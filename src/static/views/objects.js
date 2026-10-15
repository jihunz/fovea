// Objects tab: every labelled box as a crop (patch view) for fast class QA — keyboard reassign / delete, undo
// through label snapshots.
//
// Scale: 40k+ boxes. The grid is virtualised (only the rows on screen, plus a little overscan, exist in the
// DOM) and the list is a frozen working set: what was loaded stays in place after an edit — a reassigned box
// keeps its tile and shows its new class, a deleted one leaves a marked gap — and further pages continue by
// keyset from the last box already held, so an edit never makes the list skip or repeat a box.
//
// Visual comfort (docs/visual-comfort.md): a crop appears only once it is decoded, in one step; empty tiles
// are flat (no shimmer); the peek view swaps in only after its frame is decoded; results of an action are
// reported in a fixed status line, not in toasts.
import { decodedImage } from '../lib/media.js';
import { EMPTY as NO_ANN, annSchema, attrFilterText } from '../lib/annattrs.js';

const SIZES = { s: 92, m: 132, l: 188, xl: 260 };
const SIZE_ORDER = ['s', 'm', 'l', 'xl'];
const PADS = { tight: 0.06, normal: 0.2, wide: 0.6 };
const CROP_SIZES = [64, 96, 128, 160, 192, 256, 320, 384, 512, 768];   // = CROP_SIZES in fovea/api/objects.py
const CROP_RENDER = 1;      // = CROP_RENDER in fovea/api/objects.py (part of the immutable crop URL)
const MIN_CONTEXT_PX = 24;  // = MIN_CONTEXT_PX in fovea/api/objects.py (the box outline uses the same framing)
const GAP = 6, PADX = 12, PADY = 10;
const BLOCK = 240;          // boxes per request
const EXTRA = 8;            // spare rows per request, filling in for duplicates dropped after edits moved boxes
const MAX_TILES = 560;      // DOM budget: tiles alive at once (visible + overscan)
const CONCURRENCY = 5;      // crop requests in flight (browsers allow 6 per host; one stays free for the API)
const UNDO_DEPTH = 10;      // the server keeps the newest 20 automatic snapshots per dataset
// Image filters other views may hand over in the URL; kept as removable chips and passed through unchanged.
// `set` is a working-set token (POST /sets): a frozen list of image ids kept by the server for a while.
// `attr` narrows the BOXES here (core: a box whose converted attribute has the value), as it does images in Explore.
const PASS_KEYS = ['labeled', 'review', 'exclude_review', 'issue', 'seq', 'ids', 'set', 'ai', 'only_cls', 'min_boxes', 'max_boxes', 'has_label', 'attr'];
const SORTS = [['class', 'Class'], ['name', 'Image name'], ['area', 'Area'], ['aspect', 'Aspect ratio'], ['random', 'Random'], ['audit', 'AI suspicion']];
const SUSPECT = { swap: { glyph: '⇄', label: 'Possibly wrong class' }, loose: { glyph: '⤢', label: 'Loose box' }, spurious: { glyph: '✕', label: 'Box without support' } };

export function install(fovea) {
  const M = fovea.MOD;
  fovea.registerTab({ id: 'objects', label: 'Objects', icon: 'boxSelect', order: 25, render });
  fovea.keys.registerKeys('objects', 'Objects', [
    ['← → ↑ ↓', 'Move focus (⇧ extends the selection)'],
    ['0-9', 'Reassign focused or selected boxes to a class'],
    ['C', 'Pick a class to reassign to'],
    ['Del', 'Delete focused or selected boxes'],
    ['S', 'Select / deselect the focused box'],
    ['⇧ Click', 'Select a range'],
    [`${M} A`, 'Select all loaded boxes'],
    ['Esc', 'Cancel a class number being typed, else clear the selection'],
    ['Enter', 'Open the image in Annotate with the box selected'],
    ['Space', 'Hold to see the full frame'],
    [`${M} Z`, 'Undo the last change'],
    ['A', 'Accept the AI suggestion on the box'],
    ['O', 'Box outline on / off'],
    ['[ ]', 'Smaller / larger tiles'],
    ['I', 'Details panel'],
    ['Home End', 'First / last box'],
  ], 25);
}

/** Middle-ellipsis: keep the distinguishing tail (frame numbers) of long file names visible. */
function midEllipsis(s, max = 42) {
  if (!s || s.length <= max) return s || '';
  const tail = Math.ceil((max - 1) * 0.6), head = max - 1 - tail;
  return s.slice(0, head) + '…' + s.slice(-tail);
}
const basename = (p) => (p || '').split('/').pop();
/** The box's number in its image as Annotate shows it (1-based); `row` stays 0-based in URLs and API calls. */
const boxNo = (it) => `box #${it.row + 1}`;
/** replaceChildren that skips null / false (the DOM method would print them as text). */
const put = (el, ...kids) => el.replaceChildren(...kids.flat().filter((k) => k != null && k !== false));
const snapCrop = (px) => CROP_SIZES.find((s) => px <= s) || CROP_SIZES[CROP_SIZES.length - 1];

/** Where the box sits inside its crop, as fractions — the same square framing the server draws. */
export function boxInCrop(it, pad) {
  const W = it.width, H = it.height;
  if (!W || !H) return null;
  const bw = Math.abs(it.w) * W, bh = Math.abs(it.h) * H;
  const side = Math.max(Math.max(bw, bh) * (1 + 2 * pad), MIN_CONTEXT_PX, 1);
  return { l: 0.5 - bw / side / 2, t: 0.5 - bh / side / 2, w: bw / side, h: bh / side };
}

async function render(el, { ctx, dataset, fovea }) {
  const { h, icon, ui, fmt, cls, MOD, keys } = fovea;
  const { classColor, className, textOn } = fovea.colors;
  const ds = dataset;
  let annSch = NO_ANN;
  annSchema(fovea.api, ds).then((sch) => { if (sch !== NO_ANN) { annSch = sch; try { renderChips(); } catch (e) { /* view closed */ } } });
  const names = [...(ds.classes || [])];
  const q0 = ctx.query;
  const api = fovea.api;
  const boxesUrl = `/api/datasets/${ds.id}/boxes`;
  const pref = (k, d) => { try { const v = localStorage.getItem('fovea.obj.' + k); return v == null ? d : v; } catch (e) { return d; } };
  const setPref = (k, v) => { try { localStorage.setItem('fovea.obj.' + k, v); } catch (e) { /* private mode */ } };
  const ink = (c) => (typeof textOn === 'function' ? textOn(classColor(c)) : '#fff');

  // ---------------------------------------------------------------- state
  const filt = {
    classes: new Set((q0.get('cls') || '').split(',').filter((x) => /^\d+$/.test(x)).map(Number)),
    split: q0.get('split') || '', q: q0.get('q') || '',
    extra: Object.fromEntries(PASS_KEYS.filter((k) => q0.get(k)).map((k) => [k, q0.get(k)])),
  };
  let sort = SORTS.some(([v]) => v === q0.get('sort')) ? q0.get('sort') : 'class';
  let order = q0.get('order') === 'desc' || q0.get('order') === 'asc' ? q0.get('order') : (sort === 'audit' ? 'desc' : 'asc');
  let seed = Math.max(0, parseInt(q0.get('seed') || '0', 10) || 0);
  let size = SIZES[pref('size', 'm')] ? pref('size', 'm') : 'm';
  let padKey = PADS[pref('pad', 'normal')] != null ? pref('pad', 'normal') : 'normal';
  let showOutline = pref('outline', '1') !== '0';
  let showLabels = pref('labels', '1') !== '0';
  let panelOn = pref('panel', '0') === '1';
  let classCounts = null, issueLabels = null, setCount = null;

  let L = null;               // the list: { total, slots[], blocks Map, idents Map, byImage Map, ctrl }
  let cols = 1, T = SIZES[size], rowH = T + GAP;
  const tiles = new Map();    // slot index -> element
  let focus = -1, anchor = -1;
  const selected = new Set(); // slot indices
  let destroyed = false, raf = 0, blockTimer = 0, lastTop = 0;
  let gone = 0;               // held boxes that no longer match: deleted, or reassigned out of the class filter
  const undoStack = [];
  let chain = Promise.resolve();
  let digitBuf = null;        // { d, idx, Lx, timer } while waiting for a possible second digit; idx = the boxes it was typed for
  let peekOn = false, peekToken = 0;
  let restore = null;         // scroll position to return to (after coming back from Annotate)

  // ---------------------------------------------------------------- DOM
  const root = h('div', { class: 'objects' });
  const toolbar = h('div', { class: 'ob-toolbar' });
  const chipsRow = h('div', { class: 'ob-chips', role: 'toolbar', 'aria-label': 'Class filter' });
  const scroller = h('div', { class: 'ob-scroll', tabindex: '-1' });   // clicking a tile focuses the grid, not a stale button
  const canvas = h('div', { class: 'ob-canvas', role: 'list', 'aria-label': 'Boxes' });
  const msg = h('div', { class: 'ob-msg hidden' });
  const peek = h('div', { class: 'ob-peek hidden', 'aria-hidden': 'true' });
  const bulk = h('div', { class: 'bulk-bar ob-bulk hidden' });
  const gridWrap = h('div', { class: 'ob-gridwrap' }, scroller, msg, peek, bulk);
  const panel = h('aside', { class: cls('ob-panel', !panelOn && 'hidden') });
  const statusL = h('div', { class: 'ob-status-l' });
  const statusR = h('div', { class: 'ob-status-r', 'aria-live': 'polite' });
  const statusBar = h('div', { class: 'ob-status' }, statusL, statusR);
  scroller.appendChild(canvas);
  // The bar carries the background and rule, so the strip inside can fade its scrolled-off edge (mask).
  root.append(toolbar, h('div', { class: 'ob-chipbar' }, chipsRow), h('div', { class: 'ob-main' }, gridWrap, panel), statusBar);
  el.appendChild(root);
  canvas.classList.toggle('no-outline', !showOutline);
  canvas.classList.toggle('no-labels', !showLabels);

  // ---------------------------------------------------------------- toolbar
  const searchInput = h('input', { class: 'input', placeholder: 'Search file name…', value: filt.q, 'aria-label': 'Search file name' });
  searchInput.addEventListener('keydown', (e) => {
    if (e.key === 'Enter') { filt.q = searchInput.value.trim(); reload(); searchInput.blur(); }
    else if (e.key === 'Escape') { searchInput.value = filt.q; searchInput.blur(); }
  });
  // A select, not a segmented control: split folders can have long names and the toolbar must fit one line.
  const splitSel = h('select', { class: 'select', 'aria-label': 'Split', style: 'width:auto;max-width:200px', onChange: (e) => { filt.split = e.target.value; e.target.blur(); reload(); } },
    [['', 'All splits'], ...[...new Set((ds.splits || []).filter(Boolean))].map((v) => [v, v])].map(([v, l]) => h('option', { value: v, selected: v === filt.split }, l)));
  const countEl = h('span', { class: 'result-count' }, '');
  const sortSel = h('select', { class: 'select', 'aria-label': 'Order boxes by', style: 'width:auto', onChange: (e) => { sort = e.target.value; order = sort === 'audit' ? 'desc' : 'asc'; e.target.blur(); paintOrder(); reload(); } },
    SORTS.map(([v, l]) => h('option', { value: v, selected: v === sort }, `Order: ${l}`)));
  const orderBtn = h('button', { class: 'btn btn-icon', 'data-tip': 'Reverse order', 'aria-label': 'Reverse order', onClick: () => { order = order === 'asc' ? 'desc' : 'asc'; paintOrder(); reload(); } });
  const shuffleBtn = h('button', { class: 'btn btn-icon', 'data-tip': 'Shuffle again', 'aria-label': 'Shuffle again', onClick: () => { seed = 1 + Math.floor(Math.random() * 999999); reload(); } }, icon('shuffle', 14));
  const sizeSeg = ui.seg(SIZE_ORDER.map((v) => ({ value: v, label: v.toUpperCase(), tip: `Tile size ${v.toUpperCase()} ([ / ])` })), size, (v) => setSize(v));
  const displayBtn = h('button', { class: 'btn btn-icon', 'aria-haspopup': 'true', 'aria-label': 'Display options', 'data-tip': 'Display: outline, labels, context', onClick: () => displayMenu() }, icon('sliders', 14));
  const panelBtn = h('button', { class: cls('btn btn-icon', panelOn && 'active'), 'data-tip': 'Details panel (I)', 'aria-label': 'Details panel', 'aria-pressed': String(panelOn), onClick: () => togglePanel() }, icon('sidebar', 14));
  const refreshBtn = h('button', { class: 'btn btn-icon', 'data-tip': 'Reload the list', 'aria-label': 'Reload the list', onClick: () => reload() }, icon('refresh', 14));
  toolbar.append(h('div', { class: 'input-wrap ob-search' }, icon('search'), searchInput), splitSel, h('span', { class: 'spacer' }),
    countEl, sortSel, orderBtn, shuffleBtn, sizeSeg, displayBtn, panelBtn, refreshBtn);
  function paintOrder() {
    orderBtn.replaceChildren(icon(order === 'asc' ? 'arrowUp' : 'arrowDown', 14));
    orderBtn.dataset.tip = `${order === 'asc' ? 'Ascending' : 'Descending'} — click to reverse`;
    shuffleBtn.classList.toggle('hidden', sort !== 'random');
    sortSel.value = sort;
  }
  paintOrder();

  function displayMenu() {
    const outlineSw = ui.switchBtn(showOutline, (v) => setOutline(v));
    const labelsSw = ui.switchBtn(showLabels, (v) => { showLabels = v; setPref('labels', v ? '1' : '0'); canvas.classList.toggle('no-labels', !v); });
    const padSeg = ui.seg([{ value: 'tight', label: 'Tight' }, { value: 'normal', label: 'Normal' }, { value: 'wide', label: 'Wide' }], padKey, (v) => { padKey = v; setPref('pad', v); relayout(true); });
    ui.popover(displayBtn, h('div', { class: 'col gap-8 ob-display' },
      h('label', { class: 'row' }, h('span', { class: 'grow small' }, 'Box outline', h('span', { class: 'hint', style: 'display:block' }, 'O')), outlineSw),
      h('label', { class: 'row' }, h('span', { class: 'grow small' }, 'Class name on tiles'), labelsSw),
      h('div', { class: 'col gap-4' }, h('span', { class: 'small' }, 'Context around the box'), padSeg)), { width: 250 });
  }
  function setOutline(v) { showOutline = v; setPref('outline', v ? '1' : '0'); canvas.classList.toggle('no-outline', !v); }
  function setSize(v) {
    if (!SIZES[v] || v === size) return;
    size = v; setPref('size', v); sizeSeg.setValue(v); relayout(true);
  }
  function togglePanel(v = !panelOn) {
    panelOn = v; setPref('panel', v ? '1' : '0');
    panel.classList.toggle('hidden', !v); panelBtn.classList.toggle('active', v); panelBtn.setAttribute('aria-pressed', String(v));
    relayout(false);
    if (v) renderPanel();
  }

  // ---------------------------------------------------------------- filters: chips + URL
  let myQuery = location.search;   // this view's own query; by teardown time the URL already belongs to the next view
  function syncUrl() {
    fovea.router.replaceQuery({
      ...filt.extra, cls: [...filt.classes].sort((a, b) => a - b).join(',') || null, split: filt.split || null, q: filt.q || null,
      sort: sort !== 'class' ? sort : null, order: order !== (sort === 'audit' ? 'desc' : 'asc') ? order : null, seed: sort === 'random' && seed ? seed : null,
    });
    myQuery = location.search;
  }
  function listParams() {
    const p = { ...filt.extra, sort, order };
    if (filt.classes.size) p.cls = [...filt.classes].sort((a, b) => a - b).join(',');
    if (filt.split) p.split = filt.split;
    if (filt.q) p.q = filt.q;
    if (sort === 'random') p.seed = seed;
    return p;
  }
  function extraLabel(k, v) {
    const lab = (code) => (issueLabels && issueLabels[code]) || code;
    switch (k) {
      case 'issue': return `Issue: ${v.split(',').map(lab).join(', ')}`;
      case 'review': return `Review: ${v}`;
      case 'exclude_review': return `Not ${v}`;
      case 'labeled': return `Label state: ${v}`;
      case 'seq': return `Sequence: ${v}`;
      case 'ids': { const n = v.split(',').length; return `${fmt.num(n)} chosen image${n === 1 ? '' : 's'}`; }
      case 'set': return setCount != null ? `${fmt.num(setCount)} selected image${setCount === 1 ? '' : 's'}` : 'Selected images';
      case 'ai': return v === 'audit' ? 'AI audit findings' : v === 'pending' ? 'AI proposals pending' : `AI: ${v}`;
      case 'only_cls': return `Only classes ${v}`;
      case 'attr': return `Boxes with ${attrFilterText(annSch, v)}`;
      default: return `${k}: ${v}`;
    }
  }
  /** `reveal`: the filter just changed or the list just arrived — bring the active class chip into view (the
      strip scrolls sideways, and an active chip past its edge makes the list look unfiltered). Edits re-render
      the chips without moving the strip under the reader. */
  function renderChips({ reveal = false } = {}) {
    const chips = [];
    for (const [k, v] of Object.entries(filt.extra)) {
      chips.push(ui.chip(extraLabel(k, v), { cls: 'ob-fchip', onRemove: () => { delete filt.extra[k]; reload(); } }));
    }
    const ids = new Set(names.map((_, i) => i));
    if (classCounts) Object.keys(classCounts).forEach((c) => ids.add(Number(c)));
    const all = classCounts ? Object.values(classCounts).reduce((a, b) => a + b, 0) : null;
    chips.push(h('button', { class: cls('ob-cchip', !filt.classes.size && 'active'), 'aria-pressed': String(!filt.classes.size), onClick: () => { if (filt.classes.size) { filt.classes.clear(); reload(); } } },
      h('span', 'All classes'), all != null ? h('span', { class: 'n' }, fmt.num(all)) : null));
    for (const c of [...ids].sort((a, b) => a - b)) {
      const n = classCounts ? classCounts[String(c)] || 0 : null;
      const on = filt.classes.has(c);
      chips.push(h('button', {
        class: cls('ob-cchip', on && 'active', n === 0 && 'zero'), 'aria-pressed': String(on),
        'data-tip': `${on ? 'Showing' : 'Show only'} ${className(names, c)} · ⇧/${MOD}-click to combine`,
        onClick: (e) => {
          if (e.shiftKey || e.metaKey || e.ctrlKey) { on ? filt.classes.delete(c) : filt.classes.add(c); }
          else if (on && filt.classes.size === 1) filt.classes.clear();
          else { filt.classes.clear(); filt.classes.add(c); }
          reload();
        },
      }, h('span', { class: 'sw', style: `background:${classColor(c)}` }), h('span', { class: 'idx' }, String(c)), h('span', { class: 'nm' }, className(names, c)),
      n != null ? h('span', { class: 'n' }, fmt.compact(n)) : null));
    }
    chipsRow.replaceChildren(...chips);
    if (reveal) {
      const a = chipsRow.querySelector('.ob-cchip.active');
      if (a) {
        const cr = chipsRow.getBoundingClientRect(), ar = a.getBoundingClientRect(), m = 40;   // m: clear of the edge fade
        if (ar.left < cr.left + m) chipsRow.scrollLeft -= cr.left + m - ar.left;
        else if (ar.right > cr.right - m) chipsRow.scrollLeft += ar.right - (cr.right - m);
      }
    }
    paintChipEdges();
  }
  /** Fade the strip's edge on the side where more chips are hidden: the strip scrolls, and nothing else says so. */
  function paintChipEdges() {
    const max = chipsRow.scrollWidth - chipsRow.clientWidth;
    chipsRow.classList.toggle('more-l', max > 1 && chipsRow.scrollLeft > 1);
    chipsRow.classList.toggle('more-r', max > 1 && chipsRow.scrollLeft < max - 1);
  }
  chipsRow.addEventListener('scroll', paintChipEdges, { passive: true });
  // A vertical wheel scrolls the strip sideways (a mouse without a horizontal wheel reaches every chip).
  chipsRow.addEventListener('wheel', (e) => {
    if (Math.abs(e.deltaY) <= Math.abs(e.deltaX) || chipsRow.scrollWidth <= chipsRow.clientWidth) return;
    e.preventDefault();
    chipsRow.scrollLeft += e.deltaY;
  }, { passive: false });
  if (filt.extra.issue) api.get(`/api/datasets/${ds.id}/stats`).then((r) => { issueLabels = r.issue_labels || null; if (!destroyed) renderChips(); }).catch(() => {});
  // A working set arrives only through the URL (the chip can just be removed), so its size is asked once.
  if (filt.extra.set) {
    api.get(`/api/datasets/${ds.id}/images`, { set: filt.extra.set, limit: 1 })
      .then((r) => { setCount = typeof r.total === 'number' ? r.total : null; if (!destroyed) renderChips(); }).catch(() => {});
  }
  /** The server no longer holds this working set (they expire): 410 {detail: {message, code: 'set_expired'}}. */
  const setGone = (e) => !!filt.extra.set && (e.status === 410 || !!(e.body && e.body.detail && e.body.detail.code === 'set_expired'));

  // ---------------------------------------------------------------- the list
  const ident = (it) => `${it.image_id}:${it.row}:${it.xc}:${it.yc}:${it.w}:${it.h}`;
  function index(Lx, i, it) {
    it._L = Lx;
    Lx.slots[i] = it;
    Lx.idents.set(ident(it), i);
    let s = Lx.byImage.get(it.image_id);
    if (!s) Lx.byImage.set(it.image_id, s = new Set());
    s.add(i);
  }
  /** Put a page of server items into slots [a, a+BLOCK), skipping boxes the list already holds. */
  function place(Lx, b, items, requested) {
    const a = b * BLOCK, end = Math.min(a + BLOCK, Lx.total);
    let i = a;
    for (const it of items) {
      if (i >= end) break;
      if (Lx.idents.has(ident(it))) continue;   // already shown elsewhere (moved by an edit)
      index(Lx, i, it); i++;
    }
    // The server ran out before this block was full: the list shrank since it was opened (deletions). If
    // nothing after this block is loaded, the list simply ends here.
    if (i < end && items.length < requested && ![...Lx.blocks.keys()].some((k) => k > b && Lx.blocks.get(k) === 'done')) {
      Lx.total = i;
      return true;
    }
    return false;
  }
  function nearestBefore(Lx, a) {
    for (let bb = Math.floor((a - 1) / BLOCK); bb >= 0; bb--) {
      if (Lx.blocks.get(bb) !== 'done') continue;
      for (let j = Math.min(a, (bb + 1) * BLOCK) - 1; j >= bb * BLOCK; j--) if (Lx.slots[j]) return j;
    }
    return -1;
  }
  async function loadBlock(Lx, b) {
    if (Lx.blocks.has(b) && Lx.blocks.get(b) !== 'error') return;
    const a = b * BLOCK;
    if (a >= Lx.total) return;
    Lx.blocks.set(b, 'loading');
    const params = { ...Lx.params, limit: BLOCK + EXTRA };
    const j = nearestBefore(Lx, a);
    if (j >= 0) { params.after = JSON.stringify(Lx.slots[j].k); params.offset = a - j - 1; }   // continue exactly from a box we hold
    else params.offset = a;
    try {
      const res = await api.get(boxesUrl, params, { signal: Lx.ctrl.signal });
      if (Lx !== L || destroyed) return;
      const shrank = place(Lx, b, res.items, BLOCK + EXTRA);
      Lx.blocks.set(b, 'done');
      if (shrank) relayout(false);
      for (let i = a; i < a + BLOCK; i++) if (tiles.has(i)) repaint(i);
      if (restore) applyRestore();
      schedule();
      if (focus >= a && focus < a + BLOCK) { renderStatusL(); schedulePanel(); }
    } catch (e) {
      if (Lx !== L || destroyed || e.name === 'AbortError') return;
      Lx.blocks.set(b, 'error');
      for (let i = a; i < a + BLOCK; i++) if (tiles.has(i)) repaint(i);
      note(`Could not load boxes: ${e.message}`, 'danger', { label: 'Retry', run: () => { Lx.blocks.delete(b); schedule(); } });
    }
  }

  let listSeq = 0;
  async function reload({ keepPosition = false } = {}) {
    commitDigit({ ask: false });     // typed for boxes of the list that is about to go
    syncUrl();
    if (keepPosition && !restore && L) restore = { q: myQuery, focus, top: scroller.scrollTop };
    if (L) L.ctrl.abort();
    const params = listParams();
    const Lx = L = { seq: ++listSeq, total: null, slots: [], blocks: new Map(), idents: new Map(), byImage: new Map(), ctrl: new AbortController(), params };
    if (!keepPosition) { restore = null; scroller.scrollTop = 0; }
    selected.clear(); anchor = -1; focus = -1; gone = 0;
    for (const t of tiles.values()) t.remove();
    tiles.clear(); canvas.style.height = '0px';
    hidePeek(); updateBulk(); renderChips({ reveal: true }); renderStatusL();
    countEl.textContent = '';
    const wait = setTimeout(() => { if (L === Lx && Lx.total == null) showMsg(h('div', { class: 'col', style: 'align-items:center' }, ui.spinner(), h('span', { class: 'small muted' }, 'Loading boxes…'))); }, 400);
    try {
      const res = await api.get(boxesUrl, { ...params, offset: 0, limit: BLOCK + EXTRA, counts: 1 }, { signal: Lx.ctrl.signal });
      if (Lx !== L || destroyed) return;
      Lx.total = res.total;
      classCounts = res.class_counts || null;
      place(Lx, 0, res.items, BLOCK + EXTRA);
      Lx.blocks.set(0, 'done');
      renderChips({ reveal: true });
      if (sort === 'audit' && res.audit_available === false) note('No AI audit results for this dataset yet — run a label audit from the AI tab. Showing image order.', 'info');
      if (!Lx.total) showEmpty(); else hideMsg();
      relayout(false);
      renderStatusL(); renderPanel();
      if (restore) applyRestore();
    } catch (e) {
      if (Lx !== L || destroyed || e.name === 'AbortError') return;
      if (setGone(e)) {
        showMsg(ui.emptyState({ icon: 'alert', title: 'This selection is no longer available', message: e.message,
          action: h('button', { class: 'btn btn-primary', onClick: () => { delete filt.extra.set; reload(); } }, 'Show without the selection') }));
        return;
      }
      showMsg(ui.emptyState({ icon: 'alert', title: 'Could not load boxes', message: e.message, action: h('button', { class: 'btn btn-primary', onClick: () => reload() }, 'Retry') }));
    } finally { clearTimeout(wait); }
  }
  function showEmpty() {
    const others = [filt.split && `split ${filt.split}`, filt.q && `“${filt.q}”`, ...Object.entries(filt.extra).map(([k, v]) => extraLabel(k, v))].filter(Boolean);
    const picked = [...filt.classes].sort((a, b) => a - b).map((c) => className(names, c));
    if (picked.length || others.length) {
      // Say which filter empties the list: the class strip scrolls, so its active chip may be out of sight.
      const which = picked.length === 1 ? picked[0] : picked.length ? `${picked.slice(0, -1).join(', ')} or ${picked[picked.length - 1]}` : '';
      const message = !picked.length ? `Nothing matches ${others.join(' · ')}.`
        : others.length ? `No ${which} boxes match ${others.join(' · ')}.`
        : `${which} ${picked.length === 1 ? 'has' : 'have'} no boxes.`;
      showMsg(ui.emptyState({ icon: 'boxSelect', title: 'No boxes match', message, action: h('div', { class: 'row gap-8', style: 'justify-content:center' },
        picked.length ? h('button', { class: 'btn btn-primary', onClick: () => { filt.classes.clear(); reload(); } }, 'Show all classes') : null,
        h('button', { class: 'btn', onClick: () => clearFilters() }, 'Clear filters')) }));
    }
    else showMsg(ui.emptyState({ icon: 'boxSelect', title: 'No boxes yet', message: 'Boxes appear here once images are labeled.', action: fovea.link(`/d/${ds.id}/annotate`, { class: 'btn btn-primary' }, icon('pen', 14), 'Open Annotate') }));
  }
  function clearFilters() {
    filt.classes.clear(); filt.split = ''; filt.q = ''; filt.extra = {};
    searchInput.value = ''; splitSel.value = '';
    reload();
  }
  function showMsg(node) { msg.replaceChildren(node); msg.classList.remove('hidden'); }
  function hideMsg() { msg.classList.add('hidden'); msg.replaceChildren(); }
  function updateCount() {
    if (!L || L.total == null) { countEl.textContent = ''; return; }
    const n = Math.max(0, L.total - gone);
    countEl.textContent = `${fmt.num(n)} box${n === 1 ? '' : 'es'}`;
  }

  // ---------------------------------------------------------------- virtual grid
  function relayout(sizeChanged) {
    const w = scroller.clientWidth - PADX * 2;
    if (w <= 0) return;
    // Keep the same boxes in view across a column-count change (the pixel offset would land elsewhere).
    const keepIdx = L && L.total && scroller.scrollTop > 0 ? visibleRange().r0 * cols : -1;
    const base = SIZES[size];
    cols = Math.max(1, Math.floor((w + GAP) / (base + GAP)));
    T = Math.floor((w - (cols - 1) * GAP) / cols);
    rowH = T + GAP;
    canvas.style.setProperty('--tile', T + 'px');
    paintChipEdges();
    canvas.classList.toggle('small-tiles', T < 110);
    const rows = L && L.total ? Math.ceil(L.total / cols) : 0;
    canvas.style.height = rows ? `${rows * rowH - GAP + PADY * 2}px` : '0px';
    // Positions (and, for a new size or context, crop URLs) change: rebuild the tiles in range.
    for (const t of tiles.values()) t.remove();
    tiles.clear();
    if (focus >= 0 && sizeChanged) scrollToIndex(focus, true);
    else if (keepIdx >= 0) scroller.scrollTop = Math.floor(keepIdx / cols) * rowH;
    updateCount();
    schedule();
  }
  const ro = new ResizeObserver(() => relayout(false));
  ro.observe(scroller);
  function schedule() { if (!raf && !destroyed) raf = requestAnimationFrame(update); }
  scroller.addEventListener('scroll', schedule, { passive: true });

  function visibleRange() {
    const top = scroller.scrollTop, vh = scroller.clientHeight;
    const r0 = Math.max(0, Math.floor((top - PADY) / rowH)), r1 = Math.max(r0, Math.floor((top + vh - PADY) / rowH));
    return { r0, r1, top, vh };
  }
  function update() {
    raf = 0;
    if (!L || !L.total || destroyed) return;
    const { r0, r1, top, vh } = visibleRange();
    const visRows = r1 - r0 + 1;
    const ov = Math.max(0, Math.min(3, Math.floor((MAX_TILES / cols - visRows) / 2)));
    const from = Math.max(0, (r0 - ov) * cols), to = Math.min(L.total, (r1 + ov + 1) * cols);
    for (const [i, t] of tiles) if (i < from || i >= to) { t.remove(); tiles.delete(i); }
    const frag = document.createDocumentFragment();
    for (let i = from; i < to; i++) if (!tiles.has(i)) { const t = makeTile(i); tiles.set(i, t); frag.appendChild(t); }
    canvas.appendChild(frag);
    // While the scrollbar is being dragged through thousands of rows, wait for it to settle before
    // asking for pages that will never be looked at.
    const fast = Math.abs(top - lastTop) > vh * 1.5;
    lastTop = top;
    clearTimeout(blockTimer);
    const want = () => {
      if (!L) return;
      const v0 = r0 * cols, v1 = Math.min(L.total, (r1 + 1) * cols);
      const b0 = Math.floor(Math.max(0, from) / BLOCK), b1 = Math.floor(Math.max(0, Math.min(L.total, to + cols * 2) - 1) / BLOCK);
      for (let b = Math.floor(v0 / BLOCK); b <= Math.floor(Math.max(v0, v1 - 1) / BLOCK); b++) loadBlock(L, b);   // visible first
      for (let b = b0; b <= b1; b++) loadBlock(L, b);
    };
    if (fast) blockTimer = setTimeout(want, 120); else want();
    pumpCrops();
  }

  function cropUrl(it) {
    const s = snapCrop(Math.round(T * (window.devicePixelRatio || 1)));
    return `/api/crop/${it.image_id}?box=${it.xc},${it.yc},${it.w},${it.h}&pad=${PADS[padKey]}&s=${s}&v=${Math.round(it.mtime || 0)}.${CROP_RENDER}`;
  }
  function makeTile(i) {
    const x = PADX + (i % cols) * (T + GAP), y = PADY + Math.floor(i / cols) * rowH;
    const t = h('div', { class: 'ob-tile', role: 'listitem', style: `left:${x}px;top:${y}px;width:${T}px;height:${T}px` });
    t._i = i;
    paint(t);
    return t;
  }
  function repaint(i) { const t = tiles.get(i); if (t) paint(t); }
  function paint(t) {
    const i = t._i, it = L && L.slots[i];
    const blockState = L ? L.blocks.get(Math.floor(i / BLOCK)) : null;
    if (!it) {
      t.className = cls('ob-tile', 'is-empty', blockState === 'error' && 'is-failed');
      t.replaceChildren();
      t._url = null; t._img = null; t.removeAttribute('data-tip');
      return;
    }
    const url = cropUrl(it);
    const keep = t._url === url && t._img ? t._img : null;
    t._url = url; t._img = keep;
    if (!keep) {
      const hit = cropCache.get(url);
      if (hit) t._img = attachable(hit);
    }
    t.className = cls('ob-tile', t._img && 'has-img', i === focus && 'is-focus', selected.has(i) && 'is-selected',
      it.deleted && 'is-deleted', it.edited != null && !it.deleted && 'is-edited', it.conflict && 'is-conflict', it.pending && 'is-pending', t._failed === url && 'is-failed');
    t.style.setProperty('--c', classColor(it.cls));
    t.style.setProperty('--ct', ink(it.cls));
    const r = boxInCrop(it, PADS[padKey]);
    const kids = [];
    if (t._img) kids.push(t._img);
    if (r) kids.push(h('div', { class: 'ob-box', style: `left:${r.l * 100}%;top:${r.t * 100}%;width:${r.w * 100}%;height:${r.h * 100}%` }));
    kids.push(h('span', { class: 'ob-lbl' }, className(names, it.cls)));
    kids.push(h('span', { class: 'ob-check' }, icon('check', 11)));
    const marks = [];
    if (it.conflict) marks.push(h('span', { class: 'ob-mk warn' }, '!'));
    else if (it.edited != null && !it.deleted) marks.push(h('span', { class: 'ob-mk' }, icon('pen', 10)));
    if (it.suspects && it.suspects.length && !it.deleted) {
      const s = it.suspects[0];
      marks.push(h('span', { class: 'ob-mk ai' }, (SUSPECT[s.kind] || {}).glyph || '?', s.kind === 'swap' && s.cls != null ? ` ${s.cls}` : ''));
    }
    if (it.conf != null) marks.push(h('span', { class: 'ob-mk conf' }, it.conf.toFixed(2)));
    if (marks.length) kids.push(h('span', { class: 'ob-marks' }, marks));
    if (it.deleted) kids.push(h('span', { class: 'ob-gone' }, it.external ? 'Removed elsewhere' : 'Deleted'));
    t.replaceChildren(...kids);
    t.dataset.tip = tipFor(it);
    t.setAttribute('aria-label', t.dataset.tip);
    t.setAttribute('aria-selected', String(selected.has(i)));
  }
  function tipFor(it) {
    const bits = [`${basename(it.rel_path)} · ${boxNo(it)}`, className(names, it.cls)];
    if (it.edited != null && !it.deleted) bits.push(`was ${className(names, it.edited)}`);
    if (!it.deleted && filt.classes.size && !filt.classes.has(it.cls)) bits.push('no longer matches the class filter');
    if (it.conflict) bits.push(`not changed: ${it.conflict}`);
    if (it.suspects && it.suspects.length) { const s = it.suspects[0]; bits.push(`AI: ${(SUSPECT[s.kind] || {}).label || s.kind}${s.kind === 'swap' && s.cls != null ? ` → ${className(names, s.cls)}` : ''}${s.reason ? ` (${s.reason})` : ''}`); }
    if (it.stale) bits.push('index out of date — rescan');
    return bits.join(' · ');
  }

  // ---------------------------------------------------------------- crops: decoded before shown, visible first
  const cropCache = new Map();      // url -> decoded <img>, LRU: scrolling back a little shows them at once
  const inflight = new Set();
  let active = 0;
  function attachable(img) { return img.parentNode ? img.cloneNode() : img; }
  function cacheCrop(url, img) {
    cropCache.delete(url); cropCache.set(url, img);
    while (cropCache.size > 320) cropCache.delete(cropCache.keys().next().value);   // ~3 screens; the HTTP cache has the rest
  }
  function pumpCrops() {
    if (active >= CONCURRENCY || !L) return;
    const { r0, r1 } = visibleRange();
    const v0 = r0 * cols, v1 = (r1 + 1) * cols;
    const want = [];
    for (const [i, t] of tiles) if (t._url && !t._img && t._failed !== t._url && !inflight.has(t._url)) want.push(i);
    want.sort((a, b) => ((a >= v0 && a < v1) ? 0 : 1) - ((b >= v0 && b < v1) ? 0 : 1) || a - b);
    for (const i of want) {
      if (active >= CONCURRENCY) break;
      const t = tiles.get(i); if (!t || inflight.has(t._url)) continue;
      loadCrop(t._url);
    }
  }
  function loadCrop(url) {
    active++; inflight.add(url);
    decodedImage(url).then((img) => {
      img.alt = ''; img.draggable = false; img.decoding = 'async';
      cacheCrop(url, img);
      for (const t of tiles.values()) if (t._url === url && !t._img) { t._img = attachable(img); paint(t); }
    }, () => {
      for (const t of tiles.values()) if (t._url === url) { t._failed = url; paint(t); }
    }).finally(() => { active--; inflight.delete(url); if (!destroyed) pumpCrops(); });
  }

  // ---------------------------------------------------------------- focus & selection
  function scrollToIndex(i, center = false) {
    const y = PADY + Math.floor(i / cols) * rowH;
    const top = scroller.scrollTop, vh = scroller.clientHeight;
    if (center) scroller.scrollTop = Math.max(0, y - (vh - T) / 2);
    else if (y < top + PADY) scroller.scrollTop = Math.max(0, y - PADY);
    else if (y + T > top + vh - PADY) scroller.scrollTop = y + T - vh + PADY;
  }
  function setFocus(i, { extend = false, scroll = true } = {}) {
    if (!L || !L.total) return;
    i = Math.max(0, Math.min(L.total - 1, i));
    if (i !== focus || extend) commitDigit();      // a buffered digit belongs to the boxes it was typed on
    const prev = focus;
    focus = i;
    if (extend) {
      if (anchor < 0) anchor = prev >= 0 ? prev : i;
      const [a, b] = [Math.min(anchor, i), Math.max(anchor, i)];
      for (let k = a; k <= b; k++) if (L.slots[k] && !L.slots[k].deleted) selected.add(k);
      for (const [k] of tiles) repaintSel(k);
      updateBulk();
    } else anchor = i;
    if (prev >= 0) repaintSel(prev);
    repaintSel(i);
    if (scroll) scrollToIndex(i);
    renderStatusL(); schedulePanel();
    if (peekOn) showPeek();
  }
  function repaintSel(i) {
    const t = tiles.get(i); if (!t) return;
    t.classList.toggle('is-focus', i === focus);
    t.classList.toggle('is-selected', selected.has(i));
    if (t.hasAttribute('aria-selected')) t.setAttribute('aria-selected', String(selected.has(i)));
  }
  function toggleSelect(i) {
    const it = L && L.slots[i]; if (!it || it.deleted) return;
    commitDigit();
    selected.has(i) ? selected.delete(i) : selected.add(i);
    repaintSel(i); updateBulk();
  }
  function clearSelection() { commitDigit(); const was = [...selected]; selected.clear(); was.forEach(repaintSel); updateBulk(); }
  function selectAllLoaded() {
    if (!L) return;
    commitDigit();
    L.slots.forEach((it, i) => { if (it && !it.deleted) selected.add(i); });
    for (const [k] of tiles) repaintSel(k);
    updateBulk();
    const loaded = L.slots.filter(Boolean).length;
    if (loaded < L.total) note(`Selected the ${fmt.num(selected.size)} loaded boxes — scroll further to load more of the ${fmt.num(L.total)}.`, 'info');
  }
  /** Targets of an action: the selection, or the focused box. */
  function targets() {
    if (selected.size) return [...selected].sort((a, b) => a - b);
    return focus >= 0 && L && L.slots[focus] ? [focus] : [];
  }
  function updateBulk() {
    bulk.classList.toggle('hidden', !selected.size);
    canvas.classList.toggle('selecting', selected.size > 0);
    if (!selected.size) { bulk.replaceChildren(); return; }
    const reBtn = h('button', { class: 'btn', onClick: () => pickClass(reBtn, `Reassign ${fmt.num(selected.size)} boxes to`, (c) => reassign(targets(), c)) }, icon('tag', 13), 'Reassign…', h('span', { class: 'kbd' }, 'C'));
    bulk.replaceChildren(
      h('span', { class: 'strong' }, `${fmt.num(selected.size)} selected`),
      reBtn,
      h('button', { class: 'btn', onClick: () => remove(targets()) }, icon('trash', 13), 'Delete'),
      h('button', { class: 'btn btn-icon', 'data-tip': 'Clear selection (Esc)', 'aria-label': 'Clear selection', onClick: clearSelection }, icon('x', 13)));
  }

  // pointer
  canvas.addEventListener('click', (e) => {
    const t = e.target.closest('.ob-tile'); if (!t || !L || !L.slots[t._i]) return;
    const i = t._i;
    if (e.shiftKey) { setFocus(i, { extend: true, scroll: false }); return; }
    if (e.metaKey || e.ctrlKey) { toggleSelect(i); setFocus(i, { scroll: false }); return; }
    setFocus(i, { scroll: false });
  });
  canvas.addEventListener('dblclick', (e) => { const t = e.target.closest('.ob-tile'); if (t && L && L.slots[t._i]) openAnnotate(t._i); });
  canvas.addEventListener('contextmenu', (e) => {
    const t = e.target.closest('.ob-tile'); if (!t || !L || !L.slots[t._i]) return;
    e.preventDefault();
    const i = t._i, it = L.slots[i];
    if (!selected.has(i)) setFocus(i, { scroll: false });
    const many = selected.has(i) && selected.size > 1;
    ui.menu({ x: e.clientX, y: e.clientY }, [
      { label: 'Open in Annotate', icon: 'pen', kbd: '↵', onClick: () => openAnnotate(i) },
      { label: many ? `Reassign ${selected.size} boxes…` : 'Reassign…', icon: 'tag', kbd: 'C', disabled: it.deleted, onClick: () => pickClass(t, 'Reassign to', (c) => reassign(many ? targets() : [i], c)) },
      { label: many ? `Delete ${selected.size} boxes` : 'Delete box', icon: 'trash', kbd: 'Del', danger: true, disabled: it.deleted, onClick: () => remove(many ? targets() : [i]) },
      { sep: true },
      { label: selected.has(i) ? 'Deselect' : 'Select', icon: 'check', kbd: 'S', disabled: it.deleted, onClick: () => toggleSelect(i) },
      { label: 'Copy image name', icon: 'copy', onClick: () => ui.copyText(it.rel_path) },
    ]);
  });

  // ---------------------------------------------------------------- class picker
  function pickClass(anchorEl, title, onPick) {
    commitDigit();
    const input = h('input', { class: 'input input-sm', placeholder: 'Type a name or number…', 'aria-label': 'Find class' });
    const list = h('div', { class: 'ob-pick-list', role: 'listbox' });
    let sel = 0, shown = [];
    const draw = () => {
      const q = input.value.trim().toLowerCase();
      shown = names.map((n, i) => [i, n]).filter(([i, n]) => !q || String(i) === q || String(n).toLowerCase().includes(q));
      sel = Math.max(0, Math.min(sel, shown.length - 1));
      list.replaceChildren(...(shown.length ? shown.map(([i, n], k) => h('button', { class: cls('ob-pick-item', k === sel && 'active'), role: 'option', 'aria-selected': String(k === sel), onClick: () => pick(i) },
        h('span', { class: 'sw', style: `background:${classColor(i)}` }), h('span', { class: 'idx' }, String(i)), h('span', { class: 'grow truncate' }, className(names, i)))) : [h('div', { class: 'small faint', style: 'padding:6px' }, 'No class matches')]));
      const a = list.querySelector('.active'); if (a) a.scrollIntoView({ block: 'nearest' });
    };
    input.addEventListener('input', () => { sel = 0; draw(); });
    input.addEventListener('keydown', (e) => {
      // Handled here only: once the picker closes, focus is back on the page and the grid would see the key too.
      if (e.key === 'ArrowDown') { e.preventDefault(); e.stopPropagation(); sel = Math.min(shown.length - 1, sel + 1); draw(); }
      else if (e.key === 'ArrowUp') { e.preventDefault(); e.stopPropagation(); sel = Math.max(0, sel - 1); draw(); }
      else if (e.key === 'Enter') { e.preventDefault(); e.stopPropagation(); if (!e.isComposing && shown[sel]) pick(shown[sel][0]); }
    });
    const pop = ui.popover(anchorEl, h('div', { class: 'col gap-8' }, h('div', { class: 'small strong' }, title), input, list), { width: 260 });
    draw();
    setTimeout(() => input.focus(), 0);
    function pick(c) { pop.close(); onPick(c); }
  }

  // ---------------------------------------------------------------- edits
  let undoable = true;        // false once the server says it keeps no snapshots: every multi-box write then asks first
  const NOTHING = 'Focus a box first — click one or use the arrow keys.';
  async function reassign(idx, c) {
    if (c == null || c < 0 || !L) return;
    if (!idx.length) { note(NOTHING, 'info'); return; }
    if (c >= names.length) { note(names.length ? `There is no class ${c} — this dataset has ${names.length} class${names.length === 1 ? '' : 'es'} (0–${names.length - 1}).` : 'This dataset has no classes yet.', 'warn'); return; }
    const Lx = L;
    idx = idx.filter((i) => L.slots[i] && !L.slots[i].deleted && L.slots[i].cls !== c);
    if (!idx.length) return;
    if ((idx.length > 200 || (!undoable && idx.length > 1)) && !(await ui.confirm({ title: `Reassign ${fmt.num(idx.length)} boxes?`, message: `${fmt.num(idx.length)} boxes become ${className(names, c)} in their label files.${undoable ? ` You can undo this with ${MOD} Z.` : ''}`, okLabel: `Reassign ${fmt.num(idx.length)}` }))) return;
    if (Lx === L) edit(idx, { cls: c });
  }
  async function remove(idx) {
    if (!L) return;
    commitDigit();
    if (!idx.length) { note(NOTHING, 'info'); return; }
    const Lx = L;
    idx = idx.filter((i) => L.slots[i] && !L.slots[i].deleted);
    if (!idx.length) return;
    if ((idx.length > 20 || (!undoable && idx.length > 1)) && !(await ui.confirm({ title: `Delete ${fmt.num(idx.length)} boxes?`, message: `The ${fmt.num(idx.length)} boxes are removed from their label files.${undoable ? ` You can undo this with ${MOD} Z.` : ' This cannot be undone here.'}`, okLabel: `Delete ${fmt.num(idx.length)}`, danger: true }))) return;
    if (Lx === L) edit(idx, { del: true });
  }
  /** Label of the one-key fix the AI suggests for this box, or null (loose boxes are fixed in Annotate). */
  function acceptable(it) {
    const s = it && !it.deleted && it.suspects && it.suspects[0];
    if (!s) return null;
    if (s.kind === 'swap' && s.cls != null && s.cls !== it.cls) return `Make ${className(names, s.cls)}`;
    if (s.kind === 'spurious') return 'Delete (AI)';
    return null;
  }
  function acceptSuggestion(idx) {
    commitDigit();
    const groups = new Map();
    for (const i of idx) {
      const it = L.slots[i]; const s = it && !it.deleted && it.suspects && it.suspects[0];
      if (!s) continue;
      if (s.kind === 'swap' && s.cls != null && s.cls !== it.cls) { const k = `c${s.cls}`; if (!groups.has(k)) groups.set(k, { op: { cls: s.cls }, idx: [], preds: [] }); groups.get(k).idx.push(i); groups.get(k).preds.push(s.id); }
      else if (s.kind === 'spurious') { if (!groups.has('del')) groups.set('del', { op: { del: true }, idx: [], preds: [] }); groups.get('del').idx.push(i); groups.get('del').preds.push(s.id); }
    }
    if (!groups.size) { note(idx.length ? 'No AI suggestion to accept here (loose boxes are fixed in Annotate).' : 'Nothing focused.', 'info'); return; }
    for (const g of groups.values()) edit(g.idx, g.op, g.preds);
  }

  function edit(idx, op, acceptPreds = null) {
    if (!idx.length || !L) return;
    const Lx = L;
    const before = idx.map((i) => { const it = Lx.slots[i]; return { i, it, cls: it.cls, deleted: !!it.deleted, edited: it.edited, conflict: it.conflict }; });
    for (const b of before) {
      const it = b.it;
      it.pending = true; it.conflict = null;
      bump(it, -1);
      if (op.del) { it.deleted = true; selected.delete(b.i); }
      else { if (it.edited == null) it.edited = it.cls; it.cls = op.cls; if (it.edited === it.cls) it.edited = null; }
      bump(it, +1);
      repaint(b.i);
    }
    chipsLater(); updateCount();
    if (op.del) updateBulk();
    renderStatusL(); schedulePanel(); if (peekOn) showPeek();
    chain = chain.then(() => commit(Lx, before, op, acceptPreds)).catch((e) => console.error(e));
  }
  function revert(b) {
    const it = b.it;
    bump(it, -1);
    it.cls = b.cls; it.deleted = b.deleted; it.edited = b.edited; it.pending = false;
    bump(it, +1);
  }
  /** Called with -1 before and +1 after every change to a held box. Keeps the class chips' counts in step
      (every held box matches the non-class filters the counts were taken under, so this stays exact), and
      `gone`: held boxes the list no longer counts — deleted, or reassigned to a class the filter leaves out
      (the tile stays where it was, showing its new class; the count is what the filters match now). */
  const outside = (it) => it.deleted || (filt.classes.size > 0 && !filt.classes.has(it.cls));
  function bump(it, d) {
    if (it._L !== L) return;           // a late reply about a list that has been reloaded since
    if (outside(it)) gone += d;
    if (!classCounts || it.deleted) return;
    const k = String(it.cls);
    classCounts[k] = Math.max(0, (classCounts[k] || 0) + d);
  }
  let chipsTimer = 0;
  function chipsLater() { clearTimeout(chipsTimer); chipsTimer = setTimeout(() => { if (!destroyed) renderChips(); }, 150); }
  async function commit(Lx, before, op, acceptPreds) {
    const edits = before.map(({ it, cls: was }) => ({ image_id: it.image_id, row: it.row, version: it.version, cls_was: was, box: [it.xc, it.yc, it.w, it.h], ...(op.del ? { delete: true } : { cls: op.cls }) }));
    let res;
    try {
      res = await api.patch(boxesUrl, { edits });
      if (res.job) res = await followJob(res.job, op.del ? 'Deleting boxes' : 'Reassigning boxes');
    } catch (e) {
      for (const b of before) { revert(b); b.it.conflict = null; repaint(b.i); }
      updateCount(); renderStatusL(); schedulePanel(); chipsLater();
      note(`Not saved: ${e.message}`, 'danger');
      return;
    }
    const conflicts = new Map((res.conflicts || []).map((c) => [c.image_id, c.reason]));
    let nConflict = 0;
    for (const b of before) {
      const it = b.it; it.pending = false;
      if (conflicts.has(it.image_id)) { revert(b); it.conflict = conflicts.get(it.image_id); nConflict++; }
      if (Lx === L) repaint(b.i);
    }
    if (Lx === L) { updateCount(); if (nConflict) chipsLater(); }
    const done = before.filter((b) => !conflicts.has(b.it.image_id));
    // AI findings this batch carried out (acceptPreds runs parallel to `before`); a box left unchanged keeps its own.
    const accepted = acceptPreds ? before.map((b, k) => (conflicts.has(b.it.image_id) ? null : acceptPreds[k])).filter((x) => x != null) : [];
    if (res.updated) undoable = !!res.snapshot_id;
    if (res.updated) {
      undoStack.push({ snapshot_id: res.snapshot_id, image_ids: res.images_changed || [], items: done, op, n: res.updated, Lx, preds: accepted });
      while (undoStack.length > UNDO_DEPTH) undoStack.shift();
    }
    if (accepted.length) {
      // Mark the AI findings decided (before the reconcile below re-reads them). Best effort: the label change
      // itself is already saved.
      await api.post(`/api/datasets/${ds.id}/proposals/decide`, { accept: accepted, reject: [] }).catch(() => {});
      for (const b of done) b.it.suspects = null;
    }
    const what = op.del ? `Deleted ${fmt.num(res.deleted)} box${res.deleted === 1 ? '' : 'es'}` : `Reassigned ${fmt.num(res.reassigned)} box${res.reassigned === 1 ? '' : 'es'} to ${className(names, op.cls)}`;
    if (nConflict) note(`${res.updated ? what + '. ' : ''}${fmt.num(nConflict)} box${nConflict === 1 ? ' was' : 'es were'} not changed: ${[...new Set(conflicts.values())][0]}.`, 'warn', { label: 'Reload', run: () => reload({ keepPosition: true }) });
    else if (res.updated && res.snapshot_id) note(what + '.', 'ok', { label: `Undo  ${MOD}Z`, run: () => undo() });
    else if (res.updated) note(what + '. Undo is unavailable: this server keeps no label snapshots.', 'warn');
    if (Lx === L) await reconcile([...new Set(before.map((b) => b.it.image_id))]);
    renderStatusL(); schedulePanel(); if (peekOn) showPeek();
  }

  /** Re-read the given images' boxes and line the held items up with them: rows shift after a deletion,
      versions change after every write, and a file may have been changed elsewhere. */
  async function reconcile(imageIds) {
    const Lx = L;
    for (let k = 0; k < imageIds.length; k += 150) {
      const chunk = imageIds.slice(k, k + 150);
      const all = [];
      try {
        for (let off = 0; ; off += 2000) {                    // every box of these images, however many
          const res = await api.get(boxesUrl, { ids: chunk.join(','), sort: 'name', limit: 2000, offset: off });
          all.push(...res.items);
          if (off + res.items.length >= res.total || !res.items.length) break;
        }
      } catch (e) { continue; }
      if (Lx !== L || destroyed) return;
      const server = new Map();
      for (const s of all) { if (!server.has(s.image_id)) server.set(s.image_id, []); server.get(s.image_id).push(s); }
      for (const iid of chunk) {
        const srv = server.get(iid) || [];
        const used = new Set();
        const local = [...(Lx.byImage.get(iid) || [])].sort((a, b) => Lx.slots[a].row - Lx.slots[b].row);
        for (const i of local) {
          const it = Lx.slots[i];
          if (it.deleted && !it.pending) continue;
          const same = (s) => s.xc === it.xc && s.yc === it.yc && s.w === it.w && s.h === it.h;
          let m = srv.findIndex((s, j) => !used.has(j) && same(s) && s.row === it.row);
          if (m < 0) m = srv.findIndex((s, j) => !used.has(j) && same(s));
          Lx.idents.delete(ident(it));
          if (m < 0) {
            if (!it.pending) { bump(it, -1); it.deleted = true; it.external = true; bump(it, +1); selected.delete(i); }
          } else {
            used.add(m);
            const s = srv[m];
            it.row = s.row; it.version = s.version; it.stale = s.stale;
            if (!it.pending && it.cls !== s.cls) { bump(it, -1); it.cls = s.cls; bump(it, +1); }
            if (it.edited === it.cls) it.edited = null;
            if ('conf' in s) it.conf = s.conf;
            it.suspects = s.suspects || null;
            Lx.idents.set(ident(it), i);
          }
          repaint(i);
        }
      }
    }
    updateCount(); updateBulk(); chipsLater();
  }

  async function followJob(job, label) {
    const clock = ui.elapsedClock(Date.now() - (job.elapsed || 0) * 1000);
    const txt = h('span', label);
    statusR.replaceChildren(h('span', { class: 'ob-note' }, ui.spinner(), txt, clock));
    const snap = await api.watchJob(job.id, (s) => { txt.textContent = `${label} · ${fmt.num(s.done)} / ${fmt.num(s.total)}`; });
    if (snap.status !== 'done') throw new Error(snap.status === 'cancelled' ? 'cancelled — some files may have changed; reload to see them' : (snap.error || 'job failed'));
    return snap.result || {};
  }

  async function undo() {
    if (digitBuf) { clearDigit(); note('Class change cancelled.', 'info'); return; }    // the newest change was not made yet
    await chain;                     // an edit still being saved is the newest change: undo that one, not the one before
    const u = undoStack.pop();
    if (!u) { note('Nothing to undo.', 'info'); return; }
    if (!u.snapshot_id) { note('This change cannot be undone: label snapshots are not available.', 'warn'); return; }
    let kept = 0;                    // backend-data hook: mode 'undo' keeps a file edited since (Annotate, another tab)
    try {
      const res = await api.post(`/api/datasets/${ds.id}/snapshots/${u.snapshot_id}/restore`, { image_ids: u.image_ids, mode: 'undo' });
      if (res && res.job) kept = (await followJob(res.job, 'Undoing')).changed_since || 0;
    } catch (e) {
      note(e.status === 404 ? 'That change can no longer be undone (its snapshot was pruned).' : `Undo failed: ${e.message}`, 'danger');
      return;
    }
    if (u.Lx === L) {
      for (const b of u.items) {
        const it = b.it;
        bump(it, -1);
        it.cls = b.cls; it.deleted = b.deleted; it.edited = b.edited; it.conflict = null; it.external = false;
        bump(it, +1);
        repaint(b.i);
      }
      chipsLater();
    }
    // Findings accepted with A are open again: the boxes they named are back as they were.
    if (u.preds && u.preds.length) await api.post(`/api/datasets/${ds.id}/proposals/decide`, { accept: [], reject: [], pending: u.preds }).catch(() => {});
    await reconcile(u.image_ids);     // also lines up a list reloaded since the edit
    note(`Undone: ${u.op.del ? `${fmt.num(u.n)} deleted` : `${fmt.num(u.n)} reassigned`}.${kept ? ` ${fmt.plural(kept, 'image')} edited since kept as they are.` : ''}`, kept ? 'warn' : 'ok', undoStack.length ? { label: `Undo  ${MOD}Z`, run: () => undo() } : null);
    renderStatusL(); schedulePanel(); if (peekOn) showPeek();
  }

  // ---------------------------------------------------------------- Annotate hand-off
  let opening = false;
  async function openAnnotate(i) {
    const it = L && L.slots[i]; if (!it || opening) return;
    if (digitBuf) {
      const ask = needsConfirm(digitBuf.idx, digitBuf.d);
      commitDigit();
      if (ask) return;               // the confirmation decides first; Enter again opens Annotate
    }
    // Annotate reads the label file as it opens: let edits still being saved land first.
    opening = true;
    const slow = setTimeout(() => { if (!destroyed) note('Saving before opening Annotate…', 'info'); }, 400);
    try { await chain; } finally { clearTimeout(slow); opening = false; }
    if (destroyed) return;
    saveRestore();
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(filt.extra)) p.set(k, v);
    if (filt.split) p.set('split', filt.split);
    if (filt.q) p.set('q', filt.q);
    // Annotate's class filter means "images containing": include this box's class so its image stays in that list.
    if (filt.classes.size && !it.deleted) p.set('cls', [...new Set([...filt.classes, it.cls])].sort((a, b) => a - b).join(','));
    p.set('img', it.image_id);
    if (!it.deleted) p.set('box', it.row);
    fovea.router.navigate(`/d/${ds.id}/annotate?${p}`);
  }
  const POS_KEY = `fovea.obj.pos:${ds.id}`;
  function saveRestore() {
    if (!L || L.total == null) return;
    try { sessionStorage.setItem(POS_KEY, JSON.stringify({ q: myQuery, focus, top: scroller.scrollTop })); } catch (e) { /* ignore */ }
  }
  try {
    const r = JSON.parse(sessionStorage.getItem(POS_KEY) || 'null');
    // Only when coming back to the very same list (Back from Annotate); a different filter starts at the top.
    if (r && r.q === myQuery) restore = r;
  } catch (e) { /* ignore */ }
  function applyRestore() {
    const r = restore; if (!r || !L || !L.total) return;
    restore = null;
    if (r.top) scroller.scrollTop = r.top;
    if (r.focus >= 0 && r.focus < L.total) {
      focus = r.focus; anchor = r.focus;
      if (L.slots[r.focus]) { repaintSel(r.focus); renderStatusL(); schedulePanel(); }
      else restore = { q: r.q, focus: r.focus, top: 0 };   // its page is on its way
    }
    schedule();
  }

  // ---------------------------------------------------------------- status line
  let noteTimer = 0;
  function note(text, tone = 'info', action = null) {
    clearTimeout(noteTimer);
    const kids = [h('span', { class: `ob-note ${tone}` }, text)];
    if (action) kids.push(h('button', { class: 'btn btn-sm btn-ghost', onClick: action.run }, action.label));
    statusR.replaceChildren(...kids);
    // A static line in a fixed place; it is cleared after a while only so stale news is not mistaken for new.
    if (tone !== 'danger') noteTimer = setTimeout(() => { if (!destroyed && statusR.firstChild && statusR.firstChild.textContent === text) statusR.replaceChildren(); }, 20000);
  }
  function renderStatusL() {
    const it = L && focus >= 0 ? L.slots[focus] : null;
    if (!it) {
      statusL.replaceChildren(h('span', { class: 'faint' }, L && L.total ? `Arrows move · 0–9 reassign · Del delete · Enter opens in Annotate · hold Space to see the frame` : ''));
      return;
    }
    const bw = it.width ? Math.round(Math.abs(it.w) * it.width) : null, bh = it.height ? Math.round(Math.abs(it.h) * it.height) : null;
    put(statusL,
      h('span', { class: 'mono faint' }, `${fmt.num(focus + 1)} / ${fmt.num(L.total)}`),
      h('span', { class: 'ob-sw', style: `background:${classColor(it.cls)}` }),
      h('span', { class: 'strong' }, className(names, it.cls)),
      it.edited != null && !it.deleted ? h('span', { class: 'faint' }, `was ${className(names, it.edited)}`) : null,
      h('span', { class: 'mono truncate', title: it.rel_path }, midEllipsis(basename(it.rel_path), 48)),
      h('span', { class: 'faint' }, boxNo(it)),
      bw != null ? h('span', { class: 'mono faint' }, `${bw}×${bh} px`) : null,
      it.deleted ? h('span', { class: 'ob-note danger' }, it.external ? 'removed elsewhere' : 'deleted') : null,
      digitBuf ? h('span', { class: 'kbd' }, `${digitBuf.d}…`) : null);
  }

  // ---------------------------------------------------------------- full frame (peek + details panel)
  const frameImgs = new Map();   // url -> Promise<img>, small LRU
  const labelsOf = new Map();    // image_id+version -> Promise<{boxes}>
  function frameImg(url) {
    let p = frameImgs.get(url);
    if (p) { frameImgs.delete(url); frameImgs.set(url, p); return p; }
    p = decodedImage(url);
    p.catch(() => frameImgs.delete(url));
    frameImgs.set(url, p);
    while (frameImgs.size > 8) frameImgs.delete(frameImgs.keys().next().value);
    return p;
  }
  function imageBoxes(it) {
    const key = `${it.image_id}:${it.version}`;
    let p = labelsOf.get(key);
    if (!p) {
      p = api.get(`/api/datasets/${ds.id}/images/${it.image_id}/labels`).then((r) => r.boxes || []).catch(() => []);
      labelsOf.set(key, p);
      while (labelsOf.size > 40) labelsOf.delete(labelsOf.keys().next().value);
    }
    return p;
  }
  /** Full frame with every box of the image, the focused one emphasised. Resolves once decoded. */
  async function frameView(it, px, maxW, maxH) {
    const url = api.thumbUrl(it.image_id, px, it.mtime);
    const [img, boxes] = await Promise.all([frameImg(url), imageBoxes(it)]);
    const W = img.naturalWidth || it.width || 4, H = img.naturalHeight || it.height || 3;
    const k = Math.min(maxW / W, maxH / H);
    const pic = h('div', { class: 'ob-fpic', style: `width:${Math.max(1, Math.floor(W * k))}px;height:${Math.max(1, Math.floor(H * k))}px` });
    const im = img.parentNode ? img.cloneNode() : img;
    im.alt = ''; im.draggable = false;
    pic.appendChild(im);
    const same = (b) => Math.abs(b[1] - it.xc) < 1e-7 && Math.abs(b[2] - it.yc) < 1e-7 && Math.abs(b[3] - it.w) < 1e-7 && Math.abs(b[4] - it.h) < 1e-7;
    for (const b of boxes) {
      if (same(b)) continue;
      pic.appendChild(h('div', { class: 'ob-fbox', style: `--c:${classColor(b[0])};left:${(b[1] - b[3] / 2) * 100}%;top:${(b[2] - b[4] / 2) * 100}%;width:${b[3] * 100}%;height:${b[4] * 100}%` }));
    }
    const focusBox = h('div', { class: cls('ob-fbox focus', it.deleted && 'deleted'), style: `--c:${classColor(it.cls)};left:${(it.xc - it.w / 2) * 100}%;top:${(it.yc - it.h / 2) * 100}%;width:${it.w * 100}%;height:${it.h * 100}%` },
      h('span', { class: 'ob-flbl', style: `background:${classColor(it.cls)};color:${ink(it.cls)}` }, className(names, it.cls)));
    // A loose-box finding carries the tighter box the AI suggests: dashed, like every proposal. Placed before the
    // focused box so that box's class label stays readable on top.
    for (const s of it.deleted ? [] : (it.suspects || [])) {
      if (s.kind !== 'loose' || !Array.isArray(s.box)) continue;
      const [x, y, w, hh] = s.box;
      pic.appendChild(h('div', { class: 'ob-fbox suggest', style: `--c:${classColor(it.cls)};left:${(x - w / 2) * 100}%;top:${(y - hh / 2) * 100}%;width:${w * 100}%;height:${hh * 100}%` }));
    }
    pic.appendChild(focusBox);
    return pic;
  }
  async function showPeek() {
    const it = L && focus >= 0 ? L.slots[focus] : null;
    if (!it) return;
    const my = ++peekToken;
    const W = gridWrap.clientWidth - 32, H = gridWrap.clientHeight - 64;
    let pic;
    try { pic = await frameView(it, 1024, W, H); } catch (e) { if (my === peekToken && peekOn) note('The full frame could not be loaded.', 'danger'); return; }
    if (my !== peekToken || !peekOn) return;
    // Swap the whole view in one frame, only now that the picture is decoded.
    peek.replaceChildren(
      h('div', { class: 'ob-peek-top' }, h('span', { class: 'mono' }, midEllipsis(it.rel_path, 80)), h('span', { class: 'faint' }, `${boxNo(it)} · ${className(names, it.cls)}`), h('span', { class: 'spacer' }), h('span', { class: 'faint small' }, 'Release Space to return')),
      h('div', { class: 'ob-peek-stage' }, pic));
    peek.classList.remove('hidden');
  }
  function hidePeek() { peekOn = false; peekToken++; peek.classList.add('hidden'); peek.replaceChildren(); }

  let panelTimer = 0, panelToken = 0;
  function schedulePanel() { if (!panelOn) return; clearTimeout(panelTimer); panelTimer = setTimeout(renderPanel, 60); }
  async function renderPanel() {
    if (!panelOn || destroyed) return;
    const it = L && focus >= 0 ? L.slots[focus] : null;
    const my = ++panelToken;
    if (!it) { panel.replaceChildren(h('div', { class: 'ob-panel-empty small faint' }, 'Focus a box (click or arrow keys) to see it in its frame.')); return; }
    const pw = Math.max(160, panel.clientWidth - 24);
    let pic = null;
    try { pic = await frameView(it, 512, pw, pw); } catch (e) { pic = h('div', { class: 'small faint' }, 'Frame could not be loaded'); }
    if (my !== panelToken || destroyed) return;
    const bw = it.width ? Math.round(Math.abs(it.w) * it.width) : null, bh = it.height ? Math.round(Math.abs(it.h) * it.height) : null;
    const kv = (k, v) => [h('b', k), h('span', v)];
    const sus = (it.suspects || []).map((s) => h('div', { class: 'ob-sus' }, h('span', { class: 'strong' }, `${(SUSPECT[s.kind] || {}).glyph || ''} ${(SUSPECT[s.kind] || {}).label || s.kind}`),
      s.kind === 'swap' && s.cls != null ? h('span', ` → ${className(names, s.cls)}`) : null, s.conf != null ? h('span', { class: 'faint mono' }, ` ${s.conf.toFixed(2)}`) : null,
      s.kind === 'loose' ? h('div', { class: 'small faint' }, 'Dashed outline: the suggested fit. Adjust it in Annotate (Enter).') : null,
      s.reason ? h('div', { class: 'small muted' }, s.reason) : null));
    put(panel,
      h('div', { class: 'ob-panel-frame' }, pic),
      h('div', { class: 'ob-panel-sec' },
        h('div', { class: 'row' }, h('span', { class: 'ob-sw', style: `background:${classColor(it.cls)}` }), h('span', { class: 'strong grow truncate' }, `${it.cls} · ${className(names, it.cls)}`),
          it.deleted ? ui.chip(it.external ? 'Removed elsewhere' : 'Deleted', { type: 'danger' }) : it.edited != null ? ui.chip(`was ${className(names, it.edited)}`, { type: 'accent' }) : null),
        it.conflict ? h('div', { class: 'callout warn small' }, icon('alert', 14), `Not changed: ${it.conflict}`) : null,
        h('div', { class: 'ob-kv' },
          kv('Image', h('span', { class: 'mono truncate', title: it.rel_path }, midEllipsis(basename(it.rel_path), 30))), kv('Split', it.split || '—'), kv('Box', `#${it.row + 1}`),
          kv('Size', bw != null ? `${bw} × ${bh} px` : `${(it.w * 100).toFixed(1)} × ${(it.h * 100).toFixed(1)} %`),
          kv('Aspect', bw && bh ? (bw / bh).toFixed(2) : '—'), kv('Area', `${(Math.abs(it.w * it.h) * 100).toFixed(2)} % of frame`),
          it.conf != null ? kv('Conf', it.conf.toFixed(3)) : null, it.stale ? kv('Index', 'out of date — rescan') : null),
        sus.length ? h('div', { class: 'col gap-4' }, h('div', { class: 'small strong muted' }, 'AI audit'), sus) : null,
        h('div', { class: 'row wrap' },
          acceptable(it) ? h('button', { class: 'btn btn-sm', onClick: () => acceptSuggestion(targets()) }, icon('sparkles', 13), acceptable(it), h('span', { class: 'kbd' }, 'A')) : null,
          h('button', { class: 'btn btn-sm btn-primary', onClick: () => openAnnotate(focus) }, icon('pen', 13), 'Annotate', h('span', { class: 'kbd' }, '↵')),
          !it.deleted ? h('button', { class: 'btn btn-sm', onClick: (e) => pickClass(e.currentTarget, 'Reassign to', (c) => reassign(targets(), c)) }, icon('tag', 13), 'Reassign', h('span', { class: 'kbd' }, 'C')) : null,
          !it.deleted ? h('button', { class: 'btn btn-sm', onClick: () => remove(targets()) }, icon('trash', 13), 'Delete') : null)));
  }

  // ---------------------------------------------------------------- keyboard
  // With 11+ classes, "1" may be the start of "12", so it waits briefly for a second digit. The boxes it applies
  // to are taken when the key is pressed, and anything that moves the focus or the selection (or leaves the
  // view) applies it at once — never to whichever box has the focus when the wait runs out.
  function clearDigit() { if (digitBuf) { clearTimeout(digitBuf.timer); digitBuf = null; renderStatusL(); } }
  /** Whether reassign(idx, c) would ask first (the rule in reassign). */
  function needsConfirm(idx, c) {
    const n = L ? idx.filter((i) => L.slots[i] && !L.slots[i].deleted && L.slots[i].cls !== c).length : 0;
    return n > 200 || (!undoable && n > 1);
  }
  /** Apply a buffered first digit now. `ask: false` when no one is left to answer a confirmation (the view or
      the list is going away): a change that would need one is dropped instead. */
  function commitDigit({ ask = true } = {}) {
    const b = digitBuf;
    if (!b) return;
    clearDigit();
    if (b.Lx !== L || (!ask && needsConfirm(b.idx, b.d))) return;
    reassign(b.idx, b.d);
  }
  function onDigit(d) {
    const n = names.length;
    if (digitBuf) {
      const b = digitBuf; clearDigit();
      const two = b.d * 10 + d;
      if (b.Lx === L) {
        if (two < n) { reassign(b.idx, two); return; }
        reassign(b.idx, b.d);              // not a two-digit class: the first digit stands, this one starts over
      }
    }
    const idx = targets();
    if (d > 0 && d * 10 < n && idx.length) {
      digitBuf = { d, idx, Lx: L, timer: setTimeout(() => commitDigit(), 550) };
      renderStatusL();
      return;
    }
    reassign(idx, d);
  }
  const MODIFIER_KEYS = new Set(['Shift', 'Control', 'Alt', 'Meta', 'CapsLock', 'Fn', 'Process']);
  const SELECT_KEYS = new Set(['ArrowUp', 'ArrowDown', 'ArrowLeft', 'ArrowRight', 'PageUp', 'PageDown', 'Home', 'End']);
  let lastG = 0;
  const onKey = (e) => {
    if (destroyed || !L) return;
    if (fovea.isTyping() || ui.hasModal()) return;
    // Any other key ends a buffered first digit (applies it); Escape and Undo take it back.
    if (digitBuf && !(keys.digit(e) && keys.plain(e) && !e.shiftKey) && !MODIFIER_KEYS.has(e.key)) {
      if (keys.is(e, 'Escape') || keys.is(e, 'mod+z')) { e.preventDefault(); clearDigit(); note('Class change cancelled.', 'info'); return; }
      const ask = needsConfirm(digitBuf.idx, digitBuf.d);
      commitDigit();
      if (ask) { e.preventDefault(); return; }      // its confirmation takes this key's place
    }
    // `g` starts the global "go to" chord: its second key belongs to the shell, not to this view.
    if (keys.is(e, 'g')) { lastG = e.timeStamp; return; }
    if (lastG && e.timeStamp - lastG < 1000) { lastG = 0; return; }
    if (document.querySelector('.menu')) return;          // an open menu / popover owns the keyboard
    if (keys.is(e, 'mod+z')) { e.preventDefault(); undo(); return; }
    if (keys.is(e, 'mod+a')) { e.preventDefault(); selectAllLoaded(); return; }
    if (!keys.plain(e)) return;
    const d = keys.digit(e);
    if (d && !e.shiftKey) { e.preventDefault(); if (!e.repeat) onDigit(Number(d)); return; }
    const n = L.total || 0;
    const a = document.activeElement;
    const onSelect = a && a.tagName === 'SELECT';
    const onControl = a && (a.tagName === 'BUTTON' || a.tagName === 'A' || onSelect) && !canvas.contains(a);
    if (onControl && (keys.is(e, 'Enter') || keys.is(e, ' '))) return;   // the control's own activation
    if (onSelect && SELECT_KEYS.has(e.key)) return;     // a focused Split / Sort menu moves its own choice; other hotkeys still work there
    const step = (delta, extend) => { e.preventDefault(); setFocus(focus < 0 ? firstVisible() : focus + delta, { extend }); };
    if (keys.is(e, 'ArrowRight') || keys.is(e, 'shift+ArrowRight')) step(1, e.shiftKey);
    else if (keys.is(e, 'ArrowLeft') || keys.is(e, 'shift+ArrowLeft')) step(-1, e.shiftKey);
    else if (keys.is(e, 'ArrowDown') || keys.is(e, 'shift+ArrowDown')) step(cols, e.shiftKey);
    else if (keys.is(e, 'ArrowUp') || keys.is(e, 'shift+ArrowUp')) step(-cols, e.shiftKey);
    else if (keys.is(e, 'PageDown')) step(cols * Math.max(1, Math.floor(scroller.clientHeight / rowH) - 1), false);
    else if (keys.is(e, 'PageUp')) step(-cols * Math.max(1, Math.floor(scroller.clientHeight / rowH) - 1), false);
    else if (keys.is(e, 'Home')) { e.preventDefault(); setFocus(0); }
    else if (keys.is(e, 'End')) { e.preventDefault(); setFocus(n - 1); }
    else if (keys.is(e, 'Enter')) { if (focus >= 0) { e.preventDefault(); openAnnotate(focus); } }
    else if (keys.is(e, 'Escape')) { if (peekOn) hidePeek(); else if (selected.size) clearSelection(); }
    else if (keys.is(e, ' ')) { e.preventDefault(); if (!e.repeat && focus >= 0 && !peekOn) { peekOn = true; showPeek(); } }
    else if (keys.is(e, 'Delete')) { e.preventDefault(); if (!e.repeat) remove(targets()); }
    else if (keys.is(e, 's')) { if (focus >= 0) { e.preventDefault(); toggleSelect(focus); } }
    else if (keys.is(e, 'c')) {
      const t = targets(); if (!t.length) return;
      e.preventDefault();
      const anchorEl = tiles.get(focus) || statusBar;
      pickClass(anchorEl, t.length > 1 ? `Reassign ${fmt.num(t.length)} boxes to` : 'Reassign to', (c) => reassign(targets(), c));
    }
    else if (keys.is(e, 'a')) { if (!e.repeat) { e.preventDefault(); acceptSuggestion(targets()); } }
    else if (keys.is(e, 'o')) { e.preventDefault(); setOutline(!showOutline); }
    else if (keys.is(e, 'i')) { e.preventDefault(); togglePanel(); }
    else if (keys.is(e, '[')) { e.preventDefault(); setSize(SIZE_ORDER[Math.max(0, SIZE_ORDER.indexOf(size) - 1)]); }
    else if (keys.is(e, ']')) { e.preventDefault(); setSize(SIZE_ORDER[Math.min(SIZE_ORDER.length - 1, SIZE_ORDER.indexOf(size) + 1)]); }
  };
  const onKeyUp = (e) => { if ((e.code === 'Space' || e.key === ' ') && peekOn) hidePeek(); };
  const onBlur = () => { if (peekOn) hidePeek(); };
  function firstVisible() { const { r0 } = visibleRange(); return Math.min((L.total || 1) - 1, r0 * cols); }
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKeyUp);
  window.addEventListener('blur', onBlur);
  const offScan = fovea.bus.on('dataset:scanned', (id) => { if (id === ds.id) reload({ keepPosition: true }); });

  // ---------------------------------------------------------------- boot
  renderChips(); renderStatusL();
  reload({ keepPosition: !!restore });

  return () => {
    commitDigit({ ask: false });     // typed before leaving: it still counts (saved in the background)
    destroyed = true;
    saveRestore();
    if (L) L.ctrl.abort();
    cancelAnimationFrame(raf); clearTimeout(blockTimer); clearTimeout(noteTimer); clearTimeout(panelTimer); clearTimeout(chipsTimer);
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('keyup', onKeyUp);
    window.removeEventListener('blur', onBlur);
    offScan && offScan();
    ro.disconnect();
    cropCache.clear(); frameImgs.clear();
  };
}
