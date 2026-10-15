// Home: dataset registry cards + Add dataset wizard + file browser picker.
import { thumbPx, trackImage } from '../lib/media.js';
import { parseClassNames, duplicateNames } from '../lib/classnames.js';
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };

let F;
export function install(fovea) { F = fovea; F.openAddDataset = openAddDataset; F.pickPath = pickPath; }

const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};
const SPLIT_ORDER = ['train', 'val', 'valid', 'validation', 'test'];
export const sortSplits = (list) => list.slice().sort((a, b) => {
  const ia = SPLIT_ORDER.indexOf(String(a).toLowerCase()), ib = SPLIT_ORDER.indexOf(String(b).toLowerCase());
  return (ia < 0 ? 99 : ia) - (ib < 0 ? 99 : ib) || String(a).localeCompare(String(b), 'en', { numeric: true });
});
/** The card footer's split list, longest first: every name, then whole names from the start with "+N" for the
    rest, then just the count. The footer shows the first that fits — a name is never cut in the middle. */
export function splitLabels(splits) {
  const n = splits.length, sep = '\u00a0/\u00a0', out = [];
  for (let k = n; k >= 1; k--) out.push(k === n ? splits.join(sep) : `${splits.slice(0, k).join(sep)}${sep}+${n - k}`);
  if (n > 1) out.push(`${n} splits`);
  return out;
}
/** Show the longest label that fits the element's width (its width never depends on its text: flex-basis 0). */
function fitSplits(el) {
  const opts = el._labels || [];
  for (const t of opts) {
    if (el.textContent !== t) el.textContent = t;
    if (el.scrollWidth <= el.clientWidth + 1) return;
  }
}
/** What the Home filter matches: the name, the id and the last folders of the path — not the long common
    prefix every dataset shares (which made "set" match everything under ~/datasets). */
const haystack = (d) => `${d.name} ${d.id} ${(d.root_host || '').split('/').filter(Boolean).slice(-2).join(' ')}`.toLowerCase();

export async function render(ctx, fovea) {
  const { h, icon, logo, fmt, ui, link, cls } = fovea;
  const page = h('div', { class: 'page' });
  const inner = h('div', { class: 'page-inner' });
  page.appendChild(inner);
  document.title = 'Fovea';
  let datasets = fovea.state.get().datasets;
  const search = h('input', { class: 'input', placeholder: 'Filter by name or folder…', 'aria-label': 'Filter datasets', type: 'search' });
  let sort = ls.get('fovea.home.sort', 'recent'), show = 'all';
  const sortSel = h('select', { class: 'select auto', 'aria-label': 'Sort datasets', onChange: (e) => { sort = e.target.value; ls.set('fovea.home.sort', sort); renderGrid(); } },
    [['recent', 'Recently opened'], ['name', 'Name'], ['size', 'Most images'], ['added', 'Recently added']].map(([v, l]) => h('option', { value: v, selected: v === sort }, l)));
  const showSeg = ui.seg([{ value: 'all', label: 'All' }, { value: 'busy', label: 'Indexing' }, { value: 'attention', label: 'Needs attention' }], show, (v) => { show = v; renderGrid(); }, { label: 'Show' });
  const grid = h('div', { class: 'ds-cards', role: 'list' });
  const sub = h('div', { class: 'page-sub' });
  const heroSlot = h('div');
  const addCard = h('button', { class: 'ds-card add-card', type: 'button', onClick: () => openAddDataset() }, icon('plus', 28), h('div', { class: 'strong' }, 'Add dataset'), h('div', { class: 'small' }, 'Folder, data.yaml or train.txt'));
  const cards = new Map();   // id -> { el, key }
  const splitFit = typeof ResizeObserver === 'function' ? new ResizeObserver((es) => es.forEach(e => fitSplits(e.target))) : null;

  const needsAttention = (d) => (d.reachable && d.reachable.ok === false) || d.status === 'error' || d.status === 'new' || d.status === 'partial';
  const busy = (d) => d.status === 'scanning' || fovea.state.get().jobs.some(j => j.dataset_id === d.id);

  function renderGrid() {
    const { datasetsLoaded, datasetsError } = fovea.state.get();
    const q = search.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    let list = datasets.filter(d => q.every(w => haystack(d).includes(w)));
    if (show === 'busy') list = list.filter(busy); else if (show === 'attention') list = list.filter(needsAttention);
    const coll = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
    list.sort(sort === 'name' ? (a, b) => coll.compare(a.name, b.name)
      : sort === 'size' ? (a, b) => (b.image_count || 0) - (a.image_count || 0)
        : sort === 'added' ? (a, b) => (b.created_at || 0) - (a.created_at || 0)
          : (a, b) => (b.opened_at || b.created_at || 0) - (a.opened_at || a.created_at || 0));
    sub.textContent = datasetsError ? 'Could not load the dataset list' : !datasetsLoaded ? 'Loading…' : `${fmt.plural(datasets.length, 'dataset')} registered · index stored locally`;
    put(heroSlot, datasetsLoaded && !datasetsError && !datasets.length ? h('div', { class: 'hero' }, logo(56), h('div', null, h('h2', 'Welcome to Fovea'),
      h('p', 'A local-first workspace for reviewing and labeling computer-vision datasets. Register a YOLO-style folder to index images, labels and boxes, then explore, review and annotate — no upload, no accounts.'),
      h('div', { class: 'row mt-12' }, h('button', { class: 'btn btn-primary', onClick: () => openAddDataset() }, icon('plus', 14), 'Add your first dataset')))) : null);
    if (datasetsError && !datasets.length) {
      put(grid, ui.emptyState({ icon: 'wifiOff', title: 'Could not load datasets', message: datasetsError, action: h('button', { class: 'btn', onClick: () => fovea.loadDatasets().catch(() => {}) }, icon('refresh', 13), 'Try again') }));
      return;
    }
    const want = [addCard];
    const seen = new Set();
    for (const d of list) {
      seen.add(d.id);
      const job = fovea.state.get().jobs.find(j => j.dataset_id === d.id && ['queued', 'running'].includes(j.status)) || (d.job && ['queued', 'running'].includes(d.job.status) ? d.job : null);
      const key = JSON.stringify([d, job && [job.id, job.kind, job.status, job.done, job.total, job.message]]);
      let c = cards.get(d.id);
      if (!c || c.key !== key) {
        const el = card(d, job);
        if (c) { const old = c.el.querySelector('.splits'); if (old && splitFit) splitFit.unobserve(old); c.el.replaceWith(el); }
        c = { el, key }; cards.set(d.id, c);
      }
      want.push(c.el);
    }
    for (const [id, c] of cards) if (!seen.has(id)) { const sp = c.el.querySelector('.splits'); if (sp && splitFit) splitFit.unobserve(sp); c.el.remove(); cards.delete(id); }
    // Reorder in place; nodes that keep their position are not touched, so focus inside a card survives a poll.
    if (grid.firstChild && grid.firstChild.classList && grid.firstChild.classList.contains('empty')) put(grid);
    want.forEach((el, i) => { if (grid.children[i] !== el) grid.insertBefore(el, grid.children[i] || null); });
    while (grid.children.length > want.length) grid.lastChild.remove();
    if (datasetsLoaded && datasets.length && list.length === 0) grid.appendChild(h('div', { class: 'empty compact', style: 'grid-column:1/-1' }, h('p', 'No dataset matches.'), h('button', { class: 'btn btn-sm', onClick: () => { search.value = ''; show = 'all'; showSeg.setValue('all'); renderGrid(); } }, 'Clear filter')));
  }

  function card(d, job) {
    const rv = d.review || {}; const total = d.image_count || 0;
    const pct = (n) => total ? (100 * n / total) + '%' : '0%';
    const reviewed = (rv.approved || 0) + (rv.flagged || 0) + (rv.excluded || 0);
    const kind = (d.layout && d.layout.kind) || '';
    const reach = d.reachable || { ok: true };
    const href = `/d/${encodeURIComponent(d.id)}`;
    let cover;
    if (d.cover_image_id) {
      // Card covers are few and above the fold — load them straight away rather than leaving it to
      // the lazy heuristic, which can leave a registry page full of blank cards.
      const cimg = h('img', { src: fovea.api.thumbUrl(d.cover_image_id, thumbPx(320)), decoding: 'async', alt: '' });
      cover = h('div', { class: 'pic' }, cimg);
      trackImage(cimg, cover);
    } else {
      cover = h('div', { class: 'no-cover' }, icon('images', 28));
    }
    const st = !reach.ok ? 'unreachable' : (job && job.kind === 'scan') || d.status === 'scanning' ? 'scanning' : d.status;
    const stTip = { unreachable: 'Files not found at this path', scanning: 'Indexing', ready: `Indexed ${fmt.ago(d.scanned_at)}`, new: 'Not indexed yet', error: 'Indexing failed' }[st] || st;
    const splits = sortSplits((d.splits || []).filter(Boolean));
    let jobLine = null;
    if (job) {
      const started = (job.started_at || job.created_at || Date.now() / 1000) * 1000;
      jobLine = h('div', { class: 'job-line' }, icon(fovea.jobs.icon(job.kind), 12), h('span', { class: 'txt' }, fovea.jobs.text(job)),
        ui.progress(job.progress || 0, { indeterminate: !job.total }), ui.elapsedClock(started));
    }
    return h('article', { class: cls('ds-card', !reach.ok && 'unreachable'), role: 'listitem', 'aria-label': d.name },
      h('div', { class: 'cover' }, cover, kind ? ui.chip(kind, { cls: 'kind' }) : null,
        h('button', { class: 'btn btn-sm btn-icon menu-btn', 'aria-label': `Actions for ${d.name}`, 'aria-haspopup': 'menu', onClick: (e) => { e.stopPropagation(); e.preventDefault(); cardMenu(e.currentTarget, d); } }, icon('moreH', 14))),
      h('div', { class: 'body' },
        h('div', { class: 'name' }, h('a', { class: 'stretch nm', href, title: d.name }, d.name), h('span', { class: 'status-pill', title: stTip }, h('span', { class: `dot ${st}` }), h('span', { class: 'sr-only' }, stTip))),
        h('div', { class: 'path' }, ui.pathText(d.root_host, { tail: 30 })),
        jobLine,
        !reach.ok ? h('div', { class: 'warn-row' }, icon('alert', 13), h('span', { class: 'truncate' }, 'Files not found at this path — use ⋯ → Move to a new path')) : null,
        h('div', { class: 'meta' }, h('span', h('b', fmt.num(total)), total === 1 ? ' image' : ' images'), h('span', h('b', fmt.num(d.label_count)), ' labeled'), h('span', h('b', fmt.num(d.box_count)), d.box_count === 1 ? ' box' : ' boxes'),
          h('span', h('b', fmt.num((d.classes || []).length)), (d.classes || []).length === 1 ? ' class' : ' classes')),
        h('div', { class: 'review-bar', 'data-tip': `Approved ${fmt.num(rv.approved || 0)} · Flagged ${fmt.num(rv.flagged || 0)} · Excluded ${fmt.num(rv.excluded || 0)}`, role: 'img', 'aria-label': `${fmt.pct(reviewed, total)} reviewed` },
          h('i', { class: 'a', style: `width:${pct(rv.approved || 0)}` }), h('i', { class: 'f', style: `width:${pct(rv.flagged || 0)}` }), h('i', { class: 'x', style: `width:${pct(rv.excluded || 0)}` })),
        h('div', { class: 'foot' }, h('span', `${fmt.pct(reviewed, total)} reviewed`), h('span', '·'), splitsEl(splits),
          h('span', { title: d.opened_at ? `Opened ${fmt.date(d.opened_at)}` : null }, fmt.ago(d.opened_at || d.created_at))),
      ));
  }

  function splitsEl(splits) {
    if (!splits.length) return h('span', { class: 'splits' }, 'no splits');
    const labels = splitLabels(splits);
    const el = h('span', { class: 'splits', title: splits.join(' / ') }, labels[0]);
    el._labels = labels;
    if (splitFit) splitFit.observe(el);
    return el;
  }
  function cardMenu(anchor, d) {
    const base = `/d/${encodeURIComponent(d.id)}`;
    ui.menu(anchor, [
      { label: 'Open', icon: 'external', onClick: () => fovea.router.navigate(base) },
      { label: 'Explore', icon: 'grid', onClick: () => fovea.router.navigate(`${base}/explore`) },
      { label: 'Annotate', icon: 'pen', onClick: () => fovea.router.navigate(`${base}/annotate`) },
      { sep: true },
      { label: 'Rescan', icon: 'refresh', onClick: () => rescan(d, false) },
      { label: 'Re-detect layout & rescan', icon: 'scan', onClick: () => rescan(d, true) },
      { label: 'Move to a new path…', icon: 'folderOpen', onClick: () => relocate(d) },
      { label: 'Copy path', icon: 'copy', onClick: () => ui.copyText(d.root_host) },
      { sep: true },
      { label: 'Remove from Fovea…', icon: 'trash', danger: true, onClick: async () => {
        if (!await ui.confirm({ title: `Remove “${d.name}”?`, message: 'Only the Fovea index, review marks and label versions are removed. Files on disk are untouched.', okLabel: 'Remove', danger: true })) return;
        try { await fovea.api.del(`/api/datasets/${encodeURIComponent(d.id)}`); ui.toast('Dataset removed'); await fovea.loadDatasets(); }
        catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
      } },
    ]);
  }

  async function relocate(d) {
    const p = await pickPath({ title: `Where is “${d.name}” now?`, files: true, start: d.root_host, accept: ['.yaml', '.yml', '.txt'] });
    if (!p) return;
    try {
      await fovea.api.post(`/api/datasets/${encodeURIComponent(d.id)}/relocate`, { path: p });
      ui.toast('Re-indexing from the new path…', { type: 'ok' });
      fovea.jobs.refresh(); await fovea.loadDatasets();
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  async function rescan(d, redetect) {
    try {
      const r = await fovea.api.post(`/api/datasets/${encodeURIComponent(d.id)}/scan`, { redetect });
      if (r.already_running) ui.toast('Indexing is already running');
      fovea.jobs.refresh(); await fovea.loadDatasets();
    } catch (e) {
      ui.toast(fovea.plainError(e), { type: 'error', action: /not found|unreachable|no such/i.test(e.message || '') ? { label: 'Move to a new path…', onClick: () => relocate(d) } : undefined });
    }
  }

  const unsub = fovea.state.subscribe((s) => { datasets = s.datasets; renderGrid(); }, (s) => s.datasets);
  const unsub2 = fovea.state.subscribe(() => renderGrid(), (s) => `${s.datasetsLoaded}|${s.datasetsError}`);
  const unsub3 = fovea.state.subscribe(() => renderGrid(), (s) => s.jobs);
  fovea.setView('home', page, () => { unsub(); unsub2(); unsub3(); if (splitFit) splitFit.disconnect(); });

  inner.appendChild(heroSlot);
  inner.appendChild(h('div', { class: 'page-head' },
    h('div', { class: 'grow' }, h('h1', { class: 'page-title' }, 'Datasets'), sub),
    h('div', { class: 'home-tools' }, showSeg, sortSel, h('div', { class: 'input-wrap' }, icon('search'), search),
      h('button', { class: 'btn btn-primary', onClick: () => openAddDataset() }, icon('plus', 14), 'Add dataset'))));
  inner.appendChild(grid);
  search.addEventListener('input', renderGrid);
  renderGrid();
  // Review progress and counts change in other views: refresh the list each time Home is shown (the cached
  // list renders first, so this never delays the page).
  fovea.loadDatasets().catch(() => {});
}

// ------------------------------------------------------------------ file browser picker
export function pickPath({ title = 'Choose a folder', files = true, start = '', accept = null, selectFiles = true } = {}) {
  const { h, icon, ui, fmt } = F;
  return new Promise((resolve) => {
    let current = start || '';
    let seq = 0, ctrl = null;
    const pathInput = h('input', { class: 'input mono', placeholder: '/path/to/folder', value: current, 'aria-label': 'Folder path', spellcheck: 'false', autocomplete: 'off' });
    const list = h('div', { class: 'fb-list', role: 'listbox', 'aria-label': 'Folder contents', tabindex: '-1' });
    const status = h('div', { class: 'small faint', 'aria-live': 'polite' });
    const errBox = h('div');
    const load = async (p, { thenUse = false } = {}) => {
      const my = ++seq;
      if (ctrl) ctrl.abort();
      ctrl = new AbortController();
      put(list, h('div', { class: 'wait-row' }, ui.spinner()));
      put(errBox);
      try {
        const data = await F.api.get('/api/fs/browse', { path: p, files: files ? 1 : 0 }, { signal: ctrl.signal });
        if (my !== seq) return false;
        current = data.current; pathInput.value = current;
        const rows = [];
        if (data.parent) rows.push(item('cornerUpLeft', '..', null, () => load(data.parent), 'dir', 'Parent folder'));
        data.dirs.forEach(d => rows.push(item('folder', d.name, null, () => load(d.path), 'dir')));
        if (files) data.files.forEach(f => {
          const ok = !accept || accept.some(ext => f.name.toLowerCase().endsWith(ext));
          rows.push(item(f.is_image ? 'image' : 'fileText', f.name, fmt.bytes(f.size), ok && selectFiles ? () => done(f.path) : null, ok ? '' : 'faint'));
        });
        put(list, ...rows);
        const c = data.counts || {};
        status.textContent = `${fmt.plural(c.dirs ?? data.dirs.length, 'folder')} · ${fmt.plural(c.files ?? data.files.length, 'file')}${c.images ? ` (${fmt.num(c.images)} images)` : ''}${data.truncated ? ' · list shortened' : ''}`;
        if (thenUse) done(current);
        return true;
      } catch (e) {
        if (my !== seq || (e && e.name === 'AbortError')) return false;
        put(list);
        const nearest = e.body && e.body.nearest;
        put(errBox, h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('div', { class: 'grow' }, F.plainError(e),
          nearest ? h('div', { class: 'mt-4' }, h('button', { class: 'btn btn-sm', onClick: () => load(nearest) }, `Open the closest folder that exists: ${nearest}`)) : null)));
        status.textContent = '';
        return false;
      }
    };
    function item(ic, name, size, onAct, extra = '', label) {
      const row = h('div', { class: `fb-item ${extra}`, role: 'option', tabindex: '0', 'aria-label': label || name, title: name, onClick: onAct || null, 'aria-disabled': onAct ? null : 'true' },
        icon(ic), h('span', name), size ? h('span', { class: 'size' }, size) : null);
      return row;
    }
    // Arrow keys walk the list; Enter opens a folder or picks a file; Backspace goes up.
    list.addEventListener('keydown', (e) => {
      const rows = [...list.querySelectorAll('.fb-item')];
      const i = rows.indexOf(document.activeElement);
      if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); const n = Math.max(0, Math.min(rows.length - 1, i + (e.key === 'ArrowDown' ? 1 : -1))); rows[n] && rows[n].focus(); }
      else if (e.key === 'Backspace' && rows[0] && rows[0].getAttribute('aria-label') === 'Parent folder') { e.preventDefault(); rows[0].click(); }
    });
    let api;
    const done = (p) => api.close(p);  // close(result) → onClose(result) resolves exactly once
    // "Use this folder" means the folder in the path box — a typed path is opened (and checked) first.
    const useCurrent = async () => {
      const typed = pathInput.value.trim();
      if (typed && typed !== current) { await load(typed, { thenUse: true }); return; }
      done(current);
    };
    api = ui.modal({
      title, size: 'lg', onClose: (r) => { if (ctrl) ctrl.abort(); resolve(typeof r === 'string' && r ? r : null); },
      initialFocus: pathInput,
      body: h('div', { class: 'fb' },
        h('div', { class: 'fb-path' }, icon('folderOpen', 16), pathInput, h('button', { class: 'btn', onClick: () => load(pathInput.value.trim()) }, 'Go')),
        errBox, list, status),
      footer: (a) => [h('span', { class: 'small faint grow' }, files && selectFiles ? 'Click a file to pick it, or use the folder shown above.' : 'Open a folder, then use it.'), h('button', { class: 'btn', onClick: () => a.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: useCurrent }, 'Use this folder')],
    });
    pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); load(pathInput.value.trim()); } });
    load(current);
  });
}

// ------------------------------------------------------------------ add dataset wizard
let addDialog = null;
const slugPreview = (s) => {
  const raw = String(s || '').trim();
  let out = raw.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '').slice(0, 40).replace(/-+$/, '');
  if (/[^\x00-\x7f]/.test(raw)) out = out ? `${out}-xxxxxx` : 'ds-xxxxxx';
  return out || 'dataset';
};
export function openAddDataset(prefillPath = '') {
  if (typeof prefillPath !== 'string') prefillPath = '';  // may be called as a click handler
  // One wizard at a time: a second open (double click, a stray key) focuses the one already there.
  if (addDialog && addDialog.backdrop.isConnected) { const i = addDialog.el.querySelector('input'); i && i.focus(); return; }
  const { h, icon, ui, fmt } = F;
  let detected = null, detectedPath = null, detectedAi = false, seq = 0, ctrl = null, nameEdited = false, idEdited = false;
  const pathInput = h('input', { class: 'input mono', placeholder: '/data/my-dataset  ·  data.yaml  ·  train.txt', value: prefillPath, 'aria-label': 'Dataset path', spellcheck: 'false', autocomplete: 'off' });
  const nameInput = h('input', { class: 'input', placeholder: 'Dataset name', 'aria-label': 'Name', onInput: () => { nameEdited = !!nameInput.value.trim(); syncSlug(); } });
  const idInput = h('input', { class: 'input mono', placeholder: 'dataset-id', 'aria-label': 'ID', spellcheck: 'false', onInput: () => { idEdited = !!idInput.value.trim(); syncSlug(); } });
  const slugHint = h('span', { class: 'hint' });
  const result = h('div', { 'aria-live': 'polite' });
  const detectBtn = h('button', { class: 'btn', onClick: () => detect(true) }, icon('scan', 14), 'Detect');
  // Optional class names, pasted in any common form (JSON list / map, COCO categories, names: block, lines).
  const classesArea = h('textarea', { class: 'input mono', rows: 3, spellcheck: 'false', 'aria-label': 'Class names', style: 'width:100%',
    placeholder: '["person", "car"]  ·  {"0": "person", "1": "car"}  ·  names: block  ·  one per line' });
  const classesHint = h('div', { class: 'hint' });
  let pastedClasses = null;
  function syncClasses() {
    const r = parseClassNames(classesArea.value), dupes = duplicateNames(r.names);
    pastedClasses = !r.error && !dupes.length && r.names.length ? r.names : null;
    classesHint.className = (r.error || dupes.length) ? 'hint warn-text' : 'hint';
    classesHint.textContent = !classesArea.value.trim() ? 'Optional — overrides names found in data.yaml / classes.txt. Leave empty to use those (or infer them from the labels).'
      : r.error || (dupes.length ? `Duplicate names: ${dupes.join(', ')}` : `${r.names.length} class${r.names.length === 1 ? '' : 'es'}: ${r.names.slice(0, 12).map((n, i) => `${i} ${n}`).join(' · ')}${r.names.length > 12 ? ' …' : ''}`);
  }
  classesArea.addEventListener('input', syncClasses);
  syncClasses();
  let registerBtn;
  function syncSlug() {
    const typed = idInput.value.trim(), basis = typed || nameInput.value.trim();
    const slug = slugPreview(basis);
    slugHint.textContent = basis ? (typed && typed !== slug ? `Will be saved as “${slug}” (URLs allow a–z, 0–9 and “-”)` : `URL: /d/${slug}`) : 'Used in URLs';
  }
  syncSlug();

  /** `ai`: folders not named images/ and labels/ are identified by an AI provider (Settings → AI). Detection while
      typing never asks one (a half-typed path may be a home folder); Detect, Enter, Browse and Register do. */
  let inflight = null;            // {path, ai} of the detection under way
  async function detect(force = false, ai = true) {
    const p = pathInput.value.trim();
    if (!p) { put(result); detected = null; detectedPath = null; registerBtn.disabled = true; return null; }
    if (!force && p === detectedPath && detected && (detectedAi || !ai)) return detected;
    // A name-only detection (typing paused) never cancels an AI one already reading this path (Enter, Detect).
    if (!ai && inflight && inflight.ai && inflight.path === p) return null;
    const my = ++seq;
    inflight = { path: p, ai };
    if (ctrl) ctrl.abort();
    ctrl = new AbortController();
    detectBtn.disabled = true; registerBtn.disabled = true;
    const clock = ui.elapsedClock();
    const cancelBtn = h('button', { class: 'btn btn-sm btn-ghost', hidden: true, onClick: () => { if (ctrl) ctrl.abort(); } }, 'Cancel');
    setTimeout(() => { if (my === seq && cancelBtn.isConnected) cancelBtn.hidden = false; }, 2000);
    // An unknown folder layout or annotation format is read by an AI provider (with a web search for a published
    // dataset's documentation): that takes tens of seconds the first time; the answer is kept afterwards.
    const slowNote = h('div', { class: 'xs faint', hidden: true }, ai ? 'Unknown folders or annotation formats are read by AI — with a web search for a published dataset’s documentation. This can take up to a minute the first time.' : '');
    setTimeout(() => { if (my === seq && slowNote.isConnected && ai) slowNote.hidden = false; }, 4000);
    put(result, h('div', { class: 'col gap-4' }, h('div', { class: 'row small faint' }, ui.spinner(), h('span', 'Detecting the layout…'), clock, cancelBtn), slowNote));
    try {
      const res = await F.api.get('/api/fs/detect', { path: p, ai: ai ? 1 : 0 }, { signal: ctrl.signal });
      if (my !== seq) return null;
      detected = res; detectedPath = p; detectedAi = ai;
      const lay = res.layout;
      // Suggestions follow the latest detection unless the person typed their own.
      if (!nameEdited) nameInput.value = res.suggested_name || '';
      if (!idEdited) idInput.value = res.suggested_id || '';
      syncSlug();
      const bad = lay.kind === 'unknown' || !lay.sources.length;
      const total = lay.sources.reduce((n, s) => n + (s.image_count || 0), 0);
      const capped = lay.sources.some(s => s.count_capped);
      put(result, h('div', { class: 'detect-box' },
        h('div', { class: 'row wrap' }, ui.chip(lay.kind, { type: bad && lay.kind !== 'collection' ? 'danger' : 'accent' }), h('span', { class: 'small muted grow' }, LAYOUT_DESC[lay.kind] || ''),
          lay.data_yaml ? ui.chip('data.yaml', { icon: 'fileText' }) : null,
          lay.roles_from === 'ai' ? ui.chip('Folders found by AI', { icon: 'sparkles', title: [lay.roles_ai && lay.roles_ai.model, lay.roles_ai && lay.roles_ai.reason].filter(Boolean).join(' — ') || null })
            : lay.roles_from === 'names' ? ui.chip('Folders matched by name', { icon: 'folder' }) : null,
          total ? h('span', { class: 'small strong nowrap' }, `${fmt.num(total)}${capped ? '+' : ''} images`) : null),
        lay.sources.length ? h('div', { class: 'table-scroll' }, h('table', { class: 'table' }, h('thead', h('tr', h('th', 'Split'), h('th', { class: 'num' }, 'Images'), h('th', { class: 'num' }, 'Labels'), h('th', 'Source'))),
          h('tbody', lay.sources.map(s => h('tr', h('td', { class: 'truncate', style: 'max-width:160px', title: s.split || '' }, s.split || h('span', { class: 'faint' }, '—')), h('td', { class: 'num' }, fmt.num(s.image_count) + (s.count_capped ? '+' : '')),
            h('td', { class: 'num' }, s.list_file ? 'auto' : s.ann_format && !s.label_dir_exists ? h('span', { class: 'small muted', title: 'Converted from the annotations when you register' }, 'converted') : (s.label_dir_exists ? fmt.num(s.label_count) : h('span', { class: 'faint' }, 'none'))),
            h('td', { class: 'name-cell' }, ui.pathText(relTo(s.list_file_host || s.img_dir_host, res.path_host || lay.root_host), { tail: 24 }))))))) : null,
        lay.kind === 'collection' && (lay.candidates || []).length ? h('div', { class: 'col gap-6' },
          h('span', { class: 'small muted' }, 'This folder holds several datasets. Pick one:'),
          h('div', { class: 'row wrap gap-6' }, lay.candidates.map(c => h('button', { class: 'btn btn-sm', type: 'button', title: c.path_host,
            onClick: () => { pathInput.value = c.path_host; nameEdited = false; idEdited = false; detect(true); } }, icon('folder', 13), c.name))))
        : bad ? h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', `No images were found in a layout Fovea knows. Pick the dataset folder (the one with images/ and labels/), its data.yaml, or a train.txt list.${detectedAi ? '' : ' Press Detect to let AI identify folders with other names.'}`))
          : lay.classes.length ? h('div', { class: 'row wrap gap-4' }, h('span', { class: 'small muted' }, `Classes (${lay.classes.length}):`), lay.classes.slice(0, 40).map((c, i) => ui.chip(`${i} ${c}`, { cls: 'outline' })), lay.classes.length > 40 ? h('span', { class: 'small faint' }, `+${lay.classes.length - 40} more`) : null)
            : h('div', { class: 'small faint' }, 'No class names found — they will be inferred from the labels (you can name them later in Overview).'),
        formatBlock(lay, res.previews || {}, detectedAi),
        (lay.notes || []).filter(n => !/^Annotations (in |Fovea could not read)/.test(n)).map(n => h('div', { class: 'callout small' }, icon('info', 14), h('span', n))),
      ));
      registerBtn.disabled = bad;
      return res;
    } catch (e) {
      if (my !== seq) return null;
      detected = null; detectedPath = null;
      put(result, e && e.name === 'AbortError' ? h('div', { class: 'small faint' }, 'Detection cancelled.') : h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', F.plainError(e))));
      registerBtn.disabled = true;
      return null;
    } finally { if (my === seq) { detectBtn.disabled = false; clock.stop(); inflight = null; } }
  }
  const detectSoon = F.debounce(() => detect(false, false), 700);
  /** The folder mapping shown, to tell whether an AI re-detect at Register changed it. */
  const mapping = (lay) => JSON.stringify((lay.sources || []).map(s => [s.split, s.img_dir, s.label_dir, s.list_file]));

  async function register() {
    registerBtn.disabled = true;
    // One click registers: a path typed since the last detection is detected first.
    const p = pathInput.value.trim();
    const shown = p && p === detectedPath && detected ? detected : null;
    const res = shown && detectedAi ? shown : await detect(true);
    if (!res || res.layout.kind === 'unknown' || !res.layout.sources.length) { registerBtn.disabled = !res; return; }
    // Typing showed a name-only guess; the AI check found other folders: show those before anything is registered.
    if (shown && shown !== res && mapping(shown.layout) !== mapping(res.layout)) {
      ui.toast('AI found a different folder layout — check it, then register');
      registerBtn.disabled = false;
      return;
    }
    if (classesArea.value.trim() && !pastedClasses) { ui.toast('Fix the class names or clear them', { type: 'error' }); classesArea.focus(); registerBtn.disabled = false; return; }
    const lay = res.layout;
    const total = lay.sources.reduce((n, s) => n + (s.image_count || 0), 0);
    const capped = lay.sources.some(s => s.count_capped);
    const rootish = /^(\/|[A-Za-z]:\\?|\/Users\/[^/]+|\/home\/[^/]+|~)$/.test((res.path_host || p).replace(/\/+$/, '') || '/');
    if (lay.kind === 'bare' && (capped || total > 20000 || rootish)) {
      const ok = await ui.confirm({ title: 'Index this whole folder?', okLabel: 'Index it',
        message: `${res.path_host || p} has ${fmt.num(total)}${capped ? '+' : ''} images in its subfolders and no labels. Fovea will index all of them and create a labels/ folder next to them. Pick the dataset's own folder instead if this is a parent folder.` });
      if (!ok) { registerBtn.disabled = false; return; }
    }
    const body = { path: p, name: nameInput.value.trim() || undefined, id: idInput.value.trim() || undefined, classes: pastedClasses || undefined };
    try {
      let r;
      try { r = await F.api.post('/api/datasets', body); }
      catch (e) {
        // The same folder is registered already: say which one and let the person open it or add a second copy.
        const dup = e.status === 409 && /already registered as '(.*)' \(id ([^)]+)\)/.exec(e.message || '');
        if (!dup) throw e;
        const choice = await new Promise((resolve) => {
          let v = null;
          ui.modal({ title: 'Folder already registered', onClose: () => resolve(v),
            body: h('p', { style: 'margin:0' }, `This folder is already registered as “${dup[1]}”. A second registration keeps its own index and review marks; the label files are shared.`),
            footer: (a) => [h('button', { class: 'btn', onClick: () => a.close() }, 'Cancel'),
              h('button', { class: 'btn', onClick: () => { v = 'open'; a.close(); } }, `Open “${dup[1]}”`),
              h('button', { class: 'btn btn-primary', onClick: () => { v = 'again'; a.close(); } }, 'Register anyway')] });
        });
        if (choice === 'open') { m.close(); F.router.navigate(`/d/${encodeURIComponent(dup[2])}`); return; }
        if (choice !== 'again') { registerBtn.disabled = false; return; }
        r = await F.api.post('/api/datasets', { ...body, allow_duplicate: true });
      }
      m.close(); ui.toast(`Registered “${r.dataset.name}” — indexing…`, { type: 'ok' });
      F.jobs.refresh();
      await F.loadDatasets().catch(() => {}); F.router.navigate(`/d/${encodeURIComponent(r.dataset.id)}`);
    } catch (e) { ui.toast(F.plainError(e), { type: 'error' }); registerBtn.disabled = false; }
  }

  const m = addDialog = ui.modal({
    title: 'Add dataset', size: 'lg', initialFocus: pathInput, onClose: () => { if (ctrl) ctrl.abort(); seq++; addDialog = null; },
    body: h('div', { class: 'col gap-16' },
      h('div', { class: 'field' }, h('label', { class: 'label' }, 'Path', h('span', { class: 'hint' }, '— folder, data.yaml or train.txt (host paths are mapped automatically)')),
        h('div', { class: 'row' }, pathInput, h('button', { class: 'btn', onClick: async () => { const p = await pickPath({ title: 'Choose dataset folder or file', files: true, start: pathInput.value.trim(), accept: ['.yaml', '.yml', '.txt'] }); if (p) { pathInput.value = p; detect(true); } } }, icon('folderOpen', 14), 'Browse'), detectBtn)),
      result,
      h('div', { class: 'fields cols-2' }, h('div', { class: 'field' }, h('label', { class: 'label' }, 'Name'), nameInput), h('div', { class: 'field' }, h('label', { class: 'label' }, 'ID'), idInput, slugHint)),
      h('div', { class: 'field' }, h('label', { class: 'label' }, 'Class names', h('span', { class: 'hint' }, '— optional; paste JSON, a data.yaml names: block or one per line')), classesArea, classesHint),
      h('div', { class: 'small faint' }, 'Registering builds a local index (images, labels, boxes, sizes). Nothing is uploaded; image files are never modified.'),
    ),
    footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'),
      registerBtn = h('button', { class: 'btn btn-primary', disabled: true, onClick: register }, icon('database', 14), 'Register & index')],
  });
  pathInput.addEventListener('keydown', (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); detect(true); } });
  pathInput.addEventListener('input', () => { registerBtn.disabled = !pathInput.value.trim(); detectSoon(); });
  if (prefillPath) detect(true);
}

function relTo(p, root) {
  if (!p) return '';
  const r = (root || '').replace(/\/+$/, '');
  return r && p.startsWith(r + '/') ? p.slice(r.length + 1) : p;
}

const FORMAT_SOURCE = { catalog: 'built-in format', ai: 'reading rule written by AI from sample lines', web: 'reading rule from the official documentation (found by AI)' };

/** Annotations in another format than YOLO: what they are, where the reading rule came from (with the official
    links), how the check on real files went, the classes and attributes, and sample images with the converted boxes
    drawn on them — before anything is written. */
function formatBlock(lay, previews, detectedAi) {
  const { h, icon, ui, fmt } = F;
  const fmts = Object.entries(lay.formats || {});
  if (!fmts.length) return null;
  return h('div', { class: 'col gap-8' }, fmts.map(([fid, f]) => {
    const card = f.card || {};
    if (f.state === 'unreadable') {
      return h('div', { class: 'callout warn small' }, icon('alert', 14), h('span',
        `Annotations next to the images are in a format Fovea could not read${f.why ? ` (${f.why})` : ''}. The images are indexed without labels; the original files are not touched.`,
        detectedAi ? ' In Settings → AI, “Read annotation formats” lets an AI read unknown formats.' : ' Press Detect to let AI read them.'));
    }
    if (f.state === 'converted') {
      return h('div', { class: 'small muted row gap-6' }, icon('check', 13), `Labels converted earlier from ${card.name || fid} annotations${f.counts && f.counts.boxes != null ? ` (${fmt.num(f.counts.boxes)} boxes)` : ''}.`);
    }
    const ck = f.check || {};
    const attrs = ((card.spec || {}).attributes || card.attributes_hint || []);
    const regions = Object.values(card.regions || {}).map(r => r.label).filter(Boolean);
    const links = (card.urls || []).slice(0, 3).map(u => h('a', { href: u, target: '_blank', rel: 'noopener noreferrer', 'data-external': '', class: 'small' }, u.replace(/^https?:\/\//, '').replace(/\/$/, '')));
    const pv = previews[fid] || [];
    return h('div', { class: 'fmt-box col gap-6' },
      h('div', { class: 'row wrap gap-6' }, ui.chip(card.name || fid, { icon: 'fileText', type: 'accent' }),
        h('span', { class: 'small muted grow' }, FORMAT_SOURCE[card.source] || card.source || ''),
        card.model ? h('span', { class: 'xs faint', title: 'The AI model that wrote the reading rule, and what reading it cost (not booked in Runs & cost: no dataset exists yet)' }, `${card.model}${card.cached ? ' · saved answer' : card.usd != null ? ` · $${Number(card.usd).toFixed(card.usd < 0.1 ? 3 : 2)}` : ''}`) : null),
      links.length ? h('div', { class: 'row wrap gap-8' }, icon('external', 12), links) : null,
      ck.deferred ? h('div', { class: 'small muted' }, 'A large file: it is read in full while registering.')
        : h('div', { class: 'small muted' }, `Checked on ${fmt.num(ck.checked || 0)} files: ${fmt.num(ck.boxes || 0)} boxes${ck.regions ? `, ${fmt.num(ck.regions)} ${regions[0] ? regions[0].toLowerCase() + 's' : 'regions'}` : ''}${ck.bad ? `, ${fmt.num(ck.bad)} unreadable lines` : ''}${ck.oob ? `, ${fmt.num(ck.oob)} outside the image` : ''}.`),
      (ck.classes || []).length && JSON.stringify(ck.classes) !== JSON.stringify(lay.classes || []) ? h('div', { class: 'row wrap gap-4' }, h('span', { class: 'small muted' }, `Classes (${ck.classes.length}):`), ck.classes.slice(0, 30).map((c, i) => ui.chip(`${i} ${c}`, { cls: 'outline' })), ck.classes.length > 30 ? h('span', { class: 'small faint' }, `+${ck.classes.length - 30} more`) : null) : null,
      attrs.length || regions.length ? h('div', { class: 'small muted' }, [attrs.length ? `Attributes: ${attrs.map(a => a.label || a.name).join(', ')}` : '', regions.length ? `Regions that are not objects: ${regions.join(', ')}` : ''].filter(Boolean).join(' · ')) : null,
      pv.length ? h('div', { class: 'fmt-previews' }, pv.map(p => h('figure', { title: p.name },
        h('img', { src: p.data, alt: `${p.name} with converted boxes`, width: p.width, height: p.height, loading: 'lazy' }),
        h('figcaption', { class: 'xs faint truncate' }, `${p.boxes} boxes${p.regions ? ` · ${p.regions} regions` : ''}`)))) : null,
      h('div', { class: 'xs faint' }, 'Registering writes YOLO labels into a new labels folder; the original annotation files are never changed. Attributes and regions stay available in Annotate, Explore and Objects.'));
  }));
}

const LAYOUT_DESC = {
  yaml: 'Ultralytics data.yaml', ultralytics: 'images/<split> + labels/<split>', 'split-first': '<split>/images + <split>/labels',
  flat: 'images/ + labels/', mixed: 'images and labels in one folder', bare: 'images only (labels/ will be created)', list: 'train.txt image list', collection: 'several datasets in one folder', unknown: 'not recognised',
  custom: 'image and label folders with other names',
};
