// Export tab: subsets (ZIP / copy) with class remapping, near-duplicate removal and a sequence-aware split
// builder, a live preview of what will be written, review lists, and the dataset's data.yaml.
import { sortSplits } from './home.js';
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };

export function install(fovea) { fovea.registerTab({ id: 'export', label: 'Export', icon: 'download', order: 50, render }); }

async function render(el, { dataset, fovea, refresh }) {
  const { h, icon, ui, fmt, cls } = fovea;
  const { classColor } = fovea.colors;
  const api = fovea.api;
  const ds = dataset; const names = ds.classes || [];
  const enc = encodeURIComponent(ds.id);
  let alive = true;
  const stops = [];
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner' }); page.appendChild(inner); el.appendChild(page);
  inner.appendChild(h('div', { class: 'page-head' }, h('div', { class: 'grow' }, h('h1', { class: 'page-title' }, 'Export'),
    h('div', { class: 'page-sub' }, 'Build training subsets, review lists and Ultralytics data.yaml from the index. Files in the dataset are never modified unless you write into its folder explicitly.'))));
  const grid = h('div', { class: 'export-grid' }); inner.appendChild(grid);
  const card = (title, sub, body, extra = '') => h('section', { class: `card ${extra}`, 'aria-label': title }, h('div', { class: 'card-header' }, h('h3', title), sub ? h('span', { class: 'sub' }, sub) : null), h('div', { class: 'card-body' }, body));

  // ------------------------------------------------------------ options
  const allSplits = sortSplits((ds.splits || []).filter(Boolean));
  const f = {
    split: new Set(allSplits), cls: new Set(), only: false, labeled: '', excludeReview: new Set(['excluded']), includeUnlabeled: true,
    resize: 'none', max: 1280, w: 640, hh: 640, yaml: true, card: false,
    dedupe: false, map: {}, compact: false, badClass: 'extend',
    builder: false, ratios: { train: 80, val: 10, test: 10 }, seed: 0, groupBy: 'seq', stratify: true,
  };
  // Unticking every split means "nothing", but the wire format cannot say that (an empty split filter
  // means "any split") — so the UI must refuse, or it would export the whole dataset.
  const noSplit = () => allSplits.length > 0 && f.split.size === 0;
  const filtersOf = () => {
    const o = {};
    if (f.split.size && f.split.size !== allSplits.length) o.split = [...f.split].join(',');
    if (f.cls.size) { if (f.only) o.only_cls = [...f.cls].join(','); else o.cls = [...f.cls].join(','); }
    if (f.labeled) o.labeled = f.labeled;
    if (f.excludeReview.size) o.exclude_review = [...f.excludeReview].join(',');
    return o;
  };
  const resizeSpec = () => f.resize === 'none' ? null : f.resize === 'max' ? { mode: 'max', size: f.max } : { mode: 'exact', width: f.w, height: f.hh };
  const ratiosOk = () => { const s = f.ratios.train + f.ratios.val + f.ratios.test; return f.ratios.train > 0 && s > 0; };
  const optionsOf = () => {
    const o = { filters: filtersOf(), resize: resizeSpec(), include_unlabeled: f.includeUnlabeled, include_yaml: f.yaml };
    if (f.card) o.dataset_card = true;
    const cm = {};
    for (const [k, v] of Object.entries(f.map)) if (v !== 'keep') cm[k] = v === 'drop' ? null : Number(v);
    if (Object.keys(cm).length) o.class_map = cm;
    if (f.compact) o.compact = true;
    if (f.badClass !== 'extend') o.bad_class = f.badClass;
    if (f.dedupe) o.drop_near_dups = true;
    if (f.builder) {
      const sum = f.ratios.train + f.ratios.val + f.ratios.test || 1;
      o.split_builder = { ratios: { train: f.ratios.train / sum, val: f.ratios.val / sum, test: f.ratios.test / sum }, seed: f.seed, group_by: f.groupBy, stratify: f.stratify };
    }
    return o;
  };
  const resizeBad = () => (f.resize === 'max' && !(f.max >= 32 && f.max <= 16384)) || (f.resize === 'exact' && !(f.w >= 32 && f.w <= 16384 && f.hh >= 32 && f.hh <= 16384));
  const blocked = () => noSplit() ? 'Pick at least one split.' : resizeBad() ? 'Resize sizes must be between 32 and 16384 px.' : f.builder && !ratiosOk() ? 'The training share must be above 0.' : '';

  // ------------------------------------------------------------ count (fast) + preview (debounced, newest wins)
  const countEl = h('div', { class: 'preview-count', 'aria-live': 'polite' }, '…');
  const countSub = h('div', { class: 'xs faint' });
  const blockEl = h('div', { class: 'small danger-text', role: 'alert' });
  let countSeq = 0, pvSeq = 0, lastPreview = null;
  const refreshCount = fovea.debounce(async () => {
    const my = ++countSeq;
    syncActions();
    if (noSplit()) { countEl.textContent = '0 images'; return; }
    countSub.textContent = 'counting…';
    try {
      // The count must describe exactly what the export writes, including "without labels" being off.
      const r = await api.get(`/api/datasets/${enc}/images`, { ...filtersOf(), has_label: f.includeUnlabeled ? null : 1, limit: 1 });
      if (my !== countSeq || !alive) return;
      countEl.textContent = fmt.plural(r.total, 'image');
      countSub.textContent = f.dedupe || f.builder ? 'before near-duplicates are dropped' : 'match these filters';
    } catch (e) { if (my === countSeq) { countEl.textContent = '?'; countSub.textContent = fovea.plainError(e); } }
  }, 120);
  const pvBody = h('div', null, h('div', { class: 'wait-row' }, ui.spinner()));
  const pvState = h('span', { class: 'xs faint' });
  const refreshPreview = fovea.debounce(async () => {
    const my = ++pvSeq;
    if (blocked()) { pvState.textContent = ''; put(pvBody, h('div', { class: 'small faint' }, blocked())); return; }
    put(pvState, ui.spinner(), ' updating…');
    let r;
    try { r = await api.post(`/api/datasets/${enc}/export/preview`, optionsOf()); }
    catch (e) { if (my === pvSeq && alive) { pvState.textContent = ''; put(pvBody, h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', fovea.plainError(e)))); } return; }
    if (my !== pvSeq || !alive) return;
    lastPreview = r; pvState.textContent = '';
    drawPreview(r);
    drawCleanupHints(r);
  }, 450);
  const changed = () => { syncActions(); refreshCount(); refreshPreview(); };

  // ------------------------------------------------------------ subset card
  const chk = (label, checked, on, attrs = {}) => h('label', { class: 'check', ...attrs }, h('input', { type: 'checkbox', checked, onChange: (e) => { on(e.target.checked); changed(); } }), label);
  let zipBusy = false, copyJob = null, zipCtrl = null;
  const zipBtn = h('button', { class: 'btn', onClick: () => downloadZip() }, icon('download', 14), 'Download ZIP');
  const copyBtn = h('button', { class: 'btn btn-primary', onClick: () => copyTo() }, icon('folderOpen', 14), 'Copy to folder…');
  const syncActions = () => { const b = !!blocked(); zipBtn.disabled = b || zipBusy; copyBtn.disabled = b || !!copyJob; blockEl.textContent = blocked(); };
  const resizeOpts = h('div', { class: 'row wrap' });
  const num = (v, w, on, label) => h('input', { class: 'input input-sm', type: 'number', min: 32, max: 16384, style: `width:${w}px`, value: v, 'aria-label': label, onInput: (e) => { on(Number(e.target.value)); changed(); } });
  const renderResize = () => {
    put(resizeOpts, ui.seg([{ value: 'none', label: 'Original' }, { value: 'max', label: 'Max side' }, { value: 'exact', label: 'Exact' }], f.resize, v => { f.resize = v; renderResize(); changed(); }, { label: 'Resize' }));
    if (f.resize === 'max') resizeOpts.appendChild(h('div', { class: 'row gap-4' }, num(f.max, 90, v => f.max = v, 'Longest side in pixels'), h('span', { class: 'small muted' }, 'px')));
    if (f.resize === 'exact') resizeOpts.appendChild(h('div', { class: 'row gap-4' }, num(f.w, 80, v => f.w = v, 'Width in pixels'), h('span', '×'), num(f.hh, 80, v => f.hh = v, 'Height in pixels'), h('span', { class: 'small muted' }, 'px')));
  };
  renderResize();
  const jobBox = h('div', { 'aria-live': 'polite' });
  const zipBox = h('div', { 'aria-live': 'polite' });
  grid.appendChild(card('Subset', 'images + labels in the Ultralytics layout', h('div', { class: 'col gap-12' },
    allSplits.length ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Splits'), h('div', { class: 'row wrap' }, allSplits.map(s => chk(h('span', { class: 'truncate', style: 'max-width:200px', title: s }, s), true, v => v ? f.split.add(s) : f.split.delete(s))))) : null,
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Classes', h('span', { class: 'hint' }, '— images containing any of')),
      h('div', { class: 'row wrap', style: names.length > 24 ? 'max-height:180px;overflow-y:auto' : '' }, names.length ? names.map((n, i) => chk(h('span', { class: 'row gap-4', style: 'min-width:0' }, h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'truncate', style: 'max-width:180px', title: n }, `${i} ${n}`)), false, v => v ? f.cls.add(i) : f.cls.delete(i))) : h('span', { class: 'faint small' }, 'no classes')),
      h('div', { class: 'row' }, chk('Only these classes (no other class present)', false, v => f.only = v))),
    h('div', { class: 'field' }, h('label', { class: 'label', for: 'exp-labeled' }, 'Label state'), h('select', { class: 'select auto', id: 'exp-labeled', onChange: e => { f.labeled = e.target.value; changed(); } }, [['', 'Any'], ['1', 'Labeled with boxes'], ['empty', 'Empty label file'], ['nobox', 'No boxes (empty or no label file)'], ['0', 'No label file']].map(([v, l]) => h('option', { value: v }, l)))),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Review'), h('div', { class: 'row wrap' }, chk('Skip excluded', true, v => v ? f.excludeReview.add('excluded') : f.excludeReview.delete('excluded')), chk('Skip flagged', false, v => v ? f.excludeReview.add('flagged') : f.excludeReview.delete('flagged')), chk('Include images without labels', true, v => f.includeUnlabeled = v))),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Resize'), resizeOpts, h('span', { class: 'hint' }, 'Normalised YOLO labels stay valid after resizing. Resized images are re-encoded as JPEG (EXIF orientation applied).')),
    h('div', { class: 'field' }, h('div', { class: 'row wrap' }, chk('Write data.yaml', true, v => f.yaml = v),
      chk('Add a dataset card (DATASET_CARD.md)', false, v => f.card = v)),
      h('span', { class: 'hint' }, 'The card describes the export for whoever trains on it: splits, classes with their definitions, data quality, review progress, label issues and the settings used here.')),
    h('div', { class: 'divider' }),
    h('div', { class: 'export-actions' }, h('div', { class: 'count-wrap' }, countEl, countSub), zipBtn, copyBtn),
    blockEl,
    zipBox,
    jobBox,
  )));

  // ------------------------------------------------------------ clean-up & splits card
  const dedupeInfo = h('span', { class: 'hint' }, 'Keeps one image of each group of near-identical frames (found by Overview → Data quality).');
  const mapBody = h('div');
  const unnamedBox = h('div');
  const renderMap = () => {
    const opts = (i) => [['keep', 'Keep'], ['drop', 'Drop its boxes'], ...names.map((n, j) => [String(j), `→ ${j} ${n}`]).filter(([v]) => v !== String(i))];
    put(mapBody, h('div', { class: 'table-scroll', style: names.length > 14 ? 'max-height:300px;overflow-y:auto' : '' }, h('table', { class: 'table remap-table' },
      h('thead', h('tr', h('th', 'Class'), h('th', 'In the export'))),
      h('tbody', names.map((n, i) => h('tr', h('td', { class: 'name-cell' }, h('div', { class: 'row gap-6', style: 'min-width:0' }, h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'truncate', title: n }, `${i} ${n}`))),
        h('td', h('select', { class: 'select', 'aria-label': `Export class ${i} as`, onChange: (e) => { f.map[i] = e.target.value; changed(); } }, opts(i).map(([v, l]) => h('option', { value: v, selected: (f.map[i] || 'keep') === v }, l))))))))),
      h('div', { class: 'row wrap mt-8' }, chk('Renumber classes without gaps', f.compact, v => f.compact = v), h('span', { class: 'hint' }, 'After dropping or merging classes, ids become 0, 1, 2 … and data.yaml names follow.')));
  };
  renderMap();
  const ratioIn = (k, label) => h('div', { class: 'field' }, h('label', { class: 'label' }, label), h('div', { class: 'input-affix' }, h('input', { class: 'input input-sm', type: 'number', min: 0, max: 100, step: 1, value: f.ratios[k], 'aria-label': `${label} percent`, onInput: (e) => { f.ratios[k] = Math.max(0, Number(e.target.value) || 0); changed(); } }), h('span', { class: 'affix' }, '%')));
  const builderBody = h('div', { class: 'col gap-8', hidden: true },
    h('div', { class: 'split-ratios' }, ratioIn('train', 'Train'), ratioIn('val', 'Val'), ratioIn('test', 'Test')),
    h('div', { class: 'row wrap' },
      h('div', { class: 'field' }, h('label', { class: 'label', for: 'exp-group' }, 'Keep together'), h('select', { class: 'select auto', id: 'exp-group', onChange: (e) => { f.groupBy = e.target.value; changed(); } },
        [['seq', 'Frames of one sequence'], ['seq+dups', 'Sequences and near-duplicates'], ['folder', 'Images of one folder'], ['image', 'Nothing (image by image)']].map(([v, l]) => h('option', { value: v, selected: v === f.groupBy }, l)))),
      h('div', { class: 'field' }, h('label', { class: 'label', for: 'exp-seed' }, 'Seed'), h('input', { class: 'input input-sm', id: 'exp-seed', type: 'number', min: 0, step: 1, value: f.seed, style: 'width:90px', onInput: (e) => { f.seed = Math.max(0, Math.floor(Number(e.target.value) || 0)); changed(); } }))),
    chk('Balance class proportions across splits', true, v => f.stratify = v),
    h('span', { class: 'hint' }, 'Whole groups go to one split, so near-identical frames never end up in both training and validation (which inflates validation scores). The same seed gives the same split.'));
  grid.appendChild(card('Clean-up & splits', 'applied while exporting — the dataset is not changed', h('div', { class: 'col gap-12' },
    h('div', { class: 'field' }, chk(h('span', { class: 'strong' }, 'Drop near-duplicate images'), false, v => f.dedupe = v), dedupeInfo),
    h('div', { class: 'divider', style: 'margin:0' }),
    h('div', { class: 'field' }, chk(h('span', { class: 'strong' }, 'Make new train / val / test splits'), false, v => { f.builder = v; builderBody.hidden = !v; }), builderBody),
    h('div', { class: 'divider', style: 'margin:0' }),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Classes', h('span', { class: 'hint' }, '— keep, merge into another class, or drop')), names.length ? mapBody : h('span', { class: 'small faint' }, 'No classes.')),
    unnamedBox)));

  function drawCleanupHints(r) {
    const nd = r.near_dups || {};
    dedupeInfo.textContent = !nd.available ? 'Run Overview → Data quality first: near-duplicates are found there.'
      : f.dedupe ? `${fmt.plural(nd.dropped || 0, 'image')} dropped from ${fmt.plural(nd.groups || 0, 'group')} of near-identical frames.`
        : 'Keeps one image of each group of near-identical frames (found by Overview → Data quality).';
    const un = r.unnamed || {};
    put(unnamedBox, un.rows ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Class ids beyond the class list'),
      h('div', { class: 'row wrap' }, ui.seg([{ value: 'extend', label: 'Keep (add class_N names)' }, { value: 'drop', label: 'Drop those boxes' }], f.badClass, (v) => { f.badClass = v; changed(); }, { label: 'Unnamed class ids' }),
        h('span', { class: 'small muted' }, `${fmt.plural(un.rows, 'box', 'boxes')} in ${fmt.plural(un.images, 'image')}`))) : null);
  }

  // ------------------------------------------------------------ preview card
  const yamlDetails = h('details', null, h('summary', { class: 'small strong', style: 'cursor:pointer' }, 'data.yaml in the export'));
  let cardOpen = false;
  const cardDetails = h('details', { class: 'mt-8', onToggle: (e) => { cardOpen = e.currentTarget.open; } });
  grid.appendChild(h('section', { class: 'card wide', 'aria-label': 'Export preview' }, h('div', { class: 'card-header' }, h('h3', 'What will be exported'), h('span', { class: 'sub' }, 'from the index — no files are read'), h('div', { class: 'actions' }, pvState)),
    h('div', { class: 'card-body' }, pvBody)));
  function drawPreview(r) {
    const cl = r.classes || {};
    const nameList = (cl.names || []).map((n, i) => `${i} ${n}`).join(', ');
    put(yamlDetails, h('summary', { class: 'small strong', style: 'cursor:pointer' }, 'data.yaml in the export'), r.yaml ? h('pre', { class: 'mono small yaml-pre mt-8' }, r.yaml) : h('div', { class: 'small faint mt-8' }, 'No data.yaml (turned off).'));
    cardDetails.open = cardOpen;
    put(cardDetails, h('summary', { class: 'small strong', style: 'cursor:pointer' }, 'DATASET_CARD.md in the export'),
      r.card ? h('div', { class: 'col gap-8 mt-8' }, h('pre', { class: 'mono small yaml-pre card-pre' }, r.card),
        h('div', { class: 'row' }, h('span', { class: 'xs faint grow' }, 'Written next to data.yaml. The counts are those of this preview.'),
          h('button', { class: 'btn btn-sm', onClick: () => saveText(r.card, `${(String(ds.name || ds.id).normalize('NFC').replace(/[^\p{L}\p{N}._-]+/gu, '-').replace(/^[-.]+|[-.]+$/g, '') || ds.id)}-DATASET_CARD.md`) }, icon('download', 12), 'Download this card')))
        : h('div', { class: 'small faint mt-8' }, 'Not included — tick “Add a dataset card” to write one with the export.'));
    put(pvBody, 
      h('div', { class: 'row wrap gap-16 mb-12' },
        h('div', null, h('div', { class: 'preview-count' }, fmt.num(r.images)), h('div', { class: 'xs faint' }, `images · ${fmt.num(r.labeled)} with labels`)),
        h('div', null, h('div', { class: 'preview-count' }, fmt.num(r.boxes)), h('div', { class: 'xs faint' }, 'boxes')),
        h('div', null, h('div', { class: 'preview-count' }, fmt.num(cl.nc ?? 0)), h('div', { class: 'xs faint' }, cl.remapped ? 'classes (remapped)' : 'classes')),
        r.near_dups && r.near_dups.dropped ? h('div', null, h('div', { class: 'preview-count' }, fmt.num(r.near_dups.dropped)), h('div', { class: 'xs faint' }, 'near-duplicates dropped')) : null,
        r.reencoded ? h('div', null, h('div', { class: 'preview-count' }, fmt.num(r.reencoded)), h('div', { class: 'xs faint' }, 're-encoded (resized)')) : null),
      r.splits && r.splits.length ? h('div', { class: 'pv-splits mb-12' }, r.splits.map(s => h('div', { class: 'pv-split' },
        h('div', { class: 'k', title: s.name }, s.yaml_key && s.yaml_key !== s.name ? `${s.name} → ${s.yaml_key}` : s.name),
        h('div', { class: 'v' }, fmt.num(s.images)), h('div', { class: 's' }, `${fmt.pct(s.images, r.images)} · ${fmt.plural(s.boxes, 'box', 'boxes')}`)))) : null,
      nameList ? h('div', { class: 'small muted mb-8', style: 'overflow-wrap:anywhere' }, h('span', { class: 'strong' }, 'Classes: '), nameList) : null,
      r.sequences && r.sequences.total ? h('div', { class: 'small muted mb-8' }, `${fmt.plural(r.sequences.total, 'sequence')}${r.sequences.spanning_splits ? ` · ${fmt.num(r.sequences.spanning_splits)} span more than one split` : ' · none spans two splits'}`) : null,
      (r.warnings || []).length ? h('div', { class: 'col gap-4 mb-8' }, r.warnings.map(w => h('div', { class: 'callout warn small' }, icon('alert', 14), h('span', w)))) : null,
      yamlDetails, cardDetails);
  }
  function saveText(text, filename) {
    const href = URL.createObjectURL(new Blob([text], { type: 'text/markdown;charset=utf-8' }));
    const a = h('a', { href, download: filename, 'data-external': '' }); document.body.appendChild(a); a.click(); a.remove();
    setTimeout(() => URL.revokeObjectURL(href), 30000);
  }

  // ------------------------------------------------------------ actions
  async function downloadZip() {
    if (zipBusy || blocked()) return;
    zipBusy = true; syncActions();
    const ctrl = zipCtrl = new AbortController(), t0 = Date.now();
    // Planning a large archive takes seconds before the browser's download starts: after 400 ms the action row
    // says so, with an elapsed clock and a Cancel (which abandons the request; leaving the page does too).
    const busy = setTimeout(() => {
      if (zipCtrl !== ctrl || !alive) return;
      put(zipBox, h('div', { class: 'row small muted zip-busy' }, ui.spinner(), h('span', { class: 'grow' }, 'Preparing the archive…'), ui.elapsedClock(t0),
        h('button', { class: 'btn btn-sm btn-ghost', onClick: () => ctrl.abort() }, 'Cancel')));
    }, 400);
    let started = false;
    try {
      // A one-time link the browser downloads natively: the archive streams to disk with the browser's own
      // progress, instead of fetch() holding a multi-GB blob in this tab until the very end.
      const r = await api.post(`/api/datasets/${enc}/export/zip-link`, { ...optionsOf(), filename: `${ds.id}-subset` }, { signal: ctrl.signal });
      if (!alive || ctrl.signal.aborted) return;
      const a = h('a', { href: r.url, download: r.filename || `${ds.id}-subset.zip`, 'data-external': '' }); document.body.appendChild(a); a.click(); a.remove();
      started = true;
      ui.toast(`Download started${r.bytes ? ` · ${fmt.bytes(r.bytes)}` : ''}${r.images != null ? ` · ${fmt.plural(r.images, 'image')}` : ''} — follow it in your browser’s downloads`, { type: 'ok', timeout: 6000 });
      (r.warnings || []).slice(0, 2).forEach(w => ui.toast(w, { timeout: 8000 }));
    } catch (e) { if (!(e && e.name === 'AbortError') && alive) ui.toast(`Export failed: ${fovea.plainError(e)}`, { type: 'error' }); }
    finally {
      clearTimeout(busy);
      if (zipCtrl === ctrl) { zipCtrl = null; if (alive) put(zipBox); }
      // one archive per click; a cancelled request frees the button at once
      if (started) setTimeout(() => { zipBusy = false; if (alive) syncActions(); }, 2500); else { zipBusy = false; if (alive) syncActions(); }
    }
  }

  async function copyTo() {
    if (blocked() || copyJob) return;
    const parent = await fovea.pickPath({ title: 'Choose where to create the export folder', files: false, selectFiles: false });
    if (!parent || !alive) return;
    const name = await ui.prompt({ title: 'Export folder name', label: `Created inside ${parent}`, value: `${ds.id}-${new Date().toISOString().slice(0, 10)}`, mono: true,
      validate: (v) => !v.trim() ? 'Enter a folder name' : /[\\/]/.test(v.trim()) ? 'A folder name cannot contain / or \\' : null });
    if (name == null || !alive) return;
    const target = parent.replace(/[\/\\]+$/, '') + '/' + name.trim();
    await startCopy(target, false);
  }
  async function startCopy(target, overwrite) {
    try {
      const { job } = await api.post(`/api/datasets/${enc}/export/copy`, { ...optionsOf(), target_dir: target, overwrite });
      fovea.jobs.refresh();
      followCopy(job, target, { card: f.card });
    } catch (e) {
      if (e.status === 409 && e.body && e.body.target && !overwrite) {
        const ok = await ui.confirm({ title: 'The folder is not empty', message: fovea.plainError(e), okLabel: e.body.target.previous_export ? 'Replace the previous export' : 'Write into it anyway', danger: true });
        if (ok) await startCopy(target, true);
        return;
      }
      ui.toast(fovea.plainError(e), { type: 'error' });
    }
  }
  function followCopy(job, target, { card = false } = {}) {
    copyJob = job; syncActions();
    const bar = ui.progress(job.progress || 0, { indeterminate: !job.total });
    const txt = h('span', { class: 'small muted grow truncate' }, job.message || 'Starting…');
    const clock = ui.elapsedClock((job.started_at || job.created_at || Date.now() / 1000) * 1000);
    const cancel = h('button', { class: 'btn btn-sm btn-ghost', onClick: async () => { cancel.disabled = true; try { await api.post(`/api/jobs/${job.id}/cancel`); } catch (err) { ui.toast(fovea.plainError(err), { type: 'error' }); } } }, 'Cancel');
    put(jobBox, h('div', { class: 'col gap-4' }, h('div', { class: 'row' }, icon('folderOpen', 13), h('span', { class: 'small strong' }, 'Copying to'), h('span', { class: 'grow', style: 'min-width:0;display:flex' }, ui.pathText(target || (job.extra && job.extra.target_host) || '', { tail: 28 })), clock, cancel), h('div', { class: 'row' }, txt), bar));
    const w = api.watchJob(job.id, (s) => { txt.textContent = s.total ? `${fmt.num(s.done)} / ${fmt.num(s.total)}${s.message ? ` · ${s.message}` : ''}` : (s.message || ''); bar.className = `progress ${s.total ? '' : 'indeterminate'}`; bar.firstChild.style.width = `${Math.round((s.progress || 0) * 100)}%`; },
      { onConnection: (ok) => { if (!ok) txt.textContent = 'Reconnecting to the server…'; } });
    stops.push(() => w.stop());
    const finish = (node) => { clock.stop(); copyJob = null; if (!alive) return; put(jobBox, node); syncActions(); };
    w.then((snap) => {
      if (snap.status === 'cancelled') { finish(h('div', { class: 'callout small' }, icon('info', 14), h('span', 'Copy cancelled — files copied before the cancel were kept.'))); ui.toast('Copy cancelled'); return; }
      const r = snap.result || {};
      const extra = [r.failed ? `${fmt.num(r.failed)} images could not be read and were skipped` : '', r.missing ? `${fmt.num(r.missing)} missing` : '', r.skipped_same_file ? `${fmt.num(r.skipped_same_file)} source files left untouched` : ''].filter(Boolean).join(' · ');
      finish(h('div', { class: `callout ${r.failed ? 'warn' : 'ok'} small` }, icon(r.failed ? 'alert' : 'checkCircle', 14), h('div', { class: 'grow' },
        h('div', `Done: ${fmt.plural(r.images || 0, 'image')}, ${fmt.plural(r.labels || 0, 'label file')}${r.dropped_near_dups ? `, ${fmt.num(r.dropped_near_dups)} near-duplicates dropped` : ''}${card ? ', DATASET_CARD.md' : ''}.`),
        r.target ? h('div', { class: 'row gap-4', style: 'min-width:0' }, h('span', { style: 'min-width:0;display:flex' }, ui.pathText(r.target, { tail: 30 })), h('button', { class: 'btn btn-sm btn-ghost', onClick: () => ui.copyText(r.target) }, icon('copy', 12), 'Copy path')) : null,
        extra ? h('div', { class: 'xs' }, extra) : null)));
      ui.toast('Subset copied', { type: r.failed ? 'error' : 'ok' });
    }).catch((e) => {
      finish(h('div', { class: 'callout danger small' }, icon('alertCircle', 14), h('span', `Copy failed: ${fovea.plainError(e)}`)));
    });
  }
  // A copy started earlier (another tab, before leaving this page) is picked up again.
  api.get('/api/jobs', { dataset_id: ds.id, kind: 'export_copy', active: 1 }).then(({ jobs }) => { if (alive && jobs[0] && !copyJob) followCopy(jobs[0], jobs[0].extra && jobs[0].extra.target_host); }).catch(() => {});

  // ------------------------------------------------------------ lists
  const lf = { status: 'excluded', style: 'host', split: '' };
  grid.appendChild(card('Review lists', 'plain-text path lists', h('div', { class: 'col gap-12' },
    h('div', { class: 'field' }, h('label', { class: 'label', for: 'exp-list-which' }, 'Which images'), h('select', { class: 'select', id: 'exp-list-which', onChange: e => lf.status = e.target.value }, [['excluded', 'Excluded'], ['flagged', 'Flagged'], ['approved', 'Approved'], ['none', 'Unreviewed'], ['any', 'All reviewed'], ['all', 'Every image (training list)'], ['keep', 'Every image except excluded (training list)']].map(([v, l]) => h('option', { value: v }, l)))),
    allSplits.length ? h('div', { class: 'field' }, h('label', { class: 'label', for: 'exp-list-split' }, 'Split'), h('select', { class: 'select', id: 'exp-list-split', onChange: e => lf.split = e.target.value }, [['', 'All splits'], ...allSplits.map(s => [s, s])].map(([v, l]) => h('option', { value: v }, l)))) : null,
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Path style'), ui.seg([{ value: 'host', label: 'Absolute' }, { value: 'rel', label: 'Relative to the list file' }], lf.style, v => lf.style = v, { label: 'Path style' })),
    h('div', { class: 'row wrap' }, h('span', { class: 'spacer' }),
      h('button', { class: 'btn', onClick: () => listAction(false) }, icon('download', 14), 'Download .txt'),
      h('button', { class: 'btn', onClick: () => listAction(true) }, icon('save', 14), 'Write into the dataset folder')),
  )));
  async function listAction(write, overwrite = false) {
    const filters = {}; if (lf.split) filters.split = lf.split;
    if (lf.status === 'all') { /* every image */ } else if (lf.status === 'keep') filters.exclude_review = 'excluded'; else filters.review = lf.status;
    const filename = `${ds.id}-${lf.status}${lf.split ? '-' + lf.split : ''}`;
    try {
      if (write) {
        const r = await api.post(`/api/datasets/${enc}/export/list`, { filters, style: lf.style, write: true, filename, overwrite });
        ui.toast(`Wrote ${fmt.plural(r.lines, 'line')} → ${r.path}`, { type: 'ok' }); return;
      }
      const res = await fetch(`/api/datasets/${enc}/export/list`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ filters, style: lf.style, filename }) });
      if (!res.ok) {
        // Without this check the server's error JSON would be saved as the list file, looking like success.
        let msg = res.statusText; try { msg = (await res.json()).detail || msg; } catch (_) { /* not JSON */ }
        ui.toast('List export failed: ' + msg, { type: 'error' }); return;
      }
      const blob = await res.blob(); const href = URL.createObjectURL(blob);
      const a = h('a', { href, download: filename + '.txt', 'data-external': '' }); document.body.appendChild(a); a.click(); a.remove();
      setTimeout(() => URL.revokeObjectURL(href), 30000);
    } catch (e) {
      if (write && e.status === 409 && e.body && e.body.exists && !overwrite) {
        if (await ui.confirm({ title: 'Replace the existing list?', message: fovea.plainError(e), okLabel: 'Replace', danger: true })) return listAction(true, true);
        return;
      }
      ui.toast(fovea.plainError(e), { type: 'error' });
    }
  }

  // ------------------------------------------------------------ yaml
  const yamlPre = h('pre', { class: 'mono small yaml-pre' }, '…');
  const yamlStatus = h('div', { class: 'small faint break' });
  const yamlBtn = h('button', { class: 'btn', onClick: () => writeYaml() }, icon('save', 14), 'Write data.yaml');
  grid.appendChild(card('data.yaml', 'Ultralytics dataset config in the dataset folder', h('div', { class: 'col gap-12' }, yamlPre, yamlStatus, h('div', { class: 'row' }, h('span', { class: 'spacer' }), yamlBtn))));
  let yamlPlan = null;
  async function loadYaml() {
    try { yamlPlan = await api.get(`/api/datasets/${enc}/export/yaml`); }
    catch (e) { yamlPre.textContent = ''; yamlStatus.textContent = fovea.plainError(e); yamlBtn.disabled = true; return; }
    if (!alive) return;
    yamlPre.textContent = yamlPlan.proposed || '';
    put(yamlStatus, yamlPlan.exists ? `Updates ${yamlPlan.path_host}: other keys and comments are kept, the previous file is saved as data.yaml.bak.` : `Creates ${yamlPlan.path_host}.`);
    yamlBtn.lastChild.textContent = yamlPlan.exists ? 'Update data.yaml' : 'Write data.yaml';
    yamlBtn.disabled = yamlPlan.exists && yamlPlan.current === yamlPlan.proposed;
    if (yamlBtn.disabled) yamlStatus.append(' It is already up to date.');
  }
  async function writeYaml() {
    if (!yamlPlan) return;
    if (!await ui.confirm({ title: yamlPlan.exists ? 'Update data.yaml?' : 'Write data.yaml?', message: yamlStatus.textContent, okLabel: yamlPlan.exists ? 'Update' : 'Write' })) return;
    try { const r = await api.post(`/api/datasets/${enc}/export/yaml`); ui.toast(`Wrote ${r.path}`, { type: 'ok' }); refresh && refresh(); loadYaml(); }
    catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  syncActions(); refreshCount(); refreshPreview(); loadYaml();
  return () => { alive = false; stops.forEach(s => s()); if (zipCtrl) zipCtrl.abort(); };
}
