// Compare plugin UI: evaluate prediction folders or stored AI runs against ground truth, tune the confidence
// threshold on live curves, read the confusion matrix, walk the per-image errors side by side, flag images.
import { confChart, prChart, confusionTable } from './charts.js';
import { openViewer, midEllipsis, VIEWER_KEYS } from './viewer.js';

const GRID_IOUS = new Set(['0.50', '0.55', '0.60', '0.65', '0.70', '0.75', '0.80', '0.85', '0.90', '0.95']);
const METRIC_OPTS = [
  ['coco', 'COCO', 'Greedy matching by confidence, 101-point interpolated AP (pycocotools)'],
  ['ultralytics', 'Ultralytics', 'Same matching and AP as yolo val — use this to compare with Ultralytics numbers'],
  ['voc', 'VOC', 'Greedy matching by confidence, all-point AP (Pascal VOC 2010+)'],
];
const SHOW_OPTS = [['errors', 'Images with errors'], ['all', 'All images'], ['fn', 'Misses (FN)'], ['fp', 'False positives'], ['swap', 'Class swaps'], ['clean', 'No errors'], ['regress', 'B worse than A'], ['improve', 'B better than A']];
const SORT_OPTS = [['errors', 'Most errors'], ['fn', 'Most misses'], ['fp', 'Most false positives'], ['swap', 'Most swaps'], ['delta', 'Biggest regression (B − A)'], ['gt', 'Most boxes'], ['name', 'Name']];

export default function install(fovea, manifest) {
  const API = manifest.api;
  const base = (manifest.entry || '/plugins/compare/compare.js').replace(/[^/]+$/, '');
  if (!document.querySelector('link[data-cmp-css]')) {
    const link = document.createElement('link');
    link.rel = 'stylesheet'; link.href = `${base}compare.css`; link.dataset.cmpCss = '1';
    document.head.appendChild(link);
  }
  if (fovea.keys && fovea.keys.registerKeys) fovea.keys.registerKeys('compare', 'Compare viewer', VIEWER_KEYS.map(([k, d]) => [k.replace(/⌘/g, fovea.MOD || '⌘'), d]), 60);
  fovea.registerTab({ id: 'compare', label: 'Compare', icon: 'gitCompare', order: 45, render: (el, c) => render(el, c, fovea, API) });
  fovea.registerCommand({ id: 'compare.open', label: 'Compare predictions vs ground truth', icon: 'gitCompare', when: (s) => !!s.currentDataset, run: (f) => f.router.navigate(`/d/${f.state.get().currentDataset.id}/compare`) });
}

/** replaceChildren that skips null / false like h() does (the DOM method would print "null"). */
const put = (el, ...kids) => el.replaceChildren(...kids.flat(Infinity).filter(k => k != null && k !== false));
const pct = (v, d = 1) => (v == null || isNaN(v) ? '–' : `${(v * 100).toFixed(d)}%`);
const mean = (a) => (a.length ? a.reduce((x, y) => x + y, 0) / a.length : null);

/** Headline numbers of one side at threshold index k over the selected classes (null = all). */
function calc(cv, ev, k, sel) {
  const gridIdx = ev.ious.map((v, i) => (GRID_IOUS.has(Number(v).toFixed(2)) ? i : -1)).filter(i => i >= 0);
  let T = 0, Fp = 0, G = 0; const ap = [], ap95 = [], precs = [], recs = [];
  for (const [c, d] of Object.entries(cv.classes)) {
    if (sel && !sel.has(+c)) continue;
    T += d.tp[k]; Fp += d.fp[k]; G += d.gt;
    if (d.tp[k] + d.fp[k] > 0) precs.push(d.tp[k] / (d.tp[k] + d.fp[k]));
    if (d.gt > 0) {
      recs.push(d.tp[k] / d.gt);
      if (d.ap[ev.primary] != null) { ap.push(d.ap[ev.primary]); const v = gridIdx.map(i => d.ap[i]).filter(x => x != null); if (v.length) ap95.push(mean(v)); }
    }
  }
  const p = T + Fp ? T / (T + Fp) : 0, r = G ? T / G : 0, f1 = p + r ? (2 * p * r) / (p + r) : 0;
  const ok = cv.conf_available !== false;
  return { tp: T, fp: Fp, gt: G, fn: G - T, p, r, f1, map: ok && ap.length ? mean(ap) : null, map95: ok && ap95.length ? mean(ap95) : null,
    macroP: mean(precs), macroR: mean(recs), nAp: ap.length };
}
function curveOf(cv, sel) {
  const p = [], r = [], f1 = [], n = [];
  const keys = Object.entries(cv.classes).filter(([c]) => !sel || sel.has(+c));
  for (let k = 0; k <= 100; k++) {
    let T = 0, F = 0, G = 0;
    for (const [, d] of keys) { T += d.tp[k]; F += d.fp[k]; G += d.gt; }
    const pp = T + F ? T / (T + F) : 0, rr = G ? T / G : 0;
    p.push(pp); r.push(rr); f1.push(pp + rr ? (2 * pp * rr) / (pp + rr) : 0); n.push(T + F);
  }
  return { p, r, f1, n };
}
function bestK(cv, sel) {
  if (cv.conf_available === false) return null;
  const { f1 } = curveOf(cv, sel);
  let b = 0; for (let k = 0; k <= 100; k++) if (f1[k] >= f1[b]) b = k;
  return f1[b] > 0 ? { k: b, f1: f1[b] } : null;     // no threshold helps a class the model never finds
}

async function render(el, { ctx, dataset }, F, API) {
  const { h, icon, ui, fmt, cls } = F;
  const ds = dataset;
  const names = ds.classes || [];
  // An evaluation may count several classes as one (an audit's class family, or "Count as one class"): the merged
  // class keeps its smallest id and reads as its group everywhere on the results.
  const cvLabels = () => (cur && cur.meta.classes_view && cur.meta.classes_view.labels) || {};
  const cname = (c) => cvLabels()[c] || F.colors.className(names, c);
  // box labels in the viewer stay short: 'person', not 'person (0+1)'
  const evNames = () => { const out = names.slice(); for (const [c, l] of Object.entries(cvLabels())) out[+c] = l.replace(/ \([\d+]+\)$/, ''); return out; };
  const page = h('div', { class: 'page cmp' }); const inner = h('div', { class: 'page-inner wide' }); page.appendChild(inner); el.appendChild(page);
  const cleanups = [];
  let destroyed = false;

  // ------------------------------------------------------------------ setup
  const blankSrc = () => ({ type: 'folder', path: '', run_id: null, missing: 'empty', remap: '' });
  const setup = { a: blankSrc(), b: blankSrc(), iou: 0.5, conf: 0.25, metric: 'coco', split: '', skip_excluded: true, unlabeled: 'background', merge: '' };
  const dsName = (c) => F.colors.className(names, c);
  let runs = null;
  const covState = { a: null, b: null };
  const srcViews = {};

  const runBtn = h('button', { class: 'btn btn-primary', onClick: () => runEval() }, icon('play', 14), 'Run evaluation');
  const progress = h('div', { class: 'cmp-progress' });
  const splitSel = h('select', { class: 'select', style: 'width:auto', 'aria-label': 'Split', onChange: (e) => { setup.split = e.target.value; recheck(); } },
    [['', 'All splits'], ...(ds.splits || []).filter(Boolean).map(s => [s, s])].map(([v, l]) => h('option', { value: v }, l)));
  const exclCb = h('input', { type: 'checkbox', checked: true, onChange: (e) => { setup.skip_excluded = e.target.checked; recheck(); } });
  const iouVal = h('span', { class: 'mono small', style: 'width:32px' }, '0.50');
  const iouIn = h('input', { type: 'range', class: 'slider', min: 0.05, max: 0.95, step: 0.05, value: 0.5, style: 'width:140px', 'aria-label': 'IoU threshold', onInput: (e) => { setup.iou = Number(e.target.value); iouVal.textContent = setup.iou.toFixed(2); } });
  const metricSel = h('select', { class: 'select', style: 'width:auto', 'aria-label': 'Metric definition', onChange: (e) => { setup.metric = e.target.value; metricHint.textContent = METRIC_OPTS.find(m => m[0] === setup.metric)[2]; } },
    METRIC_OPTS.map(([v, l]) => h('option', { value: v }, l)));
  const metricHint = h('span', { class: 'hint' }, METRIC_OPTS[0][2]);
  const confIn = h('input', { class: 'input', type: 'number', min: 0, max: 0.99, step: 0.01, value: 0.25, onChange: (e) => { const v = Number(e.target.value); setup.conf = isFinite(v) ? Math.max(0, Math.min(0.99, v)) : 0.25; e.target.value = setup.conf; syncMore(); } });
  const missingSel = h('select', { class: 'select', onChange: (e) => { setup.a.missing = setup.b.missing = e.target.value; recheck(); syncMore(); } },
    h('option', { value: 'empty' }, 'No detections'), h('option', { value: 'skip' }, 'Skip the image'));
  const unlabeledSel = h('select', { class: 'select', onChange: (e) => { setup.unlabeled = e.target.value; syncMore(); } },
    h('option', { value: 'background' }, 'Background'), h('option', { value: 'skip' }, 'Skip the image'));
  const remapIn = { a: null, b: null };
  for (const side of ['a', 'b']) remapIn[side] = h('input', { class: 'input mono', placeholder: 'e.g. 0:1, 1:0, 5:-', 'aria-label': `Class map for ${side.toUpperCase()}`, onChange: (e) => { setup[side].remap = e.target.value.trim(); syncMore(); } });
  const mergeHint = h('span', { class: 'hint' }, 'For a detector with coarser classes, e.g. a person model on standing + fallen.');
  const mergeIn = h('input', { class: 'input mono', placeholder: 'e.g. 0+1', 'aria-label': 'Count as one class',
    onChange: (e) => { setup.merge = e.target.value.trim(); syncMore(); mergeHint.textContent = mergeText(setup.merge) || 'For a detector with coarser classes, e.g. a person model on standing + fallen.'; } });
  const mergeText = (txt) => txt.split(/[,;]/).map(g => g.split(/[+&\s]+/).filter(Boolean)).filter(g => g.length > 1)
    .map(g => g.map(x => (/^\d+$/.test(x) ? `${x} · ${dsName(+x)}` : x)).join(' + ')).join('; ');
  const more = h('div', { class: 'cmp-more', hidden: true },
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Initial confidence threshold'), confIn, h('span', { class: 'hint' }, 'Only where the results open — move it freely afterwards.')),
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Image without a prediction file'), missingSel, h('span', { class: 'hint' }, 'Ultralytics writes no file for an image where it found nothing.')),
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Image without a label file'), unlabeledSel, h('span', { class: 'hint' }, 'Background: every prediction on it is a false positive.')),
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Class map for A (prediction:dataset)'), remapIn.a, h('span', { class: 'hint' }, '“-” drops a class. Unlisted classes keep their id.')),
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Class map for B'), remapIn.b),
    h('label', { class: 'field' }, h('span', { class: 'label' }, 'Count as one class'), mergeIn, mergeHint));
  const moreCount = h('span', { class: 'cmp-more-n' });
  const moreBtn = h('button', { class: 'btn btn-ghost btn-sm', 'aria-expanded': 'false', onClick: () => { more.hidden = !more.hidden; moreBtn.setAttribute('aria-expanded', String(!more.hidden)); } }, icon('sliders', 13), 'More options', moreCount);
  /** How many advanced settings differ from the defaults: a closed panel must not hide what changes the numbers. */
  function syncMore() {
    const n = [setup.a.missing !== 'empty', setup.unlabeled !== 'background', !!setup.a.remap, !!setup.b.remap, !!setup.merge, Math.abs(setup.conf - 0.25) > 1e-9].filter(Boolean).length;
    moreCount.textContent = n ? `${n} set` : '';
  }

  function sourceEditor(side) {
    const S = setup[side];
    const body = h('div', { class: 'col', style: 'gap:6px' });
    const cov = h('div', { class: 'cmp-cov', 'aria-live': 'polite' });
    const typeSeg = ui.seg([{ value: 'folder', label: 'Folder', icon: 'folder' }, { value: 'run', label: 'AI run', icon: 'sparkles' }], S.type, (v) => { S.type = v; draw(); recheck(side); });
    const draw = () => {
      typeSeg.setValue(S.type);
      if (S.type === 'folder') {
        // A long path shows its END (the distinguishing folder) whenever the field is not being edited.
        const input = h('input', { class: 'input mono', value: S.path, title: S.path || null, 'aria-label': `Model ${side.toUpperCase()} prediction folder`,
          placeholder: side === 'a' ? 'YOLO prediction folder' : 'Optional second model',
          onInput: (e) => { S.path = e.target.value.trim(); e.target.title = S.path; recheck(side); }, onBlur: (e) => { const el = e.target; requestAnimationFrame(() => { el.scrollLeft = el.scrollWidth; }); } });
        requestAnimationFrame(() => { if (document.activeElement !== input) input.scrollLeft = input.scrollWidth; });
        const browse = h('button', { class: 'btn', onClick: async () => { const p = await F.pickPath({ title: 'Choose a prediction label folder', files: false, selectFiles: false, start: input.value.trim() || ds.root_host }); if (p) { S.path = p; draw(); recheck(side); } } }, icon('folderOpen', 14), 'Browse');
        const disc = h('button', { class: 'btn', onClick: () => discover(side), 'data-tip': 'Scan a folder for prediction outputs and see how well each matches this dataset' }, icon('scan', 14), 'Discover…');
        body.replaceChildren(h('div', { class: 'row cmp-path-row' }, input, browse, disc, side === 'b' && S.path ? h('button', { class: 'btn btn-ghost btn-icon', 'aria-label': 'Clear model B', onClick: () => { S.path = ''; draw(); recheck(side); } }, icon('x', 14)) : null));
      } else {
        if (runs == null) { body.replaceChildren(h('div', { class: 'small muted' }, 'Loading AI runs…')); loadRuns().then(draw); return; }
        const runBtn = h('button', { class: 'btn', onClick: () => runModelDialog(side), 'data-tip': 'Run a model on this machine over the scope below and evaluate it' }, icon('play', 14), 'Run a model…');
        if (!runs.length) {
          body.replaceChildren(h('div', { class: 'row' }, h('span', { class: 'small muted grow' }, 'No AI runs with stored predictions yet.'), runBtn),
            h('div', { class: 'hint' }, 'Cloud models: run a benchmark, a batch with target “Proposals” or a label audit in the ', h('a', { href: `/d/${ds.id}/ai` }, 'AI tab'), ', then pick the run here.'));
          return;
        }
        const status = (r) => (r.status && r.status !== 'done' ? ` · ${r.status}` : '');
        // Short enough to read in the closed select at 1280 px; the coverage line below says the rest.
        const model = (r) => (r.audit && r.audit.members && r.audit.members.length ? r.audit.members[0].split(' · ').pop() : String(r.model || r.provider || '').split('/').pop());
        const runText = (r) => `${r.kind ? r.kind[0].toUpperCase() + r.kind.slice(1) : 'Run'} #${r.id} · ${model(r)} · ${fmt.num(r.processed ?? r.images_with_preds)} images · ${fmt.ago(r.created_at)}${status(r)}`;
        const ok = runs.filter(r => r.evaluable !== false), stale = runs.filter(r => r.evaluable === false);
        const sel = h('select', { class: 'select cmp-run-sel', 'aria-label': `Model ${side.toUpperCase()} AI run`, onChange: (e) => { S.run_id = e.target.value ? Number(e.target.value) : null; recheck(side); } },
          h('option', { value: '' }, side === 'a' ? 'Choose a run…' : 'None'),
          ok.map(r => h('option', { value: r.id, selected: S.run_id === r.id, title: `${r.label} · ${fmt.num(r.preds)} boxes` }, runText(r))),
          stale.length ? h('optgroup', { label: 'Audits without stored detections — run again to evaluate' },
            stale.map(r => h('option', { value: r.id, disabled: true, title: r.reason || '' }, runText(r)))) : null);
        put(body, h('div', { class: 'row' }, sel, runBtn),
          !ok.length ? h('div', { class: 'hint' }, stale[0].reason || 'No AI run with stored predictions yet.') : null);
      }
    };
    draw();
    srcViews[side] = { draw, cov };
    return h('div', { class: 'cmp-src' },
      h('div', { class: 'row' }, h('span', { class: `cmp-tag ${side}` }, side.toUpperCase()), h('span', { class: 'strong small' }, side === 'a' ? 'Model A' : 'Model B (optional)'), h('span', { class: 'spacer' }), typeSeg),
      body, cov);
  }

  async function loadRuns() {
    try { runs = (await F.api.get(`${API}/runs`, { dataset_id: ds.id })).runs; } catch (e) { runs = []; }
  }

  // ---- run a local model into a stored AI run, then evaluate that run
  async function runModelDialog(side) {
    const sel = h('select', { class: 'select', 'aria-label': 'Model' }, h('option', { value: '' }, 'Loading models…'));
    const confIn = h('input', { class: 'input', type: 'number', min: 0, max: 0.9, step: 0.01, value: 0.01, style: 'width:100px', 'aria-label': 'Confidence floor' });
    const mapIn = h('input', { class: 'input mono', placeholder: 'person:0, *:skip   (model class : dataset class id)', 'aria-label': 'Class map' });
    const mapHint = h('div', { class: 'small', 'aria-live': 'polite' });
    const mapField = h('label', { class: 'field' }, h('span', { class: 'label' }, 'Class map'), mapIn, mapHint);
    const scopeLine = h('div', { class: 'small muted' });
    const okBtn = h('button', { class: 'btn btn-primary', disabled: true }, icon('play', 13), h('span', 'Run'));
    let provs = [], mapOk = true, nImages = 0;
    const m = ui.modal({
      title: 'Run a model on this dataset', size: 'lg',
      body: h('div', { class: 'col', style: 'gap:12px' },
        h('div', { class: 'small muted' }, 'Runs a model on this machine over the evaluation scope and keeps every prediction as an AI run — nothing is written to your labels and nothing appears as a proposal in Annotate. Cloud models run from the ', h('a', { href: `/d/${ds.id}/ai` }, 'AI tab'), ', which estimates the cost and enforces a budget.'),
        h('label', { class: 'field' }, h('span', { class: 'label' }, 'Model'), sel),
        h('label', { class: 'field' }, h('span', { class: 'label' }, 'Confidence floor'), confIn, h('span', { class: 'hint' }, 'Keep it low: AP and the threshold curves need the low-confidence predictions (yolo val uses 0.001).')),
        mapField, scopeLine),
      footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'), okBtn],
    });
    const choice = () => { const [pid, mid] = (sel.value || '|').split('|'); return { provider: pid, model: mid }; };
    const closedVocab = (c) => c.provider === 'yolo' || (c.provider === 'yoloe' && /-pf$/.test(c.model));
    const sync = () => { okBtn.disabled = !sel.value || !mapOk || !nImages; okBtn.lastChild.textContent = nImages ? `Run on ${fmt.num(nImages)} image${nImages === 1 ? '' : 's'}` : 'Run'; };
    let mSeq = 0;
    const preview = F.debounce(async () => {
      const c = choice(); const my = ++mSeq;
      mapField.hidden = !closedVocab(c);
      if (!closedVocab(c)) { mapOk = true; sync(); return; }
      mapHint.textContent = 'Checking the class map…';
      let cmap = null;
      try { cmap = parseMap(mapIn.value); } catch (e) { mapOk = false; mapHint.textContent = e.message; mapHint.className = 'small danger'; sync(); return; }
      try {
        const r = await F.api.post('/api/ai/mapping', { dataset_id: ds.id, provider: c.provider, model: c.model, class_map: cmap });
        if (my !== mSeq) return;
        const mapped = r.items.filter(x => x.cls != null);
        mapOk = mapped.length > 0;
        mapHint.className = mapOk ? 'small muted' : 'small danger';
        mapHint.textContent = mapOk
          ? `${mapped.slice(0, 6).map(x => `${x.model_name} → ${dsName(x.cls)}`).join(', ')}${mapped.length > 6 ? ` and ${mapped.length - 6} more` : ''} · ${fmt.num(r.items.length - mapped.length)} other model classes ignored`
          : `None of the model's ${r.items.length} classes map to this dataset — name them here, e.g. “${r.items[0] ? r.items[0].model_name : 'person'}:0”.`;
      } catch (e) { if (my === mSeq) { mapOk = false; mapHint.className = 'small danger'; mapHint.textContent = e.message; } }
      sync();
    }, 250);
    const parseMap = (txt) => {
      const out = {};
      for (const tok of txt.split(/[,;]/).map(t => t.trim()).filter(Boolean)) {
        const mm = tok.match(/^(.+?)\s*(?:->|→|:|=)\s*(.*)$/);
        if (!mm) throw new Error(`“${tok}” needs the form model_class:dataset_class`);
        const v = mm[2].trim();
        out[mm[1].trim()] = ['-', '', 'skip', 'none'].includes(v.toLowerCase()) ? 'skip' : (/^-?\d+$/.test(v) ? Number(v) : v);
      }
      return Object.keys(out).length ? out : null;
    };
    sel.addEventListener('change', preview); mapIn.addEventListener('input', preview);
    try {
      const res = await F.api.get('/api/ai/providers');
      provs = res.providers.filter(p => p.kind === 'local' && p.capabilities && p.capabilities.detect);
      sel.replaceChildren();
      const usable = provs.filter(p => p.status && p.status.available && (p.models || []).length);
      if (!usable.length) { sel.appendChild(h('option', { value: '' }, 'No local model available')); scopeLine.textContent = 'Add detector weights in Settings, or use the AI tab for a cloud model.'; return; }
      for (const p of usable) sel.appendChild(h('optgroup', { label: p.name || p.id }, p.models.map(mo => h('option', { value: `${p.id}|${mo.id}`, selected: p.id === 'yolo' && mo.default }, `${mo.name || mo.id}${mo.recommended ? ' · recommended' : ''}`))));
      const d = await F.api.post(`${API}/runs/start`, { dataset_id: ds.id, ...choice(), conf: 0.01, dry_run: true, ...scopePayload() }).catch(() => null);
      nImages = d ? d.images : 0;
      const sc = setup.split || 'all splits';
      scopeLine.textContent = nImages ? `Runs on ${fmt.num(nImages)} images · ${sc}${setup.skip_excluded ? ' · excluded images skipped' : ''} (the scope in the form)` : 'No images in this scope.';
      preview();
    } catch (e) { scopeLine.textContent = e.message; }
    okBtn.addEventListener('click', async () => {
      okBtn.disabled = true;
      try {
        const c = choice();
        const r = await F.api.post(`${API}/runs/start`, { dataset_id: ds.id, ...c, conf: Number(confIn.value) || 0.01, class_map: closedVocab(c) ? parseMap(mapIn.value) : null, ...scopePayload() });
        m.close();
        trackRun(side, r.job, r.run_id);
      } catch (e) { scopeLine.textContent = e.message; scopeLine.className = 'small danger'; okBtn.disabled = false; }
    });
  }

  const modelRuns = { a: null, b: null };      // jobs running a model for a source: evaluation waits for them
  function syncRunBtn() {
    const busy = modelRuns.a || modelRuns.b;
    runBtn.disabled = !!busy || !!job;
    runBtn.title = busy ? 'Waiting for the model run to finish' : '';
  }
  function trackRun(side, job, runId) {
    const S = setup[side];
    modelRuns[side] = job.id; syncRunBtn();
    S.type = 'run'; S.run_id = runId;
    const cov = srcViews[side].cov;
    const bar = ui.progress(0); const txt = h('span', { class: 'small' }, 'Loading the model…');
    const cancel = h('button', { class: 'btn btn-sm btn-ghost', onClick: () => F.api.post(`/api/jobs/${job.id}/cancel`).catch(() => {}) }, 'Cancel');
    cov.replaceChildren(h('div', { class: 'row' }, txt, h('span', { class: 'spacer' }), ui.elapsedClock(), cancel), bar);
    checkSeq[side]++;            // a pending coverage check must not overwrite the progress
    const w = F.api.watchJob(job.id, (snap) => { txt.textContent = snap.message || 'Running…'; bar.firstChild.style.width = `${Math.round((snap.progress || 0) * 100)}%`; });
    cleanups.push(() => w.stop && w.stop());
    const settle = () => { modelRuns[side] = null; syncRunBtn(); };
    w.then(async (snap) => {
      settle();
      if (destroyed) return;
      runs = null; await loadRuns();
      srcViews[side].draw();
      recheck(side);
      if (snap.status === 'cancelled') ui.toast(`Model run cancelled — the images it finished are kept as run #${runId}`);
    }).catch(async (e) => {
      settle();
      if (destroyed) return;
      runs = null; await loadRuns();
      S.run_id = (runs || []).some(r => r.id === runId) ? runId : null;
      srcViews[side].draw();
      cov.replaceChildren(h('div', { class: 'line danger' }, icon('alertCircle', 14), h('span', `Model run failed: ${e.message}`)));
    });
  }

  const srcSpec = (side) => {
    const S = setup[side];
    if (S.type === 'run') return S.run_id ? { type: 'run', run_id: S.run_id, remap: S.remap || undefined } : null;
    return S.path ? { type: 'folder', path: S.path, missing: S.missing, remap: S.remap || undefined } : null;
  };
  const scopePayload = () => ({ split: setup.split, skip_excluded: setup.skip_excluded, unlabeled: setup.unlabeled });

  const checkSeq = { a: 0, b: 0 };
  const checkTimers = {};
  function recheck(side) {
    for (const s of side ? [side] : ['a', 'b']) {
      clearTimeout(checkTimers[s]);
      const spec = srcSpec(s);
      const my = ++checkSeq[s];
      if (!spec) { covState[s] = null; srcViews[s].cov.replaceChildren(); continue; }
      checkTimers[s] = setTimeout(async () => {
        const wait = setTimeout(() => { if (my === checkSeq[s]) srcViews[s].cov.replaceChildren(h('div', { class: 'line' }, ui.spinner(), h('span', 'Checking which images it covers…'))); }, 400);
        try {
          const r = await F.api.post(`${API}/check`, { dataset_id: ds.id, source: spec, which: s, ...scopePayload() });
          if (my !== checkSeq[s] || destroyed) return;
          covState[s] = r.coverage; srcViews[s].cov.replaceChildren(...covView(s, spec, r.coverage));
        } catch (e) {
          if (my !== checkSeq[s] || destroyed) return;
          covState[s] = null; srcViews[s].cov.replaceChildren(h('div', { class: 'line danger' }, icon('alertCircle', 14), h('span', e.message)));
        } finally { clearTimeout(wait); }
      }, 350);
    }
  }

  function covView(side, spec, c) {
    const out = [];
    if (!c.images) { out.push(h('div', { class: 'line warn' }, icon('alert', 14), h('span', 'No images in this scope.'))); return out; }
    const ratio = c.matched / c.images;
    const tone = c.matched === 0 ? 'danger' : ratio < 0.5 ? 'warn' : 'ok';
    const ic = tone === 'ok' ? 'checkCircle' : tone === 'warn' ? 'alert' : 'xCircle';
    let text;
    if (spec.type === 'run') {
      text = `${fmt.num(c.matched)} of ${fmt.num(c.images)} images in scope have stored predictions · ${fmt.plural(c.preds, 'box', 'boxes')}${c.has_conf ? '' : ' · no confidences (AP unavailable)'}`;
    } else if (c.matched === 0) {
      text = `No prediction file matches the ${fmt.num(c.images)} images in scope (${fmt.num(c.files)} txt files in the folder).`;
    } else {
      text = `${fmt.num(c.matched)} of ${fmt.num(c.images)} images have a prediction file${c.by_stem ? ` (${fmt.num(c.by_stem)} by file name only)` : ''}${c.sample_rows ? (c.has_conf ? ' · confidence column present' : ' · no confidence column (AP unavailable)') : ''}`;
      if (ratio < 1 && setup[side].missing !== 'skip') text += ' — images without one count as “no detections”.';
    }
    out.push(h('div', { class: `line ${tone}` }, icon(ic, 14), h('span', text)));
    if (c.ambiguous) out.push(h('div', { class: 'line warn' }, icon('alert', 14), h('span', `${fmt.plural(c.ambiguous, 'image')} will be skipped: ${c.ambiguous === 1 ? 'its' : 'their'} prediction file name exists in several splits or folders.`)));
    if (spec.type === 'run') {
      const au = c.audit;
      const line = (tone, ic, text) => out.push(h('div', { class: `line ${tone}` }, icon(ic, 14), h('span', text)));
      const one = (n, a, b) => (n === 1 ? a : b);
      if (au && au.mode === 'geometry' && c.family_text) line('', 'info', `Label audit, geometry mode: the detector finds ${c.family_text}. Each family counts as one class; other classes are left out.`);
      if (au && (au.members || []).length > 1) line('', 'info', `Asked ${au.members.length} models; only the main model’s detections were kept (${au.members[0]}), so that is the model scored.`);
      if (!c.processed_known && c.matched) line('', 'info', 'The run does not record the images it processed: images where it found nothing are not evaluated, so recall may read high.');
      if (c.replaced) line('warn', 'alert', `${fmt.plural(c.replaced, 'image')} lost this run’s proposals to a later proposal run and ${one(c.replaced, 'is', 'are')} left out.`);
      if (c.accepted) line('warn', 'alert', `${fmt.num(c.accepted)} of its predictions ${one(c.accepted, 'was', 'were')} accepted into the labels and ${one(c.accepted, 'is', 'are')} still there — the ground truth contains ${one(c.accepted, 'it', 'them')}.`);
      if (au && au.fixes_applied) line('warn', 'alert', `${fmt.num(au.fixes_applied)} of its findings ${one(au.fixes_applied, 'was', 'were')} applied to the labels — the ground truth now partly comes from this model.`);
      if (c.min_conf != null && c.min_conf >= 0.1) line('', 'info', `Stored from confidence ${Number(c.min_conf).toFixed(2)} up: AP misses what the model found below that.`);
    }
    const splits = Object.entries(c.by_split || {});
    if (splits.length > 1 && ratio < 1) out.push(h('div', { class: 'splits' }, splits.map(([s, v], i) => [i ? ' · ' : null, h('span', `${s || '(no split)'} ${fmt.num(v.matched)}/${fmt.num(v.images)}`)])));
    for (const hint of c.hints || []) {
      const act = hint.kind === 'parent' ? h('button', { class: 'btn btn-sm', onClick: () => { setup[side].path = hint.path_host; srcViews[side].draw(); recheck(side); } }, 'Use parent folder')
        : hint.kind === 'split' && hint.splits.length === 1 ? h('button', { class: 'btn btn-sm', onClick: () => { setup.split = hint.splits[0]; splitSel.value = setup.split; recheck(); } }, `Evaluate ${hint.splits[0]} only`) : null;
      out.push(h('div', { class: 'hint' }, h('span', hint.text), act));
    }
    return out;
  }

  async function discover(side) {
    const baseDir = await F.pickPath({ title: 'Folder to scan for prediction outputs', files: false, selectFiles: false, start: setup[side].path || ds.root_host });
    if (!baseDir) return;
    const m = ui.modal({ title: 'Prediction folders', size: 'lg', body: h('div', { class: 'row', style: 'padding:14px;justify-content:center' }, ui.spinner(), h('span', { class: 'small muted' }, 'Scanning…')) });
    try {
      const { dirs, root_host } = await F.api.get(`${API}/discover`, { path: baseDir, dataset_id: ds.id });
      if (!m.el.isConnected) return;
      if (!dirs.length) { m.body.replaceChildren(h('div', { class: 'empty' }, h('p', 'No folders with YOLO txt files found here.'))); return; }
      const use = (d, s) => { setup[s].type = 'folder'; setup[s].path = d.path_host; srcViews[s].draw(); recheck(s); m.close(); };
      m.body.replaceChildren(
        h('div', { class: 'small muted mb-8' }, 'Inside ', h('span', { class: 'mono' }, root_host), ' — folders with confidences and the best match to this dataset first.'),
        h('table', { class: 'table cmp-disc' }, h('thead', h('tr', h('th', 'Folder'), h('th', { class: 'num' }, 'Files'), h('th', 'Conf'), h('th', { class: 'num', 'data-tip': 'Share of a sample of this dataset’s images that have a file here' }, 'Matches'), h('th', ''))),
          h('tbody', dirs.map(d => h('tr',
            h('td', { class: 'cmp-disc-dir', title: d.path_host }, h('div', { class: 'mono small' }, d.rel ? midEllipsis(d.rel, 60) : '(this folder)'),
              d.kind === 'parent' ? h('div', { class: 'xs muted' }, `All splits in one source: ${d.splits.join(', ')}`) : null),
            h('td', { class: 'num' }, fmt.num(d.files)),
            h('td', d.is_gt ? ui.chip('ground truth', { cls: 'outline' }) : d.has_conf ? ui.chip('yes', { type: 'ok' }) : ui.chip('no', { type: 'warn' })),
            h('td', { class: 'num' }, d.match == null ? '–' : pct(d.match, 0)),
            h('td', { style: 'width:1%' }, h('div', { class: 'row', style: 'justify-content:flex-end;gap:4px;flex-wrap:nowrap' }, h('button', { class: 'btn btn-sm', onClick: () => use(d, 'a') }, 'Use as A'), h('button', { class: 'btn btn-sm', onClick: () => use(d, 'b') }, 'Use as B'))))))));
    } catch (e) { if (m.el.isConnected) m.body.replaceChildren(h('div', { class: 'callout danger' }, icon('alertCircle', 14), h('span', e.message))); }
  }

  function howModal() {
    ui.modal({ title: 'How metrics are computed', size: 'lg', body: h('div', { class: 'col', style: 'gap:10px;font-size:13px;line-height:1.55' },
      h('p', { style: 'margin:0' }, h('b', 'Matching. '), 'Within an image and class, predictions are taken in descending confidence and each claims the unclaimed ground-truth box with the highest IoU ≥ the threshold (COCO). The Ultralytics option lets every prediction propose its best box and each box keep the highest-confidence prediction, exactly like ', h('span', { class: 'mono' }, 'yolo val'), '. Matching runs once over all predictions.'),
      h('p', { style: 'margin:0' }, h('b', 'AP and mAP. '), 'AP is the area under the precision envelope — 101 recall points (COCO), the Ultralytics trapezoid (a perfect class scores 99.5%) or all points (VOC). mAP is the mean over the selected classes that have ground truth; it uses every prediction, so it does not change with the confidence threshold. mAP@.5:.95 averages AP over IoU 0.50, 0.55 … 0.95.'),
      h('p', { style: 'margin:0' }, h('b', 'At a threshold. '), 'Precision, recall, F1, the confusion matrix and the per-image errors keep the predictions with confidence ≥ the threshold. Precision and recall in the tiles pool all boxes of the selected classes (micro); the line below them averages per class (macro). The best threshold maximises pooled F1.'),
      h('p', { style: 'margin:0' }, h('b', 'Confusion matrix. '), 'After same-class matching, a remaining prediction that overlaps a remaining box of another class (IoU ≥ threshold) is a class swap. Other predictions are false positives on background; other boxes are misses.'),
      h('p', { style: 'margin:0' }, h('b', 'Object size. '), 'COCO area ranges in pixels: small < 32², medium < 96², large ≥ 96². Boxes outside a range are ignored for that range.'),
      h('p', { style: 'margin:0' }, h('b', 'Classes counted as one. '), '“Count as one class” (More options) merges classes for the ground truth and every model alike: the group keeps its smallest id. A label audit in geometry mode does this for its class families (a person detector judged on standing + fallen) and leaves out every class outside them, so a car is neither a hit nor a miss.'),
      h('p', { style: 'margin:0' }, h('b', 'Without confidences. '), 'If a prediction has no score, AP is not reported (every prediction ties, so a ranking metric would be arbitrary) and the threshold cannot filter it.')) });
  }

  inner.appendChild(h('div', { class: 'card' },
    h('div', { class: 'card-header' }, h('h3', 'Evaluate predictions against ground truth'), h('span', { class: 'sub' }, 'mAP · precision / recall at any confidence · confusion matrix'), h('span', { class: 'spacer' }),
      h('button', { class: 'btn btn-ghost btn-sm', onClick: howModal }, icon('help', 13), 'How metrics are computed')),
    h('div', { class: 'card-body col', style: 'gap:12px' },
      h('div', { class: 'cmp-srcs' }, sourceEditor('a'), sourceEditor('b')),
      h('div', { class: 'cmp-scope' },
        h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'Scope'), splitSel, h('label', { class: 'check small' }, exclCb, 'Skip excluded images')),
        h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'IoU ≥'), iouIn, iouVal),
        h('div', { class: 'grp' }, h('span', { class: 'lbl' }, 'Metric'), metricSel),
        moreBtn, h('span', { class: 'spacer' }), runBtn),
      h('div', { class: 'small' }, metricHint),
      more, progress)));

  let job = null;
  async function runEval() {
    const a = srcSpec('a'), b = srcSpec('b');
    if (!a) { progress.replaceChildren(h('div', { class: 'callout warn' }, icon('alert', 14), h('span', 'Choose Model A: a prediction folder or an AI run.'))); return; }
    runBtn.disabled = true;
    const started = Date.now();
    try {
      const r = await F.api.post(`${API}/evaluate`, { dataset_id: ds.id, a, b, iou: setup.iou, conf: setup.conf, metric: setup.metric, merge: setup.merge || undefined, ...scopePayload() });
      job = r.job;
      const bar = ui.progress(0); const txt = h('span', { class: 'small muted' }, 'Queued…');
      const cancel = h('button', { class: 'btn btn-sm btn-ghost', onClick: () => F.api.post(`/api/jobs/${job.id}/cancel`).catch(() => {}) }, 'Cancel');
      progress.replaceChildren(h('div', { class: 'row' }, txt, h('span', { class: 'spacer' }), ui.elapsedClock(started), cancel), bar);
      const w = F.api.watchJob(job.id, (snap) => { txt.textContent = snap.message || 'Working…'; bar.firstChild.style.width = `${Math.round((snap.progress || 0) * 100)}%`; });
      cleanups.push(() => w.stop && w.stop());
      const snap = await w;
      if (destroyed) return;
      progress.replaceChildren();
      if (snap.status === 'cancelled') { progress.replaceChildren(h('div', { class: 'callout' }, icon('info', 14), h('span', 'Evaluation cancelled — nothing was saved.'))); return; }
      const res = snap.result || {};
      if (!res.saved) {
        const which = res.which || 'a';
        progress.replaceChildren(h('div', { class: 'callout danger' }, icon('alertCircle', 14), h('div', { class: 'col', style: 'gap:6px' }, h('span', res.reason || 'Nothing to evaluate.'),
          (res.hints || []).map(hint => h('div', { class: 'row wrap' }, h('span', { class: 'small' }, hint.text),
            hint.kind === 'parent' ? h('button', { class: 'btn btn-sm', onClick: () => { setup[which].path = hint.path_host; srcViews[which].draw(); recheck(which); progress.replaceChildren(); } }, 'Use parent folder')
              : hint.kind === 'split' && hint.splits.length === 1 ? h('button', { class: 'btn btn-sm', onClick: () => { setup.split = hint.splits[0]; splitSel.value = setup.split; recheck(); progress.replaceChildren(); } }, `Evaluate ${hint.splits[0]} only`) : null)), h('span', { class: 'small muted' }, 'The evaluation was not saved.'))));
        return;
      }
      await loadSaved();
      await loadEval(res.eval_id, { k: Math.round(setup.conf * 100) });
    } catch (e) { if (!destroyed) progress.replaceChildren(h('div', { class: 'callout danger' }, icon('alertCircle', 14), h('span', e.message))); }
    finally { job = null; syncRunBtn(); }
  }

  function prefill(ev) {
    for (const side of ['a', 'b']) {
      const src = ev.sources && ev.sources[side];
      const S = Object.assign(setup[side], blankSrc());   // in place: the source editors hold this object
      if (src) {
        S.type = src.type === 'run' ? 'run' : 'folder';
        S.path = src.type === 'run' ? '' : (src.path_host || src.path || '');
        S.run_id = src.type === 'run' ? Number(src.run_id) : null;
        S.missing = src.missing || 'empty';
        const rm = (ev.remap || {})[side] || {};
        S.remap = Object.entries(rm).map(([x, y]) => `${x}:${y < 0 ? '-' : y}`).join(', ');
      }
      remapIn[side].value = S.remap || '';
      srcViews[side].draw();
    }
    setup.iou = ev.iou || 0.5; iouIn.value = setup.iou; iouVal.textContent = Number(setup.iou).toFixed(2);
    setup.metric = ev.metric || 'coco'; metricSel.value = setup.metric; metricHint.textContent = (METRIC_OPTS.find(m => m[0] === setup.metric) || METRIC_OPTS[0])[2];
    setup.conf = ev.conf != null ? ev.conf : 0.25; confIn.value = setup.conf;
    const sc = ev.scope || {};
    setup.split = (sc.split || [])[0] || ''; splitSel.value = setup.split;
    setup.skip_excluded = sc.skip_excluded !== false; exclCb.checked = setup.skip_excluded;
    setup.unlabeled = sc.unlabeled || 'background'; unlabeledSel.value = setup.unlabeled;
    missingSel.value = setup.a.missing;
    setup.merge = (ev.merge || []).map(g => g.join('+')).join(', '); mergeIn.value = setup.merge;
    mergeHint.textContent = mergeText(setup.merge) || 'For a detector with coarser classes, e.g. a person model on standing + fallen.';
    syncMore();
    recheck();
  }

  // ------------------------------------------------------------------ saved evaluations
  const savedBody = h('div');
  const savedCount = h('span', { class: 'sub' });
  let savedAll = false;
  let saved = [];
  inner.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Saved evaluations'), savedCount), h('div', { class: 'card-body tight', style: 'overflow-x:auto' }, savedBody)));
  /** 'person (0+1) only' / 'standing + fallen as one': how the evaluation counted classes, when not one by one. */
  const cvText = (cv) => {
    if (!cv) return null;
    const lab = (c) => (cv.labels && cv.labels[c]) || dsName(c);
    if (cv.only) return `${cv.only.map(lab).join(', ')} only`;
    const grp = (cv.merge || []).map(g => lab(g[0]));
    return grp.length ? `${grp.join(', ')} as one` : null;
  };
  const scopeText = (e) => {
    const sc = e.scope || {};
    const parts = [(sc.split && sc.split.length ? sc.split.join(', ') : 'all splits'), `IoU ${Number(e.iou || 0.5).toFixed(2)}`, (e.metric || 'voc').toUpperCase()];
    if (sc.skip_excluded === false) parts.push('incl. excluded');
    if (cvText(e.classes_view)) parts.push(cvText(e.classes_view));
    return parts.join(' · ');
  };
  async function loadSaved() {
    try { saved = (await F.api.get(`${API}/evals`, { dataset_id: ds.id })).evals; } catch (e) { savedBody.replaceChildren(h('div', { class: 'callout danger' }, e.message)); return; }
    drawSaved();
  }
  function drawSaved() {
    savedCount.textContent = saved.length ? `${saved.length}` : '';
    if (!saved.length) { savedBody.replaceChildren(h('div', { class: 'small faint', style: 'padding:6px 4px' }, 'None yet — run an evaluation above.')); return; }
    const list = savedAll ? saved : saved.slice(0, 6);
    const mapTxt = (e, s) => { const sm = e.summary && e.summary[s]; if (!sm) return '–'; const v = sm.map ?? sm.overall?.map50; return v == null ? 'n/a' : pct(v); };
    savedBody.replaceChildren(h('table', { class: 'table cmp-saved' },
      h('thead', h('tr', h('th', 'Evaluation'), h('th', 'Scope'), h('th', { class: 'num' }, 'Images'), h('th', { class: 'num' }, 'A mAP@IoU'), h('th', { class: 'num' }, 'B mAP@IoU'), h('th', 'Created'), h('th', ''))),
      h('tbody', list.map(e => h('tr', { class: cls('clickable', cur && cur.meta.id === e.id && 'active'), tabindex: 0, onClick: () => loadEval(e.id),
          onKeydown: (ev) => { if (ev.key === 'Enter' && ev.target === ev.currentTarget) loadEval(e.id); } },
        h('td', h('div', { class: 'nm', title: e.name }, e.name), e.legacy ? h('span', { class: 'chip outline', 'data-tip': 'Saved by an earlier version: re-run for curves, confusion matrix and flags' }, 'older format') : null),
        h('td', { class: 'scope' }, scopeText(e)),
        h('td', { class: 'num' }, fmt.num(e.images)),
        h('td', { class: 'num' }, mapTxt(e, 'a')), h('td', { class: 'num' }, e.sources && e.sources.b ? mapTxt(e, 'b') : '–'),
        h('td', { class: 'small muted', style: 'white-space:nowrap', title: fmt.date(e.created_at) }, fmt.ago(e.created_at)),
        h('td', h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': 'Evaluation actions', onClick: (ev) => { ev.stopPropagation(); evalMenu(ev.currentTarget, e); } }, icon('moreH', 14))))))),
    saved.length > 6 ? h('div', { class: 'row', style: 'justify-content:center;padding:6px' }, h('button', { class: 'btn btn-ghost btn-sm', onClick: () => { savedAll = !savedAll; drawSaved(); } }, savedAll ? 'Show fewer' : `Show all ${saved.length}`)) : '');   // replaceChildren(null) would print "null"
  }
  function evalMenu(anchor, e) {
    ui.menu(anchor, [
      { label: 'Rename…', icon: 'pen', onClick: () => renameEval(e) },
      { label: 'Re-run with these settings', icon: 'refresh', onClick: async () => { const full = (await F.api.get(`${API}/evals/${e.id}`, { dataset_id: ds.id })).eval; prefill(full); page.scrollTo({ top: 0 }); } },
      { sep: true },
      { label: 'Delete', icon: 'trash', danger: true, onClick: () => deleteEval(e) },
    ]);
  }
  async function renameEval(e) {
    // Blur the trigger: ui.prompt hands focus back to it while Enter is still being handled, which would click it again.
    if (document.activeElement && document.activeElement !== document.body) document.activeElement.blur();
    const name = await ui.prompt({ title: 'Rename evaluation', value: e.name, okLabel: 'Rename' });
    if (!name || !name.trim()) return;
    try { await F.api.patch(`${API}/evals/${e.id}?dataset_id=${encodeURIComponent(ds.id)}`, { name: name.trim() }); await loadSaved(); if (cur && cur.meta.id === e.id) { cur.meta.name = name.trim(); titleEl && (titleEl.textContent = cur.meta.name); } }
    catch (err) { ui.toast(err.message, { type: 'error' }); }
  }
  async function deleteEval(e) {
    if (!(await ui.confirm({ title: 'Delete evaluation?', message: `${e.name} — its saved results go away. Review flags it set stay.`, okLabel: 'Delete', danger: true }))) return;
    try { await F.api.del(`${API}/evals/${e.id}?dataset_id=${encodeURIComponent(ds.id)}`); if (cur && cur.meta.id === e.id) { closeResults(); } await loadSaved(); }
    catch (err) { ui.toast(err.message, { type: 'error' }); }
  }

  // ------------------------------------------------------------------ results
  const results = h('div', { class: 'cmp-section' });
  inner.appendChild(results);
  let cur = null;           // { meta, curves }
  let titleEl = null;
  const view = { k: 25, sel: null, cmSide: 'a', pcTab: 'ap', curve: 'f1', t: { show: 'errors', sort: 'errors', side: 'a', q: '', split: '', cell: '' } };
  let parts = null;          // update hooks of the rendered sections
  let viewer = null;

  function closeResults() { cur = null; parts && parts.destroy(); parts = null; results.replaceChildren(); F.router.replaceQuery({}); drawSaved(); }

  async function loadEval(id, { k } = {}) {
    if (viewer) { viewer.close(); viewer = null; }
    const wait = setTimeout(() => { if (!destroyed) results.replaceChildren(h('div', { class: 'row', style: 'padding:24px;justify-content:center' }, ui.spinner())); }, 400);
    let res;
    try { res = await F.api.get(`${API}/evals/${id}`, { dataset_id: ds.id }); }
    catch (e) { clearTimeout(wait); results.replaceChildren(h('div', { class: 'callout danger' }, icon('alertCircle', 14), h('span', e.message))); return; }
    clearTimeout(wait);
    if (destroyed) return;
    parts && parts.destroy();
    cur = { meta: res.eval, curves: res.curves || null };
    view.k = k != null ? k : Math.round((cur.meta.conf ?? 0.25) * 100);
    view.sel = null; view.cmSide = 'a'; view.t = { show: 'errors', sort: 'errors', side: 'a', q: '', split: '', cell: '' };
    const qcls = ctx.query.get('cls');
    if (qcls && ctx.query.get('eval') === id) view.sel = new Set(qcls.split(',').filter(x => x !== '').map(Number));
    syncUrl();
    drawSaved();
    // Show what produced these numbers in the form (D5) — unless the user is already setting up another run.
    if (!srcSpec('a') && !srcSpec('b')) prefill(cur.meta.legacy ? { ...cur.meta, metric: 'coco' } : cur.meta);
    parts = cur.meta.legacy ? renderLegacy() : renderResults();
  }
  function syncUrl() { if (cur) F.router.replaceQuery({ eval: cur.meta.id, t: (view.k / 100).toFixed(2), cls: view.sel ? [...view.sel].join(',') : null }); }
  const sides = () => ['a', 'b'].filter(s => cur.meta.sources && cur.meta.sources[s] && cur.curves && cur.curves[s]);
  const classesParam = () => (view.sel ? [...view.sel].join(',') : undefined);

  function header(extra) {
    const ev = cur.meta;
    titleEl = h('h2', { title: ev.name }, ev.name);
    const sc = ev.scope || {};
    const metaLine = [`${fmt.num(ev.images)} images`, sc.split && sc.split.length ? sc.split.join(', ') : 'all splits', `IoU ≥ ${Number(ev.iou).toFixed(2)}`,
      ev.metric_label || (ev.metric || 'voc').toUpperCase(), cvText(ev.classes_view), fmt.ago(ev.created_at)].filter(Boolean).join(' · ');
    return h('div', { class: 'col', style: 'gap:2px' },
      h('div', { class: 'cmp-head' }, titleEl,
        h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': 'Rename', 'data-tip': 'Rename', onClick: () => renameEval(ev) }, icon('pen', 13)),
        h('span', { class: 'spacer' }),
        h('button', { class: 'btn btn-sm', onClick: () => { prefill(ev); page.scrollTo({ top: 0 }); }, 'data-tip': 'Load these settings into the form above' }, icon('refresh', 13), 'Re-run'),
        extra),
      h('div', { class: 'cmp-meta', title: fmt.date(ev.created_at) }, metaLine));
  }
  function warningsView(ws) {
    if (!ws || !ws.length) return null;
    return h('div', { class: 'cmp-warn' }, ws.map(w => h('div', { class: `callout small ${w.level === 'danger' ? 'danger' : w.level === 'warn' ? 'warn' : ''}` }, icon(w.level === 'info' ? 'info' : 'alert', 14), h('span', w.text))));
  }
  const sourceLabel = (side) => { const s = cur.meta.sources[side]; return s ? s.label : ''; };
  const sourceTip = (side) => { const s = cur.meta.sources[side]; return s ? (s.path_host || s.label) : ''; };

  function renderResults() {
    const ev = cur.meta, S = sides(), hasB = S.includes('b');
    const destroyers = [];
    const flagLine = h('div', { class: 'small' });
    const exportBtn = h('button', { class: 'btn btn-sm', onClick: (e) => ui.menu(e.currentTarget, [
      { label: 'Per-image CSV (current table)', icon: 'download', onClick: () => download('images') },
      { label: 'Per-class CSV', icon: 'download', onClick: () => download('classes') },
    ]) }, icon('download', 13), 'Export');
    const flagBtn = h('button', { class: 'btn btn-sm', onClick: () => flagDialog() }, icon('flag', 13), 'Flag images…');
    // Classes with ground truth that a model never predicted: usually classes it cannot detect at all (a closed
    // vocabulary, a class map, an audit's model) -- every box of them is a miss that says nothing about the model.
    const unseenBox = h('div');
    const unseen = Object.fromEntries(S.map(s => [s, Object.entries(cur.curves[s].classes).filter(([, d]) => d.gt > 0 && !d.np).map(([c]) => +c)]));
    const drawUnseen = () => {
      const all = [...new Set(S.flatMap(s => unseen[s]))].sort((x, y) => x - y);
      const withGt = new Set(S.flatMap(s => Object.entries(cur.curves[s].classes).filter(([, d]) => d.gt > 0).map(([c]) => +c)));
      const shown = all.filter(c => !view.sel || view.sel.has(c));
      if (!shown.length || all.length >= withGt.size) { unseenBox.replaceChildren(); return; }
      const list = (cs) => (cs.length > 4 ? `${cs.slice(0, 4).map(c => `${c} · ${cname(c)}`).join(', ')} and ${cs.length - 4} more` : cs.map(c => `${c} · ${cname(c)}`).join(', '));
      const who = S.filter(s => unseen[s].length).map(s => `${S.length > 1 ? `Model ${s.toUpperCase()}` : 'The model'} never predicted ${unseen[s].length === 1 ? 'class' : `${unseen[s].length} classes`}: ${list(unseen[s])}`).join('; ');
      unseenBox.replaceChildren(h('div', { class: 'callout small', style: 'margin-top:10px' }, icon('info', 14),
        h('span', { class: 'grow' }, `${who}. If it cannot detect ${all.length === 1 ? 'it' : 'them'}, every box there is a miss that says nothing about the model.`),
        h('button', { class: 'btn btn-sm', onClick: () => { const C = allClasses(); setSel(new Set(C.filter(c => !all.includes(c) && (!view.sel || view.sel.has(c))))); } }, 'Leave them out')));
    };
    put(results, header([exportBtn, flagBtn]), warningsView(ev.warnings), unseenBox, flagLine);

    // ---- threshold bar (scopes everything below it)
    const thrIn = h('input', { type: 'range', class: 'slider', min: 0, max: 1, step: 0.01, value: view.k / 100, 'aria-label': 'Confidence threshold', onInput: (e) => setK(Math.round(Number(e.target.value) * 100)) });
    const thrNum = h('input', { class: 'input input-sm thr-num mono', type: 'number', min: 0, max: 1, step: 0.01, value: (view.k / 100).toFixed(2), 'aria-label': 'Confidence threshold value',
      onChange: (e) => { const v = Number(e.target.value); if (isFinite(v)) setK(Math.round(Math.max(0, Math.min(1, v)) * 100)); } });
    const bestBox = h('span', { class: 'best' });
    const clsBtn = h('button', { class: 'btn btn-sm', onClick: (e) => classPicker(e.currentTarget) }, icon('tags', 13), h('span'));
    const thrBar = h('div', { class: 'cmp-thrbar', role: 'group', 'aria-label': 'Threshold and classes' },
      h('span', { class: 'small strong' }, 'Confidence ≥'), thrIn, thrNum, bestBox, h('span', { class: 'spacer' }), clsBtn);
    results.appendChild(thrBar);

    // ---- headline numbers: one row per model, metrics in aligned columns (A and B read straight down)
    const boardEl = h('div', { class: 'card cmp-board', role: 'table', 'aria-label': 'Headline metrics' });
    results.appendChild(boardEl);

    // ---- charts
    const chartF1 = h('div'), chartPR = h('div');
    const curveSeg = ui.seg([{ value: 'f1', label: 'F1' }, { value: 'p', label: 'Precision' }, { value: 'r', label: 'Recall' }], view.curve, (v) => { view.curve = v; drawCharts(); });
    const chartsCard = h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Threshold curves'), h('span', { class: 'sub' }, 'click a curve to set the threshold')),
      h('div', { class: 'card-body cmp-charts' },
        h('div', h('div', { class: 'cmp-chart-head' }, h('h4', 'By confidence threshold'), h('span', { class: 'spacer' }), curveSeg), chartF1),
        h('div', h('div', { class: 'cmp-chart-head' }, h('h4', 'Precision–recall as the threshold moves')), chartPR)));
    const noConf = S.filter(s => cur.curves[s].conf_available === false);
    results.appendChild(chartsCard);
    let cF1 = null, cPR = null;
    function drawCharts() {
      cF1 && cF1.destroy(); cPR && cPR.destroy();
      const color = (s) => (s === 'a' ? 'var(--cmp-a)' : 'var(--cmp-b)');     // live CSS variables follow the theme
      const ser = S.filter(s => cur.curves[s].conf_available !== false).map(s => ({ s, label: `Model ${s.toUpperCase()}`, color: color(s), cv: curveOf(cur.curves[s], view.sel) }));
      if (!ser.length) { chartF1.replaceChildren(h('div', { class: 'callout small' }, icon('info', 14), h('span', 'No confidences: threshold curves need scored predictions.'))); chartPR.replaceChildren(); return; }
      const key = view.curve === 'p' ? 'p' : view.curve === 'r' ? 'r' : 'f1';
      const marks = view.curve === 'f1' ? ser.map((x, i) => { const b = bestK(cur.curves[x.s], view.sel); return b ? { series: i, k: b.k } : null; }).filter(Boolean) : [];
      cF1 = confChart(chartF1, { series: ser.map(x => ({ label: x.label, color: x.color, values: x.cv[key] })), k: view.k, marks, yLabel: key === 'f1' ? 'F1' : key === 'p' ? 'precision' : 'recall', onPick: setK });
      cPR = prChart(chartPR, { series: ser.map(x => ({ label: x.label, color: x.color, p: x.cv.p, r: x.cv.r, tpfp: x.cv.n })), k: view.k, onPick: setK });
    }
    destroyers.push(() => { cF1 && cF1.destroy(); cPR && cPR.destroy(); });
    if (noConf.length) chartsCard.querySelector('.card-body').prepend(h('div', { class: 'callout small', style: 'grid-column:1/-1' }, icon('info', 14), h('span', `Model ${noConf.map(s => s.toUpperCase()).join(' and ')} has predictions without confidence: no curve, AP or threshold for it.`)));

    // ---- confusion matrix
    const cmBody = h('div', { class: 'cmp-cm-wrap' });
    const cmNote = h('span', { class: 'sub' });
    const cmSeg = hasB ? ui.seg([{ value: 'a', label: 'Model A' }, { value: 'b', label: 'Model B' }], view.cmSide, (v) => { view.cmSide = v; loadCm(); }) : null;
    results.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Confusion matrix'), cmNote, h('span', { class: 'spacer' }), cmSeg),
      h('div', { class: 'card-body' }, cmBody, h('div', { class: 'hint', style: 'margin-top:8px' }, 'Rows: ground truth. Columns: what the model predicted there. Shading is the share of each row. Click a cell to list its images below.'))));
    let cmSeq = 0;
    async function loadCm() {
      const my = ++cmSeq;
      cmNote.textContent = `at confidence ≥ ${(view.k / 100).toFixed(2)}`;
      try {
        const data = await F.api.get(`${API}/evals/${ev.id}/confusion`, { dataset_id: ds.id, side: view.cmSide, conf: view.k / 100, classes: classesParam() });
        if (my !== cmSeq || destroyed) return;
        if (!data.labels.length) { cmBody.replaceChildren(h('div', { class: 'small muted' }, 'No classes selected.')); return; }
        confusionTable(cmBody, { data, name: cname, textOn: F.colors.textOn || (() => '#000'), active: view.t.cell && view.t.side === view.cmSide ? view.t.cell : null,
          onCell: (g, p) => { view.t = { ...view.t, show: 'cell', cell: `${g}:${p}`, side: view.cmSide, sort: 'errors' }; drawToolbar(); loadRows(); loadCm(); imgsCard.scrollIntoView({ block: 'start' }); } });
      } catch (e) { if (my === cmSeq) cmBody.replaceChildren(h('div', { class: 'callout danger' }, e.message)); }
    }

    // ---- per class
    const pcBody = h('div', { style: 'overflow-x:auto' });
    const pcSeg = ui.seg([{ value: 'ap', label: 'AP' }, { value: 'op', label: 'At threshold' }], view.pcTab, (v) => { view.pcTab = v; drawPerClass(); });
    results.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Per class'), h('span', { class: 'sub' }, 'untick a class to leave it out of every number on this page'), h('span', { class: 'spacer' }), pcSeg),
      h('div', { class: 'card-body tight' }, pcBody)));
    const allClasses = () => [...new Set(S.flatMap(s => Object.keys(cur.curves[s].classes).map(Number)))].sort((x, y) => x - y)
      .filter(c => S.some(s => { const d = cur.curves[s].classes[c]; return d && (d.gt > 0 || d.np > 0); }));
    function drawPerClass() {
      const C = allClasses();
      const on = (c) => !view.sel || view.sel.has(c);
      const head = h('input', { type: 'checkbox', 'aria-label': 'All classes', checked: !view.sel, indeterminate: !!view.sel && view.sel.size > 0 && view.sel.size < C.length, onChange: (e) => setSel(e.target.checked ? null : new Set()) });
      const grp = (s, n) => h('th', { class: `num grp-${s}`, colspan: n }, `Model ${s.toUpperCase()}`);
      const rows = C.map(c => {
        const cells = [];
        const d = Object.fromEntries(S.map(s => [s, cur.curves[s].classes[c]]));
        const gt = Math.max(...S.map(s => (d[s] ? d[s].gt : 0)));
        const oneCalc = (s) => (d[s] ? calc({ classes: { [c]: d[s] }, conf_available: cur.curves[s].conf_available }, ev, view.k, null) : null);
        if (view.pcTab === 'ap') {
          for (const s of S) {
            const o = oneCalc(s);
            cells.push(h('td', { class: 'num' }, d[s] ? fmt.num(d[s].np) : '–'));
            cells.push(h('td', { class: cls('num', (!o || o.map == null) && 'cmp-na'), title: gt ? null : 'No ground truth for this class: no AP' }, o && o.map != null ? pct(o.map) : '–'));
            cells.push(h('td', { class: cls('num', (!o || o.map95 == null) && 'cmp-na') }, o && o.map95 != null ? pct(o.map95) : '–'));
          }
          if (hasB) { const oa = oneCalc('a'), ob = oneCalc('b'); cells.push(deltaCell(oa && oa.map, ob && ob.map)); }
        } else {
          for (const s of S) {
            const o = oneCalc(s);
            if (!o) { cells.push(...[0, 1, 2, 3, 4, 5].map(() => h('td', { class: 'num cmp-na' }, '–'))); continue; }
            const b = bestK({ classes: { [c]: d[s] }, conf_available: cur.curves[s].conf_available }, null);
            cells.push(h('td', { class: 'num' }, o.tp + o.fp ? pct(o.p) : '–'), h('td', { class: 'num' }, gt ? pct(o.r) : '–'), h('td', { class: 'num' }, gt ? pct(o.f1) : '–'),
              h('td', { class: 'num' }, fmt.num(o.fp)), h('td', { class: 'num' }, fmt.num(o.fn)),
              h('td', { class: 'num' }, b && gt ? h('button', { class: 'best-btn', 'data-tip': `Best F1 for this class: ${pct(b.f1)} — set the page threshold to it`, onClick: () => setK(b.k) }, (b.k / 100).toFixed(2)) : '–'));
          }
        }
        return h('tr', { class: on(c) ? '' : 'off' },
          h('td', h('input', { type: 'checkbox', checked: on(c), 'aria-label': `Include ${cname(c)}`, onChange: (e) => toggleClass(c, e.target.checked, C) })),
          h('td', { class: 'cls', title: cname(c) }, h('span', { class: 'swatch', style: `background:${F.colors.classColor(c)};margin-right:6px` }), `${c} · ${cname(c)}`),
          h('td', { class: 'num' }, fmt.num(gt)), ...cells);
      });
      const sub = view.pcTab === 'ap'
        ? S.flatMap(s => [h('th', { class: 'num' }, 'Preds'), h('th', { class: 'num' }, `AP@${Number(ev.iou).toFixed(2)}`), h('th', { class: 'num' }, 'AP@.5:.95')]).concat(hasB ? [h('th', { class: 'num' }, 'Δ AP')] : [])
        : S.flatMap(s => [h('th', { class: 'num' }, 'P'), h('th', { class: 'num' }, 'R'), h('th', { class: 'num' }, 'F1'), h('th', { class: 'num' }, 'FP'), h('th', { class: 'num' }, 'FN'), h('th', { class: 'num', 'data-tip': 'Confidence threshold with the best F1 for this class' }, 'Best conf')]);
      pcBody.replaceChildren(C.length ? h('table', { class: 'table cmp-pc' },
        h('thead', h('tr', h('th', { rowspan: 2, style: 'width:28px' }, head), h('th', { rowspan: 2 }, 'Class'), h('th', { rowspan: 2, class: 'num' }, 'GT'), S.map(s => grp(s, view.pcTab === 'ap' ? 3 : 6)), hasB && view.pcTab === 'ap' ? h('th', { rowspan: 1 }) : null),
          h('tr', sub)),
        h('tbody', rows)) : h('div', { class: 'small muted', style: 'padding:8px' }, 'No classes.'));
    }
    const deltaCell = (a, b) => { if (a == null || b == null) return h('td', { class: 'num cmp-na' }, '–'); const d = (b - a) * 100; return h('td', { class: 'num' }, h('span', { class: `cmp-delta ${d > 0.05 ? 'up' : d < -0.05 ? 'down' : 'flat'}`, style: 'margin:0' }, `${d > 0 ? '+' : ''}${d.toFixed(1)}`)); };
    function toggleClass(c, on, C) { const s = new Set(view.sel || C); if (on) s.add(c); else s.delete(c); setSel(s.size === C.length ? null : s); }

    // ---- object size
    const sizeBody = h('div', { style: 'overflow-x:auto' });
    results.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'By object size'), h('span', { class: 'sub' }, 'COCO area ranges in pixels')), h('div', { class: 'card-body tight' }, sizeBody)));
    function drawSizes() {
      const label = { S: 'Small  < 32²', M: 'Medium  32²–96²', L: 'Large  ≥ 96²' };
      const rows = [0, 1, 2].map(b => {
        const cells = [];
        let gtAll = 0;
        for (const s of S) {
          const sz = cur.curves[s].sizes[b];
          let T = 0, Fp = 0, G = 0; const aps = [];
          for (const [c, d] of Object.entries(sz.classes)) { if (view.sel && !view.sel.has(+c)) continue; T += d.tp[view.k]; Fp += d.fp[view.k]; G += d.gt; if (d.gt > 0 && d.ap != null) aps.push(d.ap); }
          gtAll = Math.max(gtAll, G);
          const ok = cur.curves[s].conf_available !== false;
          cells.push(h('td', { class: 'num' }, ok && aps.length ? pct(mean(aps)) : '–'), h('td', { class: 'num' }, G ? pct(T / G) : '–'), h('td', { class: 'num' }, T + Fp ? pct(T / (T + Fp)) : '–'));
        }
        return h('tr', h('td', { style: 'white-space:nowrap' }, label[cur.curves[S[0]].sizes[b].name] || cur.curves[S[0]].sizes[b].name), h('td', { class: 'num' }, fmt.num(gtAll)), ...cells);
      });
      sizeBody.replaceChildren(h('table', { class: 'table cmp-pc' },
        h('thead', h('tr', h('th', { rowspan: 2 }, 'Size'), h('th', { rowspan: 2, class: 'num' }, 'GT'), S.map(s => h('th', { class: `num grp-${s}`, colspan: 3 }, `Model ${s.toUpperCase()}`))),
          h('tr', S.flatMap(() => [h('th', { class: 'num' }, `AP@${Number(ev.iou).toFixed(2)}`), h('th', { class: 'num' }, 'Recall'), h('th', { class: 'num' }, 'Precision')]))),
        h('tbody', rows)));
    }

    // ---- per image
    const toolbar = h('div', { class: 'cmp-toolbar' });
    const countEl = h('span', { class: 'result-count' });
    const tableBody = h('tbody');
    const moreBox = h('div', { class: 'load-more', style: 'padding:10px' });
    const imgsTable = h('table', { class: 'table cmp-imgs' });
    const flagShownBtn = h('button', { class: 'btn btn-sm', onClick: () => flagShown(), 'data-tip': 'Flag every image this table lists, with a note from this evaluation' }, icon('flag', 12), 'Flag these');
    const imgsCard = h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Per image'), countEl, h('span', { class: 'spacer' }),
      flagShownBtn, h('button', { class: 'btn btn-sm', onClick: () => download('images') }, icon('download', 12), 'CSV')),
      h('div', { class: 'card-body', style: 'padding-bottom:4px' }, toolbar), h('div', { class: 'card-body tight', style: 'overflow:auto;max-height:70vh' }, imgsTable, moreBox));
    results.appendChild(imgsCard);
    const thead = h('thead', h('tr', h('th', 'Image'), h('th', 'Split'), h('th', { class: 'num' }, 'GT'),
      S.map(s => h('th', { class: `num grp-${s}`, 'data-tip': 'true positives · false positives · misses · class swaps' }, `${s.toUpperCase()} · TP / FP / FN`)), hasB ? h('th', { class: 'num', 'data-tip': 'Errors of B minus errors of A' }, 'Δ') : null, h('th', 'Review')));
    imgsTable.append(thead, tableBody);
    let rowsState = { items: [], total: 0, params: null };
    let rowsSeq = 0;
    function drawToolbar() {
      const t = view.t;
      const showSel = h('select', { class: 'select select-sm', style: 'width:auto', 'aria-label': 'Which images', onChange: (e) => { view.t = { ...view.t, show: e.target.value, cell: e.target.value === 'cell' ? view.t.cell : '' }; drawToolbar(); loadRows(); if (e.target.value !== 'cell') loadCm(); } },
        SHOW_OPTS.filter(([v]) => hasB || !['regress', 'improve'].includes(v)).map(([v, l]) => h('option', { value: v, selected: t.show === v }, l)), t.show === 'cell' ? h('option', { value: 'cell', selected: true }, 'Confusion cell') : null);
      const sideSeg = hasB && ['fn', 'fp', 'swap', 'cell'].includes(t.show) || (hasB && ['fn', 'fp', 'swap'].includes(t.sort)) ? ui.seg([{ value: 'a', label: 'A' }, { value: 'b', label: 'B' }], t.side, (v) => { view.t = { ...view.t, side: v }; loadRows(); }, { size: 'sm' }) : null;
      const sortSel = h('select', { class: 'select select-sm', style: 'width:auto', 'aria-label': 'Sort', onChange: (e) => { view.t = { ...view.t, sort: e.target.value }; drawToolbar(); loadRows(); } },
        SORT_OPTS.filter(([v]) => hasB || v !== 'delta').map(([v, l]) => h('option', { value: v, selected: t.sort === v }, l)));
      const evSplits = ev.scope && ev.scope.split && ev.scope.split.length ? ev.scope.split : (ds.splits || []).filter(Boolean);
      const splitF = evSplits.length > 1 ? h('select', { class: 'select select-sm', style: 'width:auto', 'aria-label': 'Split', onChange: (e) => { view.t = { ...view.t, split: e.target.value }; loadRows(); } },
        [['', 'All splits'], ...evSplits.map(s => [s, s])].map(([v, l]) => h('option', { value: v, selected: t.split === v }, l))) : null;
      const q = h('div', { class: 'input-wrap' }, icon('search', 14), h('input', { class: 'input input-sm', placeholder: 'Filter by path', value: t.q, 'aria-label': 'Filter by path', onInput: F.debounce((e) => { view.t = { ...view.t, q: e.target.value.trim() }; loadRows(); }, 250) }));
      let cellChip = null;
      if (t.show === 'cell' && t.cell) {
        const [g, p] = t.cell.split(':');
        const txt = g === 'bg' ? `${cname(+p)} on background` : p === 'miss' ? `${cname(+g)} missed` : g === p ? `${cname(+g)} found` : `GT ${cname(+g)} → predicted ${cname(+p)}`;
        cellChip = ui.chip(`${hasB ? `Model ${t.side.toUpperCase()}: ` : ''}${txt}`, { cls: 'accent cmp-filter-chip', onRemove: () => { view.t = { ...view.t, show: 'errors', cell: '' }; drawToolbar(); loadRows(); loadCm(); } });
      }
      put(toolbar, h('span', { class: 'small muted' }, 'Show'), showSel, sideSeg, h('span', { class: 'small muted' }, 'Sort'), sortSel, splitF, q, cellChip);
    }
    const n = (v, kind) => h('span', { class: `cmp-n ${v ? kind : 'zero'}` }, String(v));
    function rowEl(r, idx) {
      const rv = r.review;
      const errs = (s) => (r[s] ? r[s].fp + r[s].fn : 0);
      return h('tr', { class: 'clickable', tabindex: 0, onClick: () => openAt(idx), onKeydown: (e) => { if (e.key === 'Enter' && !e.repeat) { e.preventDefault(); openAt(idx); } } },
        h('td', { class: 'path', title: r.rel_path }, midEllipsis(r.rel_path, 70)),
        h('td', { class: 'small' }, r.split || '—'), h('td', { class: 'num' }, fmt.num(r.gt)),
        S.map(s => h('td', { class: 'num errs' }, r[s] ? [n(r[s].tp, 'tp'), ' ', n(r[s].fp, 'fp'), ' ', n(r[s].fn, 'fn'), r[s].sw ? [' ', h('span', { class: 'cmp-n sw', title: 'class swaps' }, `⇄${r[s].sw}`)] : null] : '–')),
        hasB ? h('td', { class: 'num' }, (() => { const d = errs('b') - errs('a'); return h('span', { class: `cmp-delta ${d < 0 ? 'up' : d > 0 ? 'down' : 'flat'}`, style: 'margin:0' }, d > 0 ? `+${d}` : String(d)); })()) : null,
        h('td', rv && rv.status ? h('span', { class: `chip ${rv.status === 'flagged' ? 'warn' : rv.status === 'approved' ? 'ok' : rv.status === 'excluded' ? 'danger' : ''}`, title: rv.note || '' }, rv.status) : null));
    }
    function rowsParams() { return { dataset_id: ds.id, conf: view.k / 100, classes: classesParam(), show: view.t.show, sort: view.t.sort, side: view.t.side, q: view.t.q, split: view.t.split, cell: view.t.cell }; }
    async function loadRows() {
      const my = ++rowsSeq;
      const params = rowsParams();
      const wait = setTimeout(() => { if (my === rowsSeq) countEl.textContent = 'Loading…'; }, 400);
      try {
        const r = await F.api.get(`${API}/evals/${ev.id}/rows`, { ...params, offset: 0, limit: 100 });
        if (my !== rowsSeq || destroyed) return;
        rowsState = { items: r.items, total: r.total, params };
        tableBody.replaceChildren(...r.items.map((x, i) => rowEl(x, i)));
        syncCount();
      } catch (e) { if (my === rowsSeq) { tableBody.replaceChildren(); countEl.textContent = ''; moreBox.replaceChildren(h('div', { class: 'callout danger' }, e.message)); } }
      finally { clearTimeout(wait); }
    }
    function syncCount() {
      countEl.textContent = `${fmt.num(rowsState.total)} image${rowsState.total === 1 ? '' : 's'}${rowsState.items.length < rowsState.total ? ` · showing ${fmt.num(rowsState.items.length)}` : ''}`;
      flagShownBtn.disabled = !rowsState.total || ['all', 'clean'].includes(view.t.show);
      flagShownBtn.lastChild.textContent = rowsState.total ? `Flag these ${fmt.num(rowsState.total)}` : 'Flag these';
      moreBox.replaceChildren();
      if (!rowsState.total) moreBox.appendChild(h('div', { class: 'small muted' }, view.t.show === 'errors' ? 'No errors at this threshold.' : 'No images match.'));
      else if (rowsState.items.length < rowsState.total) moreBox.appendChild(h('button', { class: 'btn btn-sm', onClick: (e) => { e.currentTarget.disabled = true; loadMore().catch(err => ui.toast(err.message, { type: 'error' })); } }, `Load ${fmt.num(Math.min(100, rowsState.total - rowsState.items.length))} more`));
    }
    async function loadMore() {
      const my = rowsSeq; const st = rowsState;
      const r = await F.api.get(`${API}/evals/${ev.id}/rows`, { ...st.params, offset: st.items.length, limit: 100 });
      if (my !== rowsSeq || st !== rowsState) return;
      const startIdx = st.items.length;
      st.items.push(...r.items); st.total = r.total;
      tableBody.append(...r.items.map((x, i) => rowEl(x, startIdx + i)));     // append only the new rows
      syncCount();
    }
    function openAt(idx) {
      const st = rowsState;
      // The viewer walks a frozen copy of this list (same order even if the threshold changes inside it).
      const arr = st.items.slice(); let total = st.total; let pending = null;
      const list = {
        total: () => total, loaded: () => arr.length, peek: (j) => arr[j],
        async get(j) {
          while (j >= arr.length && arr.length < total) {
            pending = pending || F.api.get(`${API}/evals/${ev.id}/rows`, { ...st.params, offset: arr.length, limit: 200 }).then(r => { arr.push(...r.items); total = r.total; }).finally(() => { pending = null; });
            await pending;
          }
          return arr[j] || null;
        },
      };
      viewer = openViewer({ F, API, ds, names: evNames(), ev, list, startIndex: idx, k: view.k, classes: classesParam(), onK: (k) => setK(k, { fromViewer: true }), onFlagChange: () => { flagsDirty = true; } });
    }
    let flagsDirty = false;
    const reviewPoll = () => { if (flagsDirty && (!viewer || !viewer.open)) { flagsDirty = false; loadRows(); refreshFlagLine(); } };
    const pollT = setInterval(reviewPoll, 600);
    destroyers.push(() => clearInterval(pollT));

    // ---- flagging
    async function refreshFlagLine() {
      try {
        const st = await F.api.get(`${API}/evals/${ev.id}/flags`, { dataset_id: ds.id });
        if (destroyed || !cur || cur.meta.id !== ev.id) return;
        if (!st.count) { flagLine.replaceChildren(); return; }
        const last = st.ops && st.ops[st.ops.length - 1];
        flagLine.replaceChildren(h('div', { class: 'callout small', style: 'margin-top:10px' }, icon('flag', 14), h('span', `${fmt.num(st.count)} image${st.count === 1 ? '' : 's'} flagged by this evaluation.`),
          h('a', { href: `/d/${ds.id}/explore?review=flagged` }, 'Review flagged images in Explore'),
          last ? h('button', { class: 'btn btn-sm', onClick: () => unflag({ op: last.op }) }, 'Undo last') : null,
          h('button', { class: 'btn btn-sm btn-ghost', onClick: () => unflag({ all: true }) }, 'Remove all flags from this evaluation')));
      } catch (e) { /* the line is optional */ }
    }
    async function unflag(body) {
      try {
        const r = await F.api.post(`${API}/evals/${ev.id}/unflag`, { dataset_id: ds.id, ...body });
        ui.toast(`Restored ${fmt.num(r.restored)} image${r.restored === 1 ? '' : 's'}${r.changed ? ` · ${fmt.num(r.changed)} changed since and left as is` : ''}`, { type: 'ok' });
        refreshFlagLine(); loadRows();
      } catch (e) { ui.toast(e.message, { type: 'error' }); }
    }
    function flagDialog() {
      const crit = { side: hasB ? 'any' : 'a', fn: true, fp: true, sw: false, min: 1 };
      let includeApproved = false;
      const preview = h('div', { class: 'small', 'aria-live': 'polite', style: 'min-height:40px' });
      let pSeq = 0, will = 0, counted = '';
      const okBtn = h('button', { class: 'btn btn-primary', disabled: true }, icon('flag', 13), 'Flag images');
      const critKey = () => JSON.stringify([crit, includeApproved]);
      const count = F.debounce(async () => {
        const my = ++pSeq, key = critKey();
        try {
          const r = await F.api.post(`${API}/evals/${ev.id}/flag`, { dataset_id: ds.id, conf: view.k / 100, classes: classesParam() || null, criteria: crit, include_approved: includeApproved, dry_run: true });
          if (my !== pSeq) return;
          will = r.will_flag + (r.will_update || 0); counted = key;
          const st = r.statuses || {};
          preview.replaceChildren(h('div', h('b', fmt.num(r.images)), ' images match.'),
            h('div', { class: 'muted' }, [
              `${fmt.num(r.will_flag)} will be flagged`,
              st.flagged ? `${fmt.num(st.flagged)} already flagged get a note line` : null,
              st.approved && !includeApproved ? `${fmt.num(st.approved)} approved left alone` : null,
              st.excluded ? `${fmt.num(st.excluded)} excluded left alone` : null,
            ].filter(Boolean).join(' · ')));
          okBtn.disabled = !will; okBtn.lastChild.textContent = will ? `Flag ${fmt.num(will)} image${will === 1 ? '' : 's'}` : 'Flag images';
        } catch (e) { if (my === pSeq) preview.replaceChildren(h('span', { class: 'danger' }, e.message)); }
      }, 200);
      // Any change makes the count on the button stale at once (the recount waits for typing to pause): no click may
      // flag a different number of images than the button says.
      const refresh = () => { ++pSeq; counted = ''; okBtn.disabled = true; okBtn.lastChild.textContent = 'Flag images'; count(); };
      const cb = (key, label) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: crit[key], onChange: (e) => { crit[key] = e.target.checked; refresh(); } }), label);
      const m = ui.modal({
        title: 'Flag images for review', size: 'lg',
        body: h('div', { class: 'col', style: 'gap:12px' },
          h('div', { class: 'small muted' }, `Sets the review status to Flagged and adds a note line naming this evaluation, at confidence ≥ ${(view.k / 100).toFixed(2)}${view.sel ? ` and ${view.sel.size} selected class${view.sel.size === 1 ? '' : 'es'}` : ''}. Excluded images are never changed.`),
          hasB ? h('div', { class: 'row' }, h('span', { class: 'small', style: 'width:150px' }, 'Where'), ui.seg([{ value: 'any', label: 'Either model' }, { value: 'a', label: 'Model A' }, { value: 'b', label: 'Model B' }], crit.side, (v) => { crit.side = v; refresh(); })) : null,
          h('div', { class: 'row wrap' }, h('span', { class: 'small', style: 'width:150px' }, 'Has'), cb('fn', 'misses (FN)'), cb('fp', 'false positives (FP)'), cb('sw', 'class swaps')),
          h('div', { class: 'row' }, h('span', { class: 'small', style: 'width:150px' }, 'At least'), h('input', { class: 'input input-sm', type: 'number', min: 1, step: 1, value: 1, style: 'width:80px', 'aria-label': 'Minimum per image',
            onInput: (e) => { crit.min = Math.max(1, Math.floor(Number(e.target.value)) || 1); refresh(); }, onChange: (e) => { e.target.value = crit.min; } }), h('span', { class: 'small muted' }, 'of those per image')),
          h('label', { class: 'check' }, h('input', { type: 'checkbox', onChange: (e) => { includeApproved = e.target.checked; refresh(); } }), 'Also flag images already approved'),
          preview),
        footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'), okBtn],
      });
      okBtn.addEventListener('click', async () => {
        if (counted !== critKey()) { refresh(); return; }
        okBtn.disabled = true;
        try {
          const r = await F.api.post(`${API}/evals/${ev.id}/flag`, { dataset_id: ds.id, conf: view.k / 100, classes: classesParam() || null, criteria: crit, include_approved: includeApproved });
          m.close();
          ui.toast(`Flagged ${fmt.num(r.flagged)} image${r.flagged === 1 ? '' : 's'}${r.updated ? ` · ${fmt.plural(r.updated, 'note')} updated` : ''}`, { type: 'ok', action: { label: 'Undo', onClick: () => unflag({ op: r.op }) } });
          refreshFlagLine(); loadRows();
        } catch (e) { preview.replaceChildren(h('span', { class: 'danger' }, e.message)); okBtn.disabled = !will; }
      });
      refresh();
    }

    async function flagShown() {
      const table = { show: view.t.show, side: view.t.side, q: view.t.q, split: view.t.split, cell: view.t.cell };
      const body = { dataset_id: ds.id, conf: view.k / 100, classes: classesParam() || null, table };
      try {
        const dry = await F.api.post(`${API}/evals/${ev.id}/flag`, { ...body, dry_run: true });
        dry.will_flag += dry.will_update || 0;
        if (!dry.will_flag) { ui.toast(dry.images ? 'Nothing to flag: these images are approved or excluded' : 'No images in this list'); return; }
        const st = dry.statuses || {};
        const skipped = [st.approved ? `${fmt.num(st.approved)} approved` : null, st.excluded ? `${fmt.num(st.excluded)} excluded` : null].filter(Boolean).join(' and ');
        const ok = await ui.confirm({ title: `Flag ${fmt.num(dry.will_flag)} image${dry.will_flag === 1 ? '' : 's'}?`, okLabel: 'Flag',
          message: `Every image in this list gets the review status Flagged and a note line naming this evaluation (confidence ≥ ${(view.k / 100).toFixed(2)}).${skipped ? ` ${skipped} stay as they are.` : ''} You can undo it.` });
        if (!ok) return;
        const r = await F.api.post(`${API}/evals/${ev.id}/flag`, body);
        ui.toast(`Flagged ${fmt.num(r.flagged)} image${r.flagged === 1 ? '' : 's'}${r.updated ? ` · ${fmt.plural(r.updated, 'note')} updated` : ''}`, { type: 'ok', action: { label: 'Undo', onClick: () => unflag({ op: r.op }) } });
        refreshFlagLine(); loadRows();
      } catch (e) { ui.toast(e.message, { type: 'error' }); }
    }
    function download(what) {
      const q = new URLSearchParams({ dataset_id: ds.id, what, conf: (view.k / 100).toFixed(2), show: view.t.show, sort: view.t.sort, side: view.t.side, q: view.t.q, split: view.t.split, cell: view.t.cell });
      if (view.sel) q.set('classes', [...view.sel].join(','));
      const a = document.createElement('a'); a.href = `${API}/evals/${ev.id}/export?${q}`; a.download = ''; document.body.appendChild(a); a.click(); a.remove();
    }

    function classPicker(anchor) {
      const C = allClasses();
      const pop = ui.popover(anchor, h('div', { class: 'col', style: 'gap:2px;max-height:360px;overflow:auto' },
        h('div', { class: 'row', style: 'padding:2px 4px 6px' }, h('button', { class: 'btn btn-sm', onClick: () => { setSel(null); pop.close(); } }, 'All'),
          h('button', { class: 'btn btn-sm', onClick: () => { setSel(new Set(C.filter(c => S.some(s => (cur.curves[s].classes[c] || {}).gt > 0)))); pop.close(); } }, 'With ground truth')),
        C.map(c => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: !view.sel || view.sel.has(c), onChange: (e) => toggleClass(c, e.target.checked, C) }),
          h('span', { class: 'swatch', style: `background:${F.colors.classColor(c)}` }), `${c} · ${cname(c)}`))), { width: 280 });
    }

    // ---- updates
    function drawTiles() {
      const best = S.map(s => ({ s, b: bestK(cur.curves[s], view.sel) })).filter(x => x.b);
      put(bestBox, best.length ? h('span', 'Best F1:') : null, best.map(({ s, b }) => h('span', { class: 'best' }, h('span', { class: `cmp-key ${s}` }), `${S.length > 1 ? s.toUpperCase() + ' ' : ''}`,
        h('span', { class: 'k' }, (b.k / 100).toFixed(2)), h('span', { class: 'muted' }, `(${pct(b.f1)})`),
        h('button', { class: 'btn btn-sm', disabled: b.k === view.k, onClick: () => setK(b.k) }, 'Use'))));
      const C = allClasses();
      clsBtn.lastChild.textContent = view.sel ? `${view.sel.size} of ${C.length} class${C.length === 1 ? '' : 'es'}` : C.length === 1 ? '1 class' : `All ${C.length} classes`;
      const stats = Object.fromEntries(S.map(s => [s, calc(cur.curves[s], ev, view.k, view.sel)]));
      const cols = [
        { key: 'map', label: `mAP@${Number(ev.iou).toFixed(2)}`, tip: 'Mean AP over the selected classes with ground truth, from every prediction — independent of the threshold', sub: (o) => `${o.nAp} class${o.nAp === 1 ? '' : 'es'} · all scores` },
        { key: 'map95', label: 'mAP@.5:.95', tip: 'AP averaged over IoU 0.50, 0.55 … 0.95, then over classes', sub: () => 'IoU 0.50 … 0.95' },
        { key: 'p', label: 'Precision', tip: 'All boxes of the selected classes at this threshold (micro)', sub: (o) => `${fmt.num(o.tp)} TP · ${fmt.num(o.fp)} FP` },
        { key: 'r', label: 'Recall', tip: 'All ground-truth boxes of the selected classes at this threshold (micro)', sub: (o) => `${fmt.num(o.fn)} missed of ${fmt.num(o.gt)}` },
        { key: 'f1', label: 'F1', tip: 'Harmonic mean of precision and recall at this threshold', sub: () => `at conf ≥ ${(view.k / 100).toFixed(2)}` },
      ];
      const cellsOf = (s) => {
        const o = stats[s], ref = s === 'b' ? stats.a : null;
        const noConf = cur.curves[s].conf_available === false;
        return cols.map(c => {
          const v = (c.key === 'map' || c.key === 'map95') && noConf ? null : o[c.key];
          let delta = null;
          if (ref && v != null && ref[c.key] != null) { const d = (v - ref[c.key]) * 100; delta = h('span', { class: `cmp-delta ${d > 0.05 ? 'up' : d < -0.05 ? 'down' : 'flat'}`, title: 'Model B minus model A, in percentage points' }, `${d > 0 ? '+' : d < 0 ? '−' : '±'}${Math.abs(d).toFixed(1)}`); }
          return h('div', { class: 'cmp-bcell', role: 'cell' }, h('div', { class: 'v' }, v == null ? 'n/a' : pct(v), delta),
            h('div', { class: 's' }, v == null && noConf ? 'needs confidences' : c.sub(o)));
        });
      };
      put(boardEl,
        h('div', { class: 'cmp-brow head', role: 'row' }, h('div', { class: 'cmp-bcell', role: 'columnheader' }), cols.map(c => h('div', { class: 'cmp-bcell', role: 'columnheader', 'data-tip': c.tip }, c.label))),
        S.map(s => h('div', { class: 'cmp-brow', role: 'row' },
          h('div', { class: 'cmp-bcell model', role: 'rowheader' }, h('span', { class: `cmp-tag ${s}` }, s.toUpperCase()),
            h('div', { class: 'col', style: 'gap:1px;min-width:0' }, h('span', { class: 'strong' }, `Model ${s.toUpperCase()}`), h('span', { class: 'lbl', title: sourceTip(s) }, sourceLabel(s)),
              h('span', { class: 'macro', 'data-tip': 'Macro average: the mean of the per-class precision and recall, each class weighing the same' }, `macro P ${pct(stats[s].macroP)} · R ${pct(stats[s].macroR)}`))),
          cellsOf(s))));
    }

    let deferT = null;
    function setK(k, { fromViewer = false } = {}) {
      k = Math.max(0, Math.min(100, k));
      if (k === view.k) return;
      view.k = k; thrIn.value = k / 100; thrNum.value = (k / 100).toFixed(2);
      drawTiles(); cF1 && cF1.setK(k); cPR && cPR.setK(k);
      if (view.pcTab === 'op') drawPerClass();
      drawSizes();
      syncUrl();
      clearTimeout(deferT);
      deferT = setTimeout(() => { loadCm(); if (!fromViewer || !viewer || !viewer.open) loadRows(); else flagsDirty = true; }, 220);
    }
    function setSel(sel) {
      view.sel = sel;
      drawTiles(); drawCharts(); drawPerClass(); drawSizes(); drawUnseen(); syncUrl();
      clearTimeout(deferT);
      deferT = setTimeout(() => { loadCm(); loadRows(); }, 120);
    }

    drawTiles(); drawCharts(); drawPerClass(); drawSizes(); drawUnseen(); drawToolbar(); loadCm(); loadRows(); refreshFlagLine();
    // The matrix shading is computed from the theme's ramp: repaint it when the theme (or the OS scheme) changes.
    const mq = window.matchMedia('(prefers-color-scheme: dark)');
    const repaint = () => requestAnimationFrame(() => loadCm());
    mq.addEventListener('change', repaint);
    destroyers.push(() => mq.removeEventListener('change', repaint));
    if (F.state && F.state.subscribe) destroyers.push(F.state.subscribe(repaint, (st) => st.theme));
    return { destroy: () => { clearTimeout(deferT); destroyers.forEach(f => f()); } };
  }

  // ------------------------------------------------------------------ evaluations saved by Compare 1.x
  function renderLegacy() {
    const ev = cur.meta;
    const hasB = !!(ev.summary && ev.summary.b);
    const S = hasB ? ['a', 'b'] : ['a'];
    results.replaceChildren(header(null),
      h('div', { class: 'callout warn', style: 'margin-top:10px' }, icon('alert', 14), h('div', { class: 'col', style: 'gap:6px' },
        h('span', `Saved by an earlier version of Compare: AP was computed only from predictions with confidence ≥ ${Number(ev.conf).toFixed(2)} (so mAP moved with the threshold) and with all-point interpolation. Re-run it for threshold-independent mAP, curves, the confusion matrix, object sizes and flagging.`),
        h('div', h('button', { class: 'btn btn-sm', onClick: () => { prefill({ ...ev, metric: 'coco' }); page.scrollTo({ top: 0 }); } }, icon('refresh', 13), 'Load settings to re-run')))));
    const tiles = h('div', { class: 'cmp-tiles' }, S.map(s => {
      const o = ev.summary[s].overall;
      const src = ev.sources && ev.sources[s];
      return h('div', { class: 'card cmp-model' }, h('div', { class: 'card-header' }, h('span', { class: `cmp-tag ${s}` }, s.toUpperCase()), h('h3', `Model ${s.toUpperCase()}`), h('span', { class: 'lbl', title: src && src.path_host }, src ? src.label : '')),
        h('div', { class: 'card-body' }, h('div', { class: 'cmp-stats' },
          ui.statTile({ label: `mAP@${ev.iou} (conf ≥ ${ev.conf})`, value: pct(o.map50) }), ui.statTile({ label: 'Precision', value: pct(o.precision), sub: `${fmt.num(o.tp)} TP · ${fmt.num(o.fp)} FP` }),
          ui.statTile({ label: 'Recall', value: pct(o.recall), sub: `${fmt.num(o.fn)} FN of ${fmt.num(o.gt)}` }), ui.statTile({ label: 'F1', value: pct(o.f1) }))));
    }));
    results.appendChild(tiles);
    const clsRows = new Map();
    for (const s of S) for (const c of ev.summary[s].classes) clsRows.set(c.cls, { ...(clsRows.get(c.cls) || {}), [s]: c });
    results.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Per class')), h('div', { class: 'card-body tight', style: 'overflow-x:auto' }, h('table', { class: 'table cmp-pc' },
      h('thead', h('tr', h('th', 'Class'), h('th', { class: 'num' }, 'GT'), S.flatMap(s => [h('th', { class: `num grp-${s}` }, `${s.toUpperCase()} · AP`), h('th', { class: 'num' }, 'P'), h('th', { class: 'num' }, 'R'), h('th', { class: 'num' }, 'FP'), h('th', { class: 'num' }, 'FN')]))),
      h('tbody', [...clsRows.keys()].sort((x, y) => x - y).map(c => { const r = clsRows.get(c); const any = r.a || r.b; return h('tr', h('td', `${c} · ${cname(c)}`), h('td', { class: 'num' }, fmt.num(any.gt)),
        S.flatMap(s => r[s] ? [h('td', { class: 'num' }, pct(r[s].ap)), h('td', { class: 'num' }, pct(r[s].precision)), h('td', { class: 'num' }, pct(r[s].recall)), h('td', { class: 'num' }, fmt.num(r[s].fp)), h('td', { class: 'num' }, fmt.num(r[s].fn))] : [0, 1, 2, 3, 4].map(() => h('td', { class: 'num cmp-na' }, '–')))); }))))));
    const body = h('tbody'); const countEl = h('span', { class: 'result-count' }); const moreBox = h('div', { class: 'load-more', style: 'padding:10px' });
    results.appendChild(h('div', { class: 'card cmp-section' }, h('div', { class: 'card-header' }, h('h3', 'Per image'), countEl, h('span', { class: 'sub' }, 'errors at the saved threshold')),
      h('div', { class: 'card-body tight', style: 'overflow:auto;max-height:70vh' }, h('table', { class: 'table cmp-imgs' }, h('thead', h('tr', h('th', 'Image'), h('th', 'Split'), h('th', { class: 'num' }, 'GT'), S.map(s => h('th', { class: `num grp-${s}` }, `${s.toUpperCase()} · TP / FP / FN`)))), body), moreBox)));
    const items = []; let total = 0;
    const nn = (v, kind) => h('span', { class: `cmp-n ${v ? kind : 'zero'}` }, String(v));
    const row = (r, idx) => h('tr', { class: 'clickable', tabindex: 0, onClick: () => openLegacy(idx), onKeydown: (e) => { if (e.key === 'Enter') openLegacy(idx); } }, h('td', { class: 'path', title: r.rel_path }, midEllipsis(r.rel_path, 70)), h('td', { class: 'small' }, r.split || '—'), h('td', { class: 'num' }, r.gt),
      S.map(s => h('td', { class: 'num errs' }, r[s] ? [nn(r[s].tp, 'tp'), ' ', nn(r[s].fp, 'fp'), ' ', nn(r[s].fn, 'fn')] : '–')));
    async function more() {
      const r = await F.api.get(`${API}/evals/${ev.id}/rows`, { dataset_id: ds.id, show: 'errors', offset: items.length, limit: 100 });
      const start = items.length; items.push(...r.items); total = r.total;
      body.append(...r.items.map((x, i) => row(x, start + i)));
      countEl.textContent = `${fmt.num(total)} images with errors${items.length < total ? ` · showing ${fmt.num(items.length)}` : ''}`;
      put(moreBox, items.length < total ? h('button', { class: 'btn btn-sm', onClick: () => more().catch(e => ui.toast(e.message, { type: 'error' })) }, 'Load 100 more') : null);
    }
    function openLegacy(idx) {
      const list = { total: () => total, loaded: () => items.length, peek: (j) => items[j], async get(j) { while (j >= items.length && items.length < total) await more(); return items[j] || null; } };
      const evLike = { ...ev, ious: [ev.iou], primary: 0, sources: Object.fromEntries(S.map(s => [s, { type: 'folder', label: (ev.sources && ev.sources[s] && ev.sources[s].label) || s.toUpperCase(), path_host: ev[`pred_${s}_host`] }])) };
      viewer = openViewer({ F, API, ds, names, ev: evLike, list, startIndex: idx, k: Math.round((ev.conf || 0) * 100), classes: undefined, onK: () => {}, onFlagChange: () => {} });
    }
    more().catch(e => moreBox.replaceChildren(h('div', { class: 'callout danger' }, e.message)));
    return { destroy: () => {} };
  }

  // ------------------------------------------------------------------ start
  drawSaved();
  await loadSaved();
  const qrun = ctx.query.get('run'), qrunB = ctx.query.get('run_b');
  if (qrun || qrunB) {
    await loadRuns();
    if (qrun) { setup.a.type = 'run'; setup.a.run_id = Number(qrun); srcViews.a.draw(); recheck('a'); }
    if (qrunB) { setup.b.type = 'run'; setup.b.run_id = Number(qrunB); srcViews.b.draw(); recheck('b'); }
  }
  const qe = ctx.query.get('eval');
  if (qe) {
    const t = Number(ctx.query.get('t'));
    loadEval(qe, { k: ctx.query.get('t') != null && isFinite(t) ? Math.round(t * 100) : undefined });
  } else if (saved.length && !qrun) {
    // Land on the latest result instead of an empty page; the setup card stays above it.
    loadEval(saved[0].id);
  }
  return () => {
    destroyed = true;
    if (viewer) viewer.close();
    parts && parts.destroy();
    cleanups.forEach(f => { try { f(); } catch (e) { /* already stopped */ } });
  };
}
