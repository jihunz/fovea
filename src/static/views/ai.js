// AI tab: provider-backed batch labeling, two-stage (detector → classifier) setup with a test preview, label audit,
// provider benchmark on your own labels, class cards (the prompts people and models share), runs, usage and cost.
import {
  AUDIT_KINDS, add, mount, reveal, openClassMap, idsScope, sameSource, notifyChanged, deviceText, errorInfo, isClosedModel, SETTINGS_URL, activeAiJobs, batchForm, basename, canClassify, canDetect, cardsMissing, classInfo, closedMap,
  defaultModel, defaultSource, describeSource, duration, firstUsable, imagesById, isClosed, isCloud, jobBox, jobTitle, loadCards,
  loadProfile, loadProviders, midEllipsis, modelClassNames, modelName, overlayThumb, patchProfile, peekModelClasses, pickSource,
  predOverlay, previewItems, priceText, profileOf, provOf, providerState, requestFor, reviewLinks, saveCard, scrub, usd, usdRange, when,
  whereText, canGenerate, providerTone, whileChecking, modelOf, modelProblem, providerBatchOf, batchStateText, batchOpen, notSentText, latestOnly,
  EFFORTS, effortSeg, effortOf, effortLabel, defaultEffort, costKind, unpricedText, relPath, pathLine,
  addClassesLink, spendCheck, confirmSpend, sourceCap, REMOTE_AI_NOTE, remoteNoteShown,
} from './ai-panel.js';

export function install(fovea) {
  fovea.registerTab({ id: 'ai', label: 'AI', icon: 'sparkles', order: 35, render });
  if (fovea.keys && fovea.keys.registerKeys) fovea.keys.registerKeys('ai', 'AI tab', [['↑ ↓', 'Class cards: previous / next class (list focused)']], 70);
}

const SECTIONS = [
  ['overview', 'Overview', 'layout'], ['label', 'Label images', 'sparkles'], ['two-stage', 'Two-stage', 'layers'],
  ['audit', 'Label audit', 'fileSearch'], ['benchmark', 'Benchmark', 'barChart'], ['cards', 'Class cards', 'tags'], ['runs', 'Runs & cost', 'clock'],
];
const KIND_LABEL = { missing: 'Missing boxes', loose: 'Loose boxes', swap: 'Class swaps', spurious: 'Boxes without support' };   // annotate hook (r4): one name for ai_spurious in every view
const KIND_HINT = {
  missing: 'The model sees an object where you have no box.',
  loose: 'Your box overlaps the model’s but is badly placed or sized.',
  swap: 'The model is confident this box is a different class.',
  spurious: 'Nothing supports this box — it may be on background.',
};
const PROPOSAL_KINDS = ['detect', 'prev_frame', 'interp', 'refine'];   // everything but audit findings
export const KIND_COUNT = { missing: ['missing box', 'missing boxes'], loose: ['loose box', 'loose boxes'], swap: ['class swap', 'class swaps'], spurious: ['box without support', 'boxes without support'] };   // shell hook (r4): Overview's suspect rows use the same words
const kindCount = (k, n) => `${n} ${(KIND_COUNT[k] || [k, k])[n === 1 ? 0 : 1]}`;
/** A review link into the audit findings, most suspect first. The order is named, not left to the view's default
    for sort=audit: "worst first" must open on the worst image whatever that default is. */
export const auditLink = (dsId, view, q = {}) => `/d/${dsId}/${view}?${new URLSearchParams({ ...q, sort: 'audit', order: 'desc' }).toString()}`;
/** "YOLO12m", "Two-stage · YOLO12m → Claude Haiku 5.5" or "2 setups" for a run row. */
function runSource(C, r) {
  const p = r.params || {}, o = p.options || {};
  if (r.kind === 'benchmark' && p.config && p.config.label) return p.config.label;
  if (r.kind === 'benchmark' && Array.isArray(p.configs)) return `${p.configs.length} setup${p.configs.length === 1 ? '' : 's'}: ${p.configs.map((c) => c.label || c.model || c.provider).join(' · ')}`;
  if (o.pipeline === 'two_stage') {
    const dt = o.detector || {}, cl = o.classifier || {};
    return `Two-stage · ${modelName(provOf(C.P, dt.provider), dt.model)} → ${modelName(provOf(C.P, cl.provider || r.provider), cl.model || r.model)}`;
  }
  return r.provider ? modelName(provOf(C.P, r.provider), r.model) : '–';
}
/** The newest benchmark: one run per setup, grouped by the job that ran them (older servers: one run, list result). */
function lastBenchmark(runs) {
  const bench = (runs || []).filter((r) => r.kind === 'benchmark' && r.status !== 'deleted');
  if (!bench.length) return null;
  const first = bench[0];
  if (Array.isArray(first.result)) return first.result.length ? { rows: first.result, created_at: first.created_at, images: first.images, iou: first.params && first.params.iou } : null;
  const group = bench.filter((r) => (first.job_id ? r.job_id === first.job_id : r.id === first.id) && r.result && r.result.f1 != null);
  if (!group.length) return null;
  return { rows: group.slice().sort((a, b) => a.id - b.id).map((r) => r.result), created_at: first.created_at, images: (first.params && first.params.images) || first.images, iou: first.params && first.params.iou };
}
const pendingOf = (r) => (r.preds && typeof r.preds === 'object' ? r.preds.pending || 0 : r.pending);
/** Class name for people: a placeholder like class_0 is replaced by its card prompt when there is one. */
function displayName(C, c) { const ci = classInfo(C.names, C.cards, c); return ci.prompt && /^class_\d+$/.test(ci.name) ? ci.prompt : ci.name; }
const RUN_KIND = { propose: 'Single image', batch: 'Batch', audit: 'Audit', benchmark: 'Benchmark', interp: 'Interpolation', prev_frame: 'Previous frame', refine: 'Refine', classify: 'Classify', preview: 'Preview', review: 'Image review',
  name_classes: 'Class names', nl_filter: 'Ask AI search' };

async function render(el, { ctx, dataset, fovea }) {
  const { h, ui, icon, fmt, link } = fovea;
  const page = h('div', { class: 'page' });
  const inner = h('div', { class: 'page-inner ai-page' });
  page.appendChild(inner); el.appendChild(page);
  let ds = dataset;
  const C = { fovea, ds, P: null, prof: null, cards: {}, names: ds.classes || [], stats: null };
  let destroySec = null, destroyed = false, current = null;
  const cleanups = [];

  const nav = h('div', { class: 'ai-nav', role: 'tablist', 'aria-label': 'AI sections' });
  const body = h('div', { class: 'ai-body' });
  const headNote = h('div', { class: 'ai-head-note' });
  add(inner, h('div', { class: 'ai-head' },
    h('div', { class: 'grow' }, h('h2', { class: 'page-title' }, 'AI'), h('div', { class: 'page-sub' }, 'Propose, audit and benchmark labels with models on this machine or in the cloud. Nothing reaches your label files until you accept it or choose to write.')),
    headNote), nav, body);

  function go(id, { push = true } = {}) {
    if (!SECTIONS.some(([s]) => s === id)) id = 'overview';
    current = id;
    if (push) fovea.router.replaceQuery({ s: id === 'overview' ? null : id });
    mount(nav, ...SECTIONS.map(([s, label, ic]) => h('button', { class: `ai-nav-btn ${s === id ? 'active' : ''}`, role: 'tab', 'aria-selected': String(s === id), type: 'button',
      onClick: () => go(s) }, icon(ic, 14), h('span', label))));
    if (destroySec) { try { destroySec(); } catch (e) { console.error(e); } destroySec = null; }
    mount(body);
    const fn = { overview: secOverview, label: secLabel, 'two-stage': secTwoStage, audit: secAudit, benchmark: secBenchmark, cards: secCards, runs: secRuns }[id];
    const res = fn(body, C, go, ctx);
    destroySec = typeof res === 'function' ? res : null;
  }
  C.go = go;

  // Delayed loading line (nothing for the first 400 ms).
  body.appendChild(h('div', { class: 'row small muted' }, ui.spinner(), 'Loading AI providers…'));
  // Fresh on every visit: a key added in Settings a moment ago must show up here (the call is cheap, no network).
  const [P, prof, cards] = await Promise.all([loadProviders(fovea, { force: true }), loadProfile(fovea, ds.id), loadCards(fovea, ds.id, { force: true }).catch(() => ({}))]);
  if (destroyed) return () => {};
  Object.assign(C, { P, prof, cards });
  renderHeadNote();
  const q = ctx && ctx.query;
  go((q && (q.get('s') || q.get('tab'))) || 'overview', { push: false });
  if (q && q.get('suggest') === '1') {
    fovea.router.replaceQuery({ suggest: null, tab: null, s: current === 'overview' ? null : current });
    openSuggestNames(fovea, ds, { P: C.P });
  }

  cleanups.push(fovea.bus.on('dataset:updated', async (d) => {
    if (d && d.id && d.id !== ds.id) return;
    const before = JSON.stringify(C.names);
    try { ds = d && d.id ? { ...ds, ...d } : await fovea.getDataset(ds.id, { force: true }); } catch (e) { return; }
    C.ds = ds; C.names = ds.classes || [];
    if (JSON.stringify(C.names) !== before && current && !destroyed) go(current, { push: false });   // classes renamed / added
  }));
  cleanups.push(fovea.bus.on('ai:profile', (e) => { if (e && e.dataset_id === ds.id) C.prof = profileOf(ds.id); }));
  // Cards written elsewhere (naming dialog, Overview drawer): the next render reads the fresh copy.
  cleanups.push(fovea.bus.on('ai:cards', async (e) => {
    if (!e || e.dataset_id !== ds.id) return;
    C.cards = await loadCards(fovea, ds.id).catch(() => C.cards);
  }));
  // A reachability probe still pending (OpenAI-compatible server): show the answer once it is in. Only the header and
  // the Overview's provider list are redrawn; a section with a form keeps what the person typed (its pickers read
  // the fresh list when opened).
  const onFresh = (fresh) => { if (destroyed) return; C.P = fresh; renderHeadNote(); if (current === 'overview') go(current, { push: false }); };
  let stopChecking = whileChecking(fovea, C.P, onFresh);
  cleanups.push(() => stopChecking());
  cleanups.push(fovea.bus.on('ai:providers', async () => {           // Settings changed a key / config
    C.P = await loadProviders(fovea, { force: true });
    if (destroyed) return;
    renderHeadNote();
    if (current) go(current, { push: false });
    stopChecking();
    stopChecking = whileChecking(fovea, C.P, onFresh);
  }));
  function renderHeadNote() {
    const ok = (C.P.providers || []).filter((p) => providerState(C.P, p).ok);
    mount(headNote, 
      C.P.error ? [h('span', { class: 'chip danger' }, icon('alertCircle', 12), C.P.error), h('button', { class: 'btn btn-sm', type: 'button', onClick: async () => {
        C.P = await loadProviders(fovea, { force: true });
        renderHeadNote();
        if (!C.P.error && current) go(current, { push: false });
      } }, icon('refresh', 13), 'Retry')]
        : h('span', { class: 'chip' }, icon('cpu', 12), `${ok.length} of ${(C.P.providers || []).length} providers ready${deviceText(C.P) ? ` · ${deviceText(C.P)}` : ''}`),
      // Every section, not only those with the privacy line: a browser on another machine cannot use cloud models.
      remoteNoteShown(C.P) ? h('span', { class: 'chip warn', title: REMOTE_AI_NOTE }, icon('shield', 12), 'Cloud AI off from this browser') : null,
      link(SETTINGS_URL, { class: 'btn btn-sm' }, icon('settings', 13), 'Providers & keys'));
  }
  return () => {
    destroyed = true;
    if (destroySec) try { destroySec(); } catch (e) { console.error(e); }
    cleanups.forEach((u) => { try { u(); } catch (e) { /* gone */ } });
  };
}

// ====================================================================== helpers
function card(h, title, sub, ...children) {
  return h('section', { class: 'card ai-card' }, h('div', { class: 'card-header' }, h('h3', title), sub ? h('span', { class: 'sub' }, sub) : null), h('div', { class: 'card-body' }, ...children));
}
function loadingLine(fovea, text) { return fovea.h('div', { class: 'row small muted' }, fovea.ui.spinner(), text); }
function errLine(fovea, text) { return fovea.h('div', { class: 'aip-msg bad' }, fovea.icon('alertCircle', 13), fovea.h('span', scrub(text))); }
const safe = (p, fb = null) => p.then((x) => x).catch(() => fb);
async function countImages(fovea, dsId, filters) {
  const r = await fovea.api.get(`/api/datasets/${dsId}/images`, { ...filters, limit: 1 });
  // A server that does not know a filter ignores it and would count every image: report "unknown" instead.
  if (r.filters && Object.keys(filters).some((k) => filters[k] != null && filters[k] !== '' && !(k in r.filters))) return null;
  return r.total;
}
/** Up to n ids from a filtered list in a random order. With a `seed`: the first n of that seeded shuffle — a fresh
    random sample for every seed (a run). Without: a window at a random offset of the dataset's stable shuffled order
    (previews and tests, where any few images do). */
async function sampleIds(fovea, dsId, filters, n, { shuffle = true, seed = null } = {}) {
  const total = (await countImages(fovea, dsId, filters)) || 0;
  if (!total) return [];
  const offset = !seed && shuffle && total > n ? Math.floor(Math.random() * (total - n)) : 0;
  const ids = [];
  for (let off = offset; ids.length < n && off < total; off += 1000) {
    const r = await fovea.api.get(`/api/datasets/${dsId}/images`, { ...filters, sort: 'random', ...(seed ? { seed } : {}), offset: off, limit: Math.min(1000, n - ids.length) });
    (r.items || []).forEach((it) => ids.push(it.id));
    if (!(r.items || []).length) break;
  }
  return ids;
}
/** A new permutation for sort=random: each run's "Random sample N" is a different N images. */
export const newSeed = () => 1 + Math.floor(Math.random() * 2147483646);
/** Per-class box counts ({cls: boxes}; classes without boxes may be missing), or null when they cannot be read. */
function classCounts(fovea, dsId) { return safe(fovea.api.get(`/api/datasets/${dsId}/classes/counts`).then((r) => (r && r.counts) || {}), null); }
/** Classes that only have a placeholder name (class_3 …), split by whether they have boxes a model could look at.
    Without counts every placeholder counts as having boxes. */
export function placeholderClasses(names, cards, counts) {
  const all = names.map((_, c) => c).filter((c) => classInfo(names, cards, c).placeholder);
  const withBoxes = counts ? all.filter((c) => (Number(counts[c] ?? counts[String(c)]) || 0) > 0) : all;
  return { all, withBoxes, empty: all.filter((c) => !withBoxes.includes(c)) };
}
/** "class_0, class_3 and 4 more". */
export function classList(names, cs, max = 3) {
  const shown = cs.slice(0, max).map((c) => names[c] || `class_${c}`);
  if (cs.length <= max) return shown.length > 1 ? `${shown.slice(0, -1).join(', ')} and ${shown[shown.length - 1]}` : shown.join('');
  return `${shown.join(', ')} and ${cs.length - max} more`;
}
function splitsOf(ds) { return [...new Set((ds.splits || []).filter(Boolean))]; }
function splitSeg(fovea, ds, value, onChange) {
  const sp = splitsOf(ds);
  if (sp.length < 2) return null;
  const s = fovea.ui.seg([{ value: '', label: 'All splits' }, ...sp.map((x) => ({ value: x, label: x }))], value || '', onChange);
  s.classList.add('ai-seg');
  return s;
}
function srcButton(fovea, C, getSrc, setSrc, { purpose = 'detect', allowTwoStage = true } = {}) {
  const { h, icon } = fovea;
  const btn = h('button', { class: 'btn aip-pick', type: 'button', 'aria-haspopup': 'menu' });
  const paint = () => {
    const d = describeSource(C.P, C.prof, getSrc());
    mount(btn, icon(d.twoStage ? 'layers' : d.cloud ? 'zap' : 'cpu', 14), h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, d.label), h('span', { class: 's truncate' }, d.sub)), icon('chevronDown', 12));
    btn.title = `${d.label} — ${d.sub}`;
    return d;
  };
  btn.addEventListener('click', () => pickSource(btn, { fovea, P: C.P, prof: C.prof, dsId: C.ds.id, current: getSrc(), purpose, allowTwoStage, onPick: (s) => { setSrc(s); paint(); } }));
  btn.paint = paint;
  paint();
  return btn;
}
function srcNote(fovea, C, src) {
  const { h, link } = fovea;
  const d = describeSource(C.P, C.prof, src);
  if (!d.ok) return h('div', { class: 'hint' }, h('span', { class: 'aip-bad' }, d.reason), d.settings ? [' ', link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings')] : null);
  return h('div', { class: 'hint' }, d.where, d.price ? ` · ${d.price}` : '');
}
function privacyLine(fovea, C) {
  const { h, icon } = fovea;
  const cloud = (C.P.providers || []).filter((p) => isCloud(p) && providerState(C.P, p).ok);
  // From another machine "No cloud provider is set up" would be wrong (they may be, for the computer Fovea runs on).
  const remote = remoteNoteShown(C.P);
  return h('div', { class: `ai-privacy ${remote ? 'ai-privacy-remote' : ''}` }, icon(remote ? 'shield' : 'info', 13), h('span',
    'Local providers keep images on this machine. A cloud provider receives the images (or crops) and your class prompts for that request only; keys stay on the server and never reach the browser.',
    remote ? ` ${REMOTE_AI_NOTE}` : cloud.length ? ` Ready in the cloud: ${cloud.map((p) => p.name).join(', ')}.` : ' No cloud provider is set up.'));
}
function kindLabel(C, k) {
  const L = C.issueLabels || {};
  return L[`ai_${k}`] || KIND_LABEL[k] || k;
}
async function issueLabels(fovea, C) {
  if (C.issueLabels) return C.issueLabels;
  const r = await safe(fovea.api.get(`/api/datasets/${C.ds.id}/stats`), null);
  C.stats = r && r.stats; C.issueLabels = (r && r.issue_labels) || {};
  return C.issueLabels;
}
function iou(a, b) {
  const ax0 = a[0] - a[2] / 2, ay0 = a[1] - a[3] / 2, ax1 = a[0] + a[2] / 2, ay1 = a[1] + a[3] / 2;
  const bx0 = b[0] - b[2] / 2, by0 = b[1] - b[3] / 2, bx1 = b[0] + b[2] / 2, by1 = b[1] + b[3] / 2;
  const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)), ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  const inter = iw * ih, u = a[2] * a[3] + b[2] * b[3] - inter;
  return u > 0 ? inter / u : 0;
}
/** Re-attach to running AI jobs of this dataset (after a reload or a tab switch). */
function activeJobsInto(fovea, C, into, { onEnd } = {}) {
  const boxes = [];
  let stopped = false;
  activeAiJobs(fovea, C.ds.id).then((list) => {
    if (stopped) return;
    list.forEach(({ job, run }) => {
      const target = run && run.params && run.params.target;
      const eco = run && run.params && run.params.speed === 'economy';
      const b = jobBox(fovea, { job, title: (run ? jobTitle(run.kind) : jobTitle(job.kind)) + (eco ? ' · economy' : ''), writes: !!target && target !== 'proposals',
        actions: () => (run && run.kind === 'batch') || /batch/.test(job.kind) ? reviewLinks(fovea, C.ds, target || 'proposals') : [], onEnd });
      boxes.push(b);
      into.appendChild(b.el);
    });
  });
  return () => { stopped = true; boxes.forEach((b) => b.stop()); };
}

// ====================================================================== overview
function secOverview(body, C, go) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt, link } = fovea;
  const tiles = h('div', { class: 'stat-grid mb-16' });
  const flow = h('div', { class: 'ai-flow' });
  const jobsEl = h('div', { class: 'col gap-8' });
  const provs = h('div', { class: 'ai-provs' });
  add(body, tiles,
    h('div', { class: 'ai-cols' },
      h('div', { class: 'col gap-16' }, card(h, 'Workflow', 'from class descriptions to reviewed labels', flow), jobsEl),
      h('div', { class: 'col gap-16' }, card(h, 'Providers', null, provs, privacyLine(fovea, C)))));
  let alive = true;
  const stopJobs = activeJobsInto(fovea, C, jobsEl, { onEnd: () => alive && load() });

  // providers list
  for (const p of C.P.providers || []) {
    const st = providerState(C.P, p), tone = providerTone(C.P, p);
    const m = defaultModel(p);
    // Configured models a local server has not pulled are named, so "ready" never hides a model that cannot run.
    const missing = st.ok ? (p.models || []).filter((x) => modelProblem(x) && modelProblem(x) !== 'cannot draw boxes') : [];
    provs.appendChild(h('div', { class: 'ai-prov' }, h('span', { class: `dot ${tone}`, title: tone === 'ready' ? 'Ready' : st.reason }),
      h('div', { class: 'grow' }, h('div', h('span', { class: 'strong' }, p.name), h('span', { class: 'faint small' }, ` · ${isCloud(p) ? 'cloud' : 'local'}`)),
        h('div', { class: `xs truncate ${st.ok ? 'muted' : tone === 'bad' ? 'aip-bad' : 'muted'}`, title: st.ok ? '' : st.reason }, st.ok ? [m ? m.name || m.id : '', p.key && p.key.present ? ` · key ${p.key.hint || 'saved'}${p.key.source === 'env' ? ' (environment)' : ''}` : ''].join('') : st.reason),
        missing.length ? h('div', { class: 'xs aip-warn truncate', title: missing.map((x) => `${x.name || x.id}: ${modelProblem(x)}`).join('\n') }, `${missing.length} configured model${missing.length === 1 ? '' : 's'} not usable: ${missing.map((x) => x.name || x.id).join(', ')}`) : null)));
  }
  if (!(C.P.providers || []).length) provs.appendChild(h('div', { class: 'hint' }, C.P.error || 'No providers on this server.'));

  async function load() {
    mount(tiles, loadingLine(fovea, 'Loading…'));
    const [usage, dsUsage, pending, audit, runs, unlabeled, counts] = await Promise.all([
      safe(fovea.api.get('/api/ai/usage', { days: 30 })), safe(fovea.api.get('/api/ai/usage', { days: 30, dataset_id: ds.id })), safe(countImages(fovea, ds.id, { ai: 'pending' })),
      safe(fovea.api.get(`/api/datasets/${ds.id}/audit/summary`)), safe(fovea.api.get('/api/ai/runs', { dataset_id: ds.id, limit: 100 }).then((r) => r.runs || [])),
      safe(countImages(fovea, ds.id, { labeled: '0' })), classCounts(fovea, ds.id),
    ]);
    if (!alive) return;
    const runsList = runs || [];
    mount(tiles,
      ui.statTile({ label: 'Spent on this dataset · 30 days', value: dsUsage ? usd(dsUsage.total_usd || 0) : '–', sub: usage ? `${usd(usage.total_usd || 0)} across all datasets` : '', icon: 'clock', onClick: () => go('runs') }),
      ui.statTile({ label: 'Images with proposals', value: pending != null ? fmt.num(pending) : '–', sub: 'waiting for review', icon: 'sparkles', tone: pending ? 'warn' : null, onClick: pending ? () => fovea.router.navigate(`/d/${ds.id}/annotate?ai=pending`) : null }),
      ui.statTile({ label: 'Suspect labels', value: audit ? fmt.num(audit.images_flagged || 0) : '–', sub: audit && audit.run ? `audit ${when(audit.run.created_at)}` : 'no audit yet', icon: 'fileSearch', tone: audit && audit.images_flagged ? 'danger' : null, onClick: () => go('audit') }),
      ui.statTile({ label: 'Without labels', value: unlabeled != null ? fmt.num(unlabeled) : '–', sub: 'images with no label file', icon: 'images', onClick: () => go('label') }));
    renderFlow({ pending, audit, runs: runsList, unlabeled, counts });
  }
  function renderFlow({ pending, audit, runs, unlabeled, counts }) {
    const names = C.names;
    const described = names.filter((_, c) => classInfo(names, C.cards, c).described).length;
    // Name the placeholder classes; a model can only suggest a name for one that has boxes to look at.
    const ph = placeholderClasses(names, C.cards, counts);
    const unnamed = ph.all.length;
    const src = defaultSource(C.P, C.prof);
    const d = describeSource(C.P, C.prof, src);
    let closedNoMatch = false;
    if (d.p && isClosed(d.p)) {
      const mn = peekModelClasses(d.p, d.model);
      closedNoMatch = !!mn && !closedMap(C.prof, d.p, d.model, mn.names, names).targets.length;
    }
    const bench = lastBenchmark(runs);
    const best = bench ? bench.rows.slice().sort((a, b) => (b.f1 || 0) - (a.f1 || 0))[0] : null;
    const steps = [
      { done: names.length > 0 && described === names.length, title: 'Describe your classes',
        text: !names.length ? 'This dataset has no classes yet.' : !unnamed ? `${described} of ${names.length} classes have a prompt or description. The same text guides people and models.`
          : [ph.withBoxes.length ? `${unnamed} of ${names.length} classes ${unnamed === 1 ? 'has' : 'have'} only a placeholder name (${classList(names, ph.all)}). Models need a description of what each class means — e.g. “person lying on the ground after a fall”.` : '',
            ph.empty.length ? `${classList(names, ph.empty)} ${ph.empty.length === 1 ? 'has' : 'have'} ${ph.withBoxes.length ? '' : 'only a placeholder name and '}no boxes, so AI cannot suggest a name — describe ${ph.empty.length === 1 ? 'it' : 'them'} by hand in Class cards.` : ''].filter(Boolean).join(' '),
        actions: [ph.withBoxes.length ? h('button', { class: 'btn btn-sm btn-primary', type: 'button', onClick: () => openSuggestNames(fovea, C.ds, { P: C.P }) }, icon('sparkles', 12), 'Suggest names with AI') : null,
          h('button', { class: `btn btn-sm ${unnamed && !ph.withBoxes.length ? 'btn-primary' : ''}`, type: 'button', onClick: () => go('cards') }, icon('tags', 12), 'Class cards')].filter(Boolean) },
      { done: !!(C.prof.source || C.prof.twoStage), title: 'Choose how AI finds objects',
        text: closedNoMatch || unnamed ? 'For posture or state classes (fall / standing / sitting), a person detector plus a classifier per box works best: the detector draws tight boxes, the classifier decides the class.' : `Current choice: ${d.label}${d.sub ? ` (${d.sub})` : ''}.`,
        actions: [h('button', { class: 'btn btn-sm', type: 'button', onClick: () => go('two-stage') }, icon('layers', 12), C.prof.twoStage ? 'Two-stage setup' : 'Set up two-stage')] },
      { done: !!bench, title: 'Measure it on your labels',
        text: best ? `Last benchmark: best F1 ${(best.f1 * 100).toFixed(0)}% — ${best.config && (best.config.label || `${best.config.provider} ${best.config.model || ''}`)} (${when(bench.created_at)}).` : 'Run 2–4 setups on a sample of labeled images and compare precision, recall, speed and cost before a long run.',
        actions: [h('button', { class: 'btn btn-sm', type: 'button', onClick: () => go('benchmark') }, icon('barChart', 12), 'Benchmark')] },
      { done: unlabeled === 0, title: 'Label new images',
        text: unlabeled ? `${fmt.num(unlabeled)} images have no label file. Run AI on them into proposals — nothing is written until you accept.` : 'Every image has a label file.',
        actions: [h('button', { class: 'btn btn-sm', type: 'button', onClick: () => go('label') }, icon('sparkles', 12), 'Label images')] },
      { done: pending === 0 && !!runs.length, title: 'Review proposals',
        text: pending ? `${fmt.num(pending)} images have proposals. In Annotate: Enter accepts, Delete rejects, ⇧Enter accepts all visible.` : 'No proposals are waiting.',
        actions: pending ? reviewLinks(fovea, ds, 'proposals') : [] },
      { done: !!(audit && audit.run), title: 'Audit existing labels',
        text: audit && audit.run ? `${fmt.num(audit.images_flagged || 0)} images flagged by the last audit (${when(audit.run.created_at)}).` : 'Compare a model against your labels to find missing, loose and swapped boxes — most likely errors first.',
        actions: [h('button', { class: 'btn btn-sm', type: 'button', onClick: () => go('audit') }, icon('fileSearch', 12), audit && audit.images_flagged ? 'Review suspects' : 'Run an audit')] },
    ];
    mount(flow, h('ol', { class: 'ai-steps' }, steps.map((s, i) => h('li', { class: `ai-step ${s.done ? 'done' : ''}` },
      h('span', { class: 'ai-step-n', 'aria-hidden': 'true' }, s.done ? icon('check', 12) : String(i + 1)),
      h('div', { class: 'grow' }, h('div', { class: 'strong' }, s.title, s.done ? h('span', { class: 'sr-only' }, ' (done)') : null), h('div', { class: 'small muted' }, s.text),
        s.actions.length ? h('div', { class: 'row wrap gap-4 mt-8' }, s.actions) : null)))));
  }
  load();
  return () => { alive = false; stopJobs(); };
}

// ====================================================================== label images (batch launcher)
function datasetScopes(fovea, C, { split, limit, ids, set }) {
  const { h, fmt } = fovea;
  const ds = C.ds;
  const withSplit = (f) => (split.value ? { ...f, split: split.value } : f);
  const lim = () => (limit.value > 0 ? limit.value : null);
  const mk = (id, label, f) => {
    const countEl = h('span', { class: 'faint' });
    return { id, label, countEl, filters: () => withSplit(f),
      count: async () => { const n = await countImages(fovea, ds.id, withSplit(f)); return lim() ? Math.min(lim(), n) : n; },
      // "Random sample N": a new sample at every Start (a seeded shuffle), so a rerun is not the same N images again.
      resolve: async () => (lim() ? { image_ids: await sampleIds(fovea, ds.id, withSplit(f), lim(), { seed: newSeed() }) } : { filters: withSplit(f) }),
      sample: (k) => sampleIds(fovea, ds.id, withSplit(f), k) };
  };
  const scopes = [
    // A selection handed over from Explore (?set=<token>, more than 1,500 images or "all matching").
    ...(set ? [mk('set', 'The selection from Explore', { set })] : []),
    mk('unlabeled', 'Images without a label file', { labeled: '0' }),
    mk('nobox', 'Images with no boxes (incl. empty label files)', { labeled: 'nobox' }),
    mk('all', 'All images', {}),
    mk('labeled', 'Labeled images', { labeled: '1' }),
  ];
  scopes.push(idsScope(fovea, ids));
  return scopes;
}
function secLabel(body, C, go, ctx) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt } = fovea;
  const qIds = ctx && ctx.query && ctx.query.get('ids');
  const qSet = ctx && ctx.query && ctx.query.get('set');
  const split = { value: (C.prof.batch && C.prof.batch.split) || '' }, limit = { value: 0 }, ids = { value: qIds || '' };
  const formHost = h('div');
  const bulk = h('div', { class: 'col gap-12' });
  const jobsEl = h('div', { class: 'col gap-8' });
  let form = null, alive = true;
  const started = [];
  const stopJobs = activeJobsInto(fovea, C, jobsEl, { onEnd: () => refreshBulk() });
  const limitIn = h('input', { class: 'input input-sm', type: 'number', min: 0, max: 100000, placeholder: 'all', style: 'width:90px', onChange: (e) => { limit.value = Math.max(0, Number(e.target.value) || 0); if (form) form.refresh(); } });
  const seg = splitSeg(fovea, ds, split.value, (v) => { split.value = v; patchProfile(fovea, ds.id, { batch: { ...(profileOf(ds.id).batch || {}), split: v } }); if (form) form.refresh(); });
  // Split and sample narrow every scope below; they sit inside the form's "Images" field.
  const scopeExtra = h('div', { class: 'ai-fields-row' }, seg ? h('div', { class: 'field' }, h('span', { class: 'hint' }, 'Split'), seg) : null,
    h('div', { class: 'field' }, h('span', { class: 'hint' }, 'Random sample'), h('div', { class: 'row' }, limitIn, h('span', { class: 'hint' }, 'images (empty = all in scope)'))));
  const DATASET_SCOPES = ['unlabeled', 'nobox', 'all', 'labeled'];
  const lastScope = () => { const v = (profileOf(ds.id).batch || {}).scope; return DATASET_SCOPES.includes(v) ? v : null; };
  /** Open on a scope that has images: the one used last on this dataset, else the first of "without a label file",
      "with no boxes", "all" that is not empty (a fully labeled dataset has no image without a label file). The person's
      own choice, made while the counts load, is never overridden. */
  async function pickScope(f, scopes) {
    const first = f.scope;
    for (const id of [...new Set([lastScope(), 'unlabeled', 'nobox', 'all'].filter(Boolean))]) {
      const sc = scopes.find((x) => x.id === id);
      if (!sc) continue;
      const n = await sc.count().catch(() => null);
      if (!alive || form !== f || f.scope !== first) return;
      if (n) { if (id !== first) f.selectScope(id); return; }
    }
  }
  function build() {
    if (form) form.destroy();
    const scopes = datasetScopes(fovea, C, { split, limit, ids, set: qSet });
    form = batchForm({ fovea, ds, P: C.P, getP: () => C.P, scopes, scope: qSet ? 'set' : qIds ? 'ids' : lastScope() || 'unlabeled', src: C.labelSrc || undefined, cards: C.cards, scopeExtra,
      onStarted: (res) => {
        if (form && DATASET_SCOPES.includes(form.scope)) patchProfile(fovea, ds.id, { batch: { ...(profileOf(ds.id).batch || {}), scope: form.scope } });
        const jb = jobBox(fovea, { job: res.job, title: 'AI labeling', writes: res.target !== 'proposals', actions: () => reviewLinks(fovea, ds, res.target),
          onEnd: () => { refreshBulk(); notifyChanged(fovea, ds.id, { labels: res.target !== 'proposals' }); } });
        started.push(jb);
        jobsEl.prepend(jb.el);
        reveal(jb.el);
      }, onClose: () => build() });
    mount(formHost, form.el, h('div', { class: 'row aib-foot' }, form.footer()));
    if (!qSet && !qIds) pickScope(form, scopes);
  }
  add(body, h('div', { class: 'ai-cols' },
    h('div', { class: 'col gap-16' }, card(h, 'Run AI on many images', 'results go to proposals unless you choose to write', formHost), jobsEl),
    h('div', { class: 'col gap-16' }, card(h, 'Proposals waiting for review', null, bulk), privacyLine(fovea, C))));
  build();

  let bulkOut = null;      // ai-core hook (r4): the last accept's summary ("… left to decide in Annotate") survives the recount
  async function refreshBulk() {
    if (!bulk.firstChild) mount(bulk, loadingLine(fovea, 'Counting…'));     // later refreshes keep the old numbers until the new ones arrive
    const n = await safe(countImages(fovea, ds.id, { ai: 'pending' }));
    if (!alive) return;
    if (n == null) { mount(bulk, h('div', { class: 'hint' }, 'Proposal counts are not available on this server yet.')); return; }
    const acc = h('input', { class: 'input input-sm', type: 'number', min: 0.05, max: 0.99, step: 0.05, value: 0.5, style: 'width:76px', 'aria-label': 'Accept at or above confidence' });
    const rej = h('input', { class: 'input input-sm', type: 'number', min: 0.05, max: 0.99, step: 0.05, value: 0.25, style: 'width:76px', 'aria-label': 'Reject below confidence' });
    const out = bulkOut || (bulkOut = h('div', { class: 'col gap-8' }));
    mount(bulk, 
      h('div', h('span', { class: 'ai-big' }, fmt.num(n)), h('span', { class: 'muted' }, ` image${n === 1 ? '' : 's'} with pending proposals`)),
      n ? h('div', { class: 'row wrap gap-4' }, reviewLinks(fovea, ds, 'proposals')) : null,
      n ? h('div', { class: 'ai-bulk' },
        h('div', { class: 'row wrap' }, h('button', { class: 'btn btn-sm', type: 'button', onClick: () => acceptAll(Number(acc.value)) }, icon('check', 12), 'Accept all with confidence ≥'), acc),
        h('div', { class: 'hint' }, 'Writes the accepted boxes into label files (snapshot first). Proposals without a confidence (cloud models) are included. A box over one of the same class is skipped; one over a box of another class stays pending for you to decide in Annotate.'),
        h('div', { class: 'row wrap' }, h('button', { class: 'btn btn-sm', type: 'button', onClick: () => rejectBelow(Number(rej.value)) }, icon('x', 12), 'Reject all below'), rej),
        h('div', { class: 'row wrap' }, h('button', { class: 'btn btn-sm btn-danger', type: 'button', onClick: () => rejectAll() }, icon('trash', 12), 'Reject every pending proposal'))) : null,
      out);
    async function acceptAll(min) {
      if (!(await ui.confirm({ title: `Accept proposals ≥ ${min.toFixed(2)}?`, okLabel: 'Accept and write', message: `Proposals with confidence ≥ ${min.toFixed(2)} (and those without a confidence) on up to ${fmt.num(n)} images are written into label files. A snapshot is saved first; Overview → History restores it. Consider reviewing a few in Annotate first.` }))) return;
      try {
        const res = await fovea.api.post(`/api/datasets/${ds.id}/proposals/accept`, { filters: {}, min_conf: min });
        if (res.job) mount(out, jobBox(fovea, { job: res.job, title: 'Accepting proposals', writes: true, onEnd: () => { notifyChanged(fovea, ds.id, { labels: true }); refreshBulk(); } }).el);
        else { mount(out, h('div', { class: 'aip-msg ok' }, icon('checkCircle', 13), `Accepted ${fmt.num(res.accepted ?? res.updated ?? 0)} proposals.`)); refreshBulk(); }
      } catch (e) { mount(out, errLine(fovea, errorInfo(e).text)); }
    }
    async function rejectBelow(min) {
      if (!(min > 0 && min < 1)) { mount(out, errLine(fovea, 'Enter a confidence between 0.05 and 0.99.')); return; }
      // Dataset-wide and final (a rejected box is never proposed again): ask first, with the count, like Reject all.
      if (!(await ui.confirm({ title: `Reject proposals below ${min.toFixed(2)}?`, danger: true, okLabel: 'Reject',
        message: `Pending proposals with confidence below ${min.toFixed(2)} on up to ${fmt.num(n)} image${n === 1 ? '' : 's'} are marked rejected. Proposals without a confidence (cloud models) are kept, and label files are not touched. This cannot be undone here: rejected proposals stay on record, so the same boxes are not proposed again.` }))) return;
      try {
        const res = await fovea.api.post(`/api/datasets/${ds.id}/proposals/reject`, { filters: {}, min_conf: min, below: true, kinds: PROPOSAL_KINDS });
        notifyChanged(fovea, ds.id);
        mount(out, h('div', { class: 'aip-msg ok' }, icon('checkCircle', 13), `Rejected ${fmt.num(res.rejected ?? 0)} proposals below ${min.toFixed(2)}.`));
        setTimeout(refreshBulk, 800);
      } catch (e) { mount(out, errLine(fovea, errorInfo(e).text)); }
    }
    async function rejectAll() {
      if (!(await ui.confirm({ title: 'Reject every pending proposal?', danger: true, okLabel: 'Reject all', message: `All pending proposals on ${fmt.num(n)} images are marked rejected. Label files are not touched; rejected proposals stay on record so the same box is not proposed again.` }))) return;
      try {
        const res = await fovea.api.post(`/api/datasets/${ds.id}/proposals/reject`, { filters: {}, kinds: PROPOSAL_KINDS });
        notifyChanged(fovea, ds.id);
        mount(out, h('div', { class: 'aip-msg ok' }, icon('checkCircle', 13), `Rejected ${fmt.num(res.rejected ?? 0)} proposals.`));
        setTimeout(refreshBulk, 800);
      } catch (e) { mount(out, errLine(fovea, errorInfo(e).text)); }
    }
  }
  refreshBulk();
  return () => { alive = false; if (form) form.destroy(); stopJobs(); started.forEach((b) => b.stop()); };
}

// ====================================================================== two-stage
function defaultTwoStage(C) {
  const P = C.P;
  const det = firstUsable(P, (p) => canDetect(p) && !isCloud(p)) || firstUsable(P, canDetect);
  // Classifier: a cloud model with a key, else a configured OpenAI-compatible server; an unconfigured local server
  // (default localhost:11434) may not be running, so it comes last.
  const score = (p) => (p.id === 'openai_compat' ? (p.config && p.config.base_url ? 2 : 0.5) : p.id === 'mock' ? 1 : isCloud(p) ? 3 : 1.5);
  const cls = ((P && P.providers) || []).filter((p) => canClassify(p) && providerState(P, p).ok).sort((a, b) => score(b) - score(a))[0] || null;
  // The cheapest recommended classifier model: a crop classification is a short, easy question.
  const cheap = cls ? (cls.models || []).filter((m) => m.recommended || m.default).sort((a, b) => ((a.price && a.price.in) ?? 1e9) - ((b.price && b.price.in) ?? 1e9))[0] || defaultModel(cls) : null;
  return {
    detector: { provider: det ? det.id : 'yolo', model: det ? (defaultModel(det) || {}).id : '', conf: 0.25, source_classes: ['person'] },
    classifier: { provider: cls ? cls.id : '', model: cheap ? cheap.id : '' },      // effort: the provider's Settings default
    classes: null, pad: 0.25,
  };
}
function secTwoStage(body, C, go) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt } = fovea;
  let ts = JSON.parse(JSON.stringify(C.prof.twoStage || defaultTwoStage(C)));
  let dirty = !C.prof.twoStage, alive = true, lastIds = null, estSeq = 0;
  const testing = latestOnly();
  const names = () => C.names;
  const detBox = h('div', { class: 'col gap-12' }), clsBox = h('div', { class: 'col gap-12' }), classesBox = h('div', { class: 'col gap-8' });
  const saveState = h('span', { class: 'hint' });
  const estEl = h('div', { class: 'small' });
  const results = h('div', { class: 'col gap-12' });
  const saveBtn = h('button', { class: 'btn btn-primary btn-sm', type: 'button', onClick: () => save() }, icon('save', 12), 'Save these defaults');
  const testBtn = h('button', { class: 'btn btn-sm', type: 'button', onClick: () => test(false) }, icon('eye', 12), 'Test on 4 labeled images');
  const shuffleBtn = h('button', { class: 'btn btn-sm btn-ghost', type: 'button', onClick: () => test(true), 'data-tip': 'Test on 4 other images' }, icon('shuffle', 12), 'Other images');
  const useBtn = h('button', { class: 'btn btn-sm', type: 'button', onClick: () => { save(); patchProfile(fovea, ds.id, { source: { pipeline: 'two_stage' } }); C.prof = profileOf(ds.id); saveState.textContent = 'Saved · Propose (I) in Annotate now uses two-stage'; } }, icon('pen', 12), 'Use in Annotate');
  const runBtn = h('button', { class: 'btn btn-sm', type: 'button', onClick: () => { save(); C.labelSrc = { pipeline: 'two_stage' }; go('label'); } }, icon('sparkles', 12), 'Run on images…');
  add(body, h('div', { class: 'ai-cols ai-cols-wide' },
    h('div', { class: 'col gap-16' },
      h('div', { class: 'callout small' }, icon('info', 14), h('span', 'A detector finds every person (tight boxes, cheap, local); a classifier looks at each box with some context and picks one of your classes — e.g. fall, standing or sitting. Boxes the classifier calls “none” are dropped; “uncertain” ones stay as low-confidence proposals.')),
      card(h, '1 · Find the objects', 'detector', detBox),
      card(h, '2 · Decide the class of each box', 'classifier', clsBox),
      card(h, '3 · Classes the classifier chooses from', null, classesBox),
      h('div', { class: 'row wrap gap-8' }, saveBtn, saveState, h('span', { class: 'spacer' }), useBtn, runBtn), estEl),
    h('div', { class: 'col gap-16' }, card(h, 'Test', 'nothing is stored', h('div', { class: 'row wrap gap-8 mb-12' }, testBtn, shuffleBtn), results))));

  // Every change is saved for this dataset right away (it is only a setup; nothing runs until you test or start).
  function changed() { save(); invalidateTest(); renderAll(); }
  /** Test results must match the setup on screen: a change drops the result shown and abandons a test still running. */
  function invalidateTest() {
    const running = testing.drop();
    if (running || results.querySelector('.ai-ts-sum')) mount(results, h('div', { class: 'hint' }, running ? 'The setup changed, so the test was stopped — run it again for this setup.' : 'The setup changed — test again to see how this setup does on the same images.'));
  }
  function save() {
    patchProfile(fovea, ds.id, { twoStage: JSON.parse(JSON.stringify(ts)) });
    C.prof = profileOf(ds.id);
    dirty = false; saveState.textContent = 'Setup saved for this dataset';
    saveBtn.classList.add('hidden');
  }
  function renderAll() { renderDet(); renderCls(); renderClasses(); estimate(); }
  function renderDet() {
    const dp = provOf(C.P, ts.detector.provider);
    const btn = srcButton(fovea, C, () => ({ provider: ts.detector.provider, model: ts.detector.model }), (s) => {
      const wasClosed = isClosedModel(dp, ts.detector.model);
      ts.detector.provider = s.provider; ts.detector.model = s.model;
      const np = provOf(C.P, s.provider);
      if (np && isClosedModel(np, s.model) !== wasClosed) ts.detector.source_classes = ['person'];
      changed();
    }, { allowTwoStage: false });
    const conf = h('input', { class: 'input input-sm', type: 'number', min: 0.05, max: 0.95, step: 0.05, value: ts.detector.conf, style: 'width:76px', onChange: (e) => { ts.detector.conf = Math.min(0.95, Math.max(0.05, Number(e.target.value) || 0.25)); changed(); } });
    const what = h('div', { class: 'col gap-4' });
    mount(detBox, h('div', { class: 'field' }, h('span', { class: 'label' }, 'Detector'), btn, srcNote(fovea, C, { provider: ts.detector.provider, model: ts.detector.model })),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Looks for'), what),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Minimum detector confidence'), h('div', { class: 'row' }, conf, h('span', { class: 'hint' }, 'Every box above this is sent to the classifier, so it also sets the cost.'))));
    const chosen = ts.detector.source_classes || [];
    if (dp && isClosedModel(dp, ts.detector.model)) {
      const draw = (mn) => {
        if (!alive) return;
        const all = Object.values((mn && mn.names) || {});
        mount(what, h('div', { class: 'aip-chips' }, chosen.map((n) => h('span', { class: 'aip-chip on' }, h('span', n),
          h('button', { class: 'aip-x', type: 'button', 'aria-label': `Remove ${n}`, onClick: () => { ts.detector.source_classes = chosen.filter((x) => x !== n); changed(); } }, icon('x', 10))))),
          h('select', { class: 'select select-sm', style: 'max-width:240px', 'aria-label': 'Add a detector class', onChange: (e) => { if (e.target.value) { ts.detector.source_classes = [...chosen, e.target.value]; changed(); } } },
            h('option', { value: '' }, all.length ? '+ Add a detector class…' : 'Loading classes…'), all.filter((n) => !chosen.includes(n)).map((n) => h('option', { value: n }, n))),
          !chosen.length ? h('div', { class: 'aip-hint warn' }, 'Pick at least one detector class (usually “person”).') : null);
      };
      const peek = peekModelClasses(dp, ts.detector.model);
      if (peek) draw(peek); else { draw(null); modelClassNames(fovea, dp, ts.detector.model).then(draw); }
    } else {
      mount(what, h('input', { class: 'input input-sm', value: chosen.join(', '), placeholder: 'person', 'aria-label': 'What the detector looks for',
        onChange: (e) => { ts.detector.source_classes = e.target.value.split(',').map((s) => s.trim()).filter(Boolean); changed(); } }),
        h('div', { class: 'hint' }, 'Comma-separated text prompts, e.g. “person”.'));
    }
  }
  function renderCls() {
    const cp = provOf(C.P, ts.classifier.provider);
    const btn = srcButton(fovea, C, () => ({ provider: ts.classifier.provider, model: ts.classifier.model }), (s) => { ts.classifier.provider = s.provider; ts.classifier.model = s.model; changed(); }, { purpose: 'classify', allowTwoStage: false });
    const padSeg = ui.seg([{ value: 0.15, label: '15%' }, { value: 0.25, label: '25%' }, { value: 0.5, label: '50%' }], ts.pad ?? 0.25, (v) => { ts.pad = v; changed(); });
    padSeg.classList.add('ai-seg');
    const anyClassifier = (C.P.providers || []).some((p) => canClassify(p));
    mount(clsBox, 
      anyClassifier ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Classifier'), btn, srcNote(fovea, C, { provider: ts.classifier.provider, model: ts.classifier.model }))
        : h('div', { class: 'aip-hint warn' }, 'No provider here can classify regions. Add a cloud key or an OpenAI-compatible server in Settings.'),
      cp && isCloud(cp) ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Effort'), effortSeg(fovea, ts.classifier.effort || defaultEffort(cp), (v) => { ts.classifier.effort = v; changed(); })) : null,
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Context around each box'), padSeg, h('div', { class: 'hint' }, 'More context helps tell lying from sitting (floor, bed, chair) but costs a little more.')));
  }
  function renderClasses() {
    const nm = names();
    if (!nm.length) { mount(classesBox, h('div', { class: 'aip-hint warn' }, 'This dataset has no classes yet — the classifier needs classes to choose from. ', addClassesLink(fovea, ds.id))); return; }
    const sel = ts.classes ? new Set(ts.classes) : null;
    const counts = (C.stats && C.stats.classes) || [];
    const rows = nm.map((_, c) => {
      const ci = classInfo(nm, C.cards, c);
      const on = !sel || sel.has(c);
      const n = (counts.find((x) => x.id === c || x.cls === c) || {}).boxes;
      const promptIn = h('input', { class: 'input input-sm', value: ci.prompt, placeholder: ci.placeholder ? 'Describe it, e.g. person lying on the ground' : `e.g. ${ci.name}`, 'aria-label': `Prompt for ${ci.name}`, disabled: cardsMissing(ds.id),
        onChange: async (e) => {
          try { await saveCard(fovea, ds.id, c, { prompt: e.target.value.trim() }); C.cards = await loadCards(fovea, ds.id); invalidateTest(); renderClasses(); }
          catch (err) { ui.toast(`Could not save the prompt: ${scrub(err.message)}`, { type: 'error' }); }
        } });
      return h('div', { class: `ai-tsc ${on ? '' : 'off'}` },
        h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: on, onChange: (e) => {
          const cur = new Set(ts.classes || nm.map((_, i) => i));
          if (e.target.checked) cur.add(c); else cur.delete(c);
          const arr = [...cur].sort((a, b) => a - b);
          ts.classes = arr.length === nm.length ? null : arr; changed();
        } }), h('i', { class: 'ai-sw', style: `background:${fovea.colors.classColor(c)}` }), h('span', { class: 'truncate', title: ci.name }, `${c} ${ci.name}`)),
        promptIn, h('span', { class: 'xs faint nowrap' }, n != null ? fmt.plural(n, 'box', 'boxes') : ''));
    });
    const undescribed = nm.map((_, c) => classInfo(nm, C.cards, c)).filter((ci) => ci.placeholder && (!sel || sel.has(ci.id)));
    mount(classesBox, 
      undescribed.length ? h('div', { class: 'callout warn small' }, icon('alert', 14), h('span', `${undescribed.length} chosen class${undescribed.length === 1 ? ' has' : 'es have'} only a placeholder name. The classifier sees just “${undescribed[0].name}”, which means nothing to it — describe each class below (it is saved to the class card).`)) : null,
      h('div', { class: 'ai-tsc-list' }, rows),
      h('div', { class: 'row wrap gap-4' }, h('button', { class: 'aip-link', type: 'button', onClick: () => { ts.classes = null; changed(); } }, 'All classes'),
        h('span', { class: 'faint' }, '·'), h('button', { class: 'aip-link', type: 'button', onClick: () => go('cards') }, 'Full class cards (definition, include / exclude, examples)')));
  }
  async function estimate() {
    const seq = ++estSeq;
    const req = await requestFor(fovea, C.P, { ...C.prof, twoStage: ts }, { pipeline: 'two_stage' }, names(), { dsId: ds.id });
    if (!alive || seq !== estSeq) return;
    if (!req.ok) { mount(estEl, h('span', { class: 'aip-bad' }, req.reason)); testBtn.disabled = shuffleBtn.disabled = runBtn.disabled = useBtn.disabled = true; return; }
    testBtn.disabled = shuffleBtn.disabled = runBtn.disabled = useBtn.disabled = false;
    try {
      const est = await fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, count: 100, ...req.body });
      if (!alive || seq !== estSeq) return;
      const ck = costKind(est, req.cloud);
      mount(estEl, h('span', { class: 'muted' }, 'Per 100 images: '), h('strong', ck === 'free' ? 'free' : ck === 'unknown' ? 'cost unknown' : usdRange(est.usd)),
        ck === 'unknown' ? h('span', { class: 'muted' }, ` (no price on file for ${unpricedText(est, 'the classifier')})`) : null,
        est.seconds_per_image ? h('span', { class: 'muted' }, ` · about ${duration(est.seconds_per_image * 100)}`) : null,
        (est.assumptions || []).length ? h('div', { class: 'xs faint' }, 'Assumes ' + est.assumptions.join('; ')) : null);
    } catch (e) { if (alive && seq === estSeq) mount(estEl, h('span', { class: 'hint' }, 'No estimate available.')); }
  }
  async function test(shuffle) {
    const run = testing.begin();
    const stale = () => !alive || !run.current();
    testBtn.disabled = shuffleBtn.disabled = true;
    const req = await requestFor(fovea, C.P, { ...C.prof, twoStage: ts }, { pipeline: 'two_stage' }, names(), { dsId: ds.id });
    if (stale()) return;
    if (!req.ok) { run.end(); testBtn.disabled = shuffleBtn.disabled = false; mount(results, errLine(fovea, req.reason)); return; }
    mount(results, h('div', { class: 'row small muted' }, ui.spinner(), 'Running on 4 images… ', ui.elapsedClock()));
    try {
      if (shuffle || !lastIds) lastIds = await sampleIds(fovea, ds.id, { labeled: '1' }, 4);
      if (stale()) return;
      if (!lastIds.length) { mount(results, h('div', { class: 'hint' }, 'No labeled images to test on.')); return; }
      const t0 = performance.now();
      const ids = lastIds;
      const [res, items] = await Promise.all([fovea.api.post('/api/ai/preview', { dataset_id: ds.id, image_ids: ids, ...req.body }, { signal: run.signal }), imagesById(fovea, ds.id, ids)]);
      if (stale()) return;
      const out = previewItems(res, ids);
      let agree = 0, differ = 0, extra = 0, missed = 0, cost = 0, nPred = 0;
      const nm = names();
      const tiles = out.map((r) => {
        const it = items.find((x) => x.id === r.image_id) || { id: r.image_id, boxes: [] };
        const gt = it.boxes || [];
        const used = new Set();
        const rows = r.preds.map((p) => {
          let best = -1, bi = 0;
          gt.forEach((g, gi) => { const v = iou([p.xc, p.yc, p.w, p.h], g.slice(1)); if (v > bi) { bi = v; best = gi; } });
          let verdict;
          if (best >= 0 && bi >= 0.5) { used.add(best); if (gt[best][0] === p.cls) { agree++; verdict = h('span', { class: 'ai-ok' }, '✓ matches your label'); } else { differ++; verdict = h('span', { class: 'ai-diff' }, `your label: ${displayName(C, gt[best][0])}`); } }
          else { extra++; verdict = h('span', { class: 'faint' }, 'not in your labels'); }
          return h('li', { class: 'ai-tsr' }, h('i', { class: 'ai-sw', style: `background:${fovea.colors.classColor(p.cls)}` }),
            h('div', { class: 'grow' }, h('div', h('strong', displayName(C, p.cls)), p.conf != null ? h('span', { class: 'faint' }, ` · det ${Number(p.conf).toFixed(2)}`) : null, ' · ', verdict),
              p.reason ? h('div', { class: 'xs muted' }, `“${scrub(p.reason)}”`) : null));
        });
        missed += gt.filter((_, gi) => !used.has(gi)).length;
        nPred += r.preds.length;
        if (r.usage && r.usage.cost_usd) cost += r.usage.cost_usd;
        return h('figure', { class: 'ai-ts-item' }, overlayThumb(fovea, it, predOverlay(fovea, { preds: r.preds, gt, names: nm, labelOf: (c) => displayName(C, c) }), { size: 512 }),
          // The path gets its own line: beside the time, cost and link its folder had no room in a ~200 px figure.
          h('figcaption', pathLine(fovea, it, { cls: 'xs faint' }),
            h('div', { class: 'row xs faint' }, h('span', { class: 'grow nowrap' }, [r.latency_ms ? `${(r.latency_ms / 1000).toFixed(1)} s` : '', r.usage && r.usage.cost_usd ? usd(r.usage.cost_usd) : ''].filter(Boolean).join(' · ')),
              fovea.link(`/d/${ds.id}/annotate?img=${it.id}`, { class: 'aip-link nowrap' }, 'Annotate')),
            r.error ? errLine(fovea, r.error) : rows.length ? h('ul', { class: 'ai-tsr-list' }, rows) : h('div', { class: 'hint' }, 'Nothing detected.'),
            (r.warnings || []).length ? h('div', { class: 'xs aip-warn' }, r.warnings.map(scrub).join('; ')) : null));
      });
      const matched = agree + differ;
      const secs = (performance.now() - t0) / 1000 / Math.max(1, out.length);
      mount(results, 
        h('div', { class: 'ai-ts-sum' },
          h('div', h('span', { class: 'ai-big' }, matched ? `${Math.round(100 * agree / matched)}%` : '–'), h('span', { class: 'muted' }, ' class agreement on boxes that match your labels')),
          h('div', { class: 'small muted' }, `${agree} agree · ${differ} differ · ${extra} not in your labels · ${missed} of your boxes not found · ${fmt.plural(nPred, 'box', 'boxes')}`),
          h('div', { class: 'small muted' }, `${cost ? `${usd(cost / Math.max(1, out.length))} per image · ` : ''}${secs.toFixed(1)} s per image`)),
        h('div', { class: 'ai-ts-grid' }, tiles),
        h('div', { class: 'xs faint' }, 'Dashed: two-stage result · thin solid: your labels. “Differ” may mean the label is wrong — check them in Annotate.'));
    } catch (e) {
      if (stale() || e.name === 'AbortError') return;
      mount(results, errLine(fovea, e.status === 404 ? 'Preview is not available on this server.' : `Test failed: ${errorInfo(e).text}`));
    } finally {
      if (run.current()) { run.end(); testBtn.disabled = shuffleBtn.disabled = false; }
    }
  }
  mount(results, h('div', { class: 'hint' }, 'Runs this setup on 4 random labeled images and compares every box with your labels: same class, a different class (a candidate label error) or not labeled at all. Nothing is stored; cloud calls are billed.'));
  saveState.textContent = dirty ? 'Suggested defaults — not saved yet' : 'Setup saved for this dataset';
  saveBtn.classList.toggle('hidden', !dirty);
  renderAll();
  issueLabels(fovea, C).then(() => { if (alive) renderClasses(); });
  const offCards = fovea.bus.on('ai:cards', (e) => { if (e && e.dataset_id === ds.id && alive) renderClasses(); });
  return () => { alive = false; testing.drop(); offCards(); };
}

// ====================================================================== label audit
function secAudit(body, C, go) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt, link } = fovea;
  const summaryEl = h('div', { class: 'col gap-16' });
  const formEl = h('div', { class: 'col gap-12' });
  const jobsEl = h('div', { class: 'col gap-8' });
  let alive = true, estSeq = 0;
  const saved = C.prof.audit || {};
  let src = saved.source || defaultSource(C.P, C.prof);
  const d0 = describeSource(C.P, C.prof, src);
  let mode = saved.mode || (d0.p && isClosedModel(d0.p, d0.model) ? 'geometry' : 'full');
  let floor = saved.conf_floor ?? 0.05, split = saved.split || '', limit = saved.limit || 0;
  let family = saved.family || null;             // {model label: [dataset classes]}
  let committee = saved.committee || null;        // {provider, model}
  const estEl = h('div', { class: 'small' });
  const errEl = h('div', { class: 'aib-err' });
  let starting = false;            // claimed before the first await: one audit per click, however fast the clicks
  const startBtn = h('button', { class: 'btn btn-primary', type: 'button', onClick: (e) => { if (!(e && e.detail > 1)) start(); } }, icon('fileSearch', 13), 'Start audit');
  const stopJobs = activeJobsInto(fovea, C, jobsEl, { onEnd: () => alive && loadSummary() });
  add(body, h('div', { class: 'ai-cols' }, h('div', { class: 'col gap-16' }, summaryEl, jobsEl), h('div', { class: 'col gap-16' }, card(h, 'Run a label audit', 'compares a model with your labels', formEl))));

  async function loadSummary() {
    mount(summaryEl, loadingLine(fovea, 'Loading the last audit…'));
    await issueLabels(fovea, C);
    let s;
    try { s = await fovea.api.get(`/api/datasets/${ds.id}/audit/summary`); }
    catch (e) { if (alive) mount(summaryEl, card(h, 'Last audit', null, h('div', { class: 'hint' }, e.status === 404 ? 'Audit results are not available on this server yet.' : scrub(e.message)))); return; }
    if (!alive) return;
    if (!s || !s.run) {
      mount(summaryEl, card(h, 'Label audit', null, h('div', { class: 'col gap-8' },
        h('p', { class: 'muted', style: 'margin:0' }, 'An audit runs a model over images you already labeled and ranks the boxes most likely to be wrong: objects you missed, boxes that are loose, classes that look swapped and boxes with nothing under them.'),
        h('p', { class: 'muted', style: 'margin:0' }, 'Findings become suspects you review in Annotate (one key fixes or dismisses each). Label files are never changed by the audit itself.'))));
      return;
    }
    const counts = s.counts || {};
    const run = s.run;
    const params = run.params || {};
    const seen = new Set();          // one row per image, even if the summary lists an image twice
    const top = (s.top || []).filter((t) => t && !seen.has(t.image_id) && seen.add(t.image_id));
    mount(summaryEl, 
      card(h, 'Findings to review', `last audit: ${runSource(C, run)} · ${when(run.created_at)}${run.images ? ` · ${fmt.num(run.images)} images` : ''}${params.mode ? ` · ${params.mode}` : ''}${run.status && run.status !== 'done' ? ` · ${run.status}` : ''}`,
        h('div', { class: 'stat-grid ai-audit-tiles mb-12' }, AUDIT_KINDS.map((k) => ui.statTile({ label: kindLabel(C, k), value: fmt.num(counts[k] || 0), sub: KIND_HINT[k], tone: counts[k] ? 'warn' : null,
          onClick: counts[k] ? () => fovea.router.navigate(auditLink(ds.id, 'explore', { issue: `ai_${k}` })) : null }))),
        h('div', { class: 'row wrap gap-8' }, h('span', h('span', { class: 'ai-big' }, fmt.num(s.images_flagged || 0)), h('span', { class: 'muted' }, ` image${s.images_flagged === 1 ? '' : 's'} with open findings${s.images_audited ? ` · ${fmt.num(s.images_audited)} audited` : ''}`)), h('span', { class: 'spacer' }),
          s.images_flagged ? link(auditLink(ds.id, 'annotate', { ai: 'audit' }), { class: 'btn btn-sm btn-primary' }, icon('pen', 12), 'Review in Annotate, worst first') : null,
          s.images_flagged ? link(auditLink(ds.id, 'explore', { ai: 'audit' }), { class: 'btn btn-sm' }, icon('grid', 12), 'Explore') : null,
          s.images_flagged ? h('button', { class: 'btn btn-sm btn-ghost', type: 'button', onClick: () => clearSuspects(s.images_flagged) }, icon('x', 12), 'Dismiss all') : null)),
      top.length ? card(h, 'Most likely wrong', 'highest audit score first', h('div', { class: 'ai-top' }, top.slice(0, 24).map((t) => {
        const kinds = Object.entries(t.counts || {}).filter(([, n]) => n);
        return link(auditLink(ds.id, 'annotate', { img: t.image_id, ai: 'audit' }), { class: 'ai-top-row', title: t.rel_path },
          h('img', { src: fovea.api.thumbUrl(t.image_id, 128), alt: '', loading: 'lazy', decoding: 'async' }),
          // The whole relative path: the same frame name can sit in two split folders (test/, test-1854-1860-포함/).
          h('div', { class: 'grow' }, h('div', { class: 'small ai-top-path' }, relPath(fovea, t.rel_path, { tail: 30 })),
            h('div', { class: 'row wrap gap-4' }, kinds.map(([k, n]) => h('span', { class: 'chip warn' }, kindCount(k, n))))),
          h('div', { class: 'ai-score' }, h('div', { class: 'bar' }, h('i', { style: `width:${Math.round(100 * Math.min(1, t.score || 0))}%` })), h('span', { class: 'xs mono' }, (t.score || 0).toFixed(2))));
      }))) : null);
  }
  async function clearSuspects(n) {
    if (!(await ui.confirm({ title: 'Dismiss every audit finding?', okLabel: 'Dismiss all', message: `The suspects on ${fmt.num(n)} images are marked rejected. Your labels are not changed.` }))) return;
    try { await fovea.api.post(`/api/datasets/${ds.id}/proposals/reject`, { filters: {}, kinds: AUDIT_KINDS }); notifyChanged(fovea, ds.id); loadSummary(); }
    catch (e) { ui.toast(scrub(e.message), { type: 'error' }); }
  }

  function remember() { patchProfile(fovea, ds.id, { audit: { source: src, mode, conf_floor: floor, split, limit, family, committee } }); C.prof = profileOf(ds.id); }
  /** In geometry mode a two-stage source audits with its detector (the classifier adds nothing to geometry). */
  function geoSource() {
    if (!src || !src.pipeline) return src;
    const ts = C.prof.twoStage;
    return ts ? { provider: ts.detector.provider, model: ts.detector.model } : src;
  }
  function renderForm() {
    const d = describeSource(C.P, C.prof, src);
    const btn = srcButton(fovea, C, () => src, (s) => {
      src = s; family = null;
      const nd = describeSource(C.P, C.prof, s);
      mode = nd.p && isClosedModel(nd.p, nd.model) ? 'geometry' : 'full';
      remember(); renderForm();
    });
    const modeSeg = ui.seg([{ value: 'geometry', label: 'Geometry' }, { value: 'full', label: 'Geometry + classes' }], mode, (v) => { mode = v; remember(); renderForm(); });
    modeSeg.classList.add('ai-seg');
    const modeHint = mode === 'geometry' ? 'Missing, loose and unsupported boxes, by model class family. Works with a stock person detector even when your classes are states like fall / standing.'
      : 'Also flags boxes whose class looks wrong. Needs a source that predicts your own classes — two-stage, an open-vocabulary model, a cloud model, or a closed model whose classes map onto yours.';
    const famEl = h('div', { class: 'col ai-gap6' });
    if (mode === 'geometry' && d.ok) renderFamily(famEl);
    const floorIn = h('input', { class: 'input input-sm', type: 'number', min: 0.01, max: 0.5, step: 0.01, value: floor, style: 'width:76px', onChange: (e) => { floor = Math.min(0.5, Math.max(0.01, Number(e.target.value) || 0.05)); remember(); estimate(); } });
    const limitIn = h('input', { class: 'input input-sm', type: 'number', min: 0, placeholder: 'all', value: limit || '', style: 'width:90px', onChange: (e) => { limit = Math.max(0, Number(e.target.value) || 0); remember(); estimate(); } });
    const seg = splitSeg(fovea, ds, split, (v) => { split = v; remember(); estimate(); });
    const comBtn = committee ? srcButton(fovea, C, () => committee, (s) => { committee = { provider: s.provider, model: s.model }; remember(); estimate(); }, { allowTwoStage: false }) : null;
    mount(formEl,
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model'), btn, srcNote(fovea, C, src),
        h('div', { class: 'hint' }, 'Tip: a model family different from the one that produced the labels avoids confirming its own mistakes.')),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Checks'), modeSeg, h('div', { class: 'hint' }, modeHint)),
      famEl,
      h('div', { class: 'fields cols-2' },
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Prediction floor'), floorIn, h('div', { class: 'hint' }, 'Low keeps weak evidence for “unsupported box” checks.')),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Random sample'), h('div', { class: 'row' }, limitIn, h('span', { class: 'hint' }, 'labeled images')))),
      seg ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Split'), seg) : null,
      h('div', { class: 'field' }, h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: !!committee, onChange: (e) => {
        if (e.target.checked) {
          const cur = describeSource(C.P, C.prof, geoSource());
          const alt = (C.P.providers || []).find((p) => canDetect(p) && providerState(C.P, p).ok && p.id !== (cur.p && cur.p.id)) || cur.p;
          committee = alt ? { provider: alt.id, model: (defaultModel(alt) || {}).id } : null;
        } else committee = null;
        remember(); renderForm();
      } }), 'Confirm with a second model (flag only what both find)'), comBtn, committee ? srcNote(fovea, C, committee) : null),
      estEl, errEl, h('div', { class: 'row' }, startBtn));
    estimate();
  }
  function renderFamily(el) {
    const nm = C.names;
    const g = describeSource(C.P, C.prof, geoSource());
    const closed = g.p && isClosedModel(g.p, g.model);
    const mn = closed ? peekModelClasses(g.p, g.model) : null;
    if (!family || !Object.keys(family).length) {
      const det = src && src.pipeline && C.prof.twoStage ? (C.prof.twoStage.detector.source_classes || ['person']) : ['person'];
      family = Object.fromEntries(det.map((k) => [k, nm.map((_, i) => i)]));
    }
    const listId = `ai-fam-${ds.id}`;
    const datalist = closed && mn ? h('datalist', { id: listId }, Object.values(mn.names).map((n) => h('option', { value: n }))) : null;
    const rename = (from, to) => {
      to = to.trim();
      if (!to || to === from || family[to]) return renderForm();
      const next = {};
      Object.entries(family).forEach(([k, v]) => { next[k === from ? to : k] = v; });
      family = next; remember(); renderForm();
    };
    const rows = Object.entries(family).map(([lab, cls]) => h('div', { class: 'ai-fam' },
      h('div', { class: 'row' },
        h('input', { class: 'input input-sm ai-fam-lab', value: lab, list: datalist ? listId : null, 'aria-label': 'Model class or prompt', onChange: (e) => rename(lab, e.target.value) }),
        h('span', { class: 'muted small' }, 'covers'), h('span', { class: 'spacer' }),
        h('button', { class: 'aip-link', type: 'button', onClick: () => { family[lab] = nm.map((_, i) => i); remember(); renderForm(); } }, 'all'),
        Object.keys(family).length > 1 ? h('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', 'aria-label': `Remove ${lab}`, onClick: () => { delete family[lab]; remember(); renderForm(); } }, icon('x', 11)) : null),
      h('div', { class: 'aip-chips' }, nm.map((n, c) => h('button', { class: `aip-chip ${cls.includes(c) ? 'on' : ''}`, type: 'button', 'aria-pressed': String(cls.includes(c)),
        onClick: () => { const s2 = new Set(family[lab]); s2.has(c) ? s2.delete(c) : s2.add(c); family[lab] = [...s2].sort((a, b) => a - b); remember(); renderForm(); } },
        cls.includes(c) ? icon('check', 10) : null, h('i', { style: `background:${fovea.colors.classColor(c)}` }), h('span', { class: 'truncate' }, `${c} ${displayName(C, c)}`)))),
      !cls.length ? h('div', { class: 'aip-hint warn' }, 'Pick at least one of your classes.') : null));
    mount(el, h('span', { class: 'ai-lbl' }, 'Class families'),
      h('div', { class: 'hint' }, `What the model looks for (${closed ? `${modelName(g.p, g.model)} classes` : 'text prompts'}) and which of your classes each one stands for — e.g. “person” covers fall, standing and sitting. Boxes are compared within a family; classes inside it are not checked in this mode.`),
      datalist, ...rows,
      h('button', { class: 'btn btn-sm btn-ghost', type: 'button', style: 'align-self:flex-start', onClick: () => { let k = 'object', i = 2; while (family[k]) k = `object ${i++}`; family[k] = []; remember(); renderForm(); } }, icon('plus', 12), 'Add a family'));
  }
  function scopeRef() {
    const f = { labeled: '1', ...(split ? { split } : {}) };
    return f;
  }
  async function bodyFor() {
    const s0 = mode === 'geometry' ? geoSource() : src;
    const req = await requestFor(fovea, C.P, C.prof, s0, C.names, { conf: floor, requireMap: mode === 'full', dsId: ds.id });
    if (!req.ok) return req;
    const { conf, classes, ...rest } = req.body;
    const b = { ...rest, options: { ...(rest.options || {}) }, conf_floor: floor, mode };
    if (mode === 'geometry') {
      delete b.options.class_map;              // the family names drive the detector in geometry mode
      const fam = Object.fromEntries(Object.entries(family || {}).filter(([k, v]) => k.trim() && v.length));
      if (!Object.keys(fam).length) return { ok: false, reason: 'Give at least one class family with one of your classes' };
      b.class_family = fam;
    }
    if (committee && committee.provider) b.committee = [{ provider: committee.provider, model: committee.model }];
    return { ok: true, body: b, cloud: req.cloud, fix: req.fix };
  }
  async function estimate() {
    const seq = ++estSeq;
    const r = await bodyFor();
    if (!alive || seq !== estSeq) return;
    if (!r.ok) {
      mount(estEl, h('span', { class: 'aip-bad' }, r.reason), r.fix === 'map' ? [' ', h('button', { class: 'aip-link', type: 'button', onClick: () => openClassMap(fovea, { ds, P: C.P, src, cards: C.cards, onDone: () => { C.prof = profileOf(ds.id); estimate(); } }) }, 'Map classes…')] : null);
      startBtn.disabled = true; return;
    }
    startBtn.disabled = starting;
    try {
      const n = await countImages(fovea, ds.id, scopeRef());
      const count = limit ? Math.min(limit, n) : n;
      const est = await fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, count, ...r.body });
      if (!alive || seq !== estSeq) return;
      const ck = costKind(est, r.cloud);
      mount(estEl, h('strong', `${fmt.num(count)} labeled images`), ' · ', ck === 'free' ? 'free' : ck === 'unknown' ? h('strong', 'cost unknown') : h('strong', usdRange(est.usd)),
        ck === 'unknown' ? h('span', { class: 'muted' }, ` (no price on file for ${unpricedText(est, describeSource(C.P, C.prof, src).label)})`) : null,
        // ai-core hook (r4): an audit asks one image at a time — per-request time, not a batch's parallel throughput
        (est.seconds_per_request ?? est.seconds_per_image) ? ` · about ${duration((est.seconds_per_request ?? est.seconds_per_image) * count)}` : '');
    } catch (e) { if (alive && seq === estSeq) mount(estEl, h('span', { class: 'hint' }, 'No estimate available.')); }
  }
  async function start() {
    if (starting) return;
    starting = true; startBtn.disabled = true;
    try { await startAudit(); } finally { starting = false; }
  }
  async function startAudit() {
    mount(errEl);
    const r = await bodyFor();
    if (!r.ok) { mount(errEl, r.reason); startBtn.disabled = false; return; }
    try {
      const payload = { dataset_id: ds.id, ...r.body };
      if (limit) payload.image_ids = await sampleIds(fovea, ds.id, scopeRef(), limit, { seed: newSeed() });
      else payload.filters = scopeRef();
      // An audit has no run budget: with no 30-day spend cap either, nothing stops a cloud one before it ends. Ask
      // first when the estimate is large (or the cost cannot be known).
      const s0 = mode === 'geometry' ? geoSource() : src;
      if (r.cloud && !sourceCap(C.P, C.prof, s0)) {
        const est = await fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, ...(payload.image_ids ? { image_ids: payload.image_ids } : { filters: payload.filters }), ...r.body }).catch(() => null);
        if (!alive) return;
        const chk = spendCheck(est);
        if (chk.ask && !(await confirmSpend(fovea, chk, { label: describeSource(C.P, C.prof, s0).label, what: 'label audit', canCap: false, est }))) { startBtn.disabled = false; return; }
      }
      const res = await fovea.api.post('/api/ai/audit', payload);
      remember();
      jobsEl.prepend(jobBox(fovea, { job: res.job, title: 'Label audit', onEnd: () => { if (alive) { loadSummary(); startBtn.disabled = false; } },
        actions: () => [link(auditLink(ds.id, 'annotate', { ai: 'audit' }), { class: 'btn btn-sm btn-primary' }, icon('pen', 12), 'Review suspects')] }).el);
    } catch (e) { mount(errEl, e.status === 404 ? 'Label audit is not available on this server yet.' : `Could not start: ${errorInfo(e).text}`); startBtn.disabled = false; }
  }
  loadSummary();
  renderForm();
  return () => { alive = false; stopJobs(); };
}

// ====================================================================== benchmark
function secBenchmark(body, C) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt } = fovea;
  const saved = C.prof.bench || {};
  let configs = (saved.configs && saved.configs.length ? saved.configs : defaultConfigs()).slice(0, 4);
  let sample = saved.sample || 30, iouT = saved.iou || 0.5, split = saved.split ?? (splitsOf(ds).includes('val') ? 'val' : '');
  let alive = true, estSeq = 0;
  const cfgEl = h('div', { class: 'col gap-8' });
  const estEl = h('div', { class: 'small' });
  const errEl = h('div', { class: 'aib-err' });
  const resEl = h('div', { class: 'col gap-12' });
  const jobsEl = h('div', { class: 'col gap-8' });
  let starting = false;            // claimed before the first await: one benchmark per click
  const startBtn = h('button', { class: 'btn btn-primary', type: 'button', onClick: (e) => { if (!(e && e.detail > 1)) start(); } }, icon('barChart', 13), 'Run benchmark');
  const sampleIn = h('input', { class: 'input input-sm', type: 'number', min: 5, max: 500, value: sample, style: 'width:80px', onChange: (e) => { sample = Math.max(5, Math.min(500, Number(e.target.value) || 30)); e.target.value = sample; remember(); estimate(); } });
  const iouSeg = ui.seg([{ value: 0.5, label: '0.50' }, { value: 0.75, label: '0.75' }], iouT, (v) => { iouT = v; remember(); });
  iouSeg.classList.add('ai-seg');
  const seg = splitSeg(fovea, ds, split, (v) => { split = v; remember(); estimate(); });
  const stopJobs = activeJobsInto(fovea, C, jobsEl, { onEnd: () => alive && loadLast() });
  add(body, h('div', { class: 'col gap-16' },
    h('div', { class: 'ai-narrow col gap-16' }, card(h, 'Compare setups on your labels', 'same images for every setup',
      h('div', { class: 'col gap-12' },
        h('div', { class: 'hint' }, 'Each setup runs on the same random sample of labeled images; its boxes are scored against your labels (precision, recall, F1, AP@IoU when it gives confidences), with speed and cost per image.'),
        cfgEl,
        h('div', { class: 'ai-fields-row' },
          h('div', { class: 'field' }, h('span', { class: 'label' }, 'Labeled images'), sampleIn),
          h('div', { class: 'field' }, h('span', { class: 'label' }, 'Match at IoU'), iouSeg),
          seg ? h('div', { class: 'field' }, h('span', { class: 'label' }, 'Split'), seg) : null),
        estEl, errEl, h('div', { class: 'row' }, startBtn))), jobsEl),
    h('div', { class: 'col gap-16' }, resEl)));

  function defaultConfigs() {
    const out = [];
    const s = defaultSource(C.P, C.prof);
    if (s) out.push(s);
    const ye = (C.P.providers || []).find((p) => p.id === 'yoloe' && providerState(C.P, p).ok);
    if (ye && !out.some((x) => x.provider === 'yoloe')) out.push({ provider: 'yoloe', model: (defaultModel(ye) || {}).id });
    if (C.prof.twoStage && !out.some((x) => x.pipeline)) out.push({ pipeline: 'two_stage' });
    if (out.length < 2) {
      const other = (C.P.providers || []).find((p) => canDetect(p) && providerState(C.P, p).ok && !out.some((x) => x.provider === p.id && !x.pipeline));
      if (other) out.push({ provider: other.id, model: (defaultModel(other) || {}).id });
    }
    return out;
  }
  function remember() { patchProfile(fovea, ds.id, { bench: { configs, sample, iou: iouT, split } }); C.prof = profileOf(ds.id); }
  function renderConfigs() {
    mount(cfgEl, ...configs.map((c, i) => {
      const d = describeSource(C.P, C.prof, c);
      return h('div', { class: 'ai-cfg' }, h('span', { class: 'ai-cfg-n' }, String.fromCharCode(65 + i)),
        srcButton(fovea, C, () => configs[i], (s) => { configs[i] = s; remember(); renderConfigs(); }),
        d.ok && d.cloud && !d.twoStage ? effortSeg(fovea, effortOf(C.P, c), (v) => { configs[i] = { ...configs[i], effort: v }; remember(); estimate(); }) : null,
        !d.ok ? h('span', { class: 'xs aip-bad' }, d.reason) : null,
        h('span', { class: 'spacer' }),
        configs.length > 1 ? h('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', 'aria-label': `Remove setup ${String.fromCharCode(65 + i)}`, onClick: () => { configs.splice(i, 1); remember(); renderConfigs(); } }, icon('x', 12)) : null);
    }), configs.length < 4 ? h('button', { class: 'btn btn-sm btn-ghost', type: 'button', style: 'align-self:flex-start', onClick: () => {
      const s = defaultSource(C.P, C.prof) || { provider: 'yolo' };
      configs.push({ ...s }); remember(); renderConfigs();
    } }, icon('plus', 12), 'Add a setup') : null);
    estimate();
  }
  async function bodies() {
    const out = [];
    for (const c of configs) {
      const r = await requestFor(fovea, C.P, C.prof, c, C.names, { conf: 0.25, dsId: ds.id });
      if (!r.ok) return { ok: false, reason: `${describeSource(C.P, C.prof, c).label}: ${r.reason}`, fix: r.fix, src: c };
      const d = describeSource(C.P, C.prof, c);
      out.push({ ...r.body, label: d.twoStage ? `Two-stage (${d.sub})` : `${d.label}${d.cloud ? ` · ${effortLabel(effortOf(C.P, c)).toLowerCase()}` : ''}`, cloud: r.cloud, src: c });
    }
    return { ok: true, configs: out };
  }
  const filt = () => ({ labeled: '1', ...(split ? { split } : {}) });
  async function estimate() {
    const seq = ++estSeq;
    const b = await bodies();
    if (!alive || seq !== estSeq) return;
    if (!b.ok) {
      mount(estEl, h('span', { class: 'aip-bad' }, b.reason), b.fix === 'map' ? [' ', h('button', { class: 'aip-link', type: 'button', onClick: () => openClassMap(fovea, { ds, P: C.P, src: b.src, cards: C.cards, onDone: () => { C.prof = profileOf(ds.id); estimate(); } }) }, 'Map classes…')] : null);
      startBtn.disabled = true; return;
    }
    if (b.configs.length < 1) { mount(estEl, h('span', { class: 'aip-bad' }, 'Add at least one setup.')); startBtn.disabled = true; return; }
    startBtn.disabled = starting;
    try {
      const avail = await countImages(fovea, ds.id, filt());
      const n = Math.min(sample, avail);
      const ests = await Promise.all(b.configs.map(({ label, cloud, src, ...body }) => fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, count: n, ...body }).catch(() => null)));
      if (!alive || seq !== estSeq) return;
      const lo = ests.reduce((a, e) => a + ((e && e.usd && e.usd[0]) || 0), 0), hi = ests.reduce((a, e) => a + ((e && e.usd && e.usd[1]) || 0), 0);
      // ai-core hook (r4): a benchmark asks one image at a time — per-request time, not a batch's parallel throughput
      const secs = ests.reduce((a, e) => a + ((e && (e.seconds_per_request ?? e.seconds_per_image)) || 0) * n, 0);
      // A cloud setup with no price on file has an unknown cost: never folded into the total as $0, never "free".
      const unknown = b.configs.filter((c, i) => ests[i] && costKind(ests[i], c.cloud) === 'unknown').map((c) => c.label);
      const price = hi ? h('strong', usdRange([lo, hi])) : unknown.length ? null : 'free';
      mount(estEl, h('strong', `${fmt.num(n)} images × ${b.configs.length} setups`), ' · ', price, unknown.length ? [price ? ' + ' : '', h('strong', 'cost unknown'), h('span', { class: 'muted' }, ` for ${unknown.join(', ')} (no price on file)`)] : null, secs ? ` · about ${duration(secs)}` : '',
        avail < sample ? h('div', { class: 'xs aip-warn' }, `Only ${fmt.num(avail)} labeled images in scope.`) : null);
    } catch (e) { if (alive && seq === estSeq) mount(estEl, h('span', { class: 'hint' }, 'No estimate available.')); }
  }
  async function start() {
    if (starting) return;
    starting = true; startBtn.disabled = true;
    try { await startBench(); } finally { starting = false; }
  }
  async function startBench() {
    mount(errEl);
    const b = await bodies();
    if (!b.ok) { mount(errEl, b.reason); startBtn.disabled = false; return; }
    try {
      const res = await fovea.api.post('/api/ai/benchmark', { dataset_id: ds.id, sample, filters: filt(), iou: iouT, configs: b.configs.map(({ cloud, src, ...c }) => c) });
      remember();
      jobsEl.prepend(jobBox(fovea, { job: res.job, title: 'Benchmark', onEnd: (snap) => {
        if (!alive) return;
        startBtn.disabled = false;
        if (snap.status === 'done' && Array.isArray(snap.result)) showResults(snap.result, { created_at: Date.now() / 1000, images: sample });
        else loadLast();
      } }).el);
    } catch (e) { mount(errEl, e.status === 404 ? 'Benchmark is not available on this server yet.' : `Could not start: ${errorInfo(e).text}`); startBtn.disabled = false; }
  }
  function showResults(rows, meta) {
    const iouUsed = Number(meta.iou || (rows[0] && rows[0].iou) || iouT);
    const apLabel = Math.abs(iouUsed - 0.5) < 1e-9 ? 'AP50' : `AP@${iouUsed.toFixed(2)}`;
    const apOf = (r) => (r.ap50 != null ? r.ap50 : r.ap != null ? r.ap : null);
    const best = rows.reduce((a, r) => (r.f1 != null && (!a || r.f1 > a.f1) ? r : a), null);
    const pct = (v) => (v == null ? '–' : `${(v * 100).toFixed(1)}%`);
    const findSrc = (cfg) => {
      if (!cfg) return null;
      if (cfg.options && cfg.options.pipeline === 'two_stage') return { pipeline: 'two_stage' };
      const e = EFFORTS.find((x) => x.send === (cfg.options && cfg.options.effort));
      return { provider: cfg.provider, model: cfg.model, ...(e ? { effort: e.value } : {}) };
    };
    mount(resEl, card(h, 'Results', `${when(meta.created_at)}${meta.images ? ` · ${fmt.num(meta.images)} images` : ''} · IoU ${iouUsed.toFixed(2)}`,
      h('div', { class: 'ai-table-wrap' }, h('table', { class: 'table ai-bench' },
        h('thead', h('tr', h('th', 'Setup'), h('th', { class: 'num' }, 'Precision'), h('th', { class: 'num' }, 'Recall'), h('th', 'F1'), h('th', { class: 'num' }, apLabel),
          h('th', { class: 'num' }, 's / image'), h('th', { class: 'num' }, '$ / image'), h('th', { class: 'num' }, 'Errors'), h('th', ''))),
        h('tbody', rows.map((r) => {
          const cfg = r.config || {};
          const src = findSrc(cfg);
          const label = cfg.label || (src && describeSource(C.P, C.prof, src).label) || `${cfg.provider} ${cfg.model || ''}`;
          return h('tr', { class: r === best ? 'best' : '' },
            h('td', h('div', { class: 'strong' }, label, r === best ? h('span', { class: 'chip ok', style: 'margin-left:6px' }, 'best F1') : null), h('div', { class: 'xs faint' }, String(cfg.provider || '').split('+').map((p) => (provOf(C.P, p) || {}).name || p).join(' → '))),
            h('td', { class: 'num' }, pct(r.precision)), h('td', { class: 'num' }, pct(r.recall)),
            h('td', h('div', { class: 'ai-f1' }, h('div', { class: 'bar' }, h('i', { style: `width:${Math.round(100 * (r.f1 || 0))}%` })), h('span', { class: 'mono xs' }, pct(r.f1)))),
            h('td', { class: 'num', title: apOf(r) == null ? 'Needs confidences — this setup reports none' : '' }, apOf(r) == null ? '—' : pct(apOf(r))),
            h('td', { class: 'num' }, r.sec_per_image != null ? Number(r.sec_per_image).toFixed(2) : '–'), h('td', { class: 'num' }, r.usd_per_image ? usd(r.usd_per_image) : '$0'),
            h('td', { class: 'num' }, r.errors ? fmt.num(r.errors) : '0'),
            h('td', src ? h('button', { class: 'btn btn-sm', type: 'button', 'data-tip': 'Make this the default for Propose and batches',
              onClick: (e) => { patchProfile(fovea, ds.id, { source: src }); C.prof = profileOf(ds.id); mount(e.currentTarget, icon('check', 12), 'In use'); } }, sameSource(src, C.prof.source) ? [icon('check', 12), 'In use'] : 'Use') : null));
        })))),
      perClass(rows, apOf, apLabel),
      h('div', { class: 'xs faint mt-8' }, `Precision: share of AI boxes that match a label. Recall: share of your boxes the AI found. ${apLabel} needs confidences (VLMs report none). Scores reflect your labels as they are — a low precision can also mean missing labels.`)));
  }
  /** Per-class AP side by side — for fall detection the rare class (fall) matters more than the average. */
  function perClass(rows, apOf, apLabel) {
    const withPc = rows.filter((r) => Array.isArray(r.per_class) && r.per_class.some((x) => x.ap != null));
    if (!withPc.length) return null;
    const classes = [...new Map(withPc.flatMap((r) => r.per_class).filter((x) => x.gt > 0).map((x) => [x.cls, x])).values()].sort((a, b) => b.gt - a.gt);
    if (!classes.length) return null;
    const cell = (r, c) => { const x = (r.per_class || []).find((y) => y.cls === c.cls); return x && x.ap != null ? `${(x.ap * 100).toFixed(1)}%` : '—'; };
    return h('details', { class: 'ai-pc mt-8' }, h('summary', { class: 'small' }, `${apLabel} per class`),
      h('div', { class: 'ai-table-wrap' }, h('table', { class: 'table' },
        h('thead', h('tr', h('th', 'Class'), h('th', { class: 'num' }, 'Your boxes'), withPc.map((r) => h('th', { class: 'num', title: (r.config || {}).label || '' }, midEllipsis((r.config || {}).label || (r.config || {}).model || '?', 22))))),
        h('tbody', classes.map((c) => h('tr', h('td', h('i', { class: 'ai-sw', style: `background:${fovea.colors.classColor(c.cls)};margin-right:6px` }), `${c.cls} ${displayName(C, c.cls)}`),
          h('td', { class: 'num' }, fmt.num(c.gt)), withPc.map((r) => h('td', { class: 'num' }, cell(r, c)))))))));
  }
  async function loadLast() {
    const runs = await safe(fovea.api.get('/api/ai/runs', { dataset_id: ds.id, kind: 'benchmark', limit: 12 }).then((r) => r.runs || []), []);
    if (!alive) return;
    const last = lastBenchmark(runs);
    if (last) showResults(last.rows, last);
    else mount(resEl, card(h, 'Results', null, h('div', { class: 'hint' }, 'No benchmark yet. A good first run: your current model vs two-stage on 30 labeled images.')));
  }
  renderConfigs();
  loadLast();
  return () => { alive = false; stopJobs(); };
}

// ====================================================================== class cards
function secCards(body, C) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt } = fovea;
  let alive = true, sel = 0, saveTimer = null;
  const sizes = new Map();          // image id -> [width, height] for example outlines
  const list = h('div', { class: 'ai-cc-list', role: 'listbox', 'aria-label': 'Classes' });
  const editor = h('div', { class: 'ai-cc-editor' });
  // First run on a dataset whose classes are only numbers (class_0 …): let a vision model name them from examples.
  const unnamed = placeholderClasses(C.names, C.cards, null).all.length;
  const genReady = (C.P.providers || []).filter((p) => canGenerate(p) && providerState(C.P, p).ok);
  // A dataset with no classes (an image-only folder) has nothing to name: the button says so instead of opening.
  const noClasses = !C.names.length;
  const suggestBtn = h('button', { class: 'btn btn-sm', type: 'button', disabled: noClasses, title: noClasses ? 'Add classes first' : null,
    onClick: () => openSuggestNames(fovea, C.ds, { P: C.P }) }, icon('sparkles', 12), 'Suggest names with AI…');
  const barText = h('span', { class: 'grow small' });
  /** Suggesting is the main action only while a placeholder class has boxes a model can look at; one without boxes
      is described by hand here. */
  function renderBar(counts) {
    const ph = placeholderClasses(C.names, C.cards, counts), n = ph.all.length;
    suggestBtn.classList.toggle('btn-primary', !!counts && ph.withBoxes.length > 0);
    if (noClasses) { mount(barText, 'This dataset has no classes yet — add them first; a card then describes each one for annotators and AI providers. ', addClassesLink(fovea, ds.id)); return; }
    barText.textContent = !n ? 'Names, prompts and definitions guide annotators and every AI provider.'
      : [ph.withBoxes.length ? `${n} of ${C.names.length} classes ${n === 1 ? 'has' : 'have'} only a placeholder name. A vision model can look at a few boxes of each and suggest a name, a prompt and a definition${genReady.length ? '' : ' (needs a cloud key or an OpenAI-compatible vision server)'}.` : '',
        ph.empty.length ? `${classList(C.names, ph.empty)} ${ph.empty.length === 1 ? 'has' : 'have'} ${ph.withBoxes.length ? '' : 'only a placeholder name and '}no boxes, so AI cannot suggest a name — describe ${ph.empty.length === 1 ? 'it' : 'them'} here by hand.` : ''].filter(Boolean).join(' ');
  }
  renderBar(undefined);
  classCounts(fovea, ds.id).then((c) => { if (alive && c) renderBar(c); });
  add(body, cardsMissing(ds.id) ? h('div', { class: 'callout warn small mb-12' }, icon('alert', 14), h('span', 'This server cannot store class cards yet — edits here are not saved.')) : null,
    h('div', { class: `ai-cc-bar ${unnamed ? 'callout' : ''} mb-12` }, unnamed ? icon('info', 14) : null, barText, suggestBtn),
    h('div', { class: 'ai-cc' }, list, editor));
  const counts = () => ((C.stats && C.stats.classes) || []);

  function renderList() {
    const nm = C.names;
    // No classes: the list says so and the editor (an empty bordered card otherwise) is not shown at all.
    if (!nm.length) { mount(list, h('div', { class: 'hint', style: 'padding:12px' }, 'No classes yet.')); mount(editor); editor.classList.add('hidden'); return; }
    editor.classList.remove('hidden');
    mount(list, ...nm.map((_, c) => {
      const ci = classInfo(nm, C.cards, c);
      const n = (counts().find((x) => x.id === c || x.cls === c) || {}).boxes;
      return h('button', { class: `ai-cc-row ${c === sel ? 'active' : ''}`, type: 'button', role: 'option', 'aria-selected': String(c === sel), onClick: () => { flush(); sel = c; renderList(); renderEditor(); } },
        h('i', { class: 'ai-sw', style: `background:${fovea.colors.classColor(c)}` }),
        h('div', { class: 'grow' }, h('div', { class: 'truncate' }, h('span', { class: 'faint mono' }, `${c} `), ci.name), h('div', { class: 'xs truncate muted' }, ci.prompt || (ci.described ? 'described' : 'no description yet'))),
        n != null ? h('span', { class: 'xs faint' }, fmt.compact(n)) : null);
    }));
  }
  let pending = null;
  function flush() { if (pending) { clearTimeout(saveTimer); const p = pending; pending = null; doSave(p.cls, p.patch); } }
  const saveEl = h('span', { class: 'hint' });
  async function doSave(cls, patch) {
    saveEl.textContent = 'Saving…';
    try { await saveCard(fovea, ds.id, cls, patch); C.cards = await loadCards(fovea, ds.id); if (alive) { saveEl.textContent = 'Saved'; renderList(); } }
    catch (e) { if (alive) mount(saveEl, h('span', { class: 'aip-bad' }, `Not saved: ${scrub(e.message)} `), h('button', { class: 'aip-link', type: 'button', onClick: () => doSave(cls, patch) }, 'Retry')); }
  }
  function queue(cls, patch) {
    pending = { cls, patch: { ...((pending && pending.cls === cls && pending.patch) || {}), ...patch } };
    saveEl.textContent = 'Unsaved';
    clearTimeout(saveTimer);
    saveTimer = setTimeout(flush, 700);
  }
  function renderEditor() {
    const nm = C.names;
    if (!nm.length) return;
    const c = sel;
    const ci = classInfo(nm, C.cards, c);
    const card0 = ci.card || {};
    const field = (key, label, hint, rows, placeholder) => {
      const input = rows ? h('textarea', { class: 'input', rows, placeholder, disabled: cardsMissing(ds.id) }) : h('input', { class: 'input', placeholder, disabled: cardsMissing(ds.id) });
      input.value = card0[key] || '';
      input.addEventListener('input', () => { queue(c, { [key]: input.value }); preview(); });
      input.addEventListener('blur', () => flush());
      return { el: h('div', { class: 'field' }, h('label', { class: 'label' }, label), input, hint ? h('div', { class: 'hint' }, hint) : null), input };
    };
    const fPrompt = field('prompt', 'What should a model look for?', 'A short noun phrase. Used as the prompt for open-vocabulary and cloud models, and shown to the two-stage classifier.', 0,
      ci.placeholder ? 'e.g. person lying on the ground after a fall' : `e.g. ${ci.name}`);
    const fDesc = field('description', 'Definition', 'One or two sentences that tell people and models what counts as this class.', 3, 'e.g. A person whose torso is on or near the floor, not by choice (not sitting, not lying on a bed).');
    const fInc = field('include', 'Box it when…', null, 2, 'e.g. partly occluded people; people mid-fall when the body is past 45°');
    const fExc = field('exclude', 'Don’t box…', null, 2, 'e.g. people lying on a bed or sofa; reflections; posters');
    const pv = h('pre', { class: 'ai-cc-pv' });
    function preview() {
      const p = fPrompt.input.value.trim() || ci.name, dsc = fDesc.input.value.trim(), inc = fInc.input.value.trim(), exc = fExc.input.value.trim();
      pv.textContent = `${c} · ${p}${dsc ? `\n  ${dsc}` : ''}${inc ? `\n  Include: ${inc}` : ''}${exc ? `\n  Exclude: ${exc}` : ''}`;
    }
    preview();
    const exEl = h('div', { class: 'ai-cc-ex' });
    const renderExamples = () => {
      const ex = (classInfo(C.names, C.cards, c).card.examples) || [];
      // The card keeps {image_id, row, box}; the crop outline also needs the image size, looked up once per image.
      const need = [...new Set(ex.filter((e) => !e.wh && !sizes.has(e.image_id)).map((e) => e.image_id))];
      if (need.length) imagesById(fovea, ds.id, need).then((items) => { items.forEach((it) => sizes.set(it.id, [it.width, it.height])); if (alive && sel === c) renderExamples(); }).catch(() => {});
      mount(exEl, ...ex.map((e, i) => h('figure', { class: `ai-ex ${e.polarity < 0 ? 'neg' : 'pos'}`, title: `${e.polarity < 0 ? 'Counter-example' : 'Example'} · image ${e.image_id}, box ${e.row + 1}` },
        cropImg(fovea, e.image_id, e.box, 112, e.wh || sizes.get(e.image_id)), h('span', { class: 'ai-ex-pol' }, e.polarity < 0 ? '−' : '+'),
        h('button', { class: 'ai-ex-x', type: 'button', 'aria-label': 'Remove example', onClick: () => { const next = ex.filter((_, j) => j !== i); doSave(c, { examples: next }).then(renderExamples); } }, icon('x', 10)))),
      h('button', { class: 'ai-ex-add', type: 'button', disabled: cardsMissing(ds.id) || ex.length >= 12, onClick: () => pickExamples(c, ex, (next) => doSave(c, { examples: next }).then(renderExamples)) }, icon('plus', 16), h('span', { class: 'xs' }, ex.length >= 12 ? 'Max 12' : 'Add')));
    };
    renderExamples();
    mount(editor, 
      h('div', { class: 'row mb-12' }, h('i', { class: 'ai-sw lg', style: `background:${fovea.colors.classColor(c)}` }), h('h3', { class: 'ai-cc-title' }, `${c} · ${ci.name}`), h('span', { class: 'spacer' }), saveEl),
      ci.placeholder ? h('div', { class: 'callout warn small mb-12' }, icon('alert', 14), h('span', `“${ci.name}” is a placeholder name. Give this class a prompt and a definition so models (and new annotators) know what it means.`)) : null,
      h('div', { class: 'col gap-12' }, fPrompt.el, fDesc.el, h('div', { class: 'fields cols-2' }, fInc.el, fExc.el),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Example boxes'), h('div', { class: 'hint' }, 'Good examples (+) and look-alikes that are not this class (−). Shown to annotators; providers that support exemplars use them too.'), exEl),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'What a model is told'), pv)));
  }
  function pickExamples(c, current, onDone) {
    const nm = C.names;
    let fcls = c, pol = 1, offset = 0;
    const chosen = new Map();
    const grid = h('div', { class: 'ai-pick-grid' });
    const status = h('span', { class: 'hint' });
    const clsSel = h('select', { class: 'select select-sm', style: 'width:auto;max-width:260px', 'aria-label': 'Class to pick boxes from', onChange: (e) => { fcls = Number(e.target.value); pol = fcls === c ? 1 : -1; polSeg.setValue(pol); offset = 0; load(); } },
      nm.map((n, i) => h('option', { value: i, selected: i === c }, `${i} ${n}`)));
    const polSeg = ui.seg([{ value: 1, label: '+ Example' }, { value: -1, label: '− Counter-example' }], pol, (v) => { pol = v; });
    polSeg.classList.add('ai-seg');
    const addBtn = h('button', { class: 'btn btn-primary', type: 'button', disabled: true }, 'Add');
    const m = ui.modal({ title: `Examples for ${c} · ${nm[c]}`, size: 'lg',
      body: h('div', { class: 'col gap-12' }, h('div', { class: 'row wrap gap-8' }, h('span', { class: 'small muted' }, 'Boxes of'), clsSel, polSeg, h('span', { class: 'spacer' }),
        h('button', { class: 'btn btn-sm btn-ghost', type: 'button', onClick: () => { offset += 24; load(); } }, icon('shuffle', 12), 'More')), grid, status),
      footer: (api) => [h('button', { class: 'btn', type: 'button', onClick: () => api.close() }, 'Cancel'), addBtn] });
    addBtn.addEventListener('click', () => {
      const next = current.concat([...chosen.values()]).slice(0, 12);
      m.close(); onDone(next);
    });
    async function load() {
      mount(grid, loadingLine(fovea, 'Loading boxes…'));
      try {
        const r = await fovea.api.get(`/api/datasets/${ds.id}/boxes`, { cls: fcls, sort: 'random', offset, limit: 24 });
        const items = r.items || [];
        if (!items.length) { mount(grid, h('div', { class: 'hint' }, 'No boxes of this class.')); return; }
        mount(grid, ...items.map((b) => {
          const key = `${b.image_id}:${b.row}`;
          const have = current.some((e) => e.image_id === b.image_id && e.row === b.row);
          const tile = h('button', { class: `ai-pick ${chosen.has(key) ? 'on' : ''}`, type: 'button', disabled: have, title: have ? 'Already an example' : basename(b.rel_path),
            onClick: () => {
              if (chosen.has(key)) chosen.delete(key);
              else if (current.length + chosen.size < 12) chosen.set(key, { image_id: b.image_id, row: b.row, polarity: pol, box: [b.xc, b.yc, b.w, b.h], wh: [b.width, b.height] });
              tile.classList.toggle('on', chosen.has(key));
              tile.dataset.pol = chosen.has(key) ? (chosen.get(key).polarity < 0 ? '−' : '+') : '';
              addBtn.disabled = !chosen.size; addBtn.textContent = chosen.size ? `Add ${chosen.size}` : 'Add';
              status.textContent = current.length + chosen.size >= 12 ? 'A card holds at most 12 examples.' : '';
            } }, cropImg(fovea, b.image_id, [b.xc, b.yc, b.w, b.h], 128, [b.width, b.height]));
          return tile;
        }));
      } catch (e) { mount(grid, errLine(fovea, e.status === 404 ? 'Box listing is not available on this server yet.' : e.message)); }
    }
    load();
  }
  renderList(); renderEditor();
  issueLabels(fovea, C).then(() => { if (alive) renderList(); });
  // ↑ ↓ switch classes while the list has focus.
  list.addEventListener('keydown', (e) => {
    if (e.key !== 'ArrowDown' && e.key !== 'ArrowUp') return;
    e.preventDefault();
    flush();
    sel = Math.max(0, Math.min(C.names.length - 1, sel + (e.key === 'ArrowDown' ? 1 : -1)));
    renderList(); renderEditor();
    const a = list.querySelector('.ai-cc-row.active'); if (a) { a.focus({ preventScroll: true }); reveal(a); }
  });
  return () => { alive = false; flush(); };
}
/** A box crop from the server (/api/crop) with the box outlined; falls back to a CSS crop of the thumbnail if that
    endpoint is missing. The server crop is a square centred on the box, pad × the longer side on every edge
    (api/objects.py crop_region), so the outline is placed with the same formula when the image size is known. */
const CROP_PAD = 0.15;
function cropImg(fovea, imageId, box, px, wh) {
  const { h, api } = fovea;
  const wrap = h('span', { class: 'ai-crop', style: `width:${px}px;height:${px}px` });
  if (!box || box.length < 4) { wrap.classList.add('missing'); return wrap; }
  const [xc, yc, w, hh] = box.map(Number);
  const img = h('img', { alt: '', loading: 'lazy', decoding: 'async', src: `/api/crop/${imageId}?box=${[xc, yc, w, hh].map((v) => v.toFixed(5)).join(',')}&pad=${CROP_PAD}&s=${px <= 128 ? 128 : 192}` });
  img.addEventListener('error', () => {
    img.remove();
    if (outline) outline.remove();
    const pw = Math.min(1, w * 1.3), ph = Math.min(1, hh * 1.3), sq = Math.max(pw, ph);
    const left = Math.max(0, Math.min(1 - sq, xc - sq / 2)), top = Math.max(0, Math.min(1 - sq, yc - sq / 2));
    wrap.style.backgroundImage = `url("${api.thumbUrl(imageId, 512)}")`;
    wrap.style.backgroundSize = `${100 / sq}% ${100 / sq}%`;
    wrap.style.backgroundPosition = `${sq >= 1 ? 0 : (100 * left) / (1 - sq)}% ${sq >= 1 ? 0 : (100 * top) / (1 - sq)}%`;
  }, { once: true });
  let outline = null;
  if (wh && wh[0] && wh[1]) {
    const W = Number(wh[0]), H = Number(wh[1]);
    const side = Math.max(w * W, hh * H, 1) * (1 + 2 * CROP_PAD);
    const side2 = Math.max(side, 24);
    const bw = (w * W) / side2, bh = (hh * H) / side2;
    outline = h('i', { class: 'ai-crop-box', style: `left:${50 - bw * 50}%;top:${50 - bh * 50}%;width:${bw * 100}%;height:${bh * 100}%` });
  }
  add(wrap, img, outline);
  return wrap;
}

// ====================================================================== suggest class names (first run on unnamed datasets)
/** Normalise POST /api/ai/name-classes answers: a bare list, or {items | suggestions | classes, usage, warnings}. */
export function normSuggestions(res) {
  const list = Array.isArray(res) ? res : (res && (res.items || res.suggestions || res.classes || res.results)) || [];
  const items = list.filter((x) => x && x.cls != null && isFinite(Number(x.cls))).map((x) => ({
    cls: Number(x.cls), name: String(x.name || '').trim(), prompt: String(x.prompt || '').trim(), description: String(x.description || '').trim(),
    confidence: x.confidence_label || (x.confidence ?? null), examples: Array.isArray(x.examples) ? x.examples.filter((e) => e && e.image_id != null) : [],
    error: x.error ? scrub(x.error) : null, note: x.reason || x.note ? scrub(x.reason || x.note) : '',
    warnings: (Array.isArray(x.warnings) ? x.warnings : []).map(scrub) }));
  const meta = Array.isArray(res) ? {} : (res || {});
  return { items, usage: meta.usage || null, warnings: (meta.warnings || []).map(scrub), latency_ms: meta.latency_ms ?? null, served_by: meta.served_by || '' };
}
/** What Stop did, from the answer: classes the server skipped carry a "Stopped …" error (never sent, not billed). */
export function stoppedText(items) {
  const skipped = items.filter((i) => /^stopped/i.test(String(i.error || ''))).length;
  return skipped
    ? `Stopped: ${skipped} class${skipped === 1 ? ' was' : 'es were'} not sent (not billed). The requests already sent finished and are billed; their suggestions are below.`
    : 'Stopped, but every request had already been sent: they finished and are billed, so their suggestions are below.';
}
/** Confidence as a word: a number (0–1) or low / medium / high. */
export function confWord(c) {
  if (c == null || c === '') return '';
  if (typeof c === 'number' || /^[\d.]+$/.test(String(c))) { const v = Number(c); return v >= 0.75 ? 'high' : v >= 0.45 ? 'medium' : 'low'; }
  return String(c).toLowerCase();
}
/** What is wrong with a set of new names, as {cls: message}. Like Overview's class editor: every name non-empty and
    unique (ignoring case) across the whole class list after the change. */
export function nameProblems(names, edits) {
  const next = names.slice();
  for (const [c, n] of Object.entries(edits)) next[Number(c)] = String(n == null ? '' : n).trim();
  for (let i = 0; i < next.length; i++) if (next[i] == null) next[i] = `class_${i}`;
  const where = new Map();
  next.forEach((n, i) => { const k = String(n).toLowerCase(); if (!where.has(k)) where.set(k, []); where.get(k).push(i); });
  const out = {};
  for (const c of Object.keys(edits).map(Number)) {
    const n = next[c];
    if (!n) { out[c] = 'Give it a name'; continue; }
    if (n.length > 200) { out[c] = 'Name is too long'; continue; }
    const other = where.get(n.toLowerCase()).filter((i) => i !== c);
    if (other.length) out[c] = `Same name as class ${other[0]}${String(other[0]) in edits ? ' (also suggested)' : ''}`;
  }
  return out;
}
/** The class list after renaming: holes (classes beyond the stored list) become class_N. */
export function renamed(names, edits) {
  const next = names.slice();
  for (const [c, n] of Object.entries(edits)) next[Number(c)] = String(n).trim();
  for (let i = 0; i < next.length; i++) if (next[i] == null || next[i] === '') next[i] = `class_${i}`;
  return next;
}
/** A rough price before running, used when the server gives none: each crop is at most 448 px on its long edge
    (≈150–270 image tokens with every provider), plus the instructions and a short JSON answer per class. */
export function namingEstimate(model, nClasses, perClass) {
  const pr = model && model.price;
  const tin = [nClasses * (perClass * 150 + 300), nClasses * (perClass * 270 + 600)];
  const tout = [nClasses * 80, nClasses * 240];
  if (!pr || pr.in == null) return { usd: [0, 0], tokens_in: tin, tokens_out: tout, priced: false };
  const cost = (i, o) => (i * pr.in + o * (pr.out ?? pr.in)) / 1e6;
  return { usd: [cost(tin[0], tout[0]), cost(tin[1], tout[1])], tokens_in: tin, tokens_out: tout, priced: true };
}

/**
 * "Suggest names with AI": a vision model looks at a few typical boxes of each class and proposes a name, a prompt and a
 * definition. The cost estimate is shown before anything is sent; each suggestion is editable and is applied per class
 * (or all at once) to the class list (PATCH dataset) and the class card. Applied rows can be reverted.
 * Exported for Overview (shell): import('./ai.js').then((m) => m.openSuggestNames(fovea, dataset)).
 */
export async function openSuggestNames(fovea, dsIn, { P: P0 = null, onApplied } = {}) {
  const { h, ui, icon, fmt } = fovea;
  let ds = dsIn, alive = true, ac = null, estSeq = 0, runToken = null;
  const body = h('div', { class: 'ai-nm col gap-12' }, loadingLine(fovea, 'Loading providers and classes…'));
  const foot = h('div', { class: 'row gap-8 ai-nm-foot' });
  const modal = ui.modal({ title: 'Suggest class names with AI', size: 'lg', className: 'ai-nm-modal', body, footer: foot,
    // Closing while suggestions are being fetched also tells the server to skip the classes it has not sent yet.
    onClose: () => { alive = false; if (ac) { if (runToken) fovea.api.post('/api/ai/name-classes/cancel', { token: runToken }).catch(() => {}); ac.abort(); } } });
  const [P, prof] = await Promise.all([P0 && !P0.error ? Promise.resolve(P0) : loadProviders(fovea, { force: true }), loadProfile(fovea, ds.id)]);
  let cards = await loadCards(fovea, ds.id, { force: true }).catch(() => ({}));
  try { ds = (await fovea.getDataset(ds.id, { force: true })) || ds; } catch (e) { /* keep the copy we have */ }
  const counts = await safe(fovea.api.get(`/api/datasets/${ds.id}/classes/counts`).then((r) => r.counts || {}), {});
  if (!alive) return modal;
  let names = (ds.classes || []).slice();
  const saved = prof.naming || {};
  let perClass = [4, 6, 8].includes(saved.per_class) ? saved.per_class : 6;
  let src = saved.source && providerState(P, provOf(P, saved.source.provider)).ok && canGenerate(provOf(P, saved.source.provider)) ? saved.source : null;
  if (!src) { const p = firstUsable(P, canGenerate, { cloudFirst: true }); src = p ? { provider: p.id, model: (defaultModel(p) || {}).id } : null; }
  const isPlaceholder = (c) => classInfo(names, cards, c).placeholder || /^class_\d+$/.test(String(names[c] || ''));
  // A class without boxes has nothing to look at: never chosen by default, never sent.
  const countsKnown = Object.keys(counts).length > 0;          // the counts list leaves out classes with no boxes
  const hasBoxes = (c) => !countsKnown || (counts[c] || 0) > 0;
  const pickable = () => names.map((_, c) => c).filter(hasBoxes);
  // Only unnamed classes with boxes are chosen by default; with none left, nothing is (renaming named classes is a
  // deliberate choice — a click on "All" — not a second bill by default).
  let sel = new Set(pickable().filter(isPlaceholder));
  const chosen = () => [...sel].filter(hasBoxes).sort((a, b) => a - b);
  const sizes = new Map();              // image id -> [w, h] for the box outlines on example crops
  let result = null;
  const rowsUi = new Map();             // cls -> {nameIn, promptIn, descIn, probEl, applyBtn, stateEl, before, applied}

  // ---------------------------------------------------------------- setup
  function renderSetup() {
    if (!alive) return;
    result = null; rowsUi.clear();
    const d = describeSource(P, prof, src);
    const anyGen = (P.providers || []).some(canGenerate);
    const pickBtn = h('button', { class: 'btn aip-pick', type: 'button', 'aria-haspopup': 'menu', style: 'max-width:420px',
      onClick: () => pickSource(pickBtn, { fovea, P, prof, dsId: ds.id, current: src, purpose: 'generate', allowTwoStage: false,
        onPick: (s2) => { src = { provider: s2.provider, model: s2.model }; patchProfile(fovea, ds.id, { naming: { source: src, per_class: perClass } }); renderSetup(); } }) },
      icon(d.cloud ? 'zap' : 'cpu', 14), h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, src ? d.label : 'Choose a model'), h('span', { class: 's truncate' }, src ? d.sub : 'needs a vision model')), icon('chevronDown', 12));
    const chips = h('div', { class: 'aip-chips' }, names.map((_, c) => {
      const on = sel.has(c) && hasBoxes(c);
      return h('button', { class: `aip-chip ${on ? 'on' : ''} ${isPlaceholder(c) ? 'undescribed' : ''}`, type: 'button', 'aria-pressed': String(on), disabled: !hasBoxes(c),
        title: hasBoxes(c) ? `${c} · ${names[c]}` : `${c} · ${names[c]} — no boxes to look at`,
        onClick: () => { if (sel.has(c)) sel.delete(c); else sel.add(c); renderSetup(); } },
        on ? icon('check', 10) : null, h('i', { style: `background:${fovea.colors.classColor(c)}` }), h('span', { class: 'truncate' }, `${c} ${names[c]}`),
        counts[c] != null ? h('span', { class: 'faint' }, ` · ${fmt.compact(counts[c])}`) : null);
    }));
    const empty = names.map((_, c) => c).filter((c) => !hasBoxes(c));
    const perSeg = ui.seg([4, 6, 8].map((v) => ({ value: v, label: String(v) })), perClass, (v) => { perClass = v; patchProfile(fovea, ds.id, { naming: { source: src, per_class: v } }); renderEstimate(); });
    perSeg.classList.add('ai-seg');
    const estEl = h('div', { class: 'aib-est', 'aria-live': 'polite' });
    const errEl = h('div', { class: 'aib-err', role: 'alert' });
    mount(body,
      h('p', { class: 'muted small', style: 'margin:0' }, 'A vision model looks at a few typical boxes of each class (the largest ones, from different images) and suggests a short English name, a prompt for detectors and a one-line definition. Nothing changes until you apply a suggestion; applied names can be reverted here or edited in Overview.'),
      !anyGen ? h('div', { class: 'callout warn small' }, icon('alert', 14), h('span', 'No provider here can describe images. Add a key for Anthropic, Google or OpenAI, or point the OpenAI-compatible provider at a server with a vision model (Ollama, LM Studio). ',
        fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings'))) : null,
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model'), pickBtn,
        src ? h('div', { class: 'hint' }, d.ok ? [d.where, d.price ? ` · ${d.price}` : ''].join('') : h('span', { class: 'aip-bad' }, d.reason, d.settings ? [' ', fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings')] : null)) : null),
      h('div', { class: 'field' }, h('div', { class: 'row gap-8' }, h('span', { class: 'label grow' }, `Classes (${chosen().length} of ${names.length})`),
        h('button', { class: 'aip-link', type: 'button', onClick: () => { sel = new Set(pickable().filter(isPlaceholder)); renderSetup(); } }, 'Unnamed only'),
        h('span', { class: 'faint' }, '·'), h('button', { class: 'aip-link', type: 'button', onClick: () => { sel = new Set(pickable()); renderSetup(); } }, 'All')),
        chips, empty.length ? h('div', { class: 'hint' }, `Class${empty.length === 1 ? '' : 'es'} ${empty.join(', ')} ${empty.length === 1 ? 'has' : 'have'} no boxes, so there is nothing to look at.`) : null),
      h('div', { class: 'field' }, h('span', { class: 'label' }, 'Boxes shown per class'), perSeg, h('div', { class: 'hint' }, 'More boxes give a steadier answer for classes that vary (e.g. fallen people in different poses) and cost a little more.')),
      estEl, errEl);
    const go = h('button', { class: 'btn btn-primary', type: 'button', disabled: !d.ok || !chosen().length, onClick: () => run(errEl) }, icon('sparkles', 13), 'Suggest names');
    mount(foot, h('button', { class: 'btn', type: 'button', onClick: () => modal.close() }, 'Cancel'), go);
    function renderEstimate() {
      const seq = ++estSeq;
      if (!names.length) { mount(estEl, h('span', { class: 'aip-bad' }, 'This dataset has no classes yet — add or label some first. ', addClassesLink(fovea, ds.id))); return; }
      if (!d.ok || !chosen().length) {
        const allNamed = !pickable().some(isPlaceholder);
        mount(estEl, h('span', { class: !src || !d.ok || !allNamed ? 'aip-bad' : 'hint' }, !src ? 'Choose a model that can describe images.' : !d.ok ? d.reason
          : allNamed ? `Every class with boxes already has a name${empty.length ? ` (${classList(names, empty)} ${empty.length === 1 ? 'has' : 'have'} no boxes)` : ''}. Choose the classes to rename above.` : 'Choose at least one class.'));
        return;
      }
      const n = chosen().length;
      const local = namingEstimate(modelOf(d.p, d.model), n, perClass);
      const show = (est, server) => {
        if (seq !== estSeq || !alive) return;
        const free = !d.cloud && !(est.usd && est.usd[1]);
        const unknown = !free && est.priced === false && !(est.usd && est.usd[1]);     // a cloud model with no price on record
        const crops = est.images || n * perClass;
        mount(estEl, h('div', { class: 'small' }, h('strong', `${fmt.num(n)} class${n === 1 ? '' : 'es'} × up to ${perClass} boxes`), ' · ',
          free ? 'free' : unknown ? 'cost unknown' : h('strong', usdRange(est.usd)), d.cloud ? h('span', { class: 'muted' }, ` · ${fmt.num(crops)} crops are sent to ${d.p.name}`) : h('span', { class: 'muted' }, ` · ${d.where}`)),
          h('div', { class: 'xs faint' }, server
            ? `${est.requests ? `${fmt.num(est.requests)} request${est.requests === 1 ? '' : 's'} · ` : ''}${est.images ? `${fmt.num(est.images)} crops · ` : ''}≈${fmt.compact(est.tokens_in[0])}–${fmt.compact(est.tokens_in[1])} input tokens${est.priced === false ? ' · this model has no price on record' : ''}.`
            : `Assumes crops of at most 448 px (≈${fmt.compact(est.tokens_in[0])}–${fmt.compact(est.tokens_in[1])} input tokens) and a short answer per class${local.priced ? '' : '; this model has no price on record'}.`));
      };
      show(local, false);
      // The server prices the exact requests it would send (dry run: nothing is sent); the rough local figure shows
      // until that answer is in, and stays if the server cannot price it.
      fovea.api.post('/api/ai/name-classes', { dataset_id: ds.id, provider: src.provider, model: d.model, classes: chosen(), per_class: perClass, dry_run: true })
        .then((r) => { const e = r && r.estimate; if (e && Array.isArray(e.usd)) show({ ...local, ...e, tokens_in: e.tokens_in || local.tokens_in }, true); })
        .catch(() => {});
    }
    renderEstimate();
  }

  // ---------------------------------------------------------------- run
  async function run(errEl) {
    const d = describeSource(P, prof, src);
    const cls = chosen();
    if (!d.ok || !cls.length) return;
    mount(errEl);
    const my = new AbortController();
    ac = my;
    const t0 = Date.now();
    // Stop asks the server to skip the classes it has not sent yet (cancel_token). Requests already sent still finish
    // and are billed, so the dialog keeps waiting and shows what they paid for; pressing again stops waiting.
    const token = `nm-${Date.now().toString(36)}-${Math.random().toString(36).slice(2, 10)}`;
    runToken = token;
    let stopping = false;
    const who = (d.p && d.p.name) || d.label;
    const line = h('div', { class: 'row small muted gap-8' });
    const showWait = () => { if (alive && ac === my) mount(body, mount(line, ui.spinner(), stopping ? `Stopping: classes not sent yet are skipped. The requests already sent to ${who} finish first (they are billed)…` : `Asking ${d.label} about ${cls.length} class${cls.length === 1 ? '' : 'es'}…`, ui.elapsedClock(t0))); };
    const stopBtn = h('button', { class: 'btn', type: 'button', onClick: async () => {
      if (stopping) { my.abort(); return; }
      stopping = true;
      stopBtn.textContent = 'Stop waiting';
      const r = await fovea.api.post('/api/ai/name-classes/cancel', { token }).catch(() => null);
      if (!r) { my.abort(); return; }           // this server cannot stop it: stop waiting (the note says what that means)
      clearTimeout(wait); showWait();
    } }, 'Stop');
    mount(foot, stopBtn);
    const wait = setTimeout(showWait, 400);
    try {
      const res = await fovea.api.post('/api/ai/name-classes', { dataset_id: ds.id, provider: src.provider, model: d.model, classes: cls, per_class: perClass, cancel_token: token }, { signal: my.signal });
      clearTimeout(wait);
      if (!alive) return;
      result = { ...normSuggestions(res), seconds: (Date.now() - t0) / 1000, model: d.label, stopped: stopping || !!(res && res.stopped) };
      if (result.served_by && result.served_by !== d.model) result.served_by = `served by ${modelName(d.p, result.served_by)}`;
      else result.served_by = '';
      patchProfile(fovea, ds.id, { naming: { source: src, per_class: perClass } });
      renderResults();
    } catch (e) {
      clearTimeout(wait);
      if (!alive) return;
      renderSetup();
      const box = body.querySelector('.aib-err');
      const ei = errorInfo(e);
      if (box) mount(box, e.name === 'AbortError' ? `Stopped waiting. Requests already sent to ${who} still finish and are billed (see Runs & cost), but their suggestions are not shown here.` : e.status === 404 ? 'This Fovea server cannot suggest class names yet.' : `Could not get suggestions: ${ei.text}`,
        ei.settings && e.name !== 'AbortError' ? [' ', fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings')] : null);
    } finally { if (ac === my) ac = null; }
  }

  // ---------------------------------------------------------------- results
  function renderResults() {
    const items = result.items.slice().sort((a, b) => a.cls - b.cls);
    const u = result.usage || {};
    const named = items.filter((i) => i.name).length;
    const facts = [named === items.length ? `${items.length} suggestion${items.length === 1 ? '' : 's'}` : `${named} of ${items.length} classes named`,
      u.cost_usd ? usd(u.cost_usd) : '', `${result.seconds.toFixed(1)} s`, result.served_by || result.model].filter(Boolean).join(' · ');
    const list = h('div', { class: 'ai-nm-list' }, items.map(row));
    const firstErr = (items.find((i) => i.error) || {}).error;
    mount(body,
      h('div', { class: 'row wrap gap-8' }, h('span', { class: 'small' }, h('strong', facts)), h('span', { class: 'spacer' }),
        h('button', { class: 'aip-link', type: 'button', onClick: () => renderSetup() }, 'Change model or classes')),
      result.stopped ? h('div', { class: 'callout small' }, icon('info', 14), h('span', stoppedText(items))) : null,
      items.length && !named && !result.stopped ? h('div', { class: 'callout warn small' }, icon('alert', 14), h('span', `No usable suggestion came back${firstErr ? ` (${firstErr})` : ''}. Try another model, or name the classes yourself below.`)) : null,
      result.warnings.length ? h('div', { class: 'xs aip-warn' }, result.warnings.join(' · ')) : null,
      items.length ? h('div', { class: 'xs faint' }, 'Edit anything before applying. Names follow the dataset’s class list (data.yaml exports use them); the prompt and definition go to the class card.') : h('div', { class: 'hint' }, 'The model returned no suggestions.'),
      list);
    const applyAll = h('button', { class: 'btn btn-primary', type: 'button', onClick: () => applyRows([...rowsUi.values()].filter((r) => !r.applied && !r.problem)) }, icon('check', 13), 'Apply all');
    mount(foot, h('button', { class: 'btn', type: 'button', onClick: () => modal.close() }, 'Close'), applyAll);
    foot.applyAll = applyAll;
    updateProblems();
  }
  function row(it) {
    const c = it.cls;
    const ci = classInfo(names, cards, c);
    const nameIn = h('input', { class: 'input input-sm mono', value: it.name, 'aria-label': `New name for class ${c}`, onInput: () => updateProblems(), spellcheck: 'false' });
    const promptIn = h('input', { class: 'input input-sm', value: it.prompt, 'aria-label': `Prompt for class ${c}` });
    const descIn = h('textarea', { class: 'input', rows: 2, 'aria-label': `Definition of class ${c}` });
    descIn.value = it.description;
    const probEl = h('div', { class: 'xs aip-warn ai-nm-prob' });
    const stateEl = h('span', { class: 'xs muted' });
    const applyBtn = h('button', { class: 'btn btn-sm', type: 'button', onClick: () => applyRows([r]) }, icon('check', 12), 'Apply');
    const crops = h('div', { class: 'ai-nm-crops' });
    const curEl = h('span', { class: 'truncate strong', title: names[c] }, names[c] || `class_${c}`);
    const r = { cls: c, nameIn, promptIn, descIn, probEl, applyBtn, stateEl, curEl, applied: false, before: null, problem: '' };
    rowsUi.set(c, r);
    const showCrops = (list) => {
      const four = list.slice(0, 4);
      const draw = () => mount(crops, four.map((e) => cropImg(fovea, e.image_id, e.box, 60, e.wh || (e.width ? [e.width, e.height] : sizes.get(e.image_id)))));
      draw();
      // Examples from the server carry {image_id, box}; the outline on the crop also needs the image size.
      const need = [...new Set(four.filter((e) => !e.wh && !e.width && !sizes.has(e.image_id)).map((e) => e.image_id))];
      if (need.length) imagesById(fovea, ds.id, need).then((items) => { items.forEach((x) => { if (x.width) sizes.set(x.id, [x.width, x.height]); }); if (alive) draw(); }).catch(() => {});
    };
    if (it.examples.length && it.examples.some((e) => e.box)) showCrops(it.examples.filter((e) => e.box));
    else fovea.api.get(`/api/datasets/${ds.id}/boxes`, { cls: c, sort: 'area', order: 'desc', limit: 4 })
      .then((res) => { if (alive) showCrops((res.items || []).map((b) => ({ image_id: b.image_id, box: [b.xc, b.yc, b.w, b.h], wh: [b.width, b.height] }))); }).catch(() => {});
    const cw = confWord(it.confidence);
    const replaces = [ci.card.prompt ? `prompt “${midEllipsis(ci.card.prompt, 40)}”` : '', ci.card.description ? 'definition' : ''].filter(Boolean);
    return h('div', { class: 'ai-nm-row' },
      h('div', { class: 'ai-nm-cur' },
        h('div', { class: 'row gap-6' }, h('i', { class: 'ai-sw', style: `background:${fovea.colors.classColor(c)}` }), h('span', { class: 'mono faint' }, String(c)),
          curEl, h('span', { class: 'spacer' }),
          cw ? h('span', { class: `chip ${cw === 'high' ? 'ok' : cw === 'low' ? 'warn' : ''}`, title: `Model confidence: ${it.confidence}` }, cw) : null),
        crops,
        counts[c] != null ? h('div', { class: 'xs faint' }, fmt.plural(counts[c], 'box', 'boxes')) : null,
        it.note ? h('div', { class: 'xs muted' }, `“${it.note}”`) : null),
      h('div', { class: 'ai-nm-fields' },
        it.error ? h('div', { class: 'xs aip-bad' }, it.error) : null,
        it.warnings.length ? h('div', { class: 'xs aip-warn' }, it.warnings.join(' · ')) : null,
        h('label', { class: 'ai-nm-f' }, h('span', { class: 'xs muted' }, 'Name'), nameIn),
        h('label', { class: 'ai-nm-f' }, h('span', { class: 'xs muted' }, 'Prompt'), promptIn),
        h('label', { class: 'ai-nm-f' }, h('span', { class: 'xs muted' }, 'Definition'), descIn),
        replaces.length ? h('div', { class: 'xs faint' }, `Replaces the card’s current ${replaces.join(' and ')}.`) : null,
        h('div', { class: 'row gap-8 ai-nm-act' }, probEl, h('span', { class: 'spacer' }), stateEl, applyBtn)));
  }
  function updateProblems() {
    const edits = {};
    rowsUi.forEach((r, c) => { if (!r.applied) edits[c] = r.nameIn.value; });
    const probs = nameProblems(names, edits);
    let ready = 0;
    rowsUi.forEach((r, c) => {
      r.problem = r.applied ? '' : probs[c] || '';
      r.probEl.textContent = r.problem;
      r.applyBtn.disabled = !!r.problem || r.busy;
      if (!r.applied && !r.problem) ready++;
    });
    const all = foot.applyAll;
    if (all) { all.disabled = !ready; mount(all, icon('check', 13), ready && ready < [...rowsUi.values()].filter((r) => !r.applied).length ? `Apply ${ready} ready` : 'Apply all'); }
  }
  function markRow(r, applied) {
    r.applied = applied;
    const now = names[r.cls] || `class_${r.cls}`;
    r.curEl.textContent = now;
    r.curEl.title = applied && r.before ? `${now} (was ${r.before.name})` : now;
    [r.nameIn, r.promptIn, r.descIn].forEach((x) => { x.disabled = applied; });
    mount(r.stateEl, applied ? [icon('check', 11), ' Applied · '] : null,
      applied ? h('button', { class: 'aip-link', type: 'button', onClick: () => revertRow(r) }, 'Revert') : null);
    r.applyBtn.classList.toggle('hidden', applied);
  }
  async function freshNames() {
    const fresh = await fovea.getDataset(ds.id, { force: true });
    ds = fresh || ds; names = (ds.classes || names).slice();
  }
  async function applyRows(list) {
    if (!list.length) return;
    list.forEach((r) => { r.busy = true; r.applyBtn.disabled = true; r.stateEl.textContent = 'Saving…'; });
    try {
      await freshNames();                        // a rename made elsewhere in the meantime is kept
      const edits = Object.fromEntries(list.map((r) => [r.cls, r.nameIn.value.trim()]));
      const probs = nameProblems(names, edits);
      const ok = list.filter((r) => !probs[r.cls]);
      list.filter((r) => probs[r.cls]).forEach((r) => { r.busy = false; r.stateEl.textContent = ''; r.probEl.textContent = probs[r.cls]; });
      if (!ok.length) return;
      const before = new Map(ok.map((r) => [r.cls, { name: names[r.cls] ?? `class_${r.cls}`, prompt: (cards[r.cls] || cards[String(r.cls)] || {}).prompt || '', description: (cards[r.cls] || cards[String(r.cls)] || {}).description || '' }]));
      const next = renamed(names, Object.fromEntries(ok.map((r) => [r.cls, edits[r.cls]])));
      if (JSON.stringify(next) !== JSON.stringify(names)) {
        const res = await fovea.api.patch(`/api/datasets/${ds.id}`, { classes: next });
        ds = (res && res.dataset) || { ...ds, classes: next };
        names = (ds.classes || next).slice();
      }
      for (const r of ok) {
        const patch = {};
        const pr = r.promptIn.value.trim(), de = r.descIn.value.trim();
        if (pr) patch.prompt = pr;
        if (de) patch.description = de;
        if (Object.keys(patch).length) await saveCard(fovea, ds.id, r.cls, patch);
      }
      cards = await loadCards(fovea, ds.id).catch(() => cards);
      ok.forEach((r) => { r.busy = false; r.before = before.get(r.cls); markRow(r, true); });
      fovea.bus.emit('dataset:updated', ds);
      onApplied && onApplied(ds);
    } catch (e) {
      list.forEach((r) => { r.busy = false; if (!r.applied) mount(r.stateEl, h('span', { class: 'aip-bad' }, `Not applied: ${errorInfo(e).text}`)); });
    } finally { updateProblems(); }
  }
  async function revertRow(r) {
    if (!r.before) return;
    r.stateEl.textContent = 'Reverting…';
    try {
      await freshNames();
      const next = renamed(names, { [r.cls]: r.before.name });
      const probs = nameProblems(names, { [r.cls]: r.before.name });
      if (probs[r.cls]) { mount(r.stateEl, h('span', { class: 'aip-bad' }, `Cannot revert: ${probs[r.cls]}`)); return; }
      const res = await fovea.api.patch(`/api/datasets/${ds.id}`, { classes: next });
      ds = (res && res.dataset) || { ...ds, classes: next };
      names = (ds.classes || next).slice();
      await saveCard(fovea, ds.id, r.cls, { prompt: r.before.prompt, description: r.before.description });
      cards = await loadCards(fovea, ds.id).catch(() => cards);
      markRow(r, false);
      fovea.bus.emit('dataset:updated', ds);
      onApplied && onApplied(ds);
    } catch (e) { mount(r.stateEl, h('span', { class: 'aip-bad' }, `Not reverted: ${errorInfo(e).text}`)); }
    finally { updateProblems(); }
  }
  renderSetup();
  return modal;
}

// ====================================================================== runs & cost
/** Stop an Economy run: cancelling its job cancels the provider batches too and files what already finished; a run
    with no live job (its provider was unavailable at the last start) is cancelled through the run itself. */
async function cancelProviderBatch(fovea, run) {
  if (run.job && ['queued', 'running'].includes(run.job.status)) return fovea.api.post(`/api/jobs/${run.job.id}/cancel`);
  return fovea.api.post(`/api/ai/runs/${run.id}/cancel`, {});
}
/** A cloud run booked at $0 because its model has no price on file (a live-listed model): "not tracked", not "$0".
    Known from the run's warnings, or from the model list (cloud model without a price) when it called the API. */
export function costUntracked(C, r) {
  if (!r || r.cost_usd) return false;
  const res = r.result && typeof r.result === 'object' && !Array.isArray(r.result) ? r.result : {};
  const warns = [...(Array.isArray(res.warnings) ? res.warnings : []), ...(Array.isArray(r.warnings) ? r.warnings : [])];
  if (warns.some((w) => /cost (is )?not tracked|no price on file/i.test(String(w)))) return true;
  const p = provOf(C.P, r.provider);
  if (!p || !isCloud(p) || !(r.input_tokens || r.output_tokens)) return false;
  const m = modelOf(p, r.model);
  return !!m && !(m.price && m.price.in != null);
}
/** A provider batch's state in a few words, for the runs table's status chip (the full wording is in its tooltip). */
const shortBatchState = (st) => ({ running: 'at provider', in_progress: 'at provider', submitted: 'submitted', validating: 'submitted', submitting: 'submitting',
  canceling: 'cancelling', cancelling: 'cancelling', ended: 'saving', ingesting: 'saving' }[st] || batchStateText(st).toLowerCase());
/** Images of an Economy run that were never submitted (Fovea stopped mid-submit; they are not resent on restart). */
const notSentOf = (r) => Number(((r && r.job && r.job.extra) || {}).not_submitted ?? ((r && r.result) || {}).not_submitted) || 0;
function secRuns(body, C) {
  const { fovea, ds } = C;
  const { h, ui, icon, fmt } = fovea;
  let alive = true, limit = 50, pollTimer = null, fastPolls = 0;
  const usageEl = h('div');
  const tableEl = h('div');
  const batchEl = h('div');
  add(body, h('div', { class: 'col gap-16' }, batchEl, card(h, 'Spending', 'last 30 days, all datasets', usageEl), card(h, 'Runs on this dataset', null, tableEl)));

  /** Provider batches still at the provider (economy runs): state, how many results are back, Cancel. */
  function renderBatches(runs) {
    const open = runs.filter(batchOpen).map((r) => ({ r, b: providerBatchOf(r) }));
    if (!open.length) { mount(batchEl); return false; }
    mount(batchEl, card(h, 'Provider batches in progress', 'economy runs · checked every minute, also after a restart',
      h('div', { class: 'ai-pb-list' }, open.map(({ r, b }) => {
        const cancelBtn = h('button', { class: 'btn btn-sm', type: 'button', disabled: /cancel/.test(b.state) || r.status === 'cancelling',
          onClick: async () => {
            if (!(await ui.confirm({ title: 'Cancel this provider batch?', okLabel: 'Cancel batch', danger: true, cancelLabel: 'Keep it',
              message: 'The provider stops the requests that have not run yet. Results that are already back are still filed as proposals (or written, for a write run); you pay only for those.' }))) return;
            cancelBtn.disabled = true; cancelBtn.textContent = 'Cancelling…';
            // The job cancels the provider batch on its next check (a second or two): follow it closely until it has.
            try { await cancelProviderBatch(fovea, r); fastPolls = 15; clearTimeout(pollTimer); pollTimer = setTimeout(() => alive && loadRuns(), 800); }
            catch (e) { cancelBtn.disabled = false; cancelBtn.textContent = 'Cancel'; ui.toast(`Not cancelled: ${errorInfo(e).text}`, { type: 'error' }); }
          } }, 'Cancel');
        const pct = b.total ? Math.round(100 * b.done / b.total) : 0;
        return h('div', { class: 'ai-pb' },
          h('div', { class: 'grow col gap-4' },
            h('div', { class: 'row wrap gap-6' }, h('span', { class: 'strong' }, runSource(C, r)), h('span', { class: 'chip info' }, batchStateText(b.state)),
              r.params && r.params.target && r.params.target !== 'proposals' ? h('span', { class: 'chip warn' }, r.params.target) : null),
            h('div', { class: 'xs muted' }, [`${fmt.num(b.total || r.images || 0)} images`, b.total ? `${fmt.num(b.done)} of ${fmt.num(b.total)} processed` : '',
              b.filed ? `${fmt.num(b.filed)} filed` : '', b.counts.errored ? `${fmt.num(b.counts.errored)} failed` : '', `submitted ${when(b.submitted_at)}`,
              b.next_check_at ? `next check ${new Date(b.next_check_at * 1000).toLocaleTimeString('en-US', { hour: 'numeric', minute: '2-digit' })}` : '',
              b.id ? `${b.batches > 1 ? `${b.batches} batches, first ` : 'batch '}${midEllipsis(b.id, 24)}` : ''].filter(Boolean).join(' · ')),
            b.not_submitted ? h('div', { class: 'xs aip-warn' }, notSentText(b.not_submitted)) : null,
            b.total ? ui.progress(pct / 100) : null),
          cancelBtn);
      }))));
    return true;
  }

  async function loadUsage() {
    mount(usageEl, loadingLine(fovea, 'Loading…'));
    let u;
    try { u = await fovea.api.get('/api/ai/usage', { days: 30 }); } catch (e) { if (alive) mount(usageEl, h('div', { class: 'hint' }, e.status === 404 ? 'Usage is not available on this server yet.' : scrub(e.message))); return; }
    if (!alive) return;
    const daily = u.daily || [];
    const byDate = new Map(daily.map((d) => [d.date || d.day, d.usd ?? d.cost_usd ?? 0]));
    const days = [];
    for (let i = 29; i >= 0; i--) { const d = new Date(Date.now() - i * 86400000); const k = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`; days.push([k, byDate.get(k) || 0, d]); }
    const max = Math.max(1e-9, ...days.map((d) => d[1]));
    const val = (v) => (v && typeof v === 'object' ? v.usd || 0 : Number(v) || 0);
    const nameOf = (k, v) => {
      const prov = v && typeof v === 'object' && v.model ? v.provider : k;
      if (String(prov).includes('+')) {            // two-stage: "yolo+mock" / "yolo12m+mock-fast"
        if (!(v && v.model)) return `Two-stage (${String(prov).split('+').map((p) => (provOf(C.P, p) || {}).name || p).join(' → ')})`;
        const ps = String(prov).split('+'), ms = String(v.model).split('+');
        return `Two-stage · ${ps.map((p, i) => modelName(provOf(C.P, p), ms[i] || '')).join(' → ')}`;
      }
      return v && typeof v === 'object' && v.model ? modelName(provOf(C.P, v.provider), v.model) : (provOf(C.P, k) || {}).name || k;
    };
    const chips = (obj) => Object.entries(obj || {}).filter(([, v]) => val(v) > 0).sort((a, b) => val(b[1]) - val(a[1])).map(([k, v]) => h('span', { class: 'chip' }, `${nameOf(k, v)} · ${usd(val(v))}`));
    mount(usageEl, h('div', { class: 'row wrap gap-16' },
      h('div', h('div', { class: 'ai-big' }, usd(u.total_usd || 0)), h('div', { class: 'xs muted' }, 'cloud spend, 30 days')),
      h('div', { class: 'ai-spark', role: 'img', 'aria-label': `Daily spend over 30 days, highest ${usd(max)}` }, days.map(([k, v, d]) => h('i', { style: `height:${v ? Math.max(6, Math.round(100 * v / max)) : 2}%`, class: v ? '' : 'zero', 'data-tip': `${d.toLocaleDateString('en-US', { month: 'short', day: 'numeric' })}: ${usd(v)}` }))),
      h('div', { class: 'col gap-4' }, h('div', { class: 'row wrap gap-4' }, chips(u.by_provider)), h('div', { class: 'row wrap gap-4' }, chips(u.by_model)))),
      h('div', { class: 'xs faint mt-8' }, 'Costs come from the providers’ reported token usage and the price table (local models cost nothing).'));
  }
  async function loadRuns() {
    if (!tableEl.firstChild) mount(tableEl, loadingLine(fovea, 'Loading runs…'));    // a refresh keeps the table until the new one is in
    let runs;
    try { const r = await fovea.api.get('/api/ai/runs', { dataset_id: ds.id, limit }); runs = r.runs || r || []; }
    catch (e) { if (alive) mount(tableEl, h('div', { class: 'hint' }, e.status === 404 ? 'Run history is not available on this server yet.' : scrub(e.message))); return; }
    if (!alive) return;
    // Pending provider batches update in place every 30 s while this page is open (static text, no animation).
    clearTimeout(pollTimer);
    if (renderBatches(runs)) pollTimer = setTimeout(() => alive && loadRuns(), fastPolls > 0 ? (fastPolls--, 1000) : 30000);
    else fastPolls = 0;
    if (!runs.length) { mount(tableEl, h('div', { class: 'hint' }, 'No AI runs on this dataset yet.')); return; }
    const srcLabel = (r) => runSource(C, r);
    mount(tableEl, h('div', { class: 'ai-table-wrap' }, h('table', { class: 'table ai-runs' },
      h('thead', h('tr', h('th', 'When'), h('th', 'Kind'), h('th', 'Model'), h('th', { class: 'num' }, 'Images'), h('th', { class: 'num', title: 'Boxes proposed or audit findings' }, 'Boxes'), h('th', { class: 'num', title: 'Still waiting for a decision' }, 'Pending'),
        h('th', { class: 'num' }, 'Tokens in / out'), h('th', { class: 'num' }, 'Cost'), h('th', 'Status'), h('th', { class: 'ai-runs-del' }, h('span', { class: 'sr-only' }, 'Delete')))),
      h('tbody', runs.map((r) => {
        const target = r.params && r.params.target;
        const dur = r.finished_at && r.created_at ? r.finished_at - r.created_at : null;
        return h('tr',
          h('td', { class: 'nowrap', title: r.created_at ? new Date(r.created_at * 1000).toLocaleString('en-US') : '' }, when(r.created_at)),
          h('td', { class: 'ai-runs-kind' }, RUN_KIND[r.kind] || String(r.kind || '').replace(/_/g, ' '), r.params && r.params.speed === 'economy' ? h('span', { class: 'chip', style: 'margin-left:4px', title: 'Provider batch API, about half price' }, 'economy') : null,
            target && target !== 'proposals' ? h('span', { class: 'chip warn', style: 'margin-left:4px' }, target) : null),
          h('td', { class: 'ai-runs-src' }, h('span', { class: 'truncate', title: srcLabel(r) }, srcLabel(r))),
          h('td', { class: 'num', title: batchOpen(r) ? 'Submitted to the provider; filed when the results are back' : '' }, fmt.num(batchOpen(r) ? providerBatchOf(r).total : r.images || 0)),
          // Text-only runs draw no boxes: an Ask AI ledger row keeps its number of questions in `boxes`.
          h('td', { class: 'num', title: r.kind === 'nl_filter' ? `${fmt.plural(r.boxes || 0, 'question')}` : '' }, ['benchmark', 'nl_filter', 'name_classes'].includes(r.kind) ? '–' : fmt.num(r.boxes || 0)), h('td', { class: 'num' }, pendingOf(r) != null && r.kind !== 'benchmark' ? fmt.num(pendingOf(r)) : '–'),
          h('td', { class: 'num mono xs' }, r.input_tokens || r.output_tokens ? `${fmt.compact(r.input_tokens || 0)} / ${fmt.compact(r.output_tokens || 0)}` : '–'),
          h('td', { class: 'num' }, costUntracked(C, r) ? h('span', { class: 'muted', title: 'This model has no price on file, so the cost of this run was not tracked (it is counted as $0 in the totals).' }, 'not tracked') : r.cost_usd ? usd(r.cost_usd) : '$0'),
          // Status stays one short chip (the time it took and an error are in its tooltip; the Kind column already says
          // "economy"), so the table fits a 1280 px window with the delete button in view.
          h('td', { class: 'ai-runs-st' }, h('span', { class: `chip ${r.status === 'done' ? 'ok' : r.status === 'error' ? 'danger' : ['running', 'waiting'].includes(r.status) ? 'info' : ''}`,
            title: [batchOpen(r) ? `Economy · ${batchStateText(providerBatchOf(r).state)}` : '', dur ? `Took ${duration(dur)}` : '', r.error ? scrub(r.error) : ''].filter(Boolean).join(' — ') || null },
            batchOpen(r) ? shortBatchState(providerBatchOf(r).state) : r.status || '–'),
            notSentOf(r) ? h('span', { class: 'chip warn', style: 'margin-left:4px', title: notSentText(notSentOf(r)) }, `${fmt.num(notSentOf(r))} never sent`) : null),
          h('td', { class: 'ai-runs-del' }, h('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', 'aria-label': `Delete run ${r.id}`, 'data-tip': batchOpen(r) ? 'Cancel the provider batch first' : 'Delete run (drops its pending proposals)', disabled: r.status === 'running' || batchOpen(r),
            onClick: async () => {
              if (!(await ui.confirm({ title: 'Delete this run?', okLabel: 'Delete run', danger: true, message: `Its ${pendingOf(r) != null ? fmt.num(pendingOf(r)) + ' ' : ''}pending proposals or audit findings are dropped. Boxes you already accepted stay in your label files.` }))) return;
              try { await fovea.api.del(`/api/ai/runs/${r.id}`); loadRuns(); } catch (e) { ui.toast(scrub(e.message), { type: 'error' }); }
            } }, icon('trash', 12))));
      })))),
    runs.length >= limit ? h('div', { class: 'row mt-8' }, h('button', { class: 'btn btn-sm', type: 'button', onClick: () => { limit += 50; loadRuns(); } }, 'Show more')) : null);
  }
  loadUsage(); loadRuns();
  return () => { alive = false; clearTimeout(pollTimer); };
}
