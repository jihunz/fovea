// AI assist panel mounted inside Annotate, plus the AI client helpers the AI tab (views/ai.js) shares: provider
// list and availability, per-dataset AI profile, class cards, the source picker, the batch form / dialog, job boxes.
// Contract with Annotate (SPEC §3.6): mountAIPanel(el, host) -> { destroy, propose, openBatch, openPicker }.
// AI output is never written straight into the open image: Propose stores pending proposals for the image it was
// asked about, and the panel hands them to Annotate only if that image is still the one on screen.

import { mount, frag, splitTail } from '../lib/dom.js';

/** Null-safe DOM insertion (skips null / false, flattens arrays) — native append() would print "null". */
export const add = (el, ...kids) => { el.appendChild(frag(...kids)); return el; };
export { mount };
/** Scroll el into view inside its own scroll box only (a .page or a dialog body). Element.scrollIntoView would also
    scroll overflow-hidden ancestors and shift the whole app shell. */
export function reveal(el) {
  const box = el && el.closest && el.closest('.modal-body, .page, .ai-cc-list');
  if (!box) return;
  const r = el.getBoundingClientRect(), b = box.getBoundingClientRect();
  if (r.bottom > b.bottom) box.scrollTop += Math.min(r.bottom - b.bottom + 12, r.top - b.top - 12);
  else if (r.top < b.top) box.scrollTop -= b.top - r.top + 12;
}
export const SETTINGS_URL = '/settings#ai-providers';
export const AUDIT_KINDS = ['missing', 'loose', 'swap', 'spurious'];
export const COCO80 = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light',
  'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
  'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
  'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife',
  'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair',
  'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
  'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'];
const STOCK_COCO = /^yolo(v?\d+|11|12|26)[nslmx]$/i;      // stock checkpoints: COCO names without loading the model
const PLACEHOLDER = /^class_\d+$/;
/** requestFor's reason (fix: 'classes') on a dataset without any class: an image-only folder registered for labeling. */
export const NO_CLASSES = 'This dataset has no classes yet — add one first.';
/** Where to add one from the AI tab or a dialog: Overview's Classes card. */
export const addClassesLink = (fovea, dsId, text = 'Add classes in Overview') => fovea.link(`/d/${dsId}`, { class: 'aip-link' }, text);
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);
const hostLooksLocal = () => { try { return LOCAL_HOSTS.has(location.hostname) || location.hostname.endsWith('.localhost'); } catch (e) { return false; } };
/** Is this browser on the computer Fovea runs on? The server decides (GET /api/meta → server.request_is_local, from the
    request's address — what its remote-AI guard checks). The address bar is only a fallback (an older server, /api/meta
    unanswered): in Docker the host's browser opens http://localhost:8000 yet reaches the container from the bridge
    gateway, so the server refuses cloud calls a "localhost" page would offer. */
export const isLocalBrowser = (P) => { const v = P && P.request_is_local; return typeof v === 'boolean' ? v : hostLooksLocal(); };
/** Does the server refuse this provider for this browser (api/ai.py _guard_remote)? A cloud provider, or a local
    server it sends the owner's saved key to (`guarded`), asked from another machine without FOVEA_ALLOW_REMOTE_AI. */
export const remoteBlocked = (P, p) => !!p && !!P && P.remote_ai_allowed === false && !isLocalBrowser(P) && (p.guarded != null ? !!p.guarded : isCloud(p));
/** Said wherever cloud providers are listed while this browser is on another machine (the Settings card says the same). */
export const REMOTE_AI_NOTE = 'You are using Fovea from another machine: cloud providers and saved API keys work only on the computer Fovea runs on (start Fovea with FOVEA_ALLOW_REMOTE_AI=1 to allow other machines). Models that run on the server itself still work.';
export const remoteNoteShown = (P) => !!P && !P.error && P.remote_ai_allowed === false && !isLocalBrowser(P);

// ------------------------------------------------------------------ formatting
export function usd(x) {
  if (x == null || isNaN(x)) return '–';
  x = Number(x);
  if (x === 0) return '$0';
  if (x < 0.0001) return '<$0.0001';
  if (x < 0.01) return '$' + String(Number(x.toPrecision(2)));
  if (x < 1000) return '$' + x.toFixed(2);
  return '$' + Math.round(x).toLocaleString('en-US');
}
export function usdRange(r) {
  if (!Array.isArray(r)) return '–';
  const [lo, hi] = r;
  if (!lo && !hi) return '$0';
  return Math.abs((hi || 0) - (lo || 0)) < 1e-9 ? `≈ ${usd(lo)}` : `≈ ${usd(lo)}–${usd(hi).slice(1)}`;
}
export function duration(sec) {
  if (sec == null || !isFinite(sec)) return '';
  if (sec < 90) return `${Math.max(1, Math.round(sec))} s`;
  if (sec < 5400) return `${Math.round(sec / 60)} min`;
  return `${(sec / 3600).toFixed(1)} h`;
}
/** Keep both ends of a file name: frame numbers live at the end. */
export function midEllipsis(s, max = 34) {
  s = String(s || '');
  if (s.length <= max) return s;
  const keep = max - 1, a = Math.ceil(keep * 0.45);
  return s.slice(0, a) + '…' + s.slice(s.length - (keep - a));
}
export const basename = (p) => String(p || '').split('/').pop();
/** A dataset-relative path with its end always visible (the shell's ui.midText). The folder matters: the same frame
    name can exist in two split folders (test/ and test-1854-1860-포함/), and a bare file name hides which one it is. */
export function relPath(fovea, p, { tail = 26, cls = '' } = {}) {
  const t = String(p || '');
  return fovea.ui && fovea.ui.midText ? fovea.ui.midText(t, { tail, cls }) : fovea.h('span', { class: `truncate ${cls}`, title: t }, midEllipsis(t, 44));
}
/** An image's path in a narrow caption (a ~160 px figure), on its own line: the split as a small chip, then the file
    name. When room runs out the name's head (a prefix every frame shares) gives way first, then the chip; the name's
    end — sequence and frame number — always stays. Full path in the tooltip; the line clips rather than paint over
    what is next to it. */
export function pathLine(fovea, it, { tail = 14, cls = '' } = {}) {
  const { h } = fovea;
  const p = String((it && it.rel_path) || ''), file = basename(p);
  const dir = p.slice(0, Math.max(0, p.length - file.length - 1));
  const split = String((it && it.split) || '') || dir.split('/').pop();
  const [a, b] = splitTail(file, tail);           // a short name has no head: all of it stays
  return h('div', { class: `ai-pathline ${cls}`, title: p }, split ? h('span', { class: 'ai-pathline-s' }, split) : null,
    b ? h('span', { class: 'ai-pathline-h' }, a) : null, h('span', { class: 'ai-pathline-t' }, b || a));
}
/** Dates in the UI language, never the browser locale (M13). */
export function when(ts) {
  if (!ts) return '–';
  return new Date(ts * 1000).toLocaleString('en-US', { month: 'short', day: 'numeric', hour: 'numeric', minute: '2-digit' });
}
/** Defence in depth: the server scrubs secrets, but an error string must never put one on screen. */
export function scrub(msg) {
  return String(msg == null ? '' : msg).replace(/sk-ant-[\w-]+/g, 'sk-ant-…').replace(/\bsk-[\w-]{8,}/g, 'sk-…')
    .replace(/AIza[\w-]{10,}/g, 'AIza…').replace(/Bearer\s+[\w.~+/=-]+/gi, 'Bearer …');
}
/** Effort levels a cloud model is asked for, with the provider value each sends. The same three as Settings → Effort
    (low / medium / high), whose saved value is the default whenever a model of that provider is chosen. */
export const EFFORTS = [
  { value: 'fast', label: 'Fast', send: 'low', tip: 'Low reasoning effort: cheaper and quicker' },
  { value: 'balanced', label: 'Balanced', send: 'medium', tip: 'Medium effort: between Fast and Accurate in quality, time and cost' },
  { value: 'accurate', label: 'Accurate', send: 'high', tip: 'High effort: better results, slower, costs more' },
];
export const effortValue = (e) => (EFFORTS.find((x) => x.value === e || x.send === e) || EFFORTS[0]).send;
/** The effort a provider's Settings default stands for: low → Fast, medium → Balanced, high → Accurate. */
export function defaultEffort(p) {
  const e = p && p.config && p.config.effort;
  return (EFFORTS.find((x) => x.send === e || x.value === e) || EFFORTS[0]).value;
}
/** A source's effort: the one chosen for it, else its provider's default from Settings. */
export const effortOf = (P, src) => (src && EFFORTS.some((x) => x.value === src.effort) ? src.effort : defaultEffort(provOf(P, src && src.provider)));
export const effortLabel = (e) => (EFFORTS.find((x) => x.value === e || x.send === e) || EFFORTS[0]).label;
/** Fast / Balanced / Accurate. */
export function effortSeg(fovea, value, onChange) {
  const s = fovea.ui.seg(EFFORTS.map(({ value: v, label, tip }) => ({ value: v, label, tip })), value || 'fast', onChange);
  s.classList.add('ai-seg');
  return s;
}
/** A request error for people: the server's {detail, code} turned into a sentence plus whether Settings can fix it. */
export function errorInfo(e) {
  const code = e && e.body && e.body.code;
  const msg = scrub((e && (e.detail || e.message)) || 'Request failed').replace(/^\w*Error: /, '');
  if (e && e.status === 403) return { text: msg, settings: false };
  if (code === 'auth') return { text: `Key rejected — check it in Settings (${msg})`, settings: true };
  if (code === 'quota') return { text: msg, settings: true };
  if (code === 'rate') return { text: `Rate limited — try again in a moment. ${msg}` };
  if (code === 'unavailable') return { text: msg, settings: true };
  return { text: msg, settings: /api key|add a key|settings/i.test(msg) };
}

// ------------------------------------------------------------------ providers
let provCache = null, provAt = 0, provInflight = null;
let serverInfo = null, serverInflight = null;
/** Where this browser is as the server sees it (GET /api/meta → server: request_is_local, remote_ai_allowed…), asked
    once per page: the request's address does not change while the page is open. Never throws; null when the server
    does not say (an older server, a failed request — asked again next time). */
export function loadServer(fovea) {
  if (serverInfo) return Promise.resolve(serverInfo);
  if (!serverInflight) {
    serverInflight = Promise.resolve().then(() => fovea.api.get('/api/meta')).then((r) => {
      const s = r && r.server;
      if (s && typeof s.request_is_local === 'boolean') serverInfo = s;
      return serverInfo;
    }).catch(() => null).finally(() => { serverInflight = null; });
  }
  return serverInflight;
}
/** GET /api/ai/providers, cached for a minute. Never throws: failures come back as { providers: [], error }. */
export async function loadProviders(fovea, { force = false } = {}) {
  if (!force && provCache && Date.now() - provAt < 60000) return provCache;
  if (provInflight) return provInflight;
  provInflight = Promise.all([fovea.api.get('/api/ai/providers'), loadCaps(fovea), loadServer(fovea)]).then(([res, caps, server]) => {
    const P = Array.isArray(res) ? { providers: res } : { ...res };
    P.providers = (P.providers || []).filter(Boolean);
    P.caps = caps;
    // isLocalBrowser / remoteBlocked read these; null = unknown (the page's host name decides).
    P.server = server || null;
    P.request_is_local = server ? server.request_is_local : null;
    if (P.remote_ai_allowed == null && server && typeof server.remote_ai_allowed === 'boolean') P.remote_ai_allowed = server.remote_ai_allowed;
    P.checking = P.providers.some(isChecking);
    provCache = P; provAt = P.checking ? 0 : Date.now();      // a pending probe is not worth caching
    return P;
  }).catch((e) => ({ providers: [], error: e && e.status === 404 ? 'This Fovea server has no AI endpoints.' : scrub(e && e.message) || 'Could not load AI providers' }))
    .finally(() => { provInflight = null; });
  return provInflight;
}
/** While a reachability probe is pending, ask again a few times so "Checking…" turns into ready / not reachable
    without a reload. Returns a cancel function. */
export function whileChecking(fovea, P, onFresh, { tries = 5, every = 2500 } = {}) {
  let n = 0, t = null, stopped = false;
  const tick = async () => {
    if (stopped || !(P && P.checking) || n++ >= tries) return;
    t = setTimeout(async () => {
      const fresh = await loadProviders(fovea, { force: true });
      if (stopped) return;
      P = fresh;
      if (!fresh.checking) onFresh(fresh); else tick();
    }, every);
  };
  tick();
  return () => { stopped = true; clearTimeout(t); };
}
export const provOf = (P, id) => ((P && P.providers) || []).find((p) => p.id === id) || null;
export const isCloud = (p) => !!p && p.kind === 'cloud';
export const caps = (p) => (p && p.capabilities) || {};
export const canDetect = (p) => !!p && (p.capabilities ? !!p.capabilities.detect : p.id !== 'sam');
export const canClassify = (p) => !!p && (p.capabilities ? !!p.capabilities.classify : isCloud(p));
/** Can answer a free-form structured question (class naming, natural-language search). */
export const canGenerate = (p) => !!p && !!caps(p).generate;
/** Closed vocabulary (stock YOLO): model classes must be mapped onto dataset classes. */
export const isClosed = (p) => !!p && (p.capabilities && 'open_vocab' in p.capabilities ? !p.capabilities.open_vocab : p.id === 'yolo');
export const modelsOf = (p) => (p && p.models) || [];
export const modelOf = (p, id) => modelsOf(p).find((m) => m.id === id) || null;
/** Why a listed model cannot run right now ('' = it can): a classification / OBB checkpoint, or a model an
    OpenAI-compatible server (Ollama, LM Studio) lists in its config but has not pulled. */
export function modelProblem(m) {
  if (!m) return 'Unknown model';
  if (m.usable === false) return scrub(m.reason) || 'cannot draw boxes';
  if (m.pulled === false || m.installed === false) return scrub(m.reason) || 'Not pulled on the server';
  if (m.available === false) return scrub(m.reason) || 'Not available on the server';
  return '';
}
const usable = (m) => !!m && !modelProblem(m);
export function defaultModel(p) {
  const ms = modelsOf(p).filter(usable), cfg = (p && (p.default_model || (p.config && p.config.default_model))) || '';
  return (cfg && ms.find((m) => m.id === cfg)) || ms.find((m) => m.default) || ms.find((m) => m.recommended) || ms[0] || null;
}
/** The compute device local models run on ("mps", "cuda", "cpu"), '' while the runtime warms up. */
export function deviceText(P) {
  const d = P && P.device;
  if (!d) return '';
  return typeof d === 'string' ? d : d.resolved || '';
}
/** Does this provider/model report a calibrated confidence? (VLMs do not: their boxes always show.) */
export function hasConfidence(p) {
  const c = caps(p);
  return 'confidence' in c ? !!c.confidence : !isCloud(p);
}
export function modelName(p, id) { const m = modelOf(p, id); return (m && (m.name || m.id)) || id || '?'; }
/** A reachability probe that has not answered yet (OpenAI-compatible servers are probed off the request thread). */
export const isChecking = (p) => {
  const st = p && p.status;
  if (!st) return false;
  return !!(st.checking || st.state === 'checking' || (st.available == null && 'available' in st)
    || (st.available === false && /^checking\b/i.test(String(st.reason || ''))));        // "Checking 127.0.0.1:11434…"
};
/** Can this provider be used from this browser right now?
    { ok, reason, settings: reason is fixed in Settings, unreachable: a local server that does not answer, checking } */
export function providerState(P, p) {
  if (!p) return { ok: false, reason: 'Not available on this server' };
  // First: from another machine Settings cannot add a key either, so "Add key in Settings…" would lead nowhere.
  if (remoteBlocked(P, p)) return { ok: false, remote: true, reason: isCloud(p) ? 'Cloud AI is off for browsers on other machines' : 'Uses the server’s saved API key — off for browsers on other machines' };
  if (p.needs_key && !(p.key && p.key.present)) return { ok: false, reason: 'Add key in Settings…', settings: true };
  const cap = capOf(P, p);
  if (cap && cap.reached) return { ok: false, reason: capReachedText(cap), settings: true, capped: true };
  if (isChecking(p)) return { ok: false, reason: 'Checking whether the server answers…', checking: true };
  if (p.status && p.status.available === false) {
    const reason = scrub(p.status.reason) || 'Unavailable';
    return { ok: false, reason, settings: true, unreachable: p.status.reachable === false || /reach|connect|refused|timed? ?out/i.test(reason) };
  }
  const ms = modelsOf(p);
  if (!ms.filter(usable).length) {
    // Say why: every listed model is a classifier / OBB checkpoint, or none of the configured ones is pulled.
    const why = ms.length ? modelProblem(ms[0]) : '';
    return { ok: false, reason: ms.length && ms.every((m) => modelProblem(m) === why) && why !== 'cannot draw boxes' ? `No usable model — ${why.replace(/^./, (c) => c.toLowerCase())}` : 'No models available', settings: true };
  }
  return { ok: true };
}
/** Status dot for a provider list: ready / checking / off (needs a key or setup) / bad (configured but not answering). */
export function providerTone(P, p) {
  const st = providerState(P, p);
  return st.ok ? 'ready' : st.checking ? 'checking' : st.unreachable ? 'bad' : 'off';
}
export function priceText(m) {
  const pr = m && m.price;
  return pr && pr.in != null ? `${usd(pr.in)} / ${usd(pr.out)} per 1M tokens` : '';
}
export function whereText(P, p) {
  if (!p) return '';
  // Fovea's own local models run on its compute device; a local server (Ollama, LM Studio) describes itself.
  if (!isCloud(p)) return ['yolo', 'yoloe', 'sam'].includes(p.id) || !p.location ? `On this machine${deviceText(P) ? ` (${deviceText(P)})` : ''} · free` : `${p.location} · free`;
  return (p.location || `Images are sent to ${p.name}`).replace(/^./, (c) => c.toUpperCase());
}
/** Pick a usable provider for a purpose, preferring local ones (private, free). */
export function firstUsable(P, test, { cloudFirst = false } = {}) {
  const ps = ((P && P.providers) || []).filter((p) => test(p) && providerState(P, p).ok && p.id !== 'mock');
  ps.sort((a, b) => (isCloud(a) - isCloud(b)) * (cloudFirst ? -1 : 1));
  return ps[0] || ((P && P.providers) || []).find((p) => test(p) && providerState(P, p).ok) || null;
}

// ------------------------------------------------------------------ 30-day spend caps (Settings → Spend cap)
/** {provider: {cap_usd, spent_30d}} for the providers that have a cap (GET /api/ai/usage). Never throws. */
export async function loadCaps(fovea) {
  try { const r = await fovea.api.get('/api/ai/usage', { days: 1 }); return (r && r.caps && typeof r.caps === 'object') ? r.caps : {}; } catch (e) { return {}; }
}
/** A cloud provider's rolling 30-day cap: {cap, spent, left, reached} or null when none is set. */
export function capOf(P, p) {
  const c = p && isCloud(p) && P && P.caps && P.caps[p.id];
  const cap = c && c.cap_usd != null ? Number(c.cap_usd) : null;
  if (cap == null || !isFinite(cap)) return null;
  const spent = Number(c.spent_30d) || 0;
  return { cap, spent, left: Math.max(0, cap - spent), reached: cap - spent <= 0, p };
}
export const capReachedText = (c) => `The 30-day spend cap for ${c.p.name} is reached (${usd(c.spent)} of ${usd(c.cap)})`;
/** The tightest cap among the cloud providers a source calls (a two-stage pipeline may call two), or null. */
export function sourceCap(P, prof, src) {
  if (!src) return null;
  const ids = src.pipeline === 'two_stage' ? [prof && prof.twoStage && prof.twoStage.detector && prof.twoStage.detector.provider, prof && prof.twoStage && prof.twoStage.classifier && prof.twoStage.classifier.provider] : [src.provider];
  const cs = [...new Set(ids.filter(Boolean))].map((id) => capOf(P, provOf(P, id))).filter(Boolean);
  return cs.length ? cs.reduce((a, b) => (b.left < a.left ? b : a)) : null;
}
/** Does this provider have a spend cap set (whether or not it is reached)? A model with no price cannot run under one. */
export const hasCap = (P, p) => !!capOf(P, p) || !!(p && isCloud(p) && p.config && p.config.spend_cap_usd != null && p.config.spend_cap_usd !== '');
/** How an estimate's price reads: 'free' (local), 'unknown' (a cloud model with no price on file: its cost is booked
    at $0 and not tracked) or 'priced'. */
export function costKind(est, cloud) {
  if (!cloud) return 'free';
  const u = est && est.usd;
  return (est && est.priced === false) || !(Array.isArray(u) && (u[0] || u[1])) ? 'unknown' : 'priced';
}
/** The models of an estimate that have no price on file, for the sentence that says so. */
export const unpricedText = (est, fallback) => ((est && Array.isArray(est.unpriced) && est.unpriced.length) ? est.unpriced.join(', ') : fallback || 'this model');
/** A cloud run that nothing would stop (no run budget, no 30-day cap) asks first once its high estimate reaches this,
    and so does one whose cost cannot be known (no price on file) once it covers SPEND_CONFIRM_IMAGES images. */
export const SPEND_CONFIRM_USD = 5;
export const SPEND_CONFIRM_IMAGES = 20;
/** Should starting this cloud run ask first? {ask, unknown, hi} from an estimate (est) for n images with no limit. */
export function spendCheck(est, { n = 0, speed = 'now', unpriced = false } = {}) {
  if (!est) return { ask: false };          // no estimate (an older server): nothing accurate to say
  const cost = speed === 'economy' && Array.isArray(est.usd_batch) ? est.usd_batch : est.usd;
  const hi = Array.isArray(cost) ? Number(cost[1]) || 0 : 0;
  const unknown = unpriced || costKind(est, true) === 'unknown';
  const imgs = est && est.images != null ? Number(est.images) : n;
  return { ask: unknown ? imgs >= SPEND_CONFIRM_IMAGES : hi >= SPEND_CONFIRM_USD, unknown, hi, cost, images: imgs };
}
/** The confirmation for spendCheck(): the estimated range (or that the cost is unknown) and what would stop the run.
    Resolves true to go ahead. `canCap`: the form has a run budget to type instead (its Cancel says "Set a cap"). */
export function confirmSpend(fovea, chk, { label = 'this model', what = 'run', canCap = true, est = null, speed = 'now' } = {}) {
  const { fmt } = fovea;
  const imgs = fmt.plural(chk.images || 0, 'image');
  if (chk.unknown) {
    return fovea.ui.confirm({ title: `Start a ${what} whose cost is unknown?`, okLabel: 'Start anyway', cancelLabel: 'Cancel',
      message: `No price is on file for ${unpricedText(est, label)}, so Fovea can neither estimate nor count what ${imgs} cost, and no spend cap can stop it. Check the provider’s pricing first.` });
  }
  return fovea.ui.confirm({ title: `Start a cloud ${what} with no spend cap?`, okLabel: speed === 'economy' ? 'Submit without a cap' : 'Start without a cap',
    cancelLabel: canCap ? 'Set a cap' : 'Cancel',
    message: `${imgs} on ${label}: ${usdRange(chk.cost)}${speed === 'economy' ? ' at the batch price' : ''} by the estimate. Nothing stops this ${what} before it ends — ${canCap ? 'a spend cap stops it early' : 'a 30-day spend cap in Settings limits what it can spend'}.` });
}

// ------------------------------------------------------------------ per-dataset AI profile (server settings, local mirror)
const LS_PROFILES = 'fovea.ai.profiles';
let profiles = null, profilesInflight = null, saveTimer = null;
const dirtyProfiles = new Set();
const parseProfiles = (v) => {
  const out = {};
  for (const [k, x] of Object.entries(v && typeof v === 'object' ? v : {})) {
    try { out[k] = typeof x === 'string' ? JSON.parse(x) : x; } catch (e) { /* unreadable entry: start fresh */ }
  }
  return out;
};
const lsRead = () => { try { return JSON.parse(localStorage.getItem(LS_PROFILES) || '{}') || {}; } catch (e) { return {}; } };
const lsWrite = (v) => { try { localStorage.setItem(LS_PROFILES, JSON.stringify(v)); } catch (e) { /* private mode */ } };
export async function loadProfile(fovea, dsId) {
  if (!profiles) {
    profilesInflight = profilesInflight || fovea.api.get('/api/settings').then((r) => (r && r.settings && r.settings.ai_profiles) || null).catch(() => null);
    const server = await profilesInflight;
    profilesInflight = null;
    profiles = profiles || { ...lsRead(), ...parseProfiles(server) };
  }
  return profileOf(dsId);
}
export function profileOf(dsId) {
  return { source: null, classes: null, maps: {}, twoStage: null, batch: {}, audit: null, bench: null, ...((profiles && profiles[dsId]) || {}) };
}
/** Remembered per dataset on the server (so the setup follows the dataset, not the browser) and mirrored locally. */
export function patchProfile(fovea, dsId, patch) {
  profiles = profiles || lsRead();
  profiles[dsId] = { ...(profiles[dsId] || {}), ...patch };
  lsWrite(profiles);
  dirtyProfiles.add(dsId);
  clearTimeout(saveTimer);
  // Each dataset's profile is one JSON string: PUT /api/settings deep-merges objects, which would resurrect keys
  // the user removed (an unmapped class, a dropped benchmark setup).
  saveTimer = setTimeout(() => {
    const patch = {};
    dirtyProfiles.forEach((id) => { patch[id] = JSON.stringify(profiles[id] || {}); });
    dirtyProfiles.clear();
    fovea.api.put('/api/settings', { ai_profiles: patch }).catch(() => {});
  }, 500);
  fovea.bus.emit('ai:profile', { dataset_id: dsId });
  return profileOf(dsId);
}

// ------------------------------------------------------------------ class cards
const cardCache = new Map();
export async function loadCards(fovea, dsId, { force = false } = {}) {
  const c = cardCache.get(dsId);
  if (c && !force && Date.now() - c.at < 30000) return c.cards;
  try {
    const res = await fovea.api.get(`/api/datasets/${dsId}/classes/cards`);
    const cards = (res && res.cards) || {};
    cardCache.set(dsId, { at: Date.now(), cards });
    return cards;
  } catch (e) {
    if (e.status === 404 || e.status === 405) { cardCache.set(dsId, { at: Date.now(), cards: {}, missing: true }); return {}; }
    throw e;
  }
}
export const cardsMissing = (dsId) => !!(cardCache.get(dsId) || {}).missing;
export async function saveCard(fovea, dsId, cls, patch) {
  const res = await fovea.api.put(`/api/datasets/${dsId}/classes/${cls}/card`, patch);
  const c = cardCache.get(dsId) || { at: Date.now(), cards: {} };
  c.cards[cls] = { ...(c.cards[cls] || {}), ...patch, ...((res && (res.card || res)) || {}) };
  cardCache.set(dsId, c);
  fovea.bus.emit('ai:cards', { dataset_id: dsId, cls });
  return c.cards[cls];
}
/** What a model is asked to find for class c: the card prompt, else the class name. */
export function classInfo(names, cards, c) {
  const name = names[c] != null && names[c] !== '' ? String(names[c]) : `class_${c}`;
  const card = (cards && (cards[c] || cards[String(c)])) || {};
  const prompt = String(card.prompt || '').trim();
  const described = !!(prompt || String(card.description || '').trim());
  return { id: c, name, card, prompt, looksFor: prompt || name, described, placeholder: PLACEHOLDER.test(name) && !described };
}

// ------------------------------------------------------------------ closed-vocabulary class mapping
const mcCache = new Map();
/** Model class names {id: name}. Stock COCO checkpoints never load the model just to list names. */
export function peekModelClasses(p, modelId) {
  const m = modelOf(p, modelId), listed = m && (m.classes || m.names);
  if (listed) return { names: Array.isArray(listed) ? Object.fromEntries(listed.map((n, i) => [i, n])) : listed };
  if (STOCK_COCO.test(modelId || '')) return { names: Object.fromEntries(COCO80.map((n, i) => [i, n])) };
  const c = mcCache.get(`${p && p.id}:${modelId}`);
  return c && !c.then ? c : null;
}
export async function modelClassNames(fovea, p, modelId) {
  const peek = peekModelClasses(p, modelId);
  if (peek) return peek;
  const k = `${p.id}:${modelId}`;
  if (!mcCache.has(k)) {
    mcCache.set(k, fovea.api.get(`/api/ai/models/${encodeURIComponent(modelId)}/classes`)
      .then((r) => ({ names: (r && r.classes) || {} }))
      .catch((e) => ({ names: {}, error: scrub(e.message) }))
      .then((v) => { mcCache.set(k, v); return v; }));
  }
  return mcCache.get(k);
}
/** Closed vocabulary for this model: stock YOLO, or a prompt-free YOLOE checkpoint. */
export const isClosedModel = (p, model) => isClosed(p) || (!!p && p.id === 'yoloe' && /-pf$/.test(model || ''));
const mapCache = new Map();
// The last mapped count per source and explicit map, kept while a fresh answer is fetched: Propose's state and the
// picker read it without waiting.
const mapKnown = new Map();
const mapKey = (dsId, p, modelId, explicit) => JSON.stringify([dsId, p.id, modelId, explicit || {}]);
const explicitOf = (prof, p, modelId) => ((prof && prof.maps && prof.maps[`${p.id}:${modelId}`]) || {});
/** How the server will map this model's classes (name, synonym or explicit; null = skipped): POST /api/ai/mapping. */
export async function serverMap(fovea, dsId, p, modelId, explicit) {
  const key = mapKey(dsId, p, modelId, explicit);
  const hit = mapCache.get(key);
  if (hit && (hit.then || Date.now() - hit.at < 30000)) return hit;
  const pr = fovea.api.post('/api/ai/mapping', { dataset_id: dsId, provider: p.id, model: modelId, class_map: explicit && Object.keys(explicit).length ? explicit : null })
    .then((r) => ({ ok: true, at: Date.now(), items: r.items || [], mapped: r.mapped ?? (r.items || []).filter((i) => i.cls != null).length }))
    .catch((e) => ({ ok: false, at: Date.now(), missing: e.status === 404 || e.status === 405, error: scrub(e.message) }))
    .then((v) => { mapCache.set(key, v); if (v.ok) mapKnown.set(key, v.mapped); return v; });
  mapCache.set(key, pr);
  return pr;
}
export function forgetMappings() { mapCache.clear(); mapKnown.clear(); }
/** How many of your classes a closed-vocabulary model's boxes can land on, with the request fields that say so:
    {mapped, class_map, classes} (classes only when resolved here, for a server without /api/ai/mapping). */
export async function closedMapping(fovea, dsId, prof, p, modelId, names) {
  // A dataset with no classes has nothing to map onto (the server answers 422): don't ask.
  if (!names || !names.length) return { mapped: 0, class_map: null, classes: null, noClasses: true };
  const explicit = explicitOf(prof, p, modelId);
  const sm = dsId ? await serverMap(fovea, dsId, p, modelId, explicit) : { ok: false, missing: true };
  if (sm.ok) return { mapped: sm.mapped, class_map: Object.keys(explicit).length ? explicit : null, classes: null };
  const mn = await modelClassNames(fovea, p, modelId);        // an older server: resolve names here
  const map = closedMap(prof, p, modelId, mn.names, names);
  const n = map.targets.length;
  if (dsId) mapKnown.set(mapKey(dsId, p, modelId, explicit), n);
  return { mapped: n, class_map: n ? map.class_map : null, classes: n ? map.targets : null };
}
/** The last known mapped count for a closed-vocabulary source, without waiting (null: not asked yet). */
export function knownMapped(dsId, prof, p, modelId) {
  const v = p ? mapKnown.get(mapKey(dsId, p, modelId, explicitOf(prof, p, modelId))) : null;
  return v == null ? null : v;
}
/** Map rows from the server's answer: [{mc, name, target (-1 = skip), auto, how}] plus {id: name} of every model class. */
function rowsFromServer(sm, explicit) {
  const mnames = Object.fromEntries(sm.items.map((i) => [i.model_id, i.model_name]));
  const rows = sm.items.filter((i) => i.cls != null || explicit[i.model_id] != null)
    .map((i) => ({ mc: Number(i.model_id), name: i.model_name, target: i.cls != null ? Number(i.cls) : -1, auto: explicit[i.model_id] == null, how: i.how }));
  return { rows, mnames, mapped: sm.mapped };
}
/** Rows model class -> dataset class: explicit choices first, else a case-insensitive name match. target -1 = skip. */
export function closedMap(prof, p, modelId, mnames, names) {
  const key = `${p.id}:${modelId}`;
  const explicit = (prof.maps && prof.maps[key]) || {};
  const byName = new Map(names.map((n, i) => [String(n).trim().toLowerCase(), i]));
  const rows = [];
  for (const [mc, nm] of Object.entries(mnames || {})) {
    const auto = byName.has(String(nm).toLowerCase()) ? byName.get(String(nm).toLowerCase()) : null;
    if (explicit[mc] != null) rows.push({ mc: +mc, name: nm, target: +explicit[mc], auto: false });
    else if (auto != null) rows.push({ mc: +mc, name: nm, target: auto, auto: true });
  }
  for (const [mc, t] of Object.entries(explicit)) if (!mnames || !(mc in mnames)) rows.push({ mc: +mc, name: `#${mc}`, target: +t, auto: false });
  const class_map = {};
  rows.forEach((r) => { if (r.target >= 0 && r.target < names.length) class_map[r.mc] = r.target; });
  return { key, rows, class_map, targets: [...new Set(Object.values(class_map))].sort((a, b) => a - b) };
}

// ------------------------------------------------------------------ sources (what Propose / batch / benchmark run)
/** A source is {provider, model, effort, prompt_mode} or {pipeline:'two_stage'} (the dataset's saved two-stage setup). */
export function describeSource(P, prof, src) {
  if (!src) return { label: 'Choose a model', sub: '', ok: false, reason: 'No model chosen' };
  if (src.pipeline === 'two_stage') {
    const ts = prof.twoStage;
    if (!ts || !ts.detector || !ts.classifier) return { label: 'Two-stage', sub: 'Not set up yet', ok: false, reason: 'Set up two-stage in the AI tab first', twoStage: true };
    const dp = provOf(P, ts.detector.provider), cp = provOf(P, ts.classifier.provider);
    const ds = providerState(P, dp), cs = providerState(P, cp);
    const what = (ts.detector.source_classes || []).join(', ') || '?';
    return { label: 'Two-stage', sub: `${what} · ${modelName(dp, ts.detector.model)} → ${modelName(cp, ts.classifier.model)}`, twoStage: true,
      cloud: isCloud(cp) || isCloud(dp), ok: ds.ok && cs.ok, reason: !ds.ok ? `Detector: ${ds.reason}` : !cs.ok ? `Classifier: ${cs.reason}` : '',
      settings: ds.settings || cs.settings, checking: (!ds.ok && ds.checking) || (ds.ok && !cs.ok && cs.checking), dp, cp,
      where: isCloud(cp) ? `Crops are sent to ${cp.name}` : whereText(P, cp) };
  }
  const p = provOf(P, src.provider);
  const st = providerState(P, p);
  // A saved model that cannot run right now falls back to a usable one; with none usable, keep naming the saved one.
  const model = p && usable(modelOf(p, src.model)) ? src.model : (defaultModel(p) || {}).id || src.model;
  return { label: p ? modelName(p, model) : src.provider, sub: p ? p.name : 'Unknown provider', p, model, cloud: isCloud(p), ok: st.ok,
    reason: st.reason || '', settings: st.settings, checking: st.checking, where: whereText(P, p), price: priceText(modelOf(p, model)) };
}
export function sameSource(a, b) {
  if (!a || !b) return false;
  if (a.pipeline || b.pipeline) return a.pipeline === b.pipeline;
  return a.provider === b.provider && a.model === b.model;
}
/** The default for a dataset with no remembered choice: a local detector (free, private), smallest recommended model. */
export function defaultSource(P, prof) {
  if (prof.source) return prof.source;
  const p = firstUsable(P, canDetect) || ((P && P.providers) || []).find(canDetect);
  return p ? { provider: p.id, model: (defaultModel(p) || {}).id } : null;      // effort: the provider's Settings default
}
/** The same default checked against this dataset's classes: the first ready local detector that can propose something
    here — a closed-vocabulary model with a class mapped onto yours, or an open-vocabulary one with a class that has a
    real name or prompt. With none of them, defaultSource's pick (whose class-map note then says what to do). */
export async function usefulDefault(fovea, P, prof, dsId, names, cards) {
  const fallback = defaultSource(P, { ...prof, source: null });
  if (!names || !names.length) return fallback;
  const named = names.some((_, c) => !classInfo(names, cards || {}, c).placeholder);
  const local = ((P && P.providers) || []).filter((p) => canDetect(p) && !isCloud(p) && p.id !== 'mock' && providerState(P, p).ok);
  for (const p of local) {
    const model = (defaultModel(p) || {}).id;
    if (!model) continue;
    if (!isClosedModel(p, model)) { if (named) return { provider: p.id, model }; continue; }
    const cm = await closedMapping(fovea, dsId, prof, p, model, names).catch(() => null);
    if (cm && cm.mapped) return { provider: p.id, model };
  }
  return fallback;
}
/** Request body shared by propose / preview / estimate / batch / benchmark: {ok, reason, body}. */
export async function requestFor(fovea, P, prof, src, names, { conf = 0.25, requireMap = true, dsId = '' } = {}) {
  const d = describeSource(P, prof, src);
  if (!d.ok) return { ok: false, reason: d.reason, settings: d.settings };
  // Every source writes boxes of your classes; with none there is nothing to propose (and nothing to map onto).
  if (!names || !names.length) return { ok: false, fix: 'classes', reason: NO_CLASSES };
  if (src.pipeline === 'two_stage') {
    const ts = prof.twoStage;
    const dp = provOf(P, ts.detector.provider), cp = provOf(P, ts.classifier.provider);
    if (!(ts.detector.source_classes || []).length) return { ok: false, reason: 'Two-stage: choose what the detector looks for' };
    const det = { provider: dp.id, model: ts.detector.model || (defaultModel(dp) || {}).id, conf: ts.detector.conf ?? 0.25,
      source_classes: ts.detector.source_classes, options: isCloud(dp) ? { effort: 'low' } : {} };
    const cls = { provider: cp.id, model: ts.classifier.model || (defaultModel(cp) || {}).id,
      options: { pad: ts.pad ?? 0.25, ...(isCloud(cp) ? { effort: effortValue(ts.classifier.effort || defaultEffort(cp)) } : {}) } };
    const classes = (ts.classes || []).filter((c) => c < names.length);
    // Top-level provider/model name the classifier (the costed, possibly cloud part); the server reads both stages from options.
    return { ok: true, cloud: d.cloud, body: { provider: cls.provider, model: cls.model, conf: det.conf, ...(classes.length ? { classes } : {}),
      options: { pipeline: 'two_stage', detector: det, classifier: cls } } };
  }
  const p = d.p;
  const body = { provider: p.id, model: d.model, conf, options: {} };
  if (isCloud(p)) body.options.effort = effortValue(effortOf(P, src));
  if (caps(p).exemplars && src.prompt_mode === 'exemplars') body.options.prompt_mode = 'exemplars';
  if (isClosedModel(p, d.model)) {
    const cm = await closedMapping(fovea, dsId, prof, p, d.model, names);
    if (cm.class_map) body.options.class_map = cm.class_map;
    if (cm.classes) body.classes = cm.classes;
    if (!cm.mapped && requireMap) return { ok: false, fix: 'map', reason: `Map a ${modelName(p, d.model)} class to one of your classes first` };
  } else {
    const cl = (prof.classes || []).filter((c) => c < names.length);
    if (prof.classes && !cl.length) return { ok: false, reason: 'Choose at least one class to look for' };
    if (cl.length && cl.length < names.length) body.classes = cl;
  }
  return { ok: true, cloud: isCloud(p), body };
}

/** Source picker: a keyboard-navigable menu grouped "On this machine" / "Cloud" (+ the dataset's two-stage pipeline). */
export function pickSource(anchor, { fovea, P, prof, dsId, current, onPick, purpose = 'detect', allowTwoStage = true }) {
  const { h, ui, router } = fovea;
  const items = [];
  const test = purpose === 'classify' ? canClassify : purpose === 'generate' ? canGenerate : canDetect;
  const opt = (title, sub, off) => h('span', { class: `aip-opt ${off ? 'off' : ''}` }, h('span', { class: 'n' }, title), sub ? h('span', { class: 's' }, sub) : null);
  // Another model of the same provider keeps an effort chosen for it; otherwise none is stored, so the provider's
  // Settings default (Fast / Balanced / Accurate) applies — also after it is changed in Settings.
  const pickOf = (p, m) => {
    const kept = current && !current.pipeline && current.provider === p.id && current.effort;
    return { provider: p.id, model: m.id, ...(kept ? { effort: kept } : {}) };
  };
  for (const [cloud, title] of [[false, 'On this machine'], [true, 'Cloud']]) {
    const ps = ((P && P.providers) || []).filter((p) => isCloud(p) === cloud && test(p));
    if (!ps.length) continue;
    items.push({ header: true, label: title });
    for (const p of ps) {
      const st = providerState(P, p);
      if (!st.ok) {
        // A configured server that does not answer: its models are listed dimmed so people see what is set up,
        // and the one actionable row says why and leads to Settings.
        items.push({ label: opt(p.name, st.settings && !/settings/i.test(st.reason) ? `${st.reason} — open Settings` : st.reason, true), icon: st.checking ? 'clock' : st.settings ? 'settings' : 'ban', disabled: !st.settings,
          onClick: st.settings ? () => router.navigate(SETTINGS_URL) : null, title: st.reason });
        if (st.unreachable || st.checking) {
          modelsOf(p).slice(0, 4).forEach((m) => items.push({ label: opt(m.name || m.id, st.checking ? 'Waiting for the server' : 'Server not reachable', true), icon: 'ban', disabled: true }));
        }
        continue;
      }
      const ms = modelsOf(p);
      const isCur = (m) => current && !current.pipeline && current.provider === p.id && current.model === m.id;
      const shown = ms.length > 6 ? ms.filter((m) => m.recommended || m.default || isCur(m)) : ms;
      for (const m of shown) {
        // A closed-vocabulary model none of whose classes lands on this dataset's would propose nothing.
        const unmapped = dsId && isClosedModel(p, m.id) && knownMapped(dsId, prof, p, m.id) === 0;
        const sub = [ms.length > 1 || cloud ? p.name : '', m.notes, priceText(m), unmapped ? 'needs a class mapping' : ''].filter(Boolean).join(' · ');
        if (!usable(m)) { items.push({ label: opt(m.name || m.id, [ms.length > 1 || cloud ? p.name : '', modelProblem(m)].filter(Boolean).join(' · '), true), icon: 'ban', disabled: true, title: modelProblem(m) }); continue; }
        items.push({ label: opt(m.name || m.id, sub), icon: isCur(m) ? 'check' : cloud ? 'zap' : 'cpu',
          onClick: () => onPick(pickOf(p, m)) });
      }
      if (shown.length < ms.length) {
        items.push({ label: `All ${p.name} models (${ms.length})…`, icon: 'list', onClick: () => setTimeout(() => {
          const el = ui.menu(anchor, ms.map((m) => (usable(m)
            ? { label: opt(m.name || m.id, [m.notes, priceText(m), dsId && isClosedModel(p, m.id) && knownMapped(dsId, prof, p, m.id) === 0 ? 'needs a class mapping' : ''].filter(Boolean).join(' · ')), icon: isCur(m) ? 'check' : cloud ? 'zap' : 'cpu',
              onClick: () => onPick(pickOf(p, m)) }
            : { label: opt(m.name || m.id, modelProblem(m), true), icon: 'ban', disabled: true, title: modelProblem(m) })));
          el.classList.add('aip-menu');
        }, 0) });
      }
    }
  }
  if (allowTwoStage && purpose === 'detect') {
    items.push({ header: true, label: 'Pipelines' });
    const d = describeSource(P, prof, { pipeline: 'two_stage' });
    if (prof.twoStage) items.push({ label: opt('Two-stage', d.ok ? d.sub : d.reason), icon: current && current.pipeline ? 'check' : 'layers', onClick: () => onPick({ pipeline: 'two_stage' }) });
    else items.push({ label: opt('Set up two-stage…', 'Detect people, then classify each (e.g. fall / standing)'), icon: 'layers', onClick: () => router.navigate(`/d/${dsId}/ai?s=two-stage`) });
  }
  if (!items.length) items.push({ label: opt('No AI providers', (P && P.error) || 'Nothing is available on this server'), icon: 'ban', disabled: true });
  items.push({ sep: true });
  items.push({ label: 'Providers & keys…', icon: 'settings', onClick: () => router.navigate(SETTINGS_URL) });
  const el = ui.menu(anchor, items);
  el.classList.add('aip-menu');
  return el;
}

// ------------------------------------------------------------------ overlays, previews
const pos = (xc, yc, w, hh) => `left:${(xc - w / 2) * 100}%;top:${(yc - hh / 2) * 100}%;width:${w * 100}%;height:${hh * 100}%;`;
/** Ground truth thin and solid, predictions dashed with a dark label chip (readable on any image and any class colour). */
export function predOverlay(fovea, { preds = [], gt = [], names = [], labels = true, labelOf = null }) {
  const { h, colors } = fovea;
  const layer = h('div', { class: 'ov aiov' });
  for (const b of gt) layer.appendChild(h('div', { class: 'aiov-gt', style: pos(b[1], b[2], b[3], b[4]) + `border-color:${colors.classColor(b[0])}` }));
  for (const p of preds) {
    const c = colors.classColor(p.cls);
    const bx = h('div', { class: 'aiov-pr', style: pos(p.xc, p.yc, p.w, p.h) + `border-color:${c}` });
    const name = labelOf ? labelOf(p.cls) : colors.className(names, p.cls);
    if (labels) bx.appendChild(h('span', { class: 'aiov-chip', style: `border-color:${c}`, title: name }, `${p.conf != null ? Number(p.conf).toFixed(2) + ' ' : ''}${name}`));
    layer.appendChild(bx);
  }
  // Chips of overlapping boxes landed on top of each other, and a chip above a box at the top edge (or running past
  // the right edge) was cut off. Place them once the overlay has a size (a ResizeObserver callback runs after layout
  // and before paint, so nothing visibly jumps).
  if (labels && preds.length && typeof ResizeObserver === 'function') {
    const ro = new ResizeObserver(() => { if (!layer.isConnected) { ro.disconnect(); return; } declutterChips(layer); });
    ro.observe(layer);
  }
  return layer;
}
/** Measure the chips of an overlay and move each with placeChips (CSS `translate`, so a re-run starts over). */
export function declutterChips(layer) {
  const L = layer.getBoundingClientRect();
  if (!L.width || !L.height) return;
  const chips = [...layer.querySelectorAll('.aiov-chip')];
  chips.forEach((c) => { c.style.translate = ''; });
  const moves = placeChips(L, chips.map((c) => ({ r: c.getBoundingClientRect(), B: c.parentElement.getBoundingClientRect() })));
  moves.forEach(({ dx, dy }, i) => { if (dx || dy) chips[i].style.translate = `${dx}px ${dy}px`; });
}
/** Where each chip goes, from rects only: L the picture, per chip r (as drawn) and B (its box). Boxes from the top
 *  down, each chip takes the first spot inside the picture that covers no chip placed before it: above its box,
 *  inside its top edge or below it — left- or right-aligned — then one or two chip heights further out. When no
 *  spot is free, the one covering the least. Returns {dx, dy} per chip, in the order given. */
export function placeChips(L, items) {
  const placed = [], out = items.map(() => ({ dx: 0, dy: 0 }));
  const cover = (x, y, w, h) => placed.reduce((s, q) => s + Math.max(0, Math.min(x + w, q.x + q.w) - Math.max(x, q.x)) * Math.max(0, Math.min(y + h, q.y + q.h) - Math.max(y, q.y)), 0);
  const order = items.map((it, i) => i).sort((a, b) => items[a].B.top - items[b].B.top || items[a].B.left - items[b].B.left);
  for (const i of order) {
    const { r, B } = items[i], w = r.width, h = r.height;
    let best = null;
    for (const y of [r.top, B.top - h, B.top, B.bottom, B.top - 2 * h, B.top + h, B.bottom + h, B.top - 3 * h, B.top + 2 * h]) {
      for (const x of [r.left, B.right - w]) {
        if (y < L.top - 0.5 || y + h > L.bottom + 0.5 || x < L.left - 0.5 || x + w > L.right + 0.5) continue;
        const area = cover(x, y, w, h);
        if (!best || area < best.area - 0.5) best = { x, y, area };
        if (area < 0.5) break;
      }
      if (best && best.area < 0.5) break;
    }
    const at = best || { x: r.left, y: r.top };
    out[i] = { dx: Math.abs(at.x - r.left) > 0.5 ? +(at.x - r.left).toFixed(1) : 0, dy: Math.abs(at.y - r.top) > 0.5 ? +(at.y - r.top).toFixed(1) : 0 };
    placed.push({ x: at.x, y: at.y, w, h });
  }
  return out;
}
/** A thumbnail with an overlay that only appears once the pixels are decoded (visual-comfort rule 1). */
export function overlayThumb(fovea, item, overlay, { size = 256 } = {}) {
  const { h, api, media } = fovea;
  const img = h('img', { src: api.thumbUrl(item.id, size, item.mtime), alt: '', draggable: 'false', decoding: 'async' });
  const ar = item.width && item.height ? `${item.width} / ${item.height}` : '1 / 1';
  const pic = h('div', { class: 'pic aiov-pic', style: `aspect-ratio:${ar}` }, img, overlay);
  media.trackImage(img, pic);
  return pic;
}
/** Normalise POST /api/ai/preview responses to [{image_id, preds, usage, latency_ms, warnings, error}]. */
export function previewItems(res, ids) {
  const list = Array.isArray(res) ? res : (res && (res.items || res.images || res.results)) || null;
  if (list) return list.map((x, i) => ({ image_id: x.image_id ?? x.id ?? ids[i], preds: x.preds || x.detections || [], usage: x.usage || null,
    latency_ms: x.latency_ms, warnings: x.warnings || [], error: x.error ? scrub(x.error) : null }));
  if (res && typeof res === 'object') return ids.map((id) => ({ image_id: id, preds: res[id] || [], usage: null, warnings: [], error: null }));
  return [];
}
/** Image rows (with ground-truth boxes) for a set of ids, in the order given. */
export async function imagesById(fovea, dsId, ids) {
  if (!ids.length) return [];
  const res = await fovea.api.get(`/api/datasets/${dsId}/images`, { ids: ids.join(','), limit: Math.min(1000, ids.length), boxes: 1 });
  const by = new Map((res.items || []).map((it) => [it.id, it]));
  return ids.map((id) => by.get(id) || { id, rel_path: `#${id}`, boxes: [] });
}

// ------------------------------------------------------------------ jobs
const TERMINAL = ['done', 'error', 'cancelled'];
export function jobTitle(kind) {
  const k = String(kind || '');
  if (/audit/.test(k)) return 'Label audit';
  if (/bench/.test(k)) return 'Benchmark';
  if (/accept/.test(k)) return 'Accepting proposals';
  if (/reject/.test(k)) return 'Rejecting proposals';
  return 'AI labeling';
}
/** The live line of a job box: progress, counts, running cost. */
export function progressText(fovea, s) {
  const { fmt } = fovea;
  const x = s.extra || {}, parts = [];
  // Economy (provider batch): done counts images already filed; the message says how far the provider is.
  if (x.speed === 'economy') parts.push(s.total ? `${fmt.num(s.done)} of ${fmt.num(s.total)} filed${x.batch && x.batch.state ? ` · ${batchStateText(x.batch.state).toLowerCase()}` : ''}` : 'Submitting…');
  else parts.push(s.total ? `${fmt.num(s.done)} / ${fmt.num(s.total)}` : s.status === 'queued' ? 'Queued' : 'Starting…');
  if (x.flagged != null) parts.push(`${fmt.num(x.flagged)} flagged`);
  else if (x.added != null) parts.push(`${fmt.plural(x.added, 'box', 'boxes')} added`);
  else if (x.boxes != null && (x.boxes || x.speed !== 'economy')) parts.push(fmt.plural(x.boxes, 'box', 'boxes'));
  if (x.cost_usd != null && (x.cost_usd > 0 || x.budget_usd)) parts.push(x.budget_usd ? `${usd(x.cost_usd)} of ${usd(x.budget_usd)}` : usd(x.cost_usd));
  if (x.errors) parts.push(`${fmt.num(x.errors)} error${x.errors === 1 ? '' : 's'}`);
  if (x.skipped) parts.push(`${fmt.num(x.skipped)} skipped`);
  if (x.not_submitted) parts.push(`${fmt.num(x.not_submitted)} never sent`);
  return parts.join(' · ');
}
/** The job's message line under its progress line. Counts and cost are already on the progress line, so the message
    shows only what it adds: a provider wait ("Anthropic Claude rate limited — retrying in 10 s", without the counts
    the server appends to it) or a phase ("Loading models…"). */
export function jobNote(s) {
  const x = (s && s.extra) || {};
  const w = typeof x.wait === 'string' ? x.wait : '';
  if (w) return scrub(w);
  const msg = String((s && s.message) || '').trim();
  const m = /^(.*?retrying in \d+\s*s)(?=\s*(?:·|$))/i.exec(msg);
  if (m) return scrub(m[1]);
  return msg && !/^\d/.test(msg) && !/^(queued|done|starting…?)$/i.test(msg) ? scrub(msg) : '';
}
/** An Economy run resumed after Fovea stopped part-way through submitting: images whose submission was never
    recorded are NOT resent (the provider may have accepted them, and a resend could bill twice). Say so plainly. */
export function notSentText(n) {
  n = Number(n) || 0;
  if (!n) return '';
  return `${n.toLocaleString('en-US')} image${n === 1 ? ' was' : 's were'} never sent to the provider: Fovea stopped while submitting the batch, and unsent images are not resubmitted on restart (a resend could be billed twice). Run AI again on them if you need their results.`;
}
/** Static summary of a finished AI job, built from its result + extra. */
export function jobSummary(fovea, snap) {
  const { fmt } = fovea;
  const r = (snap.result && typeof snap.result === 'object' && !Array.isArray(snap.result)) ? snap.result : {};
  const x = { ...(snap.extra || {}), ...r };
  const parts = [];
  const imgs = x.images_done ?? x.images ?? (snap.status === 'done' ? snap.total : snap.done);
  if (imgs != null) parts.push(fmt.plural(imgs, 'image'));
  if (x.flagged != null) parts.push(`${fmt.num(x.flagged)} flagged`);
  if (x.added != null) parts.push(`${fmt.num(x.added)} added`, x.modified ? `${fmt.num(x.modified)} changed` : '', x.deleted ? `${fmt.num(x.deleted)} deleted` : '');
  else if (x.written) parts.push(`${fmt.plural(x.boxes || 0, 'box', 'boxes')} written to ${fmt.plural(x.written, 'file')}`);
  else if (x.boxes != null) parts.push(fmt.plural(x.boxes, 'box', 'boxes'));
  if (x.duplicates) parts.push(`${fmt.num(x.duplicates)} duplicates skipped`);
  // ai-core hook (r4): boxes over another class are never written (left as proposals), loose copies, gone findings
  if (x.conflicts) parts.push(`${fmt.num(x.conflicts)} over a box of another class left to decide in Annotate`);
  if (x.merged) parts.push(`${fmt.num(x.merged)} loose ${x.merged === 1 ? 'copy' : 'copies'} removed`);
  if (x.obsolete) parts.push(`${fmt.plural(x.obsolete, 'finding')} no longer ${x.obsolete === 1 ? 'applies' : 'apply'}`);
  if (x.stale) parts.push(`${fmt.num(x.stale)} out of date`);
  if (x.cost_usd) parts.push(usd(x.cost_usd));
  if (x.errors) parts.push(`${fmt.num(x.errors)} error${x.errors === 1 ? '' : 's'}`);
  if (x.skipped) parts.push(`${fmt.num(x.skipped)} skipped`);
  const notSent = Number(x.not_submitted) || 0;
  if (notSent) parts.push(`${fmt.num(notSent)} never sent`);
  const budget = x.stopped === 'budget' || /budget|cap/i.test(String(x.stopped || ''));
  return { text: parts.filter(Boolean).join(' · '), budget, notSent, lastError: x.last_error ? scrub(x.last_error) : '', warnings: (x.warnings || []).map(scrub), snapshot: x.snapshot_id || null };
}
/** Live box for a background AI job: progress, counts, running cost, elapsed clock, Cancel; a static summary when it ends.
    For jobs that wrote label files, the summary offers to restore the snapshot taken before the write.
    The SSE stream is closed by stop() (call it when the view unmounts). */
export function jobBox(fovea, { job, title, onEnd, actions, writes = false, dsId = job.dataset_id }) {
  const { h, ui, icon } = fovea;
  if (!title && job.extra && job.extra.speed === 'economy') title = 'AI labeling · economy';
  const el = h('div', { class: 'aij', role: 'status' });
  const bar = ui.progress(job.progress || 0, { indeterminate: !job.total });
  const line = h('span', { class: 'aij-line' }, progressText(fovea, job));
  const msg = h('div', { class: 'aij-msg' });
  const cancel = h('button', { class: 'btn btn-ghost btn-sm', type: 'button', onClick: async () => {
    cancel.disabled = true; cancel.textContent = 'Cancelling…';
    try { await fovea.api.post(`/api/jobs/${job.id}/cancel`); } catch (e) { cancel.disabled = false; cancel.textContent = 'Cancel'; ui.toast(scrub(e.message), { type: 'error' }); }
  } }, 'Cancel');
  add(el, h('div', { class: 'aij-head' }, icon('sparkles', 13), h('span', { class: 'strong grow truncate' }, title || jobTitle(job.kind)),
    ui.elapsedClock(Date.now() - (job.elapsed || 0) * 1000), cancel), bar, h('div', { class: 'aij-row' }, line), msg);
  let last = job;
  const update = (s) => {
    bar.className = `progress ${s.total ? '' : 'indeterminate'}`;
    bar.firstChild.style.width = `${Math.round((s.progress || 0) * 100)}%`;
    line.textContent = progressText(fovea, s);
    const extra = s.extra || {};
    msg.textContent = [jobNote(s), extra.last_error ? `Last error: ${scrub(extra.last_error)}` : ''].filter(Boolean).join(' · ');
  };
  update(job);
  let ended = false;
  const finish = (snap) => {
    if (ended) return;
    ended = true;
    const sum = jobSummary(fovea, snap);
    const tone = snap.status === 'done' ? (sum.budget || sum.lastError || sum.notSent ? 'warn' : 'ok') : snap.status === 'cancelled' ? 'warn' : 'danger';
    const head = snap.status === 'done' ? (sum.budget ? 'Stopped at the spend cap' : 'Done') : snap.status === 'cancelled' ? 'Cancelled' : 'Failed';
    const eco = (snap.extra && snap.extra.speed === 'economy') || (snap.result && snap.result.speed === 'economy');
    const tail = snap.status === 'cancelled' ? (eco ? 'The provider batch was cancelled; results it had already finished were filed (and are billed).' : writes ? 'Label files written before the cancel keep their changes — restore them below or from Overview → History.' : 'Results so far are kept.')
      : snap.status === 'error' ? scrub(snap.error || 'Unknown error').replace(/^\w*Error: /, '') : '';
    const undo = sum.snapshot && dsId ? h('button', { class: 'btn btn-sm btn-ghost', type: 'button', onClick: async (ev) => {
      const b = ev.currentTarget;
      if (!(await ui.confirm({ title: 'Undo these label changes?', okLabel: 'Restore', message: 'Puts the label files this job wrote back the way they were before it. A file edited since the job is kept as it is. The current state is saved as a snapshot first, so this can be undone too.' }))) return;
      b.disabled = true;
      try {
        const res = await fovea.api.post(`/api/datasets/${dsId}/snapshots/${sum.snapshot}/restore`, { mode: 'undo' });   // backend-data hook: later edits are kept
        mount(b, icon('check', 12), 'Restored');
        if (res && res.job) el.appendChild(jobBox(fovea, { job: res.job, title: 'Restoring labels', onEnd: () => notifyChanged(fovea, dsId, { labels: true }) }).el);
        else notifyChanged(fovea, dsId, { labels: true });
      } catch (e) { b.disabled = false; ui.toast(`Restore failed: ${scrub(e.message)}`, { type: 'error' }); }
    } }, icon('undo', 12), 'Undo these changes') : null;
    mount(el, h('div', { class: `aij-end ${tone}` }, icon(tone === 'ok' ? 'checkCircle' : tone === 'warn' ? 'alert' : 'alertCircle', 14),
      h('div', { class: 'grow' }, h('div', h('span', { class: 'strong' }, `${title || jobTitle(job.kind)}: ${head}`), sum.text ? ` · ${sum.text}` : ''),
        tail ? h('div', { class: 'xs muted' }, tail) : null,
        sum.lastError && snap.status !== 'error' ? h('div', { class: 'xs aip-warn' }, `Last error: ${sum.lastError}`) : null,
        sum.notSent ? h('div', { class: 'xs aip-warn' }, notSentText(sum.notSent)) : null,
        sum.warnings.length ? h('div', { class: 'xs muted' }, sum.warnings.join(' · ')) : null,
        (actions || undo) ? h('div', { class: 'row wrap gap-4 mt-8' }, actions ? actions(snap) : null, writes || sum.snapshot ? undo : null) : null)));
    onEnd && onEnd(snap);
  };
  const w = fovea.api.watchJob(job.id, (s) => { last = s; if (!ended && !TERMINAL.includes(s.status)) update(s); });
  w.then(finish).catch((e) => finish({ ...last, status: 'error', error: e.message }));
  return { el, stop: () => w.stop(), id: job.id };
}
/** Tell other views that AI work changed this dataset: proposals always; label files (counts, boxes) when written. */
export async function notifyChanged(fovea, dsId, { labels = false } = {}) {
  fovea.bus.emit('proposals:changed', { dataset: dsId });
  if (!labels) return;
  fovea.bus.emit('dataset:scanned', dsId);
  try { fovea.bus.emit('dataset:updated', await fovea.getDataset(dsId, { force: true })); } catch (e) { /* the views keep their copy */ }
}
/** Active AI jobs of a dataset (to re-attach after a reload or a tab switch), each with its run when known. */
export async function activeAiJobs(fovea, dsId) {
  const [runs, js] = await Promise.all([
    fovea.api.get('/api/ai/runs', { dataset_id: dsId, limit: 20, counts: 0 }).then((r) => r.runs || r || []).catch(() => []),   // (ai-core hook) job ids only: skip the per-run prediction counts
    fovea.api.get('/api/jobs', { dataset_id: dsId }).then((r) => r.jobs || []).catch(() => []),
  ]);
  const byJob = new Map((Array.isArray(runs) ? runs : []).filter((r) => r.job_id).map((r) => [r.job_id, r]));
  return js.filter((j) => ['queued', 'running'].includes(j.status) && (byJob.has(j.id) || /^(ai|batch|audit|bench|propos|autolabel)/.test(j.kind || '')))
    .map((j) => ({ job: j, run: byJob.get(j.id) || null }));
}

// ------------------------------------------------------------------ what a source looks for
/** Class chips (open vocabulary), a model → dataset class map (closed vocabulary) or the two-stage summary.
    Edits go straight into the dataset's AI profile, so the panel, the batch form and the AI tab agree. */
export function classesEditor(fovea, { ds, getP, getSrc, getCards, getNames, onChange, compact = false, inAnnotate = compact }) {
  const { h, icon } = fovea;
  const el = h('div', { class: 'aip-classes' });
  let gen = 0;
  // Compact (the Annotate rail): the chip list folds into one summary line until asked for. Remembered per device.
  const OPEN_KEY = 'fovea.ai.panel.classesOpen';
  let open = (() => { try { return localStorage.getItem(OPEN_KEY) === '1'; } catch (e) { return false; } })();
  const setOpen = (v) => { open = v; try { localStorage.setItem(OPEN_KEY, v ? '1' : '0'); } catch (e) { /* private mode */ } render(); };
  const changed = () => { render(); onChange && onChange(); };
  const names = () => (getNames ? getNames() : ds.classes || []);
  function chip(nm, cards, c, on, onClick) {
    const ci = classInfo(nm, cards, c);
    const text = ci.prompt && PLACEHOLDER.test(ci.name) ? ci.prompt : ci.name;
    return h(onClick ? 'button' : 'span', { class: `aip-chip ${on ? 'on' : ''} ${ci.placeholder ? 'undescribed' : ''}`, type: onClick ? 'button' : null, 'aria-pressed': onClick ? String(on) : null,
      title: `${c} · ${ci.name}${ci.prompt ? ` — looks for “${ci.prompt}”` : ''}${ci.placeholder ? ' (no description)' : ''}`, onClick },
      onClick && on ? icon('check', 10) : null, h('i', { style: `background:${fovea.colors.classColor(c)}` }), h('span', { class: 'truncate' }, text));
  }
  function render() {
    const my = ++gen;
    const P = getP(), src = getSrc(), prof = profileOf(ds.id), nm = names(), cards = (getCards && getCards()) || {};
    mount(el);
    if (!P || !src) return;
    const d = describeSource(P, prof, src);
    if (!d.ok && !d.twoStage) return;
    // Where a class is added: Annotate's own Classes list (a dialog over Annotate), else Overview. In the compact rail
    // the panel's status line under Propose already says so, with a link to the class name field.
    if (!nm.length) {
      if (!compact) add(el, h('div', { class: 'aip-hint warn' }, inAnnotate ? 'This dataset has no classes yet — add one under Classes in Annotate.'
        : ['This dataset has no classes yet. ', addClassesLink(fovea, ds.id)]));
      return;
    }
    if (d.twoStage) {
      const ts = prof.twoStage;
      if (!ts) return;
      const cl = (ts.classes && ts.classes.length ? ts.classes : nm.map((_, i) => i)).filter((c) => c < nm.length);
      add(el, h('div', { class: 'aip-lbl' }, 'Classifies into'), h('div', { class: 'aip-chips' }, cl.map((c) => chip(nm, cards, c, true, null))),
        h('div', { class: 'aip-hint' }, fovea.link(`/d/${ds.id}/ai?s=two-stage`, { class: 'aip-link' }, 'Edit two-stage setup')));
      return;
    }
    if (isClosedModel(d.p, d.model)) { renderMap(d, nm, prof, my); return; }
    const sel = prof.classes ? new Set(prof.classes) : null;
    const chips = nm.map((_, c) => chip(nm, cards, c, !sel || sel.has(c), () => {
      const cur = new Set(profileOf(ds.id).classes || nm.map((_, i) => i));
      if (cur.has(c)) cur.delete(c); else cur.add(c);
      const arr = [...cur].sort((a, b) => a - b);
      patchProfile(fovea, ds.id, { classes: arr.length === nm.length ? null : arr });
      changed();
    }));
    const undescribed = nm.map((_, c) => classInfo(nm, cards, c)).filter((ci) => ci.placeholder && (!sel || sel.has(ci.id)));
    const nSel = sel ? [...sel].filter((c) => c < nm.length).length : nm.length;
    const title = src.prompt_mode === 'exemplars' ? 'Find classes like your labels' : 'Look for';
    if (compact && !open) {
      add(el, h('div', { class: 'aip-row' }, h('span', { class: 'aip-lbl' }, title),
        h('span', { class: 'aip-sum grow truncate' }, nSel === nm.length ? `all ${nm.length} classes` : `${nSel} of ${nm.length} classes`),
        h('button', { class: 'aip-link', type: 'button', 'aria-expanded': 'false', onClick: () => setOpen(true) }, 'Change')),
        undescribed.length && src.prompt_mode !== 'exemplars' ? h('div', { class: 'aip-hint warn' }, `${undescribed.length} without a name or prompt. `,
          fovea.link(`/d/${ds.id}/ai?s=cards`, { class: 'aip-link' }, 'Describe them')) : null);
      return;
    }
    add(el, h('div', { class: 'aip-row' }, h('span', { class: 'aip-lbl grow' }, title),
      sel ? h('button', { class: 'aip-link', type: 'button', onClick: () => { patchProfile(fovea, ds.id, { classes: null }); changed(); } }, 'All') : null,
      compact ? h('button', { class: 'aip-link', type: 'button', 'aria-expanded': 'true', onClick: () => setOpen(false) }, 'Done') : null),
      h('div', { class: 'aip-chips' }, chips),
      undescribed.length && src.prompt_mode !== 'exemplars' ? h('div', { class: 'aip-hint warn' }, `${undescribed.length} class${undescribed.length === 1 ? ' has' : 'es have'} no name or prompt — the model only sees “${undescribed[0].name}”. `,
        fovea.link(`/d/${ds.id}/ai?s=cards`, { class: 'aip-link' }, 'Describe them')) : h('div', { class: 'aip-hint' }, fovea.link(`/d/${ds.id}/ai?s=cards`, { class: 'aip-link' }, 'Edit class prompts')));
  }
  function renderMap(d, nm, prof, my) {
    const p = d.p, model = d.model;
    const key = `${p.id}:${model}`;
    const explicit = (prof.maps && prof.maps[key]) || {};
    const draw = (res) => {
      if (my !== gen) return;
      const { rows, mnames, error, loading } = res;
      const targets = new Set(rows.filter((r) => r.target >= 0).map((r) => r.target));
      const setMaps = (fn) => { const maps = { ...(profileOf(ds.id).maps || {}) }; maps[key] = fn({ ...(maps[key] || {}) }); patchProfile(fovea, ds.id, { maps }); changed(); };
      const setRow = (mc, t) => setMaps((m) => ({ ...m, [mc]: t }));
      const dropRow = (mc) => setMaps((m) => { delete m[mc]; return m; });
      const howText = (r) => (r.how === 'synonym' ? 'similar name' : r.how === 'name' ? 'same name' : 'auto');
      const targetSel = (r) => h('select', { class: 'select select-sm', 'aria-label': `Dataset class for ${r.name}`, onChange: (e) => setRow(r.mc, Number(e.target.value)) },
        h('option', { value: -1, selected: r.target < 0 }, '— skip —'), nm.map((n, i) => h('option', { value: i, selected: r.target === i }, `${i} ${n}`)));
      const used = new Set(rows.map((r) => r.mc));
      const addSel = h('select', { class: 'select select-sm', 'aria-label': 'Map another model class', onChange: (e) => { const mc = Number(e.target.value); if (mc >= 0) setRow(mc, 0); } },
        h('option', { value: -1 }, Object.keys(mnames).length ? '+ Map a model class…' : loading ? 'Loading model classes…' : 'No model classes'),
        Object.entries(mnames).filter(([mc]) => !used.has(Number(mc))).map(([mc, n]) => h('option', { value: mc }, `${n}`)));
      mount(el, h('div', { class: 'aip-lbl' }, `${modelName(p, model)} classes → yours`),
        rows.length ? h('div', { class: 'aip-map' }, rows.map((r) => h('div', { class: 'aip-map-row' },
          h('span', { class: 'truncate', title: `${r.name} (model class ${r.mc})${r.auto ? ` — ${howText(r)}` : ''}` }, r.name), icon('arrowRight', 11), targetSel(r),
          r.auto ? h('span', { class: 'aip-auto', title: `Matched automatically (${howText(r)})` }, 'auto') : h('button', { class: 'btn btn-ghost btn-icon btn-sm', type: 'button', 'aria-label': `Remove mapping for ${r.name}`, onClick: () => dropRow(r.mc) }, icon('x', 11)))))
          : null,
        !targets.size && !loading ? h('div', { class: 'aip-hint warn' }, `No ${modelName(p, model)} class matches your class names, so nothing would be proposed. Map one:`) : null,
        addSel,
        error ? h('div', { class: 'aip-hint warn' }, `Model classes could not be read: ${error}`) : null,
        targets.size === 1 && nm.length > 1 ? h('div', { class: 'aip-hint' }, 'One detector class can’t tell states apart (e.g. fall vs standing). ',
          fovea.link(`/d/${ds.id}/ai?s=two-stage`, { class: 'aip-link' }, 'Use two-stage')) : null);
    };
    const local = (mn) => { const m = closedMap(prof, p, model, (mn && mn.names) || {}, nm); return { rows: m.rows, mnames: (mn && mn.names) || {}, error: mn && mn.error }; };
    const peek = peekModelClasses(p, model);
    draw(peek ? local(peek) : { rows: [], mnames: {}, loading: true });
    serverMap(fovea, ds.id, p, model, explicit).then((sm) => {
      if (sm.ok) draw(rowsFromServer(sm, explicit));
      else if (sm.missing) modelClassNames(fovea, p, model).then((mn) => draw(local(mn)));
      else draw({ ...(peek ? local(peek) : { rows: [], mnames: {} }), error: sm.error });
    });
  }
  return { el, render };
}
/** The class editor for one source, in a dialog (benchmark / audit rows that need a mapping). */
export function openClassMap(fovea, { ds, P, src, cards, onDone }) {
  const ce = classesEditor(fovea, { ds, getP: () => P, getSrc: () => src, getCards: () => cards, getNames: () => ds.classes || [] });
  ce.render();
  fovea.ui.modal({ title: `What ${describeSource(P, profileOf(ds.id), src).label} looks for`, body: fovea.h('div', { class: 'aip-dlg' }, ce.el), onClose: () => onDone && onDone(),
    footer: (api) => fovea.h('button', { class: 'btn btn-primary', type: 'button', onClick: () => api.close() }, 'Done') });
}

// ------------------------------------------------------------------ speed: now vs. the provider's batch API
/** Can this source run as a provider batch ("Economy": about half price, results within 24 h)? {ok, reason} */
export function economyFor(P, prof, src) {
  const d = describeSource(P, prof, src);
  if (!d.ok) return { ok: false, reason: d.reason };
  if (d.twoStage) return { ok: false, reason: 'Two-stage runs box by box, so it has no batch mode' };
  if (!d.cloud) return { ok: false, reason: 'Local models are free and run now' };
  if (!caps(d.p).batch_api) return { ok: false, reason: `${d.p.name} has no batch API in Fovea` };
  return { ok: true, provider: d.p };
}
/** The request flag for a provider batch (POST /api/ai/batch `speed`): the server submits ONE batch, polls it every
    minute (also after a restart) and files the results like a synchronous run. */
export function withSpeed(body, speed) {
  return speed === 'economy' ? { ...body, speed: 'economy' } : body;
}
/** Price line for an estimate: the price that applies first, the other one muted. The Economy price is named only
    where Economy can be picked (ecoOk: economyFor().ok) and the server agrees (est.economy.available) — a two-stage
    run, for one, is priced per box and has no batch mode, whatever usd_batch the estimate carries. */
export function priceLine(est, speed, ecoOk = true) {
  const ecoAvail = ecoOk && !(est && est.economy && est.economy.available === false);
  const now = usdRange(est && est.usd), eco = ecoAvail && est && Array.isArray(est.usd_batch) ? usdRange(est.usd_batch) : null;
  // ai-cloud hook: OpenAI's Flex tier bills "now" at the batch price already — Economy saves nothing more.
  const flex = !!est && est.tier === 'flex';
  if (speed === 'economy' && eco) return { main: eco, other: flex ? 'the same as now on the Flex tier' : `now ${now}`, when: 'results within 24 h' };
  if (flex) return { main: now, other: eco ? 'Flex tier; Economy costs the same' : 'Flex tier', when: null };
  return { main: now, other: eco ? `Economy ${eco}, within 24 h` : null, when: null };
}
/** ai-cloud hook: is this provider's Flex tier on (Settings → Flex tier: batch prices, slower answers)? */
export const flexOn = (p) => !!(p && p.capabilities && p.capabilities.flex && p.config && p.config.batch_mode);
/** An Economy run (provider batch API) from a runs row: {id, state, counts, done, total, submitted_at, next_check_at}
    or null. The server keeps the provider batches in params.provider_batches and the last poll in result.provider. */
export function providerBatchOf(run) {
  const p = (run && run.params) || {};
  const r = (run && run.result && typeof run.result === 'object' && !Array.isArray(run.result)) ? run.result : {};
  const bs = Array.isArray(p.provider_batches) ? p.provider_batches.filter(Boolean) : [];
  if (p.speed !== 'economy' && !p.batch_id && !bs.length) return null;
  // The last poll: result.batch (job extra.batch is the same, fresher while the job runs); older rows: result.provider.
  const j = (run && run.job && run.job.extra) || {};
  const sum = [j.batch, r.batch, r.provider].find((x) => x && typeof x === 'object' && x.state) || null;
  let counts = sum && sum.counts;
  if (!counts) { counts = {}; bs.forEach((b) => Object.entries(b.counts || {}).forEach(([k, v]) => { counts[k] = (counts[k] || 0) + (Number(v) || 0); })); }
  const state = (sum && sum.state) || (bs.length
    ? (bs.every((b) => b.ingested || ['ended', 'failed'].includes(b.state)) ? 'ended' : bs.some((b) => b.state === 'canceling') ? 'canceling' : 'running')
    : ['running', 'queued'].includes(run.status) ? 'submitting' : run.status);
  const total = bs.reduce((a, b) => a + (Number(b.n) || 0), 0) || Number(p.images) || Number(run.images) || 0;
  const done = (counts.succeeded || 0) + (counts.errored || 0) + (counts.canceled || 0) + (counts.expired || 0);
  const firsts = bs.map((b) => b.submitted_at).filter(Boolean);
  return { id: p.batch_id || (bs[0] && bs[0].id) || null, batches: bs.length, state, counts, done: Math.min(done, total || done), total,
    submitted_at: firsts.length ? Math.min(...firsts) : run.created_at, checked_at: sum && sum.checked_at, next_check_at: sum && sum.next_check_at,
    filed: (sum && sum.filed) || j.images_done || r.images_done || 0, not_submitted: Number(j.not_submitted ?? r.not_submitted) || 0 };
}
/** Is this Economy run still out at the provider (or being submitted / read)? */
export const batchOpen = (run) => !!providerBatchOf(run) && ['running', 'waiting', 'queued'].includes(run.status);
const BATCH_STATE = { running: 'Processing at the provider', in_progress: 'Processing at the provider', submitted: 'Submitted', validating: 'Submitted', submitting: 'Submitting',
  canceling: 'Cancelling at the provider', cancelling: 'Cancelling at the provider', ended: 'Results ready', ingesting: 'Saving results', failed: 'Failed at the provider', done: 'Done', cancelled: 'Cancelled', error: 'Failed' };
export const batchStateText = (s) => BATCH_STATE[s] || String(s || 'Running').replace(/_/g, ' ').replace(/^./, (c) => c.toUpperCase());

// ------------------------------------------------------------------ batch form (dialog in Annotate, inline in the AI tab)
/** For answers that are only worth showing while the settings that asked for them are unchanged (a preview, a test):
    begin() starts one and returns {signal, current(), end()}; drop() abandons the running one (its fetch is aborted)
    and makes every earlier answer stale, so a late reply can never land under newer settings. */
export function latestOnly() {
  let seq = 0, ac = null;
  return {
    begin() {
      if (ac) ac.abort();
      const my = ++seq, a = new AbortController();
      ac = a;
      return { signal: a.signal, current: () => my === seq, end: () => { if (my === seq) ac = null; } };
    },
    /** Returns whether one was still running. */
    drop() { const was = !!ac; seq++; if (ac) { ac.abort(); ac = null; } return was; },
    get running() { return !!ac; },
  };
}
export const TARGETS = [
  { value: 'proposals', label: 'Proposals', tip: 'Store as proposals to review — label files are not touched' },
  { value: 'fill', label: 'Fill empty', tip: 'Write boxes only into images without a label file (empty label files are kept as background)' },
  { value: 'append', label: 'Append', tip: 'Add boxes next to the existing ones' },
  { value: 'replace', label: 'Replace', tip: 'Delete existing boxes and write the AI boxes' },
];
export const TARGET_HINT = {
  proposals: 'Boxes are stored as proposals. Review them in Annotate (Enter accepts, Delete rejects) or accept in bulk later. Label files are not touched; an image’s earlier pending proposals are replaced.',
  // The runner skips every image that has a label file: an empty one is a deliberate background image, not a gap.
  fill: 'Writes label files only for images without one. Images that already have a label file are skipped — including empty ones, which are kept as deliberate background images.',
  append: 'Writes label files: AI boxes are added next to existing boxes. A box over one of the same class is skipped; one over a box of another class stays a proposal for you to decide.',
  replace: 'Writes label files: existing boxes on these images are deleted and replaced by the AI boxes.',
};
/**
 * The batch launcher. scopes: [{id, label: string | (() => Node), count: () => Promise<n>, resolve: () => Promise<{image_ids}|{filters}>,
 * sample: (k) => Promise<int[]>}]. Returns {el, footer(api?), destroy, start}.
 */
export function batchForm(opts) {
  const { fovea, ds, P, scopes, inModal = false, host = null, onStarted, onClose } = opts;
  const { h, ui, icon, fmt } = fovea;
  const names = ds.classes || [];
  // The dataset's AI profile (class choices, model → class maps, two-stage setup) is edited in place by the class
  // editor below and by other views: read it fresh every time it is used, never from a copy taken at build time.
  const curProf = () => profileOf(ds.id);
  let src = opts.src || defaultSource(P, curProf());
  let scopeId = opts.scope || (scopes[0] && scopes[0].id);
  const saved = curProf().batch || {};
  let target = 'proposals';
  let conf = saved.conf ?? 0.25, budget = saved.budget ?? null, keepConf = !!saved.keep_conf;
  let speed = saved.speed === 'economy' ? 'economy' : 'now';
  let started = null, destroyed = false, estSeq = 0, estTimer = null;
  // Start: `starting` is claimed before the first await, so a double click (or Enter pressed twice) while the
  // first requests are still out starts exactly one job; `ready` / `blocked` come from the model and the estimate.
  let starting = false, ready = false, blocked = false, coolUntil = 0, budgetTouched = false, unpriced = false;
  const preview = latestOnly();
  const el = h('div', { class: 'aib' });
  // The picker reads the freshest provider list (a reachability probe may have answered since the form was built).
  const srcBtn = h('button', { class: 'btn aip-pick', type: 'button', 'aria-haspopup': 'menu', onClick: () => pickSource(srcBtn, { fovea, P: (opts.getP && opts.getP()) || P, prof: curProf(), dsId: ds.id, current: src, onPick: (s) => { src = s; render(); } }) });
  const scopeBox = h('div', { class: 'aib-scopes', role: 'radiogroup', 'aria-label': 'Images' });
  const targetSeg = ui.seg(TARGETS, target, (v) => { target = v; renderTarget(); refreshEstimate(); });
  targetSeg.classList.add('ai-seg');
  const targetHint = h('div', { class: 'hint' });
  const writeWarn = h('div', { class: 'callout warn small hidden' }, icon('alert', 14), h('span', 'Writes label files. Fovea saves a snapshot first — restore it from Overview → History.'));
  const keepConfChk = h('label', { class: 'check small' }, h('input', { type: 'checkbox', checked: keepConf, onChange: (e) => { keepConf = e.target.checked; } }), 'Keep the confidence column in label files');
  const confIn = h('input', { class: 'input input-sm', type: 'number', min: 0.01, max: 0.99, step: 0.05, value: conf, style: 'width:80px', onChange: (e) => { conf = Math.min(0.99, Math.max(0.01, Number(e.target.value) || 0.25)); e.target.value = conf; invalidatePreview(); refreshEstimate(); } });
  const budgetIn = h('input', { class: 'input input-sm', type: 'number', min: 0, step: 0.5, placeholder: 'none', value: budget ?? '', style: 'width:90px', onChange: (e) => { const v = e.target.value === '' ? null : Math.max(0, Number(e.target.value)); budget = v; budgetTouched = true; refreshEstimate(); } });
  const confRow = h('div', { class: 'field' }, h('span', { class: 'label' }, 'Minimum confidence'), h('div', { class: 'row' }, confIn, h('span', { class: 'hint' }, 'Lower finds more; you reject extras in one key.')));
  const budgetHint = h('span', { class: 'hint' }, 'No new image starts once the next could pass this; one already running can go over.');
  const budgetRow = h('div', { class: 'field' }, h('span', { class: 'label' }, 'Spend cap'), h('div', { class: 'row' }, h('span', { class: 'muted' }, '$'), budgetIn, budgetHint));
  const srcNote = h('div', { class: 'hint' });
  const effortBox = h('div');
  const speedOpts = [{ value: 'now', label: 'Now', tip: 'Standard price; proposals appear while the job runs' },
    { value: 'economy', label: 'Economy · −50%', tip: 'The provider’s batch API: about half price, results within 24 h' }];
  const speedSeg = ui.seg(speedOpts, speed, (v) => { speed = v; renderSpeed(); refreshEstimate(); });
  speedSeg.classList.add('ai-seg');
  const speedHint = h('div', { class: 'hint' });
  const prevHint = h('span', { class: 'hint' }, 'Runs now at the standard price; nothing is stored — check the boxes before a long run.');
  const speedRow = h('div', { class: 'field hidden' }, h('span', { class: 'label' }, 'Speed'), speedSeg, speedHint);
  const classes = classesEditor(fovea, { ds, getP: () => P, getSrc: () => src, getCards: () => opts.cards || {}, inAnnotate: !!host, onChange: () => { invalidatePreview(); refreshEstimate(); } });
  const estEl = h('div', { class: 'aib-est', 'aria-live': 'polite' }, h('span', { class: 'hint' }, 'Estimating cost and time…'));
  // The dialog's footer repeats the estimate's first line next to Start: in a short window the form scrolls, and the
  // price must be in view whenever Start is (the full estimate, with its notes, stays in the form).
  const estSum = h('div', { class: 'aib-sum muted' }, 'Estimating…');
  const setSum = (text, tone = '') => { estSum.textContent = text || ''; estSum.title = text || ''; estSum.className = `aib-sum ${tone}`; };
  const prevBtn = h('button', { class: 'btn btn-sm', type: 'button', onClick: () => runPreview() }, icon('eye', 13), 'Preview on 3 images');
  const prevEl = h('div', { class: 'aib-prev' });
  const errEl = h('div', { class: 'aib-err', role: 'alert' });
  const body = h('div', { class: 'aib-form col gap-12' },
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model'), h('div', { class: 'row' }, srcBtn, effortBox), srcNote, classes.el),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Images'), opts.scopeExtra || null, scopeBox),
    h('div', { class: 'field' }, h('span', { class: 'label' }, 'Result goes to'), targetSeg, targetHint, writeWarn, keepConfChk),
    speedRow,
    h('div', { class: 'aib-opts' }, confRow, budgetRow),
    estEl,
    h('div', { class: 'col gap-8' }, h('div', { class: 'row' }, prevBtn, prevHint), prevEl),
    errEl);
  const progressEl = h('div', { class: 'aib-progress' });
  add(el, body, progressEl);
  // The second click of a double click (detail 2) is never a second start.
  const startBtn = h('button', { class: 'btn btn-primary', type: 'button', onClick: (e) => { if (!(e && e.detail > 1)) start(); } }, icon('sparkles', 13), 'Start');
  const closeBtn = h('button', { class: 'btn', type: 'button', onClick: () => onClose && onClose() }, inModal ? 'Cancel' : 'Reset');

  function curScope() { return scopes.find((s) => s.id === scopeId) || scopes[0]; }
  function syncStart() { startBtn.disabled = !ready || blocked || !!started || starting || Date.now() < coolUntil; }
  /** The run budget: the one saved for this dataset, but never more than what is left of the provider's 30-day spend
      cap (all the server lets a run spend), or that remainder when none was saved. A value typed here wins. */
  function applyBudget() {
    if (budgetTouched) return;
    const c = sourceCap(P, curProf(), src), sb = saved.budget ?? null;
    const v = c ? (sb != null ? Math.min(sb, c.left) : c.left) : sb;
    budget = v == null ? null : v >= 0.01 ? Math.floor(v * 100 + 1e-6) / 100 : Math.floor(v * 10000 + 1e-6) / 10000;
    budgetIn.value = budget ?? '';
  }
  /** A preview must always match the settings above it: anything that changes what it would show (model, effort,
      classes, scope, confidence) drops the one on screen and abandons one still running. */
  function invalidatePreview() {
    const running = preview.drop();
    prevBtn.disabled = false;
    mount(prevEl, running ? h('div', { class: 'hint' }, 'Settings changed, so the preview was stopped — run it again for these settings.') : null);
  }
  function render() {
    invalidatePreview();          // a new model (or a source set from outside)
    const prof = curProf();
    const d = describeSource(P, prof, src);
    mount(srcBtn, icon(d.twoStage ? 'layers' : d.cloud ? 'zap' : 'cpu', 14), h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, d.label), h('span', { class: 's truncate' }, d.sub)), icon('chevronRight', 12));
    mount(srcNote, d.ok ? [d.where, d.price ? ` · ${d.price}` : ''].join('') : h('span', { class: 'aip-bad' }, d.reason, d.settings ? [' ', fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings')] : null));
    mount(effortBox);
    if (d.ok && d.cloud && !d.twoStage) {
      effortBox.appendChild(effortSeg(fovea, effortOf(P, src), (v) => { src = { ...src, effort: v }; invalidatePreview(); refreshEstimate(); }));
    }
    // Two-stage takes its detector confidence from the setup; cloud VLMs report none.
    confRow.classList.toggle('hidden', !(d.ok && !d.twoStage && d.p && hasConfidence(d.p)));
    budgetRow.classList.toggle('hidden', !d.cloud);
    unpriced = false; budgetIn.disabled = false;
    if (d.cloud) applyBudget();
    classes.render(); renderScopes(); renderTarget(); refreshEstimate();
    ready = d.ok; blocked = false; syncStart();
  }
  function budgetHintText() { return curSpeed() === 'economy' ? 'Only as many images as this covers (at the batch price) are submitted.' : 'No new image starts once the next could pass this; one already running can go over.'; }
  /** Economy is offered only where the provider has a batch API; elsewhere the row is not shown at all. */
  function curSpeed() { return speed === 'economy' && economyFor(P, curProf(), src).ok ? 'economy' : 'now'; }
  function renderSpeed() {
    const eco = economyFor(P, curProf(), src);
    // ai-cloud hook: with the Flex tier on, "now" already costs the batch price, so Economy is no discount.
    const flex = eco.ok && flexOn(eco.provider);
    speedOpts[0].tip = flex ? 'Flex tier: batch price now, answers can take minutes' : 'Standard price; proposals appear while the job runs';
    speedOpts[1].label = flex ? 'Economy' : 'Economy · −50%';
    speedOpts[1].tip = flex ? 'The provider’s batch API: the same price as the Flex tier, results within 24 h' : 'The provider’s batch API: about half price, results within 24 h';
    prevHint.textContent = previewHint(describeSource(P, curProf(), src), flex);
    speedRow.classList.toggle('hidden', !eco.ok);
    speedSeg.setValue(curSpeed());
    budgetHint.textContent = unpriced ? 'This model has no price on file, so its cost is not tracked and a cap cannot stop it.' : budgetHintText();
    speedHint.textContent = curSpeed() === 'economy'
      ? `Sent to ${eco.provider.name} as one batch at ${flex ? 'the same price as the Flex tier' : 'about half price'}. Results usually arrive within hours (at most 24 h); Fovea checks every minute, also after a restart, and files them ${target === 'proposals' ? 'as proposals' : 'into your labels'} like a normal run. Cancel any time.`
      : (flex ? `Images are processed now on ${eco.provider.name}’s Flex tier: batch prices, but answers can take minutes. Economy gives no further discount.` : 'Images are processed now at the standard price.');
    mount(startBtn, icon(curSpeed() === 'economy' ? 'clock' : 'sparkles', 13), target === 'proposals' ? (curSpeed() === 'economy' ? 'Submit batch' : 'Start') : 'Write labels…');
  }
  /** Choose a scope. Also called by a scope's own input (typing ids, the "next N" number): the run uses the scope
      whose input people are filling in, never the radio that happened to be checked before. */
  function selectScope(id) {
    if (scopeId === id) return;
    scopeId = id;
    scopeBox.querySelectorAll('input[type=radio]').forEach((r) => { r.checked = r.value === id; });
    invalidatePreview(); refreshEstimate();
  }
  function renderScopes() {
    mount(scopeBox, ...scopes.map((s) => h('label', { class: 'check aib-scope' },
      h('input', { type: 'radio', name: `aib-scope-${ds.id}`, value: s.id, checked: s.id === scopeId, onChange: () => selectScope(s.id) }),
      typeof s.label === 'function' ? s.label(() => { if (scopeId === s.id) { invalidatePreview(); refreshEstimate(); } }, () => selectScope(s.id)) : s.label,
      s.countEl || null)));
    for (const s of scopes) if (s.count && s.countEl) s.count().then((n) => { if (s.countEl) s.countEl.textContent = n != null ? ` · ${fmt.num(n)}` : ''; }).catch(() => {});
  }
  function renderTarget() {
    targetSeg.setValue(target);
    targetHint.textContent = TARGET_HINT[target];
    writeWarn.classList.toggle('hidden', target === 'proposals');
    keepConfChk.classList.toggle('hidden', target === 'proposals');
    renderSpeed();
  }
  async function refreshEstimate() {
    clearTimeout(estTimer);
    estTimer = setTimeout(async () => {
      const seq = ++estSeq;
      const sc = curScope();
      const req = await requestFor(fovea, P, curProf(), src, names, { conf, dsId: ds.id });
      if (seq !== estSeq || destroyed) return;
      if (!req.ok) {
        mount(estEl, h('span', { class: 'aip-bad' }, req.reason, req.fix === 'classes' && !host ? [' ', addClassesLink(fovea, ds.id)] : null));
        setSum(req.reason, 'bad'); ready = false; syncStart(); return;
      }
      ready = true; syncStart();
      // The previous estimate stays until the new one is in; only a slow answer shows a waiting line.
      const slow = setTimeout(() => { if (seq === estSeq && !destroyed) { mount(estEl, h('span', { class: 'row small muted' }, ui.spinner(), 'Estimating…')); setSum('Estimating…', 'muted'); } }, 400);
      try {
        const n = sc.count ? await sc.count() : null;
        const ref = n != null ? { count: n } : await sc.resolve();
        const est = await fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, ...ref, ...req.body });
        clearTimeout(slow);
        if (seq !== estSeq || destroyed) return;
        renderEstimate(est, req);
      } catch (e) {
        clearTimeout(slow);
        if (seq !== estSeq || destroyed) return;
        mount(estEl, h('span', { class: 'hint' }, e.status === 404 ? 'No estimate available on this server.' : `No estimate: ${scrub(e.message)}`));
        setSum('No estimate', 'muted');
      }
    }, 250);
  }
  function renderEstimate(est, req) {
    const n = est.images ?? 0;
    if (!n) { mount(estEl, h('span', { class: 'aip-warn' }, 'No images in this scope — choose another.')); setSum('No images in this scope', 'warn'); ready = false; syncStart(); return; }
    ready = true; blocked = false;
    const secs = est.seconds_per_image != null ? est.seconds_per_image * n : null;
    const kind = costKind(est, req.cloud), free = kind === 'free';
    const d = describeSource(P, curProf(), src);
    // A cloud model with no price on file (one a live "Refresh list" added) is booked at $0: its cost is unknown, and
    // no run budget or spend cap could stop it — the server refuses it while a spend cap is set.
    unpriced = kind === 'unknown';
    budgetIn.disabled = unpriced;
    budgetHint.textContent = unpriced ? 'This model has no price on file, so its cost is not tracked and a cap cannot stop it.' : budgetHintText();
    // Economy's price is named only where it can be picked (not for two-stage, a local model or a provider without a
    // batch API), whatever the estimate carries.
    const sp = curSpeed(), pl = priceLine(est, sp, economyFor(P, curProf(), src).ok);
    const bits = [h('strong', `${fmt.num(n)} image${n === 1 ? '' : 's'}`)];
    bits.push(' · ', free ? 'free' : unpriced ? h('strong', 'cost unknown') : h('strong', pl.main));
    if (unpriced) bits.push(h('span', { class: 'muted' }, ` (no price on file for ${unpricedText(est, d.label)})`));
    else if (!free && pl.other) bits.push(h('span', { class: 'muted' }, ` (${pl.other})`));
    if (sp === 'economy') bits.push(' · ', pl.when);
    else if (secs) bits.push(' · ', `about ${duration(secs)}`);
    setSum([fmt.plural(n, 'image'), free ? 'free' : unpriced ? 'cost unknown' : pl.main, sp === 'economy' ? pl.when : secs ? `about ${duration(secs)}` : ''].filter(Boolean).join(' · '));
    let capNote = null;
    const c = req.cloud ? sourceCap(P, curProf(), src) : null;
    if (unpriced && (c || hasCap(P, d.p) || (d.twoStage && (hasCap(P, d.cp) || hasCap(P, d.dp))))) {
      blocked = true;
      const capped = c ? c.p : d.twoStage ? [d.cp, d.dp].find((p) => hasCap(P, p)) : d.p;
      capNote = h('div', { class: 'xs aip-bad' }, `No price on file for ${unpricedText(est, d.label)}: Fovea cannot count what it costs, so the ${capped ? capped.name : 'provider'} spend cap set in Settings could not stop it. Pick a model with a known price, or remove the spend cap. `,
        fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Open Settings'));
      setSum('No price on file — a spend cap is set, so it cannot run', 'bad');
    } else if (!free && !unpriced) {
      const cost = sp === 'economy' && Array.isArray(est.usd_batch) ? est.usd_batch : est.usd;
      // The run may spend the smaller of its own budget and what is left of the provider's 30-day cap.
      const capFirst = !!c && (!(budget > 0) || c.left <= budget + 1e-9);
      const limit = capFirst ? c.left : budget > 0 ? budget : null;
      if (limit == null && Array.isArray(cost) && cost[1] >= SPEND_CONFIRM_USD) {
        capNote = h('div', { class: 'xs aip-warn' }, `No spend cap: nothing stops this run before it ends, up to ${usd(cost[1])} by the high estimate. Set a cap above to stop it early.`);
      } else if (limit != null && Array.isArray(cost) && cost[1] > limit) {
        // A provider batch is sized before it is sent: only the images the limit covers (at the high estimate) go out.
        const lo = Math.floor(n * limit / cost[1]), hi = cost[0] > limit ? Math.floor(n * limit / cost[0]) : n;
        const left = capFirst ? `Only ${usd(c.left)} is left of the ${usd(c.cap)} 30-day spend cap for ${c.p.name}` : '';
        capNote = h('div', { class: 'xs aip-warn' }, sp === 'economy'
          ? (capFirst ? `${left}: it covers about ${fmt.num(lo)} of ${fmt.num(n)} images at the batch price — only those are submitted.`
            : `The ${usd(budget)} cap covers about ${fmt.num(lo)} of ${fmt.num(n)} images at the batch price — only those are submitted. Raise the cap to send all.`)
          : `${capFirst ? `${left}, so the job` : `The ${usd(budget)} cap`} may stop ${capFirst ? '' : 'the job '}early — after roughly ${fmt.num(lo)}${hi > lo ? `–${fmt.num(hi)}` : ''} of ${fmt.num(n)} images.`);
      }
    }
    syncStart();
    const assum = (est.assumptions || []).length ? h('div', { class: 'xs faint' }, 'Assumes ' + est.assumptions.join('; ')) : null;
    // Fill empty skips (before asking the model) every image that has a label file, empty ones included.
    const fillNote = target === 'fill' ? h('div', { class: 'xs faint' }, `Counts every image in the scope; those that already have a label file (even an empty one) are skipped${free || unpriced ? '' : ', so the run may cost less'}.`) : null;
    mount(estEl, h('div', { class: 'small' }, ...bits), capNote, fillNote, assum);
  }
  async function runPreview() {
    mount(errEl);
    const run = preview.begin();
    prevBtn.disabled = true;
    const stale = () => destroyed || !run.current();
    const req = await requestFor(fovea, P, curProf(), src, names, { conf, dsId: ds.id });
    if (stale()) return;
    if (!req.ok) { run.end(); prevBtn.disabled = false; mount(errEl, req.reason); return; }
    const t0 = Date.now();
    const model = describeSource(P, curProf(), src).label;
    mount(prevEl, h('div', { class: 'row small muted' }, ui.spinner(), 'Running on 3 images… ', ui.elapsedClock(t0)));
    try {
      const ids = (await curScope().sample(3)).slice(0, 3);
      if (stale()) return;
      if (!ids.length) { mount(prevEl, h('div', { class: 'hint' }, 'No images in this scope.')); return; }
      const [res, items] = await Promise.all([fovea.api.post('/api/ai/preview', { dataset_id: ds.id, image_ids: ids, ...req.body }, { signal: run.signal }), imagesById(fovea, ds.id, ids)]);
      if (stale()) return;
      const out = previewItems(res, ids);
      mount(prevEl, h('div', { class: 'aib-prev-grid' }, out.map((r) => {
        const it = items.find((x) => x.id === r.image_id) || { id: r.image_id, boxes: [] };
        const cost = r.usage && r.usage.cost_usd ? ` · ${usd(r.usage.cost_usd)}` : '';
        const labelOf = (c) => { const ci = classInfo(names, opts.cards || {}, c); return ci.prompt && PLACEHOLDER.test(ci.name) ? ci.prompt : ci.name; };
        return h('figure', { class: 'aib-prev-item' }, overlayThumb(fovea, it, predOverlay(fovea, { preds: r.preds, gt: it.boxes || [], names, labelOf })),
          h('figcaption', { class: 'xs' }, r.error ? h('span', { class: 'aip-bad' }, r.error)
            : [h('strong', `${r.preds.length} box${r.preds.length === 1 ? '' : 'es'}`), `${cost}${r.latency_ms ? ` · ${(r.latency_ms / 1000).toFixed(1)} s` : ''}`,
              pathLine(fovea, it, { cls: 'faint' }),
              (r.warnings || []).length ? h('div', { class: 'aip-warn' }, r.warnings.map(scrub).join('; ')) : null]));
      })), h('div', { class: 'xs faint' }, `Dashed: AI boxes (${model}) · thin solid: your current labels.`));
      reveal(prevEl);
    } catch (e) {
      if (stale() || e.name === 'AbortError') return;
      mount(prevEl, h('div', { class: 'aip-bad small' }, e.status === 404 ? 'Preview is not available on this server.' : `Preview failed: ${errorInfo(e).text}`));
    } finally {
      if (run.current()) { run.end(); prevBtn.disabled = false; }
    }
  }
  async function start() {
    if (started || starting) return;
    starting = true; syncStart();
    try { await startRun(); } finally { starting = false; if (!destroyed) syncStart(); }
  }
  async function startRun() {
    mount(errEl);
    const req = await requestFor(fovea, P, curProf(), src, names, { conf, dsId: ds.id });
    if (!req.ok) { mount(errEl, req.reason); return; }
    const sc = curScope();
    let ref;
    try { ref = await sc.resolve(); } catch (e) { mount(errEl, scrub(e.message)); return; }
    const n = ref.image_ids ? ref.image_ids.length : sc.count ? await sc.count().catch(() => null) : null;
    if (ref.image_ids && !ref.image_ids.length) { mount(errEl, 'No images in this scope.'); return; }
    const sp = curSpeed();
    // From Annotate, what was edited and decided on the open image reaches the server first: a write must not race
    // the edit, and a proposals run replaces that image's undecided proposals — decisions still queued in Annotate
    // would otherwise land on the new ones (or on nothing), and a rejection would not hold back the same box.
    if (host) {
      const st = await host.flush();
      if (st !== 'clean') { mount(errEl, 'The current image has unsaved or conflicting edits — resolve that first.'); return; }
    }
    // A cloud run that nothing would stop — no run budget and no 30-day cap (or a model with no price on file, which
    // neither can stop) — asks first when the estimate is large: one click must not spend hundreds of dollars.
    const cap = req.cloud ? sourceCap(P, curProf(), src) : null;
    const noLimit = !!req.cloud && (unpriced || (!(budget > 0) && !cap));
    // The server's dry run for exactly these images: what a write would touch, and the price for the confirmation.
    const estNow = target !== 'proposals' || noLimit ? await fovea.api.post('/api/ai/estimate', { dataset_id: ds.id, ...ref, ...req.body }).catch(() => null) : null;
    if (destroyed) return;
    const chk = noLimit ? spendCheck(estNow, { n: n || 0, speed: sp, unpriced }) : { ask: false };
    const label = describeSource(P, curProf(), src).label;
    if (target === 'proposals' && chk.ask) {
      if (!(await confirmSpend(fovea, chk, { label, canCap: !chk.unknown, est: estNow, speed: sp }))) {
        if (!chk.unknown && !destroyed) { mount(errEl, 'Not started. Enter a spend cap above, then Start.'); budgetIn.focus({ preventScroll: true }); reveal(budgetIn); }
        return;
      }
    }
    if (target !== 'proposals') {
      // Exactly what the write would touch, from the server (dry run), before anything is written.
      const t = (estNow && estNow.targets) || null;
      const nImg = t ? t.images : n;
      const what = nImg != null ? `${fmt.num(nImg)} image${nImg === 1 ? '' : 's'}` : 'these images';
      const effect = !t ? TARGET_HINT[target]
        : target === 'fill' ? fillEffect(t)
          : target === 'append' ? `AI boxes are added to ${what}; ${fmt.num(t.labeled)} of them already have ${fmt.num(t.boxes)} boxes, which stay. A box over one of the same class is skipped; one over a box of another class stays a proposal for you to decide.`
            : `${fmt.num(t.boxes)} existing box${t.boxes === 1 ? '' : 'es'} on ${fmt.num(t.labeled)} labeled image${t.labeled === 1 ? '' : 's'} are deleted and replaced by the AI boxes. Images where the model finds nothing keep their labels.`;
      if (target === 'fill' && t && !t.no_label_file) { mount(errEl, 'Every image in this scope already has a label file — Fill empty would change nothing.'); return; }
      // The spend question rides on this confirmation (one dialog, not two).
      const spend = !chk.ask ? null : chk.unknown ? `No price is on file for ${unpricedText(estNow, label)}, so what this costs is neither known nor counted.`
        : `No spend cap is set: ${usdRange(chk.cost)} by the estimate, and nothing stops the run before it ends.`;
      const ok = await ui.confirm({ title: `Write AI labels to ${what}?`, okLabel: target === 'replace' ? 'Replace labels' : 'Write labels', danger: target === 'replace',
        message: h('span', `${effect} A snapshot of the current label files is saved first, so this can be undone from the job summary or Overview → History.`,
          spend ? h('div', { class: 'aip-warn mt-8' }, spend) : null) });
      if (!ok) return;
      if (target === 'replace' && (!t || t.boxes > 0) && (nImg == null || nImg > 20)) {
        const word = nImg != null ? String(nImg) : 'REPLACE';
        const typed = await ui.prompt({ title: 'Confirm replace', label: `Type ${word} to replace the labels of ${what}.`, okLabel: 'Replace' });
        if (typed == null || typed.trim() !== word) { mount(errEl, 'Not started: the confirmation did not match.'); return; }
      }
    }
    const payload = withSpeed({ dataset_id: ds.id, ...ref, ...req.body, target }, sp);
    if (target !== 'proposals' && keepConf) payload.options = { ...payload.options, keep_conf: true };
    // No budget for a model with no price on file: its spend is booked at $0, so no budget could stop it.
    if (req.cloud && budget && !unpriced) payload.budget_usd = budget;
    try {
      const res = await fovea.api.post('/api/ai/batch', payload);
      // Only a budget typed here is remembered: a prefilled one is just what was left of the 30-day cap today.
      patchProfile(fovea, ds.id, { source: src, batch: { ...(profileOf(ds.id).batch || {}), conf, budget: budgetTouched ? budget : (saved.budget ?? null), keep_conf: keepConf, speed: sp } });
      fovea.bus.emit('ai:job', { dataset_id: ds.id, job: res.job, run_id: res.run_id, target, speed: sp });
      if (!inModal) {             // inline (AI tab): the caller shows the job; the form stays ready for the next run
        coolUntil = Date.now() + 1500;        // ...after a moment: a slow second click is not a second run
        setTimeout(() => { if (!destroyed) syncStart(); }, 1600);
        onStarted && onStarted({ ...res, target });
        return;
      }
      started = res;
      body.classList.add('hidden'); estSum.classList.add('hidden');
      const jb = jobBox(fovea, { job: res.job, title: sp === 'economy' ? 'AI labeling · economy' : 'AI labeling', writes: target !== 'proposals', actions: (snap) => reviewLinks(fovea, ds, target) });
      mount(progressEl, jb.el, h('div', { class: 'hint mt-8' }, sp === 'economy'
        ? 'You can close this dialog. The provider batch keeps going even if Fovea stops; its status and a Cancel button stay in the AI panel and in AI → Runs & cost.'
        : 'You can close this dialog — the job keeps running and stays visible in the AI panel and the AI tab.'));
      started.box = jb;
      closeBtn.textContent = 'Close';
      startBtn.classList.add('hidden');
      onStarted && onStarted({ ...res, target });
    } catch (e) {
      mount(errEl, `Could not start: ${errorInfo(e).text}`);
    }
  }
  render();
  // A fresh look at the 30-day spend caps (other runs spend too): the budget prefill and the cap note use them.
  loadCaps(fovea).then((caps) => {
    if (destroyed || JSON.stringify(caps) === JSON.stringify(P.caps || {})) return;
    P.caps = caps; render();
  });
  return {
    el, start,
    footer: () => [closeBtn, startBtn],
    /** The estimate's first line, for a dialog footer (left of Cancel / Start). */
    summary: estSum,
    destroy() { destroyed = true; clearTimeout(estTimer); preview.drop(); if (started && started.box) started.box.stop(); },
    setSource(s) { src = s; render(); },
    /** The scope that is checked, and choosing one from outside (the AI tab picks one that has images). */
    get scope() { return scopeId; },
    selectScope(id) { if (scopes.some((x) => x.id === id)) selectScope(id); },
    /** The scopes' inputs changed outside the form (split, sample size): recount and re-estimate. */
    refresh() { renderScopes(); invalidatePreview(); refreshEstimate(); },
  };
}
/** The line next to "Preview on 3 images": a price only where one applies (a local model is free). */
export function previewHint(d, flex = false) {
  const tail = 'nothing is stored — check the boxes before a long run.';
  if (d && d.ok && !d.cloud) {
    const own = d.twoStage || !d.p || ['yolo', 'yoloe', 'sam'].includes(d.p.id) || !d.p.location;
    return `${own ? 'Runs on this machine' : `Runs on ${d.p.name}, free`}; ${tail}`;
  }
  return `Runs now ${flex ? 'on the Flex tier (batch price)' : 'at the standard price'}; ${tail}`;
}
/** Fill empty, exactly: only images without a label file are written; an empty label file is a deliberate background. */
export function fillEffect(t) {
  const n = Number(t.no_label_file) || 0, have = Math.max(0, (Number(t.images) || 0) - n), empty = Number(t.empty_label_file) || 0;
  const num = (v) => Number(v).toLocaleString('en-US');
  const keep = !have ? '' : empty === have
    ? `${num(have)} with an empty label file ${have === 1 ? 'is' : 'are'} left alone (kept as deliberate background).`
    : `${num(have)} that already ${have === 1 ? 'has' : 'have'} a label file ${have === 1 ? 'is' : 'are'} left alone${empty ? `, including ${num(empty)} empty one${empty === 1 ? '' : 's'} kept as deliberate background` : ''}.`;
  return [`${num(n)} image${n === 1 ? '' : 's'} without a label file ${n === 1 ? 'gets' : 'get'} AI boxes.`, keep].filter(Boolean).join(' ');
}
/** "These image ids" (pasted from Explore or a list): ids separated by commas, spaces or new lines. */
export function idsScope(fovea, holder = { value: '' }) {
  const { h } = fovea;
  const parse = () => [...new Set(String(holder.value || '').split(/[\s,;]+/).map(Number).filter((n) => Number.isInteger(n) && n > 0))];
  const area = h('textarea', { class: 'input mono', rows: 2, placeholder: 'Image ids, e.g. 12, 57, 301', style: 'width:100%', 'aria-label': 'Image ids', onClick: (e) => e.stopPropagation() });
  area.value = holder.value || '';
  // Clicking into the box or typing ids selects this scope (select() from the form); tabbing through does not.
  return { id: 'ids', label: (changed, select = () => {}) => {
    area.onpointerdown = () => select();
    area.oninput = () => { holder.value = area.value; select(); changed(); };
    area.onchange = () => changed();
    return h('span', { class: 'col gap-4 grow' }, 'These image ids', area);
  },
    count: async () => parse().length, resolve: async () => ({ image_ids: parse() }), sample: async (k) => parse().slice(0, k) };
}
/** Where to go after a job: review proposals, or check the written labels. */
export function reviewLinks(fovea, ds, target) {
  const { link, icon } = fovea;
  if (target === 'proposals') return [
    link(`/d/${ds.id}/annotate?ai=pending`, { class: 'btn btn-sm btn-primary' }, icon('pen', 12), 'Review in Annotate'),
    link(`/d/${ds.id}/explore?ai=pending`, { class: 'btn btn-sm' }, icon('grid', 12), 'Browse in Explore')];
  return [link(`/d/${ds.id}/explore`, { class: 'btn btn-sm' }, icon('grid', 12), 'Open Explore')];
}
export function openBatchDialog(opts) {
  const { fovea } = opts;
  let form = null;
  const m = fovea.ui.modal({
    title: opts.title || 'Run AI on many images', size: 'lg', className: 'aib-modal',
    body: (api) => { form = batchForm({ ...opts, inModal: true, onClose: () => api.close() }); return form.el; },
    // The price stays in view next to Start even when the form scrolls in a short window.
    footer: () => [form.summary, ...form.footer()],
    onClose: () => form && form.destroy(),
  });
  return { modal: m, form };
}

// ------------------------------------------------------------------ provider waits (rate limits) during a synchronous Propose
/** Display names of the cloud providers a request calls (a two-stage request may call two): their waits are this request's. */
export function waitNames(P, body) {
  const o = (body && body.options) || {};
  const ids = [body && body.provider, o.detector && o.detector.provider, o.classifier && o.classifier.provider];
  return [...new Set(ids.filter(Boolean))].map((id) => provOf(P, id)).filter((p) => p && isCloud(p)).map((p) => p.name);
}
/** GET /api/ai/waits answers [{reason: 'Anthropic Claude rate limited', seconds_left}] for every wait on the server;
    the longest one of these providers. */
export function pickWait(res, names) {
  const ws = ((res && res.waits) || []).filter((w) => w && w.seconds_left > 0 && names.some((n) => String(w.reason || '').startsWith(n)));
  return ws.sort((a, b) => b.seconds_left - a.seconds_left)[0] || null;
}
export const waitSentence = (w) => `${scrub(String(w.reason || 'Waiting')).replace(/^./, (c) => c.toUpperCase())} — retrying in ${Math.max(1, Math.round(w.seconds_left))} s`;
/** The nearest ancestor that scrolls vertically (Annotate's tool column), or null. */
function scrollBox(el) {
  if (typeof getComputedStyle !== 'function') return null;
  for (let n = el && el.parentElement; n && n !== document.body && n !== document.documentElement; n = n.parentElement) {
    const oy = getComputedStyle(n).overflowY;
    if ((oy === 'auto' || oy === 'scroll') && n.scrollHeight > n.clientHeight + 1) return n;
  }
  return null;
}

// ------------------------------------------------------------------ the Annotate panel
export function mountAIPanel(el, host) {
  const { fovea } = host;
  const { h, ui, icon, fmt, keys } = fovea;
  const ds = host.dataset;
  let P = null, prof = profileOf(ds.id), cards = {}, src = null;
  let destroyed = false, jobs = [];
  // Propose runs one request per image: people move on while a cloud model answers (5–30 s), and I on the next image
  // starts that image's own request. image id -> { ac, target: {id, name}, t0, shown, waitLine }.
  const running = new Map();
  const MAX_RUNNING = 3;
  const busyHere = () => { const it = host.getItem(); return (it && running.get(it.id)) || null; };
  const unsubs = [];

  // Layout (N3): the controls people aim at — model picker, Propose / Batch, the threshold — sit at the top in rows of
  // fixed height, and the status slot under the buttons keeps its height whether it is idle, busy or showing a result
  // or an error. Everything whose height varies (effort, prompt mode, where the images go, class chips, jobs) comes
  // after them, so nothing a person is about to click moves when a line appears or disappears.
  const root = h('div', { class: 'aip' });
  const pick = h('button', { class: 'btn aip-pick', type: 'button', 'aria-haspopup': 'menu', 'data-tip': 'Choose model (⇧I)', onClick: () => openPicker() });
  const gear = fovea.link(`/d/${ds.id}/ai`, { class: 'btn btn-ghost btn-icon btn-sm', 'data-tip': 'AI tab: setup, audit, benchmark', 'aria-label': 'Open the AI tab' }, icon('settings', 13));
  const note = h('div', { class: 'aip-note' });
  const effortBox = h('div', { class: 'aip-effort' });
  const thrVal = h('span', { class: 'aip-thr-v' });
  const thr = h('input', { type: 'range', class: 'slider', min: 0.05, max: 0.95, step: 0.05, 'aria-label': 'Show proposals at or above this confidence',
    onInput: (e) => { host.setProposalThreshold(Number(e.target.value)); syncThr(); } });
  const classes = classesEditor(fovea, { ds, getP: () => P, getSrc: () => src, getCards: () => cards, compact: true,
    getNames: () => (host.classNames ? host.classNames() : ds.classes || []) });
  const proposeBtn = h('button', { class: 'btn btn-sm btn-primary aip-go', type: 'button', disabled: true, onClick: () => propose() }, icon('sparkles', 12), h('span', 'Propose'), h('span', { class: 'kbd' }, 'I'));
  const batchBtn = h('button', { class: 'btn btn-sm aip-go', type: 'button', disabled: true, onClick: () => openBatch() }, icon('layers', 12), h('span', 'Batch…'));
  const status = h('div', { class: 'aip-status', 'aria-live': 'polite' });
  const jobsEl = h('div', { class: 'aip-jobs' });
  add(root,
    h('div', { class: 'aip-top' }, pick, gear),
    h('div', { class: 'aip-actions' }, proposeBtn, batchBtn),
    status,
    h('div', { class: 'aip-thr' }, h('span', { class: 'aip-lbl' }, 'Show ≥'), thr, thrVal),
    effortBox, note, classes.el, jobsEl);
  el.appendChild(root);
  mount(pick, icon('cpu', 14), h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, 'Loading AI…'), h('span', { class: 's truncate' }, 'Providers and models')));
  let idleNote = null;                // what the status slot says when nothing is running: a hint, or why Propose is off
  let idleKey = '';                   // which reason idleNote shows ('' = none), so a result line is kept until it changes
  let unmapped = false;               // a closed-vocabulary model with no class mapped onto this dataset's
  const names = () => (host.classNames ? host.classNames() : ds.classes || []);

  function syncThr() {
    let v = Number(host.getProposalThreshold());
    if (!isFinite(v)) v = 0.25;
    thr.value = String(v);
    thrVal.textContent = v.toFixed(2);
  }
  /** Fill the fixed-height status slot. `lead` is an outcome (a result, a warning, an error); with none the slot says
      what is going on: this image's request (with its provider wait), else the idle note. A request still running
      for ANOTHER image always gets a line that names that image. Full texts stay in the title. */
  function setStatus(...lead) {
    lead = lead.filter(Boolean);
    const here = busyHere(), other = otherLine();
    const shown = !!(here && here.shown);
    // Another image's line (one line) sits first, or right under this image's own; CSS trims what follows to fit.
    const nodes = shown ? [busyLine(here), ...(lead.length ? lead : [other]), here.waitLine]
      : lead.length ? [other, ...lead]
        : other ? [other, here ? null : idleNote || msgLine('idle', null, ['Press ', ui.kbd('I'), ' to propose for this image too.'])]
          : [idleNote || idleHint()];
    mount(status, ...nodes.filter(Boolean));
    status.querySelectorAll('.aip-clamp').forEach((s) => { if (!s.title) s.title = s.textContent; });
    const idle = !shown && !lead.length;
    status.dataset.idle = idle ? '1' : '';
    if (!idle) revealStatus();
  }
  /** "Proposing… 0:04 Stop" for the image on screen. */
  function busyLine(r) {
    return h('div', { class: 'aip-msg busy here' }, ui.spinner(), h('span', { class: 'grow truncate' }, 'Proposing…'), ui.elapsedClock(r.t0),
      h('button', { class: 'aip-link', type: 'button', onClick: () => r.ac.abort() }, 'Stop'));
  }
  /** The newest request still running for another image (shown once it has run 400 ms), named. */
  function otherLine() {
    const it = host.getItem();
    const rs = [...running.values()].filter((r) => r.shown && (!it || r.target.id !== it.id));
    if (!rs.length) return null;
    const r = rs[rs.length - 1], more = rs.length > 1 ? ` +${rs.length - 1}` : '';
    // The name keeps its end (the frame number) in the narrow rail: its shared head gives way first. No clock here
    // (the rail has no room for it beside the name): that image shows its own when you go back to it.
    const name = ui.midText ? ui.midText(r.target.name, { tail: 9 }) : midEllipsis(r.target.name, 20);
    return h('div', { class: 'aip-msg busy other' }, ui.spinner(),
      h('span', { class: 'grow aip-other-t', title: `Proposing for ${r.target.name}${more ? ` and ${rs.length - 1} more` : ''} — its proposals show when you go back to it` },
        h('span', { class: 'aip-other-l' }, 'Proposing for\u00a0'), name, more ? h('span', { class: 'aip-other-l' }, more) : null),
      h('button', { class: 'aip-link', type: 'button', title: `Stop waiting for ${r.target.name}`, onClick: () => r.ac.abort() }, 'Stop'));
  }
  let revealOff = null;
  /** At 1280×720 the status slot can sit below the fold of Annotate's scrolling tool column. Bring an outcome (busy,
      result, warning, error) into view by scrolling that column only — never the page — and never while the pointer
      is in the column, so nothing moves under it: then it waits until the pointer leaves the column. */
  function revealStatus() {
    if (revealOff) { revealOff(); revealOff = null; }
    const box = scrollBox(status);
    if (!box) return;
    const go = () => {
      revealOff = null;
      if (destroyed || status.dataset.idle === '1' || !root.isConnected) return;
      const r = status.getBoundingClientRect(), b = box.getBoundingClientRect();
      if (r.bottom > b.bottom) box.scrollTop += Math.max(0, Math.min(r.bottom - b.bottom + 8, r.top - b.top - 8));
      else if (r.top < b.top) box.scrollTop -= b.top - r.top + 8;
    };
    if (box.matches(':hover')) { box.addEventListener('pointerleave', go, { once: true }); revealOff = () => box.removeEventListener('pointerleave', go); }
    else go();
  }
  const msgLine = (cls, ic, text, ...after) => h('div', { class: `aip-msg ${cls}` }, ic ? icon(ic, 13) : null, h('span', { class: 'aip-clamp' }, text), ...after);
  const settingsLink = () => fovea.link(SETTINGS_URL, { class: 'aip-link' }, 'Settings');
  function bad(text, extra) { return msgLine('bad', 'alertCircle', text, extra || null); }
  function idleHint() { return msgLine('idle', null, host.getItem() ? ['Press ', ui.kbd('I'), ' to propose boxes for this image. Nothing is saved until you accept one.'] : 'Open an image to propose boxes for it.'); }

  function render() {
    if (destroyed) return;
    prof = profileOf(ds.id);
    if (!P || P.error) {
      mount(pick, icon('ban', 14), h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, 'AI unavailable'), h('span', { class: 's truncate' }, 'Providers could not be loaded')));
      idleNote = bad((P && P.error) || 'Could not load AI providers', h('button', { class: 'aip-link', type: 'button', onClick: () => load(true) }, 'Retry'));
      idleKey = 'error'; unmapped = false;
      if (!busyHere()) setStatus();
      proposeBtn.disabled = batchBtn.disabled = true;
      mount(classes.el); mount(effortBox); mount(note);
      return;
    }
    src = src || defaultSource(P, prof);
    const d = describeSource(P, prof, src);
    mount(pick, icon(d.twoStage ? 'layers' : d.cloud ? 'zap' : 'cpu', 14),
      h('span', { class: 'aip-pick-t' }, h('span', { class: 'n truncate' }, d.label), h('span', { class: 's truncate' }, d.sub || ' ')), icon('chevronRight', 12));
    pick.title = `${d.label} — ${d.sub}`;
    syncPropose(true);
    const noConf = d.ok && d.p && !d.twoStage && !hasConfidence(d.p);
    mount(note, d.ok ? [h('span', { class: d.cloud ? 'aip-cloud' : '' }, d.where), noConf ? h('span', { class: 'aip-noconf' }, ' · This model reports no confidence, so all its boxes show.') : null] : null);
    mount(effortBox);
    if (d.ok && d.cloud && !d.twoStage) {
      const s = effortSeg(fovea, effortOf(P, src), (v) => { src = { ...src, effort: v }; patchProfile(fovea, ds.id, { source: src }); });
      s.classList.add('aip-seg');
      effortBox.appendChild(s);
    }
    if (d.ok && d.p && caps(d.p).exemplars) {
      const s = ui.seg([{ value: 'text', label: 'Text', tip: 'Prompt with class names / card prompts' }, { value: 'exemplars', label: 'Your labels', tip: 'Prompt with visual examples taken from boxes you already labeled' }],
        src.prompt_mode || 'text', (v) => { src = { ...src, prompt_mode: v }; patchProfile(fovea, ds.id, { source: src }); renderClasses(); });
      s.classList.add('ai-seg', 'aip-seg');
      effortBox.appendChild(h('div', { class: 'aip-row' }, h('span', { class: 'aip-lbl' }, 'Prompt'), s));
    }
    renderClasses();
  }

  function renderClasses() { classes.render(); }

  /** Propose's state and the idle line under the buttons, where people look when a click does nothing. Propose is off
      with the reason when the source cannot run, and also when a closed-vocabulary model (stock YOLO) has no class
      mapped onto this dataset's: it would propose nothing, and the class map that says so can sit below the fold of
      the tool column. The mapping is read from the cache at once and re-checked in the background. A result line
      stays until the reason changes (force: a new source or provider list always resets it). */
  function syncPropose(force = false) {
    if (destroyed || !P || P.error) return;
    const d = describeSource(P, prof, src);
    // An image-only folder registered for labeling has no class yet: nothing can be proposed (and there is no class
    // to map a model class onto), so say that first, with a way to add one.
    const noClasses = !names().length;
    let known = null;
    if (d.ok && !noClasses && !d.twoStage && d.p && isClosedModel(d.p, d.model)) {
      known = knownMapped(ds.id, prof, d.p, d.model);
      closedMapping(fovea, ds.id, prof, d.p, d.model, names())
        .then((cm) => { if (!destroyed && cm.mapped !== known) syncPropose(); }).catch(() => {});
    }
    unmapped = known === 0;
    idleNote = noClasses ? noClassNote()
      : !d.ok ? (d.checking ? msgLine('', 'clock', d.reason)
        : bad(d.reason, d.settings ? settingsLink() : d.twoStage && !prof.twoStage ? fovea.link(`/d/${ds.id}/ai?s=two-stage`, { class: 'aip-link' }, 'Set up') : null))
        : unmapped ? mapNote(d) : null;
    const key = noClasses ? 'classes' : !d.ok ? `off:${d.reason}` : unmapped ? `map:${d.label}` : '';
    if (!busyHere() && (force || key !== idleKey || status.dataset.idle === '1')) setStatus();
    idleKey = key;
    proposeBtn.disabled = noClasses || !d.ok || unmapped || !!busyHere() || !host.getItem();
    batchBtn.disabled = !d.ok;          // the batch dialog shows the same class map up front
  }
  /** Annotate's "New class name…" field sits in the Classes section right above this panel. */
  const newClassField = () => (typeof document.querySelector === 'function' ? document.querySelector('input[aria-label="New class name"]') : null);
  function noClassNote() {
    const n = bad('No classes yet — nothing to propose', newClassField() ? h('button', { class: 'aip-link', type: 'button', onClick: () => focusIn(newClassField()) }, 'Add one')
      : addClassesLink(fovea, ds.id, 'Add one'));
    n.querySelector('.aip-clamp').title = 'This dataset has no classes yet. Add one (or several) first; proposals are boxes of your classes.';
    return n;
  }
  // "Map one" only where the class map's select exists (it is drawn for a closed-vocabulary model with classes).
  const mapLink = () => h('button', { class: 'aip-link', type: 'button', onClick: () => focusMap() }, 'Map one');
  /** One line in the narrow rail (the picker above names the model); the whole reason in the tooltip. */
  function mapNote(d) {
    const n = bad('Needs a class mapping', mapLink());
    n.querySelector('.aip-clamp').title = `No ${d.label} class maps to one of your classes, so it would propose nothing`;
    return n;
  }
  /** Bring the class map's "+ Map a model class…" select into view (scrolling the tool column only) and focus it.
      Without one (no class to map onto, or a model with no classes) the class editor below says why. */
  function focusMap() {
    const sel = classes.el.querySelector('select[aria-label="Map another model class"]');
    if (sel) focusIn(sel);
    else focusIn(classes.el.querySelector('.aip-hint') || classes.el, { focus: false });
  }
  /** Scroll an element of the tool column into view (that column only, never the page) and focus it. */
  function focusIn(target, { focus = true } = {}) {
    if (!target) return;
    const box = scrollBox(target);
    if (box) {
      const r = target.getBoundingClientRect(), b = box.getBoundingClientRect();
      if (r.bottom > b.bottom - 8) box.scrollTop += r.bottom - b.bottom + 8;
      else if (r.top < b.top + 8) box.scrollTop -= b.top - r.top + 8;
    }
    if (focus) target.focus({ preventScroll: true });
  }

  // ---------------------------------------------------------------- Propose (I)
  async function propose() {
    if (destroyed || !P) return;
    const item = host.getItem();
    if (!item || running.has(item.id)) return;          // this image's request is out: its line says so, with Stop
    if (running.size >= MAX_RUNNING) {
      setStatus(msgLine('', 'clock', `Already proposing for ${running.size} images — press I again when one finishes.`));
      return;
    }
    // Claim the slot before the first await: a double press (or I held down) must start exactly one request.
    const ac = new AbortController();
    const target = { id: item.id, name: basename(item.rel_path) };
    const r = { ac, target, t0: Date.now(), shown: false, waitLine: null };
    running.set(item.id, r);
    proposeBtn.disabled = true;
    const onScreen = () => { const cur = host.getItem(); return !!cur && cur.id === target.id; };
    const forName = () => (onScreen() ? '' : ` for ${midEllipsis(target.name, 28)}`);
    let out = null, showTimer = null, waitTimer = null;
    try {
      // What was decided and edited on this image reaches the server first: the new run then replaces only the
      // proposals still undecided there, and the server drops new boxes that repeat ones rejected here. Sent later,
      // decisions queued in Annotate would land on the new proposals (or on nothing at all).
      const st = await host.flush();
      if (destroyed) return;
      if (st !== 'clean') { out = [bad(st === 'failed' && host.unsavedReason ? `${host.unsavedReason()}.` : `Save the edits on this image first${st === 'conflict' ? ' (resolve the conflict)' : ''}, then press I again.`)]; return; }
      const names = host.classNames ? host.classNames() : (ds.classes || []);
      const req = await requestFor(fovea, P, prof, src, names, { conf: 0.05, dsId: ds.id });
      if (destroyed) return;
      if (!req.ok) {
        out = [req.fix === 'classes' ? noClassNote() : bad(req.reason, req.fix === 'map' ? mapLink() : req.settings ? settingsLink() : null)];
        return;
      }
      const t0 = performance.now();
      // A cloud provider that rate-limits (or is overloaded) makes the server wait and retry inside this one request:
      // after 2 s, ask the server about provider waits once a second and say why it takes so long.
      const waitText = h('span', { class: 'aip-clamp' });
      r.waitLine = h('div', { class: 'aip-msg warn wait hidden' }, icon('clock', 13), waitText);
      const waitFor = waitNames(P, req.body);
      const pollWaits = async () => {
        if (running.get(target.id) !== r || destroyed) return;
        let w = null;
        try { w = pickWait(await fovea.api.get('/api/ai/waits'), waitFor); }
        catch (e) { if (e && (e.status === 404 || e.status === 405)) return; }      // an older server: no wait info
        if (running.get(target.id) !== r || destroyed) return;
        waitText.textContent = w ? waitSentence(w) : '';
        waitText.title = waitText.textContent;
        r.waitLine.classList.toggle('hidden', !w);
        waitTimer = setTimeout(pollWaits, 1000);
      };
      if (req.cloud && waitFor.length) waitTimer = setTimeout(pollWaits, 2000);
      // Nothing for the first 400 ms (fast local runs show only their result), then a static line with an elapsed
      // clock — "Proposing…" on this image, "Proposing for <name>…" once people have moved on.
      showTimer = setTimeout(() => { if (running.get(target.id) === r && !destroyed) { r.shown = true; setStatus(); } }, Math.max(0, 400 - (Date.now() - r.t0)));
      const res = await fovea.api.post('/api/ai/propose', { dataset_id: ds.id, image_id: item.id, ...req.body }, { signal: ac.signal });
      if (destroyed) return;
      const preds = res.preds || res.proposals || [];
      const ms = performance.now() - t0;            // what the user waited, network and queue included
      const thrV = Number(host.getProposalThreshold()) || 0;
      const below = preds.filter((p) => p.conf != null && p.conf < thrV).length;
      const cost = res.usage && res.usage.cost_usd ? usd(res.usage.cost_usd) : '';
      const facts = [ms < 100 ? `${Math.max(1, Math.round(ms))} ms` : `${(ms / 1000).toFixed(1)} s`, cost, res.served_by && res.served_by !== req.body.model ? `served by ${res.served_by}` : ''].filter(Boolean).join(' · ');
      const warns = (res.warnings || []).map(scrub).filter((w) => !(preds.length === 0 && /^no objects found/i.test(w)));
      const warn = warns.length ? msgLine('warn', 'alert', warns.join('; ')) : null;
      if (onScreen()) {
        host.setProposals(preds);
        const main = preds.length ? `${preds.length} proposal${preds.length === 1 ? '' : 's'}${below ? ` · ${below} below ${thrV.toFixed(2)}` : ''}` : 'Nothing found on this image';
        out = [msgLine('ok', preds.length ? 'checkCircle' : 'info', [h('strong', main), ` · ${facts}`]), warn];
      } else {
        out = [msgLine('', 'info', `${preds.length} proposal${preds.length === 1 ? '' : 's'} saved for ${midEllipsis(target.name, 28)} — they show when you go back to it.`), warn];
      }
    } catch (e) {
      if (destroyed) return;
      if (e.name === 'AbortError') out = [msgLine('', 'info', `Stopped waiting${forName()}. If the model still finishes, its proposals are saved for ${onScreen() ? 'this' : 'that'} image.`)];
      else {
        const ei = errorInfo(e);
        out = [bad(`Propose failed${forName()}: ${ei.text}`, ei.settings ? settingsLink() : null)];
      }
    } finally {
      clearTimeout(showTimer); clearTimeout(waitTimer);
      if (running.get(target.id) === r) running.delete(target.id);
      if (!destroyed) { syncPropose(); setStatus(...(out || [])); }
    }
  }

  // ---------------------------------------------------------------- Batch…
  function hostScopes() {
    let n = Number((prof.batch && prof.batch.n) || 50);
    const nIn = h('input', { class: 'input input-sm', type: 'number', min: 1, max: 5000, value: n, style: 'width:76px', 'aria-label': 'Number of images',
      onClick: (e) => e.stopPropagation() });
    let cache = null;
    const nextIds = async () => {
      const at = host.getIndex();
      if (!cache || cache.at !== at || cache.n !== n) cache = { at, n, ids: await host.idsFrom(at, n) };
      return cache.ids;
    };
    const listCount = () => fovea.api.get(`/api/datasets/${ds.id}/images`, { ...(host.getFilters() || {}), limit: 1 }).then((r) => r.total);
    const spread = (ids, k) => ids.length <= k ? ids : Array.from({ length: k }, (_, i) => ids[Math.floor(i * (ids.length - 1) / Math.max(1, k - 1))]);
    const scopes = [
      { id: 'next', label: (changed, select = () => {}) => { nIn.onpointerdown = () => select(); nIn.oninput = () => select(); nIn.onchange = () => { n = Math.max(1, Math.min(5000, Number(nIn.value) || 50)); nIn.value = n; patchProfile(fovea, ds.id, { batch: { ...(profileOf(ds.id).batch || {}), n } }); changed(); }; return h('span', { class: 'row gap-4' }, 'Next', nIn, 'images from this one'); },
        count: async () => (await nextIds()).length, resolve: async () => ({ image_ids: await nextIds() }), sample: async (k) => spread(await nextIds(), k) },
      { id: 'list', label: 'Every image in the current list', countEl: h('span', { class: 'faint' }), count: listCount,
        resolve: async () => ({ filters: host.getFilters() || {} }), sample: async (k) => spread(await host.idsFrom(0, Math.min(300, 3 * k * 20)), k) },
    ];
    const it = host.getItem();
    if (it) scopes.push({ id: 'this', label: `This image (${midEllipsis(basename(it.rel_path), 28)})`, count: async () => 1, resolve: async () => ({ image_ids: [it.id] }), sample: async () => [it.id] });
    scopes.push(idsScope(fovea));
    return scopes;
  }
  function openBatch() {
    if (!P || P.error) return;
    // Classes added in Annotate since it opened are already in the host's list.
    const cur = { ...ds, classes: host.classNames ? host.classNames() : ds.classes };
    openBatchDialog({ fovea, ds: cur, P, src, host, cards, scopes: hostScopes(), title: 'Run AI on many images',
      onStarted: (res) => attachJob(res.job, res.target) });
  }
  function openPicker() {
    if (!P || P.error) { load(true); return; }
    pickSource(pick, { fovea, P, prof, dsId: ds.id, current: src, onPick: (s) => { src = s; patchProfile(fovea, ds.id, { source: s }); render(); } });
  }
  function attachJob(job, target) {
    if (jobs.some((j) => j.id === job.id)) return;
    const box = jobBox(fovea, { job, writes: target && target !== 'proposals', actions: () => reviewLinks(fovea, ds, target || 'proposals'),
      onEnd: (snap) => {
        if (destroyed) return;
        if (snap.status !== 'error') { host.reload(); notifyChanged(fovea, ds.id, { labels: !!target && target !== 'proposals' }); }
      } });
    jobs.push(box);
    jobsEl.appendChild(box.el);
  }

  // ---------------------------------------------------------------- lifecycle
  let stopChecking = () => {};
  async function load(force = false) {
    const [p, , c] = await Promise.all([loadProviders(fovea, { force }), loadProfile(fovea, ds.id), loadCards(fovea, ds.id).catch(() => ({}))]);
    if (destroyed) return;
    prof = profileOf(ds.id);
    const saved = prof.source && describeSource(p, prof, prof.source).p !== null ? prof.source : null;
    // No remembered choice: a model that can propose something on these classes (not a stock COCO model none of
    // whose classes maps onto them, when another local model is ready).
    const s = saved || (p && !p.error ? await usefulDefault(fovea, p, prof, ds.id, names(), c).catch(() => null) : null);
    // A remembered closed-vocabulary model: know whether it maps before the first paint, so Propose does not flip.
    const d0 = s && p && !p.error ? describeSource(p, prof, s) : {};
    if (d0.ok && d0.p && !d0.twoStage && isClosedModel(d0.p, d0.model)) await closedMapping(fovea, ds.id, prof, d0.p, d0.model, names()).catch(() => null);
    if (destroyed) return;
    P = p; cards = c; prof = profileOf(ds.id); src = s;
    render();
    prefetchMaps();
    stopChecking();
    stopChecking = whileChecking(fovea, P, (fresh) => { if (!destroyed) { P = fresh; render(); } });
  }
  /** Ask how the other closed-vocabulary models in the picker map (stock checkpoints only: their names need no model
      load), so the picker can say which ones need a class mapping when it opens. */
  function prefetchMaps() {
    ((P && P.providers) || []).filter((p) => canDetect(p) && providerState(P, p).ok).forEach((p) => modelsOf(p)
      .filter((m) => usable(m) && isClosedModel(p, m.id) && peekModelClasses(p, m.id)).slice(0, 8)
      .forEach((m) => closedMapping(fovea, ds.id, prof, p, m.id, names()).catch(() => {})));
  }
  syncThr();
  setStatus();
  load(true);          // fresh per mount: keys and weights change in Settings between visits
  unsubs.push(fovea.bus.on('ai:providers', () => load(true)));
  activeAiJobs(fovea, ds.id).then((list) => { if (!destroyed) list.forEach(({ job, run }) => attachJob(job, run && run.params && run.params.target)); });
  unsubs.push(host.onImageChange(() => {
    if (destroyed) return;
    syncThr();
    setStatus();          // this image's own request, if any, and a named line for one still running elsewhere
    proposeBtn.disabled = !P || !!P.error || !names().length || !describeSource(P, prof, src).ok || unmapped || !!busyHere() || !host.getItem();
  }));
  unsubs.push(fovea.bus.on('ai:cards', (e) => { if (e && e.dataset_id === ds.id) { forgetMappings(); loadCards(fovea, ds.id).then((c) => { cards = c; renderClasses(); syncPropose(); }); } }));
  // A class mapped (here or in the batch dialog) can turn Propose on: the class map and its state follow the profile.
  unsubs.push(fovea.bus.on('ai:profile', (e) => { if (e && e.dataset_id === ds.id && !destroyed) { prof = profileOf(ds.id); renderClasses(); syncPropose(); } }));
  // The threshold also moves from Annotate's own proposal controls ({ } keys, its slider): keep this slider in step.
  unsubs.push(fovea.bus.on('annotate:threshold', () => syncThr()));
  const onKey = (e) => { if (keys.is(e, '{') || keys.is(e, '}')) setTimeout(syncThr, 0); };
  document.addEventListener('keydown', onKey);
  unsubs.push(() => document.removeEventListener('keydown', onKey));
  // Classes renamed or added: the class chips and the name-based model mapping change with them.
  unsubs.push(fovea.bus.on('dataset:updated', (d) => { if (!d || d.id === ds.id) { forgetMappings(); renderClasses(); syncPropose(); } }));

  function destroy() {
    destroyed = true;
    stopChecking();
    if (revealOff) revealOff();
    running.forEach((r) => r.ac.abort());
    jobs.forEach((j) => j.stop());
    unsubs.forEach((u) => { try { u && u(); } catch (e) { /* already gone */ } });
    root.remove();
  }
  return { destroy, propose, openBatch, openPicker };
}
