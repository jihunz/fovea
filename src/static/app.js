// Fovea app shell: sidebar, routing, view/tab registry, command palette, activity, global keys, plugins.
//
// First paint does not wait for anything it does not show: the dataset list is already being fetched by the
// inline script in app.html, the shell and sidebar render at once, Home and Settings load only their own
// module, and the other views and the plugins load in the background (a dataset page waits for them).
import { h, svg, mount, clear, fmt, cls, isTyping, MOD, debounce } from './lib/dom.js';
import { icon, logo } from './lib/icons.js';
import { createStore, createBus } from './lib/store.js';
import * as apiMod from './lib/api.js';
import { router, link } from './lib/router.js';
import * as ui from './lib/ui.js';
import * as colors from './lib/colors.js';
import { boxLayer } from './lib/overlay.js';
import { ImageCursor } from './lib/cursor.js';
import * as media from './lib/media.js';
import { comfort } from './lib/comfort.js';
import * as keys from './lib/keys.js';

const { get, post, put, patch, del, watchJob, imgUrl, thumbUrl } = apiMod;
const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

const state = createStore({ datasets: [], datasetsLoaded: false, datasetsError: null, meta: null, currentDataset: null, theme: ls.get('fovea.theme', 'system'), jobs: [] });
const bus = createBus();
const tabs = [];
const commands = [];

const JOB_KIND = {
  scan: ['Indexing', 'refresh'], qa: ['Checking data quality', 'scan'], export_copy: ['Exporting', 'download'], snapshot: ['Saving a label version', 'history'],
  restore: ['Restoring labels', 'history'], ai_batch: ['AI labeling', 'sparkles'], ai_audit: ['Label audit', 'fileSearch'], ai_benchmark: ['Benchmark', 'barChart'],
  ai_accept: ['Accepting proposals', 'check'], ai_weights: ['Downloading model', 'download'], autolabel: ['AI labeling', 'sparkles'], compare: ['Comparing', 'gitCompare'],
  compare_run: ['Running a model', 'gitCompare'], bulk_labels: ['Editing labels', 'tags'], objects_edit: ['Editing boxes', 'tags'],
};
const jobLabel = (k) => (JOB_KIND[k] || [String(k || 'Job').replace(/_/g, ' ').replace(/^./, c => c.toUpperCase())])[0];
const jobIcon = (k) => (JOB_KIND[k] || [null, 'zap'])[1];
/** "Indexing · 3,600 / 7,808" — counts when there are any, plus the job's message only when it adds something
    (messages often repeat the counts or the kind: "Indexing 6,400/7,808"). */
function jobText(j) {
  const label = jobLabel(j.kind);
  if (j.status === 'queued') return `${label} · queued${j.queue_position ? ` (#${j.queue_position})` : ''}`;
  const counts = j.total ? `${fmt.num(j.done)} / ${fmt.num(j.total)}` : '';
  let msg = (j.message || '').trim();
  if (msg.toLowerCase().startsWith(label.toLowerCase())) msg = msg.slice(label.length).replace(/^[\s:·…-]+/, '');
  // Drop a "done/total" the message repeats ("Indexing 6,400/7,808"), keep what follows ("· 38,558 boxes · $7.81") —
  // also when it sits after a provider wait ("Rate limited — retrying in 12 s · 12/100 · 30 boxes").
  if (counts) {
    const same = (m) => m.replace(/[^\d/]/g, '') === `${j.done}/${j.total}`;
    msg = msg.replace(/^[A-Za-z ]*?[\d,]+\s*\/\s*[\d,]+\s*(images?)?\s*[·:,-]?\s*/i, (m) => (same(m) ? '' : m)).trim();
    msg = msg.split(/\s+·\s+/).filter(p => !(/^[\d,]+\s*\/\s*[\d,]+(\s*images?)?$/i.test(p) && same(p))).join(' · ');
  }
  if (/^(queued|starting…?|running…?|done)$/i.test(msg)) msg = '';
  return [label, counts, msg].filter(Boolean).join(' · ');
}

export const fovea = {
  h, svg, icon, logo, fmt, cls, MOD, isTyping, debounce,
  api: { get, post, put, patch, del, watchJob, imgUrl, thumbUrl, ApiError: apiMod.ApiError, onConnection: apiMod.onConnection },
  router, link, ui, colors, boxLayer, ImageCursor, state, bus, keys,
  media,   // image loading for plugins: trackImage, createFrameCache (hold-then-swap, see docs/visual-comfort.md)
  comfort, // screen dimming, image surround, rest reminder, class palette (per device)
  config: window.FOVEA || {},
  registerTab(tab) { const i = tabs.findIndex(t => t.id === tab.id); if (i >= 0) tabs[i] = tab; else tabs.push(tab); tabs.sort((a, b) => (a.order ?? 50) - (b.order ?? 50)); },
  tabs: () => tabs.slice(),
  registerCommand(cmd) { const i = commands.findIndex(c => c.id === cmd.id); if (i >= 0) commands[i] = cmd; else commands.push(cmd); },
  commands: () => commands.slice(),
  async loadDatasets() {
    try {
      const { datasets } = await get('/api/datasets');
      setDatasets(datasets);
      return datasets;
    } catch (e) {
      state.set({ datasetsError: e.message, datasetsLoaded: true });
      throw e;
    }
  },
  async getDataset(id, { touch = false, force = false } = {}) {
    const cur = state.get().currentDataset;
    if (!force && cur && cur.id === id && !touch) return cur;
    const { dataset } = await get(`/api/datasets/${encodeURIComponent(id)}`, { touch: touch ? 1 : 0 });
    state.set({ currentDataset: dataset, datasets: state.get().datasets.map(d => d.id === id ? { ...d, ...dataset } : d) });
    return dataset;
  },
  /** The dataset G-chords and the palette act on: the open one, else the most recently opened. */
  recentDataset() {
    const s = state.get();
    if (s.currentDataset && location.pathname.startsWith('/d/')) return s.currentDataset;
    return s.datasets.slice().sort((a, b) => (b.opened_at || b.created_at || 0) - (a.opened_at || a.created_at || 0))[0] || s.currentDataset || null;
  },
  theme: {
    get: () => state.get().theme,
    set(t) { state.set({ theme: t }); ls.set('fovea.theme', t); document.documentElement.setAttribute('data-theme', t); },
    cycle() { const order = ['system', 'light', 'dark']; fovea.theme.set(order[(order.indexOf(fovea.theme.get()) + 1) % 3]); },
  },
  navigate: (to, o) => router.navigate(to, o),
  jobs: { label: jobLabel, icon: jobIcon, text: jobText, refresh: () => pollJobs(true), active: () => state.get().jobs.slice() },
};
window.fovea = fovea;

let lastDatasetsJson = '';
function setDatasets(datasets) {
  const json = JSON.stringify(datasets);
  if (json === lastDatasetsJson && state.get().datasetsLoaded) return;
  lastDatasetsJson = json;
  const cur = state.get().currentDataset;
  const fresh = cur && datasets.find(d => d.id === cur.id);
  state.set({ datasets, datasetsLoaded: true, datasetsError: null, ...(fresh ? { currentDataset: { ...cur, ...fresh } } : {}) });
}

// ------------------------------------------------------------------ layout
const appEl = document.getElementById('app');
const sidebar = h('aside', { class: 'sidebar', id: 'sidebar', 'aria-label': 'Navigation' });
const main = h('main', { class: 'main' });
const netBanner = h('div', { class: 'net-banner hidden', role: 'alert' });
const navBtn = h('button', { class: 'btn btn-ghost btn-icon', 'aria-label': 'Open navigation', 'aria-controls': 'sidebar', 'aria-expanded': 'false', onClick: () => setNav(!appEl.classList.contains('nav-open')) }, icon('menu', 18));
const mobileTitle = h('span', { class: 'mb-title' }, 'Fovea');
const mobileBar = h('div', { class: 'mobile-bar' }, navBtn, logo(20), mobileTitle, h('span', { class: 'spacer' }),
  h('button', { class: 'btn btn-ghost btn-icon', 'aria-label': 'Search', onClick: () => openPalette() }, icon('search', 16)));
const navScrim = h('div', { class: 'nav-scrim', onClick: () => setNav(false) });
const view = h('div', { id: 'view', class: 'view' });
main.appendChild(mobileBar); main.appendChild(netBanner); main.appendChild(view);
mount(appEl, sidebar, main, navScrim);
fovea.viewEl = view;
function setNav(open) {
  appEl.classList.toggle('nav-open', open);
  navBtn.setAttribute('aria-expanded', String(open));
  if (open) { const a = sidebar.querySelector('.sb-item.active') || sidebar.querySelector('.sb-item'); a && a.focus({ preventScroll: true }); }
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && appEl.classList.contains('nav-open') && !ui.hasModal()) { e.preventDefault(); e.stopImmediatePropagation(); setNav(false); navBtn.focus(); }
}, true);
new MutationObserver(() => { mobileTitle.textContent = document.title.replace(/ · Fovea$/, '') || 'Fovea'; }).observe(document.querySelector('title'), { childList: true });

let currentView = { destroy: null, key: null };
export function setView(key, node, destroy) {
  if (currentView.destroy) { try { currentView.destroy(); } catch (e) { console.error(e); } }
  currentView = { destroy, key };
  if (node !== view.firstChild || view.childNodes.length !== 1) mount(view, node);
}
fovea.setView = setView;

// ------------------------------------------------------------------ sidebar
// Built once and updated in place: a poll during a long scan must not rebuild it and drop keyboard focus.
const pins = new Set((() => { try { return JSON.parse(ls.get('fovea.pins', '[]')); } catch (e) { return []; } })());
function togglePin(id) { if (pins.has(id)) pins.delete(id); else pins.add(id); ls.set('fovea.pins', JSON.stringify([...pins])); renderSidebar(); }
const sbList = h('div', { class: 'sb-list', role: 'list' });
const sbEmpty = h('div', { class: 'faint small sb-empty' });
const activityCount = h('span', { class: 'count' });
const activityBtn = h('button', { class: 'sb-item', 'aria-haspopup': 'dialog', onClick: (e) => openActivity(e.currentTarget) }, icon('activity'), h('span', { class: 'name' }, 'Activity'), activityCount);
const themeIcon = h('span', { class: 'ico-slot' });
const themeCount = h('span', { class: 'count' });
const themeBtn = h('button', { class: 'sb-item', onClick: () => fovea.theme.cycle() }, themeIcon, h('span', { class: 'name' }, 'Theme'), themeCount);
const sbHome = link('/', { class: 'sb-item' }, icon('home'), h('span', { class: 'name' }, 'Home'));
const sbSettings = link('/settings', { class: 'sb-item' }, icon('settings'), h('span', { class: 'name' }, 'Settings'));
mount(sidebar,
  link('/', { class: 'sb-brand', 'aria-label': 'Fovea home' }, logo(24), h('span', { class: 'wordmark' }, 'fovea'), h('span', { class: 'ver' }, 'v' + (fovea.config.version || ''))),
  h('nav', { class: 'sb-section', 'aria-label': 'Main' },
    sbHome,
    h('button', { class: 'sb-item', onClick: () => openPalette() }, icon('search'), h('span', { class: 'name' }, 'Search'), h('span', { class: 'count' }, `${MOD} K`)),
    activityBtn),
  h('div', { class: 'sb-section sb-datasets' },
    h('div', { class: 'sb-title' }, h('span', { id: 'sb-ds-title' }, 'Datasets'), h('span', { class: 'spacer' }),
      h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'data-tip': 'Add dataset', 'aria-label': 'Add dataset', onClick: () => fovea.openAddDataset() }, icon('plus', 14))),
    sbList, sbEmpty),
  h('div', { class: 'sb-footer' },
    sbSettings,
    h('button', { class: 'sb-item', onClick: () => showShortcuts() }, icon('keyboard'), h('span', { class: 'name' }, 'Shortcuts'), h('span', { class: 'count' }, '?')),
    themeBtn),
);
sbList.setAttribute('aria-labelledby', 'sb-ds-title');
const sbItems = new Map();
function dotState(d) {
  if (d.reachable && d.reachable.ok === false) return ['unreachable', 'Files not found at the registered path'];
  if (d.status === 'scanning' || (d.job && ['queued', 'running'].includes(d.job.status) && d.job.kind === 'scan')) return ['scanning', 'Indexing'];
  return [d.status || 'new', { ready: 'Indexed', new: 'Not indexed yet', error: 'Indexing failed' }[d.status] || d.status];
}
function sidebarOrder(list) {
  const coll = new Intl.Collator('en', { numeric: true, sensitivity: 'base' });
  return list.slice().sort((a, b) => (pins.has(b.id) - pins.has(a.id)) || coll.compare(a.name || a.id, b.name || b.id));
}
function renderSidebar() {
  const { datasets, datasetsLoaded, datasetsError, theme } = state.get();
  const path = location.pathname;
  sbHome.classList.toggle('active', path === '/'); sbHome.toggleAttribute('aria-current', path === '/');
  sbSettings.classList.toggle('active', path === '/settings'); sbSettings.toggleAttribute('aria-current', path === '/settings');
  const seen = new Set();
  const ordered = sidebarOrder(datasets);
  ordered.forEach((d, i) => {
    seen.add(d.id);
    let it = sbItems.get(d.id);
    if (!it) {
      const dot = h('span', { class: 'dot' }), name = h('span', { class: 'name' }), count = h('span', { class: 'count' }), pin = h('span', { class: 'pin' }, icon('pin', 11));
      const a = link(`/d/${encodeURIComponent(d.id)}`, { class: 'sb-item', role: 'listitem', onContextmenu: (e) => { e.preventDefault(); ui.menu({ x: e.clientX, y: e.clientY }, [
        { label: pins.has(d.id) ? 'Unpin' : 'Pin to top', icon: 'pin', onClick: () => togglePin(d.id) },
        { label: 'Explore', icon: 'grid', onClick: () => router.navigate(`/d/${encodeURIComponent(d.id)}/explore`) },
        { label: 'Annotate', icon: 'pen', onClick: () => router.navigate(`/d/${encodeURIComponent(d.id)}/annotate`) },
      ]); } }, dot, name, pin, count);
      it = { a, dot, name, count, pin, key: '' };
      sbItems.set(d.id, it);
    }
    const [st, tip] = dotState(d);
    const key = [d.name, st, d.image_count, pins.has(d.id), d.root_host].join('|');
    if (key !== it.key) {
      it.key = key;
      it.dot.className = `dot ${st}`; it.dot.setAttribute('title', tip);
      it.name.textContent = d.name;
      it.count.textContent = fmt.compact(d.image_count);
      it.a.dataset.tip = d.root_host;
      it.pin.hidden = !pins.has(d.id);
    }
    const active = path === `/d/${encodeURIComponent(d.id)}` || path.startsWith(`/d/${encodeURIComponent(d.id)}/`);
    it.a.classList.toggle('active', active); it.a.toggleAttribute('aria-current', active);
    if (sbList.children[i] !== it.a) sbList.insertBefore(it.a, sbList.children[i] || null);
  });
  for (const [id, it] of sbItems) if (!seen.has(id)) { it.a.remove(); sbItems.delete(id); }
  sbEmpty.textContent = datasetsError ? 'Could not load datasets' : !datasetsLoaded ? '' : datasets.length ? '' : 'No datasets yet';
  sbEmpty.hidden = !sbEmpty.textContent;
  const t = theme;
  if (themeIcon.dataset.t !== t) { themeIcon.dataset.t = t; mount(themeIcon, icon(t === 'dark' ? 'moon' : t === 'light' ? 'sun' : 'monitor')); themeCount.textContent = t; themeBtn.setAttribute('aria-label', `Theme: ${t} (change)`); }
}
state.subscribe(renderSidebar, (s) => s.datasets);
state.subscribe(renderSidebar, (s) => `${s.theme}|${s.datasetsLoaded}|${s.datasetsError}`);
router.after.push(() => { renderSidebar(); setNav(false); });

// ------------------------------------------------------------------ jobs: activity + polling
// One cheap poll of the active-job list (not the whole dataset list) while anything runs; the dataset list is
// refetched only when a job starts or ends.
let jobTimer = null, lastActiveKey = '', recentJobs = [];
async function pollJobs(force = false) {
  clearTimeout(jobTimer);
  try {
    const { jobs } = await get('/api/jobs', { active: 1 });
    const key = jobs.map(j => j.id).sort().join(',');
    state.set({ jobs });
    if (key !== lastActiveKey) {
      const ended = lastActiveKey && lastActiveKey.split(',').filter(id => id && !key.includes(id));
      lastActiveKey = key;
      bus.emit('jobs:changed', { jobs, ended: ended || [] });
      fovea.loadDatasets().catch(() => {});
    }
  } catch (e) { /* the connection banner reports it */ }
  const busy = state.get().jobs.length > 0;
  jobTimer = setTimeout(pollJobs, document.hidden ? 15000 : busy ? 2500 : 5000);
  if (force) return state.get().jobs;
}
document.addEventListener('visibilitychange', () => { if (!document.hidden) pollJobs(); });
apiMod.onJobStarted(() => pollJobs());
bus.on('ai:job', () => pollJobs());
state.subscribe(() => {
  const n = state.get().jobs.length;
  activityCount.textContent = n ? String(n) : '';
  activityBtn.classList.toggle('busy', n > 0);
  if (activityPop) renderActivity();
}, (s) => s.jobs);

let activityPop = null, activityBody = null;
function renderActivity() {
  if (!activityBody || !activityBody.isConnected) return;
  const active = state.get().jobs;
  const dsName = (id) => { const d = state.get().datasets.find(x => x.id === id); return d ? d.name : id || ''; };
  const row = (j, live) => {
    const started = (j.started_at || j.created_at || Date.now() / 1000) * 1000;
    return h('div', { class: 'act-row' },
      h('div', { class: 'row' }, icon(jobIcon(j.kind), 14), h('span', { class: 'grow truncate strong' }, jobLabel(j.kind)),
        live ? ui.elapsedClock(started) : h('span', { class: `xs act-st ${j.status}` }, j.status === 'done' ? `done · ${fmt.ago(j.finished_at || j.updated_at)}` : j.status)),
      h('div', { class: 'row xs muted' }, j.dataset_id ? link(`/d/${encodeURIComponent(j.dataset_id)}`, { class: 'act-ds truncate' }, dsName(j.dataset_id)) : null,
        h('span', { class: 'grow truncate' }, live ? jobText(j).replace(/^[^·]*·\s*/, '') : (j.error || j.message || ''))),
      live ? h('div', { class: 'row' }, ui.progress(j.progress || 0, { indeterminate: !j.total }), h('button', { class: 'btn btn-sm btn-ghost', onClick: async (e) => {
        e.currentTarget.disabled = true;
        try { await post(`/api/jobs/${j.id}/cancel`); pollJobs(); } catch (err) { ui.toast(err.message, { type: 'error' }); }
      } }, 'Cancel')) : null);
  };
  mount(activityBody,
    active.length ? active.map(j => row(j, true)) : h('div', { class: 'small faint act-empty' }, 'Nothing running.'),
    recentJobs.length ? h('div', { class: 'menu-label' }, 'Recent') : null,
    recentJobs.slice(0, 8).map(j => row(j, false)));
}
async function openActivity(anchor) {
  activityBody = h('div', { class: 'act-body' });
  activityPop = ui.popover(anchor, h('div', { class: 'col gap-8' }, h('div', { class: 'row' }, h('span', { class: 'strong' }, 'Activity'), h('span', { class: 'spacer' }),
    h('span', { class: 'xs faint' }, 'jobs on this server')), activityBody), { width: 340, onClose: () => { activityPop = null; } });
  renderActivity();
  try {
    const { jobs } = await get('/api/jobs');
    recentJobs = jobs.filter(j => !['queued', 'running'].includes(j.status));
    renderActivity();
  } catch (e) { /* offline: the banner says so */ }
  pollJobs();
}
fovea.openActivity = () => openActivity(activityBtn);

// ------------------------------------------------------------------ connection banner + global errors
let retryTimer = null;
apiMod.onConnection((ok) => {
  clearInterval(retryTimer);
  if (ok) { netBanner.classList.add('hidden'); clear(netBanner); return; }
  mount(netBanner, icon('wifiOff', 15), h('span', { class: 'grow' }, apiMod.NETWORK_MESSAGE + '. Retrying…'),
    h('button', { class: 'btn btn-sm', onClick: () => get('/api/jobs', { active: 1 }).catch(() => {}) }, 'Retry now'));
  netBanner.classList.remove('hidden');
  retryTimer = setInterval(() => get('/api/jobs', { active: 1 }).then(() => { fovea.loadDatasets().catch(() => {}); }).catch(() => {}), 4000);
});
window.addEventListener('unhandledrejection', (e) => {
  const r = e.reason;
  if (!r || r.name === 'AbortError') return;
  console.error(r);
  if (r instanceof apiMod.ApiError && r.network) return;     // the banner already says it
  ui.toast(plainError(r), { type: 'error' });
});
/** An error message for people: no exception class names, no raw JSON. */
export function plainError(e) {
  let m = (e && (e.message || e.detail)) || String(e || 'Something went wrong');
  m = m.replace(/^[A-Z][A-Za-z]+(Error|Exception|Unreachable|NotFound):\s*/, '');
  return m.length > 300 ? m.slice(0, 300) + '…' : m;
}
fovea.plainError = plainError;

// ------------------------------------------------------------------ shortcuts modal
const GLOBAL_KEYS = [
  [`${MOD} K`, 'Command palette: datasets, pages, images, actions'], ['?', 'Keyboard shortcuts'],
  ['G H', 'Go to Home'], ['G O', 'Overview'], ['G E', 'Explore'], ['G B', 'Objects'], ['G A', 'Annotate'], ['G I', 'AI'],
  ['G F', 'Files'], ['G C', 'Compare'], ['G X', 'Export'], ['G S', 'Settings'], ['Esc', 'Close the dialog, menu or panel on top'],
];
keys.registerKeys('global', 'Everywhere', GLOBAL_KEYS, 0);
function currentScopes() {
  const m = location.pathname.match(/^\/d\/[^/]+(?:\/([^/]+))?/);
  if (!m) return [];
  const tab = m[1] || 'overview';
  return tab === 'explore' && /(?:^|[?&])view=inspect/.test(location.search) ? ['inspect', 'explore'] : [tab];
}
export function showShortcuts() {
  if (document.querySelector('.shortcuts-modal')) return;
  ui.shortcutsModal(keys.keyGroupsFor(currentScopes()));
}
fovea.showShortcuts = showShortcuts;

// ------------------------------------------------------------------ command palette
const RECENT_KEY = 'fovea.palette.recent';
const recentRuns = () => { try { return JSON.parse(ls.get(RECENT_KEY, '[]')); } catch (e) { return []; } };
function remember(id) { if (!id) return; const r = recentRuns().filter(x => x !== id); r.unshift(id); ls.set(RECENT_KEY, JSON.stringify(r.slice(0, 20))); }
/** Every query word must appear; matches at word starts and in the label rank higher, then recent use. */
function score(it, words, recent) {
  const label = it.label.toLowerCase(), hay = `${label} ${(it.keywords || '').toLowerCase()} ${(it.sub || '').toLowerCase()}`;
  let s = 0;
  for (const w of words) {
    const i = hay.indexOf(w);
    if (i < 0) return -1;
    const li = label.indexOf(w);
    s += li === 0 ? 30 : li > 0 && /[\s›·/_-]/.test(label[li - 1]) ? 20 : li >= 0 ? 10 : i > 0 && /[\s/_-]/.test(hay[i - 1]) ? 6 : 2;
  }
  const r = recent.indexOf(it.id);
  return s + (r >= 0 ? 12 - Math.min(10, r) : 0);
}
let paletteApi = null;
export function openPalette() {
  if (paletteApi) { paletteApi.el.querySelector('input').focus(); return; }
  if (ui.hasModal()) return;                         // never stack over a dialog that is waiting for an answer
  const { datasets } = state.get();
  const cur = location.pathname.startsWith('/d/') ? state.get().currentDataset : null;
  const items = [];
  const dsPath = (d) => (d.root_host || '').split('/').filter(Boolean).slice(-2).join('/');
  if (cur) tabs.forEach(t => items.push({ id: `tab:${t.id}`, label: `${t.label}`, sub: cur.name, icon: t.icon, keywords: `${cur.name} tab page`, run: () => router.navigate(`/d/${encodeURIComponent(cur.id)}${t.id === 'overview' ? '' : '/' + t.id}`) }));
  datasets.forEach(d => items.push({ id: `ds:${d.id}`, label: d.name, sub: `${fmt.num(d.image_count)} images · ${dsPath(d)}`, icon: 'database', keywords: `${d.id} ${dsPath(d)} dataset`, run: () => router.navigate(`/d/${encodeURIComponent(d.id)}`) }));
  // "<dataset> › <page>" for every dataset — found by typing ("synth explore"), never listed when the box is empty.
  datasets.forEach(d => tabs.forEach(t => { if (!cur || cur.id !== d.id) items.push({ id: `dstab:${d.id}:${t.id}`, label: `${d.name} › ${t.label}`, sub: 'page', icon: t.icon, keywords: d.id, searchOnly: true, run: () => router.navigate(`/d/${encodeURIComponent(d.id)}${t.id === 'overview' ? '' : '/' + t.id}`) }); }));
  if (cur) {
    items.push({ id: 'act:rescan', label: `Rescan ${cur.name}`, sub: 'action', icon: 'refresh', keywords: 'index scan refresh', run: async () => { try { await post(`/api/datasets/${encodeURIComponent(cur.id)}/scan`, { redetect: false }); ui.toast('Rescan started'); pollJobs(); } catch (e) { ui.toast(plainError(e), { type: 'error' }); } } });
    items.push({ id: 'act:snapshot', label: 'Save a version of the labels', sub: cur.name, icon: 'history', keywords: 'snapshot backup history version', run: async () => { try { const r = await post(`/api/datasets/${encodeURIComponent(cur.id)}/snapshots`, {}); ui.toast(r.snapshot_id ? 'Label version saved' : 'Saving a label version…', { type: 'ok' }); bus.emit('snapshots:changed', { dataset: cur.id }); pollJobs(); } catch (e) { ui.toast(plainError(e), { type: 'error' }); } } });
  }
  items.push({ id: 'act:add', label: 'Add dataset', icon: 'plus', sub: 'action', keywords: 'register new folder', run: () => fovea.openAddDataset() });
  items.push({ id: 'act:activity', label: 'Activity — running jobs', icon: 'activity', sub: 'action', keywords: 'jobs progress cancel', run: () => fovea.openActivity() });
  items.push({ id: 'act:theme', label: 'Toggle theme', icon: 'moon', sub: 'action', keywords: 'dark light', run: () => fovea.theme.cycle() });
  items.push({ id: 'act:keys', label: 'Keyboard shortcuts', icon: 'keyboard', sub: '?', keywords: 'help keys', run: showShortcuts });
  items.push({ id: 'page:settings', label: 'Settings', icon: 'settings', sub: 'page', keywords: 'preferences ai providers keys comfort', run: () => router.navigate('/settings') });
  items.push({ id: 'page:home', label: 'Home', icon: 'home', sub: 'page', keywords: 'datasets', run: () => router.navigate('/') });
  commands.forEach(c => { if (!c.when || c.when(state.get())) items.push({ id: `cmd:${c.id}`, label: c.label, icon: c.icon || 'zap', sub: c.sub || 'command', keywords: c.keywords || '', run: () => c.run(fovea) }); });

  let sel = 0, filtered = items, imgItems = [], imgSeq = 0;
  const list = h('div', { class: 'palette-list', role: 'listbox', id: 'palette-list', 'aria-label': 'Results' });
  const input = h('input', { class: 'input', placeholder: cur ? `Search pages, datasets, actions, or images in ${cur.name}…` : 'Search datasets, pages, actions…',
    role: 'combobox', 'aria-expanded': 'true', 'aria-controls': 'palette-list', 'aria-autocomplete': 'list', autocomplete: 'off', spellcheck: 'false' });
  const render = () => {
    const words = input.value.trim().toLowerCase().split(/\s+/).filter(Boolean);
    const recent = recentRuns();
    if (!words.length) filtered = items.filter(it => !it.searchOnly).sort((a, b) => { const ra = recent.indexOf(a.id), rb = recent.indexOf(b.id); return (ra < 0 ? 99 : ra) - (rb < 0 ? 99 : rb); });
    else filtered = items.map(it => [it, score(it, words, recent)]).filter(x => x[1] >= 0).sort((a, b) => b[1] - a[1]).map(x => x[0]);
    filtered = filtered.concat(imgItems);
    sel = Math.min(sel, Math.max(0, filtered.length - 1));
    mount(list, filtered.length ? filtered.map((it, i) => h('div', { class: cls('palette-item', i === sel && 'active'), role: 'option', id: `pal-${i}`, 'aria-selected': String(i === sel),
      onMousemove: () => { if (sel !== i) { sel = i; paint(); } }, onClick: () => run(it) },
      it.thumb ? h('img', { class: 'pal-thumb', src: it.thumb, alt: '', loading: 'lazy' }) : icon(it.icon || 'circle', 16), h('span', { class: 'pal-label' }, it.labelEl || it.label), h('span', { class: 'sub' }, it.sub || ''))) :
      h('div', { class: 'palette-empty' }, words.length ? 'No matches' : 'Nothing to show'));
    paint();
  };
  const paint = () => {
    [...list.children].forEach((c, i) => { c.classList.toggle('active', i === sel); c.setAttribute('aria-selected', String(i === sel)); });
    const a = list.children[sel];
    if (a && a.id) { input.setAttribute('aria-activedescendant', a.id); a.scrollIntoView({ block: 'nearest' }); }
  };
  const searchImages = debounce(async () => {
    const q = input.value.trim();
    const seq = ++imgSeq;
    if (!cur || q.length < 2) { if (imgItems.length) { imgItems = []; render(); } return; }
    try {
      const r = await get(`/api/datasets/${encodeURIComponent(cur.id)}/images`, { q, limit: 6 });
      if (seq !== imgSeq || !paletteApi) return;
      imgItems = (r.items || []).map(it => ({ id: `img:${it.id}`, label: it.rel_path, labelEl: ui.midText(it.rel_path, { tail: 18 }), sub: 'image', thumb: thumbUrl(it.id, 64, it.mtime),
        run: () => router.navigate(`/d/${encodeURIComponent(cur.id)}/explore?img=${it.id}&view=inspect`) }));
      render();
    } catch (e) { /* the list stays as it is */ }
  }, 180);
  const run = (it) => {
    remember(it.id);
    ui.closeModals({ all: true });               // the palette and anything under it: the action owns the screen now
    Promise.resolve().then(() => it.run());
  };
  paletteApi = ui.modal({
    bare: true, className: 'palette', backdropClass: 'top', initialFocus: input, title: 'Command palette',
    body: () => h('div', { class: 'col' }, h('div', { class: 'input-wrap' }, icon('search'), input), list),
    onClose: () => { paletteApi = null; },
  });
  input.addEventListener('keydown', (e) => {
    if (e.isComposing) return;
    if (e.key === 'ArrowDown') { e.preventDefault(); sel = (sel + 1) % Math.max(1, filtered.length); paint(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = (sel - 1 + filtered.length) % Math.max(1, filtered.length); paint(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (filtered[sel]) run(filtered[sel]); }
  });
  input.addEventListener('input', () => { sel = 0; render(); searchImages(); });
  render();
}
fovea.openPalette = openPalette;

// ------------------------------------------------------------------ global keys (capture phase)
// The G chord runs in the capture phase and CONSUMES its second key, so "G A" navigates to Annotate and never
// also approves the image in Inspect. A static "G…" chip shows while the chord waits.
let chord = null, chordTimer = null;
const chordChip = h('div', { class: 'chord-chip hidden', role: 'status' }, h('span', { class: 'kbd' }, 'G'), h('span', '… H home · O overview · E explore · A annotate · F files · ? all'));
document.body.appendChild(chordChip);
function setChord(v) {
  chord = v; clearTimeout(chordTimer);
  chordChip.classList.toggle('hidden', !v);
  if (v) chordTimer = setTimeout(() => setChord(null), 1600);
}
// [explore hook] Views that listen ahead of this handler (Explore's Inspect, window capture) ask whether the chord
// still waits, so its second key is never also taken as a view shortcut.
fovea.chordPending = () => chord === 'g';
const CHORD_TABS ={ o: '', e: 'explore', b: 'objects', a: 'annotate', i: 'ai', f: 'files', x: 'export', c: 'compare' };
document.addEventListener('keydown', (e) => {
  if (e.isComposing || e.keyCode === 229) return;
  if (keys.is(e, 'mod+k') || ((e.metaKey || e.ctrlKey) && keys.letter(e) === 'k' && !e.shiftKey && !e.altKey)) {
    e.preventDefault(); e.stopImmediatePropagation();
    if (paletteApi) ui.closeModals({ all: false }); else openPalette();
    return;
  }
  if (chord === 'g') {
    if (e.key === 'Shift' || e.key === 'CapsLock') return;
    e.preventDefault(); e.stopImmediatePropagation();
    setChord(null);
    if (e.key === 'Escape' || !keys.plain(e)) return;
    const k = keys.letter(e);
    if (keys.isHelp(e)) { showShortcuts(); return; }
    if (k === 'h') { router.navigate('/'); return; }
    if (k === 's') { router.navigate('/settings'); return; }
    if (!(k in CHORD_TABS)) return;
    const ds = fovea.recentDataset();
    if (!ds) { ui.toast('Add a dataset first — G-keys open its pages'); return; }
    const tab = CHORD_TABS[k];
    if (tab && !tabs.some(t => t.id === tab) && viewsLoaded) { ui.toast(`No ${tab} page`); return; }
    router.navigate(`/d/${encodeURIComponent(ds.id)}${tab ? '/' + tab : ''}`);
    return;
  }
  if (ui.hasModal() || keys.inField(e) || e.metaKey || e.ctrlKey || e.altKey) return;
  if (keys.isHelp(e)) { e.preventDefault(); e.stopImmediatePropagation(); showShortcuts(); return; }
  if (keys.letter(e) === 'g' && !e.shiftKey && !e.repeat) { e.preventDefault(); e.stopImmediatePropagation(); setChord('g'); }
}, true);
window.addEventListener('blur', () => setChord(null));

// Sliders, selects and checkboxes picked with the pointer give focus back to the page, so the view's single-key
// shortcuts work right away (keyboard users who Tab to a control keep it).
let lastPointer = 0, lastKey = 0;
document.addEventListener('pointerdown', () => { lastPointer = performance.now(); }, true);
document.addEventListener('keydown', (e) => { if (e.key !== 'Shift') lastKey = performance.now(); }, true);
const releaseIfPointer = (e) => {
  const t = e.target;
  if (!t || !t.matches || !t.matches('input[type=range], input[type=checkbox], input[type=radio], select')) return;
  if (lastPointer <= lastKey || t.closest('.modal, .menu')) return;
  setTimeout(() => { if (document.activeElement === t) t.blur(); }, 0);
};
document.addEventListener('change', releaseIfPointer, true);
document.addEventListener('pointerup', (e) => { if (e.target && e.target.matches && e.target.matches('input[type=range]')) releaseIfPointer(e); }, true);

// ------------------------------------------------------------------ routing
router.before.push(() => {
  // Menus, tooltips and dialogs belong to the page that opened them.
  ui.closeMenu(); ui.hideTip(); ui.closeModals(); setChord(null);
  document.title = 'Fovea';
});

const VIEW_MODULES = {
  home: () => import('./views/home.js'), dataset: () => import('./views/dataset.js'), overview: () => import('./views/overview.js'),
  explore: () => import('./views/explore.js'), objects: () => import('./views/objects.js'), annotate: () => import('./views/annotate.js'),
  ai: () => import('./views/ai.js'), files: () => import('./views/files.js'), export: () => import('./views/export.js'), settings: () => import('./views/settings.js'),
};
const loaded = new Map();
function viewModule(name) {
  if (!loaded.has(name)) {
    loaded.set(name, VIEW_MODULES[name]().then((m) => { if (m.install) m.install(fovea); return m; }).catch((e) => { loaded.delete(name); throw e; }));
  }
  return loaded.get(name);
}
let viewsLoaded = false, viewsPromise = null;
/** Every built-in view (their tabs and key groups) — dataset pages need all of them. */
function coreViews() {
  if (!viewsPromise) {
    viewsPromise = Promise.all(['home', 'dataset', 'overview', 'explore', 'objects', 'annotate', 'ai', 'files', 'export'].map(viewModule))
      .then((ms) => { viewsLoaded = true; return ms; })
      .catch((e) => { viewsPromise = null; throw e; });
  }
  return viewsPromise;
}
const pluginsReady = Promise.allSettled((fovea.config.plugins || []).filter(p => p.entry).map(p =>
  import(p.entry).then((mod) => { if (mod.default) mod.default(fovea, p); })
    .catch((e) => { console.error(`[plugin:${p.id}]`, e); ui.toast(`Plugin ${p.name || p.id} failed to load`, { type: 'error' }); })));

fovea.openAddDataset = async (...a) => (await viewModule('home')).openAddDataset(...a);
fovea.pickPath = async (...a) => (await viewModule('home')).pickPath(...a);

function viewError(e) {
  console.error(e);
  setView('error', ui.emptyState({ icon: 'alert', title: 'This page failed to load', message: plainError(e), action: h('button', { class: 'btn', onClick: () => location.reload() }, 'Reload') }));
}
router.add('/', async (ctx) => { try { const m = await viewModule('home'); if (router.isCurrent(ctx)) await m.render(ctx, fovea); } catch (e) { viewError(e); } });
router.add('/settings', async (ctx) => { try { const m = await viewModule('settings'); if (router.isCurrent(ctx)) await m.render(ctx, fovea); } catch (e) { viewError(e); } });
const datasetRoute = (tabOf) => async (ctx) => {
  try {
    const [ms] = await Promise.all([coreViews(), pluginsReady]);
    if (!router.isCurrent(ctx)) return;
    await ms[1].render(ctx, fovea, tabOf(ctx));
  } catch (e) { viewError(e); }
};
router.add('/d/:id', datasetRoute(() => 'overview'));
router.add('/d/:id/:tab', datasetRoute((ctx) => ctx.params.tab));
router.notFound = () => setView('404', ui.emptyState({ icon: 'alert', title: 'Page not found', message: location.pathname, action: link('/', { class: 'btn' }, 'Go home') }));

// ------------------------------------------------------------------ boot
async function boot() {
  document.documentElement.setAttribute('data-theme', state.get().theme);
  comfort.init();
  renderSidebar();
  const boot = window.FOVEA_BOOT && window.FOVEA_BOOT.datasets;
  const first = boot ? boot.then(r => setDatasets(r.datasets)).catch(() => fovea.loadDatasets()) : fovea.loadDatasets();
  first.catch((e) => { state.set({ datasetsError: plainError(e), datasetsLoaded: true }); });
  // Pages render right away; the dataset list fills in when it arrives (it is usually already there).
  await router.start();
  pollJobs();
  // Load the remaining views while the person looks at the first page, so the next click is instant.
  const idle = window.requestIdleCallback || ((fn) => setTimeout(fn, 300));
  idle(() => { coreViews().catch(() => {}); viewModule('settings').catch(() => {}); });
}
boot();
