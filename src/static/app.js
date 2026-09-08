// Fovea app shell: sidebar, routing, view/tab registry, command palette, plugins.
import { h, svg, mount, clear, fmt, cls, isTyping, MOD, debounce } from './lib/dom.js';
import { icon, logo } from './lib/icons.js';
import { createStore, createBus } from './lib/store.js';
import * as apiMod from './lib/api.js';
import { router, link } from './lib/router.js';
import * as ui from './lib/ui.js';
import * as colors from './lib/colors.js';
import { boxLayer } from './lib/overlay.js';
import { ImageCursor } from './lib/cursor.js';

const { get, post, put, patch, del, watchJob, imgUrl, thumbUrl } = apiMod;

const state = createStore({ datasets: [], meta: null, currentDataset: null, theme: localStorage.getItem('fovea.theme') || 'system' });
const bus = createBus();
const tabs = [];
const commands = [];

export const fovea = {
  h, svg, icon, logo, fmt, cls, MOD, isTyping, debounce,
  api: { get, post, put, patch, del, watchJob, imgUrl, thumbUrl, ApiError: apiMod.ApiError },
  router, link, ui, colors, boxLayer, ImageCursor, state, bus,
  config: window.FOVEA || {},
  registerTab(tab) { const i = tabs.findIndex(t => t.id === tab.id); if (i >= 0) tabs[i] = tab; else tabs.push(tab); tabs.sort((a, b) => (a.order ?? 50) - (b.order ?? 50)); },
  tabs: () => tabs.slice(),
  registerCommand(cmd) { const i = commands.findIndex(c => c.id === cmd.id); if (i >= 0) commands[i] = cmd; else commands.push(cmd); },
  commands: () => commands.slice(),
  async loadDatasets() { const { datasets } = await get('/api/datasets'); state.set({ datasets }); return datasets; },
  async getDataset(id, { touch = false, force = false } = {}) {
    const cur = state.get().currentDataset;
    if (!force && cur && cur.id === id && !touch) return cur;
    const { dataset } = await get(`/api/datasets/${id}`, { touch: touch ? 1 : 0 });
    state.set({ currentDataset: dataset, datasets: state.get().datasets.map(d => d.id === id ? { ...d, ...dataset } : d) });
    return dataset;
  },
  theme: {
    get: () => state.get().theme,
    set(t) { state.set({ theme: t }); localStorage.setItem('fovea.theme', t); document.documentElement.setAttribute('data-theme', t); },
    cycle() { const order = ['system', 'light', 'dark']; fovea.theme.set(order[(order.indexOf(fovea.theme.get()) + 1) % 3]); },
  },
  navigate: (to, o) => router.navigate(to, o),
};
window.fovea = fovea;

// ------------------------------------------------------------------ layout
const appEl = document.getElementById('app');
const sidebar = h('aside', { class: 'sidebar' });
const main = h('main', { class: 'main' });
const view = h('div', { id: 'view', style: 'display:flex;flex-direction:column;min-height:0;flex:1;' });
main.appendChild(view);
mount(appEl, sidebar, main);
fovea.viewEl = view;

let currentView = { destroy: null, key: null };
export function setView(key, node, destroy) {
  if (currentView.destroy) { try { currentView.destroy(); } catch (e) { console.error(e); } }
  currentView = { destroy, key };
  mount(view, node);
}
fovea.setView = setView;

function renderSidebar() {
  const { datasets, currentDataset } = state.get();
  const path = location.pathname;
  const items = datasets.map(d => link(`/d/${d.id}`, { class: cls('sb-item', path.startsWith(`/d/${d.id}`) && 'active'), 'data-tip': d.root_host },
    h('span', { class: `dot ${d.status}` }), h('span', { class: 'name' }, d.name), h('span', { class: 'count' }, fmt.compact(d.image_count))));
  mount(sidebar,
    link('/', { class: 'sb-brand' }, logo(24), h('span', { class: 'wordmark' }, 'fovea'), h('span', { class: 'ver' }, 'v' + (fovea.config.version || ''))),
    h('div', { class: 'sb-section' },
      link('/', { class: cls('sb-item', path === '/' && 'active') }, icon('home'), h('span', { class: 'name' }, 'Home')),
      h('button', { class: 'sb-item', onClick: () => openPalette() }, icon('search'), h('span', { class: 'name' }, 'Search'), h('span', { class: 'count' }, `${MOD} K`)),
    ),
    h('div', { class: 'sb-section sb-datasets' },
      h('div', { class: 'sb-title' }, 'Datasets', h('span', { class: 'spacer' }), h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'data-tip': 'Add dataset', onClick: () => fovea.openAddDataset && fovea.openAddDataset() }, icon('plus', 14))),
      items.length ? items : h('div', { class: 'faint small', style: 'padding:6px 8px' }, 'No datasets yet'),
    ),
    h('div', { class: 'sb-footer' },
      link('/settings', { class: cls('sb-item', path === '/settings' && 'active') }, icon('settings'), h('span', { class: 'name' }, 'Settings')),
      h('button', { class: 'sb-item', onClick: showShortcuts }, icon('keyboard'), h('span', { class: 'name' }, 'Shortcuts'), h('span', { class: 'count' }, '?')),
      h('button', { class: 'sb-item', onClick: () => { fovea.theme.cycle(); renderSidebar(); } }, icon(state.get().theme === 'dark' ? 'moon' : state.get().theme === 'light' ? 'sun' : 'monitor'), h('span', { class: 'name' }, 'Theme'), h('span', { class: 'count' }, state.get().theme)),
    ),
  );
}
state.subscribe(renderSidebar);
router.before.push(() => { setTimeout(renderSidebar, 0); });

// Poll while something is scanning
let pollTimer = null;
function schedulePoll() {
  clearTimeout(pollTimer);
  const busy = state.get().datasets.some(d => d.status === 'scanning' || (d.job && ['queued', 'running'].includes(d.job.status)));
  if (busy) pollTimer = setTimeout(async () => { try { await fovea.loadDatasets(); } catch (e) {} schedulePoll(); }, 2500);
}
state.subscribe(schedulePoll);

// ------------------------------------------------------------------ shortcuts modal
export function showShortcuts() {
  ui.shortcutsModal([
    { title: 'Global', items: [[`${MOD} K`, 'Command palette / search'], ['?', 'This help'], ['G H', 'Go home'], ['G O', 'Overview'], ['G E', 'Explore'], ['G A', 'Annotate'], ['Esc', 'Close dialog / panel']] },
    { title: 'Explore & Inspect', items: [['← →', 'Previous / next image'], ['Enter', 'Open image (inspect)'], ['A', 'Approve'], ['F', 'Flag'], ['X', 'Exclude'], ['U', 'Clear review'], ['O', 'Toggle overlay'], ['E', 'Edit in Annotate'], ['Space', 'Auto-play'], ['Shift Click', 'Range select'], [`${MOD} A`, 'Select all loaded']] },
    { title: 'Annotate', items: [['V', 'Select tool'], ['B', 'Box tool'], ['P', 'Point tool'], ['H', 'Pan tool'], ['0-9', 'Set class (selected box or active)'], ['[ ]', 'Previous / next class'], ['Del', 'Delete box'], [`${MOD} Z`, 'Undo'], [`${MOD} ⇧ Z`, 'Redo'], ['Tab', 'Cycle boxes'], ['Esc', 'Deselect'], ['f', 'Fit to screen'], ['L', 'Toggle labels'], ['N', 'Next unlabeled'], ['C', 'Copy boxes from previous image'], ['⇧A ⇧F ⇧X', 'Approve / flag / exclude'], ['Alt ←↑→↓', 'Nudge box (⇧ = 10px)'], ['Wheel', 'Zoom'], ['Space drag', 'Pan'], ['Right click', 'Box menu']] },
  ]);
}
fovea.showShortcuts = showShortcuts;

// ------------------------------------------------------------------ command palette
let paletteOpen = false;
export function openPalette() {
  if (paletteOpen) return; paletteOpen = true;
  const { datasets, currentDataset } = state.get();
  const items = [];
  datasets.forEach(d => items.push({ label: d.name, sub: `${fmt.num(d.image_count)} images`, icon: 'database', run: () => router.navigate(`/d/${d.id}`), keywords: d.id + ' ' + d.root_host }));
  if (currentDataset) tabs.forEach(t => items.push({ label: `${currentDataset.name} › ${t.label}`, sub: 'tab', icon: t.icon, run: () => router.navigate(`/d/${currentDataset.id}/${t.id}`) }));
  items.push({ label: 'Add dataset', icon: 'plus', sub: 'action', run: () => fovea.openAddDataset && fovea.openAddDataset() });
  items.push({ label: 'Toggle theme', icon: 'moon', sub: 'action', run: () => fovea.theme.cycle() });
  items.push({ label: 'Keyboard shortcuts', icon: 'keyboard', sub: '?', run: showShortcuts });
  items.push({ label: 'Settings', icon: 'settings', sub: 'page', run: () => router.navigate('/settings') });
  commands.forEach(c => { if (!c.when || c.when(state.get())) items.push({ label: c.label, icon: c.icon || 'zap', sub: c.sub || 'command', run: () => c.run(fovea) }); });
  let sel = 0, filtered = items;
  const list = h('div', { class: 'palette-list' });
  const input = h('input', { class: 'input', placeholder: 'Search datasets, tabs, actions…', autofocus: true });
  const render = () => {
    const q = input.value.trim().toLowerCase();
    filtered = q ? items.filter(it => (it.label + ' ' + (it.keywords || '') + ' ' + it.sub).toLowerCase().includes(q)) : items;
    sel = Math.min(sel, Math.max(0, filtered.length - 1));
    mount(list, filtered.length ? filtered.map((it, i) => h('div', { class: cls('palette-item', i === sel && 'active'), onMousemove: () => { if (sel !== i) { sel = i; render(); } }, onClick: () => { run(it); } }, icon(it.icon || 'circle', 16), h('span', it.label), h('span', { class: 'sub' }, it.sub || ''))) : h('div', { class: 'palette-empty' }, 'No matches'));
  };
  const backdrop = h('div', { class: 'backdrop', style: 'align-items:flex-start' });
  const close = () => { backdrop.remove(); paletteOpen = false; document.removeEventListener('keydown', onKey, true); };
  const run = (it) => { close(); it.run(); };
  const onKey = (e) => {
    if (e.key === 'Escape') { e.stopPropagation(); close(); }
    else if (e.key === 'ArrowDown') { e.preventDefault(); sel = (sel + 1) % Math.max(1, filtered.length); render(); }
    else if (e.key === 'ArrowUp') { e.preventDefault(); sel = (sel - 1 + filtered.length) % Math.max(1, filtered.length); render(); }
    else if (e.key === 'Enter') { e.preventDefault(); if (filtered[sel]) run(filtered[sel]); }
  };
  input.addEventListener('input', () => { sel = 0; render(); });
  document.addEventListener('keydown', onKey, true);
  backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop) close(); });
  backdrop.appendChild(h('div', { class: 'palette' }, h('div', { class: 'input-wrap' }, icon('search'), input), list));
  document.body.appendChild(backdrop);
  render(); setTimeout(() => input.focus(), 10);
}
fovea.openPalette = openPalette;

// Global keys
let chord = null;
document.addEventListener('keydown', (e) => {
  if ((e.metaKey || e.ctrlKey) && e.key.toLowerCase() === 'k') { e.preventDefault(); openPalette(); return; }
  if (isTyping() || ui.hasModal() || e.metaKey || e.ctrlKey || e.altKey) return;
  if (e.key === '?') { e.preventDefault(); showShortcuts(); return; }
  if (chord === 'g') {
    chord = null;
    const ds = state.get().currentDataset;
    const k = e.key.toLowerCase();
    if (k === 'h') router.navigate('/');
    else if (ds && k === 'o') router.navigate(`/d/${ds.id}`);
    else if (ds && k === 'e') router.navigate(`/d/${ds.id}/explore`);
    else if (ds && k === 'a') router.navigate(`/d/${ds.id}/annotate`);
    else if (ds && k === 'f') router.navigate(`/d/${ds.id}/files`);
    return;
  }
  if (e.key.toLowerCase() === 'g') { chord = 'g'; setTimeout(() => chord = null, 900); }
});

// ------------------------------------------------------------------ boot
async function boot() {
  document.documentElement.setAttribute('data-theme', state.get().theme);
  const [home, dataset, overview, explore, annotate, files, exportV, settings] = await Promise.all([
    import('./views/home.js'), import('./views/dataset.js'), import('./views/overview.js'), import('./views/explore.js'),
    import('./views/annotate.js'), import('./views/files.js'), import('./views/export.js'), import('./views/settings.js'),
  ]);
  for (const m of [home, dataset, overview, explore, annotate, files, exportV, settings]) if (m.install) m.install(fovea);

  // plugins
  for (const p of (fovea.config.plugins || [])) {
    if (!p.entry) continue;
    try { const mod = await import(p.entry); if (mod.default) mod.default(fovea, p); }
    catch (e) { console.error(`[plugin:${p.id}]`, e); ui.toast(`Plugin ${p.id} failed to load`, { type: 'error' }); }
  }

  router.add('/', (ctx) => home.render(ctx, fovea));
  router.add('/settings', (ctx) => settings.render(ctx, fovea));
  router.add('/d/:id', (ctx) => dataset.render(ctx, fovea, 'overview'));
  router.add('/d/:id/:tab', (ctx) => dataset.render(ctx, fovea, ctx.params.tab));
  router.notFound = () => setView('404', ui.emptyState({ icon: 'alert', title: 'Not found', message: location.pathname }));

  try { await fovea.loadDatasets(); } catch (e) { ui.toast('Failed to load datasets: ' + e.message, { type: 'error' }); }
  get('/api/meta').then(meta => state.set({ meta })).catch(() => {});
  renderSidebar();
  await router.start();
}
boot();
