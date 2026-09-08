export function install(fovea) {}

export async function render(ctx, fovea) {
  const { h, icon, ui, fmt } = fovea;
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner', style: 'max-width:820px' }); page.appendChild(inner);
  fovea.setView('settings', page);
  inner.appendChild(h('div', { class: 'mb-16' }, h('h1', { class: 'page-title' }, 'Settings'), h('div', { class: 'page-sub' }, 'Appearance, storage and integrations.')));
  let meta; try { meta = await fovea.api.get('/api/meta'); } catch (e) { inner.appendChild(ui.emptyState({ icon: 'alert', title: 'Failed to load', message: e.message })); return; }
  const row = (label, value) => h('div', { class: 'row' }, h('span', { class: 'muted', style: 'width:180px' }, label), h('span', { class: 'grow' }, value));
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Appearance')), h('div', { class: 'card-body settings-list' },
    row('Theme', ui.seg([{ value: 'system', label: 'System', icon: 'monitor' }, { value: 'light', label: 'Light', icon: 'sun' }, { value: 'dark', label: 'Dark', icon: 'moon' }], fovea.theme.get(), v => fovea.theme.set(v))))));
  const cacheEl = h('span', `${fmt.num(meta.thumb_cache.files)} files · ${fmt.bytes(meta.thumb_cache.bytes)}`);
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Storage')), h('div', { class: 'card-body settings-list' },
    row('Version', h('span', { class: 'mono' }, meta.version)),
    row('Data directory', h('span', { class: 'mono small' }, meta.data_dir)),
    row('Thumbnail cache', h('div', { class: 'row' }, cacheEl, h('button', { class: 'btn btn-sm', onClick: async () => { const r = await fovea.api.post('/api/cache/clear'); cacheEl.textContent = '0 files'; ui.toast(`Removed ${r.removed} thumbnails`); } }, 'Clear'))),
    row('Host path mapping', meta.host_path ? h('span', { class: 'mono small' }, `${meta.host_path} → ${meta.container_mount}`) : h('span', { class: 'faint' }, 'none (running directly on host)')))));
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'AI auto-label'), h('span', { class: 'sub' }, meta.ai.available ? 'ultralytics available' : 'ultralytics not installed')), h('div', { class: 'card-body settings-list' },
    row('Model directory', h('span', { class: 'mono small' }, meta.model_dir)),
    row('Models', meta.ai.models.length ? h('div', { class: 'row wrap gap-4' }, meta.ai.models.map(m => ui.chip(`${m.name} · ${m.size_mb} MB`, { cls: 'outline', icon: 'cpu' }))) : h('span', { class: 'faint' }, 'Drop YOLO .pt weights into the model directory')))));
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Plugins'), h('span', { class: 'sub' }, `${meta.plugins.length} loaded`)), h('div', { class: 'card-body settings-list' },
    meta.plugins.length ? meta.plugins.map(p => row(p.name, h('span', { class: 'small' }, p.description || '', p.error ? ui.chip('error: ' + p.error, { type: 'danger' }) : ''))) : h('div', { class: 'small faint' }, 'Add a package under fovea/plugins/ exposing PLUGIN to extend Fovea.'))));
}
