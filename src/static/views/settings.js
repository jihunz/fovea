export function install(fovea) {}

export async function render(ctx, fovea) {
  const { h, icon, ui, fmt } = fovea;
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner', style: 'max-width:820px' }); page.appendChild(inner);
  fovea.setView('settings', page);
  inner.appendChild(h('div', { class: 'mb-16' }, h('h1', { class: 'page-title' }, 'Settings'), h('div', { class: 'page-sub' }, 'Appearance, storage and integrations.')));
  let meta; try { meta = await fovea.api.get('/api/meta'); } catch (e) { inner.appendChild(ui.emptyState({ icon: 'alert', title: 'Failed to load', message: e.message })); return; }
  const row = (label, value) => h('div', { class: 'row', style: 'align-items:flex-start' }, h('span', { class: 'muted', style: 'width:180px;flex:none;padding-top:4px' }, label), h('span', { class: 'grow', style: 'min-width:0' }, value));
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Appearance')), h('div', { class: 'card-body settings-list' },
    row('Theme', h('div', { class: 'col gap-4' }, ui.seg([{ value: 'system', label: 'System', icon: 'monitor' }, { value: 'light', label: 'Light', icon: 'sun' }, { value: 'dark', label: 'Dark', icon: 'moon' }], fovea.theme.get(), v => fovea.theme.set(v)),
      h('span', { class: 'hint' }, 'Neither theme is less tiring for everyone — people differ, and often not in the way they expect. Try both for a session.'))))));

  // ---- visual comfort (per device: depends on the room and the display)
  const c = fovea.comfort.get();
  const dimLabel = h('span', { class: 'mono small', style: 'width:40px;text-align:right' }, `${Math.round(c.dim * 100)}%`);
  const dim = h('input', { type: 'range', class: 'slider', min: 0, max: 0.6, step: 0.05, value: c.dim, style: 'width:180px',
    onInput: (e) => { const v = Number(e.target.value); dimLabel.textContent = `${Math.round(v * 100)}%`; fovea.comfort.set({ dim: v }); } });
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Visual comfort'), h('span', { class: 'sub' }, 'this device')), h('div', { class: 'card-body settings-list' },
    row('Screen dimming', h('div', { class: 'col gap-4' }, h('div', { class: 'row' }, dim, dimLabel),
      h('span', { class: 'hint' }, 'Darkens everything Fovea shows. Match the screen to the room — a bright screen in a dim room is the most tiring combination. Lowering brightness helps; colour-temperature ("blue light") filters were not shown to.'))),
    row('Image surround', h('div', { class: 'col gap-4' }, ui.seg([{ value: 'black', label: 'Black' }, { value: 'dark', label: 'Dark gray' }, { value: 'gray', label: 'Gray' }], c.stage, v => fovea.comfort.set({ stage: v })),
      h('span', { class: 'hint' }, 'Background around images in Inspect, Annotate and Compare. A lighter surround shrinks the brightness jump between an image and its edges.'))),
    row('Rest reminder', h('div', { class: 'col gap-4' }, ui.seg([{ value: 0, label: 'Off' }, { value: 30, label: '30 min' }, { value: 45, label: '45 min' }, { value: 60, label: '60 min' }], c.rest, v => fovea.comfort.set({ rest: v })),
      h('span', { class: 'hint' }, 'A quiet note after continuous work; five idle minutes count as a break. Suggests a real break — a 20-second glance was not shown to help.'))),
    row('Image changes', h('span', { class: 'small muted' }, 'Always instant: the current image stays until the next one is ready, with no fade, blank frame or animated placeholder.')))));
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
