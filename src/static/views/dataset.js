// Dataset shell: header, scan banner, tabs (core + plugins). Keeps the shell mounted across tab switches.
let shell = null; // { dsId, el, content, tabBar, header, banner, tabId, destroyTab, jobWatch }

export function install(fovea) {}

export async function render(ctx, fovea, tabId) {
  const { h, icon, ui, fmt, link, cls } = fovea;
  const id = ctx.params.id;
  const tabs = fovea.tabs();
  const tab = tabs.find(t => t.id === tabId) || tabs[0];
  if (!tab) return;

  let ds;
  try { ds = await fovea.getDataset(id, { touch: !shell || shell.dsId !== id }); }
  catch (e) { fovea.setView('ds-missing', ui.emptyState({ icon: 'alert', title: 'Dataset not found', message: e.message, action: link('/', { class: 'btn' }, 'Back home') })); return; }

  if (!shell || shell.dsId !== id || !shell.el.isConnected) {
    if (shell && shell.destroyTab) { try { shell.destroyTab(); } catch (e) {} }
    shell = buildShell(ds);
    fovea.setView('dataset:' + id, shell.el, () => { if (shell) { shell.destroyTab && shell.destroyTab(); shell.stopWatch && shell.stopWatch(); shell = null; } });
  } else {
    updateHeader(ds);
  }
  if (shell.tabId !== tab.id || !shell.content.firstChild) {
    if (shell.destroyTab) { try { shell.destroyTab(); } catch (e) {} shell.destroyTab = null; }
    shell.tabId = tab.id;
    shell.content.innerHTML = '';
    Array.from(shell.tabBar.children).forEach(b => b.classList.toggle('active', b.dataset.tab === tab.id));
    document.title = `${ds.name} · ${tab.label} · Fovea`;
    const res = await tab.render(shell.content, { ctx, dataset: ds, fovea, refresh: () => refreshDataset(ds.id) });
    if (typeof res === 'function') shell.destroyTab = res;
  } else {
    shell.onRouteUpdate && shell.onRouteUpdate(ctx);
  }

  function buildShell(ds) {
    const el = h('div', { style: 'display:flex;flex-direction:column;height:100%;min-height:0' });
    const header = h('div', { class: 'ds-header' });
    const banner = h('div', { class: 'ds-scan hidden' });
    const tabBar = h('div', { class: 'tabs' });
    const content = h('div', { style: 'flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden' });
    tabs.forEach(t => tabBar.appendChild(h('button', { class: cls('tab', t.id === tab.id && 'active'), 'data-tab': t.id, onClick: () => fovea.router.navigate(`/d/${ds.id}/${t.id === 'overview' ? '' : t.id}`.replace(/\/$/, '')) }, icon(t.icon || 'circle', 15), t.label)));
    el.appendChild(header); el.appendChild(banner); el.appendChild(tabBar); el.appendChild(content);
    const s = { dsId: ds.id, el, header, banner, tabBar, content, tabId: null, destroyTab: null, stopWatch: null };
    shell = s;
    updateHeader(ds);
    return s;
  }

  function updateHeader(ds) {
    const s = shell; if (!s) return;
    const kind = ds.layout && ds.layout.kind;
    s.header.innerHTML = '';
    s.header.appendChild(h('div', { class: 'row', style: 'align-items:flex-start' },
      h('div', { class: 'grow' },
        h('h1', ds.name, ui.chip(kind || '?', { type: 'accent' }), h('span', { class: 'status-pill' }, h('span', { class: `dot ${ds.status}` }), ds.status === 'ready' ? `indexed ${fmt.ago(ds.scanned_at)}` : ds.status)),
        h('div', { class: 'path' }, h('span', { class: 'truncate', style: 'max-width:60vw' }, ds.root_host), h('button', { class: 'btn btn-ghost btn-icon btn-sm copy', 'data-tip': 'Copy path', onClick: () => ui.copyText(ds.root_host) }, icon('copy', 12)))),
      h('div', { class: 'row' },
        h('span', { class: 'small muted nowrap' }, `${fmt.num(ds.image_count)} images · ${fmt.num(ds.label_count)} labeled · ${fmt.num(ds.box_count)} boxes · ${(ds.classes || []).length} classes`),
        link(`/d/${ds.id}/explore`, { class: 'btn btn-sm' }, icon('grid', 13), 'Explore'),
        link(`/d/${ds.id}/annotate`, { class: 'btn btn-sm btn-primary' }, icon('pen', 13), 'Annotate'),
        h('button', { class: 'btn btn-sm btn-icon', onClick: (e) => headerMenu(e.currentTarget, ds) }, icon('moreH', 14)))));
    watchJob(ds);
  }

  function headerMenu(anchor, ds) {
    ui.menu(anchor, [
      { label: 'Rescan index', icon: 'refresh', onClick: () => startScan(ds.id, false) },
      { label: 'Re-detect layout & rescan', icon: 'scan', onClick: () => startScan(ds.id, true) },
      { label: 'Rename…', icon: 'pen', onClick: async () => { const v = await ui.prompt({ title: 'Rename dataset', value: ds.name }); if (v && v.trim()) { await fovea.api.patch(`/api/datasets/${ds.id}`, { name: v.trim() }); await refreshDataset(ds.id); await fovea.loadDatasets(); } } },
      { label: 'Copy path', icon: 'copy', onClick: () => ui.copyText(ds.root_host) },
      { label: 'API: dataset JSON', icon: 'external', onClick: () => window.open(`/api/datasets/${ds.id}`, '_blank') },
      { sep: true },
      { label: 'Remove from Fovea', icon: 'trash', danger: true, onClick: async () => {
        if (await ui.confirm({ title: `Remove “${ds.name}”?`, message: 'Removes the local index and review marks only. Files are untouched.', okLabel: 'Remove', danger: true })) {
          await fovea.api.del(`/api/datasets/${ds.id}`); await fovea.loadDatasets(); fovea.router.navigate('/'); ui.toast('Dataset removed');
        } } },
    ]);
  }

  async function startScan(dsId, redetect) {
    try { const res = await fovea.api.post(`/api/datasets/${dsId}/scan`, { redetect }); ui.toast(res.already_running ? 'Scan already running' : 'Rescan started'); await refreshDataset(dsId); }
    catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  async function refreshDataset(dsId) {
    const d = await fovea.getDataset(dsId, { force: true });
    if (shell && shell.dsId === dsId) updateHeader(d);
    fovea.bus.emit('dataset:updated', d);
    return d;
  }

  function watchJob(ds) {
    const s = shell; if (!s) return;
    const job = ds.job;
    if (!job || !['queued', 'running'].includes(job.status)) { s.banner.classList.add('hidden'); return; }
    if (s.watchingJob === job.id) return;
    s.watchingJob = job.id;
    s.banner.classList.remove('hidden');
    const bar = ui.progress(job.progress || 0, { indeterminate: !job.total });
    const txt = h('span', jobText(job));
    s.banner.innerHTML = ''; s.banner.appendChild(icon('refresh', 14)); s.banner.appendChild(txt); s.banner.appendChild(bar);
    s.banner.appendChild(h('button', { class: 'btn btn-sm btn-ghost', onClick: () => fovea.api.post(`/api/jobs/${job.id}/cancel`) }, 'Cancel'));
    fovea.api.watchJob(job.id, (snap) => { txt.textContent = jobText(snap); bar.className = `progress ${snap.total ? '' : 'indeterminate'}`; bar.firstChild.style.width = `${Math.round((snap.progress || 0) * 100)}%`; })
      .then(async (snap) => { s.watchingJob = null; s.banner.classList.add('hidden'); if (snap.status === 'done') { ui.toast(`${snap.kind === 'scan' ? 'Index' : snap.kind} finished · ${fmt.num(snap.result && snap.result.images)} images`, { type: 'ok' }); }
        await fovea.loadDatasets(); await refreshDataset(ds.id); fovea.bus.emit('dataset:scanned', ds.id); })
      .catch(async (e) => { s.watchingJob = null; s.banner.classList.add('hidden'); ui.toast('Job failed: ' + e.message, { type: 'error' }); await refreshDataset(ds.id); });
  }
  function jobText(j) { const k = j.kind === 'scan' ? 'Indexing' : j.kind; return `${k}${j.total ? ` ${fmt.num(j.done)} / ${fmt.num(j.total)}` : ''}${j.message ? ` · ${j.message}` : ''}`; }
}
