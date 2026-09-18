// Dataset shell: header, scan banner, tabs (core + plugins). Keeps the shell mounted across tab switches.
let shell = null; // { dsId, el, content, tabBar, header, banner, tabId, destroyTab, jobWatch }
let tabSeq = 0;    // increments per tab render; a render that finishes after a newer one started is stale

/** Tear down one specific shell. Destroy callbacks outlive their render and must never touch a newer shell. */
function teardownShell(s) {
  try { s.destroyTab && s.destroyTab(); } catch (e) { console.error(e); }
  s.destroyTab = null;
  try { s.stopWatch && s.stopWatch(); } catch (e) { console.error(e); }
  s.stopWatch = null;
  s.tabToken = ++tabSeq;
  if (shell === s) shell = null;
}

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
    const s = buildShell(ds);                       // becomes the module-level `shell`
    // setView runs the previous view's destroy first; it is bound to ITS shell, so it cannot null this one.
    fovea.setView('dataset:' + id, s.el, () => teardownShell(s));
  } else {
    updateHeader(ds);
  }
  const s = shell;
  if (!s) return;
  if (s.tabId !== tab.id || !s.content.firstChild) {
    if (s.destroyTab) { try { s.destroyTab(); } catch (e) { console.error(e); } s.destroyTab = null; }
    const token = s.tabToken = ++tabSeq;
    s.tabId = tab.id;
    s.content.innerHTML = '';
    Array.from(s.tabBar.children).forEach(b => b.classList.toggle('active', b.dataset.tab === tab.id));
    document.title = `${ds.name} · ${tab.label} · Fovea`;
    const res = await tab.render(s.content, { ctx, dataset: ds, fovea, refresh: () => refreshDataset(ds.id) });
    if (typeof res === 'function') {
      // Tabs render asynchronously. If another tab (or dataset) took over meanwhile, this tab's listeners
      // are already live — tear it down now rather than leak its global key handlers into the next view.
      if (shell === s && s.tabToken === token) s.destroyTab = res;
      else { try { res(); } catch (e) { console.error(e); } }
    }
  }
  // Same dataset and tab: nothing to re-render. Tabs keep their query in sync via router.replaceQuery.

  function buildShell(ds) {
    const el = h('div', { style: 'display:flex;flex-direction:column;height:100%;min-height:0' });
    const header = h('div', { class: 'ds-header' });
    const banner = h('div', { class: 'ds-scan hidden' });
    const alert = h('div', { class: 'ds-alert hidden' });
    const tabBar = h('div', { class: 'tabs' });
    const content = h('div', { style: 'flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden' });
    tabs.forEach(t => tabBar.appendChild(h('button', { class: cls('tab', t.id === tab.id && 'active'), 'data-tab': t.id, onClick: () => fovea.router.navigate(`/d/${ds.id}/${t.id === 'overview' ? '' : t.id}`.replace(/\/$/, '')) }, icon(t.icon || 'circle', 15), t.label)));
    el.appendChild(header); el.appendChild(alert); el.appendChild(banner); el.appendChild(tabBar); el.appendChild(content);
    const s = { dsId: ds.id, el, header, alert, banner, tabBar, content, tabId: null, destroyTab: null, stopWatch: null };
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
    updateAlert(ds);
    watchJob(ds);
  }

  /** Loudly flag a dataset whose files are not on disk — otherwise it looks healthy while every image 404s. */
  function updateAlert(ds) {
    const s = shell; if (!s || !s.alert) return;
    const reach = ds.reachable || { ok: true };
    if (reach.ok) { s.alert.classList.add('hidden'); return; }
    const where = reach.missing_sources && reach.missing_sources.length ? reach.missing_sources[0] : ds.root_host;
    s.alert.classList.remove('hidden');
    s.alert.innerHTML = '';
    s.alert.appendChild(icon('alert', 15));
    s.alert.appendChild(h('div', { class: 'grow' },
      h('div', { class: 'strong' }, 'These files are not on this machine'),
      h('div', { class: 'mono truncate', style: 'max-width:70vw' }, where),
      h('div', { class: 'xs' }, 'The index still holds ' + fmt.num(ds.image_count) + ' images and your review marks. Point Fovea at the new location to restore them.')));
    s.alert.appendChild(h('button', { class: 'btn btn-sm', onClick: () => relocate(ds) }, icon('folderOpen', 13), 'Move to a new path…'));
  }

  async function relocate(ds) {
    const p = await fovea.pickPath({ title: `Where is “${ds.name}” now?`, files: true, start: ds.root_host, accept: ['.yaml', '.yml', '.txt'] });
    if (!p) return;
    try {
      await fovea.api.post(`/api/datasets/${ds.id}/relocate`, { path: p });
      ui.toast('Re-indexing from the new path…', { type: 'ok' });
      await fovea.loadDatasets();
      await refreshDataset(ds.id);
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  function headerMenu(anchor, ds) {
    ui.menu(anchor, [
      { label: 'Rescan index', icon: 'refresh', onClick: () => startScan(ds.id, false) },
      { label: 'Re-detect layout & rescan', icon: 'scan', onClick: () => startScan(ds.id, true) },
      { label: 'Move to a new path…', icon: 'folderOpen', onClick: () => relocate(ds) },
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
    const clock = ui.elapsedClock(Date.now() - (job.elapsed || 0) * 1000);
    s.banner.innerHTML = ''; s.banner.appendChild(icon('refresh', 14)); s.banner.appendChild(txt); s.banner.appendChild(clock); s.banner.appendChild(bar);
    s.banner.appendChild(h('button', { class: 'btn btn-sm btn-ghost', onClick: () => fovea.api.post(`/api/jobs/${job.id}/cancel`) }, 'Cancel'));
    const w = fovea.api.watchJob(job.id, (snap) => { txt.textContent = jobText(snap); bar.className = `progress ${snap.total ? '' : 'indeterminate'}`; bar.firstChild.style.width = `${Math.round((snap.progress || 0) * 100)}%`; });
    // One stream per shell, closed when the shell goes away — leaked EventSources exhaust the browser's
    // per-origin connection limit and stall every thumbnail and API call behind them.
    s.stopWatch = () => { w.stop(); s.watchingJob = null; };
    const what = job.kind === 'scan' ? 'Index' : job.kind;
    w.then(async (snap) => {
      s.watchingJob = null; s.stopWatch = null; s.banner.classList.add('hidden'); s.banner.replaceChildren();
      if (snap.status === 'done') ui.toast(`${what} finished · ${fmt.num(snap.result && snap.result.images)} images`, { type: 'ok' });
      else if (snap.status === 'cancelled') ui.toast(`${what} cancelled`);
      await fovea.loadDatasets();
      if (shell !== s) return;                     // never refresh a dataset the user has already left
      await refreshDataset(ds.id); fovea.bus.emit('dataset:scanned', ds.id);
    }).catch(async (e) => {
      s.watchingJob = null; s.stopWatch = null; s.banner.classList.add('hidden'); s.banner.replaceChildren();
      ui.toast('Job failed: ' + e.message, { type: 'error' });
      if (shell === s) await refreshDataset(ds.id);
    });
  }
  function jobText(j) { const k = j.kind === 'scan' ? 'Indexing' : j.kind; return `${k}${j.total ? ` ${fmt.num(j.done)} / ${fmt.num(j.total)}` : ''}${j.message ? ` · ${j.message}` : ''}`; }
}
