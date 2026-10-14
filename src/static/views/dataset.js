// Dataset shell: header (with the job status line), tabs (core + plugins). Keeps the shell mounted across tab switches.
//
// Nothing in the shell changes height while someone works: the running job shows in a fixed-height slot of the
// header's path line (never a banner that pushes the tabs down when it appears or goes), and the partial-index /
// files-missing notice row only appears or goes at a tab change — in between, a new problem waits in the status slot
// and a solved one stays as a static "resolved" line (docs/visual-comfort.md).
/** replaceChildren that skips null / false and flattens arrays (DOM replaceChildren would print "null"). */
const put = (el, ...c) => { el.replaceChildren(...c.flat(Infinity).filter(x => x != null && x !== false && x !== true)); return el; };
let shell = null; // { dsId, el, content, tabBar, header, jobSlot, alert, tabId, destroyTab, ... }
let tabSeq = 0;    // increments per tab render; a render that finishes after a newer one started is stale
export const DONE_LINE_MS = 4000;   // how long a finished job's one-line result stays in the status slot

const ACTIVE = ['queued', 'running'];
/** This dataset's active jobs in the order the status slot follows them: running before queued, oldest first.
    `finished` holds ids the slot already saw end (a poll fetched a moment earlier may still list them). */
export function jobsToFollow(jobs, dsId, { finished = new Set(), dsJob = null } = {}) {
  const live = (j) => j && j.dataset_id === dsId && ACTIVE.includes(j.status) && !finished.has(j.id);
  const list = (jobs || []).filter(live);
  if (dsJob && live({ ...dsJob, dataset_id: dsId }) && !list.some(j => j.id === dsJob.id)) list.unshift({ ...dsJob, dataset_id: dsId });
  return list.sort((a, b) => (a.status === 'running' ? 0 : 1) - (b.status === 'running' ? 0 : 1) || (a.created_at || 0) - (b.created_at || 0));
}
const DONE_TEXT = { restore: 'Labels restored', qa: 'Data quality checked', snapshot: 'Label version saved', export_copy: 'Export copied',
  ai_audit: 'Label audit finished', ai_accept: 'Proposals accepted', bulk_labels: 'Labels edited', objects_edit: 'Boxes edited' };
/** The one-line result a finished job leaves in the status slot: {tone, icon, text}. */
export function doneLine(label, kind, snap, err, num = (n) => String(n)) {
  if (err) return { tone: 'bad', icon: 'alertCircle', text: `${label} failed` };
  if (!snap || snap.status === 'cancelled') return { tone: 'muted', icon: 'x', text: `${label} cancelled` };
  const r = snap.result || {};
  if (kind === 'scan') return { tone: 'ok', icon: 'checkCircle', text: r.images != null ? `Index finished · ${num(r.images)} images` : 'Index finished' };
  const said = DONE_TEXT[kind];
  return { tone: 'ok', icon: 'checkCircle', text: said || `${label} finished` };
}
/** What the notice above the tabs should say: 'unreachable' | 'partial' | 'resuming' | null. `prev` is what it said
    before: a partial index that is being resumed reads 'scanning' on the server, and is still "resuming" here. */
export function alertKind(ds, jobs, prev = null) {
  const reach = (ds && ds.reachable) || { ok: true };
  if (!ds) return null;
  if (!reach.ok) return 'unreachable';
  const scanning = ds.status === 'scanning' || (jobs || []).some(j => j.dataset_id === ds.id && j.kind === 'scan' && ACTIVE.includes(j.status));
  if (ds.status === 'partial') return scanning ? 'resuming' : 'partial';
  if (scanning && (prev === 'partial' || prev === 'resuming')) return 'resuming';
  return null;
}

/** Tear down one specific shell. Destroy callbacks outlive their render and must never touch a newer shell. */
function teardownShell(s) {
  try { s.destroyTab && s.destroyTab(); } catch (e) { console.error(e); }
  s.destroyTab = null;
  try { s.stopWatch && s.stopWatch(); } catch (e) { console.error(e); }
  s.stopWatch = null;
  try { s.offJobs && s.offJobs(); } catch (e) { console.error(e); }
  clearTimeout(s.doneTimer); s.doneTimer = null;
  s.tabToken = ++tabSeq;
  if (shell === s) shell = null;
}

export function install(fovea) {}

export async function render(ctx, fovea, tabId) {
  const { h, icon, ui, fmt, link, cls } = fovea;
  const id = ctx.params.id;
  const tabs = fovea.tabs();
  const tab = tabs.find(t => t.id === tabId) || null;
  if (!tab) {
    fovea.setView('ds-tab-missing', ui.emptyState({ icon: 'alert', title: 'Page not found', message: `This dataset has no “${tabId}” page.`, action: link(`/d/${encodeURIComponent(id)}`, { class: 'btn' }, 'Overview') }));
    return;
  }

  let ds;
  try { ds = await fovea.getDataset(id, { touch: !shell || shell.dsId !== id }); }
  catch (e) {
    if (!fovea.router.isCurrent(ctx)) return;
    const missing = e.status === 404;
    document.title = 'Fovea';
    fovea.setView('ds-missing', ui.emptyState({ icon: 'alert', title: missing ? 'Dataset not found' : 'Could not open this dataset',
      message: missing ? `No dataset with the id “${id}” is registered.` : fovea.plainError(e),
      action: h('div', { class: 'row' }, missing ? null : h('button', { class: 'btn', onClick: () => fovea.router.navigate(location.pathname + location.search, { replace: true }) }, icon('refresh', 13), 'Try again'), link('/', { class: 'btn' }, 'Back home')) }));
    return;
  }
  // The user may have navigated on while the dataset loaded: never render it over another page's URL.
  if (!fovea.router.isCurrent(ctx)) return;

  if (!shell || shell.dsId !== id || !shell.el.isConnected) {
    const s = buildShell(ds);                       // becomes the module-level `shell`
    // setView runs the previous view's destroy first; it is bound to ITS shell, so it cannot null this one.
    fovea.setView('dataset:' + id, s.el, () => teardownShell(s));
  } else {
    updateHeader(ds);
  }
  const s = shell;
  if (!s) return;
  document.title = `${ds.name} · ${tab.label} · Fovea`;
  if (s.tabId !== tab.id || !s.content.firstChild) {
    await renderTab(s, tab, ctx, ds);
  } else if (ctx.search !== ctx.prevSearch) {
    // Same tab, another query (Back/Forward, a link with other filters). A tab that syncs its own query exposes
    // onQuery; any other tab is rendered again for the new query.
    const onQuery = s.destroyTab && s.destroyTab.onQuery;
    if (typeof onQuery === 'function') { if ([...ctx.query.keys()].length) { try { onQuery(ctx.query); } catch (e) { console.error(e); } } }
    else await renderTab(s, tab, ctx, ds);
  }

  async function renderTab(s, tab, ctx, ds) {
    if (s.destroyTab) { try { s.destroyTab(); } catch (e) { console.error(e); } s.destroyTab = null; }
    const token = s.tabToken = ++tabSeq;
    s.tabId = tab.id;
    // A new page is the moment the notice row may appear or go: a resolved one ("Index complete") kept its place
    // so nothing moved under the pointer, and a problem that arose meanwhile waited in the status slot.
    const cur = fovea.state.get().currentDataset;
    updateAlert(cur && cur.id === s.dsId ? cur : ds, { safe: true });
    put(s.content);
    Array.from(s.tabBar.children).forEach(b => { const on = b.dataset.tab === tab.id; b.classList.toggle('active', on); b.setAttribute('aria-selected', String(on)); b.tabIndex = on ? 0 : -1; });
    let res;
    try { res = await tab.render(s.content, { ctx, dataset: ds, fovea, refresh: () => refreshDataset(ds.id) }); }
    catch (e) {
      console.error(e);
      if (shell === s && s.tabToken === token) put(s.content, ui.emptyState({ icon: 'alert', title: `${tab.label} failed to load`, message: fovea.plainError(e), action: h('button', { class: 'btn', onClick: () => { s.tabId = null; renderTab(s, tab, ctx, ds); } }, icon('refresh', 13), 'Try again') }));
      return;
    }
    if (typeof res === 'function') {
      // Tabs render asynchronously. If another tab (or dataset) took over meanwhile, this tab's listeners
      // are already live — tear it down now rather than leak its global key handlers into the next view.
      if (shell === s && s.tabToken === token) s.destroyTab = res;
      else { try { res(); } catch (e) { console.error(e); } }
    }
  }

  function buildShell(ds) {
    const el = h('div', { class: 'ds-shell', style: 'display:flex;flex-direction:column;height:100%;min-height:0' });
    const header = h('header', { class: 'ds-header' });
    // The job status slot lives in the header's path line (fixed height): it never pushes the tabs down.
    const jobSlot = h('div', { class: 'ds-job', role: 'status', hidden: true });
    const alert = h('div', { class: 'ds-alert hidden', role: 'alert' });
    const tabBar = h('div', { class: 'tabs', role: 'tablist', 'aria-label': 'Dataset pages' });
    const content = h('div', { style: 'flex:1;min-height:0;display:flex;flex-direction:column;overflow:hidden', role: 'tabpanel' });
    const tabUrl = (t) => `/d/${encodeURIComponent(ds.id)}${t.id === 'overview' ? '' : '/' + t.id}`;
    tabs.forEach(t => tabBar.appendChild(h('a', { href: tabUrl(t), class: cls('tab', t.id === tab.id && 'active'), role: 'tab', 'aria-selected': String(t.id === tab.id), tabindex: t.id === tab.id ? '0' : '-1', 'data-tab': t.id,
      onClick: (e) => {
        if (shell && shell.tabId === t.id && !e.metaKey && !e.ctrlKey) e.preventDefault();
        // A tab picked with the pointer (detail > 0; Enter gives 0) gives its focus back: kept, the arrows, Home and
        // End would walk the tab bar instead of moving the image (Annotate) or the tile (Explore, Objects).
        if (e.detail > 0) e.currentTarget.blur();
      } }, icon(t.icon || 'circle', 15), t.label)));
    // Arrow keys move along the tab bar (the tablist pattern) while a tab has keyboard focus; Enter opens.
    // Modified arrows are the browser's (⌘← / Alt+← go back).
    tabBar.addEventListener('keydown', (e) => {
      if (e.key !== 'ArrowRight' && e.key !== 'ArrowLeft' && e.key !== 'Home' && e.key !== 'End') return;
      if (e.altKey || e.metaKey || e.ctrlKey || e.shiftKey) return;
      const list = [...tabBar.children], i = list.indexOf(document.activeElement);
      if (i < 0) return;
      const n = e.key === 'Home' ? 0 : e.key === 'End' ? list.length - 1 : (i + (e.key === 'ArrowRight' ? 1 : -1) + list.length) % list.length;
      e.preventDefault(); list.forEach((b, j) => { b.tabIndex = j === n ? 0 : -1; }); list[n].focus();
    });
    el.appendChild(header); el.appendChild(alert); el.appendChild(tabBar); el.appendChild(content);
    const s = { dsId: ds.id, el, header, alert, alertState: null, alertKey: null, notice: null, jobSlot, tabBar, content, tabId: null, destroyTab: null, stopWatch: null, watchingJob: null, doneTimer: null, finished: new Set() };
    shell = s;
    // Jobs of this dataset come from the shell's one cheap poll (no per-page polling).
    s.offJobs = fovea.state.subscribe(() => {
      if (shell !== s) return;
      syncBanner();
      const cur = fovea.state.get().currentDataset;
      if (cur && cur.id === s.dsId) updateAlert(cur);
    }, (st) => st.jobs);
    updateHeader(ds);
    return s;
  }

  function updateHeader(ds) {
    const s = shell; if (!s) return;
    const kind = ds.layout && ds.layout.kind;
    const reach = ds.reachable || { ok: true };
    const st = !reach.ok ? 'unreachable' : ds.status;
    const stText = !reach.ok ? 'files not found' : ds.status === 'ready' ? `indexed ${fmt.ago(ds.scanned_at)}` : ds.status === 'scanning' ? 'indexing' : ds.status === 'new' ? 'not indexed yet' : ds.status === 'partial' ? 'partly indexed' : ds.status;
    put(s.header, 
      h('div', { class: 'ds-top' },
        h('h1', { title: ds.name }, h('span', { class: 'ds-name' }, ds.name)),
        kind ? ui.chip(kind, { type: 'accent', title: 'Detected folder layout' }) : null,
        h('span', { class: 'status-pill', title: ds.scanned_at ? `Indexed ${fmt.date(ds.scanned_at)}` : null }, h('span', { class: `dot ${st}` }), stText),
        h('span', { class: 'spacer' }),
        h('span', { class: 'ds-counts' }, ds.status === 'scanning' && !ds.image_count ? 'Counting images…' : `${fmt.num(ds.image_count)} images · ${fmt.num(ds.label_count)} labeled · ${fmt.num(ds.box_count)} boxes · ${fmt.plural((ds.classes || []).length, 'class', 'classes')}`),
        h('button', { class: 'btn btn-sm btn-icon', 'aria-label': 'Dataset actions', 'aria-haspopup': 'menu', 'data-tip': 'Rescan, rename, move…', onClick: (e) => headerMenu(e.currentTarget, ds) }, icon('moreH', 14))),
      h('div', { class: cls('path', !s.jobSlot.hidden && 'busy') }, ui.pathText(ds.root_host, { tail: 36 }),
        h('button', { class: 'btn btn-ghost btn-icon btn-sm copy', 'data-tip': 'Copy path', 'aria-label': 'Copy path', onClick: () => ui.copyText(ds.root_host) }, icon('copy', 12)),
        s.jobSlot));
    if (fovea.router.current && fovea.router.current.params && fovea.router.current.params.id === ds.id) {
      const t = tabs.find(x => x.id === s.tabId);
      if (t) document.title = `${ds.name} · ${t.label} · Fovea`;
    }
    updateAlert(ds);
    syncBanner(ds);
  }

  /** Loudly flag a dataset whose files are not on disk — otherwise it looks healthy while every image 404s. */
  /** The notice row above the tabs (partial index, files not found). It only appears or goes at a safe point — a new
      shell or a tab change, when the content is drawn anyway; in between, a new problem waits in the status slot
      and a solved one stays as a static "resolved" line, so the tabs never move under the pointer. */
  function updateAlert(ds, { safe = false } = {}) {
    const s = shell; if (!s || !s.alert || !ds || ds.id !== s.dsId) return;
    safe = safe || !s.content.firstChild;
    const kind = alertKind(ds, fovea.state.get().jobs, s.alertState === 'resolved' ? null : s.alertState);
    const visible = !s.alert.classList.contains('hidden');
    const show = (tone, ...kids) => { s.alert.className = `ds-alert ${tone}`; put(s.alert, ...kids); };
    if (kind && !visible && !safe) { setNotice(s, kind, ds); return; }
    setNotice(s, null);
    if (kind === 'partial' || kind === 'resuming') {
      // A first index that was cancelled or cut short by a restart: the counts cover only part of the folder.
      // While it resumes the notice keeps its height (same two lines, no button).
      const key = `${kind}|${ds.image_count}`;
      if (s.alertKey === key && visible) return;
      s.alertState = kind; s.alertKey = key;
      show('warn', icon('alert', 15),
        h('div', { class: 'grow' }, h('div', { class: 'strong' }, kind === 'resuming' ? 'Resuming the index' : 'Partial index'),
          h('div', { class: 'xs' }, kind === 'resuming' ? `Indexing the rest of the folder — progress is shown above. ${fmt.num(ds.image_count)} images so far.` : `Indexing stopped before it finished — ${fmt.num(ds.image_count)} images so far. Resuming skips what is already indexed.`)),
        kind === 'partial' ? h('button', { class: 'btn btn-sm', onClick: () => startScan(ds.id, false) }, icon('refresh', 13), 'Resume indexing') : null);
      return;
    }
    if (kind === 'unreachable') {
      const reach = ds.reachable || {};
      const where = reach.missing_sources && reach.missing_sources.length ? reach.missing_sources[0] : ds.root_host;
      const key = `${kind}|${where}|${ds.image_count}`;
      if (s.alertKey === key && visible) return;
      s.alertState = kind; s.alertKey = key;
      show('', icon('alert', 15),
        h('div', { class: 'grow' },
          h('div', { class: 'strong' }, 'These files are not on this machine'),
          h('div', { class: 'mono' }, ui.pathText(where, { tail: 40 })),
          h('div', { class: 'xs' }, 'The index still holds ' + fmt.num(ds.image_count) + ' images and your review marks. Point Fovea at the new location to restore them.')),
        h('button', { class: 'btn btn-sm', onClick: () => relocate(ds) }, icon('folderOpen', 13), 'Move to a new path…'));
      return;
    }
    if (!visible) { s.alertState = null; s.alertKey = null; return; }
    if (safe || s.alertState === 'resolved') {
      if (safe) { s.alert.classList.add('hidden'); put(s.alert); s.alertState = null; s.alertKey = null; }
      return;
    }
    // The problem went away while the notice was on screen: a static "resolved" line until the next tab change.
    const was = s.alertState;
    s.alertState = 'resolved'; s.alertKey = null;
    show('ok', icon('checkCircle', 15),
      h('div', { class: 'grow' }, h('div', { class: 'strong' }, was === 'unreachable' ? 'Files found again' : 'Index complete'),
        h('div', { class: 'xs' }, `${fmt.num(ds.image_count)} images indexed. This note goes away when you open another page.`)),
      h('button', { class: 'btn btn-sm btn-ghost', onClick: () => { s.alert.classList.add('hidden'); put(s.alert); s.alertState = null; } }, 'Close'));
  }
  /** A problem that arose while a page was open waits in the status slot (fixed height) until the next tab change. */
  function setNotice(s, kind, ds) {
    const key = kind ? `${kind}|${ds.image_count}` : null;
    if ((s.notice && s.notice.key) === key) return;
    s.notice = kind && kind !== 'resuming' ? { key, kind, ds } : null;
    if (!s.watchingJob && !s.doneTimer) idleSlot(s);
  }


  async function relocate(ds) {
    const p = await fovea.pickPath({ title: `Where is “${ds.name}” now?`, files: true, start: ds.root_host, accept: ['.yaml', '.yml', '.txt'] });
    if (!p) return;
    try {
      await fovea.api.post(`/api/datasets/${encodeURIComponent(ds.id)}/relocate`, { path: p });
      ui.toast('Re-indexing from the new path…', { type: 'ok' });
      fovea.jobs.refresh();
      await fovea.loadDatasets();
      await refreshDataset(ds.id);
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  function headerMenu(anchor, ds) {
    ui.menu(anchor, [
      { label: 'Rescan index', icon: 'refresh', onClick: () => startScan(ds.id, false) },
      { label: 'Re-detect layout & rescan', icon: 'scan', onClick: () => startScan(ds.id, true) },
      { label: 'Move to a new path…', icon: 'folderOpen', onClick: () => relocate(ds) },
      { label: 'Rename…', icon: 'pen', onClick: async () => {
        const v = await ui.prompt({ title: 'Rename dataset', label: 'Name', value: ds.name, validate: (x) => x.trim() ? null : 'The name cannot be empty' });
        if (v == null || !v.trim() || v.trim() === ds.name) return;
        try { await fovea.api.patch(`/api/datasets/${encodeURIComponent(ds.id)}`, { name: v.trim() }); await refreshDataset(ds.id); await fovea.loadDatasets(); }
        catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
      } },
      { label: 'Copy path', icon: 'copy', onClick: () => ui.copyText(ds.root_host) },
      { label: 'Save a version of the labels', icon: 'history', onClick: () => saveVersion(ds) },
      { label: 'API: dataset JSON', icon: 'external', onClick: () => window.open(`/api/datasets/${encodeURIComponent(ds.id)}`, '_blank', 'noopener') },
      { sep: true },
      { label: 'Remove from Fovea…', icon: 'trash', danger: true, onClick: async () => {
        if (!await ui.confirm({ title: `Remove “${ds.name}”?`, message: 'Removes the local index, review marks and label versions from Fovea. Files on disk are untouched.', okLabel: 'Remove', danger: true })) return;
        try {
          await fovea.api.del(`/api/datasets/${encodeURIComponent(ds.id)}`);
          fovea.state.set({ currentDataset: null });
          await fovea.loadDatasets().catch(() => {});
          // Replace, not push: Back must not lead to the page of a dataset that no longer exists.
          fovea.router.navigate('/', { replace: true });
          ui.toast('Dataset removed');
        } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
      } },
    ]);
  }

  async function saveVersion(ds) {
    const name = await ui.prompt({ title: 'Save a version of the labels', label: 'Name', placeholder: 'e.g. Before relabeling class 3', okLabel: 'Save' });
    if (name == null) return;
    try {
      const r = await fovea.api.post(`/api/datasets/${encodeURIComponent(ds.id)}/snapshots`, { name: name.trim() });
      ui.toast(r.snapshot_id ? 'Label version saved' : 'Saving the label version…', { type: 'ok' });
      fovea.bus.emit('snapshots:changed', { dataset: ds.id });
      fovea.jobs.refresh();
    } catch (e) { ui.toast(fovea.plainError(e), { type: 'error' }); }
  }

  async function startScan(dsId, redetect) {
    try {
      const res = await fovea.api.post(`/api/datasets/${encodeURIComponent(dsId)}/scan`, { redetect });
      if (res.already_running) ui.toast('Indexing is already running');
      await fovea.jobs.refresh();
      await refreshDataset(dsId);
    } catch (e) {
      ui.toast(fovea.plainError(e), { type: 'error', action: /not found|unreachable/i.test(e.message || '') ? { label: 'Move to a new path…', onClick: () => relocate(fovea.state.get().currentDataset || { id: dsId, name: dsId }) } : undefined });
    }
  }

  async function refreshDataset(dsId) {
    const d = await fovea.getDataset(dsId, { force: true });
    if (shell && shell.dsId === dsId) updateHeader(d);
    fovea.bus.emit('dataset:updated', d);
    return d;
  }

  /** Show or empty the job slot. The path line keeps its height either way; with a job the path shrinks to its tail. */
  function setSlot(s, ...kids) {
    clearTimeout(s.doneTimer); s.doneTimer = null;
    const on = kids.length > 0;
    s.jobSlot.hidden = !on;
    if (s.jobSlot.parentNode) s.jobSlot.parentNode.classList.toggle('busy', on);
    put(s.jobSlot, ...kids);
  }

  /** The slot with no job to follow: the waiting notice, if any, else nothing (the path gets the whole line). */
  function idleSlot(s) {
    const n = s.notice;
    if (!n) { setSlot(s); return; }
    s.jobSlot.className = 'ds-job notice';
    if (n.kind === 'partial') setSlot(s, icon('alert', 13), h('span', { class: 'txt', title: 'Indexing stopped before it finished' }, `Partial index · ${fmt.num(n.ds.image_count)} images so far`),
      h('button', { class: 'btn btn-sm btn-ghost', onClick: () => startScan(n.ds.id, false) }, 'Resume indexing'));
    else setSlot(s, icon('alert', 13), h('span', { class: 'txt' }, 'Files not found at the registered path'),
      h('button', { class: 'btn btn-sm btn-ghost', onClick: () => relocate(n.ds) }, 'Move to a new path…'));
  }

  /** The status slot follows this dataset's active jobs: the oldest one live over SSE, "+ n more" for the rest. */
  function syncBanner(dsArg) {
    const s = shell; if (!s) return;
    const ds = dsArg || fovea.state.get().currentDataset;
    // A job this slot already saw finish may linger in a list fetched a moment earlier: never follow it twice.
    const jobsHere = jobsToFollow(fovea.state.get().jobs, s.dsId, { finished: s.finished, dsJob: ds && ds.id === s.dsId ? ds.job : null });
    if (!jobsHere.length) { if (!s.watchingJob && !s.doneTimer) idleSlot(s); return; }
    const job = (s.watchingJob && jobsHere.find(j => j.id === s.watchingJob)) || jobsHere[0];
    if (s.watchingJob === job.id) { if (s.moreEl) s.moreEl.textContent = jobsHere.length > 1 ? `+${jobsHere.length - 1} more` : ''; return; }
    watch(s, job, jobsHere.length - 1);
  }

  function watch(s, job, more) {
    if (s.stopWatch) { s.stopWatch(); s.stopWatch = null; }
    s.watchingJob = job.id;
    const bar = ui.progress(job.progress || 0, { indeterminate: !job.total });
    const txt = h('span', { class: 'txt' }, fovea.jobs.text(job));
    const clock = ui.elapsedClock(Date.now() - (job.elapsed || 0) * 1000);
    const moreEl = s.moreEl = h('button', { class: 'btn btn-sm btn-ghost', 'data-tip': 'Other jobs of this dataset (Activity)', onClick: () => fovea.openActivity() }, more > 0 ? `+${more} more` : '');
    s.jobSlot.className = `ds-job ${job.kind === 'scan' ? 'warn' : 'info'}`;
    txt.title = txt.textContent;
    setSlot(s, icon(fovea.jobs.icon(job.kind), 13), txt, clock, bar, moreEl,
      h('button', { class: 'btn btn-sm btn-ghost', onClick: async (e) => {
        e.currentTarget.disabled = true;
        try { await fovea.api.post(`/api/jobs/${job.id}/cancel`); } catch (err) { ui.toast(fovea.plainError(err), { type: 'error' }); }
      } }, 'Cancel'));
    const w = fovea.api.watchJob(job.id, (snap) => {
      txt.textContent = txt.title = fovea.jobs.text(snap);
      bar.className = `progress ${snap.total ? '' : 'indeterminate'}`;
      bar.firstChild.style.width = `${snap.total ? Math.round((snap.progress || 0) * 100) : 40}%`;
    }, { onConnection: (ok) => { if (!ok) txt.textContent = `${fovea.jobs.label(job.kind)} · reconnecting…`; } });
    // One stream per shell, closed when the shell goes away — leaked EventSources exhaust the browser's
    // per-origin connection limit and stall every thumbnail and API call behind them.
    s.stopWatch = () => { w.stop(); s.watchingJob = null; };
    const label = fovea.jobs.label(job.kind);
    const settle = async (snap, err) => {
      if (s.watchingJob !== job.id) return;
      s.watchingJob = null; s.stopWatch = null; s.finished.add(job.id);
      clock.stop();
      // The result stays in place as a static line for a few seconds (no toast, no animation); the next job of
      // this dataset, if one is waiting, takes the slot at once — and then a finished index still gets its toast.
      const line = doneLine(label, job.kind, snap, err, fmt.num);
      s.jobSlot.className = `ds-job done ${line.tone}`;
      setSlot(s, icon(line.icon, 13), h('span', { class: 'txt', title: line.text }, line.text));
      s.doneTimer = setTimeout(() => { s.doneTimer = null; if (shell === s && !s.watchingJob) idleSlot(s); }, DONE_LINE_MS);
      if (err) ui.toast(`${label} failed: ${fovea.plainError(err)}`, { type: 'error' });
      await fovea.jobs.refresh();
      // Another job of this dataset took the slot right away (a scan is followed by the quality checks): the
      // finished index still deserves a word.
      if (!err && snap.status === 'done' && (job.kind === 'scan' || job.kind === 'restore') && shell === s && s.watchingJob) ui.toast(line.text, { type: 'ok' });
      if (shell !== s) return;                     // never refresh a dataset the user has already left
      try { await refreshDataset(s.dsId); } catch (e) { /* the header keeps its last state */ }
      if (shell !== s) return;
      if (job.kind === 'scan' || job.kind === 'restore') fovea.bus.emit('dataset:scanned', s.dsId);
      fovea.bus.emit('job:finished', { dataset: s.dsId, kind: job.kind, status: err ? 'error' : snap.status, job: snap || job });
      syncBanner();
    };
    w.then((snap) => settle(snap, null)).catch((e) => settle(null, e));
  }
}
