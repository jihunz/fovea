// Explore tab: filterable gallery + Inspect overlay + review actions.
import { ISSUE_LABELS } from './issues.js';

export function install(fovea) { fovea.registerTab({ id: 'explore', label: 'Explore', icon: 'grid', order: 20, render }); }

const REVIEW_KEYS = { a: 'approved', f: 'flagged', x: 'excluded', u: null };

async function render(el, { ctx, dataset, fovea }) {
  const { h, icon, ui, fmt, cls, isTyping, MOD } = fovea;
  const { classColor, className, REVIEW } = fovea.colors;
  const ds = dataset;
  const names = ds.classes || [];
  const q0 = ctx.query;
  const filters = { split: q0.get('split') || '', cls: q0.get('cls') || '', labeled: q0.get('labeled') || '', review: q0.get('review') || '', issue: q0.get('issue') || '', q: q0.get('q') || '', seq: q0.get('seq') || '' };
  let sort = q0.get('sort') || 'id', order = q0.get('order') || 'asc';
  let tileSize = localStorage.getItem('fovea.tile') || 'm';
  let showOverlay = localStorage.getItem('fovea.overlay') !== '0';
  const selected = new Set();
  let focusIdx = -1, lastClickIdx = -1;
  let cursor = null, loadingMore = false, destroyed = false;
  const TILE_PX = { s: 150, m: 210, l: 300, xl: 420 };

  const root = h('div', { class: 'explore' });
  const toolbar = h('div', { class: 'explore-toolbar' });
  const body = h('div', { class: 'explore-body' });
  const grid = h('div', { class: 'grid', style: `--tile:${TILE_PX[tileSize]}px` });
  const sentinel = h('div', { class: 'load-more' });
  const countEl = h('span', { class: 'result-count' });
  const bulk = h('div', { class: 'bulk-bar hidden' });
  body.appendChild(grid); body.appendChild(sentinel); body.appendChild(bulk);
  root.appendChild(toolbar); root.appendChild(body); el.appendChild(root);

  // ---------------------------------------------------------------- toolbar
  const searchInput = h('input', { class: 'input', placeholder: 'Search filename…', value: filters.q });
  searchInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { filters.q = searchInput.value.trim(); reload(); } if (e.key === 'Escape') searchInput.blur(); });
  const splitSeg = ui.seg([{ value: '', label: 'All' }, ...(ds.splits || []).filter(Boolean).map(s => ({ value: s, label: s }))], filters.split, (v) => { filters.split = v; reload(); });
  const clsBtn = h('button', { class: 'btn' }, icon('tags', 14), h('span', { class: 'cls-label' }, 'Classes'), icon('chevronDown', 12));
  clsBtn.addEventListener('click', () => {
    const sel = new Set(filters.cls ? filters.cls.split(',').map(Number) : []);
    const content = h('div', { class: 'col gap-4' },
      h('div', { class: 'row' }, h('span', { class: 'small strong muted grow' }, 'Contains class'), h('button', { class: 'btn btn-ghost btn-sm', onClick: () => { filters.cls = ''; pop.close(); reload(); } }, 'Clear')),
      names.length ? names.map((n, i) => h('label', { class: 'check' }, h('input', { type: 'checkbox', checked: sel.has(i), onChange: (e) => { e.target.checked ? sel.add(i) : sel.delete(i); filters.cls = [...sel].sort((a, b) => a - b).join(','); reload(); } }), h('span', { class: 'swatch', style: `background:${classColor(i)}` }), `${i} · ${n}`)) : h('div', { class: 'faint small' }, 'No classes'));
    const pop = ui.popover(clsBtn, content);
  });
  const labeledSel = select([['', 'Any label state'], ['1', 'Labeled (has boxes)'], ['0', 'No label file'], ['empty', 'Empty label file'], ['nobox', 'No boxes']], filters.labeled, v => { filters.labeled = v; reload(); });
  const reviewSel = select([['', 'Any review'], ['none', 'Unreviewed'], ['any', 'Reviewed'], ['approved', '✓ Approved'], ['flagged', '⚑ Flagged'], ['excluded', '✕ Excluded']], filters.review, v => { filters.review = v; reload(); });
  const issueSel = select([['', 'Any issue state'], ['any', 'Has issues'], ...Object.entries(ISSUE_LABELS).map(([k, v]) => [k, v])], filters.issue, v => { filters.issue = v; reload(); });
  const sortSel = select([['id', 'Order: index'], ['name', 'Order: name'], ['boxes', 'Order: boxes'], ['size', 'Order: resolution'], ['mtime', 'Order: modified'], ['reviewed', 'Order: reviewed'], ['random', 'Order: random']], sort, v => { sort = v; reload(); });
  const orderBtn = h('button', { class: 'btn btn-icon', 'data-tip': 'Toggle direction', onClick: () => { order = order === 'asc' ? 'desc' : 'asc'; orderBtn.innerHTML = ''; orderBtn.appendChild(icon(order === 'asc' ? 'arrowUp' : 'arrowDown', 14)); reload(); } }, icon(order === 'asc' ? 'arrowUp' : 'arrowDown', 14));
  const overlayBtn = h('button', { class: cls('btn btn-icon', showOverlay && 'active'), 'data-tip': 'Toggle box overlay (O)', onClick: () => toggleOverlay() }, icon('square', 14));
  const sizeSeg = ui.seg([{ value: 's', label: 'S' }, { value: 'm', label: 'M' }, { value: 'l', label: 'L' }, { value: 'xl', label: 'XL' }], tileSize, (v) => { tileSize = v; localStorage.setItem('fovea.tile', v); grid.style.setProperty('--tile', TILE_PX[v] + 'px'); });
  toolbar.appendChild(h('div', { class: 'input-wrap' }, icon('search'), searchInput));
  toolbar.appendChild(splitSeg); toolbar.appendChild(clsBtn); toolbar.appendChild(labeledSel); toolbar.appendChild(reviewSel); toolbar.appendChild(issueSel);
  toolbar.appendChild(h('span', { class: 'spacer' }));
  toolbar.appendChild(countEl); toolbar.appendChild(sortSel); toolbar.appendChild(orderBtn); toolbar.appendChild(overlayBtn); toolbar.appendChild(sizeSeg);
  toolbar.appendChild(h('button', { class: 'btn btn-primary', 'data-tip': 'Annotate this selection', onClick: () => fovea.router.navigate(`/d/${ds.id}/annotate?${qs()}`) }, icon('pen', 14), 'Annotate'));

  function select(opts, value, onChange) { const s = h('select', { class: 'select', style: 'width:auto', onChange: (e) => onChange(e.target.value) }, opts.map(([v, l]) => h('option', { value: v, selected: v === value }, l))); return s; }
  function activeFilters() { const f = {}; for (const [k, v] of Object.entries(filters)) if (v !== '' && v != null) f[k] = v; return f; }
  function qs(extra = {}) { const p = new URLSearchParams(); for (const [k, v] of Object.entries({ ...activeFilters(), sort: sort !== 'id' ? sort : '', order: order !== 'asc' ? order : '', ...extra })) if (v) p.set(k, v); return p.toString(); }
  function syncUrl() { fovea.router.replaceQuery({ ...activeFilters(), sort: sort !== 'id' ? sort : null, order: order !== 'asc' ? order : null }); }
  function updateClsLabel() { const n = filters.cls ? filters.cls.split(',').length : 0; clsBtn.querySelector('.cls-label').textContent = n ? `${n} class${n > 1 ? 'es' : ''}` : 'Classes'; clsBtn.classList.toggle('active', n > 0); }

  // ---------------------------------------------------------------- data
  const tiles = new Map(); // index -> element
  async function reload() {
    syncUrl(); updateClsLabel();
    selected.clear(); focusIdx = -1; updateBulk();
    cursor = new fovea.ImageCursor({ dsId: ds.id, filters: activeFilters(), sort, order, pageSize: 100, boxes: true });
    grid.innerHTML = ''; tiles.clear();
    countEl.textContent = '…';
    await loadMore();
  }
  async function loadMore() {
    if (!cursor || loadingMore || destroyed) return;
    const have = cursor.loadedCount;
    if (cursor.total != null && have >= cursor.total) { sentinel.innerHTML = ''; return; }
    loadingMore = true; sentinel.innerHTML = ''; sentinel.appendChild(ui.spinner());
    const page = Math.floor(have / cursor.pageSize);
    try {
      const res = await cursor.loadPage(page);
      if (destroyed) return;
      res.items.forEach((it, i) => appendTile(page * cursor.pageSize + i, it));
      countEl.textContent = `${fmt.num(cursor.total)} image${cursor.total === 1 ? '' : 's'}`;
      if (cursor.total === 0) grid.appendChild(ui.emptyState({ icon: 'images', title: 'No images match', message: 'Try clearing a filter.', action: h('button', { class: 'btn', onClick: () => { Object.keys(filters).forEach(k => filters[k] = ''); searchInput.value = ''; reload(); } }, 'Clear filters') }));
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
    finally { loadingMore = false; sentinel.innerHTML = ''; if (cursor && cursor.total != null && cursor.loadedCount < cursor.total) sentinel.appendChild(h('button', { class: 'btn', onClick: loadMore }, `Load more (${fmt.num(cursor.total - cursor.loadedCount)} left)`)); }
  }
  const io = new IntersectionObserver((entries) => { if (entries.some(e => e.isIntersecting)) loadMore(); }, { root: body, rootMargin: '600px' });
  io.observe(sentinel);

  // ---------------------------------------------------------------- tiles
  function appendTile(index, it) {
    const t = tile(index, it);
    tiles.set(index, t);
    grid.appendChild(t);
  }
  function tile(index, it) {
    const el = h('div', { class: cls('tile', selected.has(it.id) && 'selected', focusIdx === index && 'focus'), 'data-review': it.review ? it.review.status : '', 'data-idx': index });
    const pic = h('div', { class: 'pic', style: it.width && it.height ? `aspect-ratio:${it.width}/${it.height}` : '' }, h('img', { src: fovea.api.thumbUrl(it.id, tileSize === 'xl' || tileSize === 'l' ? 512 : 256), loading: 'lazy', decoding: 'async', alt: '' }));
    if (showOverlay && it.boxes && it.boxes.length) pic.appendChild(fovea.boxLayer(it.boxes, { names, stroke: 1.5 }));
    pic.style.width = '100%'; pic.style.height = '100%';
    // fit within frame: use object-fit via wrapper sizing
    const frame = h('div', { class: 'frame' }, fitWrap(pic, it));
    el.appendChild(frame);
    el.appendChild(h('div', { class: 'badges' }, it.n_boxes ? h('span', { class: 'badge' }, String(it.n_boxes)) : null, it.split ? h('span', { class: 'badge' }, it.split) : null));
    el.appendChild(h('div', { class: 'rv' }, icon(it.review ? REVIEW[it.review.status].icon : 'check', 12)));
    el.appendChild(h('div', { class: 'sel', onClick: (e) => { e.stopPropagation(); toggleSelect(index, it, e.shiftKey); } }, icon('check', 12)));
    el.appendChild(h('div', { class: 'meta' }, it.issues && it.issues.length ? h('span', { class: 'issue-dot', 'data-tip': it.issues.map(c => ISSUE_LABELS[c] || c).join(', ') }) : null, h('span', { class: 'name', title: it.rel_path }, it.rel_path.split('/').pop()), it.width ? h('span', { class: 'faint xs' }, `${it.width}×${it.height}`) : null));
    el.addEventListener('click', (e) => {
      if (e.shiftKey || e.metaKey || e.ctrlKey) { toggleSelect(index, it, e.shiftKey); return; }
      openInspect(index);
    });
    el.addEventListener('contextmenu', (e) => { e.preventDefault(); tileMenu({ x: e.clientX, y: e.clientY }, index, it); });
    return el;
  }
  function fitWrap(pic, it) {
    // wrapper that keeps the picture's aspect ratio inside a 4:3 frame
    const w = it.width || 4, hh = it.height || 3; const frameRatio = 4 / 3; const r = w / hh;
    const wrap = h('div', { style: `position:relative;${r >= frameRatio ? 'width:100%;height:auto;' : 'height:100%;width:auto;'}aspect-ratio:${w}/${hh};max-width:100%;max-height:100%` });
    wrap.appendChild(pic);
    return wrap;
  }
  function refreshTile(index) { const old = tiles.get(index); const it = cursor.items[index]; if (!old || !it) return; const n = tile(index, it); tiles.set(index, n); old.replaceWith(n); }
  function tileMenu(pos, index, it) {
    ui.menu(pos, [
      { label: 'Inspect', icon: 'eye', onClick: () => openInspect(index), kbd: '↵' },
      { label: 'Annotate', icon: 'pen', onClick: () => fovea.router.navigate(`/d/${ds.id}/annotate?${qs({ img: it.id })}`), kbd: 'E' },
      { sep: true },
      { label: 'Approve', icon: 'check', onClick: () => review([it.id], 'approved'), kbd: 'A' },
      { label: 'Flag', icon: 'flag', onClick: () => review([it.id], 'flagged'), kbd: 'F' },
      { label: 'Exclude', icon: 'ban', onClick: () => review([it.id], 'excluded'), kbd: 'X' },
      { label: 'Clear review', icon: 'rotate', onClick: () => review([it.id], null), kbd: 'U' },
      { sep: true },
      { label: 'Copy path', icon: 'copy', onClick: async () => { const { item } = await fovea.api.get(`/api/datasets/${ds.id}/images/${it.id}`); ui.copyText(item.abs_path_host); } },
      { label: 'Open original', icon: 'external', onClick: () => window.open(fovea.api.imgUrl(it.id), '_blank') },
    ]);
  }
  function toggleOverlay() { showOverlay = !showOverlay; localStorage.setItem('fovea.overlay', showOverlay ? '1' : '0'); overlayBtn.classList.toggle('active', showOverlay); for (const idx of tiles.keys()) refreshTile(idx); if (inspect) inspect.draw(); }

  // ---------------------------------------------------------------- selection & review
  function toggleSelect(index, it, range = false) {
    if (range && lastClickIdx >= 0) { const [a, b] = [Math.min(lastClickIdx, index), Math.max(lastClickIdx, index)]; for (let i = a; i <= b; i++) { const x = cursor.items[i]; if (x) selected.add(x.id); } }
    else { selected.has(it.id) ? selected.delete(it.id) : selected.add(it.id); }
    lastClickIdx = index;
    for (const [idx, t] of tiles) { const x = cursor.items[idx]; t.classList.toggle('selected', !!x && selected.has(x.id)); }
    grid.classList.toggle('selecting', selected.size > 0);
    updateBulk();
  }
  function clearSelection() { selected.clear(); for (const t of tiles.values()) t.classList.remove('selected'); grid.classList.remove('selecting'); updateBulk(); }
  function updateBulk() {
    bulk.classList.toggle('hidden', selected.size === 0);
    if (!selected.size) return;
    bulk.innerHTML = '';
    bulk.appendChild(h('span', { class: 'strong' }, `${fmt.num(selected.size)} selected`));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => review([...selected], 'approved') }, icon('check', 13), 'Approve'));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => review([...selected], 'flagged') }, icon('flag', 13), 'Flag'));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => review([...selected], 'excluded') }, icon('ban', 13), 'Exclude'));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => review([...selected], null) }, icon('rotate', 13), 'Clear'));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => fovea.router.navigate(`/d/${ds.id}/annotate?ids=${[...selected].join(',')}`) }, icon('pen', 13), 'Annotate'));
    bulk.appendChild(h('button', { class: 'btn', onClick: () => copyList() }, icon('copy', 13), 'Copy paths'));
    bulk.appendChild(h('button', { class: 'btn btn-icon', 'data-tip': 'Deselect (Esc)', onClick: clearSelection }, icon('x', 13)));
  }
  async function copyList() {
    const ids = [...selected];
    const text = await fovea.api.api(`/api/datasets/${ds.id}/export/list`, { method: 'POST', body: { filters: { ids: ids.join(',') }, style: 'host' } });
    ui.copyText(typeof text === 'string' ? text : '', `Copied ${ids.length} paths`);
  }
  async function review(ids, status, note) {
    try {
      await fovea.api.put(`/api/datasets/${ds.id}/review`, { image_ids: ids, status, note: note || '' });
      const set = new Set(ids);
      cursor.items.forEach((it, idx) => { if (it && set.has(it.id)) { it.review = status ? { status, note: note || '', updated_at: Date.now() / 1000 } : null; if (tiles.has(idx)) refreshTile(idx); } });
      fovea.bus.emit('review:changed', { ids, status });
      if (inspect) inspect.draw();
      ui.toast(status ? `${REVIEW[status].label}: ${ids.length} image${ids.length > 1 ? 's' : ''}` : `Cleared ${ids.length}`, { timeout: 1200 });
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  // ---------------------------------------------------------------- keyboard (grid)
  function cols() { const first = grid.querySelector('.tile'); if (!first) return 1; const w = first.getBoundingClientRect().width + 10; return Math.max(1, Math.floor(grid.clientWidth / w)); }
  function setFocus(i) {
    if (!cursor || cursor.total == null) return;
    i = Math.max(0, Math.min(cursor.total - 1, i));
    const prev = tiles.get(focusIdx); prev && prev.classList.remove('focus');
    focusIdx = i;
    const t = tiles.get(i); if (t) { t.classList.add('focus'); t.scrollIntoView({ block: 'nearest' }); } else cursor.ensure(i).then(() => loadMore());
  }
  const onKey = (e) => {
    if (isTyping() || ui.hasModal() || e.metaKey || e.ctrlKey || e.altKey) { if ((e.metaKey || e.ctrlKey) && e.key === 'a' && !isTyping() && !inspect) { e.preventDefault(); cursor.items.forEach(it => it && selected.add(it.id)); toggleSelect(focusIdx, {}, false); selected.size || 0; for (const [idx, t] of tiles) t.classList.add('selected'); grid.classList.add('selecting'); updateBulk(); } return; }
    if (inspect) return; // inspect has its own handler
    const k = e.key;
    if (k === 'ArrowRight') { e.preventDefault(); setFocus(focusIdx + 1); }
    else if (k === 'ArrowLeft') { e.preventDefault(); setFocus(focusIdx - 1); }
    else if (k === 'ArrowDown') { e.preventDefault(); setFocus(focusIdx + cols()); }
    else if (k === 'ArrowUp') { e.preventDefault(); setFocus(focusIdx - cols()); }
    else if (k === 'Enter') { if (focusIdx >= 0) openInspect(focusIdx); }
    else if (k === 'Escape') { clearSelection(); }
    else if (k === 's' || k === 'S') { const it = cursor.items[focusIdx]; if (it) toggleSelect(focusIdx, it, e.shiftKey); }
    else if (k === 'o' || k === 'O') toggleOverlay();
    else if (k === 'e' || k === 'E') { const it = cursor.items[focusIdx]; if (it) fovea.router.navigate(`/d/${ds.id}/annotate?${qs({ img: it.id })}`); }
    else if (k === '/') { e.preventDefault(); searchInput.focus(); }
    else if (k.toLowerCase() in REVIEW_KEYS) { const ids = selected.size ? [...selected] : (cursor.items[focusIdx] ? [cursor.items[focusIdx].id] : []); if (ids.length) review(ids, REVIEW_KEYS[k.toLowerCase()]); }
  };
  document.addEventListener('keydown', onKey);

  // ---------------------------------------------------------------- inspect overlay
  let inspect = null;
  function openInspect(index) {
    if (inspect) inspect.close();
    inspect = createInspect(index);
  }
  function createInspect(index) {
    let idx = index, item = cursor.items[idx], showLabels = true, auto = null, speed = 400, activeBox = -1;
    const stage = h('div', { class: 'inspect-stage' });
    const side = h('div', { class: 'inspect-side' });
    const top = h('div', { class: 'inspect-top' });
    const rootEl = h('div', { class: 'inspect' }, top, stage, side);
    const overlayToggle = h('button', { class: cls('btn btn-sm', showOverlay && 'active'), 'data-tip': 'Overlay (O)', onClick: () => toggleOverlay() }, icon('square', 13), 'Boxes');
    const labelsToggle = h('button', { class: cls('btn btn-sm', showLabels && 'active'), 'data-tip': 'Labels (L)', onClick: () => { showLabels = !showLabels; labelsToggle.classList.toggle('active', showLabels); draw(); } }, icon('tag', 13), 'Labels');
    const autoBtn = h('button', { class: 'btn btn-sm', 'data-tip': 'Auto-play (Space)', onClick: () => toggleAuto() }, icon('play', 13), 'Play');
    const speedInput = h('input', { type: 'range', class: 'slider', min: 50, max: 2000, step: 10, value: speed, style: 'width:90px', onInput: (e) => { speed = Number(e.target.value); speedLbl.textContent = speed + 'ms'; if (auto) { stopAuto(); startAuto(); } } });
    const speedLbl = h('span', { class: 'xs faint mono' }, speed + 'ms');
    const posEl = h('span', { class: 'mono small' });
    const nameEl = h('span', { class: 'name grow' });
    top.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Close (Esc)', onClick: () => close() }, icon('x', 14)));
    top.appendChild(posEl); top.appendChild(nameEl);
    top.appendChild(overlayToggle); top.appendChild(labelsToggle); top.appendChild(autoBtn); top.appendChild(speedInput); top.appendChild(speedLbl);
    top.appendChild(h('button', { class: 'btn btn-sm', onClick: () => fovea.router.navigate(`/d/${ds.id}/annotate?${qs({ img: item.id })}`) }, icon('pen', 13), 'Edit (E)'));
    stage.appendChild(h('button', { class: 'inspect-nav prev', onClick: () => go(-1) }, icon('chevronLeft')));
    stage.appendChild(h('button', { class: 'inspect-nav next', onClick: () => go(1) }, icon('chevronRight')));
    const picHolder = h('div', { class: 'pic' });
    stage.appendChild(picHolder);
    document.body.appendChild(rootEl);

    async function go(delta) {
      const n = idx + delta; if (n < 0 || (cursor.total != null && n >= cursor.total)) { if (auto) stopAuto(); return; }
      const it = await cursor.ensure(n); if (!it) return;
      idx = n; item = it; activeBox = -1; draw(); cursor.prefetch(idx, 3);
      const nxt = cursor.items[idx + 1]; if (nxt) { const im = new Image(); im.src = fovea.api.imgUrl(nxt.id); }
      if (cursor.loadedCount < (cursor.total || 0) && idx > cursor.loadedCount - 30) loadMore();
      setFocus(idx);
    }
    function draw() {
      if (!item) return;
      posEl.textContent = `${fmt.num(idx + 1)} / ${fmt.num(cursor.total)}`;
      nameEl.textContent = item.rel_path;
      picHolder.innerHTML = '';
      const img = h('img', { src: fovea.api.imgUrl(item.id), alt: '', draggable: false });
      picHolder.appendChild(img);
      if (showOverlay && item.boxes && item.boxes.length) picHolder.appendChild(fovea.boxLayer(item.boxes, { names, labels: showLabels, active: activeBox, onClick: (i) => { activeBox = i; draw(); } }));
      overlayToggle.classList.toggle('active', showOverlay);
      // side panel
      const rv = item.review;
      side.innerHTML = '';
      side.appendChild(h('div', { class: 'sec' }, h('h4', 'Review'),
        h('div', { class: 'row' }, ['approved', 'flagged', 'excluded'].map(s => h('button', { class: cls('btn btn-sm review-btn grow', s, rv && rv.status === s && 'active'), onClick: () => review([item.id], rv && rv.status === s ? null : s, noteEl.value) }, icon(REVIEW[s].icon, 13), REVIEW[s].label, h('span', { class: 'kbd', style: 'margin-left:auto' }, REVIEW[s].key)))),
        noteEl = h('textarea', { class: 'input mt-8', rows: 2, placeholder: 'Note (optional) — saved with the next review action', value: rv ? rv.note : '' }),
        rv ? h('div', { class: 'xs faint mt-8' }, `${REVIEW[rv.status].label} ${fmt.ago(rv.updated_at)}`) : h('div', { class: 'xs faint mt-8' }, 'Unreviewed')));
      side.appendChild(h('div', { class: 'sec' }, h('h4', `Labels`, h('span', { class: 'badge', style: 'margin-left:6px' }, String(item.n_boxes))),
        item.boxes && item.boxes.length ? item.boxes.map((b, i) => h('div', { class: cls('label-row', i === activeBox && 'active'), onMouseenter: () => { activeBox = i; refreshOverlayOnly(); }, onClick: () => { activeBox = i; refreshOverlayOnly(); } }, h('span', { class: 'swatch', style: `background:${classColor(b[0])}` }), h('span', { class: 'name' }, `${b[0]} · ${className(names, b[0])}`), h('span', { class: 'geo' }, `${(b[3] * 100).toFixed(0)}×${(b[4] * 100).toFixed(0)}%`)))
          : h('div', { class: 'small faint' }, item.has_label ? 'Empty label file' : 'No label file'),
        item.issues && item.issues.length ? h('div', { class: 'row wrap gap-4 mt-8' }, item.issues.map(c => ui.chip(ISSUE_LABELS[c] || c, { cls: 'issue-chip', icon: 'alert' }))) : null));
      side.appendChild(h('div', { class: 'sec' }, h('h4', 'Image'), h('div', { class: 'kv' },
        h('b', 'Split'), h('span', item.split || '—'), h('b', 'Size'), h('span', item.width ? `${item.width} × ${item.height} · ${fmt.bytes(item.size)}` : fmt.bytes(item.size)),
        h('b', 'Index'), h('span', { class: 'mono' }, `#${item.id}`), h('b', 'Sequence'), h('span', { class: 'mono' }, item.seq || '—'),
        h('b', 'Path'), h('span', { class: 'mono' }, item.rel_path)),
        h('div', { class: 'row mt-8' }, h('button', { class: 'btn btn-sm', onClick: async () => { const { item: full } = await fovea.api.get(`/api/datasets/${ds.id}/images/${item.id}`); ui.copyText(full.abs_path_host); } }, icon('copy', 12), 'Copy path'),
          h('button', { class: 'btn btn-sm', onClick: () => window.open(fovea.api.imgUrl(item.id), '_blank') }, icon('external', 12), 'Original'))));
      side.appendChild(h('div', { class: 'sec xs faint' }, h('div', '← → navigate · A/F/X review · U clear · O overlay · L labels · E edit · Space play · Esc close')));
    }
    let noteEl;
    function refreshOverlayOnly() { const ov = picHolder.querySelector('.ov'); if (ov) ov.remove(); if (showOverlay && item.boxes) picHolder.appendChild(fovea.boxLayer(item.boxes, { names, labels: showLabels, active: activeBox, onClick: (i) => { activeBox = i; refreshOverlayOnly(); } })); side.querySelectorAll('.label-row').forEach((r, i) => r.classList.toggle('active', i === activeBox)); }
    function startAuto() { auto = setInterval(() => go(1), speed); autoBtn.classList.add('active'); autoBtn.innerHTML = ''; autoBtn.appendChild(icon('pause', 13)); autoBtn.appendChild(document.createTextNode('Stop')); }
    function stopAuto() { clearInterval(auto); auto = null; autoBtn.classList.remove('active'); autoBtn.innerHTML = ''; autoBtn.appendChild(icon('play', 13)); autoBtn.appendChild(document.createTextNode('Play')); }
    function toggleAuto() { auto ? stopAuto() : startAuto(); }
    const key = (e) => {
      if (ui.hasModal()) return;
      if (isTyping()) { if (e.key === 'Escape') e.target.blur(); return; }
      const k = e.key;
      if (k === 'Escape') { e.preventDefault(); close(); }
      else if (k === 'ArrowRight' || k === 'ArrowDown') { e.preventDefault(); stopAuto(); go(1); }
      else if (k === 'ArrowLeft' || k === 'ArrowUp') { e.preventDefault(); stopAuto(); go(-1); }
      else if (k === 'Home') { e.preventDefault(); go(-idx); }
      else if (k === 'End') { e.preventDefault(); if (cursor.total) go(cursor.total - 1 - idx); }
      else if (k === ' ') { e.preventDefault(); toggleAuto(); }
      else if (k === 'o' || k === 'O') toggleOverlay();
      else if (k === 'l' || k === 'L') labelsToggle.click();
      else if (k === 'e' || k === 'E') fovea.router.navigate(`/d/${ds.id}/annotate?${qs({ img: item.id })}`);
      else if (k.toLowerCase() in REVIEW_KEYS && !e.metaKey && !e.ctrlKey) { const s = REVIEW_KEYS[k.toLowerCase()]; review([item.id], s, noteEl ? noteEl.value : ''); if (s && s !== 'flagged' && !auto) setTimeout(() => go(1), 120); }
    };
    document.addEventListener('keydown', key, true);
    const close = () => { stopAuto(); document.removeEventListener('keydown', key, true); rootEl.remove(); inspect = null; setFocus(idx); };
    draw(); setFocus(idx); cursor.prefetch(idx, 3);
    return { close, draw: () => { item = cursor.items[idx] || item; draw(); } };
  }

  // ---------------------------------------------------------------- lifecycle
  await reload();
  const offScan = fovea.bus.on('dataset:scanned', (id) => { if (id === ds.id) reload(); });
  const offLabels = fovea.bus.on('labels:changed', ({ item }) => { if (!cursor) return; const i = cursor.indexOfId(item.id); if (i >= 0) { cursor.items[i] = { ...cursor.items[i], ...item }; refreshTile(i); } });
  return () => { destroyed = true; document.removeEventListener('keydown', onKey); io.disconnect(); if (inspect) inspect.close(); offScan(); offLabels(); };
}
