// Annotate tab: zoomable canvas labeling with autosave, class palette, AI auto-label, range tools.
export function install(fovea) { fovea.registerTab({ id: 'annotate', label: 'Annotate', icon: 'pen', order: 30, render }); }

const HANDLES = ['nw', 'n', 'ne', 'e', 'se', 's', 'sw', 'w'];

async function render(el, { ctx, dataset, fovea, refresh }) {
  const { h, icon, ui, fmt, cls, isTyping, MOD } = fovea;
  const { classColor, withAlpha, className, REVIEW } = fovea.colors;
  const ds = dataset;
  let names = [...(ds.classes || [])];
  const q0 = ctx.query;
  const filters = {}; for (const k of ['split', 'cls', 'labeled', 'review', 'issue', 'q', 'seq', 'ids']) if (q0.get(k)) filters[k] = q0.get(k);
  const sort = q0.get('sort') || 'id', order = q0.get('order') || 'asc';
  const cursor = new fovea.ImageCursor({ dsId: ds.id, filters, sort, order, pageSize: 200, boxes: false });

  // ---------------------------------------------------------------- state
  let idx = -1, item = null, img = null, boxes = [], sel = -1, tool = 'select', activeCls = 0;
  let scale = 1, tx = 0, ty = 0, fitScale = 1, customView = false, prevDims = null;
  let undo = [], redo = [];
  let dirty = false, saveTimer = null, saving = false, destroyed = false;
  let mouse = { x: 0, y: 0, inside: false }, drag = null, spaceDown = false;
  const opts = { propagate: localStorage.getItem('fovea.ann.propagate') === '1', labels: localStorage.getItem('fovea.ann.labels') !== '0', pointW: 0.15, pointH: 0.30, crosshair: true };
  const ai = { model: '', conf: 0.25, classes: '', mode: 'fill', map: {}, modelClasses: null, keepUnmapped: false, job: null };
  let rangeStart = null;
  let classBuf = '', classBufTimer = null;
  let autoTimer = null;

  // ---------------------------------------------------------------- DOM
  const root = h('div', { class: 'ann' });
  const tools = h('div', { class: 'ann-tools' });
  const stage = h('div', { class: 'ann-stage' });
  const side = h('div', { class: 'ann-side' });
  const bottom = h('div', { class: 'ann-bottom' });
  const canvas = h('canvas');
  const hud = h('div', { class: 'hud' });
  const stageMsg = h('div', { class: 'stage-msg' }, 'Loading…');
  stage.appendChild(canvas); stage.appendChild(hud); stage.appendChild(stageMsg);
  root.appendChild(tools); root.appendChild(stage); root.appendChild(side); root.appendChild(bottom);
  el.appendChild(root);
  const g = canvas.getContext('2d');

  // ---- left panel
  const toolBtns = {};
  const toolRow = h('div', { class: 'tool-row' }, [['select', 'pointer', 'Select', 'V'], ['box', 'boxSelect', 'Box', 'B'], ['point', 'crosshair', 'Point', 'P'], ['pan', 'hand', 'Pan', 'H']].map(([id, ic, label, key]) =>
    toolBtns[id] = h('button', { class: cls('tool-btn', tool === id && 'active'), 'data-tip': `${label} (${key})`, onClick: () => setTool(id) }, icon(ic, 16), h('span', label))));
  const classList = h('div', { class: 'class-list' });
  const newClassInput = h('input', { class: 'input input-sm', placeholder: 'New class name…', onKeydown: async (e) => { if (e.key === 'Enter' && newClassInput.value.trim()) { await addClass(newClassInput.value.trim()); newClassInput.value = ''; } } });
  const pointOpts = h('div', { class: cls('col gap-4', tool !== 'point' && 'hidden') });
  const propagateSw = ui.switchBtn(opts.propagate, v => { opts.propagate = v; localStorage.setItem('fovea.ann.propagate', v ? '1' : '0'); });
  const labelsSw = ui.switchBtn(opts.labels, v => { opts.labels = v; localStorage.setItem('fovea.ann.labels', v ? '1' : '0'); draw(); });
  const aiPanel = h('div', { class: 'ai-panel col gap-8' });
  tools.appendChild(h('div', { class: 'sec' }, h('h4', 'Tools'), toolRow));
  tools.appendChild(h('div', { class: 'sec' }, h('h4', 'Classes', h('span', { class: 'spacer' }), h('span', { class: 'xs faint' }, '0–9 hotkeys')), classList, h('div', { class: 'mt-8' }, newClassInput)));
  tools.appendChild(h('div', { class: 'sec' }, h('h4', 'Options'),
    h('div', { class: 'row', style: 'justify-content:space-between;padding:3px 0' }, h('span', { class: 'small' }, 'Propagate boxes to next unlabeled'), propagateSw),
    h('div', { class: 'row', style: 'justify-content:space-between;padding:3px 0' }, h('span', { class: 'small' }, 'Show labels (L)'), labelsSw),
    pointOpts));
  tools.appendChild(h('div', { class: 'sec' }, h('h4', icon('sparkles', 13), 'AI auto-label'), aiPanel));
  tools.appendChild(h('div', { class: 'sec xs faint' }, 'Wheel zoom · Space+drag pan · F fit · Tab cycle · Del delete · ⌘Z undo · C copy from previous · N next unlabeled · ⇧A/⇧F/⇧X review'));
  renderPointOpts();

  // ---- right panel
  const boxList = h('div', { class: 'col gap-4' });
  const boxCount = h('span', { class: 'badge' }, '0');
  const reviewRow = h('div', { class: 'row' });
  const infoKv = h('div', { class: 'kv small', style: 'display:grid;grid-template-columns:64px 1fr;gap:3px 8px' });
  const issuesRow = h('div', { class: 'row wrap gap-4 mt-8' });
  const rangeStatus = h('div', { class: 'xs faint mt-8' }, 'No range start');
  const rangeCls = h('select', { class: 'select select-sm', style: 'width:auto' });
  const rangeApply = h('button', { class: 'btn btn-sm btn-primary', disabled: true, onClick: () => applyRange() }, 'Apply to range');
  const saveState = h('span', { class: 'save-state' }, icon('check', 12), 'Saved');
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Boxes', boxCount, h('span', { class: 'spacer' }), h('button', { class: 'btn btn-ghost btn-sm', 'data-tip': 'Delete all boxes', onClick: () => { if (boxes.length) { pushHistory(); boxes = []; sel = -1; changed(); } } }, icon('trash', 12))), boxList));
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Review', h('span', { class: 'spacer' }), h('span', { class: 'xs faint' }, '⇧A ⇧F ⇧X')), reviewRow));
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Image'), infoKv, issuesRow));
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Range edit'), h('div', { class: 'small muted mb-8' }, 'Set the class of every box across a range of images (e.g. a fall sequence).'),
    h('div', { class: 'row' }, h('button', { class: 'btn btn-sm', onClick: () => { rangeStart = item ? { id: item.id, idx, name: item.rel_path } : null; rangeApply.disabled = !rangeStart; rangeStatus.textContent = rangeStart ? `Start: #${idx + 1} ${fovea.fmt ? '' : ''}${rangeStart.name.split('/').pop()}` : 'No range start'; } }, icon('bookmark', 12), 'Mark start'), rangeCls, rangeApply), rangeStatus));

  // ---- bottom bar
  const idxInput = h('input', { class: 'input input-sm idx', value: '' });
  idxInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') { const n = parseInt(idxInput.value, 10); if (!isNaN(n)) goTo(n - 1); idxInput.blur(); } if (e.key === 'Escape') idxInput.blur(); });
  idxInput.addEventListener('focus', () => idxInput.select());
  const nameEl = h('span', { class: 'name grow' });
  const zoomEl = h('span', { class: 'xs mono faint', style: 'width:44px;text-align:right' }, '100%');
  const playBtn = h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Auto-advance', onClick: () => toggleAuto() }, icon('play', 13));
  const speed = h('input', { type: 'range', class: 'slider', min: 100, max: 3000, step: 50, value: 600, style: 'width:70px', 'data-tip': 'Auto-advance interval', onInput: () => { if (autoTimer) { stopAuto(); startAuto(); } } });
  bottom.appendChild(h('div', { class: 'btn-group' },
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'First', onClick: () => goTo(0) }, icon('chevronsLeft', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Previous (←)', onClick: () => goTo(idx - 1) }, icon('chevronLeft', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Next (→)', onClick: () => goTo(idx + 1) }, icon('chevronRight', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Last', onClick: () => goTo((cursor.total || 1) - 1) }, icon('chevronsRight', 14))));
  bottom.appendChild(idxInput);
  bottom.appendChild(h('button', { class: 'btn btn-sm', 'data-tip': 'Jump to next image without boxes (N)', onClick: () => nextUnlabeled() }, icon('zap', 13), 'Next unlabeled'));
  bottom.appendChild(playBtn); bottom.appendChild(speed);
  bottom.appendChild(nameEl);
  bottom.appendChild(saveState);
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Fit (F)', onClick: () => { fit(); draw(); } }, icon('maximize', 13)));
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Zoom out', onClick: () => zoomBy(1 / 1.25) }, icon('zoomOut', 13)));
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Zoom in', onClick: () => zoomBy(1.25) }, icon('zoomIn', 13)));
  bottom.appendChild(zoomEl);

  // ---------------------------------------------------------------- panels rendering
  let classCounts = {};
  fovea.api.get(`/api/datasets/${ds.id}/stats`).then(r => { r.stats.classes.forEach(c => classCounts[c.cls] = c.boxes); renderClasses(); }).catch(() => {});
  function renderClasses() {
    classList.innerHTML = '';
    if (!names.length) classList.appendChild(h('div', { class: 'small faint' }, 'No classes yet — add one below, or just draw (class 0).'));
    names.forEach((n, i) => classList.appendChild(h('div', { class: cls('class-row', i === activeCls && 'active'), onClick: () => setActiveClass(i) },
      h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'name', title: `${i} · ${n}` }, n), h('span', { class: 'n', 'data-tip': 'boxes in dataset' }, classCounts[i] != null ? fmt.compact(classCounts[i]) : ''), i < 10 ? h('span', { class: 'kbd' }, String(i)) : h('span', { class: 'kbd', style: 'opacity:.5' }, String(i)))));
    rangeCls.innerHTML = ''; names.forEach((n, i) => rangeCls.appendChild(h('option', { value: i }, `${i} ${n}`)));
  }
  function renderPointOpts() {
    pointOpts.innerHTML = '';
    pointOpts.classList.toggle('hidden', tool !== 'point');
    const w = h('input', { type: 'range', class: 'slider', min: 0.02, max: 0.6, step: 0.005, value: opts.pointW, onInput: (e) => { opts.pointW = Number(e.target.value); wl.textContent = opts.pointW.toFixed(2); draw(); } });
    const hh = h('input', { type: 'range', class: 'slider', min: 0.02, max: 0.9, step: 0.005, value: opts.pointH, onInput: (e) => { opts.pointH = Number(e.target.value); hl.textContent = opts.pointH.toFixed(2); draw(); } });
    const wl = h('span', { class: 'xs mono' }, opts.pointW.toFixed(2)), hl = h('span', { class: 'xs mono' }, opts.pointH.toFixed(2));
    pointOpts.appendChild(h('div', { class: 'row small' }, h('span', { style: 'width:44px' }, 'Width'), w, wl));
    pointOpts.appendChild(h('div', { class: 'row small' }, h('span', { style: 'width:44px' }, 'Height'), hh, hl));
    pointOpts.appendChild(h('div', { class: 'xs faint' }, 'Point tool: click to drop a fixed-size box. Wheel adjusts width, ⇧wheel height.'));
  }
  function renderBoxes() {
    boxCount.textContent = String(boxes.length);
    boxList.innerHTML = '';
    if (!boxes.length) boxList.appendChild(h('div', { class: 'small faint' }, item && item.has_label ? 'No boxes (empty label)' : 'No boxes yet'));
    boxes.forEach((b, i) => boxList.appendChild(h('div', { class: cls('box-row', i === sel && 'active'), onClick: () => { sel = i; draw(); renderBoxes(); }, onMouseenter: () => { hover = i; draw(); }, onMouseleave: () => { hover = -1; draw(); } },
      h('span', { class: 'idx' }, String(i + 1)), h('span', { class: 'swatch', style: `background:${classColor(b[0])}` }),
      h('span', { class: 'name' }, `${b[0]} · ${className(names, b[0])}`), h('span', { class: 'geo' }, `${Math.round(b[3] * 100)}×${Math.round(b[4] * 100)}%`),
      h('button', { class: 'btn btn-ghost btn-sm btn-icon del', onClick: (e) => { e.stopPropagation(); pushHistory(); boxes.splice(i, 1); sel = -1; changed(); } }, icon('x', 12)))));
  }
  let hover = -1;
  function renderInfo() {
    if (!item) return;
    const rv = item.review;
    reviewRow.innerHTML = '';
    ['approved', 'flagged', 'excluded'].forEach(s => reviewRow.appendChild(h('button', { class: cls('btn btn-sm review-btn grow', s, rv && rv.status === s && 'active'), onClick: () => setReview(rv && rv.status === s ? null : s) }, icon(REVIEW[s].icon, 12), REVIEW[s].label)));
    infoKv.innerHTML = '';
    [['Name', item.rel_path.split('/').pop()], ['Split', item.split || '—'], ['Size', item.width ? `${item.width} × ${item.height}` : '—'], ['Seq', item.seq || '—'], ['Index', `#${item.id}`]].forEach(([k, v]) => { infoKv.appendChild(h('b', { class: 'faint' }, k)); infoKv.appendChild(h('span', { class: 'mono truncate', title: v }, v)); });
    issuesRow.innerHTML = '';
    (item.issues || []).forEach(c => issuesRow.appendChild(ui.chip(c.replace(/_/g, ' '), { type: 'warn', icon: 'alert' })));
    nameEl.textContent = item.rel_path; nameEl.title = item.rel_path;
    idxInput.value = `${idx + 1} / ${fmt.num(cursor.total)}`;
    if (!isTyping()) idxInput.blur();
  }
  function setSaveState(s, msg) { saveState.className = `save-state ${s}`; saveState.innerHTML = ''; saveState.appendChild(icon(s === 'saved' ? 'check' : s === 'dirty' ? 'clock' : s === 'saving' ? 'refresh' : 'alertCircle', 12)); saveState.appendChild(document.createTextNode(msg || (s === 'saved' ? 'Saved' : s === 'dirty' ? 'Unsaved' : s === 'saving' ? 'Saving…' : 'Save failed'))); }
  function updateHud() {
    hud.innerHTML = '';
    hud.appendChild(ui.chip(tool, { cls: 'mono' }));
    if (img) hud.appendChild(ui.chip(`${img.naturalWidth}×${img.naturalHeight}`, { cls: 'mono' }));
    hud.appendChild(ui.chip(`${activeCls} ${className(names, activeCls)}`, { cls: 'mono' }));
    zoomEl.textContent = Math.round(scale * 100) + '%';
  }

  // ---------------------------------------------------------------- classes
  function setActiveClass(i) { activeCls = i; if (sel >= 0 && boxes[sel][0] !== i) { pushHistory(); boxes[sel][0] = i; changed(); } renderClasses(); updateHud(); draw(); }
  async function addClass(name) {
    names.push(name);
    try { await fovea.api.patch(`/api/datasets/${ds.id}`, { classes: names }); await refresh(); ui.toast(`Added class ${names.length - 1}: ${name}`, { type: 'ok' }); }
    catch (e) { ui.toast(e.message, { type: 'error' }); names.pop(); }
    renderClasses();
  }

  // ---------------------------------------------------------------- view transforms
  function resizeCanvas() {
    const r = stage.getBoundingClientRect(); const dpr = window.devicePixelRatio || 1;
    canvas.width = Math.max(1, Math.round(r.width * dpr)); canvas.height = Math.max(1, Math.round(r.height * dpr));
    canvas.style.width = r.width + 'px'; canvas.style.height = r.height + 'px';
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!customView) fit();
    draw();
  }
  function fit() {
    if (!img) return;
    const cw = stage.clientWidth, ch = stage.clientHeight;
    fitScale = Math.min(cw / img.naturalWidth, ch / img.naturalHeight) * 0.96;
    scale = fitScale; tx = (cw - img.naturalWidth * scale) / 2; ty = (ch - img.naturalHeight * scale) / 2; customView = false; updateHud();
  }
  function zoomAt(sx, sy, factor) {
    if (!img) return;
    const ns = Math.max(fitScale * 0.15, Math.min(60, scale * factor));
    tx = sx - (sx - tx) * (ns / scale); ty = sy - (sy - ty) * (ns / scale); scale = ns; customView = Math.abs(scale - fitScale) > 1e-6; updateHud(); draw();
  }
  function zoomBy(f) { zoomAt(stage.clientWidth / 2, stage.clientHeight / 2, f); }
  const toImg = (sx, sy) => ({ x: (sx - tx) / scale, y: (sy - ty) / scale });
  const rectOf = (b) => { const iw = img.naturalWidth, ih = img.naturalHeight; return { x: (b[1] - b[3] / 2) * iw * scale + tx, y: (b[2] - b[4] / 2) * ih * scale + ty, w: b[3] * iw * scale, h: b[4] * ih * scale }; };
  const normOf = (x0, y0, x1, y1) => { const iw = img.naturalWidth, ih = img.naturalHeight; const a = toImg(x0, y0), b = toImg(x1, y1); let ax = Math.max(0, Math.min(iw, Math.min(a.x, b.x))), bx = Math.max(0, Math.min(iw, Math.max(a.x, b.x))), ay = Math.max(0, Math.min(ih, Math.min(a.y, b.y))), by = Math.max(0, Math.min(ih, Math.max(a.y, b.y))); return [(ax + bx) / 2 / iw, (ay + by) / 2 / ih, (bx - ax) / iw, (by - ay) / ih]; };
  function clampBox(b) { const x0 = Math.max(0, b[1] - b[3] / 2), y0 = Math.max(0, b[2] - b[4] / 2), x1 = Math.min(1, b[1] + b[3] / 2), y1 = Math.min(1, b[2] + b[4] / 2); b[1] = (x0 + x1) / 2; b[2] = (y0 + y1) / 2; b[3] = Math.max(0.001, x1 - x0); b[4] = Math.max(0.001, y1 - y0); return b; }

  // ---------------------------------------------------------------- drawing
  function draw() {
    const cw = stage.clientWidth, ch = stage.clientHeight;
    g.clearRect(0, 0, cw, ch);
    if (!img) return;
    g.imageSmoothingEnabled = scale < 2;
    g.drawImage(img, tx, ty, img.naturalWidth * scale, img.naturalHeight * scale);
    boxes.forEach((b, i) => {
      const r = rectOf(b); const color = classColor(b[0]); const active = i === sel;
      g.fillStyle = withAlpha(color, active ? 0.22 : (i === hover ? 0.18 : 0.10)); g.fillRect(r.x, r.y, r.w, r.h);
      g.lineWidth = active ? 2.5 : 1.75; g.strokeStyle = active ? '#fff' : color; g.strokeRect(r.x, r.y, r.w, r.h);
      if (active) { g.lineWidth = 1.5; g.strokeStyle = color; g.strokeRect(r.x + 2, r.y + 2, r.w - 4, r.h - 4); }
      if (opts.labels) {
        const t = `${b[0]} ${className(names, b[0])}`; g.font = '600 11px Inter, sans-serif'; const tw = g.measureText(t).width + 8;
        const ly = r.y - 16 < 0 ? r.y : r.y - 16;
        g.fillStyle = color; g.fillRect(r.x, ly, tw, 16); g.fillStyle = '#fff'; g.textBaseline = 'middle'; g.fillText(t, r.x + 4, ly + 8);
      }
      if (active) HANDLES.forEach(hn => { const p = handlePos(r, hn); g.fillStyle = '#fff'; g.strokeStyle = color; g.lineWidth = 1.5; g.beginPath(); g.rect(p.x - 4, p.y - 4, 8, 8); g.fill(); g.stroke(); });
    });
    if (drag && drag.kind === 'draw') {
      const x = Math.min(drag.x0, mouse.x), y = Math.min(drag.y0, mouse.y), w = Math.abs(mouse.x - drag.x0), hh = Math.abs(mouse.y - drag.y0);
      g.setLineDash([5, 3]); g.strokeStyle = classColor(activeCls); g.lineWidth = 1.5; g.strokeRect(x, y, w, hh); g.fillStyle = withAlpha(classColor(activeCls), 0.12); g.fillRect(x, y, w, hh); g.setLineDash([]);
      g.fillStyle = '#fff'; g.font = '10px JetBrains Mono, monospace'; g.textBaseline = 'bottom'; g.fillText(`${Math.round(w / scale)}×${Math.round(hh / scale)}`, x, y - 3);
    }
    if (mouse.inside && !drag && (tool === 'box' || tool === 'point') && !spaceDown) {
      g.save(); g.strokeStyle = 'rgba(255,255,255,.45)'; g.lineWidth = 1; g.setLineDash([4, 4]);
      g.beginPath(); g.moveTo(0, mouse.y + .5); g.lineTo(cw, mouse.y + .5); g.moveTo(mouse.x + .5, 0); g.lineTo(mouse.x + .5, ch); g.stroke();
      if (tool === 'point') { const w = opts.pointW * img.naturalWidth * scale, hh = opts.pointH * img.naturalHeight * scale; g.setLineDash([5, 3]); g.strokeStyle = classColor(activeCls); g.lineWidth = 1.5; g.strokeRect(mouse.x - w / 2, mouse.y - hh / 2, w, hh); }
      g.restore();
    }
  }
  function handlePos(r, hn) { const cx = r.x + r.w / 2, cy = r.y + r.h / 2; return { nw: { x: r.x, y: r.y }, n: { x: cx, y: r.y }, ne: { x: r.x + r.w, y: r.y }, e: { x: r.x + r.w, y: cy }, se: { x: r.x + r.w, y: r.y + r.h }, s: { x: cx, y: r.y + r.h }, sw: { x: r.x, y: r.y + r.h }, w: { x: r.x, y: cy } }[hn]; }
  function handleAt(sx, sy) { if (sel < 0 || !boxes[sel]) return null; const r = rectOf(boxes[sel]); for (const hn of HANDLES) { const p = handlePos(r, hn); if (Math.abs(sx - p.x) <= 7 && Math.abs(sy - p.y) <= 7) return hn; } return null; }
  function boxAt(sx, sy) {
    let best = -1, bestArea = Infinity;
    boxes.forEach((b, i) => { const r = rectOf(b); const pad = 3; if (sx >= r.x - pad && sx <= r.x + r.w + pad && sy >= r.y - pad && sy <= r.y + r.h + pad) { const a = r.w * r.h; if (a < bestArea) { bestArea = a; best = i; } } });
    return best;
  }
  function cursorFor(sx, sy) {
    if (spaceDown || tool === 'pan') return drag ? 'grabbing' : 'grab';
    const hn = handleAt(sx, sy);
    if (hn) return { n: 'ns-resize', s: 'ns-resize', e: 'ew-resize', w: 'ew-resize', ne: 'nesw-resize', sw: 'nesw-resize', nw: 'nwse-resize', se: 'nwse-resize' }[hn];
    if (tool === 'select') return boxAt(sx, sy) >= 0 ? 'move' : 'default';
    if (tool === 'box') return boxAt(sx, sy) >= 0 && sel === boxAt(sx, sy) ? 'move' : 'crosshair';
    return 'none';
  }

  // ---------------------------------------------------------------- mouse
  const local = (e) => { const r = canvas.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  canvas.addEventListener('mousedown', (e) => {
    if (!img) return; const p = local(e); mouse = { ...p, inside: true };
    if (e.button === 1 || spaceDown || tool === 'pan') { drag = { kind: 'pan', x0: p.x, y0: p.y, tx0: tx, ty0: ty }; e.preventDefault(); canvas.style.cursor = 'grabbing'; return; }
    if (e.button !== 0) return;
    const hn = handleAt(p.x, p.y);
    if (hn) { pushHistory(); drag = { kind: 'resize', hn, orig: [...boxes[sel]] }; return; }
    const hit = boxAt(p.x, p.y);
    if (tool === 'point') {
      if (hit >= 0 && hit === sel) { drag = { kind: 'move', x0: p.x, y0: p.y, orig: [...boxes[sel]], pushed: false }; return; }
      const c = toImg(p.x, p.y); const iw = img.naturalWidth, ih = img.naturalHeight;
      if (c.x < 0 || c.y < 0 || c.x > iw || c.y > ih) return;
      pushHistory(); boxes.push(clampBox([activeCls, c.x / iw, c.y / ih, opts.pointW, opts.pointH])); sel = boxes.length - 1; changed(); return;
    }
    if (hit >= 0 && (tool === 'select' || hit === sel)) { sel = hit; drag = { kind: 'move', x0: p.x, y0: p.y, orig: [...boxes[sel]], pushed: false }; renderBoxes(); draw(); return; }
    if (tool === 'box') { sel = -1; drag = { kind: 'draw', x0: p.x, y0: p.y }; renderBoxes(); draw(); return; }
    if (sel !== -1) { sel = -1; renderBoxes(); draw(); }
  });
  window.addEventListener('mousemove', onMove);
  function onMove(e) {
    if (!img || destroyed) return; const p = local(e); mouse = { ...p, inside: mouse.inside };
    if (!drag) { if (mouse.inside) { canvas.style.cursor = cursorFor(p.x, p.y); const hb = tool === 'select' ? boxAt(p.x, p.y) : -1; if (hb !== hover) { hover = hb; } draw(); } return; }
    if (drag.kind === 'pan') { tx = drag.tx0 + (p.x - drag.x0); ty = drag.ty0 + (p.y - drag.y0); customView = true; draw(); return; }
    if (drag.kind === 'draw') { draw(); return; }
    if (drag.kind === 'move' && sel >= 0) {
      if (!drag.pushed) { pushHistory(); drag.pushed = true; }
      const dx = (p.x - drag.x0) / (img.naturalWidth * scale), dy = (p.y - drag.y0) / (img.naturalHeight * scale);
      const b = boxes[sel]; b[1] = Math.max(b[3] / 2, Math.min(1 - b[3] / 2, drag.orig[1] + dx)); b[2] = Math.max(b[4] / 2, Math.min(1 - b[4] / 2, drag.orig[2] + dy)); draw(); return;
    }
    if (drag.kind === 'resize' && sel >= 0) {
      const o = drag.orig; const iw = img.naturalWidth, ih = img.naturalHeight;
      let x0 = (o[1] - o[3] / 2) * iw, y0 = (o[2] - o[4] / 2) * ih, x1 = (o[1] + o[3] / 2) * iw, y1 = (o[2] + o[4] / 2) * ih;
      const c = toImg(p.x, p.y); const hn = drag.hn;
      if (hn.includes('w')) x0 = c.x; if (hn.includes('e')) x1 = c.x; if (hn.includes('n')) y0 = c.y; if (hn.includes('s')) y1 = c.y;
      if (x0 > x1) [x0, x1] = [x1, x0]; if (y0 > y1) [y0, y1] = [y1, y0];
      x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(iw, x1); y1 = Math.min(ih, y1);
      boxes[sel] = [o[0], (x0 + x1) / 2 / iw, (y0 + y1) / 2 / ih, Math.max(1, x1 - x0) / iw, Math.max(1, y1 - y0) / ih]; draw();
    }
  }
  window.addEventListener('mouseup', onUp);
  function onUp(e) {
    if (!drag || destroyed) return; const p = local(e); const d = drag; drag = null;
    if (d.kind === 'draw') {
      if (Math.abs(p.x - d.x0) >= 4 && Math.abs(p.y - d.y0) >= 4) { pushHistory(); const nb = normOf(d.x0, d.y0, p.x, p.y); if (nb[2] > 0.0005 && nb[3] > 0.0005) { boxes.push([activeCls, ...nb]); sel = boxes.length - 1; changed(); return; } }
      draw(); return;
    }
    if (d.kind === 'move') { if (d.pushed) changed(); else { renderBoxes(); draw(); } return; }
    if (d.kind === 'resize') { changed(); return; }
    canvas.style.cursor = cursorFor(p.x, p.y); draw();
  }
  canvas.addEventListener('mouseenter', () => { mouse.inside = true; });
  canvas.addEventListener('mouseleave', () => { mouse.inside = false; hover = -1; draw(); });
  canvas.addEventListener('wheel', (e) => {
    e.preventDefault(); if (!img) return;
    if (tool === 'point' && !e.ctrlKey && !e.metaKey) { const d = e.deltaY > 0 ? -0.01 : 0.01; if (e.shiftKey) opts.pointH = Math.max(0.02, Math.min(0.9, opts.pointH + d)); else opts.pointW = Math.max(0.02, Math.min(0.6, opts.pointW + d)); renderPointOpts(); draw(); return; }
    const p = local(e); zoomAt(p.x, p.y, Math.exp(-e.deltaY * 0.0022));
  }, { passive: false });
  canvas.addEventListener('contextmenu', (e) => { e.preventDefault(); const p = local(e); const hit = boxAt(p.x, p.y); if (hit < 0) return; sel = hit; renderBoxes(); draw();
    ui.menu({ x: e.clientX, y: e.clientY }, [{ label: 'Change class', header: true }, ...names.slice(0, 20).map((n, i) => ({ label: `${i} · ${n}`, icon: i === boxes[hit][0] ? 'check' : 'tag', onClick: () => { pushHistory(); boxes[hit][0] = i; changed(); } })), { sep: true }, { label: 'Delete box', icon: 'trash', danger: true, onClick: () => { pushHistory(); boxes.splice(hit, 1); sel = -1; changed(); } }]); });
  canvas.addEventListener('dblclick', (e) => { const p = local(e); if (boxAt(p.x, p.y) < 0) { fit(); draw(); } });
  const ro = new ResizeObserver(() => resizeCanvas()); ro.observe(stage);

  // ---------------------------------------------------------------- history & save
  function pushHistory() { undo.push(JSON.stringify(boxes)); if (undo.length > 60) undo.shift(); redo = []; }
  function doUndo() { if (!undo.length) return; redo.push(JSON.stringify(boxes)); boxes = JSON.parse(undo.pop()); sel = Math.min(sel, boxes.length - 1); changed(false); }
  function doRedo() { if (!redo.length) return; undo.push(JSON.stringify(boxes)); boxes = JSON.parse(redo.pop()); sel = Math.min(sel, boxes.length - 1); changed(false); }
  function changed() { dirty = true; setSaveState('dirty'); renderBoxes(); draw(); clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 500); }
  async function save(force = false) {
    if (!item || (!dirty && !force) || saving) return;
    const target = item, payload = boxes.map(b => [...b]);
    saving = true; setSaveState('saving');
    try {
      const res = await fovea.api.put(`/api/datasets/${ds.id}/images/${target.id}/labels`, { boxes: payload });
      if (target === item) { dirty = JSON.stringify(payload) !== JSON.stringify(boxes); item = { ...item, ...res.item, boxes: undefined }; if (!dirty) setSaveState('saved'); }
      cursor.update({ ...res.item, boxes: res.item.boxes });
      fovea.bus.emit('labels:changed', { item: res.item });
      if (res.item.classes.some(c => c >= names.length)) { const d = await refresh(); names = [...(d.classes || [])]; renderClasses(); }
      renderInfo();
      if (dirty) { clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 300); }
    } catch (e) { setSaveState('error', 'Save failed — retrying'); ui.toast('Save failed: ' + e.message, { type: 'error' }); clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 3000); }
    finally { saving = false; }
  }
  async function flush() { clearTimeout(saveTimer); if (dirty) await save(true); while (saving) await new Promise(r => setTimeout(r, 30)); }

  // ---------------------------------------------------------------- navigation
  async function goTo(i) {
    if (cursor.total != null) i = Math.max(0, Math.min(cursor.total - 1, i));
    if (i < 0) return;
    stopAutoIfEnd(i);
    await flush();
    const prevBoxes = boxes.map(b => [...b]);
    const it = await cursor.ensure(i);
    if (!it) return;
    idx = i; item = it; sel = -1; hover = -1; undo = []; redo = []; drag = null; dirty = false;
    fovea.router.replaceQuery({ ...filters, sort: sort !== 'id' ? sort : null, order: order !== 'asc' ? order : null, img: it.id });
    renderInfo(); setSaveState('saved');
    stageMsg.textContent = ''; stageMsg.classList.add('hidden');
    const [labels] = await Promise.all([fovea.api.get(`/api/datasets/${ds.id}/images/${it.id}/labels`).catch(() => ({ boxes: [] })), loadImage(it.id)]);
    if (item !== it) return;
    boxes = (labels.boxes || []).map(b => b.slice(0, 5));
    if (!boxes.length && opts.propagate && prevBoxes.length && !labels.exists) { boxes = prevBoxes; dirty = true; setSaveState('dirty'); clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 400); }
    if (!(customView && prevDims && img && prevDims[0] === img.naturalWidth && prevDims[1] === img.naturalHeight)) fit();
    prevDims = img ? [img.naturalWidth, img.naturalHeight] : null;
    renderBoxes(); updateHud(); draw();
    cursor.prefetch(idx, 4);
    const nxt = cursor.items[idx + 1]; if (nxt) { const pre = new Image(); pre.src = fovea.api.imgUrl(nxt.id); }
  }
  function loadImage(id) {
    return new Promise((resolve) => {
      const im = new Image(); im.decoding = 'async';
      im.onload = () => { img = im; resolve(); }; im.onerror = () => { img = null; stageMsg.textContent = 'Failed to load image'; stageMsg.classList.remove('hidden'); resolve(); };
      im.src = fovea.api.imgUrl(id);
    });
  }
  async function nextUnlabeled() {
    if (!item) return;
    try {
      const { split, cls, q, seq, ids, review, issue } = filters;
      const res = await fovea.api.get(`/api/datasets/${ds.id}/images/neighbor`, { id: item.id, dir: 'next', labeled: 'nobox', split, cls, q, seq, ids, review, issue });
      if (!res.item) { ui.toast('No unlabeled image after this one'); return; }
      const pos = await cursor.positionOf(res.item.id);
      if (pos >= 0) goTo(pos); else fovea.router.navigate(`/d/${ds.id}/annotate?img=${res.item.id}`);
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }
  async function copyFromPrevious() {
    if (idx <= 0) return; const prev = cursor.items[idx - 1]; if (!prev) return;
    try { const { item: full } = await fovea.api.get(`/api/datasets/${ds.id}/images/${prev.id}`); if (!full.boxes.length) { ui.toast('Previous image has no boxes'); return; } pushHistory(); boxes = boxes.concat(full.boxes.map(b => b.slice(0, 5))); changed(); ui.toast(`Copied ${full.boxes.length} boxes from previous`, { timeout: 1200 }); }
    catch (e) { ui.toast(e.message, { type: 'error' }); }
  }
  async function setReview(status) {
    if (!item) return;
    try { await fovea.api.put(`/api/datasets/${ds.id}/review`, { image_ids: [item.id], status }); item.review = status ? { status, note: '', updated_at: Date.now() / 1000 } : null; cursor.update({ id: item.id, review: item.review }); renderInfo(); fovea.bus.emit('review:changed', { ids: [item.id], status }); }
    catch (e) { ui.toast(e.message, { type: 'error' }); }
  }
  function startAuto() { autoTimer = setInterval(() => goTo(idx + 1), Number(speed.value)); playBtn.classList.add('active'); playBtn.innerHTML = ''; playBtn.appendChild(icon('pause', 13)); }
  function stopAuto() { clearInterval(autoTimer); autoTimer = null; playBtn.classList.remove('active'); playBtn.innerHTML = ''; playBtn.appendChild(icon('play', 13)); }
  function toggleAuto() { autoTimer ? stopAuto() : startAuto(); }
  function stopAutoIfEnd(i) { if (autoTimer && cursor.total != null && i >= cursor.total - 1) stopAuto(); }

  // ---------------------------------------------------------------- range edit
  async function applyRange() {
    if (!rangeStart || !item) return;
    const c = Number(rangeCls.value); const a = Math.min(rangeStart.idx, idx), b = Math.max(rangeStart.idx, idx);
    if (!await ui.confirm({ title: 'Apply class to range', message: `Set every box in images #${a + 1}–#${b + 1} (${b - a + 1} images) to class ${c} · ${className(names, c)}? This writes label files immediately.`, okLabel: 'Apply' })) return;
    await flush();
    try {
      const ids = []; for (let i = a; i <= b; i++) { const it = await cursor.ensure(i); if (it) ids.push(it.id); }
      const res = await fovea.api.post(`/api/datasets/${ds.id}/labels/bulk`, { op: 'set_class', cls: c, image_ids: ids });
      if (res.job) { await fovea.api.watchJob(res.job.id); }
      ui.toast(`Range applied: ${res.images ?? ids.length} images`, { type: 'ok' });
      rangeStart = null; rangeApply.disabled = true; rangeStatus.textContent = 'No range start';
      const cur = idx; idx = -1; await goTo(cur);
      fovea.bus.emit('dataset:updated');
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }

  // ---------------------------------------------------------------- AI auto-label
  async function renderAI() {
    aiPanel.innerHTML = '';
    let meta;
    try { meta = await fovea.api.get('/api/ai/models'); } catch (e) { aiPanel.appendChild(h('div', { class: 'small faint' }, 'AI unavailable')); return; }
    if (!meta.available) { aiPanel.appendChild(h('div', { class: 'small faint' }, 'Install ultralytics to enable auto-labeling.')); return; }
    if (!meta.models.length) { aiPanel.appendChild(h('div', { class: 'small faint' }, 'No models found. Drop YOLO .pt weights into the model folder (see Settings).')); return; }
    if (!ai.model) ai.model = (meta.models.find(m => m.name.includes('12x')) || meta.models[0]).name;
    const modelSel = h('select', { class: 'select select-sm', onChange: (e) => { ai.model = e.target.value; ai.modelClasses = null; ai.map = {}; renderMap(); } }, meta.models.map(m => h('option', { value: m.name, selected: m.name === ai.model }, `${m.name} (${m.size_mb} MB)`)));
    const confL = h('span', { class: 'xs mono' }, ai.conf.toFixed(2));
    const conf = h('input', { type: 'range', class: 'slider', min: 0.05, max: 0.95, step: 0.05, value: ai.conf, onInput: (e) => { ai.conf = Number(e.target.value); confL.textContent = ai.conf.toFixed(2); } });
    const clsIn = h('input', { class: 'input input-sm mono', placeholder: 'model class ids, e.g. 0 (empty = all)', value: ai.classes, onChange: (e) => { ai.classes = e.target.value; renderMap(); } });
    const mapBox = h('div', { class: 'col gap-4' });
    const modeSeg = ui.seg([{ value: 'fill', label: 'Fill empty', tip: 'Only images with no boxes' }, { value: 'append', label: 'Append' }, { value: 'replace', label: 'Replace' }], ai.mode, v => ai.mode = v);
    const progress = h('div');
    aiPanel.appendChild(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model'), modelSel));
    aiPanel.appendChild(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Confidence ', confL), conf));
    aiPanel.appendChild(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model classes'), clsIn, h('div', { class: 'row' }, h('button', { class: 'btn btn-sm', onClick: () => loadModelClasses(true) }, icon('list', 12), 'Show model classes'))));
    aiPanel.appendChild(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Map to dataset classes'), mapBox));
    aiPanel.appendChild(h('div', { class: 'field' }, h('span', { class: 'label' }, 'Mode'), modeSeg));
    aiPanel.appendChild(h('div', { class: 'row' }, h('button', { class: 'btn btn-sm grow', onClick: () => detectCurrent() }, icon('wand', 12), 'This image'), h('button', { class: 'btn btn-sm btn-primary grow', onClick: () => batchDialog() }, icon('sparkles', 12), 'Batch…')));
    aiPanel.appendChild(progress);
    ai.progressEl = progress;
    async function loadModelClasses(show) {
      if (!ai.modelClasses) { try { ai.modelClasses = (await fovea.api.get(`/api/ai/models/${ai.model}/classes`)).classes; } catch (e) { ui.toast(e.message, { type: 'error' }); return; } }
      if (show) ui.modal({ title: `${ai.model} classes`, body: h('div', { class: 'row wrap gap-4' }, Object.entries(ai.modelClasses).map(([k, v]) => ui.chip(`${k} ${v}`, { cls: 'outline', onClick: () => { const set = new Set(ai.classes.split(',').map(s => s.trim()).filter(Boolean)); set.add(k); ai.classes = [...set].join(','); clsIn.value = ai.classes; renderMap(); } }))), footer: (a) => h('button', { class: 'btn btn-primary', onClick: () => a.close() }, 'Done') });
      renderMap();
    }
    function chosenModelClasses() { return ai.classes.split(',').map(s => s.trim()).filter(s => s !== '').map(Number).filter(n => !isNaN(n)); }
    function renderMap() {
      mapBox.innerHTML = '';
      const chosen = chosenModelClasses();
      if (!chosen.length) { mapBox.appendChild(h('div', { class: 'xs faint' }, 'All model classes → same class id (set ids above to map explicitly).')); return; }
      chosen.forEach(mc => {
        const mname = ai.modelClasses ? ai.modelClasses[mc] : `class ${mc}`;
        if (ai.map[mc] == null) { const byName = names.findIndex(n => n.toLowerCase() === String(mname).toLowerCase()); ai.map[mc] = byName >= 0 ? byName : (mc < names.length ? mc : 0); }
        mapBox.appendChild(h('div', { class: 'map-row' }, h('span', { class: 'truncate mono xs', title: mname }, `${mc} ${mname}`), icon('arrowRight', 12), h('select', { class: 'select select-sm', onChange: (e) => ai.map[mc] = Number(e.target.value) }, names.map((n, i) => h('option', { value: i, selected: i === ai.map[mc] }, `${i} ${n}`)))));
      });
    }
    renderMap();
    function payloadBase() { const chosen = chosenModelClasses(); return { model: ai.model, conf: ai.conf, classes: chosen.length ? chosen : null, class_map: chosen.length ? ai.map : {}, keep_unmapped: !chosen.length }; }
    async function detectCurrent() {
      if (!item) return; progress.innerHTML = ''; progress.appendChild(h('div', { class: 'row small muted' }, ui.spinner(), 'Detecting… (first run loads the model)'));
      try {
        const res = await fovea.api.post('/api/ai/detect', { image_id: item.id, ...payloadBase() });
        const dets = res.detections.map(d => d.slice(0, 5));
        pushHistory();
        if (ai.mode === 'replace') boxes = dets; else if (ai.mode === 'fill') { if (!boxes.length) boxes = dets; else { ui.toast('Image already has boxes (fill mode) — nothing added'); undo.pop(); } } else boxes = boxes.concat(dets);
        changed(); progress.innerHTML = ''; ui.toast(`${dets.length} detections`, { type: 'ok', timeout: 1500 });
      } catch (e) { progress.innerHTML = ''; ui.toast(e.message, { type: 'error' }); }
    }
    function batchDialog() {
      let scope = 'next', n = 50;
      const nIn = h('input', { class: 'input input-sm', type: 'number', min: 1, max: 5000, value: n, style: 'width:90px', onChange: (e) => n = Number(e.target.value) });
      const m = ui.modal({ title: 'Batch auto-label', body: h('div', { class: 'col gap-12' },
        h('div', { class: 'callout small' }, icon('info', 14), h('span', `Runs ${ai.model} on the server and writes label files directly (mode: ${ai.mode}). Existing boxes are ${ai.mode === 'replace' ? 'replaced' : ai.mode === 'append' ? 'kept and extended' : 'kept; only empty images are filled'}.`)),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Scope'),
          h('label', { class: 'check' }, h('input', { type: 'radio', name: 'scope', checked: true, onChange: () => scope = 'next' }), 'Next ', nIn, ' images from here'),
          h('label', { class: 'check' }, h('input', { type: 'radio', name: 'scope', onChange: () => scope = 'all' }), `Every image in the current list (${fmt.num(cursor.total)})`)),
      ), footer: (a) => [h('button', { class: 'btn', onClick: () => a.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: async () => { a.close(); await runBatch(scope, n); } }, icon('sparkles', 13), 'Start')] });
    }
    async function runBatch(scope, n) {
      await flush();
      let body = { dataset_id: ds.id, mode: ai.mode, ...payloadBase() };
      if (scope === 'next') { const ids = []; for (let i = idx; i < idx + n; i++) { const it = await cursor.ensure(i); if (!it) break; ids.push(it.id); } body.image_ids = ids; }
      else body.filters = filters;
      try {
        const { job } = await fovea.api.post('/api/ai/autolabel', body);
        const bar = ui.progress(0); const txt = h('span', { class: 'xs muted' }, 'Queued…');
        progress.innerHTML = ''; progress.appendChild(h('div', { class: 'col gap-4' }, h('div', { class: 'row' }, txt, h('span', { class: 'spacer' }), h('button', { class: 'btn btn-ghost btn-sm', onClick: () => fovea.api.post(`/api/jobs/${job.id}/cancel`) }, 'Cancel')), bar));
        const snap = await fovea.api.watchJob(job.id, (s) => { txt.textContent = s.message; bar.firstChild.style.width = `${Math.round(s.progress * 100)}%`; });
        progress.innerHTML = ''; ui.toast(`Auto-label done: ${snap.result.images} images, ${snap.result.boxes} boxes`, { type: 'ok' });
        const cur = idx; idx = -1; cursor.reset(); await goTo(cur); fovea.bus.emit('dataset:updated');
      } catch (e) { progress.innerHTML = ''; ui.toast('Auto-label failed: ' + e.message, { type: 'error' }); }
    }
  }

  // ---------------------------------------------------------------- tools & keys
  function setTool(t) { tool = t; Object.entries(toolBtns).forEach(([k, b]) => b.classList.toggle('active', k === t)); renderPointOpts(); updateHud(); draw(); if (mouse.inside) canvas.style.cursor = cursorFor(mouse.x, mouse.y); }
  function applyClassBuf() { const n = parseInt(classBuf, 10); classBuf = ''; if (isNaN(n)) return; if (sel >= 0) { if (n !== boxes[sel][0]) { pushHistory(); boxes[sel][0] = n; changed(); } activeCls = n; renderClasses(); updateHud(); } else if (n < Math.max(1, names.length)) setActiveClass(n); }
  const onKey = (e) => {
    if (destroyed || ui.hasModal()) return;
    if (isTyping()) { if (e.key === 'Escape') e.target.blur(); return; }
    const k = e.key, mod = e.metaKey || e.ctrlKey;
    if (k === ' ' && !e.repeat) { spaceDown = true; canvas.style.cursor = 'grab'; e.preventDefault(); return; }
    if (mod && k.toLowerCase() === 'z') { e.preventDefault(); e.shiftKey ? doRedo() : doUndo(); return; }
    if (mod && k.toLowerCase() === 's') { e.preventDefault(); flush(); return; }
    if (mod) return;
    if (k === 'ArrowRight' && !e.altKey) { e.preventDefault(); stopAuto(); goTo(idx + 1); return; }
    if (k === 'ArrowLeft' && !e.altKey) { e.preventDefault(); stopAuto(); goTo(idx - 1); return; }
    if (e.altKey && k.startsWith('Arrow') && sel >= 0 && img) { e.preventDefault(); pushHistory(); const st = e.shiftKey ? 10 : 1; const b = boxes[sel]; if (k === 'ArrowLeft') b[1] -= st / img.naturalWidth; if (k === 'ArrowRight') b[1] += st / img.naturalWidth; if (k === 'ArrowUp') b[2] -= st / img.naturalHeight; if (k === 'ArrowDown') b[2] += st / img.naturalHeight; clampBox(b); changed(); return; }
    if (k >= '0' && k <= '9') { e.preventDefault(); classBuf += k; clearTimeout(classBufTimer); if (names.length <= 10) applyClassBuf(); else classBufTimer = setTimeout(applyClassBuf, 550); return; }
    switch (k) {
      case 'v': case 'V': setTool('select'); break;
      case 'b': case 'B': setTool('box'); break;
      case 'p': case 'P': setTool('point'); break;
      case 'h': case 'H': setTool('pan'); break;
      case 'f': setTool(tool); fit(); draw(); break;
      case 'F': setReview(item && item.review && item.review.status === 'flagged' ? null : 'flagged'); break;
      case 'A': setReview(item && item.review && item.review.status === 'approved' ? null : 'approved'); break;
      case 'X': setReview(item && item.review && item.review.status === 'excluded' ? null : 'excluded'); break;
      case 'l': case 'L': labelsSw.click(); break;
      case 'n': case 'N': nextUnlabeled(); break;
      case 'c': case 'C': copyFromPrevious(); break;
      case 'Delete': case 'Backspace': if (sel >= 0) { e.preventDefault(); pushHistory(); boxes.splice(sel, 1); sel = -1; changed(); } break;
      case 'Escape': if (drag) { drag = null; draw(); } else if (sel >= 0) { sel = -1; renderBoxes(); draw(); } break;
      case 'Tab': e.preventDefault(); if (boxes.length) { sel = e.shiftKey ? (sel - 1 + boxes.length) % boxes.length : (sel + 1) % boxes.length; renderBoxes(); draw(); } break;
      case 'Home': e.preventDefault(); goTo(0); break;
      case 'End': e.preventDefault(); goTo((cursor.total || 1) - 1); break;
      case '[': setActiveClass(Math.max(0, activeCls - 1)); break;
      case ']': setActiveClass(Math.min(Math.max(0, names.length - 1), activeCls + 1)); break;
      default: return;
    }
  };
  const onKeyUp = (e) => { if (e.key === ' ') { spaceDown = false; if (mouse.inside) canvas.style.cursor = cursorFor(mouse.x, mouse.y); } };
  document.addEventListener('keydown', onKey); document.addEventListener('keyup', onKeyUp);
  const onUnload = () => { if (dirty && item) { navigator.sendBeacon && fetch(`/api/datasets/${ds.id}/images/${item.id}/labels`, { method: 'PUT', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ boxes }), keepalive: true }); } };
  window.addEventListener('beforeunload', onUnload);

  // ---------------------------------------------------------------- boot
  renderClasses(); updateHud(); renderAI();
  const startId = q0.get('img') ? Number(q0.get('img')) : null;
  try {
    await cursor.loadPage(0);
    if (!cursor.total) { stageMsg.textContent = 'No images match the current filter'; renderInfo(); }
    else { let start = 0; if (startId) { const pos = await cursor.positionOf(startId); if (pos >= 0) start = pos; } await goTo(start); }
  } catch (e) { stageMsg.textContent = e.message; }
  resizeCanvas();

  return () => {
    destroyed = true; clearTimeout(saveTimer); stopAuto();
    if (dirty) save(true);
    document.removeEventListener('keydown', onKey); document.removeEventListener('keyup', onKeyUp);
    window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); window.removeEventListener('beforeunload', onUnload);
    ro.disconnect();
  };
}
