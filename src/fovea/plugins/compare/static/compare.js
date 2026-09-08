// Compare plugin UI: run evaluations, browse per-class / per-image results, side-by-side viewer.
export default function install(fovea, manifest) {
  const API = manifest.api;
  fovea.registerTab({ id: 'compare', label: 'Compare', icon: 'gitCompare', order: 45, render: (el, c) => render(el, c, fovea, API) });
  fovea.registerCommand({ id: 'compare.open', label: 'Compare predictions vs GT', icon: 'gitCompare', when: (s) => !!s.currentDataset, run: (f) => f.router.navigate(`/d/${f.state.get().currentDataset.id}/compare`) });
}

async function render(el, { ctx, dataset, fovea }, F, API) {
  const { h, icon, ui, fmt, cls } = F;
  const { classColor, className } = F.colors;
  const ds = dataset; const names = ds.classes || [];
  const page = h('div', { class: 'page' }); const inner = h('div', { class: 'page-inner wide' }); page.appendChild(inner); el.appendChild(page);
  const setup = { pred_a: '', pred_b: '', iou: 0.5, conf: 0.25, split: '' };
  const results = h('div');
  let current = null; // loaded eval
  let viewer = null;

  // ------------------------------------------------------------ setup card
  const aIn = h('input', { class: 'input mono', placeholder: '/path/to/predictions/labels (YOLO txt with confidence)', onChange: e => setup.pred_a = e.target.value.trim() });
  const bIn = h('input', { class: 'input mono', placeholder: 'optional second model to compare', onChange: e => setup.pred_b = e.target.value.trim() });
  const iouL = h('span', { class: 'mono small' }, '0.50'), confL = h('span', { class: 'mono small' }, '0.25');
  const iouIn = h('input', { type: 'range', class: 'slider', min: 0.1, max: 0.95, step: 0.05, value: 0.5, onInput: e => { setup.iou = Number(e.target.value); iouL.textContent = setup.iou.toFixed(2); } });
  const confIn = h('input', { type: 'range', class: 'slider', min: 0, max: 0.95, step: 0.05, value: 0.25, onInput: e => { setup.conf = Number(e.target.value); confL.textContent = setup.conf.toFixed(2); } });
  const splitSel = h('select', { class: 'select', style: 'width:auto', onChange: e => setup.split = e.target.value }, [['', 'All splits'], ...(ds.splits || []).filter(Boolean).map(s => [s, s])].map(([v, l]) => h('option', { value: v }, l)));
  const runBtn = h('button', { class: 'btn btn-primary', onClick: () => runEval() }, icon('play', 14), 'Run evaluation');
  const progress = h('div');
  const browse = (input, key) => h('button', { class: 'btn', onClick: async () => { const p = await F.pickPath({ title: 'Choose prediction label folder', files: false, selectFiles: false, start: input.value.trim() || ds.root_host }); if (p) { input.value = p; setup[key] = p; } } }, icon('folderOpen', 14), 'Browse');
  const discoverBtn = h('button', { class: 'btn', onClick: () => discover() }, icon('scan', 14), 'Discover…');
  inner.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Evaluate predictions against ground truth'), h('span', { class: 'sub' }, 'greedy IoU matching · per-class P/R/F1 · AP@IoU')),
    h('div', { class: 'card-body col gap-12' },
      h('div', { class: 'callout small' }, icon('info', 14), h('span', 'Prediction folders hold YOLO txt files named like the images (optionally inside split subfolders) with a 6th confidence column. Ground truth comes from the dataset index.')),
      h('div', { class: 'fields cols-2' },
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model A predictions'), h('div', { class: 'row' }, aIn, browse(aIn, 'pred_a'), discoverBtn)),
        h('div', { class: 'field' }, h('span', { class: 'label' }, 'Model B predictions (optional)'), h('div', { class: 'row' }, bIn, browse(bIn, 'pred_b')))),
      h('div', { class: 'row wrap gap-16' },
        h('div', { class: 'row', style: 'width:220px' }, h('span', { class: 'small muted', style: 'width:60px' }, 'IoU ≥'), iouIn, iouL),
        h('div', { class: 'row', style: 'width:220px' }, h('span', { class: 'small muted', style: 'width:60px' }, 'conf ≥'), confIn, confL),
        splitSel, h('span', { class: 'spacer' }), runBtn),
      progress)));
  const savedList = h('div', { class: 'row wrap gap-4' });
  inner.appendChild(h('div', { class: 'row mb-12' }, h('span', { class: 'small strong muted' }, 'Saved evaluations'), savedList));
  inner.appendChild(results);

  async function discover() {
    const base = await F.pickPath({ title: 'Folder to scan for prediction outputs', files: false, selectFiles: false, start: ds.root_host });
    if (!base) return;
    const m = ui.modal({ title: 'Prediction folders found', size: 'lg', body: h('div', { class: 'row', style: 'padding:14px;justify-content:center' }, ui.spinner()) });
    try {
      const { dirs } = await F.api.get(`${API}/discover`, { path: base });
      m.body.innerHTML = '';
      if (!dirs.length) { m.body.appendChild(h('div', { class: 'empty' }, h('p', 'No folders with YOLO txt files found.'))); return; }
      m.body.appendChild(h('table', { class: 'table' }, h('thead', h('tr', h('th', 'Folder'), h('th', 'Files'), h('th', 'Conf'), h('th', ''))), h('tbody', dirs.map(d => h('tr',
        h('td', { class: 'mono small' }, d.path_host), h('td', { class: 'num' }, fmt.num(d.files)), h('td', d.has_conf ? ui.chip('yes', { type: 'ok' }) : ui.chip('no', { type: 'warn' })),
        h('td', { class: 'row' }, h('button', { class: 'btn btn-sm', onClick: () => { aIn.value = d.path_host; setup.pred_a = d.path_host; m.close(); } }, 'Use as A'), h('button', { class: 'btn btn-sm', onClick: () => { bIn.value = d.path_host; setup.pred_b = d.path_host; m.close(); } }, 'Use as B')))))));
    } catch (e) { m.body.innerHTML = ''; m.body.appendChild(h('div', { class: 'callout danger' }, e.message)); }
  }
  async function runEval() {
    setup.pred_a = aIn.value.trim(); setup.pred_b = bIn.value.trim();
    if (!setup.pred_a) { ui.toast('Model A prediction folder is required', { type: 'error' }); return; }
    runBtn.disabled = true;
    try {
      const { job, eval_id } = await F.api.post(`${API}/evaluate`, { dataset_id: ds.id, pred_a: setup.pred_a, pred_b: setup.pred_b || null, iou: setup.iou, conf: setup.conf, filters: setup.split ? { split: setup.split } : {} });
      const bar = ui.progress(0); const txt = h('span', { class: 'small muted' }, 'Queued…');
      progress.innerHTML = ''; progress.appendChild(h('div', { class: 'col gap-4' }, txt, bar));
      await F.api.watchJob(job.id, s => { txt.textContent = s.message; bar.firstChild.style.width = `${Math.round(s.progress * 100)}%`; });
      progress.innerHTML = ''; ui.toast('Evaluation finished', { type: 'ok' });
      await loadSaved(); await loadEval(eval_id);
    } catch (e) { progress.innerHTML = ''; ui.toast(e.message, { type: 'error' }); }
    finally { runBtn.disabled = false; }
  }
  async function loadSaved() {
    const { evals } = await F.api.get(`${API}/evals`, { dataset_id: ds.id });
    savedList.innerHTML = '';
    if (!evals.length) savedList.appendChild(h('span', { class: 'small faint' }, 'none yet'));
    evals.forEach(ev => savedList.appendChild(ui.chip(`${ev.name} · mAP ${(ev.summary.a.overall.map50 * 100).toFixed(1)}`, { cls: current && current.id === ev.id ? 'accent' : 'outline', icon: 'gitCompare', onClick: () => loadEval(ev.id),
      onRemove: async () => { if (await ui.confirm({ title: 'Delete evaluation?', message: ev.name, okLabel: 'Delete', danger: true })) { await F.api.del(`${API}/evals/${ev.id}?dataset_id=${ds.id}`); if (current && current.id === ev.id) { current = null; results.innerHTML = ''; } loadSaved(); } } })));
  }

  // ------------------------------------------------------------ results
  let errorsOnly = true;
  async function loadEval(id) {
    results.innerHTML = ''; results.appendChild(h('div', { class: 'row', style: 'padding:20px;justify-content:center' }, ui.spinner()));
    try { current = (await F.api.get(`${API}/evals/${id}`, { dataset_id: ds.id, errors_only: errorsOnly ? 1 : 0, limit: 500 })).eval; }
    catch (e) { results.innerHTML = ''; results.appendChild(h('div', { class: 'callout danger' }, e.message)); return; }
    F.router.replaceQuery({ eval: id });
    loadSaved();
    drawResults();
  }
  function pct(v) { return (v * 100).toFixed(1) + '%'; }
  function drawResults() {
    const ev = current; const hasB = !!ev.summary.b;
    results.innerHTML = '';
    results.appendChild(h('div', { class: 'row mb-12' }, h('h2', { style: 'margin:0;font-size:15px' }, ev.name), h('span', { class: 'small faint' }, `${fmt.num(ev.images)} images · pred files A ${fmt.num(ev.pred_files.a)}${hasB ? ` · B ${fmt.num(ev.pred_files.b)}` : ''}`), h('span', { class: 'spacer' }),
      h('button', { class: 'btn btn-sm', onClick: () => window.open(`${API}/evals/${ev.id}?dataset_id=${ds.id}&limit=100000`, '_blank') }, icon('download', 12), 'JSON')));
    const tiles = (side, label) => { const o = ev.summary[side].overall; return h('div', { class: 'card', style: 'flex:1' }, h('div', { class: 'card-header' }, h('h3', label), h('span', { class: 'sub mono' }, side === 'a' ? ev.pred_a_host : ev.pred_b_host)), h('div', { class: 'card-body' }, h('div', { class: 'stat-grid' },
      ui.statTile({ label: 'mAP@' + ev.iou, value: pct(o.map50) }), ui.statTile({ label: 'Precision', value: pct(o.precision), sub: `${fmt.num(o.tp)} TP · ${fmt.num(o.fp)} FP` }), ui.statTile({ label: 'Recall', value: pct(o.recall), sub: `${fmt.num(o.fn)} FN of ${fmt.num(o.gt)} GT` }), ui.statTile({ label: 'F1', value: pct(o.f1) })))); };
    results.appendChild(h('div', { class: 'row gap-12 mb-16', style: 'align-items:stretch' }, tiles('a', 'Model A'), hasB ? tiles('b', 'Model B') : null));
    // per class table
    const clsRows = new Map();
    ev.summary.a.classes.forEach(c => clsRows.set(c.cls, { a: c }));
    if (hasB) ev.summary.b.classes.forEach(c => clsRows.set(c.cls, { ...(clsRows.get(c.cls) || {}), b: c }));
    const cell = (c, k, isPct = true) => c ? h('td', { class: 'num' }, isPct ? pct(c[k]) : fmt.num(c[k])) : h('td', { class: 'num faint' }, '–');
    const delta = (a, b, k) => { if (!a || !b) return h('td'); const d = (b[k] - a[k]) * 100; return h('td', { class: 'num', style: `color:${d > 0 ? 'var(--ok)' : d < 0 ? 'var(--danger)' : 'var(--text-3)'}` }, (d > 0 ? '+' : '') + d.toFixed(1)); };
    results.appendChild(h('div', { class: 'card mb-16' }, h('div', { class: 'card-header' }, h('h3', 'Per class')), h('div', { class: 'card-body tight', style: 'overflow-x:auto' }, h('table', { class: 'table' },
      h('thead', h('tr', h('th', 'Class'), h('th', { class: 'right' }, 'GT'), h('th', { class: 'right' }, 'A · AP'), h('th', { class: 'right' }, 'A · P'), h('th', { class: 'right' }, 'A · R'), h('th', { class: 'right' }, 'A · FP'), h('th', { class: 'right' }, 'A · FN'),
        hasB ? [h('th', { class: 'right' }, 'B · AP'), h('th', { class: 'right' }, 'B · P'), h('th', { class: 'right' }, 'B · R'), h('th', { class: 'right' }, 'B · FP'), h('th', { class: 'right' }, 'B · FN'), h('th', { class: 'right' }, 'Δ AP')] : null)),
      h('tbody', [...clsRows.keys()].sort((x, y) => x - y).map(c => { const r = clsRows.get(c); const any = r.a || r.b; return h('tr', h('td', h('span', { class: 'swatch', style: `background:${classColor(c)};margin-right:6px` }), `${c} · ${any.name}`), h('td', { class: 'num' }, fmt.num(any.gt)),
        cell(r.a, 'ap'), cell(r.a, 'precision'), cell(r.a, 'recall'), cell(r.a, 'fp', false), cell(r.a, 'fn', false),
        hasB ? [cell(r.b, 'ap'), cell(r.b, 'precision'), cell(r.b, 'recall'), cell(r.b, 'fp', false), cell(r.b, 'fn', false), delta(r.a, r.b, 'ap')] : null); }))))));
    // per image table
    const tbl = h('table', { class: 'table' });
    const rowsBody = h('tbody');
    const drawRows = () => { rowsBody.innerHTML = ''; ev.rows.forEach((r, i) => rowsBody.appendChild(h('tr', { class: 'clickable', onClick: () => openViewer(i) },
      h('td', { class: 'mono small truncate', style: 'max-width:420px' }, r.rel_path), h('td', r.split || '—'), h('td', { class: 'num' }, r.gt),
      errCell(r.a), hasB ? errCell(r.b) : null))); };
    const errCell = (s) => s ? h('td', { class: 'num' }, h('span', { style: 'color:var(--ok)' }, s.tp), ' / ', h('span', { style: `color:${s.fp ? 'var(--danger)' : 'inherit'}` }, s.fp), ' / ', h('span', { style: `color:${s.fn ? 'var(--warn)' : 'inherit'}` }, s.fn)) : h('td', '–');
    tbl.appendChild(h('thead', h('tr', h('th', 'Image'), h('th', 'Split'), h('th', { class: 'right' }, 'GT'), h('th', { class: 'right' }, 'A · TP / FP / FN'), hasB ? h('th', { class: 'right' }, 'B · TP / FP / FN') : null)));
    tbl.appendChild(rowsBody); drawRows();
    results.appendChild(h('div', { class: 'card' }, h('div', { class: 'card-header' }, h('h3', 'Per image'), h('span', { class: 'sub' }, `${fmt.num(ev.rows_total)} rows, worst first`), h('span', { class: 'spacer' }),
      h('label', { class: 'check small' }, h('input', { type: 'checkbox', checked: errorsOnly, onChange: async (e) => { errorsOnly = e.target.checked; await loadEval(ev.id); } }), 'errors only')),
      h('div', { class: 'card-body tight', style: 'overflow:auto;max-height:60vh' }, ev.rows.length ? tbl : h('div', { class: 'empty' }, h('p', 'No rows')))));
  }

  // ------------------------------------------------------------ side-by-side viewer
  function openViewer(rowIdx) {
    if (viewer) viewer.close();
    const ev = current; const hasB = !!ev.summary.b;
    let i = rowIdx; const layers = { gt: true, pred: true, labels: true };
    const rootEl = h('div', { class: 'inspect', style: `grid-template-columns:1fr;grid-template-rows:44px 1fr` });
    const top = h('div', { class: 'inspect-top' });
    const stageWrap = h('div', { style: `display:grid;grid-template-columns:${hasB ? '1fr 1fr' : '1fr'};gap:2px;min-height:0;background:#000` });
    rootEl.appendChild(top); rootEl.appendChild(stageWrap); document.body.appendChild(rootEl);
    const pos = h('span', { class: 'mono small' }); const name = h('span', { class: 'name grow' });
    const tog = (key, label) => { const b = h('button', { class: cls('btn btn-sm', layers[key] && 'active'), onClick: () => { layers[key] = !layers[key]; b.classList.toggle('active', layers[key]); draw(); } }, label); return b; };
    top.appendChild(h('button', { class: 'btn btn-sm btn-icon', onClick: () => close() }, icon('x', 14))); top.appendChild(pos); top.appendChild(name);
    top.appendChild(tog('gt', 'GT')); top.appendChild(tog('pred', 'Predictions')); top.appendChild(tog('labels', 'Labels'));
    top.appendChild(h('button', { class: 'btn btn-sm', 'data-tip': 'Copy side-by-side PNG to clipboard (⌘C)', onClick: () => copyPng() }, icon('copy', 13), 'Copy PNG'));
    top.appendChild(h('button', { class: 'btn btn-sm', onClick: () => F.router.navigate(`/d/${ds.id}/annotate?img=${ev.rows[i].id}`) }, icon('pen', 13), 'Edit GT'));
    let detail = null, imgEl = null;
    async function draw() {
      const r = ev.rows[i]; if (!r) return;
      pos.textContent = `${i + 1} / ${ev.rows.length}`; name.textContent = r.rel_path;
      if (!detail || detail.id !== r.id) { const res = await F.api.get(`${API}/evals/${ev.id}/image/${r.id}`, { dataset_id: ds.id }); detail = { id: r.id, ...res }; }
      stageWrap.innerHTML = '';
      for (const side of hasB ? ['a', 'b'] : ['a']) {
        const d = detail.detail[side]; const s = r[side] || { tp: 0, fp: 0, fn: 0 };
        const panel = h('div', { style: 'display:flex;flex-direction:column;min-height:0' });
        panel.appendChild(h('div', { class: 'row', style: `height:30px;padding:0 10px;font-weight:600;background:${side === 'a' ? '#1e3a8a' : '#7c2d12'};color:#fff;font-size:12px` }, side === 'a' ? 'Model A' : 'Model B', h('span', { class: 'spacer' }), h('span', { class: 'mono xs' }, `TP ${s.tp} · FP ${s.fp} · FN ${s.fn}`)));
        const stage = h('div', { class: 'inspect-stage', style: 'flex:1' });
        const pic = h('div', { class: 'pic' });
        const img = h('img', { src: F.api.imgUrl(r.id), alt: '', style: `max-width:calc(${hasB ? '50vw' : '100vw'} - 24px);max-height:calc(100vh - 44px - 30px - 16px)` });
        pic.appendChild(img);
        const boxes = [], styles = [];
        if (layers.gt) detail.detail.gt.forEach((g, gi) => { boxes.push(g); styles.push({ color: d && d.gt_matched[gi] === -1 ? '#f59e0b' : '#22c55e', label: `GT ${className(names, g[0])}${d && d.gt_matched[gi] === -1 ? ' · FN' : ''}`, fill: 'transparent' }); });
        if (layers.pred && d) d.preds.forEach(p => { boxes.push(p.box); styles.push({ color: p.tp ? '#3b82f6' : '#ef4444', dashed: true, label: `${p.tp ? 'TP' : 'FP'} ${className(names, p.box[0])} ${p.conf.toFixed(2)}`, fill: 'transparent' }); });
        pic.appendChild(F.boxLayer(boxes, { names, labels: layers.labels, stroke: 2, styleFor: (b, k) => styles[k] }));
        stage.appendChild(pic); panel.appendChild(stage); stageWrap.appendChild(panel);
        panel._img = img; panel._boxes = boxes; panel._styles = styles; panel._side = side; panel._s = s;
      }
    }
    async function copyPng() {
      const panels = [...stageWrap.children]; const first = panels[0]._img; if (!first.naturalWidth) return;
      const W = first.naturalWidth, H = first.naturalHeight, barH = Math.round(H * 0.06);
      const c = document.createElement('canvas'); c.width = W * panels.length; c.height = H + barH; const g = c.getContext('2d');
      panels.forEach((p, k) => { const x = k * W; g.fillStyle = p._side === 'a' ? '#1e3a8a' : '#7c2d12'; g.fillRect(x, 0, W, barH); g.fillStyle = '#fff'; g.font = `bold ${Math.round(barH * 0.5)}px Inter, sans-serif`; g.textBaseline = 'middle'; g.fillText(`${p._side === 'a' ? 'Model A' : 'Model B'}   TP ${p._s.tp} · FP ${p._s.fp} · FN ${p._s.fn}`, x + 12, barH / 2);
        g.drawImage(p._img, x, barH, W, H);
        p._boxes.forEach((b, j) => { const st = p._styles[j]; const bw = b[3] * W, bh = b[4] * H, bx = x + (b[1] - b[3] / 2) * W, by = barH + (b[2] - b[4] / 2) * H; g.lineWidth = Math.max(2, W / 500); g.strokeStyle = st.color; g.setLineDash(st.dashed ? [10, 6] : []); g.strokeRect(bx, by, bw, bh); g.setLineDash([]);
          if (layers.labels) { g.font = `${Math.round(H * 0.022)}px Inter, sans-serif`; const tw = g.measureText(st.label).width + 8; g.fillStyle = st.color; g.fillRect(bx, by - H * 0.028, tw, H * 0.028); g.fillStyle = '#fff'; g.textBaseline = 'top'; g.fillText(st.label, bx + 4, by - H * 0.026); } }); });
      c.toBlob(async (blob) => { try { await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]); ui.toast('Copied side-by-side PNG', { type: 'ok' }); } catch (e) { ui.toast('Clipboard failed: ' + e.message, { type: 'error' }); } }, 'image/png');
    }
    const key = (e) => { if (ui.hasModal() || F.isTyping()) return; if (e.key === 'Escape') close(); else if (e.key === 'ArrowRight') { i = Math.min(ev.rows.length - 1, i + 1); draw(); } else if (e.key === 'ArrowLeft') { i = Math.max(0, i - 1); draw(); } else if ((e.metaKey || e.ctrlKey) && e.key === 'c') { e.preventDefault(); copyPng(); } };
    document.addEventListener('keydown', key, true);
    const close = () => { document.removeEventListener('keydown', key, true); rootEl.remove(); viewer = null; };
    viewer = { close }; draw();
  }

  await loadSaved();
  if (ctx.query.get('eval')) loadEval(ctx.query.get('eval'));
  return () => { if (viewer) viewer.close(); };
}
