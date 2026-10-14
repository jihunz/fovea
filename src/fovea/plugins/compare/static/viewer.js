// Side-by-side viewer for one evaluation: ground truth and each model's predictions over the same image.
// Boxes are recomputed live from the current labels and prediction files; the header says when that differs
// from the stored evaluation. Image changes hold the current frame until the next one is decoded.

const COL = {
  gtTp: '#22c55e', gtFn: '#f59e0b', gtSwap: '#d946ef',
  tp: '#3b82f6', fp: '#ef4444', swap: '#d946ef', below: '#9ca3af', out: '#9ca3af',
};
export const SIDE_COLORS = { a: '#2a78d6', b: '#eb6834' };
export const VIEWER_KEYS = [
  ['← →', 'Previous / next image'], ['[ ]', 'Confidence threshold −/+ 0.05'], ['{ }', 'Confidence threshold −/+ 0.01'],
  ['E', 'Errors only'], ['L', 'Labels on / off'], ['B', 'Show predictions below the threshold'],
  ['F', 'Flag / unflag this image'], ['Enter', 'Edit ground truth in Annotate'], ['⌘ C', 'Copy side-by-side PNG'], ['Esc', 'Close'],
];

/** Middle-ellipsis for long paths so the distinguishing frame number stays visible. */
export function midEllipsis(text, max = 64) {
  if (!text || text.length <= max) return text || '';
  const keep = max - 1, head = Math.ceil(keep * 0.45), tail = keep - head;
  return text.slice(0, head) + '…' + text.slice(text.length - tail);
}

export function openViewer({ F, API, ds, names, ev, list, startIndex, k: k0, classes, onK, onFlagChange }) {
  const { h, icon, ui, keys } = F;
  const className = (c) => F.colors.className(names, c);
  const textOn = F.colors.textOn || ((hexc) => '#fff');
  const sides = ['a', 'b'].filter(s => ev.sources && ev.sources[s]);
  const layers = { gt: true, pred: true, labels: true, errors: false, below: false };
  let i = startIndex, k = k0, seq = 0, detail = null, row = null, closed = false;

  const opener = document.activeElement;           // focus goes back there on close (the table row, usually)
  const rootEl = h('div', { class: 'inspect cmp-viewer', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Compare viewer', tabindex: -1 });
  const top = h('div', { class: 'inspect-top' });
  const stageWrap = h('div', { class: 'cmp-v-panels', style: `grid-template-columns:repeat(${sides.length}, minmax(0,1fr))` });
  const bottom = h('div', { class: 'cmp-v-bottom' });
  rootEl.append(top, stageWrap, bottom);
  document.body.appendChild(rootEl);

  const pos = h('span', { class: 'mono small cmp-v-pos' });
  const nameEl = h('span', { class: 'name grow' });
  const tog = (key, label, tip) => { const b = h('button', { class: `btn btn-sm ${layers[key] ? 'active' : ''}`, 'aria-pressed': String(layers[key]), 'data-tip': tip, onClick: () => toggle(key) }, label); b.dataset.layer = key; return b; };
  const toggles = [tog('gt', 'GT', 'Ground-truth boxes'), tog('pred', 'Predictions', 'Model boxes'), tog('labels', 'Labels', 'Box labels (L)'), tog('errors', 'Errors only', 'Only misses, false positives and swaps (E)'), tog('below', 'Below threshold', 'Also draw predictions under the threshold, faint (B)')];
  const thrVal = h('span', { class: 'mono small cmp-v-thr-val' });
  const thrIn = h('input', { type: 'range', class: 'slider', min: 0, max: 1, step: 0.01, value: k / 100, 'aria-label': 'Confidence threshold', style: 'width:110px', onInput: (e) => setK(Math.round(Number(e.target.value) * 100)) });
  const flagBtn = h('button', { class: 'btn btn-sm', onClick: () => toggleFlag(), disabled: !!ev.legacy, 'data-tip': ev.legacy ? 'Re-run this evaluation to flag images from it' : 'Flag this image for review with a note from this evaluation (F)' }, icon('flag', 13), h('span', 'Flag'));
  top.append(
    h('button', { class: 'btn btn-sm btn-icon', onClick: () => close(), 'aria-label': 'Close viewer', 'data-tip': 'Close (Esc)' }, icon('x', 14)),
    pos, nameEl, ...toggles,
    h('span', { class: 'cmp-v-thr' }, h('span', { class: 'small' }, 'conf ≥'), thrIn, thrVal),
    flagBtn,
    h('button', { class: 'btn btn-sm', onClick: () => editGt(), 'data-tip': 'Open this image in Annotate (Enter)' }, icon('pen', 13), 'Edit GT'),
    h('button', { class: 'btn btn-sm', onClick: () => copyPng(), 'data-tip': 'Copy side-by-side PNG (⌘C)' }, icon('copy', 13), 'Copy PNG'),
  );
  const legendItems = [['GT found', COL.gtTp, false], ['GT missed (FN)', COL.gtFn, false], ['TP', COL.tp, true], ['FP', COL.fp, true], ['class swap', COL.swap, true]];
  if (ev.classes_view && ev.classes_view.only) legendItems.push(['not scored', COL.out, false]);
  const statusEl = h('span', { class: 'cmp-v-status' });
  bottom.append(h('span', { class: 'cmp-v-legend' }, legendItems.map(([t, c, dashed]) => h('span', { class: 'cmp-v-key' }, h('i', { style: `border:2px ${dashed ? 'dashed' : 'solid'} ${c}` }), t))),
    h('span', { class: 'spacer' }), statusEl, h('span', { class: 'cmp-v-hint' }, '← → images · [ ] threshold · E errors · F flag · ? keys'));

  function toggle(key) {
    layers[key] = !layers[key];
    toggles.forEach(b => { if (b.dataset.layer === key) { b.classList.toggle('active', layers[key]); b.setAttribute('aria-pressed', String(layers[key])); } });
    paint();
  }
  function setK(nk) {
    nk = Math.max(0, Math.min(100, nk));
    if (nk === k) return;
    k = nk; thrIn.value = k / 100; thrVal.textContent = (k / 100).toFixed(2);
    onK && onK(k);
    load(true);
  }
  thrVal.textContent = (k / 100).toFixed(2);

  const frames = F.media.createFrameCache({ max: 5 });
  let im = null, panelsBuilt = null;

  async function load(sameImage = false) {
    const my = ++seq;
    let r;
    try { r = await list.get(i); } catch (e) { if (my === seq && !closed) statusEl.textContent = e.message; return; }
    if (my !== seq || closed) return;
    if (!r) { i = Math.max(0, Math.min(i, list.total() - 1)); return; }
    pos.textContent = `${(i + 1).toLocaleString('en-US')} / ${list.total().toLocaleString('en-US')}`;
    let det, pic;
    try {
      [det, pic] = await Promise.all([
        F.api.get(`${API}/evals/${ev.id}/image/${r.id}`, { dataset_id: ds.id, conf: k / 100, classes: classes || undefined }),
        sameImage && im ? Promise.resolve(im) : frames.get(F.api.imgUrl(r.id), { priority: 'high' }).catch(() => null),
      ]);
    } catch (e) { if (my === seq && !closed) statusEl.textContent = e.message; return; }
    if (my !== seq || closed) return;
    row = r; detail = det; im = pic;
    paint();
    const next = [i + 1, i - 1].map(j => list.peek(j)).filter(Boolean).map(x => F.api.imgUrl(x.id));
    frames.prefetch(next);
    if (i + 3 >= list.loaded()) list.get(i + 3).catch(() => {});
  }

  function boxesFor(side) {
    const d = detail.detail, sd = d.sides[side];
    const out = [];
    // labelled boxes of classes this evaluation leaves out (e.g. cars under a person audit): context, never counted
    if (layers.gt && !layers.errors) (d.gt_out || []).forEach(g => out.push({ box: g, color: COL.out, dashed: false, faint: true, label: `${className(g[0])} · not scored` }));
    if (layers.gt) d.gt.forEach((g, gi) => {
      const st = sd ? sd.gt_state[gi] : 'tp';
      if (layers.errors && st === 'tp') return;
      const color = st === 'tp' ? COL.gtTp : st === 'swap' ? COL.gtSwap : COL.gtFn;
      out.push({ box: g, color, dashed: false, label: st === 'fn' ? `FN ${className(g[0])}` : st === 'swap' ? `GT ${className(g[0])} · swapped` : `GT ${className(g[0])}` });
    });
    if (layers.pred && sd) sd.preds.forEach(p => {
      const cf = p.conf != null ? ' ' + p.conf.toFixed(2) : '';
      if (!p.active) { if (layers.below && !layers.errors) out.push({ box: p.box, color: COL.below, dashed: true, faint: true, label: `${className(p.box[0])}${cf} · below` }); return; }
      if (layers.errors && p.tp) return;
      if (p.tp) out.push({ box: p.box, color: COL.tp, dashed: true, label: `TP ${className(p.box[0])}${cf}` });
      else if (p.swap_gt >= 0) out.push({ box: p.box, color: COL.swap, dashed: true, label: `${className(p.box[0])}${cf} ≠ GT ${className(d.gt[p.swap_gt][0])}` });
      else out.push({ box: p.box, color: COL.fp, dashed: true, label: `FP ${className(p.box[0])}${cf}` });
    });
    return out;
  }

  function overlay(boxes) {
    const ov = h('div', { class: 'cmp-ov' });
    for (const b of boxes) {
      const [, xc, yc, w, hh] = b.box;
      const left = (xc - w / 2) * 100, topP = (yc - hh / 2) * 100;
      const el = h('div', { class: `cmp-bx ${b.faint ? 'faint' : ''}`, style: `left:${left}%;top:${topP}%;width:${w * 100}%;height:${hh * 100}%;border:2px ${b.dashed ? 'dashed' : 'solid'} ${b.color}` });
      if (layers.labels) {
        // Inside the box when it touches the top edge, anchored right when it starts near the right edge:
        // a label is never clipped by the image border.
        const inside = topP < 4, right = left + w * 100 > 80 && left > 55;
        el.appendChild(h('span', { class: `cmp-lbl ${inside ? 'in' : ''} ${right ? 'r' : ''}`, style: `background:${b.color};color:${textOn(b.color)}` }, b.label));
      }
      ov.appendChild(el);
    }
    return ov;
  }

  function paint() {
    if (!detail || closed) return;
    const d = detail.detail, image = detail.image;
    nameEl.textContent = midEllipsis(image.rel_path, 90); nameEl.title = image.rel_path;
    const panels = [];
    for (const side of sides) {
      const sd = d.sides[side];
      const c = sd ? sd.counts : { tp: 0, fp: 0, fn: 0, sw: 0 };
      const src = ev.sources[side];
      const chips = [];
      if (sd && sd.file_missing) chips.push(src.type === 'run' ? h('span', { class: 'cmp-v-chip warn', 'data-tip': 'The AI run behind this side was deleted since the evaluation.' }, 'run deleted')
        : h('span', { class: 'cmp-v-chip warn', 'data-tip': 'The prediction file used by the evaluation is gone (moved or deleted).' }, 'file missing'));
      else if (sd && sd.changed) chips.push(h('span', { class: 'cmp-v-chip warn', 'data-tip': sd.stored ? `At evaluation: TP ${sd.stored.tp} · FP ${sd.stored.fp} · FN ${sd.stored.fn}. Labels or predictions changed since.` : 'Changed since the evaluation.' }, 'changed since evaluation'));
      if (!d.in_eval) chips.push(h('span', { class: 'cmp-v-chip', 'data-tip': 'This image was not part of the evaluation (skipped or out of scope).' }, 'not evaluated'));
      const fileTxt = src.type === 'run' ? `${sd ? sd.preds.length : 0} stored prediction${sd && sd.preds.length === 1 ? '' : 's'}` : (sd && sd.file_host ? sd.file_host.split('/').slice(-2).join('/') : 'no prediction file');
      const head = h('div', { class: 'cmp-v-head', style: `--side:${SIDE_COLORS[side]}` },
        h('span', { class: 'cmp-v-side' }, side.toUpperCase()), h('span', { class: 'cmp-v-label', title: src.path_host || src.label }, src.label),
        h('span', { class: 'mono xs cmp-v-file', title: sd && sd.file_host ? sd.file_host : fileTxt }, fileTxt), h('span', { class: 'spacer' }), ...chips,
        h('span', { class: 'mono xs cmp-v-counts' }, `TP ${c.tp} · FP ${c.fp} · FN ${c.fn}${c.sw ? ` · ${c.sw} swap${c.sw > 1 ? 's' : ''}` : ''}`));
      const stage = h('div', { class: 'inspect-stage cmp-v-stage' });
      let img = null;
      if (im) { img = side === sides[0] ? im : im.cloneNode(); img.alt = ''; img.draggable = false; }
      const pic = h('div', { class: `pic ${img ? 'is-ready' : 'is-error'}` });
      if (img) { pic.appendChild(img); const bx = boxesFor(side); pic.appendChild(overlay(bx)); panelsBuilt = panelsBuilt || {}; panelsBuilt[side] = { img, boxes: bx, counts: c }; }
      stage.appendChild(pic);
      panels.push(h('div', { class: 'cmp-v-panel' }, head, stage));
    }
    stageWrap.replaceChildren(...panels);
    fit();
    const rv = image.review;
    const ours = rv && rv.status === 'flagged' && (rv.note || '').split('\n').some(l => l.startsWith(`Compare #${ev.id}:`));
    flagBtn.classList.toggle('active', !!ours); flagBtn.setAttribute('aria-pressed', String(!!ours));
    flagBtn.lastChild.textContent = ours ? 'Flagged' : 'Flag';
    statusEl.textContent = rv ? `${rv.status[0].toUpperCase()}${rv.status.slice(1)}${rv.note ? ` · ${rv.note.split('\n').slice(-1)[0]}` : ''}` : '';
    statusEl.title = rv && rv.note ? rv.note : '';
    statusEl.dataset.status = rv ? rv.status : '';
  }

  /** Scale each frame to fill its panel (small 640 px frames included), computed before the browser paints. */
  function fit() {
    for (const pic of stageWrap.querySelectorAll('.pic.is-ready')) {
      const img = pic.querySelector('img'); const stage = pic.parentElement;
      if (!img || !img.naturalWidth || !stage) continue;
      const aw = Math.max(50, stage.clientWidth - 16), ah = Math.max(50, stage.clientHeight - 16);
      const sc = Math.min(aw / img.naturalWidth, ah / img.naturalHeight);
      pic.style.width = `${Math.floor(img.naturalWidth * sc)}px`; pic.style.height = `${Math.floor(img.naturalHeight * sc)}px`;
    }
  }
  const ro = new ResizeObserver(() => fit());
  ro.observe(stageWrap);

  async function toggleFlag() {
    if (!detail || ev.legacy) return;
    const rv = detail.image.review;
    const ours = rv && rv.status === 'flagged' && (rv.note || '').split('\n').some(l => l.startsWith(`Compare #${ev.id}:`));
    if (rv && rv.status === 'excluded') { statusEl.textContent = 'Excluded images are not flagged — change the review in Explore first.'; return; }
    const id = detail.image.id;
    try {
      if (ours) await F.api.post(`${API}/evals/${ev.id}/unflag`, { dataset_id: ds.id, image_ids: [id] });
      else await F.api.post(`${API}/evals/${ev.id}/flag`, { dataset_id: ds.id, image_ids: [id], conf: k / 100, classes: classes || null, include_approved: true });
      const fresh = await F.api.get(`${API}/evals/${ev.id}/image/${id}`, { dataset_id: ds.id, conf: k / 100, classes: classes || undefined });
      if (closed || !detail || detail.image.id !== id) return;
      detail = fresh; paint();
      onFlagChange && onFlagChange();
    } catch (e) { statusEl.textContent = e.message; }
  }

  function editGt() { if (!row) return; close(); F.router.navigate(`/d/${ds.id}/annotate?img=${row.id}`); }

  async function copyPng() {
    const built = panelsBuilt && sides.map(s => panelsBuilt[s]).filter(Boolean);
    if (!built || !built.length || !built[0].img.naturalWidth) return;
    const W = built[0].img.naturalWidth, H = built[0].img.naturalHeight;
    const fs = Math.max(12, Math.round(H * 0.022)), barH = Math.round(fs * 2.1), footH = Math.round(fs * 2.4), gap = 4;
    const c = document.createElement('canvas'); c.width = W * built.length + gap * (built.length - 1); c.height = barH + H + footH;
    const g = c.getContext('2d');
    g.fillStyle = '#16171a'; g.fillRect(0, 0, c.width, c.height);
    built.forEach((p, n) => {
      const side = sides[n], x0 = n * (W + gap);
      g.fillStyle = SIDE_COLORS[side]; g.fillRect(x0, 0, W, barH);
      g.fillStyle = textOn(SIDE_COLORS[side]); g.font = `600 ${fs}px Inter, system-ui, sans-serif`; g.textBaseline = 'middle';
      g.fillText(`Model ${side.toUpperCase()} · ${ev.sources[side].label}`, x0 + 10, barH / 2);
      const ct = `TP ${p.counts.tp} · FP ${p.counts.fp} · FN ${p.counts.fn}${p.counts.sw ? ` · swaps ${p.counts.sw}` : ''}`;
      g.textAlign = 'right'; g.fillText(ct, x0 + W - 10, barH / 2); g.textAlign = 'left';
      g.drawImage(p.img, x0, barH, W, H);
      g.save(); g.beginPath(); g.rect(x0, barH, W, H); g.clip();
      for (const b of p.boxes) {
        const [, xc, yc, bw, bh] = b.box;
        const bx = x0 + (xc - bw / 2) * W, by = barH + (yc - bh / 2) * H, ww = bw * W, hh = bh * H;
        g.globalAlpha = b.faint ? 0.6 : 1;
        g.lineWidth = Math.max(2, W / 400); g.strokeStyle = b.color; g.setLineDash(b.dashed ? [8, 5] : []); g.strokeRect(bx, by, ww, hh); g.setLineDash([]);
        if (layers.labels) {
          const lf = Math.max(11, Math.round(fs * 0.8)); g.font = `600 ${lf}px Inter, system-ui, sans-serif`;
          const tw = g.measureText(b.label).width + 8, th = lf + 6;
          let lx = Math.min(Math.max(bx, x0), x0 + W - tw), ly = by - th;
          if (ly < barH) ly = by + 1;                     // flip inside the box at the top edge
          g.fillStyle = b.color; g.fillRect(lx, ly, tw, th);
          g.fillStyle = textOn(b.color); g.textBaseline = 'middle'; g.fillText(b.label, lx + 4, ly + th / 2);
        }
        g.globalAlpha = 1;
      }
      g.restore();
    });
    // footer: file, thresholds, legend
    const fy = barH + H + footH / 2;
    g.font = `${Math.round(fs * 0.85)}px ui-monospace, Menlo, monospace`; g.fillStyle = '#e5e7eb'; g.textBaseline = 'middle';
    g.fillText(`${detail.image.rel_path}   IoU ≥ ${ev.ious[ev.primary]}   conf ≥ ${(k / 100).toFixed(2)}`, 10, fy);
    let lx = c.width - 10; g.font = `${Math.round(fs * 0.8)}px Inter, system-ui, sans-serif`; g.textAlign = 'right';
    for (const [t, col, dashed] of [...legendItems].reverse()) {
      g.fillStyle = '#e5e7eb'; g.fillText(t, lx, fy); lx -= g.measureText(t).width + 6;
      g.strokeStyle = col; g.lineWidth = 2; g.setLineDash(dashed ? [4, 3] : []); g.strokeRect(lx - 14, fy - 6, 14, 12); g.setLineDash([]); lx -= 26;
    }
    g.textAlign = 'left';
    c.toBlob(async (blob) => {
      try { await navigator.clipboard.write([new ClipboardItem({ 'image/png': blob })]); statusEl.textContent = 'Copied side-by-side PNG'; }
      catch (e) { statusEl.textContent = `Clipboard blocked: ${e.message}`; }
    }, 'image/png');
  }

  async function step(d) {
    const n = i + d;
    if (n < 0) return;
    if (n >= list.loaded()) { const r = await list.get(n).catch(() => null); if (!r || closed) return; }
    if (n >= list.total()) return;
    i = n; load();
  }

  // A focused control of the viewer keeps the keys it acts on: Enter / Space press a button or follow a link,
  // arrows / Home / End / Page keys move the slider or a select. Every other key stays a viewer shortcut.
  const OWN_KEYS = { button: ['Enter', ' '], range: ['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown'] };
  function ownedByControl(e) {
    const el = document.activeElement;
    if (!el || !rootEl.contains(el) || el === rootEl) return false;
    const tag = el.tagName, type = (el.getAttribute('type') || '').toLowerCase();
    if (tag === 'BUTTON' || tag === 'A' || el.getAttribute('role') === 'button') return OWN_KEYS.button.includes(e.key);
    if ((tag === 'INPUT' && type === 'range') || tag === 'SELECT') return OWN_KEYS.range.includes(e.key);
    return false;
  }
  /** Tab cycles through the viewer's own controls: focus never wanders to the page underneath. */
  function trapTab(e) {
    const els = [...rootEl.querySelectorAll('button:not([disabled]), input:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])')]
      .filter(el => el.getClientRects().length);
    if (!els.length) return;
    const first = els[0], last = els[els.length - 1], a = document.activeElement;
    const to = !rootEl.contains(a) || a === rootEl ? (e.shiftKey ? last : first) : !e.shiftKey && a === last ? first : e.shiftKey && a === first ? last : null;
    if (to) { e.preventDefault(); to.focus(); }
  }
  const onKey = (e) => {
    if (closed || ui.hasModal() || F.isTyping() || ownedByControl(e)) return;
    if (e.key === 'Tab') { trapTab(e); return; }
    let handled = true;
    if (e.key === 'Escape') close();
    else if (keys.is(e, 'ArrowRight')) step(1);
    else if (keys.is(e, 'ArrowLeft')) step(-1);
    else if (keys.is(e, '[')) setK(k - 5);
    else if (keys.is(e, ']')) setK(k + 5);
    else if (keys.is(e, '{')) setK(k - 1);
    else if (keys.is(e, '}')) setK(k + 1);
    else if (keys.is(e, 'e') && keys.fresh(e)) toggle('errors');
    else if (keys.is(e, 'l') && keys.fresh(e)) toggle('labels');
    else if (keys.is(e, 'b') && keys.fresh(e)) toggle('below');
    else if (keys.is(e, 'f') && keys.fresh(e)) toggleFlag();
    else if (keys.is(e, 'Enter') && keys.fresh(e)) editGt();
    else if (keys.is(e, 'mod+c')) { if (String(window.getSelection() || '').length) { handled = false; } else copyPng(); }
    else handled = false;
    if (handled) { e.preventDefault(); e.stopPropagation(); }
  };
  document.addEventListener('keydown', onKey, true);
  function close() {
    if (closed) return;
    closed = true; seq++; frames.clear(); ro.disconnect();
    document.removeEventListener('keydown', onKey, true);
    rootEl.remove();
    if (opener && opener !== document.body && opener.isConnected) opener.focus({ preventScroll: true });
  }
  rootEl.focus({ preventScroll: true });
  load();
  return { close, get open() { return !closed; } };
}
