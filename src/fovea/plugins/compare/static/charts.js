// Small SVG charts for the Compare plugin: metric-vs-confidence lines, precision-recall curves, confusion matrix.
// Thin 2px lines, hairline grid, one y-axis, a legend for two series, hover crosshair + tooltip, click to pick.

const NS = 'http://www.w3.org/2000/svg';
const s = (tag, attrs = {}, ...kids) => { const el = document.createElementNS(NS, tag); for (const [k, v] of Object.entries(attrs)) if (v != null) el.setAttribute(k, v); kids.forEach(c => c != null && el.appendChild(typeof c === 'string' ? document.createTextNode(c) : c)); return el; };
const pct = (v) => (v == null || isNaN(v) ? '–' : `${(v * 100).toFixed(1)}%`);

/** Keep a chart sized to its container: render(width) runs now and again (debounced) on resize. */
function responsive(host, render) {
  let w = 0, t = null;
  const run = () => { const nw = Math.max(260, Math.floor(host.clientWidth)); if (nw !== w) { w = nw; render(w); } };
  const ro = new ResizeObserver(() => { clearTimeout(t); t = setTimeout(run, 60); });
  ro.observe(host);
  requestAnimationFrame(run);
  return () => ro.disconnect();
}

function tooltipEl(host) {
  let tip = host.querySelector(':scope > .cmp-tip');
  if (!tip) { tip = document.createElement('div'); tip.className = 'cmp-tip'; tip.hidden = true; host.appendChild(tip); }
  return tip;
}
function placeTip(host, tip, x, y) {
  const hw = host.clientWidth; tip.hidden = false;
  const tw = tip.offsetWidth, th = tip.offsetHeight;
  let left = x + 14; if (left + tw > hw - 4) left = x - tw - 14;
  tip.style.left = `${Math.max(4, left)}px`; tip.style.top = `${Math.max(0, y - th / 2)}px`;
}
function tipRows(tip, title, rows) {
  tip.replaceChildren();
  const head = document.createElement('div'); head.className = 'cmp-tip-h'; head.textContent = title; tip.appendChild(head);
  for (const r of rows) {
    const row = document.createElement('div'); row.className = 'cmp-tip-r';
    const key = document.createElement('i'); key.style.background = r.color; row.appendChild(key);
    const v = document.createElement('b'); v.textContent = r.value; row.appendChild(v);
    const l = document.createElement('span'); l.textContent = r.label; row.appendChild(l);
    tip.appendChild(row);
  }
}

function legend(series) {
  if (series.length < 2) return null;
  const el = document.createElement('div'); el.className = 'cmp-legend';
  for (const sr of series) {
    const it = document.createElement('span'); it.className = 'cmp-legend-i';
    const key = document.createElement('i'); key.style.background = sr.color; it.appendChild(key);
    it.appendChild(document.createTextNode(sr.label)); el.appendChild(it);
  }
  return el;
}

const M = { l: 52, r: 14, t: 12, b: 30 };          // l: room for '100%' ticks AND the rotated axis label beside them

function frame(svg, W, H, { xTicks, yTicks, xFmt, yFmt, xLabel, yLabel }) {
  const pw = W - M.l - M.r, ph = H - M.t - M.b;
  const X = (v) => M.l + v * pw, Y = (v) => M.t + (1 - v) * ph;
  const g = s('g', { class: 'cmp-axes' });
  for (const v of yTicks) { g.appendChild(s('line', { x1: M.l, x2: W - M.r, y1: Y(v), y2: Y(v), class: v === 0 ? 'cmp-base' : 'cmp-grid' })); g.appendChild(s('text', { x: M.l - 6, y: Y(v) + 3.5, 'text-anchor': 'end' }, yFmt(v))); }
  for (const v of xTicks) g.appendChild(s('text', { x: X(v), y: H - M.b + 15, 'text-anchor': v === 0 ? 'start' : v === 1 ? 'end' : 'middle' }, xFmt(v)));
  if (xLabel) g.appendChild(s('text', { x: M.l + pw / 2, y: H - 3, 'text-anchor': 'middle', class: 'cmp-axis-label' }, xLabel));
  if (yLabel) g.appendChild(s('text', { x: 9, y: M.t + ph / 2, 'text-anchor': 'middle', transform: `rotate(-90 9 ${M.t + ph / 2})`, class: 'cmp-axis-label' }, yLabel));
  svg.appendChild(g);
  return { X, Y, pw, ph };
}

const path = (pts, X, Y) => pts.map((p, i) => `${i ? 'L' : 'M'}${X(p[0]).toFixed(1)},${Y(p[1]).toFixed(1)}`).join('');

/**
 * Metric vs confidence threshold. series: [{label, color, values: [101 numbers in 0..1] }].
 * marks: [{series index, k, text}] (best-F1 dots). k: current threshold index. onPick(k).
 */
export function confChart(host, { series, k, marks = [], yLabel, onPick, height = 230 }) {
  host.classList.add('cmp-chart');
  const lg = legend(series);
  const area = document.createElement('div'); area.className = 'cmp-chart-area'; area.style.height = `${height}px`;
  host.replaceChildren(...[lg, area].filter(Boolean));
  let svgEl = null, cur = k;
  const draw = (W) => {
    const H = height;
    const svgN = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': `${yLabel} by confidence threshold` });
    const { X, Y } = frame(svgN, W, H, { xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], xFmt: v => v.toFixed(1), yFmt: v => `${Math.round(v * 100)}%`, xLabel: 'confidence threshold', yLabel });
    for (const sr of series) svgN.appendChild(s('path', { d: path(sr.values.map((v, i) => [i / 100, v]), X, Y), class: 'cmp-line', style: `stroke:${sr.color}` }));
    const thr = s('g', { class: 'cmp-thr' });
    const tline = s('line', { y1: M.t, y2: H - M.b, class: 'cmp-thr-line' });
    const tlabel = s('text', { y: M.t - 2, class: 'cmp-thr-label' });
    thr.appendChild(tline); thr.appendChild(tlabel); svgN.appendChild(thr);
    const setThr = (kk) => { const x = X(kk / 100); tline.setAttribute('x1', x); tline.setAttribute('x2', x); tlabel.setAttribute('x', x); tlabel.setAttribute('text-anchor', kk > 85 ? 'end' : kk < 15 ? 'start' : 'middle'); tlabel.textContent = `≥ ${(kk / 100).toFixed(2)}`; };
    setThr(cur);
    for (const m of marks) {
      const sr = series[m.series]; if (!sr) continue;
      const x = X(m.k / 100), y = Y(sr.values[m.k]);
      svgN.appendChild(s('circle', { cx: x, cy: y, r: 4.5, class: 'cmp-dot', style: `fill:${sr.color}` }));
    }
    const cross = s('line', { y1: M.t, y2: H - M.b, class: 'cmp-cross', visibility: 'hidden' });
    svgN.appendChild(cross);
    const dots = series.map(sr => { const d = s('circle', { r: 4, class: 'cmp-dot', style: `fill:${sr.color}`, visibility: 'hidden' }); svgN.appendChild(d); return d; });
    const hit = s('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: 'transparent', class: 'cmp-hit', tabindex: '0' });
    svgN.appendChild(hit);
    const tip = tooltipEl(area);
    const kAt = (ev) => { const r = svgN.getBoundingClientRect(); return Math.max(0, Math.min(100, Math.round(((ev.clientX - r.left) - M.l) / (W - M.l - M.r) * 100))); };
    const show = (kk, px, py) => {
      const x = X(kk / 100); cross.setAttribute('x1', x); cross.setAttribute('x2', x); cross.setAttribute('visibility', 'visible');
      series.forEach((sr, i) => { dots[i].setAttribute('cx', x); dots[i].setAttribute('cy', Y(sr.values[kk])); dots[i].setAttribute('visibility', 'visible'); });
      tipRows(tip, `confidence ≥ ${(kk / 100).toFixed(2)}`, series.map(sr => ({ color: sr.color, value: pct(sr.values[kk]), label: sr.label })));
      placeTip(area, tip, px, py);
    };
    const hide = () => { cross.setAttribute('visibility', 'hidden'); dots.forEach(d => d.setAttribute('visibility', 'hidden')); tip.hidden = true; };
    hit.addEventListener('pointermove', (ev) => { const r = area.getBoundingClientRect(); show(kAt(ev), ev.clientX - r.left, ev.clientY - r.top); });
    hit.addEventListener('pointerleave', hide);
    hit.addEventListener('click', (ev) => onPick && onPick(kAt(ev)));
    hit.addEventListener('focus', () => show(cur, X(cur / 100), H / 2));
    hit.addEventListener('blur', hide);
    hit.addEventListener('keydown', (ev) => {
      if (ev.key === 'ArrowLeft' || ev.key === 'ArrowRight') { ev.preventDefault(); ev.stopPropagation(); const kk = Math.max(0, Math.min(100, cur + (ev.key === 'ArrowRight' ? 1 : -1) * (ev.shiftKey ? 5 : 1))); onPick && onPick(kk); show(kk, X(kk / 100), H / 2); }
    });
    svgEl = svgN; svgN._setThr = setThr;
    area.replaceChildren(svgN, tip);
  };
  const stop = responsive(area, draw);
  return { setK(kk) { cur = kk; if (svgEl && svgEl._setThr) svgEl._setThr(kk); }, destroy: stop };
}

/**
 * Precision-recall as the confidence threshold sweeps. series: [{label, color, p: [101], r: [101]}]. k marks the
 * current operating point on every curve. onPick(k) with the threshold of the nearest point.
 */
export function prChart(host, { series, k, onPick, height = 230 }) {
  host.classList.add('cmp-chart');
  const lg = legend(series);
  const area = document.createElement('div'); area.className = 'cmp-chart-area'; area.style.height = `${height}px`;
  host.replaceChildren(...[lg, area].filter(Boolean));
  let api = null, cur = k;
  const draw = (W) => {
    const H = height;
    const svgN = s('svg', { width: W, height: H, viewBox: `0 0 ${W} ${H}`, role: 'img', 'aria-label': 'Precision against recall as the confidence threshold changes' });
    const { X, Y } = frame(svgN, W, H, { xTicks: [0, 0.2, 0.4, 0.6, 0.8, 1], yTicks: [0, 0.25, 0.5, 0.75, 1], xFmt: v => `${Math.round(v * 100)}%`, yFmt: v => `${Math.round(v * 100)}%`, xLabel: 'recall', yLabel: 'precision' });
    const pts = series.map(sr => sr.r.map((r, i) => ({ k: i, r, p: sr.p[i], valid: sr.tpfp ? sr.tpfp[i] > 0 : true })).filter(q => q.valid));
    series.forEach((sr, i) => svgN.appendChild(s('path', { d: path(pts[i].map(q => [q.r, q.p]), X, Y), class: 'cmp-line', style: `stroke:${sr.color}` })));
    const opDots = series.map(sr => { const d = s('circle', { r: 4.5, class: 'cmp-dot', style: `fill:${sr.color}` }); svgN.appendChild(d); return d; });
    const setOp = (kk) => series.forEach((sr, i) => { const ok = !sr.tpfp || sr.tpfp[kk] > 0; opDots[i].setAttribute('visibility', ok ? 'visible' : 'hidden'); if (ok) { opDots[i].setAttribute('cx', X(sr.r[kk])); opDots[i].setAttribute('cy', Y(sr.p[kk])); } });
    setOp(cur);
    const ring = s('circle', { r: 6, class: 'cmp-ring', visibility: 'hidden' }); svgN.appendChild(ring);
    const hit = s('rect', { x: M.l, y: M.t, width: W - M.l - M.r, height: H - M.t - M.b, fill: 'transparent', class: 'cmp-hit' });
    svgN.appendChild(hit);
    const tip = tooltipEl(area);
    const nearest = (ev) => {
      const r = svgN.getBoundingClientRect(); const mx = ev.clientX - r.left, my = ev.clientY - r.top;
      let best = null;
      pts.forEach((list, si) => list.forEach(q => { const d = (X(q.r) - mx) ** 2 + (Y(q.p) - my) ** 2; if (!best || d < best.d) best = { d, si, q }; }));
      return best && best.d < 40 * 40 ? best : null;
    };
    hit.addEventListener('pointermove', (ev) => {
      const b = nearest(ev); const ar = area.getBoundingClientRect();
      if (!b) { ring.setAttribute('visibility', 'hidden'); tip.hidden = true; return; }
      ring.setAttribute('cx', X(b.q.r)); ring.setAttribute('cy', Y(b.q.p)); ring.style.stroke = series[b.si].color; ring.setAttribute('visibility', 'visible');
      tipRows(tip, `${series[b.si].label} · confidence ≥ ${(b.q.k / 100).toFixed(2)}`, [{ color: series[b.si].color, value: pct(b.q.p), label: 'precision' }, { color: series[b.si].color, value: pct(b.q.r), label: 'recall' }]);
      placeTip(area, tip, ev.clientX - ar.left, ev.clientY - ar.top);
    });
    hit.addEventListener('pointerleave', () => { ring.setAttribute('visibility', 'hidden'); tip.hidden = true; });
    hit.addEventListener('click', (ev) => { const b = nearest(ev); if (b && onPick) onPick(b.q.k); });
    api = { setOp };
    area.replaceChildren(svgN, tip);
  };
  const stop = responsive(area, draw);
  return { setK(kk) { cur = kk; if (api) api.setOp(kk); }, destroy: stop };
}

// ------------------------------------------------------------------ confusion matrix

const hex = (c) => { c = c.trim(); if (c.startsWith('#') && c.length === 4) c = '#' + [...c.slice(1)].map(x => x + x).join(''); return c; };
const rgb = (c) => { c = hex(c); return [1, 3, 5].map(i => parseInt(c.slice(i, i + 2), 16)); };
const toHex = (a) => '#' + a.map(v => Math.round(Math.max(0, Math.min(255, v))).toString(16).padStart(2, '0')).join('');

/**
 * Confusion matrix as a table. data: {labels, matrix (rows GT + background, cols predicted + missed)}. Each row is
 * shaded by its share of that row's total (one blue ramp). onCell(gt, pred) where gt = cls|'bg', pred = cls|'miss'.
 */
export function confusionTable(host, { data, name, textOn, onCell, active }) {
  const cs = getComputedStyle(host);
  const lo = rgb(cs.getPropertyValue('--cmp-cm-lo') || '#ffffff'), hi = rgb(cs.getPropertyValue('--cmp-cm-hi') || '#184f95');
  const mix = (t) => toHex(lo.map((v, i) => v + (hi[i] - v) * t));
  const L = data.labels, Mx = data.matrix, n = L.length;
  const table = document.createElement('table'); table.className = 'cmp-cm';
  const thead = document.createElement('thead'); const hr = document.createElement('tr');
  const corner = document.createElement('th'); corner.className = 'cmp-cm-corner'; corner.innerHTML = '<span>predicted →</span><span>ground truth ↓</span>'; hr.appendChild(corner);
  // Column names may wrap, but only after _ - / · or a space (never mid-word): 'standing_' / 'person'.
  const wrapName = (th, t) => String(t).split(/(?<=[_\-/·\s])/).forEach((part, i) => { if (i) th.appendChild(document.createElement('wbr')); th.appendChild(document.createTextNode(part)); });
  [...L.map(c => name(c)), 'missed'].forEach((t, j) => { const th = document.createElement('th'); th.className = j === n ? 'cmp-cm-x' : ''; th.title = j === n ? 'No prediction reached these boxes (false negatives)' : `Predicted ${t}`; wrapName(th, t); hr.appendChild(th); });
  thead.appendChild(hr); table.appendChild(thead);
  const tbody = document.createElement('tbody');
  for (let i = 0; i <= n; i++) {
    const tr = document.createElement('tr');
    const th = document.createElement('th'); th.textContent = i === n ? 'background' : name(L[i]); th.title = i === n ? 'Predictions on nothing (false positives)' : `Ground truth ${name(L[i])}`; if (i === n) th.className = 'cmp-cm-x'; tr.appendChild(th);
    const rowTotal = Mx[i].reduce((a, b) => a + b, 0);
    for (let j = 0; j <= n; j++) {
      const v = Mx[i][j]; const td = document.createElement('td');
      if (i === n && j === n) { td.className = 'cmp-cm-void'; tr.appendChild(td); continue; }
      const share = rowTotal ? v / rowTotal : 0;
      const bg = mix(v ? 0.08 + 0.92 * share : 0);
      td.style.background = bg; td.style.color = textOn(bg);
      td.textContent = v ? v.toLocaleString('en-US') : '·';
      const g = i === n ? 'bg' : L[i], p = j === n ? 'miss' : L[j];
      const what = i === n ? `${v.toLocaleString('en-US')} predictions of ${name(L[j])} on background (false positives)`
        : j === n ? `${v.toLocaleString('en-US')} boxes of ${name(L[i])} missed (no prediction)`
          : i === j ? `${v.toLocaleString('en-US')} boxes of ${name(L[i])} found (true positives)`
            : `${v.toLocaleString('en-US')} boxes of ${name(L[i])} predicted as ${name(L[j])}`;
      td.title = `${what}${rowTotal && i < n ? ` · ${(share * 100).toFixed(1)}% of the row` : ''}`;
      td.className = `${i === j ? 'diag' : ''} ${v ? 'clickable' : ''} ${active === `${g}:${p}` ? 'active' : ''}`;
      if (v && onCell) { td.tabIndex = 0; td.addEventListener('click', () => onCell(g, p)); td.addEventListener('keydown', (e) => { if (e.key === 'Enter') onCell(g, p); }); }
      tr.appendChild(td);
    }
    tbody.appendChild(tr);
  }
  table.appendChild(tbody);
  host.replaceChildren(table);
}
