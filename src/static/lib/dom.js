// Tiny DOM helpers — no framework, no build step.
export const SVG_NS = 'http://www.w3.org/2000/svg';

function append(parent, child) {
  if (child == null || child === false || child === true) return;
  if (Array.isArray(child)) { child.forEach(c => append(parent, c)); return; }
  if (child instanceof Node) { parent.appendChild(child); return; }
  parent.appendChild(document.createTextNode(String(child)));
}

function applyProps(el, props) {
  if (!props) return;
  for (const [k, v] of Object.entries(props)) {
    if (v == null || v === false) { if (k === 'value') el.value = ''; continue; }
    if (k === 'class' || k === 'className') el.className = Array.isArray(v) ? v.filter(Boolean).join(' ') : v;
    else if (k === 'style') { if (typeof v === 'string') el.style.cssText = v; else Object.assign(el.style, v); }
    else if (k === 'dataset') Object.assign(el.dataset, v);
    else if (k === 'html') el.innerHTML = v;
    else if (k === 'ref') v(el);
    else if (k === 'attrs') for (const [a, b] of Object.entries(v)) el.setAttribute(a, b);
    else if (k.startsWith('on') && typeof v === 'function') el.addEventListener(k.slice(2).toLowerCase(), v);
    else if (k in el && (typeof v === 'boolean' || k === 'value' || k === 'checked' || k === 'selected' || k === 'indeterminate')) el[k] = v;
    else el.setAttribute(k, v === true ? '' : v);
  }
}

export function h(tag, props, ...children) {
  const el = document.createElement(tag);
  // Any primitive in the props slot is a child — including 0 and '' (a count of 0 must still render).
  if (props != null && (props instanceof Node || typeof props !== 'object' || Array.isArray(props))) { children.unshift(props); props = null; }
  applyProps(el, props);
  children.forEach(c => append(el, c));
  // Icon-only controls get their accessible name from the tooltip text, so screen readers and voice
  // control can reach them without every call site repeating the label.
  if ((tag === 'button' || tag === 'a') && props && props['data-tip'] && !el.hasAttribute('aria-label') && !el.textContent.trim()) {
    el.setAttribute('aria-label', props['data-tip']);
  }
  return el;
}

export function svg(tag, attrs, ...children) {
  const el = document.createElementNS(SVG_NS, tag);
  if (attrs) for (const [k, v] of Object.entries(attrs)) { if (v != null) el.setAttribute(k, v); }
  children.forEach(c => append(el, c));
  return el;
}

export function clear(el) { while (el.firstChild) el.removeChild(el.firstChild); return el; }
export function mount(el, ...children) { clear(el); children.forEach(c => append(el, c)); return el; }
export function frag(...children) { const f = document.createDocumentFragment(); children.forEach(c => append(f, c)); return f; }
export function on(el, evt, fn, opts) { el.addEventListener(evt, fn, opts); return () => el.removeEventListener(evt, fn, opts); }

export function debounce(fn, ms = 200) { let t; return (...a) => { clearTimeout(t); t = setTimeout(() => fn(...a), ms); }; }
export function throttle(fn, ms = 100) { let last = 0, t; return (...a) => { const now = Date.now(); const rem = ms - (now - last); if (rem <= 0) { last = now; fn(...a); } else { clearTimeout(t); t = setTimeout(() => { last = Date.now(); fn(...a); }, rem); } }; }

const pad2 = (n) => String(n).padStart(2, '0');
const MONTHS = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'];

export const fmt = {
  num(n) { if (n == null || isNaN(n)) return '–'; return Number(n).toLocaleString('en-US'); },
  compact(n) { if (n == null) return '–'; if (n < 1000) return String(n); if (n < 1e6) return (n / 1e3).toFixed(n < 1e4 ? 1 : 0) + 'k'; return (n / 1e6).toFixed(1) + 'M'; },
  bytes(b) { if (b == null) return '–'; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let n = b; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`; },
  /** A share as a percentage that never rounds to a misleading 0% or 100%: 2 of 7,808 is '<1%', 44,990 of
      45,000 is '>99%'; only a === b is 100% and only 0 is 0%. */
  pct(a, b, digits = 0) {
    if (!b || !a) return '0%';
    if (a >= b) return '100%';
    const v = 100 * a / b, s = v.toFixed(digits), lim = 10 ** -digits;
    if (Number(s) <= 0) return `<${lim}%`;
    if (Number(s) >= 100) return `>${(100 - lim).toFixed(digits)}%`;
    return s + '%';
  },
  ago(ts) {
    if (!ts) return '–'; const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return 'just now'; if (s < 3600) return `${Math.floor(s / 60)}m ago`; if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    if (s < 86400 * 30) return `${Math.floor(s / 86400)}d ago`; return fmt.day(ts);
  },
  /** Dates are written the same way whatever the browser's locale (the UI is English): 2026-10-13 15:46. */
  date(ts) {
    if (!ts) return '–';
    const d = new Date(ts * 1000);
    return `${d.getFullYear()}-${pad2(d.getMonth() + 1)}-${pad2(d.getDate())} ${pad2(d.getHours())}:${pad2(d.getMinutes())}`;
  },
  /** Oct 13, 2026 */
  day(ts) { if (!ts) return '–'; const d = new Date(ts * 1000); return `${MONTHS[d.getMonth()]} ${d.getDate()}, ${d.getFullYear()}`; },
  dur(sec) { if (sec == null) return '–'; if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)}s`; const m = Math.floor(sec / 60); return `${m}m ${Math.round(sec - m * 60)}s`; },
  /** Elapsed time as a clock: 0:07, 12:05, 1:02:03. */
  clock(sec) {
    if (sec == null || !isFinite(sec)) return '–';
    sec = Math.max(0, Math.floor(sec));
    const hh = Math.floor(sec / 3600), mm = Math.floor((sec % 3600) / 60), ss = sec % 60;
    return (hh ? `${hh}:${String(mm).padStart(2, '0')}` : String(mm)) + ':' + String(ss).padStart(2, '0');
  },
  fixed(n, d = 3) { return n == null ? '–' : Number(n).toFixed(d); },
  usd(v) { if (v == null || isNaN(v)) return '–'; v = Number(v); return v === 0 ? '$0' : v >= 0.1 ? `$${v.toFixed(2)}` : v >= 0.001 ? `$${v.toFixed(3)}` : '<$0.001'; },
  /** "1 image", "2 images" */
  plural(n, one, many = one + 's') { return `${fmt.num(n)} ${n === 1 ? one : many}`; },
};

export function cls(...names) { return names.filter(Boolean).join(' '); }

// Input types that take typed text. Sliders, checkboxes, radios, colour pickers and buttons do not: a focused
// slider must not swallow A / F / X or the arrow keys meant for the view.
const TEXT_INPUTS = new Set(['', 'text', 'search', 'email', 'url', 'tel', 'password', 'number', 'date', 'datetime-local', 'month', 'time', 'week']);
/** True when keystrokes go into a text field (where letters must type, not trigger shortcuts). */
export function isEditable(a) {
  if (!a || a === document.body) return false;
  if (a.isContentEditable) return true;
  if (a.tagName === 'TEXTAREA') return true;
  if (a.tagName === 'INPUT') return TEXT_INPUTS.has(a.type || '') && !a.disabled;
  return false;
}
export function isTyping() { return isEditable(document.activeElement); }
export function basename(p) { return (p || '').split('/').pop(); }
export const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = isMac ? '⌘' : 'Ctrl';

/** Split a string for middle ellipsis: [head, tail] where the tail keeps the distinguishing end
    (a frame number, the dataset folder). */
export function splitTail(text, tail = 14) {
  const s = String(text ?? '');
  if (s.length <= tail + 4) return [s, ''];
  // Prefer to cut at a path separator inside the tail window so the last segment stays whole.
  const cut = s.length - tail;
  const sep = s.lastIndexOf('/', s.length - 2);
  const at = sep > 0 && sep >= cut - tail && sep < s.length - 2 ? sep : cut;
  return [s.slice(0, at), s.slice(at)];
}
