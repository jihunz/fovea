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
  if (props && (props instanceof Node || typeof props !== 'object' || Array.isArray(props))) { children.unshift(props); props = null; }
  applyProps(el, props);
  children.forEach(c => append(el, c));
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

export const fmt = {
  num(n) { if (n == null || isNaN(n)) return '–'; return Number(n).toLocaleString('en-US'); },
  compact(n) { if (n == null) return '–'; if (n < 1000) return String(n); if (n < 1e6) return (n / 1e3).toFixed(n < 1e4 ? 1 : 0) + 'k'; return (n / 1e6).toFixed(1) + 'M'; },
  bytes(b) { if (b == null) return '–'; const u = ['B', 'KB', 'MB', 'GB', 'TB']; let i = 0; let n = b; while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; } return `${n < 10 && i > 0 ? n.toFixed(1) : Math.round(n)} ${u[i]}`; },
  pct(a, b, digits = 0) { if (!b) return '0%'; return (100 * a / b).toFixed(digits) + '%'; },
  ago(ts) {
    if (!ts) return '–'; const s = Math.max(0, Date.now() / 1000 - ts);
    if (s < 60) return 'just now'; if (s < 3600) return `${Math.floor(s / 60)}m ago`; if (s < 86400) return `${Math.floor(s / 3600)}h ago`;
    if (s < 86400 * 30) return `${Math.floor(s / 86400)}d ago`; return new Date(ts * 1000).toLocaleDateString();
  },
  date(ts) { return ts ? new Date(ts * 1000).toLocaleString() : '–'; },
  dur(sec) { if (sec == null) return '–'; if (sec < 60) return `${sec.toFixed(sec < 10 ? 1 : 0)}s`; const m = Math.floor(sec / 60); return `${m}m ${Math.round(sec - m * 60)}s`; },
  fixed(n, d = 3) { return n == null ? '–' : Number(n).toFixed(d); },
};

export function cls(...names) { return names.filter(Boolean).join(' '); }
export function isTyping() {
  const a = document.activeElement; if (!a) return false;
  return a.tagName === 'INPUT' || a.tagName === 'TEXTAREA' || a.tagName === 'SELECT' || a.isContentEditable;
}
export function basename(p) { return (p || '').split('/').pop(); }
export const isMac = /Mac|iPhone|iPad/.test(navigator.platform);
export const MOD = isMac ? '⌘' : 'Ctrl';
