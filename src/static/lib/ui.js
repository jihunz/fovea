import { h, clear, mount, on, isTyping } from './dom.js';
import { icon } from './icons.js';

// ------------------------------------------------------------------ toast
let stack;
function ensureStack() { if (!stack) { stack = h('div', { class: 'toast-stack' }); document.body.appendChild(stack); } return stack; }
export function toast(message, { type = 'info', timeout = 3200, action } = {}) {
  const el = h('div', { class: `toast ${type}` }, type === 'error' ? icon('alertCircle', 15) : type === 'ok' ? icon('checkCircle', 15) : null, h('span', message));
  if (action) el.appendChild(h('button', { class: 'btn btn-sm', onClick: () => { action.onClick(); remove(); } }, action.label));
  ensureStack().appendChild(el);
  const remove = () => { el.remove(); };
  if (timeout) setTimeout(remove, timeout);
  return remove;
}

// ------------------------------------------------------------------ modal
const modals = [];
export function modal({ title, body, footer, size, onClose, closable = true, className = '' }) {
  const backdrop = h('div', { class: 'backdrop' });
  const box = h('div', { class: `modal ${size || ''} ${className}` });
  const close = (result) => {
    if (!backdrop.isConnected) return;
    backdrop.remove(); modals.splice(modals.indexOf(api), 1);
    onClose && onClose(result);
  };
  const api = { close, el: box, backdrop };
  const header = h('div', { class: 'modal-header' }, h('h2', title), h('span', { class: 'spacer' }),
    closable ? h('button', { class: 'btn btn-ghost btn-icon btn-sm', onClick: () => close(), 'aria-label': 'Close' }, icon('x', 15)) : null);
  const bodyEl = h('div', { class: 'modal-body' }, typeof body === 'function' ? body(api) : body);
  box.appendChild(header); box.appendChild(bodyEl);
  const f = typeof footer === 'function' ? footer(api) : footer;
  if (f) box.appendChild(h('div', { class: 'modal-footer' }, f));
  backdrop.appendChild(box);
  backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop && closable) close(); });
  document.body.appendChild(backdrop);
  modals.push(api);
  api.body = bodyEl;
  setTimeout(() => { const first = box.querySelector('input,select,textarea,button.btn-primary'); first && first.focus(); }, 30);
  return api;
}
document.addEventListener('keydown', (e) => {
  if (e.key === 'Escape' && modals.length) { const m = modals[modals.length - 1]; if (m && m.el.isConnected) { e.stopPropagation(); m.close(); } }
}, true);
export const hasModal = () => modals.length > 0;

export function confirm({ title = 'Are you sure?', message, okLabel = 'Confirm', cancelLabel = 'Cancel', danger = false }) {
  return new Promise((resolve) => {
    const m = modal({
      title, body: h('p', { style: 'margin:0' }, message), onClose: () => resolve(false),
      footer: (api) => [
        h('button', { class: 'btn', onClick: () => api.close() }, cancelLabel),
        h('button', { class: `btn ${danger ? 'btn-danger' : 'btn-primary'}`, onClick: () => { resolve(true); api.onDone = true; api.backdrop.remove(); modals.splice(modals.indexOf(api), 1); } }, okLabel),
      ],
    });
  });
}

export function prompt({ title, label, value = '', placeholder = '', okLabel = 'OK', mono = false }) {
  return new Promise((resolve) => {
    let input;
    const submit = (api) => { const v = input.value; api.backdrop.remove(); modals.splice(modals.indexOf(api), 1); resolve(v); };
    const m = modal({
      title, onClose: () => resolve(null),
      body: (api) => h('div', { class: 'field' }, label ? h('label', { class: 'label' }, label) : null,
        input = h('input', { class: `input ${mono ? 'mono' : ''}`, value, placeholder, onKeydown: (e) => { if (e.key === 'Enter') submit(api); } })),
      footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: () => submit(api) }, okLabel)],
    });
  });
}

// ------------------------------------------------------------------ menu
let openMenu = null;
export function menu(anchor, items) {
  closeMenu();
  const el = h('div', { class: 'menu', role: 'menu' });
  for (const it of items) {
    if (!it) continue;
    if (it.sep) { el.appendChild(h('div', { class: 'menu-sep' })); continue; }
    if (it.label && !it.onClick && it.header) { el.appendChild(h('div', { class: 'menu-label' }, it.label)); continue; }
    el.appendChild(h('button', { class: `menu-item ${it.danger ? 'danger' : ''} ${it.disabled ? 'disabled' : ''}`, disabled: !!it.disabled,
      onClick: (e) => { e.stopPropagation(); closeMenu(); it.onClick && it.onClick(e); } },
      it.icon ? icon(it.icon, 14) : h('span', { class: 'ico' }), h('span', it.label), it.kbd ? kbd(it.kbd) : null));
  }
  document.body.appendChild(el);
  const r = anchor instanceof Element ? anchor.getBoundingClientRect() : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
  const mw = el.offsetWidth, mh = el.offsetHeight;
  let x = r.left, y = r.bottom + 4;
  if (x + mw > innerWidth - 8) x = Math.max(8, r.right - mw);
  if (y + mh > innerHeight - 8) y = Math.max(8, r.top - mh - 4);
  el.style.left = x + 'px'; el.style.top = y + 'px';
  openMenu = el;
  setTimeout(() => { document.addEventListener('mousedown', outside, { once: true }); }, 0);
  function outside(e) { if (!el.contains(e.target)) closeMenu(); else document.addEventListener('mousedown', outside, { once: true }); }
  return el;
}
export function closeMenu() { if (openMenu) { openMenu.remove(); openMenu = null; } }
document.addEventListener('keydown', (e) => { if (e.key === 'Escape') closeMenu(); });
window.addEventListener('resize', closeMenu);

// ------------------------------------------------------------------ popover (anchored panel that stays open)
export function popover(anchor, content, { width } = {}) {
  closeMenu();
  const el = h('div', { class: 'menu filter-pop', style: width ? `width:${width}px` : '' }, content);
  document.body.appendChild(el);
  const r = anchor.getBoundingClientRect();
  let x = r.left, y = r.bottom + 4;
  if (x + el.offsetWidth > innerWidth - 8) x = Math.max(8, r.right - el.offsetWidth);
  if (y + el.offsetHeight > innerHeight - 8) y = Math.max(8, r.top - el.offsetHeight - 4);
  el.style.left = x + 'px'; el.style.top = y + 'px';
  openMenu = el;
  setTimeout(() => { document.addEventListener('mousedown', outside); }, 0);
  function outside(e) { if (!el.contains(e.target)) { closeMenu(); document.removeEventListener('mousedown', outside); } }
  return { el, close: () => { closeMenu(); document.removeEventListener('mousedown', outside); } };
}

// ------------------------------------------------------------------ tooltips via [data-tip]
let tipEl, tipTimer;
document.addEventListener('mouseover', (e) => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (!t) return;
  clearTimeout(tipTimer);
  tipTimer = setTimeout(() => {
    if (!t.isConnected) return;
    tipEl = tipEl || h('div', { class: 'tip' });
    tipEl.textContent = t.dataset.tip; document.body.appendChild(tipEl);
    const r = t.getBoundingClientRect();
    let x = r.left + r.width / 2 - tipEl.offsetWidth / 2, y = r.bottom + 6;
    x = Math.max(6, Math.min(innerWidth - tipEl.offsetWidth - 6, x));
    if (y + tipEl.offsetHeight > innerHeight - 6) y = r.top - tipEl.offsetHeight - 6;
    tipEl.style.left = x + 'px'; tipEl.style.top = y + 'px';
  }, 450);
});
document.addEventListener('mouseout', (e) => { if (e.target.closest && e.target.closest('[data-tip]')) { clearTimeout(tipTimer); tipEl && tipEl.remove(); } });
document.addEventListener('mousedown', () => { clearTimeout(tipTimer); tipEl && tipEl.remove(); });

// ------------------------------------------------------------------ small bits
export function kbd(k) { return h('span', { class: 'kbd' }, k); }
export function spinner() { return h('span', { class: 'spinner' }); }
export function chip(text, { type = '', icon: ic, onClick, onRemove, cls = '' } = {}) {
  return h('span', { class: `chip ${type} ${onClick ? 'clickable' : ''} ${cls}`, onClick }, ic ? icon(ic, 12) : null, text,
    onRemove ? h('span', { class: 'x', onClick: (e) => { e.stopPropagation(); onRemove(); } }, icon('x', 11)) : null);
}
export function emptyState({ icon: ic = 'images', title, message, action }) {
  return h('div', { class: 'empty' }, icon(ic, 36), h('h3', title), message ? h('p', message) : null, action || null);
}
export function progress(pct, { indeterminate = false, cls = '' } = {}) {
  return h('div', { class: `progress ${indeterminate ? 'indeterminate' : ''} ${cls}` }, h('i', { style: `width:${indeterminate ? 40 : Math.round(pct * 100)}%` }));
}
export async function copyText(text, label = 'Copied') {
  try { await navigator.clipboard.writeText(text); toast(label, { type: 'ok', timeout: 1500 }); }
  catch (e) { toast('Clipboard unavailable', { type: 'error' }); }
}
export function seg(options, value, onChange, { size } = {}) {
  const el = h('div', { class: 'seg' });
  const render = (v) => { clear(el); for (const o of options) el.appendChild(h('button', { class: o.value === v ? 'active' : '', type: 'button', onClick: () => { if (o.value !== v) { v = o.value; render(v); onChange(v); } }, 'data-tip': o.tip || null }, o.icon ? icon(o.icon, 13) : null, o.label != null ? h('span', o.label) : null)); };
  render(value);
  el.setValue = (v) => render(v);
  return el;
}
export function switchBtn(checked, onChange) {
  const b = h('button', { class: 'switch', role: 'switch', 'aria-checked': String(!!checked), type: 'button' });
  b.addEventListener('click', () => { const v = b.getAttribute('aria-checked') !== 'true'; b.setAttribute('aria-checked', String(v)); onChange(v); });
  b.setValue = (v) => b.setAttribute('aria-checked', String(!!v));
  return b;
}
export function statTile({ label, value, sub, icon: ic, onClick, tone }) {
  return h('div', { class: `stat ${onClick ? 'clickable' : ''}`, onClick }, h('div', { class: 'k' }, ic ? icon(ic, 13) : null, label),
    h('div', { class: 'v', style: tone ? `color:var(--${tone})` : '' }, value), sub ? h('div', { class: 's' }, sub) : null);
}
export function shortcutsModal(groups) {
  modal({ title: 'Keyboard shortcuts', size: 'lg', body: h('div', { class: 'col gap-16' }, groups.map(g => h('div', null, h('div', { class: 'small strong muted mb-8' }, g.title),
    h('div', { class: 'shortcut-grid' }, g.items.map(([keys, desc]) => h('div', { class: 'shortcut-row' }, h('span', desc), h('span', { class: 'keys' }, keys.split(' ').map(k => kbd(k)))))))) ) });
}
export { isTyping };
