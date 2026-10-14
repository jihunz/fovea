import { h, clear, mount, on, isTyping, fmt, splitTail } from './dom.js';
import { icon } from './icons.js';

// ------------------------------------------------------------------ toast
// Toasts are for batch results and errors only (docs/visual-comfort.md). At most three are visible; a repeat of
// the same message bumps a counter instead of stacking; errors stay until read (8 s minimum, paused while the
// pointer is over them, closable) and are announced assertively.
let stack;
const MAX_TOASTS = 3;
function ensureStack() {
  if (!stack || !stack.isConnected) { stack = h('div', { class: 'toast-stack', role: 'region', 'aria-label': 'Notifications' }); document.body.appendChild(stack); }
  return stack;
}
export function toast(message, { type = 'info', timeout = 3200, action } = {}) {
  const st = ensureStack();
  const text = String(message ?? '');
  const dup = [...st.children].find(t => t._key === `${type}|${text}`);
  if (dup) { dup._count = (dup._count || 1) + 1; dup._countEl.textContent = `×${dup._count}`; dup._countEl.hidden = false; dup._arm(); return dup._remove; }
  const isErr = type === 'error';
  const countEl = h('span', { class: 'toast-count', hidden: true });
  const el = h('div', { class: `toast ${type}`, role: isErr ? 'alert' : 'status', 'aria-live': isErr ? 'assertive' : 'polite' },
    isErr ? icon('alertCircle', 15) : type === 'ok' ? icon('checkCircle', 15) : null, h('span', { class: 'msg' }, text), countEl);
  let timer = null;
  const remove = () => { clearTimeout(timer); el.remove(); };
  if (action) el.appendChild(h('button', { class: 'btn btn-sm', onClick: () => { remove(); action.onClick(); } }, action.label));
  if (isErr || action) el.appendChild(h('button', { class: 'btn btn-ghost btn-icon btn-sm toast-x', 'aria-label': 'Dismiss', onClick: remove }, icon('x', 13)));
  const life = timeout ? (isErr ? Math.max(8000, timeout) : timeout) : 0;
  const arm = () => { clearTimeout(timer); if (life) timer = setTimeout(remove, life); };
  el.addEventListener('mouseenter', () => clearTimeout(timer));
  el.addEventListener('mouseleave', arm);
  Object.assign(el, { _key: `${type}|${text}`, _countEl: countEl, _arm: arm, _remove: remove });
  st.appendChild(el);
  while (st.children.length > MAX_TOASTS) st.firstChild.remove();
  arm();
  return remove;
}

// ------------------------------------------------------------------ modal
const modals = [];
const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
/**
 * Dialog with a backdrop, focus trap, Escape and focus restore. Options:
 *   title, body, footer (node | (api) => node), size ('lg' | 'xl'), className, closable (false: Escape and the
 *   backdrop do nothing), onClose(result), initialFocus (element or selector), bare (no header/body padding —
 *   the body is the whole dialog, e.g. the command palette), backdropClass, keepOnNavigate.
 */
export function modal({ title, body, footer, size, onClose, closable = true, className = '', initialFocus, bare = false, backdropClass = '', keepOnNavigate = false, labelledBy }) {
  const restoreTo = document.activeElement;
  closeMenu(); hideTip();
  const backdrop = h('div', { class: `backdrop ${backdropClass}` });
  const box = h('div', { class: `modal ${size || ''} ${className}`, role: 'dialog', 'aria-modal': 'true', 'aria-label': typeof title === 'string' ? title : null, tabindex: '-1' });
  // Every way out (buttons, Escape, backdrop, confirm/prompt results) goes through here.
  const close = (result) => {
    if (!backdrop.isConnected) return;
    backdrop.remove();
    const i = modals.indexOf(api); if (i >= 0) modals.splice(i, 1);
    // Hand focus back to what opened the dialog; otherwise keyboard users restart at the top of the page.
    if (restoreTo && restoreTo.isConnected && restoreTo !== document.body && typeof restoreTo.focus === 'function') {
      try { restoreTo.focus({ preventScroll: true }); } catch (e) { /* element gone */ }
    }
    onClose && onClose(result);
  };
  const api = { close, el: box, backdrop, closable, keepOnNavigate };
  let bodyEl;
  if (bare) {
    bodyEl = box;
    const b = typeof body === 'function' ? body(api) : body;
    if (b) box.appendChild(b);
  } else {
    const header = h('div', { class: 'modal-header' }, h('h2', { id: labelledBy || null }, title), h('span', { class: 'spacer' }),
      closable ? h('button', { class: 'btn btn-ghost btn-icon btn-sm', onClick: () => close(), 'aria-label': 'Close' }, icon('x', 15)) : null);
    bodyEl = h('div', { class: 'modal-body' }, typeof body === 'function' ? body(api) : body);
    box.appendChild(header); box.appendChild(bodyEl);
    const f = typeof footer === 'function' ? footer(api) : footer;
    if (f) box.appendChild(h('div', { class: 'modal-footer' }, f));
  }
  backdrop.appendChild(box);
  backdrop.addEventListener('mousedown', (e) => { if (e.target === backdrop && api.closable) close(); });
  document.body.appendChild(backdrop);
  modals.push(api);
  api.body = bodyEl;
  // Focus synchronously: keystrokes typed right after the click that opened the dialog must land in it.
  const want = typeof initialFocus === 'string' ? box.querySelector(initialFocus) : initialFocus;
  const first = want || box.querySelector('input:not([type=hidden]):not([disabled]),select,textarea,button.btn-primary:not([disabled])');
  try { (first || box).focus({ preventScroll: true }); } catch (e) { /* not focusable */ }
  return api;
}
document.addEventListener('keydown', (e) => {
  if (!modals.length) return;
  const m = modals[modals.length - 1];
  if (!m || !m.el.isConnected) return;
  if (e.key === 'Escape') {
    if (e.isComposing || openMenu) return;          // an open menu takes this Escape (handled below)
    // Consumed here: one Escape closes one layer, never also Inspect or the page behind it.
    e.preventDefault(); e.stopImmediatePropagation();
    if (m.closable) m.close();
    return;
  }
  if (e.key === 'Tab') {
    // Keep focus inside the top dialog: the page behind the backdrop is not operable.
    const f = [...m.el.querySelectorAll(FOCUSABLE)].filter(n => n.getClientRects().length);
    if (!f.length) { e.preventDefault(); m.el.focus(); return; }
    const first = f[0], last = f[f.length - 1], at = document.activeElement;
    if (!m.el.contains(at)) { e.preventDefault(); (e.shiftKey ? last : first).focus(); }
    else if (e.shiftKey && at === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && at === last) { e.preventDefault(); first.focus(); }
  }
}, true);
export const hasModal = () => modals.length > 0;
/** Close every dialog that does not ask to survive navigation (palette, pickers, a dialog bound to the old page). */
export function closeModals({ all = false } = {}) { [...modals].reverse().forEach(m => { if (all || !m.keepOnNavigate) m.close(); }); }

/** Trap Tab inside `el` (a full-screen panel that is not a ui.modal, e.g. Inspect). Returns release(). */
export function trapFocus(el, { restore = true } = {}) {
  const back = document.activeElement;
  const onKey = (e) => {
    if (e.key !== 'Tab' || modals.length) return;
    const f = [...el.querySelectorAll(FOCUSABLE)].filter(n => n.getClientRects().length);
    if (!f.length) { e.preventDefault(); el.focus(); return; }
    const first = f[0], last = f[f.length - 1], at = document.activeElement;
    if (!el.contains(at)) { e.preventDefault(); (e.shiftKey ? last : first).focus(); }
    else if (e.shiftKey && at === first) { e.preventDefault(); last.focus(); }
    else if (!e.shiftKey && at === last) { e.preventDefault(); first.focus(); }
  };
  document.addEventListener('keydown', onKey, true);
  return () => {
    document.removeEventListener('keydown', onKey, true);
    if (restore && back && back.isConnected && back !== document.body) { try { back.focus({ preventScroll: true }); } catch (e) { /* gone */ } }
  };
}

export function confirm({ title = 'Are you sure?', message, okLabel = 'Confirm', cancelLabel = 'Cancel', danger = false }) {
  return new Promise((resolve) => {
    let ok = false;
    modal({
      title, body: typeof message === 'string' || message == null ? h('p', { style: 'margin:0' }, message) : message, onClose: () => resolve(ok),
      footer: (api) => [
        h('button', { class: 'btn', onClick: () => api.close() }, cancelLabel),
        h('button', { class: `btn ${danger ? 'btn-danger' : 'btn-primary'}`, onClick: () => { ok = true; api.close(); } }, okLabel),
      ],
      initialFocus: danger ? '.modal-footer .btn' : null,
    });
  });
}

export function prompt({ title, label, value = '', placeholder = '', okLabel = 'OK', mono = false, hint, validate }) {
  return new Promise((resolve) => {
    let input, err, result = null;
    const submit = (api) => {
      const v = input.value;
      const problem = validate ? validate(v) : null;
      if (problem) { err.textContent = problem; err.hidden = false; input.focus(); return; }
      result = v; api.close();
    };
    modal({
      title, onClose: () => resolve(result),
      body: (api) => h('div', { class: 'field' }, label ? h('label', { class: 'label' }, label) : null,
        input = h('input', { class: `input ${mono ? 'mono' : ''}`, value, placeholder, 'aria-label': label || title, onKeydown: (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); submit(api); } } }),
        hint ? h('span', { class: 'hint' }, hint) : null,
        err = h('span', { class: 'hint danger-text', hidden: true, role: 'alert' })),
      footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'), h('button', { class: 'btn btn-primary', onClick: () => submit(api) }, okLabel)],
    });
    input.select();
  });
}

/** Right-side panel with modal semantics (focus trap, Escape, focus restore). Returns the modal api. */
export function drawer({ title, body, footer, onClose, width = 460, className = '', initialFocus } = {}) {
  const m = modal({ title, body, footer, onClose, className: `drawer ${className}`, backdropClass: 'drawer-backdrop', initialFocus });
  m.el.style.width = `min(${width}px, 100vw)`;
  return m;
}

// ------------------------------------------------------------------ menu
let openMenu = null, menuRestore = null, menuCleanup = null;
/** Where an anchored menu / popover goes: below the anchor when it fits, else on the side with more room, capped
    to that room (it scrolls) — so it never covers its own anchor. Only when neither side has PIN_MIN px is it
    pinned to the viewport. cssMax is the panel's own CSS max-height (a .filter-pop caps itself), which an inline
    height must never raise. Pure, for tests: returns {x, y, maxHeight}. */
const PIN_MIN = 160;
export function placement(r, mw, mh, vw, vh, cssMax = Infinity) {
  const below = vh - r.bottom - 12, above = r.top - 12;
  const cap = (room) => Math.max(0, Math.min(cssMax, room));
  let y, maxHeight;
  if (mh <= below || (below >= above && below >= PIN_MIN)) { maxHeight = cap(below); y = r.bottom + 4; }
  else if (above >= PIN_MIN) { maxHeight = cap(above); y = r.top - 4 - Math.min(mh, maxHeight); }
  else { maxHeight = cap(Math.max(120, vh - 16)); y = Math.max(8, vh - 8 - Math.min(mh, maxHeight)); }
  let x = r.left;
  if (x + mw > vw - 8) x = Math.max(8, r.right - mw);
  return { x: Math.max(8, x), y: Math.round(y), maxHeight: Math.floor(maxHeight) };
}
function place(el, anchor) {
  const r = anchor instanceof Element ? anchor.getBoundingClientRect() : { left: anchor.x, right: anchor.x, top: anchor.y, bottom: anchor.y };
  el.style.maxHeight = '';                 // measure at the stylesheet's own cap
  const cs = parseFloat(getComputedStyle(el).maxHeight);
  const { x, y, maxHeight } = placement(r, el.offsetWidth, el.offsetHeight, innerWidth, innerHeight, Number.isFinite(cs) ? cs : Infinity);
  el.style.maxHeight = `${maxHeight}px`;
  el.style.left = x + 'px'; el.style.top = y + 'px';
}
export function menu(anchor, items) {
  closeMenu();
  hideTip();
  menuRestore = document.activeElement;
  const el = h('div', { class: 'menu', role: 'menu' });
  for (const it of items) {
    if (!it) continue;
    if (it.sep) { el.appendChild(h('div', { class: 'menu-sep', role: 'separator' })); continue; }
    if (it.label && !it.onClick && it.header) { el.appendChild(h('div', { class: 'menu-label' }, it.label)); continue; }
    el.appendChild(h('button', { class: `menu-item ${it.danger ? 'danger' : ''} ${it.disabled ? 'disabled' : ''}`, role: 'menuitem', tabindex: '-1', disabled: !!it.disabled, title: it.title || null,
      onClick: (e) => { e.stopPropagation(); closeMenu(true); it.onClick && it.onClick(e); } },
      it.swatch ? h('span', { class: 'swatch', style: `background:${it.swatch}` }) : it.icon ? icon(it.icon, 14) : h('span', { class: 'ico' }), h('span', { class: 'menu-text' }, it.label), it.kbd ? kbd(it.kbd) : null));
  }
  // Arrow keys move between items; Tab leaves. Handled here so views listening on document do not also react.
  el.addEventListener('keydown', (e) => {
    const list = [...el.querySelectorAll('.menu-item:not([disabled])')];
    if (!list.length) return;
    const i = list.indexOf(document.activeElement);
    const go = (n) => { e.preventDefault(); e.stopPropagation(); const t = list[(n + list.length) % list.length]; t.focus(); t.scrollIntoView({ block: 'nearest' }); };
    if (e.key === 'ArrowDown') go(i + 1);
    else if (e.key === 'ArrowUp') go(i < 0 ? -1 : i - 1);
    else if (e.key === 'Home') go(0);
    else if (e.key === 'End') go(-1);
    else if (e.key === 'Tab') { e.preventDefault(); e.stopPropagation(); closeMenu(true); }
    else if (e.key !== 'Escape' && e.key !== 'Enter' && e.key !== ' ') e.stopPropagation();
  });
  document.body.appendChild(el);
  place(el, anchor);
  openMenu = el;
  const firstItem = el.querySelector('.menu-item:not([disabled])');
  if (firstItem) firstItem.focus({ preventScroll: true });
  const outside = (e) => { if (!el.contains(e.target)) closeMenu(); };
  setTimeout(() => { if (openMenu === el) document.addEventListener('mousedown', outside, true); }, 0);
  menuCleanup = () => document.removeEventListener('mousedown', outside, true);
  return el;
}
/** restoreFocus: return focus to what opened the menu (keyboard dismissal or choosing an item). */
export function closeMenu(restoreFocus = false) {
  if (menuCleanup) { menuCleanup(); menuCleanup = null; }
  if (!openMenu) return;
  const hadFocus = openMenu.contains(document.activeElement);
  const m = openMenu; openMenu = null;
  m.remove();
  if (m._onClose) { try { m._onClose(); } catch (e) { console.error(e); } }
  const back = menuRestore; menuRestore = null;
  if ((restoreFocus || hadFocus) && back && back.isConnected && back !== document.body && typeof back.focus === 'function') {
    try { back.focus({ preventScroll: true }); } catch (e) { /* gone */ }
  }
}
export const hasMenu = () => !!openMenu;
// An Escape that closes a menu is consumed: views listening on document must not also act on it
// (e.g. Explore clearing the whole selection when the user only meant to dismiss a context menu).
document.addEventListener('keydown', (e) => { if (e.key === 'Escape' && openMenu && !e.isComposing) { closeMenu(true); e.preventDefault(); e.stopImmediatePropagation(); } }, true);
window.addEventListener('resize', () => closeMenu());

// ------------------------------------------------------------------ popover (anchored panel that stays open)
export function popover(anchor, content, { width, onClose } = {}) {
  closeMenu();
  hideTip();
  menuRestore = document.activeElement;
  const el = h('div', { class: 'menu filter-pop', style: width ? `width:${width}px` : '' }, content);
  document.body.appendChild(el);
  place(el, anchor);
  el._onClose = onClose || null;
  openMenu = el;
  const outside = (e) => { if (!el.contains(e.target) && !(anchor instanceof Element && anchor.contains(e.target))) closeMenu(); };
  setTimeout(() => { if (openMenu === el) document.addEventListener('mousedown', outside, true); }, 0);
  menuCleanup = () => document.removeEventListener('mousedown', outside, true);
  return { el, close: () => { if (openMenu === el) closeMenu(true); }, reposition: () => place(el, anchor) };
}

// ------------------------------------------------------------------ tooltips via [data-tip]
let tipEl, tipTimer, tipWatch, tipTarget = null;
export function hideTip() { clearTimeout(tipTimer); clearInterval(tipWatch); tipTarget = null; if (tipEl) tipEl.remove(); }
document.addEventListener('mouseover', (e) => {
  const t = e.target.closest && e.target.closest('[data-tip]');
  if (!t || t === tipTarget) return;
  hideTip();
  tipTarget = t;
  tipTimer = setTimeout(() => {
    if (!t.isConnected || !t.dataset.tip) return;
    tipEl = tipEl || h('div', { class: 'tip', role: 'tooltip' });
    tipEl.textContent = t.dataset.tip; document.body.appendChild(tipEl);
    const r = t.getBoundingClientRect();
    let x = r.left + r.width / 2 - tipEl.offsetWidth / 2, y = r.bottom + 6;
    x = Math.max(6, Math.min(innerWidth - tipEl.offsetWidth - 6, x));
    if (y + tipEl.offsetHeight > innerHeight - 6) y = r.top - tipEl.offsetHeight - 6;
    tipEl.style.left = x + 'px'; tipEl.style.top = y + 'px';
    // A target removed from the page never fires mouseout: drop the tip with it.
    tipWatch = setInterval(() => { if (!t.isConnected || !t.matches(':hover')) hideTip(); }, 400);
  }, 450);
});
document.addEventListener('mouseout', (e) => { const t = e.target.closest && e.target.closest('[data-tip]'); if (t && t === tipTarget && !t.contains(e.relatedTarget)) hideTip(); });
document.addEventListener('mousedown', () => hideTip());
document.addEventListener('keydown', () => { if (tipEl && tipEl.isConnected) hideTip(); }, true);

// Keyboard activation for non-native controls (rows, tiles, cards): Enter (and Space for buttons) clicks them.
document.addEventListener('keydown', (e) => {
  if (e.defaultPrevented || e.altKey || e.ctrlKey || e.metaKey || e.isComposing) return;
  const t = e.target;
  if (!t || !t.matches || t.matches('button, a[href], input, select, textarea, [contenteditable]')) return;
  const role = t.getAttribute('role');
  if (!(role === 'button' || role === 'link' || role === 'option' || role === 'tab' || role === 'menuitem' || t.hasAttribute('data-activate'))) return;
  if (e.key === 'Enter' || (e.key === ' ' && role !== 'link')) { e.preventDefault(); t.click(); }
});

// ------------------------------------------------------------------ small bits
export function kbd(k) { return h('span', { class: 'kbd' }, k); }
/** Elapsed-time readout for long operations. With no time shown a wait feels longer and less certain, and a
    countdown frustrates more than elapsed time (docs/visual-comfort.md). Ticks once a second, with tabular
    digits so nothing shifts, and stops by itself once the element leaves the page. */
export function elapsedClock(startedAtMs = Date.now()) {
  const el = h('span', { class: 'elapsed', role: 'timer', 'aria-live': 'off' });
  const render = () => { el.textContent = fmt.clock((Date.now() - startedAtMs) / 1000); };
  render();
  let attached = false;
  const born = Date.now();
  const t = setInterval(() => {
    // Stops once it has left the page (or if it was never shown at all).
    if (!el.isConnected) { if (attached || Date.now() - born > 15000) { clearInterval(t); return; } } else attached = true;
    render();
  }, 1000);
  el.stop = () => clearInterval(t);
  return el;
}
export function spinner() { return h('span', { class: 'spinner', role: 'progressbar', 'aria-label': 'Loading' }); }
export function chip(text, { type = '', icon: ic, onClick, onRemove, cls = '', title } = {}) {
  return h('span', { class: `chip ${type} ${onClick ? 'clickable' : ''} ${cls}`, onClick, title: title || null, role: onClick ? 'button' : null, tabindex: onClick ? '0' : null }, ic ? icon(ic, 12) : null, text,
    onRemove ? h('span', { class: 'x', role: 'button', tabindex: '0', 'aria-label': 'Remove', onClick: (e) => { e.stopPropagation(); onRemove(); } }, icon('x', 11)) : null);
}
export function emptyState({ icon: ic = 'images', title, message, action }) {
  return h('div', { class: 'empty' }, icon(ic, 36), h('h3', title), message ? h('p', message) : null, action || null);
}
export function progress(pct, { indeterminate = false, cls = '' } = {}) {
  return h('div', { class: `progress ${indeterminate ? 'indeterminate' : ''} ${cls}`, role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100', 'aria-valuenow': indeterminate ? null : String(Math.round(pct * 100)) },
    h('i', { style: `width:${indeterminate ? 40 : Math.round(pct * 100)}%` }));
}
/** Middle ellipsis: the head shrinks with "…", the tail (frame number, last folder) always stays visible. */
export function midText(text, { tail = 14, cls = '', title } = {}) {
  const [a, b] = splitTail(text, tail);
  return h('span', { class: `mid ${cls}`, title: title === undefined ? String(text ?? '') : title }, h('span', { class: 'h' }, a), b ? h('span', { class: 't' }, b) : null);
}
/** A path shown with its distinguishing end visible (dataset folder, file name); '~' for the home folder. */
export function pathText(path, { home = window.FOVEA && window.FOVEA.home, cls = 'mono', tail = 28 } = {}) {
  const p = String(path ?? '');
  const shown = home && p.startsWith(home + '/') ? '~' + p.slice(home.length) : p;
  return midText(shown, { tail, cls, title: p });
}
export async function copyText(text, label = 'Copied') {
  // navigator.clipboard only exists in secure contexts: opened over plain http on a LAN address (a common
  // Docker setup) it is undefined, and an unfocused tab rejects it. Fall back rather than fail.
  try { await navigator.clipboard.writeText(text); toast(label, { type: 'ok', timeout: 1500 }); return true; }
  catch (e) { /* try the legacy path */ }
  const ta = h('textarea', { style: 'position:fixed;top:-1000px;left:0;opacity:0', readonly: true });
  ta.value = text; document.body.appendChild(ta); ta.select();
  let ok = false;
  try { ok = document.execCommand('copy'); } catch (e) { ok = false; }
  ta.remove();
  if (ok) { toast(label, { type: 'ok', timeout: 1500 }); return true; }
  // Last resort: hand the text over, pre-selected, so one keystroke copies it.
  const lines = text.split('\n').length;
  modal({
    title: 'Copy manually', size: 'lg',
    body: (api) => {
      const area = h('textarea', { class: 'input mono', rows: Math.min(14, Math.max(3, lines)), readonly: true, style: 'width:100%;white-space:pre' });
      area.value = text;
      setTimeout(() => { area.focus(); area.select(); }, 0);
      return h('div', { class: 'col gap-8' }, h('div', { class: 'small muted' }, `The browser blocked clipboard access. The text is selected — press ${navigator.platform.includes('Mac') ? '⌘' : 'Ctrl+'}C.`), area);
    },
    footer: (api) => h('button', { class: 'btn btn-primary', onClick: () => api.close() }, 'Done'),
  });
  return false;
}
export function seg(options, value, onChange, { size, label } = {}) {
  const el = h('div', { class: `seg ${size === 'sm' ? 'seg-sm' : ''}`, role: 'group', 'aria-label': label || null });
  const render = (v) => {
    clear(el);
    for (const o of options) {
      el.appendChild(h('button', { class: o.value === v ? 'active' : '', type: 'button', 'aria-pressed': String(o.value === v), disabled: !!o.disabled,
        onClick: () => { if (o.value !== v) { v = o.value; render(v); onChange(v); } }, 'data-tip': o.tip || null },
        o.icon ? icon(o.icon, 13) : null, o.label != null ? h('span', o.label) : null));
    }
  };
  render(value);
  el.setValue = (v) => render(v);
  return el;
}
export function switchBtn(checked, onChange, { label } = {}) {
  const b = h('button', { class: 'switch', role: 'switch', 'aria-checked': String(!!checked), type: 'button', 'aria-label': label || null });
  b.addEventListener('click', () => { const v = b.getAttribute('aria-checked') !== 'true'; b.setAttribute('aria-checked', String(v)); onChange(v); });
  b.setValue = (v) => b.setAttribute('aria-checked', String(!!v));
  return b;
}
export function statTile({ label, value, sub, icon: ic, onClick, tone, tip }) {
  return h('div', { class: `stat ${onClick ? 'clickable' : ''}`, onClick, role: onClick ? 'link' : null, tabindex: onClick ? '0' : null, 'data-tip': tip || null },
    // a bare text node in the flex row would be clipped mid-word with no ellipsis: wrap it so it can truncate
    h('div', { class: 'k' }, ic ? icon(ic, 13) : null, typeof label === 'string' ? h('span', { class: 'truncate', title: label }, label) : label),
    h('div', { class: `v ${tone ? `tone-${tone}` : ''}` }, value), sub ? h('div', { class: 's', title: typeof sub === 'string' ? sub : null }, sub) : null);
}
export function shortcutsModal(groups) {
  modal({ title: 'Keyboard shortcuts', size: 'lg', className: 'shortcuts-modal', body: h('div', { class: 'col gap-16' }, groups.filter(g => g.items && g.items.length).map(g => h('section', null, h('h3', { class: 'small strong muted mb-8 sc-title' }, g.title),
    h('div', { class: 'shortcut-grid' }, g.items.map(([keys, desc]) => h('div', { class: 'shortcut-row' }, h('span', desc), h('span', { class: 'keys' }, String(keys).split(' ').filter(Boolean).map(k => kbd(k)))))))) ) });
}
export { isTyping };
