// Keyboard shortcuts that survive Caps Lock and non-Latin input sources (e.g. the Korean 2-set layout, where
// e.key is a jamo like 'ㅍ' for V). Letters, digits and symbols are matched by PHYSICAL key (e.code), never by
// e.key or by letter case; Shift is read from e.shiftKey only.
//
// Usage: keys.is(e, 'v'), keys.is(e, 'shift+a'), keys.is(e, 'mod+z'), keys.is(e, 'shift+Enter'), keys.is(e, '`')
// Help: views call registerKeys(scope, title, [[keys, description], ...]) once at install; the ? modal and any
// on-screen hint line read keyGroups(), so documentation cannot drift from the bindings.

import { isMac, isEditable } from './dom.js';

const SYMBOL_CODES = {
  '`': 'Backquote', '/': 'Slash', '[': 'BracketLeft', ']': 'BracketRight', '{': 'BracketLeft', '}': 'BracketRight',
  '=': 'Equal', '+': 'Equal', '-': 'Minus', ';': 'Semicolon', "'": 'Quote', ',': 'Comma', '.': 'Period', '\\': 'Backslash',
};
const SHIFTED_SYMBOLS = new Set(['{', '}', '+']);

/** 'a'..'z' for a letter key (by physical position), else ''. */
export function letter(e) {
  const c = e.code || '';
  if (c.length === 4 && c.startsWith('Key')) return c[3].toLowerCase();
  return !c && /^[a-z]$/i.test(e.key || '') ? e.key.toLowerCase() : '';   // synthetic events without a code
}

/** '0'..'9' for the digit row or the numeric keypad, else ''. A keypad key counts only while it types a digit:
    with NumLock off the same codes arrive as ArrowUp, PageUp, Insert… and must not pick a class. */
export function digit(e) {
  const c = e.code || '';
  if (/^Digit\d$/.test(c)) return c[5];
  if (/^Numpad\d$/.test(c)) return /^\d$/.test(e.key || '') ? c[6] : '';
  return !c && /^\d$/.test(e.key || '') ? e.key : '';
}

/** No Cmd / Ctrl / Alt held (Shift allowed). */
export const plain = (e) => !e.metaKey && !e.ctrlKey && !e.altKey;
/** Cmd on macOS, Ctrl elsewhere. */
export const mod = (e) => (isMac ? e.metaKey : e.ctrlKey);

/**
 * Match a key event against a spec: modifiers joined with '+', then one key.
 * Modifiers: shift, alt, mod (⌘ on Mac / Ctrl elsewhere), ctrl, meta. Unlisted modifiers must NOT be held
 * (so 'a' does not fire for ⌘A), except that Shift is implied by '{', '}' and '+'.
 * Keys: a letter, a digit, a symbol from SYMBOL_CODES, or a KeyboardEvent.key name (Enter, Escape, Delete,
 * Backspace, Tab, ArrowLeft, Home, End, PageUp, ' ' ...).
 */
export function is(e, spec) {
  const parts = spec.split('+');
  let key = parts.pop();
  if (key === '' && spec.endsWith('+')) key = '+';            // 'shift++' style
  const want = new Set(parts.map(p => p.toLowerCase()));
  if (SHIFTED_SYMBOLS.has(key)) want.add('shift');
  const needMod = want.has('mod');
  const meta = want.has('meta') || (needMod && isMac), ctrl = want.has('ctrl') || (needMod && !isMac);
  if (!!e.shiftKey !== want.has('shift') || !!e.altKey !== want.has('alt') || !!e.metaKey !== meta || !!e.ctrlKey !== ctrl) return false;
  if (/^[a-z]$/i.test(key)) return letter(e) === key.toLowerCase();
  if (/^\d$/.test(key)) return digit(e) === key;
  if (SYMBOL_CODES[key]) return e.code ? e.code === SYMBOL_CODES[key] : e.key === key;
  if (key === 'Space' || key === ' ') return e.code === 'Space' || e.key === ' ';
  if (key === 'Delete') return e.key === 'Delete' || e.key === 'Backspace';   // both keys delete on macOS keyboards
  return e.key === key;
}

/** Ignore auto-repeat for one-shot actions (review, accept): holding a key must not act on a run of images. */
export const fresh = (e) => !e.repeat;

/** The event goes into a text field (letters must type there, not trigger shortcuts). Sliders, checkboxes and
    selects are not text fields. */
export const inField = (e) => isEditable(e && e.target);

/** '?' (Shift + the slash key on most layouts, whatever the input source). */
export const isHelp = (e) => !e.metaKey && !e.ctrlKey && !e.altKey && (e.key === '?' || (e.shiftKey && e.code === 'Slash'));

// ------------------------------------------------------------------ help registry
// The ? dialog is generated from these groups, so it cannot drift from the bindings. The shell registers the
// 'global' group (order 0); each view registers its own at install.
const scopes = new Map();
/** Register (or replace) the documented bindings of one view. items: [[keys, description], ...] */
export function registerKeys(scope, title, items, order = 50) { scopes.set(scope, { scope, title, items, order }); }
/** The group of the view in front first, then the global one, then everything else. */
export function keyGroupsFor(activeScopes = []) {
  const all = keyGroups();
  const rank = (g) => { const i = activeScopes.indexOf(g.scope); return i >= 0 ? i : g.scope === 'global' ? activeScopes.length : activeScopes.length + 1 + g.order / 1000; };
  return all.slice().sort((a, b) => rank(a) - rank(b));
}
/** All registered groups, in display order. */
export function keyGroups() { return [...scopes.values()].sort((a, b) => a.order - b.order); }
/** The items of one scope (for compact hint lines). */
export function keysOf(scope) { return (scopes.get(scope) || {}).items || []; }
