// Class colours.
//
// The default palette was chosen for box outlines over photographs: every colour has CIELAB lightness 55–80
// and chroma 45–90 (visible on dark and bright frames), black or white text reaches ≥ 4.8:1 on it, and the
// colours were picked greedily for the largest CIEDE2000 distance — ids 0–9 (the digit hotkeys) are ≥ 20 apart,
// all 20 ≥ 13 apart (the old palette had pairs at 8). Ids beyond the palette get golden-angle hues instead of
// repeating class 0's colour.
//
// "cvd" is for colour-vision deficiency: Okabe–Ito order first (safe for protan/deutan/tritan), then Paul Tol's
// muted set. Chosen per device in Settings → Visual comfort.
export const PALETTES = {
  default: ['#f04b4b', '#00e15a', '#00b4ff', '#ffb400', '#b469ff', '#879600', '#ff96c3', '#00d2c3', '#b4783c', '#2d965a',
    '#5a87ff', '#ff0096', '#ffa578', '#b4d200', '#e1a5ff', '#f05a00', '#b4963c', '#ff5ae1', '#a5a5ff', '#78b45a'],
  cvd: ['#e69f00', '#56b4e9', '#009e73', '#f0e442', '#0072b2', '#d55e00', '#cc79a7', '#999999', '#882255', '#44aa99',
    '#ddcc77', '#332288', '#aa4499', '#117733', '#88ccee', '#cc6677', '#999933', '#dddddd', '#6699cc', '#ee8866'],
};
let paletteName = 'default';
/** The active palette (mutated in place when the palette changes, so importers keep a live reference). */
export const CLASS_COLORS = [...PALETTES.default];
const extra = new Map();

function hslHex(hDeg, s, l) {
  const k = (n) => (n + hDeg / 30) % 12, a = s * Math.min(l, 1 - l);
  const f = (n) => Math.round(255 * (l - a * Math.max(-1, Math.min(k(n) - 3, Math.min(9 - k(n), 1)))));
  return '#' + [f(0), f(8), f(4)].map(v => v.toString(16).padStart(2, '0')).join('');
}
export function classColor(cls) {
  const i = Math.abs(Math.trunc(Number(cls) || 0));
  if (i < CLASS_COLORS.length) return CLASS_COLORS[i];
  if (!extra.has(i)) extra.set(i, hslHex((i * 137.508) % 360, 0.72, i % 2 ? 0.6 : 0.52));
  return extra.get(i);
}
export function setPalette(name) {
  const p = PALETTES[name] ? name : 'default';
  paletteName = p;
  CLASS_COLORS.splice(0, CLASS_COLORS.length, ...PALETTES[p]);
  return p;
}
export const palette = () => paletteName;

export function withAlpha(hex, a) {
  const r = parseInt(hex.slice(1, 3), 16), g = parseInt(hex.slice(3, 5), 16), b = parseInt(hex.slice(5, 7), 16);
  return `rgba(${r},${g},${b},${a})`;
}
export const REVIEW = {
  approved: { label: 'Approved', color: 'var(--approve)', icon: 'check', key: 'A' },
  flagged: { label: 'Flagged', color: 'var(--flag)', icon: 'flag', key: 'F' },
  excluded: { label: 'Excluded', color: 'var(--exclude)', icon: 'ban', key: 'X' },
};
export function className(names, cls) { const n = names && names[cls]; return n != null && n !== '' ? n : `class_${cls}`; }

/** Readable text colour on a solid background (WCAG relative luminance): '#000' or '#fff', whichever contrasts more. */
export function textOn(hex) {
  if (typeof hex !== 'string' || !/^#[0-9a-f]{6}$/i.test(hex)) return '#fff';
  const ch = (i) => { const v = parseInt(hex.slice(i, i + 2), 16) / 255; return v <= 0.03928 ? v / 12.92 : ((v + 0.055) / 1.055) ** 2.4; };
  const L = 0.2126 * ch(1) + 0.7152 * ch(3) + 0.0722 * ch(5);
  return (L + 0.05) / 0.05 >= 1.05 / (L + 0.05) ? '#000' : '#fff';
}

/** Splits are shown as lightness steps of whatever colour is drawn (train strongest), never as hues of their
    own — a split hue would read as a class colour. */
export const SPLIT_ALPHA = [1, 0.62, 0.38, 0.22, 0.13];
export const splitAlpha = (index) => SPLIT_ALPHA[Math.min(Math.max(0, index), SPLIT_ALPHA.length - 1)];
