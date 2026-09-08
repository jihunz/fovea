export const CLASS_COLORS = [
  '#ef4444', '#3b82f6', '#f59e0b', '#10b981', '#8b5cf6', '#ec4899', '#06b6d4', '#f97316', '#84cc16', '#6366f1',
  '#14b8a6', '#e11d48', '#0ea5e9', '#a855f7', '#eab308', '#22c55e', '#f43f5e', '#2563eb', '#d946ef', '#64748b',
];
export function classColor(cls) { const i = Math.abs(Number(cls) || 0); return CLASS_COLORS[i % CLASS_COLORS.length]; }
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
