// Annotations converted from another format (VisDrone, KITTI, DOTA, VOC, COCO …): the per-box attributes they carry
// (occlusion, truncation …) and their regions that are not objects (ignored regions, DontCare, crowds). The schema —
// attribute and value names from the dataset's format card — is read once per dataset (GET …/annotations).

export const EMPTY = Object.freeze({ attributes: [], regions: {}, formats: [], counts: {}, region_counts: {} });
const cache = new Map();

/** Has the dataset annotations converted from (or waiting to be converted from) another format? */
export const hasFormats = (ds) => !!(ds && ds.layout && ds.layout.formats && Object.keys(ds.layout.formats).length);

export function annSchema(api, ds) {
  if (!hasFormats(ds)) return Promise.resolve(EMPTY);
  if (!cache.has(ds.id)) cache.set(ds.id, api.get(`/api/datasets/${encodeURIComponent(ds.id)}/annotations`).catch(() => { cache.delete(ds.id); return EMPTY; }));
  return cache.get(ds.id);
}
export function forgetSchema(dsId) { cache.delete(dsId); }

const find = (schema, name) => ((schema && schema.attributes) || []).find((a) => a.name === name);
export const attrLabel = (schema, name) => (find(schema, name) || {}).label || String(name).replace(/_/g, ' ');
export function valueLabel(schema, name, v) {
  const a = find(schema, name);
  return (a && a.values && a.values[String(v)]) || String(v);
}
export const regionLabel = (schema, kind) => ((schema && schema.regions && schema.regions[kind]) || {}).label || String(kind);

/** "Occlusion: partial (1–50%) · Truncation: none (0%)" for one box's attributes. */
export function attrText(schema, attrs) {
  return Object.entries(attrs || {}).map(([k, v]) => `${attrLabel(schema, k)}: ${valueLabel(schema, k, v)}`).join(' · ');
}

/** Pairs of an `attr` filter value ("occlusion:2,occlusion:1,region:ignored") → [[name, value], …]. */
export function attrPairs(csv) {
  return String(csv || '').split(',').map((t) => t.trim()).filter(Boolean).map((t) => {
    const i = t.indexOf(':');
    return i > 0 ? [t.slice(0, i), t.slice(i + 1)] : null;
  }).filter(Boolean);
}
/** Chip text of an `attr` filter: "Occlusion: heavy (50–100%) or partial · Ignored region". */
export function attrFilterText(schema, csv) {
  const by = new Map();
  for (const [n, v] of attrPairs(csv)) { if (!by.has(n)) by.set(n, []); by.get(n).push(v); }
  return [...by].map(([n, vs]) => (n === 'region' ? vs.map((k) => `Has ${regionLabel(schema, k).toLowerCase()}`).join(' or ')
    : `${attrLabel(schema, n)}: ${vs.map((v) => valueLabel(schema, n, v)).join(' or ')}`)).join(' · ');
}
/** Toggle one name:value token in an `attr` filter value. */
export function toggleAttr(csv, token) {
  const toks = String(csv || '').split(',').map((t) => t.trim()).filter(Boolean);
  const i = toks.indexOf(token);
  if (i >= 0) toks.splice(i, 1); else toks.push(token);
  return toks.join(',');
}
