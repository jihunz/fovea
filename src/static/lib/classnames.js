// Class names pasted as text: JSON (a list, an {id: name} map, a {name: id} map, {"names": …}, COCO "categories"),
// a data.yaml names block, a Python list, or plain names one per line / comma-separated. Pure — no DOM.

const MAX_CLASSES = 10000;
const KEYS = ['names', 'classes', 'class_names', 'categories', 'labels'];

const clean = (v) => String(v == null ? '' : v).replace(/\s+/g, ' ').trim();
const isId = (k) => /^\d+$/.test(String(k).trim());

/** {id: name} → a dense list; gaps become class_<id> (a sparse map must not build millions of names). */
function fromIdMap(entries) {
  const ids = entries.map(([k]) => Number(k));
  const top = Math.max(-1, ...ids);
  if (top >= MAX_CLASSES) throw new Error(`Class id ${top} is too large (at most ${MAX_CLASSES - 1})`);
  const out = Array.from({ length: top + 1 }, (_, i) => `class_${i}`);
  for (const [k, v] of entries) out[Number(k)] = clean(v) || `class_${k}`;
  return out;
}

/** Names out of a parsed JSON / YAML value, or null when it does not look like a class list. */
function fromValue(v) {
  if (Array.isArray(v)) {
    if (!v.length) return [];
    if (v.every((x) => typeof x === 'string' || typeof x === 'number')) return v.map(clean);
    // [{id, name}, …] (COCO categories, Label Studio, CVAT): placed by id when every item has one, else in order.
    if (v.every((x) => x && typeof x === 'object' && ('name' in x || 'label' in x))) {
      const nm = (x) => clean(x.name != null ? x.name : x.label);
      if (v.every((x) => isId(x.id))) {
        const ids = v.map((x) => Number(x.id));
        // COCO counts from 1: shift so the first category is class 0 (YOLO ids start at 0).
        const base = Math.min(...ids) === 1 && !ids.includes(0) ? 1 : 0;
        return fromIdMap(v.map((x) => [Number(x.id) - base, nm(x)]));
      }
      return v.map(nm);
    }
    return null;
  }
  if (v && typeof v === 'object') {
    for (const k of KEYS) if (k in v) { const r = fromValue(v[k]); if (r) return r; }
    const ents = Object.entries(v);
    if (!ents.length) return [];
    if (ents.every(([k]) => isId(k))) return fromIdMap(ents);                 // {"0": "person", "1": "car"}
    if (ents.every(([, x]) => isId(x))) return fromIdMap(ents.map(([k, x]) => [Number(x), k]));   // {"person": 0}
  }
  return null;
}

/** A data.yaml names block: "names:\n  0: person\n  1: car", "names: [person, car]" or "- person" lines. */
function fromYaml(text) {
  const body = text.replace(/^\s*(names|classes)\s*:\s*/m, '');
  const inline = /^\s*\[([\s\S]*)\]\s*$/.exec(body);
  if (inline) return inline[1].split(',').map((s) => clean(s.replace(/^\s*['"]|['"]\s*$/g, ''))).filter(Boolean);
  const lines = body.split(/\r?\n/).map((l) => l.replace(/\s+#.*$/, '').trim()).filter(Boolean);
  if (lines.length && lines.every((l) => /^\d+\s*:/.test(l))) {
    return fromIdMap(lines.map((l) => { const m = /^(\d+)\s*:\s*(.*)$/.exec(l); return [m[1], m[2].replace(/^['"]|['"]$/g, '')]; }));
  }
  if (lines.length && lines.every((l) => /^-\s*/.test(l))) return lines.map((l) => clean(l.replace(/^-\s*/, '').replace(/^['"]|['"]$/g, '')));
  return null;
}

/**
 * Parse pasted class names. Returns {names, format} — format 'json' | 'yaml' | 'lines' | 'commas' — or
 * {names: [], error} when the text is JSON-shaped but not a class list.
 */
export function parseClassNames(text) {
  const t = String(text || '').trim();
  if (!t) return { names: [], format: '' };
  let names = null, format = '';
  try {
    if (/^[[{]/.test(t)) {
      let v;
      try { v = JSON.parse(t); }
      catch (e) {
        // A Python list / dict ({0: 'person'}, ['a', 'b']) is not JSON: quote it the JSON way and try again.
        v = JSON.parse(t.replace(/'([^'\\]*(?:\\.[^'\\]*)*)'/g, (_m, s) => JSON.stringify(s)).replace(/([{,]\s*)(\d+)\s*:/g, '$1"$2":').replace(/,\s*([}\]])/g, '$1'));
      }
      names = fromValue(v);
      if (!names) return { names: [], format: 'json', error: 'This JSON is not a list of class names — use ["person", "car"], {"0": "person"} or {"names": […]}' };
      format = 'json';
    } else if (/^\s*(names|classes)\s*:/m.test(t) || /^\s*(\d+\s*:|-\s)/.test(t)) {
      names = fromYaml(t);
      format = names ? 'yaml' : '';
    }
  } catch (e) {
    if (/^[[{]/.test(t)) return { names: [], format: 'json', error: `Not valid JSON: ${e.message}` };
  }
  if (!names) {
    const lines = t.split(/\r?\n/).map((l) => l.trim()).filter(Boolean);
    if (lines.length > 1 && lines.every((l) => /^\d+[\s:]+\S/.test(l))) {          // "0 person" / "0: person" lines
      names = fromIdMap(lines.map((l) => { const m = /^(\d+)[\s:]+(.*)$/.exec(l); return [m[1], m[2]]; }));
      format = 'lines';
    } else if (lines.length > 1) { names = lines; format = 'lines'; }
    else { names = t.split(/[,;\t]/).map((s) => s.trim()).filter(Boolean); format = names.length > 1 ? 'commas' : 'lines'; }
    names = names.map((s) => clean(s.replace(/^['"]|['"],?$/g, '')));
  }
  if (names.length > MAX_CLASSES) return { names: [], format, error: `${names.length} names — at most ${MAX_CLASSES}` };
  return { names: names.map((n, i) => n || `class_${i}`), format };
}

/** Duplicate names (after trimming) — a dataset's class names must be unique. */
export function duplicateNames(names) {
  const seen = new Set(), dup = new Set();
  for (const n of names) { const k = clean(n); if (seen.has(k)) dup.add(k); seen.add(k); }
  return [...dup];
}
