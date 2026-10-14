// Explore tab: a virtualized gallery over the whole filtered list, Inspect (zoom, notes, AI suggestions,
// similar images) and review actions with confirmation and undo.
//
// The grid renders only the rows near the viewport (plus about a screen of overscan) inside a spacer as tall as
// the whole list, so the scrollbar spans every image and the DOM stays a few hundred nodes whether the list holds
// 100 images or 45,000. Data comes from the shared ImageCursor (a frozen working set: index i keeps naming the
// same image while reviews change who matches the filter), fetched page by page for the rows on screen.
import { ISSUE_LABELS as BASE_ISSUE_LABELS } from './issues.js';
import { thumbPx, trackImage, createImageLoader, createFrameCache } from '../lib/media.js';

const FILTER_KEYS = ['split', 'cls', 'only_cls', 'labeled', 'review', 'exclude_review', 'issue', 'ai', 'q', 'seq', 'ids', 'set', 'min_boxes', 'max_boxes', 'has_label', 'seed'];
const TILE_PX = { s: 150, m: 210, l: 300, xl: 420 };
const SIZES = ['s', 'm', 'l', 'xl'];
const GAP = 10, META_H = 30, PAD_X = 16, PAD_TOP = 14, PAD_BOTTOM = 76;
const PAGE = 100;
const OVERSCAN = 1;               // screens of rows kept mounted above and below the viewport
const REVIEW_KEYS = { a: 'approved', f: 'flagged', x: 'excluded', u: null };
const PROPOSAL_KINDS = ['detect', 'prev_frame', 'interp', 'refine'];
const AUDIT_GLYPH = { missing: '+', loose: '⤢', swap: '⇄', spurious: '✕' };
const CONFIRM_OVER = 50;          // bulk actions on more images than this ask first, with the exact count
const RESOLVE_MAX = 20000;        // "all matching" is resolved to ids (exact counts, undo) up to this size
const URL_IDS_MAX = 1500;         // up to this many ids travel in the URL (?ids=); more, or "all matching", become a server-side set (?set=)
const FACET_DELAY = 250, FACET_DELAY_EDIT = 1200;   // ms: after a filter change / after reviews and label edits
const BAND_MIN = 4;               // px of pointer travel before a drag on empty grid space becomes a selection rectangle
const MIN_SWAP_MS = 334;          // WCAG 2.3.1: at most three full-frame changes a second across unrelated images
const SUSPECT_SCAN = 3000;        // J / ⇧J look this far ahead before giving up
const GHOST = '#f59e0b';          // Inspect: boxes another image puts here (a pressed QA flag chip; explore.css uses the same amber)
const IMAGE_FLAGS = new Set(['near_dup', 'leak_split', 'dup_same_role', 'blurry', 'dark', 'overexposed', 'empty_label']);
const FLAG_CODES = new Set(['near_dup', 'leak_split', 'dup_same_role', 'dup_label_mismatch', 'seq_gap', 'seq_flicker', 'seq_jump', 'seq_count_spike', 'shape_outlier', 'blurry', 'dark', 'overexposed']);
const AUDIT_CODES = new Set(['ai_missing', 'ai_loose', 'ai_swap', 'ai_spurious']);   // issue= codes of the label audit's findings
const VERB = { approved: 'Approve', flagged: 'Flag', excluded: 'Exclude' };
const SORT_SHORT = { flag: 'Most suspicious', audit: 'Most suspect', ai_conf: 'AI confidence', set: 'Selection' };   // the toolbar button's label
// 'flag' ranks by the score of the QA flags the issue filter names (the server's key), so the menu offers it only then.
const SORTS = [['id', 'Index'], ['name', 'Name'], ['boxes', 'Boxes'], ['size', 'Resolution'], ['mtime', 'Modified'], ['reviewed', 'Reviewed'], ['random', 'Random'], ['flag', 'Most suspicious first'], ['audit', 'Most suspect (AI audit)'], ['ai_conf', 'Lowest AI confidence']];
const LABELED = [['', 'Any label state'], ['1', 'Has boxes'], ['nobox', 'No boxes'], ['empty', 'Empty label file'], ['0', 'No label file']];
const REVIEWS = [['', 'Any review'], ['none', 'Unreviewed'], ['any', 'Reviewed'], ['approved', 'Approved'], ['flagged', 'Flagged'], ['excluded', 'Excluded']];
const AI_STATES = [['', 'Any AI state'], ['pending', 'Has AI proposals'], ['audit', 'Suspected label errors'], ['none', 'No AI results']];
const FOCUSABLE = 'a[href],button:not([disabled]),input:not([disabled]),select:not([disabled]),textarea:not([disabled]),[tabindex]:not([tabindex="-1"])';
const DUP_IOU = 0.7;              // a proposal over an existing box of its class is "already labeled" (as in Annotate and the accept job)
const REF_IOU = 0.9;              // an audit finding's box is still there: IoU with the GT box it stored (runner._ref_ok, Annotate's refByGeom)
const CHORD_MS = 1600;            // the shell's G-chord window (app.js), used only when the shell does not report its chord
const STATUS_MS = 5000;           // how long a grid status line stays
const EDIT_WINDOW = 30000;        // ms: label saves this recent may still be waiting for the QA re-check (see watchQa)
const QA_FIRST = 3400, QA_POLL = 1500, QA_TRIES = 12;   // the QA engine re-checks edited labels ~3 s after the last save
// What each QA flag says about single boxes (rows of the label file), for Inspect's row markers.
const ROW_FLAG_TEXT = { seq_jump: 'jumped', seq_flicker: 'class flips', shape_outlier: 'odd shape' };
const DETAIL_FLAGS = new Set(['seq_jump', 'seq_gap', 'seq_flicker', 'seq_count_spike', 'shape_outlier', 'dup_label_mismatch', 'leak_split']);

// Label saves made anywhere in the app — Annotate saves while Explore is not even mounted. The QA engine re-checks the
// label-based flags (sequence gaps and jumps, twins, box shapes) a few seconds after the last save, so flag counts
// read before then are about to change: an Explore opened meanwhile waits for that re-check (watchQa).
const recentEdits = [];           // {ds, id, at}, newest last
function noteLabelEdit(ds, id) {
  const now = Date.now();
  while (recentEdits.length && (recentEdits.length >= 40 || now - recentEdits[0].at > EDIT_WINDOW)) recentEdits.shift();
  recentEdits.push({ ds, id, at: now });
}

/** "3 boxes added · 2 already labeled" from an accept job's result. */
function acceptSummary(r) {
  if (!r || typeof r !== 'object') return '';
  const n = (k) => r[k] || 0, boxes = (k) => `${n(k)} box${n(k) === 1 ? '' : 'es'}`;
  return [n('added') && `${boxes('added')} added`, n('modified') && `${boxes('modified')} corrected`, n('deleted') && `${boxes('deleted')} removed`,
    n('duplicates') && `${n('duplicates')} already labeled`, n('merged') && `${n('merged')} loose cop${n('merged') === 1 ? 'y' : 'ies'} removed`,
    n('obsolete') && `${n('obsolete')} no longer appl${n('obsolete') === 1 ? 'ies' : 'y'} (label changed since)`, n('stale') && `${n('stale')} outdated`,
    n('conflicts') && `${n('conflicts')} ⇄ left to decide (over another class)`].filter(Boolean).join(' · ') || 'nothing to change';
}
export const defaultOrder = (sort) => (sort === 'audit' || sort === 'flag' ? 'desc' : 'asc');
const csv = (v) => (v == null || v === '' ? [] : String(v).split(',').map(s => s.trim()).filter(Boolean));
/** Whether "Most suspicious first" has something to rank by: the issue filter names a QA flag. */
export const flagSortable = (f) => csv(f && f.issue).some(c => FLAG_CODES.has(c));
/** The order a list opens in when its URL names none: a selection in its own order; images a QA flag picked out
    most suspicious first (the highest score among the flags the issue filter names, audit findings included); the
    label audit's suspects by their audit score; anything else by index. */
export function defaultSort(f) {
  f = f || {};
  if (f.set) return 'set';
  if (flagSortable(f)) return 'flag';
  if (csv(f.issue).some(c => AUDIT_CODES.has(c)) || f.ai === 'audit') return 'audit';
  return 'id';
}
/** The sort after the filters change from `prev` to `next`. A list in its default order takes the new filters'
    default (filtering by a QA flag ranks most suspicious first; clearing it goes back to the index), like a URL
    without a sort would; a sort or direction picked by hand stays, unless it needs a filter that is gone. */
export function nextSort(prev, next, sort, order) {
  const byDefault = sort === defaultSort(prev) && order === defaultOrder(sort);
  if (byDefault || (sort === 'set' && !(next && next.set)) || (sort === 'flag' && !flagSortable(next))) {
    const s = defaultSort(next);
    return { sort: s, order: defaultOrder(s) };
  }
  return { sort, order };
}
/** The sort and direction a link to another page (Annotate) carries for the list `f`: none for the plain order every
    page starts in (index, or a selection's own); otherwise both, so that page's own defaults never decide. */
export function linkOrder(f, sort, order) {
  const base = f && f.set ? 'set' : 'id';
  return sort === base && order === 'asc' && defaultSort(f) === base ? null : { sort, order };
}
const LOCAL_HOSTS = new Set(['localhost', '127.0.0.1', '[::1]', '::1']);
/** Whether Ask AI may use provider `p` from this browser. `local` is the server's verdict on this browser's requests
    (/api/meta server.request_is_local); only without one does the page's hostname stand in — behind Docker or on a LAN
    address the hostname says nothing about where the server sees the request come from. */
export function askUsable(P, p, local = null, hostname = '') {
  if (!p || !(p.capabilities && p.capabilities.generate)) return false;
  if (p.config && p.config.enabled === false) return false;
  if (p.needs_key && !(p.key && p.key.present)) return false;
  const here = typeof local === 'boolean' ? local : LOCAL_HOSTS.has(hostname) || String(hostname).endsWith('.localhost');
  const guarded = typeof p.guarded === 'boolean' ? p.guarded : p.kind === 'cloud';   // what the server's remote guard refuses
  if (guarded && P && P.remote_ai_allowed === false && !here) return false;
  const st = p.status || {};
  return st.available !== false && !st.checking && st.state !== 'checking';
}
// The server's view of this browser does not change while the page lives: asked once, again only after a failure.
let requestIsLocal = null;
function serverSeesLocal(api) {
  if (!requestIsLocal) {
    requestIsLocal = api.get('/api/meta').then(m => {
      const v = m && m.server ? m.server.request_is_local : null;
      return typeof v === 'boolean' ? v : null;
    }, () => { requestIsLocal = null; return null; });
  }
  return requestIsLocal;
}
const clamp = (v, a, b) => Math.max(a, Math.min(b, v));
const reviewStatus = (it) => (it && it.review && it.review.status) || null;
/** A note can exist without a review status (the server keeps notes on unreviewed images). */
const noteOf = (it) => (it && ((it.review && it.review.note) || it.note)) || '';
const ls = {
  get(k, d) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { localStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};
const ss = {
  get(k) { try { return sessionStorage.getItem(k); } catch (e) { return null; } },
  set(k, v) { try { sessionStorage.setItem(k, v); } catch (e) { /* private mode */ } },
};

/** Per-box issue codes, mirroring the label parser's per-row checks (core/labels.py), so Inspect can point at
    the box behind "Tiny box" or "Box outside image" instead of only naming the problem. */
export function boxIssues(boxes, nClasses) {
  const out = boxes.map(() => []);
  boxes.forEach((b, i) => {
    const [c, xc, yc, w, hh] = b;
    if (c < 0 || (nClasses > 0 && c >= nClasses)) out[i].push('bad_class');
    if (w <= 0 || hh <= 0) out[i].push('zero_area'); else if (w * hh < 1e-4) out[i].push('tiny_box');
    const eps = 0.005;
    if (!(xc - w / 2 >= -eps && xc + w / 2 <= 1 + eps && yc - hh / 2 >= -eps && yc + hh / 2 <= 1 + eps)) out[i].push('out_of_range');
  });
  for (let i = 0; i < boxes.length; i++) for (let j = i + 1; j < boxes.length; j++) {
    const a = boxes[i], b = boxes[j];
    if (a[0] !== b[0] || iou(a, b) < 0.95) continue;
    if (!out[i].includes('duplicate_box')) out[i].push('duplicate_box');
    if (!out[j].includes('duplicate_box')) out[j].push('duplicate_box');
  }
  return out;
}
function iou(a, b) {
  const ax0 = a[1] - a[3] / 2, ay0 = a[2] - a[4] / 2, ax1 = a[1] + a[3] / 2, ay1 = a[2] + a[4] / 2;
  const bx0 = b[1] - b[3] / 2, by0 = b[2] - b[4] / 2, bx1 = b[1] + b[3] / 2, by1 = b[2] + b[4] / 2;
  const iw = Math.max(0, Math.min(ax1, bx1) - Math.max(ax0, bx0)), ih = Math.max(0, Math.min(ay1, by1) - Math.max(ay0, by0));
  const inter = iw * ih, uni = a[3] * a[4] + b[3] * b[4] - inter;
  return uni > 0 ? inter / uni : 0;
}

/** Whether a list row still matches an `issue=` filter (CSV of codes), the way the server reads it: label-file
    issues (the informational missing_label aside for "any"), QA flags, and AI audit codes — known per image only as
    a total, so any pending finding counts. null when the row does not carry its issues and flags. */
export function issueMatch(it, issueCsv) {
  const want = String(issueCsv || '').split(',').map(s => s.trim()).filter(Boolean);
  if (!want.length) return true;
  if (!it || !Array.isArray(it.issues) || !Array.isArray(it.flags)) return null;
  const probs = it.issues.filter(c => c !== 'missing_label');
  return want.some(c => (c === 'any' ? probs.length > 0 || it.flags.length > 0
    : c.startsWith('ai_') ? (it.ai_suspect || 0) > 0 : it.issues.includes(c) || it.flags.includes(c)));
}

/** Rows of the label file (0-based) a QA flag ({code, detail}) names: the boxes that jumped, flicker or have an
    odd shape. Other flags are about the whole image (or a box that is missing). */
export function qaFlagRows(flag) {
  const d = (flag && flag.detail) || {};
  const rows = flag && flag.code === 'seq_jump' ? (d.jumps || []).map(j => j && j.row)
    : flag && (flag.code === 'seq_flicker' || flag.code === 'shape_outlier') ? (d.rows || []).map(x => x && x.row) : [];
  return [...new Set(rows.filter(r => Number.isInteger(r) && r >= 0))];
}

/** Where the labels of two near-identical images differ (a dup_label_mismatch flag). Boxes pair the way the QA
    engine compares them (core/qa.py labels_disagree: greedy one-to-one, same class, IoU >= minIou) and agree; of the
    rest, a pair at that IoU with another class is kind 'class', a same-class pair that still overlaps (IoU >= 0.1)
    is 'moved', and what is left is 'only' (on one side). `mine` lists rows of this image, `theirs` indexes of the
    other image's boxes; `other` names the partner (-1 for 'only'). */
export function twinDiff(mine, theirs, minIou = 0.5) {
  const a = Array.isArray(mine) ? mine : [], b = Array.isArray(theirs) ? theirs : [];
  const ua = new Set(a.keys()), ub = new Set(b.keys()), outA = [], outB = [];
  const pair = (sameClass, floor, kind) => {
    const cand = [];
    for (const i of ua) for (const j of ub) {
      if ((a[i][0] === b[j][0]) !== sameClass) continue;
      const v = iou(a[i], b[j]);
      if (v >= floor) cand.push([v, i, j]);
    }
    cand.sort((x, y) => y[0] - x[0] || y[1] - x[1] || y[2] - x[2]);   // ties as Python's sorted(reverse=True)
    for (const [, i, j] of cand) {
      if (!ua.has(i) || !ub.has(j)) continue;
      ua.delete(i); ub.delete(j);
      if (kind) { outA.push({ row: i, kind, other: j }); outB.push({ index: j, kind, other: i }); }
    }
  };
  pair(true, minIou, null);
  pair(false, minIou, 'class');
  pair(true, Math.min(0.1, minIou), 'moved');
  for (const i of ua) outA.push({ row: i, kind: 'only', other: -1 });
  for (const j of ub) outB.push({ index: j, kind: 'only', other: -1 });
  return { mine: outA.sort((x, y) => x.row - y.row), theirs: outB.sort((x, y) => x.index - y.index) };
}

/** A proposal ({cls, xc, yc, w, h, kind}) that repeats an existing box of its class (IoU >= 0.7) adds nothing:
    "already labeled", as Annotate and the accept job count it. Audit findings and refinements are never one. */
export function duplicateOf(p, boxes) {
  if (!p || p.kind in AUDIT_GLYPH || p.kind === 'refine') return false;
  return (boxes || []).some(b => b[0] === p.cls && iou(b, [p.cls, p.xc, p.yc, p.w, p.h]) >= DUP_IOU);
}

/** The box (0-based row) of ANOTHER class a proposal lands on (IoU >= 0.7), as Annotate marks it ("⇄ conflicts #n"):
    accepting it would put a second box on that object, so ⇧Enter leaves it to decide. -1: none, or the proposal
    repeats a box of its own class (already labeled), or it is an audit finding / a refinement. */
export function conflictOf(p, boxes) {
  if (!p || p.kind in AUDIT_GLYPH || p.kind === 'refine') return -1;
  const b = [p.cls, p.xc, p.yc, p.w, p.h]; let k = -1;
  for (const [i, x] of (boxes || []).entries()) {
    if (iou(x, b) < DUP_IOU) continue;
    if (x[0] === p.cls) return -1;
    k = i;
  }
  return k;
}

/** What ⇧Enter does with the shown proposals — Annotate's acceptAllVisible, and the accept job with skip_conflicts
    (runner.split_conflicts): most confident first, a proposal that lands (IoU >= 0.7) only on boxes of another class
    — one here or one this accept just placed — is left to decide (`left`); one over a box of its own class adds
    nothing (`dupes`: the job counts them "already labeled"); the rest add a box each (`take`). */
export function acceptPlan(list, boxes) {
  const placed = (boxes || []).map(b => b.slice(0, 5)), take = [], dupes = [], left = [];
  for (const p of [...(list || [])].sort((a, b) => (b.conf ?? -1) - (a.conf ?? -1))) {
    const b = [p.cls, p.xc, p.yc, p.w, p.h], hits = placed.filter(x => iou(x, b) >= DUP_IOU);
    if (hits.length && !hits.some(x => x[0] === p.cls)) left.push(p);
    else if (hits.length) dupes.push(p);
    else { placed.push(b); take.push(p); }
  }
  return { take, dupes, left };
}

/** The label-file row (0-based) an AI audit finding is about, in `boxes` as they are now. Rows shift when a box above
    is deleted, so the stored ref_row counts only while it still is the box the audit judged — IoU >= 0.9 with the
    stored GT box (ref_box), of the same class unless a class swap — else the row that holds that box now (best IoU,
    then the one nearest the stored row). Exactly what the accept job applies a fix to (runner._find_ref_row; Annotate's
    refByGeom finds the same box). A finding stored without ref_box keeps its row only while that row still fits the
    suggestion (runner._ref_ok). -1: not about an existing box (a missed object, a plain proposal); null: that box is
    gone or changed since the audit ("label changed since"). Without `boxes`, the stored row as it is. */
export function refRow(p, boxes) {
  if (!p || !(p.kind in AUDIT_GLYPH) || p.kind === 'missing') return -1;
  const at = Number.isInteger(p.ref_row) && p.ref_row >= 0 ? p.ref_row : null;
  if (!Array.isArray(boxes)) return at ?? -1;
  const g = Array.isArray(p.ref_box) && p.ref_box.length >= 4 ? p.ref_box.slice(-4).map(Number) : null;
  const cur = at != null ? boxes[at] : null;
  if (!g || !g.every(isFinite)) {
    const s = [p.cls, p.xc, p.yc, p.w, p.h];
    return cur && (p.kind === 'loose' ? cur[0] === p.cls && iou(cur, s) >= 0.25 : iou(cur, s) >= REF_IOU) ? at : null;
  }
  const want = [p.cls, ...g], ok = (b) => p.kind === 'swap' || b[0] === p.cls;
  if (cur && ok(cur) && iou(cur, want) >= REF_IOU) return at;
  let best = null, bv = 0, bd = 0;
  for (const [i, b] of boxes.entries()) {
    if (!ok(b)) continue;
    const v = iou(b, want), d = at != null ? Math.abs(i - at) : 0;
    if (v >= REF_IOU && (best == null || v > bv || (v === bv && d < bd))) { best = i; bv = v; bd = d; }
  }
  return best;
}
/** One AI suggestion as a short line: "0.82 person", "+ 0.61 person", "⤢ tighter #3 person", "⇄ #2 → bicycle?",
    "✕ #5 no support" — findings name the box (#N, as numbered in Inspect's Labels list and in Annotate): `row` is
    the one refRow(p, boxes) found; null, a finding whose box is gone, names none ("label changed since"). A
    proposal over a box of another class says so: "0.89 person ⇄#1" (`conflict`, from conflictOf). */
export function findingLabel(p, nameOf = (c) => `class_${c}`, { row = refRow(p), conflict = -1 } = {}) {
  const ref = row != null && row >= 0 ? `#${row + 1} ` : '', gone = row === null ? ' · label changed since' : '';
  if (p.kind === 'swap') return `⇄ ${ref}→ ${nameOf(p.cls)}?${gone}`;
  if (p.kind === 'spurious') return `✕ ${ref}no support${gone}`;
  if (p.kind === 'loose') return `⤢ tighter ${ref}${nameOf(p.cls)}${gone}`;
  const conf = p.conf == null ? '—' : Number(p.conf).toFixed(2);
  return `${AUDIT_GLYPH[p.kind] ? `${AUDIT_GLYPH[p.kind]} ` : ''}${conf} ${nameOf(p.cls)}${conflict >= 0 ? ` ⇄#${conflict + 1}` : ''}`;
}

/** Class isolation by digit keys (n = number of classes, 0 when unknown). A digit that can start an existing
    two-digit class waits for a second one; a pair that names no class keeps the first digit and starts over with
    the second (as Annotate's classDigit and Objects do). Returns {cls (null: no such class), buf (the digit kept
    for a second key, '' for none), joined (cls came from two digits)}. */
export function isolateDigit(buf, d, n) {
  d = String(d);
  if (buf && n) { const two = Number(buf + d); if (two < n) return { cls: two, buf: '', joined: true }; }
  const one = Number(d);
  if (n && one >= n) return { cls: null, buf: '', joined: false };
  return { cls: one, buf: one > 0 && one * 10 < n ? d : '', joined: false };
}

/** A focused control whose own keys must win over the view's: Enter / Space press a button or a link; arrows,
    Home, End and the page keys move a select, a slider or a radio group. `inside` (the grid) is never one. */
export function controlKeys(a, inside) {
  if (!a || !a.tagName || a === (typeof document !== 'undefined' ? document.body : null) || (inside && inside.contains(a))) return { press: false, arrows: false };
  const t = a.tagName, type = String(a.type || '').toLowerCase(), role = (a.getAttribute && a.getAttribute('role')) || '';
  const arrows = t === 'SELECT' || (t === 'INPUT' && (type === 'range' || type === 'radio')) || ['slider', 'radio', 'spinbutton', 'listbox', 'menuitem', 'tab'].includes(role);
  const press = arrows || t === 'BUTTON' || t === 'A' || t === 'SUMMARY' || t === 'INPUT' || t === 'TEXTAREA'
    || ['button', 'link', 'switch', 'checkbox', 'menuitemcheckbox', 'option'].includes(role);
  return { press, arrows };
}

/** "a/b/c.jpg" -> ["a/", "b/", "c.jpg"]: a path that may wrap only after a slash. */
export function pathParts(p) {
  return String(p ?? '').split('/').map((s, i, a) => (i < a.length - 1 ? `${s}/` : s)).filter(Boolean);
}

/** How many images picking `value` for filter `dim` would show, everything else as it is, from an
    /images/facets answer; null when the answer has no such facet. '' means "any" (the facet's total). */
export function facetValue(fx, dim, value) {
  const f = fx && fx[dim];
  if (!f || typeof f !== 'object') return null;
  // Split names read from macOS disks may come decomposed (NFD) on one side and composed (NFC) on the other.
  const n = (k) => Number(f[k] ?? f[String(k).normalize('NFC')] ?? f[String(k).normalize('NFD')]) || 0;
  const sum = () => Object.values(f).reduce((a, b) => a + (Number(b) || 0), 0);
  if (dim === 'split') return value ? n(value) : sum();
  if (dim === 'review') return value ? n(value) : n('none') + n('any');
  if (dim === 'ai') return value ? n(value) : n('any') + n('none');
  if (dim === 'labeled') return value === '' || value == null ? n('file') + n('unlabeled') : n({ 1: 'labeled', nobox: 'nobox', empty: 'empty', 0: 'unlabeled' }[value]);
  return n(value);
}

/** Indices of the grid tiles a rectangle touches (grid coordinates). G: {cols, rows, tileW, tileH, rowH, padX,
    padTop, gap}. Pure layout arithmetic, so rows that are not mounted count too. */
export function gridHits(G, total, x0, y0, x1, y1) {
  const out = new Set(), colW = G.tileW + G.gap, cols = [], rows = [];
  for (let c = Math.max(0, Math.floor((x0 - G.padX) / colW)); c <= Math.min(G.cols - 1, Math.floor((x1 - G.padX) / colW)); c++) {
    const left = G.padX + c * colW; if (x1 >= left && x0 <= left + G.tileW) cols.push(c);
  }
  for (let r = Math.max(0, Math.floor((y0 - G.padTop) / G.rowH)); r <= Math.min(G.rows - 1, Math.floor((y1 - G.padTop) / G.rowH)); r++) {
    const t = G.padTop + r * G.rowH; if (y1 >= t && y0 <= t + G.tileH) rows.push(r);
  }
  for (const r of rows) for (const c of cols) { const i = r * G.cols + c; if (i < total) out.add(i); }
  return out;
}

/** An /api/ai/nl-filter answer as Explore state: only known filter keys (lists joined as CSV, split names
    composed), a sort only when Explore offers it, an order only asc/desc. */
export function askResult(r) {
  const filters = {};
  for (const [k, v] of Object.entries((r && r.filters) || {})) {
    if (!FILTER_KEYS.includes(k) || v == null || v === '' || v === false || (Array.isArray(v) && !v.length)) continue;
    filters[k] = Array.isArray(v) ? v.join(',') : String(v);
  }
  if (filters.split) filters.split = filters.split.normalize('NFC');
  const so = r && r.sort;
  const key = so && typeof so === 'object' ? so.sort || so.key : so;
  if (!key || !SORTS.some(([k]) => k === key)) return { filters, sort: null, order: null };
  const ord = (so && typeof so === 'object' && so.order) || (r && r.order);
  return { filters, sort: key, order: ord === 'asc' || ord === 'desc' ? ord : defaultOrder(key) };
}

export function install(fovea) {
  fovea.registerTab({ id: 'explore', label: 'Explore', icon: 'grid', order: 20, render });
  if (fovea.bus) fovea.bus.on('labels:changed', ({ item } = {}) => {
    const cur = fovea.state && fovea.state.get ? fovea.state.get().currentDataset : null;
    if (item && item.id != null) noteLabelEdit(cur ? cur.id : null, item.id);
  });
  const k = fovea.keys, M = fovea.MOD;
  if (!k || !k.registerKeys) return;
  k.registerKeys('explore', 'Explore grid', [
    ['← → ↑ ↓', 'Move focus'], ['PageUp PageDown', 'Page up / down'], ['Home End', 'First / last image'], ['Enter', 'Inspect'],
    ['Space', 'Hold: quick look'], ['A F X', 'Approve / flag / exclude (focused image or selection)'], ['U', 'Clear review'],
    ['⇧A', 'Approve the unreviewed rest (keeps flagged and excluded)'], ['S ⇧S', 'Select / select range'], [`${M} A`, 'Select all matching images'],
    ['Drag', 'Select a rectangle (start between tiles; ⇧ or ' + M + ' adds)'],
    ['Esc', 'Clear selection, then the AI search, then class isolation'], ['0-9', 'Show only one class’s boxes'], ['O', 'Show / hide boxes'], ['[ ]', 'Smaller / larger tiles'],
    ['J ⇧J', 'Next / previous image with a label problem (AI finding, label issue, sequence or shape flag)'], ['N', 'Next unreviewed image'], ['E', 'Edit in Annotate'], ['/', 'Search filenames'],
    ['/ then ⇧/', 'Ask AI: search in your own words (when an AI provider is set up)'], [`${M} Z`, 'Undo the last review or AI decision'],
  ], 20);
  k.registerKeys('inspect', 'Inspect', [
    ['← →', 'Previous / next image'], ['Home End', 'First / last image'], ['A F X', 'Approve / flag / exclude'], ['U', 'Clear review'],
    ['N', 'Write a note (Enter saves)'], ['Space', 'Auto-play'], ['O', 'Show / hide boxes'], ['L', 'Show / hide labels'], ['0-9', 'Show only one class'],
    ['Z', 'Zoom to the selected box / fit'], ['= -', 'Zoom in / out'], ['Wheel', 'Zoom at the pointer (drag to pan)'],
    ['⇧Enter', 'Accept the visible proposals (not those over a box of another class, ⇄); with none left, apply the suggested fixes'], ['Delete', `Reject the visible proposals and fixes (${M}Z brings them back)`],
    ['{ }', 'Proposal threshold down / up'], ['I', 'Ask AI to check the labels'], ['J ⇧J', 'Next / previous image with a label problem'],
    ['E', 'Edit in Annotate'], [`${M} Z`, 'Undo the last review, accept or rejection'], ['Esc', 'Hide a QA flag’s details, then close'],
  ], 21);
}

async function render(el, { ctx, dataset, fovea }) {
  const { h, icon, ui, fmt, cls, isTyping, keys } = fovea;
  const { classColor, className, REVIEW } = fovea.colors;
  const ds = dataset;
  let names = ds.classes || [];
  const ssKey = `fovea.explore.${ds.id}`;
  const thrKey = `fovea.ann.thr.${ds.id}`;           // shared with Annotate's proposals slider

  // ---------------------------------------------------------------- state
  function parseQuery(q) {
    const f = {};
    for (const k of FILTER_KEYS) { const v = q.get(k); if (v != null && v !== '') f[k] = k === 'split' ? v.normalize('NFC') : v; }
    const sort = q.get('sort') || defaultSort(f);     // a saved selection in its own order (like Annotate), a QA flag most suspicious first
    return { filters: f, sort, order: q.get('order') || defaultOrder(sort), img: Number(q.get('img')) || null, inspect: q.get('view') === 'inspect' };
  }
  // Entering Explore with no query at all (tab bar, header link, G E) returns to where this dataset's Explore was
  // left in this browser session: filters, sort and the focused image. An explicit query always wins.
  let q0 = ctx.query;
  if (![...q0.keys()].length) { const saved = ss.get(ssKey); if (saved) q0 = new URLSearchParams(saved); }
  const st0 = parseQuery(q0);
  let filters = st0.filters, sort = st0.sort, order = st0.order;
  let tileSize = SIZES.includes(ls.get('fovea.tile', 'm')) ? ls.get('fovea.tile', 'm') : 'm';
  let showOverlay = ls.get('fovea.overlay', '1') !== '0';
  let advance = ls.get('fovea.review.advance', '1') !== '0';
  let propThreshold = clamp(Number(ls.get(thrKey, '0.25')) || 0, 0, 1);
  let isolate = null;               // class id whose boxes alone are drawn (digit keys); null = all
  let stats = null, issueLabels = { ...BASE_ISSUE_LABELS };
  let facets = null, facetsFor = '', facetWatch = null;   // counts per choice under the other filters (see loadFacets)
  let setCount = null;              // {token, count} of the last server-side selection set made or shown here
  let askOn = false, askErr = '', askBusy = null, askProv = null, askList = [], aiApplied = null, askReady = null;   // Ask AI (see below)
  let destroyed = false;

  let cursor = null, total = null, gen = 0, L = null;
  const mounted = new Map();        // list index -> tile element (real or placeholder)
  let focusIdx = -1, anchorIdx = -1, anchorBase = new Set(), anchorBaseExcept = new Set();
  let selected = new Set(), allMode = false, except = new Set();
  const gone = new Set();           // ids reviewed/edited so that they no longer match the filter
  const undoStack = [];
  const drafts = new Map();          // image id -> note text not stored yet (see saveNote)
  let inspect = null, inspectPushed = false, quick = null;

  // ---------------------------------------------------------------- skeleton
  const root = h('div', { class: 'explore' });
  const toolbar = h('div', { class: 'explore-toolbar', role: 'toolbar', 'aria-label': 'Filters and view' });
  const chipsRow = h('div', { class: 'explore-chips hidden' });
  const banner = h('div', { class: 'explore-banner hidden', role: 'status' });
  const body = h('div', { class: 'explore-body' });
  const vgrid = h('div', { class: cls('vgrid', !showOverlay && 'no-overlay'), role: 'listbox', 'aria-multiselectable': 'true', 'aria-label': 'Images', tabindex: '-1' });
  const emptyEl = h('div', { class: 'explore-empty hidden' });
  const bulk = h('div', { class: 'bulk-bar hidden', role: 'toolbar', 'aria-label': 'Selection actions' });
  body.append(vgrid, emptyEl);
  root.append(toolbar, chipsRow, banner, body, bulk);
  el.appendChild(root);

  // ---------------------------------------------------------------- toolbar
  // The search box has two modes: filename search (live, debounced) and "Ask AI" (natural language compiled to
  // filters by /api/ai/nl-filter, sent on Enter). Ask AI exists only while a provider that can answer free-form
  // questions is set up; the ✦ button left of the box switches modes, so it never moves under the pointer.
  const searchInput = h('input', { class: 'input', placeholder: 'Search filename…', value: filters.q || '', 'aria-label': 'Search filenames', spellcheck: 'false', autocomplete: 'off' });
  const searchClear = h('button', { class: 'btn btn-ghost btn-icon btn-sm search-clear', 'aria-label': 'Clear search', 'data-tip': 'Clear (Esc)', onClick: () => {
    searchInput.value = '';
    if (askOn) { askErr = ''; renderAsk(); } else applySearch();
    searchInput.focus();
  } }, icon('x', 12));
  const applySearch = () => {
    if (askOn) return;
    const v = searchInput.value.trim();
    searchClear.classList.toggle('hidden', !searchInput.value);
    if ((filters.q || '') !== v) setFilter('q', v);
  };
  const searchSoon = fovea.debounce(applySearch, 300);
  searchInput.addEventListener('input', (e) => { searchClear.classList.toggle('hidden', !searchInput.value); if (askOn) { if (askErr) { askErr = ''; renderAsk(); } } else if (!e.isComposing) searchSoon(); });
  searchInput.addEventListener('compositionend', () => { if (!askOn) searchSoon(); });
  searchInput.addEventListener('keydown', (e) => {
    // Korean (and any IME) input: Enter or Esc while a syllable is being composed belongs to the IME.
    if (e.isComposing || e.keyCode === 229) return;
    if (keys.is(e, 'shift+/') && !searchInput.value && askAvail()) { e.preventDefault(); setAsk(!askOn, { focus: true }); return; }
    if (e.key === 'Enter') { e.preventDefault(); if (askOn) submitAsk(); else applySearch(); }
    else if (e.key === 'Escape') { e.preventDefault(); if (askOn) askEscape(); else if (searchInput.value) { searchInput.value = ''; applySearch(); } else { searchInput.blur(); focusGrid(); } }
    else if (e.key === 'ArrowDown') { e.preventDefault(); searchInput.blur(); focusGrid(); }
  });
  searchClear.classList.toggle('hidden', !searchInput.value);
  const searchIcon = h('span', { class: 'search-ico', 'aria-hidden': 'true' }, icon('search'));
  const askBtn = h('button', { class: 'btn btn-icon ask-toggle hidden', type: 'button', 'aria-pressed': 'false', 'aria-label': 'Ask AI', 'data-tip': 'Ask AI: search in your own words',
    onClick: () => setAsk(!askOn, { focus: true }) }, icon('sparkles', 14));
  const askPop = h('div', { class: 'ask-pop hidden' });
  const searchSlot = h('div', { class: 'search-slot' }, h('div', { class: 'input-wrap search' }, searchIcon, searchInput, searchClear), askPop);
  searchSlot.addEventListener('focusin', () => renderAsk());
  searchSlot.addEventListener('focusout', () => setTimeout(() => { if (!destroyed && !searchSlot.contains(document.activeElement)) renderAsk(); }, 0));

  const SPLIT_RANK = { train: 0, val: 1, valid: 1, validation: 1, test: 2 };
  // NFC: folder names read from macOS disks arrive decomposed (Korean jamo); the index stores composed text.
  const splits = [...new Set((ds.splits || []).filter(Boolean).map(s => String(s).normalize('NFC')))].sort((a, b) => (SPLIT_RANK[a] ?? 9) - (SPLIT_RANK[b] ?? 9) || a.localeCompare(b));
  const splitCtl = splits.length < 2 ? null
    : splits.length <= 4 && splits.reduce((a, s) => a + s.length, 0) <= 24
      ? ui.seg([{ value: '', label: 'All' }, ...splits.map(s => ({ value: s, label: s }))], filters.split || '', (v) => setFilter('split', v))
      : select([['', 'All splits'], ...splits.map(s => [s, s])], filters.split || '', (v) => setFilter('split', v), 'Split');
  /** Images per split under the other active filters: in the dropdown's options, or the segmented control's tips. */
  function updateSplitCounts() {
    if (!splitCtl) return;
    const vals = ['', ...splits];
    const n = (v) => facetCount('split', v);
    if (splitCtl.tagName === 'SELECT') {
      [...splitCtl.options].forEach((o, k) => { const c = n(vals[k]); o.textContent = `${k ? vals[k] : 'All splits'}${c == null ? '' : ` · ${fmt.num(c)}`}`; });
      return;
    }
    [...splitCtl.querySelectorAll('button')].forEach((b, k) => {
      const c = n(vals[k]);
      b.dataset.tip = c == null ? '' : `${fmt.plural(c, 'image')}${Object.keys(filters).some(x => x !== 'split') ? ' with the other filters' : ''}`;
      b.classList.toggle('zero', c === 0);
    });
  }
  const clsBtn = h('button', { class: 'btn', 'aria-haspopup': 'dialog', onClick: () => openClassPop() }, icon('tags', 14), h('span', { class: 'lbl' }, 'Classes'), icon('chevronDown', 12));
  const filterBtn = h('button', { class: 'btn', 'aria-haspopup': 'dialog', onClick: () => openFilterPop() }, icon('filter', 14), h('span', { class: 'lbl' }, 'Filters'), icon('chevronDown', 12));
  const countEl = h('span', { class: 'result-count', 'aria-live': 'polite' });
  // The grid's status line: results of quick single-image actions (undo, "nothing to change", a J miss) are said
  // here, next to the count, instead of in a toast. It is drawn over the toolbar's free space, right-aligned against
  // the count; a text longer than that space also covers the count for those few seconds. Nothing moves either way.
  const spacerEl = h('span', { class: 'spacer' });
  const gridStatusText = h('span', { class: 't' });
  const gridStatus = h('span', { class: 'grid-status hidden', role: 'status', 'aria-live': 'polite' }, gridStatusText);
  let gridStatusTimer = null;
  function sayInGrid(text, kind = '', title = '') {
    clearTimeout(gridStatusTimer);
    countEl.style.visibility = '';
    gridStatusText.textContent = text || ''; gridStatus.title = title || text || ''; gridStatus.dataset.status = kind;
    gridStatus.classList.toggle('hidden', !text);
    if (!text) return;
    const tb = toolbar.getBoundingClientRect(), ct = countEl.getBoundingClientRect();
    let sp = spacerEl.getBoundingClientRect();
    if (Math.abs(sp.top + sp.height / 2 - (ct.top + ct.height / 2)) > 8) sp = { left: ct.left, width: 0 };   // a wrapped toolbar (narrow window): over the count only
    gridStatus.style.cssText = `left:${Math.round(sp.left - tb.left)}px;top:${Math.round(ct.top - tb.top)}px;height:${Math.round(ct.height)}px;width:${Math.round(sp.width)}px`;
    if (!sp.width || gridStatusText.scrollWidth > gridStatusText.clientWidth + 1) {
      gridStatus.style.width = `${Math.round(ct.right - sp.left)}px`;
      countEl.style.visibility = 'hidden';
    }
    gridStatusTimer = setTimeout(() => sayInGrid(''), STATUS_MS);
  }
  const viewBtn = h('button', { class: 'btn view-btn', 'aria-haspopup': 'dialog', 'data-tip': 'Sort, tile size, boxes', onClick: () => openViewPop() }, icon('sliders', 14), h('span', { class: 'lbl' }), icon('chevronDown', 12));
  const overlayBtn = h('button', { class: cls('btn btn-icon', showOverlay && 'active'), 'aria-pressed': String(showOverlay), 'aria-label': 'Show boxes', 'data-tip': 'Show / hide boxes (O)', onClick: () => toggleOverlay() }, icon('square', 14));
  const annotateBtn = h('button', { class: 'btn btn-icon', 'aria-label': 'Annotate this list', 'data-tip': 'Annotate the images in this list', onClick: () => leaveTo(`/d/${ds.id}/annotate?${portableQuery({ img: focusedId() }).toString()}`) }, icon('pen', 14));
  toolbar.append(askBtn, searchSlot, splitCtl || '', clsBtn, filterBtn,
    spacerEl, countEl, viewBtn, overlayBtn, annotateBtn, gridStatus);

  function select(opts, value, onChange, label) {
    const s = h('select', { class: 'select', 'aria-label': label || null, onChange: (e) => { onChange(e.target.value); e.target.blur(); } },
      opts.map(([v, l]) => h('option', { value: v, selected: v === value }, l)));
    return s;
  }
  // A selection set can also be listed in the order it was frozen in (Annotate opens one that way).
  const sortList = () => [...SORTS.filter(([k]) => k !== 'flag' || sort === 'flag' || flagSortable(filters)), ...(filters.set ? [['set', 'Selection order']] : [])];
  const activeCount = () => Object.keys(filters).filter(k => k !== 'seed').length;   // seed only orders "Random"
  function updateControls() {
    if (!askOn && document.activeElement !== searchInput) searchInput.value = filters.q || '';
    searchClear.classList.toggle('hidden', !searchInput.value);
    if (splitCtl) { if (splitCtl.setValue) splitCtl.setValue(filters.split || ''); else splitCtl.value = filters.split || ''; updateSplitCounts(); }
    const nc = csv(filters.cls || filters.only_cls).length;
    clsBtn.querySelector('.lbl').textContent = nc ? `${nc} class${nc > 1 ? 'es' : ''}` : 'Classes';
    clsBtn.classList.toggle('active', nc > 0);
    const nf = ['labeled', 'review', 'issue', 'ai', 'min_boxes', 'max_boxes', 'exclude_review', 'has_label'].filter(k => filters[k]).length;
    filterBtn.querySelector('.lbl').textContent = nf ? `Filters · ${nf}` : 'Filters';
    filterBtn.classList.toggle('active', nf > 0);
    const s = sortList().find(([k]) => k === sort), dir = order === 'desc' ? '↓' : '↑';
    // The button says the sort in a word or two (the toolbar must keep its icon buttons on screen at 1280 px);
    // the tooltip and the popover have the full name.
    viewBtn.querySelector('.lbl').textContent = `${SORT_SHORT[sort] || (s ? s[1].replace(/ \(.*\)$/, '') : sort)} ${dir}`;
    viewBtn.dataset.tip = `Sort: ${s ? s[1] : sort} ${dir} · tile size, boxes`;
    renderChips();
  }

  // ---------------------------------------------------------------- filter chips (every active filter is visible and removable)
  const LABEL_OF = { labeled: Object.fromEntries(LABELED), review: Object.fromEntries(REVIEWS), ai: Object.fromEntries(AI_STATES) };
  function filterChips() {
    const out = [];
    const clsNames = (v) => { const ids = csv(v).map(Number); return ids.length > 3 ? `${ids.length} classes` : ids.map(i => `${i} · ${className(names, i)}`).join(', '); };
    for (const [k, v] of Object.entries(filters)) {
      if (k === 'seed') continue;
      let text;
      if (k === 'split') text = `Split: ${v}`;
      else if (k === 'cls') text = `Has: ${clsNames(v)}`;
      else if (k === 'only_cls') text = `Only: ${clsNames(v)}`;
      else if (k === 'labeled' || k === 'review' || k === 'ai') text = LABEL_OF[k][v] || `${k}: ${v}`;
      else if (k === 'issue') { const c = csv(v); text = c.includes('any') ? 'Any issue' : c.length > 2 ? `${c.length} issue types` : c.map(x => issueLabels[x] || x).join(' or '); }
      else if (k === 'q') text = `Name contains “${v}”`;
      else if (k === 'seq') text = `Sequence: ${v}`;
      else if (k === 'ids') { const n = csv(v).length; text = `${fmt.num(n)} chosen image${n === 1 ? '' : 's'}`; }
      else if (k === 'set') text = setCount != null && setCount.token === v ? `Selection of ${fmt.plural(setCount.count, 'image')}` : 'Saved selection';
      else if (k === 'min_boxes') text = `≥ ${v} boxes`;
      else if (k === 'max_boxes') text = `≤ ${v} boxes`;
      else if (k === 'exclude_review') text = `Not ${csv(v).join(', ')}`;
      else text = `${k}: ${v}`;
      out.push(ui.chip(text, { cls: 'filter-chip', onRemove: () => setFilter(k, '') }));
    }
    return out;
  }
  function renderChips() {
    const chips = filterChips();
    if (isolate != null) chips.push(ui.chip(`Showing class ${isolate} · ${className(names, isolate)} only`, { cls: 'filter-chip isolate', icon: 'eye', onRemove: () => setIsolate(null) }));
    // An AI search stays "applied" only while the list shows exactly what it produced; any other change ends it.
    if (aiApplied && aiApplied.state !== stateQuery().toString()) aiApplied = null;
    const note = aiApplied ? aiNote() : null;
    chipsRow.classList.toggle('hidden', !chips.length && !note);
    chipsRow.replaceChildren(...(note ? [note] : []), ...chips, activeCount() > 1 ? h('button', { class: 'btn btn-ghost btn-sm', onClick: () => clearFilters() }, 'Clear all') : '');
  }

  // ---------------------------------------------------------------- popovers: classes, filters, view
  /** A live count next to one choice of a popover (see facetCount); refreshed in place when new facets arrive. */
  function countSpan(dim, value, zeroTip) {
    const s = h('span', { class: 'n' });
    s._upd = () => {
      const n = facetCount(dim, value), row = s.parentElement;
      s.textContent = n == null ? '' : fmt.num(n);
      if (row) { row.classList.toggle('zero', n === 0); row.title = n === 0 ? zeroTip() : ''; }
    };
    return s;
  }
  /** Track the count spans of an open popover; facets arriving later update them, and the popover dims its
      numbers while they describe the previous filters. */
  function watchCounts(content) {
    const upd = () => {
      content.querySelectorAll('.n, .live-upd').forEach(s => s._upd && s._upd());
      content.querySelectorAll('select[data-dim]').forEach(s => s._upd && s._upd());
      content.classList.toggle('counts-stale', !!facets && facetsFor !== facetKeyOf(filters));
      content.querySelectorAll('.counts-note').forEach(n => { n.textContent = countsNote(n.dataset.dim, n.dataset.mode); });
    };
    facetWatch = upd;
    upd();
    return () => { if (facetWatch === upd) facetWatch = null; };
  }
  /** What the numbers mean: the whole dataset (no facets on this server, or no other filter) or the current filters. */
  function countsNote(dim, mode) {
    // In "Only" mode a class's number still counts the images that CONTAIN it (with other classes too): the
    // result of the ticked set is said on its own line (see openClassPop).
    const each = dim === 'cls' && mode === 'only' ? ' that contain each' : '';
    if (!facets) return stats ? `images in the dataset${each}` : '';
    const others = Object.keys(filters).filter(k => !(dim === 'cls' ? ['cls', 'only_cls'] : [dim]).includes(k));
    return others.length ? `images${each} with your other filters` : `images${each}`;
  }
  const zeroTipFor = (what) => () => (facets && Object.keys(filters).length ? `No image matches this ${what} together with your other filters` : `No image in this dataset ${what === 'class' ? 'has a box of this class' : 'has this'}`);

  function openClassPop() {
    const sel = new Set(csv(filters.cls || filters.only_cls).map(Number));
    let mode = filters.only_cls ? 'only' : 'any';
    const known = new Set([...((stats && stats.classes) || []).map(c => c.cls), ...Object.keys((facets && facets.cls) || {}).map(Number)]);
    const ids = [...new Set([...names.map((_, i) => i), ...known])].sort((a, b) => a - b);
    const commit = () => {
      const v = [...sel].sort((a, b) => a - b).join(',');
      const next = { ...filters }; delete next.cls; delete next.only_cls;
      if (v) next[mode === 'only' ? 'only_cls' : 'cls'] = v;
      setFilters(next);
    };
    const commitSoon = fovea.debounce(commit, 450);
    const listEl = h('div', { class: 'cls-list', role: 'group', 'aria-label': 'Classes' });
    const search = ids.length > 8 ? h('input', { class: 'input input-sm', placeholder: 'Find class…', 'aria-label': 'Find class', onInput: () => renderList() }) : null;
    const row = (i) => h('label', { class: 'check cls-row' },
      h('input', { type: 'checkbox', checked: sel.has(i), onChange: (e) => { e.target.checked ? sel.add(i) : sel.delete(i); commitSoon(); if (facetWatch) facetWatch(); } }),
      h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'grow truncate', title: className(names, i) }, `${i} · ${className(names, i)}`),
      countSpan('cls', i, zeroTipFor('class')));
    let recount = () => {};
    function renderList() {
      const q = search ? search.value.trim().toLowerCase() : '';
      const shown = ids.filter(i => !q || `${i} ${className(names, i)}`.toLowerCase().includes(q));
      listEl.replaceChildren(...(shown.length ? shown.map(row) : [h('div', { class: 'faint small', style: 'padding:6px' }, ids.length ? 'No class matches' : 'No classes')]));
      recount();
    }
    const setAll = (on) => { if (on) ids.forEach(i => sel.add(i)); else sel.clear(); renderList(); commitSoon(); };
    // "Only": the per-class numbers cannot predict the result (an image with classes 0 and 1 counts for both, but
    // matches "only 0" for neither), so the list's own count for the ticked set is said here.
    const noteEl = h('span', { class: 'xs faint counts-note', 'data-dim': 'cls', 'data-mode': mode });
    const onlyEl = h('div', { class: 'xs only-total live-upd', role: 'status' });
    onlyEl._upd = () => {                 // one line in both modes, so switching never moves the list below
      noteEl.dataset.mode = mode;
      const want = [...sel].sort((a, b) => a - b).join(','), only = mode === 'only';
      const what = only ? `Only the ticked class${sel.size === 1 ? '' : 'es'}` : `Any of the ticked class${sel.size === 1 ? '' : 'es'}`;
      onlyEl.textContent = !want ? (only ? 'Tick classes: images whose boxes are all of those classes.' : 'Tick classes: images with a box of any of them.')
        : filters[only ? 'only_cls' : 'cls'] === want && total != null ? `${what}: ${fmt.plural(total, 'image')}` : `${what}: …`;
    };
    const content = h('div', { class: 'col gap-8 cls-pop' },
      h('div', { class: 'row' }, h('span', { class: 'small strong muted grow' }, 'Images that contain'),
        ui.seg([{ value: 'any', label: 'Any of', tip: 'At least one box of a chosen class' }, { value: 'only', label: 'Only', tip: 'Boxes of the chosen classes and no others' }], mode, (v) => { mode = v; if (sel.size) commitSoon(); if (facetWatch) facetWatch(); })),
      onlyEl,
      search,
      h('div', { class: 'row gap-4' }, h('button', { class: 'btn btn-sm', onClick: () => setAll(true) }, 'All'), h('button', { class: 'btn btn-sm', onClick: () => setAll(false) }, 'None'),
        h('span', { class: 'spacer' }), noteEl),
      listEl);
    renderList();
    const stop = watchCounts(content);
    recount = () => { if (facetWatch) facetWatch(); };
    ui.popover(clsBtn, content, { width: 300, onClose: stop });
    scheduleFacets(0);
    if (search) setTimeout(() => search.focus(), 0);
  }

  function openFilterPop() {
    const codes = Object.keys(issueLabels);
    const groups = [
      ['Label file', codes.filter(c => !FLAG_CODES.has(c) && !c.startsWith('ai_'))],
      ['Data quality', codes.filter(c => FLAG_CODES.has(c))],
      ['AI audit', codes.filter(c => c.startsWith('ai_'))],
    ].filter(([, c]) => c.length);
    const chosen = new Set(csv(filters.issue));
    const boxes = new Map();             // code -> its checkbox
    const commitIssues = fovea.debounce(() => setFilter('issue', [...chosen].join(',')), 400);
    const issueRow = (code, label) => {
      // Data-quality flags have a count only once the QA checks ran: before that "0" would be a false all-clear.
      const known = !FLAG_CODES.has(code) || issueCodeCounted(code);
      const box = h('input', { type: 'checkbox', checked: chosen.has(code), onChange: (e) => {
        if (e.target.checked) {
          chosen.add(code);
          // "Any issue" and specific issues are alternatives, not a combination.
          for (const [c, b] of boxes) if (code === 'any' ? c !== 'any' : c === 'any') { chosen.delete(c); b.checked = false; }
        } else chosen.delete(code);
        commitIssues();
      } });
      boxes.set(code, box);
      return h('label', { class: 'check cls-row' }, box, h('span', { class: 'grow truncate' }, label), known ? countSpan('issue', code, zeroTipFor('issue')) : null);
    };
    const num = (key, ph) => h('input', { class: 'input input-sm', type: 'number', min: 0, step: 1, placeholder: ph, value: filters[key] || '', 'aria-label': ph,
      onChange: (e) => { const v = e.target.value.trim(); setFilter(key, v === '' ? '' : String(Math.max(0, Math.round(Number(v)) || 0))); } });
    const content = h('div', { class: 'col gap-8 filters-pop' },
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Labels'), countSelect('labeled', LABELED, 'Label state')),
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Review'), countSelect('review', REVIEWS, 'Review state')),
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'AI'), countSelect('ai', AI_STATES, 'AI state')),
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Boxes'), h('div', { class: 'row gap-4' }, num('min_boxes', 'min'), h('span', { class: 'faint' }, '–'), num('max_boxes', 'max'))),
      h('div', { class: 'small strong muted mt-8' }, 'Issues', h('span', { class: 'xs faint counts-note', 'data-dim': 'issue', style: 'font-weight:400;margin-left:6px' })),
      h('div', { class: 'issue-list' }, issueRow('any', 'Any issue or flag'),
        groups.map(([title, cs]) => [h('div', { class: 'xs faint grp' }, title), cs.map(c => issueRow(c, issueLabels[c] || c))])),
      h('div', { class: 'row' }, h('span', { class: 'spacer' }), h('button', { class: 'btn btn-sm', onClick: () => { clearFilters(); ui.closeMenu && ui.closeMenu(); } }, 'Clear all filters')));
    const stop = watchCounts(content);
    ui.popover(filterBtn, content, { width: 320, onClose: stop });
    scheduleFacets(0);
  }
  /** A filter dropdown whose options say how many images each choice would show (facets), kept current. */
  function countSelect(dim, opts, label) {
    const s = select(opts, filters[dim] || '', v => setFilter(dim, v), label);
    s.dataset.dim = dim;
    s._upd = () => [...s.options].forEach((o, k) => {
      const n = facets ? facetCount(dim, opts[k][0]) : null;
      o.textContent = n == null ? opts[k][1] : `${opts[k][1]} · ${fmt.num(n)}`;
    });
    return s;
  }

  function openViewPop() {
    const sizeSeg = ui.seg(SIZES.map(s => ({ value: s, label: s.toUpperCase() })), tileSize, (v) => setTileSize(v));
    let dirSeg = null;
    const content = h('div', { class: 'col gap-8 view-pop' },
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Sort'), select(sortList(), sort, (v) => {
        sort = v; order = defaultOrder(v);
        if (v !== 'random') delete filters.seed;
        reload(); if (dirSeg) dirSeg.setValue(order); ui.closeMenu(); openViewPop();
      }), sort === 'random' ? h('button', { class: 'btn btn-sm', 'data-tip': 'Another random order (the same one comes back with this link)', onClick: () => setFilter('seed', String(1 + Math.floor(Math.random() * 999999))) }, icon('shuffle', 12), 'Shuffle') : null),
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Order'), dirSegEl()),
      h('div', { class: 'fp-row' }, h('span', { class: 'small muted' }, 'Tiles'), sizeSeg, h('span', { class: 'xs faint' }, '[ ]')),
      h('label', { class: 'fp-row sw' }, h('span', { class: 'small' }, 'Show boxes', h('span', { class: 'xs faint' }, '  O')), ui.switchBtn(showOverlay, () => toggleOverlay())),
      h('label', { class: 'fp-row sw', title: 'After approve, exclude or clear on one image, move to the next. Flag stays.' }, h('span', { class: 'small' }, 'Advance after review'),
        ui.switchBtn(advance, (v) => { advance = v; ls.set('fovea.review.advance', v ? '1' : '0'); })),
      savedViewsEl());
    function dirSegEl() { dirSeg = ui.seg([{ value: 'asc', label: 'Ascending', icon: 'arrowUp' }, { value: 'desc', label: 'Descending', icon: 'arrowDown' }], order, (v) => { order = v; reload(); }); return dirSeg; }
    ui.popover(viewBtn, content, { width: 330 });
  }

  // Saved views: named filter + sort presets per dataset, kept in this browser.
  const viewsKey = `fovea.views.${ds.id}`;
  const readViews = () => { try { return JSON.parse(ls.get(viewsKey, '[]')) || []; } catch (e) { return []; } };
  function savedViewsEl() {
    const box = h('div', { class: 'col gap-4 saved-views' });
    const draw = () => {
      const views = readViews(), cur = stateQuery().toString();
      box.replaceChildren(h('div', { class: 'small strong muted mt-8' }, 'Saved views'),
        ...(views.length ? views.map((v, k) => h('div', { class: cls('row sv-row', v.query === cur && 'current') },
          h('button', { class: 'btn btn-ghost btn-sm grow sv-name', title: v.query || 'All images', onClick: () => { ui.closeMenu(); applyQuery(new URLSearchParams(v.query)); } }, v.name),
          h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': `Delete view ${v.name}`, onClick: () => { const all = readViews(); all.splice(k, 1); ls.set(viewsKey, JSON.stringify(all)); draw(); } }, icon('x', 12))))
          : [h('div', { class: 'xs faint' }, 'Save the current filters and sort to come back to them in one click.')]),
        h('button', { class: 'btn btn-sm', onClick: async () => {
          ui.closeMenu();
          const name = await ui.prompt({ title: 'Save this view', label: 'Name', value: filterChips().map(c => c.textContent).join(' · ').slice(0, 60) || 'All images', okLabel: 'Save' });
          if (!name || !name.trim()) return;
          const all = readViews().filter(v => v.name !== name.trim());
          all.push({ name: name.trim(), query: stateQuery().toString() });
          ls.set(viewsKey, JSON.stringify(all.slice(-20)));
          ui.toast(`Saved view “${name.trim()}”`, { type: 'ok', timeout: 1800 });
        } }, icon('bookmark', 12), 'Save current view…'));
    };
    draw();
    return box;
  }

  // ---------------------------------------------------------------- filter changes
  function setFilter(k, v) { const next = { ...filters }; if (v == null || v === '') delete next[k]; else next[k] = String(v); setFilters(next); }
  function setFilters(next, opts = {}) {
    const clean = {}; for (const k of FILTER_KEYS) if (next[k] != null && next[k] !== '') clean[k] = String(next[k]);
    if (sameFilters(clean, filters) && !opts.force) { updateControls(); return; }
    ({ sort, order } = nextSort(filters, clean, sort, order));
    filters = clean; reload(opts);
  }
  function sameFilters(a, b) { const ka = Object.keys(a), kb = Object.keys(b); return ka.length === kb.length && ka.every(k => a[k] === b[k]); }
  function clearFilters() { setFilters({}); }

  // ---------------------------------------------------------------- Ask AI: natural-language search compiled to filters
  // Only text goes out (the question, class names and class-card prompts — never images). The answer is ordinary
  // filters: they appear as chips like any other, the one-line explanation sits in front of them, and Esc (or Undo)
  // brings back the filters the list had before the first AI search.
  const askKey = 'fovea.ask.provider';
  async function loadAskProviders() {
    const [P, local] = await Promise.all([fovea.api.get('/api/ai/providers').catch(() => null), serverSeesLocal(fovea.api)]);
    if (destroyed) return;
    const all = (P && (Array.isArray(P) ? P : P.providers)) || [];
    // Local servers first (free, private), then cloud providers, the test double last.
    askList = all.filter(p => askUsable(P, p, local, location.hostname)).sort((a, b) => (a.id === 'mock') - (b.id === 'mock') || (a.kind === 'cloud') - (b.kind === 'cloud'));
    const saved = ls.get(askKey, '');
    askProv = askList.find(p => p.id === saved) || askList[0] || null;
    askBtn.classList.toggle('hidden', !askProv);
    if (!askProv && askOn) setAsk(false);
    else renderAsk();
  }
  const askAvail = () => !!askProv;
  const askWho = () => (askProv ? askProv.name || askProv.id : '');
  function setAsk(on, { focus = false } = {}) {
    on = !!on && askAvail();
    if (on === askOn) { if (focus) searchInput.focus(); return; }
    askOn = on; askErr = ''; askReady = null;
    if (!on && askBusy) { askBusy.ctl.abort(); askBusy = null; }
    searchInput.value = on ? '' : filters.q || '';
    searchClear.classList.toggle('hidden', !searchInput.value);
    renderAsk();
    if (focus) searchInput.focus();
  }
  function renderAsk() {
    if (destroyed) return;
    askBtn.classList.toggle('active', askOn);
    askBtn.setAttribute('aria-pressed', String(askOn));
    askBtn.dataset.tip = askOn ? 'Back to filename search' : `Ask AI (${askWho()}): search in your own words · in the empty box ⇧/`;
    searchIcon.replaceChildren(icon(askOn ? 'sparkles' : 'search'));
    searchInput.placeholder = askOn ? 'Ask in your own words… (Enter)' : 'Search filename…';
    searchInput.setAttribute('aria-label', askOn ? 'Ask AI to filter the images' : 'Search filenames');
    searchInput.readOnly = !!askBusy;
    const focused = searchSlot.contains(document.activeElement);
    const open = askOn && (focused || !!askBusy || !!askErr || !!askReady);
    searchSlot.classList.toggle('ask', askOn);
    searchSlot.classList.toggle('open', open);
    askPop.classList.toggle('hidden', !open);
    if (!open) return;
    if (askBusy) {
      askPop.replaceChildren(h('div', { class: 'row gap-8', role: 'status' }, ui.spinner(), h('span', { class: 'grow' }, `Asking ${askWho()}…`), askBusy.clock,
        h('button', { class: 'btn btn-sm', onMousedown: (e) => e.preventDefault(), onClick: () => askEscape() }, 'Cancel', h('span', { class: 'kbd' }, 'Esc'))));
      return;
    }
    const examples = ['fallen people in val that nobody reviewed', '박스가 5개 이상인데 검토 안 한 이미지', 'near-duplicates across splits'];
    const who = askProv && askProv.kind === 'cloud' && askProv.id !== 'mock'
      ? `Your question and the class names go to ${askWho()} — no images.` : `Answered by ${askWho()}${askProv && askProv.location ? ` (${String(askProv.location).replace(/^Images? (stay|are sent to) /i, '').replace(/\.$/, '')})` : ''}.`;
    askPop.replaceChildren(
      askReady ? askReadyEl() : '',
      askErr ? h('div', { class: 'ask-err', role: 'alert' }, icon('alert', 13), h('span', { class: 'grow' }, askErr),
        h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': 'Dismiss', onMousedown: (e) => e.preventDefault(), onClick: () => { askErr = ''; renderAsk(); } }, icon('x', 12))) : '',
      h('div', { class: 'small' }, 'Describe the images you want, in English or Korean. ', h('span', { class: 'faint' }, 'Enter searches · Esc goes back')),
      h('div', { class: 'ask-ex' }, examples.map(x => h('button', { class: 'chip clickable', type: 'button', onMousedown: (e) => e.preventDefault(),
        onClick: () => { searchInput.value = x; searchClear.classList.remove('hidden'); searchInput.focus(); } }, x))),
      h('div', { class: 'row gap-6 xs faint ask-who' }, h('span', { class: 'grow' }, who),
        askList.length > 1 ? h('button', { class: 'btn btn-ghost btn-sm', 'aria-haspopup': 'menu', onMousedown: (e) => e.preventDefault(), onClick: (e) => pickAskProvider(e.currentTarget) }, 'Change', icon('chevronDown', 11)) : null));
  }
  function pickAskProvider(anchor) {
    ui.menu(anchor, askList.map(p => ({ label: p.name || p.id, icon: p.kind === 'cloud' && p.id !== 'mock' ? 'cloud' : 'cpu', kbd: p.id === (askProv && askProv.id) ? '✓' : null,
      onClick: () => { askProv = p; ls.set(askKey, p.id); renderAsk(); searchInput.focus(); } })));
  }
  /** An answer that came back after the list changed: kept here, applied only when asked to. */
  function askReadyEl() {
    const a = askReady, res = askResult(a.r), n = Object.keys(res.filters).length;
    const expl = String(a.r.explanation || '').trim();
    return h('div', { class: 'ask-ready', role: 'status' }, icon('sparkles', 13),
      h('span', { class: 'grow', title: expl || null }, `Filters for “${a.text}” are ready`,
        h('span', { class: 'faint' }, ` — not applied, the list changed while the AI answered · ${n ? fmt.plural(n, 'filter') : 'no filters'}${expl ? `: ${expl}` : ''}`)),
      h('button', { class: 'btn btn-sm btn-primary', onMousedown: (e) => e.preventDefault(), onClick: () => { const x = askReady; askReady = null; if (x) applyAsk(x.r, x.text); } }, 'Apply'),
      h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': 'Dismiss', 'data-tip': 'Dismiss (Esc)', onMousedown: (e) => e.preventDefault(), onClick: () => { askReady = null; renderAsk(); } }, icon('x', 12)));
  }
  /** Esc in the box: cancel a pending question, then undo the AI's filters, then clear the text, then leave Ask AI. */
  function askEscape() {
    if (askBusy) { askBusy.ctl.abort(); askBusy = null; renderAsk(); return; }
    if (askReady) { askReady = null; renderAsk(); return; }
    if (askErr) { askErr = ''; renderAsk(); return; }
    if (aiApplied) { restoreAsk(); return; }
    if (searchInput.value) { searchInput.value = ''; searchClear.classList.add('hidden'); renderAsk(); return; }
    setAsk(false);
    searchInput.blur(); focusGrid();
  }
  async function submitAsk() {
    const text = searchInput.value.trim();
    if (!text || askBusy || !askProv) return;
    const ctl = new AbortController();
    // The list the question was asked about: an answer that arrives after the user moved on (other filters, an
    // image open in Inspect) must not replace what they are looking at.
    askBusy = { ctl, clock: ui.elapsedClock(), state: stateQuery().toString() };
    askErr = ''; askReady = null;
    renderAsk();
    try {
      const r = await fovea.api.post('/api/ai/nl-filter', { dataset_id: ds.id, text, provider: askProv.id }, { signal: ctl.signal });
      if (destroyed || !askBusy || askBusy.ctl !== ctl) return;
      const asked = askBusy.state;
      askBusy = null;
      if (inspect || stateQuery().toString() !== asked) { askReady = { r: r || {}, text }; renderAsk(); return; }
      applyAsk(r || {}, text);
    } catch (e) {
      if (destroyed || (e && e.name === 'AbortError') || !askBusy || askBusy.ctl !== ctl) return;
      askBusy = null;
      askErr = e && e.status === 400 && /unavailable/i.test((e.body && e.body.code) || '') ? `${askWho()} cannot answer questions right now: ${e.message}` : `Could not turn that into filters: ${e.message}`;
      renderAsk();
      if (document.activeElement !== searchInput) searchInput.focus();
    }
  }
  function applyAsk(r, text) {
    const res = askResult(r);
    const prev = aiApplied ? aiApplied.prev : { filters: { ...filters }, sort, order };
    ({ sort, order } = res.sort && (res.sort !== 'flag' || flagSortable(res.filters)) ? res : nextSort(filters, res.filters, sort, order));
    const cost = r.usage && r.usage.cost_usd ? r.usage.cost_usd : null;
    filters = res.filters;
    aiApplied = { prev, text, explanation: String(r.explanation || '').trim(), via: r.served_by || r.model || askWho(), cost, cached: !!r.cached,
      warnings: (Array.isArray(r.warnings) ? r.warnings : []).map(String).filter(Boolean) };
    aiApplied.state = stateQuery().toString();
    searchInput.blur();
    reload();
    vgrid.focus({ preventScroll: true });          // the grid's keys work at once; Esc there also undoes
  }
  /** Put back the filters (and sort) the list had before the AI search. */
  function restoreAsk() {
    if (!aiApplied) return false;
    const p = aiApplied.prev;
    aiApplied = null;
    filters = { ...p.filters }; sort = p.sort; order = p.order;
    reload();
    return true;
  }
  function aiNote() {
    const a = aiApplied;
    const cost = a.cached ? ' · same question as before, no charge' : a.cost != null ? ` · $${a.cost < 0.01 ? a.cost.toFixed(4) : a.cost.toFixed(2)}` : '';
    return h('div', { class: 'ai-note', role: 'status' }, icon('sparkles', 13),
      h('span', { class: 'ai-expl', title: `“${a.text}”${a.warnings.length ? `\n${a.warnings.join('\n')}` : ''}` }, a.explanation || `Filters for “${a.text}”`),
      h('span', { class: 'xs faint ai-via' }, `${a.via}${cost}`),
      // Parts of the question the server could not map (an unknown class or split) are said, not silently dropped.
      a.warnings.length ? h('span', { class: 'xs ai-warn', title: a.warnings.join('\n') }, `⚠ ${a.warnings[0]}${a.warnings.length > 1 ? ` (+${a.warnings.length - 1})` : ''}`) : '',
      h('button', { class: 'btn btn-ghost btn-sm', 'data-tip': 'Back to the filters you had before the AI search', onClick: () => restoreAsk() }, icon('undo', 12), 'Undo', h('span', { class: 'kbd' }, 'Esc')));
  }

  // ---------------------------------------------------------------- URL + session memory
  function stateQuery(extra = {}) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(filters)) if (v) p.set(k, v);
    if (sort !== defaultSort(filters)) p.set('sort', sort);
    if (order !== defaultOrder(sort)) p.set('order', order);
    for (const [k, v] of Object.entries(extra)) if (v != null && v !== '') p.set(k, v);
    return p;
  }
  function putOrder(p, f) {
    const o = linkOrder(f, sort, order);
    p.delete('sort'); p.delete('order');
    if (o) { p.set('sort', o.sort); p.set('order', o.order); }
    return p;
  }
  /** The list as a query for another page. */
  const portableQuery = (extra = {}) => putOrder(stateQuery(extra), filters);
  const focusedId = () => (cursor && focusIdx >= 0 && cursor.items[focusIdx] ? cursor.items[focusIdx].id : null);
  let urlTimer = null;
  function syncUrl(now = false) {
    clearTimeout(urlTimer);
    // Holding an arrow key must not flood the history API (browsers throttle more than ~100 calls per 10 s).
    if (!now) { urlTimer = setTimeout(() => syncUrl(true), 350); return; }
    if (destroyed) return;
    const id = inspect ? inspect.currentId() : focusedId();
    const p = stateQuery({ img: id, view: inspect ? 'inspect' : null });
    const keep = history.state;
    fovea.router.replaceQuery(Object.fromEntries(p));
    if (keep) history.replaceState(keep, '', location.pathname + location.search);   // keep the Inspect history marker
    ss.set(ssKey, stateQuery({ img: focusedId() }).toString());
  }
  let backUntil = 0;               // Inspect closed itself with history.back(): that entry's img is stale
  function applyQuery(q) {
    if (destroyed) return;
    const s = parseQuery(q);
    if (performance.now() < backUntil && !s.inspect && sameFilters(s.filters, filters) && s.sort === sort && s.order === order) { syncUrl(true); return; }
    if (!sameFilters(s.filters, filters) || s.sort !== sort || s.order !== order) {
      filters = s.filters; sort = s.sort; order = s.order;
      reload({ keepId: s.img, openInspect: s.inspect });
      return;
    }
    if (s.inspect && s.img) {
      if (!inspect) openInspectId(s.img, { push: false });
      else if (inspect.currentId() !== s.img) inspect.showId(s.img);
    } else if (inspect) inspect.close({ fromHistory: true });
    else if (s.img && s.img !== focusedId()) focusId(s.img);
    syncUrl(true);
  }
  // Same-tab navigations (header link, browser Back/Forward) do not re-render the tab: watch the router instead.
  const routeHook = (c) => {
    if (destroyed || !c.params || c.params.id !== ds.id || c.params.tab !== 'explore') return;
    const q = new URLSearchParams(c.query.toString());
    setTimeout(() => { if (!destroyed) applyQuery([...q.keys()].length ? q : new URLSearchParams(ss.get(ssKey) || '')); }, 0);
  };
  fovea.router.before.push(routeHook);

  // ---------------------------------------------------------------- stats (issue labels, dataset-wide counts)
  function loadStats() {
    return fovea.api.get(`/api/datasets/${ds.id}/stats`).then((r) => {
      if (destroyed) return;
      stats = r.stats || null;
      if (r.issue_labels) issueLabels = { ...issueLabels, ...r.issue_labels };
      renderChips();
      if (facetWatch) facetWatch();
    }).catch(() => {});
  }

  // ---------------------------------------------------------------- facets: counts per choice under the OTHER active filters
  // (standard faceted search: next to "val" stands how many images picking val would show with everything else as
  // it is). Fetched after the list settles (debounced), cached per filter set, and refetched more lazily after
  // reviews and label edits. A server without /images/facets falls back to dataset-wide counts from /stats.
  const facetKeyOf = (f) => JSON.stringify(Object.keys(f).filter(k => k !== 'seed').sort().map(k => [k, f[k]]));   // order does not change counts
  const facetCache = new Map();
  let facetTimer = null, facetCtl = null, facetsOff = false;
  function scheduleFacets(delay = FACET_DELAY) {
    clearTimeout(facetTimer);
    if (facetsOff || destroyed) return;
    if (!delay && facetCache.has(facetKeyOf(filters))) { loadFacets(); return; }
    facetTimer = setTimeout(loadFacets, delay);
  }
  async function loadFacets() {
    if (destroyed || facetsOff) return;
    const f = { ...filters }, key = facetKeyOf(f);
    delete f.seed;
    if (facetCache.has(key)) { useFacets(key, facetCache.get(key)); return; }
    if (facetsFor === key && facetCtl) return;            // already on its way
    if (facetCtl) facetCtl.abort();
    const ctl = facetCtl = new AbortController();
    try {
      const url = `/api/datasets/${ds.id}/images/facets`;
      const r = await fovea.api.get(url, f, { signal: ctl.signal });
      let fx = { ...(r.facets || r) };
      if (f.only_cls) {
        // "Any of" and "Only" are one choice: class counts must ignore both, not just the "Any of" filter.
        const g = { ...f }; delete g.only_cls; delete g.cls;
        const r2 = await fovea.api.get(url, g, { signal: ctl.signal });
        fx.cls = (r2.facets || r2).cls || {};
      }
      if (r.issue_labels) issueLabels = { ...issueLabels, ...r.issue_labels };
      facetCache.set(key, fx);
      if (facetCache.size > 24) facetCache.delete(facetCache.keys().next().value);
      if (ctl === facetCtl) { facetCtl = null; useFacets(key, fx); }
    } catch (e) {
      if (ctl === facetCtl) facetCtl = null;
      if (e && e.name === 'AbortError') return;
      if (e && (e.status === 404 || e.status === 405)) { facetsOff = true; if (facetWatch) facetWatch(); }
      // Anything else: keep the last counts (dimmed as stale by the popover).
    }
  }
  function useFacets(key, fx) {
    if (destroyed) return;
    facets = fx; facetsFor = key;
    updateSplitCounts();
    if (facetWatch) facetWatch();
  }
  /** Reviews and label edits change the counts: forget them, and refetch a little after the last change (a run of
      reviews coalesces) when something on screen shows them — an open popover, or split counts that depend on a
      review / label / AI filter. Otherwise the next popover fetches them. */
  const EDITABLE = ['review', 'exclude_review', 'labeled', 'has_label', 'cls', 'only_cls', 'issue', 'ai', 'min_boxes', 'max_boxes'];
  function facetsChanged(delay = FACET_DELAY_EDIT) {
    facetCache.clear(); facetsFor = '';
    if (facetWatch || Object.keys(filters).some(k => EDITABLE.includes(k))) scheduleFacets(delay);
  }

  // ---------------------------------------------------------------- QA flags after label edits
  // A label save changes QA flags only once the QA engine has re-checked it (about 3 s after the last save; a gap or
  // a jump also moves between neighbouring frames). Counts and tile flags read before that are stale, so after an
  // edit — here, in Annotate, or just before this view opened — poll the last edited image's /qa until its re-check
  // has run, then refetch the counts and re-read the flags of the edited images and the loaded frames of their
  // sequences.
  let qaTimer = null, qaTries = 0;
  const editsHere = () => { const now = Date.now(); return recentEdits.filter(x => (x.ds == null || x.ds === ds.id) && now - x.at < EDIT_WINDOW); };
  function watchQa() {
    if (destroyed) return;
    const ed = editsHere();
    if (!ed.length) return;
    clearTimeout(qaTimer); qaTries = 0;
    qaTimer = setTimeout(checkQa, Math.max(200, ed[ed.length - 1].at + QA_FIRST - Date.now()));
  }
  async function checkQa() {
    if (destroyed) return;
    const ed = editsHere(), last = ed[ed.length - 1];
    if (!last) return;
    let r = null;
    // A write without a known image (a filter-wide accept) is followed on the dataset's QA summary instead.
    try { r = last.id != null ? await fovea.api.get(`/api/datasets/${ds.id}/images/${last.id}/qa`) : await fovea.api.get(`/api/datasets/${ds.id}/qa/summary`, { auto: 0 }); }
    catch (e) {
      if (destroyed) return;
      // An image of another dataset (an edit made elsewhere) or one that is gone: forget it, look at the others.
      if (e && e.status === 404 && last.id != null) { const k = recentEdits.indexOf(last); if (k >= 0) recentEdits.splice(k, 1); checkQa(); return; }
      if (++qaTries < QA_TRIES) qaTimer = setTimeout(checkQa, QA_POLL);
      return;
    }
    if (destroyed) return;
    const busy = r && (r.refreshing || (last.id == null && r.status === 'running'));
    if (busy && ++qaTries < QA_TRIES) { qaTimer = setTimeout(checkQa, QA_POLL); return; }
    // Re-checked: the engine takes every pending edit of a dataset at once, so the edits before `last` are done too.
    for (const x of ed) { const k = recentEdits.indexOf(x); if (k >= 0) recentEdits.splice(k, 1); }
    facetsChanged(0);
    rereadFlags(ed.map(x => x.id).filter(id => id != null), ed.some(x => x.id == null));
  }
  /** Re-read the QA flags (and AI counts) of these images and of the loaded frames of their sequences; with
      `onScreen`, of every mounted tile too. */
  async function rereadFlags(ids, onScreen = false) {
    const c = cursor;
    if (!c || (!ids.length && !onScreen)) return;
    const want = new Set(), seqs = new Set();
    for (const id of ids) { const i = c.indexOfId(id); if (i >= 0) { want.add(id); if (c.items[i].seq) seqs.add(c.items[i].seq); } }
    if (onScreen) for (const [i, t] of mounted) if (!t._ph && c.items[i]) want.add(c.items[i].id);
    if (seqs.size) c.items.forEach((it) => { if (it && it.seq && seqs.has(it.seq) && want.size < 400) want.add(it.id); });
    if (!want.size) return;
    let r;
    try { r = await fovea.api.post(`/api/datasets/${ds.id}/images/by-ids`, { ids: [...want] }); } catch (e) { return; }
    if (c !== cursor || destroyed) return;
    for (const row of (r && r.items) || []) {
      const i = c.indexOfId(row.id);
      if (i < 0) continue;
      const it = c.items[i];
      c.items[i] = { ...it, flags: row.flags || [], issues: row.issues || it.issues, ai_pending: row.ai_pending ?? it.ai_pending, ai_suspect: row.ai_suspect ?? it.ai_suspect };
      if (matchesWith(c.items[i], filters)) gone.delete(row.id); else gone.add(row.id);
      refreshTile(i);
    }
    updateCount();
    if (inspect) inspect.redraw({ qa: true });
  }
  /** How many images picking `value` for filter `dim` would show, everything else as it is (facets); dataset-wide
      counts from /stats on a server without facets; null when unknown. */
  function facetCount(dim, value) {
    const v = facetValue(facets, dim, value);
    if (v != null) return v;
    if (!stats) return null;
    if (dim === 'cls') { const c = (stats.classes || []).find(x => x.cls === value); return c ? c.images || 0 : 0; }
    if (dim === 'issue' && value !== 'any') {
      const hit = [...(stats.issues || []), ...(stats.flags || [])].find(x => x.code === value);
      if (hit) return hit.count || 0;
      const a = (stats.ai && stats.ai.audit) || {}, v = a[value] ?? a[value.replace(/^ai_/, '')];
      return typeof v === 'number' ? v : (v && v.images) || 0;
    }
    return null;
  }
  /** Data-quality flags count only once the QA engine has run on this dataset. */
  const issueCodeCounted = (code) => !!((facets && facets.issue && code in facets.issue) || (stats && Array.isArray(stats.flags) && (stats.flags.length || stats.qa_computed_at)));

  // ---------------------------------------------------------------- list loading
  /** Re-query the list. The current grid stays on screen until the new first page and the visible thumbnails
      are ready, then everything swaps in one frame (no blank grid, no tiles popping in one by one). */
  async function reload({ keepId = null, keepScroll = false, openInspect: reopen = false, keepSelection = false } = {}) {
    const my = ++gen;
    if (inspect) inspect.close({ fromHistory: true, fromReload: true });
    updateControls();
    if (!keepSelection) clearSelection(true);
    countEl.textContent = '…';
    const c = new fovea.ImageCursor({ dsId: ds.id, filters: { ...filters }, sort, order, pageSize: PAGE, boxes: true });
    const waitEl = cursor ? null : showLoading();
    let pos = -1;
    try {
      await c.loadPage(0);
      if (keepId != null) pos = await c.positionOf(keepId);
    } catch (e) {
      if (my !== gen || destroyed) return;
      if (waitEl) waitEl.remove();
      showError(e);
      return;
    }
    if (my !== gen || destroyed) return;
    if (waitEl) waitEl.remove();
    // Lay out the new list without touching the grid on screen yet.
    const n = c.total || 0, NL = computeLayout(n);
    let top = 0;                          // the kept image (centred), the old position (refresh), or the top
    if (pos >= 0) top = rowTopIn(NL, pos) - Math.max(0, (body.clientHeight - NL.tileH) / 2);
    else if (keepScroll) top = body.scrollTop;
    top = clamp(top, 0, Math.max(0, gridHeight(NL) - body.clientHeight));
    await preloadVisible(c, NL, n, top);
    if (my !== gen || destroyed) return;
    // Swap in one frame.
    imgLoader.reset();
    for (const t of mounted.values()) t.remove();
    mounted.clear();
    cursor = c; total = n; gone.clear(); focusIdx = -1; anchorIdx = -1;
    applyLayout(NL);
    emptyEl.classList.add('hidden'); vgrid.classList.remove('hidden');
    if (!total) renderEmpty();
    body.scrollTop = top;
    renderWindow();
    if (filters.set && Object.keys(filters).length === 1) { setCount = { token: filters.set, count: total }; renderChips(); }
    updateCount();
    if (facetWatch) facetWatch();          // an open popover's lines that read the list's count
    loadProgress(my);
    scheduleFacets();
    if (pos >= 0) setFocus(pos, { scroll: false, dom: !reopen });
    if (reopen && pos >= 0) openInspect(pos, { push: false });
    syncUrl(true);
  }
  function showLoading() { const w = h('div', { class: 'explore-loading' }, ui.spinner()); body.appendChild(w); return w; }
  function showError(e) {
    countEl.textContent = '—';
    for (const t of mounted.values()) t.remove();
    mounted.clear(); cursor = null; total = null;
    vgrid.style.height = '0px';
    emptyEl.classList.remove('hidden');
    // A selection set lives 24 h on the server: an expired one must not leave the view stuck on an error.
    const setGone = filters.set && e && (e.status === 404 || e.status === 410 || e.status === 400);
    emptyEl.replaceChildren(ui.emptyState({ icon: 'alert', title: setGone ? 'This selection is no longer available' : 'Could not load images',
      message: setGone ? 'Selections are kept on the server for 24 hours. Select the images again, or show the whole list.' : e.message,
      action: setGone ? h('button', { class: 'btn btn-primary', onClick: () => setFilter('set', '') }, 'Show without the selection')
        : h('button', { class: 'btn btn-primary', onClick: () => reload() }, 'Retry') }));
  }
  function renderEmpty() {
    vgrid.classList.add('hidden');
    emptyEl.classList.remove('hidden');
    const scanning = ds.status === 'scanning' || (ds.job && ['queued', 'running'].includes(ds.job.status));
    if (activeCount()) {
      emptyEl.replaceChildren(ui.emptyState({ icon: 'filter', title: 'No images match these filters',
        message: 'Remove a filter, or clear them all.',
        action: h('div', { class: 'col', style: 'align-items:center' }, h('div', { class: 'row wrap', style: 'justify-content:center' }, filterChips()),
          h('button', { class: 'btn btn-primary', onClick: () => clearFilters() }, 'Clear filters')) }));
    } else if (scanning) {
      emptyEl.replaceChildren(ui.emptyState({ icon: 'refresh', title: 'Indexing…', message: 'Images appear here when the index is ready.' }));
    } else {
      emptyEl.replaceChildren(ui.emptyState({ icon: 'images', title: 'This dataset has no images', message: 'If files were added since, rescan it from the dataset menu (⋯).' }));
    }
  }
  /** Decode the thumbnails of the tiles that will be visible, up to ~300 ms, so they appear together. */
  async function preloadVisible(c, NL, n, top) {
    if (!n) return;
    const [a, b] = rangeFor(NL, n, top, 0);
    const urls = [];
    for (let i = a; i < b && urls.length < 48; i++) { const it = c.items[i]; if (it) urls.push(fovea.api.thumbUrl(it.id, NL.thumb, it.mtime)); }
    const todo = urls.filter(u => !imgLoader.isLoaded(u));
    if (!todo.length) return;
    const all = Promise.all(todo.map(u => new Promise((res) => {
      const im = new Image(); im.decoding = 'async'; im.setAttribute('fetchpriority', 'high');
      im.onload = () => { (im.decode ? im.decode() : Promise.resolve()).then(() => { imgLoader.prime(u); res(); }, () => res()); };
      im.onerror = () => res();
      im.src = u;
    })));
    await Promise.race([all, new Promise(r => setTimeout(r, 300))]);
  }

  // ---------------------------------------------------------------- virtual grid layout
  function computeLayout(n) {
    const w = body.clientWidth || el.clientWidth || 1000;
    const inner = Math.max(120, w - PAD_X * 2);
    const min = TILE_PX[tileSize];
    const cols = Math.max(1, Math.floor((inner + GAP) / (min + GAP)));
    const tileW = Math.floor((inner - GAP * (cols - 1)) / cols);
    const frameH = Math.round(tileW * 0.75);
    const tileH = frameH + META_H;
    return { cols, tileW, frameH, tileH, rowH: tileH + GAP, rows: n ? Math.ceil(n / cols) : 0, thumb: thumbPx(tileW) };
  }
  const gridHeight = (X) => (X.rows ? PAD_TOP + X.rows * X.rowH - GAP + PAD_BOTTOM : 0);
  function applyLayout(X) {
    const prev = L;
    L = X;
    vgrid.style.height = `${gridHeight(X)}px`;
    vgrid.style.setProperty('--tw', `${X.tileW}px`); vgrid.style.setProperty('--th', `${X.tileH}px`); vgrid.style.setProperty('--fh', `${X.frameH}px`);
    return !prev || prev.cols !== X.cols || prev.tileW !== X.tileW;
  }
  const layoutGrid = () => applyLayout(computeLayout(total || 0));
  const rowTopIn = (X, i) => PAD_TOP + Math.floor(i / X.cols) * X.rowH;
  const rowTop = (i) => rowTopIn(L, i);
  function placeTile(t, i) { const c = i % L.cols; t.style.transform = `translate(${PAD_X + c * (L.tileW + GAP)}px,${rowTop(i)}px)`; }
  function rangeFor(X, n, top, over) {
    const vh = body.clientHeight || 600, ov = vh * over;
    const r0 = Math.max(0, Math.floor((top - PAD_TOP - ov) / X.rowH));
    const r1 = Math.min(X.rows - 1, Math.floor((top - PAD_TOP + vh + ov) / X.rowH));
    return r1 < r0 ? [0, 0] : [r0 * X.cols, Math.min(n, (r1 + 1) * X.cols)];
  }
  const rangeAt = (top, over) => rangeFor(L, total, top, over);
  /** Mount the tiles for the rows near the viewport, drop the rest, and ask for the pages they need. */
  function renderWindow() {
    if (!cursor || !total || !L) return;
    const [a, b] = rangeAt(body.scrollTop, OVERSCAN);
    for (const [i, t] of mounted) if ((i < a || i >= b) && i !== focusIdx) unmount(i, t);
    const frag = document.createDocumentFragment(), fresh = [];
    const want = [];
    for (let i = a; i < b; i++) want.push(i);
    if (focusIdx >= 0 && focusIdx < total && (focusIdx < a || focusIdx >= b)) want.push(focusIdx);   // keeps keyboard focus alive
    let refocus = null;
    for (const i of want) {
      const it = cursor.items[i];
      let t = mounted.get(i);
      if (t && t._ph && it) { if (t.contains(document.activeElement)) refocus = i; unmount(i, t); t = null; }
      if (t) continue;
      t = it ? tile(i, it) : placeholder(i);
      mounted.set(i, t); frag.appendChild(t);
      if (it) fresh.push(t);
    }
    if (frag.firstChild) vgrid.appendChild(frag);
    for (const t of fresh) imgLoader.observe(t._img, t._pic, t);
    if (refocus != null) { const t = mounted.get(refocus); if (t) t.focus({ preventScroll: true }); }
    requestPages();
  }
  function unmount(i, t) { imgLoader.unobserve(t); t.remove(); mounted.delete(i); }
  function relayout() {
    if (!cursor || !total) { layoutGrid(); return; }
    const anchor = focusIdx >= 0 && mounted.has(focusIdx) && isOnScreen(focusIdx) ? focusIdx : firstVisibleIndex();
    const offset = anchor >= 0 ? rowTop(anchor) - body.scrollTop : 0;
    const oldThumb = L && L.thumb;
    if (!layoutGrid()) { renderWindow(); return; }
    for (const [i, t] of mounted) placeTile(t, i);
    if (L.thumb !== oldThumb) {
      for (const [i, t] of mounted) { const it = !t._ph && cursor.items[i]; if (it) imgLoader.retarget(t, fovea.api.thumbUrl(it.id, L.thumb, it.mtime)); }
    }
    if (anchor >= 0) body.scrollTop = Math.max(0, rowTop(anchor) - offset);
    renderWindow();
  }
  function firstVisibleIndex() { if (!L || !total) return -1; return clamp(Math.floor((body.scrollTop - PAD_TOP) / L.rowH) * L.cols, 0, total - 1); }
  function isOnScreen(i) { const y = rowTop(i); return y + L.tileH > body.scrollTop && y < body.scrollTop + body.clientHeight; }

  // Pages for the rows on screen. A normal scroll fetches as it goes (at most every 120 ms); while the scrollbar
  // is dragged or a flick is still flying, nothing is fetched for the rows rushing past — the request goes out
  // once the movement slows down.
  let pageTimer = null, lastScrollAt = 0, lastTop = 0, scrollSpeed = 0, lastPageLoad = 0, pageErrorShown = false;
  function requestPages() {
    clearTimeout(pageTimer);
    const now = performance.now();
    const fast = now - lastScrollAt < 100 && scrollSpeed > 6;      // px per ms: about four screens a second
    const run = () => { lastPageLoad = performance.now(); loadPages(); };
    if (!fast && now - lastPageLoad >= 120) run();
    else pageTimer = setTimeout(run, fast ? 140 : 120 - (now - lastPageLoad));
  }
  function loadPages() {
    const c = cursor;
    if (!c || !total || destroyed) return;
    const [a, b] = rangeAt(body.scrollTop, 0.5);
    if (b <= a) return;
    for (let p = Math.floor(a / PAGE); p <= Math.floor((b - 1) / PAGE); p++) {
      const s = p * PAGE, e = Math.min(total, s + PAGE) - 1;
      if (c.items[s] && c.items[e]) continue;
      c.loadPage(p).then(() => { if (c === cursor && !destroyed) onPage(); }, (err) => {
        if (c !== cursor || destroyed) return;
        if (!pageErrorShown) { pageErrorShown = true; ui.toast(`Could not load images: ${err.message}`, { type: 'error' }); setTimeout(() => { pageErrorShown = false; }, 5000); }
      });
    }
  }
  function onPage() {
    if (cursor.total != null && cursor.total !== total) {
      total = cursor.total;
      relayout(); updateCount();
      if (!total) renderEmpty();
    } else renderWindow();
  }
  let scrollTick = false;
  body.addEventListener('scroll', () => {
    const now = performance.now(), top = body.scrollTop, dt = now - lastScrollAt;
    const v = dt > 0 && dt < 200 ? Math.abs(top - lastTop) / dt : 0;
    scrollSpeed = scrollSpeed * 0.5 + v * 0.5;
    lastScrollAt = now; lastTop = top;
    if (scrollTick) return;
    scrollTick = true;
    requestAnimationFrame(() => { scrollTick = false; if (!destroyed) renderWindow(); });
  }, { passive: true });
  const ro = new ResizeObserver(() => { if (destroyed) return; relayout(); if (!gridStatus.classList.contains('hidden')) sayInGrid(''); });   // its place was measured
  ro.observe(body);
  const imgLoader = createImageLoader({ root: body, margin: '200px 0px', maxInflight: 6 });

  // ---------------------------------------------------------------- tiles
  function placeholder(i) {
    const t = h('div', { class: 'tile ph', 'data-idx': i, role: 'option', 'aria-busy': 'true', 'aria-posinset': i + 1, 'aria-setsize': total, tabindex: i === focusIdx ? '0' : '-1' },
      h('div', { class: 'frame' }), h('div', { class: 'meta' }));
    t._ph = true;
    if (i === focusIdx) t.classList.add('focus');
    placeTile(t, i);
    return t;
  }
  function tileLabel(it) {
    const s = reviewStatus(it);
    return `${it.rel_path.split('/').pop()}, ${it.n_boxes || 0} box${it.n_boxes === 1 ? '' : 'es'}${s ? `, ${REVIEW[s].label.toLowerCase()}` : ''}${noteOf(it) ? ', has a note' : ''}`;
  }
  function tile(i, it) {
    const sel = isSelected(it.id);
    const t = h('div', { class: cls('tile', sel && 'selected', i === focusIdx && 'focus', gone.has(it.id) && 'gone'), role: 'option', 'aria-selected': String(sel),
      'aria-posinset': i + 1, 'aria-setsize': total, 'aria-label': tileLabel(it), tabindex: i === focusIdx ? '0' : '-1', 'data-idx': i, 'data-review': reviewStatus(it) || '' });
    placeTile(t, i);
    const img = h('img', { decoding: 'async', alt: '', draggable: false, dataset: { src: fovea.api.thumbUrl(it.id, L.thumb, it.mtime) } });
    const pic = h('div', { class: 'pic is-idle' }, img);
    if (it.boxes && it.boxes.length) pic.appendChild(tileOverlay(it.boxes));
    const wrap = h('div', { class: 'pic-wrap' }, pic);
    shapeWrap(wrap, it.width || 4, it.height || 3);
    // The stored size can disagree with the picture (unknown at index time, or pre-rotation EXIF): the decoded
    // thumbnail has the true proportions — use them, or the boxes land beside the objects.
    img.addEventListener('load', () => {
      const nw = img.naturalWidth, nh = img.naturalHeight;
      if (!nw || !nh) return;
      const stored = it.width && it.height ? it.width / it.height : 0;
      if (!stored || Math.abs(stored - nw / nh) / (nw / nh) >= 0.02) shapeWrap(wrap, nw, nh);
    });
    t.append(h('div', { class: 'frame' }, wrap), badges(it), h('div', { class: 'rv', 'aria-hidden': 'true' }, reviewStatus(it) ? icon(REVIEW[reviewStatus(it)].icon, 12) : null),
      h('div', { class: 'sel', 'aria-hidden': 'true' }), meta(it));
    t._img = img; t._pic = pic; t._id = it.id;
    return t;
  }
  function shapeWrap(wrap, w, hh) {
    const r = w / hh;
    wrap.style.cssText = `${r >= 4 / 3 ? 'width:100%;height:auto;' : 'height:100%;width:auto;'}aspect-ratio:${w}/${hh}`;
  }
  /** Boxes of one tile as ONE svg with one path per class (not a div per box): a crowded page used to add 30k nodes. */
  function tileOverlay(boxes) {
    const by = new Map();
    for (const b of boxes) {
      if (isolate != null && b[0] !== isolate) continue;
      const x = (b[1] - b[3] / 2) * 1000, y = (b[2] - b[4] / 2) * 1000, w = b[3] * 1000, hh = b[4] * 1000;
      by.set(b[0], (by.get(b[0]) || '') + `M${x.toFixed(1)} ${y.toFixed(1)}h${w.toFixed(1)}v${hh.toFixed(1)}h${(-w).toFixed(1)}z`);
    }
    const s = fovea.svg('svg', { class: 'tov', viewBox: '0 0 1000 1000', preserveAspectRatio: 'none', 'aria-hidden': 'true' });
    for (const [c, d] of by) { const col = classColor(c); s.appendChild(fovea.svg('path', { d, fill: col, 'fill-opacity': '0.12', stroke: col, 'stroke-width': '1.5', 'vector-effect': 'non-scaling-stroke' })); }
    return s;
  }
  function problemCodes(it) {
    const out = (it.issues || []).filter(c => c !== 'missing_label');
    for (const f of it.flags || []) if (!out.includes(f)) out.push(f);
    return out;
  }
  // J's "label problem" = the labels may be wrong: AI audit findings, label-file issues and the label-related QA
  // flags (sequence gaps and jumps, odd shapes…) — wider than Overview's "suspect images", which are AI audit
  // findings only; the keymap and hints say so. Image-level flags (near-duplicate, blurry, dark…) are not label
  // errors and would make J stop everywhere.
  const isSuspect = (it) => !!it && ((it.ai_suspect || 0) > 0 || problemCodes(it).some(c => !IMAGE_FLAGS.has(c)));
  function badges(it) {
    return h('div', { class: 'badges' },
      it.n_boxes ? h('span', { class: 'badge' }, String(it.n_boxes)) : null,
      it.split && splits.length > 1 && !filters.split ? h('span', { class: 'badge split', title: it.split }, it.split) : null,
      it.ai_pending ? h('span', { class: 'badge ai', 'data-tip': `${it.ai_pending} AI proposal${it.ai_pending > 1 ? 's' : ''} waiting for a decision` }, `◌ ${it.ai_pending}`) : null,
      it.ai_suspect ? h('span', { class: 'badge suspect', 'data-tip': `${it.ai_suspect} suspected label error${it.ai_suspect > 1 ? 's' : ''} (AI audit)` }, `? ${it.ai_suspect}`) : null);
  }
  function nameEl(base, full) {
    // Middle ellipsis: the end of a filename (frame number, extension) is what tells neighbours apart.
    const cut = Math.max(0, base.length - 10);
    return h('span', { class: 'name', title: full }, h('span', { class: 'head' }, base.slice(0, cut)), h('span', { class: 'tail' }, base.slice(cut)));
  }
  function meta(it) {
    const probs = problemCodes(it);
    return h('div', { class: 'meta' },
      probs.length ? h('span', { class: 'issue-dot', 'data-tip': probs.map(c => issueLabels[c] || c).join(', ') }) : null,
      nameEl(it.rel_path.split('/').pop(), it.rel_path),
      it.width ? h('span', { class: 'dims' }, `${it.width}×${it.height}`) : null);
  }
  /** Rebuild one mounted tile in place (labels changed). The picture element is kept so it does not blink. */
  function refreshTile(i) {
    const old = mounted.get(i), it = cursor && cursor.items[i];
    if (!old || !it || old._ph) { renderWindow(); return; }
    const pic = old._pic;
    const ov = pic.querySelector('.tov'); if (ov) ov.remove();
    if (it.boxes && it.boxes.length) pic.appendChild(tileOverlay(it.boxes));
    old.querySelector('.badges').replaceWith(badges(it));
    old.querySelector('.meta').replaceWith(meta(it));
    updateTileReview(i);
  }
  function updateTileReview(i) {
    const t = mounted.get(i), it = cursor && cursor.items[i];
    if (!t || !it || t._ph) return;
    const s = reviewStatus(it);
    t.dataset.review = s || '';
    t.classList.toggle('gone', gone.has(it.id));
    t.setAttribute('aria-label', tileLabel(it));
    t.querySelector('.rv').replaceChildren(...(s ? [icon(REVIEW[s].icon, 12)] : []));
  }
  function redrawOverlays() {
    for (const [i, t] of mounted) {
      const it = !t._ph && cursor.items[i]; if (!it) continue;
      const ov = t._pic.querySelector('.tov'); if (ov) ov.remove();
      if (it.boxes && it.boxes.length) t._pic.appendChild(tileOverlay(it.boxes));
    }
  }

  // ---------------------------------------------------------------- grid pointer events (one delegated listener each)
  vgrid.addEventListener('click', (e) => {
    const t = e.target.closest('.tile');
    if (!t || !vgrid.contains(t)) return;
    const i = Number(t.dataset.idx);
    if (t._ph) { setFocus(i, { scroll: false, dom: false }); openInspect(i); return; }   // Inspect loads it
    if (e.target.closest('.sel')) { toggleSelect(i, e.shiftKey ? 'range' : 'toggle'); return; }
    if (e.shiftKey) { toggleSelect(i, 'range'); return; }
    if (e.metaKey || e.ctrlKey) { toggleSelect(i, 'toggle'); return; }
    setFocus(i, { scroll: false, dom: false });
    openInspect(i);
  });
  vgrid.addEventListener('mousedown', (e) => { if (e.shiftKey) e.preventDefault(); });   // no text selection on shift-click

  // ---------------------------------------------------------------- rubber-band selection: drag from empty grid space
  // A drag that starts between tiles (gaps, margins, below the last row) draws a selection rectangle; one that starts
  // on a tile is a click as before, and the scrollbar is left alone. Hits come from layout arithmetic (no DOM
  // queries, so it works for rows that are not mounted), the list scrolls while the pointer is near the top or bottom
  // edge, and pages that are not loaded yet are fetched when the drag ends. ⇧ or ⌘/Ctrl adds to the selection.
  let band = null;
  body.addEventListener('pointerdown', (e) => {
    if (e.button !== 0 || e.pointerType === 'touch' || band || inspect || !cursor || !total || !L) return;
    if (e.target.closest('.tile, .explore-empty, button, a, input')) return;
    const br = body.getBoundingClientRect();
    if (e.clientX - br.left >= body.clientWidth || e.clientY - br.top >= body.clientHeight) return;   // the scrollbar
    const g = vgrid.getBoundingClientRect();
    band = { id: e.pointerId, x0: e.clientX - g.left, y0: e.clientY - g.top, cx: e.clientX, cy: e.clientY, live: false,
      add: e.shiftKey || e.metaKey || e.ctrlKey, hit: new Set(), el: null, raf: 0 };
    e.preventDefault();                                    // no text selection while dragging
  });
  window.addEventListener('pointermove', onBandMove);
  window.addEventListener('pointerup', onBandUp);
  window.addEventListener('pointercancel', endBand);
  function onBandMove(e) {
    if (!band || e.pointerId !== band.id) return;
    band.cx = e.clientX; band.cy = e.clientY;
    if (!band.live) {
      const g = vgrid.getBoundingClientRect();
      if (Math.hypot(e.clientX - g.left - band.x0, e.clientY - g.top - band.y0) < BAND_MIN) return;
      band.live = true;
      band.el = h('div', { class: 'band', 'aria-hidden': 'true' });
      vgrid.appendChild(band.el);
      vgrid.classList.add('banding');
      band.raf = requestAnimationFrame(bandTick);
    }
    drawBand();
  }
  /** Keep scrolling while the pointer rests near the top or bottom edge (speed grows with closeness). */
  function bandTick() {
    if (!band || !band.live) return;
    const br = body.getBoundingClientRect(), edge = 36;
    const d = band.cy < br.top + edge ? band.cy - (br.top + edge) : band.cy > br.bottom - edge ? band.cy - (br.bottom - edge) : 0;
    if (d) { body.scrollTop += clamp(d * 0.6, -28, 28); drawBand(); }
    band.raf = requestAnimationFrame(bandTick);
  }
  function drawBand() {
    if (!band || !band.live || !L) return;
    const g = vgrid.getBoundingClientRect();
    const x1 = clamp(band.cx - g.left, 0, vgrid.clientWidth), y1 = clamp(band.cy - g.top, 0, gridHeight(L));
    const x = Math.min(band.x0, x1), y = Math.min(band.y0, y1), w = Math.abs(x1 - band.x0), hh = Math.abs(y1 - band.y0);
    band.el.style.cssText = `transform:translate(${x}px,${y}px);width:${w}px;height:${hh}px`;
    band.hit = bandHits(x, y, x + w, y + hh);
    paintBand();
  }
  const bandHits = (x0, y0, x1, y1) => gridHits({ ...L, padX: PAD_X, padTop: PAD_TOP, gap: GAP }, total, x0, y0, x1, y1);
  /** Show the selection the drag would make on the mounted tiles (nothing is stored until the drag ends). */
  function paintBand() {
    for (const [i, t] of mounted) {
      const it = !t._ph && cursor.items[i];
      const on = band.hit.has(i) || (band.add && !!it && isSelected(it.id));
      t.classList.toggle('selected', on); t.setAttribute('aria-selected', String(on));
    }
    vgrid.classList.toggle('selecting', band.hit.size > 0 || selCount() > 0);
  }
  async function onBandUp(e) {
    if (!band || e.pointerId !== band.id) return;
    const b = band;
    endBand({ keepPreview: true });                        // the preview stays until the selection below replaces it
    if (!b.live) return;                                   // a plain click between tiles: nothing to do
    const c = cursor;
    if (!b.hit.size) { if (!b.add) clearSelection(); else refreshSelection(); return; }
    let lo = Infinity, hi = -1;
    for (const i of b.hit) { if (i < lo) lo = i; if (i > hi) hi = i; }
    await ensureRange(c, lo, hi);
    if (c !== cursor || destroyed) return;
    const ids = [...b.hit].sort((a, z) => a - z).map(i => c.items[i]).filter(Boolean).map(it => it.id);
    if (allMode && b.add) ids.forEach(id => except.delete(id));
    else if (b.add) ids.forEach(id => selected.add(id));
    else { allMode = false; except = new Set(); selected = new Set(ids); }
    anchorIdx = hi; anchorBase = new Set(selected); anchorBaseExcept = new Set(except);
    refreshSelection();
    vgrid.focus({ preventScroll: true });
  }
  function endBand({ keepPreview = false } = {}) {
    if (!band) return;
    cancelAnimationFrame(band.raf);
    if (band.el) band.el.remove();
    vgrid.classList.remove('banding');
    const was = band; band = null;
    if (was.live && cursor && !keepPreview) refreshSelection();   // drop the preview; the stored selection is redrawn
  }
  vgrid.addEventListener('contextmenu', (e) => {
    const t = e.target.closest('.tile');
    if (!t || t._ph) return;
    e.preventDefault();
    tileMenu({ x: e.clientX, y: e.clientY }, Number(t.dataset.idx));
  });

  function tileMenu(pos, i) {
    const it = cursor.items[i]; if (!it) return;
    // Right-click on a selected tile acts on the whole selection, and says so.
    const inSel = isSelected(it.id) && selCount() > 1;
    const n = inSel ? selCount() : 1;
    const sfx = inSel ? ` ${fmt.num(n)} images` : '';
    const target = () => (inSel ? selectionTarget() : { ids: [it.id], count: 1 });
    ui.menu(pos, [
      { label: 'Inspect', icon: 'eye', onClick: () => openInspect(i), kbd: '↵' },
      { label: inSel ? `Annotate${sfx}` : 'Annotate', icon: 'pen', onClick: () => (inSel ? annotateSelection() : annotateId(it.id)), kbd: 'E' },
      { sep: true },
      { label: `Approve${sfx}`, icon: 'check', onClick: () => review(target(), 'approved', { quiet: !inSel }), kbd: 'A' },
      { label: `Flag${sfx}`, icon: 'flag', onClick: () => review(target(), 'flagged', { quiet: !inSel }), kbd: 'F' },
      { label: `Exclude${sfx}`, icon: 'ban', onClick: () => review(target(), 'excluded', { quiet: !inSel }), kbd: 'X' },
      { label: `Clear review${sfx}`, icon: 'rotate', onClick: () => review(target(), null, { quiet: !inSel }), kbd: 'U' },
      { sep: true },
      { label: inSel ? `Copy${sfx} paths` : 'Copy path', icon: 'copy', onClick: () => (inSel ? copyPaths(selectionTarget()) : copyPath(it.id)) },
      { label: 'Open original', icon: 'external', onClick: () => window.open(fovea.api.imgUrl(it.id), '_blank') },
      it.seq ? { label: 'Show this sequence', icon: 'film', onClick: () => setFilters({ seq: it.seq }, { keepId: it.id }) } : null,
    ]);
  }

  // ---------------------------------------------------------------- selection
  const isSelected = (id) => (allMode ? !except.has(id) : selected.has(id));
  const selCount = () => (allMode ? Math.max(0, (total || 0) - except.size) : selected.size);
  function selectionTarget() { return allMode ? { all: true, except: new Set(except), count: selCount() } : { ids: [...selected], count: selected.size }; }
  async function toggleSelect(i, mode) {
    const c = cursor; const it = c && c.items[i]; if (!it) return;
    if (mode === 'range' && anchorIdx >= 0) {
      // Anchor-based like a file manager: each shift-click REPLACES the previous range from the same anchor.
      const a = Math.min(anchorIdx, i), b = Math.max(anchorIdx, i);
      if (b - a > 2000 && !(await ui.confirm({ title: 'Select a large range?', message: `This selects ${fmt.num(b - a + 1)} images.`, okLabel: 'Select' }))) return;
      await ensureRange(c, a, b);
      if (c !== cursor) return;
      if (allMode) { const ex = new Set(anchorBaseExcept); for (let k = a; k <= b; k++) if (c.items[k]) ex.delete(c.items[k].id); except = ex; }
      else { const s = new Set(anchorBase); for (let k = a; k <= b; k++) if (c.items[k]) s.add(c.items[k].id); selected = s; }
    } else {
      if (allMode) { except.has(it.id) ? except.delete(it.id) : except.add(it.id); }
      else { selected.has(it.id) ? selected.delete(it.id) : selected.add(it.id); }
      anchorIdx = i; anchorBase = new Set(selected); anchorBaseExcept = new Set(except);
    }
    refreshSelection();
  }
  async function ensureRange(c, a, b) {
    const pages = [];
    for (let p = Math.floor(a / PAGE); p <= Math.floor(b / PAGE); p++) { const s = p * PAGE; if (!c.items[s] || !c.items[Math.min(total, s + PAGE) - 1]) pages.push(p); }
    for (let k = 0; k < pages.length; k += 4) await Promise.all(pages.slice(k, k + 4).map(p => c.loadPage(p).catch(() => null)));
  }
  function selectAllMatching() {
    if (!total) return;
    allMode = true; except = new Set(); selected = new Set(); anchorIdx = -1;
    refreshSelection();
  }
  function clearSelection(silent = false) {
    allMode = false; except = new Set(); selected = new Set(); anchorIdx = -1;
    if (!silent) refreshSelection(); else { vgrid.classList.remove('selecting'); bulk.classList.add('hidden'); for (const t of mounted.values()) { t.classList.remove('selected'); t.setAttribute('aria-selected', 'false'); } }
  }
  function refreshSelection() {
    for (const [i, t] of mounted) {
      const it = !t._ph && cursor.items[i];
      const on = !!it && isSelected(it.id);
      t.classList.toggle('selected', on); t.setAttribute('aria-selected', String(on));
    }
    vgrid.classList.toggle('selecting', selCount() > 0);
    updateBulk();
  }

  // ---------------------------------------------------------------- bulk bar
  let bulkBusy = null;   // a node shown instead of the actions while a bulk operation prepares or runs
  function updateBulk() {
    const n = selCount();
    bulk.classList.toggle('hidden', !n && !bulkBusy);
    if (!n && !bulkBusy) return;
    if (bulkBusy) { bulk.replaceChildren(bulkBusy); return; }
    const label = allMode ? (except.size ? `All ${fmt.num(total)} matching except ${fmt.num(except.size)}` : `All ${fmt.num(total)} matching selected`) : `${fmt.num(n)} selected`;
    const t = () => selectionTarget();
    bulk.replaceChildren(
      h('span', { class: 'strong' }, label),
      !allMode && total > n ? h('button', { class: 'btn btn-link', onClick: () => selectAllMatching() }, `Select all ${fmt.num(total)}`) : '',   // DOM replaceChildren prints null as text
      h('span', { class: 'sep' }),
      h('button', { class: 'btn', onClick: () => review(t(), 'approved') }, icon('check', 13), 'Approve'),
      h('button', { class: 'btn', 'data-tip': 'Approve only the unreviewed ones; flagged and excluded stay (⇧A)', onClick: () => review(t(), 'approved', { onlyUnreviewed: true }) }, 'Approve unreviewed'),
      h('button', { class: 'btn', onClick: () => review(t(), 'flagged') }, icon('flag', 13), 'Flag'),
      h('button', { class: 'btn', onClick: () => review(t(), 'excluded') }, icon('ban', 13), 'Exclude'),
      h('button', { class: 'btn', onClick: () => review(t(), null) }, icon('rotate', 13), 'Clear'),
      h('span', { class: 'sep' }),
      h('button', { class: 'btn', 'aria-haspopup': 'menu', onClick: (e) => proposalsMenu(e.currentTarget) }, icon('sparkles', 13), 'AI', icon('chevronDown', 11)),
      h('button', { class: 'btn', onClick: () => annotateSelection() }, icon('pen', 13), 'Annotate'),
      h('button', { class: 'btn btn-icon', 'aria-label': 'More: show only these, copy paths, download', 'aria-haspopup': 'menu', 'data-tip': 'Show only these · copy paths · download list or ZIP', onClick: (e) => moreMenu(e.currentTarget) }, icon('moreH', 13)),
      h('button', { class: 'btn btn-icon', 'aria-label': 'Deselect', 'data-tip': 'Deselect (Esc)', onClick: () => clearSelection() }, icon('x', 13)));
  }
  function setBusy(text) {
    if (text == null) { bulkBusy = null; updateBulk(); return null; }
    const lbl = h('span', text);
    bulkBusy = h('span', { class: 'row' }, ui.spinner(), lbl, ui.elapsedClock());
    updateBulk();
    return lbl;
  }

  // ---------------------------------------------------------------- resolving targets
  const byIdItem = (id) => { const i = cursor ? cursor.indexOfId(id) : -1; return i >= 0 ? cursor.items[i] : null; };
  async function countWith(f) { const r = await fovea.api.get(`/api/datasets/${ds.id}/images`, { ...f, limit: 1, offset: 0 }); return r.total || 0; }
  /** Every image of the current list (minus exceptions), with review state: exact counts, overwrite checks
      and undo for "all matching". Over RESOLVE_MAX the images are frozen into a server-side set instead
      (returns {filters: {set}, match, count}; `match` is the filter the loaded tiles are checked against). */
  async function resolveAll(target) {
    const f = target.filters || filters;
    const n = await countWith(f);
    if (n > RESOLVE_MAX) {
      // Too many to list one by one here: a set holds exactly these images (exceptions removed) on the server.
      const lbl = setBusy(`Preparing ${fmt.num(n)} images…`);
      let tf = null;
      try { tf = await setTarget({ all: true, filters: f, except: target.except || new Set() }); } finally { if (lbl) setBusy(null); }
      if (tf) return { filters: tf.filters, match: { ...f }, count: tf.count ?? n };
      // A server without sets: the server applies the filter itself, which cannot leave single images out.
      if (target.except && target.except.size) throw new Error(`More than ${fmt.num(RESOLVE_MAX)} images: deselecting single images is not possible on this server — narrow the filter instead.`);
      return { filters: { ...f }, count: n };
    }
    const lbl = setBusy(`Preparing ${fmt.num(n)} images…`);
    try {
      const items = [];
      for (let off = 0; off < n; off += 1000) {
        const r = await fovea.api.get(`/api/datasets/${ds.id}/images`, { ...f, sort, order, offset: off, limit: 1000 });
        items.push(...r.items);
        if (lbl) lbl.textContent = `Preparing ${fmt.num(Math.min(n, off + 1000))} / ${fmt.num(n)}…`;
        if (!r.items.length) break;
      }
      return { items: items.filter(it => !target.except.has(it.id)) };
    } finally { setBusy(null); }
  }
  async function targetItems(target) {
    if (target.ids) return { items: target.ids.map(id => byIdItem(id) || { id, review: null }) };
    return resolveAll(target);
  }

  /** Choose among buttons in a dialog; resolves with the chosen value or null. */
  function choose({ title, message, options }) {
    return new Promise((resolve) => {
      let out = null;
      ui.modal({ title, body: h('p', { style: 'margin:0' }, message), onClose: () => resolve(out),
        footer: (api) => [h('button', { class: 'btn', onClick: () => api.close() }, 'Cancel'),
          ...options.map(o => h('button', { class: cls('btn', o.primary && 'btn-primary'), onClick: () => { out = o.value; api.close(); } }, o.label))] });
    });
  }

  // ---------------------------------------------------------------- review (with confirmation and undo)
  const serial = (() => { let chain = Promise.resolve(); return (fn) => { const p = chain.then(fn, fn); chain = p.catch(() => {}); return p; }; })();
  /** Set the review status of a target ({ids} | {all, except}). Returns true when stored.
      quiet: the change is visible where the user looks (tile badge, Inspect status line) — no toast. */
  async function review(target, status, { quiet = false, onlyUnreviewed = false, confirmed = false, note } = {}) {
    let res;
    try { res = await targetItems(target); } catch (e) { ui.toast(e.message, { type: 'error' }); return false; }
    let items = res.items || null;
    if (items && onlyUnreviewed) items = items.filter(it => !reviewStatus(it));
    if (items && status && items.length > 1 && !onlyUnreviewed) {
      // Bulk review never silently overwrites a different decision someone already made.
      const others = items.filter(it => { const s = reviewStatus(it); return s && s !== status; });
      if (others.length) {
        const by = {}; others.forEach(it => { const s = reviewStatus(it); by[s] = (by[s] || 0) + 1; });
        const parts = Object.entries(by).map(([s, k]) => `${fmt.num(k)} ${REVIEW[s].label.toLowerCase()}`).join(' and ');
        const rest = items.length - others.length;
        const pick = await choose({ title: `${VERB[status]} ${fmt.num(items.length)} images?`,
          message: `${parts} of them already ${others.length === 1 ? 'has' : 'have'} a different review. Change ${others.length === 1 ? 'it' : 'them'} too, or leave ${others.length === 1 ? 'it' : 'them'} as ${others.length === 1 ? 'it is' : 'they are'}?`,
          options: [rest ? { label: `${VERB[status]} only the other ${fmt.num(rest)}`, value: 'skip' } : null, { label: `${VERB[status]} all ${fmt.num(items.length)}`, value: 'all', primary: true }].filter(Boolean) });
        if (!pick) return false;
        if (pick === 'skip') items = items.filter(it => !reviewStatus(it) || reviewStatus(it) === status);
        confirmed = true;
      }
    }
    const n = items ? items.length : res.count;
    if (!n) { flash(onlyUnreviewed ? 'Nothing unreviewed in the selection' : 'Nothing to change', { short: onlyUnreviewed ? 'Nothing unreviewed' : '' }); return false; }
    if (n > CONFIRM_OVER && !confirmed) {
      const what = status ? VERB[status] : 'Clear the review of';
      const ok = await ui.confirm({ title: `${what} ${fmt.num(n)} images?`, okLabel: status ? `${what} ${fmt.num(n)}` : `Clear ${fmt.num(n)}`, danger: !status || status === 'excluded',
        message: items ? `You can undo this with ${fovea.MOD} Z.` : `More than ${fmt.num(RESOLVE_MAX)} images: this cannot be undone from Explore.` });
      if (!ok) return false;
    }
    const prev = items ? items.map(it => [it.id, reviewStatus(it), noteOf(it)]) : null;
    const body = items ? { image_ids: items.map(it => it.id), status } : { filters: res.filters, status };
    if (note !== undefined) body.note = note;
    const ids = items ? body.image_ids : null;
    try {
      await serial(() => fovea.api.put(`/api/datasets/${ds.id}/review`, body));
    } catch (e) { ui.toast(`Review failed: ${e.message}`, { type: 'error' }); return false; }
    applyLocal(ids, status, ids ? null : { except: target.except, filters: res.match || res.filters }, note);
    // The loaded tiles carry the progress count only for images that are loaded: "all matching" and selections
    // that reach unloaded images are counted again by the server.
    if (!ids || !allLoaded(ids)) recountProgress();
    fovea.bus.emit('review:changed', { ids: ids || [], status, filters: ids ? null : res.filters });
    const label = status ? `${REVIEW[status].label} ${fmt.num(n)} image${n > 1 ? 's' : ''}` : `Cleared ${fmt.num(n)} review${n > 1 ? 's' : ''}`;
    if (prev) pushUndo(label, prev);
    if (!quiet) ui.toast(label, { timeout: 4000, action: prev ? { label: 'Undo', onClick: () => undo() } : undefined });
    if (target.all || (target.ids && target.ids.length > 1 && selCount() > 0)) clearSelection();   // done with that selection
    return true;
  }
  /** Reflect stored review changes in the loaded items and tiles, and dim images that no longer match.
      By ids, or (for a filter-based action) every loaded image that matched that filter. */
  function applyLocal(ids, status, scope, note) {
    if (!cursor) return;
    const set = ids ? new Set(ids) : null;
    const now = Date.now() / 1000;
    cursor.items.forEach((it, i) => {
      if (!it) return;
      if (set) { if (!set.has(it.id)) return; }
      else if ((scope.except && scope.except.has(it.id)) || !matchesWith(it, scope.filters || filters)) return;
      const keep = note !== undefined ? note : noteOf(it);
      reviewedDelta += (status ? 1 : 0) - (reviewStatus(it) ? 1 : 0);
      it.review = status ? { status, note: keep, updated_at: now } : null;
      it.note = keep;
      if (matchesWith(it, filters)) gone.delete(it.id); else gone.add(it.id);
      updateTileReview(i);
    });
    updateCount();
    facetsChanged();
    if (inspect) inspect.redraw();
  }
  /** Client-side check of the filters that a review or a label edit can change (the rest cannot change). */
  function matchesWith(it, f) {
    const s = reviewStatus(it), r = f.review;
    if (r === 'none' && s) return false;
    if (r === 'any' && !s) return false;
    if (r && !['none', 'any'].includes(r) && s !== r) return false;
    if (f.exclude_review && s && csv(f.exclude_review).includes(s)) return false;
    const nb = it.boxes ? it.boxes.length : it.n_boxes || 0;
    const lab = f.labeled;
    if ((lab === '1' && !(it.has_label && nb > 0)) || (lab === '0' && it.has_label) || (lab === 'empty' && !(it.has_label && nb === 0)) || (lab === 'nobox' && nb > 0)) return false;
    if (it.boxes) {
      const cs = csv(f.cls).map(Number);
      if (cs.length && !it.boxes.some(b => cs.includes(b[0]))) return false;
      const only = csv(f.only_cls).map(Number);
      if (only.length && !(nb > 0 && it.boxes.every(b => only.includes(b[0])))) return false;
    }
    if (f.min_boxes && nb < Number(f.min_boxes)) return false;
    if (f.max_boxes && nb > Number(f.max_boxes)) return false;
    // Issues and QA flags change with label edits (flags once the QA re-check ran; see rereadFlags).
    if (issueMatch(it, f.issue) === false) return false;
    return true;
  }
  // One undo stack for everything ⌘Z can take back here, newest last: reviews ({kind:'review', prev}), AI
  // suggestions rejected or accepted in Inspect ({kind:'decide', ids, imageId, snapshot?}), and bulk proposal
  // actions that cannot be taken back from Explore ({kind:'note', message}) — those stop ⌘Z from silently
  // reaching past them to an older review.
  function pushUndo(label, prev) { pushUndoEntry({ kind: 'review', label, prev }); }
  function pushUndoEntry(u) { undoStack.push(u); if (undoStack.length > 30) undoStack.shift(); }
  let undoBusy = false;
  async function undo() {
    if (undoBusy) return;
    const u = undoStack.pop();
    if (!u) { flash('Nothing to undo'); return; }
    if (u.kind === 'note') { flash(u.message); return; }
    if (u.kind === 'decide') { undoBusy = true; try { await undoDecision(u); } finally { undoBusy = false; } return; }
    const groups = new Map();
    for (const [id, s, note] of u.prev) { const k = JSON.stringify([s, note]); if (!groups.has(k)) groups.set(k, []); groups.get(k).push(id); }
    try {
      for (const [k, ids] of groups) {
        const [s, note] = JSON.parse(k);
        await serial(() => fovea.api.put(`/api/datasets/${ds.id}/review`, { image_ids: ids, status: s, note }));
        applyLocal(ids, s, null, note);
      }
      if (!allLoaded(u.prev.map(p => p[0]))) recountProgress();
      fovea.bus.emit('review:changed', { ids: u.prev.map(p => p[0]), status: undefined });
      // Undoing one image is confirmed in place; undoing a batch is a batch result (its tiles may be off screen).
      if (u.prev.length > 1 && !inspect) ui.toast(`Undone: ${u.label}`, { timeout: 4000 });
      else flash(`Undone: ${u.label}`, { short: 'Undone' });
    } catch (e) { ui.toast(`Undo failed: ${e.message}`, { type: 'error' }); }
  }
  /** Take back AI suggestions decided on one image in Inspect: rejected ones become pending again; accepted ones
      also get the image's label file back from the snapshot the accept took first. */
  async function undoDecision(u) {
    const name = u.name || '';
    flash(`Undoing: ${u.label}…`);
    try {
      if (u.snapshot != null && u.changed) {
        const r = await fovea.api.post(`/api/datasets/${ds.id}/snapshots/${u.snapshot}/restore`, { image_ids: [u.imageId] });
        if (r && r.job) await fovea.api.watchJob(r.job.id || r.job);
        noteLabelEdit(ds.id, u.imageId); watchQa();
      }
      if (u.ids.length) await fovea.api.post(`/api/datasets/${ds.id}/proposals/decide`, { accept: [], reject: [], pending: u.ids });
    } catch (e) {
      undoStack.push(u);                // still the newest action: ⌘Z can try again (both steps are idempotent)
      flash(`Undo failed: ${e.message}`, { kind: 'error' });
      return;
    }
    try { await refreshImage(u.imageId); } catch (e) { /* the next look at the image reads it again */ }
    flash(`Undone: ${u.label}${name ? ` · ${name}` : ''}`, { short: 'Undone' });
  }
  /** Re-read one image (its labels and how many AI suggestions are still pending) into the list and, when it is on
      screen there, into Inspect. */
  async function refreshImage(id) {
    const [{ item: fresh }, left] = await Promise.all([fovea.api.get(`/api/datasets/${ds.id}/images/${id}`), pendingPreds(id)]);
    // The single-image endpoint may not carry the AI counts: derive them from what is still pending.
    fresh.ai_pending = left.filter(p => !(p.kind in AUDIT_GLYPH)).length; fresh.ai_suspect = left.filter(p => p.kind in AUDIT_GLYPH).length;
    const c = inspect ? inspect.cursor() : cursor;
    const i = c ? c.indexOfId(id) : -1;
    if (i >= 0) { c.items[i] = { ...c.items[i], ...fresh }; if (c === cursor) { refreshTile(i); if (matchesWith(c.items[i], filters)) gone.delete(id); else gone.add(id); updateCount(); } }
    if (inspect) inspect.imageChanged(id, i >= 0 ? c.items[i] : fresh, left);
    facetsChanged();
    return { item: fresh, preds: left };
  }
  /** The pending AI suggestions of one image (proposals and audit findings). */
  const pendingPreds = (id) => fovea.api.get(`/api/datasets/${ds.id}/images/${id}/proposals`)
    .then(r => (r.preds || []).filter(x => (!x.state || x.state === 'pending') && (x.image_id == null || x.image_id === id)));
  /** A static confirmation where the user is looking: Inspect's status line, else the grid's (never a toast:
      toasts are for batch results and errors, docs/visual-comfort.md §3). */
  function flash(text, { kind = '', short = '' } = {}) { if (inspect) inspect.status(text, kind); else sayInGrid(short || text, kind, text); }
  /** Every one of these ids is a loaded tile (so the local progress count saw each of them). */
  function allLoaded(ids) {
    if (!cursor) return false;
    if (ids.length <= 4) return ids.every(id => cursor.indexOfId(id) >= 0);
    const have = new Set();
    cursor.items.forEach((it) => { if (it) have.add(it.id); });
    return ids.every(id => have.has(id));
  }

  async function approveRemaining() {
    if (selCount() > 0) { await review(selectionTarget(), 'approved', { onlyUnreviewed: true }); return; }
    if (!total) return;
    if (filters.review && filters.review !== 'none') { flash('Every image in this list already has a review', { short: 'All reviewed already' }); return; }
    let n, fl, ex;
    try { [n, fl, ex] = await Promise.all([countWith({ ...filters, review: 'none' }), countWith({ ...filters, review: 'flagged' }), countWith({ ...filters, review: 'excluded' })]); }
    catch (e) { ui.toast(e.message, { type: 'error' }); return; }
    if (!n) { flash('Nothing unreviewed in this list', { short: 'Nothing unreviewed' }); return; }
    const keepNote = fl || ex ? ` ${[fl ? `${fmt.num(fl)} flagged` : '', ex ? `${fmt.num(ex)} excluded` : ''].filter(Boolean).join(' and ')} stay as they are.` : '';
    const ok = await ui.confirm({ title: `Approve ${fmt.num(n)} unreviewed images?`, okLabel: `Approve ${fmt.num(n)}`,
      message: `${fmt.num(n)} of the ${fmt.num(total)} images in this list are unreviewed.${keepNote}${n <= RESOLVE_MAX ? ` Undo with ${fovea.MOD} Z.` : ''}` });
    if (!ok) return;
    await review({ all: true, filters: { ...filters, review: 'none' }, except: new Set(), count: n }, 'approved', { confirmed: true });
  }

  // ---------------------------------------------------------------- working sets: big selections travel as a server-side token
  // Up to URL_IDS_MAX explicit images go by id in the URL (?ids=, bookmarkable). More than that, and every
  // "all matching" selection, are frozen on the server (POST /sets → ?set=<token>, kept 24 h): Annotate, the AI tab,
  // exports and filter-wide actions then work on exactly the images that were selected, in the list's order.
  let setsOff = false;
  /** {token, count} for a frozen list of images; null on a server without sets. */
  async function makeSet(body) {
    if (setsOff) return null;
    try {
      const r = await fovea.api.post(`/api/datasets/${ds.id}/sets`, body);
      if (!r || !r.token) return null;
      setCount = { token: r.token, count: r.count ?? (body.ids ? body.ids.length : null) };
      return setCount;
    } catch (e) {
      if (e && (e.status === 404 || e.status === 405)) { setsOff = true; return null; }
      throw e;
    }
  }
  /** Ids in the order the list shows them. */
  function orderedIds(ids) {
    const pos = (id) => { const i = cursor ? cursor.indexOfId(id) : -1; return i < 0 ? Infinity : i; };
    return ids.map(id => [pos(id), id]).sort((a, b) => a[0] - b[0] || a[1] - b[1]).map(x => x[1]);
  }
  /** A selection target as a server-side set: {filters: {set}, count}, or null on a server without sets. */
  async function setTarget(target) {
    let body;
    if (target.ids) body = { ids: orderedIds(target.ids) };
    else if (!target.except || !target.except.size) body = { filters: { ...(target.filters || filters) }, sort, order };
    else {
      // All matching minus a few: the list's ids in order, without the exceptions.
      const r = await fovea.api.get(`/api/datasets/${ds.id}/images/ids`, { ...(target.filters || filters), sort, order });
      body = { ids: (r.ids || []).filter(id => !target.except.has(id)) };
    }
    const s = await makeSet(body);
    return s ? { filters: { set: s.token }, count: s.count } : null;
  }
  /** A selection target as filters another page can open: {ids} for a small explicit selection, else a set.
      Falls back to the current filters (all matching, no exceptions) on a server without sets; else throws. */
  async function selectionFilters(target) {
    // Ids are sorted again by the page that gets them, and "most suspicious first" ranks by the issue filter they
    // leave behind: in that order a selection becomes a set, which keeps the list's order.
    const small = target.ids && target.ids.length <= URL_IDS_MAX;
    const asIds = () => ({ filters: { ids: orderedIds(target.ids).join(',') }, count: target.ids.length });
    if (small && (sort !== 'flag' || setsOff)) return asIds();
    const tf = await setTarget(target);
    if (tf) return tf;
    if (small) return asIds();
    if (!target.ids && !(target.except && target.except.size)) return { filters: { ...(target.filters || filters) }, count: target.count, live: true };
    throw new Error(target.ids ? `This server cannot hold a selection of ${fmt.num(target.ids.length)} images — select at most ${fmt.num(URL_IDS_MAX)}, or narrow the filter and use the whole list.`
      : 'This server cannot hold “all except a few” — clear the exceptions or narrow the filter.');
  }
  /** Run a preparation step with the bulk bar showing what is going on (spinner after 400 ms, elapsed time). */
  async function preparing(text, fn) {
    const lbl = setBusy(text);
    try { return await fn(lbl); } finally { setBusy(null); }
  }
  /** Query string for another page showing exactly the selection, in this list's order. */
  function selectionQuery(tf, extra = {}) {
    const p = new URLSearchParams();
    for (const [k, v] of Object.entries(tf.filters)) if (v != null && v !== '') p.set(k, v);
    // A set was frozen in this list's order and opens in it. Ids and live filters are sorted again by the next page:
    // they carry the sort and its direction, spelled out (Annotate has defaults of its own).
    if (!tf.filters.set || tf.live) {
      if (sort === 'random' && filters.seed && !tf.live) p.set('seed', filters.seed);     // the same shuffled order
      putOrder(p, tf.filters);
    }
    for (const [k, v] of Object.entries(extra)) if (v != null && v !== '') p.set(k, v);
    return p.toString();
  }
  /** The image to open first: the focused one when it is part of the selection, else the first selected in list order. */
  function firstSelectedId() {
    const f = focusedId();
    if (f != null && isSelected(f)) return f;
    if (allMode) { for (let i = 0; i < (total || 0); i++) { const it = cursor.items[i]; if (it && !except.has(it.id)) return it.id; } return null; }
    return orderedIds([...selected])[0] ?? null;
  }

  // ---------------------------------------------------------------- other bulk actions
  /** Leave Explore for another page. The URL first gets the focused image (it is written 350 ms after a move), so
      browser Back returns to the same tile even right after an arrow key. */
  function leaveTo(url) { syncUrl(true); fovea.router.navigate(url); }
  function annotateId(id) { leaveTo(`/d/${ds.id}/annotate?${portableQuery({ img: id }).toString()}`); }
  async function annotateSelection() {
    const t = selectionTarget();
    if (!t.count) return;
    let tf;
    try { tf = await preparing(`Preparing ${fmt.num(t.count)} images for Annotate…`, () => selectionFilters(t)); }
    catch (e) { ui.toast(e.message, { type: 'error', timeout: 6000 }); return; }
    leaveTo(`/d/${ds.id}/annotate?${selectionQuery(tf, { img: firstSelectedId() })}`);
  }
  async function labelSelectionWithAI() {
    const t = selectionTarget();
    if (!t.count) return;
    let tf;
    try { tf = await preparing(`Preparing ${fmt.num(t.count)} images for AI labeling…`, () => selectionFilters(t)); }
    catch (e) { ui.toast(e.message, { type: 'error', timeout: 6000 }); return; }
    if (tf.live) { leaveTo(`/d/${encodeURIComponent(ds.id)}/ai?s=label`); ui.toast('Pick the same filters in the AI tab’s scope (this server cannot pass a selection along)', { timeout: 6000 }); return; }
    const key = tf.filters.ids ? 'ids' : 'set';
    leaveTo(`/d/${encodeURIComponent(ds.id)}/ai?s=label&${key}=${encodeURIComponent(tf.filters[key])}`);
  }
  /** Show only the selected images in the grid (the selection becomes the list: ?ids= or ?set=). */
  async function showSelection() {
    const t = selectionTarget();
    if (!t.count) return;
    let tf;
    try { tf = await preparing(`Preparing ${fmt.num(t.count)} images…`, () => selectionFilters(t)); }
    catch (e) { ui.toast(e.message, { type: 'error', timeout: 6000 }); return; }
    const keep = firstSelectedId();
    setFilters({ ...tf.filters, ...(filters.seed ? { seed: filters.seed } : {}) }, { keepId: keep });
  }
  async function copyPath(id) {
    try { const { item } = await fovea.api.get(`/api/datasets/${ds.id}/images/${id}`); await ui.copyText(item.abs_path_host || item.abs_path); }
    catch (e) { ui.toast(`Copy failed: ${e.message}`, { type: 'error' }); }
  }
  /** Filters for the export endpoints (POST bodies: no URL limit, so explicit ids of any number are fine on an
      older server; all-but-a-few needs a set). */
  async function exportFilters(target) {
    if (target.ids && (target.ids.length <= URL_IDS_MAX || setsOff)) return { ids: orderedIds(target.ids).join(',') };
    const tf = await selectionFilters(target);
    return { ...tf.filters };
  }
  async function pathList(target) {
    const f = await exportFilters(target);
    const text = await fovea.api.post(`/api/datasets/${ds.id}/export/list`, { filters: f, style: 'host' });
    return typeof text === 'string' ? text : '';
  }
  async function copyPaths(target) {
    try {
      const text = await preparing(`Collecting ${fmt.num(target.count)} paths…`, () => pathList(target));
      const n = text.split('\n').filter(Boolean).length;
      ui.copyText(text, `Copied ${fmt.num(n)} path${n === 1 ? '' : 's'}`);
    } catch (e) { ui.toast(`Copy failed: ${e.message}`, { type: 'error' }); }
  }
  function saveFile(href, name) {
    const a = h('a', { href, download: name || '', style: 'display:none' });
    document.body.appendChild(a); a.click(); a.remove();
  }
  async function downloadList(target) {
    try {
      const text = await preparing(`Collecting ${fmt.num(target.count)} paths…`, () => pathList(target));
      const url = URL.createObjectURL(new Blob([text], { type: 'text/plain;charset=utf-8' }));
      saveFile(url, `${ds.id}-selection-${text.split('\n').filter(Boolean).length}.txt`);
      setTimeout(() => URL.revokeObjectURL(url), 60000);
    } catch (e) { ui.toast(`Download failed: ${e.message}`, { type: 'error' }); }
  }
  /** ZIP of the selected images with their label files and data.yaml, streamed by the browser's own downloader. */
  async function downloadZip(target) {
    let info;
    try {
      info = await preparing(`Planning a ZIP of ${fmt.num(target.count)} images…`, async () => {
        const f = await exportFilters(target);
        return fovea.api.post(`/api/datasets/${ds.id}/export/zip-link`, { filters: f, include_unlabeled: true, include_yaml: true, filename: `${ds.id}-selection` });
      });
    } catch (e) { ui.toast(`ZIP export failed: ${e.message}`, { type: 'error' }); return; }
    if (!info || !info.url) { ui.toast('ZIP export failed: the server sent no download link', { type: 'error' }); return; }
    const n = info.images ?? target.count;
    const ok = await ui.confirm({ title: `Download ${fmt.plural(n, 'image')} as a ZIP?`, okLabel: 'Download',
      message: `${info.bytes ? `${fmt.bytes(info.bytes)} · ` : ''}images, their label files and data.yaml. For class remapping, near-duplicate removal or new splits use the Export tab.${info.missing ? ` ${fmt.num(info.missing)} image file${info.missing === 1 ? ' is' : 's are'} missing on disk and left out.` : ''}` });
    if (!ok) return;
    saveFile(info.url, info.filename || '');
  }
  function moreMenu(anchor) {
    const t = selectionTarget();
    const sfx = ` (${fmt.num(t.count)})`;
    ui.menu(anchor, [
      { label: `Show only the selection${sfx}`, icon: 'filter', onClick: () => showSelection() },
      { sep: true },
      { label: 'Copy paths', icon: 'copy', onClick: () => copyPaths(t) },
      { label: 'Download path list (.txt)', icon: 'fileText', onClick: () => downloadList(t) },
      { label: 'Download ZIP (images + labels)…', icon: 'download', onClick: () => downloadZip(t) },
    ]);
  }
  function proposalsMenu(anchor) {
    const thr = propThreshold.toFixed(2);
    ui.menu(anchor, [
      { label: 'AI proposals on the selection', header: true },
      { label: `Accept proposals ≥ ${thr}`, icon: 'check', onClick: () => bulkProposals('accept') },
      { label: 'Reject pending proposals', icon: 'x', onClick: () => bulkProposals('reject') },
      { label: `Reject proposals below ${thr}`, icon: 'minus', onClick: () => bulkProposals('reject-below') },
      { sep: true },
      { label: 'Label the selection with AI…', icon: 'sparkles', onClick: () => labelSelectionWithAI() },
    ]);
  }
  async function bulkProposals(kind) {
    const target = selectionTarget();
    let body;
    let known = null, withPending = null;
    try {
      if (target.ids) { body = { image_ids: target.ids }; known = target.ids.reduce((a, id) => a + ((byIdItem(id) || {}).ai_pending || 0), 0); }
      else if (!target.except.size) body = { filters: { ...filters } };
      else { const r = await resolveAll(target); if (r.items) { body = { image_ids: r.items.map(it => it.id) }; known = r.items.reduce((a, it) => a + (it.ai_pending || 0), 0); } else body = { filters: r.filters }; }
      if (known == null && body.filters) withPending = await countWith({ ...body.filters, ai: 'pending' });
    } catch (e) { ui.toast(e.message, { type: 'error' }); return; }
    const scope = `${fmt.num(target.count)} image${target.count === 1 ? '' : 's'}`;
    const thr = propThreshold.toFixed(2);
    const what = kind === 'accept' ? `Accept AI proposals with confidence ≥ ${thr} on ${scope}?` : kind === 'reject' ? `Reject every pending AI proposal on ${scope}?` : `Reject AI proposals below ${thr} on ${scope}?`;
    const ok = await ui.confirm({ title: what, okLabel: kind === 'accept' ? 'Accept' : 'Reject', danger: kind !== 'accept',
      message: `${known != null ? `${fmt.num(known)} proposal${known === 1 ? '' : 's'} pending. ` : withPending != null ? `${fmt.num(withPending)} of them have pending proposals. ` : ''}${kind === 'accept' ? 'Accepted boxes are written to the label files (a snapshot is taken first, so History can restore them). A proposal over a box of another class stays pending for you to decide in Annotate.' : 'Rejected proposals are kept, so the same suggestion is not offered again.'}` });
    if (!ok) return;
    try {
      if (kind === 'accept') {
        setBusy('Accepting proposals…');
        const r = await fovea.api.post(`/api/datasets/${ds.id}/proposals/accept`, { ...body, min_conf: propThreshold || undefined, kinds: PROPOSAL_KINDS, skip_conflicts: true });
        const job = r && r.job;
        const snap = job ? await fovea.api.watchJob(job.id || job) : r;
        setBusy(null);
        noteLabelEdit(ds.id, null); watchQa();      // written files are re-checked by the QA engine: followed dataset-wide
        // Proposals over a box of another class stayed pending (⇄): the images they are on can be listed at once.
        const res = (snap && snap.result) || {}, cimg = Array.isArray(res.conflict_images) ? res.conflict_images.filter(Number.isInteger) : [];
        ui.toast(snap && snap.status === 'cancelled' ? 'Accepting proposals was cancelled' : `Proposals accepted: ${acceptSummary(res)}`,
          cimg.length ? { type: 'info', timeout: 12000, action: { label: `Show ${cimg.length === 1 ? 'the image' : `${cimg.length} images`}`, onClick: () => { if (!destroyed) setFilters({ ids: cimg.join(',') }); } } } : { type: 'ok', timeout: 5000 });
        pushUndoEntry({ kind: 'note', label: 'Accepted proposals', message: `Accepted proposals are undone from Overview → History (a snapshot was saved first). ${fovea.MOD} Z again undoes the action before.` });
        refresh({ keepSelection: false });
      } else {
        const r = await fovea.api.post(`/api/datasets/${ds.id}/proposals/reject`, kind === 'reject' ? { ...body, kinds: PROPOSAL_KINDS } : { ...body, min_conf: propThreshold, below: true, kinds: PROPOSAL_KINDS });
        ui.toast(`Rejected ${fmt.num((r && r.rejected) || 0)} proposal${(r && r.rejected) === 1 ? '' : 's'}`, { type: 'ok' });
        if ((r && r.rejected) || 0) pushUndoEntry({ kind: 'note', label: 'Rejected proposals', message: `Proposals rejected on a selection cannot be taken back from Explore. ${fovea.MOD} Z again undoes the action before.` });
        refresh({ keepSelection: false });
      }
    } catch (e) { setBusy(null); ui.toast(`AI proposals: ${e.message}`, { type: 'error' }); }
  }

  // ---------------------------------------------------------------- count, banner
  // How much of this list is reviewed: counted once per load, then kept current from the reviews made here.
  // The count belongs to one filter set (progressFor): a list with other filters — a review filter, where the
  // share is 0 % or 100 % by definition and is not shown — never shows the previous list's percentage.
  let reviewedBase = null, reviewedDelta = 0, progressFor = '';
  function loadProgress(my) {
    if (progressFor !== facetKeyOf(filters)) { reviewedBase = null; reviewedDelta = 0; progressFor = ''; updateCount(); }
    recountProgress(my);
  }
  /** Count the reviewed images of this list on the server. The count waits in the review queue (serial): every
      review sent before it is in the answer and none sent after, so the local delta starts again from zero. The
      shown percentage stays until the answer arrives (no blank, no jump) while the list keeps its filters. */
  function recountProgress(my = gen) {
    if (filters.review || !total) return;
    const f = { ...filters, review: 'any' }, key = facetKeyOf(filters);
    serial(() => countWith(f).then((n) => { if (my === gen && !destroyed) { reviewedBase = n; reviewedDelta = 0; progressFor = key; updateCount(); } }, () => {}));
  }
  function updateCount() {
    if (total == null) { countEl.textContent = '—'; return; }
    const n = total - gone.size;
    const done = reviewedBase == null || progressFor !== facetKeyOf(filters) || filters.review ? null : clamp(reviewedBase + reviewedDelta, 0, total);
    // In a crowded toolbar the progress and "changed" parts shorten (ellipsis) before any control is pushed out.
    countEl.replaceChildren(h('span', { class: 'cnt' }, `${fmt.num(n)} image${n === 1 ? '' : 's'}`),
      done != null && total ? h('span', { class: 'progress-txt', title: `${fmt.num(done)} of ${fmt.num(total)} reviewed` }, ` · ${done && done < total ? Math.max(1, Math.floor(100 * done / total)) : Math.round(100 * done / total)}% reviewed`) : '',
      gone.size ? h('button', { class: 'btn btn-ghost btn-sm gone-btn', 'data-tip': 'Reviewed or edited images that no longer match the filters stay in place (dimmed) until you refresh.', onClick: () => refresh() }, `· ${fmt.num(gone.size)} changed · refresh`) : '');
  }
  function showBanner(text, actionLabel, action) {
    banner.classList.remove('hidden');
    banner.replaceChildren(icon('info', 14), h('span', { class: 'grow' }, text),
      h('button', { class: 'btn btn-sm', onClick: () => { banner.classList.add('hidden'); action(); } }, actionLabel),
      h('button', { class: 'btn btn-ghost btn-icon btn-sm', 'aria-label': 'Dismiss', onClick: () => banner.classList.add('hidden') }, icon('x', 13)));
  }
  /** Re-query the list, keeping the focused image (or the scroll position) and, optionally, the selection. */
  function refresh({ keepSelection = false } = {}) { banner.classList.add('hidden'); facetCache.clear(); facetsFor = ''; reload({ keepId: focusedId(), keepScroll: true, keepSelection }); }

  // ---------------------------------------------------------------- focus & keyboard (grid)
  function setFocus(i, { scroll = true, dom = true, center = false } = {}) {
    if (!cursor || !total) return;
    i = clamp(i, 0, total - 1);
    const prev = mounted.get(focusIdx);
    if (prev) { prev.classList.remove('focus'); prev.tabIndex = -1; }
    focusIdx = i;
    if (scroll) scrollToIndex(i, center);
    renderWindow();
    const t = mounted.get(i);
    if (t) { t.classList.add('focus'); t.tabIndex = 0; if (dom) t.focus({ preventScroll: true }); }
    syncUrl();
  }
  function focusGrid() { if (!total) return; setFocus(focusIdx >= 0 ? focusIdx : firstVisibleIndex()); }
  function scrollToIndex(i, center = false) {
    const y = rowTop(i), vh = body.clientHeight;
    if (center) body.scrollTop = Math.max(0, y - (vh - L.tileH) / 2);
    else if (y < body.scrollTop + 4) body.scrollTop = Math.max(0, y - 8);
    else if (y + L.tileH > body.scrollTop + vh - 4) body.scrollTop = y + L.tileH - vh + 8;
  }
  async function focusId(id) {
    const c = cursor; if (!c) return;
    const i = await c.positionOf(id);
    if (c !== cursor || i < 0) return;
    if (total !== c.total) { total = c.total; relayout(); }
    setFocus(i, { center: true });
  }
  function setTileSize(v) {
    if (!SIZES.includes(v) || v === tileSize) return;
    tileSize = v; ls.set('fovea.tile', v);
    relayout();
  }
  function toggleOverlay() {
    showOverlay = !showOverlay;
    ls.set('fovea.overlay', showOverlay ? '1' : '0');
    overlayBtn.classList.toggle('active', showOverlay); overlayBtn.setAttribute('aria-pressed', String(showOverlay));
    vgrid.classList.toggle('no-overlay', !showOverlay);     // pure CSS — tiles and their images are untouched
    if (inspect) inspect.redraw();
  }
  function setIsolate(c) {
    isolate = c;
    redrawOverlays(); renderChips();
    if (inspect) inspect.redraw();
  }
  /** Classes this dataset has: named ones, and any id the labels use beyond them. */
  function classCount() {
    let n = names.length;
    for (const c of (stats && stats.classes) || []) if (Number.isInteger(c.cls)) n = Math.max(n, c.cls + 1);
    for (const k of Object.keys((facets && facets.cls) || {})) if (/^\d+$/.test(k)) n = Math.max(n, Number(k) + 1);
    return n;
  }
  let digitBuf = '', digitTimer = null;
  function digitIsolate(d) {
    // A second digit within 600 ms extends the number only when that class exists (1 then 2 → class 12 with 13+
    // classes; with 12 classes it shows class 2): never a class id the dataset does not have.
    const n = classCount(), r = isolateDigit(digitBuf, d, n);
    clearTimeout(digitTimer);
    digitBuf = r.buf;
    if (digitBuf) digitTimer = setTimeout(() => { digitBuf = ''; }, 600);
    if (r.cls == null) { flash(`No class ${d} — this dataset has classes 0–${n - 1}`, { short: `No class ${d}` }); return; }
    setIsolate(!r.joined && isolate === r.cls ? null : r.cls);   // the same digit again shows every class
  }

  // The key right after a plain G belongs to the global G-chord (G E, G A …): it must never also approve,
  // edit or flag here, whichever listener happens to run first. Inspect listens on window capture, ahead of the
  // shell, so it asks the shell whether its chord (and the "G…" chip) is still waiting rather than keeping a
  // timer of its own that could run out first.
  let gAt = 0, guardEvt = null, guardRes = false;
  function chordGuard(e) {
    if (e === guardEvt) return guardRes;
    guardEvt = e;
    if (keys.letter(e) === 'g' && keys.plain(e) && !e.shiftKey) { gAt = performance.now(); return (guardRes = true); }
    const pending = typeof fovea.chordPending === 'function' ? fovea.chordPending() : !!gAt && performance.now() - gAt < CHORD_MS;
    if (e.key !== 'Shift' && e.key !== 'CapsLock') gAt = 0;      // the shell's chord also waits through these
    return (guardRes = !!pending);
  }
  const menuOpen = () => !!document.querySelector('body > .menu');

  async function nextMatch(from, dir, test) {
    const c = cursor;
    for (let i = from + dir, n = 0; i >= 0 && i < total && n < SUSPECT_SCAN; i += dir, n++) {
      const it = c.items[i] || await c.ensure(i).catch(() => null);
      if (c !== cursor || destroyed) return -2;
      if (it && test(it)) return i;
    }
    return -1;
  }
  async function jumpTo(test, dir, what) {
    const i = await nextMatch(focusIdx < 0 ? (dir > 0 ? -1 : total) : focusIdx, dir, test);
    if (i === -2) return;
    if (i < 0) { flash(`No ${what} in the next ${fmt.num(SUSPECT_SCAN)}${test === isSuspect ? ' — Filters → Issues lists them all' : ''}`, { short: `No ${what.replace(/^image with an? /, '').replace(/ image$/, '')} in the next ${fmt.num(SUSPECT_SCAN)}` }); return; }
    setFocus(i, { center: !isOnScreen(i) });
  }
  const jumpSuspect = (dir) => jumpTo(isSuspect, dir, 'image with a label problem');

  function reviewKey(e, status) {
    if (e.repeat) return;                       // holding a key must not review a run of images
    if (selCount() > 0) { review(selectionTarget(), status); return; }
    if (focusIdx < 0 && cursor && total) {
      // Nothing focused yet: a review must land on an image the user can see is meant, so this key only puts the
      // focus ring on the first image on screen and says how to go on.
      const i = firstVisibleIndex();
      const K = keys.letter(e).toUpperCase(), verb = status ? VERB[status].toLowerCase() : 'clear the review';
      if (i >= 0) { setFocus(i, { scroll: false }); flash(`Focused the first image — press ${K} again to ${verb}${status ? ' it' : ''}`, { short: `Press ${K} again to ${verb}` }); }
      return;
    }
    const it = cursor && cursor.items[focusIdx];
    if (!it) return;
    const at = focusIdx;
    review({ ids: [it.id], count: 1 }, status, { quiet: true }).then((ok) => {
      if (ok && advance && status !== 'flagged' && focusIdx === at && !inspect) setFocus(at + 1);
    });
  }

  const onKey = (e) => {
    if (destroyed || inspect || e.defaultPrevented) return;
    if (ui.hasModal() || isTyping()) return;
    if (chordGuard(e)) return;
    if (menuOpen() || !cursor || !L) return;
    if (keys.is(e, 'mod+a')) { e.preventDefault(); selectAllMatching(); return; }
    if (keys.is(e, 'mod+z')) { e.preventDefault(); undo(); return; }
    if (!keys.plain(e)) return;
    const k = e.key, rows = Math.max(1, Math.floor(body.clientHeight / L.rowH));
    // A Tab-focused toolbar button takes its own Enter / Space, and a select or slider its own arrows: the grid's
    // keys act only when the grid (or nothing in particular) has the focus.
    const ctl = controlKeys(document.activeElement, vgrid);
    if (ctl.press && (k === 'Enter' || keys.is(e, 'Space'))) return;
    if (ctl.arrows && ['ArrowRight', 'ArrowLeft', 'ArrowDown', 'ArrowUp', 'PageDown', 'PageUp', 'Home', 'End'].includes(k)) return;
    const from = focusIdx < 0 ? firstVisibleIndex() : focusIdx;
    const move = (i) => { e.preventDefault(); setFocus(i); };
    /** The image a one-image key acts on: the focused tile, or — nothing focused yet (a fresh list from a link) —
        the first one on screen, which then gets the focus ring so the result is visible. */
    const focusedOrFirst = () => {
      if (focusIdx >= 0) return focusIdx;
      if (from < 0 || !total) return -1;
      setFocus(from, { scroll: false });
      return from;
    };
    if (k === 'ArrowRight') move(focusIdx < 0 ? from : focusIdx + 1);
    else if (k === 'ArrowLeft') move(focusIdx < 0 ? from : focusIdx - 1);
    else if (k === 'ArrowDown') move(focusIdx < 0 ? from : focusIdx + L.cols);
    else if (k === 'ArrowUp') move(focusIdx < 0 ? from : focusIdx - L.cols);
    else if (k === 'PageDown') move(from + rows * L.cols);
    else if (k === 'PageUp') move(from - rows * L.cols);
    else if (k === 'Home') move(0);
    else if (k === 'End') move(total - 1);
    else if (k === 'Enter') { if (!e.repeat) { const i = focusedOrFirst(); if (i >= 0) { e.preventDefault(); openInspect(i); } } }
    else if (keys.is(e, 'Space')) { e.preventDefault(); if (!e.repeat) { const i = focusedOrFirst(); if (i >= 0) quickLook(i); } }
    else if (k === 'Escape') {
      if (selCount() > 0) clearSelection();
      else if (aiApplied) restoreAsk();
      else if (isolate != null) setIsolate(null);
      else if (document.activeElement && vgrid.contains(document.activeElement)) document.activeElement.blur();
    }
    else if (keys.is(e, 'shift+a')) { if (!e.repeat) approveRemaining(); }
    else if (keys.is(e, 's') || keys.is(e, 'shift+s')) { if (!e.repeat) { const i = focusedOrFirst(); if (i >= 0) toggleSelect(i, e.shiftKey ? 'range' : 'toggle'); } }
    else if (keys.is(e, 'o')) { if (!e.repeat) toggleOverlay(); }
    else if (keys.is(e, 'e')) { if (!e.repeat) { const i = focusedOrFirst(); const it = i >= 0 && cursor.items[i]; if (it) annotateId(it.id); } }
    else if (keys.is(e, '/')) { e.preventDefault(); searchInput.focus(); searchInput.select(); }
    else if (keys.is(e, '[')) setTileSize(SIZES[Math.max(0, SIZES.indexOf(tileSize) - 1)]);
    else if (keys.is(e, ']')) setTileSize(SIZES[Math.min(SIZES.length - 1, SIZES.indexOf(tileSize) + 1)]);
    else if (keys.is(e, 'j') || keys.is(e, 'shift+j')) { if (!e.repeat) jumpSuspect(e.shiftKey ? -1 : 1); }
    else if (keys.is(e, 'n')) { if (!e.repeat) jumpTo((it) => !reviewStatus(it), 1, 'unreviewed image'); }
    else if (keys.digit(e) && !e.shiftKey) digitIsolate(keys.digit(e));
    else if (!e.shiftKey) { const l = keys.letter(e); if (l && l in REVIEW_KEYS) reviewKey(e, REVIEW_KEYS[l]); }
  };
  // Space on a focused tile is quick look. The shell's generic "activate role=option on Enter/Space" listener
  // (lib/ui.js, on document) would click the tile — open Inspect — before onKey runs, so it is taken here first.
  vgrid.addEventListener('keydown', (e) => {
    if (!keys.is(e, 'Space') || !e.target.closest || !e.target.closest('.tile') || destroyed || inspect || ui.hasModal() || isTyping()) return;
    e.preventDefault();
    if (!e.repeat && focusIdx >= 0) quickLook(focusIdx);
  });
  const onKeyUp = (e) => { if (quick && (e.code === 'Space' || e.key === ' ')) closeQuick(); };
  document.addEventListener('keydown', onKey);
  document.addEventListener('keyup', onKeyUp);
  const onBlur = () => closeQuick();
  window.addEventListener('blur', onBlur);

  // ---------------------------------------------------------------- quick look (hold Space)
  const qlFrames = createFrameCache({ max: 3 });
  let qlEl = null;
  async function quickLook(i) {
    const it = cursor && cursor.items[i]; if (!it) return;
    const my = {}; quick = my;
    const t = mounted.get(i);
    const wait = t ? h('span', { class: 'ql-wait' }, ui.spinner()) : null;
    if (wait) t.appendChild(wait);
    const im = await qlFrames.get(fovea.api.imgUrl(it.id)).catch(() => null);
    if (wait) wait.remove();
    if (quick !== my || destroyed) return;
    if (!im) { flash('Image could not be loaded', { kind: 'error' }); quick = null; return; }
    im.alt = ''; im.draggable = false;
    const pic = h('div', { class: 'pic is-ready' }, im, showOverlay && it.boxes && it.boxes.length ? fovea.boxLayer(it.boxes, { names, labels: true, filter: isolate != null ? new Set([isolate]) : null }) : null);
    const nw = im.naturalWidth || 4, nh = im.naturalHeight || 3;
    const s = Math.min(window.innerWidth * 0.9 / nw, window.innerHeight * 0.82 / nh);
    pic.style.width = `${Math.round(nw * s)}px`; pic.style.height = `${Math.round(nh * s)}px`;
    qlEl = h('div', { class: 'quicklook', role: 'dialog', 'aria-label': `Quick look: ${it.rel_path}` }, pic, h('div', { class: 'ql-name mono' }, it.rel_path, h('span', { class: 'faint' }, ' · release Space to close')));
    document.body.appendChild(qlEl);
  }
  function closeQuick() { quick = null; if (qlEl) { qlEl.remove(); qlEl = null; } }

  // ---------------------------------------------------------------- Inspect
  function pushInspectState(id) {
    const p = stateQuery({ img: id, view: 'inspect' });
    history.pushState({ foveaInspect: true }, '', `${location.pathname}?${p.toString()}`);
    fovea.router.replaceQuery(Object.fromEntries(p));
    history.replaceState({ foveaInspect: true }, '', location.pathname + location.search);
    inspectPushed = true;
  }
  function openInspect(i, { push = true } = {}) {
    if (!cursor || i < 0 || i >= total) return;
    closeQuick();
    if (inspect) { inspect.show(i); return; }
    inspect = createInspect(i);
    const it = cursor.items[i];
    if (push) pushInspectState(it ? it.id : null); else syncUrl(true);
  }
  async function openInspectId(id, opts) {
    const c = cursor; if (!c) return;
    const i = await c.positionOf(id);
    if (c !== cursor || i < 0) return;
    if (total !== c.total) { total = c.total; relayout(); }
    openInspect(i, opts);
  }

  function createInspect(index) {
    const c0 = cursor;
    const frames = createFrameCache({ max: 6 });
    const predCache = new Map();          // image id -> Promise<pending preds>
    const qaCache = new Map();            // image id -> Promise<QA flags with details | null>
    let qaDetail = [], flagFocus = null;  // the shown image's QA flags ({code, detail}); the flag chip pressed
    const twinCache = new Map();          // image id -> Promise<its boxes | null> (the twin of a dup_label_mismatch)
    let twin = null, twinIouP = null, twinMemo = null;   // the pressed mismatch chip's twin {forId, id, boxes, minIou}; the dataset's mismatch IoU
    let idx = index, item = null, natural = [0, 0], showLabels = true, activeBox = -1, hoverBox = -1, zoomedTo = null;
    let closed = false, want = index, seq = 0, autoOn = false, autoTimer = null, preds = [], noteDirty = false;
    let speed = clamp(Number(ls.get('fovea.inspect.speed', '400')) || 400, 100, 2000);
    let similarOpen = ls.get('fovea.inspect.similar', '1') === '1';
    const view = { z: 1, cx: 0.5, cy: 0.5 };   // zoom relative to fit, and the image point at the stage centre
    const restoreFocus = document.activeElement;
    imgLoader.pause();                    // the grid is hidden: give every connection to the full-size frames

    const stage = h('div', { class: 'inspect-stage' });
    const side = h('div', { class: 'inspect-side' });
    const top = h('div', { class: 'inspect-top' });
    const rootEl = h('div', { class: 'inspect ex-inspect', role: 'dialog', 'aria-modal': 'true', 'aria-label': 'Inspect image', tabindex: '-1' }, top, stage, side);
    const overlayToggle = h('button', { class: cls('btn btn-sm', showOverlay && 'active'), 'aria-pressed': String(showOverlay), 'data-tip': 'Boxes (O)', onClick: () => toggleOverlay() }, icon('square', 13), 'Boxes');
    const labelsToggle = h('button', { class: cls('btn btn-sm', showLabels && 'active'), 'aria-pressed': String(showLabels), 'data-tip': 'Labels (L)', onClick: () => { showLabels = !showLabels; labelsToggle.classList.toggle('active', showLabels); labelsToggle.setAttribute('aria-pressed', String(showLabels)); renderOverlay(); } }, icon('tag', 13), 'Labels');
    const autoBtn = h('button', { class: 'btn btn-sm', 'data-tip': 'Auto-play (Space)', onClick: () => toggleAuto() }, icon('play', 13), 'Play');
    const speedLbl = h('span', { class: 'xs mono speed-lbl' });
    const speedInput = h('input', { type: 'range', class: 'slider', min: 100, max: 2000, step: 10, value: speed, 'aria-label': 'Auto-play interval in milliseconds', style: 'width:90px',
      // A drag gives the arrow keys back to image stepping on release; a Tab-focused slider keeps its arrows.
      onInput: (e) => { speed = Number(e.target.value); ls.set('fovea.inspect.speed', String(speed)); setSpeedLbl(); if (autoOn) schedule(); },
      onPointerup: (e) => e.target.blur() });
    const setSpeedLbl = () => {
      speedLbl.textContent = `${speed} ms`;
      speedLbl.title = speed < MIN_SWAP_MS ? `Below ${MIN_SWAP_MS} ms only between frames of the same sequence — unrelated images change at most 3 times a second.` : '';
      speedLbl.classList.toggle('limited', speed < MIN_SWAP_MS);
    };
    setSpeedLbl();
    const posEl = h('span', { class: 'mono small pos' });
    const nameEl2 = h('span', { class: 'name grow' });
    const statusEl = h('span', { class: 'inspect-status', 'aria-live': 'polite' });
    const zoomBtn = h('button', { class: 'btn btn-sm zoom-btn', 'data-tip': 'Fit ↔ actual pixels (1:1). Wheel zooms at the pointer.', onClick: () => (view.z > 1.001 ? fitView() : actualPixels()) }, 'Fit');
    top.append(h('button', { class: 'btn btn-sm btn-icon', 'aria-label': 'Close', 'data-tip': 'Close (Esc)', onClick: () => close() }, icon('x', 14)),
      posEl, nameEl2, statusEl, zoomBtn, overlayToggle, labelsToggle, autoBtn, speedInput, speedLbl,
      h('button', { class: 'btn btn-sm', onClick: () => { if (item) annotateId(item.id); } }, icon('pen', 13), 'Edit (E)'));
    const picHolder = h('div', { class: 'pic' });
    const waitEl = h('div', { class: 'inspect-wait' });
    stage.append(picHolder, waitEl,
      h('button', { class: 'inspect-nav prev', 'aria-label': 'Previous image', onClick: () => step(-1) }, icon('chevronLeft')),
      h('button', { class: 'inspect-nav next', 'aria-label': 'Next image', onClick: () => step(1) }, icon('chevronRight')));
    document.body.appendChild(rootEl);
    rootEl.focus({ preventScroll: true });

    const urlOf = (it) => fovea.api.imgUrl(it.id);
    const fetchPreds = (it) => pendingPreds(it.id).catch(() => []);
    function proposalsFor(it) {
      // List rows say whether an image has pending suggestions: only those cost a request.
      if (!it || !((it.ai_pending || 0) > 0 || (it.ai_suspect || 0) > 0)) return Promise.resolve([]);
      if (!predCache.has(it.id)) {
        const p = fetchPreds(it);
        predCache.set(it.id, p);
        if (predCache.size > 12) predCache.delete(predCache.keys().next().value);
      }
      return predCache.get(it.id);
    }
    /** The details of an image's QA flags (which box jumped, where the gap is, the twin…): fetched only for images
        whose list row carries such a flag. */
    function qaFor(it) {
      if (!it || !(it.flags || []).some(c => DETAIL_FLAGS.has(c))) return Promise.resolve(null);
      if (!qaCache.has(it.id)) {
        qaCache.set(it.id, fovea.api.get(`/api/datasets/${ds.id}/images/${it.id}/qa`).then(r => (r && r.flags) || [], () => null));
        if (qaCache.size > 12) qaCache.delete(qaCache.keys().next().value);
      }
      return qaCache.get(it.id);
    }
    function prefetchAround(n) {
      c0.prefetch(n, 3);
      const near = [n + 1, n + 2, n - 1].map(k => c0.items[k]).filter(Boolean);
      frames.prefetch(near.map(urlOf));
      near.slice(0, 1).forEach((it) => { proposalsFor(it); qaFor(it); });
    }
    /** Show image n. The current picture stays until the next one is decoded; then picture, boxes, proposals and
        panel change together in one frame. The latest request wins: a held arrow key skips frames. */
    async function show(n) {
      if (closed) return false;
      if (c0.total != null) n = Math.min(c0.total - 1, n);
      n = Math.max(0, n);
      const my = ++seq; want = n;
      posEl.textContent = `${fmt.num(n + 1)} / ${fmt.num(c0.total)}`;      // the keypress registers at once
      waitEl.replaceChildren(ui.spinner());
      let it = null, im = null, pr = [], qd = null;
      try {
        it = await c0.ensure(n);
        if (it && my === seq && !closed) [im, pr, qd] = await Promise.all([frames.get(urlOf(it)).catch(() => null), proposalsFor(it), qaFor(it)]);
      } catch (e) { if (my === seq && !closed) status(e.message, 'error'); }
      if (closed || my !== seq) return false;
      waitEl.replaceChildren();
      if (!it) { want = idx; posEl.textContent = `${fmt.num(idx + 1)} / ${fmt.num(c0.total)}`; return false; }
      const sameImage = item && item.id === it.id;
      if (!sameImage && noteDirty && noteEl && item) saveNote(item, noteEl.value);   // auto-play while typing
      idx = n; item = it; preds = pr || []; qaDetail = qd || [];
      if (!sameImage) { activeBox = firstIssueBox(it); hoverBox = -1; zoomedTo = null; flagFocus = null; }
      paint(im);
      prefetchAround(n);
      if (cursor === c0) setFocus(idx, { scroll: true, dom: false });
      scheduleSimilar(it);
      return true;
    }
    function showId(id) { const i = c0.indexOfId(id); if (i >= 0) show(i); else c0.positionOf(id).then(j => { if (j >= 0) show(j); }); }
    function step(delta) { stopAuto(); show(want + delta); }
    /** The box to select when an image opens: the first one a label issue or a QA flag points at. */
    function firstIssueBox(it) {
      if (!it.boxes || !it.boxes.length || !problemCodes(it).length) return -1;
      const k = boxIssues(it.boxes, names.length).findIndex(x => x.length);
      if (k >= 0) return k;
      const rows = [...qaRowNotes().keys()].filter(r => r < it.boxes.length).sort((a, b) => a - b);
      return rows.length ? rows[0] : -1;
    }

    // ---------------- QA flag details: which box, where it should be, which frames to compare
    const flagOf = (code) => qaDetail.find(f => f.code === code) || null;
    /** Rows of the label file a QA flag names: row -> [short notes] (seq_jump "jumped", shape_outlier "odd shape" …). */
    function qaRowNotes() {
      const out = new Map(), add = (r, text) => { if (!out.has(r)) out.set(r, []); out.get(r).push(text); };
      for (const f of qaDetail) for (const r of flagRows(f)) add(r, ROW_FLAG_TEXT[f.code] || f.code);
      const tw = twinNow();
      if (tw) for (const x of tw.mine) add(x.row, x.kind === 'class' ? `${className(names, tw.boxes[x.other][0])} there` : x.kind === 'moved' ? 'placed differently' : 'only here');
      return out;
    }
    const flagRows = qaFlagRows;
    /** Rows a chip points at: a QA flag's rows (for a mismatch: the boxes the twin does not match), or the boxes
        with that label-file issue. */
    function rowsForCode(code) {
      if (code === 'dup_label_mismatch') { const tw = twinNow(); return tw ? tw.mine.map(x => x.row) : []; }
      const f = flagOf(code);
      if (f) return flagRows(f);
      const bx = (item && item.boxes) || [];
      return boxIssues(bx, names.length).map((x, i) => (x.includes(code) ? i : -1)).filter(i => i >= 0);
    }
    /** Where the frames around this one put a box: seq_jump's expected positions, seq_gap's missing boxes. */
    function expectedBoxes() {
      const f = flagFocus && flagOf(flagFocus);
      if (!f) return [];
      const d = f.detail || {};
      const list = f.code === 'seq_jump' ? (d.jumps || []).map(j => j.expected) : f.code === 'seq_gap' ? d.boxes || [] : [];
      return list.filter(b => Array.isArray(b) && b.length >= 5);
    }
    function focusFlag(code) {
      flagFocus = flagFocus === code ? null : code;
      if (flagFocus) { const rows = rowsForCode(code); if (rows.length) activeBox = rows[0]; }
      if (flagFocus === 'dup_label_mismatch') loadTwin();
      renderOverlay(); renderSide();
    }
    // ---------------- dup_label_mismatch: the twin's boxes, compared box by box (read when its chip is pressed)
    function twinBoxes(id) {
      if (!twinCache.has(id)) {
        // a failed read is not kept: pressing the chip again asks again
        twinCache.set(id, fovea.api.get(`/api/datasets/${ds.id}/images/${id}`).then(r => (r && r.item && Array.isArray(r.item.boxes) ? r.item.boxes : null), () => { twinCache.delete(id); return null; }));
        if (twinCache.size > 8) twinCache.delete(twinCache.keys().next().value);
      }
      return twinCache.get(id);
    }
    /** The IoU at which the QA engine calls two boxes the same (a per-dataset QA setting). */
    const mismatchIou = () => (twinIouP = twinIouP || fovea.api.get(`/api/datasets/${ds.id}/qa/settings`)
      .then(r => { const v = Number(r && r.options && r.options.mismatch_iou); return v > 0 && v < 1 ? v : 0.5; }, () => 0.5));
    async function loadTwin() {
      const d = (flagOf('dup_label_mismatch') || {}).detail || {};
      if (!item || d.twin == null) { twin = null; return; }
      const forId = item.id, id = d.twin;
      if (twin && twin.forId === forId && twin.id === id && twin.boxes !== null) return;   // read already, or on its way
      twin = { forId, id, boxes: undefined, minIou: 0.5 };
      const [boxes, minIou] = await Promise.all([twinBoxes(id), mismatchIou()]);
      if (closed || !twin || twin.forId !== forId || twin.id !== id) return;
      twin = { forId, id, boxes, minIou };
      if (item && item.id === forId && flagFocus === 'dup_label_mismatch') {
        const rows = rowsForCode('dup_label_mismatch');
        if (rows.length && !rows.includes(activeBox)) activeBox = rows[0];
        renderOverlay(); renderSide();
      }
    }
    /** The pressed mismatch chip's comparison ({boxes: the twin's, mine, theirs} as twinDiff), once read; else null. */
    function twinNow() {
      if (flagFocus !== 'dup_label_mismatch' || !item || !twin || twin.forId !== item.id || !Array.isArray(twin.boxes)) return null;
      // one render asks several times (rows, notes, overlay, legend): compare once per pair of box lists
      if (!twinMemo || twinMemo.mine !== item.boxes || twinMemo.twin !== twin) {
        twinMemo = { mine: item.boxes, twin, value: { boxes: twin.boxes, ...twinDiff(item.boxes || [], twin.boxes, twin.minIou) } };
      }
      return twinMemo.value;
    }
    /** What the pressed mismatch chip draws, in words: which boxes differ, with the dashed / highlighted key. */
    function twinLegend() {
      const line = (sw, text) => h('div', { class: 'lg' }, h('span', { class: `lg-sw ${sw}`, 'aria-hidden': 'true' }), h('span', text));
      if (!twin || twin.forId !== item.id || twin.boxes === undefined) return h('div', { class: 'twin-legend faint' }, 'Reading the other image’s labels…');
      if (!Array.isArray(twin.boxes)) return h('div', { class: 'twin-legend faint' }, 'The other image’s labels could not be read.');
      const tw = twinNow();
      if (!tw) return null;
      if (!tw.mine.length && !tw.theirs.length) return h('div', { class: 'twin-legend faint' }, 'The labels agree now. The check runs again a few seconds after a label edit.');
      const boxWord = (k) => `${k} box${k === 1 ? '' : 'es'}`;
      const namesOf = (list) => { const u = [...new Set(list.map(c => className(names, c)))]; return u.slice(0, 3).join(', ') + (u.length > 3 ? ` +${u.length - 3}` : ''); };
      const of = (list, kind) => list.filter(x => x.kind === kind);
      const there = of(tw.theirs, 'only'), here = of(tw.mine, 'only'), other = of(tw.mine, 'class'), moved = of(tw.mine, 'moved');
      return h('div', { class: 'twin-legend' },
        there.length ? line('ghost', `Dashed: ${boxWord(there.length)} only in the other image (${namesOf(there.map(x => tw.boxes[x.index][0]))})`) : null,
        other.length ? line('ghost', `${boxWord(other.length)} with another class there: ${other.slice(0, 2).map(x => `${className(names, item.boxes[x.row][0])} here, ${className(names, tw.boxes[x.other][0])} there`).join(' · ')}`) : null,
        moved.length ? line('ghost', `${boxWord(moved.length)} placed differently there (dashed: its place there)`) : null,
        here.length ? line('row', `Highlighted: ${boxWord(here.length)} only here (${namesOf(here.map(x => item.boxes[x.row][0]))})`) : null,
        h('div', { class: 'faint' }, h('span', { class: 'kbd' }, 'Esc'), ' hides these'));
    }
    /** Open another frame named by a flag (previous / next frame, a twin): in this list when it is there, else
        in its sequence (or on its own) as a new history entry, so Back returns to this list. */
    async function openFrame(id, seqName) {
      if (id == null) return;
      const i = c0.indexOfId(id) >= 0 ? c0.indexOfId(id) : await c0.positionOf(id);
      if (closed) return;
      if (i >= 0) { stopAuto(); show(i); return; }
      close({ fromHistory: true });
      const q = new URLSearchParams(seqName ? { seq: seqName } : { ids: String(id) });
      q.set('img', String(id)); q.set('view', 'inspect');
      fovea.router.navigate(`/d/${ds.id}/explore?${q.toString()}`);
    }
    const pct = (v) => `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`;
    /** One line on what the pressed flag means for this image, with the frames to compare. */
    function flagDetailEl(code) {
      const f = flagOf(code), d = (f && f.detail) || {};
      const n = rowsForCode(code).length, boxWord = (k) => `${k} box${k === 1 ? '' : 'es'}`;
      let text = '';
      if (code === 'seq_jump') text = `${boxWord(n)} moved off the path the frames before and after agree on (highlighted). Dashed: where those frames put it.`;
      else if (code === 'seq_gap') text = `An object labeled in the frames before and after has no box here. Dashed: where it would be.`;
      else if (code === 'seq_flicker') text = (d.rows || []).slice(0, 3).map(x => `${className(names, x.cls)} here, ${className(names, x.expected)} in the frames around it`).join(' · ') + ' (highlighted).';
      else if (code === 'seq_count_spike') text = `${fmt.num(d.n ?? 0)} boxes here, ${fmt.num(d.prev ?? 0)} in the frame before and ${fmt.num(d.next ?? 0)} after.`;
      else if (code === 'shape_outlier') text = (d.rows || []).slice(0, 3).map(x => `${className(names, x.cls)}: ${x.kind === 'aspect' ? `${x.value > x.typical ? 'wider' : 'taller'} than usual (w/h ${x.value} vs ${x.typical})` : `${x.value > x.typical ? 'larger' : 'smaller'} than usual (${pct(x.value)} of the image vs ${pct(x.typical)})`}`).join(' · ') + ' (highlighted).';
      else if (code === 'dup_label_mismatch') text = `A near-identical image is labeled differently (${fmt.num(d.n_boxes ?? 0)} vs ${fmt.num(d.twin_n_boxes ?? 0)} boxes${d.reason && d.reason !== 'count' ? `, ${d.reason === 'class' ? 'other classes' : 'other boxes'}` : ''}).`;
      else if (code === 'leak_split') text = `A near-identical image is in ${(d.splits || []).join(', ') || 'another split'}.`;
      else if (n) text = `${boxWord(n)} with this problem (highlighted).`;
      if (!text) return null;
      const nav = [];
      if (d.prev_id != null) nav.push(h('button', { class: 'btn btn-sm', onClick: () => openFrame(d.prev_id, item.seq) }, icon('chevronLeft', 12), 'Frame before'));
      if (d.next_id != null) nav.push(h('button', { class: 'btn btn-sm', onClick: () => openFrame(d.next_id, item.seq) }, 'Frame after', icon('chevronRight', 12)));
      if (d.twin != null) nav.push(h('button', { class: 'btn btn-sm', 'data-tip': d.twin_path || '', onClick: () => openFrame(d.twin, null) }, icon('copy', 12), 'Open the other image'));
      return h('div', { class: 'flag-detail xs' }, h('div', text), code === 'dup_label_mismatch' && d.twin != null ? twinLegend() : null,
        nav.length ? h('div', { class: 'row wrap gap-6 mt-8' }, nav) : null);
    }
    function flagChip(code) {
      const label = issueLabels[code] || code;
      const can = DETAIL_FLAGS.has(code) ? !!flagOf(code) : rowsForCode(code).length > 0;
      if (!can) return ui.chip(label, { cls: 'issue-chip', icon: 'alert', title: code === 'near_dup' ? `${label}: see Similar images below` : label });
      const on = flagFocus === code;
      return h('button', { class: cls('chip issue-chip clickable', on && 'active'), type: 'button', 'aria-pressed': String(on),
        'data-tip': on ? 'Hide the details' : 'Show which box, and the frames to compare', onClick: () => focusFlag(code) }, icon('alert', 12), label);
    }
    async function refreshQa() {
      if (!item) return;
      const target = item;
      qaCache.delete(target.id);
      const qd = await qaFor(target);
      if (closed || item !== target) return;
      qaDetail = qd || [];
      if (flagFocus && !problemCodes(item).includes(flagFocus)) flagFocus = null;
      twinCache.clear(); twin = null;                     // labels changed somewhere: compare with the twin afresh
      if (flagFocus === 'dup_label_mismatch') loadTwin();
      renderOverlay(); renderSide();
    }

    // ---------------- picture, zoom and pan
    function paint(im) {
      if (im) { im.alt = ''; im.draggable = false; natural = [im.naturalWidth || 0, im.naturalHeight || 0]; } else natural = [0, 0];
      picHolder.className = `pic ${im ? 'is-ready' : 'is-error'}`;
      picHolder.replaceChildren(...[im, ...overlays()].filter(Boolean));
      nameEl2.textContent = item.rel_path; nameEl2.title = item.rel_path;
      posEl.textContent = `${fmt.num(idx + 1)} / ${fmt.num(c0.total)}`;
      overlayToggle.classList.toggle('active', showOverlay);
      layoutPic();
      renderSide();
      syncUrl();
    }
    const stageSize = () => [stage.clientWidth, stage.clientHeight];
    const fitScale = () => { const [sw, sh] = stageSize(); return natural[0] ? Math.min((sw - 24) / natural[0], (sh - 24) / natural[1]) : 1; };
    function layoutPic() {
      const [nw, nh] = natural;
      if (!nw || !nh) { picHolder.style.cssText = ''; zoomBtn.textContent = 'Fit'; return; }
      const [sw, sh] = stageSize();
      const s = fitScale() * view.z, w = nw * s, hh = nh * s;
      let x = sw / 2 - view.cx * w, y = sh / 2 - view.cy * hh;
      x = w <= sw ? (sw - w) / 2 : clamp(x, sw - w, 0);
      y = hh <= sh ? (sh - hh) / 2 : clamp(y, sh - hh, 0);
      view.cx = (sw / 2 - x) / w; view.cy = (sh / 2 - y) / hh;   // what is on screen after clamping
      picHolder.style.cssText = `left:${x.toFixed(1)}px;top:${y.toFixed(1)}px;width:${w.toFixed(1)}px;height:${hh.toFixed(1)}px`;
      picHolder.classList.toggle('pixelated', s > 2);           // past 2× show real pixels, not a blur
      zoomBtn.textContent = view.z <= 1.001 ? 'Fit' : `${Math.round(s * 100)}%`;
      stage.classList.toggle('zoomed', view.z > 1.001);
    }
    function zoomAt(z, px, py) {
      const [nw, nh] = natural; if (!nw) return;
      const [sw, sh] = stageSize();
      const r = picHolder.getBoundingClientRect(), sr = stage.getBoundingClientRect();
      const lx = px - sr.left, ly = py - sr.top;
      const fx = (px - r.left) / r.width, fy = (py - r.top) / r.height;   // image point under the pointer stays put
      view.z = clamp(z, 1, 40);
      const s = fitScale() * view.z;
      view.cx = (sw / 2 - lx) / (nw * s) + fx; view.cy = (sh / 2 - ly) / (nh * s) + fy;
      zoomedTo = null;
      layoutPic();
    }
    function zoomCenter(f) { const r = stage.getBoundingClientRect(); zoomAt(view.z * f, r.left + r.width / 2, r.top + r.height / 2); }
    function fitView() { view.z = 1; view.cx = 0.5; view.cy = 0.5; zoomedTo = null; layoutPic(); }
    /** 100 %: one image pixel per screen pixel (or fit, when the image is smaller than the stage anyway). */
    function actualPixels() { const f = fitScale(); if (!natural[0] || f >= 1) { fitView(); return; } view.z = 1 / f; zoomedTo = null; layoutPic(); }
    function zoomToBox() {
      const i = activeBox >= 0 ? activeBox : hoverBox >= 0 ? hoverBox : (item && item.boxes && item.boxes.length ? 0 : -1);
      const b = i >= 0 && item.boxes[i];
      if (!b || zoomedTo === `${item.id}:${i}`) { fitView(); return; }
      const [sw, sh] = stageSize(), f = fitScale();
      const bw = Math.max(4, b[3] * natural[0] * f), bh = Math.max(4, b[4] * natural[1] * f);
      view.z = clamp(Math.min(sw * 0.55 / bw, sh * 0.55 / bh), 1, 40); view.cx = b[1]; view.cy = b[2];
      activeBox = i; zoomedTo = `${item.id}:${i}`;
      layoutPic(); renderOverlay();
    }
    stage.addEventListener('wheel', (e) => {
      if (!natural[0]) return;
      e.preventDefault();
      zoomAt(view.z * Math.exp(-e.deltaY * (e.ctrlKey ? 0.01 : 0.0018)), e.clientX, e.clientY);
    }, { passive: false });
    stage.addEventListener('dblclick', (e) => { if (e.target.closest('.inspect-nav')) return; if (view.z > 1.001) fitView(); else zoomAt(3, e.clientX, e.clientY); });
    stage.addEventListener('pointerdown', (e) => {
      if (e.button !== 0 || view.z <= 1.001 || e.target.closest('.inspect-nav')) return;
      const sx = e.clientX, sy = e.clientY, cx0 = view.cx, cy0 = view.cy, r = picHolder.getBoundingClientRect();
      let moved = false;
      stage.setPointerCapture(e.pointerId);
      const move = (ev) => { if (!moved && Math.hypot(ev.clientX - sx, ev.clientY - sy) < 3) return; moved = true; stage.classList.add('panning'); view.cx = cx0 - (ev.clientX - sx) / r.width; view.cy = cy0 - (ev.clientY - sy) / r.height; layoutPic(); };
      const up = () => { stage.removeEventListener('pointermove', move); stage.removeEventListener('pointerup', up); stage.removeEventListener('pointercancel', up); stage.classList.remove('panning'); if (moved) stage.addEventListener('click', (ev) => ev.stopPropagation(), { capture: true, once: true }); };
      stage.addEventListener('pointermove', move); stage.addEventListener('pointerup', up); stage.addEventListener('pointercancel', up);
    });
    const ro2 = new ResizeObserver(() => { if (!closed) layoutPic(); });
    ro2.observe(stage);

    // ---------------- boxes and AI suggestions
    const isAudit = (p) => p.kind in AUDIT_GLYPH;
    /** A proposal over an existing box of its class adds nothing ("already labeled"): hidden, as in Annotate (the
        accept job skips it too). Audit findings and refinements point at existing boxes on purpose. */
    const isDup = (p) => duplicateOf(p, item && item.boxes);
    const inClass = (p) => isolate == null || p.cls === isolate;
    // Only the shown image's own suggestions, ever (a late answer for another image must not be drawn or decided here).
    const ownPred = (p) => !!item && (p.image_id == null || p.image_id === item.id);
    const visiblePreds = () => preds.filter(p => ownPred(p) && (isAudit(p) || p.conf == null || p.conf >= propThreshold) && inClass(p) && !isDup(p));
    function overlays() {
      if (!item || !showOverlay) return [];
      const out = [];
      const filter = isolate != null ? new Set([isolate]) : null;
      if (item.boxes && item.boxes.length) {
        const iss = boxIssues(item.boxes, names.length), qn = qaRowNotes();
        out.push(fovea.boxLayer(item.boxes, { names, labels: showLabels, filter, active: activeBox,
          onClick: (i) => { activeBox = activeBox === i ? -1 : i; renderOverlay(); },
          styleFor: (b, i) => (iss[i].length || qn.has(i) ? { label: `⚠ ${className(names, b[0])}` } : null) }));
      }
      // Where the frames before and after put a box that jumped or is missing here (the pressed flag chip).
      const ex = expectedBoxes();
      if (ex.length) out.push(ghostLayer(ex, () => (flagFocus === 'seq_gap' ? 'missing here?' : 'expected here'), filter));
      // The near-identical image's boxes that have no match here (the pressed dup_label_mismatch chip).
      const tw = twinNow();
      if (tw && tw.theirs.length) out.push(ghostLayer(tw.theirs.map(x => tw.boxes[x.index]), (b) => `other image: ${className(names, b[0])}`, filter));
      const vp = visiblePreds();
      if (vp.length) {
        out.push(fovea.boxLayer(vp.map(p => [p.cls, p.xc, p.yc, p.w, p.h]), { names, labels: showLabels, stroke: 1.5, dim: true,
          styleFor: (b, i) => { const p = vp[i]; return { dashed: true, color: classColor(p.cls), fill: 'transparent', label: predLabel(p) }; } }));
      }
      out.forEach(o => o.classList.add('inspect-ov'));
      return out;
    }
    /** The box a finding is about in the labels as they are now (rows shift when a box above is deleted); null: it is
        gone since the audit. The accept job applies a fix to that same box (refRow). */
    const refOf = (p) => refRow(p, item && item.boxes);
    /** The box of another class a proposal lands on ("⇄ #n"), -1 for none (conflictOf). */
    const conflictAt = (p) => conflictOf(p, item && item.boxes);
    /** What ⇧Enter would do with the shown proposals: `left` lands on a box of another class and stays to decide. */
    const planOf = (vp) => acceptPlan(vp.filter(p => !isAudit(p)), item && item.boxes);
    const predLabel = (p) => findingLabel(p, (c) => className(names, c), { row: refOf(p), conflict: conflictAt(p) });
    /** Boxes another frame puts here (a pressed flag chip): dashed amber, each label UNDER its box, so it never covers
        the label of this image's own box at the same place (those sit above their box). */
    function ghostLayer(boxes, labelOf, filter) {
      const layer = fovea.boxLayer(boxes, { names, labels: showLabels, stroke: 1.5, fill: false, filter,
        styleFor: (b) => ({ dashed: true, color: GHOST, fill: 'transparent', label: labelOf(b) }) });
      for (const lb of layer.querySelectorAll('.bx-label')) {
        const bx = lb.parentElement;
        if (parseFloat(bx.style.top) + parseFloat(bx.style.height) > 93) continue;   // no room below: above, as usual
        Object.assign(lb.style, { top: '100%', bottom: 'auto', marginBottom: '0', borderRadius: '0 0 3px 3px' });
      }
      return layer;
    }
    function renderOverlay() {
      if (!item) return;
      picHolder.querySelectorAll('.inspect-ov').forEach(o => o.remove());
      overlays().forEach(o => picHolder.appendChild(o));
      overlayToggle.classList.toggle('active', showOverlay); overlayToggle.setAttribute('aria-pressed', String(showOverlay));
      const flagged = new Set(flagFocus ? rowsForCode(flagFocus) : []);
      side.querySelectorAll('.label-row[data-i]').forEach((r) => { const i = Number(r.dataset.i); r.classList.toggle('active', i === activeBox); r.classList.toggle('flagged', flagged.has(i)); });
    }
    /** Write the visible proposals (or, with audit=true, the suggested fixes) into this image's label file. */
    async function acceptVisible(audit = false) {
      const vis = visiblePreds().filter(p => isAudit(p) === audit);
      if (!vis.length) { status(audit ? 'No suggested fixes here' : 'No proposals to accept here'); return; }
      // As Annotate's ⇧Enter: a proposal over a box of another class (⇄) is not accepted in bulk — the job leaves it
      // pending too (skip_conflicts) — and a fix whose box is gone since the audit has nothing left to apply to.
      const acc = audit ? vis.filter(p => refOf(p) !== null) : (({ take, dupes }) => [...take, ...dupes])(planOf(vis));
      if (!acc.length) {
        // Short (the status line holds ~44 characters); the panel says what to do with them.
        status(audit ? 'Nothing to apply: label changed since audit' : `Nothing accepted: ${vis.length} over another class (⇄)`, 'warn');
        return;
      }
      const target = item, name = target.rel_path.split('/').pop();
      const before = preds.filter(ownPred).map(p => p.id);      // what the server decides is read back afterwards
      status(`${audit ? 'Applying' : 'Accepting'} ${acc.length}…`);
      try {
        const kinds = audit ? [...new Set(acc.map(p => p.kind))] : PROPOSAL_KINDS;
        const r = await fovea.api.post(`/api/datasets/${ds.id}/proposals/accept`, { image_ids: [target.id], min_conf: !audit && propThreshold ? propThreshold : undefined, kinds, classes: isolate != null ? [isolate] : undefined, skip_conflicts: true });
        const snap = r && r.job ? await fovea.api.watchJob(r.job.id || r.job) : r;
        predCache.delete(target.id);
        noteLabelEdit(ds.id, target.id); watchQa();          // the label file changed: its QA flags follow
        const now = await reloadItem(target.id);
        const res = (snap && snap.result) || {};
        // ⌘Z takes it back: the label file from the snapshot the accept saved first, the suggestions to pending.
        const still = new Set(((now && now.preds) || []).map(p => p.id));
        const decided = now ? before.filter(id => !still.has(id)) : [];
        if (decided.length || res.snapshot_id != null) {
          pushUndoEntry({ kind: 'decide', label: audit ? `Applied ${fmt.plural(acc.length, 'fix', 'fixes')}` : `Accepted ${fmt.plural(acc.length, 'proposal')}`,
            name, imageId: target.id, ids: decided, snapshot: res.snapshot_id ?? null, changed: !!(res.added || res.modified || res.deleted) });
        }
        status(`${acceptSummary(res)} · ${name}`, res.conflicts ? 'warn' : 'ok');
      } catch (e) { status(`${audit ? 'Applying fixes' : 'Accept'} failed: ${e.message}`, 'error'); }
    }
    async function rejectVisible() {
      const vp = visiblePreds();
      if (!vp.length) { status('No AI suggestions to reject here'); return; }
      const target = item, name = target.rel_path.split('/').pop();
      try {
        await fovea.api.post(`/api/datasets/${ds.id}/proposals/decide`, { accept: [], reject: vp.map(p => p.id) });
        // Rejected suggestions are never offered again, so the rejection must be undoable (⌘Z sets them pending).
        pushUndoEntry({ kind: 'decide', label: `Rejected ${fmt.plural(vp.length, 'AI suggestion')}`, name, imageId: target.id, ids: vp.map(p => p.id) });
        predCache.delete(target.id);
        const gone2 = new Set(vp.map(p => p.id));
        if (item === target) { preds = preds.filter(p => !gone2.has(p.id)); renderOverlay(); renderSide(); }
        const it = byIdItem(target.id);
        if (it) {
          it.ai_pending = Math.max(0, (it.ai_pending || 0) - vp.filter(p => !isAudit(p)).length);
          it.ai_suspect = Math.max(0, (it.ai_suspect || 0) - vp.filter(isAudit).length);
          const i = cursor.indexOfId(target.id); if (i >= 0) refreshTile(i);
        }
        status(`Rejected ${vp.length} · ${name} · ${fovea.MOD} Z undoes`);
      } catch (e) { status(`Reject failed: ${e.message}`, 'error'); }
    }
    /** Re-read an image after a change (see refreshImage, which also updates Inspect through imageChanged). */
    async function reloadItem(id) { try { return await refreshImage(id); } catch (e) { return null; } }
    /** The image's labels or its pending suggestions changed (an accept, a rejection or their undo). */
    function imageChanged(id, row, left) {
      if (closed) return;
      predCache.set(id, Promise.resolve(left));
      if (item && item.id === id) { item = row || item; preds = left; renderOverlay(); renderSide(); }
    }
    function setThreshold(v) {
      propThreshold = clamp(Math.round(v * 100) / 100, 0, 1);
      ls.set(thrKey, String(propThreshold));
      renderOverlay(); renderSide();
    }
    /** While the slider is dragged: boxes and the counts line follow (once a frame); the panel is rebuilt on release,
        so the slider is never replaced under the pointer. */
    let thrRaf = 0, thrNext = null;
    function setThresholdLive(v) {
      thrNext = v;
      if (thrRaf) return;
      thrRaf = requestAnimationFrame(() => {
        thrRaf = 0;
        if (closed || thrNext == null) return;
        propThreshold = clamp(Math.round(thrNext * 100) / 100, 0, 1); thrNext = null;
        ls.set(thrKey, String(propThreshold));
        renderOverlay();
        const t = side.querySelector('.props .prop-stats'); if (t) t.textContent = propStatsText();
      });
    }
    /** "3 shown · 1 below 0.25 · 2 already labeled" (the same words as Annotate's Proposals panel). */
    function propStatsText() {
      const vp = visiblePreds();
      const below = preds.filter(p => !isAudit(p) && inClass(p) && p.conf != null && p.conf < propThreshold).length;
      const dup = preds.filter(p => inClass(p) && !(p.conf != null && !isAudit(p) && p.conf < propThreshold) && isDup(p)).length;
      return [`${vp.length} shown`, below ? `${below} below ${propThreshold.toFixed(2)}` : '', dup ? `${dup} already labeled` : ''].filter(Boolean).join(' · ');
    }

    // ---------------- side panel
    let noteEl = null, noteState = null;
    const reviewLine = (it) => {
      const s = reviewStatus(it);
      if (drafts.has(it.id)) return 'Draft note — saved with your review (A, F or X)';
      return s ? `${REVIEW[s].label} ${fmt.ago(it.review.updated_at)}` : noteOf(it) ? 'Unreviewed · note saved' : 'Unreviewed';
    };
    function renderSide() {
      if (!item) return;
      const rv = item.review, s = reviewStatus(item);
      const keepScroll = side.scrollTop;
      side.replaceChildren();
      noteEl = h('textarea', { class: 'input mt-8', rows: 2, placeholder: 'Note — Enter saves, ⇧Enter new line', 'aria-label': 'Review note', value: drafts.has(item.id) ? drafts.get(item.id) : noteOf(item) });
      noteState = h('div', { class: 'xs faint note-state' }, reviewLine(item));
      noteEl.addEventListener('input', () => { noteDirty = true; noteState.textContent = 'Unsaved note'; });
      noteEl.addEventListener('blur', () => { if (noteDirty) saveNote(item, noteEl.value); });
      noteEl.addEventListener('keydown', (e) => {
        if (e.isComposing || e.keyCode === 229) return;
        if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); noteEl.blur(); rootEl.focus({ preventScroll: true }); }
        else if (e.key === 'Enter' && (e.metaKey || e.ctrlKey)) { e.preventDefault(); const t = item; saveNote(t, noteEl.value).then(() => reviewShown(t, 'approved', true)); noteEl.blur(); rootEl.focus({ preventScroll: true }); }
        else if (e.key === 'Enter' && !e.shiftKey) { e.preventDefault(); noteEl.blur(); rootEl.focus({ preventScroll: true }); }
      });
      side.appendChild(h('div', { class: 'sec' }, h('h4', 'Review'),
        h('div', { class: 'review-row' }, ['approved', 'flagged', 'excluded'].map(x => h('button', { class: cls('btn btn-sm review-btn', x, s === x && 'active'), 'aria-pressed': String(s === x), 'aria-keyshortcuts': REVIEW[x].key, 'data-tip': `${REVIEW[x].label} (${REVIEW[x].key})`,
          onClick: () => reviewShown(item, s === x ? null : x) }, icon(REVIEW[x].icon, 13), h('span', { class: 'truncate' }, { approved: 'Approve', flagged: 'Flag', excluded: 'Exclude' }[x])))),
        noteEl, noteState));
      side.appendChild(proposalsSection());
      side.appendChild(labelsSection());
      side.appendChild(imageSection());
      side.appendChild(similarSection());
      side.appendChild(h('div', { class: 'sec xs faint hints' }, '← → navigate · A F X review · U clear · N note · Z zoom to box · O boxes · L labels · J next label problem · E edit · Space play · ? all keys'));
      side.scrollTop = keepScroll;
    }
    function proposalsSection() {
      const vp = visiblePreds(), acc = vp.filter(p => !isAudit(p)), fixes = vp.filter(isAudit);
      // ⇧Enter leaves a proposal over a box of another class (⇄) to decide, and a fix whose box is gone has nothing
      // to apply to: the buttons count what they will do.
      const left = new Set(planOf(vp).left), nAcc = acc.length - left.size;
      const live = fixes.filter(p => refOf(p) !== null), gone = fixes.length - live.length;
      const sorted = [...vp].sort((a, b) => (b.conf ?? 2) - (a.conf ?? 2));
      const hasConf = preds.some(p => !isAudit(p) && p.conf != null);
      const thrLabel = h('span', { class: 'mono xs thr-val' }, propThreshold.toFixed(2));
      // ⇧Enter accepts the proposals; with none shown it applies the suggested fixes: the hint sits on that button.
      const enterHint = h('span', { class: 'xs faint' }, ' ⇧↵');
      return h('div', { class: 'sec props' },
        h('h4', 'Proposals', preds.length ? h('span', { class: 'badge', style: 'margin-left:6px' }, String(preds.length)) : null),
        preds.length ? h('div', { class: 'xs prop-stats' }, propStatsText()) : null,
        hasConf ? h('div', { class: 'thr' }, h('span', { class: 'xs faint' }, 'Show ≥'),
          h('input', { type: 'range', class: 'slider', min: 0, max: 1, step: 0.01, value: propThreshold, 'aria-label': 'Proposal confidence threshold',
            onInput: (e) => { thrLabel.textContent = Number(e.target.value).toFixed(2); setThresholdLive(Number(e.target.value)); },
            onPointerdown: (e) => { e.target._ptr = true; },
            onChange: (e) => {
              const kbd = document.activeElement === e.target && !e.target._ptr;   // arrow keys: the rebuilt slider keeps the focus
              if (!closed) setTimeout(() => { if (closed) return; renderSide(); const s2 = kbd && side.querySelector('.props .thr .slider'); if (s2) s2.focus({ preventScroll: true }); }, 0);
            } }),
          thrLabel, h('span', { class: 'kbd' }, '{'), h('span', { class: 'kbd' }, '}')) : null,
        sorted.slice(0, 30).map(p => {
          // A finding about an existing box (tighter, other class, no support) selects that box on click; so does a
          // proposal over a box of another class (the box it would sit on).
          const r = refOf(p), cf = conflictAt(p), at = r != null && r >= 0 ? r : cf;
          const tip = r != null && r >= 0 ? `About box #${r + 1} in Labels (click to select it)`
            : r === null ? 'The box this was about was removed or changed since the audit: nothing to apply (Del dismisses it)'
            : cf >= 0 ? `Lands on box #${cf + 1} (${className(names, item.boxes[cf][0])}), a box of another class: ⇧Enter leaves it — decide in Annotate (E)`
            : left.has(p) ? 'Lands on a more confident proposal of another class: ⇧Enter leaves it — decide in Annotate (E)' : null;
          return h('div', { class: cls('label-row prop-row', at >= 0 && 'has-ref', left.has(p) && 'conflict'), 'data-tip': tip,
            onClick: at >= 0 ? () => { activeBox = at; renderOverlay(); } : null },
            h('span', { class: 'swatch dashed', style: `border-color:${classColor(p.cls)}` }), h('span', { class: 'name' }, predLabel(p)),
            left.has(p) && cf < 0 ? h('span', { class: 'geo' }, '⇄ proposal') : null,
            p.reason && r !== null ? h('span', { class: 'geo truncate', title: p.reason, style: 'max-width:90px' }, p.reason) : null);   // a gone box: its own line says why
        }),
        vp.length ? h('div', { class: 'row wrap gap-6 mt-8' },        // three buttons do not fit the 320 px panel side by side
          nAcc ? h('button', { class: 'btn btn-sm grow', 'aria-keyshortcuts': 'Shift+Enter', 'data-tip': left.size ? `Accept the proposals that land on no box of another class (${left.size} ⇄ left to decide)` : null, onClick: () => acceptVisible() }, icon('check', 12), `Accept ${nAcc}`, enterHint) : null,
          live.length ? h('button', { class: 'btn btn-sm grow', 'aria-keyshortcuts': acc.length ? null : 'Shift+Enter', 'data-tip': fixesTip(live) + (acc.length ? ' · ⇧Enter applies them once the proposals are decided' : ''), onClick: () => acceptVisible(true) },
            icon('wand', 12), `Apply ${live.length} fix${live.length === 1 ? '' : 'es'}`, acc.length ? null : enterHint) : null,
          h('button', { class: 'btn btn-sm grow', 'aria-keyshortcuts': 'Delete', onClick: () => rejectVisible() }, icon('x', 12), `${acc.length ? 'Reject' : 'Dismiss'} ${vp.length}`, h('span', { class: 'xs faint' }, ' Del'))) : null,
        left.size ? h('div', { class: 'xs mt-8 prop-left' }, `⇄ ${left.size} land${left.size === 1 ? 's' : ''} on a box of another class: ⇧Enter leaves ${left.size === 1 ? 'it' : 'them'} — decide in Annotate (E).`) : null,
        gone ? h('div', { class: 'xs mt-8 prop-left' }, `${gone} fix${gone === 1 ? '' : 'es'} no longer appl${gone === 1 ? 'ies' : 'y'}: the label changed since the audit.`) : null,
        fixes.length ? h('div', { class: 'xs faint mt-8' }, '+ missed object · ⤢ tighter box · ⇄ other class · ✕ box without support. Or fix by hand in Annotate (E).') : null,
        h('div', { class: 'row mt-8' }, h('button', { class: 'btn btn-sm grow', 'aria-keyshortcuts': 'I', disabled: asking.has(item.id),
          'data-tip': 'Check this image’s labels with the AI source chosen in the AI panel (a cloud source receives the image)', onClick: () => askAI() },
          icon('sparkles', 12), asking.has(item.id) ? 'Asking AI…' : 'Ask AI to check the labels', h('span', { class: 'xs faint' }, ' I'))));
    }
    const asking = new Set();            // ids of the images whose "Ask AI" answer is on its way
    /** "Ask AI": the single-image label audit (/api/ai/review) with the source chosen in the AI panel. Each image
        has its own request: the answer is shown only on the image it was asked about. */
    async function askAI() {
      if (!item || asking.has(item.id)) return;
      const target = item;
      let m = null;
      try { m = await import('./ai-panel.js'); } catch (e) { m = null; }
      const prof = m && m.loadProfile ? await m.loadProfile(fovea, ds.id).catch(() => null) : null;
      const src = prof && ((prof.audit && prof.audit.source) || prof.source);
      if (!src || !src.provider || src.pipeline) { status('Choose an AI source first: Annotate → AI panel, or the AI tab'); return; }
      asking.add(target.id); renderSide();
      const clock = ui.elapsedClock();
      statusEl.replaceChildren(`Asking ${src.provider}${src.model ? ` · ${src.model}` : ''} `, clock); statusEl.dataset.status = ''; statusEl.title = '';
      try {
        const r = await fovea.api.post('/api/ai/review', { dataset_id: ds.id, image_id: target.id, provider: src.provider, model: src.model || '', options: src.effort ? { effort: src.effort } : undefined });
        clock.stop();
        predCache.delete(target.id);
        const it = byIdItem(target.id);
        if (it) it.ai_suspect = (r.preds || []).length;
        // Read the findings into a local first: the user may have moved to another image meanwhile.
        const fresh = await fetchPreds(target);
        predCache.set(target.id, Promise.resolve(fresh));
        if (!closed && item && item.id === target.id) { preds = fresh; renderOverlay(); }
        const i = cursor ? cursor.indexOfId(target.id) : -1; if (i >= 0) refreshTile(i);
        const cost = r.usage && r.usage.cost_usd ? ` · $${r.usage.cost_usd.toFixed(4)}` : '';
        status(`${(r.preds || []).length ? `AI suggests ${(r.preds || []).length} fix${(r.preds || []).length === 1 ? '' : 'es'}` : 'AI found nothing to fix'}${cost} · ${target.rel_path.split('/').pop()}`, 'ok');
      } catch (e) { clock.stop(); status(`Ask AI failed: ${e.message}`, 'error'); }
      finally { asking.delete(target.id); if (!closed) renderSide(); }
    }
    function fixesTip(list) {
      const c = (k) => list.filter(p => p.kind === k).length;
      return [c('missing') && `add ${c('missing')} box(es)`, c('loose') && `tighten ${c('loose')}`, c('swap') && `change the class of ${c('swap')}`, c('spurious') && `delete ${c('spurious')}`].filter(Boolean).join(', ') + ' — a snapshot is saved first (History can restore it)';
    }
    function labelsSection() {
      const bx = item.boxes || [];
      const iss = bx.length ? boxIssues(bx, names.length) : [];
      const qn = qaRowNotes(), flagged = new Set(flagFocus ? rowsForCode(flagFocus) : []);
      const imgIssues = problemCodes(item);
      // A row's markers: label-file problems first, then what a QA flag says about this box ("jumped", "odd shape").
      const marks = (i) => [...iss[i].map(c => ({ short: issueLabels[c] ? shortIssue(c) : c, long: issueLabels[c] || c })),
        ...(qn.get(i) || []).map(t => ({ short: t, long: t }))];
      return h('div', { class: 'sec' }, h('h4', 'Labels', h('span', { class: 'badge', style: 'margin-left:6px' }, String(item.n_boxes || bx.length))),
        bx.length ? h('div', { class: 'label-list' }, bx.map((b, i) => {
          const m = marks(i);
          return h('div', { class: cls('label-row', i === activeBox && 'active', flagged.has(i) && 'flagged', isolate != null && b[0] !== isolate && 'dim'), 'data-i': i, role: 'button', tabindex: '0',
            onMouseenter: () => { hoverBox = i; }, onMouseleave: () => { hoverBox = -1; },
            onClick: () => { activeBox = activeBox === i ? -1 : i; renderOverlay(); },
            // Enter selects and zooms; preventDefault keeps the generic role=button activation (lib/ui.js) from
            // clicking the row too, which would toggle the selection straight off again.
            onKeydown: (e) => { if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); activeBox = i; zoomToBox(); } } },
            h('span', { class: 'idx', 'aria-label': `Box ${i + 1}` }, `#${i + 1}`),        // the number AI findings refer to
            h('span', { class: 'swatch', style: `background:${classColor(b[0])}` }), h('span', { class: 'name' }, `${b[0]} · ${className(names, b[0])}`),
            m.length ? h('span', { class: 'box-issue', title: m.map(x => x.long).join(', ') }, '⚠ ', m[0].short, m.length > 1 ? ` +${m.length - 1}` : '') : null,
            h('span', { class: 'geo' }, `${(b[3] * 100).toFixed(0)}×${(b[4] * 100).toFixed(0)}%`));
        }))
          : h('div', { class: 'small faint' }, item.has_label ? 'Empty label file' : 'No label file'),
        imgIssues.length ? h('div', { class: 'row wrap gap-4 mt-8 flag-chips' }, imgIssues.map(flagChip)) : null,
        flagFocus && imgIssues.includes(flagFocus) ? flagDetailEl(flagFocus) : null);
    }
    const shortIssue = (c) => ({ tiny_box: 'tiny', out_of_range: 'outside', zero_area: 'zero area', bad_class: 'bad class', duplicate_box: 'duplicate' }[c] || c);
    /** "a/b/c.jpg" as text with a line-break opportunity after each slash. */
    const slashBreaks = (p) => pathParts(p).flatMap((s, i, a) => (i < a.length - 1 ? [s, h('wbr')] : [s]));
    function imageSection() {
      return h('div', { class: 'sec' }, h('h4', 'Image'), h('div', { class: 'kv' },
        h('b', 'Split'), h('span', item.split || '—'),
        h('b', 'Size'), h('span', item.width ? `${item.width} × ${item.height} · ${fmt.bytes(item.size)}` : fmt.bytes(item.size)),
        h('b', 'Sequence'), item.seq ? h('button', { class: 'linkish mono path', 'data-tip': 'Show only this sequence', onClick: () => { const id = item.id, sq = item.seq; close({ fromHistory: true }); setFilters({ seq: sq }, { keepId: id }); } }, ...slashBreaks(item.seq)) : h('span', '—'),
        h('b', 'ID'), h('span', { class: 'mono' }, `#${item.id} · position ${fmt.num(idx + 1)}`),
        // A path wraps after a slash, never inside a file name (a name broken as ".j" / "pg" reads as two names).
        h('b', 'Path'), h('span', { class: 'mono path', title: item.rel_path }, ...slashBreaks(item.rel_path))),
        h('div', { class: 'row mt-8' }, h('button', { class: 'btn btn-sm', onClick: () => copyPath(item.id) }, icon('copy', 12), 'Copy path'),
          h('button', { class: 'btn btn-sm', onClick: () => window.open(fovea.api.imgUrl(item.id), '_blank') }, icon('external', 12), 'Original')));
    }
    // ---------------- similar images (dHash neighbours from the QA engine)
    let simBody = null, simTimer = null, simFor = null, simItems = null, simSeq = 0, simNote = '…';
    /** A one-line state of the strip ("…", "Indexing…", an error), kept so a rebuilt panel shows it again. */
    function simSay(text) { simNote = text; if (simBody) simBody.replaceChildren(h('div', { class: 'xs faint' }, text)); }
    function similarSection() {
      simBody = h('div', { class: 'sim-body' });
      const head = h('button', { class: 'sec-toggle', 'aria-expanded': String(similarOpen), onClick: () => { similarOpen = !similarOpen; ls.set('fovea.inspect.similar', similarOpen ? '1' : '0'); renderSide(); if (similarOpen) scheduleSimilar(item, 0); } },
        icon(similarOpen ? 'chevronDown' : 'chevronRight', 12), 'Similar images');
      const sec = h('div', { class: 'sec similar' }, head, similarOpen ? simBody : null);
      if (similarOpen && simFor === item.id) { if (simItems) renderSimilar(simItems); else simSay(simNote); }
      return sec;
    }
    function scheduleSimilar(it, delay = 350, { again = false } = {}) {
      clearTimeout(simTimer);
      if (!similarOpen || !it) return;
      if (simFor === it.id && simItems) { renderSimilar(simItems); return; }
      // The strip belongs to this image from now on: an answer still on its way for the previous one is dropped
      // (it would list that image's neighbours — and the image itself — as this one's).
      simItems = null; simFor = it.id;
      const my = ++simSeq;
      if (!again) simSay('…');
      // Fetched only after the image has been on screen for a moment: stepping through frames costs nothing.
      simTimer = setTimeout(async () => {
        try {
          const r = await fovea.api.get(`/api/datasets/${ds.id}/images/${it.id}/similar`, { k: 12 });
          if (closed || my !== simSeq) return;
          if (r.pending && !(r.items || []).length) {
            // The dataset's first data-quality pass is still hashing images: say so, and look again shortly.
            const j = r.job;
            simSay(`Indexing image fingerprints for similarity${j && j.total ? ` · ${fmt.num(j.done)} / ${fmt.num(j.total)}` : ''}…`);
            simTimer = setTimeout(() => { if (!closed && item === it) scheduleSimilar(it, 0, { again: true }); }, 4000);
            return;
          }
          if (r.unreadable) { simSay('This image could not be read, so it has no fingerprint.'); return; }
          simItems = (r.items || []).filter(x => x.id !== it.id);
          renderSimilar(simItems);
        } catch (e) {
          if (closed || my !== simSeq) return;
          simSay(e.status === 404 && e.message === 'Not Found' ? 'Similarity search is not available on this server.' : `Could not load similar images: ${e.message}`);
        }
      }, delay);
    }
    function renderSimilar(items) {
      if (!simBody || !item) return;
      if (!items.length) { simBody.replaceChildren(h('div', { class: 'xs faint' }, 'No near-duplicates found. Run Data quality checks (Overview) if the dataset changed.')); return; }
      const grid = h('div', { class: 'sim-grid' }, items.map(s => {
        const im = h('img', { alt: '', decoding: 'async', loading: 'eager', src: fovea.api.thumbUrl(s.id, 128, s.mtime) });
        im.setAttribute('fetchpriority', 'low');
        const pic = h('div', { class: 'pic is-loading' }, im);
        trackImage(im, pic);
        return h('button', { class: 'sim', title: `${s.rel_path}${s.split ? ` · ${s.split}` : ''} · distance ${s.dist}`, onClick: () => jumpTo(s.id, items) }, pic,
          h('span', { class: 'd' }, s.dist === 0 ? 'same' : `d${s.dist}`), s.split && s.split !== item.split ? h('span', { class: 'sp' }, s.split) : null);
      }));
      simBody.replaceChildren(grid, h('button', { class: 'btn btn-sm mt-8', onClick: () => showSet(items) }, icon('grid', 12), `Show these ${items.length + 1} in the grid`));
    }
    async function jumpTo(id, items) {
      const i = c0.indexOfId(id) >= 0 ? c0.indexOfId(id) : await c0.positionOf(id);
      if (closed) return;
      if (i >= 0) { stopAuto(); show(i); return; }
      showSet(items, id);
    }
    function showSet(items, focus) {
      const ids = [item.id, ...items.map(s => s.id)].join(',');
      const at = focus || item.id;
      close({ fromHistory: true });
      fovea.router.navigate(`/d/${ds.id}/explore?ids=${ids}&img=${at}${focus ? '&view=inspect' : ''}`);
    }

    // ---------------- review, note
    function status(text, kind = '') { statusEl.textContent = text; statusEl.title = text; statusEl.dataset.status = kind; }
    /** Review the image on screen (never one still loading), confirmed in place. */
    function reviewShown(target, s, noAdvance = false) {
      if (!target) return Promise.resolve(false);
      // A note that could not be stored on its own (older server, unreviewed image) rides along with the review.
      const draft = drafts.get(target.id);
      return review({ ids: [target.id], count: 1 }, s, { quiet: true, note: s && draft != null ? draft : undefined }).then((ok) => {
        if (!ok || closed) return ok;
        if (s && draft != null) drafts.delete(target.id);
        const name = target.rel_path.split('/').pop();
        status(s ? `${REVIEW[s].label} · ${name}` : `Review cleared · ${name}`, s || '');
        if (!noAdvance && advance && s !== 'flagged' && !autoOn) setTimeout(() => { if (!closed && item === target && want === idx) show(idx + 1); }, 120);
        return ok;
      });
    }
    /** Save the note on its own (blur / Enter), keeping the review status. For an unreviewed image the note is
        sent without a status and read back: a server that keeps notes on unreviewed images stores it; otherwise
        it stays as a draft (shown in the panel) and is saved with the next review instead of being lost. */
    function saveNote(target, text) {
      if (!target) return Promise.resolve();
      noteDirty = false;
      const cur = byIdItem(target.id) || target;
      const prevNote = drafts.has(target.id) ? drafts.get(target.id) : noteOf(cur);
      if (text === prevNote) { if (noteState && item && item.id === target.id) noteState.textContent = reviewLine(cur); return Promise.resolve(); }
      return serial(async () => {
        const s = reviewStatus(byIdItem(target.id) || target);
        try {
          // A note on its own: the server keeps the status (and the reviewed time) as they are.
          await fovea.api.put(`/api/datasets/${ds.id}/review`, { image_ids: [target.id], note: text });
          if (!s) {
            // Read it back: an older server reads "no status" as "clear" and keeps nothing for unreviewed images.
            const { item: back } = await fovea.api.get(`/api/datasets/${ds.id}/images/${target.id}`);
            if (noteOf(back) !== text) {
              if (text) drafts.set(target.id, text); else drafts.delete(target.id);
              if (!closed && item && item.id === target.id && noteState) noteState.textContent = text ? 'Draft note — saved with your review (A, F or X)' : 'Unreviewed';
              return;
            }
          }
          drafts.delete(target.id);
          applyLocal([target.id], s, null, text);
          if (!closed && item && item.id === target.id && noteState) noteState.textContent = text ? 'Note saved' : 'Note removed';
          if (!closed) status(`${text ? 'Note saved' : 'Note removed'} · ${target.rel_path.split('/').pop()}`);
        } catch (e) { if (!closed) status(`Note not saved: ${e.message}`, 'error'); }
      });
    }

    // ---------------- auto-play (never more than 3 unrelated full-frame changes a second)
    function schedule() {
      clearTimeout(autoTimer);
      const next = c0.items[idx + 1];
      const sameSeq = next && item && item.seq && next.seq === item.seq;
      autoTimer = setTimeout(async () => {
        if (!autoOn || closed) return;
        if (c0.total != null && idx >= c0.total - 1) { stopAuto(); return; }
        await show(idx + 1);
        if (autoOn && !closed) schedule();
      }, sameSeq ? speed : Math.max(speed, MIN_SWAP_MS));
    }
    function setAutoBtn() { autoBtn.classList.toggle('active', autoOn); autoBtn.replaceChildren(icon(autoOn ? 'pause' : 'play', 13), autoOn ? 'Stop' : 'Play'); }
    function startAuto() { if (noteEl && document.activeElement === noteEl) noteEl.blur(); autoOn = true; setAutoBtn(); schedule(); }
    function stopAuto() { if (!autoOn && !autoTimer) return; autoOn = false; clearTimeout(autoTimer); autoTimer = null; setAutoBtn(); }
    function toggleAuto() { autoOn ? stopAuto() : startAuto(); }

    // ---------------- keyboard: window capture phase, ahead of the page's own handlers
    function trapTab(e) {
      const f = [...rootEl.querySelectorAll(FOCUSABLE)].filter(n => n.getClientRects().length);
      if (!f.length) { e.preventDefault(); rootEl.focus(); return; }
      const first = f[0], last = f[f.length - 1], at = document.activeElement;
      if (!rootEl.contains(at) || at === rootEl) { e.preventDefault(); (e.shiftKey ? last : first).focus(); }
      else if (e.shiftKey && at === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && at === last) { e.preventDefault(); first.focus(); }
    }
    const key = (e) => {
      // A dialog or the command palette above Inspect gets the keys first (its Esc must not also close Inspect).
      if (closed || ui.hasModal() || document.querySelector('body > .backdrop')) return;
      if (e.key === 'Tab') { trapTab(e); return; }
      if (isTyping() || chordGuard(e)) return;
      const done = () => { e.preventDefault(); e.stopPropagation(); };
      // Esc first hides what a pressed QA flag chip draws (dashed boxes, highlighted rows), then closes.
      if (keys.is(e, 'Escape')) { done(); if (flagFocus) focusFlag(flagFocus); else close(); return; }
      if (keys.is(e, 'mod+z')) { done(); undo(); return; }
      if (!item) return;
      if (keys.is(e, 'shift+Enter')) {
        done();
        // Proposals first; once none is shown, the same key applies the suggested fixes (audit findings).
        if (!e.repeat) { const vp = visiblePreds(); acceptVisible(!vp.some(p => !isAudit(p)) && vp.some(isAudit)); }
        return;
      }
      if (keys.is(e, 'Delete') && preds.length) { done(); if (!e.repeat) rejectVisible(); return; }
      if (keys.is(e, '{') || keys.is(e, '}')) { done(); setThreshold(propThreshold + (keys.is(e, '}') ? 0.05 : -0.05)); status(`Showing proposals ≥ ${propThreshold.toFixed(2)}`); return; }
      if (!keys.plain(e)) return;
      const k = e.key;
      // A Tab-focused slider (speed, proposal threshold) keeps its own arrows; letters still work there.
      if (['ArrowRight', 'ArrowDown', 'ArrowLeft', 'ArrowUp', 'Home', 'End', 'PageUp', 'PageDown'].includes(k) && controlKeys(document.activeElement, null).arrows) return;
      if (k === 'ArrowRight' || k === 'ArrowDown') { done(); step(1); }
      else if (k === 'ArrowLeft' || k === 'ArrowUp') { done(); step(-1); }
      else if (k === 'Home') { done(); stopAuto(); show(0); }
      else if (k === 'End') { done(); stopAuto(); if (c0.total) show(c0.total - 1); }
      else if (keys.is(e, 'Space')) { if (e.target.closest && e.target.closest('button, [role="button"]')) return; done(); if (!e.repeat) toggleAuto(); }
      else if (keys.is(e, 'o')) { done(); if (!e.repeat) toggleOverlay(); }
      else if (keys.is(e, 'l')) { done(); if (!e.repeat) labelsToggle.click(); }
      else if (keys.is(e, 'e')) { done(); if (!e.repeat) annotateId(item.id); }
      else if (keys.is(e, 'i')) { done(); if (!e.repeat) askAI(); }
      else if (keys.is(e, 'n')) { done(); if (noteEl) { noteEl.focus(); noteEl.setSelectionRange(noteEl.value.length, noteEl.value.length); } }
      else if (keys.is(e, 'z')) { done(); if (!e.repeat) zoomToBox(); }
      else if (keys.is(e, '=') || keys.is(e, 'shift+=')) { done(); zoomCenter(1.5); }
      else if (keys.is(e, '-')) { done(); zoomCenter(1 / 1.5); }
      else if (keys.is(e, 'j') || keys.is(e, 'shift+j')) { done(); if (!e.repeat) suspect(e.shiftKey ? -1 : 1); }
      else if (keys.digit(e) && !e.shiftKey) { done(); digitIsolate(keys.digit(e)); }
      else if (!e.shiftKey && keys.letter(e) in REVIEW_KEYS) { done(); if (!e.repeat) reviewShown(item, REVIEW_KEYS[keys.letter(e)]); }
    };
    async function suspect(dir) {
      const c = c0;
      for (let i = idx + dir, n = 0; i >= 0 && i < c.total && n < SUSPECT_SCAN; i += dir, n++) {
        const it = c.items[i] || await c.ensure(i).catch(() => null);
        if (closed) return;
        if (isSuspect(it)) { stopAuto(); show(i); return; }
      }
      status(`No image with a label problem in the next ${fmt.num(SUSPECT_SCAN)}`);
    }
    window.addEventListener('keydown', key, true);

    function close({ fromHistory = false, fromReload = false } = {}) {
      if (closed) return;
      if (noteDirty && noteEl && item) saveNote(item, noteEl.value);
      closed = true; stopAuto(); seq++; frames.clear(); clearTimeout(simTimer);
      window.removeEventListener('keydown', key, true);
      ro2.disconnect();
      rootEl.remove();
      inspect = null;
      imgLoader.resume();
      if (!fromReload && cursor === c0 && !destroyed) setFocus(idx, { center: !isOnScreen(idx), dom: true });
      else if (restoreFocus && restoreFocus.isConnected) try { restoreFocus.focus({ preventScroll: true }); } catch (e) { /* gone */ }
      if (destroyed) return;
      if (!fromHistory && inspectPushed && history.state && history.state.foveaInspect) { inspectPushed = false; backUntil = performance.now() + 1500; history.back(); }
      else { inspectPushed = false; syncUrl(true); }
    }
    show(index);
    return {
      close, show, showId,
      currentId: () => (item ? item.id : (c0.items[want] || {}).id),
      cursor: () => c0, imageChanged,
      // qa: the image's QA flags may have changed (a label re-check ran): read their details again.
      redraw: ({ qa = false } = {}) => { if (!item || closed) return; item = c0.items[idx] || item; if (qa) { qaCache.clear(); refreshQa(); } renderOverlay(); renderSide(); },
      status,
    };
  }

  // ---------------------------------------------------------------- lifecycle
  // Everything global is registered above and torn down below; the first page loads in the background so the
  // shell holds this teardown from the start — leaving mid-load must not leak key handlers or observers.
  const offScan = fovea.bus.on('dataset:scanned', (id) => {
    if (id !== ds.id || destroyed) return;
    loadStats();
    // A finished background scan never yanks the list away mid-review: refresh in place when idle, else ask.
    if (inspect || selCount() > 0) showBanner('The index was updated — new or changed images appear after a refresh.', 'Refresh', () => refresh({ keepSelection: true }));
    else refresh();
  });
  const offUpdated = fovea.bus.on('dataset:updated', (d) => { if (d && d.id === ds.id && Array.isArray(d.classes)) { names = d.classes; renderChips(); } });
  const offLabels = fovea.bus.on('labels:changed', ({ item } = {}) => {
    if (destroyed || !item) return;
    watchQa();                      // the edit was recorded at install (noteLabelEdit); its flags follow in ~3 s
    if (!cursor) return;
    const i = cursor.indexOfId(item.id);
    if (i < 0) { facetsChanged(); return; }
    cursor.items[i] = { ...cursor.items[i], ...item };
    if (matchesWith(cursor.items[i], filters)) gone.delete(item.id); else gone.add(item.id);
    refreshTile(i); updateCount();
    facetsChanged();
    if (inspect) inspect.redraw();
  });
  // A finished data-quality run changes the flag counts (and the flags of the tiles on screen). The dataset page
  // reports the jobs it watches (job:finished); a run started elsewhere shows up in the shell's job list.
  const qaRuns = new Set(), qaDone = new Set();
  function qaRunEnded(id) {
    if (destroyed || (id != null && qaDone.has(id))) return;
    if (id != null) qaDone.add(id);
    facetsChanged(0); loadStats();
    rereadFlags([], true);
  }
  const offJob = fovea.bus.on('job:finished', (j) => {
    if (j && j.dataset === ds.id && j.kind === 'qa') qaRunEnded(j.job && j.job.id);
  });
  const offJobs = fovea.bus.on('jobs:changed', ({ jobs: active = [], ended = [] } = {}) => {
    for (const j of active) if (j && j.kind === 'qa' && j.dataset_id === ds.id) qaRuns.add(j.id);
    for (const id of ended) if (qaRuns.delete(id)) qaRunEnded(id);
  });

  // Settings and the AI panel announce key and provider changes: Ask AI appears or goes away with them.
  const offProviders = fovea.bus.on('ai:providers', () => { if (!destroyed) loadAskProviders(); });

  updateControls();
  loadStats();
  reload({ keepId: st0.img, openInspect: st0.inspect });
  setTimeout(() => { if (!destroyed) loadAskProviders(); }, 0);   // after the first page is requested
  watchQa();                        // labels saved just before this view opened (in Annotate) are still being re-checked

  const destroy = () => {
    destroyed = true;
    clearTimeout(urlTimer); clearTimeout(pageTimer); clearTimeout(facetTimer); clearTimeout(qaTimer); clearTimeout(gridStatusTimer);
    if (facetCtl) facetCtl.abort();
    if (askBusy) askBusy.ctl.abort();
    endBand({ keepPreview: true });
    window.removeEventListener('pointermove', onBandMove);
    window.removeEventListener('pointerup', onBandUp);
    window.removeEventListener('pointercancel', endBand);
    document.removeEventListener('keydown', onKey);
    document.removeEventListener('keyup', onKeyUp);
    window.removeEventListener('blur', onBlur);
    const hi = fovea.router.before.indexOf(routeHook); if (hi >= 0) fovea.router.before.splice(hi, 1);
    ro.disconnect(); imgLoader.dispose(); closeQuick();
    if (inspect) inspect.close({ fromHistory: true, fromReload: true });
    offScan(); offUpdated(); offLabels(); offJob(); offJobs(); offProviders();
  };
  destroy.onQuery = (q) => applyQuery(q);    // for a shell that forwards same-tab query changes
  return destroy;
}
