// Annotate tab: zoomable canvas labeling with autosave, class palette, AI proposals and suspects, sequence tools.
//
// Rendering is layered so a pointer move never repaints the picture: the image sits on its own canvas (brightness /
// contrast are a CSS filter on it), every box and proposal that is not being touched is drawn once into an offscreen
// cache, and the visible overlay is that cache plus the hovered/selected item, handles, guides and drag previews,
// redrawn at most once per animation frame. Visual comfort (docs/visual-comfort.md): the current image stays until the
// next one is decoded, then picture, boxes and every panel change in the same frame; waits show after 400 ms only.
import { createFrameCache, decodedImage, thumbPx, trackImage } from '../lib/media.js';
import { EMPTY as NO_ANN, annSchema, attrText, attrFilterText, regionLabel } from '../lib/annattrs.js';
import { ISSUE_LABELS as BASE_ISSUE_LABELS } from './issues.js';

export function install(fovea) {
  fovea.registerTab({ id: 'annotate', label: 'Annotate', icon: 'pen', order: 30, render });
  const mod = fovea.MOD === '⌘' ? '⌘' : 'Ctrl+';
  if (fovea.keys && fovea.keys.registerKeys) fovea.keys.registerKeys('annotate', 'Annotate', KEYMAP.map(([k, d]) => [k.replace(/⌘/g, mod), d]), 30);
}

// One table documents the bindings for the ? modal and the hint line, so help cannot drift from behaviour.
const KEYMAP = [
  ['V B P W H', 'Tools: select · box · point · click-to-box (SAM) · pan'],
  ['B B', 'Box tool again: drag ⇄ 2-click (corner, then corner; zoom and pan in between)'],
  ['0-9', 'Class of the selected box (else for new boxes); a 2nd digit picks 10+'],
  ['[ ]', 'Previous / next class'], ['/', 'Search classes'],
  ['Del', 'Delete box · reject proposal'], ['⇧Del', 'Reject all visible proposals'],
  ['Enter', 'Accept proposal'], ['⇧Enter', 'Accept all visible proposals'],
  ['{ }', 'Proposal threshold −/+ 0.05'], ['Tab ⇧Tab', 'Cycle proposals, then boxes'],
  ['I', 'Propose with the current AI model'], ['⇧I', 'Choose AI model'],
  ['R', 'Refine selected box (SAM)'], ['C', 'Previous frame → proposals'], ['⇧C', 'Copy previous frame boxes'],
  ['K', 'Keyframe · interpolate between keyframes'], ['`', 'Previous-frame ghost'],
  ['Alt ←↑→↓', 'Move box or focused edge (⇧ ×10)'], ['E', 'Focus an edge of the selected box'],
  [`⌘C ⌘V ⌘D`, 'Copy · paste · duplicate boxes'],
  ['O', 'Hide boxes (hold to peek)'], ['⇧O', 'Box fill 0 / 8 / 20%'], ['L', 'Labels: all · selected · off'],
  ['T', 'Crosshair'], ['M', 'Loupe'], ['Z', 'Zoom to box / back'], ['⇧Z', '100%'], ['= -', 'Zoom in / out'], ['F', 'Fit'],
  ['← →', 'Previous / next image'], ['Home End', 'First / last image'], ['N', 'Next image without boxes'], ['⇧N', 'Next image with proposals'],
  [', .', 'Previous / next frame of this sequence'], ['\\', 'Sequence filmstrip'],
  ['⇧A ⇧F ⇧X', 'Approve · flag · exclude (again: keep it, go on)'], ['⇧U', 'Clear review'], ['⌘Z ⌘⇧Z', 'Undo · redo'], ['⌘S', 'Save now'],
  ['Space drag', 'Pan'], ['Wheel', 'Zoom (trackpad: scroll pans, pinch zooms)'], ['Alt drag', 'Always move, never resize'],
];

const NOTICE_MS = 400;          // waits shorter than this show nothing at all
const DRAG_PX = 4;              // a drag is ≥ 4 screen px (Euclidean); less is a click
const MIN_PX = 2;               // a box needs ≥ 2 image px per side
const HANDLE_HIT = 8, EDGE_BAND = 5, SMALL_BOX = 24;
const COALESCE_MS = 800;        // nudges / repeated digits on one box within this window are one undo step
const HIST_CAP = 200, HIST_BOX_BUDGET = 400000;
const DUP_IOU = 0.7, PASTE_IOU = 0.9;
const PEEK_MS = 250;
const ROW_H = 26;
const AUDIT = new Set(['missing', 'loose', 'swap', 'spurious']);
// The audit glyphs and words Explore, Objects, Overview and the AI tab use for the same finding (✕ "no support").
const GLYPH = { missing: '+', loose: '⤢', swap: '⇄', spurious: '✕', prev_frame: '↺', interp: '≈', refine: '◎', copy: '⧉' };
const KIND_LABEL = { detect: 'AI', prev_frame: 'previous frame', interp: 'interpolated', refine: 'refined', missing: 'possibly missing', loose: 'loose box', swap: 'class swap', spurious: 'box without support', copy: 'from another image' };
const LABEL_MAX = 172;          // px of text in a canvas label chip (chip ≤ 180 px); longer text is ellipsized in the middle
// QA flags whose chip shows what differs (flagTargets); the dashed ghost of a pressed chip is amber, as in Inspect.
const FLAG_DETAIL = new Set(['near_dup', 'leak_split', 'dup_same_role', 'dup_label_mismatch', 'seq_gap', 'seq_jump', 'seq_flicker', 'seq_count_spike', 'shape_outlier']);
const FLAG_GHOST = '#f59e0b';

// Session-wide (outlive one Annotate view): undo stacks per image, class counts per dataset, the box clipboard.
const histByImage = new Map();
// Label files this session created on images that had none (`${ds}:${id}` -> the version we last wrote). When such
// an image is emptied again (undo, delete) the file is removed, not left as a 0-byte "background" file.
const createdFiles = new Map();
const countsCache = new Map();
let serverIssueLabels = {};
let clipboard = null;

const LS = {
  get(k, d = null) { try { const v = localStorage.getItem(k); return v == null ? d : v; } catch (e) { return d; } },
  set(k, v) { try { if (v == null) localStorage.removeItem(k); else localStorage.setItem(k, String(v)); } catch (e) { /* private mode */ } },
};

function iou(a, b) {   // [cls, xc, yc, w, h] normalised
  const ax0 = a[1] - a[3] / 2, ax1 = a[1] + a[3] / 2, ay0 = a[2] - a[4] / 2, ay1 = a[2] + a[4] / 2;
  const bx0 = b[1] - b[3] / 2, bx1 = b[1] + b[3] / 2, by0 = b[2] - b[4] / 2, by1 = b[2] + b[4] / 2;
  const iw = Math.min(ax1, bx1) - Math.max(ax0, bx0), ih = Math.min(ay1, by1) - Math.max(ay0, by0);
  if (iw <= 0 || ih <= 0) return 0;
  const inter = iw * ih, u = a[3] * a[4] + b[3] * b[4] - inter;
  return u > 0 ? inter / u : 0;
}
const copyBox = (b) => { const c = b.slice(); if (b.kind) c.kind = b.kind; if (b.issues) c.issues = b.issues; return c; };

/** A box from two opposite corners in image pixels (drag, or the two clicks of 2-click mode), clamped to the image:
    normalised [xc, yc, w, h], or null when a side is under `minPx` image pixels. */
export function boxFromCorners(a, c, W, H, minPx = MIN_PX) {
  const x0 = Math.max(0, Math.min(a.x, c.x)), y0 = Math.max(0, Math.min(a.y, c.y)), x1 = Math.min(W, Math.max(a.x, c.x)), y1 = Math.min(H, Math.max(a.y, c.y));
  if (!(x1 - x0 >= minPx && y1 - y0 >= minPx)) return null;
  return [(x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H];
}
/** Frames of a sequence in natural file-name order ("_9" before "_10"); a sequence split over train and test
    interleaves its frames here, where path order would list one split after the other. Ties: by path. */
const natural = new Intl.Collator(undefined, { numeric: true, sensitivity: 'base' });
const baseOf = (p) => String(p || '').split('/').pop();
export function filmOrder(rows) {
  return rows.slice().sort((a, b) => natural.compare(baseOf(a.rel_path), baseOf(b.rel_path)) || (a.rel_path < b.rel_path ? -1 : a.rel_path > b.rel_path ? 1 : 0));
}
const sameFrame = (a, b) => natural.compare(baseOf(a.rel_path), baseOf(b.rel_path)) === 0;
/** The copies of one frame name on the strip from index j on, walking in direction d (filmOrder keeps them
    adjacent), those in `split` first: a sequence held twice — the same video in test/ and in another folder —
    has two copies ("twins") of every frame. */
export function frameGroup(frames, j, d, split = null) {
  const out = [];
  for (let m = j; m >= 0 && m < frames.length && sameFrame(frames[m], frames[j]); m += d) out.push(m);
  const pref = split || null;
  return out.sort((a, b) => ((frames[a].split || null) === pref ? 0 : 1) - ((frames[b].split || null) === pref ? 0 : 1) || (a - b) * d);
}
/** Index of the frame before (d = -1) or after (d = 1) frames[k] that is another frame: the copies of frames[k]
    itself in other folders are skipped, and of the next frame's copies the one in `split` wins. -1 at the end. */
export function seqNeighbor(frames, k, d, split = null) {
  let j = k + d;
  while (j >= 0 && j < frames.length && sameFrame(frames[j], frames[k])) j += d;
  return j >= 0 && j < frames.length ? frameGroup(frames, j, d, split)[0] : -1;
}
/** Frame numbers on the strip: the copies of one frame in other folders share one (filmOrder keeps them adjacent).
    `of[k]` is frame k's 0-based frame number, `n` the number of distinct frames, `multi` how many have copies. */
export function frameOrdinals(frames) {
  const of = new Int32Array(frames.length);
  let n = 0, multi = 0;
  for (let k = 0; k < frames.length; k++) {
    if (k > 0 && !sameFrame(frames[k], frames[k - 1])) n++;
    else if (k > 0 && of[k - 1] === n && (k < 2 || of[k - 2] !== n)) multi++;     // the first copy of this frame
    of[k] = n;
  }
  return { of, n: frames.length ? n + 1 : 0, multi };
}
/** What a QA flag ({code, detail}) points at on its own image: label-file rows (0-based) it names, boxes where the
    frames around it put an object ([cls, xc, yc, w, h]: a seq_gap's missing box, a seq_jump's expected place), and
    `other`, the image whose boxes to compare (a twin, or the frame before) — null when the flag names none. */
export function flagTargets(flag) {
  const d = (flag && flag.detail) || {}, code = flag && flag.code;
  const rows = code === 'seq_jump' ? (d.jumps || []).map(j => j && j.row)
    : code === 'seq_flicker' || code === 'shape_outlier' ? (d.rows || []).map(x => x && x.row) : [];
  const exp = code === 'seq_jump' ? (d.jumps || []).map(j => j && j.expected) : code === 'seq_gap' ? d.boxes || [] : [];
  const other = code === 'dup_label_mismatch' || code === 'leak_split' || code === 'dup_same_role' ? d.twin : code === 'seq_flicker' || code === 'seq_count_spike' ? d.prev_id : null;
  return {
    rows: [...new Set(rows.filter(r => Number.isInteger(r) && r >= 0))],
    boxes: exp.filter(b => Array.isArray(b) && b.length >= 5 && b.slice(0, 5).every(v => v !== null && isFinite(v))).map(b => b.slice(0, 5).map(Number)),
    other: Number.isInteger(other) && other >= 0 ? other : null,
  };
}
/** A box's size for the Boxes list: image pixels when the image size is known, else its share of the image. */
export function boxSizeText(b, W, H) {
  if (W > 0 && H > 0) return `${Math.round(b[3] * W)}×${Math.round(b[4] * H)}`;
  const pc = (v) => (v * 100).toFixed(v < 0.01 ? 1 : 0);
  return `${pc(b[3])}×${pc(b[4])}%`;
}
const sameGeom = (a, b) => a.length === b.length && a.every((v, k) => Math.abs(v - b[k]) < 1e-9);
/** The order a link means when it names none: the server's default for that sort (imagesq.DEFAULT_DESC — most
    suspect first for the audit and flag sorts), so "worst first" opens on the worst image, as in Explore. */
export const defaultOrder = (sort) => (sort === 'audit' || sort === 'flag' ? 'desc' : 'asc');
// QA flag codes (core/labels.py FLAG_LABELS) and the label audit's finding codes: the issue= codes with a score.
// Label-file problems (bad_format, …) have none. The same sets as Explore's, so a link opens in the same order there.
const FLAG_CODES = new Set(['near_dup', 'leak_split', 'dup_same_role', 'dup_label_mismatch', 'seq_gap', 'seq_flicker', 'seq_jump', 'seq_count_spike', 'shape_outlier', 'blurry', 'dark', 'overexposed']);
const AUDIT_CODES = new Set(['ai_missing', 'ai_loose', 'ai_swap', 'ai_spurious']);
const issueCodes = (f) => String((f && f.issue) || '').split(',').map(c => c.trim()).filter(Boolean);
/** Whether the flag sort has something to rank by: the issue filter names a QA flag. */
export const flagSortable = (f) => issueCodes(f).some(c => FLAG_CODES.has(c));
/** The sort a list opens on when its link names none. A saved selection keeps its own order. A list of QA flags or AI
    audit findings opens most suspicious first: by the score of the flags it names (imagesq "flag", which also counts
    the audit score of audit kinds named with them), or by the audit score when it names only audit findings or
    `ai=audit`. Label-file issues have no score: such a list, like any other, stays in index order. */
export function initialSort(f) {
  f = f || {};
  if (f.set) return 'set';
  if (flagSortable(f)) return 'flag';
  if (issueCodes(f).some(c => AUDIT_CODES.has(c)) || f.ai === 'audit') return 'audit';
  return 'id';
}
/** The POST /proposals/decide body for queued decisions (proposal id -> 'accepted' | 'rejected' | 'pending'). */
export function decideBody(queue) {
  const body = { accept: [], reject: [] }, pend = [];
  for (const [id, st] of queue) { if (st === 'accepted') body.accept.push(id); else if (st === 'rejected') body.reject.push(id); else pend.push(id); }
  if (pend.length) body.pending = pend;
  return body;
}
/** Send every decision in `queue` with `post(body)`. A send that fails puts its decisions back — never over a newer
    decision made for the same proposal meanwhile — so a retry sends them again; only an answer that a retry cannot
    change (a 4xx other than 408 / 429) drops them, as does a server without a proposals store (404 / 405). Resolves
    `{ ok, sent, error, retry }`: `ok` when nothing is left to send for what was taken. */
export async function sendDecisions(queue, post) {
  if (!queue.size) return { ok: true, sent: 0, error: null, retry: false };
  const q = new Map(queue); queue.clear();
  try { await post(decideBody(q)); return { ok: true, sent: q.size, error: null, retry: false }; }
  catch (e) {
    const s = e && e.status;
    if (s === 404 || s === 405) { queue.clear(); return { ok: true, sent: 0, error: null, retry: false }; }
    const retry = !(s >= 400 && s < 500 && s !== 408 && s !== 429);
    if (retry) for (const [id, st] of q) if (!queue.has(id)) queue.set(id, st);
    return { ok: false, sent: 0, error: e, retry };
  }
}
/** Proposals as the server returned them, with the decisions still queued here applied: an image revisited before
    its decisions reached the server shows them as made, not as pending again. */
export function withQueuedDecisions(preds, queue) {
  if (queue && queue.size) for (const p of preds) if (p && p.id != null && queue.has(p.id)) p.state = queue.get(p.id);
  return preds;
}
/** The proposal threshold Explore shares (localStorage): a stored 0 is a choice, not a missing value. */
export function readThr(stored) { const v = Number(stored); return stored != null && stored !== '' && isFinite(v) && v >= 0 && v <= 1 ? v : 0.25; }
/** Alt+wheel over the Point tool, one notch = ±0.01. Shift turns the wheel into a horizontal scroll in most
    browsers, so the direction comes from whichever axis moved. */
export function wheelStep(dx, dy) { const v = dy || dx; return v > 0 ? -0.01 : v < 0 ? 0.01 : 0; }
/** Class hotkeys. The first digit applies at once; when it can start a two-digit id it stays pending, and a second
    digit replaces it — only while the same item is selected (`key`): a Tab, a click or a draw in between starts
    over, so "1, Tab, 0" gives the next box class 0, never 10. Returns the class to apply (null: no such class)
    and whether the digit may start a pair. */
export function classDigit(pending, d, key, n) {
  if (pending && pending.key === key) { const two = Number(pending.d + d); if (two < n) return { cls: two, wait: false }; }
  const one = Number(d);
  if (n && one >= n && one !== 0) return { cls: null, wait: false };
  return { cls: one, wait: one > 0 && one * 10 < n };
}
/** Pending proposals the editor would show over `boxes` (threshold, hidden classes, same-class duplicates of a box
    hidden as "already labeled") — ⇧N skips images where this is zero. Audit findings always count. */
export function shownPending(preds, boxes, thr, hidden = new Set()) {
  let n = 0;
  for (const p of preds || []) {
    if ((p.state || 'pending') !== 'pending') continue;
    const b = [Number(p.cls) || 0, +p.xc, +p.yc, +p.w, +p.h];
    if (hidden.has(b[0]) || (p.conf != null && +p.conf < thr)) continue;
    if (!AUDIT.has(p.kind) && p.kind !== 'refine' && (boxes || []).some(x => x[0] === b[0] && iou(x, b) >= DUP_IOU)) continue;
    n++;
  }
  return n;
}

/** The position box: "7133", "7,133" (as it is shown) or "7,133 / 7,808" → 7133; null when it is not a position. */
export function parsePosition(v) {
  const m = /^(\d{1,3}(?:,\d{3})+|\d+)(\s*\/.*)?$/.exec(String(v).trim());
  return m ? parseInt(m[1].replace(/,/g, ''), 10) : null;
}
/** An audit finding's `ref_box` (the GT box [xc, yc, w, h] when the audit ran; a leading class id tolerated). */
export function refGeom(v) {
  if (!Array.isArray(v) || v.length < 4) return null;
  const g = v.slice(-4).map(Number);
  return g.every(isFinite) ? g : null;
}
/** The box an audit finding is about, found by the GT geometry it stored (rows shift when a box above is deleted):
    IoU >= 0.9 with it — the rule the server's bulk accept applies (runner._ref_ok) — of the same class except for
    a class swap, the stored row breaking ties. null: no such box any more (the finding is stale). undefined: the
    finding stored no geometry, so only its row can tell. */
export function refByGeom(boxes, p) {
  if (!p.ref_box) return undefined;
  const g = [0, ...p.ref_box], ok = (b) => p.kind === 'swap' || b[0] === p.b[0];
  const at = p.ref_row != null ? boxes[p.ref_row] : null;
  if (at && ok(at) && iou(at, g) >= 0.9) return at;
  let best = null, bv = 0.9;
  for (const b of boxes) { if (!ok(b)) continue; const v = iou(b, g); if (v >= bv) { bv = v; best = b; } }
  return best;
}
/** Kinds whose accept adds a box (the others change or delete the box they point at, `p.refBox`). */
const addsBox = (p) => !((AUDIT.has(p.kind) && p.kind !== 'missing') || (p.kind === 'refine' && p.refBox));
/** What one ⇧Enter does with the visible proposals `list` (most confident first) over `boxes`: `take` is accepted,
    `dupes` rejected, `left` stays pending. A proposal that adds a box is not taken when it lands (IoU ≥ 0.7) on a box —
    one already here, one this accept adds, or a box where a fix in this same accept moves or re-classes it (a ⤢
    tighter box and the detection it came from are one object; the fixes count first, whatever their order). Of its
    own class it is a duplicate, of another class it is left for you (⇄). So ⇧Enter never puts two boxes on one object. */
export function bulkAcceptPlan(list, boxes) {
  const fixed = new Map();          // a box a fix points at -> where and what it will be
  for (const p of list) {
    if (addsBox(p) || !p.refBox || !boxes.includes(p.refBox)) continue;
    const f = fixed.get(p.refBox) || p.refBox.slice(0, 5);
    if (p.kind === 'swap') f[0] = p.b[0];
    else if ((p.kind === 'loose' || p.kind === 'refine') && !p.refBox.kind) { f[1] = p.b[1]; f[2] = p.b[2]; f[3] = p.b[3]; f[4] = p.b[4]; }
    else continue;                  // ✕ no support: the box stays in the way (its removal makes no room for a class)
    fixed.set(p.refBox, f);
  }
  // Where a fixed box was stays occupied too: of another class, a detection there is still yours to decide.
  const placed = boxes.map(b => b.slice(0, 5)).concat([...fixed.values()]), take = [], dupes = [], left = [];
  for (const p of list) {
    if (addsBox(p)) {
      const near = placed.filter(b => iou(b, p.b) >= DUP_IOU);
      if (near.length) { (near.some(b => b[0] === p.b[0]) ? dupes : left).push(p); continue; }
      placed.push(p.b.slice(0, 5));
    }
    take.push(p);
  }
  return { take, dupes, left };
}
/** Another box of `ref`'s class on the same object (IoU ≥ 0.7), after a fix moved or re-classed `ref`; null if none.
    Segmentation / pose rows (`.kind`) are never matched: a fix does not move them. */
export function dupOf(boxes, ref) {
  if (!ref || ref.kind) return null;
  return boxes.find(x => x !== ref && !x.kind && x[0] === ref[0] && iou(x, ref) >= DUP_IOU) || null;
}
/** `text` cut to at most `max` px (as `measure` reports) with "…" in the middle: the head — class id, glyph,
    confidence — and the tail, where long class names tend to differ, both stay. The text itself when it fits. */
export function middleEllipsis(text, max, measure) {
  text = String(text);
  if (measure(text) <= max) return text;
  const ch = Array.from(text);
  const cut = (n) => { const a = Math.ceil(n * 0.6); return ch.slice(0, a).join('') + '…' + ch.slice(ch.length - (n - a)).join(''); };
  let lo = 0, hi = ch.length - 1, best = '…';
  while (lo <= hi) { const mid = (lo + hi) >> 1, s = cut(mid); if (measure(s) <= max) { best = s; lo = mid + 1; } else hi = mid - 1; }
  return best;
}
/** What an interpolation (POST /interpolate, core/qa.py) did, for the Sequence panel and the note: `head` (proposals
    made, or why there are none — the server's `message`, e.g. adjacent keyframes) and `notes` on what was left out —
    keyframe boxes with no partner in the other keyframe (`warn`), and boxes the frames in between
    already have — and on boxes paired across a class change (a person who falls: `class_changes`, worded by the
    server's `message`), whose proposals take the nearer keyframe's class and are worth a look frame by frame. */
export function interpSummary(res, fromName, toName) {
  res = res || {};
  const cnt = (v) => (Array.isArray(v) ? v.length : Number(v) || 0);
  const pl = (n, one, many = one + 's') => `${n.toLocaleString('en-US')} ${n === 1 ? one : many}`;
  const n = cnt(res.preds), imgs = Array.isArray(res.preds) && !res.images ? new Set(res.preds.map(p => p.image_id)).size : cnt(res.images);
  const um = res.unmatched || {}, uf = Math.max(0, Number(um.from) || 0), ut = Math.max(0, Number(um.to) || 0);
  const sk = Math.max(0, Number(res.skipped_existing) || 0);
  const notes = [];
  if (uf || ut) {
    const where = uf && ut ? `${pl(uf, 'box', 'boxes')} in ${fromName} and ${ut.toLocaleString('en-US')} in ${toName}` : uf ? `${pl(uf, 'box', 'boxes')} in ${fromName}` : `${pl(ut, 'box', 'boxes')} in ${toName}`;
    notes.push({ text: `${where} had no partner in the other keyframe — not interpolated`, warn: true });
  }
  if (sk) notes.push({ text: `${pl(sk, 'box', 'boxes')} already on the frames in between — skipped`, warn: false });
  const msg = typeof res.message === 'string' && res.message.trim() ? res.message.trim() : '';
  const cc = Array.isArray(res.class_changes) ? res.class_changes.length : Math.max(0, Number(res.class_changes) || 0);
  // The server words a class change itself (which classes, and what the proposals do); any other message with
  // proposals made is said as it is. Without proposals and class changes, the message is the reason there are none.
  if (cc) notes.push({ text: msg || `${pl(cc, 'box', 'boxes')} change${cc === 1 ? 's' : ''} class between the keyframes: the proposals take the class of the nearer keyframe`, warn: false });
  else if (msg && n) notes.push({ text: msg, warn: false });
  const reason = cc ? '' : msg;
  const head = n ? `Interpolated ${pl(n, 'proposal')} on ${pl(imgs, 'frame')} between ${fromName} and ${toName}`
    : reason ? `Nothing to interpolate between ${fromName} and ${toName} — ${/^[A-Z][a-z]/.test(reason) ? reason[0].toLowerCase() + reason.slice(1) : reason}`
    : sk ? `Nothing new to interpolate between ${fromName} and ${toName}`
    : `No boxes to interpolate between ${fromName} and ${toName}`;
  // The transient note says the same in a few words; the Sequence panel keeps the sentences.
  const brief = [uf + ut && `${pl(uf + ut, 'box', 'boxes')} without a partner`, sk && `${sk.toLocaleString('en-US')} already labeled`,
    cc && pl(cc, 'class change')].filter(Boolean).join(' · ');
  return { n, imgs, head, notes, brief, warn: notes.some(x => x.warn) };
}

async function render(el, { ctx, dataset, fovea, refresh }) {
  const { h, icon, ui, fmt, cls, isTyping, MOD } = fovea;
  const K = fovea.keys;
  const { classColor, withAlpha, className, REVIEW } = fovea.colors;
  const textOn = fovea.colors.textOn || (() => '#fff');
  const ds = dataset;
  let names = [...(ds.classes || [])];
  const q0 = ctx.query;
  // Every list filter Explore can hand over (api/common.py FILTER_KEYS), so "Annotate this list" opens the same list.
  // `set` is a server-side working set (POST /sets → token): an ordered id list, like `ids` without its size limit;
  // it keeps its own order unless the link asks for another sort.
  const FKEYS = ['split', 'cls', 'only_cls', 'labeled', 'review', 'exclude_review', 'issue', 'q', 'seq', 'ids', 'set', 'ai', 'min_boxes', 'max_boxes', 'has_label', 'seed', 'attr'];
  let filters = {}; for (const k of FKEYS) if (q0.get(k)) filters[k] = q0.get(k);
  // A link without a sort opens in the order its filters imply (initialSort: a QA-flag or audit list most suspicious
  // first). Such an implied sort follows the filters — opening a sequence shows it in index order, and going back
  // to the list restores the flag order — while a sort the link names stays.
  let sort = q0.get('sort') || initialSort(filters);
  let sortImplied = sort === initialSort(filters);
  let order = q0.get('order') === 'asc' || q0.get('order') === 'desc' ? q0.get('order') : defaultOrder(sort);
  let cursor = new fovea.ImageCursor({ dsId: ds.id, filters, sort, order, pageSize: 200, boxes: false });

  // ---------------------------------------------------------------- state
  // sizeKnown: W × H is this image's size (decoded, or from the index when the file is missing), not a leftover.
  let idx = -1, item = null, img = null, imgFull = true, W = 1, H = 1, sizeKnown = false;
  let boxes = [], props = [], runs = {}, sel = null, hov = null, tool = 'select', activeCls = 0;
  // Annotations converted from another format: regions that are not objects (drawn hatched, never selectable) and the
  // attribute names of the dataset's format card (each label row carries its own attributes as row.attrs).
  let regions = [], annSch = NO_ANN;
  annSchema(fovea.api, ds).then((sch) => { if (sch === NO_ANN) return; annSch = sch; try { renderList(); renderInfo(); updateHud(); } catch (e) { /* view closed */ } });
  let scale = 1, tx = 0, ty = 0, fitScale = 1, customView = false, holdFit = false, prevDims = null, zoomBack = null;
  let corner = null;        // 2-click box mode: the first corner, in image pixels (survives zoom and pan)
  let undo = [], redo = [];
  let dirty = false, saveTimer = null, saving = false, destroyed = false;
  // loading: a navigation is in flight. Panels and canvas still show the image on screen; every mutation is blocked
  // so an edit can never be queued against an image that is about to be replaced.
  let loading = false, navToken = 0, want = -1, lastNavAt = 0, booted = false;
  let version = null, baseBoxes = [], savedCounts = {}, labelsError = null, missingFile = false;
  let saveRetries = 0, lastFailToast = 0, conflictOpen = false, conflictPending = null, lastOutcome = 'ok';
  let mouse = { x: 0, y: 0, inside: false }, drag = null, spaceDown = false, focusEdge = null, theirsOverlay = null;
  // approveArmed: the image a first ⇧A over pending proposals was pressed on, and when (a second ⇧A within 3 s on
  // that same image approves anyway). classBuf: a first class digit that may still become 10+, and the item it went to.
  let hiddenAll = false, peek = null, hiddenCls = new Set(), noteTimer = null, approveArmed = null;
  let classBuf = '', classBufFor = null, classBufTimer = null, autoTimer = null, autoOn = false, uidSeq = 1;
  let keyframe = null, seqStatus = '', seqAction = null, ghost = null, rangeStart = null, propsApi = true;
  let flagFocus = null;     // the pressed QA flag chip: what it shows over this image (see focusFlag)
  let touched = false;      // the image on screen was edited or had proposals decided during this visit
  const decisionQueue = new Map();
  const reviewStats = { accepted: 0, rejected: 0, edited: 0, warned: false };   // automation-bias check, this session
  // decSending: the decide request on its way; decFail: the save-state text while queued decisions keep failing.
  let decTimer = null, decSending = null, decFail = null, decFails = 0;
  const opts = {
    propagate: LS.get('fovea.ann.propagate') === '1',
    labels: LS.get('fovea.ann.labelmode') || (LS.get('fovea.ann.labels') === '0' ? 'off' : 'all'),
    fill: LS.get('fovea.ann.fill'),                        // null = auto: 8%, none above 40 boxes
    crosshair: LS.get('fovea.ann.crosshair') !== '0',
    loupe: LS.get('fovea.ann.loupe') === '1',
    ghost: LS.get('fovea.ann.ghost') === '1',
    advance: LS.get('fovea.review.advance') !== '0',
    deselect: LS.get('fovea.ann.deselect') === '1',
    wheel: LS.get('fovea.ann.wheel') || 'auto',
    boxMode: LS.get('fovea.ann.boxmode') === 'click2' ? 'click2' : 'drag',
    film: LS.get('fovea.ann.film') == null ? null : LS.get('fovea.ann.film') === '1',   // null: on when the dataset has sequences
    pointW: 0.15, pointH: 0.30,
  };
  let adjust = (() => { try { return { b: 100, c: 100, ...JSON.parse(LS.get('fovea.ann.adjust') || '{}') }; } catch (e) { return { b: 100, c: 100 }; } })();
  let thr = readThr(LS.get(`fovea.ann.thr.${ds.id}`));
  const frames = createFrameCache({ max: 6 });

  // ---------------------------------------------------------------- DOM
  const root = h('div', { class: 'ann' });
  const tools = h('div', { class: 'ann-tools' });
  const stage = h('div', { class: 'ann-stage' });
  const side = h('div', { class: 'ann-side' });
  const bottom = h('div', { class: 'ann-bottom' });
  const cImg = h('canvas', { class: 'ann-img' });
  const cOv = h('canvas', { class: 'ann-ov', tabindex: '-1', 'aria-label': 'Image and boxes' });
  const hud = h('div', { class: 'hud' });
  const noteEl = h('div', { class: 'ann-note hidden', role: 'status' });
  const loupe = h('canvas', { class: 'ann-loupe hidden', width: 160, height: 160 });
  const minimap = h('canvas', { class: 'ann-minimap hidden' });
  const stageMsg = h('div', { class: 'stage-msg hidden' });
  [cImg, cOv, hud, noteEl, loupe, minimap, stageMsg].forEach(n => stage.appendChild(n));
  root.appendChild(tools); root.appendChild(stage); root.appendChild(side); root.appendChild(bottom);
  el.appendChild(root);
  const gi = cImg.getContext('2d'), go = cOv.getContext('2d'), gl = loupe.getContext('2d'), gm = minimap.getContext('2d');
  const cache = document.createElement('canvas'), gc = cache.getContext('2d');
  const FONT = '600 11px ' + (getComputedStyle(document.body).fontFamily || 'sans-serif');
  const MONO = '10.5px ' + (getComputedStyle(document.documentElement).getPropertyValue('--mono') || 'monospace');

  // ---- left panel
  const toolBtns = {};
  const TOOLS = [['select', 'pointer', 'Select', 'V'], ['box', 'boxSelect', 'Box', 'B'], ['point', 'crosshair', 'Point', 'P'], ['sam', 'target', 'Click', 'W'], ['pan', 'hand', 'Pan', 'H']];
  const toolRow = h('div', { class: 'tool-row ann-tool-row' }, TOOLS.map(([id, ic, label, key]) =>
    toolBtns[id] = h('button', { class: cls('tool-btn', tool === id && 'active'), 'aria-pressed': String(tool === id), 'data-tip': id === 'sam' ? 'Click an object to get a box (SAM) (W)' : id === 'box' ? 'Box (B) — B again: drag ⇄ 2-click' : `${label} (${key})`, onClick: () => (id === 'box' && tool === 'box' ? toggleBoxMode() : setTool(id)) }, icon(ic, 16), h('span', label))));
  const classHead = h('span', { class: 'xs faint ann-class-target' }, '0–9 hotkeys');
  const classList = h('div', { class: 'class-list ann-class-list', role: 'listbox', 'aria-label': 'Classes' });
  const newClassInput = h('input', { class: 'input input-sm', placeholder: 'New class name…', 'aria-label': 'New class name', onKeydown: async (e) => { if (e.key === 'Enter' && !e.isComposing && newClassInput.value.trim()) { await addClass(newClassInput.value.trim()); newClassInput.value = ''; } } });
  const pointOpts = h('div', { class: cls('col gap-4', tool !== 'point' && 'hidden') });
  const optRow = (label, ctl, tip) => h('div', { class: 'ann-opt' }, h('span', { class: 'small', title: tip || null }, label), ctl);
  const labelSeg = ui.seg([{ value: 'all', label: 'All' }, { value: 'sel', label: 'Selected', tip: 'Selected and hovered box only' }, { value: 'off', label: 'Off' }], opts.labels, v => setLabelMode(v), { label: 'Labels (L)' });
  const fillSeg = ui.seg([{ value: 'auto', label: 'Auto', tip: '8%, none above 40 boxes' }, { value: '0', label: '0' }, { value: '0.08', label: '8%' }, { value: '0.2', label: '20%' }], opts.fill == null ? 'auto' : opts.fill, v => setFill(v === 'auto' ? null : v), { label: 'Box fill (⇧O)' });
  const wheelSeg = ui.seg([{ value: 'auto', label: 'Auto', tip: 'Mouse wheel zooms; trackpad scroll pans, pinch zooms' }, { value: 'zoom', label: 'Zoom' }, { value: 'pan', label: 'Pan' }], opts.wheel, v => { opts.wheel = v; LS.set('fovea.ann.wheel', v); }, { label: 'Scroll wheel' });
  // Every switch and segmented control is named by the text beside it (screen readers otherwise say just "switch, on").
  const sw = (key, lsKey, label, after) => ui.switchBtn(opts[key], v => { opts[key] = v; LS.set(lsKey, v ? '1' : '0'); after && after(v); }, { label });
  const crossSw = sw('crosshair', 'fovea.ann.crosshair', 'Crosshair (T)', () => { updateHud(); invalidate(); });
  const loupeSw = sw('loupe', 'fovea.ann.loupe', 'Loupe (M)', () => { updateHud(); invalidate(); });
  const advanceSw = sw('advance', 'fovea.review.advance', 'Advance after ⇧A / ⇧X');
  const deselectSw = sw('deselect', 'fovea.ann.deselect', 'Deselect after drawing');
  const aiBox = h('div', { class: 'ann-ai' });
  const hintLine = h('div', { class: 'xs faint ann-hint' });
  tools.appendChild(h('div', { class: 'sec' }, h('h4', 'Tools'), toolRow, pointOpts));
  tools.appendChild(h('div', { class: 'sec' }, h('h4', 'Classes', h('span', { class: 'spacer' }), classHead), classList, h('div', { class: 'mt-8' }, newClassInput)));
  tools.appendChild(h('div', { class: 'sec' }, h('h4', icon('sparkles', 13), 'AI'), aiBox));
  tools.appendChild(h('div', { class: 'sec ann-options' }, h('h4', 'Options'),
    optRow('Labels (L)', labelSeg), optRow('Box fill (⇧O)', fillSeg),
    optRow('Crosshair (T)', crossSw), optRow('Loupe (M)', loupeSw),
    optRow('Advance after ⇧A / ⇧X', advanceSw, 'After approving or excluding, go to the next image'),
    optRow('Deselect after drawing', deselectSw, 'Off: a new box stays selected, so a digit sets its class'),
    optRow('Scroll wheel', wheelSeg)));
  tools.appendChild(h('div', { class: 'sec' }, hintLine, h('button', { class: 'btn btn-ghost btn-sm mt-8', onClick: () => fovea.showShortcuts && fovea.showShortcuts() }, icon('keyboard', 12), 'All shortcuts (?)')));

  // ---- right panel
  const wsBody = h('div', { class: 'col gap-4' });
  const propSec = h('div', { class: 'sec ann-props' });
  const boxCount = h('span', { class: 'badge' }, '0');
  const vlist = h('div', { class: 'ann-vlist', role: 'listbox', tabindex: '0', 'aria-label': 'Boxes on this image' });
  const vinner = h('div', { class: 'ann-vlist-inner' });
  vlist.appendChild(vinner);
  const boxEmpty = h('div', { class: 'row small faint hidden ann-box-empty' });
  const reviewRow = h('div', { class: 'ann-review' });
  const reviewNote = h('div', { class: 'xs faint hidden' });
  const infoKv = h('div', { class: 'kv small ann-kv' });
  const issuesRow = h('div', { class: 'row wrap gap-4 ann-issues' });
  const flagInfo = h('div', { class: 'ann-flag-detail xs hidden', role: 'status' });
  const seqBody = h('div', { class: 'col gap-4' });
  const rangeBody = h('div', { class: 'col gap-8' });
  side.appendChild(h('div', { class: 'sec ann-ws' }, h('h4', 'Working set'), wsBody));
  side.appendChild(propSec);
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Boxes', boxCount, h('span', { class: 'spacer' }),
    h('button', { class: 'btn btn-ghost btn-sm btn-icon', 'aria-label': 'Delete all boxes on this image', 'data-tip': 'Delete all boxes', onClick: () => deleteAll() }, icon('trash', 12))), vlist, boxEmpty));
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Review', h('span', { class: 'spacer' }), h('span', { class: 'xs faint' }, '⇧A ⇧F ⇧X · ⇧U')), reviewRow, reviewNote));
  // Problems first: a pressed flag chip's detail opens under it, in view, and pushes only the facts below it down.
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Image'), issuesRow, flagInfo, infoKv));
  side.appendChild(h('div', { class: 'sec' }, h('h4', icon('film', 13), 'Sequence'), seqBody));
  side.appendChild(h('div', { class: 'sec' }, h('h4', 'Range edit'), rangeBody));

  // ---- bottom bar
  const idxInput = h('input', { class: 'input input-sm idx', value: '', 'aria-label': 'Position in list — type a number, #id or part of a file name', 'data-tip': 'Go to: position, #id or part of a file name' });
  const idxText = () => (item ? `${fmt.num(idx + 1)} / ${fmt.num(cursor.total)}` : cursor.total ? '' : '0 / 0');
  idxInput.addEventListener('focus', () => idxInput.select());
  idxInput.addEventListener('blur', () => { idxInput.value = idxText(); });
  // Enter and Escape end here: the box blurs before the keydown would reach the editor's own keys, where Enter
  // would also accept the selected proposal and Escape clear the selection.
  idxInput.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); idxInput.value = idxText(); idxInput.blur(); return; }
    if (e.key !== 'Enter' || e.isComposing) return;
    e.preventDefault(); e.stopPropagation();
    const v = idxInput.value.trim();
    idxInput.blur();
    jumpTo(v);
  });
  /** Position ("120"), image id ("#7809") or part of a file name ("frame_0042") within the working set. */
  async function jumpTo(v) {
    if (!v || v === idxText()) return;
    const pos = parsePosition(v);
    if (pos != null && pos >= 1) { goTo(Math.min(pos, cursor.total || pos) - 1); return; }
    const idm = /^#\s*(\d+)$/.exec(v) || /^id\s*:?\s*(\d+)$/i.exec(v);
    try {
      let id = idm ? Number(idm[1]) : null;
      if (id == null) {
        const r = await fovea.api.get(`/api/datasets/${ds.id}/images`, { ...filters, q: v, sort, order, limit: 1, offset: 0 });
        const hit = r.items && r.items[0];
        if (!hit) { note(`No image in this list matches “${v}”`, 'warn'); return; }
        id = hit.id;
      }
      const at = await cursor.positionOf(id);
      if (at >= 0) goTo(at); else note(idm ? `Image #${id} is not in this working set` : `No image in this list matches “${v}”`, 'warn');
    } catch (e) { note(e.message, 'error'); }
  }
  const nameEl = h('span', { class: 'name grow' });
  const zoomEl = h('span', { class: 'xs mono faint ann-zoom' }, '100%');
  const playBtn = h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Auto-advance', 'aria-label': 'Auto-advance', onClick: () => toggleAuto() }, icon('play', 13));
  const speed = h('input', { type: 'range', class: 'slider', min: 100, max: 3000, step: 50, value: 600, style: 'width:70px', 'aria-label': 'Auto-advance interval', 'data-tip': 'Auto-advance interval', onInput: () => { if (autoOn) scheduleAuto(); } });
  const saveState = h('span', { class: 'save-state saved', role: 'status' }, icon('check', 12), 'Saved');
  const adjustBtn = h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Brightness / contrast', 'aria-label': 'Brightness and contrast', onClick: () => openAdjust() }, icon('sliders', 13));
  const panelBtn = h('button', { class: 'btn btn-sm btn-icon ann-panel-btn', 'data-tip': 'Side panel', 'aria-label': 'Toggle side panel', onClick: () => root.classList.toggle('side-open') }, icon('sidebar', 13));
  bottom.appendChild(h('div', { class: 'btn-group' },
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'First (Home)', 'aria-label': 'First image', onClick: () => goTo(0) }, icon('chevronsLeft', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Previous (←)', 'aria-label': 'Previous image', onClick: () => step(-1) }, icon('chevronLeft', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Next (→)', 'aria-label': 'Next image', onClick: () => step(1) }, icon('chevronRight', 14)),
    h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Last (End)', 'aria-label': 'Last image', onClick: () => goTo((cursor.total || 1) - 1) }, icon('chevronsRight', 14))));
  bottom.appendChild(idxInput);
  bottom.appendChild(h('button', { class: 'btn btn-sm', 'data-tip': 'Next image without boxes (N)', onClick: () => nextUnlabeled() }, icon('zap', 13), h('span', { class: 'ann-hide-narrow' }, 'Next unlabeled')));
  bottom.appendChild(playBtn); bottom.appendChild(speed);
  bottom.appendChild(nameEl);
  bottom.appendChild(saveState);
  bottom.appendChild(adjustBtn);
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Fit (F)', 'aria-label': 'Fit image', onClick: () => { fit(); invalidate(true); } }, icon('maximize', 13)));
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Zoom out (-)', 'aria-label': 'Zoom out', onClick: () => zoomBy(1 / 1.25) }, icon('zoomOut', 13)));
  bottom.appendChild(h('button', { class: 'btn btn-sm btn-icon', 'data-tip': 'Zoom in (=)', 'aria-label': 'Zoom in', onClick: () => zoomBy(1.25) }, icon('zoomIn', 13)));
  bottom.appendChild(zoomEl);
  bottom.appendChild(panelBtn);

  // ---------------------------------------------------------------- host API for the AI panel (SPEC §3.6)
  const imageListeners = new Set();
  const host = {
    fovea, dataset: ds,
    getItem: () => item,
    getIndex: () => idx,
    getFilters: () => ({ ...filters }),
    getSort: () => ({ sort, order }),
    async idsFrom(index, n) {
      const out = [];
      for (let i = Math.max(0, index); out.length < n; i++) { const it = await cursor.ensure(i); if (!it) break; out.push(it.id); }
      return out;
    },
    isEditable: () => editable(),
    // The panel words a failure as unsaved edits; when only decisions are held back, the note says so.
    flush: async () => { const r = await flush(); if (r === 'failed' && !dirty && !destroyed) note(notSavedMsg(), 'warn'); return r; },
    unsavedReason: () => notSavedMsg(),
    reload: () => goTo(idx, { reload: true }),
    setProposals(preds) {
      if (!item || !Array.isArray(preds)) return;
      // A response for an image the user already left must never land on the one now on screen.
      if (preds.some(p => p && p.image_id != null && p.image_id !== item.id)) return;
      const incoming = preds.map(normProp);
      const kinds = new Set(incoming.map(p => p.kind));
      if (!kinds.size) kinds.add('detect');
      // ai-ui hook (proposal ids reused by the server): a new proposal can get the id of one the run just replaced; a
      // decision still queued for that id, or a decided copy kept here, belongs to the old one, never to the new one.
      const fresh = new Set(incoming.map(p => p.id).filter(id => id != null));
      fresh.forEach(id => decisionQueue.delete(id));
      props = props.filter(p => (p.local || p.state !== 'pending' || !kinds.has(p.kind)) && (p.local || p.id == null || !fresh.has(p.id))).concat(incoming);
      if (sel && sel.t === 'prop') sel = null;
      propsChanged();
    },
    getProposalThreshold: () => thr,
    setProposalThreshold: (v) => setThr(v),
    onImageChange(cb) { imageListeners.add(cb); return () => imageListeners.delete(cb); },
    classNames: () => names.slice(),
  };
  let panel = null;
  import('./ai-panel.js').then((m) => {
    if (destroyed) return;
    const r = m.mountAIPanel(aiBox, host);
    panel = typeof r === 'function' ? { destroy: r } : (r || {});
  }).catch((e) => { console.error(e); aiBox.replaceChildren(h('div', { class: 'small faint' }, 'AI panel could not be loaded.')); });

  // ---------------------------------------------------------------- class palette & picker
  let classCounts = {}, issueLabels = { ...BASE_ISSUE_LABELS, ...serverIssueLabels };
  function loadCounts() {
    const c = countsCache.get(ds.id);
    if (c && Date.now() - c.at < 5 * 60e3) { classCounts = c.counts; renderClasses(); return; }
    // Boxes per class only (one index-only GROUP BY), still deferred so it never competes with the first image.
    const run = () => fovea.api.get(`/api/datasets/${ds.id}/classes/counts`).then((r) => {
      const counts = {}; for (const [k, v] of Object.entries(r.counts || {})) counts[k] = Number(v) || 0;
      countsCache.set(ds.id, { counts, at: Date.now() });
      if (r.issue_labels) mergeIssueLabels(r.issue_labels);
      if (destroyed) return;
      classCounts = counts; renderClasses();
    }).catch(() => {});
    (window.requestIdleCallback || ((f) => setTimeout(f, 300)))(run, { timeout: 1200 });
  }
  // Issue / flag wording belongs to the server. The shared table covers every known code; a code it does not know
  // (a newer server) fetches the server's labels once, from /stats, only when such a code is actually on screen.
  let labelsAsked = false;
  function mergeIssueLabels(l) { serverIssueLabels = { ...serverIssueLabels, ...l }; issueLabels = { ...issueLabels, ...l }; }
  function issueLabel(c) {
    if (issueLabels[c]) return issueLabels[c];
    if (!labelsAsked) {
      labelsAsked = true;
      fovea.api.get(`/api/datasets/${ds.id}/stats`).then((r) => { if (r.issue_labels) mergeIssueLabels(r.issue_labels); if (!destroyed) { renderWorkingSet(); renderInfo(); } }).catch(() => {});
    }
    return c.replace(/_/g, ' ');
  }
  function bumpCounts(before, after) {
    const keys = new Set([...Object.keys(before), ...Object.keys(after)]);
    let changedAny = false;
    for (const k of keys) { const d = (after[k] || 0) - (before[k] || 0); if (d) { classCounts[k] = Math.max(0, (classCounts[k] || 0) + d); changedAny = true; } }
    if (changedAny) { const c = countsCache.get(ds.id); if (c) c.counts = classCounts; classRows.forEach((r, i) => { const n = r.querySelector('.n'); if (n) n.textContent = classCounts[i] != null ? fmt.compact(classCounts[i]) : ''; }); }
  }
  const countByClass = (list) => { const o = {}; for (const b of list) o[b[0]] = (o[b[0]] || 0) + 1; return o; };
  const classRows = [];
  function renderClasses() {
    classList.replaceChildren(); classRows.length = 0;
    if (!names.length) classList.appendChild(h('div', { class: 'small faint' }, 'No classes yet — add one below, or just draw (class 0).'));
    const hl = hlCls();
    names.forEach((n, i) => {
      const hidden = hiddenCls.has(i);
      const row = h('div', { class: cls('class-row', i === hl && 'active', hidden && 'is-hidden') },
        h('button', { class: 'class-pick', title: `${i} · ${n}`, 'aria-pressed': String(i === hl), onClick: () => pickClass(i) },
          h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'name' }, n),
          h('span', { class: 'n', 'data-tip': 'boxes in dataset' }, classCounts[i] != null ? fmt.compact(classCounts[i]) : ''),
          h('span', { class: cls('kbd', i >= 10 && 'kbd-dim') }, String(i))),
        h('button', { class: 'btn btn-ghost btn-sm btn-icon class-eye', 'aria-label': `${hidden ? 'Show' : 'Hide'} class ${n}`, 'aria-pressed': String(hidden), 'data-tip': 'Hide class · Alt-click: show only this class', onClick: (e) => toggleClassVisible(i, e.altKey) }, icon(hidden ? 'eyeOff' : 'eye', 12)));
      classRows.push(row); classList.appendChild(row);
    });
    renderRange();
  }
  /** Only the highlighted class changed: update in place (a rebuild would drop keyboard focus in the list). */
  // Runs on every selection change: only the class list itself scrolls to the row (scrollIntoView would also move
  // the whole left panel when it is scrolled down to the AI section), and only when the highlight moved.
  let shownHl = -1;
  function markActiveClass() {
    const hl = hlCls();
    if (hl === shownHl && classRows[hl] && classRows[hl].classList.contains('active')) return;
    shownHl = hl;
    classRows.forEach((r, i) => { r.classList.toggle('active', i === hl); const b = r.firstChild; if (b) b.setAttribute('aria-pressed', String(i === hl)); });
    const a = classRows[hl];
    if (a && classList.scrollHeight > classList.clientHeight) {
      const lr = classList.getBoundingClientRect(), ar = a.getBoundingClientRect();
      if (ar.top < lr.top) classList.scrollTop -= lr.top - ar.top; else if (ar.bottom > lr.bottom) classList.scrollTop += ar.bottom - lr.bottom;
    }
  }
  function updateClassHead() {
    const s = selectedItem();
    classHead.textContent = s ? (s.t === 'prop' ? 'applies to the proposal' : `applies to box #${s.i + 1}`) : '0–9 hotkeys';
    classHead.classList.toggle('accent-text', !!s);
  }
  function toggleClassVisible(i, solo) {
    if (solo) {
      const onlyThis = hiddenCls.size === names.length - 1 && !hiddenCls.has(i);
      hiddenCls = onlyThis ? new Set() : new Set(names.map((_, k) => k).filter(k => k !== i));
    } else if (hiddenCls.has(i)) hiddenCls.delete(i); else hiddenCls.add(i);
    if (sel && sel.t === 'box' && boxes[sel.i] && hiddenCls.has(boxes[sel.i][0])) setSel(null);
    renderClasses(); recalcProps(); updateHud(); invalidate();
  }
  const recentCls = [];
  function noteRecent(c) { const k = recentCls.indexOf(c); if (k >= 0) recentCls.splice(k, 1); recentCls.unshift(c); recentCls.length = Math.min(recentCls.length, 5); }

  let pop = null;
  function closePop() { if (!pop) return; const p = pop; pop = null; p.el.remove(); document.removeEventListener('mousedown', p.outside, true); if (p.restore && p.restore.isConnected) { try { p.restore.focus({ preventScroll: true }); } catch (e) { /* gone */ } } }
  /** Anchored popover. Class picker: type to filter (name or id), ↑↓ Enter, Esc; recent classes first. */
  function openPop({ x, y, title, actions = [], pick = null, extra = null }) {
    closePop();
    const restore = document.activeElement;
    const box = h('div', { class: 'ann-pop', role: 'dialog', 'aria-label': title || 'Menu' });
    if (title) box.appendChild(h('div', { class: 'menu-label' }, title));
    actions.forEach(a => { if (!a) return; if (a.sep) { box.appendChild(h('div', { class: 'menu-sep' })); return; } box.appendChild(h('button', { class: cls('menu-item', a.danger && 'danger'), disabled: !!a.disabled, onClick: () => { closePop(); a.onClick(); } }, icon(a.icon || 'circle', 14), h('span', a.label), a.kbd ? h('span', { class: 'kbd' }, a.kbd) : null)); });
    if (extra) box.appendChild(extra);
    let input = null, list = null, items = [], cur = 0;
    if (pick) {
      if (actions.length) box.appendChild(h('div', { class: 'menu-sep' }));
      input = h('input', { class: 'input input-sm', placeholder: pick.placeholder || 'Search classes…', 'aria-label': 'Filter classes' });
      list = h('div', { class: 'ann-pop-list', role: 'listbox' });
      box.appendChild(h('div', { class: 'ann-pop-search' }, input)); box.appendChild(list);
      const draw = () => {
        const q = input.value.trim().toLowerCase();
        const all = names.map((n, i) => i);
        const order = [...recentCls.filter(i => i < names.length), ...all.filter(i => !recentCls.includes(i))];
        items = order.filter(i => !q || String(i) === q || names[i].toLowerCase().includes(q) || String(i).startsWith(q));
        cur = Math.min(cur, Math.max(0, items.length - 1));
        list.replaceChildren(...(items.length ? items.map((i, k) => h('button', { class: cls('menu-item', k === cur && 'active', i === pick.current && 'current'), role: 'option', 'aria-selected': String(k === cur), onMousemove: () => { if (cur !== k) { cur = k; draw(); } }, onClick: () => { closePop(); pick.onPick(i); } },
          h('span', { class: 'swatch', style: `background:${classColor(i)}` }), h('span', { class: 'grow truncate' }, `${i} · ${names[i]}`), i === pick.current ? icon('check', 12) : null, i < 10 ? h('span', { class: 'kbd' }, String(i)) : null))
          : [h('div', { class: 'small faint', style: 'padding:6px 9px' }, names.length ? 'No class matches' : 'No classes yet')]));
        const a = list.children[cur]; if (a && a.scrollIntoView) a.scrollIntoView({ block: 'nearest' });
      };
      input.addEventListener('input', () => { cur = 0; draw(); });
      input.addEventListener('keydown', (e) => {
        if (e.key === 'ArrowDown') { e.preventDefault(); cur = Math.min(items.length - 1, cur + 1); draw(); }
        else if (e.key === 'ArrowUp') { e.preventDefault(); cur = Math.max(0, cur - 1); draw(); }
        else if (e.key === 'Enter' && !e.isComposing) { e.preventDefault(); const i = items[cur]; closePop(); if (i != null) pick.onPick(i); }
        else if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closePop(); }
        e.stopPropagation();
      });
      draw();
    }
    box.addEventListener('keydown', (e) => { if (e.key === 'Escape') { e.preventDefault(); e.stopPropagation(); closePop(); } });
    document.body.appendChild(box);
    const bw = box.offsetWidth, bh = box.offsetHeight;
    box.style.left = Math.max(8, Math.min(innerWidth - bw - 8, x)) + 'px';
    box.style.top = Math.max(8, Math.min(innerHeight - bh - 8, y)) + 'px';
    const outside = (e) => { if (!box.contains(e.target)) closePop(); };
    document.addEventListener('mousedown', outside, true);
    pop = { el: box, outside, restore };
    (input || box.querySelector('button') || box).focus({ preventScroll: true });
  }
  function classSearch() {
    const s = selectedItem();
    const r = stage.getBoundingClientRect();
    const at = mouse.inside ? { x: r.left + mouse.x + 12, y: r.top + mouse.y + 12 } : { x: r.left + r.width / 2 - 120, y: r.top + 60 };
    openPop({ ...at, title: s ? 'Set class of the selected ' + (s.t === 'prop' ? 'proposal' : 'box') : 'Class for new boxes', pick: { current: s ? itemCls(s) : activeCls, onPick: (i) => pickClass(i) } });
  }

  // ---------------------------------------------------------------- view transforms
  let dpr = window.devicePixelRatio || 1;
  function resizeCanvas() {
    const r = stage.getBoundingClientRect(); dpr = window.devicePixelRatio || 1;
    const w = Math.max(1, Math.round(r.width * dpr)), hh = Math.max(1, Math.round(r.height * dpr));
    for (const c of [cImg, cOv, cache]) { c.width = w; c.height = hh; }
    cImg.style.width = cOv.style.width = r.width + 'px'; cImg.style.height = cOv.style.height = r.height + 'px';
    if (!customView) fit();
    invalidate(true);
  }
  // Fitting leaves the filmstrip's band free, so a fitted picture is never under the strip.
  const viewH = () => Math.max(40, stage.clientHeight - filmBand());
  function fit() {
    if (!img) return;
    const cw = stage.clientWidth, ch = viewH();
    fitScale = Math.min(cw / W, ch / H) * 0.96;
    scale = fitScale; tx = (cw - W * scale) / 2; ty = (ch - H * scale) / 2; customView = false; holdFit = false; zoomBack = null; updateZoom();
  }
  function setView(ns, ntx, nty) {
    scale = ns; tx = ntx; ty = nty; customView = Math.abs(scale - fitScale) > 1e-6 || Math.abs(tx - (stage.clientWidth - W * scale) / 2) > 0.5;
    updateZoom(); invalidate(true);
  }
  function zoomAt(sx, sy, factor) {
    if (!img) return;
    const ns = Math.max(fitScale * 0.15, Math.min(60, scale * factor));
    setView(ns, sx - (sx - tx) * (ns / scale), sy - (sy - ty) * (ns / scale));
  }
  function zoomBy(f) { const p = mouse.inside ? mouse : { x: stage.clientWidth / 2, y: viewH() / 2 }; zoomAt(p.x, p.y, f); }
  function zoomTo(ns) { zoomBy(ns / scale); }
  function zoomToBox() {
    const s = selectedItem();
    if (!s) { zoomTo(1 / dpr); return; }
    const b = itemBox(s);
    if (zoomBack && zoomBack.key === itemKey(s)) { const z = zoomBack; zoomBack = null; setView(z.scale, z.tx, z.ty); return; }
    const prev = { scale, tx, ty, key: itemKey(s) };
    const cw = stage.clientWidth, ch = viewH();
    const bw = Math.max(b[3] * W, 2), bh = Math.max(b[4] * H, 2);
    const ns = Math.max(fitScale * 0.15, Math.min(60, Math.min(cw * 0.55 / bw, ch * 0.55 / bh)));
    setView(ns, cw / 2 - b[1] * W * ns, ch / 2 - b[2] * H * ns);
    zoomBack = prev;
  }
  const toImg = (sx, sy) => ({ x: (sx - tx) / scale, y: (sy - ty) / scale });
  const rectOf = (b) => ({ x: (b[1] - b[3] / 2) * W * scale + tx, y: (b[2] - b[4] / 2) * H * scale + ty, w: b[3] * W * scale, h: b[4] * H * scale });
  function updateZoom() { const t = Math.round(scale * dpr * 100) + '%'; if (zoomEl.textContent !== t) zoomEl.textContent = t; }

  // ---------------------------------------------------------------- selection helpers
  const selectedItem = () => (sel && (sel.t === 'box' ? boxes[sel.i] : props[sel.i]) ? sel : null);
  const itemBox = (s) => (s.t === 'box' ? boxes[s.i] : props[s.i].b);
  const itemCls = (s) => itemBox(s)[0];
  const itemKey = (s) => (s.t === 'box' ? 'b' + s.i : 'p' + props[s.i].uid);
  // activeCls is the class for new boxes: only a palette pick or a digit with nothing selected changes it. Selecting
  // a box highlights that box's class in the list (hlCls) and leaves the class for new boxes alone.
  const hlCls = () => { const s = selectedItem(); return s ? itemCls(s) : activeCls; };
  const selKey = () => { const s = selectedItem(); return s ? itemKey(s) : '-'; };
  function setSel(s, { scroll = true } = {}) {
    sel = s; focusEdge = null;
    if (classBuf && selKey() !== classBufFor) clearClassBuf();      // a pending first digit belongs to the item it went to
    markActiveClass();
    updateClassHead(); invalidate(); renderList(); renderPropList(); updateHud();
    if (scroll && s && s.t === 'box') scrollListTo(s.i);
  }

  // ---------------------------------------------------------------- proposals model
  function normProp(p) {
    return { uid: uidSeq++, id: p.id ?? null, b: [Number(p.cls) || 0, +p.xc, +p.yc, +p.w, +p.h], conf: p.conf == null ? null : +p.conf,
      kind: p.kind || 'detect', label: p.label || '', reason: p.reason || '', run_id: p.run_id ?? null, ref_row: p.ref_row ?? null,
      ref_box: refGeom(p.ref_box), state: p.state || 'pending', local: !!p.local, refBox: p.refBox || null, points: p.points || null };
  }
  const isPending = (p) => p.state === 'pending';
  // A proposal for an existing box (loose / swap / spurious / refine) points at that box object. ref_row is the row
  // at audit time; if the file changed since, the matching row is searched by geometry instead.
  function resolveRef(p) {
    if (p.refBox && boxes.includes(p.refBox)) return p.refBox;
    const r = refByGeom(boxes, p);
    if (r !== undefined) return (p.refBox = r);
    if (p.ref_row == null) return null;
    const minIou = p.kind === 'loose' ? 0.3 : 0.6;
    const at = boxes[p.ref_row];
    if (at && iou(at, p.b) >= minIou) return (p.refBox = at);
    let best = null, bi = minIou;
    for (const b of boxes) { const v = iou(b, p.b); if (v >= bi) { bi = v; best = b; } }
    return (p.refBox = best);
  }
  let visProps = [], propStats = { pending: 0, below: 0, dup: 0, hiddenCls: 0 };
  function recalcProps() {
    const vis = []; const st = { pending: 0, below: 0, dup: 0, hiddenCls: 0 };
    for (const p of props) {
      p.vis = false; p.dup = null; p.conflict = -1;
      if (!isPending(p)) continue;
      st.pending++;
      if (AUDIT.has(p.kind) || (p.kind === 'refine' && p.refBox)) { resolveRef(p); p.stale = AUDIT.has(p.kind) && p.kind !== 'missing' && !p.refBox; }
      if (hiddenCls.has(p.b[0])) { st.hiddenCls++; continue; }
      // A box SAM drew where the user just clicked (W, R) is shown whatever its score: they asked for it.
      if (p.conf != null && p.conf < thr && !(p.local && p.kind === 'refine')) { st.below++; continue; }
      if (!AUDIT.has(p.kind) && !(p.kind === 'refine' && p.refBox)) {
        for (let k = 0; k < boxes.length; k++) {
          if (iou(boxes[k], p.b) >= DUP_IOU) { if (boxes[k][0] === p.b[0]) { p.dup = 'same'; break; } p.conflict = k; }
        }
        if (p.dup === 'same') { st.dup++; continue; }
      }
      p.vis = true; vis.push(p);
    }
    vis.sort((a, b) => (b.conf ?? -1) - (a.conf ?? -1));        // most confident first; no confidence last
    visProps = vis; propStats = st;
    if (sel && sel.t === 'prop' && props[sel.i] && isPending(props[sel.i]) && !props[sel.i].vis) sel = null;   // see selectProp
  }
  function propsChanged() { recalcProps(); renderProps(); invalidate(); updateHud(); filmTouch(); if (flagFocus) renderFlagInfo(); }
  function setThr(v) {
    v = Math.max(0, Math.min(1, Math.round(Number(v) * 100) / 100));
    if (!isFinite(v) || v === thr) return;
    thr = v; LS.set(`fovea.ann.thr.${ds.id}`, v);
    if (sel && sel.t === 'prop' && props[sel.i] && !props[sel.i].vis) sel = null;
    propsChanged();
    fovea.bus.emit('annotate:threshold', { value: thr });
  }
  // Decisions are sent when the user leaves the image (and on unload), never per keypress: until then an undo is
  // purely local, and the boxes they produced are already on disk when the server hears about them.
  function queueDecision(p, state) {
    if (p.local || p.id == null) return;
    touched = true;
    decisionQueue.set(p.id, state);
  }
  /** Decisions reach the server only after the boxes they produced are on disk. One send at a time: a send already
      on its way is waited for (a failed one puts its decisions back, and they go with these). Resolves true when
      nothing is left unsent. A failure keeps the decisions queued — the save state says "Decisions not saved", an
      image visited again shows them as made — and retries in the background (4 s, then longer); an answer a retry
      cannot change drops them, and says so. */
  async function flushDecisions() {
    clearTimeout(decTimer);
    for (let round = 0; round < 3; round++) {
      while (decSending) await decSending;
      clearTimeout(decTimer);
      if (!decisionQueue.size) return true;
      if (dirty || saving) { if (!destroyed) decTimer = setTimeout(() => flushDecisions(), 500); return false; }
      const p = sendDecisions(decisionQueue, (body) => fovea.api.post(`/api/datasets/${ds.id}/proposals/decide`, body));
      decSending = p;
      let r;
      try { r = await p; } finally { if (decSending === p) decSending = null; }
      if (r.ok) {
        decFails = 0;
        if (r.sent) fovea.bus.emit('proposals:changed', { dataset: ds.id, source: 'annotate' });
        if (decFail && !decisionQueue.size) { decFail = null; if (!destroyed && !dirty && !saving && !loading && !labelsError) setSaveState('saved'); }
        continue;                                    // decisions made during the send go now too
      }
      decFails++;
      const why = (r.error && r.error.message) || 'network error';
      if (destroyed) { if (r.retry) finalDecisions(false); else ui.toast(`Proposal decisions were not saved: ${why}`, { type: 'error', timeout: 8000 }); return false; }   // the view is gone: one last try, then a toast
      decFail = r.retry ? 'Decisions not saved' : null;
      if (!dirty && !saving && !loading) setSaveState('error', 'Decisions not saved');   // else the label state shows; decFail follows it
      if (Date.now() - lastFailToast > 8000) { lastFailToast = Date.now(); ui.toast(`Could not record proposal decisions: ${why}${r.retry ? ' — retrying' : ''}`, { type: 'error' }); }
      if (r.retry) decTimer = setTimeout(() => flushDecisions(), 4000 * 2 ** Math.min(2, decFails - 1));   // 4, 8, then every 16 s
      return false;
    }
    return !decisionQueue.size;
  }
  /** Turn a proposal into the label change it suggests. Returns the box index the change landed on (or -1). */
  function materialize(p) {
    const ref = p.refBox && boxes.includes(p.refBox) ? p.refBox : null;
    let at = -1;
    p.merged = null;
    if (p.kind === 'spurious') { if (ref) boxes.splice(boxes.indexOf(ref), 1); }
    else if (p.kind === 'swap') { if (ref) { ref[0] = p.b[0]; at = settleFix(p, ref); } }
    else if ((p.kind === 'loose' || p.kind === 'refine') && ref) { if (ref.kind) { note('Segmentation / pose rows keep their shape — only class and deletion apply', 'warn'); at = boxes.indexOf(ref); } else { ref[1] = p.b[1]; ref[2] = p.b[2]; ref[3] = p.b[3]; ref[4] = p.b[4]; at = settleFix(p, ref); } }
    else if (AUDIT.has(p.kind) && p.kind !== 'missing') { /* stale: the row it refers to is gone */ }
    else { boxes.push([p.b[0], p.b[1], p.b[2], p.b[3], p.b[4]]); at = boxes.length - 1; }
    p.state = 'accepted';
    queueDecision(p, 'accepted');
    return at;
  }
  /** A fix just moved or re-classed `ref`. If that puts it on another box of its class (IoU ≥ 0.7) — typically the
      tighter box already accepted from the detection the ⤢ finding came from — the two would label one object
      twice, so `ref` goes and the box already there stays; `p.merged` keeps the numbers for the note. Returns the
      index of the box the fix ended on. */
  function settleFix(p, ref) {
    const o = dupOf(boxes, ref);
    if (!o) return boxes.indexOf(ref);
    const gone = boxes.indexOf(ref);
    boxes.splice(gone, 1);
    const at = boxes.indexOf(o);
    p.merged = { gone: gone + 1, kept: at + 1 };
    return at;
  }
  /** After an accept: a fix that was merged into the box already there says so; a box the accept made or moved that
      lands on a box of another class says that (noteOverlap). */
  function noteAccepted(p, at) {
    if (p.merged) {
      const { gone, kept } = p.merged;
      const what = p.kind === 'swap' ? `Box #${gone} as ${className(names, p.b[0])}` : `${p.kind === 'refine' ? 'Refined' : 'Tightened'} box #${gone}`;
      note(`${what} would duplicate #${kept} — removed #${gone} instead · ${MOD}Z undoes the accept`, 'warn', true);
    } else if (at >= 0) noteOverlap(at);
  }
  function nextPendingAfter(p) {
    const k = visProps.indexOf(p);
    const rest = visProps.filter(x => x !== p && isPending(x));
    if (!rest.length) return null;
    const after = visProps.slice(k + 1).find(x => x !== p && isPending(x));
    return after || rest[0];
  }
  // Only a proposal the editor shows can be selected: Enter accepts the selection, and a hidden one (under the
  // threshold, a hidden class, a duplicate of a box) must never be accepted unseen.
  function selectProp(p) { const i = p && p.vis ? props.indexOf(p) : -1; setSel(i >= 0 ? { t: 'prop', i } : null); }
  /** A static note (no modal) when nearly everything is accepted untouched: a sign of rubber-stamping. */
  function countDecision(kind, n = 1) {
    reviewStats[kind] += n;
    const { accepted, rejected, edited } = reviewStats, total = accepted + rejected + edited;
    if (!reviewStats.warned && total >= 200 && edited === 0 && accepted / total >= 0.98) {
      reviewStats.warned = true;
      note('Most proposals were accepted without any change — consider spot-checking at a lower threshold', 'warn', true);
    }
  }
  /** After an accept added box `bi`: say so when it lands on another box (IoU ≥ 0.7). Of another class it is two
      classes on one object (the ⇄ a proposal row shows), of the same class a duplicate — and C would copy either. */
  function noteOverlap(bi) {
    const b = boxes[bi]; if (!b) return;
    const k = boxes.findIndex((x, j) => j !== bi && iou(x, b) >= DUP_IOU);
    if (k < 0) return;
    const what = boxes[k][0] !== b[0] ? `(${className(names, b[0])}) overlaps #${k + 1} (${className(names, boxes[k][0])}) — two classes on one object?` : `duplicates #${k + 1}`;
    note(`Box #${bi + 1} ${what} ${MOD}Z undoes the accept`, 'warn', true);
  }
  function acceptProp(p) {
    if (!p || !editable() || !isPending(p)) return;
    if (p.stale) { note('The label row this suggestion refers to has changed — rejected instead', 'warn'); rejectProp(p); return; }
    const next = nextPendingAfter(p);
    pushHistory('accept proposal');
    const at = materialize(p);
    changed();
    countDecision('accepted');
    if (next) selectProp(next); else setSel(at >= 0 ? { t: 'box', i: at } : null);
    noteAccepted(p, at);
  }
  function rejectProp(p) {
    if (!p || !editable() || !isPending(p)) return;
    const next = nextPendingAfter(p);
    pushHistory('reject proposal');
    p.state = 'rejected'; queueDecision(p, 'rejected');
    propsChanged(); countDecision('rejected');
    if (next) selectProp(next); else setSel(null);
  }
  function acceptAllVisible() {
    if (!editable()) return;
    const list = visProps.filter(p => isPending(p) && !p.stale);
    if (!list.length) { note('No visible proposals to accept'); return; }
    // Most confident first; a proposal that would land on a box — one here, one this accept adds, or where a fix
    // in it moves a box — is rejected (same class) or left to decide (⇄): see bulkAcceptPlan.
    const { take, dupes, left } = bulkAcceptPlan(list, boxes);
    if (!take.length && !dupes.length) { note(`Nothing accepted: ${left.length === 1 ? 'the proposal overlaps' : `all ${left.length} overlap`} a box of another class (⇄) — decide one by one (Enter · Del)`, 'warn'); return; }
    pushHistory('accept all');
    const made = [];
    for (const p of take) { const at = materialize(p); if (at >= 0) made.push({ box: boxes[at], conf: p.conf }); }
    for (const p of dupes) { p.state = 'rejected'; queueDecision(p, 'rejected'); }
    // A fix that would have put its box on another box of its class removed it instead (settleFix).
    const merged = take.filter(p => p.merged).length;
    changed(); countDecision('accepted', take.length); if (dupes.length) countDecision('rejected', dupes.length);
    // Spot-check (automation-bias safeguard): one accepted box from the least confident third.
    const withConf = made.filter(m => m.conf != null && boxes.includes(m.box)).sort((a, b) => a.conf - b.conf);
    const pool = withConf.slice(0, Math.max(1, Math.ceil(withConf.length / 3)));
    const pickM = pool.length ? pool[Math.floor(Math.random() * pool.length)] : null;
    const at = pickM ? boxes.indexOf(pickM.box) : -1;
    setSel(at >= 0 ? { t: 'box', i: at } : null);
    const extra = (dupes.length ? ` · ${dupes.length} duplicate${dupes.length === 1 ? '' : 's'} rejected` : '')
      + (merged ? ` · ${merged} fixed box${merged === 1 ? '' : 'es'} matched a box already there — removed, not doubled` : '')
      + (left.length ? ` · ${left.length} overlapping another class left to decide (⇄)` : '');
    note(`Accepted ${take.length}` + (at >= 0 ? ` · spot-check box #${at + 1} (${pickM.conf.toFixed(2)})` : '') + extra + ' — Esc to dismiss', left.length || merged ? 'warn' : 'ok', true);
  }
  function rejectAllVisible() {
    if (!editable()) return;
    const list = visProps.filter(isPending);
    if (!list.length) { note('No visible proposals to reject'); return; }
    pushHistory('reject all');
    for (const p of list) { p.state = 'rejected'; queueDecision(p, 'rejected'); }
    setSel(null); propsChanged(); countDecision('rejected', list.length);
    note(`Rejected ${list.length} proposal${list.length === 1 ? '' : 's'} — ${MOD}Z to undo`);
  }
  /** Pending proposals this editor hides — under the threshold, or duplicates of a same-class box ("already
      labeled") — can never be decided one by one; this rejects them, one undo step. */
  const hiddenPending = () => props.filter(p => isPending(p) && !p.vis && !hiddenCls.has(p.b[0]));
  function rejectHidden() {
    if (!editable()) return;
    const list = hiddenPending();
    if (!list.length) return;
    pushHistory('reject hidden');
    for (const p of list) { p.state = 'rejected'; queueDecision(p, 'rejected'); }
    if (sel && sel.t === 'prop' && !isPending(props[sel.i])) sel = null;
    propsChanged(); countDecision('rejected', list.length);
    note(`Rejected ${list.length} hidden proposal${list.length === 1 ? '' : 's'} — ${MOD}Z to undo`);
  }
  /** Leaving an image that was worked on: pending server proposals that duplicate a same-class box are redundant by
      definition (accepting one adds a second box on the object), so they are rejected with the other decisions —
      ⇧N, the strip's ◌ and ai=pending then stop pointing at an image where nothing is left to decide. */
  function dropRedundant() {
    if (!touched || !item || labelsError) return;
    for (const p of props) {
      if (!isPending(p) || p.local || p.id == null || !addsBox(p) || AUDIT.has(p.kind)) continue;
      if (boxes.some(b => b[0] === p.b[0] && iou(b, p.b) >= DUP_IOU)) { p.state = 'rejected'; queueDecision(p, 'rejected'); }
    }
  }
  /** Editing a proposal accepts it: the edit continues on the box it became. Returns that box index or -1. */
  function beginEditProp(i) {
    const p = props[i];
    if (!p || !isPending(p)) return -1;
    countDecision('edited');
    if (p.kind === 'spurious') { const ref = resolveRef(p); p.state = 'rejected'; queueDecision(p, 'rejected'); return ref ? boxes.indexOf(ref) : -1; }
    if (p.stale) return -1;
    const at = materialize(p);
    if (p.merged) noteAccepted(p, at);
    return at;
  }
  function addLocalProps(list, kind, label) {
    const made = list.map(b => normProp({ cls: b[0], xc: b[1], yc: b[2], w: b[3], h: b[4], conf: b[5] ?? null, kind, label, local: true }));
    props = props.concat(made);
    propsChanged();
    return made;
  }

  // ---------------------------------------------------------------- drawing (rAF-coalesced, layered)
  let needImg = true, needCache = true, rafId = 0;
  function invalidate(all = false, cacheToo = true) { if (all) needImg = true; if (cacheToo) needCache = true; if (!rafId && !destroyed) rafId = requestAnimationFrame(frame); }
  const overlayOnly = () => invalidate(false, false);
  function frame() {
    rafId = 0;
    if (destroyed) return;
    if (needImg) { drawImageLayer(); needImg = false; }
    if (needCache) { drawCache(); needCache = false; }
    drawOverlay();
    drawLoupe();
    drawMinimap();
  }
  function drawImageLayer() {
    gi.setTransform(1, 0, 0, 1, 0, 0); gi.clearRect(0, 0, cImg.width, cImg.height);
    if (!img) return;
    gi.setTransform(dpr, 0, 0, dpr, 0, 0);
    const eff = scale * W / (img.naturalWidth || W);
    gi.imageSmoothingEnabled = eff * dpr < 2;
    gi.drawImage(img, tx, ty, W * scale, H * scale);
  }
  const fillAlpha = () => (opts.fill == null ? (boxes.length > 40 ? 0 : 0.08) : Number(opts.fill));
  const twCache = new Map();
  function textW(g, t) { let w = twCache.get(t); if (w == null) { g.font = FONT; w = g.measureText(t).width; if (twCache.size > 4000) twCache.clear(); twCache.set(t, w); } return w; }
  // A label longer than LABEL_MAX is shortened in the middle ("0 person_lying_…after_fall"), never cut off mid-word.
  const fitCache = new Map();
  function fitLabel(g, t) {
    if (textW(g, t) <= LABEL_MAX) return t;
    let v = fitCache.get(t);
    if (v == null) { v = middleEllipsis(t, LABEL_MAX, (x) => textW(g, x)); if (fitCache.size > 2000) fitCache.clear(); fitCache.set(t, v); }
    return v;
  }
  const chipW = (g, t) => textW(g, fitLabel(g, t)) + 8;
  const labelText = (c) => `${c} ${className(names, c)}`;
  function propText(p) {
    const nm = className(names, p.b[0]);
    if (p.kind === 'swap') return `⇄ ${p.refBox ? className(names, p.refBox[0]) : '?'} → ${nm}`;
    if (p.kind === 'spurious') return `${GLYPH.spurious} no support? ${nm}`;
    const g0 = GLYPH[p.kind] && p.kind !== 'detect' ? GLYPH[p.kind] + ' ' : '';
    const conf = p.conf != null ? p.conf.toFixed(2) + ' ' : '';
    const cf = p.conflict >= 0 ? ` ⇄#${p.conflict + 1}` : '';
    return `${g0}${conf}${nm}${cf}`;
  }
  function chip(g, x, y, text, { bg, fg, border = null, cw, ch }) {
    text = fitLabel(g, text);
    const tw = textW(g, text) + 8, th = 16;
    x = Math.max(0, Math.min(cw - tw, x)); y = Math.max(0, Math.min(ch - th, y));
    g.fillStyle = bg; g.fillRect(x, y, tw, th);
    if (border) { g.strokeStyle = border; g.lineWidth = 1; g.strokeRect(x + 0.5, y + 0.5, tw - 1, th - 1); }
    g.save(); g.beginPath(); g.rect(x, y, tw, th); g.clip();
    g.fillStyle = fg; g.font = FONT; g.textBaseline = 'middle'; g.fillText(text, x + 4, y + th / 2 + 0.5);
    g.restore();
    return { x, y, w: tw, h: th };
  }
  /** Place a label chip above, inside or below a box, skipping it when every spot collides with one already placed. */
  function placeLabel(r, tw, placed, cw, ch) {
    const th = 16, cand = [[r.x, r.y - th], [r.x, r.y], [r.x, r.y + r.h]];
    for (const [cx, cy] of cand) {
      const x = Math.max(0, Math.min(cw - tw, cx)), y = Math.max(0, Math.min(ch - th, cy));
      if (!placed.hit(x, y, tw, th)) { placed.add(x, y, tw, th); return { x, y }; }
    }
    return null;
  }
  function placer(cw, ch) {
    const C = 64, grid = new Map();
    const keys = (x, y, w, hh) => { const out = []; for (let gx = Math.floor(x / C); gx <= Math.floor((x + w) / C); gx++) for (let gy = Math.floor(y / C); gy <= Math.floor((y + hh) / C); gy++) out.push(gx + ':' + gy); return out; };
    return {
      hit(x, y, w, hh) { for (const k of keys(x, y, w, hh)) { const l = grid.get(k); if (l) for (const r of l) if (x < r.x + r.w && x + w > r.x && y < r.y + r.h && y + hh > r.y) return true; } return false; },
      add(x, y, w, hh) { const r = { x, y, w, h: hh }; for (const k of keys(x, y, w, hh)) { let l = grid.get(k); if (!l) grid.set(k, l = []); l.push(r); } },
    };
  }
  function strokeBox(g, r, color, width, dash) {
    g.lineWidth = width; g.strokeStyle = color;
    if (dash) g.setLineDash(dash);
    g.strokeRect(r.x, r.y, r.w, r.h);
    if (dash) g.setLineDash([]);
  }
  let hatch = null;
  function hatchPattern() {
    if (hatch) return hatch;
    const t = document.createElement('canvas'); t.width = t.height = 8;
    const c = t.getContext('2d');
    c.fillStyle = 'rgba(20,20,20,.32)'; c.fillRect(0, 0, 8, 8);
    c.strokeStyle = 'rgba(235,235,235,.55)'; c.lineWidth = 1; c.beginPath(); c.moveTo(0, 8); c.lineTo(8, 0); c.stroke();
    hatch = gc.createPattern(t, 'repeat');
    return hatch;
  }
  function drawCache() {
    const cw = stage.clientWidth, ch = stage.clientHeight;
    gc.setTransform(1, 0, 0, 1, 0, 0); gc.clearRect(0, 0, cache.width, cache.height);
    if (!img || hiddenAll) return;
    gc.setTransform(dpr, 0, 0, dpr, 0, 0);
    const skipB = sel && sel.t === 'box' ? sel.i : -1, skipP = sel && sel.t === 'prop' ? sel.i : -1;
    if (opts.ghost && ghost && ghost.forId === (item && item.id) && ghost.boxes && ghost.boxes.length) {
      // Two-tone and static: dark 2 px dots with light 1 px dots in the gaps, so the outline reads on white walls
      // and on dark floors alike (a single grey dash vanished on light frames). No animation (visual comfort).
      const rs = ghost.boxes.map(rectOf);
      gc.setLineDash([3, 3]);
      gc.lineDashOffset = 0; gc.lineWidth = 2; gc.strokeStyle = 'rgba(0,0,0,.8)';
      for (const r of rs) gc.strokeRect(r.x, r.y, r.w, r.h);
      gc.lineDashOffset = 3; gc.lineWidth = 1; gc.strokeStyle = 'rgba(255,255,255,.95)';
      for (const r of rs) gc.strokeRect(r.x, r.y, r.w, r.h);
      gc.setLineDash([]); gc.lineDashOffset = 0;
    }
    if (regions.length) {
      // Regions that are not objects (ignored regions, DontCare, crowds): hatched, dashed outline, never selectable.
      gc.fillStyle = hatchPattern(); gc.setLineDash([4, 3]); gc.lineWidth = 1; gc.strokeStyle = 'rgba(235,235,235,.75)';
      for (const g of regions) {
        const r = rectOf([0, ...g.box]); if (r.x > cw || r.y > ch || r.x + r.w < 0 || r.y + r.h < 0) continue;
        gc.fillRect(r.x, r.y, r.w, r.h); gc.strokeRect(r.x, r.y, r.w, r.h);
      }
      gc.setLineDash([]);
    }
    const fa = fillAlpha();
    for (let i = 0; i < boxes.length; i++) {
      if (i === skipB) continue;
      const b = boxes[i]; if (hiddenCls.has(b[0])) continue;
      const r = rectOf(b); if (r.x > cw || r.y > ch || r.x + r.w < 0 || r.y + r.h < 0) continue;
      const color = classColor(b[0]);
      if (fa > 0) { gc.fillStyle = withAlpha(color, fa); gc.fillRect(r.x, r.y, r.w, r.h); }
      strokeBox(gc, r, color, 1.75);
    }
    const fg = flagFocus && item && flagFocus.forId === item.id ? flagFocus : null;
    if (fg) {
      // A pressed QA flag: the boxes it names get an amber ring; the twin's / neighbour's boxes, or where the frames
      // around put one, an amber dashed outline over a dark one (reads on bright and dark frames). Static.
      for (const b of fg.refs) {
        const i = boxes.indexOf(b); if (i < 0 || i === skipB || hiddenCls.has(b[0])) continue;
        const r = rectOf(b); strokeBox(gc, { x: r.x - 3, y: r.y - 3, w: r.w + 6, h: r.h + 6 }, FLAG_GHOST, 2);
      }
      // Of another image's boxes, those this image also has (same class, IoU ≥ 0.5) are drawn faint and unlabeled;
      // what differs is outlined last, over the box labels, and its labels are placed first: it is what is asked about.
      gc.setLineDash([3, 4]); gc.lineWidth = 1; gc.strokeStyle = withAlpha(FLAG_GHOST, 0.6);
      for (const b of fg.boxes) if (!flagDiffers(b)) { const r = rectOf(b); gc.strokeRect(r.x - 2, r.y - 2, r.w + 4, r.h + 4); }
      gc.setLineDash([]);
    }
    const odd = fg ? fg.boxes.filter(b => flagDiffers(b)) : [];
    const strokeOdd = () => {
      if (!odd.length) return;
      const rs = odd.map(rectOf);
      gc.setLineDash([7, 4]);
      gc.lineWidth = 3.5; gc.strokeStyle = 'rgba(0,0,0,.72)'; for (const r of rs) gc.strokeRect(r.x, r.y, r.w, r.h);
      gc.lineWidth = 1.75; gc.strokeStyle = FLAG_GHOST; for (const r of rs) gc.strokeRect(r.x, r.y, r.w, r.h);
      gc.setLineDash([]);
    };
    for (const p of visProps) {
      const i = props.indexOf(p); if (i === skipP) continue;
      const r = rectOf(p.b); strokeBox(gc, r, classColor(p.b[0]), 1.5, [6, 4]);
    }
    if (opts.labels !== 'all') { strokeOdd(); return; }
    const placed = placer(cw, ch);
    for (const b of odd) {
      const r = rectOf(b); if (r.x > cw || r.y > ch || r.x + r.w < 0 || r.y + r.h < 0) continue;
      const t = `${fg.tag} ${className(names, b[0])}`, tw = chipW(gc, t);
      const at = placeLabel(r, tw, placed, cw, ch); if (!at) continue;
      chip(gc, at.x, at.y, t, { bg: 'rgba(0,0,0,.78)', fg: '#fde68a', border: FLAG_GHOST, cw, ch });
    }
    for (let i = 0; i < boxes.length; i++) {
      if (i === skipB) continue;
      const b = boxes[i]; if (hiddenCls.has(b[0])) continue;
      const r = rectOf(b); if (r.x > cw || r.y > ch || r.x + r.w < 0 || r.y + r.h < 0) continue;
      // Tiny boxes get the class id only: a full tag would hide the object it names.
      const full = labelText(b[0]); const t = r.w < 28 && r.h < 28 ? String(b[0]) : full;
      const tw = chipW(gc, t);
      const at = placeLabel(r, tw, placed, cw, ch); if (!at) continue;
      const color = classColor(b[0]);
      chip(gc, at.x, at.y, t, { bg: color, fg: textOn(color), cw, ch });
    }
    for (const p of visProps) {
      if (props.indexOf(p) === skipP) continue;
      const r = rectOf(p.b); const t = propText(p); const tw = chipW(gc, t);
      const at = placeLabel(r, tw, placed, cw, ch); if (!at) continue;
      chip(gc, at.x, at.y, t, { bg: 'rgba(0,0,0,.72)', fg: '#fff', border: classColor(p.b[0]), cw, ch });
    }
    strokeOdd();
  }
  function handlePoints(r) {
    // Below SMALL_BOX on screen the corner handles sit outside the box, so its whole inside is a move target.
    const o = Math.min(r.w, r.h) < SMALL_BOX ? 6 : 0;
    return { nw: { x: r.x - o, y: r.y - o }, ne: { x: r.x + r.w + o, y: r.y - o }, se: { x: r.x + r.w + o, y: r.y + r.h + o }, sw: { x: r.x - o, y: r.y + r.h + o } };
  }
  function drawOverlay() {
    const cw = stage.clientWidth, ch = stage.clientHeight;
    go.setTransform(1, 0, 0, 1, 0, 0); go.clearRect(0, 0, cOv.width, cOv.height);
    if (!img) return;
    go.drawImage(cache, 0, 0);
    go.setTransform(dpr, 0, 0, dpr, 0, 0);
    if (!hiddenAll) {
      // hovered (drawn over the cache), then the selected item last so nothing covers its handles
      const hv = hov && (hov.t === 'box' ? boxes[hov.i] && !hiddenCls.has(boxes[hov.i][0]) : props[hov.i] && props[hov.i].vis) && !(sel && sel.t === hov.t && sel.i === hov.i) ? hov : null;
      if (hv) {
        const b = itemBox(hv), r = rectOf(b), color = classColor(b[0]);
        if (hv.t === 'box') { go.fillStyle = withAlpha(color, 0.16); go.fillRect(r.x, r.y, r.w, r.h); strokeBox(go, r, color, 2.5); }
        else strokeBox(go, r, color, 2.5, [6, 4]);
        if (opts.labels === 'sel') { const t = hv.t === 'box' ? labelText(b[0]) : propText(props[hv.i]); const isP = hv.t === 'prop'; chip(go, r.x, r.y - 16 < 0 ? r.y : r.y - 16, t, isP ? { bg: 'rgba(0,0,0,.72)', fg: '#fff', border: color, cw, ch } : { bg: color, fg: textOn(color), cw, ch }); }
      }
      const s = selectedItem();
      if (s) {
        const b = itemBox(s), r = rectOf(b), color = classColor(b[0]);
        if (s.t === 'box') {
          go.fillStyle = withAlpha(color, 0.22); go.fillRect(r.x, r.y, r.w, r.h);
          strokeBox(go, r, '#fff', 2.5); strokeBox(go, { x: r.x + 2, y: r.y + 2, w: r.w - 4, h: r.h - 4 }, color, 1.5);
        } else {
          go.fillStyle = withAlpha(color, 0.04); go.fillRect(r.x, r.y, r.w, r.h);
          strokeBox(go, r, '#fff', 2.5, [6, 4]); strokeBox(go, r, color, 1.5, [6, 4]);
          const p = props[s.i];
          if (p.refBox && boxes.includes(p.refBox)) { const rr = rectOf(p.refBox); go.setLineDash([2, 2]); strokeBox(go, rr, 'rgba(255,255,255,.8)', 1); go.setLineDash([]); }
          if (p.points) for (const [px, py, pos] of p.points) { const x = px * W * scale + tx, y = py * H * scale + ty; go.fillStyle = pos ? '#22c55e' : '#ef4444'; go.beginPath(); go.arc(x, y, 4.5, 0, Math.PI * 2); go.fill(); go.strokeStyle = '#fff'; go.lineWidth = 1.5; go.stroke(); }
        }
        if (focusEdge && s.t === 'box') {
          const e = { left: [r.x, r.y, r.x, r.y + r.h], right: [r.x + r.w, r.y, r.x + r.w, r.y + r.h], top: [r.x, r.y, r.x + r.w, r.y], bottom: [r.x, r.y + r.h, r.x + r.w, r.y + r.h] }[focusEdge];
          go.lineWidth = 3; go.strokeStyle = '#fff'; go.beginPath(); go.moveTo(e[0], e[1]); go.lineTo(e[2], e[3]); go.stroke();
        }
        if (opts.labels !== 'off') { const t = s.t === 'box' ? labelText(b[0]) : propText(props[s.i]); chip(go, r.x, r.y - 16 < 0 ? r.y + r.h : r.y - 16, t, s.t === 'prop' ? { bg: 'rgba(0,0,0,.72)', fg: '#fff', border: color, cw, ch } : { bg: color, fg: textOn(color), cw, ch }); }
        if (!(s.t === 'box' && boxes[s.i].kind)) {
          const hp = handlePoints(r);
          for (const k of Object.keys(hp)) { const p = hp[k]; go.fillStyle = '#fff'; go.strokeStyle = color; go.lineWidth = 1.5; go.beginPath(); go.rect(p.x - 4, p.y - 4, 8, 8); go.fill(); go.stroke(); }
        }
      }
    }
    if (theirsOverlay) { go.setLineDash([4, 3]); for (const b of theirsOverlay) strokeBox(go, rectOf(b), '#fbbf24', 1.5); go.setLineDash([]); }
    if (corner && tool === 'box') {             // 2-click: the placed first corner
      const x = corner.x * scale + tx, y = corner.y * scale + ty;
      go.save(); go.lineWidth = 3; go.strokeStyle = 'rgba(0,0,0,.6)'; go.beginPath(); go.moveTo(x - 7, y); go.lineTo(x + 7, y); go.moveTo(x, y - 7); go.lineTo(x, y + 7); go.stroke();
      go.lineWidth = 1.5; go.strokeStyle = classColor(activeCls); go.stroke(); go.restore();
    }
    const draftA = drag && drag.kind === 'draw' && drag.started ? drag.a : corner && tool === 'box' && mouse.inside ? corner : null;
    if (draftA) {
      const a = draftA, c = toImg(mouse.x, mouse.y);
      const x0 = Math.max(0, Math.min(a.x, c.x)), y0 = Math.max(0, Math.min(a.y, c.y)), x1 = Math.min(W, Math.max(a.x, c.x)), y1 = Math.min(H, Math.max(a.y, c.y));
      const r = { x: x0 * scale + tx, y: y0 * scale + ty, w: (x1 - x0) * scale, h: (y1 - y0) * scale };
      const color = classColor(activeCls);
      go.fillStyle = withAlpha(color, 0.12); go.fillRect(r.x, r.y, r.w, r.h); strokeBox(go, r, color, 1.5, [5, 3]);
      chip(go, r.x, r.y - 18 < 0 ? r.y + r.h + 2 : r.y - 18, `${Math.round(x1 - x0)}×${Math.round(y1 - y0)} px`, { bg: 'rgba(0,0,0,.72)', fg: '#fff', cw, ch });
    }
    const guides = opts.crosshair && mouse.inside && !spaceDown && (tool === 'box' || tool === 'point' || tool === 'sam' || (drag && (drag.kind === 'draw' || drag.kind === 'resize')));
    if (guides) {
      go.save(); go.strokeStyle = 'rgba(255,255,255,.5)'; go.lineWidth = 1; go.setLineDash([4, 4]);
      go.beginPath(); go.moveTo(0, Math.round(mouse.y) + 0.5); go.lineTo(cw, Math.round(mouse.y) + 0.5); go.moveTo(Math.round(mouse.x) + 0.5, 0); go.lineTo(Math.round(mouse.x) + 0.5, ch); go.stroke();
      go.restore();
    }
    if (tool === 'point' && mouse.inside && !drag) {
      const w = opts.pointW * W * scale, hh = opts.pointH * H * scale;
      strokeBox(go, { x: mouse.x - w / 2, y: mouse.y - hh / 2, w, h: hh }, classColor(activeCls), 1.5, [5, 3]);
    }
  }
  function drawLoupe() {
    const on = opts.loupe && mouse.inside && img && !spaceDown;
    loupe.classList.toggle('hidden', !on);
    if (!on) return;
    const S = 160, Z = 4, cw = stage.clientWidth, ch = stage.clientHeight;
    let lx = mouse.x + 24, ly = mouse.y + 24;
    if (lx + S > cw) lx = mouse.x - 24 - S; if (ly + S > ch) ly = mouse.y - 24 - S;
    loupe.style.left = Math.max(0, lx) + 'px'; loupe.style.top = Math.max(0, ly) + 'px';
    const ld = window.devicePixelRatio || 1;
    if (loupe.width !== S * ld) { loupe.width = loupe.height = S * ld; loupe.style.width = loupe.style.height = S + 'px'; }
    gl.setTransform(ld, 0, 0, ld, 0, 0);
    gl.fillStyle = '#000'; gl.fillRect(0, 0, S, S);
    const c = toImg(mouse.x, mouse.y), span = S / (scale * Z);   // image px across the loupe
    const sx = (img.naturalWidth || W) / W;
    gl.imageSmoothingEnabled = false;
    gl.drawImage(img, (c.x - span / 2) * sx, (c.y - span / 2) * sx, span * sx, span * sx, 0, 0, S, S);
    const k = S / span, ox = c.x - span / 2, oy = c.y - span / 2;
    const lr = (b) => ({ x: ((b[1] - b[3] / 2) * W - ox) * k, y: ((b[2] - b[4] / 2) * H - oy) * k, w: b[3] * W * k, h: b[4] * H * k });
    if (!hiddenAll) {
      for (const b of boxes) if (!hiddenCls.has(b[0])) strokeBox(gl, lr(b), classColor(b[0]), 1);
      for (const p of visProps) strokeBox(gl, lr(p.b), classColor(p.b[0]), 1, [4, 3]);
    }
    const la = drag && drag.kind === 'draw' && drag.started ? drag.a : corner && tool === 'box' ? corner : null;
    if (la) { const a = la; strokeBox(gl, { x: (Math.min(a.x, c.x) - ox) * k, y: (Math.min(a.y, c.y) - oy) * k, w: Math.abs(c.x - a.x) * k, h: Math.abs(c.y - a.y) * k }, classColor(activeCls), 1, [4, 3]); }
    gl.strokeStyle = 'rgba(255,255,255,.85)'; gl.lineWidth = 1; gl.beginPath(); gl.moveTo(S / 2 + 0.5, 0); gl.lineTo(S / 2 + 0.5, S); gl.moveTo(0, S / 2 + 0.5); gl.lineTo(S, S / 2 + 0.5); gl.stroke();
  }
  let miniSrc = null;
  function drawMinimap() {
    const on = img && scale > fitScale * 2;
    minimap.classList.toggle('hidden', !on);
    if (!on) return;
    const md = window.devicePixelRatio || 1, MW = 140, MH = Math.max(40, Math.round(MW * H / W));
    if (minimap.width !== MW * md || minimap.height !== MH * md) { minimap.width = MW * md; minimap.height = MH * md; minimap.style.width = MW + 'px'; minimap.style.height = MH + 'px'; miniSrc = null; }
    if (!miniSrc || miniSrc.img !== img) {
      const c = document.createElement('canvas'); c.width = MW * md; c.height = MH * md;
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height); miniSrc = { img, c };
    }
    gm.setTransform(1, 0, 0, 1, 0, 0); gm.drawImage(miniSrc.c, 0, 0);
    gm.setTransform(md, 0, 0, md, 0, 0);
    const kx = MW / (W * scale), ky = MH / (H * scale);
    gm.strokeStyle = '#fff'; gm.lineWidth = 1.5; gm.strokeRect(-tx * kx, -ty * ky, stage.clientWidth * kx, stage.clientHeight * ky);
  }
  // Click or drag on the minimap centres the view there.
  function minimapJump(e) {
    const r = minimap.getBoundingClientRect();
    const ix = Math.max(0, Math.min(1, (e.clientX - r.left) / r.width)) * W, iy = Math.max(0, Math.min(1, (e.clientY - r.top) / r.height)) * H;
    setView(scale, stage.clientWidth / 2 - ix * scale, stage.clientHeight / 2 - iy * scale);
  }
  minimap.addEventListener('mousedown', (e) => {
    e.preventDefault(); e.stopPropagation(); minimapJump(e);
    const mv = (ev) => minimapJump(ev), up = () => { window.removeEventListener('mousemove', mv); window.removeEventListener('mouseup', up); };
    window.addEventListener('mousemove', mv); window.addEventListener('mouseup', up);
  });

  // ---------------------------------------------------------------- hit testing
  let grid = null;   // spatial index over boxes, built on demand above 200 boxes
  function buildGrid() {
    const G = 32, cells = new Map();
    boxes.forEach((b, i) => {
      const x0 = Math.max(0, Math.floor((b[1] - b[3] / 2) * G)), x1 = Math.min(G - 1, Math.floor((b[1] + b[3] / 2) * G));
      const y0 = Math.max(0, Math.floor((b[2] - b[4] / 2) * G)), y1 = Math.min(G - 1, Math.floor((b[2] + b[4] / 2) * G));
      for (let gx = x0; gx <= x1; gx++) for (let gy = y0; gy <= y1; gy++) { const k = gx * G + gy; let l = cells.get(k); if (!l) cells.set(k, l = []); l.push(i); }
    });
    grid = { G, cells, n: boxes.length };
  }
  function boxCandidates(c) {
    if (boxes.length <= 200) return boxes.map((_, i) => i);
    if (!grid || grid.n !== boxes.length) buildGrid();
    const G = grid.G, gx = Math.floor(c.x / W * G), gy = Math.floor(c.y / H * G), out = new Set();
    for (let dx = -1; dx <= 1; dx++) for (let dy = -1; dy <= 1; dy++) { const l = grid.cells.get((gx + dx) * G + gy + dy); if (l) l.forEach(i => out.add(i)); }
    return [...out];
  }
  function hitsAt(sx, sy) {
    if (hiddenAll) return [];
    const out = [], pad = 3, c = toImg(sx, sy);
    const inside = (b) => { const r = rectOf(b); return sx >= r.x - pad && sx <= r.x + r.w + pad && sy >= r.y - pad && sy <= r.y + r.h + pad ? r.w * r.h : -1; };
    for (const p of visProps) { const a = inside(p.b); if (a >= 0) out.push({ t: 'prop', i: props.indexOf(p), a: a - 0.5 }); }
    for (const i of boxCandidates(c)) { const b = boxes[i]; if (hiddenCls.has(b[0])) continue; const a = inside(b); if (a >= 0) out.push({ t: 'box', i, a }); }
    out.sort((x, y) => x.a - y.a);
    return out;
  }
  function handleAt(sx, sy) {
    const s = selectedItem(); if (!s) return null;
    if (s.t === 'box' && boxes[s.i].kind) return null;          // segmentation / pose rows are geometry-locked
    const r = rectOf(itemBox(s));
    let best = null, bd = HANDLE_HIT;
    const hp = handlePoints(r);
    for (const k of Object.keys(hp)) { const d = Math.hypot(sx - hp[k].x, sy - hp[k].y); if (d <= bd) { bd = d; best = { corner: k }; } }
    if (best) return best;
    if (Math.min(r.w, r.h) < SMALL_BOX) return null;
    const band = EDGE_BAND, inY = sy > r.y + band && sy < r.y + r.h - band, inX = sx > r.x + band && sx < r.x + r.w - band;
    if (inY && Math.abs(sx - r.x) <= band) return { edge: 'w' };
    if (inY && Math.abs(sx - (r.x + r.w)) <= band) return { edge: 'e' };
    if (inX && Math.abs(sy - r.y) <= band) return { edge: 'n' };
    if (inX && Math.abs(sy - (r.y + r.h)) <= band) return { edge: 's' };
    return null;
  }
  const CURSORS = { nw: 'nwse-resize', se: 'nwse-resize', ne: 'nesw-resize', sw: 'nesw-resize', w: 'ew-resize', e: 'ew-resize', n: 'ns-resize', s: 'ns-resize' };
  function cursorFor(sx, sy, alt) {
    if (spaceDown || tool === 'pan') return drag && drag.kind === 'pan' ? 'grabbing' : 'grab';
    if (!alt && editable() && !corner) { const hh = handleAt(sx, sy); if (hh) return CURSORS[hh.corner || hh.edge]; }
    const hits = hitsAt(sx, sy);
    const s = selectedItem();
    if (tool === 'select') return hits.length ? 'move' : 'default';
    if (tool === 'box') return opts.boxMode !== 'click2' && hits.length && s && hits.some(x => x.t === s.t && x.i === s.i) ? 'move' : 'crosshair';
    if (tool === 'sam') return 'crosshair';
    return 'none';
  }
  let lastCursor = '';
  function setCursor(c) { if (c !== lastCursor) { cOv.style.cursor = c; lastCursor = c; } }

  // ---------------------------------------------------------------- mouse
  const local = (e) => { const r = cOv.getBoundingClientRect(); return { x: e.clientX - r.left, y: e.clientY - r.top }; };
  cOv.addEventListener('mousedown', (e) => {
    closePop(); clearNote();
    if (!img) return;
    const p = local(e); mouse = { ...p, inside: true };
    pokeAuto();
    if (e.button === 1 || spaceDown || tool === 'pan') { drag = { kind: 'pan', x0: p.x, y0: p.y, tx0: tx, ty0: ty }; e.preventDefault(); setCursor('grabbing'); return; }
    if (e.button !== 0 || loading) return;
    const a = toImg(p.x, p.y);
    const s = selectedItem();
    const click2 = tool === 'box' && opts.boxMode === 'click2';
    // 2-click: with the first corner placed, the next press is the second corner (released where the pointer ends).
    if (click2 && corner) { drag = { kind: 'draw', a: { ...corner }, started: true, sx: p.x, sy: p.y, two: true }; overlayOnly(); return; }
    if (s && !e.altKey && editable()) {
      const hh = handleAt(p.x, p.y);
      if (hh) { drag = { kind: 'resize', hh, target: { ...s }, a, orig: itemBox(s).slice(), started: false, sx: p.x, sy: p.y }; return; }
    }
    const hits = hitsAt(p.x, p.y);
    if (tool === 'sam') { samClick(a, e.shiftKey); return; }
    // In 2-click mode a press inside the selected box is a corner, not a move (Alt-drag still moves).
    const selHit = s && !click2 && hits.find(x => x.t === s.t && x.i === s.i);
    if (tool === 'point' && !selHit) {
      if (!editable() || a.x < 0 || a.y < 0 || a.x > W || a.y > H) return;
      pushHistory('point');
      const b = [activeCls, a.x / W, a.y / H, opts.pointW, opts.pointH];
      clampCentre(b); boxes.push(b); changed(); setSel(opts.deselect ? null : { t: 'box', i: boxes.length - 1 });
      return;
    }
    if (tool === 'select' || tool === 'point' || (tool === 'box' && selHit) || (tool === 'box' && e.altKey && hits.length)) {
      if (!hits.length) { if (sel) setSel(null); return; }
      const target = selHit ? { t: s.t, i: s.i } : { t: hits[0].t, i: hits[0].i };
      if (!selHit) setSel(target);
      drag = { kind: 'move', target, a, orig: itemBox(target).slice(), started: false, sx: p.x, sy: p.y, hits, wasSelected: !!selHit };
      return;
    }
    if (tool === 'box') { drag = { kind: 'draw', a, started: false, sx: p.x, sy: p.y, hits }; overlayOnly(); }
  });
  window.addEventListener('mousemove', onMove);
  function onMove(e) {
    if (!img || destroyed) return;
    const p = local(e), last = mouse;
    mouse = { ...p, inside: mouse.inside };
    if (!drag) {
      if (!mouse.inside) return;
      setCursor(cursorFor(p.x, p.y, e.altKey));
      if (!loading) {
        const hits = tool === 'select' || tool === 'box' ? hitsAt(p.x, p.y) : [];
        const nh = hits.length ? { t: hits[0].t, i: hits[0].i } : null;
        if ((nh && (!hov || nh.t !== hov.t || nh.i !== hov.i)) || (!nh && hov)) { hov = nh; overlayOnly(); }
      }
      if (corner || (opts.crosshair && (tool === 'box' || tool === 'point' || tool === 'sam'))) overlayOnly();
      else if (opts.loupe || tool === 'point') overlayOnly();
      return;
    }
    if (spaceDown && drag.kind !== 'pan') {             // Space mid-drag pans; anchors stay in image space
      setView(scale, tx + (p.x - last.x), ty + (p.y - last.y)); return;
    }
    if (drag.kind === 'pan') { setView(scale, drag.tx0 + (p.x - drag.x0), drag.ty0 + (p.y - drag.y0)); return; }
    const dist = Math.hypot(p.x - drag.sx, p.y - drag.sy);
    if (drag.kind === 'draw') { if (!drag.started && dist >= DRAG_PX) drag.started = true; overlayOnly(); return; }
    if (!drag.started) {
      if (dist < (drag.kind === 'move' ? 3 : 2) || !editable()) return;
      if (!startEdit(drag)) { drag = null; return; }
    }
    const b = boxes[drag.boxIndex]; if (!b) return;
    const c = toImg(p.x, p.y), o = drag.orig;
    if (drag.kind === 'move') {
      if (b.kind) return;
      b[1] = Math.max(o[3] / 2, Math.min(1 - o[3] / 2, o[1] + (c.x - drag.a.x) / W));
      b[2] = Math.max(o[4] / 2, Math.min(1 - o[4] / 2, o[2] + (c.y - drag.a.y) / H));
    } else {
      let x0 = (o[1] - o[3] / 2) * W, y0 = (o[2] - o[4] / 2) * H, x1 = (o[1] + o[3] / 2) * W, y1 = (o[2] + o[4] / 2) * H;
      const k = drag.hh.corner || drag.hh.edge, dx = c.x - drag.a.x, dy = c.y - drag.a.y;   // grab offset kept: the edge never jumps to the pointer
      if (k.includes('w')) x0 += dx; if (k.includes('e')) x1 += dx; if (k.includes('n')) y0 += dy; if (k.includes('s')) y1 += dy;
      if (x0 > x1) [x0, x1] = [x1, x0]; if (y0 > y1) [y0, y1] = [y1, y0];
      x0 = Math.max(0, x0); y0 = Math.max(0, y0); x1 = Math.min(W, x1); y1 = Math.min(H, y1);
      const mw = Math.min(W, MIN_PX), mh = Math.min(H, MIN_PX);
      if (x1 - x0 < mw) { if (x0 + mw <= W) x1 = x0 + mw; else x0 = x1 - mw; }
      if (y1 - y0 < mh) { if (y0 + mh <= H) y1 = y0 + mh; else y0 = y1 - mh; }
      b[1] = (x0 + x1) / 2 / W; b[2] = (y0 + y1) / 2 / H; b[3] = (x1 - x0) / W; b[4] = (y1 - y0) / H;
    }
    overlayOnly(); scheduleRowUpdate(drag.boxIndex);
  }
  /** First real movement of a move/resize: one history entry, and a proposal being edited becomes a box. */
  function startEdit(d) {
    pushHistory(d.kind === 'move' ? 'move' : 'resize');
    let bi = d.target.i;
    if (d.target.t === 'prop') {
      bi = beginEditProp(d.target.i);
      if (bi < 0) { undo.pop(); return false; }
      setSel({ t: 'box', i: bi }, { scroll: false });
      d.orig = boxes[bi].slice();
      recalcProps(); renderProps(); invalidate();
    } else invalidate();    // the selected box leaves the cache for the overlay layer
    if (boxes[bi].kind) {
      restore(undo.pop()); recalcProps(); renderList(); renderProps(); invalidate();
      note('Segmentation / pose rows keep their shape here — change the class or delete instead', 'warn'); return false;
    }
    d.boxIndex = bi; d.started = true;
    return true;
  }
  window.addEventListener('mouseup', onUp);
  function onUp(e) { if (!drag || destroyed) return; finishDrag(local(e)); }
  function finishDrag(p) {
    const d = drag; drag = null;
    if (!d) return;
    if (d.kind === 'pan') { setCursor(cursorFor(p.x, p.y)); return; }
    if (d.kind === 'draw') {
      if (!d.started) {
        if (tool === 'box' && opts.boxMode === 'click2') {
          // First corner of a 2-click box (clamped to the image, so a click in the margin snaps to the edge).
          if (!editable()) return;
          corner = { x: Math.max(0, Math.min(W, d.a.x)), y: Math.max(0, Math.min(H, d.a.y)) };
          if (sel) setSel(null);
          updateHud(); overlayOnly(); return;
        }
        // A click with the Box tool on a box selects it; on empty canvas it clears the selection.
        if (d.hits && d.hits.length) setSel({ t: d.hits[0].t, i: d.hits[0].i }); else if (sel) setSel(null);
        overlayOnly(); return;
      }
      if (!editable()) { overlayOnly(); return; }
      const nb = boxFromCorners(d.a, toImg(p.x, p.y), W, H);
      // A rejected second corner keeps the first one: click again further away (or zoom in first).
      if (!nb) { note(d.two ? 'Box too thin — click the opposite corner further away, or Esc to start over' : 'Box too thin — zoom in (Z or =) to draw it', 'warn'); overlayOnly(); return; }
      corner = null;
      pushHistory('draw');
      boxes.push([activeCls, ...nb]);
      noteRecent(activeCls);
      changed();
      setSel(opts.deselect ? null : { t: 'box', i: boxes.length - 1 });
      return;
    }
    if (!d.started) {
      // A click on an already-selected item that has others under it selects the next one down the stack.
      if (d.wasSelected && d.hits && d.hits.length > 1) {
        const k = d.hits.findIndex(x => x.t === d.target.t && x.i === d.target.i);
        const n = d.hits[(k + 1) % d.hits.length]; setSel({ t: n.t, i: n.i });
      }
      return;
    }
    const b = boxes[d.boxIndex];
    if (b && d.target.t === 'box' && sameGeom(b.slice(0, 5), d.orig.slice(0, 5))) {
      undo.pop(); invalidate(); renderList(); return;       // moved and put back exactly: not an edit
    }
    changed();
  }
  cOv.addEventListener('mouseenter', () => { mouse.inside = true; overlayOnly(); });
  cOv.addEventListener('mouseleave', () => { mouse.inside = false; if (hov) hov = null; overlayOnly(); });
  cOv.addEventListener('wheel', (e) => {
    e.preventDefault(); if (!img) return;
    const p = local(e);
    const unit = e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? stage.clientHeight : 1;
    let dx = e.deltaX * unit, dy = e.deltaY * unit;
    if (e.shiftKey && !dx) { dx = dy; dy = 0; }
    if (tool === 'point' && e.altKey) {
      const d = wheelStep(dx, dy); if (!d) return;
      if (e.shiftKey) opts.pointH = Math.max(0.02, Math.min(0.9, opts.pointH + d)); else opts.pointW = Math.max(0.02, Math.min(0.6, opts.pointW + d));
      renderPointOpts(); overlayOnly(); return;
    }
    // Pinch on a trackpad arrives as ctrl+wheel. A physical wheel moves in coarse vertical notches; a trackpad
    // scrolls in small steps, often with a horizontal part — those pan instead of zooming.
    const trackpad = !!e.deltaX || (e.wheelDeltaY ? e.wheelDeltaY === -3 * e.deltaY : e.deltaMode === 0 && Math.abs(e.deltaY) < 50);
    const notch = !trackpad;
    const mode = e.ctrlKey || e.metaKey ? 'zoom' : opts.wheel === 'auto' ? (notch ? 'zoom' : 'pan') : opts.wheel;
    if (mode === 'zoom') zoomAt(p.x, p.y, Math.exp(-dy * (e.ctrlKey && !notch ? 0.01 : 0.0022)));
    else setView(scale, tx - dx, ty - dy);
  }, { passive: false });
  cOv.addEventListener('contextmenu', (e) => {
    e.preventDefault(); if (loading || !img) return;
    const p = local(e); const hits = hitsAt(p.x, p.y); if (!hits.length) return;
    const s = selectedItem(); const t = s && hits.find(x => x.t === s.t && x.i === s.i) ? s : { t: hits[0].t, i: hits[0].i };
    setSel(t);
    if (t.t === 'prop') {
      const pr = props[t.i];
      openPop({ x: e.clientX, y: e.clientY, title: `${KIND_LABEL[pr.kind] || 'Proposal'}${pr.conf != null ? ' · ' + pr.conf.toFixed(2) : ''}`, actions: [
        { label: 'Accept', icon: 'check', kbd: 'Enter', onClick: () => acceptProp(pr), disabled: !editable() },
        { label: 'Reject', icon: 'x', kbd: 'Del', onClick: () => rejectProp(pr), disabled: !editable() },
        { label: 'Zoom to it', icon: 'zoomIn', kbd: 'Z', onClick: () => zoomToBox() },
      ], pick: { current: pr.b[0], placeholder: 'Accept as class…', onPick: (c) => applyClass(c) } });
      return;
    }
    openPop({ x: e.clientX, y: e.clientY, title: `Box #${t.i + 1}`, actions: [
      { label: 'Delete', icon: 'trash', kbd: 'Del', danger: true, onClick: () => deleteSelected(), disabled: !editable() },
      { label: 'Duplicate', icon: 'copy', kbd: `${MOD}D`, onClick: () => duplicateSelected(), disabled: !editable() },
      { label: 'Refine with SAM', icon: 'target', kbd: 'R', onClick: () => refineSelected(), disabled: !editable() },
      { label: 'Zoom to box', icon: 'zoomIn', kbd: 'Z', onClick: () => zoomToBox() },
    ], pick: { current: boxes[t.i][0], placeholder: 'Change class…', onPick: (c) => applyClass(c) } });
  });
  cOv.addEventListener('dblclick', (e) => { const p = local(e); if (!hitsAt(p.x, p.y).length) { fit(); invalidate(true); } });
  const ro = new ResizeObserver(() => resizeCanvas()); ro.observe(stage);
  let dprMq = null;
  const onDpr = () => { resizeCanvas(); watchDpr(); };
  function watchDpr() { if (dprMq) dprMq.removeEventListener('change', onDpr); dprMq = matchMedia(`(resolution: ${window.devicePixelRatio || 1}dppx)`); dprMq.addEventListener('change', onDpr, { once: true }); }
  watchDpr();
  const onBlur = () => { if (spaceDown) { spaceDown = false; setCursor(cursorFor(mouse.x, mouse.y)); } if (drag) finishDrag(mouse); if (peek) { hiddenAll = peek.prev; peek = null; updateHud(); invalidate(); } };
  window.addEventListener('blur', onBlur);

  // ---------------------------------------------------------------- history
  function snapshot(op, key) {
    return { op, key, t: performance.now(), boxes: boxes.map(copyBox), props: props.map(p => ({ ...p, b: p.b.slice(), points: p.points ? p.points.map(x => x.slice()) : null, refIdx: p.refBox ? boxes.indexOf(p.refBox) : -1 })), sel: sel ? { ...sel } : null };
  }
  const pkey = (p) => (p.id != null ? 's' + p.id : 'l' + p.uid);
  function trimHistory() {
    let total = undo.reduce((s, e) => s + e.boxes.length + 1, 0);
    while (undo.length > HIST_CAP || (undo.length > 10 && total > HIST_BOX_BUDGET)) { total -= undo[0].boxes.length + 1; undo.shift(); }
  }
  /** One undo step. With `key`, repeats of the same small edit (nudges, a digit pair) within 800 ms share it. */
  function pushHistory(op, key = null) {
    const now = performance.now(), top = undo[undo.length - 1];
    if (key && top && top.key === key && now - top.t < COALESCE_MS) { top.t = now; return false; }
    undo.push(snapshot(op, key)); redo = []; trimHistory();
    return true;
  }
  function restore(s) {
    const cur = new Map(props.map(p => [pkey(p), p]));
    boxes = s.boxes.map(copyBox);
    const restored = s.props.map(x => { const p = { ...x, b: x.b.slice(), refBox: x.refIdx >= 0 ? boxes[x.refIdx] : null }; delete p.refIdx; return p; });
    const keys = new Set(restored.map(pkey));
    // Server proposals whose decision this undo reverses are told so (pending again, or the earlier verdict).
    for (const p of restored) { const c = cur.get(pkey(p)); if (!p.local && p.id != null && (!c || c.state !== p.state)) queueDecision(p, p.state); }
    props = restored.concat(props.filter(p => !keys.has(pkey(p))));
    sel = s.sel && (s.sel.t === 'box' ? boxes[s.sel.i] : props[s.sel.i]) ? s.sel : null;
    grid = null;
  }
  function doUndo() { if (!undo.length || !editable()) return; const s = undo.pop(); redo.push(snapshot(s.op)); restore(s); changed(); note(`Undid ${s.op}`); }
  function doRedo() { if (!redo.length || !editable()) return; const s = redo.pop(); undo.push(snapshot(s.op)); restore(s); changed(); note(`Redid ${s.op}`); }
  const histKey = (id) => `${ds.id}:${id}`;
  function stashHistory() {
    if (!item) return;
    const k = histKey(item.id);
    histByImage.delete(k);
    if (undo.length || redo.length) {
      histByImage.set(k, { undo, redo, version });
      while (histByImage.size > 50) histByImage.delete(histByImage.keys().next().value);
    }
  }
  function takeHistory(id, ver) {
    const e = histByImage.get(histKey(id)); histByImage.delete(histKey(id));
    if (e && e.version === ver) { undo = e.undo; redo = e.redo; } else { undo = []; redo = []; }
  }

  // ---------------------------------------------------------------- edit operations
  function editable() { return !!item && !loading && !destroyed && !!img && !labelsError; }
  function changed() {
    if (!editable()) return;          // backstop: nothing may be queued against an image that is loading
    grid = null;
    clearClassBuf();                  // a draw, accept, paste or undo ends a digit pair: the next digit starts over
    touched = true;
    dirty = true; setSaveState('dirty');
    recalcProps(); renderList(); renderProps(); updateHud(); invalidate(); filmTouch();
    // An undo / redo restores `sel` without setSel: the class list's highlight and its "applies to box #n" follow here.
    markActiveClass(); updateClassHead();
    if (flagFocus) renderFlagInfo();      // "n boxes here" and what the other image has that this one lacks
    clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 500);
  }
  function clampCentre(b) { b[3] = Math.min(1, b[3]); b[4] = Math.min(1, b[4]); b[1] = Math.max(b[3] / 2, Math.min(1 - b[3] / 2, b[1])); b[2] = Math.max(b[4] / 2, Math.min(1 - b[4] / 2, b[2])); return b; }
  /** Class for the selected box, or accept-as-class for a selected proposal; a digit pair within 800 ms is one step.
      The class for new boxes (activeCls) changes only with nothing selected. */
  function applyClass(c, key = null) {
    if (key !== 'digit') clearClassBuf();
    const s = selectedItem();
    noteRecent(c);
    if (!s || !editable()) { setPalette(c); return; }
    if (s.t === 'prop') {
      const p = props[s.i]; if (!isPending(p)) return;
      pushHistory('accept proposal'); countDecision(c === p.b[0] ? 'accepted' : 'edited');
      let bi;
      if (p.kind === 'swap' || p.kind === 'spurious') { p.merged = null; const ref = resolveRef(p); bi = ref ? boxes.indexOf(ref) : -1; p.state = p.kind === 'swap' && c === p.b[0] ? 'accepted' : 'rejected'; queueDecision(p, p.state); }
      else bi = materialize(p);
      if (bi >= 0) boxes[bi][0] = c;
      changed(); setSel(bi >= 0 ? { t: 'box', i: bi } : null);
      if (p.merged) noteAccepted(p, bi); else if (bi >= 0 && addsBox(p)) noteOverlap(bi);
      return;
    }
    const b = boxes[s.i];
    if (b[0] !== c) { pushHistory('class', key ? `${key}:${s.i}` : null); b[0] = c; changed(); }
    markActiveClass(); updateHud(); invalidate();
  }
  function setPalette(i) { activeCls = i; markActiveClass(); updateHud(); overlayOnly(); }
  /** A class chosen with the mouse, the class search or [ ]: never the second half of a digit pair. */
  function pickClass(i) { clearClassBuf(); if (selectedItem()) applyClass(i); else setPalette(i); }
  function deleteSelected() {
    const s = selectedItem(); if (!s || !editable()) return;
    if (s.t === 'prop') { rejectProp(props[s.i]); return; }
    pushHistory('delete'); boxes.splice(s.i, 1); sel = null; changed(); setSel(null);
  }
  async function deleteAll() {
    if (!boxes.length || !editable()) return;
    const name = item ? item.rel_path.split('/').pop() : 'this image';
    if (!await ui.confirm({ title: 'Delete all boxes?', message: `Remove all ${boxes.length} boxes from ${name}? ${MOD}Z undoes it.`, okLabel: 'Delete all', danger: true })) return;
    if (!editable()) return;
    pushHistory('delete all'); boxes = []; sel = null; changed(); setSel(null);
  }
  function nudge(dirKey, big) {
    const s = selectedItem(); if (!s || !editable()) return;
    let bi = s.i;
    if (s.t === 'prop') { pushHistory('accept proposal'); bi = beginEditProp(s.i); if (bi < 0) { undo.pop(); return; } setSel({ t: 'box', i: bi }); }
    const b = boxes[bi]; if (b.kind) { note('Segmentation / pose rows keep their shape', 'warn'); return; }
    const st = big ? 10 : 1;
    const dx = dirKey === 'ArrowLeft' ? -st : dirKey === 'ArrowRight' ? st : 0, dy = dirKey === 'ArrowUp' ? -st : dirKey === 'ArrowDown' ? st : 0;
    const before = b.slice(0, 5);
    if (focusEdge) {
      let x0 = (b[1] - b[3] / 2) * W, x1 = (b[1] + b[3] / 2) * W, y0 = (b[2] - b[4] / 2) * H, y1 = (b[2] + b[4] / 2) * H;
      if (focusEdge === 'left') x0 = Math.max(0, Math.min(x1 - MIN_PX, x0 + dx)); else if (focusEdge === 'right') x1 = Math.min(W, Math.max(x0 + MIN_PX, x1 + dx));
      else if (focusEdge === 'top') y0 = Math.max(0, Math.min(y1 - MIN_PX, y0 + dy)); else y1 = Math.min(H, Math.max(y0 + MIN_PX, y1 + dy));
      b[1] = (x0 + x1) / 2 / W; b[2] = (y0 + y1) / 2 / H; b[3] = (x1 - x0) / W; b[4] = (y1 - y0) / H;
    } else {
      // Moving into the edge stops the box there; it never shrinks it.
      b[1] = Math.max(b[3] / 2, Math.min(1 - b[3] / 2, b[1] + dx / W)); b[2] = Math.max(b[4] / 2, Math.min(1 - b[4] / 2, b[2] + dy / H));
    }
    if (sameGeom(before, b.slice(0, 5))) return;
    const after = b.slice(0, 5); b.splice(0, 5, ...before);
    pushHistory(focusEdge ? 'edge move' : 'nudge', `${focusEdge ? 'edge' : 'nudge'}:${bi}`);
    b.splice(0, 5, ...after);
    changed();
  }
  function duplicateSelected() {
    const s = selectedItem(); if (!s || s.t !== 'box' || !editable()) return;
    const b = boxes[s.i], off = 10 / scale;
    const nb = [b[0], b[1] + off / W, b[2] + off / H, b[3], b[4]]; clampCentre(nb);
    pushHistory('duplicate'); boxes.push(nb); changed(); setSel({ t: 'box', i: boxes.length - 1 });
  }
  function copyBoxes() {
    const s = selectedItem();
    const list = s && s.t === 'box' ? [boxes[s.i]] : boxes;
    if (!list.length) { note('Nothing to copy'); return; }
    clipboard = { boxes: list.map(b => b.slice(0, 5)), from: item ? item.rel_path.split('/').pop() : '' };
    note(`Copied ${list.length} box${list.length === 1 ? '' : 'es'} — ${MOD}V pastes into any image`);
  }
  function pasteBoxes() {
    if (!clipboard || !editable()) { if (!clipboard) note(`Nothing copied yet — select a box and press ${MOD}C`); return; }
    const add = clipboard.boxes.filter(b => !boxes.some(x => x[0] === b[0] && iou(x, b) >= PASTE_IOU));
    if (!add.length) { note('Those boxes are already here'); return; }
    pushHistory('paste'); add.forEach(b => boxes.push(b.slice())); changed();
    setSel(add.length === 1 ? { t: 'box', i: boxes.length - 1 } : null);
    note(`Pasted ${add.length}` + (clipboard.boxes.length - add.length ? ` · ${clipboard.boxes.length - add.length} already present` : ''));
  }

  // ---------------------------------------------------------------- save
  const labelUrl = (id) => `/api/datasets/${ds.id}/images/${id}/labels`;
  /** An image without a label file stays without one when its boxes go back to none (draw then undo, or delete
      them all): the server removes the file this session created, or writes nothing. Only an image that had a
      file keeps an empty one — a deliberate background frame ("Mark as background" makes one on purpose). */
  const emptyMeansNoFile = (it) => !it.has_label || createdFiles.has(histKey(it.id));
  function labelBody(it, rows, base) {
    const body = { boxes: rows };
    if (base !== undefined) body.base_version = base;
    if (!rows.length && emptyMeansNoFile(it)) body.remove_if_empty = true;
    return body;
  }
  function noteFileWrite(it, hadFile, n, res) {
    const k = histKey(it.id), has = !!(res.item && res.item.has_label);
    if (!has) createdFiles.delete(k);
    else if ((!hadFile && n) || createdFiles.has(k)) { createdFiles.set(k, res.version ?? null); while (createdFiles.size > 500) createdFiles.delete(createdFiles.keys().next().value); }
  }
  const labelRows = (res) => (res.boxes || []).map((b, i) => {
    const c = res.confs && res.confs[i];
    const row = c != null ? [...b.slice(0, 5), c] : [...b.slice(0, 5)];   // keep a prediction file's confidence column
    const kind = res.row_kinds && res.row_kinds[i]; if (kind && kind !== 'box') row.kind = kind;
    const iss = res.row_issues && res.row_issues[i]; if (iss && iss.length) row.issues = iss;
    const at = res.attrs && res.attrs[i]; if (at) row.attrs = at;
    return row;
  });
  /**
   * Persist `boxes` for the current image. Resolves true when what is on disk now matches what the user
   * sees, false when it does not (failure, or a conflict the user has not resolved). Never throws.
   */
  async function save(force = false) {
    if (!item || loading) return !dirty;
    if (!dirty && !force) return true;
    if (saving) return false;
    if (conflictOpen || conflictPending) { lastOutcome = 'conflict'; return false; }
    const target = item, payload = boxes.map(b => [...b]), base = version, hadFile = !!target.has_label;
    saving = true;
    clearTimeout(savingNote); savingNote = setTimeout(() => { if (saving && target === item) setSaveState('saving'); }, NOTICE_MS);
    try {
      const res = await fovea.api.put(labelUrl(target.id), labelBody(target, payload, base));
      saveRetries = 0; lastOutcome = 'ok';
      noteFileWrite(target, hadFile, payload.length, res);
      forgetLabels(target.id);            // C / ⇧C / the ghost on the next frame read this file again
      const after = countByClass(res.item && res.item.boxes ? res.item.boxes : payload);
      if (target === item) { bumpCounts(savedCounts, after); savedCounts = after; }
      if (target === item) {
        version = res.version ?? null;
        const editedMeanwhile = JSON.stringify(payload) !== JSON.stringify(boxes.map(b => [...b]));
        const reconcile = res.rejected && !editedMeanwhile;
        if (reconcile) {
          boxes = (res.item.boxes || []).map(b => [...b]); sel = null; grid = null;
          ui.toast(`${res.rejected} invalid box${res.rejected > 1 ? 'es were' : ' was'} not saved`, { type: 'error' });
        }
        baseBoxes = (res.item.boxes || payload).map(b => b.slice(0, 5));
        dirty = editedMeanwhile;
        Object.assign(item, res.item); delete item.boxes;     // keep identity: navigation staleness checks rely on it
        setSaveState(dirty ? 'dirty' : 'saved');
        if (reconcile) { recalcProps(); renderList(); renderProps(); invalidate(); }
        renderInfo(); renderEmptyState(boxes.length);
      }
      { const { boxes: _omit, ...row } = res.item; cursor.update(row); }
      fovea.bus.emit('labels:changed', { item: res.item });
      scheduleFlagCheck(target.id);
      if (!destroyed && (res.item.classes || []).some(c => c >= names.length)) {
        try { const d = await refresh(); names = [...(d.classes || [])]; renderClasses(); } catch (_) { /* palette refresh is cosmetic */ }
      }
      if (dirty && !destroyed && target === item) { clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), 300); }
      return true;
    } catch (e) {
      if (e.status === 409) {
        lastOutcome = 'conflict';
        if (target !== item) {
          ui.toast(`Edits to ${target.rel_path.split('/').pop()} were not saved: the file was changed elsewhere`, { type: 'error', timeout: 8000 });
          return false;
        }
        saving = false;
        const choice = await resolveConflict(e);
        if (choice === 'mine' || choice === 'merge') return await save(true);
        if (choice === 'theirs') { lastOutcome = 'ok'; return true; }
        return false;
      }
      lastOutcome = 'error';
      saveRetries++;
      const giveUp = saveRetries >= 5 || destroyed || target !== item || (e.status >= 400 && e.status < 500);
      if (target === item) setSaveState('error', giveUp ? 'Save failed — not written' : 'Save failed — retrying');
      if (Date.now() - lastFailToast > 8000) { lastFailToast = Date.now(); ui.toast('Save failed: ' + (e.message || 'network error'), { type: 'error' }); }
      if (!giveUp) { clearTimeout(saveTimer); saveTimer = setTimeout(() => save(), Math.min(20000, 1500 * 2 ** saveRetries)); }
      return false;
    } finally { saving = false; clearTimeout(savingNote); }
  }
  let savingNote = null, loadingNote = null;
  // The QA engine re-checks the label-based flags (sequence gaps, twin mismatches, box shapes) about 3 s after a
  // save, without a job. Read the saved images' flags once that has run so the chips and the strip update in place.
  const flagIds = new Set();
  let flagTimer = null, flagTries = 0;
  function scheduleFlagCheck(id) { flagIds.add(id); flagTries = 0; clearTimeout(flagTimer); flagTimer = setTimeout(checkFlags, 3600); }
  async function checkFlags() {
    if (destroyed) return;
    const ids = [...flagIds].slice(-20);
    flagIds.clear();
    for (const id of ids) {
      let r;
      try { r = await fovea.api.get(`/api/datasets/${ds.id}/images/${id}/qa`); } catch (_) { continue; }
      if (destroyed) return;
      if (r.refreshing && flagTries < 6) { flagIds.add(id); continue; }        // the re-check has not run yet
      const flags = (r.flags || []).map(f => f.code);
      cursor.update({ id, flags }); filmSetRow(id, { flags });
      if (item && item.id === id && (item.flags || []).join() !== flags.join()) { item.flags = flags; renderInfo(); }
      if (item && item.id === id && flagFocus && flagFocus.forId === id) refreshFlagFocus();   // its detail changed with the labels
    }
    if (flagIds.size) { flagTries++; clearTimeout(flagTimer); flagTimer = setTimeout(checkFlags, 1500); }
    filmRefresh();          // a gap or a twin mismatch also changes on the other frames of the sequence
  }

  /** Three-way merge of my edits and the file's new content, against the version this editor loaded. */
  function match(a, b) {
    const used = new Set(), pairs = new Map();
    a.forEach((x, i) => {
      let best = -1, bv = 0.5;
      b.forEach((y, j) => { if (used.has(j)) return; const v = iou(x, y) + (x[0] === y[0] ? 0.01 : 0); if (v > bv) { bv = v; best = j; } });
      if (best >= 0) { used.add(best); pairs.set(i, best); }
    });
    return pairs;
  }
  function merge3(base, mine, theirs) {
    const bm = match(base, mine), bt = match(base, theirs);
    const out = theirs.map(b => b.slice()), drop = new Set();
    let mineAdded = 0, mineChanged = 0, mineDeleted = 0;
    base.forEach((b, i) => {
      const mi = bm.get(i), ti = bt.get(i);
      if (mi == null) { if (ti != null && sameGeom(theirs[ti].slice(0, 5), b.slice(0, 5))) { drop.add(ti); mineDeleted++; } return; }
      const m = mine[mi];
      if (!sameGeom(m.slice(0, 5), b.slice(0, 5)) && ti != null) { out[ti] = m.slice(); mineChanged++; }
    });
    const matchedMine = new Set(bm.values());
    mine.forEach((m, i) => {
      if (matchedMine.has(i)) return;
      if (theirs.some(t => t[0] === m[0] && iou(t, m) >= DUP_IOU)) return;
      out.push(m.slice()); mineAdded++;
    });
    return { boxes: out.filter((_, j) => !drop.has(j)), mineAdded, mineChanged, mineDeleted };
  }
  /** The file changed under this editor. Show what differs and let the user merge, keep theirs or overwrite. */
  function resolveConflict(err) {
    if (conflictOpen) return Promise.resolve(null);
    conflictOpen = true; conflictPending = null;
    const d = (err.body && err.body.detail) || {};
    const theirs = Array.isArray(d.boxes) ? d.boxes.map(b => b.slice()) : [];
    const mine = boxes.map(b => [...b]);
    const m = merge3(baseBoxes, mine, theirs);
    const theirAdded = theirs.filter(t => !baseBoxes.some(b => sameGeom(b.slice(0, 5), t.slice(0, 5)))).length;
    const theirRemoved = baseBoxes.filter(b => !theirs.some(t => sameGeom(b.slice(0, 5), t.slice(0, 5)))).length;
    theirsOverlay = theirs; overlayOnly();
    setSaveState('error', 'Conflict — not saved');
    return new Promise((resolve) => {
      let decided = null;
      ui.modal({
        title: 'Labels changed elsewhere',
        body: h('div', { class: 'col gap-8' },
          h('p', { style: 'margin:0' }, `${item ? item.rel_path.split('/').pop() : 'This image'} was modified outside this editor — in another tab, by a bulk operation or by another tool. The file's boxes are outlined in amber on the image.`),
          h('div', { class: 'ann-conflict small' },
            h('div', null, h('b', 'On disk now: '), `${theirs.length} box${theirs.length === 1 ? '' : 'es'} (${theirAdded} added, ${theirRemoved} removed or changed since you opened it)`),
            h('div', null, h('b', 'Your version: '), `${mine.length} box${mine.length === 1 ? '' : 'es'}`),
            h('div', null, h('b', 'Merged: '), `${m.boxes.length} — their changes plus your ${m.mineAdded} new, ${m.mineChanged} changed, ${m.mineDeleted} deleted`))),
        onClose: () => { conflictOpen = false; theirsOverlay = null; overlayOnly(); if (!decided) { conflictPending = d; setSaveState('error', 'Conflict — choose how to resolve'); renderConflictBtn(); } resolve(decided); },
        footer: (api) => [
          h('button', { class: 'btn', onClick: () => { decided = 'theirs'; version = d.version ?? null; pushHistory('load file'); boxes = theirs.map(b => [...b]); baseBoxes = theirs.map(b => b.slice(0, 5)); sel = null; dirty = false; grid = null; setSaveState('saved'); recalcProps(); renderList(); renderProps(); invalidate(); api.close(); } }, 'Use the file on disk'),
          h('button', { class: 'btn', onClick: () => { decided = 'mine'; version = d.version ?? null; baseBoxes = theirs.map(b => b.slice(0, 5)); dirty = true; api.close(); } }, 'Overwrite with mine'),
          h('button', { class: 'btn btn-primary', onClick: () => { decided = 'merge'; version = d.version ?? null; pushHistory('merge'); boxes = m.boxes; baseBoxes = theirs.map(b => b.slice(0, 5)); sel = null; grid = null; dirty = true; recalcProps(); renderList(); renderProps(); invalidate(); api.close(); } }, 'Merge both'),
        ],
      });
    });
  }
  function renderConflictBtn() {
    if (!conflictPending) return;
    saveState.replaceChildren(icon('alertCircle', 12), document.createTextNode('Conflict — '), h('button', { class: 'btn btn-ghost btn-sm', onClick: () => { const d = conflictPending; conflictPending = null; resolveConflict({ body: { detail: d } }).then(c => { if (c === 'mine' || c === 'merge') save(true); }); } }, 'Resolve…'));
  }

  const sleep = (ms) => new Promise(r => setTimeout(r, ms));
  /** Save until clean: the label file, then the proposal decisions. Resolves 'clean', 'conflict' (a changed-elsewhere
      dialog is undecided) or 'failed' — also when the decisions could not be recorded (sent twice, 600 ms apart; once
      for a navigation, which goes on with them queued): the AI panel must not propose over rejections the server
      has not heard of. `dirty` tells the two failures apart. */
  async function flush({ nav = false } = {}) {
    clearTimeout(saveTimer);
    for (let attempt = 0; attempt < 4; attempt++) {
      while (saving) await sleep(25);
      if (!dirty) {
        if (await flushDecisions()) return 'clean';
        if (dirty || saving) continue;                 // edited meanwhile: that is saved first
        if (nav || !decisionQueue.size) return 'failed';
        await sleep(600);
        if (await flushDecisions()) return 'clean';
        if (dirty || saving) continue;
        return 'failed';
      }
      if (conflictOpen || conflictPending) return 'conflict';
      if (await save(true)) continue;          // written — loop in case of edits made during the round trip
      if (!dirty) continue;                    // nothing left to write: the decisions, at the top
      if (lastOutcome === 'conflict') return 'conflict';
      if (attempt >= 1) return 'failed';
      clearTimeout(saveTimer);                 // flush drives this one retry itself
      await sleep(600);
    }
    return dirty || !(await flushDecisions()) ? 'failed' : 'clean';
  }
  /** Why flush() did not come back clean: the boxes on this image, or proposal decisions the server has not heard of. */
  const notSavedMsg = () => (dirty || conflictPending || conflictOpen ? 'The edits on this image are not saved yet — resolve that first'
    : 'Proposal decisions are not saved yet (see the save state) — try again in a moment');
  const compactRows = (list) => list.map(b => b.map((v, k) => (k === 0 ? v : Math.round(v * 1e6) / 1e6)));
  /** Write that outlives this view. 'leave' (in-app tab switch): an ordinary request — the page stays alive.
      'unload': keepalive, whose body is capped at 64 KiB, so rows are rounded to the file's 6 decimals. */
  function finalSave(reason) {
    if (!dirty || !item || loading || conflictOpen || conflictPending) return;
    const it = item, name = it.rel_path.split('/').pop(), url = labelUrl(it.id), hadFile = !!it.has_label;
    // A save still in flight is our own and will move the file's version; otherwise stay version-checked.
    const body = labelBody(it, reason === 'unload' ? compactRows(boxes) : boxes.map(b => [...b]), saving ? undefined : version);
    const json = JSON.stringify(body);
    const keepalive = reason === 'unload' && json.length < 63000;
    prevLabels.delete(it.id);
    fetch(url, { method: 'PUT', keepalive, headers: { 'Content-Type': 'application/json' }, body: json })
      .then(async (r) => {
        if (!r.ok && reason !== 'unload') ui.toast(`Edits to ${name} were not saved (${r.status === 409 ? 'the file was changed elsewhere' : 'HTTP ' + r.status})`, { type: 'error', timeout: 8000 });
        else if (r.ok) { fovea.bus.emit('labels:changed', { item: { id: it.id } }); try { noteFileWrite(it, hadFile, body.boxes.length, await r.json()); } catch (_) { /* page going away */ } }
      })
      .catch(() => { if (reason !== 'unload') ui.toast(`Edits to ${name} were not saved`, { type: 'error', timeout: 8000 }); });
    dirty = false;
  }
  /** Decisions that outlive this view. On unload: keepalive, fire and forget. On leaving the tab (the page stays):
      a failed send is tried once more after 2 s, and only then reported — the view that could retry is gone. */
  function finalDecisions(keepalive) {
    if (!decisionQueue.size) return;
    const body = JSON.stringify(decideBody(decisionQueue)), n = decisionQueue.size;
    decisionQueue.clear();
    const send = () => fetch(`/api/datasets/${ds.id}/proposals/decide`, { method: 'POST', keepalive, headers: { 'Content-Type': 'application/json' }, body })
      .then(r => (r.ok || r.status === 404 || r.status === 405 ? 'ok' : r.status >= 400 && r.status < 500 && r.status !== 408 && r.status !== 429 ? 'drop' : 'retry'), () => 'retry');
    if (keepalive) { send(); return; }
    send().then(o => (o === 'retry' ? sleep(2000).then(send) : o)).then((o) => {
      if (o !== 'ok') ui.toast(`${n} proposal decision${n === 1 ? ' was' : 's were'} not saved — ${n === 1 ? 'it shows' : 'they show'} as pending again`, { type: 'error', timeout: 9000 });
    });
  }

  // ---------------------------------------------------------------- navigation
  const step = (d, source = 'click') => { stopAuto(); goTo((loading && want >= 0 ? want : idx) + d, { source }); };
  /**
   * Show image i. The current picture stays until the next one is decoded; then picture, boxes, proposals and every
   * panel change in one frame. The newest request wins, but a held arrow key still shows every frame that is decoded
   * on the way (never a frozen screen), and large frames are stepped through as 1024 px renditions while the key is
   * held — the original replaces the rendition, same picture, once stepping stops.
   */
  async function goTo(i, { reload = false, source = 'jump' } = {}) {
    if (destroyed) return;
    if (cursor.total != null) i = Math.max(0, Math.min(cursor.total - 1, i));
    if (i < 0) return;
    if (!reload && i === want && (loading || i === idx)) return;     // already there or on the way
    if (drag) finishDrag(mouse);                                     // a drag in progress is committed, never dropped
    closePop();
    const my = ++navToken; want = i;
    const now = performance.now(), rapid = (source === 'key' || source === 'auto') && now - lastNavAt < 260;
    lastNavAt = now;
    stopAutoIfEnd(i); clearClassBuf();
    if (!loading) dropRedundant();          // decided along with this image's other decisions, in the flush below
    if (dirty || saving || decisionQueue.size || conflictPending) {
      const flushed = await flush({ nav: true });
      if (my !== navToken) return;
      if (flushed === 'conflict') { stopAuto(); want = idx; if (conflictPending) renderConflictBtn(); return; }
      // Only unsaved boxes hold the user here: proposal decisions that did not get through stay queued and are
      // retried (the save state says so; this image shows them as made when it is opened again).
      if (flushed === 'failed' && dirty) {
        stopAuto();
        const name = item ? item.rel_path.split('/').pop() : 'this image';
        const leave = await ui.confirm({ title: 'Changes not saved', message: `Your edits to ${name} could not be written. Leave this image and discard them?`, okLabel: 'Discard & continue', danger: true });
        if (!leave || my !== navToken) { if (my === navToken) want = idx; return; }
        while (saving) await sleep(25);
        if (my !== navToken) return;
        dirty = false; clearTimeout(saveTimer);
      }
    }
    // Propagate means exactly one step forward from an image whose boxes are on screen.
    const prevFrame = !reload && !loading && item && i === idx + 1 && opts.propagate && boxes.length
      ? { boxes: boxes.map(b => b.slice(0, 5)), seq: item.seq, name: item.rel_path.split('/').pop() } : null;
    if (!loading && item) stashHistory();          // a reload of the same image gets it back if the file is unchanged
    loading = true; updateBusy();
    clearTimeout(loadingNote);
    loadingNote = setTimeout(() => { if (loading && my === navToken) { setSaveState('loading', 'Loading…'); if (!img) showStageMsg('Loading…'); } }, NOTICE_MS);
    let it = null;
    try { it = await cursor.ensure(i); }
    catch (e) { if (my === navToken) { loading = false; want = idx; updateBusy(); if (img) note(e.message, 'error'); else showStageMsg(e.message, true); restoreSaveState(); } return; }
    if (my !== navToken) return;
    if (!it) { loading = false; want = idx; updateBusy(); restoreSaveState(); return; }
    const big = (it.width || 0) * (it.height || 0) > 4.2e6;
    const fullUrl = fovea.api.imgUrl(it.id);
    const useRend = big && rapid && !frames.ready(fullUrl);
    const url = useRend ? fovea.api.thumbUrl(it.id, 1024, it.mtime) : fullUrl;
    const ac = new AbortController();
    inflight.push({ my, ac, url });
    while (inflight.length > 3) { const old = inflight.shift(); old.ac.abort(); frames.cancel(old.url); }
    const [labels, pr, im] = await Promise.all([
      fovea.api.get(labelUrl(it.id), null, { signal: ac.signal }).catch((e) => ({ error: e })),
      loadProposals(it.id, ac.signal),
      frames.get(url, { priority: 'high' }).catch(() => null),
      filmShown() && it.seq ? loadFilm(it.seq).then(d => filmWarm(d, it.id)).catch(() => null) : null,   // the strip changes with the picture
    ]);
    const k = inflight.findIndex(x => x.my === my); if (k >= 0) inflight.splice(k, 1);
    if (destroyed) return;
    if (my !== navToken) {
      // Superseded. A frame between the one on screen and the destination is still shown (view only).
      if (!labels.error && im && between(i)) present(it, i, labels, pr, im, { transit: true, rend: useRend });
      return;
    }
    clearTimeout(loadingNote);
    present(it, i, labels, pr, im, { transit: false, rend: useRend, prevFrame, reload });
  }
  const inflight = [];
  let shownIdx = -1;
  function between(i) { return want !== shownIdx && (want > shownIdx ? i > shownIdx && i < want : i < shownIdx && i > want); }
  async function loadProposals(id, signal) {
    if (!propsApi) return { preds: [], runs: {} };
    try { return await fovea.api.get(`/api/datasets/${ds.id}/images/${id}/proposals`, null, { signal }); }
    catch (e) { if (e.status === 404 && /not found/i.test(e.message) && !/image/i.test(e.message)) propsApi = false; return { preds: [], runs: {}, error: e }; }
  }
  function present(it, i, labels, pr, im, { transit, rend, prevFrame = null, reload = false }) {
    const sameImage = item && item.id === it.id;
    filmCommit();                                  // the strip keeps what was just done on the image being left
    const keepSel = reload && sameImage && sel ? { ...sel } : null;
    item = it; idx = i; shownIdx = i; img = im; imgFull = !rend; missingFile = !im;
    if (im) {
      const nw = im.naturalWidth, nh = im.naturalHeight;
      if (rend && it.width && it.height) {
        // A rendition is drawn at the original's size; its aspect tells which way round the original is shown.
        const flip = Math.abs(nw / nh - it.height / it.width) < Math.abs(nw / nh - it.width / it.height);
        W = flip ? it.height : it.width; H = flip ? it.width : it.height;
      } else { W = nw; H = nh; }
      sizeKnown = true;
    } else if (it.width > 0 && it.height > 0) { W = it.width; H = it.height; sizeKnown = true; }   // file missing: the indexed size
    else sizeKnown = false;                        // never the size of the image shown before
    labelsError = labels.error || null;
    regions = labelsError ? [] : (labels.regions || []);
    if (labelsError) { boxes = []; version = null; baseBoxes = []; savedCounts = {}; }
    else {
      version = labels.version ?? null;
      boxes = labelRows(labels); baseBoxes = boxes.map(b => b.slice(0, 5)); savedCounts = countByClass(boxes);
      if (typeof labels.exists === 'boolean') {
        it.has_label = labels.exists;          // the file itself, not the list row read when the list was opened
        // A file this session created stays "ours" (emptied → removed) only while it is exactly what we wrote.
        const k = histKey(it.id);
        if (createdFiles.has(k) && (!labels.exists || createdFiles.get(k) !== version)) createdFiles.delete(k);
      }
    }
    runs = (pr && pr.runs) || {};
    props = withQueuedDecisions(((pr && pr.preds) || []).map(normProp), decisionQueue);
    sel = null; hov = null; focusEdge = null; grid = null; dirty = false; saveRetries = 0; zoomBack = null;
    if (!sameImage) {
      corner = null; touched = false;
      approveArmed = null; showReviewNote('');      // a "proposals still pending" warning belongs to the image it was shown on
      // So does a note that waits for Esc ("spot-check box #12", "⌘Z undoes the accept"): on the next image it
      // would point at another box. Short notes ("Skipped 3 frames…") are about the step itself and stay.
      if (note.sticky) clearNote();
    }
    if (ghost && ghost.forId !== it.id) ghost = null;
    if (flagFocus && flagFocus.forId !== it.id) flagFocus = null;
    else if (flagFocus) flagFocus.refs = flagFocus.rows.map(r => boxes[r]).filter(Boolean);   // reloaded: new box objects
    loading = transit;
    if (!transit) {
      takeHistory(it.id, version);
      if (prevFrame && !labelsError && !labels.exists && im && (prevFrame.seq || null) === (it.seq || null)) {
        // Previous frame's boxes as proposals: nothing is written until accepted, so nothing to undo and an
        // intentionally empty label file (a negative sample) is never touched.
        addLocalProps(prevFrame.boxes, 'prev_frame', 'previous frame');
        note(`${prevFrame.boxes.length} boxes from the previous frame proposed — Enter accepts one, ⇧Enter all`, 'info', true);
      }
    }
    recalcProps();
    if (keepSel && (keepSel.t === 'box' ? boxes[keepSel.i] : props[keepSel.i])) sel = keepSel;
    if (!booted && !transit) {
      booted = true;
      const row = q0.get('box');
      if (row != null && boxes[Number(row)]) sel = { t: 'box', i: Number(row) };
    }
    if (!((customView || holdFit) && prevDims && img && prevDims[0] === W && prevDims[1] === H)) fit();
    if (img) prevDims = [W, H];
    if (!img) showStageMsg('Image file not found on disk — it may have been moved or deleted. Labels are read-only until it is back.', true);
    else if (labelsError) showStageMsg(`Could not load this image's labels (${labelsError.message}). Editing is disabled — move away and back to retry.`, true);
    else showStageMsg('', false);
    markActiveClass();   // the class list shows the selected box's class (?box=), else the class for new boxes
    fovea.router.replaceQuery({ ...filters, sort: sortImplied ? null : sort, order: order !== defaultOrder(sort) ? order : null, img: it.id });
    // everything below lands in the same frame as the new picture
    renderInfo(); renderList(); renderProps(); renderSeq(); renderFilm(); updateHud(); updateClassHead(); updateBusy();
    if (labelsError) setSaveState('error', 'Labels not loaded'); else setSaveState('saved');
    invalidate(true);
    if (sel && sel.t === 'box') scrollListTo(sel.i, true);
    if (transit) return;
    cursor.prefetch(idx, 4);
    const near = [idx + 1, idx + 2, idx - 1].map(k => cursor.items[k]).filter(Boolean);
    frames.prefetch(near.map(x => (rend && (x.width || 0) * (x.height || 0) > 4.2e6 ? fovea.api.thumbUrl(x.id, 1024, x.mtime) : fovea.api.imgUrl(x.id))));
    if (filmShown()) for (const x of near.slice(0, 2)) if (x.seq && x.seq !== it.seq) loadFilm(x.seq).then(d => filmWarm(d, x.id)).catch(() => {});   // the next sequence's strip, ahead of time
    if (rend) {
      const target = it;
      frames.get(fovea.api.imgUrl(it.id), { priority: 'high' }).then((full) => {
        if (destroyed || item !== target || !full) return;
        img = full; imgFull = true; miniSrc = null; invalidate(true);
      }).catch(() => {});
    }
    if (opts.ghost) loadGhost();
    imageListeners.forEach(cb => { try { cb(item, idx); } catch (e) { console.error(e); } });
  }
  function showStageMsg(t, show = true) { stageMsg.textContent = t; stageMsg.classList.toggle('hidden', !show || !t); }
  function restoreSaveState() { if (labelsError) setSaveState('error', 'Labels not loaded'); else setSaveState(dirty ? 'dirty' : 'saved'); }
  function updateBusy() { root.classList.toggle('is-loading', loading); }

  async function nextUnlabeled() {
    if (!item) return;
    try {
      // Search the working set itself — same filters, same order — so the result is reachable and never behind.
      const res = await fovea.api.get(`/api/datasets/${ds.id}/images/neighbor`, { ...filters, sort, order, id: item.id, dir: 'next', need: 'nobox' });
      if (!res.item) { note('No image without boxes after this one in the current list'); return; }
      const pos = await cursor.positionOf(res.item.id);
      if (pos >= 0) goTo(pos); else note('That image is not in this working set — refresh the list to include it', 'warn');
    } catch (e) { note(e.message, 'error'); }
  }
  /** ⇧N: the next image in this list with a proposal to decide here. Images whose pending proposals this editor
      would all hide (under the threshold, a hidden class, or duplicates of a same-class box) are skipped and counted. */
  async function nextWithProposals() {
    if (!item) return;
    const token = navToken;
    let from = item.id, skipped = 0;
    try {
      for (let tries = 0; tries < 25; tries++) {
        const res = await fovea.api.get(`/api/datasets/${ds.id}/images/neighbor`, { ...filters, ai: 'pending', sort, order, id: from, dir: 'next' });
        if (token !== navToken || destroyed) return;
        if (!res.item) break;
        const [pr, lab] = await Promise.all([loadProposals(res.item.id), fovea.api.get(labelUrl(res.item.id)).catch(() => ({ boxes: [] }))]);
        if (token !== navToken || destroyed) return;
        if (shownPending(pr.preds, (lab.boxes || []).map(b => b.slice(0, 5)), thr, hiddenCls) > 0) {
          const pos = await cursor.positionOf(res.item.id);
          if (token !== navToken || destroyed) return;
          if (pos < 0) { note('That image is not in this working set', 'warn'); return; }
          if (skipped) note(`Skipped ${skipped} image${skipped === 1 ? '' : 's'} whose proposals are all hidden here (below ${thr.toFixed(2)} or already labeled)`);
          goTo(pos); return;
        }
        skipped++; from = res.item.id;
      }
      note(skipped ? `No proposals to decide in the next ${skipped} image${skipped === 1 ? '' : 's'} with pending ones — all hidden here (below ${thr.toFixed(2)} or already labeled). { lowers the threshold; "Reject hidden" clears them.`
        : 'No image with pending proposals after this one in the current list');
    } catch (e) { note(e.message, 'error'); }
  }
  /** ⇧A ⇧F ⇧X set the status — again on an image that already has it keeps it and moves on, as Explore's A F X do.
      ⇧U and the panel's pressed button (toggle) clear it. A ⇧A over pending proposals warns first; a second ⇧A on
      the same image within 3 s approves anyway. */
  async function review(status, { toggle = false } = {}) {
    if (!item) return;
    const target = item, cur = target.review ? target.review.status : null;
    const st = toggle && cur === status ? null : status;
    const advance = () => { if (st && opts.advance && (st === 'approved' || st === 'excluded') && target === item && !loading) goTo(idx + 1, { source: 'review' }); return st && opts.advance && (st === 'approved' || st === 'excluded'); };
    if (st === cur) {                            // already so: nothing to write
      approveArmed = null; showReviewNote('');
      if (!advance()) note(st ? `Already ${REVIEW[st].label.toLowerCase()} — ⇧U clears it` : 'Not reviewed yet');
      return;
    }
    const open = visProps.filter(isPending).length;
    const armed = !!approveArmed && approveArmed.id === target.id && Date.now() - approveArmed.at < 3000;
    if (st === 'approved' && open > 0 && !armed) {
      approveArmed = { id: target.id, at: Date.now() };
      showReviewNote(`${open} proposal${open === 1 ? '' : 's'} still pending — ⇧A again to approve anyway, or decide them first`);
      return;
    }
    approveArmed = null; showReviewNote('');
    try {
      await fovea.api.put(`/api/datasets/${ds.id}/review`, { image_ids: [target.id], status: st });
      target.review = st ? { status: st, note: (target.review && target.review.note) || '', updated_at: Date.now() / 1000 } : null;
      cursor.update({ id: target.id, review: target.review });
      filmSetRow(target.id, { review: target.review });
      if (target === item) { renderInfo(); filmTouch(); }
      fovea.bus.emit('review:changed', { ids: [target.id], status: st });
      advance();
    } catch (e) { ui.toast(e.message, { type: 'error' }); }
  }
  function showReviewNote(t) { reviewNote.textContent = t; reviewNote.classList.toggle('hidden', !t); }

  // ---------------------------------------------------------------- previous frame: ghost, C / ⇧C
  // Which frame is the previous one is cached per image; that frame's boxes are cached apart and dropped whenever
  // this editor (or a bulk write) changes its file, so the ghost never shows boxes the file no longer holds.
  // C and ⇧C, which write what they read, always re-read the file.
  const prevOf = new Map();         // image id -> { id, name } of its previous frame, or null
  const prevLabels = new Map();     // frame id -> its boxes as last read
  function forgetLabels(id) {
    prevLabels.delete(id);
    if (opts.ghost && ghost && ghost.srcId === id && item) loadGhost();
  }
  /** The previous frame of the same sequence (by file name), else the previous image in this list; with boxes. */
  async function prevFrameOf(it, i, { fresh = false } = {}) {
    let prev;
    if (prevOf.has(it.id)) prev = prevOf.get(it.id);
    else {
      let row = null;
      if (it.seq) {
        // The frame before it in file-name order (the filmstrip's order): a sequence split over train and test
        // interleaves its frames, so path order would give the previous frame of the same split instead. A copy
        // of this very frame in another folder is not its previous frame (see seqNeighbor).
        const data = await loadFilm(it.seq).catch(() => null);
        const k = data ? data.pos.get(it.id) : null;
        if (k != null) { const j = seqNeighbor(data.frames, k, -1, it.split); row = j >= 0 ? data.frames[j] : null; }
        else row = (await fovea.api.get(`/api/datasets/${ds.id}/images/neighbor`, { seq: it.seq, split: it.split || undefined, sort: 'name', id: it.id, dir: 'prev' })).item;
      } else if (i > 0) row = await cursor.ensure(i - 1);
      const base = row ? baseOf(row.rel_path) : '';
      prev = row ? { id: row.id, name: row.split && row.split !== it.split ? `${row.split}/${base}` : base } : null;
      prevOf.set(it.id, prev); if (prevOf.size > 200) prevOf.delete(prevOf.keys().next().value);
    }
    if (!prev) return null;
    let boxes = fresh ? null : prevLabels.get(prev.id);
    if (!boxes) {
      const lab = await fovea.api.get(labelUrl(prev.id));
      boxes = (lab.boxes || []).map(b => b.slice(0, 5));
      prevLabels.delete(prev.id); prevLabels.set(prev.id, boxes); if (prevLabels.size > 30) prevLabels.delete(prevLabels.keys().next().value);
    }
    return { ...prev, boxes };
  }
  async function loadGhost() {
    if (!item) return;
    const target = item;
    try { const p = await prevFrameOf(target, idx); if (item !== target) return; ghost = { forId: target.id, srcId: p ? p.id : null, boxes: p ? p.boxes : [], name: p ? p.name : null }; invalidate(); updateHud(); }
    catch (e) { ghost = null; }
  }
  function toggleGhost() {
    opts.ghost = !opts.ghost; LS.set('fovea.ann.ghost', opts.ghost ? '1' : '0');
    if (opts.ghost) loadGhost(); else invalidate();
    renderSeq(); updateHud();
  }
  async function copyPrevious(direct) {
    if (!editable()) return;
    const token = navToken, target = item;
    let src;
    try { src = await prevFrameOf(target, idx, { fresh: true }); } catch (e) { note(e.message, 'error'); return; }
    // The answer may arrive after the user moved on: it must never land on another image.
    if (navToken !== token || item !== target || !editable()) return;
    if (!src) { note('No previous frame for this image'); return; }
    if (!src.boxes.length) { note(`${src.name} has no boxes`); return; }
    const fresh = src.boxes.filter(b => !boxes.some(x => x[0] === b[0] && iou(x, b) >= DUP_IOU));
    const skipped = src.boxes.length - fresh.length;
    if (!fresh.length) { note(`All ${src.boxes.length} boxes of ${src.name} are already here`); return; }
    // Two boxes on one object (another class, or a duplicate) are never copied blindly: ⇧C leaves out a box that
    // would land on one already here or on one it copies first; C proposes everything, and accept-all skips them.
    const clashes = (list) => { const keep = [], out = []; for (const b of list) { if (boxes.some(x => iou(x, b) >= DUP_IOU) || keep.some(x => iou(x, b) >= DUP_IOU)) out.push(b); else keep.push(b); } return { keep, out }; };
    if (direct) {
      const { keep, out } = clashes(fresh);
      if (!keep.length) { note(`Nothing copied: every box of ${src.name} would land on another box here — C proposes them to decide one by one`, 'warn'); return; }
      pushHistory('copy previous'); keep.forEach(b => boxes.push(b.slice())); changed();
      note(`Copied ${keep.length} from ${src.name}` + (skipped ? ` · ${skipped} already present` : '') + (out.length ? ` · ${out.length} left out: each would land on another box (two classes on one object, or a duplicate) — C proposes them` : '') + ` — ${MOD}Z to undo`, out.length ? 'warn' : 'info', !!out.length);
      return;
    }
    const already = fresh.filter(b => props.some(p => isPending(p) && p.kind === 'prev_frame' && p.b[0] === b[0] && iou(p.b, b) >= PASTE_IOU));
    const add = fresh.filter(b => !already.includes(b));
    if (!add.length) { note('Those proposals are already shown'); return; }
    const made = addLocalProps(add, 'prev_frame', 'previous frame');
    selectProp(made[0]);
    const overlapping = clashes(add).out.length;
    note(`${add.length} proposal${add.length === 1 ? '' : 's'} from ${src.name}` + (skipped ? ` · ${skipped} already labeled` : '') + (overlapping ? ` · ${overlapping} overlap another box or proposal — ⇧Enter leaves those to you` : '') + ' — Enter accepts, ⇧Enter all', overlapping ? 'warn' : 'info', true);
  }

  // ---------------------------------------------------------------- SAM: refine (R) and click-to-box (W)
  async function refineSelected() {
    const s = selectedItem(); if (!s || !editable()) return;
    let bi = s.i;
    if (s.t === 'prop') { pushHistory('accept proposal'); bi = beginEditProp(s.i); if (bi < 0) { undo.pop(); return; } changed(); setSel({ t: 'box', i: bi }); }
    const b = boxes[bi], token = navToken, target = item;
    const done = waitNote('Refining with SAM…');
    try {
      const res = await fovea.api.post('/api/ai/refine', { dataset_id: ds.id, image_id: target.id, box: [b[1], b[2], b[3], b[4]] });
      done();
      if (navToken !== token || item !== target) return;
      if (!res || !res.box) { note('SAM found no object there', 'warn'); return; }
      const p = normProp({ cls: b[0], xc: res.box[0], yc: res.box[1], w: res.box[2], h: res.box[3], conf: res.score ?? null, kind: 'refine', label: 'SAM', local: true, refBox: b });
      props = props.concat([p]); propsChanged(); selectProp(p);
      note('Refined box shown dashed — Enter replaces the box, Del keeps the original', 'info', true);
    } catch (e) { done(); if (navToken === token && item === target) note('Refine failed: ' + e.message, 'error'); }
  }
  /** A wait shown as a note after 400 ms; the returned function ends it (only if that note is still the one shown). */
  function waitNote(text) {
    const t = setTimeout(() => note(text, 'info', true), NOTICE_MS);
    return () => { clearTimeout(t); if (noteEl.textContent === text) clearNote(); };
  }
  async function samClick(a, negative) {
    if (!editable() || a.x < 0 || a.y < 0 || a.x > W || a.y > H) return;
    const s = selectedItem(), sp = s && s.t === 'prop' && props[s.i].kind === 'refine' && props[s.i].points && isPending(props[s.i]) ? props[s.i] : null;
    const pt = [a.x / W, a.y / H, negative ? 0 : 1];
    const inside = sp && pt[0] >= sp.b[1] - sp.b[3] / 2 && pt[0] <= sp.b[1] + sp.b[3] / 2 && pt[1] >= sp.b[2] - sp.b[4] / 2 && pt[1] <= sp.b[2] + sp.b[4] / 2;
    if (negative && !sp) { note('⇧-click adds a negative point to the selected SAM box'); return; }
    const points = sp && (negative || inside) ? sp.points.concat([pt]) : [pt];
    const token = navToken, target = item;
    const done = waitNote('Finding the object…');
    try {
      const res = await fovea.api.post('/api/ai/refine', { dataset_id: ds.id, image_id: target.id, points });
      done();
      if (navToken !== token || item !== target) return;
      if (!res || !res.box) { note('SAM found no object there', 'warn'); return; }
      const more = sp && points.length > 1;
      const nb = [more ? sp.b[0] : activeCls, ...res.box.slice(0, 4).map(Number)];
      // The object already has a box of this class: a proposal there would only be a hidden duplicate ("already
      // labeled") that Enter still accepted. Point at that box instead; R refines it.
      const dupK = boxes.findIndex(x => x[0] === nb[0] && iou(x, nb) >= DUP_IOU);
      if (dupK >= 0) {
        if (more) { props = props.filter(x => x !== sp); propsChanged(); }     // the box being refined became that box
        setSel({ t: 'box', i: dupK });
        note(`Already labeled as #${dupK + 1} (${className(names, nb[0])}) — R refines that box`);
        return;
      }
      if (more) { sp.b = nb; sp.conf = res.score ?? sp.conf; sp.points = points; propsChanged(); return; }
      const p = normProp({ cls: nb[0], xc: nb[1], yc: nb[2], w: nb[3], h: nb[4], conf: res.score ?? null, kind: 'refine', label: 'SAM', local: true, points });
      props = props.concat([p]); propsChanged(); selectProp(p);
      if (!p.vis) { note(`Class ${nb[0]} · ${className(names, nb[0])} is hidden — show it in the class list to see the box`, 'warn'); return; }
      note('Enter accepts · click inside to add a point · ⇧-click to exclude an area · a digit sets the class', 'info', true);
    } catch (e) { done(); if (navToken === token && item === target) note('Click-to-box failed: ' + e.message, 'error'); }
  }

  // ---------------------------------------------------------------- keyframes & interpolation (K)
  // Interpolation runs between two frames of one sequence in one split (core/qa.py interpolate). A sequence split
  // over two folders interleaves them on the strip, so while a keyframe is set , and . stay in its split.
  const sameTrack = (k, it) => !!k && !!it && k.seq === it.seq && (k.split || null) === (it.split || null);
  async function keyframeKey() {
    if (!item) return;
    if (!item.seq) { note('This image has no sequence — interpolation needs two frames of one sequence', 'warn'); return; }
    if (keyframe && keyframe.seq === item.seq && !sameTrack(keyframe, item)) {
      note(`The keyframe is in ${keyframe.split || 'another split'}, this frame in ${item.split || 'another split'} — interpolation needs both in one split. , and . now stay in ${keyframe.split || 'its split'} (Clear in the Sequence panel starts over here).`, 'warn', true);
      return;
    }
    if (!keyframe || keyframe.seq !== item.seq) {
      const other = keyframe && keyframe.seq !== item.seq;
      keyframe = { id: item.id, name: item.rel_path.split('/').pop(), seq: item.seq, split: item.split || null, rel: item.rel_path };
      seqStatus = other ? 'Keyframe moved to this sequence.' : ''; seqAction = null;
      const twoSplits = filmData && filmData.seq === item.seq && filmData.splits.length > 1;
      renderSeq(); drawTiles(); note('Keyframe set — go to another frame of this sequence and press K to interpolate between them' + (twoSplits ? ` (, and . stay in ${item.split})` : '')); return;
    }
    if (keyframe.id === item.id) { keyframe = null; seqStatus = ''; seqAction = null; renderSeq(); drawTiles(); note('Keyframe cleared'); return; }
    await interpolate();
  }
  async function interpolate() {
    if (!keyframe || !item || !sameTrack(keyframe, item) || keyframe.id === item.id) return;
    if (await flush() !== 'clean') { note(notSavedMsg(), 'warn'); return; }
    const a = keyframe, b = { id: item.id, name: item.rel_path.split('/').pop(), seq: item.seq, split: item.split || null, rel: item.rel_path };
    const [from, to] = a.rel < b.rel ? [a, b] : [b, a];
    seqStatus = 'Interpolating…'; renderSeq();
    seqAction = null;
    try {
      const res = await fovea.api.post(`/api/datasets/${ds.id}/interpolate`, { from_image_id: from.id, to_image_id: to.id });
      // What was left out is said too: keyframe boxes with no partner, boxes already on the frames in between, boxes
      // paired across a class change (the server's message), and the server's reason when nothing ran.
      const sum = interpSummary(res, from.name, to.name), { n, imgs } = sum;
      const extra = sum.brief ? ' · ' + sum.brief : '';     // in a few words: the Sequence panel has the sentences
      if (n) {
        // ⇧N only looks ahead. Marking the later keyframe second (the usual way) leaves the new proposals behind
        // this frame, so the way to them is offered: the first frame in between, from where ⇧N steps on.
        const first = firstInterpolated(from, Array.isArray(res.preds) ? res.preds : []);
        const behind = to.id === b.id;
        seqStatus = [sum.head + (behind ? ' — they are before this frame: Review opens the first, then ⇧N steps through them' : ' — ⇧N reviews them'), ...sum.notes];
        if (first) seqAction = { label: `Review from ${first.label}`, tip: `Open ${first.name}, the first frame with an interpolated proposal`, icon: behind ? 'chevronLeft' : 'chevronRight', run: () => reviewInterpolated(first.id) };
        note((behind && first ? `${n} interpolated proposal${n === 1 ? '' : 's'} on the ${imgs} frame${imgs === 1 ? '' : 's'} before this one — “Review” in the Sequence panel opens the first` : `${n} interpolated proposal${n === 1 ? '' : 's'} — ⇧N reviews them`) + extra, sum.warn ? 'warn' : 'info');
      } else { seqStatus = [sum.head, ...sum.notes]; note(sum.head + extra, 'warn'); }
      keyframe = b;                                      // chain: the next K interpolates onward from here
      fovea.bus.emit('proposals:changed', { dataset: ds.id, source: 'annotate' });
      filmRefresh();                                     // the frames in between now carry proposals
    } catch (e) { seqStatus = 'Interpolation failed: ' + e.message; note(seqStatus, 'error'); }
    renderSeq();
  }
  /** The first frame after keyframe `from` (strip order) that received an interpolated proposal. */
  function firstInterpolated(from, preds) {
    const ids = new Set(preds.map(p => p.image_id));
    if (!ids.size) return null;
    const data = filmCache.get(from.seq), k = data ? data.pos.get(from.id) : null;
    if (k != null) for (let j = k + 1; j < data.frames.length; j++) if (ids.has(data.frames[j].id)) return { id: data.frames[j].id, label: `frame ${fmt.num(j + 1)}`, name: baseOf(data.frames[j].rel_path) };
    const id = Math.min(...ids);
    return { id, label: `#${id}`, name: `#${id}` };
  }
  async function reviewInterpolated(id) {
    let pos = cursor.indexOfId(id);
    if (pos < 0) pos = await cursor.positionOf(id);
    if (destroyed) return;
    if (pos >= 0) { stopAuto(); goTo(pos); } else note('That frame is not in this working set — “All frames” in the filmstrip opens the whole sequence', 'warn');
  }

  // ---------------------------------------------------------------- sequence filmstrip (\ toggles · , and . step)
  // An overlay along the bottom of the stage: showing or hiding it never resizes the canvas, and the picture stays
  // where it is until the next fit (fitting then leaves the strip's band free). Frames are in natural file-name
  // order — a sequence split over train and test interleaves there — only the tiles in view exist, and each
  // thumbnail appears once decoded. The strip of the next image's sequence is loaded with that image, so both
  // change in the same frame; jumps scroll it instantly, a third in, never with an animation.
  const FILM_H = 84, FILM_TH = 58, FILM_GAP = 4, FILM_PAD = 3, FILM_MAX = 5000, FILM_PAGE = 1000;
  const filmCache = new Map();     // seq -> { seq, frames, pos: Map(id -> index), splits, total, partial, inSet, stale }
  const filmLoads = new Map();
  const tileEls = new Map();       // frame index -> tile element (only the tiles in view)
  let hasSeqs = null, filmData = null, filmTw = FILM_TH, filmLastK = -1, filmErr = null, filmRaf = 0, filmTouchRaf = 0, seqStepToken = 0, seqReturn = null;
  function filmShown() { return opts.film === true || (opts.film == null && hasSeqs === true); }
  function filmBand() { return filmShown() ? FILM_H : 0; }
  const filmLabel = h('div', { class: 'ann-film-label' });
  const filmInner = h('div', { class: 'ann-film-inner', role: 'list' });
  const filmScroll = h('div', { class: 'ann-film-scroll' }, filmInner);
  const film = h('div', { class: 'ann-film hidden', role: 'navigation', 'aria-label': 'Sequence filmstrip', style: `height:${FILM_H}px` }, filmLabel, filmScroll);
  stage.appendChild(film);

  /** Every frame of a sequence (up to FILM_MAX around the current image), with which of them this working set holds. */
  function loadFilm(seq) {
    const c = filmCache.get(seq);
    if (c && !c.stale) return c.inSet === undefined ? filmMembership(c) : Promise.resolve(c);
    if (filmLoads.has(seq)) return filmLoads.get(seq);
    const p = (async () => {
      const q = { seq, sort: 'name', order: 'asc' };
      const first = await fovea.api.get(`/api/datasets/${ds.id}/images`, { ...q, offset: 0, limit: FILM_PAGE });
      let rows = first.items || [];
      const total = first.total ?? rows.length;
      let start = 0;
      if (total > rows.length) {
        if (total > FILM_MAX) {               // a very long sequence: the FILM_MAX frames around the current image
          const cur = item && item.seq === seq ? item.id : null;
          const at = cur != null ? (await fovea.api.get(`/api/datasets/${ds.id}/images/position`, { ...q, id: cur })).position : 0;
          start = Math.max(0, Math.min(total - FILM_MAX, (at || 0) - FILM_MAX / 2));
          if (start > 0) rows = [];
        }
        const end = Math.min(total, start + FILM_MAX), pages = [];
        for (let off = start + rows.length; off < end; off += FILM_PAGE) pages.push(fovea.api.get(`/api/datasets/${ds.id}/images`, { ...q, offset: off, limit: Math.min(FILM_PAGE, end - off), count: 0 }));
        for (const r of await Promise.all(pages)) rows = rows.concat(r.items || []);
      }
      rows = filmOrder(rows);
      const data = { seq, frames: rows, pos: new Map(rows.map((r, k) => [r.id, k])), ord: frameOrdinals(rows), splits: [...new Set(rows.map(r => r.split))], total, partial: rows.length < total, inSet: undefined, stale: false };
      const old = filmCache.get(seq); if (old && old.inSet !== undefined && !old.stale) data.inSet = old.inSet;
      filmCache.delete(seq); filmCache.set(seq, data);
      while (filmCache.size > 12) filmCache.delete(filmCache.keys().next().value);
      return data.inSet === undefined ? filmMembership(data) : data;
    })();
    filmLoads.set(seq, p);
    p.finally(() => { if (filmLoads.get(seq) === p) filmLoads.delete(seq); }).catch(() => {});
    return p;
  }
  // Without filters every frame is in the working set. Otherwise the live query says which are (the frozen cursor
  // also keeps an image that left the filter after it was opened — see filmInSet).
  function filmMembership(data) {
    if (data.memberP) return data.memberP;
    const f = { ...filters };
    if (!Object.keys(f).length) { data.inSet = null; return Promise.resolve(data); }
    if (f.seq && f.seq !== data.seq) { data.inSet = new Set(); return Promise.resolve(data); }
    const p = data.memberP = fovea.api.get(`/api/datasets/${ds.id}/images/ids`, { ...f, seq: data.seq, limit: FILM_MAX * 2 })
      .then((r) => { if (data.memberP === p) { data.inSet = new Set(r.ids || []); data.memberP = null; } return data; },
        (e) => { if (data.memberP === p) data.memberP = null; throw e; });
    return p;
  }
  // byId (not indexOfId, which scans every loaded row on a miss): this runs for every frame of the strip.
  const filmInSet = (data, id) => !data || data.inSet == null || data.inSet.has(id) || (cursor.byId ? cursor.byId.has(id) : cursor.indexOfId(id) >= 0);
  function forgetMembership() { for (const c of filmCache.values()) { c.inSet = undefined; c.memberP = null; } }
  /** A frame's row with what is on screen now for the image being edited (boxes, decisions, review). */
  function liveRow(f) {
    if (!item || f.id !== item.id) return f;
    const out = { ...f, review: item.review || null, issues: item.issues || f.issues, flags: item.flags || f.flags };
    if (labelsError || !img) return out;
    let pend = 0, sus = 0; for (const p of props) if (!p.local && isPending(p)) { if (AUDIT.has(p.kind)) sus++; else pend++; }
    return { ...out, n_boxes: boxes.length, has_label: boxes.length > 0 || (!!item.has_label && !(dirty && emptyMeansNoFile(item))), classes: [...new Set(boxes.map(b => b[0]))].sort((a, b) => a - b), ai_pending: pend, ai_suspect: sus };
  }
  function filmCommit() {
    if (!item || !item.seq) return;
    const data = filmCache.get(item.seq), k = data ? data.pos.get(item.id) : null;
    if (k != null) data.frames[k] = liveRow(data.frames[k]);
  }
  function filmSetRow(id, patch) { for (const d of filmCache.values()) { const k = d.pos.get(id); if (k != null) d.frames[k] = { ...d.frames[k], ...patch }; } }
  function makeTile() {
    const t = h('div', { class: 'fs-tile', role: 'listitem' });
    t._pic = h('span', { class: 'pic fs-pic is-idle' }); t._img = h('img', { alt: '', draggable: 'false' }); t._img.decoding = 'async'; t._pic.appendChild(t._img);
    t._marks = h('span', { class: 'fs-marks' }); t._n = h('span', { class: 'fs-n' }); t._kf = h('span', { class: 'fs-kf' }); t._bar = h('span', { class: 'fs-bar' });
    t.append(t._pic, t._marks, t._n, t._kf, t._bar);
    return t;
  }
  const tileW = (data) => { const f0 = data.frames[0] || {}; const ar = f0.width && f0.height ? f0.width / f0.height : img ? W / H : 4 / 3; return Math.max(40, Math.min(112, Math.round(FILM_TH * ar))); };
  const thumbOf = (f, tw) => fovea.api.thumbUrl(f.id, thumbPx(Math.max(tw, FILM_TH)), f.mtime);
  // Thumbnails of a sequence about to be shown are decoded first (≤ 250 ms, usually ahead of time from the prefetch),
  // so its tiles appear complete in the same frame as the picture instead of popping in one by one.
  const warmed = new Map();
  function filmWarm(data, id) {
    const k = data ? data.pos.get(id) : null;
    if (k == null || (filmData && filmData.seq === data.seq)) return data;
    const tw = tileW(data), n = Math.ceil((filmScroll.clientWidth || 800) / (tw + FILM_GAP)) + 5;   // the view plus its overscan
    const a = Math.max(0, Math.min(data.frames.length - n, k - Math.ceil(n / 2))), z = Math.min(data.frames.length, a + n), wait = [];
    for (let j = a; j < z; j++) {
      const url = thumbOf(data.frames[j], tw);
      if (!warmed.has(url)) { const p = decodedImage(url, { priority: 'low' }); p.catch(() => {}); warmed.set(url, p); wait.push(p.catch(() => null)); }
    }
    while (warmed.size > 120) warmed.delete(warmed.keys().next().value);
    return Promise.race([Promise.all(wait), sleep(250)]).then(() => data);
  }
  function fillTile(t, k) {
    const f = liveRow(filmData.frames[k]), cur = !!item && f.id === item.id;
    t.style.left = (FILM_PAD + k * (filmTw + FILM_GAP)) + 'px'; t.style.width = filmTw + 'px'; t.dataset.k = k;
    if (t._id !== f.id) {
      t._id = f.id;
      t._img.setAttribute('fetchpriority', 'low');
      t._img.src = thumbOf(f, filmTw);
      trackImage(t._img, t._pic);
    }
    const inSet = filmInSet(filmData, f.id);
    const sig = [cur, inSet, f.review && f.review.status, f.n_boxes, f.has_label, f.ai_pending, f.ai_suspect, (f.issues || []).join(), (f.flags || []).join(), (f.classes || []).join(), keyframe && keyframe.id === f.id].join('|');
    if (t._sig === sig) return;
    t._sig = sig;
    t.className = cls('fs-tile', cur && 'current', !inSet && 'out', f.review && `rv-${f.review.status}`);
    if (cur) t.setAttribute('aria-current', 'true'); else t.removeAttribute('aria-current');
    const probs = (f.issues || []).filter(c => c !== 'missing_label').concat(f.flags || []);
    t._marks.replaceChildren(...[f.ai_pending ? h('i', { class: 'fs-m ai' }, `◌${f.ai_pending}`) : null, f.ai_suspect ? h('i', { class: 'fs-m sus' }, `?${f.ai_suspect}`) : null, probs.length ? h('i', { class: 'fs-m warn' }, '!') : null].filter(Boolean));
    t._n.textContent = f.has_label || f.n_boxes ? String(f.n_boxes || 0) : '–';
    t._n.classList.toggle('zero', !f.n_boxes);
    t._kf.textContent = keyframe && keyframe.id === f.id ? 'K' : '';
    const cl = f.classes || [];
    t._bar.replaceChildren(...cl.slice(0, 8).map(c => h('i', { style: `background:${classColor(c)}` })));
    const bits = [baseOf(f.rel_path), f.split, f.has_label || f.n_boxes ? `${f.n_boxes || 0} box${f.n_boxes === 1 ? '' : 'es'}` : 'no label file'];
    if (cl.length) bits.push(cl.map(c => className(names, c)).join(', '));
    if (f.ai_pending) bits.push(`${f.ai_pending} proposal${f.ai_pending === 1 ? '' : 's'}`);
    if (f.ai_suspect) bits.push(`${f.ai_suspect} suspected error${f.ai_suspect === 1 ? '' : 's'}`);
    if (probs.length) bits.push(probs.map(c => issueLabel(c)).join(', '));
    if (f.review) bits.push(REVIEW[f.review.status] ? REVIEW[f.review.status].label : f.review.status);
    if (!inSet) bits.push('not in this working set');
    t.title = bits.join(' · ');
  }
  function drawTiles() {
    if (!filmData || !filmShown()) { for (const t of tileEls.values()) { t._img.removeAttribute('src'); t.remove(); } tileEls.clear(); return; }
    const n = filmData.frames.length, span = filmTw + FILM_GAP;
    const sl = filmScroll.scrollLeft, vw = filmScroll.clientWidth || 800;
    const a = Math.max(0, Math.floor(sl / span) - 3), z = Math.min(n, Math.ceil((sl + vw) / span) + 3);
    // A tile leaving the view drops its src, which cancels a thumbnail still downloading.
    for (const [k, t] of tileEls) if (k < a || k >= z) { t._img.removeAttribute('src'); t.remove(); tileEls.delete(k); }
    for (let k = a; k < z; k++) { let t = tileEls.get(k); if (!t) { t = makeTile(); tileEls.set(k, t); filmInner.appendChild(t); } fillTile(t, k); }
  }
  function filmTouch() {
    if (filmTouchRaf || !filmData || !item || !filmShown()) return;
    filmTouchRaf = requestAnimationFrame(() => { filmTouchRaf = 0; if (!filmData || !item) return; const k = filmData.pos.get(item.id), t = k != null ? tileEls.get(k) : null; if (t) fillTile(t, k); });
  }
  function renderFilmLabel() {
    const kids = [];
    if (!item) kids.push(h('div', { class: 'fs-title' }, 'No image'));
    else if (!item.seq) kids.push(h('div', { class: 'fs-title' }, 'No sequence'));
    else {
      kids.push(h('div', { class: 'fs-title', title: `Sequence ${item.seq}` }, item.seq));
      if (filmData) {
        // Frames, not images: a sequence held in two folders (twins) has two images of most frames, and , . step
        // over the copy — so "Frame k of n" counts distinct frame names, and the copies are told apart.
        const k = filmData.pos.get(item.id), n = filmData.frames.length, nf = filmData.ord.n, copies = n - nf;
        const twins = copies > 0 ? `${fmt.num(n)} images of ${plural(nf, 'frame')}: ${plural(filmData.ord.multi, 'frame is', 'frames are')} in more than one folder${filmData.splits.length > 1 ? ` (${filmData.splits.join(', ')})` : ''} — , and . step over the copies` : null;
        kids.push(h('div', { class: 'fs-sub', title: twins }, filmData.partial ? `${fmt.num(n)} of ${fmt.num(filmData.total)} images` : k != null ? `Frame ${fmt.num(filmData.ord.of[k] + 1)} of ${fmt.num(nf)}` : `${plural(nf, 'frame')}`));
        if (filmData.splits.length > 1) kids.push(h('div', { class: 'fs-sub', title: filmData.splits.join(', ') }, `Spans ${filmData.splits.join(', ')}`));
        const out = filmData.inSet ? filmData.frames.reduce((s, f) => s + (filmInSet(filmData, f.id) ? 0 : 1), 0) : 0;
        if (seqReturn && filters.seq === item.seq && Object.keys(filters).length === 1) kids.push(h('button', { class: 'btn btn-sm fs-btn', 'data-tip': 'Return to the list you were working through', onClick: () => backFromSequence() }, icon('chevronLeft', 12), 'Back to list'));
        else if (out > 0) kids.push(h('button', { class: 'btn btn-sm fs-btn', 'data-tip': `${fmt.num(out)} frame${out === 1 ? ' is' : 's are'} not in this working set — work through the whole sequence instead`, onClick: () => openSequence() }, `All ${fmt.num(n)} frames`));
      } else if (filmErr) kids.push(h('div', { class: 'fs-sub' }, 'Frames could not be loaded'));
    }
    filmLabel.replaceChildren(...kids);
  }
  function renderFilm() {
    const on = filmShown() && !!item;          // nothing to show in an empty working set
    film.classList.toggle('hidden', !on); root.classList.toggle('film-on', on);
    if (!on) { if (tileEls.size) drawTiles(); return; }
    const data = item && item.seq ? filmCache.get(item.seq) || null : null;
    if (data !== filmData) {
      // Fresh rows for the sequence already shown (a refresh) keep the tiles: their thumbnails never blink.
      const same = !!data && !!filmData && data.seq === filmData.seq;
      if (!same) { for (const t of tileEls.values()) { t._img.removeAttribute('src'); t.remove(); } tileEls.clear(); filmLastK = -1; }
      filmData = data;
      if (data) {
        if (!same) filmTw = tileW(data);
        filmInner.style.width = (2 * FILM_PAD + data.frames.length * (filmTw + FILM_GAP)) + 'px';
      } else filmInner.style.width = '0px';
    }
    if (item && item.seq && (!data || data.stale || data.inSet === undefined)) {
      const seq = item.seq;
      loadFilm(seq).then(() => { filmErr = null; if (!destroyed && item && item.seq === seq && filmCache.get(seq) !== filmData) renderFilm(); else if (!destroyed) { renderFilmLabel(); drawTiles(); } })
        .catch((e) => { filmErr = e; if (!destroyed && item && item.seq === seq) renderFilmLabel(); });
    }
    renderFilmLabel();
    if (!filmData || !item) { drawTiles(); return; }
    const k = filmData.pos.get(item.id);
    if (k != null) {
      const span = filmTw + FILM_GAP, vw = filmScroll.clientWidth || 800, x = FILM_PAD + k * span, sl = filmScroll.scrollLeft;
      if (filmLastK < 0 || x < sl + span * 0.5 || x + filmTw > sl + vw - span * 0.5) {
        const fwd = filmLastK < 0 || k >= filmLastK;
        filmScroll.scrollLeft = Math.max(0, x - (filmLastK < 0 ? vw / 2 : fwd ? vw / 3 : vw * 2 / 3));
      }
      filmLastK = k;
    }
    drawTiles();
  }
  filmScroll.addEventListener('scroll', () => { if (!filmRaf) filmRaf = requestAnimationFrame(() => { filmRaf = 0; drawTiles(); }); }, { passive: true });
  // A vertical wheel scrolls the strip sideways; nothing on the strip zooms the picture.
  filmScroll.addEventListener('wheel', (e) => { if (Math.abs(e.deltaY) > Math.abs(e.deltaX)) { e.preventDefault(); filmScroll.scrollLeft += e.deltaY * (e.deltaMode === 1 ? 16 : e.deltaMode === 2 ? filmScroll.clientWidth : 1); } }, { passive: false });
  filmInner.addEventListener('mousedown', (e) => { if (e.target.closest('.fs-tile')) e.preventDefault(); });   // keyboard focus stays where it was
  filmInner.addEventListener('click', (e) => { const t = e.target.closest('.fs-tile'); const f = t && filmData ? filmData.frames[Number(t.dataset.k)] : null; if (f) jumpToFrame(f); });
  async function jumpToFrame(f) {
    if (item && f.id === item.id) return;
    const data = filmData;
    let pos = cursor.indexOfId(f.id);
    if (pos < 0 && filmInSet(data, f.id)) pos = await cursor.positionOf(f.id);
    if (destroyed) return;
    if (pos >= 0) { stopAuto(); goTo(pos); return; }
    note(`${baseOf(f.rel_path)} is not in this working set — “All frames” opens the whole sequence`, 'warn');
  }
  /** , and . — the previous / next frame of this sequence that the working set holds (others are skipped and counted). */
  async function stepSeq(d) {
    if (!item) return;
    const from = loading && want >= 0 && cursor.items[want] ? cursor.items[want] : item;
    if (!from.seq) { note('This image has no sequence'); return; }
    const token = ++seqStepToken;
    let data;
    try { data = await loadFilm(from.seq); } catch (e) { note(e.message, 'error'); return; }
    if (token !== seqStepToken || destroyed) return;
    const k = data.pos.get(from.id);
    if (k == null) { note('This frame is outside the part of the sequence that is loaded', 'warn'); return; }
    // With a keyframe in this sequence, only frames of the keyframe's split (interpolation cannot cross splits).
    const track = keyframe && keyframe.seq === from.seq && data.splits.length > 1 ? (keyframe.split || null) : undefined;
    // One step is one frame name: copies of the same frame in other folders (twins) are not another frame, and of
    // the next frame's copies the one in this frame's folder comes first (seqNeighbor).
    const fr = data.frames;
    let skipped = 0, otherSplit = 0, j = k + d;
    while (j >= 0 && j < fr.length && sameFrame(fr[j], fr[k])) j += d;
    while (j >= 0 && j < fr.length && skipped + otherSplit <= 500) {
      const group = frameGroup(fr, j, d, from.split);
      j += group.length * d;
      let tried = false;
      for (const m of group) {
        const f = fr[m];
        if (track !== undefined && (f.split || null) !== track) continue;
        tried = true;
        if (!filmInSet(data, f.id)) continue;
        let pos = cursor.indexOfId(f.id);
        if (pos < 0) pos = await cursor.positionOf(f.id);
        if (token !== seqStepToken || destroyed) return;
        if (pos >= 0) {
          if (skipped) note(`Skipped ${skipped} frame${skipped === 1 ? '' : 's'} not in this working set`);
          else if (track !== undefined && (from.split || null) !== track) note(`Back in ${track}, the keyframe's split`);
          goTo(pos, { source: 'key' }); return;
        }
      }
      if (tried) skipped++; else otherSplit++;
    }
    note((d > 0 ? `Last frame of ${from.seq}` : `First frame of ${from.seq}`) + (track !== undefined ? ` in ${track}` : '') + ' in this working set');
  }
  async function openSequence() {
    if (!item || !item.seq) return;
    const back = { filters: { ...filters }, id: item.id };
    if (await setFilters({ seq: item.seq }, { keepReturn: true })) seqReturn = back;
    renderFilm(); renderWorkingSet();
  }
  /** Back to the list the sequence was opened from: at the frame on screen if that list holds it, else where the user was. */
  async function backFromSequence() { const r = seqReturn; seqReturn = null; if (r) await setFilters(r.filters, { fallbackId: r.id }); renderFilm(); }
  function toggleFilm() {
    opts.film = !filmShown(); LS.set('fovea.ann.film', opts.film ? '1' : '0');
    holdFit = true;              // the picture stays where it is: nothing moves when the strip comes or goes (F re-fits)
    renderFilm(); renderSeq(); updateHud();
    note(opts.film ? 'Filmstrip on — , and . step through this sequence' : 'Filmstrip off — \\ shows it again');
  }
  let filmRefreshTimer = null;
  /** Something outside this editor changed labels or proposals (AI batch, range edit, interpolation). */
  function filmRefresh() {
    clearTimeout(filmRefreshTimer);
    filmRefreshTimer = setTimeout(() => {
      for (const c of filmCache.values()) c.stale = true;
      prevLabels.clear();                // label files may have changed too (an AI batch writing labels)
      if (!destroyed && filmShown() && item && item.seq) renderFilm();
    }, 300);
  }

  // ---------------------------------------------------------------- auto-advance
  // Steps from the frame on screen once it has been shown for the interval. It waits while the user edits: a
  // pointer press, a drag, unsaved changes or a save in flight pause it (shown on the button).
  function scheduleAuto() {
    clearTimeout(autoTimer);
    autoTimer = setTimeout(async () => {
      if (!autoOn || destroyed) return;
      if (drag || dirty || saving || loading) { playBtn.classList.add('paused'); playBtn.dataset.tip = 'Paused while you edit'; scheduleAuto(); return; }
      playBtn.classList.remove('paused'); playBtn.dataset.tip = 'Auto-advance';
      if (cursor.total != null && idx >= cursor.total - 1) { stopAuto(); return; }
      await goTo(idx + 1, { source: 'auto' });
      if (autoOn && !destroyed) scheduleAuto();
    }, Number(speed.value));
  }
  function pokeAuto() { if (autoOn) scheduleAuto(); }
  function setPlayBtn() { playBtn.classList.toggle('active', autoOn); playBtn.classList.remove('paused'); playBtn.replaceChildren(icon(autoOn ? 'pause' : 'play', 13)); }
  function startAuto() { autoOn = true; setPlayBtn(); scheduleAuto(); }
  function stopAuto() { const was = autoOn || autoTimer; autoOn = false; clearTimeout(autoTimer); autoTimer = null; if (was) setPlayBtn(); }
  function toggleAuto() { autoOn ? stopAuto() : startAuto(); }
  function stopAutoIfEnd(i) { if (autoOn && cursor.total != null && i >= cursor.total - 1) stopAuto(); }

  // ---------------------------------------------------------------- range edit
  let rangeJob = null;
  let rangeTo = 0, rangeOnly = '';
  // The result line (and its Undo) lives outside the rebuilt part: renderRange runs again whenever the class list
  // is redrawn — right after an apply, when the class counts are reloaded — and must not wipe the result.
  const rangeStatus = h('div', { class: 'xs faint ann-range-status', role: 'status' });
  const setRangeStatus = (...kids) => rangeStatus.replaceChildren(...kids);
  function renderRange() {
    rangeBody.replaceChildren();
    const clsSel = h('select', { class: 'select select-sm', 'aria-label': 'New class', onChange: (e) => { rangeTo = Number(e.target.value); } }, names.map((n, i) => h('option', { value: i, selected: i === rangeTo }, `${i} ${n}`)));
    const onlySel = h('select', { class: 'select select-sm', 'aria-label': 'Only boxes of class', onChange: (e) => { rangeOnly = e.target.value; } }, [h('option', { value: '' }, 'every box'), ...names.map((n, i) => h('option', { value: i, selected: String(i) === rangeOnly }, `only class ${i} ${n}`))]);
    const startTxt = rangeStart ? `From ${rangeStart.name}` : 'No range start';
    rangeBody.appendChild(h('div', { class: 'small muted' }, 'Set the class of boxes across a run of images (e.g. a fall sequence).'));
    rangeBody.appendChild(h('div', { class: 'row' },
      h('button', { class: 'btn btn-sm', onClick: () => { if (!item) return; rangeStart = { id: item.id, name: item.rel_path.split('/').pop() }; if (!rangeJob) setRangeStatus(); renderRange(); } }, icon('bookmark', 12), 'Mark start'),
      rangeStart ? h('button', { class: 'btn btn-ghost btn-sm', onClick: () => { rangeStart = null; renderRange(); } }, 'Clear') : null));
    rangeBody.appendChild(h('div', { class: 'xs faint truncate', title: startTxt }, startTxt));
    rangeBody.appendChild(h('div', { class: 'ann-range-grid' }, h('span', { class: 'xs faint' }, 'Set to'), clsSel, h('span', { class: 'xs faint' }, 'Apply to'), onlySel));
    rangeBody.appendChild(h('button', { class: 'btn btn-sm btn-primary', disabled: !rangeStart || !!rangeJob, onClick: () => applyRange(Number(clsSel.value), onlySel.value === '' ? null : Number(onlySel.value)) }, 'Apply to range'));
    rangeBody.appendChild(rangeStatus);
  }
  const plural = (n, one, many = one + 's') => `${fmt.num(n)} ${n === 1 ? one : many}`;
  /** Everything a range edit (or its undo) changed outside this image: counts, the strip, other views. */
  function afterBulkWrite() { countsCache.delete(ds.id); prevLabels.clear(); loadCounts(); fovea.bus.emit('dataset:updated'); }
  async function applyRange(c, only) {
    if (!rangeStart || !item || rangeJob) return;
    let a = cursor.indexOfId(rangeStart.id);
    if (a < 0) a = await cursor.positionOf(rangeStart.id);
    if (a < 0) { setRangeStatus('The start image is no longer in this list'); return; }
    const [lo, hi] = [Math.min(a, idx), Math.max(a, idx)], n = hi - lo + 1;
    const what = only == null ? 'every box' : `boxes of class ${only} · ${className(names, only)}`;
    if (!await ui.confirm({ title: 'Apply class to range', message: `Set ${what} in ${plural(n, 'image')} (positions ${lo + 1}–${hi + 1}) to class ${c} · ${className(names, c)}? This writes label files now.`, okLabel: 'Apply' })) return;
    if (await flush() !== 'clean') { ui.toast(notSavedMsg(), { type: 'error' }); return; }
    rangeJob = true; renderRange(); setRangeStatus();
    const clock = ui.elapsedClock();
    const showWait = setTimeout(() => { setRangeStatus(ui.spinner(), ' Writing… ', clock); }, NOTICE_MS);
    try {
      const ids = []; for (let i = lo; i <= hi; i++) { const it = await cursor.ensure(i); if (it) ids.push(it.id); }
      const body = { op: 'set_class', cls: c, image_ids: ids }; if (only != null) body.only_cls = only;
      const res = await fovea.api.post(`/api/datasets/${ds.id}/labels/bulk`, body);
      let result = res, cancelled = false;
      if (res.job) { const snap = await fovea.api.watchJob(res.job.id, (s) => { setRangeStatus(ui.spinner(), ` ${s.message || 'Writing…'} `, clock); }); cancelled = snap.status === 'cancelled'; result = snap.result || {}; }
      clearTimeout(showWait);
      rangeStart = null; rangeJob = null;
      const nImg = result.images ?? ids.length, nBox = result.changed ?? 0;
      const done = cancelled ? 'Cancelled — images processed before the cancel were changed' : `Done: ${plural(nImg, 'image')}, ${plural(nBox, 'box', 'boxes')} changed`;
      setRangeStatus(done, res.snapshot_id ? undoRangeBtn(res.snapshot_id, done) : null);
      renderRange();
      afterBulkWrite();
      await goTo(idx, { reload: true });
    } catch (e) { clearTimeout(showWait); rangeJob = null; renderRange(); setRangeStatus('Failed: ' + e.message); }
  }
  const undoRangeBtn = (snapshotId, done) => h('button', { class: 'btn btn-ghost btn-sm', 'data-tip': 'Put the label files back as they were before this range edit', onClick: () => undoRange(snapshotId, done) }, 'Undo');
  async function undoRange(snapshotId, done) {
    if (rangeJob) return;
    if (await flush() !== 'clean') { ui.toast(notSavedMsg(), { type: 'error' }); return; }
    rangeJob = true; renderRange();
    const clock = ui.elapsedClock();
    setRangeStatus(done);
    const showWait = setTimeout(() => { setRangeStatus(ui.spinner(), ' Undoing… ', clock); }, NOTICE_MS);
    try {
      const r = await fovea.api.post(`/api/datasets/${ds.id}/snapshots/${snapshotId}/restore`, { mode: 'undo' });   // backend-data hook: an image edited since keeps that edit
      const fin = r.job ? await fovea.api.watchJob(r.job.id) : null;
      clearTimeout(showWait); rangeJob = null;
      setRangeStatus(fin && fin.result && fin.result.changed_since ? `Range edit undone · ${fin.message}` : 'Range edit undone'); renderRange();
      afterBulkWrite();
      await goTo(idx, { reload: true });
    } catch (e) { clearTimeout(showWait); rangeJob = null; renderRange(); setRangeStatus(`Undo failed: ${e.message} · `, undoRangeBtn(snapshotId, done)); }
  }

  // ---------------------------------------------------------------- panels
  function setSaveState(s, msg) {
    if (s === 'saved' && decFail) { s = 'error'; msg = decFail; }      // the boxes are on disk, some decisions are not
    if (msg === 'Decisions not saved') saveState.dataset.tip = decFail ? `Accept / reject decisions did not reach the server — retrying in the background (${MOD === '⌘' ? '⌘' : 'Ctrl+'}S retries now)` : 'Accept / reject decisions were refused by the server — they show as pending when this image is opened again';
    else delete saveState.dataset.tip;
    saveState.className = `save-state ${s}`;
    saveState.replaceChildren(icon(s === 'saved' ? 'check' : s === 'dirty' ? 'clock' : s === 'saving' ? 'refresh' : s === 'loading' ? 'clock' : 'alertCircle', 12),
      document.createTextNode(msg || (s === 'saved' ? 'Saved' : s === 'dirty' ? 'Unsaved' : s === 'saving' ? 'Saving…' : s === 'loading' ? 'Loading…' : 'Save failed')));
  }
  // Box list: virtualised (dense frames have hundreds of boxes), rows updated in place.
  const rowEls = new Map();
  let listRaf = 0; const rowDirty = new Set();
  function scheduleRowUpdate(i) { rowDirty.add(i); if (!listRaf) listRaf = requestAnimationFrame(() => { listRaf = 0; for (const k of rowDirty) { const r = rowEls.get(k); if (r) fillRow(r, k); } rowDirty.clear(); }); }
  vlist.addEventListener('scroll', () => renderList(), { passive: true });
  vlist.addEventListener('keydown', (e) => {
    if (!boxes.length) return;
    const cur = sel && sel.t === 'box' ? sel.i : -1;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') { e.preventDefault(); e.stopPropagation(); const n = e.key === 'ArrowDown' ? Math.min(boxes.length - 1, cur + 1) : Math.max(0, cur - 1); setSel({ t: 'box', i: n }); }
  });
  function fillRow(r, i) {
    const b = boxes[i]; if (!b) return;
    const active = sel && sel.t === 'box' && sel.i === i;
    r.className = cls('box-row', active && 'active', hiddenCls.has(b[0]) && 'is-hidden');
    r.setAttribute('aria-selected', String(!!active));
    r.style.top = (i * ROW_H) + 'px';
    r.dataset.i = i;
    r._idx.textContent = String(i + 1);
    r._sw.style.background = classColor(b[0]);
    const nm = `${b[0]} · ${className(names, b[0])}`; if (r._name.textContent !== nm) r._name.textContent = nm;
    r._name.title = b.attrs ? attrText(annSch, b.attrs) : '';
    r._geo.textContent = boxSizeText(b, sizeKnown ? W : 0, sizeKnown ? H : 0);
    r._geo.title = sizeKnown ? 'width × height in image pixels' : 'width × height as a share of the image (its size is not known)';
    r.classList.toggle('flagged', !!flagFocus && flagFocus.forId === (item && item.id) && flagFocus.refs.includes(b));
    r._warn.textContent = b.kind ? '🔒' : b.issues ? '!' : '';
    r._warn.title = b.kind ? `${b.kind} row: class and delete only` : b.issues ? b.issues.join(', ').replace(/_/g, ' ') : '';
    r._del.setAttribute('aria-label', `Delete box ${i + 1}`);
  }
  function makeRow() {
    const r = h('div', { class: 'box-row', role: 'option' });
    r._idx = h('span', { class: 'idx' }); r._sw = h('span', { class: 'swatch' }); r._name = h('span', { class: 'name' });
    r._warn = h('span', { class: 'warn' }); r._geo = h('span', { class: 'geo', title: 'width × height in image pixels' });
    r._del = h('button', { class: 'btn btn-ghost btn-sm btn-icon del', tabindex: '-1', onClick: (e) => { e.stopPropagation(); const i = Number(r.dataset.i); if (!editable() || !boxes[i]) return; pushHistory('delete'); boxes.splice(i, 1); sel = null; changed(); setSel(null); } }, icon('x', 12));
    r.append(r._idx, r._sw, r._name, r._warn, r._geo, r._del);
    r.addEventListener('click', () => { if (loading) return; const i = Number(r.dataset.i); if (boxes[i]) setSel({ t: 'box', i }, { scroll: false }); });
    r.addEventListener('mouseenter', () => { const i = Number(r.dataset.i); hov = { t: 'box', i }; overlayOnly(); });
    r.addEventListener('mouseleave', () => { hov = null; overlayOnly(); });
    return r;
  }
  function renderList() {
    const n = boxes.length;
    boxCount.textContent = String(n);
    renderEmptyState(n);
    vlist.classList.toggle('hidden', n === 0);
    vinner.style.height = (n * ROW_H) + 'px';
    vlist.style.height = Math.min(n * ROW_H + 2, Math.max(160, Math.round(innerHeight * 0.36))) + 'px';
    const top = vlist.scrollTop, hgt = vlist.clientHeight || 300;
    const a = Math.max(0, Math.floor(top / ROW_H) - 4), z = Math.min(n, Math.ceil((top + hgt) / ROW_H) + 4);
    for (const [k, r] of rowEls) if (k < a || k >= z) { r.remove(); rowEls.delete(k); }
    for (let i = a; i < z; i++) { let r = rowEls.get(i); if (!r) { r = makeRow(); rowEls.set(i, r); vinner.appendChild(r); } fillRow(r, i); }
  }
  /** No boxes: say whether the image has an empty label file (a deliberate background frame) or none at all, and
      switch between the two on purpose — emptying an image never does it by accident (see labelBody). */
  function renderEmptyState(n) {
    const hasFile = !!item && !!item.has_label && !(dirty && emptyMeansNoFile(item));
    const sig = item && !labelsError && !n ? `${hasFile}|${editable()}|${item.id}` : '';
    boxEmpty.classList.toggle('hidden', n > 0);
    if (boxEmpty._sig === sig) return;
    boxEmpty._sig = sig;
    if (!sig) { boxEmpty.replaceChildren(); return; }
    boxEmpty.replaceChildren(h('span', { class: 'grow' }, hasFile ? 'No boxes — empty label file (background frame)' : 'No boxes yet — no label file'),
      h('button', { class: 'btn btn-ghost btn-sm', disabled: !editable(), onClick: () => setFileState(!hasFile),
        'data-tip': hasFile ? 'Remove the empty label file: the image counts as unlabeled again' : 'Write an empty label file: this image shows no objects (a negative sample)' },
        hasFile ? 'Remove file' : 'Mark as background'));
  }
  async function setFileState(empty) {
    if (!editable() || boxes.length) return;
    const target = item;
    if (await flush() !== 'clean' || target !== item || boxes.length || !editable()) return;
    try {
      const body = { boxes: [], base_version: version }; if (!empty) body.remove_if_empty = true;
      const res = await fovea.api.put(labelUrl(target.id), body);
      createdFiles.delete(histKey(target.id));           // either way a deliberate state, not an accident to tidy up
      forgetLabels(target.id);
      if (target !== item) return;
      version = res.version ?? null; baseBoxes = [];
      Object.assign(item, res.item); delete item.boxes;
      { const { boxes: _omit, ...row } = res.item; cursor.update(row); }
      fovea.bus.emit('labels:changed', { item: res.item });
      renderList(); renderInfo(); filmTouch();
      note(empty ? 'Marked as background — an empty label file was written' : 'Label file removed — the image counts as unlabeled');
    } catch (e) { note(e.message, 'error'); }
  }
  function scrollListTo(i, instant) {
    const top = i * ROW_H, vt = vlist.scrollTop, vh = vlist.clientHeight;
    if (top < vt) vlist.scrollTop = top; else if (top + ROW_H > vt + vh) vlist.scrollTop = top + ROW_H - vh;
    if (!instant) renderList();
  }

  // Proposals panel
  let propListEl = null;
  function renderProps() {
    propSec.replaceChildren();
    const st = propStats, nVis = visProps.length;
    const runIds = [...new Set(props.filter(isPending).map(p => p.run_id).filter(x => x != null))];
    const src = runIds.length === 1 && runs[runIds[0]] ? `${runs[runIds[0]].model || runs[runIds[0]].provider || ''}` : runIds.length > 1 ? `${runIds.length} runs` : '';
    propSec.appendChild(h('h4', 'Proposals', st.pending ? h('span', { class: 'badge accent' }, String(st.pending)) : null, h('span', { class: 'spacer' }), src ? h('span', { class: 'xs faint truncate ann-prop-src', title: src }, src) : null));
    if (!st.pending) {
      const decided = props.filter(p => !isPending(p)).length;
      // Worded as the AI panel words it ("Press I to propose…"), with the same key badges.
      propSec.appendChild(h('div', { class: 'xs faint ann-prop-empty' }, ...(decided
        ? [`All ${decided} decided. Press `, ui.kbd('I'), ' to propose again, or ', ui.kbd('C'), ' to use the previous frame.']
        : ['None for this image. Press ', ui.kbd('I'), ' to propose with the current model, or ', ui.kbd('C'), ' to use the previous frame.'])));
      propListEl = null; return;
    }
    propSec.appendChild(propStatsRow());
    const thrLabel = h('span', { class: 'xs mono' }, thr.toFixed(2));
    propSec.appendChild(h('div', { class: 'ann-thr' }, h('span', { class: 'xs faint' }, 'Show ≥'),
      h('input', { type: 'range', class: 'slider', min: 0, max: 1, step: 0.01, value: thr, 'aria-label': 'Proposal confidence threshold', onInput: (e) => { thrLabel.textContent = Number(e.target.value).toFixed(2); setThrLive(Number(e.target.value)); } }),
      thrLabel, thrStepBtn(-1), thrStepBtn(1)));
    const ed = editable();
    propSec.appendChild(h('div', { class: 'row ann-prop-actions' },
      h('button', { class: 'btn btn-sm btn-primary grow', disabled: !ed || !nVis, onClick: () => acceptAllVisible(), 'data-tip': 'Accept all shown (⇧Enter)' }, icon('check', 12), `Accept ${nVis}`),
      h('button', { class: 'btn btn-sm grow', disabled: !ed || !nVis, onClick: () => rejectAllVisible(), 'data-tip': 'Reject all shown (⇧Del)' }, icon('x', 12), `Reject ${nVis}`)));
    propListEl = h('div', { class: 'ann-prop-list', role: 'listbox', 'aria-label': 'Proposals' });
    const shown = visProps.slice(0, 150);
    for (const p of shown) {
      const i = props.indexOf(p), active = sel && sel.t === 'prop' && sel.i === i;
      const desc = p.kind === 'swap' && p.refBox ? `${className(names, p.refBox[0])} → ${className(names, p.b[0])}`
        : p.kind === 'loose' && p.refBox ? `tighter box for #${boxes.indexOf(p.refBox) + 1}`
        : p.kind === 'spurious' && p.refBox ? `#${boxes.indexOf(p.refBox) + 1} no support`
        : p.kind === 'refine' && p.refBox ? `refine #${boxes.indexOf(p.refBox) + 1}`
        : className(names, p.b[0]);
      const extra = p.stale ? 'label changed since' : p.conflict >= 0 ? `⇄ conflicts #${p.conflict + 1}` : (p.reason || '');
      propListEl.appendChild(h('div', { class: cls('prop-row', active && 'active'), role: 'option', 'aria-selected': String(!!active), title: [KIND_LABEL[p.kind], p.label, p.reason].filter(Boolean).join(' · '), onClick: () => selectProp(p) },
        h('span', { class: 'swatch', style: `background:${classColor(p.b[0])}` }),
        h('span', { class: 'mono xs conf' }, p.conf != null ? p.conf.toFixed(2) : '—'),
        GLYPH[p.kind] && p.kind !== 'detect' ? h('span', { class: 'glyph', title: KIND_LABEL[p.kind] }, GLYPH[p.kind]) : null,
        h('span', { class: 'name' }, desc, extra ? h('span', { class: 'xs faint' }, ' ' + extra) : null),
        h('button', { class: 'btn btn-ghost btn-sm btn-icon', tabindex: '-1', 'aria-label': 'Accept', disabled: !ed, onClick: (e) => { e.stopPropagation(); acceptProp(p); } }, icon('check', 12)),
        h('button', { class: 'btn btn-ghost btn-sm btn-icon', tabindex: '-1', 'aria-label': 'Reject', disabled: !ed, onClick: (e) => { e.stopPropagation(); rejectProp(p); } }, icon('x', 12))));
    }
    if (visProps.length > shown.length) propListEl.appendChild(h('div', { class: 'xs faint', style: 'padding:4px 6px' }, `and ${visProps.length - shown.length} more — raise the threshold or accept in bulk`));
    propSec.appendChild(propListEl);
    const a = propListEl.querySelector('.prop-row.active'); if (a && a.scrollIntoView) a.scrollIntoView({ block: 'nearest' });
  }
  /** The { and } keys as buttons: one step of 0.05 down or up. The panel is redrawn by the step, so a button the user
      Tabbed to is focused again in the new panel (Enter / Space keep stepping). */
  function thrStepBtn(d) {
    const k = d < 0 ? '{' : '}', what = d < 0 ? 'Lower' : 'Raise';
    return h('button', { type: 'button', class: 'kbd ann-thr-step', 'data-step': k, 'aria-label': `${what} threshold`, 'data-tip': `${what} the threshold by 0.05 (${k})`,
      disabled: d < 0 ? thr <= 0 : thr >= 1,
      onClick: (e) => {
        const viaTab = tabFocused === e.currentTarget;
        stepThr(d);
        const b = propSec.querySelector(`.ann-thr-step[data-step="${k}"]`);
        if (b && viaTab && !b.disabled) { b.focus({ preventScroll: true }); tabFocused = b; }
      } }, k);
  }
  function stepThr(d) { setThr(thr + 0.05 * d); note(`Showing proposals ≥ ${thr.toFixed(2)}`); }
  /** "7 shown · 3 below 0.25 · 2 already labeled", and a way to decide the ones not shown (else they stay pending for
      ever). Its height is fixed: the threshold slider under it never moves while it is dragged. */
  function propStatsRow() {
    const st = propStats, bits = [`${visProps.length} shown`];
    if (st.below) bits.push(`${st.below} below ${thr.toFixed(2)}`);
    if (st.dup) bits.push(`${st.dup} already labeled`);
    if (st.hiddenCls) bits.push(`${st.hiddenCls} hidden class`);
    const nHidden = st.below + st.dup;
    return h('div', { class: 'row ann-prop-stats' }, h('span', { class: 'xs faint grow', title: bits.join(' · ') }, bits.join(' · ')),
      nHidden ? h('button', { class: 'btn btn-ghost btn-sm', disabled: !editable(), 'data-tip': `Reject the ${nHidden} proposal${nHidden === 1 ? '' : 's'} not shown here: under the threshold, or duplicates of a box of the same class`, onClick: () => rejectHidden() }, 'Reject hidden') : null);
  }
  const renderPropList = () => { if (!propListEl) return; propListEl.querySelectorAll('.prop-row').forEach((r) => r.classList.remove('active')); renderProps(); };
  let thrRaf = 0, thrPending = null;
  function applyThrLive() {
    if (thrRaf) { cancelAnimationFrame(thrRaf); thrRaf = 0; }
    if (thrPending == null) return;
    const v2 = Math.round(thrPending * 100) / 100; thrPending = null;
    if (v2 !== thr) { thr = v2; LS.set(`fovea.ann.thr.${ds.id}`, v2); recalcProps(); invalidate(); updateHud(); const t = propSec.querySelector('.ann-prop-stats'); if (t) t.replaceWith(propStatsRow()); }
  }
  function setThrLive(v) { thrPending = v; if (!thrRaf) thrRaf = requestAnimationFrame(() => { thrRaf = 0; applyThrLive(); }); }
  // A released slider redraws the panel (the list follows the new threshold). The value it was moved to is applied
  // first — an arrow key fires input and change in one go, before the frame that would apply it — and a slider
  // that had keyboard focus keeps it, so the next arrow moves it again instead of changing the image.
  propSec.addEventListener('change', (e) => {
    if (!e.target || e.target.type !== 'range') return;
    const focused = document.activeElement === e.target, viaTab = focused && tabFocused === e.target;
    applyThrLive(); renderProps();
    if (focused) { const s = propSec.querySelector('input[type="range"]'); if (s) { s.focus({ preventScroll: true }); if (viaTab) tabFocused = s; } }
    fovea.bus.emit('annotate:threshold', { value: thr });
  });

  const FILTER_NAMES = { labeled: { nobox: 'No boxes', 1: 'Labeled', labeled: 'Labeled', 0: 'Unlabeled', unlabeled: 'Unlabeled', empty: 'Empty label' }, review: { none: 'Unreviewed', any: 'Reviewed', approved: 'Approved', flagged: 'Flagged', excluded: 'Excluded' }, ai: { pending: 'AI proposals pending', audit: 'AI suspects', none: 'No AI proposals' } };
  function filterChips() {
    const out = [];
    for (const [k, v] of Object.entries(filters)) {
      if (!v) continue;
      let t = v;
      if (k === 'split') t = `Split: ${v}`;
      else if (k === 'cls') t = 'Class: ' + v.split(',').map(c => className(names, Number(c))).join(', ');
      else if (k === 'only_cls') t = 'Only: ' + v.split(',').map(c => className(names, Number(c))).join(', ');
      else if (k === 'exclude_review') t = 'Not ' + v.split(',').map(x => ((FILTER_NAMES.review || {})[x] || x).toLowerCase()).join(', ');
      else if (k === 'min_boxes') t = `≥ ${v} boxes`;
      else if (k === 'max_boxes') t = `≤ ${v} boxes`;
      else if (k === 'has_label') t = v === '0' || v === 'false' ? 'No label file' : 'Has a label file';
      else if (k === 'seed') continue;           // the permutation of a random sort, not a filter
      else if (k === 'labeled' || k === 'review' || k === 'ai') t = (FILTER_NAMES[k] || {})[v] || `${k}: ${v}`;
      else if (k === 'issue') t = 'Issue: ' + v.split(',').map(c => (c === 'any' ? 'any' : issueLabel(c))).join(', ');
      else if (k === 'q') t = `Name contains “${v}”`;
      else if (k === 'seq') t = `Sequence: ${v}`;
      else if (k === 'ids') t = `${fmt.num(v.split(',').length)} selected images`;
      else if (k === 'set') t = 'Saved selection';
      else if (k === 'attr') t = attrFilterText(annSch, v);
      out.push(ui.chip(t, { cls: 'outline' }));
    }
    return out;
  }
  function renderWorkingSet() {
    const chips = filterChips();
    const total = cursor.total;
    if (total != null) sizeIdx();          // as soon as the list's size is known: before the first image is shown
    wsBody.replaceChildren(...[
      h('div', { class: 'small' }, total == null ? 'Loading…' : `${fmt.num(total)} image${total === 1 ? '' : 's'}`, chips.length ? h('span', { class: 'faint' }, ' matching') : h('span', { class: 'faint' }, ' — whole dataset'),
        sort === 'flag' || sort === 'audit' ? h('span', { class: 'faint', 'data-tip': sort === 'flag' ? 'Ordered by the score of the QA flags in the filter' : 'Ordered by the label-audit score' },
          ` · ${order === 'desc' ? 'most' : 'least'} ${sort === 'flag' ? 'suspicious' : 'suspect'} first`) : null),
      chips.length ? h('div', { class: 'row wrap gap-4' }, chips) : null,
      h('div', { class: 'row wrap gap-4 ann-ws-actions' },
        seqReturn ? h('button', { class: 'btn btn-sm', 'data-tip': 'Return to the list you were working through before opening this sequence', onClick: () => backFromSequence() }, icon('chevronLeft', 12), 'Back to list') : null,
        chips.length ? h('button', { class: 'btn btn-sm', onClick: () => setFilters({}) }, icon('x', 12), 'Clear filters') : null,
        h('button', { class: 'btn btn-ghost btn-sm', 'data-tip': 'Re-run the query: picks up images that started matching since the list was opened', onClick: () => setFilters(filters) }, icon('refresh', 12), 'Refresh'),
        h('a', { class: 'btn btn-ghost btn-sm', href: exploreHref() }, icon('grid', 12), 'Explore'))].filter(Boolean));
  }
  function exploreHref() {
    const p = new URLSearchParams(); for (const [k, v] of Object.entries(filters)) if (v) p.set(k, v);
    // Explore opens a list in the order its filters imply by the same rule (initialSort), so a sort is passed on
    // when it is another one — "Index" over a QA-flag list included — or one Explore would not imply by itself.
    if (!sortImplied || (sort !== 'id' && sort !== 'set')) p.set('sort', sort);
    if (order !== defaultOrder(sort)) p.set('order', order);
    if (item) p.set('img', item.id);
    const s = p.toString(); return `/d/${ds.id}/explore${s ? '?' + s : ''}`;
  }
  /** Re-open the working set with other filters, staying on the current image when it is still in the list. */
  async function setFilters(f, { keepReturn = false, fallbackId = null } = {}) {
    if (await flush() !== 'clean' && dirty) { ui.toast('Save the edits on this image first', { type: 'error' }); return false; }
    if (!keepReturn) seqReturn = null;
    filters = { ...f }; for (const k of Object.keys(filters)) if (!filters[k]) delete filters[k];
    // The order these filters imply; a flag sort the link named, once its flag filter is gone, has nothing to rank by.
    if (sortImplied || (sort === 'flag' && !flagSortable(filters))) { sort = initialSort(filters); order = defaultOrder(sort); sortImplied = true; }
    forgetMembership(); prevOf.clear();          // outside a sequence, "previous" is the previous image of the list
    const keep = item && item.id;
    cursor = new fovea.ImageCursor({ dsId: ds.id, filters, sort, order, pageSize: 200, boxes: false });
    idx = -1; shownIdx = -1; want = -1;
    try { await cursor.loadPage(0); } catch (e) { if (!listGone(e)) showStageMsg(e.message, true); return false; }
    renderWorkingSet();
    if (!cursor.total) { emptyState(); return true; }
    let pos = keep ? await cursor.positionOf(keep) : 0;
    if (pos < 0 && fallbackId != null) pos = await cursor.positionOf(fallbackId);
    if (pos < 0) pos = 0;
    await goTo(pos, { reload: true });
    return true;
  }
  /** A saved selection (`set=` token) expires after a day: say so instead of showing a bare error. */
  function listGone(e) {
    if (!filters.set || !(e && (e.status === 404 || e.status === 410 || e.status === 400))) return false;
    cursor.total = 0;
    emptyState('This saved selection has expired or no longer exists — selections are kept for a day. Make it again in Explore.');
    return true;
  }
  function emptyState(msg) {
    item = null; img = null; boxes = []; props = []; sel = null; recalcProps(); invalidate(true);
    renderList(); renderProps(); renderInfo(); renderWorkingSet(); renderFilm();
    stageMsg.replaceChildren(h('div', { class: 'ann-empty' }, h('div', null, msg || (filters.set ? 'This saved selection is empty' : 'No images match the current filter')),
      h('div', { class: 'row', style: 'justify-content:center' }, Object.keys(filters).length ? h('button', { class: 'btn btn-sm', onClick: () => setFilters({}) }, 'Clear filters') : null,
        h('a', { class: 'btn btn-sm', href: exploreHref() }, 'Back to Explore'))));
    stageMsg.classList.remove('hidden');
  }

  function renderInfo() {
    renderWorkingSet();
    reviewRow.replaceChildren();
    const rv = item && item.review;
    ['approved', 'flagged', 'excluded'].forEach(s => reviewRow.appendChild(h('button', { class: cls('btn btn-sm review-btn', s, rv && rv.status === s && 'active'), disabled: !item, 'aria-pressed': String(!!(rv && rv.status === s)), onClick: () => review(s, { toggle: true }) }, icon(REVIEW[s].icon, 12), REVIEW[s].label)));
    infoKv.replaceChildren();
    if (!item) { nameEl.replaceChildren(); nameEl._path = null; if (document.activeElement !== idxInput) idxInput.value = cursor.total ? '' : '0 / 0'; issuesRow.replaceChildren(); flagFocus = null; renderFlagInfo(); return; }
    const total = cursor.total;
    const rows = [['Name', item.rel_path.split('/').pop()], ['Split', item.split || '—'], ['Size', item.width ? `${item.width} × ${item.height}` : (img ? `${W} × ${H}` : '—')], ['Seq', item.seq || '—'], ['Position', `${fmt.num(idx + 1)} of ${fmt.num(total)}`], ['ID', `#${item.id}`]];
    if (regions.length) {
      const byKind = {}; regions.forEach(g => { byKind[g.kind] = (byKind[g.kind] || 0) + 1; });
      rows.push(['Regions', Object.entries(byKind).map(([k, n]) => `${n} ${regionLabel(annSch, k).toLowerCase()}`).join(', ')]);
    }
    rows.forEach(([k, v]) => { infoKv.appendChild(h('b', { class: 'faint' }, k)); infoKv.appendChild(h('span', { class: 'mono truncate', title: v }, v)); });
    issuesRow.replaceChildren();
    (item.issues || []).forEach(c => issuesRow.appendChild(ui.chip(issueLabel(c), { type: 'warn', icon: 'alert' })));
    if (flagFocus && !(item.flags || []).includes(flagFocus.code)) { flagFocus = null; flagLoad++; updateHud(); invalidate(); renderList(); }   // re-checked away
    (item.flags || []).forEach((c) => {
      const on = !!flagFocus && flagFocus.code === c;
      // A chip that can show what differs is a toggle button (dashed boxes on the image, a line of detail below).
      const chipEl = !FLAG_DETAIL.has(c) ? ui.chip(issueLabel(c), { type: 'info', icon: 'info', title: issueLabel(c) })
        : h('button', { class: cls('chip info clickable ann-flag-chip', on && 'active'), type: 'button', 'aria-pressed': String(on),
          'data-tip': on ? 'Hide what differs (Esc)' : 'Show what differs on the image', onClick: () => focusFlag(c) }, icon('info', 12), issueLabel(c));
      // A frame missing a box its neighbours both have: the fix is one click away (POST …/fill-gap → proposals).
      issuesRow.appendChild(c !== 'seq_gap' ? chipEl : h('span', { class: 'ann-flag-fix' }, chipEl,
        h('button', { class: 'btn btn-sm', disabled: !editable() || gapBusy, 'data-tip': 'Propose the boxes the frames before and after have and this one lacks (Enter accepts)', onClick: () => fillGap() }, icon('film', 12), 'Fill from neighbours')));
    });
    renderFlagInfo();
    // Middle ellipsis: the frame number and extension at the end of the path always stay visible.
    if (nameEl._path !== item.rel_path) { nameEl._path = item.rel_path; nameEl.replaceChildren(ui.midText ? ui.midText(item.rel_path, { tail: 24 }) : item.rel_path); nameEl.title = item.rel_path; }
    sizeIdx();
    // Never overwrite what the user is typing in the position box.
    if (document.activeElement !== idxInput) idxInput.value = idxText();
  }
  /** The position box fits "246,800 / 246,800" for this list (digits are tabular in it), so it never clips the
      total and never changes width while stepping through the list — only when the list's size changes. */
  let idxSized = '';
  function sizeIdx() {
    const t = fmt.num(cursor.total || 0), sig = `${t}`;
    if (sig === idxSized) return;
    idxSized = sig;
    idxInput.style.width = `calc(${Math.max(11, 2 * t.length + 3)}ch + 24px)`;
  }

  // ---------------------------------------------------------------- QA flag details (the chips under Image)
  // A pressed chip shows what differs, from GET …/images/{id}/qa: the rows it names get an amber ring (the first is
  // selected), and the twin's / near-identical image's / frame before's boxes, or where the frames around put a box,
  // a static amber dashed ghost — legend in the HUD, a line of detail with the frames to compare under the chips.
  // Pressing it again, Esc or another image clears it.
  let flagLoad = 0;
  async function focusFlag(code, { refresh = false } = {}) {
    if (!refresh && flagFocus && flagFocus.code === code && item && flagFocus.forId === item.id) { clearFlagFocus(); return; }
    if (!item) return;
    const target = item, my = ++flagLoad;
    const done = refresh ? () => {} : waitNote('Reading the flag…');
    try {
      const res = await fovea.api.get(`/api/datasets/${ds.id}/images/${target.id}/qa`);
      const f = (res.flags || []).find(x => x.code === code);
      if (!f) {
        done();
        if (destroyed || my !== flagLoad || item !== target) return;
        // Checked again since the chips were drawn: show the flags as they are now.
        const codes = (res.flags || []).map(x => x.code);
        item.flags = codes; cursor.update({ id: target.id, flags: codes }); filmSetRow(target.id, { flags: codes });
        if (flagFocus && flagFocus.code === code) clearFlagFocus(); else renderInfo();
        if (!refresh) note('This flag is gone — the image was checked again', 'info');
        return;
      }
      const t = flagTargets(f), d = f.detail || {};
      let otherId = t.other;
      if (code === 'near_dup' && otherId == null) {
        const sim = await fovea.api.get(`/api/datasets/${ds.id}/images/${target.id}/similar`, { k: 1 });
        const s = (sim.items || [])[0]; otherId = s ? s.id : null;
      }
      let other = null;
      if (otherId != null) {
        const [lab, rows] = await Promise.all([fovea.api.get(labelUrl(otherId)), fovea.api.post(`/api/datasets/${ds.id}/images/by-ids`, { ids: [otherId] }).catch(() => null)]);
        const row = rows && (rows.items || [])[0];
        other = { id: otherId, path: (row && row.rel_path) || d.twin_path || `#${otherId}`, split: row ? row.split : null, seq: row ? row.seq : null, boxes: (lab.boxes || []).map(b => b.slice(0, 5)) };
      }
      done();
      if (destroyed || my !== flagLoad || item !== target) return;
      const rel = code === 'seq_flicker' || code === 'seq_count_spike' ? 'before' : code === 'near_dup' ? 'similar' : 'twin';
      flagFocus = { code, forId: target.id, flag: f, rows: t.rows, refs: t.rows.map(r => boxes[r]).filter(Boolean),
        boxes: t.boxes.length ? t.boxes : other ? other.boxes : [], other: t.boxes.length ? null : other, rel,
        tag: code === 'seq_gap' ? 'missing?' : code === 'seq_jump' ? 'expected' : rel };
      // The first box it names is selected (a digit fixes a flicker, a drag a jump) — not on a refresh after a save.
      if (!refresh && flagFocus.refs.length) setSel({ t: 'box', i: boxes.indexOf(flagFocus.refs[0]) });
      renderInfo(); renderList(); updateHud(); invalidate();
    } catch (e) { done(); if (!destroyed && my === flagLoad && item === target && !refresh) note('Could not read this flag: ' + e.message, 'error'); }
  }
  /** A ghost box that shows a difference: where the frames around put an object, or another image's box with no box
      of its class here (IoU ≥ 0.5). */
  const flagDiffers = (b) => !flagFocus || !flagFocus.other || !boxes.some(x => x[0] === b[0] && iou(x, b) >= 0.5);
  function clearFlagFocus() {
    if (!flagFocus) return;
    flagFocus = null; flagLoad++;
    renderInfo(); renderList(); updateHud(); invalidate();
  }
  /** The flag was checked again after a save (checkFlags): show its fresh detail, or drop it when it is gone. */
  function refreshFlagFocus() {
    if (!flagFocus || !item || flagFocus.forId !== item.id) return;
    const code = flagFocus.code;
    if (!(item.flags || []).includes(code)) { clearFlagFocus(); return; }
    focusFlag(code, { refresh: true });           // the ghost on screen stays until the fresh one replaces it
  }
  function renderFlagInfo() {
    const fg = flagFocus && item && flagFocus.forId === item.id ? flagFocus : null;
    flagInfo.classList.toggle('hidden', !fg);
    if (!fg) { flagInfo.replaceChildren(); return; }
    const d = fg.flag.detail || {}, nm = (c) => className(names, c), nRows = fg.refs.length;
    const pct = (v) => `${(v * 100).toFixed(v < 0.01 ? 2 : 1)}%`;
    const oth = fg.other, oName = oth ? baseOf(oth.path) : '';
    const here = boxes.length;
    let text = '';
    switch (fg.code) {
      case 'seq_jump': text = `${plural(nRows, 'box', 'boxes')} moved off the path the frames before and after agree on (ringed). Dashed: where those frames put it.`; break;
      case 'seq_gap': text = 'An object labeled in the frames before and after has no box here. Dashed: where it would be.'; break;
      case 'seq_flicker': text = (d.rows || []).slice(0, 3).map(x => `${nm(x.cls)} here, ${nm(x.expected)} in the frames around it`).join(' · ') + ` (ringed).${oth ? ` Dashed: the frame before, ${oName}.` : ''}`; break;
      case 'seq_count_spike': text = `${plural(d.n ?? here, 'box', 'boxes')} here, ${fmt.num(d.prev ?? 0)} in the frame before and ${fmt.num(d.next ?? 0)} after.${oth ? ` Dashed: the frame before (${plural(oth.boxes.length, 'box', 'boxes')}).` : ''}`; break;
      case 'shape_outlier': text = (d.rows || []).slice(0, 3).map(x => `${nm(x.cls)}: ${x.kind === 'aspect' ? `${x.value > x.typical ? 'wider' : 'taller'} than usual (w/h ${x.value} vs ${x.typical})` : `${x.value > x.typical ? 'larger' : 'smaller'} than usual (${pct(x.value)} of the image vs ${pct(x.typical)})`}`).join(' · ') + ' (ringed).'; break;
      case 'dup_label_mismatch': text = `A near-identical image is labeled differently${d.reason && d.reason !== 'count' ? ` (${d.reason === 'class' ? 'other classes' : 'other boxes'})` : ''}: ${plural(here, 'box', 'boxes')} here, ${fmt.num(oth ? oth.boxes.length : d.twin_n_boxes ?? 0)} there.${oth ? ' Dashed: its boxes.' : ''}`; break;
      case 'leak_split': text = `A near-identical image is in ${(d.splits || []).join(', ') || 'another split'}.`; break;
      case 'dup_same_role': text = `The same frame is also in ${(d.splits || []).join(', ') || 'another folder'}, a folder of the same role${d.role ? ` (${d.role})` : ''}.`; break;
      case 'near_dup': text = oth ? `Near-identical to ${oName}${d.exact ? ' (an exact copy)' : ''}.` : 'No near-identical image was found any more.'; break;
      default: text = '';
    }
    if (oth && (fg.code === 'leak_split' || fg.code === 'dup_same_role' || fg.code === 'near_dup')) text += ` Dashed: its ${plural(oth.boxes.length, 'box', 'boxes')}${oth.boxes.length !== here ? ` (${fmt.num(here)} here)` : ''}.`;
    // A twin's / near-identical image's boxes this one lacks (not yet proposed): one click makes them proposals. Not
    // for the frame before — its objects moved; C proposes that frame's boxes.
    const missing = oth && fg.rel !== 'before' ? oth.boxes.filter(b => flagDiffers(b) && !props.some(p => isPending(p) && p.b[0] === b[0] && iou(p.b, b) >= PASTE_IOU)) : [];
    const btn = (label, ic, tip, fn) => h('button', { class: 'btn btn-sm', type: 'button', 'data-tip': tip, onClick: fn }, icon(ic, 12), label);
    const nav = [];
    if (d.prev_id != null) nav.push(btn('Frame before', 'chevronLeft', 'Open the frame before this one', () => openFlagImage(d.prev_id, 'seq')));
    if (d.next_id != null) nav.push(btn('Frame after', 'chevronRight', 'Open the frame after this one', () => openFlagImage(d.next_id, 'seq')));
    if (oth && fg.rel !== 'before') nav.push(btn('Open it', 'copy', oth.path, () => openFlagImage(oth.id, 'pair')));
    if (missing.length && editable()) nav.push(btn(`Propose ${plural(missing.length, 'box', 'boxes')} it has`, 'sparkles', `Its boxes with no box of the same class here, as proposals (Enter accepts) — from ${oName}`, () => proposeFrom(missing, oName, fg.rel)));
    flagInfo.replaceChildren(...[h('div', text), nav.length ? h('div', { class: 'row wrap gap-4 mt-8' }, nav) : null].filter(Boolean));
  }
  /** The other image a flag names: in this working set → go there; else Annotate opened on it (this image and that
      one for a twin, its sequence for a neighbouring frame) as a new history entry, so Back returns to this list. */
  async function openFlagImage(id, how) {
    if (!item) return;
    const from = item;
    let pos = cursor.indexOfId(id);
    if (pos < 0) { try { pos = await cursor.positionOf(id); } catch (e) { pos = -1; } }
    if (destroyed || item !== from) return;
    if (pos >= 0) { stopAuto(); goTo(pos); return; }
    const q = new URLSearchParams(how === 'seq' && from.seq ? { seq: from.seq, sort: 'name' } : { ids: `${from.id},${id}` });
    q.set('img', String(id));
    fovea.router.navigate(`/d/${ds.id}/annotate?${q.toString()}`);
  }
  /** The other image's boxes that have no box of the same class here, as local proposals (nothing written until accepted). */
  function proposeFrom(list, name, rel) {
    if (!editable() || !list.length) return;
    const add = list.filter(b => !props.some(p => isPending(p) && p.b[0] === b[0] && iou(p.b, b) >= PASTE_IOU));
    if (!add.length) { note('Those proposals are already shown'); return; }
    const made = addLocalProps(add, 'copy', rel === 'before' ? 'frame before' : rel === 'similar' ? 'near-identical image' : 'twin');
    const shown = made.filter(p => p.vis);
    if (shown.length) selectProp(shown[0]);
    note(`${plural(add.length, 'proposal')} from ${name} — Enter accepts, ⇧Enter all, Del rejects`, 'info', true);
  }
  /** seq_gap fix: interpolated proposals for the objects the nearest labeled frames on both sides have and this one
      lacks (core/qa.py fill_gap reads the label files, so this image is saved first). */
  let gapBusy = false;
  async function fillGap() {
    if (!editable() || gapBusy) return;
    const target = item, token = navToken;
    if (await flush() !== 'clean') { note(notSavedMsg(), 'warn'); return; }
    if (navToken !== token || item !== target || !editable()) return;
    gapBusy = true; renderInfo();
    const done = waitNote('Filling from the neighbouring frames…');
    try {
      const res = await fovea.api.post(`/api/datasets/${ds.id}/images/${target.id}/fill-gap`, {});
      done();
      if (destroyed || navToken !== token || item !== target) return;
      const preds = (res.preds || []).filter(p => p.image_id == null || p.image_id === target.id);
      if (!preds.length) { note(res.message || 'Nothing to fill here', 'info'); return; }
      // The server replaced this image's earlier pending interpolations with these.
      const incoming = preds.map(normProp);
      props = props.filter(p => p.local || !isPending(p) || p.kind !== 'interp').concat(incoming);
      if (sel && sel.t === 'prop') sel = null;
      propsChanged();
      const shown = incoming.filter(p => p.vis);
      if (shown.length) selectProp(shown[0]);
      fovea.bus.emit('proposals:changed', { dataset: ds.id, source: 'annotate' });
      const from = (preds[0].reason || '').replace(/^Filled from /, '');
      note(`${preds.length} box${preds.length === 1 ? '' : 'es'} proposed from ${from || 'the neighbouring frames'}` +
        (shown.length < preds.length ? ` · ${preds.length - shown.length} hidden here` : '') + ' — Enter accepts, ⇧Enter all', 'info', true);
    } catch (e) { done(); if (!destroyed && navToken === token && item === target) note('Fill from neighbours failed: ' + e.message, 'error'); }
    finally { gapBusy = false; if (!destroyed && item === target) renderInfo(); }
  }
  function renderSeq() {
    seqBody.replaceChildren();
    const ghostSw = ui.switchBtn(opts.ghost, () => toggleGhost(), { label: 'Previous-frame ghost (`)' });
    const propSw = ui.switchBtn(opts.propagate, (v) => { opts.propagate = v; LS.set('fovea.ann.propagate', v ? '1' : '0'); }, { label: 'Propose boxes on the next new frame' });
    seqBody.appendChild(h('div', { class: 'xs faint truncate', title: item && item.seq || '' }, item && item.seq ? `Sequence ${item.seq}` : 'No sequence for this image'));
    seqBody.appendChild(optRow('Filmstrip (\\)', ui.switchBtn(filmShown(), () => toggleFilm(), { label: 'Filmstrip (\\)' }), 'Every frame of this sequence along the bottom of the image — , and . step through them'));
    seqBody.appendChild(optRow('Previous-frame ghost (`)', ghostSw, 'Outline the previous frame\'s boxes'));
    seqBody.appendChild(optRow('Propose boxes on the next new frame', propSw, 'Stepping → onto a frame with no label file shows the previous frame\'s boxes as proposals'));
    const kfTxt = keyframe ? `Keyframe: ${keyframe.name}` : 'Keyframe: none — K marks this frame';
    seqBody.appendChild(h('div', { class: 'row ann-kf-row' }, h('span', { class: 'xs truncate grow', title: kfTxt }, kfTxt),
      keyframe ? h('button', { class: 'btn btn-ghost btn-sm', onClick: () => { keyframe = null; seqStatus = ''; seqAction = null; renderSeq(); drawTiles(); } }, 'Clear') : null));
    const canInterp = sameTrack(keyframe, item) && keyframe.id !== item.id;
    if (keyframe && item && keyframe.seq === item.seq && !sameTrack(keyframe, item)) seqBody.appendChild(h('div', { class: 'xs faint' }, `This frame is in ${item.split || 'another split'}; the keyframe in ${keyframe.split || 'another'} — interpolation needs both in one split`));
    seqBody.appendChild(h('div', { class: 'row' },
      h('button', { class: 'btn btn-sm', disabled: !item || !item.seq, onClick: () => keyframeKey() }, icon('bookmark', 12), keyframe && keyframe.id === (item && item.id) ? 'Unmark (K)' : 'Keyframe (K)'),
      h('button', { class: 'btn btn-sm btn-primary', disabled: !canInterp, onClick: () => interpolate(), 'data-tip': 'Proposals on every frame between the keyframe and this one' }, 'Interpolate')));
    // seqStatus: a line, or lines of which {text, warn} ones say what an interpolation left out.
    for (const x of [].concat(seqStatus || [])) if (x) seqBody.appendChild(h('div', { class: cls('xs', x.warn ? 'warn-text' : 'faint') }, typeof x === 'string' ? x : x.text));
    if (seqAction) seqBody.appendChild(h('div', { class: 'row' }, h('button', { class: 'btn btn-sm ann-seq-act', 'data-tip': seqAction.tip || null, onClick: () => seqAction && seqAction.run() }, icon(seqAction.icon, 12), h('span', { class: 'truncate' }, seqAction.label))));
  }
  function renderPointOpts() {
    pointOpts.replaceChildren();
    pointOpts.classList.toggle('hidden', tool !== 'point');
    const wl = h('span', { class: 'xs mono' }, opts.pointW.toFixed(2)), hl = h('span', { class: 'xs mono' }, opts.pointH.toFixed(2));
    const w = h('input', { type: 'range', class: 'slider', min: 0.02, max: 0.6, step: 0.005, value: opts.pointW, 'aria-label': 'Point box width', onInput: (e) => { opts.pointW = Number(e.target.value); wl.textContent = opts.pointW.toFixed(2); overlayOnly(); } });
    const hh = h('input', { type: 'range', class: 'slider', min: 0.02, max: 0.9, step: 0.005, value: opts.pointH, 'aria-label': 'Point box height', onInput: (e) => { opts.pointH = Number(e.target.value); hl.textContent = opts.pointH.toFixed(2); overlayOnly(); } });
    pointOpts.appendChild(h('div', { class: 'row small' }, h('span', { style: 'width:44px' }, 'Width'), w, wl));
    pointOpts.appendChild(h('div', { class: 'row small' }, h('span', { style: 'width:44px' }, 'Height'), hh, hl));
    pointOpts.appendChild(h('div', { class: 'xs faint' }, 'Click drops a fixed-size box. Alt+wheel: width, Alt+⇧+wheel: height.'));
  }
  let hudSig = '';
  function updateHud() {
    const chips = [[tool === 'sam' ? 'click-to-box' : tool === 'box' ? `box · ${opts.boxMode === 'click2' ? (corner ? '2-click 1/2' : '2-click') : 'drag'}` : tool, corner ? 'mono accent' : 'mono']];
    if (img) chips.push([`${W}×${H}${imgFull ? '' : ' · preview'}`, 'mono']);
    const sNow = selectedItem();         // the class for new boxes; named so when a box of another class is selected
    chips.push([`${sNow && itemCls(sNow) !== activeCls ? 'new boxes: ' : ''}${activeCls} ${className(names, activeCls)}`, 'mono', classColor(activeCls)]);
    if (classBuf) chips.push([`class ${classBuf}_`, 'mono accent']);
    const sBox = sNow && sNow.t === 'box' ? boxes[sNow.i] : null;
    if (sBox && sBox.attrs) chips.push([attrText(annSch, sBox.attrs), 'ann-attrs']);    // occlusion, truncation … of the selected box
    if (hiddenAll) chips.push([peek ? 'Boxes hidden while O is held' : 'Boxes hidden (O)', 'warn']);
    if (hiddenCls.size) chips.push([`${hiddenCls.size} class${hiddenCls.size === 1 ? '' : 'es'} hidden`, 'warn']);
    if (opts.labels !== 'all') chips.push([opts.labels === 'off' ? 'Labels off (L)' : 'Labels: selected (L)', '']);
    if (opts.fill != null) chips.push([`Fill ${Math.round(Number(opts.fill) * 100)}%`, '']);
    if (opts.ghost) chips.push([ghost && ghost.name ? `Ghost: ${ghost.name}` : 'Ghost', '']);
    if (focusEdge) chips.push([`Edge: ${focusEdge} (Alt+arrows)`, 'accent']);
    const fg = flagFocus && item && flagFocus.forId === item.id ? flagFocus : null;
    if (fg) {
      // The legend of the amber dashed ghost: what it is, and how to clear it.
      const what = fg.code === 'seq_gap' ? 'missing box' : fg.code === 'seq_jump' ? 'where the frames around put it'
        : fg.other ? `${fg.rel === 'before' ? 'frame before' : fg.rel === 'similar' ? 'near-identical image' : 'twin'}: ${plural(fg.other.boxes.length, 'box', 'boxes')}${fg.rel === 'before' || fg.other.boxes.length * boxes.length > 50000 ? '' : (n => (n ? `, ${fmt.num(n)} not here` : ''))(fg.boxes.filter(flagDiffers).length)}` : '';
      chips.push([fg.boxes.length && what ? `${what} · Esc clears` : `${issueLabel(fg.code)} · Esc clears`, fg.boxes.length ? 'flag-legend' : 'flag-legend ring']);
    }
    const sig = JSON.stringify(chips);
    if (sig === hudSig) return;
    hudSig = sig;
    hud.replaceChildren(...chips.map(([t, c, sw]) => { const e = ui.chip(t, { cls: c, title: c === 'ann-attrs' ? t : undefined }); if (sw) e.prepend(h('span', { class: 'swatch', style: `background:${sw}` })); if (c.startsWith('flag-legend')) e.prepend(h('span', { class: 'ann-legend-dash', 'aria-hidden': 'true' })); return e; }));
  }
  function note(text, kind = 'info', sticky = false) {
    clearTimeout(noteTimer);
    noteEl.className = `ann-note ${kind}`; noteEl.textContent = text;
    note.sticky = sticky;
    if (!sticky) noteTimer = setTimeout(clearNote, 6000);
  }
  function clearNote() { clearTimeout(noteTimer); noteEl.className = 'ann-note hidden'; noteEl.textContent = ''; note.sticky = false; }

  function setTool(t) {
    if (t !== 'box' && t !== 'pan') corner = null;      // the Pan tool between the two corners keeps the first one
    tool = t; Object.entries(toolBtns).forEach(([k, b]) => { b.classList.toggle('active', k === t); b.setAttribute('aria-pressed', String(k === t)); });
    renderPointOpts(); updateHud(); overlayOnly(); if (mouse.inside) setCursor(cursorFor(mouse.x, mouse.y));
  }
  function toggleBoxMode() {
    opts.boxMode = opts.boxMode === 'click2' ? 'drag' : 'click2'; LS.set('fovea.ann.boxmode', opts.boxMode);
    corner = null;
    setTool('box');
    note(opts.boxMode === 'click2' ? 'Box: 2-click — click one corner, then the opposite corner (zoom and pan in between) · Esc clears' : 'Box: drag');
  }
  function setLabelMode(v) { opts.labels = v; LS.set('fovea.ann.labelmode', v); labelSeg.setValue(v); updateHud(); invalidate(); }
  function setFill(v) { opts.fill = v; LS.set('fovea.ann.fill', v); fillSeg.setValue(v == null ? 'auto' : v); updateHud(); invalidate(); }
  async function addClass(name) {
    names.push(name);
    try { await fovea.api.patch(`/api/datasets/${ds.id}`, { classes: names }); await refresh(); note(`Added class ${names.length - 1}: ${name}`, 'ok'); }
    catch (e) { ui.toast(e.message, { type: 'error' }); names.pop(); }
    renderClasses();
  }
  function openAdjust() {
    const r = adjustBtn.getBoundingClientRect();
    const mk = (key, label) => {
      const val = h('span', { class: 'xs mono' }, `${adjust[key]}%`);
      return h('div', { class: 'ann-opt' }, h('span', { class: 'small', style: 'width:72px' }, label),
        h('input', { type: 'range', class: 'slider', min: 40, max: 220, step: 5, value: adjust[key], 'aria-label': label, onInput: (e) => { adjust[key] = Number(e.target.value); val.textContent = `${adjust[key]}%`; applyAdjust(); } }), val);
    };
    const body = h('div', { class: 'col gap-8', style: 'padding:6px 9px;width:240px' }, mk('b', 'Brightness'), mk('c', 'Contrast'),
      h('div', { class: 'row' }, h('span', { class: 'xs faint grow' }, 'Display only — files are not changed'), h('button', { class: 'btn btn-sm', onClick: () => { adjust = { b: 100, c: 100 }; applyAdjust(); closePop(); } }, 'Reset')));
    openPop({ x: r.left - 200, y: r.top - 130, title: 'Image adjustments', extra: body });
  }
  function applyAdjust() {
    const f = adjust.b === 100 && adjust.c === 100 ? '' : `brightness(${adjust.b / 100}) contrast(${adjust.c / 100})`;
    cImg.style.filter = f; loupe.style.filter = f;
    adjustBtn.classList.toggle('active', !!f);
    LS.set('fovea.ann.adjust', JSON.stringify(adjust));
  }
  applyAdjust();

  // ---------------------------------------------------------------- keyboard
  function clearClassBuf() { clearTimeout(classBufTimer); classBufFor = null; if (classBuf) { classBuf = ''; updateHud(); } }
  /** First digit applies at once. When it can start a valid two-digit id, a second digit within 600 ms on the same
      item replaces it in the same undo step ("1" then "2" → class 12); see classDigit. */
  function digitKey(d) {
    const r = classDigit(classBuf ? { d: classBuf, key: classBufFor } : null, d, selKey(), names.length);
    clearClassBuf();
    if (r.cls == null) { note(`No class ${d} — this dataset has ${names.length} class${names.length === 1 ? '' : 'es'}. Add one in the Classes panel.`, 'warn'); return; }
    applyClass(r.cls, 'digit');
    // The pair is tied to where the first digit landed (a proposal it accepted is now that box).
    if (r.wait) { classBuf = d; classBufFor = selKey(); updateHud(); classBufTimer = setTimeout(clearClassBuf, 600); }
  }
  function cycle(dir) {
    const order2 = [...visProps.map(p => ({ t: 'prop', i: props.indexOf(p) })), ...boxes.map((b, i) => ({ t: 'box', i })).filter(x => !hiddenCls.has(boxes[x.i][0]))];
    if (!order2.length) return;
    const k = sel ? order2.findIndex(x => x.t === sel.t && x.i === sel.i) : -1;
    const n = k < 0 ? (dir > 0 ? 0 : order2.length - 1) : (k + dir + order2.length) % order2.length;
    setSel(order2[n]);
  }
  const MODIFIERS = new Set(['Shift', 'Alt', 'Meta', 'Control', 'CapsLock', 'Fn']);
  const ARROWS_OWN = new Set(['ArrowLeft', 'ArrowRight', 'ArrowUp', 'ArrowDown', 'Home', 'End', 'PageUp', 'PageDown']);
  const onKey = (e) => {
    // A key a control already handled (the position box, a list) is not also an editor key; neither is any key
    // while a menu is open (the ⇧I model picker: Enter and Space activate its items).
    if (destroyed || e.defaultPrevented || ui.hasModal() || pop || (ui.hasMenu && ui.hasMenu())) return;
    if (isTyping()) { if (e.key === 'Escape') e.target.blur(); return; }
    if (classBuf && !MODIFIERS.has(e.key) && !K.digit(e)) clearClassBuf();      // only a digit completes a pair
    const c = e.code, plain = K.plain(e);
    if (K.is(e, 'mod+z')) { e.preventDefault(); doUndo(); return; }
    if (K.is(e, 'mod+shift+z') || K.is(e, 'mod+y')) { e.preventDefault(); doRedo(); return; }
    if (K.is(e, 'mod+s')) { e.preventDefault(); flush(); return; }
    if (K.is(e, 'mod+c')) { if (String(window.getSelection && window.getSelection()).length) return; e.preventDefault(); copyBoxes(); return; }
    if (K.is(e, 'mod+v')) { e.preventDefault(); pasteBoxes(); return; }
    if (K.is(e, 'mod+d')) { e.preventDefault(); duplicateSelected(); return; }
    if (e.metaKey || e.ctrlKey) return;
    if (!note.sticky && !noteEl.classList.contains('hidden') && !['Shift', 'Alt', 'Meta', 'Control'].includes(e.key)) clearNote();
    // A control the user Tabbed to keeps Enter and Space (activate it). A button that merely kept focus after a
    // click, or got it back when a dialog closed, does not: Enter then accepts a proposal as everywhere else.
    const f = document.activeElement, kbdFocus = f && f === tabFocused && f !== cOv;
    if (kbdFocus && (e.key === 'Enter' || c === 'Space')) return;
    // A slider or select the user Tabbed to keeps its arrows, Home and End (they change its value). One that only
    // kept focus after a mouse click does not: there the arrows still step through the images (G19).
    if (kbdFocus && ARROWS_OWN.has(e.key) && f.matches && f.matches('input[type="range"], select')) return;
    if (c === 'Space' && !e.altKey && (mouse.inside || !f || f === document.body || f === cOv)) { e.preventDefault(); if (!e.repeat) { spaceDown = true; setCursor('grab'); overlayOnly(); } return; }
    if ((e.key === 'ArrowRight' || e.key === 'ArrowLeft') && !e.altKey) { e.preventDefault(); stopAuto(); goTo((loading && want >= 0 ? want : idx) + (e.key === 'ArrowRight' ? 1 : -1), { source: 'key' }); return; }
    if (e.key === 'Home' && plain) { e.preventDefault(); stopAuto(); goTo(0); return; }
    if (e.key === 'End' && plain) { e.preventDefault(); stopAuto(); goTo((cursor.total || 1) - 1); return; }
    if (K.is(e, 'shift+a') || K.is(e, 'shift+f') || K.is(e, 'shift+x') || K.is(e, 'shift+u')) { e.preventDefault(); if (!e.repeat) review({ KeyA: 'approved', KeyF: 'flagged', KeyX: 'excluded', KeyU: null }[c]); return; }
    if (K.is(e, 'f')) { fit(); invalidate(true); return; }
    if (K.is(e, 'o')) { if (!e.repeat) { peek = { prev: hiddenAll, at: performance.now() }; hiddenAll = !hiddenAll; updateHud(); invalidate(); } return; }
    if (K.is(e, 'shift+o')) { const seq = ['0', '0.08', '0.2']; const cur = opts.fill == null ? '0.08' : opts.fill; setFill(seq[(seq.indexOf(cur) + 1) % seq.length]); note(`Box fill ${Math.round(Number(opts.fill) * 100)}%`); return; }
    if (K.is(e, 'l')) { const m = { all: 'sel', sel: 'off', off: 'all' }[opts.labels] || 'all'; setLabelMode(m); note(m === 'all' ? 'Labels: all boxes' : m === 'sel' ? 'Labels: selected and hovered box only' : 'Labels off'); return; }
    if (K.is(e, 't')) { crossSw.click(); note(opts.crosshair ? 'Crosshair on' : 'Crosshair off'); return; }
    if (K.is(e, 'm')) { loupeSw.click(); note(opts.loupe ? 'Loupe on — follows the pointer' : 'Loupe off'); return; }
    if (K.is(e, '`')) { toggleGhost(); note(opts.ghost ? 'Previous-frame ghost on' : 'Ghost off'); return; }
    if (K.is(e, 'z')) { zoomToBox(); return; }
    if (K.is(e, 'shift+z')) { zoomTo(1 / dpr); return; }
    if (K.is(e, '=') || K.is(e, '+')) { zoomBy(1.25); return; }
    if (K.is(e, '-')) { zoomBy(1 / 1.25); return; }
    if (K.is(e, '/')) { e.preventDefault(); classSearch(); return; }
    if (K.is(e, '{') || K.is(e, '}')) { stepThr(K.is(e, '}') ? 1 : -1); return; }
    if (K.is(e, 'v')) { setTool('select'); return; }
    if (K.is(e, 'b')) { if (tool === 'box') { if (!e.repeat) toggleBoxMode(); } else setTool('box'); return; }
    if (K.is(e, 'p')) { setTool('point'); return; }
    if (K.is(e, 'w')) { setTool('sam'); return; }
    if (K.is(e, 'h')) { setTool('pan'); return; }
    if (K.is(e, 'n')) { if (!e.repeat) nextUnlabeled(); return; }
    if (K.is(e, 'shift+n')) { if (!e.repeat) nextWithProposals(); return; }
    if (K.is(e, 'i')) { if (!e.repeat) { if (panel && panel.propose) panel.propose(); else note('The AI panel is not ready yet', 'warn'); } return; }
    if (K.is(e, 'shift+i')) { if (!e.repeat) { if (panel && panel.openPicker) panel.openPicker(); else note('The AI panel is not ready yet', 'warn'); } return; }
    if (K.is(e, 'k')) { if (!e.repeat) keyframeKey(); return; }
    if (K.is(e, ',') || K.is(e, '.')) { e.preventDefault(); stopAuto(); stepSeq(K.is(e, '.') ? 1 : -1); return; }
    if (K.is(e, '\\')) { if (!e.repeat) toggleFilm(); return; }
    if (loading && e.key !== 'Escape') return;           // navigation above still works
    if (e.altKey && !e.metaKey && !e.ctrlKey && e.key.startsWith('Arrow')) { e.preventDefault(); nudge(e.key, e.shiftKey); return; }
    const d = K.digit(e);
    if (d && plain && !e.shiftKey) { e.preventDefault(); if (!e.repeat) digitKey(d); return; }
    if (K.is(e, '[')) { setActiveStep(-1); return; }
    if (K.is(e, ']')) { setActiveStep(1); return; }
    if (K.is(e, 'c')) { if (!e.repeat) copyPrevious(false); return; }
    if (K.is(e, 'shift+c')) { if (!e.repeat) copyPrevious(true); return; }
    if (K.is(e, 'r')) { if (!e.repeat) refineSelected(); return; }
    if (K.is(e, 'e')) { const s = selectedItem(); if (!s || s.t !== 'box') { note('Select a box first, then E focuses its edges'); return; } const seq = [null, 'left', 'top', 'right', 'bottom']; focusEdge = seq[(seq.indexOf(focusEdge) + 1) % seq.length]; updateHud(); overlayOnly(); return; }
    if (e.key === 'Enter' && !e.isComposing && e.keyCode !== 229) {
      if (e.repeat) return;
      e.preventDefault();
      if (e.shiftKey) { acceptAllVisible(); return; }
      const s = selectedItem();
      if (s && s.t === 'prop' && props[s.i].vis) acceptProp(props[s.i]);
      else if (visProps.length) selectProp(visProps.find(isPending) || visProps[0]);
      return;
    }
    if (e.key === 'Delete' || e.key === 'Backspace') {
      e.preventDefault();
      if (e.shiftKey) { if (!e.repeat) rejectAllVisible(); return; }
      deleteSelected(); return;
    }
    if (e.key === 'Escape') {
      if (drag) {
        // A true cancel: geometry back to where the drag started, and its history entry dropped.
        const d2 = drag; drag = null;
        // Only a move / resize pushed a history entry when it started; a draw adds its box on release.
        if (d2.started && (d2.kind === 'move' || d2.kind === 'resize') && undo.length) { const s = undo.pop(); restore(s); recalcProps(); renderList(); renderProps(); }
        invalidate(); return;
      }
      if (corner) { corner = null; updateHud(); overlayOnly(); note('Corner cleared'); return; }
      if (note.sticky) { clearNote(); return; }
      if (focusEdge) { focusEdge = null; updateHud(); overlayOnly(); return; }
      if (flagFocus) { clearFlagFocus(); return; }       // the pressed flag chip's ghost (the HUD says "Esc clears")
      if (sel) { setSel(null); return; }
      if (classBuf) clearClassBuf();
      return;
    }
    if (e.key === 'Tab') {
      // Over the canvas Tab cycles proposals then boxes. Elsewhere only with something selected and focus not on
      // a control — otherwise Tab has to move focus, or keyboard users are trapped on this page.
      const f = document.activeElement;
      const onControl = f && f !== document.body && f !== cOv;
      if (!mouse.inside && (onControl || !sel)) return;
      e.preventDefault(); cycle(e.shiftKey ? -1 : 1); return;
    }
  };
  function setActiveStep(d) {
    const n = Math.max(0, Math.min(Math.max(0, names.length - 1), hlCls() + d));
    pickClass(n);
  }
  const onKeyUp = (e) => {
    if (e.code === 'Space' || e.key === ' ') { spaceDown = false; if (mouse.inside) setCursor(cursorFor(mouse.x, mouse.y)); overlayOnly(); }
    if (e.code === 'KeyO' && peek) {
      if (performance.now() - peek.at >= PEEK_MS) { hiddenAll = peek.prev; invalidate(); }   // held: it was a peek
      peek = null; updateHud();
    }
    if (e.key === 'Alt' && mouse.inside && !drag) setCursor(cursorFor(mouse.x, mouse.y, false));
  };
  let tabbedAt = 0, tabFocused = null;
  const onTabDown = (e) => { if (e.key === 'Tab') tabbedAt = performance.now(); };
  const onFocusIn = (e) => { tabFocused = performance.now() - tabbedAt < 400 ? e.target : null; };
  document.addEventListener('keydown', onTabDown, true); document.addEventListener('focusin', onFocusIn);
  document.addEventListener('keydown', onKey); document.addEventListener('keyup', onKeyUp);
  const onBeforeUnload = (e) => {
    if (dirty && item && !conflictOpen) {
      const json = JSON.stringify({ boxes: compactRows(boxes) });
      if (json.length >= 63000) { flush(); e.preventDefault(); e.returnValue = ''; return; }   // too big for keepalive: ask to stay
    }
    dropRedundant(); finalSave('unload'); finalDecisions(true);
  };
  window.addEventListener('beforeunload', onBeforeUnload);
  const offDs = fovea.bus.on ? fovea.bus.on('dataset:updated', (d) => { if (d && d.id === ds.id && Array.isArray(d.classes)) { names = [...d.classes]; renderClasses(); updateHud(); invalidate(); } if (!d || d.id === ds.id) filmRefresh(); }) : null;
  // Proposals or labels changed by someone else (an AI batch, a scan): the strip's per-frame markers are re-read.
  const offProps = fovea.bus.on ? fovea.bus.on('proposals:changed', (d) => { if (!d || ((d.dataset || d.dataset_id) === ds.id && d.source !== 'annotate')) filmRefresh(); }) : null;
  const offScan = fovea.bus.on ? fovea.bus.on('dataset:scanned', (id) => { if (!id || id === ds.id || (id && id.id === ds.id)) filmRefresh(); }) : null;

  function renderHint() { hintLine.textContent = 'V B P W H tools · 0–9 class · / search · Enter accept · O hide · Z zoom · Space pan · , . frames · \\ filmstrip'; }

  // ---------------------------------------------------------------- boot
  renderClasses(); renderPointOpts(); renderProps(); renderSeq(); renderInfo(); updateHud(); renderHint(); renderList(); loadCounts();
  const startId = q0.get('img') ? Number(q0.get('img')) : null;
  const firstWait = setTimeout(() => { if (!item && !destroyed) showStageMsg('Loading…', true); }, NOTICE_MS);
  // Boot in the background so the shell receives this view's teardown immediately.
  (async () => {
    try {
      // The strip's default (on for datasets with real sequences) is known before the first fit, so it never moves it.
      const seqP = opts.film == null ? fovea.api.get(`/api/datasets/${ds.id}/sequences`, { limit: 1 }).then((r) => { hasSeqs = (r.total || 0) > 0; }, () => { hasSeqs = false; }) : null;
      try { await cursor.loadPage(0); } catch (e) { await seqP; if (destroyed) return; if (listGone(e)) { clearTimeout(firstWait); return; } throw e; }
      await seqP;
      if (destroyed) return;
      root.classList.toggle('film-on', filmShown());
      renderWorkingSet();
      if (!cursor.total) emptyState();
      else {
        let start = 0;
        if (startId) {
          const pos = await cursor.positionOf(startId);
          if (destroyed) return;
          if (pos >= 0) start = pos;
          else note('That image is not in the current list — starting from the first image', 'warn');
        }
        await goTo(start);
      }
    } catch (e) { if (!destroyed) showStageMsg(e.message, true); }
    clearTimeout(firstWait);
    if (!destroyed) resizeCanvas();
  })();

  return () => {
    destroyed = true; navToken++;                 // any navigation still in flight bails at its next check
    clearTimeout(saveTimer); clearTimeout(classBufTimer); clearTimeout(loadingNote); clearTimeout(savingNote); clearTimeout(noteTimer); clearTimeout(decTimer); clearTimeout(firstWait); stopAuto();
    if (rafId) cancelAnimationFrame(rafId);
    closePop();
    try { panel && panel.destroy && panel.destroy(); } catch (e) { console.error(e); }
    imageListeners.clear();
    inflight.forEach(x => x.ac.abort());
    frames.clear();
    stashHistory();
    // Let an in-flight save land first (it updates `version`), then write whatever is still pending.
    if (dirty && item && (conflictPending || conflictOpen)) ui.toast(`Edits to ${item.rel_path.split('/').pop()} were not saved: the file changed elsewhere and the conflict was left open`, { type: 'error', timeout: 9000 });
    dropRedundant();
    const last = () => { if (saving) { setTimeout(last, 30); return; } finalSave('leave'); finalDecisions(false); };
    last();
    document.removeEventListener('keydown', onKey); document.removeEventListener('keyup', onKeyUp);
    document.removeEventListener('keydown', onTabDown, true); document.removeEventListener('focusin', onFocusIn);
    window.removeEventListener('mousemove', onMove); window.removeEventListener('mouseup', onUp); window.removeEventListener('beforeunload', onBeforeUnload); window.removeEventListener('blur', onBlur);
    if (dprMq) dprMq.removeEventListener('change', onDpr);
    if (offDs) offDs();
    if (offProps) offProps();
    if (offScan) offScan();
    clearTimeout(filmRefreshTimer); clearTimeout(flagTimer); if (filmRaf) cancelAnimationFrame(filmRaf); if (filmTouchRaf) cancelAnimationFrame(filmTouchRaf);
    for (const t of tileEls.values()) t._img.removeAttribute('src');
    ro.disconnect();
  };
}
