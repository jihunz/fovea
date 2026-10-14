"""Annotate's frozen working set (static/lib/cursor.js) and the pure helpers of views/annotate.js, exercised in Node,
plus the label-save contract Annotate relies on (an emptied image without a label file stays without one).

The list a person reviews must not shift under them: labeling "No boxes" or approving "Unreviewed" removes each
finished image from the live query, and OFFSET paging then skipped one image per edit. These tests walk such a list
while it changes and check that every image is visited exactly once, in order — also against a server whose OFFSET
is wrong (SQLite 3.51.0 skips one row per matching box under EXISTS filters such as cls=). Skipped without Node.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

CURSOR = Path(__file__).resolve().parent.parent / "static" / "lib" / "cursor.js"
NODE = shutil.which("node")

MOCK_API = r"""
// Simulated server: ids 1..N in id order. Filter labeled=nobox hides labeled ids; review=none hides reviewed ids.
// badOffset: OFFSET > 0 lands further down the list than asked (what SQLite 3.51.0 does under EXISTS filters);
// LIMIT, keyset (after=) and COUNT-based positions stay right, as they do there.
export const db = { N: 1000, labeled: new Set(), reviewed: new Set(), added: new Set(), calls: 0, badOffset: false, paths: {} };
const count = (path) => { const k = path.split('/').pop(); db.paths[k] = (db.paths[k] || 0) + 1; };
function live(q) {
  const out = [];
  for (let i = 1; i <= db.N + 50; i++) {
    if (i > db.N && !db.added.has(i)) continue;
    if (q.labeled === 'nobox' && db.labeled.has(i)) continue;
    if (q.review === 'none' && db.reviewed.has(i)) continue;
    out.push(i);
  }
  return out;
}
export async function get(path, q) {
  db.calls++; count(path);
  const L = live(q);
  if (path.endsWith('/images')) {
    const from = q.after != null ? L.filter(x => x > Number(q.after)) : L;
    const off = db.badOffset && q.offset > 0 ? Math.min(from.length, Math.floor(q.offset * 1.7) + 3) : q.offset;
    return { total: L.length, items: from.slice(off, off + q.limit).map(id => ({ id })) };
  }
  if (path.endsWith('/images/ids')) return { total: L.length, ids: L.slice(0, q.limit) };
  if (path.endsWith('/neighbor')) { const id = Number(q.id); const n = L.find(x => x > id); return { item: n ? { id: n } : null }; }
  if (path.endsWith('/position')) { const i = L.indexOf(Number(q.id)); return { position: i >= 0 ? i : null }; }
  throw new Error('unexpected ' + path);
}
export async function post(path, body) {
  db.calls++; count(path);
  if (path.endsWith('/images/by-ids')) return { items: body.ids.map(id => ({ id })) };
  throw new Error('unexpected POST ' + path);
}
"""

SCENARIOS = r"""
import { ImageCursor } from './cursor.mjs';
import { db } from './mockapi.mjs';
const out = {};
const reset = () => { db.labeled.clear(); db.reviewed.clear(); db.added.clear(); db.calls = 0; };

// 1. Label 2 of every 3 images of "No boxes" while stepping through it.
reset();
let c = new ImageCursor({ dsId: 'x', filters: { labeled: 'nobox' }, pageSize: 100 });
await c.loadPage(0);
let seen = [];
for (let i = 0; ; i++) { const it = await c.ensure(i); if (!it) break; seen.push(it.id); if (i % 3 !== 2) db.labeled.add(it.id); }
out.walk = { n: seen.length, unique: new Set(seen).size, inOrder: seen.every((v, k) => v === k + 1), total: c.total };

// 2. Approve every image of "Unreviewed" (the Explore/Annotate review loop).
reset();
c = new ImageCursor({ dsId: 'x', filters: { review: 'none' }, pageSize: 120 });
await c.loadPage(0);
seen = [];
for (let i = 0; ; i++) { const it = await c.ensure(i); if (!it) break; seen.push(it.id); db.reviewed.add(it.id); }
out.review = { n: seen.length, unique: new Set(seen).size, inOrder: seen.every((v, k) => v === k + 1) };

// 3. Unchanged list: one request per page (fast path), far jump then back-fill stays aligned.
reset();
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
await c.loadPage(0);
for (let i = 0; i < 1000; i++) await c.ensure(i);
out.calls = db.calls;
reset();
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
await c.loadPage(0);
const last = await c.ensure(999), mid = await c.ensure(500);
out.jump = { last: last && last.id, mid: mid && mid.id };

// 4. positionOf beyond the loaded frontier after edits.
reset();
c = new ImageCursor({ dsId: 'x', filters: { labeled: 'nobox' }, pageSize: 100 });
await c.loadPage(0);
for (let i = 0; i < 60; i++) { const it = await c.ensure(i); db.labeled.add(it.id); }
const pos = await c.positionOf(800);
out.position = { id: c.items[pos] && c.items[pos].id, before: (await c.ensure(pos - 1)).id };

// 5. An image that starts matching inside the part already loaded stays out until refresh(); one that appears
//    beyond it is simply part of the rest of the list.
reset();
for (let i = 1; i <= 100; i++) db.reviewed.add(i);
c = new ImageCursor({ dsId: 'x', filters: { review: 'none' }, pageSize: 100 });
await c.loadPage(0); await c.ensure(250);
db.reviewed.delete(50);
out.joinedBefore = await c.positionOf(50);
db.added.add(1001);
out.tail = await c.positionOf(1001);
await c.refresh();
out.joinedAfter = await c.positionOf(50);

// 6. A stale request's failure must not delete a newer in-flight entry for the same page.
reset();
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
const p1 = c.loadPage(0); c.reset(); const p2 = c.loadPage(0);
out.dedupe = c.loading.get(0) === p2;
await Promise.allSettled([p1, p2]);

// 7. A server whose OFFSET is wrong: walking, labeling while walking, far jumps and positionOf still line up.
reset(); db.badOffset = true;
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
await c.loadPage(0);
seen = [];
for (let i = 0; ; i++) { const it = await c.ensure(i); if (!it) break; seen.push(it.id); }
out.badWalk = { n: seen.length, unique: new Set(seen).size, inOrder: seen.every((v, k) => v === k + 1), total: c.total };
reset(); db.badOffset = true;
c = new ImageCursor({ dsId: 'x', filters: { labeled: 'nobox' }, pageSize: 100 });
await c.loadPage(0);
seen = [];
for (let i = 0; ; i++) { const it = await c.ensure(i); if (!it) break; seen.push(it.id); if (i % 3 !== 2) db.labeled.add(it.id); }
out.badLabelWalk = { n: seen.length, unique: new Set(seen).size, inOrder: seen.every((v, k) => v === k + 1) };
reset(); db.badOffset = true;
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
await c.loadPage(0);
const bl = await c.ensure(999), bm = await c.ensure(500), bp = await c.positionOf(777);
out.badJump = { last: bl && bl.id, mid: bm && bm.id, pos: bp, at: c.items[bp] && c.items[bp].id };
reset(); db.badOffset = true;
c = new ImageCursor({ dsId: 'x', filters: {}, pageSize: 100 });
out.badCold = await c.positionOf(640);          // nothing loaded yet: the first read is already far down
out.badColdId = c.items[out.badCold] && c.items[out.badCold].id;
db.badOffset = false;

// 8. A deep link (Annotate opened at the 800th image of a filtered list): positionOf finds the first live image after
//    the loaded part and its position; the page fetch that follows reuses them instead of asking again.
reset(); db.paths = {};
c = new ImageCursor({ dsId: 'x', filters: { labeled: 'nobox' }, pageSize: 100 });
await c.loadPage(0);
db.paths = {};
const deep = await c.positionOf(800);
out.deep = { at: c.items[deep] && c.items[deep].id, neighbor: db.paths.neighbor || 0, position: db.paths.position || 0, images: db.paths.images || 0 };
out.deepBefore = (await c.ensure(deep - 1)).id;
// The target opens a page (index 800 = page 8's first row): the OFFSET check of that page is its own position, known.
reset(); db.paths = {};
c = new ImageCursor({ dsId: 'x', filters: { labeled: 'nobox' }, pageSize: 100 });
await c.loadPage(0);
db.paths = {};
const edge = await c.positionOf(801);
out.deepEdge = { at: c.items[edge] && c.items[edge].id, neighbor: db.paths.neighbor || 0, position: db.paths.position || 0 };
// ...and a stale answer (the list was re-queried in between) is never reused.
c.hint = { anchor: 100, item: { id: 5 }, position: 0, v: c.version - 1, at: Date.now() };
out.staleHint = c._takeHint(100, c.version);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def results(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("cursor")
    src = CURSOR.read_text(encoding="utf-8").replace("from './api.js'", "from './mockapi.mjs'")
    (d / "cursor.mjs").write_text(src, encoding="utf-8")
    (d / "mockapi.mjs").write_text(MOCK_API, encoding="utf-8")
    (d / "run.mjs").write_text(SCENARIOS, encoding="utf-8")
    proc = subprocess.run([NODE, str(d / "run.mjs")], capture_output=True, text=True, timeout=60, cwd=d)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_walk_while_labeling_visits_every_image_once_in_order(results):
    assert results["walk"] == {"n": 1000, "unique": 1000, "inOrder": True, "total": 1000}


def test_review_loop_never_skips(results):
    assert results["review"] == {"n": 1000, "unique": 1000, "inOrder": True}


def test_unchanged_list_costs_one_request_per_page(results):
    assert results["calls"] <= 10 + 1


def test_far_jump_and_backfill_stay_aligned(results):
    assert results["jump"] == {"last": 1000, "mid": 501}


def test_position_of_image_beyond_loaded_pages(results):
    assert results["position"] == {"id": 800, "before": 799}


def test_new_matches_join_only_after_refresh(results):
    assert results["joinedBefore"] == -1
    assert results["joinedAfter"] == 0         # first in id order once the list is re-queried
    assert results["tail"] == 900          # 900 unreviewed images before it; the unloaded tail is still live


def test_stale_failure_does_not_drop_newer_request(results):
    assert results["dedupe"] is True


def test_wrong_server_offset_walk_stays_in_order(results):
    assert results["badWalk"] == {"n": 1000, "unique": 1000, "inOrder": True, "total": 1000}
    assert results["badLabelWalk"] == {"n": 1000, "unique": 1000, "inOrder": True}


def test_wrong_server_offset_far_jumps_and_positions(results):
    assert results["badJump"] == {"last": 1000, "mid": 501, "pos": 776, "at": 777}
    assert results["badCold"] == 639 and results["badColdId"] == 640


def test_deep_position_does_not_repeat_neighbor_and_position(results):
    # position(target) + neighbor(anchor) + position(first after anchor), then the page read and its OFFSET check:
    # before, the page fetch asked neighbor(anchor) and position(first after anchor) a second time
    assert results["deep"] == {"at": 800, "neighbor": 1, "position": 3, "images": 1}
    assert results["deepBefore"] == 799
    assert results["deepEdge"] == {"at": 801, "neighbor": 1, "position": 2}
    assert results["staleHint"] is None


# ---------------------------------------------------------------- views/annotate.js pure helpers + keymap

ANNOTATE = Path(__file__).resolve().parent.parent / "static" / "views" / "annotate.js"
ANN_SCENARIOS = r"""
import { boxFromCorners, filmOrder, install, classDigit, defaultOrder, readThr, wheelStep, shownPending, seqNeighbor, frameGroup, parsePosition, refGeom, refByGeom, frameOrdinals, flagTargets, boxSizeText, bulkAcceptPlan, dupOf, middleEllipsis, interpSummary, initialSort, decideBody, sendDecisions, withQueuedDecisions } from './views/annotate.js';
const out = {};
const r6 = (v) => v && v.map(x => Math.round(x * 1e6) / 1e6);
out.box = r6(boxFromCorners({ x: 100, y: 50 }, { x: 20, y: 10 }, 200, 100));        // corners in either order
out.clamped = r6(boxFromCorners({ x: -30, y: -5 }, { x: 50, y: 40 }, 200, 100));    // a corner in the margin snaps to the edge
out.thin = boxFromCorners({ x: 10, y: 10 }, { x: 11, y: 60 }, 200, 100);           // 1 px wide: rejected
out.outside = boxFromCorners({ x: 210, y: 10 }, { x: 260, y: 60 }, 200, 100);      // entirely outside the image
const rows = ['test/clip_10.jpg', 'train/clip_9.jpg', 'test/clip_2.jpg', 'train/clip_2.jpg', 'val/clip_100.jpg', 'train/clip_011.jpg']
  .map((rel_path, id) => ({ id, rel_path }));
out.order = filmOrder(rows).map(r => r.rel_path);
out.untouched = rows[0].rel_path;
// Class digits (12 classes): a first "1" may become 10/11, but only on the item it went to.
out.d1 = classDigit(null, '1', 'b0', 12);
out.pairSame = classDigit({ d: '1', key: 'b0' }, '0', 'b0', 12);
out.pairOther = classDigit({ d: '1', key: 'b0' }, '0', 'b1', 12);          // Tab in between: a fresh 0
out.pairOtherOne = classDigit({ d: '1', key: 'b0' }, '1', 'b5', 12);       // a draw in between: a fresh 1 (may pair again)
out.pairTooBig = classDigit({ d: '1', key: 'b0' }, '5', 'b0', 12);         // 15 does not exist: 5
out.noClass = classDigit(null, '7', '-', 5);
out.zero = classDigit(null, '0', '-', 0);
out.few = classDigit(null, '1', '-', 5);
out.orders = ['audit', 'id', 'name', 'set', 'random', 'flag'].map(defaultOrder);
// The sort a link without one opens on: QA-flag lists most suspicious first, audit-only lists by audit score.
out.initSort = {
  none: initialSort({}), flag: initialSort({ issue: 'dup_label_mismatch' }), flags: initialSort({ issue: 'seq_gap,near_dup' }),
  audit: initialSort({ issue: 'ai_loose' }), audits: initialSort({ issue: 'ai_missing,ai_swap' }), mixed: initialSort({ issue: 'ai_loose,near_dup' }),
  parser: initialSort({ issue: 'bad_format' }), parserFlag: initialSort({ issue: 'bad_format,seq_gap' }), any: initialSort({ issue: 'any' }),
  aiAudit: initialSort({ ai: 'audit' }), aiPending: initialSort({ ai: 'pending' }), set: initialSort({ set: 'tok', issue: 'near_dup' }),
  spaced: initialSort({ issue: ' near_dup , ' }), nothing: initialSort(null), unknown: initialSort({ issue: 'some_future_code' }),
};
// Proposal decisions: a failed send keeps them (never over a newer one), only an unchangeable answer drops them.
const dq = (pairs) => new Map(pairs);
const dres = {};
dres.body = decideBody(dq([[1, 'rejected'], [2, 'accepted'], [3, 'pending'], [4, 'rejected']]));
dres.bodyNoPending = decideBody(dq([[5, 'accepted']]));
let q = dq([[1, 'rejected'], [2, 'accepted']]), seen = null;
let r = await sendDecisions(q, async (b) => { seen = b; });
dres.ok = { ok: r.ok, sent: r.sent, left: q.size, body: seen };
q = dq([[1, 'rejected'], [2, 'accepted']]);
r = await sendDecisions(q, async () => { throw Object.assign(new Error('Failed to fetch'), { status: 0 }); });
dres.net = { ok: r.ok, retry: r.retry, left: [...q.entries()], msg: r.error.message };
q = dq([[1, 'rejected'], [2, 'accepted']]);
r = await sendDecisions(q, async () => { q.set(1, 'pending'); q.set(7, 'rejected'); throw Object.assign(new Error('busy'), { status: 503 }); });
dres.newer = { ok: r.ok, retry: r.retry, left: [...q.entries()].sort((a, b) => a[0] - b[0]) };
dres.codes = {};
for (const st of [408, 429, 500, 403, 422, 404, 405]) {
  q = dq([[1, 'rejected']]);
  r = await sendDecisions(q, async () => { throw Object.assign(new Error('x'), { status: st }); });
  dres.codes[st] = { ok: r.ok, retry: r.retry, left: q.size };
}
q = new Map(); r = await sendDecisions(q, async () => { throw new Error('never called'); });
dres.empty = { ok: r.ok, sent: r.sent };
const loaded = [{ id: 1, state: 'pending' }, { id: 2, state: 'pending' }, { id: null, state: 'pending' }, { id: 3, state: 'accepted' }];
dres.overlay = withQueuedDecisions(loaded, dq([[1, 'rejected'], [3, 'pending'], [9, 'accepted']])).map(p => p.state);
dres.overlayNone = withQueuedDecisions([{ id: 1, state: 'pending' }], new Map()).map(p => p.state);
out.dec = dres;
out.thr = [readThr(null), readThr('0'), readThr('0.4'), readThr('abc'), readThr(''), readThr('2')];
out.wheel = [wheelStep(0, 100), wheelStep(0, -100), wheelStep(100, 0), wheelStep(-100, 0), wheelStep(0, 0)];
const box = [0, 0.5, 0.5, 0.2, 0.2];
const preds = [
  { cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, conf: 0.9, kind: 'detect', state: 'pending' },     // same class on a box: "already labeled"
  { cls: 1, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, conf: 0.9, kind: 'detect', state: 'pending' },     // another class: shown (⇄)
  { cls: 0, xc: 0.1, yc: 0.1, w: 0.1, h: 0.1, conf: 0.1, kind: 'detect', state: 'pending' },     // under the threshold
  { cls: 0, xc: 0.8, yc: 0.8, w: 0.1, h: 0.1, conf: 0.9, kind: 'detect', state: 'rejected' },    // decided
];
out.shown = shownPending(preds, [box], 0.25);
out.shownOnlyHidden = shownPending([preds[0], preds[2]], [box], 0.25);
out.shownZeroThr = shownPending([preds[0], preds[2]], [box], 0);
out.shownHiddenCls = shownPending([preds[1]], [box], 0.25, new Set([1]));
out.shownAudit = shownPending([{ ...preds[0], kind: 'swap' }], [box], 0.25);
// A sequence held twice (the same video in a/ and b/: "twins"), plus a frame only b/ has.
const twins = filmOrder(['b/x_00007.jpg', 'a/x_00000.jpg', 'a/x_00050.jpg', 'b/x_00000.jpg', 'a/x_00007.jpg', 'b/x_00042.jpg', 'b/x_00050.jpg']
  .map((rel_path, id) => ({ id, rel_path, split: rel_path[0] })));
const at = (p) => twins.findIndex(f => f.rel_path === p), name = (k) => (k >= 0 ? twins[k].rel_path : null);
out.twinOrder = twins.map(f => f.rel_path);
out.twinPrev = [name(seqNeighbor(twins, at('b/x_00007.jpg'), -1, 'b')), name(seqNeighbor(twins, at('a/x_00007.jpg'), -1, 'a')),
  name(seqNeighbor(twins, at('a/x_00000.jpg'), -1, 'a')), name(seqNeighbor(twins, at('b/x_00000.jpg'), -1, 'b'))];
out.twinNext = [name(seqNeighbor(twins, at('b/x_00007.jpg'), 1, 'b')), name(seqNeighbor(twins, at('a/x_00007.jpg'), 1, 'a')),
  name(seqNeighbor(twins, at('b/x_00042.jpg'), 1, 'a')), name(seqNeighbor(twins, at('b/x_00050.jpg'), 1, 'b'))];
out.group = frameGroup(twins, at('b/x_00050.jpg'), -1, 'b').map(name);
// "Frame k of n" counts frames, not images: the copies of a frame in another folder share its number.
const ord = frameOrdinals(twins);
out.ord = { of: [...ord.of], n: ord.n, multi: ord.multi };
const three = frameOrdinals(filmOrder(['a/f_1.jpg', 'b/f_1.jpg', 'c/f_1.jpg', 'a/f_2.jpg'].map((rel_path, id) => ({ id, rel_path }))));
out.ord3 = { of: [...three.of], n: three.n, multi: three.multi };
out.ord0 = { n: frameOrdinals([]).n, multi: frameOrdinals([]).multi };
// QA flag chips: which rows a flag names, where the frames around put a box, which other image to compare.
out.ft = {
  gap: flagTargets({ code: 'seq_gap', detail: { boxes: [[8, 0.75, 0.45, 0.07, 0.06]], prev_id: 136, next_id: 138 } }),
  jump: flagTargets({ code: 'seq_jump', detail: { jumps: [{ row: 2, box: [0, 0.1, 0.1, 0.1, 0.1], expected: [0, 0.3, 0.3, 0.1, 0.1] }, { row: 'x', expected: [1, 2] }], prev_id: 4, next_id: 6 } }),
  twin: flagTargets({ code: 'dup_label_mismatch', detail: { twin: 136, twin_path: 'train/a.jpg', n_boxes: 6, twin_n_boxes: 7 } }),
  leak: flagTargets({ code: 'leak_split', detail: { twin: 9, splits: ['val'] } }),
  same: flagTargets({ code: 'dup_same_role', detail: { twin: 11, splits: ['test'], role: 'test' } }),
  flicker: flagTargets({ code: 'seq_flicker', detail: { rows: [{ row: 1, cls: 3, expected: 0 }, { row: 1, cls: 3, expected: 0 }, { row: -1 }], prev_id: 40, next_id: 42 } }),
  spike: flagTargets({ code: 'seq_count_spike', detail: { n: 9, prev: 3, next: 3, prev_id: 7, next_id: 8 } }),
  shape: flagTargets({ code: 'shape_outlier', detail: { rows: [{ row: 0, kind: 'aspect', value: 4, typical: 0.4 }] } }),
  near: flagTargets({ code: 'near_dup', detail: { group: 3, dist: 1, size: 2 } }),
  empty: flagTargets(null),
};
out.geo = [boxSizeText([0, 0.5, 0.5, 0.25, 0.1], 640, 480), boxSizeText([0, 0.5, 0.5, 0.04, 0.23], 0, 0), boxSizeText([0, 0.5, 0.5, 0.004, 0.5], 0, 0)];
// A sequence split over two folders frame by frame (no twins): every frame is a step, across folders.
const inter = filmOrder(['train/c_1.jpg', 'test/c_2.jpg', 'train/c_3.jpg'].map((rel_path, id) => ({ id, rel_path, split: rel_path.split('/')[0] })));
out.interPrev = inter[seqNeighbor(inter, 1, -1, 'test')].rel_path;
out.interNext = inter[seqNeighbor(inter, 1, 1, 'test')].rel_path;
out.pos = ['7133', '7,133', '7,133 / 7,808', ' 12 ', '0', 'frame_0042', '#12', '1,2', '7133/x'].map(parsePosition);
out.refGeom = [refGeom([3, 0.5, 0.5, 0.2, 0.2]), refGeom([0.5, 0.5, 0.2, 0.2]), refGeom(null), refGeom('x'), refGeom([1, 2])];
const gt = [[0, 0.1, 0.1, 0.05, 0.05], [3, 0.5, 0.5, 0.2, 0.2], [3, 0.8, 0.8, 0.1, 0.1]];
const loose = { kind: 'loose', b: [3, 0.5, 0.5, 0.17, 0.17], ref_row: 2, ref_box: [0.5, 0.5, 0.2, 0.2] };   // made when the box was row 2
out.refShifted = gt.indexOf(refByGeom(gt, loose));                       // a row above it was deleted: found by geometry
out.refRowOk = gt.indexOf(refByGeom(gt, { ...loose, ref_row: 1 }));
out.refGone = refByGeom([gt[0], gt[2]], loose);                           // the box itself was deleted: stale
out.refOtherCls = refByGeom([[5, 0.5, 0.5, 0.2, 0.2]], loose);            // its class changed since: stale
out.refSwap = refByGeom([[5, 0.5, 0.5, 0.2, 0.2]], { kind: 'swap', b: [3, 0.5, 0.5, 0.2, 0.2], ref_row: 4, ref_box: [0.5, 0.5, 0.2, 0.2] }) !== null;
out.refNone = refByGeom(gt, { kind: 'loose', b: [3, 0.5, 0.5, 0.2, 0.2], ref_row: 2, ref_box: null }) === undefined;
// ⇧Enter: a ⤢ tighter-box fix and the detection it came from are one object (train/images_GOPR0804_00076.jpg).
const bx = [[0, 0.1, 0.1, 0.05, 0.05], [1, 0.3, 0.3, 0.1, 0.1], [0, 0.55, 0.38, 0.08, 0.25], [1, 0.8, 0.2, 0.1, 0.1]];
const tight = [0, 0.547327, 0.377894, 0.041937, 0.159249];
const det = { kind: 'detect', b: tight, conf: 0.91 }, fix = { kind: 'loose', b: tight, conf: 0.91, refBox: bx[2] };
const plan = (list) => { const r = bulkAcceptPlan(list, bx); return { take: r.take.map(p => p.kind), dupes: r.dupes.map(p => p.kind), left: r.left.map(p => p.kind) }; };
out.planDetFirst = plan([det, fix]);
out.planFixFirst = plan([fix, det]);
// the same detection as class 2 on a box a ⇄ fix turns into class 2: one object, a duplicate once the swap lands
const swap = { kind: 'swap', b: [2, 0.3, 0.3, 0.1, 0.1], conf: 0.8, refBox: bx[1] }, det2 = { kind: 'detect', b: [2, 0.3, 0.3, 0.1, 0.1], conf: 0.7 };
out.planSwap = plan([swap, det2]);
// a detection of another class where the loose box was: still yours to decide (⇄), as without the fix
out.planOldPlace = plan([fix, { kind: 'detect', b: [3, 0.55, 0.38, 0.08, 0.25], conf: 0.6 }]);
// ✕ no support on box #2 does not make room for a detection of another class there
out.planSpurious = plan([{ kind: 'spurious', b: [0, 0.55, 0.38, 0.08, 0.25], conf: 0.5, refBox: bx[2] }, { kind: 'detect', b: [3, 0.55, 0.38, 0.08, 0.25], conf: 0.4 }]);
out.planTwoDet = plan([{ kind: 'detect', b: [4, 0.7, 0.7, 0.1, 0.1], conf: 0.9 }, { kind: 'interp', b: [4, 0.7, 0.7, 0.1, 0.1], conf: null }, { kind: 'missing', b: [4, 0.9, 0.9, 0.05, 0.05], conf: 0.4 }]);
out.planStaleRef = plan([{ kind: 'loose', b: tight, conf: 0.9, refBox: [0, 0.55, 0.38, 0.08, 0.25] }, det]);   // its box is gone
// a fix that moved its box onto a box of its class: dupOf finds that box (never itself, another class, or a polygon row)
const moved = [0, 0.547327, 0.377894, 0.041937, 0.159249], kept = [0, 0.548, 0.378, 0.042, 0.16], poly = Object.assign([0, 0.547327, 0.377894, 0.041937, 0.159249], { kind: 'seg' });
out.dupOf = [dupOf([moved, kept], moved) === kept, dupOf([moved, [1, ...kept.slice(1)]], moved), dupOf([moved], moved), dupOf([moved, poly], moved), dupOf([poly, kept], poly), dupOf([moved, [0, 0.9, 0.9, 0.05, 0.05]], moved)];
// canvas label chips: shortened in the middle to the cap, the class id and the distinguishing tail kept
const m7 = (s) => Array.from(s).length * 7;
const longName = '0 person_lying_on_the_ground_after_fall';
const el = middleEllipsis(longName, 172, m7);
out.ell = { text: el, w: m7(el), maximal: m7(el) + 7 > 172, short: middleEllipsis('0 person', 172, m7), tiny: middleEllipsis(longName, 5, m7) };
// interpolation status: what the server left out is said
out.interp = {
  unmatched: interpSummary({ preds: [{ image_id: 5 }, { image_id: 5 }], images: 1, unmatched: { from: 2, to: 0 }, skipped_existing: 0, matched: 2 }, 'A.jpg', 'B.jpg'),
  adjacent: interpSummary({ run_id: null, images: 0, preds: [], frames: 0, message: 'The keyframes are adjacent: no frames in between' }, 'A.jpg', 'B.jpg'),
  existing: interpSummary({ preds: [], images: 0, unmatched: { from: 0, to: 0 }, skipped_existing: 4 }, 'A.jpg', 'B.jpg'),
  both: interpSummary({ preds: [{ image_id: 1 }], images: 1, unmatched: { from: 1, to: 3 }, skipped_existing: 1 }, 'A.jpg', 'B.jpg'),
  none: interpSummary({ preds: [], images: 0 }, 'A.jpg', 'B.jpg'),
  counts: interpSummary({ preds: 3, images: 2 }, 'A.jpg', 'B.jpg'),
  // a person who falls between the keyframes: paired across classes, the server's message says so (qa.interpolate)
  fall: interpSummary({ preds: [{ image_id: 5 }, { image_id: 6 }], images: 2, unmatched: { from: 1, to: 0 }, skipped_existing: 0, matched: 1,
    class_changes: [{ from_box: 0, to_box: 0, from_cls: 0, to_cls: 1 }],
    message: '1 box changes class between the keyframes (standing → fallen): the proposals take the class of the nearer keyframe' }, 'A.jpg', 'B.jpg'),
  fallAllThere: interpSummary({ preds: [], images: 0, unmatched: { from: 0, to: 0 }, skipped_existing: 3, class_changes: [{ from_cls: 0, to_cls: 1 }],
    message: '1 box changes class between the keyframes (standing → fallen): the proposals take the class of the nearer keyframe' }, 'A.jpg', 'B.jpg'),
  changesNoMsg: interpSummary({ preds: [{ image_id: 5 }], images: 1, class_changes: [{}, {}] }, 'A.jpg', 'B.jpg'),
  otherMsg: interpSummary({ preds: [{ image_id: 5 }], images: 1, message: 'Weighted by position: the file names carry no frame numbers' }, 'A.jpg', 'B.jpg'),
};
const reg = [];
install({ registerTab() {}, MOD: '⌘', keys: { registerKeys: (scope, title, items) => reg.push({ scope, items }) } });
out.keys = reg.length ? reg[0].items.map(([k]) => k) : [];
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def ann(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("annotate")
    static = ANNOTATE.parent.parent
    (d / "package.json").write_text('{"type": "module"}', encoding="utf-8")
    for rel in ("views/annotate.js", "views/issues.js", "lib/media.js"):
        (d / rel).parent.mkdir(parents=True, exist_ok=True)
        (d / rel).write_text((static / rel).read_text(encoding="utf-8"), encoding="utf-8")
    (d / "run.js").write_text(ANN_SCENARIOS, encoding="utf-8")
    proc = subprocess.run([NODE, str(d / "run.js")], capture_output=True, text=True, timeout=60, cwd=d)
    assert proc.returncode == 0, proc.stderr
    return json.loads(proc.stdout.strip().splitlines()[-1])


def test_box_from_corners_any_order_and_clamped(ann):
    assert ann["box"] == [0.3, 0.3, 0.4, 0.4]
    assert ann["clamped"] == [0.125, 0.2, 0.25, 0.4]
    assert ann["thin"] is None and ann["outside"] is None


def test_filmstrip_order_is_natural_file_name_order_across_splits(ann):
    assert ann["order"] == ["test/clip_2.jpg", "train/clip_2.jpg", "train/clip_9.jpg", "test/clip_10.jpg",
                            "train/clip_011.jpg", "val/clip_100.jpg"]
    assert ann["untouched"] == "test/clip_10.jpg"          # a sorted copy; the caller's rows keep their order


def test_annotate_registers_sequence_and_box_mode_keys(ann):
    keys = ann["keys"]
    assert ", ." in keys and "\\" in keys and "B B" in keys
    assert "⌘C ⌘V ⌘D" in keys                             # the modifier glyph follows the platform (fovea.MOD)


def test_second_class_digit_pairs_only_on_the_same_item(ann):
    assert ann["d1"] == {"cls": 1, "wait": True}
    assert ann["pairSame"] == {"cls": 10, "wait": False}
    # "1, Tab, 0" and "draw, 1, draw, 1": the second digit starts over on the new item (never class 10 / 11)
    assert ann["pairOther"] == {"cls": 0, "wait": False}
    assert ann["pairOtherOne"] == {"cls": 1, "wait": True}
    assert ann["pairTooBig"] == {"cls": 5, "wait": False}
    assert ann["noClass"]["cls"] is None
    assert ann["zero"] == {"cls": 0, "wait": False} and ann["few"] == {"cls": 1, "wait": False}


def test_default_order_matches_the_server_and_explore(ann):
    # imagesq.DEFAULT_DESC = {"audit", "flag"}: "worst first" links carry no order and must open on the most suspect image
    assert ann["orders"] == ["desc", "asc", "asc", "asc", "asc", "desc"]


def test_a_flag_or_audit_list_opens_most_suspicious_first(ann):
    # Explore / Overview hand over ?issue=dup_label_mismatch without a sort: index order buried the worst of 734 twins
    s = ann["initSort"]
    assert s["none"] == "id" and s["nothing"] == "id"
    assert s["flag"] == "flag" and s["flags"] == "flag" and s["spaced"] == "flag"
    assert s["audit"] == "audit" and s["audits"] == "audit" and s["aiAudit"] == "audit"
    assert s["mixed"] == "flag"              # imagesq's flag key takes the audit score of named audit kinds too
    assert s["parser"] == "id" and s["any"] == "id" and s["aiPending"] == "id"     # no score to sort by
    assert s["parserFlag"] == "flag"
    assert s["set"] == "set"                 # a saved selection keeps its own order
    assert s["unknown"] == "id"              # as Explore: only known QA flags rank (imagesq ranks only those)


def test_annotate_sort_follows_the_filters_only_when_the_link_named_none(ann):
    src = ANNOTATE.read_text(encoding="utf-8")
    assert "let sort = q0.get('sort') || initialSort(filters);" in src
    body = src[src.index("  async function setFilters("):]
    body = body[:body.index("\n  }\n")]
    assert "if (sortImplied || (sort === 'flag' && !flagSortable(filters))) { sort = initialSort(filters);" in body
    # the URL keeps a sort only when it is not the one the filters imply (a reload re-derives it)
    assert "sort: sortImplied ? null : sort" in src


def test_decisions_are_sent_together_and_kept_when_a_send_fails(ann):
    d = ann["dec"]
    assert d["body"] == {"accept": [2], "reject": [1, 4], "pending": [3]}
    assert d["bodyNoPending"] == {"accept": [5], "reject": []}
    assert d["ok"] == {"ok": True, "sent": 2, "left": 0, "body": {"accept": [2], "reject": [1]}}
    # a network failure keeps every decision for the retry (it used to be reported "clean")
    assert d["net"]["ok"] is False and d["net"]["retry"] is True and d["net"]["msg"] == "Failed to fetch"
    assert sorted(d["net"]["left"]) == [[1, "rejected"], [2, "accepted"]]
    # a decision made while the send was on its way wins over the failed one it replaces
    assert d["newer"] == {"ok": False, "retry": True, "left": [[1, "pending"], [2, "accepted"], [7, "rejected"]]}
    assert d["codes"]["408"] == d["codes"]["429"] == d["codes"]["500"] == {"ok": False, "retry": True, "left": 1}
    # an answer a retry cannot change: dropped and reported, never retried for ever
    assert d["codes"]["403"] == d["codes"]["422"] == {"ok": False, "retry": False, "left": 0}
    # a server without a proposals store: nothing to keep
    assert d["codes"]["404"] == d["codes"]["405"] == {"ok": True, "retry": False, "left": 0}
    assert d["empty"] == {"ok": True, "sent": 0}


def test_an_image_opened_again_shows_its_unsent_decisions(ann):
    d = ann["dec"]
    assert d["overlay"] == ["rejected", "pending", "pending", "pending"]
    assert d["overlayNone"] == ["pending"]


def test_flush_reports_decisions_that_did_not_get_through(ann):
    src = ANNOTATE.read_text(encoding="utf-8")
    body = src[src.index("  async function flush("):]
    body = body[:body.index("\n  }\n")]
    # 'clean' only when flushDecisions() says nothing is left unsent; a navigation goes on with them queued
    assert "if (await flushDecisions()) return 'clean';" in body
    assert "await flushDecisions(); return 'clean';" not in body
    assert "return 'failed';" in body
    assert "flush({ nav: true })" in src and "if (flushed === 'failed' && dirty) {" in src
    assert "props = withQueuedDecisions(" in src


def test_threshold_keys_are_buttons_that_step_it(ann):
    src = ANNOTATE.read_text(encoding="utf-8")
    assert "h('span', { class: 'kbd' }, '{')" not in src and "h('span', { class: 'kbd' }, '}')" not in src
    assert "thrStepBtn(-1), thrStepBtn(1)" in src
    btn = src[src.index("  function thrStepBtn(d) {"):]
    btn = btn[:btn.index("\n  }\n")]
    assert "h('button', { type: 'button', class: 'kbd ann-thr-step'" in btn
    assert "'aria-label': `${what} threshold`" in btn and "'Lower' : 'Raise'" in btn
    assert "function stepThr(d) { setThr(thr + 0.05 * d);" in src
    assert "stepThr(K.is(e, '}') ? 1 : -1)" in src           # the keys go the same way


def test_stored_threshold_zero_is_kept(ann):
    assert ann["thr"] == [0.25, 0, 0.4, 0.25, 0.25, 0.25]


def test_point_tool_wheel_steps_both_ways_on_either_axis(ann):
    # Alt+Shift+wheel arrives as a horizontal delta: down / right shrinks, up / left grows
    assert ann["wheel"] == [-0.01, 0.01, -0.01, 0.01, 0]


def test_next_with_proposals_counts_only_what_the_editor_shows(ann):
    assert ann["shown"] == 1                 # only the other-class one; the duplicate, the low one and the decided one are not
    assert ann["shownOnlyHidden"] == 0       # an image like this is skipped by ⇧N
    assert ann["shownZeroThr"] == 1          # at threshold 0 the low one shows; the duplicate never does
    assert ann["shownHiddenCls"] == 0
    assert ann["shownAudit"] == 1            # audit findings point at a box on purpose: always shown


def test_annotate_registers_clear_review_key(ann):
    assert "⇧U" in ann["keys"]


def test_previous_frame_skips_the_same_frame_in_another_folder(ann):
    assert ann["twinOrder"] == ["a/x_00000.jpg", "b/x_00000.jpg", "a/x_00007.jpg", "b/x_00007.jpg", "b/x_00042.jpg",
                                "a/x_00050.jpg", "b/x_00050.jpg"]
    # the frame before b/00007 is 00000 — not a/00007, its twin — and the copy in its own folder
    assert ann["twinPrev"] == ["b/x_00000.jpg", "a/x_00000.jpg", None, None]
    # forward: the twin is skipped, the own folder preferred, a frame only the other folder has is still a step
    assert ann["twinNext"] == ["b/x_00042.jpg", "b/x_00042.jpg", "a/x_00050.jpg", None]
    assert ann["group"] == ["b/x_00050.jpg", "a/x_00050.jpg"]


def test_filmstrip_counts_frames_not_twin_copies(ann):
    # a/ and b/ hold the same video: 7 images, 4 frames (00000, 00007, 00042, 00050), 3 of them in both folders
    assert ann["ord"] == {"of": [0, 0, 1, 1, 2, 3, 3], "n": 4, "multi": 3}
    assert ann["ord3"] == {"of": [0, 0, 0, 1], "n": 2, "multi": 1}      # three copies are still one frame
    assert ann["ord0"] == {"n": 0, "multi": 0}


def test_flag_chip_targets(ann):
    ft = ann["ft"]
    assert ft["gap"] == {"rows": [], "boxes": [[8, 0.75, 0.45, 0.07, 0.06]], "other": None}
    # seq_jump: the moved row and where the frames around put it; a malformed entry is dropped
    assert ft["jump"] == {"rows": [2], "boxes": [[0, 0.3, 0.3, 0.1, 0.1]], "other": None}
    # twins: compare with the other image's boxes
    assert ft["twin"] == {"rows": [], "boxes": [], "other": 136}
    assert ft["leak"]["other"] == 9 and ft["same"]["other"] == 11
    # flicker / count spike: the frame before is the one to compare with; rows once each
    assert ft["flicker"] == {"rows": [1], "boxes": [], "other": 40}
    assert ft["spike"] == {"rows": [], "boxes": [], "other": 7}
    assert ft["shape"] == {"rows": [0], "boxes": [], "other": None}
    assert ft["near"] == {"rows": [], "boxes": [], "other": None}      # found with /similar instead
    assert ft["empty"] == {"rows": [], "boxes": [], "other": None}


def test_box_size_without_a_known_image_size_is_a_share_not_zero(ann):
    # the image file is missing and the index has no size either: never "0×0"
    assert ann["geo"] == ["160×48", "4×23%", "0.4×50%"]


def test_sequence_split_frame_by_frame_still_steps_across_folders(ann):
    assert ann["interPrev"] == "train/c_1.jpg" and ann["interNext"] == "train/c_3.jpg"


def test_position_box_accepts_the_number_as_it_is_shown(ann):
    assert ann["pos"] == [7133, 7133, 7133, 12, 0, None, None, None, 7133]


def test_audit_finding_targets_its_box_by_stored_geometry(ann):
    assert ann["refGeom"] == [[0.5, 0.5, 0.2, 0.2], [0.5, 0.5, 0.2, 0.2], None, None, None]
    assert ann["refShifted"] == 1        # rows shifted: still the box it was made for, never the row's new occupant
    assert ann["refRowOk"] == 1
    assert ann["refGone"] is None and ann["refOtherCls"] is None      # stale, not retargeted to a neighbour
    assert ann["refSwap"] is True        # a class swap's box keeps its old class until the swap is accepted
    assert ann["refNone"] is True        # no stored geometry: the row (and the old IoU search) decides


def test_shift_enter_never_doubles_a_box_a_fix_already_tightens(ann):
    # the detection and the ⤢ fix from it are one object, whatever their order: the fix moves box #3, the
    # detection is a duplicate of where it lands (before, both were accepted: two identical rows)
    assert ann["planDetFirst"] == {"take": ["loose"], "dupes": ["detect"], "left": []}
    assert ann["planFixFirst"] == {"take": ["loose"], "dupes": ["detect"], "left": []}
    assert ann["planSwap"] == {"take": ["swap"], "dupes": ["detect"], "left": []}
    assert ann["planOldPlace"] == {"take": ["loose"], "dupes": [], "left": ["detect"]}
    assert ann["planSpurious"] == {"take": ["spurious"], "dupes": [], "left": ["detect"]}
    assert ann["planTwoDet"] == {"take": ["detect", "missing"], "dupes": ["interp"], "left": []}
    # a fix whose box is gone moves nothing: the detection is judged against the boxes as they are
    assert ann["planStaleRef"] == {"take": ["loose", "detect"], "dupes": [], "left": []}


def test_a_fix_landing_on_a_box_of_its_class_finds_that_box(ann):
    # used by Enter on a ⤢ / ⇄ fix after the detection was accepted: the moved box goes, the one there stays
    assert ann["dupOf"] == [True, None, None, None, None, None]


def test_canvas_labels_are_shortened_in_the_middle_not_cut(ann):
    e = ann["ell"]
    assert e["w"] <= 172 and e["maximal"]
    assert e["text"].startswith("0 person_lying") and e["text"].endswith("fall") and e["text"].count("…") == 1
    assert e["short"] == "0 person" and e["tiny"] == "…"


def test_interpolation_status_says_what_was_left_out(ann):
    i = ann["interp"]
    assert i["unmatched"]["head"] == "Interpolated 2 proposals on 1 frame between A.jpg and B.jpg"
    assert i["unmatched"]["notes"] == [{"text": "2 boxes in A.jpg had no partner in the other keyframe — not interpolated", "warn": True}]
    assert i["unmatched"]["warn"] is True and i["unmatched"]["n"] == 2 and i["unmatched"]["imgs"] == 1
    # adjacent keyframes that both have boxes: the server's reason, never "No boxes to interpolate"
    assert i["adjacent"]["head"] == "Nothing to interpolate between A.jpg and B.jpg — the keyframes are adjacent: no frames in between"
    assert i["adjacent"]["notes"] == [] and i["adjacent"]["n"] == 0
    assert i["existing"]["head"] == "Nothing new to interpolate between A.jpg and B.jpg"
    assert i["existing"]["notes"] == [{"text": "4 boxes already on the frames in between — skipped", "warn": False}]
    assert i["existing"]["warn"] is False
    assert [x["text"] for x in i["both"]["notes"]] == ["1 box in A.jpg and 3 in B.jpg had no partner in the other keyframe — not interpolated",
                                                     "1 box already on the frames in between — skipped"]
    assert i["none"]["head"] == "No boxes to interpolate between A.jpg and B.jpg" and i["none"]["notes"] == []
    assert (i["counts"]["n"], i["counts"]["imgs"]) == (3, 2)


def test_interpolation_status_says_which_boxes_change_class(ann):
    i = ann["interp"]
    msg = "1 box changes class between the keyframes (standing → fallen): the proposals take the class of the nearer keyframe"
    # proposals made AND a fall paired across classes: the server's message is said, with the unpaired box
    assert i["fall"]["head"] == "Interpolated 2 proposals on 2 frames between A.jpg and B.jpg"
    assert i["fall"]["notes"] == [{"text": "1 box in A.jpg had no partner in the other keyframe — not interpolated", "warn": True},
                                  {"text": msg, "warn": False}]
    # nothing new (all already labeled): the class-change message is a note, never the reason nothing ran
    assert i["fallAllThere"]["head"] == "Nothing new to interpolate between A.jpg and B.jpg"
    assert [x["text"] for x in i["fallAllThere"]["notes"]] == ["3 boxes already on the frames in between — skipped", msg]
    assert i["changesNoMsg"]["notes"] == [{"text": "2 boxes change class between the keyframes: the proposals take the class of the nearer keyframe", "warn": False}]
    assert i["otherMsg"]["notes"] == [{"text": "Weighted by position: the file names carry no frame numbers", "warn": False}]
    assert i["otherMsg"]["head"].startswith("Interpolated 1 proposal on 1 frame")
    # the transient note says it in a few words (the Sequence panel keeps the sentences)
    assert i["fall"]["brief"] == "1 box without a partner · 1 class change"
    assert i["both"]["brief"] == "4 boxes without a partner · 1 already labeled"
    assert i["fallAllThere"]["brief"] == "3 already labeled · 1 class change"
    assert i["none"]["brief"] == "" and i["otherMsg"]["brief"] == ""


def test_undo_and_redo_resync_the_class_list(ann):
    # ⌘Z / ⇧⌘Z restore the selection without setSel; every edit path ends in changed(), which must re-mark the
    # highlighted class and the "applies to box #n" header (they kept class_3 and box #5 after the box was gone)
    src = ANNOTATE.read_text(encoding="utf-8")
    body = src[src.index("  function changed() {"):]
    body = body[:body.index("\n  }\n")]
    assert "markActiveClass()" in body and "updateClassHead()" in body
    restore = src[src.index("  function restore(s) {"):]
    assert "sel = s.sel" in restore[:restore.index("\n  }\n")]
    assert "restore(s); changed();" in src


def test_an_unsupported_box_reads_the_same_in_every_view(ann):
    # one glyph and one wording for ai_spurious: Annotate, Explore and Objects draw ✕; the words say "support"
    import re
    static = ANNOTATE.parent.parent
    src = ANNOTATE.read_text(encoding="utf-8")
    glyph = re.search(r"const GLYPH = \{[^}]*spurious: '([^']+)'", src).group(1)
    explore = re.search(r"const AUDIT_GLYPH = \{[^}]*spurious: '([^']+)'", (static / "views/explore.js").read_text(encoding="utf-8")).group(1)
    objects = re.search(r"spurious: \{ glyph: '([^']+)'", (static / "views/objects.js").read_text(encoding="utf-8")).group(1)
    assert glyph == explore == objects == "✕"
    assert re.search(r"const KIND_LABEL = \{[^}]*spurious: 'box without support'", src)
    assert "spurious?" not in src and "looks spurious" not in src
    assert "no support? ${nm}" in src and "no support`" in src


# ---------------------------------------------------------------- server contract: emptied images keep "no label file"

@pytest.fixture()
def client():
    from fastapi.testclient import TestClient
    from fovea.main import app
    return TestClient(app)


@pytest.fixture()
def lds(tmp_path, client):
    """One image with a box, one without a label file, one with an empty label file (a deliberate background)."""
    import uuid
    from PIL import Image
    from fovea import db
    from fovea.core import layout
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"ann-{uuid.uuid4().hex[:6]}"
    for name, rows in (("boxed", "0 0.5 0.5 0.2 0.2\n"), ("nofile", None), ("empty", "")):
        ip = root / "images" / "train" / f"{name}.jpg"
        ip.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (64, 48), (90, 90, 90)).save(ip)
        lp = root / "labels" / "train" / f"{name}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        if rows is not None:
            lp.write_text(rows)
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "ann-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["person"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    ids = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
    yield ds_id, ids, root
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _labels(client, ds, iid):
    r = client.get(f"/api/datasets/{ds}/images/{iid}/labels")
    assert r.status_code == 200, r.text
    return r.json()


def _unlabeled(client, ds):
    return client.get(f"/api/datasets/{ds}/images/ids", params={"labeled": "0"}).json()["ids"]


def test_draw_then_undo_on_an_image_without_a_label_file_removes_the_file(client, lds):
    ds, ids, root = lds
    iid, lp = ids["nofile"], root / "labels" / "train" / "nofile.txt"
    assert iid in _unlabeled(client, ds)
    r = client.put(f"/api/datasets/{ds}/images/{iid}/labels", json={"boxes": [[0, 0.5, 0.5, 0.2, 0.2]], "base_version": None})
    assert r.status_code == 200, r.text
    assert lp.is_file() and r.json()["item"]["has_label"]
    v = r.json()["version"]
    # the editor emptied it again (undo): the file goes, the image is "no label file" again — not a 0-byte background
    r = client.put(f"/api/datasets/{ds}/images/{iid}/labels", json={"boxes": [], "base_version": v, "remove_if_empty": True})
    assert r.status_code == 200, r.text
    assert not lp.exists()
    assert not r.json()["item"]["has_label"] and r.json()["version"] is None
    assert _labels(client, ds, iid)["exists"] is False
    assert iid in _unlabeled(client, ds)


def test_remove_if_empty_creates_nothing_and_needs_an_empty_list(client, lds):
    ds, ids, root = lds
    r = client.put(f"/api/datasets/{ds}/images/{ids['nofile']}/labels", json={"boxes": [], "base_version": None, "remove_if_empty": True})
    assert r.status_code == 200, r.text
    assert not (root / "labels" / "train" / "nofile.txt").exists()
    # boxes win over the flag: a non-empty save is written as always
    r = client.put(f"/api/datasets/{ds}/images/{ids['boxed']}/labels", json={"boxes": [[0, 0.4, 0.4, 0.2, 0.2]], "remove_if_empty": True})
    assert r.status_code == 200 and (root / "labels" / "train" / "boxed.txt").read_text().startswith("0 0.4")


def test_plain_empty_save_still_writes_a_background_file(client, lds):
    ds, ids, root = lds
    lp = root / "labels" / "train" / "boxed.txt"
    r = client.put(f"/api/datasets/{ds}/images/{ids['boxed']}/labels", json={"boxes": []})
    assert r.status_code == 200, r.text
    assert lp.is_file() and lp.read_text() == ""          # deleting every box of a labeled image: a negative sample
    r = client.put(f"/api/datasets/{ds}/images/{ids['nofile']}/labels", json={"boxes": [], "base_version": None})
    assert r.status_code == 200 and (root / "labels" / "train" / "nofile.txt").is_file()   # "Mark as background"


def test_remove_if_empty_is_version_checked(client, lds):
    ds, ids, root = lds
    lp = root / "labels" / "train" / "empty.txt"
    r = client.put(f"/api/datasets/{ds}/images/{ids['empty']}/labels", json={"boxes": [], "base_version": "stale", "remove_if_empty": True})
    assert r.status_code == 409
    assert lp.is_file(), "a file changed elsewhere is never removed"
