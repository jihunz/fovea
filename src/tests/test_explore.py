"""Explore (static/views/explore.js): its pure helpers run in Node, and the server contracts it relies on run through
the API — facet counts under the other filters, working sets (?set=) for big selections, the set filter in review
and export, and the natural-language filter answer. Node-based tests are skipped when Node is not installed.
Run: python -m pytest tests/test_explore.py -q"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
EXPLORE = SRC / "static" / "views" / "explore.js"
NODE = shutil.which("node")


# ------------------------------------------------------------------ Node: pure helpers of explore.js

SCENARIOS = r"""
import { facetValue, gridHits, askResult, boxIssues, issueMatch, qaFlagRows, duplicateOf, pathParts, isolateDigit, controlKeys, findingLabel, refRow, twinDiff, conflictOf, acceptPlan, defaultSort, defaultOrder, nextSort, flagSortable, askUsable, linkOrder } from './views/explore.mjs';
const out = {};
const fx = {
  split: { train: 5, val: 3, 'test-폭': 2 },            // a decomposed (NFD) Korean split name
  review: { approved: 2, any: 2, none: 8 },
  labeled: { labeled: 7, unlabeled: 1, empty: 2, nobox: 2, file: 9 },
  ai: { pending: 1, audit: 4, any: 4, none: 6 },
  cls: { 0: 6, 1: 4 }, issue: { tiny_box: 3, near_dup: 2, any: 5 }, total: 10,
};
out.facets = {
  all_splits: facetValue(fx, 'split', ''), val: facetValue(fx, 'split', 'val'),
  korean_nfc: facetValue(fx, 'split', 'test-폭'), missing_split: facetValue(fx, 'split', 'other'),
  review_any_choice: facetValue(fx, 'review', ''), unreviewed: facetValue(fx, 'review', 'none'), flagged: facetValue(fx, 'review', 'flagged'),
  labeled_all: facetValue(fx, 'labeled', ''), has_boxes: facetValue(fx, 'labeled', '1'), no_file: facetValue(fx, 'labeled', '0'),
  empty_file: facetValue(fx, 'labeled', 'empty'), nobox: facetValue(fx, 'labeled', 'nobox'),
  ai_all: facetValue(fx, 'ai', ''), ai_pending: facetValue(fx, 'ai', 'pending'),
  cls1: facetValue(fx, 'cls', 1), cls9: facetValue(fx, 'cls', 9), tiny: facetValue(fx, 'issue', 'tiny_box'), any_issue: facetValue(fx, 'issue', 'any'),
  no_facets: facetValue(null, 'split', 'val'), no_dim: facetValue({ total: 3 }, 'cls', 0),
};

// 4 columns of 100 px tiles (tile 75 px high) with 10 px gaps, 16 px side padding, 14 px top padding.
const G = { cols: 4, rows: 3, tileW: 100, tileH: 75, rowH: 85, padX: 16, padTop: 14, gap: 10 };
const hits = (x0, y0, x1, y1, total = 12) => [...gridHits(G, total, x0, y0, x1, y1)].sort((a, b) => a - b);
out.band = {
  gap_start_two_by_two: hits(121, 5, 240, 120),          // from the gap after column 0 into column 2, rows 0-1
  inside_one_gap: hits(117, 20, 125, 60),                // only the gap between columns 0 and 1: nothing
  whole_grid: hits(0, 0, 1000, 1000),
  past_the_end: hits(0, 0, 1000, 1000, 10),              // a short last row
  left_margin_column0: hits(2, 30, 20, 40),
  below_last_row_gap: hits(16, 14 + 75 + 2, 450, 14 + 85 - 1), // the gap between rows 0 and 1
};

out.ask = {
  arrays_and_unknown: askResult({ filters: { cls: [1, 3], review: 'none', bogus: 'x', min_boxes: 5, issue: [] }, explanation: 'x' }),
  korean_split: askResult({ filters: { split: 'test-폭' } }),
  sort_string: askResult({ filters: {}, sort: 'audit' }),
  sort_object: askResult({ filters: {}, sort: { key: 'boxes', order: 'desc' } }),
  sort_unknown: askResult({ filters: { review: 'approved' }, sort: 'colour' }),
  bad_order: askResult({ filters: {}, sort: 'name', order: 'sideways' }),
  empty: askResult({}),
};

out.boxIssues = boxIssues([[0, 0.5, 0.5, 0.2, 0.2], [0, 0.5, 0.5, 0.2, 0.2], [7, 0.99, 0.5, 0.1, 0.1], [1, 0.5, 0.5, 0.001, 0.001]], 3);

// issue= matching of a list row after a label edit / QA re-check (null: the row does not carry issues + flags)
const row = (issues, flags, ai_suspect = 0) => ({ issues, flags, ai_suspect });
out.issue = {
  flag_still_there: issueMatch(row([], ['seq_jump']), 'seq_jump'),
  flag_gone: issueMatch(row([], []), 'seq_jump'),
  one_of_two: issueMatch(row(['tiny_box'], []), 'seq_jump,tiny_box'),
  any_with_flag: issueMatch(row([], ['blurry']), 'any'),
  any_only_missing_label: issueMatch(row(['missing_label'], []), 'any'),
  explicit_missing_label: issueMatch(row(['missing_label'], []), 'missing_label'),
  ai_code_with_finding: issueMatch(row([], [], 2), 'ai_missing'),
  ai_code_without: issueMatch(row([], [], 0), 'ai_missing'),
  unknown_row: issueMatch({ id: 1 }, 'seq_gap'),
  no_filter: issueMatch(row([], []), ''),
};
out.rows = {
  jump: qaFlagRows({ code: 'seq_jump', detail: { jumps: [{ row: 4, box: [0, .5, .5, .1, .1], expected: [0, .54, .51, .05, .12] }, { row: 4 }, { row: 1 }] } }),
  flicker: qaFlagRows({ code: 'seq_flicker', detail: { rows: [{ row: 2, cls: 1, expected: 0 }] } }),
  shape: qaFlagRows({ code: 'shape_outlier', detail: { rows: [{ row: 0, kind: 'aspect' }, { row: 0, kind: 'area' }, { row: 3 }] } }),
  gap: qaFlagRows({ code: 'seq_gap', detail: { boxes: [[0, .5, .5, .1, .1]] } }),
  bad: qaFlagRows({ code: 'seq_jump', detail: { jumps: [{ row: -1 }, { row: 'x' }, null] } }),
  none: qaFlagRows(null),
};
const boxes = [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]];
out.dup = {
  same_box_same_class: duplicateOf({ cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, kind: 'detect' }, boxes),
  shifted_a_bit: duplicateOf({ cls: 0, xc: 0.51, yc: 0.5, w: 0.2, h: 0.2, kind: 'detect' }, boxes),
  other_class: duplicateOf({ cls: 1, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, kind: 'detect' }, boxes),
  elsewhere: duplicateOf({ cls: 0, xc: 0.8, yc: 0.8, w: 0.1, h: 0.1, kind: 'detect' }, boxes),
  audit_finding: duplicateOf({ cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, kind: 'loose' }, boxes),
  refine: duplicateOf({ cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, kind: 'refine' }, boxes),
  no_boxes: duplicateOf({ cls: 0, xc: 0.5, yc: 0.5, w: 0.2, h: 0.2, kind: 'detect' }, null),
};
out.path = { nested: pathParts('train/images_GOPR0804_00006.jpg'), flat: pathParts('a.jpg'), korean: pathParts('test-1854-1860-포함/x_01.jpg'), empty: pathParts('') };

// digit keys → class isolation: run a key sequence (each key within the 600 ms window) and collect the classes shown
const digits = (seq, n) => { let buf = ''; return [...seq].map((d) => { const r = isolateDigit(buf, d, n); buf = r.buf; return r.cls; }); };
out.digits = {
  twelve_2_5: digits('25', 12), twelve_1_2: digits('12', 12), twelve_1_1: digits('11', 12), twelve_8_0: digits('80', 12),
  twelve_1_0: digits('10', 12), five_7: digits('7', 5), five_3_4: digits('34', 5), twenty_2_5: digits('25', 26),
  ten_1_0: digits('10', 10), unknown_9: digits('9', 0), zero_first: digits('01', 12),
  waits: [isolateDigit('', '1', 12).buf, isolateDigit('', '2', 12).buf, isolateDigit('', '2', 26).buf, isolateDigit('', '0', 12).buf],
};

// which focused control keeps Enter / Space / arrows (fake elements: tagName, type, role)
const el = (tagName, type = '', role = '') => ({ tagName, type, getAttribute: (k) => (k === 'role' ? role || null : null) });
const grid = { contains: (a) => a && a.inGrid === true };
const tileEl = Object.assign(el('DIV', '', 'option'), { inGrid: true });
out.ctl = {
  button: controlKeys(el('BUTTON'), grid), link: controlKeys(el('A'), grid), select: controlKeys(el('SELECT'), grid),
  range: controlKeys(el('INPUT', 'range'), grid), radio: controlKeys(el('INPUT', 'radio'), grid), checkbox: controlKeys(el('INPUT', 'checkbox'), grid),
  tile: controlKeys(tileEl, grid), div: controlKeys(el('DIV'), grid), none: controlKeys(null, grid), role_button: controlKeys(el('DIV', '', 'button'), grid),
  slider_role: controlKeys(el('DIV', '', 'slider'), grid),
};

const nm = (c) => ['person', 'bicycle', 'car'][c] || `class_${c}`;
out.labels = {
  loose: findingLabel({ kind: 'loose', cls: 0, ref_row: 2, conf: 0.4 }, nm),
  swap: findingLabel({ kind: 'swap', cls: 1, ref_row: 0 }, nm),
  spurious: findingLabel({ kind: 'spurious', cls: 2, ref_row: 5 }, nm),
  missing: findingLabel({ kind: 'missing', cls: 0, conf: 0.61, ref_row: 3 }, nm),
  detect: findingLabel({ kind: 'detect', cls: 2, conf: 0.825 }, nm),
  no_conf: findingLabel({ kind: 'detect', cls: 1, conf: null }, nm),
  loose_no_ref: findingLabel({ kind: 'loose', cls: 0, ref_row: null }, nm),
  refs: [refRow({ kind: 'loose', ref_row: 2 }), refRow({ kind: 'missing', ref_row: 2 }), refRow({ kind: 'detect', ref_row: 1 }), refRow({ kind: 'swap', ref_row: -1 }), refRow(null)],
};

// dup_label_mismatch in Inspect: which boxes of the twin have no match here (drawn dashed), which rows have none there
const T = [[0, .3, .3, .1, .2], [1, .6, .6, .1, .1], [0, .8, .2, .05, .05]];
out.twin = {
  same: twinDiff(T, T.map(b => [...b])),
  extra_there: twinDiff(T.slice(0, 2), T),
  extra_here: twinDiff(T, T.slice(0, 2)),
  other_class: twinDiff(T, [[2, .3, .3, .1, .2], T[1], T[2]]),
  moved: twinDiff(T, [[0, .35, .3, .1, .2], T[1], T[2]]),                 // IoU 1/3: same object, placed differently
  far: twinDiff(T, [[0, .9, .9, .1, .2], T[1], T[2]]),
  near_default: twinDiff(T, [[0, .32, .3, .1, .2], T[1], T[2]]),          // IoU 2/3 agrees at the default 0.5 …
  near_strict: twinDiff(T, [[0, .32, .3, .1, .2], T[1], T[2]], 0.9),      // … not at a stricter QA setting
  tie: twinDiff([[0, .49, .5, .2, .2], [0, .51, .5, .2, .2]], [[0, .5, .5, .2, .2]]),   // equal IoU: as Python's reverse sort
  none_here: twinDiff(null, T.slice(0, 1)), none_there: twinDiff(T.slice(0, 1), undefined),
};
// …and agrees with the QA engine (core/qa.py labels_disagree) on whether two label sets differ: seeded random pairs
let seed = 11;
const rnd = () => (seed = (seed * 1103515245 + 12345) % 2147483648) / 2147483648;
out.twinPairs = [];
for (let k = 0; k < 400; k++) {
  const n = 1 + Math.floor(rnd() * 6), s = rnd() * 0.1;
  const a = Array.from({ length: n }, () => [Math.floor(rnd() * 3), 0.1 + rnd() * 0.8, 0.1 + rnd() * 0.8, 0.02 + rnd() * 0.2, 0.02 + rnd() * 0.2]);
  let b = a.map(x => [x[0], x[1] + (rnd() - 0.5) * s, x[2] + (rnd() - 0.5) * s, x[3] * (1 + (rnd() - 0.5) * s * 3), x[4] * (1 + (rnd() - 0.5) * s * 3)]);
  const r = rnd();
  if (r < 0.15) b.pop();
  else if (r < 0.3) b.push([Math.floor(rnd() * 3), rnd(), rnd(), 0.05, 0.05]);
  else if (r < 0.45) b[0][0] = (b[0][0] + 1) % 3;
  if (rnd() < 0.5) b = b.reverse();
  const minIou = [0.3, 0.5, 0.7, 0.9][k % 4];
  out.twinPairs.push({ a, b, minIou, d: twinDiff(a, b, minIou) });
}

// ⇄ in Inspect: a proposal over a box of ANOTHER class (Annotate's rule), and what ⇧Enter does with the shown ones
const L = [[1, 0.22, 0.89, 0.096, 0.22], [1, 0.54, 0.91, 0.097, 0.17], [0, 0.8, 0.2, 0.1, 0.1]];
const on = (cls, b, conf, id = 0, kind = 'detect') => ({ id, cls, xc: b[1] + 0.003, yc: b[2] - 0.002, w: b[3] * 1.03, h: b[4] * 0.98, conf, kind });
const free = (cls, conf, id, x = 0.3) => ({ id, cls, xc: x, yc: 0.4, w: 0.06, h: 0.12, conf, kind: 'detect' });
out.conflict = {
  first: conflictOf(on(0, L[0], 0.89), L), second: conflictOf(on(0, L[1], 0.81), L),
  same_class: conflictOf(on(0, L[2], 0.7), L), free: conflictOf(free(0, 0.7, 1), L),
  loose: conflictOf(on(0, L[0], 0.9, 0, 'loose'), L), refine: conflictOf(on(0, L[0], 0.9, 0, 'refine'), L),
  no_boxes: conflictOf(on(0, L[0], 0.89), undefined),
  label: findingLabel(on(0, L[0], 0.89), nm, { conflict: 0 }), label_free: findingLabel(free(0, 0.7, 1), nm, { conflict: -1 }),
};
const plan = (list, boxes) => { const r = acceptPlan(list, boxes); return { take: r.take.map(p => p.id), dupes: r.dupes.map(p => p.id), left: r.left.map(p => p.id) }; };
out.plan = {
  conflicts_only: plan([on(0, L[0], 0.89, 1), on(0, L[1], 0.81, 2), on(0, L[0], 0.77, 3)], L),
  mixed: plan([on(0, L[0], 0.89, 1), free(0, 0.7, 8), on(0, L[1], 0.81, 2)], L),
  // two new proposals on one free object: the likelier one is placed; of another class the other waits, of its class it is a copy
  new_other_class: plan([free(2, 0.6, 5), free(0, 0.9, 4)], L),
  new_same_class: plan([free(0, 0.6, 5), free(0, 0.9, 4)], L),
  null_conf_last: plan([free(0, null, 6), free(2, 0.3, 7)], L),
  over_own_class: plan([on(0, L[2], 0.9, 9)], L),
  nothing: plan([], L), no_boxes: plan([on(0, L[0], 0.89, 1)], null),
};

// Which box a finding is about once rows shifted (Inspect names it, the accept job applies the fix to it)
const G0 = [[1, .26, .93, .073, .139], [1, .293, .583, .09, .089], [1, .466, .589, .066, .135], [0, .492, .129, .017, .066], [0, .408, .129, .015, .034], [8, .837, .477, .064, .089]];
const F = (kind, row, cls = G0[row][0], extra = {}) => ({ kind, cls, ref_row: row, ref_box: G0[row].slice(1), xc: G0[row][1], yc: G0[row][2], w: G0[row][3], h: G0[row][4], ...extra });
const shifted = G0.slice(1);                     // box #1 deleted: every row below moves up one
out.refs = {
  unchanged: refRow(F('spurious', 4), G0), shifted: refRow(F('spurious', 4), shifted), loose_shifted: refRow(F('loose', 3), shifted),
  swap_shifted: refRow(F('swap', 2, 0), shifted),  // a swap names a box of the OTHER class
  gone: refRow(F('spurious', 4), G0.slice(0, 4)),  // the flagged box itself was deleted
  other_class_now: refRow(F('spurious', 3), G0.map((b, i) => (i === 3 ? [5, ...b.slice(1)] : b))),
  redrawn: refRow(F('loose', 3), G0.map((b, i) => (i === 3 ? [0, .52, .16, .03, .06] : b))),
  // two identical copies of the box: the one nearest the stored row wins (runner._find_ref_row)
  tie_nearest: refRow(F('spurious', 4), [G0[4], G0[0], G0[1], G0[2], G0[5], G0[4]]),   // copies at rows 0 and 5; row 4 is another box now
  no_ref_box_kept: refRow({ ...F('spurious', 4), ref_box: null }, G0),
  no_ref_box_shifted: refRow({ ...F('spurious', 4), ref_box: null }, shifted),
  no_ref_box_no_row: refRow({ ...F('spurious', 4), ref_box: null, ref_row: null }, G0),
  no_boxes_stored_row: refRow(F('spurious', 4)), missing: refRow({ ...F('missing', 4) }, G0),
  label_shifted: findingLabel(F('spurious', 4), nm, { row: refRow(F('spurious', 4), shifted) }),
  label_gone: findingLabel(F('spurious', 4), nm, { row: null }),
  label_loose_gone: findingLabel(F('loose', 3), nm, { row: null }),
};

// Seeded random cases the Python side replays through the server's own rules (runner.split_conflicts +
// geometry.split_duplicates, runner._find_ref_row): Inspect must count and name exactly what the accept job does.
let s2 = 7;
const r2 = () => (s2 = (s2 * 1103515245 + 12345) % 2147483648) / 2147483648;
const rbox = () => [Math.floor(r2() * 3), 0.1 + r2() * 0.8, 0.1 + r2() * 0.8, 0.04 + r2() * 0.2, 0.04 + r2() * 0.2];
const jit = (b, s) => [b[0], b[1] + (r2() - 0.5) * s, b[2] + (r2() - 0.5) * s, b[3] * (1 + (r2() - 0.5) * s * 4), b[4] * (1 + (r2() - 0.5) * s * 4)];
out.planCases = [];
for (let k = 0; k < 300; k++) {
  const boxes = Array.from({ length: Math.floor(r2() * 5) }, rbox), props = [];
  for (let j = 0, n = 1 + Math.floor(r2() * 5); j < n; j++) {
    const base = boxes.length && r2() < 0.6 ? boxes[Math.floor(r2() * boxes.length)] : (props.length && r2() < 0.5 ? [0, props[0].xc, props[0].yc, props[0].w, props[0].h] : rbox());
    const b = jit(base, r2() * 0.08);
    props.push({ id: j + 1, cls: Math.floor(r2() * 3), xc: b[1], yc: b[2], w: b[3], h: b[4], conf: r2() < 0.15 ? null : Math.round(r2() * 100) / 100, kind: 'detect' });
  }
  out.planCases.push({ boxes, props, plan: plan(props, boxes) });
}
out.refCases = [];
for (let k = 0; k < 400; k++) {
  const orig = Array.from({ length: 1 + Math.floor(r2() * 6) }, rbox), row = Math.floor(r2() * orig.length);
  const kind = ['loose', 'swap', 'spurious'][k % 3], g = orig[row];
  const sug = jit(g, 0.05);
  const p = { kind, cls: kind === 'swap' ? (g[0] + 1) % 3 : g[0], ref_row: r2() < 0.05 ? null : row, ref_box: r2() < 0.2 ? null : g.slice(1), xc: sug[1], yc: sug[2], w: sug[3], h: sug[4] };
  let now = orig.map(b => b.slice());
  const e = r2();
  if (e < 0.2 && row > 0) now.splice(Math.floor(r2() * row), 1);                 // a box above deleted: rows shift
  else if (e < 0.3) now.splice(row, 1);                                           // the box itself deleted
  else if (e < 0.4) now[row] = jit(now[row], 0.15);                               // redrawn
  else if (e < 0.5) now[row][0] = (now[row][0] + 2) % 3;                          // its class changed
  else if (e < 0.6) now.splice(Math.floor(r2() * (now.length + 1)), 0, g.slice());  // an identical copy added
  else if (e < 0.7) now = now.reverse();
  out.refCases.push({ p, boxes: now, row: refRow(p, now) });
}

// The order a list opens in without a sort in its URL, and how it follows filter changes (a sort picked by hand stays)
out.sorting = {
  defaults: [{}, { issue: 'dup_label_mismatch' }, { issue: 'near_dup,seq_gap' }, { issue: 'tiny_box,near_dup' }, { issue: 'ai_missing' },
    { issue: 'ai_swap,blurry' }, { ai: 'audit' }, { ai: 'pending' }, { issue: 'any' }, { issue: 'tiny_box' }, { set: 'tok', issue: 'near_dup' },
    { ai: 'audit', issue: 'seq_jump' }].map(f => [defaultSort(f), defaultOrder(defaultSort(f))]),
  sortable: [flagSortable({ issue: 'blurry' }), flagSortable({ issue: 'ai_loose' }), flagSortable({}), flagSortable(null)],
  // [prev filters, next filters, sort, order] → {sort, order}
  follow: [
    nextSort({}, { issue: 'near_dup' }, 'id', 'asc'),                              // default index → most suspicious first
    nextSort({ issue: 'near_dup' }, {}, 'flag', 'desc'),                           // filter cleared → back to the index
    nextSort({}, { issue: 'near_dup' }, 'boxes', 'asc'),                           // picked by hand: kept
    nextSort({ issue: 'near_dup' }, { issue: 'near_dup', split: 'val' }, 'id', 'asc'),   // index picked over the default: kept
    nextSort({ issue: 'near_dup' }, { issue: 'blurry' }, 'flag', 'asc'),           // direction flipped by hand: kept
    nextSort({ issue: 'near_dup' }, { split: 'val' }, 'flag', 'asc'),              // flag score with no flag left: default again
    nextSort({}, { ai: 'audit' }, 'id', 'asc'),
    nextSort({}, { issue: 'ai_spurious' }, 'id', 'asc'),
    nextSort({ set: 't' }, {}, 'set', 'asc'),
    nextSort({ issue: 'near_dup' }, { set: 't' }, 'flag', 'desc'),                 // "show only the selection": the set's own order
    nextSort({}, { split: 'val' }, 'id', 'desc'),                                  // index, descending by hand: kept
  ],
  // what a link to Annotate carries: [filters, sort, order]
  link: [[{}, 'id', 'asc'], [{ issue: 'near_dup' }, 'flag', 'desc'], [{ issue: 'near_dup' }, 'id', 'asc'], [{}, 'id', 'desc'],
    [{ set: 't' }, 'set', 'asc'], [{ ids: '1,2' }, 'boxes', 'asc'], [{ ai: 'audit' }, 'audit', 'desc'], [{ ids: '4,5' }, 'id', 'asc']].map(a => linkOrder(...a)),
};

// Ask AI gate: the server's verdict on this browser (request_is_local) decides, the hostname only without one
const prov = (o) => ({ id: 'anthropic', kind: 'cloud', guarded: true, needs_key: true, key: { present: true }, capabilities: { generate: true }, status: { available: true }, ...o });
const P0 = { remote_ai_allowed: false }, P1 = { remote_ai_allowed: true };
out.askGate = {
  docker_localhost_but_remote: askUsable(P0, prov(), false, 'localhost'),
  lan_name_but_local: askUsable(P0, prov(), true, 'fovea-lan.test'),
  no_verdict_localhost: askUsable(P0, prov(), null, 'localhost'),
  no_verdict_dot_localhost: askUsable(P0, prov(), null, 'fovea.localhost'),
  no_verdict_lan: askUsable(P0, prov(), null, '192.168.0.12'),
  remote_allowed: askUsable(P1, prov(), false, '192.168.0.12'),
  local_model_remote: askUsable(P0, prov({ id: 'openai_compat', kind: 'local', guarded: false, needs_key: false }), false, '192.168.0.12'),
  keyed_gateway_remote: askUsable(P0, prov({ id: 'openai_compat', kind: 'local', guarded: true, needs_key: false }), false, 'localhost'),
  old_server_cloud_remote: askUsable(P0, prov({ guarded: undefined }), false, 'localhost'),
  no_generate: askUsable(P0, prov({ capabilities: {} }), true, 'localhost'),
  no_key: askUsable(P0, prov({ key: { present: false } }), true, 'localhost'),
  unreachable: askUsable(P0, prov({ status: { available: false } }), true, 'localhost'),
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_out(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("explore-node")
    (d / "views").mkdir()
    (d / "lib").mkdir()
    # Explore's own imports are browser modules: stub them (only their names are needed to load the file).
    (d / "views" / "issues.mjs").write_text("export const ISSUE_LABELS = {};\n")
    (d / "lib" / "media.mjs").write_text("export const thumbPx = () => 256, trackImage = () => {}, createImageLoader = () => ({}), createFrameCache = () => ({});\n")
    (d / "lib" / "annattrs.mjs").write_text((EXPLORE.parent.parent / "lib" / "annattrs.js").read_text(encoding="utf-8"),
                                            encoding="utf-8")     # pure: used as is
    src = EXPLORE.read_text(encoding="utf-8").replace("'./issues.js'", "'./issues.mjs'").replace("'../lib/media.js'", "'../lib/media.mjs'")
    src = src.replace("'../lib/annattrs.js'", "'../lib/annattrs.mjs'")
    (d / "views" / "explore.mjs").write_text(src, encoding="utf-8")
    (d / "run.mjs").write_text(SCENARIOS, encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_explore_js_parses_as_a_module():
    if not NODE:
        pytest.skip("node is not installed")
    r = subprocess.run([NODE, "--input-type=module", "--check"], input=EXPLORE.read_text(encoding="utf-8"), capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr


def test_facet_counts_per_choice(node_out):
    f = node_out["facets"]
    assert f["all_splits"] == 10 and f["val"] == 3 and f["missing_split"] == 0
    assert f["korean_nfc"] == 2, "a composed split name must find the decomposed facet key"
    assert f["review_any_choice"] == 10 and f["unreviewed"] == 8 and f["flagged"] == 0
    assert (f["labeled_all"], f["has_boxes"], f["no_file"], f["empty_file"], f["nobox"]) == (10, 7, 1, 2, 2)
    assert f["ai_all"] == 10 and f["ai_pending"] == 1
    assert (f["cls1"], f["cls9"], f["tiny"], f["any_issue"]) == (4, 0, 3, 5)
    assert f["no_facets"] is None and f["no_dim"] is None, "no facet answer: fall back to dataset-wide counts"


def test_rubber_band_hits_follow_the_layout(node_out):
    b = node_out["band"]
    assert b["gap_start_two_by_two"] == [1, 2, 5, 6]
    assert b["inside_one_gap"] == []
    assert b["whole_grid"] == list(range(12))
    assert b["past_the_end"] == list(range(10))
    assert b["left_margin_column0"] == [0]
    assert b["below_last_row_gap"] == []


def test_ai_filter_answer_is_sanitised(node_out):
    a = node_out["ask"]
    assert a["arrays_and_unknown"]["filters"] == {"cls": "1,3", "review": "none", "min_boxes": "5"}
    assert a["arrays_and_unknown"]["sort"] is None
    assert a["korean_split"]["filters"]["split"] == "test-폭"
    assert (a["sort_string"]["sort"], a["sort_string"]["order"]) == ("audit", "desc")
    assert (a["sort_object"]["sort"], a["sort_object"]["order"]) == ("boxes", "desc")
    assert a["sort_unknown"]["sort"] is None and a["sort_unknown"]["filters"] == {"review": "approved"}
    assert a["bad_order"]["order"] == "asc"
    assert a["empty"] == {"filters": {}, "sort": None, "order": None}


def test_flag_filters_open_most_suspicious_first(node_out):
    d = node_out["sorting"]["defaults"]
    assert d[0] == ["id", "asc"]
    assert d[1] == ["flag", "desc"], "Overview → Data quality → Review (?issue=dup_label_mismatch) must not open by index"
    assert d[2] == ["flag", "desc"] and d[3] == ["flag", "desc"], "any QA flag code among the issues ranks by flag score"
    assert d[4] == ["audit", "desc"] and d[6] == ["audit", "desc"], "audit findings / ai=audit rank by the audit score"
    assert d[5] == ["flag", "desc"], "flag + audit codes: the server's flag key takes the max of both"
    assert d[7] == ["id", "asc"] and d[8] == ["id", "asc"] and d[9] == ["id", "asc"], "no flag to rank by: index"
    assert d[10] == ["set", "asc"], "a saved selection keeps its own order"
    assert d[11] == ["flag", "desc"]
    assert node_out["sorting"]["sortable"] == [True, False, False, False]


def test_sort_follows_the_filters_unless_picked_by_hand(node_out):
    f = [(x["sort"], x["order"]) for x in node_out["sorting"]["follow"]]
    assert f[0] == ("flag", "desc") and f[1] == ("id", "asc")
    assert f[2] == ("boxes", "asc"), "a sort picked by hand survives adding a flag filter"
    assert f[3] == ("id", "asc"), "Index picked over 'most suspicious first' stays when another filter changes"
    assert f[4] == ("flag", "asc"), "a direction flipped by hand stays"
    assert f[5] == ("id", "asc"), "the flag score needs a flag filter: without one the default comes back"
    assert f[6] == ("audit", "desc") and f[7] == ("audit", "desc")
    assert f[8] == ("id", "asc") and f[9] == ("set", "asc")
    assert f[10] == ("id", "desc")


def test_flag_sort_is_offered_and_handed_on_with_its_direction():
    text = EXPLORE.read_text(encoding="utf-8")
    assert "['flag', 'Most suspicious first']" in text
    # the URL leaves out only Explore's own default; Annotate links spell sort AND direction out (its defaults differ)
    assert "if (sort !== defaultSort(filters)) p.set('sort', sort);" in text
    assert "annotate?${stateQuery(" not in text, "Annotate links must use portableQuery (explicit order)"
    assert text.count("annotate?${portableQuery(") == 2
    sel = text[text.index("function selectionQuery"):]
    sel = sel[:sel.index("\n  }\n")]
    assert "putOrder(p, tf.filters)" in sel and "defaultOrder(" not in sel


def test_links_to_annotate_spell_out_any_order_but_the_plain_one(node_out):
    lk = node_out["sorting"]["link"]
    assert lk[0] is None and lk[4] is None and lk[7] is None, "index / a selection's own order: nothing to say"
    assert lk[1] == {"sort": "flag", "order": "desc"}
    assert lk[2] == {"sort": "id", "order": "asc"}, "Index picked over a flag list's default must reach Annotate"
    assert lk[3] == {"sort": "id", "order": "desc"} and lk[5] == {"sort": "boxes", "order": "asc"}
    assert lk[6] == {"sort": "audit", "order": "desc"}


def test_ask_ai_gate_uses_the_servers_view_of_this_browser(node_out):
    g = node_out["askGate"]
    assert g["docker_localhost_but_remote"] is False, "Docker/LAN: localhost in the address bar, but the server sees a remote client"
    assert g["lan_name_but_local"] is True, "a LAN name the server counts as local (FOVEA_LOCAL_NETS / same machine)"
    assert g["no_verdict_localhost"] is True and g["no_verdict_dot_localhost"] is True and g["no_verdict_lan"] is False
    assert g["remote_allowed"] is True and g["local_model_remote"] is True
    assert g["keyed_gateway_remote"] is False, "a local server that is sent the owner's key is guarded like a cloud one"
    assert g["old_server_cloud_remote"] is False
    assert g["no_generate"] is False and g["no_key"] is False and g["unreachable"] is False
    text = EXPLORE.read_text(encoding="utf-8")
    assert "api.get('/api/meta')" in text and "request_is_local" in text
    assert "location.hostname" in text[text.index("async function loadAskProviders"):][:600], "the hostname is only the fallback argument"


def test_box_issues_still_mirror_the_parser(node_out):
    iss = node_out["boxIssues"]
    assert "duplicate_box" in iss[0] and "duplicate_box" in iss[1]
    assert "bad_class" in iss[2] and "out_of_range" in iss[2]
    assert "tiny_box" in iss[3]


def test_issue_filter_matching_after_edits(node_out):
    m = node_out["issue"]
    assert m["flag_still_there"] is True and m["flag_gone"] is False, "a re-checked flag decides whether the tile still matches"
    assert m["one_of_two"] is True and m["any_with_flag"] is True
    assert m["any_only_missing_label"] is False, "issue=any ignores the informational missing_label, like the server"
    assert m["explicit_missing_label"] is True
    assert m["ai_code_with_finding"] is True and m["ai_code_without"] is False
    assert m["unknown_row"] is None and m["no_filter"] is True


def test_qa_flag_rows_point_at_boxes(node_out):
    r = node_out["rows"]
    assert r["jump"] == [4, 1] and r["flicker"] == [2] and r["shape"] == [0, 3]
    assert r["gap"] == [] and r["bad"] == [] and r["none"] == []


def test_duplicate_proposals_are_already_labeled(node_out):
    d = node_out["dup"]
    assert d["same_box_same_class"] is True and d["shifted_a_bit"] is True
    assert d["other_class"] is False and d["elsewhere"] is False
    assert d["audit_finding"] is False and d["refine"] is False and d["no_boxes"] is False


def test_paths_break_only_after_a_slash(node_out):
    p = node_out["path"]
    assert p["nested"] == ["train/", "images_GOPR0804_00006.jpg"]
    assert p["flat"] == ["a.jpg"] and p["korean"] == ["test-1854-1860-포함/", "x_01.jpg"] and p["empty"] == []


def test_digit_isolation_never_names_a_missing_class(node_out):
    d = node_out["digits"]
    # 12 classes (0-11): "2 5" is class 2 then class 5 (never 25); "1 2" is 1 then 2 (12 does not exist)
    assert d["twelve_2_5"] == [2, 5] and d["twelve_1_2"] == [1, 2]
    assert d["twelve_1_1"] == [1, 11] and d["twelve_1_0"] == [1, 10] and d["twelve_8_0"] == [8, 0]
    assert d["five_7"] == [None], "a digit past the last class isolates nothing"
    assert d["five_3_4"] == [3, 4] and d["ten_1_0"] == [1, 0]
    assert d["twenty_2_5"] == [2, 25], "with 26 classes the pair is a class"
    assert d["unknown_9"] == [9] and d["zero_first"] == [0, 1]
    assert d["waits"] == ["1", "", "2", ""], "only a digit that can start an existing class waits for a second one"


def test_focused_controls_keep_their_own_keys(node_out):
    c = node_out["ctl"]
    for k in ("button", "link", "checkbox", "role_button"):
        assert c[k] == {"press": True, "arrows": False}, k
    for k in ("select", "range", "radio", "slider_role"):
        assert c[k]["arrows"] is True and c[k]["press"] is True, k
    for k in ("tile", "div", "none"):
        assert c[k] == {"press": False, "arrows": False}, k


def test_ai_findings_name_their_box(node_out):
    lb = node_out["labels"]
    assert lb["loose"] == "⤢ tighter #3 person" and lb["swap"] == "⇄ #1 → bicycle?" and lb["spurious"] == "✕ #6 no support"
    assert lb["missing"] == "+ 0.61 person", "a missed object has no box yet"
    assert lb["detect"] == "0.82 car" or lb["detect"] == "0.83 car"
    assert lb["no_conf"] == "— bicycle" and lb["loose_no_ref"] == "⤢ tighter person"
    assert lb["refs"] == [2, -1, -1, -1, -1]


def test_proposals_over_another_class_are_marked_and_left_by_shift_enter(node_out):
    c = node_out["conflict"]
    assert c["first"] == 0 and c["second"] == 1, "a proposal on a box of another class names that box (⇄ #n)"
    assert c["same_class"] == -1 and c["free"] == -1 and c["no_boxes"] == -1
    assert c["loose"] == -1 and c["refine"] == -1, "audit findings and refinements point at boxes on purpose"
    assert c["label"] == "0.89 person ⇄#1" and c["label_free"] == "0.70 person"
    p = node_out["plan"]
    assert p["conflicts_only"] == {"take": [], "dupes": [], "left": [1, 2, 3]}, "⇧Enter must not stack boxes on boxes of another class"
    assert p["mixed"] == {"take": [8], "dupes": [], "left": [1, 2]}
    assert p["new_other_class"] == {"take": [4], "dupes": [], "left": [5]}, "the likelier of two new proposals is placed first"
    assert p["new_same_class"] == {"take": [4], "dupes": [5], "left": []}
    assert p["null_conf_last"] == {"take": [7], "dupes": [], "left": [6]}, "no confidence sorts last, as in Annotate"
    assert p["over_own_class"] == {"take": [], "dupes": [9], "left": []}
    assert p["nothing"] == {"take": [], "dupes": [], "left": []} and p["no_boxes"]["take"] == [1]


def test_shift_enter_plan_matches_the_accept_job(node_out):
    """Inspect counts "Accept N" and the ⇄ left to decide with acceptPlan; the accept job (skip_conflicts) decides
    with runner.split_conflicts + geometry.split_duplicates. Both must agree, case for case."""
    from fovea.ai import geometry, runner
    for k, case in enumerate(node_out["planCases"]):
        ok, left = runner.split_conflicts(case["props"], case["boxes"])
        fresh, dups = geometry.split_duplicates(ok, case["boxes"])
        got = {"take": [d["id"] for d in fresh], "dupes": [d["id"] for d in dups], "left": [d["id"] for d in left]}
        assert got == case["plan"], (k, case)
    kinds = {key for c in node_out["planCases"] for key in ("take", "dupes", "left") if c["plan"][key]}
    assert kinds == {"take", "dupes", "left"}, "the random cases must exercise every outcome"


def test_findings_follow_their_box_when_rows_shift(node_out):
    r = node_out["refs"]
    assert r["unchanged"] == 4
    assert r["shifted"] == 3 and r["loose_shifted"] == 2 and r["swap_shifted"] == 1, "box #1 deleted: each finding moves up one"
    assert r["gone"] is None and r["other_class_now"] is None and r["redrawn"] is None, "a box gone or changed since: no #n"
    assert r["tie_nearest"] == 5, "of two identical copies, the one nearest the stored row (as the accept job picks)"
    assert r["no_ref_box_kept"] == 4 and r["no_ref_box_shifted"] is None and r["no_ref_box_no_row"] is None
    assert r["no_boxes_stored_row"] == 4 and r["missing"] == -1
    assert r["label_shifted"] == "✕ #4 no support"
    assert r["label_gone"] == "✕ no support · label changed since" and r["label_loose_gone"] == "⤢ tighter person · label changed since"


def test_inspect_names_the_box_the_accept_job_fixes(node_out):
    """The row Inspect names for a finding (refRow) is the row the accept job applies it to (runner._find_ref_row),
    over random edits since the audit: rows shifted, the box deleted, redrawn, re-classed, copied, reordered."""
    from fovea.ai import runner
    seen = set()
    for k, case in enumerate(node_out["refCases"]):
        p = dict(case["p"])
        want, _known = runner._find_ref_row(case["boxes"], p, p["kind"], set())
        assert case["row"] == want, (k, case)
        seen.add("none" if want is None else "same" if want == p["ref_row"] else "moved")
    assert seen == {"none", "same", "moved"}, seen


def test_twin_diff_names_the_boxes_that_differ(node_out):
    t = node_out["twin"]
    assert t["same"] == {"mine": [], "theirs": []}
    assert t["extra_there"] == {"mine": [], "theirs": [{"index": 2, "kind": "only", "other": -1}]}, "drawn dashed: the box only the twin has"
    assert t["extra_here"] == {"mine": [{"row": 2, "kind": "only", "other": -1}], "theirs": []}, "highlighted: the row only this image has"
    assert t["other_class"] == {"mine": [{"row": 0, "kind": "class", "other": 0}], "theirs": [{"index": 0, "kind": "class", "other": 0}]}
    assert t["moved"]["mine"] == [{"row": 0, "kind": "moved", "other": 0}] and t["moved"]["theirs"] == [{"index": 0, "kind": "moved", "other": 0}]
    assert t["far"]["mine"] == [{"row": 0, "kind": "only", "other": -1}] and t["far"]["theirs"] == [{"index": 0, "kind": "only", "other": -1}]
    assert t["near_default"] == {"mine": [], "theirs": []}
    assert t["near_strict"]["mine"] == [{"row": 0, "kind": "moved", "other": 0}], "the dataset's mismatch IoU decides what agrees"
    assert t["tie"] == {"mine": [{"row": 0, "kind": "only", "other": -1}], "theirs": []}, "ties resolve like core/qa.py (larger row first)"
    assert t["none_here"]["theirs"] == [{"index": 0, "kind": "only", "other": -1}] and t["none_there"]["mine"] == [{"row": 0, "kind": "only", "other": -1}]


def test_twin_diff_agrees_with_the_qa_engine(node_out):
    """Inspect draws a dup_label_mismatch from twinDiff: it must call two label sets different exactly when the QA
    engine does (else a flagged image shows "the labels agree", or an agreeing pair shows differences), and every
    box it reports is consistent (one partner at most, 'class' = other class at the IoU, 'moved' = same class below it)."""
    from fovea.core.labels import iou
    from fovea.core.qa import labels_disagree
    pairs = node_out["twinPairs"]
    differ = 0
    for p in pairs:
        a, b, t, d = p["a"], p["b"], p["minIou"], p["d"]
        why = labels_disagree(a, b, t)
        empty = not d["mine"] and not d["theirs"]
        assert (why is None) == empty, (a, b, t, why, d)
        differ += why is not None
        rows, idx = [x["row"] for x in d["mine"]], [x["index"] for x in d["theirs"]]
        assert len(set(rows)) == len(rows) and len(set(idx)) == len(idx)
        if why == "count":
            assert any(x["kind"] == "only" for x in d["mine"] + d["theirs"])
        paired = {(x["row"], x["other"], x["kind"]) for x in d["mine"] if x["kind"] != "only"}
        assert paired == {(x["other"], x["index"], x["kind"]) for x in d["theirs"] if x["kind"] != "only"}
        for i, j, kind in paired:
            v = iou(a[i], b[j])
            if kind == "class":
                assert a[i][0] != b[j][0] and v >= t
            else:
                assert a[i][0] == b[j][0] and min(0.1, t) <= v < t, (a[i], b[j], v, t)
    assert 60 < differ < len(pairs) - 60, "the random pairs cover both outcomes"


# ------------------------------------------------------------------ source checks (cheap drift guards)

def _fn(text, name):
    """The body of `function name(` / `async function name(` up to the next function at the same indent."""
    m = re.search(rf"\n(\s*)(?:async )?function {name}\(.*?\n\1(?:async )?function |\n(\s*)(?:async )?function {name}\(.*", text, re.S)
    assert m, name
    return m.group(0)


def test_ask_ai_findings_stay_on_their_image():
    """Inspect's "Ask AI": the follow-up read of the findings is awaited into a local and assigned only while the
    same image is on screen; the busy state is per image; Delete/⇧Enter only see the shown image's suggestions."""
    text = EXPLORE.read_text(encoding="utf-8")
    assert "preds = await fetchPreds" not in text, "assigning after an await shows image A's findings on image B"
    body = _fn(text, "askAI")
    assert re.search(r"const fresh = await fetchPreds\(target\);", body)
    assert re.search(r"if \(!closed && item && item\.id === target\.id\) \{ preds = fresh;", body)
    assert "asking.has(item.id)" in body and "const asking = new Set()" in text
    assert re.search(r"const visiblePreds = \(\) => preds\.filter\(p => ownPred\(p\)", text)
    assert re.search(r"const ownPred = \(p\) => !!item && \(p\.image_id == null \|\| p\.image_id === item\.id\)", text)


def test_inspect_decisions_are_undoable_in_order():
    """Delete (reject) and ⇧Enter (accept) in Inspect go on the same undo stack as reviews; ⌘Z takes back the newest
    action — never an older review behind them."""
    text = EXPLORE.read_text(encoding="utf-8")
    rej, acc = _fn(text, "rejectVisible"), _fn(text, "acceptVisible")
    assert "pushUndoEntry({ kind: 'decide'" in rej and "pushUndoEntry({ kind: 'decide'" in acc
    assert "snapshot: res.snapshot_id" in acc
    und = _fn(text, "undo")
    assert "u.kind === 'decide'" in und and "u.kind === 'note'" in und
    dec = _fn(text, "undoDecision")
    assert "/snapshots/${u.snapshot}/restore" in dec and "pending: u.ids" in dec
    bulk = _fn(text, "bulkProposals")
    assert bulk.count("kind: 'note'") == 2, "bulk accept/reject stop ⌘Z from silently undoing an older review"


def test_late_ai_search_answer_does_not_replace_the_list():
    text = EXPLORE.read_text(encoding="utf-8")
    sub = _fn(text, "submitAsk")
    assert "state: stateQuery().toString()" in sub
    assert re.search(r"if \(inspect \|\| stateQuery\(\)\.toString\(\) !== asked\) \{ askReady = ", sub)
    assert "askReady" in _fn(text, "askEscape") and "askReadyEl()" in _fn(text, "renderAsk")


def test_similar_strip_is_bound_to_its_image_at_once():
    text = EXPLORE.read_text(encoding="utf-8")
    body = _fn(text, "scheduleSimilar")
    head = body.split("simTimer = setTimeout")[0]
    assert "simFor = it.id" in head and "++simSeq" in head, "the token must be set before the timer, not when it fires"
    assert "my !== simSeq" in body


def test_leaving_explore_writes_the_focused_image_first():
    """syncUrl writes img= 350 ms after a move: E or the Annotate button right after an arrow must flush it, so
    browser Back returns to the same tile."""
    text = EXPLORE.read_text(encoding="utf-8")
    assert re.search(r"function leaveTo\(url\) \{ syncUrl\(true\); fovea\.router\.navigate\(url\); \}", text)
    assert "function annotateId(id) { leaveTo(" in text
    nav = re.findall(r"fovea\.router\.navigate\(", text)
    assert len(nav) == 3, "only leaveTo and Inspect's in-Explore jumps (which close Inspect first) navigate directly"


def test_review_progress_belongs_to_its_filters():
    text = EXPLORE.read_text(encoding="utf-8")
    upd = _fn(text, "updateCount")
    assert "progressFor !== facetKeyOf(filters)" in upd and "filters.review ? null" in upd
    assert "progressFor = key" in _fn(text, "recountProgress")


def test_grid_keys_leave_focused_controls_alone():
    text = EXPLORE.read_text(encoding="utf-8")
    on = re.search(r"const onKey = \(e\) => \{(.*?)\n  \};", text, re.S).group(1)
    assert "controlKeys(document.activeElement, vgrid)" in on
    assert re.search(r"if \(ctl\.press && \(k === 'Enter' \|\| keys\.is\(e, 'Space'\)\)\) return;", on)
    assert "ctl.arrows" in on
    ins = re.search(r"const key = \(e\) => \{(.*?)\n    \};", text, re.S).group(1)
    assert "controlKeys(document.activeElement, null).arrows" in ins
    assert "focusedOrFirst()" in on, "E / Enter / Space / S act on the first tile when nothing is focused yet"


def test_inspect_text_is_readable_in_the_light_theme():
    """Inspect is dark in both themes: its faint text and badges must not take the light theme's tokens."""
    css = (SRC / "static" / "css" / "explore.css").read_text(encoding="utf-8")
    block = re.search(r"\.inspect\.ex-inspect \{(.*?)\}", css, re.S).group(1)
    t3 = re.search(r"--text-3: (#[0-9a-fA-F]{6})", block).group(1)
    faint = re.search(r"\.inspect-side \.faint \{ color: (#[0-9a-fA-F]{6}); \}", css).group(1)
    for fg in (t3, faint):
        for bg in ("#222325", "#141517"):          # a button in the panel, the panel
            l1, l2 = sorted((_luminance(fg), _luminance(bg)), reverse=True)
            assert (l1 + 0.05) / (l2 + 0.05) >= 4.5, (fg, bg)
    assert re.search(r"\.inspect-side \.sec h4 \.badge \{ background: rgba\(255,255,255,\.1\); color: #cfcfcf; \}", css)
    assert "ex-inspect" in EXPLORE.read_text(encoding="utf-8")


def test_toolbar_parts_give_way_before_controls_leave_the_screen():
    css = (SRC / "static" / "css" / "explore.css").read_text(encoding="utf-8")
    assert re.search(r"\.explore-toolbar \.result-count \{ flex: 0 50 auto; min-width: \d+px; overflow: hidden;", css)
    assert re.search(r"\.explore-toolbar \.view-btn \.lbl \{ min-width: 0; overflow: hidden; text-overflow: ellipsis; \}", css)
    assert re.search(r"\.explore-toolbar > \.select \{ flex: 0 1000 auto;", css)

def test_inspect_defers_to_the_shells_g_chord():
    """Inspect listens on window capture, ahead of the shell: it must ask the shell whether the G chord still waits
    (and its fallback window must equal the shell's), or a key 0.9-1.6 s after G reviews / asks the AI."""
    app = (SRC / "static" / "app.js").read_text(encoding="utf-8")
    text = EXPLORE.read_text(encoding="utf-8")
    assert re.search(r"fovea\.chordPending\s*=\s*\(\)\s*=>\s*chord\s*===\s*'g'", app), "app.js must expose fovea.chordPending()"
    assert "fovea.chordPending()" in text
    shell_ms = int(re.search(r"chordTimer = setTimeout\(\(\) => setChord\(null\), (\d+)\)", app).group(1))
    assert int(re.search(r"const CHORD_MS = (\d+)", text).group(1)) == shell_ms


def test_mismatch_chip_draws_what_differs_and_esc_hides_it():
    """The "Duplicates labeled differently" chip reads the twin's boxes and draws the ones with no match here as dashed
    ghosts (labelled under the box, never over this image's own label), highlights the rows with no match there, and
    explains both; Esc hides what a pressed chip draws before it closes Inspect."""
    text = EXPLORE.read_text(encoding="utf-8")
    assert "/api/datasets/${ds.id}/images/${id}`" in _fn(text, "twinBoxes")
    lt = _fn(text, "loadTwin")
    assert "mismatchIou()" in lt and "twin.forId !== forId" in lt, "a late answer for another image is dropped"
    assert "/qa/settings" in text and "mismatch_iou" in text, "the dataset's own agreement IoU, as the QA engine uses"
    assert re.search(r"if \(code === 'dup_label_mismatch'\) \{ const tw = twinNow\(\); return tw \? tw\.mine\.map\(x => x\.row\)", text)
    ov = _fn(text, "overlays")
    assert "ghostLayer(tw.theirs.map(x => tw.boxes[x.index])" in ov and "`other image: ${" in ov
    assert "ghostLayer(ex," in ov, "the sequence flags' expected boxes use the same ghost style"
    gl = _fn(text, "ghostLayer")
    assert "top: '100%'" in gl and "bottom: 'auto'" in gl
    assert "twinLegend()" in _fn(text, "flagDetailEl")
    leg = _fn(text, "twinLegend")
    for s in ("only in the other image", "with another class there", "placed differently there", "only here", "Esc"):
        assert s in leg, s
    ins = re.search(r"const key = \(e\) => \{(.*?)\n    \};", text, re.S).group(1)
    assert "if (keys.is(e, 'Escape')) { done(); if (flagFocus) focusFlag(flagFocus); else close(); return; }" in ins
    rq = _fn(text, "refreshQa")
    assert "twinCache.clear(); twin = null;" in rq, "after a label re-check the twin is read again"


def test_a_failed_tile_shows_only_its_reason():
    """fovea.css hides a failed tile's <img> with `.pic.is-error > img` (0,2,1); explore.css's `.vgrid .pic img
    { display: block }` has the same specificity and loads later, so explore.css must hide it again — or the browser's
    broken-image box fills half the tile and pushes "file missing" aside."""
    css = (SRC / "static" / "css" / "explore.css").read_text(encoding="utf-8")
    shown = css.index(".vgrid .pic img { display: block;")
    hidden = re.search(r"\.vgrid \.pic\.is-error > img \{ display: none; \}", css)
    assert hidden and hidden.start() > shown


def test_label_row_enter_is_not_also_a_click():
    """lib/ui.js clicks any role=button on Enter unless the event was handled: the row's own Enter (select + zoom)
    must prevent that, or the click toggles the selection straight off again."""
    text = EXPLORE.read_text(encoding="utf-8")
    m = re.search(r"onKeydown: \(e\) => \{ if \(e\.key === 'Enter'[^}]*\}", text)
    assert m and "e.preventDefault()" in m.group(0) and "zoomToBox()" in m.group(0)


def test_single_image_results_in_the_grid_are_not_toasts():
    """docs/visual-comfort.md §3: quick single-item results are confirmed in place (the grid's status line), toasts
    are for batch results and errors."""
    text = EXPLORE.read_text(encoding="utf-8")
    body = re.search(r"function flash\([^)]*\) \{(.*?)\n", text).group(1)
    assert "sayInGrid" in body and "toast" not in body


def _luminance(hexcol):
    rgb = [int(hexcol[i:i + 2], 16) / 255 for i in (1, 3, 5)]
    lin = [c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4 for c in rgb]
    return 0.2126 * lin[0] + 0.7152 * lin[1] + 0.0722 * lin[2]


def test_ai_proposal_badge_is_readable_over_any_frame():
    """White 10.5 px text needs 4.5:1; a translucent badge lightens over a white frame, so it must be opaque."""
    css = (SRC / "static" / "css" / "explore.css").read_text(encoding="utf-8")
    m = re.search(r"\.badge\.ai \{ background: (#[0-9a-fA-F]{6}); color: #fff; \}", css)
    assert m, "the .badge.ai background must be an opaque #rrggbb"
    ratio = (1.05) / (_luminance(m.group(1)) + 0.05)
    assert ratio >= 4.5, ratio


def _explore_keys():
    text = EXPLORE.read_text(encoding="utf-8")
    keys = json.loads(re.search(r"const FILTER_KEYS = (\[[^\]]*\])", text).group(1).replace("'", '"'))
    sorts = re.findall(r"\['([a-z_]+)', '[^']+'\]", re.search(r"const SORTS = \[(.*?)\];", text).group(1))
    return keys, sorts


def test_every_explore_filter_and_sort_is_known_to_the_server():
    from fovea.api.common import FILTER_KEYS
    from fovea.core.imagesq import SORT_KEYS
    keys, sorts = _explore_keys()
    assert "set" in keys
    assert set(keys) <= set(FILTER_KEYS), set(keys) - set(FILTER_KEYS)
    assert sorts and set(sorts) <= set(SORT_KEYS), set(sorts) - set(SORT_KEYS)


# ------------------------------------------------------------------ server contracts Explore relies on

def _make_dataset(tmp_path):
    """train: 6 images, val: 4 images. Classes 0/1; image k of train has class k % 2, every image a class-1 box."""
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"ds-{uuid.uuid4().hex[:6]}"
    for split, n in (("train", 6), ("val", 4)):
        for k in range(n):
            ip = root / "images" / split / f"{split}_{k:03d}.jpg"
            ip.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (96, 72), (40 + 20 * k, 90, 120)).save(ip)
            lp = root / "labels" / split / f"{split}_{k:03d}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            rows = [f"1 0.700000 0.400000 0.150000 0.250000"]
            if split == "train" and k % 2 == 0:
                rows.insert(0, "0 0.400000 0.500000 0.200000 0.300000")
            lp.write_text("\n".join(rows) + "\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "exp-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["standing", "fallen"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fastapi.testclient import TestClient
    from fovea.ai import registry
    from fovea.main import app
    registry.reset()
    return TestClient(app)


@pytest.fixture()
def ds(tmp_path):
    ds_id = _make_dataset(tmp_path)
    yield ds_id
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _ids(client, ds, **q):
    r = client.get(f"/api/datasets/{ds}/images/ids", params=q)
    assert r.status_code == 200, r.text
    return r.json()["ids"]


def test_facets_count_each_choice_under_the_other_filters(client, ds):
    r = client.get(f"/api/datasets/{ds}/images/facets", params={"split": "val", "cls": "0"})
    assert r.status_code == 200, r.text
    fx = r.json()["facets"]
    for dim in ("split", "review", "labeled", "cls", "issue", "ai"):
        assert isinstance(fx.get(dim), dict), dim
    # split ignores the split filter (train has class-0 images; val has none) …
    assert fx["split"].get("train") == 3 and fx["split"].get("val", 0) == 0
    # … and cls ignores the class filter: under split=val, class 1 is in all 4 images, class 0 in none.
    assert int(fx["cls"].get("1", fx["cls"].get(1, 0))) == 4
    assert fx["review"]["none"] + fx["review"]["any"] == 0, "review counts follow split=val AND cls=0 (no image)"
    assert {"labeled", "unlabeled", "empty", "nobox", "file"} <= set(fx["labeled"])
    assert {"pending", "audit", "any", "none"} <= set(fx["ai"])


def test_selection_sets_keep_the_order_and_count(client, ds):
    all_ids = _ids(client, ds, sort="name", order="desc")
    pick = all_ids[1:8]
    r = client.post(f"/api/datasets/{ds}/sets", json={"ids": pick})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["token"] and s["count"] == len(pick)
    assert _ids(client, ds, set=s["token"]) == pick, "a selection keeps the order Explore listed it in"
    # 'all matching' frozen from filters + sort, and 'all matching except a few' frozen from ids
    r = client.post(f"/api/datasets/{ds}/sets", json={"filters": {"split": "train"}, "sort": "name", "order": "asc"})
    assert r.json()["count"] == 6
    every = _ids(client, ds, split="train", sort="name", order="asc")
    r = client.post(f"/api/datasets/{ds}/sets", json={"ids": [i for i in every if i not in every[:2]]})
    assert r.json()["count"] == 4
    lst = client.get(f"/api/datasets/{ds}/images", params={"set": r.json()["token"], "limit": 50}).json()
    assert lst["total"] == 4 and not ({it["id"] for it in lst["items"]} & set(every[:2]))


def test_set_filter_drives_review_and_exports(client, ds):
    pick = _ids(client, ds, split="val")[:3]
    tok = client.post(f"/api/datasets/{ds}/sets", json={"ids": pick}).json()["token"]
    r = client.put(f"/api/datasets/{ds}/review", json={"filters": {"set": tok}, "status": "approved"})
    assert r.status_code == 200, r.text
    approved = _ids(client, ds, review="approved")
    assert sorted(approved) == sorted(pick)
    text = client.post(f"/api/datasets/{ds}/export/list", json={"filters": {"set": tok}, "style": "rel"}).text
    assert len([ln for ln in text.splitlines() if ln.strip()]) == 3
    z = client.post(f"/api/datasets/{ds}/export/zip-link", json={"filters": {"set": tok}, "include_unlabeled": True, "include_yaml": True})
    assert z.status_code == 200, z.text
    assert z.json()["url"] and z.json().get("images") == 3
    z2 = client.post(f"/api/datasets/{ds}/export/zip-link", json={"filters": {"ids": ",".join(map(str, pick[:2]))}})
    assert z2.json().get("images") == 2


def test_flag_sort_ranks_the_filtered_flags_and_sets_keep_it(client, ds):
    """What Explore's default for ?issue=<flag> asks the server: sort=flag, most suspicious first by the score of the
    flags the filter names (another flag on the image does not count), and a selection set frozen in that order."""
    import time
    from fovea.jobs import jobs
    t0 = time.time()
    while jobs.active_all(ds) and time.time() - t0 < 60:     # the QA run a scan queues rewrites image_flags
        time.sleep(0.05)
    db.execute("DELETE FROM image_flags WHERE dataset_id=?", (ds,))
    ids = _ids(client, ds, sort="id")
    scores = {ids[0]: 0.2, ids[3]: 0.9, ids[5]: 0.5, ids[7]: 0.7}
    db.executemany("INSERT INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
                   [(i, ds, "dup_label_mismatch", sc, "{}") for i, sc in scores.items()]
                   + [(ids[0], ds, "blurry", 5.0, "{}")])          # a stronger flag of another code
    want = sorted(scores, key=lambda i: -scores[i])
    assert _ids(client, ds, issue="dup_label_mismatch", sort="flag", order="desc") == want
    assert _ids(client, ds, issue="dup_label_mismatch", sort="flag") == want, "flag is descending by default on the server too"
    assert _ids(client, ds, issue="dup_label_mismatch", sort="flag", order="asc") == want[::-1]
    lst = client.get(f"/api/datasets/{ds}/images", params={"issue": "dup_label_mismatch", "sort": "flag", "order": "desc", "limit": 2}).json()
    assert [it["id"] for it in lst["items"]] == want[:2] and lst["total"] == 4
    tok = client.post(f"/api/datasets/{ds}/sets", json={"filters": {"issue": "dup_label_mismatch"}, "sort": "flag", "order": "desc"}).json()["token"]
    assert _ids(client, ds, set=tok) == want, "Annotate opens a set in the order it was frozen in"


def test_expired_selection_is_410_with_a_code(client, ds):
    r = client.get(f"/api/datasets/{ds}/images", params={"set": "no-such-set", "limit": 5})
    assert r.status_code == 410
    d = r.json()["detail"]
    assert d["code"] == "set_expired" and "24 h" in d["message"]


def test_natural_language_filter_answer_fits_explore(client, ds):
    keys, sorts = _explore_keys()
    r = client.post("/api/ai/nl-filter", json={"dataset_id": ds, "text": "검토 안 한 val 이미지 중 fallen", "provider": "mock"})
    if r.status_code in (404, 405):
        pytest.skip("this server has no /api/ai/nl-filter")
    assert r.status_code == 200, r.text
    out = r.json()
    assert isinstance(out.get("filters"), dict) and isinstance(out.get("explanation", ""), str)
    assert set(out["filters"]) <= set(keys), set(out["filters"]) - set(keys)
    sk = out.get("sort")
    sk = sk.get("sort") or sk.get("key") if isinstance(sk, dict) else sk
    assert sk in (None, "") or sk in sorts or sk == "seed"
    # the compiled filters are usable as-is by the image list
    q = {k: (",".join(map(str, v)) if isinstance(v, list) else v) for k, v in out["filters"].items()}
    assert client.get(f"/api/datasets/{ds}/images", params={**q, "limit": 1}).status_code == 200


# ------------------------------------------------------------------ QA flags after a label edit (what Explore polls)

def _seq_dataset(tmp_path):
    """train/camX_0001..0005: a textured scene with one static box; in frame 3 the box slipped 45 px (seq_jump)."""
    import random
    from PIL import ImageDraw
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    W, H = 320, 240
    rng = random.Random(11)
    bg = Image.new("RGB", (W, H), (120, 120, 120))
    d = ImageDraw.Draw(bg)
    for _ in range(40):
        x, y, w, h = rng.randrange(W), rng.randrange(H), rng.randrange(8, 70), rng.randrange(8, 60)
        d.rectangle([x, y, x + w, y + h], fill=tuple(rng.randrange(256) for _ in range(3)))
    S = [1, 0.75, 0.30, 60 / W, 40 / H]
    root = tmp_path / f"seq-{uuid.uuid4().hex[:6]}"
    for k in range(1, 6):
        im = bg.copy()
        ImageDraw.Draw(im).rectangle([(S[1] - S[3] / 2) * W, (S[2] - S[4] / 2) * H, (S[1] + S[3] / 2) * W, (S[2] + S[4] / 2) * H], fill=(230, 230, 40))
        ip = root / "images" / "train" / f"camX_{k:04d}.jpg"
        ip.parent.mkdir(parents=True, exist_ok=True)
        im.save(ip, "JPEG", quality=92)
        box = [1, S[1] - 45 / W, S[2], S[3], S[4]] if k == 3 else S
        lp = root / "labels" / "train" / f"camX_{k:04d}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(f"{box[0]} {box[1]:.6f} {box[2]:.6f} {box[3]:.6f} {box[4]:.6f}\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "exq-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["a", "b"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id, S


def test_flag_details_and_the_recheck_explore_waits_for(client, tmp_path):
    from fovea.core import qa
    from fovea.jobs import Job, jobs
    ds_id, S = _seq_dataset(tmp_path)
    try:
        t0 = __import__("time").time()
        while jobs.active_for(ds_id, "qa") is not None and __import__("time").time() - t0 < 60:   # a scan hook may have queued one
            __import__("time").sleep(0.05)
        qa.run_qa(ds_id, Job(id="qa-" + ds_id, kind="qa"), ["sequence"])
        ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        i3 = ids["train/camX_0003.jpg"]
        r = client.get(f"/api/datasets/{ds_id}/images/{i3}/qa").json()
        assert r["refreshing"] is False
        jump = next(f for f in r["flags"] if f["code"] == "seq_jump")
        j = jump["detail"]["jumps"][0]
        # Inspect reads exactly these: the row it highlights, the dashed expected box, the frames to compare
        assert j["row"] == 0 and len(j["expected"]) == 5 and j["expected"][0] == 1
        assert jump["detail"]["prev_id"] == ids["train/camX_0002.jpg"] and jump["detail"]["next_id"] == ids["train/camX_0004.jpg"]
        rows = client.post(f"/api/datasets/{ds_id}/images/by-ids", json={"ids": [i3]}).json()["items"]
        assert "seq_jump" in rows[0]["flags"], "rereadFlags takes tile flags from /images/by-ids"
        fx = client.get(f"/api/datasets/{ds_id}/images/facets", params={"issue": "seq_jump"}).json()["facets"]
        assert fx["issue"].get("seq_jump") == 1
        # fix the box: the image is 'refreshing' until the debounced re-check has run, then the flag is gone
        lab = client.get(f"/api/datasets/{ds_id}/images/{i3}/labels").json()
        put = client.put(f"/api/datasets/{ds_id}/images/{i3}/labels", json={"boxes": [list(S)], "base_version": lab["version"]})
        assert put.status_code == 200, put.text
        assert client.get(f"/api/datasets/{ds_id}/images/{i3}/qa").json()["refreshing"] is True
        assert client.get(f"/api/datasets/{ds_id}/qa/summary", params={"auto": 0}).json()["refreshing"] >= 1
        assert qa.flush_label_rechecks(ds_id, timeout=30)
        r = client.get(f"/api/datasets/{ds_id}/images/{i3}/qa").json()
        assert r["refreshing"] is False and not [f for f in r["flags"] if f["code"] == "seq_jump"]
        assert client.get(f"/api/datasets/{ds_id}/qa/summary", params={"auto": 0}).json()["refreshing"] == 0
        fx = client.get(f"/api/datasets/{ds_id}/images/facets", params={"issue": "seq_jump"}).json()["facets"]
        assert not fx["issue"].get("seq_jump"), "facets fetched after the re-check are current (no stale server cache)"
        assert "seq_jump" not in client.post(f"/api/datasets/{ds_id}/images/by-ids", json={"ids": [i3]}).json()["items"][0]["flags"]
    finally:
        for t in ("ai_preds", "ai_runs", "image_flags", "image_hash", "boxes", "images"):
            db.execute(f"DELETE FROM {t} WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


# ------------------------------------------------------------------ what ⌘Z in Inspect sends (undo of a decision)

def _wait_job(job_id, timeout=60):
    import time
    from fovea.jobs import jobs
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(job_id)
        if j and j.status in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


def test_inspect_reject_and_accept_can_be_taken_back(client, ds):
    """Inspect's Delete rejects the shown suggestions and ⇧Enter accepts them; ⌘Z then calls proposals/decide
    {pending} (and, after an accept, restores the image's label file from the accept's snapshot). Rejected
    suggestions are never offered again, so this round trip must bring them back exactly."""
    from fovea.ai import store
    iid = _ids(client, ds, split="val")[0]
    lp = db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))["label_path"]
    before = Path(lp).read_bytes()
    run = store.create_run(ds, "propose", "mock", "mock-fast", {})
    preds = store.insert_preds(run, ds, iid, [
        {"cls": 0, "xc": 0.25, "yc": 0.3, "w": 0.2, "h": 0.2, "conf": 0.9, "kind": "detect"},
        {"cls": 1, "xc": 0.2, "yc": 0.8, "w": 0.1, "h": 0.1, "conf": 0.8, "kind": "detect"}], "pending")
    ids = [p["id"] for p in preds]
    pending = lambda: sorted(p["id"] for p in client.get(f"/api/datasets/{ds}/images/{iid}/proposals").json()["preds"]
                             if p.get("state", "pending") == "pending" and p["image_id"] == iid)
    assert pending() == sorted(ids)
    # Delete, then ⌘Z
    assert client.post(f"/api/datasets/{ds}/proposals/decide", json={"accept": [], "reject": ids}).status_code == 200
    assert pending() == []
    r = client.post(f"/api/datasets/{ds}/proposals/decide", json={"accept": [], "reject": [], "pending": ids})
    assert r.status_code == 200 and r.json()["updated"] == 2
    assert pending() == sorted(ids)
    # ⇧Enter, then ⌘Z: the label file comes back byte for byte, the suggestions are pending again
    r = client.post(f"/api/datasets/{ds}/proposals/accept", json={"image_ids": [iid], "kinds": ["detect"]})
    assert r.status_code == 200, r.text
    job = _wait_job(r.json()["job"]["id"])
    res = job.to_dict()["result"] or job.to_dict()["extra"]
    assert job.status == "done" and res["added"] == 2 and res.get("snapshot_id") is not None
    assert Path(lp).read_bytes() != before and pending() == []
    r = client.post(f"/api/datasets/{ds}/snapshots/{res['snapshot_id']}/restore", json={"image_ids": [iid]})
    assert r.status_code == 200, r.text
    assert _wait_job(r.json()["job"]["id"]).status == "done"
    assert Path(lp).read_bytes() == before
    # The restore itself puts the accept's decisions back (snapshot write log); Explore's ⌘Z still sends decide
    # {pending} after it (for an older server, or a reject that has no snapshot) — a no-op here, never an error.
    assert pending() == sorted(ids)
    r = client.post(f"/api/datasets/{ds}/proposals/decide", json={"accept": [], "reject": [], "pending": ids})
    assert r.status_code == 200 and r.json()["updated"] in (0, 2)
    assert pending() == sorted(ids)
    row = client.get(f"/api/datasets/{ds}/images/{iid}").json()["item"]
    assert row["n_boxes"] == 1, "the restored file is indexed again (one box, as before the accept)"
    db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds,))
    db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))


# ------------------------------------------------------------------ ⇄ in Inspect and the grid; findings after a row shift

def test_inspect_and_grid_accepts_leave_cross_class_overlaps_to_decide():
    """⇧Enter in Inspect counts and sends only what acceptPlan takes and asks the job to skip conflicts; the grid's
    bulk accept asks the same and offers the images left; Inspect names finding boxes as they are now (refOf)."""
    text = EXPLORE.read_text(encoding="utf-8")
    body = _fn(text, "acceptVisible")
    assert "planOf(vis)" in body and "skip_conflicts: true" in body
    assert "refOf(p) !== null" in body, "a fix whose box is gone has nothing to apply to"
    bulk = _fn(text, "bulkProposals")
    assert "skip_conflicts: true" in bulk and "conflict_images" in bulk
    sec = _fn(text, "proposalsSection")
    assert "planOf(vp).left" in sec and "Accept ${nAcc}" in sec
    assert "refRow(p)" not in sec, "rows must be resolved against the current labels (refOf), not the stored ref_row"
    assert re.search(r"const refOf = \(p\) => refRow\(p, item && item\.boxes\)", text)
    assert re.search(r"const predLabel = \(p\) => findingLabel\(p, .*\{ row: refOf\(p\), conflict: conflictAt\(p\) \}\)", text)


def test_accept_with_skip_conflicts_writes_only_free_proposals(client, ds):
    """What Inspect's ⇧Enter sends: a proposal over a box of another class stays pending (⇄, decided in Annotate) and
    is counted, a same-class copy is "already labeled", a free one is written; the job lists the images left."""
    from fovea.ai import store
    iid = _ids(client, ds, split="val")[0]          # one box: class 1 at (0.7, 0.4, 0.15, 0.25)
    lp = Path(db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))["label_path"])
    run = store.create_run(ds, "propose", "mock", "mock-fast", {})
    preds = store.insert_preds(run, ds, iid, [
        {"cls": 0, "xc": 0.702, "yc": 0.401, "w": 0.152, "h": 0.248, "conf": 0.9, "kind": "detect"},   # on the class-1 box
        {"cls": 1, "xc": 0.701, "yc": 0.4, "w": 0.15, "h": 0.25, "conf": 0.8, "kind": "detect"},       # repeats it
        {"cls": 0, "xc": 0.2, "yc": 0.7, "w": 0.1, "h": 0.1, "conf": 0.7, "kind": "detect"}], "pending")  # a free spot
    try:
        r = client.post(f"/api/datasets/{ds}/proposals/accept", json={"image_ids": [iid], "kinds": ["detect", "prev_frame", "interp", "refine"], "skip_conflicts": True})
        assert r.status_code == 200, r.text
        job = _wait_job(r.json()["job"]["id"])
        res = job.to_dict()["result"] or {}
        assert job.status == "done", job.error
        assert (res["added"], res["duplicates"], res["conflicts"]) == (1, 1, 1), res
        assert res.get("conflict_images") == [iid]
        rows = [ln.split()[0] for ln in lp.read_text().splitlines() if ln.strip()]
        assert sorted(rows) == ["0", "1"], "only the free proposal is written: no class-0 box on the class-1 object"
        left = [p["id"] for p in client.get(f"/api/datasets/{ds}/images/{iid}/proposals").json()["preds"] if p.get("state", "pending") == "pending"]
        assert left == [preds[0]["id"]]
    finally:
        db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds,))
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))
