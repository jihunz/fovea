"""Objects (patch view): every labelled box as a crop, for fast class QA.

  GET   /api/datasets/{ds}/boxes   one row per box; every image filter + box class; sorted; offset or keyset paging
  GET   /api/crop/{image_id}       a square, EXIF-aware crop centred on one box, cached on disk
  PATCH /api/datasets/{ds}/boxes   reassign / delete boxes by (image_id, row), snapshot first

A box is addressed by (image_id, row): its 0-based position among the parsed boxes of its label file, which
is the order the index stores them in (boxes.id per image). An edit rewrites only the lines it targets, so
everything else in the file (confidence columns, polygon tails, number formatting, unparsable lines) is
kept byte for byte.
"""
from __future__ import annotations

import hashlib
import json
import logging
import math
import os
import re
import stat
import tempfile
import threading
import time
from array import array
from collections import OrderedDict, defaultdict
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Dict, List, NamedTuple, Optional, Sequence, Tuple

from fastapi import APIRouter, Body, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse
from PIL import Image, ImageOps

from .. import db
from ..config import DATA_DIR, MAX_IMAGE_PIXELS
from ..core import imagesq
from ..core import labels as labels_mod
from ..core.labels import LABEL_LOCK, MAX_CLASS_ID, parse_label_text, read_label_snapshot
from ..core.scanner import refresh_dataset_counts, refresh_image
from ..core.thumbs import ImageTooLarge, png_over_cap, to_display_rgb
from ..jobs import Job, jobs
from .common import filters_from, get_dataset_or_404

router = APIRouter(tags=["objects"])
log = logging.getLogger(__name__)

# ------------------------------------------------------------------ listing

MAX_LIMIT = 2000
SUSPECT_KINDS = ("swap", "loose", "spurious")     # audit findings that point at an existing box (ref_row)

# Aspect in PIXELS: normalised w/h is skewed by the frame shape (a square object on 16:9 reads as 0.56).
_ASPECT = ("(CASE WHEN b.h > 0 THEN (b.w * COALESCE(NULLIF(i.width, 0), 1)) * 1.0 / "
           "(b.h * COALESCE(NULLIF(i.height, 0), 1)) ELSE 0.0 END)")
# Stable under edits: geometry and image only (box ids change whenever an image is re-read, classes when
# reassigned), so a shuffled list does not reshuffle itself after every keystroke. {seed} is an int literal.
_RANDOM = ("((b.image_id * 2654435761 + CAST(ABS(b.xc) * 1000003 AS INTEGER) * 40503 + "
           "CAST(ABS(b.yc) * 1000003 AS INTEGER) * 9973 + {seed} * 7919) % 4294967296)")

# sort -> (key expressions before the b.id tie-break, needs the images join). One direction for the whole
# tuple, so keyset paging is a single row-value comparison.
SORTS: Dict[str, Tuple[Tuple[str, ...], bool]] = {
    "class": (("b.cls", "i.rel_path"), True),
    "area": (("(b.w * b.h)",), False),
    "aspect": ((_ASPECT,), True),
    "random": ((_RANDOM,), False),
    "audit": (("COALESCE(sx.s, -1.0)", "COALESCE(au.score, 0.0)"), False),
    "name": (("i.rel_path",), True),
}
DEFAULT_ORDER = {"audit": "desc"}

# Per-box suspicion from pending audit findings (ObjectLab-style), matched to boxes by the ground-truth
# geometry whenever the finding carries it, which survives rows shifting after a deletion (and coming back on
# undo): `swap` and `spurious` store the GT box as their own box; `loose` stores the suggested tighter box,
# with the GT box in `ref_box` (JSON [xc, yc, w, h]) on audits that record it. An older `loose` finding
# without `ref_box` can only go by its row. _suspects() applies the same rules to one page.
_GEOM_EQ = ("ABS(bb.xc - {a}.{x}) < 1e-6 AND ABS(bb.yc - {a}.{y}) < 1e-6 "
            "AND ABS(bb.w - {a}.{w}) < 1e-6 AND ABS(bb.h - {a}.{h}) < 1e-6")
_HAS_REF = "rx IS NOT NULL AND ry IS NOT NULL AND rw IS NOT NULL AND rh IS NOT NULL"


def _ref_coord(k: int, name: str) -> str:
    """SQL for one coordinate of ai_preds.ref_box: a JSON [xc, yc, w, h] (a leading class id tolerated) or
    {xc, yc, w, h}; NULL when the column holds anything else, which then falls back to ref_row."""
    return (f"(CASE WHEN json_valid(ref_box) THEN COALESCE(json_extract(ref_box, '$.{name}'), "
            f"json_extract(ref_box, '$[' || (MAX(0, json_array_length(ref_box) - 4) + {k}) || ']')) END)")


def _audit_cte(with_ref_box: bool, scope: Optional[str] = None) -> str:
    """The per-box suspicion CTE (params: those of `scope`, then the dataset id twice). `scope` is a SELECT of
    image ids: only those images' findings are read. A box's suspicion depends on its own image's findings
    only, so a short list scoped to its images costs milliseconds instead of a pass over every pending finding
    of the dataset (~0.2 s at 96k findings)."""
    coords = ([_ref_coord(k, n) for k, n in enumerate(("xc", "yc", "w", "h"))] if with_ref_box
              else ["NULL"] * 4)
    fi = f"fi AS MATERIALIZED ({scope}), " if scope else ""
    only, only_p = ((" AND image_id IN (SELECT image_id FROM fi)", " AND p.image_id IN (SELECT image_id FROM fi)")
                    if scope else ("", ""))
    return f"""
WITH {fi}lb AS MATERIALIZED (
  SELECT image_id, ref_row, COALESCE(conf, 0.5) AS s,
         {coords[0]} AS rx, {coords[1]} AS ry, {coords[2]} AS rw, {coords[3]} AS rh
  FROM ai_preds WHERE dataset_id = ? AND state = 'pending' AND kind = 'loose'{only}
), lr AS MATERIALIZED (
  SELECT image_id, ref_row, MAX(s) AS s FROM lb WHERE ref_row IS NOT NULL AND NOT ({_HAS_REF}) GROUP BY image_id, ref_row
), sus AS (
  SELECT bb.id AS bid, COALESCE(p.conf, 0.5) AS s FROM ai_preds p
  JOIN boxes bb ON bb.image_id = p.image_id AND {_GEOM_EQ.format(a='p', x='xc', y='yc', w='w', h='h')}
  WHERE p.dataset_id = ? AND p.state = 'pending' AND p.kind IN ('swap', 'spurious'){only_p}
  UNION ALL
  SELECT bb.id, lb.s FROM lb
  JOIN boxes bb ON bb.image_id = lb.image_id AND {_GEOM_EQ.format(a='lb', x='rx', y='ry', w='rw', h='rh')}
  WHERE {_HAS_REF}
  UNION ALL
  SELECT rk.bid, lr.s FROM lr JOIN (
    SELECT bb.id AS bid, bb.image_id, ROW_NUMBER() OVER (PARTITION BY bb.image_id ORDER BY bb.id) - 1 AS rn
    FROM boxes bb WHERE bb.image_id IN (SELECT image_id FROM lr)
  ) rk ON rk.image_id = lr.image_id AND rk.rn = lr.ref_row
), sx AS (SELECT bid, MAX(s) AS s FROM sus GROUP BY bid)
"""


_ref_box_seen = False


def has_ref_box() -> bool:
    """Whether ai_preds has the `ref_box` column (added by a later migration; older databases lack it)."""
    global _ref_box_seen
    if not _ref_box_seen:
        _ref_box_seen = any(r["name"] == "ref_box" for r in db.query("PRAGMA table_info(ai_preds)"))
    return _ref_box_seen


def ref_geom(v: Any) -> Optional[Tuple[float, float, float, float]]:
    """(xc, yc, w, h) from a stored ref_box, by the rules of _ref_coord; None means "use ref_row"."""
    if v in (None, ""):
        return None
    try:
        obj = json.loads(v) if isinstance(v, (str, bytes)) else v
    except ValueError:
        return None
    if isinstance(obj, dict):
        vals = [obj.get(k) for k in ("xc", "yc", "w", "h")]
    elif isinstance(obj, list) and len(obj) >= 4:
        vals = obj[len(obj) - 4:]
    else:
        return None
    if not all(isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x) for x in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2]), float(vals[3])


_REVIEW_REF = re.compile(r"(?<![\w.])r\.")


def _csv_ints(v: Optional[str]) -> List[int]:
    out = []
    for x in str(v or "").split(","):
        x = x.strip()
        if x.lstrip("-").isdigit():
            try:
                # Clamped to SQLite's 64-bit range (a larger id is a binding OverflowError; it matches nothing
                # either way); "²".isdigit() is True but int() rejects it.
                out.append(min(max(int(x), -(1 << 63)), (1 << 63) - 1))
            except ValueError:
                continue
    return out


_AUDIT_JOIN = " LEFT JOIN sx ON sx.bid = b.id LEFT JOIN ai_audit au ON au.image_id = b.image_id"
SCOPE_SHARE = 3                  # a list holding at most 1/3 of the dataset's boxes reads only its own images' findings


def _box_query(ds_id: str, img_f: Dict[str, Any], classes: List[int], sort: str, seed: int):
    """(prefix, prefix params, from+where sql, params, key expressions) for one filtered box list. The prefix
    is the whole-dataset form of the audit CTE: it names the list (frozen orders are keyed on it); a query
    that reads a short list runs it scoped to the list's images instead (_scoped_prefix)."""
    exprs_t, needs_images = SORTS[sort]
    exprs = [e.replace("{seed}", str(int(seed))) for e in exprs_t] + ["b.id"]
    # attr= narrows the BOXES here (a box with occlusion:2), region= stays an image filter
    pairs = imagesq.attr_pairs(img_f.get("attr")) if img_f.get("attr") else []
    box_attrs = [(n, v) for n, v in (pairs or []) if n != "region"]
    if box_attrs:
        regions = [f"{n}:{v}" for n, v in pairs if n == "region"]
        img_f = {**{k: v for k, v in img_f.items() if k != "attr"}, **({"attr": ",".join(regions)} if regions else {})}
    img_where, img_params = imagesq.build_where(ds_id, img_f)
    filtered = img_where.strip() != "i.dataset_id = ?"
    join_i = needs_images or filtered
    sql = "FROM boxes b"
    params: List[Any] = []
    prefix, prefix_params = "", []
    if join_i:
        sql += " JOIN images i ON i.id = b.image_id"
        if _REVIEW_REF.search(img_where):
            sql += " LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path"
    if sort == "audit":
        prefix, prefix_params = _audit_cte(has_ref_box()), [ds_id, ds_id]
        sql += _AUDIT_JOIN
    where = ["b.dataset_id = ?"]
    params.append(ds_id)
    if join_i:
        where.append(img_where)
        params.extend(img_params)
    if classes:
        where.append("b.cls IN (%s)" % ",".join("?" * len(classes)))
        params.extend(classes)
    by_name: Dict[str, List[str]] = {}
    for n, v in box_attrs:
        by_name.setdefault(n, []).append(v)
    for n, vals in by_name.items():
        where.append("b.id IN (SELECT a.box_id FROM box_attrs a WHERE a.dataset_id = ? AND "
                     f"CAST(json_extract(a.attrs, ?) AS TEXT) IN ({','.join('?' * len(vals))}))")
        params.extend([ds_id, '$."' + n.replace('"', "") + '"'] + vals)
    return prefix, prefix_params, f"{sql} WHERE {' AND '.join(where)}", params, exprs


def _scoped_prefix(ds_id: str, body: str, params: List[Any], total: int) -> Optional[Tuple[str, List[Any]]]:
    """The audit CTE limited to the images of this list's boxes (prefix, prefix params), or None when the
    list is a large share of the dataset: finding its images then costs more than the findings it skips."""
    if _AUDIT_JOIN not in body:
        return None
    ds = db.query_one("SELECT box_count FROM datasets WHERE id=?", (ds_id,))
    if not (total <= ORDER_MIN_BOXES or total * SCOPE_SHARE <= int((ds or {}).get("box_count") or 0)):
        return None
    scope = "SELECT DISTINCT b.image_id AS image_id " + body.replace(_AUDIT_JOIN, "")
    return _audit_cte(has_ref_box(), scope), list(params) + [ds_id, ds_id]


def _images_prefix(ds_id: str, prefix: str, ids_json: str) -> Tuple[str, List[Any]]:
    """The list's prefix for a query that reads only the images in `ids_json` (a JSON array of image ids)."""
    if not prefix:
        return "", []
    return _audit_cte(has_ref_box(), "SELECT value AS image_id FROM json_each(?)"), [ids_json, ds_id, ds_id]


_ds_cls_index: List[bool] = []


def _steered(body: str) -> str:
    """`body` for reading a whole list in the audit order. Without image filters that reads only b.id and
    b.image_id, which idx_boxes_ds_cls covers; with one dataset in the planner statistics SQLite rates the
    non-covering idx_boxes_ds_area the same and may take it — a row lookup per box, 2x slower at 300k boxes."""
    if _AUDIT_JOIN not in body or " JOIN images i " in body:
        return body
    if not _ds_cls_index:
        _ds_cls_index.append(any(r["name"] == "idx_boxes_ds_cls" for r in db.query("PRAGMA index_list(boxes)")))
    return body.replace("FROM boxes b ", "FROM boxes b INDEXED BY idx_boxes_ds_cls ", 1) if _ds_cls_index[0] else body


def _parse_after(after: Optional[str], n: int) -> Optional[list]:
    if after in (None, ""):
        return None
    try:
        key = json.loads(after)
    except ValueError:
        raise HTTPException(400, "after must be a JSON array (the k of the last item you have)")
    if (not isinstance(key, list) or len(key) != n
            or not all(isinstance(v, (int, float, str)) and not isinstance(v, bool) for v in key)
            or any(isinstance(v, int) and not -(1 << 63) <= v < (1 << 63) for v in key)):    # SQLite INTEGER range
        raise HTTPException(400, f"after must be the {n}-element sort key (k) of an item from this sort")
    return key


def _chunks(seq: Sequence, n: int = 500):
    for i in range(0, len(seq), n):
        yield seq[i:i + n]


# ------------------------------------------------------------------ frozen orders
# A list sorted by a computed key (AI suspicion, aspect, class + image name, shuffle) costs SQLite a full sort of
# every matching box for every page: 0.3-0.7 s a page at 260k boxes, deep offsets and keyset continuations alike
# (the audit order also re-derives each box's suspicion from the pending findings). Such a list is therefore
# frozen once — the whole ordered key list: the sort keys, the box id and its image, ~32 bytes a box — and its
# later pages are slices of it; `after` is a binary search in it. A first page is still answered by SQL (a one-off
# peek such as Overview's class examples never pays for, or holds, a whole list); a list opened for browsing
# (`counts=1`, the Objects grid) starts freezing in the background right then, so the pages after it are ready.
#
# A frozen order is used only while its fingerprint holds: the label index (datasets.updated_at and counts, the
# highest box id — box ids change whenever an image is re-read) plus whatever else the query reads (pending AI
# rows and audit scores, reviews, QA flags), and for ORDER_TTL seconds. Label writes are what moves while a list
# is reviewed (every reassign or delete), so they do not void it: every writer announces the images it wrote
# (LABEL_WRITE_HOOKS), and an order whose label fingerprint moved by announced writes only is patched — those
# images' entries out, their current keys in: milliseconds, not a new sort. Anything else that moved the label
# index (a rescan, a re-read nobody announced) voids it. When only the other parts moved (an AI decision, a
# review, new QA flags) or the TTL ran out, a keyset continuation — the grid scrolling on from boxes it holds,
# which came from this order — is still answered from it while a new order is sorted in the background; other
# pages use SQL or wait for the new sort. A list SQLite reads straight off one covering index (sort=area, through
# idx_boxes_ds_area) is paged by SQL at any depth.

ORDER_TTL = 120.0
STALE_MAX = 900.0                # older than this, an order is not served even to a continuation
ORDER_MIN_BOXES = 2000           # smaller lists sort in a few milliseconds: not worth holding
ORDER_BUDGET = 1_500_000         # boxes held across every frozen order (least recently used dropped beyond)
ORDER_MAX = 8
SHALLOW_ROWS = 2000              # pages above this depth (and without `after`) are answered by SQL
PATCH_MAX_IMAGES = 2000          # more images written since an order was taken: sort it again
LOG_IMAGES = 20_000              # announced writes remembered per dataset (image ids), for patching orders
LOG_WRITES = 4096                # ... and at most this many writes

_frozen: "OrderedDict[tuple, _Frozen]" = OrderedDict()
_frozen_lock = threading.Lock()
_frozen_build: Dict[tuple, threading.Lock] = {}
_frozen_patch: Dict[tuple, threading.Lock] = {}
_label_gen: Dict[str, int] = defaultdict(int)          # announced label writes per dataset
_label_log: Dict[str, List[Tuple[int, Tuple[int, ...]]]] = defaultdict(list)   # recent ones: (gen, image ids)
_label_floor: Dict[str, int] = defaultdict(int)        # writes up to this gen have left the log
_label_mark: Dict[str, tuple] = {}                     # (gen, dataset roll-up) right after the latest one
_top_mark: List[Optional[int]] = [None]                # highest box id right after the latest one of any dataset
_warming: set = set()
_warmer = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fovea-box-order")   # one sort at a time, own connection


class _Q(NamedTuple):
    """One box list: the CTE prefix (whole-dataset form) and its params, FROM + WHERE and its params, sort keys."""
    ds_id: str
    prefix: str
    pparams: tuple
    body: str
    params: tuple
    exprs: tuple

    @property
    def key(self) -> tuple:
        return (self.ds_id, self.prefix, self.body, ", ".join(self.exprs), self.pparams + self.params)


class _Frozen:
    """One list's sort keys in ascending order, column-wise (the last column is the box id), each entry's image,
    the fingerprint it holds for — (label part, the rest) — and the last announced label write it includes.
    Never changed once made (pages are read from it without a lock): a patch makes a new one."""
    __slots__ = ("version", "cols", "imgs", "gen", "text", "n", "at")

    def __init__(self, version: tuple, cols: list, imgs: array, gen: int, text: Optional[List[bool]] = None,
                 at: Optional[float] = None) -> None:
        self.version, self.cols, self.imgs, self.gen = version, cols, imgs, gen
        self.text = text if text is not None else [isinstance(c, list) and bool(c) and isinstance(c[0], str)
                                                   for c in cols]
        self.n = len(cols[-1]) if cols else 0
        self.at = time.monotonic() if at is None else at

    def key(self, i: int) -> tuple:
        return tuple(c[i] for c in self.cols)

    def accepts(self, after: Sequence[Any]) -> bool:
        """Whether `after` compares with these keys the way SQLite compares them (text with text, numbers with numbers)."""
        return len(after) == len(self.cols) and all(
            isinstance(v, str) == t and (t or math.isfinite(v)) for v, t in zip(after, self.text))

    def position(self, after: Optional[Sequence[Any]], desc: bool) -> int:
        """Reading position of the first key beyond `after` (0 without one)."""
        if after is None:
            return 0
        a = tuple(after)
        lo, hi = 0, self.n
        while lo < hi:                       # first index whose key is > a (ascending) or >= a (descending)
            mid = (lo + hi) // 2
            k = self.key(mid)
            if k < a or (not desc and k == a):
                lo = mid + 1
            else:
                hi = mid
        return self.n - lo if desc else lo


@contextmanager
def _key_lock(table: Dict[tuple, threading.Lock], key: tuple):
    """One holder per list (the grid asks for several pages at once: sort or patch once, not once per page)."""
    with _frozen_lock:
        lk = table.setdefault(key, threading.Lock())
    with lk:
        try:
            yield
        finally:
            with _frozen_lock:
                if table.get(key) is lk:
                    table.pop(key, None)


def _label_state(conn, ds_id: str) -> tuple:
    """The label part of a fingerprint: the dataset's roll-up (updated_at moves with every re-index) and the
    highest box id (an image's boxes get new ids whenever it is re-read)."""
    ds = conn.execute("SELECT updated_at, scanned_at, image_count, label_count, box_count FROM datasets WHERE id=?",
                      (ds_id,)).fetchone()
    return (tuple(ds) if ds else None, conn.execute("SELECT MAX(id) FROM boxes").fetchone()[0])


def note_index_change(ds_id: str, image_ids: Sequence[int]) -> None:
    """These images' boxes were re-read from their label files and the dataset's roll-up refreshed: the
    dataset's frozen orders patch them in on their next use instead of being sorted again. Called by the
    LABEL_WRITE_HOOK, and for a re-read that wrote nothing (an edit refused as stale, a no-op edit)."""
    conn = db.connect()
    with _frozen_lock:
        state = _label_state(conn, ds_id)        # under the lock: marks are taken in the order they are numbered
        _label_gen[ds_id] += 1
        gen = _label_gen[ds_id]
        log = _label_log[ds_id]
        log.append((gen, tuple(sorted({int(i) for i in image_ids}))))
        held = sum(len(ids) for _g, ids in log)
        while len(log) > 1 and (held > LOG_IMAGES or len(log) > LOG_WRITES):
            g0, ids0 = log.pop(0)
            held -= len(ids0)
            _label_floor[ds_id] = g0
        _label_mark[ds_id] = (gen, state[0])
        _top_mark[0] = state[1]


def _on_label_write(ds_id: str, image_ids: List[int]) -> None:
    """LABEL_WRITE_HOOK (runs after the write, the re-index and the roll-up): see note_index_change."""
    note_index_change(ds_id, image_ids)


def drop_frozen_orders(ds_id: Optional[str] = None) -> None:
    with _frozen_lock:
        for k in [k for k in _frozen if ds_id is None or k[0] == ds_id]:
            _frozen.pop(k, None)


_write_hooks = getattr(labels_mod, "LABEL_WRITE_HOOKS", None)
if isinstance(_write_hooks, list) and not any(getattr(h, "__qualname__", "") == "_on_label_write"
                                              and getattr(h, "__module__", "") == __name__ for h in _write_hooks):
    _write_hooks.append(_on_label_write)


def _pending_fp(conn, ds_id: str, kinds: Optional[Sequence[str]]) -> tuple:
    """Which pending AI rows exist (a decision changes the set, not only its size): count + two id checksums,
    read from the covering index idx_ai_preds_ds."""
    sql = ("SELECT COUNT(*), TOTAL(id), TOTAL((id * 2654435761) % 1000003) FROM ai_preds "
           "WHERE dataset_id = ? AND state = 'pending'")
    args: List[Any] = [ds_id]
    if kinds:
        sql += " AND kind IN (%s)" % ",".join("?" * len(kinds))
        args.extend(kinds)
    return tuple(conn.execute(sql, args).fetchone())


def _order_version(ds_id: str, prefix: str, body: str) -> tuple:
    """Fingerprint of everything a box list reads: (label part, the rest). Cheap: a few index-only aggregates,
    only for the tables the query touches (pending AI rows for `ai=` / `issue=ai_*` and the audit order, flags,
    reviews)."""
    conn = db.connect()
    parts: List[Any] = []
    if "ai_preds" in body:
        parts.append(_pending_fp(conn, ds_id, None))
    elif "ai_preds" in prefix:
        parts.append(_pending_fp(conn, ds_id, SUSPECT_KINDS))
    if "ai_audit" in body:
        parts.append(tuple(conn.execute("SELECT COUNT(*), TOTAL(score), TOTAL(image_id * score) FROM ai_audit "
                                        "WHERE dataset_id = ?", (ds_id,)).fetchone()))
    if "image_flags" in body:
        parts.append(tuple(conn.execute("SELECT COUNT(*), TOTAL(rowid) FROM image_flags WHERE dataset_id = ?",
                                        (ds_id,)).fetchone()))
    if "reviews" in body:
        parts.append(tuple(conn.execute("SELECT COUNT(*), MAX(updated_at) FROM reviews WHERE dataset_id = ?",
                                        (ds_id,)).fetchone()))
    if "box_attrs" in body or "image_regions" in body:
        parts.append(tuple(conn.execute("SELECT COUNT(*), TOTAL(box_id) FROM box_attrs WHERE dataset_id = ?",
                                        (ds_id,)).fetchone()))
    return (_label_state(conn, ds_id), tuple(parts))


def _columns(rows: List[tuple], n: int) -> Optional[list]:
    """Rows of sort keys -> compact columns (int64 / float64 arrays; text interned); None for a column of mixed
    or NULL values, which only the SQL path compares exactly like SQLite."""
    cols: list = []
    for c in (list(zip(*rows)) if rows else [()] * n):
        kinds = {type(v) for v in c}
        if kinds <= {int}:
            cols.append(array("q", c))
        elif kinds == {float}:
            cols.append(array("d", c))
        elif kinds == {int, float}:
            cols.append(list(c))
        elif kinds == {str}:
            seen: Dict[str, str] = {}
            cols.append([seen.setdefault(v, v) for v in c])
        else:
            return None
    return cols


def _page_plan(sql: str, params: List[Any]) -> str:
    """How SQLite would produce one page: 'sort' (it sorts the whole list first), 'index' (one covering-index
    walk in list order: any page costs milliseconds) or 'walk' (list order from an index, but every row skipped
    by OFFSET is joined or looked up)."""
    try:
        plan = [str(r[3]) for r in db.connect().execute("EXPLAIN QUERY PLAN " + sql, params).fetchall()]
    except Exception:  # noqa: BLE001 — a plan we cannot read: assume the expensive one
        return "sort"
    if any("ORDER BY" in d and "TEMP B-TREE" in d for d in plan):
        return "sort"
    return "index" if len(plan) == 1 and "COVERING INDEX" in plan[0] else "walk"


def _frozen_order(q: _Q, page_sql: str, page_params: List[Any], shallow: bool, warm: bool, cont: bool = False,
                  count: Optional[Callable[[], int]] = None, later: Optional[list] = None) -> Optional[_Frozen]:
    """The frozen order for this list (patched, or built now, when out of date); None means "page it with SQL".
    `shallow`: a first page (no `after`, small offset); `warm`: the list is being opened for browsing; `cont`: a
    keyset continuation, which an order that only the non-label parts outdated still answers while a new one
    is sorted in the background; `count`: the list's length, worked out on demand; `later`: background sorts to
    start once this page is answered (not alongside it: both would compete for the same CPU)."""
    key = q.key
    sort_later = later.append if later is not None else _sort_later
    with _frozen_lock:
        fz = _frozen.get(key)
    if fz is not None:
        version = _order_version(q.ds_id, q.prefix, q.body)
        cur = _caught_up(q, fz, version)
        keep = False
        if cur is not None:
            age = time.monotonic() - cur.at
            fresh = cur.version == version and age < ORDER_TTL
            keep = age < STALE_MAX
            if fresh or (cont and keep):
                if not fresh:
                    sort_later(q)
                with _frozen_lock:
                    if _frozen.get(key) is cur:
                        _frozen.move_to_end(key)
                return cur
            if keep:
                sort_later(q)            # held for the continuations to come; this page uses SQL or the new sort
        if not keep:
            with _frozen_lock:
                if _frozen.get(key) in (fz, cur):
                    _frozen.pop(key, None)
    plan = _page_plan(page_sql, page_params)
    if plan == "index":
        return None
    if shallow:
        if warm and plan == "sort":
            sort_later(q)
        return None
    total = count() if (q.prefix and count is not None) else None
    if total is not None and total < ORDER_MIN_BOXES:
        return None                      # a short audit-sorted list: SQL scoped to its images answers any page
    return _build_frozen(q, total)


def _caught_up(q: _Q, fz: _Frozen, version: tuple) -> Optional[_Frozen]:
    """`fz` brought up to the current label index — patched when announced writes are all that moved it — or
    None when it cannot be: boxes were re-read without anyone saying which."""
    with _frozen_lock:
        gen, mark, top = _label_gen.get(q.ds_id, 0), _label_mark.get(q.ds_id), _top_mark[0]
    now, then = version[0], fz.version[0]
    if fz.gen == gen and now == then:
        return fz
    # The roll-up is where the dataset's latest announced write left it, and the highest box id is where it or
    # theirs (another dataset's) left it: nothing else re-read boxes since.
    if not ((now[0] == then[0]) if fz.gen == gen else (mark == (gen, now[0]))) or now[1] not in (then[1], top):
        return None
    with _key_lock(_frozen_patch, q.key):
        with _frozen_lock:
            held = _frozen.get(q.key)
        if held is not None and held is not fz and held.gen == gen and held.version[0] == now:
            return held                  # another page patched it meanwhile
        ids = _written_since(q.ds_id, fz.gen)
        if ids is None or len(ids) > PATCH_MAX_IMAGES:
            return None
        try:
            new = (_patched(fz, q, ids, (now, fz.version[1]), gen) if ids else
                   _Frozen((now, fz.version[1]), fz.cols, fz.imgs, gen, fz.text, fz.at))
        except Exception:  # noqa: BLE001 — sort it again instead
            log.debug("patching a box order failed", exc_info=True)
            return None
        if new is not None:
            with _frozen_lock:
                if _frozen.get(q.key) is fz:
                    _frozen[q.key] = new
        return new


def _written_since(ds_id: str, gen: int) -> Optional[set]:
    """Images of every announced write after `gen`; None when the log no longer reaches back that far."""
    with _frozen_lock:
        if gen < _label_floor[ds_id]:
            return None
        out: set = set()
        for g, ids in _label_log[ds_id]:
            if g > gen:
                out.update(ids)
        return out


def _patched(fz: _Frozen, q: _Q, ids: set, version: tuple, gen: int) -> Optional[_Frozen]:
    """A copy of `fz` with the entries of these images replaced by their current keys (only boxes the list's
    filters still match). None when a new key does not fit a column (it would not compare like SQLite)."""
    ids_json = json.dumps(sorted(ids))
    pre, pp = _images_prefix(q.ds_id, q.prefix, ids_json)
    cur = db.connect().execute(
        f"{pre} SELECT {', '.join(q.exprs)}, b.image_id {q.body} AND b.image_id IN (SELECT value FROM json_each(?)) "
        f"ORDER BY {', '.join(e + ' ASC' for e in q.exprs)}", [*pp, *q.params, ids_json])
    cur.row_factory = None
    rows = cur.fetchall()
    if not all(_fits(v, c, t) for r in rows for v, c, t in zip(r, fz.cols, fz.text)):
        return None
    drop = _positions(fz.imgs, ids)
    cols = [_spliced(c, drop, ()) for c in fz.cols]
    imgs = _spliced(fz.imgs, drop, ())
    at = [_insertion(cols, len(imgs), r[:-1]) for r in rows]     # rows ascend, so these do too
    cols = [_spliced(c, (), [(p, r[k]) for p, r in zip(at, rows)]) for k, c in enumerate(cols)]
    imgs = _spliced(imgs, (), [(p, r[-1]) for p, r in zip(at, rows)])
    return _Frozen(version, cols, imgs, gen, fz.text, fz.at)


def _fits(v: Any, col: Any, text: bool) -> bool:
    if text:
        return isinstance(v, str)
    if isinstance(col, array):
        return type(v) is (int if col.typecode == "q" else float)
    return type(v) in (int, float)


def _positions(imgs: array, ids: set) -> List[int]:
    """Ascending positions of the entries of these images: a byte search per image (a few hundred µs for 300k
    entries), or one pass for many images."""
    if len(ids) > 32:
        return [j for j, v in enumerate(imgs) if v in ids]
    buf, w, out = imgs.tobytes(), imgs.itemsize, []
    for iid in ids:
        pat = array(imgs.typecode, [iid]).tobytes()
        j = buf.find(pat)
        while j >= 0:
            if j % w:                    # straddles two entries
                j = buf.find(pat, j + 1)
                continue
            out.append(j // w)
            j = buf.find(pat, j + w)
    out.sort()
    return out


def _spliced(col: Any, drop: Sequence[int], ins: Sequence[Tuple[int, Any]]) -> Any:
    """`col` without the entries at the ascending positions `drop`, or with `ins` = [(position, value)] added
    (ascending; a value goes before the entry now at its position). Built from slices: a patch costs a copy of
    each column, not a Python loop over it."""
    out = col[:0]
    last = 0
    for j in drop:
        out += col[last:j]
        last = j + 1
    for p, v in ins:
        out += col[last:p]
        out.append(v)
        last = p
    out += col[last:]
    return out


def _insertion(cols: list, n: int, key: Sequence[Any]) -> int:
    """Where `key` goes among ascending columns (keys are unique: the box id is the last one)."""
    key = tuple(key)
    lo, hi = 0, n
    while lo < hi:
        mid = (lo + hi) // 2
        if tuple(c[mid] for c in cols) < key:
            lo = mid + 1
        else:
            hi = mid
    return lo


def _sort_later(q: _Q) -> None:
    """Sort this list on the background thread (once at a time per list)."""
    with _frozen_lock:
        if q.key in _warming:
            return
        _warming.add(q.key)
    _warmer.submit(_warm, q)


def _warm(q: _Q) -> None:
    try:
        _build_frozen(q)
    except Exception:  # noqa: BLE001 — only a head start: the next page builds it again or pages by SQL
        log.debug("box order warm-up failed", exc_info=True)
    finally:
        with _frozen_lock:
            _warming.discard(q.key)


def _build_frozen(q: _Q, total: Optional[int] = None) -> Optional[_Frozen]:
    key = q.key
    with _key_lock(_frozen_build, key):
        with _frozen_lock:
            fz = _frozen.get(key)
            gen = _label_gen.get(q.ds_id, 0)          # before reading: a write during the read is patched in later
        version = _order_version(q.ds_id, q.prefix, q.body)
        if fz is not None and fz.version == version and fz.gen == gen and time.monotonic() - fz.at < ORDER_TTL:
            return fz
        pre, pp = q.prefix, list(q.pparams)
        if q.prefix:
            if total is None:
                total = db.query_one(f"SELECT COUNT(*) AS c {q.body.replace(_AUDIT_JOIN, '')}", list(q.params))["c"]
            pre, pp = _scoped_prefix(q.ds_id, q.body, list(q.params), total) or (pre, pp)
        cur = db.connect().execute(f"{pre} SELECT {', '.join(q.exprs)}, b.image_id {_steered(q.body)} ORDER BY "
                                   + ", ".join(f"{e} ASC" for e in q.exprs), pp + list(q.params))
        cur.row_factory = None
        cols = _columns(cur.fetchall(), len(q.exprs) + 1)
        if cols is None or not isinstance(cols[-1], array):
            return None
        imgs = cols.pop()
        fz = _Frozen(version, cols, imgs, gen)
        if fz.n >= ORDER_MIN_BOXES:
            with _frozen_lock:
                _frozen[key] = fz
                _frozen.move_to_end(key)
                held = sum(f.n for f in _frozen.values())
                while len(_frozen) > 1 and (len(_frozen) > ORDER_MAX or held > ORDER_BUDGET):
                    _k, old = _frozen.popitem(last=False)
                    held -= old.n
        return fz


def _forget_frozen(fz: _Frozen) -> None:
    with _frozen_lock:
        for k in [k for k, v in _frozen.items() if v is fz]:
            _frozen.pop(k, None)


def list_boxes(ds_id: str, img_f: Dict[str, Any], classes: List[int], sort: str = "class", order: Optional[str] = None,
               offset: int = 0, limit: int = 200, after: Optional[list] = None, seed: int = 0,
               counts: bool = False) -> dict:
    if sort not in SORTS:
        raise HTTPException(400, f"Unknown sort: {sort}")
    order = (order or DEFAULT_ORDER.get(sort, "asc")).lower()
    desc = order == "desc"
    prefix, pparams, body, params, exprs = _box_query(ds_id, img_f, classes, sort, seed)
    q = _Q(ds_id, prefix, tuple(pparams), body, tuple(params), tuple(exprs))

    keyset = ""
    kparams: List[Any] = []
    if after is not None:
        keyset = f" AND ({', '.join(exprs)}) {'<' if desc else '>'} ({', '.join('?' * len(exprs))})"
        kparams = list(after)
    direction = "DESC" if desc else "ASC"
    order_sql = ", ".join(f"{e} {direction}" for e in exprs)
    keys_sql = ", ".join(f"{e} AS k{n}" for n, e in enumerate(exprs))

    read_body = _steered(body)

    def page_sql(pre: str) -> str:
        # Sort only (id, key) pairs: dragging every column of 40k rows through the sorter is 4x slower.
        return f"{pre} SELECT b.id AS bid, {keys_sql} {read_body}{keyset} ORDER BY {order_sql} LIMIT ? OFFSET ?"
    tail = params + kparams + [limit, offset]

    # Counting needs no sort key: neither the audit CTE nor the images join an ordering by name brings along.
    _p0, _pp0, count_body, count_params, _e = _box_query(ds_id, img_f, classes, "area", seed)
    counted: List[int] = []

    def count() -> int:
        if not counted:
            counted.append(db.query_one(f"SELECT COUNT(*) AS c {count_body}", count_params)["c"])
        return counted[0]

    items: Optional[List[dict]] = None
    total = 0
    later: List[_Q] = []
    for _attempt in range(2):
        fz = _frozen_order(q, page_sql(prefix), pparams + tail,
                           shallow=after is None and offset + limit <= SHALLOW_ROWS, warm=counts,
                           cont=after is not None, count=count, later=later)
        if fz is None or (after is not None and not fz.accepts(after)):
            break
        start = fz.position(after, desc) + offset
        rows = range(start, min(fz.n, start + limit))
        page = [{"bid": fz.cols[-1][j], **{f"k{n}": c[j] for n, c in enumerate(fz.cols)}}
                for j in ((fz.n - 1 - r) if desc else r for r in rows)]
        items, total = _hydrate(page, len(exprs)), fz.n
        if len(items) == len(page):
            break
        _forget_frozen(fz)            # boxes it holds are gone: a write it was not told about; sort again
        items = None
    if items is None:
        total = count()
        pre, pp = (_scoped_prefix(ds_id, body, params, total) if prefix else None) or (prefix, pparams)
        items = _hydrate(db.query(page_sql(pre), pp + tail), len(exprs))
    out = {"total": total, "items": items, "sort": sort, "order": order, "offset": offset}
    if counts:
        _p, _pp, cbody, cparams, _e = _box_query(ds_id, img_f, [], "area", seed)
        out["class_counts"] = {str(r["cls"]): r["n"] for r in
                               db.query(f"SELECT b.cls AS cls, COUNT(*) AS n {cbody} GROUP BY b.cls", cparams)}
    if sort == "audit":
        out["audit_available"] = bool(db.query_one(
            "SELECT EXISTS(SELECT 1 FROM ai_preds WHERE dataset_id=? AND state='pending' AND kind IN (%s)) "
            "OR EXISTS(SELECT 1 FROM ai_audit WHERE dataset_id=?) AS x" % ",".join("?" * len(SUSPECT_KINDS)),
            [ds_id, *SUSPECT_KINDS, ds_id])["x"])
    if not (counted and counted[0] < ORDER_MIN_BOXES):      # a short list would not be held anyway
        for lq in later:
            _sort_later(lq)
    return out


def _hydrate(page: List[dict], n_keys: int) -> List[dict]:
    """Full rows for one page of box ids, in page order: geometry, image, row, label-file version."""
    if not page:
        return []
    bids = [r["bid"] for r in page]
    rows: Dict[int, dict] = {}
    for chunk in _chunks(bids):
        for r in db.query(
                "SELECT b.id, b.image_id, b.cls, b.xc, b.yc, b.w, b.h, i.rel_path, i.split, i.width, i.height, "
                "i.mtime, i.label_path FROM boxes b JOIN images i ON i.id = b.image_id "
                "WHERE b.id IN (SELECT value FROM json_each(?))", [json.dumps(chunk)]):
            rows[r["id"]] = r
    image_ids = sorted({r["image_id"] for r in rows.values()})
    # row = position among the image's boxes in index order (= label-file order)
    row_of: Dict[int, int] = {}
    n_of: Dict[int, int] = defaultdict(int)
    for chunk in _chunks(image_ids):
        for r in db.query("SELECT id, image_id FROM boxes WHERE image_id IN (SELECT value FROM json_each(?)) "
                          "ORDER BY image_id, id", [json.dumps(chunk)]):
            row_of[r["id"]] = n_of[r["image_id"]]
            n_of[r["image_id"]] += 1
    # The version is the label file's content hash, exactly what Annotate's editor uses; reading the file
    # also yields the confidence column and tells us when the index no longer matches the file.
    label_of = {r["image_id"]: r["label_path"] for r in rows.values()}
    files: Dict[int, Tuple[Optional[str], list, bool]] = {}
    for iid in image_ids:
        lp = label_of.get(iid)
        version, confs, n_file = None, [], 0
        if lp:
            fb, confs, _issues, version = read_label_snapshot(Path(lp))
            n_file = len(fb)
        # The index no longer describes the file (edited outside Fovea since the last scan): rows may not line up.
        files[iid] = (version, confs, n_file != n_of[iid])
    suspects = _suspects(image_ids, rows, row_of)
    items = []
    for p in page:
        r = rows.get(p["bid"])
        if not r:
            continue                       # deleted between the two queries
        row = row_of.get(r["id"], 0)
        version, confs, stale = files[r["image_id"]]
        it = {
            "image_id": r["image_id"], "row": row, "cls": r["cls"],
            "xc": r["xc"], "yc": r["yc"], "w": r["w"], "h": r["h"],
            "rel_path": r["rel_path"], "split": r["split"], "width": r["width"], "height": r["height"],
            "mtime": r["mtime"], "version": version,
            "k": [p[f"k{n}"] for n in range(n_keys)],
        }
        if not stale and row < len(confs) and confs[row] is not None:
            it["conf"] = confs[row]
        if stale:
            it["stale"] = True
        sus = suspects.get((r["image_id"], row))
        if sus:
            it["suspects"] = sus
        items.append(it)
    return items


def _suspects(image_ids: List[int], rows: Dict[int, dict], row_of: Dict[int, int]) -> Dict[Tuple[int, int], List[dict]]:
    """Pending audit findings per (image_id, row) for the page's images (the matching rules of _audit_cte)."""
    page_geom: Dict[int, List[Tuple[int, Tuple[float, float, float, float]]]] = defaultdict(list)
    for r in rows.values():
        page_geom[r["image_id"]].append((row_of.get(r["id"], 0), (r["xc"], r["yc"], r["w"], r["h"])))
    with_ref = has_ref_box()
    out: Dict[Tuple[int, int], List[dict]] = defaultdict(list)
    for chunk in _chunks(image_ids):
        for r in db.query(
                "SELECT id, image_id, ref_row, kind, cls, conf, reason, xc, yc, w, h%s FROM ai_preds "
                "WHERE image_id IN (SELECT value FROM json_each(?)) AND state = 'pending' "
                "AND kind IN (%s) ORDER BY COALESCE(conf, 0.5) DESC"
                % (", ref_box" if with_ref else "", ",".join("?" * len(SUSPECT_KINDS))),
                [json.dumps(chunk), *SUSPECT_KINDS]):
            gt = ref_geom(r.get("ref_box")) if r["kind"] == "loose" else (r["xc"], r["yc"], r["w"], r["h"])
            if gt is None:
                if r["ref_row"] is None:
                    continue
                row = r["ref_row"]
            else:
                hits = [n for n, g in page_geom.get(r["image_id"], ())
                        if all(abs(g[k] - gt[k]) < 1e-6 for k in range(4))]
                if not hits:
                    continue       # the box it names is not on this page (or no longer exists)
                row = r["ref_row"] if r["ref_row"] in hits else hits[0]    # identical duplicates: trust the row
            out[(r["image_id"], row)].append({
                "id": r["id"], "kind": r["kind"], "cls": r["cls"], "conf": r["conf"], "reason": r["reason"],
                "box": [r["xc"], r["yc"], r["w"], r["h"]]})
    return out


@router.get("/api/datasets/{ds_id}/boxes")
def get_boxes(ds_id: str, request: Request, cls: Optional[str] = Query(None), sort: str = Query("class"),
              order: Optional[str] = Query(None), offset: int = Query(0, ge=0), limit: int = Query(200, ge=1, le=MAX_LIMIT),
              after: Optional[str] = Query(None), seed: int = Query(0, ge=0, le=1_000_000), counts: int = Query(0)):
    """Boxes of a dataset as one list. `cls` filters BOX classes (not "images containing"); every image
    filter (split, q, review, issue, seq, ids, set, ai…) narrows by the box's image. `after` (the `k` of the
    last item you hold) continues a list exactly where it was, even after edits moved or removed items;
    `offset` is then counted from there."""
    get_dataset_or_404(ds_id)
    img_f = filters_from(request, skip=("cls",))   # [backend-data hook] cls = BOX classes here, parsed below
    img_f.pop("cls", None)
    sort = sort if sort in SORTS else "class"
    order = order.lower() if isinstance(order, str) and order.lower() in ("asc", "desc") else None
    n_keys = len(SORTS[sort][0]) + 1
    return list_boxes(ds_id, img_f, _csv_ints(cls), sort, order, offset, limit, _parse_after(after, n_keys), seed,
                      bool(counts))


# ------------------------------------------------------------------ crops

CROP_DIR = DATA_DIR / "crops"
CROP_SIZES = (64, 96, 128, 160, 192, 256, 320, 384, 512, 768)
CROP_RENDER = 1                  # bump when the drawing changes: cache files and immutable URLs are keyed on it
CROP_FILL = (28, 29, 32)         # area beyond the frame edge; the dark image surround
MIN_CONTEXT_PX = 24              # a crop never shows fewer source pixels than this, however tiny the box
CROP_CACHE_BYTES = int(os.environ.get("FOVEA_CROP_CACHE_MB", "2048")) * 1024 * 1024
DECODED_BUDGET_PX = 48_000_000   # decoded frames kept for neighbouring crops of the same image (~140 MB RGB)


def snap_crop_size(s: int) -> int:
    for c in CROP_SIZES:
        if s <= c:
            return c
    return CROP_SIZES[-1]


def crop_region(width: float, height: float, box: Sequence[float], pad: float,
                min_px: float = MIN_CONTEXT_PX) -> Tuple[float, float, float]:
    """(x0, y0, side) in image pixels: a square centred on the box, `pad` of the box's longer side added on
    every edge. Square crops keep the object at the centre of every tile, with the same framing for all.
    The UI places its box outline with this same formula."""
    xc, yc, w, h = (float(v) for v in box)
    side = max(abs(w) * width, abs(h) * height) * (1.0 + 2.0 * pad)
    side = max(side, float(min_px), 1.0)
    return xc * width - side / 2.0, yc * height - side / 2.0, side


def _probe(abs_path: str) -> Tuple[int, int, int, str]:
    """(display width, display height, orientation, format) from the header only."""
    with Image.open(abs_path) as im:
        w, h = im.size
        try:
            # (qa-engine hook) never decode a PNG beyond the cap just to look for a trailing eXIf chunk
            orient = 1 if png_over_cap(im) else int(im.getexif().get(0x0112) or 1)
        except Exception:  # noqa: BLE001 — malformed EXIF is "no orientation"
            orient = 1
        if orient in (5, 6, 7, 8):
            w, h = h, w
        return w, h, orient, im.format or ""


_dec_lock = threading.Lock()
_decoded: "OrderedDict[tuple, Image.Image]" = OrderedDict()
_dec_px = 0
_dec_key_locks: Dict[tuple, threading.Lock] = {}


def _decode(abs_path: str, reduce: int) -> Image.Image:
    with Image.open(abs_path) as im:
        if reduce > 1 and im.format == "JPEG":
            # JPEG decodes at 1/2, 1/4, 1/8 for a fraction of the cost; asked-for size is a lower bound.
            im.draft("RGB", (max(1, -(-im.width // reduce)), max(1, -(-im.height // reduce))))
        if im.width * im.height > MAX_IMAGE_PIXELS:     # (qa-engine hook) a few-MB PNG can declare 60k x 60k
            raise ImageTooLarge(f"{im.width}x{im.height} exceeds the {MAX_IMAGE_PIXELS / 1e6:.0f} MP decode limit")
        out = ImageOps.exif_transpose(im)   # labels are relative to the upright picture, as browsers show it
        out = to_display_rgb(out)
        out.load()
        return out


def _decoded_image(abs_path: str, mtime: float, reduce: int) -> Image.Image:
    """Decoded, upright RGB frame; a sorted list shows several boxes of one image side by side, and each
    tile is its own request, so the decode is shared."""
    global _dec_px
    key = (abs_path, mtime, reduce)
    with _dec_lock:
        im = _decoded.get(key)
        if im is not None:
            _decoded.move_to_end(key)
            return im
        klock = _dec_key_locks.setdefault(key, threading.Lock())
    with klock:
        with _dec_lock:
            im = _decoded.get(key)
            if im is not None:
                return im
        try:
            im = _decode(abs_path, reduce)
        finally:
            with _dec_lock:
                _dec_key_locks.pop(key, None)
        with _dec_lock:
            _decoded[key] = im
            _dec_px += im.width * im.height
            while _dec_px > DECODED_BUDGET_PX and len(_decoded) > 1:
                _k, old = _decoded.popitem(last=False)
                _dec_px -= old.width * old.height
        return im


def render_crop(abs_path: str, mtime: float, box: Sequence[float], pad: float, size: int) -> Image.Image:
    full_w, full_h, _orient, fmt = _probe(abs_path)
    x0, y0, side = crop_region(full_w, full_h, box, pad)
    need = size / side                               # output pixels per source pixel
    reduce = 1
    if fmt == "JPEG":
        for r in (8, 4, 2):
            if need * r <= 1.0:
                reduce = r
                break
    src = _decoded_image(abs_path, mtime, reduce)
    sx, sy = src.width / full_w, src.height / full_h
    rx0, ry0, lx, ly = x0 * sx, y0 * sy, side * sx, side * sy
    kx, ky = size / lx, size / ly
    canvas = Image.new("RGB", (size, size), CROP_FILL)
    ix0, iy0 = max(rx0, 0.0), max(ry0, 0.0)
    ix1, iy1 = min(rx0 + lx, float(src.width)), min(ry0 + ly, float(src.height))
    if ix1 > ix0 and iy1 > iy0:
        ox0, oy0 = round((ix0 - rx0) * kx), round((iy0 - ry0) * ky)
        ox1, oy1 = round((ix1 - rx0) * kx), round((iy1 - ry0) * ky)
        if ox1 > ox0 and oy1 > oy0:
            # Map the rounded output rectangle back to exact (fractional) source coordinates, so the box sits
            # where crop_region says to within half an output pixel, at any zoom.
            sb = (max(0.0, rx0 + ox0 / kx), max(0.0, ry0 + oy0 / ky),
                  min(float(src.width), rx0 + ox1 / kx), min(float(src.height), ry0 + oy1 / ky))
            upscale = (ox1 - ox0) > (sb[2] - sb[0])
            part = src.resize((ox1 - ox0, oy1 - oy0),
                              Image.Resampling.BICUBIC if upscale else Image.Resampling.LANCZOS,
                              box=sb, reducing_gap=None if upscale else 2.0)
            canvas.paste(part, (ox0, oy0))
    return canvas


_render_locks: Dict[str, threading.Lock] = {}
_render_guard = threading.Lock()
_renders = 0


def _crop_file(abs_path: str, st: os.stat_result, box: Sequence[float], pad: float, size: int) -> Optional[Path]:
    global _renders
    base = (f"{abs_path}|{st.st_mtime:.3f}|{st.st_size}|{box[0]:.6f},{box[1]:.6f},{box[2]:.6f},{box[3]:.6f}"
            f"|{pad:.2f}|{size}|r{CROP_RENDER}")
    key = hashlib.sha1(base.encode("utf-8", "surrogateescape")).hexdigest()
    out = CROP_DIR / key[:2] / f"{key}.jpg"
    if out.is_file():
        return out
    with _render_guard:
        lock = _render_locks.setdefault(key, threading.Lock())
    with lock:
        if out.is_file():
            return out
        tmp = out.with_suffix(f".{threading.get_ident()}.tmp")
        try:
            out.parent.mkdir(parents=True, exist_ok=True)
            im = render_crop(abs_path, st.st_mtime, box, pad, size)
            im.save(tmp, "JPEG", quality=86)
            os.replace(tmp, out)
        except Exception:  # noqa: BLE001 — undecodable file: the caller answers 415
            try:
                tmp.unlink()
            except OSError:
                pass
            return None
        finally:
            with _render_guard:
                _render_locks.pop(key, None)
    with _render_guard:
        _renders += 1
        due = _renders % 4000 == 0
    if due:
        threading.Thread(target=prune_crop_cache, daemon=True).start()
    return out


def _parse_box(box: str) -> Tuple[float, float, float, float]:
    try:
        vals = [float(v) for v in box.split(",")]
    except ValueError:
        raise HTTPException(400, "box must be xc,yc,w,h (normalised)")
    if len(vals) != 4 or not all(math.isfinite(v) for v in vals) or any(abs(v) > 100 for v in vals):
        raise HTTPException(400, "box must be four finite numbers: xc,yc,w,h (normalised)")
    return vals[0], vals[1], vals[2], vals[3]


@router.get("/api/crop/{image_id}")
def crop(image_id: int, request: Request, box: str = Query(...), pad: float = Query(0.2, ge=0, le=3),
         s: int = Query(192, ge=16, le=2048), v: Optional[str] = Query(None)):
    """JPEG crop of one box with context. Pass `v` (the image mtime) to let the browser cache it for good —
    the URL then changes whenever the image or the box does."""
    b = _parse_box(box)
    row = db.query_one("SELECT abs_path FROM images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(404, "Image not found")
    try:
        st = os.stat(row["abs_path"])
    except OSError:
        raise HTTPException(404, "Image file not found")
    out = _crop_file(row["abs_path"], st, b, round(pad, 2), snap_crop_size(s))
    if out is None:
        raise HTTPException(415, "Could not decode this image")
    etag = f'"{out.stem}"'
    headers = {"ETag": etag, "Cache-Control": "private, max-age=604800, immutable" if v else "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(out, media_type="image/jpeg", headers=headers)


def crop_cache_stats() -> dict:
    n = total = 0
    for root, _dirs, files in os.walk(CROP_DIR):
        for f in files:
            try:
                total += os.stat(os.path.join(root, f)).st_size
                n += 1
            except OSError:
                pass
    return {"files": n, "bytes": total, "limit_bytes": CROP_CACHE_BYTES}


def prune_crop_cache(limit: Optional[int] = None) -> int:
    """Keep the crop cache under its size limit, oldest files first. Crops are cheap to redraw, and edited
    boxes leave their old crops behind, so the cache must not grow without bound."""
    limit = CROP_CACHE_BYTES if limit is None else limit
    entries = []
    total = 0
    for root, _dirs, files in os.walk(CROP_DIR):
        for f in files:
            p = os.path.join(root, f)
            try:
                st = os.stat(p)
            except OSError:
                continue
            entries.append((st.st_mtime, st.st_size, p))
            total += st.st_size
    if total <= limit:
        return 0
    removed = 0
    target = int(limit * 0.8)
    for _m, size, p in sorted(entries):
        if total <= target:
            break
        try:
            os.unlink(p)
            total -= size
            removed += 1
        except OSError:
            pass
    return removed


@router.get("/api/crops/cache")
def get_crop_cache():
    return crop_cache_stats()


@router.post("/api/crops/cache/clear")
def clear_crop_cache():
    return {"removed": prune_crop_cache(limit=0)}


# ------------------------------------------------------------------ batch edits

MAX_EDITS = 50_000
ASYNC_IMAGES = 400               # more label files than this: run as a background job
_GEOM_TOL = 1e-7


class _Conflict(Exception):
    pass


def _norm_edit(e: Any) -> dict:
    if not isinstance(e, dict):
        raise HTTPException(400, "each edit must be an object")
    try:
        iid, row = int(e["image_id"]), int(e["row"])
    except (KeyError, TypeError, ValueError):
        raise HTTPException(400, "each edit needs integer image_id and row")
    if row < 0:
        raise HTTPException(400, "row must be >= 0")
    delete = e.get("delete") is True
    new_cls = None
    if not delete:
        try:
            c = float(e.get("cls"))
        except (TypeError, ValueError):
            raise HTTPException(400, "each edit needs cls (an integer class id) or delete: true")
        if not math.isfinite(c) or c != int(c) or not 0 <= c <= MAX_CLASS_ID:
            raise HTTPException(400, f"invalid class id: {e.get('cls')!r}")
        new_cls = int(c)
    expect = None
    if e.get("box") is not None and e.get("cls_was") is not None:
        try:
            bx = [float(x) for x in e["box"]]
            expect = (int(e["cls_was"]), bx)
        except (TypeError, ValueError):
            raise HTTPException(400, "box must be [xc, yc, w, h] and cls_was an integer")
        if len(bx) != 4:
            raise HTTPException(400, "box must be [xc, yc, w, h]")
    version = e.get("version")
    if version is not None and not isinstance(version, str):
        raise HTTPException(400, "version must be a string")
    if version is None and expect is None:
        raise HTTPException(400, "each edit needs the label version it was made against (or box + cls_was)")
    return {"image_id": iid, "row": row, "delete": delete, "cls": new_cls, "expect": expect, "version": version}


def _digest(data: bytes) -> Optional[str]:
    fn = getattr(labels_mod, "_digest", None)
    return fn(data) if fn else None


def _line_box(line: str) -> Optional[list]:
    """The box this label line contributes to the index, decided by the index's own parser."""
    if not line.strip():
        return None
    parsed = parse_label_text(line)[0]
    return list(parsed[0]) if len(parsed) == 1 else None


def _same(box: Sequence[float], expect: Tuple[int, List[float]]) -> bool:
    return int(box[0]) == expect[0] and all(abs(float(box[k + 1]) - expect[1][k]) <= _GEOM_TOL for k in range(4))


def _write_atomic(path: Path, data: bytes) -> None:
    """Atomic replace that keeps the file's permission bits (same contract as labels.write_label_file)."""
    # backend-data hook: a symlinked label is written through its link when it stays inside the dataset, and a
    # link out of the dataset is refused (reported per image as a conflict) — never replaced by a regular file.
    try:
        path = labels_mod.write_target(path)
    except labels_mod.LabelLinkOutside as e:
        raise _Conflict(e.strerror)
    try:
        mode: Optional[int] = stat.S_IMODE(os.stat(path).st_mode)
    except OSError:
        mode = None
    fd, tmp = tempfile.mkstemp(prefix=".fovea-", suffix=".txt", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


_LEAD = re.compile(r"(\s*)(\S+)(.*)", re.S)


def apply_edits_to_file(lp: Path, edits: List[dict]) -> dict:
    """Apply one image's edits to its label file under LABEL_LOCK. Raises _Conflict (file untouched).

    Each edit names a row and, normally, the version of the file it was made against. When the version still
    matches, the row is used as is. When it does not (another tab, Annotate, an external tool wrote the file)
    an edit that carries what it expects to find (cls_was + box) is re-located by that content — the user's
    intent is still unambiguous — and anything else is refused."""
    with LABEL_LOCK:
        try:
            data = lp.read_bytes()
        except OSError:
            raise _Conflict("label file is missing")
        current = _digest(data)
        if current is None:
            current = read_label_snapshot(lp)[3]
        text = data.decode("utf-8", "surrogateescape")
        lines = text.splitlines(keepends=True)
        row_line: List[int] = []
        row_box: List[list] = []
        for li, line in enumerate(lines):
            b = _line_box(line)
            if b is not None:
                row_line.append(li)
                row_box.append(b)
        actions: Dict[int, Optional[int]] = {}       # row -> new class, None = delete
        for e in edits:
            r = e["row"]
            in_range = r < len(row_box)
            fresh = e["version"] is not None and e["version"] == current
            if e["expect"] is not None:
                if not (in_range and _same(row_box[r], e["expect"])):
                    hits = [k for k, b in enumerate(row_box) if _same(b, e["expect"]) and k not in actions]
                    if len(hits) != 1:
                        raise _Conflict("the box was changed elsewhere" if not hits else "the box is ambiguous")
                    r = hits[0]
            elif not (fresh and in_range):
                raise _Conflict("the label file was changed elsewhere")
            if e["delete"]:
                actions[r] = None
            elif not (r in actions and actions[r] is None):     # a delete of the same row wins
                actions[r] = e["cls"]
        reassigned = deleted = 0
        drop = set()
        for r, c in actions.items():
            li = row_line[r]
            if c is None:
                drop.add(li)
                deleted += 1
                continue
            m = _LEAD.match(lines[li])
            if m and int(row_box[r][0]) != c:
                lines[li] = f"{m.group(1)}{c}{m.group(3)}"
                reassigned += 1
        new_text = "".join(line for li, line in enumerate(lines) if li not in drop)
        if new_text == text:
            return {"changed": False, "reassigned": 0, "deleted": 0, "version": current}
        new_data = new_text.encode("utf-8", "surrogateescape")
        _write_atomic(lp, new_data)
        version = _digest(new_data) or read_label_snapshot(lp)[3]
        return {"changed": True, "reassigned": reassigned, "deleted": deleted, "version": version}


def _snapshot(ds_id: str, label_paths: List[str], image_ids: List[int], name: str) -> Optional[int]:
    try:
        from ..core.snapshots import snapshot_before   # backend-data; absent on older trees
    except ImportError:
        return None
    return snapshot_before(ds_id, label_paths, "objects", image_ids=image_ids, name=name)


def _notify_written(ds_id: str, image_ids: List[int]) -> None:
    """Tell label-write listeners (QA recomputation, caches keyed on labels) which images changed. A failing
    listener never fails the edit: the files are already written."""
    fn = getattr(labels_mod, "notify_label_write", None)     # backend-data's hook registry
    if fn is None:
        return
    try:
        fn(ds_id, list(image_ids))
    except Exception:  # noqa: BLE001
        log.exception("label-write hooks failed for dataset %s", ds_id)


def _describe(n_reassign: int, n_delete: int) -> str:
    parts = []
    if n_reassign:
        parts.append(f"{n_reassign} reassigned")
    if n_delete:
        parts.append(f"{n_delete} deleted")
    return "Objects: " + (", ".join(parts) or "no changes")


def run_edits(ds_id: str, by_image: Dict[int, List[dict]], job: Optional[Job] = None) -> dict:
    ds = get_dataset_or_404(ds_id)
    classes = db.loads(ds["classes"])
    n_classes = len(classes) if ds.get("classes_source") != "inferred" and classes else None
    ids = list(by_image)
    imgs: Dict[int, dict] = {}
    for chunk in _chunks(ids):
        for r in db.query("SELECT id, label_path FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?))",
                          [ds_id, json.dumps(chunk)]):
            imgs[r["id"]] = r
    conflicts = [{"image_id": iid, "reason": "image not found in this dataset"} for iid in ids if iid not in imgs]
    targets = [iid for iid in ids if iid in imgs and imgs[iid]["label_path"]]
    conflicts += [{"image_id": iid, "reason": "image has no label file"} for iid in ids if iid in imgs and not imgs[iid]["label_path"]]
    n_re = sum(1 for iid in targets for e in by_image[iid] if not e["delete"])
    n_del = sum(1 for iid in targets for e in by_image[iid] if e["delete"])
    snapshot_id = None
    wlog = None
    if targets:
        # Before the first write: this is what Undo restores.
        snapshot_id = _snapshot(ds_id, [imgs[i]["label_path"] for i in targets], targets, _describe(n_re, n_del))
        # backend-data hook: record what this edit wrote, so its Undo (restore mode 'undo') reverts only files
        # still as this edit left them, and the undo point keeps only the files it changed.
        try:
            from ..core.snapshots import WriteLog
            wlog = WriteLog(snapshot_id) if snapshot_id else None
        except ImportError:
            pass
    changed: List[int] = []
    reread: List[int] = []           # re-indexed without a write (refused edits): new box ids all the same
    versions: Dict[str, Optional[str]] = {}
    reassigned = deleted = 0
    max_cls = -1
    try:
        for k, iid in enumerate(targets):
            if job is not None and k % 25 == 0:
                job.update(done=k, total=len(targets), message=f"Editing labels {k:,}/{len(targets):,}")
            lp = Path(imgs[iid]["label_path"])
            try:
                res = apply_edits_to_file(lp, by_image[iid])
            except _Conflict as c:
                conflicts.append({"image_id": iid, "reason": str(c)})
                refresh_image(iid, n_classes, update_counts=False)    # the index may simply be out of date
                reread.append(iid)
                continue
            versions[str(iid)] = res["version"]
            if res["changed"]:
                if wlog is not None:          # backend-data hook
                    wlog.wrote(imgs[iid]["label_path"], version=res["version"])
                changed.append(iid)
                reassigned += res["reassigned"]
                deleted += res["deleted"]
                max_cls = max([max_cls] + [e["cls"] for e in by_image[iid] if not e["delete"]])
                refresh_image(iid, n_classes, update_counts=False)
    finally:
        if targets:
            refresh_dataset_counts(ds_id)
        if changed:
            _notify_written(ds_id, changed)      # also when a later file failed: these were written
        if targets:
            # Frozen box orders patch in every image re-read here and take the new roll-up (which moves even for
            # a no-op edit) as announced; without this the next deep page would sort the whole list again.
            try:
                note_index_change(ds_id, changed + reread)
            except Exception:  # noqa: BLE001 — only a speed-up: an order it misses is sorted again
                log.debug("could not note the re-indexed images", exc_info=True)
        if wlog is not None:                     # backend-data hook: no file changed -> no undo point
            snapshot_id = wlog.close()
    if ds.get("classes_source") == "inferred" and len(classes) <= max_cls < len(classes) + 32:
        classes = classes + [f"class_{i}" for i in range(len(classes), max_cls + 1)]
        db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(classes), ds_id))
    if job is not None:
        job.update(done=len(targets), total=len(targets), message=_describe(reassigned, deleted))
    return {"updated": reassigned + deleted, "reassigned": reassigned, "deleted": deleted,
            "images_changed": changed, "conflicts": conflicts, "snapshot_id": snapshot_id, "versions": versions}


@router.patch("/api/datasets/{ds_id}/boxes")
def patch_boxes(ds_id: str, payload: dict = Body(...)):
    """{edits:[{image_id, row, version, cls | delete:true, cls_was?, box?}]} — snapshot first, then each label
    file under LABEL_LOCK with a version check; a file that changed elsewhere is reported in `conflicts`
    (untouched) unless its edits carry cls_was + box and still find their box unambiguously."""
    ds = get_dataset_or_404(ds_id)
    raw = payload.get("edits") if isinstance(payload, dict) else None
    if not isinstance(raw, list) or not raw:
        raise HTTPException(400, "edits must be a non-empty list")
    if len(raw) > MAX_EDITS:
        raise HTTPException(413, f"at most {MAX_EDITS:,} edits per request")
    classes = db.loads(ds["classes"])
    declared = ds.get("classes_source") != "inferred" and bool(classes)
    by_image: Dict[int, List[dict]] = defaultdict(list)
    for e in raw:
        n = _norm_edit(e)
        if declared and n["cls"] is not None and n["cls"] >= len(classes):
            raise HTTPException(400, f"class {n['cls']} is not in this dataset's class list (0-{len(classes) - 1})")
        by_image[n["image_id"]].append(n)
    if len(by_image) > ASYNC_IMAGES:
        job = jobs.submit("objects_edit", lambda j: run_edits(ds_id, dict(by_image), j), dataset_id=ds_id,
                          message=f"Editing {len(raw):,} boxes")
        return {"job": job.to_dict()}
    return run_edits(ds_id, dict(by_image))
