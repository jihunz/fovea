"""Shared image query builder (filters / sorting) used by the API and exporters."""
from __future__ import annotations

import json
import re
import secrets
import threading
import time
import unicodedata
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import db
from .labels import AI_AUDIT_KINDS, AI_PROPOSAL_KINDS, FLAG_LABELS, INFO_ISSUE_CODES

_PROPOSAL_IN = "(%s)" % ",".join(f"'{k}'" for k in AI_PROPOSAL_KINDS)
_AUDIT_IN = "(%s)" % ",".join(f"'{k}'" for k in AI_AUDIT_KINDS.values())

# Sort key expression and how ties break: True = by id in the same direction, False = by id ascending
# (natural order inside a group of equal keys), None = the key is unique by itself. This one table
# drives ORDER BY, "position of an image" and "next/previous image", so the three can never disagree.
# The plain-column expressions match expression indexes in db.POST_SCHEMA character for character.
SORT_KEYS = {
    "name": ("i.rel_path", True),
    "id": ("i.id", None),
    "boxes": ("i.n_boxes", False),
    "size": ("(COALESCE(i.width,0)*COALESCE(i.height,0))", False),
    "mtime": ("COALESCE(i.mtime,0)", False),
    "reviewed": ("COALESCE(CASE WHEN r.status IS NOT NULL THEN r.updated_at END,0)", False),
    "random": ("((i.id * 2654435761) % 4294967296)", None),   # odd multiplier: a bijection, so unique
    # Label audit: most likely wrong first (descending by default); images never audited sort last.
    "audit": ("COALESCE((SELECT a.score FROM ai_audit a WHERE a.image_id = i.id), -1)", False),
    # Least confident pending proposal first; a proposal without a calibrated confidence counts as 1 and
    # an image without proposals sorts last.
    "ai_conf": ("COALESCE((SELECT MIN(COALESCE(p.conf, 1)) FROM ai_preds p WHERE p.image_id = i.id"
                f" AND p.state = 'pending' AND p.kind IN {_PROPOSAL_IN}), 2)", False),
    # QA flags: most suspicious first — the highest image_flags.score among the flag codes the `issue` filter names
    # (any flag when it names none); images without one sort last. The expression depends on the filter: _flag_expr.
    "flag": ("COALESCE((SELECT MAX(g.score) FROM image_flags g WHERE g.image_id = i.id), -1)", False),
}
DEFAULT_DESC = {"audit", "flag"}

# "Has a problem": a label-parser issue other than the informational ones (no label file yet is progress, not
# a defect), or any QA flag. issue=any, the Overview "Issues" tile and Annotate's next-issue all use this.
_INFO_JSON = "[" + ",".join(f'"{c}"' for c in sorted(INFO_ISSUE_CODES)) + "]"
_INFO_ONLY = f"'{_INFO_JSON}'"
PROBLEM_SQL = (f"((i.issues != '[]' AND i.issues != {_INFO_ONLY}) "
               "OR EXISTS (SELECT 1 FROM image_flags g WHERE g.image_id = i.id))")


def problem_clause(dataset_id: str) -> Tuple[str, List[Any]]:
    """PROBLEM_SQL with the flag part as an uncorrelated IN over this dataset's flags (one index read per query
    instead of a probe per image) — for queries that can bind a parameter."""
    return (f"((i.issues != '[]' AND i.issues != {_INFO_ONLY}) "
            "OR +i.id IN (SELECT g.image_id FROM image_flags g WHERE g.dataset_id = ?))"), [dataset_id]


REVIEW_STATUSES = ("approved", "flagged", "excluded")


def nfc(v: Any) -> str:
    """Filter values as typed (NFC) must match names indexed from macOS (stored NFC, see the scanner)."""
    return unicodedata.normalize("NFC", str(v))


def _csv(v: Optional[str]) -> List[str]:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        v = ",".join(str(x) for x in v)
    return [x.strip() for x in str(v).split(",") if x.strip() != ""]


def _fit64(n: int) -> int:
    """Clamp to SQLite's 64-bit INTEGER (binding a larger int is an OverflowError, i.e. a 500). Exact for the
    comparisons it feeds: no row holds such a value, so cls=10**30 still matches nothing."""
    return min(max(n, -(1 << 63)), (1 << 63) - 1)


def _scalar(v: Any) -> str:
    """A filter value that must be one string (lists/dicts from a JSON payload are ignored)."""
    if v is None or isinstance(v, (list, dict, bool)):
        return ""
    return str(v).strip()


# ------------------------------------------------------------------ filter values
# Every filter fails CLOSED. Bulk writers (labels/bulk, review, proposals accept/reject, AI batches, export, sets)
# resolve their targets through these filters, so a value that is not understood must never be read as "no
# filter": api.common.filters_from answers 400 {code:'bad_filter', key, value} and build_where (also called directly
# by other modules) makes such a filter match nothing. Values are compared trimmed and lower-cased.
LABELED_VALUES = ("1", "0", "true", "false", "labeled", "unlabeled", "empty", "nobox", "file")
REVIEW_VALUES = REVIEW_STATUSES + ("none", "any")          # review= takes one or several (CSV: either matches)
AI_VALUES = ("pending", "audit", "any", "none")
HAS_LABEL_VALUES = ("1", "0", "true", "false", "yes", "no")
ENUM_KEYS = ("labeled", "review", "exclude_review", "ai", "has_label")
_INT_KEYS = ("cls", "only_cls", "ids")                       # CSV of ids >= 0
_COUNT_KEYS = ("min_boxes", "max_boxes")                     # one integer
_TEXT_KEYS = ("split", "q", "seq", "issue", "set")            # free text: a value nothing has already matches nothing


def _absent(v: Any) -> bool:
    """No filter: None, blank text, an empty list (except for ids — see filter_problem)."""
    return v is None or (isinstance(v, str) and v.strip() == "") or (isinstance(v, (list, tuple)) and not v)


def _word(v: Any) -> str:
    """One enum value, normalised: True/False as 'true'/'false', numbers as text, trimmed and lower-cased."""
    if isinstance(v, bool):
        return "true" if v else "false"
    if isinstance(v, (list, tuple)):
        return ",".join(_word(x) for x in v)
    return str(v).strip().lower()


def _id_list(v: Any) -> Optional[List[int]]:
    """CSV (or list) of non-negative integers; None when any item is not one (or there is none at all)."""
    out = []
    for x in _csv(v):
        try:
            n = int(x)                # "²".isdigit() is True but int("²") raises: never a 500
        except ValueError:
            return None
        if n < 0:
            return None
        out.append(_fit64(n))
    return out or None


def _count(v: Any) -> Optional[int]:
    if isinstance(v, bool):
        return None
    if isinstance(v, (list, tuple)) and len(v) == 1:
        v = v[0]
    try:
        return _fit64(int(str(v).strip()))
    except (TypeError, ValueError):
        return None


def _allowed(key: str) -> str:
    vocab = {"labeled": LABELED_VALUES, "review": REVIEW_VALUES, "exclude_review": REVIEW_STATUSES,
             "ai": AI_VALUES, "has_label": HAS_LABEL_VALUES}.get(key)
    if vocab:
        return ", ".join(vocab)
    if key in _INT_KEYS:
        return f"{'image' if key == 'ids' else 'class'} ids (whole numbers, 0 or more)"
    return "a whole number"


def filter_problem(key: str, v: Any) -> Optional[str]:
    """Why `v` is not a valid value of filter `key` (None: it is valid or absent)."""
    if key == "ids" and isinstance(v, (list, tuple)) and not v:
        return "ids is an empty list — no images are selected"      # a selection of nothing is not "no filter"
    if _absent(v):
        return None
    if isinstance(v, dict) or (isinstance(v, (list, tuple)) and any(isinstance(x, (list, tuple, dict)) for x in v)):
        return f"{key} must be text or a list of values"
    if key in ENUM_KEYS:
        if key in ("review", "exclude_review"):
            vocab = REVIEW_VALUES if key == "review" else REVIEW_STATUSES
            toks = _csv(_word(v))
            ok = bool(toks) and all(t in vocab for t in toks)
        else:
            ok = _word(v) in {"labeled": LABELED_VALUES, "ai": AI_VALUES, "has_label": HAS_LABEL_VALUES}[key]
    elif key in _INT_KEYS:
        ok = _id_list(v) is not None
    elif key in _COUNT_KEYS:
        ok = _count(v) is not None
    else:
        return None
    if ok:
        return None
    shown = v if isinstance(v, (str, int, float)) and not isinstance(v, bool) else json.dumps(v, default=str)
    return f"Unknown {key} filter value {str(shown)[:80]!r} — use {_allowed(key)}"


def canonical_filters(f: Dict[str, Any]) -> Dict[str, Any]:
    """Enum values in the form build_where compares (trimmed, lower-case). Assumes filter_problem found none."""
    out = dict(f)
    for key in ENUM_KEYS:
        if key in out and not _absent(out[key]):
            out[key] = _word(out[key])
    return out


def _seed(f: Optional[Dict[str, Any]]) -> Optional[int]:
    if not f:
        return None
    try:
        s = int(str(f.get("seed")).strip())
    except (TypeError, ValueError):
        return None
    return s % 2147483647 or None


def _random_expr(seed: Optional[int]) -> str:
    """A seeded shuffle that is still a bijection of the id (so keys never tie): an odd affine map mod 2^32,
    a xor-shift (x ^ x>>15, written with | and & since SQLite has no xor), then an odd multiply. Products
    stay below 2^63 for any realistic id."""
    if not seed:
        return SORT_KEYS["random"][0]
    a = (2654435761 + 2 * (seed % 1000003) * 40503) | 1
    b = (seed * 2246822519) % 4294967296
    c = ((seed * 3266489917) % 1073741824) | 1
    h1 = f"((i.id * {a} + {b}) % 4294967296)"
    h2 = f"(({h1} | ({h1} >> 15)) - ({h1} & ({h1} >> 15)))"
    return f"(({h2} * {c}) % 4294967296)"


def _unusable(f: Dict[str, Any]) -> bool:
    """A filter value build_where cannot read: unknown enum words, ids that are not ids, nested objects."""
    return any(filter_problem(k, f[k]) for k in ENUM_KEYS + _INT_KEYS + _COUNT_KEYS + _TEXT_KEYS if k in f)


def _text(v: Any) -> str:
    """A free-text filter value (a JSON list is read as CSV, like api.common.filters_from does)."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v).strip()
    return _scalar(v)


def build_where(dataset_id: str, f: Dict[str, Any]) -> Tuple[str, List[Any]]:
    """WHERE clause over `images i LEFT JOIN reviews r` for the shared filter keys (api.common.FILTER_KEYS).
    Fails closed: a filter whose value is not understood matches nothing (see filter_problem)."""
    where = ["i.dataset_id = ?"]
    params: List[Any] = [dataset_id]
    if _unusable(f):
        return "i.dataset_id = ? AND 0", params

    splits = [nfc(s) for s in _csv(f.get("split"))]
    if splits:
        where.append("i.split IN (%s)" % ",".join("?" * len(splits)))
        params.extend(splits)

    # Classes, like flags and AI rows below, are matched with an uncorrelated IN (...) read once from a covering
    # index (dataset_id, cls, image_id) rather than probed per image.
    cls = _id_list(f.get("cls"))
    if cls:
        where.append("+i.id IN (SELECT b.image_id FROM boxes b WHERE b.dataset_id = ? AND b.cls IN (%s))" % ",".join("?" * len(cls)))
        params.extend([dataset_id] + cls)
    only_cls = _id_list(f.get("only_cls"))
    if only_cls:
        where.append("i.n_boxes > 0 AND +i.id NOT IN (SELECT b.image_id FROM boxes b WHERE b.dataset_id = ? AND b.cls NOT IN (%s))"
                     % ",".join("?" * len(only_cls)))
        params.extend([dataset_id] + only_cls)

    has_label = "" if _absent(f.get("has_label")) else _word(f.get("has_label"))
    if has_label in ("1", "true", "yes"):
        where.append("i.has_label = 1")            # exports with "include images without labels" off
    elif has_label in ("0", "false", "no"):
        where.append("i.has_label = 0")

    labeled = "" if _absent(f.get("labeled")) else _word(f.get("labeled"))
    if labeled in ("1", "true", "labeled"):
        where.append("i.has_label = 1 AND i.n_boxes > 0")
    elif labeled in ("0", "false", "unlabeled"):
        where.append("i.has_label = 0")
    elif labeled == "empty":
        where.append("i.has_label = 1 AND i.n_boxes = 0")
    elif labeled == "nobox":
        where.append("i.n_boxes = 0")
    elif labeled == "file":                       # "has a label file" (what the Overview Labeled tile counts)
        where.append("i.has_label = 1")

    # review= one value or several (CSV: an image matching either is kept), e.g. "approved,flagged" or "none,flagged".
    review = set(_csv(_word(f.get("review")))) if not _absent(f.get("review")) else set()
    if review and not {"any", "none"} <= review:
        ors: List[str] = []
        statuses = [s for s in REVIEW_STATUSES if s in review]
        if "any" in review:
            ors.append("r.status IS NOT NULL")
        elif len(statuses) == 1:
            ors.append("r.status = ?")
            params.append(statuses[0])
        elif statuses:
            ors.append("r.status IN (%s)" % ",".join("?" * len(statuses)))
            params.extend(statuses)
        if "none" in review:
            ors.append("r.status IS NULL")
        where.append(ors[0] if len(ors) == 1 else "(" + " OR ".join(ors) + ")")
    exclude = _csv(_word(f.get("exclude_review"))) if not _absent(f.get("exclude_review")) else []
    if exclude:
        where.append("(r.status IS NULL OR r.status NOT IN (%s))" % ",".join("?" * len(exclude)))
        params.extend(exclude)

    issues = _csv(f.get("issue"))
    if issues:
        clause, args = _issue_clause(issues, dataset_id)
        where.append(clause)
        params.extend(args)

    # Flags and AI rows are matched with an uncorrelated IN (...) over this dataset's rows, read from a covering
    # index once per query — not a probe per image — so the cost does not depend on the planner's statistics.
    # The unary + keeps every such IN a membership test: used as an index constraint, two of them (or one plus
    # the working-set join of _from) made SQLite walk one whole list per row of the other — quadratic.
    ai = "" if _absent(f.get("ai")) else _word(f.get("ai"))
    if ai in ("pending", "audit", "any"):
        kinds = {"pending": f" AND p.kind IN {_PROPOSAL_IN}", "audit": f" AND p.kind IN {_AUDIT_IN}", "any": ""}[ai]
        where.append(f"+i.id IN (SELECT p.image_id FROM ai_preds p WHERE p.dataset_id = ? AND p.state = 'pending'{kinds})")
        params.append(dataset_id)
    elif ai == "none":
        where.append("+i.id NOT IN (SELECT p.image_id FROM ai_preds p WHERE p.dataset_id = ? AND p.state = 'pending')")
        params.append(dataset_id)

    q = nfc(_text(f.get("q")))
    if q:
        where.append("i.rel_path LIKE ? ESCAPE '\\'")
        params.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")

    seq = _text(f.get("seq"))
    if seq:
        where.append("i.seq = ?")
        params.append(nfc(seq))

    ids = _id_list(f.get("ids"))
    if ids:
        # One JSON parameter instead of one per id: a large selection must not hit SQLite's variable limit.
        where.append("+i.id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(ids))

    token = _text(f.get("set"))
    if token:
        sid = set_id(token, dataset_id)
        if sid is None:
            where.append("0")             # an unknown or expired set matches nothing, never "everything"
        else:
            where.append("+i.id IN (SELECT w.image_id FROM work_set_items w WHERE w.set_id = ?)")
            params.append(sid)

    for key, op in (("min_boxes", ">="), ("max_boxes", "<=")):
        n = None if _absent(f.get(key)) else _count(f.get(key))
        if n is None:
            continue                                  # absent (a bad value returned "matches nothing" above)
        where.append(f"i.n_boxes {op} ?")
        params.append(n)

    return " AND ".join(where), params


def _issue_clause(codes: List[str], dataset_id: str) -> Tuple[str, List[Any]]:
    """`issue=` matches label-parser issues, QA flags (image_flags) and AI audit findings (pending ai_preds
    of an audit kind). `any` means any parser issue or any QA flag."""
    if "any" in codes:
        return problem_clause(dataset_id)
    ors: List[str] = []
    params: List[Any] = []
    flags = [c for c in codes if c in FLAG_LABELS]
    kinds = [AI_AUDIT_KINDS[c] for c in codes if c in AI_AUDIT_KINDS]
    for code in codes:
        if code in FLAG_LABELS or code in AI_AUDIT_KINDS:
            continue
        ors.append("i.issues LIKE ?")
        params.append(f'%"{code}"%')
    if flags:
        ors.append("+i.id IN (SELECT g.image_id FROM image_flags g WHERE g.dataset_id = ? AND g.code IN (%s))"
                   % ",".join("?" * len(flags)))
        params.extend([dataset_id] + flags)
    if kinds:
        ors.append("+i.id IN (SELECT p.image_id FROM ai_preds p WHERE p.dataset_id = ? AND p.state = 'pending' AND p.kind IN (%s))"
                   % ",".join("?" * len(kinds)))
        params.extend([dataset_id] + kinds)
    return "(" + " OR ".join(ors) + ")", params


_SET_POS = "w.pos"


def _set_order_expr(f: Optional[Dict[str, Any]], keyed: bool = False) -> Optional[str]:
    """Sort key "set": the image's position in the working set named by the `set` filter. In list queries it is
    `w.pos` of a joined work_set_items row (see _from), so a page walks the set's primary key instead of sorting
    every member; for one image's key (`keyed`) it is a lookup that gives -1 outside the set. The set id is an
    integer from the database, so it is inlined like the seeded-random constants."""
    token = _scalar((f or {}).get("set"))
    sid = set_id(token) if token else None
    if sid is None:
        return None
    if not keyed:
        return _SET_POS
    return f"COALESCE((SELECT w.pos FROM work_set_items w WHERE w.set_id = {int(sid)} AND w.image_id = i.id), -1)"


_CODE_RE = re.compile(r"^[a-z0-9_]+$")


def _flag_expr(f: Optional[Dict[str, Any]]) -> str:
    """Sort key "flag" for this filter: the highest score among the QA flags the `issue` filter names (every flag
    when it names none), and for AI audit kinds the image's audit score — all "higher is more suspect". The flag
    codes are known constants, inlined like the seeded-random ones, so the key binds the same way in ORDER BY,
    position and neighbor. Each image costs one probe of image_flags' primary key (image_id, code)."""
    codes = _csv((f or {}).get("issue"))
    flags = sorted({c for c in codes if c in FLAG_LABELS and _CODE_RE.match(c)})
    audit = any(c in AI_AUDIT_KINDS for c in codes)
    parts = []
    if flags or not audit:
        where = (" AND g.code IN (%s)" % ",".join(f"'{c}'" for c in flags)) if flags else ""
        parts.append(f"COALESCE((SELECT MAX(g.score) FROM image_flags g WHERE g.image_id = i.id{where}), -1)")
    if audit:
        parts.append(SORT_KEYS["audit"][0])
    return parts[0] if len(parts) == 1 else f"MAX({', '.join(parts)})"


def _sort_spec(sort: Optional[str], order: Optional[str], f: Optional[Dict[str, Any]] = None, keyed: bool = False):
    if sort == "set" or (not sort and f and f.get("set")):
        # A working set keeps the order it was frozen in unless another sort is asked for.
        expr = _set_order_expr(f, keyed)
        if expr is not None:
            return expr, None, (order or "").lower() == "desc"
    key = sort if sort in SORT_KEYS else "id"
    expr, tie = SORT_KEYS[key]
    if key == "random":
        expr = _random_expr(_seed(f))
    elif key == "flag":
        expr = _flag_expr(f)
    o = (order or "").lower()
    desc = o == "desc" or (o not in ("asc", "desc") and key in DEFAULT_DESC)
    return expr, tie, desc


def order_clause(sort: Optional[str], order: Optional[str], reverse: bool = False,
                 f: Optional[Dict[str, Any]] = None) -> str:
    expr, tie, desc = _sort_spec(sort, order, f)
    main_desc = desc != reverse
    clause = f"{expr} {'DESC' if main_desc else 'ASC'}"
    if tie is None:
        return clause
    tie_desc = main_desc if tie else reverse
    return f"{clause}, i.id {'DESC' if tie_desc else 'ASC'}"


def _beyond(sort: Optional[str], order: Optional[str], after: bool,
            f: Optional[Dict[str, Any]] = None) -> Tuple[str, bool]:
    """Predicate for rows strictly before (or after) a row in this ordering. Returns (sql, needs_tie):
    bind (key,) when needs_tie is False, else (key, key, id)."""
    expr, tie, desc = _sort_spec(sort, order, f)
    lt, gt = ("<", ">") if not after else (">", "<")
    main_op = gt if desc else lt
    if tie is None:
        return f"{expr} {main_op} ?", False
    tie_op = (gt if desc else lt) if tie else lt
    return f"({expr} {main_op} ? OR ({expr} = ? AND i.id {tie_op} ?))", True


_JOIN = "FROM images i LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path"
_R_REF = re.compile(r"\br\.")


def _from(*sql: str, f: Optional[Dict[str, Any]] = None) -> str:
    """FROM clause for a query that returns no review column: the reviews join only when the WHERE / ORDER BY
    reads one (SQLite does not drop an unused LEFT JOIN by itself; it costs one probe per image and keeps count /
    facet queries from being answered from an index alone), and the working set's rows when they order it."""
    base = _JOIN if any(_R_REF.search(x or "") for x in sql) else "FROM images i"
    if any(_SET_POS in (x or "") for x in sql):
        sid = set_id(_scalar((f or {}).get("set")))
        base += f" JOIN work_set_items w ON w.image_id = i.id AND w.set_id = {int(sid) if sid is not None else -1}"
    return base


def _bind(key: Any, image_id: int, needs_tie: bool) -> List[Any]:
    return [key, key, image_id] if needs_tie else [key]


def _key_of(dataset_id: str, image_id: int, sort: Optional[str], order: Optional[str],
            f: Optional[Dict[str, Any]]) -> Optional[dict]:
    expr = _sort_spec(sort, order, f, keyed=True)[0]
    return db.query_one(f"SELECT {expr} AS k {_JOIN} WHERE i.id = ? AND i.dataset_id = ?", (image_id, dataset_id))


def position(dataset_id: str, f: Dict[str, Any], image_id: int, sort: Optional[str] = None,
             order: Optional[str] = None) -> Optional[int]:
    """0-based index of an image in the filtered, sorted list — None when it is not in that list."""
    where, params = build_where(dataset_id, f)
    expr = _sort_spec(sort, order, f, keyed=True)[0]
    cur = db.query_one(f"SELECT {expr} AS k {_JOIN} WHERE {where} AND i.id = ?", params + [image_id])
    if not cur:
        return None
    pred, needs_tie = _beyond(sort, order, after=False, f=f)
    row = db.query_one(f"SELECT COUNT(*) AS c {_from(where, pred, f=f)} WHERE {where} AND {pred}",
                       params + _bind(cur["k"], image_id, needs_tie))
    return int(row["c"]) if row else None


BASE_SELECT = """
SELECT i.id, i.rel_path, i.split, i.width, i.height, i.size, i.mtime, i.has_label, i.n_boxes,
       i.classes, i.issues, i.seq, i.abs_path, i.label_path,
       r.status AS review_status, r.note AS review_note, r.updated_at AS review_at
FROM images i
LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path
"""


def row_to_item(row: dict, with_paths: bool = False) -> dict:
    item = {
        "id": row["id"], "rel_path": row["rel_path"], "split": row["split"],
        "width": row["width"], "height": row["height"], "size": row["size"],
        "has_label": bool(row["has_label"]), "n_boxes": row["n_boxes"], "mtime": row["mtime"],
        "classes": db.loads(row["classes"]), "issues": db.loads(row["issues"]), "seq": row["seq"],
        # `review` is set only for a reviewed image; a note survives clearing the status and is always in `note`.
        "review": ({"status": row["review_status"], "note": row["review_note"] or "", "updated_at": row["review_at"]}
                   if row.get("review_status") else None),
        "note": row.get("review_note") or "",
    }
    if with_paths:
        from ..paths import to_host
        item["abs_path"] = row["abs_path"]
        item["abs_path_host"] = to_host(row["abs_path"])
        item["label_path"] = row["label_path"]
        item["label_path_host"] = to_host(row["label_path"]) if row["label_path"] else None
    return item


def attach_ai(items: List[dict]) -> None:
    """Per-row AI / QA badges, computed for the returned page only: `ai_pending` (pending proposals),
    `ai_suspect` (pending audit findings) and `flags` (QA flag codes)."""
    if not items:
        return
    by_id = {it["id"]: it for it in items}
    for it in items:
        it["ai_pending"], it["ai_suspect"], it["flags"] = 0, 0, []
    ids = json.dumps(list(by_id))
    for r in db.query(f"""SELECT image_id, SUM(kind IN {_PROPOSAL_IN}) pend, SUM(kind IN {_AUDIT_IN}) sus FROM ai_preds
                          WHERE image_id IN (SELECT value FROM json_each(?)) AND state = 'pending' GROUP BY image_id""", (ids,)):
        it = by_id.get(r["image_id"])
        if it:
            it["ai_pending"], it["ai_suspect"] = int(r["pend"] or 0), int(r["sus"] or 0)
    for r in db.query("SELECT image_id, code FROM image_flags WHERE image_id IN (SELECT value FROM json_each(?)) ORDER BY code",
                      (ids,)):
        it = by_id.get(r["image_id"])
        if it:
            it["flags"].append(r["code"])


def count_images(dataset_id: str, f: Dict[str, Any]) -> int:
    where, params = build_where(dataset_id, f)
    row = db.query_one(f"SELECT COUNT(*) AS c {_from(where)} WHERE {where}", params)
    return int(row["c"]) if row else 0


def list_images(dataset_id: str, f: Dict[str, Any], offset: int = 0, limit: int = 80,
                sort: Optional[str] = None, order: Optional[str] = None, with_boxes: bool = False,
                with_paths: bool = False, after_id: Optional[int] = None, with_total: bool = True,
                with_ai: bool = True) -> Tuple[Optional[int], List[dict]]:
    """One page of the filtered, sorted list. `after_id` pages by key (rows strictly after that image in
    this order) instead of OFFSET — stable while edits change the list, and O(page) at any depth.
    `with_total=False` skips the COUNT for follow-up pages (total is then None)."""
    db.maybe_refresh_planner_stats()
    where, params = build_where(dataset_id, f)
    total = count_images(dataset_id, f) if with_total else None
    if after_id is not None:
        cur = _key_of(dataset_id, after_id, sort, order, f)
        if cur is not None:
            pred, needs_tie = _beyond(sort, order, after=True, f=f)
            where = f"{where} AND {pred}"
            params = params + _bind(cur["k"], after_id, needs_tie)
    # Sort narrow ids first, then read the wide rows of that one page: the sorter never carries paths and review
    # columns for every matching image (5-8x faster for the computed sorts audit / ai_conf at 45k images).
    order_by = order_clause(sort, order, f=f)
    ids = [r[0] for r in db.connect().execute(
        f"SELECT i.id {_from(where, order_by, f=f)} WHERE {where} ORDER BY {order_by} LIMIT ? OFFSET ?", params + [limit, offset])]
    return total, items_by_ids(dataset_id, ids, with_paths=with_paths, with_boxes=with_boxes, with_ai=with_ai)


def list_ids(dataset_id: str, f: Dict[str, Any], sort: Optional[str] = None, order: Optional[str] = None,
             limit: int = 500_000) -> List[int]:
    """Every id of the filtered list in its order: a frozen working set for a review session (items that
    stop matching the filter while being reviewed are neither skipped nor repeated)."""
    where, params = build_where(dataset_id, f)
    order_by = order_clause(sort, order, f=f)
    rows = db.connect().execute(f"SELECT i.id {_from(where, order_by, f=f)} WHERE {where} ORDER BY {order_by} LIMIT ?",
                                params + [limit]).fetchall()
    return [r[0] for r in rows]


def items_by_ids(dataset_id: str, ids: List[int], with_paths: bool = False, with_boxes: bool = False,
                 with_ai: bool = True) -> List[dict]:
    """Rows for these ids in the given order (ids no longer in the dataset are left out)."""
    if not ids:
        return []
    rows = db.query(f"{BASE_SELECT} WHERE i.dataset_id = ? AND i.id IN (SELECT value FROM json_each(?))",
                    (dataset_id, json.dumps([int(x) for x in ids])))
    by_id = {r["id"]: row_to_item(r, with_paths) for r in rows}
    items = [by_id[i] for i in ids if i in by_id]
    if with_boxes and items:
        attach_boxes(items)
    if with_ai:
        attach_ai(items)
    return items


def iter_images(dataset_id: str, f: Dict[str, Any], sort: Optional[str] = None, order: Optional[str] = None,
                chunk: int = 2000):
    """Every matching image, for long jobs (export, compare). The id list is taken once up front: paging
    with OFFSET over a filter the user keeps changing (reviewing, excluding) skipped or repeated images."""
    ids = list_ids(dataset_id, f, sort, order)
    for k in range(0, len(ids), chunk):
        yield from items_by_ids(dataset_id, ids[k:k + chunk], with_paths=True, with_ai=False)


# ------------------------------------------------------------------ working sets
# An ordered id list frozen on the server: Explore's "all matching" and large selections hand a short token to
# Annotate / the AI tab / exports instead of an id list in the URL. Every image filter takes `set=<token>`.

SET_MAX = 200_000
SET_TTL = 24 * 3600.0                  # kept 24 h after its last use
SET_KEEP_PER_DATASET = 50              # older sets beyond this are dropped when a new one is made
_TOKEN_RE = re.compile(r"^[A-Za-z0-9_-]{6,64}$")
_set_use: Dict[int, float] = {}        # set id -> last use in this process (written to used_at on the next write)
_set_lock = threading.Lock()


class SetError(ValueError):
    """A working set could not be made or used. `code`: set_too_large | set_expired | invalid."""

    def __init__(self, message: str, code: str, status: int = 400):
        super().__init__(message)
        self.code, self.status = code, status


def valid_token(token: Any) -> bool:
    return isinstance(token, str) and bool(_TOKEN_RE.match(token))


def _live(row: Optional[dict]) -> bool:
    if not row:
        return False
    with _set_lock:
        used = max(float(row.get("used_at") or 0), float(row.get("created_at") or 0), _set_use.get(row["id"], 0.0))
    return time.time() - used <= SET_TTL


def set_info(token: Any, dataset_id: Optional[str] = None, touch: bool = True) -> Optional[dict]:
    """The working set behind a token: {id, token, dataset_id, count, params, created_at, expires_at}. None when
    the token is unknown, expired or (with dataset_id) belongs to another dataset."""
    if not valid_token(token):
        return None
    row = db.query_one("SELECT id, token, dataset_id, n, params, created_at, used_at FROM work_sets WHERE token=?", (token,))
    if not row or (dataset_id is not None and row["dataset_id"] != dataset_id) or not _live(row):
        return None
    now = time.time()
    if touch:
        with _set_lock:
            _set_use[row["id"]] = now
    with _set_lock:
        last = max(float(row["used_at"] or 0), float(row["created_at"] or 0), _set_use.get(row["id"], 0.0))
    return {"id": row["id"], "token": row["token"], "dataset_id": row["dataset_id"], "count": row["n"],
            "params": db.loads(row["params"], {}), "created_at": row["created_at"], "expires_at": last + SET_TTL}


def set_id(token: Any, dataset_id: Optional[str] = None) -> Optional[int]:
    info = set_info(token, dataset_id)
    return info["id"] if info else None


def _expire_sets(conn, dataset_id: str) -> None:
    """Write remembered uses, then drop expired sets and the oldest beyond SET_KEEP_PER_DATASET (one write txn)."""
    with _set_lock:
        uses = list(_set_use.items())
    if uses:
        conn.executemany("UPDATE work_sets SET used_at = MAX(COALESCE(used_at, 0), ?) WHERE id = ?", [(t, s) for s, t in uses])
    conn.execute("DELETE FROM work_sets WHERE COALESCE(used_at, created_at, 0) < ?", (time.time() - SET_TTL,))
    conn.execute("""DELETE FROM work_sets WHERE dataset_id = ? AND id IN (
                      SELECT id FROM work_sets WHERE dataset_id = ? ORDER BY COALESCE(used_at, created_at) DESC, id DESC
                      LIMIT -1 OFFSET ?)""", (dataset_id, dataset_id, SET_KEEP_PER_DATASET - 1))
    alive = {r[0] for r in conn.execute("SELECT id FROM work_sets").fetchall()}
    with _set_lock:
        for s in [s for s in _set_use if s not in alive]:
            _set_use.pop(s, None)


def _unique_ints(values: Sequence[Any]) -> List[int]:
    out, seen = [], set()
    for v in values or ():
        if isinstance(v, bool):
            continue
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i >= 0 and i not in seen:
            seen.add(i)
            out.append(i)
    return out


def create_set(dataset_id: str, ids: Optional[Sequence[Any]] = None, filters: Optional[Dict[str, Any]] = None,
               sort: Optional[str] = None, order: Optional[str] = None, source: str = "") -> dict:
    """Freeze an ordered id list and return {token, count, created_at, expires_at}.

    * ids: kept in the given order (duplicates and ids of other datasets dropped); with filters too, only the
      ids that also match them.
    * filters (+ sort / order): every matching id, in that order. Neither: the whole dataset.
    More than SET_MAX ids is refused (SetError set_too_large) rather than silently cut."""
    filters = dict(filters or {})
    if ids is not None:
        want = _unique_ints(ids)
        if len(want) > SET_MAX:
            raise SetError(f"A working set holds at most {SET_MAX:,} images ({len(want):,} given)", "set_too_large", 413)
        found: set = set()
        where, params = build_where(dataset_id, filters)
        for k in range(0, len(want), 50_000):
            chunk = json.dumps(want[k:k + 50_000])
            found.update(r[0] for r in db.connect().execute(
                f"SELECT i.id {_from(where)} WHERE {where} AND i.id IN (SELECT value FROM json_each(?))", params + [chunk]))
        keep = [i for i in want if i in found]
    else:
        keep = list_ids(dataset_id, filters, sort, order, limit=SET_MAX + 1)
        if len(keep) > SET_MAX:
            raise SetError(f"A working set holds at most {SET_MAX:,} images; narrow the filters", "set_too_large", 413)
    token = secrets.token_hex(8)
    now = time.time()
    params_json = db.dumps({"filters": filters, "sort": sort, "order": order, "from": source or ("ids" if ids is not None else "filters")})
    with db.transaction() as conn:
        _expire_sets(conn, dataset_id)
        sid = conn.execute("INSERT INTO work_sets(token, dataset_id, n, params, created_at, used_at) VALUES(?,?,?,?,?,?)",
                           (token, dataset_id, len(keep), params_json, now, now)).lastrowid
        conn.executemany("INSERT INTO work_set_items(set_id, pos, image_id) VALUES(?,?,?)",
                         ((sid, k, iid) for k, iid in enumerate(keep)))
    if len(keep) >= db.SMALL_TABLE_ROWS:
        db.maybe_refresh_planner_stats()
    return {"token": token, "count": len(keep), "created_at": now, "expires_at": now + SET_TTL}


def delete_set(token: Any, dataset_id: str) -> bool:
    if not valid_token(token):
        return False
    with db.transaction() as conn:
        gone = conn.execute("DELETE FROM work_sets WHERE token=? AND dataset_id=?", (token, dataset_id)).rowcount
    return bool(gone)


def attach_boxes(items: List[dict]) -> None:
    ids = [it["id"] for it in items]
    by_id = {it["id"]: it for it in items}
    for it in items:
        it["boxes"] = []
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        rows = db.query("SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE image_id IN (%s) ORDER BY id" % ",".join("?" * len(chunk)), chunk)
        for r in rows:
            by_id[r["image_id"]]["boxes"].append([r["cls"], r["xc"], r["yc"], r["w"], r["h"]])


NEEDS = {
    "nobox": "i.n_boxes = 0", "label": "i.has_label = 1", "issue": PROBLEM_SQL, "unreviewed": "r.status IS NULL",
    "proposals": f"EXISTS (SELECT 1 FROM ai_preds p WHERE p.image_id = i.id AND p.state = 'pending' AND p.kind IN {_PROPOSAL_IN})",
    "suspect": f"EXISTS (SELECT 1 FROM ai_preds p WHERE p.image_id = i.id AND p.state = 'pending' AND p.kind IN {_AUDIT_IN})",
    "flag": "EXISTS (SELECT 1 FROM image_flags g WHERE g.image_id = i.id)",
}


def need_clause(need: Optional[str], dataset_id: str) -> Tuple[str, List[Any]]:
    """NEEDS[need] as a parameterised clause; the table-backed ones as uncorrelated IN (...) over this dataset."""
    if need == "issue":
        return problem_clause(dataset_id)
    if need == "proposals":
        return f"+i.id IN (SELECT p.image_id FROM ai_preds p WHERE p.dataset_id = ? AND p.state = 'pending' AND p.kind IN {_PROPOSAL_IN})", [dataset_id]
    if need == "suspect":
        return f"+i.id IN (SELECT p.image_id FROM ai_preds p WHERE p.dataset_id = ? AND p.state = 'pending' AND p.kind IN {_AUDIT_IN})", [dataset_id]
    if need == "flag":
        return "+i.id IN (SELECT g.image_id FROM image_flags g WHERE g.dataset_id = ?)", [dataset_id]
    return (NEEDS[need], []) if need in NEEDS else ("", [])


def neighbor(dataset_id: str, f: Dict[str, Any], current_id: int, direction: str = "next",
             sort: Optional[str] = None, order: Optional[str] = None, need: Optional[str] = None) -> Optional[dict]:
    """The next/previous image after `current_id` in the list's own order (filters + sort), optionally
    only among images that also satisfy `need` (e.g. "nobox" for Annotate's next-unlabeled). The
    current image does not have to match the filters; its sort key is the starting point."""
    where, params = build_where(dataset_id, f)
    need_sql, need_params = need_clause(need, dataset_id)
    if need_sql:
        where, params = f"{where} AND {need_sql}", params + need_params
    cur = _key_of(dataset_id, current_id, sort, order, f)
    if not cur:
        return None
    after = direction != "prev"
    pred, needs_tie = _beyond(sort, order, after=after, f=f)
    order_by = order_clause(sort, order, reverse=not after, f=f)
    hit = db.connect().execute(f"SELECT i.id {_from(where, pred, order_by, f=f)} WHERE {where} AND {pred} ORDER BY {order_by} LIMIT 1",
                               params + _bind(cur["k"], current_id, needs_tie)).fetchone()
    items = items_by_ids(dataset_id, [hit[0]]) if hit else []
    return items[0] if items else None


# ------------------------------------------------------------------ facets
# Standard faceted-search counts: each facet is counted under every OTHER active filter, so a value's count is
# exactly how many images picking it would show. Facets whose key is not active all share one WHERE clause; an
# active facet gets its own (the filters without itself). Results are cached per data version + filters.

FACET_KEYS = ("split", "review", "labeled", "cls", "issue", "ai")
FACET_TTL = 120.0
FACET_CACHE_MAX = 64
_facet_cache: Dict[tuple, Tuple[float, dict]] = {}
_facet_lock = threading.Lock()


def _without(f: Dict[str, Any], *keys: str) -> Dict[str, Any]:
    return {k: v for k, v in f.items() if k not in keys}


def _active(f: Dict[str, Any], key: str) -> bool:
    return f.get(key) not in (None, "", [])


def _versions(dataset_id: str) -> Dict[str, tuple]:
    ds = db.query_one("SELECT updated_at, scanned_at, image_count, label_count, box_count, classes FROM datasets WHERE id=?",
                      (dataset_id,)) or {}
    rv = db.query_one("SELECT COUNT(*) n, MAX(updated_at) t FROM reviews WHERE dataset_id=?", (dataset_id,)) or {}
    ai = db.query_one("SELECT COUNT(*) n, MAX(id) m, TOTAL(state = 'pending') p FROM ai_preds WHERE dataset_id=?",
                      (dataset_id,)) or {}
    fl = db.query_one("SELECT COUNT(*) n, TOTAL(rowid) s FROM image_flags WHERE dataset_id=?", (dataset_id,)) or {}
    return {"images": tuple(ds.values()), "reviews": tuple(rv.values()), "ai": tuple(ai.values()), "flags": tuple(fl.values())}


def data_version(dataset_id: str) -> tuple:
    """Cheap fingerprint of everything an image query can depend on: the label index (every label write and scan
    bumps datasets.updated_at / counts), review marks, pending AI rows and QA flags. A few index-only aggregates."""
    v = _versions(dataset_id)
    return v["images"], v["reviews"], v["ai"], v["flags"]


# Per-image facts from the small side tables (QA flags, pending AI rows, review marks), read once per change of
# that table and counted in Python for the matching ids: probing them per image, or sending SQLite an id list
# per facet, cost more than the image pass itself at 45k images.
_side_cache: Dict[Tuple[str, str], Tuple[tuple, dict]] = {}
SIDE_CACHE_DATASETS = 4


def _side_map(dataset_id: str, kind: str, version: tuple) -> dict:
    key = (dataset_id, kind)
    with _facet_lock:
        hit = _side_cache.get(key)
    if hit and hit[0] == version:
        return hit[1]
    conn = db.connect()
    out: Dict[int, Any] = {}
    if kind == "flags":
        for iid, code in conn.execute("SELECT g.image_id, g.code FROM image_flags g WHERE g.dataset_id = ?", (dataset_id,)):
            out[iid] = out.get(iid, ()) + (code,)
    elif kind == "ai":
        for iid, k in conn.execute("SELECT q.image_id, q.kind FROM ai_preds q WHERE q.dataset_id = ? AND q.state = 'pending'",
                                   (dataset_id,)):
            out.setdefault(iid, set()).add(k)
    else:
        for iid, status in conn.execute("""SELECT i.id, r.status FROM reviews r JOIN images i ON i.dataset_id = r.dataset_id
                                           AND i.rel_path = r.rel_path WHERE r.dataset_id = ? AND r.status IS NOT NULL""",
                                        (dataset_id,)):
            out[iid] = status
    with _facet_lock:
        _side_cache[key] = (version, out)
        datasets = list(dict.fromkeys(k[0] for k in _side_cache))
        for old in datasets[:-SIDE_CACHE_DATASETS]:
            for k in [k for k in _side_cache if k[0] == old]:
                _side_cache.pop(k, None)
    return out


def invalidate_facets(dataset_id: Optional[str] = None) -> None:
    with _facet_lock:
        for cache in (_facet_cache, _side_cache):
            for k in [k for k in cache if dataset_id is None or k[0] == dataset_id]:
                cache.pop(k, None)


def facets(dataset_id: str, f: Dict[str, Any], use_cache: bool = True) -> Dict[str, Any]:
    """Counts per filter value under the other active filters, cached per (data version, filters)."""
    return facets_cached(dataset_id, f, use_cache)[0]


def facets_cached(dataset_id: str, f: Dict[str, Any], use_cache: bool = True) -> Tuple[Dict[str, Any], bool]:
    """(facets, served from cache)."""
    db.maybe_refresh_planner_stats()
    key = (dataset_id, tuple(sorted((k, str(v)) for k, v in f.items() if k != "seed")))
    versions = _versions(dataset_id)
    version = (versions["images"], versions["reviews"], versions["ai"], versions["flags"])
    now = time.time()
    if use_cache:
        with _facet_lock:
            hit = _facet_cache.get(key)
        if hit and hit[1][0] == version and now - hit[0] < FACET_TTL:
            return hit[1][1], True
    out = _compute_facets(dataset_id, f, versions)
    with _facet_lock:
        _facet_cache[key] = (now, (version, out))
        if len(_facet_cache) > FACET_CACHE_MAX:
            for k in sorted(_facet_cache, key=lambda k: _facet_cache[k][0])[:len(_facet_cache) - FACET_CACHE_MAX]:
                _facet_cache.pop(k, None)
    return out, False


class _Everything:
    """`iid in _ALL` is always true: the id set of an unfiltered pass."""

    def __contains__(self, _item: Any) -> bool:
        return True


_ALL = _Everything()


def _compute_facets(dataset_id: str, f: Dict[str, Any], versions: Optional[Dict[str, tuple]] = None) -> Dict[str, Any]:
    """One pass over the matching images per distinct WHERE (the full filter set, plus one per ACTIVE facet key)
    yields the split / labeled / class / parser-issue counts, the total and the matching ids; flags, pending AI
    rows and reviews are counted from per-image maps of their tables (_side_map) for those ids. An expensive filter
    (a text search, a class) is therefore evaluated once per distinct WHERE, never once per facet."""
    conn = db.connect()
    versions = versions or _versions(dataset_id)
    scans: Dict[str, dict] = {}

    def tag_of(key: Optional[str]) -> str:
        return key if (key and _active(f, key)) else ""

    def scan_for(key: Optional[str]) -> dict:
        tag = tag_of(key)
        if tag in scans:
            return scans[tag]
        w, p = build_where(dataset_id, _without(f, tag) if tag else f)
        res = {"total": 0, "split": {}, "issues": {}, "cls": {}, "parser_problems": 0,
               "labeled": dict.fromkeys(("labeled", "unlabeled", "empty", "nobox", "file"), 0)}
        cols = "i.split, i.has_label, i.n_boxes > 0, i.issues, i.classes"
        if w == "i.dataset_id = ?":
            # The whole dataset: grouped by SQLite from the covering index idx_images_ds_split, no id set needed.
            groups = {tuple(r[:-1]): r[-1] for r in conn.execute(
                f"SELECT {cols}, COUNT(*) FROM images i WHERE i.dataset_id = ? GROUP BY {cols}", (dataset_id,))}
            res["ids"] = _ALL
            res["problems"] = {r[0] for r in conn.execute(
                f"SELECT i.id FROM images i WHERE i.dataset_id = ? AND i.issues != '[]' AND i.issues != {_INFO_ONLY}",
                (dataset_id,))}
        else:
            # Grouped by SQLite too, with each group's ids as one string: far cheaper than a Python row per image.
            groups, ids, problems = {}, set(), set()
            for *attrs, c, id_list in conn.execute(
                    f"SELECT {cols}, COUNT(*), group_concat(i.id) {_from(w)} WHERE {w} GROUP BY {cols}", p):
                groups[tuple(attrs)] = c
                got = map(int, id_list.split(","))
                if attrs[3] != "[]" and attrs[3] != _INFO_JSON:
                    got = list(got)
                    problems.update(got)
                ids.update(got)
            res["ids"], res["problems"] = ids, problems
        lb = res["labeled"]
        for (split, has_label, has_boxes, issues, classes), c in groups.items():
            res["total"] += c
            res["split"][split] = res["split"].get(split, 0) + c
            if has_label:
                lb["file"] += c
                lb["labeled" if has_boxes else "empty"] += c
            else:
                lb["unlabeled"] += c
            if not has_boxes:
                lb["nobox"] += c
            if issues and issues != "[]":
                codes = db.loads(issues)
                for code in codes:
                    res["issues"][code] = res["issues"].get(code, 0) + c
                if any(code not in INFO_ISSUE_CODES for code in codes):
                    res["parser_problems"] += c
            if classes and classes != "[]":
                for cls in db.loads(classes):          # images.classes: the distinct class ids of the image's boxes
                    res["cls"][int(cls)] = res["cls"].get(int(cls), 0) + c
        scans[tag] = res
        return res

    from .layout import split_order_key
    out: Dict[str, Any] = {}
    split = scan_for("split")["split"]
    out["split"] = {k: split[k] for k in sorted(split, key=lambda x: split_order_key(x or ""))}
    out["labeled"] = dict(scan_for("labeled")["labeled"])
    out["cls"] = dict(sorted(scan_for("cls")["cls"].items()))

    sc = scan_for("review")
    reviews = _side_map(dataset_id, "reviews", versions["images"] + versions["reviews"])
    rv: Dict[str, int] = {}
    for iid, status in reviews.items():
        if iid in sc["ids"]:
            rv[status] = rv.get(status, 0) + 1
    rv["any"] = sum(rv.values())
    rv["none"] = sc["total"] - rv["any"]
    out["review"] = rv

    sc = scan_for("issue")
    issues: Dict[str, int] = dict(sc["issues"])
    flagged_only = 0
    for iid, codes in _side_map(dataset_id, "flags", versions["flags"]).items():
        if iid in sc["ids"]:
            for code in codes:
                issues[code] = issues.get(code, 0) + 1
            if iid not in sc["problems"]:
                flagged_only += 1                   # issue=any: a parser problem OR a QA flag
    issues["any"] = sc["parser_problems"] + flagged_only
    kind_code = {v: k for k, v in AI_AUDIT_KINDS.items()}
    ai_map = _side_map(dataset_id, "ai", versions["ai"])
    for iid, kinds in ai_map.items():
        if iid in sc["ids"]:
            for kind in kinds:
                if kind in kind_code:
                    issues[kind_code[kind]] = issues.get(kind_code[kind], 0) + 1
    out["issue"] = issues

    sc = scan_for("ai")
    pend = audit = anyp = 0
    for iid, kinds in ai_map.items():
        if iid in sc["ids"]:
            anyp += 1
            pend += any(k in AI_PROPOSAL_KINDS for k in kinds)
            audit += any(k in kind_code for k in kinds)
    out["ai"] = {"pending": pend, "audit": audit, "any": anyp, "none": sc["total"] - anyp}

    out["total"] = scan_for(None)["total"]
    return out
