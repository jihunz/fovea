"""Shared image query builder (filters / sorting) used by the API and exporters."""
from __future__ import annotations

import json
from typing import Any, Dict, List, Optional, Tuple

from .. import db

# Sort key expression and how ties break: True = by id in the same direction, False = by id ascending
# (natural order inside a group of equal keys), None = the key is unique by itself. This one table
# drives ORDER BY, "position of an image" and "next/previous image", so the three can never disagree.
SORT_KEYS = {
    "name": ("i.rel_path", True),
    "id": ("i.id", None),
    "boxes": ("i.n_boxes", False),
    "size": ("(COALESCE(i.width,0)*COALESCE(i.height,0))", False),
    "mtime": ("COALESCE(i.mtime,0)", False),
    "reviewed": ("COALESCE(r.updated_at,0)", False),
    "random": ("((i.id * 2654435761) % 4294967296)", None),   # odd multiplier: a bijection, so unique
}

REVIEW_STATUSES = ("approved", "flagged", "excluded")


def _csv(v: Optional[str]) -> List[str]:
    if v is None:
        return []
    return [x.strip() for x in str(v).split(",") if x.strip() != ""]


def build_where(dataset_id: str, f: Dict[str, Any]) -> Tuple[str, List[Any]]:
    where = ["i.dataset_id = ?"]
    params: List[Any] = [dataset_id]

    splits = _csv(f.get("split"))
    if splits:
        where.append("i.split IN (%s)" % ",".join("?" * len(splits)))
        params.extend(splits)

    cls = [int(x) for x in _csv(f.get("cls")) if x.lstrip("-").isdigit()]
    if cls:
        where.append("EXISTS (SELECT 1 FROM boxes b WHERE b.image_id = i.id AND b.cls IN (%s))" % ",".join("?" * len(cls)))
        params.extend(cls)
    only_cls = [int(x) for x in _csv(f.get("only_cls")) if x.lstrip("-").isdigit()]
    if only_cls:
        where.append("i.n_boxes > 0 AND NOT EXISTS (SELECT 1 FROM boxes b WHERE b.image_id = i.id AND b.cls NOT IN (%s))" % ",".join("?" * len(only_cls)))
        params.extend(only_cls)

    if f.get("has_label") in ("1", 1, True):
        where.append("i.has_label = 1")            # exports with "include images without labels" off

    labeled = f.get("labeled")
    if labeled in ("1", 1, True, "labeled"):
        where.append("i.has_label = 1 AND i.n_boxes > 0")
    elif labeled in ("0", 0, False, "unlabeled"):
        where.append("i.has_label = 0")
    elif labeled == "empty":
        where.append("i.has_label = 1 AND i.n_boxes = 0")
    elif labeled == "nobox":
        where.append("i.n_boxes = 0")

    review = f.get("review")
    if review in REVIEW_STATUSES:
        where.append("r.status = ?")
        params.append(review)
    elif review == "none":
        where.append("r.status IS NULL")
    elif review == "any":
        where.append("r.status IS NOT NULL")
    exclude = _csv(f.get("exclude_review"))
    if exclude:
        where.append("(r.status IS NULL OR r.status NOT IN (%s))" % ",".join("?" * len(exclude)))
        params.extend(exclude)

    issues = _csv(f.get("issue"))
    if issues:
        if "any" in issues:
            where.append("i.issues != '[]'")
        else:
            ors = []
            for code in issues:
                ors.append("i.issues LIKE ?")
                params.append(f'%"{code}"%')
            where.append("(" + " OR ".join(ors) + ")")

    q = f.get("q")
    q = str(q).strip() if isinstance(q, (str, int, float)) and not isinstance(q, bool) else ""   # ignore list/dict
    if q:
        where.append("i.rel_path LIKE ? ESCAPE '\\'")
        params.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")

    seq = f.get("seq")
    if seq not in (None, "") and not isinstance(seq, (list, dict)):
        where.append("i.seq = ?")
        params.append(str(seq))

    ids = [int(x) for x in _csv(f.get("ids")) if x.isdigit()]
    if ids:
        # One JSON parameter instead of one per id: a large selection must not hit SQLite's variable limit.
        where.append("i.id IN (SELECT value FROM json_each(?))")
        params.append(json.dumps(ids))

    for key, op in (("min_boxes", ">="), ("max_boxes", "<=")):
        v = f.get(key)
        if v in (None, ""):
            continue
        try:
            n = int(str(v).strip())          # parse first: a clause without its parameter is a 500
        except (TypeError, ValueError):
            continue
        where.append(f"i.n_boxes {op} ?")
        params.append(n)

    return " AND ".join(where), params


def _sort_spec(sort: Optional[str], order: Optional[str]):
    expr, tie = SORT_KEYS.get(sort or "id", SORT_KEYS["id"])
    return expr, tie, (order or "").lower() == "desc"


def order_clause(sort: Optional[str], order: Optional[str], reverse: bool = False) -> str:
    expr, tie, desc = _sort_spec(sort, order)
    main_desc = desc != reverse
    clause = f"{expr} {'DESC' if main_desc else 'ASC'}"
    if tie is None:
        return clause
    tie_desc = main_desc if tie else reverse
    return f"{clause}, i.id {'DESC' if tie_desc else 'ASC'}"


def _beyond(sort: Optional[str], order: Optional[str], after: bool) -> Tuple[str, bool]:
    """Predicate for rows strictly before (or after) a row in this ordering. Returns (sql, needs_tie):
    bind (key,) when needs_tie is False, else (key, key, id)."""
    expr, tie, desc = _sort_spec(sort, order)
    lt, gt = ("<", ">") if not after else (">", "<")
    main_op = gt if desc else lt
    if tie is None:
        return f"{expr} {main_op} ?", False
    tie_op = (gt if desc else lt) if tie else lt
    return f"({expr} {main_op} ? OR ({expr} = ? AND i.id {tie_op} ?))", True


_JOIN = "FROM images i LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path"


def _bind(key: Any, image_id: int, needs_tie: bool) -> List[Any]:
    return [key, key, image_id] if needs_tie else [key]


def position(dataset_id: str, f: Dict[str, Any], image_id: int, sort: Optional[str] = None,
             order: Optional[str] = None) -> Optional[int]:
    """0-based index of an image in the filtered, sorted list — None when it is not in that list."""
    where, params = build_where(dataset_id, f)
    expr = _sort_spec(sort, order)[0]
    cur = db.query_one(f"SELECT {expr} AS k {_JOIN} WHERE {where} AND i.id = ?", params + [image_id])
    if not cur:
        return None
    pred, needs_tie = _beyond(sort, order, after=False)
    row = db.query_one(f"SELECT COUNT(*) AS c {_JOIN} WHERE {where} AND {pred}", params + _bind(cur["k"], image_id, needs_tie))
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
        "has_label": bool(row["has_label"]), "n_boxes": row["n_boxes"],
        "classes": db.loads(row["classes"]), "issues": db.loads(row["issues"]), "seq": row["seq"],
        "review": ({"status": row["review_status"], "note": row["review_note"] or "", "updated_at": row["review_at"]}
                   if row.get("review_status") else None),
    }
    if with_paths:
        from ..paths import to_host
        item["abs_path"] = row["abs_path"]
        item["abs_path_host"] = to_host(row["abs_path"])
        item["label_path"] = row["label_path"]
        item["label_path_host"] = to_host(row["label_path"]) if row["label_path"] else None
    return item


def count_images(dataset_id: str, f: Dict[str, Any]) -> int:
    where, params = build_where(dataset_id, f)
    row = db.query_one(f"SELECT COUNT(*) AS c FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where}", params)
    return int(row["c"]) if row else 0


def list_images(dataset_id: str, f: Dict[str, Any], offset: int = 0, limit: int = 80,
                sort: Optional[str] = None, order: Optional[str] = None, with_boxes: bool = False,
                with_paths: bool = False) -> Tuple[int, List[dict]]:
    where, params = build_where(dataset_id, f)
    total = count_images(dataset_id, f)
    rows = db.query(f"{BASE_SELECT} WHERE {where} ORDER BY {order_clause(sort, order)} LIMIT ? OFFSET ?",
                    params + [limit, offset])
    items = [row_to_item(r, with_paths) for r in rows]
    if with_boxes and items:
        attach_boxes(items)
    return total, items


def iter_images(dataset_id: str, f: Dict[str, Any], sort: Optional[str] = None, order: Optional[str] = None,
                chunk: int = 2000):
    where, params = build_where(dataset_id, f)
    offset = 0
    while True:
        rows = db.query(f"{BASE_SELECT} WHERE {where} ORDER BY {order_clause(sort, order)} LIMIT ? OFFSET ?",
                        params + [chunk, offset])
        if not rows:
            return
        for r in rows:
            yield row_to_item(r, with_paths=True)
        if len(rows) < chunk:
            return
        offset += chunk


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


NEEDS = {"nobox": "i.n_boxes = 0", "label": "i.has_label = 1", "issue": "i.issues != '[]'", "unreviewed": "r.status IS NULL"}


def neighbor(dataset_id: str, f: Dict[str, Any], current_id: int, direction: str = "next",
             sort: Optional[str] = None, order: Optional[str] = None, need: Optional[str] = None) -> Optional[dict]:
    """The next/previous image after `current_id` in the list's own order (filters + sort), optionally
    only among images that also satisfy `need` (e.g. "nobox" for Annotate's next-unlabeled). The
    current image does not have to match the filters; its sort key is the starting point."""
    where, params = build_where(dataset_id, f)
    if need in NEEDS:
        where = f"{where} AND {NEEDS[need]}"
    expr = _sort_spec(sort, order)[0]
    cur = db.query_one(f"SELECT {expr} AS k {_JOIN} WHERE i.id = ? AND i.dataset_id = ?", (current_id, dataset_id))
    if not cur:
        return None
    after = direction != "prev"
    pred, needs_tie = _beyond(sort, order, after=after)
    row = db.query_one(f"{BASE_SELECT} WHERE {where} AND {pred} ORDER BY {order_clause(sort, order, reverse=not after)} LIMIT 1",
                       params + _bind(cur["k"], current_id, needs_tie))
    return row_to_item(row) if row else None
