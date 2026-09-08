"""Shared image query builder (filters / sorting) used by the API and exporters."""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from .. import db

SORTS = {
    "name": "i.rel_path {o}, i.id {o}",
    "id": "i.id {o}",
    "boxes": "i.n_boxes {o}, i.id ASC",
    "size": "(COALESCE(i.width,0)*COALESCE(i.height,0)) {o}, i.id ASC",
    "mtime": "i.mtime {o}, i.id ASC",
    "reviewed": "COALESCE(r.updated_at, 0) {o}, i.id ASC",
    "random": "((i.id * 2654435761) % 4294967296) {o}",
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

    q = (f.get("q") or "").strip()
    if q:
        where.append("i.rel_path LIKE ? ESCAPE '\\'")
        params.append("%" + q.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%")

    seq = f.get("seq")
    if seq:
        where.append("i.seq = ?")
        params.append(seq)

    ids = [int(x) for x in _csv(f.get("ids")) if x.isdigit()]
    if ids:
        where.append("i.id IN (%s)" % ",".join("?" * len(ids)))
        params.extend(ids)

    for key, op in (("min_boxes", ">="), ("max_boxes", "<=")):
        v = f.get(key)
        if v not in (None, ""):
            try:
                where.append(f"i.n_boxes {op} ?")
                params.append(int(v))
            except ValueError:
                pass

    return " AND ".join(where), params


def order_clause(sort: Optional[str], order: Optional[str]) -> str:
    o = "DESC" if (order or "").lower() == "desc" else "ASC"
    tmpl = SORTS.get(sort or "id", SORTS["id"])
    return tmpl.format(o=o)


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


def neighbor(dataset_id: str, f: Dict[str, Any], current_id: int, direction: str = "next",
             sort: Optional[str] = None, order: Optional[str] = None) -> Optional[dict]:
    """Next/prev image id in id order matching the filters (used for 'next unlabeled')."""
    where, params = build_where(dataset_id, f)
    if direction == "prev":
        row = db.query_one(f"{BASE_SELECT} WHERE {where} AND i.id < ? ORDER BY i.id DESC LIMIT 1", params + [current_id])
    else:
        row = db.query_one(f"{BASE_SELECT} WHERE {where} AND i.id > ? ORDER BY i.id ASC LIMIT 1", params + [current_id])
    return row_to_item(row) if row else None
