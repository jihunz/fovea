from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import HTTPException, Request

from .. import db
from ..jobs import jobs
from ..paths import to_host

FILTER_KEYS = ("split", "cls", "only_cls", "labeled", "review", "exclude_review", "issue", "q", "seq", "ids",
               "min_boxes", "max_boxes")


def filters_from(source: Any) -> Dict[str, Any]:
    if isinstance(source, Request):
        qp = source.query_params
        return {k: qp.get(k) for k in FILTER_KEYS if qp.get(k) not in (None, "")}
    if isinstance(source, dict):
        return {k: source.get(k) for k in FILTER_KEYS if source.get(k) not in (None, "", [])}
    return {}


def get_dataset_or_404(ds_id: str) -> dict:
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    if not row:
        raise HTTPException(404, f"Dataset not found: {ds_id}")
    return row


def dataset_public(row: dict, with_job: bool = True) -> dict:
    layout = db.loads(row["layout"], {})
    reviews = {r["status"]: r["c"] for r in db.query("SELECT status, COUNT(*) c FROM reviews WHERE dataset_id=? GROUP BY status", (row["id"],))}
    out = {
        "id": row["id"], "name": row["name"], "root": row["root"], "root_host": to_host(row["root"]),
        "layout": layout, "classes": db.loads(row["classes"]), "classes_source": row.get("classes_source", "inferred"),
        "description": row["description"], "status": row["status"],
        "image_count": row["image_count"], "label_count": row["label_count"], "box_count": row["box_count"],
        "cover_image_id": row["cover_image_id"],
        "created_at": row["created_at"], "updated_at": row["updated_at"], "scanned_at": row["scanned_at"],
        "opened_at": row["opened_at"],
        "review": {"approved": reviews.get("approved", 0), "flagged": reviews.get("flagged", 0), "excluded": reviews.get("excluded", 0)},
        "splits": [s.get("split", "") for s in layout.get("sources", [])],
    }
    if with_job:
        job = jobs.active_for(row["id"])
        out["job"] = job.to_dict() if job else None
    return out
