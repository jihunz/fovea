from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional

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


def check_reachable(row: dict, sample: int = 5) -> dict:
    """Cheap probe (a handful of stat calls) telling the UI whether this dataset's files are
    actually on disk right now. An index built elsewhere — inside Docker, or on a drive that is
    no longer mounted — otherwise looks perfectly healthy while every image 404s."""
    layout = db.loads(row["layout"], {})
    sources = layout.get("sources", [])
    missing_sources: List[str] = []
    for src in sources:
        target = src.get("list_file") or src.get("img_dir")
        if target and not Path(target).exists():
            missing_sources.append(to_host(target))

    # Sample SPREAD across the id range, not the first N rows: a handful of deleted files at the
    # start of a dataset must not masquerade as "the whole dataset is gone".
    missing_files = 0
    checked = 0
    if sample > 0 and row.get("image_count"):
        bounds = db.query_one("SELECT MIN(id) lo, MAX(id) hi FROM images WHERE dataset_id=?", (row["id"],))
        if bounds and bounds["lo"] is not None:
            lo, hi = bounds["lo"], bounds["hi"]
            seen = set()
            for k in range(sample):
                at = lo + ((hi - lo) * k) // max(1, sample - 1) if sample > 1 else lo
                r = db.query_one(
                    "SELECT id, abs_path FROM images WHERE dataset_id=? AND id >= ? ORDER BY id LIMIT 1",
                    (row["id"], at),
                ) or db.query_one(
                    "SELECT id, abs_path FROM images WHERE dataset_id=? ORDER BY id DESC LIMIT 1", (row["id"],)
                )
                if not r or r["id"] in seen:
                    continue
                seen.add(r["id"])
                checked += 1
                if not os.path.isfile(r["abs_path"]):
                    missing_files += 1

    root_exists = bool(row.get("root")) and Path(row["root"]).exists()
    all_sources_gone = bool(sources) and len(missing_sources) == len(sources)
    all_files_gone = checked > 0 and missing_files == checked
    return {
        "ok": not (all_sources_gone or all_files_gone) and (root_exists or not sources),
        "root_exists": root_exists,
        "missing_sources": missing_sources[:5],
        "source_count": len(sources),
        "sampled": checked,
        "sample_missing": missing_files,
    }


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
        "reachable": check_reachable(row),
    }
    if with_job:
        job = jobs.active_for(row["id"])
        out["job"] = job.to_dict() if job else None
    return out
