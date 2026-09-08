"""Compare plugin: evaluate prediction label folders against the dataset's ground truth."""
from __future__ import annotations

import json
import uuid
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ... import db
from ...api.common import filters_from, get_dataset_or_404
from ...jobs import jobs
from ...paths import resolve, to_host
from ...plugin_api import Plugin
from .evaluate import discover_pred_dirs, evaluate, image_detail

router = APIRouter()
PLUGIN_ID = "compare"


@router.get("/discover")
def discover(path: str = Query(...)):
    root = resolve(path)
    if not root.is_dir():
        raise HTTPException(404, f"Not a directory: {path}")
    found = discover_pred_dirs(root)
    for f in found:
        f["path_host"] = to_host(f["path"])
    return {"dirs": found}


@router.post("/evaluate")
def start_evaluate(payload: dict = Body(...)):
    ds_id = payload.get("dataset_id") or ""
    ds = get_dataset_or_404(ds_id)
    pred_a = resolve(payload.get("pred_a") or "")
    if not pred_a.is_dir():
        raise HTTPException(400, f"Prediction folder A not found: {payload.get('pred_a')}")
    pred_b = resolve(payload["pred_b"]) if payload.get("pred_b") else None
    if pred_b is not None and not pred_b.is_dir():
        raise HTTPException(400, f"Prediction folder B not found: {payload.get('pred_b')}")
    iou_thr = float(payload.get("iou", 0.5))
    conf_thr = float(payload.get("conf", 0.25))
    filters = filters_from(payload.get("filters") or {})
    names = db.loads(ds["classes"])
    eval_id = uuid.uuid4().hex[:10]
    name = payload.get("name") or f"{pred_a.name}{' vs ' + pred_b.name if pred_b else ''} @IoU{iou_thr:g}/conf{conf_thr:g}"

    def run(job):
        res = evaluate(job, ds_id, names, pred_a, pred_b, iou_thr, conf_thr, filters)
        rec = {"id": eval_id, "name": name, "dataset_id": ds_id, "pred_a": str(pred_a), "pred_b": str(pred_b) if pred_b else None,
               "pred_a_host": to_host(pred_a), "pred_b_host": to_host(pred_b) if pred_b else None,
               "iou": iou_thr, "conf": conf_thr, "filters": filters, "created_at": db.now(), **res}
        db.execute("INSERT INTO plugin_data(plugin, dataset_id, key, value, updated_at) VALUES(?,?,?,?,?) ON CONFLICT(plugin, dataset_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at",
                   (PLUGIN_ID, ds_id, eval_id, db.dumps(rec), db.now()))
        return {"eval_id": eval_id, "images": res["images"], "summary": res["summary"]}

    job = jobs.submit("compare", run, dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict(), "eval_id": eval_id}


def _load(ds_id: str, eval_id: str) -> dict:
    row = db.query_one("SELECT value FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?", (PLUGIN_ID, ds_id, eval_id))
    if not row:
        raise HTTPException(404, "Evaluation not found")
    return json.loads(row["value"])


@router.get("/evals")
def list_evals(dataset_id: str = Query(...)):
    out = []
    for r in db.query("SELECT key, value, updated_at FROM plugin_data WHERE plugin=? AND dataset_id=? ORDER BY updated_at DESC", (PLUGIN_ID, dataset_id)):
        rec = json.loads(r["value"])
        out.append({k: rec.get(k) for k in ("id", "name", "pred_a_host", "pred_b_host", "iou", "conf", "images", "created_at", "summary")})
    return {"evals": out}


@router.get("/evals/{eval_id}")
def get_eval(eval_id: str, dataset_id: str = Query(...), errors_only: int = Query(0), limit: int = Query(500), offset: int = Query(0)):
    rec = _load(dataset_id, eval_id)
    rows = rec["rows"]
    if errors_only:
        rows = [r for r in rows if (r.get("a", {}).get("fp", 0) + r.get("a", {}).get("fn", 0) + r.get("b", {}).get("fp", 0) + r.get("b", {}).get("fn", 0)) > 0]
    total = len(rows)
    rec = {**rec, "rows": rows[offset:offset + limit], "rows_total": total}
    return {"eval": rec}


@router.get("/evals/{eval_id}/image/{image_id}")
def get_eval_image(eval_id: str, image_id: int, dataset_id: str = Query(...)):
    rec = _load(dataset_id, eval_id)
    img = db.query_one("SELECT id, rel_path, split, width, height FROM images WHERE id=? AND dataset_id=?", (image_id, dataset_id))
    if not img:
        raise HTTPException(404, "Image not found")
    det = image_detail(image_id, img["rel_path"], img["split"], Path(rec["pred_a"]), Path(rec["pred_b"]) if rec.get("pred_b") else None, rec["iou"], rec["conf"])
    return {"image": img, "detail": det}


@router.delete("/evals/{eval_id}")
def delete_eval(eval_id: str, dataset_id: str = Query(...)):
    db.execute("DELETE FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?", (PLUGIN_ID, dataset_id, eval_id))
    return {"ok": True}


PLUGIN = Plugin(
    id=PLUGIN_ID, name="Compare", description="Evaluate prediction folders (YOLO txt with confidence) against ground truth: P/R/F1, AP@IoU, per-image errors, side-by-side viewer.",
    version="1.0.0", router=router, static_dir=Path(__file__).parent / "static", entry="compare.js",
    nav=[{"id": "compare", "label": "Compare", "icon": "gitCompare", "order": 45}],
)
