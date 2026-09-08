from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException

from .. import db
from ..core import yolo
from ..core.labels import read_label_file, sanitize_boxes, write_label_file
from ..core.scanner import refresh_image
from ..jobs import jobs
from .common import get_dataset_or_404
from .datasets import _select_target_ids

router = APIRouter(prefix="/api/ai", tags=["ai"])


@router.get("/models")
def models():
    return {"available": yolo.available(), "models": yolo.list_models()}


@router.get("/models/{name}/classes")
def model_classes(name: str):
    if not yolo.available():
        raise HTTPException(503, "ultralytics is not installed")
    try:
        return {"classes": yolo.model_classes(name)}
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Failed to load model: {e}")


def _map_dets(dets: List[List[float]], class_map: Optional[Dict], keep_unmapped: bool) -> List[List[float]]:
    out = []
    cmap = {int(k): int(v) for k, v in (class_map or {}).items() if v is not None and str(v) != ""}
    for d in dets:
        c = int(d[0])
        if c in cmap:
            out.append([cmap[c]] + list(d[1:]))
        elif keep_unmapped or not cmap:
            out.append(list(d))
    return out


@router.post("/detect")
def detect(payload: dict = Body(...)):
    if not yolo.available():
        raise HTTPException(503, "ultralytics is not installed")
    image_id = int(payload.get("image_id") or 0)
    row = db.query_one("SELECT abs_path FROM images WHERE id=?", (image_id,))
    if not row:
        raise HTTPException(404, "Image not found")
    try:
        dets = yolo.detect(payload.get("model") or "", row["abs_path"], float(payload.get("conf", 0.25)),
                           payload.get("classes") or None, payload.get("imgsz"))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(500, f"Detection failed: {e}")
    mapped = _map_dets(dets, payload.get("class_map"), bool(payload.get("keep_unmapped", True)))
    return {"detections": mapped, "raw": dets, "count": len(mapped)}


@router.post("/autolabel")
def autolabel(payload: dict = Body(...)):
    """Background auto-labelling job. mode: replace | append | fill (only images without boxes)."""
    if not yolo.available():
        raise HTTPException(503, "ultralytics is not installed")
    ds_id = payload.get("dataset_id") or ""
    row = get_dataset_or_404(ds_id)
    targets = _select_target_ids(ds_id, payload)
    if not targets:
        raise HTTPException(400, "No target images")
    model = payload.get("model") or ""
    conf = float(payload.get("conf", 0.25))
    classes = payload.get("classes") or None
    class_map = payload.get("class_map") or {}
    keep_unmapped = bool(payload.get("keep_unmapped", False))
    mode = payload.get("mode", "fill")
    imgsz = payload.get("imgsz")
    ds_classes = db.loads(row["classes"])
    n_classes = len(ds_classes) if row.get("classes_source") != "inferred" and ds_classes else None

    def run(job):
        yolo.load(model)
        done_imgs = boxes_added = skipped = 0
        job.update(0, len(targets), f"Auto-label with {model}")
        for k, iid in enumerate(targets):
            img = db.query_one("SELECT abs_path, label_path, n_boxes FROM images WHERE id=?", (iid,))
            if not img or not img["label_path"]:
                continue
            lp = Path(img["label_path"])
            existing = read_label_file(lp)[0] if lp.is_file() else []
            if mode == "fill" and existing:
                skipped += 1
                job.update(done=k + 1, message=f"{k + 1}/{len(targets)} (skipped labeled)")
                continue
            dets = yolo.detect(model, img["abs_path"], conf, classes, imgsz)
            new = sanitize_boxes([d[:5] for d in _map_dets(dets, class_map, keep_unmapped)])
            if not new and mode != "replace" and not lp.is_file():
                skipped += 1  # nothing detected: do not create an empty label file
                job.update(done=k + 1, message=f"{k + 1}/{len(targets)} (no detections)")
                continue
            final = (existing + new) if mode == "append" else new
            write_label_file(lp, final)
            refresh_image(iid, n_classes)
            done_imgs += 1
            boxes_added += len(new)
            job.update(done=k + 1, message=f"{k + 1}/{len(targets)} · {boxes_added} boxes")
        return {"images": done_imgs, "boxes": boxes_added, "skipped": skipped}

    job = jobs.submit("autolabel", run, dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict()}
