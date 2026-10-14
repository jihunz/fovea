"""Legacy facade over the AI registry's local YOLO provider (kept for /api/meta and older callers).

``available()`` never imports torch/ultralytics (a cached ``find_spec``): the old implementation imported them on
the first /api/meta call and stalled every other request for 10+ s (finding C13). ``list_models()`` lists detector
weights only — MODEL_DIR also holds SAM checkpoints and text encoders now.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..ai import local, weights


def available() -> bool:
    return local.ultralytics_status()[0]


def list_models() -> List[dict]:
    out = []
    for w in weights.local_weights("yolo"):
        if w["task"] == "classify":
            continue
        out.append({"name": w["name"], "file": w["file"], "size_mb": w["size_mb"], "task": w["task"],
                    "loaded": local.CACHE.is_loaded("yolo", w["name"])})
    return out


def model_classes(name: str) -> Dict[int, str]:
    from ..ai.providers.yolo import model_names
    return model_names(name, "yolo")


def load(name: str):
    from ..ai.providers.yolo import load_entry
    return load_entry(name, "yolo").obj


def detect(name: str, image_path: str, conf: float = 0.25, classes: Optional[List[int]] = None,
           imgsz: Optional[int] = None, iou: float = 0.7) -> List[List[float]]:
    """Raw model-id detections ``[[model_cls, xc, yc, w, h, conf]]`` — callers must map ids before writing."""
    from ..ai.providers.yolo import clean_imgsz, load_entry
    entry = load_entry(name, "yolo")
    raw = local.predict(entry, image_path, conf=conf, iou=iou, imgsz=clean_imgsz(imgsz), classes=classes)
    return [[d["mcls"], d["xc"], d["yc"], d["w"], d["h"], d["conf"]] for d in raw]


def unload(name: Optional[str] = None) -> None:
    local.CACHE.unload("yolo", name)
