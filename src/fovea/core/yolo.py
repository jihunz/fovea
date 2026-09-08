"""Optional Ultralytics YOLO integration (lazy; degrades gracefully when not installed)."""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..config import MODEL_DIR

_models: Dict[str, Any] = {}
_lock = threading.Lock()


def available() -> bool:
    try:
        import ultralytics  # noqa: F401
        return True
    except Exception:
        return False


def list_models() -> List[dict]:
    out = []
    if MODEL_DIR.is_dir():
        for p in sorted(MODEL_DIR.iterdir()):
            if p.suffix.lower() in (".pt", ".onnx", ".engine") and p.is_file():
                out.append({"name": p.stem, "file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1),
                            "loaded": p.stem in _models})
    return out


def _path_for(name: str) -> Path:
    for ext in (".pt", ".onnx", ".engine"):
        p = MODEL_DIR / f"{name}{ext}"
        if p.is_file():
            return p
    raise FileNotFoundError(f"Model not found: {name}")


def load(name: str):
    with _lock:
        m = _models.get(name)
        if m is not None:
            return m
        from ultralytics import YOLO
        m = YOLO(str(_path_for(name)))
        _models[name] = m
        return m


def model_classes(name: str) -> Dict[int, str]:
    m = load(name)
    names = getattr(m, "names", None) or {}
    if isinstance(names, list):
        return {i: n for i, n in enumerate(names)}
    return {int(k): str(v) for k, v in names.items()}


def detect(name: str, image_path: str, conf: float = 0.25, classes: Optional[List[int]] = None,
           imgsz: Optional[int] = None, iou: float = 0.7) -> List[List[float]]:
    m = load(name)
    kwargs: Dict[str, Any] = {"conf": conf, "verbose": False, "iou": iou}
    if classes:
        kwargs["classes"] = list(classes)
    if imgsz:
        kwargs["imgsz"] = int(imgsz)
    results = m(str(image_path), **kwargs)
    dets: List[List[float]] = []
    for r in results:
        b = getattr(r, "boxes", None)
        if b is None:
            continue
        xywhn = b.xywhn.tolist()
        cls = b.cls.tolist()
        cf = b.conf.tolist()
        for i in range(len(xywhn)):
            xc, yc, w, h = xywhn[i]
            dets.append([int(cls[i]), round(xc, 6), round(yc, 6), round(w, 6), round(h, 6), round(float(cf[i]), 4)])
    return dets


def unload(name: Optional[str] = None) -> None:
    with _lock:
        if name:
            _models.pop(name, None)
        else:
            _models.clear()
