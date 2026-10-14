"""Closed-vocabulary Ultralytics detectors (YOLO11/12/26, RT-DETR …) from MODEL_DIR.

Model classes are mapped to dataset classes by an explicit map first, then by name (case/space/plural-insensitive plus
synonyms). Detections of unmapped model classes are skipped and counted in the warnings — a raw COCO id is never
written into a dataset. Non-detection checkpoints (image classifiers) are refused with a clear message; OBB boxes
become their axis-aligned extent; seg/pose checkpoints contribute their boxes.
"""
from __future__ import annotations

import os
import threading
import time
from collections import Counter
from typing import Dict, List, Optional

from .. import local, weights
from ..base import DetectRequest, DetectResult, Detection, Provider, ProviderError, Usage
from ..classes import map_vocab

RECOMMENDED = ("yolo26l", "yolo26m", "yolo12m", "yolo11m")
_names_cache: Dict[tuple, Dict[int, str]] = {}
_names_lock = threading.Lock()


def clean_imgsz(v, default: int = 640) -> int:
    if v in (None, ""):
        return default
    try:
        n = int(v)
    except (TypeError, ValueError):
        raise ProviderError("imgsz must be an integer", "invalid") from None
    if not 32 <= n <= 4096:
        raise ProviderError("imgsz must be between 32 and 4096", "invalid")
    return max(32, (n // 32) * 32)


def clean_float(v, name: str, lo: float, hi: float, default: float) -> float:
    if v in (None, ""):
        return default
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ProviderError(f"{name} must be a number", "invalid") from None
    if not lo <= f <= hi:
        raise ProviderError(f"{name} must be between {lo} and {hi}", "invalid")
    return f


def _load(path: str, family: str):
    def loader():
        local.ensure_ultralytics()
        device = local.resolve_device()
        if family == "yoloe":
            from ultralytics import YOLOE
            m = YOLOE(path)
        else:
            from ultralytics import YOLO
            m = YOLO(path)
        task = getattr(m, "task", "detect")
        if task == "classify":
            raise ProviderError(f"{os.path.basename(path)} is an image classifier, not a detector — pick a detection "
                                "model", "invalid")
        if task not in ("detect", "segment", "pose", "obb"):
            raise ProviderError(f"{os.path.basename(path)} is a {task} model and produces no boxes", "invalid")
        local.warmup_predict(m, device)
        return m, device
    return loader


def load_entry(name: str, family: str = "yolo", mode: str = "nms") -> local.Loaded:
    try:
        path = str(weights.resolve(name, family))
    except ValueError as e:
        raise ProviderError(str(e), "invalid") from None
    local.ensure_ultralytics()
    device = local.resolve_device()
    return local.CACHE.get((family, path, mode, device), _load(path, family), family, name)


def model_names(name: str, family: str = "yolo") -> Dict[int, str]:
    """Class names of a checkpoint WITHOUT keeping the model in memory (finding C14): from the loaded model when it is
    cached, else from the checkpoint metadata (cached by path + mtime)."""
    try:
        p = weights.resolve(name, family)
    except ValueError as e:
        raise ProviderError(str(e), "invalid") from None
    key = (str(p), p.stat().st_mtime)
    with _names_lock:
        if key in _names_cache:
            return _names_cache[key]
    local.ensure_ultralytics()
    names = None
    for e in list(local.CACHE._entries.values()):         # noqa: SLF001 — read-only peek
        if e.key[1] == str(p):
            names = getattr(e.obj, "names", None)
            break
    if names is None:
        if p.suffix == ".pt":
            from ultralytics.nn.tasks import torch_safe_load
            ckpt, _ = torch_safe_load(str(p))
            mdl = (ckpt.get("ema") or ckpt.get("model")) if isinstance(ckpt, dict) else None
            names = getattr(mdl, "names", None)
            task = getattr(mdl, "task", None) or (getattr(mdl, "yaml", {}) or {}).get("task")
            del ckpt, mdl
            if task == "classify":
                raise ProviderError(f"{name} is an image classifier, not a detector", "invalid")
        if names is None:
            from ultralytics import YOLO
            m = YOLO(str(p))
            names = m.names
            del m
    if isinstance(names, (list, tuple)):
        names = dict(enumerate(names))
    out = {int(k): str(v) for k, v in (names or {}).items()}
    with _names_lock:
        _names_cache[key] = out
    return out


def same_config(reqs: List[DetectRequest]) -> bool:
    """Every request asks the same model the same question (only then can they share one forward pass)."""
    r0 = reqs[0]

    def same(a, b):
        return a is b or a == b
    return all(r.model == r0.model and r.conf == r0.conf and same(r.options, r0.options) and same(r.classes, r0.classes)
               for r in reqs[1:])


def skipped_warning(skipped: Counter) -> Optional[str]:
    if not skipped:
        return None
    n = sum(skipped.values())
    top = ", ".join(f"{k} ×{v}" for k, v in skipped.most_common(4))
    more = "…" if len(skipped) > 4 else ""
    return f"{n} box{'es' if n != 1 else ''} of classes not in this dataset skipped ({top}{more})"


class YoloProvider(Provider):
    id = "yolo"
    name = "Ultralytics YOLO"
    kind = "local"
    location = "Runs on this machine — images never leave it"
    needs_key = False
    capabilities = {"detect": True, "classify": False, "refine": False, "open_vocab": False, "exemplars": False,
                    "batch_api": False, "confidence": True, "closed_vocab": True, "tiles": True, "batch_infer": True}

    def status(self) -> dict:
        ok, reason = local.ultralytics_status()
        if not ok:
            return {"available": False, "reason": reason}
        if not [w for w in weights.local_weights("yolo") if w["task"] != "classify"]:
            return {"available": False, "reason": "No detector weights in the model folder — download one in Settings"}
        return {"available": True, "reason": None}

    def models(self, refresh: bool = False) -> List[dict]:
        ws = [w for w in weights.local_weights("yolo")]
        rec = next((r for r in RECOMMENDED if any(w["name"] == r for w in ws)), None)
        smallest = min((w for w in ws if w["task"] != "classify"), key=lambda w: w["size_mb"], default=None)
        out = []
        for w in ws:
            usable = w["task"] != "classify"
            ms = local.speed_ms("yolo", w["name"])
            notes = [f"{w['size_mb']:.0f} MB", w["task"] if w["task"] != "detect" else "COCO-style closed vocabulary"]
            if not usable:
                notes.append("image classifier — cannot draw boxes")
            if ms:
                notes.append(f"~{ms:.0f} ms/image here")
            out.append({"id": w["name"], "name": w["name"], "notes": " · ".join(notes), "recommended": w["name"] == rec,
                        "default": bool(smallest and w["name"] == smallest["name"]), "price": None, "max_edge": None,
                        "family": "yolo", "task": w["task"], "size_mb": w["size_mb"], "usable": usable,
                        "loaded": local.CACHE.is_loaded("yolo", w["name"])})
        return out

    def test(self) -> dict:
        t0 = time.time()
        st = self.status()
        if not st["available"]:
            return {"ok": False, "message": st["reason"], "latency_ms": 0}
        try:
            local.ensure_ultralytics()
            dev = local.resolve_device()
        except ProviderError as e:
            return {"ok": False, "message": str(e), "latency_ms": int((time.time() - t0) * 1000)}
        return {"ok": True, "message": f"ultralytics ready on {dev}", "latency_ms": int((time.time() - t0) * 1000)}

    def estimate(self, width, height, model, n_classes, options):
        ms = local.speed_ms("yolo", model)
        return {"usd": [0.0, 0.0], "tokens_in": [0, 0], "tokens_out": [0, 0], "seconds": (ms / 1000) if ms else None}

    def _mapping(self, req: DetectRequest) -> Dict[int, int]:
        names = model_names(req.model)
        try:
            mapping, _how = map_vocab(names, req.classes, (req.options or {}).get("class_map"))
        except ValueError as e:
            raise ProviderError(str(e), "invalid") from None
        if not mapping:
            want = ", ".join(c.name for c in req.classes[:6]) + ("…" if len(req.classes) > 6 else "")
            raise ProviderError(
                f"None of {req.model}'s {len(names)} classes match the classes being labeled ({want}). Map them in "
                "the class map, use YOLOE (open vocabulary) or a two-stage pipeline.", "invalid")
        return mapping

    @staticmethod
    def _result(raw: List[dict], mapping: Dict[int, int], ms: int, served: str) -> DetectResult:
        dets: List[Detection] = []
        skipped: Counter = Counter()
        for d in raw:
            did = mapping.get(d["mcls"])
            if did is None:
                skipped[d["label"]] += 1
                continue
            dets.append(Detection(did, d["xc"], d["yc"], d["w"], d["h"], d["conf"], d["label"]))
        warnings = [w for w in [skipped_warning(skipped)] if w]
        return DetectResult(dets, Usage(), ms, warnings, served, covered=sorted(set(mapping.values())))

    def detect(self, req: DetectRequest) -> DetectResult:
        t0 = time.time()
        o = req.options or {}
        imgsz = clean_imgsz(o.get("imgsz"))
        iou = clean_float(o.get("iou"), "iou", 0.05, 0.95, 0.7)
        conf = clean_float(req.conf, "conf", 0.0, 1.0, 0.25)
        mapping = self._mapping(req)
        entry = load_entry(req.model, "yolo")
        raw = local.predict(entry, req.image_path, conf=conf, iou=iou, imgsz=imgsz,
                            classes=sorted(mapping) if o.get("filter_classes") else None,
                            max_det=int(o.get("max_objects") or 300), tile=o.get("tiles"), augment=bool(o.get("tta")))
        return self._result(raw, mapping, int((time.time() - t0) * 1000), f"{req.model} on {entry.device}")

    def detect_many(self, reqs: List[DetectRequest]) -> List[object]:
        """One forward pass for the whole chunk (requests of one run share model, classes and options)."""
        if not reqs:
            return []
        r0 = reqs[0]
        o = r0.options or {}
        if o.get("tiles") or not same_config(reqs):
            return super().detect_many(reqs)
        t0 = time.time()
        imgsz = clean_imgsz(o.get("imgsz"))
        iou = clean_float(o.get("iou"), "iou", 0.05, 0.95, 0.7)
        conf = clean_float(r0.conf, "conf", 0.0, 1.0, 0.25)
        mapping = self._mapping(r0)
        entry = load_entry(r0.model, "yolo")
        arrays = local.decode_many([r.image_path for r in reqs])
        ok = [k for k, a in enumerate(arrays) if not isinstance(a, Exception)]
        raws = local.predict_arrays(entry, [arrays[k] for k in ok], conf=conf, iou=iou, imgsz=imgsz,
                                    classes=sorted(mapping) if o.get("filter_classes") else None,
                                    max_det=int(o.get("max_objects") or 300), augment=bool(o.get("tta")))
        ms = int((time.time() - t0) * 1000 / max(1, len(ok)))
        out: List[object] = [(a if isinstance(a, ProviderError) else ProviderError(f"Could not read the image: {a}", "bad_output"))
                             if isinstance(a, Exception)
                             else None for a in arrays]
        for k, raw in zip(ok, raws):
            out[k] = self._result(raw, mapping, ms, f"{r0.model} on {entry.device}")
        return out


PROVIDER = YoloProvider()
