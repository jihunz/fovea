"""Segment Anything as a box tool: tighten a rough box, or turn clicks into a box (SAM2.1-t default, MobileSAM fallback).

The image encoder runs once per image (0.15-0.7 s on MPS); its features are cached for the last few images, so
every further click on the same image costs one prompt decode (~50 ms). The box comes from the mask, ignoring specks
smaller than a fifth of the main region, so stray pixels do not loosen it.
"""
from __future__ import annotations

import os
import time
from collections import OrderedDict
from typing import List, Optional

from .. import local, weights
from ..base import Provider, ProviderError
from ..geometry import clamp_box

PREFERRED = ("sam2.1_t", "mobile_sam", "sam2.1_s", "sam2_t", "sam2.1_b", "sam_b")
FEATURE_CACHE = 3


def _pick(model: str) -> str:
    names = [w["name"] for w in weights.local_weights("sam")]
    if model:
        if model not in names:
            raise ProviderError(f"SAM model not found in the model folder: {model}", "invalid")
        return model
    for p in PREFERRED:
        if p in names:
            return p
    if names:
        return names[0]
    raise ProviderError("No SAM weights in the model folder — download sam2.1_t or mobile_sam in Settings", "unavailable")


def _loader(path: str):
    def load():
        local.ensure_ultralytics()
        device = local.resolve_device()
        from ultralytics.models.sam import Predictor, SAM2Predictor
        stem = os.path.basename(path).lower()
        cls = SAM2Predictor if stem.startswith("sam2") else Predictor
        pred = cls(overrides=dict(conf=0.0, task="segment", mode="predict", imgsz=1024, model=path, device=device,
                                  save=False, verbose=False))
        pred.setup_model(model=None, verbose=False)
        try:   # compile the encoder and both decoder paths now, not on the user's first click
            import numpy as np
            pred.set_image(np.full((256, 256, 3), 127, dtype=np.uint8))
            pred(bboxes=[[64, 64, 192, 192]], multimask_output=False)
            pred(points=[[[128, 128]]], labels=[[1]], multimask_output=True)
        except Exception:  # noqa: BLE001 — warm-up is best effort
            pass
        finally:
            pred.reset_image()
        return pred, device
    return load


def _mask_box(mask, W: int, H: int) -> Optional[list]:
    """Tight box of the main region(s) of a boolean mask (H x W): drops specks < 20 % of the largest component."""
    import numpy as np
    m = np.asarray(mask).astype("uint8")
    if m.sum() == 0:
        return None
    try:
        import cv2
        n, lab, stats, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
        if n > 2:
            areas = stats[1:, cv2.CC_STAT_AREA]
            keep = [i + 1 for i, a in enumerate(areas) if a >= 0.2 * areas.max()]
            m = np.isin(lab, keep).astype("uint8")
    except Exception:  # noqa: BLE001 — fall back to the raw mask extent
        pass
    ys, xs = np.nonzero(m)
    if xs.size == 0:
        return None
    b = clamp_box((xs.min() + xs.max() + 1) / 2 / W, (ys.min() + ys.max() + 1) / 2 / H,
                  (xs.max() + 1 - xs.min()) / W, (ys.max() + 1 - ys.min()) / H)
    return list(b) if b else None


def warm(model: str = "") -> None:
    name = _pick(model or "")
    path = str(weights.resolve(name, "sam"))
    local.ensure_ultralytics()
    device = local.resolve_device()
    local.CACHE.get(("sam", path, "prompt", device), _loader(path), "sam", name)


class SamProvider(Provider):
    id = "sam"
    name = "Segment Anything"
    kind = "local"
    location = "Runs on this machine — images never leave it"
    needs_key = False
    capabilities = {"detect": False, "classify": False, "refine": True, "open_vocab": False, "exemplars": False,
                    "batch_api": False, "confidence": True}

    def status(self) -> dict:
        ok, reason = local.ultralytics_status()
        if not ok:
            return {"available": False, "reason": reason}
        if not weights.local_weights("sam"):
            return {"available": False, "reason": "No SAM weights in the model folder — download sam2.1_t in Settings"}
        return {"available": True, "reason": None}

    def models(self, refresh: bool = False) -> List[dict]:
        ws = weights.local_weights("sam")
        try:
            default = _pick("")
        except ProviderError:
            default = None
        out = []
        for w in ws:
            ms = local.speed_ms("sam", w["name"])
            notes = [f"{w['size_mb']:.0f} MB", "box refine · click-to-box"]
            if ms:
                notes.append(f"~{ms:.0f} ms per new image here")
            out.append({"id": w["name"], "name": w["name"], "notes": " · ".join(notes), "recommended": w["name"] == "sam2.1_t",
                        "default": w["name"] == default, "price": None, "max_edge": None, "family": "sam",
                        "size_mb": w["size_mb"], "loaded": local.CACHE.is_loaded("sam", w["name"])})
        return out

    def test(self) -> dict:
        st = self.status()
        if not st["available"]:
            return {"ok": False, "message": st["reason"], "latency_ms": 0}
        t0 = time.time()
        local.ensure_ultralytics()
        return {"ok": True, "message": f"ready on {local.resolve_device()} (default model {_pick('')})",
                "latency_ms": int((time.time() - t0) * 1000)}

    def refine(self, image_path: str, box=None, points=None, model: str = "") -> dict:
        name = _pick(model or "")
        try:
            path = str(weights.resolve(name, "sam"))
        except ValueError as e:
            raise ProviderError(str(e), "invalid") from None
        local.ensure_ultralytics()
        device = local.resolve_device()
        entry = local.CACHE.get(("sam", path, "prompt", device), _loader(path), "sam", name)
        try:
            st = os.stat(image_path)
        except OSError:
            raise ProviderError("Image file not found on disk", "invalid") from None
        local.check_pixels(image_path)          # SAM's loader would decode a decompression bomb whole
        img_key = (image_path, st.st_mtime, st.st_size)
        t0 = time.time()
        with entry.lock:
            pred = entry.obj
            feats: OrderedDict = entry.state.setdefault("features", OrderedDict())
            if img_key in feats:
                pred.setup_source(image_path)
                pred.features = feats[img_key][0]
                W, H = feats[img_key][1]
                encoded = False
            else:
                W, H = local.image_size(image_path)
                pred.set_image(image_path)
                feats[img_key] = (pred.features, (W, H))
                while len(feats) > FEATURE_CACHE:
                    feats.popitem(last=False)
                encoded = True
            feats.move_to_end(img_key)
            kw = {}
            if box is not None:
                xc, yc, w, h = (float(v) for v in box[:4])
                kw["bboxes"] = [[(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H]]
            if points:
                kw["points"] = [[[float(p[0]) * W, float(p[1]) * H] for p in points]]
                kw["labels"] = [[1 if len(p) < 3 or int(p[2]) else 0 for p in points]]
            if not kw:
                raise ProviderError("Give a box or at least one point", "invalid")
            multi = box is None
            res = pred(multimask_output=multi, **kw)
            r = res[0]
            cands = []
            masks = getattr(r, "masks", None)
            confs = r.boxes.conf.tolist() if r.boxes is not None else []
            if masks is not None and masks.data is not None:
                data = masks.data.cpu().numpy()
                for k in range(data.shape[0]):
                    b = _mask_box(data[k], W, H)
                    if b:
                        cands.append({"box": [round(v, 6) for v in b], "score": round(float(confs[k]) if k < len(confs) else 0.0, 4)})
            elif r.boxes is not None:
                for (x0, y0, x1, y1), s in zip(r.boxes.xyxy.tolist(), confs):
                    b = clamp_box((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)
                    if b:
                        cands.append({"box": list(b), "score": round(float(s), 4)})
        ms = (time.time() - t0) * 1000
        if encoded:
            entry.record_ms(ms)
        if not cands:
            raise ProviderError("SAM found no object there", "bad_output")
        cands.sort(key=lambda c: -c["score"])
        return {"box": cands[0]["box"], "score": cands[0]["score"], "candidates": cands, "model": name,
                "device": entry.device, "latency_ms": int(ms), "encoded": encoded}


PROVIDER = SamProvider()
