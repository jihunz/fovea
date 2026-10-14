"""YOLOE-26 open-vocabulary detection: text prompts from class cards, or visual prototypes built from existing labels.

* ``prompt_mode='text'`` (default): each class's prompt (card prompt or name) is embedded with the MobileCLIP text tower
  (``mobileclip2_b.ts`` in MODEL_DIR, loaded by absolute path — never downloaded into the CWD). Embeddings are cached in
  memory and on disk per (weights, prompts), so the 250 MB text tower only loads when a prompt changes.
* ``prompt_mode='exemplars'``: the caller passes ``options.exemplars = {cls_id: [(image_path, [[x0,y0,x1,y1] px …]), …]}``
  ``options.exemplar_key`` and ``options.exemplar_class_keys`` (content hashes of the reference sets picked, see
  ``pipeline.build_exemplars``). Each class's prototype is the normalised mean of its visual-prompt embeddings, cached
  per class key: a label save rebuilds only a class whose references changed. Works for classes CLIP cannot name
  ("class_3"), needs no text tower.
* ``-pf`` (prompt-free) checkpoints have a fixed 4,585-name vocabulary and are mapped like closed-vocab models.

Text and exemplar mode use separate model objects: the active vocabulary is model state and is set under the
model lock right before predict.
"""
from __future__ import annotations

import hashlib
import threading
import time
from collections import Counter, OrderedDict
from pathlib import Path
from typing import Dict, List, Tuple

from ... import config as app_config
from .. import geometry, local, weights
from ..base import DetectRequest, DetectResult, Detection, Provider, ProviderError, Usage
from ..classes import map_vocab
from .yolo import clean_float, clean_imgsz, load_entry, model_names, same_config, skipped_warning

TEXT_TOWERS = {"mobileclip2:b": "mobileclip2_b.ts", "mobileclip:blt": "mobileclip_blt.ts"}
_pe_mem: "OrderedDict[str, object]" = OrderedDict()
_pe_lock = threading.Lock()


def _pe_dir() -> Path:
    d = Path(app_config.DATA_DIR) / "ai_cache" / "yoloe_pe"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _remember(key: str, pe) -> None:
    with _pe_lock:
        _pe_mem[key] = pe
        _pe_mem.move_to_end(key)
        while len(_pe_mem) > 128:              # prototypes are a few KB: whole sets plus one entry per class
            _pe_mem.popitem(last=False)


def _recall(key: str):
    with _pe_lock:
        pe = _pe_mem.get(key)
        if pe is not None:
            _pe_mem.move_to_end(key)
        return pe


def text_tower_file(entry) -> str:
    variant = getattr(entry.obj.model, "text_model", "mobileclip:blt")
    if variant.startswith("clip:"):
        return ""
    return TEXT_TOWERS.get(variant, "mobileclip2_b.ts")


def text_ready(name: str) -> Tuple[bool, str]:
    """Can text prompts run for this checkpoint? Cheap (no model load): file + package checks only."""
    tower = "mobileclip2_b.ts" if "26" in name else "mobileclip_blt.ts"
    if not (Path(weights.MODEL_DIR) / tower).is_file():
        return False, f"text prompts need {tower} in the model folder (download it in Settings)"
    if not local.has_module("clip"):
        return False, "text prompts need the ultralytics CLIP package (pip install git+https://github.com/ultralytics/CLIP.git)"
    return True, ""


class YoloeProvider(Provider):
    id = "yoloe"
    name = "YOLOE (open vocabulary)"
    kind = "local"
    location = "Runs on this machine — images never leave it"
    needs_key = False
    capabilities = {"detect": True, "classify": False, "refine": False, "open_vocab": True, "exemplars": True,
                    "batch_api": False, "confidence": True, "tiles": True, "batch_infer": True}

    def status(self) -> dict:
        ok, reason = local.ultralytics_status()
        if not ok:
            return {"available": False, "reason": reason}
        if not weights.local_weights("yoloe"):
            return {"available": False, "reason": "No YOLOE weights in the model folder — download yoloe-26s-seg in Settings"}
        return {"available": True, "reason": None}

    def models(self, refresh: bool = False) -> List[dict]:
        out = []
        ws = weights.local_weights("yoloe")
        for w in ws:
            pf = w["name"].endswith("-pf")
            ok, why = text_ready(w["name"])
            ms = local.speed_ms("yoloe", w["name"])
            notes = [f"{w['size_mb']:.0f} MB"]
            notes.append("fixed 4,585-name vocabulary" if pf else ("text prompts + exemplars" if ok else f"exemplars only — {why}"))
            if ms:
                notes.append(f"~{ms:.0f} ms/image here")
            out.append({"id": w["name"], "name": w["name"], "notes": " · ".join(notes), "recommended": "26s" in w["name"],
                        "default": w is ws[0], "price": None, "max_edge": None, "family": "yoloe", "task": w["task"],
                        "size_mb": w["size_mb"], "prompt_free": pf, "text_prompts": (not pf) and ok,
                        "loaded": local.CACHE.is_loaded("yoloe", w["name"])})
        return out

    def test(self) -> dict:
        st = self.status()
        if not st["available"]:
            return {"ok": False, "message": st["reason"], "latency_ms": 0}
        t0 = time.time()
        local.ensure_ultralytics()
        msgs = [f"ultralytics ready on {local.resolve_device()}"]
        for m in self.models():
            if not m["prompt_free"] and not m["text_prompts"]:
                msgs.append(f"{m['id']}: {text_ready(m['id'])[1]}")
        return {"ok": True, "message": "; ".join(msgs), "latency_ms": int((time.time() - t0) * 1000)}

    def estimate(self, width, height, model, n_classes, options):
        ms = local.speed_ms("yoloe", model)
        return {"usd": [0.0, 0.0], "tokens_in": [0, 0], "tokens_out": [0, 0], "seconds": (ms / 1000) if ms else None}

    # ---- vocabularies
    def _text_pe(self, entry, queries: List[str]):
        key = "text|" + entry.name + "|" + "\x1f".join(queries)
        pe = _recall(key)
        if pe is not None:
            return pe
        import torch
        disk = _pe_dir() / (hashlib.sha1(key.encode()).hexdigest() + ".pt")
        if disk.is_file():
            try:
                pe = torch.load(str(disk), map_location="cpu", weights_only=True)
                _remember(key, pe)
                return pe
            except Exception:  # noqa: BLE001 — recompute a corrupt cache file
                pass
        tower = text_tower_file(entry)
        tpath = Path(weights.MODEL_DIR) / tower if tower else None
        if not tpath or not tpath.is_file():
            raise ProviderError(f"YOLOE text prompts need {tower or 'a text encoder'} in the model folder — download it "
                                "in Settings › Model weights, or switch to Your labels (exemplar prompts)", "unavailable")
        if not local.has_module("clip"):
            raise ProviderError("YOLOE text prompts need the ultralytics CLIP package "
                                "(pip install git+https://github.com/ultralytics/CLIP.git)", "unavailable")
        from ultralytics.nn.text_model import MobileCLIPTS
        # The TorchScript text tower runs on the CPU: on MPS it took 9 s to load and 20-30 s per call here,
        # on the CPU 0.3 s + ~1 s. Only the small auxiliary text head runs on the model's device.
        tm = MobileCLIPTS(torch.device("cpu"), weight=str(tpath))
        with torch.no_grad():
            feats = [tm.encode_text(t).detach() for t in tm.tokenize(list(queries)).split(80)]
            feats = feats[0] if len(feats) == 1 else torch.cat(feats, dim=0)
            feats = feats.reshape(-1, len(queries), feats.shape[-1])
        del tm
        with entry.lock:
            mdl = entry.obj.model
            dev = next(mdl.parameters()).device
            with torch.no_grad():
                pe = mdl.model[-1].get_tpe(feats.to(dev)).detach().cpu()
        local.free_device_memory()
        try:
            torch.save(pe, str(disk))
        except OSError:
            pass
        _remember(key, pe)
        return pe

    def _exemplar_pe(self, entry, req: DetectRequest, imgsz: int):
        o = req.options or {}
        ex: Dict = o.get("exemplars") or {}
        key = "vp|" + entry.name + "|" + str(o.get("exemplar_key") or "") + "|" + ",".join(str(c.id) for c in req.classes)
        pe = _recall(key)
        if pe is not None:
            return pe
        missing = [c.name for c in req.classes if not ex.get(c.id) and not ex.get(str(c.id))]
        if missing:
            raise ProviderError("Exemplar prompts need labeled examples of every class; none for: " + ", ".join(missing[:6]),
                                "invalid")
        import numpy as np
        import torch
        from ultralytics.models.yolo.yoloe import YOLOEVPDetectPredictor, YOLOEVPSegPredictor
        seg = getattr(entry.obj, "task", "segment") == "segment"
        cls_ = YOLOEVPSegPredictor if seg else YOLOEVPDetectPredictor
        ckeys = o.get("exemplar_class_keys") or {}
        protos = []
        with entry.lock:
            vp = entry.state.get("vp_predictor")
            for c in req.classes:
                ck = ckeys.get(c.id) or ckeys.get(str(c.id))
                ckey = f"vpc|{entry.name}|{ck}" if ck else None
                hit = _recall(ckey) if ckey else None
                if hit is not None:                  # this class's references are unchanged: reuse its prototype
                    protos.append(hit)
                    continue
                if vp is None:
                    vp = cls_(overrides=dict(task="segment" if seg else "detect", mode="predict", imgsz=imgsz, batch=1,
                                             save=False, verbose=False, device=entry.device))
                    vp.setup_model(model=entry.obj.model, verbose=False)
                    entry.state["vp_predictor"] = vp
                refs = ex.get(c.id) or ex.get(str(c.id)) or []
                vpes = []
                for path, boxes in refs:
                    if not boxes:
                        continue
                    vp.set_prompts({"bboxes": np.array(boxes, dtype=np.float32), "cls": np.zeros(len(boxes), dtype=int)})
                    try:
                        local.check_pixels(str(path))      # too large to decode: skipped like an unreadable one
                        vpes.append(vp.get_vpe(str(path)).detach().float().cpu())
                    except Exception:  # noqa: BLE001 — an unreadable reference image is skipped
                        continue
                if not vpes:
                    raise ProviderError(f"Could not read any example image for {c.name}", "invalid")
                proto = torch.nn.functional.normalize(torch.cat(vpes).mean(0), dim=-1)
                if ckey:
                    _remember(ckey, proto)
                protos.append(proto)
        pe = torch.cat(protos)[None]                          # (1, n_classes, D)
        _remember(key, pe)
        return pe

    @staticmethod
    def _args(req: DetectRequest):
        o = req.options or {}
        imgsz = clean_imgsz(o.get("imgsz"))
        iou = clean_float(o.get("iou"), "iou", 0.05, 0.95, 0.7)
        conf = clean_float(req.conf, "conf", 0.0, 1.0, 0.25)
        if not req.classes:
            raise ProviderError("No classes to look for", "invalid")
        return o, imgsz, iou, conf

    def _pf_mapping(self, req: DetectRequest) -> Dict[int, int]:
        names = model_names(req.model, "yoloe")
        try:
            mapping, _ = map_vocab(names, req.classes, (req.options or {}).get("class_map"))
        except ValueError as e:
            raise ProviderError(str(e), "invalid") from None
        if not mapping:
            raise ProviderError(f"None of {req.model}'s vocabulary matches the classes being labeled", "invalid")
        return mapping

    @staticmethod
    def _pf_result(raw: List[dict], mapping: Dict[int, int], ms: int, served: str) -> DetectResult:
        dets, skipped = [], Counter()
        for d in raw:
            did = mapping.get(d["mcls"])
            if did is None:
                skipped[d["label"]] += 1
            else:
                dets.append(Detection(did, d["xc"], d["yc"], d["w"], d["h"], d["conf"], d["label"]))
        return DetectResult(dets, Usage(), ms, [w for w in [skipped_warning(skipped)] if w], served,
                            covered=sorted(set(mapping.values())))

    def _vocabulary(self, req: DetectRequest, imgsz: int):
        """(entry, mode, specs kept, prompt embeddings, vocabulary token, warnings) for text / exemplar prompts."""
        o = req.options or {}
        mode = (o.get("prompt_mode") or "text").lower()
        if mode not in ("text", "exemplars"):
            raise ProviderError("prompt_mode must be text or exemplars", "invalid")
        entry = load_entry(req.model, "yoloe", "vp" if mode == "exemplars" else "text")
        warnings: List[str] = []
        specs = list(req.classes)
        if mode == "text":
            queries, seen, keep = [], set(), []
            for c in specs:
                q = c.query
                if q.lower() in seen:
                    warnings.append(f"{c.name} has the same prompt as another class ({q!r}) and was skipped")
                    continue
                seen.add(q.lower())
                queries.append(q)
                keep.append(c)
            specs = keep
            pe = self._text_pe(entry, queries)
            token = ("text", tuple(queries))
        else:
            pe = self._exemplar_pe(entry, req, imgsz)
            token = ("vp", str(o.get("exemplar_key") or ""), tuple(c.id for c in specs))
        return entry, mode, specs, pe, token, warnings

    @staticmethod
    def _set_vocab(entry, specs, pe, token) -> Dict[str, object]:
        """Make ``specs`` the model's active vocabulary (caller holds entry.lock). Unique internal names keep
        r.names -> class mapping exact even when two prompts render alike."""
        vocab = [f"c{c.id}" for c in specs]
        if entry.state.get("vocab") != token:
            entry.obj.model.set_classes(vocab, pe.to(next(entry.obj.model.parameters()).device))
            if entry.obj.predictor is not None:
                entry.obj.predictor.model.names = entry.obj.model.names
            entry.state["vocab"] = token
        return {v: c for v, c in zip(vocab, specs)}

    @staticmethod
    def _open_result(raw: List[dict], by_vocab, mode: str, o: dict, ms: int, served: str, specs,
                     warnings: List[str]) -> DetectResult:
        # One object, one class: the NMS-free head can score the same box for several prompts.
        if o.get("agnostic", True):
            for d in raw:
                d["cls"] = 0
            raw = geometry.nms(raw, 0.7, class_aware=False)
        dets = []
        for d in raw:
            c = by_vocab.get(d["label"])
            if c is None:
                continue
            dets.append(Detection(c.id, d["xc"], d["yc"], d["w"], d["h"], d["conf"], c.query if mode == "text" else c.name))
        return DetectResult(dets, Usage(), ms, list(warnings), served, covered=[c.id for c in specs])

    def detect(self, req: DetectRequest) -> DetectResult:
        t0 = time.time()
        o, imgsz, iou, conf = self._args(req)
        max_det = int(o.get("max_objects") or 300)
        if req.model.endswith("-pf"):
            mapping = self._pf_mapping(req)
            entry = load_entry(req.model, "yoloe", "pf")
            raw = local.predict(entry, req.image_path, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, tile=o.get("tiles"))
            return self._pf_result(raw, mapping, int((time.time() - t0) * 1000), f"{req.model} on {entry.device}")
        entry, mode, specs, pe, token, warnings = self._vocabulary(req, imgsz)
        with entry.lock:
            by_vocab = self._set_vocab(entry, specs, pe, token)
            raw = local.predict(entry, req.image_path, conf=conf, iou=iou, imgsz=imgsz, max_det=max_det, tile=o.get("tiles"))
        return self._open_result(raw, by_vocab, mode, o, int((time.time() - t0) * 1000),
                                 f"{req.model} ({'text prompts' if mode == 'text' else 'exemplars'}) on {entry.device}",
                                 specs, warnings)

    def detect_many(self, reqs: List[DetectRequest]) -> List[object]:
        """One forward pass for the whole chunk; the vocabulary is set once (requests share model, classes, options)."""
        if not reqs:
            return []
        r0 = reqs[0]
        if (r0.options or {}).get("tiles") or not same_config(reqs):
            return super().detect_many(reqs)
        t0 = time.time()
        o, imgsz, iou, conf = self._args(r0)
        max_det = int(o.get("max_objects") or 300)
        pf = r0.model.endswith("-pf")
        if pf:
            mapping = self._pf_mapping(r0)
            entry = load_entry(r0.model, "yoloe", "pf")
        else:
            entry, mode, specs, pe, token, warnings = self._vocabulary(r0, imgsz)
        arrays = local.decode_many([r.image_path for r in reqs])
        ok = [k for k, a in enumerate(arrays) if not isinstance(a, Exception)]
        with entry.lock:
            if not pf:
                by_vocab = self._set_vocab(entry, specs, pe, token)
            raws = local.predict_arrays(entry, [arrays[k] for k in ok], conf=conf, iou=iou, imgsz=imgsz, max_det=max_det)
        ms = int((time.time() - t0) * 1000 / max(1, len(ok)))
        out: List[object] = [(a if isinstance(a, ProviderError) else ProviderError(f"Could not read the image: {a}", "bad_output"))
                             if isinstance(a, Exception)
                             else None for a in arrays]
        for k, raw in zip(ok, raws):
            if pf:
                out[k] = self._pf_result(raw, mapping, ms, f"{r0.model} on {entry.device}")
            else:
                out[k] = self._open_result(raw, by_vocab, mode, o, ms, f"{r0.model} ({'text prompts' if mode == 'text' else 'exemplars'}) on {entry.device}",
                                           specs, warnings)
        return out


PROVIDER = YoloeProvider()
