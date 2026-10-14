"""Orchestration: one image in, dataset-class proposals out — for any provider, single- or two-stage.

Every caller (propose, preview, batch, audit, benchmark) goes through :func:`detect_image`, so validation, class
specs from class cards, exemplar prototypes, clamping, conf floors and duplicate suppression behave identically
everywhere. Nothing here writes label files or the database.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import db
from . import classes as C
from . import geometry, registry
from .base import (ClassSpec, ClassifyRequest, Crop, DetectRequest, Provider, ProviderError, Usage)
from .base import carry_usage  # ai-cloud hook: a failing call still books what was already billed
from .secrets import scrub

MODEL_ID_RE = re.compile(r"^[\w.:/@+\-]{1,160}$")
PSEUDO_BASE = 100000            # two-stage source classes get ids far above any dataset class
MAX_CROPS_PER_CALL = 20


# ------------------------------------------------------------------ validation helpers

def as_conf(v: Any, default: float = 0.25, name: str = "conf") -> float:
    if v is None or v == "":
        return default
    if isinstance(v, bool):
        raise ProviderError(f"{name} must be a number between 0 and 1", "invalid")
    try:
        f = float(v)
    except (TypeError, ValueError):
        raise ProviderError(f"{name} must be a number between 0 and 1", "invalid") from None
    if not math.isfinite(f) or not 0.0 <= f <= 1.0:
        raise ProviderError(f"{name} must be a number between 0 and 1", "invalid")
    return f


def opt_int(options: dict, name: str, default: int, lo: int, hi: int) -> int:
    v = options.get(name)
    if v in (None, ""):
        return default
    try:
        if isinstance(v, bool):
            raise ValueError
        n = int(v)
    except (TypeError, ValueError):
        raise ProviderError(f"options.{name} must be an integer", "invalid") from None
    if not lo <= n <= hi:
        raise ProviderError(f"options.{name} must be between {lo} and {hi}", "invalid")
    return n


LOCAL_TILE_PROVIDERS = ("yolo", "yoloe")      # ultralytics: tiles = a tile SIZE in pixels


def tiles_option(provider: str, tiles: Any) -> Any:
    """``options.tiles`` validated for the provider family (raises ProviderError):

    * local YOLO / YOLOE — ``true`` / ``"auto"`` (tiles of the model's input size) or ``n`` = tile size in pixels
      (64-4096);
    * cloud VLMs and OpenAI-compatible servers — ``true`` / ``"auto"`` (a grid only when the image is much larger
      than what the model sees), ``n`` = an n×n grid (1-4) or ``[cols, rows]`` (each 1-4).

    ``None`` / ``false`` / ``0`` mean no tiling. Returns the normalised value."""
    if tiles is None or tiles is False or (isinstance(tiles, (int, float)) and not isinstance(tiles, bool) and tiles == 0):
        return None
    if tiles is True or (isinstance(tiles, str) and tiles.strip().lower() == "auto"):
        return True if provider in LOCAL_TILE_PROVIDERS else "auto"
    if provider in LOCAL_TILE_PROVIDERS:
        n = tiles.strip() if isinstance(tiles, str) else tiles
        if isinstance(n, str) and re.fullmatch(r"[0-9]{1,5}", n):
            n = int(n)
        if not isinstance(n, int) or isinstance(n, bool) or not 64 <= n <= 4096:
            raise ProviderError("options.tiles for YOLO / YOLOE is true, \"auto\" or a tile size in pixels "
                                "(64-4096)", "invalid")
        return n
    grid = tiles if isinstance(tiles, list) else [tiles, tiles]
    if len(grid) != 2 or not all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 4 for v in grid):
        raise ProviderError("options.tiles for this provider is true, \"auto\", n (an n×n grid, 1-4) or [cols, rows] "
                            "(each 1-4)", "invalid")
    return [int(grid[0]), int(grid[1])] if isinstance(tiles, list) else int(tiles)


_JPEG_EXTS = (".jpg", ".jpeg", ".jpe", ".jfif")


def pixel_error(img_row: dict, provider: str = "") -> Optional[ProviderError]:
    """ProviderError 'too_large' when the image declares more pixels than FOVEA_MAX_IMAGE_PIXELS allows to decode
    (the indexed size; the file header when it was never indexed), else None. Checked before any model sees the
    image: local decoders (cv2 / ultralytics / PIL) would decode such a file whole. Cloud providers decode JPEGs at
    a reduced scale and check the cap themselves, so a large JPEG is left to them."""
    from ..config import MAX_IMAGE_PIXELS
    from . import local
    path = str(img_row.get("abs_path") or "")
    if provider and registry.is_cloud(provider) and path.lower().endswith(_JPEG_EXTS):
        return None
    w, h = img_row.get("width"), img_row.get("height")
    if w and h:
        return None if int(w) * int(h) <= MAX_IMAGE_PIXELS else local.too_large_error(os.path.basename(path), int(w), int(h))
    try:
        local.check_pixels(path)
    except ProviderError as e:
        return e
    return None


def provider_for(pid: str, need: str = "detect") -> Provider:
    if not isinstance(pid, str) or not pid:
        raise ProviderError("Choose a provider", "invalid")
    try:
        p = registry.get(pid)
    except KeyError:
        raise ProviderError(f"Unknown provider: {pid}", "invalid") from None
    st = registry.safe_status(p)
    if not st["available"]:
        raise ProviderError(f"{p.name or pid} is not available: {st.get('reason') or 'unknown reason'}", "unavailable")
    if not (getattr(p, "capabilities", {}) or {}).get(need):
        raise ProviderError(f"{p.name or pid} cannot {need} — pick another provider", "invalid")
    return p


def check_model(pid: str, model: Optional[str]) -> str:
    """Local providers accept only models that exist in their list (no paths — finding C12); cloud ids are
    syntax-checked (live listings change faster than any static list)."""
    model = (model or "").strip()
    if not model:
        model = registry.default_model(pid) or ""
        if not model:
            raise ProviderError("Choose a model", "invalid")
    if not MODEL_ID_RE.match(model) or ".." in model:
        raise ProviderError(f"Unknown model: {model!r}", "invalid")
    if not registry.is_cloud(pid):
        known = {m.get("id"): m for m in registry.models(pid)}
        if model not in known:
            raise ProviderError(f"Unknown model for {pid}: {model}", "invalid")
        if known[model].get("usable") is False:
            raise ProviderError(f"{model} cannot draw boxes: {known[model].get('notes', '')}", "invalid")
    return model


def specs_for(ds_row: dict, cls_ids: Optional[Sequence[int]]) -> List[ClassSpec]:
    try:
        specs = C.class_specs(ds_row, cls_ids if cls_ids else None)
    except ValueError as e:
        raise ProviderError(str(e), "invalid") from None
    if not specs:
        raise ProviderError("This dataset has no classes yet — add class names in Overview first", "invalid")
    return specs


def _refs_key(refs: Sequence[tuple]) -> str:
    """Content key of one class's reference set: which images (and their file versions) with which boxes."""
    blob = json.dumps([[int(iid), round(float(mt or 0), 3), [[round(float(v), 1) for v in b] for b in boxes]]
                       for iid, mt, boxes in refs], separators=(",", ":"))
    return hashlib.sha1(blob.encode()).hexdigest()[:20]


def build_exemplars(ds_row: dict, specs: Sequence[ClassSpec], per_class: int = 8,
                    exclude: Sequence[int] = ()) -> Tuple[Dict[int, list], str, Dict[int, str]]:
    """Up to ``per_class`` labeled images per class (deterministic spread over the dataset), with that class's boxes in
    pixels. ``exclude`` keeps evaluated images out of their own prompts (audit / benchmark).

    Returns (exemplars, key, class keys). The keys hash the reference sets actually picked — image ids, image file
    versions and boxes — so the provider's prototypes are rebuilt only when a reference changed (a class's own key:
    only that class), not after every label save anywhere in the dataset (finding: 3-4 s per Propose)."""
    from ..config import MAX_IMAGE_PIXELS as max_px     # never a prototype from an image too large to decode
    per_class = max(1, min(int(per_class or 8), 32))
    excl = set(int(i) for i in exclude or ())
    out: Dict[int, list] = {}
    class_keys: Dict[int, str] = {}
    for s in specs:
        rows = db.query(
            "SELECT b.image_id, b.xc, b.yc, b.w, b.h, i.abs_path, i.width, i.height, i.mtime FROM boxes b "
            "JOIN images i ON i.id=b.image_id WHERE b.dataset_id=? AND b.cls=? AND b.w*b.h >= 0.0015 "
            "ORDER BY (b.image_id * 2654435761) % 1000003, b.id LIMIT 600", (ds_row["id"], s.id))
        by_img: Dict[int, dict] = {}
        for r in rows:
            if r["image_id"] in excl or not r["width"] or not r["height"] or r["width"] * r["height"] > max_px:
                continue
            e = by_img.get(r["image_id"])
            if e is None:
                if len(by_img) >= per_class:
                    continue
                if not os.path.isfile(r["abs_path"]):
                    continue
                e = by_img[r["image_id"]] = {"path": r["abs_path"], "boxes": [], "mtime": r["mtime"]}
            if len(e["boxes"]) < 6:
                W, H = r["width"], r["height"]
                x0, y0, x1, y1 = geometry.xyxy((r["xc"], r["yc"], r["w"], r["h"]))
                e["boxes"].append([max(0.0, x0 * W), max(0.0, y0 * H), min(W, x1 * W), min(H, y1 * H)])
        if by_img:
            out[s.id] = [(e["path"], e["boxes"]) for e in by_img.values()]
            class_keys[s.id] = _refs_key([(iid, e["mtime"], e["boxes"]) for iid, e in by_img.items()])
    key = ds_row["id"] + ":" + hashlib.sha1(json.dumps(sorted(class_keys.items())).encode()).hexdigest()[:20]
    return out, key, class_keys


_ex_cache: Dict[tuple, Tuple[float, Dict[int, list], str, Dict[int, str]]] = {}


def exemplars_cached(ds_row: dict, specs: Sequence[ClassSpec], per_class, exclude: Sequence[int] = ()):
    """build_exemplars memoised for 30 s: a batch reuses one prototype set instead of re-querying per image (the
    prototypes themselves are cached by the provider under the returned keys). Returns (exemplars, key, class keys)."""
    try:
        per = int(per_class or 8)
    except (TypeError, ValueError):
        raise ProviderError("exemplars_per_class must be an integer", "invalid") from None
    k = (ds_row["id"], tuple(s.id for s in specs), per, hash(tuple(sorted(int(i) for i in exclude or ()))))
    hit = _ex_cache.get(k)
    if hit and time.time() - hit[0] < 30:
        return hit[1], hit[2], hit[3]
    ex, key, class_keys = build_exemplars(ds_row, specs, per, exclude)
    if len(_ex_cache) > 32:
        _ex_cache.clear()
    _ex_cache[k] = (time.time(), ex, key, class_keys)
    return ex, key, class_keys


# ------------------------------------------------------------------ detection

def _postprocess(dets, specs: Sequence[ClassSpec], conf: float, max_objects: int, kind: str = "detect") -> Tuple[List[dict], int]:
    allowed = {s.id for s in specs}
    out, dropped = [], 0
    for d in dets:
        def get(k, d=d):
            return d.get(k) if isinstance(d, dict) else getattr(d, k, None)
        cls = int(get("cls") if get("cls") is not None else -1)
        if cls not in allowed:
            dropped += 1
            continue
        b = geometry.clamp_box(float(get("xc")), float(get("yc")), float(get("w")), float(get("h")))
        if b is None:
            dropped += 1
            continue
        c = get("conf")
        c = None if c is None else round(float(c), 4)
        if c is not None and c < conf:
            continue
        out.append({"cls": cls, "xc": b[0], "yc": b[1], "w": b[2], "h": b[3], "conf": c,
                    "label": str(get("label") or "")[:200], "kind": kind, "reason": str(get("reason") or "")})
    out = geometry.nms(out, 0.8, class_aware=True)          # many-to-one class maps can stack near-identical boxes
    out.sort(key=lambda p: -(p["conf"] if p["conf"] is not None else 1.0))
    return out[:max_objects], dropped


class DetectPlan:
    """A validated single-stage detect configuration, shared by every image of a run: the provider, model, class
    specs (with exemplar prototypes when asked), confidence floor and options."""

    def __init__(self, p: Provider, model: str, specs: List[ClassSpec], conf: float, max_objects: int, options: dict,
                 warnings: List[str]):
        self.provider = p
        self.model = model
        self.specs = specs
        self.conf = conf
        self.max_objects = max_objects
        self.options = options
        self.warnings = warnings

    def request(self, img_row: dict) -> DetectRequest:
        return DetectRequest(image_path=img_row["abs_path"], classes=self.specs, model=self.model, conf=self.conf,
                             options=self.options)

    def finish(self, res, t0: float) -> dict:
        """Provider result → ``{preds, usage, latency_ms, warnings, served_by, model, covered}``."""
        preds, dropped = _postprocess(res.detections, self.specs, self.conf, self.max_objects)
        warnings = list(self.warnings) + [scrub(w) for w in (res.warnings or [])]
        if dropped:
            warnings.append(f"{dropped} box{'es' if dropped != 1 else ''} outside the image or the requested classes dropped")
        covered = getattr(res, "covered", None)
        ids = {s.id for s in self.specs}
        covered = [s.id for s in self.specs] if covered is None else [c for c in covered if c in ids]
        return {"preds": preds, "usage": (res.usage or Usage()).tagged(self.provider.id),
                "latency_ms": res.latency_ms or int((time.time() - t0) * 1000),
                "warnings": warnings, "served_by": res.served_by or self.model, "model": self.model, "covered": covered}


def plan_detect(ds_row: dict, provider: str, model: str, conf: Any = None, cls_ids: Optional[Sequence[int]] = None,
                options: Optional[dict] = None, exclude_exemplars: Sequence[int] = ()) -> DetectPlan:
    """Validate a single-stage detect configuration once (raises ProviderError) — see :class:`DetectPlan`."""
    options = dict(options or {})
    if options.get("pipeline") not in (None, "", "single"):
        raise ProviderError("options.pipeline must be 'single' or 'two_stage'", "invalid")
    p = provider_for(provider, "detect")
    model = check_model(provider, model)
    specs = specs_for(ds_row, cls_ids)
    conf_v = as_conf(conf, 0.25 if not registry.is_cloud(provider) else 0.0)
    extra_warn: List[str] = []
    max_objects = opt_int(options, "max_objects", 300, 1, 1000)
    if "tiles" in options:
        options["tiles"] = tiles_option(provider, options.get("tiles"))
    if provider == "yoloe" and (options.get("prompt_mode") or "text") == "exemplars":
        ex, key, class_keys = exemplars_cached(ds_row, specs, options.get("exemplars_per_class", 8), exclude_exemplars)
        options["exemplars"], options["exemplar_key"], options["exemplar_class_keys"] = ex, key, class_keys
        missing = [s.name for s in specs if s.id not in ex]
        if missing and cls_ids:
            raise ProviderError("Exemplar prompts need labeled examples of each class; none yet for: "
                                + ", ".join(missing[:6]) + ". Label a few, or pick only classes that have labels.", "invalid")
        if missing:                      # "all classes": look only for the ones that have examples
            specs = [s for s in specs if s.id in ex]
            if not specs:
                raise ProviderError("Exemplar prompts need a few labeled boxes per class — this dataset has none yet",
                                    "invalid")
            extra_warn = [f"No labeled examples for {', '.join(missing[:4])}{'…' if len(missing) > 4 else ''} — "
                          "not looked for"]
    return DetectPlan(p, model, specs, conf_v, max_objects, options, extra_warn)


def detect_image(ds_row: dict, img_row: dict, provider: str, model: str, conf: Any = None,
                 cls_ids: Optional[Sequence[int]] = None, options: Optional[dict] = None,
                 exclude_exemplars: Sequence[int] = ()) -> dict:
    """Run one provider (or the two-stage pipeline) on one image. Returns
    ``{preds, usage: Usage, latency_ms, warnings, served_by, model}``; raises ProviderError."""
    options = dict(options or {})
    if not os.path.isfile(img_row["abs_path"]):
        raise ProviderError("The image file is missing on disk", "invalid")
    two = (options.get("pipeline") or "") == "two_stage"
    det = options.get("detector") if two and isinstance(options.get("detector"), dict) else {}
    err = pixel_error(img_row, str(det.get("provider") or "") if two else provider)
    if err is not None:
        raise err
    if two:
        return two_stage(ds_row, img_row, cls_ids, options)
    plan = plan_detect(ds_row, provider, model, conf, cls_ids, options, exclude_exemplars)
    t0 = time.time()
    return plan.finish(plan.provider.detect(plan.request(img_row)), t0)


def batch_capable(provider: str, options: Optional[dict] = None) -> bool:
    """Can this configuration infer several images per model call (local ultralytics, no tiling, single stage)?"""
    o = options or {}
    if (o.get("pipeline") or "") == "two_stage" or o.get("tiles"):
        return False
    try:
        p = registry.get(provider)
    except KeyError:
        return False
    return bool((getattr(p, "capabilities", {}) or {}).get("batch_infer"))


def detect_images(ds_row: dict, img_rows: Sequence[dict], provider: str, model: str, conf: Any = None,
                  cls_ids: Optional[Sequence[int]] = None, options: Optional[dict] = None,
                  exclude_exemplars: Sequence[int] = ()) -> List[Any]:
    """Several images in one model call where the provider batches (``Provider.detect_many``). Returns one entry per
    image, in order: the :func:`detect_image` dict, or the ProviderError of that image. A configuration error
    (bad model, no classes …) raises for the whole call."""
    options = dict(options or {})
    if (options.get("pipeline") or "") == "two_stage":
        out: List[Any] = []
        for img in img_rows:
            try:
                out.append(detect_image(ds_row, img, provider, model, conf, cls_ids, options, exclude_exemplars))
            except ProviderError as e:
                out.append(e)
        return out
    plan = plan_detect(ds_row, provider, model, conf, cls_ids, options, exclude_exemplars)
    slots: List[Any] = [None] * len(img_rows)
    reqs, where = [], []
    for k, img in enumerate(img_rows):
        if not os.path.isfile(img["abs_path"]):
            slots[k] = ProviderError("The image file is missing on disk", "invalid")
            continue
        err = pixel_error(img, provider)
        if err is not None:                 # never handed to the chunk's decoders
            slots[k] = err
            continue
        reqs.append(plan.request(img))
        where.append(k)
    if reqs:
        t0 = time.time()
        many = getattr(plan.provider, "detect_many", None)
        results = many(reqs) if callable(many) else [_safe_detect(plan.provider, r) for r in reqs]
        share = max(1, len(reqs))
        for k, res in zip(where, results):
            if isinstance(res, Exception):
                slots[k] = res if isinstance(res, ProviderError) else ProviderError(
                    scrub(f"{type(res).__name__}: {res}"), "bad_output")
            else:
                out1 = plan.finish(res, t0)
                out1["latency_ms"] = res.latency_ms or int((time.time() - t0) * 1000 / share)
                slots[k] = out1
    return slots


def _safe_detect(p: Provider, req: DetectRequest):
    try:
        return p.detect(req)
    except ProviderError as e:
        return e


def _source_specs(det_provider: str, det_model: str, source: Sequence) -> Tuple[List[ClassSpec], Dict]:
    """Two-stage detector vocabulary: model class names / ids (closed vocab) or free-text prompts."""
    if not source:
        source = ["person"]
    specs: List[ClassSpec] = []
    explicit: Dict = {"*": "skip"}
    if det_provider in ("yolo",) or (det_provider == "yoloe" and det_model.endswith("-pf")):
        from .providers.yolo import model_names
        names = model_names(det_model, "yolo" if det_provider == "yolo" else "yoloe")
        by_name = {C.norm_name(v): k for k, v in names.items()}
        for k, s in enumerate(source):
            mid = None
            if isinstance(s, int) or (isinstance(s, str) and s.strip().isdigit()):
                mid = int(s)
            else:
                mid = by_name.get(C.norm_name(str(s)))
                if mid is None:
                    canon = C.canonical(str(s))
                    mid = next((i for i, n in names.items() if C.canonical(n) == canon), None)
            if mid is None or mid not in names:
                raise ProviderError(f"{det_model} has no class {s!r}", "invalid")
            specs.append(ClassSpec(id=PSEUDO_BASE + k, name=names[mid]))
            explicit[mid] = PSEUDO_BASE + k
        return specs, explicit
    for k, s in enumerate(source):
        txt = str(s).strip()
        if not txt or len(txt) > 200:
            raise ProviderError("Source classes must be short names, e.g. person", "invalid")
        specs.append(ClassSpec(id=PSEUDO_BASE + k, name=txt, prompt=txt))
    return specs, {}


def two_stage(ds_row: dict, img_row: dict, cls_ids: Optional[Sequence[int]], options: dict) -> dict:
    """Detector proposes boxes for source classes (e.g. COCO person); a classifier assigns each crop a dataset class
    (e.g. fallen / standing) or rejects it. Box geometry and conf come from the detector; the classifier's verdict and
    short reason go into ``reason``."""
    det = options.get("detector") or {}
    clf = options.get("classifier") or {}
    if not isinstance(det, dict) or not isinstance(clf, dict):
        raise ProviderError("two_stage needs options.detector and options.classifier objects", "invalid")
    dp = provider_for(det.get("provider") or "", "detect")
    dmodel = check_model(dp.id, det.get("model"))
    cp = provider_for(clf.get("provider") or "", "classify")
    cmodel = check_model(cp.id, clf.get("model"))
    specs = specs_for(ds_row, cls_ids)
    dconf = as_conf(det.get("conf"), 0.25, "detector conf")
    source = det.get("source_classes") or det.get("classes") or ["person"]
    if not isinstance(source, list) or len(source) > 20:
        raise ProviderError("detector.source_classes must be a list of up to 20 names", "invalid")
    sspecs, explicit = _source_specs(dp.id, dmodel, source)
    dopts = det.get("options") or {}
    if not isinstance(dopts, dict):
        raise ProviderError("detector.options must be an object", "invalid")
    dopts = {k: v for k, v in dopts.items() if k not in ("class_map", "pipeline")}
    if "tiles" in dopts:
        dopts["tiles"] = tiles_option(dp.id, dopts.get("tiles"))
    if explicit:
        dopts["class_map"] = explicit
    t0 = time.time()
    dres = dp.detect(DetectRequest(image_path=img_row["abs_path"], classes=sspecs, model=dmodel, conf=dconf, options=dopts))
    boxes, _ = _postprocess(dres.detections, sspecs, dconf, opt_int(options, "max_objects", 100, 1, 1000))
    usage = Usage().add((dres.usage or Usage()).tagged(dp.id))
    warnings = [scrub(w) for w in (dres.warnings or []) if "skipped" not in w]
    preds: List[dict] = []
    none_n = unc_n = 0
    by_id = {s.id: s for s in specs}
    copts = clf.get("options") or {}
    if not isinstance(copts, dict):
        raise ProviderError("classifier.options must be an object", "invalid")
    copts = dict(copts)
    for k in range(0, len(boxes), MAX_CROPS_PER_CALL):
        chunk = boxes[k:k + MAX_CROPS_PER_CALL]
        crops = [Crop(index=k + i, xc=b["xc"], yc=b["yc"], w=b["w"], h=b["h"], hint=b["label"]) for i, b in enumerate(chunk)]
        try:
            cres = cp.classify(ClassifyRequest(image_path=img_row["abs_path"], crops=crops, classes=specs, model=cmodel,
                                               options=copts))
        except Exception as e:     # ai-cloud hook: the detector and earlier chunks were billed already
            carry_usage(e, usage)
            raise
        usage.add((cres.usage or Usage()).tagged(cp.id))
        warnings += [scrub(w) for w in (cres.warnings or [])]
        verdicts = {it.index: it for it in cres.items}
        for i, b in enumerate(chunk):
            it = verdicts.get(k + i)
            if it is None or it.verdict == "uncertain" or (it.verdict == "label" and it.cls is None):
                unc_n += 1
                continue
            if it.verdict == "none" or it.cls not in by_id:
                none_n += 1
                continue
            cf = it.confidence
            why = (it.reason or "").strip()
            if cf is not None:
                why = (why + " · " if why else "") + f"classifier {cf:.2f}"
            preds.append({"cls": it.cls, "xc": b["xc"], "yc": b["yc"], "w": b["w"], "h": b["h"], "conf": b["conf"],
                          "label": f"{b['label']} → {by_id[it.cls].name}", "kind": "detect", "reason": why[:500]})
    if none_n:
        warnings.append(f"{none_n} detected box{'es' if none_n != 1 else ''} rejected by the classifier")
    if unc_n:
        warnings.append(f"{unc_n} box{'es' if unc_n != 1 else ''} the classifier was unsure about were left out")
    return {"preds": preds, "usage": usage, "latency_ms": int((time.time() - t0) * 1000), "warnings": warnings,
            "served_by": f"{dres.served_by or dmodel} + {cmodel}", "model": f"{dmodel}+{cmodel}",
            "detected": len(boxes), "covered": [s.id for s in specs]}


# ------------------------------------------------------------------ classify / refine

def classify_boxes(ds_row: dict, img_row: dict, boxes: Sequence, provider: str, model: str,
                   cls_ids: Optional[Sequence[int]] = None, options: Optional[dict] = None) -> dict:
    p = provider_for(provider, "classify")
    model = check_model(provider, model)
    specs = specs_for(ds_row, cls_ids)
    names = C.dataset_classes(ds_row)
    crops: List[Crop] = []
    if not isinstance(boxes, list) or not boxes:
        raise ProviderError("boxes must be a non-empty list of [cls, xc, yc, w, h]", "invalid")
    if len(boxes) > 200:
        raise ProviderError("At most 200 boxes per request", "invalid")
    for i, b in enumerate(boxes):
        try:
            if not isinstance(b, list):
                raise TypeError
            cls = None if b[0] is None else int(b[0])
            xc, yc, w, h = (float(v) for v in b[1:5])
        except (TypeError, ValueError, IndexError, OverflowError):
            raise ProviderError(f"Box {i} is not [cls, xc, yc, w, h]", "invalid") from None
        if geometry.clamp_box(xc, yc, w, h) is None:
            raise ProviderError(f"Box {i} is empty or outside the image", "invalid")
        hint = names[cls] if cls is not None and 0 <= cls < len(names) else ""
        crops.append(Crop(index=i, xc=xc, yc=yc, w=w, h=h, hint=hint))
    err = pixel_error(img_row, provider)
    if err is not None:
        raise err
    usage, items, warnings, t0 = Usage(), [], [], time.time()
    for k in range(0, len(crops), MAX_CROPS_PER_CALL):
        try:
            res = p.classify(ClassifyRequest(image_path=img_row["abs_path"], crops=crops[k:k + MAX_CROPS_PER_CALL],
                                             classes=specs, model=model, options=dict(options or {})))
        except Exception as e:     # ai-cloud hook: earlier chunks were billed already
            carry_usage(e, usage)
            raise
        usage.add((res.usage or Usage()).tagged(p.id))
        warnings += [scrub(w) for w in (res.warnings or [])]
        items += res.items
    by_index = {it.index: it for it in items}
    out = []
    for i, b in enumerate(boxes):
        it = by_index.get(i)
        given = None if b[0] is None else int(b[0])
        if it is None:
            out.append({"index": i, "cls": None, "verdict": "uncertain", "confidence": None, "reason": "no answer",
                        "agrees": None})
            continue
        agrees = None if it.verdict != "label" or given is None else (it.cls == given)
        out.append({"index": i, "cls": it.cls, "verdict": it.verdict, "confidence": it.confidence,
                    "reason": scrub(it.reason or "")[:500], "agrees": agrees})
    return {"items": out, "usage": usage, "latency_ms": int((time.time() - t0) * 1000), "warnings": warnings,
            "model": model}


def refine(img_row: dict, box: Optional[Sequence[float]], points: Optional[Sequence], model: str = "") -> dict:
    if box is not None and not isinstance(box, list):
        raise ProviderError("box must be [xc, yc, w, h] (normalised)", "invalid")
    if points is not None and not isinstance(points, list):
        raise ProviderError("points must be [[x, y, 1|0], …] (normalised)", "invalid")
    if box is not None:
        try:
            bx = [float(v) for v in box][:4]
        except (TypeError, ValueError):
            raise ProviderError("box must be [xc, yc, w, h] (normalised)", "invalid") from None
        if len(bx) != 4 or not all(math.isfinite(v) for v in bx) or geometry.clamp_box(*bx) is None:
            raise ProviderError("box must be [xc, yc, w, h] inside the image", "invalid")
        box = bx
    pts = []
    for q in points or []:
        if not isinstance(q, list):
            raise ProviderError("points must be [[x, y, 1|0], …] (normalised)", "invalid")
        try:
            x, y = float(q[0]), float(q[1])
            lab = int(q[2]) if len(q) > 2 else 1
        except (TypeError, ValueError, IndexError):
            raise ProviderError("points must be [[x, y, 1|0], …] (normalised)", "invalid") from None
        if not (0 <= x <= 1 and 0 <= y <= 1) or lab not in (0, 1):
            raise ProviderError("points must lie inside the image and carry label 1 or 0", "invalid")
        pts.append([x, y, lab])
    if box is None and not any(p_[2] == 1 for p_ in pts):
        raise ProviderError("Give a box or at least one positive point", "invalid")
    if len(pts) > 20:
        raise ProviderError("At most 20 points", "invalid")
    p = provider_for("sam", "refine")      # after the shape checks: a malformed request is 422 even without SAM
    if model and model not in {m.get("id") for m in registry.models("sam")}:
        raise ProviderError(f"Unknown SAM model: {model}", "invalid")
    if not os.path.isfile(img_row["abs_path"]):
        raise ProviderError("The image file is missing on disk", "invalid")
    err = pixel_error(img_row, "sam")
    if err is not None:
        raise err
    res = p.refine(img_row["abs_path"], box, pts or None, model or "")
    if box is not None:
        res["iou_with_input"] = round(geometry.iou4(box, res["box"]), 4)
    return res
