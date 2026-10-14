"""Text helpers built on ``Provider.generate_json``: naming unnamed classes from example crops, and compiling a
natural-language search ("넘어진 사람이 있는 val 이미지", "blurry frames with more than 3 boxes") into Explore filters.

Both work with any provider whose ``capabilities["generate"]`` is set (Claude, Gemini, OpenAI, an OpenAI-compatible
server, the mock). Local-only setups get ``ProviderError(code='unavailable')`` (the API answers 400). Every answer is
validated here: names are made snake_case and unique, filters keep only known keys and values that exist in this
dataset — a model can suggest, it can never inject a filter the app does not understand.
"""
from __future__ import annotations

import io
import json
import math
import os
import re
import statistics
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import db
from . import classes as C
from . import pipeline, registry
from .base import Provider, ProviderError, Usage
from .base import billed_usage  # ai-cloud hook: failed calls that were billed
from .secrets import scrub

GENERATE_ORDER = ("anthropic", "gemini", "openai", "openai_compat", "mock")
CONF_WORDS = {"low": 0.35, "medium": 0.65, "high": 0.9}
MAX_IMAGES_PER_REQUEST = 20          # providers accept at most 20 images in one generate_json call
CROP_EDGE = 384
CROP_PAD = 0.2


# ------------------------------------------------------------------ provider choice

def generators() -> List[str]:
    """Providers that can answer free-form questions here, in preference order (cloud first, then a local server)."""
    out = []
    for pid in GENERATE_ORDER:
        if pid not in registry.ids():
            continue
        p = registry.get(pid)
        if not (getattr(p, "capabilities", {}) or {}).get("generate"):
            continue
        if registry.safe_status(p).get("available"):
            out.append(pid)
    return out


def pick(provider: Optional[str], model: Optional[str], what: str, cheapest: bool = False) -> Tuple[Provider, str]:
    """(provider, model) for a generate_json task. No provider given → the first configured one."""
    if provider:
        try:
            p = registry.get(provider)
        except KeyError:
            raise ProviderError(f"Unknown provider: {provider}", "invalid") from None
        if not (getattr(p, "capabilities", {}) or {}).get("generate"):
            raise ProviderError(f"{p.name or provider} cannot {what} — it only draws boxes. Use Claude, Gemini, OpenAI or "
                                "an OpenAI-compatible server (add one in Settings).", "unavailable")
        st = registry.safe_status(p)
        if not st.get("available"):
            raise ProviderError(f"{p.name or provider} is not available: {st.get('reason') or 'unknown reason'}",
                                "unavailable")
    else:
        avail = generators()
        if not avail:
            raise ProviderError(f"To {what}, add a cloud AI provider (Claude, Gemini or OpenAI) or an OpenAI-compatible "
                                "server in Settings — the local detectors cannot write text.", "unavailable")
        p = registry.get(avail[0])
    model = (model or "").strip() if isinstance(model, str) else ""
    if not model:
        model = cheapest_model(p.id) if cheapest else (registry.default_model(p.id) or "")
    return p, pipeline.check_model(p.id, model)


def cheapest_model(pid: str) -> str:
    """The provider's cheapest priced model (a short text answer: ~1.5k tokens in, ~200 out), else its default."""
    best, best_cost = None, None
    try:
        lst = registry.models(pid)
    except Exception:  # noqa: BLE001
        lst = []
    for m in lst:
        pr = m.get("price") if isinstance(m.get("price"), dict) else None
        if not pr or pr.get("in") is None or m.get("available") is False:
            continue
        cost = float(pr["in"]) * 1500 + float(pr.get("out") or pr["in"]) * 200
        if best_cost is None or cost < best_cost:
            best, best_cost = m.get("id"), cost
    return best or registry.default_model(pid) or ""


def _price(pid: str, model: str) -> Optional[dict]:
    try:
        for m in registry.models(pid):
            if m.get("id") == model and isinstance(m.get("price"), dict):
                return m["price"]
    except Exception:  # noqa: BLE001
        pass
    try:
        from . import vlm
        return vlm.model_price(pid, model)
    except Exception:  # noqa: BLE001
        return None


def _usage_dict(u: Usage) -> dict:
    return {"input_tokens": int(u.input_tokens), "output_tokens": int(u.output_tokens), "cost_usd": round(float(u.cost_usd), 6)}


def _ledger(ds_id: str, kind: str, provider: str, model: str, usage: Usage, params: dict, aggregate: bool) -> int:
    """Record spend in ai_runs. Many tiny calls (searches) share one row per dataset, provider, model and day."""
    from . import store
    if aggregate:
        day0 = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0).timestamp()
        r = db.query_one("SELECT id FROM ai_runs WHERE dataset_id=? AND kind=? AND provider=? AND model=? AND created_at>=? "
                         "AND status='done' ORDER BY id DESC LIMIT 1", (ds_id, kind, provider, model, day0))
        if r:
            store.add_usage(r["id"], usage, images=0)
            db.execute("UPDATE ai_runs SET boxes=boxes+1 WHERE id=?", (r["id"],))   # boxes = number of questions here
            return int(r["id"])
    rid = store.create_run(ds_id, kind, provider, model, params, status="done")
    store.add_usage(rid, usage, images=0)
    if aggregate:
        db.execute("UPDATE ai_runs SET boxes=1 WHERE id=?", (rid,))
    store.finish_run(rid, "done")
    return rid


# ------------------------------------------------------------------ class naming

def _class_stats(ds_id: str, cls: int) -> dict:
    r = db.query_one("SELECT COUNT(*) n, COUNT(DISTINCT b.image_id) imgs FROM boxes b WHERE b.dataset_id=? AND b.cls=?",
                     (ds_id, cls)) or {}
    shapes = db.query("SELECT b.w*COALESCE(i.width,1) pw, b.h*COALESCE(i.height,1) ph, b.w*b.h a FROM boxes b JOIN images i "
                      "ON i.id=b.image_id WHERE b.dataset_id=? AND b.cls=? ORDER BY (b.id * 2654435761) % 1000003 LIMIT 400",
                      (ds_id, cls))
    ar = [s["pw"] / s["ph"] for s in shapes if s["ph"] and s["pw"]]
    area = [s["a"] for s in shapes if s["a"]]
    return {"boxes": int(r.get("n") or 0), "images": int(r.get("imgs") or 0),
            "aspect": round(statistics.median(ar), 2) if ar else None,
            "area": round(statistics.median(area), 4) if area else None}


def _shape_words(st: dict) -> str:
    a = st.get("aspect")
    if not a:
        return "no boxes"
    shape = "wider than tall" if a >= 1.25 else ("taller than wide" if a <= 0.8 else "about square")
    size = f", about {st['area'] * 100:.1f}% of the image" if st.get("area") else ""
    return f"{shape} (median width/height {a:.2f}){size}"


def pick_examples(ds_id: str, cls: int, n: int) -> List[dict]:
    """Up to ``n`` typical boxes of a class: large and central ones first, from different sequences / folders and
    images, so the model sees the class rather than one scene."""
    from ..config import MAX_IMAGE_PIXELS       # never pick an image too large to decode
    rows = db.query(
        "SELECT b.image_id, b.xc, b.yc, b.w, b.h, i.abs_path, i.rel_path, i.seq, i.width, i.height FROM boxes b "
        "JOIN images i ON i.id=b.image_id WHERE b.dataset_id=? AND b.cls=? AND b.w*b.h >= 0.0015 "
        "AND COALESCE(i.width * i.height, 0) <= ? "
        "ORDER BY (b.w*b.h) * (1.0 - MIN(0.9, ABS(b.xc-0.5) + ABS(b.yc-0.5))) DESC, b.id LIMIT 600",
        (ds_id, cls, MAX_IMAGE_PIXELS))
    if len(rows) < n:
        rows += db.query("SELECT b.image_id, b.xc, b.yc, b.w, b.h, i.abs_path, i.rel_path, i.seq, i.width, i.height "
                         "FROM boxes b JOIN images i ON i.id=b.image_id WHERE b.dataset_id=? AND b.cls=? AND b.w*b.h < 0.0015 "
                         "AND COALESCE(i.width * i.height, 0) <= ? ORDER BY b.w*b.h DESC LIMIT 50",
                         (ds_id, cls, MAX_IMAGE_PIXELS))
    # Skip the very largest 5 % (often truncated or odd frames) when there is plenty to choose from.
    if len(rows) > 60:
        rows = rows[len(rows) // 20:]
    picked: List[dict] = []
    seen_group, seen_img = set(), set()
    for pass_ in (0, 1):
        for r in rows:
            if len(picked) >= n:
                break
            group = r["seq"] or os.path.dirname(r["rel_path"] or "")
            if r["image_id"] in seen_img or (pass_ == 0 and group in seen_group):
                continue
            if not os.path.isfile(r["abs_path"] or ""):
                continue
            seen_img.add(r["image_id"])
            seen_group.add(group)
            picked.append(r)
    return picked


def crop_jpeg(path: str, box: Sequence[float], edge: int = CROP_EDGE, pad: float = CROP_PAD) -> bytes:
    """The box with some context, the object outlined in red, long edge ``edge`` px (small crops are upscaled)."""
    from PIL import Image, ImageDraw, ImageOps
    from ..config import MAX_IMAGE_PIXELS
    from ..core.thumbs import to_display_rgb
    from .local import too_large_error
    try:
        with warnings.catch_warnings():         # checked below, after draft scaling
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            im0 = Image.open(path)
    except Image.DecompressionBombError:
        raise too_large_error(os.path.basename(path)) from None
    with im0:
        im0.draft("RGB", (max(edge * 3, 1024), max(edge * 3, 1024)))
        if im0.width * im0.height > MAX_IMAGE_PIXELS:          # after JPEG draft scaling, like thumbnails
            raise too_large_error(os.path.basename(path), im0.width, im0.height)
        im = to_display_rgb(ImageOps.exif_transpose(im0))
        W, H = im.size
        xc, yc, w, h = (float(v) for v in box[:4])
        bw, bh = w * W, h * H
        px, py = max(12.0, bw * pad), max(12.0, bh * pad)
        x0, y0 = max(0, int(xc * W - bw / 2 - px)), max(0, int(yc * H - bh / 2 - py))
        x1, y1 = min(W, int(math.ceil(xc * W + bw / 2 + px))), min(H, int(math.ceil(yc * H + bh / 2 + py)))
        crop = im.crop((x0, y0, max(x0 + 1, x1), max(y0 + 1, y1))).convert("RGB")
    s = edge / max(crop.size)
    crop = crop.resize((max(1, round(crop.width * s)), max(1, round(crop.height * s))), Image.LANCZOS)
    d = ImageDraw.Draw(crop)
    lw = max(2, round(edge / 160))
    bx0 = (xc * W - bw / 2 - x0) * s
    by0 = (yc * H - bh / 2 - y0) * s
    d.rectangle([bx0, by0, bx0 + bw * s, by0 + bh * s], outline=(255, 40, 40), width=lw)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=85)
    return buf.getvalue()


NAME_SYSTEM = (
    "You name the object classes of a computer-vision detection dataset. The classes have placeholder names (only "
    "numbers). For each class you get example crops: the labeled object is outlined in red in every crop (the rest is "
    "context). Describe what the outlined objects of that class have in common.\n"
    "For each class return:\n"
    "- name: a short English snake_case name (1-3 words, e.g. fallen_person, forklift, hard_hat). Be specific enough "
    "to tell it apart from the other classes listed (pose, state or role when that is what differs).\n"
    "- prompt: a short English noun phrase an open-vocabulary detector can search for (e.g. \"person lying on the "
    "floor\").\n"
    "- description: one sentence for a human annotator: what belongs to the class and how it differs from similar "
    "classes.\n"
    "- confidence: low, medium or high. Use low when the crops show different things or too little to tell.\n"
    "Do not invent details you cannot see. Answer only with the JSON object.")


def _name_props() -> dict:
    return {"name": {"type": "string", "description": "snake_case English name"},
            "prompt": {"type": "string", "description": "noun phrase for an open-vocabulary detector"},
            "description": {"type": "string", "description": "one sentence definition for annotators"},
            "confidence": {"type": "string", "enum": ["low", "medium", "high"]}}


def _snake(s: Any) -> str:
    s = re.sub(r"[^0-9a-zA-Z]+", "_", str(s or "").strip().lower())
    s = re.sub(r"_+", "_", s).strip("_")[:40].strip("_")
    if s and s[0].isdigit():
        s = "c_" + s
    return s


def _conf(v: Any) -> Optional[float]:
    if v is None or v == "":
        return None
    if isinstance(v, (int, float)) and not isinstance(v, bool):
        return round(max(0.0, min(1.0, float(v))), 3)
    return CONF_WORDS.get(str(v).strip().lower())


def name_classes(ds_row: dict, provider: Optional[str], model: Optional[str], cls_ids: Optional[Sequence[int]],
                 per_class: int = 6, dry_run: bool = False, cancel: Any = None) -> dict:
    """Suggest a name, a detector prompt and a one-line definition for each class from a few typical boxes.

    ONE request per class (or one for all classes when they fit in 20 images). Returns
    ``{items: [{cls, current_name, name, prompt, description, confidence, confidence_label, examples, boxes, images,
    error?}], usage, latency_ms, warnings, provider, model, served_by}``; with ``dry_run`` only ``{estimate, items}``
    (the boxes that would be sent, nothing is sent)."""
    ds_id = ds_row["id"]
    names = C.dataset_classes(ds_row)
    if not names:
        raise ProviderError("This dataset has no classes yet", "invalid")
    try:
        per = int(per_class if per_class not in (None, "") else 6)
    except (TypeError, ValueError):
        raise ProviderError("per_class must be an integer between 4 and 8", "invalid") from None
    if not 4 <= per <= 8:
        raise ProviderError("per_class must be between 4 and 8", "invalid")
    if cls_ids:
        ids = []
        for c in cls_ids:
            if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c < len(names):
                raise ProviderError(f"Class {c!r} is not in this dataset (it has {len(names)} classes)", "invalid")
            if c not in ids:
                ids.append(c)
    else:
        ids = list(range(len(names)))
    if len(ids) > 200:
        raise ProviderError("At most 200 classes per request", "invalid")
    p, model = pick(provider, model, "suggest class names")
    cards = C.cards(ds_id)
    stats = {c: _class_stats(ds_id, c) for c in ids}
    examples = {c: pick_examples(ds_id, c, per) for c in ids}
    items: Dict[int, dict] = {}
    for c in ids:
        items[c] = {"cls": c, "current_name": names[c], "boxes": stats[c]["boxes"], "images": stats[c]["images"],
                    "examples": [{"image_id": e["image_id"], "box": [round(e["xc"], 6), round(e["yc"], 6), round(e["w"], 6),
                                                                      round(e["h"], 6)], "rel_path": e["rel_path"]}
                                 for e in examples[c]]}
    todo = [c for c in ids if examples[c]]
    for c in ids:
        if not examples[c]:
            items[c]["error"] = "No labeled boxes of this class to look at"
    # Context about every class (also those not asked) helps telling look-alikes apart (standing vs fallen person).
    all_stats = stats if len(ids) == len(names) else {**{c: _class_stats(ds_id, c) for c in range(len(names))
                                                         if c not in stats and len(names) <= 60}, **stats}

    def context_lines(exclude: Sequence[int]) -> str:
        lines = []
        for c in range(len(names)):
            if c in exclude or c not in all_stats:
                continue
            nm = names[c] if not C.is_placeholder(names[c]) else f"class {c} (unnamed)"
            lines.append(f"- {nm}: boxes {_shape_words(all_stats[c])}")
        return ("Other classes in this dataset:\n" + "\n".join(lines[:40]) + "\n") if lines else ""

    def block(c: int, first_image: int) -> str:
        cd = cards.get(c) or {}
        hint = ""
        if not C.is_placeholder(names[c]):
            hint += f" Current name: {names[c]}."
        if (cd.get("description") or "").strip():
            hint += f" Existing note: {cd['description'].strip()[:200]}."
        n = len(examples[c])
        return (f"Class {c}.{hint} Images {first_image}-{first_image + n - 1} are its examples ({n} crops from {n} "
                f"different images). It has {stats[c]['boxes']:,} labeled boxes in {stats[c]['images']:,} images; boxes "
                f"are {_shape_words(stats[c])}.\n")

    together = sum(len(examples[c]) for c in todo) <= MAX_IMAGES_PER_REQUEST and len(todo) > 1
    requests: List[Tuple[List[int], str, dict]] = []
    head = f"Dataset: {ds_row.get('name') or ds_id}.\n"
    if together:
        k, text = 1, head + "Name these classes:\n"
        for c in todo:
            text += block(c, k)
            k += len(examples[c])
        text += context_lines(todo)
        schema = {"type": "object", "properties": {"classes": {"type": "array", "items": {
            "type": "object", "properties": {"cls": {"type": "integer", "enum": list(todo)}, **_name_props()},
            "required": ["cls", "name", "prompt", "description", "confidence"], "additionalProperties": False}}},
            "required": ["classes"], "additionalProperties": False}
        requests.append((list(todo), text, schema))
    else:
        for c in todo:
            text = head + block(c, 1) + context_lines([c])
            schema = {"type": "object", "properties": _name_props(),
                      "required": ["name", "prompt", "description", "confidence"], "additionalProperties": False}
            requests.append(([c], text, schema))

    if dry_run:
        return {"estimate": naming_estimate(p, model, requests, examples), "items": [items[c] for c in ids],
                "provider": p.id, "model": model}
    from .runner import refuse_unpriced
    refuse_unpriced([(p.id, model)])           # a spend cap must hold: no model whose cost would be booked at $0

    def run_one(req):
        cs, text, schema = req
        # --- ai-ui hook (r2): Stop in the naming dialog (``cancel``, a threading.Event set by
        # POST /api/ai/name-classes/cancel) — a request not sent yet is never sent; those already out finish (billed).
        if cancel is not None and cancel.is_set():
            return cs, None, Usage(), ProviderError("Stopped before this class was sent", "cancelled"), 0.0
        # --- end ai-ui hook
        imgs = []
        for c in cs:
            for e in examples[c]:
                try:
                    imgs.append(crop_jpeg(e["abs_path"], (e["xc"], e["yc"], e["w"], e["h"])))
                except Exception as ex:  # noqa: BLE001 — one unreadable image: send the others
                    imgs.append(None)
                    items[c].setdefault("warnings", []).append(scrub(f"{e['rel_path']}: {ex}")[:200])
        imgs = [b for b in imgs if b]
        t0 = time.time()
        try:
            obj, usage = p.generate_json(NAME_SYSTEM, text, schema, model, images=imgs, options={"max_tokens": 1500})
        except ProviderError as e:     # ai-cloud hook: a refused / unreadable answer is billed all the same
            return cs, None, billed_usage(e) or Usage(), e, time.time() - t0
        return cs, obj, usage, None, time.time() - t0

    t0 = time.time()
    usage = Usage()
    warnings: List[str] = []
    fatal: Optional[ProviderError] = None
    conc = 1 if together else max(1, min(4, len(requests)))
    try:
        from .config import provider_config
        conc = min(conc, int(provider_config(p.id).get("concurrency") or 4))
    except Exception:  # noqa: BLE001
        pass
    with ThreadPoolExecutor(max_workers=max(1, conc), thread_name_prefix="fovea-ai-name") as pool:
        for cs, obj, u, err, _secs in pool.map(run_one, requests):
            usage.add(u)
            if err is not None:
                for c in cs:
                    items[c]["error"] = scrub(str(err))[:300]
                if err.fatal and fatal is None:
                    fatal = err
                continue
            answers = []
            if isinstance(obj, dict) and isinstance(obj.get("classes"), list):
                answers = [a for a in obj["classes"] if isinstance(a, dict)]
            elif isinstance(obj, dict):
                answers = [{**obj, "cls": cs[0]}] if len(cs) == 1 else []
            got = set()
            for a in answers:
                try:
                    c = int(a.get("cls"))
                except (TypeError, ValueError):
                    continue
                if c not in cs or c in got:
                    continue
                got.add(c)
                name = _snake(a.get("name"))
                items[c].update(name=name, prompt=str(a.get("prompt") or "").strip()[:120],
                                description=str(a.get("description") or "").strip()[:300],
                                confidence=_conf(a.get("confidence")),
                                confidence_label=str(a.get("confidence") or "").strip().lower()[:10] or None)
                if not name:
                    items[c]["error"] = "The model gave no usable name"
            for c in cs:
                if c not in got and "error" not in items[c]:
                    items[c]["error"] = "The model gave no answer for this class"
    if fatal is not None and not any(it.get("name") for it in items.values()):
        if usage.cost_usd or usage.input_tokens:      # ai-cloud hook: requests billed before the fatal error
            from . import store
            rid = _ledger(ds_id, "name_classes", p.id, model, usage, {"classes": ids, "per_class": per}, aggregate=False)
            store.finish_run(rid, "error", error=scrub(str(fatal))[:500])
        raise fatal
    # Unique names across the whole class list (a suggestion must not collide with another class).
    taken = {names[c].lower() for c in range(len(names)) if c not in ids}
    for c in ids:
        it = items[c]
        if not it.get("name"):
            continue
        base, k = it["name"], 2
        while it["name"].lower() in taken:
            it["name"] = f"{base}_{k}"[:44]
            k += 1
        if it["name"] != base:
            warnings.append(f"class {c}: '{base}' was suggested twice — renamed to '{it['name']}'")
        taken.add(it["name"].lower())
    if usage.cost_usd or usage.input_tokens:
        _ledger(ds_id, "name_classes", p.id, model, usage, {"classes": ids, "per_class": per}, aggregate=False)
    return {"items": [items[c] for c in ids], "usage": _usage_dict(usage), "latency_ms": int((time.time() - t0) * 1000),
            "warnings": warnings, "provider": p.id, "model": model, "served_by": model, "requests": len(requests)}


def crop_size(e: dict, edge: int = CROP_EDGE, pad: float = CROP_PAD) -> Tuple[int, int]:
    """The size :func:`crop_jpeg` will send for example ``e`` (box + context, long edge ``edge``), from the indexed
    image size — a tall person crop is far smaller than ``edge`` × ``edge``."""
    W, H = int(e.get("width") or 0), int(e.get("height") or 0)
    if W <= 0 or H <= 0:
        return edge, edge
    xc, yc, w, h = (float(e[k]) for k in ("xc", "yc", "w", "h"))
    bw, bh = w * W, h * H
    px, py = max(12.0, bw * pad), max(12.0, bh * pad)
    x0, y0 = max(0, int(xc * W - bw / 2 - px)), max(0, int(yc * H - bh / 2 - py))
    x1, y1 = min(W, int(math.ceil(xc * W + bw / 2 + px))), min(H, int(math.ceil(yc * H + bh / 2 + py)))
    cw, ch = max(1, x1 - x0), max(1, y1 - y0)
    s = edge / max(cw, ch)
    return max(1, round(cw * s)), max(1, round(ch * s))


def naming_estimate(p: Provider, model: str, requests, examples) -> dict:
    crops = [crop_size(e) for cs, _t, _s in requests for c in cs for e in examples[c]]
    n_img = len(crops)
    img_tokens = getattr(p, "image_tokens", None)
    sized: Dict[Tuple[int, int], int] = {}

    def tokens(wh: Tuple[int, int]) -> int:
        if wh not in sized:
            n = 0
            if callable(img_tokens):
                try:
                    n = int(img_tokens(model, wh[0], wh[1], {}))
                except Exception:  # noqa: BLE001
                    n = 0
            sized[wh] = n or 260
        return sized[wh]
    img_in = sum(tokens(wh) for wh in crops)          # each crop priced at the size it is actually sent
    text_in = sum(len(NAME_SYSTEM) // 4 + len(t) // 4 + 40 for _cs, t, _s in requests)
    tin = [text_in + img_in, int((text_in + img_in) * 1.15)]
    n_cls = sum(len(cs) for cs, _t, _s in requests)
    tout = [n_cls * 70, n_cls * 400]                 # reasoning models may think before answering
    pr = _price(p.id, model)
    usd = [0.0, 0.0]
    if pr and pr.get("in") is not None:
        usd = [round((tin[0] * float(pr["in"]) + tout[0] * float(pr.get("out") or pr["in"])) / 1e6, 6),
               round((tin[1] * float(pr["in"]) + tout[1] * float(pr.get("out") or pr["in"])) / 1e6, 6)]
    return {"usd": usd, "tokens_in": tin, "tokens_out": tout, "requests": len(requests), "images": n_img,
            "priced": bool(pr) or not registry.is_cloud(p.id), "provider": p.id, "model": model}


# ------------------------------------------------------------------ natural-language search

LABELED_VALUES = {"labeled": "has boxes", "unlabeled": "no label file yet", "empty": "label file without boxes",
                  "nobox": "no boxes (no file or empty file)", "file": "has a label file"}
REVIEW_VALUES = {"approved": "approved by a reviewer", "flagged": "flagged by a reviewer", "excluded": "excluded",
                 "none": "not reviewed yet", "any": "reviewed (any decision)"}
AI_VALUES = {"pending": "has AI proposals waiting for review", "audit": "has AI label-audit findings",
             "any": "has any pending AI suggestion", "none": "no pending AI suggestion"}
SORT_HELP = {"name": "file name", "boxes": "number of boxes", "size": "image size", "mtime": "file date",
             "reviewed": "review date", "random": "shuffled", "audit": "most likely mislabeled first (label audit)",
             "ai_conf": "least confident AI proposal first"}
_nl_cache: Dict[tuple, Tuple[float, dict]] = {}
NL_CACHE_S = 600


def _issue_labels() -> Dict[str, str]:
    try:
        from ..core.labels import ISSUE_LABELS
        return dict(ISSUE_LABELS)
    except ImportError:  # pragma: no cover
        return {}


def _sort_keys() -> List[str]:
    try:
        from ..core.imagesq import SORT_KEYS
        return [k for k in SORT_KEYS if k != "id"]
    except ImportError:  # pragma: no cover
        return list(SORT_HELP)


def nl_context(ds_row: dict) -> dict:
    ds_id = ds_row["id"]
    names = C.dataset_classes(ds_row)
    cards = C.cards(ds_id)
    counts = {r["cls"]: r["n"] for r in db.query("SELECT cls, COUNT(*) n FROM boxes WHERE dataset_id=? GROUP BY cls",
                                                 (ds_id,))}
    splits = [r["split"] for r in db.query("SELECT split, COUNT(*) n FROM images WHERE dataset_id=? GROUP BY split "
                                           "ORDER BY n DESC", (ds_id,)) if r["split"]]
    return {"names": names, "cards": cards, "counts": counts, "splits": splits, "issues": _issue_labels(),
            "sorts": _sort_keys()}


def nl_prompts(ds_row: dict, ctx: dict, text: str) -> Tuple[str, str, dict]:
    names, cards = ctx["names"], ctx["cards"]
    cls_lines = []
    for i, n in enumerate(names[:300]):
        cd = cards.get(i) or {}
        extra = (cd.get("prompt") or "").strip()
        line = f"class {i}: {n}" + (f" | prompt: {extra[:80]}" if extra else "")
        desc = (cd.get("description") or "").strip()
        if desc:
            line += f" | {desc[:120]}"
        line += f" | {ctx['counts'].get(i, 0):,} boxes"
        cls_lines.append(line)
    issue_lines = [f"{k}: {v}" for k, v in ctx["issues"].items()]
    system = (
        "You turn a search request about the images of an object-detection dataset into Fovea filters. The request "
        "may be in Korean or English. Use only the values listed. An empty list, an empty string or -1 means \"no "
        "filter\". Leave a filter empty unless the request asks for it. Several values in one list mean any of them.\n"
        f"Dataset: {ds_row.get('name') or ds_row['id']}.\n"
        "Classes (cls = images containing ANY of these classes; only_cls = images whose boxes are ALL of these "
        "classes):\n" + "\n".join(cls_lines) + "\n"
        f"Splits: {', '.join(ctx['splits']) or '(none)'}.\n"
        "labeled: " + "; ".join(f"{k} = {v}" for k, v in LABELED_VALUES.items()) + ".\n"
        "review: " + "; ".join(f"{k} = {v}" for k, v in REVIEW_VALUES.items()) + ".\n"
        "ai: " + "; ".join(f"{k} = {v}" for k, v in AI_VALUES.items()) + ".\n"
        "issue codes (problems found in labels and images):\n" + "\n".join(issue_lines) + "\n"
        "q = text that must appear in the file path (folder or file name); seq = an exact sequence (video) name; "
        "min_boxes / max_boxes = number of boxes per image, counting boxes of ALL classes (there is no per-class "
        "count filter).\n"
        "sort: " + "; ".join(f"{k} = {SORT_HELP.get(k, k)}" for k in ctx["sorts"]) + " (empty = keep the current order); "
        "order: asc, desc or empty for the default.\n"
        "explanation: one short line, in the language of the request, describing exactly the filters you set — "
        "nothing more. If part of the request cannot be expressed with these filters (for example a count of one "
        "class, or excluding a class), say plainly which part is not applied.")
    cls_enum = [str(i) for i in range(min(len(names), 300))]
    str_list = lambda vals: {"type": "array", "items": {"type": "string", "enum": vals} if vals else {"type": "string"}}  # noqa: E731
    schema = {"type": "object", "properties": {
        "filters": {"type": "object", "properties": {
            "split": str_list(ctx["splits"]),
            "cls": str_list(cls_enum), "only_cls": str_list(cls_enum),
            "labeled": {"type": "string", "enum": [""] + list(LABELED_VALUES)},
            "review": {"type": "string", "enum": [""] + list(REVIEW_VALUES)},
            "exclude_review": str_list(["approved", "flagged", "excluded"]),
            "issue": str_list(list(ctx["issues"])),
            "ai": {"type": "string", "enum": [""] + list(AI_VALUES)},
            "q": {"type": "string"}, "seq": {"type": "string"},
            "min_boxes": {"type": "integer"}, "max_boxes": {"type": "integer"}},
            "required": ["split", "cls", "only_cls", "labeled", "review", "exclude_review", "issue", "ai", "q", "seq",
                         "min_boxes", "max_boxes"], "additionalProperties": False},
        "sort": {"type": "string", "enum": [""] + ctx["sorts"]},
        "order": {"type": "string", "enum": ["", "asc", "desc"]},
        "explanation": {"type": "string"}},
        "required": ["filters", "sort", "order", "explanation"], "additionalProperties": False}
    user = f"Request: {text.strip()}"
    return system, user, schema


def _as_list(v: Any) -> List[str]:
    if v is None or v == "":
        return []
    if isinstance(v, (list, tuple)):
        return [str(x).strip() for x in v if str(x).strip()]
    return [s.strip() for s in str(v).split(",") if s.strip()]


def validate_filters(ds_row: dict, ctx: dict, raw: Any) -> Tuple[Dict[str, Any], List[str]]:
    """Keep only filter keys and values Fovea understands and this dataset has. Returns (filters, warnings)."""
    from ..api.common import FILTER_KEYS
    f: Dict[str, Any] = {}
    warn: List[str] = []
    if not isinstance(raw, dict):
        return f, (["The model returned no filters"] if raw not in (None, {}) else [])
    names = ctx["names"]
    by_name = {}
    for i, n in enumerate(names):
        for k in C.name_keys(n):
            by_name.setdefault(k, i)

    def class_ids(vals) -> List[int]:
        out = []
        for v in _as_list(vals):
            i = None
            if re.fullmatch(r"-?\d+", v):
                i = int(v)
            else:
                i = next((by_name[k] for k in C.name_keys(v) if k in by_name), None)
            if i is None or not 0 <= i < len(names):
                warn.append(f"ignored unknown class {v!r}")
                continue
            if i not in out:
                out.append(i)
        return out

    for key, val in raw.items():
        if key not in FILTER_KEYS or key in ("ids", "seed", "has_label", "set"):
            if val not in (None, "", [], -1) and key not in ("ids", "seed", "set"):
                warn.append(f"ignored unknown filter {key!r}")
            continue
        if key in ("cls", "only_cls"):
            ids = class_ids(val)
            if ids:
                f[key] = ids
        elif key == "split":
            ok = [s for s in _as_list(val) if s in ctx["splits"]]
            bad = [s for s in _as_list(val) if s not in ctx["splits"]]
            if bad:
                warn.append(f"ignored unknown split {', '.join(bad)}")
            if ok:
                f[key] = ok
        elif key == "issue":
            ok = [s for s in _as_list(val) if s in ctx["issues"] or s == "any"]
            if len(ok) < len(_as_list(val)):
                warn.append("ignored unknown issue codes")
            if ok:
                f[key] = ok
        elif key == "labeled":
            v = str(val or "").strip()
            if v in LABELED_VALUES:
                f[key] = v
            elif v:
                warn.append(f"ignored labeled={v!r}")
        elif key == "review":
            v = str(val or "").strip()
            if v in REVIEW_VALUES:
                f[key] = v
            elif v:
                warn.append(f"ignored review={v!r}")
        elif key == "exclude_review":
            ok = [s for s in _as_list(val) if s in ("approved", "flagged", "excluded")]
            if ok:
                f[key] = ok
        elif key == "ai":
            v = str(val or "").strip()
            if v in AI_VALUES:
                f[key] = v
            elif v:
                warn.append(f"ignored ai={v!r}")
        elif key == "q":
            v = str(val or "").strip()[:200]
            if v:
                f[key] = v
        elif key == "seq":
            v = str(val or "").strip()[:300]
            if v:
                if db.query_one("SELECT 1 FROM images WHERE dataset_id=? AND seq=? LIMIT 1", (ds_row["id"], v)):
                    f[key] = v
                else:
                    warn.append(f"no sequence named {v!r} — searched the file paths instead")
                    f["q"] = f.get("q") or v
        elif key in ("min_boxes", "max_boxes"):
            try:
                n = int(val)
            except (TypeError, ValueError):
                continue
            if n >= 0 and not (key == "min_boxes" and n == 0):
                f[key] = min(n, 100000)
    if "min_boxes" in f and "max_boxes" in f and f["min_boxes"] > f["max_boxes"]:
        f["min_boxes"], f["max_boxes"] = f["max_boxes"], f["min_boxes"]
    return f, warn


def nl_filter(ds_row: dict, text: Any, provider: Optional[str] = None, model: Optional[str] = None) -> dict:
    """Natural-language search → ``{filters, sort, order, explanation, warnings, usage, provider, model, served_by,
    cached}``. Cheap: the provider's cheapest model unless one is given; identical questions within 10 minutes are
    answered from a cache (no second charge)."""
    if not isinstance(text, str) or not text.strip():
        raise ProviderError("Type what you are looking for", "invalid")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > 500:
        raise ProviderError("Keep the question under 500 characters", "invalid")
    p, model = pick(provider, model, "answer questions", cheapest=True)
    from .runner import refuse_unpriced
    refuse_unpriced([(p.id, model)])
    ctx = nl_context(ds_row)
    system, user, schema = nl_prompts(ds_row, ctx, text)
    key = (ds_row["id"], p.id, model, text.lower(), hash(system), hash(json.dumps(schema, sort_keys=True)))
    hit = _nl_cache.get(key)
    if hit and time.time() - hit[0] < NL_CACHE_S:
        return {**hit[1], "cached": True, "usage": {"input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0}}
    t0 = time.time()
    try:
        obj, usage = p.generate_json(system, user, schema, model, images=None,
                                     options={"effort": "low", "max_tokens": 800})
    except ProviderError as e:         # ai-cloud hook: a refused / unreadable answer is billed all the same
        if billed_usage(e) is not None:
            _ledger(ds_row["id"], "nl_filter", p.id, model, billed_usage(e), {"questions": 1}, aggregate=True)
        raise
    if not isinstance(obj, dict):
        if usage.cost_usd or usage.input_tokens:      # ai-cloud hook
            _ledger(ds_row["id"], "nl_filter", p.id, model, usage, {"questions": 1}, aggregate=True)
        raise ProviderError("The model did not answer with filters", "bad_output")
    filters, warnings = validate_filters(ds_row, ctx, obj.get("filters"))
    sort = str(obj.get("sort") or "").strip()
    sort = sort if sort in ctx["sorts"] else None
    order = str(obj.get("order") or "").strip().lower()
    order = order if (sort and order in ("asc", "desc")) else None
    expl = re.sub(r"\s+", " ", str(obj.get("explanation") or "")).strip()[:240]
    if usage.cost_usd or usage.input_tokens:
        _ledger(ds_row["id"], "nl_filter", p.id, model, usage, {"questions": 1}, aggregate=True)
    out = {"filters": filters, "sort": sort, "order": order, "explanation": expl or ("No filter matched the request"
                                                                                     if not filters else ""),
           "warnings": warnings, "usage": _usage_dict(usage), "latency_ms": int((time.time() - t0) * 1000),
           "provider": p.id, "model": model, "served_by": model, "cached": False}
    if len(_nl_cache) > 256:
        _nl_cache.clear()
    _nl_cache[key] = (time.time(), out)
    return out
