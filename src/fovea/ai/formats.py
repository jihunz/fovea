"""Read an annotation format the catalog does not know (core/annformats.py): an AI provider turns sample lines into a
reading rule — and, when Settings allows looking it up (the default), first identifies the dataset and follows its
official documentation found with the provider's web search, plus the dataset's Hugging Face card when one exists.

What leaves the machine: the dataset's folder names, a few sample annotation lines with their image sizes — never an
image — and, for the look-up, the guessed dataset name in a Hugging Face search. Every answer becomes a format card
that core.layout only uses after `annformats.check` ran it over real files. Answers are cached for 30 days per sample,
so re-detecting the same dataset costs nothing. These calls are not booked in Runs & cost (no dataset exists yet).
"""
from __future__ import annotations

import hashlib
import json
import re
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from .. import db
from .base import ProviderError

MODES = ("web", "ai", "catalog")
CACHE_KEY = "ai.format_cards"
CACHE_S = 30 * 86400
HF_TIMEOUT = 6.0

SYSTEM = ("You read the annotation files of an object-detection dataset and write the rule that turns each line into "
          "one box (or polygon) with its class and attributes. Use column numbers counted from 0 after splitting a "
          "line by its delimiter. Coordinates may be pixels (compare them with the image sizes given) or normalised "
          "to 0..1. Class values that mark areas which are not objects (ignored regions, don't-care, crowds) are "
          "`regions`, not classes. Per-object flags such as occlusion, truncation or difficulty are `attributes`.")

_VAL = {"type": "object", "properties": {"value": {"type": "string"}, "meaning": {"type": "string"}},
        "required": ["value", "meaning"]}
SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string", "description": "the dataset's official name and version when recognised "
                                                     "(e.g. VisDrone2019-DET), else empty"},
        "format_name": {"type": "string"},
        "delimiter": {"type": "string", "enum": ["comma", "space", "tab", "semicolon"]},
        "header_prefixes": {"type": "array", "items": {"type": "string"},
                            "description": "lines starting with one of these are not annotations"},
        "geometry": {"type": "string", "enum": ["ltwh", "xyxy", "cxcywh", "poly"],
                     "description": "ltwh: left, top, width, height; xyxy: two corners; cxcywh: centre and size; "
                                    "poly: x1 y1 x2 y2 … corner points"},
        "geometry_columns": {"type": "array", "items": {"type": "integer"}},
        "units": {"type": "string", "enum": ["pixel", "normalized"]},
        "class_column": {"type": "integer", "description": "-1 when every line is the same class"},
        "classes": {"type": "array", "items": {
            "type": "object", "properties": {"value": {"type": "string", "description": "as written in the file"},
                                             "name": {"type": "string"},
                                             "definition": {"type": "string",
                                                            "description": "one sentence, from the official "
                                                                           "documentation when there is one"}},
            "required": ["value", "name", "definition"]}},
        "regions": {"type": "array", "items": {
            "type": "object", "properties": {"value": {"type": "string"}, "name": {"type": "string"},
                                             "description": {"type": "string"}},
            "required": ["value", "name", "description"]}},
        "attributes": {"type": "array", "items": {
            "type": "object", "properties": {"name": {"type": "string"}, "column": {"type": "integer"},
                                             "values": {"type": "array", "items": _VAL}},
            "required": ["name", "column", "values"]}},
        "urls": {"type": "array", "items": {"type": "string"},
                 "description": "official pages the rule follows (empty when it is read from the samples alone)"},
        "license": {"type": "string"},
        "citation": {"type": "string"},
        "notes": {"type": "string"},
        "confidence": {"type": "number"},
    },
    "required": ["dataset", "format_name", "delimiter", "header_prefixes", "geometry", "geometry_columns", "units",
                 "class_column", "classes", "regions", "attributes", "urls", "license", "citation", "notes",
                 "confidence"],
}


def mode() -> str:
    from .config import ai_settings
    m = ai_settings().get("annotation_formats") or "web"
    return m if m in MODES else "web"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", (s or "").lower()).strip("-")[:40] or "custom"


def card_from_answer(ans: dict, source: str, provider: str = "", model: str = "") -> Optional[dict]:
    """A format card (core.annformats) from a model answer, or None when the answer is not a usable rule."""
    if not isinstance(ans, dict):
        return None
    delim = {"comma": ",", "space": "space", "tab": "tab", "semicolon": ";"}.get(str(ans.get("delimiter") or ""))
    geom = str(ans.get("geometry") or "")
    try:
        cols = [int(c) for c in ans.get("geometry_columns") or []]
        ccol = int(ans.get("class_column") if ans.get("class_column") is not None else -1)
    except (TypeError, ValueError):
        return None
    if delim is None or geom not in ("ltwh", "xyxy", "cxcywh", "poly") or any(c < 0 or c > 64 for c in cols):
        return None
    if (geom == "poly" and (len(cols) < 6 or len(cols) % 2)) or (geom != "poly" and len(cols) != 4):
        return None
    classes: List[str] = []
    values: Dict[str, int] = {}
    defs: Dict[str, dict] = {}
    for c in ans.get("classes") or []:
        if not isinstance(c, dict):
            continue
        v, name = str(c.get("value") or "").strip(), str(c.get("name") or "").strip()[:64]
        if not name or v in values:
            continue
        if name not in classes:
            classes.append(name)
        values[v or name] = classes.index(name)
        if c.get("definition"):
            defs[name] = {"description": str(c["definition"]).strip()[:400]}
    regions, region_cards = {}, {}
    for r in ans.get("regions") or []:
        if isinstance(r, dict) and str(r.get("value") or "").strip():
            kind = _slug(r.get("name") or "region")
            regions[str(r["value"]).strip()] = kind
            region_cards[kind] = {"label": str(r.get("name") or kind)[:40],
                                  "description": str(r.get("description") or "")[:300]}
    attrs = []
    for a in (ans.get("attributes") or [])[:8]:
        try:
            col = int(a.get("column"))
        except (TypeError, ValueError, AttributeError):
            continue
        if col < 0 or col > 64:
            continue
        vals = {str(v.get("value")): str(v.get("meaning") or "")[:60] for v in a.get("values") or []
                if isinstance(v, dict) and str(v.get("value") or "") != ""}
        attrs.append({"name": _slug(a.get("name") or f"attr_{col}").replace("-", "_"),
                      "label": str(a.get("name") or f"Attribute {col}")[:40], "column": col, "values": vals})
    spec: Dict[str, Any] = {"delimiter": delim, "geometry": {"type": geom, "columns": cols,
                                                            "units": "normalized" if ans.get("units") == "normalized"
                                                            else "pixel"},
                            "attributes": attrs, "regions": regions}
    prefixes = [p for p in (ans.get("header_prefixes") or []) if isinstance(p, str) and p.strip()][:6]
    if prefixes:
        spec["skip"] = "^(" + "|".join(re.escape(p.strip()) for p in prefixes) + ")"
    if ccol < 0:
        spec["class"] = {"column": None}
        classes = classes[:1] or ["object"]
    elif values and all(re.fullmatch(r"-?\d+", v) for v in values):
        spec["class"] = {"column": ccol, "values": values}
    else:
        spec["class"] = {"column": ccol, "from_data": True}       # class names written in the file
    spec["min_columns"] = max(cols + [ccol] + [a["column"] for a in attrs]) + 1
    name = str(ans.get("dataset") or ans.get("format_name") or "Custom format").strip()[:60]
    return {
        "id": f"{source}-{_slug(name)}", "name": name, "kind": "delimited", "ext": ".txt", "single": False,
        "source": source, "provider": provider, "model": model, "spec": spec, "classes": classes,
        "class_defs": defs, "regions": region_cards,
        "urls": [u for u in (ans.get("urls") or []) if isinstance(u, str) and re.match(r"https?://", u)][:6],
        "license": str(ans.get("license") or "")[:200], "citation": str(ans.get("citation") or "")[:300],
        "notes": str(ans.get("notes") or "")[:600], "confidence": ans.get("confidence"),
    }


def _sample_block(sn, img_dir: Path, ann: Path, root: Path) -> str:
    """Up to three annotation files, each with its image size, for the prompt."""
    from ..core import annformats as af
    out = []
    for f in sn.files[:3]:
        size = None
        rel = f.relative_to(ann).with_suffix("") if f.is_relative_to(ann) else Path(f.stem)
        for ext in (".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"):
            for cand in (img_dir / rel.with_suffix(ext), img_dir / (f.stem + ext)):
                if cand.is_file():
                    size = af.image_size(cand)
                    break
            if size:
                break
        lines = af.head_lines(f, 10)
        out.append(f"--- {f.name}" + (f" (image {size[0]}×{size[1]} px)" if size else "") + "\n" + "\n".join(lines))
    return "\n".join(out)


def _hf_card(query: str) -> str:
    """The README of the best Hugging Face dataset match for `query` (a guessed dataset name), or ''."""
    from . import vlm
    q = re.sub(r"[^\w\- ]+", " ", query).strip()[:60]
    if len(q) < 3:
        return ""
    try:
        r = vlm._http("GET", "https://huggingface.co/api/datasets", "Hugging Face", timeout=HF_TIMEOUT,
                      params={"search": q, "limit": 3, "sort": "downloads"})
        hits = r.json()
        for h in hits if isinstance(hits, list) else []:
            ds = h.get("id") if isinstance(h, dict) else None
            if not ds or not re.fullmatch(r"[\w.\-]+(/[\w.\-]+)?", ds):
                continue
            t = vlm._http("GET", f"https://huggingface.co/datasets/{ds}/raw/main/README.md", "Hugging Face",
                          timeout=HF_TIMEOUT)
            if t.text.strip():
                return f"Hugging Face dataset card of {ds} (it may describe a re-packaged copy):\n{t.text[:6000]}"
    except Exception:  # noqa: BLE001 — the card is a hint; no network just means no hint
        return ""
    return ""


def _guess_name(root: Path, ann: Path) -> str:
    """A dataset name to look up from the folder names: 'VisDrone2019-DET-train' → 'VisDrone2019-DET'."""
    for name in [*reversed(dataset_label(root).split("/")), ann.parent.name, ann.name]:
        toks = [t for t in re.split(r"[-_ .]+", name) if t]
        toks = [t for t in toks if t.lower() not in ("train", "val", "valid", "test", "dev", "images", "labels",
                                                     "annotations", "data", "dataset", "raw")]
        if toks and any(re.search(r"[a-z]{3,}", t, re.I) for t in toks):
            return "-".join(toks)
    return ""


def _pick(web: bool, allow_guarded: bool, cheapest: bool = False):
    from . import assist, registry
    for pid in assist.generators():
        p = registry.get(pid)
        if web and not (getattr(p, "capabilities", {}) or {}).get("web_search"):
            continue
        if not allow_guarded and registry.is_guarded(pid):
            continue
        try:
            p, model = assist.pick(pid, None, "read annotation formats", cheapest=cheapest)
        except ProviderError:
            continue
        if registry.is_cloud(pid):
            from .runner import caps_left
            left = caps_left([pid])
            if left is not None and left < 0.05:
                continue
        return p, model
    return None, None


def _cache_get(key: str) -> Optional[list]:
    c = db.kv_get(CACHE_KEY, {}) or {}
    hit = c.get(key) if isinstance(c, dict) else None
    if hit and time.time() - float(hit.get("at") or 0) < CACHE_S:
        return hit.get("cards")
    return None


def remember(key: Optional[str], card: dict) -> None:
    """Keep a rule that passed the check on the real files (only those: a rule that failed is asked for again)."""
    if not key:
        return
    c = db.kv_get(CACHE_KEY, {}) or {}
    c = c if isinstance(c, dict) else {}
    if len(c) > 200:
        c = dict(sorted(c.items(), key=lambda kv: kv[1].get("at", 0))[-100:])
    c[key] = {"at": time.time(), "cards": [{**card, "cached": True}]}
    db.kv_set(CACHE_KEY, c)


ID_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "dataset": {"type": "string", "description": "official name and version of the published dataset these files "
                                                     "come from, or empty"},
        "published": {"type": "boolean", "description": "true only for a dataset with public documentation"},
        "confidence": {"type": "number"},
    },
    "required": ["dataset", "published", "confidence"],
}


_GENERIC = {"data", "dataset", "datasets", "raw", "images", "imgs", "train", "val", "test", "annotations", "labels",
            "download", "downloads", "extracted", "unzipped", "files", "source", "src"}


def dataset_label(root: Path) -> str:
    """The dataset folder's name with its parents while it is generic: CARPK_devkit/data, not just "data"."""
    parts, p = [root.name], root
    while p.name.lower() in _GENERIC and len(parts) < 3 and p.parent != p and p.parent.name:
        p = p.parent
        parts.insert(0, p.name)
    return "/".join(parts)


def _context(sn, img_dir: Path, ann: Path, root: Path) -> tuple:
    samples = _sample_block(sn, img_dir, ann, root)
    hint = (f"Dataset folder: {dataset_label(root)}\nAnnotation folder: "
            f"{ann.relative_to(root) if ann.is_relative_to(root) else ann.name}\nImage folder: "
            f"{img_dir.relative_to(root) if img_dir.is_relative_to(root) else img_dir.name}")
    return hint, samples


def _ask(p, model: str, user: str, options: dict, spent: dict, source: str) -> Optional[dict]:
    obj, u = p.generate_json(SYSTEM, user, SCHEMA, model, images=None, options=options)
    spent["usd"] = spent.get("usd", 0.0) + float(u.cost_usd or 0)
    c = card_from_answer(obj, source if source != "web" else ("web" if (obj or {}).get("urls") else "ai"), p.id, model)
    if c:
        c["usd"] = round(spent["usd"], 4)
    return c


def infer_card(sn, img_dir: Path, ann: Path, root: Path, use_setting: bool = True, allow_guarded: bool = True,
               failed: Optional[List[dict]] = None) -> dict:
    """{"cards": [card], "key": cache key} — a reading rule to check on the real files (core.layout keeps it with
    `remember` once it passed) — or {"why": …}. With the look-up allowed, the cheapest model first says whether the
    files come from a published dataset; only then does a model with web search read its official documentation
    (and its Hugging Face card). Otherwise, or when that fails, the rule is read from the samples alone. Only
    plain-text formats (one .txt per image) are inferred; .xml / .json formats come from the catalog."""
    m = mode() if use_setting else "web"
    if m == "catalog":
        return {"why": "reading unknown formats with AI is turned off in Settings"}
    if sn is None or sn.ext != ".txt" or sn.single or not sn.lines:
        return {"why": "only one-.txt-per-image formats can be read by AI"}
    hint, samples = _context(sn, img_dir, ann, root)
    key = hashlib.sha1(f"{m}\n{hint}\n{samples}".encode("utf-8", "surrogatepass")).hexdigest()
    hit = _cache_get(key)
    if hit:
        return {"cards": hit, "key": None}
    tried = "".join(f"\nA known rule ({t.get('name')}) did not fit: {'; '.join(t.get('problems') or [])}"
                    for t in failed or [])
    why: List[str] = []
    spent: Dict[str, float] = {}
    name = ""
    if m == "web":
        p, model = _pick(False, allow_guarded, cheapest=True)
        if p is not None:
            try:
                obj, u = p.generate_json(
                    "You recognise published computer-vision datasets from their folder names and annotation files.",
                    f"{hint}\nSample annotation files:\n{samples}\n\nDo these files come from a published dataset "
                    "with public documentation? Name it with its version, or leave `dataset` empty.",
                    ID_SCHEMA, model, images=None, options={"effort": "low", "max_tokens": 300})
                spent["usd"] = float(u.cost_usd or 0)
                if isinstance(obj, dict) and obj.get("published") and float(obj.get("confidence") or 0) >= 0.4:
                    name = str(obj.get("dataset") or "").strip()[:80]
            except (ProviderError, TypeError, ValueError) as e:
                why.append(f"{p.name or p.id}: {e}"[:200])
    if name:
        p, model = _pick(True, allow_guarded)
        if p is None:
            why.append("no AI provider with web search is set up")
        else:
            hf = _hf_card(name) if p.id != "mock" else ""
            user = (f"{hint}\nThe files look like the published dataset {name}.\n{tried}\n"
                    f"Sample annotation files:\n{samples}\n\n{hf + chr(10) * 2 if hf else ''}"
                    "Search the web for this dataset's official annotation format documentation (its site, its GitHub "
                    "toolkit, its paper) and write the reading rule exactly as that documentation defines it — class "
                    "ids and names, regions that are not objects, attribute meanings — checked against the samples. "
                    "List the URLs you used.")
            try:
                c = _ask(p, model, user, {"web_search": True, "max_searches": 5, "effort": "medium",
                                          "max_tokens": 4000}, spent, "web")
                if c:
                    return {"cards": [c], "key": key}
                why.append(f"{p.name or p.id} gave no usable rule")
            except ProviderError as e:
                why.append(f"{p.name or p.id}: {e}"[:200])
    p, model = _pick(False, allow_guarded)
    if p is None:
        why.append("no AI provider that can write text is set up")
        return {"why": "; ".join(why)}
    user = (f"{hint}\n{tried}\nSample annotation files:\n{samples}\n\n"
            "Write the reading rule from the samples. Name classes only when the files or folder names say what they "
            "are; otherwise use class_<value>. Say in a definition when a name or meaning is inferred.")
    try:
        c = _ask(p, model, user, {"effort": "medium", "max_tokens": 3000}, spent, "ai")
        if c:
            return {"cards": [c], "key": key}
        why.append(f"{p.name or p.id} gave no usable rule")
    except ProviderError as e:
        why.append(f"{p.name or p.id}: {e}"[:200])
    return {"why": "; ".join(why) or "no answer"}


def repair_card(sn, img_dir: Path, ann: Path, root: Path, card: dict, problems: List[str],
                allow_guarded: bool = True) -> dict:
    """One more try after a rule failed the check: the model sees its rule, what went wrong and the lines again.
    {"cards": [card], "key": …} or {"why": …}."""
    p, model = _pick(False, allow_guarded)
    if p is None or sn is None or sn.ext != ".txt" or sn.single:
        return {"why": "no AI provider to correct the rule"}
    hint, samples = _context(sn, img_dir, ann, root)
    key = hashlib.sha1(f"{mode()}\n{hint}\n{samples}".encode("utf-8", "surrogatepass")).hexdigest()
    rule = {k: card.get(k) for k in ("name", "spec", "classes")}
    user = (f"{hint}\nSample annotation files:\n{samples}\n\nThis reading rule was applied to the real files and "
            f"failed: {'; '.join(problems)}\nThe rule: {json.dumps(rule, ensure_ascii=False)[:3000]}\n"
            "Lines are split by the delimiter; brackets and quotes around a value are removed; column numbers count "
            "from 0. Write a corrected rule that reads every sample line, keeping the dataset's class names.")
    spent = {"usd": float(card.get("usd") or 0)}
    try:
        c = _ask(p, model, user, {"effort": "medium", "max_tokens": 3000}, spent, "ai")
    except ProviderError as e:
        return {"why": f"{p.name or p.id}: {e}"[:200]}
    if c is None:
        return {"why": f"{p.name or p.id} gave no usable rule"}
    for k in ("urls", "license", "citation"):           # the documentation the first rule was read from still holds
        if card.get(k) and not c.get(k):
            c[k] = card[k]
    if card.get("source") == "web":
        c["source"] = "web"
        c["id"] = card.get("id") or c["id"]
    if card.get("class_defs") and not c.get("class_defs"):
        c["class_defs"] = card["class_defs"]
    return {"cards": [c], "key": key}


def clear_cache() -> None:
    db.kv_set(CACHE_KEY, {})
