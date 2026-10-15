"""Find a dataset's image and label folders when they are not called images/ and labels/.

Adding a dataset whose tree uses other names (imgs/ + annotations/, train_images/ + train_labels/, Photos/2024/…)
asks a text model (any provider with ``capabilities["generate"]``: Claude, Gemini, OpenAI, an OpenAI-compatible
server, the mock) which folders hold the images and which the YOLO .txt labels. Only the folder listing goes out —
relative folder names, file counts and three sample file names per folder — never an image or a label.

This is the default (Settings → AI → "Find dataset folders"). Every answer is checked against the listing: a path
that is not in it, has no images or lies outside the dataset folder is dropped. When no provider is configured, the
spend cap is reached, or nothing survives the check, ``core.layout`` matches conventional folder names instead.
The answer for one listing is kept for an hour, so the add dialog's re-detects and the registration that follows
cost one call. These calls are not booked in Runs & cost (no dataset exists yet); each is one short text request.
"""
from __future__ import annotations

import hashlib
import threading
import time
from pathlib import Path, PurePosixPath
from typing import Dict, List, Optional

from .base import ProviderError

CACHE_S = 3600
_cache: Dict[tuple, tuple] = {}
_lock = threading.Lock()

SYSTEM = ("You map the folders of an object-detection dataset. Images are pictures (.jpg, .png …); labels are YOLO "
          ".txt files, one per image with the same file name. You are given the dataset's folder listing and answer "
          "which folders hold the images and which folder holds each one's labels. Use only paths from the listing, "
          "exactly as written. Only folders of YOLO .txt files are label folders (the listing marks each folder's "
          "first .txt line as YOLO or not). A folder of .txt in another format (VisDrone, KITTI, DOTA …), of .xml "
          "(Pascal VOC) or of .json (LabelMe) that belongs to an images folder is its `annotations` — Fovea converts "
          "those — never its `labels`.")

SCHEMA = {
    "type": "object",
    "properties": {
        "sources": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "split": {"type": "string", "description": "train, val or test; a short folder-derived name when the "
                                                               "folders are not splits; empty when there is one set"},
                    "images": {"type": "string", "description": "the folder that holds the images (a listed path)"},
                    "labels": {"type": "string", "description": "the folder with this images folder's YOLO .txt labels "
                                                                "(a listed path); the images folder itself when labels "
                                                                "sit beside the images; empty when there are none yet"},
                    "annotations": {"type": "string", "description": "the folder with this images folder's annotations "
                                                                     "in another format (a listed path); empty when "
                                                                     "there are none"},
                },
                "required": ["split", "images", "labels", "annotations"],
            },
        },
        "reason": {"type": "string", "description": "one short sentence"},
    },
    "required": ["sources"],
}


def mode() -> str:
    from .config import ai_settings
    return ai_settings().get("folder_roles") or "ai"


def _prompt(root: Path, lines: List[str]) -> str:
    return (f"Dataset folder: {root.name}\n"
            "Folders (path relative to the dataset folder · files directly inside · counts below it · sample names):\n"
            + "\n".join(lines) +
            "\n\nList every folder that holds this dataset's images (the deepest folder that holds them directly, or "
            "a parent when its sub-folders are only image folders of one set), with its split and its label folder.")


def _pick_provider(allow_guarded: bool):
    from . import assist, registry
    for pid in assist.generators():
        if not allow_guarded and registry.is_guarded(pid):
            continue
        try:
            p, model = assist.pick(pid, None, "find dataset folders", cheapest=True)
        except ProviderError:
            continue
        if registry.is_cloud(pid):
            from .runner import caps_left
            left = caps_left([pid])
            if left is not None and left < 0.005:
                continue
        return p, model
    return None, None


def _clean_rel(s) -> Optional[str]:
    s = str(s or "").strip().strip("/").replace("\\", "/")
    if s in ("", "."):
        return "." if s == "." else None
    pp = PurePosixPath(s)
    if pp.is_absolute() or ".." in pp.parts:
        return None
    return pp.as_posix()


def validate(root: Path, inv, raw) -> List[tuple]:
    """[(split, img_dir, label_dir, recursive, annotations)] from a model answer; anything not in the listing is
    dropped, and a "label" folder whose .txt are not YOLO becomes the annotations to convert."""
    from ..core.layout import default_label_dir, norm_split, split_names
    by_rel = {d.rel: d for d in inv}
    out, seen = [], set()
    for item in (raw or {}).get("sources") or []:
        if not isinstance(item, dict):
            continue
        img = _clean_rel(item.get("images"))
        d = by_rel.get(img or "")
        if d is None or d.tree_images <= 0 or img in seen:
            continue
        lab_rel = _clean_rel(item.get("labels"))
        lab = by_rel.get(lab_rel) if lab_rel else None
        ann_rel = _clean_rel(item.get("annotations"))
        ann = by_rel.get(ann_rel) if ann_rel else None
        if lab is not None and lab.rel == img and d.txt_kind == "other":
            ann, lab = ann or d, None                           # .txt of another format beside the images
        elif lab is not None and lab.rel != img and not lab.yolo_labels:
            ann, lab = ann or (lab if lab.tree_ann else None), None   # VisDrone .txt, .xml, .json: converted
        if ann is not None and not ann.tree_ann:
            ann = None
        label_dir = lab.path if lab is not None else default_label_dir(d.path, root)
        split = norm_split(str(item.get("split") or "").strip())[:40]
        out.append((split, d.path, label_dir, True, ann.path if ann is not None else None))
        seen.add(img)
    # A parent and its own sub-folder both listed would index the same images twice: keep the deeper ones.
    rels = [str(t[1]) for t in out]
    out = [t for t in out if not any(o != str(t[1]) and o.startswith(str(t[1]) + "/") for o in rels)]
    if len(out) > 1 and any(not t[0] for t in out):
        out = [(t[0] or _split_of(t[1], root), *t[1:]) for t in out]     # several sets left unnamed: name them
    names = split_names(sorted({t[0] for t in out}))              # 'visdrone2019-det-val' → val
    return [(names.get(t[0], t[0]),) + tuple(t[1:]) for t in out]


def _split_of(img: Path, root: Path) -> str:
    """A split for an image folder the model left unnamed: 'train' in its name (train_images), else its parent's
    name (train/imgs, siteA/imgs), else its own name."""
    from ..core.layout import dir_role, norm_split
    role, split = dir_role(img.name)
    if split:
        return split
    if img.parent != root:
        return img.parent.name
    return "" if role else norm_split(img.name)


def find_roles(root: Path, inv, use_setting: bool = True, allow_guarded: bool = True) -> Optional[dict]:
    """``{pairs, provider, model, reason}`` — or ``{pairs: [], why}`` when AI could not answer (the caller falls back
    to folder names), or None when folder detection by AI is turned off."""
    if use_setting and mode() != "ai":
        return None
    from ..core.layout import INVENTORY_MAX_DIRS, inventory_lines
    lines = inventory_lines(inv)
    if not lines:
        return {"pairs": [], "why": "no images in this folder"}
    # A home folder, a drive or a tree too big to list is not one dataset: its folder names are not sent anywhere.
    home = Path.home()
    if root == Path(root.anchor) or root in (home, home.parent) or len(inv) >= INVENTORY_MAX_DIRS:
        return {"pairs": [], "why": "this folder is too large to be one dataset — pick the dataset's own folder"}
    p, model = _pick_provider(allow_guarded)
    if p is None:
        return {"pairs": [], "why": "no AI provider that can write text is set up"}
    user = _prompt(root, lines)
    key = (str(root), p.id, model, hashlib.sha1(user.encode("utf-8", "surrogatepass")).hexdigest())
    with _lock:
        hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_S:
        raw = hit[1]
    else:
        try:
            raw, _usage = p.generate_json(SYSTEM, user, SCHEMA, model, images=None,
                                          options={"effort": "low", "max_tokens": 800})
        except ProviderError as e:
            return {"pairs": [], "why": f"{p.name or p.id}: {e}"[:200]}
        with _lock:
            if len(_cache) > 128:
                _cache.clear()
            _cache[key] = (time.time(), raw)
    pairs = validate(root, inv, raw if isinstance(raw, dict) else {})
    if not pairs:
        return {"pairs": [], "why": f"{p.name or p.id} named no image folder that exists"}
    reason = " ".join(str((raw or {}).get("reason") or "").split())[:240]
    return {"pairs": pairs, "provider": p.id, "model": model, "reason": reason}


def clear_cache() -> None:
    with _lock:
        _cache.clear()
