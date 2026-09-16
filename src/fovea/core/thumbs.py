"""Thumbnail generation with an on-disk cache keyed by path + mtime + size."""
from __future__ import annotations

import functools
import hashlib
import os
import threading
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

from ..config import THUMB_DIR, THUMB_SIZES

Image.MAX_IMAGE_PIXELS = None
_locks: dict = {}
_locks_guard = threading.Lock()


def snap_size(s: int) -> int:
    for cand in THUMB_SIZES:
        if s <= cand:
            return cand
    return THUMB_SIZES[-1]


HIGH_BIT_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")
WEB_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif"}   # what every browser decodes natively


@functools.lru_cache(maxsize=200_000)
def _probe(abs_path: str, mtime: float) -> Tuple[int, str]:
    """(EXIF orientation, PIL mode) from the header only — once per file version per process."""
    try:
        with Image.open(abs_path) as im:
            try:
                orient = int(im.getexif().get(0x0112) or 1)
            except Exception:  # noqa: BLE001 — a malformed EXIF block is simply "no orientation"
                orient = 1
            return orient, im.mode
    except Exception:  # noqa: BLE001
        return 1, ""


def _key(abs_path: str, size, mtime: float, orient: int = 1, mode: str = "") -> str:
    base = f"{abs_path}|{size}|{mtime:.3f}"
    # Only images whose rendering changed get a new key, so existing caches of ordinary images stay valid.
    if orient not in (0, 1):
        base += f"|o{orient}"
    if mode in HIGH_BIT_MODES:
        base += "|n2"
    return hashlib.sha1(base.encode("utf-8", "surrogateescape")).hexdigest()


def to_display_rgb(im: Image.Image) -> Image.Image:
    """8-bit RGB for display. High-bit-depth images (16-bit, 32-bit int, float) are stretched over their
    real value range — a plain convert("RGB") clips them to a flat white or black rectangle."""
    if im.mode == "RGB":
        return im
    if im.mode == "L":
        return im.convert("RGB")
    if im.mode in HIGH_BIT_MODES:
        f = im.convert("F")
        lo, hi = f.getextrema()
        if hi <= lo:
            return Image.new("RGB", im.size, (128, 128, 128))
        scale = 255.0 / (hi - lo)
        return f.point(lambda v: v * scale + (-lo * scale)).convert("L").convert("RGB")
    try:
        return im.convert("RGB")
    except Exception:  # noqa: BLE001
        return ImageOps.autocontrast(im.convert("L")).convert("RGB")


def _render(abs_path: str, out: Path, key: str, size: Optional[int], quality: int) -> Optional[Path]:
    if out.is_file():
        return out
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if out.is_file():
            return out
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_suffix(".tmp")
        try:
            with Image.open(abs_path) as im:
                if size:
                    try:
                        im.draft("RGB", (size * 2, size * 2))
                    except Exception:  # noqa: BLE001
                        pass
                    im.thumbnail((size, size), Image.Resampling.BILINEAR)
                # Browsers rotate photos by their EXIF orientation; a rendition that did not would show
                # the picture sideways with its (upright-normalised) boxes on the wrong objects.
                im = ImageOps.exif_transpose(im)
                im = to_display_rgb(im)
                im.save(tmp, "JPEG", quality=quality, optimize=bool(size))
            os.replace(tmp, out)
        except Exception:  # noqa: BLE001
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            return None
        finally:
            with _locks_guard:
                _locks.pop(key, None)
    return out


def thumb_for(abs_path: str, size: int) -> Optional[Path]:
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return None
    size = snap_size(size)
    orient, mode = _probe(abs_path, mtime)
    key = _key(abs_path, size, mtime, orient, mode)
    return _render(abs_path, THUMB_DIR / key[:2] / f"{key}.jpg", key, size, 82)


def needs_rendition(abs_path: str) -> bool:
    return Path(abs_path).suffix.lower() not in WEB_EXTS


def rendition_for(abs_path: str) -> Optional[Path]:
    """Full-resolution JPEG of an image browsers cannot decode (TIFF), cached beside the thumbnails.
    Native resolution on purpose: a downscaled proxy would quietly cap the detail available to label."""
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return None
    orient, mode = _probe(abs_path, mtime)
    key = _key(abs_path, "full", mtime, orient, mode)
    return _render(abs_path, THUMB_DIR / "full" / key[:2] / f"{key}.jpg", key, None, 92)


def cache_stats() -> dict:
    n = 0
    total = 0
    for root, _dirs, files in os.walk(THUMB_DIR):
        for f in files:
            n += 1
            try:
                total += os.stat(os.path.join(root, f)).st_size
            except OSError:
                pass
    return {"files": n, "bytes": total}


def clear_cache() -> int:
    n = 0
    for root, _dirs, files in os.walk(THUMB_DIR):
        for f in files:
            try:
                os.unlink(os.path.join(root, f))
                n += 1
            except OSError:
                pass
    return n
