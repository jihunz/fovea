"""Thumbnail generation with an on-disk cache keyed by path + mtime + size."""
from __future__ import annotations

import hashlib
import os
import threading
from pathlib import Path
from typing import Optional

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


def _key(abs_path: str, size: int, mtime: float) -> str:
    return hashlib.sha1(f"{abs_path}|{size}|{mtime:.3f}".encode("utf-8", "surrogateescape")).hexdigest()


def thumb_for(abs_path: str, size: int) -> Optional[Path]:
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return None
    size = snap_size(size)
    key = _key(abs_path, size, mtime)
    out = THUMB_DIR / key[:2] / f"{key}.jpg"
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
                try:
                    im.draft("RGB", (size * 2, size * 2))
                except Exception:
                    pass
                im.thumbnail((size, size), Image.Resampling.BILINEAR)
                if im.mode not in ("RGB", "L"):
                    try:
                        im = im.convert("RGB")
                    except Exception:
                        im = ImageOps.autocontrast(im.convert("L")).convert("RGB")
                elif im.mode == "L":
                    im = im.convert("RGB")
                im.save(tmp, "JPEG", quality=82, optimize=True)
            os.replace(tmp, out)
        except Exception:
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
