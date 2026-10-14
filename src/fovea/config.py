from __future__ import annotations

import os
from pathlib import Path

APP_ROOT = Path(__file__).resolve().parent.parent  # src/
PKG_ROOT = Path(__file__).resolve().parent          # src/fovea

DATA_DIR = Path(os.environ.get("FOVEA_DATA_DIR", APP_ROOT / "data")).expanduser()
THUMB_DIR = DATA_DIR / "thumbs"
DB_PATH = DATA_DIR / "fovea.db"
MODEL_DIR = Path(os.environ.get("FOVEA_MODEL_DIR", APP_ROOT / "model")).expanduser()
STATIC_DIR = APP_ROOT / "static"
TEMPLATE_DIR = APP_ROOT / "templates"

# Host <-> container path mapping (Docker). HOST_PATH is mounted at CONTAINER_MOUNT.
HOST_PATH = os.environ.get("HOST_PATH", "").strip()
CONTAINER_MOUNT = os.environ.get("CONTAINER_MOUNT", "/host").strip() or "/host"


def _env_num(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except ValueError:
        return default


def _optional_formats() -> set:
    """Formats Ultralytics trains on that this Pillow build can also decode. HEIC/HEIF need pillow-heif
    (registered here, once, so every Image.open in the app can read them)."""
    exts = {".mpo"}                                  # multi-picture JPEG (some cameras): Pillow reads frame 0
    try:
        from PIL import features
        if features.check("jpg_2000"):
            exts |= {".jp2", ".j2k", ".jpeg2000"}
        if features.check("avif"):
            exts.add(".avif")
    except Exception:  # noqa: BLE001
        pass
    try:
        from pillow_heif import register_heif_opener   # optional dependency
        register_heif_opener()
        exts |= {".heic", ".heif"}
    except Exception:  # noqa: BLE001
        pass
    return exts


# .dng is deliberately absent: Pillow opens a DNG's small embedded preview, so sizes and labels would not match.
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"} | _optional_formats()
LABEL_EXT = ".txt"
SPLIT_NAMES = ("train", "val", "valid", "validation", "test", "eval")
THUMB_SIZES = (128, 256, 512, 1024, 2048)            # 2048: a screen-sized rendition for Annotate / Inspect
TEXT_PREVIEW_MAX = 512 * 1024
# Decoding is refused above this many pixels (after JPEG draft scaling): a few-MB PNG can declare 60k x 60k
# pixels and cost 10 GB to decode. Raise it for genuinely huge rasters (e.g. 2_000_000_000).
MAX_IMAGE_PIXELS = int(_env_num("FOVEA_MAX_IMAGE_PIXELS", 250_000_000))
# (qa-engine hook) Pillow's own decompression-bomb guard follows the same cap, process-wide: Image.open refuses
# rasters declaring more than twice this. Every decoder still checks MAX_IMAGE_PIXELS after draft scaling; no
# module may lift the guard (Image.MAX_IMAGE_PIXELS = None once did, for the whole server).
from PIL import Image as _PILImage  # noqa: E402

_PILImage.MAX_IMAGE_PIXELS = MAX_IMAGE_PIXELS
# Thumbnail/rendition cache cap; the oldest entries are removed beyond it (0 = unbounded).
THUMB_CACHE_MAX_BYTES = int(_env_num("FOVEA_THUMB_CACHE_MAX_GB", 10) * 1024 ** 3)

DATA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
