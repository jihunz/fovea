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

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
LABEL_EXT = ".txt"
SPLIT_NAMES = ("train", "val", "valid", "validation", "test", "eval")
THUMB_SIZES = (128, 256, 512, 1024)
TEXT_PREVIEW_MAX = 512 * 1024

DATA_DIR.mkdir(parents=True, exist_ok=True)
THUMB_DIR.mkdir(parents=True, exist_ok=True)
