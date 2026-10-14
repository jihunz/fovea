from __future__ import annotations

import importlib.util
import threading
from typing import Any

from fastapi import APIRouter, Body, HTTPException, Query, Request

from .. import __version__, db, security
from ..config import CONTAINER_MOUNT, DATA_DIR, HOST_PATH, MODEL_DIR
from ..core import thumbs
from ..paths import PATH_MAPPINGS

router = APIRouter(prefix="/api", tags=["meta"])
_plugins_cache = []
_settings_lock = threading.Lock()
_ultralytics: dict = {}


def set_plugins(plugins) -> None:
    _plugins_cache.clear()
    _plugins_cache.extend(plugins)


def _ultralytics_installed() -> bool:
    """find_spec only: importing ultralytics (and torch) costs 10+ s and must never happen on a request."""
    if "ok" not in _ultralytics:
        _ultralytics["ok"] = importlib.util.find_spec("ultralytics") is not None
    return _ultralytics["ok"]


def _model_files() -> list:
    out = []
    try:
        for p in sorted(MODEL_DIR.iterdir()):
            if p.is_file() and p.suffix.lower() in (".pt", ".onnx", ".engine"):
                out.append({"name": p.stem, "file": p.name, "size_mb": round(p.stat().st_size / 1e6, 1)})
    except OSError:
        pass
    return out


def _settings() -> dict:
    cur = db.kv_get("settings", {}) or {}
    return security.strip_secrets(cur) if isinstance(cur, dict) else {}


@router.get("/meta")
def meta(request: Request):
    """Cheap by design (the shell may call it at boot): no filesystem walk, no ultralytics import."""
    return {
        "version": __version__,
        "data_dir": str(DATA_DIR), "model_dir": str(MODEL_DIR),
        "host_path": HOST_PATH, "container_mount": CONTAINER_MOUNT, "mappings": PATH_MAPPINGS,
        "ai": {"available": _ultralytics_installed(), "models": _model_files()},
        # Running counter; null until the first background count finishes. GET /api/cache/stats for details.
        "thumb_cache": thumbs.cache_stats(),
        "plugins": [p.manifest() for p in _plugins_cache],
        "settings": _settings(),
        "server": security.server_info(request),
    }


@router.get("/plugins")
def plugins():
    return {"plugins": [p.manifest() for p in _plugins_cache]}


@router.get("/settings")
def get_settings():
    return {"settings": _settings()}


def _deep_merge(base: dict, patch: dict) -> dict:
    """RFC 7386-style merge: nested objects merge, null deletes, everything else replaces."""
    out = dict(base)
    for k, v in patch.items():
        if v is None:
            out.pop(k, None)
        elif isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _deep_merge(out[k], v)
        elif isinstance(v, dict):
            out[k] = _deep_merge({}, v)
        else:
            out[k] = v
    return out


@router.put("/settings")
def put_settings(payload: Any = Body(...)):
    """Merge non-secret UI settings. API keys are refused here: this blob is returned by GET /api/settings
    and /api/meta. Provider keys go to PUT /api/ai/providers/{id}/key, which never echoes them."""
    if not isinstance(payload, dict):
        raise HTTPException(400, "Settings must be a JSON object")
    bad = security.find_secrets(payload)
    if bad:
        raise HTTPException(400, f"Settings cannot hold API keys or other secrets ({', '.join(bad[:5])}). "
                                 "Add provider keys under Settings → AI providers instead.")
    with _settings_lock:
        cur = _settings()
        merged = _deep_merge(cur, payload)
        db.kv_set("settings", merged)
    return {"settings": merged}


@router.get("/cache/stats")
def cache_stats(refresh: int = Query(0, description="1 = recount the cache folder in the background")):
    return {"thumb_cache": thumbs.cache_stats(refresh=bool(refresh))}


@router.post("/cache/clear")
def clear_cache():
    return {"removed": thumbs.clear_cache(), "thumb_cache": thumbs.cache_stats()}
