from __future__ import annotations

from fastapi import APIRouter, Body

from .. import __version__, db
from ..config import CONTAINER_MOUNT, DATA_DIR, HOST_PATH, MODEL_DIR
from ..core import thumbs, yolo
from ..paths import PATH_MAPPINGS, to_host

router = APIRouter(prefix="/api", tags=["meta"])
_plugins_cache = []


def set_plugins(plugins) -> None:
    _plugins_cache.clear()
    _plugins_cache.extend(plugins)


@router.get("/meta")
def meta():
    return {
        "version": __version__,
        "data_dir": str(DATA_DIR), "model_dir": str(MODEL_DIR),
        "host_path": HOST_PATH, "container_mount": CONTAINER_MOUNT, "mappings": PATH_MAPPINGS,
        "ai": {"available": yolo.available(), "models": yolo.list_models()},
        "thumb_cache": thumbs.cache_stats(),
        "plugins": [p.manifest() for p in _plugins_cache],
        "settings": db.kv_get("settings", {}),
    }


@router.get("/plugins")
def plugins():
    return {"plugins": [p.manifest() for p in _plugins_cache]}


@router.get("/settings")
def get_settings():
    return {"settings": db.kv_get("settings", {})}


@router.put("/settings")
def put_settings(payload: dict = Body(...)):
    cur = db.kv_get("settings", {}) or {}
    cur.update(payload or {})
    db.kv_set("settings", cur)
    return {"settings": cur}


@router.post("/cache/clear")
def clear_cache():
    return {"removed": thumbs.clear_cache()}
