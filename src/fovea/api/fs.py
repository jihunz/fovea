from __future__ import annotations

from fastapi import APIRouter, HTTPException, Query

from ..core.layout import detect, slugify
from ..paths import browse, resolve, to_host

router = APIRouter(prefix="/api/fs", tags=["fs"])


@router.get("/browse")
def api_browse(path: str = Query(""), files: int = Query(1)):
    try:
        return browse(path, show_files=bool(files))
    except PermissionError as e:
        raise HTTPException(403, str(e))


@router.get("/detect")
def api_detect(path: str = Query(...)):
    try:
        lay = detect(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not detect layout: {e}")
    p = resolve(path)
    name = p.stem if p.is_file() else p.name
    return {"layout": lay.to_dict(), "suggested_name": name or "dataset", "suggested_id": slugify(name),
            "path_host": to_host(p)}
