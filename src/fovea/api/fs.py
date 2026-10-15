from __future__ import annotations

import unicodedata

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from ..core.layout import detect, slugify
from ..paths import BrowseNotFound, browse, resolve, to_host
from ..security import ai_access_allowed

router = APIRouter(prefix="/api/fs", tags=["fs"])


@router.get("/browse")
def api_browse(path: str = Query(""), files: int = Query(1), limit: int = Query(2000, ge=1, le=20000)):
    try:
        return browse(path, show_files=bool(files), max_files=limit)
    except BrowseNotFound as e:
        # 404 plus the closest folder that does exist, so the picker can offer it instead of jumping home.
        return JSONResponse(status_code=404, content={"detail": str(e),
                                                      "nearest": to_host(e.nearest) if e.nearest else None})
    except PermissionError as e:
        raise HTTPException(403, str(e))
    except OSError as e:
        # e.g. an unmounted network share: a clear message instead of a 500
        raise HTTPException(503, f"Cannot read this folder: {e.strerror or e}")


_LIST_SUFFIXES = (".yaml", ".yml", ".txt")


def suggested_name(p) -> str:
    """A dataset name for a picked path: a data.yaml / train.txt (or an images/ or split folder) is named
    after the dataset folder that holds it, not 'data' / 'train' / 'images'."""
    try:
        from ..core.layout import suggest_name          # backend-data's rule (generic folder names too)
        name = suggest_name(p)
    except ImportError:
        if p.is_file() or p.suffix.lower() in _LIST_SUFFIXES:
            name = p.parent.name if p.suffix.lower() in _LIST_SUFFIXES else p.stem
        else:
            name = p.name
    return unicodedata.normalize("NFC", name or "")


@router.get("/detect")
def api_detect(request: Request, path: str = Query(...), ai: int = Query(1)):
    """`ai=0`: folders with other names than images/ and labels/ are matched by name only (no AI call)."""
    try:
        lay = detect(path, ai=None if ai else False, allow_guarded=ai_access_allowed(request))
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not detect layout: {e}")
    p = resolve(path)
    name = suggested_name(p)
    return {"layout": lay.to_dict(), "suggested_name": name or "dataset", "suggested_id": slugify(name),
            "path_host": to_host(p), "previews": _format_previews(lay)}


def _format_previews(lay) -> dict:
    """For each format waiting for conversion: a few sample images with the converted boxes drawn on them, so the
    reading rule can be checked by eye before anything is written."""
    from pathlib import Path

    from ..core import annformats as af
    out = {}
    for fid, f in (lay.formats or {}).items():
        if f.get("state") != "pending" or not f.get("card"):
            continue
        src = next((s for s in lay.sources if s.ann_format == fid and s.ann_path and s.img_dir), None)
        if src is None:
            continue
        try:
            out[fid] = af.preview(f["card"], Path(src.img_dir), Path(src.ann_path), src.recursive)
        except Exception:  # noqa: BLE001 — a preview is a convenience; detection itself succeeded
            out[fid] = []
    return out
