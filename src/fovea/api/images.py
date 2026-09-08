from __future__ import annotations

import mimetypes
import os
from pathlib import Path

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from .. import db
from ..core.thumbs import thumb_for

router = APIRouter(prefix="/api", tags=["images"])


def _image_row(image_id: int) -> dict:
    row = db.query_one("SELECT abs_path, mtime FROM images WHERE id=?", (image_id,))
    if not row or not os.path.isfile(row["abs_path"]):
        raise HTTPException(404, "Image not found")
    return row


@router.get("/img/{image_id}")
def full_image(image_id: int, request: Request):
    row = _image_row(image_id)
    p = Path(row["abs_path"])
    st = p.stat()
    etag = f'W/"{image_id}-{int(st.st_mtime)}-{st.st_size}"'
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304)
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "image/jpeg",
                        headers={"ETag": etag, "Cache-Control": "private, max-age=3600"})


@router.get("/thumb/{image_id}")
def thumbnail(image_id: int, s: int = Query(256, ge=32, le=2048)):
    row = _image_row(image_id)
    out = thumb_for(row["abs_path"], s)
    if out is None:
        raise HTTPException(415, "Could not render thumbnail")
    return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "private, max-age=604800"})
