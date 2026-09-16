from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from .. import db
from ..core.thumbs import needs_rendition, rendition_for, thumb_for

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
    headers = {"ETag": etag, "Cache-Control": "private, max-age=3600"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    if needs_rendition(str(p)):
        # TIFF and friends index and thumbnail fine, but no browser can display them: Inspect would be
        # blank and Annotate inert. Serve a full-resolution JPEG rendition instead.
        out = rendition_for(str(p))
        if out is None:
            raise HTTPException(415, "This image format cannot be displayed")
        return FileResponse(out, media_type="image/jpeg", headers=headers)
    return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "image/jpeg", headers=headers)


@router.get("/thumb/{image_id}")
def thumbnail(image_id: int, request: Request, s: int = Query(256, ge=32, le=2048), v: Optional[str] = Query(None)):
    row = _image_row(image_id)
    out = thumb_for(row["abs_path"], s)
    if out is None:
        raise HTTPException(415, "Could not render thumbnail")
    # The cache file name encodes path, size, mtime and rendering, so it is a precise ETag. A URL that carries
    # the image version (?v=) can be cached for good; an unversioned one must revalidate (cheap 304) or a
    # replaced image would keep its old thumbnail in the browser for a week.
    etag = f'"{out.stem}"'
    headers = {"ETag": etag, "Cache-Control": "private, max-age=604800, immutable" if v else "private, no-cache"}
    if request.headers.get("if-none-match") == etag:
        return Response(status_code=304, headers=headers)
    return FileResponse(out, media_type="image/jpeg", headers=headers)
