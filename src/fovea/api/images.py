from __future__ import annotations

import mimetypes
import os
from pathlib import Path
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request, Response
from fastapi.responses import FileResponse

from .. import db
from ..core.thumbs import needs_rendition, passthrough_ok, probe, render_tag, rendition_render, thumb_render

router = APIRouter(prefix="/api", tags=["images"])

# A URL that names the image version (?v=<mtime>) never changes meaning, so the browser may keep it for
# good. An unversioned URL must revalidate (a cheap 304): image ids are reused after a dataset is deleted
# or relocated, and an edited file keeps its id — max-age would show the old picture for an hour.
IMMUTABLE = "private, max-age=31536000, immutable"
REVALIDATE = "private, no-cache"


def _image_row(image_id: int) -> dict:
    row = db.query_one("SELECT abs_path, mtime FROM images WHERE id=?", (image_id,))
    if not row or not os.path.isfile(row["abs_path"]):
        raise HTTPException(404, "Image not found")
    return row


def _not_modified(request: Request, etag: str) -> bool:
    inm = request.headers.get("if-none-match")
    if not inm:
        return False
    return etag in [t.strip() for t in inm.split(",")] or inm.strip() == "*"


def _mime(p: Path) -> str:
    return mimetypes.guess_type(p.name)[0] or {".avif": "image/avif", ".webp": "image/webp"}.get(p.suffix.lower(), "image/jpeg")


@router.get("/img/{image_id}")
def full_image(image_id: int, request: Request, v: Optional[str] = Query(None)):
    row = _image_row(image_id)
    p = Path(row["abs_path"])
    st = p.stat()
    rendition = needs_rendition(str(p))
    tag = render_tag(str(p)) if rendition else "o"
    etag = f'W/"{image_id}-{int(st.st_mtime * 1000)}-{st.st_size}-{tag}"'
    headers = {"ETag": etag, "Cache-Control": IMMUTABLE if v else REVALIDATE}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    if rendition:
        # TIFF, JPEG 2000, HEIC and high-bit-depth PNGs index and thumbnail fine, but a browser shows them
        # blank or near-black: serve a full-resolution 8-bit JPEG rendition instead.
        out, err = rendition_render(str(p))
        if out is None:
            raise HTTPException(415, f"This image cannot be displayed: {err or 'unsupported format'}")
        return FileResponse(out, media_type="image/jpeg", headers=headers)
    return FileResponse(p, media_type=_mime(p), headers=headers)


@router.get("/thumb/{image_id}")
def thumbnail(image_id: int, request: Request, s: int = Query(256, ge=32, le=4096), v: Optional[str] = Query(None)):
    row = _image_row(image_id)
    path = row["abs_path"]
    pr = probe(path)
    cache = IMMUTABLE if v else REVALIDATE
    if passthrough_ok(path, s, pr):
        # Already a browser-native image no bigger than the tile: the original is the best thumbnail.
        p = Path(path)
        st = p.stat()
        etag = f'"o{image_id}-{int(st.st_mtime * 1000)}-{st.st_size}"'
        headers = {"ETag": etag, "Cache-Control": cache}
        if _not_modified(request, etag):
            return Response(status_code=304, headers=headers)
        return FileResponse(p, media_type=_mime(p), headers=headers)
    out, err = thumb_render(path, s)
    if out is None:
        raise HTTPException(415, f"Could not render a thumbnail: {err or 'unsupported image'}")
    # The cache file name encodes path, size, mtime and rendering, so it is a precise ETag.
    etag = f'"{out.stem}"'
    headers = {"ETag": etag, "Cache-Control": cache}
    if _not_modified(request, etag):
        return Response(status_code=304, headers=headers)
    return FileResponse(out, media_type="image/jpeg", headers=headers)
