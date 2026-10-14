from __future__ import annotations

import os
import re
import secrets
import threading
import time
import unicodedata
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
from urllib.parse import quote

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse, PlainTextResponse, Response, StreamingResponse

from .. import db
from ..core import export as ex
from ..jobs import jobs
from ..paths import to_host
from .common import filters_from, get_dataset_or_404

router = APIRouter(prefix="/api/datasets", tags=["export"])


def _safe_name(s: str, default: str) -> str:
    """A file name from user text: Unicode letters and digits survive (a Korean split name must not
    collapse into another split's file name), NFC-normalised, path separators and controls removed."""
    s = unicodedata.normalize("NFC", s or "")
    s = re.sub(r"[^\w.\-]+", "-", s, flags=re.UNICODE).strip("-.")
    return s[:120] or default


def _disposition(fname: str, kind: str = "attachment") -> str:
    ascii_name = re.sub(r"[^A-Za-z0-9._-]+", "-", unicodedata.normalize("NFKD", fname).encode("ascii", "ignore").decode()) or "export"
    return f"{kind}; filename=\"{ascii_name}\"; filename*=UTF-8''{quote(fname)}"


def _checked(payload: dict) -> dict:
    """Type-check the free-text fields before anything is planned: a number or a list used to reach
    unicodedata.normalize / str.strip and come back as a 500."""
    for key in ("filename", "target_dir"):
        v = payload.get(key)
        if v is not None and not isinstance(v, str):
            raise HTTPException(400, f"{key} must be a string")
    return payload


def _options(payload: dict) -> ex.ExportOptions:
    try:
        return ex.parse_options(payload)
    except ValueError as e:
        raise HTTPException(400, str(e))


def _export_filters(payload: dict) -> dict:
    f = filters_from(payload.get("filters") or {})
    if not payload.get("include_unlabeled", True):
        f["has_label"] = "1"                   # so counts, progress and the archive agree
    return f


def _names(row: dict) -> list:
    return db.loads(row["classes"]) or []


def _zip_name(row: dict, payload: dict) -> str:
    return _safe_name(payload.get("filename") or f"{row['id']}-{time.strftime('%Y%m%d-%H%M%S')}", "export") + ".zip"


def _zip_response(row: dict, payload: dict, prepared: Optional[ex.ZipPrepared] = None) -> StreamingResponse:
    opts = _options(payload)
    headers = {"Content-Disposition": _disposition(_zip_name(row, payload)), "Cache-Control": "no-store"}
    if prepared is not None:
        # Every member is known: an exact Content-Length lets the browser show progress and time left.
        headers["Content-Length"] = str(prepared.length)
        headers["X-Fovea-Export-Images"] = str(prepared.result.get("images", 0))
        return StreamingResponse(ex.stream_prepared(prepared), media_type="application/zip", headers=headers)
    gen = ex.stream_zip(row["id"], _names(row), _export_filters(payload), opts=opts, include_manifest=True)
    return StreamingResponse(gen, media_type="application/zip", headers=headers)


def _prepare(row: dict, payload: dict) -> Optional[ex.ZipPrepared]:
    opts = _options(payload)
    if opts.resize is not None:
        return None
    try:
        return ex.prepare_zip(row["id"], _names(row), _export_filters(payload), opts)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{ds_id}/export/zip")
def export_zip(ds_id: str, payload: dict = Body(...)):
    """Stream a ZIP directly (API clients). The web UI uses zip-link instead."""
    row = get_dataset_or_404(ds_id)
    _checked(payload)
    return _zip_response(row, payload, _prepare(row, payload))


_LINKS: Dict[str, Tuple[float, str, dict, Optional[ex.ZipPrepared]]] = {}
_LINKS_LOCK = threading.Lock()
LINK_TTL = 300.0


@router.post("/{ds_id}/export/zip-link")
def export_zip_link(ds_id: str, payload: dict = Body(...)):
    """A one-time URL for the same export. The browser downloads it natively — streaming to disk with its
    own progress UI — where fetch() + blob() would hold the entire archive in the tab's memory first.
    Options are validated here (400 before any download starts); without resizing the archive is planned
    here too, so the download carries an exact size."""
    row = get_dataset_or_404(ds_id)
    _checked(payload)
    prepared = _prepare(row, payload)
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _LINKS_LOCK:
        for k in [k for k, v in _LINKS.items() if now - v[0] > LINK_TTL]:
            del _LINKS[k]
        _LINKS[token] = (now, ds_id, payload, prepared)
    info: Dict[str, Any] = {"url": f"/api/datasets/{ds_id}/export/zip/{token}", "expires_in": int(LINK_TTL),
                            "filename": _zip_name(row, payload)}
    if prepared is not None:
        r = prepared.result
        info.update(bytes=prepared.length, images=r.get("images"), labels=r.get("labels"), missing=r.get("missing"),
                    dropped_near_dups=r.get("dropped_near_dups"), split_counts=r.get("split_counts"),
                    warnings=r.get("warnings"))
    return info


@router.get("/{ds_id}/export/zip/{token}")
def export_zip_download(ds_id: str, token: str):
    with _LINKS_LOCK:
        entry = _LINKS.pop(token, None)          # single use
    if not entry or entry[1] != ds_id or time.time() - entry[0] > LINK_TTL:
        raise HTTPException(404, "This download link has expired — start the export again")
    return _zip_response(get_dataset_or_404(ds_id), entry[2], entry[3])


@router.post("/{ds_id}/export/preview")
def export_preview(ds_id: str, payload: dict = Body(...)):
    """What an export with these options would contain (counts per split and class, near-duplicates
    dropped, unnamed class ids, sequences spanning splits, data.yaml, warnings), from the index only.
    Pass target_dir to also check the copy destination."""
    row = get_dataset_or_404(ds_id)
    _checked(payload)
    opts = _options(payload)
    try:
        return ex.preview(ds_id, _names(row), _export_filters(payload), opts,
                          (payload.get("target_dir") or "").strip() or None)
    except ValueError as e:
        raise HTTPException(400, str(e))


@router.post("/{ds_id}/export/copy")
def export_copy(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    _checked(payload)
    target = (payload.get("target_dir") or "").strip()
    if not target:
        raise HTTPException(400, "target_dir is required")
    opts = _options(payload)
    try:
        t = ex.check_copy_target(ds_id, target)       # fail fast, before a job is queued
    except ValueError as e:
        raise HTTPException(400, str(e))
    info = ex.inspect_target(t)
    if info.get("error"):
        raise HTTPException(400, f"{info['path']}: {info['error']}")
    if info["entries"] and not opts.overwrite:
        n = f"{info['entries']:,}{'+' if info['entries'] >= 1000 else ''}"
        what = "a previous Fovea export" if info["previous_export"] else f"{n} items"
        return JSONResponse(status_code=409, content={
            "detail": f"{info['path']} already contains {what}. Choose another folder name, or confirm to replace it"
                      + (" (files of the previous export that this one does not write are removed)." if info["previous_export"]
                         else " (existing files with the same names are overwritten; other files are left as they are)."),
            "target": info})
    busy = [j for j in jobs.active_all(ds_id, ["export_copy"]) if (j.extra or {}).get("target") == str(t)]
    if busy:
        raise HTTPException(409, "An export into this folder is already running")
    f = _export_filters(payload)
    names = _names(row)
    job = jobs.submit("export_copy", lambda j: ex.copy_subset(j, ds_id, names, f, target, opts=opts),
                      dataset_id=ds_id, message="Queued")
    job.extra = {"target": str(t), "target_host": to_host(t)}
    return {"job": job.to_dict()}


@router.post("/{ds_id}/export/list")
def export_list(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    _checked(payload)
    f = filters_from(payload.get("filters") or {})
    style = payload.get("style") or "host"
    if style not in ("host", "rel", "container"):
        raise HTTPException(400, "style must be host, rel or container")
    root = Path(row["root"])
    text = ex.render_list(ds_id, f, style, list_dir=str(root))
    if payload.get("write"):
        fname = _safe_name(payload.get("filename") or "fovea-list", "list") + ".txt"
        out = root / fname
        try:
            ex.refuse_fovea_dirs(out, to_host(out))
        except ValueError as e:
            raise HTTPException(400, str(e))
        used = {str(Path(s["list_file"])) for s in (db.loads(row["layout"], {}) or {}).get("sources", []) if s.get("list_file")}
        if str(out) in used:
            raise HTTPException(409, f"{fname} is one of this dataset's own image lists — choose another name")
        if out.exists() and not payload.get("overwrite"):
            return JSONResponse(status_code=409, content={
                "detail": f"{to_host(out)} already exists — confirm to replace it", "exists": True, "path": to_host(out)})
        tmp = out.with_name(f".{fname}.fovea-tmp")
        try:
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, out)
        except OSError as e:
            if tmp.exists():
                tmp.unlink()
            raise HTTPException(403 if isinstance(e, PermissionError) else 500, f"Could not write {to_host(out)}: {e.strerror or e}")
        return {"path": to_host(out), "lines": text.count("\n"), "style": style,
                "relative_to": to_host(root) if style == "rel" else None}
    fname = _safe_name(payload.get("filename") or f"{row['id']}-list", "list") + ".txt"
    return PlainTextResponse(text, headers={"Content-Disposition": _disposition(fname)})


@router.get("/{ds_id}/card.md")
def dataset_card_md(ds_id: str, request: Request, inline: int = Query(0)):
    """DATASET_CARD.md for the dataset as indexed — or for the images matching the usual image filters given
    in the query (split, cls, review, issue, set, ...): splits, classes with their class-card definitions,
    data quality, review progress and label issues. Exports can include the same card (dataset_card: true),
    which then also records the export settings. inline=1 shows it in the browser instead of downloading."""
    row = get_dataset_or_404(ds_id)
    try:
        text = ex.card_for_dataset(ds_id, _names(row), filters_from(request))
    except ValueError as e:
        raise HTTPException(400, str(e))
    fname = f"{_safe_name(row.get('name') or ds_id, 'dataset')}-{ex.CARD}"
    return Response(text, media_type=("text/plain" if inline else "text/markdown") + "; charset=utf-8",
                    headers={"Content-Disposition": _disposition(fname, "inline" if inline else "attachment"),
                             "Cache-Control": "no-store"})


@router.get("/{ds_id}/export/yaml")
def export_yaml_preview(ds_id: str):
    """The data.yaml 'Write/Update data.yaml' would produce, next to the current file (for a diff/confirm)."""
    row = get_dataset_or_404(ds_id)
    try:
        p = ex.plan_data_yaml(row["root"], _names(row), db.loads(row["layout"], {}) or {})
    except ValueError as e:
        raise HTTPException(400, str(e))
    p.pop("path", None)
    return p


@router.post("/{ds_id}/export/yaml")
def export_yaml(ds_id: str):
    row = get_dataset_or_404(ds_id)
    lay = db.loads(row["layout"], {}) or {}
    try:
        out = ex.write_data_yaml(row["root"], _names(row), lay)
    except ValueError as e:
        raise HTTPException(400, str(e))
    except PermissionError as e:
        raise HTTPException(403, f"Permission denied: {e.filename or e}")
    lay["data_yaml"] = out
    lay["data_yaml_host"] = to_host(out)
    db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), ds_id))
    bak = Path(out).with_name(Path(out).name + ".bak")
    return {"path": to_host(out), "backup": to_host(bak) if bak.exists() else None}
