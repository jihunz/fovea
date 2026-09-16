from __future__ import annotations

import os
import re
import secrets
import threading
import time
from pathlib import Path
from typing import Dict, Tuple

from fastapi import APIRouter, Body, HTTPException
from fastapi.responses import PlainTextResponse, StreamingResponse

from .. import db
from ..core import export as ex
from ..jobs import jobs
from ..paths import to_host
from .common import filters_from, get_dataset_or_404

router = APIRouter(prefix="/api/datasets", tags=["export"])


def _safe_name(s: str, default: str) -> str:
    s = re.sub(r"[^A-Za-z0-9._-]+", "-", s or "").strip("-.")
    return s or default


def _export_filters(payload: dict) -> dict:
    f = filters_from(payload.get("filters") or {})
    if not payload.get("include_unlabeled", True):
        f["has_label"] = "1"                   # so counts, progress and the archive agree
    return f


def _zip_response(row: dict, payload: dict) -> StreamingResponse:
    names = db.loads(row["classes"])
    gen = ex.stream_zip(row["id"], names, _export_filters(payload), payload.get("resize"),
                        bool(payload.get("include_unlabeled", True)), bool(payload.get("include_yaml", True)))
    fname = _safe_name(payload.get("filename") or f"{row['id']}-{int(time.time())}", "export") + ".zip"
    return StreamingResponse(gen, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/{ds_id}/export/zip")
def export_zip(ds_id: str, payload: dict = Body(...)):
    """Stream a ZIP directly (API clients). The web UI uses zip-link instead."""
    return _zip_response(get_dataset_or_404(ds_id), payload)


_LINKS: Dict[str, Tuple[float, str, dict]] = {}
_LINKS_LOCK = threading.Lock()
LINK_TTL = 300.0


@router.post("/{ds_id}/export/zip-link")
def export_zip_link(ds_id: str, payload: dict = Body(...)):
    """A one-time URL for the same export. The browser downloads it natively — streaming to disk with its
    own progress UI — where fetch() + blob() would hold the entire archive in the tab's memory first."""
    get_dataset_or_404(ds_id)
    token = secrets.token_urlsafe(24)
    now = time.time()
    with _LINKS_LOCK:
        for k in [k for k, (t, _d, _p) in _LINKS.items() if now - t > LINK_TTL]:
            del _LINKS[k]
        _LINKS[token] = (now, ds_id, payload)
    return {"url": f"/api/datasets/{ds_id}/export/zip/{token}", "expires_in": int(LINK_TTL)}


@router.get("/{ds_id}/export/zip/{token}")
def export_zip_download(ds_id: str, token: str):
    with _LINKS_LOCK:
        entry = _LINKS.pop(token, None)          # single use
    if not entry or entry[1] != ds_id or time.time() - entry[0] > LINK_TTL:
        raise HTTPException(404, "This download link has expired — start the export again")
    return _zip_response(get_dataset_or_404(ds_id), entry[2])


@router.post("/{ds_id}/export/copy")
def export_copy(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    target = (payload.get("target_dir") or "").strip()
    if not target:
        raise HTTPException(400, "target_dir is required")
    try:
        ex.check_copy_target(ds_id, target)       # fail fast, before a job is queued
    except ValueError as e:
        raise HTTPException(400, str(e))
    f = _export_filters(payload)
    names = db.loads(row["classes"])
    job = jobs.submit("export_copy", lambda j: ex.copy_subset(j, ds_id, names, f, target, payload.get("resize"),
                                                               bool(payload.get("include_unlabeled", True)),
                                                               bool(payload.get("include_yaml", True))),
                      dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict()}


@router.post("/{ds_id}/export/list")
def export_list(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    f = filters_from(payload.get("filters") or {})
    text = ex.render_list(ds_id, f, payload.get("style") or "host")
    if payload.get("write"):
        fname = _safe_name(payload.get("filename") or "fovea-list", "list") + ".txt"
        out = Path(row["root"]) / fname
        used = {str(Path(s["list_file"])) for s in (db.loads(row["layout"], {}) or {}).get("sources", []) if s.get("list_file")}
        if str(out) in used:
            raise HTTPException(409, f"{fname} is one of this dataset's own image lists — choose another name")
        tmp = out.with_name(f".{fname}.fovea-tmp")
        tmp.write_text(text, encoding="utf-8")
        os.replace(tmp, out)
        return {"path": to_host(out), "lines": text.count("\n")}
    fname = _safe_name(payload.get("filename") or f"{row['id']}-list", "list") + ".txt"
    return PlainTextResponse(text, headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/{ds_id}/export/yaml")
def export_yaml(ds_id: str):
    row = get_dataset_or_404(ds_id)
    out = ex.write_data_yaml(row["root"], db.loads(row["classes"]), db.loads(row["layout"], {}))
    lay = db.loads(row["layout"], {})
    lay["data_yaml"] = out
    lay["data_yaml_host"] = to_host(out)
    db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), ds_id))
    return {"path": to_host(out)}
