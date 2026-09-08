from __future__ import annotations

import re
import time
from pathlib import Path

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


@router.post("/{ds_id}/export/zip")
def export_zip(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    f = filters_from(payload.get("filters") or {})
    names = db.loads(row["classes"])
    gen = ex.stream_zip(ds_id, names, f, payload.get("resize"), bool(payload.get("include_unlabeled", True)),
                        bool(payload.get("include_yaml", True)))
    fname = _safe_name(payload.get("filename") or f"{row['id']}-{int(time.time())}", "export") + ".zip"
    return StreamingResponse(gen, media_type="application/zip",
                             headers={"Content-Disposition": f'attachment; filename="{fname}"'})


@router.post("/{ds_id}/export/copy")
def export_copy(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    target = (payload.get("target_dir") or "").strip()
    if not target:
        raise HTTPException(400, "target_dir is required")
    f = filters_from(payload.get("filters") or {})
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
        out.write_text(text, encoding="utf-8")
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
