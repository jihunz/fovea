"""Label snapshots: automatic before bulk / AI writes, manual versions, diff and restore."""
from __future__ import annotations

import time
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

from ..core import snapshots as snap
from ..jobs import jobs
from .common import get_dataset_or_404

router = APIRouter(prefix="/api/datasets", tags=["snapshots"])

SYNC_WAIT = 2.0     # a small manual snapshot finishes inside the request and returns its id


def _public(row: dict) -> dict:
    # `label`: the name people recognise an unnamed version by ('Class remap · 23:50'), as restores record it.
    return {**{k: row[k] for k in ("id", "name", "kind", "op", "n_files", "created_at")}, "label": snap.describe(row)}


def _snapshot_or_404(ds_id: str, sid: int) -> dict:
    row = snap.get_snapshot(ds_id, sid)
    if not row:
        raise HTTPException(404, "Snapshot not found")
    return row


@router.get("/{ds_id}/snapshots")
def list_snapshots(ds_id: str, limit: int = Query(200, ge=1, le=1000)):
    get_dataset_or_404(ds_id)
    # Retention: the newest keep_auto versions of more than small_files files, and separately the newest
    # keep_small smaller ones (a run of Objects key presses never pushes out a bulk / AI undo point).
    return {"snapshots": [_public(r) for r in snap.list_snapshots(ds_id, limit)], "keep_auto": snap.KEEP_AUTO,
            "keep_small": snap.KEEP_SMALL, "small_files": snap.SMALL_FILES}


@router.post("/{ds_id}/snapshots")
def create_snapshot(ds_id: str, payload: Optional[dict] = Body(None)):
    """Manual snapshot ("Save version") of every label file in the dataset."""
    get_dataset_or_404(ds_id)
    name = (payload or {}).get("name")
    if name is not None and not isinstance(name, str):
        raise HTTPException(400, "name must be a string")
    name = (name or "").strip()[:120] or time.strftime("Version %Y-%m-%d %H:%M")

    def run(job):
        job.update(message="Saving label files")
        sid = snap.snapshot_dataset(ds_id, name=name)
        row = snap.get_snapshot(ds_id, sid) or {}
        n = int(row.get("n_files") or 0)
        # Not job.update(): the version is saved, so a cancel arriving now must not mark the job cancelled.
        job.done, job.total, job.updated_at = n, n, time.time()
        job.message = f"Saved {n:,} label file{'s' if n != 1 else ''}"
        return {"snapshot_id": sid, "n_files": n}

    job = jobs.submit("snapshot", run, dataset_id=ds_id, message="Queued")
    deadline = time.time() + SYNC_WAIT
    while job.status in ("queued", "running") and time.time() < deadline:
        time.sleep(0.02)
    out = {"job": job.to_dict()}
    if job.status == "done" and job.result:
        out["snapshot_id"] = job.result["snapshot_id"]
    elif job.status == "error":
        raise HTTPException(500, job.error or "Snapshot failed")
    return out


@router.get("/{ds_id}/snapshots/{sid}")
def get_snapshot(ds_id: str, sid: int):
    get_dataset_or_404(ds_id)
    return {"snapshot": _public(_snapshot_or_404(ds_id, sid))}


@router.get("/{ds_id}/snapshots/{sid}/diff")
def snapshot_diff(ds_id: str, sid: int, offset: int = Query(0, ge=0), limit: int = Query(50, ge=1, le=500)):
    """What changed since the snapshot (snapshot -> label files on disk now)."""
    get_dataset_or_404(ds_id)
    _snapshot_or_404(ds_id, sid)
    return snap.diff(ds_id, sid, offset, limit)


@router.post("/{ds_id}/snapshots/{sid}/restore")
def restore_snapshot(ds_id: str, sid: int, payload: Optional[dict] = Body(None)):
    """Put the snapshot's label files back (optionally only for some images). Runs as a job; an automatic
    snapshot of the current files is taken first, so the restore can itself be undone.

    `mode`: 'full' (default; History "Restore version") or 'undo' (an op's own Undo: only the files the op wrote,
    and only while they still hold what it wrote — later edits are kept and counted in `changed_since`)."""
    get_dataset_or_404(ds_id)
    row = _snapshot_or_404(ds_id, sid)
    mode = (payload or {}).get("mode") or "full"
    if mode not in snap.RESTORE_MODES:
        raise HTTPException(400, "mode must be 'full' or 'undo'")
    image_ids = (payload or {}).get("image_ids")
    if image_ids is not None:
        if not isinstance(image_ids, list):
            raise HTTPException(400, "image_ids must be a list of image ids")
        try:
            image_ids = [int(x) for x in image_ids]
        except (TypeError, ValueError):
            raise HTTPException(400, "image_ids must be a list of image ids")
    active = jobs.active_for(ds_id, "scan")
    if active:
        raise HTTPException(409, "Indexing is running — wait for it to finish (or cancel it) before restoring")
    job = jobs.submit("restore", lambda j: snap.restore(ds_id, sid, image_ids, job=j, mode=mode), dataset_id=ds_id,
                      message=f"{'Undoing' if mode == 'undo' else 'Restoring'} “{snap.describe(row)}”")
    return {"job": job.to_dict()}


@router.delete("/{ds_id}/snapshots/{sid}")
def delete_snapshot(ds_id: str, sid: int):
    get_dataset_or_404(ds_id)
    _snapshot_or_404(ds_id, sid)
    try:
        snap.delete_snapshot(ds_id, sid)
    except snap.SnapshotBusy as e:
        raise HTTPException(409, detail={"message": str(e), "code": "snapshot_in_use"})
    return {"ok": True}
