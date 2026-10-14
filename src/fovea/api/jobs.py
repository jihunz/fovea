from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse

from .. import db, security
from ..jobs import Job, jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


def provider_batch_of(job: Job) -> Optional[str]:
    """The cloud provider a cancel of this job would call with the owner's API key, or None. Cancelling an Economy
    run (provider batch API) cancels its batches at the provider; every other job only stops spending."""
    if not job.kind.startswith("ai_") or job.status not in ("queued", "running"):
        return None
    row = db.query_one("SELECT provider, params FROM ai_runs WHERE job_id=? ORDER BY id DESC LIMIT 1", (job.id,))
    if not row:
        return None
    params = db.loads(row["params"], {}) or {}
    if not isinstance(params, dict) or params.get("speed") != "economy":
        return None
    pid = row["provider"] or str(params.get("provider") or "")
    try:
        from ..ai import registry
        cloud = registry.is_cloud(pid)
    except Exception:  # noqa: BLE001 — cannot tell: treat it as the owner's cloud key (fail closed)
        cloud = True
    return pid if cloud else None


STOPS_BATCH = "this stops an Economy batch at the cloud provider."
# Relocating does not cancel an Economy run: it detaches the job (datasets._cancel_dataset_jobs) and, once the rescan
# has finished, reconnects to the same provider batch with the owner's key (datasets._reattach_economy). That
# reconnect is why a remote relocate stays refused.
RECONNECTS_BATCH = ("this pauses an Economy batch and reconnects to it at the cloud provider with Fovea's API key "
                    "once the files are re-indexed.")


def guard_cancel(request: Optional[Request], targets, action: str = "Cancel", effect: str = STOPS_BATCH) -> None:
    """403 when a request from another machine would act on a provider batch with the owner's key — the same rule
    as /api/ai/runs/{id}/cancel, for this route and dataset delete (both cancel the batch at the provider) and
    relocate (reconnects to it). `effect` says which, in the error. No request means an in-process call (tests,
    scripts), which the network cannot reach."""
    if request is None or security.ai_access_allowed(request):
        return
    for j in targets:
        if provider_batch_of(j):
            raise HTTPException(403, f"{action}: {effect} " + security.REMOTE_AI_MESSAGE)


def guard_dataset_cancel(request: Optional[Request], dataset_id: str, action: str) -> None:
    """guard_cancel for every live job of a dataset: deleting a dataset cancels them all; relocating it cancels
    all but its Economy runs, which it pauses and re-attaches (RECONNECTS_BATCH)."""
    if request is None or security.ai_access_allowed(request):
        return
    effect = RECONNECTS_BATCH if action == "Relocate" else STOPS_BATCH
    guard_cancel(request, jobs.active_all(dataset_id), action, effect)


@router.get("")
def list_jobs(dataset_id: Optional[str] = Query(None), kind: Optional[str] = Query(None),
              active: int = Query(0, description="1 = only queued/running jobs (cheap to poll)")):
    return {"jobs": [j.to_dict() for j in jobs.list(dataset_id, kind, active=bool(active))[:50]]}


@router.get("/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return {"job": j.to_dict()}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str, request: Request):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    guard_cancel(request, [j])
    j.cancel()
    return {"job": j.to_dict()}


TERMINAL = ("done", "error", "cancelled")


@router.get("/{job_id}/events")
async def job_events(job_id: str):
    """Server-sent job snapshots. Async on purpose: a synchronous generator that sleeps between changes
    pins a thread-pool worker for the job's whole life, and every sync endpoint shares that pool."""
    async def gen():
        last = None
        last_sent = time.monotonic()
        deadline = last_sent + 6 * 3600
        while time.monotonic() < deadline:
            job = jobs.get(job_id)
            if job is None:
                yield f"data: {json.dumps({'status': 'missing', 'id': job_id})}\n\n"
                return
            snap = job.to_dict()
            key = (snap["status"], snap["done"], snap["total"], snap["message"], snap["queue_position"],
                   json.dumps(snap["extra"], sort_keys=True, default=str))
            if key != last:
                yield f"data: {json.dumps(snap, default=str)}\n\n"
                last, last_sent = key, time.monotonic()
            elif time.monotonic() - last_sent > 15:
                yield ": keepalive\n\n"        # an SSE comment: keeps proxies from closing a quiet stream
                last_sent = time.monotonic()
            if snap["status"] in TERMINAL:
                return
            if jobs.shutting_down.is_set():
                # The server is stopping: end the stream so the shutdown is not held up by an open page.
                # The browser's EventSource reconnect then fails and the UI falls back to polling.
                return
            await asyncio.sleep(0.25)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
