from __future__ import annotations

import asyncio
import json
import time
from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from ..jobs import jobs

router = APIRouter(prefix="/api/jobs", tags=["jobs"])


@router.get("")
def list_jobs(dataset_id: Optional[str] = Query(None), kind: Optional[str] = Query(None)):
    return {"jobs": [j.to_dict() for j in jobs.list(dataset_id, kind)[:50]]}


@router.get("/{job_id}")
def get_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
    return {"job": j.to_dict()}


@router.post("/{job_id}/cancel")
def cancel_job(job_id: str):
    j = jobs.get(job_id)
    if not j:
        raise HTTPException(404, "Job not found")
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
            key = (snap["status"], snap["done"], snap["total"], snap["message"])
            if key != last:
                yield f"data: {json.dumps(snap)}\n\n"
                last, last_sent = key, time.monotonic()
            elif time.monotonic() - last_sent > 15:
                yield ": keepalive\n\n"        # an SSE comment: keeps proxies from closing a quiet stream
                last_sent = time.monotonic()
            if snap["status"] in TERMINAL:
                return
            await asyncio.sleep(0.25)
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
