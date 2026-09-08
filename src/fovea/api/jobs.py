from __future__ import annotations

import json
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


@router.get("/{job_id}/events")
def job_events(job_id: str):
    def gen():
        for snap in jobs.events(job_id):
            yield f"data: {json.dumps(snap)}\n\n"
    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})
