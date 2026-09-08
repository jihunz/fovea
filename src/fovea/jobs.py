"""Tiny in-process background job manager with progress + SSE streaming."""
from __future__ import annotations

import threading
import time
import traceback
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterator, List, Optional


class JobCancelled(Exception):
    pass


@dataclass
class Job:
    id: str
    kind: str
    dataset_id: str = ""
    status: str = "queued"  # queued|running|done|error|cancelled
    done: int = 0
    total: int = 0
    message: str = ""
    result: Any = None
    error: Optional[str] = None
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)

    @property
    def progress(self) -> float:
        if self.status in ("done",):
            return 1.0
        if self.total <= 0:
            return 0.0
        return min(1.0, self.done / self.total)

    def update(self, done: Optional[int] = None, total: Optional[int] = None, message: Optional[str] = None) -> None:
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        self.updated_at = time.time()
        if self._cancel.is_set():
            raise JobCancelled()

    def cancel(self) -> None:
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def to_dict(self) -> dict:
        return {
            "id": self.id, "kind": self.kind, "dataset_id": self.dataset_id, "status": self.status,
            "done": self.done, "total": self.total, "progress": round(self.progress, 4),
            "message": self.message, "result": self.result, "error": self.error,
            "created_at": self.created_at, "updated_at": self.updated_at, "finished_at": self.finished_at,
            "elapsed": round((self.finished_at or time.time()) - self.created_at, 2),
        }


class JobManager:
    def __init__(self, workers: int = 3):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self._pool = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fovea-job")

    def submit(self, kind: str, fn: Callable[[Job], Any], dataset_id: str = "", message: str = "") -> Job:
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, dataset_id=dataset_id, message=message)
        with self._lock:
            self._jobs[job.id] = job
            self._gc()

        def _run():
            job.status = "running"
            job.updated_at = time.time()
            try:
                job.result = fn(job)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = f"{type(e).__name__}: {e}"
                traceback.print_exc()
            finally:
                job.finished_at = time.time()
                job.updated_at = job.finished_at

        self._pool.submit(_run)
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def list(self, dataset_id: Optional[str] = None, kind: Optional[str] = None) -> List[Job]:
        jobs = list(self._jobs.values())
        if dataset_id is not None:
            jobs = [j for j in jobs if j.dataset_id == dataset_id]
        if kind is not None:
            jobs = [j for j in jobs if j.kind == kind]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def active_for(self, dataset_id: str, kind: Optional[str] = None) -> Optional[Job]:
        for j in self.list(dataset_id, kind):
            if j.status in ("queued", "running"):
                return j
        return None

    def events(self, job_id: str, interval: float = 0.25, timeout: float = 3600 * 6) -> Iterator[dict]:
        """Yield job snapshots until the job reaches a terminal state."""
        start = time.time()
        last = None
        while time.time() - start < timeout:
            job = self.get(job_id)
            if job is None:
                yield {"status": "missing", "id": job_id}
                return
            snap = job.to_dict()
            key = (snap["status"], snap["done"], snap["total"], snap["message"])
            if key != last:
                yield snap
                last = key
            if snap["status"] in ("done", "error", "cancelled"):
                return
            time.sleep(interval)

    def _gc(self, keep: int = 200) -> None:
        if len(self._jobs) <= keep:
            return
        finished = [j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")]
        finished.sort(key=lambda j: j.finished_at or 0)
        for j in finished[: len(self._jobs) - keep]:
            self._jobs.pop(j.id, None)


jobs = JobManager()
