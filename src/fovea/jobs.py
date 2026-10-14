"""Tiny in-process background job manager with progress + SSE streaming."""
from __future__ import annotations

import logging
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional

log = logging.getLogger("fovea.jobs")


class JobCancelled(Exception):
    pass


class JobError(Exception):
    """An expected, user-facing failure (bad input, missing files, disk full). Its message is shown as is,
    and it is logged as one warning line instead of a traceback."""


# Failures the user can act on: their message is the whole story, without the exception class name.
EXPECTED_ERRORS = (JobError, ValueError, OSError)
# ...and those that are logged as one line: a ValueError may just as well be a bug, so it keeps its traceback.
QUIET_ERRORS = (JobError, OSError)


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
    pool: str = "default"
    created_at: float = field(default_factory=time.time)
    started_at: Optional[float] = None
    updated_at: float = field(default_factory=time.time)
    finished_at: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)   # job-specific live info, e.g. {"cost_usd": 0.12, "budget_usd": 5}
    _cancel: threading.Event = field(default_factory=threading.Event, repr=False)
    _manager: Any = field(default=None, repr=False)
    # (ai-core hook) called once when the job is cancelled before it started: its function never runs, so records
    # created for it up front (ai_runs rows) must be finished by someone else.
    on_unstarted_cancel: Optional[Callable[["Job"], None]] = field(default=None, repr=False)
    # Set before cancel() by whoever takes the job's work over (relocate: an Economy run is re-attached after its
    # rescan). The job stops, but its work is not over: economy's JobCancelled handler and runner.finish_unstarted
    # (a job cancelled while queued) leave its provider batches alone.
    detached: bool = field(default=False, repr=False)

    @property
    def progress(self) -> float:
        if self.status in ("done",):
            return 1.0
        if self.total <= 0:
            return 0.0
        return min(1.0, self.done / self.total)

    def update(self, done: Optional[int] = None, total: Optional[int] = None, message: Optional[str] = None,
               extra: Optional[Dict[str, Any]] = None) -> None:
        if done is not None:
            self.done = done
        if total is not None:
            self.total = total
        if message is not None:
            self.message = message
        if extra:
            self.extra = {**self.extra, **extra}
        self.updated_at = time.time()
        if self._cancel.is_set():
            raise JobCancelled()

    def check(self) -> None:
        """Raise JobCancelled when a cancel was requested (for loops that do not report progress)."""
        if self._cancel.is_set():
            raise JobCancelled()

    def cancel(self) -> None:
        self._cancel.set()
        # A queued job never reaches a worker check-point: finish it now so its status, the dataset banner
        # and active_for() agree immediately instead of hours later when a worker picks it up.
        if self.status == "queued":
            self.status = "cancelled"
            self.message = "Cancelled before it started"
            self.finished_at = self.updated_at = time.time()
            cb, self.on_unstarted_cancel = self.on_unstarted_cancel, None     # (ai-core hook)
            if cb is not None:
                try:
                    cb(self)
                except Exception:  # noqa: BLE001 — a hook problem must never break a cancel
                    log.exception("Job %s (%s): cancel hook failed", self.kind, self.id)

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def queue_position(self) -> Optional[int]:
        """1-based place among the queued jobs of the same pool, None once running or finished."""
        if self.status != "queued" or self._manager is None:
            return None
        return self._manager.queue_position(self)

    def to_dict(self) -> dict:
        end = self.finished_at or time.time()
        return {
            "id": self.id, "kind": self.kind, "dataset_id": self.dataset_id, "status": self.status,
            "done": self.done, "total": self.total, "progress": round(self.progress, 4),
            "message": self.message, "result": self.result, "error": self.error, "extra": self.extra,
            "created_at": self.created_at, "started_at": self.started_at, "updated_at": self.updated_at,
            "finished_at": self.finished_at, "queue_position": self.queue_position(),
            # Time spent working, not waiting in the queue: the UI's elapsed clock starts from this.
            "elapsed": round(end - self.started_at, 2) if self.started_at else 0.0,
            "waited": round((self.started_at or end) - self.created_at, 2),
        }


def _user_facing(e: BaseException) -> bool:
    """An error class that declares its message user-facing (ai.base.ProviderError: bad key, quota, refusal...)."""
    return bool(getattr(type(e), "user_facing", False))


def _error_text(e: BaseException) -> str:
    from .security import scrub
    text = str(e) if (isinstance(e, EXPECTED_ERRORS) or _user_facing(e)) and str(e) else f"{type(e).__name__}: {e}"
    return scrub(text)[:2000]


class JobManager:
    # Separate pools so a long AI batch (minutes to hours, network-bound) can never starve a scan or an export.
    # ai-core hook: 'ai_wait' runs Economy (provider batch API) jobs, which mostly sleep for up to 24 h between polls.
    POOLS = {"default": 3, "ai": 2, "ai_wait": 16}

    def __init__(self, workers: Optional[Dict[str, int]] = None):
        self._jobs: Dict[str, Job] = {}
        self._lock = threading.Lock()
        self.shutting_down = threading.Event()
        sizes = {**self.POOLS, **(workers or {})}
        self._pools = {name: ThreadPoolExecutor(max_workers=n, thread_name_prefix=f"fovea-{name}") for name, n in sizes.items()}

    def submit(self, kind: str, fn: Callable[[Job], Any], dataset_id: str = "", message: str = "",
               pool: str = "default") -> Job:
        pool = pool if pool in self._pools else "default"
        job = Job(id=uuid.uuid4().hex[:12], kind=kind, dataset_id=dataset_id, message=message, pool=pool, _manager=self)
        with self._lock:
            self._jobs[job.id] = job
            self._gc()
        if self.shutting_down.is_set():
            job.cancel()
            return job

        def _run():
            if job.cancelled:           # cancelled while queued: cancel() already finished it
                return
            job.status = "running"
            job.started_at = job.updated_at = time.time()
            try:
                job.result = fn(job)
                job.status = "done"
            except JobCancelled:
                job.status = "cancelled"
            except Exception as e:  # noqa: BLE001
                job.status = "error"
                job.error = _error_text(e)
                if isinstance(e, QUIET_ERRORS) or _user_facing(e):
                    log.warning("Job %s (%s) failed: %s", job.kind, job.id, job.error)
                else:
                    log.exception("Job %s (%s) crashed", job.kind, job.id)
            finally:
                job.finished_at = time.time()
                job.updated_at = job.finished_at

        try:
            self._pools[pool].submit(_run)
        except RuntimeError:            # executor already shut down (server exiting)
            job.cancel()
        return job

    def get(self, job_id: str) -> Optional[Job]:
        return self._jobs.get(job_id)

    def _snapshot(self) -> List[Job]:
        """Every job, copied under the lock: submit() inserts (and _gc removes) from other threads, and
        iterating the live dict meanwhile raises 'dictionary changed size during iteration'."""
        with self._lock:
            return list(self._jobs.values())

    def list(self, dataset_id: Optional[str] = None, kind: Optional[str] = None, active: bool = False) -> List[Job]:
        jobs = self._snapshot()
        if dataset_id is not None:
            jobs = [j for j in jobs if j.dataset_id == dataset_id]
        if kind is not None:
            kinds = set(kind.split(",")) if isinstance(kind, str) else set(kind)
            jobs = [j for j in jobs if j.kind in kinds]
        if active:
            jobs = [j for j in jobs if j.status in ("queued", "running")]
        return sorted(jobs, key=lambda j: j.created_at, reverse=True)

    def active_for(self, dataset_id: str, kind: Optional[str] = None) -> Optional[Job]:
        """The newest queued/running job of this dataset (and kind). Use active_all() to act on every one."""
        for j in self.list(dataset_id, kind):
            if j.status in ("queued", "running"):
                return j
        return None

    def active_all(self, dataset_id: str, kinds: Optional[Iterable[str]] = None) -> List[Job]:
        kinds = set(kinds) if kinds else None
        return [j for j in self.list(dataset_id) if j.status in ("queued", "running") and (kinds is None or j.kind in kinds)]

    def cancel_all(self, dataset_id: str, kinds: Optional[Iterable[str]] = None, wait: float = 0.0) -> List[Job]:
        """Cancel every queued/running job of a dataset (delete / relocate must not leave one behind).
        With wait > 0, block up to that many seconds for running jobs to stop."""
        hit = self.active_all(dataset_id, kinds)
        for j in hit:
            j.cancel()
        if wait > 0:
            self.wait(hit, wait)
        return hit

    @staticmethod
    def wait(jobs: Iterable[Job], timeout: float) -> bool:
        """True when every job finished within timeout seconds."""
        deadline = time.monotonic() + timeout
        pending = list(jobs)
        while pending and time.monotonic() < deadline:
            pending = [j for j in pending if j.status in ("queued", "running")]
            if pending:
                time.sleep(0.05)
        return not [j for j in pending if j.status in ("queued", "running")]

    def queue_position(self, job: Job) -> Optional[int]:
        queued = sorted((j for j in self._snapshot() if j.status == "queued" and j.pool == job.pool),
                        key=lambda j: j.created_at)
        for i, j in enumerate(queued):
            if j is job:
                return i + 1
        return None

    def begin_shutdown(self) -> None:
        """Called as soon as the server is asked to stop: SSE streams end and every job is cancelled, so
        Ctrl+C / docker stop finish in seconds instead of waiting for a long export or AI batch."""
        if self.shutting_down.is_set():
            return
        self.shutting_down.set()
        for j in self._snapshot():
            if j.status in ("queued", "running"):
                j.cancel()

    def shutdown(self, wait: float = 5.0) -> None:
        self.begin_shutdown()
        running = [j for j in self._snapshot() if j.status == "running"]
        for p in self._pools.values():
            p.shutdown(wait=False, cancel_futures=True)
        if running and not self.wait(running, wait):
            log.warning("%d job(s) still stopping at shutdown: %s", len(running),
                        ", ".join(f"{j.kind}:{j.id}" for j in running if j.status == "running"))

    def _gc(self, keep: int = 200) -> None:
        """Drop the oldest finished jobs. Called by submit() with self._lock held."""
        if len(self._jobs) <= keep:
            return
        finished = [j for j in self._jobs.values() if j.status in ("done", "error", "cancelled")]
        finished.sort(key=lambda j: j.finished_at or 0)
        for j in finished[: len(self._jobs) - keep]:
            self._jobs.pop(j.id, None)


jobs = JobManager()
