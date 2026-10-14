"""Economy mode: a batch run through the provider's own batch API (Anthropic Message Batches, OpenAI /v1/batches,
Gemini batchGenerateContent) — about half the price, results within 24 h.

One run (``ai_runs.kind='batch'``, ``params.speed='economy'``) lives in one job (kind ``ai_batch``) that runs in its own
``ai_wait`` pool, so a day-long wait never holds one of the two AI workers:

1. submit — the images go to ``Provider.batch_submit`` in chunks of :data:`SUBMIT_CHUNK`. Each provider batch id is
   recorded in ``params.provider_batches`` (``params.batch_id`` = the first) and its image ids in kv
   ``ai.economy.<run_id>`` before anything else happens, so a restart can always find them again.
2. wait — ``batch_status`` every :data:`POLL_S` seconds (60); the job message, ``job.extra.batch`` and
   ``run.result.batch`` (``{id, state, counts, batches, submitted_at, checked_at, next_check_at, filed}``) show the
   provider's progress. The run's status stays ``running``.
3. ingest — ``batch_results`` → the same :class:`runner.BatchSink` as the synchronous path (proposals, or a write target
   after a snapshot, label-write hooks), costed at batch prices by the provider.

Server shutdown ends the job WITHOUT touching the provider batches; :func:`resume` (startup scan) re-attaches to every
Economy run still ``running``. A user cancel cancels the provider batches too, then collects whatever the provider
had already finished (those requests are billed either way).
"""
from __future__ import annotations

import inspect
import logging
import math
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from .. import db
from ..jobs import Job, JobCancelled, jobs
from . import pipeline, registry, runner, store
from .base import ProviderError, Usage
from .base import billed_usage  # ai-cloud hook: failed batch answers that were billed
from .secrets import scrub

log = logging.getLogger("fovea.ai")

POLL_S = float(os.environ.get("FOVEA_AI_BATCH_POLL_S") or 60)
GIVE_UP_S = 26 * 3600               # provider batches expire after 24 h: polls failing past that end the run
SUBMIT_CHUNK = 1000                 # images per batch_submit call (providers split further by their own size limits)
WAIT_POOL = "ai_wait"
WAIT_WORKERS = 16
STARTED_AT = time.time()            # runs created by this process are never "resumed" by the startup scan
_pool_lock = threading.Lock()
_resumed: Dict[str, bool] = {}


def kv_key(run_id: int) -> str:
    return f"ai.economy.{int(run_id)}"


def available(provider: str) -> Tuple[bool, Optional[str]]:
    """Can ``provider`` run Economy batches here?"""
    try:
        p = registry.get(provider)
    except KeyError:
        return False, "Unknown provider"
    if not (getattr(p, "capabilities", {}) or {}).get("batch_api"):
        return False, f"{getattr(p, 'name', provider) or provider} has no batch API in Fovea — use Now"
    return True, None


def _pool() -> str:
    """The dedicated pool for jobs that mostly wait (``JobManager.POOLS['ai_wait']``; added on first use to a job
    manager built without it)."""
    pools = getattr(jobs, "_pools", None)
    if not isinstance(pools, dict):
        return "ai"
    if WAIT_POOL not in pools:
        with _pool_lock:
            if WAIT_POOL not in pools:
                pools[WAIT_POOL] = ThreadPoolExecutor(max_workers=WAIT_WORKERS, thread_name_prefix=f"fovea-{WAIT_POOL}")
    return WAIT_POOL


def _fmt_age(seconds: float) -> str:
    s = max(0, int(seconds))
    if s < 90:
        return f"{s} s"
    if s < 5400:
        return f"{s // 60} min"
    return f"{s // 3600} h {s % 3600 // 60:02d} min"


# ------------------------------------------------------------------ start

def start(ds_row: dict, image_ids: List[int], provider: str, model: str, conf, cls_ids, options: dict, target: str,
          budget: Optional[float], budget_from_cap: bool = False) -> Tuple[Job, int]:
    """Validate and queue an Economy run (called by runner.start_batch with speed='economy'). ``budget_from_cap``:
    the budget is what the 30-day spend cap leaves, not one the user set."""
    options = dict(options or {})
    if (options.get("pipeline") or "") == "two_stage":
        raise ProviderError("Economy (batch API) runs one provider — use Now for a two-stage pipeline", "invalid")
    ok, why = available(provider)
    if not ok:
        raise ProviderError(why or "No batch API", "invalid")
    pipeline.plan_detect(ds_row, provider, model, conf, cls_ids, options)        # validates classes / options now
    per_image = runner._per_image_hi(ds_row, provider, model, options, batch=True)  # noqa: SLF001
    stopped = None
    ids = list(image_ids)
    if budget is not None and per_image > 0:
        fit = int(math.floor((budget + 1e-9) / per_image))
        if fit <= 0:
            raise ProviderError(f"The budget ({runner._fmt_usd(budget)}) does not cover one image at the batch price "  # noqa: SLF001
                                f"({runner._fmt_usd(per_image)})", "quota")
        if fit < len(ids):
            ids, stopped = ids[:fit], ("cap" if budget_from_cap else "budget")
    keep_conf = bool(options.pop("keep_conf", False))
    params = {"provider": provider, "model": model, "conf": conf, "classes": cls_ids, "options": options,
              "target": target, "budget_usd": budget, "keep_conf": keep_conf, "images": len(ids), "speed": "economy",
              "requested_images": len(image_ids), "provider_batches": [], "batch_id": None,
              # what each image may cost at the batch price (high estimate): until the provider bills the batch,
              # images × this is held against the spend cap (runner.reserved_usd), so runs started meanwhile see it
              "per_image_usd": round(per_image, 8)}
    run_id = store.create_run(ds_row["id"], "batch", provider, model, params)
    db.kv_set(kv_key(run_id), {"image_ids": ids, "batches": {}})
    store.update_result(run_id, {"speed": "economy", "stopped": stopped, "run_id": run_id, "images_done": 0,
                                 "cost_usd": 0.0, "budget_usd": budget})
    return _launch(run_id, ds_row["id"], "Submitting to the batch API…"), run_id


def _launch(run_id: int, ds_id: str, message: str) -> Job:
    job = jobs.submit("ai_batch", lambda j: _lifecycle(j, run_id), dataset_id=ds_id, message=message, pool=_pool())
    store.set_run_job(run_id, job.id)
    runner.watch_unstarted(job, [run_id])
    return job


# ------------------------------------------------------------------ lifecycle

class _Run:
    def __init__(self, job: Job, run_id: int):
        self.job = job
        self.run_id = run_id
        run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
        if not run:
            raise ProviderError("The run was deleted", "invalid")
        self.run = store.run_public(run)
        self.params = self.run["params"] if isinstance(self.run["params"], dict) else {}
        self.ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (self.run["dataset_id"],))
        if not self.ds_row:
            raise ProviderError("The dataset was removed", "invalid")
        self.provider_id = self.params.get("provider") or self.run["provider"]
        self.p = registry.get(self.provider_id)        # KeyError: provider gone (mock off, module missing)
        self.pname = getattr(self.p, "name", "") or self.provider_id
        self.state = db.kv_get(kv_key(run_id), {}) or {}
        self.state.setdefault("image_ids", [])
        self.state.setdefault("batches", {})
        self.batches: List[dict] = list(self.params.get("provider_batches") or [])
        res = self.run.get("result") if isinstance(self.run.get("result"), dict) else {}
        self.stats: Dict[str, Any] = {"cost_usd": 0.0, "budget_usd": self.params.get("budget_usd"), "images_done": 0,
                                      "boxes": 0, "errors": 0, "skipped": 0, "duplicates": 0, "written": 0,
                                      "expired": 0, "canceled": 0, "run_id": run_id, "last_error": None,
                                      "warnings": [], "speed": "economy", "not_submitted": 0, "processed": [],
                                      "conflicts": 0, "poll_error": None}
        self.stats.update({k: v for k, v in res.items() if k != "processed" and (k in self.stats or k in (
            "stopped", "snapshot_id", "snapshot_ids"))})
        for a, b in res.get("image_ranges") or []:          # images already read before a restart
            self.stats["processed"].extend(range(int(a), int(b) + 1))
        self.sink = runner.BatchSink(self.ds_row, run_id, self.params.get("target") or "proposals",
                                     bool(self.params.get("keep_conf")), self.stats)
        self.total = len(self.state["image_ids"])
        self._plan = None
        self._seen: Dict[str, set] = {}
        self.tally = runner.WarnTally()
        self.fatal_since: Optional[float] = None       # first of a run of fatal-coded poll errors

    @property
    def plan(self):
        if self._plan is None:
            pr = self.params
            self._plan = pipeline.plan_detect(self.ds_row, self.provider_id, pr.get("model") or self.run["model"],
                                              pr.get("conf"), pr.get("classes"), dict(pr.get("options") or {}))
        return self._plan

    # ---- persistence
    def save(self, summary: Optional[dict] = None) -> None:
        if self._seen:
            self.state["seen"] = {k: sorted(v) for k, v in self._seen.items()}
            db.kv_set(kv_key(self.run_id), self.state)
        store.update_params(self.run_id, {"provider_batches": self.batches,
                                          "batch_id": self.batches[0]["id"] if self.batches else None})
        if summary is not None:
            self.stats["batch"] = summary
        res = {k: v for k, v in self.stats.items() if k != "processed"}
        res["image_ranges"] = runner.id_ranges(self.stats["processed"])
        res["submitted"] = sum(int(b.get("n") or 0) for b in self.batches)
        store.update_result(self.run_id, res)

    def submitted_ids(self) -> set:
        return {int(i) for ids in self.state["batches"].values() for i in ids}

    def seen(self, batch_id: str) -> set:
        """custom_ids of ``batch_id`` already filed (persisted with the run's kv state)."""
        cur = self._seen.get(batch_id)
        if cur is None:
            cur = self._seen[batch_id] = set(str(x) for x in (self.state.get("seen") or {}).get(batch_id, []))
        return cur

    # ---- phases
    def submit(self, resume: bool) -> None:
        if resume:
            # A crash between "provider accepted the batch" and "id recorded" cannot be told apart from "never sent":
            # never resubmit on resume (that could bill twice) — report what was not sent instead.
            done = self.submitted_ids()
            self.stats["not_submitted"] = len([i for i in self.state["image_ids"] if int(i) not in done])
            return
        ids = [int(i) for i in self.state["image_ids"]]
        plan = self.plan
        sent = 0
        for k in range(0, len(ids), SUBMIT_CHUNK):
            self.job.check()
            chunk = ids[k:k + SUBMIT_CHUNK]
            rows = {r["id"]: r for c in store.chunked(chunk) for r in db.query(
                f"SELECT * FROM images WHERE dataset_id=? AND id IN ({','.join('?' * len(c))})", [self.ds_row["id"], *c])}
            items, item_ids = [], []
            for iid in chunk:
                img = rows.get(iid)
                why = self.sink.skip_reason(img)
                if why is None and not os.path.isfile(img["abs_path"]):
                    why = "missing"
                if why:
                    self.stats["skipped"] += 1
                    continue
                err = pipeline.pixel_error(img, self.provider_id)
                if err is not None:                  # too large to decode: one image's error, not the whole batch's
                    self.stats["errors"] += 1
                    self.stats["last_error"] = scrub(str(err))
                    continue
                items.append({"custom_id": str(iid), "request": plan.request(img)})
                item_ids.append(iid)
            if not items:
                continue
            bid = self.p.batch_submit(items)
            if not isinstance(bid, str) or not bid:
                raise ProviderError(f"{self.pname} returned no batch id", "bad_output")
            self.state["batches"][bid] = item_ids
            db.kv_set(kv_key(self.run_id), self.state)
            self.batches.append({"id": bid, "n": len(item_ids), "state": "running", "counts": {}, "ingested": False,
                                 "submitted_at": time.time()})
            self.save()
            sent += len(item_ids)
            self.job.update(0, self.total, f"Submitted {sent:,} of {self.total:,} images to the {self.pname} batch API",
                            extra=self.extra())

    def poll_once(self) -> dict:
        """Check every unfinished provider batch; ingest those that ended. Returns the provider summary."""
        counts: Dict[str, int] = {}
        states = []
        poll_error: Optional[ProviderError] = None
        poll_failed = False
        for b in self.batches:
            if b.get("ingested"):
                states.append("ended")
                for k, v in (b.get("counts") or {}).items():
                    counts[k] = counts.get(k, 0) + int(v or 0)
                continue
            try:
                st = self.p.batch_status(b["id"])
            except ProviderError as e:
                if e.fatal and (e.status == 404 or "unknown batch" in str(e).lower()):
                    b.update(state="failed", ingested=True, error=scrub(str(e))[:300])
                    self.stats["errors"] += int(b.get("n") or 0)
                    self.stats["last_error"] = scrub(str(e))
                    continue
                # A rejected key, a billing hold or a hiccup: the batch is already at the provider and keeps running
                # (and billing) there — keep waiting, with a longer pause, rather than abandon it. Cancel stops it.
                self.stats["last_error"] = self.stats["poll_error"] = scrub(str(e))
                poll_failed = True
                if e.fatal:
                    poll_error = e
                states.append(b.get("state") or "running")
                continue
            b["state"] = st.get("state") or "running"
            b["counts"] = _counts(st)
            if b["state"] in ("ended", "failed"):
                self.ingest(b)
            states.append(b["state"])
            for k, v in b["counts"].items():
                counts[k] = counts.get(k, 0) + v
        if poll_error is not None:
            self.fatal_since = self.fatal_since or time.time()
            first = min((x.get("submitted_at") or time.time() for x in self.batches), default=time.time())
            if time.time() - first > GIVE_UP_S:
                raise ProviderError(f"{scrub(str(poll_error))} — the provider batch could not be checked for "
                                    f"{_fmt_age(time.time() - self.fatal_since)}", poll_error.code)
        else:
            self.fatal_since = None
        if not poll_failed and self.stats.get("poll_error"):
            # Polling works again: a failed check was a wait, not an error of the run — no stale warning under "Done".
            if self.stats.get("last_error") == self.stats["poll_error"]:
                self.stats["last_error"] = None
            self.stats["poll_error"] = None
        summary = self.summary(states, counts)
        if poll_error is not None:
            summary["error"] = scrub(str(poll_error))[:300]
            summary["next_check_at"] = time.time() + self.next_poll_s()
        self.save(summary)
        return summary

    def next_poll_s(self) -> float:
        """POLL_S normally; while the provider keeps refusing (key, billing) back off up to 15 minutes."""
        if self.fatal_since is None:
            return POLL_S
        return min(max(POLL_S, 900.0), POLL_S * 2 ** min(6, int((time.time() - self.fatal_since) // max(POLL_S, 1)) + 1))

    def summary(self, states: Optional[List[str]] = None, counts: Optional[Dict[str, int]] = None) -> dict:
        """``{id, state, counts, batches, submitted_at, checked_at, next_check_at, filed}`` over all provider batches."""
        now = time.time()
        if states is None:
            states = [b.get("state") or "running" for b in self.batches]
        if counts is None:
            counts = {k: sum(int((b.get("counts") or {}).get(k) or 0) for b in self.batches) for k in COUNT_KEYS}
        return {"id": self.batches[0]["id"] if self.batches else None,
                "state": "ended" if all(s in ("ended", "failed") for s in states) else
                ("canceling" if "canceling" in states else "running"),
                "counts": counts, "batches": len(self.batches),
                "submitted_at": min((b.get("submitted_at") or now for b in self.batches), default=now),
                "checked_at": now, "next_check_at": now + POLL_S, "filed": int(self.stats.get("images_done") or 0)}

    def ingest(self, b: dict, quiet: bool = False) -> None:
        """Read one ended provider batch into proposals / label files (idempotent per batch)."""
        if b.get("ingested"):
            return
        if b.get("state") == "failed":
            b["ingested"] = True
            self.stats["errors"] += int(b.get("n") or 0)
            self.stats["last_error"] = f"{self.pname} rejected the batch" + (f": {b['error']}" if b.get("error") else "")
            return
        ids = [int(i) for i in self.state["batches"].get(b["id"], [])]
        rows = {r["id"]: r for c in store.chunked(ids) for r in db.query(
            f"SELECT * FROM images WHERE dataset_id=? AND id IN ({','.join('?' * len(c))})", [self.ds_row["id"], *c])}
        plan = self.plan
        items = [{"custom_id": str(i), "request": plan.request(rows[i])} for i in ids if i in rows]
        if self.sink.target != "proposals" and not b.get("snapshot_id"):
            sid = self.sink.snapshot([i for i in ids if i in rows])
            if sid:
                b["snapshot_id"] = sid
                self.stats.setdefault("snapshot_ids", []).append(sid)
                self.stats.setdefault("snapshot_id", sid)
        # Results already filed (a cancel or a server stop can interrupt reading): read again from the start, skip
        # those — their cost, boxes and label writes are booked exactly once.
        seen = self.seen(b["id"])
        usage = Usage()
        n = 0
        try:
            for cid, res in _results(self.p, b["id"], items):
                n += 1
                if str(cid) in seen:
                    continue
                seen.add(str(cid))
                try:
                    iid = int(str(cid))
                except ValueError:
                    continue
                img = rows.get(iid)
                if img is None:
                    self.stats["skipped"] += 1
                    continue
                if isinstance(res, BaseException):
                    spent = billed_usage(res)    # ai-cloud hook: a refused / unreadable answer is billed (batch price)
                    if spent is not None:
                        usage.add(spent)
                        store.add_usage(self.run_id, spent, images=0)
                        self.stats["cost_usd"] = round(float(self.stats["cost_usd"] or 0) + float(spent.cost_usd or 0), 6)
                    code = getattr(res, "code", "")
                    if code in ("expired", "canceled"):
                        self.stats[code] += 1
                    else:
                        self.stats["errors"] += 1
                        self.stats["last_error"] = scrub(str(res))
                    continue
                out = plan.finish(res, time.time())
                ok = False
                try:
                    if self.sink.skip_reason(img):
                        self.stats["skipped"] += 1
                    else:
                        self.tally.add(out.get("warnings"))
                        self.sink.apply(iid, img, out)
                        ok = True
                        self.stats["processed"].append(iid)
                        self.stats["images_done"] += 1
                except OSError as e:          # one unwritable label file: count it, keep reading
                    self.stats["errors"] += 1
                    self.stats["write_errors"] = int(self.stats.get("write_errors") or 0) + 1
                    self.stats["last_error"] = runner.write_error_text(img.get("label_path"), e)
                finally:
                    usage.add(out["usage"])
                    store.add_usage(self.run_id, out["usage"], images=1, boxes=len(out["preds"]) if ok else 0)
                    self.stats["cost_usd"] = round(float(self.stats["cost_usd"] or 0)
                                                   + float(out["usage"].cost_usd or 0), 6)
                if len(seen) % 200 == 0:
                    self.save()               # keep the filed set on disk: a restart resumes without double counting
                if n % 50 == 0 and not quiet:
                    self.stats["warnings"] = self.tally.lines()
                    self.job.update(done=self.done_count(),
                                    message=f"Reading {self.pname} batch results · {n:,}/{len(items):,}",
                                    extra=self.extra())
        finally:
            self.stats["warnings"] = self.tally.lines()
            self.sink.announce()
            self.save()
        b["ingested"] = True
        b["ingested_at"] = time.time()
        self.save()

    def done_count(self) -> int:
        return sum(int(b.get("n") or 0) for b in self.batches if b.get("ingested")) + int(self.stats["skipped"] or 0)

    def extra(self) -> dict:
        return {k: v for k, v in self.stats.items() if k != "processed"}

    def wait_message(self, summary: dict) -> str:
        c = summary.get("counts") or {}
        n = sum(int(b.get("n") or 0) for b in self.batches)
        done = int(c.get("succeeded", 0)) + int(c.get("errored", 0)) + int(c.get("canceled", 0)) + int(c.get("expired", 0))
        first = min((b.get("submitted_at") or time.time() for b in self.batches), default=time.time())
        parts = [f"Economy batch at {self.pname}", f"{min(done, n):,} of {n:,} back",
                 f"sent {_fmt_age(time.time() - first)} ago"]
        if self.stats.get("images_done"):
            parts.append(f"{self.stats['images_done']:,} read")
        return " · ".join(parts)

    def sleep(self, seconds: float) -> None:
        end = time.time() + max(0.0, seconds)
        while True:
            self.job.check()
            left = end - time.time()
            if left <= 0:
                return
            time.sleep(min(1.0, left))

    def cancel_provider(self) -> None:
        """User cancel: stop the provider batches, then collect whatever had already finished (billed anyway)."""
        open_ = [b for b in self.batches if not b.get("ingested")]
        for b in open_:
            try:
                self.p.batch_cancel(b["id"])
                b["state"] = "canceling"
            except Exception as e:  # noqa: BLE001 — best effort; the provider expires it within 24 h anyway
                self.stats["last_error"] = scrub(f"Could not cancel {b['id']}: {e}")
        deadline = time.time() + 12
        while open_ and time.time() < deadline:
            for b in list(open_):
                try:
                    st = self.p.batch_status(b["id"])
                except Exception:  # noqa: BLE001
                    continue
                b["state"] = st.get("state") or b.get("state")
                b["counts"] = _counts(st)
                if b["state"] in ("ended", "failed"):
                    try:
                        self.ingest(b, quiet=True)
                    except Exception as e:  # noqa: BLE001
                        self.stats["last_error"] = scrub(f"{type(e).__name__}: {e}")
                        b["ingested"] = True
                    open_.remove(b)
            if open_:
                time.sleep(1.0)
        self.save(self.summary())


COUNT_KEYS = ("processing", "succeeded", "errored", "canceled", "expired")


def _counts(st: dict) -> Dict[str, int]:
    """The provider's per-request counts, without a 'total' (the UI sums the parts)."""
    c = st.get("counts") or {}
    return {k: int(c.get(k) or 0) for k in COUNT_KEYS}


def _results(p, batch_id: str, items: List[dict]):
    """``batch_results`` with the request list when the provider takes it (decode without provider-side state)."""
    fn = p.batch_results
    try:
        takes_items = "items" in inspect.signature(fn).parameters
    except (TypeError, ValueError):
        takes_items = False
    return fn(batch_id, items=items) if takes_items else fn(batch_id)


def _lifecycle(job: Job, run_id: int, resume: bool = False) -> dict:
    try:
        r = _Run(job, run_id)
    except KeyError:
        raise ProviderError("The provider of this Economy run is not available right now — it is picked up again at the "
                            "next start", "unavailable") from None
    except ProviderError as e:
        store.finish_run(run_id, "error", error=str(e))
        raise
    status, fatal = "done", None
    waits = runner.JobWaits(job)

    def on_wait(seconds: int, reason: str) -> None:        # provider retries while submitting / polling / reading
        waits.listener(seconds, reason)
        w = waits.text()
        base = (job.message or "").split(" · ", 1)[-1] if "retrying in" in (job.message or "") else job.message
        job.update(message=f"{w} · {base}" if w else base)
    vlm = runner._vlm()                                     # noqa: SLF001
    ctx = vlm.wait_listener(on_wait) if (vlm is not None and hasattr(vlm, "wait_listener")) else None
    if ctx is not None:
        ctx.__enter__()
    try:
        store.set_status(run_id, "running")
        r.submit(resume or bool(r.batches))
        if not r.batches:
            raise ProviderError("Nothing was submitted (no images left to label)", "invalid")
        while True:
            summary = r.poll_once()
            if all(b.get("ingested") for b in r.batches):
                break
            msg = r.wait_message(summary)
            if summary.get("error"):
                msg = f"{summary['error']} — still waiting; checking again in {_fmt_age(r.next_poll_s())} · " + msg
            job.update(done=r.done_count(), total=r.total, message=msg, extra=r.extra())
            r.sleep(r.next_poll_s())
    except JobCancelled:
        # backend-data hook: relocating the dataset detaches the job (``job.detached``) instead of cancelling it.
        if jobs.shutting_down.is_set() or getattr(job, "detached", False):
            # Server stopping (or the dataset being relocated): the provider keeps working; resume() re-attaches at
            # the next start (or once the relocation's rescan has finished).
            if not r.batches:
                store.finish_run(run_id, "error", error="The server stopped before the images were submitted")
            r.save()
            r.sink.announce()
            raise
        status = "cancelled"
        if ctx is not None:                 # retries while cancelling must not be aborted by the cancel itself
            ctx.__exit__(None, None, None)
            ctx = None
        r.cancel_provider()
    except ProviderError as e:
        status, fatal = "error", scrub(str(e))
    except Exception as e:  # noqa: BLE001
        status, fatal = "error", scrub(f"{type(e).__name__}: {e}")
        log.exception("Economy run %s failed", run_id)
    finally:
        if ctx is not None:
            ctx.__exit__(None, None, None)
        r.sink.close()
    result = {k: v for k, v in r.stats.items() if k != "processed"}
    result.update(status=status, image_ranges=runner.id_ranges(r.stats["processed"]),
                  submitted=sum(int(b.get("n") or 0) for b in r.batches))
    store.update_params(run_id, {"provider_batches": r.batches})
    store.finish_run(run_id, status, result=result, error=fatal)
    if status in ("done", "cancelled"):
        db.execute("DELETE FROM kv WHERE key=?", (kv_key(run_id),))
    runner.after_bulk_insert()
    if status == "error":
        raise ProviderError(fatal or "Economy batch failed", "invalid")
    if status == "cancelled":
        raise JobCancelled()
    msg = f"{r.stats['images_done']:,} images · {r.stats['boxes']:,} boxes · {runner._fmt_usd(r.stats['cost_usd'])} (batch price)"  # noqa: SLF001
    if r.stats.get("conflicts"):
        msg += f" · {r.stats['conflicts']:,} over a box of another class left as proposals"
    if r.stats.get("stopped") == "budget":
        msg = "Stopped at the budget · " + msg
    elif r.stats.get("stopped") == "cap":
        msg = "Sized to the 30-day spend cap set in Settings · " + msg
    job.update(done=r.total, total=r.total, message=msg, extra=result)
    return result


# ------------------------------------------------------------------ restart / cancel without a live job

def resume(force: bool = False) -> List[int]:
    """Startup scan: re-attach a job to every Economy run whose provider batches are still out. Idempotent per
    database file; runs created by this process are skipped (their own job is driving them)."""
    key = str(getattr(db, "DB_PATH", ""))
    if _resumed.get(key) and not force:
        return []
    _resumed[key] = True
    started: List[int] = []
    for r in db.query("SELECT id, dataset_id, job_id, params, created_at FROM ai_runs WHERE kind='batch' AND "
                      "status='running'"):
        params = db.loads(r["params"], {})
        if not isinstance(params, dict) or params.get("speed") != "economy":
            continue
        if (r["created_at"] or 0) >= STARTED_AT and not force:
            continue
        j = jobs.get(r["job_id"]) if r["job_id"] else None
        if j is not None and j.status in ("queued", "running"):
            continue
        if not params.get("provider_batches"):
            store.finish_run(r["id"], "error", error="The server stopped before the images were submitted — start the "
                                                     "run again")
            continue
        try:
            registry.get(params.get("provider") or "")
        except KeyError:
            log.warning("Economy run %s: provider %s unavailable; left waiting", r["id"], params.get("provider"))
            continue
        job = jobs.submit("ai_batch", lambda jb, rid=r["id"]: _lifecycle(jb, rid, resume=True), dataset_id=r["dataset_id"],
                          message="Reconnecting to the batch API…", pool=_pool())
        store.set_run_job(r["id"], job.id)
        runner.watch_unstarted(job, [r["id"]])
        started.append(r["id"])
    if started:
        log.info("Resumed %d Economy batch run(s): %s", len(started), started)
    return started


def cancel_detached(run_id: int) -> dict:
    """Cancel an Economy run that has no live job (e.g. its provider is unavailable): cancel the provider batches
    best effort and mark the run cancelled."""
    run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    if not run:
        return {"cancelled": False}
    params = db.loads(run["params"], {}) or {}
    errors = []
    try:
        p = registry.get(params.get("provider") or run["provider"])
        for b in params.get("provider_batches") or []:
            if not b.get("ingested"):
                try:
                    p.batch_cancel(b["id"])
                except Exception as e:  # noqa: BLE001
                    errors.append(scrub(str(e))[:200])
    except KeyError:
        errors.append("provider unavailable — its batches expire on their own within 24 h")
    store.finish_run(run_id, "cancelled", error="; ".join(errors) or None)
    db.execute("DELETE FROM kv WHERE key=?", (kv_key(run_id),))
    return {"cancelled": True, "errors": errors}


def start_resume() -> Optional[threading.Thread]:
    """App startup (``main.py`` lifespan → ``api.ai.startup``): add late columns, then re-attach Economy runs, in a
    background thread so a slow provider import never delays the first page. Nothing happens at import time."""
    if not getattr(db, "_initialised", True):
        return None

    def _go():
        try:
            store.ensure_schema()
            resume()
        except Exception:  # noqa: BLE001 — never break startup
            log.exception("Resuming Economy batch runs failed")
    th = threading.Thread(target=_go, name="fovea-ai-resume", daemon=True)
    th.start()
    return th
