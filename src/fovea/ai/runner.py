"""Background AI jobs (pool 'ai'): batch auto-label, label audit, provider benchmark, bulk proposal accept.

Shared rules:
* everything is validated BEFORE a job is queued (bad provider / model / class / conf → 422 at once, finding C8);
* label files are written only by write targets and accept, always after a snapshot (``core.snapshots``), per image
  under ``LABEL_LOCK`` with a fresh re-read, then ``refresh_image`` and one ``refresh_dataset_counts`` at the end;
* the job's ``extra`` carries live counters ``{cost_usd, budget_usd, images_done, boxes, errors, skipped}`` and the
  message changes with them (the SSE stream keys on the message);
* a fatal ProviderError (auth, quota, unavailable, invalid) aborts the job; per-image errors are counted;
  10 failures in a row also abort (a provider that is down should not burn through 40k images of retries).
"""
from __future__ import annotations

import errno
import json
import logging
import math
import random
import re
import threading
import time
from collections import Counter, defaultdict
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from .. import db
from ..core.labels import LABEL_LOCK, read_label_details, read_label_rows, with_confs, write_label_file
from ..core.scanner import refresh_dataset_counts, refresh_image
from ..jobs import Job, JobCancelled, JobError, jobs
from . import geometry, pipeline, registry, store
from .base import ProviderError, Usage
from .base import billed_usage, carry_usage  # ai-cloud hook: failed provider calls that were billed
from .secrets import scrub

TARGETS = ("proposals", "fill", "append", "replace")
MAX_CONSECUTIVE_ERRORS = 10
log = logging.getLogger("fovea.ai")

try:  # backend-data's snapshots (undo for bulk writes)
    from ..core.snapshots import snapshot_before as _snapshot_before
except ImportError:  # pragma: no cover — older tree without snapshots
    _snapshot_before = None


def id_ranges(ids) -> List[List[int]]:
    """[[first, last], ...] of consecutive ids: the compact record of which images a run processed (Compare
    reads ``result.image_ranges`` to count images where a model found nothing)."""
    out: List[List[int]] = []
    for i in sorted(set(int(x) for x in ids)):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def snapshot(ds_id: str, image_ids: Sequence[int], op: str) -> Optional[int]:
    if _snapshot_before is None or not image_ids:
        return None
    rows: List[dict] = []
    for chunk in store.chunked(list(image_ids), 500):
        rows += db.query(f"SELECT id, label_path FROM images WHERE id IN ({','.join('?' * len(chunk))}) "
                         "AND label_path IS NOT NULL", chunk)
    if not rows:
        return None
    return _snapshot_before(ds_id, [r["label_path"] for r in rows], op, [r["id"] for r in rows])


def write_log(sids: Sequence[Optional[int]]):
    """backend-data hook: a ``core.snapshots.WriteLog`` recording what a write op leaves in each label file, so its
    Undo (restore mode 'undo') reverts only files still as the op left them — never a later hand edit — and its
    snapshot keeps only the files the op wrote. None without snapshots."""
    ids = sorted({int(x) for x in sids if x})
    if not ids:
        return None
    try:
        from ..core.snapshots import WriteLog
    except ImportError:  # pragma: no cover — older tree
        return None
    return WriteLog(ids)


def n_classes_of(ds_row: dict) -> Optional[int]:
    classes = db.loads(ds_row["classes"])
    return len(classes) if ds_row.get("classes_source") != "inferred" and classes else None


def spent_30d(provider: str) -> float:
    """What ``provider`` billed in the last 30 days — including its share of two-stage, audit and review runs that
    are booked under another (or a composite) provider name."""
    return store.spent_since(provider, time.time() - 30 * 86400)


def caps_left(providers: Sequence[str], exclude_run: Optional[int] = None) -> Optional[float]:
    """The smallest remaining 30-day spend cap over the cloud providers involved (None = no cap applies)."""
    left = [c for c in (cap_remaining(p, exclude_run) for p in set(p for p in providers if p) if registry.is_cloud(p))
            if c is not None]
    return min(left) if left else None


# Money other runs have committed but not billed yet. A provider bills an Economy run only when its batch is read,
# up to 24 h after it was sized against the cap, and a running Now run still has images to go: a run started
# meanwhile must be sized against what is really left, or every run gets the whole remaining cap.
_holds: Dict[int, Tuple[Tuple[str, ...], float]] = {}     # Now runs in this process: run id -> (providers, usd)
_holds_lock = threading.Lock()


def hold(run_id: int, providers: Sequence[str], usd: Optional[float]) -> None:
    """Record (or with ``usd`` None / <= 0 drop) what a running Now run may still spend."""
    with _holds_lock:
        if usd is None or not math.isfinite(usd) or usd <= 0:
            _holds.pop(int(run_id), None)
        else:
            _holds[int(run_id)] = (tuple(p for p in providers if p), float(usd))


def reserved_usd(provider: str, exclude_run: Optional[int] = None) -> float:
    """What ``provider``'s other runs have committed but not billed: each Economy run still waiting at the provider
    (images sized × the batch price's high estimate, minus what it billed so far) and each Now run's remaining
    images (see :func:`hold`)."""
    total = 0.0
    with _holds_lock:
        held = {rid: v for rid, v in _holds.items() if rid != exclude_run and provider in v[0]}
    if held:
        live = set()
        for chunk in store.chunked(list(held)):
            live.update(r["id"] for r in db.query(f"SELECT id FROM ai_runs WHERE status='running' AND id IN "
                                                  f"({','.join('?' * len(chunk))})", chunk))
        for rid in list(held):
            if rid in live:
                total += held[rid][1]
            else:                                    # ended without clearing its hold (a crash in its job)
                hold(rid, (), None)
    for r in db.query("SELECT id, params, cost_usd FROM ai_runs WHERE kind='batch' AND status='running' AND provider=?",
                      (provider,)):
        if r["id"] == exclude_run:
            continue
        params = db.loads(r["params"], {}) or {}
        if not isinstance(params, dict) or params.get("speed") != "economy":
            continue
        try:
            per, n = float(params.get("per_image_usd") or 0), int(params.get("images") or 0)
        except (TypeError, ValueError):
            continue
        total += max(0.0, per * n - float(r["cost_usd"] or 0))
    return total


class WarnTally:
    """Per-image warnings folded into a few job-level lines. Counted warnings ("2 boxes of classes not in this
    dataset skipped (car ×2, chair ×1)") are summed across images — counts and the class tallies in parentheses —
    instead of one line per distinct per-image count; other warnings are counted by how many images gave them."""

    _COUNTED = re.compile(r"^(\d[\d,]*)\s+(.+?)(?:\s*\(([^()]*)\))?\s*$")
    _ITEM = re.compile(r"^(.+?)\s*×\s*(\d+)$")
    _PLURAL = {"box": "boxes", "crop": "crops", "image": "images", "proposal": "proposals", "object": "objects",
               "tile": "tiles", "row": "rows"}
    _SINGULAR = {**{v: k for k, v in _PLURAL.items()}, **{k: k for k in _PLURAL}}
    _NOUN = re.compile(r"\b(" + "|".join(sorted(_SINGULAR, key=len, reverse=True)) + r")\b")

    def __init__(self):
        self._counted: Dict[str, list] = {}         # key -> [total, Counter(details), images]
        self._plain: Counter = Counter()

    def add(self, warnings: Optional[Sequence[str]]) -> None:
        for w in warnings or []:
            w = str(w).strip()
            m = self._COUNTED.match(w)
            if not m:
                self._plain[w[:160]] += 1
                continue
            n = int(m.group(1).replace(",", ""))
            key = self._NOUN.sub(lambda mm: self._SINGULAR.get(mm.group(0), mm.group(0)), m.group(2), count=1)[:160]
            ent = self._counted.setdefault(key, [0, Counter(), 0])
            ent[0] += n
            ent[2] += 1
            for item in (m.group(3) or "").split(","):
                item = item.strip().rstrip("…").strip()
                mm = self._ITEM.match(item)
                if mm:
                    ent[1][mm.group(1).strip()] += int(mm.group(2))
                elif item:
                    ent[1][item] += 0

    def lines(self, limit: int = 5) -> List[str]:
        out: List[Tuple[int, str]] = []
        for key, (n, details, _imgs) in self._counted.items():
            noun = self._NOUN.sub(lambda mm: self._PLURAL.get(mm.group(0), mm.group(0)), key, count=1) if n != 1 else key
            text = f"{n:,} {noun}"
            tally = [f"{k} ×{v}" if v else k for k, v in details.most_common(6)]
            if tally:
                text += f" ({', '.join(tally)}{'…' if len(details) > 6 else ''})"
            out.append((n, text))
        for w, n in self._plain.items():
            out.append((n, f"{w} ({n}×)" if n > 1 else w))
        out.sort(key=lambda t: -t[0])
        return [t for _n, t in out[:limit]]


FATAL_WRITE_ERRNOS = (errno.ENOSPC, errno.EROFS, getattr(errno, "EDQUOT", errno.ENOSPC))


def write_error_text(label_path: Any, e: BaseException) -> str:
    """'Cannot write <host path>: Permission denied' — the label file the user knows, not a temp file path."""
    try:
        from ..paths import to_host
        where = to_host(str(label_path)) if label_path else ""
    except Exception:  # noqa: BLE001
        where = str(label_path or "")
    why = getattr(e, "strerror", None) or f"{type(e).__name__}: {e}"
    return scrub(f"Cannot write {where}: {why}" if where else f"Cannot write the label file: {why}")[:500]


def cap_remaining(provider: str, exclude_run: Optional[int] = None) -> Optional[float]:
    """Rolling 30-day spend cap from Settings minus what was billed and what other running runs have committed
    (None = no cap). ``exclude_run``: the run asking (its own commitment is not in its way)."""
    from .config import spend_cap
    cap = spend_cap(provider)
    if cap is None:
        return None
    return max(0.0, cap - spent_30d(provider) - reserved_usd(provider, exclude_run))


def _fmt_usd(v: float) -> str:
    return f"${v:,.2f}" if v >= 0.1 else f"${v:,.4f}"


# ------------------------------------------------------------------ batch

def _validate_detect(ds_row, provider, model, conf, cls_ids, options) -> Tuple[str, List[str]]:
    """Validate a detect configuration without running it; returns (model, providers involved)."""
    options = options or {}
    if (options.get("pipeline") or "") == "two_stage":
        det, clf = options.get("detector") or {}, options.get("classifier") or {}
        if not isinstance(det, dict) or not isinstance(clf, dict):
            raise ProviderError("two_stage needs options.detector and options.classifier", "invalid")
        dp = pipeline.provider_for(det.get("provider") or "", "detect")
        dm = pipeline.check_model(dp.id, det.get("model"))
        cp = pipeline.provider_for(clf.get("provider") or "", "classify")
        cm = pipeline.check_model(cp.id, clf.get("model"))
        pipeline.as_conf(det.get("conf"), 0.25, "detector conf")
        pipeline.specs_for(ds_row, cls_ids)
        src = det.get("source_classes") or det.get("classes") or ["person"]
        if not isinstance(src, list) or not src or len(src) > 20:
            raise ProviderError("detector.source_classes must be a list of up to 20 names", "invalid")
        pipeline._source_specs(dp.id, dm, src)          # noqa: SLF001 — validates names against the model
        dopts = det.get("options")
        if dopts is not None and not isinstance(dopts, dict):
            raise ProviderError("detector.options must be an object", "invalid")
        if dopts and "tiles" in dopts:
            pipeline.tiles_option(dp.id, dopts["tiles"])
        return f"{dm}+{cm}", [dp.id, cp.id]
    pipeline.provider_for(provider, "detect")
    m = pipeline.check_model(provider, model)
    pipeline.specs_for(ds_row, cls_ids)
    pipeline.as_conf(conf)
    if "tiles" in options:
        pipeline.tiles_option(provider, options["tiles"])
    return m, [provider]


def validate_detect(ds_row, provider, model, conf, cls_ids, options) -> Tuple[str, List[str]]:
    return _validate_detect(ds_row, provider, model, conf, cls_ids, options)


def _per_image_hi(ds_row, provider, model, options, batch: bool = False) -> float:
    try:
        from .estimate import estimate
        e = estimate(ds_row, None, 1, provider, model, options or {})
        if batch and e.get("usd_batch"):
            return float(e["usd_batch"][1] or 0)
        return float(e["usd"][1] or 0)
    except Exception:  # noqa: BLE001
        return 0.0


def model_priced(pid: str, model: str) -> bool:
    """Can Fovea price this model's calls? Local providers are free; a cloud model missing from the price table (one
    a live "Refresh list" added) is booked at $0 — no budget or spend cap could ever stop it."""
    if not pid or not registry.is_cloud(pid):
        return True
    try:
        e = registry.get(pid).estimate(1024, 768, model, 1, {}) or {}
    except Exception:  # noqa: BLE001 — an estimate problem is not a missing price
        return True
    if e.get("priced") is not None:
        return bool(e["priced"])
    return float((e.get("usd") or [0.0, 0.0])[-1] or 0) > 0


def config_models(provider: Any, model: Any, options: Optional[dict]) -> List[Tuple[str, str]]:
    """(provider, model) of every model a detect configuration calls — a two-stage pipeline's detector and classifier.
    Only call after the configuration was validated."""
    o = options if isinstance(options, dict) else {}
    if (o.get("pipeline") or "") == "two_stage":
        out = []
        for k in ("detector", "classifier"):
            v = o.get(k) if isinstance(o.get(k), dict) else {}
            pid = v.get("provider")
            if isinstance(pid, str) and pid:
                out.append((pid, pipeline.check_model(pid, v.get("model"))))
        return out
    return [(provider, pipeline.check_model(provider, model))] if isinstance(provider, str) and provider else []


def refuse_unpriced(models: Sequence[Tuple[str, str]], budget: Optional[float] = None) -> None:
    """Refuse (ProviderError 'invalid') a cloud model with no price on file when a run budget or the provider's
    30-day spend cap has to hold: its spend is booked at $0, so neither could ever stop it."""
    from .config import spend_cap
    bad = [(p, m) for p, m in dict.fromkeys(models) if not model_priced(p, m)]
    if not bad:
        return
    capped = [p for p in dict.fromkeys(p for p, _m in bad) if spend_cap(p) is not None]
    if budget is None and not capped:
        return
    names = ", ".join(m for _p, m in bad)
    limits = (["the budget"] if budget is not None else []) + \
        [f"the {registry.NAMES.get(p, p)} spend cap set in Settings" for p in capped]
    fix = "remove " + " and ".join((["the budget"] if budget is not None else []) + (["the spend cap"] if capped else []))
    raise ProviderError(f"No price on file for {names}: Fovea cannot count what it costs, so {' and '.join(limits)} "
                        f"could not stop it. Pick a model with a known price, or {fix} to run it with the cost "
                        "not tracked.", "invalid")


def finish_unstarted(run_ids: Sequence[int]) -> None:
    """End the runs of a job cancelled while it was still queued: its function never runs, and that is the only code
    that finishes them — they would stay 'running' (shown as "interrupted") and an Economy run would turn into an
    error at the next start. During a server shutdown nothing is done: those runs really were interrupted, and an
    Economy run's batches are re-attached at the next start."""
    for rid in run_ids:
        hold(int(rid), (), None)
    if jobs.shutting_down.is_set():
        return
    for rid in run_ids:
        r = db.query_one("SELECT id, status, params, result, job_id FROM ai_runs WHERE id=?", (int(rid),))
        if not r or r["status"] != "running":
            continue
        params = db.loads(r["params"], {}) or {}
        if params.get("speed") == "economy":
            from . import economy
            if params.get("provider_batches"):    # batches are out at the provider: cancel them there too
                # backend-io hook: unless a relocation detached the job — it re-attaches the run after its rescan
                if getattr(jobs.get(r["job_id"]) if r["job_id"] else None, "detached", False):
                    continue
                threading.Thread(target=economy.cancel_detached, args=(int(rid),), name="fovea-ai-cancel",
                                 daemon=True).start()
                continue
            db.execute("DELETE FROM kv WHERE key=?", (economy.kv_key(int(rid)),))
        res = db.loads(r["result"], {}) or {}
        store.finish_run(int(rid), "cancelled", result={**(res if isinstance(res, dict) else {}), "status": "cancelled",
                                                        "started": False})


def watch_unstarted(job: Job, run_ids: Sequence[int]) -> None:
    """Finish ``run_ids`` if ``job`` is cancelled before it starts (jobs.py hook)."""
    ids = [int(i) for i in run_ids]
    if hasattr(job, "on_unstarted_cancel"):
        job.on_unstarted_cancel = lambda _j: finish_unstarted(ids)
    if job.status == "cancelled" and job.started_at is None:      # cancelled before the hook was set
        finish_unstarted(ids)


MIN_SPEED_IMAGES = 16       # fewer images after the first chunk say little (two chunks can finish together)


def record_speed(provider: str, model: str, batched: bool, images: int, seconds: float, chunk: int = 1) -> None:
    """A local job's measured seconds per image (after model loading) → the estimate's speed table."""
    if images < max(MIN_SPEED_IMAGES, 3 * chunk) or seconds < 1.0:
        return
    try:
        from . import local
        device = local.resolve_device() if local.is_imported() else ""
        local.record_throughput(provider, model, "batch" if batched else "single", images, seconds, device)
    except Exception:  # noqa: BLE001 — a hint for estimates only
        log.debug("recording job speed failed", exc_info=True)


class SpeedMeter:
    """A local job's throughput for the estimate's speed table, robust to pauses. Time is ``time.monotonic`` (no
    wall-clock jumps), read in windows of at least ``window`` images; a window far slower than the run's median (the
    machine slept, the process was stopped) is left out instead of turning a 45 s job into "about 55 min" next time."""

    def __init__(self, window: int = 16):
        self.window = max(1, int(window))
        self.t0: Optional[float] = None
        self.n0 = 0
        self.windows: List[Tuple[float, int]] = []     # (seconds, images)

    @property
    def started(self) -> bool:
        return self.t0 is not None

    def start(self, images_done: int) -> None:
        self.t0, self.n0 = time.monotonic(), int(images_done)

    def tick(self, images_done: int, final: bool = False) -> None:
        if self.t0 is None:
            return
        dn = int(images_done) - self.n0
        if dn <= 0 or (dn < self.window and not final):
            return
        now = time.monotonic()
        self.windows.append((now - self.t0, dn))
        self.t0, self.n0 = now, int(images_done)

    def result(self) -> Tuple[int, float]:
        """(images, seconds) over the windows kept."""
        ws = [(dt, n) for dt, n in self.windows if n > 0]
        if len(ws) >= 3:
            rates = sorted(dt / n for dt, n in ws)
            med = rates[len(rates) // 2]
            ws = [(dt, n) for dt, n in ws if dt / n <= 4 * med + 1e-3]
        return sum(n for _dt, n in ws), sum(dt for dt, _n in ws)


def notify_written(ds_id: str, image_ids: Sequence[int]) -> None:
    """Announce label-file writes to the LABEL_WRITE_HOOKS (QA recomputes its label checks for those images)."""
    if not image_ids:
        return
    try:
        from ..core.labels import notify_label_write
    except ImportError:  # pragma: no cover — older tree without hooks
        return
    try:
        notify_label_write(ds_id, list(image_ids))
    except Exception:  # noqa: BLE001 — a hook problem must never fail the job that wrote the labels
        log.exception("label-write hooks failed for %s", ds_id)


def after_bulk_insert() -> None:
    """Planner statistics after a job inserted many ai_preds rows: backend-data's central ``db.refresh_planner_stats``
    when present (it re-analyses drifted tables and makes every connection re-plan), else our own re-analyse."""
    fn = getattr(db, "refresh_planner_stats", None)
    if callable(fn):
        try:
            fn(["ai_preds", "ai_audit", "boxes"])
            return
        except Exception:  # noqa: BLE001 — statistics are an optimisation; never fail the job
            log.debug("db.refresh_planner_stats failed", exc_info=True)
    store.refresh_planner_stats(throttle=False)


class JobWaits:
    """Provider retry waits seen by THIS job's worker threads ("Rate limited — retrying in 12 s"), fed by
    ``vlm.wait_listener`` — the listener also aborts a long wait at once when the job is cancelled."""

    def __init__(self, job: Job):
        self.job = job
        self._w: Dict[int, Tuple[float, str]] = {}

    def listener(self, seconds: int, reason: str) -> None:
        tid = threading.get_ident()
        if seconds > 0:
            self._w[tid] = (time.time() + seconds, reason)
        else:
            self._w.pop(tid, None)
        self.job.check()

    def wrap(self, fn, *args):
        vlm = _vlm()
        if vlm is None or not hasattr(vlm, "wait_listener"):
            return fn(*args)
        try:
            with vlm.wait_listener(self.listener):
                return fn(*args)
        finally:
            self._w.pop(threading.get_ident(), None)

    def text(self) -> Optional[str]:
        now = time.time()
        live = [(until - now, why) for until, why in list(self._w.values()) if until > now]
        if not live:
            return None
        left, why = max(live)
        why = (why or "waiting").strip()
        return f"{why[:1].upper()}{why[1:]} — retrying in {int(math.ceil(left))} s"


def _vlm():
    try:
        from . import vlm
        return vlm
    except Exception:  # noqa: BLE001 — httpx / PIL missing: no cloud providers, no waits to report
        return None


class BatchSink:
    """Where a batch's per-image detections go — shared by the synchronous path and Economy (provider batch) runs.

    ``proposals`` stores pending preds and never touches label files. Write targets re-read the label file under
    LABEL_LOCK, never create or empty a file because a model saw nothing, dedupe Append at IoU >= 0.7 (a box over a
    labeled object of another class is not appended either: it waits as a pending proposal, counted in
    ``conflicts``), keep the confidence column only with ``keep_conf``; every write is announced to the label-write
    hooks."""

    def __init__(self, ds_row: dict, run_id: int, target: str, keep_conf: bool, stats: Dict[str, Any]):
        self.ds_id = ds_row["id"]
        self.run_id = run_id
        self.target = target
        self.keep_conf = keep_conf
        self.n_classes = n_classes_of(ds_row)
        self.stats = stats
        self.written: List[int] = []
        self._announced = 0
        self._wlog = None                    # backend-data hook: write_log() of this run's snapshot(s)

    def skip_reason(self, img: Optional[dict]) -> Optional[str]:
        if not img:
            return "gone"
        if self.target == "fill" and img["has_label"]:
            return "labeled"
        if self.target != "proposals" and not img["label_path"]:
            return "no_label_path"
        return None

    def apply(self, iid: int, img: dict, out: dict) -> None:
        ds_id, run_id, target, stats = self.ds_id, self.run_id, self.target, self.stats
        preds, n_reg = store.drop_in_regions(iid, out["preds"])     # never written into an ignored region
        if n_reg:
            stats["in_ignored_regions"] = int(stats.get("in_ignored_regions") or 0) + n_reg
        if target == "proposals":
            preds, n_rej = store.drop_rejected(iid, preds)     # what the user rejected here stays rejected
            if n_rej:
                stats["already_rejected"] = int(stats.get("already_rejected") or 0) + n_rej
            store.insert_preds(run_id, ds_id, iid, preds, "pending", replace_pending=("detect",))
            stats["boxes"] += len(preds)
            return
        lp = Path(img["label_path"])
        with LABEL_LOCK:
            exists = lp.is_file()
            existing = read_label_rows(lp)[0] if exists else []
            if target == "fill" and exists:
                stats["skipped"] += 1        # labeled while inference ran: respect it
                store.insert_preds(run_id, ds_id, iid, [{**p, "state": "rejected", "reason": "image was labeled meanwhile"}
                                                        for p in preds], "rejected")
                return
            if not preds:
                stats["skipped"] += 1        # never create or empty a label file because a model saw nothing
                return
            conflicts: List[dict] = []
            if target == "append":
                # A box over a labeled object of ANOTHER class would give that object two classes: never appended —
                # it waits as a proposal (⇄ in Annotate), the bulk-accept rule.
                ok, conflicts = split_conflicts(preds, existing, DUP_IOU)
                fresh, dups = geometry.split_duplicates(ok, existing)
            else:
                fresh, dups = preds, []
            rows = [[p["cls"], p["xc"], p["yc"], p["w"], p["h"]] + ([p["conf"]] if self.keep_conf and p["conf"] is not None else [])
                    for p in fresh]
            new_rows = (existing + rows) if target == "append" else rows
            if target == "append" and not rows:
                stats["duplicates"] += len(dups)
                store.insert_preds(run_id, ds_id, iid, [{**p, "reason": "already labeled"} for p in dups], "rejected")
                self._park(iid, conflicts)
                return
            write_label_file(lp, new_rows)
            self._log_write(img["label_path"])     # backend-data hook (under LABEL_LOCK: reads back this write)
        self._park(iid, conflicts)
        refresh_image(iid, self.n_classes, update_counts=False)
        self.written.append(iid)
        stats["written"] += 1
        stats["boxes"] += len(rows)
        stats["duplicates"] += len(dups)
        store.insert_preds(run_id, ds_id, iid, fresh, "accepted")
        if dups:
            store.insert_preds(run_id, ds_id, iid, [{**p, "reason": "already labeled"} for p in dups], "rejected")
        if len(self.written) - self._announced >= 200:      # long runs: keep QA flags current as we go
            self.announce()

    def _park(self, iid: int, conflicts: List[dict]) -> None:
        """Append's conflicts (a box over a labeled object of another class) become pending proposals of this run,
        unless the user rejected that suggestion before or the same one is already pending."""
        if not conflicts:
            return
        stats = self.stats
        stats["conflicts"] = int(stats.get("conflicts") or 0) + len(conflicts)
        imgs = stats.setdefault("conflict_images", [])
        if isinstance(imgs, list) and len(imgs) < 200 and iid not in imgs:
            imgs.append(iid)
        keep, n_rej = store.drop_rejected(iid, conflicts)
        if n_rej:
            stats["already_rejected"] = int(stats.get("already_rejected") or 0) + n_rej
        pend = db.query(f"SELECT cls, xc, yc, w, h FROM ai_preds WHERE image_id=? AND state='pending' AND kind IN "
                        f"({','.join('?' * len(store.DETECT_KINDS))})", [iid, *store.DETECT_KINDS])
        keep = [p for p in keep if not any(int(q["cls"]) == int(p["cls"]) and geometry.iou4(
            (p["xc"], p["yc"], p["w"], p["h"]), (q["xc"], q["yc"], q["w"], q["h"])) >= DUP_IOU for q in pend)]
        if keep:
            store.insert_preds(self.run_id, self.ds_id, iid, [
                {**p, "reason": p.get("reason") or "over a labeled box of another class — decide in Annotate"}
                for p in keep], "pending")

    def keep(self, iid: int, out: dict) -> int:
        """A result that came back after the run stopped (cancel, fatal error): billed, so kept — always as pending
        proposals (replacing this image's earlier ones), never written to a label file. Returns the boxes stored."""
        preds, _n_rej = store.drop_rejected(iid, out["preds"])
        return len(store.insert_preds(self.run_id, self.ds_id, iid, preds, "pending", replace_pending=("detect",)))

    def announce(self) -> None:
        fresh = self.written[self._announced:]
        self._announced = len(self.written)
        notify_written(self.ds_id, fresh)

    def snapshot(self, image_ids: Sequence[int]) -> Optional[int]:
        snap_ids = list(image_ids)
        if self.target == "fill":            # fill only ever creates missing files: snapshot just those
            snap_ids = [r["id"] for chunk in store.chunked(snap_ids) for r in db.query(
                f"SELECT id FROM images WHERE has_label=0 AND id IN ({','.join('?' * len(chunk))})", chunk)]
        return snapshot(self.ds_id, snap_ids, f"ai_{self.target}")

    def _snapshot_ids(self) -> List[int]:
        return sorted({int(x) for x in [self.stats.get("snapshot_id"), *(self.stats.get("snapshot_ids") or [])] if x})

    def _log_write(self, label_path: str) -> None:
        """backend-data hook: record the write against this run's snapshot(s). An Economy run takes one per
        provider batch and may continue in a new process, so the log follows the ids kept in ``stats``."""
        sids = self._snapshot_ids()
        if self._wlog is None or self._wlog.sids != sids:
            if self._wlog is not None:
                self._wlog.flush()
            self._wlog = write_log(sids)
        if self._wlog is not None:
            self._wlog.wrote(label_path)

    def close(self) -> None:
        if self.target != "proposals":
            refresh_dataset_counts(self.ds_id)
        self.announce()
        # backend-data hook: a synchronous run is over — its undo point keeps only the files it wrote (none: no undo
        # point). An Economy run may still resume and ingest more into the same snapshots: flush only.
        if self.target != "proposals":
            wlog = self._wlog if self._wlog is not None and self._wlog.sids == self._snapshot_ids() else \
                write_log(self._snapshot_ids())
            self._wlog = None
            if wlog is not None:
                if self.stats.get("speed") == "economy":
                    wlog.close(trim=False, drop_empty=False)
                elif wlog.close() is None:
                    self.stats["snapshot_id"] = None


SPEEDS = ("now", "economy")


def parse_speed(speed: Any) -> str:
    s = str(speed or "now").strip().lower()
    s = {"standard": "now", "sync": "now", "batch": "economy", "batch_api": "economy"}.get(s, s)
    if s not in SPEEDS:
        raise ProviderError("speed must be 'now' or 'economy'", "invalid")
    return s


def start_batch(ds_row: dict, image_ids: List[int], provider: str, model: str, conf, cls_ids, options: dict,
                target: str, budget_usd: Optional[float], concurrency: Optional[int],
                speed: str = "now") -> Tuple[Job, int]:
    if target not in TARGETS:
        raise ProviderError(f"target must be one of {', '.join(TARGETS)}", "invalid")
    options = dict(options or {})
    speed = parse_speed(speed)
    model_v, involved = _validate_detect(ds_row, provider, model, conf, cls_ids, options)
    if not image_ids:
        raise ProviderError("No images selected", "invalid")
    cloud = any(registry.is_cloud(p) for p in involved)
    if concurrency is not None:
        try:
            concurrency = int(concurrency)
        except (TypeError, ValueError):
            raise ProviderError("concurrency must be an integer", "invalid") from None
        if not 1 <= concurrency <= 16:
            raise ProviderError("concurrency must be between 1 and 16", "invalid")
    budget = None
    if budget_usd not in (None, ""):
        try:
            budget = float(budget_usd)
        except (TypeError, ValueError):
            raise ProviderError("budget_usd must be a number", "invalid") from None
        if not math.isfinite(budget) or budget < 0:
            raise ProviderError("budget_usd must be zero or more", "invalid")
    if cloud:
        refuse_unpriced(config_models(provider, model, options), budget)
    caps = [c for c in (cap_remaining(p) for p in set(involved)) if c is not None]
    budget_from_cap = False         # the run stops at the spend cap, not at a budget the user set
    if caps:
        cap = min(caps)
        if cap <= 0:
            held = sum(reserved_usd(p) for p in set(involved) if registry.is_cloud(p))
            raise ProviderError("The monthly spend cap set in Settings is reached for this provider"
                                + (f" ({_fmt_usd(held)} of it is held by runs still going)" if held > 0 else ""), "quota")
        budget_from_cap = budget is None or cap < budget
        budget = cap if budget is None else min(budget, cap)
    if speed == "economy":
        from . import economy
        try:
            return economy.start(ds_row, image_ids, provider, model_v, conf, cls_ids, options, target, budget,
                                 budget_from_cap=budget_from_cap)
        except ProviderError as e:
            if e.code != "quota" or not budget_from_cap:
                raise
            held = sum(reserved_usd(p) for p in set(involved) if registry.is_cloud(p))
            raise ProviderError(f"The 30-day spend cap set in Settings has {_fmt_usd(budget or 0)} left"
                                + (f" ({_fmt_usd(held)} is held by runs still going)" if held > 0 else "")
                                + " — not enough for one image at the batch price", "quota") from None
    infer_batch = pipeline.opt_int(options, "infer_batch", 8, 1, 32)
    chunk = infer_batch if (not cloud and pipeline.batch_capable(provider, options)) else 1
    if cloud:
        from .config import provider_config
        conc = concurrency or min(int(provider_config(p).get("concurrency") or 4) for p in involved)
    else:
        # One local model: inference is serialised by its lock. With chunked inference a second worker decodes
        # the next chunk while the first one runs on the GPU.
        conc = 2 if chunk > 1 else 1
    keep_conf = bool(options.pop("keep_conf", False))
    ds_id = ds_row["id"]
    params = {"provider": provider, "model": model_v, "conf": conf, "classes": cls_ids, "options": options,
              "target": target, "budget_usd": budget, "concurrency": conc, "keep_conf": keep_conf,
              "images": len(image_ids), "speed": "now", "infer_batch": chunk}
    run_provider = provider if (options.get("pipeline") or "") != "two_stage" else "+".join(involved)
    run_id = store.create_run(ds_id, "batch", run_provider, model_v, params)
    per_image_hi = _per_image_hi(ds_row, provider, model, options) if cloud else 0.0

    def commitment(spent: float, images_left: int, per_image: float) -> float:
        """What this run may still spend: its images to go at the per-image reservation, within its budget."""
        c = per_image * max(0, images_left)
        return min(c, max(0.0, budget - spent)) if budget is not None else c
    if cloud:          # other runs started while this one is queued or running see this as taken
        hold(run_id, involved, commitment(0.0, len(image_ids), per_image_hi))

    def run(job: Job):
        stats: Dict[str, Any] = {"cost_usd": 0.0, "budget_usd": budget, "images_done": 0, "boxes": 0, "errors": 0,
                                 "skipped": 0, "duplicates": 0, "written": 0, "run_id": run_id, "last_error": None,
                                 "warnings": [], "speed": "now", "write_errors": 0, "already_rejected": 0,
                                 "conflicts": 0}
        sink = BatchSink(ds_row, run_id, target, keep_conf, stats)
        waits = JobWaits(job)
        total = len(image_ids)
        usage_total = Usage()
        status, stop_reason, fatal_msg = "done", None, None
        local_first = not cloud
        tally = WarnTally()
        consecutive = 0
        processed: List[int] = []
        t_start = None
        meter = SpeedMeter(max(MIN_SPEED_IMAGES, 2 * chunk))   # job throughput once the first chunk is in
        # Spend control for cloud runs: an image's real cost is only known when it returns, and reasoning models can
        # bill several times the estimate. With a budget, the first image runs alone; after that every image in
        # flight reserves the largest per-image cost seen so far (at least the estimate's high end), and the 30-day
        # caps are re-read (other runs spend too).
        cost_seen = {"n": 0, "max": 0.0}
        cap_state = {"left": caps_left(involved, run_id) if cloud else None, "at": time.time(), "cost": 0.0}

        def reserve() -> float:
            return max(per_image_hi, cost_seen["max"] * 1.1)

        def cap_left() -> Optional[float]:
            if cloud and time.time() - cap_state["at"] >= 1.0:
                cap_state.update(left=caps_left(involved, run_id), at=time.time(), cost=stats["cost_usd"])
            if cap_state["left"] is None:
                return None
            return cap_state["left"] - max(0.0, stats["cost_usd"] - cap_state["cost"])   # this run's spend since the read

        def work(ids: List[int]):
            """[(iid, img, out | ProviderError | Exception | None, skip_reason)] for one chunk of images."""
            rows = {r["id"]: r for chunk_ in store.chunked(ids) for r in db.query(
                f"SELECT * FROM images WHERE dataset_id=? AND id IN ({','.join('?' * len(chunk_))})", [ds_id, *chunk_])}
            res, todo = [], []
            for iid in ids:
                img = rows.get(iid)
                why = sink.skip_reason(img)
                res.append([iid, img, None, why])
                if why is None:
                    todo.append(len(res) - 1)
            if not todo:
                return res
            if len(todo) == 1 or chunk == 1:
                for k in todo:
                    try:
                        res[k][2] = pipeline.detect_image(ds_row, res[k][1], provider, model, conf, cls_ids, options)
                    except JobCancelled:
                        raise
                    except Exception as e:  # noqa: BLE001 — reported per image
                        res[k][2] = e
                return res
            try:
                outs = pipeline.detect_images(ds_row, [res[k][1] for k in todo], provider, model, conf, cls_ids, options)
            except ProviderError as e:      # a configuration error: every image of the chunk failed the same way
                outs = [e] * len(todo)
            for k, o in zip(todo, outs):
                res[k][2] = o
            return res

        def message(done: int) -> str:
            parts = [f"{done:,}/{total:,}"]
            if stats["boxes"]:
                parts.append(f"{stats['boxes']:,} boxes")
            if cloud:
                parts.append(_fmt_usd(stats["cost_usd"]) + (f" of {_fmt_usd(budget)}" if budget is not None else ""))
            if stats["errors"]:
                parts.append(f"{stats['errors']} errors")
            if stats["skipped"]:
                parts.append(f"{stats['skipped']} skipped")
            if stats.get("conflicts"):
                parts.append(f"{stats['conflicts']:,} over a box of another class left as proposals")
            w = waits.text()
            if w:
                parts.insert(0, w)
            return " · ".join(parts)

        def failed(msg: str, fatal: bool, streak: bool = True) -> None:
            nonlocal consecutive, status, fatal_msg
            stats["errors"] += 1
            stats["last_error"] = msg
            if not streak:                   # a property of this image (too large to decode), not of the provider
                return
            consecutive += 1
            if fatal:
                status, fatal_msg = "error", msg
            elif consecutive >= MAX_CONSECUTIVE_ERRORS and status != "error":
                status, fatal_msg = "error", f"{consecutive} images in a row failed — last error: {msg}"

        def handle(iid: int, img: Optional[dict], out: Any, skip: Optional[str], late: bool = False) -> None:
            """One image's outcome. ``late``: it finished after the run stopped (cancel, fatal error) — the provider
            billed it, so it is kept, but as pending proposals whatever the target: nothing is written to label
            files after a stop."""
            nonlocal consecutive, local_first, t_start
            late = late or job.cancelled
            if skip:
                stats["skipped"] += 1
                return
            if isinstance(out, BaseException):
                spent = billed_usage(out)    # ai-cloud hook: refused / unreadable / cut-off answers are billed too
                if spent is not None:
                    usage_total.add(spent)
                    stats["cost_usd"] = round(usage_total.cost_usd, 6)
                    cost_seen["n"] += 1
                    cost_seen["max"] = max(cost_seen["max"], float(spent.cost_usd or 0))
                    store.add_usage(run_id, spent, images=0)
                if isinstance(out, JobCancelled) or late:
                    return
                msg = scrub(str(out) if isinstance(out, ProviderError) else f"{type(out).__name__}: {out}")
                too_large = isinstance(out, ProviderError) and out.code == "too_large"
                failed(msg, isinstance(out, ProviderError) and out.fatal, streak=not too_large)
                return
            u = out["usage"]
            usage_total.add(u)
            stats["cost_usd"] = round(usage_total.cost_usd, 6)
            cost_seen["n"] += 1
            cost_seen["max"] = max(cost_seen["max"], float(u.cost_usd or 0))
            n_boxes = 0
            if late:
                try:
                    n_boxes = sink.keep(iid, out)
                except Exception:  # noqa: BLE001 — a database hiccup: the image is still billed below
                    log.exception("AI batch %s: keeping the late result of image %s failed", run_id, iid)
                store.add_usage(run_id, u, images=1, boxes=n_boxes)
                stats["late"] = int(stats.get("late") or 0) + 1
                stats["boxes"] += n_boxes
                stats["images_done"] += 1
                processed.append(iid)
                return
            tally.add(out.get("warnings"))
            ok = False
            try:
                sink.apply(iid, img, out)
                ok = True
            except OSError as e:          # one unwritable label file must not end the run
                stats["write_errors"] += 1
                failed(write_error_text(img.get("label_path") if img else None, e),
                       getattr(e, "errno", None) in FATAL_WRITE_ERRNOS)
            except Exception as e:  # noqa: BLE001 — a database hiccup on one image: count it, keep going
                log.exception("AI batch %s: storing image %s failed", run_id, iid)
                failed(scrub(f"{type(e).__name__}: {e}")[:500], False)
            finally:                     # the provider billed this image whatever happened next
                store.add_usage(run_id, u, images=1, boxes=len(out["preds"]) if ok else 0)
            if not ok:
                return
            consecutive = 0
            processed.append(iid)
            stats["images_done"] += 1
            if local_first and stats["images_done"] == 1:
                local_first = False
                t_start = time.time()

        pool = ThreadPoolExecutor(max_workers=conc, thread_name_prefix="fovea-ai-batch")
        inflight: Dict[Any, List[int]] = {}
        chunks = [image_ids[i:i + chunk] for i in range(0, total, chunk)]
        it = iter(chunks)
        done = 0
        exhausted = False
        try:
            if target != "proposals":
                job.update(0, total, f"Saving a snapshot of {total:,} label files (undo)…", extra=stats)
                stats["snapshot_id"] = sink.snapshot(image_ids)
            job.update(0, total, f"Loading {model_v}…" if local_first else "Starting…", extra=stats)
            while True:
                while not exhausted and len(inflight) < conc and status == "done":
                    n_inflight = sum(len(v) for v in inflight.values())
                    if cloud and budget is not None and n_inflight and not cost_seen["n"]:
                        break                        # wait for the first real cost before running more in parallel
                    left = cap_left()
                    # When the next image only fails to fit because of images still in flight, wait for them: their
                    # real cost replaces their reservation. Stop only when nothing is in flight and it still does not fit.
                    if budget is not None and stats["cost_usd"] + reserve() * (n_inflight + 1) > budget + 1e-9:
                        if n_inflight:
                            break
                        stop_reason = "cap" if budget_from_cap else "budget"
                        exhausted = True
                        break
                    if left is not None and reserve() * (n_inflight + 1) > left + 1e-9:
                        if n_inflight:
                            break
                        stop_reason = "cap"          # the 30-day spend cap (other runs count too)
                        exhausted = True
                        break
                    try:
                        ids = next(it)
                    except StopIteration:
                        exhausted = True
                        break
                    inflight[pool.submit(waits.wrap, work, ids)] = ids
                if not inflight:
                    break
                finished, _ = wait(list(inflight), return_when=FIRST_COMPLETED, timeout=1.0)
                for f in finished:
                    ids = inflight.pop(f)
                    try:
                        entries = f.result()
                    except JobCancelled:
                        done += len(ids)
                        continue
                    except Exception as e:  # noqa: BLE001 — e.g. the database is locked: count the chunk as failed
                        entries = [[iid, None, e, None] for iid in ids]
                    for entry in entries:
                        done += 1
                        handle(*entry)
                    if not cloud and not meter.started and stats["images_done"]:
                        meter.start(stats["images_done"])                       # model loading is behind us
                    meter.tick(stats["images_done"])
                if t_start and stats["images_done"] > 1:
                    stats["images_per_s"] = round((stats["images_done"] - 1) / max(1e-6, time.time() - t_start), 2)
                stats["warnings"] = tally.lines()
                if cloud:
                    hold(run_id, involved, 0.0 if exhausted and not inflight else
                         commitment(stats["cost_usd"], total - done, reserve()))
                job.update(done=done, message=message(done), extra=dict(stats))   # raises JobCancelled when cancelled
                if status == "error":
                    break
        except JobCancelled:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001 — the run must end with its real status and error, never "interrupted"
            log.exception("AI batch %s failed", run_id)
            status = "error"
            fatal_msg = write_error_text(getattr(e, "filename", None), e) if isinstance(e, OSError) else \
                scrub(f"{type(e).__name__}: {e}")[:500]
        finally:
            for f in inflight:
                f.cancel()
            pool.shutdown(wait=True, cancel_futures=True)
            for f in inflight:                       # images that finished after the run stopped were billed too
                if f.cancelled() or f.exception() is not None:
                    continue
                for entry in f.result():
                    try:
                        handle(*entry, late=True)
                    except Exception:  # noqa: BLE001 — never lose the run's final status over one late image
                        log.exception("AI batch %s: keeping a late result failed", run_id)
            hold(run_id, (), None)
            try:
                sink.close()
            except Exception:  # noqa: BLE001
                log.exception("AI batch %s: refreshing counts failed", run_id)
        meter.tick(stats["images_done"], final=True)
        if meter.started and (options.get("pipeline") or "") != "two_stage":
            record_speed(provider, model_v, chunk > 1, *meter.result(), chunk)
        stats["stopped"] = stop_reason
        stats["warnings"] = tally.lines()
        result = {**stats, "status": status, "image_ranges": id_ranges(processed)}
        store.finish_run(run_id, status, result=result, error=fatal_msg)
        after_bulk_insert()
        if status in ("error", "cancelled"):
            # The job summary must match Runs & cost: images that finished after the stop are in both.
            late = int(stats.get("late") or 0)
            tail = (f" · {late:,} image{'s' if late != 1 else ''} finished after the "
                    f"{'cancel' if status == 'cancelled' else 'error'} (billed, kept as proposals)") if late else ""
            try:
                job.update(done=done, message=message(done) + tail, extra=dict(stats))
            except JobCancelled:
                pass
        if status == "error":
            raise ProviderError(fatal_msg or "Batch failed", "invalid")
        if status == "cancelled":
            raise JobCancelled()
        msg = message(done)
        if stop_reason == "budget":
            msg = f"Stopped at the budget ({_fmt_usd(budget or 0)}) · " + msg
        elif stop_reason == "cap":
            msg = "Stopped at the 30-day spend cap set in Settings · " + msg
        job.update(done=done, message=msg, extra=dict(stats))
        return result

    job = jobs.submit("ai_batch", run, dataset_id=ds_id, message="Queued", pool="ai")
    store.set_run_job(run_id, job.id)
    watch_unstarted(job, [run_id])
    return job, run_id


# ------------------------------------------------------------------ audit

def _family_fn(class_family: Dict[str, List[int]], pseudo_names: Dict[int, str]):
    fam = {k.lower(): set(int(c) for c in v) for k, v in class_family.items()}

    def same(gt_cls: int, pred_cls: int) -> bool:
        name = pseudo_names.get(int(pred_cls))
        if name is None:
            return int(gt_cls) == int(pred_cls)
        return int(gt_cls) in fam.get(name.lower(), set())
    return same


def prepare_audit(ds_row: dict, provider: str, model: str, conf_floor, class_family, mode: str,
                  committee: Optional[list], options: Optional[dict]) -> dict:
    """Validate an audit configuration (shared by the audit job and the single-image review)."""
    ds_id = ds_row["id"]
    if mode not in ("geometry", "full"):
        raise ProviderError("mode must be 'geometry' or 'full'", "invalid")
    options = dict(options or {})
    members = [{"provider": provider, "model": model, "options": options}]
    for m in committee or []:
        if not isinstance(m, dict):
            raise ProviderError("committee entries must be {provider, model}", "invalid")
        if not isinstance(m.get("options") or {}, dict):
            raise ProviderError("committee options must be an object", "invalid")
        members.append({"provider": m.get("provider"), "model": m.get("model"), "options": dict(m.get("options") or {})})
    if len(members) > 4:
        raise ProviderError("At most 3 committee members besides the main model", "invalid")
    names = [str(c) for c in db.loads(ds_row["classes"])]
    fam: Dict[str, List[int]] = {}
    if mode == "geometry":
        if not class_family or not isinstance(class_family, dict):
            raise ProviderError("Geometry mode needs class_family, e.g. {\"person\": [0, 1]}", "invalid")
        for k, v in class_family.items():
            if not isinstance(v, list) or not v:
                raise ProviderError("class_family values must be lists of dataset class ids", "invalid")
            for c in v:
                if not isinstance(c, int) or isinstance(c, bool) or not 0 <= c < len(names):
                    raise ProviderError(f"class_family: {c!r} is not a class of this dataset", "invalid")
            fam[str(k)] = [int(c) for c in v]
    resolved = []
    involved: List[str] = []
    for m in members:
        if not isinstance(m["provider"], (str, type(None))) or not isinstance(m["model"], (str, type(None))):
            raise ProviderError("committee entries must be {provider, model} with string values", "invalid")
        p = pipeline.provider_for(m["provider"] or "", "detect")
        mm = pipeline.check_model(p.id, m["model"])
        if mode == "geometry":
            pipeline._source_specs(p.id, mm, list(fam))     # noqa: SLF001 — validates family names
        elif (m["options"].get("pipeline") or "") == "two_stage" or "tiles" in m["options"]:
            _validate_detect(ds_row, p.id, mm, None, None, m["options"])    # before the job is queued
        resolved.append({**m, "provider": p.id, "model": mm})
        involved += providers_of(p.id, m["options"])
    floor = pipeline.as_conf(conf_floor, 0.05, "conf_floor")
    hi = pipeline.as_conf(options.get("hi"), 0.5, "options.hi")
    # Missing objects found in geometry mode get the family's most common class as a starting point.
    fam_default = {}
    for k, v in fam.items():
        r = db.query_one(f"SELECT cls, COUNT(*) n FROM boxes WHERE dataset_id=? AND cls IN ({','.join('?' * len(v))}) "
                         "GROUP BY cls ORDER BY n DESC LIMIT 1", [ds_id, *v])
        fam_default[k] = int(r["cls"]) if r else v[0]
    involved = sorted(set(involved))
    cloud = any(registry.is_cloud(p) for p in involved)
    if cloud:          # a spend cap must hold: refuse a model whose spend would be booked at $0
        refuse_unpriced([pm for m in resolved for pm in (config_models(m["provider"], m["model"], m["options"])
                                                        if mode == "full" else [(m["provider"], m["model"])])])
    return {"members": resolved, "conf_floor": floor, "hi": hi, "class_family": fam, "fam_default": fam_default,
            "mode": mode, "providers": involved, "cloud": cloud}


def providers_of(provider: Any, options: Optional[dict]) -> List[str]:
    """Every provider a detect configuration calls: the provider itself, or a two-stage pipeline's detector and
    classifier (``options.pipeline='two_stage'``). Non-string values are left out (the caller validates them)."""
    o = options if isinstance(options, dict) else {}
    if (o.get("pipeline") or "") == "two_stage":
        out = []
        for k in ("detector", "classifier"):
            v = o.get(k)
            pid = v.get("provider") if isinstance(v, dict) else None
            if isinstance(pid, str) and pid:
                out.append(pid)
        return out
    return [provider] if isinstance(provider, str) and provider else []


def _member_detect(ds_row: dict, img: dict, cfg: dict, m: dict, exclude: Sequence[int] = ()) -> dict:
    mopts = dict(m["options"])
    if cfg["mode"] == "geometry":
        return _detect_family(ds_row, img, m["provider"], m["model"], cfg["conf_floor"], list(cfg["class_family"]), mopts)
    # Exemplar prototypes are built once per audit (a per-image hold-out would rebuild them for every image).
    return pipeline.detect_image(ds_row, img, m["provider"], m["model"], cfg["conf_floor"], None, mopts,
                                 exclude_exemplars=list(exclude))


def _member_detect_many(ds_row: dict, imgs: List[dict], cfg: dict, m: dict) -> List[Any]:
    """One chunked forward pass for a local member (see pipeline.detect_images); entries are outs or errors."""
    if cfg["mode"] == "geometry":
        return _detect_family_many(ds_row, imgs, m["provider"], m["model"], cfg["conf_floor"], list(cfg["class_family"]),
                                   dict(m["options"]))
    return pipeline.detect_images(ds_row, imgs, m["provider"], m["model"], cfg["conf_floor"], None, dict(m["options"]))


def _raw_rows(out: dict, cfg: dict) -> List[dict]:
    """The main model's detections, kept as state 'info' / kind 'detect' so Compare can evaluate an audit run like a
    batch run. Geometry mode detects family names (COCO 'person'): those are stored as the family's most common
    dataset class and keep the family name as their label (run params say so: raw_detections = 'family')."""
    pseudo = {pipeline.PSEUDO_BASE + i: n for i, n in enumerate(cfg["class_family"])}
    rows = []
    for p in out["preds"]:
        cls = int(p["cls"])
        label = p.get("label") or ""
        if cfg["mode"] == "geometry":
            fname = pseudo.get(cls, "")
            cls = int(cfg["fam_default"].get(fname, cls))
            label = label or fname
        if cls >= pipeline.PSEUDO_BASE:
            continue
        rows.append({"cls": cls, "xc": p["xc"], "yc": p["yc"], "w": p["w"], "h": p["h"], "conf": p.get("conf"),
                     "label": label, "kind": "detect", "reason": ""})
    return rows


def audit_image(ds_row: dict, img: dict, cfg: dict, run_id: int, exclude: Sequence[int] = (),
                pre: Optional[Dict[int, dict]] = None, store_raw: bool = True) -> dict:
    """Audit one labeled image against every committee member; store pending audit preds (replacing this image's
    earlier ones), the main model's raw detections (state 'info', for Compare) and its ai_audit score. ``pre`` holds
    detections already computed per member index (chunked local inference). Raises ProviderError from the provider."""
    ds_id = ds_row["id"]
    rows = [r[:5] for r in read_label_rows(Path(img["label_path"]))[0]]
    fam, mode = cfg["class_family"], cfg["mode"]
    pseudo_names = {pipeline.PSEUDO_BASE + i: n for i, n in enumerate(fam)}
    fam_classes = {c for v in fam.values() for c in v}
    usage = Usage()
    per_member: List[List[dict]] = []
    raw: List[dict] = []
    for mi, m in enumerate(cfg["members"]):
        out = (pre or {}).get(mi)
        if out is None:
            try:
                out = _member_detect(ds_row, img, cfg, m, exclude)
            except Exception as e:     # ai-cloud hook: earlier committee members were billed already
                carry_usage(e, usage)
                raise
        if mode == "geometry":
            same = _family_fn(fam, pseudo_names)
            covered = fam_classes
        else:
            same = None
            covered = set(out.get("covered") or [])
        usage.add(out["usage"])
        if mi == 0 and store_raw:
            raw = _raw_rows(out, cfg)
        # Only judge boxes of classes this model can find: a car detector says nothing about benches. ref_row keeps
        # pointing at the row's position in the whole file.
        keep = [i for i, r in enumerate(rows) if int(r[0]) in covered]
        gt = [rows[i] for i in keep]
        others = [rows[i] for i in range(len(rows)) if i not in set(keep)]
        findings = geometry.label_issues(gt, out["preds"], hi=cfg["hi"], same_family=same, others=others)
        for f in findings:
            if f["ref_row"] is not None:
                f["ref_row"] = keep[f["ref_row"]]
            if f.get("dup_of") is not None:          # name the other box the way Annotate numbers rows (1-based)
                f["dup_of"] = keep[f["dup_of"]]
                other = f"box #{f['dup_of'] + 1}"
                if f.get("dup_cls") is not None and int(f["dup_cls"]) != int(f["cls"]):
                    nm = (db.loads(ds_row["classes"]) or [])
                    cn = nm[int(f["dup_cls"])] if 0 <= int(f["dup_cls"]) < len(nm) else f"class {f['dup_cls']}"
                    other += f" ({cn})"
                f["reason"] = f"labeled twice: the same object as {other} (IoU {f.get('dup_iou', 0):.2f})"
        if mode == "geometry":
            for f in findings:
                if f["kind"] in ("missing", "swap"):
                    fname = pseudo_names.get(f["cls"], "")
                    f["cls"] = cfg["fam_default"].get(fname, f["cls"])
                    if f["kind"] == "missing":
                        f["reason"] = f"{fname or 'object'} found (confidence {f['conf']:.2f}) but not labeled"
        per_member.append(findings)
    kept = [f for f in _merge_committee(per_member) if f["score"] >= 0.05]
    preds = [{"cls": f["cls"], "xc": f["box"][0], "yc": f["box"][1], "w": f["box"][2], "h": f["box"][3],
              "conf": f.get("conf"), "label": f.get("label", ""), "kind": f["kind"], "ref_row": f["ref_row"],
              "ref_box": f.get("ref_box"), "reason": f["reason"], "_score": f["score"]} for f in kept]
    preds, n_rej = store.drop_rejected(img["id"], preds)    # a finding the user dismissed is not raised again
    kept = [{"kind": p["kind"], "score": p.pop("_score")} for p in preds]
    stored = store.insert_preds(run_id, ds_id, img["id"], preds, "pending", replace_pending=store.AUDIT_KINDS)
    if raw:
        store.insert_preds(run_id, ds_id, img["id"], raw, "info")
    counts = dict(Counter(f["kind"] for f in kept))
    score = geometry.pool_score([f["score"] for f in kept])
    store.upsert_audit(ds_id, img["id"], run_id, score, counts)
    store.add_usage(run_id, usage, images=1, boxes=len(kept))
    return {"preds": stored, "score": score, "counts": counts, "usage": usage, "gt": len(rows), "raw": len(raw),
            "already_rejected": n_rej}


def _audit_chunk(cfg: dict) -> int:
    """Images per model call for an audit: chunked only when every member is a local batch-capable detector."""
    if len(cfg["members"]) != 1 or cfg["cloud"]:
        return 1
    m = cfg["members"][0]
    if not pipeline.batch_capable(m["provider"], m["options"]):
        return 1
    return pipeline.opt_int(m["options"], "infer_batch", 8, 1, 32)


def start_audit(ds_row: dict, image_ids: List[int], provider: str, model: str, conf_floor, class_family,
                mode: str, committee: Optional[list], options: Optional[dict]) -> Tuple[Job, int]:
    ds_id = ds_row["id"]
    cfg = prepare_audit(ds_row, provider, model, conf_floor, class_family, mode, committee, options)
    if not image_ids:
        raise ProviderError("No images selected", "invalid")
    labeled = []
    for chunk in store.chunked(image_ids):
        labeled += [r["id"] for r in db.query(f"SELECT id FROM images WHERE dataset_id=? AND has_label=1 AND id IN "
                                                f"({','.join('?' * len(chunk))}) ORDER BY id", [ds_id, *chunk])]
    if not labeled:
        raise ProviderError("None of the selected images has a label file to audit", "invalid")
    params = {k: v for k, v in cfg.items() if k != "cloud"}
    params["images"] = len(labeled)
    params["raw_detections"] = "family" if mode == "geometry" else "dataset"
    resolved = cfg["members"]
    run_id = store.create_run(ds_id, "audit", "+".join(cfg["providers"] or sorted({m["provider"] for m in resolved})),
                              "+".join(m["model"] for m in resolved), params)
    cloud = cfg["cloud"]
    step = _audit_chunk(cfg)
    if cloud:
        left = caps_left(cfg["providers"])
        if left is not None and left <= 0:
            store.finish_run(run_id, "error", error="The monthly spend cap set in Settings is reached")
            raise ProviderError("The monthly spend cap set in Settings is reached for this provider", "quota")

    def run(job: Job):
        stats = {"images_done": 0, "flagged": 0, "findings": 0, "errors": 0, "skipped": 0, "cost_usd": 0.0,
                 "counts": {k: 0 for k in store.AUDIT_KINDS}, "run_id": run_id, "last_error": None, "raw_detections": 0}
        total = len(labeled)
        usage_total = Usage()
        status, fatal = "done", None
        consecutive = 0
        processed: List[int] = []
        state = {"msg": "Loading models…" if not cloud else "Starting…", "k": 0}
        waits = JobWaits(job)

        def on_wait(seconds: int, reason: str) -> None:
            waits.listener(seconds, reason)
            w = waits.text()
            job.update(message=f"{w} · {state['msg']}" if w else state["msg"])

        def chunk_outs(ids: List[int]):
            """(imgs, {image id: member-0 out | error}) for a chunk, computed ahead of the audit loop."""
            imgs = [r for i in ids for r in [db.query_one("SELECT * FROM images WHERE id=?", (i,))] if r]
            imgs = [r for r in imgs if r["label_path"] and Path(r["label_path"]).is_file()]
            if step == 1 or not imgs:
                return imgs, {}
            try:
                outs = _member_detect_many(ds_row, imgs, cfg, cfg["members"][0])
            except ProviderError as e:
                outs = [e] * len(imgs)
            return imgs, {r["id"]: o for r, o in zip(imgs, outs)}

        def book_failed(e: BaseException) -> None:
            """ai-cloud hook: an image that failed after billed calls (refusal, unreadable answer, a later committee
            member failing) still costs money — count it in the run, the ledger and the 30-day cap."""
            spent = billed_usage(e)
            if spent is not None:
                usage_total.add(spent)
                stats["cost_usd"] = round(usage_total.cost_usd, 6)
                store.add_usage(run_id, spent, images=0)

        def one(img: dict, pre_out) -> None:
            nonlocal consecutive
            if isinstance(pre_out, BaseException):
                raise pre_out
            res = audit_image(ds_row, img, cfg, run_id, pre={0: pre_out} if pre_out is not None else None)
            consecutive = 0
            usage_total.add(res["usage"])
            processed.append(img["id"])
            stats["images_done"] += 1
            stats["findings"] += len(res["preds"])
            stats["flagged"] += 1 if res["preds"] else 0
            stats["raw_detections"] += res.get("raw", 0)
            stats["already_rejected"] = int(stats.get("already_rejected") or 0) + int(res.get("already_rejected") or 0)
            for kk, n in res["counts"].items():
                stats["counts"][kk] = stats["counts"].get(kk, 0) + n
            stats["cost_usd"] = round(usage_total.cost_usd, 6)

        chunks = [labeled[i:i + step] for i in range(0, total, step)]
        meter = SpeedMeter(max(MIN_SPEED_IMAGES, 2 * step))
        ahead = ThreadPoolExecutor(max_workers=1, thread_name_prefix="fovea-ai-audit")   # next chunk on the GPU
        vlm = _vlm()
        ctx = vlm.wait_listener(on_wait) if (vlm is not None and hasattr(vlm, "wait_listener")) else None
        try:
            if ctx is not None:
                ctx.__enter__()
            job.update(0, total, state["msg"], extra=stats)
            nxt = ahead.submit(chunk_outs, chunks[0]) if chunks and step > 1 else None
            k = 0
            for ci, ids in enumerate(chunks):
                if step > 1:
                    imgs, pre = nxt.result()
                    nxt = ahead.submit(chunk_outs, chunks[ci + 1]) if ci + 1 < len(chunks) else None
                else:
                    imgs, pre = chunk_outs(ids)
                stats["skipped"] += len(ids) - len(imgs)
                for img in imgs:
                    try:
                        one(img, pre.get(img["id"]))
                    except JobCancelled as e:
                        book_failed(e)
                        raise
                    except ProviderError as e:
                        book_failed(e)
                        msg = scrub(str(e))
                        stats["errors"] += 1
                        stats["last_error"] = msg
                        if e.code == "too_large":      # this image only: not a sign the provider is failing
                            continue
                        consecutive += 1
                        if e.fatal or consecutive >= MAX_CONSECUTIVE_ERRORS:
                            status, fatal = "error", msg
                            break
                    except Exception as e:  # noqa: BLE001 — unreadable image / label, model hiccup: count it, keep going
                        book_failed(e)
                        stats["errors"] += 1
                        stats["last_error"] = scrub(f"{type(e).__name__}: {e}")
                        consecutive += 1
                        if consecutive >= MAX_CONSECUTIVE_ERRORS:
                            status, fatal = "error", f"{consecutive} images in a row failed — last error: {stats['last_error']}"
                            break
                if status == "error":
                    break
                k += len(ids)
                if not cloud and not meter.started:
                    meter.start(stats["images_done"])                           # model loading is behind us
                meter.tick(stats["images_done"])
                msg = f"{k:,}/{total:,} · {stats['flagged']:,} images with findings"
                if cloud:
                    msg += f" · {_fmt_usd(stats['cost_usd'])}"
                state["msg"] = msg
                job.update(done=k, message=msg, extra=dict(stats))
                if cloud and k < total:
                    left = caps_left(cfg["providers"])       # this run's spend is already on the ledger
                    if left is not None and left <= 0:
                        stats["stopped"] = "cap"
                        state["msg"] = msg = "Stopped at the 30-day spend cap set in Settings · " + msg
                        job.update(message=msg, extra=dict(stats))
                        break
        except JobCancelled:
            status = "cancelled"
        except Exception as e:  # noqa: BLE001 — end the run with its real status and error, never "interrupted"
            log.exception("AI audit %s failed", run_id)
            status, fatal = "error", scrub(f"{type(e).__name__}: {e}")[:500]
        finally:
            if ctx is not None:
                ctx.__exit__(None, None, None)
            ahead.shutdown(wait=True, cancel_futures=True)
        meter.tick(stats["images_done"], final=True)
        if meter.started and len(cfg["members"]) == 1:
            m0 = cfg["members"][0]
            if (m0["options"].get("pipeline") or "") != "two_stage":
                record_speed(m0["provider"], m0["model"], step > 1, *meter.result(), step)
        result = {**stats, "status": status, "image_ranges": id_ranges(processed)}
        store.finish_run(run_id, status, result=result, error=fatal)
        after_bulk_insert()
        if status in ("error", "cancelled"):
            try:                                     # the job summary shows what the run row books
                job.update(extra=dict(stats))
            except JobCancelled:
                pass
        if status == "error":
            raise ProviderError(fatal or "Audit failed", "invalid")
        if status == "cancelled":
            raise JobCancelled()
        return result

    job = jobs.submit("ai_audit", run, dataset_id=ds_id, message="Queued", pool="ai")
    store.set_run_job(run_id, job.id)
    watch_unstarted(job, [run_id])
    return job, run_id


def _detect_family(ds_row, img, provider, model, floor, family_names, options) -> dict:
    """Detector run whose classes are the family names (e.g. COCO 'person'), mapped to pseudo ids."""
    from .base import DetectRequest
    p = pipeline.provider_for(provider, "detect")
    specs, explicit = pipeline._source_specs(provider, model, family_names)   # noqa: SLF001
    opts = {k: v for k, v in options.items() if k not in ("pipeline", "class_map")}
    if explicit:
        opts["class_map"] = explicit
    err = pipeline.pixel_error(img, provider)
    if err is not None:
        raise err
    t0 = time.time()
    res = p.detect(DetectRequest(image_path=img["abs_path"], classes=specs, model=model, conf=floor, options=opts))
    preds, _ = pipeline._postprocess(res.detections, specs, floor, 300)        # noqa: SLF001
    return {"preds": preds, "usage": (res.usage or Usage()).tagged(p.id), "latency_ms": int((time.time() - t0) * 1000),
            "warnings": res.warnings or []}


def _detect_family_many(ds_row, imgs, provider, model, floor, family_names, options) -> List[Any]:
    """_detect_family for a chunk of images in one model call."""
    from .base import DetectRequest
    p = pipeline.provider_for(provider, "detect")
    specs, explicit = pipeline._source_specs(provider, model, family_names)   # noqa: SLF001
    opts = {k: v for k, v in options.items() if k not in ("pipeline", "class_map")}
    if explicit:
        opts["class_map"] = explicit
    t0 = time.time()
    out: List[Any] = [pipeline.pixel_error(img, provider) for img in imgs]   # too large: never decoded
    todo = [k for k, e in enumerate(out) if e is None]
    reqs = [DetectRequest(image_path=imgs[k]["abs_path"], classes=specs, model=model, conf=floor, options=opts)
            for k in todo]
    for k, res in zip(todo, p.detect_many(reqs) if reqs else []):
        if isinstance(res, Exception):
            out[k] = res
            continue
        preds, _ = pipeline._postprocess(res.detections, specs, floor, 300)        # noqa: SLF001
        out[k] = {"preds": preds, "usage": (res.usage or Usage()).tagged(p.id), "warnings": res.warnings or [],
                  "latency_ms": res.latency_ms or int((time.time() - t0) * 1000 / max(1, len(reqs)))}
    return out


def _merge_committee(per_member: List[List[dict]]) -> List[dict]:
    """Agreement-weighted merge: a finding's score is the members' mean (a member that did not report it counts 0)."""
    n = len(per_member)
    if n == 1:
        return per_member[0]
    keyed: Dict[tuple, List[dict]] = defaultdict(list)
    missing: List[List[dict]] = []
    for findings in per_member:
        for f in findings:
            if f["kind"] == "missing":
                for cl in missing:
                    if geometry.iou4(cl[0]["box"], f["box"]) >= 0.5:
                        cl.append(f)
                        break
                else:
                    missing.append([f])
            else:
                keyed[(f["kind"], f["ref_row"])].append(f)
    out = []
    for group in list(keyed.values()) + missing:
        best = max(group, key=lambda f: f["score"])
        out.append({**best, "score": round(sum(f["score"] for f in group) / n, 4),
                    "reason": best["reason"] + (f" · {len(group)}/{n} models agree" if n > 1 else "")})
    out.sort(key=lambda f: -f["score"])
    return out


def audit_summary(ds_id: str, top: int = 20) -> dict:
    # The audit the findings come from: an audit cancelled before it looked at any image is not "the last audit".
    run = db.query_one("SELECT * FROM ai_runs WHERE dataset_id=? AND kind='audit' AND status!='deleted' AND NOT "
                       "(status='cancelled' AND images=0) ORDER BY id DESC LIMIT 1", (ds_id,))
    counts = {k: 0 for k in store.AUDIT_KINDS}
    for r in db.query("SELECT kind, COUNT(*) n FROM ai_preds WHERE dataset_id=? AND state='pending' AND kind IN "
                      f"({','.join('?' * len(store.AUDIT_KINDS))}) GROUP BY kind", [ds_id, *store.AUDIT_KINDS]):
        counts[r["kind"]] = r["n"]
    flagged = db.query_one(
        "SELECT COUNT(DISTINCT image_id) n FROM ai_preds WHERE dataset_id=? AND state='pending' AND kind IN "
        f"({','.join('?' * len(store.AUDIT_KINDS))})", [ds_id, *store.AUDIT_KINDS])
    audited = db.query_one("SELECT COUNT(*) n FROM ai_audit WHERE dataset_id=?", (ds_id,))
    # An uncorrelated membership test: SQLite 3.51 answered the correlated EXISTS (… kind IN (…)) with one row per
    # matching kind, so an image with loose + swap findings was listed twice and "top 6" showed fewer images.
    rows = db.query("SELECT a.image_id, a.score, a.counts, i.rel_path, i.split, i.mtime FROM ai_audit a JOIN images i ON "
                    "i.id=a.image_id WHERE a.dataset_id=? AND a.score > 0 AND a.image_id IN (SELECT p.image_id FROM "
                    "ai_preds p WHERE p.dataset_id=? AND p.state='pending' AND p.kind IN "
                    f"({','.join('?' * len(store.AUDIT_KINDS))})) ORDER BY a.score DESC, a.image_id LIMIT ?",
                    [ds_id, ds_id, *store.AUDIT_KINDS, int(top)])
    seen: set = set()
    rows = [r for r in rows if not (r["image_id"] in seen or seen.add(r["image_id"]))]   # belt and braces
    # What is still pending per image — ai_audit.counts is what the audit found, and findings decided since then
    # (accepted / rejected in Annotate) must not be listed again.
    left: Dict[int, Dict[str, int]] = defaultdict(dict)
    if rows:
        top_ids = [r["image_id"] for r in rows]
        for r in db.query(f"SELECT image_id, kind, COUNT(*) n FROM ai_preds WHERE dataset_id=? AND state='pending' AND "
                          f"kind IN ({','.join('?' * len(store.AUDIT_KINDS))}) AND image_id IN "
                          f"({','.join('?' * len(top_ids))}) GROUP BY image_id, kind",
                          [ds_id, *store.AUDIT_KINDS, *top_ids]):
            left[int(r["image_id"])][r["kind"]] = int(r["n"])
    return {"run": store.run_public(run) if run else None, "counts": counts,
            "images_flagged": int(flagged["n"] or 0) if flagged else 0,
            "images_audited": int(audited["n"] or 0) if audited else 0,
            "top": [{"image_id": r["image_id"], "rel_path": r["rel_path"], "split": r["split"], "mtime": r["mtime"],
                     "score": round(r["score"], 4),
                     "counts": {k: n for k, n in sorted(left.get(int(r["image_id"]), {}).items(),
                                                        key=lambda kv: store.AUDIT_KINDS.index(kv[0]))}}
                    for r in rows]}


# ------------------------------------------------------------------ benchmark

def sample_labeled(ds_id: str, n: int, seed: int = 0, within: Optional[Sequence[int]] = None) -> List[int]:
    if within is not None:
        ids = []
        for chunk in store.chunked(list(within)):
            ids += [r["id"] for r in db.query(f"SELECT id FROM images WHERE dataset_id=? AND has_label=1 AND id IN "
                                              f"({','.join('?' * len(chunk))})", [ds_id, *chunk])]
    else:
        ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? AND has_label=1", (ds_id,))]
    ids.sort()
    rnd = random.Random(seed)
    if len(ids) > n:
        ids = sorted(rnd.sample(ids, n))
    return ids


def start_benchmark(ds_row: dict, image_ids: List[int], configs: List[dict], iou: float = 0.5) -> Tuple[Job, List[int]]:
    from ..plugins.compare.evaluate import average_precision, match_image
    ds_id = ds_row["id"]
    if not isinstance(configs, list) or not 1 <= len(configs) <= 6:
        raise ProviderError("Pick 1 to 6 configurations to compare", "invalid")
    iou = pipeline.as_conf(iou, 0.5, "iou")
    if not 0.1 <= iou <= 0.95:
        raise ProviderError("iou must be between 0.1 and 0.95", "invalid")
    if not image_ids:
        raise ProviderError("No labeled images to benchmark on", "invalid")
    resolved = []
    for c in configs:
        if not isinstance(c, dict) or not isinstance(c.get("options") or {}, dict):
            raise ProviderError("Each config must be {provider, model, options}", "invalid")
        opts = dict(c.get("options") or {})
        prov = c.get("provider") or ""
        m, involved = _validate_detect(ds_row, prov, c.get("model"), c.get("conf"), c.get("classes"), opts)
        if any(registry.is_cloud(p) for p in involved):
            refuse_unpriced(config_models(prov, c.get("model"), opts))
        resolved.append({"provider": prov if opts.get("pipeline") != "two_stage" else "+".join(involved), "_provider": prov,
                         "model": m, "_model": c.get("model"), "options": opts, "conf": c.get("conf"),
                         "classes": c.get("classes"), "label": c.get("label") or f"{prov} · {m}",
                         "cloud": any(registry.is_cloud(p) for p in involved)})
    run_ids = [store.create_run(ds_id, "benchmark", r["provider"], r["model"],
                                {"config": {k: v for k, v in r.items() if not k.startswith("_")}, "iou": iou,
                                 "images": len(image_ids)}) for r in resolved]
    names = [str(c) for c in db.loads(ds_row["classes"])]

    def run(job: Job):
        vlm = _vlm()
        if vlm is None or not hasattr(vlm, "wait_listener"):
            return _run(job)
        waits = JobWaits(job)

        def on_wait(seconds: int, reason: str) -> None:
            waits.listener(seconds, reason)
            w = waits.text()
            base = (job.message or "").split(" · ", 1)[-1] if (job.message or "").find("retrying in") >= 0 else job.message
            job.update(message=f"{w} · {base}" if w else base)
        with vlm.wait_listener(on_wait):
            return _run(job)

    def _run(job: Job):
        total = len(image_ids) * len(resolved)
        results = []
        job.update(0, total, "Starting…", extra={"configs": len(resolved), "images": len(image_ids)})
        step = 0
        try:
            for cfg, run_id in zip(resolved, run_ids):
                local_floor = 0.05 if not cfg["cloud"] else 0.0
                thr = pipeline.as_conf(cfg["conf"], 0.25 if not cfg["cloud"] else 0.0)
                want = set(int(x) for x in cfg["classes"]) if cfg["classes"] else None
                tp = fp = fn = 0
                records: Dict[int, list] = defaultdict(list)
                n_gt: Counter = Counter()
                errors, secs, usage, has_conf, imgs = 0, 0.0, Usage(), True, 0
                bench_ids: List[int] = []
                last_error = None
                for iid in image_ids:
                    step += 1
                    img = db.query_one("SELECT * FROM images WHERE id=?", (iid,))
                    if not img:
                        continue
                    gt = [r[:5] for r in read_label_rows(Path(img["label_path"]))[0]] if img["label_path"] else []
                    if want is not None:
                        gt = [g for g in gt if int(g[0]) in want]
                    t0 = time.time()
                    try:
                        out = pipeline.detect_image(ds_row, img, cfg["_provider"], cfg["_model"], min(thr, local_floor) if not cfg["cloud"] else thr,
                                                    cfg["classes"], cfg["options"], exclude_exemplars=image_ids)
                    except ProviderError as e:
                        if billed_usage(e) is not None:     # ai-cloud hook: a refused / unreadable answer is billed
                            usage.add(billed_usage(e))
                            store.add_usage(run_id, billed_usage(e), images=0)
                        errors += 1
                        last_error = scrub(str(e))
                        if e.fatal:
                            raise
                        job.update(done=step, message=f"{cfg['label']}: {step:,}/{total:,}")
                        continue
                    except Exception as e:  # noqa: BLE001 — one bad image must not sink the comparison
                        if billed_usage(e) is not None:     # ai-cloud hook
                            usage.add(billed_usage(e))
                            store.add_usage(run_id, billed_usage(e), images=0)
                        errors += 1
                        last_error = scrub(f"{type(e).__name__}: {e}")
                        job.update(done=step, message=f"{cfg['label']}: {step:,}/{total:,}")
                        continue
                    secs += time.time() - t0
                    imgs += 1
                    usage.add(out["usage"])
                    store.add_usage(run_id, out["usage"], images=1, boxes=len(out["preds"]))
                    store.insert_preds(run_id, ds_id, iid, out["preds"], "info")
                    bench_ids.append(iid)
                    allp = [[p["cls"], p["xc"], p["yc"], p["w"], p["h"], 1.0 if p["conf"] is None else p["conf"]]
                            for p in out["preds"]]
                    if any(p["conf"] is None for p in out["preds"]):
                        has_conf = False
                    atthr = [p for p in allp if p[5] >= thr]
                    ptp, gm = match_image(gt, atthr, iou)
                    tp += sum(ptp)
                    fp += len(ptp) - sum(ptp)
                    fn += sum(1 for m in gm if m == -1)
                    ptp_all, _ = match_image(gt, allp, iou)
                    for p, ok in zip(allp, ptp_all):
                        records[int(p[0])].append((p[5], ok))
                    for g in gt:
                        n_gt[int(g[0])] += 1
                    job.update(done=step, message=f"{cfg['label']}: {step:,}/{total:,}")
                prec = tp / (tp + fp) if tp + fp else 0.0
                rec = tp / (tp + fn) if tp + fn else 0.0
                f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
                aps = [average_precision(records[c], n_gt[c]) for c in n_gt if n_gt[c] > 0]
                ap = round(sum(aps) / len(aps), 4) if (aps and has_conf) else None
                per_class = []
                for c in sorted(set(n_gt) | set(records)):
                    per_class.append({"cls": c, "name": names[c] if 0 <= c < len(names) else f"class_{c}",
                                      "gt": n_gt[c], "ap": round(average_precision(records[c], n_gt[c]), 4) if has_conf else None})
                row = {"config": {k: v for k, v in cfg.items() if not k.startswith("_") and k != "cloud"}, "run_id": run_id,
                       "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                       "ap50" if abs(iou - 0.5) < 1e-9 else "ap": ap, "iou": iou, "conf_threshold": thr,
                       "images": imgs, "tp": tp, "fp": fp, "fn": fn,
                       "sec_per_image": round(secs / imgs, 3) if imgs else None,
                       "usd_per_image": round(usage.cost_usd / imgs, 6) if imgs else None,
                       "cost_usd": round(usage.cost_usd, 6), "errors": errors, "last_error": last_error,
                       "per_class": per_class, "image_ranges": id_ranges(bench_ids)}
                if "ap50" not in row:
                    row["ap50"] = None
                results.append(row)
                store.finish_run(run_id, "done", result=row, images=imgs)
        except JobCancelled:
            for rid in run_ids[len(results):]:
                store.finish_run(rid, "cancelled")
            raise
        except ProviderError as e:
            for rid in run_ids[len(results):]:
                store.finish_run(rid, "error", error=str(e))
            raise ProviderError(scrub(str(e)), e.code) from None
        after_bulk_insert()
        job.update(done=total, message="Done", extra={"configs": len(resolved), "images": len(image_ids)})
        return results

    job = jobs.submit("ai_benchmark", run, dataset_id=ds_id, message="Queued", pool="ai")
    for rid in run_ids:
        store.set_run_job(rid, job.id)
    watch_unstarted(job, run_ids)
    return job, run_ids


# ------------------------------------------------------------------ bulk accept

def _ref_ok(row: Sequence[float], pred: dict, kind: str) -> bool:
    """Is GT row ``row`` still the box the audit finding was computed for? With the stored ``ref_box`` (the GT box
    at audit time) that is exact; older findings fall back to comparing with the suggested box."""
    rb = (float(row[1]), float(row[2]), float(row[3]), float(row[4]))
    ref = pred.get("ref_box")
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except ValueError:
            ref = None
    if isinstance(ref, (list, tuple)) and len(ref) >= 4:
        same_box = geometry.iou4(rb, [float(v) for v in list(ref)[-4:]]) >= 0.9
        return same_box and (kind == "swap" or int(row[0]) == int(pred["cls"]))
    pb = (pred["xc"], pred["yc"], pred["w"], pred["h"])
    if kind == "loose":
        return int(row[0]) == int(pred["cls"]) and geometry.iou4(rb, pb) >= 0.25
    return geometry.iou4(rb, pb) >= 0.9


EXTENDED_KINDS = ("seg", "pose", "ext")
DUP_IOU = 0.7          # = Annotate's DUP_IOU: a box over another box (IoU >= 0.7) is the same object


def _stored_ref_box(pred: dict) -> Optional[Tuple[float, ...]]:
    ref = pred.get("ref_box")
    if isinstance(ref, str):
        try:
            ref = json.loads(ref)
        except ValueError:
            return None
    if isinstance(ref, (list, tuple)) and len(ref) >= 4:
        try:
            return tuple(float(v) for v in list(ref)[-4:])
        except (TypeError, ValueError):
            return None
    return None


def _find_ref_row(rows: List[list], pred: dict, kind: str, skip: set) -> Tuple[Optional[int], bool]:
    """(row index, known) of the label row an audit finding is about: its ``ref_row`` while that row still is the box
    the audit judged, else the row that holds that box now — an edit above it shifted the rows (IoU >= 0.9 with the
    stored ``ref_box``; same class unless a swap). ``known`` is False for a finding without ``ref_box`` (stored before
    it existed): then nothing can tell where its box went."""
    ref = pred.get("ref_row")
    if ref is not None and 0 <= int(ref) < len(rows) and int(ref) not in skip and _ref_ok(rows[int(ref)], pred, kind):
        return int(ref), True
    rb = _stored_ref_box(pred)
    if rb is None:
        return None, False
    best = None
    for i, r in enumerate(rows):
        if i in skip or (kind != "swap" and int(r[0]) != int(pred["cls"])):
            continue
        v = geometry.iou4((float(r[1]), float(r[2]), float(r[3]), float(r[4])), rb)
        if v >= 0.9:
            cand = (-v, abs(i - int(ref)) if ref is not None else 0, i)
            if best is None or cand < best:
                best = cand
    return (best[2] if best else None), True


def split_conflicts(adds: List[dict], kept: List[list], thr: float = 0.7) -> Tuple[List[dict], List[dict]]:
    """[explore hook] Annotate's ⇧Enter rule: most confident first, a proposal that lands on a box (IoU >= thr) of
    ANOTHER class only — one already in the file or one accepted just before it — is not accepted in bulk (it stays
    pending for the user to decide). Same-class overlaps pass through: ``split_duplicates`` calls them duplicates.
    Returns (proposals that may be written, conflicting ones), the first list in that confidence order."""
    placed = [(int(r[0]), (float(r[1]), float(r[2]), float(r[3]), float(r[4]))) for r in kept]
    ok: List[dict] = []
    left: List[dict] = []
    for p in sorted(adds, key=lambda d: -(d["conf"] if d.get("conf") is not None else -1.0)):
        b = (float(p["xc"]), float(p["yc"]), float(p["w"]), float(p["h"]))
        hits = {c for c, eb in placed if geometry.iou4(b, eb) >= thr}
        if hits and int(p["cls"]) not in hits:
            left.append(p)
            continue
        ok.append(p)
        if not hits:
            placed.append((int(p["cls"]), b))
    return ok, left


def apply_to_rows(rows: List[list], preds: List[dict],
                  row_kinds: Optional[Sequence[str]] = None,
                  skip_conflicts: bool = True) -> Tuple[List[list], Dict[str, List[int]]]:
    """Apply accepted proposals to one image's label rows. Returns (new rows, {outcome: [pred ids]}).

    Outcomes: ``added`` / ``modified`` / ``deleted`` (written); ``duplicates`` (an object proposal over a box of its
    class: nothing to write); ``conflicts`` (an object proposal over a box of ANOTHER class — one object would carry two
    classes: not written, left pending for a person, as Annotate's ⇧Enter does; see ``split_conflicts``; only an
    explicit ``skip_conflicts=False`` writes them); ``merged`` (a 'loose' fix whose tighter box is already in the file
    as another row of that class — its detection was accepted as a proposal meanwhile: the loose row is removed rather
    than becoming a second copy); ``stale`` (which row a finding meant cannot be told: left pending); ``obsolete``
    (the box a finding was about was removed or changed since the audit: the finding no longer applies); ``extended``.

    An audit finding follows its box (``ref_box``) when an edit above it shifted the rows.
    ``row_kinds`` (``read_label_details``) marks segmentation / pose rows: a 'loose' finding would turn such a row into
    a rectangle, so it is left pending (outcome ``extended``); a class swap or a delete keeps the polygon's tokens."""
    rows = [list(r) for r in rows]
    kinds = list(row_kinds or [])
    out: Dict[str, List[int]] = defaultdict(list)
    delete: set = set()
    adds: List[dict] = []
    tightened: List[Tuple[int, tuple]] = []      # (row, box) set by this call's loose fixes
    for p in preds:
        k = p["kind"]
        if k in ("loose", "swap", "spurious"):
            ref, known = _find_ref_row(rows, p, k, delete)
            if ref is None:
                out["obsolete" if known else "stale"].append(p["id"])
                continue
            r = rows[ref]
            if k == "loose":
                if ref < len(kinds) and kinds[ref] in EXTENDED_KINDS:
                    out["extended"].append(p["id"])
                    continue
                nb = (float(p["xc"]), float(p["yc"]), float(p["w"]), float(p["h"]))
                # Two rows "tightened" onto one detection (findings stored before the audit matched one-to-one: maybe
                # two objects the detector merged) would leave the same box twice: keep the second finding pending.
                if any(rr != ref and int(rows[rr][0]) == int(r[0]) and geometry.iou4(nb, b) >= DUP_IOU
                       for rr, b in tightened):
                    out["stale"].append(p["id"])
                    continue
                # The tighter box already is another row of this class: moving this row onto it would leave two
                # identical boxes — this row is the redundant loose copy of that object.
                if any(i != ref and i not in delete and int(o[0]) == int(r[0])
                       and geometry.iou4(nb, (float(o[1]), float(o[2]), float(o[3]), float(o[4]))) >= DUP_IOU
                       for i, o in enumerate(rows)):
                    delete.add(ref)
                    out["merged"].append(p["id"])
                    continue
                r[1:5] = list(nb)
                tightened.append((ref, nb))
                out["modified"].append(p["id"])
            elif k == "swap":
                r[0] = int(p["cls"])
                out["modified"].append(p["id"])
            else:
                delete.add(ref)
                out["deleted"].append(p["id"])
        else:
            adds.append(p)
    kept = [r for i, r in enumerate(rows) if i not in delete]
    if skip_conflicts:                                       # [explore hook] — on by default (ai-core r4)
        adds, conflicts = split_conflicts(adds, kept, DUP_IOU)
        out["conflicts"] += [p["id"] for p in conflicts]
    fresh, dups = geometry.split_duplicates(adds, kept)
    out["duplicates"] += [p["id"] for p in dups]
    out["added"] += [p["id"] for p in fresh]
    kept += [[int(p["cls"]), p["xc"], p["yc"], p["w"], p["h"]] for p in fresh]
    return kept, out


def start_accept(ds_row: dict, image_ids: Optional[List[int]], min_conf, classes, kinds,
                 skip_conflicts: bool = True) -> Job:
    ds_id = ds_row["id"]
    kinds = list(kinds or store.DETECT_KINDS)
    bad = [k for k in kinds if k not in store.ALL_KINDS]
    if bad:
        raise ProviderError(f"Unknown proposal kinds: {', '.join(bad)}", "invalid")
    mc = None if min_conf in (None, "") else pipeline.as_conf(min_conf, 0.0, "min_conf")
    cls = [int(c) for c in classes] if classes else None
    # Which proposals the user asked for, as of now. Their state is re-checked per image when the job gets there:
    # anything decided in the meantime (rejected in Annotate, accepted elsewhere) is left alone.
    preds = store.pending_for_images(ds_id, image_ids, kinds, mc, cls)
    by_img: Dict[int, List[int]] = defaultdict(list)
    for p in preds:
        by_img[p["image_id"]].append(p["id"])
    n_classes = n_classes_of(ds_row)

    def run(job: Job):
        stats = {"images": 0, "added": 0, "modified": 0, "deleted": 0, "duplicates": 0, "stale": 0, "skipped": 0,
                 "conflicts": 0,                                   # [explore hook] skip_conflicts
                 "merged": 0, "obsolete": 0, "conflict_images": [],
                 "extended": 0, "decided_meanwhile": 0, "errors": 0, "last_error": None}
        ids = sorted(by_img)
        changed_ids: List[int] = []
        fatal = None
        consecutive = 0
        job.update(0, len(ids), "Saving a snapshot…", extra=stats)
        if ids:
            stats["snapshot_id"] = snapshot(ds_id, ids, "ai_accept")
        wlog = write_log([stats.get("snapshot_id")])          # backend-data hook
        try:
            for k, iid in enumerate(ids):
                img = db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))
                if not img or not img["label_path"]:
                    stats["skipped"] += 1
                    continue
                lp = Path(img["label_path"])
                claimed: List[dict] = []
                outcome: Dict[str, List[int]] = {}
                changed = False
                try:
                    with LABEL_LOCK:
                        claimed = store.claim_pending(ds_id, by_img[iid])
                        stats["decided_meanwhile"] += len(by_img[iid]) - len(claimed)
                        if claimed:
                            det = read_label_details(lp) if lp.is_file() else None
                            rows = with_confs(det["boxes"], det["confs"]) if det else []
                            new_rows, outcome = apply_to_rows(rows, claimed, det["row_kinds"] if det else None,
                                                              skip_conflicts=skip_conflicts)
                            changed = bool(outcome.get("added") or outcome.get("modified") or outcome.get("deleted")
                                           or outcome.get("merged"))
                            if changed:
                                write_label_file(lp, new_rows)
                                if wlog is not None:             # backend-data hook
                                    wlog.wrote(img["label_path"])
                except OSError as e:              # unwritable file: leave its proposals pending, go on
                    store.release([p["id"] for p in claimed])
                    msg = write_error_text(lp, e)
                    stats["errors"] += 1
                    stats["last_error"] = msg
                    consecutive += 1
                    if getattr(e, "errno", None) in FATAL_WRITE_ERRNOS or consecutive >= MAX_CONSECUTIVE_ERRORS:
                        fatal = msg if consecutive < MAX_CONSECUTIVE_ERRORS else \
                            f"{consecutive} images in a row failed — last error: {msg}"
                        break
                    continue
                except Exception:
                    store.release([p["id"] for p in claimed])
                    raise
                consecutive = 0
                # stale findings, polygon rows and skipped conflicts go back to pending: nothing was done with them
                store.release(outcome.get("stale", []) + outcome.get("extended", []) + outcome.get("conflicts", []))
                # the box an audit finding was about is gone (or was redrawn) since the audit: the finding is answered,
                # not pending for good (rejected without decided_at: a later audit may raise it again)
                store.resolve(outcome.get("obsolete", []), "the box changed or was removed since the audit")
                if wlog is not None and claimed:                 # [backend-data hook] restoring the snapshot makes
                    back = set(outcome.get("stale", []) + outcome.get("extended", []) + outcome.get("conflicts", [])
                               + outcome.get("obsolete", []))   # what this image accepted pending again
                    wlog.decided(iid, [p for p in claimed if p["id"] not in back])
                if outcome.get("conflicts") and len(stats["conflict_images"]) < 200:
                    stats["conflict_images"].append(iid)
                # [compare hook] a duplicate stays 'accepted' but wrote nothing: say so, so Compare does not count it
                # as one of the model's boxes in the labels (same reason text as an Append run's duplicates)
                for dup_ids in store.chunked(outcome.get("duplicates", [])):
                    db.execute(f"UPDATE ai_preds SET reason='already labeled' WHERE id IN ({','.join('?' * len(dup_ids))})", dup_ids)
                if changed:
                    refresh_image(iid, n_classes, update_counts=False)
                    changed_ids.append(iid)
                    stats["images"] += 1
                for key in ("added", "modified", "deleted", "duplicates", "stale", "extended", "conflicts", "merged",
                            "obsolete"):
                    stats[key] += len(outcome.get(key, []))
                if k % 10 == 0 or k == len(ids) - 1:
                    job.update(done=k + 1, message=f"{k + 1:,}/{len(ids):,} images · {stats['added']:,} boxes added",
                               extra=dict(stats))
        finally:
            refresh_dataset_counts(ds_id)
            notify_written(ds_id, changed_ids)
            if wlog is not None:                                 # backend-data hook: nothing written -> no undo point
                stats["snapshot_id"] = wlog.close()
        if fatal:
            job.update(extra=dict(stats))
            raise JobError(f"{fatal} · {stats['images']:,} images were updated before it stopped")
        msg = f"{stats['images']:,} images updated"
        if stats["errors"]:
            msg += f" · {stats['errors']} could not be written"
        if stats["conflicts"]:                                   # [explore hook]
            msg += f" · {stats['conflicts']} over a box of another class left to decide in Annotate"
        if stats["merged"]:
            msg += f" · {stats['merged']} loose box{'es' if stats['merged'] != 1 else ''} removed (already boxed tighter)"
        if stats["obsolete"]:
            msg += (f" · {stats['obsolete']} finding{'s' if stats['obsolete'] != 1 else ''} no longer "
                    f"appl{'y' if stats['obsolete'] != 1 else 'ies'} (the box changed since the audit)")
        if stats["extended"]:
            msg += f" · {stats['extended']} polygon/pose rows left as they are"
        job.update(done=len(ids), message=msg, extra=dict(stats))
        return stats

    return jobs.submit("ai_accept", run, dataset_id=ds_id, message="Queued", pool="default")
