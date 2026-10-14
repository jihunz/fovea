"""Dataset QA: near-duplicates / split leakage, sequence consistency, shape outliers, image quality, interpolation."""
from __future__ import annotations

import threading
import time
from typing import Optional

from fastapi import APIRouter, Body, HTTPException, Query

from .. import db
from ..core import qa, scanner
from ..jobs import jobs
from .common import get_dataset_or_404

router = APIRouter(prefix="/api/datasets", tags=["qa"])

# Every successful (re)scan or relocation queues an incremental QA run.
if qa.on_scan not in scanner.POST_SCAN_HOOKS:
    scanner.POST_SCAN_HOOKS.append(qa.on_scan)


def register_label_hook() -> bool:
    """Every label write (save, bulk edit, accepted proposals, restore, ...) re-checks the label-based flags of
    the edited images' sequences and twins a few seconds later. True when core.labels offers the hook list."""
    from ..core import labels  # noqa: PLC0415
    hooks = getattr(labels, "LABEL_WRITE_HOOKS", None)
    if hooks is None:
        return False
    if qa.on_label_write not in hooks:
        hooks.append(qa.on_label_write)
    return True


register_label_hook()

_auto_lock = threading.Lock()
_auto_at: dict = {}
AUTO_EVERY = 30.0      # seconds between automatic re-runs of one dataset triggered by reads


def _labels():
    try:
        from ..core.labels import FLAG_LABELS   # noqa: PLC0415 — backend-data owns the shared label table
        return {**qa.LABELS, **{k: v for k, v in FLAG_LABELS.items() if k not in qa.LABELS}}
    except ImportError:
        return dict(qa.LABELS)


def _auto_queue(ds_id: str, checks, reason: str):
    """Reads keep results fresh without a button: a dataset indexed before QA existed, new images that were
    never hashed, or labels edited since the last run start a background job (at most every AUTO_EVERY s)."""
    with _auto_lock:
        last = _auto_at.get(ds_id, 0.0)
        if time.time() - last < AUTO_EVERY:
            return None
        _auto_at[ds_id] = time.time()
    job, _already = qa.queue_qa(ds_id, checks, message=reason)
    return job


def _bad(e: Exception):
    raise HTTPException(400, str(e))


@router.post("/{ds_id}/qa")
def start_qa(ds_id: str, payload: Optional[dict] = Body(None)):
    """Run (or re-run) the QA checks. `checks`: any of duplicates | sequence | shape | quality | labels
    (= the label-based three) or flag codes; default all. `options`: threshold overrides, remembered for the
    dataset (see core.qa.DEFAULTS). Incremental: only images whose file changed are re-hashed."""
    row = get_dataset_or_404(ds_id)
    payload = payload or {}
    checks = payload.get("checks")
    if checks is not None and not isinstance(checks, list):
        raise HTTPException(400, "checks must be a list")
    options = payload.get("options")
    if options is not None and not isinstance(options, dict):
        raise HTTPException(400, "options must be an object")
    if not row["image_count"] and row["status"] in ("new", "scanning"):
        raise HTTPException(409, "The dataset has not been indexed yet")
    try:
        job, already = qa.queue_qa(ds_id, checks, options, message="Queued")
    except ValueError as e:
        _bad(e)
    return {"job": job.to_dict(), "already_running": already}


@router.get("/{ds_id}/qa/summary")
def qa_summary(ds_id: str, auto: int = Query(1)):
    """Results of the last QA run plus freshness. With auto=1 (default) a missing or stale result queues a
    background run; poll /api/jobs/{job.id} or this endpoint."""
    row = get_dataset_or_404(ds_id)
    s = qa.load_summary(ds_id) or {}
    active = qa.active_job(ds_id)
    scanning = jobs.active_for(ds_id, "scan") is not None
    stale_imgs = qa.stale_images(ds_id) if row["image_count"] else 0
    label_at = min((s.get("checks_at") or {}).get(c, 0) for c in qa.LABEL_CHECKS) if s.get("checks_at") else 0
    # labels edited since the last label check, minus those an incremental re-check covered or is about to
    stale_lbls, refreshing = qa.label_freshness(ds_id, label_at)
    # images whose files were missing at the last run (unmounted drive) stay stale until they come back:
    # re-running for them every AUTO_EVERY seconds would only find them missing again
    missing = int((s.get("hash") or {}).get("missing") or 0)
    # duplicate results written by an older engine mean something else (e.g. leaks by folder, not split role)
    outdated = bool(s.get("checks_at", {}).get("duplicates")) and int(s.get("engine") or 1) < qa.ENGINE_VERSION
    if auto and active is None and not scanning and row["image_count"] and row["status"] != "unreachable":
        if not s.get("computed_at"):
            active = _auto_queue(ds_id, None, "Checking data quality")
        elif stale_imgs > missing:
            active = _auto_queue(ds_id, None, "Checking new images")
        elif stale_lbls:
            active = _auto_queue(ds_id, list(qa.LABEL_CHECKS), "Rechecking edited labels")
        elif outdated:
            active = _auto_queue(ds_id, ["duplicates"], "Updating data-quality checks")
    if active is None:
        active = qa.active_job(ds_id)     # queued meanwhile (a scan hook, another tab): show it, never 'stale'
    if active is not None:
        status = "running"
    elif not s.get("computed_at"):
        status = "never"
    elif stale_imgs or stale_lbls or outdated:
        status = "stale"
    else:
        status = "ok"
    counts = qa.flag_counts(ds_id)
    return {
        "status": status,
        "computed_at": s.get("computed_at"),
        "counts": {c: counts.get(c, 0) for c in qa.CODES},
        "dup_groups": s.get("dup_groups", 0),
        "exact_dups": s.get("exact_dups", 0),
        "unique_images": max(0, row["image_count"] - s.get("exact_dups", 0)),
        "split_spanning_sequences": s.get("split_spanning_sequences", []),
        "sequences": s.get("sequences", 0),
        "images": row["image_count"],
        "hashed": s.get("hashed", 0),
        "unreadable": s.get("unreadable", 0),
        "stale_images": stale_imgs,
        "stale_labels": stale_lbls,
        # edited images whose flags are being re-checked right now (seconds; no job, no banner)
        "refreshing": refreshing,
        "recheck": qa.last_recheck(ds_id),      # the latest such re-check: {at, images, seconds, counts, ...}
        # indexing is running: checks (re)start by themselves when it ends
        "scanning": scanning,
        "quality": s.get("quality"),
        "options": {**qa.DEFAULTS, **(s.get("options") or {})},
        "timings": s.get("timings", {}),
        "labels": _labels(),
        "checks": {k: list(v) for k, v in qa.CHECKS.items()},
        "job": active.to_dict() if active is not None else None,
    }


@router.get("/{ds_id}/qa/duplicates")
def qa_duplicates(ds_id: str, offset: int = Query(0, ge=0), limit: int = Query(20, ge=1, le=200),
                  cross_split: int = Query(0), members: int = Query(60, ge=2, le=500)):
    """Near-duplicate groups, biggest first. Each member has `dist` (256-bit dHash bits from the group's
    leader), `leader` (the group's centre), `keep` (the member Export's "drop near-duplicates" keeps:
    approved, then labeled, most boxes, oldest) and `exact` (a byte-identical copy of the leader).
    `cross_split=1`: only groups whose members feed more than one split role (train / val / test)."""
    get_dataset_or_404(ds_id)
    return qa.duplicate_groups(ds_id, offset, limit, bool(cross_split), members)


@router.get("/{ds_id}/images/{image_id}/similar")
def similar_images(ds_id: str, image_id: int, k: int = Query(24, ge=1, le=200), max_dist: int = Query(64, ge=0, le=256)):
    """Most similar images by 256-bit dHash (dist 0 = same picture, <= 2 near-duplicate, ~35 same camera
    view at another moment). Before the dataset's first QA run this queues one and returns `pending`."""
    row = get_dataset_or_404(ds_id)
    res = qa.similar(ds_id, image_id, k, max_dist)
    if res.get("missing"):
        raise HTTPException(404, "Image not found")
    s = qa.load_summary(ds_id)
    if not (s or {}).get("computed_at") and res.get("hashed", 0) < max(2, row["image_count"] // 2):
        active = jobs.active_for(ds_id, "qa") or _auto_queue(ds_id, None, "Checking data quality")
        res["pending"] = True
        res["job"] = active.to_dict() if active is not None else None
    return res


@router.get("/{ds_id}/images/{image_id}/qa")
def image_qa(ds_id: str, image_id: int):
    """This image's QA flags with their details (the twin of a mismatch, the expected box of a gap, ...) and
    `refreshing` (a save of its labels is about to be re-checked)."""
    get_dataset_or_404(ds_id)
    if not db.query_one("SELECT 1 FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id)):
        raise HTTPException(404, "Image not found")
    labels = _labels()
    rows = db.query("SELECT code, score, detail FROM image_flags WHERE image_id=? ORDER BY score DESC, code", (image_id,))
    return {"flags": [{"code": r["code"], "label": labels.get(r["code"], r["code"]), "score": r["score"],
                       "detail": db.loads(r["detail"], {})} for r in rows],
            # its labels were just saved: the label-based flags above are re-checked within a few seconds
            "refreshing": image_id in qa.refreshing_ids(ds_id)}


@router.post("/{ds_id}/interpolate")
def interpolate(ds_id: str, payload: dict = Body(...)):
    """Linear interpolation between two keyframes of one sequence: pending `interp` proposals on every frame
    strictly between them (boxes paired by Hungarian on centre distance + IoU: same class anywhere, another
    class only in place, e.g. a person who falls; such pairs are listed in `class_changes` and take the class
    of the nearer keyframe). Weighted by frame numbers when the names carry them. Never writes label files."""
    get_dataset_or_404(ds_id)
    try:
        a, b = int(payload.get("from_image_id")), int(payload.get("to_image_id"))
    except (TypeError, ValueError):
        raise HTTPException(400, "from_image_id and to_image_id are required") from None
    try:
        return qa.interpolate(ds_id, a, b)
    except qa.QAError as e:
        _bad(e)


@router.post("/{ds_id}/images/{image_id}/fill-gap")
def fill_gap(ds_id: str, image_id: int):
    """Pending `interp` proposals for objects that the nearest labeled frames on both sides have and this
    frame lacks (the fix for a `seq_gap` flag)."""
    get_dataset_or_404(ds_id)
    try:
        return qa.fill_gap(ds_id, image_id)
    except qa.QAError as e:
        _bad(e)


@router.get("/{ds_id}/qa/settings")
def qa_settings(ds_id: str):
    """Per-dataset thresholds: `options` (effective), `defaults`, `overrides` (what differs from the
    defaults) and `fields` [{key, label, help, check, unit, step, min, max, default, value, integer, advanced,
    codes}] — everything a settings form needs."""
    get_dataset_or_404(ds_id)
    return qa.settings(ds_id)


@router.put("/{ds_id}/qa/settings")
def put_qa_settings(ds_id: str, payload: Optional[dict] = Body(None)):
    """Save threshold overrides: {options?: {key: number}, reset?: true | [keys], rerun?: true}. Sending a
    default value (or resetting) removes the override. When a value changed, the checks it drives re-run
    (quality-only changes take seconds: no re-hashing). Returns qa_settings + {changed, job}."""
    row = get_dataset_or_404(ds_id)
    payload = payload or {}
    options = payload.get("options") or {}
    reset = payload.get("reset")
    if not isinstance(options, dict):
        raise HTTPException(400, "options must be an object")
    if reset is not None and reset is not True and reset is not False and not (
            isinstance(reset, list) and all(isinstance(k, str) for k in reset)):
        raise HTTPException(400, "reset must be true or a list of option names")
    rerun = bool(payload.get("rerun", True)) and bool(row["image_count"])
    try:
        return qa.update_settings(ds_id, options, reset if reset is not False else None, rerun=rerun)
    except ValueError as e:
        _bad(e)
