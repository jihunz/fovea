from __future__ import annotations

import os
import threading
import time
import unicodedata
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fastapi import HTTPException, Request

from .. import db
from ..jobs import jobs
from ..paths import to_host

# Image list filters shared by Explore, Annotate, Objects, exports and AI batches (core.imagesq.build_where).
# `seed` is not a filter but travels with them: it picks the permutation of sort=random. `set` is a working-set
# token (POST /api/datasets/{ds}/sets): only those images, in the set's order unless a sort is given.
FILTER_KEYS = ("split", "cls", "only_cls", "labeled", "review", "exclude_review", "issue", "q", "seq", "ids",
               "min_boxes", "max_boxes", "has_label", "ai", "seed", "set")

SET_EXPIRED_MESSAGE = ("This working set has expired or does not exist (sets are kept 24 h after their last use). "
                       "Go back to Explore and select the images again.")


def _filter_value(v: Any) -> Any:
    """Query strings give one string; JSON payloads may give lists (ids, cls) — join those as CSV, the form
    build_where reads. Nested objects never get here: filters_from refuses them (bad_filter)."""
    if isinstance(v, (list, tuple)):
        return ",".join(str(x) for x in v)
    return v


def bad_filter(key: str, value: Any, message: str) -> HTTPException:
    """400 {message, code:'bad_filter', key, value}: a filter value the server does not understand. Never ignored —
    every bulk writer resolves its images through these filters, so a dropped filter would act on the whole dataset."""
    if not isinstance(value, (str, int, float, bool, type(None))):
        value = str(value)
    if isinstance(value, str) and len(value) > 200:
        value = value[:200] + "…"
    return HTTPException(400, detail={"message": message, "code": "bad_filter", "key": key, "value": value})


def check_set(token: Any, ds_id: Optional[str] = None) -> None:
    """410 {message, code:'set_expired'} for a working-set token that is unknown, expired or of another dataset —
    an empty list with no explanation would look like 'no images match'."""
    from ..core.imagesq import set_info
    if set_info(token, ds_id, touch=False) is None:
        raise HTTPException(410, detail={"message": SET_EXPIRED_MESSAGE, "code": "set_expired"})


def filters_from(source: Any, ds_id: Optional[str] = None, skip: Sequence[str] = ()) -> Dict[str, Any]:
    """The image filters of a request's query string or of a JSON `filters` object, validated: a value outside a
    filter's vocabulary (labeled, review, ai…), ids or class ids that are not whole numbers >= 0, an empty `ids`
    list or a nested object is a 400 bad_filter, never silently dropped. `skip`: keys the caller reads itself with
    another meaning (Objects' `cls` = box classes)."""
    from ..core.imagesq import canonical_filters, filter_problem
    keys = [k for k in FILTER_KEYS if k not in skip]
    if source is None:
        return {}
    if isinstance(source, Request):
        qp = source.query_params
        raw = {k: qp.get(k) for k in keys if qp.get(k) not in (None, "")}
        ds_id = ds_id or source.path_params.get("ds_id")
    elif isinstance(source, dict):
        raw = {k: source[k] for k in keys if k in source}
        ds_id = ds_id or (source.get("dataset_id") if isinstance(source.get("dataset_id"), str) else None)
    else:
        raise bad_filter("filters", source, "filters must be an object")
    out: Dict[str, Any] = {}
    for k, v in raw.items():
        # Checked as sent, before lists are joined: an empty `ids` list or a nested object must not just vanish.
        msg = filter_problem(k, v) if k != "seed" else None
        if msg:
            raise bad_filter(k, v, msg)
        v = _filter_value(v)
        if v not in (None, "", []):
            out[k] = v
    out = canonical_filters(out)
    if "set" in out:
        out["set"] = str(out["set"]).strip()
        check_set(out["set"], ds_id)
    return out


def get_dataset_or_404(ds_id: str) -> dict:
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    if not row or row.get("status") == "deleting":      # deleted: its rows are being removed in the background
        raise HTTPException(404, f"Dataset not found: {ds_id}")
    return row


def cached_reachable(row: dict, sample: int = 5) -> dict:
    """Cached reachability (see check_reachable). The dataset list is polled while jobs run; probing every
    dataset's disk on each poll cost ~11 queries and up to 5 stat calls per dataset, and a hung network mount
    hung the whole list. Results are kept for REACH_TTL seconds and a probe that does not answer within
    REACH_WAIT seconds is reported as `checking` (it finishes in the background)."""
    key = (row["id"], row.get("root"), row.get("image_count"), row.get("layout"))
    now = time.time()
    with _reach_lock:
        hit = _reach_cache.get(row["id"])
        if hit and hit[0] == key and now - hit[1] < REACH_TTL:
            return hit[2]
        fut = _reach_inflight.get(row["id"])
        if fut is None or fut.done():
            fut = _reach_pool.submit(_probe_and_store, dict(row), key, sample)
            _reach_inflight[row["id"]] = fut
    try:
        return fut.result(timeout=REACH_WAIT)
    except Exception:  # noqa: BLE001 — timeout (slow/hung mount) or a probe error
        if hit and hit[0][0] == row["id"]:
            return hit[2]
        return {"ok": True, "checking": True, "root_exists": None, "missing_sources": [], "source_count": 0,
                "sampled": 0, "sample_missing": 0}


REACH_TTL = 30.0
REACH_WAIT = 1.5
_reach_cache: Dict[str, tuple] = {}
_reach_inflight: Dict[str, Any] = {}
_reach_lock = threading.Lock()
_reach_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fovea-reach")


def _store_fresh(row: dict) -> dict:
    key = (row["id"], row.get("root"), row.get("image_count"), row.get("layout"))
    return _probe_and_store(dict(row), key, 5)


def forget_reachability(ds_id: str) -> None:
    with _reach_lock:
        _reach_cache.pop(ds_id, None)


def _probe_and_store(row: dict, key: tuple, sample: int) -> dict:
    res = check_reachable(row, sample)
    with _reach_lock:
        _reach_cache[row["id"]] = (key, time.time(), res)
    return res


def check_reachable(row: dict, sample: int = 5) -> dict:
    """Cheap probe (a handful of stat calls) telling the UI whether this dataset's files are
    actually on disk right now. An index built elsewhere — inside Docker, or on a drive that is
    no longer mounted — otherwise looks perfectly healthy while every image 404s."""
    layout = db.loads(row["layout"], {})
    sources = layout.get("sources", [])
    missing_sources: List[str] = []
    for src in sources:
        target = src.get("list_file") or src.get("img_dir")
        if target and not Path(target).exists():
            missing_sources.append(to_host(target))

    # Sample SPREAD across the id range, not the first N rows: a handful of deleted files at the
    # start of a dataset must not masquerade as "the whole dataset is gone".
    missing_files = 0
    checked = 0
    if sample > 0 and row.get("image_count"):
        bounds = db.query_one("SELECT MIN(id) lo, MAX(id) hi FROM images WHERE dataset_id=?", (row["id"],))
        if bounds and bounds["lo"] is not None:
            lo, hi = bounds["lo"], bounds["hi"]
            seen = set()
            for k in range(sample):
                at = lo + ((hi - lo) * k) // max(1, sample - 1) if sample > 1 else lo
                r = db.query_one(
                    "SELECT id, abs_path FROM images WHERE dataset_id=? AND id >= ? ORDER BY id LIMIT 1",
                    (row["id"], at),
                ) or db.query_one(
                    "SELECT id, abs_path FROM images WHERE dataset_id=? ORDER BY id DESC LIMIT 1", (row["id"],)
                )
                if not r or r["id"] in seen:
                    continue
                seen.add(r["id"])
                checked += 1
                if not os.path.isfile(r["abs_path"]):
                    missing_files += 1

    root_exists = bool(row.get("root")) and Path(row["root"]).exists()
    all_sources_gone = bool(sources) and len(missing_sources) == len(sources)
    all_files_gone = checked > 0 and missing_files == checked
    return {
        "ok": not (all_sources_gone or all_files_gone) and (root_exists or not sources),
        "root_exists": root_exists,
        "missing_sources": missing_sources[:5],
        "source_count": len(sources),
        "sampled": checked,
        "sample_missing": missing_files,
    }


SPLIT_RANK = {"train": 0, "val": 1, "test": 2}


def split_rank(name: str):
    """train, val, test, then other named splits in natural order (fold2 before fold10), then the split-less group."""
    from ..core.layout import split_order_key
    return split_order_key(name or "")


def ordered_splits(names) -> List[str]:
    return sorted(set(names), key=split_rank)


def _last_scan(ds_id: str) -> Optional[dict]:
    """The latest scan's report. Reports written before problems became structured carried `problems` as plain
    strings: present them in the current shape [{code, label, severity, count, examples}]."""
    rep = db.kv_get(f"scan:{ds_id}")
    if not isinstance(rep, dict):
        return rep
    probs = rep.get("problems")
    if isinstance(probs, list) and probs and not all(isinstance(p, dict) for p in probs):
        texts = [str(p) for p in probs if not isinstance(p, dict)]
        rep = dict(rep)
        rep["problems"] = [p for p in probs if isinstance(p, dict)] + [
            {"code": "other", "label": "Problems reported by the scan", "severity": "warn", "count": len(texts),
             "examples": texts[:5]}]
    rep.setdefault("problems", [])
    rep.setdefault("notes", [])
    return rep


def dataset_public(row: dict, with_job: bool = True, fresh: bool = False) -> dict:
    """`fresh` probes the disk now (one dataset); the dataset list uses the cached probe."""
    layout = db.loads(row["layout"], {})
    if isinstance(layout.get("sources"), list):
        # Split names come from folder names, which macOS hands out decomposed (NFD Hangul); images.split, filters
        # and facet keys are NFC, so the public copy is NFC too or a lookup by ds.splits finds nothing.
        for s in layout["sources"]:
            if isinstance(s, dict) and isinstance(s.get("split"), str):
                s["split"] = unicodedata.normalize("NFC", s["split"])
        # Datasets registered before detection ordered its sources still list train, val, test, others.
        layout["sources"] = sorted(layout["sources"], key=lambda s: split_rank(s.get("split", "") if isinstance(s, dict) else ""))
    reviews = {r["status"]: r["c"] for r in db.query("SELECT r.status, COUNT(*) c FROM reviews r JOIN images i ON i.dataset_id=r.dataset_id AND i.rel_path=r.rel_path WHERE r.dataset_id=? GROUP BY r.status", (row["id"],))}
    out = {
        "id": row["id"], "name": row["name"], "root": row["root"], "root_host": to_host(row["root"]),
        "layout": layout, "classes": db.loads(row["classes"]), "classes_source": row.get("classes_source", "inferred"),
        "description": row["description"], "status": row["status"],
        "image_count": row["image_count"], "label_count": row["label_count"], "box_count": row["box_count"],
        "cover_image_id": row["cover_image_id"],
        "created_at": row["created_at"], "updated_at": row["updated_at"], "scanned_at": row["scanned_at"],
        "opened_at": row["opened_at"],
        "review": {"approved": reviews.get("approved", 0), "flagged": reviews.get("flagged", 0), "excluded": reviews.get("excluded", 0)},
        "splits": ordered_splits(s.get("split", "") for s in layout.get("sources", [])),
        "reachable": _store_fresh(row) if fresh else cached_reachable(row),
        "last_scan": _last_scan(row["id"]),
    }
    if with_job:
        job = jobs.active_for(row["id"])
        out["job"] = job.to_dict() if job else None
    return out
