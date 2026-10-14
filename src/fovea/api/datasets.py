from __future__ import annotations

import errno
import json
import logging
import math
import mimetypes
import os
import threading
import time
import unicodedata
from pathlib import Path, PurePosixPath
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse

from .. import db
from ..config import IMAGE_EXTS, TEXT_PREVIEW_MAX
from ..core import imagesq, stats
from ..core.labels import (ALL_ISSUE_LABELS, LABEL_LOCK, MAX_CLASS_ID, MAX_INFERRED_CLASSES, ExtendedRowsChanged,
                           LabelLinkOutside, inferred_names, notify_label_write, read_label_details, read_label_rows,
                           read_label_snapshot, validate_boxes, write_label_file)
from ..core.layout import detect, slugify, suggest_name
from ..core.scanner import (refresh_dataset_counts, refresh_image, remap_reviews,  # noqa: F401 — re-exported
                             revalidate_classes, scan_dataset)
from ..jobs import jobs
from ..paths import resolve, to_host
from .jobs import guard_dataset_cancel  # backend-io hook: a remote client may not cancel the owner's provider batch
from .common import check_set, dataset_public, filters_from, forget_reachability, get_dataset_or_404, split_rank

router = APIRouter(prefix="/api/datasets", tags=["datasets"])
log = logging.getLogger("fovea.datasets")

BULK_INLINE_MAX = 3000      # larger bulk label ops run as a background job


def _str_field(payload: dict, key: str, required: bool = False, max_len: int = 4096) -> Optional[str]:
    v = payload.get(key)
    if v is None:
        if required:
            raise HTTPException(400, f"{key} is required")
        return None
    if not isinstance(v, str):
        raise HTTPException(400, f"{key} must be a string")
    return v.strip()[:max_len]


def _body(payload: Any) -> dict:
    if not isinstance(payload, dict):
        raise HTTPException(400, "Expected a JSON object")
    return payload


def _n_classes(row: dict) -> Optional[int]:
    classes = db.loads(row["classes"])
    return len(classes) if row.get("classes_source") != "inferred" and classes else None


def _economy_jobs(ds_id: str) -> set:
    """Ids of the jobs driving this dataset's Economy runs (a provider batch API wait of up to a day)."""
    out = set()
    for r in db.query("SELECT job_id, params FROM ai_runs WHERE dataset_id=? AND kind='batch' AND status='running' "
                      "AND job_id IS NOT NULL", (ds_id,)):
        params = db.loads(r["params"], {}) or {}
        if isinstance(params, dict) and params.get("speed") == "economy":
            out.add(r["job_id"])
    return out


def _reattach_economy(ds_id: str) -> None:
    """Re-attach this dataset's Economy runs that a relocation detached, now that the rescan put every image at
    its new path (economy.resume skips any run whose job is still alive)."""
    try:
        orphan = False
        for r in db.query("SELECT job_id, params FROM ai_runs WHERE dataset_id=? AND kind='batch' AND status='running'",
                          (ds_id,)):
            params = db.loads(r["params"], {}) or {}
            j = jobs.get(r["job_id"]) if r["job_id"] else None
            if isinstance(params, dict) and params.get("speed") == "economy" and (j is None or j.status not in ("queued", "running")):
                orphan = True
        if orphan:
            from ..ai import economy
            economy.resume(force=True)
    except Exception:  # noqa: BLE001 — the run stays 'running' and is picked up at the next start
        log.exception("re-attaching Economy runs of %s failed", ds_id)


def _scan_job(ds_id: str):
    def run(j):
        result = scan_dataset(ds_id, j)
        _reattach_economy(ds_id)
        return result
    return run


def _cancel_dataset_jobs(ds_id: str, wait: float = 10.0, detach: Optional[set] = None) -> bool:
    """Cancel every queued/running job of the dataset and wait for them to stop. True when all stopped.
    Jobs in `detach` (Economy runs, when relocating) are detached instead: the job stops, but the run stays
    'running' and its provider batch is not cancelled (economy checks ``job.detached``); _reattach_economy
    picks it up again."""
    detached = [j for j in jobs.list(ds_id) if detach and j.id in detach and j.status in ("queued", "running")]
    for j in detached:
        j.detached = True
    cancel_all = getattr(jobs, "cancel_all", None)
    if cancel_all is not None:
        hit = cancel_all(ds_id, wait=wait)
    else:
        hit = [j for j in jobs.list(ds_id) if j.status in ("queued", "running")]
        for j in hit:
            j.cancel()
        deadline = time.time() + wait
        while time.time() < deadline and any(j.status in ("queued", "running") for j in hit):
            time.sleep(0.05)
    for j in detached:
        if j.status not in ("queued", "running"):        # say what happened instead of a bare "cancelled"
            j.message = "Paused while the dataset is re-indexed at its new location — the provider batch keeps running"
    return not [j for j in hit if j.status in ("queued", "running")]


# ------------------------------------------------------------------ registry

@router.get("")
def list_datasets():
    rows = db.query("SELECT * FROM datasets ORDER BY COALESCE(opened_at, created_at) DESC")
    for r in rows:
        if r["status"] == "deleting":
            _start_purge(r["id"])          # left half-deleted by a previous process: finish it
    return {"datasets": [dataset_public(r) for r in rows if r["status"] != "deleting"]}


def _detect_or_http(path: str):
    try:
        lay = detect(path, estimate=False)       # the add dialog already showed counts; do not walk the tree again
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not detect layout: {e}")
    if lay.kind == "collection":
        raise HTTPException(400, "; ".join(lay.notes) or "This folder holds several datasets — add each one separately")
    if lay.kind == "unknown" or not lay.sources:
        raise HTTPException(400, "; ".join(lay.notes) or "No dataset found at this path")
    return lay


def _refuse_fovea_dirs(lay=None, path: Optional[str] = None) -> None:
    """A dataset may not be, lie inside or contain Fovea's own data folder (index, settings, secrets.json with the
    API keys, caches) or its model folder — its root nor any source folder a data.yaml points at. Its file
    browser would otherwise hand those files to any client, and its label writes could land among them.
    `path`: the folder as typed, checked before layout detection walks it."""
    from ..core.export import refuse_fovea_dirs
    from ..core.labels import dataset_bases
    bases = [str(resolve(path))] if path else dataset_bases(lay.root, lay.to_dict())
    for base in bases:
        try:
            refuse_fovea_dirs(base, "This dataset folder", contains=True)
        except ValueError as e:
            raise HTTPException(400, detail={"message": str(e), "code": "fovea_folder"})


def _same_place(lay) -> Optional[dict]:
    """An already registered dataset at the same root (and the same data.yaml / list file, if any)."""
    for r in db.query("SELECT id, name, root, layout FROM datasets WHERE root=? AND status!='deleting'", (lay.root,)):
        other = db.loads(r["layout"], {})
        mine = {s.list_file for s in lay.sources if s.list_file}
        theirs = {s.get("list_file") for s in other.get("sources", []) if s.get("list_file")}
        if (other.get("data_yaml") or None) == (lay.data_yaml or None) and mine == theirs:
            return r
    return None


@router.post("")
def create_dataset(payload: Any = Body(...)):
    payload = _body(payload)
    path = _str_field(payload, "path", required=True)
    if not path:
        raise HTTPException(400, "path is required")
    name = _str_field(payload, "name", max_len=200)
    want_id = _str_field(payload, "id", max_len=64)
    description = _str_field(payload, "description") or ""
    user_classes = payload.get("classes")
    if user_classes is not None and (not isinstance(user_classes, list) or not all(isinstance(c, (str, int)) for c in user_classes)):
        raise HTTPException(400, "classes must be a list of names")
    _refuse_fovea_dirs(path=path)
    lay = _detect_or_http(path)
    _refuse_fovea_dirs(lay)
    if not payload.get("allow_duplicate"):
        dup = _same_place(lay)
        if dup:
            raise HTTPException(409, f"This folder is already registered as '{dup['name']}' (id {dup['id']}). "
                                     "Open that dataset, or add it again with allow_duplicate.")
    p = resolve(path)
    name = unicodedata.normalize("NFC", name or suggest_name(p) or "dataset")
    ds_id = slugify(want_id or name)
    base_id, n = ds_id, 2
    while db.query_one("SELECT 1 FROM datasets WHERE id=?", (ds_id,)):
        ds_id = f"{base_id}-{n}"
        n += 1
    if user_classes:
        classes, source = [str(c).strip() or f"class_{i}" for i, c in enumerate(user_classes)], "user"
    else:
        classes = lay.classes
        source = lay.classes_from or ("yaml" if lay.data_yaml and lay.classes else ("file" if lay.classes else "inferred"))
        if not classes:
            source = "inferred"
    now = db.now()
    db.execute("""INSERT INTO datasets(id, name, root, layout, classes, classes_source, description, status, created_at, updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""",
               (ds_id, name, lay.root, db.dumps(lay.to_dict()), db.dumps([str(c) for c in classes]), source,
                description, "new", now, now))
    job = jobs.submit("scan", lambda j: scan_dataset(ds_id, j), dataset_id=ds_id, message="Queued")
    row = get_dataset_or_404(ds_id)
    return {"dataset": dataset_public(row), "job": job.to_dict()}


@router.get("/{ds_id}")
def get_dataset(ds_id: str, touch: int = Query(0)):
    row = get_dataset_or_404(ds_id)
    if touch:
        db.execute("UPDATE datasets SET opened_at=? WHERE id=?", (db.now(), ds_id))
    return {"dataset": dataset_public(row, fresh=True)}


@router.post("/{ds_id}/relocate")
def relocate_dataset(ds_id: str, payload: Any = Body(...), request: Request = None):
    """Point an existing dataset at a new root and re-index it, keeping its id and review marks.
    This is the repair path for an index whose files moved — a different machine, an unmounted
    drive, or an index built inside Docker (container paths) now opened directly on the host."""
    payload = _body(payload)
    get_dataset_or_404(ds_id)
    path = _str_field(payload, "path", required=True)
    if not path:
        raise HTTPException(400, "path is required")
    _refuse_fovea_dirs(path=path)
    lay = _detect_or_http(path)
    _refuse_fovea_dirs(lay)
    # A scan still writing rows for the old root would race the update below and re-index twice. Economy runs
    # are detached, not cancelled: ids survive the relocation, so their provider batches (and the hours already
    # waited) stay valid and the runs are re-attached once the rescan has finished.
    guard_dataset_cancel(request, ds_id, "Relocate")     # backend-io hook: remote AI guard
    if not _cancel_dataset_jobs(ds_id, wait=10.0, detach=_economy_jobs(ds_id)):
        raise HTTPException(409, "A job for this dataset is still stopping — try again in a moment")

    # Every row is kept, and with it every image id that AI proposals and audit findings, QA flags, snapshots,
    # class-card examples and working sets refer to: the rescan updates each row in place by its index key
    # (rel_path), and carries a moved file's references over to its new row (scanner.carry_over). Until a row has
    # been seen at the new root it has no label path, so nothing can be written into the old location; no
    # scanned_at makes an interrupted rescan 'partial' (resumable) instead of a quietly half-moved 'ready'.
    with db.transaction() as conn:
        conn.execute("UPDATE datasets SET root=?, layout=?, status='new', scanned_at=NULL, updated_at=? WHERE id=?",
                     (lay.root, db.dumps(lay.to_dict()), db.now(), ds_id))
        conn.execute("UPDATE images SET label_path=NULL WHERE dataset_id=?", (ds_id,))
    forget_reachability(ds_id)
    imagesq.invalidate_facets(ds_id)

    def run(j):
        result = dict(scan_dataset(ds_id, j) or {})
        result.update(remap_reviews(ds_id))
        _reattach_economy(ds_id)
        return result

    job = jobs.submit("scan", run, dataset_id=ds_id, message="Queued")
    return {"dataset": dataset_public(get_dataset_or_404(ds_id), fresh=True), "job": job.to_dict()}


@router.patch("/{ds_id}")
def update_dataset(ds_id: str, payload: Any = Body(...)):
    payload = _body(payload)
    row = get_dataset_or_404(ds_id)
    sets: List[str] = []
    params: List[Any] = []
    classes_changed = False
    if "name" in payload:
        name = _str_field(payload, "name", max_len=200)
        if name:
            sets.append("name=?"); params.append(unicodedata.normalize("NFC", name))
    if "description" in payload:
        sets.append("description=?"); params.append(_str_field(payload, "description") or "")
    if "classes" in payload:
        cl = payload["classes"]
        if not isinstance(cl, list) or not all(isinstance(c, (str, int, float)) or c is None for c in cl):
            raise HTTPException(400, "classes must be a list of names")
        names = [str(c).strip()[:200] if c is not None and str(c).strip() else f"class_{i}" for i, c in enumerate(cl)]
        sets.append("classes=?"); params.append(db.dumps(names))
        sets.append("classes_source=?"); params.append("user")
        classes_changed = names != db.loads(row["classes"]) or row.get("classes_source") != "user"
    if sets:
        sets.append("updated_at=?"); params.append(db.now())
        params.append(ds_id)
        db.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE id=?", params)
    if classes_changed:
        revalidate_classes(ds_id)            # 'Class id out of range' must follow the list right away
    if payload.get("write_yaml"):
        from ..core.export import write_data_yaml
        row = get_dataset_or_404(ds_id)
        try:
            out = write_data_yaml(row["root"], db.loads(row["classes"]), db.loads(row["layout"], {}))
        except PermissionError as e:
            raise HTTPException(403, f"Cannot write data.yaml: {e.strerror or e}")
        except (OSError, ValueError) as e:
            raise HTTPException(400, f"Cannot write data.yaml: {getattr(e, 'strerror', None) or e}")
        lay = db.loads(row["layout"], {})
        lay["data_yaml"] = out
        lay["data_yaml_host"] = to_host(out)
        db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), ds_id))
    return {"dataset": dataset_public(get_dataset_or_404(ds_id))}


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str, request: Request = None):
    get_dataset_or_404(ds_id)
    # Every job of the dataset (a scan, an export, an AI batch) — not only the newest one.
    guard_dataset_cancel(request, ds_id, "Delete")       # backend-io hook: remote AI guard
    _cancel_dataset_jobs(ds_id, wait=10.0)
    # Gone for every list and endpoint at once; the rows (tens of thousands of images, boxes, snapshot files, AI
    # rows) are deleted on a background thread in short transactions, so other datasets' saves never stall.
    db.execute("UPDATE datasets SET status='deleting', updated_at=? WHERE id=?", (db.now(), ds_id))
    stats.invalidate(ds_id)
    imagesq.invalidate_facets(ds_id)
    forget_reachability(ds_id)
    _start_purge(ds_id)
    return {"ok": True}


DELETE_CHUNK = 1000             # rows per transaction (<= ~0.1 s) while a deleted dataset's rows are removed
_purging: Dict[str, threading.Thread] = {}
_purge_lock = threading.Lock()


def _start_purge(ds_id: str) -> None:
    with _purge_lock:
        th = _purging.get(ds_id)
        if th is not None and th.is_alive():
            return
        th = threading.Thread(target=_purge_dataset, args=(ds_id,), name=f"fovea-delete-{ds_id}"[:60], daemon=True)
        _purging[ds_id] = th
        th.start()


def wait_for_deletes(timeout: float = 60.0) -> bool:
    """Block until every background dataset deletion finished (tests). True when none is left running."""
    end = time.monotonic() + timeout
    while True:
        with _purge_lock:
            alive = [t for t in _purging.values() if t.is_alive()]
        if not alive:
            return True
        if time.monotonic() > end:
            return False
        alive[0].join(min(0.2, max(0.0, end - time.monotonic())))


def _chunked_delete(sql: str, params: tuple, chunk: int = 0) -> int:
    """Run `DELETE ... IN (SELECT ... LIMIT ?)` until nothing is left, one short transaction per chunk."""
    chunk = chunk or DELETE_CHUNK
    total = 0
    while True:
        with db.transaction() as conn:
            n = conn.execute(sql, params + (chunk,)).rowcount or 0
        total += n
        if n < chunk:
            return total


def _purge_dataset(ds_id: str) -> None:
    """Delete a dataset marked 'deleting' in short transactions: its snapshots (each with only its own blobs
    garbage-collected), AI predictions, QA rows, working sets, boxes, images and reviews in chunks, then the
    dataset row itself (whose cascade has little left to do). Safe to run again after an interruption."""
    try:
        _cancel_dataset_jobs(ds_id, wait=10.0)          # a job that was still stopping when the request returned
        from ..core.snapshots import drop_dataset
        drop_dataset(ds_id)
        for sql in (
            "DELETE FROM ai_preds WHERE id IN (SELECT id FROM ai_preds WHERE dataset_id=? LIMIT ?)",
            "DELETE FROM image_flags WHERE rowid IN (SELECT rowid FROM image_flags WHERE dataset_id=? LIMIT ?)",
            "DELETE FROM image_hash WHERE image_id IN (SELECT image_id FROM image_hash WHERE dataset_id=? LIMIT ?)",
            "DELETE FROM ai_audit WHERE image_id IN (SELECT image_id FROM ai_audit WHERE dataset_id=? LIMIT ?)",
            "DELETE FROM boxes WHERE id IN (SELECT id FROM boxes WHERE dataset_id=? LIMIT ?)",
            "DELETE FROM reviews WHERE rowid IN (SELECT rowid FROM reviews WHERE dataset_id=? LIMIT ?)",
        ):
            _chunked_delete(sql, (ds_id,))
        # an image row sits in a dozen indexes and its delete cascades: half the chunk keeps a step near 0.05 s
        _chunked_delete("DELETE FROM images WHERE id IN (SELECT id FROM images WHERE dataset_id=? LIMIT ?)", (ds_id,),
                        max(1, DELETE_CHUNK // 2))
        for ws in db.query("SELECT id FROM work_sets WHERE dataset_id=?", (ds_id,)):
            _chunked_delete("DELETE FROM work_set_items WHERE set_id=? AND pos IN "
                            "(SELECT pos FROM work_set_items WHERE set_id=? LIMIT ?)", (ws["id"], ws["id"]))
        with db.transaction() as conn:
            conn.execute("DELETE FROM plugin_data WHERE dataset_id=?", (ds_id,))
            conn.execute("DELETE FROM kv WHERE key=?", (f"scan:{ds_id}",))
            conn.execute("DELETE FROM datasets WHERE id=? AND status='deleting'", (ds_id,))   # cascades the rest
    except Exception:  # noqa: BLE001 — stays 'deleting' (hidden); the next dataset list resumes it
        log.exception("deleting dataset %s failed", ds_id)
    finally:
        stats.invalidate(ds_id)
        imagesq.invalidate_facets(ds_id)


@router.post("/{ds_id}/scan")
def rescan(ds_id: str, payload: Optional[dict] = Body(None)):
    row = get_dataset_or_404(ds_id)
    active = jobs.active_for(ds_id, "scan")
    if active:
        return {"job": active.to_dict(), "already_running": True}
    if payload and payload.get("redetect"):
        try:
            lay = detect(row["root"], estimate=False)
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Re-detect failed: {e}")
        if lay.sources:
            db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay.to_dict()), ds_id))
            if lay.classes and row.get("classes_source") in ("inferred", "yaml", "file"):
                db.execute("UPDATE datasets SET classes=?, classes_source=? WHERE id=?",
                           (db.dumps(lay.classes), lay.classes_from or ("yaml" if lay.data_yaml else "file"), ds_id))
    job = jobs.submit("scan", _scan_job(ds_id), dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict()}


@router.get("/{ds_id}/stats")
def get_stats(ds_id: str):
    row = get_dataset_or_404(ds_id)
    st = stats.dataset_stats(ds_id, db.loads(row["classes"]))
    return {"stats": st, "issue_labels": ALL_ISSUE_LABELS}


@router.get("/{ds_id}/classes/counts")
def class_counts(ds_id: str):
    """Boxes per class — what Annotate's class palette needs, without the full Overview statistics."""
    get_dataset_or_404(ds_id)
    return {"counts": {str(k): v for k, v in stats.class_counts(ds_id).items()}}


# ------------------------------------------------------------------ class cards

CARD_TEXT_FIELDS = ("description", "include", "exclude", "prompt")
CARD_MAX_EXAMPLES = 24


def _card_public(row: Optional[dict], name: str) -> dict:
    row = row or {}
    return {
        "name": name,
        "description": row.get("description", ""), "include": row.get("include", ""),
        "exclude": row.get("exclude", ""), "prompt": row.get("prompt", ""),
        "examples": db.loads(row.get("examples"), []) if row else [],
        "updated_at": row.get("updated_at"),
    }


def class_cards(ds_id: str) -> Dict[int, dict]:
    """Every class of the dataset with its card (empty fields when none was written). AI prompts use
    `prompt or name` as the class prompt and description + include / exclude as its definition."""
    row = get_dataset_or_404(ds_id)
    names = db.loads(row["classes"])
    stored = {r["cls"]: r for r in db.query("SELECT * FROM class_cards WHERE dataset_id=?", (ds_id,))}
    out: Dict[int, dict] = {}
    for cls in sorted(set(range(len(names))) | set(stored)):
        out[cls] = _card_public(stored.get(cls), names[cls] if 0 <= cls < len(names) else f"class_{cls}")
    return out


@router.get("/{ds_id}/classes/cards")
def get_class_cards(ds_id: str):
    return {"cards": {str(k): v for k, v in class_cards(ds_id).items()}}


def _clean_examples(ds_id: str, value: Any) -> List[dict]:
    if not isinstance(value, list):
        raise HTTPException(400, "examples must be a list of {image_id, row, polarity}")
    out: List[dict] = []
    for ex in value[:CARD_MAX_EXAMPLES]:
        if not isinstance(ex, dict):
            raise HTTPException(400, "each example must be an object {image_id, row, polarity}")
        try:
            iid, row = int(ex.get("image_id")), int(ex.get("row", 0))
            pol = int(ex.get("polarity", 1))
        except (TypeError, ValueError, OverflowError):     # JSON Infinity -> int() overflows
            raise HTTPException(400, "example image_id / row / polarity must be integers")
        if pol not in (1, -1) or row < 0:
            raise HTTPException(400, "example polarity must be 1 or -1 and row >= 0")
        if not db.query_one("SELECT 1 FROM images WHERE id=? AND dataset_id=?", (iid, ds_id)):
            raise HTTPException(400, f"example image {iid} is not in this dataset")
        item = {"image_id": iid, "row": row, "polarity": pol}
        if isinstance(ex.get("box"), list) and len(ex["box"]) >= 4:
            try:
                box = [float(v) for v in ex["box"][:4]]
                if all(math.isfinite(v) for v in box):      # NaN would make the cards response unserialisable
                    item["box"] = box
            except (TypeError, ValueError, OverflowError):
                pass
        if isinstance(ex.get("note"), str):
            item["note"] = ex["note"][:200]
        out.append(item)
    return out


@router.put("/{ds_id}/classes/{cls}/card")
def put_class_card(ds_id: str, cls: int, payload: Any = Body(...)):
    payload = _body(payload)
    row = get_dataset_or_404(ds_id)
    names = db.loads(row["classes"])
    if cls < 0 or cls > MAX_CLASS_ID or (cls >= len(names) and not db.query_one(
            "SELECT 1 FROM boxes WHERE dataset_id=? AND cls=? LIMIT 1", (ds_id, cls))):
        raise HTTPException(404, f"Class {cls} is not in this dataset's class list")
    cur = db.query_one("SELECT * FROM class_cards WHERE dataset_id=? AND cls=?", (ds_id, cls)) or {}
    vals = {k: cur.get(k, "") for k in CARD_TEXT_FIELDS}
    for k in CARD_TEXT_FIELDS:
        if k in payload:
            v = payload[k]
            if v is not None and not isinstance(v, str):
                raise HTTPException(400, f"{k} must be a string")
            vals[k] = (v or "").strip()[:4000]
    examples = db.loads(cur.get("examples"), []) if cur else []
    if "examples" in payload:
        examples = _clean_examples(ds_id, payload["examples"])
    now = db.now()
    db.execute("""INSERT INTO class_cards(dataset_id, cls, description, include, exclude, prompt, examples, updated_at)
                  VALUES(?,?,?,?,?,?,?,?) ON CONFLICT(dataset_id, cls) DO UPDATE SET description=excluded.description,
                  include=excluded.include, exclude=excluded.exclude, prompt=excluded.prompt, examples=excluded.examples,
                  updated_at=excluded.updated_at""",
               (ds_id, cls, vals["description"], vals["include"], vals["exclude"], vals["prompt"], db.dumps(examples), now))
    stored = db.query_one("SELECT * FROM class_cards WHERE dataset_id=? AND cls=?", (ds_id, cls))
    card = _card_public(stored, names[cls] if cls < len(names) else f"class_{cls}")
    return {**card, "cls": cls, "card": card}       # the card itself, also nested under "card"


# ------------------------------------------------------------------ images

@router.get("/{ds_id}/images")
def list_images(ds_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(80, ge=1, le=1000),
                sort: Optional[str] = Query(None), order: Optional[str] = Query(None),
                boxes: int = Query(0), paths: int = Query(0), after: Optional[int] = Query(None),
                count: int = Query(1)):
    """One page of the filtered list. `after=<image id>` pages by key (rows after that image in this order;
    stable under edits, fast at any depth); `count=0` skips the total for follow-up pages."""
    get_dataset_or_404(ds_id)
    f = filters_from(request)
    total, items = imagesq.list_images(ds_id, f, offset, limit, sort, order, with_boxes=bool(boxes), with_paths=bool(paths),
                                       after_id=after, with_total=bool(count))
    return {"total": total, "offset": offset, "limit": limit, "items": items, "filters": f}


@router.get("/{ds_id}/images/ids")
def list_image_ids(ds_id: str, request: Request, sort: Optional[str] = Query(None), order: Optional[str] = Query(None),
                   limit: int = Query(500_000, ge=1, le=2_000_000)):
    """Every id of the filtered, sorted list: a frozen working set for review sessions and 'select all N'."""
    get_dataset_or_404(ds_id)
    f = filters_from(request)
    ids = imagesq.list_ids(ds_id, f, sort, order, limit)
    return {"total": len(ids), "ids": ids, "filters": f}


@router.post("/{ds_id}/images/by-ids")
def images_by_ids(ds_id: str, payload: Any = Body(...)):
    """Rows for a slice of an id working set, in the given order (ids no longer indexed are left out)."""
    payload = _body(payload)
    get_dataset_or_404(ds_id)
    ids = _int_ids(payload.get("ids"))[:2000]
    items = imagesq.items_by_ids(ds_id, ids, with_boxes=bool(payload.get("boxes")), with_paths=bool(payload.get("paths")))
    return {"items": items, "missing": [i for i in ids if i not in {it["id"] for it in items}]}


@router.get("/{ds_id}/images/facets")
def image_facets(ds_id: str, request: Request):
    """Counts per filter value, each under all the OTHER active filters (standard faceted search: how many images
    picking that value would show). Cached per data version + filters."""
    get_dataset_or_404(ds_id)
    f = filters_from(request)
    out, cached = imagesq.facets_cached(ds_id, f)
    return {"facets": out, "filters": f, "issue_labels": ALL_ISSUE_LABELS, "cached": cached}


# ------------------------------------------------------------------ working sets

def _set_http_error(e: "imagesq.SetError") -> HTTPException:
    return HTTPException(e.status, detail={"message": str(e), "code": e.code})


@router.post("/{ds_id}/sets")
def create_work_set(ds_id: str, payload: Any = Body(None)):
    """Freeze an ordered image-id list: {ids?: [...], filters?: {...}, sort?, order?} -> {token, count, expires_at}.
    Every image filter then takes `set=<token>` (only these images; in this order unless another sort is asked
    for). Kept 24 h after its last use; at most 200,000 ids."""
    payload = _body(payload if payload is not None else {})
    get_dataset_or_404(ds_id)
    ids = payload.get("ids")
    if ids is not None and not isinstance(ids, list):
        raise HTTPException(400, "ids must be a list of image ids")
    if payload.get("filters") is not None and not isinstance(payload.get("filters"), dict):
        raise HTTPException(400, "filters must be an object")
    sort, order = payload.get("sort"), payload.get("order")
    if not isinstance(sort, (str, type(None))) or not isinstance(order, (str, type(None))):
        raise HTTPException(400, "sort and order must be strings")
    f = filters_from(payload.get("filters") or {}, ds_id)
    try:
        out = imagesq.create_set(ds_id, ids, f, sort, order, source=_str_field(payload, "source", max_len=40) or "")
    except imagesq.SetError as e:
        raise _set_http_error(e)
    return {**out, "dataset_id": ds_id}


@router.get("/{ds_id}/sets/{token}")
def get_work_set(ds_id: str, token: str):
    """Whether a working set is still alive (410 code set_expired when not), its size and how it was made."""
    get_dataset_or_404(ds_id)
    info = imagesq.set_info(token, ds_id)
    if info is None:
        check_set(token, ds_id)
    return {"token": info["token"], "count": info["count"], "created_at": info["created_at"],
            "expires_at": info["expires_at"], "params": info["params"], "dataset_id": ds_id}


@router.delete("/{ds_id}/sets/{token}")
def delete_work_set(ds_id: str, token: str):
    get_dataset_or_404(ds_id)
    return {"deleted": imagesq.delete_set(token, ds_id)}


@router.get("/{ds_id}/images/neighbor")
def neighbor(ds_id: str, request: Request, id: int = Query(...), dir: str = Query("next"),
             sort: Optional[str] = Query(None), order: Optional[str] = Query(None), need: Optional[str] = Query(None)):
    get_dataset_or_404(ds_id)
    item = imagesq.neighbor(ds_id, filters_from(request), id, dir, sort, order, need)
    return {"item": item}


@router.get("/{ds_id}/images/position")
def image_position(ds_id: str, request: Request, id: int = Query(...), sort: Optional[str] = Query(None), order: Optional[str] = Query(None)):
    """0-based position of an image inside the filtered, sorted list; null when the image is not in it."""
    get_dataset_or_404(ds_id)
    if not db.query_one("SELECT 1 FROM images WHERE id=? AND dataset_id=?", (id, ds_id)):
        raise HTTPException(404, "Image not found")
    return {"position": imagesq.position(ds_id, filters_from(request), id, sort, order)}


@router.get("/{ds_id}/images/{image_id}")
def get_image(ds_id: str, image_id: int):
    get_dataset_or_404(ds_id)
    row = db.query_one(f"{imagesq.BASE_SELECT} WHERE i.id=? AND i.dataset_id=?", (image_id, ds_id))
    if not row:
        raise HTTPException(404, "Image not found")
    item = imagesq.row_to_item(row, with_paths=True)
    imagesq.attach_boxes([item])
    imagesq.attach_ai([item])
    return {"item": item}


@router.get("/{ds_id}/images/{image_id}/labels")
def get_labels(ds_id: str, image_id: int):
    row = get_dataset_or_404(ds_id)
    img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id))
    if not img:
        raise HTTPException(404, "Image not found")
    lp = Path(img["label_path"]) if img["label_path"] else None
    det = read_label_details(lp, _n_classes(row)) if lp and lp.is_file() else None
    if det is None:
        det = {"boxes": [], "confs": [], "issues": set(), "row_issues": [], "row_kinds": [], "version": None}
    return {"image_id": image_id, "boxes": det["boxes"], "confs": det["confs"], "issues": sorted(det["issues"]),
            "exists": det["version"] is not None, "version": det["version"],
            # Which box each issue belongs to, and which rows are polygons / keypoints (geometry-locked).
            "row_issues": det["row_issues"], "row_kinds": det["row_kinds"],
            "label_path": str(lp) if lp else None, "label_path_host": to_host(lp) if lp else None}


AUTO_EXTEND_LIMIT = 32   # an inferred class list grows to cover a new id only when the gap is this small


def _write_error(e: OSError, lp: Path) -> HTTPException:
    where = to_host(lp)
    if isinstance(e, PermissionError) or e.errno in (errno.EACCES, errno.EPERM, errno.EROFS):
        return HTTPException(403, f"Cannot write the label file {where}: permission denied (read-only folder?)")
    if e.errno == errno.ENOSPC:
        return HTTPException(507, f"Cannot write the label file {where}: the disk is full")
    return HTTPException(500, f"Cannot write the label file {where}: {e.strerror or 'I/O error'}")


# --- annotate hook (remove_if_empty) ---
def _remove_label_file(lp: Path) -> bool:
    """An editor that emptied an image which had no label file (draw, then undo) sends remove_if_empty: the image goes
    back to "no label file" instead of becoming a 0-byte background file, and nothing is created when there is none.
    A symlinked label is written through as before. True when nothing is left to write."""
    if lp.is_symlink():
        return False
    lp.unlink(missing_ok=True)
    return True
# --- end annotate hook ---


@router.put("/{ds_id}/images/{image_id}/labels")
def put_labels(ds_id: str, image_id: int, payload: Any = Body(...)):
    payload = _body(payload)
    row = get_dataset_or_404(ds_id)
    img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id))
    if not img:
        raise HTTPException(404, "Image not found")
    if not img["label_path"]:
        raise HTTPException(400, "No label path for this image")
    if not isinstance(payload.get("boxes"), list):
        # A body without a box list used to write an empty file: never guess "delete everything".
        raise HTTPException(400, "boxes must be a list (send [] to remove every box)")
    lp = Path(img["label_path"])
    sent = payload["boxes"]
    boxes, rejected = validate_boxes(sent)
    if sent and not boxes:
        raise HTTPException(400, f"None of the {len(sent)} box(es) could be stored (malformed rows) — nothing was written")

    # Optimistic concurrency: the editor sends the version it loaded. If the file changed underneath it
    # (another tab, a bulk op, an external tool) we refuse rather than silently resurrect or erase boxes.
    with LABEL_LOCK:     # endpoints run on a thread pool: the check and the write must not interleave
        if "base_version" in payload:
            b, c, _i, current = read_label_snapshot(lp)
            if payload.get("base_version") != current:
                theirs = [list(x[:5]) + ([cf] if cf is not None else []) for x, cf in zip(b, c)]
                raise HTTPException(409, detail={
                    "message": "The label file was changed elsewhere since it was loaded.",
                    "version": current, "boxes": theirs,
                })
        try:
            if not (not boxes and payload.get("remove_if_empty") is True and _remove_label_file(lp)):   # annotate hook
                write_label_file(lp, boxes, strict=True, dataset_id=ds_id)
        except LabelLinkOutside as e:
            # 403, not 409: the editor reads 409 as "changed elsewhere" and would offer to merge with a file it cannot write.
            raise HTTPException(403, detail={"message": f"Cannot save {to_host(lp)}: {e.strerror}.", "code": "label_link_outside"})
        except ExtendedRowsChanged as e:
            b, c, _i, current = read_label_snapshot(lp)
            raise HTTPException(409, detail={"message": str(e), "code": "extended_rows", "version": current,
                                             "boxes": [list(x[:5]) + ([cf] if cf is not None else []) for x, cf in zip(b, c)]})
        except OSError as e:
            raise _write_error(e, lp)
        b, c, _i, version = read_label_snapshot(lp)
        boxes = [list(x[:5]) + ([cf] if cf is not None else []) for x, cf in zip(b, c)]   # exactly what is on disk
    classes = db.loads(row["classes"])
    n_classes = _n_classes(row)
    max_cls = max((int(b[0]) for b in boxes if int(b[0]) < MAX_INFERRED_CLASSES), default=-1)   # larger: bad_class
    if (row.get("classes_source") == "inferred"
            and len(classes) <= max_cls < min(len(classes) + AUTO_EXTEND_LIMIT, MAX_INFERRED_CLASSES)):
        classes = inferred_names(classes, max_cls)
        db.execute("UPDATE datasets SET classes=? WHERE id=? AND classes_source='inferred'", (db.dumps(classes), ds_id))
    refresh_image(image_id, n_classes)
    notify_label_write(ds_id, [image_id])
    r2 = db.query_one(f"{imagesq.BASE_SELECT} WHERE i.id=?", (image_id,))
    item = imagesq.row_to_item(r2, with_paths=True)
    item["boxes"] = boxes
    return {"item": item, "saved": len(boxes), "rejected": rejected, "version": version}


def _int_field(payload: dict, key: str, required: bool = True, lo: int = 0, hi: int = MAX_CLASS_ID) -> Optional[int]:
    v = payload.get(key)
    if v is None or v == "":
        if required:
            raise HTTPException(400, f"{key} is required")
        return None
    if isinstance(v, bool):
        raise HTTPException(400, f"{key} must be an integer")
    try:
        f = float(v)
    except (TypeError, ValueError, OverflowError):
        raise HTTPException(400, f"{key} must be an integer")
    if not math.isfinite(f) or f != int(f) or not (lo <= int(f) <= hi):   # 'NaN' / 'inf' parse as floats
        raise HTTPException(400, f"{key} must be an integer between {lo} and {hi}")
    return int(f)


def _class_limit(row: dict) -> int:
    """Highest class id a bulk op may write: the class list (or, for an inferred list, a small gap above it)."""
    classes = db.loads(row["classes"])
    if row.get("classes_source") == "inferred":
        return max(min(len(classes) + AUTO_EXTEND_LIMIT, MAX_INFERRED_CLASSES) - 1, len(classes) - 1, 0)
    return max(len(classes) - 1, 0)


def _validate_bulk(row: dict, payload: dict) -> dict:
    """Check a bulk request completely before any file is touched (a bad value used to 500 mid-job after
    some files were already rewritten)."""
    op = payload.get("op")
    if op not in ("set_class", "remap", "delete_class", "clear"):
        raise HTTPException(400, "op must be one of set_class, remap, delete_class, clear")
    hi = _class_limit(row)
    spec: Dict[str, Any] = {"op": op}
    if op == "set_class":
        spec["cls"] = _int_field(payload, "cls", hi=hi)
        spec["only_cls"] = _int_field(payload, "only_cls", required=False)
    elif op == "delete_class":
        spec["cls"] = _int_field(payload, "cls")
    elif op == "remap":
        mapping = payload.get("mapping")
        if not isinstance(mapping, dict) or not mapping:
            raise HTTPException(400, "mapping must be an object {from_class: to_class}")
        clean: Dict[int, int] = {}
        for k, v in mapping.items():
            src = _int_field({"from": k}, "from")
            dst = _int_field({"to": v}, "to", hi=hi)
            clean[src] = dst
        spec["mapping"] = clean
    if payload.get("image_ids") is not None and not isinstance(payload.get("image_ids"), list):
        raise HTTPException(400, "image_ids must be a list")
    rng = payload.get("range")
    if rng is not None:
        if not isinstance(rng, dict):
            raise HTTPException(400, "range must be {from_id, to_id}")
        _int_field(rng, "from_id", hi=2 ** 62)
        _int_field(rng, "to_id", hi=2 ** 62)
    if payload.get("filters") is not None and not isinstance(payload.get("filters"), dict):
        raise HTTPException(400, "filters must be an object")
    return spec


@router.post("/{ds_id}/labels/bulk")
def bulk_labels(ds_id: str, payload: Any = Body(...)):
    """Bulk label ops: set_class | remap | delete_class | clear over a selection. The label files about to
    change are snapshotted first (response `snapshot_id`; restoring it undoes the op)."""
    payload = _body(payload)
    row = get_dataset_or_404(ds_id)
    spec = _validate_bulk(row, payload)
    op = spec["op"]
    targets = _select_target_ids(ds_id, payload)
    if not targets:
        return {"changed": 0, "images": 0, "snapshot_id": None}
    n_classes = _n_classes(row)

    def run(job=None):
        from ..core.snapshots import WriteLog, snapshot_images
        if job:
            job.update(done=0, total=len(targets), message="Saving a snapshot of the label files")
        sid = snapshot_images(ds_id, targets, op=f"bulk {op}", only_existing=True, track=True)
        wlog = WriteLog(sid)          # what this op wrote: its Undo reverts only files still as it left them
        changed = images = 0
        written: List[int] = []
        refused: List[str] = []
        try:
            for k, iid in enumerate(targets):
                if job and k % 20 == 0:
                    job.update(done=k, total=len(targets), message=f"{op} {k:,}/{len(targets):,}")
                img = db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))
                if not img or not img["label_path"]:
                    continue
                lp = Path(img["label_path"])
                try:
                    with LABEL_LOCK:
                        n_changed = _bulk_rewrite(lp, spec, ds_id)
                        if n_changed is not None:
                            wlog.wrote(img["label_path"])
                except LabelLinkOutside:
                    refused.append(to_host(lp))       # a link out of the dataset: left alone, reported
                    continue
                if n_changed is None:
                    continue
                changed += n_changed
                refresh_image(iid, n_classes, update_counts=False)
                written.append(iid)
                images += 1
        finally:
            refresh_dataset_counts(ds_id)   # once, even if the job is cancelled part-way
            notify_label_write(ds_id, written)
            sid = wlog.close()              # the undo point keeps exactly the files changed (none: no undo point)
        if spec.get("cls") is not None and op == "set_class":
            _grow_inferred(ds_id, spec["cls"])
        elif op == "remap":
            _grow_inferred(ds_id, max(spec["mapping"].values()))
        out = {"changed": changed, "images": images, "snapshot_id": sid}
        if refused:
            out["skipped_links"] = len(refused)
            out["problems"] = [f"{p}: the label file is a link to a file outside the dataset — not changed" for p in refused[:5]]
        return out

    if len(targets) > BULK_INLINE_MAX:
        job = jobs.submit("bulk_labels", run, dataset_id=ds_id, message=op)
        return {"job": job.to_dict()}
    try:
        return run()
    except OSError as e:
        raise HTTPException(403 if isinstance(e, PermissionError) else 500,
                            f"Bulk {op} stopped: {e.strerror or e} ({to_host(getattr(e, 'filename', '') or '')})")


def _grow_inferred(ds_id: str, max_cls: int) -> None:
    row = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (ds_id,))
    if not row or row["classes_source"] != "inferred":
        return
    classes = db.loads(row["classes"])
    if len(classes) <= max_cls < MAX_INFERRED_CLASSES:
        db.execute("UPDATE datasets SET classes=? WHERE id=? AND classes_source='inferred'",
                   (db.dumps(inferred_names(classes, max_cls)), ds_id))


def _bulk_rewrite(lp: Path, spec: dict, ds_id: Optional[str] = None) -> Optional[int]:
    """Apply one bulk op to one label file. Returns the number of changed (re-classed or removed) rows, or
    None if the file was not touched. Polygon / keypoint rows keep their geometry (write_label_file)."""
    if not lp.is_file():
        return None           # nothing to change — and "clear" must not invent an empty label file
    rows, _issues = read_label_rows(lp)   # conf stays inside each row, so filtering cannot misassign it
    op = spec["op"]
    if op == "clear":
        new: List[list] = []
    else:
        new = []
        for b in rows:
            c = int(b[0])
            if op == "set_class":
                if spec.get("only_cls") is None or c == spec["only_cls"]:
                    b = [spec["cls"]] + list(b[1:])
            elif op == "remap":
                if c in spec["mapping"]:
                    b = [spec["mapping"][c]] + list(b[1:])
            elif op == "delete_class":
                if c == spec["cls"]:
                    continue
            new.append(b)
    if new == rows:
        return None
    write_label_file(lp, new, dataset_id=ds_id)
    if op in ("clear", "delete_class"):
        return len(rows) - len(new)
    return sum(1 for a, b2 in zip(rows, new) if int(a[0]) != int(b2[0]))


def _int_ids(values: Any) -> List[int]:
    """Coerce a client-supplied id list, dropping anything that is not an integer. A stray null must
    not surface as a 500 with a raw TypeError."""
    out: List[int] = []
    if not isinstance(values, (list, tuple)):
        return out
    for v in values:
        if isinstance(v, bool):
            continue
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _payload_filters(ds_id: str, payload: dict) -> Dict[str, Any]:
    """The validated `filters` object of a write (absent: none). Anything but an object is a 400, never "all images"."""
    raw = payload.get("filters")
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise HTTPException(400, detail={"message": "filters must be an object", "code": "bad_filter", "key": "filters"})
    return filters_from(raw, ds_id)


def _select_target_ids(ds_id: str, payload: dict) -> List[int]:
    """The images a write acts on. image_ids, when sent, is the whole selection (an empty or unusable list selects
    nothing — it never falls through to the filters); else a range and/or filters."""
    if payload.get("image_ids") is not None:
        ids = _int_ids(payload["image_ids"])
        if not ids:
            return []
        rows = db.query("SELECT id FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?)) ORDER BY id",
                        (ds_id, json.dumps(ids)))
        return [r["id"] for r in rows]
    rng = payload.get("range")
    if isinstance(rng, dict) and rng.get("from_id") is not None and rng.get("to_id") is not None:
        try:
            a, b = sorted((int(rng["from_id"]), int(rng["to_id"])))
        except (TypeError, ValueError):
            raise HTTPException(400, "range from_id / to_id must be image ids")
        f = _payload_filters(ds_id, payload)
        where, params = imagesq.build_where(ds_id, f)
        rows = db.query(f"SELECT i.id FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where} AND i.id BETWEEN ? AND ? ORDER BY i.id", params + [a, b])
        return [r["id"] for r in rows]
    if payload.get("filters") is not None:
        f = _payload_filters(ds_id, payload)
        where, params = imagesq.build_where(ds_id, f)
        rows = db.query(f"SELECT i.id FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where} ORDER BY i.id", params)
        return [r["id"] for r in rows]
    return []


# ------------------------------------------------------------------ review

@router.put("/{ds_id}/review")
def set_review(ds_id: str, payload: Any = Body(...)):
    """Set or clear a review status, and/or a note, for a selection.

    * {status, note?}: status approved|flagged|excluded|null. Without `note` an existing note is kept.
    * {note} without `status`: only the note changes (the status stays as it is).
    Clearing a status keeps the note; a row is deleted only when it has neither."""
    payload = _body(payload)
    get_dataset_or_404(ds_id)
    has_status = "status" in payload
    has_note = "note" in payload
    if not has_status and not has_note:
        raise HTTPException(400, "send a status (approved|flagged|excluded|null) and/or a note")
    status = payload.get("status")
    if has_status and status not in ("approved", "flagged", "excluded", None):
        raise HTTPException(400, "status must be approved|flagged|excluded|null")
    note = payload.get("note")
    if has_note and note is not None and not isinstance(note, str):
        raise HTTPException(400, "note must be a string")
    note = (note or "")[:4000]
    if payload.get("image_ids") is not None and not isinstance(payload.get("image_ids"), list):
        raise HTTPException(400, "image_ids must be a list")
    if payload.get("rel_paths") is not None and not isinstance(payload.get("rel_paths"), list):
        raise HTTPException(400, "rel_paths must be a list")
    ids = _int_ids(payload.get("image_ids"))
    rels: List[str] = []
    asked = [unicodedata.normalize("NFC", r) for r in (payload.get("rel_paths") or []) if isinstance(r, str)]
    if asked:
        # Only paths that are images of this dataset: unknown paths used to create orphan review rows.
        for i in range(0, len(asked), 800):
            chunk = asked[i:i + 800]
            rels.extend(r["rel_path"] for r in db.query(
                "SELECT rel_path FROM images WHERE dataset_id=? AND rel_path IN (%s)" % ",".join("?" * len(chunk)), [ds_id] + chunk))
    if payload.get("filters") is not None and payload.get("image_ids") is None and payload.get("rel_paths") is None:
        # Filters only when no explicit selection was sent: image_ids that are all unusable select nothing.
        ids = _select_target_ids(ds_id, {"filters": payload["filters"]})
    if ids:
        for i in range(0, len(ids), 800):
            chunk = ids[i:i + 800]
            rels.extend(r["rel_path"] for r in db.query("SELECT rel_path FROM images WHERE dataset_id=? AND id IN (%s)" % ",".join("?" * len(chunk)), [ds_id] + chunk))
    rels = list(dict.fromkeys(rels))
    if not rels:
        return {"updated": 0, "status": status}
    now = db.now()
    with db.transaction() as conn:
        if has_status and status is None and has_note:
            conn.executemany("""INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,NULL,?,?)
                                ON CONFLICT(dataset_id, rel_path) DO UPDATE SET status=NULL, note=excluded.note,
                                updated_at=excluded.updated_at""", [(ds_id, r, note, now) for r in rels])
        elif has_status and status is None:
            for i in range(0, len(rels), 800):
                chunk = rels[i:i + 800]
                conn.execute("UPDATE reviews SET status=NULL, updated_at=? WHERE dataset_id=? AND rel_path IN (%s)"
                             % ",".join("?" * len(chunk)), [now, ds_id] + chunk)
        elif has_status:
            keep = "note=excluded.note" if has_note else "note=reviews.note"
            conn.executemany(f"""INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)
                                ON CONFLICT(dataset_id, rel_path) DO UPDATE SET status=excluded.status, {keep}, updated_at=excluded.updated_at""",
                             [(ds_id, r, status, note, now) for r in rels])
        else:
            # Note only: the status (and the time it was reviewed, which the 'reviewed' sort uses) is untouched.
            conn.executemany("""INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,NULL,?,?)
                                ON CONFLICT(dataset_id, rel_path) DO UPDATE SET note=excluded.note,
                                updated_at=CASE WHEN reviews.status IS NULL THEN excluded.updated_at ELSE reviews.updated_at END""",
                             [(ds_id, r, note, now) for r in rels])
        for i in range(0, len(rels), 800):
            chunk = rels[i:i + 800]
            conn.execute("DELETE FROM reviews WHERE dataset_id=? AND status IS NULL AND note='' AND rel_path IN (%s)"
                         % ",".join("?" * len(chunk)), [ds_id] + chunk)
    out = {"updated": len(rels), "status": status if has_status else None}
    if has_note:
        out["note"] = note
    return out


@router.get("/{ds_id}/sequences")
def sequences(ds_id: str, split: Optional[str] = Query(None), limit: int = Query(400, ge=1, le=100_000)):
    get_dataset_or_404(ds_id)
    params: List[Any] = [ds_id]
    where = "dataset_id=? AND seq != ''"
    if split:
        where += " AND split=?"; params.append(unicodedata.normalize("NFC", split))
    rows = db.query(f"SELECT seq, split, COUNT(*) n, MIN(id) first_id, SUM(n_boxes) boxes FROM images WHERE {where} "
                    "GROUP BY seq, split HAVING n > 1", params)
    rows.sort(key=lambda r: (split_rank(r["split"]), r["seq"]))
    return {"sequences": rows[:limit], "total": len(rows)}


# ------------------------------------------------------------------ files

def _dataset_bases(row: dict) -> List[Path]:
    bases = [Path(row["root"]).resolve()]
    for s in (db.loads(row["layout"], {}) or {}).get("sources", []):
        for k in ("img_dir", "label_dir", "base"):
            if s.get(k):
                bases.append(Path(s[k]).resolve())
    return bases


def _safe_child(row: dict, rel: str) -> Path:
    """A path inside the dataset, as the user sees it (symlinks unresolved).

    Containment is checked on the resolved target, which may lie under the dataset root OR under one of its
    own source folders — so `images/train -> /data/train` works — while a link to anywhere else (say, a
    `notes -> /etc` shipped inside a downloaded dataset) is still refused: this endpoint returns content."""
    root = Path(row["root"])
    parts = [x for x in PurePosixPath(rel or "").parts if x not in ("", ".")]
    if (rel or "").startswith("/") or ".." in parts:
        raise HTTPException(400, "Path escapes dataset root")
    cand = root.joinpath(*parts)
    if not cand.exists():
        # A typed (NFC) name for a folder stored decomposed (NFD), or the reverse, on a file system that
        # does not treat them as the same name.
        for form in ("NFD", "NFC"):
            alt = root.joinpath(*[unicodedata.normalize(form, x) for x in parts])
            if alt.exists():
                cand = alt
                break
    real = cand.resolve()
    from ..core.export import refuse_fovea_dirs
    try:
        # Never Fovea's own folders, even inside a dataset registered before that was refused: secrets.json
        # (API keys), the database and the caches must not be readable through the file browser.
        refuse_fovea_dirs(real, "This path")
    except ValueError:
        raise HTTPException(403, "This path is inside Fovea's own data or model folder")
    for base in _dataset_bases(row):
        try:
            real.relative_to(base)
            return cand
        except ValueError:
            continue
    raise HTTPException(400, "Path escapes dataset root")


TEXT_KINDS = (".txt", ".yaml", ".yml", ".md", ".json", ".csv", ".names", ".cache", ".log", ".xml", ".tsv", ".ini", ".cfg")
FILES_PAGE_MAX = 5000


def _image_ids_for(ds_id: str, folder: Path, names: List[str]) -> Dict[str, int]:
    """Indexed image ids for files of one folder, in ONE query (it used to be one unindexed lookup per
    entry: minutes for a 24k-file folder). Names are matched in both Unicode normal forms: the stored
    path is what the file system returned, the requested folder may have been typed (NFC)."""
    if not names:
        return {}
    want: Dict[str, str] = {}
    for n in names:
        full = os.path.join(str(folder), n)
        for form in (full, unicodedata.normalize("NFC", full), unicodedata.normalize("NFD", full)):
            want[form] = n
    found: Dict[str, int] = {}
    keys = list(want)
    for k in range(0, len(keys), 20000):
        for r in db.query("SELECT id, abs_path FROM images WHERE dataset_id=? AND abs_path IN (SELECT value FROM json_each(?))",
                          (ds_id, json.dumps(keys[k:k + 20000]))):
            found[want[r["abs_path"]]] = r["id"]
    return found


@router.get("/{ds_id}/files")
def list_files(ds_id: str, path: str = Query(""), offset: int = Query(0, ge=0),
               limit: int = Query(FILES_PAGE_MAX, ge=1, le=FILES_PAGE_MAX)):
    row = get_dataset_or_404(ds_id)
    root = Path(row["root"])
    rel_dir = path.strip("/")
    d = _safe_child(row, rel_dir)
    if not d.is_dir():
        raise HTTPException(404, "Not a directory")
    listing = []
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.name.startswith("."):
                    continue
                try:
                    is_dir = e.is_dir()           # follows links: images/train -> /data/train is a folder
                except OSError:
                    is_dir = False
                listing.append((not is_dir, e.name.casefold(), e.name, is_dir))
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    except OSError as e:
        raise HTTPException(503, f"Cannot read this folder: {e.strerror or e}")
    listing.sort()
    total = len(listing)
    page = listing[offset:offset + limit]           # per-entry work only for the page that is returned
    images = [name for _f, _k, name, is_dir in page if not is_dir and Path(name).suffix.lower() in IMAGE_EXTS]
    ids = _image_ids_for(ds_id, d, images)
    entries = []
    for _f, _k, name, is_dir in page:
        child = d / name
        try:
            st = child.stat()
        except OSError:
            st = None
        suf = Path(name).suffix.lower()
        kind = "dir" if is_dir else ("image" if suf in IMAGE_EXTS else ("text" if suf in TEXT_KINDS else "file"))
        e = {"name": name, "rel": f"{rel_dir}/{name}" if rel_dir else name, "is_dir": is_dir,
             "size": None if (is_dir or st is None) else st.st_size, "mtime": st.st_mtime if st else None, "kind": kind}
        if kind == "image":
            e["image_id"] = ids.get(name)
        entries.append(e)
    crumbs = []
    acc = ""
    for part in [p for p in rel_dir.split("/") if p]:
        acc = f"{acc}/{part}" if acc else part
        crumbs.append({"name": part, "rel": acc})
    return {"root_host": to_host(root), "path": rel_dir, "crumbs": crumbs, "entries": entries,
            "total": total, "offset": offset, "limit": limit, "truncated": offset + len(page) < total}


def _decode_preview(data: bytes, cut: bool):
    """Text of a preview chunk: (text, encoding). A chunk cut by the size limit ends at its last line break,
    never inside a line or a multi-byte character."""
    if cut:
        nl = data.rfind(b"\n")
        if nl > len(data) // 2:
            data = data[:nl + 1]
    for enc in ("utf-8-sig", "cp949"):
        try:
            return data.decode(enc), ("utf-8" if enc == "utf-8-sig" else enc), len(data)
        except UnicodeDecodeError:
            if enc == "utf-8-sig" and cut:
                # The cut may still split a character when there was no line break to cut at.
                for trim in (1, 2, 3):
                    try:
                        return data[:-trim].decode("utf-8-sig"), "utf-8", len(data) - trim
                    except UnicodeDecodeError:
                        continue
            continue
    return data.decode("latin-1"), "latin-1", len(data)


@router.get("/{ds_id}/file")
def get_file(ds_id: str, path: str = Query(...), raw: int = Query(0), offset: int = Query(0, ge=0),
             limit: int = Query(TEXT_PREVIEW_MAX, ge=1, le=4 * TEXT_PREVIEW_MAX)):
    row = get_dataset_or_404(ds_id)
    p = _safe_child(row, path.strip("/"))
    if not p.is_file():
        raise HTTPException(404, "File not found")
    if p.suffix.lower() in IMAGE_EXTS and not raw:
        try:
            from ..core.thumbs import needs_rendition, rendition_for
            if needs_rendition(str(p)):
                # TIFF and friends: browsers cannot show them, so preview a JPEG rendition (Download stays raw).
                out = rendition_for(str(p))
                if out is None:
                    raise HTTPException(415, "This image format cannot be previewed")
                return FileResponse(out, media_type="image/jpeg", headers={"Cache-Control": "private, no-cache"})
        except ImportError:
            pass
        return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream",
                            headers={"Cache-Control": "private, no-cache"})
    if raw:
        # Never serve a dataset file as a document on this origin: an .html/.svg/.js shipped inside a
        # dataset would run with full access to this API. Everything downloads.
        return FileResponse(p, media_type="application/octet-stream", filename=p.name)
    size = p.stat().st_size
    with p.open("rb") as fh:
        fh.seek(offset)
        data = fh.read(limit)
    if size == 0:
        return {"path": path, "size": 0, "empty": True, "binary": False, "text": "", "offset": 0,
                "next_offset": None, "truncated": False, "encoding": None}
    if b"\x00" in data[:8192]:
        return {"path": path, "size": size, "empty": False, "binary": True, "text": "", "offset": offset,
                "next_offset": None, "truncated": False, "encoding": None}
    cut = offset + len(data) < size
    text, enc, used = _decode_preview(data, cut)
    end = offset + used
    return {"path": path, "size": size, "empty": False, "binary": False, "text": text, "encoding": enc,
            "offset": offset, "shown": used, "next_offset": end if end < size else None, "truncated": end < size}
