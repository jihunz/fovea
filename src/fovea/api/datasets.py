from __future__ import annotations

import mimetypes
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import FileResponse

from .. import db
from ..config import IMAGE_EXTS, TEXT_PREVIEW_MAX
from ..core import imagesq, stats
from ..core.labels import (ISSUE_LABELS, LABEL_LOCK, read_label_rows, read_label_snapshot, validate_boxes,
                           write_label_file)
from ..core.layout import detect, slugify
from ..core.scanner import refresh_dataset_counts, refresh_image, scan_dataset
from ..jobs import jobs
from ..paths import resolve, to_host
from .common import dataset_public, filters_from, get_dataset_or_404

router = APIRouter(prefix="/api/datasets", tags=["datasets"])


# ------------------------------------------------------------------ registry

@router.get("")
def list_datasets():
    rows = db.query("SELECT * FROM datasets ORDER BY COALESCE(opened_at, created_at) DESC")
    return {"datasets": [dataset_public(r) for r in rows]}


@router.post("")
def create_dataset(payload: dict = Body(...)):
    path = (payload.get("path") or "").strip()
    if not path:
        raise HTTPException(400, "path is required")
    try:
        lay = detect(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not detect layout: {e}")
    if lay.kind == "unknown" or not lay.sources:
        raise HTTPException(400, "; ".join(lay.notes) or "No dataset found at this path")
    p = resolve(path)
    name = (payload.get("name") or (p.stem if p.is_file() else p.name) or "dataset").strip()
    ds_id = slugify(payload.get("id") or name)
    base_id, n = ds_id, 2
    while db.query_one("SELECT 1 FROM datasets WHERE id=?", (ds_id,)):
        ds_id = f"{base_id}-{n}"
        n += 1
    classes = payload.get("classes") if isinstance(payload.get("classes"), list) else lay.classes
    source = "user" if payload.get("classes") else ("yaml" if lay.data_yaml and lay.classes else ("file" if lay.classes else "inferred"))
    now = db.now()
    db.execute("""INSERT INTO datasets(id, name, root, layout, classes, classes_source, description, status, created_at, updated_at)
                  VALUES(?,?,?,?,?,?,?,?,?,?)""",
               (ds_id, name, lay.root, db.dumps(lay.to_dict()), db.dumps([str(c) for c in classes]), source,
                payload.get("description") or "", "new", now, now))
    job = jobs.submit("scan", lambda j: scan_dataset(ds_id, j), dataset_id=ds_id, message="Queued")
    row = get_dataset_or_404(ds_id)
    return {"dataset": dataset_public(row), "job": job.to_dict()}


@router.get("/{ds_id}")
def get_dataset(ds_id: str, touch: int = Query(0)):
    row = get_dataset_or_404(ds_id)
    if touch:
        db.execute("UPDATE datasets SET opened_at=? WHERE id=?", (db.now(), ds_id))
    return {"dataset": dataset_public(row)}


@router.post("/{ds_id}/relocate")
def relocate_dataset(ds_id: str, payload: dict = Body(...)):
    """Point an existing dataset at a new root and re-index it, keeping its id and review marks.
    This is the repair path for an index whose files moved — a different machine, an unmounted
    drive, or an index built inside Docker (container paths) now opened directly on the host."""
    get_dataset_or_404(ds_id)
    path = (payload.get("path") or "").strip()
    if not path:
        raise HTTPException(400, "path is required")
    try:
        lay = detect(path)
    except FileNotFoundError as e:
        raise HTTPException(404, str(e))
    except Exception as e:  # noqa: BLE001
        raise HTTPException(400, f"Could not detect layout: {e}")
    if lay.kind == "unknown" or not lay.sources:
        raise HTTPException(400, "; ".join(lay.notes) or "No dataset found at this path")

    db.execute("UPDATE datasets SET root=?, layout=?, status='new', updated_at=? WHERE id=?",
               (lay.root, db.dumps(lay.to_dict()), db.now(), ds_id))
    # Stale rows still point at the old location; the rescan re-creates them from the new root.
    with db.transaction() as conn:
        conn.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
        conn.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
    job = jobs.submit("scan", lambda j: scan_dataset(ds_id, j), dataset_id=ds_id, message="Queued")
    return {"dataset": dataset_public(get_dataset_or_404(ds_id)), "job": job.to_dict()}


@router.patch("/{ds_id}")
def update_dataset(ds_id: str, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    sets: List[str] = []
    params: List[Any] = []
    if "name" in payload and str(payload["name"]).strip():
        sets.append("name=?"); params.append(str(payload["name"]).strip())
    if "description" in payload:
        sets.append("description=?"); params.append(str(payload["description"] or ""))
    if "classes" in payload and isinstance(payload["classes"], list):
        names = [str(c).strip() or f"class_{i}" for i, c in enumerate(payload["classes"])]
        sets.append("classes=?"); params.append(db.dumps(names))
        sets.append("classes_source=?"); params.append("user")
    if sets:
        sets.append("updated_at=?"); params.append(db.now())
        params.append(ds_id)
        db.execute(f"UPDATE datasets SET {', '.join(sets)} WHERE id=?", params)
    if payload.get("write_yaml"):
        from ..core.export import write_data_yaml
        row = get_dataset_or_404(ds_id)
        out = write_data_yaml(row["root"], db.loads(row["classes"]), db.loads(row["layout"], {}))
        lay = db.loads(row["layout"], {})
        lay["data_yaml"] = out
        lay["data_yaml_host"] = to_host(out)
        db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), ds_id))
    return {"dataset": dataset_public(get_dataset_or_404(ds_id))}


@router.delete("/{ds_id}")
def delete_dataset(ds_id: str):
    get_dataset_or_404(ds_id)
    job = jobs.active_for(ds_id)
    if job:
        job.cancel()
    with db.transaction() as conn:
        conn.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
        conn.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
        conn.execute("DELETE FROM reviews WHERE dataset_id=?", (ds_id,))
        conn.execute("DELETE FROM plugin_data WHERE dataset_id=?", (ds_id,))
        conn.execute("DELETE FROM datasets WHERE id=?", (ds_id,))
    return {"ok": True}


@router.post("/{ds_id}/scan")
def rescan(ds_id: str, payload: Optional[dict] = Body(None)):
    row = get_dataset_or_404(ds_id)
    active = jobs.active_for(ds_id, "scan")
    if active:
        return {"job": active.to_dict(), "already_running": True}
    if payload and payload.get("redetect"):
        try:
            lay = detect(row["root"])
            if lay.sources:
                db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay.to_dict()), ds_id))
                if lay.classes and row.get("classes_source") in ("inferred", "yaml", "file"):
                    db.execute("UPDATE datasets SET classes=?, classes_source=? WHERE id=?",
                               (db.dumps(lay.classes), "yaml" if lay.data_yaml else "file", ds_id))
        except Exception as e:  # noqa: BLE001
            raise HTTPException(400, f"Re-detect failed: {e}")
    job = jobs.submit("scan", lambda j: scan_dataset(ds_id, j), dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict()}


@router.get("/{ds_id}/stats")
def get_stats(ds_id: str):
    row = get_dataset_or_404(ds_id)
    return {"stats": stats.dataset_stats(ds_id, db.loads(row["classes"])), "issue_labels": ISSUE_LABELS}


# ------------------------------------------------------------------ images

@router.get("/{ds_id}/images")
def list_images(ds_id: str, request: Request, offset: int = Query(0, ge=0), limit: int = Query(80, ge=1, le=1000),
                sort: Optional[str] = Query(None), order: Optional[str] = Query(None),
                boxes: int = Query(0), paths: int = Query(0)):
    get_dataset_or_404(ds_id)
    f = filters_from(request)
    total, items = imagesq.list_images(ds_id, f, offset, limit, sort, order, with_boxes=bool(boxes), with_paths=bool(paths))
    return {"total": total, "offset": offset, "limit": limit, "items": items, "filters": f}


@router.get("/{ds_id}/images/neighbor")
def neighbor(ds_id: str, request: Request, id: int = Query(...), dir: str = Query("next")):
    get_dataset_or_404(ds_id)
    item = imagesq.neighbor(ds_id, filters_from(request), id, dir)
    return {"item": item}


@router.get("/{ds_id}/images/position")
def image_position(ds_id: str, request: Request, id: int = Query(...), sort: Optional[str] = Query(None), order: Optional[str] = Query(None)):
    """0-based position of an image inside the filtered/sorted list (id & name sorts only)."""
    get_dataset_or_404(ds_id)
    f = filters_from(request)
    where, params = imagesq.build_where(ds_id, f)
    cur = db.query_one("SELECT id, rel_path FROM images WHERE id=? AND dataset_id=?", (id, ds_id))
    if not cur:
        raise HTTPException(404, "Image not found")
    desc = (order or "").lower() == "desc"
    if (sort or "id") == "name":
        cmp = "(i.rel_path > ? OR (i.rel_path = ? AND i.id > ?))" if desc else "(i.rel_path < ? OR (i.rel_path = ? AND i.id < ?))"
        extra = [cur["rel_path"], cur["rel_path"], id]
    elif (sort or "id") == "id":
        cmp = "i.id > ?" if desc else "i.id < ?"
        extra = [id]
    else:
        return {"position": None}
    row = db.query_one(f"SELECT COUNT(*) c FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where} AND {cmp}", params + extra)
    return {"position": int(row["c"]) if row else None}


@router.get("/{ds_id}/images/{image_id}")
def get_image(ds_id: str, image_id: int):
    get_dataset_or_404(ds_id)
    row = db.query_one(f"{imagesq.BASE_SELECT} WHERE i.id=? AND i.dataset_id=?", (image_id, ds_id))
    if not row:
        raise HTTPException(404, "Image not found")
    item = imagesq.row_to_item(row, with_paths=True)
    imagesq.attach_boxes([item])
    return {"item": item}


@router.get("/{ds_id}/images/{image_id}/labels")
def get_labels(ds_id: str, image_id: int):
    row = get_dataset_or_404(ds_id)
    img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id))
    if not img:
        raise HTTPException(404, "Image not found")
    classes = db.loads(row["classes"])
    n_classes = len(classes) if row.get("classes_source") != "inferred" and classes else None
    lp = Path(img["label_path"]) if img["label_path"] else None
    version = None
    if lp and lp.is_file():
        boxes, confs, issues, version = read_label_snapshot(lp, n_classes)
        exists = version is not None
    else:
        boxes, confs, issues, exists = [], [], set(), False
    return {"image_id": image_id, "boxes": boxes, "confs": confs, "issues": sorted(issues), "exists": exists,
            "version": version,
            "label_path": str(lp) if lp else None, "label_path_host": to_host(lp) if lp else None}


AUTO_EXTEND_LIMIT = 32   # an inferred class list grows to cover a new id only when the gap is this small


@router.put("/{ds_id}/images/{image_id}/labels")
def put_labels(ds_id: str, image_id: int, payload: dict = Body(...)):
    row = get_dataset_or_404(ds_id)
    img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id))
    if not img:
        raise HTTPException(404, "Image not found")
    if not img["label_path"]:
        raise HTTPException(400, "No label path for this image")
    lp = Path(img["label_path"])
    boxes, rejected = validate_boxes(payload.get("boxes") or [])

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
        write_label_file(lp, boxes)
        b, c, _i, version = read_label_snapshot(lp)
        boxes = [list(x[:5]) + ([cf] if cf is not None else []) for x, cf in zip(b, c)]   # exactly what is on disk
    classes = db.loads(row["classes"])
    n_classes = len(classes) if row.get("classes_source") != "inferred" and classes else None
    max_cls = max((int(b[0]) for b in boxes), default=-1)
    if row.get("classes_source") == "inferred" and len(classes) <= max_cls < len(classes) + AUTO_EXTEND_LIMIT:
        classes = classes + [f"class_{i}" for i in range(len(classes), max_cls + 1)]
        db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(classes), ds_id))
    refresh_image(image_id, n_classes)
    r2 = db.query_one(f"{imagesq.BASE_SELECT} WHERE i.id=?", (image_id,))
    item = imagesq.row_to_item(r2, with_paths=True)
    item["boxes"] = boxes
    return {"item": item, "saved": len(boxes), "rejected": rejected, "version": version}


@router.post("/{ds_id}/labels/bulk")
def bulk_labels(ds_id: str, payload: dict = Body(...)):
    """Bulk label ops: set_class | remap | delete_class | clear over a selection."""
    row = get_dataset_or_404(ds_id)
    op = payload.get("op")
    if op not in ("set_class", "remap", "delete_class", "clear"):
        raise HTTPException(400, "Unknown op")
    targets = _select_target_ids(ds_id, payload)
    if not targets:
        return {"changed": 0, "images": 0}
    classes = db.loads(row["classes"])
    n_classes = len(classes) if row.get("classes_source") != "inferred" and classes else None

    def run(job=None):
        changed = images = 0
        mapping = {int(k2): int(v) for k2, v in (payload.get("mapping") or {}).items()} if op == "remap" else {}
        try:
            for k, iid in enumerate(targets):
                if job and k % 20 == 0:
                    job.update(done=k, total=len(targets), message=f"{op} {k}/{len(targets)}")
                img = db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))
                if not img or not img["label_path"]:
                    continue
                lp = Path(img["label_path"])
                with LABEL_LOCK:
                    n_changed = _bulk_rewrite(lp, op, payload, mapping)
                if n_changed is None:
                    continue
                changed += n_changed
                refresh_image(iid, n_classes, update_counts=False)
                images += 1
        finally:
            refresh_dataset_counts(ds_id)   # once, even if the job is cancelled part-way
        return {"changed": changed, "images": images}

    if len(targets) > 3000:
        job = jobs.submit("bulk_labels", run, dataset_id=ds_id, message=op)
        return {"job": job.to_dict()}
    return run()


def _bulk_rewrite(lp: Path, op: str, payload: dict, mapping: Dict[int, int]) -> Optional[int]:
    """Apply one bulk op to one label file. Returns the number of changed rows, or None if untouched."""
    if not lp.is_file():
        return None           # nothing to change — and "clear" must not invent an empty label file
    rows, _issues = read_label_rows(lp)   # conf stays inside each row, so filtering cannot misassign it
    new = []
    for b in rows:
        if op == "set_class":
            if payload.get("only_cls") is None or int(b[0]) == int(payload["only_cls"]):
                b = [int(payload["cls"])] + list(b[1:])
        elif op == "remap":
            if int(b[0]) in mapping:
                b = [mapping[int(b[0])]] + list(b[1:])
        elif op == "delete_class":
            if int(b[0]) == int(payload["cls"]):
                continue
        new.append(b)
    if op == "clear":
        new = []
    if new == rows:
        return None
    write_label_file(lp, new)
    return sum(1 for a, b2 in zip(rows, new) if a != b2) + abs(len(rows) - len(new))


def _int_ids(values: Any) -> List[int]:
    """Coerce a client-supplied id list, dropping anything that is not an integer. A stray null must
    not surface as a 500 with a raw TypeError."""
    out: List[int] = []
    for v in values or []:
        try:
            out.append(int(v))
        except (TypeError, ValueError):
            continue
    return out


def _select_target_ids(ds_id: str, payload: dict) -> List[int]:
    if payload.get("image_ids"):
        ids = _int_ids(payload["image_ids"])
        if not ids:
            return []
        rows = db.query("SELECT id FROM images WHERE dataset_id=? AND id IN (%s) ORDER BY id" % ",".join("?" * len(ids)), [ds_id] + ids)
        return [r["id"] for r in rows]
    rng = payload.get("range")
    if rng and rng.get("from_id") is not None and rng.get("to_id") is not None:
        a, b = sorted((int(rng["from_id"]), int(rng["to_id"])))
        f = filters_from(payload.get("filters") or {})
        where, params = imagesq.build_where(ds_id, f)
        rows = db.query(f"SELECT i.id FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where} AND i.id BETWEEN ? AND ? ORDER BY i.id", params + [a, b])
        return [r["id"] for r in rows]
    if payload.get("filters") is not None:
        f = filters_from(payload.get("filters") or {})
        where, params = imagesq.build_where(ds_id, f)
        rows = db.query(f"SELECT i.id FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path WHERE {where} ORDER BY i.id", params)
        return [r["id"] for r in rows]
    return []


# ------------------------------------------------------------------ review

@router.put("/{ds_id}/review")
def set_review(ds_id: str, payload: dict = Body(...)):
    get_dataset_or_404(ds_id)
    status = payload.get("status")
    if status not in ("approved", "flagged", "excluded", None):
        raise HTTPException(400, "status must be approved|flagged|excluded|null")
    note = str(payload.get("note") or "")
    ids = payload.get("image_ids") or []
    rels = list(payload.get("rel_paths") or [])
    if payload.get("filters") is not None and not ids and not rels:
        ids = _select_target_ids(ds_id, {"filters": payload["filters"]})
    if ids:
        ids = _int_ids(ids)
        for i in range(0, len(ids), 800):
            chunk = ids[i:i + 800]
            rels.extend(r["rel_path"] for r in db.query("SELECT rel_path FROM images WHERE dataset_id=? AND id IN (%s)" % ",".join("?" * len(chunk)), [ds_id] + chunk))
    if not rels:
        return {"updated": 0}
    now = db.now()
    with db.transaction() as conn:
        if status is None:
            for i in range(0, len(rels), 800):
                chunk = rels[i:i + 800]
                conn.execute("DELETE FROM reviews WHERE dataset_id=? AND rel_path IN (%s)" % ",".join("?" * len(chunk)), [ds_id] + chunk)
        else:
            conn.executemany("""INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)
                                ON CONFLICT(dataset_id, rel_path) DO UPDATE SET status=excluded.status, note=excluded.note, updated_at=excluded.updated_at""",
                             [(ds_id, r, status, note, now) for r in rels])
    return {"updated": len(rels), "status": status}


@router.get("/{ds_id}/sequences")
def sequences(ds_id: str, split: Optional[str] = Query(None), limit: int = Query(400)):
    get_dataset_or_404(ds_id)
    params: List[Any] = [ds_id]
    where = "dataset_id=?"
    if split:
        where += " AND split=?"; params.append(split)
    rows = db.query(f"SELECT seq, split, COUNT(*) n, MIN(id) first_id, SUM(n_boxes) boxes FROM images WHERE {where} GROUP BY seq, split HAVING n > 1 ORDER BY split, seq LIMIT ?", params + [limit])
    return {"sequences": rows}


# ------------------------------------------------------------------ files

def _safe_child(root: Path, rel: str) -> Path:
    cand = (root / rel).resolve() if rel else root.resolve()
    try:
        cand.relative_to(root.resolve())
    except ValueError:
        raise HTTPException(400, "Path escapes dataset root")
    return cand


@router.get("/{ds_id}/files")
def list_files(ds_id: str, path: str = Query("")):
    row = get_dataset_or_404(ds_id)
    root = Path(row["root"])
    d = _safe_child(root, path.strip("/"))
    if not d.is_dir():
        raise HTTPException(404, "Not a directory")
    entries = []
    try:
        for child in sorted(d.iterdir(), key=lambda p: (not p.is_dir(), p.name.lower())):
            if child.name.startswith("."):
                continue
            try:
                st = child.stat()
            except OSError:
                continue
            is_dir = child.is_dir()
            rel = child.relative_to(root).as_posix()
            e = {"name": child.name, "rel": rel, "is_dir": is_dir, "size": None if is_dir else st.st_size, "mtime": st.st_mtime,
                 "kind": "dir" if is_dir else ("image" if child.suffix.lower() in IMAGE_EXTS else ("text" if child.suffix.lower() in (".txt", ".yaml", ".yml", ".md", ".json", ".csv", ".names") else "file"))}
            if not is_dir and e["kind"] == "image":
                img = db.query_one("SELECT id FROM images WHERE dataset_id=? AND abs_path=?", (ds_id, str(child)))
                e["image_id"] = img["id"] if img else None
            entries.append(e)
    except PermissionError:
        raise HTTPException(403, "Permission denied")
    crumbs = []
    acc = ""
    for part in [p for p in path.strip("/").split("/") if p]:
        acc = f"{acc}/{part}" if acc else part
        crumbs.append({"name": part, "rel": acc})
    return {"root_host": to_host(root), "path": path.strip("/"), "crumbs": crumbs, "entries": entries[:5000], "truncated": len(entries) > 5000}


@router.get("/{ds_id}/file")
def get_file(ds_id: str, path: str = Query(...), raw: int = Query(0)):
    row = get_dataset_or_404(ds_id)
    root = Path(row["root"])
    p = _safe_child(root, path.strip("/"))
    if not p.is_file():
        raise HTTPException(404, "File not found")
    if raw or p.suffix.lower() in IMAGE_EXTS:
        return FileResponse(p, media_type=mimetypes.guess_type(p.name)[0] or "application/octet-stream")
    size = p.stat().st_size
    with p.open("rb") as fh:
        data = fh.read(TEXT_PREVIEW_MAX)
    text = data.decode("utf-8", errors="replace")
    return {"path": path, "size": size, "truncated": size > TEXT_PREVIEW_MAX, "text": text}
