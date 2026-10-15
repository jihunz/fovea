"""Annotations converted from another format, in the index.

* ``convert_pending`` runs at the start of a scan: every source whose annotations are in another format and whose label
  folder Fovea has not finished writing is converted (core.annformats.convert) — once; later scans read the YOLO
  labels like any others. All sources of a dataset share one class list (every split must use the same ids).
* ``sync`` re-reads each converted folder's sidecar into ``box_attrs`` (a box's attributes, kept while the box has the
  geometry it was converted with) and ``image_regions`` (areas that are not objects). It runs after every scan
  (POST_SCAN_HOOKS) and after a label save (scanner.refresh_image).
* ``schema`` / ``regions_in`` / ``drop_in_regions`` serve the UI, the filters and the AI proposal pipeline.
"""
from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .. import db
from . import annformats as af
from .layout import Layout, Source

log = logging.getLogger(__name__)
_lock = threading.Lock()


def _layout(dataset_id: str) -> Tuple[Optional[dict], Optional[Layout]]:
    row = db.query_one("SELECT id, layout, classes, classes_source FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        return None, None
    return row, Layout.from_dict(db.loads(row["layout"], {}))


def converted_dirs(lay: Layout) -> List[Path]:
    return [Path(s.label_dir) for s in lay.sources if s.ann_format and s.label_dir and not s.list_file]


# ------------------------------------------------------------------------------------------------- converting

def _pending(lay: Layout) -> List[Source]:
    return [s for s in lay.sources if s.ann_format and s.ann_path and s.img_dir and s.label_dir
            and not af.is_converted(Path(s.label_dir))]


def convert_pending(dataset_id: str, job=None) -> bool:
    """Convert every source still waiting for it; update the layout, the class list (unless the user named the
    classes) and the class cards (official definitions, never over a card someone wrote). True when anything ran."""
    row, lay = _layout(dataset_id)
    if lay is None:
        return False
    todo = _pending(lay)
    if not todo:
        return False
    by_fmt: Dict[str, List[Source]] = {}
    for s in todo:
        by_fmt.setdefault(s.ann_format, []).append(s)
    done_classes: List[str] = []
    for fid, srcs in by_fmt.items():
        card = (lay.formats.get(fid) or {}).get("card") or af.catalog_card(fid)
        if not card:
            log.warning("format %s of %s has no card; not converted", fid, dataset_id)
            continue
        # one id space for every split: collect class names over all of this format's sources first
        classes: Optional[List[str]] = None
        names: List[str] = []
        for s in srcs:
            r = af.Reader(card, Path(s.ann_path), Path(s.img_dir), s.recursive)
            if not r.from_data:
                classes = r.classes
                break
            rels = [rel for rel, _p in af.list_images(Path(s.img_dir), s.recursive) if r.has_annotation(rel)]
            r.collect_classes(rels)
            names += [c for c in r.classes if c not in names]
        if classes is None:
            order = [c for c in card.get("classes") or [] if c in names]
            classes = order + [c for c in names if c not in order]
        total = {"images": 0, "labels": 0, "boxes": 0, "regions": 0, "missing": 0, "bad": 0, "unknown": 0}
        for k, s in enumerate(srcs):
            msg = f"Converting {card.get('name') or fid} annotations"
            if len(srcs) > 1:
                msg += f" ({s.split or Path(s.img_dir).name}, {k + 1}/{len(srcs)})"

            def progress(i: int, n: int, _m: str = msg) -> None:
                if job is not None:
                    job.update(done=i, total=n, message=f"{_m} {i:,}/{n:,}")
            res = af.convert(card, Path(s.img_dir), Path(s.ann_path), Path(s.label_dir), s.recursive,
                             progress=progress, classes=classes)
            classes = res["classes"]
            s.label_dir_exists = True
            s.label_count = res["labels"]
            for key in total:
                total[key] += int(res.get(key) or 0)
        f = lay.formats.setdefault(fid, {"card": card})
        f.update({"state": "converted", "counts": total, "classes": classes})
        f.pop("check", None)
        done_classes = done_classes or list(classes or [])
        lay.notes = [n for n in lay.notes if not n.startswith("Annotations in ")]
        lay.notes.append(f"Converted {card.get('name') or fid} annotations to YOLO labels: {total['labels']:,} "
                         f"images, {total['boxes']:,} boxes" + (f", {total['regions']:,} regions"
                                                                if total["regions"] else "")
                         + " — the original files are unchanged")
    db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay.to_dict()), dataset_id))
    if done_classes and (row["classes_source"] in ("inferred", "format", "") or not db.loads(row["classes"])):
        db.execute("UPDATE datasets SET classes=?, classes_source='format' WHERE id=?",
                   (db.dumps(done_classes), dataset_id))
    seed_class_cards(dataset_id)
    return True


def seed_class_cards(dataset_id: str) -> int:
    """Official definitions of the dataset's classes (from its format card) into empty class cards."""
    row, lay = _layout(dataset_id)
    if lay is None:
        return 0
    defs: Dict[str, dict] = {}
    for f in lay.formats.values():
        defs.update(((f.get("card") or {}).get("class_defs")) or {})
    if not defs:
        return 0
    names = db.loads(row["classes"]) or []
    have = {r["cls"]: r for r in db.query("SELECT * FROM class_cards WHERE dataset_id=?", (dataset_id,))}
    n = 0
    for i, name in enumerate(names):
        d = defs.get(name)
        if not d:
            continue
        cur = have.get(i)
        if cur and any((cur.get(k) or "").strip() for k in ("description", "include", "exclude", "prompt")):
            continue                                            # someone wrote this card: leave it
        db.execute("""INSERT INTO class_cards(dataset_id, cls, description, include, exclude, prompt, examples, updated_at)
                      VALUES(?,?,?,?,?,?,COALESCE((SELECT examples FROM class_cards WHERE dataset_id=? AND cls=?),'[]'),?)
                      ON CONFLICT(dataset_id, cls) DO UPDATE SET description=excluded.description,
                        include=excluded.include, exclude=excluded.exclude, updated_at=excluded.updated_at""",
                   (dataset_id, i, d.get("description") or "", d.get("include") or "", d.get("exclude") or "", "",
                    dataset_id, i, db.now()))
        n += 1
    return n


# ------------------------------------------------------------------------------------------------- attributes / regions

_dirs_cache: Dict[str, Tuple[str, List[str]]] = {}


def _dirs_for(dataset_id: str) -> List[str]:
    row = db.query_one("SELECT layout FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        return []
    raw = row["layout"] or ""
    hit = _dirs_cache.get(dataset_id)
    if hit and hit[0] == raw:
        return hit[1]
    dirs = [str(d) for d in converted_dirs(Layout.from_dict(db.loads(raw, {})))]
    _dirs_cache[dataset_id] = (raw, dirs)
    return dirs


def _entry_for(label_path: str, sides: Dict[str, Dict[str, dict]]) -> Optional[dict]:
    if not label_path:
        return None
    for d, side in sides.items():
        base = d.rstrip("/") + "/"
        if label_path.startswith(base):
            rel = label_path[len(base):]
            dot = rel.rfind(".")
            return side.get(rel[:dot] if dot > 0 else rel)
    return None


def sync(dataset_id: str, image_ids: Optional[Sequence[int]] = None) -> int:
    """Rebuild box_attrs and image_regions for the dataset (or some of its images) from the conversion sidecars.
    Returns the number of boxes that carry attributes."""
    dirs = _dirs_for(dataset_id)
    with _lock:
        if not dirs:
            if image_ids is None:
                with db.transaction() as tx:
                    tx.execute("DELETE FROM box_attrs WHERE dataset_id=?", (dataset_id,))
                    tx.execute("DELETE FROM image_regions WHERE dataset_id=?", (dataset_id,))
            return 0
        sides = {d: af.load_sidecar(Path(d)) for d in dirs}
        if image_ids is None:
            imgs = db.query("SELECT id, label_path FROM images WHERE dataset_id=?", (dataset_id,))
        else:
            ids = [int(i) for i in image_ids]
            imgs = []
            for k in range(0, len(ids), 500):
                chunk = ids[k:k + 500]
                imgs += db.query(f"SELECT id, label_path FROM images WHERE id IN ({','.join('?' * len(chunk))})",
                                 chunk)
        entries = {r["id"]: _entry_for(r["label_path"] or "", sides) for r in imgs}
        want = [i for i, e in entries.items() if e]
        attr_rows: List[tuple] = []
        region_rows: List[tuple] = []
        for k in range(0, len(want), 500):
            chunk = want[k:k + 500]
            boxes: Dict[int, List[dict]] = {}
            for b in db.query(f"SELECT id, image_id, cls, xc, yc, w, h FROM boxes WHERE image_id IN "
                              f"({','.join('?' * len(chunk))}) ORDER BY image_id, id", chunk):
                boxes.setdefault(b["image_id"], []).append(b)
            for iid in chunk:
                e = entries[iid]
                bl = boxes.get(iid, [])
                for b, a in zip(bl, af.match_attrs(e, [[b["cls"], b["xc"], b["yc"], b["w"], b["h"]] for b in bl])):
                    if a:
                        attr_rows.append((b["id"], iid, dataset_id, json.dumps(a, separators=(",", ":"))))
                for kind, xc, yc, w, h in e.get("regions") or []:
                    region_rows.append((iid, dataset_id, str(kind), float(xc), float(yc), float(w), float(h)))
        with db.transaction() as tx:
            if image_ids is None:
                tx.execute("DELETE FROM box_attrs WHERE dataset_id=?", (dataset_id,))
                tx.execute("DELETE FROM image_regions WHERE dataset_id=?", (dataset_id,))
            else:
                for k in range(0, len(imgs), 500):
                    chunk = [r["id"] for r in imgs[k:k + 500]]
                    q = ",".join("?" * len(chunk))
                    tx.execute(f"DELETE FROM box_attrs WHERE image_id IN ({q})", chunk)
                    tx.execute(f"DELETE FROM image_regions WHERE image_id IN ({q})", chunk)
            if attr_rows:
                tx.executemany("INSERT OR REPLACE INTO box_attrs(box_id, image_id, dataset_id, attrs) VALUES(?,?,?,?)",
                               attr_rows)
            if region_rows:
                tx.executemany("INSERT INTO image_regions(image_id, dataset_id, kind, xc, yc, w, h) "
                               "VALUES(?,?,?,?,?,?,?)", region_rows)
        return len(attr_rows)


def on_scan(dataset_id: str) -> None:
    """POST_SCAN_HOOKS entry."""
    sync(dataset_id)


def sync_image(dataset_id: str, image_id: int) -> None:
    """After a label save: only datasets with converted annotations do any work here."""
    if _dirs_for(dataset_id):
        sync(dataset_id, [image_id])


# ------------------------------------------------------------------------------------------------- reading

def schema(dataset_id: str) -> dict:
    """{attributes: [{name, label, type, values: {code: meaning}}], regions: {kind: {label, description}}, formats:
    [{id, name, source, urls, …}]} of the dataset's converted / pending formats."""
    _row, lay = _layout(dataset_id)
    out = {"attributes": [], "regions": {}, "formats": []}
    if lay is None:
        return out
    seen = set()
    for fid, f in lay.formats.items():
        card = f.get("card") or {}
        out["formats"].append({"id": fid, "name": card.get("name") or fid, "source": card.get("source") or "",
                               "state": f.get("state"), "urls": card.get("urls") or [],
                               "license": card.get("license") or "", "citation": card.get("citation") or "",
                               "notes": card.get("notes") or "", "provider": card.get("provider") or "",
                               "model": card.get("model") or "", "counts": f.get("counts") or {},
                               "why": f.get("why") or ""})
        for a in af.attribute_schema(card):
            if a["name"] not in seen:
                seen.add(a["name"])
                out["attributes"].append(a)
        out["regions"].update(card.get("regions") or {})
    return out


def regions_in(image_id: int) -> List[dict]:
    return [{"kind": r["kind"], "box": [r["xc"], r["yc"], r["w"], r["h"]]}
            for r in db.query("SELECT kind, xc, yc, w, h FROM image_regions WHERE image_id=? ORDER BY id", (image_id,))]


def attrs_for_rows(dataset_id: str, label_path: str, boxes: List[List[float]]) -> List[Optional[dict]]:
    """Attributes of each label row (cls, xc, yc, w, h) as Annotate shows them — matched live from the sidecar, so
    a row moved since the last save reads as a new annotation at once."""
    dirs = _dirs_for(dataset_id)
    if not dirs:
        return [None] * len(boxes)
    e = _entry_for(label_path or "", {d: af.load_sidecar(Path(d)) for d in dirs})
    return af.match_attrs(e, boxes)


def regions_by_image(image_ids: Iterable[int]) -> Dict[int, List[Tuple[float, float, float, float]]]:
    ids = [int(i) for i in image_ids]
    out: Dict[int, List[Tuple[float, float, float, float]]] = {}
    for k in range(0, len(ids), 500):
        chunk = ids[k:k + 500]
        for r in db.query(f"SELECT image_id, xc, yc, w, h FROM image_regions WHERE image_id IN "
                          f"({','.join('?' * len(chunk))})", chunk):
            out.setdefault(r["image_id"], []).append((r["xc"], r["yc"], r["w"], r["h"]))
    return out


def _inside(xc: float, yc: float, r: Sequence[float]) -> bool:
    return abs(xc - r[0]) <= r[2] / 2 and abs(yc - r[1]) <= r[3] / 2


def drop_in_regions(image_id: int, preds: List[dict]) -> Tuple[List[dict], int]:
    """AI proposals / findings whose centre lies inside one of the image's regions that are not objects (VisDrone's
    ignored regions, KITTI DontCare, COCO crowds) are dropped: those areas are not labelled object by object, so a box
    there could never be accepted or audited against anything."""
    regs = [(r["xc"], r["yc"], r["w"], r["h"]) for r in
            db.query("SELECT xc, yc, w, h FROM image_regions WHERE image_id=?", (image_id,))]
    if not regs or not preds:
        return preds, 0
    kept = [p for p in preds if not any(_inside(float(p.get("xc", 0)), float(p.get("yc", 0)), r) for r in regs)]
    return kept, len(preds) - len(kept)


def register_hook() -> None:
    from . import scanner
    if on_scan not in scanner.POST_SCAN_HOOKS:
        scanner.POST_SCAN_HOOKS.insert(0, on_scan)
