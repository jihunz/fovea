"""Dataset indexing: walks sources, reads image headers + YOLO labels into SQLite."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from PIL import Image

from .. import db
from ..config import IMAGE_EXTS, LABEL_EXT
from ..jobs import Job
from ..paths import resolve
from .labels import read_label_file
from .layout import Layout, Source, is_image, label_path_for_image

Image.MAX_IMAGE_PIXELS = None

_SEQ_RE = re.compile(r"^(.*?)[\-_ .]*(?:frame|img|image|fr|f)?[\-_ .]*(\d+)$", re.I)
_HASH_RE = re.compile(r"_[0-9a-fA-F]{8}_")


def seq_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    m = _SEQ_RE.match(stem)
    if m and m.group(1):
        return m.group(1)
    return stem


def _norm_stem(stem: str) -> str:
    return _HASH_RE.sub("_", stem)


class FuzzyLabelIndex:
    """Fallback matcher for label files whose stem differs by an embedded hash."""

    def __init__(self):
        self._maps: Dict[str, Dict[str, Path]] = {}

    def find(self, label_dir: Path, stem: str) -> Optional[Path]:
        key = str(label_dir)
        m = self._maps.get(key)
        if m is None:
            m = {}
            try:
                with os.scandir(label_dir) as it:
                    for e in it:
                        if e.is_file() and e.name.lower().endswith(LABEL_EXT):
                            m[_norm_stem(Path(e.name).stem)] = Path(e.path)
            except OSError:
                pass
            self._maps[key] = m
        return m.get(_norm_stem(stem))


def enumerate_source(src: Source) -> Iterator[Tuple[str, str, str, str]]:
    """Yield (rel_key, split, abs_image, abs_label_target)."""
    split = src.split or ""
    if src.list_file:
        lf = Path(src.list_file)
        base = Path(src.base) if src.base else lf.parent
        try:
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError:
            return
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            p = Path(s)
            cand = resolve(s) if p.is_absolute() else (base / p)
            if not cand.exists():
                alt = lf.parent / p
                cand = alt if alt.exists() else cand
            if not cand.is_file() or cand.suffix.lower() not in IMAGE_EXTS:
                continue
            cand = cand.resolve()
            try:
                rel = cand.relative_to(base.resolve()).as_posix()
            except ValueError:
                rel = "/".join(cand.parts[-2:])
            key = f"{split}/{rel}" if split else rel
            yield key, split, str(cand), str(label_path_for_image(cand))
        return

    img_dir = Path(src.img_dir)
    label_dir = Path(src.label_dir) if src.label_dir else label_path_for_image(img_dir / "x.jpg").parent
    for root, dirs, files in os.walk(img_dir):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        rp = Path(root)
        for f in sorted(files):
            if f.startswith("."):
                continue
            dot = f.rfind(".")
            if dot <= 0 or f[dot:].lower() not in IMAGE_EXTS:
                continue
            p = rp / f
            rel = p.relative_to(img_dir).as_posix()
            key = f"{split}/{rel}" if split else rel
            lbl = label_dir / rel
            lbl = lbl.with_suffix(LABEL_EXT)
            yield key, split, str(p), str(lbl)


def read_dims(path: str) -> Optional[Tuple[int, int]]:
    try:
        with Image.open(path) as im:
            return int(im.width), int(im.height)
    except Exception:
        return None


UPSERT_SQL = """
INSERT INTO images(dataset_id, rel_path, split, abs_path, label_path, width, height, size, mtime, lmtime,
                   has_label, n_boxes, classes, issues, seq, scan_token)
VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(dataset_id, rel_path) DO UPDATE SET
  split=excluded.split, abs_path=excluded.abs_path, label_path=excluded.label_path,
  width=excluded.width, height=excluded.height, size=excluded.size, mtime=excluded.mtime, lmtime=excluded.lmtime,
  has_label=excluded.has_label, n_boxes=excluded.n_boxes, classes=excluded.classes, issues=excluded.issues,
  seq=excluded.seq, scan_token=excluded.scan_token
RETURNING id
"""


def scan_dataset(dataset_id: str, job: Job) -> dict:
    ds = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not ds:
        raise ValueError("dataset not found")
    layout = Layout.from_dict(db.loads(ds["layout"], {}))
    classes = db.loads(ds["classes"])
    validate_classes = bool(classes) and ds.get("classes_source", "inferred") != "inferred"
    n_classes = len(classes) if validate_classes else None

    db.execute("UPDATE datasets SET status='scanning', updated_at=? WHERE id=?", (db.now(), dataset_id))

    entries: List[Tuple[str, str, str, str]] = []
    for src in layout.sources:
        job.update(message=f"Listing {src.split or 'images'}…")
        entries.extend(enumerate_source(src))
    total = len(entries)
    job.update(done=0, total=total, message="Indexing")

    existing: Dict[str, dict] = {}
    for r in db.query("SELECT id, rel_path, mtime, size, width, height FROM images WHERE dataset_id=?", (dataset_id,)):
        existing[r["rel_path"]] = r

    fuzzy = FuzzyLabelIndex()
    conn = db.connect()
    token = job.id
    n_labeled = 0
    n_boxes_total = 0
    max_cls = -1
    t0 = time.time()
    conn.execute("BEGIN")
    try:
        for idx, (key, split, abs_img, abs_lbl) in enumerate(entries):
            try:
                st = os.stat(abs_img)
            except OSError:
                continue
            issues: set = set()
            prev = existing.get(key)
            dims = None
            if prev and prev["mtime"] == st.st_mtime and prev["size"] == st.st_size and prev["width"]:
                dims = (prev["width"], prev["height"])
            else:
                dims = read_dims(abs_img)
                if dims is None:
                    issues.add("image_unreadable")

            lbl_path = Path(abs_lbl)
            has_label = 0
            boxes: list = []
            lmtime = None
            if lbl_path.is_file():
                has_label = 1
            else:
                alt = fuzzy.find(lbl_path.parent, lbl_path.stem) if lbl_path.parent.is_dir() else None
                if alt is not None:
                    lbl_path = alt
                    has_label = 1
            if has_label:
                try:
                    lmtime = lbl_path.stat().st_mtime
                except OSError:
                    lmtime = None
                boxes, _confs, lbl_issues = read_label_file(lbl_path, n_classes)
                issues |= lbl_issues
                n_labeled += 1
            else:
                issues.add("missing_label")

            cls_ids = sorted({int(b[0]) for b in boxes})
            if cls_ids:
                max_cls = max(max_cls, cls_ids[-1])
            n_boxes_total += len(boxes)
            cur = conn.execute(UPSERT_SQL, (
                dataset_id, key, split, abs_img, str(lbl_path),
                dims[0] if dims else None, dims[1] if dims else None, st.st_size, st.st_mtime, lmtime,
                has_label, len(boxes), db.dumps(cls_ids), db.dumps(sorted(issues)), seq_key(key), token,
            ))
            image_id = cur.fetchone()[0]
            conn.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
            if boxes:
                conn.executemany("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)",
                                 [(image_id, dataset_id, int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes])
            if (idx + 1) % 400 == 0:
                conn.execute("COMMIT")
                job.update(done=idx + 1, message=f"Indexing {idx + 1:,}/{total:,}")
                conn.execute("BEGIN")
        conn.execute("DELETE FROM boxes WHERE dataset_id=? AND image_id IN (SELECT id FROM images WHERE dataset_id=? AND scan_token != ?)",
                     (dataset_id, dataset_id, token))
        conn.execute("DELETE FROM images WHERE dataset_id=? AND scan_token != ?", (dataset_id, token))
        conn.execute("COMMIT")
    except Exception:
        try:
            conn.execute("ROLLBACK")
        except Exception:
            pass
        db.execute("UPDATE datasets SET status='error', updated_at=? WHERE id=?", (db.now(), dataset_id))
        raise

    job.update(done=total, message="Finalising")
    if not classes or (ds.get("classes_source") == "inferred" and max_cls + 1 > len(classes)):
        classes = [f"class_{i}" for i in range(max_cls + 1)] if max_cls >= 0 else []
        db.execute("UPDATE datasets SET classes=?, classes_source='inferred' WHERE id=?", (db.dumps(classes), dataset_id))
    cover = db.query_one("SELECT id FROM images WHERE dataset_id=? AND n_boxes>0 ORDER BY id LIMIT 1", (dataset_id,)) \
        or db.query_one("SELECT id FROM images WHERE dataset_id=? ORDER BY id LIMIT 1", (dataset_id,))
    counts = db.query_one("SELECT COUNT(*) c, SUM(has_label) l, SUM(n_boxes) b FROM images WHERE dataset_id=?", (dataset_id,))
    db.execute("""UPDATE datasets SET status='ready', image_count=?, label_count=?, box_count=?, cover_image_id=?,
                  scanned_at=?, updated_at=? WHERE id=?""",
               (counts["c"] or 0, counts["l"] or 0, counts["b"] or 0, cover["id"] if cover else None,
                db.now(), db.now(), dataset_id))
    return {"images": counts["c"] or 0, "labeled": counts["l"] or 0, "boxes": counts["b"] or 0,
            "elapsed": round(time.time() - t0, 2)}


def refresh_image(image_id: int, n_classes: Optional[int] = None) -> Optional[dict]:
    """Re-read one image's label from disk and update the index (after a save)."""
    row = db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
    if not row:
        return None
    lbl = Path(row["label_path"]) if row["label_path"] else None
    issues: set = set()
    boxes: list = []
    has_label = 0
    lmtime = None
    if lbl and lbl.is_file():
        has_label = 1
        lmtime = lbl.stat().st_mtime
        boxes, _c, issues = read_label_file(lbl, n_classes)
    else:
        issues.add("missing_label")
    if row["width"] is None:
        issues.add("image_unreadable")
    cls_ids = sorted({int(b[0]) for b in boxes})
    with db.transaction() as conn:
        conn.execute("UPDATE images SET has_label=?, n_boxes=?, classes=?, issues=?, lmtime=? WHERE id=?",
                     (has_label, len(boxes), db.dumps(cls_ids), db.dumps(sorted(issues)), lmtime, image_id))
        conn.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
        if boxes:
            conn.executemany("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)",
                             [(image_id, row["dataset_id"], int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes])
        counts = conn.execute("SELECT COUNT(*) c, SUM(has_label) l, SUM(n_boxes) b FROM images WHERE dataset_id=?",
                              (row["dataset_id"],)).fetchone()
        conn.execute("UPDATE datasets SET image_count=?, label_count=?, box_count=?, updated_at=? WHERE id=?",
                     (counts[0] or 0, counts[1] or 0, counts[2] or 0, db.now(), row["dataset_id"]))
    return db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
