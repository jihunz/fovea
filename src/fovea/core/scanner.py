"""Dataset indexing: walks sources, reads image headers + YOLO labels into SQLite."""
from __future__ import annotations

import os
import re
import time
from pathlib import Path
from collections import Counter
from typing import Dict, Iterator, List, Optional, Set, Tuple

from PIL import Image

from .. import db
from ..config import IMAGE_EXTS, LABEL_EXT
from ..jobs import Job, JobCancelled
from ..paths import resolve, to_host
from .labels import read_label_file
from .layout import Layout, Source, label_path_for_image

Image.MAX_IMAGE_PIXELS = None

_SEQ_RE = re.compile(r"^(.*?)[\-_ .]*(?:frame|img|image|fr|f)?[\-_ .]*(\d+)$", re.I)
# An embedded 8-char hex hash, e.g. "..._170a28e1_...". Must contain at least one a-f letter:
# eight plain digits are a zero-padded frame number, not a hash, and treating them as one made
# "cam_00000007_left" and "cam_00000042_left" the same file.
_HASH_RE = re.compile(r"_(?=[0-9]*[a-fA-F])[0-9a-fA-F]{8}_")


def seq_key(rel_path: str) -> str:
    stem = Path(rel_path).stem
    m = _SEQ_RE.match(stem)
    if m and m.group(1):
        return m.group(1)
    return stem


def _norm_stem(stem: str) -> str:
    return _HASH_RE.sub("_", stem)


class FuzzyLabelIndex:
    """Fallback for label files whose stem differs from the image's only by an embedded hash.

    This path decides which file an edit is written to, so it is deliberately conservative — a wrong
    match silently overwrites another image's ground truth. A label file is only offered when:
      * the image stem itself carries a hash segment (the premise of the fallback),
      * the file is not the exact label of any image in this scan (it already has an owner),
      * exactly one such file normalises to the key, and
      * exactly one label-less image in that folder is asking for that key.
    Anything ambiguous is left unmatched and surfaces as `missing_label`, which is recoverable.
    """

    def __init__(self, owned: Set[str], demand: Counter):
        self._owned = owned
        self._demand = demand
        self._maps: Dict[str, Dict[str, List[Path]]] = {}

    @staticmethod
    def wants(stem: str) -> bool:
        return bool(_HASH_RE.search(stem))

    def find(self, label_dir: Path, stem: str) -> Optional[Path]:
        if not self.wants(stem):
            return None
        d = str(label_dir)
        key = _norm_stem(stem)
        if self._demand.get((d, key), 0) != 1:
            return None
        m = self._maps.get(d)
        if m is None:
            m = {}
            try:
                with os.scandir(label_dir) as it:
                    for e in it:
                        if not (e.is_file() and e.name.lower().endswith(LABEL_EXT)):
                            continue
                        if e.path in self._owned:
                            continue
                        try:
                            e.path.encode("utf-8")
                        except UnicodeEncodeError:
                            continue
                        m.setdefault(_norm_stem(Path(e.name).stem), []).append(Path(e.path))
            except OSError:
                pass
            self._maps[d] = m
        found = m.get(key) or []
        return found[0] if len(found) == 1 else None


def _utf8_ok(*values: str) -> bool:
    """Linux filenames are arbitrary bytes; Python decodes the undecodable ones to lone surrogates,
    which SQLite (and every UTF-8 writer) rejects. One such name used to abort the whole scan."""
    try:
        for v in values:
            v.encode("utf-8")
        return True
    except UnicodeEncodeError:
        return False


def enumerate_source(src: Source, errors: Optional[List[str]] = None,
                     skipped: Optional[Counter] = None) -> Iterator[Tuple[str, str, str, str]]:
    """Yield (rel_key, split, abs_image, abs_label_target).

    `errors` collects listing failures. A failure must never look like "this folder is empty" —
    the caller would then prune every indexed row for it.
    """
    errors = errors if errors is not None else []
    skipped = skipped if skipped is not None else Counter()
    split = src.split or ""
    if src.list_file:
        lf = Path(src.list_file)
        base = Path(src.base) if src.base else lf.parent
        try:
            lines = lf.read_text(encoding="utf-8", errors="ignore").splitlines()
        except OSError as e:
            errors.append(f"{to_host(lf)}: {e.strerror or e}")
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
            lbl = str(label_path_for_image(cand))
            if not _utf8_ok(key, str(cand), lbl, split):
                skipped["undecodable_name"] += 1
                continue
            yield key, split, str(cand), lbl
        return

    img_dir = Path(src.img_dir)
    label_dir = Path(src.label_dir) if src.label_dir else label_path_for_image(img_dir / "x.jpg").parent

    def _onerror(e: OSError) -> None:
        errors.append(f"{to_host(getattr(e, 'filename', '') or img_dir)}: {e.strerror or e}")

    for root, dirs, files in os.walk(img_dir, onerror=_onerror):
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
            lbl = str((label_dir / rel).with_suffix(LABEL_EXT))
            if not _utf8_ok(key, str(p), lbl, split):
                skipped["undecodable_name"] += 1
                continue
            yield key, split, str(p), lbl


def read_dims(path: str) -> Optional[Tuple[int, int]]:
    """Display size: EXIF orientations 5-8 swap width and height, as every browser and YOLO trainer does."""
    try:
        with Image.open(path) as im:
            w, h = int(im.width), int(im.height)
            try:
                if int(im.getexif().get(0x0112) or 1) in (5, 6, 7, 8):
                    w, h = h, w
            except Exception:  # noqa: BLE001 — malformed EXIF: keep the raster size
                pass
            return w, h
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


class DatasetUnreachable(FileNotFoundError):
    """The dataset's files could not be listed. Raised BEFORE the index is touched."""


def refresh_dataset_counts(dataset_id: str) -> dict:
    counts = db.query_one("SELECT COUNT(*) c, SUM(has_label) l, SUM(n_boxes) b FROM images WHERE dataset_id=?", (dataset_id,))
    c, l, b = (counts["c"] or 0), (counts["l"] or 0), (counts["b"] or 0)
    db.execute("UPDATE datasets SET image_count=?, label_count=?, box_count=?, updated_at=? WHERE id=?",
               (c, l, b, db.now(), dataset_id))
    return {"images": c, "labeled": l, "boxes": b}


def _settle_status(dataset_id: str) -> None:
    """Put a dataset back into a truthful resting state after an interrupted scan."""
    refresh_dataset_counts(dataset_id)
    db.execute("UPDATE datasets SET status = CASE WHEN image_count > 0 THEN 'ready' ELSE 'new' END, updated_at=? WHERE id=?",
               (db.now(), dataset_id))


def scan_dataset(dataset_id: str, job: Job) -> dict:
    ds = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not ds:
        raise ValueError("dataset not found")
    db.execute("UPDATE datasets SET status='scanning', updated_at=? WHERE id=?", (db.now(), dataset_id))
    conn = db.connect()
    try:
        return _scan(ds, job, conn)
    except JobCancelled:
        # A user-requested cancel is not an error, and must not leave the dataset parked at 'scanning'.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        _settle_status(dataset_id)
        raise
    except DatasetUnreachable:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        db.execute("UPDATE datasets SET status='unreachable', updated_at=? WHERE id=?", (db.now(), dataset_id))
        raise
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        db.execute("UPDATE datasets SET status='error', updated_at=? WHERE id=?", (db.now(), dataset_id))
        raise


def _scan(ds: dict, job: Job, conn) -> dict:
    dataset_id = ds["id"]
    layout = Layout.from_dict(db.loads(ds["layout"], {}))
    classes = db.loads(ds["classes"])
    validate_classes = bool(classes) and ds.get("classes_source", "inferred") != "inferred"
    n_classes = len(classes) if validate_classes else None

    # ---- listing. A source that is missing OR could not be fully listed (permissions, a mount that came
    # back empty-handed) is "unreachable": its existing rows are kept, never pruned. Treating a failed
    # listing as "this folder is empty now" is how an entire index used to vanish on a rescan.
    reachable: List[Source] = []
    unreachable: List[Source] = []
    problems: List[str] = []
    skipped: Counter = Counter()
    entries: List[Tuple[str, str, str, str]] = []
    for src in layout.sources:
        target = src.list_file or src.img_dir
        if not target or not Path(target).exists():
            unreachable.append(src)
            problems.append(f"{to_host(target) if target else '(no path)'}: not found")
            continue
        errs: List[str] = []
        label = src.split or "images"
        got: List[Tuple[str, str, str, str]] = []
        for k, entry in enumerate(enumerate_source(src, errs, skipped), 1):
            got.append(entry)
            if k % 2000 == 0:
                job.update(message=f"Listing {label}… {len(entries) + k:,} found")   # also a cancel point
        job.update(message=f"Listing {label}… {len(entries) + len(got):,} found")
        if errs:
            unreachable.append(src)
            problems.extend(errs[:3])
        else:
            reachable.append(src)
            entries.extend(got)

    if layout.sources and not reachable:
        shown = "; ".join(problems[:3]) + (" …" if len(problems) > 3 else "")
        raise DatasetUnreachable(f"Dataset files are not reachable — nothing was changed in the index. {shown}")

    total = len(entries)
    if total == 0 and db.query_one("SELECT 1 FROM images WHERE dataset_id=? LIMIT 1", (dataset_id,)):
        raise DatasetUnreachable(
            "The dataset folders were listed but no images were found, so the existing index was kept. "
            "If the images really were deleted, remove the dataset or move it to a new path."
        )

    if unreachable:
        job.update(message=f"{len(unreachable)} source(s) unreachable — keeping their existing rows")
    job.update(done=0, total=total, message="Indexing")

    existing: Dict[str, dict] = {}
    for r in db.query("SELECT id, rel_path, mtime, size, width, height FROM images WHERE dataset_id=?", (dataset_id,)):
        existing[r["rel_path"]] = r

    # ---- fuzzy fallback bookkeeping (see FuzzyLabelIndex): which label files already belong to an image,
    # and how many label-less, hash-named images compete for each normalised key.
    owned: Set[str] = {e[3] for e in entries}
    demand: Counter = Counter()
    for _key, _split, _img, abs_lbl in entries:
        lp = Path(abs_lbl)
        if FuzzyLabelIndex.wants(lp.stem) and not os.path.isfile(abs_lbl):
            demand[(str(lp.parent), _norm_stem(lp.stem))] += 1
    fuzzy = FuzzyLabelIndex(owned, demand)

    token = job.id
    max_cls = -1
    t0 = time.time()
    conn.execute("BEGIN")
    for idx, (key, split, abs_img, abs_lbl) in enumerate(entries):
        try:
            st = os.stat(abs_img)
        except OSError:
            continue
        issues: set = set()
        prev = existing.get(key)
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
        elif lbl_path.parent.is_dir():
            alt = fuzzy.find(lbl_path.parent, lbl_path.stem)
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
        else:
            issues.add("missing_label")

        cls_ids = sorted({int(b[0]) for b in boxes})
        if cls_ids:
            max_cls = max(max_cls, cls_ids[-1])
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
            job.update(done=idx + 1, message=f"Indexing {idx + 1:,}/{total:,}")   # cancel point, outside a txn
            conn.execute("BEGIN")

    # ---- prune rows this pass did not re-see, but only within splits every one of whose sources was read.
    prune = "dataset_id=? AND scan_token != ?"
    prune_args: List[object] = [dataset_id, token]
    prunable = True
    if unreachable:
        gone_splits = {s.split or "" for s in unreachable}
        keep = sorted({s.split or "" for s in reachable} - gone_splits)
        if keep:
            prune += " AND split IN (%s)" % ",".join("?" * len(keep))
            prune_args.extend(keep)
        else:
            prunable = False
    if prunable:
        conn.execute(f"DELETE FROM boxes WHERE dataset_id=? AND image_id IN (SELECT id FROM images WHERE {prune})",
                     [dataset_id] + prune_args)
        conn.execute(f"DELETE FROM images WHERE {prune}", prune_args)
    conn.execute("COMMIT")

    # The index is committed; from here on nothing is cancellable, so do not go through job.update().
    job.done, job.message = total, "Finalising"
    if not classes or (ds.get("classes_source") == "inferred" and max_cls + 1 > len(classes)):
        classes = [f"class_{i}" for i in range(max_cls + 1)] if max_cls >= 0 else []
        db.execute("UPDATE datasets SET classes=?, classes_source='inferred' WHERE id=?", (db.dumps(classes), dataset_id))
    cover = db.query_one("SELECT id FROM images WHERE dataset_id=? AND n_boxes>0 ORDER BY id LIMIT 1", (dataset_id,)) \
        or db.query_one("SELECT id FROM images WHERE dataset_id=? ORDER BY id LIMIT 1", (dataset_id,))
    counts = refresh_dataset_counts(dataset_id)
    db.execute("UPDATE datasets SET status='ready', cover_image_id=?, scanned_at=?, updated_at=? WHERE id=?",
               (cover["id"] if cover else None, db.now(), db.now(), dataset_id))
    result = {**counts, "elapsed": round(time.time() - t0, 2)}
    if skipped:
        result["skipped"] = dict(skipped)
    if unreachable:
        result["unreachable_sources"] = len(unreachable)
        result["problems"] = problems[:5]
    notes = []
    if skipped.get("undecodable_name"):
        notes.append(f"{skipped['undecodable_name']:,} file(s) skipped: name is not valid UTF-8")
    if unreachable:
        notes.append(f"{len(unreachable)} source(s) could not be read; their rows were kept")
    job.message = "; ".join(notes) if notes else "Done"
    return result


def refresh_image(image_id: int, n_classes: Optional[int] = None, update_counts: bool = True) -> Optional[dict]:
    """Re-read one image's label from disk and update the index (after a save).

    Bulk callers pass update_counts=False and call refresh_dataset_counts() once afterwards: the
    roll-up is a full aggregate over the dataset, and repeating it per image made a 50k-image bulk op
    run one table scan per image under the write lock."""
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
    width, height = row["width"], row["height"]
    if width is None:
        # Indexed while unreadable (e.g. still being copied in). Try again rather than keep the image
        # flagged forever with an unknown size until the next full rescan.
        dims = read_dims(row["abs_path"])
        if dims:
            width, height = dims
        else:
            issues.add("image_unreadable")
    cls_ids = sorted({int(b[0]) for b in boxes})
    with db.transaction() as conn:
        conn.execute("UPDATE images SET has_label=?, n_boxes=?, classes=?, issues=?, lmtime=?, width=?, height=? WHERE id=?",
                     (has_label, len(boxes), db.dumps(cls_ids), db.dumps(sorted(issues)), lmtime, width, height, image_id))
        conn.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
        if boxes:
            conn.executemany("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)",
                             [(image_id, row["dataset_id"], int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes])
    if update_counts:
        refresh_dataset_counts(row["dataset_id"])
    return db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
