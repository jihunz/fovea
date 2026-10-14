"""Dataset indexing: walks sources, reads image headers + YOLO labels into SQLite."""
from __future__ import annotations

import os
import re
import sqlite3
import stat
import time
import traceback
import unicodedata
import warnings
from collections import Counter
from pathlib import Path, PurePosixPath
from typing import Callable, Dict, Iterator, List, Optional, Set, Tuple

from PIL import Image

from .. import db
from ..config import IMAGE_EXTS, LABEL_EXT, MAX_IMAGE_PIXELS
from ..jobs import Job, JobCancelled
from ..paths import to_host
from .labels import MAX_INFERRED_CLASSES, dataset_bases, inferred_names, read_label_file
from .layout import (UNMATCHED_NOTE, Layout, Source, entry_path, find_unmatched_labels, iter_files,
                     label_path_for_image)

# Called with the dataset id after every successful scan (a relocation is a scan too). The QA engine
# appends a hook that queues its incremental checks. A failing hook is logged and never fails the scan.
POST_SCAN_HOOKS: List[Callable[[str], None]] = []

_SEQ_RE = re.compile(r"^(.*?)[\-_ .]*(?:frame|img|image|fr|f)?[\-_ .]*(\d+)$", re.I)
# An embedded 8-char hex hash, e.g. "..._170a28e1_...". Must contain at least one a-f letter:
# eight plain digits are a zero-padded frame number, not a hash, and treating them as one made
# "cam_00000007_left" and "cam_00000042_left" the same file.
_HASH_RE = re.compile(r"_(?=[0-9]*[a-fA-F])[0-9a-fA-F]{8}_")
_BATCH_ROWS = 400        # index rows per write transaction ...
_BATCH_SECONDS = 0.2     # ... or less: a label save must never wait long behind a rescan


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def seq_key(rel_path: str) -> str:
    """Sequence (video / burst) key of an image from its path inside its source folder: the folder plus the
    file name without its frame number. 'MOT17-02/img1/000001.jpg' -> 'MOT17-02/img1',
    'camA/clip_001.jpg' -> 'camA/clip', 'images_GOPR0804_00006.jpg' -> 'images_GOPR0804'. The split is not
    part of it, so a sequence spread over train and test keeps one key (split leakage stays visible)."""
    p = PurePosixPath(rel_path)
    parent = p.parent.as_posix()
    parent = "" if parent in (".", "") else parent
    stem = p.stem
    m = _SEQ_RE.match(stem)
    prefix = m.group(1) if m else stem
    if prefix and parent:
        return f"{parent}/{prefix}"
    # 'frame_0001' (only a frame word before the number) -> 'frame'; '000001' in a folder -> the folder
    return prefix or parent or re.sub(r"[\-_ .]*\d+$", "", stem) or stem


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


def _list_rel(cand: Path, base: Path, taken: Set[str], split: str) -> str:
    """Index key part for a list entry: its path below the list's base folder; for an entry outside it the
    shortest tail of its path (>= 2 parts) not already used in this split — two different files named
    images/0001.jpg must not share a key."""
    try:
        return cand.relative_to(base).as_posix()
    except ValueError:
        pass
    parts = [p for p in cand.parts if p not in ("/", "")]
    for n in range(2, len(parts) + 1):
        rel = "/".join(parts[-n:])
        if (f"{split}/{rel}" if split else rel) not in taken:
            return rel
    return "/".join(parts)


def _printable(s: str) -> str:
    """A file name that may hold undecodable bytes (lone surrogates), made safe to show and to store."""
    return s.encode("utf-8", "surrogateescape").decode("utf-8", "replace")


def enumerate_source(src: Source, errors: Optional[List[str]] = None,
                     skipped: Optional[Counter] = None, peers=(), disc: str = "",
                     notes: Optional[List[str]] = None,
                     examples: Optional[Dict[str, List[str]]] = None) -> Iterator[Tuple[str, str, str, str, str]]:
    """Yield (rel_key, split, abs_image, abs_label_target, seq).

    rel_key / split / seq are NFC-normalised (macOS hands out decomposed Hangul); the absolute paths are
    exactly what the file system returned. `disc` is inserted after the split when a split has several
    source folders, so cam_a/0001.jpg and cam_b/0001.jpg stay two images. `errors` collects listing
    failures. A failure must never look like "this folder is empty" — the caller would then prune every
    indexed row for it. `notes` collects the first few skipped entries for the scan report; `examples`
    (kind -> up to 5 texts) the first few of each kind."""
    errors = errors if errors is not None else []
    skipped = skipped if skipped is not None else Counter()
    notes = notes if notes is not None else []

    def _skip(kind: str, text: str) -> None:
        skipped[kind] += 1
        if examples is not None:
            lst = examples.setdefault(kind, [])
            if len(lst) < 5:
                lst.append(_printable(text)[:300])

    split = nfc(src.split or "")
    prefix = "/".join(x for x in (split, nfc(disc)) if x)

    def _key(rel: str) -> str:
        rel = nfc(rel)
        return f"{prefix}/{rel}" if prefix else rel

    if src.list_file:
        lf = Path(src.list_file)
        base = Path(os.path.normpath(src.base)) if src.base else lf.parent
        try:
            lines = lf.read_text(encoding="utf-8-sig", errors="ignore").splitlines()
        except OSError as e:
            errors.append(f"{to_host(lf)}: {e.strerror or e}")
            return
        taken: Set[str] = set()
        for raw in lines:
            s = raw.strip()
            if not s or s.startswith("#"):
                continue
            cand = entry_path(s, base)
            if not cand.exists():
                tries = [entry_path(s, lf.parent)]
                rel_s = s.replace("\\", "/")
                if rel_s.startswith("data/"):            # CVAT "YOLO 1.1": entries are data/obj_train_data/x.jpg
                    tries += [entry_path(rel_s[5:], base), entry_path(rel_s[5:], lf.parent)]
                cand = next((t for t in tries if t.exists()), cand)
            if not cand.exists():
                kind = "broken_link" if os.path.islink(cand) else "missing_list_entry"
                _skip(kind, s)
                if len(notes) < 5:
                    notes.append(f"{kind.replace('_', ' ')}: {s}")
                continue
            if not cand.is_file() or cand.suffix.lower() not in IMAGE_EXTS:
                _skip("not_an_image", s)
                continue
            # Keep the path as listed (symlinks unresolved): the label lives beside the LINK, as Ultralytics
            # reads it, and an edit must land inside the dataset, not next to the link target.
            cand = Path(os.path.normpath(cand))
            rel = _list_rel(cand, base, taken, split)
            key = _key(rel)
            taken.add(key)
            lbl = str(label_path_for_image(cand))
            if not _utf8_ok(key, str(cand), lbl, split):
                _skip("undecodable_name", str(cand))
                continue
            yield key, split, str(cand), lbl, nfc(seq_key(rel))
        return

    img_dir = Path(src.img_dir)
    label_dir = Path(src.label_dir) if src.label_dir else label_path_for_image(img_dir / "x.jpg").parent

    def _onerror(e: OSError) -> None:
        errors.append(f"{to_host(getattr(e, 'filename', '') or img_dir)}: {e.strerror or e}")

    for root, files in iter_files(img_dir, getattr(src, "recursive", True), onerror=_onerror,
                                  peers=[p for p in peers if p != src.img_dir]):
        rp = Path(root)
        for f in files:
            dot = f.rfind(".")
            if dot <= 0 or f[dot:].lower() not in IMAGE_EXTS:
                continue
            p = rp / f
            rel = p.relative_to(img_dir).as_posix()
            key = _key(rel)
            lbl = str((label_dir / rel).with_suffix(LABEL_EXT))
            if not _utf8_ok(key, str(p), lbl, split):
                _skip("undecodable_name", to_host(str(p)))
                continue
            yield key, split, str(p), lbl, nfc(seq_key(rel))


def _header_size(path: str) -> Optional[Tuple[int, int]]:
    """Image size from the header only, bypassing Pillow's decompression-bomb guard (which is about
    decoding pixels; this never decodes)."""
    Image.init()
    with open(path, "rb") as fp:
        prefix = fp.read(16)
        for fmt in Image.ID:
            factory, accept = Image.OPEN[fmt]
            try:
                if accept and not accept(prefix):
                    continue
                fp.seek(0)
                im = factory(fp, path)
                return int(im.size[0]), int(im.size[1])
            except Exception:  # noqa: BLE001 — not this format
                continue
    return None


def read_dims(path: str) -> Optional[Tuple[int, int]]:
    """Display size: EXIF orientations 5-8 swap width and height, as every browser and YOLO trainer does."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                w, h = int(im.width), int(im.height)
                try:
                    # (qa-engine hook) a PNG's getexif() decodes the whole raster (trailing eXIf chunk): not
                    # for one beyond the decode cap
                    if im.format == "PNG" and "exif" not in im.info and w * h > MAX_IMAGE_PIXELS:
                        pass
                    elif int(im.getexif().get(0x0112) or 1) in (5, 6, 7, 8):
                        w, h = h, w
                except Exception:  # noqa: BLE001 — malformed EXIF: keep the raster size
                    pass
                return w, h
    except Image.DecompressionBombError:
        try:
            return _header_size(path)
        except Exception:  # noqa: BLE001
            return None
    except Exception:
        return None


UPSERT_SQL = """
INSERT INTO images(dataset_id, rel_path, split, abs_path, label_path, width, height, size, mtime, lmtime, lsize,
                   has_label, n_boxes, classes, issues, seq, scan_token)
VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
ON CONFLICT(dataset_id, rel_path) DO UPDATE SET
  split=excluded.split, abs_path=excluded.abs_path, label_path=excluded.label_path,
  width=excluded.width, height=excluded.height, size=excluded.size, mtime=excluded.mtime, lmtime=excluded.lmtime,
  lsize=excluded.lsize, has_label=excluded.has_label, n_boxes=excluded.n_boxes, classes=excluded.classes,
  issues=excluded.issues, seq=excluded.seq, scan_token=excluded.scan_token
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


def _infer_classes(dataset_id: str) -> None:
    """Grow an inferred class list (class_0..N) to cover the highest class id in the index, up to
    MAX_INFERRED_CLASSES: a larger id stays outside the list and is flagged bad_class (revalidate_classes).
    Re-reads the current list and only writes over the exact value it read: names a user typed while the
    first scan was running (source 'user') are never overwritten."""
    row = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        return
    classes = db.loads(row["classes"])
    if classes and row["classes_source"] != "inferred":
        return
    mx = (db.query_one("SELECT MAX(cls) m FROM boxes WHERE dataset_id=? AND cls < ?",
                       (dataset_id, MAX_INFERRED_CLASSES)) or {}).get("m")
    mx = -1 if mx is None else int(mx)
    if mx + 1 <= len(classes):
        return
    new = inferred_names(classes, mx)
    db.execute("UPDATE datasets SET classes=?, classes_source='inferred' WHERE id=? AND classes=? "
               "AND (classes_source='inferred' OR classes='[]')", (db.dumps(new), dataset_id, row["classes"]))


def _pick_cover(dataset_id: str) -> Optional[int]:
    cover = db.query_one("SELECT id FROM images WHERE dataset_id=? AND n_boxes>0 ORDER BY id LIMIT 1", (dataset_id,)) \
        or db.query_one("SELECT id FROM images WHERE dataset_id=? ORDER BY id LIMIT 1", (dataset_id,))
    return cover["id"] if cover else None


def revalidate_classes(dataset_id: str) -> dict:
    """Recompute the `bad_class` issue from the boxes table after the class list changed (or after a scan
    that skipped unchanged labels). Without this, shrinking the class list left no issue and growing it
    left stale ones until a full re-parse."""
    row = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        return {"added": 0, "removed": 0}
    classes = db.loads(row["classes"])
    # An inferred list covers every id below MAX_INFERRED_CLASSES (_infer_classes grows it); larger ids never get a name.
    n = len(classes) if row["classes_source"] != "inferred" and classes else MAX_INFERRED_CLASSES
    bad = {r["image_id"] for r in db.query("SELECT DISTINCT image_id FROM boxes WHERE dataset_id=? AND (cls < 0 OR cls >= ?)",
                                             (dataset_id, n))}
    flagged = {r["id"]: r["issues"] for r in db.query(
        "SELECT id, issues FROM images WHERE dataset_id=? AND issues LIKE '%\"bad_class\"%'", (dataset_id,))}
    updates = []
    for iid in bad - set(flagged):
        cur = db.query_one("SELECT issues FROM images WHERE id=?", (iid,))
        if cur is not None:
            updates.append((db.dumps(sorted(set(db.loads(cur["issues"])) | {"bad_class"})), iid))
    removed = 0
    for iid, issues in flagged.items():
        if iid not in bad:
            updates.append((db.dumps(sorted(set(db.loads(issues)) - {"bad_class"})), iid))
            removed += 1
    if updates:
        with db.transaction() as conn:
            conn.executemany("UPDATE images SET issues=? WHERE id=?", updates)
    return {"added": len(updates) - removed, "removed": removed}


def _settle_status(dataset_id: str, first_scan: bool) -> None:
    """Put a dataset back into a truthful resting state after an interrupted scan. An interrupted FIRST scan
    is 'partial' (a rescan resumes it); class names and a cover are derived from what was indexed so far."""
    counts = refresh_dataset_counts(dataset_id)
    _infer_classes(dataset_id)
    status = "new" if not counts["images"] else ("partial" if first_scan else "ready")
    db.execute("UPDATE datasets SET status=?, cover_image_id=COALESCE(cover_image_id, ?), updated_at=? WHERE id=? "
               "AND status!='deleting'",
               (status, _pick_cover(dataset_id), db.now(), dataset_id))


def run_post_scan_hooks(dataset_id: str) -> None:
    for hook in list(POST_SCAN_HOOKS):
        try:
            hook(dataset_id)
        except Exception:  # noqa: BLE001
            traceback.print_exc()


def scan_dataset(dataset_id: str, job: Job) -> dict:
    ds = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,))
    if not ds:
        raise ValueError("dataset not found")
    first_scan = ds.get("scanned_at") is None
    # 'deleting' (DELETE /datasets) is final: a scan that was still stopping must never bring the dataset back.
    db.execute("UPDATE datasets SET status='scanning', updated_at=? WHERE id=? AND status!='deleting'", (db.now(), dataset_id))
    conn = db.connect()
    try:
        result = _scan(ds, job, conn)
    except JobCancelled:
        # A user-requested cancel is not an error, and must not leave the dataset parked at 'scanning'.
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        _settle_status(dataset_id, first_scan)
        raise
    except DatasetUnreachable:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        db.execute("UPDATE datasets SET status='unreachable', updated_at=? WHERE id=? AND status!='deleting'",
                   (db.now(), dataset_id))
        raise
    except Exception:
        if conn.in_transaction:
            conn.execute("ROLLBACK")
        db.execute("UPDATE datasets SET status='error', updated_at=? WHERE id=? AND status!='deleting'",
                   (db.now(), dataset_id))
        raise
    run_post_scan_hooks(dataset_id)
    return result


def _discriminators(sources: List[Source]) -> Dict[int, str]:
    """For a split fed by several folders, each folder's path below their common parent (cam_a, cam_b)."""
    groups: Dict[str, List[Source]] = {}
    for s in sources:
        if s.img_dir and not s.list_file:
            groups.setdefault(s.split or "", []).append(s)
    out: Dict[int, str] = {}
    for group in groups.values():
        if len(group) < 2:
            continue
        paths = [os.path.normpath(s.img_dir) for s in group]
        try:
            common = os.path.commonpath(paths)
        except ValueError:
            common = ""
        for s, p in zip(group, paths):
            rel = os.path.relpath(p, common) if common else p.strip("/")
            out[id(s)] = Path(rel).as_posix() if rel not in (".", "") else Path(p).name
    return out


def _disambiguate(entries: List[Tuple[str, str, str, str, str]], entry_src: List[int],
                  sources: List[Source]) -> List[Tuple[str, str, str, str, str]]:
    """A split fed by several folders whose file names clash (cam_a/0001.jpg, cam_b/0001.jpg) gets each
    folder's name inserted into its keys. Only splits that actually clash are re-keyed, so the keys (and the
    review marks attached to them) of every other dataset stay exactly as they were."""
    owners: Dict[str, set] = {}
    for e, si in zip(entries, entry_src):
        owners.setdefault(e[0], set()).add(si)
    clashing = {e[1] for e in entries if len(owners.get(e[0], ())) > 1}
    if not clashing:
        return entries
    discs = _discriminators(sources)
    out = []
    for e, si in zip(entries, entry_src):
        d = discs.get(id(sources[si]))
        if d and e[1] in clashing:
            key, split = e[0], e[1]
            rest = key[len(split) + 1:] if split and key.startswith(split + "/") else key
            e = ("/".join(x for x in (split, nfc(d), rest) if x),) + tuple(e[1:])
        out.append(e)
    return out


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
    examples: Dict[str, List[str]] = {}       # problem kind -> first few examples (the scan report)
    list_errors = sources_missing = 0

    def example(kind: str, text: str) -> None:
        lst = examples.setdefault(kind, [])
        if len(lst) < 5:
            lst.append(_printable(text)[:300])

    entries: List[Tuple[str, str, str, str, str]] = []
    entry_src: List[int] = []
    peer_dirs = [s.img_dir for s in layout.sources if s.img_dir]   # links into these would index images twice
    for si, src in enumerate(layout.sources):
        target = src.list_file or src.img_dir
        if not target or not Path(target).exists():
            unreachable.append(src)
            problems.append(f"{to_host(target) if target else '(no path)'}: not found")
            example("unreachable", to_host(target) if target else "(no path)")
            sources_missing += 1
            continue
        errs: List[str] = []
        label = src.split or "images"
        got: List[Tuple[str, str, str, str, str]] = []
        for k, entry in enumerate(enumerate_source(src, errs, skipped, peers=peer_dirs, examples=examples), 1):
            got.append(entry)
            if k % 2000 == 0:
                job.update(message=f"Listing {label}… {len(entries) + k:,} found")   # also a cancel point
        job.update(message=f"Listing {label}… {len(entries) + len(got):,} found")
        if errs:
            unreachable.append(src)
            problems.extend(errs[:3])
            list_errors += len(errs)
            for e in errs[:5]:
                example("list_error", e)
        else:
            reachable.append(src)
            entries.extend(got)
            entry_src.extend([si] * len(got))

    if layout.sources and not reachable:
        shown = "; ".join(problems[:3]) + (" …" if len(problems) > 3 else "")
        raise DatasetUnreachable(f"Dataset files are not reachable — nothing was changed in the index. {shown}")

    entries = _disambiguate(entries, entry_src, layout.sources)

    # ---- one key per image: two different files with the same key (two list entries outside the base, a
    # clash the folder prefix could not resolve) used to overwrite each other's row, boxes and review.
    unique: List[Tuple[str, str, str, str, str]] = []
    first_abs: Dict[str, str] = {}
    for e in entries:
        prev = first_abs.get(e[0])
        if prev is None:
            first_abs[e[0]] = e[2]
            unique.append(e)
        elif os.path.realpath(prev) == os.path.realpath(e[2]):
            skipped["listed_twice"] += 1
            example("listed_twice", to_host(e[2]))
        else:
            skipped["name_collision"] += 1
            example("name_collision", f"{to_host(e[2])} (same key as {to_host(prev)})")
    entries = unique

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
    for r in db.query("SELECT id, rel_path, split, abs_path, label_path, mtime, size, width, height, lmtime, lsize, seq, has_label "
                      "FROM images WHERE dataset_id=?", (dataset_id,)):
        existing[r["rel_path"]] = r

    # ---- fuzzy fallback bookkeeping (see FuzzyLabelIndex): which label files already belong to an image,
    # and how many label-less, hash-named images compete for each normalised key.
    owned: Set[str] = {e[3] for e in entries}
    demand: Counter = Counter()
    for _key, _split, _img, abs_lbl, _seq in entries:
        lp = Path(abs_lbl)
        if FuzzyLabelIndex.wants(lp.stem) and not os.path.isfile(abs_lbl):
            demand[(str(lp.parent), _norm_stem(lp.stem))] += 1
    fuzzy = FuzzyLabelIndex(owned, demand)

    link_bases = dataset_bases(ds["root"], db.loads(ds["layout"], {}))
    token = job.id
    t0 = time.time()
    stats = Counter()
    touch: List[int] = []
    upserts: List[tuple] = []
    batch_t = time.time()

    def flush() -> None:
        """Write one batch in a short transaction. All file I/O for the batch already happened, so a label
        save waiting on the write lock waits for SQL only, never for disk reads."""
        nonlocal touch, upserts, batch_t
        if touch or upserts:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if touch:
                    conn.executemany("UPDATE images SET scan_token=? WHERE id=?", [(token, i) for i in touch])
                for row, boxes in upserts:
                    image_id = conn.execute(UPSERT_SQL, row).fetchone()[0]
                    conn.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
                    if boxes:
                        conn.executemany("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)",
                                         [(image_id, dataset_id, int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes])
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        touch, upserts, batch_t = [], [], time.time()

    for idx, (key, split, abs_img, abs_lbl, seq) in enumerate(entries):
        try:
            st = os.stat(abs_img)
        except OSError as e:
            skipped["stat_error"] += 1
            example("stat_error", f"{to_host(abs_img)} ({e.strerror or e})")
            continue
        prev = existing.get(key)
        lbl_path = Path(abs_lbl)
        try:
            lst = os.stat(abs_lbl)
            has_label = 1 if stat.S_ISREG(lst.st_mode) else 0
        except OSError:
            lst, has_label = None, 0
        if not has_label and lbl_path.parent.is_dir():
            alt = fuzzy.find(lbl_path.parent, lbl_path.stem)
            if alt is not None:
                lbl_path, has_label = alt, 1
                try:
                    lst = os.stat(alt)
                except OSError:
                    lst, has_label = None, 0
        lmtime = lst.st_mtime if (has_label and lst) else None
        lsize = lst.st_size if (has_label and lst) else None

        # ---- unchanged image and label (same stat signature): keep its row and boxes, just mark it seen.
        if (prev and prev["width"] and prev["mtime"] == st.st_mtime and prev["size"] == st.st_size
                and prev["label_path"] == str(lbl_path) and prev["lmtime"] == lmtime and prev["lsize"] == lsize
                and bool(prev["has_label"]) == bool(has_label) and prev["split"] == split
                and prev["abs_path"] == abs_img and prev["seq"] == seq):
            touch.append(prev["id"])
            stats["unchanged"] += 1
        else:
            issues: set = set()
            if prev and prev["mtime"] == st.st_mtime and prev["size"] == st.st_size and prev["width"]:
                dims = (prev["width"], prev["height"])
            else:
                dims = read_dims(abs_img)
                if dims is None:
                    if _foreign_link(abs_img, link_bases):
                        # A file link out of the dataset that is not even an image (a downloaded dataset may ship
                        # images/x.jpg -> ~/.ssh/id_rsa): indexing it would serve and export that file's bytes.
                        skipped["link_outside"] += 1
                        example("link_outside", to_host(abs_img))
                        continue
                    issues.add("image_unreadable")
            boxes: list = []
            if has_label:
                boxes, _confs, lbl_issues = read_label_file(lbl_path, n_classes)
                issues |= lbl_issues
            else:
                issues.add("missing_label")
            cls_ids = sorted({int(b[0]) for b in boxes})
            upserts.append(((
                dataset_id, key, split, abs_img, str(lbl_path),
                dims[0] if dims else None, dims[1] if dims else None, st.st_size, st.st_mtime, lmtime, lsize,
                has_label, len(boxes), db.dumps(cls_ids), db.dumps(sorted(issues)), seq, token,
            ), boxes))
            stats["indexed"] += 1
        if len(touch) + len(upserts) >= _BATCH_ROWS or time.time() - batch_t > _BATCH_SECONDS:
            flush()
            job.update(done=idx + 1, message=f"Indexing {idx + 1:,}/{total:,}")   # cancel point, outside a txn
    flush()

    # ---- prune rows this pass did not re-see, but only within splits every one of whose sources was read.
    prune = "dataset_id=? AND scan_token != ?"
    prune_args: List[object] = [dataset_id, token]
    prunable = True
    if unreachable:
        gone_splits = {nfc(s.split or "") for s in unreachable}
        keep = sorted({nfc(s.split or "") for s in reachable} - gone_splits)
        if keep:
            prune += " AND split IN (%s)" % ",".join("?" * len(keep))
            prune_args.extend(keep)
        else:
            prunable = False
    pruned = 0
    if prunable:
        carry_over(dataset_id, prune, prune_args, token, existing)
        with db.transaction() as tx:
            tx.execute(f"DELETE FROM boxes WHERE dataset_id=? AND image_id IN (SELECT id FROM images WHERE {prune})",
                       [dataset_id] + prune_args)
            pruned = tx.execute(f"DELETE FROM images WHERE {prune}", prune_args).rowcount or 0

    # The index is committed; from here on nothing is cancellable, so do not go through job.update().
    job.done, job.message = total, "Finalising"
    moved = remap_reviews(dataset_id) if pruned else {}
    _infer_classes(dataset_id)
    revalidate_classes(dataset_id)
    orphan_labels = _refresh_layout(dataset_id, layout)
    counts = refresh_dataset_counts(dataset_id)
    db.execute("UPDATE datasets SET status='ready', cover_image_id=?, scanned_at=?, updated_at=? WHERE id=? "
               "AND status!='deleting'",
               (_pick_cover(dataset_id), db.now(), db.now(), dataset_id))
    result = {**counts, "elapsed": round(time.time() - t0, 2), "indexed": stats["indexed"],
              "unchanged": stats["unchanged"], "removed": pruned}
    if moved.get("reviews_moved"):
        result["reviews_moved"] = moved["reviews_moved"]
    if skipped:
        result["skipped"] = dict(skipped)
    if unreachable:
        result["unreachable_sources"] = len(unreachable)
    counts_by_kind = dict(skipped)
    counts_by_kind["unreachable"] = sources_missing
    counts_by_kind["list_error"] = list_errors
    if orphan_labels:
        counts_by_kind["labels_without_images"] = sum(int(u.get("files") or 0) for u in orphan_labels)
        for u in orphan_labels[:5]:
            example("labels_without_images", f"{u['path_host']} ({int(u.get('files') or 0):,} files)")
    result["problems"] = scan_problems(counts_by_kind, examples)
    result["notes"] = [p["note"] for p in result["problems"]]
    job.message = "; ".join(n for n, p in zip(result["notes"], result["problems"]) if p["severity"] == "warn") or "Done"
    db.kv_set(f"scan:{dataset_id}", {"at": db.now(), **result})
    try:
        db.refresh_planner_stats(("images", "boxes", "reviews"))
    except Exception:  # noqa: BLE001 — statistics are an optimisation; the scan itself succeeded
        traceback.print_exc()
    return result


# Problems a scan reports (dataset.last_scan.problems), most serious first. `info` ones are not defects.
SCAN_PROBLEMS = {
    "unreachable": ("warn", "Source folders or lists not found", "{n:,} source(s) not found; their indexed images were kept"),
    "list_error": ("warn", "Folders that could not be listed", "{n:,} listing error(s); those sources' images were kept"),
    "stat_error": ("warn", "Images that could not be read", "{n:,} image(s) could not be read and were skipped"),
    "missing_list_entry": ("warn", "List entries not found", "{n:,} list entr{y} not found"),
    "broken_link": ("warn", "Broken links", "{n:,} broken link(s) skipped"),
    "name_collision": ("warn", "Images with the same name as another image in the split",
                       "{n:,} image(s) skipped: same name as another image in the split"),
    "undecodable_name": ("warn", "File names that are not valid UTF-8", "{n:,} file(s) skipped: name is not valid UTF-8"),
    "not_an_image": ("info", "List entries that are not images", "{n:,} list entr{y} skipped: not an image"),
    "labels_without_images": ("info", "Label files with no matching images folder",
                              "{n:,} label file(s) in folders without images — not indexed"),
    "listed_twice": ("info", "Images listed twice", "{n:,} duplicate list entr{y} ignored"),
    "link_outside": ("warn", "Links to files outside the dataset",
                     "{n:,} linked file(s) skipped: they point outside the dataset and are not images"),
}


def scan_problems(counts: Dict[str, int], examples: Dict[str, List[str]]) -> List[dict]:
    """[{code, label, severity, count, note, examples (<= 5)}] for every kind with a count, warnings first."""
    out = []
    for code, (severity, label, note) in SCAN_PROBLEMS.items():
        n = int(counts.get(code) or 0)
        if n <= 0:
            continue
        out.append({"code": code, "label": label, "severity": severity, "count": n,
                    "note": note.format(n=n, y="y" if n == 1 else "ies"), "examples": list(examples.get(code, []))[:5]})
    for code, n in counts.items():                 # a kind added later without a description still shows up
        if code not in SCAN_PROBLEMS and n:
            out.append({"code": code, "label": code.replace("_", " ").capitalize(), "severity": "warn", "count": int(n),
                        "note": f"{int(n):,} × {code.replace('_', ' ')}", "examples": list(examples.get(code, []))[:5]})
    out.sort(key=lambda p: p["severity"] != "warn")
    return out


def _suffix_len(a: str, b: str) -> int:
    pa, pb = PurePosixPath(a).parts, PurePosixPath(b).parts
    n = 0
    while n < min(len(pa), len(pb)) and pa[-1 - n] == pb[-1 - n]:
        n += 1
    return n


def _prefix_len(a: str, b: str) -> int:
    pa, pb = PurePosixPath(a).parts, PurePosixPath(b).parts
    n = 0
    while n < min(len(pa), len(pb)) - 1 and pa[n] == pb[n]:
        n += 1
    return n


def match_moves(orphans: List[str], candidates: List[str]) -> Dict[str, str]:
    """old index key -> new index key for files that only moved. A key moves only to an unambiguous match — the
    candidate with the same file name sharing the longest trailing run of path components (then the longest
    leading run: the split, when a folder level was added in between), unique on both sides; anything else is
    left unmatched."""
    from collections import defaultdict
    by_name = defaultdict(list)
    for rel in candidates:
        by_name[PurePosixPath(rel).name].append(rel)
    best = {}
    for old in orphans:
        cands = by_name.get(PurePosixPath(nfc(old)).name, [])
        if not cands:
            continue
        scored = sorted((((_suffix_len(nfc(old), c), _prefix_len(nfc(old), c)), c) for c in cands), reverse=True)
        if len(scored) > 1 and scored[0][0] == scored[1][0]:
            continue                                   # ambiguous: two equally good new homes
        best[old] = scored[0][1]
    claimed = defaultdict(list)
    for old, new in best.items():
        claimed[new].append(old)
    return {olds[0]: new for new, olds in claimed.items() if len(olds) == 1}


def remap_reviews(dataset_id: str) -> dict:
    """Re-attach review marks after a re-index changed image paths (e.g. a relocation that detects a
    different layout, turning train/0001.jpg into 0001.jpg). Marks are keyed by rel_path, so without this
    they silently orphan. A mark moves only to an unambiguous match (match_moves) among the unreviewed paths."""
    current = {r["rel_path"] for r in db.query("SELECT rel_path FROM images WHERE dataset_id=?", (dataset_id,))}
    marks = [r["rel_path"] for r in db.query("SELECT rel_path FROM reviews WHERE dataset_id=?", (dataset_id,))]
    orphans = [m for m in marks if m not in current]
    if not orphans:
        return {"reviews_moved": 0, "reviews_unmatched": 0}
    taken = set(marks)
    moves = match_moves(orphans, [rel for rel in current if rel not in taken])
    with db.transaction() as conn:
        conn.executemany("UPDATE reviews SET rel_path=? WHERE dataset_id=? AND rel_path=?",
                         [(new, dataset_id, old) for old, new in moves.items()])
    return {"reviews_moved": len(moves), "reviews_unmatched": len(orphans) - len(moves)}


def carry_over(dataset_id: str, prune_where: str, prune_args: List[object], token: str,
               existing: Dict[str, dict]) -> int:
    """Before rows that a scan did not see again are pruned: when such a file only moved (a relocation that
    detects a different layout, a renamed folder) and now has a new row, move everything that refers to the old
    row's id — AI proposals and audit findings, QA hashes and flags, snapshot files, working-set entries and
    class-card examples — onto the new id, matched like review marks. Deleting the old row would otherwise
    cascade them away. Returns the number of rows carried over."""
    gone = {r["rel_path"]: r["id"] for r in db.query(f"SELECT id, rel_path FROM images WHERE {prune_where}", prune_args)}
    if not gone:
        return 0
    fresh = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=? AND scan_token=?",
                                                      (dataset_id, token)) if r["rel_path"] not in existing}
    if not fresh:
        return 0
    pairs = [(fresh[new], gone[old]) for old, new in match_moves(list(gone), list(fresh)).items()]
    if not pairs:
        return 0
    remap = {old: new for new, old in pairs}
    with db.transaction() as conn:
        conn.executemany("UPDATE ai_preds SET image_id=? WHERE image_id=?", pairs)          # indexed by image_id
        for table in ("ai_audit", "image_hash", "image_flags"):
            conn.executemany(f"UPDATE OR IGNORE {table} SET image_id=? WHERE image_id=?", pairs)
        # snapshot_files and work_set_items have no image_id index: one pass over this dataset's rows each.
        conn.execute("CREATE TEMP TABLE IF NOT EXISTS _fovea_moved(old INTEGER PRIMARY KEY, new INTEGER NOT NULL)")
        conn.execute("DELETE FROM _fovea_moved")
        conn.executemany("INSERT INTO _fovea_moved(old, new) VALUES(?,?)", [(o, n) for n, o in pairs])
        for table, owner in (("snapshot_files", "snapshot_id IN (SELECT id FROM snapshots WHERE dataset_id=?)"),
                             ("work_set_items", "set_id IN (SELECT id FROM work_sets WHERE dataset_id=?)")):
            conn.execute(f"UPDATE {table} SET image_id=(SELECT m.new FROM _fovea_moved m WHERE m.old={table}.image_id) "
                         f"WHERE {owner} AND image_id IN (SELECT old FROM _fovea_moved)", (dataset_id,))
        conn.execute("DELETE FROM _fovea_moved")
        for card in conn.execute("SELECT cls, examples FROM class_cards WHERE dataset_id=?", (dataset_id,)).fetchall():
            examples = db.loads(card[1], [])
            if any(isinstance(e, dict) and e.get("image_id") in remap for e in examples):
                for e in examples:
                    if isinstance(e, dict) and e.get("image_id") in remap:
                        e["image_id"] = remap[e["image_id"]]
                conn.execute("UPDATE class_cards SET examples=? WHERE dataset_id=? AND cls=?",
                             (db.dumps(examples), dataset_id, card[0]))
    return len(pairs)


def _foreign_link(path: str, bases: List[str]) -> bool:
    """A file symlink whose target lies outside the dataset's folders (and outside the folder it is listed in)."""
    if not os.path.islink(path):
        return False
    real = os.path.realpath(path)
    for base in (*bases, os.path.realpath(os.path.dirname(path))):
        if real == base or real.startswith(base.rstrip(os.sep) + os.sep):
            return False
    return True


def _refresh_layout(dataset_id: str, layout: Layout) -> List[dict]:
    """label_dir_exists was computed when the dataset was added; after the first annotation of a bare
    dataset Overview kept saying '(will be created)'. Refresh it on every scan, and re-check the labels-only
    folders (returned for the scan report). """
    changed = False
    for s in layout.sources:
        exists = bool(s.label_dir) and Path(s.label_dir).is_dir() if not s.list_file else True
        if exists != s.label_dir_exists:
            s.label_dir_exists = exists
            changed = True
    try:
        orphans = find_unmatched_labels(layout)
    except Exception:  # noqa: BLE001 — a report detail must never fail a finished scan
        orphans = list(layout.unmatched_labels)
    if orphans != list(layout.unmatched_labels):
        changed = True
    if changed:
        row = db.query_one("SELECT layout FROM datasets WHERE id=?", (dataset_id,))
        lay = db.loads(row["layout"], {}) if row else {}
        for d, s in zip(lay.get("sources", []), layout.sources):
            d["label_dir_exists"] = s.label_dir_exists
        lay["unmatched_labels"] = orphans
        notes = [n for n in lay.get("notes", []) if not str(n).endswith(UNMATCHED_NOTE)]
        root_host = to_host(layout.root)
        for u in orphans[:5]:
            rel = u["path_host"]
            if rel.startswith(root_host.rstrip("/") + "/"):
                rel = rel[len(root_host.rstrip("/")) + 1:]
            notes.append(f"{rel}: {int(u.get('files') or 0):,} {UNMATCHED_NOTE}")
        lay["notes"] = notes
        db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), dataset_id))
    return orphans


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
    lmtime = lsize = None
    if lbl and lbl.is_file():
        has_label = 1
        try:
            st = lbl.stat()
            lmtime, lsize = st.st_mtime, st.st_size
        except OSError:
            pass
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
    for attempt in range(3):
        try:
            with db.transaction() as conn:
                conn.execute("UPDATE images SET has_label=?, n_boxes=?, classes=?, issues=?, lmtime=?, lsize=?, width=?, height=? "
                             "WHERE id=?",
                             (has_label, len(boxes), db.dumps(cls_ids), db.dumps(sorted(issues)), lmtime, lsize, width, height,
                              image_id))
                conn.execute("DELETE FROM boxes WHERE image_id=?", (image_id,))
                if boxes:
                    conn.executemany("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)",
                                     [(image_id, row["dataset_id"], int(b[0]), b[1], b[2], b[3], b[4]) for b in boxes])
            break
        except sqlite3.OperationalError as e:
            # The label file is already written; a busy index must not turn the user's own save into an
            # error (or a false "changed elsewhere" on retry). Wait and try again.
            if "locked" not in str(e) or attempt == 2:
                raise
            time.sleep(0.5 * (attempt + 1))
    if update_counts:
        refresh_dataset_counts(row["dataset_id"])
    return db.query_one("SELECT * FROM images WHERE id=?", (image_id,))
