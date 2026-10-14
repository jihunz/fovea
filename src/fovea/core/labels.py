"""YOLO label file parsing, validation and writing."""
from __future__ import annotations

import errno
import hashlib
import logging
import math
import os
import stat
import tempfile
import threading
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Sequence, Set, Tuple

Box = List[float]  # [cls, xc, yc, w, h] or [cls, xc, yc, w, h, conf]

log = logging.getLogger("fovea.labels")

# Problems the label parser finds in one label file (stored in images.issues).
PARSER_ISSUE_LABELS = {
    "missing_label": "No label file",
    "empty_label": "Empty label file",
    "bad_format": "Malformed line",
    "bad_class": "Invalid class id",
    "out_of_range": "Box outside image",
    "zero_area": "Zero-area box",
    "tiny_box": "Tiny box (< 0.01% area)",
    "duplicate_box": "Duplicate box",
    "image_unreadable": "Image unreadable",
}
LABEL_ISSUE_CODES = frozenset(PARSER_ISSUE_LABELS)
# "No label file" is progress (not labeled yet), not a defect: it stays filterable but is not counted as a
# problem in dataset health.
INFO_ISSUE_CODES = frozenset({"missing_label"})

# Dataset QA flags (image_flags.code), computed by the QA engine rather than the label parser.
FLAG_LABELS = {
    "near_dup": "Near-duplicate image",
    "leak_split": "Duplicate across splits",
    "dup_same_role": "Same frame in two folders of one split",   # qa-engine: informational, not a leak
    "dup_label_mismatch": "Duplicates labeled differently",
    "seq_gap": "Box missing in one frame of a sequence",
    "seq_flicker": "Class flickers in a sequence",
    "seq_jump": "Box jumps in a sequence",
    "seq_count_spike": "Box count spike in a sequence",
    "shape_outlier": "Unusual box shape for its class",
    "blurry": "Blurry image",
    "dark": "Dark image",
    "overexposed": "Overexposed image",
}
# AI label-audit findings: pending ai_preds of these kinds.
AI_AUDIT_KINDS = {"ai_missing": "missing", "ai_loose": "loose", "ai_swap": "swap", "ai_spurious": "spurious"}
AI_AUDIT_LABELS = {
    "ai_missing": "AI: likely missed object",
    "ai_loose": "AI: loose box",
    "ai_swap": "AI: possibly wrong class",
    "ai_spurious": "AI: box without support",
}
AI_PROPOSAL_KINDS = ("detect", "prev_frame", "interp", "refine")
# Every human-readable label for an `issue=` code (parser issues, QA flags, AI audit findings), so no view
# hard-codes them; served as `issue_labels` by /stats and /images/facets.
ISSUE_LABELS = {**PARSER_ISSUE_LABELS, **FLAG_LABELS, **AI_AUDIT_LABELS}
ALL_ISSUE_LABELS = ISSUE_LABELS

MAX_CLASS_ID = 2 ** 31 - 1   # anything larger cannot be stored and is certainly not a real class id
# A made-up class list (class_0..class_N for a dataset that names none, a yaml `nc`, the gaps of a sparse `names`
# map) never holds more entries than this. One stray row such as '2000000000 0.5 0.5 0.1 0.1' used to build two
# billion names and run the server out of memory; an id at or above it is flagged bad_class instead.
MAX_INFERRED_CLASSES = 4096


def inferred_names(classes: Sequence[str], max_cls: int) -> List[str]:
    """`classes` grown with class_<id> names to cover `max_cls`, never past MAX_INFERRED_CLASSES entries."""
    top = min(int(max_cls), MAX_INFERRED_CLASSES - 1)
    return list(classes) + [f"class_{i}" for i in range(len(classes), top + 1)]

# Every read-modify-write of a label file (editor saves, bulk ops, auto-label) holds this lock, so a
# version check and the write that follows it can never interleave with another writer in this process.
LABEL_LOCK = threading.RLock()

# Called as hook(dataset_id, image_ids) after label files were written and re-indexed: the editor's PUT,
# labels/bulk, snapshot restore, the proposals-accept job, Objects edits and AI write targets. The QA engine
# hangs its debounced re-check of label-based flags here. A hook runs on the writer's thread, so it must be
# cheap or only schedule work; one that raises is logged and never fails the write.
LABEL_WRITE_HOOKS: List[Callable[[str, List[int]], None]] = []


def notify_label_write(dataset_id: str, image_ids: Iterable[Optional[int]]) -> List[int]:
    """Tell every LABEL_WRITE_HOOK that these images' label files changed. Call it AFTER the write and the
    index refresh (refresh_image), outside LABEL_LOCK. Returns the de-duplicated ids that were announced."""
    ids: List[int] = []
    seen: Set[int] = set()
    for v in image_ids or ():
        if v is None or isinstance(v, bool):
            continue
        try:
            i = int(v)
        except (TypeError, ValueError):
            continue
        if i not in seen:
            seen.add(i)
            ids.append(i)
    if not ids or not dataset_id:
        return ids
    for hook in list(LABEL_WRITE_HOOKS):
        try:
            hook(dataset_id, list(ids))
        except Exception:  # noqa: BLE001 — a hook must never turn a saved label into an error
            log.exception("label-write hook %r failed for dataset %s", getattr(hook, "__name__", hook), dataset_id)
    return ids


class LabelLinkOutside(OSError):
    """The label file is a symbolic link to a file outside the dataset's folders. Writing through it could
    overwrite anything (a downloaded dataset may ship `labels/x.txt -> ~/.zshrc`); replacing the link would
    silently fork a label shared by link. Fovea does neither and refuses the write."""

    def __init__(self, path: str):
        super().__init__(errno.EPERM, "the label file is a link to a file outside the dataset, so Fovea does not "
                                      "write through it (edit the file it points to, or replace the link)", path)


def _inside(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


def dataset_bases(root: str, layout: Optional[dict]) -> List[str]:
    """Real paths of a dataset's folders: its root and every source's images / labels / list-base folder (a
    split may live outside the root, e.g. a data.yaml pointing at /data/train)."""
    out = [root]
    for s in (layout or {}).get("sources", []) or []:
        for k in ("img_dir", "label_dir", "base"):
            if s.get(k):
                out.append(s[k])
        if s.get("list_file"):
            out.append(os.path.dirname(s["list_file"]))
    return list(dict.fromkeys(os.path.realpath(p) for p in out if p))


def _bases_of(dataset_id: Optional[str], label_path: str) -> List[str]:
    from .. import db                      # labels.py stays importable without a database for parsing
    if dataset_id:
        row = db.query_one("SELECT root, layout FROM datasets WHERE id=?", (dataset_id,))
    else:                                  # rare (only for a link): find the dataset that indexes this label file
        row = db.query_one("SELECT d.root, d.layout FROM images i JOIN datasets d ON d.id = i.dataset_id "
                           "WHERE i.label_path=? LIMIT 1", (label_path,))
    return dataset_bases(row["root"], db.loads(row["layout"], {})) if row else []


def write_target(path: str | Path, dataset_id: Optional[str] = None) -> Path:
    """The file a write of label `path` must replace. A plain path is itself. A symlinked label file is written
    THROUGH the link (the file it points to is the label; replacing the link would leave the shared original
    stale) when its target lies inside the dataset — its root or one of its source folders — and refused with
    LabelLinkOutside otherwise. Every label writer (editor PUT, bulk ops, restore, Objects, AI) goes through it."""
    p = Path(path)
    if not os.path.islink(p):
        return p
    real = os.path.realpath(p)
    if any(_inside(real, b) for b in _bases_of(dataset_id, str(p))):
        return Path(real)
    raise LabelLinkOutside(str(p))


def iou(a: Sequence[float], b: Sequence[float]) -> float:
    ax0, ay0, ax1, ay1 = a[1] - a[3] / 2, a[2] - a[4] / 2, a[1] + a[3] / 2, a[2] + a[4] / 2
    bx0, by0, bx1, by1 = b[1] - b[3] / 2, b[2] - b[4] / 2, b[1] + b[3] / 2, b[2] + b[4] / 2
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = a[3] * a[4] + b[3] * b[4] - inter
    return inter / union if union > 0 else 0.0


def _duplicate_rows(boxes: List[Box], threshold: float = 0.95, budget: int = 5_000_000) -> Set[int]:
    """Indices of rows that duplicate another row. Exact search without an arbitrary size cap.

    Sort by (class, left edge) and sweep: two boxes can only overlap if they share a class and the
    later one starts before the earlier one ends, so the inner walk stops at the first box that
    starts past the current right edge. No true duplicate is ever skipped (unlike grid bucketing,
    which misses pairs that straddle a cell boundary)."""
    n = len(boxes)
    found: Set[int] = set()
    if n < 2:
        return found
    order = sorted(range(n), key=lambda i: (boxes[i][0], boxes[i][1] - boxes[i][3] / 2))
    ops = 0
    for p in range(n):
        bi = boxes[order[p]]
        right = bi[1] + bi[3] / 2
        for q in range(p + 1, n):
            bj = boxes[order[q]]
            if bj[0] != bi[0] or bj[1] - bj[3] / 2 >= right:
                break
            ops += 1
            if ops > budget:          # pathological overlap; give up rather than stall the scan
                return found
            if iou(bi, bj) > threshold:
                found.update((order[p], order[q]))
    return found


def _has_duplicate(boxes: List[Box], threshold: float = 0.95) -> bool:
    return bool(_duplicate_rows(boxes, threshold))


class LabelRow:
    """One parsed label line.

    `kind` is "box" (5 values), "conf" (box + confidence), or for Ultralytics segmentation / OBB ("seg":
    cls + an even number of polygon coordinates), pose ("pose": cls xc yc w h + k*(x y v)) and anything
    else longer ("ext"). Extended rows are displayed and edited as their bounding box, but their raw
    tokens (`tail`, everything after the class id) are written back verbatim — Fovea never turns a polygon
    or a skeleton into a rectangle."""

    __slots__ = ("cls", "box", "conf", "kind", "tail", "issues")

    def __init__(self, cls: int, box: List[float], conf: Optional[float], kind: str, tail: Optional[List[str]],
                 issues: Set[str]):
        self.cls, self.box, self.conf, self.kind, self.tail, self.issues = cls, box, conf, kind, tail, issues

    @property
    def extended(self) -> bool:
        return self.kind in ("seg", "pose", "ext")

    def as_box(self) -> Box:
        return [self.cls] + list(self.box)


def _is_pose(vals: Sequence[float]) -> bool:
    """Pose-SHAPED: 4 box values + k*(x y v) with every v in {0, 1, 2}. A polygon whose vertices touch the image
    border can have that shape too (x = 0 or y = 1 in a visibility slot), so this alone decides nothing."""
    k3 = len(vals) - 4
    return k3 >= 3 and k3 % 3 == 0 and all(vals[4 + 3 * j + 2] in (0.0, 1.0, 2.0) for j in range(k3 // 3))


POSE_MARGIN = 0.1      # a keypoint may lie this fraction of its box's size outside the box ...
POSE_SLACK = 0.02      # ... plus this much (normalised): annotators' boxes are not always tight


def _pose_fits(vals: Sequence[float]) -> bool:
    """A pose row's keypoints belong to its own box: each one lies inside it (with a margin), or is unlabelled
    and written as 0 0 0. A polygon read as a pose row almost never passes: its 'keypoints' are vertices far
    from the 'box' that its first two vertices make up."""
    xc, yc, w, h = vals[0], vals[1], vals[2], vals[3]
    if not (w > 0 and h > 0):
        return False
    mx, my = POSE_SLACK + POSE_MARGIN * w, POSE_SLACK + POSE_MARGIN * h
    x0, x1, y0, y1 = xc - w / 2 - mx, xc + w / 2 + mx, yc - h / 2 - my, yc + h / 2 + my
    for j in range((len(vals) - 4) // 3):
        x, y, v = vals[4 + 3 * j], vals[5 + 3 * j], vals[6 + 3 * j]
        if (x0 <= x <= x1 and y0 <= y <= y1) or (v == 0 and x == 0 and y == 0):
            continue
        return False
    return True


def looks_like_pose(vals: Sequence[float]) -> bool:
    """One row judged on its own (prediction files): pose-shaped AND its keypoints fit its box."""
    return _is_pose(vals) and _pose_fits(vals)


def _file_is_pose(ext_vals: List[List[float]]) -> bool:
    """Decide for a whole file whether its pose-shaped rows are pose rows. Ultralytics pose files use one row
    length throughout; a segmentation file whose polygon happens to be pose-shaped is told apart by the
    keypoints: in a pose file most rows have every keypoint inside their own box. (A pose-shaped row with an
    odd number of values cannot be a polygon and is a pose row either way.)"""
    if not ext_vals:
        return False
    n = len(ext_vals[0])
    if any(len(v) != n for v in ext_vals) or not all(_is_pose(v) for v in ext_vals):
        return False
    return sum(1 for v in ext_vals if _pose_fits(v)) * 2 > len(ext_vals)


def _classify_extended(vals: List[float], pose_file: bool = False) -> Tuple[str, List[float]]:
    """(kind, [xc, yc, w, h]) for a row with more than 5 values after the class id. An even-length row is a
    polygon unless the file as a whole reads as a pose file (_file_is_pose)."""
    if _is_pose(vals) and (pose_file or len(vals) % 2 == 1):
        return "pose", vals[:4]
    if len(vals) % 2 == 0:
        xs, ys = vals[0::2], vals[1::2]
        x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
        return "seg", [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]
    return "ext", vals[:4]


def parse_label_rows(text: str, n_classes: Optional[int] = None) -> Tuple[List[LabelRow], Set[str]]:
    """Parse YOLO label text into rows plus the file's issue codes. Each row also carries its own issues
    (which box is tiny, out of range, ...)."""
    rows: List[LabelRow] = []
    issues: Set[str] = set()
    if text.startswith("\ufeff"):          # UTF-8 BOM: float('\ufeff0') fails and the first box vanished
        text = text[1:]
    parsed: List[Tuple[List[str], float, List[float]]] = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line:
            continue
        parts = line.split()
        if len(parts) < 5:
            issues.add("bad_format")
            continue
        try:
            cls_f = float(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            issues.add("bad_format")
            continue
        parsed.append((parts, cls_f, vals))
    pose_file = _file_is_pose([v for _p, _c, v in parsed if len(v) > 5 and all(math.isfinite(x) for x in v)])
    for parts, cls_f, vals in parsed:
        if len(vals) <= 5:
            xc, yc, w, h = vals[:4]
            conf = vals[4] if len(vals) == 5 else None
            kind, tail = ("conf" if conf is not None else "box"), None
        else:
            if not all(math.isfinite(v) for v in vals):
                issues.add("bad_format")
                continue
            kind, (xc, yc, w, h) = _classify_extended(vals, pose_file)
            conf, tail = None, parts[1:]
        # nan/inf parse fine as floats but cannot be stored (NOT NULL), serialised (JSON) or drawn.
        if not all(math.isfinite(v) for v in (cls_f, xc, yc, w, h)) or abs(cls_f) > MAX_CLASS_ID:
            issues.add("bad_format")
            continue
        row_issues: Set[str] = set()
        if conf is not None and not math.isfinite(conf):
            row_issues.add("bad_format")
            conf = None
        if cls_f != int(cls_f):              # "1.7" is not class 1: flag it rather than truncate silently
            row_issues.add("bad_format")
        cls = int(cls_f)
        # Without a declared list the inferred one grows to cover the id, up to MAX_INFERRED_CLASSES.
        if cls < 0 or cls >= (n_classes if n_classes else MAX_INFERRED_CLASSES):
            row_issues.add("bad_class")
        if w <= 0 or h <= 0:
            row_issues.add("zero_area")
        elif w * h < 1e-4:
            row_issues.add("tiny_box")
        eps = 0.005
        if not (-eps <= xc - w / 2 and xc + w / 2 <= 1 + eps and -eps <= yc - h / 2 and yc + h / 2 <= 1 + eps):
            row_issues.add("out_of_range")
        issues |= row_issues
        rows.append(LabelRow(cls, [xc, yc, w, h], conf, kind, tail, row_issues))
    dups = _duplicate_rows([r.as_box() for r in rows])
    if dups:
        issues.add("duplicate_box")
        for i in dups:
            rows[i].issues.add("duplicate_box")
    if not rows and not issues:
        issues.add("empty_label")
    return rows, issues


def parse_label_text(text: str, n_classes: Optional[int] = None) -> Tuple[List[Box], List[Optional[float]], Set[str]]:
    rows, issues = parse_label_rows(text, n_classes)
    return [r.as_box() for r in rows], [r.conf for r in rows], issues


def _decode(data: bytes) -> str:
    return data.decode("utf-8-sig", errors="replace")


def read_label_file(path: Path, n_classes: Optional[int] = None) -> Tuple[List[Box], List[Optional[float]], Set[str]]:
    try:
        text = _decode(Path(path).read_bytes())
    except OSError:
        return [], [], {"missing_label"}
    return parse_label_text(text, n_classes)


def read_label_details(path: Path, n_classes: Optional[int] = None) -> Optional[dict]:
    """Everything the editor needs about one label file, from one read: boxes, confs, file issues, the
    issues of each row, each row's kind (extended rows are geometry-locked), and the version. None when
    the file does not exist."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return None
    rows, issues = parse_label_rows(_decode(data), n_classes)
    return {
        "boxes": [r.as_box() for r in rows], "confs": [r.conf for r in rows], "issues": issues,
        "row_issues": [sorted(r.issues) for r in rows], "row_kinds": [r.kind for r in rows],
        "version": _digest(data),
    }


def with_confs(boxes: List[Box], confs: List[Optional[float]]) -> List[Box]:
    """Fold a parallel confs list into the boxes as a 6th element. Code that filters or reorders boxes
    must carry the confidence inside each row — a positional side list silently misassigns scores."""
    return [list(b[:5]) + ([c] if c is not None else []) for b, c in zip(boxes, confs)]


def read_label_rows(path: Path, n_classes: Optional[int] = None) -> Tuple[List[Box], Set[str]]:
    boxes, confs, issues = read_label_file(path, n_classes)
    return with_confs(boxes, confs), issues


def format_box(b: Sequence[float]) -> str:
    s = f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
    if len(b) >= 6 and b[5] is not None:
        s += f" {float(b[5]):.6f}"
    return s


def validate_boxes(boxes: Iterable[Sequence[float]]) -> Tuple[List[Box], int]:
    """Structural validation for boxes a USER is writing. Returns (kept, rejected_count).

    This deliberately does not clamp, recentre or drop boxes for being tiny or slightly out of frame:
    those are the user's annotations, the index already flags them as issues, and silently "fixing"
    them on an unrelated edit is data loss. Only rows that cannot be stored at all are rejected."""
    out: List[Box] = []
    rejected = 0
    for b in boxes or []:
        try:
            if not isinstance(b, (list, tuple)) or len(b) < 5:
                raise ValueError
            c = float(b[0])
            vals = [float(x) for x in b[1:5]]
            conf = float(b[5]) if len(b) >= 6 and b[5] is not None else None
        except (TypeError, ValueError):
            rejected += 1
            continue
        if (not math.isfinite(c) or c < 0 or c != int(c) or c > MAX_CLASS_ID
                or not all(math.isfinite(v) for v in vals)):
            rejected += 1
            continue
        row: Box = [int(c)] + vals
        if conf is not None and math.isfinite(conf):
            row.append(conf)
        out.append(row)
    return out, rejected


def sanitize_boxes(boxes: Iterable[Sequence[float]]) -> List[Box]:
    """Clamp and drop degenerate boxes. Only for MACHINE-generated boxes (model detections) —
    never for a user's existing annotations; use validate_boxes for those."""
    out: List[Box] = []
    for b in boxes:
        if len(b) < 5:
            continue
        try:
            cls = int(float(b[0]))
            xc, yc, w, h = (float(x) for x in b[1:5])
        except (TypeError, ValueError, OverflowError):
            continue
        if cls < 0 or not all(math.isfinite(v) for v in (xc, yc, w, h)):
            continue
        x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        if x1 - x0 <= 0.0005 or y1 - y0 <= 0.0005:
            continue
        out.append([cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    return out


def _same_rows(existing: List[Box], wanted: List[Box]) -> bool:
    if len(existing) != len(wanted):
        return False
    for a, b in zip(existing, wanted):
        if int(a[0]) != int(b[0]):
            return False
        if any(abs(float(a[i]) - float(b[i])) >= 5e-7 for i in range(1, 5)):
            return False
        ac = a[5] if len(a) >= 6 else None
        bc = b[5] if len(b) >= 6 else None
        if (ac is None) != (bc is None) or (ac is not None and abs(float(ac) - float(bc)) >= 5e-7):
            return False
    return True


def _digest(data: bytes) -> str:
    return hashlib.blake2b(data, digest_size=12).hexdigest()


def read_label_snapshot(path: Path, n_classes: Optional[int] = None):
    """Read a label file once: (boxes, confs, issues, version). The version always describes exactly the
    bytes the boxes were parsed from; version is None when the file does not exist."""
    try:
        data = Path(path).read_bytes()
    except OSError:
        return [], [], {"missing_label"}, None
    boxes, confs, issues = parse_label_text(_decode(data), n_classes)
    return boxes, confs, issues, _digest(data)


class ExtendedRowsChanged(ValueError):
    """An edit moved or resized a segmentation / pose / OBB row. Fovea shows those rows as their bounding
    box, so it cannot know what the polygon or skeleton should become — it refuses rather than replace it
    with a rectangle."""


_GEOM_TOL = 2e-6


def _same_geom(a: Sequence[float], b: Sequence[float]) -> bool:
    return all(abs(float(a[i]) - float(b[i])) <= _GEOM_TOL for i in range(4))


def _fmt(v: float) -> str:
    return f"{float(v):.6f}"


def _box_as_extended(b: Sequence[float], style: str, pose_len: int) -> str:
    """A plain box written into a file of extended rows, in that file's own row format: a rectangle polygon
    for segmentation / OBB files (Ultralytics reads every row of such a file as a polygon, so a 5-value row
    would become a 2-point polygon), or the box with zeroed, invisible keypoints for pose files."""
    cls, xc, yc, w, h = int(b[0]), float(b[1]), float(b[2]), float(b[3]), float(b[4])
    if style == "pose":
        return " ".join([str(cls), _fmt(xc), _fmt(yc), _fmt(w), _fmt(h)] + ["0"] * max(0, pose_len - 4))
    x0, y0, x1, y1 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
    return " ".join([str(cls)] + [_fmt(v) for v in (x0, y0, x1, y0, x1, y1, x0, y1)])


ALIGN_MAX_PAIRS = 600   # above this many same-geometry (row, existing row) pairs, match greedily instead


def _match_rows(existing: List[LabelRow], rows: List[Box]) -> List[int]:
    """For each incoming row, the index of the existing row it is (same geometry), or -1.

    Rows are matched as two ordered lists — an editor keeps the file's order, deleting and re-classing rows in
    place — by the order-preserving matching with the most matches, then the most rows keeping their class. So
    two annotations with one bounding box (a square and a diamond polygon, say, under different classes) each
    keep their own shape when one is deleted or both are re-classed. Rows left over (an unusual re-ordering)
    fall back to geometry alone, preferring the same class."""
    pairs = [(i, j) for i, b in enumerate(rows) for j, r in enumerate(existing) if _same_geom(r.box, b[1:5])]
    match = [-1] * len(rows)
    used = [False] * len(existing)
    if len(pairs) <= ALIGN_MAX_PAIRS:
        big = len(rows) + 1                  # one more match always beats any number of kept classes
        score: List[int] = []
        prev: List[int] = []
        for k, (i, j) in enumerate(pairs):
            w = big + (1 if int(rows[i][0]) == existing[j].cls else 0)
            best, arg = 0, -1
            for m in range(k):
                pi, pj = pairs[m]
                if pi < i and pj < j and score[m] > best:
                    best, arg = score[m], m
            score.append(best + w)
            prev.append(arg)
        k = max(range(len(pairs)), key=lambda m: score[m], default=-1)
        while k >= 0:
            i, j = pairs[k]
            match[i], used[j] = j, True
            k = prev[k]
    for i, b in enumerate(rows):
        if match[i] >= 0:
            continue
        cands = [j for j, r in enumerate(existing) if not used[j] and _same_geom(r.box, b[1:5])]
        if cands:
            j = min(cands, key=lambda j: (existing[j].cls != int(b[0]), j))
            match[i], used[j] = j, True
    return match


def _extended_lines(existing: List[LabelRow], rows: List[Box], strict: bool) -> List[str]:
    ext = [r for r in existing if r.extended]
    kinds = Counter(r.kind for r in ext)
    style = "pose" if kinds.get("pose", 0) > len(ext) / 2 else "seg"
    pose_len = max((len(r.tail) for r in ext if r.kind == "pose"), default=4)
    match = _match_rows(existing, rows)
    taken = {j for j in match if j >= 0}
    lines: List[str] = []
    unmatched_new = 0
    for b, j in zip(rows, match):
        if j >= 0 and existing[j].extended:
            lines.append(" ".join([str(int(b[0]))] + existing[j].tail))
            continue
        if j < 0 and not any(_same_geom(r.box, b[1:5]) for r in existing if not r.extended):
            unmatched_new += 1
        lines.append(_box_as_extended(b, style, pose_len))
    used = [j in taken for j, r in enumerate(existing) if r.extended]
    if strict and unmatched_new and not all(used):
        gone = [i for i, u in enumerate(used) if not u]
        raise ExtendedRowsChanged(
            f"{len(gone)} segmentation/pose row(s) were moved or resized. Fovea can change their class or delete "
            "them, but not reshape a polygon or skeleton; edit those rows in a polygon tool.")
    return lines


def _semantic(lines: List[str]) -> List[Tuple[float, ...]]:
    out = []
    for ln in lines:
        try:
            out.append(tuple(float(x) for x in ln.split()))
        except ValueError:
            out.append((float("nan"),))
    return out


def label_text_for(rows: Iterable[Sequence[float]], existing_text: str = "", strict: bool = False) -> str:
    """The exact file content `write_label_file` would produce for `rows` over a file currently holding
    `existing_text` (extended rows keep their raw tokens)."""
    rows = [list(b) for b in rows]
    existing, _ = parse_label_rows(existing_text) if existing_text else ([], set())
    if any(r.extended for r in existing):
        lines = _extended_lines(existing, rows, strict)
    else:
        if any(len(r) >= 6 and r[5] is not None for r in rows):
            # A file is either all 5-column or all 6-column: YOLO readers reject ragged rows. A box added by
            # hand to a prediction file is as certain as it gets.
            rows = [r if len(r) >= 6 and r[5] is not None else [*r[:5], 1.0] for r in rows]
        else:
            rows = [r[:5] for r in rows]
        lines = [format_box(b) for b in rows]
    return ("\n".join(lines) + "\n") if lines else ""


def write_label_file(path: Path, boxes: Iterable[Sequence[float]], strict: bool = False,
                     dataset_id: Optional[str] = None) -> int:
    """Write YOLO labels atomically.

    A write that would not change the file is skipped entirely — not its bytes, its mtime, nor the
    precision the author wrote coordinates with — so merely opening an image never rewrites it.
    An existing file keeps its permission bits (mkstemp would otherwise leave every rewritten label
    at 0600, which breaks shared dataset folders).

    Segmentation / pose / OBB rows are matched to the incoming boxes by geometry and written back with
    their original tokens (and the incoming class id). With strict=True (an editor save) a moved or
    resized extended row raises ExtendedRowsChanged instead of being flattened into a rectangle.

    A symlinked label file is written through its link when the link stays inside the dataset, and refused
    (LabelLinkOutside, an OSError) otherwise — see write_target."""
    path = write_target(path, dataset_id)
    rows = [list(b) for b in boxes]
    mode: Optional[int] = None
    text = ""
    if path.is_file():
        try:
            text = _decode(path.read_bytes())
        except OSError:
            text = ""
    data = label_text_for(rows, text, strict)
    n_lines = data.count("\n")
    if path.is_file():
        try:
            if text == data:
                return n_lines
            existing, _issues = parse_label_rows(text)
            if any(r.extended for r in existing):
                if _semantic(data.splitlines()) == _semantic([ln.strip() for ln in text.splitlines() if ln.strip()]):
                    return n_lines
            elif _same_rows([r.as_box() + ([r.conf] if r.conf is not None else []) for r in existing], _rows_of(data)):
                return n_lines
        except Exception:  # noqa: BLE001 — never let a comparison stop a real save
            pass
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mode = None
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask

    fd, tmp = tempfile.mkstemp(prefix=".fovea-", suffix=".txt", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return n_lines


def _rows_of(data: str) -> List[Box]:
    boxes, confs, _ = parse_label_text(data)
    return with_confs(boxes, confs)


def export_label_text(path: Optional[str | Path]) -> str:
    """Label text for a training export: plain rows lose any confidence column (trainers reject 6-column
    rows), while segmentation / pose / OBB rows are copied verbatim."""
    if not path:
        return ""
    p = Path(path)
    try:
        text = _decode(p.read_bytes())
    except OSError:
        return ""
    rows, _ = parse_label_rows(text)
    if any(r.extended for r in rows):
        return label_text_for([r.as_box() for r in rows], text)
    return "".join(format_box(r.as_box()) + "\n" for r in rows)
