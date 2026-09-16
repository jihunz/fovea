"""YOLO label file parsing, validation and writing."""
from __future__ import annotations

import hashlib
import math
import os
import stat
import tempfile
import threading
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

Box = List[float]  # [cls, xc, yc, w, h] or [cls, xc, yc, w, h, conf]

ISSUE_LABELS = {
    "missing_label": "No label file",
    "empty_label": "Empty label file",
    "bad_format": "Unparsable line",
    "bad_class": "Class id out of range",
    "out_of_range": "Box outside image",
    "zero_area": "Zero-area box",
    "tiny_box": "Tiny box (< 0.01% area)",
    "duplicate_box": "Duplicate box",
    "image_unreadable": "Image unreadable",
}

MAX_CLASS_ID = 2 ** 31 - 1   # anything larger cannot be stored and is certainly not a real class id

# Every read-modify-write of a label file (editor saves, bulk ops, auto-label) holds this lock, so a
# version check and the write that follows it can never interleave with another writer in this process.
LABEL_LOCK = threading.RLock()


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


def _has_duplicate(boxes: List[Box], threshold: float = 0.95, budget: int = 5_000_000) -> bool:
    """Exact duplicate search without an arbitrary size cap.

    Sort by (class, left edge) and sweep: two boxes can only overlap if they share a class and the
    later one starts before the earlier one ends, so the inner walk stops at the first box that
    starts past the current right edge. No true duplicate is ever skipped (unlike grid bucketing,
    which misses pairs that straddle a cell boundary)."""
    n = len(boxes)
    if n < 2:
        return False
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
                return False
            if iou(bi, bj) > threshold:
                return True
    return False


def parse_label_text(text: str, n_classes: Optional[int] = None) -> Tuple[List[Box], List[Optional[float]], Set[str]]:
    boxes: List[Box] = []
    confs: List[Optional[float]] = []
    issues: Set[str] = set()
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
            xc, yc, w, h = (float(x) for x in parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
        except ValueError:
            issues.add("bad_format")
            continue
        # nan/inf parse fine as floats but cannot be stored (NOT NULL), serialised (JSON) or drawn.
        if not all(math.isfinite(v) for v in (cls_f, xc, yc, w, h)) or abs(cls_f) > MAX_CLASS_ID:
            issues.add("bad_format")
            continue
        if conf is not None and not math.isfinite(conf):
            issues.add("bad_format")
            conf = None
        cls = int(cls_f)
        if cls < 0 or (n_classes is not None and n_classes > 0 and cls >= n_classes):
            issues.add("bad_class")
        if w <= 0 or h <= 0:
            issues.add("zero_area")
        elif w * h < 1e-4:
            issues.add("tiny_box")
        eps = 0.005
        if not (-eps <= xc - w / 2 and xc + w / 2 <= 1 + eps and -eps <= yc - h / 2 and yc + h / 2 <= 1 + eps):
            issues.add("out_of_range")
        boxes.append([cls, xc, yc, w, h])
        confs.append(conf)
    if _has_duplicate(boxes):
        issues.add("duplicate_box")
    if not boxes and not issues:
        issues.add("empty_label")
    return boxes, confs, issues


def read_label_file(path: Path, n_classes: Optional[int] = None) -> Tuple[List[Box], List[Optional[float]], Set[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], [], {"missing_label"}
    return parse_label_text(text, n_classes)


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
    boxes, confs, issues = parse_label_text(data.decode("utf-8", errors="replace"), n_classes)
    return boxes, confs, issues, _digest(data)


def write_label_file(path: Path, boxes: Iterable[Sequence[float]]) -> int:
    """Write YOLO labels atomically.

    A write that would not change the file is skipped entirely — not its bytes, its mtime, nor the
    precision the author wrote coordinates with — so merely opening an image never rewrites it.
    An existing file keeps its permission bits (mkstemp would otherwise leave every rewritten label
    at 0600, which breaks shared dataset folders)."""
    path = Path(path)
    rows = [list(b) for b in boxes]
    if any(len(r) >= 6 and r[5] is not None for r in rows):
        # A file is either all 5-column or all 6-column: YOLO readers reject ragged rows. A box added by
        # hand to a prediction file is as certain as it gets.
        rows = [r if len(r) >= 6 and r[5] is not None else [*r[:5], 1.0] for r in rows]
    else:
        rows = [r[:5] for r in rows]
    lines = [format_box(b) for b in rows]
    data = ("\n".join(lines) + "\n") if lines else ""

    mode: Optional[int] = None
    if path.is_file():
        try:
            text = path.read_text(encoding="utf-8", errors="replace")
            if text == data:
                return len(lines)
            existing, confs, _issues = parse_label_text(text)
            if _same_rows(with_confs(existing, confs), rows):
                return len(lines)
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
    return len(lines)
