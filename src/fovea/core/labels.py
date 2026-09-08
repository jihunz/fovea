"""YOLO label file parsing, validation and writing."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Set, Tuple

Box = List[float]  # [cls, xc, yc, w, h]

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
            cls = int(float(parts[0]))
            xc, yc, w, h = (float(x) for x in parts[1:5])
            conf = float(parts[5]) if len(parts) >= 6 else None
        except ValueError:
            issues.add("bad_format")
            continue
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
    # duplicates (same class, near-identical geometry)
    n = len(boxes)
    if n > 1 and n <= 400:
        for i in range(n):
            for j in range(i + 1, n):
                if boxes[i][0] == boxes[j][0] and iou(boxes[i], boxes[j]) > 0.95:
                    issues.add("duplicate_box")
                    break
            if "duplicate_box" in issues:
                break
    if not boxes and not issues:
        issues.add("empty_label")
    return boxes, confs, issues


def read_label_file(path: Path, n_classes: Optional[int] = None) -> Tuple[List[Box], List[Optional[float]], Set[str]]:
    try:
        text = path.read_text(encoding="utf-8", errors="replace")
    except OSError:
        return [], [], {"missing_label"}
    return parse_label_text(text, n_classes)


def format_box(b: Sequence[float], conf: Optional[float] = None) -> str:
    s = f"{int(b[0])} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}"
    if conf is not None:
        s += f" {conf:.4f}"
    return s


def sanitize_boxes(boxes: Iterable[Sequence[float]]) -> List[Box]:
    out: List[Box] = []
    for b in boxes:
        if len(b) < 5:
            continue
        cls = max(0, int(float(b[0])))
        xc, yc, w, h = (float(x) for x in b[1:5])
        x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
        x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
        if x1 - x0 <= 0.0005 or y1 - y0 <= 0.0005:
            continue
        out.append([cls, (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0])
    return out


def write_label_file(path: Path, boxes: Iterable[Sequence[float]]) -> int:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [format_box(b) for b in boxes]
    data = ("\n".join(lines) + "\n") if lines else ""
    fd, tmp = tempfile.mkstemp(prefix=".fovea-", suffix=".txt", dir=str(path.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(data)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return len(lines)
