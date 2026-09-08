"""GT vs prediction evaluation (greedy IoU matching, per-class P/R/F1 and AP@IoU)."""
from __future__ import annotations

import os
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ... import db
from ...core.imagesq import iter_images
from ...core.labels import iou, read_label_file
from ...jobs import Job
from ...config import LABEL_EXT


def pred_file_for(pred_dir: Path, rel_path: str, split: str) -> Optional[Path]:
    rel = Path(rel_path)
    cands = [pred_dir / rel.with_suffix(LABEL_EXT)]
    if split and rel_path.startswith(split + "/"):
        cands.append(pred_dir / Path(rel_path[len(split) + 1:]).with_suffix(LABEL_EXT))
    cands.append(pred_dir / (rel.stem + LABEL_EXT))
    for c in cands:
        if c.is_file():
            return c
    return None


def load_preds(pred_dir: Path, rel_path: str, split: str, conf_thr: float) -> List[List[float]]:
    p = pred_file_for(pred_dir, rel_path, split)
    if p is None:
        return []
    boxes, confs, _ = read_label_file(p)
    out = []
    for b, c in zip(boxes, confs):
        conf = 1.0 if c is None else c
        if conf >= conf_thr:
            out.append([int(b[0]), b[1], b[2], b[3], b[4], conf])
    return out


def match_image(gt: List[List[float]], preds: List[List[float]], iou_thr: float) -> Tuple[List[bool], List[int]]:
    """Return (pred_is_tp[], gt_matched_by_pred_index[] (-1 if FN))."""
    order = sorted(range(len(preds)), key=lambda i: -preds[i][5])
    gt_matched = [-1] * len(gt)
    pred_tp = [False] * len(preds)
    for pi in order:
        p = preds[pi]
        best, best_iou = -1, iou_thr
        for gi, g in enumerate(gt):
            if gt_matched[gi] != -1 or int(g[0]) != int(p[0]):
                continue
            v = iou(g, p)
            if v >= best_iou:
                best, best_iou = gi, v
        if best >= 0:
            gt_matched[best] = pi
            pred_tp[pi] = True
    return pred_tp, gt_matched


def average_precision(records: List[Tuple[float, bool]], n_gt: int) -> float:
    if n_gt == 0 or not records:
        return 0.0
    records = sorted(records, key=lambda r: -r[0])
    tp = fp = 0
    precisions, recalls = [], []
    for conf, is_tp in records:
        if is_tp:
            tp += 1
        else:
            fp += 1
        precisions.append(tp / (tp + fp))
        recalls.append(tp / n_gt)
    # all-point interpolation
    mrec = [0.0] + recalls + [1.0]
    mpre = [0.0] + precisions + [0.0]
    for i in range(len(mpre) - 2, -1, -1):
        mpre[i] = max(mpre[i], mpre[i + 1])
    ap = 0.0
    for i in range(1, len(mrec)):
        if mrec[i] != mrec[i - 1]:
            ap += (mrec[i] - mrec[i - 1]) * mpre[i]
    return ap


class SideAcc:
    def __init__(self):
        self.tp = defaultdict(int); self.fp = defaultdict(int); self.fn = defaultdict(int)
        self.records: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
        self.n_gt = defaultdict(int)

    def add(self, gt, preds, pred_tp, gt_matched):
        for g, m in zip(gt, gt_matched):
            c = int(g[0]); self.n_gt[c] += 1
            if m == -1:
                self.fn[c] += 1
        for p, ok in zip(preds, pred_tp):
            c = int(p[0])
            if ok:
                self.tp[c] += 1
            else:
                self.fp[c] += 1
            self.records[c].append((p[5], ok))

    def summary(self, names: List[str]) -> dict:
        classes = sorted(set(self.n_gt) | set(self.tp) | set(self.fp))
        rows, aps = [], []
        T = F = N = 0
        for c in classes:
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            T += tp; F += fp; N += fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            ap = average_precision(self.records[c], self.n_gt[c])
            if self.n_gt[c] > 0:
                aps.append(ap)
            rows.append({"cls": c, "name": names[c] if 0 <= c < len(names) else f"class_{c}", "gt": self.n_gt[c],
                         "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4), "ap": round(ap, 4)})
        prec = T / (T + F) if T + F else 0.0
        rec = T / (T + N) if T + N else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"classes": rows, "overall": {"gt": T + N, "tp": T, "fp": F, "fn": N, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                                             "map50": round(sum(aps) / len(aps), 4) if aps else 0.0}}


def gt_boxes_for(image_id: int) -> List[List[float]]:
    return [[r["cls"], r["xc"], r["yc"], r["w"], r["h"]] for r in db.query("SELECT cls, xc, yc, w, h FROM boxes WHERE image_id=? ORDER BY id", (image_id,))]


def evaluate(job: Job, dataset_id: str, names: List[str], pred_a: Path, pred_b: Optional[Path], iou_thr: float, conf_thr: float,
             filters: dict) -> dict:
    from ...core.imagesq import count_images
    total = count_images(dataset_id, filters)
    job.update(0, total, "Evaluating")
    acc = {"a": SideAcc(), "b": SideAcc() if pred_b else None}
    rows = []
    n_pred_files = {"a": 0, "b": 0}
    for k, item in enumerate(iter_images(dataset_id, filters)):
        if k % 100 == 0:
            job.update(done=k, message=f"{k:,}/{total:,}")
        gt = gt_boxes_for(item["id"])
        row = {"id": item["id"], "rel_path": item["rel_path"], "split": item["split"], "gt": len(gt)}
        for side, pdir in (("a", pred_a), ("b", pred_b)):
            if pdir is None:
                continue
            preds = load_preds(pdir, item["rel_path"], item["split"], conf_thr)
            if pred_file_for(pdir, item["rel_path"], item["split"]) is not None:
                n_pred_files[side] += 1
            pred_tp, gt_matched = match_image(gt, preds, iou_thr)
            acc[side].add(gt, preds, pred_tp, gt_matched)
            row[side] = {"tp": sum(pred_tp), "fp": len(preds) - sum(pred_tp), "fn": sum(1 for m in gt_matched if m == -1)}
        rows.append(row)
    rows.sort(key=lambda r: -(r.get("a", {}).get("fn", 0) + r.get("a", {}).get("fp", 0) + r.get("b", {}).get("fn", 0) + r.get("b", {}).get("fp", 0)))
    result = {
        "images": len(rows), "pred_files": n_pred_files,
        "summary": {"a": acc["a"].summary(names), "b": acc["b"].summary(names) if acc["b"] else None},
        "rows": rows[:20000],
    }
    job.update(done=total, message="Done")
    return result


def image_detail(image_id: int, rel_path: str, split: str, pred_a: Path, pred_b: Optional[Path], iou_thr: float, conf_thr: float) -> dict:
    gt = gt_boxes_for(image_id)
    out = {"gt": gt}
    for side, pdir in (("a", pred_a), ("b", pred_b)):
        if pdir is None:
            continue
        preds = load_preds(pdir, rel_path, split, conf_thr)
        pred_tp, gt_matched = match_image(gt, preds, iou_thr)
        out[side] = {"preds": [{"box": p[:5], "conf": p[5], "tp": ok} for p, ok in zip(preds, pred_tp)],
                     "gt_matched": gt_matched, "file": str(pred_file_for(pdir, rel_path, split) or "")}
    return out


def discover_pred_dirs(root: Path, max_depth: int = 5, limit: int = 300) -> List[dict]:
    """Find folders containing YOLO prediction txt files (>=6 columns) beneath root."""
    found = []
    root = root.resolve()
    for dirpath, dirs, files in os.walk(root):
        depth = len(Path(dirpath).relative_to(root).parts)
        dirs[:] = [d for d in dirs if not d.startswith(".")]
        if depth >= max_depth:
            dirs[:] = []
        txts = [f for f in files if f.lower().endswith(LABEL_EXT)]
        if len(txts) >= 3:
            sample = Path(dirpath) / txts[0]
            try:
                first = sample.read_text(encoding="utf-8", errors="ignore").strip().splitlines()
                cols = len(first[0].split()) if first else 0
            except OSError:
                cols = 0
            found.append({"path": dirpath, "files": len(txts), "has_conf": cols >= 6})
            if len(found) >= limit:
                break
    return found
