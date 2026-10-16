"""Dataset health statistics computed from the index."""
from __future__ import annotations

import threading
import time
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

from .. import db
from .labels import AI_AUDIT_KINDS, ALL_ISSUE_LABELS, FLAG_LABELS, INFO_ISSUE_CODES, ISSUE_LABELS
from .layout import split_order_key

SPLIT_ORDER = "CASE i.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 WHEN 'test' THEN 2 WHEN '' THEN 9 ELSE 3 END"
AREA_BINS = [
    ("< 0.1%", 0.001), ("0.1–1%", 0.01), ("1–5%", 0.05), ("5–20%", 0.2), ("> 20%", 9.0),
]
ASPECT_LABELS = ["<0.5", "0.5–0.8", "≈1", "1.25–2", ">2"]
HEAT_N = 12
UNDER_REPRESENTED = 0.1      # a class with < 10% of the median class's boxes is flagged as under-represented
CACHE_TTL = 600.0

_cache: Dict[str, Tuple[tuple, float, dict]] = {}
_cache_lock = threading.Lock()


def _version(dataset_id: str) -> Optional[tuple]:
    """Everything the cached part depends on. Every label write ends in refresh_dataset_counts (which bumps
    updated_at), a class-list edit bumps it too, and a scan sets scanned_at."""
    row = db.query_one("SELECT updated_at, scanned_at, image_count, box_count, label_count, classes, classes_source "
                       "FROM datasets WHERE id=?", (dataset_id,))
    return tuple(row.values()) if row else None


def invalidate(dataset_id: Optional[str] = None) -> None:
    with _cache_lock:
        if dataset_id is None:
            _cache.clear()
        else:
            _cache.pop(dataset_id, None)


def dataset_stats(dataset_id: str, class_names: List[str]) -> Dict[str, Any]:
    """Overview statistics. The heavy part (passes over every box) is cached per dataset version; review,
    QA-flag and AI counts are always fresh (they change without touching the label index)."""
    key = (_version(dataset_id), tuple(class_names))
    now = time.time()
    with _cache_lock:
        hit = _cache.get(dataset_id)
    if hit and hit[0] == key and now - hit[1] < CACHE_TTL:
        heavy = hit[2]
    else:
        heavy = _heavy(dataset_id, class_names)
        with _cache_lock:
            _cache[dataset_id] = (key, now, heavy)
    out = dict(heavy)
    out.update(_light(dataset_id))
    out["cached"] = bool(hit and hit[2] is heavy)
    return out


def class_counts(dataset_id: str) -> Dict[int, int]:
    """Boxes per class only (Annotate's class palette): one index-only GROUP BY."""
    return {int(r["cls"]): r["c"] for r in db.query("SELECT cls, COUNT(*) c FROM boxes WHERE dataset_id=? GROUP BY cls", (dataset_id,))}


def _heavy(dataset_id: str, class_names: List[str]) -> Dict[str, Any]:
    info = tuple(INFO_ISSUE_CODES)
    info_only = ["'[" + ",".join(f'"{c}"' for c in info) + "]'"] if info else []
    problem_expr = "issues != '[]'" + "".join(f" AND issues != {x}" for x in info_only)
    tot = db.query_one(f"""
        SELECT COUNT(*) images, COALESCE(SUM(has_label),0) labeled,
               COALESCE(SUM(CASE WHEN has_label=1 AND n_boxes=0 THEN 1 ELSE 0 END),0) empty,
               COALESCE(SUM(CASE WHEN n_boxes>0 THEN 1 ELSE 0 END),0) with_boxes,
               COALESCE(SUM(n_boxes),0) boxes,
               COALESCE(SUM(CASE WHEN {problem_expr} THEN 1 ELSE 0 END),0) with_issues,
               COALESCE(AVG(CASE WHEN has_label=1 THEN n_boxes END),0) avg_boxes
        FROM images WHERE dataset_id=?""", (dataset_id,)) or {}
    splits = db.query(f"""
        SELECT i.split, COUNT(*) images, COALESCE(SUM(i.has_label),0) labeled, COALESCE(SUM(i.n_boxes),0) boxes
        FROM images i WHERE i.dataset_id=? GROUP BY i.split ORDER BY {SPLIT_ORDER}, i.split""", (dataset_id,))
    splits.sort(key=lambda s: split_order_key(s["split"] or ""))      # other splits in natural order (fold2, fold10)
    split_names = [s["split"] for s in splits]

    cls_rows = db.query("""
        SELECT b.cls, i.split, COUNT(*) boxes, COUNT(DISTINCT b.image_id) images
        FROM boxes b JOIN images i ON i.id=b.image_id WHERE b.dataset_id=? GROUP BY b.cls, i.split""", (dataset_id,))
    classes: Dict[int, dict] = {}
    for r in cls_rows:
        c = classes.setdefault(int(r["cls"]), {"cls": int(r["cls"]), "boxes": 0, "images": 0, "by_split": {}})
        c["boxes"] += r["boxes"]
        c["images"] += r["images"]
        c["by_split"][r["split"]] = r["boxes"]
    for i, _name in enumerate(class_names):
        classes.setdefault(i, {"cls": i, "boxes": 0, "images": 0, "by_split": {}})
    counts_nonzero = sorted(c["boxes"] for c in classes.values() if c["boxes"] > 0)
    median = counts_nonzero[len(counts_nonzero) // 2] if counts_nonzero else 0
    labeled_splits = [s["split"] for s in splits if s["boxes"] > 0]
    class_list = []
    for cid in sorted(classes):
        c = classes[cid]
        c["name"] = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        c["unknown"] = not (0 <= cid < len(class_names))
        c["empty"] = c["boxes"] == 0
        c["under_represented"] = bool(median) and (c["boxes"] == 0 or c["boxes"] < UNDER_REPRESENTED * median)
        # A class that is labeled in some splits but has no box at all in another (that has boxes) cannot be
        # learned or evaluated there.
        c["missing_in"] = [s for s in labeled_splits if c["boxes"] > 0 and not c["by_split"].get(s)]
        class_list.append(c)

    bpi = db.query("""SELECT MIN(n_boxes, 10) k, COUNT(*) c FROM images WHERE dataset_id=? AND has_label=1
                      GROUP BY k ORDER BY k""", (dataset_id,))
    # Boxes per image by class: which classes the boxes of each bucket (the image's box count, 10 = 10+) are, and
    # each class's own count per image (how many images hold 1, 2 … 10+ boxes of it).
    bpi_cls = db.query("""SELECT MIN(i.n_boxes, 10) k, b.cls, COUNT(*) c FROM boxes b JOIN images i ON i.id = b.image_id
                          WHERE b.dataset_id=? AND i.has_label=1 GROUP BY k, b.cls""", (dataset_id,))
    per_cls = db.query("""SELECT cls, MIN(n, 10) k, COUNT(*) c FROM (SELECT image_id, cls, COUNT(*) n FROM boxes
                          WHERE dataset_id=? GROUP BY image_id, cls) GROUP BY cls, k""", (dataset_id,))
    dims = db.query("""SELECT width, height, COUNT(*) c FROM images WHERE dataset_id=? AND width IS NOT NULL
                       GROUP BY width, height ORDER BY c DESC LIMIT 8""", (dataset_id,))
    dims_total = db.query_one("SELECT COUNT(*) n FROM (SELECT 1 FROM images WHERE dataset_id=? AND width IS NOT NULL "
                              "GROUP BY width, height)", (dataset_id,)) or {"n": 0}

    # Area, aspect and position in ONE pass. Aspect is measured in pixels: normalised w/h is stretched by the
    # image's own aspect (a square object on a 16:9 frame has w/h = 0.56).
    ar = ("CASE WHEN i.width > 0 AND i.height > 0 THEN (b.w * i.width) / (b.h * i.height) ELSE b.w / b.h END")
    rows = db.query(f"""
        SELECT CASE WHEN b.w*b.h < 0.001 THEN 0 WHEN b.w*b.h < 0.01 THEN 1 WHEN b.w*b.h < 0.05 THEN 2
                    WHEN b.w*b.h < 0.2 THEN 3 ELSE 4 END abin,
               CASE WHEN b.h <= 0 THEN -1 WHEN {ar} < 0.5 THEN 0 WHEN {ar} < 0.8 THEN 1 WHEN {ar} < 1.25 THEN 2
                    WHEN {ar} < 2 THEN 3 ELSE 4 END rbin,
               MIN({HEAT_N - 1}, MAX(0, CAST(b.xc*{HEAT_N} AS INTEGER))) gx,
               MIN({HEAT_N - 1}, MAX(0, CAST(b.yc*{HEAT_N} AS INTEGER))) gy, COUNT(*) c
        FROM boxes b JOIN images i ON i.id = b.image_id WHERE b.dataset_id=? GROUP BY abin, rbin, gx, gy""", (dataset_id,))
    area_counts: Counter = Counter()
    aspect_counts: Counter = Counter()
    heat = [[0] * HEAT_N for _ in range(HEAT_N)]
    for r in rows:
        area_counts[int(r["abin"])] += r["c"]
        if int(r["rbin"]) >= 0:
            aspect_counts[int(r["rbin"])] += r["c"]
        heat[int(r["gy"])][int(r["gx"])] += r["c"]

    issue_counter: Counter = Counter()
    for r in db.query("SELECT issues, COUNT(*) c FROM images WHERE dataset_id=? AND issues != '[]' GROUP BY issues", (dataset_id,)):
        for code in db.loads(r["issues"]):
            issue_counter[code] += r["c"]
    issues = [{"code": code, "label": ISSUE_LABELS.get(code, code), "count": n,
               "severity": "info" if code in INFO_ISSUE_CODES else "warn"}
              for code, n in sorted(issue_counter.items(), key=lambda kv: (kv[0] in INFO_ISSUE_CODES, -kv[1]))]

    seqs = db.query_one("SELECT COUNT(*) n FROM (SELECT seq FROM images WHERE dataset_id=? AND seq != '' "
                        "GROUP BY seq HAVING COUNT(*) > 1)", (dataset_id,)) or {"n": 0}

    images = tot.get("images", 0) or 0
    labeled = tot.get("labeled", 0) or 0
    return {
        "totals": {
            "images": images, "labeled": labeled,
            "unlabeled": images - labeled, "empty": tot.get("empty", 0),
            "with_boxes": tot.get("with_boxes", 0),
            "boxes": tot.get("boxes", 0), "with_issues": tot.get("with_issues", 0),
            "avg_boxes": round(tot.get("avg_boxes", 0) or 0, 2), "classes": len(class_list),
            "sequences": seqs.get("n", 0),
            "empty_classes": sum(1 for c in class_list if c["empty"]),
        },
        "splits": splits,
        "split_names": split_names,
        "classes": class_list,
        "empty_classes": [c["cls"] for c in class_list if c["empty"]],
        "boxes_per_image": [{"k": int(r["k"]), "count": r["c"]} for r in bpi],
        "boxes_per_image_classes": [{"k": int(r["k"]), "cls": int(r["cls"]), "boxes": r["c"]} for r in bpi_cls],
        "class_per_image": [{"cls": int(r["cls"]), "k": int(r["k"]), "images": r["c"]} for r in per_cls],
        "dimensions": [{"width": r["width"], "height": r["height"], "count": r["c"]} for r in dims],
        "dimensions_distinct": dims_total.get("n", 0),
        "box_area": [{"label": lbl, "count": area_counts.get(i, 0)} for i, (lbl, _) in enumerate(AREA_BINS)],
        "box_aspect": [{"label": lbl, "count": aspect_counts.get(i, 0)} for i, lbl in enumerate(ASPECT_LABELS)],
        "heatmap": {"n": HEAT_N, "cells": heat, "max": max((max(row) for row in heat), default=0)},
        "issues": issues,
    }


def _light(dataset_id: str) -> Dict[str, Any]:
    reviews = {r["status"]: r["c"] for r in db.query(
        "SELECT r.status, COUNT(*) c FROM reviews r JOIN images i ON i.dataset_id=r.dataset_id AND i.rel_path=r.rel_path "
        "WHERE r.dataset_id=? AND r.status IS NOT NULL GROUP BY r.status", (dataset_id,))}
    flags = [{"code": r["code"], "label": FLAG_LABELS.get(r["code"], r["code"]), "count": r["c"]}
             for r in db.query("SELECT code, COUNT(*) c FROM image_flags WHERE dataset_id=? GROUP BY code ORDER BY c DESC",
                               (dataset_id,))]
    kind_code = {v: k for k, v in AI_AUDIT_KINDS.items()}
    ai_rows = db.query("SELECT kind, COUNT(*) n, COUNT(DISTINCT image_id) images FROM ai_preds WHERE dataset_id=? "
                       "AND state='pending' GROUP BY kind", (dataset_id,))
    proposals = sum(r["n"] for r in ai_rows if r["kind"] not in kind_code)
    audit = {kind_code[r["kind"]]: r["images"] for r in ai_rows if r["kind"] in kind_code}
    from .imagesq import problem_clause
    sql, params = problem_clause(dataset_id)
    problems = db.query_one(f"SELECT COUNT(*) c FROM images i WHERE i.dataset_id=? AND {sql}", [dataset_id] + params) or {"c": 0}
    return {
        # Images with a parser issue (not just "no label file") or a QA flag: exactly what Explore's issue=any shows.
        "with_problems": problems["c"],
        "reviews": {"approved": reviews.get("approved", 0), "flagged": reviews.get("flagged", 0),
                    "excluded": reviews.get("excluded", 0)},
        "flags": flags,
        "ai": {"pending_proposals": proposals, "audit": audit},
        "issue_labels": ALL_ISSUE_LABELS,
    }
