"""Dataset health statistics computed from the index."""
from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List

from .. import db
from .labels import ISSUE_LABELS

SPLIT_ORDER = "CASE i.split WHEN 'train' THEN 0 WHEN 'val' THEN 1 WHEN 'test' THEN 2 WHEN '' THEN 9 ELSE 3 END"
AREA_BINS = [
    ("< 0.1%", 0.001), ("0.1–1%", 0.01), ("1–5%", 0.05), ("5–20%", 0.2), ("> 20%", 9.0),
]
HEAT_N = 12


def dataset_stats(dataset_id: str, class_names: List[str]) -> Dict[str, Any]:
    tot = db.query_one("""
        SELECT COUNT(*) images, COALESCE(SUM(has_label),0) labeled,
               COALESCE(SUM(CASE WHEN has_label=1 AND n_boxes=0 THEN 1 ELSE 0 END),0) empty,
               COALESCE(SUM(n_boxes),0) boxes,
               COALESCE(SUM(CASE WHEN issues != '[]' THEN 1 ELSE 0 END),0) with_issues,
               COALESCE(AVG(CASE WHEN has_label=1 THEN n_boxes END),0) avg_boxes
        FROM images WHERE dataset_id=?""", (dataset_id,)) or {}
    splits = db.query(f"""
        SELECT i.split, COUNT(*) images, COALESCE(SUM(i.has_label),0) labeled, COALESCE(SUM(i.n_boxes),0) boxes
        FROM images i WHERE i.dataset_id=? GROUP BY i.split ORDER BY {SPLIT_ORDER}, i.split""", (dataset_id,))
    cls_rows = db.query("""
        SELECT b.cls, i.split, COUNT(*) boxes, COUNT(DISTINCT b.image_id) images
        FROM boxes b JOIN images i ON i.id=b.image_id WHERE b.dataset_id=? GROUP BY b.cls, i.split""", (dataset_id,))
    classes: Dict[int, dict] = {}
    for r in cls_rows:
        c = classes.setdefault(int(r["cls"]), {"cls": int(r["cls"]), "boxes": 0, "images": 0, "by_split": {}})
        c["boxes"] += r["boxes"]
        c["images"] += r["images"]
        c["by_split"][r["split"]] = r["boxes"]
    for i, name in enumerate(class_names):
        classes.setdefault(i, {"cls": i, "boxes": 0, "images": 0, "by_split": {}})
    class_list = []
    for cid in sorted(classes):
        c = classes[cid]
        c["name"] = class_names[cid] if 0 <= cid < len(class_names) else f"class_{cid}"
        c["unknown"] = not (0 <= cid < len(class_names))
        class_list.append(c)

    bpi = db.query("""SELECT MIN(n_boxes, 10) k, COUNT(*) c FROM images WHERE dataset_id=? AND has_label=1
                      GROUP BY k ORDER BY k""", (dataset_id,))
    dims = db.query("""SELECT width, height, COUNT(*) c FROM images WHERE dataset_id=? AND width IS NOT NULL
                       GROUP BY width, height ORDER BY c DESC LIMIT 8""", (dataset_id,))
    dims_total = db.query_one("SELECT COUNT(DISTINCT width || 'x' || height) n FROM images WHERE dataset_id=? AND width IS NOT NULL", (dataset_id,)) or {"n": 0}
    area = db.query("""
        SELECT CASE WHEN w*h < 0.001 THEN 0 WHEN w*h < 0.01 THEN 1 WHEN w*h < 0.05 THEN 2 WHEN w*h < 0.2 THEN 3 ELSE 4 END bin,
               COUNT(*) c FROM boxes WHERE dataset_id=? GROUP BY bin""", (dataset_id,))
    area_counts = {int(r["bin"]): r["c"] for r in area}
    heat_rows = db.query(f"""
        SELECT MIN({HEAT_N - 1}, MAX(0, CAST(xc*{HEAT_N} AS INTEGER))) gx,
               MIN({HEAT_N - 1}, MAX(0, CAST(yc*{HEAT_N} AS INTEGER))) gy, COUNT(*) c
        FROM boxes WHERE dataset_id=? GROUP BY gx, gy""", (dataset_id,))
    heat = [[0] * HEAT_N for _ in range(HEAT_N)]
    for r in heat_rows:
        heat[int(r["gy"])][int(r["gx"])] = r["c"]
    aspect = db.query("""SELECT CASE WHEN w/h < 0.5 THEN 0 WHEN w/h < 0.8 THEN 1 WHEN w/h < 1.25 THEN 2 WHEN w/h < 2 THEN 3 ELSE 4 END bin,
                         COUNT(*) c FROM boxes WHERE dataset_id=? AND h > 0 GROUP BY bin""", (dataset_id,))
    aspect_counts = {int(r["bin"]): r["c"] for r in aspect}

    issue_counter: Counter = Counter()
    for r in db.query("SELECT issues, COUNT(*) c FROM images WHERE dataset_id=? AND issues != '[]' GROUP BY issues", (dataset_id,)):
        for code in db.loads(r["issues"]):
            issue_counter[code] += r["c"]
    issues = [{"code": code, "label": ISSUE_LABELS.get(code, code), "count": n}
              for code, n in sorted(issue_counter.items(), key=lambda kv: -kv[1])]

    reviews = {r["status"]: r["c"] for r in db.query("SELECT status, COUNT(*) c FROM reviews WHERE dataset_id=? GROUP BY status", (dataset_id,))}
    seqs = db.query_one("SELECT COUNT(DISTINCT seq) n FROM images WHERE dataset_id=?", (dataset_id,)) or {"n": 0}

    return {
        "totals": {
            "images": tot.get("images", 0), "labeled": tot.get("labeled", 0),
            "unlabeled": tot.get("images", 0) - tot.get("labeled", 0), "empty": tot.get("empty", 0),
            "boxes": tot.get("boxes", 0), "with_issues": tot.get("with_issues", 0),
            "avg_boxes": round(tot.get("avg_boxes", 0) or 0, 2), "classes": len(class_list),
            "sequences": seqs.get("n", 0),
        },
        "splits": splits,
        "classes": class_list,
        "boxes_per_image": [{"k": int(r["k"]), "count": r["c"]} for r in bpi],
        "dimensions": [{"width": r["width"], "height": r["height"], "count": r["c"]} for r in dims],
        "dimensions_distinct": dims_total.get("n", 0),
        "box_area": [{"label": lbl, "count": area_counts.get(i, 0)} for i, (lbl, _) in enumerate(AREA_BINS)],
        "box_aspect": [{"label": lbl, "count": aspect_counts.get(i, 0)} for i, lbl in enumerate(["<0.5", "0.5–0.8", "≈1", "1.25–2", ">2"])],
        "heatmap": {"n": HEAT_N, "cells": heat, "max": max((max(row) for row in heat), default=0)},
        "issues": issues,
        "reviews": {"approved": reviews.get("approved", 0), "flagged": reviews.get("flagged", 0), "excluded": reviews.get("excluded", 0)},
    }
