"""Box math shared by providers, jobs and the audit (pure Python; boxes are normalised YOLO [xc, yc, w, h])."""
from __future__ import annotations

import math
from typing import Iterable, List, Optional, Sequence, Tuple

Box4 = Sequence[float]   # xc, yc, w, h (normalised)


def xyxy(b: Box4) -> Tuple[float, float, float, float]:
    return b[0] - b[2] / 2, b[1] - b[3] / 2, b[0] + b[2] / 2, b[1] + b[3] / 2


def iou4(a: Box4, b: Box4) -> float:
    ax0, ay0, ax1, ay1 = xyxy(a)
    bx0, by0, bx1, by1 = xyxy(b)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def clamp_box(xc: float, yc: float, w: float, h: float) -> Optional[Tuple[float, float, float, float]]:
    """Clip to the image; None when degenerate or not finite."""
    if not all(math.isfinite(v) for v in (xc, yc, w, h)):
        return None
    x0, y0 = max(0.0, xc - w / 2), max(0.0, yc - h / 2)
    x1, y1 = min(1.0, xc + w / 2), min(1.0, yc + h / 2)
    if x1 - x0 <= 0.0005 or y1 - y0 <= 0.0005:
        return None
    return (round((x0 + x1) / 2, 6), round((y0 + y1) / 2, 6), round(x1 - x0, 6), round(y1 - y0, 6))


def from_xyxy_px(x0: float, y0: float, x1: float, y1: float, W: float, H: float):
    if W <= 0 or H <= 0:
        return None
    x0, x1 = sorted((x0, x1))
    y0, y1 = sorted((y0, y1))
    return clamp_box((x0 + x1) / 2 / W, (y0 + y1) / 2 / H, (x1 - x0) / W, (y1 - y0) / H)


def ios4(a: Box4, b: Box4) -> float:
    """Intersection over the smaller box: 1.0 when one box lies inside the other."""
    ax0, ay0, ax1, ay1 = xyxy(a)
    bx0, by0, bx1, by1 = xyxy(b)
    iw = min(ax1, bx1) - max(ax0, bx0)
    ih = min(ay1, by1) - max(ay0, by0)
    if iw <= 0 or ih <= 0:
        return 0.0
    small = min(a[2] * a[3], b[2] * b[3])
    return iw * ih / small if small > 0 else 0.0


def nms(items: List[dict], iou_thr: float = 0.5, class_aware: bool = True, ios_thr: Optional[float] = None) -> List[dict]:
    """Greedy NMS over dicts with cls/xc/yc/w/h/conf (conf None counts as 1). With ``ios_thr`` a box mostly inside a
    better-scored one is also suppressed (merging tiled predictions: an object cut by a tile edge)."""
    order = sorted(items, key=lambda d: -(d.get("conf") if d.get("conf") is not None else 1.0))
    kept: List[dict] = []
    for d in order:
        b = (d["xc"], d["yc"], d["w"], d["h"])
        dup = False
        for k in kept:
            if class_aware and k["cls"] != d["cls"]:
                continue
            kb = (k["xc"], k["yc"], k["w"], k["h"])
            if iou4(b, kb) > iou_thr or (ios_thr is not None and ios4(b, kb) > ios_thr):
                dup = True
                break
        if not dup:
            kept.append(d)
    return kept


def split_duplicates(new: List[dict], existing: Iterable[Sequence[float]], thr: float = 0.7) -> Tuple[List[dict], List[dict]]:
    """Split proposals into (fresh, duplicates): a duplicate overlaps an existing label row [cls, xc, yc, w, h, ...]
    of the same class with IoU >= thr (finding C9: repeated Append must not stack identical boxes)."""
    ex = [(int(r[0]), (float(r[1]), float(r[2]), float(r[3]), float(r[4]))) for r in existing]
    fresh, dup = [], []
    for d in new:
        b = (d["xc"], d["yc"], d["w"], d["h"])
        if any(c == int(d["cls"]) and iou4(b, eb) >= thr for c, eb in ex):
            dup.append(d)
        else:
            fresh.append(d)
            ex.append((int(d["cls"]), b))      # also dedupes within the new set
    return fresh, dup


def tiles(W: int, H: int, tile: int = 640, overlap: float = 0.2) -> List[Tuple[int, int, int, int]]:
    """SAHI-style overlapping windows (x0, y0, x1, y1) covering a W x H image."""
    tile = max(64, int(tile))
    if W <= tile and H <= tile:
        return [(0, 0, W, H)]
    step = max(1, int(tile * (1 - overlap)))
    xs = list(range(0, max(W - tile, 0) + 1, step)) or [0]
    ys = list(range(0, max(H - tile, 0) + 1, step)) or [0]
    if xs[-1] + tile < W:
        xs.append(W - tile)
    if ys[-1] + tile < H:
        ys.append(H - tile)
    return [(x, y, min(x + tile, W), min(y + tile, H)) for y in ys for x in xs]


# ------------------------------------------------------------------ label audit (ObjectLab-style)

def is_duplicate(gg: float, gs: float, gd: float) -> bool:
    """Is a GT box the same object as another GT row whose detection it overlaps? ``gg`` / ``gs`` = IoU / IoS of the
    two GT boxes, ``gd`` = IoU of this box with that detection. Measured on vfp290k (val, yolo12m): copies of a person
    shifted by half a width overlap their twin at IoU 0.30-0.58 and its detection at 0.45-0.69, while a fallen person
    next to a standing helper overlaps the helper's detection at only 0.31-0.37."""
    return gg >= 0.5 or gs >= 0.85 or (gd >= 0.45 and (gg >= 0.3 or gs >= 0.5))

def label_issues(gt: List[Sequence[float]], preds: List[dict], *, hi: float = 0.5, cover: float = 0.3, good: float = 0.8,
                 same_family=None, others: Sequence[Sequence[float]] = ()) -> List[dict]:
    """Compare ground truth rows ``[cls, xc, yc, w, h]`` with predictions (dicts with cls/xc/yc/w/h/conf, from a LOW
    confidence floor). Returns findings sorted by suspicion:

    * ``missing``  — a confident prediction no GT box covers (box = prediction)
    * ``loose``    — a GT box whose matched same-class prediction overlaps only partly (box = suggested tighter box)
    * ``swap``     — a GT box confidently predicted as another class (box = GT, cls = suggested class)
    * ``spurious`` — a GT box no prediction supports at all, or a second label on an object another row already
      labels (box = GT)

    GT rows and same-family predictions are matched one-to-one first (greedy by IoU, as ObjectLab / COCO evaluation
    do): a detection explains at most one GT box, so two rows on one object are never both "tightened" to the same box
    — the row the detection fits best gets the ``loose`` fix, the other one is reported as a duplicate.

    Findings about a GT box carry ``ref_row`` (its row index in ``gt``) and ``ref_box`` (its geometry), so a fix can
    still be matched to the right box after the label file was edited.

    ``same_family(gt_cls, pred_cls) -> bool`` widens "same class" (geometry mode: COCO person covers fall and standing).
    Predictions without confidence count as 1.0. ``others`` are label rows of classes not being judged: a prediction
    sitting on one of them (IoU >= 0.5) is not reported as missing."""
    same = same_family or (lambda a, b: int(a) == int(b))
    P = [(d, (d["xc"], d["yc"], d["w"], d["h"]), d.get("conf") if d.get("conf") is not None else 1.0) for d in preds]
    G = [(int(r[0]), (float(r[1]), float(r[2]), float(r[3]), float(r[4]))) for r in gt]
    I = [[iou4(gb, pb) for (_d, pb, _c) in P] for (_gc, gb) in G]
    match_g: dict = {}                           # GT row -> its prediction (one-to-one)
    match_p: dict = {}                           # prediction -> its GT row
    for v, i, j in sorted(((I[i][j], i, j) for i in range(len(G)) for j in range(len(P))
                           if I[i][j] >= cover and same(G[i][0], P[j][0]["cls"])), key=lambda t: (-t[0], t[1], t[2])):
        if i not in match_g and j not in match_p:
            match_g[i], match_p[j] = j, i
    out: List[dict] = []
    for j, (d, pb, c) in enumerate(P):
        if c < hi:
            continue
        covered_same = any(same(G[i][0], d["cls"]) and I[i][j] >= cover for i in range(len(G)))
        covered_any = any(I[i][j] >= 0.5 for i in range(len(G))) or any(
            iou4((float(o[1]), float(o[2]), float(o[3]), float(o[4])), pb) >= 0.5 for o in others)
        if not covered_same and not covered_any:
            out.append({"kind": "missing", "ref_row": None, "cls": int(d["cls"]), "box": pb, "score": float(c),
                        "conf": float(c), "label": d.get("label", ""),
                        "reason": f"{d.get('label') or 'object'} found with confidence {c:.2f} but not labeled"})
    for i, (gcls, gb) in enumerate(G):
        s_idx = [j for j in range(len(P)) if same(gcls, P[j][0]["cls"])]
        o_idx = [j for j in range(len(P)) if not same(gcls, P[j][0]["cls"]) and P[j][2] >= hi]
        bs = max((I[i][j] for j in s_idx), default=0.0)        # best same-family overlap, matched elsewhere or not
        bo = max((I[i][j] for j in o_idx), default=0.0)
        if i in match_g:
            j = match_g[i]
            d, pb, c = P[j]
            m = I[i][j]
            if m < good and c >= hi:
                out.append({"kind": "loose", "ref_row": i, "ref_box": gb, "cls": gcls, "box": pb, "score": float(c * (1 - m)),
                            "conf": float(c), "label": d.get("label", ""),
                            "reason": f"box overlaps the detection only {m:.2f} IoU (confidence {c:.2f})"})
        elif o_idx and bo >= 0.5 and bs < cover:
            j = max(o_idx, key=lambda j: I[i][j])
            d, pb, c = P[j]
            out.append({"kind": "swap", "ref_row": i, "ref_box": gb, "cls": int(d["cls"]), "box": gb, "score": float(c * bo),
                        "conf": float(c), "label": d.get("label", ""),
                        "reason": f"looks like {d.get('label') or d['cls']} (confidence {c:.2f}, IoU {bo:.2f})"})
        elif bs >= cover:
            # The detections on this box already explain other rows. Overlapping such a row (and its detection) a
            # lot = the same object labeled twice: report it, so accepting every fix deletes the copy instead of
            # stacking two equal boxes. Otherwise the detector merged two objects and says nothing about this one.
            best = None
            for j in s_idx:
                if I[i][j] < cover or j not in match_p:
                    continue
                k = match_p[j]
                gg, gs = iou4(gb, G[k][1]), ios4(gb, G[k][1])
                if is_duplicate(gg, gs, I[i][j]):
                    sc = float(P[j][2] * max(gg, I[i][j]))
                    if best is None or sc > best[0]:
                        best = (sc, k, gg)
            if best is not None:
                sc, k, gg = best
                out.append({"kind": "spurious", "ref_row": i, "ref_box": gb, "cls": gcls, "box": gb, "score": sc,
                            "conf": None, "label": "", "dup_of": k, "dup_iou": round(gg, 2), "dup_cls": G[k][0],
                            "reason": f"labeled twice: the same object as another box (IoU {gg:.2f})"})
        elif bs < cover and bo < cover and max(I[i], default=0.0) < cover:
            out.append({"kind": "spurious", "ref_row": i, "ref_box": gb, "cls": gcls, "box": gb,
                        "score": 0.5 * float(1 - max(I[i], default=0.0)), "conf": None, "label": "",
                        "reason": "no detection supports this box"})
    out.sort(key=lambda f: -f["score"])
    return out


def pool_score(scores: Sequence[float], temperature: float = 0.1) -> float:
    """Soft-max pooling: dominated by the worst finding, but several medium findings add up."""
    if not scores:
        return 0.0
    m = max(scores)
    ws = [math.exp((s - m) / temperature) for s in scores]
    return round(sum(w * s for w, s in zip(ws, scores)) / sum(ws), 4)
