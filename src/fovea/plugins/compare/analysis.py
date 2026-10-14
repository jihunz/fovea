"""Threshold-dependent views of a stored evaluation: per-image error counts, the confusion matrix, the per-image
table and the live viewer detail. All of them read the packed arrays (no label files) except ``image_detail``."""
from __future__ import annotations

import threading
from collections import OrderedDict
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

from ... import db
from .evaluate import GRID, ClassView, apply_remap, evaluate_image, iou_matrix, pred_class_fn, read_pred_file

_lock = threading.Lock()
_counts: "OrderedDict[tuple, tuple]" = OrderedDict()


def grid_k(conf) -> int:
    try:
        v = float(conf)
    except (TypeError, ValueError):
        return 25
    if v != v:                                   # NaN (a JSON body or ?conf=nan): the default, not a 500
        return 25
    return max(0, min(GRID - 1, int(round(max(-1.0, min(2.0, v)) * 100))))


def parse_classes(spec) -> Optional[FrozenSet[int]]:
    """'0,1' -> {0, 1}; '' / None -> None (all classes)."""
    if spec in (None, "", "all"):
        return None
    out = set()
    for tok in str(spec).split(","):
        tok = tok.strip()
        if tok.lstrip("-").isascii() and tok.lstrip("-").isdigit():     # '²' is a digit to isdigit(), not to int()
            out.add(int(tok))
    return frozenset(out)


def image_counts(eid: str, data: dict, side: str, k: int, classes: Optional[FrozenSet[int]]):
    """Per image (tp, fp, fn, swaps) arrays at threshold index k for the given classes."""
    key = (eid, side, k, classes)
    with _lock:
        if key in _counts:
            _counts.move_to_end(key)
            return _counts[key]
    n = data["n"]
    sd = data["sides"][side]
    pmask = 1 << data["primary"]
    tp = [0] * n; fp = [0] * n; fn = [0] * n; sw = [0] * n
    p_img, p_cls, p_lvl, p_bits = sd["p_img"], sd["p_cls"], sd["p_lvl"], sd["p_bits"]
    for i in range(len(p_img)):
        if p_lvl[i] >= k and (classes is None or p_cls[i] in classes):
            if p_bits[i] & pmask:
                tp[p_img[i]] += 1
            else:
                fp[p_img[i]] += 1
    g_img, g_cls, g_lvl = data["g_img"], data["g_cls"], sd["g_lvl"]
    for g in range(len(g_img)):
        if g_lvl[g] < k and (classes is None or g_cls[g] in classes):
            fn[g_img[g]] += 1
    s_img, s_lo, s_hi, s_g, s_p, s_n = sd["s_img"], sd["s_lo"], sd["s_hi"], sd["s_g"], sd["s_p"], sd["s_n"]
    for s in range(len(s_img)):
        if s_lo[s] <= k <= s_hi[s] and (classes is None or (s_g[s] in classes and s_p[s] in classes)):
            sw[s_img[s]] += s_n[s]
    res = (tp, fp, fn, sw)
    with _lock:
        _counts[key] = res
        _counts.move_to_end(key)
        while len(_counts) > 24:
            _counts.popitem(last=False)
    return res


def forget(eid: str) -> None:
    with _lock:
        for key in [key for key in _counts if key[0] == eid]:
            _counts.pop(key, None)


def confusion(data: dict, side: str, k: int, classes: Optional[FrozenSet[int]]) -> dict:
    """Rows = ground-truth class (+ 'background' for false positives on nothing), columns = predicted class
    (+ 'missed' for boxes no prediction reached). The diagonal equals TP at this threshold."""
    sd = data["sides"][side]
    pmask = 1 << data["primary"]
    seen = set(int(c) for c in data["g_cls"]) | set(int(c) for c in sd["p_cls"])
    labels = sorted(c for c in seen if classes is None or c in classes)
    pos = {c: i for i, c in enumerate(labels)}
    n = len(labels)
    M = [[0] * (n + 1) for _ in range(n + 1)]       # [gt or bg][pred or miss]
    tp = [0] * n; fpc = [0] * n; fnc = [0] * n
    for i in range(len(sd["p_img"])):
        c = sd["p_cls"][i]
        if sd["p_lvl"][i] < k or c not in pos:
            continue
        if sd["p_bits"][i] & pmask:
            tp[pos[c]] += 1
        else:
            fpc[pos[c]] += 1
    for g in range(len(data["g_img"])):
        c = data["g_cls"][g]
        if c in pos and sd["g_lvl"][g] < k:
            fnc[pos[c]] += 1
    swap_by_pred = [0] * n; swap_by_gt = [0] * n
    for s in range(len(sd["s_img"])):
        if sd["s_lo"][s] <= k <= sd["s_hi"][s] and sd["s_g"][s] in pos and sd["s_p"][s] in pos:
            gi, pi = pos[sd["s_g"][s]], pos[sd["s_p"][s]]
            M[gi][pi] += sd["s_n"][s]
            swap_by_pred[pi] += sd["s_n"][s]
            swap_by_gt[gi] += sd["s_n"][s]
    for i in range(n):
        M[i][i] = tp[i]
        M[n][i] = fpc[i] - swap_by_pred[i]
        M[i][n] = fnc[i] - swap_by_gt[i]
    return {"k": k, "conf": k / 100, "labels": labels, "matrix": M,
            "gt_totals": [tp[i] + fnc[i] for i in range(n)], "pred_totals": [tp[i] + fpc[i] for i in range(n)]}


def cell_images(data: dict, side: str, k: int, classes, cell: str) -> List[int]:
    """Image indexes contributing to one confusion cell 'g:p' (p may be 'miss', g may be 'bg')."""
    try:
        g_s, p_s = cell.split(":", 1)
    except ValueError:
        return []
    sd = data["sides"][side]
    pmask = 1 << data["primary"]
    hits = set()
    if g_s == "bg" or p_s == "miss":
        c = int(p_s) if g_s == "bg" else int(g_s)
        per: Dict[int, int] = {}
        if g_s == "bg":
            for i in range(len(sd["p_img"])):
                if sd["p_cls"][i] == c and sd["p_lvl"][i] >= k and not sd["p_bits"][i] & pmask:
                    per[sd["p_img"][i]] = per.get(sd["p_img"][i], 0) + 1
        else:
            for g in range(len(data["g_img"])):
                if data["g_cls"][g] == c and sd["g_lvl"][g] < k:
                    per[data["g_img"][g]] = per.get(data["g_img"][g], 0) + 1
        for s in range(len(sd["s_img"])):
            if not (sd["s_lo"][s] <= k <= sd["s_hi"][s]):
                continue
            if classes is not None and not (sd["s_g"][s] in classes and sd["s_p"][s] in classes):
                continue
            if (g_s == "bg" and sd["s_p"][s] == c) or (p_s == "miss" and sd["s_g"][s] == c):
                per[sd["s_img"][s]] = per.get(sd["s_img"][s], 0) - sd["s_n"][s]
        return sorted(i for i, v in per.items() if v > 0)
    g, p = int(g_s), int(p_s)
    if g == p:
        for i in range(len(sd["p_img"])):
            if sd["p_cls"][i] == g and sd["p_lvl"][i] >= k and sd["p_bits"][i] & pmask:
                hits.add(sd["p_img"][i])
    else:
        for s in range(len(sd["s_img"])):
            if sd["s_g"][s] == g and sd["s_p"][s] == p and sd["s_lo"][s] <= k <= sd["s_hi"][s]:
                hits.add(sd["s_img"][s])
    return sorted(hits)


SHOWS = ("errors", "all", "fn", "fp", "swap", "clean", "regress", "improve", "cell")
SORTS = ("errors", "fn", "fp", "swap", "delta", "gt", "name")


def table(eid: str, data: dict, k: int, classes, *, show: str = "errors", sort: str = "errors", side: str = "a",
          q: str = "", split: str = "", cell: str = "") -> Tuple[List[int], Dict[str, tuple]]:
    """Image indexes in display order for the per-image table, plus the per-side counts used."""
    sides = list(data["sides"])
    side = side if side in sides else sides[0]
    counts = {s: image_counts(eid, data, s, k, classes) for s in sides}
    n = data["n"]
    gt_n = _gt_counts(data, classes)

    def err(s, i):
        tp, fp, fn, sw = counts[s]
        return fp[i] + fn[i]

    idx: Iterable[int] = range(n)
    if show == "errors":
        idx = [i for i in idx if any(err(s, i) for s in sides)]
    elif show == "clean":
        idx = [i for i in idx if not any(err(s, i) for s in sides)]
    elif show == "fn":
        idx = [i for i in idx if counts[side][2][i]]
    elif show == "fp":
        idx = [i for i in idx if counts[side][1][i]]
    elif show == "swap":
        idx = [i for i in idx if counts[side][3][i]]
    elif show in ("regress", "improve") and "b" in counts:
        sign = 1 if show == "regress" else -1
        idx = [i for i in idx if sign * (err("b", i) - err("a", i)) > 0]
    elif show == "cell" and cell:
        idx = cell_images(data, side, k, classes, cell)
    else:
        idx = list(idx)
    if split:
        try:
            si = data["splits"].index(split)
            idx = [i for i in idx if data["split"][i] == si]
        except ValueError:
            idx = []
    if q:
        ql = q.lower()
        idx = [i for i in idx if ql in data["rel"][i].lower()]
    idx = list(idx)
    if sort == "name":
        idx.sort(key=lambda i: data["rel"][i])
    elif sort == "gt":
        idx.sort(key=lambda i: (-gt_n[i], i))
    elif sort == "delta" and "b" in counts:
        idx.sort(key=lambda i: (-(err("b", i) - err("a", i)), -err("b", i), i))
    elif sort in ("fn", "fp", "swap"):
        col = {"fp": 1, "fn": 2, "swap": 3}[sort]
        idx.sort(key=lambda i: (-counts[side][col][i], -err(side, i), i))
    else:
        idx.sort(key=lambda i: (-sum(err(s, i) for s in sides), -sum(counts[s][2][i] for s in sides), i))
    return idx, counts


def _gt_counts(data: dict, classes) -> List[int]:
    out = [0] * data["n"]
    for g in range(len(data["g_img"])):
        if classes is None or data["g_cls"][g] in classes:
            out[data["g_img"][g]] += 1
    return out


def row_items(data: dict, idx: Sequence[int], counts: Dict[str, tuple], classes) -> List[dict]:
    gt_n = _gt_counts(data, classes) if idx else []
    items = []
    for i in idx:
        it = {"i": i, "id": int(data["img"][i]), "rel_path": data["rel"][i], "split": data["splits"][data["split"][i]],
              "gt": gt_n[i]}
        for s, (tp, fp, fn, sw) in counts.items():
            it[s] = {"tp": tp[i], "fp": fp[i], "fn": fn[i], "sw": sw[i]}
        items.append(it)
    return items


def attach_reviews(ds_id: str, items: List[dict]) -> None:
    rels = [it["rel_path"] for it in items]
    found: Dict[str, dict] = {}
    for c in range(0, len(rels), 500):
        chunk = rels[c:c + 500]
        for r in db.query("SELECT rel_path, status, note FROM reviews WHERE dataset_id=? AND rel_path IN (%s)" % ",".join("?" * len(chunk)), [ds_id, *chunk]):
            found[r["rel_path"]] = {"status": r["status"], "note": r["note"] or ""}
    for it in items:
        it["review"] = found.get(it["rel_path"])


def stored_counts(eid: str, data: dict, image_id: int, k: int, classes) -> Optional[Dict[str, dict]]:
    i = data["index"].get(int(image_id))
    if i is None:
        return None
    out = {}
    for s in data["sides"]:
        tp, fp, fn, sw = image_counts(eid, data, s, k, classes)
        out[s] = {"tp": tp[i], "fp": fp[i], "fn": fn[i], "sw": sw[i]}
    return out


# ------------------------------------------------------------------ live viewer detail

def live_preds(meta: dict, side: str, image_id: int, file_rel: str) -> Tuple[List[list], dict]:
    """Current predictions of one side for one image, re-read now (the file may have changed or moved)."""
    src = meta["sources"][side]
    info = {"file": "", "file_missing": False}
    if src["type"] == "run":
        run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (int(src["run_id"]),))
        to_cls = pred_class_fn(run) if run else (lambda c, label=None: int(c))
        rows = db.query("SELECT cls, xc, yc, w, h, conf, label FROM ai_preds WHERE run_id=? AND image_id=? AND kind='detect' ORDER BY id",
                        (int(src["run_id"]), int(image_id)))
        preds = [[to_cls(r["cls"], r["label"]), r["xc"], r["yc"], r["w"], r["h"], r["conf"]] for r in rows]
        if not run:
            info["file_missing"] = True                  # the run was deleted since the evaluation
    else:
        root = Path(src["path"])
        if file_rel:
            f = root / file_rel
            info["file"] = str(f)
            info["file_host"] = src.get("path_host", src["path"]).rstrip("/") + "/" + file_rel
            if not f.is_file():
                info["file_missing"] = True
                preds = []
            else:
                preds = read_pred_file(f)
        else:
            preds = []
    remap = {int(a): int(b) for a, b in ((meta.get("remap") or {}).get(side) or {}).items()}
    return ClassView.from_meta(meta).rows(apply_remap(preds, remap)), info


def image_detail(ds_id: str, eid: str, meta: dict, data: Optional[dict], image: dict, k: int, classes,
                 files: Optional[Dict[str, str]] = None) -> dict:
    raw_gt = [[int(r["cls"]), r["xc"], r["yc"], r["w"], r["h"]] for r in
              db.query("SELECT cls, xc, yc, w, h FROM boxes WHERE image_id=? ORDER BY id", (image["id"],))]
    cv = ClassView.from_meta(meta)
    gt = cv.rows(raw_gt)
    ious = meta["ious"]
    primary = meta["primary"]
    t = ious[primary]
    idx = data["index"].get(int(image["id"])) if data else None
    stored = stored_counts(eid, data, image["id"], k, classes) if data is not None else None
    # boxes of classes this evaluation leaves out: drawn as context, never counted
    out = {"gt": gt, "gt_out": [g for g in raw_gt if cv.cls(g[0]) is None], "k": k, "sides": {}, "in_eval": idx is not None}
    for side in meta["sources"]:
        if not meta["sources"][side]:
            continue
        if files is not None:
            file_rel = files.get(side, "")
        else:
            file_rel = data["sides"][side]["files"][idx] if (data is not None and idx is not None) else ""
        preds, info = live_preds(meta, side, image["id"], file_rel)
        r = evaluate_image(gt, preds, ious, primary, meta.get("metric", "coco"), image.get("width"), image.get("height"))
        order = sorted(range(len(preds)), key=lambda i: -(preds[i][5] if preds[i][5] is not None else 1.0))
        P = [preds[i] for i in order]
        active = [r.lvl[p] >= k for p in range(len(P))]
        tp_p = [bool(r.bits[p] & (1 << primary)) for p in range(len(P))]
        claimed = {r.gt_of[p] for p in range(len(P)) if active[p] and tp_p[p]}
        # class swaps at this threshold: remaining active predictions take an unclaimed box of another class
        M = iou_matrix(P, gt) if P and gt else [[] for _ in P]
        swap_gt = [-1] * len(P)
        for p in range(len(P)):
            if not active[p] or tp_p[p]:
                continue
            best, best_iou = -1, t
            for g in range(len(gt)):
                if g in claimed or gt[g][0] == P[p][0]:
                    continue
                if M[p][g] >= best_iou and (best < 0 or M[p][g] > M[p][best]):
                    best, best_iou = g, M[p][g]
            if best >= 0:
                claimed.add(best)
                swap_gt[p] = best
        gt_state = ["fn"] * len(gt)
        for p in range(len(P)):
            if active[p] and tp_p[p] and r.gt_of[p] >= 0:
                gt_state[r.gt_of[p]] = "tp"
            elif swap_gt[p] >= 0:
                gt_state[swap_gt[p]] = "swap"
        def want(c):
            return classes is None or int(c) in classes
        cnt = {"tp": sum(1 for p in range(len(P)) if active[p] and tp_p[p] and want(P[p][0])),
               "fp": sum(1 for p in range(len(P)) if active[p] and not tp_p[p] and want(P[p][0])),
               "fn": sum(1 for g in range(len(gt)) if gt_state[g] != "tp" and want(gt[g][0])),
               "sw": sum(1 for p in range(len(P)) if swap_gt[p] >= 0 and want(P[p][0]) and want(gt[swap_gt[p]][0]))}
        st = stored.get(side) if stored else None
        out["sides"][side] = {
            **info,
            "preds": [{"box": [int(P[p][0]), P[p][1], P[p][2], P[p][3], P[p][4]], "conf": P[p][5], "active": active[p],
                       "tp": tp_p[p], "gt": r.gt_of[p] if tp_p[p] else -1, "swap_gt": swap_gt[p], "iou": round(max(M[p]) if M[p] else 0.0, 3)}
                      for p in range(len(P))],
            "gt_state": gt_state, "counts": cnt, "stored": st,
            "changed": bool(st is not None and (st["tp"], st["fp"], st["fn"]) != (cnt["tp"], cnt["fp"], cnt["fn"])) or info.get("file_missing", False),
        }
    return out


def note_line(eid: str, meta: dict, k: int, per_side: Dict[str, dict]) -> str:
    parts = []
    for side, c in per_side.items():
        bits = [f"{c[x]} {lbl}" for x, lbl in (("fn", "FN"), ("fp", "FP"), ("sw", "swapped")) if c.get(x)]
        if bits:
            parts.append(f"{side.upper()} " + ", ".join(bits))
    return f"Compare #{eid}: " + ("; ".join(parts) or "no errors") + f" (conf ≥ {k / 100:.2f}, IoU {meta['ious'][meta['primary']]:g})"


def strip_line(note: str, eid: str) -> str:
    return "\n".join(l for l in (note or "").split("\n") if not l.startswith(f"Compare #{eid}:")).strip()


def with_line(note: str, eid: str, line: str) -> str:
    base = strip_line(note, eid)
    return (base + "\n" + line).strip() if base else line
