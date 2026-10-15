"""Ground truth vs prediction evaluation for the Compare plugin.

How the numbers are defined (the UI repeats this under "How metrics are computed"):

* Matching runs ONCE per image over every prediction, at IoU 0.50:0.05:0.95 plus the chosen IoU.
  ``coco`` / ``voc``: predictions in descending confidence each claim the unclaimed ground-truth box of the same
  class with the highest IoU >= t (pycocotools). ``ultralytics``: every detection proposes its best-IoU box and
  each box keeps the highest-ranked detection proposing it (``DetectionValidator.match_predictions``), so mAP equals
  ``yolo val`` on the same predictions.
* AP: area under the precision envelope. ``coco`` = 101 recall points; ``ultralytics`` = its trapezoid over a
  101-point interpolation (1.0 at recall 0, so a perfect class scores 0.995); ``voc`` = all-point. A class without
  ground truth has no AP (its predictions still count as false positives). mAP = mean over the evaluated classes.
* Operating point: TP/FP/FN, P/R/F1, the confusion matrix, per-image errors and flags keep the predictions with
  conf >= t out of that one matching. For greedy matching this is exactly what matching only those predictions gives.
  Thresholds live on a 0.01 grid (k = 0..100, active iff ``floor(conf*100) >= k``) so the UI moves t without re-running.
* Confusion matrix: after same-class matching, each remaining prediction (by confidence) claims the remaining box of
  another class with the highest IoU >= t -- a class swap. Everything else is a background false positive or a miss.
"""
from __future__ import annotations

import array
import base64
import math
import os
import unicodedata
import zlib
from bisect import bisect_left, bisect_right
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from ... import db
from ...config import LABEL_EXT
from ...jobs import Job

IOU_GRID: Tuple[float, ...] = tuple(round(0.5 + 0.05 * i, 2) for i in range(10))
GRID = 101                                   # confidence thresholds 0.00, 0.01, ... 1.00
METRICS = ("coco", "ultralytics", "voc")
METRIC_LABELS = {
    "coco": "COCO: greedy by confidence, 101-point AP",
    "ultralytics": "Ultralytics: IoU-ranked matching, 101-point AP (same as yolo val)",
    "voc": "VOC: greedy by confidence, all-point AP",
}
SIZE_BUCKETS = (("S", 0.0, 32.0 ** 2), ("M", 32.0 ** 2, 96.0 ** 2), ("L", 96.0 ** 2, float("inf")))
PRED_KINDS = ("detect",)                     # ai_preds kinds that are a model's raw output over the whole image
DETECT_KINDS = ("detect", "prev_frame", "interp", "refine")     # what a later proposal run replaces (ai.store)
AUDIT_KINDS = ("missing", "loose", "swap", "spurious")          # audit findings (not predictions)
PSEUDO_BASE = 100_000                        # = ai.pipeline.PSEUDO_BASE: detector classes standing for a class family
SPLIT_WORDS = {"train", "val", "valid", "validation", "test", "eval", "training", "testing"}


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def conf_level(conf: Optional[float]) -> int:
    """Grid index of the highest threshold this confidence still passes (conf >= k/100)."""
    if conf is None:
        return 100
    return max(0, min(100, int(math.floor(conf * 100 + 1e-6))))


def size_bucket(area_px: float) -> int:
    for i, (_, lo, hi) in enumerate(SIZE_BUCKETS):
        if lo <= area_px < hi:
            return i
    return len(SIZE_BUCKETS) - 1


# ------------------------------------------------------------------ prediction files

def _is_pose(vals: Sequence[float]) -> bool:
    # backend-data hook: pose-SHAPED is not enough — a polygon touching two image borders has 0/1 in every
    # visibility slot too; core.labels.looks_like_pose also wants the keypoints inside the row's own box.
    from ...core.labels import looks_like_pose
    return looks_like_pose(vals)


def parse_pred_text(text: str) -> List[list]:
    """Rows ``[cls, xc, yc, w, h, conf|None]`` from YOLO prediction text. Accepts boxes (5 or 6 columns), and
    segment / OBB polygons and pose rows (as their bounding box), each with an optional trailing confidence."""
    out: List[list] = []
    if text.startswith("﻿"):
        text = text[1:]
    for line in text.splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        try:
            c = float(parts[0])
            vals = [float(x) for x in parts[1:]]
        except ValueError:
            continue
        if not math.isfinite(c) or c < 0 or c != int(c) or not all(math.isfinite(v) for v in vals):
            continue
        conf: Optional[float] = None
        if len(vals) <= 5:
            xc, yc, w, h = vals[:4]
            if len(vals) == 5:
                conf = vals[4]
        elif _is_pose(vals):
            xc, yc, w, h = vals[:4]
        elif _is_pose(vals[:-1]):
            xc, yc, w, h = vals[:4]
            conf = vals[-1]
        else:
            coords = vals
            if len(coords) % 2 == 1:          # Ultralytics writes the confidence after the polygon
                conf, coords = coords[-1], coords[:-1]
            xs, ys = coords[0::2], coords[1::2]
            x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
            xc, yc, w, h = (x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0
        if w <= 0 or h <= 0:
            continue
        out.append([int(c), xc, yc, w, h, conf])
    return out


def read_pred_file(path: Path) -> List[list]:
    try:
        return parse_pred_text(Path(path).read_bytes().decode("utf-8", errors="replace"))
    except OSError:
        return []


class PredFolder:
    """A folder of YOLO prediction txt files, indexed by ONE directory walk. Lookups are dictionary hits: no stat
    call per image, which matters on network mounts. Keys are NFC so a Hangul folder name written by macOS (NFD)
    still matches the index."""

    def __init__(self, root: Path, max_files: int = 3_000_000):
        self.root = Path(root)
        self.by_key: Dict[str, str] = {}
        self.by_stem: Dict[str, List[str]] = defaultdict(list)
        self.truncated = False
        stack = [""]
        while stack:
            rel_dir = stack.pop()
            try:
                with os.scandir(self.root / rel_dir if rel_dir else self.root) as it:
                    entries = list(it)
            except OSError:
                continue
            for e in entries:
                name = e.name
                if name.startswith("."):
                    continue
                rel = f"{rel_dir}/{name}" if rel_dir else name
                try:
                    if e.is_dir():
                        stack.append(rel)
                        continue
                except OSError:
                    continue
                if not name.lower().endswith(LABEL_EXT) or name.lower() == "classes.txt":
                    continue
                key = nfc(rel[: -len(LABEL_EXT)])
                self.by_key[key] = rel
                self.by_stem[key.rsplit("/", 1)[-1]].append(rel)
                if len(self.by_key) >= max_files:
                    self.truncated = True
                    return

    @property
    def files(self) -> int:
        return len(self.by_key)

    def keys_for(self, rel_path: str, split: str) -> List[Tuple[str, bool]]:
        """Candidate keys in priority order, each with ``exact``: True when the key keeps the whole relative path
        (so no other image can produce it), False for the split-stripped key, which a flat folder shares between
        'train/x.jpg' and 'val/x.jpg'."""
        base = nfc(rel_path).rsplit(".", 1)[0] if "." in rel_path.rsplit("/", 1)[-1] else nfc(rel_path)
        split = nfc(split or "")
        keys = [(base, True)]
        if split and base.startswith(split + "/"):
            keys.append((base[len(split) + 1:], False))      # the split folder itself was chosen
        parts = base.split("/")
        if "images" in parts[:-1]:                             # images/<split>/x  or  <split>/images/x
            i = len(parts) - 2 - parts[:-1][::-1].index("images")
            keys.append(("/".join(parts[:i] + ["labels"] + parts[i + 1:]), True))
            keys.append(("/".join(parts[:i] + parts[i + 1:]), True))
        keys.append(("labels/" + base, True))                  # a run or dataset folder with labels/<split>/x
        if split and not base.startswith(split + "/"):
            keys.append((f"{split}/{base}", True))
            keys.append((f"labels/{split}/{base}", True))
        return keys

    def lookup(self, rel_path: str, split: str, stem_images: Dict[str, int]) -> Tuple[Optional[str], str]:
        """(relative file, how) with how in 'path' (whole relative path), 'split' (the split folder was chosen),
        'stem' (file name only), 'none' or 'ambiguous'. Callers must still reject a file that several images reach
        through a 'split' or 'stem' match (see FolderSource.prepare)."""
        for k, exact in self.keys_for(rel_path, split):
            hit = self.by_key.get(k)
            if hit is not None:
                return hit, "path" if exact else "split"
        stem = nfc(rel_path).rsplit("/", 1)[-1]
        stem = stem.rsplit(".", 1)[0] if "." in stem else stem
        files = self.by_stem.get(stem)
        if not files:
            return None, "none"
        split_n = nfc(split or "")
        in_split = [f for f in files if split_n and split_n in nfc(f).split("/")[:-1]]
        if len(files) == 1 and stem_images.get(stem, 0) <= 1:
            return files[0], "stem"
        if len(in_split) == 1:
            return in_split[0], "stem"
        return None, "ambiguous"

    @staticmethod
    def resolve_claims(matched: List[Optional[str]], how: List[str]) -> None:
        """A file reached by more than one image through a lossy match ('split' / 'stem') belongs to none of them:
        a flat folder cannot say whether x.txt is the prediction for train/x.jpg or val/x.jpg."""
        claims: Dict[str, List[int]] = defaultdict(list)
        for i, f in enumerate(matched):
            if f is not None:
                claims[f].append(i)
        for idxs in claims.values():
            if len(idxs) > 1:
                for i in idxs:
                    if how[i] in ("split", "stem"):
                        matched[i], how[i] = None, "ambiguous"


# ------------------------------------------------------------------ prediction sources

class Source:
    """Where one model's predictions come from. ``covers(image_id)`` is False for images this source says nothing
    about (an AI run that did not process the image, or a folder without a file when missing='skip')."""
    kind = "folder"
    label = ""
    has_files = False

    def prepare(self, items: List[dict]) -> None: ...
    def covers(self, idx: int) -> bool: return True
    def ambiguous(self, idx: int) -> bool: return False
    def preds(self, idx: int) -> List[list]: return []
    def file(self, idx: int) -> str: return ""


class FolderSource(Source):
    kind = "folder"
    has_files = True

    def __init__(self, root: Path, missing: str = "empty", folder: Optional[PredFolder] = None):
        self.root = Path(root)
        self.folder = folder or PredFolder(self.root)
        self.missing = missing
        self.label = folder_label(self.root)
        self.matched: List[Optional[str]] = []
        self.how: List[str] = []

    def prepare(self, items: List[dict]) -> None:
        stems: Dict[str, int] = defaultdict(int)
        for it in items:
            s = nfc(it["rel_path"]).rsplit("/", 1)[-1]
            stems[s.rsplit(".", 1)[0] if "." in s else s] += 1
        self.matched, self.how = [], []
        for it in items:
            f, how = self.folder.lookup(it["rel_path"], it["split"], stems)
            self.matched.append(f)
            self.how.append(how)
        PredFolder.resolve_claims(self.matched, self.how)

    def covers(self, idx: int) -> bool:
        return self.matched[idx] is not None or self.missing != "skip"

    def ambiguous(self, idx: int) -> bool:
        return self.how[idx] == "ambiguous"

    def preds(self, idx: int) -> List[list]:
        f = self.matched[idx]
        return read_pred_file(self.root / f) if f else []

    def file(self, idx: int) -> str:
        return self.matched[idx] or ""


def run_info(dataset_id: str, run_id: int) -> Optional[dict]:
    run = db.query_one("SELECT * FROM ai_runs WHERE id=? AND dataset_id=?", (run_id, dataset_id))
    if not run:
        return None
    marks = ",".join("?" * len(PRED_KINDS))
    st = db.query_one(f"""SELECT COUNT(*) n, COUNT(DISTINCT image_id) imgs, SUM(conf IS NULL) noconf,
                              SUM(state='accepted') accepted, MIN(conf) min_conf FROM ai_preds WHERE run_id=? AND kind IN ({marks})""",
                      (run_id, *PRED_KINDS)) or {}
    params = db.loads(run.get("params"), {})
    processed, replaced = processed_images(run)
    out = {"id": run["id"], "kind": run["kind"], "provider": run["provider"], "model": run["model"],
           "status": run["status"], "created_at": run["created_at"], "finished_at": run.get("finished_at"),
           "images": run["images"], "preds": int(st.get("n") or 0), "images_with_preds": int(st.get("imgs") or 0),
           "preds_without_conf": int(st.get("noconf") or 0), "accepted": int(st.get("accepted") or 0),
           "min_conf": st.get("min_conf"),
           "processed_known": processed is not None, "processed": len(processed) if processed is not None else None,
           "replaced": len(replaced), "label": run_label(run), "evaluable": int(st.get("n") or 0) > 0}
    audit = audit_info(run, params)
    if audit:
        out["audit"] = audit
        if not out["evaluable"]:
            out["reason"] = ("This audit kept only its findings, not the model's detections (audits made by an earlier "
                             "version did). Run the audit again to evaluate the model.")
    return out


ACCEPT_IOU = 0.7                             # = ai.geometry.split_duplicates: an accept over such a box writes nothing
ALREADY_LABELED = "already labeled"          # ai_preds.reason of a prediction an accept found already in the labels


def accepted_in_labels(run_id: int, image_ids: Iterable[int]) -> int:
    """How many of a run's accepted predictions on these images are in the labels NOW: a label box of the same class
    at IoU >= 0.7, each box standing for one prediction. A box removed since (a delete, a History restore of an older
    version — restoring the accept's own version also puts its proposals back to pending) no longer makes the ground
    truth optimistic, and an accept that found the box already there wrote nothing of the model's."""
    want = {int(i) for i in image_ids}
    if not want:
        return 0
    marks = ",".join("?" * len(PRED_KINDS))
    by_img: Dict[int, List[dict]] = defaultdict(list)
    for r in db.query(f"""SELECT image_id, cls, xc, yc, w, h, conf FROM ai_preds WHERE run_id=? AND state='accepted'
                          AND reason != ? AND kind IN ({marks})""", (int(run_id), ALREADY_LABELED, *PRED_KINDS)):
        if r["image_id"] in want:
            by_img[r["image_id"]].append(r)
    if not by_img:
        return 0
    gt = gt_for(sorted(by_img))
    n = 0
    for iid, rows in by_img.items():
        boxes = gt.get(iid) or []
        if not boxes:
            continue
        rows.sort(key=lambda r: -(r["conf"] if r["conf"] is not None else 1.0))
        preds = [[int(r["cls"]), r["xc"], r["yc"], r["w"], r["h"]] for r in rows]
        M = iou_matrix(preds, boxes)
        taken = [False] * len(boxes)
        for pi, p in enumerate(preds):
            best, hit = ACCEPT_IOU, -1
            for j, g in enumerate(boxes):
                if not taken[j] and int(g[0]) == p[0] and M[pi][j] >= best:
                    best, hit = M[pi][j], j
            if hit >= 0:
                taken[hit] = True
                n += 1
    return n


def audit_info(run: dict, params: Optional[dict] = None) -> Optional[dict]:
    """What an audit run's raw detections mean: in geometry mode the detector found a class FAMILY (e.g. COCO
    'person' standing for dataset classes 0 and 1), so it can only be judged at the family level."""
    if run.get("kind") != "audit":
        return None
    params = params if params is not None else db.loads(run.get("params"), {})
    params = params if isinstance(params, dict) else {}
    family: Dict[str, List[int]] = {}
    for name, cls in (params.get("class_family") or {}).items():
        try:
            ids = sorted({int(c) for c in cls})
        except (TypeError, ValueError):
            continue
        if ids:
            family[str(name)] = ids
    mode = params.get("mode") or ("geometry" if family else "full")
    members = [f"{m.get('provider') or ''} · {str(m.get('model') or '').rsplit('/', 1)[-1]}".strip(" ·")
               for m in (params.get("members") or []) if isinstance(m, dict)]
    fixes = db.query_one(f"SELECT COUNT(*) n FROM ai_preds WHERE run_id=? AND state='accepted' AND kind IN ({','.join('?' * len(AUDIT_KINDS))})",
                         (run["id"], *AUDIT_KINDS))
    return {"mode": mode, "family": family if mode == "geometry" else {}, "members": members,
            "conf_floor": params.get("conf_floor"), "fixes_applied": int((fixes or {}).get("n") or 0)}


def processed_images(run: dict) -> Tuple[Optional[set], set]:
    """(images the run processed or None when unknown, images whose stored predictions a later run replaced).

    Recorded ranges / ids win. Otherwise every image the run left a row for (any kind, plus the audit scores it
    still owns) is the answer when that adds up to the run's own image counter -- each processed image then has a
    row, so none is missing. A proposals batch loses an image's pending predictions when a later proposal run
    touches the same image; those images would read as 'found nothing' and are reported instead."""
    params, result = db.loads(run.get("params"), {}), db.loads(run.get("result"), {})
    params = params if isinstance(params, dict) else {}
    ids = _processed_ids(params, result if isinstance(result, dict) else {})
    rid = int(run["id"])
    if ids is None and run.get("images"):
        have = {r["image_id"] for r in db.query("SELECT DISTINCT image_id FROM ai_preds WHERE run_id=?", (rid,))}
        if run.get("kind") == "audit":
            have |= {r["image_id"] for r in db.query("SELECT image_id FROM ai_audit WHERE run_id=?", (rid,))}
        if len(have) == int(run["images"]):
            ids = have
    replaced: set = set()
    if ids and run.get("kind") == "batch" and params.get("target", "proposals") == "proposals":
        marks = ",".join("?" * len(DETECT_KINDS))
        st = db.query_one(f"SELECT COUNT(*) n FROM ai_preds WHERE run_id=? AND kind IN ({marks})", (rid, *DETECT_KINDS))
        if st and int(run.get("boxes") or 0) > int(st["n"] or 0):
            mine = {r["image_id"] for r in db.query(f"SELECT DISTINCT image_id FROM ai_preds WHERE run_id=? AND kind IN ({marks})", (rid, *DETECT_KINDS))}
            later = {r["image_id"] for r in db.query(f"SELECT DISTINCT image_id FROM ai_preds WHERE dataset_id=? AND run_id>? AND kind IN ({marks})",
                                                     (run["dataset_id"], rid, *DETECT_KINDS))}
            replaced = (ids - mine) & later
    return ids, replaced


def _processed_ids(params: dict, result: dict) -> Optional[set]:
    """Images a run actually processed (including those where it found nothing), when the run recorded them:
    ``image_ranges`` [[first_id, last_id], ...] (compact, written by Compare's model runs) or ``image_ids``."""
    for src in (result, params):
        if not isinstance(src, dict):
            continue
        rng = src.get("image_ranges")
        if isinstance(rng, list) and rng:
            try:
                out = set()
                for a, b in rng:
                    out.update(range(int(a), int(b) + 1))
                return out
            except (TypeError, ValueError):
                return None
        ids = src.get("image_ids")
        if isinstance(ids, list) and ids and len(ids) != 200:      # ai-core truncates long param lists to 200
            try:
                return {int(x) for x in ids}
            except (TypeError, ValueError):
                return None
    return None


def id_ranges(ids: Iterable[int]) -> List[List[int]]:
    out: List[List[int]] = []
    for i in sorted(set(int(x) for x in ids)):
        if out and i == out[-1][1] + 1:
            out[-1][1] = i
        else:
            out.append([i, i])
    return out


def run_label(run: dict) -> str:
    model = run.get("model") or ""
    model = model.rsplit("/", 1)[-1]
    who = " · ".join(x for x in (run.get("provider") or "", model) if x)
    return f"{(run.get('kind') or 'run').capitalize()} #{run['id']}" + (f" · {who}" if who else "")


def pred_class_fn(run: dict):
    """(cls, raw label) -> evaluation class for one run's stored predictions. Geometry-mode audits store the
    detector's family classes (PSEUDO_BASE + k, label 'person'): they become the family's smallest dataset class,
    which the evaluation's ClassView then counts as the whole family."""
    audit = audit_info(run) if run.get("kind") == "audit" else None
    fam = (audit or {}).get("family") or {}
    if not fam:
        return lambda c, label=None: int(c)
    names = list(fam)
    by_label = {n.lower(): min(v) for n, v in fam.items()}

    def fn(c, label=None):
        c = int(c)
        hit = by_label.get(str(label or "").strip().lower())
        if hit is not None:
            return hit
        if c >= PSEUDO_BASE and c - PSEUDO_BASE < len(names):
            return min(fam[names[c - PSEUDO_BASE]])
        return c
    return fn


class RunSource(Source):
    """Stored predictions of one ``ai_runs`` row (kind 'detect' rows, whatever their review state: a proposal
    accepted or rejected later is still what the model said, and an audit keeps its raw detections as 'info')."""
    kind = "run"

    def __init__(self, dataset_id: str, run_id: int):
        self.dataset_id, self.run_id = dataset_id, int(run_id)
        run = db.query_one("SELECT * FROM ai_runs WHERE id=? AND dataset_id=?", (self.run_id, dataset_id))
        if not run:
            raise LookupError(f"AI run #{run_id} not found for this dataset")
        self.run = run
        self.label = run_label(run)
        self.audit = audit_info(run)
        to_cls = pred_class_fn(run)
        self.by_image: Dict[int, List[list]] = defaultdict(list)
        marks = ",".join("?" * len(PRED_KINDS))
        for r in db.query(f"SELECT image_id, cls, xc, yc, w, h, conf, label FROM ai_preds WHERE run_id=? AND kind IN ({marks}) ORDER BY image_id, id",
                          (self.run_id, *PRED_KINDS)):
            self.by_image[r["image_id"]].append([to_cls(r["cls"], r["label"]), r["xc"], r["yc"], r["w"], r["h"], r["conf"]])
        processed, self.replaced = processed_images(run)
        self.processed_known = processed is not None
        self.covered_ids = (processed - self.replaced) if processed is not None else set(self.by_image)
        self.ids: List[int] = []

    def class_groups(self) -> List[List[int]]:
        """Dataset classes this source can only be judged on together (an audit's class family)."""
        return [v for v in ((self.audit or {}).get("family") or {}).values() if len(v) > 1]

    def class_scope(self) -> Optional[List[int]]:
        """The dataset classes this source covers at all, when the run says so (geometry audit: its families)."""
        fam = (self.audit or {}).get("family") or {}
        return sorted({c for v in fam.values() for c in v}) if fam else None

    def family_labels(self) -> Dict[int, str]:
        return {min(v): name for name, v in ((self.audit or {}).get("family") or {}).items()}

    def prepare(self, items: List[dict]) -> None:
        self.ids = [it["id"] for it in items]

    def covers(self, idx: int) -> bool:
        return self.ids[idx] in self.covered_ids

    def preds(self, idx: int) -> List[list]:
        return [list(r) for r in self.by_image.get(self.ids[idx], [])]


def folder_label(p: Path) -> str:
    """'predA' for .../predA, 'predA/val' when the folder itself is a split or a 'labels' folder."""
    p = Path(p)
    if p.name.lower() in SPLIT_WORDS or p.name.lower() == "labels":
        return f"{p.parent.name}/{p.name}" if p.parent.name else p.name
    return p.name or str(p)


def apply_remap(rows: List[list], remap: Optional[Dict[int, int]]) -> List[list]:
    if not remap:
        return rows
    out = []
    for r in rows:
        c = remap.get(int(r[0]), int(r[0]))
        if c is None or c < 0:
            continue
        out.append([c] + list(r[1:]))
    return out


class ClassView:
    """How classes are counted in one evaluation, for ground truth and every model alike.

    ``merge``: groups of classes counted as one class (its smallest id), e.g. [[0, 1]] when a person detector is
    judged on 'standing' + 'fallen'. ``only``: the classes the evaluation covers at all; boxes of other classes,
    labelled or predicted, are left out (neither a hit nor a miss). Applied after each source's own class map."""

    def __init__(self, merge: Optional[Sequence[Sequence[int]]] = None, only: Optional[Iterable[int]] = None,
                 labels: Optional[Dict[int, str]] = None):
        parent: Dict[int, int] = {}

        def find(x: int) -> int:
            while parent.setdefault(x, x) != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        for grp in merge or []:
            g = sorted({int(c) for c in grp})
            for c in g[1:]:
                a, b = find(g[0]), find(c)
                if a != b:
                    parent[max(a, b)] = min(a, b)
        members: Dict[int, List[int]] = defaultdict(list)
        for c in list(parent):
            members[find(c)].append(c)
        self.groups = {r: sorted(m) for r, m in members.items() if len(m) > 1}
        self.rep = {c: r for r, m in self.groups.items() for c in m}
        self.only = frozenset(self.rep.get(int(c), int(c)) for c in only) if only is not None else None
        self.labels = {int(k): str(v) for k, v in (labels or {}).items() if int(k) in self.groups}

    @property
    def active(self) -> bool:
        return bool(self.groups) or self.only is not None

    def cls(self, c: int) -> Optional[int]:
        r = self.rep.get(int(c), int(c))
        return None if self.only is not None and r not in self.only else r

    def rows(self, rows: List[list]) -> List[list]:
        if not self.active:
            return rows
        out = []
        for r in rows:
            c = self.cls(r[0])
            if c is not None:
                out.append([c] + list(r[1:]))
        return out

    def to_meta(self, names: Sequence[str]) -> Optional[dict]:
        if not self.active:
            return None
        nm = lambda c: names[c] if 0 <= c < len(names) and names[c] else f"class_{c}"   # noqa: E731
        labels = {}
        for r, m in self.groups.items():
            parts = "+".join(str(c) for c in m)
            labels[str(r)] = f"{self.labels[r]} ({parts})" if r in self.labels else " + ".join(nm(c) for c in m)
        return {"merge": [m for _, m in sorted(self.groups.items())], "only": sorted(self.only) if self.only is not None else None,
                "labels": labels}

    @classmethod
    def from_meta(cls, meta: Optional[dict]) -> "ClassView":
        cv = (meta or {}).get("classes_view") or {}
        view = cls(cv.get("merge") or [], cv.get("only"))
        return view


def parse_merge(spec: Any) -> List[List[int]]:
    """'0+1, 2+3' or [[0, 1], [2, 3]] -> [[0, 1], [2, 3]]. Raises ValueError."""
    if not spec:
        return []
    groups: List[Any]
    if isinstance(spec, (list, tuple)):
        groups = list(spec)
    else:
        groups = [g for g in str(spec).replace(";", ",").split(",") if g.strip()]
    out: List[List[int]] = []
    for g in groups:
        toks = g if isinstance(g, (list, tuple)) else [t for t in str(g).replace("&", "+").replace(" ", "+").split("+") if t.strip()]
        try:
            ids = sorted({int(str(t).strip()) for t in toks})
        except ValueError:
            raise ValueError(f"Merge group '{g}' needs class ids joined by +, e.g. 0+1") from None
        if any(i < 0 for i in ids):
            raise ValueError(f"Merge group '{g}' has a negative class id")
        if len(ids) < 2:
            raise ValueError(f"Merge group '{g}' needs at least two classes, e.g. 0+1")
        out.append(ids)
    return out


def parse_remap(spec: Any) -> Dict[int, int]:
    """'0:1, 1:0, 5:-' or {"0": 1} -> {0: 1, 1: 0, 5: -1}. '-' (or -1) drops the class. Raises ValueError."""
    if not spec:
        return {}
    pairs: List[Tuple[Any, Any]]
    if isinstance(spec, dict):
        pairs = list(spec.items())
    else:
        pairs = []
        for tok in str(spec).replace(";", ",").split(","):
            tok = tok.strip()
            if not tok:
                continue
            for sep in ("->", "→", ":", "="):
                if sep in tok:
                    a, b = tok.split(sep, 1)
                    pairs.append((a.strip(), b.strip()))
                    break
            else:
                raise ValueError(f"Class map entry '{tok}' needs the form from:to (e.g. 0:1, 3:-)")
    out: Dict[int, int] = {}
    for a, b in pairs:
        try:
            src = int(str(a).strip())
            dst = -1 if str(b).strip() in ("-", "", "none", "drop", "None") else int(str(b).strip())
        except ValueError:
            raise ValueError(f"Class map entry '{a}:{b}' is not a pair of class ids") from None
        if src < 0 or dst < -1:
            raise ValueError(f"Class map entry '{a}:{b}' has a negative class id")
        out[src] = dst
    return out


# ------------------------------------------------------------------ matching

def _xyxy(rows: Sequence[Sequence[float]]) -> List[Tuple[float, float, float, float, float]]:
    return [(r[1] - r[3] / 2, r[2] - r[4] / 2, r[1] + r[3] / 2, r[2] + r[4] / 2, r[3] * r[4]) for r in rows]


def iou_matrix(preds: Sequence[Sequence[float]], gts: Sequence[Sequence[float]]) -> List[List[float]]:
    """Class-agnostic IoU, rows = predictions, columns = ground truth."""
    P, G = _xyxy(preds), _xyxy(gts)
    M = []
    for px0, py0, px1, py1, pa in P:
        row = []
        for gx0, gy0, gx1, gy1, ga in G:
            iw = (px1 if px1 < gx1 else gx1) - (px0 if px0 > gx0 else gx0)
            if iw <= 0:
                row.append(0.0)
                continue
            ih = (py1 if py1 < gy1 else gy1) - (py0 if py0 > gy0 else gy0)
            if ih <= 0:
                row.append(0.0)
                continue
            inter = iw * ih
            union = pa + ga - inter
            row.append(inter / union if union > 0 else 0.0)
        M.append(row)
    return M


def _greedy(pcls: Sequence[int], gcls: Sequence[int], M: List[List[float]], t: float,
            cands: Optional[List[List[int]]] = None) -> List[int]:
    """COCO rule. Predictions must already be in descending confidence. Returns the GT index per prediction (-1)."""
    taken = [False] * len(gcls)
    out = [-1] * len(pcls)
    for p in range(len(pcls)):
        row = M[p]
        best, best_iou = -1, t
        for g in (cands[p] if cands is not None else range(len(gcls))):
            if taken[g] or gcls[g] != pcls[p]:
                continue
            v = row[g]
            if v >= best_iou:
                best, best_iou = g, v
        if best >= 0:
            taken[best] = True
            out[p] = best
    return out


def _ultralytics(pcls: Sequence[int], gcls: Sequence[int], M: List[List[float]], t: float,
                 cands: Optional[List[List[int]]] = None) -> List[int]:
    """ultralytics DetectionValidator.match_predictions (non-scipy): each detection proposes its highest-IoU box,
    each box keeps the lowest-index (highest-confidence) detection that proposed it."""
    best_for: Dict[int, Tuple[float, int]] = {}
    for p in range(len(pcls)):
        row = M[p]
        best, best_iou = -1, -1.0
        for g in (cands[p] if cands is not None else range(len(gcls))):
            if gcls[g] != pcls[p]:
                continue
            v = row[g]
            if v >= t and v > best_iou:
                best, best_iou = g, v
        if best >= 0:
            best_for[p] = (best_iou, best)
    out = [-1] * len(pcls)
    claimed = set()
    for p in sorted(best_for):
        g = best_for[p][1]
        if g not in claimed:
            claimed.add(g)
            out[p] = g
    return out


def match_image(gt: List[List[float]], preds: List[List[float]], iou_thr: float) -> Tuple[List[bool], List[int]]:
    """Greedy COCO matching of one image at one IoU. Rows: gt [cls,xc,yc,w,h], preds [cls,xc,yc,w,h,conf,...]
    in any order. Returns (pred_is_tp[], gt_matched_by_pred_index[] with -1 for a miss)."""
    order = sorted(range(len(preds)), key=lambda i: -(preds[i][5] if len(preds[i]) > 5 and preds[i][5] is not None else 1.0))
    sp = [preds[i] for i in order]
    M = iou_matrix(sp, gt)
    m = _greedy([int(p[0]) for p in sp], [int(g[0]) for g in gt], M, iou_thr)
    pred_tp = [False] * len(preds)
    gt_matched = [-1] * len(gt)
    for k, g in enumerate(m):
        if g >= 0:
            pred_tp[order[k]] = True
            gt_matched[g] = order[k]
    return pred_tp, gt_matched


# ------------------------------------------------------------------ AP

def _interp(x: float, xp: Sequence[float], fp: Sequence[float]) -> float:
    """numpy.interp for a non-decreasing xp (duplicates take the rightmost value), like Ultralytics relies on."""
    if x < xp[0]:
        return fp[0]
    if x > xp[-1]:
        return fp[-1]
    j = bisect_right(xp, x) - 1
    if j >= len(xp) - 1 or xp[j] == x:
        return fp[j]
    return fp[j] + (fp[j + 1] - fp[j]) * (x - xp[j]) / (xp[j + 1] - xp[j])


_X101 = [i * 0.01 for i in range(100)] + [1.0]


def ap_from_flags(flags: Sequence[int], n_gt: int, method: str = "coco") -> Optional[float]:
    """AP of one class from TP flags in descending confidence. None when the class has no ground truth."""
    if n_gt <= 0:
        return None
    if not flags:
        return 0.0
    tp = fp = 0
    rec: List[float] = []
    prec: List[float] = []
    for f in flags:
        if f:
            tp += 1
        else:
            fp += 1
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    if method == "voc":
        mrec = [0.0] + rec + [1.0]
        mpre = [0.0] + prec + [0.0]
        for i in range(len(mpre) - 2, -1, -1):
            if mpre[i + 1] > mpre[i]:
                mpre[i] = mpre[i + 1]
        return sum((mrec[i] - mrec[i - 1]) * mpre[i] for i in range(1, len(mrec)) if mrec[i] != mrec[i - 1])
    if method == "ultralytics":
        mrec = [0.0] + rec + [rec[-1], 1.0]
        mpre = [1.0] + prec + [0.0, 0.0]
        for i in range(len(mpre) - 2, -1, -1):
            if mpre[i + 1] > mpre[i]:
                mpre[i] = mpre[i + 1]
        y = [_interp(x, mrec, mpre) for x in _X101]
        return sum((y[i] + y[i + 1]) / 2 * (_X101[i + 1] - _X101[i]) for i in range(100))
    # coco: precision envelope sampled at recall 0, 0.01, ... 1 (pycocotools accumulate)
    for i in range(len(prec) - 2, -1, -1):
        if prec[i + 1] > prec[i]:
            prec[i] = prec[i + 1]
    s = 0.0
    for k in range(GRID):
        i = bisect_left(rec, k / 100)
        if i < len(rec):
            s += prec[i]
    return s / GRID


def pr_envelope(flags: Sequence[int], n_gt: int) -> List[float]:
    """Interpolated precision at recall 0..1 in steps of 0.01 (what COCO averages), for plotting."""
    out = [0.0] * GRID
    if n_gt <= 0 or not flags:
        return out
    tp = fp = 0
    rec, prec = [], []
    for f in flags:
        tp += 1 if f else 0
        fp += 0 if f else 1
        rec.append(tp / n_gt)
        prec.append(tp / (tp + fp))
    for i in range(len(prec) - 2, -1, -1):
        if prec[i + 1] > prec[i]:
            prec[i] = prec[i + 1]
    for k in range(GRID):
        i = bisect_left(rec, k / 100)
        if i < len(rec):
            out[k] = round(prec[i], 5)
    return out


def average_precision(records: List[Tuple[float, bool]], n_gt: int, method: str = "voc") -> float:
    """Legacy helper: AP from (conf, is_tp) records (all-point by default). 0.0 when there is nothing to rank."""
    if n_gt == 0 or not records:
        return 0.0
    ranked = sorted(records, key=lambda r: -r[0])
    return ap_from_flags([1 if ok else 0 for _, ok in ranked], n_gt, method) or 0.0


# ------------------------------------------------------------------ legacy accumulator (benchmarks, tests)

def pred_file_for(pred_dir: Path, rel_path: str, split: str) -> Optional[Path]:
    """Single-image lookup used by older callers; evaluations index the folder once instead."""
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
    """Rows [cls, xc, yc, w, h, conf, has_conf] with conf >= conf_thr. A row without a score is kept at 1.0
    and marked has_conf=False, so ranking metrics can refuse to invent an AP."""
    p = pred_file_for(pred_dir, rel_path, split)
    if p is None:
        return []
    out = []
    for r in read_pred_file(p):
        conf = 1.0 if r[5] is None else r[5]
        if conf >= conf_thr:
            out.append([r[0], r[1], r[2], r[3], r[4], conf, r[5] is not None])
    return out


class SideAcc:
    """Micro P/R/F1 + per-class AP accumulator over matched images (kept for benchmarks and old callers)."""

    def __init__(self, method: str = "voc"):
        self.method = method
        self.tp = defaultdict(int); self.fp = defaultdict(int); self.fn = defaultdict(int)
        self.records: Dict[int, List[Tuple[float, bool]]] = defaultdict(list)
        self.n_gt = defaultdict(int)
        self.with_conf = 0
        self.without_conf = 0

    def add(self, gt, preds, pred_tp, gt_matched):
        for g, m in zip(gt, gt_matched):
            c = int(g[0]); self.n_gt[c] += 1
            if m == -1:
                self.fn[c] += 1
        for p, ok in zip(preds, pred_tp):
            if (len(p) > 6 and not p[6]) or (len(p) > 5 and p[5] is None):
                self.without_conf += 1
            else:
                self.with_conf += 1
            c = int(p[0])
            if ok:
                self.tp[c] += 1
            else:
                self.fp[c] += 1
            self.records[c].append((p[5] if len(p) > 5 and p[5] is not None else 1.0, ok))

    def summary(self, names: List[str]) -> dict:
        # AP ranks predictions by confidence. Without scores every prediction ties and the curve's order is
        # arbitrary: report it as unavailable rather than a plausible lie.
        ranked = self.without_conf == 0
        classes = sorted(set(self.n_gt) | set(self.tp) | set(self.fp))
        rows, aps = [], []
        T = F = N = 0
        for c in classes:
            tp, fp, fn = self.tp[c], self.fp[c], self.fn[c]
            T += tp; F += fp; N += fn
            prec = tp / (tp + fp) if tp + fp else 0.0
            rec = tp / (tp + fn) if tp + fn else 0.0
            f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
            ap = average_precision(self.records[c], self.n_gt[c], self.method) if self.n_gt[c] > 0 else None
            if ap is not None:
                aps.append(ap)
            rows.append({"cls": c, "name": names[c] if 0 <= c < len(names) else f"class_{c}", "gt": self.n_gt[c],
                         "tp": tp, "fp": fp, "fn": fn, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                         "ap": round(ap, 4) if (ranked and ap is not None) else None})
        prec = T / (T + F) if T + F else 0.0
        rec = T / (T + N) if T + N else 0.0
        f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
        return {"classes": rows, "conf_available": ranked, "preds_without_conf": self.without_conf,
                "overall": {"gt": T + N, "tp": T, "fp": F, "fn": N, "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
                            "map50": (round(sum(aps) / len(aps), 4) if aps else 0.0) if ranked else None}}


# ------------------------------------------------------------------ per-image evaluation

class ImageResult:
    """Everything one side contributes for one image: per-prediction rows (sorted by confidence) and outcomes."""
    __slots__ = ("cls", "conf", "lvl", "bits", "gt_of", "size", "bucket", "gt_lvl", "swaps")

    def __init__(self):
        self.cls: List[int] = []; self.conf: List[float] = []; self.lvl: List[int] = []
        self.bits: List[int] = []; self.gt_of: List[int] = []; self.size: List[int] = []
        self.bucket: List[List[int]] = []      # per size bucket: per prediction 1 TP / 0 FP / -1 ignored
        self.gt_lvl: List[int] = []            # per GT: level of the prediction matching it at the primary IoU, -1
        self.swaps: List[Tuple[int, int, int, int, int]] = []   # (lo, hi, gt_cls, pred_cls, n) confusion runs


def evaluate_image(gt: List[List[float]], preds: List[list], ious: Sequence[float], primary: int, method: str,
                   W: Optional[int], H: Optional[int]) -> ImageResult:
    """Match one image at every IoU in ``ious``. ``preds`` rows are [cls, xc, yc, w, h, conf|None]."""
    res = ImageResult()
    order = sorted(range(len(preds)), key=lambda i: -(preds[i][5] if preds[i][5] is not None else 1.0))
    P = [preds[i] for i in order]
    pcls = [int(p[0]) for p in P]
    gcls = [int(g[0]) for g in gt]
    M = iou_matrix(P, gt) if P and gt else [[] for _ in P]
    lo_t = min(ious)
    cands = [[g for g in range(len(gt)) if M[p][g] >= lo_t] for p in range(len(P))]
    matcher = _ultralytics if method == "ultralytics" else _greedy
    bits = [0] * len(P)
    primary_gt = [-1] * len(P)
    for j, t in enumerate(ious):
        m = matcher(pcls, gcls, M, t, cands)
        for p, g in enumerate(m):
            if g >= 0:
                bits[p] |= 1 << j
        if j == primary:
            primary_gt = m
    Wp, Hp = (W or 640), (H or 640)
    res.cls = pcls
    res.conf = [p[5] if p[5] is not None else 1.0 for p in P]
    res.lvl = [conf_level(p[5]) for p in P]
    res.bits = bits
    res.gt_of = primary_gt
    res.size = [size_bucket(p[3] * Wp * p[4] * Hp) for p in P]
    gt_lvl = [-1] * len(gt)
    for p, g in enumerate(primary_gt):
        if g >= 0:
            gt_lvl[g] = res.lvl[p]
    res.gt_lvl = gt_lvl
    t = ious[primary]
    # COCO area-range matching (greedy, at the primary IoU): boxes outside the range are ignored, a prediction
    # matched to an ignored box is ignored, an unmatched prediction outside the range is ignored.
    gsize = [size_bucket(g[3] * Wp * g[4] * Hp) for g in gt]
    for b in range(len(SIZE_BUCKETS)):
        ign = [s != b for s in gsize]
        g_order = sorted(range(len(gt)), key=lambda g: ign[g])
        taken = [False] * len(gt)
        outc = []
        for p in range(len(P)):
            best, best_iou = -1, min(t, 1 - 1e-10)
            for g in g_order:
                if taken[g]:
                    continue
                if best > -1 and not ign[best] and ign[g]:
                    break
                if gcls[g] != pcls[p] or M[p][g] < best_iou:
                    continue
                best, best_iou = g, M[p][g]
            if best == -1:
                outc.append(-1 if res.size[p] != b else 0)
            else:
                taken[best] = True
                outc.append(-1 if ign[best] else 1)
        res.bucket.append(outc)
    res.swaps = _swap_runs(pcls, res.lvl, primary_gt, gcls, M, t)
    return res


def _swap_runs(pcls, lvl, primary_gt, gcls, M, t) -> List[Tuple[int, int, int, int, int]]:
    """Class confusions as runs over the threshold grid: (lo, hi, gt_cls, pred_cls, n) means n predictions of
    pred_cls sit on unmatched boxes of gt_cls for every threshold index k in [lo, hi]."""
    swap_c = [sorted((g for g in range(len(gcls)) if gcls[g] != pcls[p] and M[p][g] >= t), key=lambda g: -M[p][g])
              if primary_gt[p] < 0 else [] for p in range(len(pcls))]
    if not any(swap_c):
        return []
    levels = sorted(set(lvl), reverse=True)
    runs: List[Tuple[int, int, int, int, int]] = []
    open_: Dict[Tuple[int, int], Tuple[int, int, int]] = {}    # cell -> (n, hi, lo)
    for j, v in enumerate(levels):
        lo = levels[j + 1] + 1 if j + 1 < len(levels) else 0
        active = [p for p in range(len(pcls)) if lvl[p] >= v]
        claimed = {primary_gt[p] for p in active if primary_gt[p] >= 0}
        cells: Dict[Tuple[int, int], int] = defaultdict(int)
        for p in active:                                  # already in descending confidence
            for g in swap_c[p]:
                if g not in claimed:
                    claimed.add(g)
                    cells[(gcls[g], pcls[p])] += 1
                    break
        for cell in list(open_):
            n, hi, _lo = open_[cell]
            if cells.get(cell) != n:
                runs.append((_lo, hi, cell[0], cell[1], n))
                del open_[cell]
        for cell, n in cells.items():
            if cell in open_:
                open_[cell] = (n, open_[cell][1], lo)
            else:
                open_[cell] = (n, v, lo)
    for cell, (n, hi, lo) in open_.items():
        runs.append((lo, hi, cell[0], cell[1], n))
    return runs


# ------------------------------------------------------------------ whole-dataset evaluation

def scope_items(dataset_id: str, filters: dict) -> List[dict]:
    """Every image in scope, read in ONE query ordered by id: a frozen list, so label edits during the job can
    neither skip nor repeat an image (OFFSET paging over a live filter can)."""
    from ...core.imagesq import build_where
    where, params = build_where(dataset_id, filters)
    rows = db.query(f"""SELECT i.id, i.rel_path, i.split, i.width, i.height, i.has_label
                        FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path
                        WHERE {where} ORDER BY i.id""", params)
    return rows


def outside_regions(rows: List[List[float]], regions) -> List[List[float]]:
    """Boxes whose centre is not inside one of the image's regions that are not objects (ignored regions, DontCare,
    crowds — core.annindex): as the VisDrone and KITTI protocols do, those areas count neither hits nor misses."""
    if not regions:
        return rows
    return [r for r in rows if not any(abs(r[1] - g[0]) <= g[2] / 2 and abs(r[2] - g[1]) <= g[3] / 2 for g in regions)]


def gt_for(ids: Sequence[int]) -> Dict[int, List[List[float]]]:
    out: Dict[int, List[List[float]]] = defaultdict(list)
    for i in range(0, len(ids), 500):
        chunk = list(ids[i:i + 500])
        for r in db.query("SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE image_id IN (%s) ORDER BY id" % ",".join("?" * len(chunk)), chunk):
            out[r["image_id"]].append([int(r["cls"]), r["xc"], r["yc"], r["w"], r["h"]])
    return out


class SideData:
    """Columnar per-prediction / per-GT results of one side (what the stored blob holds)."""

    def __init__(self):
        self.p_img = array.array("I"); self.p_cls = array.array("i"); self.p_lvl = array.array("b")
        self.p_conf = array.array("f"); self.p_bits = array.array("H"); self.p_gt = array.array("i")
        self.p_off = array.array("I", [0])
        self.g_lvl = array.array("b")
        self.s_img = array.array("I"); self.s_lo = array.array("b"); self.s_hi = array.array("b")
        self.s_g = array.array("i"); self.s_p = array.array("i"); self.s_n = array.array("H")
        self.files: List[str] = []
        self.noconf = 0
        # in-memory only (curves): exact confidences and size-bucket outcomes
        self.conf_d: List[float] = []
        self.bucket: List[List[int]] = [[] for _ in SIZE_BUCKETS]
        self.p_size: List[int] = []


def evaluate(job: Optional[Job], dataset_id: str, items: List[dict], sources: Dict[str, Source], *, iou: float,
             method: str, unlabeled: str = "background", remap: Optional[Dict[str, Dict[int, int]]] = None,
             classview: Optional[ClassView] = None) -> dict:
    """Evaluate every side over ``items``. Returns {'images', 'skipped', 'data': packed dict, 'curves', ...}."""
    remap = remap or {}
    cv = classview or ClassView()
    ious = sorted(set(IOU_GRID) | {round(float(iou), 4)})
    primary = ious.index(round(float(iou), 4))
    for s in sources.values():
        s.prepare(items)
    skipped = {"ambiguous": 0, "not_covered": 0, "unlabeled": 0}
    keep: List[int] = []
    for idx, it in enumerate(items):
        if any(s.ambiguous(idx) for s in sources.values()):
            skipped["ambiguous"] += 1
        elif not all(s.covers(idx) for s in sources.values()):
            skipped["not_covered"] += 1
        elif unlabeled == "skip" and not it.get("has_label"):
            skipped["unlabeled"] += 1
        else:
            keep.append(idx)
    total = len(keep)
    if job:
        job.update(0, total, f"Matching 0/{total:,}")
    img_ids = array.array("I"); rels: List[str] = []; splits: List[str] = []; split_ix = array.array("H")
    split_pos: Dict[str, int] = {}
    g_img = array.array("I"); g_cls = array.array("i"); g_size = array.array("b"); g_off = array.array("I", [0])
    sides = {k: SideData() for k in sources}
    matched_files = {k: 0 for k in sources}
    unlabeled_bg = 0
    for start in range(0, total, 500):
        chunk = keep[start:start + 500]
        gts = gt_for([items[i]["id"] for i in chunk])
        from ...core.annindex import regions_by_image
        regs = regions_by_image([items[i]["id"] for i in chunk])
        for n, idx in enumerate(chunk):
            k = start + n
            if job and k % 100 == 0:
                job.update(done=k, message=f"Matching {k:,}/{total:,}")
            it = items[idx]
            img_ids.append(it["id"]); rels.append(it["rel_path"])
            sp = it["split"] or ""
            if sp not in split_pos:
                split_pos[sp] = len(splits); splits.append(sp)
            split_ix.append(split_pos[sp])
            if not it.get("has_label"):
                unlabeled_bg += 1
            gt = cv.rows(outside_regions(gts.get(it["id"], []), regs.get(it["id"])))
            W, H = it.get("width"), it.get("height")
            for g in gt:
                g_img.append(k); g_cls.append(int(g[0]))
                g_size.append(size_bucket(g[3] * (W or 640) * g[4] * (H or 640)))
            g_off.append(len(g_img))
            for side, src in sources.items():
                sd = sides[side]
                f = src.file(idx)
                sd.files.append(f)
                if f:
                    matched_files[side] += 1
                preds = cv.rows(outside_regions(apply_remap(src.preds(idx), remap.get(side)), regs.get(it["id"])))
                sd.noconf += sum(1 for p in preds if p[5] is None)
                r = evaluate_image(gt, preds, ious, primary, method, W, H)
                for p in range(len(r.cls)):
                    sd.p_img.append(k); sd.p_cls.append(r.cls[p]); sd.p_lvl.append(r.lvl[p])
                    sd.p_conf.append(r.conf[p]); sd.conf_d.append(r.conf[p]); sd.p_bits.append(r.bits[p])
                    sd.p_gt.append(r.gt_of[p]); sd.p_size.append(r.size[p])
                    for b in range(len(SIZE_BUCKETS)):
                        sd.bucket[b].append(r.bucket[b][p])
                sd.p_off.append(len(sd.p_img))
                for lv in r.gt_lvl:
                    sd.g_lvl.append(lv)
                for lo, hi, gc, pc, cnt in r.swaps:
                    sd.s_img.append(k); sd.s_lo.append(lo); sd.s_hi.append(hi)
                    sd.s_g.append(gc); sd.s_p.append(pc); sd.s_n.append(min(cnt, 65535))
    if job:
        job.update(done=total, message="Computing curves")
    curves = {side: side_curves(sides[side], g_cls, g_size, ious, primary, method) for side in sources}
    data = {"v": 2, "n": total, "ious": ious, "primary": primary, "splits": splits,
            "img": img_ids, "split": split_ix, "rel": rels,
            "g_img": g_img, "g_cls": g_cls, "g_size": g_size, "g_off": g_off,
            "sides": {side: {"p_img": sd.p_img, "p_cls": sd.p_cls, "p_lvl": sd.p_lvl, "p_conf": sd.p_conf,
                             "p_bits": sd.p_bits, "p_gt": sd.p_gt, "p_off": sd.p_off, "g_lvl": sd.g_lvl,
                             "s_img": sd.s_img, "s_lo": sd.s_lo, "s_hi": sd.s_hi, "s_g": sd.s_g, "s_p": sd.s_p,
                             "s_n": sd.s_n, "files": sd.files} for side, sd in sides.items()}}
    return {"images": total, "skipped": skipped, "unlabeled_background": unlabeled_bg if unlabeled != "skip" else 0,
            "pred_files": matched_files, "ious": ious, "primary": primary, "curves": curves, "data": data}


def side_curves(sd: SideData, g_cls, g_size, ious: Sequence[float], primary: int, method: str) -> dict:
    """Per-class AP at every IoU, precision envelope, and TP/FP counts on the confidence grid (overall and per
    object size). Threshold-dependent numbers are derived from these by the client without another request."""
    n = len(sd.p_cls)
    by_cls: Dict[int, List[int]] = defaultdict(list)
    for i in range(n):
        by_cls[sd.p_cls[i]].append(i)
    n_gt: Dict[int, int] = defaultdict(int)
    n_gt_b: List[Dict[int, int]] = [defaultdict(int) for _ in SIZE_BUCKETS]
    for c, s in zip(g_cls, g_size):
        n_gt[c] += 1
        n_gt_b[s][c] += 1
    pmask = 1 << primary
    classes = {}
    for c in sorted(set(n_gt) | set(by_cls)):
        idx = by_cls.get(c, [])
        idx.sort(key=lambda i: -sd.conf_d[i])            # stable: image order, then confidence order in an image
        ap = []
        for j in range(len(ious)):
            m = 1 << j
            ap.append(_r(ap_from_flags([1 if sd.p_bits[i] & m else 0 for i in idx], n_gt[c], method)))
        prim = [1 if sd.p_bits[i] & pmask else 0 for i in idx]
        tp_at, fp_at = _grid_counts(idx, sd.p_lvl, prim)
        classes[str(c)] = {"gt": n_gt[c], "np": len(idx), "ap": ap, "pr": pr_envelope(prim, n_gt[c]),
                           "tp": tp_at, "fp": fp_at}
    sizes = []
    for b, (name, _, _) in enumerate(SIZE_BUCKETS):
        per = {}
        for c in sorted(set(n_gt_b[b]) | set(by_cls)):
            idx = [i for i in by_cls.get(c, []) if sd.bucket[b][i] >= 0]
            flags = [sd.bucket[b][i] for i in idx]
            if not idx and not n_gt_b[b][c]:
                continue
            tp_at, fp_at = _grid_counts(idx, sd.p_lvl, flags)
            per[str(c)] = {"gt": n_gt_b[b][c], "ap": _r(ap_from_flags(flags, n_gt_b[b][c], method)), "tp": tp_at, "fp": fp_at}
        sizes.append({"name": name, "classes": per})
    return {"classes": classes, "sizes": sizes, "preds": n, "preds_without_conf": sd.noconf,
            "conf_available": sd.noconf == 0}


def _r(v: Optional[float]) -> Optional[float]:
    return None if v is None else round(v, 6)


def _grid_counts(idx: Sequence[int], lvl, flags: Sequence[int]) -> Tuple[List[int], List[int]]:
    """TP and FP counts of predictions active at each grid threshold k (level >= k)."""
    tp_h = [0] * GRID
    fp_h = [0] * GRID
    for i, f in zip(idx, flags):
        if f:
            tp_h[lvl[i]] += 1
        else:
            fp_h[lvl[i]] += 1
    for k in range(GRID - 2, -1, -1):
        tp_h[k] += tp_h[k + 1]
        fp_h[k] += fp_h[k + 1]
    return tp_h, fp_h


def summarize(curves: dict, k: int, classes: Optional[Iterable[int]] = None, ious: Sequence[float] = IOU_GRID,
              primary: int = 0) -> dict:
    """Headline numbers of one side at threshold index k over the given classes (all when None)."""
    cl = curves["classes"]
    wanted = None if classes is None else {int(c) for c in classes}
    keys = [c for c in cl if wanted is None or int(c) in wanted]
    T = F = G = 0
    ap50, ap5095 = [], []
    on_grid = {round(x, 4) for x in IOU_GRID}
    grid_idx = [i for i, v in enumerate(ious) if round(v, 4) in on_grid]
    for c in keys:
        d = cl[c]
        T += d["tp"][k]; F += d["fp"][k]; G += d["gt"]
        if d["gt"] > 0 and d["ap"][primary] is not None:
            ap50.append(d["ap"][primary])
            vals = [d["ap"][i] for i in grid_idx if d["ap"][i] is not None]
            if vals:
                ap5095.append(sum(vals) / len(vals))
    p = T / (T + F) if T + F else 0.0
    r = T / G if G else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    ok = curves.get("conf_available", True)
    return {"gt": G, "tp": T, "fp": F, "fn": G - T, "precision": round(p, 4), "recall": round(r, 4), "f1": round(f1, 4),
            "map": round(sum(ap50) / len(ap50), 4) if (ap50 and ok) else None,
            "map5095": round(sum(ap5095) / len(ap5095), 4) if (ap5095 and ok) else None,
            "conf_available": ok}


# ------------------------------------------------------------------ packing

_TYPES = {"I", "i", "b", "f", "H", "d"}


def pack(obj: Any) -> str:
    """JSON-safe, compact encoding of nested dicts/lists with array.array leaves (zlib + base64 per array)."""
    def enc(o):
        if isinstance(o, array.array):
            return {"$a": o.typecode, "b": base64.b64encode(zlib.compress(o.tobytes(), 6)).decode("ascii")}
        if isinstance(o, dict):
            return {k: enc(v) for k, v in o.items()}
        if isinstance(o, list) and o and isinstance(o[0], str):
            return {"$s": base64.b64encode(zlib.compress("\x00".join(o).encode("utf-8"), 6)).decode("ascii"), "n": len(o)}
        if isinstance(o, list):
            return [enc(v) for v in o]
        return o
    return db.dumps(enc(obj))


def unpack(text: str) -> Any:
    import json

    def dec(o):
        if isinstance(o, dict):
            if "$a" in o and o["$a"] in _TYPES:
                a = array.array(o["$a"])
                a.frombytes(zlib.decompress(base64.b64decode(o["b"])))
                return a
            if "$s" in o:
                s = zlib.decompress(base64.b64decode(o["$s"])).decode("utf-8")
                return s.split("\x00") if o.get("n", 0) else []
            return {k: dec(v) for k, v in o.items()}
        if isinstance(o, list):
            return [dec(v) for v in o]
        return o
    return dec(json.loads(text))


# ------------------------------------------------------------------ discovery

def discover_pred_dirs(root: Path, sample: Optional[List[dict]] = None, max_depth: int = 6, limit: int = 300) -> List[dict]:
    """Folders beneath ``root`` holding YOLO txt files, plus parents whose split subfolders hold them ('use the
    parent for all splits'). With ``sample`` (dataset images), each folder gets the share of images it matches."""
    root = Path(root).resolve()
    per_dir: Dict[str, List[str]] = {}
    stack = [""]
    while stack and len(per_dir) < 5000:
        rel_dir = stack.pop()
        depth = len(rel_dir.split("/")) if rel_dir else 0
        try:
            with os.scandir(root / rel_dir if rel_dir else root) as it:
                entries = list(it)
        except OSError:
            continue
        txts = []
        for e in entries:
            if e.name.startswith("."):
                continue
            try:
                if e.is_dir():
                    if depth < max_depth:
                        stack.append(f"{rel_dir}/{e.name}" if rel_dir else e.name)
                    continue
            except OSError:
                continue
            if e.name.lower().endswith(LABEL_EXT) and e.name.lower() != "classes.txt":
                txts.append(e.name)
        if txts:
            per_dir[rel_dir] = txts
    per_dir = {d: t for d, t in per_dir.items() if _looks_yolo(root, d, t)}
    found = [_dir_entry(root, rel_dir, txts, len(txts), None) for rel_dir, txts in per_dir.items()]
    # parents of split folders: offer them as one source for every split
    children: Dict[str, List[str]] = defaultdict(list)
    for rel_dir in per_dir:
        if rel_dir:
            parent, _, name = rel_dir.rpartition("/")
            children[parent].append(name)
    for parent, names in children.items():
        if len(names) >= 2 and parent not in per_dir:
            subs = [f"{parent}/{n}" if parent else n for n in sorted(names)]
            entry = _dir_entry(root, parent, [], sum(len(per_dir[d]) for d in subs), sorted(names))
            entry["has_conf"] = _has_conf(root, parent, [f"{n}/{f}" for n, d in zip(sorted(names), subs) for f in sorted(per_dir[d])[:2]])
            found.append(entry)
    if sample:
        found.sort(key=lambda e: (not e["has_conf"], e["kind"] != "parent", -e["files"]))
        for e in found[:40]:
            folder = PredFolder(root / e["rel"] if e["rel"] else root)
            stems: Dict[str, int] = defaultdict(int)
            for it in sample:
                s = nfc(it["rel_path"]).rsplit("/", 1)[-1]
                stems[s.rsplit(".", 1)[0] if "." in s else s] += 1
            res = [folder.lookup(it["rel_path"], it["split"], stems) for it in sample]
            matched, how = [r[0] for r in res], [r[1] for r in res]
            PredFolder.resolve_claims(matched, how)
            hits = sum(1 for h in how if h in ("path", "split", "stem"))
            e["match"] = round(hits / len(sample), 3)
    found.sort(key=lambda e: (not e["has_conf"], -(e.get("match") or 0), e["kind"] != "parent", e["rel"]))
    return found[:limit]


def _looks_yolo(root: Path, rel_dir: str, names: Sequence[str]) -> bool:
    """False for folders whose txt files are notes or lists (dataset_info.txt, train.txt), not YOLO rows."""
    seen = 0
    for name in sorted(names)[:3]:
        try:
            with open(root / rel_dir / name if rel_dir else root / name, "rb") as fh:
                head = fh.read(2048).decode("utf-8", errors="ignore")
        except OSError:
            continue
        if not head.strip():
            continue                                  # an empty file is a valid "no detections"
        seen += 1
        if parse_pred_text(head.rsplit("\n", 1)[0] if len(head) == 2048 else head):
            return True
    return seen == 0


def _has_conf(root: Path, rel_dir: str, names: Sequence[str]) -> bool:
    votes = 0
    checked = 0
    for name in names[:5]:
        try:
            with open(root / rel_dir / name if rel_dir else root / name, "rb") as fh:
                head = fh.read(4096).decode("utf-8", errors="ignore")
        except OSError:
            continue
        rows = parse_pred_text(head.rsplit("\n", 1)[0] if len(head) == 4096 else head)
        if not rows:
            continue
        checked += 1
        votes += 1 if all(r[5] is not None for r in rows) else 0
    return checked > 0 and votes * 2 > checked


def _dir_entry(root: Path, rel_dir: str, txts: List[str], files: int, splits: Optional[List[str]]) -> dict:
    path = root / rel_dir if rel_dir else root
    return {"path": str(path), "rel": rel_dir, "files": files, "kind": "parent" if splits else "folder",
            "splits": splits, "has_conf": _has_conf(root, rel_dir, sorted(txts)) if txts else False}
