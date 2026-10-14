"""Exports: ZIP streaming, server-side subset copy, list files, data.yaml.

Every export runs from a *plan* built once from a snapshot of the matching image ids, so reviewing or
rescanning during a long export can neither skip nor duplicate images. The plan applies, in order: the
image filters, "drop near-duplicates" (keep one image per near-duplicate group), the split assignment
(original splits, or the sequence-aware split builder) and output naming (collisions get a suffix).
Label rows are rewritten through the class remap; segmentation / pose / OBB rows keep their coordinates
verbatim, confidence columns are dropped (an export is a training set).
"""
from __future__ import annotations

import errno
import io
import json
import logging
import math
import os
import random
import re
import shutil
import stat
import tempfile
import threading
import time
import unicodedata
import zipfile
from collections import Counter, OrderedDict, defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Dict, Iterable, Iterator, List, Optional, Set, Tuple

import yaml

from .. import __version__, db
from ..jobs import Job, JobCancelled, JobError
from ..paths import is_windows_path, resolve, to_host
from . import imagesq
from .imagesq import build_where
from .labels import MAX_CLASS_ID
from .thumbs import ImageTooLarge, decode_display, probe

log = logging.getLogger("fovea.export")

MANIFEST = "fovea-export.json"
CARD = "DATASET_CARD.md"
RESIZE_MIN, RESIZE_MAX = 16, 8192
YAML_KEYS = ("train", "val", "test")
_SPLIT_ALIASES = {"train": "train", "training": "train", "val": "val", "valid": "val", "validation": "val",
                  "eval": "val", "dev": "val", "test": "test", "testing": "test"}


# ============================================================================== options

def _as_int(v: Any, what: str) -> int:
    if isinstance(v, bool):
        raise ValueError(f"{what} must be a whole number")
    if isinstance(v, int):
        return v
    if isinstance(v, float) and v.is_integer():
        return int(v)
    if isinstance(v, str) and re.fullmatch(r"\s*-?\d+\s*", v):
        return int(v)
    raise ValueError(f"{what} must be a whole number, got {v!r}")


def _resize_spec(resize: Optional[dict]) -> Optional[Tuple[str, int, int]]:
    """('max', side, 0) | ('exact', w, h) | None. Raises ValueError for anything else: an empty or zero
    size used to produce a 'successful' archive that held only data.yaml."""
    if not resize:
        return None
    if not isinstance(resize, dict):
        raise ValueError("resize must be an object like {mode: 'max', size: 1280}")
    mode = resize.get("mode") or "none"
    if mode == "none":
        return None

    def side(key: str, label: str) -> int:
        if resize.get(key) in (None, ""):
            raise ValueError(f"Resize: {label} is required")
        n = _as_int(resize.get(key), f"Resize {label}")
        if not RESIZE_MIN <= n <= RESIZE_MAX:
            raise ValueError(f"Resize {label} must be between {RESIZE_MIN} and {RESIZE_MAX} px")
        return n

    if mode == "max":
        return ("max", side("size", "max side"), 0)
    if mode == "exact":
        return ("exact", side("width", "width"), side("height", "height"))
    raise ValueError(f"Unknown resize mode {mode!r} (use none, max or exact)")


@dataclass
class SplitBuilder:
    ratios: Dict[str, float]
    seed: int = 0
    group_by: str = "seq"            # seq | seq+dups | folder | image | regex
    regex: Optional[str] = None
    stratify: bool = True

    def describe(self) -> dict:
        return {"ratios": self.ratios, "seed": self.seed, "group_by": self.group_by, "regex": self.regex,
                "stratify": self.stratify}


@dataclass
class ExportOptions:
    resize: Optional[Tuple[str, int, int]] = None
    include_unlabeled: bool = True
    include_yaml: bool = True
    class_map: Dict[int, Optional[int]] = field(default_factory=dict)
    names_override: Optional[List[str]] = None
    compact: bool = False
    bad_class: str = "extend"        # extend | drop: rows whose class id has no name
    drop_near_dups: bool = False
    split_builder: Optional[SplitBuilder] = None
    split_map: Dict[str, str] = field(default_factory=dict)
    overwrite: bool = False
    dataset_card: bool = False       # write DATASET_CARD.md next to data.yaml

    def describe(self) -> dict:
        return {"resize": list(self.resize) if self.resize else None, "include_unlabeled": self.include_unlabeled,
                "include_yaml": self.include_yaml, "class_map": {str(k): v for k, v in self.class_map.items()},
                "names": self.names_override, "compact": self.compact, "bad_class": self.bad_class,
                "drop_near_dups": self.drop_near_dups, "split_map": self.split_map,
                "split_builder": self.split_builder.describe() if self.split_builder else None,
                "dataset_card": self.dataset_card}


def _first(payload: dict, *keys: str, default: Any = None) -> Any:
    for k in keys:
        if k in payload and payload[k] is not None:
            return payload[k]
    return default


def _parse_class_map(raw: Any) -> Dict[int, Optional[int]]:
    """{"0": 1, "1": 0, "3": null} (null / -1 / "drop" removes that class's rows) or [[0, 1], ...]."""
    if raw in (None, "", {}, []):
        return {}
    items: Iterable
    if isinstance(raw, dict):
        items = raw.items()
    elif isinstance(raw, list) and all(isinstance(x, (list, tuple)) and len(x) == 2 for x in raw):
        items = raw
    else:
        raise ValueError("class_map must be an object like {\"0\": 1, \"2\": null}")
    out: Dict[int, Optional[int]] = {}
    for k, v in items:
        src = _as_int(k, "class_map key")
        if src < 0:
            raise ValueError("class_map keys must be class ids >= 0")
        if v is None or v == "drop" or (not isinstance(v, bool) and isinstance(v, (int, float, str)) and str(v).strip() == "-1"):
            out[src] = None
            continue
        dst = _as_int(v, f"class_map[{k}]")
        if not 0 <= dst <= 9999:
            raise ValueError(f"class_map[{k}] must be a class id between 0 and 9999 (or null to drop)")
        out[src] = dst
    return out


def _parse_ratios(raw: Any) -> Dict[str, float]:
    if isinstance(raw, (list, tuple)) and len(raw) in (2, 3):
        raw = dict(zip(YAML_KEYS, raw))
    if not isinstance(raw, dict):
        raise ValueError("split ratios must look like {train: 0.8, val: 0.1, test: 0.1}")
    vals: Dict[str, float] = {}
    for k, v in raw.items():
        key = _SPLIT_ALIASES.get(str(k).lower())
        if key is None:
            raise ValueError(f"Unknown split {k!r} in ratios (use train, val, test)")
        try:
            f = float(v or 0)
        except (TypeError, ValueError):
            raise ValueError(f"Ratio for {k} must be a number")
        if f < 0:
            raise ValueError("Split ratios cannot be negative")
        vals[key] = vals.get(key, 0.0) + f
    total = sum(vals.values())
    if total <= 0:
        raise ValueError("Split ratios must add up to more than 0")
    if vals.get("train", 0) <= 0:
        raise ValueError("The split builder needs a train share above 0")
    return {k: v / total for k, v in vals.items() if v > 0}


def _parse_builder(raw: Any) -> Optional[SplitBuilder]:
    if not raw or raw is False:
        return None
    if raw is True:
        raw = {}
    if not isinstance(raw, dict) or raw.get("enabled") is False:
        return None
    ratio_src = raw.get("ratios") or {k: raw[k] for k in ("train", "val", "test") if k in raw} or {"train": 0.8, "val": 0.2}
    group_by = str(raw.get("group_by") or raw.get("group") or "seq").lower()
    if group_by in ("sequence", "sequences"):
        group_by = "seq"
    if group_by not in ("seq", "seq+dups", "folder", "image", "regex"):
        raise ValueError("group_by must be one of seq, seq+dups, folder, image, regex")
    rx = raw.get("regex") or raw.get("pattern")
    if group_by == "regex":
        if not rx:
            raise ValueError("group_by 'regex' needs a regex")
        try:
            re.compile(rx)
        except re.error as e:
            raise ValueError(f"Invalid group regex: {e}")
    seed = _as_int(raw.get("seed", 0) or 0, "seed")
    return SplitBuilder(ratios=_parse_ratios(ratio_src), seed=seed, group_by=group_by, regex=rx,
                        stratify=bool(raw.get("stratify", True)))


def parse_options(payload: Optional[dict]) -> ExportOptions:
    """Validate an export request body (shared by ZIP, copy and preview). Raises ValueError -> HTTP 400.
    Accepted aliases keep the API forgiving while the UI evolves: class_map|remap|class_remap,
    drop_near_dups|drop_near_duplicates|dedupe, split_builder|splits|resplit, dataset_card|include_card|card."""
    p = payload or {}
    names = _first(p, "names", "class_names")
    if names is not None:
        if not isinstance(names, list) or not all(isinstance(n, (str, int, float)) for n in names):
            raise ValueError("names must be a list of class names")
        names = [str(n).strip() or f"class_{i}" for i, n in enumerate(names)]
    bad = str(_first(p, "bad_class", "bad_class_rows", default="extend")).lower()
    if bad not in ("extend", "drop"):
        raise ValueError("bad_class must be 'extend' or 'drop'")
    split_map = _first(p, "split_map", default={}) or {}
    if not isinstance(split_map, dict):
        raise ValueError("split_map must map split names to train, val or test")
    sm = {}
    for k, v in split_map.items():
        key = _SPLIT_ALIASES.get(str(v).lower())
        if key is None:
            raise ValueError(f"split_map[{k!r}] must be train, val or test")
        sm[str(k)] = key
    return ExportOptions(
        resize=_resize_spec(p.get("resize")),
        include_unlabeled=bool(p.get("include_unlabeled", True)),
        include_yaml=bool(p.get("include_yaml", True)),
        class_map=_parse_class_map(_first(p, "class_map", "remap", "class_remap")),
        names_override=names, compact=bool(p.get("compact", False)), bad_class=bad,
        drop_near_dups=bool(_first(p, "drop_near_dups", "drop_near_duplicates", "dedupe", "near_dups", default=False)),
        split_builder=_parse_builder(_first(p, "split_builder", "splits", "resplit", "make_splits")),
        split_map=sm, overwrite=bool(p.get("overwrite", False)),
        dataset_card=bool(_first(p, "dataset_card", "include_card", "include_dataset_card", "card", default=False)),
    )


# ============================================================================== classes

@dataclass
class ClassPlan:
    names: List[str]                         # output class names (nc = len) before 'extend'
    mapping: Dict[int, Optional[int]]        # every source id with an explicit mapping (None = drop)
    remapped: bool

    def map(self, cls: int) -> Optional[int]:
        return self.mapping.get(cls, cls)


def class_plan(names: List[str], opts: ExportOptions) -> ClassPlan:
    src_ids = set(range(len(names))) | set(opts.class_map)
    mapping = {c: opts.class_map.get(c, c) for c in src_ids}
    if opts.compact:
        used = sorted({v for v in mapping.values() if v is not None})
        dense = {old: i for i, old in enumerate(used)}
        mapping = {c: (dense[v] if v is not None else None) for c, v in mapping.items()}
    targets = sorted({v for v in mapping.values() if v is not None})
    if opts.names_override is not None:
        out = list(opts.names_override)
        if targets and targets[-1] >= len(out):
            raise ValueError(f"names lists {len(out)} classes but the class map uses id {targets[-1]}")
    else:
        size = max([len(names) if not opts.compact else 0] + [t + 1 for t in targets])
        out = [names[i] if i < len(names) else f"class_{i}" for i in range(size)]
        lowest: Dict[int, int] = {}                 # one pass: a lookup per target was O(classes²) (5 s at 20k)
        for c, v in mapping.items():
            if v is not None and (v not in lowest or c < lowest[v]):
                lowest[v] = c
        for t in targets:
            # A merged class takes the name of the source that keeps its own id, else of the lowest source id.
            keep = t if mapping.get(t) == t else lowest[t]
            out[t] = names[keep] if keep < len(names) else f"class_{keep}"
        if opts.compact:
            out = out[: (targets[-1] + 1) if targets else 0]
    remapped = any(c != v for c, v in mapping.items()) or bool(opts.class_map)
    return ClassPlan(names=out, mapping=mapping, remapped=remapped)


@dataclass
class LabelStats:
    rows_in: int = 0
    rows_out: int = 0
    dropped_class: int = 0      # rows of classes mapped to "drop"
    dropped_invalid: int = 0    # unparsable rows / non-integer class ids / nan or inf coordinates
    dropped_unnamed: int = 0    # bad_class='drop': ids beyond the class list
    conf_stripped: int = 0
    max_cls: int = -1
    unnamed_rows: int = 0       # ids beyond the class list that were kept (names extended)

    def add(self, o: "LabelStats") -> None:
        for k in ("rows_in", "rows_out", "dropped_class", "dropped_invalid", "dropped_unnamed", "conf_stripped", "unnamed_rows"):
            setattr(self, k, getattr(self, k) + getattr(o, k))
        self.max_cls = max(self.max_cls, o.max_cls)


def transform_label_text(text: str, cp: Optional[ClassPlan] = None, nc: Optional[int] = None,
                         drop_unnamed: bool = False) -> Tuple[str, LabelStats]:
    """Rewrite one label file for a training export.

    5 values: a box (re-formatted to 6 decimals). 6: a box plus confidence (dropped). 7+: segmentation,
    pose or OBB — the coordinates are kept verbatim, only the class id is remapped. Rows the index treats as
    malformed (labels.parse_label_rows) are dropped and counted the same way: nan/inf in a coordinate passes float()
    but makes Ultralytics discard the whole image as corrupt."""
    st = LabelStats()
    out: List[str] = []
    for raw in text.lstrip("﻿").splitlines():
        parts = raw.split()
        if not parts:
            continue
        st.rows_in += 1
        try:
            vals = [float(t) for t in parts] if len(parts) >= 5 else None     # each token parsed once
        except ValueError:
            vals = None
        if (vals is None or not vals[0].is_integer() or not 0 <= vals[0] <= MAX_CLASS_ID
                or not all(math.isfinite(v) for v in (vals[1:5] if len(vals) <= 6 else vals[1:]))):
            st.dropped_invalid += 1             # a 6th value is a confidence: dropped below, so it may be anything
            continue
        cls = int(vals[0])
        new = cp.map(cls) if cp else cls
        if new is None:
            st.dropped_class += 1
            continue
        if nc is not None and new >= nc:
            if drop_unnamed:
                st.dropped_unnamed += 1
                continue
            st.unnamed_rows += 1
        st.max_cls = max(st.max_cls, new)
        if len(parts) == 5 or len(parts) == 6:
            if len(parts) == 6:
                st.conf_stripped += 1
            xc, yc, w, h = vals[1:5]
            out.append(f"{new} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}")
        else:
            out.append(" ".join([str(new)] + parts[1:]))
        st.rows_out += 1
    return ("\n".join(out) + "\n") if out else "", st


def _label_lines(label_path: Optional[str]) -> str:
    """A label file as plain training rows (no confidence column; seg/pose rows verbatim)."""
    if not label_path:
        return ""
    p = Path(label_path)
    if not p.is_file():
        return ""
    return transform_label_text(p.read_text(encoding="utf-8", errors="replace"))[0]


# ============================================================================== yaml

def yaml_key_for(split: str, split_map: Optional[Dict[str, str]] = None) -> str:
    """The data.yaml key a source split feeds: train/val/test aliases map directly, a custom name by its
    leading word ('test-1854-1860' -> test), anything else trains."""
    if split_map and split in split_map:
        return split_map[split]
    s = unicodedata.normalize("NFC", split or "").lower()
    if s in _SPLIT_ALIASES:
        return _SPLIT_ALIASES[s]
    head = re.split(r"[^0-9a-z]+", s)[0] if s else ""
    return _SPLIT_ALIASES.get(head, "train")


def data_yaml_text(names: List[str], splits: Iterable[str], path: Optional[str] = None,
                   split_dirs: Optional[Dict[str, List[str]]] = None, header: str = "",
                   warnings: Optional[List[str]] = None, split_map: Optional[Dict[str, str]] = None) -> str:
    """An Ultralytics data.yaml for an export. `path` is omitted unless given: Ultralytics then resolves
    the split folders against the yaml's own folder (a relative 'path: .' would be resolved against the
    trainer's working directory). train and val are always present — a missing one points at the other."""
    warnings = warnings if warnings is not None else []
    if split_dirs is None:
        split_dirs = defaultdict(list)
        named = [s for s in splits if s]
        for s in sorted(set(splits), key=lambda x: (x != "", x)):
            if s == "":
                split_dirs["train"].append("images/unsplit" if named else "images")
            else:
                split_dirs[yaml_key_for(s, split_map)].append(f"images/{s}")
    doc: Dict[str, Any] = {}
    if path:
        doc["path"] = path
    for k in YAML_KEYS:
        dirs = list(dict.fromkeys(split_dirs.get(k) or []))
        if dirs:
            doc[k] = dirs[0] if len(dirs) == 1 else dirs
    if "train" not in doc and "val" not in doc:
        doc["train"] = doc["val"] = "images"
    elif "val" not in doc:
        doc["val"] = doc["train"]
        warnings.append("This export has no validation split: data.yaml points val at the training images "
                        "(metrics will be optimistic). Use the split builder to make one.")
    elif "train" not in doc:
        doc["train"] = doc["val"]
        warnings.append("This export has no training split: data.yaml points train at the validation images.")
    order = {k: doc[k] for k in ("path", "train", "val", "test") if k in doc}
    order["nc"] = len(names)
    order["names"] = {i: n for i, n in enumerate(names)}
    body = yaml.safe_dump(order, sort_keys=False, allow_unicode=True)
    return (header + body) if header else body


# ============================================================================== plan

@dataclass
class PlanItem:
    id: int
    rel_path: str
    split: str
    seq: str
    abs_path: str
    label_path: Optional[str]
    has_label: bool
    n_boxes: int
    width: Optional[int]
    height: Optional[int]
    mtime: Optional[float]
    review: Optional[str]
    out_split: str = ""
    img_rel: str = ""
    lbl_rel: str = ""
    verbatim: bool = True
    renamed: bool = False
    size: Optional[int] = None          # file size at plan time (ZIP Content-Length)
    label_text: Optional[str] = None    # pre-rendered label rows (ZIP Content-Length)
    label_stats: Optional["LabelStats"] = None    # what rendering label_text did (shared, read-only)
    label_ident: Optional[tuple] = None           # (inode, size, mtime_ns) of the label file rendered


@dataclass
class ExportPlan:
    dataset_id: str
    items: List[PlanItem]
    classes: ClassPlan
    opts: ExportOptions
    filters: Dict[str, Any]
    matched: int = 0
    dropped_near_dups: int = 0
    near_dup_groups: int = 0
    near_dup_available: bool = True
    split_dirs: Dict[str, List[str]] = field(default_factory=dict)
    split_counts: Dict[str, int] = field(default_factory=dict)
    groups: Dict[str, int] = field(default_factory=dict)
    renamed: int = 0
    warnings: List[str] = field(default_factory=list)
    created_at: float = field(default_factory=time.time)
    missing: List[str] = field(default_factory=list)


_PLAN_SELECT = """
SELECT i.id, i.rel_path, i.split, i.seq, i.abs_path, i.label_path, i.has_label, i.n_boxes, i.width, i.height,
       i.mtime, r.status AS review
FROM images i LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path
"""


def snapshot_rows(dataset_id: str, filters: Dict[str, Any]) -> List[dict]:
    """The matching images, read once in one query (ordered by path): an export works from this snapshot,
    so edits that change filter membership while it runs cannot shift an OFFSET window."""
    where, params = build_where(dataset_id, filters)
    return db.query(f"{_PLAN_SELECT} WHERE {where} ORDER BY i.split, i.rel_path, i.id", params)


def _near_dup_groups(dataset_id: str, ids: Set[int]) -> Tuple[Dict[int, int], bool]:
    """image id -> near-duplicate group (only images in `ids`), and whether the QA hashes exist at all."""
    groups: Dict[int, int] = {}
    try:
        rows = db.query("SELECT image_id, detail FROM image_flags WHERE dataset_id=? AND code='near_dup'", (dataset_id,))
        hashed = db.query_one("SELECT 1 AS x FROM image_hash WHERE dataset_id=? LIMIT 1", (dataset_id,)) is not None
    except Exception:  # noqa: BLE001 — older database without the QA tables
        return {}, False
    for r in rows:
        if r["image_id"] not in ids:
            continue
        d = db.loads(r["detail"], {}) or {}
        g = d.get("group") if isinstance(d, dict) else None
        if g is not None:
            groups[r["image_id"]] = int(g)
    return groups, hashed or bool(rows)


def _keep_rank(it: PlanItem) -> tuple:
    """Which member of a near-duplicate group survives: approved first, then labeled, more boxes, oldest id."""
    return (0 if it.review == "approved" else 1, 0 if it.has_label else 1, -it.n_boxes, it.id)


class _UnionFind:
    def __init__(self):
        self.p: Dict[Any, Any] = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[rb] = ra


def _group_keys(items: List[PlanItem], sb: SplitBuilder, dup_group: Dict[int, int]) -> Dict[int, str]:
    rx = re.compile(sb.regex) if sb.group_by == "regex" and sb.regex else None
    keys: Dict[int, str] = {}
    for it in items:
        if sb.group_by == "image":
            k = f"#{it.id}"
        elif sb.group_by == "folder":
            k = "dir:" + str(PurePosixPath(it.rel_path).parent)
        elif rx is not None:
            m = rx.search(it.rel_path)
            k = ("re:" + (m.group(1) if m and m.groups() else m.group(0))) if m else f"#{it.id}"
        else:
            # The sequence alone, not split + sequence: frames of one video sitting in two source splits
            # (the leak this builder exists to fix) must become one group.
            k = f"seq:{it.seq}" if it.seq else f"#{it.id}"
        keys[it.id] = k
    if sb.group_by == "seq+dups" and dup_group:
        uf = _UnionFind()
        for it in items:
            uf.find(keys[it.id])
            g = dup_group.get(it.id)
            if g is not None:
                uf.union(keys[it.id], f"dup:{g}")
        keys = {iid: uf.find(k) for iid, k in keys.items()}
    return keys


def _class_presence(dataset_id: str, ids: Set[int], cp: ClassPlan) -> Dict[int, Counter]:
    """image id -> Counter(output class -> boxes), from the index (no label file reads)."""
    out: Dict[int, Counter] = defaultdict(Counter)
    if not ids:
        return out
    for r in db.query("SELECT image_id, cls, COUNT(*) AS n FROM boxes WHERE dataset_id=? GROUP BY image_id, cls", (dataset_id,)):
        if r["image_id"] in ids:
            new = cp.map(int(r["cls"]))
            if new is not None:
                out[r["image_id"]][new] += r["n"]
    return out


def assign_splits(items: List[PlanItem], sb: SplitBuilder, dup_group: Dict[int, int],
                  presence: Dict[int, Counter]) -> Dict[str, int]:
    """Assign whole groups (sequences by default) to train/val/test by ratio, reproducibly for a seed.

    Groups are shuffled with the seed, then placed largest-first (within the shuffled order of similar
    sizes) into the split furthest below its target share — of images, and with stratify also of each
    class's boxes — so frames of one sequence never straddle train and val."""
    keys = _group_keys(items, sb, dup_group)
    members: Dict[str, List[PlanItem]] = defaultdict(list)
    for it in items:
        members[keys[it.id]].append(it)
    order = sorted(members)
    rng = random.Random(sb.seed)
    rng.shuffle(order)
    # Large groups first (they are the hard ones to balance); the shuffle decides among similar sizes.
    order.sort(key=lambda k: -int(len(members[k]).bit_length()))
    splits = [s for s in YAML_KEYS if s in sb.ratios]
    n_total = len(items)
    cls_total: Counter = Counter()
    for it in items:
        cls_total.update(presence.get(it.id, Counter()))
    cur_img = {s: 0 for s in splits}
    cur_cls = {s: Counter() for s in splits}
    assigned: Dict[str, str] = {}
    for k in order:
        grp = members[k]
        g_cls: Counter = Counter()
        for it in grp:
            g_cls.update(presence.get(it.id, Counter()))

        def need(s: str, g_cls: Counter = g_cls) -> float:
            target = sb.ratios[s]
            score = (target * n_total - cur_img[s]) / max(1, n_total)
            if sb.stratify and g_cls:
                w = sum(g_cls.values())
                for c, n in g_cls.items():
                    score += (n / w) * (target * cls_total[c] - cur_cls[s][c]) / max(1, cls_total[c])
                score /= 2
            return score

        best = max(splits, key=lambda s: (need(s), sb.ratios[s]))
        assigned[k] = best
        cur_img[best] += len(grp)
        cur_cls[best].update(g_cls)
    # Every split with a share gets at least one group when there are enough groups to go round.
    for s in splits:
        if cur_img[s] == 0 and len(members) >= len(splits):
            donor = max(splits, key=lambda x: cur_img[x])
            cands = [k for k, v in assigned.items() if v == donor]
            if len(cands) > 1:
                k = min(cands, key=lambda x: (len(members[x]), x))
                assigned[k] = s
                cur_img[donor] -= len(members[k])
                cur_img[s] += len(members[k])
    for k, grp in members.items():
        for it in grp:
            it.out_split = assigned[k]
    return Counter(assigned.values())


def _strip_split(rel: str, split: str) -> str:
    if split and rel.startswith(split + "/"):
        return rel[len(split) + 1:]
    return rel


def _out_ext(it: PlanItem, spec: Optional[Tuple[str, int, int]]) -> Tuple[str, bool]:
    """(extension of the exported image, copied verbatim?)."""
    ext = PurePosixPath(it.rel_path).suffix.lower()
    if spec is None:
        return ext, True
    if spec[0] == "max":
        w, h = it.width, it.height
        if not w or not h:
            pr = probe(it.abs_path)
            w, h = pr.display_size
        if w and h and max(w, h) <= spec[1]:
            return ext, True          # already within the limit: no re-encode, no generation loss
    return ".jpg", False


def build_plan(dataset_id: str, names: List[str], filters: Dict[str, Any], opts: ExportOptions,
               with_files: bool = False) -> ExportPlan:
    """Everything an export will do, decided up front (cheap: one query plus index lookups).
    with_files also stats every image and renders every label file (for an exact ZIP Content-Length)."""
    cp = class_plan(names, opts)
    rows = snapshot_rows(dataset_id, filters)
    items = [PlanItem(id=r["id"], rel_path=r["rel_path"], split=r["split"] or "", seq=r["seq"] or "",
                      abs_path=r["abs_path"], label_path=r["label_path"], has_label=bool(r["has_label"]),
                      n_boxes=r["n_boxes"] or 0, width=r["width"], height=r["height"], mtime=r["mtime"],
                      review=r["review"]) for r in rows]
    if not opts.include_unlabeled:
        items = [it for it in items if it.has_label]
    plan = ExportPlan(dataset_id=dataset_id, items=items, classes=cp, opts=opts, filters=dict(filters), matched=len(items))

    ids = {it.id for it in items}
    dup_group: Dict[int, int] = {}
    if opts.drop_near_dups or (opts.split_builder and opts.split_builder.group_by == "seq+dups"):
        dup_group, plan.near_dup_available = _near_dup_groups(dataset_id, ids)
        if not plan.near_dup_available:
            plan.warnings.append("Near-duplicates have not been computed for this dataset yet — run the data "
                                 "quality check (Overview) first; no images were dropped as duplicates.")
    if opts.drop_near_dups and dup_group:
        by_group: Dict[int, List[PlanItem]] = defaultdict(list)
        for it in items:
            g = dup_group.get(it.id)
            if g is not None:
                by_group[g].append(it)
        drop: Set[int] = set()
        for grp in by_group.values():
            if len(grp) > 1:
                keep = min(grp, key=_keep_rank)
                drop.update(it.id for it in grp if it is not keep)
        plan.near_dup_groups = sum(1 for g in by_group.values() if len(g) > 1)
        plan.dropped_near_dups = len(drop)
        items = [it for it in items if it.id not in drop]
        plan.items = items

    sb = opts.split_builder
    if sb:
        presence = _class_presence(dataset_id, {it.id for it in items}, cp) if sb.stratify else {}
        plan.groups = dict(assign_splits(items, sb, dup_group, presence))
    else:
        named = any(it.split for it in items)
        for it in items:
            it.out_split = it.split if it.split else ("unsplit" if named else "")

    # Output names: images/<split>/<path inside the split>; collisions (a.jpg + a.png under resize, or two
    # source splits merged by the split builder) get a suffix instead of overwriting each other.
    taken: Set[str] = set()
    for it in items:
        inner = _strip_split(it.rel_path, it.split)
        ext, it.verbatim = _out_ext(it, opts.resize)
        stem = str(PurePosixPath(inner).with_suffix(""))
        base = f"{it.out_split}/" if it.out_split else ""
        cand, n = stem, 0
        while True:
            key = unicodedata.normalize("NFC", f"{base}{cand}").casefold()
            if key not in taken:
                break
            n += 1
            src_ext = PurePosixPath(inner).suffix.lstrip(".").lower()
            cand = f"{stem}_{src_ext}" if n == 1 and src_ext and src_ext != ext.lstrip(".") else f"{stem}_{n}"
        taken.add(key)
        if cand != stem:
            it.renamed = True
            plan.renamed += 1
        it.img_rel = f"images/{base}{cand}{ext}"
        it.lbl_rel = f"labels/{base}{cand}.txt"

    counts = Counter(it.out_split for it in items)
    plan.split_counts = dict(counts)
    split_dirs: Dict[str, List[str]] = defaultdict(list)
    for s in sorted(counts, key=lambda x: (x in ("", "unsplit"), x)):
        d = f"images/{s}" if s else "images"
        key = s if sb else ("train" if s in ("", "unsplit") else yaml_key_for(s, opts.split_map))
        split_dirs[key].append(d)
    plan.split_dirs = dict(split_dirs)
    if plan.renamed:
        plan.warnings.append(f"{plan.renamed:,} images share a name with another exported image and get a suffix "
                             "(e.g. a_png.jpg); their label files are renamed to match.")
    if with_files:
        _attach_files(plan, cp, opts.bad_class == "drop")
    return plan


class _RenderedLabels:
    """Rendered label files, keyed by the file's identity (path, inode, size, mtime_ns) and the class
    transform, so planning the same export again (every Download click plans) only stats the label files
    instead of reading and rewriting each one. A file edited since — by Fovea (atomic replace: new inode) or
    by anything else (new mtime/size) — gets a new key and is read again. LRU, bounded by bytes."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self._d: "OrderedDict[tuple, Tuple[str, LabelStats, int]]" = OrderedDict()
        self._bytes = 0
        self._max = max_bytes
        self._lock = threading.Lock()

    def clear(self) -> None:
        with self._lock:
            self._d.clear()
            self._bytes = 0

    def render(self, path: str, cp: ClassPlan, nc: int, drop_unnamed: bool,
               sig: tuple) -> Tuple[Optional[str], LabelStats, Optional[tuple]]:
        """(text, stats, file identity); text None when the file has disappeared. stats is shared: read only."""
        try:
            st = os.stat(path)
        except OSError:
            return None, LabelStats(), None
        ident = _ident(st)
        with self._lock:
            hit = self._d.get((path, ident, sig))
            if hit is not None:
                self._d.move_to_end((path, ident, sig))
                return hit[0], hit[1], ident
        try:
            with open(path, "rb") as fh:
                data = fh.read()
                ident = _ident(os.fstat(fh.fileno()))     # the identity of the bytes actually read
        except OSError:
            return None, LabelStats(), None
        text, stats = transform_label_text(data.decode("utf-8", errors="replace"), cp, nc, drop_unnamed)
        cost = len(text) + len(path) + 300
        with self._lock:
            if (path, ident, sig) not in self._d:
                self._d[(path, ident, sig)] = (text, stats, cost)
                self._bytes += cost
                while self._bytes > self._max and self._d:
                    self._bytes -= self._d.popitem(last=False)[1][2]
        return text, stats, ident


def _ident(st: os.stat_result) -> tuple:
    return st.st_ino, st.st_size, st.st_mtime_ns


RENDERED_LABELS = _RenderedLabels()
_FILE_WORKERS = 8
_SLOW_FILE_S = 0.0003      # per item (stat + label): above this the storage is remote-like and threads pay off
_CHUNK = 256


def _map_chunks(run, items: List[Any]) -> Iterator[Tuple[Any, Any]]:
    """(item, result) for run(chunk) over chunks of items, in order. The first chunk runs inline and times the
    storage: a local disk answers in microseconds, where threads only add GIL contention, while a network
    share costs a round trip per file, which threads overlap."""
    chunks = [items[i:i + _CHUNK] for i in range(0, len(items), _CHUNK)]
    if not chunks:
        return
    t = time.perf_counter()
    first = run(chunks[0])
    slow = (time.perf_counter() - t) / len(chunks[0]) > _SLOW_FILE_S
    yield from zip(chunks[0], first)
    if len(chunks) == 1:
        return
    if not slow:
        for c in chunks[1:]:
            yield from zip(c, run(c))
        return
    pool = ThreadPoolExecutor(max_workers=min(_FILE_WORKERS, len(chunks) - 1), thread_name_prefix="fovea-plan")
    try:
        for c, res in zip(chunks[1:], pool.map(run, chunks[1:])):
            yield from zip(c, res)
    finally:
        pool.shutdown(wait=True, cancel_futures=True)       # a consumer that stops early (all()) skips the rest


def _attach_files(plan: ExportPlan, cp: ClassPlan, drop_unnamed: bool) -> None:
    """Stat every image and render every label file (exact ZIP Content-Length), in plan order."""
    nc = len(cp.names)
    sig = (tuple(sorted(cp.mapping.items())), nc, drop_unnamed)

    def run(chunk: List[PlanItem]) -> List[Optional[tuple]]:
        out: List[Optional[tuple]] = []
        for it in chunk:
            try:
                size = os.stat(it.abs_path).st_size
            except OSError:
                out.append(None)
                continue
            out.append((size,) + (RENDERED_LABELS.render(it.label_path, cp, nc, drop_unnamed, sig)
                                  if it.has_label and it.label_path else (None, None, None)))
        return out

    for it, r in _map_chunks(run, plan.items):
        if r is None:
            plan.missing.append(it.rel_path)
        else:
            it.size, it.label_text, it.label_stats, it.label_ident = r


def _files_unchanged(plan: ExportPlan) -> bool:
    """Every planned image still has its planned size and every rendered label file its identity."""
    def run(chunk: List[PlanItem]) -> List[bool]:
        out = []
        for it in chunk:
            if it.size is None:
                out.append(True)
                continue
            try:
                ok = os.stat(it.abs_path).st_size == it.size
                if ok and it.label_ident is not None:
                    ok = _ident(os.stat(it.label_path)) == it.label_ident
            except OSError:
                ok = False
            out.append(ok)
        return out

    return all(ok for _it, ok in _map_chunks(run, plan.items))


def _render_label(it: PlanItem, cp: ClassPlan, nc: int, drop_unnamed: bool) -> Tuple[Optional[str], LabelStats]:
    """(label text, stats); None when the label file has disappeared since the scan."""
    try:
        text = Path(it.label_path).read_text(encoding="utf-8", errors="replace") if it.label_path else None
    except OSError:
        text = None
    if text is None:
        return None, LabelStats()
    return transform_label_text(text, cp, nc, drop_unnamed)


# More output classes than any detector trains on: one stray row such as '2000000000 0.5 0.5 0.1 0.1' used to make
# 'extend' build two billion class_N names (the preview request ran the server out of memory).
MAX_EXPORT_CLASSES = 100_000


def final_names(plan: ExportPlan, max_cls: int) -> Tuple[List[str], List[int]]:
    """Output names, extended with class_<id> for ids beyond the list (bad_class='extend')."""
    names = list(plan.classes.names)
    extended = []
    if plan.opts.bad_class == "extend" and max_cls >= max(len(names), MAX_EXPORT_CLASSES):
        raise ValueError(f"Class id {max_cls:,} is far beyond the class list ({len(names):,} names), so keeping it "
                         f"would write a data.yaml with {max_cls + 1:,} classes. Choose 'Drop those boxes' for "
                         "unnamed class ids, or fix those rows (Explore → Issues → Invalid class id).")
    if plan.opts.bad_class == "extend" and max_cls >= len(names):
        for i in range(len(names), max_cls + 1):
            names.append(f"class_{i}")
            extended.append(i)
    return names, extended


def _yaml_header(plan: ExportPlan, zip_mode: bool) -> str:
    when = time.strftime("%Y-%m-%d %H:%M", time.localtime(plan.created_at))
    lines = [f"# Exported by Fovea {__version__} on {when} from dataset '{plan.dataset_id}'."]
    if zip_mode:
        lines.append("# Paths are relative to this file (Ultralytics). For YOLOv5, add: path: /absolute/path/to/this/folder")
    return "\n".join(lines) + "\n"


def _manifest(plan: ExportPlan, result: dict, files: Optional[List[str]] = None, status: str = "done") -> str:
    doc = {
        "fovea_version": __version__, "status": status, "dataset_id": plan.dataset_id,
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z", time.localtime(plan.created_at)),
        "filters": plan.filters, "options": plan.opts.describe(), "classes": result.get("names"),
        "counts": {k: result.get(k) for k in ("images", "labels", "missing", "failed", "dropped_near_dups",
                                               "renamed", "split_counts", "label_rows", "converted_8bit")},
        "skipped": result.get("skipped", [])[:1000],
    }
    if files is not None:
        doc["files"] = files
    return json.dumps(doc, ensure_ascii=False, indent=1) + "\n"


# ============================================================================== image production

def _encode(it: PlanItem, spec: Tuple[str, int, int]) -> Tuple[bytes, bool]:
    """(JPEG bytes, high-bit source converted to 8-bit?). Upright per EXIF (the tag is not carried over),
    high-bit-depth sources stretched for display, colour profile kept."""
    pr = probe(it.abs_path)
    if spec[0] == "max":
        im, icc = decode_display(it.abs_path, max_side=spec[1], pr=pr, hq=True)
    else:
        im, icc = decode_display(it.abs_path, exact=(spec[1], spec[2]), pr=pr, hq=True)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=92, **({"icc_profile": icc} if icc else {}))
    return buf.getvalue(), pr.high_bit


def _ordered_map(fn, items: List[PlanItem], workers: int, stop: Optional[threading.Event] = None):
    """Yield (item, future) in input order while up to 2*workers items are processed ahead."""
    if workers <= 1:
        for it in items:
            if stop is not None and stop.is_set():
                return
            fut = _Done(fn, it)
            yield it, fut
        return
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="fovea-export") as ex:
        window: deque = deque()
        it_iter = iter(items)
        for it in it_iter:
            window.append((it, ex.submit(fn, it)))
            if len(window) >= workers * 2:
                break
        try:
            while window:
                it, fut = window.popleft()
                yield it, fut
                if stop is not None and stop.is_set():
                    return
                nxt = next(it_iter, None)
                if nxt is not None:
                    window.append((nxt, ex.submit(fn, nxt)))
        finally:
            for _it, fut in window:
                fut.cancel()


class _Done:
    """A synchronous stand-in for a Future."""

    def __init__(self, fn, it):
        self._fn, self._it = fn, it

    def result(self):
        return self._fn(self._it)


def _workers(spec) -> int:
    return 1 if spec is None else max(1, min(4, (os.cpu_count() or 2) - 1))


# ============================================================================== ZIP

class _ExportAborted(Exception):
    """The download was abandoned: stop producing."""


class _ByteQueue:
    """Hand-off between the zip producer thread and the HTTP response, bounded by BYTES (64 MB) rather than
    by item count: 64 queued 50 MB TIFFs used to pin GBs of memory."""

    def __init__(self, max_bytes: int = 64 * 1024 * 1024):
        self._q: deque = deque()
        self._bytes = 0
        self._max = max_bytes
        self._cv = threading.Condition()

    def put(self, chunk: Optional[bytes], stop: threading.Event) -> None:
        with self._cv:
            while chunk is not None and self._bytes > 0 and self._bytes + len(chunk) > self._max:
                if stop.is_set():
                    raise _ExportAborted()
                self._cv.wait(0.5)
            if stop.is_set() and chunk is not None:
                raise _ExportAborted()
            self._q.append(chunk)
            self._bytes += len(chunk) if chunk else 0
            self._cv.notify_all()

    def get(self) -> Optional[bytes]:
        with self._cv:
            while not self._q:
                self._cv.wait(0.5)
            chunk = self._q.popleft()
            self._bytes -= len(chunk) if chunk else 0
            self._cv.notify_all()
            return chunk

    def drain(self) -> None:
        with self._cv:
            self._q.clear()
            self._bytes = 0
            self._cv.notify_all()


class _QueueFile:
    """Write-only sink coalescing zip writes into large chunks for the HTTP response.

    zipfile issues one tiny write per header field; sending each as its own HTTP chunk cost a thread hop
    and an ASGI send (~3 ms) per 30-byte write (8 MB/s). Chunks start small (the download begins at once)
    and grow to 1 MB. Once `stop` is set every write raises, so an abandoned export unwinds at once."""

    STEPS = (32 * 1024, 256 * 1024, 1024 * 1024)

    def __init__(self, q: _ByteQueue, stop: threading.Event):
        self._q = q
        self._stop = stop
        self._pos = 0
        self._buf = bytearray()
        self._sent = 0

    def _limit(self) -> int:
        return self.STEPS[min(self._sent, len(self.STEPS) - 1)]

    def write(self, data) -> int:
        if self._stop.is_set():
            raise _ExportAborted()
        n = len(data)
        self._buf += data
        self._pos += n
        if len(self._buf) >= self._limit():
            self._push()
        return n

    def _push(self) -> None:
        if self._buf:
            chunk = bytes(self._buf)
            self._buf.clear()
            self._q.put(chunk, self._stop)
            self._sent += 1

    def finish(self) -> None:
        self._push()

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return False

    def flush(self) -> None:
        pass


class _CountingFile:
    """Discards bytes, counts them: a dry run of the archive gives the exact Content-Length."""

    def __init__(self):
        self._pos = 0

    def write(self, data) -> int:
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return False

    def flush(self) -> None:
        pass


def _zinfo(name: str, mtime: Optional[float], size: int = 0) -> zipfile.ZipInfo:
    t = time.localtime(mtime if mtime and mtime > 315532800 else time.time())   # zip dates start in 1980
    zi = zipfile.ZipInfo(name, date_time=t[:6])
    zi.compress_type = zipfile.ZIP_STORED
    zi.external_attr = 0o644 << 16
    zi.file_size = size
    return zi


_ZERO = bytes(1024 * 1024)


def _zip_put_file(zf: zipfile.ZipFile, name: str, src: str, size: int, mtime: Optional[float]) -> None:
    """Stream a file into the archive in 1 MB slices (never the whole file in memory). The size must not
    change while copying: a file edited mid-export would corrupt the announced length."""
    zi = _zinfo(name, mtime, size)
    with open(src, "rb") as fh, zf.open(zi, "w") as dst:
        left = size
        while left > 0:
            chunk = fh.read(min(left, 1024 * 1024))
            if not chunk:
                raise OSError(f"{src} shrank while it was being exported")
            dst.write(chunk)
            left -= len(chunk)


def _zip_put_bytes(zf: zipfile.ZipFile, name: str, data: bytes, mtime: Optional[float] = None) -> None:
    zf.writestr(_zinfo(name, mtime, len(data)), data)


def _dry_size(plan: ExportPlan, yaml_text: Optional[str], manifest_text: Optional[str],
              card_text: Optional[str] = None) -> int:
    """Exact archive length for a no-resize export: replay the archive with zero bytes of the same sizes."""
    sink = _CountingFile()
    with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED) as zf:
        for it in plan.items:
            if it.size is None:
                continue
            zi = _zinfo(it.img_rel, it.mtime, it.size)
            with zf.open(zi, "w") as dst:
                left = it.size
                while left > 0:
                    n = min(left, len(_ZERO))
                    dst.write(memoryview(_ZERO)[:n])
                    left -= n
            if it.label_text is not None:
                _zip_put_bytes(zf, it.lbl_rel, it.label_text.encode("utf-8"), it.mtime)
        if yaml_text is not None:
            _zip_put_bytes(zf, "data.yaml", yaml_text.encode("utf-8"), plan.created_at)
        if card_text is not None:
            _zip_put_bytes(zf, CARD, card_text.encode("utf-8"), plan.created_at)
        if manifest_text is not None:
            _zip_put_bytes(zf, MANIFEST, manifest_text.encode("utf-8"), plan.created_at)
    return sink.tell()


def zip_length(members: Iterable[Tuple[str, int]]) -> int:
    """Exact length of the stored (uncompressed) archive _run_zip writes for these (name, size) members, in
    order — arithmetic instead of a dry run (which CRC'd every byte of every image). The sink is not
    seekable, so every member is followed by a data descriptor; ZIP64 extras follow zipfile's own rules:
    local header when size*1.05 > ZIP64_LIMIT (zipfile reserves room for compression growth), central
    directory when the size or the header offset exceeds it, end records when the count or the directory
    does. _length_model_ok() checks this model against zipfile itself once per process."""
    limit = zipfile.ZIP64_LIMIT
    pos = cdir = count = 0
    for name, size in members:
        n = len(name.encode("utf-8"))
        local64 = size * 1.05 > limit
        k = (2 if size > limit else 0) + (1 if pos > limit else 0)
        cdir += 46 + n + (4 + 8 * k if k else 0)
        pos += 30 + n + (20 if local64 else 0) + size + (24 if local64 else 16)
        count += 1
    end = 22
    if count > zipfile.ZIP_FILECOUNT_LIMIT or pos > limit or cdir > limit:
        end += 56 + 20                       # ZIP64 end of central directory record + locator
    return pos + cdir + end


def _prepared_members(plan: ExportPlan, yaml_text: Optional[str], manifest_text: Optional[str],
                      card_text: Optional[str]) -> Iterator[Tuple[str, int]]:
    """The members stream_prepared writes, in the same order."""
    for it in plan.items:
        if it.size is None:
            continue
        yield it.img_rel, it.size
        if it.label_text is not None:
            yield it.lbl_rel, len(it.label_text.encode("utf-8"))
    for name, text in (("data.yaml", yaml_text), (CARD, card_text), (MANIFEST, manifest_text)):
        if text is not None:
            yield name, len(text.encode("utf-8"))


_MODEL_CHECK: Dict[str, bool] = {}


def _length_model_ok() -> bool:
    """zip_length() agrees with this Python's zipfile (a future zipfile writing extra fields would otherwise
    announce a wrong Content-Length and fail every download); False falls back to the dry run."""
    ok = _MODEL_CHECK.get("ok")
    if ok is None:
        sample = [("images/a.jpg", 1000), ("labels/한글 é.txt", 0), ("images/" + "b" * 300 + ".png", 70000)]
        sink = _CountingFile()
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, size in sample:
                _zip_put_bytes(zf, name, bytes(size), None)
        ok = _MODEL_CHECK["ok"] = sink.tell() == zip_length(sample)
        if not ok:
            log.warning("ZIP length model disagrees with this Python's zipfile: using dry runs for exports")
    return ok


@dataclass
class ZipPrepared:
    """A no-resize ZIP whose every member is known: streamed with an exact Content-Length."""
    plan: ExportPlan
    yaml_text: Optional[str]
    manifest_text: Optional[str]
    length: int
    result: dict
    card_text: Optional[str] = None


PREPARED_TTL = 60.0
_PREPARED: "OrderedDict[tuple, Tuple[float, ZipPrepared]]" = OrderedDict()
_PREPARED_LOCK = threading.Lock()


def _prepared_key(dataset_id: str, names: List[str], filters: Dict[str, Any], opts: ExportOptions,
                  include_manifest: bool) -> Optional[tuple]:
    """What a prepared archive depends on: the request, and the index (label writes, scans, reviews, QA flags,
    AI rows — imagesq.data_version) plus the class cards the dataset card quotes. Files on disk are
    re-checked separately (_files_unchanged)."""
    version = getattr(imagesq, "data_version", None)
    if version is None:
        return None
    try:
        cards = db.query_one("SELECT COUNT(*) AS n, MAX(updated_at) AS t FROM class_cards WHERE dataset_id=?",
                             (dataset_id,)) or {}
        return (dataset_id, tuple(names), json.dumps(filters, sort_keys=True, default=str),
                json.dumps(opts.describe(), sort_keys=True, default=str), include_manifest,
                version(dataset_id), tuple(cards.values()))
    except Exception:  # noqa: BLE001 — an older database: just plan every time
        return None


def prepare_zip(dataset_id: str, names: List[str], filters: Dict[str, Any], opts: ExportOptions,
                include_manifest: bool = True) -> Optional[ZipPrepared]:
    """Plan a ZIP up front. Returns None when the archive length cannot be known (resize re-encodes).
    A plan is reused for PREPARED_TTL seconds while nothing it depends on changed (a second Download click,
    or the same export from Explore and the Export tab)."""
    if opts.resize is not None:
        return None
    key = _prepared_key(dataset_id, names, filters, opts, include_manifest)
    now = time.monotonic()
    with _PREPARED_LOCK:
        for k in [k for k, (t, _p) in _PREPARED.items() if now - t > PREPARED_TTL]:
            del _PREPARED[k]
        hit = _PREPARED.get(key) if key is not None else None
    if hit is not None and _files_unchanged(hit[1].plan):
        return hit[1]
    prep = _prepare_zip(dataset_id, names, filters, opts, include_manifest)
    if key is not None:
        with _PREPARED_LOCK:
            _PREPARED[key] = (now, prep)
            while len(_PREPARED) > 2:                 # a plan of 45k images holds a few tens of MB
                _PREPARED.popitem(last=False)
    return prep


def _prepare_zip(dataset_id: str, names: List[str], filters: Dict[str, Any], opts: ExportOptions,
                 include_manifest: bool) -> ZipPrepared:
    plan = build_plan(dataset_id, names, filters, opts, with_files=True)
    stats = LabelStats()
    n_lbl = 0
    for it in plan.items:
        if it.size is not None and it.label_text is not None:
            n_lbl += 1
            stats.add(it.label_stats)
    out_names, extended = final_names(plan, stats.max_cls)
    split_counts = Counter(it.out_split for it in plan.items if it.size is not None)
    result = _result_base(plan, out_names, extended)
    result.update(images=sum(1 for it in plan.items if it.size is not None), labels=n_lbl,
                  missing=len(plan.missing), split_counts=dict(split_counts),
                  skipped=[{"path": p, "reason": "missing"} for p in plan.missing], label_rows=stats.__dict__)
    yaml_text = data_yaml_text(out_names, [], split_dirs=_present_dirs(plan, split_counts),
                               header=_yaml_header(plan, True), warnings=result["warnings"]) if opts.include_yaml else None
    card_text = dataset_card(dataset_id, plan, out_names=out_names, skipped=result["skipped"], result=result) \
        if opts.dataset_card else None
    manifest_text = _manifest(plan, result) if include_manifest else None
    if _length_model_ok():
        length = zip_length(_prepared_members(plan, yaml_text, manifest_text, card_text))
    else:
        length = _dry_size(plan, yaml_text, manifest_text, card_text)
    return ZipPrepared(plan, yaml_text, manifest_text, length, result, card_text)


def _present_dirs(plan: ExportPlan, counts: Counter) -> Dict[str, List[str]]:
    """Split folders that really received images (a split whose every file failed is left out)."""
    present = {f"images/{s}" if s else "images" for s, n in counts.items() if n}
    return {k: [d for d in v if d in present] for k, v in plan.split_dirs.items() if any(d in present for d in v)}


def _result_base(plan: ExportPlan, out_names: List[str], extended: List[int]) -> dict:
    warnings = list(plan.warnings)
    if extended:
        warnings.append(f"Label rows use class ids without a name ({', '.join(map(str, extended[:10]))}"
                        f"{'…' if len(extended) > 10 else ''}): data.yaml lists them as class_<id> so trainers accept them.")
    return {"names": out_names, "extended_classes": extended, "dropped_near_dups": plan.dropped_near_dups,
            "near_dup_groups": plan.near_dup_groups, "renamed": plan.renamed, "matched": plan.matched,
            "groups": plan.groups, "warnings": warnings, "missing": 0, "failed": 0, "skipped": []}


def stream_prepared(prep: ZipPrepared) -> Iterator[bytes]:
    """Stream a prepared archive; it must come out exactly prep.length bytes long, or the stream aborts."""
    plan = prep.plan

    def produce(zf: zipfile.ZipFile, stop: threading.Event):
        for it in plan.items:
            if stop.is_set():
                raise _ExportAborted()
            if it.size is None:
                continue
            try:
                _zip_put_file(zf, it.img_rel, it.abs_path, it.size, it.mtime)
            except OSError as e:
                raise RuntimeError(f"{it.rel_path} changed or disappeared during the export ({e.strerror or e})")
            if it.label_text is not None:
                _zip_put_bytes(zf, it.lbl_rel, it.label_text.encode("utf-8"), it.mtime)
        if prep.yaml_text is not None:
            _zip_put_bytes(zf, "data.yaml", prep.yaml_text.encode("utf-8"), plan.created_at)
        if prep.card_text is not None:
            _zip_put_bytes(zf, CARD, prep.card_text.encode("utf-8"), plan.created_at)
        if prep.manifest_text is not None:
            _zip_put_bytes(zf, MANIFEST, prep.manifest_text.encode("utf-8"), plan.created_at)

    sent = 0
    for chunk in _run_zip(produce):
        sent += len(chunk)
        yield chunk
    if sent != prep.length:
        raise RuntimeError(f"ZIP export failed part-way: {sent} bytes written, {prep.length} announced")


def _run_zip(produce) -> Iterator[bytes]:
    """Run produce(zipfile, stop) on a producer thread and yield the archive bytes as they come."""
    q = _ByteQueue()
    stop = threading.Event()
    failure: List[BaseException] = []

    def run():
        sink = _QueueFile(q, stop)
        try:
            with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
                produce(zf, stop)
            sink.finish()
        except _ExportAborted:
            pass
        except BaseException as e:  # noqa: BLE001 — reported to the consumer, which aborts the response
            failure.append(e)
        finally:
            try:
                q.put(None, stop)        # end-of-stream marker
            except _ExportAborted:
                pass

    t = threading.Thread(target=run, daemon=True, name="fovea-zip")
    t.start()
    try:
        while True:
            chunk = q.get()
            if chunk is None:
                break
            yield chunk
        if failure:
            # Raising mid-stream drops the connection, so the browser marks the download failed. A clean
            # end here would hand the user a truncated archive that looks complete.
            raise RuntimeError(f"ZIP export failed part-way: {failure[0]}")
    finally:
        stop.set()                            # the client left (or we are done): let the producer unwind
        q.drain()


def stream_zip(dataset_id: str, names: List[str], filters: Dict[str, Any], resize: Optional[dict] = None,
               include_unlabeled: bool = True, include_yaml: bool = True, opts: Optional[ExportOptions] = None,
               include_manifest: bool = False) -> Iterator[bytes]:
    """Stream a ZIP (images + labels + data.yaml [+ fovea-export.json]) built while it downloads.
    Undecodable images are skipped and listed in the manifest."""
    if opts is None:
        opts = ExportOptions(resize=_resize_spec(resize), include_unlabeled=include_unlabeled, include_yaml=include_yaml)
    spec = opts.resize
    holder: Dict[str, Any] = {}

    def produce(zf: zipfile.ZipFile, stop: threading.Event):
        plan = build_plan(dataset_id, names, filters, opts)
        nc = len(plan.classes.names)
        stats = LabelStats()
        n_img = n_lbl = converted = 0
        skipped: List[dict] = []
        counts: Counter = Counter()

        def work(it: PlanItem):
            if not os.path.isfile(it.abs_path):
                return ("missing", None)
            if it.verbatim:
                return ("file", os.path.getsize(it.abs_path))
            try:
                return ("bytes", _encode(it, spec))
            except ImageTooLarge as e:
                return ("failed", str(e))
            except Exception as e:  # noqa: BLE001 — an unreadable image is skipped and reported, not fatal
                return ("failed", f"could not be decoded ({type(e).__name__})")

        for it, fut in _ordered_map(work, plan.items, _workers(spec), stop):
            if stop.is_set():
                raise _ExportAborted()
            kind, val = fut.result()
            if kind in ("missing", "failed"):
                skipped.append({"path": it.rel_path, "reason": "missing" if kind == "missing" else val})
                continue
            if kind == "file":
                _zip_put_file(zf, it.img_rel, it.abs_path, val, it.mtime)
            else:
                data, deep = val
                converted += deep
                _zip_put_bytes(zf, it.img_rel, data, it.mtime)
            n_img += 1
            counts[it.out_split] += 1
            if it.has_label and it.label_path:
                text, st = _render_label(it, plan.classes, nc, opts.bad_class == "drop")
                if text is not None:
                    _zip_put_bytes(zf, it.lbl_rel, text.encode("utf-8"), it.mtime)
                    stats.add(st)
                    n_lbl += 1
        out_names, extended = final_names(plan, stats.max_cls)
        result = _result_base(plan, out_names, extended)
        result.update(images=n_img, labels=n_lbl, converted_8bit=converted, split_counts=dict(counts), skipped=skipped,
                      missing=sum(1 for s in skipped if s["reason"] == "missing"),
                      failed=sum(1 for s in skipped if s["reason"] != "missing"), label_rows=stats.__dict__)
        if opts.include_yaml:
            zf_yaml = data_yaml_text(out_names, [], split_dirs=_present_dirs(plan, counts),
                                     header=_yaml_header(plan, True), warnings=result["warnings"])
            _zip_put_bytes(zf, "data.yaml", zf_yaml.encode("utf-8"), plan.created_at)
        if opts.dataset_card:
            card = dataset_card(dataset_id, plan, out_names=out_names, skipped=skipped, result=result)
            _zip_put_bytes(zf, CARD, card.encode("utf-8"), plan.created_at)
        if include_manifest:
            _zip_put_bytes(zf, MANIFEST, _manifest(plan, result).encode("utf-8"), plan.created_at)
        holder["result"] = result

    yield from _run_zip(produce)


# ============================================================================== copy to folder

def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def _fovea_dirs() -> List[Tuple[Path, str]]:
    from .. import config          # read at call time: tests and the CLI may point these elsewhere
    return [(config.DATA_DIR, "Fovea's own data folder"), (config.MODEL_DIR, "Fovea's model folder")]


def refuse_fovea_dirs(path: Any, what: str, contains: bool = False) -> None:
    """ValueError when `path` lies in Fovea's data folder (index, settings, API keys, caches) or its model folder —
    or, with contains=True, holds one of them. An export copied into DATA_DIR turned it into a dataset that could
    then be registered and its secrets.json read through the file endpoints."""
    p = Path(os.path.realpath(path))
    for d, name in _fovea_dirs():
        rd = Path(os.path.realpath(d))
        if _inside(p, rd):
            raise ValueError(f"{what} is inside {name} ({to_host(rd)}). Choose a folder outside it.")
        if contains and _inside(rd, p):
            raise ValueError(f"{what} contains {name} ({to_host(rd)}). Choose a folder outside it.")


def check_copy_target(dataset_id: str, target_dir: str) -> Path:
    """Refuse an export folder that overlaps the dataset. Exporting a dataset into itself makes every
    destination path a source path: a resize re-encodes the originals in place and the label pass
    rewrites the source labels — all while the job reports success."""
    target = resolve(target_dir)
    if not target.is_absolute():
        if is_windows_path(target_dir):
            raise ValueError("Windows paths only work when Fovea runs in Docker with HOST_PATH set to the folder "
                             "that contains them")
        raise ValueError("Target must be an absolute path")
    target = Path(os.path.realpath(target))
    refuse_fovea_dirs(target, "The export folder", contains=True)
    row = db.query_one("SELECT root, layout FROM datasets WHERE id=?", (dataset_id,))
    if not row:
        raise ValueError("Dataset not found")
    protected = [row["root"]]
    for s in (db.loads(row["layout"], {}) or {}).get("sources", []):
        protected += [s.get("img_dir"), s.get("label_dir")]
        if s.get("list_file"):
            protected.append(str(Path(s["list_file"]).parent))
    for p in dict.fromkeys(x for x in protected if x):     # ordered: the dataset root is reported first
        rp = Path(os.path.realpath(p))
        if _inside(target, rp) or _inside(rp, target):
            raise ValueError(f"The export folder overlaps this dataset ({to_host(rp)}). Choose a folder outside it.")
    return target


_OS_LITTER = {".DS_Store", "Thumbs.db", "desktop.ini", ".localized"}     # a fresh Finder/Explorer folder is still empty


def inspect_target(target: Path) -> dict:
    """What is already in an export folder: {exists, entries (capped at 1000), previous_export}."""
    info = {"path": to_host(target), "exists": target.exists(), "is_dir": target.is_dir(), "entries": 0,
            "previous_export": False}
    if target.exists() and not target.is_dir():
        info["error"] = "A file with this name already exists"
        return info
    if target.is_dir():
        try:
            with os.scandir(target) as it:
                for e in it:
                    if e.name in _OS_LITTER:
                        continue
                    info["entries"] += 1
                    if e.name == MANIFEST:
                        info["previous_export"] = True
                    if info["entries"] >= 1000:
                        break
        except OSError as e:
            info["error"] = f"Cannot read the folder ({e.strerror or e})"
    return info


_FATAL_WRITE = {errno.ENOSPC, errno.EACCES, errno.EROFS, errno.EPERM, getattr(errno, "EDQUOT", -1), errno.EIO}


def _atomic_write(dst: Path, data: bytes) -> None:
    tmp = dst.with_name(f".{dst.name}.fovea-tmp")
    try:
        with open(tmp, "wb") as fh:
            fh.write(data)
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _atomic_copy(src: Path, dst: Path) -> None:
    tmp = dst.with_name(f".{dst.name}.fovea-tmp")
    try:
        shutil.copyfile(src, tmp)
        try:
            shutil.copystat(src, tmp)
        except OSError:
            pass
        os.replace(tmp, dst)
    finally:
        if tmp.exists():
            try:
                tmp.unlink()
            except OSError:
                pass


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def _write_failure(e: OSError, dst: Path) -> JobError:
    reason = {errno.ENOSPC: "the disk is full", errno.EROFS: "the disk is read-only",
              errno.EACCES: "permission denied", errno.EPERM: "permission denied",
              getattr(errno, "EDQUOT", -1): "the disk quota is exceeded"}.get(e.errno, e.strerror or str(e))
    return JobError(f"Export stopped: cannot write {to_host(dst)} — {reason}. Files written so far were kept.")


def copy_subset(job: Job, dataset_id: str, names: List[str], filters: Dict[str, Any], target_dir: str,
                resize: Optional[dict] = None, include_unlabeled: bool = True, include_yaml: bool = True,
                opts: Optional[ExportOptions] = None) -> dict:
    if opts is None:
        opts = ExportOptions(resize=_resize_spec(resize), include_unlabeled=include_unlabeled, include_yaml=include_yaml)
    spec = opts.resize
    target = check_copy_target(dataset_id, target_dir)
    info = inspect_target(target)
    if info.get("error"):
        raise ValueError(f"{info['path']}: {info['error']}")
    if info["entries"] and not opts.overwrite:
        raise ValueError(f"The folder {info['path']} is not empty ({info['entries']:,}{'+' if info['entries'] >= 1000 else ''} "
                         "items). Choose another folder, or confirm to replace the previous export.")
    old_files: List[str] = []
    if info["previous_export"]:
        try:
            old = json.loads((target / MANIFEST).read_text(encoding="utf-8"))
            old_files = [f for f in old.get("files", []) if isinstance(f, str)]
        except (OSError, ValueError):
            old_files = []
    try:
        target.mkdir(parents=True, exist_ok=True)
    except OSError as e:
        raise _write_failure(e, target)

    plan = build_plan(dataset_id, names, filters, opts)
    total = len(plan.items)
    nc = len(plan.classes.names)
    job.update(0, total, f"Copying 0/{total:,}")
    n_img = n_lbl = same = converted = 0
    skipped: List[dict] = []
    written: List[str] = []
    counts: Counter = Counter()
    stats = LabelStats()
    made_dirs: Set[Path] = set()
    last_report = time.monotonic()
    status = "failed"

    def work(it: PlanItem):
        src = Path(it.abs_path)
        if not src.is_file():
            return ("missing", None)
        if it.verbatim:
            return ("file", None)
        try:
            return ("bytes", _encode(it, spec))
        except ImageTooLarge as e:
            return ("failed", str(e))
        except Exception as e:  # noqa: BLE001 — a source that cannot be decoded is skipped and reported
            return ("failed", f"could not be decoded ({type(e).__name__})")

    def ensure_dir(d: Path):
        if d not in made_dirs:
            d.mkdir(parents=True, exist_ok=True)
            made_dirs.add(d)

    try:
        for idx, (it, fut) in enumerate(_ordered_map(work, plan.items, _workers(spec))):
            now = time.monotonic()
            if idx % 25 == 0 or now - last_report > 0.5:
                job.update(done=idx, message=f"Copying {idx:,}/{total:,}")
                last_report = now
            kind, val = fut.result()
            if kind in ("missing", "failed"):
                skipped.append({"path": it.rel_path, "reason": "missing" if kind == "missing" else val})
                continue
            dst_img = target / it.img_rel
            try:
                ensure_dir(dst_img.parent)
                if _same_file(dst_img, Path(it.abs_path)):
                    same += 1          # backstop for layouts the folder check cannot see: never overwrite a source
                    continue
                if kind == "file":
                    try:
                        _atomic_copy(Path(it.abs_path), dst_img)
                    except OSError as e:
                        if not os.path.isfile(it.abs_path) or e.filename == it.abs_path:
                            skipped.append({"path": it.rel_path, "reason": f"unreadable ({e.strerror or e})"})
                            continue
                        raise
                else:
                    data, deep = val
                    converted += deep
                    _atomic_write(dst_img, data)
            except OSError as e:
                if e.errno in _FATAL_WRITE:
                    raise _write_failure(e, dst_img)
                skipped.append({"path": it.rel_path, "reason": f"could not be written ({e.strerror or e})"})
                continue
            n_img += 1
            counts[it.out_split] += 1
            written.append(it.img_rel)
            if it.has_label and it.label_path:
                text, st = _render_label(it, plan.classes, nc, opts.bad_class == "drop")
                if text is not None:
                    dst_lbl = target / it.lbl_rel
                    if not _same_file(dst_lbl, Path(it.label_path)):
                        try:
                            ensure_dir(dst_lbl.parent)
                            _atomic_write(dst_lbl, text.encode("utf-8"))
                        except OSError as e:
                            raise _write_failure(e, dst_lbl)
                        written.append(it.lbl_rel)
                        stats.add(st)
                        n_lbl += 1
        status = "done"
    except JobCancelled:
        status = "cancelled"
        raise
    finally:
        out_names, extended = final_names(plan, stats.max_cls)
        result = _result_base(plan, out_names, extended)
        result.update(images=n_img, labels=n_lbl, skipped_same_file=same, converted_8bit=converted,
                      split_counts=dict(counts), skipped=skipped[:1000], target=to_host(target),
                      missing=sum(1 for s in skipped if s["reason"] == "missing"),
                      failed=sum(1 for s in skipped if s["reason"] != "missing"), label_rows=stats.__dict__)
        if include_yaml and opts.include_yaml and status == "done":
            text = data_yaml_text(out_names, [], path=to_host(target), split_dirs=_present_dirs(plan, counts),
                                  header=_yaml_header(plan, False), warnings=result["warnings"])
            try:
                _atomic_write(target / "data.yaml", text.encode("utf-8"))
                written.append("data.yaml")
            except OSError as e:
                raise _write_failure(e, target / "data.yaml")
        if opts.dataset_card and status == "done":
            card = dataset_card(dataset_id, plan, out_names=out_names, skipped=skipped, result=result)
            try:
                _atomic_write(target / CARD, card.encode("utf-8"))
                written.append(CARD)
            except OSError as e:
                raise _write_failure(e, target / CARD)
        # Replacing a previous Fovea export: remove ITS files that this export did not write again (only
        # paths its manifest lists, inside the folder) so stale images can never leak into training.
        removed = 0
        if status == "done" and old_files:
            keep = set(written) | {MANIFEST}
            for rel in old_files:
                if rel in keep or rel.startswith("/") or ".." in PurePosixPath(rel).parts:
                    continue
                p = target / rel
                try:
                    if p.is_file() and not p.is_symlink() and _inside(Path(os.path.realpath(p)), target):
                        p.unlink()
                        removed += 1
                except OSError:
                    pass
        result["removed_stale"] = removed
        try:
            _atomic_write(target / MANIFEST, _manifest(plan, result, files=written, status=status).encode("utf-8"))
        except OSError:
            pass
    job.update(done=total, message="Done")
    return result


# ============================================================================== preview

def preview(dataset_id: str, names: List[str], filters: Dict[str, Any], opts: ExportOptions,
            target_dir: Optional[str] = None) -> dict:
    """What an export would contain, from the index (no file reads): per-split image / box / class counts,
    near-duplicates dropped, unnamed class ids, sequences spanning splits, the data.yaml, warnings."""
    plan = build_plan(dataset_id, names, filters, opts)
    ids = {it.id for it in plan.items}
    cp = plan.classes
    nc = len(cp.names)
    per_split: Dict[str, dict] = {}
    boxes_by_img: Dict[int, Counter] = defaultdict(Counter)
    raw_by_img = _boxes_by_image(dataset_id, ids)
    dropped_rows = unnamed_rows = 0
    unnamed_imgs: Set[int] = set()
    max_cls = -1
    for iid, cnt in raw_by_img.items():
        for c, n in cnt.items():
            new = cp.map(c)
            if new is None or c < 0:
                dropped_rows += n
                continue
            if new >= nc:
                unnamed_rows += n
                unnamed_imgs.add(iid)
                if opts.bad_class == "drop":
                    continue
            boxes_by_img[iid][new] += n
            max_cls = max(max_cls, new)
    for it in plan.items:
        s = per_split.setdefault(it.out_split, {"name": it.out_split or "(root)", "images": 0, "labeled": 0,
                                                "boxes": 0, "classes": defaultdict(lambda: {"images": 0, "boxes": 0})})
        s["images"] += 1
        s["labeled"] += it.has_label
        for c, n in boxes_by_img.get(it.id, {}).items():
            s["boxes"] += n
            s["classes"][c]["images"] += 1
            s["classes"][c]["boxes"] += n
    out_names, extended = final_names(plan, max_cls)
    result = _result_base(plan, out_names, extended)
    warnings = result["warnings"]
    # Sequences whose frames land in more than one split: validation frames nearly identical to training ones.
    seq_splits: Dict[str, Set[str]] = defaultdict(set)
    for it in plan.items:
        if it.seq:
            seq_splits[it.seq].add(_yaml_key(it.out_split, opts))     # the split a trainer sees
    spanning = sum(1 for v in seq_splits.values() if len(v) > 1)
    if spanning and not opts.split_builder:
        warnings.append(f"{spanning:,} sequences have frames in more than one split — validation scores will be "
                        "inflated. The split builder keeps each sequence in one split.")
    issue_rows = db.query("SELECT i.id, i.issues FROM images i WHERE i.dataset_id=? AND i.issues != '[]'", (dataset_id,)) if ids else []
    issue_counts: Counter = Counter()
    for r in issue_rows:
        if r["id"] in ids:
            for code in db.loads(r["issues"], []):
                issue_counts[code] += 1
    if issue_counts.get("out_of_range"):
        warnings.append(f"{issue_counts['out_of_range']:,} images have boxes outside the image: Ultralytics skips such "
                        "images when training (Explore → Issues to fix them).")
    if issue_counts.get("bad_format"):
        # transform_label_text drops these lines (the index does too): the image ships with its valid boxes only.
        warnings.append(f"{issue_counts['bad_format']:,} images have unparsable label lines (text, nan/inf, a fractional "
                        "class id): the export leaves the unusable lines out (Explore → Issues to fix them).")
    if dropped_rows:
        warnings.append(f"{dropped_rows:,} boxes of dropped classes are left out.")
    if unnamed_rows and opts.bad_class == "drop":
        warnings.append(f"{unnamed_rows:,} boxes with class ids beyond the class list are left out.")
    yaml_text = data_yaml_text(out_names, [], path=None, split_dirs=plan.split_dirs, header="",
                               warnings=warnings) if opts.include_yaml else None
    splits = []
    for key in sorted(per_split, key=lambda k: (YAML_KEYS.index(k) if k in YAML_KEYS else 3, k)):
        s = per_split[key]
        s["classes"] = {str(c): v for c, v in sorted(s["classes"].items())}
        s["share"] = round(s["images"] / max(1, len(plan.items)), 4)
        s["yaml_key"] = key if opts.split_builder else ("train" if key in ("", "unsplit") else yaml_key_for(key, opts.split_map))
        splits.append(s)
    out = {
        "images": len(plan.items), "matched": plan.matched, "labeled": sum(1 for it in plan.items if it.has_label),
        "boxes": sum(sum(c.values()) for iid, c in boxes_by_img.items()),
        "splits": splits, "classes": {"names": out_names, "nc": len(out_names), "extended": extended,
                                      "remapped": cp.remapped, "mapping": {str(k): v for k, v in sorted(cp.mapping.items())}},
        "dropped_class_rows": dropped_rows,
        "unnamed": {"rows": unnamed_rows, "images": len(unnamed_imgs), "policy": opts.bad_class},
        "near_dups": {"available": plan.near_dup_available, "groups": plan.near_dup_groups, "dropped": plan.dropped_near_dups},
        "sequences": {"total": len(seq_splits), "spanning_splits": spanning},
        "groups": plan.groups, "renamed": plan.renamed, "issues": dict(issue_counts),
        "resize": {"mode": opts.resize[0], "size": list(opts.resize[1:])} if opts.resize else None,
        "reencoded": sum(1 for it in plan.items if not it.verbatim),
        "yaml": yaml_text, "warnings": warnings,
        "card": dataset_card(dataset_id, plan, out_names=out_names, result=result, raw_boxes=raw_by_img)
        if opts.dataset_card else None,
    }
    if target_dir:
        try:
            t = check_copy_target(dataset_id, target_dir)
            out["target"] = inspect_target(t)
        except ValueError as e:
            out["target"] = {"path": target_dir, "error": str(e)}
    return out


# ============================================================================== dataset card

_FILTER_LABELS = {"split": "Split", "cls": "Contains class", "only_cls": "Only classes", "labeled": "Labeled",
                  "review": "Review status", "exclude_review": "Review status is not", "issue": "Issue",
                  "q": "Path contains", "seq": "Sequence", "ids": "Selected images", "min_boxes": "At least boxes",
                  "max_boxes": "At most boxes", "has_label": "Has a label file", "ai": "AI suggestions",
                  "set": "Saved selection"}
_REVIEW_ROWS = (("approved", "Approved"), ("flagged", "Flagged"), ("excluded", "Excluded"), (None, "Not reviewed"))


def _md(text: Any, limit: int = 0) -> str:
    """Text for one line or table cell: whitespace collapsed, Markdown/HTML specials escaped."""
    s = " ".join(str("" if text is None else text).split())
    if limit and len(s) > limit:
        s = s[: limit - 1].rstrip() + "…"
    for a, b in (("\\", "\\\\"), ("|", "\\|"), ("*", "\\*"), ("`", "\\`"), ("<", "\\<")):
        s = s.replace(a, b)
    return s


def _para(text: Any, limit: int = 0) -> str:
    """_md for text that starts a line: a leading #, >, list marker or numbered item would change the block."""
    s = _md(text, limit)
    m = re.match(r"\d+(?=[.)](?:\s|$))", s)
    if m:
        return s[:m.end()] + "\\" + s[m.end():]
    return ("\\" + s) if s[:1] in ("#", ">", "+", "=", "-") else s


def _code(text: Any) -> str:
    """A code span (paths, ids) that also survives inside a table cell."""
    s = " ".join(str(text).splitlines()).replace("|", "\\|")
    fence = "``" if "`" in s else "`"
    pad = " " if fence == "``" else ""
    return f"{fence}{pad}{s}{pad}{fence}"


def _n(v: Any) -> str:
    return f"{int(v or 0):,}"


def _count(n: int, word: str, plural: str = "") -> str:
    """'1 group', '2 groups', '1,204 images'."""
    return f"{int(n or 0):,} {word if int(n or 0) == 1 else (plural or word + 's')}"


def _pct(a: float, b: float) -> str:
    return f"{100.0 * a / b:.1f}%" if b else "–"


def _table(head: List[str], rows: List[List[str]], right: Iterable[int] = ()) -> List[str]:
    right = set(right)
    out = ["| " + " | ".join(head) + " |",
           "|" + "|".join("--:" if i in right else "---" for i in range(len(head))) + "|"]
    out += ["| " + " | ".join(r) + " |" for r in rows]
    return out


def _boxes_by_image(dataset_id: str, ids: Set[int]) -> Dict[int, Dict[int, int]]:
    """image id -> {source class id: boxes}, from the index (no label file reads). Plain tuples straight off
    the cursor: at 45k images this is ~130k rows."""
    out: Dict[int, Dict[int, int]] = {}
    if ids:
        cur = db.execute("SELECT image_id, cls, COUNT(*) FROM boxes WHERE dataset_id=? GROUP BY image_id, cls", (dataset_id,))
        for iid, c, n in cur:
            if iid in ids:
                d = out.get(iid)
                if d is None:
                    d = out[iid] = {}
                d[int(c)] = d.get(int(c), 0) + n
    return out


def _yaml_key(split: str, opts: ExportOptions) -> str:
    if opts.split_builder:
        return split if split in YAML_KEYS else "train"
    return "train" if split in ("", "unsplit") else yaml_key_for(split, opts.split_map)


def _split_sort(opts: ExportOptions):
    def key(s: str) -> tuple:
        k = _yaml_key(s, opts)
        return (s in ("", "unsplit"), YAML_KEYS.index(k) if k in YAML_KEYS else 3, s)
    return key


def _filter_text(k: str, v: Any, src_names: List[str]) -> str:
    if k == "ids":
        return f"{len([x for x in str(v).split(',') if x.strip()]):,} images picked by hand"
    if k == "set":
        return "a selection saved in Fovea"
    if k in ("cls", "only_cls"):
        parts = []
        for c in str(v).split(","):
            c = c.strip()
            parts.append(f"{c} ({_md(src_names[int(c)])})" if c.isdigit() and int(c) < len(src_names) else _md(c))
        return ", ".join(parts)
    if k == "has_label":
        return "yes" if str(v).lower() in ("1", "true", "yes") else "no"
    return _md(v)


def _qa_meta(dataset_id: str) -> Optional[dict]:
    try:
        from .qa import load_summary   # noqa: PLC0415 — QA engine (optional at import time)
        return load_summary(dataset_id)
    except Exception:  # noqa: BLE001 — older database or QA module unavailable
        return None


def _issue_tables() -> Tuple[dict, dict, dict, dict]:
    try:
        from .labels import AI_AUDIT_KINDS, AI_AUDIT_LABELS, FLAG_LABELS, PARSER_ISSUE_LABELS  # noqa: PLC0415
        return dict(PARSER_ISSUE_LABELS), dict(FLAG_LABELS), dict(AI_AUDIT_KINDS), dict(AI_AUDIT_LABELS)
    except ImportError:
        return {}, {}, {}, {}


def _when(ts: float) -> str:
    return time.strftime("%Y-%m-%d %H:%M %z", time.localtime(ts))


def dataset_card(dataset_id: str, plan: ExportPlan, exported: bool = True, out_names: Optional[List[str]] = None,
                 skipped: Iterable[dict] = (), result: Optional[dict] = None,
                 raw_boxes: Optional[Dict[int, Dict[int, int]]] = None) -> str:
    """DATASET_CARD.md: what is in this export (or, with exported=False, in the dataset/selection) and how
    it was made — splits, classes with their class-card definitions, data quality, review progress, label
    issues and the export settings. Built from the index only (counts as last scanned); images the export
    could not write (`skipped`) are left out of every count."""
    row = db.query_one("SELECT * FROM datasets WHERE id=?", (dataset_id,)) or {}
    src_names = [str(n) for n in (db.loads(row.get("classes"), []) or [])]
    gone = {s.get("path") for s in skipped if isinstance(s, dict)}
    items = [it for it in plan.items if it.rel_path not in gone]
    ids = {it.id for it in items}
    opts, cp = plan.opts, plan.classes
    nc = len(cp.names)
    drop_unnamed = opts.bad_class == "drop"
    by_img: Dict[int, Dict[int, int]] = {}
    max_cls = -1
    for iid, cnt in (raw_boxes if raw_boxes is not None else _boxes_by_image(dataset_id, ids)).items():
        if iid not in ids:
            continue
        out: Dict[int, int] = {}
        for c, n in cnt.items():
            new = cp.map(c)
            if new is None or (new >= nc and drop_unnamed):
                continue
            out[new] = out.get(new, 0) + n
            if new > max_cls:
                max_cls = new
        if out:
            by_img[iid] = out
    names = list(out_names) if out_names is not None else final_names(plan, max_cls)[0]
    # One pass over the images: per split [images, labeled, background, boxes], per class boxes by split.
    per_split: Dict[str, List[int]] = defaultdict(lambda: [0, 0, 0, 0])
    per_cls: Dict[int, Counter] = defaultdict(Counter)
    imgs_cls: Counter = Counter()
    rv: Counter = Counter()
    total_boxes = 0
    for it in items:
        st = per_split[it.out_split]
        st[0] += 1
        bx = by_img.get(it.id)
        if it.has_label:
            st[1] += 1
            if not bx:
                st[2] += 1
        if bx:
            for c, n in bx.items():
                per_cls[c][it.out_split] += n
                imgs_cls[c] += 1
                st[3] += n
                total_boxes += n
        rv[it.review if it.review in ("approved", "flagged", "excluded") else None] += 1
    splits = sorted(per_split, key=_split_sort(opts))
    parser_labels, flag_labels, audit_kinds, audit_labels = _issue_tables()
    L: List[str] = []

    # ---- header
    title = row.get("name") or dataset_id
    L += [f"# {_md(title)} — dataset card", ""]
    if (row.get("description") or "").strip():
        L += [_para(row["description"], 600), ""]
    labeled = sum(v[1] for v in per_split.values())
    facts = [["Dataset", f"{_md(title)} ({_code(dataset_id)})"],
             ["Source folder", _code(to_host(row["root"])) if row.get("root") else "–"],
             ["Exported" if exported else "Generated", f"{_when(plan.created_at)} · Fovea {__version__}"],
             ["Images", f"{_n(len(items))} ({_n(labeled)} with a label file)"],
             ["Boxes", f"{_n(total_boxes)} in {_n(len(names))} classes"],
             ["Splits", " · ".join(f"{_md(s or '(root)')} {_n(per_split[s][0])}" for s in splits) or "–"]]
    if exported:
        facts.append(["Format", "YOLO: `images/<split>/…`, `labels/<split>/….txt`, `data.yaml`"])
    if result and (result.get("missing") or result.get("failed")):
        facts.append(["Not exported", f"{_n(result.get('missing'))} missing, {_n(result.get('failed'))} unreadable "
                                       "(left out of every count below)"])
    L += _table(["", ""], facts) + [""]

    # ---- splits
    L += ["## Splits", ""]
    rows = []
    for s in splits:
        n_img, n_lab, n_bg, n_box = per_split[s]
        rows.append([_md(s or "(root)"), _yaml_key(s, opts), _n(n_img), _pct(n_img, len(items)), _n(n_lab), _n(n_bg),
                     _n(n_img - n_lab), _n(n_box)])
    L += _table(["Split", "data.yaml", "Images", "Share", "Labeled", "Background", "Unlabeled", "Boxes"], rows,
                right=range(2, 8))
    L += ["", "_Background_: a label file with no boxes. _Unlabeled_: no label file (Ultralytics also trains on "
              "these as background).", ""]

    # ---- classes
    cards: Dict[int, dict] = {}
    try:
        for r in db.query("SELECT cls, description, include, exclude, prompt FROM class_cards WHERE dataset_id=?", (dataset_id,)):
            cards[int(r["cls"])] = r
    except Exception:  # noqa: BLE001 — database from before class cards
        cards = {}
    split_cols = splits if len(splits) <= 4 else []

    def sources(t: int) -> List[int]:
        """Source class ids feeding output id t ([] when t's own class was dropped and nothing maps to it)."""
        srcs = sorted(c for c, v in cp.mapping.items() if v == t)
        return srcs if srcs or t in cp.mapping else [t]

    def card_for(t: int) -> dict:
        srcs = sources(t)
        for c in ([t] if t in srcs else []) + srcs:
            card = cards.get(c)
            if card and any((card[k] or "").strip() for k in ("description", "include", "exclude", "prompt")):
                return card
        return {}

    L += ["## Classes", ""]
    head = ["ID", "Name"] + (["Source ids"] if cp.remapped else []) + ["Boxes", "Share", "Images"] + \
        [_md(s or "(root)") for s in split_cols] + ["Description"]
    rows = []
    n_num = 3 + len(split_cols)
    first_num = 3 if cp.remapped else 2
    for t, name in enumerate(names):
        card = card_for(t)
        desc = (card.get("description") or "").strip() or (card.get("prompt") or "").strip()
        b = sum(per_cls[t].values())
        r = [str(t), _md(name)] + ([", ".join(map(str, sources(t))) or "– (dropped)"] if cp.remapped else []) + \
            [_n(b), _pct(b, total_boxes), _n(imgs_cls[t])] + [_n(per_cls[t][s]) for s in split_cols] + [_md(desc, 140) or "–"]
        rows.append(r)
    L += _table(head, rows, right=range(first_num, first_num + n_num)) + [""]
    notes = []
    empty = [t for t in range(len(names)) if not sum(per_cls[t].values())]
    if empty:
        notes.append(f"No boxes in this {'export' if exported else 'selection'}: "
                     + ", ".join(f"{t} ({_md(names[t])})" for t in empty[:20]) + ("…" if len(empty) > 20 else "") + ".")
    extended = [t for t in range(len(cp.names), len(names))]
    unnamed = [t for t in range(len(cp.names)) if re.fullmatch(r"class_\d+", names[t] or "")]
    if unnamed and len(unnamed) == len(cp.names):
        notes.append("Class names were not set (they read `class_<id>`): name the classes in Fovea "
                     "(Overview → Classes) before training or sharing this dataset.")
    elif unnamed:
        notes.append("Classes without a name: " + ", ".join(map(str, unnamed[:20])) + ("…" if len(unnamed) > 20 else "") + ".")
    if extended:
        notes.append("Class ids beyond the class list are kept and named `class_<id>`: "
                     + ", ".join(map(str, extended[:20])) + ("…" if len(extended) > 20 else "") + ".")
    L += [f"- {x}" for x in notes] + ([""] if notes else [])

    defs = []
    for t, name in enumerate(names):
        card = card_for(t)
        if not card:
            continue
        defs += [f"**{t} · {_md(name)}**", ""]
        for key, label in (("description", "Description"), ("include", "Includes"), ("exclude", "Excludes"),
                           ("prompt", "AI prompt")):
            v = (card.get(key) or "").strip()
            if v:
                defs.append(f"- {label}: {_md(v, 1000)}")
        defs.append("")
    if defs:
        L += ["### Class definitions", "", "From the class cards (labeling guidelines) in Fovea.", ""] + defs

    # ---- data quality
    L += ["## Data quality", ""]
    flags: Dict[str, Set[int]] = defaultdict(set)
    dup_group: Dict[int, int] = {}
    try:
        for iid, code, detail in db.execute("SELECT image_id, code, detail FROM image_flags WHERE dataset_id=?", (dataset_id,)):
            if iid not in ids:
                continue
            flags[code].add(iid)
            if code == "near_dup":
                d = db.loads(detail, {}) or {}
                if isinstance(d, dict) and d.get("group") is not None:
                    dup_group[iid] = int(d["group"])
        hashed = db.query_one("SELECT 1 AS x FROM image_hash WHERE dataset_id=? LIMIT 1", (dataset_id,)) is not None
    except Exception:  # noqa: BLE001 — database from before the QA tables
        hashed = False
    meta = _qa_meta(dataset_id) or {}
    # Leaks are judged on the data.yaml split a trainer sees: two source folders that both feed `test`
    # (test, test-1854-1860) sharing a video is not a train/val leak.
    split_of = {it.id: _yaml_key(it.out_split, opts) for it in items}
    groups: Dict[int, List[int]] = defaultdict(list)
    for iid, g in dup_group.items():
        groups[g].append(iid)
    multi = {g: m for g, m in groups.items() if len(m) > 1}
    leaks = {g: m for g, m in multi.items() if len({split_of[i] for i in m}) > 1}
    # (qa-engine hook) Leaks by the QA engine's own definition (a copy within leak_dist, or the same view of a
    # video, in another split), judged on THIS export's images and splits: the card and the Overview's
    # "Duplicate across splits" are one definition. The groups above remain the fallback.
    leaked: Optional[Set[int]] = None
    if hashed:
        try:
            from .qa import leaked_images   # noqa: PLC0415 — None: no QA results of its own to judge by
            leaked = leaked_images(dataset_id, split_of)
        except Exception:  # noqa: BLE001 — QA module unavailable: count the groups instead
            leaked = None
    seq_splits: Dict[str, Set[str]] = defaultdict(set)
    for it in items:
        if it.seq:
            seq_splits[it.seq].add(split_of[it.id])
    spanning = sum(1 for v in seq_splits.values() if len(v) > 1)
    if not hashed and not meta.get("computed_at"):
        L += ["> The data-quality checks have not been run on this dataset yet (Fovea → Overview → Data quality), "
              "so near-duplicates, leaks between splits and image quality are unknown.", ""]
    else:
        when = meta.get("computed_at")
        if when:
            L += [f"Checks last run: {_when(float(when))}.", ""]
    rows = []
    if hashed or meta.get("computed_at"):
        dropped = plan.dropped_near_dups
        parts = [f"{_count(len(multi), 'group')} of look-alike images"] if multi else []
        if dropped:
            parts.append(f"{_count(dropped, 'image')} dropped by this export (one kept per group)")
        rows.append(["Near-duplicate images", _n(sum(len(m) for m in multi.values())), "; ".join(parts) or "–"])
        if leaked is not None:      # (qa-engine hook)
            rows.append(["Duplicates across splits (leak)", _n(len(leaked)),
                         "a copy, or the same view of its video, in another split" if leaked else "–"])
        else:
            rows.append(["Near-duplicates across splits (leak)", _n(sum(len(m) for m in leaks.values())),
                         f"{_count(len(leaks), 'group')} with members in more than one split" if leaks else "–"])
    span_imgs = sum(1 for it in items if it.seq and len(seq_splits[it.seq]) > 1)
    rows.append(["Sequences spanning splits", _n(span_imgs), f"{_n(spanning)} of {_count(len(seq_splits), 'sequence')}"
                 + (" (frames of one video in two splits inflate validation scores)" if spanning else "")])
    for code, label in flag_labels.items():
        if code in ("near_dup", "leak_split") or not flags.get(code):
            continue
        rows.append([_md(label), _n(len(flags[code])), "–"])
    L += _table(["Check", "Images", "Note"], rows, right=[1]) + [""]

    # ---- review progress
    L += ["## Review progress", ""]
    reviewed = len(items) - rv[None]
    L += [f"{_n(reviewed)} of {_count(len(items), 'image')} reviewed ({_pct(reviewed, len(items))}).", ""]
    L += _table(["Status", "Images", "Share"], [[label, _n(rv[k]), _pct(rv[k], len(items))] for k, label in _REVIEW_ROWS],
                right=[1, 2]) + [""]

    # ---- label issues
    L += ["## Label issues", ""]
    issue_imgs: Counter = Counter()
    for iid, issues in db.execute("SELECT id, issues FROM images WHERE dataset_id=? AND issues != '[]'", (dataset_id,)):
        if iid in ids:
            for code in set(db.loads(issues, []) or []):
                issue_imgs[code] += 1
    rows = [[_md(parser_labels.get(code, code)), _n(n)] for code, n in sorted(issue_imgs.items(), key=lambda kv: -kv[1])
            if code != "missing_label"]
    audit: Dict[str, Set[int]] = defaultdict(set)
    pending: Set[int] = set()
    try:
        kind_code = {v: k for k, v in audit_kinds.items()}
        for iid, kind in db.execute("SELECT image_id, kind FROM ai_preds WHERE dataset_id=? AND state='pending'", (dataset_id,)):
            if iid not in ids:
                continue
            if kind in kind_code:
                audit[kind_code[kind]].add(iid)
            else:
                pending.add(iid)
    except Exception:  # noqa: BLE001 — database from before AI tables
        pass
    for code, imgs in sorted(audit.items()):
        rows.append([_md(audit_labels.get(code, code)) + " (open finding)", _n(len(imgs))])
    if rows:
        L += _table(["Issue", "Images"], rows, right=[1]) + [""]
    else:
        L += ["No label-file problems found.", ""]
    if pending:
        L += [f"{_count(len(pending), 'image')} with AI-suggested boxes waiting for review (not part of the labels).", ""]

    # ---- export settings / selection
    real_filters = {k: v for k, v in plan.filters.items() if k != "seed"}
    if exported:
        L += ["## Export settings", ""]
        sel = "; ".join(f"{_FILTER_LABELS.get(k, _md(k))}: {_filter_text(k, v, src_names)}" for k, v in real_filters.items())
        L.append(f"- Selection: {sel if sel else 'every image of the dataset'} — {_count(plan.matched, 'image')} matched")
        L.append("- Images without a label file: " + ("included" if opts.include_unlabeled else "left out"))
        if cp.remapped:
            moved = [f"{c} → {v}" for c, v in sorted(opts.class_map.items()) if v is not None and v != c]
            dropped_cls = [str(c) for c, v in sorted(cp.mapping.items()) if v is None]
            parts = []
            if moved:
                parts.append("remapped " + ", ".join(moved))
            if dropped_cls:
                parts.append("dropped " + ", ".join(dropped_cls))
            if opts.compact:
                parts.append("renumbered to consecutive ids")
            L.append("- Classes: " + ("; ".join(parts) or "remapped") + " (source id → exported id)")
        else:
            L.append("- Classes: ids unchanged")
        if opts.names_override is not None:
            L.append("- Class names: set for this export")
        L.append("- Label rows whose class id has no name: " + ("dropped" if drop_unnamed else "kept, named `class_<id>`"))
        L.append("- Near-duplicates: " + (f"dropped, one image kept per group (approved first, then labeled, then most "
                                           f"boxes) — {_count(plan.dropped_near_dups, 'image')} removed"
                                           if opts.drop_near_dups else "kept"))
        sb = opts.split_builder
        if sb:
            ratios = " · ".join(f"{k} {round(sb.ratios[k] * 100)}%" for k in YAML_KEYS if k in sb.ratios)
            grouping = {"seq": "whole sequences", "seq+dups": "whole sequences, near-duplicates together",
                        "folder": "whole folders", "image": "single images", "regex": f"regex {_code(sb.regex)}"}[sb.group_by]
            L.append(f"- Splits: rebuilt by Fovea — {ratios}; seed {sb.seed}; kept together: {grouping}"
                     + ("; balanced per class" if sb.stratify else ""))
        else:
            mapped = [f"{_md(s)} → {_yaml_key(s, opts)}" for s in splits if s and _yaml_key(s, opts) != s]
            L.append("- Splits: the dataset's own split folders" + (f" (data.yaml: {', '.join(mapped)})" if mapped else ""))
        if opts.resize is None:
            L.append("- Image files: originals, unchanged")
        elif opts.resize[0] == "max":
            L.append(f"- Image files: longest side at most {opts.resize[1]} px (larger ones re-encoded as JPEG, "
                     "quality 92; smaller ones copied unchanged)")
        else:
            L.append(f"- Image files: resized to {opts.resize[1]}×{opts.resize[2]} px (JPEG, quality 92)")
        L.append("- Labels: YOLO txt; confidence columns dropped; segmentation, pose and OBB rows kept as they are")
        L.append("")
    elif real_filters:
        L += ["## Selection", ""]
        L += [f"- {_FILTER_LABELS.get(k, _md(k))}: {_filter_text(k, v, src_names)}" for k, v in real_filters.items()]
        L.append("")

    # ---- warnings
    warns = list((result or {}).get("warnings") or plan.warnings)
    if not any(_yaml_key(s, opts) == "val" for s in splits) and items:
        w = "There is no validation split" + (": data.yaml points val at the training images."
                                              if exported and opts.include_yaml else ".")
        if not any(x.startswith("This export has no validation split") for x in warns):
            warns.append(w)
    if spanning and not opts.split_builder:
        warns.append(f"{spanning:,} sequences have frames in more than one split — validation scores will be inflated. "
                     "The split builder keeps each sequence in one split.")
    if leaked:                      # (qa-engine hook)
        warns.append(f"{_count(len(leaked), 'image')} {'has' if len(leaked) == 1 else 'have'} a copy, or the same "
                     f"view of {'its' if len(leaked) == 1 else 'their'} video, in another split (a leak between splits).")
    elif leaks and leaked is None:
        warns.append(f"{_count(len(leaks), 'group')} of near-duplicate images "
                     f"{'has' if len(leaks) == 1 else 'have'} members in more than one split (a leak between splits).")
    warns = list(dict.fromkeys(warns))
    if warns:
        L += ["## Warnings", ""] + [f"- {_md(w)}" for w in warns] + [""]

    L.append(f"_Generated by Fovea {__version__} from its index on {_when(plan.created_at)}. Counts are images and "
             "boxes as last scanned; label files edited since may differ._")
    return "\n".join(L) + "\n"


def card_for_dataset(dataset_id: str, names: List[str], filters: Dict[str, Any]) -> str:
    """The standalone card: the dataset (or the images matching `filters`) as indexed, no export settings."""
    plan = build_plan(dataset_id, names, filters, ExportOptions())
    return dataset_card(dataset_id, plan, exported=False)


# ============================================================================== lists

def render_list(dataset_id: str, filters: Dict[str, Any], style: str = "host", list_dir: Optional[str] = None) -> str:
    """One image path per line. 'rel' paths are written './'-relative to the folder the list lives in —
    the form Ultralytics resolves against the list file (a bare 'split/name.jpg' index key resolves nowhere)."""
    lines = []
    base = Path(list_dir) if list_dir else None
    for r in snapshot_rows(dataset_id, filters):
        if style == "rel":
            if base is not None:
                rel = os.path.relpath(r["abs_path"], base).replace(os.sep, "/")
                lines.append(rel if rel.startswith("../") else "./" + rel)
            else:
                lines.append(r["rel_path"])
        elif style == "container":
            lines.append(r["abs_path"])
        else:
            lines.append(to_host(r["abs_path"]))
    return "\n".join(lines) + ("\n" if lines else "")


# ============================================================================== data.yaml in the dataset

_TOP_KEY = re.compile(r"^([^\s#'\"-][^:#]*?|'[^']*'|\"[^\"]*\")\s*:(?:\s|$)")


def _yaml_blocks(lines: List[str]) -> List[Tuple[str, int, int]]:
    """Top-level (key, start, end) line ranges. A block runs over indented lines, blank lines and column-0
    list items ('- a' under 'train:'), up to the next top-level key or column-0 comment; trailing blank
    lines stay outside it."""
    starts = []
    for i, ln in enumerate(lines):
        m = _TOP_KEY.match(ln)
        if m:
            starts.append((m.group(1).strip().strip("'\""), i))
    blocks = []
    for k, s in starts:
        e = s + 1
        while e < len(lines) and (lines[e][:1] in (" ", "\t") or (lines[e][:1] == "-" and not lines[e].startswith("---"))
                                  or not lines[e].strip()):
            e += 1
        while e > s + 1 and not lines[e - 1].strip():
            e -= 1
        blocks.append((k, s, e))
    return blocks


def _edit_yaml(text: str, doc: Dict[str, Any], owned: Iterable[str]) -> Optional[str]:
    """Replace only the owned top-level keys in the original text (comments and other keys survive).
    None when the result would not parse back to `doc` (flow mappings, anchors...)."""
    lines = text.splitlines(keepends=True)
    if lines and not lines[-1].endswith("\n"):
        lines[-1] += "\n"
    blocks = {k: (s, e) for k, s, e in _yaml_blocks(lines)}
    edits = []
    appended = []
    for k in owned:
        new = yaml.safe_dump({k: doc[k]}, sort_keys=False, allow_unicode=True) if k in doc else ""
        if k in blocks:
            s, e = blocks[k]
            edits.append((s, e, new))
        elif new:
            appended.append(new)
    for s, e, new in sorted(edits, reverse=True):
        lines[s:e] = [new]
    out = "".join(lines) + "".join(appended)
    try:
        if yaml.safe_load(out) != doc:
            return None
    except yaml.YAMLError:
        return None
    return out


def _resolves_to(value: Any, base: Path, wanted: List[str]) -> bool:
    vals = value if isinstance(value, list) else [value]
    if not vals or not all(isinstance(v, str) for v in vals):
        return False
    got = []
    for v in vals:
        p = resolve(v) if (Path(v).is_absolute() or is_windows_path(v)) else (base / v)
        got.append(os.path.realpath(p))
    return sorted(got) == sorted(os.path.realpath(w) for w in wanted)


def plan_data_yaml(dataset_root: str, names: List[str], layout: dict) -> dict:
    """The data.yaml Fovea would write for this dataset: {path, exists, current, proposed, changed,
    comments_kept}. Values that still point at the right folders are kept as written (relative stays
    relative); a relative 'path:' becomes absolute because Ultralytics resolves it against the trainer's
    working directory."""
    root = Path(dataset_root)
    out = Path(layout.get("data_yaml") or (root / "data.yaml"))
    doc: Dict[str, Any] = {}
    current = None
    if out.is_file():
        try:
            current = out.read_text(encoding="utf-8")
            loaded = yaml.safe_load(current)
        except (OSError, UnicodeDecodeError, yaml.YAMLError) as e:
            raise ValueError(f"{to_host(out)} could not be read as YAML — fix or remove it first ({e})")
        if isinstance(loaded, dict):
            doc = dict(loaded)
    ydir = out.parent
    orig_path = doc.get("path")
    orig_keys = set(doc)
    if current is not None and orig_path is None and ydir.resolve() == root.resolve():
        base = ydir                      # the file never had a path: Ultralytics reads it beside the yaml
    else:
        base = root
        keep = isinstance(orig_path, str) and (Path(orig_path).is_absolute() or is_windows_path(orig_path)) \
            and os.path.realpath(resolve(orig_path)) == os.path.realpath(root)
        if not keep:
            doc["path"] = to_host(root)
    splits: Dict[str, List[str]] = {}
    for s in layout.get("sources", []):
        if s.get("recursive") is False:
            continue        # loose files beside split folders: a YOLO directory entry would pull in every split
        split = yaml_key_for(s.get("split") or "train")
        target = s.get("list_file") or s.get("img_dir")
        if target:
            splits.setdefault(split, []).append(target)
    for split, targets in splits.items():
        if split in doc and _resolves_to(doc[split], base, targets):
            continue
        vals = []
        for t in targets:
            try:
                vals.append(Path(t).relative_to(base).as_posix())
            except ValueError:
                vals.append(to_host(t))
        doc[split] = vals[0] if len(vals) == 1 else vals     # a split made of several folders stays a list
    doc["nc"] = len(names)
    doc["names"] = {i: n for i, n in enumerate(names)}
    owned = ["path", "train", "val", "test", "nc", "names"]
    text = _edit_yaml(current, doc, [k for k in owned if k in doc or k in orig_keys]) if current else None
    comments_kept = text is not None
    if text is None:
        text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    return {"path": str(out), "path_host": to_host(out), "exists": current is not None, "current": current,
            "proposed": text, "changed": text != current, "comments_kept": comments_kept or current is None}


def write_data_yaml(dataset_root: str, names: List[str], layout: dict) -> str:
    """Point the dataset's YAML at its current splits and classes.

    Updates the file the dataset was detected from (data.yaml, dataset.yaml, ...) instead of creating a
    second one beside it, keeps every key it does not own (kpt_shape, download, flip_idx, ...) and the
    comments, backs the old file up to <name>.bak, and replaces the file atomically."""
    p = plan_data_yaml(dataset_root, names, layout)
    out = Path(p["path"])
    refuse_fovea_dirs(out, to_host(out))
    if not p["changed"]:
        return str(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    mode = stat.S_IMODE(out.stat().st_mode) if out.is_file() else None
    if p["current"] is not None:
        bak = out.with_name(out.name + ".bak")
        try:
            bak.write_text(p["current"], encoding="utf-8")
        except OSError as e:
            raise ValueError(f"Could not back up {to_host(out)} before rewriting it ({e.strerror or e})")
    fd, tmp = tempfile.mkstemp(prefix=".fovea-", suffix=".yaml", dir=str(out.parent))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(p["proposed"])
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, out)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    return str(out)
