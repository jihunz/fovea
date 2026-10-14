"""Dataset QA that needs no model: perceptual hashes and image statistics, near-duplicates and split leakage,
label consistency across near-identical frames, temporal checks inside sequences, per-class box-shape
outliers, image quality, and keyframe interpolation / gap filling (as pending proposals).

Hashes (`image_hash`): `dhash` is the classic 64-bit dHash (9x8 gradient signs). It survives re-encoding
and resizing, but it only sees the scene: on 7,808 real CCTV frames two consecutive frames of a static
camera were within 3 bits 67% of the time, so on its own it would call most of every video a duplicate.
`b0..b3` hold a 256-bit dHash (17x16, four signed 64-bit words, b0 most significant) that also sees a
person move; frames <= 2 bits apart on it are visually the same picture. Every `dist` the API reports
is a Hamming distance on the 256-bit hash (0 identical, <= 2 near-duplicate, ~35 same camera view at
another moment, > 64 unrelated). `sharpness` is 1 - the Crete-Roffet blur effect (higher is sharper;
contrast-independent, unlike Laplacian variance, which ranks every night frame as blurry).
"""
from __future__ import annotations

import logging
import math
import os
import re
import statistics
import sys
import threading
import time
import traceback
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Set, Tuple

from PIL import Image, ImageChops, ImageFilter, ImageStat

from .. import db
from ..config import MAX_IMAGE_PIXELS
from ..jobs import Job, JobCancelled, jobs
from .labels import iou, read_label_file

log = logging.getLogger("fovea.qa")

LABELS = {
    "near_dup": "Near-duplicate image",
    "leak_split": "Duplicate across splits",
    "dup_same_role": "Same frame in two folders of one split",
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
CHECKS: Dict[str, Tuple[str, ...]] = {
    "duplicates": ("near_dup", "leak_split", "dup_same_role", "dup_label_mismatch"),
    "sequence": ("seq_gap", "seq_flicker", "seq_jump", "seq_count_spike"),
    "shape": ("shape_outlier",),
    "quality": ("blurry", "dark", "overexposed"),
}
LABEL_CHECKS = ("duplicates", "sequence", "shape")     # depend on label files: rerun when labels change
CODES = tuple(c for codes in CHECKS.values() for c in codes)

# Tuned on the real 7,808-frame VFP290K clone (fall detection, 168 videos, static and hand-held cameras).
DEFAULTS: Dict[str, float] = {
    "dup_dist": 2,          # 256-bit distance for near_dup groups and dup_label_mismatch
    "leak_dist": 6,         # 256-bit distance for a twin in another split (also catches re-encoded copies)
    "scene_dist": 8,        # 64-bit distance: "same camera view" (sequence leakage evidence)
    "mismatch_iou": 0.5,    # boxes of two near-identical frames agree when matched one-to-one at this IoU
    "seq_cut": 20,          # 64-bit distance between consecutive frames that ends a shot
    "seq_video": 12,        # median 64-bit distance between consecutive frames for a group to count as video
    "shape_z": 5.0,         # robust z-score (MAD) of log aspect / log area within a class (4.0 flagged 4% of VFP290K)
    "shape_min": 30,        # boxes a class needs before its shape statistics mean anything
    "blur_floor": 0.30,     # Crete-Roffet blur effect: >= this AND robust z (within a brightness stratum) >= blur_z
    "blur_z": 4.2,
    "dark_floor": 25.0,     # mean luma (0-255): < this AND robust z <= -dark_z
    "dark_z": 2.5,
    "bright_floor": 215.0,  # mean luma: > this AND robust z >= bright_z
    "bright_z": 2.5,
}
OPTION_LIMITS = {"dup_dist": (0, 16), "leak_dist": (0, 24), "scene_dist": (0, 24), "mismatch_iou": (0.1, 0.95),
                 "seq_cut": (4, 64), "seq_video": (2, 40), "shape_z": (2.0, 20.0), "shape_min": (5, 100000),
                 "blur_floor": (0.0, 0.9), "blur_z": (1.0, 20.0), "dark_floor": (0.0, 128.0), "dark_z": (0.0, 20.0),
                 "bright_floor": (128.0, 255.0), "bright_z": (0.0, 20.0)}

# Bumped when stored duplicate results change meaning; an older result is re-checked on the next summary read.
# 2: leaks are judged on split roles (dup_same_role for two folders of one role); `keep` is no longer stored.
ENGINE_VERSION = 2
ANALYSIS_PX = 320           # images are decoded (JPEG: DCT-downscaled) to fit this before any statistic
HASH_THREADS = max(1, min(4, (os.cpu_count() or 2) - 1))
HASH_CHUNK = 256
PLUGIN_KEY = ("qa", "summary")
GROUPS_KEY = "dup_groups"
MAX_INTERP_FRAMES = 2000

_SEQ_RE = re.compile(r"^(.*?)[\-_ .]*(?:frame|img|image|fr|f)?[\-_ .]*(\d+)$", re.I)
_NUM_RE = re.compile(r"(\d+)")

_np_mod: Any = False


def _numpy():
    """numpy when installed (faster blur statistic and similarity search), else None: the QA engine works
    with Pillow alone."""
    global _np_mod
    if _np_mod is False:
        try:
            import numpy  # noqa: PLC0415
            _np_mod = numpy
        except Exception:  # noqa: BLE001
            _np_mod = None
    return _np_mod


_POP8: Any = None


def _popcount(np, a):
    """Set bits per element of an unsigned integer array, as uint8 — np.bitwise_count, which only exists from
    NumPy 2.0. requirements.txt allows 1.26 (pip keeps it where another package pins numpy<2), so without it
    the bytes are counted through a 256-entry table."""
    count = getattr(np, "bitwise_count", None)
    if count is not None:
        return count(a)
    global _POP8
    if _POP8 is None:
        _POP8 = np.array([bin(i).count("1") for i in range(256)], dtype=np.uint8)
    a = np.ascontiguousarray(a)
    per_byte = _POP8[a.view(np.uint8)].reshape(a.shape + (a.dtype.itemsize,))
    return per_byte.sum(axis=-1, dtype=np.uint8)


# ------------------------------------------------------------------ small helpers

def natural_key(s: str) -> list:
    return [int(t) if t.isdigit() else t.casefold() for t in _NUM_RE.split(s)]


def frame_number(rel_path: str) -> Optional[int]:
    m = _SEQ_RE.match(PurePosixPath(rel_path).stem)
    return int(m.group(2)) if m else None


def to_signed(v: int) -> int:
    return v - (1 << 64) if v >= (1 << 63) else v


def to_unsigned(v: int) -> int:
    return v + (1 << 64) if v < 0 else v


def fine_hash(row: dict) -> Optional[int]:
    if row.get("b0") is None:
        return None
    v = 0
    for k in ("b0", "b1", "b2", "b3"):
        v = (v << 64) | to_unsigned(int(row[k]))
    return v


def coarse_hash(row: dict) -> Optional[int]:
    return None if row.get("dhash") is None else to_unsigned(int(row["dhash"]))


def hamming(a: int, b: int) -> int:
    return (a ^ b).bit_count()


def robust_z(values: Sequence[float], floor_mad: float) -> Tuple[float, float]:
    """(median, scaled MAD) — MAD is floored so a near-constant column does not turn noise into z = 50."""
    med = statistics.median(values)
    mad = statistics.median(abs(v - med) for v in values) * 1.4826
    return med, max(mad, floor_mad)


def _centre_dist(a: Sequence[float], b: Sequence[float], W: float, H: float) -> float:
    return math.hypot((a[1] - b[1]) * W, (a[2] - b[2]) * H)


def _diag(b: Sequence[float], W: float, H: float) -> float:
    return math.hypot(b[3] * W, b[4] * H)


def assign(cost: List[List[float]]) -> List[Tuple[int, int]]:
    """Minimum-cost one-to-one assignment (Hungarian / Kuhn-Munkres, O(n^2 m)) for a rectangular matrix.
    Returns (row, col) pairs; with more rows than columns the extra rows stay unassigned."""
    n = len(cost)
    if n == 0:
        return []
    m = len(cost[0])
    if m == 0:
        return []
    if n > m:
        return [(i, j) for j, i in assign([list(col) for col in zip(*cost)])]
    INF = float("inf")
    u = [0.0] * (n + 1)
    v = [0.0] * (m + 1)
    p = [0] * (m + 1)
    way = [0] * (m + 1)
    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (m + 1)
        used = [False] * (m + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            row = cost[i0 - 1]
            for j in range(1, m + 1):
                if not used[j]:
                    cur = row[j - 1] - u[i0] - v[j]
                    if cur < minv[j]:
                        minv[j] = cur
                        way[j] = j0
                    if minv[j] < delta:
                        delta = minv[j]
                        j1 = j
            for j in range(m + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while True:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1
            if j0 == 0:
                break
    return [(p[j] - 1, j - 1) for j in range(1, m + 1) if p[j] != 0]


# ------------------------------------------------------------------ per-image features

def _dhash(g: Image.Image, w: int, h: int) -> int:
    px = g.resize((w + 1, h), Image.Resampling.BOX).tobytes()
    v = 0
    stride = w + 1
    for r in range(h):
        row = px[r * stride:(r + 1) * stride]
        for c in range(w):
            v = (v << 1) | (row[c + 1] > row[c])
    return v


def blur_effect(g: Image.Image) -> float:
    """No-reference blur (Crete-Roffet et al. 2007): how much of the image's local variation survives a 9-px
    box blur. ~0.21 for a sharp daylight frame, ~0.29 for a typical night frame, 1.0 for a flat image. Max
    over both directions, so horizontal motion blur counts."""
    w, h = g.size
    if w < 16 or h < 16:
        return 0.0
    np = _numpy()
    out = []
    if np is not None:
        a = np.asarray(g, dtype=np.float32)
        for x in (a, a.T):                                 # rows, then columns
            c = np.cumsum(x, axis=1, dtype=np.float32)
            c = np.concatenate([np.zeros((x.shape[0], 1), np.float32), c], axis=1)
            blurred = (c[:, 9:] - c[:, :-9]) * (1 / 9)     # blurred[:, k] = 9-px mean centred on k + 4
            df = np.abs(x[5:-5, 5:-6] - x[5:-5, 6:-5])     # |x[j+1] - x[j]| for j = 5 .. n-7 (margins skipped)
            db_ = np.abs(blurred[5:-5, 2:-1] - blurred[5:-5, 1:-2])   # the same j in the blurred image
            s_f = float(df.sum())
            s_v = float(np.maximum(df - db_, 0).sum())
            out.append(1.0 - s_v / s_f if s_f > 0 else 1.0)
    else:
        box = (5, 5, w - 5, h - 5)
        for dx, dy in ((1, 0), (0, 1)):
            blurred = g.filter(ImageFilter.BoxBlur((4, 0) if dx else (0, 4)))
            df = ImageChops.difference(g, ImageChops.offset(g, dx, dy))
            db_ = ImageChops.difference(blurred, ImageChops.offset(blurred, dx, dy))
            s_f = ImageStat.Stat(df.crop(box)).sum[0]
            s_v = ImageStat.Stat(ImageChops.subtract(df, db_).crop(box)).sum[0]   # subtract clips at 0
            out.append(1.0 - s_v / s_f if s_f > 0 else 1.0)
    return float(max(out))


class TooLarge(Exception):
    """The image declares more pixels than FOVEA_MAX_IMAGE_PIXELS allows to decode (after JPEG draft scaling)."""


# EXIF orientation -> the transpose that shows the picture upright (as ImageOps.exif_transpose does)
_ORIENT = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
           5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
           8: Image.Transpose.ROTATE_90}
BIG_DECODE_PX = 24_000_000                       # rasters above this decode one at a time (memory, not CPU)
_big_decode = threading.BoundedSemaphore(1)


def _analysis_image(path: str) -> Optional[Image.Image]:
    """The upright greyscale picture, at most ANALYSIS_PX on its long side. None when it cannot be decoded;
    raises TooLarge when the decode itself would exceed the pixel cap (a few-MB PNG can declare 60k x 60k)."""
    try:
        with Image.open(path) as im:
            try:
                im.draft("L", (ANALYSIS_PX, ANALYSIS_PX))      # JPEG: decode at 1/2..1/8 scale in the DCT domain
            except Exception:  # noqa: BLE001
                pass
            px = im.width * im.height                          # what the decode below allocates (per band)
            if px > MAX_IMAGE_PIXELS:
                raise TooLarge(f"{im.width}x{im.height}")
            big = px > BIG_DECODE_PX
            if big:
                _big_decode.acquire()
            try:
                try:                        # (a PNG's getexif() already decodes it, looking for a trailing eXIf)
                    orient = int(im.getexif().get(0x0112) or 1)
                except Exception:  # noqa: BLE001 — malformed EXIF: keep the raster as stored
                    orient = 1
                if im.mode in ("I", "I;16", "I;16B", "I;16L", "I;16N", "F"):
                    from .thumbs import to_display_rgb   # noqa: PLC0415
                    g = to_display_rgb(im).convert("L")
                else:
                    g = im.convert("L")
                # greyscale first, then upright: the same pixels as transposing the colour raster, at a third of
                # the memory
                if orient in _ORIENT:
                    g = g.transpose(_ORIENT[orient])
                if max(g.size) > ANALYSIS_PX:
                    g.thumbnail((ANALYSIS_PX, ANALYSIS_PX), Image.Resampling.BILINEAR)
            finally:
                if big:
                    _big_decode.release()
    except TooLarge:
        raise
    except Image.DecompressionBombError as e:                  # refused by Pillow at open (> 2x its limit)
        raise TooLarge(str(e)) from None
    except Exception:  # noqa: BLE001
        return None
    return g


def image_features(path: str) -> Optional[dict]:
    """dHash 64 + 256 bit, mean luma, contrast (std) and sharpness from one downscaled decode. None when the
    file cannot be decoded or is too large to decode."""
    try:
        return _features(path)
    except TooLarge:
        return None


def _features(path: str) -> Optional[dict]:
    g = _analysis_image(path)
    if g is None or g.width < 2 or g.height < 2:
        return None
    st = ImageStat.Stat(g)
    return {
        "dhash": _dhash(g, 8, 8),
        "fine": _dhash(g, 16, 16),
        "luma": float(st.mean[0]),
        "contrast": float(st.stddev[0]),
        "sharpness": round(1.0 - blur_effect(g), 6),
    }


def _hash_row(image_id: int, dataset_id: str, mtime, f: Optional[dict]) -> tuple:
    if f is None:
        return (image_id, dataset_id, mtime, None, None, None, None, None, None, None, None, image_id)
    words = [to_signed((f["fine"] >> s) & ((1 << 64) - 1)) for s in (192, 128, 64, 0)]
    return (image_id, dataset_id, mtime, to_signed(f["dhash"]), *words, f["luma"], f["contrast"], f["sharpness"], image_id)


_HASH_SQL = """INSERT OR REPLACE INTO image_hash(image_id, dataset_id, mtime, dhash, b0, b1, b2, b3, luma, contrast, sharpness)
SELECT ?,?,?,?,?,?,?,?,?,?,? WHERE EXISTS (SELECT 1 FROM images WHERE id = ?)"""

_generation: Dict[str, int] = defaultdict(int)       # bumped whenever a dataset's hashes change (similar-search cache)


def hash_batches(dataset_id: str):
    """Hash every image that is new or whose file changed since it was hashed (mtime) — computed, never
    written: yields ('hashing', total) once, then ('hash', rows, done, total, images_per_s) per chunk for
    _apply_hash() to store, and returns the counts. (A QA worker process computes; the server writes.)"""
    todo = db.query("""SELECT i.id, i.abs_path, i.mtime FROM images i LEFT JOIN image_hash h ON h.image_id = i.id
                       WHERE i.dataset_id = ? AND (h.image_id IS NULL OR h.mtime IS NOT i.mtime) ORDER BY i.id""",
                    (dataset_id,))
    total = len(todo)
    done = failed = missing = too_large = 0
    t0 = time.time()
    if not total:
        return {"hashed": 0, "failed": 0, "missing": 0, "too_large": 0, "seconds": 0.0}
    yield ("hashing", total)

    def feat(r: dict):
        try:
            return _features(r["abs_path"])
        except TooLarge:
            return TooLarge
    with ThreadPoolExecutor(HASH_THREADS, thread_name_prefix="fovea-qa-hash") as ex:
        for start in range(0, total, HASH_CHUNK):
            chunk = todo[start:start + HASH_CHUNK]
            feats = list(ex.map(feat, chunk))
            rows = []
            for r, f in zip(chunk, feats):
                if f is TooLarge:         # remembered like an undecodable file (no hash) until the file changes
                    too_large += 1
                    f = None
                elif f is None and not os.path.isfile(r["abs_path"]):
                    missing += 1          # unmounted / moved: retry next run, never remember it as undecodable
                    continue
                failed += f is None
                rows.append(_hash_row(r["id"], dataset_id, r["mtime"], f))
            done += len(chunk)
            yield ("hash", rows, done, total, round(done / max(time.time() - t0, 1e-6)))
    return {"hashed": done - missing, "failed": failed, "missing": missing, "too_large": too_large,
            "seconds": round(time.time() - t0, 2)}


def _apply_hash(dataset_id: str, msg: tuple, job: Optional[Job] = None, progress: Optional[Callable] = None) -> None:
    """Store one hash_batches() message and report it."""
    if msg[0] == "hashing":
        if job is not None:
            job.update(done=0, total=msg[1], message="Hashing images", extra={"phase": "hashing"})
        return
    _kind, rows, done, total, rate = msg
    with db.transaction() as conn:
        conn.executemany(_HASH_SQL, rows)
    _generation[dataset_id] += 1
    if job is not None:
        job.update(done=done, total=total, message="Hashing images",       # counts travel in done/total
                   extra={"phase": "hashing", "images_per_s": rate})
    if progress:
        progress(done, total)


def _drain(gen, handle: Callable[[tuple], Any]) -> Any:
    """Feed every message of a generator to `handle`; returns the generator's return value."""
    while True:
        try:
            msg = next(gen)
        except StopIteration as stop:
            return stop.value
        handle(msg)


def update_hashes(dataset_id: str, job: Optional[Job] = None, progress: Optional[Callable] = None) -> dict:
    """Hash (and store) every image that is new or whose file changed since it was hashed. Returns counts."""
    return _drain(hash_batches(dataset_id), lambda msg: _apply_hash(dataset_id, msg, job, progress))


# ------------------------------------------------------------------ loading

_role_cache: Dict[str, str] = {}


def split_role(split: str) -> str:
    """The data.yaml split (train / val / test) a source folder feeds, exactly as Export maps it (by name:
    'test-1854-1860' feeds test, an unknown or empty name trains). Leaks are judged on roles: two folders of
    one role sharing frames do not put evaluation frames into training."""
    split = split or ""
    r = _role_cache.get(split)
    if r is None:
        from .export import yaml_key_for   # noqa: PLC0415 — the one mapping Export writes data.yaml with
        r = _role_cache[split] = yaml_key_for(split)
    return r


class Item:
    __slots__ = ("id", "rel_path", "split", "role", "seq", "W", "H", "size", "has_label", "n_boxes", "coarse", "fine",
                 "luma", "contrast", "sharpness", "lmtime", "boxes")

    def __init__(self, r: dict):
        self.id = r["id"]
        self.rel_path = r["rel_path"]
        self.split = r["split"] or ""
        self.role = split_role(self.split)
        self.seq = r["seq"] or ""
        self.W = float(r["width"] or 1)
        self.H = float(r["height"] or 1)
        self.size = r["size"]
        self.has_label = bool(r["has_label"])
        self.n_boxes = int(r["n_boxes"] or 0)
        self.coarse = coarse_hash(r)
        self.fine = fine_hash(r)
        self.luma = r["luma"]
        self.contrast = r["contrast"]
        self.sharpness = r["sharpness"]
        self.lmtime = r.get("lmtime")
        self.boxes: List[list] = []


_ITEM_COLS = """i.id, i.rel_path, i.split, i.seq, i.width, i.height, i.size, i.has_label, i.n_boxes, i.lmtime,
                h.dhash, h.b0, h.b1, h.b2, h.b3, h.luma, h.contrast, h.sharpness"""


def load_items(dataset_id: str, with_boxes: bool = True) -> List[Item]:
    rows = db.query(f"""SELECT {_ITEM_COLS} FROM images i LEFT JOIN image_hash h ON h.image_id = i.id
                        WHERE i.dataset_id = ? ORDER BY i.id""", (dataset_id,))
    items = [Item(r) for r in rows]
    if with_boxes and items:
        by_id = {it.id: it for it in items}
        cur = db.connect().execute("SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE dataset_id = ? ORDER BY image_id, id",
                                   (dataset_id,))
        for image_id, cls, xc, yc, w, h in cur:
            it = by_id.get(image_id)
            if it is not None:
                it.boxes.append([int(cls), xc, yc, w, h])
    return items


def load_subset(dataset_id: str, image_ids: Iterable[int], with_boxes: bool = True) -> List[Item]:
    """load_items() for some images only (same columns, id order and box order)."""
    ids = sorted({int(i) for i in image_ids})
    items: List[Item] = []
    for s in range(0, len(ids), 5000):
        chunk = db.dumps(ids[s:s + 5000])
        rows = db.query(f"""SELECT {_ITEM_COLS} FROM images i LEFT JOIN image_hash h ON h.image_id = i.id
                            WHERE i.dataset_id = ? AND i.id IN (SELECT value FROM json_each(?)) ORDER BY i.id""",
                        (dataset_id, chunk))
        part = [Item(r) for r in rows]
        by_id = {it.id: it for it in part}
        for image_id, cls, xc, yc, w, h in db.connect().execute(
                "SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE image_id IN (SELECT value FROM json_each(?)) "
                "ORDER BY image_id, id", (chunk,)) if with_boxes else ():
            it = by_id.get(image_id)
            if it is not None:
                it.boxes.append([int(cls), xc, yc, w, h])
        items.extend(part)
    return items


# ------------------------------------------------------------------ near-duplicates, leakage, label mismatch

def near_pairs(hashes: Sequence[int], max_dist: int, bits: int = 256) -> List[Tuple[int, int, int]]:
    """All index pairs (i < j) of DISTINCT hashes within `max_dist` bits. Pigeonhole LSH: split the hash
    into max_dist + 1 bands; two hashes that differ in <= max_dist bits agree exactly on at least one band,
    so only bucket-mates are compared — no pair within the threshold is ever missed."""
    n_bands = max(1, min(int(max_dist) + 1, bits))
    edges = [round(k * bits / n_bands) for k in range(n_bands + 1)]
    seen: Set[Tuple[int, int]] = set()
    out: List[Tuple[int, int, int]] = []
    np = _numpy()
    for b in range(n_bands):
        lo, hi = edges[b], edges[b + 1]
        mask = (1 << (hi - lo)) - 1
        buckets: Dict[int, List[int]] = defaultdict(list)
        for i, h in enumerate(hashes):
            buckets[(h >> lo) & mask].append(i)
        for members in buckets.values():
            m = len(members)
            if m < 2:
                continue
            if np is not None and m > 400:
                _bucket_pairs_np(np, hashes, members, max_dist, seen, out, bits)
                continue
            for x in range(m):
                i = members[x]
                hi_ = hashes[i]
                for y in range(x + 1, m):
                    j = members[y]
                    d = (hi_ ^ hashes[j]).bit_count()
                    if d <= max_dist:
                        key = (i, j) if i < j else (j, i)
                        if key not in seen:
                            seen.add(key)
                            out.append((key[0], key[1], d))
    return out


def _bucket_pairs_np(np, hashes, members, max_dist, seen, out, bits) -> None:
    """A big bucket (static camera: thousands of frames share a band) compared blockwise in numpy."""
    words = bits // 64
    arr = np.array([[(hashes[i] >> (64 * (words - 1 - w))) & ((1 << 64) - 1) for w in range(words)] for i in members],
                   dtype=np.uint64)
    idx = np.array(members)
    for s in range(0, len(members), 512):
        block = arr[s:s + 512]
        d = np.zeros((len(block), len(members)), dtype=np.int32)
        for w in range(words):
            d += _popcount(np, block[:, w:w + 1] ^ arr[None, :, w]).astype(np.int32)
        rows, cols = np.nonzero(d <= max_dist)
        for r, c in zip(rows.tolist(), cols.tolist()):
            if s + r >= c:
                continue
            i, j = int(idx[s + r]), int(idx[c])
            key = (i, j) if i < j else (j, i)
            if key not in seen:
                seen.add(key)
                out.append((key[0], key[1], int(d[r, c])))


def labels_disagree(a: List[list], b: List[list], min_iou: float) -> Optional[str]:
    """None when two label sets describe the same objects (same count and classes, every box paired
    one-to-one with a same-class box at IoU >= min_iou); else 'count' | 'class' | 'boxes'."""
    if len(a) != len(b):
        return "count"
    if sorted(x[0] for x in a) != sorted(x[0] for x in b):
        return "class"
    cand = sorted(((iou(x, y), i, j) for i, x in enumerate(a) for j, y in enumerate(b) if x[0] == y[0]), reverse=True)
    ua: Set[int] = set()
    ub: Set[int] = set()
    for v, i, j in cand:
        if v < min_iou:
            break
        if i in ua or j in ub:
            continue
        ua.add(i)
        ub.add(j)
    return None if len(ua) == len(a) else "boxes"


_REASON_RANK = {"class": 3, "count": 2, "boxes": 1}


class _KeepRow:
    """Just what Export's keep rule reads of an image."""
    __slots__ = ("id", "review", "has_label", "n_boxes")

    def __init__(self, image_id: int, review: Optional[str], has_label: Any, n_boxes: Any):
        self.id, self.review, self.has_label, self.n_boxes = image_id, review, bool(has_label), int(n_boxes or 0)


def keep_member(rows: Iterable[_KeepRow]) -> Optional[int]:
    """The member of a near-duplicate group that "Drop near-duplicates" keeps when an export holds the whole
    group: Export's own rule (approved first, then labeled, most boxes, oldest id), so the duplicates view can
    never name another image. Decided when asked: a review or label edit changes it at once."""
    from .export import _keep_rank   # noqa: PLC0415 — the single implementation of the rule
    best = min(rows, key=_keep_rank, default=None)
    return best.id if best is not None else None


def check_duplicates(items: List[Item], opt: dict, flags: dict, info: dict) -> None:
    hashed = [it for it in items if it.fine is not None]
    if len(hashed) < 2:
        info.update(dup_groups=0, exact_dups=0, pairs=0)
        return
    dup_d, leak_d, mis_d = int(opt["dup_dist"]), int(opt["leak_dist"]), int(opt["dup_dist"])
    # identical 256-bit hashes form cliques (exact copies, all-black frames): collapse them to one node so a
    # thousand identical frames cost one comparison, not half a million.
    nodes: Dict[int, List[Item]] = defaultdict(list)
    for it in hashed:
        nodes[it.fine].append(it)
    keys = list(nodes)
    t0 = time.time()
    pairs = near_pairs(keys, max(dup_d, leak_d, mis_d))
    info["pair_seconds"] = round(time.time() - t0, 3)
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)       # node index -> [(other node, dist)]
    for i, j, d in pairs:
        adj[i].append((j, d))
        adj[j].append((i, d))

    # ---- near_dup groups: greedy leader ("star") clustering, so every member is within dup_dist of its
    # group's leader (union-find would chain a slowly changing video into one giant group).
    def node_weight(n: int) -> int:
        return len(nodes[keys[n]]) - 1 + sum(len(nodes[keys[o]]) for o, d in adj[n] if d <= dup_d)

    candidates = [n for n in range(len(keys)) if node_weight(n) > 0]
    candidates.sort(key=lambda n: (-node_weight(n), min(it.id for it in nodes[keys[n]])))
    group_of: Dict[int, int] = {}
    groups: List[List[Tuple[Item, int]]] = []
    for n in candidates:
        if n in group_of:
            continue
        members: List[Tuple[Item, int]] = [(it, 0) for it in nodes[keys[n]]]
        for o, d in sorted(adj[n], key=lambda t: t[1]):
            if d <= dup_d and o not in group_of:
                members.extend((it, d) for it in nodes[keys[o]])
                group_of[o] = len(groups)
        if len(members) < 2:
            continue
        group_of[n] = len(groups)
        groups.append(members)
    # stable numbering: biggest groups first
    groups.sort(key=lambda g: (-len(g), min(it.id for it, _ in g)))
    # byte-identical copies: same 256-bit hash AND same file size, 64-bit hash and pixel statistics
    copy_of: Dict[int, int] = {}
    for members in nodes.values():
        if len(members) < 2:
            continue
        sig: Dict[tuple, List[Item]] = defaultdict(list)
        for it in members:
            sig[(it.size, it.coarse, it.luma, it.contrast, it.sharpness)].append(it)
        for same in sig.values():
            if len(same) > 1:
                first = min(x.id for x in same)
                for x in same:
                    copy_of[x.id] = first
    for gid, members in enumerate(groups, 1):
        leader = min((it for it, d in members if d == 0), key=lambda it: it.id)
        splits = sorted({it.split for it, _ in members})
        for it, d in members:
            detail = {"group": gid, "dist": d, "size": len(members), "leader": it is leader,
                      "exact": it.id in copy_of, "splits": splits}
            if it.id in copy_of and copy_of[it.id] != it.id:
                detail["copy_of"] = copy_of[it.id]
            flags[(it.id, "near_dup")] = (1.0 - d / (dup_d + 1), detail)
    info["dup_groups"] = len(groups)
    info["exact_dups"] = sum(1 for iid, first in copy_of.items() if iid != first)
    # [group, size, [splits], [member ids: leader first, then by distance and file order]] — what
    # the duplicates listing pages through without re-grouping every flag on each request
    index = []
    for gid, members in enumerate(groups, 1):
        lead = min(it.id for it, d in members if d == 0)
        order = sorted(members, key=lambda m: (m[0].id != lead, m[1], natural_key(m[0].rel_path)))
        index.append([gid, len(members), sorted({it.split for it, _ in members}), [it.id for it, _ in order]])
    info["groups_index"] = index
    info["pairs"] = len(pairs)

    twin_flags(nodes, keys, adj, leak_d, flags)
    mismatch_flags(nodes, keys, pairs, opt, flags)


def twin_flags(nodes: Dict[int, List[Item]], keys: List[int], adj: Dict[int, List[Tuple[int, int]]], leak_d: int,
               flags: dict) -> None:
    """Near-identical images (<= leak_dist) in other folders. In a folder of another ROLE (train / val / test,
    as Export maps folders to data.yaml) the frame is a leak_split: evaluation scores are inflated. In another
    folder of the same role (test + test-1854-1860) it is dup_same_role: the frame counts twice there, but
    nothing leaks. `adj` holds node index -> [(other node, dist)] for every pair within leak_d."""
    per_node: List[Dict[str, Tuple[int, int]]] = []          # node -> {folder: (images, smallest id)}
    for key in keys:
        c: Dict[str, Tuple[int, int]] = {}
        for it in nodes[key]:
            n_, lo = c.get(it.split, (0, it.id))
            c[it.split] = (n_ + 1, min(lo, it.id))
        per_node.append(c)
    for n, key in enumerate(keys):
        near = [(0, n)] + sorted((d, o) for o, d in adj.get(n, ()) if d <= leak_d)
        if len(per_node[n]) < 2 and len(near) == 1:
            continue
        for s in per_node[n]:
            role = split_role(s)
            found = {"leak_split": [0, 999, None, set()], "dup_same_role": [0, 999, None, set()]}
            for d, o in near:
                for s2, (v, lo) in per_node[o].items():
                    if s2 == s:
                        continue
                    f = found["dup_same_role" if split_role(s2) == role else "leak_split"]
                    f[0] += v
                    f[3].add(s2)
                    if d < f[1] or (d == f[1] and lo < f[2]):
                        f[1], f[2] = d, lo
            for code, (twins, best_d, best_id, other) in found.items():
                if not twins:
                    continue
                detail = {"twins": twins, "dist": best_d, "twin": best_id, "splits": sorted(other)}
                if code == "leak_split":
                    detail = {"reason": "twin", **detail, "roles": sorted({split_role(x) for x in other})}
                else:
                    detail["role"] = role
                for it in nodes[key]:
                    if it.split == s:
                        flags[(it.id, code)] = (1.0 - best_d / (leak_d + 1), dict(detail))


def mismatch_flags(nodes: Dict[int, List[Item]], keys: List[int], pairs: List[Tuple[int, int, int]], opt: dict,
                   flags: dict) -> None:
    """dup_label_mismatch: near-identical frames (<= dup_dist) whose labels disagree. Within a set of
    mutually near-identical frames the odd ones out are flagged; a tie flags both sides. `nodes` maps a
    256-bit hash to its images (in id order), `pairs` are (node index, node index, dist) of `keys`."""
    mis_d = int(opt["dup_dist"])
    mm_pairs: List[Tuple[Item, Item, int]] = []
    for members in nodes.values():
        lab = [it for it in members if it.has_label]
        if 1 < len(lab) <= 40:
            mm_pairs.extend((lab[x], lab[y], 0) for x in range(len(lab)) for y in range(x + 1, len(lab)))
        elif len(lab) > 40:
            mm_pairs.extend((lab[0], o, 0) for o in lab[1:])
    for i, j, d in pairs:
        if d > mis_d:
            continue
        la = [it for it in nodes[keys[i]] if it.has_label][:20]
        lb = [it for it in nodes[keys[j]] if it.has_label][:20]
        mm_pairs.extend((x, y, d) for x in la for y in lb)
    by_id = {it.id: it for members in nodes.values() for it in members}
    agree: Dict[int, int] = Counter()
    disagree: Dict[int, List[Tuple[Item, int, str]]] = defaultdict(list)
    for a, b, d in mm_pairs:
        why = labels_disagree(a.boxes, b.boxes, float(opt["mismatch_iou"]))
        if why is None:
            agree[a.id] += 1
            agree[b.id] += 1
        else:
            disagree[a.id].append((b, d, why))
            disagree[b.id].append((a, d, why))
    for iid, lst in disagree.items():
        it = by_id[iid]
        # an image that agrees with more near-identical frames than it disagrees with is the consistent one
        if agree.get(iid, 0) > len(lst):
            continue
        other, d, why = max(lst, key=lambda t: (_REASON_RANK[t[2]], -t[1], -t[0].id))
        flags[(iid, "dup_label_mismatch")] = (_REASON_RANK[why] / 3.0, {
            "twin": other.id, "twin_path": other.rel_path, "dist": d, "reason": why, "twins": len(lst),
            "n_boxes": it.n_boxes, "twin_n_boxes": other.n_boxes})


def spanning_sequences(items: List[Item], flags: dict, opt: dict, limit: int = 200) -> List[dict]:
    """Sequences (images.seq) whose frames feed more than one split ROLE (train / val / test, as Export maps
    folders): frames of one video on both sides of a train/evaluation boundary. A sequence is evidence of
    leakage when its frames in different roles show the same camera view (64-bit dHash <= scene_dist); those
    frames get `leak_split` with reason 'seq'. Name-only matches (e.g. phone photos all called IMG_xxxx) are
    listed but not flagged. A video split over two folders of one role (test, test-1854-1860) is not listed."""
    by_seq: Dict[str, List[Item]] = defaultdict(list)
    for it in items:
        if it.seq:
            by_seq[it.seq].append(it)
    out = []
    scene_d = int(opt["scene_dist"])
    np = _numpy()
    for seq, members in by_seq.items():
        by_role = Counter(it.role for it in members)
        if len(by_role) < 2:
            continue
        by_split = Counter(it.split for it in members)
        hashed = [it for it in members if it.coarse is not None]
        matched: Set[int] = set()
        checked = True
        if len(hashed) <= 6000 and np is not None:
            arr = np.array([it.coarse for it in hashed], dtype=np.uint64)
            rol = np.array([it.role for it in hashed])
            for s in range(0, len(hashed), 256):                 # 256 x 6000 block: ~12 MB at a time
                d = _popcount(np, arr[s:s + 256, None] ^ arr[None, :])
                close = (d <= scene_d) & (rol[s:s + 256, None] != rol[None, :])
                for r in np.nonzero(close.any(axis=1))[0].tolist():
                    matched.add(hashed[s + r].id)
        elif len(hashed) <= 1500:
            for x in hashed:
                for y in hashed:
                    if x.role != y.role and hamming(x.coarse, y.coarse) <= scene_d:
                        matched.add(x.id)
                        break
        else:
            checked = False
        twins = sum(1 for it in members if (it.id, "leak_split") in flags)
        for it in members:
            if it.id in matched and (it.id, "leak_split") not in flags:
                flags[(it.id, "leak_split")] = (0.5, {
                    "reason": "seq", "seq": seq, "splits": sorted(s for s in by_split if split_role(s) != it.role),
                    "roles": sorted(r for r in by_role if r != it.role)})
        out.append({"seq": seq, "splits": sorted(by_split), "images": len(members), "by_split": dict(by_split),
                    "roles": sorted(by_role), "by_role": dict(by_role),
                    "same_view": len(matched), "twins": twins, "evidence": bool(matched) or twins > 0,
                    "checked": checked})
    out.sort(key=lambda r: (not r["evidence"], -r["same_view"], -r["images"], r["seq"]))
    return out[:limit]


def leaked_images(dataset_id: str, roles: Dict[int, str], opt: Optional[dict] = None) -> Optional[Set[int]]:
    """The images among `roles` (image id -> the split it is exported to) that leak, by the leak_split
    definition of a QA run (a near-identical image within leak_dist, or a same-view frame of its sequence,
    in another split), restricted to these images and judged on THESE roles. An export's dataset card uses it
    so its leak count and the Overview's are one definition, whatever the export selects or re-splits.
    None when the dataset has no duplicate check of its own to go by (never run, or near-duplicate flags a QA
    run did not write: they carry no `leader`)."""
    s = load_summary(dataset_id) or {}
    if not s.get("computed_at") or "duplicates" not in (s.get("checks_at") or {}):
        return None
    if db.query_one("SELECT 1 AS x FROM image_flags WHERE dataset_id=? AND code='near_dup' "
                    "AND json_extract(detail, '$.leader') IS NULL LIMIT 1", (dataset_id,)):
        return None
    opt = opt or options_for(dataset_id)
    items = [it for it in load_subset(dataset_id, roles, with_boxes=False) if it.id in roles]
    for it in items:
        it.split = it.role = roles[it.id]
    flags: Dict[Tuple[int, str], Tuple[float, dict]] = {}
    nodes: Dict[int, List[Item]] = defaultdict(list)
    for it in items:
        if it.fine is not None:
            nodes[it.fine].append(it)
    keys = list(nodes)
    leak_d = int(opt["leak_dist"])
    adj: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for i, j, d in near_pairs(keys, leak_d):
        adj[i].append((j, d))
        adj[j].append((i, d))
    twin_flags(nodes, keys, adj, leak_d, flags)
    spanning_sequences(items, flags, opt, limit=0)
    return {iid for (iid, code) in flags if code == "leak_split"}


# ------------------------------------------------------------------ sequences

def sequence_groups(items: List[Item], opt: dict) -> List[List[List[Item]]]:
    """Shots to run temporal checks on: images grouped by (split, seq) in natural file order, kept only when
    consecutive frames really look like video (median 64-bit distance <= seq_video, so a folder of
    unrelated photos called IMG_0001.. is not a 'sequence'), and cut wherever two consecutive frames are
    > seq_cut apart (a different shot)."""
    groups: Dict[Tuple[str, str], List[Item]] = defaultdict(list)
    for it in items:
        if it.seq:
            groups[(it.split, it.seq)].append(it)
    shots: List[List[List[Item]]] = []
    for members in groups.values():
        if len(members) < 3:
            continue
        members.sort(key=lambda it: natural_key(it.rel_path))
        dists = [hamming(a.coarse, b.coarse) for a, b in zip(members, members[1:])
                 if a.coarse is not None and b.coarse is not None]
        if len(dists) >= max(2, (len(members) - 1) // 2):
            if statistics.median(dists) > opt["seq_video"]:
                continue
        elif _SEQ_RE.match(PurePosixPath(members[0].rel_path).stem) is None:
            continue
        cur: List[Item] = [members[0]]
        seq_shots: List[List[Item]] = []
        for a, b in zip(members, members[1:]):
            if a.coarse is not None and b.coarse is not None and hamming(a.coarse, b.coarse) > opt["seq_cut"]:
                seq_shots.append(cur)
                cur = []
            cur.append(b)
        seq_shots.append(cur)
        shots.append([s for s in seq_shots if len(s) >= 3])
    return shots


def _link(A: List[list], B: List[list], W: float, H: float, steps: int = 1, class_aware: bool = False) -> Dict[int, int]:
    """Frame-to-frame box links (A row -> B row) for sparsely sampled CCTV: a pair is a candidate when the
    boxes overlap or their centres are within ~0.75 diagonals per step, with sizes within 3x. Assignment is
    optimal (Hungarian) on 1 - IoU + normalised centre distance (+0.5 for a class change)."""
    if not A or not B:
        return {}
    big = 1e6
    cost = []
    any_ok = False
    for a in A:
        row = []
        da = _diag(a, W, H)
        area_a = a[3] * a[4]
        for b in B:
            if class_aware and a[0] != b[0]:
                row.append(big)
                continue
            db_ = _diag(b, W, H)
            ratio = (b[3] * b[4]) / area_a if area_a > 0 else big
            v = iou(a, b)
            cd = _centre_dist(a, b, W, H) / max(da, db_, 1e-6)
            if (v >= 0.1 or cd <= 0.75 * steps) and (1 / 3) <= ratio <= 3:
                row.append((1 - v) + cd + (0.5 if a[0] != b[0] else 0.0))
                any_ok = True
            else:
                row.append(big)
        cost.append(row)
    if not any_ok:
        return {}
    return {i: j for i, j in assign(cost) if cost[i][j] < big}


def _weight(prev: Item, cur: Item, nxt: Item) -> float:
    fp, fc, fn = frame_number(prev.rel_path), frame_number(cur.rel_path), frame_number(nxt.rel_path)
    if fp is not None and fc is not None and fn is not None and fp < fc < fn:
        return (fc - fp) / (fn - fp)
    return 0.5


def lerp_box(a: Sequence[float], c: Sequence[float], t: float) -> list:
    return [int(a[0])] + [round(a[k] + (c[k] - a[k]) * t, 6) for k in (1, 2, 3, 4)]


def missing_boxes(prev: List[list], cur: List[list], nxt: List[list], W: float, H: float, w: float,
                  lp: Optional[Dict[int, int]] = None, ln: Optional[Dict[int, int]] = None) -> List[list]:
    """The seq_gap rule, shared by the detector and fill_gap so the fix proposes exactly what the flag reports:
    the expected boxes (interpolated at `w`) of objects present in `prev` and in `nxt` -- linked to each other
    over `cur`, same class -- that have no box in `cur`. Matching is one-to-one throughout: a box of `cur`
    linked to a neighbour's box is that object's and explains nothing else, and a box linked to neither (an
    orphan) near the expected place explains ONE missing object. (vfp290k GOPR0807_00266: two class-8 objects
    18 px apart, one unlabeled; the labeled one's box overlaps the missing one's place at IoU 0.31, which a
    many-to-one "any box nearby" test took for the missing box.)
    `lp` (prev -> cur) and `ln` (cur -> nxt) are the frame links when the caller already has them."""
    if lp is None:
        lp = _link(prev, cur, W, H)
    if ln is None:
        ln = _link(cur, nxt, W, H)
    into_n = set(ln.values())
    a_free = [i for i in range(len(prev)) if i not in lp]
    c_free = [j for j in range(len(nxt)) if j not in into_n]
    if not a_free or not c_free:
        return []
    skip = _link([prev[i] for i in a_free], [nxt[j] for j in c_free], W, H, steps=2, class_aware=True)
    expected = [lerp_box(prev[a_free[ia]], nxt[c_free[jc]], w) for ia, jc in skip.items()]
    into_cur = set(lp.values())
    orphans = [b for k, b in enumerate(cur) if k not in into_cur and k not in ln]
    explained = _explained(expected, orphans, W, H)
    return [e for q, e in enumerate(expected) if q not in explained]


def _explained(expected: List[list], boxes: List[list], W: float, H: float) -> Set[int]:
    """Indices of `expected` that a box of `boxes` accounts for, one box per expected box: same class and
    overlapping (IoU >= 0.1) or with its centre within 0.75 of the expected diagonal (sparse frames move a lot)."""
    if not expected or not boxes:
        return set()
    big = 1e6
    cost = []
    for e in expected:
        de = max(_diag(e, W, H), 1e-6)
        row = []
        for b in boxes:
            v = iou(b, e) if b[0] == e[0] else 0.0
            cd = _centre_dist(b, e, W, H) / de
            row.append((1 - v) + cd if b[0] == e[0] and (v >= 0.1 or cd <= 0.75) else big)
        cost.append(row)
    return {i for i, j in assign(cost) if cost[i][j] < big}


def check_sequences(items: List[Item], opt: dict, flags: dict, info: dict) -> None:
    shots = sequence_groups(items, opt)
    n_seq = n_frames = 0
    for seq_shots in shots:
        if seq_shots:
            n_seq += 1
        for shot in seq_shots:
            frames = [it for it in shot if it.has_label]          # unlabeled frames are progress, not errors
            n_frames += len(frames)
            if len(frames) < 3:
                continue
            W, H = frames[0].W, frames[0].H
            links = [_link(a.boxes, b.boxes, W, H) for a, b in zip(frames, frames[1:])]
            back = [{j: i for i, j in lk.items()} for lk in links]
            for t in range(1, len(frames) - 1):
                p, cur, n = frames[t - 1], frames[t], frames[t + 1]
                lp, ln = links[t - 1], links[t]               # p -> cur, cur -> n
                # --- count spike
                nc, np_, nn = len(cur.boxes), len(p.boxes), len(n.boxes)
                d1, d2 = nc - np_, nc - nn
                need = max(2, math.ceil(0.25 * max(np_, nn)))
                if d1 * d2 > 0 and min(abs(d1), abs(d2)) >= need:
                    flags[(cur.id, "seq_count_spike")] = (min(1.0, min(abs(d1), abs(d2)) / 5), {
                        "n": nc, "prev": np_, "next": nn, "prev_id": p.id, "next_id": n.id})
                w = _weight(p, cur, n)
                # --- gap (fill_gap proposes exactly these boxes)
                gaps = missing_boxes(p.boxes, cur.boxes, n.boxes, W, H, w, lp, ln)
                if gaps:
                    flags[(cur.id, "seq_gap")] = (min(1.0, len(gaps) / 2), {
                        "boxes": gaps, "prev_id": p.id, "next_id": n.id})
                # --- jump: the object stands still in p and n (same class, IoU >= 0.5) but its box in cur is
                # somewhere else or another size: a one-frame localisation slip.
                jumps = []
                for k, b in enumerate(cur.boxes):
                    ia, jc = back[t - 1].get(k), ln.get(k)
                    if ia is None or jc is None:
                        continue
                    a, c = p.boxes[ia], n.boxes[jc]
                    if a[0] != c[0] or iou(a, c) < 0.5:
                        continue
                    e = lerp_box(a, c, w)
                    v = iou(b, e)
                    if v >= 0.3:
                        continue
                    # tiny far-field boxes lose IoU over a pixel or two: also require a real shift or resize
                    shift = _centre_dist(b, e, W, H)
                    ratio = (b[3] * b[4]) / max(e[3] * e[4], 1e-12)
                    if shift >= max(0.5 * _diag(e, W, H), 8.0) or not (1 / 2.25 <= ratio <= 2.25):
                        jumps.append({"row": k, "box": b, "expected": e, "iou": round(v, 2)})
                if jumps:
                    flags[(cur.id, "seq_jump")] = (min(1.0, len(jumps) / 2), {
                        "jumps": jumps, "prev_id": p.id, "next_id": n.id})
            # --- flicker: follow link chains; a run of 1-2 frames whose class differs from equal classes on
            # both sides of it
            chains: List[List[Tuple[int, int]]] = []
            started: Set[Tuple[int, int]] = set()
            for t in range(len(frames)):
                for r in range(len(frames[t].boxes)):
                    if (t, r) in started or (t > 0 and r in back[t - 1]):
                        continue
                    chain = [(t, r)]
                    tt, rr = t, r
                    while tt < len(frames) - 1 and rr in links[tt]:
                        rr = links[tt][rr]
                        tt += 1
                        chain.append((tt, rr))
                    started.update(chain)
                    if len(chain) >= 3:
                        chains.append(chain)
            for chain in chains:
                cls = [frames[t].boxes[r][0] for t, r in chain]
                k = 1
                while k < len(cls) - 1:
                    for run in (1, 2):
                        if k + run < len(cls) and cls[k - 1] == cls[k + run] and all(c != cls[k - 1] for c in cls[k:k + run]):
                            for q in range(k, k + run):
                                t, r = chain[q]
                                fr = frames[t]
                                prev = flags.get((fr.id, "seq_flicker"), (0, {"rows": []}))[1]["rows"]
                                prev.append({"row": r, "cls": cls[q], "expected": cls[k - 1]})
                                flags[(fr.id, "seq_flicker")] = (1.0 if run == 1 else 0.7, {
                                    "rows": prev, "prev_id": frames[chain[k - 1][0]].id,
                                    "next_id": frames[chain[k + run][0]].id})
                            k += run
                            break
                    k += 1
    info["sequences"] = n_seq
    info["sequence_frames"] = n_frames


# ------------------------------------------------------------------ shapes

ShapeRow = Tuple[int, int, int, float, float, float, float, float, float]   # image id, row, cls, xc, yc, w, h, W, H
SHAPE_EDGE = 0.004


def _box_shape(b: Sequence[float], W: float, H: float, edge: float = SHAPE_EDGE) -> Optional[Tuple[Optional[float], float]]:
    """(log pixel aspect or None when the box is cut by the image border, log area) of one box; None for a
    degenerate box."""
    w, h = b[3] * W, b[4] * H
    if w <= 0 or h <= 0:
        return None
    cut = (b[1] - b[3] / 2 <= edge or b[2] - b[4] / 2 <= edge or b[1] + b[3] / 2 >= 1 - edge
           or b[2] + b[4] / 2 >= 1 - edge)
    return (None if cut else math.log(w / h)), math.log(max(b[3] * b[4], 1e-9))


def _shape_hits(r: int, cls: int, la: Optional[float], ls: float, model: Sequence, zt: float) -> List[dict]:
    """model = [aspect median, aspect MAD or None (too few uncut boxes), area median, area MAD, boxes]."""
    am, amad, sm, smad = model[0], model[1], model[2], model[3]
    za = (la - am) / amad if la is not None and amad else 0.0
    zs = (ls - sm) / smad
    hits = []
    if abs(za) >= zt:
        hits.append({"row": r, "cls": cls, "kind": "aspect", "z": round(za, 1),
                     "value": round(math.exp(la), 3), "typical": round(math.exp(am), 3)})
    if abs(zs) >= zt:
        hits.append({"row": r, "cls": cls, "kind": "area", "z": round(zs, 1),
                     "value": round(math.exp(ls), 5), "typical": round(math.exp(sm), 5)})
    return hits


def _add_shape_hits(flags: dict, image_id: int, hits: List[dict]) -> None:
    score, detail = flags.get((image_id, "shape_outlier"), (0.0, {"rows": []}))
    detail["rows"].extend(hits)
    flags[(image_id, "shape_outlier")] = (max(score, max(abs(h["z"]) for h in hits)), detail)


def _trim_shape_rows(flags: dict) -> None:
    for (_iid, code), (_score, detail) in list(flags.items()):
        if code == "shape_outlier":
            detail["rows"] = sorted(detail["rows"], key=lambda h: -abs(h["z"]))[:8]


def check_shapes(items: List[Item], opt: dict, flags: dict, info: dict) -> None:
    """Per class, robust z of log aspect (in pixels) and log area. A box cut off by the image border has an
    arbitrary aspect (on VFP290K most aspect 'outliers' were people half out of frame), so its aspect is
    neither scored nor part of the class statistics; its area still is."""
    shape_pass([(it.id, r, int(b[0]), b[1], b[2], b[3], b[4], it.W, it.H) for it in items for r, b in enumerate(it.boxes)],
               opt, flags, info)


def shape_rows(dataset_id: str) -> List[ShapeRow]:
    """Every box of a dataset as a ShapeRow straight from the index — exactly what check_shapes sees through
    load_items(), without building an Item per image (a label edit re-runs the whole shape pass)."""
    conn = db.connect()
    # two plain scans beat a per-box join to images (0.35 s instead of 0.9 s for 260k boxes)
    size = {i: (float(W), float(H)) for i, W, H in conn.execute(
        "SELECT id, COALESCE(NULLIF(width, 0), 1), COALESCE(NULLIF(height, 0), 1) FROM images WHERE dataset_id = ? "
        "ORDER BY id", (dataset_id,))}
    out: List[ShapeRow] = []
    last, r = None, 0
    for iid, cls, xc, yc, w, h in conn.execute(
            "SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE dataset_id = ? ORDER BY image_id, id", (dataset_id,)):
        wh = size.get(iid)
        if wh is None:
            continue
        r = r + 1 if iid == last else 0
        last = iid
        out.append((iid, r, int(cls), xc, yc, w, h, wh[0], wh[1]))
    return out


def shape_pass(rows: List[ShapeRow], opt: dict, flags: dict, info: dict) -> None:
    """shape_outlier over a whole dataset's boxes (rows in image, file order). Sets info['shape_classes'] and
    info['shape_model'] {cls: [aspect median, aspect MAD|None, area median, area MAD, boxes]}."""
    np = _numpy()
    if np is not None and rows:
        _shape_pass_np(np, rows, opt, flags, info)
        return
    per_cls: Dict[int, List[Tuple[int, int, Optional[float], float]]] = defaultdict(list)
    for iid, r, cls, xc, yc, w, h, W, H in rows:
        s = _box_shape((cls, xc, yc, w, h), W, H)
        if s is not None:
            per_cls[cls].append((iid, r, s[0], s[1]))
    zt, n_min = float(opt["shape_z"]), int(opt["shape_min"])
    stats_out: Dict[int, dict] = {}
    model_out: Dict[str, list] = {}
    for cls, crow in per_cls.items():
        if len(crow) < n_min:
            continue
        aspects = [x[2] for x in crow if x[2] is not None]
        am, amad = robust_z(aspects, 0.05) if len(aspects) >= n_min else (0.0, None)
        sm, smad = robust_z([x[3] for x in crow], 0.05)
        model = [am, amad, sm, smad, len(crow)]
        model_out[str(cls)] = model
        stats_out[cls] = {"aspect": round(math.exp(am), 3) if aspects else None, "area": round(math.exp(sm), 5), "boxes": len(crow)}
        for iid, r, la, ls in crow:
            hits = _shape_hits(r, cls, la, ls, model, zt)
            if hits:
                _add_shape_hits(flags, iid, hits)
    _trim_shape_rows(flags)
    info["shape_classes"] = stats_out
    info["shape_model"] = model_out


def _shape_pass_np(np, rows: List[ShapeRow], opt: dict, flags: dict, info: dict) -> None:
    """shape_pass in numpy (the same arithmetic, element-wise): ~10x faster, so a label edit can afford an
    exact re-run over every box of the dataset instead of re-scoring against stale class statistics. The
    per-box arrays are handed back in info['shape_arrays'] for the re-check cache."""
    A = _shape_arrays(np, rows)
    _shape_from_arrays(np, A, opt, flags, info)
    info["shape_arrays"] = A


def _shape_arrays(np, rows: List[ShapeRow]) -> dict:
    """Per-box log aspect / log area / border cut of ShapeRows, in (image, row) order."""
    a = np.asarray(rows, dtype=np.float64).reshape(-1, 9)
    xc, yc, w, h, W, H = (a[:, k] for k in range(3, 9))
    wp, hp = w * W, h * H
    e = SHAPE_EDGE
    with np.errstate(all="ignore"):
        la = np.log(wp / hp)
        ls = np.log(np.maximum(w * h, 1e-9))
    return {"iid": a[:, 0].astype(np.int64), "row": a[:, 1].astype(np.int64), "cls": a[:, 2].astype(np.int64),
            "ok": (wp > 0) & (hp > 0), "la": la, "ls": ls,
            "cut": (xc - w / 2 <= e) | (yc - h / 2 <= e) | (xc + w / 2 >= 1 - e) | (yc + h / 2 >= 1 - e)}


def _shape_from_arrays(np, A: dict, opt: dict, flags: dict, info: dict) -> None:
    iid, row, cls, cut, la, ls = A["iid"], A["row"], A["cls"], A["cut"], A["la"], A["ls"]
    idx = np.nonzero(A["ok"])[0]
    idx = idx[np.argsort(cls[idx], kind="stable")]          # grouped by class, file order inside a class
    cuts = np.flatnonzero(np.diff(cls[idx])) + 1
    groups = sorted(np.split(idx, cuts), key=lambda g: int(g[0])) if len(idx) else []   # classes by first appearance
    zt, n_min = float(opt["shape_z"]), int(opt["shape_min"])

    def rz(v):
        med = float(np.median(v))
        return med, max(float(np.median(np.abs(v - med))) * 1.4826, 0.05)

    stats_out: Dict[int, dict] = {}
    model_out: Dict[str, list] = {}
    for g in groups:
        c = int(cls[g[0]])
        if len(g) < n_min:
            continue
        uncut = g[~cut[g]]
        am, amad = rz(la[uncut]) if len(uncut) >= n_min else (0.0, None)
        sm, smad = rz(ls[g])
        model = [am, amad, sm, smad, int(len(g))]
        model_out[str(c)] = model
        stats_out[c] = {"aspect": round(math.exp(am), 3) if len(uncut) else None, "area": round(math.exp(sm), 5), "boxes": int(len(g))}
        za = np.where(cut[g], 0.0, (la[g] - am) / amad) if amad else np.zeros(len(g))
        zs = (ls[g] - sm) / smad
        for k in g[(np.abs(za) >= zt) | (np.abs(zs) >= zt)].tolist():
            hits = _shape_hits(int(row[k]), c, None if cut[k] else float(la[k]), float(ls[k]), model, zt)
            if hits:
                _add_shape_hits(flags, int(iid[k]), hits)
    _trim_shape_rows(flags)
    info["shape_classes"] = stats_out
    info["shape_model"] = model_out


# Per-dataset box-shape arrays kept between label edits (numpy only), so an edit re-runs the exact shape pass
# without reading every box back from SQLite (~1 s at 260k boxes): the edited images' rows are swapped in.
# Refreshed by every full shape pass; rebuilt whenever its box count disagrees with the index.
_shape_cache: Dict[str, dict] = {}
_shape_cache_lock = threading.Lock()
SHAPE_CACHE_DATASETS = 4


def _shape_cache_put(dataset_id: str, A: dict) -> None:
    with _shape_cache_lock:
        _shape_cache.pop(dataset_id, None)
        _shape_cache[dataset_id] = A
        while len(_shape_cache) > SHAPE_CACHE_DATASETS:
            _shape_cache.pop(next(iter(_shape_cache)))


def _shape_cache_take(dataset_id: str) -> Optional[dict]:
    with _shape_cache_lock:
        return _shape_cache.pop(dataset_id, None)


def _shape_arrays_after_edit(np, dataset_id: str, edited: Set[int], items: Dict[int, Item]) -> Tuple[dict, bool]:
    """The dataset's shape arrays with the edited images' boxes replaced by their current rows. (arrays,
    from_cache): rebuilt from the index when there is no cache or the result disagrees with box_count."""
    n_boxes = int((db.query_one("SELECT box_count FROM datasets WHERE id=?", (dataset_id,)) or {}).get("box_count") or 0)
    A = _shape_cache_take(dataset_id)
    if A is not None:
        fresh = [(it.id, r, int(b[0]), b[1], b[2], b[3], b[4], it.W, it.H)
                 for i in sorted(edited) if (it := items.get(i)) is not None for r, b in enumerate(it.boxes)]
        keep = ~np.isin(A["iid"], np.fromiter(edited, dtype=np.int64, count=len(edited)))
        B = _shape_arrays(np, fresh)
        A = {k: np.concatenate([A[k][keep], B[k]]) for k in A}
        order = np.lexsort((A["row"], A["iid"]))
        A = {k: v[order] for k, v in A.items()}
        if len(A["iid"]) == n_boxes:
            return A, True
    return _shape_arrays(np, shape_rows(dataset_id)), False


# ------------------------------------------------------------------ quality

def check_quality(items: List[Item], opt: dict, flags: dict, info: dict) -> None:
    """dark / overexposed: mean luma beyond an absolute floor AND a robust z against this dataset, so a
    uniformly dark CCTV set is not flagged wholesale. blurry: the blur effect is compared with images of
    similar brightness (luma quantile strata): night frames are softer by nature (median blur 0.29 vs 0.21
    by day on VFP290K), so one global threshold flagged sharp night frames and missed blurred day frames."""
    hashed = [it for it in items if it.luma is not None and it.sharpness is not None]
    if len(hashed) < 5:
        return
    lm, lmad = robust_z([it.luma for it in hashed], 2.0)
    for it in hashed:
        zl = (it.luma - lm) / lmad
        if it.luma < opt["dark_floor"] and zl <= -opt["dark_z"]:
            flags[(it.id, "dark")] = (round(-zl, 2), {"luma": round(it.luma, 1), "median": round(lm, 1), "z": round(zl, 1)})
        elif it.luma > opt["bright_floor"] and zl >= opt["bright_z"]:
            flags[(it.id, "overexposed")] = (round(zl, 2), {"luma": round(it.luma, 1), "median": round(lm, 1), "z": round(zl, 1)})
    n_bins = max(1, min(8, len(hashed) // 60))
    lumas = sorted(it.luma for it in hashed)
    edges = [lumas[len(lumas) * k // n_bins] for k in range(1, n_bins)]
    strata: Dict[int, List[Item]] = defaultdict(list)
    for it in hashed:
        strata[sum(1 for e in edges if it.luma >= e)].append(it)
    medians = []
    for members in strata.values():
        bm, bmad = robust_z([1.0 - it.sharpness for it in members], 0.01)
        medians.append(round(bm, 3))
        for it in members:
            if (it.contrast or 0) < 5:          # a flat frame (black, lens cap) has no detail to be blurry
                continue
            b = 1.0 - it.sharpness
            zb = (b - bm) / bmad
            if b >= opt["blur_floor"] and zb >= opt["blur_z"]:
                flags[(it.id, "blurry")] = (round(zb, 2), {"blur": round(b, 3), "median": round(bm, 3), "z": round(zb, 1)})
    info["quality"] = {"luma_median": round(lm, 1), "blur_medians": sorted(medians), "strata": n_bins}


# ------------------------------------------------------------------ the job

def normalise_checks(checks: Optional[Iterable[str]]) -> List[str]:
    if not checks:
        return list(CHECKS)
    out = []
    for c in checks:
        c = str(c).strip()
        if c == "labels":
            out.extend(LABEL_CHECKS)
        elif c in CHECKS:
            out.append(c)
        elif c in CODES:
            out.extend(k for k, codes in CHECKS.items() if c in codes)
        else:
            raise ValueError(f"Unknown check: {c}")
    return [c for c in CHECKS if c in set(out)]


def options_for(dataset_id: str, overrides: Optional[dict] = None) -> dict:
    saved = (load_summary(dataset_id) or {}).get("options") or {}
    opt = {**DEFAULTS, **{k: v for k, v in saved.items() if k in DEFAULTS}}
    for k, v in (overrides or {}).items():
        if k not in DEFAULTS:
            raise ValueError(f"Unknown option: {k}")
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"Option {k} must be a number") from None
        lo, hi = OPTION_LIMITS.get(k, (-1e9, 1e9))
        if not (lo <= v <= hi) or math.isnan(v):
            raise ValueError(f"Option {k} must be between {lo:g} and {hi:g}")
        opt[k] = int(v) if isinstance(DEFAULTS[k], int) else v
    return opt


_summary_lock = threading.RLock()       # read-modify-write of the summary blob (jobs, settings, label re-checks)


def save_options(dataset_id: str, overrides: dict, reset: Iterable[str] = ()) -> dict:
    """Validate threshold overrides and store them with the dataset's QA results (only values that differ
    from DEFAULTS, so sending a default resets it; keys in `reset` go back to their default). Returns the
    effective options."""
    reset = set(reset)
    bad = reset - set(DEFAULTS)
    if bad:
        raise ValueError(f"Unknown option: {sorted(bad)[0]}")
    with _summary_lock:
        opt = options_for(dataset_id, overrides)
        for k in reset:
            opt[k] = DEFAULTS[k]
        summary = load_summary(dataset_id) or {}
        summary["options"] = {k: opt[k] for k in DEFAULTS if opt[k] != DEFAULTS[k]}
        _save_summary(dataset_id, summary)
    return opt


def load_summary(dataset_id: str) -> Optional[dict]:
    row = db.query_one("SELECT value FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?",
                       (PLUGIN_KEY[0], dataset_id, PLUGIN_KEY[1]))
    return db.loads(row["value"], {}) if row else None


def _save_plugin(dataset_id: str, key: str, value: Any) -> None:
    db.execute("""INSERT INTO plugin_data(plugin, dataset_id, key, value, updated_at) VALUES(?,?,?,?,?)
                  ON CONFLICT(plugin, dataset_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
               (PLUGIN_KEY[0], dataset_id, key, db.dumps(value), db.now()))


def _save_summary(dataset_id: str, summary: dict) -> None:
    _save_plugin(dataset_id, PLUGIN_KEY[1], summary)


def _scan_settled(dataset_id: str, job: Job, wait: float = 3.0) -> bool:
    """A rescan prunes and rewrites rows; checking half-indexed data would flag nonsense. The post-scan
    hook queues this job while its scan job is still finishing, so a short wait covers that case. A scan
    that is really still running will queue a fresh QA run itself when it ends (POST_SCAN_HOOKS), so this
    job then steps aside instead of holding a worker for the whole scan."""
    t0 = time.time()
    while True:
        active = jobs.active_for(dataset_id, "scan")
        if active is None:
            return True
        if time.time() - t0 > wait:
            return False
        job.update(message="Waiting for indexing to finish")
        time.sleep(0.1)


def run_qa(dataset_id: str, job: Job, checks: Optional[Iterable[str]] = None, options: Optional[dict] = None) -> dict:
    if not db.query_one("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)):
        raise ValueError("dataset not found")
    checks = normalise_checks(checks)
    # replace the queue message at once: a running job must never still read "Queued after indexing"
    job.update(message="Starting", extra={"checks": checks, "phase": "starting"})
    opt = save_options(dataset_id, options) if options else options_for(dataset_id)
    if not _scan_settled(dataset_id, job):
        job.update(message="Indexing in progress: checks will run when it finishes")
        return {"deferred": True, "checks": checks}
    t0 = time.time()
    sink = _RunSink(dataset_id, job)
    in_worker = compute_run(dataset_id, checks, opt, bool({"duplicates", "sequence", "quality"} & set(checks)), job, sink)
    res, info = sink.result, sink.result["info"]
    timings = {**({"hash": sink.hash_info["seconds"]} if sink.hash_info else {}), **res["timings"]}
    if "shape_arrays" in info:
        _shape_cache_put(dataset_id, info.pop("shape_arrays"))
    codes = [c for name in checks for c in CHECKS[name]]
    rows = [(iid, dataset_id, code, score, detail, iid) for iid, code, score, detail in sink.rows]
    with db.transaction() as conn:
        conn.execute("DELETE FROM image_flags WHERE dataset_id=? AND code IN (%s)" % ",".join("?" * len(codes)),
                     [dataset_id, *codes])
        conn.executemany("""INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail)
                            SELECT ?,?,?,?,? WHERE EXISTS (SELECT 1 FROM images WHERE id = ?)""", rows)
    if not db.query_one("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)):
        raise ValueError("dataset was removed while it was being checked")
    with _summary_lock:
        _finish_summary(dataset_id, checks, res, info, timings, sink.hash_info, t0)
    _generation[dataset_id] += 1
    _refresh_planner_stats(len(rows))
    job.update(message="Done", extra={"phase": "done"})
    return {"checks": checks, "counts": res["counts"], "images": res["images"], "seconds": round(time.time() - t0, 2),
            "hashed": (sink.hash_info or {}).get("hashed", 0), "process": in_worker}


def _finish_summary(dataset_id: str, checks: List[str], res: dict, info: dict, timings: dict,
                    hash_info: Optional[dict], t0: float) -> None:
    now = db.now()
    latest = load_summary(dataset_id) or {}          # thresholds may have been changed while this ran
    summary = {
        **latest,
        "computed_at": now,
        "checks_at": {**latest.get("checks_at", {}), **{c: res["snapshot_at"] for c in checks}},
        "images": res["images"],
        "hashed": res["hashed_images"],
        "unreadable": (db.query_one("SELECT COUNT(*) c FROM image_hash WHERE dataset_id=? AND dhash IS NULL",
                                    (dataset_id,)) or {"c": 0})["c"],
        "timings": {**latest.get("timings", {}), **timings, "total": round(time.time() - t0, 2)},
        "hash": hash_info if hash_info is not None else latest.get("hash", {}),
    }
    if "duplicates" in checks:
        summary.update(dup_groups=info.get("dup_groups", 0), exact_dups=info.get("exact_dups", 0),
                       split_spanning_sequences=res["spanning"], engine=ENGINE_VERSION)
        _save_plugin(dataset_id, GROUPS_KEY, info.get("groups_index", []))
    if "sequence" in checks:
        summary.update(sequences=info.get("sequences", 0), sequence_frames=info.get("sequence_frames", 0))
    if "quality" in checks and "quality" in info:
        summary["quality"] = info["quality"]
    if "shape" in checks:
        summary["shape_model"] = info.get("shape_model", {})
    _save_summary(dataset_id, summary)


FLAG_CHUNK = 5000


def _compute(dataset_id: str, checks: List[str], opt: dict, rehash: bool):
    """Everything a QA run computes, as a stream of messages for _RunSink; it reads the index and never writes
    it: ('hashing' | 'hash', ...) from hash_batches(), ('hashed', counts), ('phase', name, message),
    ('flags', [(image id, code, score, detail json)]) in chunks, then ('done', results)."""
    if rehash:
        yield ("hashed", (yield from hash_batches(dataset_id)))
    yield ("phase", "loading", "Loading labels")
    t = time.time()
    snapshot_at = db.now()          # labels saved after this moment are newer than these results
    items = load_items(dataset_id)
    timings = {"load": round(time.time() - t, 2)}
    flags: Dict[Tuple[int, str], Tuple[float, dict]] = {}
    info: Dict[str, Any] = {}
    spanning = None
    steps = [("duplicates", "Finding near-duplicates", check_duplicates),
             ("sequence", "Checking sequences", check_sequences),
             ("shape", "Checking box shapes", check_shapes),
             ("quality", "Checking image quality", check_quality)]
    for name, msg, fn in steps:
        if name not in checks:
            continue
        yield ("phase", name, msg)
        t = time.time()
        fn(items, opt, flags, info)
        if name == "duplicates":
            spanning = spanning_sequences(items, flags, opt)
        timings[name] = round(time.time() - t, 2)
    yield ("phase", "saving", "Saving results")
    rows = [(iid, code, float(score), db.dumps(detail)) for (iid, code), (score, detail) in flags.items()]
    counts = Counter(code for (_iid, code) in flags)
    del flags
    for k in range(0, len(rows), FLAG_CHUNK):
        yield ("flags", rows[k:k + FLAG_CHUNK])
    yield ("done", {"snapshot_at": snapshot_at, "timings": timings, "images": len(items),
                    "hashed_images": sum(1 for it in items if it.fine is not None), "counts": dict(counts),
                    "spanning": spanning, "info": info})


class _RunSink:
    """The server's side of a QA run: stores hash rows as they arrive, reports progress on the job, and
    collects the flag rows and results for run_qa's one write transaction."""

    def __init__(self, dataset_id: str, job: Job):
        self.dataset_id, self.job = dataset_id, job
        self.hash_info: Optional[dict] = None
        self.rows: List[tuple] = []
        self.result: Optional[dict] = None

    def __call__(self, msg: tuple) -> None:
        kind = msg[0]
        if kind in ("hashing", "hash"):
            _apply_hash(self.dataset_id, msg, self.job)
        elif kind == "hashed":
            self.hash_info = msg[1]
        elif kind == "phase":
            self.job.update(message=msg[2], extra={"phase": msg[1]})
        elif kind == "flags":
            self.rows.extend(msg[1])
        elif kind == "done":
            self.result = msg[1]


# ------------------------------------------------------------------ worker process
# The checks are pure Python and CPU-bound: seconds at 47k images (hashing a new dataset: minutes). On a thread
# of the server they hold the GIL, and every request thread waits up to a switch interval (5 ms) for it after
# each SQLite step: a 100-row Explore page took 1 s instead of 6 ms while a check ran, and Annotate missed
# frames. So a large run computes in a child process that only reads the index; the server keeps the writes
# (hash rows as they arrive, then the flags transaction), which release the GIL while SQLite works.
PROCESS_MIN_IMAGES = 2000       # smaller runs finish in a fraction of a second here; a process start costs ~0.15 s
                                # (at 7.8k images an in-thread run stalled requests 150 ms, a worker run 2 ms)
WORKER_START_TIMEOUT = 60.0
_worker_broken: List[str] = []  # why a worker could not start in this process: compute in the job thread instead
_live_workers: Dict[int, Any] = {}   # pid -> Popen of the workers running now

# Run with `python -c`: nothing re-imports the server's __main__ (as multiprocessing's spawn would), and no
# resource-tracker process is left behind. The working directory -c puts first on sys.path is dropped before
# any import; the server's own sys.path is installed for fovea's.
_WORKER_BOOT = """import sys
if sys.path and sys.path[0] in ('', '.'):
    del sys.path[0]
import signal
signal.signal(signal.SIGINT, signal.SIG_IGN)
import pickle
args = pickle.load(sys.stdin.buffer)
sys.path[:] = args["sys_path"]
from fovea.core.qa import _worker_main
_worker_main(args)
"""


class _WorkerUnavailable(Exception):
    """The worker process could not start (or died before it said it was ready): compute in this process."""


def compute_run(dataset_id: str, checks: List[str], opt: dict, rehash: bool, job: Job,
                handle: Callable[[tuple], None]) -> bool:
    """Feed _compute()'s messages to `handle`, computed in a worker process when the dataset is large enough
    for its passes to starve the server's request threads. True when a worker computed them."""
    n = (db.query_one("SELECT COUNT(*) c FROM images WHERE dataset_id=?", (dataset_id,)) or {"c": 0})["c"]
    kw = {"dataset_id": dataset_id, "checks": list(checks), "opt": dict(opt), "rehash": rehash}
    return run_task("run", kw, handle, job.check, in_worker=n >= PROCESS_MIN_IMAGES)


def run_task(task: str, kw: dict, handle: Callable[[tuple], None], check: Callable[[], None], in_worker: bool) -> bool:
    """Feed the messages of _WORKER_TASKS[task](**kw) to `handle`: from a worker process when `in_worker`
    (and one can start), else computed right here. `check` raises to abandon the task (JobCancelled). True
    when a worker computed it."""
    if in_worker and not _worker_broken:
        try:
            ok, why = shared_index_safe()
            if not ok:
                raise _WorkerUnavailable(why)
            _run_in_worker(task, kw, handle, check)
            return True
        except _WorkerUnavailable as e:
            _worker_broken.append(str(e))
            log.warning("Data quality checks run in the server process: %s", e)
    _drain(_WORKER_TASKS[task](**kw), handle)
    return False


# A worker opens the index as a second SQLite process, which is only safe where POSIX file locks reach across
# processes. On a Docker Desktop bind mount on macOS (`./data:/app/data`) they do not: the worker's connection
# believes it is the only one, and when it closes it checkpoints and DELETES the server's -wal/-shm files. Every
# new server connection then fails with "disk I/O error" until a restart, and commits can be lost. So before the
# first worker, a child process opens and closes a probe database next to the index while this process holds it
# open in WAL mode, exactly as a worker would; if our -wal/-shm did not survive that, no worker ever starts here.
_LOCK_PROBE_CHILD = """import sqlite3, sys
c = sqlite3.connect(sys.argv[1], timeout=10)
c.execute("SELECT COUNT(*) FROM t").fetchone()
c.close()
"""
LOCK_PROBE_TIMEOUT = 30.0
_lock_probe: Dict[Tuple[str, str], Tuple[bool, str]] = {}     # (index folder, python) -> result
_lock_probe_lock = threading.Lock()


def shared_index_safe() -> Tuple[bool, str]:
    """(True, '') when a second process may open the index (probed once per directory per server process),
    else (False, why)."""
    folder = str(Path(db.DB_PATH).parent)
    key = (folder, sys.executable or "")
    with _lock_probe_lock:
        hit = _lock_probe.get(key)
        if hit is None:
            hit = _lock_probe[key] = probe_cross_process_locks(Path(folder))
            if not hit[0]:
                hit = _lock_probe[key] = (False, f"{hit[1]} [{folder}]")
        return hit


def probe_cross_process_locks(directory: Path, child: Optional[str] = None) -> Tuple[bool, str]:
    """Does SQLite's WAL survive another process opening and closing the database in `directory`?"""
    import sqlite3  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    import uuid  # noqa: PLC0415
    path = Path(directory) / f".fovea-lock-probe-{os.getpid()}-{uuid.uuid4().hex[:8]}.db"
    side = [Path(f"{path}{s}") for s in ("-wal", "-shm", "-journal")]
    conn = None
    try:
        if not sys.executable:
            return False, "no Python executable to probe file locking with"
        conn = sqlite3.connect(str(path), timeout=10, isolation_level=None)
        if str(conn.execute("PRAGMA journal_mode=WAL").fetchone()[0]).lower() != "wal":
            return False, "SQLite cannot use WAL mode in this folder"
        conn.execute("CREATE TABLE t(x)")
        conn.execute("INSERT INTO t VALUES (1)")
        r = subprocess.run([sys.executable, "-I", "-c", child or _LOCK_PROBE_CHILD, str(path)], stdin=subprocess.DEVNULL,
                           stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, timeout=LOCK_PROBE_TIMEOUT, close_fds=True)
        if r.returncode != 0:
            tail = r.stderr.decode("utf-8", "replace").strip().splitlines()[-1:] or [""]
            return False, f"the file-locking probe failed (exit code {r.returncode}{': ' + tail[0][:200] if tail[0] else ''})"
        if not (side[0].exists() and side[1].exists()):
            return False, ("file locks do not reach across processes in this folder (a Docker Desktop bind mount?): "
                           "another process deleted the write-ahead log of a database this one had open")
        conn.execute("INSERT INTO t VALUES (2)")
        if conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] != 2:
            return False, "another process's view of the database diverged from this one's"
        return True, ""
    except (OSError, sqlite3.Error, subprocess.SubprocessError, ValueError) as e:
        return False, f"the file-locking probe failed: {e}"
    finally:
        if conn is not None:
            try:
                conn.close()
            except sqlite3.Error:
                pass
        for p in [path, *side]:
            try:
                p.unlink()
            except FileNotFoundError:
                pass
            except OSError:
                log.debug("Could not remove %s", p, exc_info=True)


def _run_in_worker(task: str, kw: dict, handle: Callable[[tuple], None], check: Callable[[], None]) -> None:
    import pickle  # noqa: PLC0415
    import queue  # noqa: PLC0415
    import subprocess  # noqa: PLC0415
    if not sys.executable:
        raise _WorkerUnavailable("no Python executable")
    try:
        proc = subprocess.Popen([sys.executable, "-c", _WORKER_BOOT], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                close_fds=True, start_new_session=os.name != "nt")
    except (OSError, ValueError) as e:
        raise _WorkerUnavailable(str(e)) from e
    _live_workers[proc.pid] = proc
    msgs: "queue.Queue[Optional[tuple]]" = queue.Queue()

    def read() -> None:                 # pickles from our own child over a private pipe
        try:
            while True:
                msgs.put(pickle.load(proc.stdout))
        except EOFError:
            pass
        except Exception as e:  # noqa: BLE001 — a truncated stream (the worker was killed mid-message)
            msgs.put(("broken", repr(e)))
        finally:
            msgs.put(None)
            proc.stdout.close()

    def send(obj: Any) -> None:
        pickle.dump(obj, proc.stdin, protocol=pickle.HIGHEST_PROTOCOL)
        proc.stdin.flush()

    finished = ready = False
    try:
        try:
            send({"sys_path": list(sys.path), "db_path": str(db.DB_PATH), "task": task, "kw": kw})
        except OSError as e:                        # it exited before reading anything
            raise _WorkerUnavailable(f"exit code {proc.poll()}: {e}") from e
        threading.Thread(target=read, name="fovea-qa-worker-io", daemon=True).start()
        started = time.monotonic()
        while True:
            try:
                msg = msgs.get(timeout=0.25)
            except queue.Empty:
                check()                             # cancel / server shutdown: the finally clause stops it
                if not ready and time.monotonic() - started > WORKER_START_TIMEOUT:
                    raise _WorkerUnavailable(f"no answer within {WORKER_START_TIMEOUT:g} s") from None
                continue
            if msg is None or msg[0] == "broken":
                try:
                    code = proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    code = None
                if not ready:
                    raise _WorkerUnavailable(f"exit code {code}")
                raise RuntimeError(f"The data quality worker stopped unexpectedly (exit code {code})")
            kind = msg[0]
            if kind == "ready":
                ready = True
                continue
            if kind == "error":
                _raise_worker_error(msg)
            check()
            handle(msg)
            if kind == "hashed":
                try:
                    send("ok")                      # its hash rows are committed: it may read them back now
                except OSError:
                    pass                            # it died: the next read says so
            elif kind == "done":
                finished = True
                return
    finally:
        try:
            proc.stdin.close()
        except OSError:
            pass
        try:
            proc.wait(timeout=5 if finished else 0)
        except subprocess.TimeoutExpired:
            proc.kill()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                log.warning("Data quality worker %s did not exit", proc.pid)
        _live_workers.pop(proc.pid, None)


_WORKER_ERRORS = {"ValueError": ValueError, "QAError": ValueError, "OSError": OSError, "FileNotFoundError": OSError,
                  "PermissionError": OSError, "MemoryError": MemoryError}


def _raise_worker_error(msg: tuple) -> None:
    _kind, etype, text, tb = msg
    cls = _WORKER_ERRORS.get(etype)
    if cls is not None and text:
        raise cls(text)
    log.error("The data quality worker failed:\n%s", tb)
    raise RuntimeError(f"{etype}: {text}" if text else etype)


def _worker_main(args: dict) -> None:
    """The worker process (started by _run_in_worker): runs one task of _WORKER_TASKS and streams its messages
    to the server over stdout. Read-only on the index."""
    import pickle  # noqa: PLC0415
    out = os.fdopen(os.dup(1), "wb")
    try:                                # stray prints (a library, a warning) must not corrupt the stream
        os.dup2(2, 1)
    except OSError:
        os.dup2(os.open(os.devnull, os.O_WRONLY), 1)
    sys.stdout = sys.stderr

    def send(obj: Any) -> None:
        pickle.dump(obj, out, protocol=pickle.HIGHEST_PROTOCOL)
        out.flush()
    try:
        db.DB_PATH = Path(args["db_path"])
        db.connect().execute("PRAGMA query_only=1")
        send(("ready", os.getpid()))
        for msg in _WORKER_TASKS[args["task"]](**args["kw"]):
            send(msg)
            if msg[0] == "hashed" and pickle.load(sys.stdin.buffer) != "ok":
                break
    except (BrokenPipeError, ConnectionResetError, EOFError):
        pass                            # the server stopped listening (cancelled, or gone): nothing left to flush
    except BaseException as e:  # noqa: BLE001 — reported to the server, which fails the job with it
        try:
            send(("error", type(e).__name__, str(e), traceback.format_exc()))
        except Exception:  # noqa: BLE001
            pass
    # Leave without closing the index connection: a closing SQLite connection that thinks it is the last one
    # checkpoints and deletes the WAL, and the server is still using it (see shared_index_safe). The OS drops
    # this process's locks; it never wrote, so there is nothing to finish.
    for f in (out, sys.stderr):
        try:
            f.flush()
        except (OSError, ValueError):
            pass
    os._exit(0)


def _refresh_planner_stats(n_rows: int) -> None:
    """After a large flag write, let backend-data's helper re-check the planner statistics (so ?issue=near_dup
    keeps using the image_flags index). Best effort: statistics are an optimisation."""
    if n_rows < 500:
        return
    fn = getattr(db, "refresh_planner_stats", None)
    if fn is None:
        return
    try:
        fn(("image_flags", "image_hash"))
    except Exception:  # noqa: BLE001
        log.warning("Could not refresh planner statistics after a QA run", exc_info=True)


# ------------------------------------------------------------------ queueing

_queue_lock = threading.Lock()
_rerun: Dict[str, Set[str]] = {}


def _job_checks(job: Job) -> Set[str]:
    return set((job.extra or {}).get("checks") or CHECKS)


def queue_qa(dataset_id: str, checks: Optional[Iterable[str]] = None, options: Optional[dict] = None,
             message: str = "Queued") -> Tuple[Job, bool]:
    """Start a QA job, or return the active one (asking it to run once more afterwards when the request
    needs fresher data than the running job will produce, or checks the queued one will not run).
    Returns (job, already_running)."""
    checks = normalise_checks(checks)
    if options:
        save_options(dataset_id, options)        # validated; remembered for every later run (hook, auto)
    with _queue_lock:
        active = active_job(dataset_id)
        if active is not None:
            if active.status == "running" or options or not set(checks) <= _job_checks(active):
                _rerun.setdefault(dataset_id, set()).update(checks)
            return active, True

        def fn(job: Job):
            try:
                return run_qa(dataset_id, job, checks)
            finally:
                _after(dataset_id, job)

        job = jobs.submit("qa", fn, dataset_id=dataset_id, message=message)
        job.extra = {"checks": checks, **(job.extra or {})}
        return job, False


def _after(dataset_id: str, job: Job) -> None:
    """Re-run what was asked for while `job` ran. Waits until the job has really finished: queue_qa would
    otherwise still see it running and park the request again, where nobody would pick it up."""
    with _queue_lock:
        again = _rerun.pop(dataset_id, None)
    if not again:
        return

    def later():
        deadline = time.monotonic() + 30
        while job.status in ("queued", "running") and time.monotonic() < deadline:
            time.sleep(0.02)
        if db.query_one("SELECT 1 FROM datasets WHERE id=?", (dataset_id,)) and not jobs.shutting_down.is_set():
            queue_qa(dataset_id, sorted(again), message="Queued")

    threading.Thread(target=later, name="fovea-qa-rerun", daemon=True).start()


def active_job(dataset_id: str) -> Optional[Job]:
    """The QA job to show for a dataset: the running one, else the next queued one, else None."""
    act = jobs.active_all(dataset_id, ["qa"])
    if not act:
        return None
    running = [j for j in act if j.status == "running"]
    return running[0] if running else act[-1]


def on_scan(dataset_id: str) -> None:
    """POST_SCAN_HOOKS entry: incremental QA after every successful (re)scan or relocation."""
    queue_qa(dataset_id, None, message="Queued after indexing")


# ------------------------------------------------------------------ label edits: incremental re-check
# A label save changes what the label-based checks would say about that image, its sequence and its
# near-identical twins. LABEL_WRITE_HOOKS (core.labels) call on_label_write() after every write path; the
# edits are coalesced per dataset and, once the annotator pauses, re-checked in a background thread for the
# affected sequences / twins / boxes only (milliseconds), so flags never go stale and no job banner flickers
# on every save. Edits too large for that (a bulk relabel) queue one ordinary label-check job instead.

LABEL_DEBOUNCE = 3.0      # seconds without a new save before re-checking a dataset ...
LABEL_MAX_WAIT = 15.0     # ... but never later than this after the first pending save (steady annotating)
INCR_MAX_IMAGES = 3000    # more edited images than this (or a quarter of the dataset): one full label-check job
INCR_MAX_FRAMES = 30000   # ... likewise when their sequences + near-duplicate neighbourhood exceed this
INCR_MAX_IMAGES_PY = 200  # without numpy the near-duplicate neighbourhood search is pure Python
SHAPE_FULL_MAX_BOXES = 600_000   # up to this many boxes a re-check redoes the exact shape pass (~0.5 s in numpy)
RECHECK_PROCESS_FRAMES = 2000    # sequence frames + twins from which a re-check computes in a worker process
SEQ_CODES = CHECKS["sequence"]


class _LabelRefresher:
    def __init__(self):
        self.cv = threading.Condition()
        self.pending: Dict[str, Set[int]] = {}
        self.first: Dict[str, float] = {}
        self.last: Dict[str, float] = {}
        self.hold: Dict[str, float] = {}
        self.running: Dict[str, Set[int]] = {}
        # image id -> label mtime its flags were last re-checked at (since the last full label check)
        self.covered: Dict[str, Dict[int, Optional[float]]] = defaultdict(dict)
        self.last_run: Dict[str, dict] = {}
        self.thread: Optional[threading.Thread] = None

    def add(self, dataset_id: str, image_ids: Iterable[int]) -> None:
        ids = set()
        for i in image_ids or ():
            try:
                ids.add(int(i))
            except (TypeError, ValueError):
                continue
        if not ids or not dataset_id:
            return
        now = time.monotonic()
        with self.cv:
            p = self.pending.setdefault(dataset_id, set())
            if not p:
                self.first[dataset_id] = now
            p.update(ids)
            self.last[dataset_id] = now
            if self.thread is None or not self.thread.is_alive():
                self.thread = threading.Thread(target=self._loop, name="fovea-qa-labels", daemon=True)
                self.thread.start()
            self.cv.notify_all()

    def _due(self, ds: str) -> float:
        return max(min(self.first[ds] + LABEL_MAX_WAIT, self.last[ds] + LABEL_DEBOUNCE), self.hold.get(ds, 0.0))

    def busy(self, ds: str) -> Set[int]:
        with self.cv:
            return set(self.pending.get(ds, ())) | set(self.running.get(ds, ()))

    def _take(self, force: Optional[str] = None) -> Optional[Tuple[str, Set[int]]]:
        """Under self.cv: the next dataset whose edits are due (or `force`), moved to running."""
        cands = [ds for ds, p in self.pending.items() if p and ds not in self.running]
        if force is not None:
            cands = [ds for ds in cands if ds == force]
        if not cands:
            return None
        ds = min(cands, key=self._due)
        if force is None and self._due(ds) > time.monotonic():
            return None
        ids = self.pending.pop(ds)
        for d in (self.first, self.last, self.hold):
            d.pop(ds, None)
        self.running[ds] = ids
        return ds, ids

    def _loop(self) -> None:
        while not jobs.shutting_down.is_set():
            with self.cv:
                nxt = self._take()
                if nxt is None:
                    dues = [self._due(ds) for ds, p in self.pending.items() if p and ds not in self.running]
                    self.cv.wait(max(0.05, min(dues) - time.monotonic()) if dues else 30.0)
                    continue
            ds, ids = nxt
            if jobs.active_for(ds, "qa") is not None or jobs.active_for(ds, "scan") is not None:
                # a full run (or a rescan, which queues one) is under way: re-check after it, from its results
                with self.cv:
                    self.running.pop(ds, None)
                    p = self.pending.setdefault(ds, set())
                    p.update(ids)
                    t = time.monotonic()
                    self.first.setdefault(ds, t)
                    self.last.setdefault(ds, t)
                    self.hold[ds] = t + 1.0
                continue
            self._process(ds, ids)

    def _process(self, ds: str, ids: Set[int]) -> Optional[dict]:
        res: Optional[dict] = None
        try:
            res = recheck_labels(ds, ids)
            if res.get("full"):
                queue_qa(ds, list(LABEL_CHECKS), message="Rechecking edited labels")
            with self.cv:
                self.covered[ds].update(res.get("covered") or {})
                self.last_run[ds] = {**{k: v for k, v in res.items() if k != "covered"}, "at": db.now()}
        except JobCancelled:
            pass                             # the server is stopping
        except Exception:  # noqa: BLE001 — a hook must never break a label save; the next full run catches up
            log.exception("Re-checking edited labels of dataset %s failed", ds)
        finally:
            with self.cv:
                self.running.pop(ds, None)
                idle = not self.pending.get(ds)
                self.cv.notify_all()
        if res is not None and idle and not res.get("full"):
            try:
                _advance_label_checks(ds)
            except Exception:  # noqa: BLE001
                log.exception("Could not record label re-check freshness for %s", ds)
        return res

    def flush(self, dataset_id: Optional[str] = None, timeout: float = 30.0) -> bool:
        """Run every pending re-check now, in this thread (tests, shutdown). True when nothing is left."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.cv:
                target = dataset_id or next((d for d, p in self.pending.items() if p and d not in self.running), None)
                busy = bool(self.running.get(dataset_id)) if dataset_id else bool(self.running)
                waiting = bool(target and self.pending.get(target) and target not in self.running)
            if waiting and (jobs.active_for(target, "qa") is not None or jobs.active_for(target, "scan") is not None):
                time.sleep(0.05)             # as in _loop: re-check after the full run, not under it
                continue
            with self.cv:
                nxt = self._take(force=target) if waiting else None
            if nxt is None:
                if not busy and not waiting:
                    return True
                time.sleep(0.02)
                continue
            self._process(*nxt)
        return False


_refresher = _LabelRefresher()


def on_label_write(dataset_id: str, image_ids: Iterable[int]) -> None:
    """LABEL_WRITE_HOOKS entry: remember the edited images; the re-check runs once the edits pause."""
    _refresher.add(dataset_id, image_ids)


def flush_label_rechecks(dataset_id: Optional[str] = None, timeout: float = 30.0) -> bool:
    return _refresher.flush(dataset_id, timeout)


def refreshing_ids(dataset_id: str) -> Set[int]:
    """Images whose saved labels are queued for (or in) an incremental re-check."""
    return _refresher.busy(dataset_id)


def last_recheck(dataset_id: str) -> Optional[dict]:
    """The latest incremental re-check of a dataset in this process: {at, images, seconds, counts, ...}."""
    with _refresher.cv:
        r = _refresher.last_run.get(dataset_id)
        return dict(r) if r else None


def label_freshness(dataset_id: str, since: Optional[float]) -> Tuple[int, int]:
    """(stale, refreshing): images whose labels changed after the last full label check (`since`) and whose
    flags do not reflect that yet, and images queued for (or in) an incremental re-check right now."""
    busy = _refresher.busy(dataset_id)
    if not since:
        return 0, len(busy)
    with _refresher.cv:
        covered = dict(_refresher.covered.get(dataset_id, {}))
    rows = db.query("SELECT id, lmtime FROM images WHERE dataset_id=? AND lmtime > ? ORDER BY id", (dataset_id, since))
    stale = sum(1 for r in rows if r["id"] not in busy and covered.get(r["id"], -1.0) != r["lmtime"])
    return stale, len(busy)


def _advance_label_checks(dataset_id: str) -> None:
    """When every label edited since the last full label check has been re-checked at its current version,
    move the label checks' timestamp forward, so the summary (and a restart) see them as fresh."""
    with _summary_lock:
        s = load_summary(dataset_id) or {}
        at = s.get("checks_at") or {}
        if not all(c in at for c in LABEL_CHECKS):
            return
        since = min(at[c] for c in LABEL_CHECKS)
        rows = db.query("SELECT id, lmtime FROM images WHERE dataset_id=? AND lmtime > ? ORDER BY id", (dataset_id, since))
        with _refresher.cv:
            if _refresher.pending.get(dataset_id) or _refresher.running.get(dataset_id):
                return
            covered = _refresher.covered.get(dataset_id, {})
            if any(covered.get(r["id"], -1.0) != r["lmtime"] for r in rows):
                return
            if rows:
                newest = max(r["lmtime"] for r in rows)
                s["checks_at"] = {**at, **{c: max(at[c], newest) for c in LABEL_CHECKS}}
                _save_summary(dataset_id, s)
            _refresher.covered.pop(dataset_id, None)


def _near_columns(dataset_id: str, hashes: Iterable[int], max_dist: int) -> Tuple[List[int], List[int], List[int]]:
    """(ids, fine hashes, columns): the dataset's hashed images and the positions of those within max_dist of
    ANY of `hashes`. Queries are deduplicated (a static camera's 8,000 identical frames are one query) and only
    the union is kept — one 64 x N distance block and one flag per image at a time, never a list per query
    (8,000 twins x 8,000 neighbours once took 4 GB)."""
    ids, fines, words = _hash_matrix(dataset_id)
    uniq = list(dict.fromkeys(hashes))
    if not ids or not uniq:
        return ids, fines, []
    np = _numpy()
    if np is not None and words is not None:
        hit = np.zeros(len(ids), dtype=bool)
        mask = (1 << 64) - 1
        for s in range(0, len(uniq), 64):
            block = uniq[s:s + 64]
            qw = np.array([[(hv >> sh) & mask for sh in (192, 128, 64, 0)] for hv in block], dtype=np.uint64)
            d = np.zeros((len(block), len(ids)), dtype=np.int32)
            for w in range(4):
                d += _popcount(np, qw[:, w:w + 1] ^ words[None, :, w]).astype(np.int32)
            hit |= (d <= max_dist).any(axis=0)
        return ids, fines, np.flatnonzero(hit).tolist()
    return ids, fines, [k for k, f in enumerate(fines) if any((f ^ hv).bit_count() <= max_dist for hv in uniq)]


def neighbour_ids(dataset_id: str, hashes: Iterable[int], max_dist: int) -> Set[int]:
    """Every hashed image of the dataset within max_dist (256-bit) of any of `hashes` (images with exactly
    those hashes included)."""
    ids, _fines, cols = _near_columns(dataset_id, hashes, max_dist)
    return {ids[c] for c in cols}


def recheck_labels(dataset_id: str, image_ids: Iterable[int]) -> dict:
    """Re-run the label-based checks for images whose labels changed, touching only what those edits can
    change: seq_* on the frames of their sequences, dup_label_mismatch on their near-identical neighbourhood,
    shape_outlier on their own boxes (scored
    against the class statistics of the last full run, or a full shape pass when the classes drifted).
    Returns counts, `covered` {image id: label mtime re-checked}, or {'full': True} when one ordinary
    label-check job would be cheaper."""
    t0 = time.time()
    s = load_summary(dataset_id) or {}
    done = s.get("checks_at") or {}
    if not s.get("computed_at") or not any(c in done for c in LABEL_CHECKS):
        return {"skipped": "not checked yet", "images": 0}
    ids = sorted({int(i) for i in image_ids})
    rows: List[dict] = []
    for k in range(0, len(ids), 5000):
        rows += db.query("SELECT id, split, seq FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?))",
                         (dataset_id, db.dumps(ids[k:k + 5000])))
    if not rows:
        return {"images": 0}
    n_all = (db.query_one("SELECT COUNT(*) c FROM images WHERE dataset_id=?", (dataset_id,)) or {"c": 0})["c"]
    np = _numpy()
    cap = INCR_MAX_IMAGES if np is not None else INCR_MAX_IMAGES_PY
    if len(rows) > cap or (n_all >= 400 and len(rows) > n_all / 4):
        return {"full": True, "images": len(rows)}
    opt = options_for(dataset_id)
    edited = {r["id"] for r in rows}

    # sequences: every frame of each affected (split, seq)
    seq_scope: Set[int] = set()
    if "sequence" in done:
        groups = {(r["split"] or "", r["seq"]) for r in rows if r["seq"]}
        seqs = sorted({g[1] for g in groups})
        for k in range(0, len(seqs), 2000):
            for m in db.query("SELECT id, split, seq FROM images WHERE dataset_id=? AND seq IN (SELECT value FROM json_each(?))",
                              (dataset_id, db.dumps(seqs[k:k + 2000]))):
                if ((m["split"] or ""), m["seq"]) in groups:
                    seq_scope.add(m["id"])
    if len(seq_scope) > INCR_MAX_FRAMES:
        return {"full": True, "images": len(rows)}

    # near-identical neighbourhood: R = edited + their twins (flags may change), S = R + twins of R (context)
    mm_scope: Set[int] = set()
    mm_context: Set[int] = set()
    if "duplicates" in done:
        mis_d = int(opt["dup_dist"])
        hashes = {r["image_id"]: fine_hash(r) for r in db.query(
            "SELECT image_id, b0, b1, b2, b3 FROM image_hash WHERE dataset_id=? AND b0 IS NOT NULL "
            "AND image_id IN (SELECT value FROM json_each(?))", (dataset_id, db.dumps(sorted(edited))))}
        if hashes:
            ids_all, fines_all, cols = _near_columns(dataset_id, hashes.values(), mis_d)
            mm_scope = set(hashes) | {ids_all[c] for c in cols}
            if len(seq_scope) + len(mm_scope) > INCR_MAX_FRAMES:      # before anything grows with it
                return {"full": True, "images": len(rows)}
            mm_context = set(mm_scope)
            # the ring's own twins: searched once per distinct hash not searched above
            ring_h = {fines_all[c] for c in cols if ids_all[c] not in hashes} - set(hashes.values())
            if ring_h:
                if np is None and len(ring_h) > INCR_MAX_IMAGES_PY:   # pure Python: rings x images comparisons
                    return {"full": True, "images": len(rows)}
                mm_context |= neighbour_ids(dataset_id, ring_h, mis_d)
    if len(seq_scope) + len(mm_context) > INCR_MAX_FRAMES:
        return {"full": True, "images": len(rows)}

    by_id = {it.id: it for it in load_subset(dataset_id, edited)}
    # sequences and twins: pure Python, in a worker process when the scope is large (a static camera's
    # 8,000-frame sequence) so that the re-check never holds the GIL against the request threads
    seq_mm = _RowSink()
    in_worker = False
    if seq_scope or mm_scope:
        in_worker = run_task("recheck", {"dataset_id": dataset_id, "opt": opt, "seq_scope": sorted(seq_scope),
                                         "mm_scope": sorted(mm_scope), "mm_context": sorted(mm_context)},
                             seq_mm, _check_shutdown,
                             in_worker=n_all >= PROCESS_MIN_IMAGES and len(seq_scope) + len(mm_context) >= RECHECK_PROCESS_FRAMES)
    timings: Dict[str, Any] = {"sequence": 0.0, "duplicates": 0.0, **(seq_mm.result.get("timings") or {})}
    flags: Dict[Tuple[int, str], Tuple[float, dict]] = {}

    t = time.time()
    shape_scope: Optional[Set[int]] = set()        # None: every image (an exact pass over all boxes)
    shape_model: Optional[dict] = None
    if "shape" in done:
        model = s.get("shape_model")
        n_boxes = int((db.query_one("SELECT box_count FROM datasets WHERE id=?", (dataset_id,)) or {}).get("box_count") or 0)
        exact_up_to = SHAPE_FULL_MAX_BOXES if np is not None else SHAPE_FULL_MAX_BOXES // 6
        if n_boxes <= exact_up_to or model is None or _shape_drifted(dataset_id, model, opt):
            # class statistics move with every edit (a bulk class change moved 11 borderline boxes of
            # VFP290K across z = 5): re-run the whole pass, which is cheap in numpy, instead of re-scoring
            info: Dict[str, Any] = {}
            sf: Dict[Tuple[int, str], Tuple[float, dict]] = {}
            if np is not None:
                A, cached = _shape_arrays_after_edit(np, dataset_id, edited, by_id)
                _shape_from_arrays(np, A, opt, sf, info)
                _shape_cache_put(dataset_id, A)
                timings["shape_cache"] = cached
            else:
                shape_pass(shape_rows(dataset_id), opt, sf, info)
            flags.update(sf)
            shape_model = info["shape_model"]
            shape_scope = None
        else:
            zt = float(opt["shape_z"])
            sf = {}
            for i in edited:
                it = by_id.get(i)
                if it is None:
                    continue
                for r, b in enumerate(it.boxes):
                    m = model.get(str(b[0]))
                    shp = _box_shape(b, it.W, it.H) if m is not None else None
                    if shp is not None:
                        hits = _shape_hits(r, b[0], shp[0], shp[1], m, zt)
                        if hits:
                            _add_shape_hits(sf, it.id, hits)
            _trim_shape_rows(sf)
            flags.update(sf)
            shape_scope = set(edited)
    timings["shape"] = round(time.time() - t, 3)

    scopes: Dict[str, Optional[Set[int]]] = {code: seq_scope for code in SEQ_CODES}
    scopes["dup_label_mismatch"] = mm_scope
    scopes["shape_outlier"] = shape_scope

    def in_scope(iid: int, code: str) -> bool:
        sc = scopes.get(code, set())
        return sc is None or iid in sc

    rows_out = [(iid, dataset_id, code, score, detail, iid) for iid, code, score, detail in seq_mm.rows if in_scope(iid, code)]
    rows_out += [(iid, dataset_id, code, float(score), db.dumps(detail), iid)
                 for (iid, code), (score, detail) in flags.items() if in_scope(iid, code)]
    with db.transaction() as conn:
        for code, scope in scopes.items():
            if scope is None:
                conn.execute("DELETE FROM image_flags WHERE dataset_id=? AND code=?", (dataset_id, code))
                continue
            if not scope:
                continue
            ids_s = sorted(scope)
            for k in range(0, len(ids_s), 20000):
                conn.execute("DELETE FROM image_flags WHERE dataset_id=? AND code=? AND image_id IN (SELECT value FROM json_each(?))",
                             (dataset_id, code, db.dumps(ids_s[k:k + 20000])))
        conn.executemany("""INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail)
                            SELECT ?,?,?,?,? WHERE EXISTS (SELECT 1 FROM images WHERE id = ?)""", rows_out)
    if shape_model is not None:
        with _summary_lock:
            cur = load_summary(dataset_id) or {}
            cur["shape_model"] = shape_model
            _save_summary(dataset_id, cur)
    counts = Counter(r[2] for r in rows_out)
    return {"images": len(rows), "sequence_frames": len(seq_scope), "twins": len(mm_scope),
            "shape": "full" if shape_scope is None else len(shape_scope), "counts": dict(counts),
            "timings": timings, "seconds": round(time.time() - t0, 3), "process": in_worker,
            "covered": {i: by_id[i].lmtime if by_id[i].lmtime is not None else -1.0 for i in edited if i in by_id}}


def _recheck_pass(dataset_id: str, opt: dict, seq_scope: List[int], mm_scope: List[int], mm_context: List[int]):
    """recheck_labels' CPU-bound part as a message stream (see _compute): seq_* over `seq_scope` (every frame
    of the edited images' sequences) and dup_label_mismatch for `mm_scope` (the edited images and their
    near-identical twins), judged among `mm_context` (the twins' own twins as well)."""
    by_id = {it.id: it for it in load_subset(dataset_id, set(seq_scope) | set(mm_context))}
    flags: Dict[Tuple[int, str], Tuple[float, dict]] = {}
    timings: Dict[str, float] = {}
    t = time.time()
    if seq_scope:
        check_sequences([by_id[i] for i in seq_scope if i in by_id], opt, flags, {})
    timings["sequence"] = round(time.time() - t, 3)
    t = time.time()
    if mm_scope:
        scope = set(mm_scope)
        nodes: Dict[int, List[Item]] = defaultdict(list)
        for i in mm_context:
            it = by_id.get(i)
            if it is not None and it.fine is not None:
                nodes[it.fine].append(it)
        keys = list(nodes)
        mm: Dict[Tuple[int, str], Tuple[float, dict]] = {}
        mismatch_flags(nodes, keys, near_pairs(keys, int(opt["dup_dist"])), opt, mm)
        flags.update({k: v for k, v in mm.items() if k[0] in scope})
    timings["duplicates"] = round(time.time() - t, 3)
    rows = [(iid, code, float(score), db.dumps(detail)) for (iid, code), (score, detail) in flags.items()]
    for k in range(0, len(rows), FLAG_CHUNK):
        yield ("flags", rows[k:k + FLAG_CHUNK])
    yield ("done", {"timings": timings})


class _RowSink:
    """Collects a task's flag rows and its final ('done', result)."""

    def __init__(self):
        self.rows: List[tuple] = []
        self.result: dict = {}

    def __call__(self, msg: tuple) -> None:
        if msg[0] == "flags":
            self.rows.extend(msg[1])
        elif msg[0] == "done":
            self.result = msg[1]


def _check_shutdown() -> None:
    if jobs.shutting_down.is_set():
        raise JobCancelled()


# what a worker process can run: name -> generator function of picklable keyword arguments
_WORKER_TASKS: Dict[str, Callable[..., Any]] = {"run": _compute, "recheck": _recheck_pass}


def _shape_drifted(dataset_id: str, model: dict, opt: dict) -> bool:
    """True when per-class box counts moved enough since the last full shape pass that its statistics no
    longer describe the classes (a class crossed shape_min, or grew/shrank by > 0.5%)."""
    n_min = int(opt["shape_min"])
    now = {int(r["cls"]): int(r["c"]) for r in db.query(
        "SELECT cls, COUNT(*) c FROM boxes WHERE dataset_id=? GROUP BY cls", (dataset_id,))}
    for cls in set(now) | {int(c) for c in model}:
        n_now = now.get(cls, 0)
        m = model.get(str(cls))
        n_then = int(m[4]) if m else 0
        if (n_now >= n_min) != (m is not None):
            return True
        if m and abs(n_now - n_then) > max(5, 0.005 * n_then):
            return True
    return False


# ------------------------------------------------------------------ per-dataset thresholds (settings)

OPTION_INFO: Dict[str, dict] = {
    "dup_dist": {"label": "Near-duplicate distance", "group": "duplicates", "unit": "bits", "step": 1,
                 "help": "How many of 256 hash bits two pictures may differ in and still count as the same picture. "
                         "0 = identical only; 2 catches re-encoded copies."},
    "leak_dist": {"label": "Leak distance", "group": "duplicates", "unit": "bits", "step": 1,
                  "help": "Bits a picture in another split may differ by and still count as a leaked copy."},
    "scene_dist": {"label": "Same camera view", "group": "duplicates", "unit": "bits", "step": 1, "advanced": True,
                   "help": "64-bit distance at which frames of one sequence in two splits count as the same view."},
    "mismatch_iou": {"label": "Label agreement IoU", "group": "duplicates", "step": 0.05, "advanced": True,
                     "help": "Boxes of two near-identical pictures agree when they overlap at least this much."},
    "seq_cut": {"label": "Shot cut", "group": "sequence", "unit": "bits", "step": 1, "advanced": True,
                "help": "64-bit distance between consecutive frames that starts a new shot."},
    "seq_video": {"label": "Video likeness", "group": "sequence", "unit": "bits", "step": 1, "advanced": True,
                  "help": "Median distance between consecutive frames above which a folder is not treated as video."},
    "shape_z": {"label": "Unusual box shape", "group": "shape", "unit": "z", "step": 0.5,
                "help": "Robust z-score of a box's aspect or area within its class. Lower flags more boxes."},
    "shape_min": {"label": "Boxes per class", "group": "shape", "step": 1, "advanced": True,
                  "help": "A class needs this many boxes before its shapes are judged."},
    "blur_z": {"label": "Blur sensitivity", "group": "quality", "unit": "z", "step": 0.1,
               "help": "How far an image's blur must stand out from images of similar brightness. Lower flags more."},
    "blur_floor": {"label": "Minimum blur", "group": "quality", "step": 0.01,
                   "help": "Blur effect (0 sharp – 1 flat) an image must reach before it can be called blurry."},
    "dark_floor": {"label": "Dark below brightness", "group": "quality", "unit": "0–255", "step": 1,
                   "help": "Mean brightness under which an image can be dark."},
    "dark_z": {"label": "Dark: distance from typical", "group": "quality", "unit": "z", "step": 0.1, "advanced": True,
               "help": "…and how far below this dataset's typical brightness it must be (keeps a dark CCTV set from being flagged wholesale)."},
    "bright_floor": {"label": "Overexposed above brightness", "group": "quality", "unit": "0–255", "step": 1,
                     "help": "Mean brightness over which an image can be overexposed."},
    "bright_z": {"label": "Overexposed: distance from typical", "group": "quality", "unit": "z", "step": 0.1, "advanced": True,
                 "help": "…and how far above this dataset's typical brightness it must be."},
}
OPTION_CHECK = {k: v["group"] for k, v in OPTION_INFO.items()}


def settings(dataset_id: str) -> dict:
    """The threshold contract for a settings UI: effective values, defaults, overrides, limits and copy."""
    saved = (load_summary(dataset_id) or {}).get("options") or {}
    opt = options_for(dataset_id)
    return {
        "options": opt,
        "defaults": dict(DEFAULTS),
        "overrides": {k: v for k, v in saved.items() if k in DEFAULTS},
        "fields": [{"key": k, "label": OPTION_INFO[k]["label"], "help": OPTION_INFO[k]["help"],
                    "check": OPTION_INFO[k]["group"], "unit": OPTION_INFO[k].get("unit"), "step": OPTION_INFO[k]["step"],
                    "min": OPTION_LIMITS[k][0], "max": OPTION_LIMITS[k][1], "default": DEFAULTS[k], "value": opt[k],
                    "integer": isinstance(DEFAULTS[k], int), "advanced": bool(OPTION_INFO[k].get("advanced")),
                    "codes": list(CHECKS[OPTION_INFO[k]["group"]])} for k in DEFAULTS],
    }


def update_settings(dataset_id: str, options: Optional[dict] = None, reset: Any = None, rerun: bool = True) -> dict:
    """Save threshold overrides (`reset`: True or a list of keys back to default) and, when anything changed,
    re-run the checks those thresholds drive. Returns settings() + {changed, job}."""
    before = options_for(dataset_id)
    keys = list(DEFAULTS) if reset is True else list(reset or [])
    after = save_options(dataset_id, options or {}, keys)
    changed = sorted(k for k in DEFAULTS if before[k] != after[k])
    job = None
    if changed and rerun:
        checks = sorted({OPTION_CHECK[k] for k in changed})
        job, _ = queue_qa(dataset_id, checks, message="Rechecking with new thresholds")
    return {**settings(dataset_id), "changed": changed, "job": job.to_dict() if job is not None else None}


def stale_images(dataset_id: str) -> int:
    # ORDER BY i.id steers the planner to (dataset_id, id): with ANALYZE statistics it otherwise walks the
    # random-order expression index and visits rows in random rowid order (363 ms instead of 11 at 47k images)
    row = db.query_one("""SELECT COUNT(*) c FROM (SELECT i.id FROM images i LEFT JOIN image_hash h ON h.image_id = i.id
                          WHERE i.dataset_id = ? AND (h.image_id IS NULL OR h.mtime IS NOT i.mtime) ORDER BY i.id)""",
                       (dataset_id,))
    return int(row["c"]) if row else 0


def flag_counts(dataset_id: str) -> Dict[str, int]:
    return {r["code"]: r["c"] for r in db.query(
        "SELECT code, COUNT(*) c FROM image_flags WHERE dataset_id=? GROUP BY code ORDER BY code", (dataset_id,))}


# ------------------------------------------------------------------ duplicates listing / similar search

_groups_cache: Dict[str, Tuple[float, list]] = {}


def _groups_index(dataset_id: str) -> list:
    row = db.query_one("SELECT updated_at FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?",
                       (PLUGIN_KEY[0], dataset_id, GROUPS_KEY))
    if not row:
        return []
    hit = _groups_cache.get(dataset_id)
    if hit and hit[0] == row["updated_at"]:
        return hit[1]
    full = db.query_one("SELECT value, updated_at FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?",
                        (PLUGIN_KEY[0], dataset_id, GROUPS_KEY))
    index = db.loads(full["value"], [])
    _groups_cache[dataset_id] = (full["updated_at"], index)
    return index


def duplicate_groups(dataset_id: str, offset: int = 0, limit: int = 20, cross_split: bool = False,
                     member_limit: int = 60) -> dict:
    """Near-duplicate groups of the last QA run, biggest first, paged; member rows are read fresh (labels,
    reviews and splits may have changed since). `cross_split` keeps the groups whose members feed more than
    one split role (a leak); each group names the member Export keeps (`keep`, decided now over ALL members,
    also those beyond `member_limit`)."""
    index = _groups_index(dataset_id)
    if cross_split:
        index = [g for g in index if len({split_role(x) for x in g[2]}) > 1]
    page = index[offset:offset + limit]
    want = [iid for g in page for iid in g[3][:member_limit]]
    rows: Dict[int, dict] = {}
    for s in range(0, len(want), 2000):
        for m in db.query("""SELECT i.id, i.rel_path, i.split, i.n_boxes, i.classes, i.has_label, i.mtime, f.detail
                             FROM images i LEFT JOIN image_flags f ON f.image_id = i.id AND f.code = 'near_dup'
                             WHERE i.id IN (SELECT value FROM json_each(?)) AND i.dataset_id = ?""",
                          (db.dumps(want[s:s + 2000]), dataset_id)):
            rows[m["id"]] = m
    keep_rows: Dict[int, _KeepRow] = {}
    every = [iid for g in page for iid in g[3]]
    for s in range(0, len(every), 5000):
        for m in db.query("""SELECT i.id, i.has_label, i.n_boxes, r.status FROM images i
                             LEFT JOIN reviews r ON r.dataset_id = i.dataset_id AND r.rel_path = i.rel_path
                             WHERE i.id IN (SELECT value FROM json_each(?)) AND i.dataset_id = ?""",
                          (db.dumps(every[s:s + 5000]), dataset_id)):
            keep_rows[m["id"]] = _KeepRow(m["id"], m["status"], m["has_label"], m["n_boxes"])
    groups = []
    for gid, size, splits, ids in page:
        keep = keep_member(keep_rows[i] for i in ids if i in keep_rows)
        members = []
        for iid in ids[:member_limit]:
            m = rows.get(iid)
            if m is None:
                continue                      # removed by a rescan since the last QA run
            d = db.loads(m["detail"], {}) if m["detail"] else {}
            members.append({
                "image_id": iid, "rel_path": m["rel_path"], "split": m["split"], "dist": d.get("dist", 0),
                "n_boxes": m["n_boxes"], "classes": db.loads(m["classes"]), "has_label": bool(m["has_label"]),
                "mtime": m["mtime"], "leader": bool(d.get("leader")), "keep": iid == keep,
                "exact": bool(d.get("exact"))})
        roles = sorted({split_role(x) for x in splits})
        groups.append({"group": gid, "size": size, "splits": splits, "roles": roles, "cross_split": len(roles) > 1,
                       "keep": keep, "members": members})
    return {"total": len(index), "offset": offset, "limit": limit, "groups": groups}


_sim_cache: Dict[str, Tuple[int, float, Any]] = {}
_sim_lock = threading.Lock()


def _hash_matrix(dataset_id: str):
    """(ids, fine hashes [, numpy words]) for a dataset, cached until its hashes change."""
    gen = _generation[dataset_id]
    with _sim_lock:
        hit = _sim_cache.get(dataset_id)
        if hit and hit[0] == gen and time.time() - hit[1] < 300:
            return hit[2]
    rows = db.query("SELECT image_id, b0, b1, b2, b3 FROM image_hash WHERE dataset_id=? AND b0 IS NOT NULL ORDER BY image_id",
                    (dataset_id,))
    ids = [r["image_id"] for r in rows]
    fines = [fine_hash(r) for r in rows]
    np = _numpy()
    words = None
    if np is not None and rows:
        words = np.array([[to_unsigned(r[k]) for k in ("b0", "b1", "b2", "b3")] for r in rows], dtype=np.uint64)
    data = (ids, fines, words)
    with _sim_lock:
        _sim_cache[dataset_id] = (gen, time.time(), data)
    return data


def similar(dataset_id: str, image_id: int, k: int = 24, max_dist: int = 64) -> dict:
    img = db.query_one("""SELECT i.id, i.abs_path, i.mtime, h.b0, h.b1, h.b2, h.b3, h.mtime hm
                          FROM images i LEFT JOIN image_hash h ON h.image_id = i.id WHERE i.id=? AND i.dataset_id=?""",
                       (image_id, dataset_id))
    if not img:
        return {"items": [], "missing": True}
    q = fine_hash(img) if img["hm"] == img["mtime"] else None
    if q is None:
        f = image_features(img["abs_path"])
        if f is None:
            return {"items": [], "unreadable": True}
        with db.transaction() as conn:
            conn.execute(_HASH_SQL, _hash_row(image_id, dataset_id, img["mtime"], f))
        _generation[dataset_id] += 1
        q = f["fine"]
    ids, fines, words = _hash_matrix(dataset_id)
    np = _numpy()
    if words is not None and len(ids) > 2000:
        qw = np.array([(q >> s) & ((1 << 64) - 1) for s in (192, 128, 64, 0)], dtype=np.uint64)
        d = _popcount(np, words ^ qw[None, :]).sum(axis=1)
        cand = np.nonzero(d <= max_dist)[0]
        scored = [(int(d[c]), ids[c]) for c in cand.tolist() if ids[c] != image_id]
    else:
        scored = [(hamming(q, h), i) for i, h in zip(ids, fines) if i != image_id]
        scored = [s for s in scored if s[0] <= max_dist]
    scored.sort()
    top = scored[:k]
    if not top:
        return {"items": [], "hashed": len(ids)}
    rows = {r["id"]: r for r in db.query(
        "SELECT id, rel_path, split, mtime, seq, n_boxes, classes, width, height FROM images WHERE id IN (SELECT value FROM json_each(?))",
        (db.dumps([i for _d, i in top]),))}
    items = []
    for d, i in top:
        r = rows.get(i)
        if r:
            items.append({"id": i, "rel_path": r["rel_path"], "split": r["split"], "dist": d, "mtime": r["mtime"],
                          "seq": r["seq"], "n_boxes": r["n_boxes"], "classes": db.loads(r["classes"]),
                          "width": r["width"], "height": r["height"]})
    return {"items": items, "hashed": len(ids)}


# ------------------------------------------------------------------ interpolation / gap filling

class QAError(ValueError):
    """A user-facing refusal (bad request), with a message fit for the UI."""


def _frames_of(dataset_id: str, split: str, seq: str) -> List[dict]:
    rows = db.query("SELECT id, rel_path, split, seq, width, height, has_label, n_boxes, label_path FROM images "
                    "WHERE dataset_id=? AND split=? AND seq=?", (dataset_id, split, seq))
    rows.sort(key=lambda r: natural_key(r["rel_path"]))
    return rows


def _db_boxes(image_ids: List[int]) -> Dict[int, List[list]]:
    out: Dict[int, List[list]] = {i: [] for i in image_ids}
    for s in range(0, len(image_ids), 500):
        chunk = image_ids[s:s + 500]
        for r in db.query("SELECT image_id, cls, xc, yc, w, h FROM boxes WHERE image_id IN (SELECT value FROM json_each(?)) ORDER BY image_id, id",
                          (db.dumps(chunk),)):
            out[r["image_id"]].append([int(r["cls"]), r["xc"], r["yc"], r["w"], r["h"]])
    return out


def _file_boxes(row: dict) -> List[list]:
    """Current boxes from the label file itself (the editor may have saved a moment ago)."""
    lp = row.get("label_path")
    if lp and os.path.isfile(lp):
        boxes, _c, _i = read_label_file(lp)
        return [[int(b[0]), b[1], b[2], b[3], b[4]] for b in boxes]
    return _db_boxes([row["id"]])[row["id"]]


# A box that changes class costs as much as one that no longer overlaps at all, and may only do so in place:
# overlapping, or centres within 0.75 of the larger diagonal, with areas within 4x (standing 50x150 px -> lying
# 150x60 px is 1.2x). On the 8,412 labeled triples of consecutive frames in vfp290k (one frame interpolated from
# its neighbours) this raised recall from 0.729 to 0.753 at IoU 0.3, precision 0.840 -> 0.838; 77% of the
# class-change boxes land on a real box. Looser gates (1.25, 1.5) added no recall and cost precision.
KEYFRAME_CHANGE_COST = 1.0
KEYFRAME_CHANGE_DIST = 0.75
KEYFRAME_CHANGE_AREA = 4.0


def match_keyframes(A: List[list], B: List[list], W: float, H: float) -> List[Tuple[int, int]]:
    """Pair the boxes of two keyframes one-to-one (Hungarian on 1 - IoU + centre distance / larger diagonal).
    Boxes of the same class pair wherever they went: the user chose the keyframes, so there is no distance
    gate. A box may also pair with a box of another class, because in a fall dataset the class is the state
    (the person who falls between the keyframes is 'standing' in one and 'fallen' in the other), but only
    in place (see KEYFRAME_CHANGE_*) and at the cost of one non-overlapping box: a same-class partner that
    is about as near still wins, and nothing pairs across the frame."""
    if not A or not B:
        return []
    big = 1e6
    cost = []
    for a in A:
        row = []
        da, area_a = _diag(a, W, H), a[3] * a[4]
        for b in B:
            v = iou(a, b)
            cd = _centre_dist(a, b, W, H) / max(da, _diag(b, W, H), 1e-6)
            c = (1 - v) + cd
            if a[0] != b[0]:
                ratio = (b[3] * b[4]) / area_a if area_a > 0 else big
                in_place = (v > 0 or cd <= KEYFRAME_CHANGE_DIST) and 1 / KEYFRAME_CHANGE_AREA <= ratio <= KEYFRAME_CHANGE_AREA
                c = c + KEYFRAME_CHANGE_COST if in_place else big
            row.append(c)
        cost.append(row)
    return sorted((i, j) for i, j in assign(cost) if cost[i][j] < big)


def _class_label(names: Sequence[Any], c: int) -> str:
    name = names[c] if 0 <= c < len(names) else None
    return str(name) if name not in (None, "") and str(name) != str(c) else f"class {c}"


def _store_interp(dataset_id: str, model: str, params: dict, preds: List[Tuple[int, list, str]],
                  image_ids: List[int]) -> Tuple[int, List[dict]]:
    now = db.now()
    with db.transaction() as conn:
        if image_ids:
            conn.execute("DELETE FROM ai_preds WHERE dataset_id=? AND state='pending' AND kind='interp' "
                         "AND image_id IN (SELECT value FROM json_each(?))", (dataset_id, db.dumps(image_ids)))
        cur = conn.execute(
            "INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at, finished_at) "
            "VALUES(?,?,?,?,?,?,?,?,?,?,?)",
            (dataset_id, "interp", "fovea", model, db.dumps(params), "done", len({p[0] for p in preds}), len(preds),
             db.dumps({"preds": len(preds)}), now, now))
        run_id = cur.lastrowid
        conn.executemany(
            "INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, label, kind, reason, state, created_at) "
            "VALUES(?,?,?,?,?,?,?,?,NULL,'','interp',?,'pending',?)",
            [(run_id, dataset_id, iid, b[0], b[1], b[2], b[3], b[4], reason, now) for iid, b, reason in preds])
    out = [{"id": r["id"], "image_id": r["image_id"], "cls": r["cls"], "xc": r["xc"], "yc": r["yc"], "w": r["w"],
            "h": r["h"], "conf": r["conf"], "label": r["label"], "kind": r["kind"], "reason": r["reason"],
            "state": r["state"], "run_id": run_id}
           for r in db.query("SELECT * FROM ai_preds WHERE run_id=? ORDER BY image_id, id", (run_id,))]
    return run_id, out


# An interpolated box is already labeled when an existing box of its class(es) matches it at this IoU, one
# existing box per interpolated box. On vfp290k (one box deleted from a fully labeled frame between two
# keyframes 2-4 labeled frames apart, 37,357 frames) the old many-to-one test at 0.7 re-proposed 71.7% of the
# deleted boxes with 2.14 duplicates of existing boxes per frame; one-to-one at 0.5: 71.3% with 1.44.
EXISTING_IOU = 0.5


def already_labeled(boxes: List[list], existing: List[list], classes: Optional[List[Set[int]]] = None,
                    thr: float = EXISTING_IOU) -> Set[int]:
    """Indices of `boxes` the frame already has a box for: matched one-to-one (Hungarian on 1 - IoU) to an
    existing box of the same class (or of any of `classes[i]`) at IoU >= thr. One existing box never stands
    for two objects, however close they are."""
    if not boxes or not existing:
        return set()
    big = 1e6
    cost = []
    for q, b in enumerate(boxes):
        cs = classes[q] if classes else {b[0]}
        row = []
        for e in existing:
            v = iou(e, b) if e[0] in cs else 0.0
            row.append(1 - v if v >= thr else big)
        cost.append(row)
    return {i for i, j in assign(cost) if cost[i][j] < big}


def interpolate(dataset_id: str, from_id: int, to_id: int) -> dict:
    if from_id == to_id:
        raise QAError("Pick two different keyframes")
    rows = {r["id"]: r for r in db.query(
        "SELECT id, rel_path, split, seq, width, height, label_path FROM images WHERE dataset_id=? AND id IN (?,?)",
        (dataset_id, from_id, to_id))}
    if from_id not in rows or to_id not in rows:
        raise QAError("Keyframe image not found in this dataset")
    a, b = rows[from_id], rows[to_id]
    if (a["split"], a["seq"]) != (b["split"], b["seq"]) or not a["seq"]:
        raise QAError("Both keyframes must be in the same sequence (and split)")
    frames = _frames_of(dataset_id, a["split"], a["seq"])
    pos = {r["id"]: k for k, r in enumerate(frames)}
    ia, ib = pos[from_id], pos[to_id]
    if ia > ib:
        ia, ib = ib, ia
        a, b = b, a
    between = frames[ia + 1:ib]
    if len(between) > MAX_INTERP_FRAMES:
        raise QAError(f"The keyframes are {len(between):,} frames apart; interpolate at most {MAX_INTERP_FRAMES:,} at once")
    A, B = _file_boxes(a), _file_boxes(b)
    if not A or not B:
        raise QAError("Both keyframes need boxes: " + ("the first" if not A else "the second") + " has none")
    if not between:
        return {"run_id": None, "images": 0, "preds": [], "frames": 0, "message": "The keyframes are adjacent: no frames in between"}
    W, H = float(a["width"] or 1), float(a["height"] or 1)
    pairs = match_keyframes(A, B, W, H)
    if not pairs:
        raise QAError("No box in one keyframe pairs with a box in the other: boxes pair with their own class "
                      "anywhere, or with another class only where they overlap or nearly do")
    fa, fb = frame_number(a["rel_path"]), frame_number(b["rel_path"])
    nums = [frame_number(r["rel_path"]) for r in between]
    use_nums = fa is not None and fb is not None and fa < fb and all(n is not None and fa < n < fb for n in nums)
    existing = _db_boxes([r["id"] for r in between])
    reason = f"Interpolated between {PurePosixPath(a['rel_path']).name} and {PurePosixPath(b['rel_path']).name}"
    ds = db.query_one("SELECT classes FROM datasets WHERE id=?", (dataset_id,))
    names = db.loads(ds["classes"], []) if ds else []
    # A pair whose class changes between the keyframes (a fall) takes the class of the nearer keyframe: the
    # moment it changed is unknown, and these are proposals the user reviews frame by frame anyway.
    changes = [{"from_box": i, "to_box": j, "from_cls": int(A[i][0]), "to_cls": int(B[j][0])}
               for i, j in pairs if A[i][0] != B[j][0]]
    why = {(c["from_box"], c["to_box"]): f"{reason}; class changes {_class_label(names, c['from_cls'])} → "
           f"{_class_label(names, c['to_cls'])}: this frame takes the nearer keyframe's" for c in changes}
    preds: List[Tuple[int, list, str]] = []
    skipped = 0
    # an object that changes class is already labeled here when a box of either class covers it
    compat = [{int(A[i][0]), int(B[j][0])} for i, j in pairs]
    for k, r in enumerate(between, 1):
        t = (nums[k - 1] - fa) / (fb - fa) if use_nums else k / (len(between) + 1)
        boxes = []
        for i, j in pairs:
            box = lerp_box(A[i], B[j], t)
            if A[i][0] != B[j][0] and t >= 0.5:
                box[0] = int(B[j][0])
            boxes.append(box)
        have = already_labeled(boxes, existing[r["id"]], compat)
        skipped += len(have)
        preds.extend((r["id"], box, why.get(pairs[q], reason)) for q, box in enumerate(boxes) if q not in have)
    params = {"from_image_id": a["id"], "to_image_id": b["id"], "by": "frame_number" if use_nums else "position",
              "matched": len(pairs), "class_changes": len(changes)}
    run_id, out = _store_interp(dataset_id, "interpolate", params, preds, [r["id"] for r in between])
    res = {"run_id": run_id, "images": len({p[0] for p in preds}), "frames": len(between), "preds": out,
           "matched": len(pairs), "unmatched": {"from": len(A) - len(pairs), "to": len(B) - len(pairs)},
           "class_changes": changes, "skipped_existing": skipped, "by": params["by"]}
    if changes:
        n = len(changes)
        kinds = sorted({f"{_class_label(names, c['from_cls'])} → {_class_label(names, c['to_cls'])}" for c in changes})
        res["message"] = (f"{n} box{'es' if n > 1 else ''} change{'' if n > 1 else 's'} class between the keyframes "
                          f"({', '.join(kinds)}): the proposals take the class of the nearer keyframe")
    return res


def fill_gap(dataset_id: str, image_id: int) -> dict:
    row = db.query_one("SELECT id, rel_path, split, seq, width, height, label_path, has_label FROM images WHERE id=? AND dataset_id=?",
                       (image_id, dataset_id))
    if not row:
        raise QAError("Image not found in this dataset")
    if not row["seq"]:
        raise QAError("This image is not part of a sequence")
    frames = _frames_of(dataset_id, row["split"], row["seq"])
    k = next(i for i, r in enumerate(frames) if r["id"] == image_id)
    prev = next((r for r in reversed(frames[:k]) if r["has_label"] and r["n_boxes"]), None)
    nxt = next((r for r in frames[k + 1:] if r["has_label"] and r["n_boxes"]), None)
    if prev is None or nxt is None:
        raise QAError("Gap filling needs a labeled frame on both sides in the same sequence")
    W, H = float(row["width"] or 1), float(row["height"] or 1)
    boxes = _db_boxes([prev["id"], nxt["id"]])
    A, B = boxes[prev["id"]], boxes[nxt["id"]]
    cur = _file_boxes(row)
    t = _weight(_Path(prev["rel_path"]), _Path(row["rel_path"]), _Path(nxt["rel_path"]))
    gaps = missing_boxes(A, cur, B, W, H, t)            # the seq_gap rule itself: the fix fills what the flag reports
    if not gaps:
        return {"run_id": None, "preds": [], "from_image_id": prev["id"], "to_image_id": nxt["id"],
                "message": "Nothing to fill: every object in the neighbouring frames already has a box here"}
    reason = f"Filled from {PurePosixPath(prev['rel_path']).name} and {PurePosixPath(nxt['rel_path']).name}"
    run_id, out = _store_interp(dataset_id, "fill-gap", {"image_id": image_id, "from_image_id": prev["id"], "to_image_id": nxt["id"]},
                                [(image_id, g, reason) for g in gaps], [image_id])
    return {"run_id": run_id, "preds": out, "from_image_id": prev["id"], "to_image_id": nxt["id"]}


class _Path:
    """Just enough of an Item for _weight()."""
    __slots__ = ("rel_path",)

    def __init__(self, rel_path: str):
        self.rel_path = rel_path
