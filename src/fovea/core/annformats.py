"""Annotation formats other than YOLO: recognise, check, preview and convert them.

A dataset whose annotations are not YOLO .txt — VisDrone's pixel CSV, KITTI, DOTA, Pascal VOC .xml, COCO or LabelMe
.json, CVAT .xml, or a format nobody catalogued — is described by a *format card*: what the files are, how one line or
element becomes a box (the reading rule), the official class names and definitions, extra per-box attributes
(occlusion, truncation …) and *regions* that are not objects (VisDrone's ignored regions, KITTI's DontCare, COCO
crowds). Cards come from the catalog below (no AI, no network), else from an AI provider that read sample lines or the
dataset's official documentation (ai/formats.py). Whatever its source, a card is only used after `check` ran its
reading rule over real files and the boxes landed inside their images.

Conversion never touches the original files. YOLO labels go to a new folder (labels/ beside the images, or <name>_yolo
when that name is taken) with two hidden files: .fovea-format.json (the card, the class list and the counts) and
.fovea-annotations.jsonl (per image: each box's geometry with its attributes, and the regions). The index reads the
second one to show attributes and regions for as long as a box keeps its geometry (core.annattrs).
"""
from __future__ import annotations

import base64
import copy
import io
import json
import os
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from ..config import IMAGE_EXTS, LABEL_EXT

SIDECAR = ".fovea-annotations.jsonl"
MARKER = ".fovea-format.json"
MAX_FILE_BYTES = 64 * 1024 * 1024        # one per-image annotation file
MAX_SINGLE_BYTES = 2 * 1024 * 1024 * 1024
SLACK = 0.02                             # a box may overhang its image by 2% before it counts as outside
NOT_LABELS = {"classes.txt", "labels.txt", "obj.names", "readme.txt", "license.txt", "train.txt", "val.txt",
              "valid.txt", "test.txt", "requirements.txt", "notes.txt", "readme.md"}
ANN_EXTS = (".txt", ".xml", ".json")

_NUM = re.compile(r"^[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?$")
_INT = re.compile(r"^[-+]?\d+$")


def is_number(s: str) -> bool:
    return bool(_NUM.match(s or ""))


# ------------------------------------------------------------------------------------------------- YOLO or not

def yolo_line(line: str) -> bool:
    """A YOLO row: space-separated numbers, an integer class id, then normalised values (a box, a polygon or a pose
    row — pose visibility flags may be 2)."""
    toks = line.split()
    if len(toks) < 5 or not all(is_number(t) for t in toks):
        return False
    c = float(toks[0])
    if c < 0 or c != int(c):
        return False
    vals = [float(t) for t in toks[1:]]
    if not all(-0.05 <= v <= 1.05 for v in vals[:4]):
        return False
    return sum(1 for v in vals if -0.05 <= v <= 1.05) >= 0.6 * len(vals)


def txt_kind(lines) -> str:
    """'yolo' when (nearly) every sampled line is a YOLO row, 'empty' when there is no line, else 'other'."""
    ls = [ln.strip() for ln in lines if ln.strip() and not ln.lstrip().startswith("#")]
    if not ls:
        return "empty"
    return "yolo" if sum(1 for ln in ls if yolo_line(ln)) >= 0.9 * len(ls) else "other"


def head_lines(path: Path, n: int = 8, limit: int = 65536) -> List[str]:
    try:
        with open(path, "rb") as fh:
            data = fh.read(limit)
    except OSError:
        return []
    return [ln for ln in data.decode("utf-8-sig", errors="replace").splitlines() if ln.strip()][:n]


def tree_txt_kind(d: Path, files: int = 5, max_dirs: int = 60) -> str:
    """What the .txt files in a folder (or below it) hold: 'yolo' | 'other' | 'empty' | '' (none found)."""
    lines: List[str] = []
    seen = 0
    for k, (cur, dirs, names) in enumerate(os.walk(d)):
        if k >= max_dirs or seen >= files:
            break
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for n in sorted(names)[:4000]:
            if seen >= files:
                break
            low = n.lower()
            if n.startswith(".") or not low.endswith(LABEL_EXT) or low in NOT_LABELS:
                continue
            seen += 1
            lines += head_lines(Path(cur) / n, 6)
    return txt_kind(lines) if seen else ""


# ------------------------------------------------------------------------------------------------- the catalog

VOC_CLASSES = ["aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
               "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]
KITTI_TYPES = ["Car", "Van", "Truck", "Pedestrian", "Person_sitting", "Cyclist", "Tram", "Misc"]
DOTA_CLASSES = ["plane", "ship", "storage-tank", "baseball-diamond", "tennis-court", "basketball-court",
                "ground-track-field", "harbor", "bridge", "large-vehicle", "small-vehicle", "helicopter", "roundabout",
                "soccer-ball-field", "swimming-pool", "container-crane", "airport", "helipad"]
_DOTA_HEADER = r"^(imagesource|gsd)\s*:"

CATALOG: List[Dict[str, Any]] = [
    {
        "id": "visdrone-det", "name": "VisDrone-DET", "kind": "delimited", "ext": ".txt", "single": False,
        "urls": ["https://github.com/VisDrone/VisDrone-Dataset", "https://github.com/VisDrone/VisDrone2018-DET-toolkit"],
        "citation": "Zhu et al., “Detection and Tracking Meet Drones Challenge”, IEEE TPAMI.",
        "notes": "One file per image: <left>,<top>,<width>,<height>,<score>,<category>,<truncation>,<occlusion> in "
                 "pixels. Category 0 marks ignored regions and 11 'others'; neither is a class, and the official "
                 "evaluation skips both. Definitions are summarised from the VisDrone documentation.",
        "spec": {"delimiter": ",", "min_columns": 8,
                 "geometry": {"type": "ltwh", "columns": [0, 1, 2, 3], "units": "pixel"},
                 "class": {"column": 5, "values": {str(i + 1): i for i in range(10)}},
                 "regions": {"0": "ignored", "11": "others"},
                 "attributes": [
                     {"name": "truncation", "label": "Truncation", "column": 6,
                      "values": {"0": "none (0%)", "1": "partial (1–50%)"}},
                     {"name": "occlusion", "label": "Occlusion", "column": 7,
                      "values": {"0": "none (0%)", "1": "partial (1–50%)", "2": "heavy (50–100%)"}}]},
        "classes": ["pedestrian", "people", "bicycle", "car", "van", "truck", "tricycle", "awning-tricycle", "bus",
                    "motor"],
        "class_defs": {
            "pedestrian": {"description": "A person standing or walking.",
                           "exclude": "People sitting, lying, riding or in any other pose (those are 'people')."},
            "people": {"description": "A person in any pose other than standing or walking — sitting, lying, "
                                      "riding a bicycle or motor …",
                       "exclude": "People standing or walking (those are 'pedestrian')."},
            "bicycle": {"description": "A bicycle."},
            "car": {"description": "A car."},
            "van": {"description": "A van."},
            "truck": {"description": "A truck."},
            "tricycle": {"description": "A tricycle without an awning.",
                         "exclude": "Tricycles with an awning (those are 'awning-tricycle')."},
            "awning-tricycle": {"description": "A tricycle with an awning (a canopy over the seats)."},
            "bus": {"description": "A bus."},
            "motor": {"description": "A motorcycle or scooter."},
        },
        "regions": {"ignored": {"label": "Ignored region",
                                "description": "An area not annotated object by object (crowds, unclear objects); "
                                               "not evaluated."},
                    "others": {"label": "Others",
                               "description": "An object outside the ten classes; not evaluated."}},
    },
    {
        "id": "kitti", "name": "KITTI object", "kind": "delimited", "ext": ".txt", "single": False,
        "urls": ["https://www.cvlibs.net/datasets/kitti/eval_object.php"],
        "citation": "Geiger, Lenz, Urtasun, “Are we ready for Autonomous Driving? The KITTI Vision Benchmark "
                    "Suite”, CVPR 2012.",
        "license": "CC BY-NC-SA 3.0",
        "notes": "One file per image: <type> <truncated> <occluded> <alpha> <left> <top> <right> <bottom> <3D "
                 "dimensions, location and rotation> [score], boxes in pixels. DontCare marks regions where objects "
                 "were not labelled.",
        "spec": {"delimiter": "space", "min_columns": 15,
                 "geometry": {"type": "xyxy", "columns": [4, 5, 6, 7], "units": "pixel"},
                 "class": {"column": 0, "from_data": True},
                 "regions": {"DontCare": "dontcare"},
                 "attributes": [
                     {"name": "truncated", "label": "Truncated", "column": 1, "type": "float"},
                     {"name": "occluded", "label": "Occluded", "column": 2,
                      "values": {"0": "fully visible", "1": "partly occluded", "2": "largely occluded",
                                 "3": "unknown"}}]},
        "classes": KITTI_TYPES,
        "class_defs": {"Person_sitting": {"description": "A person sitting (on a chair, a bench …)."},
                       "Cyclist": {"description": "A person riding a bicycle, with the bicycle."},
                       "Misc": {"description": "Other vehicles (trailers, segways …)."}},
        "regions": {"dontcare": {"label": "DontCare",
                                 "description": "A region where objects were not labelled (too far away, unclear)."}},
    },
    {
        "id": "dota", "name": "DOTA", "kind": "delimited", "ext": ".txt", "single": False,
        "urls": ["https://captain-whu.github.io/DOTA/"],
        "citation": "Xia et al., “DOTA: A Large-scale Dataset for Object Detection in Aerial Images”, CVPR 2018.",
        "notes": "One file per image: x1 y1 x2 y2 x3 y3 x4 y4 <category> <difficult>, four corners of an oriented box "
                 "in pixels, after optional imagesource: / gsd: header lines. Oriented boxes are kept as 4-point "
                 "polygons (the YOLO-OBB layout); Annotate shows such rows locked.",
        "spec": {"delimiter": "space", "min_columns": 9, "skip": _DOTA_HEADER,
                 "geometry": {"type": "poly", "columns": [0, 1, 2, 3, 4, 5, 6, 7], "units": "pixel"},
                 "class": {"column": 8, "from_data": True},
                 "attributes": [{"name": "difficult", "label": "Difficult", "column": 9,
                                 "values": {"0": "no", "1": "yes"}}]},
        "classes": DOTA_CLASSES,
        "class_defs": {},
        "regions": {},
    },
    {
        "id": "voc", "name": "Pascal VOC XML", "kind": "voc", "ext": ".xml", "single": False,
        "urls": ["http://host.robots.ox.ac.uk/pascal/VOC/"],
        "citation": "Everingham et al., “The PASCAL Visual Object Classes (VOC) Challenge”, IJCV 2010.",
        "notes": "One .xml per image (<object><name>, <bndbox> in pixels). difficult, truncated, occluded and pose "
                 "become attributes.",
        "spec": {"class": {"from_data": True}},
        "classes": VOC_CLASSES, "class_defs": {}, "regions": {},
        "attributes_hint": [{"name": "difficult", "label": "Difficult", "values": {"0": "no", "1": "yes"}},
                            {"name": "truncated", "label": "Truncated", "values": {"0": "no", "1": "yes"}},
                            {"name": "occluded", "label": "Occluded", "values": {"0": "no", "1": "yes"}},
                            {"name": "pose", "label": "Pose"}],
    },
    {
        "id": "labelme", "name": "LabelMe JSON", "kind": "labelme", "ext": ".json", "single": False,
        "urls": ["https://github.com/wkentaro/labelme"],
        "notes": "One .json per image (shapes with a label and points). Rectangles become boxes and polygons "
                 "polygons; points, lines and circles are skipped. Shape flags become attributes.",
        "spec": {"class": {"from_data": True}}, "classes": [], "class_defs": {}, "regions": {},
    },
    {
        "id": "coco", "name": "COCO JSON", "kind": "coco", "ext": ".json", "single": True,
        "urls": ["https://cocodataset.org/#format-data"],
        "citation": "Lin et al., “Microsoft COCO: Common Objects in Context”, ECCV 2014.",
        "notes": "One .json for many images (images, annotations with a pixel bbox [x, y, w, h], categories). Classes "
                 "follow the category ids; crowd annotations (iscrowd = 1) become regions. Segmentation masks are "
                 "converted to their boxes.",
        "spec": {"regions": {"crowd": "crowd"}}, "classes": [], "class_defs": {},
        "regions": {"crowd": {"label": "Crowd", "description": "A crowd annotated as one region (iscrowd = 1); "
                                                                "not evaluated object by object."}},
    },
    {
        "id": "cvat", "name": "CVAT for images XML", "kind": "cvat", "ext": ".xml", "single": True,
        "urls": ["https://docs.cvat.ai/docs/manual/advanced/formats/format-cvat/"],
        "notes": "One annotations.xml for many images (<image> with <box> and <polygon>, pixel coordinates). The "
                 "occluded flag and <attribute> values become attributes.",
        "spec": {"class": {"from_data": True}}, "classes": [], "class_defs": {}, "regions": {},
    },
]
_BY_ID = {c["id"]: c for c in CATALOG}


def catalog_card(fid: str) -> Optional[dict]:
    c = _BY_ID.get(fid)
    if c is None:
        return None
    card = copy.deepcopy(c)
    card["source"] = "catalog"
    return card


def attribute_schema(card: dict) -> List[dict]:
    """[{name, label, type, values: {code: meaning}}] — what the UI shows and filters on."""
    out = []
    for a in (card.get("spec") or {}).get("attributes") or card.get("attributes_hint") or []:
        out.append({"name": a["name"], "label": a.get("label") or a["name"].replace("_", " ").capitalize(),
                    "type": a.get("type") or ("enum" if a.get("values") else "text"),
                    "values": dict(a.get("values") or {})})
    return out


# ------------------------------------------------------------------------------------------------- sniffing

@dataclass
class Sniff:
    path: Path
    single: bool                       # one file for many images (COCO, CVAT)
    ext: str
    files: List[Path] = field(default_factory=list)
    lines: List[str] = field(default_factory=list)
    objs: list = field(default_factory=list)       # parsed .json objects / .xml roots of the sampled files
    head: str = ""                                 # single files: the start (and end) of the text
    hint: str = ""                                 # lower-cased path names (dataset names show up there)
    count: int = 0                                 # annotation files of `ext` seen

    def sample_text(self, n: int = 24) -> List[str]:
        return self.lines[:n] if self.lines else [self.head[:1500]]


def _pick_files(d: Path, limit: int = 6, max_entries: int = 20000) -> Tuple[str, List[Path], int]:
    counts: Dict[str, List[Path]] = {e: [] for e in ANN_EXTS}
    n_by = {e: 0 for e in ANN_EXTS}
    seen = 0
    for cur, dirs, names in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for n in sorted(names):
            seen += 1
            low = n.lower()
            dot = low.rfind(".")
            ext = low[dot:] if dot > 0 else ""
            if ext not in counts or n.startswith(".") or low in NOT_LABELS:
                continue
            n_by[ext] += 1
            if len(counts[ext]) < limit:
                counts[ext].append(Path(cur) / n)
        if seen > max_entries:
            break
    ext = max(ANN_EXTS, key=lambda e: n_by[e])
    return (ext, counts[ext], n_by[ext]) if n_by[ext] else ("", [], 0)


def _read_head(p: Path, head: int = 262144, tail: int = 65536) -> str:
    try:
        size = p.stat().st_size
        with open(p, "rb") as fh:
            a = fh.read(head)
            b = b""
            if size > head + tail:
                fh.seek(size - tail)
                b = fh.read(tail)
    except OSError:
        return ""
    return (a + b"\n" + b).decode("utf-8-sig", errors="replace")


def sniff(path: Path, hint: str = "") -> Optional[Sniff]:
    path = Path(path)
    if path.is_file():
        ext = path.suffix.lower()
        if ext not in (".json", ".xml"):
            return None
        return Sniff(path=path, single=True, ext=ext, files=[path], head=_read_head(path), hint=hint.lower(), count=1)
    if not path.is_dir():
        return None
    ext, files, n = _pick_files(path)
    if not ext:
        return None
    s = Sniff(path=path, single=False, ext=ext, files=files, hint=hint.lower(), count=n)
    for f in files:
        if ext == ".txt":
            s.lines += head_lines(f, 12)
            continue
        try:
            if f.stat().st_size > 8 * 1024 * 1024:
                continue
            raw = f.read_bytes()
            s.objs.append(json.loads(raw.decode("utf-8-sig")) if ext == ".json" else ET.fromstring(raw))
        except Exception:  # noqa: BLE001 — a malformed sample file just does not vote
            continue
    return s


_WRAP = "()[]{}<>\"'"


def _split(line: str, delim: str) -> List[str]:
    """One line's tokens. Brackets and quotes around a value are dropped — NWPU VHR-10 writes "(563,478),(630,573),1",
    other tools wrap names in quotes — and a trailing delimiter (VisDrone ends lines with a comma) adds no column."""
    if delim == "space":
        parts = line.split()
    elif delim == "tab":
        parts = line.split("\t")
    else:
        parts = line.split(delim)
    parts = [t.strip().strip(_WRAP).strip() for t in parts]
    while parts and parts[-1] == "":
        parts.pop()
    return parts


def _fp_visdrone(s: Sniff) -> float:
    rows = [_split(ln.strip(), ",") for ln in s.lines]
    if not rows:
        return 0.0
    ok = 0
    for t in rows:
        if len(t) != 8 or not all(_INT.match(x) for x in t):
            continue
        v = [int(x) for x in t]
        if v[4] in (0, 1) and 0 <= v[5] <= 11 and v[6] in (0, 1, 2) and v[7] in (0, 1, 2) and v[2] >= 0 and v[3] >= 0:
            ok += 1
    return (0.85 + (0.1 if "visdrone" in s.hint else 0.0)) if ok >= 0.95 * len(rows) else 0.0


def _fp_kitti(s: Sniff) -> float:
    rows = [ln.split() for ln in s.lines]
    if not rows:
        return 0.0
    ok = [t for t in rows if len(t) in (15, 16) and not is_number(t[0]) and all(is_number(x) for x in t[1:])]
    if len(ok) < 0.95 * len(rows):
        return 0.0
    known = sum(1 for t in ok if t[0] in KITTI_TYPES or t[0] == "DontCare")
    return 0.8 + (0.15 if known >= 0.9 * len(ok) else 0.0)


def _fp_dota(s: Sniff) -> float:
    hdr = re.compile(_DOTA_HEADER, re.I)
    rows = [ln.split() for ln in s.lines if not hdr.match(ln.strip())]
    if not rows:
        return 0.0
    ok = sum(1 for t in rows if len(t) in (9, 10) and all(is_number(x) for x in t[:8]) and not is_number(t[8])
             and (len(t) == 9 or t[9] in ("0", "1")))
    if ok < 0.95 * len(rows):
        return 0.0
    return 0.8 + (0.15 if any(hdr.match(ln.strip()) for ln in s.lines) or "dota" in s.hint else 0.0)


def _fp_voc(s: Sniff) -> float:
    roots = [o for o in s.objs if isinstance(o, ET.Element)]
    ok = [r for r in roots if r.tag == "annotation" and (r.find("object") is not None or r.find("size") is not None)]
    return 0.95 if roots and len(ok) == len(roots) else 0.0


def _fp_labelme(s: Sniff) -> float:
    objs = [o for o in s.objs if isinstance(o, dict)]
    ok = [o for o in objs if isinstance(o.get("shapes"), list) and ("imagePath" in o or "imageWidth" in o)]
    return 0.95 if objs and len(ok) == len(objs) else 0.0


def _fp_coco(s: Sniff) -> float:
    h = s.head
    return 0.9 if re.search(r'"images"\s*:\s*\[', h) and '"file_name"' in h and (
        '"annotations"' in h or '"categories"' in h) else 0.0


def _fp_cvat(s: Sniff) -> float:
    h = s.head[:65536]
    return 0.95 if "<annotations" in h and re.search(r"<image\s", h) else 0.0


_FINGERPRINTS: Dict[str, Callable[[Sniff], float]] = {
    "visdrone-det": _fp_visdrone, "kitti": _fp_kitti, "dota": _fp_dota, "voc": _fp_voc, "labelme": _fp_labelme,
    "coco": _fp_coco, "cvat": _fp_cvat,
}


def recognize(path: Path, hint: str = "") -> Tuple[Optional[dict], Optional[Sniff], str]:
    """(card, sniff, how): how = 'catalog' (a known format), 'yolo' / 'empty' (nothing to convert), 'unknown' (an
    AI may read it) or 'none' (no annotation files)."""
    s = sniff(path, hint)
    if s is None:
        return None, None, "none"
    if s.ext == ".txt":
        k = txt_kind(s.lines)
        if k in ("yolo", "empty"):
            return None, s, k
    best, score = None, 0.0
    for c in CATALOG:
        if c["ext"] != s.ext or bool(c["single"]) != s.single:
            continue
        v = _FINGERPRINTS[c["id"]](s)
        if v > score:
            best, score = c, v
    if best is not None and score >= 0.6:
        return catalog_card(best["id"]), s, "catalog"
    return None, s, "unknown"


# ------------------------------------------------------------------------------------------------- parsing

@dataclass
class Parsed:
    rows: List[list] = field(default_factory=list)      # YOLO rows: [cls, xc, yc, w, h] or [cls, x1, y1, …, xn, yn]
    attrs: List[dict] = field(default_factory=list)     # one per row
    regions: List[list] = field(default_factory=list)   # [kind, xc, yc, w, h]
    total: int = 0      # annotation entries read
    bad: int = 0        # entries that could not be read
    unknown: int = 0    # entries whose class value the card does not map
    oob: int = 0        # boxes clearly outside the image
    tiny: int = 0       # zero-area boxes (after clipping to the image)
    dropped: int = 0    # entries skipped on purpose (unsupported shapes)
    unknown_values: Dict[str, int] = field(default_factory=dict)


def _canon(v: str) -> str:
    """'1', '1.0', ' 1 ' → '1' for whole numbers; anything else as written."""
    v = (v or "").strip()
    if _NUM.match(v):
        f = float(v)
        if f == int(f) and abs(f) < 1e15:
            return str(int(f))
    return v


def _corners(gtype: str, v: List[float]) -> Tuple[float, float, float, float]:
    if gtype == "ltwh":
        return v[0], v[1], v[0] + v[2], v[1] + v[3]
    if gtype == "cxcywh":
        return v[0] - v[2] / 2, v[1] - v[3] / 2, v[0] + v[2] / 2, v[1] + v[3] / 2
    return v[0], v[1], v[2], v[3]                  # xyxy


def _norm_box(x1: float, y1: float, x2: float, y2: float, W: float, H: float, out: Parsed) -> Optional[List[float]]:
    x1, x2 = min(x1, x2), max(x1, x2)
    y1, y2 = min(y1, y2), max(y1, y2)
    if x1 < -SLACK * W or y1 < -SLACK * H or x2 > W * (1 + SLACK) or y2 > H * (1 + SLACK):
        out.oob += 1
    cx1, cx2 = min(max(x1, 0.0), W), min(max(x2, 0.0), W)
    cy1, cy2 = min(max(y1, 0.0), H), min(max(y2, 0.0), H)
    if cx2 - cx1 <= 0 or cy2 - cy1 <= 0:
        out.tiny += 1
        return None
    return [(cx1 + cx2) / 2 / W, (cy1 + cy2) / 2 / H, (cx2 - cx1) / W, (cy2 - cy1) / H]


def _norm_poly(pts: List[float], W: float, H: float, out: Parsed) -> Optional[List[float]]:
    xs, ys = pts[0::2], pts[1::2]
    if len(xs) < 3 or len(xs) != len(ys):
        out.bad += 1
        return None
    if min(xs) < -SLACK * W or min(ys) < -SLACK * H or max(xs) > W * (1 + SLACK) or max(ys) > H * (1 + SLACK):
        out.oob += 1
    nx = [min(max(x / W, 0.0), 1.0) for x in xs]
    ny = [min(max(y / H, 0.0), 1.0) for y in ys]
    if max(nx) - min(nx) <= 0 or max(ny) - min(ny) <= 0:
        out.tiny += 1
        return None
    return [v for xy in zip(nx, ny) for v in xy]


def row_bbox(row: List[float]) -> List[float]:
    """The box Fovea's label parser reports for a written row (core.labels._classify_extended), computed from the
    6-decimal values that are written to disk — so it matches the index exactly."""
    vals = [float(f"{v:.6f}") for v in row[1:]]
    if len(vals) == 4:
        return vals
    xs, ys = vals[0::2], vals[1::2]
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    return [(x0 + x1) / 2, (y0 + y1) / 2, x1 - x0, y1 - y0]


def row_text(row: List[float]) -> str:
    return " ".join([str(int(row[0]))] + [f"{float(v):.6f}" for v in row[1:]])


def _add(out: Parsed, raw_cls: str, geom: Optional[List[float]], is_poly: bool, attrs: dict, card: dict,
         index: Dict[str, int]) -> None:
    if geom is None:
        return
    region = (card.get("spec") or {}).get("regions", {}).get(raw_cls)
    if region:
        bb = row_bbox([0] + geom) if is_poly else geom
        out.regions.append([region] + [round(v, 6) for v in bb])
        return
    k = index.get(raw_cls)
    if k is None:
        out.unknown += 1
        out.unknown_values[raw_cls] = out.unknown_values.get(raw_cls, 0) + 1
        return
    out.rows.append([k] + geom)
    out.attrs.append(attrs)


def parse_delimited(text: str, card: dict, W: float, H: float, index: Dict[str, int]) -> Parsed:
    spec = card["spec"]
    out = Parsed()
    skip = re.compile(spec["skip"], re.I) if spec.get("skip") else None
    g = spec["geometry"]
    gcols = [int(c) for c in g["columns"]]
    if g.get("units") == "normalized":
        W = H = 1.0
    ccol = (spec.get("class") or {}).get("column")
    delim = spec.get("delimiter", "space")
    for raw in text.splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or (skip and skip.match(line)):
            continue
        out.total += 1
        toks = _split(line, delim)
        if len(toks) < int(spec.get("min_columns") or 0) or max(gcols) >= len(toks):
            out.bad += 1
            continue
        try:
            v = [float(toks[i]) for i in gcols]
        except ValueError:
            out.bad += 1
            continue
        raw_cls = _canon(toks[ccol]) if ccol is not None and 0 <= ccol < len(toks) else ""
        attrs = {}
        for a in spec.get("attributes") or []:
            col = int(a["column"])
            if col >= len(toks):
                continue
            attrs[a["name"]] = (round(float(toks[col]), 4) if a.get("type") == "float" and is_number(toks[col])
                                else _canon(toks[col]))
        if g["type"] == "poly":
            _add(out, raw_cls, _norm_poly(v, W, H, out), True, attrs, card, index)
        else:
            _add(out, raw_cls, _norm_box(*_corners(g["type"], v), W, H, out), False, attrs, card, index)
    return out


def _xml_num(el: Optional[ET.Element], tag: str) -> Optional[float]:
    t = el.findtext(tag) if el is not None else None
    try:
        return float(t) if t is not None else None
    except ValueError:
        return None


def parse_voc(root: ET.Element, card: dict, W: float, H: float, index: Dict[str, int]) -> Parsed:
    out = Parsed()
    size = root.find("size")
    sw, sh = _xml_num(size, "width"), _xml_num(size, "height")
    if sw and sh and sw > 0 and sh > 0:      # boxes are in the frame the XML was written for
        W, H = sw, sh
    for obj in root.iter("object"):
        out.total += 1
        name = (obj.findtext("name") or "").strip()
        bb = obj.find("bndbox")
        vals = [_xml_num(bb, t) for t in ("xmin", "ymin", "xmax", "ymax")]
        if bb is None or any(v is None for v in vals):
            out.bad += 1
            continue
        attrs = {}
        for t in ("difficult", "truncated", "occluded", "pose"):
            v = obj.findtext(t)
            if v is not None and v.strip():
                attrs[t] = _canon(v)
        for a in obj.iter("attribute"):           # CVAT's VOC export: <attributes><attribute><name/><value/>
            n, val = a.findtext("name"), a.findtext("value")
            if n:
                attrs[n.strip()] = _canon(val or "")
        _add(out, name, _norm_box(vals[0], vals[1], vals[2], vals[3], W, H, out), False, attrs, card, index)
    return out


def parse_labelme(doc: dict, card: dict, W: float, H: float, index: Dict[str, int]) -> Parsed:
    out = Parsed()
    try:
        dw, dh = float(doc.get("imageWidth") or 0), float(doc.get("imageHeight") or 0)
    except (TypeError, ValueError):
        dw = dh = 0.0
    if dw > 0 and dh > 0:
        W, H = dw, dh
    for sh in doc.get("shapes") or []:
        out.total += 1
        if not isinstance(sh, dict):
            out.bad += 1
            continue
        label = str(sh.get("label") or "").strip()
        st = (sh.get("shape_type") or "polygon").lower()
        try:
            pts = [float(c) for p in sh.get("points") or [] for c in p[:2]]
        except (TypeError, ValueError):
            out.bad += 1
            continue
        attrs = {k: _canon(str(v).lower() if isinstance(v, bool) else str(v))
                 for k, v in (sh.get("flags") or {}).items() if isinstance(k, str)}
        if st == "rectangle" and len(pts) >= 4:
            _add(out, label, _norm_box(pts[0], pts[1], pts[2], pts[3], W, H, out), False, attrs, card, index)
        elif st == "polygon" and len(pts) >= 6:
            _add(out, label, _norm_poly(pts, W, H, out), True, attrs, card, index)
        else:
            out.dropped += 1
    return out


def _coco_entry(anns: List[dict], img: dict, card: dict, W: float, H: float, index: Dict[str, int]) -> Parsed:
    out = Parsed()
    try:
        dw, dh = float(img.get("width") or 0), float(img.get("height") or 0)
    except (TypeError, ValueError):
        dw = dh = 0.0
    if dw > 0 and dh > 0:
        W, H = dw, dh
    for a in anns:
        out.total += 1
        bb = a.get("bbox")
        if not (isinstance(bb, list) and len(bb) == 4):
            out.bad += 1
            continue
        try:
            x, y, w, h = (float(v) for v in bb)
        except (TypeError, ValueError):
            out.bad += 1
            continue
        attrs = {}
        if isinstance(a.get("attributes"), dict):
            attrs = {str(k): _canon(str(v).lower() if isinstance(v, bool) else str(v))
                     for k, v in a["attributes"].items()}
        raw_cls = "crowd" if a.get("iscrowd") in (1, True) else str(a.get("category_id"))
        _add(out, raw_cls, _norm_box(x, y, x + w, y + h, W, H, out), False, attrs, card, index)
    return out


def _cvat_points(s: str) -> List[float]:
    out = []
    for p in (s or "").split(";"):
        xy = p.split(",")
        if len(xy) == 2 and is_number(xy[0]) and is_number(xy[1]):
            out += [float(xy[0]), float(xy[1])]
    return out


def _cvat_entry(el: ET.Element, card: dict, W: float, H: float, index: Dict[str, int]) -> Parsed:
    out = Parsed()
    try:
        dw, dh = float(el.get("width") or 0), float(el.get("height") or 0)
    except ValueError:
        dw = dh = 0.0
    if dw > 0 and dh > 0:
        W, H = dw, dh
    for sh in el:
        if sh.tag not in ("box", "polygon", "polyline", "points", "ellipse", "mask", "skeleton", "cuboid", "tag"):
            continue
        out.total += 1
        label = (sh.get("label") or "").strip()
        attrs = {}
        if sh.get("occluded") is not None:
            attrs["occluded"] = _canon(sh.get("occluded"))
        for a in sh.iter("attribute"):
            if a.get("name"):
                attrs[a.get("name")] = _canon(a.text or "")
        if sh.tag == "box":
            try:
                v = [float(sh.get(k)) for k in ("xtl", "ytl", "xbr", "ybr")]
            except (TypeError, ValueError):
                out.bad += 1
                continue
            _add(out, label, _norm_box(*v, W, H, out), False, attrs, card, index)
        elif sh.tag == "polygon":
            _add(out, label, _norm_poly(_cvat_points(sh.get("points")), W, H, out), True, attrs, card, index)
        else:
            out.dropped += 1
    return out


# ------------------------------------------------------------------------------------------------- reading a source

def _natural(s: str):
    return [(0, int(t), "") if t.isdigit() else (1, 0, t) for t in re.split(r"(\d+)", s.casefold()) if t]


def list_images(img_dir: Path, recursive: bool = True, limit: Optional[int] = None,
                skip: Tuple[str, ...] = ()) -> List[Tuple[str, Path]]:
    """(rel posix path, path) of the images below `img_dir`, sorted — the order the scanner indexes them in."""
    out: List[Tuple[str, Path]] = []
    skip_n = {os.path.normpath(s) for s in skip if s}
    for cur, dirs, names in os.walk(img_dir, followlinks=True):
        dirs[:] = sorted((x for x in dirs if not x.startswith(".") and os.path.normpath(os.path.join(cur, x))
                          not in skip_n), key=_natural) if recursive else []
        for n in sorted(names, key=_natural):
            dot = n.rfind(".")
            if n.startswith(".") or dot <= 0 or n[dot:].lower() not in IMAGE_EXTS:
                continue
            p = Path(cur) / n
            out.append((p.relative_to(img_dir).as_posix(), p))
            if limit and len(out) >= limit:
                return out
    return out


def image_size(path: Path) -> Optional[Tuple[int, int]]:
    from .scanner import read_dims          # noqa: PLC0415 — scanner imports layout, which imports this module
    return read_dims(str(path))


class Reader:
    """A card applied to one image folder and its annotations (a folder of per-image files, or one file)."""

    def __init__(self, card: dict, ann: Path, img_dir: Path, recursive: bool = True):
        self.card, self.ann, self.img_dir, self.recursive = card, Path(ann), Path(img_dir), recursive
        self.kind = card.get("kind") or "delimited"
        self.single = bool(card.get("single"))
        self.ext = card.get("ext") or ".txt"
        self.classes: List[str] = list(card.get("classes") or [])
        self.index: Dict[str, int] = {}
        self._entries: Optional[Dict[str, Any]] = None
        self._names: Dict[str, int] = {}
        spec_cls = (card.get("spec") or {}).get("class") or {}
        if self.kind == "delimited" and spec_cls.get("values"):
            self.index = {str(k): int(v) for k, v in spec_cls["values"].items()}
            if not self.classes:
                self.classes = [f"class_{i}" for i in range(max(self.index.values(), default=-1) + 1)]
        elif self.kind == "delimited" and spec_cls.get("column") is None:
            self.index = {"": 0}
            self.classes = self.classes[:1] or ["object"]
        self.from_data = not self.index and self.kind != "coco"
        if self.single:
            self._load_single()

    # -- single-file formats: COCO json / CVAT xml
    def _load_single(self) -> None:
        if self.ann.stat().st_size > MAX_SINGLE_BYTES:
            raise ValueError(f"{self.ann.name} is larger than {MAX_SINGLE_BYTES // 2**30} GB")
        entries: Dict[str, Any] = {}
        if self.kind == "coco":
            doc = json.loads(self.ann.read_bytes().decode("utf-8-sig"))
            cats = sorted((c for c in doc.get("categories") or [] if isinstance(c, dict) and "id" in c),
                          key=lambda c: int(c["id"]))
            self.classes = [str(c.get("name") or f"class_{c['id']}") for c in cats]
            self.index = {str(c["id"]): i for i, c in enumerate(cats)}
            by_img: Dict[Any, List[dict]] = {}
            for a in doc.get("annotations") or []:
                if isinstance(a, dict):
                    by_img.setdefault(a.get("image_id"), []).append(a)
            for im in doc.get("images") or []:
                if isinstance(im, dict) and im.get("file_name"):
                    entries[str(im["file_name"]).replace("\\", "/")] = (by_img.get(im.get("id"), []), im)
        else:
            root = ET.parse(self.ann).getroot()
            meta = [n.text.strip() for n in root.iter("name") if n.text and n.text.strip()
                    and _parent_tag(root, n) == "label"]
            for n in meta:
                if n not in self._names:
                    self._names[n] = len(self._names)
            for im in root.iter("image"):
                if im.get("name"):
                    entries[im.get("name").replace("\\", "/")] = im
        self._entries = entries
        self._by_base: Dict[str, List[str]] = {}
        for k in entries:
            self._by_base.setdefault(PurePosixPath(k).name, []).append(k)

    def _single_entry(self, rel: str):
        e = self._entries or {}
        if rel in e:
            return e[rel]
        keys = [k for k in self._by_base.get(PurePosixPath(rel).name, []) if k == rel or k.endswith("/" + rel)
                or rel.endswith("/" + k)] or self._by_base.get(PurePosixPath(rel).name, [])
        return e[keys[0]] if len(keys) == 1 else None

    def entry_names(self) -> List[str]:
        return list((self._entries or {}).keys())

    # -- per-image files
    def ann_file(self, rel: str) -> Optional[Path]:
        stem = PurePosixPath(rel).with_suffix(self.ext)
        for cand in (self.ann / stem, self.ann / stem.name):
            if cand.is_file():
                return cand
        return None

    def has_annotation(self, rel: str) -> bool:
        return (self._single_entry(rel) is not None) if self.single else self.ann_file(rel) is not None

    # -- classes found in the data (KITTI types, VOC names, DOTA categories …)
    def _raw_values(self, rel: str) -> List[str]:
        spec = self.card.get("spec") or {}
        regions = spec.get("regions") or {}
        vals: List[str] = []
        if self.single:
            e = self._single_entry(rel)
            if e is not None and self.kind == "cvat":
                vals = [(sh.get("label") or "").strip() for sh in e if sh.tag in ("box", "polygon")]
        else:
            f = self.ann_file(rel)
            if f is None:
                return []
            try:
                if self.kind == "delimited":
                    col = int((spec.get("class") or {}).get("column") or 0)
                    skip = re.compile(spec["skip"], re.I) if spec.get("skip") else None
                    for ln in _read_text(f).splitlines():
                        ln = ln.strip()
                        if not ln or ln.startswith("#") or (skip and skip.match(ln)):
                            continue
                        t = _split(ln, spec.get("delimiter", "space"))
                        if col < len(t):
                            vals.append(_canon(t[col]))
                elif self.kind == "voc":
                    vals = [(o.findtext("name") or "").strip() for o in ET.parse(f).getroot().iter("object")]
                elif self.kind == "labelme":
                    doc = json.loads(_read_text(f))
                    vals = [str(s.get("label") or "").strip() for s in doc.get("shapes") or [] if isinstance(s, dict)]
            except Exception:  # noqa: BLE001 — unreadable files are counted when parsed
                return []
        return [v for v in vals if v and v not in regions]

    def collect_classes(self, rels: List[str]) -> None:
        """Class list for formats that name classes in the data: the catalog's order first (KITTI, VOC, DOTA), then
        the other names in natural order."""
        if not self.from_data:
            return
        seen = dict(self._names)
        for rel in rels:
            for v in self._raw_values(rel):
                seen.setdefault(v, len(seen))
        order = [c for c in self.card.get("classes") or [] if c in seen]
        rest = sorted((v for v in seen if v not in order), key=_natural)
        if self.kind == "cvat" and self._names:     # CVAT lists its labels in the task's order
            order, rest = [n for n in self._names], sorted((v for v in seen if v not in self._names), key=_natural)
        self.classes = order + rest
        self.index = {n: i for i, n in enumerate(self.classes)}

    def parse(self, rel: str, W: float, H: float) -> Optional[Parsed]:
        """The image's annotation as YOLO rows, or None when it has no annotation."""
        if self.single:
            e = self._single_entry(rel)
            if e is None:
                return None
            if self.kind == "coco":
                return _coco_entry(e[0], e[1], self.card, W, H, self.index)
            return _cvat_entry(e, self.card, W, H, self.index)
        f = self.ann_file(rel)
        if f is None:
            return None
        try:
            if f.stat().st_size > MAX_FILE_BYTES:
                return Parsed(total=1, bad=1)
            if self.kind == "delimited":
                return parse_delimited(_read_text(f), self.card, W, H, self.index)
            if self.kind == "voc":
                return parse_voc(ET.parse(f).getroot(), self.card, W, H, self.index)
            if self.kind == "labelme":
                return parse_labelme(json.loads(_read_text(f)), self.card, W, H, self.index)
        except Exception:  # noqa: BLE001 — a malformed file is one bad entry, not a failed conversion
            return Parsed(total=1, bad=1)
        return None


def _parent_tag(root: ET.Element, node: ET.Element) -> str:
    for p in root.iter():
        for c in p:
            if c is node:
                return p.tag
    return ""


def _read_text(p: Path) -> str:
    return p.read_bytes().decode("utf-8-sig", errors="replace")


# ------------------------------------------------------------------------------------------------- checking

def _spread(items: list, n: int) -> list:
    if len(items) <= n:
        return items
    step = len(items) / n
    return [items[int(i * step)] for i in range(n)]


def check(card: dict, img_dir: Path, ann: Path, recursive: bool = True, sample: int = 200,
          seconds: float = 4.0, skip: Tuple[str, ...] = ()) -> dict:
    """Apply the card's reading rule to a spread sample of the folder's images. Returns the counts and `ok` with the
    `problems` that make it unusable — a card is only ever used when this passes."""
    t0 = time.monotonic()
    st: Dict[str, Any] = {"images": 0, "annotated": 0, "entries": 0, "boxes": 0, "regions": 0, "bad": 0,
                          "unknown": 0, "oob": 0, "tiny": 0, "dropped": 0, "unknown_values": {}, "class_counts": {},
                          "attr_values": {}}
    try:
        reader = Reader(card, ann, img_dir, recursive)
    except Exception as e:  # noqa: BLE001
        return {**st, "ok": False, "problems": [f"Could not read {Path(ann).name}: {e}"], "classes": []}
    imgs = list_images(img_dir, recursive, limit=20000, skip=skip)
    picks = [x for x in _spread(imgs, sample * 3) if reader.has_annotation(x[0])][:sample]
    st["images"], st["annotated"] = len(imgs), len(picks)
    reader.collect_classes([r for r, _p in picks])
    counts: Dict[int, int] = {}
    for rel, p in picks:
        if time.monotonic() - t0 > seconds:
            break
        dims = image_size(p)
        if not dims:
            continue
        r = reader.parse(rel, float(dims[0]), float(dims[1]))
        if r is None:
            continue
        st["entries"] += r.total
        st["boxes"] += len(r.rows)
        st["regions"] += len(r.regions)
        for k in ("bad", "unknown", "oob", "tiny", "dropped"):
            st[k] += getattr(r, k)
        for v, n in r.unknown_values.items():
            st["unknown_values"][v] = st["unknown_values"].get(v, 0) + n
        for row, a in zip(r.rows, r.attrs):
            counts[int(row[0])] = counts.get(int(row[0]), 0) + 1
            for k, v in a.items():
                if isinstance(v, str) and len(st["attr_values"].setdefault(k, {})) < 50:
                    st["attr_values"][k][v] = st["attr_values"][k].get(v, 0) + 1
    st["class_counts"] = {str(k): v for k, v in sorted(counts.items())}
    st["classes"] = reader.classes
    st["checked"] = len(picks)
    problems = []
    n = max(st["entries"], 1)
    if not picks:
        problems.append("No image has an annotation file with the same name")
    elif st["annotated"] < 0.2 * min(len(imgs), sample * 3) and len(imgs) > 20:
        problems.append(f"Only {st['annotated']} of the sampled images have an annotation")
    if st["bad"] > 0.02 * n:
        problems.append(f"{st['bad']} of {st['entries']} entries could not be read with this rule")
    if st["unknown"] > 0.02 * n:
        vals = ", ".join(list(st["unknown_values"])[:6])
        problems.append(f"{st['unknown']} entries have class values the rule does not map ({vals})")
    if st["oob"] > 0.05 * n:
        problems.append(f"{st['oob']} boxes lie outside their image — the coordinates are read wrongly")
    if st["tiny"] > 0.05 * n:
        problems.append(f"{st['tiny']} boxes have no area")
    if picks and st["entries"] and not st["boxes"]:
        problems.append("The rule found no boxes")
    st["ok"] = not problems
    st["problems"] = problems
    st["seconds"] = round(time.monotonic() - t0, 2)
    return st


# ------------------------------------------------------------------------------------------------- previews

PALETTE = ['#f04b4b', '#00e15a', '#00b4ff', '#ffb400', '#b469ff', '#879600', '#ff96c3', '#00d2c3', '#b4783c', '#2d965a',
           '#5a87ff', '#ff0096', '#ffa578', '#b4d200', '#e1a5ff', '#f05a00', '#b4963c', '#ff5ae1', '#a5a5ff', '#78b45a']


def class_color(i: int) -> str:
    """static/lib/colors.js classColor (default palette, golden-angle hues beyond it)."""
    i = abs(int(i))
    if i < len(PALETTE):
        return PALETTE[i]
    h, s, ll = (i * 137.508) % 360, 0.72, (0.6 if i % 2 else 0.52)
    a = s * min(ll, 1 - ll)

    def f(n):
        k = (n + h / 30) % 12
        return round(255 * (ll - a * max(-1, min(k - 3, min(9 - k, 1)))))
    return "#%02x%02x%02x" % (f(0), f(8), f(4))


def preview(card: dict, img_dir: Path, ann: Path, recursive: bool = True, n: int = 4, edge: int = 420,
            skip: Tuple[str, ...] = (), seconds: float = 3.0) -> List[dict]:
    """A few sample images with the converted boxes (class colours), polygons and hatched regions drawn on them, as
    JPEG data URLs — what the add dialog shows before anything is written."""
    from PIL import Image, ImageDraw, ImageOps
    t0 = time.monotonic()
    try:
        reader = Reader(card, ann, img_dir, recursive)
    except Exception:  # noqa: BLE001
        return []
    imgs = [x for x in _spread(list_images(img_dir, recursive, limit=5000, skip=skip), 60)
            if reader.has_annotation(x[0])]
    reader.collect_classes([r for r, _p in imgs])
    out = []
    for rel, p in imgs:
        if len(out) >= n or time.monotonic() - t0 > seconds:
            break
        dims = image_size(p)
        if not dims:
            continue
        r = reader.parse(rel, float(dims[0]), float(dims[1]))
        if r is None or not (r.rows or r.regions):
            continue
        try:
            with Image.open(p) as im:
                im.draft("RGB", (edge * 2, edge * 2))
                im = ImageOps.exif_transpose(im).convert("RGB")
                im.thumbnail((edge, edge))
                base = im.convert("RGBA")
        except Exception:  # noqa: BLE001
            continue
        W, H = base.size
        for kind, xc, yc, w, h in r.regions:
            x0, y0 = int((xc - w / 2) * W), int((yc - h / 2) * H)
            rw, rh = max(1, int(w * W)), max(1, int(h * H))
            tile = Image.new("RGBA", (rw, rh), (40, 40, 40, 90))
            d = ImageDraw.Draw(tile)
            for k in range(-rh, rw, 6):
                d.line([(k, 0), (k + rh, rh)], fill=(230, 230, 230, 150), width=1)
            d.rectangle([0, 0, rw - 1, rh - 1], outline=(230, 230, 230, 220))
            base.alpha_composite(tile, (max(0, x0), max(0, y0)))
        draw = ImageDraw.Draw(base)
        for row in r.rows:
            col = class_color(int(row[0]))
            if len(row) == 5:
                xc, yc, w, h = row[1:5]
                draw.rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H], outline=col,
                               width=2)
            else:
                pts = [(row[i] * W, row[i + 1] * H) for i in range(1, len(row) - 1, 2)]
                draw.polygon(pts, outline=col)
        buf = io.BytesIO()
        base.convert("RGB").save(buf, "JPEG", quality=80)
        out.append({"name": rel, "width": W, "height": H, "boxes": len(r.rows), "regions": len(r.regions),
                    "data": "data:image/jpeg;base64," + base64.b64encode(buf.getvalue()).decode("ascii")})
    return out


# ------------------------------------------------------------------------------------------------- converting

def output_dir_for(label_dir: Path, ann: Path) -> Path:
    """Where converted labels go: the source's label folder unless that is the annotation folder itself (YOLO .txt
    beside VisDrone .txt would overwrite them) or a folder of someone else's files; then <name>_yolo."""
    label_dir, ann = Path(label_dir), Path(ann)
    if ann.is_dir() and os.path.normpath(label_dir) == os.path.normpath(ann):
        label_dir = ann.parent / f"{ann.name}_yolo"
    if label_dir.is_dir() and not (label_dir / MARKER).is_file() and any(
            n.lower().endswith(LABEL_EXT) for n in _names(label_dir, 2000)):
        label_dir = label_dir.parent / f"{label_dir.name}_yolo"
    return label_dir


def _names(d: Path, limit: int) -> List[str]:
    try:
        with os.scandir(d) as it:
            return [e.name for _, e in zip(range(limit), it)]
    except OSError:
        return []


def read_marker(label_dir: Path) -> Optional[dict]:
    """The .fovea-format.json of a folder Fovea converted into — `state` 'converting' while (or when) a conversion
    did not finish, 'done' after."""
    try:
        m = json.loads((Path(label_dir) / MARKER).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return m if isinstance(m, dict) else None


def is_converted(label_dir: Path) -> bool:
    m = read_marker(label_dir)
    return bool(m) and m.get("state", "done") == "done"


def convert(card: dict, img_dir: Path, ann: Path, out_dir: Path, recursive: bool = True,
            progress: Optional[Callable[[int, int], None]] = None, skip: Tuple[str, ...] = (),
            classes: Optional[List[str]] = None) -> dict:
    """Write YOLO labels for every image of the folder that has an annotation, plus the format marker and the
    attributes / regions sidecar. `classes`: a class list fixed by an earlier source of the same dataset (every
    split must share one id space). Raises ValueError when `out_dir` holds label files Fovea did not write."""
    img_dir, ann, out_dir = Path(img_dir), Path(ann), Path(out_dir)
    if out_dir.is_dir() and read_marker(out_dir) is None and any(
            n.lower().endswith(LABEL_EXT) for n in _names(out_dir, 5000)):
        raise ValueError(f"{out_dir} already holds label files — not overwriting them")
    reader = Reader(card, ann, img_dir, recursive)
    imgs = list_images(img_dir, recursive, skip=skip)
    have = [(rel, p) for rel, p in imgs if reader.has_annotation(rel)]
    if classes is not None and reader.from_data:
        reader.classes = list(classes)
        reader.index = {c: i for i, c in enumerate(reader.classes)}
        extra = Reader(card, ann, img_dir, recursive)
        extra.collect_classes([r for r, _p in have])
        for c in extra.classes:
            if c not in reader.index:
                reader.index[c] = len(reader.classes)
                reader.classes.append(c)
    else:
        reader.collect_classes([r for r, _p in have])
    out_dir.mkdir(parents=True, exist_ok=True)
    marker = {"format": card.get("id"), "name": card.get("name"), "source": card.get("source"),
              "annotations": str(ann), "images": str(img_dir), "created": time.time(), "state": "converting",
              "card": {k: v for k, v in card.items() if k != "check"}}
    _write_json(out_dir / MARKER, marker)        # an interrupted run is redone, never mistaken for someone's labels
    totals = {"images": len(imgs), "labels": 0, "boxes": 0, "regions": 0, "missing": len(imgs) - len(have),
              "bad": 0, "unknown": 0, "oob": 0, "tiny": 0, "dropped": 0, "unreadable": 0}
    side_tmp = out_dir / (SIDECAR + ".tmp")
    with open(side_tmp, "w", encoding="utf-8") as side:
        for k, (rel, p) in enumerate(have):
            if progress and k % 100 == 0:
                progress(k, len(have))
            dims = image_size(p)
            if not dims:
                totals["unreadable"] += 1
                continue
            r = reader.parse(rel, float(dims[0]), float(dims[1]))
            if r is None:
                continue
            target = out_dir / PurePosixPath(rel).with_suffix(LABEL_EXT)
            target.parent.mkdir(parents=True, exist_ok=True)
            text = "".join(row_text(row) + "\n" for row in r.rows)
            tmp = target.with_name(target.name + ".fovea-tmp")
            tmp.write_text(text, encoding="utf-8")
            os.replace(tmp, target)
            totals["labels"] += 1
            totals["boxes"] += len(r.rows)
            totals["regions"] += len(r.regions)
            for key in ("bad", "unknown", "oob", "tiny", "dropped"):
                totals[key] += getattr(r, key)
            rows = [[[round(v, 6) for v in row_bbox(row)], a] for row, a in zip(r.rows, r.attrs) if a]
            if rows or r.regions:
                side.write(json.dumps({"rel": str(PurePosixPath(rel).with_suffix("")), "rows": rows,
                                       "regions": r.regions}, separators=(",", ":")) + "\n")
    os.replace(side_tmp, out_dir / SIDECAR)
    marker.update({"state": "done", "finished": time.time(), "counts": totals, "classes": reader.classes})
    _write_json(out_dir / MARKER, marker)
    if progress:
        progress(len(have), len(have))
    return {**totals, "classes": reader.classes, "out_dir": str(out_dir)}


def _write_json(p: Path, obj: dict) -> None:
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(obj, ensure_ascii=False, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ------------------------------------------------------------------------------------------------- sidecar

_side_cache: Dict[str, Tuple[float, int, Dict[str, dict]]] = {}


def load_sidecar(label_dir: Path) -> Dict[str, dict]:
    """{rel without extension: {"rows": [[[xc, yc, w, h], attrs], …], "regions": [[kind, xc, yc, w, h], …]}} of a
    converted label folder (cached until the file changes)."""
    p = Path(label_dir) / SIDECAR
    try:
        st = p.stat()
    except OSError:
        return {}
    key = str(p)
    hit = _side_cache.get(key)
    if hit and hit[0] == st.st_mtime and hit[1] == st.st_size:
        return hit[2]
    out: Dict[str, dict] = {}
    with open(p, encoding="utf-8") as fh:
        for ln in fh:
            try:
                d = json.loads(ln)
            except ValueError:
                continue
            if isinstance(d, dict) and d.get("rel"):
                out[str(d["rel"])] = d
    if len(_side_cache) > 32:
        _side_cache.clear()
    _side_cache[key] = (st.st_mtime, st.st_size, out)
    return out


GEOM_TOL = 2e-6


def match_attrs(entry: Optional[dict], boxes: List[List[float]]) -> List[Optional[dict]]:
    """Attributes of each current box (cls, xc, yc, w, h): those of the converted box with the same geometry. A box
    that was moved or resized is a new annotation and has none; a re-classed box keeps them."""
    if not entry or not entry.get("rows"):
        return [None] * len(boxes)
    pool: Dict[Tuple[int, int], List[int]] = {}
    rows = entry["rows"]
    for j, (g, _a) in enumerate(rows):
        pool.setdefault((round(g[0] * 1e4), round(g[1] * 1e4)), []).append(j)
    used = set()
    out: List[Optional[dict]] = []
    for b in boxes:
        hit = None
        for j in pool.get((round(float(b[1]) * 1e4), round(float(b[2]) * 1e4)), []):
            g = rows[j][0]
            if j not in used and all(abs(float(b[i + 1]) - g[i]) <= GEOM_TOL for i in range(4)):
                hit = j
                break
        if hit is None:
            out.append(None)
        else:
            used.add(hit)
            out.append(rows[hit][1])
    return out
