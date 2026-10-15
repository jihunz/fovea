"""Dataset layout detection.

Supported layouts
-----------------
* ``yaml``        : Ultralytics data.yaml (path/train/val/test + names)
* ``ultralytics`` : root/images/<split>, root/labels/<split>
* ``split-first`` : root/<split>/images, root/<split>/labels
* ``flat``        : root/images, root/labels
* ``mixed``       : images and .txt labels side by side (also CVAT "YOLO 1.1" obj_<split>_data folders)
* ``bare``        : folder of images without labels (labels/ will be created on save)
* ``list``        : a train.txt style list of image paths (labels via images->labels rule)
* ``collection``  : a folder holding several datasets — nothing is registered; ``candidates`` lists them
* ``custom``      : image / label folders with other names (imgs/ + annotations/, train_images/ …) — found by an AI
                    provider reading the folder listing (the default, ai/folders.py), else by conventional names

Annotations in another format than YOLO (VisDrone, KITTI, DOTA, VOC, COCO, LabelMe, CVAT, or one an AI reads) are
attached to their source (``ann_path``, ``ann_format``) and described in ``formats`` (core/annformats.py); the scan
converts them into the source's label folder before indexing.
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

from ..config import IMAGE_EXTS, LABEL_EXT, SPLIT_NAMES
from ..paths import resolve, to_host
from . import annformats as af
from .labels import MAX_INFERRED_CLASSES

SPLIT_ALIASES = {"valid": "val", "validation": "val", "eval": "test", "evaluation": "test", "training": "train",
                 "testing": "test"}
_SPLIT_LIKE = set(SPLIT_NAMES) | set(SPLIT_ALIASES.keys()) | {"train", "val", "test"}

# The add-dataset preview counts images and labels; on a huge tree or a slow network share that walk is
# capped (counts are then lower bounds, flagged `count_capped`).
ESTIMATE_SECONDS = 4.0
_budget = threading.local()


def norm_split(name: str) -> str:
    n = name.strip().lower()
    return SPLIT_ALIASES.get(n, n)


_SPLIT_RANK = {"train": 0, "val": 1, "test": 2}
_DIGITS = re.compile(r"(\d+)")


def natural_key(name: str) -> tuple:
    """'fold2' before 'fold10'; case-insensitive, then exact (so the order is total)."""
    parts = _DIGITS.split((name or "").casefold())
    key = tuple((0, int(p), "") if p.isdecimal() else (1, 0, p) for p in parts if p != "") + ((-1, 0, ""),)
    return key, name or ""


def split_order_key(name: str) -> tuple:
    """train, val, test, then every other split in natural order, then the split-less group — the one order every
    table, list and chart uses."""
    return (_SPLIT_RANK.get(name, 3 if name else 9), natural_key(name))


def sort_sources(sources: List["Source"]) -> List["Source"]:
    """Stable: several folders of one split keep their relative order."""
    return sorted(sources, key=lambda s: split_order_key(s.split or ""))


def is_split_name(name: str) -> bool:
    return norm_split(name) in ("train", "val", "test") or name.lower() in _SPLIT_LIKE


def split_names(names: List[str]) -> Dict[str, str]:
    """Split names for sibling folders. A train / val / test word inside longer names is read as the split when every
    folder shares the words before it: 'VisDrone2019-DET-train', '…-val', '…-test-dev' → train, val, test-dev (Export
    maps a split by its leading word, so 'visdrone2019-det-val' used to feed training). Other names stay as they are."""
    toks = {n: [t for t in _TOKEN.split((n or "").lower()) if t] for n in names}
    cut: Optional[Dict[str, int]] = {}
    for n, t in toks.items():
        i = next((k for k, x in enumerate(t) if is_split_name(x)), None)
        if not i:                              # no split word, or the name starts with it: nothing to strip
            cut = None
            break
        cut[n] = i
    out = {n: norm_split(n) for n in names}
    if cut and len({tuple(toks[n][:cut[n]]) for n in names}) == 1:
        for n in names:
            rest = toks[n][cut[n]:]
            out[n] = "-".join([norm_split(rest[0])] + rest[1:])
    return out


@dataclass
class Source:
    split: str
    img_dir: str = ""            # absolute (container) path; empty for list sources
    label_dir: str = ""          # absolute; may not exist yet
    list_file: Optional[str] = None
    base: Optional[str] = None   # base for relative paths in list files
    label_dir_exists: bool = False
    image_count: Optional[int] = None
    label_count: Optional[int] = None
    recursive: bool = True       # False: only files directly in img_dir (loose images beside split folders)
    count_capped: bool = False   # the estimate walk hit its time budget: counts are lower bounds
    ann_path: str = ""           # annotations in another format (a folder, or one COCO / CVAT file)
    ann_format: str = ""         # their format card id (Layout.formats); converted into label_dir by the scan

    def to_public(self) -> dict:
        d = asdict(self)
        d["img_dir_host"] = to_host(self.img_dir) if self.img_dir else ""
        d["label_dir_host"] = to_host(self.label_dir) if self.label_dir else ""
        d["list_file_host"] = to_host(self.list_file) if self.list_file else None
        return d


@dataclass
class Layout:
    kind: str
    root: str
    sources: List[Source] = field(default_factory=list)
    data_yaml: Optional[str] = None
    classes: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)
    classes_from: str = ""                                    # 'yaml' | 'file' | '' (where `classes` came from)
    candidates: List[dict] = field(default_factory=list)      # kind 'collection': [{name, path_host, kind}]
    unmatched_labels: List[dict] = field(default_factory=list)  # label folders no source uses: [{path_host, files}]
    roles_from: str = ""                                      # kind 'custom': 'ai' | 'names' (how folders were found)
    roles_ai: Optional[dict] = None                           # {provider, model, reason} when an AI provider found them
    formats: Dict[str, dict] = field(default_factory=dict)    # card id -> {card, state: pending|converted, check, counts}

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "root": self.root, "root_host": to_host(self.root),
            "sources": [s.to_public() for s in self.sources],
            "data_yaml": self.data_yaml, "data_yaml_host": to_host(self.data_yaml) if self.data_yaml else None,
            "classes": self.classes, "notes": self.notes, "classes_from": self.classes_from,
            "candidates": self.candidates, "unmatched_labels": self.unmatched_labels,
            "roles_from": self.roles_from, "roles_ai": self.roles_ai, "formats": self.formats,
        }

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        srcs = []
        for s in d.get("sources", []):
            srcs.append(Source(
                split=s.get("split", ""), img_dir=s.get("img_dir", ""), label_dir=s.get("label_dir", ""),
                list_file=s.get("list_file"), base=s.get("base"), label_dir_exists=s.get("label_dir_exists", False),
                image_count=s.get("image_count"), label_count=s.get("label_count"),
                recursive=s.get("recursive", True), count_capped=s.get("count_capped", False),
                ann_path=s.get("ann_path", ""), ann_format=s.get("ann_format", ""),
            ))
        return Layout(kind=d.get("kind", "unknown"), root=d.get("root", ""), sources=srcs,
                      data_yaml=d.get("data_yaml"), classes=list(d.get("classes", [])), notes=list(d.get("notes", [])),
                      classes_from=d.get("classes_from", ""), candidates=list(d.get("candidates", [])),
                      unmatched_labels=list(d.get("unmatched_labels", [])), roles_from=d.get("roles_from", ""),
                      roles_ai=d.get("roles_ai"), formats=dict(d.get("formats") or {}))


# --------------------------------------------------------------------------- helpers

def is_image(p: os.DirEntry | Path) -> bool:
    name = p.name
    dot = name.rfind(".")
    return dot > 0 and name[dot:].lower() in IMAGE_EXTS


def has_images_shallow(d: Path) -> bool:
    try:
        with os.scandir(d) as it:
            for e in it:
                if e.is_file(follow_symlinks=True) and is_image(e):
                    return True
    except OSError:
        pass
    return False


def _subdirs(d: Path) -> List[Path]:
    try:
        return [c for c in sorted(d.iterdir()) if not c.name.startswith(".") and c.is_dir()]
    except OSError:
        return []


def _within(path: str, base: str) -> bool:
    return path == base or path.startswith(base.rstrip(os.sep) + os.sep)


_WIN_ABS = re.compile(r"^(?:[A-Za-z]:[\\/]|\\\\)")


def entry_path(s: str, base: Path) -> Path:
    """A path written in a data.yaml or a list file: absolute (POSIX, a Windows drive / UNC path, or a host
    path mapped into Docker) or relative to `base`. Backslashes count as separators in relative entries.
    Symlinks are NOT resolved — an image listed through a link keeps the link's path."""
    s = (s or "").strip().strip('"').strip("'")
    if s.startswith("~"):
        s = os.path.expanduser(s)
    if s.startswith("/") or _WIN_ABS.match(s):
        p = Path(s)
        if p.is_absolute() and p.exists():
            return Path(os.path.normpath(s))
        return resolve(s)               # HOST_PATH mapping and C:\ -> /c/… normalisation live there
    return Path(os.path.normpath(base / s.replace("\\", "/")))


def _over_budget() -> bool:
    deadline = getattr(_budget, "deadline", None)
    if deadline is not None and time.monotonic() > deadline:
        _budget.capped = True
        return True
    return False


def iter_files(top: Path, recursive: bool = True, onerror=None, peers=()):
    """Yield (dirpath, sorted visible file names) under `top`.

    Symlinked folders are followed — `images/train -> /data/train` is part of the user's dataset — except
    a link whose target lies inside, or above, `top` or any `peers` folder (the dataset's other sources):
    following those would index the same images twice under another path, or loop. Each real directory
    is still visited at most once. Non-recursive mode lists only `top` itself."""
    if not recursive:
        try:
            with os.scandir(top) as it:
                names = sorted(e.name for e in it if not e.name.startswith(".") and e.is_file(follow_symlinks=True))
        except OSError as e:
            if onerror:
                onerror(e)
            return
        yield str(top), names
        return
    guarded = {os.path.realpath(str(p)) for p in (*peers, top) if p}
    seen = set()
    for root, dirs, files in os.walk(top, onerror=onerror, followlinks=True):
        try:
            st = os.stat(root)
        except OSError as e:
            if onerror:
                onerror(e)
            dirs[:] = []
            continue
        if (st.st_dev, st.st_ino) in seen:
            dirs[:] = []
            continue
        seen.add((st.st_dev, st.st_ino))
        keep = []
        for d in sorted(x for x in dirs if not x.startswith(".")):
            full = os.path.join(root, d)
            if os.path.islink(full):
                real = os.path.realpath(full)
                if any(_within(real, g) or _within(g, real) for g in guarded):
                    continue
            keep.append(d)
        dirs[:] = keep
        yield root, sorted(f for f in files if not f.startswith("."))


def count_images(d: Path, limit: Optional[int] = None, recursive: bool = True) -> int:
    n = 0
    for _root, files in iter_files(d, recursive):
        for f in files:
            dot = f.rfind(".")
            if dot > 0 and f[dot:].lower() in IMAGE_EXTS:
                n += 1
                if limit and n >= limit:
                    return n
        if _over_budget():          # a huge tree (a home folder) must not stall detection
            return n
    return n


def count_labels(d: Path, recursive: bool = True) -> int:
    if not d.is_dir():
        return 0
    n = 0
    for _root, files in iter_files(d, recursive):
        n += sum(1 for f in files if f.lower().endswith(LABEL_EXT))
        if _over_budget():
            break
    return n


def derive_label_dir(img_dir: Path) -> Path:
    """Ultralytics rule: replace the *last* 'images' path component with 'labels'."""
    parts = list(img_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i].lower() == "images":
            parts[i] = "labels"
            return Path(*parts)
    return img_dir.parent / "labels"


def label_path_for_image(img_path: Path) -> Path:
    return derive_label_dir(img_path.parent) / (img_path.stem + LABEL_EXT)


def _read_text(p: Path) -> str:
    return p.read_text(encoding="utf-8-sig")


def find_yaml(d: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = d / name
        if p.is_file():
            return p
    try:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".yaml", ".yml") and p.is_file():
                try:
                    doc = yaml.safe_load(_read_text(p)) or {}
                except Exception:
                    continue
                if isinstance(doc, dict) and ("names" in doc or "train" in doc):
                    return p
    except OSError:
        pass
    return None


def parse_names(doc: dict, notes: Optional[List[str]] = None) -> List[str]:
    """Class names of a data.yaml. The gaps of a sparse {id: name} map and a bare `nc` are filled with made-up
    class_<id> names, never past MAX_INFERRED_CLASSES of them (a key like 2000000000 used to build two billion
    names): a larger id is left out — its rows are flagged bad_class — and said in `notes`."""
    names = doc.get("names")
    if isinstance(names, dict):
        out: List[str] = []
        try:
            keys = sorted(int(k) for k in names.keys())
        except Exception:
            return [str(v) for v in names.values()]
        limit = max(MAX_INFERRED_CLASSES, len(keys))
        kept = [k for k in keys if k < limit]
        if notes is not None and len(kept) < len(keys):
            notes.append(f"names: class id {keys[-1]:,} is out of range — ids from {limit:,} up were left out "
                         "(their boxes show as Invalid class id)")
        for k in range(kept[-1] + 1 if kept else 0):
            out.append(str(names.get(k, names.get(str(k), f"class_{k}"))))
        return out
    if isinstance(names, list):
        return [str(n) for n in names]
    nc = doc.get("nc")
    if isinstance(nc, int) and not isinstance(nc, bool) and nc > 0:
        if notes is not None and nc > MAX_INFERRED_CLASSES:
            notes.append(f"nc: {nc:,} classes without names — only the first {MAX_INFERRED_CLASSES:,} were named class_0…")
        return [f"class_{i}" for i in range(min(nc, MAX_INFERRED_CLASSES))]
    return []


def read_classes_file(d: Path) -> List[str]:
    for name in ("classes.txt", "obj.names", "labels.txt"):
        p = d / name
        if p.is_file():
            try:
                lines = [ln.strip() for ln in _read_text(p).splitlines()]
                lines = [ln for ln in lines if ln]
                if lines and all(len(ln) < 64 for ln in lines):
                    return lines
            except Exception:
                pass
    return []


def _set_classes_from_file(lay: Layout, d: Path) -> Layout:
    cls = read_classes_file(d)
    if cls:
        lay.classes, lay.classes_from = cls, "file"
    return lay


def _mk_dir_source(split: str, img_dir: Path, label_dir: Optional[Path] = None, estimate: bool = True,
                   recursive: bool = True) -> Source:
    label_dir = label_dir or derive_label_dir(img_dir)
    src = Source(split=split, img_dir=str(img_dir), label_dir=str(label_dir), label_dir_exists=label_dir.is_dir(),
                 recursive=recursive)
    if estimate:
        _budget.capped = False
        src.image_count = count_images(img_dir, recursive=recursive)
        src.label_count = count_labels(label_dir, recursive=recursive)
        src.count_capped = bool(getattr(_budget, "capped", False))
    return src


def _mk_list_source(split: str, list_file: Path, base: Optional[Path] = None, estimate: bool = True) -> Source:
    src = Source(split=split, list_file=str(list_file), base=str(base or list_file.parent), label_dir_exists=True)
    if estimate:
        try:
            with list_file.open("r", encoding="utf-8", errors="ignore") as fh:
                src.image_count = sum(1 for ln in fh if ln.strip() and not ln.strip().startswith("#"))
        except OSError:
            src.image_count = None
    return src


# --------------------------------------------------------------------------- detectors

def from_yaml(yaml_path: Path, root: Optional[Path] = None, estimate: bool = True) -> Layout:
    doc = yaml.safe_load(_read_text(yaml_path)) or {}
    if not isinstance(doc, dict):
        raise ValueError("YAML root must be a mapping")
    ydir = yaml_path.parent
    base = ydir
    notes: List[str] = []
    if doc.get("path"):
        # A relative `path:` is relative to the yaml file (never to the server's working directory). An
        # absolute one from another machine falls back to the yaml folder, like Ultralytics does.
        pf = str(doc["path"])
        cand = entry_path(pf, ydir)
        if cand.is_dir():
            base = cand.resolve()
        else:
            notes.append(f"path: {pf} not found — paths are read relative to the yaml folder")
    layout = Layout(kind="yaml", root=str(root or base), data_yaml=str(yaml_path), classes=parse_names(doc, notes),
                    notes=notes)
    if layout.classes:
        layout.classes_from = "yaml"
    for split in ("train", "val", "test"):
        v = doc.get(split)
        if not v:
            continue
        entries = v if isinstance(v, list) else [v]
        for entry in entries:
            e = str(entry).strip()
            tries = [entry_path(e, base), entry_path(e, ydir)]
            norm = e.replace("\\", "/")
            if norm.startswith("../"):
                # Roboflow writes 'train: ../train/images'; Ultralytics retries without the '../'.
                tries += [entry_path(norm[3:], base), entry_path(norm[3:], ydir)]
            cand = next((t for t in tries if t.exists()), tries[0])
            if cand.is_dir():
                layout.sources.append(_mk_dir_source(split, cand.resolve(), estimate=estimate))
            elif cand.is_file() and cand.suffix.lower() == ".txt":
                layout.sources.append(_mk_list_source(split, cand, base, estimate=estimate))
            else:
                layout.notes.append(f"{split}: path not found ({entry})")
    if not layout.sources:
        layout.notes.append(f"{yaml_path.name} has no resolvable train/val/test entries")
    return layout


def from_list(txt: Path, split: str = "", estimate: bool = True) -> Layout:
    split = split or (norm_split(txt.stem) if norm_split(txt.stem) in ("train", "val", "test") else "")
    lay = Layout(kind="list", root=str(txt.parent))
    lay.sources.append(_mk_list_source(split, txt, estimate=estimate))
    return _set_classes_from_file(lay, txt.parent)


def _sidecar_labels(top: Path, max_dirs: int = 64) -> bool:
    """Do images anywhere below `top` (sampled) have a same-name .txt next to them (mixed layout)?"""
    for k, (_root, files) in enumerate(iter_files(top)):
        if k >= max_dirs:
            break
        stems, txts = set(), set()
        for f in files:
            dot = f.rfind(".")
            if dot <= 0:
                continue
            ext = f[dot:].lower()
            if ext in IMAGE_EXTS:
                stems.add(f[:dot])
            elif ext == LABEL_EXT:
                txts.add(f[:dot])
        if stems & txts:
            return True
    return False


def _stray_txt(top: Path, skip: Optional[Path] = None, max_dirs: int = 400) -> int:
    """.txt files somewhere under `top` (sampled), outside `skip`."""
    n = 0
    for k, (root, files) in enumerate(iter_files(top)):
        if k >= max_dirs or _over_budget():
            break
        if skip is not None and _within(root, str(skip)):
            continue
        n += sum(1 for f in files if f.lower().endswith(LABEL_EXT) and f.lower() not in ("classes.txt", "labels.txt"))
    return n


def _child_kind(c: Path) -> Optional[str]:
    """How a sub-folder looks: 'root' (a dataset of its own), 'split' (<split>/images of a split-first
    dataset), 'images' (a folder of images) or None."""
    if find_yaml(c):
        return "root"
    if (c / "images").is_dir():
        if any(is_split_name(d.name) for d in _subdirs(c / "images")):
            return "root"
        return "split"
    if any((g / "images").is_dir() for g in _subdirs(c)):
        return "root"
    if has_images_shallow(c):
        return "images"
    return None


def _collection(p: Path, kids: List[tuple]) -> Layout:
    lay = Layout(kind="collection", root=str(p))
    lay.candidates = [{"name": c.name, "path_host": to_host(c), "kind": k} for c, k in kids]
    names = ", ".join(c.name for c, _k in kids[:8]) + (" …" if len(kids) > 8 else "")
    lay.notes.append(f"This folder holds {len(kids)} datasets ({names}). Add each one separately.")
    return lay


def _cvat(p: Path, estimate: bool) -> Optional[Layout]:
    """CVAT "YOLO 1.1" export: obj.names / obj.data and obj_<split>_data/ folders with sidecar labels."""
    if not ((p / "obj.names").is_file() or (p / "obj.data").is_file()):
        return None
    dirs = [d for d in _subdirs(p) if re.fullmatch(r"obj_(\w+?)_data", d.name)]
    if not dirs:
        return None
    lay = Layout(kind="mixed", root=str(p))
    for d in dirs:
        split = norm_split(re.fullmatch(r"obj_(\w+?)_data", d.name).group(1))
        lay.sources.append(_mk_dir_source(split, d, d, estimate=estimate))
    return _set_classes_from_file(lay, p)


def _unmatched_label_dirs(lay: Layout) -> None:
    """Label folders that no source reads from (e.g. labels/test-fall0class with 1,008 files but no
    images/test-fall0class): surface them instead of ignoring them silently."""
    if not lay.sources or lay.kind in ("list", "collection"):
        return
    used = {os.path.normpath(s.label_dir) for s in lay.sources if s.label_dir}
    root = Path(lay.root)
    for top in [root / "labels"] + [d / "labels" for d in _subdirs(root)]:
        if not top.is_dir():
            continue
        nt = os.path.normpath(str(top))
        if not any(_within(nt, u) for u in used):
            n = count_labels(top, recursive=False)       # .txt files directly in labels/
            if n:
                lay.unmatched_labels.append({"path_host": to_host(top), "files": n})
        for d in _subdirs(top):
            nd = os.path.normpath(str(d))
            if any(_within(nd, u) or _within(u, nd) for u in used):
                continue
            n = count_labels(d)
            if n:
                lay.unmatched_labels.append({"path_host": to_host(d), "files": n})
    for u in lay.unmatched_labels[:5]:
        rel = u["path_host"]
        try:
            rel = Path(u["path_host"]).relative_to(to_host(root)).as_posix()
        except ValueError:
            pass
        lay.notes.append(f"{rel}: {u['files']:,} {UNMATCHED_NOTE}")


UNMATCHED_NOTE = "label file(s) with no matching images folder — not indexed"


def find_unmatched_labels(lay: Layout, seconds: float = 5.0) -> List[dict]:
    """Labels-only folders of an already registered layout, re-checked now (a rescan reports them, and drops a
    folder whose images were added since). Bounded by `seconds` like the add-dialog estimate."""
    probe = Layout(kind=lay.kind, root=lay.root, sources=list(lay.sources))
    _budget.deadline, _budget.capped = time.monotonic() + seconds, False
    try:
        _unmatched_label_dirs(probe)
    finally:
        _budget.deadline = None
    return probe.unmatched_labels


# --------------------------------------------------------------------------- folder roles (other folder names)
# A tree that does not call its folders images/ and labels/ (imgs/ + annotations/, train_images/ + train_labels/,
# <split>/JPEGImages …) is mapped by role: an AI provider reads the folder listing (ai/folders.py, the default), and
# when none is available or its answer does not check out, these conventional names are matched instead.

IMAGE_DIR_NAMES = {"images", "image", "imgs", "img", "jpegimages", "jpeg", "jpg", "jpgs", "png", "pngs", "pictures",
                   "picture", "pics", "pic", "photos", "photo", "frames", "frame", "rgb"}
LABEL_DIR_NAMES = {"labels", "label", "annotations", "annotation", "annots", "annot", "anns", "ann", "yolo",
                   "yololabels", "labelsyolo", "yolotxt", "txt", "gt", "groundtruth", "bboxes", "bbox", "boxes",
                   "targets", "target"}
_TOKEN = re.compile(r"[^a-z0-9]+")
INVENTORY_MAX_DIRS = 400
INVENTORY_MAX_DEPTH = 4


def dir_role(name: str) -> tuple:
    """('image' | 'label' | None, split) of a folder name: 'imgs' → image, 'train_labels' → (label, 'train'),
    'JPEGImages' → image. A name with both words is what its last one says ('image_labels' → label)."""
    toks = [t for t in _TOKEN.split((name or "").lower()) if t]
    key = "".join(toks)
    split = next((norm_split(t) for t in toks if is_split_name(t)), "")
    if key in IMAGE_DIR_NAMES:
        return "image", split
    if key in LABEL_DIR_NAMES:
        return "label", split
    for t in reversed(toks):
        if t in LABEL_DIR_NAMES:
            return "label", split
        if t in IMAGE_DIR_NAMES:
            return "image", split
    return None, split


@dataclass
class DirInfo:
    rel: str                     # posix path below the root, '.' for the root itself
    path: Path
    depth: int
    images: int = 0              # directly inside
    txt: int = 0                 # label-like .txt directly inside
    other: dict = field(default_factory=dict)   # {'.xml': n, '.json': n} — other annotation formats
    sample: List[str] = field(default_factory=list)
    tree_images: int = 0         # in this folder and below
    tree_txt: int = 0
    img_stems: set = field(default_factory=set)  # a few image / label file stems, to tell which labels belong where
    txt_stems: set = field(default_factory=set)
    ann_stems: set = field(default_factory=set)  # stems of .txt / .xml / .json directly inside
    txt_kind: str = ""           # what the .txt directly inside hold: 'yolo' | 'other' (VisDrone, KITTI …) | 'empty'
    first_line: str = ""
    tree_other: int = 0          # folders at or below this one whose .txt are not YOLO
    tree_ann: int = 0            # annotation files of another format at or below (.txt that are not YOLO, .xml, .json)

    @property
    def yolo_labels(self) -> bool:
        """Holds YOLO label files (at or below it) and nothing that reads as another format."""
        return self.tree_txt > 0 and not self.tree_other


def folder_inventory(root: Path, max_dirs: int = INVENTORY_MAX_DIRS, max_depth: int = INVENTORY_MAX_DEPTH,
                     seconds: float = 3.0) -> List[DirInfo]:
    """The folders under `root` (hidden ones skipped, symlinks not followed, bounded in count, depth and time) with
    what each holds — what an AI provider is shown, and what the name fallback reads."""
    out: List[DirInfo] = []
    by_path = {}
    seen = set()
    deadline = time.monotonic() + seconds
    for cur, dirs, files in os.walk(root, followlinks=True):   # linked split folders are the user's data
        try:
            st = os.stat(cur)
        except OSError:
            dirs[:] = []
            continue
        if (st.st_dev, st.st_ino) in seen:                       # a link back to a folder already listed: a loop
            dirs[:] = []
            continue
        seen.add((st.st_dev, st.st_ino))
        rel = os.path.relpath(cur, root)
        depth = 0 if rel == "." else rel.count(os.sep) + 1
        dirs[:] = sorted(d for d in dirs if not d.startswith(".") and d != "__pycache__") if depth < max_depth else []
        d = DirInfo(rel=Path(rel).as_posix(), path=Path(cur), depth=depth)
        for f in files:
            if f.startswith("."):
                continue
            dot = f.rfind(".")
            ext = f[dot:].lower() if dot > 0 else ""
            if ext in IMAGE_EXTS:
                d.images += 1
            elif ext == LABEL_EXT and f.lower() not in af.NOT_LABELS:
                d.txt += 1
            elif ext in (".xml", ".json"):
                d.other[ext] = d.other.get(ext, 0) + 1
            else:
                continue
            if len(d.sample) < 3:
                d.sample.append(f)
            if ext in IMAGE_EXTS and len(d.img_stems) < 200:
                d.img_stems.add(f[:dot])
            elif ext == LABEL_EXT and len(d.txt_stems) < 200:
                d.txt_stems.add(f[:dot])
            if ext in af.ANN_EXTS and len(d.ann_stems) < 200:
                d.ann_stems.add(f[:dot])
        if d.txt:
            lines: List[str] = []
            for f in sorted(x for x in files if x.lower().endswith(LABEL_EXT) and not x.startswith(".")
                            and x.lower() not in af.NOT_LABELS)[:3]:
                lines += af.head_lines(Path(cur) / f, 6)
            d.txt_kind = af.txt_kind(lines)
            d.first_line = lines[0].strip()[:120] if lines else ""
        out.append(d)
        by_path[cur] = d
        if len(out) >= max_dirs or time.monotonic() > deadline:
            break
    for d in out:                                    # roll the counts up to every ancestor that was listed
        up = d.path
        while True:
            a = by_path.get(str(up))
            if a is not None:
                a.tree_images += d.images
                a.tree_txt += d.txt
                a.tree_other += 1 if d.txt_kind == "other" else 0
                a.tree_ann += (d.txt if d.txt_kind == "other" else 0) + sum(d.other.values())
            if str(up) == str(root) or up.parent == up:
                break
            up = up.parent
    return out


def inventory_lines(inv: List[DirInfo], limit: int = 200) -> List[str]:
    """The listing sent to an AI provider: only folders that hold (or lead to) images or annotation files."""
    lines = []
    for d in inv:
        if not (d.tree_images or d.tree_txt or d.other):
            continue
        extra = "".join(f" {k[1:]}={v}" for k, v in sorted(d.other.items()))
        eg = f"  e.g. {', '.join(d.sample)}" if d.sample else ""
        first = f"  first .txt line: {d.first_line!r} ({'YOLO' if d.txt_kind == 'yolo' else 'not YOLO'})" if d.first_line else ""
        lines.append(f"{d.rel}  images={d.images} txt={d.txt}{extra}  (below: images={d.tree_images} txt={d.tree_txt}){eg}"
                     f"{first}")
        if len(lines) >= limit:
            break
    return lines


def default_label_dir(img_dir: Path, root: Path) -> Path:
    """Where labels of an image folder that has none yet are written: its image-named component swapped for
    'labels' (imgs/train → labels/train), else a labels/ folder inside it (as for a bare folder)."""
    try:
        rel = img_dir.relative_to(root).parts
    except ValueError:
        rel = ()
    for i in range(len(rel) - 1, -1, -1):
        if dir_role(rel[i])[0] == "image":
            parts = list(rel)
            parts[i] = "labels"
            return root.joinpath(*parts)
    return img_dir / "labels"


def _overlap(a: set, b: set) -> int:
    return len(a & b)


def _best_label_dir(img: DirInfo, cands: List[DirInfo], split: str = "") -> Optional[DirInfo]:
    """The label-named folder of YOLO .txt files that best fits `img`: same split word, shared file stems, most files.
    A folder of .txt in another format (VisDrone's annotations/) is not a label folder — see _best_ann_dir."""
    ok = [c for c in cands if c.yolo_labels]
    if not ok:
        return None
    return max(ok, key=lambda c: (dir_role(c.path.name)[1] == split and bool(split),
                                  _overlap(c.txt_stems, img.img_stems), c.tree_txt))


def _best_ann_dir(img: DirInfo, cands: List[DirInfo], split: str = "") -> Optional[DirInfo]:
    """The sibling folder holding the images' annotations in another format (.txt that are not YOLO, .xml, .json)."""
    ok = [c for c in cands if c.tree_ann > 0 and c is not img]
    if not ok:
        return None
    best = max(ok, key=lambda c: (dir_role(c.path.name)[1] == split and bool(split), _overlap(c.ann_stems, img.img_stems),
                                  dir_role(c.path.name)[0] == "label", c.tree_ann))
    return best if (_overlap(best.ann_stems, img.img_stems) or not best.ann_stems) else None


def roles_by_names(root: Path, inv: List[DirInfo]) -> List[tuple]:
    """[(split, img_dir, label_dir, recursive, annotations)] from conventional folder names, or [] when no folder is
    image-named. `annotations`: a folder of another format to convert into label_dir (None when there is none)."""
    by_rel = {d.rel: d for d in inv}

    def kids(d: DirInfo) -> List[DirInfo]:
        pre = "" if d.rel == "." else d.rel + "/"
        return [c for c in inv if c.depth == d.depth + 1 and c.rel.startswith(pre)]

    imgs = [d for d in inv if d.rel != "." and d.tree_images > 0 and dir_role(d.path.name)[0] == "image"]
    imgs = [d for d in imgs if not any(o is not d and d.rel.startswith(o.rel + "/") for o in imgs)]   # topmost only
    out = []
    for I in imgs:
        parent = by_rel.get(Path(I.rel).parent.as_posix()) or by_rel.get(".")
        sib_labels = [c for c in kids(parent) if dir_role(c.path.name)[0] == "label"] if parent else []
        sib_ann = [c for c in kids(parent) if c is not I and c.tree_ann > 0] if parent else []
        role_split = dir_role(I.path.name)[1]
        split_kids = [c for c in kids(I) if is_split_name(c.path.name) and c.tree_images > 0]
        if split_kids:
            L = _best_label_dir(I, sib_labels)
            A = None if L else _best_ann_dir(I, sib_ann)
            for S in split_kids:
                lab = by_rel.get(f"{L.rel}/{S.path.name}") if L else None
                ann = by_rel.get(f"{A.rel}/{S.path.name}") if A else None
                out.append((norm_split(S.path.name), S.path, lab.path if lab else (L.path / S.path.name if L else
                            default_label_dir(S.path, root)), True, ann.path if ann else None))
            if I.images:
                out.append(("", I.path, L.path if L else default_label_dir(I.path, root), False, A.path if A else None))
            continue
        if parent and parent.rel != ".":
            split = norm_split(parent.path.name) if is_split_name(parent.path.name) else (role_split or parent.path.name)
        else:
            split = role_split
        if I.txt and I.txt_kind != "other" and _overlap(I.img_stems, I.txt_stems):
            out.append((split, I.path, I.path, True, None))                 # labels beside the images
            continue
        L = _best_label_dir(I, sib_labels, role_split)
        A = None
        if L is None:
            A = _best_ann_dir(I, sib_ann, role_split)
            if A is None and I.tree_ann and _overlap(I.img_stems, I.ann_stems):
                A = I                                                        # LabelMe .json beside the images
        out.append((split, I.path, L.path if L else default_label_dir(I.path, root), True, A.path if A else None))
    names = split_names(sorted({t[0] for t in out}))
    return [(names.get(t[0], t[0]),) + tuple(t[1:]) for t in out]


def layout_from_roles(root: Path, pairs: List[tuple], estimate: bool, how: str) -> Layout:
    lay = Layout(kind="custom", root=str(root), roles_from=how)
    for split, img, lab, recursive, *rest in pairs:
        src = _mk_dir_source(split, Path(img), Path(lab), estimate=estimate, recursive=recursive)
        src.ann_path = str(rest[0]) if rest and rest[0] else ""
        lay.sources.append(src)

    def rel(p: str) -> str:
        try:
            r = Path(p).relative_to(root).as_posix()
        except ValueError:
            return p
        return (r if r != "." else Path(p).name) + "/"
    shown = ", ".join(f"{rel(s.img_dir)} → {rel(s.label_dir)}{'' if s.label_dir_exists else ' (new)'}"
                      for s in lay.sources[:4]) + (" …" if len(lay.sources) > 4 else "")
    lay.notes.append(f"Folders found {'by AI' if how == 'ai' else 'by their names'}: {shown}")
    return lay


def _needs_roles(lay: Layout) -> str:
    """Did the images/ + labels/ rules miss this tree? 'rebuild': nothing found, or images below the folder (in imgs/,
    photos/train …) indexed as one split-less pile. 'labels': the image folders are right but their labels/ is missing
    while .txt files sit elsewhere (images/ + annotations/) — only the label folders are re-pointed. '': neither."""
    if lay.kind in ("yaml", "list", "collection", "mixed", "custom"):
        return ""
    if lay.kind == "unknown":
        return "rebuild"
    srcs = [s for s in lay.sources if not s.list_file]
    if not srcs:
        return ""
    root = Path(lay.root)
    if lay.kind == "bare" and not has_images_shallow(root):
        return "rebuild"
    if any(not s.label_dir_exists for s in srcs) and _stray_txt(root, max_dirs=200) > 0:
        return "rebuild" if lay.kind == "bare" else "labels"
    return ""


def _path_key(p: str) -> str:
    """A folder's identity: its real path (on a case-insensitive disk images/ and Images/ are one folder)."""
    try:
        return os.path.normcase(os.path.realpath(p)).lower() if os.path.exists(p) else os.path.normpath(p)
    except OSError:
        return os.path.normpath(p)


def _repoint_labels(lay: Layout, pairs: List[tuple]) -> int:
    """Give sources whose label folder is missing the existing YOLO label folder `pairs` name for the same images
    (or for their parent, as <labels>/<split>), or else the folder of another format their annotations are in. The
    image sources themselves — linked splits, loose images — stay. Returns how many label folders were re-pointed."""
    by_img = {_path_key(str(t[1])): (Path(t[2]), t[4] if len(t) > 4 else None) for t in pairs}
    changed = 0
    for s in lay.sources:
        if s.list_file or s.label_dir_exists or not s.img_dir:
            continue
        img = os.path.normpath(s.img_dir)
        lab, ann = by_img.get(_path_key(img), (None, None))
        if lab is None and _path_key(os.path.dirname(img)) in by_img:   # images/ paired as a whole: labels per split
            lab, ann = by_img[_path_key(os.path.dirname(img))]
            lab, ann = lab / os.path.basename(img), (Path(ann) / os.path.basename(img) if ann else None)
        if ann and Path(ann).exists() and not s.ann_path:
            s.ann_path = str(ann)
        if lab is None or not lab.is_dir() or os.path.normpath(str(lab)) == img \
                or af.tree_txt_kind(lab) == "other":
            continue
        s.label_dir, s.label_dir_exists = str(lab), True
        s.label_count = count_labels(lab, recursive=s.recursive) if s.image_count is not None else None
        changed += 1
    return changed


def _sibling_labels(lay: Layout) -> None:
    """The image folder itself was picked (…/imgs) and its labels sit in a label-named sibling (…/annotations)."""
    for s in lay.sources:
        if s.list_file or s.label_dir_exists or not s.img_dir:
            continue
        img = Path(s.img_dir)
        me = DirInfo(rel=img.name, path=img, depth=1)
        try:
            with os.scandir(img) as it:
                me.img_stems = {e.name.rsplit(".", 1)[0] for _, e in zip(range(200), it) if e.is_file() and is_image(e)}
        except OSError:
            continue
        cands = []
        for c in _subdirs(img.parent):
            if c == img or dir_role(c.name)[0] != "label":
                continue
            d = DirInfo(rel=c.name, path=c, depth=1)
            try:
                with os.scandir(c) as it:
                    names = [e.name for _, e in zip(range(2000), it) if e.name.lower().endswith(LABEL_EXT)]
            except OSError:
                continue
            d.tree_txt, d.txt_stems = len(names), {n[:-len(LABEL_EXT)] for n in names[:200]}
            cands.append(d)
        best = _best_label_dir(me, cands, dir_role(img.name)[1])
        if best is not None and _overlap(best.txt_stems, me.img_stems) and af.tree_txt_kind(best.path) != "other":
            s.label_dir, s.label_dir_exists = str(best.path), True
            s.label_count = count_labels(best.path, recursive=s.recursive) if s.image_count is not None else None
            lay.kind = "flat"
            lay.notes = [n for n in lay.notes if "No labels found" not in n]
            lay.notes.append(f"Labels found in the sibling folder {best.path.name}/")


def resolve_roles(lay: Layout, estimate: bool, ai: Optional[bool], allow_guarded: bool = True) -> Layout:
    """`lay` re-mapped by folder role when the standard rules missed the tree (see _needs_roles); else `lay`.
    `ai`: None = the Settings choice (AI by default), False = conventional names only."""
    if lay.kind == "bare" and has_images_shallow(Path(lay.root)):
        _sibling_labels(lay)
    need = _needs_roles(lay)
    if not need:
        return lay
    root = Path(lay.root)
    inv = folder_inventory(root)
    pairs, how, info, why = [], "", None, ""
    if ai is not False:
        from ..ai import folders as ai_folders            # lazy: the core layer does not load the AI stack up front
        try:
            res = ai_folders.find_roles(root, inv, use_setting=ai is None, allow_guarded=allow_guarded)
        except Exception as e:  # noqa: BLE001 — never let a provider failure break detection; names take over
            res, why = None, str(e)
        if res and res.get("pairs"):
            pairs, how, info = res["pairs"], "ai", {k: res.get(k) for k in ("provider", "model", "reason")}
        elif res:
            why = res.get("why") or ""
    if not pairs:
        pairs = roles_by_names(root, inv)
        how = "names" if pairs else ""
    if need == "labels":
        if pairs and _repoint_labels(lay, pairs):
            lay.roles_from, lay.roles_ai = how, info
            shown = ", ".join(sorted({Path(s.label_dir).relative_to(root).as_posix() + "/" for s in lay.sources
                                      if s.label_dir_exists and _within(s.label_dir, str(root))}))
            lay.notes.append(f"Labels found {'by AI' if how == 'ai' else 'by folder name'} in {shown}")
        elif pairs and any(s.ann_path for s in lay.sources):
            lay.roles_from, lay.roles_ai = how, info
        return lay
    if not pairs:
        if why:
            lay.notes.append(f"AI folder detection: {why}")
        return lay
    out = layout_from_roles(root, pairs, estimate, how)
    out.roles_ai = info
    if how == "names" and why:
        out.notes.append(f"AI folder detection did not answer ({why}) — matched by folder names instead")
    out.classes, out.classes_from = lay.classes, lay.classes_from
    out.data_yaml = lay.data_yaml
    out.notes += [n for n in lay.notes if "No labels found" not in n and "were not used as labels" not in n
                  and "No images" not in n]
    if not any(s.label_dir_exists for s in out.sources):
        out.notes.append("No labels found — label files will be created when you annotate")
    return out


# --------------------------------------------------------------------------- annotation formats

def _image_names(d: Path, limit: int = 300) -> List[str]:
    out: List[str] = []
    for cur, dirs, names in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith("."))
        for n in sorted(names):
            dot = n.rfind(".")
            if dot > 0 and n[dot:].lower() in IMAGE_EXTS and not n.startswith("."):
                out.append(n)
                if len(out) >= limit:
                    return out
    return out


def _ann_stems(d: Path, exts, max_entries: int = 6000, recursive: bool = True) -> set:
    out: set = set()
    seen = 0
    for cur, dirs, names in os.walk(d):
        dirs[:] = sorted(x for x in dirs if not x.startswith(".")) if recursive else []
        for n in names:
            seen += 1
            low = n.lower()
            dot = low.rfind(".")
            if dot > 0 and low[dot:] in exts and not n.startswith(".") and low not in af.NOT_LABELS:
                out.add(n[:dot])
        if seen > max_entries:
            break
    return out


def _inside_root(p: Path, root: Path) -> bool:
    return _within(os.path.normpath(str(p)), os.path.normpath(str(root)))


def find_annotations(src: Source, root: Path) -> Optional[Path]:
    """Where this image folder's annotations in another format are: .xml / .json beside the images (LabelMe, VOC), a
    folder whose .txt / .xml / .json files carry the images' names — next to the image folder inside the dataset
    (VisDrone annotations/, VOC Annotations/), or a label-named sibling when the image folder itself was picked — or
    one COCO / CVAT file that names the images, beside the folder, in an annotations/ folder or at the dataset root.
    Folders of other datasets next to this one are never read."""
    img = Path(src.img_dir)
    names = _image_names(img)
    if not names:
        return None
    stems = {n.rsplit(".", 1)[0] for n in names}
    label = os.path.normpath(src.label_dir) if src.label_dir else ""
    at_root = os.path.normpath(str(img)) == os.path.normpath(str(root))
    if at_root:
        cands = _subdirs(img) + [x for x in _subdirs(img.parent) if x != img and dir_role(x.name)[0] == "label"]
    else:
        cands = [x for x in _subdirs(img.parent) if x != img and _inside_root(x, root)]
    cands = [c for c in cands if os.path.normpath(str(c)) != label]
    best, score = None, 0.0
    for c in [img] + cands:
        found = _ann_stems(c, (".xml", ".json") if c == img else af.ANN_EXTS, recursive=c != img)
        if not found:
            continue
        r = len(stems & found) / len(stems) + (0.05 if dir_role(c.name)[0] == "label" else 0.0)
        if r >= 0.3 and r > score:
            best, score = c, r
    if best is not None:
        return best
    return find_single_file(img, root, names[:40])


def find_single_file(img: Path, root: Path, probe: List[str], min_share: float = 0.3) -> Optional[Path]:
    """One COCO / CVAT file that lists most of `probe` (image file names), near the image folder."""
    best, score = None, 0.0
    for base in dict.fromkeys([img.parent, img.parent / "annotations", root, root / "annotations", img]):
        if not base.is_dir():
            continue
        files = [f for f in sorted(base.iterdir())[:400] if f.suffix.lower() in (".json", ".xml") and f.is_file()]
        for f in files[:30]:
            card, _s, how = af.recognize(f)
            if how != "catalog" or not card or not card.get("single"):
                continue
            try:
                if f.stat().st_size > 300 * 1024 * 1024:
                    return f                      # too big to search here; the conversion matches image names
                text = f.read_bytes().decode("utf-8", errors="replace")
            except OSError:
                continue
            hits = sum(1 for n in probe if n in text)
            if hits >= min_share * len(probe) and hits / len(probe) > score:
                best, score = f, hits / len(probe)
    return best


_SPLIT_PREFIX = re.compile(r"^(train|training|val|valid|validation|test|testing)(\d{2,4})?$")


def split_of_folder(name: str) -> str:
    """'train2017' → train, 'val2017' → val, 'images_test' → test; else the folder name."""
    for t in _TOKEN.split((name or "").lower()):
        m = _SPLIT_PREFIX.match(t)
        if m:
            return norm_split(m.group(1))
    return norm_split(name)


def _per_folder_files(lay: Layout, root: Path) -> bool:
    """A folder of image folders with one COCO / CVAT file per folder (coco/train2017 + val2017 with
    annotations/instances_train2017.json …): one source per image folder, each with its own file."""
    if len(lay.sources) != 1 or lay.sources[0].list_file or not lay.sources[0].img_dir:
        return False
    if os.path.normpath(lay.sources[0].img_dir) != os.path.normpath(str(root)) or has_images_shallow(root):
        return False
    subs = [d for d in _subdirs(root) if has_images_shallow(d)]
    if not subs:
        return False
    found = []
    for d in subs:
        f = find_single_file(d, root, _image_names(d, 40), min_share=0.5)
        if f is not None:
            found.append((d, f))
    if not found:
        return False
    estimate = lay.sources[0].image_count is not None
    lay.sources = []
    for d, f in found:
        src = _mk_dir_source(split_of_folder(d.name), d, root / "labels" / d.name, estimate=estimate)
        src.ann_path = str(f)
        lay.sources.append(src)
    lay.kind = "custom"
    lay.notes = [n for n in lay.notes if "No labels found" not in n and "were not used as labels" not in n]
    lay.notes.append("Image folders " + ", ".join(d.name for d, _f in found[:6]) + " each have their own "
                     "annotation file")
    return True


def _small_check(st: dict) -> dict:
    keep = ("images", "annotated", "checked", "entries", "boxes", "regions", "bad", "unknown", "oob", "tiny",
            "dropped", "ok", "problems", "classes", "class_counts", "attr_values", "deferred")
    out = {k: st[k] for k in keep if k in st}
    out["unknown_values"] = dict(list((st.get("unknown_values") or {}).items())[:10])
    return out


def _format_for(src: Source, ann: Path, root: Path, ai: Optional[bool], allow_guarded: bool,
                known: Dict[str, dict]) -> tuple:
    """(card, check) for one source's annotations — the catalog first, then a card another split already uses, then
    an AI provider (sample lines, and the official documentation when Settings allows looking it up). A card is only
    returned when `check` passed on the real files. Else (None, {how, why, tried})."""
    img = Path(src.img_dir)
    hint = " ".join((str(root), str(ann), src.img_dir))
    card, sn, how = af.recognize(ann, hint)
    if how in ("yolo", "empty", "none"):
        return None, {"how": how}
    big = ann.is_file() and ann.stat().st_size > 200 * 1024 * 1024
    tried: List[dict] = []
    if card is not None:
        if big:          # a huge COCO file: parsed once, by the conversion job (the fingerprint is unambiguous)
            return card, {"ok": True, "deferred": True, "problems": [], "classes": card.get("classes") or []}
        st = af.check(card, img, ann, src.recursive)
        if st["ok"]:
            return card, st
        tried.append({"name": card.get("name"), "problems": st["problems"]})
    for k in known.values():
        if k is card:
            continue
        st = af.check(k, img, ann, src.recursive, sample=60, seconds=2.0)
        if st["ok"]:
            return k, st
    why = ""
    if ai is not False and sn is not None:
        from ..ai import formats as ai_formats            # lazy: the core layer does not load the AI stack up front
        try:
            res = ai_formats.infer_card(sn, img, ann, root, use_setting=ai is None, allow_guarded=allow_guarded,
                                        failed=tried)
        except Exception as e:  # noqa: BLE001 — a provider failure leaves the annotations unconverted, nothing more
            res = {"why": str(e)[:200]}
        for c in res.get("cards") or []:
            st = af.check(c, img, ann, src.recursive)
            if st["ok"]:
                ai_formats.remember(res.get("key"), c)
                return c, st
            tried.append({"name": c.get("name"), "problems": st["problems"], "source": c.get("source")})
            # one repair round: the model sees what went wrong on the real files
            try:
                rep = ai_formats.repair_card(sn, img, ann, root, c, st["problems"], allow_guarded=allow_guarded)
            except Exception as e:  # noqa: BLE001
                rep = {"why": str(e)[:200]}
            for c2 in rep.get("cards") or []:
                st2 = af.check(c2, img, ann, src.recursive)
                if st2["ok"]:
                    ai_formats.remember(rep.get("key"), c2)
                    return c2, st2
                tried.append({"name": c2.get("name"), "problems": st2["problems"], "source": c2.get("source")})
        why = res.get("why") or ""
    return None, {"how": "unknown", "why": why, "tried": tried}


def _rel(p: str, root: Path) -> str:
    try:
        r = Path(p).relative_to(root).as_posix()
    except ValueError:
        return to_host(p)
    return r if r != "." else Path(p).name


def resolve_formats(lay: Layout, ai: Optional[bool] = None, allow_guarded: bool = True) -> Layout:
    """Attach annotations in another format to each image source and pick the format card that reads them (see
    _format_for). The source's label_dir becomes where the scan writes the converted YOLO labels; a label folder Fovea
    converted earlier (its .fovea-format.json) is recognised as such and never converted again."""
    if lay.kind in ("list", "collection", "unknown") or not lay.sources:
        return lay
    root = Path(lay.root)
    known: Dict[str, dict] = {}
    unread: List[str] = []
    if lay.kind == "bare":
        _per_folder_files(lay, root)
    for s in lay.sources:
        if s.list_file or not s.img_dir:
            continue
        if s.label_dir and s.label_dir_exists:
            marker = af.read_marker(Path(s.label_dir))
            if marker and marker.get("state", "done") == "converting" and marker.get("format") and \
                    marker.get("annotations"):                  # a conversion that did not finish: redo it
                card = marker.get("card") or {}
                s.ann_format, s.ann_path = marker["format"], marker["annotations"]
                s.label_dir_exists = False
                f = lay.formats.setdefault(marker["format"], {"card": card, "state": "pending", "check": {},
                                                              "sources": 0})
                f["sources"] += 1
                continue
            if marker:                                          # labels Fovea converted earlier
                card = marker.get("card") or {}
                fid = marker.get("format") or card.get("id") or "converted"
                s.ann_format, s.ann_path = fid, marker.get("annotations") or s.ann_path
                f = lay.formats.setdefault(fid, {"card": card, "state": "converted", "counts": {}, "sources": 0})
                f["sources"] += 1
                for k, v in (marker.get("counts") or {}).items():
                    if isinstance(v, int):
                        f["counts"][k] = f["counts"].get(k, 0) + v
                f["classes"] = marker.get("classes") or f.get("classes")
                continue
            if af.tree_txt_kind(Path(s.label_dir)) != "other":
                continue
            s.ann_path = s.label_dir                            # a labels/ folder holding another format
        if not s.ann_path:
            found = find_annotations(s, root)
            if found is None:
                continue
            s.ann_path = str(found)
        ann = Path(s.ann_path)
        card, st = _format_for(s, ann, root, ai, allow_guarded, known)
        if card is None:
            if st.get("how") in ("yolo", "empty") and ann.is_dir() and not s.label_dir_exists:
                s.label_dir, s.label_dir_exists = str(ann), True    # YOLO labels in a folder with another name
                s.label_count = count_labels(ann, recursive=s.recursive) if s.image_count is not None else None
            elif st.get("how") == "unknown":
                why = "; ".join(t["problems"][0] for t in st.get("tried") or [] if t.get("problems")) or st.get("why")
                unread.append(f"{_rel(s.ann_path, root)}{f' ({why})' if why else ''}")
                lay.formats.setdefault("unknown", {"card": {"id": "unknown", "name": "Unknown format"},
                                                   "state": "unreadable", "why": why, "sources": 0})["sources"] += 1
            if st.get("how") != "unknown":
                s.ann_path = ""
            continue
        fid = card.get("id") or "custom"
        known[fid] = card
        s.ann_format = fid
        out = af.output_dir_for(Path(s.label_dir) if s.label_dir else Path(s.img_dir) / "labels", ann)
        s.label_dir, s.label_dir_exists = str(out), False
        s.label_count = 0 if s.image_count is not None else None
        f = lay.formats.setdefault(fid, {"card": card, "state": "pending", "check": _small_check(st), "sources": 0})
        f["sources"] += 1
    pend = [f for f in lay.formats.values() if f.get("state") == "pending"]
    if pend:
        names = ", ".join(sorted({f["card"].get("name") or "?" for f in pend}))
        outs = ", ".join(sorted({_rel(s.label_dir, root) + "/" for s in lay.sources if s.ann_format
                                 and not s.label_dir_exists}))[:300]
        lay.notes = [n for n in lay.notes if "No labels found" not in n and "were not used as labels" not in n]
        lay.notes.append(f"Annotations in {names} format — converted to YOLO labels in {outs} when you register "
                         "(the original files are not changed)")
        if not lay.classes:
            cls = next((f["card"].get("classes") for f in pend if f["card"].get("spec", {}).get("class", {})
                        .get("values")), None) or next((f.get("check", {}).get("classes") for f in pend), None)
            if cls:
                lay.classes, lay.classes_from = list(cls), "format"
    conv = [f for f in lay.formats.values() if f.get("state") == "converted"]
    if conv and not lay.classes:
        cls = next((f.get("classes") for f in conv if f.get("classes")), None)
        if cls:
            lay.classes, lay.classes_from = list(cls), "format"
    if unread:
        lay.notes.append("Annotations Fovea could not read: " + "; ".join(unread[:3]))
    return lay


def detect(raw: str, estimate: bool = True, _depth: int = 0, ai: Optional[bool] = None,
           allow_guarded: bool = True) -> Layout:
    """`ai`: how folders with other names than images/ and labels/ are found — None follows Settings (an AI provider
    by default, conventional names when none answers), False uses the names only. `allow_guarded`: False when the
    request may not reach a cloud provider (another machine)."""
    if _depth == 0:
        _budget.deadline = time.monotonic() + (ESTIMATE_SECONDS if estimate else 10.0)
        _budget.capped = False
    try:
        lay = _detect(raw, estimate, _depth)
        if _depth == 0:
            lay = resolve_roles(lay, estimate, ai, allow_guarded)
            lay = resolve_formats(lay, ai, allow_guarded)
            lay.sources = sort_sources(lay.sources)
            _budget.deadline = max(_budget.deadline, time.monotonic() + 2.0)   # orphan-label counts stay bounded too
            _unmatched_label_dirs(lay)
            if any(s.count_capped for s in lay.sources):
                lay.notes.append("Large folder: the counts above are a lower bound (indexing counts everything)")
        return lay
    finally:
        if _depth == 0:
            _budget.deadline = None


def _detect(raw: str, estimate: bool, _depth: int) -> Layout:
    p = resolve(raw)
    if not p or not p.exists():
        raise FileNotFoundError(f"Path not found: {raw}")
    if p.is_file():
        suf = p.suffix.lower()
        if suf in (".yaml", ".yml"):
            return from_yaml(p, estimate=estimate)
        if suf == ".txt":
            return from_list(p, estimate=estimate)
        if suf in IMAGE_EXTS:
            p = p.parent
        else:
            raise ValueError(f"Unsupported file type: {p.name}")

    yaml_classes: List[str] = []
    y = find_yaml(p)
    if y is not None:
        lay = from_yaml(y, root=p, estimate=estimate)
        if lay.sources:
            return lay
        yaml_classes = lay.classes

    def finish(lay: Layout) -> Layout:
        if not lay.classes:
            _set_classes_from_file(lay, Path(lay.root))
        if not lay.classes and yaml_classes:
            # A data.yaml whose paths do not resolve still names the classes (Roboflow, a copied dataset).
            lay.classes, lay.classes_from = yaml_classes, "yaml"
            lay.notes.append(f"{y.name}: its paths were not found, so the folder layout is used with its class names")
        return lay

    cvat = _cvat(p, estimate)
    if cvat is not None:
        return finish(cvat)

    # The images/ folder itself (or images/<split>) was picked: the dataset root is above it.
    if p.name.lower() == "images" and _depth == 0 and ((p.parent / "labels").is_dir()
                                                       or any(is_split_name(d.name) for d in _subdirs(p))):
        lay = _detect(str(p.parent), estimate, _depth + 1)
        if lay.sources:
            lay.notes.insert(0, "Using the dataset folder that contains images/")
            return lay

    # root/images[/<split>]
    if (p / "images").is_dir():
        img_root = p / "images"
        subs = _subdirs(img_root)
        split_subs = [d for d in subs if is_split_name(d.name)]
        if split_subs:
            lay = Layout(kind="ultralytics", root=str(p))
            for d in split_subs:
                lay.sources.append(_mk_dir_source(norm_split(d.name), d, p / "labels" / d.name, estimate=estimate))
            others = [d for d in subs if d not in split_subs and count_images(d, 1)]
            for d in others:
                lay.sources.append(_mk_dir_source(d.name, d, p / "labels" / d.name, estimate=estimate))
            if has_images_shallow(img_root):
                # Split folders plus loose images (a stray render, a half-finished manual split). Treating the
                # whole tree as one split-less folder erased every image's split; skipping the loose files would
                # hide them. Index them as their own, non-recursive, split-less source.
                lay.sources.append(_mk_dir_source("", img_root, p / "labels", estimate=estimate, recursive=False))
                lay.notes.append("Images directly inside images/ (outside the split folders) are indexed without a split")
        else:
            lay = Layout(kind="flat", root=str(p))
            lay.sources.append(_mk_dir_source("", img_root, p / "labels", estimate=estimate))
        return finish(lay)

    # root/<split>/images — unless the sub-folders are datasets of their own
    kids = [(c, _child_kind(c)) for c in _subdirs(p)]
    kids = [(c, k) for c, k in kids if k]
    roots = [c for c, k in kids if k == "root"]
    splits = [c for c, k in kids if k == "split"]
    if roots:
        if len(kids) == 1 and _depth < 2:
            lay = _detect(str(roots[0]), estimate, _depth + 1)
            lay.notes.insert(0, f"Detected dataset in subfolder '{roots[0].name}'")
            return lay
        if not has_images_shallow(p):
            return _collection(p, kids)
    if splits:
        lay = Layout(kind="split-first", root=str(p))
        names = split_names([d.name for d in splits])
        for d in splits:
            lay.sources.append(_mk_dir_source(names[d.name], d / "images", d / "labels", estimate=estimate))
        if not all(is_split_name(d.name) for d in splits) and len(splits) > 1:
            lay.notes.append("Folders " + ", ".join(d.name for d in splits[:6]) + " are indexed as splits of one dataset")
        return finish(lay)

    # bare / mixed folder of images
    if has_images_shallow(p) or count_images(p, 1):
        split = norm_split(p.name) if is_split_name(p.name) else ""
        if _sidecar_labels(p):
            lay = Layout(kind="mixed", root=str(p))
            lay.sources.append(_mk_dir_source(split, p, p, estimate=estimate))
        else:
            # Ultralytics' images -> labels rule: <ds>/images/train -> <ds>/labels/train. A folder not under
            # an images/ component gets a labels/ folder inside it.
            label_dir = derive_label_dir(p) if "images" in (x.lower() for x in p.parts) else p / "labels"
            if label_dir.is_dir():
                lay = Layout(kind="flat", root=str(p))
            else:
                lay = Layout(kind="bare", root=str(p), notes=["No labels found — labels/ will be created when you annotate"])
                stray = _stray_txt(p)
                if stray:
                    lay.notes.append(f"{stray:,} .txt file(s) elsewhere in this folder were not used as labels — "
                                     "if they are labels, add the folder that contains images/ and labels/ instead")
            lay.sources.append(_mk_dir_source(split, p, label_dir, estimate=estimate))
        return finish(lay)

    if kids and _depth < 2:
        if len(kids) == 1:
            lay = _detect(str(kids[0][0]), estimate, _depth + 1)
            lay.notes.insert(0, f"Detected dataset in subfolder '{kids[0][0].name}'")
            return lay
        return _collection(p, kids)

    return Layout(kind="unknown", root=str(p), notes=["No images, images/ folder, split folders or data.yaml found"])


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    """URL-safe dataset id. Non-ASCII names (Korean, accents) keep whatever ASCII they have plus a short hash
    of the full name, so '한글 데이터셋' and '다른 데이터셋' get distinct ids instead of both 'dataset'."""
    raw = (name or "").strip()
    s = _SLUG_RE.sub("-", raw.lower()).strip("-")[:40].strip("-")
    if re.search(r"[^\x00-\x7f]", raw):
        h = hashlib.sha1(raw.encode("utf-8", "surrogatepass")).hexdigest()[:6]
        s = f"{s}-{h}" if s else f"ds-{h}"
    return s[:48] or "dataset"


_GENERIC_STEMS = {"data", "dataset", "train", "val", "valid", "test", "images", "obj", "custom"}


def suggest_name(p: Path) -> str:
    """A display name for a dataset path: the folder name, or for a data.yaml / train.txt / images/ / split
    folder the name of the dataset folder that holds it (not 'data', 'train' or 'images')."""
    p = Path(p)
    if p.is_file():
        return p.parent.name or p.stem if p.stem.lower() in _GENERIC_STEMS else p.stem
    up = p
    while up.name.lower() in _GENERIC_STEMS and up.parent != up and up.parent.name:
        up = up.parent
    return up.name or p.name or "dataset"
