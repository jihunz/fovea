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
"""
from __future__ import annotations

import hashlib
import os
import re
import threading
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from ..config import IMAGE_EXTS, LABEL_EXT, SPLIT_NAMES
from ..paths import resolve, to_host
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

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "root": self.root, "root_host": to_host(self.root),
            "sources": [s.to_public() for s in self.sources],
            "data_yaml": self.data_yaml, "data_yaml_host": to_host(self.data_yaml) if self.data_yaml else None,
            "classes": self.classes, "notes": self.notes, "classes_from": self.classes_from,
            "candidates": self.candidates, "unmatched_labels": self.unmatched_labels,
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
            ))
        return Layout(kind=d.get("kind", "unknown"), root=d.get("root", ""), sources=srcs,
                      data_yaml=d.get("data_yaml"), classes=list(d.get("classes", [])), notes=list(d.get("notes", [])),
                      classes_from=d.get("classes_from", ""), candidates=list(d.get("candidates", [])),
                      unmatched_labels=list(d.get("unmatched_labels", [])))


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


def detect(raw: str, estimate: bool = True, _depth: int = 0) -> Layout:
    if _depth == 0:
        _budget.deadline = time.monotonic() + (ESTIMATE_SECONDS if estimate else 10.0)
        _budget.capped = False
    try:
        lay = _detect(raw, estimate, _depth)
        if _depth == 0:
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
        for d in splits:
            lay.sources.append(_mk_dir_source(norm_split(d.name), d / "images", d / "labels", estimate=estimate))
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
