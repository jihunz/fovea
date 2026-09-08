"""Dataset layout detection.

Supported layouts
-----------------
* ``yaml``        : Ultralytics data.yaml (path/train/val/test + names)
* ``ultralytics`` : root/images/<split>, root/labels/<split>
* ``split-first`` : root/<split>/images, root/<split>/labels
* ``flat``        : root/images, root/labels
* ``mixed``       : images and .txt labels side by side in one folder
* ``bare``        : folder of images without labels (labels/ will be created on save)
* ``list``        : a train.txt style list of image paths (labels via images->labels rule)
"""
from __future__ import annotations

import os
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import List, Optional

import yaml

from ..config import IMAGE_EXTS, LABEL_EXT, SPLIT_NAMES
from ..paths import resolve, to_host

SPLIT_ALIASES = {"valid": "val", "validation": "val", "eval": "test", "evaluation": "test", "training": "train",
                 "testing": "test"}
_SPLIT_LIKE = set(SPLIT_NAMES) | set(SPLIT_ALIASES.keys()) | {"train", "val", "test"}


def norm_split(name: str) -> str:
    n = name.strip().lower()
    return SPLIT_ALIASES.get(n, n)


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

    def to_dict(self) -> dict:
        return {
            "kind": self.kind, "root": self.root, "root_host": to_host(self.root),
            "sources": [s.to_public() for s in self.sources],
            "data_yaml": self.data_yaml, "data_yaml_host": to_host(self.data_yaml) if self.data_yaml else None,
            "classes": self.classes, "notes": self.notes,
        }

    @staticmethod
    def from_dict(d: dict) -> "Layout":
        srcs = []
        for s in d.get("sources", []):
            srcs.append(Source(
                split=s.get("split", ""), img_dir=s.get("img_dir", ""), label_dir=s.get("label_dir", ""),
                list_file=s.get("list_file"), base=s.get("base"), label_dir_exists=s.get("label_dir_exists", False),
                image_count=s.get("image_count"), label_count=s.get("label_count"),
            ))
        return Layout(kind=d.get("kind", "unknown"), root=d.get("root", ""), sources=srcs,
                      data_yaml=d.get("data_yaml"), classes=list(d.get("classes", [])), notes=list(d.get("notes", [])))


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


def count_images(d: Path, limit: Optional[int] = None) -> int:
    n = 0
    for _root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        for f in files:
            dot = f.rfind(".")
            if dot > 0 and f[dot:].lower() in IMAGE_EXTS:
                n += 1
                if limit and n >= limit:
                    return n
    return n


def count_labels(d: Path) -> int:
    n = 0
    if not d.is_dir():
        return 0
    for _root, dirs, files in os.walk(d):
        dirs[:] = [x for x in dirs if not x.startswith(".")]
        n += sum(1 for f in files if f.lower().endswith(LABEL_EXT))
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


def find_yaml(d: Path) -> Optional[Path]:
    for name in ("data.yaml", "data.yml", "dataset.yaml", "dataset.yml"):
        p = d / name
        if p.is_file():
            return p
    try:
        for p in sorted(d.iterdir()):
            if p.suffix.lower() in (".yaml", ".yml") and p.is_file():
                try:
                    doc = yaml.safe_load(p.read_text(encoding="utf-8")) or {}
                except Exception:
                    continue
                if isinstance(doc, dict) and ("names" in doc or "train" in doc):
                    return p
    except OSError:
        pass
    return None


def parse_names(doc: dict) -> List[str]:
    names = doc.get("names")
    if isinstance(names, dict):
        out: List[str] = []
        try:
            keys = sorted(int(k) for k in names.keys())
        except Exception:
            return [str(v) for v in names.values()]
        for k in range(keys[-1] + 1 if keys else 0):
            out.append(str(names.get(k, names.get(str(k), f"class_{k}"))))
        return out
    if isinstance(names, list):
        return [str(n) for n in names]
    nc = doc.get("nc")
    if isinstance(nc, int) and nc > 0:
        return [f"class_{i}" for i in range(nc)]
    return []


def read_classes_file(d: Path) -> List[str]:
    for name in ("classes.txt", "obj.names", "labels.txt"):
        p = d / name
        if p.is_file():
            try:
                lines = [ln.strip() for ln in p.read_text(encoding="utf-8").splitlines()]
                lines = [ln for ln in lines if ln]
                if lines and all(len(ln) < 64 for ln in lines):
                    return lines
            except Exception:
                pass
    return []


def _mk_dir_source(split: str, img_dir: Path, label_dir: Optional[Path] = None, estimate: bool = True) -> Source:
    label_dir = label_dir or derive_label_dir(img_dir)
    src = Source(split=split, img_dir=str(img_dir), label_dir=str(label_dir), label_dir_exists=label_dir.is_dir())
    if estimate:
        src.image_count = count_images(img_dir)
        src.label_count = count_labels(label_dir)
    return src


def _mk_list_source(split: str, list_file: Path, base: Optional[Path] = None, estimate: bool = True) -> Source:
    src = Source(split=split, list_file=str(list_file), base=str(base or list_file.parent), label_dir_exists=True)
    if estimate:
        try:
            with list_file.open("r", encoding="utf-8", errors="ignore") as fh:
                src.image_count = sum(1 for ln in fh if ln.strip())
        except OSError:
            src.image_count = None
    return src


# --------------------------------------------------------------------------- detectors

def from_yaml(yaml_path: Path, root: Optional[Path] = None, estimate: bool = True) -> Layout:
    doc = yaml.safe_load(yaml_path.read_text(encoding="utf-8")) or {}
    if not isinstance(doc, dict):
        raise ValueError("YAML root must be a mapping")
    base = yaml_path.parent
    if doc.get("path"):
        pf = str(doc["path"])
        cand = resolve(pf)
        if not cand.is_absolute() or not cand.exists():
            cand = (yaml_path.parent / pf).resolve()
        if cand.is_dir():
            base = cand
    layout = Layout(kind="yaml", root=str(root or base), data_yaml=str(yaml_path), classes=parse_names(doc))
    for split in ("train", "val", "test"):
        v = doc.get(split)
        if not v:
            continue
        entries = v if isinstance(v, list) else [v]
        for entry in entries:
            e = Path(str(entry))
            cand = resolve(str(e)) if e.is_absolute() else (base / e).resolve()
            if not cand.exists():
                alt = (yaml_path.parent / e).resolve()
                cand = alt if alt.exists() else cand
            if cand.is_dir():
                layout.sources.append(_mk_dir_source(split, cand, estimate=estimate))
            elif cand.is_file() and cand.suffix.lower() == ".txt":
                layout.sources.append(_mk_list_source(split, cand, base, estimate=estimate))
            else:
                layout.notes.append(f"{split}: path not found ({entry})")
    if not layout.sources:
        layout.notes.append("data.yaml has no resolvable train/val/test entries")
    return layout


def from_list(txt: Path, split: str = "", estimate: bool = True) -> Layout:
    split = split or norm_split(txt.stem) if norm_split(txt.stem) in ("train", "val", "test") else split
    lay = Layout(kind="list", root=str(txt.parent))
    lay.sources.append(_mk_list_source(split, txt, estimate=estimate))
    cls = read_classes_file(txt.parent)
    if cls:
        lay.classes = cls
    return lay


def detect(raw: str, estimate: bool = True, _depth: int = 0) -> Layout:
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

    y = find_yaml(p)
    if y is not None:
        lay = from_yaml(y, root=p, estimate=estimate)
        if lay.sources:
            return lay

    # root/images[/<split>]
    if (p / "images").is_dir():
        img_root = p / "images"
        subs = [d for d in sorted(img_root.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        split_subs = [d for d in subs if norm_split(d.name) in ("train", "val", "test") or d.name.lower() in _SPLIT_LIKE]
        if split_subs and not has_images_shallow(img_root):
            lay = Layout(kind="ultralytics", root=str(p))
            for d in split_subs:
                lay.sources.append(_mk_dir_source(norm_split(d.name), d, p / "labels" / d.name, estimate=estimate))
            others = [d for d in subs if d not in split_subs and count_images(d, 1)]
            for d in others:
                lay.sources.append(_mk_dir_source(d.name, d, p / "labels" / d.name, estimate=estimate))
        else:
            lay = Layout(kind="flat", root=str(p))
            lay.sources.append(_mk_dir_source("", img_root, p / "labels", estimate=estimate))
        lay.classes = read_classes_file(p)
        return lay

    # root/<split>/images
    split_dirs = []
    try:
        for d in sorted(p.iterdir()):
            if d.is_dir() and not d.name.startswith(".") and (d / "images").is_dir():
                split_dirs.append(d)
    except OSError:
        pass
    if split_dirs:
        lay = Layout(kind="split-first", root=str(p))
        for d in split_dirs:
            lay.sources.append(_mk_dir_source(norm_split(d.name), d / "images", d / "labels", estimate=estimate))
        lay.classes = read_classes_file(p)
        return lay

    # bare / mixed folder of images
    if has_images_shallow(p) or count_images(p, 1):
        mixed = False
        try:
            with os.scandir(p) as it:
                stems = set()
                txts = set()
                for e in it:
                    if e.is_file():
                        if is_image(e):
                            stems.add(Path(e.name).stem)
                        elif e.name.lower().endswith(LABEL_EXT):
                            txts.add(Path(e.name).stem)
                mixed = bool(stems & txts)
        except OSError:
            pass
        if mixed:
            lay = Layout(kind="mixed", root=str(p))
            lay.sources.append(_mk_dir_source("", p, p, estimate=estimate))
        elif (p / "labels").is_dir():
            lay = Layout(kind="flat", root=str(p))
            lay.sources.append(_mk_dir_source("", p, p / "labels", estimate=estimate))
        else:
            lay = Layout(kind="bare", root=str(p), notes=["No labels found — labels/ will be created when you annotate"])
            lay.sources.append(_mk_dir_source("", p, p / "labels", estimate=estimate))
        lay.classes = read_classes_file(p)
        return lay

    # look one level down
    if _depth < 1:
        try:
            children = [d for d in sorted(p.iterdir()) if d.is_dir() and not d.name.startswith(".")]
        except OSError:
            children = []
        cands = [c for c in children if (c / "images").is_dir() or find_yaml(c) or has_images_shallow(c)]
        if len(cands) == 1:
            lay = detect(str(cands[0]), estimate=estimate, _depth=_depth + 1)
            lay.notes.insert(0, f"Detected dataset in subfolder '{cands[0].name}'")
            return lay
        if len(cands) > 1:
            lay = Layout(kind="unknown", root=str(p))
            lay.notes.append("Multiple candidate datasets found: " + ", ".join(c.name for c in cands[:8]))
            return lay

    return Layout(kind="unknown", root=str(p), notes=["No images, images/ folder, split folders or data.yaml found"])


_SLUG_RE = re.compile(r"[^a-z0-9]+")


def slugify(name: str) -> str:
    s = _SLUG_RE.sub("-", name.strip().lower()).strip("-")
    return s[:48] or "dataset"
