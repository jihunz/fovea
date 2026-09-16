"""Exports: ZIP streaming, server-side subset copy, list files, data.yaml."""
from __future__ import annotations

import io
import os
import queue
import shutil
import threading
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import yaml
from PIL import Image

from .. import db
from ..jobs import Job
from ..paths import resolve, to_host
from .imagesq import iter_images
from .labels import format_box, read_label_file, write_label_file

Image.MAX_IMAGE_PIXELS = None


def _resize_spec(resize: Optional[dict]) -> Optional[Tuple[str, int, int]]:
    if not resize or resize.get("mode", "none") == "none":
        return None
    mode = resize.get("mode")
    if mode == "max":
        return ("max", int(resize.get("size", 1280)), 0)
    if mode == "exact":
        return ("exact", int(resize.get("width", 640)), int(resize.get("height", 640)))
    return None


def _process_ext(src: Path, spec: Optional[Tuple[str, int, int]]) -> str:
    """Extension _process_image will produce, known before any work is done."""
    return src.suffix.lower() if spec is None else ".jpg"


def _process_image(src: Path, spec: Optional[Tuple[str, int, int]]) -> Tuple[bytes, str]:
    """Return (bytes, extension). Without resize the file is copied verbatim."""
    if spec is None:
        return src.read_bytes(), src.suffix.lower()
    with Image.open(src) as im:
        if im.mode not in ("RGB", "L"):
            im = im.convert("RGB")
        if spec[0] == "max":
            im.thumbnail((spec[1], spec[1]), Image.Resampling.LANCZOS)
        else:
            im = im.resize((spec[1], spec[2]), Image.Resampling.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=92)
        return buf.getvalue(), ".jpg"


def data_yaml_text(names: List[str], splits: Iterable[str], path: str = ".") -> str:
    doc: Dict[str, Any] = {"path": path}
    splits = list(splits)
    for s in ("train", "val", "test"):
        if s in splits:
            doc[s] = f"images/{s}"
    if "" in splits:
        doc.setdefault("train", "images")
    doc["nc"] = len(names)
    doc["names"] = {i: n for i, n in enumerate(names)}
    return yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)


def _label_lines(label_path: Optional[str]) -> str:
    if not label_path:
        return ""
    p = Path(label_path)
    if not p.is_file():
        return ""
    # An export is a training set (it ships a data.yaml): confidence columns from prediction files would
    # make trainers such as Ultralytics reject the file, so exports always carry plain 5-column rows.
    boxes, _c, _i = read_label_file(p)
    return "".join(format_box(b) + "\n" for b in boxes)


def _target_names(item: dict) -> Tuple[str, str]:
    """Return (images/..., labels/...) relative paths inside the export."""
    split = item["split"] or ""
    rel = item["rel_path"]
    if split and rel.startswith(split + "/"):
        rel = rel[len(split) + 1:]
    base = f"{split}/" if split else ""
    return f"images/{base}{rel}", f"labels/{base}{Path(rel).with_suffix('.txt').as_posix()}"


class _QueueFile:
    def __init__(self, q: "queue.Queue[Optional[bytes]]"):
        self._q = q
        self._pos = 0

    def write(self, data: bytes) -> int:
        self._q.put(bytes(data))
        self._pos += len(data)
        return len(data)

    def tell(self) -> int:
        return self._pos

    def seekable(self) -> bool:
        return False

    def flush(self) -> None:
        pass


def stream_zip(dataset_id: str, names: List[str], filters: Dict[str, Any], resize: Optional[dict] = None,
               include_unlabeled: bool = True, include_yaml: bool = True) -> Iterator[bytes]:
    spec = _resize_spec(resize)
    q: "queue.Queue[Optional[bytes]]" = queue.Queue(maxsize=64)

    def produce():
        try:
            splits_seen = set()
            with zipfile.ZipFile(_QueueFile(q), "w", zipfile.ZIP_STORED) as zf:
                for item in iter_images(dataset_id, filters):
                    if not include_unlabeled and not item["has_label"]:
                        continue
                    src = Path(item["abs_path"])
                    if not src.is_file():
                        continue
                    img_rel, lbl_rel = _target_names(item)
                    try:
                        data, ext = _process_image(src, spec)
                    except Exception:
                        continue
                    if spec is not None:
                        img_rel = str(Path(img_rel).with_suffix(ext))
                    zf.writestr(img_rel, data)
                    if item["has_label"]:
                        zf.writestr(lbl_rel, _label_lines(item.get("label_path")))
                    splits_seen.add(item["split"] or "")
                if include_yaml:
                    zf.writestr("data.yaml", data_yaml_text(names, splits_seen))
        finally:
            q.put(None)

    t = threading.Thread(target=produce, daemon=True)
    t.start()
    while True:
        chunk = q.get()
        if chunk is None:
            break
        yield chunk
    t.join()


def _inside(child: Path, parent: Path) -> bool:
    try:
        child.relative_to(parent)
        return True
    except ValueError:
        return False


def check_copy_target(dataset_id: str, target_dir: str) -> Path:
    """Refuse an export folder that overlaps the dataset. Exporting a dataset into itself makes every
    destination path a source path: a resize re-encodes the originals in place and the label pass
    rewrites the source labels — all while the job reports success."""
    target = resolve(target_dir)
    if not target.is_absolute():
        raise ValueError("Target must be an absolute path")
    target = Path(os.path.realpath(target))
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


def _same_file(a: Path, b: Path) -> bool:
    try:
        return a.exists() and os.path.samefile(a, b)
    except OSError:
        return False


def copy_subset(job: Job, dataset_id: str, names: List[str], filters: Dict[str, Any], target_dir: str,
                resize: Optional[dict] = None, include_unlabeled: bool = True, include_yaml: bool = True) -> dict:
    spec = _resize_spec(resize)
    target = check_copy_target(dataset_id, target_dir)
    target.mkdir(parents=True, exist_ok=True)
    from .imagesq import count_images
    total = count_images(dataset_id, filters)
    job.update(0, total, "Copying")
    n_img = n_lbl = failed = collisions = 0
    splits_seen = set()
    for idx, item in enumerate(iter_images(dataset_id, filters)):
        if idx % 50 == 0:
            job.update(done=idx, message=f"Copying {idx:,}/{total:,}")
        if not include_unlabeled and not item["has_label"]:
            continue
        src = Path(item["abs_path"])
        if not src.is_file():
            continue
        img_rel, lbl_rel = _target_names(item)
        dst_img = target / img_rel
        dst_img.parent.mkdir(parents=True, exist_ok=True)
        try:
            if spec is not None:
                dst_img = dst_img.with_suffix(_process_ext(src, spec))
            if _same_file(dst_img, src):
                collisions += 1           # backstop for layouts the folder check cannot see: never overwrite a source
                continue
            if spec is None:
                shutil.copy2(src, dst_img)
            else:
                data, _ext = _process_image(src, spec)
                dst_img.write_bytes(data)
            n_img += 1
        except Exception:  # noqa: BLE001 — one unreadable image must not stop the export; it is counted
            failed += 1
            continue
        if item["has_label"] and item.get("label_path"):
            lp = Path(item["label_path"])
            dst_lbl = target / lbl_rel
            if lp.is_file() and not _same_file(dst_lbl, lp):
                boxes, _c, _i = read_label_file(lp)
                write_label_file(dst_lbl, boxes)
                n_lbl += 1
        splits_seen.add(item["split"] or "")
    if include_yaml:
        (target / "data.yaml").write_text(data_yaml_text(names, splits_seen), encoding="utf-8")
    job.update(done=total, message="Done")
    return {"images": n_img, "labels": n_lbl, "failed": failed, "skipped_same_file": collisions, "target": to_host(target)}


def render_list(dataset_id: str, filters: Dict[str, Any], style: str = "host") -> str:
    lines = []
    for item in iter_images(dataset_id, filters):
        if style == "rel":
            lines.append(item["rel_path"])
        elif style == "container":
            lines.append(item["abs_path"])
        else:
            lines.append(item["abs_path_host"])
    return "\n".join(lines) + ("\n" if lines else "")


def write_data_yaml(dataset_root: str, names: List[str], layout: dict) -> str:
    root = Path(dataset_root)
    doc: Dict[str, Any] = {"path": to_host(root)}
    for s in layout.get("sources", []):
        split = s.get("split") or "train"
        if s.get("list_file"):
            doc.setdefault(split, to_host(s["list_file"]))
        elif s.get("img_dir"):
            try:
                rel = Path(s["img_dir"]).relative_to(root).as_posix()
            except ValueError:
                rel = to_host(s["img_dir"])
            doc.setdefault(split, rel)
    doc["nc"] = len(names)
    doc["names"] = {i: n for i, n in enumerate(names)}
    text = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    out = root / "data.yaml"
    out.write_text(text, encoding="utf-8")
    return str(out)
