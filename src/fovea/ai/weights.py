"""Model weights in ``MODEL_DIR``: classification by family, safe name resolution, curated downloads.

Only names that exist in ``MODEL_DIR`` (exact file stem) can ever be loaded — no paths, no ``..`` — because a
``.pt`` file is a pickle and loading one runs code (finding C12). Downloads happen only on an explicit user action,
only for the curated list below, into ``MODEL_DIR/<name>.part`` and then an atomic rename.
"""
from __future__ import annotations

import os
import re
import time
import urllib.request
from pathlib import Path
from typing import List, Optional

from ..config import MODEL_DIR

EXTS = (".pt", ".onnx", ".engine", ".ts")
ASSET_BASE = "https://github.com/ultralytics/assets/releases/download/v8.4.0/"

AGPL = "AGPL-3.0"
APACHE = "Apache-2.0"

# name (file), family, approx size (MB), license, note. Sizes from the GitHub release (v8.4.0).
CURATED: List[dict] = [
    {"name": "yolo26n.pt", "family": "yolo", "size_mb": 5.5, "license": AGPL, "note": "COCO 80 classes · fastest (CPU-friendly)"},
    {"name": "yolo26s.pt", "family": "yolo", "size_mb": 20.4, "license": AGPL, "note": "COCO 80 classes · fast"},
    {"name": "yolo26m.pt", "family": "yolo", "size_mb": 44.3, "license": AGPL, "note": "COCO 80 classes · balanced"},
    {"name": "yolo26l.pt", "family": "yolo", "size_mb": 53.2, "license": AGPL, "note": "COCO 80 classes · as accurate as yolo12x at half the cost"},
    {"name": "yolo26x.pt", "family": "yolo", "size_mb": 118.7, "license": AGPL, "note": "COCO 80 classes · most accurate"},
    {"name": "yoloe-26s-seg.pt", "family": "yoloe", "size_mb": 31.1, "license": AGPL, "note": "Open vocabulary (text prompts / exemplars) · laptop"},
    {"name": "yoloe-26m-seg.pt", "family": "yoloe", "size_mb": 70.4, "license": AGPL, "note": "Open vocabulary · balanced"},
    {"name": "yoloe-26l-seg.pt", "family": "yoloe", "size_mb": 79.4, "license": AGPL, "note": "Open vocabulary · most accurate (GPU)"},
    {"name": "mobile_sam.pt", "family": "sam", "size_mb": 40.7, "license": APACHE, "note": "Box refine / click-to-box · fastest, CPU-viable"},
    {"name": "sam2.1_t.pt", "family": "sam", "size_mb": 78.1, "license": APACHE, "note": "Box refine / click-to-box · recommended"},
    {"name": "sam2.1_s.pt", "family": "sam", "size_mb": 92.3, "license": APACHE, "note": "Box refine / click-to-box · higher quality"},
    {"name": "sam2.1_b.pt", "family": "sam", "size_mb": 161.9, "license": APACHE, "note": "Box refine / click-to-box · best quality, slower"},
    {"name": "mobileclip2_b.ts", "family": "text", "size_mb": 253.8, "license": "apple-amlr",
     "note": "Text encoder needed for YOLOE-26 text prompts (not for exemplars)"},
]
_CURATED_BY_NAME = {c["name"]: c for c in CURATED}

_NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._\-]{0,127}$")
_TASK_SUFFIX = {"-cls": "classify", "-seg": "segment", "-pose": "pose", "-obb": "obb", "-sem": "semantic",
                "-depth": "depth"}


def family_of(filename: str) -> str:
    n = filename.lower()
    stem, ext = os.path.splitext(n)
    if ext == ".ts" or stem.startswith(("mobileclip", "clip")):
        return "text"
    if stem.startswith("yoloe"):
        return "yoloe"
    if stem.startswith(("sam", "mobile_sam", "sam2", "sam3")) or stem.startswith("efficientsam"):
        return "sam"
    if "world" in stem:
        return "world"            # YOLO-World: not supported (YOLOE replaces it)
    if stem.startswith("fastsam"):
        return "other"
    return "yolo"


def task_of(stem: str) -> str:
    s = stem.lower()
    for suf, task in _TASK_SUFFIX.items():
        if s.endswith(suf) or (suf + "-") in s or s.endswith(suf + "-pf"):
            return task
    return "detect"


def _scan() -> List[dict]:
    """Weight files directly in MODEL_DIR. A symbolic link there (a large checkpoint shared from another folder) is
    the owner's own choice and counts like a file when it points at a regular file with a weights extension — the
    listing and :func:`resolve` must agree, or a model is offered as Ready and then refused as unknown."""
    out = []
    try:
        entries = sorted(Path(MODEL_DIR).iterdir()) if Path(MODEL_DIR).is_dir() else []
    except OSError:
        entries = []
    for p in entries:
        try:
            if not p.is_file() or p.suffix.lower() not in EXTS or p.name.startswith("."):
                continue
            if p.is_symlink() and p.resolve().suffix.lower() not in EXTS:
                continue
            st = p.stat()
        except (OSError, RuntimeError):         # RuntimeError: a symlink loop
            continue
        out.append({"name": p.stem, "file": p.name, "path": str(p), "family": family_of(p.name), "task": task_of(p.stem),
                    "size_mb": round(st.st_size / 1e6, 1), "mtime": st.st_mtime})
    return out


def local_weights(family: Optional[str] = None) -> List[dict]:
    items = _scan()
    return [w for w in items if family is None or w["family"] == family]


def resolve(name: str, family: Optional[str] = None) -> Path:
    """Map a model id (file stem, or exact file name) to its absolute path inside MODEL_DIR, or raise ValueError.
    Anything that is not exactly one of the listed files — paths, ``..``, other folders — is refused."""
    name = (name or "").strip()
    if not name or not _NAME_RE.match(name) or "/" in name or "\\" in name or ".." in name:
        raise ValueError(f"Unknown model: {name!r}")
    root = Path(MODEL_DIR).resolve()
    for w in _scan():
        if w["name"] == name or w["file"] == name:
            if family and w["family"] != family:
                raise ValueError(f"{name} is a {w['family']} model, not a {family} model")
            # The listed entry itself must sit directly in MODEL_DIR (a name from a request can never reach another
            # folder); a symlink there may point anywhere — the owner put it in their model folder.
            p = root / w["file"]
            if p.parent != root or not p.is_file():
                raise ValueError(f"Unknown model: {name!r}")
            return p
    raise ValueError(f"Model not found in {MODEL_DIR}: {name}")


def weights_list() -> List[dict]:
    """Curated downloads (with on-disk state) followed by any other weights already in MODEL_DIR."""
    local = {w["file"]: w for w in _scan()}
    out = []
    for c in CURATED:
        w = local.get(c["name"])
        out.append({**c, "id": Path(c["name"]).stem, "downloaded": w is not None,
                    "size_mb": w["size_mb"] if w else c["size_mb"], "url": ASSET_BASE + c["name"], "curated": True})
    for f, w in local.items():
        if f in _CURATED_BY_NAME:
            continue
        lic = AGPL if w["family"] in ("yolo", "yoloe") else (APACHE if w["family"] == "sam" else "")
        out.append({"name": f, "id": w["name"], "family": w["family"], "size_mb": w["size_mb"], "downloaded": True,
                    "license": lic, "url": None, "note": "Already in the model folder", "curated": False,
                    "task": w["task"]})
    return out


def curated(name: str) -> Optional[dict]:
    name = (name or "").strip()
    if name in _CURATED_BY_NAME:
        return _CURATED_BY_NAME[name]
    for c in CURATED:
        if Path(c["name"]).stem == name:
            return c
    return None


def download(name: str, job=None, base: Optional[str] = None, timeout: float = 60.0) -> dict:
    """Download one curated file into MODEL_DIR (``.part`` + rename). Reports progress through ``job.update``."""
    c = curated(name)
    if c is None:
        raise ValueError(f"{name!r} is not in the curated download list")
    dest = Path(MODEL_DIR) / c["name"]
    if dest.is_file():
        return {"name": c["name"], "downloaded": True, "already": True, "size_mb": round(dest.stat().st_size / 1e6, 1)}
    Path(MODEL_DIR).mkdir(parents=True, exist_ok=True)
    part = dest.with_name(dest.name + ".part")
    url = (base or ASSET_BASE) + c["name"]
    req = urllib.request.Request(url, headers={"User-Agent": "fovea"})
    expected = int(c["size_mb"] * 1e6)
    t0 = time.time()
    got = 0
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp, open(part, "wb") as fh:
            total = int(resp.headers.get("Content-Length") or 0) or expected
            if job:
                job.update(0, total, f"Downloading {c['name']}…")
            last = 0.0
            while True:
                chunk = resp.read(1 << 20)
                if not chunk:
                    break
                fh.write(chunk)
                got += len(chunk)
                if job and time.time() - last > 0.3:
                    last = time.time()
                    job.update(min(got, total), total, f"Downloading {c['name']} · {got / 1e6:.0f} / {total / 1e6:.0f} MB")
        if got < 1000:
            raise IOError(f"Download of {c['name']} returned only {got} bytes")
        os.replace(part, dest)
    finally:
        if part.exists():
            try:
                part.unlink()
            except OSError:
                pass
    return {"name": c["name"], "downloaded": True, "size_mb": round(got / 1e6, 1), "seconds": round(time.time() - t0, 1)}
