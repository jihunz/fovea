"""Path resolution: host<->container mapping, Windows normalisation, browsing."""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import List, Optional, Tuple

from .config import CONTAINER_MOUNT, HOST_PATH, IMAGE_EXTS

_WIN_DRIVE_RE = re.compile(r"^([A-Za-z]):[/\\]+")


def normalize_win_path(p: str) -> str:
    """C://Users/foo -> /c/Users/foo (docker-style); backslashes -> slashes."""
    m = _WIN_DRIVE_RE.match(p)
    if m:
        p = "/" + m.group(1).lower() + "/" + p[m.end():]
    return p.replace("\\", "/")


def is_windows_path(p: str) -> bool:
    return bool(_WIN_DRIVE_RE.match(p or "")) or (p or "").startswith("\\\\")


_HOST_IS_WINDOWS = is_windows_path(HOST_PATH)
PATH_MAPPINGS: List[Tuple[str, str]] = []
if HOST_PATH:
    PATH_MAPPINGS.append((HOST_PATH.rstrip("/\\"), CONTAINER_MOUNT))
    _norm = normalize_win_path(HOST_PATH).rstrip("/")
    if _norm != HOST_PATH.rstrip("/\\"):
        PATH_MAPPINGS.append((_norm, CONTAINER_MOUNT))


def _prefix_rest(test: str, pfx: str) -> Optional[str]:
    """The remainder of test after host prefix pfx, or None. Windows prefixes compare case-insensitively
    (C:\\Data and c:\\data are the same folder)."""
    if not pfx:
        return None
    a, b = (test.lower(), pfx.lower()) if _HOST_IS_WINDOWS else (test, pfx)
    if a == b:
        return ""
    for sep in ("/", "\\"):
        if a.startswith(b + sep):
            return test[len(pfx) + 1:]
    return None


def resolve(raw: str, base: Optional[str | os.PathLike] = None) -> Path:
    """Resolve a user-supplied path (host or container form) to a local Path.

    Returns the candidate even when it does not exist so callers can report it — and a host path under
    HOST_PATH maps into the container mount even before it exists (a new export folder, B5). A relative
    path is taken relative to `base` when given (never silently to the server's working directory)."""
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return Path("")
    s = os.path.expanduser(s)
    cand = Path(s)
    if base is not None and not cand.is_absolute() and not is_windows_path(s):
        cand = Path(base) / s.replace("\\", "/")
    if cand.exists():
        return cand.resolve()
    norm = normalize_win_path(s)
    first_mapped: Optional[Path] = None
    for host_prefix, mount in PATH_MAPPINGS:
        for test in (s, norm):
            for pfx in (host_prefix, normalize_win_path(host_prefix)):
                rest = _prefix_rest(test, pfx)
                if rest is None:
                    continue
                mapped = Path(os.path.normpath(Path(mount) / rest.replace("\\", "/").lstrip("/")))
                if mapped.exists():
                    return mapped.resolve()
                if first_mapped is None and Path(mount).is_dir():
                    first_mapped = mapped
    if norm != s:
        m = _WIN_DRIVE_RE.match(s)
        if m:
            direct = Path(CONTAINER_MOUNT) / s[m.end():].replace("\\", "/")
            if direct.exists():
                return direct.resolve()
    if first_mapped is not None:
        return first_mapped
    return cand


def to_host(path: str | os.PathLike) -> str:
    """Map a container path back to the host form for display."""
    s = str(path)
    for host_prefix, mount in PATH_MAPPINGS:
        if s == mount or s.startswith(mount + "/"):
            return host_prefix + s[len(mount):]
    return s


def default_browse_root() -> Path:
    mount = Path(CONTAINER_MOUNT)
    if HOST_PATH and mount.is_dir():
        return mount
    return Path.home()


def nearest_existing(p: Path) -> Optional[Path]:
    """The closest existing ancestor of p (for 'folder not found' messages)."""
    cur = p
    seen = 0
    while seen < 256:
        try:
            if cur.is_dir():
                return cur
        except OSError:
            pass
        if cur.parent == cur:
            return None
        cur = cur.parent
        seen += 1
    return None


class BrowseNotFound(FileNotFoundError):
    def __init__(self, requested: str, nearest: Optional[Path]):
        super().__init__(f"Folder not found: {requested}")
        self.requested = requested
        self.nearest = nearest


BROWSE_MAX_FILES = 2000
BROWSE_MAX_DIRS = 5000


def _nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def browse(raw: str, show_files: bool = True, max_files: int = BROWSE_MAX_FILES) -> dict:
    """List one folder for the picker. os.scandir answers is_dir from the directory entry itself, so a
    60k-file folder costs a fraction of a second instead of one stat per file; files are capped (the
    totals are still reported) and image files are only counted when files are hidden."""
    if raw and raw.strip():
        root = resolve(raw, base=default_browse_root())
        if not root.exists():
            raise BrowseNotFound(raw.strip(), nearest_existing(root))
    else:
        root = default_browse_root()
    selected = None
    if root.is_file():
        selected = root.name
        root = root.parent
    dirs, files = [], []
    n_dirs = n_files = n_images = 0
    try:
        with os.scandir(root) as it:
            for e in it:
                name = e.name
                if name.startswith("."):
                    continue
                try:
                    is_dir = e.is_dir()          # follows symlinks: a linked dataset folder is a folder
                except OSError:
                    continue
                if is_dir:
                    n_dirs += 1
                    if len(dirs) < BROWSE_MAX_DIRS:
                        dirs.append(name)
                    continue
                n_files += 1
                dot = name.rfind(".")
                is_image = dot > 0 and name[dot:].lower() in IMAGE_EXTS
                n_images += is_image
                if show_files and len(files) < max_files:
                    files.append((name, e, is_image))
    except PermissionError:
        raise PermissionError(f"Permission denied: {to_host(root)}")
    dirs.sort(key=lambda n: _nfc(n).casefold())
    files.sort(key=lambda t: _nfc(t[0]).casefold())
    out_files = []
    for name, e, is_image in files:
        try:
            size = e.stat().st_size
        except OSError:
            size = None
        out_files.append({"name": _nfc(name), "path": to_host(root / name), "size": size, "is_image": is_image})
    parent = root.parent if root != root.parent else None
    return {
        "current": to_host(root),
        "parent": to_host(parent) if parent else None,
        "dirs": [{"name": _nfc(n), "path": to_host(root / n)} for n in dirs],
        "files": out_files,
        "selected": selected,
        "counts": {"dirs": n_dirs, "files": n_files, "images": n_images},
        "truncated": (show_files and n_files > len(out_files)) or n_dirs > len(dirs),
    }
