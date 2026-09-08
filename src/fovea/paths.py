"""Path resolution: host<->container mapping, Windows normalisation, browsing."""
from __future__ import annotations

import os
import re
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


PATH_MAPPINGS: List[Tuple[str, str]] = []
if HOST_PATH:
    PATH_MAPPINGS.append((HOST_PATH.rstrip("/\\"), CONTAINER_MOUNT))
    _norm = normalize_win_path(HOST_PATH).rstrip("/")
    if _norm != HOST_PATH.rstrip("/\\"):
        PATH_MAPPINGS.append((_norm, CONTAINER_MOUNT))


def resolve(raw: str) -> Path:
    """Resolve a user-supplied path (host or container form) to a local Path.
    Returns the candidate even when it does not exist so callers can report it."""
    s = (raw or "").strip().strip('"').strip("'")
    if not s:
        return Path("")
    s = os.path.expanduser(s)
    cand = Path(s)
    if cand.exists():
        return cand.resolve()
    norm = normalize_win_path(s)
    for host_prefix, mount in PATH_MAPPINGS:
        for test in (s, norm):
            for pfx in (host_prefix, normalize_win_path(host_prefix)):
                if pfx and (test == pfx or test.startswith(pfx + "/") or test.startswith(pfx + "\\")):
                    rel = test[len(pfx):].lstrip("/\\")
                    mapped = Path(mount) / rel
                    if mapped.exists():
                        return mapped.resolve()
    if norm != s:
        m = _WIN_DRIVE_RE.match(s)
        if m:
            direct = Path(CONTAINER_MOUNT) / s[m.end():].replace("\\", "/")
            if direct.exists():
                return direct.resolve()
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


def browse(raw: str, show_files: bool = True) -> dict:
    root = resolve(raw) if raw else default_browse_root()
    if not root.exists():
        root = default_browse_root()
    if root.is_file():
        root = root.parent
    dirs, files = [], []
    try:
        for child in sorted(root.iterdir(), key=lambda p: p.name.lower()):
            if child.name.startswith("."):
                continue
            try:
                if child.is_dir():
                    dirs.append({"name": child.name, "path": to_host(child)})
                elif show_files and child.is_file():
                    files.append({"name": child.name, "path": to_host(child), "size": child.stat().st_size,
                                  "is_image": child.suffix.lower() in IMAGE_EXTS})
            except OSError:
                continue
    except PermissionError:
        raise PermissionError(f"Permission denied: {to_host(root)}")
    parent = root.parent if root != root.parent else None
    return {
        "current": to_host(root),
        "parent": to_host(parent) if parent else None,
        "dirs": dirs,
        "files": files,
    }
