"""API keys for cloud AI providers.

Precedence: environment variable > ``DATA_DIR/secrets.json`` (mode 0600, written atomically). Keys never leave the
server: endpoints only ever see :func:`key_info` (``{present, source, hint}``), and :func:`scrub` removes anything
key-shaped from text that is about to reach a client, a log line or a job result.

Provider modules read their key with ``get_key(self.id)`` at call time (never cache it on the instance), so a key the
user saves or removes in Settings takes effect on the next request.
"""
from __future__ import annotations

import json
import os
import re
import tempfile
import threading
from pathlib import Path
from typing import Dict, Iterable, Optional

from ..config import DATA_DIR

# Environment variables that may hold each provider's key; the first one set wins.
ENV_KEYS: Dict[str, tuple] = {
    "anthropic": ("ANTHROPIC_API_KEY",),
    "gemini": ("GEMINI_API_KEY", "GOOGLE_API_KEY"),
    "openai": ("OPENAI_API_KEY",),
    "openai_compat": ("OPENAI_COMPAT_API_KEY",),
}

MAX_KEY_LEN = 512
_lock = threading.RLock()


def _path() -> Path:
    return Path(DATA_DIR) / "secrets.json"


def _load() -> Dict[str, str]:
    p = _path()
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {str(k): str(v) for k, v in data.items() if isinstance(v, str) and v} if isinstance(data, dict) else {}


def _save(data: Dict[str, str]) -> None:
    p = _path()
    p.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=".secrets-", suffix=".json", dir=str(p.parent))
    try:
        os.fchmod(fd, 0o600)                      # never readable by others, not even for an instant
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=1, sort_keys=True)
        os.replace(tmp, p)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)
    try:
        os.chmod(p, 0o600)
    except OSError:
        pass


def env_names(provider: str) -> tuple:
    return ENV_KEYS.get(provider, ())


def _from_env(provider: str) -> Optional[str]:
    for name in env_names(provider):
        v = (os.environ.get(name) or "").strip()
        if v:
            return v
    return None


def get_key(provider: str) -> Optional[str]:
    """The key to use for ``provider`` right now, or None."""
    v = _from_env(provider)
    if v:
        return v
    with _lock:
        return _load().get(provider) or None


def hint(key: Optional[str]) -> Optional[str]:
    """'…ab12' — enough to tell two keys apart, never enough to use one."""
    if not key:
        return None
    return "…" + key[-4:] if len(key) >= 12 else "…"


def key_info(provider: str) -> dict:
    v = _from_env(provider)
    if v:
        names = [n for n in env_names(provider) if (os.environ.get(n) or "").strip()]
        return {"present": True, "source": "env", "hint": hint(v), "env": names[0] if names else None}
    with _lock:
        stored = _load().get(provider)
    if stored:
        return {"present": True, "source": "file", "hint": hint(stored), "env": None}
    return {"present": False, "source": None, "hint": None, "env": None}


def validate_key(key: str) -> str:
    key = (key or "").strip()
    if not key:
        raise ValueError("The key is empty")
    if len(key) > MAX_KEY_LEN:
        raise ValueError("The key is too long")
    if any(c.isspace() for c in key) or not key.isprintable():
        raise ValueError("The key contains spaces or control characters")
    return key


def set_key(provider: str, key: str) -> dict:
    key = validate_key(key)
    with _lock:
        data = _load()
        data[provider] = key
        _save(data)
    return key_info(provider)


def clear_key(provider: str) -> dict:
    with _lock:
        data = _load()
        if provider in data:
            data.pop(provider)
            _save(data)
    return key_info(provider)


def all_keys() -> Iterable[str]:
    """Every key currently known (env + file) — only for scrubbing."""
    out = []
    for p in set(ENV_KEYS) | set(_load()):
        k = get_key(p)
        if k and len(k) >= 8:
            out.append(k)
    return out


_KEY_PATTERNS = [
    re.compile(r"sk-ant-[A-Za-z0-9_\-]{8,}"),
    re.compile(r"sk-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"AIza[0-9A-Za-z_\-]{20,}"),
    re.compile(r"(?i)(bearer\s+)[A-Za-z0-9._\-+/=]{8,}"),
    re.compile(r"(?i)((?:x-api-key|x-goog-api-key|api[_-]?key|authorization)[\"']?\s*[:=]\s*[\"']?)[^\s\"',}]{6,}"),
    re.compile(r"(?i)([?&]key=)[^&\s\"']+"),
]


def scrub(text) -> str:
    """Remove anything key-shaped (and every known key verbatim) from ``text``."""
    s = "" if text is None else str(text)
    if not s:
        return s
    for k in all_keys():
        if k in s:
            s = s.replace(k, "[redacted]")
    for pat in _KEY_PATTERNS:
        if pat.groups:
            s = pat.sub(lambda m: m.group(1) + "[redacted]", s)
        else:
            s = pat.sub("[redacted]", s)
    return s
