"""Non-secret AI configuration, stored in the ``kv`` table.

* ``ai.providers`` — ``{provider_id: {base_url, default_model, effort, enabled, concurrency, spend_cap_usd,
  batch_mode, coord}}`` (only keys the user changed are stored; :func:`provider_config` fills defaults).
* ``ai.settings`` — process-wide settings: ``{device: auto|cpu|mps|cuda, keep_models: 1..4, folder_roles: ai|names}``
  (``folder_roles``: how a dataset's image / label folders with other names than images/ and labels/ are found).

Keys never live here (see ``secrets.py``); anything named like a key/secret/token is rejected outright.
Provider modules read their settings with ``provider_config(self.id)`` at call time.
"""
from __future__ import annotations

import math
import re
import threading
from typing import Any, Dict, Optional
from urllib.parse import urlparse

from .. import db

PROVIDERS_KEY = "ai.providers"
SETTINGS_KEY = "ai.settings"

EFFORTS = ("low", "medium", "high")
COORDS = ("auto", "qwen3", "qwen25", "gemini", "norm")
DEVICES = ("auto", "cpu", "mps", "cuda")
FOLDER_ROLES = ("ai", "names")

DEFAULTS: Dict[str, Any] = {
    "base_url": "", "default_model": "", "effort": "low", "enabled": True, "concurrency": 4,
    "spend_cap_usd": None, "batch_mode": False, "coord": "auto",
}
SETTINGS_DEFAULTS: Dict[str, Any] = {"device": "auto", "keep_models": 2, "folder_roles": "ai"}

_SECRETISH = re.compile(r"(?i)(key|secret|token|password|passwd|credential)")
_lock = threading.Lock()


class ConfigError(ValueError):
    pass


def _clean_value(name: str, v: Any) -> Any:
    if name == "base_url":
        s = str(v or "").strip()
        if not s:
            return ""
        if len(s) > 512:
            raise ConfigError("base_url is too long")
        u = urlparse(s)
        if u.scheme not in ("http", "https") or not u.netloc:
            raise ConfigError("base_url must be an http(s) URL, e.g. http://localhost:11434/v1")
        if u.username or u.password:
            raise ConfigError("Put credentials in the key field, not in the URL")
        if u.query or u.fragment:
            raise ConfigError("base_url must not contain a query string")
        return s.rstrip("/")
    if name == "default_model":
        s = str(v or "").strip()
        if len(s) > 160 or (s and not re.fullmatch(r"[\w.:/@+\-]+", s)):
            raise ConfigError("default_model is not a valid model id")
        return s
    if name == "effort":
        s = str(v or "").strip().lower()
        s = {"fast": "low", "accurate": "high"}.get(s, s)
        if s not in EFFORTS:
            raise ConfigError(f"effort must be one of {', '.join(EFFORTS)}")
        return s
    if name in ("enabled", "batch_mode"):
        if not isinstance(v, bool):
            raise ConfigError(f"{name} must be true or false")
        return v
    if name == "concurrency":
        try:
            n = int(v)
        except (TypeError, ValueError):
            raise ConfigError("concurrency must be an integer") from None
        if not 1 <= n <= 16:
            raise ConfigError("concurrency must be between 1 and 16")
        return n
    if name == "spend_cap_usd":
        if v is None or v == "":
            return None
        try:
            f = float(v)
        except (TypeError, ValueError):
            raise ConfigError("spend_cap_usd must be a number") from None
        if not math.isfinite(f) or f < 0:
            raise ConfigError("spend_cap_usd must be zero or more")
        return round(f, 4)
    if name == "coord":
        s = str(v or "").strip().lower()
        if s not in COORDS:
            raise ConfigError(f"coord must be one of {', '.join(COORDS)}")
        return s
    raise ConfigError(f"Unknown setting: {name}")


def _all() -> Dict[str, Dict[str, Any]]:
    data = db.kv_get(PROVIDERS_KEY, {}) or {}
    return data if isinstance(data, dict) else {}


def stored_config(provider: str) -> Dict[str, Any]:
    cur = _all().get(provider) or {}
    return {k: v for k, v in cur.items() if k in DEFAULTS}


def provider_config(provider: str) -> Dict[str, Any]:
    """Defaults merged with what the user saved. Always safe to send to the client."""
    return {**DEFAULTS, **stored_config(provider)}


def update_provider_config(provider: str, patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        raise ConfigError("Expected an object")
    for k in patch:
        if _SECRETISH.search(str(k)):
            raise ConfigError("Keys are saved with the key field (PUT …/key), never as settings")
        if k not in DEFAULTS:
            raise ConfigError(f"Unknown setting: {k}")
    clean = {k: _clean_value(k, v) for k, v in patch.items()}
    with _lock:
        data = _all()
        cur = {k: v for k, v in (data.get(provider) or {}).items() if k in DEFAULTS}
        cur.update(clean)
        data[provider] = cur
        db.kv_set(PROVIDERS_KEY, data)
    return provider_config(provider)


def ai_settings() -> Dict[str, Any]:
    data = db.kv_get(SETTINGS_KEY, {}) or {}
    data = data if isinstance(data, dict) else {}
    return {**SETTINGS_DEFAULTS, **{k: v for k, v in data.items() if k in SETTINGS_DEFAULTS}}


def update_ai_settings(patch: Dict[str, Any]) -> Dict[str, Any]:
    if not isinstance(patch, dict):
        raise ConfigError("Expected an object")
    clean: Dict[str, Any] = {}
    for k, v in patch.items():
        if k == "device":
            s = str(v or "auto").strip().lower()
            if s not in DEVICES:
                raise ConfigError(f"device must be one of {', '.join(DEVICES)}")
            clean[k] = s
        elif k == "keep_models":
            try:
                n = int(v)
            except (TypeError, ValueError):
                raise ConfigError("keep_models must be an integer") from None
            if not 1 <= n <= 4:
                raise ConfigError("keep_models must be between 1 and 4")
            clean[k] = n
        elif k == "folder_roles":
            s = str(v or "ai").strip().lower()
            if s not in FOLDER_ROLES:
                raise ConfigError(f"folder_roles must be one of {', '.join(FOLDER_ROLES)}")
            clean[k] = s
        else:
            raise ConfigError(f"Unknown setting: {k}")
    with _lock:
        data = {**ai_settings(), **clean}
        db.kv_set(SETTINGS_KEY, data)
    return data


def spend_cap(provider: str) -> Optional[float]:
    return provider_config(provider).get("spend_cap_usd")
