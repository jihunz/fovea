"""Provider registry: which providers exist, whether they can run here, and their public description.

Provider modules are imported lazily (``fovea.ai.providers.<id>``) so the app starts without torch / anthropic /
httpx, and a provider whose module or dependency is missing is listed as unavailable with the reason instead of
breaking the list. ``mock`` exists only when ``FOVEA_AI_MOCK=1``.
"""
from __future__ import annotations

import importlib
import json
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from . import config, secrets
from .base import Provider, ProviderError

ORDER = ["yolo", "yoloe", "sam", "anthropic", "gemini", "openai", "openai_compat", "mock"]
CLOUD = {"anthropic", "gemini", "openai", "openai_compat", "mock"}
NAMES = {
    "yolo": "Ultralytics YOLO", "yoloe": "YOLOE (open vocabulary)", "sam": "Segment Anything",
    "anthropic": "Anthropic Claude", "gemini": "Google Gemini", "openai": "OpenAI", "openai_compat": "OpenAI-compatible server",
    "mock": "Mock (tests)",
}
MODEL_CACHE_TTL = 600.0

_lock = threading.Lock()
_providers: Dict[str, Provider] = {}
_errors: Dict[str, str] = {}
_models_cache: Dict[str, tuple] = {}


def mock_enabled() -> bool:
    return os.environ.get("FOVEA_AI_MOCK", "").strip().lower() in ("1", "true", "yes", "on")


def ids() -> List[str]:
    return [p for p in ORDER if p != "mock" or mock_enabled()]


class Missing(Provider):
    """Placeholder for a provider whose module could not be imported."""

    def __init__(self, pid: str, reason: str):
        self.id = pid
        self.name = NAMES.get(pid, pid)
        self.kind = "cloud" if pid in CLOUD else "local"
        self.needs_key = pid in ("anthropic", "gemini", "openai")
        self.capabilities = {}
        self._reason = reason

    def status(self) -> dict:
        return {"available": False, "reason": self._reason}


def get(pid: str) -> Provider:
    """The provider instance; raises KeyError for an unknown / disabled id."""
    if pid not in ids():
        raise KeyError(pid)
    p = _providers.get(pid)
    if p is not None:
        return p
    with _lock:
        p = _providers.get(pid)
        if p is not None:
            return p
        try:
            mod = importlib.import_module(f"fovea.ai.providers.{pid}")
            p = mod.PROVIDER
            _errors.pop(pid, None)
            _providers[pid] = p
        except ModuleNotFoundError as e:
            missing = getattr(e, "name", "") or ""
            reason = ("Not installed in this build" if missing.endswith(f"providers.{pid}")
                      else f"Missing Python package: {missing or e}")
            return Missing(pid, reason)
        except Exception as e:  # noqa: BLE001 — a broken provider must not break the list
            return Missing(pid, f"Failed to load: {secrets.scrub(f'{type(e).__name__}: {e}')}")
    return p


def is_cloud(pid: str) -> bool:
    p = _providers.get(pid)
    kind = getattr(p, "kind", None) if p else None
    return (kind == "cloud") if kind else pid in CLOUD


def uses_owner_key(pid: str) -> bool:
    """A provider that counts as local (openai_compat on a loopback URL) but sends the owner's API key: typically a
    local gateway (LiteLLM, an OpenRouter proxy) that forwards to a paid cloud model. Ollama, LM Studio and llama.cpp
    need no key."""
    if is_cloud(pid) or not secrets.env_names(pid):
        return False
    try:
        return bool(secrets.get_key(pid))
    except Exception:  # noqa: BLE001 — an unreadable secrets file: treat as keyed (the guard stays on)
        return True


def is_guarded(pid: str) -> bool:
    """Requests from other machines may not call it (api.ai._guard_remote): a cloud provider, or a local one that
    sends the owner's API key."""
    return is_cloud(pid) or uses_owner_key(pid)


def safe_status(p: Provider) -> dict:
    try:
        st = p.status() or {}
    except Exception as e:  # noqa: BLE001
        st = {"available": False, "reason": secrets.scrub(f"{type(e).__name__}: {e}")}
    out = {"available": bool(st.get("available")), "reason": st.get("reason")}
    cfg = config.provider_config(p.id)
    if out["available"] and cfg.get("enabled") is False:
        out = {"available": False, "reason": "Turned off in Settings"}
    if out["available"] and getattr(p, "needs_key", False) and not secrets.get_key(p.id):
        out = {"available": False, "reason": "Add an API key in Settings"}
    for k in ("loading",):
        if k in st:
            out[k] = st[k]
    return out


def _self_caching(p: Provider) -> bool:
    """Providers that cache their live model listing themselves and re-derive per-model availability on every
    call (VLM providers: openai_compat marks models a server has not pulled from its 60 s reachability probe).
    Caching their RESULT here would freeze those marks for 10 minutes."""
    return callable(getattr(p, "merge_models", None))


def forget_models(pid: str) -> None:
    """Drop the cached model list of ``pid`` (key / config change, Test)."""
    _models_cache.pop(pid, None)


def models(pid: str, refresh: bool = False) -> List[dict]:
    """Model list; live lists are cached for 10 minutes and fall back to the static one on any error. Availability
    marks are never served from this cache (see :func:`_self_caching`)."""
    p = get(pid)
    if not refresh:
        hit = _models_cache.get(pid)
        if hit and time.time() - hit[0] < MODEL_CACHE_TTL and not _self_caching(p):
            return hit[1]
        try:
            return p.models(refresh=False) or []
        except Exception:  # noqa: BLE001
            return []
    try:
        lst = p.models(refresh=True) or []
        if not _self_caching(p):
            _models_cache[pid] = (time.time(), lst)
        return lst
    except ProviderError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ProviderError(secrets.scrub(f"Could not list models: {type(e).__name__}: {e}"), "network") from e


def model_ids(pid: str) -> List[str]:
    return [m.get("id") for m in models(pid) if m.get("id")]


def default_model(pid: str) -> Optional[str]:
    lst = models(pid)
    cfg = config.provider_config(pid)
    ids_ = [m.get("id") for m in lst]
    if cfg.get("default_model") and (cfg["default_model"] in ids_ or is_cloud(pid)):
        return cfg["default_model"]
    for m in lst:
        if m.get("default"):
            return m.get("id")
    return ids_[0] if ids_ else None


def prices_as_of() -> Optional[str]:
    p = Path(__file__).with_name("prices.json")
    try:
        data = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data.get("as_of") if isinstance(data, dict) else None


def entry(pid: str) -> dict:
    p = get(pid)
    kind = getattr(p, "kind", "local") or "local"
    needs_key = bool(getattr(p, "needs_key", False))
    key = secrets.key_info(pid) if (needs_key or pid in secrets.ENV_KEYS) else None
    if key is not None:
        key = {k: key[k] for k in ("present", "source", "hint", "env")}
    try:
        lst = models(pid)
    except Exception:  # noqa: BLE001
        lst = []
    cfg = config.provider_config(pid)
    d_model = None
    try:
        d_model = default_model(pid)
    except Exception:  # noqa: BLE001
        pass
    lst = [{**m, "default": m.get("id") == d_model} for m in lst]
    location = getattr(p, "location", "") or ("Runs on this machine — images never leave it" if kind == "local"
                                              else "Images are sent to the provider")
    if kind == "local" and key and key.get("present"):
        # A local server that takes the owner's key is most likely a gateway to a paid model: never "stay here".
        host = ""
        try:
            from .vlm import url_host
            host = url_host(p.base_url()) if hasattr(p, "base_url") else ""
        except Exception:  # noqa: BLE001
            host = ""
        location = (f"Images go to {host or 'this server'} with your API key — a gateway there may forward them to "
                    "a paid cloud model (only this computer can use it)")
    return {
        "id": pid, "name": getattr(p, "name", "") or NAMES.get(pid, pid), "kind": kind,
        "location": location, "guarded": is_guarded(pid),
        "needs_key": needs_key, "capabilities": dict(getattr(p, "capabilities", {}) or {}),
        "status": safe_status(p), "key": key, "config": cfg, "models": lst, "default_model": d_model,
        "endpoint": _endpoint(p, cfg),
    }


def _endpoint(p: Any, cfg: dict) -> Optional[dict]:
    """Where an HTTP provider sends requests now, and why: Settings, its FOVEA_<PROVIDER>_BASE_URL, or the built-in
    default (Settings shows an empty URL field otherwise). Never the URL's ``user:password@``, query or fragment."""
    if not callable(getattr(p, "base_url", None)):
        return None
    try:
        from urllib.parse import urlparse
        from .vlm import redact_url
        env = getattr(p, "base_url_env", "") or ""
        src = ("settings" if str((cfg or {}).get("base_url") or "").strip()
               else "env" if env and os.environ.get(env, "").strip() else "default")
        url = urlparse(redact_url(p.base_url()))._replace(query="", fragment="").geturl()
        return {"url": url, "source": src, "env": env or None}
    except Exception:  # noqa: BLE001
        return None


def remote_ai_allowed() -> bool:
    return os.environ.get("FOVEA_ALLOW_REMOTE_AI", "").strip().lower() in ("1", "true", "yes", "on")


def reset() -> None:
    """Tests: forget imported providers and cached model lists."""
    with _lock:
        _providers.clear()
        _models_cache.clear()
