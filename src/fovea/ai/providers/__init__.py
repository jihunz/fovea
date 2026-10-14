"""Concrete AI providers. Each module exposes ``PROVIDER`` (an instance of :class:`fovea.ai.base.Provider`).

The registry (``fovea.ai.registry``) imports a module only when its provider is first needed, so a missing optional
dependency (torch, anthropic, httpx) never breaks the app. Conventions for provider modules:

* key: ``from ..secrets import get_key`` → ``get_key(self.id)`` at call time (never cache it on the instance);
* settings: ``from ..config import provider_config`` → ``provider_config(self.id)`` (base_url, default_model, effort,
  concurrency, coord, …) at call time;
* errors: raise :class:`fovea.ai.base.ProviderError` with a ``code``; never put a key, header or full URL with a
  query string in the message (callers also pass every message through ``secrets.scrub``).
"""
