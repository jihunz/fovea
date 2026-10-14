"""Test helpers that behave the same on every FastAPI / Starlette release requirements.txt allows (backend-io).

- ``iter_routes(app)``: every route the app serves with its full path. FastAPI 0.14x keeps ``include_router()`` lazy:
  ``app.routes`` then holds one router object per include (its routes unprefixed inside it) instead of a prefixed
  copy of each route, so ``for r in app.routes: r.path`` silently misses every router-defined endpoint there.
- ``client_from(app, addr)``: a TestClient whose requests come from ``addr``. ``TestClient(client=...)`` needs
  Starlette 0.42+, and FastAPI 0.115.x brings 0.41.
"""
from __future__ import annotations

from typing import Any, Iterator


def iter_routes(app: Any) -> Iterator[tuple[str, Any]]:
    """(full path, route) for every endpoint of ``app`` (an app, a router or a list of routes), through
    ``include_router()`` and ``Mount`` at any depth. A mount without routes (StaticFiles) yields ``<path>/{path:path}``."""
    yield from _walk(getattr(app, "routes", app), "")


def _walk(routes: Any, prefix: str) -> Iterator[tuple[str, Any]]:
    for r in routes:
        expand = getattr(r, "effective_route_contexts", None)
        if callable(expand):                       # FastAPI 0.14x include_router(): contexts carry the prefixed path
            for ctx in expand():
                eff = getattr(ctx, "starlette_route", None)
                path = getattr(eff, "path", None) or getattr(ctx, "path", "")
                yield from _one(getattr(ctx, "original_route", r), prefix + path)
        else:
            yield from _one(r, prefix + (getattr(r, "path", "") or ""))


def _one(route: Any, path: str) -> Iterator[tuple[str, Any]]:
    from starlette.routing import Host, Mount
    if isinstance(route, Mount):
        sub = list(route.routes or [])
        if sub:
            yield from _walk(sub, path.rstrip("/"))
        else:
            yield path.rstrip("/") + "/{path:path}", route
    elif isinstance(route, Host):
        yield from _walk(route.routes or [], path)
    elif path:
        yield path, route


def route_paths(app: Any) -> set[str]:
    return {p for p, _ in iter_routes(app)}


def client_from(app: Any, addr: tuple[str, int], **kw: Any):
    """TestClient(app, **kw) whose requests (HTTP and websocket) report ``addr`` as the client address."""
    from starlette.testclient import TestClient     # what fastapi.testclient re-exports

    async def as_client(scope, receive, send):
        if scope["type"] in ("http", "websocket"):
            scope = dict(scope, client=tuple(addr))
        await app(scope, receive, send)

    return TestClient(as_client, **kw)
