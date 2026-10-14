"""Fovea application factory."""
from __future__ import annotations

import asyncio
import base64
import gzip
import hashlib
import io
import json
import logging
import math
import os
import re
import time
import types
import typing
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.exception_handlers import http_exception_handler
from fastapi.exceptions import RequestValidationError
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.routing import APIRoute
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import TypeAdapter
from starlette.datastructures import Headers, MutableHeaders

from . import __version__, db, security
from .api import ai, datasets, export, fs, images, jobs as jobs_api, meta, objects, qa, snapshots
from .config import CONTAINER_MOUNT, HOST_PATH, STATIC_DIR, TEMPLATE_DIR
from .jobs import jobs
from .plugin_api import discover

log = logging.getLogger("fovea")
security.install_log_redaction()

_asset_cache = {"at": 0.0, "v": ""}


def asset_version() -> str:
    """Content version of the static files (path + mtime + size), recomputed at most every 2 s so an edited
    file gets a new version on the next page load."""
    now = time.monotonic()
    if now - _asset_cache["at"] < 2.0 and _asset_cache["v"]:
        return _asset_cache["v"]
    h = hashlib.sha1()
    roots = [STATIC_DIR] + [Path(p.static_dir) for p in globals().get("PLUGINS", []) if p.static_dir]
    for root in roots:
        for dirpath, _dirs, files in os.walk(root):
            for f in sorted(files):
                try:
                    st = os.stat(os.path.join(dirpath, f))
                except OSError:
                    continue
                h.update(f"{dirpath}/{f}|{st.st_mtime_ns}|{st.st_size}".encode("utf-8", "surrogateescape"))
    _asset_cache.update(at=now, v=h.hexdigest()[:12])
    return _asset_cache["v"]


class VersionedStatic(StaticFiles):
    """A URL carrying the current asset version (?v=...) is cached for good; anything else revalidates
    (ETag/Last-Modified), so UI updates are still picked up immediately."""

    def file_response(self, full_path, stat_result, scope, status_code=200):
        resp = super().file_response(full_path, stat_result, scope, status_code)
        qs = scope.get("query_string", b"").decode("latin-1")
        v = next((x[2:] for x in qs.split("&") if x.startswith("v=")), "")
        resp.headers["Cache-Control"] = "public, max-age=31536000, immutable" if v and v == asset_version() else "no-cache"
        return resp


RevalidatingStatic = VersionedStatic       # former name, kept for plugins that import it


_COMPRESSIBLE = ("application/json", "text/html", "text/css", "text/javascript", "application/javascript",
                 "text/plain", "image/svg+xml", "text/csv", "application/x-yaml", "text/yaml")


class SelectiveGZip:
    """gzip for text responses only (JSON lists, HTML, JS/CSS). Images, ZIP downloads and SSE pass through
    untouched: compressing JPEGs wastes CPU, and a gzipped event stream would be buffered."""

    def __init__(self, app, minimum_size: int = 1024, compresslevel: int = 5):
        self.app, self.minimum_size, self.compresslevel = app, minimum_size, compresslevel

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or "gzip" not in Headers(scope=scope).get("accept-encoding", ""):
            return await self.app(scope, receive, send)
        st = {"mode": "plain", "start": None, "gz": None, "buf": None}

        async def route(message):
            kind = message["type"]
            if kind == "http.response.start":
                h = Headers(raw=message.get("headers") or [])
                if h.get("content-type", "").lower().startswith(_COMPRESSIBLE) and "content-encoding" not in h:
                    st.update(mode="pending", start=message)      # decided by the first body chunk
                    return
                st["mode"] = "plain"
                return await send(message)
            if kind != "http.response.body" or st["mode"] == "plain":
                return await send(message)
            body, more = message.get("body", b""), message.get("more_body", False)
            if st["mode"] == "pending":
                start = st["start"]
                if not more and len(body) < self.minimum_size:
                    st["mode"] = "plain"
                    await send(start)
                    return await send(message)
                buf = io.BytesIO()
                st.update(mode="gzip", buf=buf, gz=gzip.GzipFile(mode="wb", fileobj=buf, compresslevel=self.compresslevel, mtime=0))
                headers = MutableHeaders(scope=start)
                headers["Content-Encoding"] = "gzip"
                headers.add_vary_header("Accept-Encoding")
                if "content-length" in headers:
                    del headers["content-length"]
                if not more:
                    st["gz"].write(body)
                    st["gz"].close()
                    data = buf.getvalue()
                    headers["Content-Length"] = str(len(data))
                    await send(start)
                    return await send({"type": "http.response.body", "body": data, "more_body": False})
                await send(start)
            gz, buf = st["gz"], st["buf"]
            gz.write(body)
            if not more:
                gz.close()
            data = buf.getvalue()
            buf.seek(0)
            buf.truncate()
            await send({"type": "http.response.body", "body": data, "more_body": more})

        await self.app(scope, receive, route)


INT64_MIN, INT64_MAX = -(1 << 63), (1 << 63) - 1
_INT = TypeAdapter(int)
_BIG_NUMBER = re.compile(rb"\d{19}|[eE][+]?\d{2}|Infinity|NaN")    # cheap pre-check before parsing a body
_route_ints: Dict[int, Tuple[List[str], List[str]]] = {}


def _int_annotation(ann: Any) -> bool:
    if ann is int:
        return True
    origin = typing.get_origin(ann)
    if origin is typing.Annotated:
        return _int_annotation(typing.get_args(ann)[0])
    if origin in (typing.Union, types.UnionType, list, set, frozenset, tuple):
        return any(_int_annotation(a) for a in typing.get_args(ann) if a is not type(None) and a is not Ellipsis)
    return False


def _flat_params(dependant: Any) -> Tuple[list, list]:
    """(path params, query params) of a route's dependant and all of its sub-dependencies. Walked here because
    fastapi.dependencies.utils.get_flat_dependant is internal and gone from newer FastAPI releases (the server
    did not start on them); the Dependant fields read here are the same from 0.100 to the current release."""
    path: list = []
    query: list = []
    stack, seen = [dependant], set()
    while stack:
        dep = stack.pop()
        if dep is None or id(dep) in seen:
            continue
        seen.add(id(dep))
        path.extend(getattr(dep, "path_params", None) or ())
        query.extend(getattr(dep, "query_params", None) or ())
        stack.extend(getattr(dep, "dependencies", None) or ())
    return path, query


def _int_params(route: APIRoute) -> Tuple[List[str], List[str]]:
    """(path names, query aliases) of a route's integer parameters, computed once per route."""
    hit = _route_ints.get(id(route))
    if hit is None:
        path, query = _flat_params(route.dependant)
        hit = tuple(list(dict.fromkeys(f.alias for f in fields if _int_annotation(f.field_info.annotation)))
                    for fields in (path, query))
        _route_ints[id(route)] = hit
    return hit


def _parsed_int(v: Any) -> Optional[int]:
    try:
        return _INT.validate_python(v)      # what FastAPI's own validation would make of it
    except ValueError:
        return None                         # not an int at all: FastAPI reports that itself


def _range_error(loc: tuple, value: float, raw: Any) -> dict:
    """The error FastAPI itself returns for a `ge=INT64_MIN, le=INT64_MAX` parameter."""
    lo = value < 0
    bound = INT64_MIN if lo else INT64_MAX
    return {"type": "greater_than_equal" if lo else "less_than_equal", "loc": loc,
            "msg": f"Input should be {'greater' if lo else 'less'} than or equal to {bound}",
            "input": str(raw), "ctx": {"ge" if lo else "le": bound}}


def _json_out_of_range(doc: Any, limit: int = 10) -> List[dict]:
    """Errors for numbers in a JSON document that no 64-bit integer can hold (iterative: nesting depth is the
    client's choice)."""
    out: List[dict] = []
    stack = [(doc, ("body",))]
    while stack and len(out) < limit:
        node, loc = stack.pop()
        if isinstance(node, dict):
            stack.extend((v, loc + (k,)) for k, v in reversed(list(node.items())))
        elif isinstance(node, list):
            stack.extend((v, loc + (i,)) for i, v in reversed(list(enumerate(node))))
        elif isinstance(node, bool):
            continue
        elif isinstance(node, int):
            if not INT64_MIN <= node <= INT64_MAX:
                out.append(_range_error(loc, node, node))
        elif isinstance(node, float) and not (math.isfinite(node) and INT64_MIN < node < INT64_MAX):
            # A whole float (1e30) is accepted where an int is expected, so it has to fit as well.
            out.append(_range_error(loc, node, node) if math.isfinite(node) else
                       {"type": "finite_number", "loc": loc, "msg": "Input should be a finite number", "input": str(node)})
    return out


async def int_range_guard(request: Request) -> None:
    """Every integer the API accepts must fit SQLite's 64-bit INTEGER: a larger one used to reach a query
    and fail parameter binding with a 500 (OverflowError). Checked once here for every route, plugins
    included: an integer path id out of range is 404 (no such row can exist), a query parameter or a JSON
    body number out of range is a 422 shaped exactly like FastAPI's own `Query(le=...)` errors. Strings that
    hold numbers (`cls=1,2`, filters) are clamped where they are parsed."""
    route = request.scope.get("route")
    if isinstance(route, APIRoute):
        path_names, query_names = _int_params(route)
        for name in path_names:
            n = _parsed_int(request.path_params.get(name))
            if n is not None and not INT64_MIN <= n <= INT64_MAX:
                raise HTTPException(404, "Not found")
        errors = []
        for name in query_names:
            for raw in request.query_params.getlist(name):
                n = _parsed_int(raw)
                if n is not None and not INT64_MIN <= n <= INT64_MAX:
                    errors.append(_range_error(("query", name), n, raw))
        if errors:
            raise RequestValidationError(errors)
    if request.method in security.UNSAFE_METHODS and "json" in request.headers.get("content-type", "").lower():
        body = await request.body()             # cached by Starlette: the endpoint reads the same bytes
        if body and _BIG_NUMBER.search(body):
            try:
                doc = json.loads(body)
            except (ValueError, RecursionError):
                return                          # FastAPI answers malformed JSON itself
            errors = _json_out_of_range(doc)
            if errors:
                raise RequestValidationError(errors)


@asynccontextmanager
async def lifespan(_app):
    ai.startup()            # ai-core hook: re-attach Economy batch runs (background thread; nothing at import time)
    yield
    # Cancel queued/running jobs and stop the worker pools, so Ctrl+C / docker stop take seconds, not the
    # remaining length of an export or an AI batch.
    await asyncio.to_thread(jobs.shutdown, 3.0)


# /api/docs is served by api_docs() below (pinned Swagger UI with SRI + CSP), not by FastAPI's default page.
app = FastAPI(title="Fovea", version=__version__, docs_url=None, redoc_url=None, openapi_url="/api/openapi.json",
              lifespan=lifespan, dependencies=[Depends(int_range_guard)])     # also applies to included routers
app.add_middleware(SelectiveGZip)
app.add_middleware(security.GuardMiddleware)        # added last = outermost: refuses before anything runs
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

db.init_db()
# recover datasets interrupted mid-scan by a previous process
db.execute("UPDATE datasets SET status = CASE WHEN image_count > 0 THEN 'ready' ELSE 'new' END WHERE status='scanning'")

app.mount("/static", VersionedStatic(directory=str(STATIC_DIR)), name="static")
for r in (meta.router, fs.router, datasets.router, images.router, ai.router, export.router, jobs_api.router,
          qa.router, objects.router, snapshots.router):
    app.include_router(r)

PLUGINS = discover()
for p in PLUGINS:
    if p.router is not None:
        app.include_router(p.router, prefix=f"/api/plugins/{p.id}", tags=[f"plugin:{p.id}"])
    if p.static_dir and Path(p.static_dir).is_dir():
        app.mount(f"/plugins/{p.id}", VersionedStatic(directory=str(p.static_dir)), name=f"plugin-{p.id}")
meta.set_plugins(PLUGINS)


@app.exception_handler(HTTPException)
async def _http_error(request: Request, exc: HTTPException):
    # Error details can quote provider responses or URLs: never let a key ride along to the browser.
    if isinstance(exc.detail, str):
        exc.detail = security.scrub(exc.detail)
    return await http_exception_handler(request, exc)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    # Starlette re-raises after this response, so uvicorn logs the traceback (through the redacting filter).
    return JSONResponse(status_code=500, content={"detail": security.scrub(f"{type(exc).__name__}: {exc}")[:500]})


def _shell(request: Request) -> HTMLResponse:
    v = asset_version()
    config = {
        "version": __version__, "assetVersion": v,
        "hostPath": HOST_PATH, "containerMount": CONTAINER_MOUNT,
        "plugins": [p.manifest() for p in PLUGINS],
    }
    resp = templates.TemplateResponse(request, "app.html", {"config_json": json.dumps(config), "asset_version": v})
    resp.headers["Cache-Control"] = "no-cache"
    return resp


FAVICON_SVG = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='12' fill='none' stroke='#0f766e' stroke-width='3.5'/>"
               "<circle cx='16' cy='16' r='4.5' fill='#0f766e'/></svg>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


# The API docs page runs Swagger UI on Fovea's own origin, where any script passes the Origin check and can drive the
# whole API (point a cloud provider at another host, then send the owner's key there). FastAPI's default page loads a
# floating 'swagger-ui-dist@5' from a CDN without integrity: so the release is pinned, each file carries its SRI hash
# (from jsDelivr's package listing), and a CSP admits only those two files plus the init script below.
# To upgrade: change the version and take both hashes from
# https://data.jsdelivr.com/v1/packages/npm/swagger-ui-dist@<version>?structure=flat ("sha256-" + hash).
SWAGGER_UI_VERSION = "5.32.15"
_SWAGGER_CDN = f"https://cdn.jsdelivr.net/npm/swagger-ui-dist@{SWAGGER_UI_VERSION}/"
_SWAGGER_SRI = {"swagger-ui-bundle.js": "sha256-p+NE8ncLLwdSfOgo4JUWJpg7jy3Nt6gmaJwCMgI/mVs=",
                "swagger-ui.css": "sha256-1/OfdkqhjHtH3QW5r1YT43PkrA81V8JpPVLQq8JGTXY="}
_SWAGGER_INIT = ("window.ui = SwaggerUIBundle({url: '/api/openapi.json', dom_id: '#swagger-ui', layout: 'BaseLayout', "
                 "deepLinking: true, showExtensions: true, showCommonExtensions: true, validatorUrl: null, "
                 "presets: [SwaggerUIBundle.presets.apis, SwaggerUIBundle.SwaggerUIStandalonePreset]});")
_SWAGGER_CSP = "; ".join([
    "default-src 'none'",
    f"script-src {_SWAGGER_CDN}swagger-ui-bundle.js 'sha256-"
    f"{base64.b64encode(hashlib.sha256(_SWAGGER_INIT.encode()).digest()).decode()}'",
    f"style-src {_SWAGGER_CDN}swagger-ui.css 'unsafe-inline'",       # Swagger UI sets style attributes
    "img-src 'self' data:", "font-src 'self' data:", "connect-src 'self'",
    "base-uri 'none'", "form-action 'none'", "frame-ancestors 'none'",
])
_SWAGGER_HTML = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fovea API</title><link rel="icon" href="/favicon.ico">
<link rel="stylesheet" href="{_SWAGGER_CDN}swagger-ui.css" integrity="{_SWAGGER_SRI['swagger-ui.css']}" crossorigin="anonymous">
</head><body><div id="swagger-ui"></div>
<script src="{_SWAGGER_CDN}swagger-ui-bundle.js" integrity="{_SWAGGER_SRI['swagger-ui-bundle.js']}" crossorigin="anonymous"></script>
<script>{_SWAGGER_INIT}</script>
</body></html>
"""


@app.get("/api/docs", include_in_schema=False)
def api_docs():
    return HTMLResponse(_SWAGGER_HTML, headers={"Content-Security-Policy": _SWAGGER_CSP, "Cache-Control": "no-cache"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _shell(request)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return _shell(request)


@app.get("/d/{rest:path}", response_class=HTMLResponse)
def dataset_pages(request: Request, rest: str):
    return _shell(request)
