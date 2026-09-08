"""Fovea application factory."""
from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from . import __version__, db
from .api import ai, datasets, export, fs, images, jobs as jobs_api, meta
from .config import CONTAINER_MOUNT, HOST_PATH, STATIC_DIR, TEMPLATE_DIR
from .plugin_api import discover

app = FastAPI(title="Fovea", version=__version__, docs_url="/api/docs", redoc_url=None, openapi_url="/api/openapi.json")
templates = Jinja2Templates(directory=str(TEMPLATE_DIR))

db.init_db()
# recover datasets interrupted mid-scan by a previous process
db.execute("UPDATE datasets SET status = CASE WHEN image_count > 0 THEN 'ready' ELSE 'new' END WHERE status='scanning'")

app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")
for r in (meta.router, fs.router, datasets.router, images.router, ai.router, export.router, jobs_api.router):
    app.include_router(r)

PLUGINS = discover()
for p in PLUGINS:
    if p.router is not None:
        app.include_router(p.router, prefix=f"/api/plugins/{p.id}", tags=[f"plugin:{p.id}"])
    if p.static_dir and Path(p.static_dir).is_dir():
        app.mount(f"/plugins/{p.id}", StaticFiles(directory=str(p.static_dir)), name=f"plugin-{p.id}")
meta.set_plugins(PLUGINS)


@app.exception_handler(Exception)
async def _unhandled(request: Request, exc: Exception):
    return JSONResponse(status_code=500, content={"detail": f"{type(exc).__name__}: {exc}"})


def _shell(request: Request) -> HTMLResponse:
    config = {
        "version": __version__,
        "hostPath": HOST_PATH, "containerMount": CONTAINER_MOUNT,
        "plugins": [p.manifest() for p in PLUGINS],
    }
    return templates.TemplateResponse(request, "app.html", {"config_json": json.dumps(config)})


FAVICON_SVG = ("<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 32 32'><circle cx='16' cy='16' r='12' fill='none' stroke='#0f766e' stroke-width='3.5'/>"
               "<circle cx='16' cy='16' r='4.5' fill='#0f766e'/></svg>")


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    from fastapi.responses import Response
    return Response(FAVICON_SVG, media_type="image/svg+xml", headers={"Cache-Control": "public, max-age=86400"})


@app.get("/", response_class=HTMLResponse)
def index(request: Request):
    return _shell(request)


@app.get("/settings", response_class=HTMLResponse)
def settings_page(request: Request):
    return _shell(request)


@app.get("/d/{rest:path}", response_class=HTMLResponse)
def dataset_pages(request: Request, rest: str):
    return _shell(request)
