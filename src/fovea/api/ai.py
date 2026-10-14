"""AI endpoints: providers/keys/config, weights, propose/preview/estimate/batch, classify/refine, audit, benchmark,
runs/usage, and the per-dataset proposal store (contract: SPEC §3.4).

Errors are JSON ``{detail, code}`` (``code`` = ProviderError code). Keys never appear in any response, error or
run record; cloud providers and key changes are refused to non-local clients unless FOVEA_ALLOW_REMOTE_AI=1.
"""
from __future__ import annotations

import math
import re
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query, Request
from fastapi.responses import JSONResponse

from .. import db
from ..ai import assist, economy
from ..ai import config as ai_config
from ..ai import estimate as ai_estimate
from ..ai import local, pipeline, registry, runner, secrets, store, weights
from ..ai.base import ProviderError, Usage
from ..ai.base import billed_usage  # ai-cloud hook: failed calls that were billed
from ..core import imagesq
from ..jobs import jobs
from .common import filters_from, get_dataset_or_404

router = APIRouter(tags=["ai"])

STATUS = {"invalid": 422, "unavailable": 503, "auth": 502, "quota": 402, "rate": 429, "overloaded": 503,
          "refusal": 502, "bad_output": 502, "network": 502, "too_large": 422}
MAX_IDS = 200_000

try:
    from ..security import is_local_request as _is_local, register_secret as _register_secret
except ImportError:  # pragma: no cover — security.py not present in this tree
    def _is_local(request) -> bool:  # type: ignore
        return True

    def _register_secret(value) -> None:  # type: ignore
        return None

REMOTE_MESSAGE = ("Cloud AI providers and API keys can only be used from the computer Fovea runs on. "
                  "Start Fovea with FOVEA_ALLOW_REMOTE_AI=1 to allow other machines.")


def _register_known_keys() -> None:
    for pid in secrets.ENV_KEYS:
        try:
            k = secrets.get_key(pid)
        except Exception:  # noqa: BLE001
            k = None
        if k:
            _register_secret(k)


_register_known_keys()


def startup() -> None:
    """Called once by the app's startup (``main.py`` lifespan), never at import: Economy (provider batch) runs keep
    going at the provider while Fovea is stopped — re-attach them and add the late AI columns, in the background."""
    economy.start_resume()


def _internal(e: Exception) -> JSONResponse:
    """An unexpected failure inside a model call: a clean JSON error (never a dropped connection), key-scrubbed."""
    import logging
    logging.getLogger("fovea.ai").error("AI request failed: %s", secrets.scrub(f"{type(e).__name__}: {e}"), exc_info=True)
    return JSONResponse(status_code=500, content={"detail": secrets.scrub(f"{type(e).__name__}: {e}")[:500],
                                                  "code": "internal"})


def _book_failed(e: BaseException, ds_id: str, kind: str, provider: str, model: str, params: dict) -> None:
    """ai-cloud hook: a single request that failed after the provider billed it (a refusal, unreadable or cut-off
    output) becomes an error run carrying that usage, so Runs & cost, the usage totals and the spend caps count it."""
    spent = billed_usage(e)
    if spent is None:
        return
    try:
        rid = store.create_run(ds_id, kind, provider or "", model or "", params, status="error")
        store.add_usage(rid, spent, images=0)
        store.finish_run(rid, "error", error=secrets.scrub(str(e))[:500])
    except Exception:  # noqa: BLE001 — bookkeeping must never replace the provider's own error
        import logging
        logging.getLogger("fovea.ai").exception("recording the cost of a failed %s request failed", kind)


def _err(e: ProviderError) -> JSONResponse:
    headers = {}
    if e.code == "rate" and e.retry_after:
        headers["Retry-After"] = str(int(math.ceil(e.retry_after)))
    return JSONResponse(status_code=STATUS.get(e.code, 422), content={"detail": secrets.scrub(str(e)), "code": e.code},
                        headers=headers)


def _guard_remote(request: Request, providers, action: str = "") -> None:
    if registry.remote_ai_allowed() or _is_local(request):
        return
    # Cloud providers, and a local server that is sent the owner's API key (a gateway to a paid model).
    if any(registry.is_guarded(p) for p in providers if isinstance(p, str) and p):
        raise HTTPException(403, (f"{action}: " if action else "") + REMOTE_MESSAGE)


def _guard_keys(request: Request, what: str = "API keys") -> None:   # shell hook (r4): `what` names the refused change
    if registry.remote_ai_allowed() or _is_local(request):
        return
    raise HTTPException(403, f"{what}: " + REMOTE_MESSAGE)


def _providers_in(provider: Any, options: Optional[dict]) -> List[str]:
    """Every provider a detect configuration would call (a two-stage pipeline's detector AND classifier) — what the
    remote guard and the spend caps must look at."""
    return runner.providers_of(provider, options)


def _members_providers(provider: Any, options: Optional[dict], committee: Any) -> List[str]:
    """Providers an audit / review calls: the main configuration and every committee member, two-stage included."""
    out = _providers_in(provider, options)
    for m in committee if isinstance(committee, list) else []:
        if isinstance(m, dict):
            out += _providers_in(m.get("provider"), m.get("options") if isinstance(m.get("options"), dict) else None)
    return out


def _check_cap(providers: List[str]) -> None:
    for p in set(x for x in providers if isinstance(x, str)):
        if p and registry.is_cloud(p):
            rem = runner.cap_remaining(p)
            if rem is not None and rem <= 0:
                raise ProviderError(f"The monthly spend cap for {registry.NAMES.get(p, p)} set in Settings is reached",
                                    "quota")


def _check_priced(provider: Any, model: Any, options: Optional[dict]) -> None:
    """A cloud model with no price on file is booked at $0: refuse it while a provider spend cap must hold."""
    if any(registry.is_cloud(p) for p in _providers_in(provider, options) if p):
        runner.refuse_unpriced(runner.config_models(provider, model, options))


_INT_RE = re.compile(r"-?[0-9]{1,18}")       # ASCII digits only: str.isdigit() also accepts '²' and '٣'


def _as_int(v: Any) -> Optional[int]:
    """An integer from JSON: an int (not a bool), an integral finite float, or an ASCII-digit string; else None."""
    if isinstance(v, bool):
        return None
    if isinstance(v, int):
        return v
    if isinstance(v, float):
        return int(v) if math.isfinite(v) and v.is_integer() and abs(v) < 2 ** 62 else None
    if isinstance(v, str) and _INT_RE.fullmatch(v.strip()):
        return int(v.strip())
    return None


def _int(v: Any, name: str) -> int:
    n = _as_int(v)
    if n is None:
        raise ProviderError(f"{name} must be an integer", "invalid")
    return n


def _int_list(v: Any, name: str, limit: int = MAX_IDS) -> List[int]:
    if v is None:
        return []
    if not isinstance(v, list):
        raise ProviderError(f"{name} must be a list of integers", "invalid")
    if len(v) > limit:
        raise ProviderError(f"{name}: at most {limit:,} items", "invalid")
    out = []
    for x in v:
        n = _as_int(x)
        if n is None:
            raise ProviderError(f"{name} must be a list of integers", "invalid")
        out.append(n)
    return out


def _str(payload: dict, key: str, default: str = "") -> str:
    """A string field (provider, model, mode …): missing / null → ``default``; any other type → 422."""
    v = payload.get(key) if isinstance(payload, dict) else None
    if v is None or v == "":
        return default
    if not isinstance(v, str):
        raise ProviderError(f"{key} must be a string", "invalid")
    return v.strip()


def _dataset(payload: dict) -> dict:
    ds_id = payload.get("dataset_id")
    if not isinstance(ds_id, str) or not ds_id:
        raise ProviderError("dataset_id is required", "invalid")
    return get_dataset_or_404(ds_id)


def _image(ds_id: str, image_id: Any) -> dict:
    iid = _int(image_id, "image_id")
    img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (iid, ds_id))
    if not img:
        raise HTTPException(404, "Image not found")
    return img


def _classes(payload: dict) -> Optional[List[int]]:
    v = payload.get("classes")
    if v in (None, []):
        return None
    return _int_list(v, "classes", 1000)


def _options(payload: dict) -> dict:
    o = payload.get("options") or {}
    if not isinstance(o, dict):
        raise ProviderError("options must be an object", "invalid")
    return dict(o)


def target_ids(ds_id: str, payload: dict, required: bool = True) -> Optional[List[int]]:
    """``image_ids`` (restricted to this dataset, order kept) or ``filters`` (Explore filter object, {} = all)."""
    if payload.get("image_ids") is not None:
        ids = _int_list(payload.get("image_ids"), "image_ids")
        found = set()
        for chunk in store.chunked(ids):
            found.update(r["id"] for r in db.query(
                f"SELECT id FROM images WHERE dataset_id=? AND id IN ({','.join('?' * len(chunk))})", [ds_id, *chunk]))
        seen, out = set(), []
        for i in ids:
            if i in found and i not in seen:
                seen.add(i)
                out.append(i)
        return out
    if payload.get("filters") is not None:
        f = payload.get("filters")
        if not isinstance(f, dict):
            raise ProviderError("filters must be an object", "invalid")
        sort, order = payload.get("sort"), payload.get("order")
        if hasattr(imagesq, "list_ids") and isinstance(sort, (str, type(None))) and isinstance(order, (str, type(None))):
            return list(imagesq.list_ids(ds_id, filters_from(f), sort, order))   # the order the user sees
        where, params = imagesq.build_where(ds_id, filters_from(f))
        rows = db.query(f"SELECT i.id FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND "
                        f"r.rel_path=i.rel_path WHERE {where} ORDER BY i.id", params)
        return [r["id"] for r in rows]
    if required:
        raise ProviderError("Give image_ids or filters", "invalid")
    return None


def _usage(u: Usage) -> dict:
    return {"input_tokens": int(u.input_tokens), "output_tokens": int(u.output_tokens), "cost_usd": round(float(u.cost_usd), 6)}


def _job_of(run: dict) -> Optional[dict]:
    if not run.get("job_id"):
        return None
    j = jobs.get(run["job_id"])
    return j.to_dict() if j else None


# ================================================================== providers / keys / config

@router.get("/api/ai/providers")
def list_providers():
    return {"providers": [registry.entry(pid) for pid in registry.ids()], "prices_as_of": registry.prices_as_of(),
            "remote_ai_allowed": registry.remote_ai_allowed(), "device": local.device_info(),
            "settings": ai_config.ai_settings(), "loaded": local.CACHE.loaded()}


def _provider_or_404(pid: str):
    if pid not in registry.ids():
        raise HTTPException(404, f"Unknown provider: {pid}")
    return registry.get(pid)


@router.put("/api/ai/providers/{pid}/key")
def put_key(pid: str, request: Request, payload: dict = Body(...)):
    p = _provider_or_404(pid)
    _guard_keys(request)
    if not (getattr(p, "needs_key", False) or pid in secrets.ENV_KEYS):
        raise HTTPException(422, f"{p.name or pid} does not use an API key")
    key = payload.get("key") if isinstance(payload, dict) else None
    if not isinstance(key, str):
        raise HTTPException(422, "key must be a string")
    try:
        info = secrets.set_key(pid, key)
    except ValueError as e:
        raise HTTPException(422, str(e)) from None
    _register_secret(secrets.get_key(pid))
    registry._models_cache.pop(pid, None)            # noqa: SLF001 — a new key may see other models
    return {"key": {k: info[k] for k in ("present", "source", "hint", "env")}, "provider": registry.entry(pid)}


@router.delete("/api/ai/providers/{pid}/key")
def delete_key(pid: str, request: Request):
    _provider_or_404(pid)
    _guard_keys(request)
    info = secrets.clear_key(pid)
    registry._models_cache.pop(pid, None)            # noqa: SLF001
    return {"key": {k: info[k] for k in ("present", "source", "hint", "env")}, "provider": registry.entry(pid)}


@router.put("/api/ai/providers/{pid}/config")
def put_config(pid: str, request: Request, payload: dict = Body(...)):
    p = _provider_or_404(pid)
    # Judged by what the provider IS, not by its current url: openai_compat counts as local while it points at
    # localhost, yet a remote client re-pointing it (base_url) would ship the stored key and the owner's images to a
    # host of its choice, and enabled / spend cap / default model of a paid provider are the owner's call too.
    if pid in registry.CLOUD or getattr(p, "needs_key", False) or secrets.env_names(pid):
        _guard_keys(request, "Provider settings")
    try:
        ai_config.update_provider_config(pid, payload)
    except ai_config.ConfigError as e:
        raise HTTPException(422, str(e)) from None
    registry._models_cache.pop(pid, None)            # noqa: SLF001
    return registry.entry(pid)


@router.post("/api/ai/providers/{pid}/test")
def test_provider(pid: str, request: Request):
    p = _provider_or_404(pid)
    _guard_remote(request, [pid], "Test")
    registry.forget_models(pid)                     # a Test re-reads what the server serves: no stale marks after it
    t0 = time.time()
    try:
        res = p.test() or {}
    except ProviderError as e:
        return {"ok": False, "message": secrets.scrub(str(e)), "code": e.code, "latency_ms": int((time.time() - t0) * 1000)}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "message": secrets.scrub(f"{type(e).__name__}: {e}"), "latency_ms": int((time.time() - t0) * 1000)}
    return {"ok": bool(res.get("ok")), "message": secrets.scrub(res.get("message") or ""),
            "latency_ms": int(res.get("latency_ms") or (time.time() - t0) * 1000)}


@router.get("/api/ai/providers/{pid}/models")
def provider_models(pid: str, request: Request, refresh: int = Query(0)):
    _provider_or_404(pid)
    if refresh:
        _guard_remote(request, [pid], "Model list")
    try:
        lst = registry.models(pid, refresh=bool(refresh))
    except ProviderError as e:
        return {"models": registry.models(pid), "error": secrets.scrub(str(e)), "code": e.code, "live": False}
    d = registry.default_model(pid)
    return {"models": [{**m, "default": m.get("id") == d} for m in lst], "live": bool(refresh), "default_model": d}


@router.get("/api/ai/settings")
def get_ai_settings():
    return {"settings": ai_config.ai_settings(), "device": local.device_info(), "loaded": local.CACHE.loaded()}


@router.put("/api/ai/settings")
def put_ai_settings(payload: dict = Body(...)):
    before = ai_config.ai_settings()
    try:
        s = ai_config.update_ai_settings(payload)
    except ai_config.ConfigError as e:
        raise HTTPException(422, str(e)) from None
    if s.get("device") != before.get("device"):
        local.CACHE.unload()                       # models are bound to a device; reload on next use
    return {"settings": s, "device": local.device_info(), "loaded": local.CACHE.loaded()}


_warming: Dict[tuple, float] = {}


@router.post("/api/ai/warm")
def warm(payload: dict = Body(...)):
    """Load a local model in the background (Annotate calls this when the AI panel opens) so the first proposal does
    not pay for loading + kernel compilation. Returns at once: {state: ready | loading | cloud}."""
    try:
        provider = _str(payload or {}, "provider")
        model_in = _str(payload or {}, "model")      # read first: a malformed request is 422 even without the provider
        pipeline.provider_for(provider, "refine" if provider == "sam" else "detect")
        if registry.is_cloud(provider):
            return {"state": "cloud"}
        model = pipeline.check_model(provider, model_in)
    except ProviderError as e:
        return _err(e)
    fam = {"yolo": "yolo", "yoloe": "yoloe", "sam": "sam"}.get(provider, provider)
    if local.CACHE.is_loaded(fam, model):
        return {"state": "ready", "model": model}
    key = (provider, model)
    if time.time() - _warming.get(key, 0) < 120:
        return {"state": "loading", "model": model}
    _warming[key] = time.time()

    def _load():
        try:
            if provider == "sam":
                from ..ai.providers.sam import warm as sam_warm
                sam_warm(model)
            else:
                from ..ai.providers.yolo import load_entry
                load_entry(model, fam, "text" if provider == "yoloe" and not model.endswith("-pf") else
                           ("pf" if provider == "yoloe" else "nms"))
        except Exception:  # noqa: BLE001 — the real request will report the error
            pass
        finally:
            _warming.pop(key, None)

    import threading
    threading.Thread(target=_load, name=f"fovea-warm-{provider}", daemon=True).start()
    return {"state": "loading", "model": model}


@router.post("/api/ai/unload")
def unload_models(payload: dict = Body(default={})):
    n = local.CACHE.unload((payload or {}).get("family"), (payload or {}).get("name"))
    return {"unloaded": n, "loaded": local.CACHE.loaded()}


# ================================================================== weights

@router.get("/api/ai/weights")
def list_weights():
    return {"weights": weights.weights_list(), "model_dir": str(weights.MODEL_DIR)}


@router.post("/api/ai/weights/download")
def download_weights(payload: dict = Body(...)):
    name = (payload or {}).get("name")
    c = weights.curated(name if isinstance(name, str) else "")
    if c is None:
        raise HTTPException(422, "Only models from the curated list can be downloaded")
    for j in jobs.list(kind="ai_weights"):
        if j.status in ("queued", "running") and (j.extra or {}).get("name") == c["name"]:
            return {"job": j.to_dict()}

    def run(job):
        job.update(extra={"name": c["name"]})
        res = weights.download(c["name"], job)
        registry._models_cache.clear()             # noqa: SLF001
        return res

    job = jobs.submit("ai_weights", run, message=f"Downloading {c['name']}…")
    job.extra = {"name": c["name"]}
    return {"job": job.to_dict()}


# ================================================================== single image

@router.post("/api/ai/propose")
def propose(request: Request, payload: dict = Body(...)):
    t_start = db.now()          # the run's duration in Runs & cost is the whole request, not the bookkeeping after it
    try:
        ds = _dataset(payload)
        img = _image(ds["id"], payload.get("image_id"))
        provider = _str(payload, "provider")
        options = _options(payload)
        provs = _providers_in(provider, options)
        _guard_remote(request, provs, "Propose")
        _check_cap(provs)
        _check_priced(provider, _str(payload, "model"), options)
        out = pipeline.detect_image(ds, img, provider, _str(payload, "model"), payload.get("conf"), _classes(payload),
                                    options)
    except ProviderError as e:
        if billed_usage(e) is not None:
            _book_failed(e, ds["id"], "propose", provider if options.get("pipeline") != "two_stage" else "+".join(provs),
                         _str(payload, "model"), {"options": options, "image_id": img["id"]})
        return _err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        if billed_usage(e) is not None:
            _book_failed(e, ds["id"], "propose", provider if options.get("pipeline") != "two_stage" else "+".join(provs),
                         _str(payload, "model"), {"options": options, "image_id": img["id"]})
        return _internal(e)
    run_provider = provider if options.get("pipeline") != "two_stage" else "+".join(provs)
    run_id = store.create_run(ds["id"], "propose", run_provider, out["model"],
                              {"conf": payload.get("conf"), "classes": payload.get("classes"), "options": options,
                               "image_id": img["id"]}, status="done", created_at=t_start)
    fresh, n_rej = store.drop_rejected(img["id"], out["preds"])    # rejected here before: not raised again
    preds = store.insert_preds(run_id, ds["id"], img["id"], fresh, "pending", replace_pending=("detect",))
    store.add_usage(run_id, out["usage"], images=1, boxes=len(preds))
    warnings = list(out["warnings"])
    if n_rej:
        warnings.append(f"{n_rej} proposal{'s' if n_rej != 1 else ''} you rejected before not shown again")
    store.finish_run(run_id, "done", result={"warnings": warnings, "latency_ms": out["latency_ms"],
                                             "already_rejected": n_rej})
    store.refresh_planner_stats()
    if not out["preds"]:
        warnings.append("No objects found" + (f" at confidence ≥ {float(payload['conf']):.2f}"
                                              if payload.get("conf") not in (None, "") else ""))
    return {"run_id": run_id, "image_id": img["id"], "preds": preds, "usage": _usage(out["usage"]),
            "latency_ms": out["latency_ms"], "warnings": warnings, "served_by": out["served_by"]}


@router.post("/api/ai/preview")
def preview(request: Request, payload: dict = Body(...)):
    t_start = db.now()
    try:
        ds = _dataset(payload)
        ids = _int_list(payload.get("image_ids"), "image_ids", 4)
        if not ids:
            raise ProviderError("Give 1-4 image_ids", "invalid")
        provider = _str(payload, "provider")
        options = _options(payload)
        provs = _providers_in(provider, options)
        _guard_remote(request, provs, "Preview")
        _check_cap(provs)
        model = _str(payload, "model")
        runner.validate_detect(ds, provider, model, payload.get("conf"), _classes(payload), options)
        _check_priced(provider, model, options)
    except ProviderError as e:
        return _err(e)
    items, total, served = [], Usage(), ""
    for iid in ids:
        img = db.query_one("SELECT * FROM images WHERE id=? AND dataset_id=?", (iid, ds["id"]))
        if not img:
            items.append({"image_id": iid, "preds": [], "error": "Image not found", "warnings": []})
            continue
        try:
            out = pipeline.detect_image(ds, img, provider, _str(payload, "model"), payload.get("conf"),
                                        _classes(payload), options)
        except ProviderError as e:
            if billed_usage(e) is not None:          # ai-cloud hook: billed although it failed
                total.add(billed_usage(e))
            items.append({"image_id": iid, "rel_path": img["rel_path"], "preds": [], "error": secrets.scrub(str(e)),
                          "code": e.code, "warnings": []})
            if e.fatal:
                break
            continue
        except Exception as e:  # noqa: BLE001
            if billed_usage(e) is not None:          # ai-cloud hook
                total.add(billed_usage(e))
            items.append({"image_id": iid, "rel_path": img["rel_path"], "preds": [], "code": "internal", "warnings": [],
                          "error": secrets.scrub(f"{type(e).__name__}: {e}")[:500]})
            continue
        total.add(out["usage"])
        served, model = out["served_by"], out["model"]
        items.append({"image_id": iid, "rel_path": img["rel_path"], "preds": out["preds"], "warnings": out["warnings"],
                      "latency_ms": out["latency_ms"], "usage": _usage(out["usage"])})
    if total.cost_usd or total.input_tokens:
        rid = store.create_run(ds["id"], "preview", "+".join(p for p in provs if p), model,
                               {"options": options, "image_ids": ids}, status="done", created_at=t_start)
        store.add_usage(rid, total, images=len(items))
        store.finish_run(rid, "done")
    return {"items": items, "usage": _usage(total), "served_by": served}


def _target_summary(ids: List[int]) -> dict:
    """What a write target would touch — for the confirmation before Replace / Append on many images."""
    out = {"images": len(ids), "labeled": 0, "boxes": 0, "no_label_file": 0, "empty_label_file": 0}
    for chunk in store.chunked(ids):
        r = db.query_one(
            "SELECT SUM(CASE WHEN has_label=1 AND n_boxes>0 THEN 1 ELSE 0 END) lab, SUM(n_boxes) b, "
            "SUM(CASE WHEN has_label=0 THEN 1 ELSE 0 END) nofile, SUM(CASE WHEN has_label=1 AND n_boxes=0 THEN 1 ELSE 0 END) "
            f"emp FROM images WHERE id IN ({','.join('?' * len(chunk))})", chunk) or {}
        out["labeled"] += int(r.get("lab") or 0)
        out["boxes"] += int(r.get("b") or 0)
        out["no_label_file"] += int(r.get("nofile") or 0)
        out["empty_label_file"] += int(r.get("emp") or 0)
    return out


@router.post("/api/ai/estimate")
def estimate(payload: dict = Body(...)):
    try:
        ds = _dataset(payload)
        ids = target_ids(ds["id"], payload, required=False)
        count = payload.get("count")
        if ids is None and count is None:
            raise ProviderError("Give image_ids, filters or count", "invalid")
        if count is not None:
            count = _int(count, "count")
        conc = payload.get("concurrency")
        if conc is not None:
            conc = _int(conc, "concurrency")
            if not 1 <= conc <= 16:
                raise ProviderError("concurrency must be between 1 and 16", "invalid")
        out = ai_estimate.estimate(ds, ids, count, _str(payload, "provider"), _str(payload, "model"),
                                   _options(payload), concurrency=conc)
        if ids is not None:
            out["targets"] = _target_summary(ids)
        o = _options(payload)
        if (o.get("pipeline") or "") == "two_stage":
            out["economy"] = {"available": False, "reason": "Two-stage runs box by box, so it has no batch mode"}
        else:
            ok, why = economy.available(_str(payload, "provider"))
            out["economy"] = {"available": ok and bool(out.get("usd_batch")), "reason": why}
        return out
    except ProviderError as e:
        return _err(e)


@router.post("/api/ai/classify")
def classify(request: Request, payload: dict = Body(...)):
    t_start = db.now()
    try:
        ds = _dataset(payload)
        img = _image(ds["id"], payload.get("image_id"))
        provider = _str(payload, "provider")
        _guard_remote(request, [provider], "Classify")
        _check_cap([provider])
        _check_priced(provider, _str(payload, "model"), None)
        out = pipeline.classify_boxes(ds, img, payload.get("boxes"), provider, _str(payload, "model"),
                                      _classes(payload), _options(payload))
    except ProviderError as e:
        if billed_usage(e) is not None:
            _book_failed(e, ds["id"], "classify", provider, _str(payload, "model"), {"image_id": img["id"]})
        return _err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        if billed_usage(e) is not None:
            _book_failed(e, ds["id"], "classify", provider, _str(payload, "model"), {"image_id": img["id"]})
        return _internal(e)
    if out["usage"].cost_usd or out["usage"].input_tokens:
        rid = store.create_run(ds["id"], "classify", provider, out["model"], {"image_id": img["id"],
                                                                             "boxes": len(payload.get("boxes") or [])},
                               status="done", created_at=t_start)
        store.add_usage(rid, out["usage"], images=1)
        store.finish_run(rid, "done")
    return {"items": out["items"], "usage": _usage(out["usage"]), "latency_ms": out["latency_ms"],
            "warnings": out["warnings"]}


@router.post("/api/ai/refine")
def refine(payload: dict = Body(...)):
    try:
        ds = _dataset(payload)
        img = _image(ds["id"], payload.get("image_id"))
        return pipeline.refine(img, payload.get("box"), payload.get("points"), _str(payload, "model"))
    except ProviderError as e:
        return _err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return _internal(e)


@router.post("/api/ai/review")
def review(request: Request, payload: dict = Body(...)):
    """"Ask AI" about one labeled image now: the label audit for a single image (missing / loose / swapped /
    unsupported boxes), stored as pending audit suggestions exactly like an audit job's."""
    try:
        ds = _dataset(payload)
        img = _image(ds["id"], payload.get("image_id"))
        provider = _str(payload, "provider")
        committee = payload.get("committee") or []
        if not isinstance(committee, list):
            raise ProviderError("committee must be a list", "invalid")
        provs = _members_providers(provider, _options(payload), committee)    # two-stage classifiers included
        _guard_remote(request, provs, "Review")
        _check_cap(provs)
        if not img["label_path"] or not Path(img["label_path"]).is_file():
            raise ProviderError("This image has no label file to review", "invalid")
        cfg = runner.prepare_audit(ds, provider, _str(payload, "model"), payload.get("conf_floor"),
                                   payload.get("class_family"), _str(payload, "mode", "full"), committee, _options(payload))
        run_id = store.create_run(ds["id"], "review", "+".join(cfg["providers"] or sorted({m["provider"] for m in
                                                                                          cfg["members"]})),
                                  "+".join(m["model"] for m in cfg["members"]),
                                  {**{k: v for k, v in cfg.items() if k != "cloud"}, "image_id": img["id"]})
        try:
            res = runner.audit_image(ds, img, cfg, run_id, store_raw=False)
        except Exception as e:
            if billed_usage(e) is not None:          # ai-cloud hook: billed although it failed
                store.add_usage(run_id, billed_usage(e), images=0)
            store.finish_run(run_id, "error", error=str(e))
            raise
        store.finish_run(run_id, "done", result={"score": res["score"], "counts": res["counts"]})
    except ProviderError as e:
        return _err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return _internal(e)
    return {"run_id": run_id, "image_id": img["id"], "preds": res["preds"], "score": res["score"], "counts": res["counts"],
            "usage": _usage(res["usage"])}


@router.post("/api/ai/mapping")
def mapping(payload: dict = Body(...)):
    """How a closed-vocabulary model's classes map onto this dataset (for the class-map editor): every model class
    with its automatic or explicit target, or null when it would be skipped."""
    try:
        ds = _dataset(payload)
        provider = _str(payload, "provider", "yolo")
        if provider not in ("yolo", "yoloe"):
            raise ProviderError("Only closed-vocabulary local models have a class map", "invalid")
        model = pipeline.check_model(provider, _str(payload, "model"))
        from ..ai.classes import map_vocab
        from ..ai.providers.yolo import model_names
        if provider == "yoloe" and not model.endswith("-pf"):
            raise ProviderError("YOLOE text/exemplar prompts use the dataset classes directly — no map needed", "invalid")
        names = model_names(model, provider)
        specs = pipeline.specs_for(ds, _classes(payload))
        cmap = payload.get("class_map") or None
        if cmap is not None and not isinstance(cmap, dict):
            raise ProviderError("class_map must be an object", "invalid")
        try:
            m, how = map_vocab(names, specs, cmap)
        except ValueError as e:
            raise ProviderError(str(e), "invalid") from None
    except ProviderError as e:
        return _err(e)
    return {"model": model, "dataset_classes": [{"id": s.id, "name": s.name} for s in specs],
            "items": [{"model_id": k, "model_name": v, "cls": m.get(k), "how": how.get(v) if k in m else None}
                      for k, v in sorted(names.items())],
            "mapped": len(m)}


# ================================================================== jobs

@router.post("/api/ai/batch")
def batch(request: Request, payload: dict = Body(...)):
    try:
        ds = _dataset(payload)
        provider = _str(payload, "provider")
        options = _options(payload)
        provs = _providers_in(provider, options)
        _guard_remote(request, provs, "Batch")
        ids = target_ids(ds["id"], payload)
        speed = payload.get("speed")
        if speed is None and (payload.get("economy") is True or options.pop("batch_api", None) is True):
            speed = "economy"
        job, run_id = runner.start_batch(ds, ids, provider, _str(payload, "model"), payload.get("conf"),
                                         _classes(payload), options, payload.get("target") or "proposals",
                                         payload.get("budget_usd"), payload.get("concurrency"), speed=speed)
    except ProviderError as e:
        return _err(e)
    return {"job": job.to_dict(), "run_id": run_id, "images": len(ids), "speed": runner.parse_speed(speed)}


@router.post("/api/ai/audit")
def audit(request: Request, payload: dict = Body(...)):
    try:
        ds = _dataset(payload)
        provider = _str(payload, "provider")
        committee = payload.get("committee") or []
        if not isinstance(committee, list):
            raise ProviderError("committee must be a list", "invalid")
        provs = _members_providers(provider, _options(payload), committee)    # two-stage classifiers included
        _guard_remote(request, provs, "Audit")
        _check_cap(provs)
        ids = target_ids(ds["id"], payload)
        job, run_id = runner.start_audit(ds, ids, provider, _str(payload, "model"), payload.get("conf_floor"),
                                         payload.get("class_family"), _str(payload, "mode", "full"), committee,
                                         _options(payload))
    except ProviderError as e:
        return _err(e)
    return {"job": job.to_dict(), "run_id": run_id}


@router.post("/api/ai/benchmark")
def benchmark(request: Request, payload: dict = Body(...)):
    try:
        ds = _dataset(payload)
        configs = payload.get("configs")
        if not isinstance(configs, list):
            raise ProviderError("configs must be a list", "invalid")
        provs = []
        for c in configs:
            if isinstance(c, dict):
                provs += _providers_in(c.get("provider"), c.get("options") if isinstance(c.get("options"), dict) else None)
        _guard_remote(request, provs, "Benchmark")
        _check_cap(provs)
        within = target_ids(ds["id"], payload, required=False)
        n = payload.get("sample")
        if n is not None or within is None:
            n = _int(n if n is not None else 30, "sample")
            if not 1 <= n <= 500:
                raise ProviderError("sample must be between 1 and 500", "invalid")
            ids = runner.sample_labeled(ds["id"], n, seed=_int(payload.get("seed") or 0, "seed"), within=within)
        else:
            ids = runner.sample_labeled(ds["id"], len(within), within=within)
            if len(ids) > 500:
                raise ProviderError("Benchmark at most 500 images (use sample)", "invalid")
        job, run_ids = runner.start_benchmark(ds, ids, configs, payload.get("iou", 0.5))
    except ProviderError as e:
        return _err(e)
    return {"job": job.to_dict(), "run_ids": run_ids, "images": len(ids)}


# ================================================================== runs / usage

@router.get("/api/ai/runs")
def list_runs(dataset_id: Optional[str] = Query(None), kind: Optional[str] = Query(None),
              limit: int = Query(50, ge=1, le=500), offset: int = Query(0, ge=0),
              counts: int = Query(1, description="0 = skip the per-state prediction counts (callers that only need "
                                                 "the runs and their jobs)")):
    where, params = ["status != 'deleted'"], []
    if dataset_id:
        where.append("dataset_id=?")
        params.append(dataset_id)
    if kind:
        kinds = [k for k in kind.split(",") if re.fullmatch(r"[a-z_]{1,20}", k)]
        if kinds:
            where.append(f"kind IN ({','.join('?' * len(kinds))})")
            params += kinds
    w = " AND ".join(where)
    total = db.query_one(f"SELECT COUNT(*) n FROM ai_runs WHERE {w}", params)["n"]
    rows = db.query(f"SELECT * FROM ai_runs WHERE {w} ORDER BY id DESC LIMIT ? OFFSET ?", params + [limit, offset])
    out = []
    for r in rows:
        d = _run_view(r)
        out.append(d)
    if rows and counts:
        store.ensure_schema()                       # idx_ai_preds_run_state: the count reads the index only
        ids = [r["id"] for r in rows]
        per_run = defaultdict(dict)
        for c in db.query(f"SELECT run_id, state, COUNT(*) n FROM ai_preds WHERE run_id IN ({','.join('?' * len(ids))}) "
                          "GROUP BY run_id, state", ids):
            per_run[c["run_id"]][c["state"]] = c["n"]
        for d in out:
            d["preds"] = per_run.get(d["id"], {})
    return {"runs": out, "total": total}


def _run_view(r: dict) -> dict:
    d = store.run_public(r)
    d["job"] = _job_of(r)
    j = d["job"]
    if d["status"] == "running" and j and j["status"] == "cancelled" and not j.get("started_at"):
        runner.finish_unstarted([r["id"]])          # cancelled while queued (normally settled by the job's hook)
        r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (r["id"],)) or r
        d = {**store.run_public(r), "job": j}
    if d["status"] == "running" and not (d["job"] and d["job"]["status"] in ("queued", "running")):
        params = d.get("params") if isinstance(d.get("params"), dict) else {}
        if params.get("speed") == "economy" and params.get("provider_batches"):
            d["status"] = "waiting"              # still at the provider; picked up when its provider is available
        else:
            d["status"] = "interrupted"          # the server restarted while it ran
    return d


@router.get("/api/ai/runs/{run_id}")
def get_run(run_id: int):
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    if not r or r["status"] == "deleted":
        raise HTTPException(404, "Run not found")
    return {"run": _run_view(r)}


def _stop_run(r: dict, wait: float = 20.0) -> None:
    """Cancel a run's live job (an Economy job also cancels its provider batches) and wait for it to stop, so nothing
    lands after the caller acts; an Economy run without a live job is cancelled at the provider directly."""
    j = jobs.get(r["job_id"]) if r.get("job_id") else None
    if j and j.status in ("queued", "running"):
        j.cancel()
        t0 = time.time()
        while j.status in ("queued", "running") and time.time() - t0 < wait:
            time.sleep(0.05)
        return
    params = db.loads(r.get("params"), {}) or {}
    if r["status"] == "running" and params.get("speed") == "economy":
        economy.cancel_detached(r["id"])


@router.post("/api/ai/runs/{run_id}/cancel")
def cancel_run(run_id: int, request: Request):
    """Stop a running job or an Economy run (its provider batches are cancelled too; results that were already back
    are still filed). Finished runs are returned unchanged."""
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    if not r or r["status"] == "deleted":
        raise HTTPException(404, "Run not found")
    params = db.loads(r.get("params"), {}) or {}
    _guard_remote(request, [r["provider"]] if params.get("speed") == "economy" else [], "Cancel")
    if r["status"] == "running":
        _stop_run(r)
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    return {"run": _run_view(r)}


@router.delete("/api/ai/runs/{run_id}")
def delete_run(run_id: int, request: Request):
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    if not r or r["status"] == "deleted":
        raise HTTPException(404, "Run not found")
    params = db.loads(r.get("params"), {}) or {}
    # Deleting a live Economy run cancels its provider batches with the owner's key: same rule as Cancel.
    _guard_remote(request, [r["provider"]] if params.get("speed") == "economy" and r["status"] == "running" else [],
                  "Delete")
    _stop_run(r, wait=15)                      # let the in-flight image finish so nothing lands after the delete
    db.execute("DELETE FROM kv WHERE key=?", (economy.kv_key(run_id),))
    return store.delete_run(run_id)


@router.get("/api/ai/usage")
def usage(days: int = Query(30, ge=1, le=3650), dataset_id: Optional[str] = Query(None)):
    since = time.time() - days * 86400
    # Interpolation ("Fill from neighbours", kind 'interp', provider 'fovea') is Fovea's own geometry, not a model.
    where, params = "created_at >= ? AND kind != 'interp'", [since]
    if dataset_id:
        where += " AND dataset_id=?"
        params.append(dataset_id)
    rows = db.query(f"SELECT id, provider, model, kind, images, input_tokens, output_tokens, cost_usd, created_at FROM "
                    f"ai_runs WHERE {where}", params)
    # by_provider: what each provider billed (a two-stage run's classifier, every committee member), from the
    # per-provider ledger; by_model keeps the run's own provider/model label.
    parts = store.usage_parts([r["id"] for r in rows])
    by_p: Dict[str, dict] = defaultdict(lambda: {"usd": 0.0, "runs": 0, "images": 0, "input_tokens": 0, "output_tokens": 0})
    by_m: Dict[str, dict] = defaultdict(lambda: {"usd": 0.0, "runs": 0, "images": 0, "input_tokens": 0, "output_tokens": 0,
                                                 "provider": "", "model": ""})
    daily: Dict[str, dict] = defaultdict(lambda: {"usd": 0.0, "runs": 0, "images": 0})
    total = 0.0
    for r in rows:
        c = float(r["cost_usd"] or 0)
        total += c
        agg = by_m[f"{r['provider']}/{r['model']}"]
        agg["usd"] += c
        agg["runs"] += 1
        agg["images"] += int(r["images"] or 0)
        agg["input_tokens"] += int(r["input_tokens"] or 0)
        agg["output_tokens"] += int(r["output_tokens"] or 0)
        agg.update(provider=r["provider"], model=r["model"])
        # No ledger rows (a free run, or one stopped before any request): book it under the provider that would have
        # billed it — a two-stage 'yolo+anthropic' run is Anthropic's, like store.add_usage does — never the raw string.
        shares = parts.get(r["id"]) or [{"provider": store._fallback_provider(r["provider"]) or "?", "cost_usd": c,  # noqa: SLF001
                                         "input_tokens": r["input_tokens"], "output_tokens": r["output_tokens"]}]
        for sh in shares:
            pa = by_p[sh["provider"] or "?"]
            pa["usd"] += float(sh["cost_usd"] or 0)
            pa["runs"] += 1
            pa["images"] += int(r["images"] or 0)
            pa["input_tokens"] += int(sh["input_tokens"] or 0)
            pa["output_tokens"] += int(sh["output_tokens"] or 0)
        day = datetime.fromtimestamp(r["created_at"] or 0).strftime("%Y-%m-%d")
        daily[day]["usd"] += c
        daily[day]["runs"] += 1
        daily[day]["images"] += int(r["images"] or 0)
    rnd = lambda d: {k: (round(v, 6) if isinstance(v, float) else v) for k, v in d.items()}  # noqa: E731
    caps = {}
    for pid in registry.ids():
        cap = ai_config.spend_cap(pid)
        if cap is not None:
            caps[pid] = {"cap_usd": cap, "spent_30d": round(runner.spent_30d(pid), 6),
                         # committed but not billed yet: Economy runs waiting at the provider, Now runs still going
                         "reserved_usd": round(runner.reserved_usd(pid), 6)}
    return {"days": days, "total_usd": round(total, 6), "by_provider": {k: rnd(v) for k, v in by_p.items()},
            "by_model": {k: rnd(v) for k, v in by_m.items()},
            "daily": [{"date": d, **rnd(v)} for d, v in sorted(daily.items())], "caps": caps,
            "unpriced": _unpriced_usage(rows, parts)}    # shell hook (r3): runs booked at $0 for want of a price


# --- shell hook (r3): Settings → Usage shows runs whose cloud model has no price on file apart from the dollar
# totals ("cost unknown"), instead of counting them as $0.
def _unpriced_usage(rows: List[dict], parts: Dict[int, List[dict]]) -> dict:
    """{runs, images, by_provider: {pid: {runs, images}}, models: ["provider/model"]} over the runs a cloud provider
    billed without a price: a ledger share with tokens but no cost, or a cloud model the price table does not know."""
    known: Dict[tuple, bool] = {}

    def priced(pid: str, model: str) -> bool:
        if (pid, model) not in known:
            known[(pid, model)] = runner.model_priced(pid, model)
        return known[(pid, model)]

    out: Dict[str, Any] = {"runs": 0, "images": 0, "by_provider": {}, "models": []}
    for r in rows:
        bad = {sh["provider"] for sh in parts.get(r["id"]) or [] if sh.get("provider") and registry.is_cloud(sh["provider"])
               and not float(sh.get("cost_usd") or 0) and (int(sh.get("input_tokens") or 0) or int(sh.get("output_tokens") or 0))}
        pids, models = str(r["provider"] or "").split("+"), str(r["model"] or "").split("+")
        pairs = list(zip(pids, models)) if len(pids) == len(models) else []
        if not float(r["cost_usd"] or 0):
            bad |= {pid for pid, m in pairs if pid and registry.is_cloud(pid) and not priced(pid, m)}
        if not bad:
            continue
        out["runs"] += 1
        out["images"] += int(r["images"] or 0)
        for pid in sorted(bad):
            pa = out["by_provider"].setdefault(pid, {"runs": 0, "images": 0})
            pa["runs"] += 1
            pa["images"] += int(r["images"] or 0)
        key = f"{r['provider']}/{r['model']}"
        if key not in out["models"]:
            out["models"].append(key)
    return out
# --- end shell hook


# ================================================================== text helpers (generate_json)

def _assist_err(e: ProviderError) -> JSONResponse:
    """'unavailable' (no provider here can write text) is a 400 for these endpoints: the request cannot work in
    this setup, it is not a temporary outage."""
    if e.code == "unavailable":
        return JSONResponse(status_code=400, content={"detail": secrets.scrub(str(e)), "code": e.code})
    return _err(e)


@router.post("/api/ai/name-classes")
def name_classes(request: Request, payload: dict = Body(...)):
    """Suggest a name, a detector prompt and a definition per class from a few typical boxes (first run on unnamed
    datasets). ``dry_run: true`` returns the cost estimate and the boxes that would be sent, without sending."""
    try:
        ds = _dataset(payload)
        provider = payload.get("provider") or None
        if provider is not None and not isinstance(provider, str):
            raise ProviderError("provider must be a string", "invalid")
        _guard_remote(request, [provider] if provider else [p for p in assist.generators()[:1]], "Name classes")
        dry = bool(payload.get("dry_run") or payload.get("estimate_only"))
        if not dry:
            _check_cap([provider] if provider else assist.generators()[:1])
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ProviderError("model must be a string", "invalid")
        token = _naming_token(payload.get("cancel_token")) if not dry else None      # ai-ui hook (r2): Stop
        try:
            return assist.name_classes(ds, provider, model, _classes(payload), payload.get("per_class", 6), dry_run=dry,
                                       cancel=_naming_cancel(token, create=True) if token else None)
        finally:
            if token:
                _NAMING_CANCEL.pop(token, None)
    except ProviderError as e:
        return _assist_err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return _internal(e)


# --- ai-ui hook (r2): Stop in the naming dialog and provider waits for a synchronous Propose -------------------------
_NAMING_CANCEL: Dict[str, Any] = {}          # cancel token -> (threading.Event, created_at)


def _naming_token(v: Any) -> Optional[str]:
    return v if isinstance(v, str) and re.fullmatch(r"[A-Za-z0-9_.-]{8,80}", v) else None


def _naming_cancel(token: str, create: bool = False):
    """The Event behind a cancel token. A Stop that arrives before its request registered still counts: the token is
    kept (10 min at most) and the request finds it already set."""
    import threading
    now = time.time()
    for k, (_ev, at) in list(_NAMING_CANCEL.items()):
        if now - at > 600:
            _NAMING_CANCEL.pop(k, None)
    if token not in _NAMING_CANCEL:
        if not create and len(_NAMING_CANCEL) > 1000:
            return None
        _NAMING_CANCEL[token] = (threading.Event(), now)
    return _NAMING_CANCEL[token][0]


@router.post("/api/ai/name-classes/cancel")
def name_classes_cancel(payload: dict = Body(...)):
    """Stop a running "Suggest names": classes whose request was not sent yet are skipped (not billed); requests
    already sent finish and are billed, and the running call returns what they found."""
    token = _naming_token(payload.get("token"))
    if not token:
        return JSONResponse(status_code=422, content={"detail": "token must be the cancel_token of a request", "code": "invalid"})
    ev = _naming_cancel(token)
    if ev is not None:
        ev.set()
    return {"cancelled": ev is not None}


@router.get("/api/ai/waits")
def provider_waits():
    """Provider retry waits in progress on this server, longest first: ``{waits: [{reason, seconds_left}]}`` (e.g.
    "Anthropic Claude rate limited", 32). Lets a synchronous Propose say why it takes long. No secrets, no network."""
    try:
        from ..ai import vlm
        return {"waits": vlm.current_waits()}
    except Exception:  # noqa: BLE001 — cloud dependencies missing: nothing can wait
        return {"waits": []}
# --- end ai-ui hook


@router.post("/api/ai/nl-filter")
def nl_filter(request: Request, payload: dict = Body(...)):
    """Natural-language search (Korean or English) → validated Explore filters + optional sort + one-line
    explanation. Uses the provider's cheapest model unless ``model`` is given."""
    try:
        ds = _dataset(payload)
        provider = payload.get("provider") or None
        if provider is not None and not isinstance(provider, str):
            raise ProviderError("provider must be a string", "invalid")
        model = payload.get("model")
        if model is not None and not isinstance(model, str):
            raise ProviderError("model must be a string", "invalid")
        _guard_remote(request, [provider] if provider else assist.generators()[:1], "Ask AI")
        _check_cap([provider] if provider else assist.generators()[:1])
        return assist.nl_filter(ds, payload.get("text"), provider, model)
    except ProviderError as e:
        return _assist_err(e)
    except HTTPException:
        raise
    except Exception as e:  # noqa: BLE001
        return _internal(e)


# ================================================================== proposals (per dataset)

@router.get("/api/datasets/{ds_id}/images/{image_id}/proposals")
def image_proposals(ds_id: str, image_id: int):
    get_dataset_or_404(ds_id)
    if not db.query_one("SELECT id FROM images WHERE id=? AND dataset_id=?", (image_id, ds_id)):
        raise HTTPException(404, "Image not found")
    return store.image_proposals(ds_id, image_id)


@router.post("/api/datasets/{ds_id}/proposals/decide")
def decide(ds_id: str, payload: dict = Body(...)):
    get_dataset_or_404(ds_id)
    try:
        acc = _int_list(payload.get("accept") or [], "accept", 20000)
        rej = _int_list(payload.get("reject") or [], "reject", 20000)
        pend = _int_list(payload.get("pending") or [], "pending", 20000)
    except ProviderError as e:
        return _err(e)
    if set(acc) & set(rej) or (set(acc) | set(rej)) & set(pend):
        raise HTTPException(422, "A proposal cannot have two decisions")
    # `pending` undoes an earlier decision (Annotate's undo after leaving and returning to an image).
    return {"updated": store.decide(ds_id, acc, rej) + store.undecide(ds_id, pend)}


def _kinds(payload: dict, default) -> List[str]:
    k = payload.get("kinds")
    if k in (None, []):
        return list(default)
    if not isinstance(k, list) or any(x not in store.ALL_KINDS for x in k):
        raise ProviderError(f"kinds must be a list of {', '.join(store.ALL_KINDS)}", "invalid")
    return list(k)


@router.post("/api/datasets/{ds_id}/proposals/accept")
def accept(ds_id: str, payload: dict = Body(...)):
    ds = get_dataset_or_404(ds_id)
    try:
        ids = target_ids(ds_id, payload)
        kinds = _kinds(payload, store.DETECT_KINDS)
        # [explore hook] skip_conflicts: proposals over a box of another class stay pending (Annotate's ⇧Enter rule).
        # ai-core r4: on for every caller; only an explicit false writes such a box anyway.
        job = runner.start_accept(ds, ids, payload.get("min_conf"), _classes(payload), kinds,
                                  skip_conflicts=payload.get("skip_conflicts") is not False)
    except ProviderError as e:
        return _err(e)
    return {"job": job.to_dict()}


@router.post("/api/datasets/{ds_id}/proposals/reject")
def reject(ds_id: str, payload: dict = Body(...)):
    get_dataset_or_404(ds_id)
    try:
        ids = target_ids(ds_id, payload)
        kinds = _kinds(payload, store.ALL_KINDS)
        mc = payload.get("min_conf")
        mc = None if mc in (None, "") else pipeline.as_conf(mc, 0.0, "min_conf")
        below = bool(payload.get("below"))
        if below and mc is None:
            raise ProviderError("below needs min_conf", "invalid")
        preds = store.pending_for_images(ds_id, ids, kinds, mc, _classes(payload), below=below)
    except ProviderError as e:
        return _err(e)
    n = store.set_state([p["id"] for p in preds], "rejected")
    return {"rejected": n, "images": len({p["image_id"] for p in preds})}


@router.get("/api/datasets/{ds_id}/audit/summary")
def audit_summary(ds_id: str, top: int = Query(20, ge=1, le=200)):
    get_dataset_or_404(ds_id)
    return runner.audit_summary(ds_id, top)


# ================================================================== legacy (pre-registry) endpoints

@router.get("/api/ai/models")
def legacy_models():
    """Closed-vocabulary detector weights (kept for the compare plugin and older pages)."""
    from ..core import yolo
    return {"available": yolo.available(), "models": yolo.list_models()}


@router.get("/api/ai/models/{name}/classes")
def legacy_model_classes(name: str):
    from ..ai.providers.yolo import model_names
    try:
        fam = "yoloe" if name.startswith("yoloe") else "yolo"
        return {"classes": model_names(name, fam)}
    except ProviderError as e:
        return _err(e)


@router.post("/api/ai/detect")
def legacy_detect(payload: dict = Body(...)):
    """Old Annotate call: detections for one image mapped to dataset classes, nothing stored. Model classes map by
    name unless ``class_map`` says otherwise; unmapped detections are dropped (never written as raw ids)."""
    try:
        iid = _int(payload.get("image_id"), "image_id")
        img = db.query_one("SELECT * FROM images WHERE id=?", (iid,))
        if not img:
            raise HTTPException(404, "Image not found")
        ds = get_dataset_or_404(img["dataset_id"])
        cmap = payload.get("class_map") or None
        if cmap is not None and not isinstance(cmap, dict):
            raise ProviderError("class_map must be an object", "invalid")
        opts = {"class_map": cmap, "imgsz": payload.get("imgsz")}
        out = pipeline.detect_image(ds, img, "yolo", _str(payload, "model"), payload.get("conf"), None, opts)
    except ProviderError as e:
        return _err(e)
    dets = [[p["cls"], p["xc"], p["yc"], p["w"], p["h"], p["conf"]] for p in out["preds"]]
    return {"detections": dets, "count": len(dets), "warnings": out["warnings"]}


@router.post("/api/ai/autolabel")
def legacy_autolabel(request: Request, payload: dict = Body(...)):
    mode = payload.get("mode", "fill")
    if mode not in ("fill", "append", "replace"):
        raise HTTPException(422, "mode must be fill, append or replace")
    body = {**payload, "provider": "yolo", "target": mode,
            "options": {"class_map": payload.get("class_map") or None, "imgsz": payload.get("imgsz")},
            "classes": None}
    if payload.get("image_ids") is None and payload.get("filters") is None:
        raise HTTPException(422, "Give image_ids or filters")
    return batch(request, body)
