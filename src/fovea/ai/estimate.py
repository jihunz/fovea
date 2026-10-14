"""Cost / time estimate for running a provider (or a two-stage pipeline) over N images — shown before every run."""
from __future__ import annotations

import re
import statistics
from typing import List, Optional, Sequence

from .. import db
from . import local, pipeline, registry, store
from .base import ProviderError

# Rough per-image seconds when nothing has been measured yet on this machine: (gpu/mps, cpu) for a medium-size model.
# One image per call (tiles, two-stage) …
LOCAL_GUESS = {"yolo": (0.12, 0.8), "yoloe": (0.15, 1.0), "sam": (0.5, 2.0)}
# … and jobs with chunked inference (8 images per call, the next chunk decoding meanwhile): measured 24.6 img/s for
# yolo12m on Apple MPS over 1,000 640-px frames — the guess stays a little slower than that.
JOB_GUESS = {"yolo": (0.05, 0.4), "yoloe": (0.07, 0.5), "sam": (0.5, 2.0)}
SIZE_FACTOR = {"n": 0.35, "s": 0.6, "m": 1.0, "l": 1.4, "x": 2.2}


def _size_factor(model: str) -> float:
    """yolo26n … yolo12x / yoloe-26s-seg: the size letter right after the version digits scales the guess."""
    m = re.search(r"\d([nsmlx])(?=$|[-_.])", (model or "").lower())
    return SIZE_FACTOR.get(m.group(1), 1.0) if m else 1.0


def _local_seconds(pid: str, model: str, options: dict) -> tuple:
    """(seconds per image in a job, 'measured' | 'guess') for a local model: measured job throughput > measured
    per-call latency > a per-device guess."""
    fam = pid if pid in LOCAL_GUESS else "yolo"
    mode = "batch" if pipeline.batch_capable(pid, options) else "single"
    s = local.job_seconds(pid, model, mode)
    if s:
        return round(s, 4), "measured"
    ms = local.speed_ms(fam, model)
    if ms:
        return round(ms / 1000, 3), "measured"
    gpu = local.is_imported() and local.resolve_device() != "cpu"
    table = JOB_GUESS if mode == "batch" else LOCAL_GUESS
    return round(table[fam][0 if gpu or not local.is_imported() else 1] * _size_factor(model), 3), "guess"


def _sizes(ds_id: str, image_ids: Optional[Sequence[int]]) -> tuple:
    if image_ids:
        sample = list(image_ids)[:400]
        rows = db.query(f"SELECT width, height FROM images WHERE id IN ({','.join('?' * len(sample))})", sample)
    else:
        rows = db.query("SELECT width, height FROM images WHERE dataset_id=? AND width IS NOT NULL "
                        "ORDER BY (id * 2654435761) % 1000003 LIMIT 400", (ds_id,))
    ws = [r["width"] for r in rows if r["width"]]
    hs = [r["height"] for r in rows if r["height"]]
    if not ws:
        return 1280, 720
    return int(statistics.median(ws)), int(statistics.median(hs))


def parallel_requests(pids: Sequence[str], requested: Optional[int] = None) -> int:
    """Requests a Now run sends at once to these cloud providers: Settings › Parallel requests (or the run's own
    ``concurrency``), the smallest over the providers involved, held down by a provider's limiter while it backs off
    after rate limits (runner.start_batch / vlm.AdaptiveLimiter)."""
    from .config import provider_config
    out: List[int] = []
    for pid in pids:
        try:
            n = int(provider_config(pid).get("concurrency") or 4)
        except (TypeError, ValueError):
            n = 4
        lim = getattr(registry.get(pid), "limiter", None)
        if callable(lim):
            try:
                n = min(n, int(getattr(lim(), "limit", n) or n))
            except Exception:  # noqa: BLE001 — the setting alone is a fine estimate
                pass
        out.append(max(1, n))
    n = min(out) if out else 1
    if requested:
        n = int(requested)
    return max(1, min(16, n))


def _boxes_per_image(ds_id: str) -> float:
    r = db.query_one("SELECT AVG(n_boxes) a FROM images WHERE dataset_id=? AND has_label=1 AND n_boxes > 0", (ds_id,))
    return float(r["a"]) if r and r["a"] else 2.0


def _one(pid: str, model: str, W: int, H: int, n_classes: int, options: dict) -> dict:
    p = registry.get(pid)
    try:
        e = p.estimate(W, H, model, n_classes, options) or {}
    except ProviderError:
        raise
    except Exception:  # noqa: BLE001 — an estimate must never block a run
        e = {}
    out = {"usd": list(e.get("usd") or [0.0, 0.0]), "tokens_in": list(e.get("tokens_in") or [0, 0]),
           "tokens_out": list(e.get("tokens_out") or [0, 0]), "seconds": e.get("seconds"),
           "usd_batch": e.get("usd_batch"), "priced": True}
    if e.get("tier"):                  # ai-cloud hook: "flex" = synchronous requests billed at batch prices (OpenAI)
        out["tier"] = e["tier"]
    if registry.is_cloud(pid):
        # A model missing from the price table (added by a live "Refresh list") is booked at $0: say so.
        pr = e.get("priced")
        out["priced"] = bool(pr) if pr is not None else float(out["usd"][1] or 0) > 0
    else:
        out["seconds"], how = _local_seconds(pid, model, options or {})
        out[how] = True
    if out["usd_batch"] is None and (getattr(p, "capabilities", {}) or {}).get("batch_api"):
        out["usd_batch"] = [round(v * 0.5, 6) for v in out["usd"]]
    return out


def estimate(ds_row: dict, image_ids: Optional[Sequence[int]], count: Optional[int], provider: str, model: str,
             options: Optional[dict], concurrency: Optional[int] = None) -> dict:
    """Cost and time of a run over ``image_ids`` (or ``count`` images). ``seconds_per_image`` is what one image adds
    to a batch (Now) run: per-request latency ÷ the parallel requests it sends to a cloud provider, a local model's
    measured job throughput; ``seconds_per_request`` is one request on its own (an audit or a benchmark asks one image
    at a time)."""
    options = dict(options or {})
    n = len(image_ids) if image_ids is not None else int(count or 0)
    if n < 0:
        raise ProviderError("count must be zero or more", "invalid")
    W, H = _sizes(ds_row["id"], image_ids)
    n_classes = max(1, len(db.loads(ds_row["classes"])))
    assumptions: List[str] = [f"median image size {W}×{H}"]
    if (options.get("pipeline") or "") == "two_stage":
        det, clf = options.get("detector") or {}, options.get("classifier") or {}
        dpid, cpid = det.get("provider") or "", clf.get("provider") or ""
        for pid in (dpid, cpid):
            try:
                registry.get(pid)
            except KeyError:
                raise ProviderError(f"Unknown provider: {pid}", "invalid") from None
        dmodel = pipeline.check_model(dpid, det.get("model"))
        cmodel = pipeline.check_model(cpid, clf.get("model"))
        d = _one(dpid, dmodel, W, H, len(det.get("source_classes") or ["person"]), det.get("options") or {})
        k = _boxes_per_image(ds_row["id"])
        cp = registry.get(cpid)
        ce = None
        if hasattr(cp, "estimate_classify"):
            try:
                ce = cp.estimate_classify(max(1, round(k)), cmodel, n_classes, clf.get("options") or {})
            except Exception:  # noqa: BLE001
                ce = None
        if ce is None:
            ce = _one(cpid, cmodel, 448, 448, n_classes, {**(clf.get("options") or {}), "task": "classify"})
            mult = max(1.0, k / 4)
            ce = {kk: ([v * mult for v in vv] if isinstance(vv, list) else vv) for kk, vv in ce.items()}
            assumptions.append(f"classifier cost approximated from one {round(k)}-crop request per image")
        assumptions.append(f"about {k:.1f} detected objects per image (dataset average)")
        per = {
            "usd": [d["usd"][0] + ce["usd"][0], d["usd"][1] + ce["usd"][1]],
            "tokens_in": [d["tokens_in"][0] + ce["tokens_in"][0], d["tokens_in"][1] + ce["tokens_in"][1]],
            "tokens_out": [d["tokens_out"][0] + ce["tokens_out"][0], d["tokens_out"][1] + ce["tokens_out"][1]],
            "seconds": (d.get("seconds") or 0) + (ce.get("seconds") or 0) or None,
            "usd_batch": None if not ce.get("usd_batch") else [d["usd"][0] + ce["usd_batch"][0], d["usd"][1] + ce["usd_batch"][1]],
        }
        cloud = registry.is_cloud(dpid) or registry.is_cloud(cpid)
        cloud_pids = [x for x in (dpid, cpid) if registry.is_cloud(x)]
        local_s = (d.get("seconds") or 0) if not registry.is_cloud(dpid) else 0   # one local detector: serialised
        c_priced = ce.get("priced")
        if c_priced is None:
            c_priced = not registry.is_cloud(cpid) or float((ce.get("usd") or [0, 0])[1] or 0) > 0
        unpriced = [m for m, ok in ((dmodel, d.get("priced", True)), (cmodel, c_priced)) if not ok]
    else:
        try:
            registry.get(provider)
        except KeyError:
            raise ProviderError(f"Unknown provider: {provider}", "invalid") from None
        model = pipeline.check_model(provider, model)
        per = _one(provider, model, W, H, n_classes, options)
        cloud = registry.is_cloud(provider)
        cloud_pids = [provider] if cloud else []
        local_s = 0
        unpriced = [] if per.get("priced", True) else [model]
        if per.get("tier") == "flex":    # ai-cloud hook
            assumptions.append("the Flex tier (Settings): batch prices now, answers can take minutes — Economy costs "
                               "the same")
        if per.get("guess"):
            assumptions.append("speed is a rough guess until this model has run here once")
        elif per.get("measured"):
            assumptions.append("speed measured on this machine")
    if unpriced:
        assumptions.append(f"no price on file for {', '.join(unpriced)} — its cost is unknown and is not tracked")
    per_request = per.get("seconds")
    per_image = per_request
    conc = None
    if cloud_pids and per_request:
        conc = parallel_requests(cloud_pids, concurrency)
        eff = min(conc, n) if n else conc
        per_image = max(local_s, per_request / max(1, eff))
        if eff > 1:
            assumptions.append(f"{eff} requests at a time (Settings › Parallel requests)")
    if cloud:
        assumptions.append("price table as of " + (registry.prices_as_of() or "unknown date"))
        assumptions.append("output tokens vary with how many objects each image has")
    else:
        assumptions.append("runs locally — no cost")
    pending = None
    if image_ids:
        sample = list(image_ids)[:5000]
        r = db.query_one(f"SELECT COUNT(DISTINCT image_id) n FROM ai_preds WHERE state='pending' AND kind IN "
                         f"({','.join('?' * len(store.DETECT_KINDS))}) AND image_id IN ({','.join('?' * len(sample))})",
                         [*store.DETECT_KINDS, *sample])
        pending = int(r["n"] or 0) if r else 0
    return {
        "images": n,
        "usd": [round(per["usd"][0] * n, 4), round(per["usd"][1] * n, 4)],
        "usd_batch": [round(per["usd_batch"][0] * n, 4), round(per["usd_batch"][1] * n, 4)] if per.get("usd_batch") else None,
        "tokens_in": [int(per["tokens_in"][0] * n), int(per["tokens_in"][1] * n)],
        "tokens_out": [int(per["tokens_out"][0] * n), int(per["tokens_out"][1] * n)],
        "seconds_per_image": round(per_image, 4) if per_image else per_image,
        "seconds_per_request": per_request,
        "concurrency": conc,
        "seconds_total": round(per_image * n, 2) if per_image else None,
        "per_image_usd": [round(v, 6) for v in per["usd"]],
        "tier": per.get("tier"),           # ai-cloud hook: "flex" when "now" is already billed at the batch price
        "cloud": cloud,
        "priced": not unpriced,            # False: a cloud model with no price on file — show "cost unknown", not $0
        "unpriced": unpriced,
        "images_with_pending_proposals": pending,
        "assumptions": assumptions,
    }
