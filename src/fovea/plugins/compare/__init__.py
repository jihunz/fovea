"""Compare plugin: evaluate prediction folders or stored AI runs against the dataset's ground truth."""
from __future__ import annotations

import csv
import io
import os
import threading
import time
import uuid
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, Body, HTTPException, Query
from fastapi.responses import Response

from ... import db
from ...api.common import get_dataset_or_404
from ...jobs import jobs
from ...paths import resolve, to_host
from ...plugin_api import Plugin
from . import analysis, store
from .evaluate import (GRID, METRIC_LABELS, METRICS, PRED_KINDS, ClassView, FolderSource, PredFolder, RunSource,
                       accepted_in_labels, discover_pred_dirs, evaluate, folder_label, id_ranges, parse_merge, parse_remap, processed_images,
                       run_info, scope_items, summarize)

router = APIRouter()
PLUGIN_ID = store.PLUGIN_ID


# ------------------------------------------------------------------ helpers

def _ds(ds_id: Any) -> dict:
    # POST bodies are free-form JSON: a list or a huge int would reach SQLite as a bad parameter (500)
    if not isinstance(ds_id, str) or not ds_id:
        raise HTTPException(400, "dataset_id must be a non-empty string")
    return get_dataset_or_404(ds_id)


def _names(ds: dict) -> List[str]:
    return [str(c) for c in db.loads(ds["classes"])]


def _cname(names: List[str], c: int) -> str:
    return names[c] if 0 <= c < len(names) and names[c] not in (None, "") else f"class_{c}"


def _scope(payload: dict) -> dict:
    """Image filters of an evaluation: split(s), excluded images out by default."""
    raw = payload.get("split")
    if raw is None and isinstance(payload.get("filters"), dict):
        raw = payload["filters"].get("split")
    splits = [s for s in (raw if isinstance(raw, list) else str(raw or "").split(",")) if str(s).strip()]
    skip_excluded = payload.get("skip_excluded", True) not in (False, 0, "0", "false")
    unlabeled = payload.get("unlabeled") if payload.get("unlabeled") in ("background", "skip") else "background"
    return {"split": [str(s).strip() for s in splits], "skip_excluded": skip_excluded, "unlabeled": unlabeled}


def _filters(scope: dict) -> dict:
    f: Dict[str, Any] = {}
    if scope["split"]:
        f["split"] = ",".join(scope["split"])
    if scope["skip_excluded"]:
        f["exclude_review"] = "excluded"
    return f


def _source_spec(ds_id: str, raw: Any, which: str) -> Optional[dict]:
    """Validate one prediction source from a request. Legacy form: a plain folder path string."""
    if raw in (None, "", {}):
        return None
    if isinstance(raw, str):
        raw = {"type": "folder", "path": raw}
    if not isinstance(raw, dict):
        raise HTTPException(400, f"Model {which.upper()}: invalid source")
    kind = raw.get("type") or ("run" if raw.get("run_id") is not None else "folder")
    try:
        remap = parse_remap(raw.get("remap"))
    except ValueError as e:
        raise HTTPException(400, f"Model {which.upper()}: {e}")
    if kind == "run":
        try:
            run_id = int(raw.get("run_id"))
        except (TypeError, ValueError, OverflowError):
            raise HTTPException(400, f"Model {which.upper()}: choose an AI run")
        if not 0 < run_id < 2 ** 63:                      # SQLite INTEGER range: a larger id is a 500, not a miss
            raise HTTPException(404, f"Model {which.upper()}: AI run #{run_id} not found for this dataset")
        info = run_info(ds_id, run_id)
        if not info:
            raise HTTPException(404, f"Model {which.upper()}: AI run #{run_id} not found for this dataset")
        if not info["preds"]:
            raise HTTPException(400, f"Model {which.upper()}: " + (info.get("reason") or f"AI run #{run_id} has no stored predictions to evaluate"))
        spec = {"type": "run", "run_id": run_id, "label": info["label"], "kind": info["kind"], "remap": remap}
        if info.get("audit"):
            spec["audit"] = {k: info["audit"][k] for k in ("mode", "family", "members")}
            if len(info["audit"]["members"]) > 1:          # a committee audit keeps its main model's detections only
                spec["label"] = f"Audit #{run_id} · {info['audit']['members'][0]}"
        return spec
    path = str(raw.get("path") or "").strip()
    if not path:
        raise HTTPException(400, f"Model {which.upper()}: choose a prediction folder")
    p = resolve(path)
    if not p.is_dir():
        raise HTTPException(400, f"Model {which.upper()}: folder not found: {path}")
    missing = raw.get("missing") if raw.get("missing") in ("empty", "skip") else "empty"
    return {"type": "folder", "path": str(p), "path_host": to_host(p), "label": folder_label(p), "missing": missing, "remap": remap}


def _family_text(names: List[str], family: Dict[str, List[int]]) -> str:
    """'“person” = 0 · standing, 1 · fallen' for the coverage line and warnings."""
    return "; ".join(f"“{n}” = " + ", ".join(f"{c} · {_cname(names, c)}" for c in v) for n, v in family.items())


def _class_view(payload: dict, specs: Dict[str, dict]) -> ClassView:
    """Merged classes the user asked for, plus what an audit's class family imposes: its detector can only be
    judged at the family level, and only on the family's classes (a person detector says nothing about cars)."""
    try:
        merge = parse_merge(payload.get("merge"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    only: Optional[set] = None
    labels: Dict[int, str] = {}
    for spec in specs.values():
        fam = ((spec or {}).get("audit") or {}).get("family") or {}
        if not fam:
            continue
        merge += [v for v in fam.values() if len(v) > 1]
        only = (only or set()) | {c for v in fam.values() for c in v}
        labels.update({min(v): name for name, v in fam.items()})
    return ClassView(merge, only, labels)


def _build_source(ds_id: str, spec: dict, folder: Optional[PredFolder] = None):
    if spec["type"] == "run":
        return RunSource(ds_id, spec["run_id"])
    return FolderSource(Path(spec["path"]), spec.get("missing", "empty"), folder)


_folders: "OrderedDict[str, tuple]" = OrderedDict()
_folders_lock = threading.Lock()


def _folder_cached(path: str) -> PredFolder:
    """Prediction-folder index reused by the pre-check for a short while (typing a path re-checks often)."""
    try:
        mt = Path(path).stat().st_mtime
    except OSError:
        mt = 0
    now = time.monotonic()
    with _folders_lock:
        hit = _folders.get(path)
        if hit and hit[0] == mt and now - hit[1] < 45:
            return hit[2]
    folder = PredFolder(Path(path))
    with _folders_lock:
        _folders[path] = (mt, now, folder)
        while len(_folders) > 8:
            _folders.popitem(last=False)
    return folder


def _coverage(ds_id: str, spec: dict, items: List[dict], folder: Optional[PredFolder] = None) -> dict:
    """How many images in scope this source has predictions for, by split, and what to do about a poor match."""
    by_split: Dict[str, Dict[str, int]] = {}
    for it in items:
        by_split.setdefault(it["split"] or "", {"images": 0, "matched": 0})["images"] += 1
    out: Dict[str, Any] = {"images": len(items), "matched": 0, "ambiguous": 0, "by_split": by_split, "hints": []}
    if spec["type"] == "run":
        info = run_info(ds_id, spec["run_id"]) or {}
        run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (spec["run_id"],)) or {}
        covered, replaced = processed_images(run) if run else (None, set())
        if covered is None:
            marks = ",".join("?" * len(PRED_KINDS))
            covered = {r["image_id"] for r in db.query(f"SELECT DISTINCT image_id FROM ai_preds WHERE run_id=? AND kind IN ({marks})",
                                                          (spec["run_id"], *PRED_KINDS))}
        covered = covered - replaced
        scoped = set()
        for it in items:
            if it["id"] in covered:
                scoped.add(it["id"])
                out["matched"] += 1
                by_split[it["split"] or ""]["matched"] += 1
        # boxes and confidences of the images in scope, not of the whole run (it may span other splits)
        marks = ",".join("?" * len(PRED_KINDS))
        preds = noconf = 0
        for r in db.query(f"""SELECT image_id, COUNT(*) n, SUM(conf IS NULL) noconf FROM ai_preds WHERE run_id=? AND kind IN ({marks})
                              GROUP BY image_id""", (spec["run_id"], *PRED_KINDS)):
            if r["image_id"] in scoped:
                preds += r["n"]
                noconf += r["noconf"] or 0
        out.update({"preds": preds, "run_preds": info.get("preds", 0), "has_conf": not noconf,
                    "accepted": accepted_in_labels(spec["run_id"], scoped), "processed_known": info.get("processed_known", False),
                    "run_images": info.get("images", 0), "images_with_preds": info.get("images_with_preds", 0),
                    "min_conf": info.get("min_conf"), "replaced": len(replaced & {it["id"] for it in items}),
                    "kind": info.get("kind")})
        if info.get("audit"):
            out["audit"] = info["audit"]
            out["family_text"] = _family_text(_names(_ds(ds_id)), info["audit"].get("family") or {})
        if items and out["matched"] < 0.5 * len(items):
            # a run usually covers one split: say which, so the scope can follow it (one click in the form)
            good = sorted(s for s, v in by_split.items() if v["matched"] and v["matched"] >= 0.5 * v["images"])
            if good and len(good) < len(by_split):
                out["hints"].append({"kind": "split", "splits": good,
                                     "text": f"This run covers {', '.join(s or '(no split)' for s in good)}: evaluate "
                                             f"{'it' if len(good) == 1 else 'them'} alone to leave the other images out."})
        return out
    folder = folder or _folder_cached(spec["path"])
    src = FolderSource(Path(spec["path"]), spec.get("missing", "empty"), folder)
    src.prepare(items)
    sample_files = []
    for idx, it in enumerate(items):
        how = src.how[idx]
        if how in ("path", "split", "stem"):
            out["matched"] += 1
            by_split[it["split"] or ""]["matched"] += 1
            if len(sample_files) < 8:
                sample_files.append(src.matched[idx])
        elif how == "ambiguous":
            out["ambiguous"] += 1
    out["files"] = folder.files
    out["truncated"] = folder.truncated
    out["by_stem"] = sum(1 for h in src.how if h == "stem")
    confs = [r[5] is not None for f in sample_files for r in _read_head(Path(spec["path"]) / f)]
    out["has_conf"] = bool(confs) and all(confs)
    out["sample_rows"] = len(confs)
    if items and out["matched"] < 0.5 * len(items):
        out["hints"] = _folder_hints(ds_id, spec, folder, items)
    return out


def _read_head(path: Path) -> List[list]:
    from .evaluate import read_pred_file
    return read_pred_file(path)[:20]


def _folder_hints(ds_id: str, spec: dict, folder: PredFolder, items: List[dict]) -> List[dict]:
    """Suggestions when a folder matches few images: other splits it does match, or a parent with split folders."""
    hints: List[dict] = []
    every = db.query("SELECT id, rel_path, split, has_label FROM images WHERE dataset_id=? ORDER BY id", (ds_id,))
    stride = max(1, len(every) // 3000)
    sample = every[::stride]
    src = FolderSource(Path(spec["path"]), "empty", folder)
    hit_split: Dict[str, int] = {}
    tot_split: Dict[str, int] = {}
    by: Dict[str, List[dict]] = {}
    for it in sample:
        by.setdefault(it["split"], []).append(it)
    for sp, group in by.items():                 # one split at a time: which split do these files belong to?
        src.prepare(group)
        tot_split[sp] = len(group)
        hit_split[sp] = sum(1 for h in src.how if h in ("path", "split", "stem"))
    in_scope = {it["split"] for it in items}
    good = [s for s, n in hit_split.items() if n and n >= 0.5 * tot_split[s]]
    if good and not set(good) & in_scope:
        hints.append({"kind": "split", "splits": sorted(good),
                      "text": f"These files match images in {', '.join(sorted(s or '(no split)' for s in good))}, not in the chosen scope."})
    root = Path(spec["path"])
    parent = root.parent
    try:
        subs = {p.name for p in parent.iterdir() if p.is_dir()}
    except OSError:
        subs = set()
    wanted = {s for s in in_scope if s}
    if wanted and wanted <= subs and root.name not in wanted:
        hints.append({"kind": "parent", "path": str(parent), "path_host": to_host(parent),
                      "text": f"The parent folder {parent.name}/ has {', '.join(sorted(wanted))}/ — use it to match every split."})
    return hints


def _meta_or_404(ds_id: str, eid: str) -> dict:
    meta = store.get_meta(ds_id, eid)
    if meta is None:
        raise HTTPException(404, "Evaluation not found")
    return meta


def _v2(meta: dict) -> None:
    if meta.get("legacy"):
        raise HTTPException(409, "This evaluation was saved by an earlier version of Compare. Re-run it to use this feature.")


def _best(curves: dict, classes=None) -> Optional[dict]:
    """Confidence threshold with the highest F1 over all boxes of the given classes (ties: the higher threshold)."""
    if not curves.get("conf_available", True):
        return None
    best = None
    for k in range(GRID):
        s = summarize(curves, k, classes)
        if best is None or s["f1"] >= best["f1"]:
            best = {"k": k, "conf": k / 100, "f1": s["f1"], "precision": s["precision"], "recall": s["recall"]}
    return best if best and best["f1"] > 0 else None


def _list_entry(m: dict) -> dict:
    keys = ("id", "name", "created_at", "images", "iou", "conf", "metric", "scope", "legacy", "summary", "classes_view")
    out = {k: m.get(k) for k in keys}
    if m.get("legacy"):
        out["sources"] = {"a": {"type": "folder", "label": Path(m.get("pred_a") or "").name, "path_host": m.get("pred_a_host")},
                          "b": ({"type": "folder", "label": Path(m.get("pred_b") or "").name, "path_host": m.get("pred_b_host")} if m.get("pred_b") else None)}
        f = m.get("filters") or {}
        out["scope"] = {"split": [s for s in str(f.get("split") or "").split(",") if s], "skip_excluded": False, "unlabeled": "background"}
        out["metric"] = "voc"
    else:
        out["sources"] = {s: ({k: v for k, v in src.items() if k in ("type", "label", "path_host", "run_id")} if src else None)
                          for s, src in (m.get("sources") or {}).items()}
    return out


# ------------------------------------------------------------------ sources

@router.get("/discover")
def discover(path: str = Query(...), dataset_id: Optional[str] = Query(None)):
    root = resolve(path)
    if not root.is_dir():
        raise HTTPException(404, f"Not a directory: {path}")
    sample = None
    if dataset_id:
        rows = db.query("SELECT id, rel_path, split FROM images WHERE dataset_id=? ORDER BY id", (dataset_id,))
        stride = max(1, len(rows) // 400)
        sample = rows[::stride] or None
    found = discover_pred_dirs(root, sample)
    gt_dirs = set()
    if dataset_id:
        row = db.query_one("SELECT layout FROM datasets WHERE id=?", (dataset_id,))
        for src in (db.loads(row["layout"], {}) if row else {}).get("sources", []):
            if src.get("label_dir"):
                gt_dirs.add(str(Path(src["label_dir"]).resolve()))
    gt_parents = {str(Path(d).parent) for d in gt_dirs}
    for f in found:
        f["path_host"] = to_host(f["path"])
        rp = str(Path(f["path"]).resolve())
        f["is_gt"] = rp in gt_dirs or (f["kind"] == "parent" and rp in gt_parents)
    found.sort(key=lambda e: e["is_gt"])            # stable: the dataset's own labels go last
    return {"root": str(root), "root_host": to_host(root), "dirs": found}


@router.get("/runs")
def list_runs(dataset_id: str = Query(...)):
    """AI runs with stored predictions (batch, audit, benchmark ...) usable as a prediction source. Audits that kept
    only their findings are listed too, marked not evaluable with the reason, so nobody hunts for them."""
    ds = _ds(dataset_id)
    marks = ",".join("?" * len(PRED_KINDS))
    rows = db.query(f"""SELECT r.id FROM ai_runs r WHERE r.dataset_id=? AND r.kind NOT IN ('propose','interp','prev_frame') AND r.status != 'deleted'
                        AND (r.kind = 'audit' OR EXISTS (SELECT 1 FROM ai_preds p WHERE p.run_id=r.id AND p.kind IN ({marks})))
                        ORDER BY r.created_at DESC LIMIT 200""", (dataset_id, *PRED_KINDS))
    names = _names(ds)
    runs = [run_info(dataset_id, r["id"]) for r in rows]
    for r in runs:
        if r and r.get("audit") and r["audit"].get("family"):
            r["family_text"] = _family_text(names, r["audit"]["family"])
    return {"runs": [r for r in runs if r]}


@router.post("/check")
def check_source(payload: dict = Body(...)):
    """Pre-flight for one source: how many images in scope it matches (before anything runs)."""
    ds_id = payload.get("dataset_id") or ""
    _ds(ds_id)
    spec = _source_spec(ds_id, payload.get("source"), str(payload.get("which") or "a"))
    if spec is None:
        raise HTTPException(400, "No source")
    scope = _scope(payload)
    items = scope_items(ds_id, _filters(scope))
    return {"coverage": _coverage(ds_id, spec, items), "label": spec["label"]}


def _class_map(spec: Any) -> Optional[Dict[str, Any]]:
    """'person:0, 2:-, *:skip' or {..} -> the class_map dict ai-core's closed-vocabulary providers take."""
    if not spec:
        return None
    if isinstance(spec, dict):
        return spec
    out: Dict[str, Any] = {}
    for tok in str(spec).replace(";", ",").split(","):
        tok = tok.strip()
        if not tok:
            continue
        for sep in ("->", "→", ":", "="):
            if sep in tok:
                a, b = (x.strip() for x in tok.split(sep, 1))
                break
        else:
            raise HTTPException(400, f"Class map entry '{tok}' needs the form model_class:dataset_class (e.g. person:0)")
        out[a] = "skip" if b in ("-", "", "skip", "none") else (int(b) if b.lstrip("-").isdigit() else b)
    return out


@router.post("/runs/start")
def start_model_run(payload: dict = Body(...)):
    """Run a LOCAL model over the evaluation scope and keep every prediction (state 'info': never shown as proposals)
    as an AI run Compare can evaluate. Cloud models run from the AI tab, where cost estimates and budgets live."""
    ds_id = payload.get("dataset_id") or ""
    ds = _ds(ds_id)
    try:
        from ...ai import pipeline, registry, runner
        from ...ai import store as ai_store
        from ...ai.base import ProviderError
        from ...ai.secrets import scrub
    except ImportError as e:  # pragma: no cover - the AI package is part of the app, but keep Compare usable
        raise HTTPException(503, f"AI providers are not available: {e}")
    provider = str(payload.get("provider") or "")
    try:
        if registry.is_cloud(provider):
            raise HTTPException(400, "Cloud models run from the AI tab (cost estimate and budget): run a benchmark or a batch "
                                     "with target “Proposals”, then pick that run here.")
        conf = pipeline.as_conf(payload.get("conf"), 0.01, "conf")
        options: Dict[str, Any] = {}
        cmap = _class_map(payload.get("class_map"))
        if cmap:
            options["class_map"] = cmap
        model, _involved = runner.validate_detect(ds, provider, payload.get("model"), conf, None, options)
    except ProviderError as e:
        raise HTTPException(400, scrub(str(e)))
    scope = _scope(payload)
    items = scope_items(ds_id, _filters(scope))
    if not items:
        raise HTTPException(400, "No images in this scope")
    if payload.get("dry_run"):
        return {"images": len(items), "model": model, "provider": provider}
    label = f"{provider} · {model}"
    params = {"config": {"provider": provider, "model": model, "conf": conf, "options": options, "label": label},
              "iou": 0.5, "images": len(items), "source": "compare", "scope": scope}
    run_id = ai_store.create_run(ds_id, "benchmark", provider, model, params)

    # Several frames per model call where the local provider batches them (ai-core's detect_images): same boxes,
    # fewer round trips through the model.
    batched = hasattr(pipeline, "detect_images") and hasattr(pipeline, "batch_capable") and pipeline.batch_capable(provider, options)
    step = 8 if batched else 1
    # thousands of new ai_preds rows: keep the planner statistics truthful (ai-core's shared helper)
    after_insert = getattr(runner, "after_bulk_insert", None) or (lambda: ai_store.refresh_planner_stats(throttle=False))

    def run(job):
        done: List[int] = []
        errors, last_error, t0 = 0, None, time.time()
        status = "done"

        def detect_chunk(imgs: List[dict]) -> List[Any]:
            if step > 1:
                return pipeline.detect_images(ds, imgs, provider, model, conf, None, options)
            out: List[Any] = []
            for img in imgs:
                try:
                    out.append(pipeline.detect_image(ds, img, provider, model, conf, None, options))
                except ProviderError as e:
                    out.append(e)
            return out

        try:
            for k0 in range(0, len(items), step):
                rate = f" · {len(done) / (time.time() - t0):.1f} images/s" if done else ""
                # counts first: the job banner drops a leading "done/total" it already shows
                job.update(done=k0, total=len(items), message=f"{k0:,}/{len(items):,} · {model}{rate}")
                imgs = []
                for it in items[k0:k0 + step]:
                    img = db.query_one("SELECT * FROM images WHERE id=?", (it["id"],))
                    if not img:
                        continue
                    if not os.path.isfile(img["abs_path"]):      # ai-core treats this as fatal; here it is one image
                        errors += 1
                        last_error = "Image file missing on disk"
                        continue
                    imgs.append(img)
                try:
                    outs = detect_chunk(imgs) if imgs else []
                except ProviderError as e:                       # a configuration error: the whole call failed
                    outs = [e] * len(imgs)
                except Exception as e:  # noqa: BLE001 - one unreadable image must not end the run
                    if len(imgs) == 1:
                        outs = [e]
                    else:                                        # find the bad frame: retry one by one
                        step_outs = []
                        for img in imgs:
                            try:
                                step_outs.append(pipeline.detect_image(ds, img, provider, model, conf, None, options))
                            except Exception as e1:  # noqa: BLE001
                                step_outs.append(e1)
                        outs = step_outs
                for img, out in zip(imgs, outs):
                    if isinstance(out, BaseException):
                        errors += 1
                        last_error = scrub(str(out) if isinstance(out, ProviderError) else f"{type(out).__name__}: {out}")
                        if isinstance(out, ProviderError) and getattr(out, "fatal", False):
                            raise out
                        continue
                    ai_store.insert_preds(run_id, ds_id, img["id"], out["preds"], "info")
                    ai_store.add_usage(run_id, out["usage"], images=1, boxes=len(out["preds"]))
                    done.append(img["id"])
        except Exception as e:
            status = "cancelled" if type(e).__name__ == "JobCancelled" else "error"
            ai_store.finish_run(run_id, status, result={"image_ranges": id_ranges(done), "errors": errors, "source": "compare",
                                                        "last_error": last_error}, error=None if status == "cancelled" else scrub(str(e)),
                                images=len(done))
            after_insert()
            raise
        job.update(done=len(items), total=len(items), message=f"Scoring {len(done):,} images")
        result = _run_summary(ds_id, run_id, done, items, conf)
        result.update({"image_ranges": id_ranges(done), "errors": errors, "last_error": last_error, "source": "compare",
                       "config": params["config"], "run_id": run_id,
                       "sec_per_image": round((time.time() - t0) / len(done), 3) if done else None, "usd_per_image": 0.0, "cost_usd": 0.0})
        ai_store.finish_run(run_id, "done" if done else "error", result=result, images=len(done),
                            error=None if done else (last_error or "No image could be processed"))
        after_insert()
        if not done:
            raise RuntimeError(last_error or "No image could be processed")
        job.update(message=f"Done · {len(done):,} images" + (f" · {errors:,} failed" if errors else ""))
        return {"run_id": run_id, "images": len(done), "errors": errors, "last_error": last_error}

    job = jobs.submit("compare_run", run, dataset_id=ds_id, message="Loading the model", pool="ai")
    ai_store.set_run_job(run_id, job.id)
    return {"job": job.to_dict(), "run_id": run_id, "images": len(items)}


def _run_summary(ds_id: str, run_id: int, done: List[int], items: List[dict], conf: float) -> dict:
    """Benchmark-style headline numbers for the AI tab (IoU 0.5, conf >= 0.25, COCO AP)."""
    keep = set(done)
    sel = [it for it in items if it["id"] in keep]
    if not sel:
        return {}
    src = RunSource(ds_id, run_id)
    src.covered_ids = keep
    res = evaluate(None, ds_id, sel, {"a": src}, iou=0.5, method="coco")
    k = max(25, int(round(conf * 100)))
    sm = summarize(res["curves"]["a"], k, None, res["ious"], res["primary"])
    return {"precision": sm["precision"], "recall": sm["recall"], "f1": sm["f1"], "ap50": sm["map"], "iou": 0.5,
            "conf_threshold": k / 100, "images": len(sel), "tp": sm["tp"], "fp": sm["fp"], "fn": sm["fn"]}


# ------------------------------------------------------------------ evaluation

@router.post("/evaluate")
def start_evaluate(payload: dict = Body(...)):
    ds_id = payload.get("dataset_id") or ""
    ds = _ds(ds_id)
    a = _source_spec(ds_id, payload.get("a") if payload.get("a") is not None else payload.get("pred_a"), "a")
    if a is None:
        raise HTTPException(400, "Model A: choose a prediction folder or an AI run")
    b = _source_spec(ds_id, payload.get("b") if payload.get("b") is not None else payload.get("pred_b"), "b")
    try:
        iou_thr = float(payload.get("iou", 0.5))
        conf_thr = float(payload.get("conf", 0.25))
    except (TypeError, ValueError):
        raise HTTPException(400, "iou and conf must be numbers")
    if not 0 < iou_thr <= 1:
        raise HTTPException(400, "IoU threshold must be > 0 and ≤ 1")
    if not 0 <= conf_thr < 1:
        raise HTTPException(400, "Confidence threshold must be ≥ 0 and < 1")
    metric = payload.get("metric") or "coco"
    if metric not in METRICS:
        raise HTTPException(400, f"metric must be one of {', '.join(METRICS)}")
    scope = _scope(payload)
    filters = _filters(scope)
    names = _names(ds)
    specs = {"a": a, **({"b": b} if b else {})}
    cv = _class_view(payload, specs)
    merge_user = parse_merge(payload.get("merge"))        # what the form said, for "Re-run" (validated just above)
    eid = uuid.uuid4().hex[:10]
    user_name = str(payload.get("name") or "").strip()
    name = user_name or (f"{a['label']} vs {b['label']}" if b else a["label"])
    k0 = analysis.grid_k(conf_thr)

    def run(job):
        out = work(job)
        # Activity lists a finished job by its last message: say how it ended, not the last step it reported.
        # Set directly: job.update() raises on a late cancel, which would mark a saved evaluation cancelled.
        job.message = (f"Done · {out['images']:,} image{'' if out['images'] == 1 else 's'}" if out.get("saved")
                       else f"Not saved · {out.get('reason') or 'nothing to evaluate'}")
        return out

    def work(job):
        items = scope_items(ds_id, filters)
        if not items:
            return {"saved": False, "reason": "No images in this scope."}
        sources = {side: _build_source(ds_id, spec) for side, spec in specs.items()}
        job.update(0, len(items), "Indexing prediction files")
        res = evaluate(job, ds_id, items, sources, iou=iou_thr, method=metric, unlabeled=scope["unlabeled"],
                       remap={side: spec.get("remap") or {} for side, spec in specs.items()}, classview=cv)
        warnings: List[dict] = []
        for side, spec in specs.items():
            who = f"Model {side.upper()}"
            if spec["type"] == "folder":
                matched = res["pred_files"][side]
                if matched == 0 and res["images"]:
                    hints = _folder_hints(ds_id, spec, sources[side].folder, items)
                    return {"saved": False, "reason": f"{who}: no prediction file matched any of the {res['images']:,} images in scope.",
                            "hints": hints, "which": side}
                if res["images"] and matched < 0.5 * res["images"]:
                    warnings.append({"level": "warn", "text": f"{who}: only {matched:,} of {res['images']:,} images {_v(matched, 'has', 'have')} a prediction file. "
                                                              f"The others count as 'no detections' — every box on them is a miss."})
            else:
                warnings += _run_warnings(who, spec, sources[side], names)
            if res["curves"][side]["preds_without_conf"]:
                nc = res["curves"][side]["preds_without_conf"]
                warnings.append({"level": "info", "text": f"{who}: {_n(nc, 'prediction')} {_v(nc, 'has', 'have')} no confidence. "
                                                          f"AP needs ranked scores and is not reported; the threshold cannot filter them."})
        if res["images"] == 0:
            reasons = ", ".join(f"{v:,} {k.replace('_', ' ')}" for k, v in res["skipped"].items() if v)
            return {"saved": False, "reason": f"Every image was skipped ({reasons})."}
        if res["skipped"]["ambiguous"]:
            amb = res["skipped"]["ambiguous"]
            warnings.append({"level": "warn", "text": f"{_n(amb, 'image')} {_v(amb, 'was', 'were')} skipped: {_v(amb, 'its', 'their')} prediction file name is ambiguous "
                                                      f"(the same file name exists in several splits or folders). Use split subfolders, or evaluate one split at a time."})
        if res["skipped"]["not_covered"]:
            runs_only = all(s["type"] == "run" or s.get("missing") != "skip" for s in specs.values())
            why = "an AI run covers only the images it processed" if runs_only else "a source has no predictions for them"
            nc = res["skipped"]["not_covered"]
            warnings.append({"level": "info", "text": f"{_n(nc, 'image')} in scope {_v(nc, 'was', 'were')} not evaluated: {why}."})
        if res["unlabeled_background"]:
            ub = res["unlabeled_background"]
            warnings.append({"level": "info", "text": f"{_n(ub, 'image')} {_v(ub, 'has', 'have')} no label file and {_v(ub, 'counts', 'count')} as background "
                                                      f"(every prediction on {_v(ub, 'it', 'them')} is a false positive)."})
        curves = res["curves"]
        summary = {side: summarize(curves[side], k0, None, res["ious"], res["primary"]) for side in specs}
        best = {side: _best(curves[side]) for side in specs}
        classes_seen = sorted({int(c) for side in specs for c in curves[side]["classes"]})
        meta = {"v": 2, "id": eid, "dataset_id": ds_id, "name": name, "named": bool(user_name), "created_at": db.now(),
                "sources": {side: {k: v for k, v in spec.items() if k != "remap"} for side, spec in specs.items()},
                "remap": {side: {str(k): v for k, v in (spec.get("remap") or {}).items()} for side, spec in specs.items()},
                "classes_view": cv.to_meta(names), "merge": merge_user,
                "iou": iou_thr, "conf": k0 / 100, "metric": metric, "scope": scope, "filters": filters,
                "images": res["images"], "skipped": res["skipped"], "pred_files": res["pred_files"],
                "ious": res["ious"], "primary": res["primary"], "summary": summary, "best": best,
                "classes_seen": classes_seen, "names": names, "warnings": warnings}
        meta["sources"].setdefault("b", None)
        store.save(ds_id, meta, curves, res["data"])
        return {"saved": True, "eval_id": eid, "images": res["images"], "summary": summary}

    job = jobs.submit("compare", run, dataset_id=ds_id, message="Queued")
    return {"job": job.to_dict(), "eval_id": eid}


def _n(n: int, one: str, many: str = "") -> str:
    """'1 image' / '1,204 images'."""
    return f"{n:,} {one if n == 1 else many or one + 's'}"


def _v(n: int, one: str, many: str) -> str:
    """The word that agrees with a count: _v(1, 'was', 'were')."""
    return one if n == 1 else many


def _run_warnings(who: str, spec: dict, src: RunSource, names: List[str]) -> List[dict]:
    """What a reader must know before trusting the numbers of an AI-run source."""
    out: List[dict] = []
    rid = spec["run_id"]
    audit = src.audit or {}
    if audit.get("family"):
        out.append({"level": "info", "text": f"{who} is a label audit in geometry mode: its detector finds "
                                             f"{_family_text(names, audit['family'])}. Each family is counted as one class "
                                             f"and boxes of other classes are left out of every number."})
    if len(audit.get("members") or []) > 1:
        out.append({"level": "info", "text": f"{who}: this audit asked {len(audit['members'])} models; it keeps the detections of "
                                             f"its main model only ({audit['members'][0]}), so that is the model scored here."})
    if src.replaced:
        nr = len(src.replaced)
        out.append({"level": "warn", "text": f"{who}: {_n(nr, 'image')} lost this run's proposals to a later proposal run "
                                             f"and {_v(nr, 'is', 'are')} left out ({_v(nr, 'its', 'their')} stored predictions are gone)."})
    if not src.processed_known:
        out.append({"level": "info", "text": f"{who}: the run does not record which images it processed, so only the "
                                             f"images it stored predictions for are evaluated. Recall may look higher than it is."})
    # only accepted boxes the labels still hold (not those deleted since or taken out by restoring an older version,
    # nor accepts of a box already there; restoring the accept's own version puts its proposals back to pending)
    acc = accepted_in_labels(rid, [i for i in src.ids if i in src.covered_ids])
    if acc:
        out.append({"level": "warn", "text": f"{who}: {acc:,} of these predictions {_v(acc, 'was', 'were')} accepted into the labels and "
                                             f"{_v(acc, 'is', 'are')} still there, so the ground truth contains {_v(acc, 'it', 'them')} "
                                             f"and the scores are optimistic."})
    fx = audit.get("fixes_applied") or 0
    if fx:
        out.append({"level": "warn", "text": f"{who}: {_n(fx, 'finding')} of this audit {_v(fx, 'was', 'were')} applied to the labels, "
                                             f"so the ground truth now partly comes from this model and the scores are optimistic."})
    mc = db.query_one(f"SELECT MIN(conf) m FROM ai_preds WHERE run_id=? AND kind IN ({','.join('?' * len(PRED_KINDS))}) AND conf IS NOT NULL",
                      (rid, *PRED_KINDS))
    floor = (mc or {}).get("m")
    if floor is not None and floor >= 0.1:
        out.append({"level": "info", "text": f"{who}: predictions were stored only from confidence {floor:.2f} up, so AP leaves out "
                                             f"everything the model found below that (yolo val keeps scores down to 0.001)."})
    return out


@router.get("/evals")
def list_evals(dataset_id: str = Query(...)):
    return {"evals": [_list_entry(m) for m in store.list_meta(dataset_id)]}


@router.get("/evals/{eval_id}")
def get_eval(eval_id: str, dataset_id: str = Query(...)):
    meta = _meta_or_404(dataset_id, eval_id)
    out = {"eval": {**meta, "metric_label": METRIC_LABELS.get(meta.get("metric") or "voc", "")}}
    if meta.get("legacy"):
        out["eval"]["sources"] = _list_entry(meta)["sources"]
        out["eval"]["scope"] = _list_entry(meta)["scope"]
        return out
    out["curves"] = store.get_curves(dataset_id, eval_id)
    return out


@router.patch("/evals/{eval_id}")
def rename_eval(eval_id: str, dataset_id: str = Query(...), payload: dict = Body(...)):
    name = str(payload.get("name") or "").strip()
    if not name:
        raise HTTPException(400, "Name is required")
    meta = store.update_meta(dataset_id, eval_id, {"name": name[:200], "named": True})
    if meta is None:
        raise HTTPException(404, "Evaluation not found")
    return {"eval": _list_entry(meta)}


@router.delete("/evals/{eval_id}")
def delete_eval(eval_id: str, dataset_id: str = Query(...)):
    store.delete(dataset_id, eval_id)
    analysis.forget(eval_id)
    return {"ok": True}


def _data_or_409(dataset_id: str, eval_id: str) -> tuple:
    meta = _meta_or_404(dataset_id, eval_id)
    _v2(meta)
    data = store.get_data(dataset_id, eval_id)
    if data is None:
        raise HTTPException(404, "Evaluation data is missing")
    return meta, data


@router.get("/evals/{eval_id}/confusion")
def get_confusion(eval_id: str, dataset_id: str = Query(...), side: str = Query("a"), conf: float = Query(0.25),
                  classes: Optional[str] = Query(None)):
    meta, data = _data_or_409(dataset_id, eval_id)
    if side not in data["sides"]:
        raise HTTPException(400, f"No model {side.upper()} in this evaluation")
    return analysis.confusion(data, side, analysis.grid_k(conf), analysis.parse_classes(classes))


def _rows_args(data, conf, classes, show, sort, side):
    if show not in analysis.SHOWS:
        raise HTTPException(400, f"show must be one of {', '.join(analysis.SHOWS)}")
    if sort not in analysis.SORTS:
        raise HTTPException(400, f"sort must be one of {', '.join(analysis.SORTS)}")
    if not isinstance(side, str) or side not in data["sides"]:
        side = "a"
    return analysis.grid_k(conf), analysis.parse_classes(classes), side


@router.get("/evals/{eval_id}/rows")
def get_rows(eval_id: str, dataset_id: str = Query(...), conf: float = Query(0.25), classes: Optional[str] = Query(None),
             show: str = Query("errors"), sort: str = Query("errors"), side: str = Query("a"), q: str = Query(""),
             split: str = Query(""), cell: str = Query(""), offset: int = Query(0, ge=0), limit: int = Query(100, ge=1, le=1000)):
    meta = _meta_or_404(dataset_id, eval_id)
    if meta.get("legacy"):
        rows = store.get_legacy_rows(dataset_id, eval_id)
        if show == "errors":
            rows = [r for r in rows if any((r.get(s) or {}).get("fp", 0) + (r.get(s) or {}).get("fn", 0) for s in ("a", "b"))]
        if q:
            rows = [r for r in rows if q.lower() in r["rel_path"].lower()]
        page = [{**r, "i": offset + n} for n, r in enumerate(rows[offset:offset + limit])]
        analysis.attach_reviews(dataset_id, page)
        return {"total": len(rows), "offset": offset, "items": page, "legacy": True}
    data = store.get_data(dataset_id, eval_id)
    if data is None:
        raise HTTPException(404, "Evaluation data is missing")
    k, cls_set, side = _rows_args(data, conf, classes, show, sort, side)
    idx, counts = analysis.table(eval_id, data, k, cls_set, show=show, sort=sort, side=side, q=q.strip(), split=split, cell=cell)
    page = analysis.row_items(data, idx[offset:offset + limit], counts, cls_set)
    analysis.attach_reviews(dataset_id, page)
    return {"total": len(idx), "offset": offset, "items": page, "k": k}


@router.get("/evals/{eval_id}/image/{image_id}")
def get_eval_image(eval_id: str, image_id: int, dataset_id: str = Query(...), conf: Optional[float] = Query(None),
                   classes: Optional[str] = Query(None)):
    meta = _meta_or_404(dataset_id, eval_id)
    img = db.query_one("""SELECT i.id, i.rel_path, i.split, i.width, i.height, i.has_label, r.status, r.note
                          FROM images i LEFT JOIN reviews r ON r.dataset_id=i.dataset_id AND r.rel_path=i.rel_path
                          WHERE i.id=? AND i.dataset_id=?""", (image_id, dataset_id))
    if not img:
        raise HTTPException(404, "Image not found")
    image = {k: img[k] for k in ("id", "rel_path", "split", "width", "height", "has_label")}
    image["review"] = {"status": img["status"], "note": img["note"] or ""} if img["status"] else None
    k = analysis.grid_k(conf if conf is not None else meta.get("conf", 0.25))
    if meta.get("legacy"):
        return {"image": image, "detail": _legacy_detail(meta, image, k)}
    data = store.get_data(dataset_id, eval_id)
    return {"image": image, "detail": analysis.image_detail(dataset_id, eval_id, meta, data, image, k, analysis.parse_classes(classes))}


def _legacy_detail(meta: dict, image: dict, k: int) -> dict:
    """Evaluations saved by Compare 1.x have no per-image data: recompute the image live from its folders."""
    from .evaluate import pred_file_for
    m = {"sources": {}, "ious": [meta.get("iou", 0.5)], "primary": 0, "metric": "voc", "remap": {}}
    files: Dict[str, str] = {}
    for side in ("a", "b"):
        d = meta.get(f"pred_{side}")
        if not d:
            continue
        f = pred_file_for(Path(d), image["rel_path"], image["split"])
        m["sources"][side] = {"type": "folder", "path": d, "path_host": meta.get(f"pred_{side}_host") or d}
        files[side] = str(f.relative_to(d)) if f else ""
    det = analysis.image_detail(meta.get("dataset_id") or "", meta["id"], m, None, image, k, None, files=files)
    det["in_eval"] = True                 # no per-image record to check against; the row list came from this evaluation
    return det


# ------------------------------------------------------------------ flags (review marks)

def _apply_flags(ds_id: str, eid: str, meta: dict, targets: Dict[int, str], include_approved: bool, op: str) -> dict:
    """Flag images with a note line naming this evaluation. Never touches excluded images; approved ones only when
    asked. Remembers the previous state of each image so the marks can be undone."""
    data = store.get_data(ds_id, eid)
    rels = {int(data["img"][i]): data["rel"][i] for i in range(data["n"])} if data else {}
    rows = {}
    rel_list = [rels[i] for i in targets if i in rels]
    for c in range(0, len(rel_list), 500):
        chunk = rel_list[c:c + 500]
        for r in db.query("SELECT rel_path, status, note FROM reviews WHERE dataset_id=? AND rel_path IN (%s)" % ",".join("?" * len(chunk)), [ds_id, *chunk]):
            rows[r["rel_path"]] = r
    res = {"flagged": 0, "updated": 0, "skipped_excluded": 0, "skipped_approved": 0}
    now = db.now()
    with db.transaction() as conn:
        ledger = store.get_flags(ds_id, eid)
        entries = ledger.setdefault("entries", {})
        for image_id, line in targets.items():
            rel = rels.get(image_id)
            if rel is None:
                continue
            cur = rows.get(rel)
            status = cur["status"] if cur else None
            if status == "excluded":
                res["skipped_excluded"] += 1
                continue
            if status == "approved" and not include_approved:
                res["skipped_approved"] += 1
                continue
            note = analysis.with_line(cur["note"] if cur else "", eid, line)
            # one entry per operation: undoing an operation restores the state right before it, not an older one
            entry = entries.setdefault(rel, {"image_id": image_id, "stack": []})
            entry["stack"].append({"op": op, "prev": {"status": status, "note": (cur["note"] if cur else "") or ""}})
            conn.execute("""INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)
                            ON CONFLICT(dataset_id, rel_path) DO UPDATE SET status=excluded.status, note=excluded.note, updated_at=excluded.updated_at""",
                         (ds_id, rel, "flagged", note, now))
            res["updated" if status == "flagged" else "flagged"] += 1
        ops = ledger.setdefault("ops", [])
        ops.append({"op": op, "at": now, "n": res["flagged"] + res["updated"]})
        ledger["ops"] = ops[-20:]
        store.put_flags(conn, ds_id, eid, ledger)
    return res


def _id_list(raw: Any, field: str) -> List[int]:
    """Image ids from a JSON body: a list of ints (or digit strings); anything else is a 400, not a 500."""
    if not isinstance(raw, list):
        raise HTTPException(400, f"{field} must be a list of image ids")
    out = []
    for x in raw:
        if isinstance(x, str) and x.isascii() and x.isdigit():
            x = int(x)
        if isinstance(x, int) and not isinstance(x, bool) and 0 < x < 2 ** 63:
            out.append(x)
    return out


def _flag_targets(eid: str, meta: dict, data: dict, k: int, cls_set, crit: dict) -> Dict[int, str]:
    sides = list(data["sides"])
    want_sides = sides if crit.get("side") in (None, "any") else [crit["side"]] if crit["side"] in sides else sides
    kinds = [x for x in ("fn", "fp", "sw") if crit.get(x, x != "sw")]
    try:
        minimum = max(1, int(crit.get("min") or 1))
    except (TypeError, ValueError):
        minimum = 1
    counts = {s: analysis.image_counts(eid, data, s, k, cls_set) for s in sides}
    col = {"tp": 0, "fp": 1, "fn": 2, "sw": 3}
    targets: Dict[int, str] = {}
    for i in range(data["n"]):
        if not any(sum(counts[s][col[x]][i] for x in kinds) >= minimum for s in want_sides):
            continue
        per = {s: {x: counts[s][col[x]][i] for x in ("fn", "fp", "sw")} for s in sides}
        targets[int(data["img"][i])] = analysis.note_line(eid, meta, k, per)
    return targets


@router.post("/evals/{eval_id}/flag")
def flag_images(eval_id: str, payload: dict = Body(...)):
    ds_id = payload.get("dataset_id") or ""
    _ds(ds_id)
    meta, data = _data_or_409(ds_id, eval_id)
    k = analysis.grid_k(payload.get("conf", meta.get("conf", 0.25)))
    cls_set = analysis.parse_classes(payload.get("classes"))
    include_approved = bool(payload.get("include_approved"))
    if isinstance(payload.get("table"), dict):
        # exactly the images the per-image table shows (its filter, split, search and confusion cell)
        t = payload["table"]
        k_t, cls_t, side_t = _rows_args(data, k / 100, payload.get("classes"), t.get("show") or "errors", t.get("sort") or "errors", t.get("side") or "a")
        idx, _ = analysis.table(eval_id, data, k_t, cls_t, show=t.get("show") or "errors", sort="name", side=side_t,
                                q=str(t.get("q") or "").strip(), split=str(t.get("split") or ""), cell=str(t.get("cell") or ""))
        sides = list(data["sides"])
        counts = {s: analysis.image_counts(eval_id, data, s, k, cls_set) for s in sides}
        targets = {int(data["img"][i]): analysis.note_line(eval_id, meta, k, {s: {"fn": counts[s][2][i], "fp": counts[s][1][i], "sw": counts[s][3][i]} for s in sides})
                   for i in idx}
    elif payload.get("image_ids"):
        ids = _id_list(payload["image_ids"], "image_ids")
        sides = list(data["sides"])
        targets = {}
        for iid in ids:
            st = analysis.stored_counts(eval_id, data, iid, k, cls_set)
            if st is not None:
                targets[iid] = analysis.note_line(eval_id, meta, k, {s: st[s] for s in sides})
    else:
        crit = payload.get("criteria") or {}
        if not isinstance(crit, dict):
            raise HTTPException(400, "criteria must be an object")
        targets = _flag_targets(eval_id, meta, data, k, cls_set, crit)
    if payload.get("dry_run"):
        rels = {int(data["img"][i]): data["rel"][i] for i in range(data["n"])}
        statuses: Dict[str, int] = {}
        rel_list = [rels[i] for i in targets if i in rels]
        for c in range(0, len(rel_list), 500):
            chunk = rel_list[c:c + 500]
            for r in db.query("SELECT status, COUNT(*) n FROM reviews WHERE dataset_id=? AND rel_path IN (%s) GROUP BY status" % ",".join("?" * len(chunk)), [ds_id, *chunk]):
                statuses[r["status"]] = statuses.get(r["status"], 0) + r["n"]
        skipped = statuses.get("excluded", 0) + (0 if include_approved else statuses.get("approved", 0))
        return {"images": len(targets), "statuses": statuses, "will_update": statuses.get("flagged", 0),
                "will_flag": len(targets) - skipped - statuses.get("flagged", 0)}
    op = uuid.uuid4().hex[:8]
    res = _apply_flags(ds_id, eval_id, meta, targets, include_approved, op)
    return {**res, "op": op}


@router.post("/evals/{eval_id}/unflag")
def unflag_images(eval_id: str, payload: dict = Body(...)):
    """Undo flags this evaluation set: restore each image's previous review state, unless someone changed the
    mark since (then it is left alone and counted as 'changed')."""
    ds_id = payload.get("dataset_id") or ""
    _ds(ds_id)
    _meta_or_404(ds_id, eval_id)
    ledger = store.get_flags(ds_id, eval_id)
    entries = ledger.get("entries", {})
    ids = set(_id_list(payload.get("image_ids") or [], "image_ids"))
    op = payload.get("op")
    plan = []                                     # (rel, state to go back to, drop the whole entry?)
    for rel, e in entries.items():
        stack = e.get("stack") or []
        if not stack:
            continue
        if payload.get("all") or (ids and e.get("image_id") in ids):
            plan.append((rel, stack[0]["prev"], True))
        elif op and stack[-1]["op"] == op:
            plan.append((rel, stack[-1]["prev"], len(stack) == 1))
    res = {"restored": 0, "changed": 0}
    now = db.now()
    tag = f"Compare #{eval_id}:"
    with db.transaction() as conn:
        for rel, prev, drop in plan:
            e = entries[rel]
            if drop:
                entries.pop(rel, None)
            else:
                e["stack"].pop()
            cur = conn.execute("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path=?", (ds_id, rel)).fetchone()
            ours = cur is not None and cur["status"] == "flagged" and any(l.startswith(tag) for l in (cur["note"] or "").split("\n"))
            if not ours:
                res["changed"] += 1                # someone changed the mark since: leave it, forget ours
                entries.pop(rel, None)
                continue
            # keep what a person wrote since; put back this evaluation's previous line, if it had one
            note = analysis.strip_line(cur["note"], eval_id)
            old_line = next((l for l in (prev.get("note") or "").split("\n") if l.startswith(tag)), None)
            if old_line:
                note = analysis.with_line(note, eval_id, old_line)
            # NULL is a real previous state (unreviewed, with a note): reviews.status is nullable
            status = prev.get("status")
            if status is None and not note:
                conn.execute("DELETE FROM reviews WHERE dataset_id=? AND rel_path=?", (ds_id, rel))
            else:
                conn.execute("UPDATE reviews SET status=?, note=?, updated_at=? WHERE dataset_id=? AND rel_path=?",
                             (status, note, now, ds_id, rel))
            res["restored"] += 1
        if op:
            ledger["ops"] = [o for o in ledger.get("ops", []) if o.get("op") != op]
        store.put_flags(conn, ds_id, eval_id, ledger)
    res["remaining"] = len(entries)
    return res


@router.get("/evals/{eval_id}/flags")
def flag_state(eval_id: str, dataset_id: str = Query(...)):
    ledger = store.get_flags(dataset_id, eval_id)
    return {"count": len(ledger.get("entries", {})), "ops": ledger.get("ops", [])[-5:]}


# ------------------------------------------------------------------ export

@router.get("/evals/{eval_id}/export")
def export_csv(eval_id: str, dataset_id: str = Query(...), what: str = Query("images"), conf: float = Query(0.25),
               classes: Optional[str] = Query(None), show: str = Query("all"), sort: str = Query("errors"),
               side: str = Query("a"), q: str = Query(""), split: str = Query(""), cell: str = Query("")):
    ds = _ds(dataset_id)
    meta, data = _data_or_409(dataset_id, eval_id)
    names = _names(ds)
    for c, label in ((meta.get("classes_view") or {}).get("labels") or {}).items():    # a merged class reads as its group
        c = int(c)
        names = names + [""] * max(0, c + 1 - len(names))
        names[c] = label
    k, cls_set, side = _rows_args(data, conf, classes, show, sort, side)
    buf = io.StringIO()
    w = csv.writer(buf)
    sides = list(data["sides"])
    if what == "classes":
        curves = store.get_curves(dataset_id, eval_id) or {}
        w.writerow(["class", "name", "gt"] + [f"{s}_{c}" for s in sides for c in ("ap", "ap50_95", "tp", "fp", "fn", "precision", "recall", "f1", "best_conf", "best_f1")])
        all_cls = sorted({int(c) for s in sides for c in curves[s]["classes"]})
        for c in all_cls:
            if cls_set is not None and c not in cls_set:
                continue
            row = [c, _csv_safe(_cname(names, c)), max(curves[s]["classes"].get(str(c), {}).get("gt", 0) for s in sides)]
            for s in sides:
                one = summarize(curves[s], k, [c], meta["ious"], meta["primary"])
                b = _best(curves[s], [c])
                row += [_f(one["map"]), _f(one["map5095"]), one["tp"], one["fp"], one["fn"], _f(one["precision"]), _f(one["recall"]),
                        _f(one["f1"]), _f(b["conf"]) if b else "", _f(b["f1"]) if b else ""]
            w.writerow(row)
    else:
        idx, counts = analysis.table(eval_id, data, k, cls_set, show=show, sort=sort, side=side, q=q.strip(), split=split, cell=cell)
        items = analysis.row_items(data, idx, counts, cls_set)
        analysis.attach_reviews(dataset_id, items)
        w.writerow(["image_id", "rel_path", "split", "gt"] + [f"{s}_{c}" for s in sides for c in ("tp", "fp", "fn", "swapped")] + ["review"])
        for it in items:
            w.writerow([it["id"], _csv_safe(it["rel_path"]), _csv_safe(it["split"]), it["gt"]] + [it[s][c] for s in sides for c in ("tp", "fp", "fn", "sw")] +
                       [(it.get("review") or {}).get("status") or ""])
    fname = f"compare-{eval_id}-{what}-conf{k:02d}.csv"
    return Response(content="﻿" + buf.getvalue(), media_type="text/csv; charset=utf-8",
                    headers={"Content-Disposition": f'attachment; filename="{fname}"'})


def _f(v) -> str:
    return "" if v is None else f"{v:.4f}"


_FORMULA_START = ("=", "+", "-", "@", "\t", "\r")


def _csv_safe(v: Any) -> Any:
    """A text cell from the dataset (class name from data.yaml, file name, split folder) that a spreadsheet would run
    as a formula gets a leading apostrophe (OWASP CSV injection). Only text fields pass through here, so a negative
    number is never touched."""
    return "'" + v if isinstance(v, str) and v.startswith(_FORMULA_START) else v


PLUGIN = Plugin(
    id=PLUGIN_ID, name="Compare",
    description="Evaluate prediction folders or stored AI runs against ground truth: mAP@.5 and @.5:.95, P/R/F1 at any "
                "confidence, PR / F1 curves with the best threshold, confusion matrix, object sizes, per-image errors, "
                "side-by-side viewer, flag images for review.",
    version="2.0.0", router=router, static_dir=Path(__file__).parent / "static", entry="compare.js",
    nav=[{"id": "compare", "label": "Compare", "icon": "gitCompare", "order": 45}],
)
