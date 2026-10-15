"""Model runs (``ai_runs``) and predictions / proposals (``ai_preds``).

A proposal is a pending prediction tied to the image it was computed for — never to whatever image happens to be open
in a browser when the answer arrives (finding C1). Nothing here touches label files.
"""
from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .. import db
from .base import Usage
from .secrets import scrub

DETECT_KINDS = ("detect", "prev_frame", "interp", "refine")
AUDIT_KINDS = ("missing", "loose", "swap", "spurious")
ALL_KINDS = DETECT_KINDS + AUDIT_KINDS
_SECRET_PARAM = ("key", "secret", "token", "password", "authorization")

_schema_ok: Dict[str, bool] = {}


def ensure_schema() -> None:
    """Schema ai-core added after the base DDL, added in place on older databases (idempotent, cheap after the
    first call per database file): ``ai_preds.ref_box`` = the ground-truth box ``[xc, yc, w, h]`` (JSON) an audit
    finding refers to — ``ref_row`` alone goes stale as soon as the label file is edited; ``ai_run_usage`` = each
    run's spend per provider that billed it (a two-stage run's classifier, every member of a committee), so spend
    caps and usage totals count a provider's money wherever it was spent; ``idx_ai_preds_run_state`` = per-run
    prediction counts read from the index alone."""
    key = str(getattr(db, "DB_PATH", ""))
    if _schema_ok.get(key):
        return
    cols = {r["name"] for r in db.query("PRAGMA table_info(ai_preds)")}
    if cols and "ref_box" not in cols:
        try:
            db.execute("ALTER TABLE ai_preds ADD COLUMN ref_box TEXT")
        except Exception as e:  # noqa: BLE001 — another connection added it first
            if "duplicate column" not in str(e).lower():
                raise
    if cols:
        db.execute("CREATE TABLE IF NOT EXISTS ai_run_usage (run_id INTEGER NOT NULL REFERENCES ai_runs(id) ON DELETE "
                   "CASCADE, provider TEXT NOT NULL, input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT "
                   "NULL DEFAULT 0, cost_usd REAL NOT NULL DEFAULT 0, created_at REAL, PRIMARY KEY(run_id, provider))")
        db.execute("CREATE INDEX IF NOT EXISTS idx_ai_run_usage_p ON ai_run_usage(provider, created_at)")
        # Runs & cost counts every run's predictions per state: index-only with (run_id, state) — 26 ms instead of
        # 210 ms over 585k audit detections (it read each row through idx_ai_preds_run, then sorted for GROUP BY).
        if {"run_id", "state"} <= cols:
            db.execute("CREATE INDEX IF NOT EXISTS idx_ai_preds_run_state ON ai_preds(run_id, state)")
        _schema_ok[key] = True


def ref_box_json(box: Any) -> Optional[str]:
    """[xc, yc, w, h] → compact JSON for ``ai_preds.ref_box`` (None for anything else)."""
    if box is None:
        return None
    try:
        vals = [round(float(v), 6) for v in list(box)[-4:]]
    except (TypeError, ValueError):
        return None
    return json.dumps(vals, separators=(",", ":")) if len(vals) == 4 else None


def _clean_params(params: Any) -> Any:
    """Run params are shown in the UI and exported: strip anything that could hold a secret, recursively."""
    if isinstance(params, dict):
        return {k: _clean_params(v) for k, v in params.items()
                if not any(s in str(k).lower() for s in _SECRET_PARAM) and k not in ("exemplars",)}
    if isinstance(params, list):
        return [_clean_params(v) for v in params[:200]]
    if isinstance(params, str):
        return scrub(params)[:2000]
    return params


def create_run(dataset_id: str, kind: str, provider: str = "", model: str = "", params: Optional[dict] = None,
               job_id: Optional[str] = None, status: str = "running", created_at: Optional[float] = None) -> int:
    """``created_at``: when the work began, for a run recorded after it finished (a single-image Propose is booked
    once its answer is in — Runs & cost shows finished_at - created_at as its duration)."""
    cur = db.execute(
        "INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, job_id, created_at) VALUES(?,?,?,?,?,?,?,?)",
        (dataset_id, kind, provider, model, db.dumps(_clean_params(params or {})), status, job_id,
         float(created_at) if created_at else db.now()))
    return int(cur.lastrowid)


def set_run_job(run_id: int, job_id: str) -> None:
    db.execute("UPDATE ai_runs SET job_id=? WHERE id=?", (job_id, run_id))


def _fallback_provider(run_provider: str) -> str:
    """Who billed the unattributed part of a run's usage: the run's provider, or for a composite run
    ('yolo+anthropic') its first cloud member."""
    parts = [p for p in str(run_provider or "").split("+") if p]
    if len(parts) <= 1:
        return parts[0] if parts else ""
    from .registry import CLOUD
    return next((p for p in parts if p in CLOUD), parts[0])


def add_usage(run_id: int, usage: Usage, images: int = 0, boxes: int = 0) -> None:
    ensure_schema()
    tin, tout, usd = int(usage.input_tokens), int(usage.output_tokens), float(usage.cost_usd)
    parts = {k: [int(v[0]), int(v[1]), float(v[2])] for k, v in (getattr(usage, "by_provider", None) or {}).items() if k}
    rest = [tin - sum(v[0] for v in parts.values()), tout - sum(v[1] for v in parts.values()),
            usd - sum(v[2] for v in parts.values())]
    with db.transaction() as conn:
        conn.execute("UPDATE ai_runs SET input_tokens=input_tokens+?, output_tokens=output_tokens+?, cost_usd=cost_usd+?, "
                     "images=images+?, boxes=boxes+? WHERE id=?", (tin, tout, usd, images, boxes, run_id))
        if rest[0] > 0 or rest[1] > 0 or rest[2] > 1e-12:
            r = conn.execute("SELECT provider FROM ai_runs WHERE id=?", (run_id,)).fetchone()
            fb = _fallback_provider(r["provider"] if r else "")
            if fb:
                cur = parts.setdefault(fb, [0, 0, 0.0])
                for i in range(3):
                    cur[i] += max(rest[i], 0)
        for pid, (a, b, c) in parts.items():
            if not (a or b or c):
                continue
            conn.execute("INSERT INTO ai_run_usage(run_id, provider, input_tokens, output_tokens, cost_usd, created_at) "
                         "VALUES(?,?,?,?,?,(SELECT created_at FROM ai_runs WHERE id=?)) ON CONFLICT(run_id, provider) DO "
                         "UPDATE SET input_tokens=input_tokens+excluded.input_tokens, output_tokens=output_tokens+"
                         "excluded.output_tokens, cost_usd=cost_usd+excluded.cost_usd",
                         (run_id, pid, int(a), int(b), float(c), run_id))


def spent_since(provider: str, since: float) -> float:
    """Money ``provider`` billed since ``since`` — its share of every run (two-stage classifier, committee member),
    plus runs recorded before the per-provider ledger existed whose provider string names it."""
    ensure_schema()
    a = db.query_one("SELECT COALESCE(SUM(cost_usd),0) s FROM ai_run_usage WHERE provider=? AND created_at >= ?",
                     (provider, since))
    b = db.query_one("SELECT COALESCE(SUM(r.cost_usd),0) s FROM ai_runs r WHERE r.created_at >= ? AND r.cost_usd > 0 "
                     "AND instr('+' || r.provider || '+', ?) > 0 AND NOT EXISTS (SELECT 1 FROM ai_run_usage u "
                     "WHERE u.run_id=r.id)", (since, f"+{provider}+"))
    return float((a or {}).get("s") or 0) + float((b or {}).get("s") or 0)


def usage_parts(run_ids: Sequence[int]) -> Dict[int, List[dict]]:
    """{run id: [{provider, input_tokens, output_tokens, cost_usd}]} from the per-provider ledger."""
    ensure_schema()
    out: Dict[int, List[dict]] = {}
    for chunk in chunked([int(i) for i in run_ids]):
        for r in db.query(f"SELECT * FROM ai_run_usage WHERE run_id IN ({','.join('?' * len(chunk))})", chunk):
            out.setdefault(r["run_id"], []).append(r)
    return out


def finish_run(run_id: int, status: str, result: Optional[dict] = None, error: Optional[str] = None,
               images: Optional[int] = None, boxes: Optional[int] = None) -> None:
    sets = ["status=?", "finished_at=?"]
    params: List[Any] = [status, db.now()]
    if result is not None:
        sets.append("result=?")
        params.append(db.dumps(result))
    if error is not None:
        sets.append("error=?")
        params.append(scrub(error)[:2000])
    if images is not None:
        sets.append("images=?")
        params.append(images)
    if boxes is not None:
        sets.append("boxes=?")
        params.append(boxes)
    db.execute(f"UPDATE ai_runs SET {', '.join(sets)} WHERE id=?", params + [run_id])


def update_params(run_id: int, patch: Dict[str, Any]) -> dict:
    """Merge ``patch`` into a run's params (economy batches record their provider batch ids here)."""
    with db.transaction() as conn:
        r = conn.execute("SELECT params FROM ai_runs WHERE id=?", (run_id,)).fetchone()
        params = db.loads(r["params"], {}) if r else {}
        params = {**(params if isinstance(params, dict) else {}), **_clean_params(patch)}
        conn.execute("UPDATE ai_runs SET params=? WHERE id=?", (db.dumps(params), run_id))
    return params


def update_result(run_id: int, result: Dict[str, Any]) -> None:
    """Live result of a run that is still going (Economy runs show the provider's progress here)."""
    db.execute("UPDATE ai_runs SET result=? WHERE id=?", (db.dumps(result), run_id))


def set_status(run_id: int, status: str) -> None:
    db.execute("UPDATE ai_runs SET status=? WHERE id=?", (status, run_id))


def get_run(run_id: int) -> Optional[dict]:
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    return run_public(r) if r else None


def run_public(r: dict) -> dict:
    out = dict(r)
    out["params"] = db.loads(r.get("params"), {})
    out["result"] = db.loads(r.get("result"), {})
    out["cost_usd"] = round(float(r.get("cost_usd") or 0), 6)
    return out


def pred_public(r: dict) -> dict:
    rb = r.get("ref_box")
    if rb:
        try:
            rb = [float(v) for v in json.loads(rb)][-4:]
        except (TypeError, ValueError):
            rb = None
    return {"id": r["id"], "run_id": r["run_id"], "image_id": r["image_id"], "cls": r["cls"], "xc": r["xc"], "yc": r["yc"],
            "w": r["w"], "h": r["h"], "conf": r["conf"], "label": r["label"], "kind": r["kind"], "reason": r["reason"],
            "ref_row": r["ref_row"], "ref_box": rb or None, "state": r["state"]}


def insert_preds(run_id: int, dataset_id: str, image_id: int, preds: Iterable[dict], state: str = "pending",
                 replace_pending: Sequence[str] = ()) -> List[dict]:
    """Store predictions for one image; optionally drop this image's earlier PENDING preds of the given kinds first
    (a new proposal run replaces the previous suggestions instead of piling up). Returns the stored rows."""
    rows = list(preds)
    now = db.now()
    ensure_schema()
    with db.transaction() as conn:
        if replace_pending:
            q = ",".join("?" * len(replace_pending))
            conn.execute(f"DELETE FROM ai_preds WHERE image_id=? AND state='pending' AND kind IN ({q})",
                         [image_id, *replace_pending])
        ids = []
        for p in rows:
            cur = conn.execute(
                "INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, label, kind, ref_row, ref_box, "
                "reason, state, created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (run_id, dataset_id, image_id, int(p["cls"]), float(p["xc"]), float(p["yc"]), float(p["w"]), float(p["h"]),
                 None if p.get("conf") is None else float(p["conf"]), str(p.get("label") or "")[:200],
                 p.get("kind") or "detect", p.get("ref_row"), ref_box_json(p.get("ref_box")),
                 str(p.get("reason") or "")[:500], p.get("state") or state, now))
            ids.append(cur.lastrowid)
    if not ids:
        return []
    q = ",".join("?" * len(ids))
    return [pred_public(r) for r in db.query(f"SELECT * FROM ai_preds WHERE id IN ({q}) ORDER BY id", ids)]


def image_proposals(dataset_id: str, image_id: int) -> dict:
    rows = db.query("SELECT * FROM ai_preds WHERE dataset_id=? AND image_id=? AND state='pending' ORDER BY "
                    "CASE WHEN conf IS NULL THEN 1 ELSE 0 END, conf DESC, id", (dataset_id, image_id))
    preds = [pred_public(r) for r in rows]
    run_ids = sorted({p["run_id"] for p in preds})
    runs = {}
    if run_ids:
        q = ",".join("?" * len(run_ids))
        for r in db.query(f"SELECT id, provider, model, kind, created_at, params FROM ai_runs WHERE id IN ({q})", run_ids):
            params = db.loads(r["params"], {})
            runs[str(r["id"])] = {"provider": r["provider"], "model": r["model"], "kind": r["kind"],
                                  "created_at": r["created_at"],
                                  "pipeline": (params.get("options") or {}).get("pipeline") if isinstance(params, dict) else None}
    return {"preds": preds, "runs": runs}


def decide(dataset_id: str, accept: Sequence[int], reject: Sequence[int]) -> int:
    n = 0
    now = db.now()
    for ids, state in ((accept, "accepted"), (reject, "rejected")):
        ids = [int(i) for i in ids or []]
        for i in range(0, len(ids), 500):
            chunk = ids[i:i + 500]
            q = ",".join("?" * len(chunk))
            cur = db.execute(f"UPDATE ai_preds SET state=?, decided_at=? WHERE dataset_id=? AND state='pending' "
                             f"AND id IN ({q})", [state, now, dataset_id, *chunk])
            n += cur.rowcount or 0
    return n


def undecide(dataset_id: str, ids: Sequence[int]) -> int:
    """Put accepted/rejected proposals back to pending (the caller only sends ids it decided itself; benchmark
    'info' rows never match)."""
    n = 0
    ids = [int(i) for i in ids or []]
    for i in range(0, len(ids), 500):
        chunk = ids[i:i + 500]
        q = ",".join("?" * len(chunk))
        cur = db.execute(f"UPDATE ai_preds SET state='pending', decided_at=NULL WHERE dataset_id=? "
                         f"AND state IN ('accepted','rejected') AND id IN ({q})", [dataset_id, *chunk])
        n += cur.rowcount or 0
    return n


def chunked(ids: Sequence[int], n: int = 500):
    for i in range(0, len(ids), n):
        yield list(ids[i:i + n])


def pending_for_images(dataset_id: str, image_ids: Optional[Sequence[int]], kinds: Sequence[str],
                       min_conf: Optional[float] = None, classes: Optional[Sequence[int]] = None,
                       below: bool = False) -> List[dict]:
    """Pending preds of ``kinds`` (optionally conf >= min_conf, or < min_conf with ``below``) for the given images
    (None = every image in the dataset)."""
    kq = ",".join("?" * len(kinds))
    base = f"SELECT * FROM ai_preds WHERE dataset_id=? AND state='pending' AND kind IN ({kq})"
    params: List[Any] = [dataset_id, *kinds]
    if min_conf is not None:
        if below:
            base += " AND conf IS NOT NULL AND conf < ?"
        else:
            base += " AND (conf IS NULL OR conf >= ?)"
        params.append(float(min_conf))
    if classes:
        base += f" AND cls IN ({','.join('?' * len(classes))})"
        params += [int(c) for c in classes]
    if image_ids is None:
        return db.query(base + " ORDER BY image_id, id", params)
    out: List[dict] = []
    for chunk in chunked(list(image_ids)):
        out += db.query(base + f" AND image_id IN ({','.join('?' * len(chunk))}) ORDER BY image_id, id", params + chunk)
    return out


def claim_pending(dataset_id: str, pred_ids: Sequence[int]) -> List[dict]:
    """Atomically take the still-PENDING preds among ``pred_ids`` (state → accepted) and return them. A proposal the
    user rejected (or accepted in Annotate) after a bulk accept was queued is not pending any more, so it is never
    claimed — the decision made last in time wins, and the label file and the decision state cannot disagree."""
    ids = [int(i) for i in pred_ids]
    if not ids:
        return []
    out: List[dict] = []
    now = db.now()
    with db.transaction() as conn:
        for chunk in chunked(ids):
            q = ",".join("?" * len(chunk))
            rows = [dict(r) for r in conn.execute(f"SELECT * FROM ai_preds WHERE dataset_id=? AND state='pending' AND id "
                                                  f"IN ({q}) ORDER BY id", [dataset_id, *chunk]).fetchall()]
            if rows:
                got = [r["id"] for r in rows]
                conn.execute(f"UPDATE ai_preds SET state='accepted', decided_at=? WHERE id IN ({','.join('?' * len(got))})",
                             [now, *got])
            out += rows
    return out


def release(pred_ids: Sequence[int]) -> int:
    """Undo :func:`claim_pending` for preds that were not applied after all (stale, extended rows, write failed)."""
    n = 0
    for chunk in chunked([int(i) for i in pred_ids]):
        cur = db.execute(f"UPDATE ai_preds SET state='pending', decided_at=NULL WHERE state='accepted' AND id IN "
                         f"({','.join('?' * len(chunk))})", chunk)
        n += cur.rowcount or 0
    return n


def resolve(pred_ids: Sequence[int], reason: str) -> int:
    """Close claimed preds that no longer apply (an audit finding whose box is gone): state 'rejected' WITHOUT
    ``decided_at`` — nobody rejected them, so :func:`drop_rejected` never suppresses a later audit's finding."""
    n = 0
    for chunk in chunked([int(i) for i in pred_ids]):
        cur = db.execute(f"UPDATE ai_preds SET state='rejected', decided_at=NULL, reason=? WHERE state='accepted' AND id "
                         f"IN ({','.join('?' * len(chunk))})", [str(reason)[:500], *chunk])
        n += cur.rowcount or 0
    return n


_POINT_KINDS = ("detect", "prev_frame", "interp", "refine", "missing")   # "there is an object here" suggestions


def drop_in_regions(image_id: int, preds: Sequence[dict]) -> Tuple[List[dict], int]:
    """Leave out "there is an object here" suggestions whose centre lies in one of the image's regions that are not
    objects (ignored regions, DontCare, crowds of annotations converted from another format — core.annindex)."""
    preds = list(preds)
    objs = [p for p in preds if (p.get("kind") or "detect") in _POINT_KINDS]
    if not objs:
        return preds, 0
    from ..core import annindex
    kept, n = annindex.drop_in_regions(int(image_id), objs)
    if not n:
        return preds, 0
    keep = {id(p) for p in kept}
    return [p for p in preds if (p.get("kind") or "detect") not in _POINT_KINDS or id(p) in keep], n


def drop_rejected(image_id: int, preds: Sequence[dict], iou: float = 0.7) -> Tuple[List[dict], int]:
    """Leave out new suggestions the user already rejected on this image: same class, IoU >= ``iou`` with a pred
    the user decided 'rejected' (``decided_at`` set — rows a job stored as rejected itself do not count). Object
    suggestions (detect / prev_frame / interp / refine / missing) match each other; label-fix findings (loose / swap /
    spurious) match their own kind. Suggestions inside regions that are not objects (drop_in_regions) are left out
    too. Returns (kept, number dropped)."""
    preds = list(preds)
    if not preds:
        return preds, 0
    preds, in_regions = drop_in_regions(image_id, preds)
    from . import geometry
    rej = db.query("SELECT cls, xc, yc, w, h, kind FROM ai_preds WHERE image_id=? AND state='rejected' AND "
                   "decided_at IS NOT NULL", (int(image_id),))
    if not rej:
        return preds, in_regions

    def group(kind: str) -> str:
        return "object" if (kind or "detect") in _POINT_KINDS else kind
    kept, dropped = [], 0
    for p in preds:
        g = group(p.get("kind") or "detect")
        box = (float(p["xc"]), float(p["yc"]), float(p["w"]), float(p["h"]))
        if any(group(r["kind"]) == g and int(r["cls"]) == int(p["cls"])
               and geometry.iou4(box, (r["xc"], r["yc"], r["w"], r["h"])) >= iou for r in rej):
            dropped += 1
            continue
        kept.append(p)
    return kept, dropped + in_regions


def set_state(pred_ids: Sequence[int], state: str) -> int:
    n = 0
    now = db.now()
    for chunk in chunked([int(i) for i in pred_ids]):
        cur = db.execute(f"UPDATE ai_preds SET state=?, decided_at=? WHERE state='pending' AND id IN "
                         f"({','.join('?' * len(chunk))})", [state, now, *chunk])
        n += cur.rowcount or 0
    return n


def upsert_audit(dataset_id: str, image_id: int, run_id: int, score: float, counts: Dict[str, int]) -> None:
    db.execute("INSERT INTO ai_audit(image_id, dataset_id, run_id, score, counts, created_at) VALUES(?,?,?,?,?,?) "
               "ON CONFLICT(image_id) DO UPDATE SET dataset_id=excluded.dataset_id, run_id=excluded.run_id, "
               "score=excluded.score, counts=excluded.counts, created_at=excluded.created_at",
               (image_id, dataset_id, run_id, float(score), json.dumps(counts), db.now()))


def delete_run(run_id: int) -> dict:
    """Drop a run's pending proposals, its raw detections (state 'info': audit / benchmark detections kept for
    Compare) and audit rows. The run row stays (status 'deleted') when it cost money so the usage ledger keeps
    telling the truth; free runs are removed entirely."""
    r = db.query_one("SELECT * FROM ai_runs WHERE id=?", (run_id,))
    if not r:
        return {"deleted": False}
    with db.transaction() as conn:
        n = conn.execute("DELETE FROM ai_preds WHERE run_id=? AND state='pending'", (run_id,)).rowcount or 0
        conn.execute("DELETE FROM ai_preds WHERE run_id=? AND state='info'", (run_id,))
        conn.execute("DELETE FROM ai_audit WHERE run_id=?", (run_id,))
        if float(r["cost_usd"] or 0) > 0:
            conn.execute("UPDATE ai_runs SET status='deleted' WHERE id=?", (run_id,))
        else:
            conn.execute("DELETE FROM ai_runs WHERE id=?", (run_id,))
    return {"deleted": True, "preds_removed": n}


_stats_checked = 0.0


def refresh_planner_stats(force: bool = False, throttle: bool = True) -> bool:
    """Keep SQLite's planner statistics for the AI tables truthful.

    ``ANALYZE`` at startup can record ai_preds when it held a handful of rows; every long-lived connection then plans
    "is there a pending proposal for this image" as a full scan per image — 5-10 s per Explore page after a batch
    filled the table (measured on 7.8k images / 6k proposals). When the table has outgrown its statistics, re-analyse
    and touch the schema, so every connection re-plans its cached statements with the new numbers."""
    global _stats_checked
    import time as _t
    if throttle and not force and _t.time() - _stats_checked < 30:
        return False
    _stats_checked = _t.time()
    try:
        row = db.query_one("SELECT stat FROM sqlite_stat1 WHERE tbl='ai_preds' AND idx='idx_ai_preds_img'")
    except Exception:  # noqa: BLE001 — no sqlite_stat1 yet: default estimates already favour the indexes
        return False
    if not row:
        return False
    try:
        recorded = int(str(row["stat"]).split()[0])
    except (ValueError, IndexError):
        recorded = 0
    n = db.query_one("SELECT COUNT(*) n FROM ai_preds")["n"]
    if not force and n <= max(200, 4 * recorded):
        return False
    try:
        db.execute("ANALYZE ai_preds")
        db.execute("ANALYZE ai_audit")
        db.execute("CREATE TABLE IF NOT EXISTS _fovea_stats_epoch(x INTEGER)")   # schema change: all connections
        db.execute("DROP TABLE IF EXISTS _fovea_stats_epoch")                    # re-plan with the new statistics
    except Exception:  # noqa: BLE001 — statistics are an optimisation; never fail the caller
        return False
    return True
