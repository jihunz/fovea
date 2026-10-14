"""Persistence of evaluations in ``plugin_data``.

One evaluation is three rows so that listing never parses per-image data:
``<id>`` small metadata + headline numbers · ``<id>/curves`` per-class curves · ``<id>/data`` packed per-image arrays.
``<id>/flags`` remembers which review marks this evaluation set (for undo). Evaluations saved by Compare 1.x kept
everything in one row with a ``rows`` list; they are split into ``<id>`` + ``<id>/legacy`` the first time they are
listed, so the list stays fast.
"""
from __future__ import annotations

import json
import threading
from collections import OrderedDict
from typing import Any, Dict, List, Optional

from ... import db
from .evaluate import pack, unpack

PLUGIN_ID = "compare"
_lock = threading.Lock()
_cache: "OrderedDict[tuple, Any]" = OrderedDict()
_CACHE_MAX = 6


def _get(ds_id: str, key: str) -> Optional[dict]:
    return db.query_one("SELECT value, updated_at FROM plugin_data WHERE plugin=? AND dataset_id=? AND key=?", (PLUGIN_ID, ds_id, key))


def _put(conn, ds_id: str, key: str, value: str) -> None:
    conn.execute("""INSERT INTO plugin_data(plugin, dataset_id, key, value, updated_at) VALUES(?,?,?,?,?)
                    ON CONFLICT(plugin, dataset_id, key) DO UPDATE SET value=excluded.value, updated_at=excluded.updated_at""",
                 (PLUGIN_ID, ds_id, key, value, db.now()))


def _cached(key: tuple, load):
    with _lock:
        if key in _cache:
            _cache.move_to_end(key)
            return _cache[key]
    val = load()
    with _lock:
        _cache[key] = val
        _cache.move_to_end(key)
        while len(_cache) > _CACHE_MAX:
            _cache.popitem(last=False)
    return val


def forget(ds_id: str, eval_id: str) -> None:
    with _lock:
        for k in [k for k in _cache if k[0] == ds_id and k[1] == eval_id]:
            _cache.pop(k, None)


def save(ds_id: str, meta: dict, curves: dict, data: dict) -> None:
    eid = meta["id"]
    with db.transaction() as conn:
        _put(conn, ds_id, f"{eid}/data", pack(data))
        _put(conn, ds_id, f"{eid}/curves", db.dumps(curves))
        _put(conn, ds_id, eid, db.dumps(meta))
    forget(ds_id, eid)


def _migrate_legacy(ds_id: str, key: str, rec: dict) -> dict:
    """Split a Compare 1.x record (rows inline) into metadata + a separate legacy row blob."""
    meta = {k: v for k, v in rec.items() if k != "rows"}
    meta.update({"v": 1, "legacy": True, "rows_total": len(rec.get("rows") or [])})
    with db.transaction() as conn:
        _put(conn, ds_id, f"{key}/legacy", db.dumps({"rows": rec.get("rows") or []}))
        _put(conn, ds_id, key, db.dumps(meta))
    return meta


def list_meta(ds_id: str) -> List[dict]:
    out = []
    rows = db.query("SELECT key, value, updated_at FROM plugin_data WHERE plugin=? AND dataset_id=? AND instr(key, '/') = 0 ORDER BY updated_at DESC",
                    (PLUGIN_ID, ds_id))
    for r in rows:
        try:
            rec = json.loads(r["value"])
        except ValueError:
            continue
        if "rows" in rec:
            rec = _migrate_legacy(ds_id, r["key"], rec)
        rec.setdefault("id", r["key"])
        out.append(rec)
    out.sort(key=lambda m: -(m.get("created_at") or 0))
    return out


def get_meta(ds_id: str, eid: str) -> Optional[dict]:
    row = _get(ds_id, eid)
    if not row:
        return None
    rec = json.loads(row["value"])
    if "rows" in rec:
        rec = _migrate_legacy(ds_id, eid, rec)
    rec.setdefault("id", eid)
    return rec


def get_curves(ds_id: str, eid: str) -> Optional[dict]:
    def load():
        row = _get(ds_id, f"{eid}/curves")
        return json.loads(row["value"]) if row else None
    return _cached((ds_id, eid, "curves"), load)


def get_data(ds_id: str, eid: str) -> Optional[dict]:
    def load():
        row = _get(ds_id, f"{eid}/data")
        if not row:
            return None
        data = unpack(row["value"])
        data["index"] = {int(i): k for k, i in enumerate(data["img"])}
        return data
    return _cached((ds_id, eid, "data"), load)


def get_legacy_rows(ds_id: str, eid: str) -> List[dict]:
    def load():
        row = _get(ds_id, f"{eid}/legacy")
        return json.loads(row["value"]).get("rows", []) if row else []
    return _cached((ds_id, eid, "legacy"), load)


def update_meta(ds_id: str, eid: str, patch: Dict[str, Any]) -> Optional[dict]:
    meta = get_meta(ds_id, eid)
    if meta is None:
        return None
    meta.update(patch)
    with db.transaction() as conn:
        _put(conn, ds_id, eid, db.dumps(meta))
    return meta


def delete(ds_id: str, eid: str) -> None:
    db.execute("DELETE FROM plugin_data WHERE plugin=? AND dataset_id=? AND (key=? OR key LIKE ?)", (PLUGIN_ID, ds_id, eid, f"{eid}/%"))
    forget(ds_id, eid)


def get_flags(ds_id: str, eid: str) -> dict:
    row = _get(ds_id, f"{eid}/flags")
    return json.loads(row["value"]) if row else {"entries": {}, "ops": []}


def put_flags(conn, ds_id: str, eid: str, ledger: dict) -> None:
    _put(conn, ds_id, f"{eid}/flags", db.dumps(ledger))
