"""SQLite storage (stdlib only). One connection per thread, WAL mode."""
from __future__ import annotations

import json
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, Optional

from .config import DB_PATH

_local = threading.local()
_init_lock = threading.Lock()
_initialised = False

SCHEMA = """
CREATE TABLE IF NOT EXISTS datasets (
  id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  root TEXT NOT NULL,
  layout TEXT NOT NULL DEFAULT '{}',
  classes TEXT NOT NULL DEFAULT '[]',
  classes_source TEXT NOT NULL DEFAULT 'inferred',
  description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'new',
  image_count INTEGER NOT NULL DEFAULT 0,
  label_count INTEGER NOT NULL DEFAULT 0,
  box_count INTEGER NOT NULL DEFAULT 0,
  cover_image_id INTEGER,
  created_at REAL, updated_at REAL, scanned_at REAL, opened_at REAL
);
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  split TEXT NOT NULL DEFAULT '',
  abs_path TEXT NOT NULL,
  label_path TEXT,
  width INTEGER, height INTEGER, size INTEGER, mtime REAL, lmtime REAL,
  has_label INTEGER NOT NULL DEFAULT 0,
  n_boxes INTEGER NOT NULL DEFAULT 0,
  classes TEXT NOT NULL DEFAULT '[]',
  issues TEXT NOT NULL DEFAULT '[]',
  seq TEXT NOT NULL DEFAULT '',
  scan_token TEXT NOT NULL DEFAULT '',
  UNIQUE(dataset_id, rel_path)
);
CREATE INDEX IF NOT EXISTS idx_images_ds_split ON images(dataset_id, split);
CREATE INDEX IF NOT EXISTS idx_images_ds_nboxes ON images(dataset_id, n_boxes);
CREATE INDEX IF NOT EXISTS idx_images_ds_label ON images(dataset_id, has_label);
CREATE INDEX IF NOT EXISTS idx_images_ds_seq ON images(dataset_id, seq);
CREATE TABLE IF NOT EXISTS boxes (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL,
  cls INTEGER NOT NULL,
  xc REAL NOT NULL, yc REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_boxes_image ON boxes(image_id);
CREATE INDEX IF NOT EXISTS idx_boxes_ds_cls ON boxes(dataset_id, cls);
CREATE TABLE IF NOT EXISTS reviews (
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  status TEXT NOT NULL,
  note TEXT NOT NULL DEFAULT '',
  updated_at REAL,
  PRIMARY KEY(dataset_id, rel_path)
);
CREATE TABLE IF NOT EXISTS kv (key TEXT PRIMARY KEY, value TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS plugin_data (
  plugin TEXT NOT NULL, dataset_id TEXT NOT NULL DEFAULT '', key TEXT NOT NULL,
  value TEXT NOT NULL, updated_at REAL,
  PRIMARY KEY(plugin, dataset_id, key)
);
"""


def connect() -> sqlite3.Connection:
    conn = getattr(_local, "conn", None)
    if conn is None:
        DB_PATH.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(DB_PATH), timeout=30, isolation_level=None)
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA busy_timeout=30000")
        _local.conn = conn
    return conn


def init_db() -> None:
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = connect()
        conn.executescript(SCHEMA)
        _initialised = True


@contextmanager
def transaction():
    conn = connect()
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield conn
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        raise


def query(sql: str, params: Iterable[Any] = ()) -> list[dict]:
    cur = connect().execute(sql, tuple(params))
    return [dict(r) for r in cur.fetchall()]


def query_one(sql: str, params: Iterable[Any] = ()) -> Optional[dict]:
    cur = connect().execute(sql, tuple(params))
    row = cur.fetchone()
    return dict(row) if row else None


def execute(sql: str, params: Iterable[Any] = ()) -> sqlite3.Cursor:
    return connect().execute(sql, tuple(params))


def executemany(sql: str, rows: Iterable[Iterable[Any]]) -> None:
    connect().executemany(sql, rows)


def now() -> float:
    return time.time()


def loads(s: Optional[str], default=None):
    if not s:
        return default if default is not None else []
    try:
        return json.loads(s)
    except Exception:
        return default if default is not None else []


def dumps(o: Any) -> str:
    return json.dumps(o, ensure_ascii=False, separators=(",", ":"))


def kv_get(key: str, default=None):
    row = query_one("SELECT value FROM kv WHERE key=?", (key,))
    return json.loads(row["value"]) if row else default


def kv_set(key: str, value) -> None:
    execute("INSERT INTO kv(key,value) VALUES(?,?) ON CONFLICT(key) DO UPDATE SET value=excluded.value",
            (key, json.dumps(value)))
