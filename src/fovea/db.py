"""SQLite storage (stdlib only). One connection per thread, WAL mode."""
from __future__ import annotations

import json
import re
import sqlite3
import threading
import time
from contextlib import contextmanager
from typing import Any, Iterable, List, Optional

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
-- AUTOINCREMENT: an image id is never reused after a dataset is deleted or relocated, so a cached
-- /api/img/<id> can never show another dataset's picture. rel_path / split / seq are NFC-normalised (macOS
-- stores Hangul file names decomposed); abs_path / label_path stay byte-exact for file access.
CREATE TABLE IF NOT EXISTS images (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  split TEXT NOT NULL DEFAULT '',
  abs_path TEXT NOT NULL,
  label_path TEXT,
  width INTEGER, height INTEGER, size INTEGER, mtime REAL, lmtime REAL,
  lsize INTEGER,                            -- label file size; with lmtime lets a rescan skip unchanged labels
  has_label INTEGER NOT NULL DEFAULT 0,
  n_boxes INTEGER NOT NULL DEFAULT 0,
  classes TEXT NOT NULL DEFAULT '[]',
  issues TEXT NOT NULL DEFAULT '[]',
  seq TEXT NOT NULL DEFAULT '',
  scan_token TEXT NOT NULL DEFAULT '',
  UNIQUE(dataset_id, rel_path)
);
-- (dataset_id, split, has_label, n_boxes, issues, classes): covers the facet / count queries, which then never
-- touch the wide rows (absolute paths). INDEX_SHAPES upgrades older databases.
CREATE INDEX IF NOT EXISTS idx_images_ds_split ON images(dataset_id, split, has_label, n_boxes, issues, classes);
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
CREATE INDEX IF NOT EXISTS idx_boxes_ds_cls ON boxes(dataset_id, cls, image_id);
CREATE TABLE IF NOT EXISTS reviews (
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL,
  status TEXT,                              -- NULL: unreviewed, but the row keeps a note
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

-- AI: one row per model run (single-image propose, batch, audit, benchmark, interpolation).
CREATE TABLE IF NOT EXISTS ai_runs (
  id INTEGER PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  kind TEXT NOT NULL,                       -- propose | batch | audit | benchmark | interp | prev_frame
  provider TEXT NOT NULL DEFAULT '', model TEXT NOT NULL DEFAULT '',
  params TEXT NOT NULL DEFAULT '{}',        -- json; never contains secrets
  status TEXT NOT NULL DEFAULT 'running',   -- running | done | error | cancelled
  images INTEGER NOT NULL DEFAULT 0, boxes INTEGER NOT NULL DEFAULT 0,
  input_tokens INTEGER NOT NULL DEFAULT 0, output_tokens INTEGER NOT NULL DEFAULT 0,
  cost_usd REAL NOT NULL DEFAULT 0,
  job_id TEXT, error TEXT, result TEXT NOT NULL DEFAULT '{}',
  created_at REAL, finished_at REAL
);
CREATE INDEX IF NOT EXISTS idx_ai_runs_ds ON ai_runs(dataset_id, created_at);
-- AI: predictions / proposals. Never written to label files until accepted.
-- AUTOINCREMENT: a deleted proposal's id is never handed out again (a re-propose deletes the image's pending preds,
-- the newest ids, right before inserting new ones), so a queued decision, an Annotate undo or a snapshot_preds row
-- naming an old id can never land on a newer proposal. Older databases are rebuilt by _m_ai_preds_autoincrement.
CREATE TABLE IF NOT EXISTS ai_preds (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  run_id INTEGER NOT NULL REFERENCES ai_runs(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  cls INTEGER NOT NULL, xc REAL NOT NULL, yc REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  conf REAL,                                -- NULL when the source has no calibrated confidence
  label TEXT NOT NULL DEFAULT '',           -- raw label from the model (before class mapping)
  kind TEXT NOT NULL DEFAULT 'detect',      -- detect | prev_frame | interp | refine | missing | loose | swap | spurious
  ref_row INTEGER,                          -- audit kinds: the ground-truth row the finding refers to
  reason TEXT NOT NULL DEFAULT '',
  state TEXT NOT NULL DEFAULT 'pending',    -- pending | accepted | rejected | info
  created_at REAL, decided_at REAL,
  ref_box TEXT                              -- audit kinds: that GT box as JSON [xc, yc, w, h] (rows shift on edits);
                                            -- older databases get it from ai.store.ensure_schema (ALTER TABLE)
);
-- (image_id, state, kind, conf): "pending proposals of this image" and the ai_conf sort read the index only.
CREATE INDEX IF NOT EXISTS idx_ai_preds_img ON ai_preds(image_id, state, kind, conf);
CREATE INDEX IF NOT EXISTS idx_ai_preds_ds ON ai_preds(dataset_id, state, kind, image_id);
CREATE INDEX IF NOT EXISTS idx_ai_preds_run ON ai_preds(run_id);
-- AI label audit: one score per image (latest audit run wins).
CREATE TABLE IF NOT EXISTS ai_audit (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL, run_id INTEGER, score REAL NOT NULL DEFAULT 0,
  counts TEXT NOT NULL DEFAULT '{}', created_at REAL
);
CREATE INDEX IF NOT EXISTS idx_ai_audit_ds ON ai_audit(dataset_id, score);
-- Label snapshots: content-addressed, deduplicated label-file versions for undoing bulk / AI writes.
CREATE TABLE IF NOT EXISTS label_blobs (sha TEXT PRIMARY KEY, data BLOB NOT NULL);
CREATE TABLE IF NOT EXISTS snapshots (
  id INTEGER PRIMARY KEY,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  name TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'auto',   -- auto | manual | deleting (being dropped)
  op TEXT NOT NULL DEFAULT '', n_files INTEGER NOT NULL DEFAULT 0, created_at REAL,
  tracked INTEGER NOT NULL DEFAULT 0        -- 1: the op records what it wrote (snapshot_files.after_ver), so an
                                            -- Undo reverts only files still as the op left them
);
CREATE TABLE IF NOT EXISTS snapshot_files (
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
  image_id INTEGER, label_path TEXT NOT NULL,
  sha TEXT,                                 -- NULL: the label file did not exist
  rel_path TEXT,                            -- the image's index key then: finds the file again after a re-index
  sig TEXT,                                 -- 'mtime_ns:ctime_ns:size:inode' then: a diff skips files whose stat matches
  after_ver TEXT,                           -- tracked ops: the labels version (digest) the op left; '-' = it left no
                                            -- file; NULL = the op did not write this file
  PRIMARY KEY(snapshot_id, label_path)
);
-- AI proposal decisions an op made, kept with its snapshot (a bulk accept marks proposals 'accepted' as it writes
-- their boxes): restoring the snapshot for an image puts each of its preds back to `state` while it still holds
-- `after_state`, so undoing an accept makes its proposals pending again instead of losing them.
CREATE TABLE IF NOT EXISTS snapshot_preds (
  snapshot_id INTEGER NOT NULL REFERENCES snapshots(id) ON DELETE CASCADE,
  pred_id INTEGER NOT NULL,                 -- ai_preds.id (no FK: a pred deleted since is simply not found)
  image_id INTEGER NOT NULL,
  state TEXT NOT NULL,                      -- what a restore puts back ('pending' for an accept)
  after_state TEXT NOT NULL,                -- what the op left; a pred decided otherwise since is left alone
  reason TEXT,                              -- ai_preds.reason before the op (NULL: keep the current one)
  PRIMARY KEY(snapshot_id, pred_id)
);
-- Dataset QA: perceptual hash / image statistics and flags that do not come from the label parser.
CREATE TABLE IF NOT EXISTS image_hash (
  image_id INTEGER PRIMARY KEY REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL, mtime REAL,
  dhash INTEGER, b0 INTEGER, b1 INTEGER, b2 INTEGER, b3 INTEGER,
  luma REAL, contrast REAL, sharpness REAL
);
CREATE INDEX IF NOT EXISTS idx_image_hash_ds ON image_hash(dataset_id);
CREATE INDEX IF NOT EXISTS idx_image_hash_b0 ON image_hash(dataset_id, b0);
CREATE INDEX IF NOT EXISTS idx_image_hash_b1 ON image_hash(dataset_id, b1);
CREATE INDEX IF NOT EXISTS idx_image_hash_b2 ON image_hash(dataset_id, b2);
CREATE INDEX IF NOT EXISTS idx_image_hash_b3 ON image_hash(dataset_id, b3);
CREATE TABLE IF NOT EXISTS image_flags (
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL, code TEXT NOT NULL,
  score REAL NOT NULL DEFAULT 0, detail TEXT NOT NULL DEFAULT '{}',
  PRIMARY KEY(image_id, code)
);
CREATE INDEX IF NOT EXISTS idx_image_flags_ds ON image_flags(dataset_id, code, image_id);
-- Class cards: per-class labeling guideline shared by people and AI prompts.
CREATE TABLE IF NOT EXISTS class_cards (
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  cls INTEGER NOT NULL,
  description TEXT NOT NULL DEFAULT '', include TEXT NOT NULL DEFAULT '', exclude TEXT NOT NULL DEFAULT '',
  prompt TEXT NOT NULL DEFAULT '',          -- what an AI should look for (defaults to the class name)
  examples TEXT NOT NULL DEFAULT '[]',      -- json [{image_id, row, polarity: 1|-1}]
  updated_at REAL,
  PRIMARY KEY(dataset_id, cls)
);
-- Annotations converted from another format (core.annformats / core.annindex): each box's attributes while it keeps
-- the geometry it was converted with (occlusion, truncation …), and the image's regions that are not objects
-- (ignored regions, DontCare, crowds). Both are re-read from the conversion's sidecar after every scan and save.
CREATE TABLE IF NOT EXISTS box_attrs (
  box_id INTEGER PRIMARY KEY REFERENCES boxes(id) ON DELETE CASCADE,
  image_id INTEGER NOT NULL,
  dataset_id TEXT NOT NULL,
  attrs TEXT NOT NULL DEFAULT '{}'
);
CREATE INDEX IF NOT EXISTS idx_box_attrs_ds ON box_attrs(dataset_id, image_id);
CREATE TABLE IF NOT EXISTS image_regions (
  id INTEGER PRIMARY KEY,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL,
  kind TEXT NOT NULL,
  xc REAL NOT NULL, yc REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_regions_image ON image_regions(image_id);
CREATE INDEX IF NOT EXISTS idx_regions_ds ON image_regions(dataset_id, image_id);
-- Working sets: an ordered image-id list frozen server-side (Explore "all matching", large selections), used by
-- every image filter as `set=<token>`. Kept 24 h after last use (core.imagesq).
CREATE TABLE IF NOT EXISTS work_sets (
  id INTEGER PRIMARY KEY,
  token TEXT NOT NULL UNIQUE,
  dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  n INTEGER NOT NULL DEFAULT 0,
  params TEXT NOT NULL DEFAULT '{}',        -- json {filters, sort, order, from}: how it was frozen (informational)
  created_at REAL, used_at REAL
);
CREATE INDEX IF NOT EXISTS idx_work_sets_ds ON work_sets(dataset_id, used_at);
CREATE TABLE IF NOT EXISTS work_set_items (
  set_id INTEGER NOT NULL REFERENCES work_sets(id) ON DELETE CASCADE,
  pos INTEGER NOT NULL,                     -- 0-based order in the set
  image_id INTEGER NOT NULL,                -- no FK: a deleted image simply drops out (the filter joins images)
  PRIMARY KEY(set_id, pos)
) WITHOUT ROWID;
CREATE INDEX IF NOT EXISTS idx_work_set_items_img ON work_set_items(set_id, image_id);
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


# Indexes on columns that older databases only gain through a migration, so they run after MIGRATIONS.
# The expression indexes match imagesq.SORT_KEYS exactly: ORDER BY + LIMIT then walks an index instead of
# sorting the whole dataset for every page.
POST_SCHEMA = """
CREATE INDEX IF NOT EXISTS idx_images_ds_id ON images(dataset_id, id);
CREATE INDEX IF NOT EXISTS idx_images_ds_abs ON images(dataset_id, abs_path);
CREATE INDEX IF NOT EXISTS idx_images_ds_mtime ON images(dataset_id, COALESCE(mtime,0));
CREATE INDEX IF NOT EXISTS idx_images_ds_area ON images(dataset_id, (COALESCE(width,0)*COALESCE(height,0)));
CREATE INDEX IF NOT EXISTS idx_images_ds_rnd ON images(dataset_id, ((id * 2654435761) % 4294967296));
CREATE INDEX IF NOT EXISTS idx_snapshots_ds ON snapshots(dataset_id, created_at);
CREATE INDEX IF NOT EXISTS idx_snapshot_files_sha ON snapshot_files(sha);
-- objects hook: Objects sort=area (api/objects.py SORTS) walks this index instead of sorting every box per page
CREATE INDEX IF NOT EXISTS idx_boxes_ds_area ON boxes(dataset_id, (w * h));
"""


# Indexes whose column list grew after databases already had them under the same name: rebuilt once at start-up.
# The trailing image_id makes "images of this dataset with a flag / pending AI row / box of class c" an index-only
# read (imagesq filters them with an uncorrelated IN (...)).
INDEX_SHAPES = {
    "idx_images_ds_split": ("images", ("dataset_id", "split", "has_label", "n_boxes", "issues", "classes")),
    "idx_boxes_ds_cls": ("boxes", ("dataset_id", "cls", "image_id")),
    "idx_ai_preds_ds": ("ai_preds", ("dataset_id", "state", "kind", "image_id")),
    "idx_ai_preds_img": ("ai_preds", ("image_id", "state", "kind", "conf")),
    "idx_image_flags_ds": ("image_flags", ("dataset_id", "code", "image_id")),
}


def _ensure_index_shapes(conn: sqlite3.Connection) -> List[str]:
    rebuilt = []
    for name, (table, cols) in INDEX_SHAPES.items():
        have = tuple(r[2] for r in conn.execute(f"PRAGMA index_info({name})").fetchall())
        if have == cols:
            continue
        conn.execute(f"DROP INDEX IF EXISTS {name}")
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {table}({', '.join(cols)})")   # two processes may race here
        rebuilt.append(name)
    if rebuilt and conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'").fetchone():
        for name in rebuilt:
            conn.execute("DELETE FROM sqlite_stat1 WHERE idx=?", (name,))
            conn.execute(f"ANALYZE {name}")
    return rebuilt


def _columns(conn: sqlite3.Connection, table: str) -> dict:
    return {r[1]: r for r in conn.execute(f"PRAGMA table_info({table})").fetchall()}


def _table_sql(conn: sqlite3.Connection, table: str) -> str:
    row = conn.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name=?", (table,)).fetchone()
    return (row[0] if row else "") or ""


def _schema_block(table: str) -> str:
    """The CREATE TABLE statement for `table` from SCHEMA (the current shape). It ends at the first ';' that
    completes it — a column comment may itself contain ');'."""
    start = end = SCHEMA.index(f"CREATE TABLE IF NOT EXISTS {table} (")
    while True:
        end = SCHEMA.index(";", end) + 1
        if sqlite3.complete_statement(SCHEMA[start:end]):
            return SCHEMA[start:end]


def _rebuild(conn: sqlite3.Connection, table: str) -> None:
    """SQLite's 12-step table rebuild: new shape from SCHEMA, copy the common columns, swap. Runs with
    foreign keys off (set by the caller, outside the transaction) so dependants are neither cascaded nor
    orphaned; the ids are copied verbatim. The drop takes the table's indexes with it: each is re-created as
    it was, including those other modules add outside SCHEMA (ai.store's idx_ai_preds_run_state)."""
    old_cols = list(_columns(conn, table))
    indexes = [r[0] for r in conn.execute("SELECT sql FROM sqlite_master WHERE type='index' AND tbl_name=? "
                                          "AND sql IS NOT NULL", (table,)).fetchall()]
    ddl = _schema_block(table).replace(f"CREATE TABLE IF NOT EXISTS {table} (", f"CREATE TABLE {table}__new (")
    conn.execute(f"DROP TABLE IF EXISTS {table}__new")
    conn.execute(ddl)
    new_cols = list(_columns(conn, f"{table}__new"))
    common = ",".join(c for c in old_cols if c in new_cols)
    conn.execute(f"INSERT INTO {table}__new({common}) SELECT {common} FROM {table}")
    conn.execute(f"DROP TABLE {table}")
    conn.execute(f"ALTER TABLE {table}__new RENAME TO {table}")
    for stmt in indexes:
        conn.execute(stmt)


def _m_images_lsize(conn: sqlite3.Connection) -> None:
    if "lsize" not in _columns(conn, "images"):
        conn.execute("ALTER TABLE images ADD COLUMN lsize INTEGER")


def _m_reviews_nullable_status(conn: sqlite3.Connection) -> None:
    col = _columns(conn, "reviews").get("status")
    if col is not None and col[3]:          # notnull flag: clearing a review used to delete its note
        _rebuild(conn, "reviews")


def _m_images_autoincrement(conn: sqlite3.Connection) -> None:
    if "AUTOINCREMENT" not in _table_sql(conn, "images").upper():
        _rebuild(conn, "images")
        for stmt in re.findall(r"CREATE INDEX IF NOT EXISTS \w+ ON images\([^)]*\)", SCHEMA):
            conn.execute(stmt)              # the drop took the old indexes with it


def _m_nfc_paths(conn: sqlite3.Connection) -> None:
    """Hangul (and other) names indexed on macOS were stored decomposed (NFD): an NFC search, split filter
    or Files lookup found nothing. Normalise the stored keys; a row whose NFC key already exists is kept as
    is rather than merged."""
    import unicodedata

    def nfc(v):
        return unicodedata.normalize("NFC", v) if isinstance(v, str) else v

    rows = conn.execute("SELECT id, dataset_id, rel_path, split, seq FROM images").fetchall()
    for r in rows:
        rel, split, seq = nfc(r[2]), nfc(r[3]), nfc(r[4])
        if (rel, split, seq) == (r[2], r[3], r[4]):
            continue
        try:
            conn.execute("UPDATE images SET rel_path=?, split=?, seq=? WHERE id=?", (rel, split, seq, r[0]))
        except sqlite3.IntegrityError:
            continue
    for r in conn.execute("SELECT dataset_id, rel_path FROM reviews").fetchall():
        rel = nfc(r[1])
        if rel != r[1]:
            try:
                conn.execute("UPDATE reviews SET rel_path=? WHERE dataset_id=? AND rel_path=?", (rel, r[0], r[1]))
            except sqlite3.IntegrityError:
                continue


def _m_snapshot_files_rel_path(conn: sqlite3.Connection) -> None:
    if "rel_path" not in _columns(conn, "snapshot_files"):
        conn.execute("ALTER TABLE snapshot_files ADD COLUMN rel_path TEXT")


def _m_snapshot_files_sig(conn: sqlite3.Connection) -> None:
    if "sig" not in _columns(conn, "snapshot_files"):
        conn.execute("ALTER TABLE snapshot_files ADD COLUMN sig TEXT")


def _m_snapshot_undo_tracking(conn: sqlite3.Connection) -> None:
    if "tracked" not in _columns(conn, "snapshots"):
        conn.execute("ALTER TABLE snapshots ADD COLUMN tracked INTEGER NOT NULL DEFAULT 0")
    if "after_ver" not in _columns(conn, "snapshot_files"):
        conn.execute("ALTER TABLE snapshot_files ADD COLUMN after_ver TEXT")


def _m_ai_preds_autoincrement(conn: sqlite3.Connection) -> None:
    """SQLite gave a new pred the highest live id + 1, so a re-propose reused the ids of the pending preds it had
    just deleted. Ids deleted before this step are known only where snapshot_preds still names them: the
    sequence starts past those too."""
    if "AUTOINCREMENT" in _table_sql(conn, "ai_preds").upper():
        return
    _rebuild(conn, "ai_preds")
    top = conn.execute("SELECT COALESCE(MAX(id), 0) FROM ai_preds").fetchone()[0]
    if conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='snapshot_preds'").fetchone():
        top = max(top, conn.execute("SELECT COALESCE(MAX(pred_id), 0) FROM snapshot_preds").fetchone()[0])
    if top and not conn.execute("UPDATE sqlite_sequence SET seq=MAX(seq, ?) WHERE name='ai_preds'", (top,)).rowcount:
        conn.execute("INSERT INTO sqlite_sequence(name, seq) VALUES('ai_preds', ?)", (top,))


# (name, step, needs foreign keys off). Each step is idempotent: a fresh database already has the current
# shape from SCHEMA, so every step is a no-op there. PRAGMA user_version records how far a database got.
MIGRATIONS = [
    ("images.lsize", _m_images_lsize, False),
    ("reviews.status nullable", _m_reviews_nullable_status, True),
    ("images AUTOINCREMENT", _m_images_autoincrement, True),
    ("NFC paths", _m_nfc_paths, False),
    ("snapshot_files.rel_path", _m_snapshot_files_rel_path, False),
    ("snapshot_files.sig", _m_snapshot_files_sig, False),
    ("snapshot undo tracking", _m_snapshot_undo_tracking, False),
    ("ai_preds AUTOINCREMENT", _m_ai_preds_autoincrement, True),
]
SCHEMA_VERSION = len(MIGRATIONS)


def migrate(conn: sqlite3.Connection) -> List[str]:
    """Bring an existing database to SCHEMA_VERSION. Each step runs in its own write transaction that
    re-reads user_version first, so two processes starting at once cannot both apply a step."""
    applied: List[str] = []
    version = conn.execute("PRAGMA user_version").fetchone()[0]
    for n, (name, step, fk_off) in enumerate(MIGRATIONS, 1):
        if version >= n:
            continue
        if fk_off:
            conn.execute("PRAGMA foreign_keys=OFF")     # only takes effect outside a transaction
        try:
            conn.execute("BEGIN IMMEDIATE")
            try:
                if conn.execute("PRAGMA user_version").fetchone()[0] < n:
                    # Old databases may already hold orphan rows; only a step that ADDS violations is wrong.
                    before = len(conn.execute("PRAGMA foreign_key_check").fetchall()) if fk_off else 0
                    step(conn)
                    if fk_off:
                        after = len(conn.execute("PRAGMA foreign_key_check").fetchall())
                        if after > before:
                            raise sqlite3.IntegrityError(f"migration '{name}' broke {after - before} foreign key(s)")
                    conn.execute(f"PRAGMA user_version={n}")
                    applied.append(name)
                conn.execute("COMMIT")
            except Exception:
                conn.execute("ROLLBACK")
                raise
        finally:
            if fk_off:
                conn.execute("PRAGMA foreign_keys=ON")
        version = n
    return applied


def init_db() -> None:
    global _initialised
    with _init_lock:
        if _initialised:
            return
        conn = connect()
        conn.executescript(SCHEMA)
        applied = migrate(conn)
        conn.executescript(POST_SCHEMA)
        _ensure_index_shapes(conn)
        # A first index interrupted by a previous process (crash, Ctrl+C) is partial, not complete: say so,
        # so the UI can offer to resume it. Rescans of a complete index keep 'ready' (startup code in main
        # resets any other 'scanning' row).
        conn.execute("""UPDATE datasets SET
                          image_count=(SELECT COUNT(*) FROM images i WHERE i.dataset_id=datasets.id),
                          label_count=(SELECT COALESCE(SUM(has_label),0) FROM images i WHERE i.dataset_id=datasets.id),
                          box_count=(SELECT COALESCE(SUM(n_boxes),0) FROM images i WHERE i.dataset_id=datasets.id)
                        WHERE status='scanning'""")
        conn.execute("UPDATE datasets SET status='partial' WHERE status='scanning' AND scanned_at IS NULL AND image_count > 0")
        if applied:
            conn.execute("ANALYZE")
        else:
            conn.execute("PRAGMA optimize")
        _initialised = True
    try:
        refresh_planner_stats()
    except sqlite3.Error:
        pass                                  # statistics are an optimisation; never fail start-up over them


# ------------------------------------------------------------------ planner statistics
# Tables whose size swings with jobs (a scan, an AI batch or audit, a QA pass, a working set). SQLite plans from
# sqlite_stat1; numbers recorded while such a table held a handful of rows make it plan "is there a pending
# proposal / a flag for this image" as a full scan per image (seconds per Explore page once a batch filled the
# table). Every job that inserts or deletes many rows calls refresh_planner_stats() at its end; read paths call
# maybe_refresh_planner_stats(), which re-checks in the background at most once a minute.
VOLATILE_TABLES = ("images", "boxes", "reviews", "ai_runs", "ai_preds", "ai_audit", "image_flags", "image_hash",
                   "work_set_items", "box_attrs", "image_regions")
SMALL_TABLE_ROWS = 1000          # below this, no statistics at all: SQLite's defaults favour the indexes
STATS_DRIFT = 2.0                # re-analyse when the row count grew or shrank by this factor (plus slack)
SELECTIVITY_PROBE = 256          # distinct leading-index values counted per check (index seeks, not a scan)
SELECTIVITY_SLACK = 16           # rows per value: below this a recorded selectivity is never "drifted"
_stats_lock = threading.Lock()
_stats_state = {"at": 0.0, "running": False, "last": []}


def _recorded_rows(conn: sqlite3.Connection, table: str) -> Optional[int]:
    best: Optional[int] = None
    for (stat,) in conn.execute("SELECT stat FROM sqlite_stat1 WHERE tbl=?", (table,)).fetchall():
        try:
            n = int(str(stat).split()[0])
        except (ValueError, IndexError):
            continue
        best = n if best is None else max(best, n)
    return best


def _q(name: str) -> str:
    return '"' + str(name).replace('"', '""') + '"'


def _distinct_leading(conn: sqlite3.Connection, table: str, index: str, col: str, cap: int) -> int:
    """How many distinct values the index's leading column holds, counted by seeking from each value to the next
    (O(values · log rows), never a scan), stopping at `cap`."""
    t, i, c = _q(table), _q(index), _q(col)
    nxt = f"SELECT MIN({c}) FROM {t} INDEXED BY {i}"
    return int(conn.execute(f"""WITH RECURSIVE v(x, k) AS (SELECT ({nxt}), 1
                                  UNION ALL SELECT ({nxt} WHERE {c} > v.x), k + 1 FROM v WHERE v.x IS NOT NULL AND k < ?)
                                SELECT COUNT(x) FROM v""", (cap,)).fetchone()[0])


def _selectivity_drifted(conn: sqlite3.Connection, table: str, n: int) -> bool:
    """Whether some index's recorded rows-per-value of its leading column (the 2nd number of its sqlite_stat1 row)
    is far from today's. The row count alone misses it: ai_preds analysed while it held ONE run of 50k rows records
    'rows per run_id = 50000'; forty more small runs later the count has not doubled, but the planner still believes
    a run_id lookup reads 50k rows and answers Compare's "runs with predictions" by scanning the table per run."""
    seen: dict = {}
    notnull = {r[1]: bool(r[3]) for r in conn.execute(f"PRAGMA table_info({_q(table)})").fetchall()}
    for idx, stat in conn.execute("SELECT idx, stat FROM sqlite_stat1 WHERE tbl=? AND idx IS NOT NULL", (table,)).fetchall():
        try:
            recorded = int(str(stat).split()[1])
        except (ValueError, IndexError):
            continue
        try:
            cols = sorted(tuple(r) for r in conn.execute(f"PRAGMA index_info({_q(idx)})").fetchall())   # (seqno, cid, name)
            col = cols[0][2] if cols else None
            if not col or not notnull.get(col):
                continue                     # a dropped index, an expression, or NULLs (which MIN() would not count)
            if col not in seen:
                seen[col] = _distinct_leading(conn, table, idx, col, SELECTIVITY_PROBE)
        except sqlite3.Error:
            continue
        values = seen[col]
        if not values:
            continue
        now = n / values                     # rows per value today (an upper bound when the count was capped)
        if recorded > STATS_DRIFT * now + SELECTIVITY_SLACK:
            return True
        if values < SELECTIVITY_PROBE and now > STATS_DRIFT * recorded + SELECTIVITY_SLACK:
            return True
    return False


def refresh_planner_stats(tables: Optional[Iterable[str]] = None, force: bool = False) -> List[str]:
    """Keep the planner statistics of the volatile tables truthful; returns the tables whose statistics changed.

    A table that drifted from its recorded row count by STATS_DRIFT, or whose indexes' recorded selectivity no
    longer matches (_selectivity_drifted: many small AI runs after one big one), is re-analysed; a near-empty table
    loses its statistics (defaults then favour the indexes, which is right for every query Fovea runs). Then a
    schema touch makes every connection — each request thread keeps one — reload the statistics and re-plan cached
    statements. Cheap when nothing drifted (one COUNT per table plus a few hundred index seeks). Call it from a
    job thread, not a request thread."""
    conn = connect()
    if not conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name='sqlite_stat1'").fetchone():
        return []                    # never analysed: the default estimates already favour the indexes
    names = [t for t in (tables or VOLATILE_TABLES) if t in VOLATILE_TABLES]
    changed: List[str] = []
    for t in names:
        recorded = _recorded_rows(conn, t)
        n = conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
        if n < SMALL_TABLE_ROWS:
            if recorded is not None:
                conn.execute("DELETE FROM sqlite_stat1 WHERE tbl=?", (t,))
                changed.append(t)
            continue
        if (force or recorded is None or n > STATS_DRIFT * recorded + 500
                or n * STATS_DRIFT + 500 < recorded or _selectivity_drifted(conn, t, n)):
            conn.execute(f"ANALYZE {t}")
            changed.append(t)
    if changed:
        # A schema change is the only signal every other connection hears: they reload sqlite_stat1 with the
        # schema and re-prepare their statements with the new numbers.
        conn.execute("CREATE TABLE IF NOT EXISTS _fovea_stats_epoch(x INTEGER)")
        conn.execute("DROP TABLE IF EXISTS _fovea_stats_epoch")
    with _stats_lock:
        _stats_state["last"] = changed
    return changed


def maybe_refresh_planner_stats(interval: float = 60.0) -> bool:
    """Read-path safety net for jobs that forgot refresh_planner_stats(): at most once per `interval`, check the
    statistics on a background thread (never on the caller's). True when a check was started."""
    now = time.monotonic()
    with _stats_lock:
        if _stats_state["running"] or now - _stats_state["at"] < interval:
            return False
        _stats_state["at"], _stats_state["running"] = now, True

    def run() -> None:
        try:
            refresh_planner_stats()
        except sqlite3.Error:
            pass
        finally:
            with _stats_lock:
                _stats_state["running"] = False
            conn = getattr(_local, "conn", None)
            if conn is not None:
                _local.conn = None
                conn.close()

    threading.Thread(target=run, name="fovea-planner-stats", daemon=True).start()
    return True


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
