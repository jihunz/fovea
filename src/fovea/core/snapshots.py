"""Label snapshots: the bytes of label files before a bulk / AI write, so every such write can be undone.

Blobs are content-addressed (sha256 of the raw bytes, stored gzipped) and shared between snapshots, so
snapshotting the same unchanged files again costs a row per file and nothing more. A file that did not
exist is recorded with sha NULL: restoring it removes the file again.

An operation that records what it wrote (WriteLog) makes its snapshot *tracked*: each file it wrote keeps the
version it left there, and the snapshot shrinks to those files when the op ends. Restore then has two modes:
'full' (History: put the version back) and 'undo' (an op's own Undo button: revert a file only while it still
holds exactly what the op wrote — an edit made since is kept and reported, never overwritten or deleted).

Retention and deletion run off the request path: snapshot_before() only queues a prune, which a background
thread runs in short transactions (a 46k-file snapshot is dropped in chunks, and only its own blobs are
garbage-collected), so an Objects key press never waits for the eviction of a whole-dataset version.
"""
from __future__ import annotations

import gzip
import hashlib
import json
import logging
import math
import os
import stat
import tempfile
import threading
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from .. import db
from .labels import (LABEL_LOCK, MAX_CLASS_ID, dataset_bases, notify_label_write, parse_label_text,
                     write_target)
from .labels import _digest as _label_version

log = logging.getLogger("fovea.snapshots")

# Retention of automatic snapshots per dataset (manual ones are kept until deleted), in two rings so a run of
# small edits (one Objects key press = one snapshot of one file) can never push out the only undo point of a
# bulk edit or an AI write over thousands of files.
KEEP_AUTO = 20                 # newest automatic snapshots of more than SMALL_FILES files
KEEP_SMALL = 50                # newest automatic snapshots of at most SMALL_FILES files
SMALL_FILES = 20
DROP_CHUNK = 2000              # snapshot_files rows deleted per transaction (~0.1 s) when a snapshot is dropped
GONE = "-"                     # snapshot_files.after_ver: the op left no file there
_gc_lock = threading.Lock()

# Snapshots in use right now (a restore reads its blobs file by file; a diff reads them too). prune() never drops
# one and delete_snapshot() refuses one, so a version cannot vanish half-way through being restored.
_pins: Counter = Counter()
_pin_lock = threading.Lock()


class SnapshotBusy(RuntimeError):
    """The snapshot is being restored or compared right now."""


@contextmanager
def pinned(*sids: Optional[int]) -> Iterator[None]:
    ids = [int(s) for s in sids if s]
    with _pin_lock:
        for s in ids:
            _pins[s] += 1
    try:
        yield
    finally:
        with _pin_lock:
            for s in ids:
                _pins[s] -= 1
                if _pins[s] <= 0:
                    del _pins[s]


def is_pinned(sid: int) -> bool:
    with _pin_lock:
        return _pins.get(int(sid), 0) > 0


def _read(path: str) -> Optional[bytes]:
    return _read_sig(path)[0]


def _sig_of(st: os.stat_result) -> str:
    return f"{st.st_mtime_ns}:{st.st_ctime_ns}:{st.st_size}:{st.st_ino}"


def _read_sig(path: str) -> Tuple[Optional[bytes], Optional[str]]:
    """(bytes, stat signature) of a label file from one open; (None, None) when there is no such file. A changed
    file always gets a new signature: an atomic replace makes a new inode, an in-place write a new ctime."""
    try:
        with open(path, "rb") as fh:
            st = os.fstat(fh.fileno())
            if not stat.S_ISREG(st.st_mode):
                return None, None
            return fh.read(), _sig_of(st)
    except (FileNotFoundError, IsADirectoryError, NotADirectoryError):
        return None, None


def _stat_sig(path: str) -> Optional[str]:
    """The stat signature now: None when the file does not exist, 'error' when it cannot be checked."""
    try:
        st = os.stat(path)
    except (FileNotFoundError, NotADirectoryError):
        return None
    except OSError:
        return "error"
    return _sig_of(st) if stat.S_ISREG(st.st_mode) else None


def _unchanged(sha: Optional[str], snap_sig: Optional[str], now_sig: Optional[str]) -> bool:
    """Without reading the file: True when it certainly still holds what the snapshot recorded."""
    if now_sig == "error":
        return False
    if sha is None:
        return now_sig is None
    return snap_sig is not None and now_sig == snap_sig


def _sha(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _chunks(seq: Sequence, n: int = 500) -> Iterator[Sequence]:
    for k in range(0, len(seq), n):
        yield seq[k:k + n]


def _rel_paths(dataset_id: str, image_ids: Sequence[int]) -> Dict[int, str]:
    out: Dict[int, str] = {}
    ids = sorted({int(i) for i in image_ids})
    for chunk in _chunks(ids, 20000):
        for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?))",
                          (dataset_id, json.dumps(list(chunk)))):
            out[r["id"]] = r["rel_path"]
    return out


def snapshot_before(dataset_id: str, label_paths: Sequence[str], op: str,
                    image_ids: Optional[Sequence[Optional[int]]] = None, name: str = "",
                    kind: str = "auto", keep: Sequence[int] = (), track: bool = False) -> int:
    """Store the current bytes of `label_paths` (parallel `image_ids` optional) and return the snapshot id.

    Call it BEFORE a bulk write. File reads happen outside any transaction; the rows go in with one short
    write transaction. Unreadable files (permissions) are skipped rather than recorded as absent — a
    restore must never delete a file that merely could not be read. Each file also records its image's index
    key (rel_path), so a restore finds the file again after the dataset was re-indexed under new ids.

    `track`: the op will record its writes with a WriteLog (which also marks the snapshot tracked by itself).
    Retention runs later on a background thread (schedule_prune). `keep` is accepted for compatibility: a
    snapshot in use is protected by pinned(), which the background prune honours."""
    ids = list(image_ids) if image_ids is not None else [None] * len(label_paths)
    if len(ids) != len(label_paths):
        raise ValueError("image_ids must be parallel to label_paths")
    seen = set()
    files: List[Tuple[Optional[int], str, Optional[str], Optional[str]]] = []
    blobs: Dict[str, bytes] = {}
    for iid, lp in zip(ids, label_paths):
        if not lp or lp in seen:
            continue
        seen.add(lp)
        try:
            data, sig = _read_sig(lp)
        except OSError:
            continue
        if data is None:
            files.append((iid, lp, None, None))
            continue
        sha = _sha(data)
        if sha not in blobs:
            blobs[sha] = data
        files.append((iid, lp, sha, sig))
    rels = _rel_paths(dataset_id, [f[0] for f in files if f[0] is not None])
    shas = list(blobs)
    known = set()
    for chunk in _chunks(shas):
        known.update(r["sha"] for r in db.query("SELECT sha FROM label_blobs WHERE sha IN (%s)" % ",".join("?" * len(chunk)), chunk))
    new_blobs = {sha: gzip.compress(data, 6) for sha, data in blobs.items() if sha not in known}
    with db.transaction() as conn:
        # gc_blobs() may have dropped a blob between the check above and this write transaction (another snapshot
        # sharing it was deleted). Write transactions are serialised, so the re-check here is race-free; only
        # the (rare) blobs that vanished are compressed while the lock is held.
        still = set()
        for chunk in _chunks(sorted(known)):
            still.update(r[0] for r in conn.execute("SELECT sha FROM label_blobs WHERE sha IN (%s)" % ",".join("?" * len(chunk)), chunk))
        for sha in known - still:
            new_blobs[sha] = gzip.compress(blobs[sha], 6)
        cur = conn.execute("INSERT INTO snapshots(dataset_id, name, kind, op, n_files, created_at, tracked) "
                           "VALUES(?,?,?,?,?,?,?)", (dataset_id, name or "", kind, op or "", len(files), db.now(),
                                                    1 if track else 0))
        sid = cur.lastrowid
        conn.executemany("INSERT OR IGNORE INTO label_blobs(sha, data) VALUES(?,?)", list(new_blobs.items()))
        conn.executemany("INSERT OR IGNORE INTO snapshot_files(snapshot_id, image_id, label_path, sha, rel_path, sig) "
                         "VALUES(?,?,?,?,?,?)",
                         [(sid, iid, lp, sha, rels.get(iid) if iid is not None else None, sig) for iid, lp, sha, sig in files])
    if kind == "auto":
        schedule_prune(dataset_id)
    return int(sid)


def snapshot_images(dataset_id: str, image_ids: Iterable[int], op: str, name: str = "", kind: str = "auto",
                    only_existing: bool = False, track: bool = False) -> int:
    """snapshot_before for a set of indexed images (their current label paths). `only_existing` leaves out
    images without a label file — right for an op that only rewrites existing files (a bulk class change):
    restoring it must not delete label files the user creates later for those images."""
    ids = [int(x) for x in image_ids]
    rows: List[dict] = []
    for k in range(0, len(ids), 20000):
        rows += db.query("SELECT id, label_path FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?)) "
                         "AND label_path IS NOT NULL", (dataset_id, json.dumps(ids[k:k + 20000])))
    if only_existing:
        rows = [r for r in rows if os.path.isfile(r["label_path"])]
    return snapshot_before(dataset_id, [r["label_path"] for r in rows], op, [r["id"] for r in rows], name, kind,
                           track=track)


def snapshot_dataset(dataset_id: str, name: str = "", kind: str = "manual", op: str = "manual") -> int:
    """Every label file of the dataset (absent ones included, so a restore also removes files created later)."""
    rows = db.query("SELECT id, label_path FROM images WHERE dataset_id=? AND label_path IS NOT NULL ORDER BY id", (dataset_id,))
    return snapshot_before(dataset_id, [r["label_path"] for r in rows], op, [r["id"] for r in rows], name, kind)


def prune(dataset_id: str, keep: int = KEEP_AUTO, exclude: Sequence[int] = (), keep_small: int = KEEP_SMALL) -> int:
    """Drop automatic snapshots beyond the newest `keep` of more than SMALL_FILES files and the newest
    `keep_small` small ones — never one in `exclude` nor one in use (being restored or compared) — and finish
    dropping any left half-deleted. Returns how many were dropped. Short transactions only (see _drop)."""
    with _pin_lock:
        skip = {int(x) for x in exclude} | set(_pins)
        rows = db.query("SELECT id, n_files FROM snapshots WHERE dataset_id=? AND kind='auto' "
                        "ORDER BY created_at DESC, id DESC", (dataset_id,))
        big = small = 0
        old: List[int] = []
        for r in rows:
            if int(r["n_files"] or 0) <= SMALL_FILES:
                small += 1
                if small > keep_small and r["id"] not in skip:
                    old.append(r["id"])
            else:
                big += 1
                if big > keep and r["id"] not in skip:
                    old.append(r["id"])
        _mark_deleting(old)          # gone for get/list/restore from here on, before the slow part
    left = [r["id"] for r in db.query("SELECT id FROM snapshots WHERE dataset_id=? AND kind='deleting'", (dataset_id,))]
    _drop(sorted(set(old) | set(left)))
    return len(old)


def drop_dataset(dataset_id: str) -> int:
    """Every snapshot of a dataset that is being deleted, in short transactions (its jobs were cancelled, so
    nothing is restoring them). Returns how many were dropped."""
    ids = [r["id"] for r in db.query("SELECT id FROM snapshots WHERE dataset_id=?", (dataset_id,))]
    _mark_deleting(ids)
    _drop(ids)
    return len(ids)


def _mark_deleting(ids: Sequence[int]) -> None:
    if ids:
        with db.transaction() as conn:
            conn.executemany("UPDATE snapshots SET kind='deleting' WHERE id=?", [(int(i),) for i in ids])


def _drop(ids: Sequence[int]) -> None:
    """Delete snapshots already marked 'deleting': their file rows in chunks of DROP_CHUNK (each its own short
    transaction, so other writers interleave), then the row, then the blobs only they referenced."""
    for sid in ids:
        shas = [r["sha"] for r in db.query("SELECT DISTINCT sha FROM snapshot_files WHERE snapshot_id=? AND sha IS NOT NULL",
                                           (sid,))]
        for table in ("snapshot_files", "snapshot_preds"):
            while True:
                with db.transaction() as conn:
                    n = conn.execute(f"DELETE FROM {table} WHERE rowid IN (SELECT rowid FROM {table} "
                                     "WHERE snapshot_id=? LIMIT ?)", (sid, DROP_CHUNK)).rowcount or 0
                if n < DROP_CHUNK:
                    break
        with db.transaction() as conn:
            conn.execute("DELETE FROM snapshots WHERE id=?", (sid,))
        forget_diff(sid)
        gc_blobs(shas)


def gc_blobs(shas: Optional[Iterable[str]] = None) -> int:
    """Delete blobs no snapshot file refers to: only among `shas` when given (what a dropped snapshot held),
    else every blob (a full scan — after a dataset is deleted)."""
    if shas is None:
        with _gc_lock, db.transaction() as conn:
            cur = conn.execute("DELETE FROM label_blobs WHERE NOT EXISTS "
                               "(SELECT 1 FROM snapshot_files f WHERE f.sha = label_blobs.sha)")
            return cur.rowcount or 0
    gone = 0
    for chunk in _chunks(sorted({s for s in shas if s})):
        with _gc_lock, db.transaction() as conn:
            cur = conn.execute("DELETE FROM label_blobs WHERE sha IN (%s) AND NOT EXISTS "
                               "(SELECT 1 FROM snapshot_files f WHERE f.sha = label_blobs.sha)" % ",".join("?" * len(chunk)),
                               list(chunk))
            gone += cur.rowcount or 0
    return gone


def delete_snapshot(dataset_id: str, sid: int) -> bool:
    """Raises SnapshotBusy while the snapshot is being restored or compared."""
    with _pin_lock:
        if _pins.get(int(sid), 0) > 0:
            raise SnapshotBusy("This version is being restored or compared right now — try again when that has finished")
        with db.transaction() as conn:
            cur = conn.execute("UPDATE snapshots SET kind='deleting' WHERE id=? AND dataset_id=? AND kind!='deleting'",
                               (sid, dataset_id))
            gone = bool(cur.rowcount)
    if gone:
        _drop([int(sid)])
    return gone


def get_snapshot(dataset_id: str, sid: int) -> Optional[dict]:
    return db.query_one("SELECT id, name, kind, op, n_files, created_at, tracked FROM snapshots "
                        "WHERE id=? AND dataset_id=? AND kind!='deleting'", (sid, dataset_id))


def list_snapshots(dataset_id: str, limit: int = 200) -> List[dict]:
    return db.query("SELECT id, name, kind, op, n_files, created_at, tracked FROM snapshots WHERE dataset_id=? "
                    "AND kind!='deleting' ORDER BY created_at DESC, id DESC LIMIT ?", (dataset_id, limit))


# ------------------------------------------------------------------ background retention

class _Housekeeper:
    """Runs prune() for datasets that took a new automatic snapshot, on one daemon thread, DELAY seconds after
    the first request (requests meanwhile are coalesced). Never on the writer's request thread."""

    DELAY = 0.3

    def __init__(self) -> None:
        self._cv = threading.Condition()
        self._due: Dict[str, float] = {}
        self._busy = False
        self._thread: Optional[threading.Thread] = None

    def request(self, dataset_id: str) -> None:
        with self._cv:
            self._due.setdefault(dataset_id, time.monotonic() + self.DELAY)
            if self._thread is None or not self._thread.is_alive():
                self._thread = threading.Thread(target=self._loop, name="fovea-snapshot-prune", daemon=True)
                self._thread.start()
            self._cv.notify_all()

    def _loop(self) -> None:
        while True:
            with self._cv:
                while True:
                    if self._due:
                        ds, due = min(self._due.items(), key=lambda kv: kv[1])
                        wait = due - time.monotonic()
                        if wait <= 0:
                            del self._due[ds]
                            self._busy = True
                            break
                        self._cv.wait(wait)
                    else:
                        self._cv.notify_all()          # settle() waiters
                        self._cv.wait(60)
            try:
                prune(ds)                              # skips every snapshot pinned (in use) right now
            except Exception:  # noqa: BLE001 — retention is housekeeping: the next snapshot asks again
                log.exception("pruning label snapshots of %s failed", ds)
            finally:
                with self._cv:
                    self._busy = False
                    self._cv.notify_all()

    def settle(self, timeout: float = 30.0) -> bool:
        """Wait until every requested prune ran (tests; shutdown). True when idle."""
        end = time.monotonic() + timeout
        with self._cv:
            for ds in self._due:
                self._due[ds] = time.monotonic()      # no need to wait out the debounce
            self._cv.notify_all()
            while self._due or self._busy:
                left = end - time.monotonic()
                if left <= 0:
                    return False
                self._cv.wait(min(left, 0.2))
        return True


_housekeeper = _Housekeeper()


def schedule_prune(dataset_id: str) -> None:
    _housekeeper.request(dataset_id)


def settle(timeout: float = 30.0) -> bool:
    return _housekeeper.settle(timeout)


# ------------------------------------------------------------------ what an op wrote

class WriteLog:
    """What a tracked operation wrote, recorded against the snapshot(s) taken before it.

    Call wrote(path) right after each label-file write — while still holding LABEL_LOCK, so what is read back is
    the op's own write (or pass `version` / `data` when the writer has them) — and close() when the op ends. An
    Undo (restore mode 'undo') then reverts a file only while it still holds what the op left there, and close()
    drops from the snapshot every file the op did not write: the undo point covers what the op changed and
    nothing else (an AI 'fill' that skipped an image never deletes the label made for it by hand later).

    Records are buffered and flushed every FLUSH_EVERY files and at close(); a write recorded into several
    snapshots (an Economy run takes one per provider batch) lands in whichever holds that file."""

    FLUSH_EVERY = 200
    _UNSET = object()

    def __init__(self, sids: Union[int, Iterable[Optional[int]], None]):
        if sids is None:
            ids: List[int] = []
        elif isinstance(sids, int):
            ids = [sids]
        else:
            ids = [int(x) for x in sids if x]
        self.sids = sorted(set(ids))
        self._buf: Dict[str, str] = {}
        self._preds: List[Tuple[int, int, str, str, Optional[str]]] = []
        self._lock = threading.Lock()
        self.recorded = 0
        if self.sids:
            with db.transaction() as conn:
                conn.execute("UPDATE snapshots SET tracked=1 WHERE tracked=0 AND id IN (%s)" % ",".join("?" * len(self.sids)),
                             self.sids)

    def wrote(self, label_path, version: Optional[str] = None, data=_UNSET) -> None:
        """`data`: the bytes now in the file (None: the op removed it / left none); default: read it back."""
        if not self.sids or not label_path:
            return
        if version is None:
            if data is WriteLog._UNSET:
                try:
                    with open(label_path, "rb") as fh:
                        data = fh.read()
                except (FileNotFoundError, NotADirectoryError):
                    data = None
                except OSError:
                    return                 # cannot tell what is there: unrecorded, so an Undo leaves it alone
            version = GONE if data is None else _label_version(data)
        with self._lock:
            self._buf[str(label_path)] = version
            full = len(self._buf) >= self.FLUSH_EVERY
        if full:
            self.flush()

    def decided(self, image_id: int, preds: Iterable, state: str = "pending", after_state: str = "accepted") -> None:
        """Record AI proposals this op decided on `image_id` (pred rows as dicts with `id` and `reason` — e.g. what
        ai.store.claim_pending returned — or bare ids): restoring the snapshot for that image puts each back to
        `state` while it still holds `after_state`, with the reason it had before."""
        if not self.sids:
            return
        rows = []
        for p in preds:
            pid, reason = (p.get("id"), p.get("reason")) if isinstance(p, dict) else (p, None)
            if pid is not None:
                rows.append((int(pid), int(image_id), state, after_state, reason))
        if not rows:
            return
        with self._lock:
            self._preds.extend(rows)
            full = len(self._preds) >= self.FLUSH_EVERY * 10
        if full:
            self.flush()

    def flush(self) -> None:
        with self._lock:
            buf, self._buf = self._buf, {}
            preds, self._preds = self._preds, []
        if (not buf and not preds) or not self.sids:
            return
        q = ",".join("?" * len(self.sids))
        try:
            with db.transaction() as conn:
                if buf:
                    conn.executemany(f"UPDATE snapshot_files SET after_ver=? WHERE label_path=? AND snapshot_id IN ({q})",
                                     [(v, p, *self.sids) for p, v in buf.items()])
                if preds:
                    # The snapshot that holds the image's file (an Economy run has several), else the first one.
                    home: Dict[int, int] = {}
                    if len(self.sids) > 1:
                        for chunk in _chunks(sorted({r[1] for r in preds})):
                            home.update((r[0], r[1]) for r in conn.execute(
                                f"SELECT image_id, MIN(snapshot_id) FROM snapshot_files WHERE snapshot_id IN ({q}) AND "
                                f"image_id IN ({','.join('?' * len(chunk))}) GROUP BY image_id", [*self.sids, *chunk]))
                    conn.executemany("INSERT OR REPLACE INTO snapshot_preds(snapshot_id, pred_id, image_id, state, "
                                     "after_state, reason) VALUES(?,?,?,?,?,?)",
                                     [(home.get(iid, self.sids[0]), pid, iid, st, after, reason)
                                      for pid, iid, st, after, reason in preds])
            self.recorded += len(buf)
        except Exception:  # noqa: BLE001 — never fail the op over its undo bookkeeping; retried at the next flush
            log.exception("recording label writes for snapshot(s) %s failed", self.sids)
            with self._lock:
                self._buf = {**buf, **self._buf}
                self._preds = preds + self._preds

    def close(self, trim: bool = True, drop_empty: bool = True) -> Optional[int]:
        """Flush, then (trim) keep only the files the op wrote. A snapshot left with no file is dropped
        (drop_empty) unless it is in use. Returns the first remaining snapshot id, or None."""
        self.flush()
        kept: List[int] = []
        for sid in self.sids:
            if trim:
                shas = [r["sha"] for r in db.query("SELECT DISTINCT sha FROM snapshot_files WHERE snapshot_id=? AND "
                                                   "after_ver IS NULL AND sha IS NOT NULL", (sid,))]
                while True:
                    with db.transaction() as conn:
                        n = conn.execute("DELETE FROM snapshot_files WHERE rowid IN (SELECT rowid FROM snapshot_files "
                                         "WHERE snapshot_id=? AND after_ver IS NULL LIMIT ?)", (sid, DROP_CHUNK)).rowcount or 0
                    if n < DROP_CHUNK:
                        break
                gc_blobs(shas)
                left = db.query_one("SELECT COUNT(*) n FROM snapshot_files WHERE snapshot_id=?", (sid,))["n"]
                forget_diff(sid)
                if not left and drop_empty and not is_pinned(sid):
                    with _pin_lock:
                        if _pins.get(sid, 0) == 0:
                            _mark_deleting([sid])
                            dropped = True
                        else:
                            dropped = False
                    if dropped:
                        _drop([sid])
                        continue
                db.execute("UPDATE snapshots SET n_files=? WHERE id=? AND kind!='deleting'", (left, sid))
            if db.query_one("SELECT 1 FROM snapshots WHERE id=? AND kind!='deleting'", (sid,)):
                kept.append(sid)
        return kept[0] if kept else None


_OP_TEXT = {
    "bulk set_class": "Bulk class change", "bulk delete_class": "Class deletion", "bulk remap": "Class remap",
    "bulk clear": "Label clearing", "ai_accept": "Accepting AI proposals", "objects": "Objects edit",
}


def describe(snap: dict) -> str:
    """A name people recognise for a version: its own name, else what it was taken before and when
    ('AI append write · 23:50'). History never shows snapshot ids, so neither may anything that names one."""
    if snap.get("name"):
        return str(snap["name"])
    op = str(snap.get("op") or "")
    if snap.get("kind") == "manual" or op == "manual":
        what = "Saved version"
    elif op in _OP_TEXT:
        what = _OP_TEXT[op]
    elif op.startswith("bulk"):
        what = "Bulk label edit"
    elif op.startswith("ai_"):
        what = f"AI {op[3:].replace('_', ' ')} write"
    elif op.startswith("restore"):
        what = "Undo point of a restore"
    elif op.startswith("range"):
        what = "Range edit"
    else:
        what = op.replace("_", " ").capitalize() or "Automatic version"
    at = snap.get("created_at")
    if not at:
        return what
    lt, today = time.localtime(at), time.localtime()
    when = time.strftime("%H:%M" if lt[:3] == today[:3] else "%b %d %H:%M", lt)
    return f"{what} · {when}"


def _blob(sha: Optional[str]) -> Optional[bytes]:
    if sha is None:
        return None
    row = db.query_one("SELECT data FROM label_blobs WHERE sha=?", (sha,))
    if not row:
        return None
    return gzip.decompress(row["data"])


def _blobs(shas: Iterable[Optional[str]]) -> Dict[str, bytes]:
    want = sorted({s for s in shas if s})
    out: Dict[str, bytes] = {}
    for chunk in _chunks(want):
        for r in db.query("SELECT sha, data FROM label_blobs WHERE sha IN (%s)" % ",".join("?" * len(chunk)), chunk):
            out[r["sha"]] = gzip.decompress(r["data"])
    return out


def _rows_full(text: str) -> List[list]:
    boxes, _c, _i = parse_label_text(text)
    return [[int(b[0])] + [round(float(v), 6) for v in b[1:5]] for b in boxes]


def _rows(data: Optional[bytes]) -> List[list]:
    """The boxes of a label file for comparing versions: [cls, xc, yc, w, h] rounded to 6 places — exactly what
    parse_label_text keeps — with a fast path for plain 5/6-column files (a diff may parse tens of thousands)."""
    if not data:
        return []
    text = data.decode("utf-8-sig", errors="replace")
    out: List[list] = []
    isfinite = math.isfinite
    for raw in text.splitlines():
        parts = raw.split()
        n = len(parts)
        if n < 5:
            continue                       # blank, or malformed: the parser drops it too
        if n > 6:
            return _rows_full(text)        # segmentation / pose rows: their box comes from the full parser
        try:
            c, xc, yc, w, h = (float(x) for x in parts[:5])
            if n == 6:
                float(parts[5])
        except ValueError:
            continue
        if not (isfinite(c) and isfinite(xc) and isfinite(yc) and isfinite(w) and isfinite(h)) or abs(c) > MAX_CLASS_ID:
            continue
        out.append([int(c), round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)])
    return out


def _files(sid: int, image_ids: Optional[Iterable[int]] = None) -> List[dict]:
    rows = db.query("SELECT image_id, label_path, sha, rel_path, sig, after_ver FROM snapshot_files WHERE snapshot_id=? "
                    "ORDER BY image_id, label_path", (sid,))
    if image_ids is not None:
        want = {int(x) for x in image_ids}
        rows = [r for r in rows if r["image_id"] in want]
    return rows


def _bases(dataset_id: str) -> List[str]:
    row = db.query_one("SELECT root, layout FROM datasets WHERE id=?", (dataset_id,))
    return dataset_bases(row["root"], db.loads(row["layout"], {})) if row else []


def _inside(path: str, bases: List[str]) -> bool:
    real = os.path.realpath(path)
    return any(real == b or real.startswith(b.rstrip(os.sep) + os.sep) for b in bases)


def _locate(dataset_id: str, files: List[dict]) -> List[Tuple[Optional[str], Optional[int], Optional[str]]]:
    """Where each snapshot file lives now: (label path, image id, rel_path), or (None, None, None).

    The image's current label path when it is still indexed — by id, else by its recorded index key (a re-index
    may have given it a new id). A file whose image is gone is restored to its recorded path only while that path
    is still inside the dataset's folders: after a relocation the old root is not the dataset any more, and a
    restore must never re-create it."""
    ids = sorted({f["image_id"] for f in files if f["image_id"] is not None})
    by_id: Dict[int, dict] = {}
    for chunk in _chunks(ids, 20000):
        for r in db.query("SELECT id, rel_path, label_path FROM images WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?))",
                          (dataset_id, json.dumps(list(chunk)))):
            by_id[r["id"]] = r
    rels = sorted({f["rel_path"] for f in files if f.get("rel_path") and f["image_id"] not in by_id})
    by_rel: Dict[str, dict] = {}
    for chunk in _chunks(rels, 20000):
        for r in db.query("SELECT id, rel_path, label_path FROM images WHERE dataset_id=? AND rel_path IN (SELECT value FROM json_each(?))",
                          (dataset_id, json.dumps(list(chunk)))):
            by_rel[r["rel_path"]] = r
    bases: Optional[List[str]] = None
    out: List[Tuple[Optional[str], Optional[int], Optional[str]]] = []
    for f in files:
        img = by_id.get(f["image_id"]) if f["image_id"] is not None else None
        if img is None and f.get("rel_path"):
            img = by_rel.get(f["rel_path"])
        if img is not None and img.get("label_path"):
            out.append((img["label_path"], img["id"], img["rel_path"]))
            continue
        if bases is None:
            bases = _bases(dataset_id)
        if img is None and _inside(f["label_path"], bases):
            out.append((f["label_path"], None, f.get("rel_path")))
        else:
            out.append((None, None, None))
    return out


# ------------------------------------------------------------------ diff
# The per-file comparison is cached per snapshot together with each file's stat signature. A later call re-stats
# the files and recomputes only those whose signature changed, so paging the diff or reopening History costs a
# stat per file instead of reading, hashing and parsing every file again — and edits made outside Fovea are still
# seen. The first comparison skips reading a file whose stat still matches the one the snapshot recorded.
DIFF_CACHE_MAX = 4             # snapshots whose comparison is kept in memory
DIFF_IDLE = 600.0              # ... for at most this long after their last use
_diff_cache: Dict[int, dict] = {}
_diff_lock = threading.Lock()
_diff_sid_locks: Dict[int, threading.Lock] = {}


def _line_counts(data: Optional[bytes]) -> Counter:
    if not data:
        return Counter()
    return Counter(ln for ln in (raw.strip() for raw in data.decode("utf-8-sig", errors="replace").splitlines()) if ln)


def _line_rows(lines: Iterable[str], memo: Dict[tuple, object]) -> List[tuple]:
    """Rows of individual label lines as (cls, xc, yc, w, h) rounded like _rows. `memo` maps a line's geometry tokens
    to their parsed values: a re-classed box costs one parse for both versions."""
    out: List[tuple] = []
    isfinite = math.isfinite
    for line in lines:
        parts = line.split()
        n = len(parts)
        if n < 5:
            continue
        if n > 6:
            out.extend(tuple(r) for r in _rows_full(line))
            continue
        key = tuple(parts[1:])
        g = memo.get(key)
        if g is None:
            try:
                xc, yc, w, h = (float(x) for x in parts[1:5])
                if n == 6:
                    float(parts[5])
                g = (round(xc, 6), round(yc, 6), round(w, 6), round(h, 6)) \
                    if isfinite(xc) and isfinite(yc) and isfinite(w) and isfinite(h) else False
            except ValueError:
                g = False
            memo[key] = g
        if g is False:
            continue
        try:
            c = float(parts[0])
        except ValueError:
            continue
        if isfinite(c) and abs(c) <= MAX_CLASS_ID:
            out.append((int(c),) + g)
    return out


def _compare(sha: Optional[str], data: Optional[bytes], before: Optional[bytes]) -> Optional[dict]:
    """None when the file matches the snapshot, else what changed (boxes per class). Lines present in both versions
    cancel out before anything is parsed (a row is a function of its line alone)."""
    if (data is None and sha is None) or (data is not None and sha == _sha(data)):
        return None
    a, b = _line_counts(before), _line_counts(data)
    memo: Dict[tuple, object] = {}
    rb, ra = Counter(_line_rows((a - b).elements(), memo)), Counter(_line_rows((b - a).elements(), memo))
    removed, added = rb - ra, ra - rb
    return {"exists": data is not None, "added": _by_cls(added), "removed": _by_cls(removed),
            "n_added": sum(added.values()), "n_removed": sum(removed.values())}


def _by_cls(rows: Counter) -> Tuple[Tuple[int, int], ...]:
    """((cls, boxes), ...): compact, since a big diff keeps one per changed file in memory."""
    out: Counter = Counter()
    for k, n in rows.items():
        out[k[0]] += n
    return tuple(sorted(out.items()))


def _diff_state(dataset_id: str, sid: int) -> dict:
    ds = db.query_one("SELECT root, scanned_at FROM datasets WHERE id=?", (dataset_id,)) or {}
    where = (ds.get("root"), ds.get("scanned_at"))          # a re-index can move files: locate them again
    with _diff_lock:
        lock = _diff_sid_locks.setdefault(sid, threading.Lock())
    with lock:
        with _diff_lock:
            st = _diff_cache.get(sid)
        if st is None or st["dataset"] != dataset_id or st["where"] != where:
            files = _files(sid)
            recs = []
            for f, (path, iid, rel) in zip(files, _locate(dataset_id, files)):
                if path is None:
                    continue                          # no longer part of the dataset: nothing to compare with
                recs.append({"path": path, "image_id": iid, "rel_path": rel or Path(path).name, "sha": f["sha"],
                             "snap_sig": f.get("sig"), "sig": False, "info": None})
            st = {"dataset": dataset_id, "where": where, "recs": recs}
        stale = []
        for rec in st["recs"]:
            sig = _stat_sig(rec["path"])
            if sig == rec["sig"] and sig != "error":
                continue
            if rec["sig"] is False and _unchanged(rec["sha"], rec["snap_sig"], sig):
                rec["sig"], rec["info"] = sig, None     # first look, and the stat is the snapshot's own
                continue
            stale.append(rec)
        if stale:
            befores = _blobs(rec["sha"] for rec in stale)
            for rec in stale:
                try:
                    data, sig = _read_sig(rec["path"])
                except OSError:
                    rec["sig"], rec["info"] = "error", None
                    continue
                rec["info"] = _compare(rec["sha"], data, befores.get(rec["sha"]) if rec["sha"] else None)
                rec["sig"] = sig
        st["at"] = time.time()
        with _diff_lock:
            _diff_cache[sid] = st
            idle = [k for k, v in _diff_cache.items() if time.time() - v.get("at", 0) > DIFF_IDLE]
            for k in idle:
                _diff_cache.pop(k, None)
            while len(_diff_cache) > DIFF_CACHE_MAX:
                _diff_cache.pop(min(_diff_cache, key=lambda k: _diff_cache[k].get("at", 0)), None)
        return st


def diff(dataset_id: str, sid: int, offset: int = 0, limit: int = 50) -> dict:
    """What changed between the snapshot and the label files on disk now."""
    with pinned(sid):
        st = _diff_state(dataset_id, sid)
        changed = [r for r in st["recs"] if r["info"] is not None]
        added: Counter = Counter()
        removed: Counter = Counter()
        for r in changed:
            for c, n in r["info"]["added"]:
                added[c] += n
            for c, n in r["info"]["removed"]:
                removed[c] += n
        per_class = {str(c): {"added": added.get(c, 0), "removed": removed.get(c, 0)} for c in sorted(set(added) | set(removed))}
        summary = {"images_changed": len(changed), "boxes_added": sum(added.values()),
                   "boxes_removed": sum(removed.values()), "per_class": per_class,
                   "proposals": _preds_to_put_back(sid)}
        page = changed[offset:offset + limit]
        befores = _blobs(r["sha"] for r in page)
        items = []
        for r in page:
            try:
                now = _read(r["path"])
            except OSError:
                now = None
            items.append({"image_id": r["image_id"], "rel_path": r["rel_path"],
                          "before": _rows(befores.get(r["sha"]) if r["sha"] else None), "after": _rows(now),
                          "existed": r["sha"] is not None, "exists": r["info"]["exists"],
                          "added": r["info"]["n_added"], "removed": r["info"]["n_removed"]})
    return {"summary": summary, "items": items, "total": len(changed), "offset": offset, "limit": limit}


def forget_diff(sid: Optional[int] = None) -> None:
    with _diff_lock:
        if sid is None:
            _diff_cache.clear()
        else:
            _diff_cache.pop(sid, None)


# ------------------------------------------------------------------ restore

def _write_bytes(path: Path, data: bytes, dataset_id: Optional[str] = None) -> None:
    """Atomic replace keeping an existing file's permission bits (like labels.write_label_file). A symlinked
    label is written through its link when that stays inside the dataset, refused otherwise (write_target)."""
    path = write_target(path, dataset_id)
    if path.exists():
        try:
            mode = stat.S_IMODE(os.stat(path).st_mode)
        except OSError:
            mode = None
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        umask = os.umask(0)
        os.umask(umask)
        mode = 0o666 & ~umask
    fd, tmp = tempfile.mkstemp(prefix=".fovea-", suffix=".txt", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as fh:
            fh.write(data)
        if mode is not None:
            os.chmod(tmp, mode)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


RESTORE_MODES = ("full", "undo")


def _proposals_text(proposals: Dict[str, int]) -> List[str]:
    return [f"{n:,} AI proposal{'s' if n != 1 else ''} {'pending' if st == 'pending' else st} again"
            for st, n in sorted(proposals.items()) if n]


def _restore_message(restored: int, removed: int, kept: int, changed_since: int, errors: int,
                     proposals: Optional[Dict[str, int]] = None) -> str:
    parts = [f"Restored {restored:,} file{'s' if restored != 1 else ''}"]
    if removed:
        parts.append(f"removed {removed:,}")
    if changed_since:
        parts.append(f"{changed_since:,} edited since — kept")
    if kept:
        parts.append(f"{kept:,} changed meanwhile and kept")
    if errors:
        parts.append(f"{errors:,} failed")
    parts += _proposals_text(proposals or {})
    return " · ".join(parts)


# ------------------------------------------------------------------ AI proposal decisions (snapshot_preds)

def _pred_rows(sid: int, image_ids: Optional[Iterable[int]] = None) -> List[dict]:
    rows = db.query("SELECT pred_id, image_id, state, after_state, reason FROM snapshot_preds WHERE snapshot_id=? "
                    "ORDER BY image_id, pred_id", (sid,))
    if image_ids is not None:
        want = {int(x) for x in image_ids}
        rows = [r for r in rows if r["image_id"] in want]
    return rows


def _preds_to_put_back(sid: int) -> int:
    """How many of the snapshot's recorded proposal decisions a restore would undo (preds still as the op left them)."""
    r = db.query_one("SELECT COUNT(*) n FROM snapshot_preds s JOIN ai_preds p ON p.id = s.pred_id "
                     "WHERE s.snapshot_id=? AND p.state = s.after_state", (sid,))
    return int(r["n"]) if r else 0


def _put_back_preds(dataset_id: str, rows: List[dict], blocked: set, undo_id: Optional[int]) -> Dict[str, int]:
    """Put the proposal decisions a snapshot recorded back (each pred only while it still holds what the op left),
    except on `blocked` images — those whose label file this restore did not put back (edited since, failed, not
    reached). The reverse goes into the restore's own undo point, so undoing the restore re-decides them too.
    Returns {state: count}."""
    todo = [r for r in rows if r["image_id"] not in blocked]
    out: Counter = Counter()
    if not todo:
        return {}
    now = db.now()
    with db.transaction() as conn:
        for chunk in _chunks(todo):
            cur = {r[0]: (r[1], r[2]) for r in conn.execute(
                "SELECT id, state, reason FROM ai_preds WHERE dataset_id=? AND id IN (SELECT value FROM json_each(?))",
                (dataset_id, json.dumps([r["pred_id"] for r in chunk])))}
            go = [r for r in chunk if cur.get(r["pred_id"], (None,))[0] == r["after_state"]]
            if not go:
                continue
            conn.executemany("UPDATE ai_preds SET state=?, decided_at=?, reason=COALESCE(?, reason) WHERE id=? AND state=?",
                             [(r["state"], None if r["state"] == "pending" else now, r["reason"], r["pred_id"],
                               r["after_state"]) for r in go])
            if undo_id:
                conn.executemany("INSERT OR REPLACE INTO snapshot_preds(snapshot_id, pred_id, image_id, state, after_state, "
                                 "reason) VALUES(?,?,?,?,?,?)",
                                 [(undo_id, r["pred_id"], r["image_id"], r["after_state"], r["state"],
                                   cur[r["pred_id"]][1]) for r in go])
            out.update(r["state"] for r in go)
    return dict(out)


def restore(dataset_id: str, sid: int, image_ids: Optional[Iterable[int]] = None, job=None, mode: str = "full") -> dict:
    """Put the snapshot's bytes back. An automatic snapshot of the files about to change is taken first, so
    a restore is itself undoable. A file the snapshot recorded as absent is removed (the other path that deletes a
    label file is Annotate's remove_if_empty). Each image is then re-indexed.

    mode 'full' (History "Restore version") puts every file of the snapshot back. mode 'undo' (the Undo of the
    op that took the snapshot) only reverts files the op wrote, and only while they still hold exactly what it
    left there: a file edited since (Annotate, another tool) is kept and counted in `changed_since`. A snapshot
    whose op did not record its writes (older versions) is restored in full either way.

    A file saved by someone else while the restore runs is never overwritten silently: each file is written
    only while it still holds exactly the bytes the undo snapshot recorded (checked under LABEL_LOCK, which every
    label writer holds for its own check-and-write); a newer edit is kept and reported."""
    from .scanner import refresh_dataset_counts, refresh_image

    if mode not in RESTORE_MODES:
        raise ValueError(f"mode must be one of {', '.join(RESTORE_MODES)}")
    with pinned(sid):
        snap = get_snapshot(dataset_id, sid)
        if not snap:
            raise LookupError("Snapshot not found")
        undo = mode == "undo" and bool(snap.get("tracked"))
        image_ids = None if image_ids is None else [int(x) for x in image_ids]
        files = _files(sid, image_ids)
        if undo:
            files = [f for f in files if f.get("after_ver") is not None]     # only what the op wrote
        prows = _pred_rows(sid, image_ids)
        blocked: set = set()                       # images whose file is not put back: their proposals stay decided
        todo = []
        not_found = changed_since = 0
        since: List[str] = []
        for f, (path, iid, _rel) in zip(files, _locate(dataset_id, files)):
            if path is None:
                not_found += 1                     # its image and folder are no longer part of the dataset
                blocked.add(f["image_id"])
                continue
            if _unchanged(f["sha"], f.get("sig"), _stat_sig(path)):
                continue                           # its stat is still the snapshot's: no need to read it
            try:
                now = _read(path)
            except OSError as e:
                todo.append((f, path, iid, e))
                continue
            if (now is None and f["sha"] is None) or (now is not None and f["sha"] == _sha(now)):
                continue
            if undo and (GONE if now is None else _label_version(now)) != f["after_ver"]:
                changed_since += 1                 # edited after the op: undoing must not lose that edit
                since.append(Path(path).name)
                blocked.update((f["image_id"], iid))
                continue
            todo.append((f, path, iid, None))
        problems: List[str] = []
        if not_found:
            problems.append(f"{not_found:,} file(s) skipped: their images are no longer in this dataset")
        if changed_since:
            problems.append(f"{changed_since:,} file(s) were edited after this change and were kept as they are"
                            + (f" ({', '.join(since[:3])}{' …' if changed_since > 3 else ''})" if since else ""))
        if not todo:
            # The files already match (or were all edited since): the op's proposal decisions are still undone.
            props = _put_back_preds(dataset_id, prows, blocked, None) if prows else {}
            if job is not None:
                job.done, job.total, job.updated_at = 0, 0, time.time()
                job.message = " · ".join((["Nothing restored", f"{changed_since:,} edited since — kept"] if changed_since else
                                          ["Nothing to restore — the label files already match"]) + _proposals_text(props))
            return {"restored": 0, "removed": 0, "unchanged": len(files) - not_found - changed_since, "errors": 0,
                    "kept": 0, "changed_since": changed_since, "not_found": not_found, "problems": problems,
                    "snapshot_id": None, "mode": "undo" if undo else "full", "proposals": props}
        undo_id = snapshot_before(dataset_id, [p for _f, p, _i, _e in todo], f"restore {describe(snap)}",
                                  [i for _f, _p, i, _e in todo], keep=[sid], track=True)
        undo_sha = {r["label_path"]: r["sha"] for r in
                    db.query("SELECT label_path, sha FROM snapshot_files WHERE snapshot_id=?", (undo_id,))}
        wlog = WriteLog(undo_id)
        ds = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (dataset_id,)) or {}
        classes = db.loads(ds.get("classes"))
        n_classes = len(classes) if ds.get("classes_source") != "inferred" and classes else None
        restored = removed = errors = kept = 0
        written: List[int] = []
        total = len(todo)
        # Until its file is put back, an image's proposal decisions stay as they are (cancel, failure, newer edit).
        not_done = {x for f, _p, i, _e in todo for x in (f["image_id"], i)}
        props: Dict[str, int] = {}
        try:
            with pinned(undo_id):
                for k, (f, path, iid, err) in enumerate(todo):
                    if job is not None and k % 50 == 0:
                        job.update(done=k, total=total, message=f"Restoring {k:,}/{total:,}")
                    if err is not None:
                        errors += 1
                        problems.append(f"{Path(path).name}: {getattr(err, 'strerror', None) or err}")
                        continue
                    p = Path(path)
                    try:
                        with LABEL_LOCK:
                            if path not in undo_sha:
                                raise OSError("could not be read before the restore, so it was left as it is")
                            cur = _read(path)
                            if (None if cur is None else _sha(cur)) != undo_sha[path]:
                                # Saved after the undo snapshot was taken: overwriting it now would lose that edit
                                # for good (the undo snapshot does not hold it).
                                kept += 1
                                problems.append(f"{p.name}: changed during the restore — the newer edit was kept")
                                continue
                            if f["sha"] is None:
                                if p.is_file():
                                    p.unlink()
                                    removed += 1
                                    wlog.wrote(path, data=None)
                            else:
                                data = _blob(f["sha"])
                                if data is None:
                                    raise OSError("snapshot data missing")
                                _write_bytes(p, data, dataset_id)
                                restored += 1
                                wlog.wrote(path, data=data)
                    except OSError as e:
                        errors += 1
                        problems.append(f"{p.name}: {e.strerror or e}")
                        continue
                    not_done.difference_update((f["image_id"], iid))
                    if iid is not None:
                        refresh_image(iid, n_classes, update_counts=False)
                        written.append(iid)
        finally:
            refresh_dataset_counts(dataset_id)
            notify_label_write(dataset_id, written)
            if prows:
                try:
                    props = _put_back_preds(dataset_id, prows, blocked | not_done, undo_id)
                except Exception:  # noqa: BLE001 — the files are restored; report, do not fail the restore over this
                    log.exception("putting AI proposal decisions of snapshot %s back failed", sid)
            # The restore's own undo point: only the files it actually changed (none: no undo point at all).
            undo_id = wlog.close()
        if job is not None:
            # Not job.update(): the files are written, so this is no longer a cancel point.
            job.done, job.total, job.updated_at = total, total, time.time()
            job.message = _restore_message(restored, removed, kept, changed_since, errors, props)
        return {"restored": restored, "removed": removed, "errors": errors, "kept": kept,
                "changed_since": changed_since, "not_found": not_found, "problems": problems[:5],
                "unchanged": len(files) - not_found - changed_since - total, "snapshot_id": undo_id,
                "mode": "undo" if undo else "full", "proposals": props}
