"""backend-data: index, layout detection, label parsing/writing, image queries, stats, snapshots, class cards.

Run: cd src && python -m pytest tests/test_backend_data.py -q
"""
from __future__ import annotations

import os
import sqlite3
import sys
import threading
import time
import unicodedata
import uuid
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fastapi import HTTPException  # noqa: E402

from fovea import db  # noqa: E402
from fovea.core import imagesq, layout, labels, snapshots, stats  # noqa: E402
from fovea.core.scanner import POST_SCAN_HOOKS, scan_dataset, seq_key  # noqa: E402
from fovea.jobs import Job, JobCancelled  # noqa: E402


# --------------------------------------------------------------------------- helpers

def _img(path: Path, size=(64, 48), color=(120, 120, 120)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path)


def _lbl(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def _register(root, classes=None, source="inferred", raw_path=None):
    db.init_db()
    lay = layout.detect(str(raw_path or root))
    ds_id = "bd-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(classes or []), source, "", "new", now, now))
    return ds_id


def _scan(ds_id, job=None):
    """Scan without the post-scan hooks other roles register (the QA engine would flag these synthetic,
    identical images as near-duplicates from a background job while a test inspects them)."""
    saved = POST_SCAN_HOOKS[:]
    POST_SCAN_HOOKS.clear()
    try:
        return scan_dataset(ds_id, job or Job(id=uuid.uuid4().hex[:8], kind="scan"))
    finally:
        POST_SCAN_HOOKS[:] = saved


def _drop(ds_id):
    db.execute("DELETE FROM reviews WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _ultra(tmp_path, n=3, name="ds", rows="0 0.5 0.5 0.2 0.2\n"):
    root = tmp_path / name
    for split in ("train", "val"):
        for k in range(n):
            _img(root / "images" / split / f"f{k:02d}.jpg")
            _lbl(root / "labels" / split / f"f{k:02d}.txt", rows)
    return root


def _rels(ds_id):
    return sorted(r["rel_path"] for r in db.query("SELECT rel_path FROM images WHERE dataset_id=?", (ds_id,)))


def _iid(ds_id, rel):
    return db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path=?", (ds_id, rel))["id"]


@pytest.fixture(autouse=True)
def _own_label_write_hooks():
    """Other roles' label-write hooks (the QA engine's debounced re-check) would start background jobs on the
    synthetic datasets these tests create and drop; tests that need hooks add their own. The app is imported
    first, so every hook registered at import time is in the saved list (one registered during a test while the
    list is cleared would otherwise be lost for the test files that run after this one)."""
    import fovea.main  # noqa: F401
    saved = labels.LABEL_WRITE_HOOKS[:]
    labels.LABEL_WRITE_HOOKS.clear()
    yield
    labels.LABEL_WRITE_HOOKS[:] = saved


@pytest.fixture()
def scanned(tmp_path):
    root = _ultra(tmp_path)
    ds_id = _register(root)
    _scan(ds_id)
    yield ds_id, root
    _drop(ds_id)


# --------------------------------------------------------------------------- A1 extended rows

SEG = "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40\n"
POSE = "0 0.5 0.5 0.2 0.4 0.5 0.4 2 0.5 0.6 1 0.45 0.55 0\n"


def test_segmentation_and_pose_rows_survive_edits_bulk_ops_and_export(tmp_path):
    from fovea.api.datasets import bulk_labels, get_labels, put_labels

    root = tmp_path / "seg"
    _img(root / "images" / "train" / "a.jpg")
    _lbl(root / "labels" / "train" / "a.txt", SEG + "1 0.5 0.5 0.6 0.5 0.6 0.6\n")
    _img(root / "images" / "train" / "p.jpg")
    _lbl(root / "labels" / "train" / "p.txt", POSE)
    ds_id = _register(root)
    try:
        _scan(ds_id)
        a, p = _iid(ds_id, "train/a.jpg"), _iid(ds_id, "train/p.jpg")
        got = get_labels(ds_id, a)
        assert got["row_kinds"] == ["seg", "seg"] and got["confs"] == [None, None]
        assert got["boxes"][0][1:] == pytest.approx([0.25, 0.25, 0.3, 0.3])     # bbox of the polygon
        # the editor sends back exactly what it got (plus a class change): polygons keep every vertex
        boxes = [[3] + got["boxes"][0][1:], got["boxes"][1]]
        put_labels(ds_id, a, {"boxes": boxes, "base_version": got["version"]})
        text = (root / "labels" / "train" / "a.txt").read_text()
        assert text.splitlines()[0] == "3 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40"
        # moving a polygon's box is refused (409), never flattened into a rectangle
        got = get_labels(ds_id, a)
        moved = [[3, 0.3, 0.3, 0.3, 0.3], got["boxes"][1], [2, 0.8, 0.8, 0.1, 0.1]]
        with pytest.raises(HTTPException) as e:
            put_labels(ds_id, a, {"boxes": moved, "base_version": got["version"]})
        assert e.value.status_code == 409 and e.value.detail["code"] == "extended_rows"
        assert (root / "labels" / "train" / "a.txt").read_text() == text
        # adding a box to a segmentation file writes it as a rectangle polygon (Ultralytics reads every row
        # of such a file as a polygon)
        put_labels(ds_id, a, {"boxes": got["boxes"] + [[2, 0.5, 0.5, 0.2, 0.2]]})
        last = (root / "labels" / "train" / "a.txt").read_text().splitlines()[-1].split()
        assert len(last) == 9 and last[0] == "2"
        # bulk ops change the class and keep the geometry
        res = bulk_labels(ds_id, {"op": "set_class", "cls": 1, "image_ids": [a, p]})
        assert res["images"] == 2
        assert (root / "labels" / "train" / "a.txt").read_text().splitlines()[0] == "1 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40"
        assert (root / "labels" / "train" / "p.txt").read_text().split()[5:] == POSE.split()[5:]
        # export copies polygon rows verbatim
        out = labels.export_label_text(root / "labels" / "train" / "a.txt")
        assert out.splitlines()[0] == "1 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40"
    finally:
        _drop(ds_id)


def test_pose_rows_are_recognised_and_padded():
    rows, issues = labels.parse_label_rows(POSE)
    assert rows[0].kind == "pose" and rows[0].box == [0.5, 0.5, 0.2, 0.4] and not issues
    text = labels.label_text_for([rows[0].as_box(), [1, 0.2, 0.2, 0.1, 0.1]], POSE)
    first, second = text.splitlines()
    assert first == POSE.strip() and len(second.split()) == len(first.split())


# --------------------------------------------------------------------------- A3 BOM, A30 fractional ids

def test_utf8_bom_keeps_the_first_box_and_class_name(tmp_path):
    f = tmp_path / "x.txt"
    f.write_bytes("\ufeff0 0.5 0.5 0.2 0.2\r\n1 0.3 0.3 0.1 0.1\r\n".encode())
    boxes, _c, issues = labels.read_label_file(f)
    assert len(boxes) == 2 and not issues
    assert labels.read_label_snapshot(f)[0] == boxes
    (tmp_path / "classes.txt").write_bytes("\ufeffperson\nfall\n".encode())
    assert layout.read_classes_file(tmp_path) == ["person", "fall"]
    lst = tmp_path / "train.txt"
    _img(tmp_path / "images" / "a.jpg")
    lst.write_bytes("\ufeff./images/a.jpg\n".encode())
    from fovea.core.scanner import enumerate_source
    src = layout.from_list(lst).sources[0]
    assert [e[0] for e in enumerate_source(src)] == ["train/images/a.jpg"]


def test_fractional_class_ids_are_flagged_not_silently_truncated():
    rows, issues = labels.parse_label_rows("1.7 0.5 0.5 0.1 0.1\n0 0.5 0.5 0.1 0.1\n")
    assert "bad_format" in issues and rows[0].issues == {"bad_format"} and rows[1].issues == set()


def test_row_issues_name_the_offending_box():
    rows, issues = labels.parse_label_rows("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.001 0.001\n0 0.99 0.5 0.2 0.2\n")
    assert [sorted(r.issues) for r in rows] == [[], ["tiny_box"], ["out_of_range"]]


# --------------------------------------------------------------------------- A4/A5/A6/A17/A24 layout

def test_picking_the_images_folder_or_a_split_folder_finds_the_labels(tmp_path):
    root = _ultra(tmp_path)
    lay = layout.detect(str(root / "images"))
    assert lay.kind == "ultralytics" and Path(lay.root) == root.resolve()
    lay = layout.detect(str(root / "images" / "train"))
    src = lay.sources[0]
    assert lay.kind == "flat" and src.split == "train" and src.label_dir.endswith("labels/train") and src.label_count == 3


def test_cvat_yolo_export_is_mixed_with_names(tmp_path):
    root = tmp_path / "cvat"
    (root / "obj.names").parent.mkdir(parents=True)
    (root / "obj.names").write_text("helmet\nhead\n")
    (root / "obj.data").write_text("classes = 2\ntrain = data/train.txt\n")
    _img(root / "obj_train_data" / "a.jpg")
    _lbl(root / "obj_train_data" / "a.txt", "1 0.5 0.5 0.2 0.2\n")
    (root / "train.txt").write_text("data/obj_train_data/a.jpg\n")
    lay = layout.detect(str(root))
    assert lay.kind == "mixed" and lay.classes == ["helmet", "head"] and lay.sources[0].split == "train"
    ds_id = _register(root)
    try:
        _scan(ds_id)
        assert db.query_one("SELECT has_label, n_boxes FROM images WHERE dataset_id=?", (ds_id,)) == {"has_label": 1, "n_boxes": 1}
        # its train.txt (entries prefixed 'data/') resolves too
        from fovea.core.scanner import enumerate_source
        assert len(list(enumerate_source(layout.from_list(root / "train.txt").sources[0]))) == 1
    finally:
        _drop(ds_id)


def test_roboflow_yaml_keeps_names_and_resolves_dotdot_paths(tmp_path):
    root = tmp_path / "rf"
    for split in ("train", "valid"):
        _img(root / split / "images" / "a.jpg")
        _lbl(root / split / "labels" / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text("train: ../train/images\nval: ../valid/images\nnc: 2\nnames: ['helmet', 'head']\n")
    for target in (root, root / "data.yaml"):
        lay = layout.detect(str(target))
        assert lay.kind == "yaml" and lay.classes == ["helmet", "head"] and len(lay.sources) == 2, target


def test_yaml_names_survive_when_its_paths_do_not_resolve(tmp_path):
    root = tmp_path / "copied"
    _img(root / "train" / "images" / "a.jpg")
    (root / "train" / "labels").mkdir(parents=True)
    (root / "data.yaml").write_text("train: /elsewhere/train/images\nnames: [person]\n")
    lay = layout.detect(str(root))
    assert lay.kind == "split-first" and lay.classes == ["person"] and lay.classes_from == "yaml"


def test_relative_yaml_path_is_relative_to_the_yaml_not_the_server_cwd(tmp_path, monkeypatch):
    root = _ultra(tmp_path, name="yds")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames: [a]\n")
    elsewhere = tmp_path / "server_cwd"
    (elsewhere / "images" / "train").mkdir(parents=True)
    monkeypatch.chdir(elsewhere)
    lay = layout.detect(str(root / "data.yaml"))
    assert Path(lay.root) == root.resolve()
    assert all(Path(s.img_dir).is_relative_to(root.resolve()) for s in lay.sources)


def test_windows_and_backslash_paths_in_yaml_and_lists(tmp_path, monkeypatch):
    from fovea import paths
    root = _ultra(tmp_path, name="win")
    assert layout.entry_path("images\\train", root) == root / "images" / "train"
    mount = tmp_path
    monkeypatch.setattr(paths, "PATH_MAPPINGS", [("D:\\data", str(mount))])
    monkeypatch.setattr(paths, "CONTAINER_MOUNT", str(mount))
    assert layout.entry_path("D:\\data\\win\\images\\val", tmp_path / "nowhere").resolve() == (root / "images" / "val").resolve()
    (root / "data.yaml").write_text("train: images\\train\nval: D:\\data\\win\\images\\val\nnames: [a]\n")
    lay = layout.detect(str(root))
    assert lay.kind == "yaml" and {s.split for s in lay.sources} == {"train", "val"}


def test_a_folder_of_datasets_is_a_collection_not_one_dataset(tmp_path):
    parent = tmp_path / "variants"
    _ultra(parent, name="ultra")
    sf = parent / "splitfirst"
    _img(sf / "train" / "images" / "a.jpg")
    (sf / "train" / "labels").mkdir(parents=True)
    _img(parent / "images-only" / "x.jpg")
    yd = _ultra(parent, name="yamlds")
    (yd / "data.yaml").write_text("train: images/train\nnames: [a]\n")
    lay = layout.detect(str(parent))
    assert lay.kind == "collection" and not lay.sources
    assert {c["name"] for c in lay.candidates} == {"ultra", "splitfirst", "images-only", "yamlds"}
    # one dataset in a sub-folder is still found
    only = tmp_path / "wrap"
    _ultra(only, name="inner")
    lay = layout.detect(str(only))
    assert lay.kind == "ultralytics" and "inner" in lay.notes[0]


def test_labels_only_folders_are_reported(tmp_path):
    root = _ultra(tmp_path)
    for k in range(4):
        _lbl(root / "labels" / "orphan" / f"x{k}.txt", "0 0.5 0.5 0.1 0.1\n")
    lay = layout.detect(str(root))
    assert [u["files"] for u in lay.unmatched_labels] == [4]
    assert any("labels/orphan" in n and "no matching images" in n for n in lay.notes)


def test_registration_never_walks_the_tree_and_the_preview_estimate_is_capped(tmp_path, monkeypatch):
    root = _ultra(tmp_path)
    calls = []
    real = layout.count_images

    def spy(*a, **k):
        calls.append(a)
        return real(*a, **k)
    monkeypatch.setattr(layout, "count_images", spy)
    lay = layout.detect(str(root), estimate=False)
    assert all(k[1:] and k[1] == 1 for k in calls) and lay.sources[0].image_count is None
    monkeypatch.setattr(layout, "ESTIMATE_SECONDS", -1.0)      # budget already spent: counts are lower bounds
    lay = layout.detect(str(root))
    assert all(s.count_capped for s in lay.sources)


def test_slug_and_name_suggestions():
    assert layout.slugify("Factory camera 3") == "factory-camera-3"
    a, b = layout.slugify("한글 데이터셋"), layout.slugify("다른 데이터셋")
    assert a != b and a.startswith("ds-") and a.isascii()
    assert layout.suggest_name(Path("/x/helmets/images")) == "helmets"


# --------------------------------------------------------------------------- scanner

def test_two_folders_of_one_split_do_not_collide(tmp_path):
    root = tmp_path / "cams"
    for cam in ("cam_a", "cam_b"):
        for k in range(3):
            _img(root / "images" / cam / f"{k:04d}.jpg")
            _lbl(root / "labels" / cam / f"{k:04d}.txt", "0 0.5 0.5 0.2 0.2\n")
    (root / "data.yaml").write_text("train: [images/cam_a, images/cam_b]\nnames: [a]\n")
    ds_id = _register(root)
    try:
        res = _scan(ds_id)
        assert res["images"] == 6
        assert _rels(ds_id)[:2] == ["train/cam_a/0000.jpg", "train/cam_a/0001.jpg"]
        assert len({r["abs_path"] for r in db.query("SELECT abs_path FROM images WHERE dataset_id=?", (ds_id,))}) == 6
    finally:
        _drop(ds_id)


def test_folders_without_clashing_names_keep_their_keys_and_reviews_follow_a_rekey(tmp_path):
    root = tmp_path / "cams2"
    for cam, names in (("cam_a", ("a1", "a2")), ("cam_b", ("b1", "b2"))):
        for n in names:
            _img(root / "images" / cam / f"{n}.jpg")
    (root / "data.yaml").write_text("train: [images/cam_a, images/cam_b]\nnames: [a]\n")
    ds_id = _register(root)
    try:
        _scan(ds_id)
        assert _rels(ds_id) == ["train/a1.jpg", "train/a2.jpg", "train/b1.jpg", "train/b2.jpg"]
        db.execute("INSERT INTO reviews(dataset_id, rel_path, status, updated_at) VALUES(?, 'train/a1.jpg', 'approved', 1)", (ds_id,))
        _img(root / "images" / "cam_b" / "a2.jpg")         # now the names clash: the split is re-keyed
        res = _scan(ds_id)
        assert "train/cam_a/a1.jpg" in _rels(ds_id) and len(_rels(ds_id)) == 5 and res["reviews_moved"] == 1
        assert db.query_one("SELECT rel_path FROM reviews WHERE dataset_id=?", (ds_id,))["rel_path"] == "train/cam_a/a1.jpg"
    finally:
        _drop(ds_id)


def test_list_entries_outside_the_base_with_the_same_name_stay_distinct(tmp_path):
    for d in ("a", "b"):
        _img(tmp_path / d / "images" / "0001.jpg")
    lst = tmp_path / "lists" / "train.txt"
    lst.parent.mkdir()
    lst.write_text(f"{tmp_path / 'a' / 'images' / '0001.jpg'}\n{tmp_path / 'b' / 'images' / '0001.jpg'}\n")
    ds_id = _register(lst.parent, raw_path=lst)
    try:
        _scan(ds_id)
        assert len(_rels(ds_id)) == 2
    finally:
        _drop(ds_id)


def test_symlinked_list_entries_keep_the_link_path(tmp_path):
    pool = tmp_path / "pool"
    _img(pool / "p1.jpg")
    root = tmp_path / "ds"
    (root / "images" / "train").mkdir(parents=True)
    os.symlink(pool / "p1.jpg", root / "images" / "train" / "p1.jpg")
    _lbl(root / "labels" / "train" / "p1.txt", "0 0.5 0.5 0.2 0.2\n")
    (root / "train.txt").write_text("./images/train/p1.jpg\n")
    ds_id = _register(root, raw_path=root / "train.txt")
    try:
        _scan(ds_id)
        row = db.query_one("SELECT rel_path, label_path, has_label FROM images WHERE dataset_id=?", (ds_id,))
        assert row["has_label"] == 1 and row["rel_path"] == "train/images/train/p1.jpg"
        assert row["label_path"].endswith("ds/labels/train/p1.txt")
    finally:
        _drop(ds_id)


def test_scan_counts_skipped_list_entries_and_refreshes_label_dir_state(tmp_path):
    root = tmp_path / "bare"
    _img(root / "a.jpg")
    ds_id = _register(root)
    try:
        res = _scan(ds_id)
        lay = db.loads(db.query_one("SELECT layout FROM datasets WHERE id=?", (ds_id,))["layout"], {})
        assert lay["sources"][0]["label_dir_exists"] is False and res["images"] == 1
        _lbl(root / "labels" / "a.txt", "0 0.5 0.5 0.1 0.1\n")
        _scan(ds_id)
        lay = db.loads(db.query_one("SELECT layout FROM datasets WHERE id=?", (ds_id,))["layout"], {})
        assert lay["sources"][0]["label_dir_exists"] is True
    finally:
        _drop(ds_id)
    lst = tmp_path / "l" / "train.txt"
    _img(tmp_path / "l" / "images" / "ok.jpg")
    lst.write_text("images/ok.jpg\nimages/missing.jpg\n")
    ds_id = _register(lst.parent, raw_path=lst)
    try:
        res = _scan(ds_id)
        assert res["skipped"]["missing_list_entry"] == 1 and "1 list entry not found" in res["notes"]
        assert db.kv_get(f"scan:{ds_id}")["skipped"]["missing_list_entry"] == 1
    finally:
        _drop(ds_id)


def test_unchanged_rescan_skips_labels_and_picks_up_changes(scanned):
    ds_id, root = scanned
    res = _scan(ds_id)
    assert res["unchanged"] == 6 and res["indexed"] == 0
    time.sleep(0.01)
    (root / "labels" / "train" / "f00.txt").write_text("1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n")
    res = _scan(ds_id)
    assert res["indexed"] == 1 and res["unchanged"] == 5
    assert db.query_one("SELECT n_boxes FROM images WHERE dataset_id=? AND rel_path='train/f00.jpg'", (ds_id,))["n_boxes"] == 2


def test_scan_never_holds_the_write_lock_across_progress_updates(tmp_path):
    root = _ultra(tmp_path, n=30)
    ds_id = _register(root)
    probes = []

    class Probe(Job):
        def update(self, *a, **k):
            other = sqlite3.connect(str(db.DB_PATH), timeout=0.2)
            try:
                other.execute("BEGIN IMMEDIATE")
                other.execute("ROLLBACK")
                probes.append(True)
            except sqlite3.OperationalError:
                probes.append(False)
            finally:
                other.close()
            return super().update(*a, **k)
    try:
        import fovea.core.scanner as sc
        old = sc._BATCH_ROWS
        sc._BATCH_ROWS = 10
        try:
            scan_dataset(ds_id, Probe(id="probe", kind="scan"))
        finally:
            sc._BATCH_ROWS = old
        assert probes and all(probes)
    finally:
        _drop(ds_id)


def test_cancelled_first_scan_is_partial_with_classes_and_cover(tmp_path):
    root = _ultra(tmp_path, n=30, rows="3 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)

    class Cancelling(Job):
        def update(self, done=None, total=None, message=None, extra=None):
            if done and done >= 20:
                self.cancel()
            return super().update(done, total, message, extra)
    try:
        import fovea.core.scanner as sc
        old = sc._BATCH_ROWS
        sc._BATCH_ROWS = 10
        try:
            with pytest.raises(JobCancelled):
                scan_dataset(ds_id, Cancelling(id="c", kind="scan"))
        finally:
            sc._BATCH_ROWS = old
        row = db.query_one("SELECT status, image_count, classes, cover_image_id, scanned_at FROM datasets WHERE id=?", (ds_id,))
        assert row["status"] == "partial" and 0 < row["image_count"] < 60 and row["scanned_at"] is None
        assert db.loads(row["classes"]) == [f"class_{i}" for i in range(4)] and row["cover_image_id"]
        _scan(ds_id)                                       # a rescan resumes and completes it
        row = db.query_one("SELECT status, image_count FROM datasets WHERE id=?", (ds_id,))
        assert row == {"status": "ready", "image_count": 60}
    finally:
        _drop(ds_id)


def test_class_names_typed_during_the_first_scan_are_kept(tmp_path):
    root = _ultra(tmp_path, n=20, rows="2 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)

    class Renaming(Job):
        def update(self, *a, **k):
            db.execute("UPDATE datasets SET classes=?, classes_source='user' WHERE id=?",
                       (db.dumps(["person", "fall", "wheelchair"]), ds_id))
            return super().update(*a, **k)
    try:
        scan_dataset(ds_id, Renaming(id="r", kind="scan"))
        row = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (ds_id,))
        assert db.loads(row["classes"]) == ["person", "fall", "wheelchair"] and row["classes_source"] == "user"
    finally:
        _drop(ds_id)


def test_sequence_keys_use_the_folder_and_drop_frame_numbers():
    assert seq_key("MOT17-02/img1/000001.jpg") == "MOT17-02/img1"
    assert seq_key("MOT17-04/img1/000001.jpg") == "MOT17-04/img1"
    assert seq_key("frame_0001.jpg") == seq_key("frame_0002.jpg")
    assert seq_key("camA/clip_001.jpg") != seq_key("camB/clip_001.jpg")
    assert seq_key("images_GOPR0804_00006.jpg") == "images_GOPR0804"


def test_post_scan_hooks_run_after_a_successful_scan_only(tmp_path):
    root = _ultra(tmp_path)
    ds_id = _register(root)
    seen = []
    saved = POST_SCAN_HOOKS[:]
    POST_SCAN_HOOKS[:] = [seen.append]
    try:
        scan_dataset(ds_id, Job(id="h1", kind="scan"))
        assert seen == [ds_id]
        POST_SCAN_HOOKS.insert(0, lambda _d: 1 / 0)       # a broken hook never fails the scan or later hooks
        scan_dataset(ds_id, Job(id="h2", kind="scan"))
        assert seen == [ds_id, ds_id]
    finally:
        POST_SCAN_HOOKS[:] = saved
        _drop(ds_id)


# --------------------------------------------------------------------------- A10 Unicode

def test_decomposed_hangul_names_match_nfc_search_split_and_files(tmp_path):
    from fovea.api.datasets import list_files
    nfd = unicodedata.normalize("NFD", "test-포함")
    root = tmp_path / "kr"
    _img(root / "images" / nfd / "a.jpg")
    _lbl(root / "labels" / nfd / "a.txt", "0 0.5 0.5 0.2 0.2\n")
    _img(root / "images" / "train" / "b.jpg")
    ds_id = _register(root)
    try:
        _scan(ds_id)
        assert imagesq.count_images(ds_id, {"q": "포함"}) == 1
        assert imagesq.count_images(ds_id, {"split": "test-포함"}) == 1
        res = list_files(ds_id, path="images/test-포함", offset=0, limit=100)
        assert res["entries"][0]["image_id"] is not None
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- queries (§2, A11, A12, F16, G42)

def _seed_ai(ds_id, iid_pending, iid_audit, iid_flag):
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'batch', ?)", (ds_id, db.now())).lastrowid
    db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind) VALUES(?,?,?,0,.5,.5,.1,.1,0.3,'detect')",
               (run, ds_id, iid_pending))
    db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, ref_row) VALUES(?,?,?,0,.5,.5,.1,.1,0.9,'swap',0)",
               (run, ds_id, iid_audit))
    db.execute("INSERT INTO ai_audit(image_id, dataset_id, run_id, score) VALUES(?,?,?,0.8)", (iid_audit, ds_id, run))
    db.execute("INSERT INTO image_flags(image_id, dataset_id, code, detail) VALUES(?,?, 'near_dup', '{}')", (iid_flag, ds_id))


def test_image_query_contract_for_ai_and_qa(scanned):
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY id", (ds_id,))]
    _seed_ai(ds_id, ids[0], ids[1], ids[2])
    count = lambda **f: imagesq.count_images(ds_id, f)  # noqa: E731
    assert count(issue="near_dup") == 1 and count(issue="ai_swap") == 1 and count(issue="ai_missing") == 0
    assert count(issue="near_dup,ai_swap") == 2 and count(issue="any") == 1
    assert count(ai="pending") == 1 and count(ai="audit") == 1 and count(ai="none") == 4
    _t, items = imagesq.list_images(ds_id, {}, 0, 10)
    by = {it["id"]: it for it in items}
    assert (by[ids[0]]["ai_pending"], by[ids[1]]["ai_suspect"], by[ids[2]]["flags"]) == (1, 1, ["near_dup"])
    assert imagesq.list_ids(ds_id, {}, "audit")[0] == ids[1]            # descending by default
    assert imagesq.list_ids(ds_id, {}, "ai_conf")[0] == ids[0]
    for sort in ("audit", "ai_conf"):
        order = imagesq.list_ids(ds_id, {}, sort)
        for pos, iid in enumerate(order):
            assert imagesq.position(ds_id, {}, iid, sort) == pos
            nxt = imagesq.neighbor(ds_id, {}, iid, "next", sort)
            assert (nxt["id"] if nxt else None) == (order[pos + 1] if pos + 1 < len(order) else None)
    assert imagesq.neighbor(ds_id, {}, ids[0], "next", need="suspect")["id"] == ids[1]


def test_keyset_paging_seeded_random_and_frozen_iteration(scanned):
    ds_id, _root = scanned
    full = imagesq.list_ids(ds_id, {}, "random", None)
    assert sorted(full) == sorted(set(full)) and len(full) == 6
    a, b = imagesq.list_ids(ds_id, {"seed": "7"}, "random"), imagesq.list_ids(ds_id, {"seed": "8"}, "random")
    assert sorted(a) == sorted(full) and a != b
    total, first = imagesq.list_images(ds_id, {"seed": "7"}, 0, 3, "random")
    rest_total, rest = imagesq.list_images(ds_id, {"seed": "7"}, 0, 3, "random", after_id=first[-1]["id"], with_total=False)
    assert total == 6 and rest_total is None and [i["id"] for i in first + rest] == a
    # iterating keeps its id snapshot even when the filter set changes underneath it
    it = imagesq.iter_images(ds_id, {"review": "none"}, chunk=2)
    got = [next(it)["id"]]
    db.execute("INSERT INTO reviews(dataset_id, rel_path, status, updated_at) SELECT dataset_id, rel_path, 'approved', 1 "
               "FROM images WHERE dataset_id=? AND id != ?", (ds_id, got[0]))
    got += [x["id"] for x in it]
    assert len(got) == 6


def test_labeled_file_filter_matches_the_overview_tile(scanned):
    ds_id, root = scanned
    (root / "labels" / "train" / "f00.txt").write_text("")
    _scan(ds_id)
    assert imagesq.count_images(ds_id, {"labeled": "file"}) == 6
    assert imagesq.count_images(ds_id, {"labeled": "labeled"}) == 5
    f = imagesq.facets(ds_id, {"split": "train"})
    assert f["labeled"]["file"] == 3 and f["split"] == {"train": 3, "val": 3} and f["labeled"]["empty"] == 1


def test_unusable_filter_values_match_nothing_never_everything():
    """build_where is also called directly (export, Objects, Compare): a value it cannot read matches NOTHING —
    every bulk writer resolves its images through it, so a dropped filter would act on the whole dataset."""
    closed = ("i.dataset_id = ? AND 0", ["x"])
    for f in ({"cls": "²,a,1"}, {"only_cls": "bogus"}, {"ids": "1,a"}, {"ids": "-1"}, {"ids": []}, {"ids": ","},
              {"ids": [[1]]}, {"q": {"x": 1}}, {"seq": {"a": 1}}, {"labeled": "Yes"}, {"review": "bogus"},
              {"review": "approved,bogus"}, {"ai": "bogus"}, {"ai": "pending,audit"}, {"exclude_review": "bogus"},
              {"min_boxes": "2.5"}, {"max_boxes": "abc"}, {"has_label": "maybe"}, {"split": "train", "cls": "x"}):
        assert imagesq.build_where("x", f) == closed, f
        assert imagesq.filter_problem(*next((k, v) for k, v in f.items() if k != "split")), f
    # Values it can read, in any case and spacing; blank values and empty lists (other than ids) are no filter.
    assert imagesq.build_where("x", {"seq": ["a"], "ids": " 1, 2 "})[1] == ["x", "a", "[1, 2]"]
    assert imagesq.build_where("x", {"review": " Approved ", "ai": "PENDING", "labeled": True}) == \
        imagesq.build_where("x", {"review": "approved", "ai": "pending", "labeled": "1"})
    assert imagesq.build_where("x", {"cls": [], "split": "", "q": "  ", "min_boxes": None}) == ("i.dataset_id = ?", ["x"])
    assert imagesq.build_where("x", {"cls": str(10 ** 30)})[1][-1] == (1 << 63) - 1      # clamped: matches nothing


# --------------------------------------------------------------------------- stats (A13, A14, A26, A34, M7)

def test_stats_are_cached_until_a_label_changes_and_aspect_is_in_pixels(tmp_path):
    from fovea.api.datasets import put_labels
    root = tmp_path / "wide"
    _img(root / "images" / "a.jpg", size=(1920, 1080))
    # a 200x200 px box on a 16:9 frame: w=0.104, h=0.185 — square, not "0.5-0.8"
    _lbl(root / "labels" / "a.txt", "0 0.5 0.5 0.104167 0.185185\n")
    _img(root / "images" / "b.jpg")
    ds_id = _register(root, classes=["sq", "never"], source="user")
    try:
        _scan(ds_id)
        s = stats.dataset_stats(ds_id, ["sq", "never"])
        assert [b["count"] for b in s["box_aspect"]] == [0, 0, 1, 0, 0]
        assert s["empty_classes"] == [1] and s["classes"][1]["empty"] and s["classes"][1]["under_represented"]
        assert s["totals"]["with_issues"] == 0                                 # b.jpg has no label: progress, not a problem
        assert [i["severity"] for i in s["issues"] if i["code"] == "missing_label"] == ["info"]
        assert s["with_problems"] == 0 == imagesq.count_images(ds_id, {"issue": "any"})
        assert imagesq.count_images(ds_id, {"issue": "missing_label"}) == 1                # still filterable
        assert "near_dup" in s["issue_labels"] and stats.dataset_stats(ds_id, ["sq", "never"])["cached"]
        put_labels(ds_id, _iid(ds_id, "a.jpg"), {"boxes": []})
        s2 = stats.dataset_stats(ds_id, ["sq", "never"])
        assert not s2["cached"] and s2["totals"]["boxes"] == 0
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- API: labels, bulk, review

def test_label_put_validation_and_readonly_folder(scanned):
    from fovea.api.datasets import put_labels
    ds_id, root = scanned
    iid = _iid(ds_id, "train/f00.jpg")
    for body in ({}, {"boxes": "x"}, {"boxes": [["a"]]}):
        with pytest.raises(HTTPException) as e:
            put_labels(ds_id, iid, body)
        assert e.value.status_code == 400
    assert (root / "labels" / "train" / "f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    if os.geteuid() != 0:
        os.chmod(root / "labels" / "train", 0o555)
        try:
            with pytest.raises(HTTPException) as e:
                put_labels(ds_id, iid, {"boxes": [[1, 0.5, 0.5, 0.2, 0.2]]})
            assert e.value.status_code == 403 and ".fovea-" not in e.value.detail and "f00.txt" in e.value.detail
        finally:
            os.chmod(root / "labels" / "train", 0o755)


def test_bulk_ops_validate_snapshot_and_count_correctly(scanned):
    from fovea.api.datasets import bulk_labels
    ds_id, root = scanned
    ids = [_iid(ds_id, "train/f00.jpg")]
    (root / "labels" / "train" / "f00.txt").write_text("1 0.1 0.1 0.1 0.1\n0 0.5 0.5 0.2 0.2\n0 0.7 0.7 0.2 0.2\n0 0.3 0.3 0.2 0.2\n")
    for body in ({"op": "set_class"}, {"op": "set_class", "cls": "x"}, {"op": "set_class", "cls": -3},
                 {"op": "remap", "mapping": {"a": 1}}, {"op": "remap", "mapping": {}}, {"op": "nope"},
                 {"op": "delete_class", "cls": 1, "range": {"from_id": "a", "to_id": 3}}):
        with pytest.raises(HTTPException) as e:
            bulk_labels(ds_id, {**body, "image_ids": ids})
        assert e.value.status_code == 400, body
    res = bulk_labels(ds_id, {"op": "delete_class", "cls": 1, "image_ids": ids})
    assert res["changed"] == 1 and res["snapshot_id"]
    res = bulk_labels(ds_id, {"op": "set_class", "cls": 2, "image_ids": ids})
    assert res["changed"] == 3
    diff = snapshots.diff(ds_id, res["snapshot_id"])
    assert diff["summary"]["per_class"] == {"0": {"added": 0, "removed": 3}, "2": {"added": 3, "removed": 0}}


def test_clearing_a_review_keeps_the_note_and_notes_update_alone(scanned):
    from fovea.api.datasets import set_review
    ds_id, _root = scanned
    iid = _iid(ds_id, "train/f01.jpg")
    set_review(ds_id, {"image_ids": [iid], "status": "approved", "note": "partly occluded"})
    set_review(ds_id, {"image_ids": [iid], "status": None})
    item = imagesq.items_by_ids(ds_id, [iid])[0]
    assert item["review"] is None and item["note"] == "partly occluded"
    assert imagesq.count_images(ds_id, {"review": "none", "ids": str(iid)}) == 1
    set_review(ds_id, {"image_ids": [iid], "status": "flagged"})
    assert imagesq.items_by_ids(ds_id, [iid])[0]["review"]["note"] == "partly occluded"
    set_review(ds_id, {"image_ids": [iid], "note": "fixed"})               # note only: status stays
    item = imagesq.items_by_ids(ds_id, [iid])[0]
    assert item["review"]["status"] == "flagged" and item["note"] == "fixed"
    set_review(ds_id, {"image_ids": [iid], "status": None, "note": ""})    # nothing left: row removed
    assert not db.query_one("SELECT 1 FROM reviews WHERE dataset_id=?", (ds_id,))
    assert set_review(ds_id, {"rel_paths": ["does/not/exist.jpg", 5], "status": "approved"})["updated"] == 0
    assert set_review(ds_id, {"image_ids": [str(iid)], "status": "approved"})["updated"] == 1
    with pytest.raises(HTTPException):
        set_review(ds_id, {"image_ids": [iid]})


def test_class_list_edits_revalidate_bad_class(scanned):
    from fovea.api.datasets import update_dataset
    ds_id, root = scanned
    (root / "labels" / "train" / "f00.txt").write_text("2 0.5 0.5 0.2 0.2\n")
    _scan(ds_id)
    update_dataset(ds_id, {"classes": ["person"]})
    assert imagesq.count_images(ds_id, {"issue": "bad_class"}) == 1
    update_dataset(ds_id, {"classes": ["a", "b", "c"]})
    assert imagesq.count_images(ds_id, {"issue": "bad_class"}) == 0


def test_dataset_registration_validates_and_refuses_duplicates(tmp_path):
    from fovea.api.datasets import create_dataset, delete_dataset
    root = _ultra(tmp_path, name="한글 데이터셋")
    for body in ({"path": 123}, {"path": str(root), "name": 5}, {"id": 7}, [1]):
        with pytest.raises(HTTPException) as e:
            create_dataset(body)
        assert e.value.status_code == 400
    first = create_dataset({"path": str(root)})["dataset"]
    try:
        assert first["name"] == "한글 데이터셋" and first["id"].startswith("ds-")
        with pytest.raises(HTTPException) as e:
            create_dataset({"path": str(root)})
        assert e.value.status_code == 409
        second = create_dataset({"path": str(root), "allow_duplicate": True})["dataset"]
        delete_dataset(second["id"])
    finally:
        delete_dataset(first["id"])
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nnames: [a]\n")
    ds = create_dataset({"path": str(root / "data.yaml")})["dataset"]
    try:
        assert ds["name"] == "한글 데이터셋" and ds["classes_source"] == "yaml"
    finally:
        delete_dataset(ds["id"])


def test_files_listing_pages_and_previews(tmp_path):
    from fovea.api.datasets import get_file, list_files
    root = _ultra(tmp_path, n=12)
    (root / "notes.txt").write_text("line one\n" * 10)
    (root / "empty.txt").write_text("")
    (root / "bin.txt").write_bytes(b"\x00\x01\x02abc")
    (root / "kr.csv").write_bytes("이름,값\n".encode("cp949"))
    import numpy as np
    Image.fromarray(np.zeros((8, 8), dtype="uint16")).save(root / "scan.tif")
    ds_id = _register(root)
    try:
        _scan(ds_id)
        res = list_files(ds_id, path="images/train", offset=5, limit=4)
        assert res["total"] == 12 and len(res["entries"]) == 4 and res["truncated"]
        assert all(e["image_id"] for e in res["entries"])
        assert get_file(ds_id, path="empty.txt", raw=0, offset=0, limit=100)["empty"]
        assert get_file(ds_id, path="bin.txt", raw=0, offset=0, limit=100)["binary"]
        kr = get_file(ds_id, path="kr.csv", raw=0, offset=0, limit=100)
        assert kr["encoding"] == "cp949" and kr["text"].startswith("이름")
        part = get_file(ds_id, path="notes.txt", raw=0, offset=0, limit=25)
        assert part["text"] == "line one\nline one\n" and part["next_offset"] == 18 and part["truncated"]
        tif = get_file(ds_id, path="scan.tif", raw=0, offset=0, limit=10)
        assert tif.media_type == "image/jpeg"
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- snapshots (§3.1, A27)

def test_snapshot_diff_restore_is_undoable_and_retention(scanned):
    from fovea.api.datasets import bulk_labels
    ds_id, root = scanned
    f00, f01 = root / "labels" / "train" / "f00.txt", root / "labels" / "train" / "f01.txt"
    before = f00.read_text()
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? AND split='train'", (ds_id,))]
    sid = bulk_labels(ds_id, {"op": "clear", "image_ids": ids})["snapshot_id"]
    assert f00.read_text() == ""
    d = snapshots.diff(ds_id, sid)
    assert d["summary"]["images_changed"] == 3 and d["summary"]["boxes_removed"] == 3 and d["items"][0]["before"]
    # a file created after the snapshot: absent in the snapshot -> removed by a restore of a full snapshot
    new_img = root / "images" / "val" / "new.jpg"
    _img(new_img)
    _scan(ds_id)
    full = snapshots.snapshot_dataset(ds_id, name="v1")           # records new.jpg's label as absent
    res = snapshots.restore(ds_id, sid)
    assert f00.read_text() == before and res["restored"] == 3 and res["snapshot_id"]
    assert snapshots.diff(ds_id, sid)["summary"]["images_changed"] == 0
    assert db.query_one("SELECT n_boxes FROM images WHERE dataset_id=? AND rel_path='train/f00.jpg'", (ds_id,))["n_boxes"] == 1
    undo = snapshots.restore(ds_id, res["snapshot_id"])                # the restore itself is undoable
    assert f01.read_text() == "" and undo["restored"] == 3
    snapshots.restore(ds_id, sid, image_ids=[_iid(ds_id, "train/f01.jpg")])
    assert f01.read_text() == before and f00.read_text() == ""
    (root / "labels" / "val" / "new.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    snapshots.restore(ds_id, full)
    assert not (root / "labels" / "val" / "new.txt").exists() and f00.read_text() == ""   # the state at 'v1'
    for _ in range(snapshots.KEEP_SMALL + 3):
        snapshots.snapshot_images(ds_id, ids, "test")                 # 3 files: small versions
    assert snapshots.settle()                                          # retention runs on a background thread
    kinds = [s["kind"] for s in snapshots.list_snapshots(ds_id)]
    assert kinds.count("auto") == snapshots.KEEP_SMALL and kinds.count("manual") == 1
    held = {r["sha"] for r in db.query("SELECT sha FROM snapshot_files WHERE snapshot_id=? AND sha IS NOT NULL", (full,))}
    assert held
    snapshots.delete_snapshot(ds_id, full)
    assert not snapshots.get_snapshot(ds_id, full)
    orphan = db.query("SELECT sha FROM label_blobs b WHERE NOT EXISTS (SELECT 1 FROM snapshot_files f WHERE f.sha=b.sha)")
    assert not held & {r["sha"] for r in orphan}            # the dropped version's blobs went with it (or are shared)


# --------------------------------------------------------------------------- class cards (§3.2)

def test_class_cards(scanned):
    from fovea.api.datasets import class_cards, get_class_cards, put_class_card
    ds_id, _root = scanned
    db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(["person", "fall"]), ds_id))
    iid = _iid(ds_id, "train/f00.jpg")
    card = put_class_card(ds_id, 1, {"prompt": "person lying on the ground", "include": "partly visible",
                                     "examples": [{"image_id": iid, "row": 0, "polarity": -1}]})["card"]
    assert card["name"] == "fall" and card["examples"] == [{"image_id": iid, "row": 0, "polarity": -1}]
    card = put_class_card(ds_id, 1, {"description": "on the floor"})["card"]
    assert card["prompt"] == "person lying on the ground" and card["description"] == "on the floor"
    cards = get_class_cards(ds_id)["cards"]
    assert set(cards) == {"0", "1"} and cards["0"]["prompt"] == ""
    assert class_cards(ds_id)[1]["include"] == "partly visible"
    for cls, body in ((9, {"prompt": "x"}), (0, {"prompt": 5}), (0, {"examples": [{"image_id": 10 ** 9}]})):
        with pytest.raises(HTTPException):
            put_class_card(ds_id, cls, body)


# --------------------------------------------------------------------------- A33 migrations

OLD_SCHEMA = """
CREATE TABLE datasets (id TEXT PRIMARY KEY, name TEXT NOT NULL, root TEXT NOT NULL, layout TEXT NOT NULL DEFAULT '{}',
  classes TEXT NOT NULL DEFAULT '[]', classes_source TEXT NOT NULL DEFAULT 'inferred', description TEXT NOT NULL DEFAULT '',
  status TEXT NOT NULL DEFAULT 'new', image_count INTEGER NOT NULL DEFAULT 0, label_count INTEGER NOT NULL DEFAULT 0,
  box_count INTEGER NOT NULL DEFAULT 0, cover_image_id INTEGER, created_at REAL, updated_at REAL, scanned_at REAL, opened_at REAL);
CREATE TABLE images (id INTEGER PRIMARY KEY, dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE,
  rel_path TEXT NOT NULL, split TEXT NOT NULL DEFAULT '', abs_path TEXT NOT NULL, label_path TEXT,
  width INTEGER, height INTEGER, size INTEGER, mtime REAL, lmtime REAL, has_label INTEGER NOT NULL DEFAULT 0,
  n_boxes INTEGER NOT NULL DEFAULT 0, classes TEXT NOT NULL DEFAULT '[]', issues TEXT NOT NULL DEFAULT '[]',
  seq TEXT NOT NULL DEFAULT '', scan_token TEXT NOT NULL DEFAULT '', UNIQUE(dataset_id, rel_path));
CREATE TABLE boxes (id INTEGER PRIMARY KEY, image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL, cls INTEGER NOT NULL, xc REAL NOT NULL, yc REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL);
CREATE TABLE reviews (dataset_id TEXT NOT NULL REFERENCES datasets(id) ON DELETE CASCADE, rel_path TEXT NOT NULL,
  status TEXT NOT NULL, note TEXT NOT NULL DEFAULT '', updated_at REAL, PRIMARY KEY(dataset_id, rel_path));
"""


def test_old_databases_are_migrated_in_place(tmp_path):
    path = tmp_path / "old.db"
    c = sqlite3.connect(path)
    c.executescript(OLD_SCHEMA)
    nfd = unicodedata.normalize("NFD", "포함/a.jpg")
    c.execute("INSERT INTO datasets(id,name,root,status,image_count) VALUES('d','d','/x','scanning',2)")
    c.execute("INSERT INTO images(id,dataset_id,rel_path,abs_path) VALUES(5,'d',?,'/x/a.jpg')", (nfd,))
    c.execute("INSERT INTO images(id,dataset_id,rel_path,abs_path) VALUES(9,'d','b.jpg','/x/b.jpg')")
    c.execute("INSERT INTO boxes(image_id,dataset_id,cls,xc,yc,w,h) VALUES(5,'d',0,.5,.5,.1,.1)")
    c.execute("INSERT INTO reviews(dataset_id,rel_path,status,note) VALUES('d',?,'approved','hi')", (nfd,))
    c.commit()
    c.executescript(db.SCHEMA)
    applied = db.migrate(c)
    assert applied == [name for name, _s, _f in db.MIGRATIONS]
    assert c.execute("PRAGMA user_version").fetchone()[0] == db.SCHEMA_VERSION
    assert c.execute("SELECT rel_path FROM images WHERE id=5").fetchone()[0] == "포함/a.jpg"
    assert c.execute("SELECT rel_path, note FROM reviews").fetchone() == ("포함/a.jpg", "hi")
    assert c.execute("SELECT COUNT(*) FROM boxes WHERE image_id=5").fetchone()[0] == 1
    c.execute("PRAGMA foreign_keys=ON")
    c.execute("DELETE FROM images WHERE id=9")
    c.execute("INSERT INTO images(dataset_id,rel_path,abs_path) VALUES('d','c.jpg','/x/c.jpg')")
    assert c.execute("SELECT MAX(id) FROM images").fetchone()[0] == 10          # ids are never reused
    c.execute("INSERT INTO reviews(dataset_id,rel_path,status,note) VALUES('d','z',NULL,'note only')")
    assert db.migrate(c) == []                                                   # idempotent
    c.close()


# --------------------------------------------------------------------------- routing

def test_list_ids_and_facets_routes_are_not_shadowed(scanned):
    try:
        from fastapi.testclient import TestClient
        from fovea.main import app
    except Exception as e:  # noqa: BLE001 — another role's module failing to import is not this test's subject
        pytest.skip(f"app import failed: {e}")
    ds_id, _root = scanned
    c = TestClient(app)
    r = c.get(f"/api/datasets/{ds_id}/images/ids", params={"sort": "name"})
    assert r.status_code == 200 and r.json()["total"] == 6
    r = c.get(f"/api/datasets/{ds_id}/images/facets")
    assert r.status_code == 200 and r.json()["facets"]["total"] == 6
    r = c.post(f"/api/datasets/{ds_id}/images/by-ids", json={"ids": r.json() and [1, 2]})
    assert r.status_code == 200
    r = c.get(f"/api/datasets/{ds_id}/classes/counts")
    assert r.json()["counts"] == {"0": 6}
    r = c.get(f"/api/datasets/{ds_id}/stats")
    assert "ai_swap" in r.json()["issue_labels"] and r.json()["stats"]["issue_labels"]["near_dup"]
    r = c.get(f"/api/datasets/{ds_id}/snapshots")
    assert r.status_code == 200 and r.json()["snapshots"] == []
    r = c.post(f"/api/datasets/{ds_id}/snapshots", json={"name": "v1"})
    assert r.status_code == 200 and r.json().get("snapshot_id")


# =========================================================================== wave 2a

# --------------------------------------------------------------------------- label-write hooks

def test_label_write_hooks_fire_after_put_bulk_and_restore_and_never_fail_a_write(scanned):
    from fovea.api.datasets import bulk_labels, get_labels, put_labels
    ds_id, root = scanned
    calls = []

    def boom(ds, ids):
        raise RuntimeError("hook bug")

    def spy(ds, ids):
        # runs after the index refresh: the new box count is already visible
        calls.append((ds, list(ids), [(db.query_one("SELECT n_boxes FROM images WHERE id=?", (i,)) or {}).get("n_boxes")
                                      for i in ids]))

    labels.LABEL_WRITE_HOOKS[:] = [boom, spy]
    try:
        a, b = _iid(ds_id, "train/f00.jpg"), _iid(ds_id, "train/f01.jpg")
        got = get_labels(ds_id, a)
        put_labels(ds_id, a, {"boxes": got["boxes"] + [[0, 0.2, 0.2, 0.1, 0.1]], "base_version": got["version"]})
        assert calls == [(ds_id, [a], [2])]                       # the raising hook did not stop the save or the spy
        res = bulk_labels(ds_id, {"op": "clear", "image_ids": [a, b, a]})
        assert calls[-1][:2] == (ds_id, [a, b]) and calls[-1][2] == [0, 0]
        snapshots.restore(ds_id, res["snapshot_id"])
        assert calls[-1][0] == ds_id and sorted(zip(calls[-1][1], calls[-1][2])) == sorted([(a, 2), (b, 1)])
        n = len(calls)
        v0 = _iid(ds_id, "val/f00.jpg")
        bulk_labels(ds_id, {"op": "clear", "image_ids": [v0]})
        bulk_labels(ds_id, {"op": "clear", "image_ids": [v0]})          # nothing left to change: nobody is told
        assert len(calls) == n + 1 and calls[-1][1] == [v0]
        assert labels.notify_label_write(ds_id, [None, str(v0), v0, True, "x"]) == [v0]   # junk dropped, de-duplicated
        assert len(calls) == n + 2 and calls[-1][1] == [v0]
        assert labels.notify_label_write("", [v0]) == [v0] and len(calls) == n + 2        # no dataset: nobody is told
        assert labels.notify_label_write(ds_id, []) == [] and len(calls) == n + 2
    finally:
        labels.LABEL_WRITE_HOOKS.clear()


# --------------------------------------------------------------------------- working sets

def test_working_sets_freeze_order_and_apply_to_every_query(scanned):
    ds_id, _root = scanned
    ids = imagesq.list_ids(ds_id, {}, "name")
    order = [ids[4], ids[0], ids[2], ids[5]]
    ws = imagesq.create_set(ds_id, order + [ids[0], 999999999, "x"])           # duplicates / foreign / junk dropped
    tok = ws["token"]
    assert ws["count"] == 4 and ws["expires_at"] > time.time() + 23 * 3600
    f = {"set": tok}
    assert imagesq.list_ids(ds_id, f) == order                                  # stored order by default
    assert [it["id"] for it in imagesq.list_images(ds_id, f, 0, 10)[1]] == order
    assert imagesq.list_ids(ds_id, f, "name") == [i for i in ids if i in order]  # any other sort applies inside it
    assert imagesq.list_ids(ds_id, f, "set", "desc") == order[::-1]
    assert imagesq.count_images(ds_id, f) == 4
    assert imagesq.count_images(ds_id, {**f, "split": "val"}) == len([i for i in order if i in ids[3:]])
    for pos, iid in enumerate(order):
        assert imagesq.position(ds_id, f, iid) == pos
        nxt = imagesq.neighbor(ds_id, f, iid, "next")
        assert (nxt["id"] if nxt else None) == (order[pos + 1] if pos + 1 < 4 else None)
        prv = imagesq.neighbor(ds_id, f, iid, "prev")
        assert (prv["id"] if prv else None) == (order[pos - 1] if pos else None)
    assert imagesq.position(ds_id, f, ids[1]) is None
    assert imagesq.neighbor(ds_id, f, ids[1], "next")["id"] == order[0]          # from outside: the first one
    assert [it["id"] for it in imagesq.iter_images(ds_id, f, chunk=3)] == order  # exports / long jobs
    assert imagesq.facets(ds_id, f)["total"] == 4 and sum(imagesq.facets(ds_id, f)["split"].values()) == 4
    # from filters + sort, and ids restricted by filters
    ws2 = imagesq.create_set(ds_id, filters={"split": "val"}, sort="name", order="desc")
    assert imagesq.list_ids(ds_id, {"set": ws2["token"]}) == ids[3:][::-1]
    ws3 = imagesq.create_set(ds_id, ids=order, filters={"split": "train"})
    assert imagesq.list_ids(ds_id, {"set": ws3["token"]}) == [i for i in order if i in ids[:3]]
    assert imagesq.create_set(ds_id)["count"] == 6                               # nothing given: the whole dataset
    # a set of another dataset, an unknown or an expired token match nothing (never everything)
    assert imagesq.count_images("other-ds", {"set": tok}) == 0
    assert imagesq.count_images(ds_id, {"set": "nope-unknown"}) == 0
    sid2 = db.query_one("SELECT id FROM work_sets WHERE token=?", (ws2["token"],))["id"]
    db.execute("UPDATE work_sets SET created_at = created_at - 90000, used_at = used_at - 90000 WHERE id=?", (sid2,))
    imagesq._set_use.pop(sid2, None)                  # last used 25 h ago (also in this process)
    assert imagesq.set_info(ws2["token"], ds_id) is None and imagesq.count_images(ds_id, {"set": ws2["token"]}) == 0
    imagesq.create_set(ds_id, [ids[0]])                                          # creating one expires the old
    assert db.query_one("SELECT 1 FROM work_sets WHERE token=?", (ws2["token"],)) is None
    assert imagesq.delete_set(tok, ds_id) and imagesq.set_info(tok) is None
    assert db.query_one("SELECT COUNT(*) c FROM work_set_items w WHERE NOT EXISTS (SELECT 1 FROM work_sets s WHERE s.id = w.set_id)")["c"] == 0


def test_working_set_limits_and_http_contract(scanned, monkeypatch):
    try:
        from fastapi.testclient import TestClient
        from fovea.main import app
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"app import failed: {e}")
    from fovea.api.common import filters_from
    ds_id, _root = scanned
    c = TestClient(app)
    ids = imagesq.list_ids(ds_id, {}, "name")
    r = c.post(f"/api/datasets/{ds_id}/sets", json={"ids": [ids[3], ids[1]]})
    assert r.status_code == 200 and r.json()["count"] == 2
    tok = r.json()["token"]
    r = c.get(f"/api/datasets/{ds_id}/images/ids", params={"set": tok})
    assert r.json()["ids"] == [ids[3], ids[1]] and r.json()["filters"]["set"] == tok
    r = c.get(f"/api/datasets/{ds_id}/images", params={"set": tok, "sort": "name"})
    assert [it["id"] for it in r.json()["items"]] == [ids[1], ids[3]]
    assert c.get(f"/api/datasets/{ds_id}/images/position", params={"set": tok, "id": ids[1]}).json()["position"] == 1
    assert c.get(f"/api/datasets/{ds_id}/images/neighbor", params={"set": tok, "id": ids[3]}).json()["item"]["id"] == ids[1]
    assert c.get(f"/api/datasets/{ds_id}/images/facets", params={"set": tok}).json()["facets"]["total"] == 2
    assert c.get(f"/api/datasets/{ds_id}/sets/{tok}").json()["count"] == 2
    r = c.post(f"/api/datasets/{ds_id}/sets", json={"filters": {"split": "val"}, "sort": "name"})
    assert r.json()["count"] == 3
    # expired / unknown -> 410 with a code, never an empty list without explanation
    for url, params in ((f"/api/datasets/{ds_id}/images", {"set": "deadbeef0000"}),
                        (f"/api/datasets/{ds_id}/images/ids", {"set": "x"}),
                        (f"/api/datasets/{ds_id}/sets/deadbeef0000", {})):
        r = c.get(url, params=params)
        assert r.status_code == 410 and r.json()["detail"]["code"] == "set_expired", (url, r.text)
    with pytest.raises(HTTPException) as e:
        filters_from({"set": "deadbeef0000", "split": "val"})
    assert e.value.status_code == 410
    assert filters_from({"set": tok})["set"] == tok
    # bulk ops and reviews take a set through their filters
    r = c.put(f"/api/datasets/{ds_id}/review", json={"status": "flagged", "filters": {"set": tok}})
    assert r.json()["updated"] == 2
    # limits
    monkeypatch.setattr(imagesq, "SET_MAX", 3)
    r = c.post(f"/api/datasets/{ds_id}/sets", json={"filters": {}})
    assert r.status_code == 413 and r.json()["detail"]["code"] == "set_too_large"
    r = c.post(f"/api/datasets/{ds_id}/sets", json={"ids": ids})
    assert r.status_code == 413
    assert c.post(f"/api/datasets/{ds_id}/sets", json={"ids": "1,2"}).status_code == 400
    assert c.delete(f"/api/datasets/{ds_id}/sets/{tok}").json()["deleted"] is True


# --------------------------------------------------------------------------- facets under filters

def _facets_brute(ds_id, f):
    """Reference: each facet value counted with count_images under the other filters."""
    def cnt(key, extra):
        g = {k: v for k, v in f.items() if k != key}
        g.update(extra)
        return imagesq.count_images(ds_id, g)
    split = {s: cnt("split", {"split": s}) for s in ("train", "val")}
    review = {s: cnt("review", {"review": s}) for s in ("approved", "flagged", "none", "any")}
    labeled = {s: cnt("labeled", {"labeled": s}) for s in ("labeled", "unlabeled", "empty", "nobox", "file")}
    cls = {c: cnt("cls", {"cls": str(c)}) for c in (0, 1, 2)}
    issue = {c: cnt("issue", {"issue": c}) for c in ("near_dup", "blurry", "ai_swap", "ai_missing", "any", "tiny_box")}
    ai = {s: cnt("ai", {"ai": s}) for s in ("pending", "audit", "any", "none")}
    return {"split": split, "review": review, "labeled": labeled, "cls": cls, "issue": issue, "ai": ai}


def test_facets_count_each_facet_under_the_other_filters_and_cache_by_version(tmp_path):
    root = tmp_path / "fac"
    rows = {"train": ["0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n", "1 0.5 0.5 0.2 0.2\n", "", None, "2 0.5 0.5 0.001 0.001\n"],
            "val": ["0 0.5 0.5 0.2 0.2\n", "2 0.5 0.5 0.2 0.2\n", None]}
    for split, texts in rows.items():
        for k, t in enumerate(texts):
            _img(root / "images" / split / f"i{k}.jpg")
            if t is not None:
                _lbl(root / "labels" / split / f"i{k}.txt", t)
    ds_id = _register(root)
    try:
        _scan(ds_id)
        ids = imagesq.list_ids(ds_id, {}, "name")          # train i0..i4, val i0..i2
        run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'audit', ?)", (ds_id, db.now())).lastrowid
        for iid, kind in ((ids[0], "detect"), (ids[0], "swap"), (ids[1], "missing"), (ids[5], "swap"), (ids[6], "detect")):
            db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind) VALUES(?,?,?,0,.5,.5,.1,.1,?)",
                       (run, ds_id, iid, kind))
        for iid, code in ((ids[0], "near_dup"), (ids[2], "near_dup"), (ids[3], "blurry"), (ids[6], "near_dup")):
            db.execute("INSERT INTO image_flags(image_id, dataset_id, code) VALUES(?,?,?)", (iid, ds_id, code))
        from fovea.api.datasets import set_review
        set_review(ds_id, {"status": "approved", "image_ids": [ids[0], ids[5]]})
        set_review(ds_id, {"status": "flagged", "image_ids": [ids[1]]})
        tok = imagesq.create_set(ds_id, [ids[6], ids[0], ids[3], ids[1]])["token"]
        for f in ({}, {"split": "train"}, {"cls": "1"}, {"split": "val", "review": "approved"},
                  {"issue": "near_dup", "ai": "audit"}, {"labeled": "labeled", "cls": "0,2", "issue": "any"},
                  {"review": "none", "ai": "none", "split": "train"}, {"q": "i1"}, {"only_cls": "0,1"},
                  {"set": tok}, {"set": tok, "issue": "any", "review": "approved"}, {"ids": f"{ids[2]},{ids[7]}"},
                  {"exclude_review": "approved", "ai": "pending"}, {"min_boxes": "2"}):
            got = imagesq.facets(ds_id, f, use_cache=False)
            want = _facets_brute(ds_id, f)
            assert got["total"] == imagesq.count_images(ds_id, f), f
            for facet, values in want.items():
                for value, n in values.items():
                    assert got[facet].get(value, 0) == n, (f, facet, value, got[facet])
        assert list(imagesq.facets(ds_id, {})["split"]) == ["train", "val"]
        # the parameterised need= clauses select exactly what NEEDS does
        for need in ("issue", "proposals", "suspect", "flag", "nobox", "unreviewed"):
            sql, params = imagesq.need_clause(need, ds_id)
            a = {r["id"] for r in db.query(f"SELECT i.id {imagesq._JOIN} WHERE i.dataset_id=? AND {sql}", [ds_id] + params)}
            b = {r["id"] for r in db.query(f"SELECT i.id {imagesq._JOIN} WHERE i.dataset_id=? AND {imagesq.NEEDS[need]}", [ds_id])}
            assert a == b and (a or need in ("nobox",)), need
        # cache: same version -> cached; any review / AI / flag / label change -> recomputed
        imagesq.invalidate_facets(ds_id)
        f = {"split": "train"}
        first, cached = imagesq.facets_cached(ds_id, f)
        assert not cached and imagesq.facets_cached(ds_id, f) == (first, True)
        assert imagesq.facets_cached(ds_id, {**f, "seed": "3"})[1] is True        # seed does not change counts
        set_review(ds_id, {"status": "excluded", "image_ids": [ids[2]]})
        out, cached = imagesq.facets_cached(ds_id, f)
        assert not cached and out["review"]["excluded"] == 1
        db.execute("UPDATE ai_preds SET state='rejected' WHERE image_id=? AND kind='swap'", (ids[0],))
        out, cached = imagesq.facets_cached(ds_id, f)
        assert not cached and out["issue"].get("ai_swap", 0) == 0
        db.execute("DELETE FROM image_flags WHERE image_id=?", (ids[3],))
        out, cached = imagesq.facets_cached(ds_id, f)
        assert not cached and "blurry" not in out["issue"]
        from fovea.api.datasets import bulk_labels
        bulk_labels(ds_id, {"op": "clear", "image_ids": [ids[1]]})
        out, cached = imagesq.facets_cached(ds_id, f)
        assert not cached and out["cls"].get(1, 0) == 1
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- N1: split order

def test_sources_and_splits_are_listed_train_val_test_then_others_naturally(tmp_path):
    root = tmp_path / "ord"
    for split in ("test", "fold10", "train", "val", "fold2", "test-1854-포함"):
        _img(root / "images" / split / "a.jpg")
        _lbl(root / "labels" / split / "a.txt", "0 0.5 0.5 0.1 0.1\n")
    lay = layout.detect(str(root))
    assert [s.split for s in lay.sources] == ["train", "val", "test", "fold2", "fold10", "test-1854-포함"]
    assert sorted(["b", "a10", "a2", "A1", "a", ""], key=layout.natural_key) == ["", "a", "A1", "a2", "a10", "b"]
    from fovea.api.common import dataset_public, ordered_splits
    assert ordered_splits(["", "test", "x10", "train", "x9", "val"]) == ["train", "val", "test", "x9", "x10", ""]
    ds_id = _register(root)
    try:
        # a dataset stored before detection sorted its sources is still shown in that order
        d = db.loads(db.query_one("SELECT layout FROM datasets WHERE id=?", (ds_id,))["layout"])
        d["sources"] = d["sources"][::-1]
        db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(d), ds_id))
        _scan(ds_id)
        pub = dataset_public(db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,)), with_job=False)
        assert [s["split"] for s in pub["layout"]["sources"]] == ["train", "val", "test", "fold2", "fold10", "test-1854-포함"]
        assert pub["splits"] == ["train", "val", "test", "fold2", "fold10", "test-1854-포함"]
        st = stats.dataset_stats(ds_id, [])
        assert [s["split"] for s in st["splits"]] == ["train", "val", "test", "fold2", "fold10", "test-1854-포함"]
        assert list(imagesq.facets(ds_id, {})["split"]) == ["train", "val", "test", "fold2", "fold10", "test-1854-포함"]
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- last_scan report

def test_last_scan_report_lists_problems_with_counts_and_examples(tmp_path):
    from fovea.api.common import dataset_public
    root = tmp_path / "rep"
    _img(root / "images" / "ok.jpg")
    lines = ["images/ok.jpg"] + [f"images/missing{k}.jpg" for k in range(7)] + ["notes.md"]
    (root / "notes.md").write_text("x")
    (root / "train.txt").write_text("\n".join(lines) + "\n")
    for k in range(3):
        _lbl(root / "labels" / "orphan" / f"x{k}.txt", "0 0.5 0.5 0.1 0.1\n")
    ds_id = _register(root, raw_path=root / "train.txt")
    try:
        res = _scan(ds_id)
        by = {p["code"]: p for p in res["problems"]}
        assert by["missing_list_entry"]["count"] == 7 and len(by["missing_list_entry"]["examples"]) == 5
        assert by["missing_list_entry"]["severity"] == "warn" and by["missing_list_entry"]["examples"][0] == "images/missing0.jpg"
        assert by["not_an_image"]["count"] == 1 and by["not_an_image"]["severity"] == "info"
        assert res["problems"][0]["severity"] == "warn"                  # warnings first
        assert "7 list entries not found" in res["notes"] and all(isinstance(n, str) for n in res["notes"])
        pub = dataset_public(db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,)), with_job=False)
        assert pub["last_scan"]["problems"] == res["problems"]
        # an older report (problems as plain strings) is presented in the same shape
        db.kv_set(f"scan:{ds_id}", {"at": 1, "images": 1, "problems": ["a: not found", "b"], "notes": ["x"]})
        old = dataset_public(db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,)), with_job=False)["last_scan"]
        assert old["problems"] == [{"code": "other", "label": "Problems reported by the scan", "severity": "warn",
                                    "count": 2, "examples": ["a: not found", "b"]}]
    finally:
        _drop(ds_id)
    # labels-only folders: reported by every scan, and dropped once their images exist
    root = _ultra(tmp_path, name="orph")
    for k in range(4):
        _lbl(root / "labels" / "orphan" / f"x{k}.txt", "0 0.5 0.5 0.1 0.1\n")
    ds_id = _register(root)
    try:
        res = _scan(ds_id)
        p = [x for x in res["problems"] if x["code"] == "labels_without_images"][0]
        assert p["count"] == 4 and p["examples"][0].endswith("labels/orphan (4 files)")
        for k in range(4):
            _img(root / "images" / "orphan" / f"x{k}.jpg")
        lay = layout.Layout.from_dict(db.loads(db.query_one("SELECT layout FROM datasets WHERE id=?", (ds_id,))["layout"]))
        lay.sources.append(layout.Source(split="orphan", img_dir=str(root / "images" / "orphan"),
                                         label_dir=str(root / "labels" / "orphan")))
        db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay.to_dict()), ds_id))
        res = _scan(ds_id)
        stored = db.loads(db.query_one("SELECT layout FROM datasets WHERE id=?", (ds_id,))["layout"])
        assert not [x for x in res["problems"] if x["code"] == "labels_without_images"]
        assert stored["unmatched_labels"] == [] and not any("no matching images" in n for n in stored["notes"])
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- planner statistics

def test_planner_statistics_follow_the_volatile_tables(scanned):
    ds_id, _root = scanned
    t0 = time.time()
    while db._stats_state["running"] and time.time() - t0 < 10:      # a background check from a read path
        time.sleep(0.05)
    db._stats_state["at"] = time.monotonic() + 3600                    # ... and none during this test
    conn = db.connect()
    conn.execute("ANALYZE")
    conn.execute("DELETE FROM sqlite_stat1 WHERE tbl IN ('ai_preds', 'image_flags')")
    # statistics recorded while ai_preds held 3 rows (what a startup ANALYZE does on a fresh install)
    conn.execute("INSERT INTO sqlite_stat1(tbl, idx, stat) VALUES('ai_preds', 'idx_ai_preds_img', '3 1 1')")
    conn.execute("INSERT INTO sqlite_stat1(tbl, idx, stat) VALUES('image_flags', 'idx_image_flags_ds', '5 5 1')")
    iid = _iid(ds_id, "train/f00.jpg")
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'audit', ?)", (ds_id, db.now())).lastrowid
    conn.executemany("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind) VALUES(?,?,?,0,.5,.5,.1,.1,'missing')",
                     [(run, ds_id, iid)] * (db.SMALL_TABLE_ROWS + 200))
    try:
        changed = db.refresh_planner_stats()
        assert "ai_preds" in changed and "image_flags" in changed      # grown -> analysed; near-empty -> no stats
        n = int(conn.execute("SELECT stat FROM sqlite_stat1 WHERE tbl='ai_preds' AND idx='idx_ai_preds_img'").fetchone()[0].split()[0])
        assert n >= db.SMALL_TABLE_ROWS
        assert conn.execute("SELECT COUNT(*) FROM sqlite_stat1 WHERE tbl='image_flags'").fetchone()[0] == 0
        assert db.refresh_planner_stats(("ai_preds",)) == []            # nothing drifted: nothing to do
        assert db.refresh_planner_stats(("not_a_table; DROP TABLE images",)) == []
        assert imagesq.count_images(ds_id, {"ai": "audit"}) == 1
        assert db.maybe_refresh_planner_stats() is False                # throttled
    finally:
        conn.execute("DELETE FROM ai_runs WHERE id=?", (run,))
        db.refresh_planner_stats()
        db._stats_state["at"] = 0.0


def test_covering_indexes_are_upgraded_in_place():
    c = sqlite3.connect(":memory:")
    c.executescript("""CREATE TABLE images(id INTEGER PRIMARY KEY, dataset_id, split, has_label, n_boxes, issues, classes);
        CREATE TABLE boxes(id INTEGER PRIMARY KEY, dataset_id, cls, image_id);
        CREATE TABLE ai_preds(id INTEGER PRIMARY KEY, dataset_id, state, kind, image_id, conf);
        CREATE INDEX idx_ai_preds_img ON ai_preds(image_id, state);
        CREATE TABLE image_flags(image_id, dataset_id, code);
        CREATE INDEX idx_images_ds_split ON images(dataset_id, split);
        CREATE INDEX idx_boxes_ds_cls ON boxes(dataset_id, cls);
        CREATE INDEX idx_ai_preds_ds ON ai_preds(dataset_id, state, kind);
        CREATE INDEX idx_image_flags_ds ON image_flags(dataset_id, code);
        INSERT INTO images VALUES(1, 'd', 'train', 1, 2, '[]', '[0]'); ANALYZE;""")
    assert sorted(db._ensure_index_shapes(c)) == sorted(db.INDEX_SHAPES)
    for name, (_table, cols) in db.INDEX_SHAPES.items():
        assert tuple(r[2] for r in c.execute(f"PRAGMA index_info({name})")) == cols
    assert db._ensure_index_shapes(c) == []                                   # idempotent
    plan = " ".join(r[3] for r in c.execute("EXPLAIN QUERY PLAN SELECT split, has_label, n_boxes > 0, issues, classes, COUNT(*) "
                                             "FROM images WHERE dataset_id = 'd' GROUP BY 1, 2, 3, 4, 5"))
    assert "COVERING INDEX idx_images_ds_split" in plan
    c.close()


# =========================================================================== round-1 review fixes

def _wait_jobs(ds_id, timeout=30.0):
    from fovea.jobs import jobs
    deadline = time.time() + timeout
    while time.time() < deadline:
        if not [j for j in jobs.list(ds_id) if j.status in ("queued", "running")]:
            return
        time.sleep(0.02)
    raise TimeoutError(ds_id)


def _put(ds_id, iid, boxes):
    from fovea.api.datasets import get_labels, put_labels
    return put_labels(ds_id, iid, {"boxes": boxes, "base_version": get_labels(ds_id, iid)["version"]})


@pytest.fixture()
def no_scan_hooks(monkeypatch):
    """Relocation runs the real scan job, which would start other roles' post-scan work on these datasets."""
    from fovea.core import scanner
    monkeypatch.setattr(scanner, "POST_SCAN_HOOKS", [])


# --------------------------------------------------------------------------- restore vs. concurrent saves

def test_restore_never_overwrites_a_save_made_while_it_runs(scanned, monkeypatch):
    from fovea.core import scanner
    ds_id, root = scanned
    train = [_iid(ds_id, f"train/f0{k}.jpg") for k in range(3)]
    v1 = snapshots.snapshot_dataset(ds_id, name="v1")
    for iid in train:
        _put(ds_id, iid, [[1, 0.3, 0.3, 0.1, 0.1]])
    last = train[-1]
    orig, fired = scanner.refresh_image, []

    def save_meanwhile(iid, *a, **k):
        if not fired:                     # after the undo snapshot and the first restored file: the user saves
            fired.append(_put(ds_id, last, [[2, 0.7, 0.7, 0.1, 0.1]])["saved"])
        return orig(iid, *a, **k)

    monkeypatch.setattr(scanner, "refresh_image", save_meanwhile)
    res = snapshots.restore(ds_id, v1)
    monkeypatch.setattr(scanner, "refresh_image", orig)
    assert fired == [1] and res["kept"] == 1 and res["restored"] == 2 and res["errors"] == 0
    assert any("changed during the restore" in p for p in res["problems"])
    assert (root / "labels/train/f02.txt").read_text().startswith("2 0.700000")          # the newer save is kept
    assert (root / "labels/train/f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    # the undo snapshot can still put back exactly what was overwritten — and holds nothing else: the file kept
    # during the restore is not in it, so undoing the restore cannot overwrite that newer save either
    assert snapshots.get_snapshot(ds_id, res["snapshot_id"])["n_files"] == 2
    undo = snapshots.restore(ds_id, res["snapshot_id"])
    assert undo["restored"] == 2 and (root / "labels/train/f00.txt").read_text().startswith("1 0.300000")
    assert (root / "labels/train/f02.txt").read_text().startswith("2 0.700000")


# --------------------------------------------------------------------------- snapshot lifetime

def test_a_snapshot_in_use_is_never_pruned_or_deleted(scanned, monkeypatch):
    from fastapi.testclient import TestClient
    from fovea.main import app
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    s = snapshots.snapshot_images(ds_id, ids, "bulk set_class")
    with snapshots.pinned(s):
        for _ in range(snapshots.KEEP_SMALL + 2):
            snapshots.snapshot_images(ds_id, ids[:1], "objects")
        assert snapshots.settle()
        assert snapshots.get_snapshot(ds_id, s)                       # the oldest auto, yet kept while in use
        with pytest.raises(snapshots.SnapshotBusy):
            snapshots.delete_snapshot(ds_id, s)
        r = TestClient(app).delete(f"/api/datasets/{ds_id}/snapshots/{s}")
        assert r.status_code == 409 and r.json()["detail"]["code"] == "snapshot_in_use"
        assert not db.query("SELECT 1 FROM snapshot_files f LEFT JOIN label_blobs b ON b.sha=f.sha "
                            "WHERE f.snapshot_id=? AND f.sha IS NOT NULL AND b.sha IS NULL", (s,))
    snapshots.prune(ds_id)
    assert not snapshots.get_snapshot(ds_id, s)                       # released: normal retention again


def test_restore_pins_its_snapshot_against_pruning_by_edits_made_meanwhile(tmp_path, monkeypatch):
    from fovea.api.datasets import bulk_labels
    from fovea.core import scanner
    root = tmp_path / "pr"
    for k in range(6):
        _img(root / "images/train" / f"f{k}.jpg")
        _lbl(root / "labels/train" / f"f{k}.txt", f"0 0.{k + 1} 0.5 0.1 0.1\n")     # each file its own blob
    ds_id = _register(root)
    _scan(ds_id)
    try:
        ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY rel_path", (ds_id,))]
        s = bulk_labels(ds_id, {"op": "set_class", "cls": 3, "image_ids": ids})["snapshot_id"]
        for _ in range(snapshots.KEEP_AUTO - 1):                        # s is now the oldest of 20 automatic versions
            snapshots.snapshot_images(ds_id, ids[:1], "objects")
        orig, fired = scanner.refresh_image, []

        def edit_meanwhile(iid, *a, **k):
            if not fired:
                fired.append(snapshots.snapshot_images(ds_id, ids[:1], "objects"))   # another auto version
            return orig(iid, *a, **k)

        monkeypatch.setattr(scanner, "refresh_image", edit_meanwhile)
        res = snapshots.restore(ds_id, s)
        assert fired and res["errors"] == 0 and res["restored"] == 6 and snapshots.get_snapshot(ds_id, s)
        assert [(root / f"labels/train/f{k}.txt").read_text()[0] for k in range(6)] == ["0"] * 6
    finally:
        _drop(ds_id)


def test_snapshot_blobs_survive_a_garbage_collection_racing_the_insert(scanned, monkeypatch):
    from contextlib import contextmanager
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    only = snapshots.snapshot_dataset(ds_id, name="only holder of these blobs")
    real, fired = db.transaction, []

    @contextmanager
    def racing():
        if not fired:                     # between the 'known blobs' check and the insert: the holder is deleted
            fired.append(1)
            monkeypatch.setattr(db, "transaction", real)
            snapshots.delete_snapshot(ds_id, only)
        with real() as conn:
            yield conn

    monkeypatch.setattr(db, "transaction", racing)
    s = snapshots.snapshot_images(ds_id, ids, "bulk clear")
    monkeypatch.setattr(db, "transaction", real)
    assert fired and not db.query_one("SELECT 1 FROM snapshots WHERE name='only holder of these blobs'")
    assert not db.query("SELECT 1 FROM snapshot_files f LEFT JOIN label_blobs b ON b.sha=f.sha "
                        "WHERE f.snapshot_id=? AND f.sha IS NOT NULL AND b.sha IS NULL", (s,))


# --------------------------------------------------------------------------- relocation keeps ids

def _seed_refs(ds_id, iid):
    """An AI proposal, an audit finding, a QA flag + hash, a class-card example and a working set on one image."""
    from fovea.api.datasets import put_class_card
    now = db.now()
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, created_at) VALUES(?,?,?,?,?)",
                     (ds_id, "batch", "mock", "m", now)).lastrowid
    db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, state, created_at) "
               "VALUES(?,?,?,0,.5,.5,.1,.1,.4,'detect','pending',?)", (run, ds_id, iid, now))
    db.execute("INSERT INTO ai_audit(image_id, dataset_id, run_id, score, created_at) VALUES(?,?,?,0.9,?)", (iid, ds_id, run, now))
    db.execute("INSERT INTO image_flags(image_id, dataset_id, code) VALUES(?,?,'blurry')", (iid, ds_id))
    db.execute("INSERT INTO image_hash(image_id, dataset_id, dhash) VALUES(?,?,7)", (iid, ds_id))
    put_class_card(ds_id, 0, {"examples": [{"image_id": iid, "row": 0, "polarity": 1}]})
    return imagesq.create_set(ds_id, [iid], {}, None, None)["token"]


def _refs(ds_id, iid, token):
    return {
        "preds": db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE image_id=? AND state='pending'", (iid,))["n"],
        "audit": db.query_one("SELECT COUNT(*) n FROM ai_audit WHERE image_id=?", (iid,))["n"],
        "flags": db.query_one("SELECT COUNT(*) n FROM image_flags WHERE image_id=?", (iid,))["n"],
        "hash": db.query_one("SELECT COUNT(*) n FROM image_hash WHERE image_id=?", (iid,))["n"],
        "card": [e["image_id"] for e in db.loads(db.query_one(
            "SELECT examples FROM class_cards WHERE dataset_id=? AND cls=0", (ds_id,))["examples"])],
        "set": imagesq.list_ids(ds_id, {"set": token}, None),
    }


@pytest.mark.parametrize("how", ["copy", "move"])
def test_relocation_keeps_ids_and_restore_writes_into_the_new_root_only(tmp_path, how, no_scan_hooks):
    import shutil
    from fovea.api.datasets import relocate_dataset
    root = _ultra(tmp_path, name="old")
    ds_id = _register(root)
    _scan(ds_id)
    try:
        ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        a = ids["train/f00.jpg"]
        token = _seed_refs(ds_id, a)
        v1 = snapshots.snapshot_dataset(ds_id, name="v1")
        _put(ds_id, a, [[1, 0.3, 0.3, 0.1, 0.1]])
        new = tmp_path / "new"
        (shutil.copytree if how == "copy" else shutil.move)(str(root), str(new))
        relocate_dataset(ds_id, {"path": str(new)})
        _wait_jobs(ds_id)
        assert {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))} == ids
        assert _refs(ds_id, a, token) == {"preds": 1, "audit": 1, "flags": 1, "hash": 1, "card": [a], "set": [a]}
        assert db.query_one("SELECT label_path FROM images WHERE id=?", (a,))["label_path"].startswith(str(new))
        d = snapshots.diff(ds_id, v1)
        assert d["summary"]["images_changed"] == 1 and d["items"][0]["image_id"] == a
        res = snapshots.restore(ds_id, v1)
        assert res["restored"] == 1 and res["not_found"] == 0
        assert (new / "labels/train/f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
        if how == "copy":
            assert (root / "labels/train/f00.txt").read_text().startswith("1 0.300000")    # the old copy untouched
        else:
            assert not root.exists()                                                     # never re-created
    finally:
        _drop(ds_id)


def test_relocation_to_a_new_layout_carries_references_over_to_the_new_rows(tmp_path, no_scan_hooks):
    import shutil
    from fovea.api.datasets import relocate_dataset
    root = tmp_path / "old"
    for k in range(3):
        _img(root / "images/train" / f"u{k}.jpg")
        _lbl(root / "labels/train" / f"u{k}.txt", "0 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)
    _scan(ds_id)
    try:
        a = _iid(ds_id, "train/u0.jpg")
        token = _seed_refs(ds_id, a)
        v1 = snapshots.snapshot_dataset(ds_id, name="v1")
        new = tmp_path / "new"
        for sub in ("images", "labels"):                  # one folder level more: every index key changes
            shutil.copytree(root / sub / "train", new / sub / "train" / "cam1")
        relocate_dataset(ds_id, {"path": str(new)})
        _wait_jobs(ds_id)
        b = _iid(ds_id, "train/cam1/u0.jpg")
        assert b != a and not db.query_one("SELECT 1 FROM images WHERE id=?", (a,))
        assert _refs(ds_id, b, token) == {"preds": 1, "audit": 1, "flags": 1, "hash": 1, "card": [b], "set": [b]}
        _put(ds_id, b, [])
        res = snapshots.restore(ds_id, v1)               # the snapshot's files follow their images too
        assert res["restored"] == 1 and (new / "labels/train/cam1/u0.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    finally:
        _drop(ds_id)


def test_relocation_never_writes_into_the_old_root_and_an_interrupted_one_is_partial(tmp_path, monkeypatch):
    import shutil
    from fovea.api import datasets as api
    from fovea.core import scanner
    root = _ultra(tmp_path, name="old")
    ds_id = _register(root)
    _scan(ds_id)
    try:
        a = _iid(ds_id, "train/f00.jpg")
        new = tmp_path / "new"
        shutil.copytree(root, new)
        monkeypatch.setattr(api, "jobs", type("NoJobs", (), {"submit": staticmethod(lambda *a, **k: Job(id="x", kind="scan")),
                                                           "list": staticmethod(lambda *a, **k: []),
                                                           "cancel_all": staticmethod(lambda *a, **k: [])})())
        api.relocate_dataset(ds_id, {"path": str(new)})          # the rescan has not run yet
        with pytest.raises(HTTPException) as e:
            api.put_labels(ds_id, a, {"boxes": [[1, 0.3, 0.3, 0.1, 0.1]]})
        assert e.value.status_code == 400
        assert (root / "labels/train/f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
        job = Job(id="relo", kind="scan")
        seen = []

        def cancel_midway(*a, **k):
            seen.append(1)
            if len(seen) == 3:
                job.cancel()
        monkeypatch.setattr(scanner, "POST_SCAN_HOOKS", [])
        orig_update = Job.update

        def update(self, *a, **k):
            cancel_midway()
            return orig_update(self, *a, **k)
        monkeypatch.setattr(Job, "update", update)
        with pytest.raises(JobCancelled):
            scanner.scan_dataset(ds_id, job)
        assert db.query_one("SELECT status FROM datasets WHERE id=?", (ds_id,))["status"] == "partial"
    finally:
        _drop(ds_id)


def test_restore_finds_files_by_index_key_and_never_writes_outside_the_dataset(scanned, tmp_path):
    ds_id, root = scanned
    a = _iid(ds_id, "train/f00.jpg")
    v1 = snapshots.snapshot_dataset(ds_id, name="v1")
    _put(ds_id, a, [])
    db.execute("DELETE FROM images WHERE id=?", (a,))              # re-indexed under a new id
    _scan(ds_id)
    a2 = _iid(ds_id, "train/f00.jpg")
    assert a2 != a
    outside = tmp_path / "elsewhere" / "labels" / "x.txt"          # recorded path outside the dataset, image gone
    db.execute("INSERT INTO snapshot_files(snapshot_id, image_id, label_path, sha, rel_path) "
               "SELECT ?, NULL, ?, sha, NULL FROM snapshot_files WHERE snapshot_id=? LIMIT 1", (v1, str(outside), v1))
    res = snapshots.restore(ds_id, v1)
    assert res["restored"] == 1 and res["not_found"] == 1 and not outside.parent.exists()
    assert (root / "labels/train/f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"
    assert db.query_one("SELECT n_boxes FROM images WHERE id=?", (a2,))["n_boxes"] == 1


# --------------------------------------------------------------------------- diff speed and correctness

def test_diff_rechecks_by_stat_and_reads_only_changed_files(scanned, monkeypatch):
    ds_id, root = scanned
    s = snapshots.snapshot_dataset(ds_id, name="v1")
    reads = []
    real = snapshots._read_sig
    monkeypatch.setattr(snapshots, "_read_sig", lambda p: (reads.append(p), real(p))[1])
    snapshots.forget_diff()
    assert snapshots.diff(ds_id, s)["total"] == 0 and reads == []          # every stat matches the snapshot's
    f = root / "labels/val/f01.txt"
    f.write_text("0 0.5 0.5 0.2 0.2\n4 0.1 0.1 0.05 0.05\n")              # an edit outside Fovea, no TTL to wait for
    d = snapshots.diff(ds_id, s)
    assert reads == [str(f)] * 2                       # compared once, and read again for the page's item
    assert d["summary"] == {"images_changed": 1, "boxes_added": 1, "boxes_removed": 0,
                            "per_class": {"4": {"added": 1, "removed": 0}}, "proposals": 0}
    assert d["items"][0]["after"] == [[0, 0.5, 0.5, 0.2, 0.2], [4, 0.1, 0.1, 0.05, 0.05]]
    reads.clear()
    assert snapshots.diff(ds_id, s, offset=0, limit=1)["total"] == 1 and reads == [str(f)]   # only the page's item
    f.unlink()
    d = snapshots.diff(ds_id, s)
    assert d["summary"]["boxes_removed"] == 1 and d["items"][0]["exists"] is False


def test_fast_label_comparison_matches_the_parser():
    import random
    rnd = random.Random(7)
    pieces = ["0 0.5 0.5 0.2 0.2", "1 0.500000 0.5 0.2 0.20000001", "2 0.1 0.1 0.1 0.1 0.93", "3 0.1 0.1 nan 0.1",
              "x 0.1 0.1 0.1 0.1", "4 0.1 0.1", "", "  5   0.3 0.3 0.1 0.1  ", "6 0.2 0.2 0.1 0.1 abc", "1.5 0.2 0.2 0.1 0.1",
              "7 0.2 0.2 0.1 0.1 inf", "0 0.10 0.10 0.40 0.10 0.40 0.40 0.10 0.40", "9e99 0.1 0.1 0.1 0.1", "-1 .1 .1 .1 .1"]
    for _ in range(300):
        a = "\n".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 6)))
        b = "\n".join(rnd.choice(pieces) for _ in range(rnd.randint(0, 6)))
        if rnd.random() < 0.2:
            a = "﻿" + a
        assert snapshots._rows(a.encode()) == snapshots._rows_full(a)
        info = snapshots._compare(snapshots._sha(a.encode()) + "x", b.encode(), a.encode())
        ra, rb = Counter(tuple(r) for r in snapshots._rows_full(a)), Counter(tuple(r) for r in snapshots._rows_full(b))
        added, removed = rb - ra, ra - rb
        assert info["n_added"] == sum(added.values()) and info["n_removed"] == sum(removed.values())
        assert dict(info["added"]) == dict(Counter(k[0] for k in added.elements()))
        assert dict(info["removed"]) == dict(Counter(k[0] for k in removed.elements()))


def test_restore_job_ends_at_its_total_and_names_unnamed_versions(scanned):
    from fovea.api.datasets import bulk_labels
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    s = bulk_labels(ds_id, {"op": "set_class", "cls": 2, "image_ids": ids})["snapshot_id"]
    job = Job(id="r", kind="restore")
    res = snapshots.restore(ds_id, s, job=job)
    assert (job.done, job.total) == (6, 6) and job.message == "Restored 6 files"
    op = snapshots.get_snapshot(ds_id, res["snapshot_id"])["op"]
    assert op.startswith("restore Bulk class change · ") and "#" not in op
    when = time.strftime("%H:%M", time.localtime(snapshots.get_snapshot(ds_id, s)["created_at"]))
    assert snapshots.describe(snapshots.get_snapshot(ds_id, s)) == f"Bulk class change · {when}"
    for op, text in (("ai_append", "AI append write"), ("ai_accept", "Accepting AI proposals"), ("objects", "Objects edit"),
                     ("restore X", "Undo point of a restore"), ("bulk clear", "Label clearing")):
        assert snapshots.describe({"name": "", "kind": "auto", "op": op, "created_at": None}) == text
    assert snapshots.describe({"name": "Before relabel", "op": "manual"}) == "Before relabel"


# --------------------------------------------------------------------------- input validation

@pytest.mark.parametrize("body", [
    {"op": "set_class", "cls": "NaN"}, {"op": "set_class", "cls": "inf"}, {"op": "set_class", "cls": "-inf"},
    {"op": "set_class", "cls": 10 ** 400}, {"op": "set_class", "cls": 1, "only_cls": "inf"},
    {"op": "remap", "mapping": {"0": "NaN"}}, {"op": "remap", "mapping": {"Infinity": 1}}, {"op": "delete_class", "cls": 1e400},
])
def test_bulk_rejects_non_finite_numbers_with_400(scanned, body):
    from fovea.api.datasets import bulk_labels
    ds_id, root = scanned
    with pytest.raises(HTTPException) as e:
        bulk_labels(ds_id, {**body, "image_ids": [_iid(ds_id, "train/f00.jpg")]})
    assert e.value.status_code == 400
    assert (root / "labels/train/f00.txt").read_text() == "0 0.5 0.5 0.2 0.2\n"


def test_class_card_examples_with_non_finite_numbers_are_refused_or_dropped(scanned):
    from fovea.api.datasets import put_class_card
    ds_id, _root = scanned
    iid = _iid(ds_id, "train/f00.jpg")
    with pytest.raises(HTTPException) as e:
        put_class_card(ds_id, 0, {"examples": [{"image_id": float("inf")}]})
    assert e.value.status_code == 400
    card = put_class_card(ds_id, 0, {"examples": [{"image_id": iid, "box": [float("nan"), 0.5, 0.1, 0.1]}]})["card"]
    assert card["examples"] == [{"image_id": iid, "row": 0, "polarity": 1}]


# --------------------------------------------------------------------------- symlinked label files

def test_a_symlinked_label_is_written_through_inside_the_dataset_and_refused_outside(tmp_path):
    from fastapi.testclient import TestClient
    from fovea.api.datasets import bulk_labels
    from fovea.main import app
    root = tmp_path / "sl"
    shared, outside = root / "shared_labels" / "f0.txt", tmp_path / "private" / "notes.txt"
    _lbl(shared, "0 0.5 0.5 0.2 0.2\n")
    _lbl(outside, "1 0.5 0.5 0.2 0.2\n")                           # e.g. a label shared from another project
    for k in range(3):
        _img(root / "images/train" / f"f{k}.jpg")
    (root / "labels/train").mkdir(parents=True)
    os.symlink(shared, root / "labels/train/f0.txt")
    os.symlink(outside, root / "labels/train/f1.txt")
    _lbl(root / "labels/train/f2.txt", "0 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)
    _scan(ds_id)
    c = TestClient(app)
    try:
        a, b = _iid(ds_id, "train/f0.jpg"), _iid(ds_id, "train/f1.jpg")
        _put(ds_id, a, [[1, 0.3, 0.3, 0.1, 0.1]])
        assert (root / "labels/train/f0.txt").is_symlink() and shared.read_text().startswith("1 0.300000")
        v = c.get(f"/api/datasets/{ds_id}/images/{b}/labels").json()["version"]
        r = c.put(f"/api/datasets/{ds_id}/images/{b}/labels", json={"boxes": [[1, 0.3, 0.3, 0.1, 0.1]], "base_version": v})
        assert r.status_code == 403 and r.json()["detail"]["code"] == "label_link_outside"
        assert (root / "labels/train/f1.txt").is_symlink() and outside.read_text() == "1 0.5 0.5 0.2 0.2\n"
        res = bulk_labels(ds_id, {"op": "set_class", "cls": 0, "image_ids": [a, b]})
        assert res["skipped_links"] == 1 and res["images"] == 1 and outside.read_text() == "1 0.5 0.5 0.2 0.2\n"
        assert shared.read_text().startswith("0 0.300000")
        undo = snapshots.restore(ds_id, res["snapshot_id"])
        assert undo["restored"] == 1 and (root / "labels/train/f0.txt").is_symlink() and shared.read_text().startswith("1 ")
        lv = c.get(f"/api/datasets/{ds_id}/images/{a}/labels").json()["version"]
        r = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": a, "row": 0, "cls": 0, "version": lv}]})
        assert r.status_code == 200 and (root / "labels/train/f0.txt").is_symlink() and shared.read_text().startswith("0 ")
        s2 = snapshots.snapshot_images(ds_id, [b], "objects")
        db.execute("UPDATE snapshot_files SET sha=(SELECT sha FROM snapshot_files WHERE snapshot_id=? AND image_id=?), "
                   "sig=NULL WHERE snapshot_id=?", (res["snapshot_id"], a, s2))     # pretend that version of f1 differed
        out = snapshots.restore(ds_id, s2)
        assert out["errors"] == 1 and "outside the dataset" in out["problems"][0] and outside.read_text() == "1 0.5 0.5 0.2 0.2\n"
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- file links out of a downloaded dataset

def test_scan_skips_file_links_out_of_the_dataset_that_are_not_images(tmp_path):
    root = tmp_path / "dl"
    secret = tmp_path / "home" / ".ssh" / "id_rsa"
    _lbl(secret, "TOP-SECRET id_rsa\n")
    _img(tmp_path / "store" / "real.jpg")
    for k in range(2):
        _img(root / "images/train" / f"f{k}.jpg")
        _lbl(root / "labels/train" / f"f{k}.txt", "0 0.5 0.5 0.2 0.2\n")
    os.symlink(secret, root / "images/train/leak.jpg")
    os.symlink(secret, root / "labels/train/leak.txt")
    os.symlink(tmp_path / "store" / "real.jpg", root / "images/train/linked.jpg")   # a split built from links: kept
    _img(root / "images/train" / "broken.jpg")
    (root / "images/train" / "broken.jpg").write_bytes(b"not an image")             # the dataset's own bad file: kept
    ds_id = _register(root)
    try:
        res = _scan(ds_id)
        assert _rels(ds_id) == ["train/broken.jpg", "train/f0.jpg", "train/f1.jpg", "train/linked.jpg"]
        p = [x for x in res["problems"] if x["code"] == "link_outside"]
        assert p and p[0]["count"] == 1 and p[0]["severity"] == "warn" and p[0]["examples"][0].endswith("leak.jpg")
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- indexes / migrations

def test_ai_conf_sort_reads_pending_proposals_from_a_covering_index(scanned):
    ds_id, _root = scanned
    where, params = imagesq.build_where(ds_id, {"ai": "pending"})
    ob = imagesq.order_clause("ai_conf", None, f={"ai": "pending"})
    sql = f"SELECT i.id {imagesq._from(where, ob, f={'ai': 'pending'})} WHERE {where} ORDER BY {ob} LIMIT 80"
    plan = " ".join(r["detail"] for r in db.query("EXPLAIN QUERY PLAN " + sql, params))
    assert "COVERING INDEX idx_ai_preds_img" in plan


def test_snapshot_files_gain_rel_path_and_sig_on_old_databases(tmp_path):
    c = sqlite3.connect(tmp_path / "s.db")
    c.executescript("CREATE TABLE snapshot_files (snapshot_id INTEGER NOT NULL, image_id INTEGER, label_path TEXT NOT NULL, "
                    "sha TEXT, PRIMARY KEY(snapshot_id, label_path)); INSERT INTO snapshot_files VALUES(1, 2, '/x', 'ab');")
    c.executescript("CREATE TABLE snapshots (id INTEGER PRIMARY KEY, dataset_id TEXT NOT NULL, name TEXT NOT NULL DEFAULT '', "
                    "kind TEXT NOT NULL DEFAULT 'auto', op TEXT NOT NULL DEFAULT '', n_files INTEGER NOT NULL DEFAULT 0, "
                    "created_at REAL); INSERT INTO snapshots(id, dataset_id) VALUES(1, 'd');")
    c.execute("PRAGMA user_version=4")
    c.executescript(db.SCHEMA)
    assert db.migrate(c) == ["snapshot_files.rel_path", "snapshot_files.sig", "snapshot undo tracking",
                             "ai_preds AUTOINCREMENT"]
    assert c.execute("SELECT image_id, rel_path, sig, after_ver FROM snapshot_files").fetchone() == (2, None, None, None)
    assert c.execute("SELECT tracked FROM snapshots").fetchone() == (0,)      # an old version: restored in full
    c.close()


# --------------------------------------------------------------------------- final: ai_preds ids are never reused

def test_schema_blocks_end_at_their_statement_not_inside_a_comment():
    """ai_preds' ref_box comment contains ');' — the rebuild must still get the whole CREATE TABLE."""
    import re
    for table in re.findall(r"CREATE TABLE IF NOT EXISTS (\w+) \(", db.SCHEMA):
        blk = db._schema_block(table)
        assert sqlite3.complete_statement(blk) and blk.count(";") >= 1, table
        sqlite3.connect(":memory:").execute(blk)                  # exactly one statement
    assert "ref_box TEXT" in db._schema_block("ai_preds")


def test_ai_pred_ids_are_never_reused_after_a_re_propose(scanned):
    """A re-propose deletes the image's pending preds (the newest ids) and inserts new ones: those must get new ids,
    or an Annotate undo / queued decision naming an old id lands on a new proposal."""
    from fovea.ai import store
    ds_id, _root = scanned
    iid = _iid(ds_id, "train/f00.jpg")
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'propose', ?)", (ds_id, db.now())).lastrowid
    try:
        box = {"cls": 0, "xc": .5, "yc": .5, "w": .2, "h": .2, "conf": .9}
        first = [p["id"] for p in store.insert_preds(run, ds_id, iid, [box, dict(box, xc=.3)])]
        store.decide(ds_id, [], [first[0]])                        # Annotate rejects one, keeps the other pending
        again = [p["id"] for p in store.insert_preds(run, ds_id, iid, [dict(box, cls=1), dict(box, cls=1, xc=.7)],
                                                     replace_pending=("detect",))]
        assert min(again) > max(first) and not set(again) & set(first)
        assert db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE id=?", (first[1],))["n"] == 0
        # the undo of the stale proposal's decision finds nothing; the new proposals keep their state
        assert store.undecide(ds_id, [first[1]]) == 0 and store.decide(ds_id, [first[1]], []) == 0
        assert {r["state"] for r in db.query("SELECT state FROM ai_preds WHERE id IN (?, ?)", again)} == {"pending"}
        # deleting the newest row (no successor) does not free its id either
        db.execute("DELETE FROM ai_preds WHERE id=?", (max(again),))
        nxt = store.insert_preds(run, ds_id, iid, [box])[0]["id"]
        assert nxt > max(again)
    finally:
        db.execute("DELETE FROM ai_runs WHERE id=?", (run,))


OLD_AI_PREDS_DDL = """CREATE TABLE IF NOT EXISTS ai_preds (
  id INTEGER PRIMARY KEY,
  run_id INTEGER NOT NULL REFERENCES ai_runs(id) ON DELETE CASCADE,
  dataset_id TEXT NOT NULL,
  image_id INTEGER NOT NULL REFERENCES images(id) ON DELETE CASCADE,
  cls INTEGER NOT NULL, xc REAL NOT NULL, yc REAL NOT NULL, w REAL NOT NULL, h REAL NOT NULL,
  conf REAL, label TEXT NOT NULL DEFAULT '', kind TEXT NOT NULL DEFAULT 'detect', ref_row INTEGER,
  reason TEXT NOT NULL DEFAULT '', state TEXT NOT NULL DEFAULT 'pending', created_at REAL, decided_at REAL
);"""
OLD_AI_PREDS_DATA = """
DROP INDEX idx_ai_preds_ds;
CREATE INDEX idx_ai_preds_ds ON ai_preds(dataset_id, state, kind);
CREATE INDEX idx_ai_preds_run_state ON ai_preds(run_id, state);
INSERT INTO datasets(id, name, root) VALUES('d', 'd', '/x');
INSERT INTO images(id, dataset_id, rel_path, abs_path) VALUES(3, 'd', 'a.jpg', '/x/a.jpg');
INSERT INTO ai_runs(id, dataset_id, kind) VALUES(1, 'd', 'propose'), (2, 'd', 'audit');
INSERT INTO ai_preds(id, run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, ref_row, state, reason)
  VALUES(2, 1, 'd', 3, 0, .5, .5, .2, .2, .9, 'detect', NULL, 'rejected', 'r'),
        (5, 2, 'd', 3, 1, .4, .4, .1, .1, NULL, 'loose', 0, 'pending', '');
INSERT INTO snapshots(id, dataset_id) VALUES(1, 'd');
INSERT INTO snapshot_preds VALUES(1, 9, 3, 'pending', 'accepted', NULL);
"""


def test_old_ai_preds_tables_are_rebuilt_with_autoincrement(tmp_path):
    """Rows, columns, indexes (also ai.store's own) and foreign keys survive; the sequence starts past every id
    a snapshot still names, so neither a deleted proposal's id nor one an undo point refers to comes back."""
    c = sqlite3.connect(tmp_path / "p.db", isolation_level=None)
    c.executescript(db.SCHEMA.replace(db._schema_block("ai_preds"), OLD_AI_PREDS_DDL))    # a 2.0 database before the fix
    c.executescript(OLD_AI_PREDS_DATA)
    c.execute(f"PRAGMA user_version={db.SCHEMA_VERSION - 1}")
    c.executescript(db.SCHEMA)
    assert "AUTOINCREMENT" not in db._table_sql(c, "ai_preds").upper()
    assert db.migrate(c) == ["ai_preds AUTOINCREMENT"]
    assert "AUTOINCREMENT" in db._table_sql(c, "ai_preds").upper()
    assert c.execute("SELECT id, run_id, cls, conf, kind, ref_row, state, reason, ref_box FROM ai_preds ORDER BY id").fetchall() \
        == [(2, 1, 0, .9, "detect", None, "rejected", "r", None), (5, 2, 1, None, "loose", 0, "pending", "", None)]
    idx = {r[0] for r in c.execute("SELECT name FROM sqlite_master WHERE type='index' AND tbl_name='ai_preds'")}
    assert {"idx_ai_preds_img", "idx_ai_preds_ds", "idx_ai_preds_run", "idx_ai_preds_run_state"} <= idx
    assert c.execute("SELECT name FROM sqlite_master WHERE name LIKE '%__new'").fetchall() == []
    assert c.execute("SELECT seq FROM sqlite_sequence WHERE name='ai_preds'").fetchone() == (9,)
    c.execute("PRAGMA foreign_keys=ON")
    ins = "INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h) VALUES(1, 'd', 3, 0, .1, .1, .1, .1)"
    assert c.execute(ins).lastrowid == 10                          # past snapshot_preds' 9, not MAX(id)+1 = 6
    c.execute("DELETE FROM ai_preds WHERE id=10")
    assert c.execute(ins).lastrowid == 11                          # a deleted id is never handed out again
    with pytest.raises(sqlite3.IntegrityError):                    # the foreign keys came along
        c.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h) VALUES(99, 'd', 3, 0, 0, 0, 0, 0)")
    c.execute("DELETE FROM ai_runs WHERE id=2")                    # ... including ON DELETE CASCADE
    assert c.execute("SELECT COUNT(*) FROM ai_preds WHERE run_id=2").fetchone() == (0,)
    assert db.migrate(c) == []                                     # idempotent
    db._ensure_index_shapes(c)                                     # the old, shorter idx_ai_preds_ds gets today's shape
    assert tuple(r[2] for r in c.execute("PRAGMA index_info(idx_ai_preds_ds)")) == db.INDEX_SHAPES["idx_ai_preds_ds"][1]
    c.close()


def test_a_fresh_database_creates_ai_preds_with_autoincrement(tmp_path):
    c = sqlite3.connect(tmp_path / "f.db", isolation_level=None)
    c.executescript(db.SCHEMA)
    assert "AUTOINCREMENT" in db._table_sql(c, "ai_preds").upper()
    assert db.migrate(c) == [name for name, _s, _f in db.MIGRATIONS]   # every step is a no-op on today's shape
    assert c.execute("SELECT COUNT(*) FROM sqlite_sequence WHERE name='ai_preds'").fetchone() == (0,)
    c.close()


def test_moved_files_match_by_trailing_then_leading_path_components():
    from fovea.core.scanner import match_moves
    got = match_moves(["test/c1/a.jpg", "test-2/c1/a.jpg", "train/0001.jpg", "val/0001.jpg", "x/q.jpg", "v/k.jpg"],
                      ["test/x/c1/a.jpg", "test-2/x/c1/a.jpg", "0001.jpg", "q.jpg", "z/q.jpg", "k.jpg"])
    # the split decides between equal trailing paths; two old files claiming one new file, or a tie, move nowhere
    assert got == {"test/c1/a.jpg": "test/x/c1/a.jpg", "test-2/c1/a.jpg": "test-2/x/c1/a.jpg", "v/k.jpg": "k.jpg"}


# --------------------------------------------------------------------------- round 2: undo mode (what an op wrote)

def _lines(p: Path):
    return p.read_text().strip().splitlines() if p.exists() else None


def test_undo_reverts_only_files_still_as_the_op_left_them(scanned):
    """An Objects / bulk / AI write's own Undo must never take back an edit made after it (it used to: the restore
    put every file of the snapshot back). History's full restore still does what it says."""
    from fovea.api.datasets import bulk_labels
    ds_id, root = scanned
    ids = [_iid(ds_id, f"train/f0{k}.jpg") for k in range(3)]
    f = [root / "labels/train" / f"f0{k}.txt" for k in range(3)]
    res = bulk_labels(ds_id, {"op": "set_class", "cls": 1, "image_ids": ids})
    sid = res["snapshot_id"]
    snap = snapshots.get_snapshot(ds_id, sid)
    assert snap["tracked"] == 1 and snap["n_files"] == 3
    assert {r["after_ver"] for r in db.query("SELECT after_ver FROM snapshot_files WHERE snapshot_id=?", (sid,))} \
        == {labels._digest(f[0].read_bytes())}
    _put(ds_id, ids[0], [[1, 0.5, 0.5, 0.2, 0.2], [0, 0.2, 0.2, 0.1, 0.1]])     # Annotate adds a box afterwards
    out = snapshots.restore(ds_id, sid, mode="undo")
    assert out["mode"] == "undo" and out["restored"] == 2 and out["changed_since"] == 1
    assert any("edited after this change" in p for p in out["problems"])
    assert _lines(f[0]) == ["1 0.500000 0.500000 0.200000 0.200000", "0 0.200000 0.200000 0.100000 0.100000"]
    assert _lines(f[1]) == _lines(f[2]) == ["0 0.5 0.5 0.2 0.2"]
    # the undo is itself undoable, and its own undo point holds only the two files it changed
    assert snapshots.get_snapshot(ds_id, out["snapshot_id"])["n_files"] == 2
    again = snapshots.restore(ds_id, out["snapshot_id"], mode="undo")
    assert again["restored"] == 2 and _lines(f[1])[0].startswith("1 ")
    # History (full mode) puts the version back whatever happened since
    full = snapshots.restore(ds_id, sid)
    assert full["mode"] == "full" and full["restored"] == 3 and _lines(f[0]) == ["0 0.5 0.5 0.2 0.2"]


def test_undo_of_an_untracked_snapshot_is_a_full_restore_and_nothing_to_do_says_so(scanned):
    ds_id, root = scanned
    ids = [_iid(ds_id, "train/f00.jpg")]
    sid = snapshots.snapshot_images(ds_id, ids, "test")            # an older kind of version: no write records
    _put(ds_id, ids[0], [[1, 0.3, 0.3, 0.1, 0.1]])
    assert snapshots.get_snapshot(ds_id, sid)["tracked"] == 0
    out = snapshots.restore(ds_id, sid, mode="undo")
    assert out["mode"] == "full" and out["restored"] == 1
    job = Job(id="j", kind="restore")
    out = snapshots.restore(ds_id, sid, job=job)
    assert out["restored"] == 0 and job.message == "Nothing to restore — the label files already match"
    with pytest.raises(ValueError):
        snapshots.restore(ds_id, sid, mode="sideways")


def test_write_log_trims_the_snapshot_to_the_files_written_and_drops_an_empty_one(scanned):
    """An AI 'fill' snapshots every image without a label file up front; the files it never wrote (failed
    images) must not be part of its undo point — undoing it used to delete labels made by hand for those."""
    ds_id, root = scanned
    imgs = [root / "images/val" / f"n{k}.jpg" for k in range(3)]
    for p in imgs:
        _img(p)
    _scan(ds_id)
    new = [_iid(ds_id, f"val/n{k}.jpg") for k in range(3)]
    lps = {r["id"]: r["label_path"] for r in db.query("SELECT id, label_path FROM images WHERE id IN (?,?,?)", new)}
    sid = snapshots.snapshot_images(ds_id, new, "ai_fill")
    wlog = snapshots.WriteLog(sid)
    labels.write_label_file(Path(lps[new[0]]), [[0, 0.5, 0.5, 0.2, 0.2]])
    wlog.wrote(lps[new[0]])
    assert wlog.close() == sid
    assert snapshots.get_snapshot(ds_id, sid)["n_files"] == 1
    _put(ds_id, new[1], [[1, 0.4, 0.4, 0.2, 0.2]])                    # labelled by hand after the fill
    out = snapshots.restore(ds_id, sid, mode="undo")
    assert out["removed"] == 1 and not Path(lps[new[0]]).exists() and Path(lps[new[1]]).exists()
    out = snapshots.restore(ds_id, sid)                              # even History's full restore leaves it alone
    assert Path(lps[new[1]]).exists() and out["restored"] == 0
    empty = snapshots.snapshot_images(ds_id, new, "ai_fill")
    assert snapshots.WriteLog(empty).close() is None and not snapshots.get_snapshot(ds_id, empty)


def test_ai_fill_undo_keeps_a_label_made_by_hand_for_an_image_the_batch_skipped(tmp_path, monkeypatch):
    """Review probe p10 end to end: a fill batch that fails on some images, a hand label on one of those, then the
    AI panel's Undo (restore mode 'undo')."""
    from fastapi.testclient import TestClient
    from fovea.main import app
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    root = tmp_path / "fill"
    for k in range(6):
        _img(root / "images/train" / f"f{k:03d}.jpg", color=(k * 40, 100, 50))
    (root / "labels/train").mkdir(parents=True)
    ds_id = _register(root, classes=["person", "fallen"], source="user")
    _scan(ds_id)
    c = TestClient(app)
    try:
        all_ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY rel_path", (ds_id,))]
        r = c.post("/api/ai/batch", json={"dataset_id": ds_id, "image_ids": all_ids, "provider": "mock", "model": "mock-fast",
                                          "target": "fill", "options": {"mock_error": "bad_output", "mock_error_every": 2}})
        assert r.status_code == 200, r.text
        _wait_jobs(ds_id, 60)
        from fovea.jobs import jobs
        res = jobs.get(r.json()["job"]["id"]).result
        written = [i for i in all_ids if db.query_one("SELECT has_label FROM images WHERE id=?", (i,))["has_label"]]
        skipped = [i for i in all_ids if i not in written]
        assert written and skipped and res["snapshot_id"]
        assert snapshots.get_snapshot(ds_id, res["snapshot_id"])["n_files"] == len(written)
        hand = skipped[0]
        assert c.put(f"/api/datasets/{ds_id}/images/{hand}/labels", json={"boxes": [[0, .5, .5, .3, .6]]}).status_code == 200
        lp = Path(db.query_one("SELECT label_path FROM images WHERE id=?", (hand,))["label_path"])
        j = c.post(f"/api/datasets/{ds_id}/snapshots/{res['snapshot_id']}/restore", json={"mode": "undo"}).json()["job"]
        _wait_jobs(ds_id, 60)
        done = jobs.get(j["id"])
        assert done.status == "done" and done.result["removed"] == len(written), done.to_dict()
        assert lp.exists() and lp.read_text().startswith("0 0.500000")
        assert c.post(f"/api/datasets/{ds_id}/snapshots/{res['snapshot_id']}/restore", json={"mode": "x"}).status_code == 400
    finally:
        _drop(ds_id)


def test_objects_undo_keeps_a_box_added_in_annotate_afterwards(scanned):
    """Review probe p1: Objects reassigns a box (snapshot 1), Annotate then adds a box to the same image; the
    Objects Cmd+Z must not take that box away."""
    from fastapi.testclient import TestClient
    from fovea.main import app
    ds_id, root = scanned
    c = TestClient(app)
    iid = _iid(ds_id, "train/f00.jpg")
    lp = root / "labels/train/f00.txt"
    lab = c.get(f"/api/datasets/{ds_id}/images/{iid}/labels").json()
    r = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": iid, "row": 0, "version": lab["version"], "cls": 1}]}).json()
    sid = r["snapshot_id"]
    assert r["updated"] == 1 and snapshots.get_snapshot(ds_id, sid)["tracked"] == 1
    lab = c.get(f"/api/datasets/{ds_id}/images/{iid}/labels").json()
    assert c.put(f"/api/datasets/{ds_id}/images/{iid}/labels",
                 json={"boxes": lab["boxes"] + [[0, .2, .2, .1, .1]], "base_version": lab["version"]}).status_code == 200
    j = c.post(f"/api/datasets/{ds_id}/snapshots/{sid}/restore", json={"image_ids": [iid], "mode": "undo"}).json()["job"]
    _wait_jobs(ds_id)
    from fovea.jobs import jobs
    done = jobs.get(j["id"])
    assert done.result["changed_since"] == 1 and done.result["restored"] == 0
    assert done.message == "Nothing restored · 1 edited since — kept"
    assert len(_lines(lp)) == 2 and _lines(lp)[0].startswith("1 ")
    # an Objects edit that changed nothing (a stale version) leaves no undo point behind
    r = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": iid, "row": 0, "version": "stale", "cls": 0}]}).json()
    assert r["updated"] == 0 and (r["snapshot_id"] is None or not snapshots.get_snapshot(ds_id, r["snapshot_id"]))


# --------------------------------------------------------------------------- round 2: retention tiers, off-thread

def test_small_edits_never_push_out_a_big_undo_point(scanned, monkeypatch):
    """Review probe p2: 20 one-box Objects edits used to evict the only undo point of a whole-dataset AI write."""
    ds_id, _root = scanned
    monkeypatch.setattr(snapshots, "SMALL_FILES", 2)                # the 6-image dataset: 3+ files is "big"
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    big = snapshots.snapshot_images(ds_id, ids, "ai_replace")
    for _ in range(snapshots.KEEP_SMALL + 5):
        snapshots.snapshot_images(ds_id, ids[:1], "objects")
    assert snapshots.settle()
    snaps = snapshots.list_snapshots(ds_id, 1000)
    assert snapshots.get_snapshot(ds_id, big)
    assert len([s for s in snaps if s["n_files"] <= 2]) == snapshots.KEEP_SMALL
    for _ in range(snapshots.KEEP_AUTO):
        snapshots.snapshot_images(ds_id, ids, "bulk set_class")
    assert snapshots.settle()
    assert not snapshots.get_snapshot(ds_id, big)                  # 20 newer big versions: normal retention
    assert len([s for s in snapshots.list_snapshots(ds_id, 1000) if s["n_files"] > 2]) == snapshots.KEEP_AUTO


def test_pruning_runs_off_the_writers_thread_in_short_transactions(scanned, monkeypatch):
    """Review probe p3: an Objects key press used to cascade-delete an evicted 46k-file snapshot (and scan every
    blob) under the write lock on its own request thread."""
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    called = []
    monkeypatch.setattr(snapshots, "prune", lambda *a, **k: called.append(threading.current_thread().name))
    snapshots.snapshot_images(ds_id, ids[:1], "objects")
    assert snapshots.settle() and called == ["fovea-snapshot-prune"]
    monkeypatch.undo()
    # a dropped snapshot goes in chunks, its blobs with it (only those nothing else holds)
    monkeypatch.setattr(snapshots, "DROP_CHUNK", 2)
    for k, iid in enumerate(ids):
        p = db.query_one("SELECT label_path FROM images WHERE id=?", (iid,))["label_path"]
        Path(p).write_text(f"0 0.{k + 1}1 0.5 0.1 0.1\n")              # every file its own blob
    s = snapshots.snapshot_images(ds_id, ids, "test")
    shas = {r["sha"] for r in db.query("SELECT sha FROM snapshot_files WHERE snapshot_id=?", (s,))}
    statements = []
    conn = db.connect()
    conn.set_trace_callback(lambda q: statements.append(q) if q.startswith("DELETE FROM snapshot_files") else None)
    try:
        assert snapshots.delete_snapshot(ds_id, s)
    finally:
        conn.set_trace_callback(None)
    assert len(statements) >= 3                                    # 6 rows, 2 per transaction
    assert not db.query("SELECT 1 FROM snapshot_files WHERE snapshot_id=?", (s,))
    assert not db.query("SELECT sha FROM label_blobs WHERE sha IN (%s)" % ",".join("?" * len(shas)), list(shas))


def test_save_version_job_ends_with_a_final_message(scanned):
    from fastapi.testclient import TestClient
    from fovea.main import app
    ds_id, _root = scanned
    r = TestClient(app).post(f"/api/datasets/{ds_id}/snapshots", json={"name": "v"}).json()
    _wait_jobs(ds_id)
    from fovea.jobs import jobs
    j = jobs.get(r["job"]["id"])
    assert j.status == "done" and j.message == "Saved 6 label files" and (j.done, j.total) == (6, 6)
    listing = TestClient(app).get(f"/api/datasets/{ds_id}/snapshots").json()
    assert listing["keep_small"] == snapshots.KEEP_SMALL and listing["small_files"] == snapshots.SMALL_FILES


# --------------------------------------------------------------------------- round 2: polygon vs pose, row matching

def test_a_polygon_touching_two_borders_is_not_read_as_a_pose_row():
    poly = "0 0.10 0.60 0.30 0.62 0.32 0.95 0.00 0.98 0.12 1.00"
    r = labels.parse_label_rows(poly)[0][0]
    assert r.kind == "seg" and [round(v, 6) for v in r.box] == [0.16, 0.8, 0.32, 0.4]
    # real pose rows: visible keypoints inside their box, unlabelled ones as 0 0 0; one odd row does not flip a file
    pose = ["0 0.5 0.5 0.2 0.4 0.5 0.4 2 0.45 0.6 1", "0 0.3 0.3 0.2 0.2 0.3 0.3 2 0 0 0",
            "1 0.7 0.7 0.2 0.2 0.7 0.7 1 0.95 0.1 1"]
    rows, _ = labels.parse_label_rows("\n".join(pose))
    assert [r.kind for r in rows] == ["pose", "pose", "pose"] and rows[0].box == [0.5, 0.5, 0.2, 0.4]
    # a pose-shaped row with an odd number of values cannot be a polygon
    assert labels.parse_label_rows("0 0.5 0.5 0.2 0.4 0.5 0.4 2 0.9 0.9 1 0.1 0.1 0")[0][0].kind == "pose"
    assert not labels.looks_like_pose([0.10, 0.60, 0.30, 0.62, 0.32, 0.95, 0.00, 0.98, 0.12, 1.00])
    assert labels.looks_like_pose([0.5, 0.5, 0.2, 0.4, 0.5, 0.4, 2, 0.45, 0.6, 1])


def test_compare_reads_the_border_polygon_prediction_as_a_polygon():
    from fovea.plugins.compare.evaluate import parse_pred_text
    (row,) = parse_pred_text("0 0.10 0.60 0.30 0.62 0.32 0.95 0.00 0.98 0.12 1.00\n")
    assert [round(v, 6) for v in row[1:5]] == [0.16, 0.8, 0.32, 0.4]


def test_two_shapes_with_one_bounding_box_each_keep_their_own_polygon(tmp_path):
    """Review finding: deleting one of two polygons with the same bbox gave the kept object the other's shape."""
    from fovea.api.datasets import bulk_labels
    sq, di = "1 0.2 0.2 0.6 0.2 0.6 0.6 0.2 0.6", "0 0.2 0.4 0.4 0.2 0.6 0.4 0.4 0.6"
    t = f"{sq}\n{di}\n"
    rows, _ = labels.parse_label_rows(t)
    assert labels.label_text_for([r.as_box() for r in rows if r.cls != 1], t) == di + "\n"
    assert labels.label_text_for([r.as_box() for r in rows if r.cls != 0], t) == sq + "\n"
    swapped = labels.label_text_for([[1 - r.cls] + r.box for r in rows], t)
    assert swapped == f"0{sq[1:]}\n1{di[1:]}\n"                     # classes swap in place, shapes stay
    root = tmp_path / "shapes"
    _img(root / "images/train/a.jpg")
    _lbl(root / "labels/train/a.txt", t)
    ds_id = _register(root)
    _scan(ds_id)
    try:
        bulk_labels(ds_id, {"op": "delete_class", "cls": 1, "image_ids": [_iid(ds_id, "train/a.jpg")]})
        assert (root / "labels/train/a.txt").read_text() == di + "\n"
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- round 2: dataset delete, relocate

def test_deleting_a_dataset_hides_it_at_once_and_removes_rows_in_the_background(scanned, monkeypatch):
    from fastapi.testclient import TestClient
    from fovea.api import datasets as api
    from fovea.main import app
    ds_id, _root = scanned
    c = TestClient(app)
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]
    snapshots.snapshot_images(ds_id, ids, "bulk set_class")
    db.execute("INSERT INTO image_flags(image_id, dataset_id, code) VALUES(?,?,?)", (ids[0], ds_id, "blurry"))
    tok = c.post(f"/api/datasets/{ds_id}/sets", json={"ids": ids}).json()["token"]
    monkeypatch.setattr(api, "DELETE_CHUNK", 2)                    # many short transactions
    gate = threading.Event()
    orig = api._purge_dataset
    monkeypatch.setattr(api, "_purge_dataset", lambda d: (gate.wait(10), orig(d)))
    assert c.delete(f"/api/datasets/{ds_id}").json() == {"ok": True}
    assert c.get(f"/api/datasets/{ds_id}").status_code == 404
    assert ds_id not in [d["id"] for d in c.get("/api/datasets").json()["datasets"]]
    assert db.query_one("SELECT status FROM datasets WHERE id=?", (ds_id,))["status"] == "deleting"
    from fovea.core.scanner import _settle_status
    _settle_status(ds_id, False)                                   # a scan that was still stopping
    assert db.query_one("SELECT status FROM datasets WHERE id=?", (ds_id,))["status"] == "deleting"
    gate.set()
    assert api.wait_for_deletes(30)
    for table in ("images", "boxes", "snapshots", "image_flags", "work_sets"):
        assert not db.query(f"SELECT 1 FROM {table} WHERE dataset_id=?", (ds_id,)), table
    assert not db.query("SELECT 1 FROM datasets WHERE id=?", (ds_id,))
    assert not db.query("SELECT 1 FROM work_set_items i JOIN work_sets s ON s.id=i.set_id WHERE s.token=?", (tok,))


def test_fovea_own_folders_can_never_become_or_be_read_through_a_dataset(tmp_path, monkeypatch):
    """Review (security): export into DATA_DIR, register it, read secrets.json through the file browser."""
    from fastapi.testclient import TestClient
    from fovea import config
    from fovea.main import app
    home = tmp_path / "home"
    data, model = home / "fovea-data", home / "fovea-model"
    _ultra(data, name="exported")                                   # an export someone wrote into the data folder
    (data / "secrets.json").write_text('{"anthropic": "sk-ant-api03-SECRET"}')
    model.mkdir(parents=True)
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "MODEL_DIR", model)
    c = TestClient(app)
    for path in (data, data / "exported", home, model):
        r = c.post("/api/datasets", json={"path": str(path)})
        assert r.status_code == 400 and r.json()["detail"]["code"] == "fovea_folder", (path, r.text)
    elsewhere = _ultra(tmp_path, name="elsewhere")
    ds = c.post("/api/datasets", json={"path": str(elsewhere)}).json()["dataset"]["id"]
    try:
        _wait_jobs(ds)
        r = c.post(f"/api/datasets/{ds}/relocate", json={"path": str(data / "exported")})
        assert r.status_code == 400 and r.json()["detail"]["code"] == "fovea_folder"
        # a dataset registered before this was refused (or holding a link into the data folder) still cannot read it
        os.symlink(data, elsewhere / "fd")
        for q in ({"path": "fd/secrets.json"}, {"path": "fd/secrets.json", "raw": 1}):
            r = c.get(f"/api/datasets/{ds}/file", params=q)
            assert r.status_code == 403 and "SECRET" not in r.text
        assert c.get(f"/api/datasets/{ds}/files", params={"path": "fd"}).status_code == 403
        db.execute("UPDATE datasets SET root=? WHERE id=?", (str(home), ds))
        r = c.get(f"/api/datasets/{ds}/file", params={"path": "fovea-data/secrets.json"})
        assert r.status_code == 403 and "SECRET" not in r.text
    finally:
        _drop(ds)


def test_relocating_detaches_an_economy_run_instead_of_cancelling_its_provider_batch(tmp_path, monkeypatch, no_scan_hooks):
    """Review probe p7: relocate used to cancel the Economy job as a user cancel — and with it the provider batch."""
    import shutil
    from fovea.ai import economy, store
    from fovea.api.datasets import relocate_dataset
    from fovea.jobs import jobs
    root = _ultra(tmp_path, name="eco")
    ds_id = _register(root)
    _scan(ds_id)
    seen = {}

    def waiting(job):
        while not job.cancelled:
            time.sleep(0.01)
        seen["detached"] = getattr(job, "detached", False)
        raise JobCancelled()

    resumed = []
    monkeypatch.setattr(economy, "resume", lambda force=False: resumed.append(force) or [])
    try:
        run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {"speed": "economy", "provider_batches": [{"id": "b1"}]})
        job = jobs.submit("ai_batch", waiting, dataset_id=ds_id, message="Economy")
        store.set_run_job(run_id, job.id)
        time.sleep(0.05)
        new = tmp_path / "eco-moved"
        shutil.copytree(root, new)
        relocate_dataset(ds_id, {"path": str(new)})
        _wait_jobs(ds_id)
        assert seen == {"detached": True}
        assert db.query_one("SELECT status FROM ai_runs WHERE id=?", (run_id,))["status"] == "running"
        assert resumed == [True]                                    # re-attached once the rescan finished
    finally:
        _drop(ds_id)


# =========================================================================== round-3 review fixes

def test_a_huge_class_id_never_builds_billions_of_class_names(tmp_path):
    """r3: one row '2000000000 0.5 0.5 0.1 0.1' in a dataset without names made the scan build two billion
    class_N names (RSS 1 GB -> 3 GB in 3 s, then the watchdog killed the server)."""
    from fovea.api.datasets import put_labels
    root = tmp_path / "huge"
    for k, row in enumerate(("0 0.5 0.5 0.1 0.1\n", "2000000000 0.5 0.5 0.1 0.1\n", "40 0.5 0.5 0.1 0.1\n")):
        _img(root / "images" / "train" / f"f{k}.jpg")
        _lbl(root / "labels" / "train" / f"f{k}.txt", row)
    ds_id = _register(root)
    try:
        t0 = time.time()
        _scan(ds_id)
        assert time.time() - t0 < 10
        row = db.query_one("SELECT classes, classes_source FROM datasets WHERE id=?", (ds_id,))
        assert db.loads(row["classes"]) == [f"class_{i}" for i in range(41)] and row["classes_source"] == "inferred"
        huge = _iid(ds_id, "train/f1.jpg")
        assert imagesq.count_images(ds_id, {"issue": "bad_class"}) == 1
        assert imagesq.list_ids(ds_id, {"issue": "bad_class"}) == [huge]
        # revalidation (after any class-list change or rescan) keeps the flag on an inferred list too
        from fovea.core.scanner import revalidate_classes
        revalidate_classes(ds_id)
        assert imagesq.list_ids(ds_id, {"issue": "bad_class"}) == [huge]
        # an editor save of that row re-reads the file: still flagged; the list grows for a near id only
        put_labels(ds_id, huge, {"boxes": [[2000000000, 0.5, 0.5, 0.1, 0.1], [45, 0.3, 0.3, 0.1, 0.1]]})
        assert imagesq.list_ids(ds_id, {"issue": "bad_class"}) == [huge]
        assert len(db.loads(db.query_one("SELECT classes FROM datasets WHERE id=?", (ds_id,))["classes"])) == 46
        names = db.loads(db.query_one("SELECT classes FROM datasets WHERE id=?", (ds_id,))["classes"])
        st = stats.dataset_stats(ds_id, names)
        assert {c["cls"] for c in st["classes"]} >= {0, 40, 2000000000}
        # the cap itself: the highest id that still gets a made-up name is MAX_INFERRED_CLASSES - 1
        cap = labels.MAX_INFERRED_CLASSES
        (root / "labels" / "train" / "f2.txt").write_text(f"{cap - 1} 0.5 0.5 0.1 0.1\n{cap} 0.5 0.5 0.1 0.1\n")
        _scan(ds_id)
        names = db.loads(db.query_one("SELECT classes FROM datasets WHERE id=?", (ds_id,))["classes"])
        assert len(names) == cap and names[-1] == f"class_{cap - 1}"
        assert sorted(imagesq.list_ids(ds_id, {"issue": "bad_class"})) == sorted([huge, _iid(ds_id, "train/f2.jpg")])
    finally:
        _drop(ds_id)


def test_parser_flags_ids_beyond_the_inferred_cap_and_grows_nothing_past_it():
    cap = labels.MAX_INFERRED_CLASSES
    _b, _c, issues = labels.parse_label_text(f"{cap - 1} 0.5 0.5 0.1 0.1\n")
    assert "bad_class" not in issues
    _b, _c, issues = labels.parse_label_text(f"{cap} 0.5 0.5 0.1 0.1\n")
    assert "bad_class" in issues
    _b, _c, issues = labels.parse_label_text(f"{cap + 5} 0.5 0.5 0.1 0.1\n", cap + 10)   # a declared list wins
    assert "bad_class" not in issues
    assert labels.inferred_names(["a"], 3) == ["a", "class_1", "class_2", "class_3"]
    assert len(labels.inferred_names([], 2_000_000_000)) == cap


def test_yaml_names_map_or_nc_never_makes_up_billions_of_names(tmp_path):
    notes = []
    assert layout.parse_names({"names": {0: "a", 1: "b", 2000000000: "c"}}, notes) == ["a", "b"]
    assert notes and "2,000,000,000" in notes[0]
    assert layout.parse_names({"names": {0: "a", 3: "d"}}) == ["a", "class_1", "class_2", "d"]
    notes = []
    assert len(layout.parse_names({"nc": 2_000_000_000}, notes)) == labels.MAX_INFERRED_CLASSES and notes
    assert layout.parse_names({"nc": 3}) == ["class_0", "class_1", "class_2"]
    big = {i: f"n{i}" for i in range(labels.MAX_INFERRED_CLASSES + 10)}                # a real, long list is kept
    assert len(layout.parse_names({"names": big})) == labels.MAX_INFERRED_CLASSES + 10
    root = tmp_path / "y"
    _img(root / "images" / "train" / "a.jpg")
    _lbl(root / "labels" / "train" / "a.txt", "2000000000 0.5 0.5 0.1 0.1\n")
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nnames:\n  0: person\n  2000000000: ghost\n")
    t0 = time.time()
    lay = layout.detect(str(root))
    assert time.time() - t0 < 5 and lay.classes == ["person"] and any("2,000,000,000" in n for n in lay.notes)


def test_bulk_ops_and_editor_saves_never_grow_an_inferred_list_past_the_cap():
    from fovea.api.datasets import _class_limit
    cap = labels.MAX_INFERRED_CLASSES
    row = {"classes": db.dumps([f"class_{i}" for i in range(cap - 4)]), "classes_source": "inferred"}
    assert _class_limit(row) == cap - 1
    assert _class_limit({"classes": "[]", "classes_source": "inferred"}) == 31
    assert _class_limit({"classes": db.dumps(["a", "b"]), "classes_source": "user"}) == 1


def _seed_flags(ds_id, scores):
    db.execute("DELETE FROM image_flags WHERE dataset_id=?", (ds_id,))
    for iid, code, score in scores:
        db.execute("INSERT INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?, '{}')",
                   (iid, ds_id, code, score))


def test_flag_sort_puts_the_most_suspect_first_in_every_query(scanned):
    """r3: ?issue=dup_label_mismatch listed 734 images in index order although the flags carry a score."""
    ds_id, _root = scanned
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY id", (ds_id,))]
    a, b, c, d, e, g = ids
    _seed_flags(ds_id, [(a, "dup_label_mismatch", 0.33), (b, "dup_label_mismatch", 1.0), (c, "dup_label_mismatch", 0.67),
                        (d, "dup_label_mismatch", 1.0), (e, "near_dup", 0.95), (a, "near_dup", 0.99)])
    f = {"issue": "dup_label_mismatch"}
    want = [b, d, c, a]                     # score desc; equal scores keep index order
    assert imagesq.list_ids(ds_id, f, "flag") == want
    assert imagesq.list_ids(ds_id, f, "flag", "asc") == [a, c, b, d]
    _t, items = imagesq.list_images(ds_id, f, 0, 2, "flag")
    _t2, rest = imagesq.list_images(ds_id, f, 0, 10, "flag", after_id=items[-1]["id"], with_total=False)
    assert [it["id"] for it in items + rest] == want
    for order in (None, "asc"):
        seq = imagesq.list_ids(ds_id, f, "flag", order)
        for pos, iid in enumerate(seq):
            assert imagesq.position(ds_id, f, iid, "flag", order) == pos
            nxt = imagesq.neighbor(ds_id, f, iid, "next", "flag", order)
            prv = imagesq.neighbor(ds_id, f, iid, "prev", "flag", order)
            assert (nxt["id"] if nxt else None) == (seq[pos + 1] if pos + 1 < len(seq) else None)
            assert (prv["id"] if prv else None) == (seq[pos - 1] if pos else None)
    # the key follows the filter: near_dup scores for issue=near_dup, every flag without one (unflagged last)
    assert imagesq.list_ids(ds_id, {"issue": "near_dup"}, "flag") == [a, e]
    assert imagesq.list_ids(ds_id, {}, "flag") == [b, d, a, e, c, g]
    assert imagesq.list_ids(ds_id, {"issue": "near_dup,dup_label_mismatch"}, "flag") == [b, d, a, e, c]
    # AI audit kinds sort by the image's audit score, mixed with flags by the higher of the two
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'audit', ?)", (ds_id, db.now())).lastrowid
    try:
        for iid, score in ((g, 0.4), (c, 0.9)):
            db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind, ref_row) "
                       "VALUES(?,?,?,0,.5,.5,.1,.1,'missing',0)", (run, ds_id, iid))
            db.execute("INSERT INTO ai_audit(image_id, dataset_id, run_id, score) VALUES(?,?,?,?)", (iid, ds_id, run, score))
        assert imagesq.list_ids(ds_id, {"issue": "ai_missing"}, "flag") == [c, g]
        assert imagesq.list_ids(ds_id, {"issue": "ai_missing,near_dup"}, "flag") == [a, e, c, g]
    finally:
        db.execute("DELETE FROM ai_audit WHERE run_id=?", (run,))
        db.execute("DELETE FROM ai_runs WHERE id=?", (run,))
    # over HTTP, with the order the server picks for the key (descending) and an explicit one
    from fastapi.testclient import TestClient
    from fovea.main import app
    cl = TestClient(app)
    r = cl.get(f"/api/datasets/{ds_id}/images", params={"issue": "dup_label_mismatch", "sort": "flag"})
    assert r.status_code == 200 and [it["id"] for it in r.json()["items"]] == want
    r = cl.get(f"/api/datasets/{ds_id}/images/ids", params={"issue": "dup_label_mismatch", "sort": "flag", "order": "desc"})
    assert r.json()["ids"] == want
    r = cl.get(f"/api/datasets/{ds_id}/images/position", params={"issue": "dup_label_mismatch", "sort": "flag", "id": a})
    assert r.json()["position"] == 3
    _seed_flags(ds_id, [])


def test_planner_statistics_follow_rows_per_run_not_only_the_row_count(scanned):
    """r3: ai_preds analysed while it held ONE run of 50k rows kept 'rows per run_id = 50000' after forty small runs
    (the row count had not doubled), so Compare's runs list scanned ai_preds per run (126 ms -> 0.1 ms with fresh stats)."""
    ds_id, _root = scanned
    t0 = time.time()
    while db._stats_state["running"] and time.time() - t0 < 10:
        time.sleep(0.05)
    db._stats_state["at"] = time.monotonic() + 3600
    conn = db.connect()
    iid = _iid(ds_id, "train/f00.jpg")
    runs = []
    insert = ("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind, state) "
              "VALUES(?,?,?,0,.5,.5,.1,.1,'detect','info')")
    query = """SELECT r.id FROM ai_runs r WHERE r.dataset_id=? AND r.kind NOT IN ('propose','interp','prev_frame')
               AND r.status != 'deleted' AND (r.kind = 'audit' OR EXISTS (SELECT 1 FROM ai_preds p WHERE p.run_id=r.id
               AND p.kind IN ('detect','refine'))) ORDER BY r.created_at DESC LIMIT 200"""
    try:
        base = conn.execute("SELECT COUNT(*) FROM ai_preds").fetchone()[0]
        runs.append(db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'batch', ?)", (ds_id, db.now())).lastrowid)
        conn.executemany(insert, [(runs[0], ds_id, iid)] * 20000)
        conn.execute("ANALYZE ai_preds")
        for k in range(30):
            runs.append(db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'batch', ?)",
                                   (ds_id, db.now() + k + 1)).lastrowid)
            conn.executemany(insert, [(runs[-1], ds_id, iid)] * 500)
        n = conn.execute("SELECT COUNT(*) FROM ai_preds").fetchone()[0]
        assert n < 2 * (base + 20000) + 500                    # the row-count test alone would not re-analyse
        changed = db.refresh_planner_stats(["ai_preds", "ai_audit", "boxes"])     # what an AI job calls at its end
        assert "ai_preds" in changed
        plan = " ".join(r[3] for r in conn.execute("EXPLAIN QUERY PLAN " + query, (ds_id,)))
        assert "USING INDEX idx_ai_preds_run" in plan and "SCAN p" not in plan, plan
        assert db.refresh_planner_stats(["ai_preds"]) == []                       # settled: no re-analyse loop
        assert "ai_runs" in db.VOLATILE_TABLES
    finally:
        for r in runs:
            conn.execute("DELETE FROM ai_preds WHERE run_id=?", (r,))
            conn.execute("DELETE FROM ai_runs WHERE id=?", (r,))
        db.refresh_planner_stats()
        db._stats_state["at"] = 0.0


# =========================================================================== round-4 review fixes

def test_image_filters_fail_closed_on_reads_and_every_bulk_writer(scanned):
    """r4: a filter value the server does not understand (ids=bogus, review=approved,bogus, ai=Pending…) was dropped,
    so GET /images/ids listed every image and labels/bulk, PUT /review, proposals accept/reject acted on the whole
    dataset. Now it is a 400 {code:'bad_filter', key, value}; known values work in any case, review takes CSV."""
    from fastapi.testclient import TestClient
    from fovea.main import app
    ds_id, root = scanned
    c = TestClient(app)
    base = f"/api/datasets/{ds_id}"
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY rel_path", (ds_id,))]
    before = {p: p.read_text() for p in (root / "labels").rglob("*.txt")}
    for key, value in (("ids", "bogus"), ("ids", "-1"), ("ids", "img_a,img_b"), ("review", "bogus"),
                       ("review", "approved,bogus"), ("ai", "bogus"), ("ai", "pending,audit"), ("labeled", "Yes"),
                       ("cls", "bogus"), ("only_cls", "x"), ("exclude_review", "bogus"), ("min_boxes", "bogus"),
                       ("max_boxes", "2.5"), ("has_label", "maybe")):
        for url, extra in (("/images/ids", {}), ("/images", {}), ("/images/facets", {}), ("/images/neighbor", {"id": ids[0]}),
                           ("/images/position", {"id": ids[0]}), ("/boxes", {})):
            if url == "/boxes" and key == "cls":
                continue                    # Objects reads cls itself (box classes)
            r = c.get(base + url, params={key: value, **extra})
            assert r.status_code == 400, (key, value, url, r.text)
            d = r.json()["detail"]
            assert d["code"] == "bad_filter" and d["key"] == key and d["value"] == value and key in d["message"]
    assert c.get(base + "/boxes", params={"cls": "²"}).status_code == 200     # Objects' cls = box classes, its own parser
    # Known values: any case and spacing; review= takes several (either matches); has_label=0 means "no label file".
    c.put(base + "/review", json={"image_ids": ids[:2], "status": "approved"})
    c.put(base + "/review", json={"image_ids": ids[2:3], "status": "excluded"})
    n = lambda **q: c.get(base + "/images/ids", params=q).json()["total"]  # noqa: E731
    assert n(review=" Approved ") == 2 and n(review="approved,excluded") == 3 and n(review="none,excluded") == 4
    assert n(review="any,none") == 6 and n(ai="PENDING") == 0 and n(labeled="1") == 6 and n(has_label="0") == 0
    assert c.get(base + "/images/ids", params={"review": "Approved"}).json()["filters"]["review"] == "approved"
    # Bulk writers: a bad filter is refused before anything is written.
    for path, body in (
            ("/labels/bulk", {"op": "set_class", "cls": 1, "only_cls": 0, "filters": {"ids": "img_a,img_b"}}),
            ("/labels/bulk", {"op": "clear", "filters": {"ids": [[1]]}}),
            ("/labels/bulk", {"op": "clear", "filters": {"ids": []}}),
            ("/labels/bulk", {"op": "clear", "filters": {"review": {"x": 1}}}),
            ("/proposals/accept", {"filters": {"ai": "bogus"}}),
            ("/proposals/reject", {"filters": {"labeled": "yes"}}),
            ("/sets", {"filters": {"exclude_review": "nope"}}),
            ("/export/preview", {"filters": {"min_boxes": "many"}})):
        r = c.post(base + path, json=body)
        assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_filter", (path, r.text)
    for body in ({"status": "flagged", "note": "check", "filters": {"review": "approved,bogus"}},
                 {"status": "approved", "filters": {"ai": "bogus"}}, {"status": "approved", "filters": "bogus"},
                 {"status": "approved", "filters": ["x"]}):
        r = c.put(base + "/review", json=body)
        assert r.status_code == 400 and r.json()["detail"]["code"] == "bad_filter", (body, r.text)
    # An explicit selection that holds no usable id selects nothing; it never falls through to the filters.
    assert c.put(base + "/review", json={"status": "flagged", "image_ids": ["x"], "filters": {}}).json()["updated"] == 0
    assert c.post(base + "/labels/bulk", json={"op": "clear", "image_ids": [], "filters": {}}).json()["changed"] == 0
    assert {p: p.read_text() for p in (root / "labels").rglob("*.txt")} == before
    assert db.query_one("SELECT COUNT(*) n FROM reviews WHERE dataset_id=? AND status='flagged'", (ds_id,))["n"] == 0
    # CSV review in a write acts on exactly those images.
    r = c.put(base + "/review", json={"status": "flagged", "filters": {"review": "approved,excluded"}})
    assert r.json()["updated"] == 3 and n(review="flagged") == 3 and n(review="none") == 3
    c.close()


def _accept_setup(ds_id):
    """Pending proposals on four images: A gets a new box (+ a duplicate of its box), B only a duplicate, C a new box,
    D one below the accept threshold."""
    A, B, C, D = (_iid(ds_id, r) for r in ("train/f00.jpg", "train/f01.jpg", "train/f02.jpg", "val/f00.jpg"))
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'batch', ?)", (ds_id, db.now())).lastrowid
    ins = ("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, reason) "
           "VALUES(?,?,?,?,?,?,?,?,?,'detect',?)")
    p = {}
    for name, iid, cls, box, conf in (("a_new", A, 1, (.2, .2, .1, .1), .9), ("a_dup", A, 0, (.5, .5, .2, .2), .8),
                                      ("b_dup", B, 0, (.5, .5, .2, .2), .7), ("c_new", C, 1, (.8, .8, .1, .1), .9),
                                      ("d_low", D, 1, (.2, .2, .1, .1), .1)):
        p[name] = db.execute(ins, (run, ds_id, iid, cls, *box, conf, f"model: {name}")).lastrowid
    return (A, B, C, D), p


def _states(p):
    return {k: db.query_one("SELECT state FROM ai_preds WHERE id=?", (v,))["state"] for k, v in p.items()}


def test_restoring_an_accept_snapshot_puts_its_proposals_back_to_pending(scanned):
    """r4: History → Restore of the snapshot a bulk accept took put the label files back but left every proposal
    'accepted' — the AI results were lost (re-running a cloud model bills again). The accept job now records what it
    accepted with the snapshot; a restore (full, partial or undo) makes those pending again — also on images whose
    proposals were only duplicates (no file in the snapshot) — and its own undo point accepts them again."""
    from fastapi.testclient import TestClient
    from fovea.jobs import jobs
    from fovea.main import app
    ds_id, root = scanned
    c = TestClient(app)
    base = f"/api/datasets/{ds_id}"
    (A, B, C, D), p = _accept_setup(ds_id)
    la, lc = root / "labels/train/f00.txt", root / "labels/train/f02.txt"
    orig_a, orig_c = la.read_text(), lc.read_text()
    j = c.post(base + "/proposals/accept", json={"image_ids": [A, B, C, D], "min_conf": 0.25}).json()["job"]
    _wait_jobs(ds_id, 60)
    res = jobs.get(j["id"]).result
    sid = res["snapshot_id"]
    assert sid and res["added"] == 2 and res["duplicates"] == 2
    assert _states(p) == {"a_new": "accepted", "a_dup": "accepted", "b_dup": "accepted", "c_new": "accepted", "d_low": "pending"}
    assert snapshots.get_snapshot(ds_id, sid)["n_files"] == 2                  # A and C were written; B was not
    assert db.query_one("SELECT COUNT(*) n FROM snapshot_preds WHERE snapshot_id=?", (sid,))["n"] == 4
    after_a = la.read_text()
    assert after_a != orig_a
    assert c.get(f"{base}/snapshots/{sid}/diff").json()["summary"]["proposals"] == 4
    # C is edited by hand after the accept: the op's own Undo keeps that edit, and C's proposal stays accepted.
    lc.write_text(lc.read_text() + "0 0.3 0.7 0.1 0.1\n")
    j = c.post(f"{base}/snapshots/{sid}/restore", json={"mode": "undo"}).json()["job"]
    _wait_jobs(ds_id)
    done = jobs.get(j["id"])
    assert done.result["restored"] == 1 and done.result["changed_since"] == 1, done.result
    assert done.result["proposals"] == {"pending": 3} and "3 AI proposals pending again" in done.message
    assert la.read_text() == orig_a
    assert _states(p) == {"a_new": "pending", "a_dup": "pending", "b_dup": "pending", "c_new": "accepted", "d_low": "pending"}
    row = db.query_one("SELECT reason, decided_at FROM ai_preds WHERE id=?", (p["b_dup"],))
    assert row["reason"] == "model: b_dup" and row["decided_at"] is None     # not 'already labeled' any more
    assert c.get(f"{base}/images/ids", params={"ai": "pending"}).json()["total"] == 3
    # Undoing the restore (its own undo point) writes A's accepted box back and accepts A's and B's proposals again.
    undo_id = done.result["snapshot_id"]
    j = c.post(f"{base}/snapshots/{undo_id}/restore", json={}).json()["job"]
    _wait_jobs(ds_id)
    done = jobs.get(j["id"])
    assert done.result["proposals"] == {"accepted": 3} and la.read_text() == after_a
    assert _states(p)["a_new"] == _states(p)["b_dup"] == "accepted"
    assert db.query_one("SELECT reason FROM ai_preds WHERE id=?", (p["b_dup"],))["reason"] == "already labeled"
    # A partial History restore of C alone: C's file goes back as it was before the accept, its proposal is pending.
    j = c.post(f"{base}/snapshots/{sid}/restore", json={"image_ids": [C]}).json()["job"]
    _wait_jobs(ds_id)
    assert jobs.get(j["id"]).result["proposals"] == {"pending": 1} and lc.read_text() == orig_c
    assert _states(p) == {"a_new": "accepted", "a_dup": "accepted", "b_dup": "accepted", "c_new": "pending", "d_low": "pending"}
    # A full restore when the files already match still undoes the decisions; a second one has nothing left to do.
    la.write_text(orig_a)
    j = c.post(f"{base}/snapshots/{sid}/restore", json={}).json()["job"]
    _wait_jobs(ds_id)
    assert jobs.get(j["id"]).result["proposals"] == {"pending": 3}
    assert set(_states(p).values()) == {"pending"}
    j = c.post(f"{base}/snapshots/{sid}/restore", json={}).json()["job"]
    _wait_jobs(ds_id)
    assert jobs.get(j["id"]).result["proposals"] == {}
    # Deleting the version drops what it recorded.
    assert c.delete(f"{base}/snapshots/{sid}").status_code == 200
    assert db.query_one("SELECT COUNT(*) n FROM snapshot_preds WHERE snapshot_id=?", (sid,))["n"] == 0
    c.close()


def test_an_accept_of_duplicates_only_leaves_no_undo_point(scanned):
    """Nothing written -> no snapshot (as before); its recorded decisions go with it."""
    from fastapi.testclient import TestClient
    from fovea.jobs import jobs
    from fovea.main import app
    ds_id, _root = scanned
    c = TestClient(app)
    (_A, B, _C, _D), p = _accept_setup(ds_id)
    j = c.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [B]}).json()["job"]
    _wait_jobs(ds_id, 60)
    assert jobs.get(j["id"]).result["snapshot_id"] is None and _states(p)["b_dup"] == "accepted"
    assert db.query_one("SELECT COUNT(*) n FROM snapshot_preds s LEFT JOIN snapshots n ON n.id = s.snapshot_id "
                        "WHERE n.id IS NULL")["n"] == 0
    c.close()
