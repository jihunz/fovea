"""Fast unit tests for Fovea core logic (no server). Run: python -m pytest src/tests -q"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea.core import layout  # noqa: E402
from fovea.core.labels import iou, parse_label_text, sanitize_boxes, write_label_file, read_label_file  # noqa: E402
from fovea.plugins.compare.evaluate import average_precision, match_image  # noqa: E402


def _img(path: Path, size=(64, 48)):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, (120, 120, 120)).save(path)


def test_parse_label_text_issues():
    boxes, confs, issues = parse_label_text("0 0.5 0.5 0.2 0.2\n1 0.9 0.5 0.4 0.2 0.77\nbad line\n", n_classes=2)
    assert len(boxes) == 2 and confs[1] == 0.77 and confs[0] is None
    assert "bad_format" in issues and "out_of_range" in issues
    _, _, empty = parse_label_text("")
    assert empty == {"empty_label"}
    _, _, dup = parse_label_text("0 0.5 0.5 0.2 0.2\n0 0.5 0.5 0.2 0.2\n")
    assert "duplicate_box" in dup
    _, _, badcls = parse_label_text("7 0.5 0.5 0.2 0.2\n", n_classes=3)
    assert "bad_class" in badcls


def test_sanitize_and_roundtrip(tmp_path):
    boxes = sanitize_boxes([[0, 0.05, 0.5, 0.2, 0.2], [1, 0.5, 0.5, 0, 0.1], [2.0, 0.95, 0.5, 0.3, 0.3], [0, 1.4, 0.5, 0.3, 0.3]])
    assert len(boxes) == 2  # zero-area and fully-outside boxes dropped, partial box clamped
    assert boxes[1][1] + boxes[1][3] / 2 <= 1.0 + 1e-9
    assert boxes[0][1] - boxes[0][3] / 2 >= 0  # clamped to image
    out = tmp_path / "a.txt"
    write_label_file(out, boxes)
    back, _, issues = read_label_file(out)
    assert len(back) == 2 and not issues


def test_iou():
    assert iou([0, .5, .5, .2, .2], [0, .5, .5, .2, .2]) == pytest.approx(1.0)
    assert iou([0, .2, .2, .2, .2], [0, .8, .8, .2, .2]) == 0.0


def test_layout_detection(tmp_path):
    # ultralytics: images/<split> + labels/<split>
    root = tmp_path / "ds"
    for split in ("train", "val"):
        _img(root / "images" / split / "a.jpg")
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    lay = layout.detect(str(root))
    assert lay.kind == "ultralytics" and {s.split for s in lay.sources} == {"train", "val"}
    # data.yaml takes precedence and provides names
    (root / "data.yaml").write_text("path: .\ntrain: images/train\nval: images/val\nnames:\n  0: person\n  1: fall\n")
    lay = layout.detect(str(root))
    assert lay.kind == "yaml" and lay.classes == ["person", "fall"] and len(lay.sources) == 2
    # split-first
    root2 = tmp_path / "ds2"
    _img(root2 / "train" / "images" / "x.png")
    (root2 / "train" / "labels").mkdir(parents=True)
    lay = layout.detect(str(root2))
    assert lay.kind == "split-first" and lay.sources[0].label_dir.endswith("train/labels")
    # bare folder
    root3 = tmp_path / "ds3"
    _img(root3 / "img1.jpg")
    lay = layout.detect(str(root3))
    assert lay.kind == "bare" and lay.sources[0].label_dir.endswith("labels")
    # list file
    lst = tmp_path / "train.txt"
    lst.write_text(str(root / "images" / "train" / "a.jpg") + "\n")
    lay = layout.detect(str(lst))
    assert lay.kind == "list" and lay.sources[0].split == "train"


def test_matching_and_ap():
    gt = [[0, .5, .5, .2, .2], [0, .2, .2, .1, .1]]
    preds = [[0, .5, .5, .21, .19, .9], [0, .8, .8, .1, .1, .6], [1, .2, .2, .1, .1, .7]]
    pred_tp, gt_matched = match_image(gt, preds, 0.5)
    assert pred_tp == [True, False, False] and gt_matched == [0, -1]
    assert average_precision([(0.9, True), (0.8, False), (0.7, True)], 2) == pytest.approx(0.8333, abs=1e-3)
    assert average_precision([], 3) == 0.0


# --------------------------------------------------------------------------- unreachable datasets
def _mini_dataset(tmp_path):
    """A tiny ultralytics-layout dataset on disk."""
    root = tmp_path / "ds"
    for split in ("train", "val"):
        _img(root / "images" / split / "a.jpg")
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    return root


def test_scan_refuses_to_wipe_an_unreachable_dataset(tmp_path, monkeypatch):
    """An index whose files have vanished (unmounted drive, moved folder, container paths opened on
    the host) must fail loudly and keep its rows — never report success with 0 images."""
    import uuid
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    db.init_db()
    root = _mini_dataset(tmp_path)
    lay = layout.detect(str(root))
    ds_id = "t-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute(
        "INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps([]), "inferred", "", "new", now, now),
    )
    try:
        first = scan_dataset(ds_id, Job(id="j1", kind="scan"))
        assert first["images"] == 2

        # the whole dataset disappears
        root.rename(tmp_path / "moved-away")
        with pytest.raises(FileNotFoundError):
            scan_dataset(ds_id, Job(id="j2", kind="scan"))

        kept = db.query_one("SELECT COUNT(*) c FROM images WHERE dataset_id=?", (ds_id,))
        assert kept["c"] == 2, "an unreachable rescan must not delete the index"
        row = db.query_one("SELECT status FROM datasets WHERE id=?", (ds_id,))
        assert row["status"] == "unreachable"

        # and it recovers once the files are back
        (tmp_path / "moved-away").rename(root)
        again = scan_dataset(ds_id, Job(id="j3", kind="scan"))
        assert again["images"] == 2
    finally:
        db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_check_reachable_samples_across_the_id_range(tmp_path):
    """A few deleted files at the start of a dataset must not be reported as "everything is gone"."""
    import uuid
    from fovea import db
    from fovea.api.common import check_reachable
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    db.init_db()
    root = tmp_path / "many"
    (root / "labels").mkdir(parents=True, exist_ok=True)
    for i in range(12):
        _img(root / "images" / f"{i:03d}.jpg")
    lay = layout.detect(str(root))
    ds_id = "r-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute(
        "INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps([]), "inferred", "", "new", now, now),
    )
    try:
        scan_dataset(ds_id, Job(id="j", kind="scan"))
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        assert check_reachable(row)["ok"] is True

        # delete only the first two files — the dataset is still overwhelmingly present
        for i in range(2):
            (root / "images" / f"{i:03d}.jpg").unlink()
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        probe = check_reachable(row)
        assert probe["ok"] is True, "a biased head-of-list sample must not raise a false alarm"

        # now delete everything
        for p in (root / "images").iterdir():
            p.unlink()
        row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
        assert check_reachable(row)["ok"] is False
    finally:
        db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_partial_unreachable_keeps_the_missing_split(tmp_path):
    """One split vanishing must not take the others down, and must not erase its own rows either."""
    import uuid
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    db.init_db()
    root = _mini_dataset(tmp_path)          # train + val
    lay = layout.detect(str(root))
    ds_id = "p-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute(
        "INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps([]), "inferred", "", "new", now, now),
    )
    try:
        assert scan_dataset(ds_id, Job(id="a", kind="scan"))["images"] == 2

        # only the val split disappears
        (root / "images" / "val" / "a.jpg").unlink()
        (root / "images" / "val").rmdir()
        scan_dataset(ds_id, Job(id="b", kind="scan"))

        splits = {r["split"] for r in db.query("SELECT DISTINCT split FROM images WHERE dataset_id=?", (ds_id,))}
        assert "train" in splits, "the reachable split must still be indexed"
        assert "val" in splits, "the unreachable split must keep its existing rows, not be erased"
    finally:
        db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_write_label_file_is_a_noop_when_nothing_changed(tmp_path):
    """Opening an image must never rewrite its label file — not its bytes, not its mtime, and not
    the precision the annotations were authored with."""
    import time
    p = tmp_path / "a.txt"

    # a file authored elsewhere with full float precision
    original = "1 0.5515613555908203 0.5003909468650818 0.09941123425960541 0.08842962980270386\n"
    p.write_text(original)
    before_mtime = p.stat().st_mtime
    time.sleep(0.01)

    boxes, _c, _i = read_label_file(p)
    write_label_file(p, boxes)                       # round-trip with no edits
    assert p.read_text() == original, "a no-op save must not reformat the file"
    assert p.stat().st_mtime == before_mtime, "a no-op save must not touch mtime"

    # a real edit still writes
    boxes[0][1] = 0.25
    write_label_file(p, boxes)
    assert p.read_text() != original
    assert read_label_file(p)[0][0][1] == pytest.approx(0.25)


# --------------------------------------------------------------------------- scanner hardening (audit)
def _register(tmp_root, ds_prefix="s"):
    import uuid
    from fovea import db
    db.init_db()
    lay = layout.detect(str(tmp_root))
    ds_id = f"{ds_prefix}-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute(
        "INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
        " VALUES(?,?,?,?,?,?,?,?,?,?)",
        (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps([]), "inferred", "", "new", now, now),
    )
    return ds_id


def _drop(ds_id):
    from fovea import db
    db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_zero_padded_frame_numbers_never_borrow_another_frames_label(tmp_path):
    """cam_00000099_left has no label. It must NOT be matched to cam_00000007_left.txt — the next edit
    would overwrite frame 7's ground truth."""
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    root = tmp_path / "seq"
    for n in (7, 42, 99):
        _img(root / "images" / f"cam_{n:08d}_left.jpg")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "cam_00000007_left.txt").write_text("0 0.1 0.1 0.05 0.05\n")
    (root / "labels" / "cam_00000042_left.txt").write_text("1 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)
    try:
        scan_dataset(ds_id, Job(id="f", kind="scan"))
        row = db.query_one("SELECT has_label, label_path, issues FROM images WHERE dataset_id=? AND rel_path LIKE ?",
                           (ds_id, "%cam_00000099_left.jpg"))
        assert row["has_label"] == 0, "an unlabeled frame must not inherit another frame's label"
        assert row["label_path"].endswith("cam_00000099_left.txt")
        assert "missing_label" in row["issues"]
    finally:
        _drop(ds_id)


def test_hash_fallback_refuses_a_label_that_belongs_to_another_image(tmp_path):
    """img_aaaaaaaa_x has its own label; img_bbbbbbbb_x has none. Both normalise to img_x, but the only
    candidate already has an owner, so it must stay unmatched."""
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    root = tmp_path / "h"
    _img(root / "images" / "img_aaaaaaaa_x.jpg")
    _img(root / "images" / "img_bbbbbbbb_x.jpg")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "img_aaaaaaaa_x.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)
    try:
        scan_dataset(ds_id, Job(id="h", kind="scan"))
        rows = {r["rel_path"]: r for r in db.query("SELECT rel_path, has_label, label_path FROM images WHERE dataset_id=?", (ds_id,))}
        b = rows["img_bbbbbbbb_x.jpg"]
        assert b["has_label"] == 0
        assert b["label_path"].endswith("img_bbbbbbbb_x.txt")
    finally:
        _drop(ds_id)


def test_hash_fallback_still_matches_the_one_unambiguous_case(tmp_path):
    """The feature itself keeps working: a single image whose label differs only by its hash."""
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    root = tmp_path / "ok"
    _img(root / "images" / "clip_1a2b3c4d_frame.jpg")
    (root / "labels").mkdir(parents=True)
    (root / "labels" / "clip_9f8e7d6c_frame.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    ds_id = _register(root)
    try:
        scan_dataset(ds_id, Job(id="ok", kind="scan"))
        r = db.query_one("SELECT has_label, n_boxes, label_path FROM images WHERE dataset_id=?", (ds_id,))
        assert r["has_label"] == 1 and r["n_boxes"] == 1
        assert r["label_path"].endswith("clip_9f8e7d6c_frame.txt")
    finally:
        _drop(ds_id)


def test_empty_listing_never_wipes_an_existing_index(tmp_path):
    """Folders that still exist but list nothing (a mount that came back empty) must not prune."""
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    root = _mini_dataset(tmp_path)
    ds_id = _register(root)
    try:
        assert scan_dataset(ds_id, Job(id="e1", kind="scan"))["images"] == 2
        for split in ("train", "val"):
            (root / "images" / split / "a.jpg").unlink()
        with pytest.raises(FileNotFoundError):
            scan_dataset(ds_id, Job(id="e2", kind="scan"))
        assert db.query_one("SELECT COUNT(*) c FROM images WHERE dataset_id=?", (ds_id,))["c"] == 2
    finally:
        _drop(ds_id)


def test_unreadable_folder_is_treated_as_unreachable_not_empty(tmp_path):
    import os
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    if os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    root = _mini_dataset(tmp_path)
    ds_id = _register(root)
    val = root / "images" / "val"
    try:
        scan_dataset(ds_id, Job(id="p1", kind="scan"))
        val.chmod(0)
        scan_dataset(ds_id, Job(id="p2", kind="scan"))           # train still readable
        splits = {r["split"] for r in db.query("SELECT DISTINCT split FROM images WHERE dataset_id=?", (ds_id,))}
        assert splits == {"train", "val"}, "rows of a folder we could not read must be kept"
    finally:
        val.chmod(0o755)
        _drop(ds_id)


def test_cancelled_scan_restores_a_resting_status(tmp_path):
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job, JobCancelled

    root = _mini_dataset(tmp_path)
    ds_id = _register(root)
    try:
        scan_dataset(ds_id, Job(id="c1", kind="scan"))
        job = Job(id="c2", kind="scan")
        job.cancel()                                              # cancelled before listing finishes
        with pytest.raises(JobCancelled):
            scan_dataset(ds_id, job)
        row = db.query_one("SELECT status, image_count FROM datasets WHERE id=?", (ds_id,))
        assert row["status"] == "ready", f"a cancel must not leave status={row['status']!r}"
        assert row["image_count"] == 2
    finally:
        _drop(ds_id)


def test_undecodable_filenames_are_skipped_not_fatal(tmp_path, monkeypatch):
    """Linux filenames are bytes; one that is not UTF-8 used to abort the whole scan."""
    from fovea.core import scanner

    root = _mini_dataset(tmp_path)
    lay = layout.detect(str(root))
    src = next(s for s in lay.sources if s.split == "train")
    real_walk = os.walk

    def fake_walk(top, **kw):
        for r, d, f in real_walk(top, **kw):
            yield r, d, f + [os.fsdecode(b"caf\xe9.jpg")]
    monkeypatch.setattr(scanner.os, "walk", fake_walk)

    skipped = scanner.Counter()
    got = list(scanner.enumerate_source(src, [], skipped))
    assert [g[0] for g in got] == ["train/a.jpg"]
    assert skipped["undecodable_name"] == 1


# ---------------------------------------------------------------- label editing API


def _scanned(tmp_path):
    from fovea import db
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job

    root = _mini_dataset(tmp_path)
    ds_id = _register(root)
    scan_dataset(ds_id, Job(id="lbl", kind="scan"))
    img = db.query_one("SELECT id, label_path FROM images WHERE dataset_id=? ORDER BY id LIMIT 1", (ds_id,))
    return ds_id, img["id"], Path(img["label_path"])


def test_label_save_refuses_to_overwrite_a_file_changed_elsewhere(tmp_path):
    from fastapi import HTTPException
    from fovea.api.datasets import get_labels, put_labels

    ds_id, iid, lp = _scanned(tmp_path)
    try:
        loaded = get_labels(ds_id, iid)
        assert loaded["version"] and loaded["boxes"] == [[0, 0.5, 0.5, 0.2, 0.2]]

        res = put_labels(ds_id, iid, {"boxes": [[0, 0.4, 0.4, 0.2, 0.2]], "base_version": loaded["version"]})
        assert res["version"] != loaded["version"] and res["rejected"] == 0

        # another writer changes the file; an editor still holding the first version must be refused
        lp.write_text("1 0.1 0.1 0.1 0.1\n")
        with pytest.raises(HTTPException) as exc:
            put_labels(ds_id, iid, {"boxes": [[0, 0.3, 0.3, 0.2, 0.2]], "base_version": res["version"]})
        assert exc.value.status_code == 409
        assert exc.value.detail["boxes"] == [[1, 0.1, 0.1, 0.1, 0.1]]
        assert lp.read_text() == "1 0.1 0.1 0.1 0.1\n", "a refused save must not touch the file"

        # "keep mine": retry against the version reported by the conflict
        ok = put_labels(ds_id, iid, {"boxes": [[0, 0.3, 0.3, 0.2, 0.2]], "base_version": exc.value.detail["version"]})
        assert ok["item"]["boxes"] == [[0, 0.3, 0.3, 0.2, 0.2]]

        # an identical rewrite by someone else is not a conflict (content, not timestamps)
        os.utime(lp, (1, 1))
        put_labels(ds_id, iid, {"boxes": [[0, 0.35, 0.3, 0.2, 0.2]], "base_version": ok["version"]})

        # a file that did not exist when loaded, but does now, is a conflict too
        lp.unlink()
        missing = get_labels(ds_id, iid)
        assert missing["version"] is None and missing["boxes"] == []
        lp.write_text("0 0.5 0.5 0.5 0.5\n")
        with pytest.raises(HTTPException):
            put_labels(ds_id, iid, {"boxes": [], "base_version": None})
    finally:
        _drop(ds_id)


def test_label_save_keeps_user_boxes_and_rejects_only_unstorable_rows(tmp_path):
    from fovea.api.datasets import put_labels

    ds_id, iid, lp = _scanned(tmp_path)
    try:
        res = put_labels(ds_id, iid, {"boxes": [
            [0, 0.5, 0.5, 0.0004, 0.0004],      # tiny: the user's call, flagged as an issue, never dropped
            [0, 0.99, 0.5, 0.1, 0.1],           # partly out of frame: kept as drawn
            [-1, 0.5, 0.5, 0.1, 0.1], [2.5, 0.5, 0.5, 0.1, 0.1], ["x", 0.5, 0.5, 0.1, 0.1],
            [0, float("nan"), 0.5, 0.1, 0.1], [0, 0.5, 0.5],
        ]})
        assert res["rejected"] == 5 and res["saved"] == 2
        assert [b[1] for b in res["item"]["boxes"]] == [0.5, 0.99]
        assert len(lp.read_text().splitlines()) == 2
    finally:
        _drop(ds_id)


def test_label_files_never_get_ragged_confidence_columns(tmp_path):
    out = tmp_path / "pred.txt"
    write_label_file(out, [[0, 0.5, 0.5, 0.2, 0.2, 0.61], [1, 0.3, 0.3, 0.1, 0.1]])
    lines = out.read_text().splitlines()
    assert [len(l.split()) for l in lines] == [6, 6], "a hand-drawn box in a prediction file gets conf 1.0"
    assert lines[1].endswith(" 1.000000")
    write_label_file(out, [[0, 0.5, 0.5, 0.2, 0.2]])
    assert out.read_text() == "0 0.500000 0.500000 0.200000 0.200000\n"


def test_bulk_ops_keep_confidence_with_its_own_box(tmp_path):
    from fovea import db
    from fovea.api.datasets import bulk_labels

    ds_id, iid, lp = _scanned(tmp_path)
    try:
        lp.write_text("0 0.1 0.1 0.1 0.1 0.9\n1 0.5 0.5 0.1 0.1 0.2\n0 0.8 0.8 0.1 0.1 0.4\n")
        bulk_labels(ds_id, {"op": "delete_class", "cls": 1, "image_ids": [iid]})
        rows = [l.split() for l in lp.read_text().splitlines()]
        assert [(r[1], r[5]) for r in rows] == [("0.100000", "0.900000"), ("0.800000", "0.400000")]

        other = db.query_one("SELECT label_path FROM images WHERE dataset_id=? AND id<>?", (ds_id, iid))
        Path(other["label_path"]).unlink()
        bulk_labels(ds_id, {"op": "clear", "image_ids": [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=?", (ds_id,))]})
        assert not Path(other["label_path"]).exists(), "clear must not invent label files"
        assert lp.read_text() == ""
    finally:
        _drop(ds_id)


def test_export_labels_are_plain_training_rows(tmp_path):
    from fovea.core.export import _label_lines

    p = tmp_path / "l.txt"
    p.write_text("0 0.5 0.5 0.2 0.2 0.93\n")
    assert _label_lines(str(p)) == "0 0.500000 0.500000 0.200000 0.200000\n"


# ---------------------------------------------------------------- ordering, reviews, export safety


def test_export_copy_refuses_to_write_into_the_dataset(tmp_path):
    from fovea.core.export import check_copy_target, copy_subset
    from fovea.jobs import Job

    root = _mini_dataset(tmp_path)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2 0.91\n")
    ds_id = _register(root)
    try:
        from fovea.core.scanner import scan_dataset
        scan_dataset(ds_id, Job(id="e", kind="scan"))
        before = (root / "images" / "train" / "a.jpg").read_bytes()
        for bad in (root, root / "images", root.parent):
            with pytest.raises(ValueError):
                check_copy_target(ds_id, str(bad))
        with pytest.raises(ValueError):
            copy_subset(Job(id="c", kind="export_copy"), ds_id, ["c0"], {}, str(root), resize={"mode": "max", "size": 16})
        assert (root / "images" / "train" / "a.jpg").read_bytes() == before
        assert (root / "labels" / "train" / "a.txt").read_text() == "0 0.5 0.5 0.2 0.2 0.91\n"
        out = tmp_path / "elsewhere"
        res = copy_subset(Job(id="c2", kind="export_copy"), ds_id, ["c0"], {}, str(out))
        assert res["images"] == 2 and (out / "labels" / "train" / "a.txt").read_text() == "0 0.500000 0.500000 0.200000 0.200000\n"
    finally:
        _drop(ds_id)


def test_dataset_files_are_never_served_as_active_content(tmp_path):
    from fastapi.testclient import TestClient
    from fovea.main import app

    root = _mini_dataset(tmp_path)
    (root / "README.html").write_text("<script>fetch('/api/fs/browse?path=/')</script>")
    (root / "logo.svg").write_text("<svg xmlns='http://www.w3.org/2000/svg'><script>alert(1)</script></svg>")
    ds_id = _register(root)
    try:
        c = TestClient(app)
        for name in ("README.html", "logo.svg"):
            r = c.get(f"/api/datasets/{ds_id}/file", params={"path": name, "raw": 1})
            assert r.status_code == 200
            assert r.headers["content-type"] == "application/octet-stream"
            assert r.headers["content-disposition"].startswith("attachment")
            assert r.headers["x-content-type-options"] == "nosniff"
        img = c.get(f"/api/datasets/{ds_id}/file", params={"path": "images/train/a.jpg"})
        assert img.headers["content-type"] == "image/jpeg"
    finally:
        _drop(ds_id)
