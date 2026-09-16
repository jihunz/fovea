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
