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
