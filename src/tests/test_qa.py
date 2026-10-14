"""QA engine: hashes and image statistics, near-duplicates / leakage / label mismatch, sequence checks,
shape outliers, image quality, similar search, interpolation and gap filling. Run: python -m pytest src/tests -q"""
from __future__ import annotations

import itertools
import os
import random
import shutil
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw, ImageFilter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout, qa  # noqa: E402
from fovea.jobs import Job, jobs  # noqa: E402

W, H = 320, 240


# --------------------------------------------------------------------------- synthetic data
def _scene(seed: int) -> Image.Image:
    """A textured background with hard edges (so it is sharp) that differs per seed."""
    rng = random.Random(seed)
    im = Image.new("RGB", (W, H), (rng.randrange(60, 200),) * 3)
    d = ImageDraw.Draw(im)
    for _ in range(40):
        x, y = rng.randrange(W), rng.randrange(H)
        w, h = rng.randrange(8, 70), rng.randrange(8, 60)
        col = tuple(rng.randrange(256) for _ in range(3))
        (d.rectangle if rng.random() < 0.6 else d.ellipse)([x, y, x + w, y + h], fill=col)
    return im


def _draw_box(im: Image.Image, b, fill=(20, 20, 20)) -> None:
    _c, xc, yc, w, h = b
    ImageDraw.Draw(im).rectangle([(xc - w / 2) * W, (yc - h / 2) * H, (xc + w / 2) * W, (yc + h / 2) * H], fill=fill)


def _write(root: Path, split: str, name: str, im: Image.Image, boxes, quality: int = 92) -> Path:
    p = root / "images" / split / f"{name}.jpg"
    p.parent.mkdir(parents=True, exist_ok=True)
    im.save(p, "JPEG", quality=quality)
    if boxes is not None:
        lp = root / "labels" / split / f"{name}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text("".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes))
    return p


def P(k):  # the walking person of sequence camA, frame k (moves 6.4 px per frame)
    return [0, 0.30 + 0.02 * k, 0.6, 0.08, 0.30]


S = [1, 0.75, 0.30, 60 / W, 40 / H]          # a static object in camA


def _build(root: Path) -> None:
    bg = _scene(1)
    for k in range(1, 13):
        im = bg.copy()
        _draw_box(im, P(k))
        _draw_box(im, S, fill=(230, 230, 40))
        boxes = [P(k), list(S)]
        if k == 6:
            boxes = [list(S)]                                           # seq_gap: the person lost its box
        if k == 4:
            boxes = [P(k), [1, S[1] - 45 / W, S[2], S[3], S[4]]]        # seq_jump: static box slipped 45 px
        if k == 8:
            boxes += [[3, 0.1, 0.1, 0.05, 0.05], [3, 0.2, 0.15, 0.05, 0.05], [3, 0.1, 0.9, 0.05, 0.05]]   # count spike
        if k == 9:
            boxes = [P(k), [2] + S[1:]]                                 # seq_flicker: 1 -> 2 -> 1
        _write(root, "train", f"camA_{k:04d}", im, boxes)
    # an exact copy of a train frame in val (leakage + exact duplicate), and the video continuing in val
    a3 = root / "images" / "train" / "camA_0003.jpg"
    (root / "images" / "val").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(a3, root / "images" / "val" / "copy_camA_0003.jpg")
    (root / "labels" / "val").mkdir(parents=True, exist_ok=True)
    shutil.copyfile(root / "labels" / "train" / "camA_0003.txt", root / "labels" / "val" / "copy_camA_0003.txt")
    im = bg.copy()
    _draw_box(im, P(13))
    _draw_box(im, S, fill=(230, 230, 40))
    _write(root, "val", "camA_0013", im, [P(13), list(S)])
    # unrelated photos that merely share a name prefix: never a "sequence"; they carry class 4 for shape stats
    rng = random.Random(7)
    for k in range(1, 7):
        boxes = []
        for _ in range(6):
            w = 0.06 * rng.uniform(0.9, 1.1)
            boxes.append([4, rng.uniform(0.2, 0.8), rng.uniform(0.2, 0.8), w, w * 1.6])
        if k == 2:
            boxes[0] = [4, 0.5, 0.5, 0.6, 0.06]                       # far too wide for class 4
        _write(root, "train", f"IMG_{k:04d}", _scene(100 + k), boxes)
    # image quality
    _write(root, "train", "night", Image.new("RGB", (W, H), (6, 6, 8)), [])
    _write(root, "train", "soft", _scene(55).filter(ImageFilter.GaussianBlur(5)), [])
    for k in range(1, 9):
        _write(root, "train", f"misc{k}", _scene(300 + k), [])
    # two identical pictures labeled differently
    sc = _scene(77)
    _draw_box(sc, [0, 0.5, 0.5, 0.1, 0.3])
    a = _write(root, "test", "scene_1", sc, [[0, 0.5, 0.5, 0.1, 0.3]])
    b = root / "images" / "test" / "scene_2.jpg"
    shutil.copyfile(a, b)
    (root / "labels" / "test" / "scene_2.txt").write_text("0 0.500000 0.500000 0.100000 0.300000\n1 0.200000 0.200000 0.100000 0.100000\n")
    # keyframes for interpolation: camB frames 10 .. 20 in val, only 10, 14 and 20 labeled
    bgb = _scene(2)
    for f in (10, 12, 14, 15, 20):
        x = 0.2 + 0.4 * (f - 10) / 10
        im = bgb.copy()
        _draw_box(im, [0, x, 0.5, 0.1, 0.3])
        boxes = None
        if f in (10, 20):
            boxes = [[0, x, 0.5, 0.1, 0.3], [1, 0.8, 0.2, 0.1, 0.1]]
        elif f == 14:
            boxes = [[1, 0.8, 0.2, 0.1, 0.1]]
        _write(root, "val", f"camB_{f:04d}", im, boxes)


def _register(root: Path) -> str:
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "qa-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps([]), "inferred", "", "new", now, now))
    return ds_id


def _drop(ds_id: str) -> None:
    for t in ("ai_preds", "ai_runs", "image_flags", "image_hash", "boxes", "images", "plugin_data", "datasets"):
        col = "id" if t == "datasets" else "dataset_id"
        db.execute(f"DELETE FROM {t} WHERE {col}=?", (ds_id,))


def _wait_idle(ds_id: str, kind: str = "qa", timeout: float = 60) -> None:
    t0 = time.time()
    while jobs.active_for(ds_id, kind) is not None:
        assert time.time() - t0 < timeout, f"{kind} job did not finish"
        time.sleep(0.05)


@pytest.fixture(scope="module")
def ds(tmp_path_factory):
    """One scanned synthetic dataset, QA computed once; the scan hook is detached so the tests stay
    deterministic (test_post_scan_hook_queues_qa covers it)."""
    from fovea.core import scanner
    from fovea.api import qa as qa_api  # noqa: F401 — registers the hook, which the module fixture removes
    hooks = list(scanner.POST_SCAN_HOOKS)
    scanner.POST_SCAN_HOOKS[:] = [h for h in hooks if h is not qa.on_scan]
    root = tmp_path_factory.mktemp("qa") / "ds"
    _build(root)
    ds_id = _register(root)
    scanner.scan_dataset(ds_id, Job(id="scan-" + ds_id, kind="scan"))
    result = qa.run_qa(ds_id, Job(id="qa-" + ds_id, kind="qa"))
    ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
    yield {"id": ds_id, "root": root, "ids": ids, "result": result}
    _drop(ds_id)
    scanner.POST_SCAN_HOOKS[:] = [h for h in hooks if h is not qa.on_scan]


def _flags(ds_id, code):
    return {r["image_id"]: db.loads(r["detail"], {}) for r in db.query(
        "SELECT image_id, detail FROM image_flags WHERE dataset_id=? AND code=?", (ds_id, code))}


def _id(d, rel):
    return d["ids"][rel]


# --------------------------------------------------------------------------- pure helpers
def test_near_pairs_finds_every_pair_within_the_threshold():
    rng = random.Random(3)
    base = [rng.getrandbits(256) for _ in range(300)]
    hashes = list(base)
    for h in base[:120]:                      # plant near copies at every distance 1..8
        v = h
        for b in rng.sample(range(256), rng.randrange(1, 9)):
            v ^= 1 << b
        hashes.append(v)
    # a big bucket (> 400 share the low band) exercises the blockwise numpy path
    common = rng.getrandbits(256)
    for _ in range(450):
        v = common
        for b in rng.sample(range(40, 256), rng.randrange(0, 12)):
            v ^= 1 << b
        hashes.append(v)
    hashes = list(dict.fromkeys(hashes))       # near_pairs works on distinct hashes
    for k in (2, 6):
        got = {(i, j): d for i, j, d in qa.near_pairs(hashes, k)}
        want = {(i, j): (hashes[i] ^ hashes[j]).bit_count() for i, j in itertools.combinations(range(len(hashes)), 2)
                if (hashes[i] ^ hashes[j]).bit_count() <= k}
        assert got == want


def test_hungarian_assignment_is_optimal():
    rng = random.Random(5)
    for n, m in ((4, 4), (3, 5), (5, 3), (1, 1)):
        for _ in range(20):
            cost = [[rng.random() for _ in range(m)] for _ in range(n)]
            pairs = qa.assign(cost)
            assert len(pairs) == min(n, m) and len({i for i, _ in pairs}) == len(pairs) == len({j for _, j in pairs})
            best = min(sum(cost[i][p[i]] for i in range(n)) for p in itertools.permutations(range(m), n)) if n <= m else \
                min(sum(cost[p[j]][j] for j in range(m)) for p in itertools.permutations(range(n), m))
            assert sum(cost[i][j] for i, j in pairs) == pytest.approx(best)
    assert qa.assign([]) == []


def test_labels_disagree_reasons():
    a = [[0, 0.5, 0.5, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]]
    assert qa.labels_disagree(a, [[1, 0.2, 0.2, 0.1, 0.1], [0, 0.51, 0.5, 0.2, 0.2]], 0.5) is None   # order-free
    assert qa.labels_disagree(a, a[:1], 0.5) == "count"
    assert qa.labels_disagree(a, [[0, 0.5, 0.5, 0.2, 0.2], [2, 0.2, 0.2, 0.1, 0.1]], 0.5) == "class"
    assert qa.labels_disagree(a, [[0, 0.8, 0.8, 0.2, 0.2], [1, 0.2, 0.2, 0.1, 0.1]], 0.5) == "boxes"


def test_natural_order_and_frame_numbers():
    names = ["f_10.jpg", "f_9.jpg", "f_100.jpg", "f_1.jpg"]
    assert sorted(names, key=qa.natural_key) == ["f_1.jpg", "f_9.jpg", "f_10.jpg", "f_100.jpg"]
    assert qa.frame_number("train/images_GOPR0804_00035.jpg") == 35
    assert qa.frame_number("cam/frame0007.png") == 7


def test_features_hashes_and_blur(tmp_path):
    sc = _scene(11)
    a = tmp_path / "a.jpg"
    sc.save(a, quality=95)
    sc.resize((W // 2, H // 2)).save(tmp_path / "half.jpg", quality=80)       # a re-encoded, resized copy
    _scene(12).save(tmp_path / "other.jpg", quality=95)
    sc.filter(ImageFilter.GaussianBlur(4)).save(tmp_path / "blur.jpg", quality=95)
    Image.new("RGB", (W, H), (5, 5, 5)).save(tmp_path / "dark.jpg")
    f = {n: qa.image_features(str(tmp_path / f"{n}.jpg")) for n in ("a", "half", "other", "blur", "dark")}
    assert qa.hamming(f["a"]["dhash"], f["half"]["dhash"]) <= 4
    assert qa.hamming(f["a"]["fine"], f["half"]["fine"]) <= 12
    assert qa.hamming(f["a"]["fine"], f["other"]["fine"]) > 60
    assert f["blur"]["sharpness"] < f["a"]["sharpness"] - 0.15
    assert f["dark"]["luma"] < 10
    assert qa.image_features(str(tmp_path / "missing.jpg")) is None
    (tmp_path / "junk.jpg").write_bytes(b"not an image")
    assert qa.image_features(str(tmp_path / "junk.jpg")) is None


def test_blur_effect_without_numpy_matches_numpy(monkeypatch):
    g = _scene(21).convert("L")
    gb = g.filter(ImageFilter.GaussianBlur(3))
    with_np = (qa.blur_effect(g), qa.blur_effect(gb))
    monkeypatch.setattr(qa, "_np_mod", None)
    without = (qa.blur_effect(g), qa.blur_effect(gb))
    assert without[0] == pytest.approx(with_np[0], abs=0.02) and without[1] == pytest.approx(with_np[1], abs=0.02)
    assert with_np[1] > with_np[0] + 0.2


def test_sixteen_bit_and_palette_images_hash(tmp_path):
    Image.new("I;16", (64, 48), 30000).save(tmp_path / "deep.png")
    Image.new("P", (64, 48), 3).save(tmp_path / "pal.png")
    assert qa.image_features(str(tmp_path / "deep.png")) is not None
    assert qa.image_features(str(tmp_path / "pal.png")) is not None


# --------------------------------------------------------------------------- the checks on a real scan
def test_every_image_is_hashed(ds):
    n = db.query_one("SELECT COUNT(*) c FROM images WHERE dataset_id=?", (ds["id"],))["c"]
    assert ds["result"]["hashed"] == n
    assert db.query_one("SELECT COUNT(*) c FROM image_hash WHERE dataset_id=? AND b0 IS NOT NULL", (ds["id"],))["c"] == n


def test_exact_copy_across_splits_is_a_duplicate_and_a_leak(ds):
    a3, cp = _id(ds, "train/camA_0003.jpg"), _id(ds, "val/copy_camA_0003.jpg")
    dup = _flags(ds["id"], "near_dup")
    assert dup[a3]["group"] == dup[cp]["group"]
    assert dup[cp]["exact"] and dup[cp]["copy_of"] == min(a3, cp) or dup[a3].get("copy_of") == min(a3, cp)
    leak = _flags(ds["id"], "leak_split")
    assert leak[cp]["reason"] == "twin" and leak[cp]["dist"] == 0 and leak[cp]["twin"] == a3 and leak[cp]["splits"] == ["train"]
    assert leak[a3]["twin"] == cp
    s = qa.load_summary(ds["id"])
    assert s["exact_dups"] >= 2           # the copied frame and the scene_1 / scene_2 pair
    # the camA video continues in val: listed as a sequence spanning splits, with same-view evidence
    span = {r["seq"]: r for r in s["split_spanning_sequences"]}
    assert span["camA"]["splits"] == ["train", "val"] and span["camA"]["evidence"]
    assert leak[_id(ds, "val/camA_0013.jpg")]["reason"] in ("seq", "twin")


def test_identical_frames_with_different_labels(ds):
    mm = _flags(ds["id"], "dup_label_mismatch")
    s1, s2 = _id(ds, "test/scene_1.jpg"), _id(ds, "test/scene_2.jpg")
    assert mm[s1]["twin"] == s2 and mm[s2]["twin"] == s1 and mm[s1]["reason"] == "count"
    # identical labels on the leaked copy are not a mismatch
    assert _id(ds, "val/copy_camA_0003.jpg") not in mm


def test_sequence_gap_jump_flicker_and_count_spike(ds):
    i = lambda k: _id(ds, f"train/camA_{k:04d}.jpg")  # noqa: E731
    gap = _flags(ds["id"], "seq_gap")
    assert i(6) in gap
    (e,) = gap[i(6)]["boxes"]
    assert e[0] == 0 and e[1] == pytest.approx(P(6)[1], abs=0.01) and e[2] == pytest.approx(0.6, abs=0.01)
    assert gap[i(6)]["prev_id"] == i(5) and gap[i(6)]["next_id"] == i(7)
    jump = _flags(ds["id"], "seq_jump")
    assert i(4) in jump and jump[i(4)]["jumps"][0]["expected"][1] == pytest.approx(S[1], abs=0.01)
    flick = _flags(ds["id"], "seq_flicker")
    assert i(9) in flick and flick[i(9)]["rows"][0] == {"row": 1, "cls": 2, "expected": 1}
    spike = _flags(ds["id"], "seq_count_spike")
    assert spike[i(8)]["n"] == 5 and spike[i(8)]["prev"] == 2 and spike[i(8)]["next"] == 2
    # nothing temporal is claimed about frames that merely share a name (IMG_0001.. are unrelated photos)
    seq_codes = ("seq_gap", "seq_jump", "seq_flicker", "seq_count_spike")
    img_ids = {v for k, v in ds["ids"].items() if "/IMG_" in k}
    for code in seq_codes:
        assert not (set(_flags(ds["id"], code)) & img_ids), code
    clean = {i(k) for k in (1, 2, 11, 12)}
    for code in seq_codes:
        assert not (set(_flags(ds["id"], code)) & clean), code


def test_shape_outlier_and_quality(ds):
    shape = _flags(ds["id"], "shape_outlier")
    wide = _id(ds, "train/IMG_0002.jpg")
    assert wide in shape and shape[wide]["rows"][0]["row"] == 0 and shape[wide]["rows"][0]["kind"] == "aspect"
    assert sum(1 for k, v in ds["ids"].items() if "/IMG_" in k and v in shape) == 1
    assert _id(ds, "train/night.jpg") in _flags(ds["id"], "dark")
    blurry = _flags(ds["id"], "blurry")
    assert _id(ds, "train/soft.jpg") in blurry
    assert len(blurry) == 1, "only the blurred picture is blurry"


def test_dark_dataset_is_not_flagged_wholesale():
    class It:
        def __init__(self, i, luma):
            self.id, self.luma, self.sharpness, self.contrast = i, luma, 0.78, 30.0
    items = [It(i, 18 + (i % 7)) for i in range(200)]          # a uniformly dark CCTV set
    flags: dict = {}
    qa.check_quality(items, dict(qa.DEFAULTS), flags, {})
    assert not [k for k in flags if k[1] == "dark"]
    items.append(It(999, 1.0))                                   # ... except a black frame
    qa.check_quality(items, dict(qa.DEFAULTS), flags, {})
    assert [k for k in flags if k[1] == "dark"] == [(999, "dark")]


def test_incremental_rehash_only_changed_images(ds, monkeypatch):
    from fovea.core import scanner
    monkeypatch.setattr(scanner, "POST_SCAN_HOOKS", [])
    again = qa.run_qa(ds["id"], Job(id="qa2", kind="qa"), ["quality"])
    assert again["hashed"] == 0
    p = ds["root"] / "images" / "train" / "misc1.jpg"
    _scene(999).save(p, quality=90)
    os.utime(p, (time.time() + 5, time.time() + 5))
    scanner.scan_dataset(ds["id"], Job(id="rescan", kind="scan"))
    assert qa.stale_images(ds["id"]) == 1
    third = qa.run_qa(ds["id"], Job(id="qa3", kind="qa"), ["quality"])
    assert third["hashed"] == 1 and qa.stale_images(ds["id"]) == 0


def test_missing_file_is_retried_not_remembered_as_unreadable(ds):
    iid = _id(ds, "train/misc2.jpg")
    p = ds["root"] / "images" / "train" / "misc2.jpg"
    keep = p.read_bytes()
    db.execute("DELETE FROM image_hash WHERE image_id=?", (iid,))
    p.unlink()
    try:
        res = qa.update_hashes(ds["id"])
        assert res["missing"] == 1 and res["hashed"] == 0
        assert db.query_one("SELECT 1 FROM image_hash WHERE image_id=?", (iid,)) is None
    finally:
        p.write_bytes(keep)
    assert qa.update_hashes(ds["id"])["hashed"] == 1


# --------------------------------------------------------------------------- API
def test_summary_duplicates_similar_and_image_flags(ds):
    from fovea.api import qa as api
    s = api.qa_summary(ds["id"], auto=0)
    assert s["status"] == "ok" and s["counts"]["near_dup"] >= 4 and s["counts"]["leak_split"] >= 2
    assert set(s["counts"]) == set(qa.CODES) and s["labels"]["seq_gap"]
    assert s["hashed"] == s["images"]
    d = api.qa_duplicates(ds["id"], 0, 50, 1, 60)
    assert d["total"] >= 1 and all(g["cross_split"] for g in d["groups"])
    a3, cp = _id(ds, "train/camA_0003.jpg"), _id(ds, "val/copy_camA_0003.jpg")
    grp = next(g for g in d["groups"] if a3 in {m["image_id"] for m in g["members"]})
    assert cp in {m["image_id"] for m in grp["members"]} and grp["members"][0]["leader"]
    assert sum(m["keep"] for m in grp["members"]) == 1
    every = api.qa_duplicates(ds["id"], 0, 200, 0, 60)
    assert every["total"] >= d["total"]
    sim = api.similar_images(ds["id"], a3, 5, 64)
    assert sim["items"][0]["id"] == cp and sim["items"][0]["dist"] == 0
    assert all(x["dist"] <= 64 for x in sim["items"]) and a3 not in {x["id"] for x in sim["items"]}
    assert [x["dist"] for x in sim["items"]] == sorted(x["dist"] for x in sim["items"])
    flags = api.image_qa(ds["id"], cp)["flags"]
    assert {"near_dup", "leak_split"} <= {f["code"] for f in flags} and all(f["label"] for f in flags)


def test_similar_search_numpy_and_python_agree(ds, monkeypatch):
    a3 = _id(ds, "train/camA_0003.jpg")
    with_np = qa.similar(ds["id"], a3, 10, 256)["items"]
    monkeypatch.setattr(qa, "_np_mod", None)
    qa._sim_cache.clear()
    without = qa.similar(ds["id"], a3, 10, 256)["items"]
    qa._sim_cache.clear()
    assert [x["dist"] for x in with_np] == [x["dist"] for x in without]


def test_bad_requests_are_400(ds):
    from fastapi import HTTPException
    from fovea.api import qa as api
    with pytest.raises(HTTPException) as e:
        api.start_qa(ds["id"], {"checks": ["nonsense"]})
    assert e.value.status_code == 400
    with pytest.raises(HTTPException):
        api.start_qa(ds["id"], {"options": {"dup_dist": 99}})
    with pytest.raises(HTTPException):
        api.start_qa(ds["id"], {"options": {"no_such": 1}})
    with pytest.raises(HTTPException):
        api.interpolate(ds["id"], {"from_image_id": "x"})
    with pytest.raises(HTTPException) as e:
        api.interpolate(ds["id"], {"from_image_id": _id(ds, "train/camA_0001.jpg"), "to_image_id": _id(ds, "val/camB_0010.jpg")})
    assert "same sequence" in e.value.detail
    with pytest.raises(HTTPException):
        api.fill_gap(ds["id"], _id(ds, "train/night.jpg"))
    with pytest.raises(HTTPException) as e:
        api.similar_images(ds["id"], 10 ** 9, 5, 64)
    assert e.value.status_code == 404
    assert qa.normalise_checks(["labels"]) == ["duplicates", "sequence", "shape"]
    assert qa.normalise_checks(["blurry"]) == ["quality"]


def test_interpolation_between_keyframes(ds):
    ds_id = ds["id"]
    k10, k20 = _id(ds, "val/camB_0010.jpg"), _id(ds, "val/camB_0020.jpg")
    res = qa.interpolate(ds_id, k20, k10)                    # order does not matter
    assert res["frames"] == 3 and res["by"] == "frame_number" and res["matched"] == 2
    assert res["skipped_existing"] == 1                      # camB_0014 already has the static box
    by_img = {}
    for p in res["preds"]:
        assert p["kind"] == "interp" and p["state"] == "pending" and p["conf"] is None and p["run_id"] == res["run_id"]
        by_img.setdefault(p["image_id"], []).append(p)
    for f in (12, 14, 15):
        people = [p for p in by_img[_id(ds, f"val/camB_{f:04d}.jpg")] if p["cls"] == 0]
        assert len(people) == 1 and people[0]["xc"] == pytest.approx(0.2 + 0.4 * (f - 10) / 10, abs=1e-4)
    assert len(by_img[_id(ds, "val/camB_0014.jpg")]) == 1
    run = db.query_one("SELECT kind, provider, status, boxes FROM ai_runs WHERE id=?", (res["run_id"],))
    assert run == {"kind": "interp", "provider": "fovea", "status": "done", "boxes": len(res["preds"])}
    # running it again replaces the pending proposals instead of stacking duplicates
    again = qa.interpolate(ds_id, k10, k20)
    n = db.query_one("SELECT COUNT(*) c FROM ai_preds WHERE dataset_id=? AND kind='interp' AND state='pending'", (ds_id,))["c"]
    assert n == len(again["preds"]) == len(res["preds"])
    # a decided proposal is history, not replaced
    db.execute("UPDATE ai_preds SET state='accepted' WHERE id=?", (again["preds"][0]["id"],))
    qa.interpolate(ds_id, k10, k20)
    assert db.query_one("SELECT state FROM ai_preds WHERE id=?", (again["preds"][0]["id"],))["state"] == "accepted"
    adj = qa.interpolate(ds_id, _id(ds, "train/camA_0001.jpg"), _id(ds, "train/camA_0002.jpg"))
    assert adj["preds"] == [] and adj["run_id"] is None
    with pytest.raises(qa.QAError):
        qa.interpolate(ds_id, k10, k10)
    with pytest.raises(qa.QAError):                           # camB_0012 has no boxes
        qa.interpolate(ds_id, k10, _id(ds, "val/camB_0012.jpg"))
    db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds_id,))


def test_fill_gap_proposes_the_missing_box(ds):
    ds_id = ds["id"]
    g = _id(ds, "train/camA_0006.jpg")
    res = qa.fill_gap(ds_id, g)
    assert res["from_image_id"] == _id(ds, "train/camA_0005.jpg") and res["to_image_id"] == _id(ds, "train/camA_0007.jpg")
    (p,) = res["preds"]
    assert p["cls"] == 0 and p["xc"] == pytest.approx(P(6)[1], abs=1e-3) and p["image_id"] == g and p["kind"] == "interp"
    # nothing to fill on a complete frame
    ok = qa.fill_gap(ds_id, _id(ds, "train/camA_0011.jpg"))
    assert ok["preds"] == [] and "Nothing to fill" in ok["message"]
    db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds_id,))


def test_labels_edit_makes_summary_stale_and_auto_rechecks(ds):
    from fovea.api import qa as api
    from fovea.core.scanner import refresh_image
    i5 = _id(ds, "train/camA_0005.jpg")
    lp = ds["root"] / "labels" / "train" / "camA_0005.txt"
    old = lp.read_text()
    lp.write_text("0 0.400000 0.600000 0.080000 0.300000\n")       # the static box disappears in frame 5
    t = time.time()
    os.utime(lp, (t, t))
    refresh_image(i5)
    api._auto_at.pop(ds["id"], None)
    s = api.qa_summary(ds["id"], auto=1)
    assert s["stale_labels"] == 1 and s["status"] == "running"
    _wait_idle(ds["id"])
    s = api.qa_summary(ds["id"], auto=0)
    assert s["status"] == "ok" and s["stale_labels"] == 0
    assert i5 in _flags(ds["id"], "seq_gap")
    lp.write_text(old)
    refresh_image(i5)
    qa.run_qa(ds["id"], Job(id="qa-restore", kind="qa"), ["labels"])
    assert i5 not in _flags(ds["id"], "seq_gap")


def test_post_scan_hook_queues_qa(tmp_path):
    from fovea.api import qa as api  # noqa: F401 — import registers the hook
    from fovea.core import scanner
    root = tmp_path / "hook"
    for k in range(3):
        _write(root, "train", f"v_{k:03d}", _scene(40 + k), [[0, 0.5, 0.5, 0.1, 0.1]])
    ds_id = _register(root)
    saved = list(scanner.POST_SCAN_HOOKS)
    scanner.POST_SCAN_HOOKS[:] = saved + ([qa.on_scan] if qa.on_scan not in saved else [])
    try:
        scanner.scan_dataset(ds_id, Job(id="hook-scan", kind="scan"))
        assert jobs.active_for(ds_id, "qa") is not None or (qa.load_summary(ds_id) or {}).get("computed_at")
        _wait_idle(ds_id)
        s = qa.load_summary(ds_id)
        assert s and s["hashed"] == 3
    finally:
        scanner.POST_SCAN_HOOKS[:] = [h for h in saved if h is not qa.on_scan]
        _wait_idle(ds_id)
        _drop(ds_id)


def test_routes_are_reachable_over_http(ds):
    from fastapi.testclient import TestClient
    from fovea.main import app
    c = TestClient(app)
    a3 = _id(ds, "train/camA_0003.jpg")
    r = c.get(f"/api/datasets/{ds['id']}/images/{a3}/similar?k=3")
    assert r.status_code == 200 and r.json()["items"]
    assert c.get(f"/api/datasets/{ds['id']}/images/{a3}/qa").status_code == 200
    assert c.get(f"/api/datasets/{ds['id']}/qa/duplicates?cross_split=1").json()["total"] >= 1
    assert c.get(f"/api/datasets/{ds['id']}/qa/summary?auto=0").json()["status"] in ("ok", "stale")
    r = c.post(f"/api/datasets/{ds['id']}/images/{_id(ds, 'train/camA_0006.jpg')}/fill-gap")
    assert r.status_code == 200 and len(r.json()["preds"]) == 1
    db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds["id"],))
    assert c.get("/api/datasets/nope/qa/summary").status_code == 404


def test_threshold_options_are_remembered_and_resettable(ds):
    from fovea.api import qa as api
    api.start_qa(ds["id"], {"checks": ["shape"], "options": {"shape_z": 2.5}})
    _wait_idle(ds["id"])
    s = api.qa_summary(ds["id"], auto=0)
    assert s["options"]["shape_z"] == 2.5 and s["options"]["dup_dist"] == qa.DEFAULTS["dup_dist"]
    assert qa.options_for(ds["id"])["shape_z"] == 2.5          # later runs (hook, auto) use it too
    assert s["unique_images"] == s["images"] - s["exact_dups"]
    api.start_qa(ds["id"], {"checks": ["shape"], "options": {"shape_z": qa.DEFAULTS["shape_z"]}})
    _wait_idle(ds["id"])
    assert "shape_z" not in (qa.load_summary(ds["id"])["options"] or {})
    assert _id(ds, "train/IMG_0002.jpg") in _flags(ds["id"], "shape_outlier")


# --------------------------------------------------------------------------- wave 2a: label-write re-checks
LABEL_CODES = ("seq_gap", "seq_jump", "seq_flicker", "seq_count_spike", "dup_label_mismatch", "shape_outlier")


def _label_path(d, rel_img):
    split, name = rel_img.split("/", 1)
    return d["root"] / "labels" / split / (Path(name).stem + ".txt")


_tick = [0.0]


def _set_labels(d, rel_img, text, notify=True):
    """Write a label file the way an editor would (new mtime, index refreshed, hook announced)."""
    from fovea.core import labels as labels_mod
    from fovea.core.scanner import refresh_image
    lp = _label_path(d, rel_img)
    old = lp.read_text() if lp.exists() else None
    lp.write_text(text)
    _tick[0] = max(time.time(), _tick[0] + 0.01)     # strictly increasing label mtimes, like real saves
    os.utime(lp, (_tick[0], _tick[0]))
    iid = _id(d, rel_img)
    refresh_image(iid)
    if notify:
        labels_mod.notify_label_write(d["id"], [iid])
    return old


def _rows(boxes):
    return "".join(f"{b[0]} {b[1]:.6f} {b[2]:.6f} {b[3]:.6f} {b[4]:.6f}\n" for b in boxes)


def _all_label_flags(ds_id):
    return {(r["image_id"], r["code"]): db.loads(r["detail"], {}) for r in db.query(
        "SELECT image_id, code, detail FROM image_flags WHERE dataset_id=? AND code IN (%s)" % ",".join("?" * len(LABEL_CODES)),
        (ds_id, *LABEL_CODES))}


@pytest.fixture
def edits(ds):
    """Label edits made by a test are reverted afterwards, and the label checks recomputed in full."""
    saved = {}

    def edit(rel_img, text, notify=True):
        old = _set_labels(ds, rel_img, text, notify)
        saved.setdefault(rel_img, old)
    yield edit
    for rel_img, old in saved.items():
        lp = _label_path(ds, rel_img)
        if old is None:
            lp.unlink(missing_ok=True)
        else:
            _set_labels(ds, rel_img, old, notify=False)
    qa.flush_label_rechecks(ds["id"])
    _wait_idle(ds["id"])
    qa.run_qa(ds["id"], Job(id="qa-revert", kind="qa"), ["labels"])


def test_label_write_hook_is_registered():
    from fovea.api import qa as api
    from fovea.core import labels as labels_mod
    assert api.register_label_hook()
    assert labels_mod.LABEL_WRITE_HOOKS.count(qa.on_label_write) == 1


def test_label_edit_rechecks_its_sequence_without_a_job(ds, edits):
    from fovea.api import qa as api
    i5 = _id(ds, "train/camA_0005.jpg")
    assert i5 not in _flags(ds["id"], "seq_gap")
    edits("train/camA_0005.jpg", _rows([P(5)]))                 # the static object loses its box in frame 5
    s = api.qa_summary(ds["id"], auto=0)
    assert s["refreshing"] >= 1 and s["stale_labels"] == 0     # queued for the incremental re-check, not "stale"
    assert api.image_qa(ds["id"], i5)["refreshing"]
    assert qa.flush_label_rechecks(ds["id"])
    assert not api.image_qa(ds["id"], i5)["refreshing"]
    gap = _flags(ds["id"], "seq_gap")
    assert i5 in gap and gap[i5]["boxes"][0][0] == 1
    # flags are fresh: the summary neither reports stale labels nor queues a full run
    api._auto_at.pop(ds["id"], None)
    s = api.qa_summary(ds["id"], auto=1)
    assert s["status"] == "ok" and s["stale_labels"] == 0 and s["refreshing"] == 0 and s["job"] is None
    assert jobs.active_for(ds["id"], "qa") is None
    last = qa._refresher.last_run[ds["id"]]
    assert last["images"] == 1 and last["sequence_frames"] == 12 and "covered" not in last
    # ... and the label checks' timestamp moved forward, so a restart does not see them as stale either
    qa._refresher.covered.pop(ds["id"], None)
    assert api.qa_summary(ds["id"], auto=0)["stale_labels"] == 0
    # undo: the gap disappears again on the next save
    edits("train/camA_0005.jpg", _rows([P(5), S]))
    assert qa.flush_label_rechecks(ds["id"])
    assert i5 not in _flags(ds["id"], "seq_gap")


def test_incremental_recheck_matches_a_full_run(ds, edits):
    i = lambda k: f"train/camA_{k:04d}.jpg"  # noqa: E731
    edits(i(6), _rows([P(6), S]))                               # the gap is fixed
    edits(i(9), _rows([P(9), [3] + S[1:]]))                     # a different flicker
    edits(i(11), _rows([P(11)]))                                # a new gap
    edits("test/scene_2.jpg", _rows([[0, 0.5, 0.5, 0.1, 0.3]]))  # the twins now agree
    edits("val/camB_0014.jpg", _rows([[1, 0.8, 0.2, 0.1, 0.1], [0, 0.36, 0.5, 0.1, 0.3]]))
    assert qa.flush_label_rechecks(ds["id"])
    incremental = _all_label_flags(ds["id"])
    qa.run_qa(ds["id"], Job(id="qa-full", kind="qa"), ["labels"])
    full = _all_label_flags(ds["id"])
    assert incremental == full                                   # every label flag, details included
    assert (_id(ds, i(6)), "seq_gap") not in incremental and (_id(ds, i(11)), "seq_gap") in incremental
    s1, s2 = _id(ds, "test/scene_1.jpg"), _id(ds, "test/scene_2.jpg")
    # scene_1 was not edited, but its twin was: its mismatch flag is gone too
    assert (s1, "dup_label_mismatch") not in incremental and (s2, "dup_label_mismatch") not in incremental


def _group_of(ds_id, image_id):
    return next(g for g in qa.duplicate_groups(ds_id, 0, 200)["groups"] if image_id in {m["image_id"] for m in g["members"]})


def _export_keeps(ds_id, ids):
    """What Export's "drop near-duplicates" keeps of these images (the rule the duplicates view must name)."""
    from fovea.core import export
    plan = export.build_plan(ds_id, [], {"ids": ",".join(map(str, ids))}, export.ExportOptions(drop_near_dups=True))
    return [it.id for it in plan.items]


def test_mismatch_and_keep_follow_an_edit_of_one_twin(ds, edits):
    a3, cp = _id(ds, "train/camA_0003.jpg"), _id(ds, "val/copy_camA_0003.jpg")
    g = _group_of(ds["id"], a3)
    assert g["keep"] == min(a3, cp) and {m["image_id"] for m in g["members"] if m["keep"]} == {min(a3, cp)}
    assert _export_keeps(ds["id"], [a3, cp]) == [g["keep"]]
    newer = "val/copy_camA_0003.jpg" if cp > a3 else "train/camA_0003.jpg"
    edits(newer, _rows([P(3), S, [3, 0.1, 0.1, 0.05, 0.05]]))   # the newer copy gains a box
    assert qa.flush_label_rechecks(ds["id"])
    assert _group_of(ds["id"], a3)["keep"] == max(a3, cp) == _export_keeps(ds["id"], [a3, cp])[0]   # the richer copy
    mm = _flags(ds["id"], "dup_label_mismatch")
    assert mm[a3]["twin"] == cp and mm[cp]["twin"] == a3 and mm[cp]["reason"] == "count"


def test_keep_is_exports_rule_and_follows_reviews(ds):
    """The duplicates view names the image Export keeps: approved first, then labeled, most boxes, oldest —
    decided when asked, so approving an image (no label write, no QA run) changes it at once."""
    a3, cp = _id(ds, "train/camA_0003.jpg"), _id(ds, "val/copy_camA_0003.jpg")
    newer = max(a3, cp)
    rel = db.query_one("SELECT rel_path FROM images WHERE id=?", (newer,))["rel_path"]
    assert _group_of(ds["id"], a3)["keep"] == min(a3, cp)
    db.execute("INSERT OR REPLACE INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)",
               (ds["id"], rel, "approved", "", db.now()))
    try:
        g = _group_of(ds["id"], a3)
        assert g["keep"] == newer and _export_keeps(ds["id"], [a3, cp]) == [newer]
        assert "keep" not in _flags(ds["id"], "near_dup")[a3]        # never stored: it could only go stale
    finally:
        db.execute("DELETE FROM reviews WHERE dataset_id=? AND rel_path=?", (ds["id"], rel))
    assert _group_of(ds["id"], a3)["keep"] == min(a3, cp)


def test_shape_rescored_against_stored_class_statistics(ds, edits, monkeypatch):
    monkeypatch.setattr(qa, "SHAPE_FULL_MAX_BOXES", 0)          # the path for datasets with huge box counts
    shape = lambda: _flags(ds["id"], "shape_outlier")  # noqa: E731
    wide, other = _id(ds, "train/IMG_0002.jpg"), _id(ds, "train/IMG_0003.jpg")
    assert wide in shape() and other not in shape()
    model = qa.load_summary(ds["id"])["shape_model"]
    assert "4" in model and len(model["4"]) == 5
    from fovea.core import labels as labels_mod
    lines = _label_path(ds, "train/IMG_0002.jpg").read_text().splitlines()
    edits("train/IMG_0002.jpg", "4 0.500000 0.500000 0.060000 0.096000\n" + "\n".join(lines[1:]) + "\n", notify=False)
    lines3 = _label_path(ds, "train/IMG_0003.jpg").read_text().splitlines()
    edits("train/IMG_0003.jpg", "\n".join(lines3) + "\n4 0.500000 0.300000 0.600000 0.060000\n", notify=False)
    labels_mod.notify_label_write(ds["id"], [wide, other])     # one bulk write of two files
    assert qa.flush_label_rechecks(ds["id"])
    assert wide not in shape()
    assert other in shape() and shape()[other]["rows"][0]["row"] == 6
    assert qa._refresher.last_run[ds["id"]]["shape"] == 2          # just the two edited images were scored


def test_shape_statistics_recomputed_when_a_class_appears(ds, edits, monkeypatch):
    monkeypatch.setattr(qa, "SHAPE_FULL_MAX_BOXES", 0)
    many = [[5, 0.05 + 0.03 * k, 0.5, 0.02, 0.05] for k in range(30)] + [[5, 0.5, 0.2, 0.6, 0.02]]
    edits("train/misc3.jpg", _rows(many))
    assert qa.flush_label_rechecks(ds["id"])
    assert qa._refresher.last_run[ds["id"]]["shape"] == "full"
    assert "5" in qa.load_summary(ds["id"])["shape_model"]
    hits = _flags(ds["id"], "shape_outlier")[_id(ds, "train/misc3.jpg")]["rows"]
    assert hits[0]["cls"] == 5 and hits[0]["row"] == 30


def test_saves_are_debounced_and_coalesced(ds, monkeypatch):
    calls = []
    qa.flush_label_rechecks()                                    # writes left over by other tests' datasets
    monkeypatch.setattr(qa, "LABEL_DEBOUNCE", 0.3)
    monkeypatch.setattr(qa, "recheck_labels",
                        lambda d, ids: (calls.append((d, sorted(ids))) if d == ds["id"] else None) or {"images": len(ids)})
    a, b = _id(ds, "train/camA_0001.jpg"), _id(ds, "train/camA_0002.jpg")
    qa.on_label_write(ds["id"], [a])
    time.sleep(0.1)
    qa.on_label_write(ds["id"], [b, a])
    qa.on_label_write(ds["id"], ["junk", None])
    assert qa._refresher.busy(ds["id"]) == {a, b}
    t0 = time.time()
    while not calls and time.time() - t0 < 5:
        time.sleep(0.02)
    assert calls == [(ds["id"], sorted([a, b]))]
    assert time.time() - t0 >= 0.15                              # waited for the edits to pause
    time.sleep(0.4)
    assert len(calls) == 1 and qa._refresher.busy(ds["id"]) == set()


def test_steady_saves_are_rechecked_within_the_max_wait(ds, monkeypatch):
    calls = []
    qa.flush_label_rechecks()
    monkeypatch.setattr(qa, "LABEL_DEBOUNCE", 0.5)
    monkeypatch.setattr(qa, "LABEL_MAX_WAIT", 1.0)
    monkeypatch.setattr(qa, "recheck_labels",
                        lambda d, ids: (calls.append(sorted(ids)) if d == ds["id"] else None) or {"images": len(ids)})
    ids = [_id(ds, f"train/camA_{k:04d}.jpg") for k in range(1, 9)]
    t0 = time.time()
    for iid in ids:                                              # a save every 0.25 s never pauses for 0.5 s
        qa.on_label_write(ds["id"], [iid])
        time.sleep(0.25)
    assert calls and time.time() - t0 < 2.5
    qa.flush_label_rechecks(ds["id"])
    assert sorted(i for c in calls for i in c) == sorted(ids)


def test_a_large_edit_queues_one_label_check_job(ds, monkeypatch):
    monkeypatch.setattr(qa, "INCR_MAX_IMAGES", 2)
    ids = [_id(ds, f"train/camA_{k:04d}.jpg") for k in range(1, 6)]
    assert qa.recheck_labels(ds["id"], ids) == {"full": True, "images": 5}
    qa.on_label_write(ds["id"], ids)
    assert qa.flush_label_rechecks(ds["id"])
    job = jobs.list(ds["id"], "qa")[0]
    assert job.extra["checks"] == list(qa.LABEL_CHECKS)
    _wait_idle(ds["id"])
    assert job.status == "done"


def test_recheck_errors_never_reach_the_writer(ds, monkeypatch):
    from fovea.core import labels as labels_mod
    logged = []

    def boom(d, ids):
        raise RuntimeError("broken")
    monkeypatch.setattr(qa, "recheck_labels", boom)
    monkeypatch.setattr(qa.log, "exception", lambda msg, *a, **k: logged.append(msg % a))
    labels_mod.notify_label_write(ds["id"], [_id(ds, "train/camA_0001.jpg")])
    assert qa.flush_label_rechecks(ds["id"])
    assert qa._refresher.busy(ds["id"]) == set()
    assert logged and "Re-checking edited labels" in logged[0]


def test_recheck_skips_a_dataset_that_was_never_checked(tmp_path):
    root = tmp_path / "never"
    _write(root, "train", "v_000", _scene(5), [[0, 0.5, 0.5, 0.1, 0.1]])
    ds_id = _register(root)
    try:
        from fovea.core import scanner
        saved = list(scanner.POST_SCAN_HOOKS)
        scanner.POST_SCAN_HOOKS[:] = []
        try:
            scanner.scan_dataset(ds_id, Job(id="scan-never", kind="scan"))
        finally:
            scanner.POST_SCAN_HOOKS[:] = saved
        iid = db.query_one("SELECT id FROM images WHERE dataset_id=?", (ds_id,))["id"]
        assert qa.recheck_labels(ds_id, [iid])["images"] == 0
        assert qa.recheck_labels("no-such-dataset", [1])["images"] == 0
    finally:
        _drop(ds_id)


def test_put_labels_endpoint_announces_the_write(ds, edits):
    """End to end through backend-data's editor save (skipped while that path does not announce writes yet)."""
    from fastapi.testclient import TestClient
    from fovea.main import app
    i5 = _id(ds, "train/camA_0005.jpg")
    edits("train/camA_0005.jpg", _rows([P(5), S]), notify=False)            # registers the revert
    qa.flush_label_rechecks(ds["id"])
    c = TestClient(app)
    r = c.put(f"/api/datasets/{ds['id']}/images/{i5}/labels", json={"boxes": [P(5)]})
    assert r.status_code == 200, r.text
    if not qa._refresher.busy(ds["id"]) and i5 not in _flags(ds["id"], "seq_gap"):
        pytest.skip("PUT labels does not call notify_label_write in this tree yet")
    assert qa.flush_label_rechecks(ds["id"])
    assert i5 in _flags(ds["id"], "seq_gap")


# --------------------------------------------------------------------------- wave 2a: summary status, reruns, settings
def _blocking_run(monkeypatch):
    gate = threading.Event()
    started = threading.Event()
    seen = []

    def fake(ds_id, job, checks=None, options=None):
        job.update(extra={"checks": qa.normalise_checks(checks)})
        seen.append(qa.normalise_checks(checks))
        started.set()
        while not gate.is_set():
            job.update(done=1, total=10, message="Hashing images")
            time.sleep(0.02)
        return {"checks": checks}
    monkeypatch.setattr(qa, "run_qa", fake)
    return gate, started, seen


def test_summary_is_running_whenever_a_qa_job_is_queued_or_running(ds, monkeypatch):
    from fovea.api import qa as api
    gate, started, seen = _blocking_run(monkeypatch)
    queued = Job(id="queued-" + uuid.uuid4().hex[:6], kind="qa", dataset_id=ds["id"], message="Queued after indexing")
    jobs._jobs[queued.id] = queued                                # a queued job (pool busy) is "running" too
    try:
        s = api.qa_summary(ds["id"], auto=0)
        assert s["status"] == "running" and s["job"]["id"] == queued.id and s["job"]["status"] == "queued"
        queued.status = "cancelled"
        job, already = qa.queue_qa(ds["id"], ["quality"], message="Queued after indexing")
        assert not already and started.wait(5)
        newer = Job(id="newer-" + uuid.uuid4().hex[:6], kind="qa", dataset_id=ds["id"], message="Queued")
        jobs._jobs[newer.id] = newer                              # newest, but the running one is what to show
        s = api.qa_summary(ds["id"], auto=1)
        assert s["status"] == "running" and s["job"]["id"] == job.id and s["job"]["message"] == "Hashing images"
        assert s["job"]["total"] == 10 and s["job"]["extra"]["checks"] == ["quality"]
        newer.status = "cancelled"
        jobs._jobs.pop(newer.id, None)
    finally:
        gate.set()
        jobs._jobs.pop(queued.id, None)
        _wait_idle(ds["id"])


def test_a_request_during_a_run_is_rerun_after_it(ds, monkeypatch):
    gate, started, seen = _blocking_run(monkeypatch)
    try:
        first, _ = qa.queue_qa(ds["id"], ["quality"])
        assert started.wait(5)
        again, already = qa.queue_qa(ds["id"], ["shape"])
        assert already and again is first
    finally:
        gate.set()
    t0 = time.time()
    while len(seen) < 2 and time.time() - t0 < 10:
        time.sleep(0.02)
    assert seen == [["quality"], ["shape"]]
    _wait_idle(ds["id"])


def test_qa_settings_contract_and_rerun(ds):
    from fastapi.testclient import TestClient
    from fovea.main import app
    c = TestClient(app)
    url = f"/api/datasets/{ds['id']}/qa/settings"
    g = c.get(url).json()
    assert set(g["options"]) == set(qa.DEFAULTS) and g["overrides"] == {}
    f = {x["key"]: x for x in g["fields"]}
    assert set(f) == set(qa.DEFAULTS)
    assert f["blur_z"]["check"] == "quality" and f["blur_z"]["codes"] == ["blurry", "dark", "overexposed"]
    assert f["dup_dist"]["integer"] and f["dup_dist"]["min"] == 0 and f["dup_dist"]["max"] == 16 and f["dup_dist"]["label"]
    assert not f["blur_z"]["advanced"] and f["seq_cut"]["advanced"]
    r = c.put(url, json={"options": {"blur_z": 3.0, "dark_floor": 40}})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["changed"] == ["blur_z", "dark_floor"] and body["overrides"] == {"blur_z": 3.0, "dark_floor": 40}
    assert body["job"] and body["job"]["extra"]["checks"] == ["quality"]
    _wait_idle(ds["id"])
    assert c.get(f"/api/datasets/{ds['id']}/qa/summary?auto=0").json()["options"]["blur_z"] == 3.0
    same = c.put(url, json={"options": {"blur_z": 3.0}}).json()
    assert same["changed"] == [] and same["job"] is None
    r = c.put(url, json={"reset": ["blur_z"], "rerun": False}).json()
    assert r["changed"] == ["blur_z"] and r["job"] is None and r["overrides"] == {"dark_floor": 40}
    assert c.put(url, json={"options": {"dup_dist": 99}}).status_code == 400
    assert c.put(url, json={"options": {"nope": 1}}).status_code == 400
    assert c.put(url, json={"reset": ["nope"]}).status_code == 400
    assert c.put(url, json={"reset": "yes"}).status_code == 400
    r = c.put(url, json={"reset": True}).json()
    assert r["overrides"] == {} and r["changed"] == ["dark_floor"] and r["job"]["extra"]["checks"] == ["quality"]
    _wait_idle(ds["id"])
    assert c.get("/api/datasets/nope/qa/settings").status_code == 404
    assert _id(ds, "train/soft.jpg") in _flags(ds["id"], "blurry")


def test_shape_recheck_is_exact_after_a_bulk_class_change(ds, edits):
    """A class change on many boxes moves the class statistics: boxes of images nobody touched can cross the
    threshold. The re-check re-runs the whole (numpy) pass, so it agrees with a full run."""
    from fovea.core import labels as labels_mod
    changed = []
    for k in range(1, 13):
        rel = f"train/camA_{k:04d}.jpg"
        rows = [ln.split() for ln in _label_path(ds, rel).read_text().splitlines() if ln.strip()]
        edits(rel, "".join(" ".join(["4"] + r[1:]) + "\n" for r in rows if r[0] == "0") +
              "".join(" ".join(r) + "\n" for r in rows if r[0] != "0"), notify=False)
        changed.append(_id(ds, rel))
    labels_mod.notify_label_write(ds["id"], changed)
    assert qa.flush_label_rechecks(ds["id"])
    last = qa._refresher.last_run[ds["id"]]
    assert last["shape"] == "full" and last["timings"]["shape_cache"] is True     # swapped rows, no re-read
    incremental = _flags(ds["id"], "shape_outlier")
    qa.run_qa(ds["id"], Job(id="qa-shape-full", kind="qa"), ["shape"])
    assert incremental == _flags(ds["id"], "shape_outlier")
    assert qa.load_summary(ds["id"])["shape_model"]["4"][4] == 36 + 11
    # a cache that disagrees with the index (a write that bypassed the hooks) is rebuilt, still exact
    A = qa._shape_cache[ds["id"]]
    qa._shape_cache[ds["id"]] = {k: v[3:] for k, v in A.items()}
    edits("train/camA_0012.jpg", _rows([P(12)]))
    assert qa.flush_label_rechecks(ds["id"])
    assert qa._refresher.last_run[ds["id"]]["timings"]["shape_cache"] is False
    incremental = _flags(ds["id"], "shape_outlier")
    qa.run_qa(ds["id"], Job(id="qa-shape-full2", kind="qa"), ["shape"])
    assert incremental == _flags(ds["id"], "shape_outlier")


def test_shape_pass_numpy_and_python_agree(ds, monkeypatch):
    rng = random.Random(9)
    rows = qa.shape_rows(ds["id"])
    for k in range(3000):                       # plus a synthetic class with outliers, cut and degenerate boxes
        w, h = rng.uniform(0.02, 0.2), rng.uniform(0.02, 0.2)
        if k % 97 == 0:
            w, h = 0.9, 0.01
        if k % 211 == 0:
            w = 0.0
        rows.append((10 ** 6 + k // 3, k % 3, 7, rng.uniform(0.0, 1.0), rng.uniform(0.0, 1.0), w, h, 640.0, 480.0))
    opt = dict(qa.DEFAULTS)
    f_np, i_np = {}, {}
    qa.shape_pass(rows, opt, f_np, i_np)
    monkeypatch.setattr(qa, "_np_mod", None)
    f_py, i_py = {}, {}
    qa.shape_pass(rows, opt, f_py, i_py)
    assert f_np == f_py and i_np["shape_model"] == i_py["shape_model"] and i_np["shape_classes"] == i_py["shape_classes"]
    assert any(k[0] >= 10 ** 6 for k in f_np)


def test_neighbour_search_matches_brute_force(ds, monkeypatch):
    ids, fines, _w = qa._hash_matrix(ds["id"])
    by_id = dict(zip(ids, fines))
    a3, cp = _id(ds, "train/camA_0003.jpg"), _id(ds, "val/copy_camA_0003.jpg")
    query = [by_id[i] for i in (a3, cp, _id(ds, "test/scene_1.jpg"), _id(ds, "train/night.jpg"))]
    for max_dist in (0, 2, 40):
        want = {i for i, f in by_id.items() if any((f ^ h).bit_count() <= max_dist for h in query)}
        assert qa.neighbour_ids(ds["id"], query, max_dist) == want
        monkeypatch.setattr(qa, "_np_mod", None)
        qa._sim_cache.clear()
        assert qa.neighbour_ids(ds["id"], query, max_dist) == want
        monkeypatch.undo()
        qa._sim_cache.clear()
    assert {a3, cp} <= qa.neighbour_ids(ds["id"], [by_id[a3]], 0)


def test_neighbourhood_of_a_static_camera_costs_one_flag_per_image(monkeypatch):
    """8,000 identical frames of a static camera: the re-check's neighbourhood is their union (8,000 ids),
    never a list per query image (64 million (id, dist) pairs, 4 GB, as the first version built)."""
    import tracemalloc
    np = pytest.importorskip("numpy")
    rng = random.Random(4)
    h, other = rng.getrandbits(256), rng.getrandbits(256)
    ids = list(range(1, 8002))
    fines = [h] * 8000 + [other]
    words = np.array([[(f >> s) & ((1 << 64) - 1) for s in (192, 128, 64, 0)] for f in fines], dtype=np.uint64)
    monkeypatch.setattr(qa, "_hash_matrix", lambda _ds: (ids, fines, words))
    tracemalloc.start()
    try:
        got = qa.neighbour_ids("static", [h] * 7999, 2)
        peak = tracemalloc.get_traced_memory()[1]
    finally:
        tracemalloc.stop()
    assert got == set(range(1, 8001)) and peak < 16e6, peak


# --------------------------------------------------------------------------- split roles, pixel cap, scale
def _build_roles(root: Path) -> None:
    """One video split over two folders of the TEST role (test, test-1854: one frame in both), another video
    in train whose frame 2 was copied into val (a real leak)."""
    bg = _scene(61)
    for k in range(1, 9):
        im = bg.copy()
        _draw_box(im, P(k))
        _write(root, "test" if k <= 5 else "test-1854", f"cam_{k:04d}", im, [P(k)])
    shutil.copyfile(root / "images" / "test" / "cam_0003.jpg", root / "images" / "test-1854" / "cam_0003.jpg")
    shutil.copyfile(root / "labels" / "test" / "cam_0003.txt", root / "labels" / "test-1854" / "cam_0003.txt")
    bg2 = _scene(62)
    (root / "images" / "val").mkdir(parents=True)
    (root / "labels" / "val").mkdir(parents=True)
    for k in range(1, 6):
        im = bg2.copy()
        _draw_box(im, [0, 0.2 + 0.1 * k, 0.4, 0.1, 0.3])
        _write(root, "train", f"walk_{k:04d}", im, [[0, 0.2 + 0.1 * k, 0.4, 0.1, 0.3]])
    shutil.copyfile(root / "images" / "train" / "walk_0002.jpg", root / "images" / "val" / "walk_0002.jpg")
    shutil.copyfile(root / "labels" / "train" / "walk_0002.txt", root / "labels" / "val" / "walk_0002.txt")


@pytest.fixture(scope="module")
def roles_ds(tmp_path_factory):
    from fovea.core import scanner
    hooks = list(scanner.POST_SCAN_HOOKS)
    scanner.POST_SCAN_HOOKS[:] = [h for h in hooks if h is not qa.on_scan]
    root = tmp_path_factory.mktemp("roles") / "ds"
    _build_roles(root)
    ds_id = _register(root)
    try:
        scanner.scan_dataset(ds_id, Job(id="scan-" + ds_id, kind="scan"))
    finally:
        scanner.POST_SCAN_HOOKS[:] = hooks
    qa.run_qa(ds_id, Job(id="qa-" + ds_id, kind="qa"))
    ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
    yield {"id": ds_id, "root": root, "ids": ids}
    _drop(ds_id)


def test_two_folders_of_one_split_role_are_not_a_leak(roles_ds):
    d = roles_ds
    splits = {r["split"] for r in db.query("SELECT DISTINCT split FROM images WHERE dataset_id=?", (d["id"],))}
    assert splits == {"train", "val", "test", "test-1854"}
    assert qa.split_role("test-1854") == qa.split_role("test") == "test" and qa.split_role("") == "train"
    t3, x3 = _id(d, "test/cam_0003.jpg"), _id(d, "test-1854/cam_0003.jpg")
    w2, v2 = _id(d, "train/walk_0002.jpg"), _id(d, "val/walk_0002.jpg")
    leak = _flags(d["id"], "leak_split")
    walk = {v for k, v in d["ids"].items() if "/walk_" in k}
    assert set(leak) <= walk and {w2, v2} <= set(leak)           # only the train / val video leaks
    assert leak[v2]["reason"] == "twin" and leak[v2]["splits"] == ["train"] and leak[v2]["roles"] == ["train"]
    assert leak[v2]["twin"] == w2 and leak[w2]["twin"] == v2
    assert {leak[i]["reason"] for i in walk - {w2, v2} if i in leak} <= {"seq"}   # same view of that video
    same = _flags(d["id"], "dup_same_role")
    assert same[t3]["twin"] == x3 and same[t3]["splits"] == ["test-1854"] and same[t3]["role"] == "test"
    assert same[x3]["twin"] == t3 and same[t3]["dist"] == 0
    assert w2 not in same and v2 not in same
    s = qa.load_summary(d["id"])
    span = {r["seq"]: r for r in s["split_spanning_sequences"]}
    assert "cam" not in span                                      # one video, two test folders: not listed
    assert span["walk"]["roles"] == ["train", "val"] and span["walk"]["splits"] == ["train", "val"]
    dup = qa.duplicate_groups(d["id"], 0, 50, cross_split=True)
    assert [sorted(m["image_id"] for m in g["members"]) for g in dup["groups"]] == [sorted([w2, v2])]
    every = qa.duplicate_groups(d["id"], 0, 50)
    tg = next(g for g in every["groups"] if t3 in {m["image_id"] for m in g["members"]})
    assert tg["splits"] == ["test", "test-1854"] and tg["roles"] == ["test"] and not tg["cross_split"]


def test_same_role_overlap_is_labeled_and_filterable(roles_ds):
    from fastapi.testclient import TestClient
    from fovea.core import labels as labels_mod
    from fovea.main import app
    d = roles_ds
    assert labels_mod.FLAG_LABELS["dup_same_role"] == qa.LABELS["dup_same_role"]
    c = TestClient(app)
    s = c.get(f"/api/datasets/{d['id']}/qa/summary?auto=0").json()
    assert s["counts"]["dup_same_role"] >= 2 and s["counts"]["leak_split"] == len(_flags(d["id"], "leak_split")) >= 2
    assert s["labels"]["dup_same_role"] and "dup_same_role" in s["checks"]["duplicates"]
    r = c.get(f"/api/datasets/{d['id']}/images", params={"issue": "dup_same_role", "limit": 50}).json()
    assert {_id(d, "test/cam_0003.jpg"), _id(d, "test-1854/cam_0003.jpg")} <= {it["id"] for it in r["items"]}


def test_dataset_card_counts_leaks_like_the_overview(roles_ds):
    """One definition: the card's leak row is the QA engine's leak_split, judged on the export's own roles."""
    from fovea.core import export
    d = roles_ds
    flagged = set(_flags(d["id"], "leak_split"))
    all_ids = set(d["ids"].values())
    roles = {r["id"]: qa.split_role(r["split"]) for r in db.query("SELECT id, split FROM images WHERE dataset_id=?", (d["id"],))}
    assert qa.leaked_images(d["id"], roles) == flagged
    # an export that maps test-1854 to val makes the shared test frames a leak (and sees the video in two splits)
    remapped = {i: ("val" if r["split"] == "test-1854" else roles[i]) for r in db.query(
        "SELECT id, split FROM images WHERE dataset_id=?", (d["id"],)) for i in [r["id"]]}
    got = qa.leaked_images(d["id"], remapped)
    assert flagged <= got and {_id(d, "test/cam_0003.jpg"), _id(d, "test-1854/cam_0003.jpg")} <= got
    # a selection without the val copy has nothing to leak into
    assert qa.leaked_images(d["id"], {i: roles[i] for i in all_ids - {_id(d, "val/walk_0002.jpg")}}) == set()
    card = export.card_for_dataset(d["id"], [], {})
    row = next(ln for ln in card.splitlines() if "across splits" in ln)
    assert f"| {len(flagged)} |" in row.replace("**", ""), row
    span = next(ln for ln in card.splitlines() if ln.startswith("| Sequences spanning splits"))
    assert "1 of 2 sequences" in span, span
    # no duplicate check of its own to go by: the card keeps counting near-duplicate groups
    t3 = _id(d, "test/cam_0003.jpg")
    row = db.query_one("SELECT detail FROM image_flags WHERE image_id=? AND code='near_dup'", (t3,))
    db.execute("UPDATE image_flags SET detail=? WHERE image_id=? AND code='near_dup'", (db.dumps({"group": 1}), t3))
    try:
        assert qa.leaked_images(d["id"], roles) is None
    finally:
        db.execute("UPDATE image_flags SET detail=? WHERE image_id=? AND code='near_dup'", (row["detail"], t3))
    assert qa.leaked_images("no-such-dataset", roles) is None


def test_results_of_an_older_engine_are_rechecked(roles_ds):
    from fovea.api import qa as api
    d = roles_ds
    assert qa.load_summary(d["id"])["engine"] == qa.ENGINE_VERSION
    with qa._summary_lock:
        s = qa.load_summary(d["id"])
        s["engine"] = 1                                           # written before leaks were judged by role
        qa._save_summary(d["id"], s)
    assert api.qa_summary(d["id"], auto=0)["status"] == "stale"
    api._auto_at.pop(d["id"], None)
    r = api.qa_summary(d["id"], auto=1)
    assert r["status"] == "running" and r["job"]["extra"]["checks"] == ["duplicates"]
    _wait_idle(d["id"])
    assert qa.load_summary(d["id"])["engine"] == qa.ENGINE_VERSION
    assert api.qa_summary(d["id"], auto=0)["status"] == "ok"


def test_pillow_bomb_guard_is_never_lifted():
    import PIL.Image
    from fovea import config
    from fovea.api import objects  # noqa: F401 — importing a decoder module must not lift the guard
    from fovea.ai import vlm  # noqa: F401
    assert PIL.Image.MAX_IMAGE_PIXELS == config.MAX_IMAGE_PIXELS


def test_rasters_beyond_the_pixel_cap_are_not_decoded(tmp_path, monkeypatch):
    import PIL.Image
    big = tmp_path / "big.png"
    Image.new("L", (1200, 1000), 90).save(big)                    # 1.2 MP: over the cap below
    jpg = tmp_path / "big.jpg"
    _scene(3).resize((1200, 1000)).save(jpg, quality=90)
    monkeypatch.setattr(qa, "MAX_IMAGE_PIXELS", 1_000_000)
    with pytest.raises(qa.TooLarge):
        qa._analysis_image(str(big))
    assert qa.image_features(str(big)) is None
    assert qa.image_features(str(jpg)) is not None                # JPEG decodes at 1/8 scale: well under the cap
    monkeypatch.setattr(PIL.Image, "MAX_IMAGE_PIXELS", 500_000)   # Pillow itself refuses > 2x its limit at open
    with pytest.raises(qa.TooLarge):
        qa._analysis_image(str(big))


def test_hashing_counts_too_large_images(tmp_path, monkeypatch):
    root = tmp_path / "huge"
    for k in range(3):
        _write(root, "train", f"v_{k:03d}", _scene(80 + k), [[0, 0.5, 0.5, 0.1, 0.1]])
    p = root / "images" / "train" / "huge.png"
    Image.new("RGB", (1500, 1000), (40, 40, 40)).save(p)
    ds_id = _register(root)
    try:
        from fovea.core import scanner
        saved = list(scanner.POST_SCAN_HOOKS)
        scanner.POST_SCAN_HOOKS[:] = []
        try:
            scanner.scan_dataset(ds_id, Job(id="scan-huge", kind="scan"))
        finally:
            scanner.POST_SCAN_HOOKS[:] = saved
        monkeypatch.setattr(qa, "MAX_IMAGE_PIXELS", 1_000_000)
        res = qa.update_hashes(ds_id)
        assert res["too_large"] == 1 and res["failed"] == 1 and res["hashed"] == 4
        assert qa.update_hashes(ds_id)["hashed"] == 0                # remembered until the file changes
    finally:
        _drop(ds_id)


def test_objects_and_ai_decoders_refuse_oversized_rasters(tmp_path, monkeypatch):
    from fovea import config
    from fovea.ai import vlm
    from fovea.ai.base import ProviderError
    from fovea.api import objects
    from fovea.core.thumbs import ImageTooLarge
    big = tmp_path / "big.png"
    Image.new("L", (1200, 1000), 90).save(big)
    monkeypatch.setattr(objects, "MAX_IMAGE_PIXELS", 1_000_000)
    with pytest.raises(ImageTooLarge):
        objects._decode(str(big), 1)
    monkeypatch.setattr(config, "MAX_IMAGE_PIXELS", 1_000_000)
    with pytest.raises(ProviderError) as e:
        vlm.open_image(str(big))
    assert "too large" in str(e.value)
    monkeypatch.undo()
    assert objects._decode(str(big), 1).size == (1200, 1000) and vlm.open_image(str(big)).size == (1200, 1000)


def test_header_probes_never_decode_an_oversized_png(tmp_path, monkeypatch):
    """Pillow's PNG getexif() decodes every pixel to look for a trailing eXIf chunk: the scanner, thumbnail and
    crop probes must not do that for a PNG beyond the cap (a 389 KB file declaring 20000 x 20000 cost 436 MB)."""
    from PIL import PngImagePlugin
    from fovea.api import objects
    from fovea.core import scanner, thumbs
    big = tmp_path / "big.png"
    Image.new("L", (1200, 1000), 90).save(big)
    loads = []
    real = PngImagePlugin.PngImageFile.load
    monkeypatch.setattr(PngImagePlugin.PngImageFile, "load", lambda self: loads.append(1) or real(self))
    for mod in (scanner, thumbs):
        monkeypatch.setattr(mod, "MAX_IMAGE_PIXELS", 1_000_000)
    thumbs._probe.cache_clear()
    assert scanner.read_dims(str(big)) == (1200, 1000)
    pr = thumbs.probe(str(big))
    assert pr.ok and (pr.width, pr.height, pr.orient) == (1200, 1000, 1)
    assert objects._probe(str(big))[:3] == (1200, 1000, 1)
    assert loads == []
    monkeypatch.setattr(thumbs, "MAX_IMAGE_PIXELS", 10_000_000)       # under the cap: EXIF is read as before
    thumbs._probe.cache_clear()
    assert thumbs.probe(str(big)).ok and loads
    thumbs._probe.cache_clear()


def test_static_camera_edit_rechecks_the_whole_group_or_queues_a_job(tmp_path, monkeypatch):
    """A save in a group of identical frames re-checks every twin (incremental == full); a neighbourhood
    beyond INCR_MAX_FRAMES queues the ordinary job before building anything."""
    root = tmp_path / "static"
    frame = root / "images" / "train" / "cam_0000.jpg"
    _write(root, "train", "cam_0000", _scene(90), [[0, 0.5, 0.5, 0.2, 0.3]])
    for k in range(1, 120):
        dst = root / "images" / "train" / f"cam_{k:04d}.jpg"
        os.link(frame, dst)
        (root / "labels" / "train" / f"cam_{k:04d}.txt").write_text("0 0.500000 0.500000 0.200000 0.300000\n")
    ds_id = _register(root)
    d = {"id": ds_id, "root": root, "ids": {}}
    try:
        from fovea.core import scanner
        saved = list(scanner.POST_SCAN_HOOKS)
        scanner.POST_SCAN_HOOKS[:] = []
        try:
            scanner.scan_dataset(ds_id, Job(id="scan-static", kind="scan"))
        finally:
            scanner.POST_SCAN_HOOKS[:] = saved
        qa.run_qa(ds_id, Job(id="qa-static", kind="qa"), ["duplicates", "sequence"])
        d["ids"] = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        _set_labels(d, "train/cam_0050.jpg", _rows([[0, 0.5, 0.5, 0.2, 0.3], [1, 0.1, 0.1, 0.05, 0.05]]), notify=False)
        res = qa.recheck_labels(ds_id, [_id(d, "train/cam_0050.jpg")])
        assert res["twins"] == 120 and not res.get("full")
        inc = _all_label_flags(ds_id)
        qa.run_qa(ds_id, Job(id="qa-static-full", kind="qa"), ["labels"])
        assert inc == _all_label_flags(ds_id) and (_id(d, "train/cam_0050.jpg"), "dup_label_mismatch") in inc
        monkeypatch.setattr(qa, "INCR_MAX_FRAMES", 150)              # 120 sequence frames + 120 twins
        built = []
        monkeypatch.setattr(qa, "load_subset", lambda *a, **k: built.append(1) or [])
        assert qa.recheck_labels(ds_id, [_id(d, "train/cam_0050.jpg")]) == {"full": True, "images": 1}
        assert not built
    finally:
        _drop(ds_id)


# --------------------------------------------------------------------------- round 2: worker process, NumPy 1.x
def _every_flag(ds_id):
    return {(r["image_id"], r["code"]): (r["score"], db.loads(r["detail"], {})) for r in db.query(
        "SELECT image_id, code, score, detail FROM image_flags WHERE dataset_id=?", (ds_id,))}


_SUMMARY_KEYS = ("images", "hashed", "dup_groups", "exact_dups", "split_spanning_sequences", "sequences",
                 "sequence_frames", "shape_model", "quality")


def _in_server_only(*_a, **_k):
    raise AssertionError("computed in the server process")


def test_a_large_run_computes_in_a_worker_process(ds, monkeypatch):
    """At scale the checks hold the GIL for seconds and starve every request thread (a 100-row page took 1 s):
    a large run computes in a child process and the server only stores what it sends — the same results."""
    qa.run_qa(ds["id"], Job(id="qa-thread", kind="qa"))
    want_flags, want_summary = _every_flag(ds["id"]), qa.load_summary(ds["id"])
    want_groups = qa.duplicate_groups(ds["id"], 0, 500)["groups"]
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    for name in ("load_items", "check_duplicates", "check_sequences", "check_shapes", "check_quality",
                 "spanning_sequences"):
        monkeypatch.setattr(qa, name, _in_server_only)          # the worker has its own, unpatched copies
    qa._shape_cache.pop(ds["id"], None)
    job = Job(id="qa-worker", kind="qa")
    res = qa.run_qa(ds["id"], job)
    assert res["process"] is True and res["images"] == want_summary["images"] and job.extra["phase"] == "done"
    assert _every_flag(ds["id"]) == want_flags
    got = qa.load_summary(ds["id"])
    assert {k: got.get(k) for k in _SUMMARY_KEYS} == {k: want_summary.get(k) for k in _SUMMARY_KEYS}
    assert set(got["timings"]) >= {"load", "duplicates", "sequence", "shape", "quality", "total"}
    assert qa.duplicate_groups(ds["id"], 0, 500)["groups"] == want_groups
    assert ds["id"] in qa._shape_cache                         # the worker's shape arrays feed the re-check cache
    assert not qa._live_workers


def test_worker_hashes_are_stored_by_the_server(ds, monkeypatch):
    from fovea.core import scanner
    monkeypatch.setattr(scanner, "POST_SCAN_HOOKS", [])
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    p = ds["root"] / "images" / "train" / "misc3.jpg"
    keep = p.read_bytes()
    scanner.scan_dataset(ds["id"], Job(id="rescan-w0", kind="scan"))  # whatever an earlier test left stale
    qa.update_hashes(ds["id"])
    try:
        _scene(4321).save(p, quality=90)
        os.utime(p, (time.time() + 7, time.time() + 7))
        scanner.scan_dataset(ds["id"], Job(id="rescan-w", kind="scan"))
        assert qa.stale_images(ds["id"]) == 1
        gen = qa._generation[ds["id"]]
        res = qa.run_qa(ds["id"], Job(id="qa-w-hash", kind="qa"), ["quality"])
        assert res["process"] is True and res["hashed"] == 1 and qa.stale_images(ds["id"]) == 0
        assert qa._generation[ds["id"]] > gen                   # the similar-search cache sees the new hash
        assert qa.load_summary(ds["id"])["hash"]["hashed"] == 1
    finally:
        p.write_bytes(keep)
        os.utime(p, (time.time() + 9, time.time() + 9))
        scanner.scan_dataset(ds["id"], Job(id="rescan-w2", kind="scan"))
        qa.update_hashes(ds["id"])


def test_cancelling_a_run_stops_its_worker(ds, monkeypatch):
    from fovea.jobs import JobCancelled
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    before = _every_flag(ds["id"])
    job = Job(id="qa-cancel", kind="qa")
    out = {}

    def run():
        try:
            out["res"] = qa.run_qa(ds["id"], job, ["labels"])
        except BaseException as e:  # noqa: BLE001
            out["err"] = e
    th = threading.Thread(target=run)
    th.start()
    t0 = time.time()
    while not qa._live_workers and time.time() - t0 < 10:
        time.sleep(0.002)
    procs = list(qa._live_workers.values())
    assert procs, "no worker started"
    job.cancel()
    th.join(15)
    assert isinstance(out.get("err"), JobCancelled), out
    assert all(p.poll() is not None for p in procs) and not qa._live_workers
    assert _every_flag(ds["id"]) == before                     # nothing half-written


def test_worker_errors_fail_the_job_with_their_message(ds):
    seen = []
    with pytest.raises(RuntimeError, match="KeyError"):
        qa._run_in_worker("run", {"dataset_id": ds["id"], "checks": ["duplicates"], "opt": {}, "rehash": False},
                          seen.append, lambda: None)
    with pytest.raises(ValueError, match="invalid literal"):     # a ValueError stays user-facing
        qa._run_in_worker("run", {"dataset_id": ds["id"], "checks": ["duplicates"],
                                  "opt": {**qa.DEFAULTS, "dup_dist": "x"}, "rehash": False}, seen.append, lambda: None)
    assert not qa._live_workers and not [m for m in seen if m[0] in ("flags", "done")]


@pytest.mark.parametrize("exe", ["/nonexistent/python3", shutil.which("false") or "/usr/bin/false"])
def test_a_worker_that_cannot_start_falls_back_to_the_job_thread(ds, monkeypatch, exe):
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    monkeypatch.setattr(qa, "_worker_broken", [])
    before = _every_flag(ds["id"])
    monkeypatch.setattr(sys, "executable", exe)
    res = qa.run_qa(ds["id"], Job(id="qa-fallback", kind="qa"), ["labels"])
    assert res["process"] is False and qa._worker_broken and _every_flag(ds["id"]) == before
    started = []
    monkeypatch.setattr(qa, "_run_in_worker", lambda *a: started.append(1))
    assert qa.run_qa(ds["id"], Job(id="qa-fallback2", kind="qa"), ["shape"])["process"] is False
    assert not started                                         # remembered: no second attempt in this process


def test_a_large_recheck_runs_in_a_worker_process(ds, edits, monkeypatch):
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    monkeypatch.setattr(qa, "RECHECK_PROCESS_FRAMES", 0)
    for name in ("check_sequences", "mismatch_flags", "near_pairs"):
        monkeypatch.setattr(qa, name, _in_server_only)
    i = lambda k: f"train/camA_{k:04d}.jpg"  # noqa: E731
    edits(i(6), _rows([P(6), S]))                               # the gap is fixed
    edits(i(11), _rows([P(11)]))                                # a new gap
    edits("test/scene_2.jpg", _rows([[0, 0.5, 0.5, 0.1, 0.3]]))  # the twins now agree
    assert qa.flush_label_rechecks(ds["id"])
    last = qa._refresher.last_run[ds["id"]]
    assert last["process"] is True and last["images"] == 3
    incremental = _all_label_flags(ds["id"])
    monkeypatch.undo()
    qa.run_qa(ds["id"], Job(id="qa-full-w", kind="qa"), ["labels"])
    assert incremental == _all_label_flags(ds["id"])
    assert (_id(ds, i(11)), "seq_gap") in incremental and (_id(ds, i(6)), "seq_gap") not in incremental


def test_popcount_without_numpy_2(monkeypatch):
    """np.bitwise_count first appeared in NumPy 2.0; requirements.txt allows 1.26."""
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    a = rng.integers(0, np.iinfo(np.uint64).max, size=(7, 33), dtype=np.uint64, endpoint=True)
    a[0, 0], a[0, 1] = 0, np.iinfo(np.uint64).max

    def want(x):
        return np.array([int(v).bit_count() for v in x.ravel()], dtype=np.int64).reshape(x.shape)
    native = qa._popcount(np, a) if hasattr(np, "bitwise_count") else None
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    for x in (a, a[:, ::2], a.T, a[3], a[:0], a[:, 5:6] ^ a[None, 2, :]):
        got = qa._popcount(np, x)
        assert got.dtype == np.uint8 and got.shape == x.shape and (got.astype(np.int64) == want(x)).all()
    if native is not None:
        assert native.dtype == np.uint8 and (native == qa._popcount(np, a)).all()


def test_every_numpy_path_works_without_bitwise_count(ds, monkeypatch):
    """A NumPy 1.x install: near-duplicate buckets, sequence leakage, the re-check neighbourhood, similar
    images and a whole run give the same answers as with NumPy 2 (and as pure Python)."""
    np = pytest.importorskip("numpy")
    rng = random.Random(11)
    base = rng.getrandbits(256)
    # 450 hashes that differ only in their low 128 bits: the high bands are one bucket of 450 (> 400: numpy)
    hashes = [base ^ (1 << rng.randrange(128)) ^ (1 << rng.randrange(128)) for _ in range(450)]
    hashes += [rng.getrandbits(256) for _ in range(50)]
    big_buckets = []
    real_bucket_pairs = qa._bucket_pairs_np
    monkeypatch.setattr(qa, "_bucket_pairs_np", lambda *a: big_buckets.append(len(a[2])) or real_bucket_pairs(*a))
    want_pairs = sorted(qa.near_pairs(hashes, 3))
    assert big_buckets and max(big_buckets) >= 450
    qa.run_qa(ds["id"], Job(id="qa-np2", kind="qa"), ["duplicates", "sequence", "shape"])
    want_flags = _every_flag(ds["id"])
    a3 = _id(ds, "train/camA_0003.jpg")
    ids, fines, words = qa._hash_matrix(ds["id"])
    far = [rng.getrandbits(256) for _ in range(2100)]          # > 2000 hashes: similar() takes the numpy path
    big = (ids + [-(k + 1) for k in range(len(far))], fines + far,
           np.vstack([words, np.array([[(f >> s) & ((1 << 64) - 1) for s in (192, 128, 64, 0)] for f in far], dtype=np.uint64)]))
    monkeypatch.setattr(qa, "_hash_matrix", lambda _ds: big)
    want_similar = qa.similar(ds["id"], a3, 10, 40)["items"]
    query = [fines[ids.index(a3)], fines[ids.index(_id(ds, "test/scene_1.jpg"))]]
    want_near = qa.neighbour_ids(ds["id"], query, 2)

    monkeypatch.delattr(np, "bitwise_count", raising=False)
    big_buckets.clear()
    assert sorted(qa.near_pairs(hashes, 3)) == want_pairs and len(want_pairs) > 400 and big_buckets
    assert qa.similar(ds["id"], a3, 10, 40)["items"] == want_similar and want_similar
    assert qa.neighbour_ids(ds["id"], query, 2) == want_near and a3 in want_near
    monkeypatch.undo()
    monkeypatch.delattr(np, "bitwise_count", raising=False)
    res = qa.run_qa(ds["id"], Job(id="qa-np1", kind="qa"), ["duplicates", "sequence", "shape"])
    assert res["process"] is False and _every_flag(ds["id"]) == want_flags
    assert any(c == "leak_split" and d[1].get("reason") == "seq" for (_i, c), d in want_flags.items())
    monkeypatch.undo()
    monkeypatch.setattr(qa, "_np_mod", None)                   # and pure Python agrees with both
    assert sorted(qa.near_pairs(hashes, 3)) == want_pairs


# --------------------------------------------------------------------------- round 4: class changes, shared index
def _px(c, x0, y0, x1, y1, side=640):
    return [c, (x0 + x1) / 2 / side, (y0 + y1) / 2 / side, (x1 - x0) / side, (y1 - y0) / side]


def test_keyframe_matching_follows_a_box_that_changes_class():
    """In a fall dataset the class is the state: the person who falls is 'standing' in one keyframe and
    'fallen' in the other. Same-class boxes still pair anywhere; another class only in place."""
    stand_l, stand_r = _px(0, 170, 250, 220, 400), _px(0, 430, 250, 480, 400)
    fallen_l = _px(1, 150, 360, 300, 420)
    m = lambda a, b: qa.match_keyframes(a, b, 640, 640)  # noqa: E731
    assert m([stand_l, stand_r], [fallen_l, stand_r]) == [(0, 0), (1, 1)]     # the journey's two people
    assert m([stand_r, stand_l], [stand_r, fallen_l]) == [(0, 0), (1, 1)]     # order does not matter
    assert m([stand_l], [fallen_l]) == [(0, 0)]                               # the faller alone
    # someone else walks in while the first person falls: the faller keeps their box
    assert m([stand_l], [fallen_l, _px(0, 560, 250, 610, 400)]) == [(0, 0)]
    # a lone person of one class is still followed wherever it went (no gate within a class) ...
    assert m([stand_l], [_px(0, 560, 250, 610, 400)]) == [(0, 0)]
    # ... while another class never pairs across the frame, and a same-class box in place wins
    assert m([_px(0, 20, 250, 70, 400)], [_px(1, 480, 560, 630, 620)]) == []
    assert m([stand_l], [_px(1, 172, 250, 222, 400), _px(0, 174, 252, 224, 402)]) == [(0, 1)]
    # every same-class pairing of the existing keyframe test is unchanged
    assert m([[0, 0.2, 0.5, 0.1, 0.3], [1, 0.8, 0.2, 0.1, 0.1]], [[0, 0.6, 0.5, 0.1, 0.3], [1, 0.8, 0.2, 0.1, 0.1]]) == [(0, 0), (1, 1)]


def test_interpolating_a_fall_proposes_the_falling_person(tmp_path):
    """Keyframe 10: two people standing; keyframe 30: the left one has fallen. Every frame in between gets
    a box for both; the faller's takes the nearer keyframe's class and says so."""
    root = tmp_path / "fall"
    bg = _scene(5)
    stand_l, stand_r, fallen_l = _px(0, 170, 250, 220, 400), _px(0, 430, 250, 480, 400), _px(1, 150, 360, 300, 420)
    for f in (10, 15, 20, 25, 30):
        boxes = {10: [stand_l, stand_r], 30: [fallen_l, stand_r]}.get(f)
        if f == 25:                                   # already labeled here: lying, where the lerp puts it
            boxes = [[1] + qa.lerp_box(stand_l, fallen_l, 0.75)[1:]]
        _write(root, "val", f"fall_{f:04d}", bg.copy(), boxes)
    ds_id = _register(root)
    try:
        from fovea.core import scanner
        saved = list(scanner.POST_SCAN_HOOKS)
        scanner.POST_SCAN_HOOKS[:] = []
        try:
            scanner.scan_dataset(ds_id, Job(id="scan-fall", kind="scan"))
        finally:
            scanner.POST_SCAN_HOOKS[:] = saved
        db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(["standing", "fallen"]), ds_id))
        d = {"ids": {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}}
        f = lambda k: _id(d, f"val/fall_{k:04d}.jpg")  # noqa: E731
        res = qa.interpolate(ds_id, f(10), f(30))
        assert res["matched"] == 2 and res["unmatched"] == {"from": 0, "to": 0}
        assert res["class_changes"] == [{"from_box": 0, "to_box": 0, "from_cls": 0, "to_cls": 1}]
        assert "standing → fallen" in res["message"]
        by = {}
        for p in res["preds"]:
            by.setdefault(p["image_id"], []).append(p)
        assert set(by) == {f(15), f(20), f(25)}
        faller = {k: [p for p in by[f(k)] if p["xc"] < 0.6] for k in (15, 20, 25)}
        (p15,), (p20,) = faller[15], faller[20]
        assert p15["cls"] == 0 and p20["cls"] == 1               # the nearer keyframe's class (t = 0.25, 0.5)
        assert p15["xc"] == pytest.approx(qa.lerp_box(stand_l, fallen_l, 0.25)[1], abs=1e-4)
        assert "class changes standing → fallen" in p15["reason"]
        assert [p["reason"] for p in by[f(15)] if p["xc"] > 0.6] == ["Interpolated between fall_0010.jpg and fall_0030.jpg"]
        assert faller[25] == [] and res["skipped_existing"] == 1  # its fallen box covers the faller already
        assert [p["cls"] for p in by[f(25)]] == [0]               # the other person still gets one
        run = db.query_one("SELECT params FROM ai_runs WHERE id=?", (res["run_id"],))
        assert db.loads(run["params"])["class_changes"] == 1
        # the faller alone in both keyframes: no longer a 400
        for k, boxes in ((10, [stand_l]), (30, [fallen_l])):
            (root / "labels" / "val" / f"fall_{k:04d}.txt").write_text(_rows(boxes))
        alone = qa.interpolate(ds_id, f(10), f(30))
        assert alone["matched"] == 1 and len(alone["class_changes"]) == 1 and {p["image_id"] for p in alone["preds"]} == {f(15), f(20)}
        # a different class across the frame is another object, not a class change
        (root / "labels" / "val" / "fall_0030.txt").write_text(_rows([_px(1, 0, 560, 150, 620)]))
        (root / "labels" / "val" / "fall_0010.txt").write_text(_rows([_px(0, 540, 250, 590, 400)]))
        with pytest.raises(qa.QAError, match="another class only where they overlap"):
            qa.interpolate(ds_id, f(10), f(30))
    finally:
        _drop(ds_id)


def test_lock_probe_passes_where_file_locks_work(tmp_path):
    assert qa.probe_cross_process_locks(tmp_path) == (True, "")
    assert list(tmp_path.iterdir()) == []                      # it cleans up after itself


_DOCKER_DESKTOP_CHILD = """import os, sqlite3, sys
c = sqlite3.connect(sys.argv[1]); c.execute("SELECT COUNT(*) FROM t").fetchone(); c.close()
for s in ("-wal", "-shm"):
    os.unlink(sys.argv[1] + s)           # what its close does on a Docker Desktop bind mount (no shared locks)
"""


def test_a_folder_without_shared_file_locks_never_gets_a_worker(ds, monkeypatch, tmp_path):
    """On a Docker Desktop bind mount a second SQLite process deletes the server's -wal/-shm when it closes,
    and every new request then fails with 'disk I/O error'. The probe sees that, and the checks run in the
    server process instead (same results)."""
    ok, why = qa.probe_cross_process_locks(tmp_path, _DOCKER_DESKTOP_CHILD)
    assert not ok and "do not reach across processes" in why and list(tmp_path.iterdir()) == []
    ok, why = qa.probe_cross_process_locks(tmp_path, "raise SystemExit(3)")
    assert not ok and "exit code 3" in why
    want = qa.run_qa(ds["id"], Job(id="qa-locks-0", kind="qa"), ["labels"])
    want_flags = _every_flag(ds["id"])
    monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 0)
    monkeypatch.setattr(qa, "RECHECK_PROCESS_FRAMES", 0)
    monkeypatch.setattr(qa, "_worker_broken", [])
    monkeypatch.setattr(qa, "_lock_probe", {})
    monkeypatch.setattr(qa, "_LOCK_PROBE_CHILD", _DOCKER_DESKTOP_CHILD)
    monkeypatch.setattr(qa, "_run_in_worker", _in_server_only)
    res = qa.run_qa(ds["id"], Job(id="qa-locks", kind="qa"), ["labels"])
    assert res["process"] is False and res["images"] == want["images"] and _every_flag(ds["id"]) == want_flags
    assert len(qa._worker_broken) == 1 and "do not reach across processes" in qa._worker_broken[0]
    folder = str(Path(db.DB_PATH).parent)
    assert qa._lock_probe[(folder, sys.executable)][0] is False
    assert not [p for p in Path(folder).iterdir() if p.name.startswith(".fovea-lock-probe")]
    assert Path(f"{db.DB_PATH}-wal").exists() and Path(f"{db.DB_PATH}-shm").exists()


def test_a_worker_never_closes_the_index_behind_the_server(tmp_path, monkeypatch):
    """A worker leaves without sqlite3_close: a close that believes it is the last connection checkpoints
    and deletes the WAL. Here nothing else has the database open, so a close WOULD delete it."""
    import sqlite3
    path = tmp_path / "index.db"
    monkeypatch.setattr(db, "DB_PATH", path)
    monkeypatch.setattr(db, "_initialised", False)
    err = []

    def init():                                                # a thread of its own: its own connection
        try:
            db.init_db()
            db.connect().close()
            db._local.conn = None
        except Exception as e:  # noqa: BLE001
            err.append(e)
    th = threading.Thread(target=init)
    th.start()
    th.join(30)
    assert not err and not Path(f"{path}-wal").exists()        # its close was the last: the WAL went with it
    seen = []
    qa._run_in_worker("run", {"dataset_id": "no-such-dataset", "checks": ["sequence"], "opt": dict(qa.DEFAULTS),
                              "rehash": False}, seen.append, lambda: None)
    assert [m[0] for m in seen][-1] == "done" and not qa._live_workers
    assert Path(f"{path}-wal").exists() and Path(f"{path}-shm").exists()
    with sqlite3.connect(str(path)) as c:                      # and the index is intact
        assert c.execute("PRAGMA integrity_check").fetchone()[0] == "ok"


# --------------------------------------------------------------------------- fill-gap matches one-to-one, like seq_gap
# vfp290k train/images_GOPR0807: frame 00266 has 2 class-8 boxes where 00264, 00265 and 00268 have 3. The missing
# one sits 18 px above the labeled one, which overlaps its expected place at IoU 0.31: a many-to-one "any box of the
# class nearby" test took that box for the missing one and answered "Nothing to fill" while seq_gap reported the gap.
GOPR0807 = {
    264: [[0, 0.281250, 0.506481, 0.041667, 0.170370], [0, 0.359375, 0.384259, 0.017708, 0.085185],
          [0, 0.565365, 0.423611, 0.023438, 0.125000], [0, 0.792969, 0.455556, 0.036979, 0.125926],
          [8, 0.678550, 0.433316, 0.085673, 0.075744], [8, 0.750784, 0.454078, 0.056467, 0.055117],
          [8, 0.748695, 0.482382, 0.075989, 0.044667]],
    265: [[0, 0.278646, 0.505556, 0.038542, 0.170370], [0, 0.359896, 0.383333, 0.017708, 0.085185],
          [0, 0.564323, 0.423148, 0.024479, 0.124074], [0, 0.798958, 0.467130, 0.033333, 0.149074],
          [8, 0.679313, 0.433638, 0.085737, 0.075824], [8, 0.751095, 0.488580, 0.077583, 0.056175],
          [8, 0.752461, 0.452885, 0.059316, 0.053391]],
    266: [[0, 0.275000, 0.504630, 0.035417, 0.168519], [0, 0.358594, 0.382407, 0.017188, 0.085185],
          [0, 0.562760, 0.422685, 0.027604, 0.123148], [0, 0.799479, 0.468056, 0.034375, 0.158333],
          [8, 0.678532, 0.433057, 0.090306, 0.075894], [8, 0.750141, 0.486804, 0.079397, 0.055739]],
    268: [[0, 0.272135, 0.509259, 0.036979, 0.172222], [0, 0.358594, 0.382407, 0.017188, 0.085185],
          [0, 0.562240, 0.421296, 0.029687, 0.122222], [0, 0.808854, 0.457870, 0.034375, 0.152778],
          [8, 0.678464, 0.431747, 0.087593, 0.075415], [8, 0.755867, 0.468335, 0.086886, 0.088414],
          [8, 0.752293, 0.486048, 0.079141, 0.054619]],
    270: [[0, 0.269792, 0.513889, 0.041667, 0.172222], [0, 0.358594, 0.383333, 0.017188, 0.085185],
          [0, 0.562760, 0.422685, 0.028646, 0.121296], [0, 0.815104, 0.450000, 0.034375, 0.144444],
          [8, 0.679873, 0.431614, 0.088292, 0.075679], [8, 0.760818, 0.467828, 0.090260, 0.086579],
          [8, 0.752038, 0.484051, 0.078585, 0.053755]],
}
GOPR0807_MISSING = [8, 0.7536, 0.458, 0.0685, 0.065]


def _close(box, ref, tol=6e-4):
    return box[0] == ref[0] and all(abs(a - b) <= tol for a, b in zip(box[1:], ref[1:]))


def test_missing_box_next_to_a_labeled_one_is_found_by_the_shared_rule():
    P, C, N = GOPR0807[265], GOPR0807[266], GOPR0807[268]
    from fovea.core.labels import iou
    labeled = C[5]
    assert 0.25 < iou(labeled, GOPR0807_MISSING) < 0.4             # the box the old test mistook for the missing one
    (gap,) = qa.missing_boxes(P, C, N, 640, 640, 1 / 3)
    assert _close(gap, GOPR0807_MISSING)
    # every complete frame of the stretch has nothing missing
    for a, b, c, w in ((264, 265, 266, 0.5), (266, 268, 270, 0.5)):
        assert qa.missing_boxes(GOPR0807[a], GOPR0807[b], GOPR0807[c], 640, 640, w) == []
    # once the box is there, nothing is missing any more
    assert qa.missing_boxes(P, C + [[8, 0.7536, 0.458, 0.0685, 0.065]], N, 640, 640, 1 / 3) == []


def test_one_orphan_box_explains_one_missing_object():
    """Two objects of a class both missing their box; one unlinked box lies near both expected places: it
    stands for one of them, the other is still missing."""
    a, b = [0, 0.30, 0.5, 0.06, 0.2], [0, 0.37, 0.5, 0.06, 0.2]
    far = [1, 0.9, 0.1, 0.05, 0.05]
    between = [0, 0.335, 0.5, 0.06, 0.2]                             # overlaps both expected places
    out = qa.missing_boxes([a, b, far], [far, between], [a, b, far], 640, 640, 0.5)
    assert len(out) == 1 and out[0][0] == 0
    assert qa._explained([a, b], [between], 640, 640) in ({0}, {1})
    assert qa._explained([a, b], [between, list(a)], 640, 640) == {0, 1}


def test_already_labeled_matches_one_to_one_at_iou_half():
    p1, p2 = [0, 0.50, 0.5, 0.1, 0.3], [0, 0.52, 0.5, 0.1, 0.3]     # one person just behind another
    one = [0, 0.51, 0.5, 0.1, 0.3]                                    # IoU 0.82 with each of them
    assert qa.already_labeled([p1, p2], [one]) in ({0}, {1})          # one box is one object
    assert qa.already_labeled([p1, p2], [one, [0, 0.52, 0.5, 0.1, 0.3]]) == {0, 1}
    near = [0, 0.535, 0.5, 0.1, 0.3]                                  # IoU 0.48 / 0.74 with p1 / p2
    assert qa.already_labeled([p1], [near]) == set()
    shifted = [0, 0.53, 0.5, 0.1, 0.3]                                # IoU 0.54: the same object, drifted
    assert qa.already_labeled([p1], [shifted]) == {0}
    assert qa.already_labeled([p1], [[1] + shifted[1:]]) == set()     # another class is not it ...
    assert qa.already_labeled([p1], [[1] + shifted[1:]], [{0, 1}]) == {0}   # ... unless the class changes here


def _scan_alone(root):
    from fovea.core import scanner
    ds_id = _register(root)
    saved = list(scanner.POST_SCAN_HOOKS)
    scanner.POST_SCAN_HOOKS[:] = []
    try:
        scanner.scan_dataset(ds_id, Job(id="scan-" + ds_id, kind="scan"))
    finally:
        scanner.POST_SCAN_HOOKS[:] = saved
    ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
    return ds_id, ids


def _frame_640(bg, boxes):
    im = bg.copy()
    d = ImageDraw.Draw(im)
    for c, xc, yc, w, h in boxes:
        d.rectangle([(xc - w / 2) * 640, (yc - h / 2) * 640, (xc + w / 2) * 640, (yc + h / 2) * 640],
                    outline=(240, 240, 240) if c else (20, 20, 20), width=2)
    return im


def test_fill_gap_fills_what_seq_gap_reports_next_to_a_labeled_box(tmp_path):
    root = tmp_path / "gopr"
    bg = _scene(21).resize((640, 640))
    for f, boxes in GOPR0807.items():
        p = root / "images" / "train" / f"images_GOPR0807_{f:05d}.jpg"
        p.parent.mkdir(parents=True, exist_ok=True)
        _frame_640(bg, boxes).save(p, "JPEG", quality=92)
        lp = root / "labels" / "train" / f"images_GOPR0807_{f:05d}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(_rows(boxes))
    ds_id, ids = _scan_alone(root)
    try:
        f = lambda k: ids[f"train/images_GOPR0807_{k:05d}.jpg"]  # noqa: E731
        qa.run_qa(ds_id, Job(id="qa-" + ds_id, kind="qa"), checks=["sequence"])
        gap = _flags(ds_id, "seq_gap")
        assert set(gap) == {f(266)}
        (flagged,) = gap[f(266)]["boxes"]
        assert _close(flagged, GOPR0807_MISSING) and (gap[f(266)]["prev_id"], gap[f(266)]["next_id"]) == (f(265), f(268))
        res = qa.fill_gap(ds_id, f(266))
        assert (res["from_image_id"], res["to_image_id"]) == (f(265), f(268)) and "message" not in res
        (p,) = res["preds"]
        assert p["image_id"] == f(266) and p["kind"] == "interp" and p["state"] == "pending"
        assert [p["cls"], p["xc"], p["yc"], p["w"], p["h"]] == flagged      # exactly the flagged box
        assert p["reason"] == "Filled from images_GOPR0807_00265.jpg and images_GOPR0807_00268.jpg"
        # the complete frames around it have nothing to fill
        for k in (265, 268):
            ok = qa.fill_gap(ds_id, f(k))
            assert ok["preds"] == [] and ok["run_id"] is None and ok["message"].startswith("Nothing to fill")
        # over HTTP too
        from fastapi.testclient import TestClient
        from fovea.main import app
        r = TestClient(app).post(f"/api/datasets/{ds_id}/images/{f(266)}/fill-gap")
        assert r.status_code == 200
        (hp,) = r.json()["preds"]
        assert _close([hp["cls"], hp["xc"], hp["yc"], hp["w"], hp["h"]], GOPR0807_MISSING)
        # accepted (written to the label file): no longer missing, so nothing to fill
        lp = root / "labels" / "train" / "images_GOPR0807_00266.txt"
        lp.write_text(_rows(GOPR0807[266] + [flagged]))
        again = qa.fill_gap(ds_id, f(266))
        assert again["preds"] == [] and again["message"].startswith("Nothing to fill")
    finally:
        _drop(ds_id)


def test_interpolate_keeps_a_neighbour_that_shares_its_box(tmp_path):
    """Keyframes 10 and 20 have two people one just behind the other; frame 15 already has a box for one of
    them that overlaps both interpolated boxes at IoU >= 0.5: one is labeled, the other is proposed."""
    root = tmp_path / "pair"
    bg = _scene(8)
    p1, p2 = [0, 0.50, 0.5, 0.1, 0.3], [0, 0.52, 0.5, 0.1, 0.3]
    for k, boxes in ((10, [p1, p2]), (15, [[0, 0.51, 0.5, 0.1, 0.3]]), (20, [p1, p2])):
        _write(root, "val", f"pair_{k:04d}", bg.copy(), boxes)
    ds_id, ids = _scan_alone(root)
    try:
        f = lambda k: ids[f"val/pair_{k:04d}.jpg"]  # noqa: E731
        res = qa.interpolate(ds_id, f(10), f(20))
        assert res["matched"] == 2 and res["skipped_existing"] == 1
        (p,) = res["preds"]
        assert p["image_id"] == f(15) and p["cls"] == 0
        # a drifted box (IoU 0.54 with the interpolated one) is that object: nothing proposed for it
        (root / "labels" / "val" / "pair_0010.txt").write_text(_rows([p1]))
        (root / "labels" / "val" / "pair_0020.txt").write_text(_rows([p1]))
        (root / "labels" / "val" / "pair_0015.txt").write_text(_rows([[0, 0.53, 0.5, 0.1, 0.3]]))
        db.execute("DELETE FROM boxes WHERE image_id=?", (f(15),))
        db.execute("INSERT INTO boxes(dataset_id, image_id, cls, xc, yc, w, h) VALUES(?,?,0,0.53,0.5,0.1,0.3)", (ds_id, f(15)))
        drift = qa.interpolate(ds_id, f(10), f(20))
        assert drift["preds"] == [] and drift["skipped_existing"] == 1
    finally:
        _drop(ds_id)
