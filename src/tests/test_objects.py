"""Objects (patch view): box listing, crops, batch reassign / delete. Run: python -m pytest tests -q

Floats are compared by hand, not with pytest.approx: approx probes numpy, and a background QA job may be
importing numpy in another thread at that moment."""
from __future__ import annotations

import io
import json
import os
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout  # noqa: E402

RED, BLUE, GREEN = (230, 20, 20), (20, 40, 230), (20, 200, 60)


def _scene(path: Path, size, rects):
    """An image with filled rectangles at normalised boxes [(xc, yc, w, h, colour)]."""
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    im = Image.new("RGB", size, (128, 128, 128))
    d = ImageDraw.Draw(im)
    for xc, yc, w, h, col in rects:
        d.rectangle([round((xc - w / 2) * W), round((yc - h / 2) * H), round((xc + w / 2) * W) - 1, round((yc + h / 2) * H) - 1], fill=col)
    im.save(path, quality=95)


def _dataset(tmp_path, files):
    """files: {rel_stem: (size, rects, label_text, split)} -> registered + scanned dataset id."""
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / "ds"
    for stem, (size, rects, text, split) in files.items():
        _scene(root / "images" / split / f"{stem}.jpg", size, rects)
        lp = root / "labels" / split / f"{stem}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_bytes(text.encode() if isinstance(text, str) else text)
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "obj-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["person", "fall", "bike"]), "inferred", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id, root


def _drop(ds_id):
    db.execute("DELETE FROM snapshots WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM boxes WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM images WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _client():
    from fastapi.testclient import TestClient
    from fovea.main import app
    return TestClient(app)


def _three(tmp_path):
    return _dataset(tmp_path, {
        "a": ((200, 100), [(0.25, 0.5, 0.2, 0.4, RED), (0.75, 0.5, 0.3, 0.6, BLUE)],
              "0 0.25 0.5 0.2 0.4\n1 0.75 0.5 0.3 0.6\n", "train"),
        "b": ((100, 100), [(0.5, 0.5, 0.1, 0.1, GREEN)], "2 0.5 0.5 0.1 0.1\n", "val"),
        "c": ((100, 100), [], "0 0.1 0.1 0.05 0.05\n0 0.9 0.9 0.5 0.5\n1 0.5 0.5 0.4 0.2\n", "train"),
    })


# ------------------------------------------------------------------ listing


def test_box_list_rows_filters_and_counts(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        r = c.get(f"/api/datasets/{ds_id}/boxes", params={"counts": 1}).json()
        assert r["total"] == 6 and r["class_counts"] == {"0": 3, "1": 2, "2": 1}
        by_path = {}
        for it in r["items"]:
            by_path.setdefault(it["rel_path"], []).append(it)
            assert it["version"] and len(it["k"]) == 3
        assert sorted(it["row"] for it in by_path["train/c.jpg"]) == [0, 1, 2]
        a = sorted(by_path["train/a.jpg"], key=lambda x: x["row"])
        assert [x["cls"] for x in a] == [0, 1] and a[0]["width"] == 200 and a[0]["height"] == 100
        # cls filters BOXES (an image holding class 0 does not bring its class-1 box along)
        r1 = c.get(f"/api/datasets/{ds_id}/boxes", params={"cls": "1"}).json()
        assert r1["total"] == 2 and {it["cls"] for it in r1["items"]} == {1}
        # image filters narrow by the box's image; class counts ignore the class filter but keep the rest
        rv = c.get(f"/api/datasets/{ds_id}/boxes", params={"split": "train", "cls": "0", "counts": 1}).json()
        assert rv["total"] == 3 and rv["class_counts"] == {"0": 3, "1": 2}
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"q": "b.jpg"}).json()["total"] == 1
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"after": "[1]"}).status_code == 400
        assert c.get("/api/datasets/nope/boxes").status_code == 404
    finally:
        _drop(ds_id)


def test_every_sort_orders_correctly_and_keyset_matches_offset_paging(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        area = [it["w"] * it["h"] for it in c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "area"}).json()["items"]]
        assert area == sorted(area)
        asp = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "aspect", "order": "desc"}).json()["items"]
        px = [it["w"] * it["width"] / (it["h"] * it["height"]) for it in asp]
        assert px == sorted(px, reverse=True)
        # aspect is in pixels: 0.2x0.4 on a 200x100 frame is a 40x40 square, not 0.5
        sq = next(it for it in asp if it["rel_path"] == "train/a.jpg" and it["row"] == 0)
        assert abs(sq["k"][0] - 1.0) < 1e-9
        cl = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "class"}).json()["items"]
        assert [it["cls"] for it in cl] == sorted(it["cls"] for it in cl)
        for sort in ("class", "area", "aspect", "random", "audit", "name"):
            full = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": sort, "seed": 3}).json()["items"]
            ident = [(it["image_id"], it["row"]) for it in full]
            assert len(set(ident)) == 6
            paged = [(it["image_id"], it["row"]) for off in range(0, 6, 2) for it in
                     c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": sort, "seed": 3, "offset": off, "limit": 2}).json()["items"]]
            assert paged == ident, sort
            ks = [(it["image_id"], it["row"]) for it in
                  c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": sort, "seed": 3, "after": json.dumps(full[1]["k"]), "offset": 1}).json()["items"]]
            assert ks == ident[3:], sort
        shuffled = [(it["image_id"], it["row"]) for it in c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "random", "seed": 11}).json()["items"]]
        assert sorted(shuffled) == sorted(ident)
    finally:
        _drop(ds_id)


def test_keyset_continuation_survives_deletes_before_the_cursor(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        full = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "area"}).json()["items"]
        first = full[:3]
        # delete the first two (already "loaded") boxes; offset paging would now skip two unseen boxes
        edits = [{"image_id": it["image_id"], "row": it["row"], "version": it["version"], "delete": True} for it in first[:2]]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": edits}).json()
        assert res["deleted"] == 2 and not res["conflicts"]
        rest = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "area", "after": json.dumps(first[2]["k"])}).json()
        geo = lambda it: (it["image_id"], it["xc"], it["yc"], it["w"], it["h"])   # noqa: E731
        assert [geo(x) for x in rest["items"]] == [geo(x) for x in full[3:]]
        assert rest["total"] == 4
    finally:
        _drop(ds_id)


def test_audit_sort_puts_flagged_boxes_first_and_lists_suspects(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        b_img = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/c.jpg'", (ds_id,))["id"]
        run = db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'audit', ?)", (ds_id, db.now())).lastrowid
        db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, ref_row, reason, state)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?)", (run, ds_id, b_img, 1, 0.9, 0.9, 0.5, 0.5, 0.8, "swap", 1, "looks fallen", "pending"))
        r = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()
        assert r["audit_available"] is True and r["order"] == "desc"
        top = r["items"][0]
        assert (top["image_id"], top["row"]) == (b_img, 1)
        assert top["suspects"][0]["kind"] == "swap" and top["suspects"][0]["cls"] == 1
        assert all("suspects" not in it for it in r["items"][1:])
        # deleting the box above it shifts rows; the finding still names the right box (matched by geometry)
        row0 = next(it for it in r["items"] if it["image_id"] == b_img and it["row"] == 0)
        c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": b_img, "row": 0, "version": row0["version"], "delete": True}]})
        top = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()["items"][0]
        assert (top["image_id"], top["row"], top["xc"]) == (b_img, 0, 0.9) and top["suspects"][0]["kind"] == "swap"
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        _drop(ds_id)


# ------------------------------------------------------------------ crops


def test_crop_region_is_a_centred_square_with_context():
    from fovea.api.objects import crop_region
    x0, y0, side = crop_region(200, 100, (0.25, 0.5, 0.2, 0.4), 0.25)   # 40x40 px box at (50, 50)
    assert abs(side - 60) < 1e-9 and abs(x0 - 20) < 1e-9 and abs(y0 - 20) < 1e-9
    _x, _y, tiny = crop_region(1000, 1000, (0.5, 0.5, 0.001, 0.001), 0.2)
    assert tiny == 24          # never fewer than MIN_CONTEXT_PX source pixels


def _px(im, x, y):
    return im.getpixel((int(x), int(y)))


def _close(a, b, tol=40):
    return all(abs(p - q) <= tol for p, q in zip(a, b))


def test_crops_show_the_labelled_object_where_the_outline_is_drawn(tmp_path):
    from fovea.api.objects import crop_region, render_crop
    p = tmp_path / "scene.jpg"
    # a 40x40 red square at (50,50) and a blue 60x60 square touching the right edge of a 200x100 frame
    _scene(p, (200, 100), [(0.25, 0.5, 0.2, 0.4, RED), (0.85, 0.5, 0.3, 0.6, BLUE)])
    for box, col in (((0.25, 0.5, 0.2, 0.4), RED), ((0.85, 0.5, 0.3, 0.6), BLUE)):
        im = render_crop(str(p), os.stat(p).st_mtime, box, 0.25, 192)
        assert im.size == (192, 192)
        x0, y0, side = crop_region(200, 100, box, 0.25)
        bw, bh = box[2] * 200, box[3] * 100
        l, t, r_, b_ = (0.5 - bw / side / 2) * 192, (0.5 - bh / side / 2) * 192, (0.5 + bw / side / 2) * 192, (0.5 + bh / side / 2) * 192
        assert _close(_px(im, 96, 96), col)
        assert _close(_px(im, l + 4, t + 4), col) and _close(_px(im, r_ - 4, b_ - 4), col)   # inside the outline
        assert not _close(_px(im, l - 6, 96), col) and not _close(_px(im, 96, t - 6), col)  # just outside
    # beyond the right edge the crop is the neutral fill, not stretched or shifted pixels
    edge = render_crop(str(p), os.stat(p).st_mtime, (0.85, 0.5, 0.3, 0.6), 0.25, 192)
    assert _close(_px(edge, 190, 96), (28, 29, 32), tol=12)


def test_crops_follow_exif_orientation_and_downscaled_jpeg_decode(tmp_path):
    from fovea.api.objects import render_crop
    # Stored landscape 400x200 with orientation 6 (rotate 90 CW to display): displayed 200x400.
    raw = Image.new("RGB", (400, 200), (128, 128, 128))
    ImageDraw.Draw(raw).rectangle([20, 20, 59, 59], fill=RED)        # stored top-left corner
    exif = Image.Exif()
    exif[0x0112] = 6
    p = tmp_path / "rot.jpg"
    raw.save(p, exif=exif, quality=95)
    # after a 90 CW rotation the stored top-left lands top-right: display x in [140,180), y in [20,60)
    im = render_crop(str(p), os.stat(p).st_mtime, (0.8, 0.1, 0.2, 0.1), 0.2, 128)
    assert _close(_px(im, 64, 64), RED)
    # a big frame with a small output decodes at a JPEG draft scale but crops the same spot
    big = tmp_path / "big.jpg"
    _scene(big, (3200, 2400), [(0.5, 0.5, 0.5, 0.5, GREEN)])
    im2 = render_crop(str(big), os.stat(big).st_mtime, (0.5, 0.5, 0.5, 0.5), 0.2, 96)
    assert _close(_px(im2, 48, 48), GREEN) and not _close(_px(im2, 3, 48), GREEN)


def test_crops_of_high_bit_depth_and_transparent_images_are_visible(tmp_path):
    from fovea.api.objects import render_crop
    # 16-bit grayscale (thermal / depth / medical): values far above 255 must be stretched, not clipped to white
    g = Image.new("I;16", (100, 100), 1000)
    ImageDraw.Draw(g).rectangle([40, 40, 59, 59], fill=3000)
    p16 = tmp_path / "g16.png"
    g.save(p16)
    im = render_crop(str(p16), os.stat(p16).st_mtime, (0.5, 0.5, 0.2, 0.2), 0.5, 64)
    centre, edge = _px(im, 32, 32), _px(im, 4, 4)
    assert centre[0] > 200 and edge[0] < 60
    rgba = Image.new("RGBA", (80, 80), (0, 0, 0, 0))
    ImageDraw.Draw(rgba).rectangle([30, 30, 49, 49], fill=(20, 200, 60, 255))
    pa = tmp_path / "a.png"
    rgba.save(pa)
    assert _close(_px(render_crop(str(pa), os.stat(pa).st_mtime, (0.5, 0.5, 0.25, 0.25), 0.2, 64), 32, 32), GREEN)


def test_crop_endpoint_caches_and_validates(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        iid = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/a.jpg'", (ds_id,))["id"]
        url = f"/api/crop/{iid}?box=0.25,0.5,0.2,0.4&pad=0.2&s=100&v=1"
        r = c.get(url)
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert "immutable" in r.headers["cache-control"]
        assert Image.open(io.BytesIO(r.content)).size == (128, 128)          # sizes snap to a small set
        assert c.get(url, headers={"if-none-match": r.headers["etag"]}).status_code == 304
        assert "no-cache" in c.get(url.replace("&v=1", "")).headers["cache-control"]
        assert c.get(f"/api/crop/{iid}?box=nan,0.5,0.1,0.1").status_code == 400
        assert c.get(f"/api/crop/{iid}?box=0.5,0.5").status_code == 400
        assert c.get("/api/crop/987654321?box=0.5,0.5,0.1,0.1").status_code == 404
        assert c.get("/api/crops/cache").json()["files"] >= 1
    finally:
        _drop(ds_id)


# ------------------------------------------------------------------ batch edits


def test_edits_touch_only_their_lines_and_keep_everything_else_byte_exact(tmp_path):
    text = ("0 0.250000 0.5 0.2 0.4 0.91\r\n"                       # prediction row with confidence, CRLF
            "not a box\r\n"                                         # unparsable line: kept, not a row
            "1 0.1 0.1 0.3 0.1 0.4 0.3 0.1 0.3\r\n"                 # segmentation polygon
            "2 0.75 0.5 0.3 0.6 0.55\r\n")
    ds_id, root = _dataset(tmp_path, {"a": ((200, 100), [], text, "train")})
    lp = root / "labels" / "train" / "a.txt"
    try:
        c = _client()
        items = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "name"}).json()["items"]
        assert [it["row"] for it in items] == [0, 1, 2] and abs(items[0]["conf"] - 0.91) < 1e-9
        v = items[0]["version"]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [
            {"image_id": items[1]["image_id"], "row": 1, "version": v, "cls": 2},       # polygon keeps its tail
            {"image_id": items[2]["image_id"], "row": 2, "version": v, "delete": True},
        ]}).json()
        assert res["reassigned"] == 1 and res["deleted"] == 1 and not res["conflicts"]
        assert lp.read_bytes() == b"0 0.250000 0.5 0.2 0.4 0.91\r\nnot a box\r\n2 0.1 0.1 0.3 0.1 0.4 0.3 0.1 0.3\r\n"
        assert res["versions"][str(items[0]["image_id"])] != v
        after = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "name"}).json()["items"]
        assert [(it["row"], it["cls"]) for it in after] == [(0, 0), (1, 2)]
        assert db.query_one("SELECT box_count FROM datasets WHERE id=?", (ds_id,))["box_count"] == 2
        if res["snapshot_id"] is not None:
            from fovea.core import snapshots
            assert snapshots.get_snapshot(ds_id, res["snapshot_id"])["n_files"] == 1
    finally:
        _drop(ds_id)


def test_stale_versions_are_refused_unless_the_box_can_be_found_again(tmp_path):
    ds_id, root = _three(tmp_path)
    lp = root / "labels" / "train" / "c.txt"
    try:
        c = _client()
        items = [it for it in c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "name"}).json()["items"]
                 if it["rel_path"] == "train/c.jpg"]
        target = items[2]                                            # row 2: "1 0.5 0.5 0.4 0.2"
        lp.write_text("0 0.9 0.9 0.5 0.5\n1 0.5 0.5 0.4 0.2\n")       # someone deleted row 0 elsewhere
        bare = {"image_id": target["image_id"], "row": 2, "version": target["version"], "cls": 2}
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [bare]}).json()
        assert res["updated"] == 0 and res["conflicts"][0]["image_id"] == target["image_id"]
        assert lp.read_text() == "0 0.9 0.9 0.5 0.5\n1 0.5 0.5 0.4 0.2\n", "a refused edit must not touch the file"
        # with what it expects to find, the edit is re-located to the box's new row
        located = dict(bare, cls_was=1, box=[target["xc"], target["yc"], target["w"], target["h"]])
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [located]}).json()
        assert res["reassigned"] == 1 and not res["conflicts"]
        assert lp.read_text() == "0 0.9 0.9 0.5 0.5\n2 0.5 0.5 0.4 0.2\n"
        # a box that is gone is a conflict, not a guess
        gone = dict(located, cls_was=0, box=[0.1, 0.1, 0.05, 0.05], delete=True)
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [gone]}).json()
        assert res["conflicts"] and lp.read_text() == "0 0.9 0.9 0.5 0.5\n2 0.5 0.5 0.4 0.2\n"
    finally:
        _drop(ds_id)


def test_edit_validation(tmp_path):
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        it = c.get(f"/api/datasets/{ds_id}/boxes").json()["items"][0]
        base = {"image_id": it["image_id"], "row": it["row"], "version": it["version"]}
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": []}).status_code == 400
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [dict(base, cls=-1)]}).status_code == 400
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [dict(base, cls=1.5)]}).status_code == 400
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [dict(base)]}).status_code == 400
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": it["image_id"], "row": 0, "cls": 1}]}).status_code == 400
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [dict(base, image_id=987654321, cls=1)]}).json()
        assert res["conflicts"][0]["reason"].startswith("image not found")
        # a declared class list (data.yaml) is a contract: no writing ids outside it
        db.execute("UPDATE datasets SET classes_source='yaml' WHERE id=?", (ds_id,))
        assert c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [dict(base, cls=7)]}).status_code == 400
    finally:
        _drop(ds_id)


def test_undo_through_snapshot_restore(tmp_path):
    pytest.importorskip("fovea.core.snapshots")
    from fovea.core import snapshots
    ds_id, root = _three(tmp_path)
    lp = root / "labels" / "train" / "a.txt"
    before = lp.read_bytes()
    try:
        c = _client()
        items = [it for it in c.get(f"/api/datasets/{ds_id}/boxes").json()["items"] if it["rel_path"] == "train/a.jpg"]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [
            {"image_id": it["image_id"], "row": it["row"], "version": it["version"], "delete": True} for it in items]}).json()
        assert lp.read_bytes() == b"" and res["snapshot_id"]
        r = c.post(f"/api/datasets/{ds_id}/snapshots/{res['snapshot_id']}/restore", json={"image_ids": res["images_changed"]})
        assert r.status_code == 200
        job = r.json().get("job")
        if job:
            import time
            from fovea.jobs import jobs
            for _ in range(200):
                if jobs.get(job["id"]).status in ("done", "error", "cancelled"):
                    break
                time.sleep(0.02)
        assert lp.read_bytes() == before
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"q": "a.jpg"}).json()["total"] == 2
        assert snapshots.get_snapshot(ds_id, res["snapshot_id"]) is not None
    finally:
        _drop(ds_id)


def test_large_batches_run_as_a_job(tmp_path, monkeypatch):
    import time
    from fovea.api import objects
    from fovea.jobs import jobs
    ds_id, root = _three(tmp_path)
    monkeypatch.setattr(objects, "ASYNC_IMAGES", 1)
    try:
        c = _client()
        items = c.get(f"/api/datasets/{ds_id}/boxes", params={"cls": "0"}).json()["items"]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [
            {"image_id": it["image_id"], "row": it["row"], "version": it["version"], "cls": 2} for it in items]}).json()
        job = jobs.get(res["job"]["id"])
        for _ in range(300):
            if job.status in ("done", "error", "cancelled"):
                break
            time.sleep(0.02)
        assert job.status == "done" and job.result["reassigned"] == 3
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"cls": "0"}).json()["total"] == 0
    finally:
        _drop(ds_id)


# ------------------------------------------------------------------ wave 2a: label-write hooks, working sets, ref_box


def _ensure_ref_box_column():
    """ai_preds.ref_box arrives with a later migration; add it here when this database predates it."""
    if not any(r["name"] == "ref_box" for r in db.query("PRAGMA table_info(ai_preds)")):
        db.execute("ALTER TABLE ai_preds ADD COLUMN ref_box TEXT")


def _finding(ds_id, image_id, kind, cls, box, ref_row, conf, ref_box=None, run=None):
    run = run or db.execute("INSERT INTO ai_runs(dataset_id, kind, created_at) VALUES(?, 'audit', ?)", (ds_id, db.now())).lastrowid
    cols = "run_id, dataset_id, image_id, cls, xc, yc, w, h, conf, kind, ref_row, reason, state"
    vals = [run, ds_id, image_id, cls, *box, conf, kind, ref_row, f"{kind} finding", "pending"]
    if ref_box is not None:
        cols += ", ref_box"
        vals.append(ref_box)
    return db.execute(f"INSERT INTO ai_preds({cols}) VALUES({','.join('?' * len(vals))})", vals).lastrowid, run


def test_ref_box_parsing_matches_the_sql_reading():
    from fovea.api.objects import _ref_coord, ref_geom
    samples = ["[0.5, 0.25, 0.4, 0.2]", "[1, 0.5, 0.25, 0.4, 0.2]", '{"xc": 0.5, "yc": 0.25, "w": 0.4, "h": 0.2}',
               "0.5,0.25,0.4,0.2", "[0.5, 0.25]", '"text"', "not json", '{"xc": 0.5}', None]
    want = [(0.5, 0.25, 0.4, 0.2)] * 3 + [None] * 6
    assert [ref_geom(s) for s in samples] == want
    assert ref_geom('[NaN, 0, 0, 0]') is None and ref_geom("[true, 0.5, 0.5, 0.1]") is None
    sql = "SELECT " + ", ".join(_ref_coord(k, n) for k, n in enumerate(("xc", "yc", "w", "h"))) + " FROM (SELECT ? AS ref_box)"
    for s, w in zip(samples, want):
        got = tuple(db.query_one(sql, [s]).values())
        got = None if any(v is None for v in got) else tuple(float(v) for v in got)
        assert got == w, s


def test_loose_findings_follow_their_ref_box_when_rows_shift(tmp_path):
    _ensure_ref_box_column()
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        img = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/c.jpg'", (ds_id,))["id"]
        # GT row 2 is "1 0.5 0.5 0.4 0.2"; the finding suggests a tighter box and (deliberately) names a wrong row:
        # the stored GT geometry wins over the row.
        pid, run = _finding(ds_id, img, "loose", 1, (0.5, 0.5, 0.3, 0.15), 0, 0.7, ref_box=json.dumps([0.5, 0.5, 0.4, 0.2]))
        r = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()
        top = r["items"][0]
        assert (top["image_id"], top["row"], top["xc"], top["w"]) == (img, 2, 0.5, 0.4)
        assert top["suspects"][0]["id"] == pid and top["suspects"][0]["kind"] == "loose"
        assert top["suspects"][0]["box"] == [0.5, 0.5, 0.3, 0.15]          # the suggested fit, for drawing
        assert all("suspects" not in it for it in r["items"][1:])
        # delete row 0: the box moves to row 1 and its finding moves with it (badge and sort agree)
        row0 = next(it for it in r["items"] if it["image_id"] == img and it["row"] == 0)
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": img, "row": 0, "version": row0["version"], "delete": True}]}).json()
        assert res["deleted"] == 1
        r = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()
        top = r["items"][0]
        assert (top["image_id"], top["row"], top["xc"]) == (img, 1, 0.5) and top["suspects"][0]["id"] == pid
        assert all("suspects" not in it for it in r["items"][1:])
        # the same box seen through a plain list carries the same badge
        named = [it for it in c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "name", "ids": str(img)}).json()["items"]]
        assert [bool(it.get("suspects")) for it in named] == [False, True]
        # once the GT box itself is gone, the finding names no box at all (it never falls back to its row)
        gone = next(it for it in named if it["row"] == 1)
        c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": img, "row": 1, "version": gone["version"], "delete": True}]})
        r = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()
        assert all("suspects" not in it for it in r["items"])
        assert r["audit_available"] is True
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        _drop(ds_id)


def test_loose_findings_without_ref_box_still_go_by_row(tmp_path, monkeypatch):
    from fovea.api import objects
    _ensure_ref_box_column()
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        img = db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/c.jpg'", (ds_id,))["id"]
        pid, run = _finding(ds_id, img, "loose", 0, (0.9, 0.9, 0.45, 0.45), 1, 0.6)                    # ref_box NULL
        bad, _ = _finding(ds_id, img, "loose", 1, (0.5, 0.5, 0.3, 0.15), 2, 0.4, ref_box="0.5,0.5,0.4,0.2", run=run)  # unreadable
        for column in (True, False):          # with the column (NULL / unreadable value), and on a database without it
            monkeypatch.setattr(objects, "has_ref_box", lambda column=column: column)
            items = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit"}).json()["items"]
            assert [(it["row"], it["suspects"][0]["id"]) for it in items[:2]] == [(1, pid), (2, bad)], column
            assert all(it["image_id"] == img for it in items[:2]) and all("suspects" not in it for it in items[2:])
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        _drop(ds_id)


def test_box_edits_notify_label_write_listeners(tmp_path, monkeypatch):
    import time
    from fovea.api import objects
    from fovea.core import labels as labels_mod
    from fovea.jobs import jobs
    ds_id, root = _three(tmp_path)
    calls = []
    monkeypatch.setattr(labels_mod, "notify_label_write", lambda ds, ids: calls.append((ds, sorted(ids))), raising=False)
    try:
        c = _client()
        items = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "name"}).json()["items"]
        by = {}
        for it in items:
            by.setdefault(it["rel_path"], []).append(it)
        a, cc = by["train/a.jpg"][0], by["train/c.jpg"][0]
        (root / "labels" / "train" / "c.txt").write_text("0 0.9 0.9 0.5 0.5\n")          # c changed elsewhere: conflict
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [
            {"image_id": a["image_id"], "row": 0, "version": a["version"], "cls": 2},
            {"image_id": cc["image_id"], "row": 0, "version": cc["version"], "cls": 2}]}).json()
        assert res["reassigned"] == 1 and len(res["conflicts"]) == 1
        assert calls == [(ds_id, [a["image_id"]])], "only images whose label file was written"
        # an edit that changes nothing writes nothing and notifies nobody
        calls.clear()
        same = c.get(f"/api/datasets/{ds_id}/boxes", params={"ids": str(a["image_id"]), "sort": "name"}).json()["items"][0]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": a["image_id"], "row": 0, "version": same["version"], "cls": 2}]}).json()
        assert res["updated"] == 0 and calls == []
        # a failing listener never fails the edit: the file is already written
        def boom(ds, ids):
            raise RuntimeError("listener failed")
        monkeypatch.setattr(labels_mod, "notify_label_write", boom, raising=False)
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": a["image_id"], "row": 0, "version": same["version"], "cls": 1}]})
        assert res.status_code == 200 and res.json()["reassigned"] == 1
        assert (root / "labels" / "train" / "a.txt").read_text().startswith("1 0.25")
        # the background-job path notifies too, once, with every written image
        monkeypatch.setattr(labels_mod, "notify_label_write", lambda ds, ids: calls.append((ds, sorted(ids))), raising=False)
        monkeypatch.setattr(objects, "ASYNC_IMAGES", 1)
        cur = c.get(f"/api/datasets/{ds_id}/boxes", params={"cls": "0,1"}).json()["items"]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [
            {"image_id": it["image_id"], "row": it["row"], "version": it["version"], "cls": 2} for it in cur]}).json()
        job = jobs.get(res["job"]["id"])
        for _ in range(300):
            if job.status in ("done", "error", "cancelled"):
                break
            time.sleep(0.02)
        assert job.status == "done"
        assert calls == [(ds_id, sorted({it["image_id"] for it in cur}))]
    finally:
        _drop(ds_id)


def test_label_write_hook_registry_sees_box_edits(tmp_path):
    """End to end through backend-data's registry (core.labels.LABEL_WRITE_HOOKS), when this tree has it."""
    import time
    from fovea.core import labels as labels_mod
    hooks = getattr(labels_mod, "LABEL_WRITE_HOOKS", None)
    if hooks is None or not hasattr(labels_mod, "notify_label_write"):
        pytest.skip("core.labels has no LABEL_WRITE_HOOKS registry in this tree")
    ds_id, _root = _three(tmp_path)
    seen = []
    hook = lambda ds, ids: seen.append((ds, sorted(ids)))   # noqa: E731
    hooks.append(hook)
    try:
        c = _client()
        it = c.get(f"/api/datasets/{ds_id}/boxes", params={"q": "b.jpg"}).json()["items"][0]
        res = c.patch(f"/api/datasets/{ds_id}/boxes", json={"edits": [{"image_id": it["image_id"], "row": 0, "version": it["version"], "delete": True}]}).json()
        assert res["deleted"] == 1
        for _ in range(100):            # a registry may run its hooks off the request thread
            if seen:
                break
            time.sleep(0.02)
        assert (ds_id, [it["image_id"]]) in seen
    finally:
        hooks.remove(hook)
        _drop(ds_id)


def test_working_set_filter_narrows_the_box_list(tmp_path):
    from fovea.api.common import FILTER_KEYS
    from fovea.core import imagesq
    if "set" not in FILTER_KEYS or not hasattr(imagesq, "create_set"):
        pytest.skip("working sets (set=) are not in this tree yet")
    ds_id, _root = _three(tmp_path)
    try:
        c = _client()
        ids = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        tok = imagesq.create_set(ds_id, ids=[ids["train/c.jpg"], ids["val/b.jpg"]])["token"]
        got = c.get(f"/api/datasets/{ds_id}/boxes", params={"set": tok, "counts": 1}).json()
        assert got["total"] == 4 and {it["rel_path"] for it in got["items"]} == {"train/c.jpg", "val/b.jpg"}
        assert got["class_counts"] == {"0": 2, "1": 1, "2": 1}
        # combines with the box-class filter and the other image filters, and with keyset paging
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"set": tok, "cls": "0"}).json()["total"] == 2
        assert c.get(f"/api/datasets/{ds_id}/boxes", params={"set": tok, "split": "val"}).json()["total"] == 1
        first = c.get(f"/api/datasets/{ds_id}/boxes", params={"set": tok, "sort": "area", "limit": 1}).json()["items"][0]
        rest = c.get(f"/api/datasets/{ds_id}/boxes", params={"set": tok, "sort": "area", "after": json.dumps(first["k"])}).json()
        assert len(rest["items"]) == 3 and all(it["rel_path"] in ("train/c.jpg", "val/b.jpg") for it in rest["items"])
        # an unknown or expired token is an explicit error, never the whole dataset
        bad = c.get(f"/api/datasets/{ds_id}/boxes", params={"set": "0123456789abcdef"})
        assert bad.status_code == 410 and bad.json()["detail"]["code"] == "set_expired"
        # Enter -> Annotate hands the token on; Annotate's image list honours it the same way
        imgs = c.get(f"/api/datasets/{ds_id}/images", params={"set": tok, "limit": 1}).json()
        assert imgs["total"] == 2
    finally:
        _drop(ds_id)


# ------------------------------------------------------------------ frozen orders (one sort per list, not per page)


def _many(tmp_path):
    """14 images, 3 boxes each, with ties in every sort key (equal areas, classes, aspects) and two frame shapes."""
    files = {}
    for k in range(14):
        size = (200, 100) if k % 3 else (100, 100)
        lines = [f"{k % 3} 0.25 0.5 0.2 0.4", f"{(k + 1) % 3} 0.5 0.5 {0.1 + 0.05 * (k % 4):.2f} 0.2",
                 f"{k % 2} 0.75 0.25 0.1 0.1"]
        files[f"f{k:02d}"] = (size, [], "\n".join(lines) + "\n", "train" if k % 4 else "val")
    return _dataset(tmp_path, files)


def _pages(c, ds_id, params, step):
    """The whole list read the way the grid reads it: a first page, then keyset continuations with an offset."""
    first = c.get(f"/api/datasets/{ds_id}/boxes", params={**params, "limit": step}).json()
    out, total = list(first["items"]), first["total"]
    while len(out) < total:
        res = c.get(f"/api/datasets/{ds_id}/boxes", params={**params, "limit": step, "offset": 1,
                                                             "after": json.dumps(out[-2]["k"])}).json()
        if not res["items"]:
            break
        out.extend(res["items"])
    return total, out


def test_frozen_orders_page_exactly_like_sql(tmp_path, monkeypatch):
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    try:
        c = _client()
        _f, _p = _finding(ds_id, db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/f05.jpg'", (ds_id,))["id"],
                          "swap", 2, (0.5, 0.5, 0.15, 0.2), 1, 0.9)
        ident = lambda items: [(it["image_id"], it["row"], it["cls"], it["k"]) for it in items]   # noqa: E731
        cases = [(s, o, f) for s in ("class", "area", "aspect", "random", "audit", "name") for o in ("asc", "desc")
                 for f in ({}, {"split": "train"}, {"cls": "0,2"})]
        monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
        monkeypatch.setattr(objects, "SHALLOW_ROWS", 0)
        frozen = {}
        for s, o, f in cases:
            params = {"sort": s, "order": o, "seed": 5, **f}
            total, items = _pages(c, ds_id, params, 7)
            assert len(items) == total and len({(it["image_id"], it["row"]) for it in items}) == total, (s, o, f)
            offs = [it for off in range(0, total, 9) for it in
                    c.get(f"/api/datasets/{ds_id}/boxes", params={**params, "offset": off, "limit": 9}).json()["items"]]
            assert ident(offs) == ident(items), (s, o, f)
            frozen[(s, o, str(f))] = (total, ident(items))
        assert objects._frozen, "lists were frozen"
        objects.drop_frozen_orders()
        monkeypatch.setattr(objects, "_frozen_order", lambda *a, **k: None)          # plain SQL paging
        for s, o, f in cases:
            total, items = _pages(c, ds_id, {"sort": s, "order": o, "seed": 5, **f}, 7)
            assert (total, ident(items)) == frozen[(s, o, str(f))], (s, o, f)
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        objects.drop_frozen_orders()
        _drop(ds_id)


def test_frozen_orders_follow_label_writes_and_ai_decisions(tmp_path, monkeypatch):
    from fovea.api import objects
    from fovea.core.scanner import refresh_image
    ds_id, root = _many(tmp_path)
    monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
    monkeypatch.setattr(objects, "SHALLOW_ROWS", 0)
    url = f"/api/datasets/{ds_id}/boxes"
    try:
        c = _client()
        img = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        top = lambda **p: c.get(url, params={"sort": "audit", "limit": 3, **p}).json()["items"]   # noqa: E731
        before = top()
        assert not any(it.get("suspects") for it in before) and objects._frozen
        # a finding written straight into the table (no hook): the audit order sees it on the next page
        a, run = _finding(ds_id, img["train/f05.jpg"], "swap", 2, (0.5, 0.5, 0.15, 0.2), 1, 0.9)
        b, _ = _finding(ds_id, img["train/f06.jpg"], "spurious", 0, (0.75, 0.25, 0.1, 0.1), 2, 0.8, run=run)
        got = top()
        assert [(it["image_id"], it["row"]) for it in got[:2]] == [(img["train/f05.jpg"], 1), (img["train/f06.jpg"], 2)]
        # accept one finding and reopen another: same number pending, a different set — the order changes
        db.execute("UPDATE ai_preds SET state='accepted' WHERE id=?", (a,))
        c2, _ = _finding(ds_id, img["train/f07.jpg"], "spurious", 1, (0.75, 0.25, 0.1, 0.1), 2, 0.7, run=run)
        db.execute("UPDATE ai_preds SET state='rejected' WHERE id=?", (c2,))
        top()                                                     # freezes: {b} pending
        db.execute("UPDATE ai_preds SET state='rejected' WHERE id=?", (b,))
        db.execute("UPDATE ai_preds SET state='pending' WHERE id=?", (c2,))
        got = top()
        assert (got[0]["image_id"], got[0]["row"]) == (img["train/f07.jpg"], 2) and got[0]["suspects"][0]["id"] == c2
        # an edit through the API (label-write hook) moves the box in a frozen class order at once
        cls_list = c.get(url, params={"sort": "class", "limit": 200}).json()["items"]
        it = next(x for x in cls_list if x["cls"] == 0)
        c.patch(url, json={"edits": [{"image_id": it["image_id"], "row": it["row"], "version": it["version"], "cls": 2}]})
        after = c.get(url, params={"sort": "class", "limit": 200}).json()["items"]
        moved = next(x for x in after if (x["image_id"], x["row"]) == (it["image_id"], it["row"]))
        assert moved["cls"] == 2 and after.index(moved) > len([x for x in after if x["cls"] < 2])
        assert [x["cls"] for x in after] == sorted(x["cls"] for x in after)
        # a file changed outside Fovea and re-indexed without any announcement: new box ids void the order too
        lp = root / "labels" / "train" / "f01.txt"
        lp.write_text(lp.read_text() + "0 0.1 0.1 0.05 0.05\n")
        refresh_image(img["train/f01.jpg"], update_counts=False)
        res = c.get(url, params={"sort": "class", "limit": 3}).json()      # a page without any of f01's old boxes
        assert res["total"] == 43
        assert (res["items"][0]["rel_path"], res["items"][0]["cls"], res["items"][0]["xc"]) == ("train/f01.jpg", 0, 0.1)
        # rewriting the image that holds the highest box ids gives its boxes the same ids again; the writer's
        # label-write announcement is what voids the order then
        from fovea.core.labels import notify_label_write
        last = db.query_one("SELECT image_id FROM boxes WHERE id = (SELECT MAX(id) FROM boxes)")["image_id"]
        assert last == img["train/f01.jpg"]
        c.get(url, params={"sort": "class", "limit": 3})
        lp.write_text(lp.read_text().replace("0 0.1 0.1 0.05 0.05", "2 0.1 0.1 0.05 0.05"))
        refresh_image(last, update_counts=False)
        notify_label_write(ds_id, [last])
        res = c.get(url, params={"sort": "class", "limit": 3}).json()
        assert res["items"][0]["rel_path"] != "train/f01.jpg" and [x["cls"] for x in res["items"]] == [0, 0, 0]
        # a frozen order that missed a write never drops boxes from a page: it is sorted again
        c.get(url, params={"sort": "aspect", "limit": 5})
        monkeypatch.setattr(objects, "_order_version", lambda *a, **k: ("frozen",))
        c.get(url, params={"sort": "aspect", "limit": 5})
        lp = root / "labels" / "train" / "f02.txt"
        lp.write_text("1 0.5 0.5 0.3 0.3\n")
        refresh_image(img["train/f02.jpg"], update_counts=False)
        full = c.get(url, params={"sort": "aspect", "limit": 200}).json()
        assert len(full["items"]) == full["total"] == 41
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        objects.drop_frozen_orders()
        _drop(ds_id)


def test_small_lists_are_not_held_and_area_pages_off_the_index(tmp_path):
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    try:
        c = _client()
        objects.drop_frozen_orders()
        deep = c.get(f"/api/datasets/{ds_id}/boxes", params={"sort": "audit", "offset": 2000}).json()
        assert deep["total"] == 42 and deep["items"] == []
        assert not objects._frozen, "42 boxes sort in no time: nothing is held"
        _pre, _pp, body, params, exprs = objects._box_query(ds_id, {}, [], "area", 0)
        sql = f"SELECT b.id, {', '.join(exprs)} {body} ORDER BY {', '.join(e + ' ASC' for e in exprs)} LIMIT 5 OFFSET 30"
        if "idx_boxes_ds_area" in {r["name"] for r in db.query("PRAGMA index_list(boxes)")}:
            assert objects._page_plan(sql, params + []) == "index"
        assert objects._page_plan(sql.replace("(b.w * b.h)", "b.xc"), params) == "sort"
    finally:
        _drop(ds_id)


def test_first_pages_use_sql_and_a_browsed_list_freezes_in_the_background(tmp_path, monkeypatch):
    import time
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
    url = f"/api/datasets/{ds_id}/boxes"
    try:
        c = _client()
        objects.drop_frozen_orders()
        # a one-off peek (Overview's class examples: a first page with a fresh shuffle each time) holds nothing
        for seed in (1, 2, 3):
            assert len(c.get(url, params={"sort": "random", "seed": seed, "cls": "0", "limit": 4}).json()["items"]) == 4
        time.sleep(0.2)
        assert not objects._frozen and not objects._warming
        # the grid opening a list (counts=1) answers by SQL and freezes the order for the pages to come
        first = c.get(url, params={"sort": "aspect", "limit": 10, "counts": 1}).json()
        for _ in range(200):
            if objects._frozen and not objects._warming:
                break
            time.sleep(0.01)
        assert len(objects._frozen) == 1
        rest = c.get(url, params={"sort": "aspect", "limit": 50, "after": json.dumps(first["items"][-1]["k"])}).json()
        sql = c.get(url, params={"sort": "aspect", "limit": 60}).json()["items"]
        assert [(x["image_id"], x["row"]) for x in first["items"] + rest["items"]] == [(x["image_id"], x["row"]) for x in sql]
    finally:
        objects.drop_frozen_orders()
        _drop(ds_id)


def _ident(items):
    return [(it["image_id"], it["row"], it["cls"], it["k"]) for it in items]


def _sql_pages(c, ds_id, params, step, monkeypatch):
    """The same list paged by plain SQL (no frozen order), for comparison."""
    from fovea.api import objects
    with monkeypatch.context() as m:
        m.setattr(objects, "_frozen_order", lambda *a, **k: None)
        return _pages(c, ds_id, params, step)


def _count_builds(monkeypatch):
    from fovea.api import objects
    builds = []
    real = objects._build_frozen
    monkeypatch.setattr(objects, "_build_frozen", lambda q, total=None: builds.append(q.key) or real(q, total))
    return builds


def test_announced_writes_patch_frozen_orders_instead_of_sorting_them_again(tmp_path, monkeypatch):
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
    monkeypatch.setattr(objects, "SHALLOW_ROWS", 0)
    url = f"/api/datasets/{ds_id}/boxes"
    try:
        c = _client()
        img = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        _f, run = _finding(ds_id, img["train/f05.jpg"], "swap", 2, (0.5, 0.5, 0.15, 0.2), 1, 0.9)
        _finding(ds_id, img["train/f06.jpg"], "spurious", 0, (0.75, 0.25, 0.1, 0.1), 2, 0.8, run=run)
        cases = [("class", "asc", {}), ("class", "desc", {"cls": "0"}), ("aspect", "desc", {}),
                 ("random", "asc", {"split": "train"}), ("audit", "desc", {}), ("audit", "asc", {"cls": "0,1"}),
                 ("name", "asc", {"cls": "2"})]
        for s, o, f in cases:
            _pages(c, ds_id, {"sort": s, "order": o, "seed": 3, **f}, 7)
        assert len(objects._frozen) == len(cases)
        rows = c.get(url, params={"sort": "name", "limit": 200}).json()["items"]
        builds = _count_builds(monkeypatch)
        patched, real_patch = set(), objects._patched
        monkeypatch.setattr(objects, "_patched", lambda fz, q, *a: patched.add(q.key) or real_patch(fz, q, *a))
        pick = lambda name, row: next(x for x in rows if x["rel_path"] == name and x["row"] == row)   # noqa: E731
        a, b, d, e = pick("train/f03.jpg", 0), pick("train/f05.jpg", 1), pick("train/f07.jpg", 0), pick("train/f09.jpg", 2)
        assert a["cls"] == 0
        edits = [
            [{"image_id": a["image_id"], "row": 0, "version": a["version"], "cls": 2}],     # leaves the cls=0 list
            [{"image_id": b["image_id"], "row": 1, "version": b["version"], "delete": True}],   # rows after it shift
            [{"image_id": d["image_id"], "row": 0, "version": "stale", "cls": 1}],      # refused, but the image is re-read
            [{"image_id": e["image_id"], "row": 2, "version": e["version"], "cls": e["cls"]}],   # changes nothing
        ]
        for ed in edits:
            assert c.patch(url, json={"edits": ed}).status_code == 200
        for s, o, f in cases:
            params = {"sort": s, "order": o, "seed": 3, **f}
            total, items = _pages(c, ds_id, params, 7)
            sql_total, sql_items = _sql_pages(c, ds_id, params, 7, monkeypatch)
            assert (total, _ident(items)) == (sql_total, _ident(sql_items)), (s, o, f)
        assert builds == [] and len(patched) == len(cases), "every order followed the edits by a patch, not a new sort"
        moved = [x for x in _pages(c, ds_id, {"sort": "class", "seed": 3}, 50)[1] if x["image_id"] == a["image_id"] and x["row"] == 0]
        assert len(moved) == 1 and moved[0]["cls"] == 2
        assert not any(x["image_id"] == a["image_id"] and x["row"] == 0
                       for x in _pages(c, ds_id, {"sort": "class", "order": "desc", "seed": 3, "cls": "0"}, 7)[1])
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        objects.drop_frozen_orders()
        _drop(ds_id)


def test_a_decision_keeps_continuations_on_the_held_order_while_a_new_one_is_sorted(tmp_path, monkeypatch):
    import time
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
    url = f"/api/datasets/{ds_id}/boxes"

    def settle():
        for _ in range(300):
            if objects._frozen and not objects._warming:
                return
            time.sleep(0.01)
    try:
        c = _client()
        img = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        a, run = _finding(ds_id, img["train/f05.jpg"], "swap", 2, (0.5, 0.5, 0.15, 0.2), 1, 0.9)
        _finding(ds_id, img["train/f06.jpg"], "spurious", 0, (0.75, 0.25, 0.1, 0.1), 2, 0.8, run=run)
        p = {"sort": "audit", "limit": 5}
        first = c.get(url, params={**p, "counts": 1}).json()["items"]          # SQL; the order sorts in the background
        settle()
        held = list(objects._frozen.values())[0]
        assert [(x["image_id"], x["row"]) for x in first[:2]] == [(img["train/f05.jpg"], 1), (img["train/f06.jpg"], 2)]
        after = json.dumps(first[1]["k"])
        before = c.get(url, params={**p, "after": after}).json()["items"]
        later, real_later = [], objects._sort_later
        monkeypatch.setattr(objects, "_sort_later", lambda q: later.append(q.key))
        builds = _count_builds(monkeypatch)
        db.execute("UPDATE ai_preds SET state='accepted', decided_at=? WHERE id=?", (db.now(), a))     # an AI decision
        # the grid scrolling on from boxes it holds: the held order answers at once and is sorted again behind it
        cont = c.get(url, params={**p, "after": after}).json()["items"]
        assert _ident(cont) == _ident(before) and not builds and later
        # a first page is exact: the accepted finding no longer puts its box on top
        top = c.get(url, params=p).json()["items"]
        assert (top[0]["image_id"], top[0]["row"]) == (img["train/f06.jpg"], 2) and not builds
        assert list(objects._frozen.values())[0] is held
        monkeypatch.setattr(objects, "_sort_later", real_later)
        c.get(url, params={**p, "after": after})
        settle()
        fresh = list(objects._frozen.values())[0]
        assert fresh is not held and fresh.version[1] != held.version[1] and builds
        total, items = _pages(c, ds_id, {"sort": "audit"}, 7)
        sql_total, sql_items = _sql_pages(c, ds_id, {"sort": "audit"}, 7, monkeypatch)
        assert (total, _ident(items)) == (sql_total, _ident(sql_items))
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        objects.drop_frozen_orders()
        _drop(ds_id)


def test_short_audit_lists_read_only_their_own_images_findings(tmp_path, monkeypatch):
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    url = f"/api/datasets/{ds_id}/boxes"
    try:
        c = _client()
        img = {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}
        _ensure_ref_box_column()
        _f, run = _finding(ds_id, img["train/f05.jpg"], "swap", 2, (0.5, 0.5, 0.15, 0.2), 1, 0.9)
        _finding(ds_id, img["train/f06.jpg"], "spurious", 0, (0.75, 0.25, 0.1, 0.1), 2, 0.8, run=run)
        _finding(ds_id, img["val/f04.jpg"], "loose", 1, (0.25, 0.5, 0.18, 0.38), 0, 0.7, ref_box="[0.25, 0.5, 0.2, 0.4]", run=run)
        _finding(ds_id, img["train/f01.jpg"], "loose", 0, (0.75, 0.25, 0.12, 0.1), 2, 0.6, run=run)    # by row only
        sqls = []
        real_query = db.query
        monkeypatch.setattr(db, "query", lambda sql, *a, **k: sqls.append(sql) or real_query(sql, *a, **k))
        builds = _count_builds(monkeypatch)
        cases = [{"cls": "0"}, {"cls": "1", "split": "train"}, {"q": "f0"}, {"split": "val"}, {}]
        got = {}
        for f in cases:
            for o in ("desc", "asc"):
                params = {"sort": "audit", "order": o, **f}
                first = c.get(url, params={**params, "limit": 4}).json()
                more = c.get(url, params={**params, "limit": 4, "offset": 1, "after": json.dumps(first["items"][-1]["k"])}).json()
                deep = c.get(url, params={**params, "limit": 4, "offset": 2500}).json()
                got[(str(f), o)] = (first["total"], _ident(first["items"]), _ident(more["items"]), deep["items"],
                                    [x.get("suspects") for x in first["items"]])
        assert any("fi AS MATERIALIZED" in s for s in sqls), "short lists ran the scoped CTE"
        assert any("INDEXED BY idx_boxes_ds_cls" in s for s in sqls)
        assert builds == [], "a short audit-sorted list is paged by SQL at any depth: nothing to hold"
        assert got[(str({"split": "val"}), "desc")][1][0][:2] == (img["val/f04.jpg"], 0)
        assert [x[:2] for x in got[(str({"cls": "0"}), "desc")][1][:2]] == [(img["train/f05.jpg"], 1), (img["train/f06.jpg"], 2)]
        # the same lists with the CTE over every finding of the dataset
        monkeypatch.setattr(objects, "_scoped_prefix", lambda *a, **k: None)
        sqls.clear()
        for f in cases:
            for o in ("desc", "asc"):
                params = {"sort": "audit", "order": o, **f}
                first = c.get(url, params={**params, "limit": 4}).json()
                more = c.get(url, params={**params, "limit": 4, "offset": 1, "after": json.dumps(first["items"][-1]["k"])}).json()
                deep = c.get(url, params={**params, "limit": 4, "offset": 2500}).json()
                assert got[(str(f), o)] == (first["total"], _ident(first["items"]), _ident(more["items"]), deep["items"],
                                            [x.get("suspects") for x in first["items"]]), (f, o)
        assert not any("fi AS MATERIALIZED" in s for s in sqls)
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
        objects.drop_frozen_orders()
        _drop(ds_id)


def test_whole_dataset_audit_order_reads_the_covering_class_index(tmp_path):
    from fovea.api import objects
    ds_id, _root = _many(tmp_path)
    try:
        _pre, _pp, body, params, _exprs = objects._box_query(ds_id, {}, [], "audit", 0)
        assert "INDEXED BY idx_boxes_ds_cls" in objects._steered(body)
        _pre, _pp, fbody, _fp, _e = objects._box_query(ds_id, {"split": "train"}, [], "audit", 0)
        assert objects._steered(fbody) == fbody            # an image filter: the planner picks the join order
        _pre, _pp, abody, _ap, _e = objects._box_query(ds_id, {}, [], "aspect", 0)
        assert objects._steered(abody) == abody
    finally:
        _drop(ds_id)


def test_box_orders_are_sorted_again_when_patching_cannot_follow(tmp_path, monkeypatch):
    from fovea.api import objects
    from fovea.core.scanner import refresh_image
    ds_id, _root = _many(tmp_path)
    other, root2 = _three(tmp_path / "other")
    monkeypatch.setattr(objects, "ORDER_MIN_BOXES", 0)
    monkeypatch.setattr(objects, "SHALLOW_ROWS", 0)
    url = f"/api/datasets/{ds_id}/boxes"
    params = {"sort": "class", "order": "desc"}
    try:
        c = _client()
        _pages(c, ds_id, params, 7)
        builds = _count_builds(monkeypatch)
        # another dataset's announced edit moves the highest box id: this order still holds, untouched
        it = c.get(f"/api/datasets/{other}/boxes", params={"limit": 1}).json()["items"][0]
        assert c.patch(f"/api/datasets/{other}/boxes", json={"edits": [{"image_id": it["image_id"], "row": it["row"], "version": it["version"], "cls": 2}]}).json()["updated"] == 1
        _pages(c, ds_id, params, 7)
        assert [k for k in builds if k[0] == ds_id] == []
        # more written images than the log keeps since the order was taken: sorted again
        monkeypatch.setattr(objects, "LOG_IMAGES", 1)
        rows = c.get(url, params={"sort": "name", "limit": 200}).json()["items"]
        builds.clear()
        for x in [r for r in rows if r["row"] == 0][:3]:
            assert c.patch(url, json={"edits": [{"image_id": x["image_id"], "row": 0, "version": x["version"], "cls": (x["cls"] + 1) % 3}]}).json()["updated"] == 1
        total, items = _pages(c, ds_id, params, 7)
        assert builds and (total, _ident(items)) == (lambda r: (r[0], _ident(r[1])))(_sql_pages(c, ds_id, params, 7, monkeypatch))
        # another dataset's boxes re-read without an announcement: nothing says what moved, so sort again
        _pages(c, ds_id, params, 7)
        builds.clear()
        (root2 / "labels" / "train" / "a.txt").write_text("1 0.25 0.5 0.2 0.4\n")
        refresh_image(db.query_one("SELECT id FROM images WHERE dataset_id=? AND rel_path='train/a.jpg'", (other,))["id"], update_counts=False)
        _pages(c, ds_id, params, 7)
        assert builds
    finally:
        objects.drop_frozen_orders()
        _drop(ds_id)
        _drop(other)


def test_order_patch_helpers():
    from array import array
    from fovea.api import objects
    imgs = array("q", [5 << 32, 0, 5, 7, 5, 9])          # the bytes of 5 also start half-way into the first entry
    assert objects._positions(imgs, {5}) == [2, 4]
    assert objects._positions(imgs, {5, 9, 0}) == [1, 2, 4, 5]
    assert objects._positions(imgs, set(range(40)) | {5 << 32}) == [0, 1, 2, 3, 4, 5]
    assert objects._spliced(imgs, [1, 4], ()) == array("q", [5 << 32, 5, 7, 9])
    assert objects._spliced(["a", "c"], (), [(0, "0"), (1, "b"), (2, "d")]) == ["0", "a", "b", "c", "d"]
    cols = [array("d", [0.1, 0.2, 0.2]), array("q", [3, 1, 8])]
    assert [objects._insertion(cols, 3, k) for k in ((0.0, 1), (0.2, 0), (0.2, 5), (0.3, 0))] == [0, 1, 2, 3]
    assert objects._fits(1.0, cols[0], False) and not objects._fits(1, cols[0], False) and not objects._fits(None, cols[1], False)
    assert objects._fits("x", ["a"], True) and not objects._fits(2, ["a"], True) and objects._fits(2, [1, 0.5], False)
