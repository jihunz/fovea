"""AI platform (ai-core): secrets, config, registry, weights, class mapping, geometry, proposals, batch / audit /
benchmark / accept jobs, estimate, two-stage, guards — driven through the API with the deterministic mock provider.
Local ultralytics tests run only when ultralytics and the weights are present. Run: python -m pytest tests -q"""
from __future__ import annotations

import http.server
import re
import json
import os
import stat
import sys
import threading
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.ai import classes as C  # noqa: E402
from fovea.ai import config as ai_config  # noqa: E402
from fovea.ai import geometry, registry, secrets, store, weights  # noqa: E402
from fovea.ai.base import ClassSpec  # noqa: E402
from fovea.core import layout  # noqa: E402

CLASSES = ["person", "fall", "bike"]


# ------------------------------------------------------------------ fixtures

def _scene(path: Path, size, rects):
    path.parent.mkdir(parents=True, exist_ok=True)
    W, H = size
    im = Image.new("RGB", size, (128, 128, 128))
    d = ImageDraw.Draw(im)
    for xc, yc, w, h in rects:
        d.rectangle([round((xc - w / 2) * W), round((yc - h / 2) * H), round((xc + w / 2) * W), round((yc + h / 2) * H)],
                    fill=(200, 60, 60))
    im.save(path, quality=92)


def _make_dataset(tmp_path, n=12, unlabeled=(), empty=()):
    """n images in images/train; label files with 1-3 boxes except `unlabeled` (no file) and `empty` (empty file)."""
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"ds-{uuid.uuid4().hex[:6]}"
    for k in range(n):
        rects = [(0.2 + 0.25 * j, 0.5, 0.15, 0.3 + 0.05 * j) for j in range(1 + k % 3)]
        _scene(root / "images" / "train" / f"img_{k:03d}.jpg", (320, 240), rects)
        lp = root / "labels" / "train" / f"img_{k:03d}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        if k in unlabeled:
            continue
        if k in empty:
            lp.write_text("")
            continue
        lp.write_text("".join(f"{(k + j) % 3} {xc:.6f} {yc:.6f} {w:.6f} {h:.6f}\n" for j, (xc, yc, w, h) in enumerate(rects)))
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "aic-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(CLASSES), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id, root


def _ids(ds_id):
    return [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY rel_path", (ds_id,))]


def _wait(job_id, timeout=60):
    from fovea.jobs import jobs
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(job_id)
        if j and j.status in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fastapi.testclient import TestClient
    from fovea.main import app
    registry.reset()
    return TestClient(app)


@pytest.fixture()
def ds(tmp_path):
    ds_id, root = _make_dataset(tmp_path, n=12, unlabeled=(10,), empty=(11,))
    yield ds_id, root
    db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _label_text(root, k):
    p = root / "labels" / "train" / f"img_{k:03d}.txt"
    return p.read_text() if p.exists() else None


# ------------------------------------------------------------------ secrets / config

def test_secrets_precedence_file_mode_and_scrub(monkeypatch, tmp_path):
    monkeypatch.setattr(secrets, "DATA_DIR", tmp_path)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    assert secrets.key_info("anthropic")["present"] is False
    info = secrets.set_key("anthropic", "  sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV-1234  ")
    assert info == {"present": True, "source": "file", "hint": "…1234", "env": None}
    p = tmp_path / "secrets.json"
    assert stat.S_IMODE(p.stat().st_mode) == 0o600
    assert secrets.get_key("anthropic") == "sk-ant-api03-ABCDEFGHIJKLMNOPQRSTUV-1234"
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-env-key-ZZZZZZZZZZZZ9999")
    assert secrets.get_key("anthropic").endswith("9999") and secrets.key_info("anthropic")["source"] == "env"
    msg = secrets.scrub("401 for key sk-ant-env-key-ZZZZZZZZZZZZ9999 and Bearer abcdefghijkl and AIzaSyA1234567890123456789")
    assert "ZZZZ" not in msg and "abcdefghijkl" not in msg and "AIzaSy" not in msg
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert secrets.clear_key("anthropic")["present"] is False
    with pytest.raises(ValueError):
        secrets.set_key("openai", "has space inside")


def test_config_validation():
    db.init_db()                       # also when this test runs first on a fresh data dir
    with pytest.raises(ai_config.ConfigError):
        ai_config.update_provider_config("openai_compat", {"api_key": "sk-123"})
    with pytest.raises(ai_config.ConfigError):
        ai_config.update_provider_config("openai_compat", {"base_url": "ftp://x"})
    with pytest.raises(ai_config.ConfigError):
        ai_config.update_provider_config("openai_compat", {"base_url": "http://u:p@host/v1"})
    with pytest.raises(ai_config.ConfigError):
        ai_config.update_provider_config("anthropic", {"concurrency": 99})
    cfg = ai_config.update_provider_config("openai_compat", {"base_url": "http://localhost:11434/v1/", "effort": "Accurate",
                                                             "coord": "qwen25"})
    assert cfg["base_url"] == "http://localhost:11434/v1" and cfg["effort"] == "high" and cfg["coord"] == "qwen25"
    with pytest.raises(ai_config.ConfigError):
        ai_config.update_ai_settings({"device": "tpu"})


def test_keys_never_leave_the_server(client, monkeypatch, tmp_path):
    monkeypatch.setattr(secrets, "DATA_DIR", tmp_path)
    key = "sk-proj-SECRETSECRETSECRET-7777"
    r = client.put("/api/ai/providers/openai/key", json={"key": key})
    assert r.status_code == 200 and r.json()["key"]["hint"] == "…7777"
    for url in ("/api/ai/providers", "/api/ai/providers/openai/models", "/api/ai/settings", "/api/ai/usage"):
        assert "SECRETSECRET" not in client.get(url).text
    r = client.put("/api/ai/providers/openai/config", json={"openai_key": key})
    assert r.status_code == 422
    assert client.delete("/api/ai/providers/openai/key").json()["key"]["present"] is False
    assert client.put("/api/ai/providers/yolo/key", json={"key": "x" * 20}).status_code == 422


def test_remote_clients_cannot_use_cloud_or_keys(client, ds, monkeypatch):
    ds_id, _ = ds
    from fovea.api import ai as ai_api
    monkeypatch.setattr(ai_api, "_is_local", lambda req: False)
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    iid = _ids(ds_id)[0]
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 403
    assert client.put("/api/ai/providers/anthropic/key", json={"key": "sk-ant-zzzzzzzzzzzzzzzz"}).status_code == 403
    monkeypatch.setenv("FOVEA_ALLOW_REMOTE_AI", "1")
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 200


def test_mock_only_with_env(monkeypatch):
    monkeypatch.delenv("FOVEA_AI_MOCK", raising=False)
    assert "mock" not in registry.ids()
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    assert "mock" in registry.ids()


# ------------------------------------------------------------------ weights / mapping / geometry

def test_weights_resolve_refuses_paths(tmp_path, monkeypatch):
    md = tmp_path / "model"
    md.mkdir()
    (md / "yolo26n.pt").write_bytes(b"x" * 10)
    (md / "sam2.1_t.pt").write_bytes(b"x" * 10)
    (md / "mobileclip2_b.ts").write_bytes(b"x" * 10)
    (tmp_path / "evil.pt").write_bytes(b"x")
    monkeypatch.setattr(weights, "MODEL_DIR", md)
    assert weights.resolve("yolo26n") == (md / "yolo26n.pt").resolve()
    for bad in ("../evil", "../model/yolo26n", "/etc/passwd", "yolo26n/../../evil", "..", ""):
        with pytest.raises(ValueError):
            weights.resolve(bad)
    with pytest.raises(ValueError):
        weights.resolve("sam2.1_t", "yolo")
    fams = {w["name"]: w["family"] for w in weights.local_weights()}
    assert fams == {"yolo26n": "yolo", "sam2.1_t": "sam", "mobileclip2_b": "text"}
    lst = {w["name"]: w for w in weights.weights_list()}
    assert lst["yolo26n.pt"]["downloaded"] and not lst["yolo26s.pt"]["downloaded"]


def test_weights_download_curated_only(tmp_path, monkeypatch):
    md = tmp_path / "model"
    md.mkdir()
    srv_dir = tmp_path / "srv"
    srv_dir.mkdir()
    (srv_dir / "yolo26n.pt").write_bytes(os.urandom(5000))

    class H(http.server.SimpleHTTPRequestHandler):
        def __init__(self, *a, **k):
            super().__init__(*a, directory=str(srv_dir), **k)

        def log_message(self, *a):
            pass

    httpd = http.server.ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    try:
        monkeypatch.setattr(weights, "MODEL_DIR", md)
        res = weights.download("yolo26n", base=f"http://127.0.0.1:{httpd.server_port}/")
        assert res["downloaded"] and (md / "yolo26n.pt").stat().st_size == 5000
        assert not list(md.glob("*.part"))
        assert weights.download("yolo26n.pt")["already"]
        with pytest.raises(ValueError):
            weights.download("../../etc/passwd")
    finally:
        httpd.shutdown()


def test_map_vocab_by_name_synonym_and_explicit():
    coco = {0: "person", 1: "bicycle", 5: "bus", 27: "tie", 62: "tv"}
    specs = [ClassSpec(0, "Person"), ClassSpec(1, "fall"), ClassSpec(2, "bike"), ClassSpec(3, "monitor")]
    m, how = C.map_vocab(coco, specs)
    assert m == {0: 0, 1: 2, 62: 3} and how["person"] == "name" and how["bicycle"] == "synonym"
    m, _ = C.map_vocab(coco, specs, {"0": 1, "tv": None})
    assert m[0] == 1 and 62 not in m
    m, _ = C.map_vocab(coco, specs, {"0": 1, "*": "skip"})
    assert m == {0: 1}
    with pytest.raises(ValueError):
        C.map_vocab(coco, specs, {"0": 7})                # target outside the classes being labeled
    with pytest.raises(ValueError):
        C.map_vocab(coco, specs, {"999": 0})
    placeholder = [ClassSpec(0, "class_0"), ClassSpec(1, "class_1")]
    assert C.map_vocab(coco, placeholder)[0] == {}         # 'class_0' never matches anything by name


def test_geometry_dedupe_nms_and_label_issues():
    ex = [[0, 0.5, 0.5, 0.2, 0.2]]
    new = [{"cls": 0, "xc": 0.505, "yc": 0.5, "w": 0.2, "h": 0.2}, {"cls": 1, "xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2},
           {"cls": 1, "xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2}]
    fresh, dup = geometry.split_duplicates(new, ex)
    assert len(dup) == 2 and len(fresh) == 1 and fresh[0]["cls"] == 1
    gt = [[0, 0.2, 0.5, 0.2, 0.4], [1, 0.6, 0.5, 0.2, 0.2], [0, 0.9, 0.1, 0.05, 0.05], [0, 0.5, 0.85, 0.2, 0.2]]
    preds = [{"cls": 0, "xc": 0.2, "yc": 0.5, "w": 0.2, "h": 0.4, "conf": 0.9},          # matches row 0
             {"cls": 0, "xc": 0.6, "yc": 0.5, "w": 0.2, "h": 0.2, "conf": 0.9},          # row 1 looks like class 0
             {"cls": 1, "xc": 0.8, "yc": 0.8, "w": 0.1, "h": 0.1, "conf": 0.8},          # nothing there in GT
             {"cls": 0, "xc": 0.52, "yc": 0.85, "w": 0.30, "h": 0.25, "conf": 0.9}]      # row 3 is loose
    kinds = {(f["kind"], f["ref_row"]) for f in geometry.label_issues(gt, preds)}
    assert ("swap", 1) in kinds and ("missing", None) in kinds and ("spurious", 2) in kinds and ("loose", 3) in kinds
    assert geometry.pool_score([0.9, 0.1]) > 0.85 and geometry.pool_score([]) == 0.0


# ------------------------------------------------------------------ propose / proposals

def test_propose_stores_and_replaces_pending(client, ds):
    ds_id, root = ds
    iid = _ids(ds_id)[0]
    before = _label_text(root, 0)
    body = {"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}
    r = client.post("/api/ai/propose", json=body)
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["run_id"] and j["preds"] and j["usage"]["cost_usd"] > 0 and "served_by" in j
    assert all(set(p) >= {"id", "cls", "xc", "yc", "w", "h", "conf", "label", "kind", "reason"} for p in j["preds"])
    assert _label_text(root, 0) == before                    # proposals never touch label files
    r2 = client.post("/api/ai/propose", json=body).json()
    got = client.get(f"/api/datasets/{ds_id}/images/{iid}/proposals").json()
    assert {p["id"] for p in got["preds"]} == {p["id"] for p in r2["preds"]}
    assert str(r2["run_id"]) in got["runs"] and got["runs"][str(r2["run_id"])]["provider"] == "mock"
    ids = [p["id"] for p in r2["preds"]]
    d = client.post(f"/api/datasets/{ds_id}/proposals/decide", json={"accept": ids[:1], "reject": ids[1:]}).json()
    assert d["updated"] == len(ids)
    assert client.get(f"/api/datasets/{ds_id}/images/{iid}/proposals").json()["preds"] == []


def test_propose_validation(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    base = {"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}
    for bad, code in (({"conf": "abc"}, 422), ({"conf": 2}, 422), ({"provider": "nope"}, 422), ({"classes": [7]}, 422),
                      ({"classes": ["x"]}, 422), ({"image_id": 999999999}, 404), ({"options": {"pipeline": "x"}}, 422)):
        r = client.post("/api/ai/propose", json={**base, **bad})
        assert r.status_code == code, (bad, r.text)
        if code == 422:
            assert r.json().get("code") == "invalid" and r.json().get("detail")
    r = client.post("/api/ai/propose", json={**base, "conf": None})
    assert r.status_code == 200
    r = client.post("/api/ai/propose", json={**base, "provider": "yolo", "model": "../../../etc/evil"})
    assert r.status_code in (422, 503)
    r = client.post("/api/ai/detect", json={"image_id": iid, "model": "../model/yolo12m"})
    assert r.status_code in (422, 503)


def test_two_stage_with_mock(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[1]
    opts = {"pipeline": "two_stage", "detector": {"provider": "mock", "model": "mock-fast", "source_classes": ["person"]},
            "classifier": {"provider": "mock", "model": "mock-accurate"}}
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "", "classes": [0, 1],
                                              "options": opts})
    assert r.status_code == 200, r.text
    j = r.json()
    assert all(p["cls"] in (0, 1) and "→" in p["label"] for p in j["preds"])
    assert all("classifier" in p["reason"] for p in j["preds"])
    assert j["usage"]["cost_usd"] > 0.001                     # detector + classifier both billed
    bad = {**opts, "classifier": {"provider": "yolo", "model": "x"}}
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "options": bad})
    assert r.status_code in (422, 503)


def test_classify_and_preview(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[2]
    r = client.post("/api/ai/classify", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast",
                                               "boxes": [[0, 0.2, 0.5, 0.15, 0.3], [1, 0.45, 0.5, 0.3, 0.1]]})
    assert r.status_code == 200, r.text
    items = r.json()["items"]
    assert [i["index"] for i in items] == [0, 1] and all("agrees" in i for i in items)
    r = client.post("/api/ai/classify", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast",
                                               "boxes": [[0, 2, 2, 0.1, 0.1]]})
    assert r.status_code == 422
    ids = _ids(ds_id)[:3]
    r = client.post("/api/ai/preview", json={"dataset_id": ds_id, "image_ids": ids, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 200 and len(r.json()["items"]) == 3
    assert not db.query(f"SELECT 1 FROM ai_preds WHERE image_id IN ({','.join('?' * 3)}) AND state='pending' "
                        "AND run_id IN (SELECT id FROM ai_runs WHERE kind='preview')", ids)
    assert client.post("/api/ai/preview", json={"dataset_id": ds_id, "image_ids": _ids(ds_id)[:5], "provider": "mock",
                                                 "model": "mock-fast"}).status_code == 422


# ------------------------------------------------------------------ batch

def _batch(client, ds_id, **kw):
    body = {"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "target": "proposals", **kw}
    if "image_ids" not in body and "filters" not in body:
        body["filters"] = {}
    r = client.post("/api/ai/batch", json=body)
    assert r.status_code == 200, r.text
    return r.json(), _wait(r.json()["job"]["id"])


def test_batch_proposals_never_write(client, ds):
    ds_id, root = ds
    before = [_label_text(root, k) for k in range(12)]
    res, job = _batch(client, ds_id)
    assert job.status == "done", job.error
    assert [_label_text(root, k) for k in range(12)] == before
    ex = job.extra
    assert ex["images_done"] == 12 and ex["boxes"] > 0 and ex["cost_usd"] == pytest.approx(0.012, abs=1e-6)
    run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (res["run_id"],))
    assert run["status"] == "done" and run["images"] == 12 and run["cost_usd"] == pytest.approx(0.012, abs=1e-6)
    n = db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE run_id=? AND state='pending'", (res["run_id"],))["n"]
    assert n == ex["boxes"]


def test_batch_append_twice_adds_no_duplicates_and_fill_respects_empty_files(client, ds):
    ds_id, root = ds
    _res, job = _batch(client, ds_id, target="append", image_ids=_ids(ds_id)[:4])
    assert job.status == "done", job.error
    first = [_label_text(root, k) for k in range(4)]
    _res, job2 = _batch(client, ds_id, target="append", image_ids=_ids(ds_id)[:4])
    assert job2.status == "done"
    assert [_label_text(root, k) for k in range(4)] == first and job2.extra["duplicates"] > 0
    # fill: img 10 has no label file -> gets labels; img 11 has an EMPTY file (a deliberate negative) -> untouched
    _res, job3 = _batch(client, ds_id, target="fill", image_ids=_ids(ds_id)[9:12])
    assert job3.status == "done"
    assert _label_text(root, 10) and _label_text(root, 11) == ""
    assert job3.extra["skipped"] == 2
    if job3.extra.get("snapshot_id"):
        assert db.query_one("SELECT op FROM snapshots WHERE id=?", (job3.extra["snapshot_id"],))["op"] == "ai_fill"


def test_batch_keep_conf_and_replace(client, ds):
    ds_id, root = ds
    _res, job = _batch(client, ds_id, target="replace", image_ids=[_ids(ds_id)[5]], options={"keep_conf": True})
    assert job.status == "done", job.error
    rows = [ln.split() for ln in _label_text(root, 5).splitlines()]
    assert rows and all(len(r) == 6 for r in rows)
    _res, job = _batch(client, ds_id, target="replace", image_ids=[_ids(ds_id)[6]])
    rows = [ln.split() for ln in _label_text(root, 6).splitlines()]
    assert rows and all(len(r) == 5 for r in rows)


def test_batch_budget_and_errors(client, ds):
    ds_id, _ = ds
    _res, job = _batch(client, ds_id, budget_usd=0.0035)
    assert job.status == "done" and job.result["stopped"] == "budget" and job.extra["images_done"] <= 3
    res, job = _batch(client, ds_id, options={"mock_error": "auth"})
    assert job.status == "error" and "auth" in job.error.lower()
    assert db.query_one("SELECT status FROM ai_runs WHERE id=?", (res["run_id"],))["status"] == "error"
    _res, job = _batch(client, ds_id, options={"mock_error": "bad_output", "mock_error_every": 3})
    assert job.status == "done" and 0 < job.extra["errors"] < 12 and job.extra["last_error"]
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                            "target": "merge", "filters": {}})
    assert r.status_code == 422                                # unknown mode is never treated as replace (C8)
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "target": "append"})
    assert r.status_code == 422                                # no target images given


def test_batch_cancel(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "options": {"mock_latency_ms": 150}, "concurrency": 1})
    from fovea.jobs import jobs
    time.sleep(0.4)
    jobs.get(r.json()["job"]["id"]).cancel()
    job = _wait(r.json()["job"]["id"])
    assert job.status == "cancelled"
    assert db.query_one("SELECT status FROM ai_runs WHERE id=?", (r.json()["run_id"],))["status"] == "cancelled"


# ------------------------------------------------------------------ accept / reject

def test_accept_and_reject_jobs(client, ds):
    ds_id, root = ds
    ids = _ids(ds_id)
    run_id = store.create_run(ds_id, "batch", "mock", "m", {})
    # image 0: GT one class-0 box at (0.2,0.5,0.15,0.3)
    store.insert_preds(run_id, ds_id, ids[0], [
        {"cls": 0, "xc": 0.2, "yc": 0.5, "w": 0.15, "h": 0.3, "conf": 0.9},          # duplicate of GT
        {"cls": 2, "xc": 0.7, "yc": 0.3, "w": 0.1, "h": 0.1, "conf": 0.8},           # new
        {"cls": 2, "xc": 0.7, "yc": 0.8, "w": 0.1, "h": 0.1, "conf": 0.2},           # low conf
    ])
    store.insert_preds(run_id, ds_id, ids[1], [
        {"cls": 1, "xc": 0.2, "yc": 0.5, "w": 0.15, "h": 0.3, "conf": 0.9, "kind": "swap", "ref_row": 0},
        {"cls": 2, "xc": 0.45, "yc": 0.5, "w": 0.15, "h": 0.35, "conf": None, "kind": "spurious", "ref_row": 1},
        {"cls": 0, "xc": 0.9, "yc": 0.9, "w": 0.05, "h": 0.05, "conf": 0.9, "kind": "loose", "ref_row": 0},  # stale
    ])
    r = client.post(f"/api/datasets/{ds_id}/proposals/reject", json={"image_ids": [ids[0]], "min_conf": 0.5, "below": True})
    assert r.json()["rejected"] == 1
    r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": ids[:2], "min_conf": 0.5,
                                                                    "kinds": ["detect", "swap", "spurious", "loose"]})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    res = job.result
    assert res["added"] == 1 and res["duplicates"] == 1 and res["modified"] == 1 and res["deleted"] == 1 and res["stale"] == 1
    rows0 = [ln.split() for ln in _label_text(root, 0).splitlines()]
    assert len(rows0) == 2 and rows0[1][0] == "2"
    rows1 = [ln.split() for ln in _label_text(root, 1).splitlines()]
    assert len(rows1) == 1 and rows1[0][0] == "1"
    assert db.query_one("SELECT n_boxes FROM images WHERE id=?", (ids[1],))["n_boxes"] == 1
    left = db.query("SELECT kind FROM ai_preds WHERE run_id=? AND state='pending'", (run_id,))
    assert [x["kind"] for x in left] == ["loose"]
    assert client.post(f"/api/datasets/{ds_id}/proposals/accept", json={}).status_code == 422


# ------------------------------------------------------------------ audit / benchmark / estimate / runs

def test_audit_full_and_geometry(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "mode": "full"})
    assert r.status_code == 200, r.text
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.result["images_done"] == 11                     # 10 has no label file; 11's EMPTY file is audited too
    s = client.get(f"/api/datasets/{ds_id}/audit/summary").json()
    assert s["run"]["id"] == r.json()["run_id"] and s["images_audited"] >= 11
    assert sum(s["counts"].values()) == job.result["findings"]
    if s["top"]:
        assert s["top"][0]["score"] >= s["top"][-1]["score"]
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "mode": "geometry", "class_family": {"person": [0, 1]}})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "mock",
                                               "model": "mock-fast", "mode": "geometry"}).status_code == 422


def test_benchmark_and_estimate(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/benchmark", json={"dataset_id": ds_id, "sample": 8, "configs": [
        {"provider": "mock", "model": "mock-fast"},
        {"provider": "", "options": {"pipeline": "two_stage", "detector": {"provider": "mock", "model": "mock-fast"},
                                     "classifier": {"provider": "mock", "model": "mock-fast"}}}]})
    assert r.status_code == 200, r.text
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    rows = job.result
    assert len(rows) == 2 and rows[0]["images"] == 8
    assert 0.5 < rows[0]["recall"] <= 1 and rows[0]["ap50"] is not None and rows[0]["usd_per_image"] > 0
    e = client.post("/api/ai/estimate", json={"dataset_id": ds_id, "count": 1000, "provider": "mock", "model": "mock-fast"}).json()
    assert e["images"] == 1000 and e["usd"][0] < e["usd"][1] and e["usd_batch"] and e["assumptions"]
    e = client.post("/api/ai/estimate", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                               "options": {"pipeline": "two_stage",
                                                           "detector": {"provider": "mock", "model": "mock-fast"},
                                                           "classifier": {"provider": "mock", "model": "mock-fast"}}}).json()
    assert e["images"] == 12 and e["usd"][1] > 0


def test_runs_usage_and_delete(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[3]
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast",
                                              "options": {"api_key": "sk-proj-NEVERSTOREDNEVERSTORED"}})
    assert r.status_code == 200
    runs = client.get("/api/ai/runs", params={"dataset_id": ds_id}).json()["runs"]
    assert runs and "NEVERSTORED" not in json.dumps(runs)
    u = client.get("/api/ai/usage", params={"days": 7}).json()
    assert u["total_usd"] > 0 and "mock" in u["by_provider"] and u["daily"]
    run_id = r.json()["run_id"]
    d = client.delete(f"/api/ai/runs/{run_id}").json()
    assert d["deleted"] and d["preds_removed"] >= 1
    assert client.get(f"/api/datasets/{ds_id}/images/{iid}/proposals").json()["preds"] == []
    assert client.get(f"/api/ai/runs/{run_id}").status_code == 404
    assert client.get("/api/ai/usage", params={"days": 7}).json()["total_usd"] >= u["total_usd"]   # spend is kept


def test_providers_list_is_cheap(client):
    t0 = time.time()
    j = client.get("/api/ai/providers").json()
    assert time.time() - t0 < 2.0
    ids = [p["id"] for p in j["providers"]]
    assert ids[:3] == ["yolo", "yoloe", "sam"] and "mock" in ids
    for p in j["providers"]:
        assert set(p) >= {"id", "name", "kind", "location", "needs_key", "capabilities", "status", "key", "config", "models"}
    assert "device" in j and "remote_ai_allowed" in j and "prices_as_of" in j
    assert client.get("/api/ai/providers/nope/models").status_code == 404


# ------------------------------------------------------------------ local ultralytics (optional)

def _have_local(name):
    from fovea.ai import local
    return local.ultralytics_status()[0] and any(w["name"] == name for w in weights.local_weights())


def _bus():
    import ultralytics
    return str(Path(ultralytics.__file__).parent / "assets" / "bus.jpg")


@pytest.mark.skipif(not _have_local("yolo12m"), reason="ultralytics or yolo12m weights not available")
def test_local_yolo_predictor_state_is_not_shared():
    """C2: a class filter / conf from one call must never leak into another, also under concurrency."""
    from fovea.ai import local
    from fovea.ai.providers.yolo import load_entry
    e = load_entry("yolo12m", "yolo")
    path = _bus()
    full = local.predict(e, path, conf=0.25, iou=0.7, imgsz=640)
    assert {d["mcls"] for d in full} >= {0, 5}
    only_bus = local.predict(e, path, conf=0.25, iou=0.7, imgsz=640, classes=[5])
    assert {d["mcls"] for d in only_bus} == {5}
    again = local.predict(e, path, conf=0.25, iou=0.7, imgsz=640)
    assert {d["mcls"] for d in again} == {d["mcls"] for d in full}
    errors = []

    def worker(classes, conf):
        for _ in range(4):
            out = local.predict(e, path, conf=conf, iou=0.7, imgsz=640, classes=classes)
            if classes and any(d["mcls"] not in classes for d in out):
                errors.append(("class leak", classes))
            if any(d["conf"] < conf - 1e-6 for d in out):
                errors.append(("conf leak", conf))

    ts = [threading.Thread(target=worker, args=a) for a in (([0], 0.85), (None, 0.1), ([5], 0.3))]
    for t in ts:
        t.start()
    for t in ts:
        t.join()
    assert not errors


@pytest.mark.skipif(not _have_local("yolo12m"), reason="ultralytics or yolo12m weights not available")
def test_local_yolo_maps_by_name_and_skips_unmapped(client, tmp_path):
    from fovea.ai.base import DetectRequest
    from fovea.ai.providers.yolo import PROVIDER
    res = PROVIDER.detect(DetectRequest(_bus(), [ClassSpec(0, "people"), ClassSpec(1, "car")], "yolo12m", 0.25))
    assert res.detections and {d.cls for d in res.detections} <= {0, 1}
    assert all(d.label == "person" for d in res.detections if d.cls == 0)
    assert any("bus" in w for w in res.warnings)               # the bus has no dataset class: skipped and reported
    from fovea.ai.base import ProviderError
    with pytest.raises(ProviderError) as ei:
        PROVIDER.detect(DetectRequest(_bus(), [ClassSpec(0, "class_0")], "yolo12m", 0.25))
    assert ei.value.code == "invalid" and "match" in str(ei.value)


def test_review_single_image_and_target_summary(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[4]
    r = client.post("/api/ai/review", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert all(p["kind"] in store.AUDIT_KINDS for p in j["preds"]) and "score" in j
    row = db.query_one("SELECT run_id, score FROM ai_audit WHERE image_id=?", (iid,))
    assert row and row["run_id"] == j["run_id"]
    # a second review replaces the first one's pending suggestions instead of stacking them
    r2 = client.post("/api/ai/review", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}).json()
    pend = db.query("SELECT run_id FROM ai_preds WHERE image_id=? AND state='pending'", (iid,))
    assert {p["run_id"] for p in pend} <= {r2["run_id"]}
    no_label = _ids(ds_id)[10]
    assert client.post("/api/ai/review", json={"dataset_id": ds_id, "image_id": no_label, "provider": "mock",
                                                "model": "mock-fast"}).status_code == 422
    e = client.post("/api/ai/estimate", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast"}).json()
    t = e["targets"]
    assert t["images"] == 12 and t["no_label_file"] == 1 and t["empty_label_file"] == 1 and t["labeled"] == 10 and t["boxes"] > 0


def test_warm_and_legacy_validation(client, ds):
    ds_id, _ = ds
    assert client.post("/api/ai/warm", json={"provider": "mock", "model": "mock-fast"}).json()["state"] == "cloud"
    assert client.post("/api/ai/warm", json={"provider": "nope"}).status_code == 422
    r = client.post("/api/ai/autolabel", json={"dataset_id": ds_id, "model": "yolo12m", "mode": "Append", "filters": {}})
    assert r.status_code == 422
    r = client.post("/api/ai/autolabel", json={"dataset_id": ds_id, "model": "yolo12m", "mode": "append"})
    assert r.status_code == 422


def test_delete_running_run_cancels_its_job(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "options": {"mock_latency_ms": 200}, "concurrency": 1}).json()
    time.sleep(0.3)
    assert client.delete(f"/api/ai/runs/{r['run_id']}").json()["deleted"]
    job = _wait(r["job"]["id"])
    assert job.status == "cancelled"
    assert not db.query("SELECT 1 FROM ai_preds WHERE run_id=? AND state='pending'", (r["run_id"],))


@pytest.mark.skipif(not _have_local("yolo12m"), reason="ultralytics or yolo12m weights not available")
def test_mapping_endpoint(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/mapping", json={"dataset_id": ds_id, "provider": "yolo", "model": "yolo12m"})
    assert r.status_code == 200, r.text
    j = r.json()
    by = {i["model_name"]: i for i in j["items"]}
    assert by["person"]["cls"] == 0 and by["bicycle"]["cls"] == 2 and by["bus"]["cls"] is None and j["mapped"] == 2
    j = client.post("/api/ai/mapping", json={"dataset_id": ds_id, "provider": "yolo", "model": "yolo12m",
                                             "class_map": {"0": 1, "1": -1}}).json()
    by = {i["model_name"]: i for i in j["items"]}
    assert by["person"]["cls"] == 1 and by["person"]["how"] == "explicit" and by["bicycle"]["cls"] is None
    assert client.post("/api/ai/mapping", json={"dataset_id": ds_id, "provider": "yolo",
                                                 "model": "../yolo12m"}).status_code == 422


def test_planner_stats_refresh_after_ai_tables_grow(client, ds):
    """Stale ANALYZE numbers for a near-empty ai_preds turned 'has pending proposals' into a full scan per image."""
    ds_id, _ = ds
    db.execute("ANALYZE ai_preds")
    db.execute("INSERT OR REPLACE INTO sqlite_stat1(tbl, idx, stat) VALUES('ai_preds', 'idx_ai_preds_img', '3 3 3')")
    run_id = store.create_run(ds_id, "batch", "mock", "m", {})
    for iid in _ids(ds_id):
        store.insert_preds(run_id, ds_id, iid, [{"cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.1, "conf": 0.5}] * 20)
    assert store.refresh_planner_stats(throttle=False)
    stat = db.query_one("SELECT stat FROM sqlite_stat1 WHERE tbl='ai_preds' AND idx='idx_ai_preds_img'")["stat"]
    assert int(stat.split()[0]) >= 240
    plan = " ".join(r["detail"] for r in db.query(
        "EXPLAIN QUERY PLAN SELECT i.id FROM images i WHERE i.dataset_id=? AND EXISTS (SELECT 1 FROM ai_preds p "
        "WHERE p.image_id=i.id AND p.state='pending' AND p.kind IN ('detect'))", (ds_id,)))
    assert "idx_ai_preds_img" in plan
    assert not store.refresh_planner_stats(throttle=False)      # up to date now: nothing to do


def test_without_ultralytics_local_providers_degrade(client, ds, monkeypatch):
    from fovea.ai import local
    monkeypatch.setitem(local._spec_cache, "ultralytics", False)
    j = client.get("/api/ai/providers").json()
    st = {p["id"]: p["status"] for p in j["providers"]}
    assert st["yolo"]["available"] is False and "ultralytics" in st["yolo"]["reason"]
    assert j["device"]["available"] is False
    ds_id, _ = ds
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": _ids(ds_id)[0], "provider": "yolo",
                                              "model": "yolo12m"})
    assert r.status_code == 503 and r.json()["code"] == "unavailable"
    from fovea.core import yolo
    assert yolo.available() is False


def test_tiles_and_task_helpers():
    from fovea.ai.local import _cut_by_tile
    assert weights.task_of("yolo11n-cls") == "classify" and weights.task_of("yolo26s-obb") == "obb"
    assert weights.task_of("yolo12m") == "detect" and weights.family_of("yolov8s-worldv2.pt") == "world"
    wins = geometry.tiles(1000, 800, 640, 0.2)
    assert wins[0] == (0, 0, 640, 640) and wins[-1][2:] == (1000, 800)
    assert geometry.tiles(320, 240) == [(0, 0, 320, 240)]
    W, H = 1000, 800
    inner_cut = {"xc": 630 / W, "yc": 0.3, "w": 40 / W, "h": 0.1}      # touches the tile's right edge at x=640
    assert _cut_by_tile(inner_cut, (0, 0, 640, 640), W, H)
    at_image_edge = {"xc": 20 / W, "yc": 0.3, "w": 40 / W, "h": 0.1}   # touches x=0: the image border, not a cut
    assert not _cut_by_tile(at_image_edge, (0, 0, 640, 640), W, H)
    big = {"cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.4, "h": 0.4, "conf": 0.9}
    part = {"cls": 0, "xc": 0.4, "yc": 0.5, "w": 0.15, "h": 0.3, "conf": 0.8}
    assert len(geometry.nms([big, part], 0.5)) == 2 and geometry.nms([big, part], 0.5, ios_thr=0.7) == [big]


# ================================================================== wave 2a: generate_json, Economy batches, naming,
# natural-language search, chunked local inference, provider waits, raw audit detections, ref_box

def test_provider_contract_defaults_and_mock_flags(client):
    from fovea.ai.base import DetectRequest, Provider, ProviderError
    p = Provider()
    for call in (lambda: p.generate_json("s", "u", {"type": "object"}, "m"), lambda: p.batch_submit([]),
                 lambda: p.batch_status("x"), lambda: list(p.batch_results("x")), lambda: p.batch_cancel("x")):
        with pytest.raises(ProviderError) as e:
            call()
        assert e.value.code == "invalid"
    # detect_many default: one detect per request, errors in place
    class Two(Provider):
        def detect(self, req):
            if req.image_path == "bad":
                raise ProviderError("nope", "bad_output")
            from fovea.ai.base import DetectResult
            return DetectResult([])
    out = Two().detect_many([DetectRequest("ok", [], "m"), DetectRequest("bad", [], "m")])
    assert not isinstance(out[0], Exception) and isinstance(out[1], ProviderError)
    caps = {x["id"]: x["capabilities"] for x in client.get("/api/ai/providers").json()["providers"]}
    assert caps["mock"]["generate"] and caps["mock"]["batch_api"]


def _economy(client, ds_id, monkeypatch, delay=0.3, **kw):
    from fovea.ai import economy
    monkeypatch.setattr(economy, "POLL_S", 0.1)
    opts = {"mock_batch_delay_s": delay, **kw.pop("options", {})}
    body = {"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "target": "proposals", "speed": "economy",
            "options": opts, **kw}
    if "image_ids" not in body and "filters" not in body:
        body["filters"] = {}
    r = client.post("/api/ai/batch", json=body)
    return r


def test_economy_batch_proposals_at_half_price(client, ds, monkeypatch):
    ds_id, root = ds
    before = [_label_text(root, k) for k in range(12)]
    r = _economy(client, ds_id, monkeypatch)
    assert r.status_code == 200, r.text
    assert r.json()["speed"] == "economy" and r.json()["job"]["kind"] == "ai_batch"
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.pool == "ai_wait"                                 # a day-long wait never holds an AI worker
    assert [_label_text(root, k) for k in range(12)] == before
    run = client.get(f"/api/ai/runs/{r.json()['run_id']}").json()["run"]
    assert run["status"] == "done" and run["params"]["speed"] == "economy"
    assert run["params"]["batch_id"] and run["params"]["batch_id"].startswith("mockbatch_")
    assert run["params"]["provider_batches"][0]["ingested"] is True
    assert run["result"]["images_done"] == 12 and run["result"]["batch"]["state"] == "ended"
    assert set(run["result"]["batch"]["counts"]) == {"processing", "succeeded", "errored", "canceled", "expired"}
    assert run["cost_usd"] == pytest.approx(12 * 0.0005, abs=1e-6)       # batch price = half the synchronous one
    n = db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE run_id=? AND state='pending'", (run["id"],))["n"]
    assert n == run["result"]["boxes"] > 0
    assert db.kv_get(f"ai.economy.{run['id']}") is None                  # bookkeeping cleaned up when done
    est = client.post("/api/ai/estimate", json={"dataset_id": ds_id, "filters": {}, "provider": "mock",
                                                 "model": "mock-fast"}).json()
    assert est["usd_batch"][1] < est["usd"][1] and est["economy"]["available"] is True


def test_economy_write_target_snapshots_and_announces(client, ds, monkeypatch):
    from fovea.core import labels as L
    seen = []
    monkeypatch.setattr(L, "LABEL_WRITE_HOOKS", [lambda d, ids: seen.append((d, list(ids)))])
    ds_id, root = ds
    ids = _ids(ds_id)[:3]
    r = _economy(client, ds_id, monkeypatch, image_ids=ids, target="replace")
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.result["written"] == 3 and job.result.get("snapshot_id")
    assert sorted(i for _d, x in seen for i in x) == sorted(ids) and all(d == ds_id for d, _x in seen)


def test_economy_validation(client, ds, monkeypatch):
    ds_id, _ = ds
    two = {"pipeline": "two_stage", "detector": {"provider": "mock", "model": "mock-fast"},
           "classifier": {"provider": "mock", "model": "mock-fast"}}
    r = _economy(client, ds_id, monkeypatch, options=two)
    assert r.status_code == 422 and "two-stage" in r.json()["detail"].lower()
    r = _economy(client, ds_id, monkeypatch, budget_usd=0.0001)
    assert r.status_code == 402 and r.json()["code"] == "quota"
    r = _economy(client, ds_id, monkeypatch, speed="tomorrow")
    assert r.status_code == 422
    from fovea.ai.providers import mock
    monkeypatch.setattr(mock.MockProvider, "capabilities", {**mock.MockProvider.capabilities, "batch_api": False})
    r = _economy(client, ds_id, monkeypatch)
    assert r.status_code == 422 and "batch api" in r.json()["detail"].lower()


def test_economy_budget_limits_what_is_submitted(client, ds, monkeypatch):
    ds_id, _ = ds
    r = _economy(client, ds_id, monkeypatch, budget_usd=0.0025)          # ~0.0006 per image at the batch price
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.result["stopped"] == "budget" and 0 < job.result["submitted"] < 12


def test_economy_cancel_cancels_the_provider_batch(client, ds, monkeypatch):
    ds_id, _ = ds
    r = _economy(client, ds_id, monkeypatch, delay=60)
    run_id = r.json()["run_id"]
    t0 = time.time()
    while not (store.get_run(run_id)["params"].get("provider_batches")) and time.time() - t0 < 10:
        time.sleep(0.05)
    bid = store.get_run(run_id)["params"]["batch_id"]
    c = client.post(f"/api/ai/runs/{run_id}/cancel", json={})
    assert c.status_code == 200, c.text
    assert c.json()["run"]["status"] == "cancelled"
    from fovea.ai.providers.mock import PROVIDER
    assert PROVIDER.batch_status(bid)["counts"]["canceled"] == 12      # the provider was told to stop
    assert _wait(r.json()["job"]["id"]).status == "cancelled"
    assert client.post("/api/ai/runs/999999/cancel", json={}).status_code == 404


def test_economy_resume_after_restart(client, ds, monkeypatch):
    """A run whose job died with the server (provider batch still out) is picked up by the startup scan."""
    from fovea.ai import economy, pipeline as P
    from fovea.ai.providers.mock import PROVIDER
    monkeypatch.setattr(economy, "POLL_S", 0.1)
    ds_id, _ = ds
    ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    ids = _ids(ds_id)[:5]
    plan = P.plan_detect(ds_row, "mock", "mock-fast", None, None, {"mock_batch_delay_s": 0.2})
    imgs = {r["id"]: r for r in db.query(f"SELECT * FROM images WHERE id IN ({','.join('?' * len(ids))})", ids)}
    bid = PROVIDER.batch_submit([{"custom_id": str(i), "request": plan.request(imgs[i])} for i in ids])
    run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {
        "provider": "mock", "model": "mock-fast", "target": "proposals", "options": {"mock_batch_delay_s": 0.2},
        "speed": "economy", "images": 5, "batch_id": bid,
        "provider_batches": [{"id": bid, "n": 5, "state": "running", "counts": {}, "ingested": False,
                              "submitted_at": time.time()}]})
    db.execute("UPDATE ai_runs SET created_at=?, job_id='gone' WHERE id=?", (economy.STARTED_AT - 60, run_id))
    db.kv_set(economy.kv_key(run_id), {"image_ids": ids, "batches": {bid: ids}})
    assert client.get(f"/api/ai/runs/{run_id}").json()["run"]["status"] == "waiting"
    monkeypatch.setattr(economy, "_resumed", {})
    assert run_id in economy.resume()
    assert economy.resume() == []                                         # once per process and database
    job_id = db.query_one("SELECT job_id FROM ai_runs WHERE id=?", (run_id,))["job_id"]
    job = _wait(job_id)
    assert job.status == "done", job.error
    run = store.get_run(run_id)
    assert run["status"] == "done" and run["images"] == 5 and run["cost_usd"] == pytest.approx(5 * 0.0005, abs=1e-6)
    # a run created by this process is never resumed behind its own job's back
    mine = store.create_run(ds_id, "batch", "mock", "mock-fast", {"speed": "economy", "provider_batches": [{"id": bid}]})
    monkeypatch.setattr(economy, "_resumed", {})
    assert mine not in economy.resume()
    store.finish_run(mine, "done")


def test_name_classes_with_mock(client, ds):
    ds_id, _ = ds
    db.execute("UPDATE datasets SET classes=? WHERE id=?", (db.dumps(["class_0", "class_1", "class_2"]), ds_id))
    r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                   "per_class": 4, "dry_run": True})
    assert r.status_code == 200, r.text
    est = r.json()["estimate"]
    assert est["requests"] == 1 and est["images"] == 12 and est["usd"][1] > 0       # 3 classes x 4 crops fit one call
    assert not db.query("SELECT 1 FROM ai_runs WHERE dataset_id=? AND kind='name_classes'", (ds_id,))
    r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                   "per_class": 4})
    assert r.status_code == 200, r.text
    j = r.json()
    items = {it["cls"]: it for it in j["items"]}
    assert set(items) == {0, 1, 2}
    names = [it["name"] for it in items.values()]
    assert all(re.fullmatch(r"[a-z][a-z0-9_]*", n) for n in names) and len(set(names)) == 3
    assert all(it["prompt"] and it["description"] and it["confidence"] is not None for it in items.values())
    assert all(1 <= len(it["examples"]) <= 4 and it["examples"][0]["image_id"] for it in items.values())
    assert j["usage"]["cost_usd"] > 0 and j["usage"]["input_tokens"] > 12 * 200           # the crops were sent
    run = db.query_one("SELECT * FROM ai_runs WHERE dataset_id=? AND kind='name_classes'", (ds_id,))
    assert run and run["cost_usd"] > 0
    # one request per class when the crops do not fit one call (at most 20 images per request)
    r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                   "per_class": 8, "classes": [0, 1, 2], "dry_run": True}).json()
    n_crops = sum(len(it["examples"]) for it in r["items"])
    assert r["estimate"]["images"] == n_crops and r["estimate"]["requests"] == (1 if n_crops <= 20 else 3)
    from fovea.ai import assist
    import fovea.ai.assist as A
    old = A.MAX_IMAGES_PER_REQUEST
    A.MAX_IMAGES_PER_REQUEST = 5
    try:
        r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                       "per_class": 4, "dry_run": True}).json()
        assert r["estimate"]["requests"] == 3 and assist.MAX_IMAGES_PER_REQUEST == 5
    finally:
        A.MAX_IMAGES_PER_REQUEST = old
    assert client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock",
                                                      "per_class": 3}).status_code == 422
    assert client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock",
                                                      "classes": [7]}).status_code == 422
    r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "yolo"})
    assert r.status_code == 400 and r.json()["code"] == "unavailable"     # local detectors cannot write text


def test_crop_marks_the_object(ds):
    from fovea.ai.assist import crop_jpeg
    import io as _io
    ds_id, root = ds
    path = str(root / "images" / "train" / "img_000.jpg")
    data = crop_jpeg(path, (0.2, 0.5, 0.15, 0.3), edge=256)
    im = Image.open(_io.BytesIO(data))
    assert max(im.size) == 256
    px = [im.getpixel((x, im.height // 2)) for x in range(im.width)]
    assert any(p[0] > 200 and p[1] < 90 for p in px)                       # red outline present


def test_nl_filter_english_korean_and_validation(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/nl-filter", json={"dataset_id": ds_id, "provider": "mock",
                                                "text": "fall images in the train split with at least 2 boxes"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["filters"]["cls"] == [1] and j["filters"]["split"] == ["train"] and j["filters"]["min_boxes"] == 2
    assert j["explanation"] and j["usage"]["cost_usd"] > 0 and j["cached"] is False
    again = client.post("/api/ai/nl-filter", json={"dataset_id": ds_id, "provider": "mock",
                                                    "text": "fall images in the train split with at least 2 boxes"}).json()
    assert again["cached"] is True and again["usage"]["cost_usd"] == 0 and again["filters"] == j["filters"]
    k = client.post("/api/ai/nl-filter", json={"dataset_id": ds_id, "provider": "mock",
                                                "text": "넘어진 사람이 있는 흐린 이미지, 검증 세트"}).json()
    assert k["filters"]["cls"] == [1] and k["filters"]["issue"] == ["blurry"]
    assert "split" not in k["filters"]                                      # no val split here: dropped, not invented
    from fovea.api.common import FILTER_KEYS
    assert set(k["filters"]) <= set(FILTER_KEYS)
    runs = db.query("SELECT boxes FROM ai_runs WHERE dataset_id=? AND kind='nl_filter'", (ds_id,))
    assert len(runs) == 1 and runs[0]["boxes"] == 2                         # one ledger row per day, two questions
    assert client.post("/api/ai/nl-filter", json={"dataset_id": ds_id, "provider": "mock", "text": " "}).status_code == 422
    r = client.post("/api/ai/nl-filter", json={"dataset_id": ds_id, "provider": "sam", "text": "fallen"})
    assert r.status_code == 400 and r.json()["code"] == "unavailable"


def test_nl_filter_validation_drops_unknown_values(ds):
    from fovea.ai import assist
    ds_id, _ = ds
    ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    ctx = assist.nl_context(ds_row)
    f, warn = assist.validate_filters(ds_row, ctx, {
        "foo": "x", "ids": [1, 2], "cls": ["9", "fall", "bike", "nope"], "split": ["train", "nope"],
        "labeled": "bogus", "review": "approved", "issue": ["blurry", "evil"], "min_boxes": -1, "max_boxes": 3,
        "seq": "no-such-seq", "ai": "audit"})
    assert f == {"cls": [1, 2], "split": ["train"], "review": "approved", "issue": ["blurry"], "max_boxes": 3,
                 "q": "no-such-seq", "ai": "audit"}
    assert any("foo" in w for w in warn) and any("nope" in w for w in warn)
    system, user, schema = assist.nl_prompts(ds_row, ctx, "x")
    props = schema["properties"]["filters"]["properties"]
    assert props["cls"]["items"]["enum"] == ["0", "1", "2"] and "blurry" in props["issue"]["items"]["enum"]
    assert set(schema["properties"]["filters"]["required"]) == set(props)    # strict-schema friendly
    assert "class 1: fall" in system


def test_chunked_local_inference_in_batch_and_audit(client, ds, monkeypatch):
    """Local batch-capable providers get several images per model call (C3)."""
    from fovea.ai.providers import mock
    calls = []
    orig = mock.MockProvider.detect_many

    def many(self, reqs):
        calls.append(len(reqs))
        return orig(self, reqs)
    monkeypatch.setattr(mock.MockProvider, "detect_many", many, raising=False)
    monkeypatch.setattr(mock.MockProvider, "capabilities", {**mock.MockProvider.capabilities, "batch_infer": True})
    monkeypatch.setattr(mock.PROVIDER, "kind", "local", raising=False)
    ds_id, _ = ds
    res, job = _batch(client, ds_id, options={"infer_batch": 4})
    assert job.status == "done", job.error
    assert calls and max(calls) == 4 and sum(calls) == 12 and job.extra["images_done"] == 12
    assert db.query_one("SELECT params FROM ai_runs WHERE id=?", (res["run_id"],))["params"].count('"infer_batch":4')
    calls.clear()
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "mode": "full", "options": {"infer_batch": 5}})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert calls and max(calls) == 5 and job.result["images_done"] == 11
    assert client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                               "options": {"infer_batch": 99}}).status_code == 422


def test_batch_status_shows_provider_waits(client, ds, monkeypatch):
    from fovea.ai import vlm
    from fovea.ai.providers import mock
    from fovea.jobs import jobs
    orig = mock.MockProvider.detect
    hit = {"n": 0}

    def slow(self, req):
        hit["n"] += 1
        if hit["n"] == 2:
            vlm._wait(2.5, "Mock rate limited")
        return orig(self, req)
    monkeypatch.setattr(mock.MockProvider, "detect", slow)
    ds_id, _ = ds
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "concurrency": 1})
    seen = set()
    t0 = time.time()
    while time.time() - t0 < 20:
        j = jobs.get(r.json()["job"]["id"])
        seen.add(j.message)
        if j.status in ("done", "error", "cancelled"):
            break
        time.sleep(0.05)
    assert any(re.search(r"Mock rate limited — retrying in \d+ s", m or "") for m in seen), seen
    assert j.status == "done"


def test_job_waits_text_and_cancel_aborts_wait():
    from fovea.ai.runner import JobWaits
    from fovea.jobs import Job, JobCancelled
    job = Job(id="w", kind="ai_batch")
    w = JobWaits(job)
    w.listener(12, "anthropic rate limited")
    assert w.text() == "Anthropic rate limited — retrying in 12 s"
    w.listener(0, "anthropic rate limited")
    assert w.text() is None
    job.cancel()
    with pytest.raises(JobCancelled):
        w.listener(5, "x")


def test_audit_keeps_raw_detections_and_ref_box(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "mode": "full"})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    run_id = r.json()["run_id"]
    raw = db.query("SELECT * FROM ai_preds WHERE run_id=? AND state='info'", (run_id,))
    assert raw and all(p["kind"] == "detect" for p in raw) and job.result["raw_detections"] == len(raw)
    assert job.result["image_ranges"] and store.get_run(run_id)["params"]["raw_detections"] == "dataset"
    cols = {c["name"] for c in db.query("PRAGMA table_info(ai_preds)")}
    assert "ref_box" in cols
    refs = db.query("SELECT kind, ref_row, ref_box FROM ai_preds WHERE run_id=? AND state='pending' AND kind IN "
                    "('loose','swap','spurious')", (run_id,))
    assert refs and all(x["ref_box"] and len(json.loads(x["ref_box"])) == 4 for x in refs)
    some = db.query_one("SELECT image_id FROM ai_preds WHERE run_id=? AND kind='loose' AND state='pending'", (run_id,))
    if some:
        props = client.get(f"/api/datasets/{ds_id}/images/{some['image_id']}/proposals").json()["preds"]
        loose = [p for p in props if p["kind"] == "loose"]
        assert loose and isinstance(loose[0]["ref_box"], list) and len(loose[0]["ref_box"]) == 4
    # the single-image review does not pile up raw detections
    iid = _ids(ds_id)[0]
    rv = client.post("/api/ai/review", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert rv.status_code == 200 and not db.query("SELECT 1 FROM ai_preds WHERE run_id=? AND state='info'",
                                                  (rv.json()["run_id"],))
    # deleting the run drops its raw detections too
    assert client.delete(f"/api/ai/runs/{run_id}").status_code == 200
    assert not db.query("SELECT 1 FROM ai_preds WHERE run_id=? AND state='info'", (run_id,))


def test_geometry_audit_raw_detections_use_family_default(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast", "filters": {},
                                            "mode": "geometry", "class_family": {"person": [0, 1]}})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    raw = db.query("SELECT cls, label FROM ai_preds WHERE run_id=? AND state='info'", (r.json()["run_id"],))
    assert raw and {p["cls"] for p in raw} <= {0, 1} and all(p["label"] for p in raw)
    assert store.get_run(r.json()["run_id"])["params"]["raw_detections"] == "family"


def test_accept_uses_ref_box_to_detect_stale_findings(client, ds):
    ds_id, root = ds
    ids = _ids(ds_id)
    run_id = store.create_run(ds_id, "audit", "mock", "m", {})
    # image 0 has one GT row (cls 0 at 0.2,0.5,0.15,0.3); a loose finding computed for exactly that box
    store.insert_preds(run_id, ds_id, ids[0], [{"cls": 0, "xc": 0.21, "yc": 0.5, "w": 0.12, "h": 0.28, "conf": 0.9,
                                                "kind": "loose", "ref_row": 0, "ref_box": [0.2, 0.5, 0.15, 0.3]}])
    # image 3 (one GT row as well): the finding refers to a box that is no longer there (edited since)
    store.insert_preds(run_id, ds_id, ids[3], [{"cls": 0, "xc": 0.21, "yc": 0.5, "w": 0.12, "h": 0.28, "conf": 0.9,
                                                "kind": "loose", "ref_row": 0, "ref_box": [0.6, 0.6, 0.1, 0.1]}])
    from fovea.core import labels as L
    seen = []
    L.LABEL_WRITE_HOOKS.append(lambda d, x: seen.append(list(x)))
    try:
        r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [ids[0], ids[3]], "kinds": ["loose"]})
        job = _wait(r.json()["job"]["id"])
    finally:
        L.LABEL_WRITE_HOOKS.pop()
    assert job.status == "done", job.error
    # the box the second finding was about is gone: the finding no longer applies (answered, not pending for good)
    assert job.result["modified"] == 1 and job.result["obsolete"] == 1 and job.result["stale"] == 0
    gone = db.query_one("SELECT state, decided_at, reason FROM ai_preds WHERE image_id=? AND kind='loose'", (ids[3],))
    assert gone["state"] == "rejected" and gone["decided_at"] is None and "since the audit" in gone["reason"]
    assert seen == [[ids[0]]]
    rows = [ln.split() for ln in _label_text(root, 0).splitlines()]
    assert float(rows[0][3]) == pytest.approx(0.12, abs=1e-4)


def test_store_ensure_schema_adds_ref_box_to_old_tables(tmp_path, monkeypatch):
    import sqlite3
    from fovea.ai import store as S
    p = tmp_path / "old.db"
    conn = sqlite3.connect(str(p), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("CREATE TABLE ai_preds (id INTEGER PRIMARY KEY, run_id INTEGER, image_id INTEGER, ref_row INTEGER)")
    monkeypatch.setattr(db, "query", lambda sql, params=(): [dict(r) for r in conn.execute(sql, tuple(params)).fetchall()])
    monkeypatch.setattr(db, "execute", lambda sql, params=(): conn.execute(sql, tuple(params)))
    monkeypatch.setattr(db, "DB_PATH", p)
    S.ensure_schema()
    S.ensure_schema()                                                      # idempotent
    assert "ref_box" in {r["name"] for r in conn.execute("PRAGMA table_info(ai_preds)")}
    assert S.ref_box_json([0.1, 0.2, 0.3, 0.4]) == "[0.1,0.2,0.3,0.4]" and S.ref_box_json("x") is None


# ------------------------------------------------------------------ review round 1 (r1) regressions

def test_no_resume_thread_at_import_and_ai_wait_pool_declared(client):
    import threading as th
    from fovea.ai import economy
    from fovea.jobs import JobManager, jobs
    assert JobManager.POOLS.get("ai_wait") and "ai_wait" in jobs._pools            # noqa: SLF001
    j = jobs.submit("t", lambda job: job.pool, pool="ai_wait")
    assert _wait(j.id).result == "ai_wait"
    assert not [t for t in th.enumerate() if t.name == "fovea-ai-resume" and t.is_alive()]
    calls = []
    orig = economy.resume
    economy.resume = lambda force=False: calls.append(1) or []
    try:
        import asyncio
        from fovea import main

        async def startup_only():                 # the app's startup half (its shutdown would stop the job pools)
            await main.lifespan(main.app).__aenter__()
        asyncio.run(startup_only())               # the app's startup re-attaches Economy runs, nothing else does
        t0 = time.time()
        while not calls and time.time() - t0 < 5:
            time.sleep(0.02)
    finally:
        economy.resume = orig
    assert calls == [1]


def test_registry_never_freezes_model_availability(client, monkeypatch):
    p = registry.get("mock")
    state = {"have": False}
    monkeypatch.setattr(type(p), "merge_models", lambda self, live: [], raising=False)
    monkeypatch.setattr(type(p), "models", lambda self, refresh=False: [{"id": "m1", "available": state["have"]}])
    assert registry.models("mock", refresh=True)[0]["available"] is False
    state["have"] = True                          # the model was pulled / the server came up
    assert registry.models("mock")[0]["available"] is True
    assert [m for m in client.get("/api/ai/providers").json()["providers"] if m["id"] == "mock"][0]["models"][0]["available"]


def test_accept_never_writes_proposals_rejected_while_it_was_queued(client, ds):
    from fovea.jobs import jobs
    ds_id, root = ds
    iid = _ids(ds_id)[10]                          # no label file yet
    run_id = store.create_run(ds_id, "propose", "mock", "m", {})
    preds = store.insert_preds(run_id, ds_id, iid, [{"cls": 0, "xc": 0.3, "yc": 0.3, "w": 0.1, "h": 0.1, "conf": 0.9},
                                                    {"cls": 1, "xc": 0.7, "yc": 0.7, "w": 0.1, "h": 0.1, "conf": 0.9}])
    gate = threading.Event()
    blockers = [jobs.submit("blocker", lambda j: gate.wait(20)) for _ in range(jobs._pools["default"]._max_workers)]  # noqa: SLF001
    try:
        r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid]})
        assert jobs.get(r.json()["job"]["id"]).status == "queued"
        assert client.post(f"/api/datasets/{ds_id}/proposals/decide",
                           json={"reject": [preds[0]["id"]]}).json()["updated"] == 1
    finally:
        gate.set()
    job = _wait(r.json()["job"]["id"])
    for b in blockers:
        _wait(b.id)
    assert job.status == "done" and job.result["added"] == 1 and job.result["decided_meanwhile"] == 1
    rows = [ln.split() for ln in _label_text(root, 10).splitlines()]
    assert len(rows) == 1 and rows[0][0] == "1"
    states = {p["id"]: p["state"] for p in db.query("SELECT id, state FROM ai_preds WHERE run_id=?", (run_id,))}
    assert states == {preds[0]["id"]: "rejected", preds[1]["id"]: "accepted"}


def test_accept_loose_finding_never_flattens_a_polygon(client, ds):
    ds_id, root = ds
    iid = _ids(ds_id)[0]
    lp = root / "labels" / "train" / "img_000.txt"
    poly = "0 0.40 0.40 0.60 0.38 0.62 0.60 0.45 0.63 0.38 0.50\n"
    lp.write_text(poly)
    from fovea.core.scanner import refresh_image
    refresh_image(iid, 3)
    run_id = store.create_run(ds_id, "audit", "mock", "m", {})
    box = [0.5, 0.505, 0.24, 0.25]
    store.insert_preds(run_id, ds_id, iid, [{"cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.2, "h": 0.2, "conf": 0.9, "kind": "loose",
                                             "ref_row": 0, "ref_box": box},
                                            {"cls": 2, "xc": 0.5, "yc": 0.505, "w": 0.24, "h": 0.25, "conf": 0.9,
                                             "kind": "swap", "ref_row": 0, "ref_box": box}])
    r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid], "kinds": ["loose", "swap"]})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.result["extended"] == 1 and job.result["modified"] == 1
    assert lp.read_text().split()[1:] == poly.split()[1:] and lp.read_text().split()[0] == "2"   # class changed, shape kept
    left = db.query("SELECT kind, state FROM ai_preds WHERE run_id=? ORDER BY id", (run_id,))
    assert [(x["kind"], x["state"]) for x in left] == [("loose", "pending"), ("swap", "accepted")]


@pytest.mark.skipif(hasattr(os, "geteuid") and os.geteuid() == 0, reason="root ignores directory permissions")
def test_batch_unwritable_label_files_end_the_run_cleanly(client, tmp_path):
    ds_id, root = _make_dataset(tmp_path, n=14, unlabeled=tuple(range(14)))
    ro = root / "labels" / "train"
    os.chmod(ro, 0o555)
    try:
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                                "target": "fill", "concurrency": 1})
        job = _wait(r.json()["job"]["id"])
    finally:
        os.chmod(ro, 0o755)
    run = client.get(f"/api/ai/runs/{r.json()['run_id']}").json()["run"]
    # every file fails: after MAX_CONSECUTIVE_ERRORS in a row the run ends as an error — recorded, never "interrupted"
    assert job.status == "error" and "in a row" in job.error
    assert run["status"] == "error" and "Permission denied" in run["error"] and str(ro) in run["error"]
    assert ".fovea-" not in run["error"] and run["result"]["write_errors"] == 10
    assert run["images"] == 10 and run["cost_usd"] == pytest.approx(0.010, abs=1e-6)   # billed images are booked
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_batch_unwritable_split_does_not_stop_other_splits(client, tmp_path):
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        pytest.skip("root ignores directory permissions")
    root = tmp_path / "two"
    for split in ("train", "val"):
        for k in range(3):
            _scene(root / "images" / split / f"f{k}.jpg", (160, 120), [(0.5, 0.5, 0.2, 0.3)])
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    r = client.post("/api/datasets", json={"path": str(root), "id": "aic-two-" + uuid.uuid4().hex[:6]})
    ds_id = r.json()["dataset"]["id"]
    from fovea.jobs import jobs
    for j in jobs.list(ds_id):
        _wait(j.id)
    db.execute("UPDATE datasets SET classes=?, classes_source='user' WHERE id=?", (db.dumps(CLASSES), ds_id))
    os.chmod(root / "labels" / "train", 0o555)
    try:
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                                "target": "fill"})
        job = _wait(r.json()["job"]["id"])
    finally:
        os.chmod(root / "labels" / "train", 0o755)
    assert job.status == "done", job.error
    assert job.result["write_errors"] == 3 and job.result["errors"] == 3 and job.result["written"] == 3
    assert sorted(p.name for p in (root / "labels" / "val").glob("*.txt")) == ["f0.txt", "f1.txt", "f2.txt"]
    run = client.get(f"/api/ai/runs/{r.json()['run_id']}").json()["run"]
    assert run["status"] == "done" and run["images"] == 6
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_economy_cancel_while_reading_results_books_each_image_once(client, ds, monkeypatch):
    from fovea.ai.providers import mock as mock_mod
    from fovea.jobs import jobs
    ds_id, _ = ds
    cls = type(registry.get("mock"))
    orig = cls.batch_results
    state = {"job": None, "first": True}

    def patched(self, batch_id, items=None):
        for n, x in enumerate(orig(self, batch_id, items)):
            if state["first"] and n == 5:
                state["first"] = False
                state["job"].cancel()             # Cancel pressed while the results are being read
            yield x
    monkeypatch.setattr(cls, "batch_results", patched)
    from fovea.ai import economy
    real_update = economy._Run.ingest

    def ingest(self, b, quiet=False):             # report progress every result so the cancel lands mid-ingest
        self.job.update(message="reading")
        return real_update(self, b, quiet)
    monkeypatch.setattr(economy._Run, "ingest", ingest)
    r = _economy(client, ds_id, monkeypatch, delay=0)
    state["job"] = jobs.get(r.json()["job"]["id"])
    _wait(r.json()["job"]["id"])
    run = store.get_run(r.json()["run_id"])
    n_imgs = db.query_one("SELECT COUNT(DISTINCT image_id) n FROM ai_preds WHERE run_id=?", (run["id"],))["n"]
    assert run["images"] == n_imgs == 12
    assert run["cost_usd"] == pytest.approx(12 * mock_mod.DETECT_USD * mock_mod.BATCH_DISCOUNT, abs=1e-6)


def test_economy_fatal_poll_error_keeps_waiting_for_the_submitted_batch(client, ds, monkeypatch):
    from fovea.ai.base import ProviderError
    ds_id, _ = ds
    cls = type(registry.get("mock"))
    orig = cls.batch_status
    fails = {"n": 2}

    def flaky(self, batch_id):
        if fails["n"] > 0:
            fails["n"] -= 1
            raise ProviderError("Mock rejected the API key (401)", "auth", status=401)
        return orig(self, batch_id)
    monkeypatch.setattr(cls, "batch_status", flaky)
    from fovea.ai import economy
    monkeypatch.setattr(economy, "POLL_S", 0.05)
    r = _economy(client, ds_id, monkeypatch, delay=0.2)
    job = _wait(r.json()["job"]["id"], timeout=60)
    run = store.get_run(r.json()["run_id"])
    assert job.status == "done", job.error
    assert run["status"] == "done" and run["images"] == 12 and fails["n"] == 0
    # polling recovered: the failed checks were a wait, not an error — no stale "Last error" under "Done"
    assert not run["result"].get("last_error") and not (job.extra or {}).get("last_error")


def test_ai_inputs_of_the_wrong_type_are_422_not_500(client, ds):
    from fastapi.testclient import TestClient
    from fovea.main import app
    c = TestClient(app, raise_server_exceptions=False)
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    base = {"dataset_id": ds_id, "image_id": iid, "image_ids": [iid], "provider": "mock", "model": "mock-fast"}
    cases = [(f"/api/datasets/{ds_id}/proposals/decide", {"reject": ["²"]}),
             (f"/api/datasets/{ds_id}/proposals/decide", {"accept": [1e309]}),
             ("/api/ai/propose", {**base, "model": 5}), ("/api/ai/propose", {**base, "provider": ["mock"]}),
             ("/api/ai/propose", {**base, "image_id": "١"}),
             ("/api/ai/refine", {**base, "points": 5}), ("/api/ai/refine", {**base, "box": 5}),
             ("/api/ai/review", {**base, "committee": [{"provider": "mock", "model": "mock-fast", "options": 5}]}),
             ("/api/ai/benchmark", {**base, "configs": [{"provider": "mock", "options": "x"}]}),
             ("/api/ai/classify", {**base, "boxes": [{}]}), ("/api/ai/warm", {"provider": "yolo", "model": 5}),
             ("/api/ai/propose", {**base, "options": {"pipeline": "two_stage", "detector": [], "classifier": {}}}),
             ("/api/ai/audit", {**base, "options": {"pipeline": "two_stage", "detector": {"provider": 5}}})]
    for path, body in cases:
        r = c.post(path, content=json.dumps(body, allow_nan=True), headers={"content-type": "application/json"})
        assert r.status_code in (400, 404, 422), (path, body, r.status_code, r.text)


def test_remote_clients_cannot_repoint_providers_or_bypass_the_guard(client, ds, monkeypatch):
    ds_id, _ = ds
    from fovea.api import ai as ai_api
    monkeypatch.setattr(ai_api, "_is_local", lambda req: False)
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    iid = _ids(ds_id)[0]
    # openai_compat counts as local while it points at localhost: re-pointing it would ship its key elsewhere
    for body in ({"base_url": "http://attacker.example/v1"}, {"base_url": "http://127.0.0.1:9/v1"}, {"spend_cap_usd": 99}):
        assert client.put("/api/ai/providers/openai_compat/config", json=body).status_code == 403
    assert client.put("/api/ai/providers/yolo/config", json={"enabled": True}).status_code == 200
    two = {"pipeline": "two_stage", "detector": {"provider": "yolo", "model": "yolo12m"},
           "classifier": {"provider": "mock", "model": "mock-fast"}}
    r = client.post("/api/ai/review", json={"dataset_id": ds_id, "image_id": iid, "provider": "yolo", "model": "yolo12m",
                                            "options": two})
    assert r.status_code == 403
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "yolo", "model": "yolo12m",
                                           "committee": [{"provider": "yolo", "model": "yolo12m", "options": two}]})
    assert r.status_code == 403
    # a live Economy run: DELETE is refused like Cancel (it would cancel the provider batch with the owner's key)
    run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {"speed": "economy", "provider_batches": [{"id": "b"}]})
    assert client.post(f"/api/ai/runs/{run_id}/cancel").status_code == 403
    assert client.delete(f"/api/ai/runs/{run_id}").status_code == 403
    store.finish_run(run_id, "done")


def test_spend_cap_counts_two_stage_classifier_spend(client, ds, monkeypatch, tmp_path):
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    db.execute("DELETE FROM ai_runs")
    monkeypatch.setattr(ai_config, "spend_cap", lambda pid: 0.0001 if pid == "mock" else None)
    from fovea.ai import runner
    two = {"pipeline": "two_stage", "detector": {"provider": "mock", "model": "mock-fast"},
           "classifier": {"provider": "mock", "model": "mock-fast"}}
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "options": two})
    assert r.status_code == 200
    run = store.get_run(r.json()["run_id"])
    assert run["provider"] == "mock+mock" or "+" in run["provider"]
    assert runner.spent_30d("mock") == pytest.approx(run["cost_usd"], abs=1e-9) and run["cost_usd"] > 0
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "options": two})
    assert r.status_code == 402
    u = client.get("/api/ai/usage").json()
    assert set(u["by_provider"]) == {"mock"} and u["caps"]["mock"]["spent_30d"] == pytest.approx(run["cost_usd"], abs=1e-6)
    # usage attributed per provider: detector and classifier shares sum to the run's total
    parts = store.usage_parts([run["id"]])[run["id"]]
    assert sum(p["cost_usd"] for p in parts) == pytest.approx(run["cost_usd"], abs=1e-9)
    # a run recorded before the ledger existed (composite provider string) still counts
    old = store.create_run(ds_id, "propose", "yolo+mock", "x", {}, status="done")
    db.execute("UPDATE ai_runs SET cost_usd=0.5 WHERE id=?", (old,))
    assert runner.spent_30d("mock") == pytest.approx(run["cost_usd"] + 0.5, abs=1e-9)
    db.execute("DELETE FROM ai_runs")


def test_usage_tagging_and_ledger_split_by_provider():
    from fovea.ai.base import Usage
    u = Usage().add(Usage(10, 5, 0.01).tagged("a")).add(Usage(20, 1, 0.02).tagged("b")).add(Usage(1, 1, 0.0).tagged("c"))
    assert u.by_provider == {"a": [10, 5, 0.01], "b": [20, 1, 0.02], "c": [1, 1, 0.0]}
    assert Usage(1, 1, 0.5).tagged("x").tagged("y").by_provider == {"x": [1, 1, 0.5]}     # first attribution wins
    assert Usage().tagged("x").by_provider == {}


def test_batch_budget_runs_first_image_alone_and_reserves_real_costs(client, ds, monkeypatch):
    from fovea.ai import runner
    from fovea.ai.providers import mock as mock_mod
    ds_id, _ = ds
    monkeypatch.setattr(runner, "_per_image_hi", lambda *a, **k: 0.0002)       # the estimate is far too low …
    monkeypatch.setattr(mock_mod, "DETECT_USD", 0.002)                          # … reasoning made each image cost 10×
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "budget_usd": 0.005, "concurrency": 4, "options": {"mock_latency_ms": 50}})
    job = _wait(r.json()["job"]["id"])
    run = store.get_run(r.json()["run_id"])
    assert job.status == "done" and job.result["stopped"] == "budget"
    assert run["cost_usd"] <= 0.005 + 1e-9 and run["images"] == 2              # 4 in parallel would have cost $0.008


def test_warn_tally_folds_per_image_counts():
    from fovea.ai.runner import WarnTally
    t = WarnTally()
    t.add(["1 box of classes not in this dataset skipped (car ×1)"])
    t.add(["2 boxes of classes not in this dataset skipped (car ×1, chair ×1)", "No labeled examples for x — not looked for"])
    t.add(["No labeled examples for x — not looked for", "3 boxes outside the image or the requested classes dropped"])
    lines = t.lines()
    assert lines[0] == "3 boxes of classes not in this dataset skipped (car ×2, chair ×1)"
    assert "3 boxes outside the image or the requested classes dropped" in lines
    assert "No labeled examples for x — not looked for (2×)" in lines and len(lines) == 3
    t2 = WarnTally()
    t2.add(["1 box of classes not in this dataset skipped (car ×1)"])
    assert t2.lines() == ["1 box of classes not in this dataset skipped (car ×1)"]
    t3 = WarnTally()
    t3.add(["3 crops got no answer", "1 crop got no answer", "1 detected box rejected by the classifier"])
    t3.add(["2 detected boxes rejected by the classifier"])
    assert t3.lines() == ["4 crops got no answer", "3 detected boxes rejected by the classifier"]


def test_tiles_option_per_provider_family():
    from fovea.ai.base import ProviderError
    from fovea.ai.pipeline import tiles_option
    assert tiles_option("anthropic", "auto") == "auto" and tiles_option("anthropic", True) == "auto"
    assert tiles_option("gemini", 2) == 2 and tiles_option("openai_compat", [3, 2]) == [3, 2]
    assert tiles_option("yolo", "auto") is True and tiles_option("yoloe", 640) == 640 and tiles_option("yolo", "800") == 800
    assert tiles_option("anthropic", None) is None and tiles_option("yolo", 0) is None
    for pid, bad in (("anthropic", 640), ("anthropic", [2]), ("anthropic", [0, 2]), ("anthropic", "x"), ("yolo", 2),
                     ("yolo", [2, 2]), ("yolo", 1e9), ("anthropic", [True, 2])):
        with pytest.raises(ProviderError):
            tiles_option(pid, bad)


def test_rejected_proposals_are_not_raised_again(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[10]
    a = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}).json()
    assert a["preds"]
    client.post(f"/api/datasets/{ds_id}/proposals/decide", json={"reject": [p["id"] for p in a["preds"]]})
    b = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}).json()
    assert b["preds"] == [] and any("rejected before" in w for w in b["warnings"])
    assert not any(w.startswith("No objects found") for w in b["warnings"])
    res, job = _batch(client, ds_id, image_ids=[iid])
    assert job.status == "done" and job.result["already_rejected"] == len(a["preds"]) and job.result["boxes"] == 0
    # undoing the rejection (back to pending) lets them come back
    client.post(f"/api/datasets/{ds_id}/proposals/decide", json={"pending": [p["id"] for p in a["preds"]]})
    c = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"}).json()
    assert len(c["preds"]) == len(a["preds"])


# ------------------------------------------------------------------ review round 2 (r2) regressions

def test_cancelling_a_queued_ai_job_ends_its_runs_as_cancelled(client, ds):
    """A job cancelled while queued never runs its function — its run rows must still end 'cancelled' (not
    'running' shown as "interrupted (the server restarted)")."""
    from fovea.jobs import jobs
    ds_id, _ = ds
    slow = {"options": {"mock_latency_ms": 250}, "concurrency": 1}
    busy = [client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                "filters": {}, **slow}).json() for _ in range(2)]   # fill the 2-worker pool
    try:
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                "filters": {}}).json()
        a = client.post("/api/ai/audit", json={"dataset_id": ds_id, "provider": "mock", "model": "mock-fast",
                                                "filters": {}, "mode": "full"}).json()
        b = client.post("/api/ai/benchmark", json={"dataset_id": ds_id, "sample": 4, "configs": [
            {"provider": "mock", "model": "mock-fast"}, {"provider": "mock", "model": "mock-accurate"}]}).json()
        assert jobs.get(r["job"]["id"]).status == "queued" and jobs.get(a["job"]["id"]).status == "queued"
        # through the run (AI tab) …
        run = client.post(f"/api/ai/runs/{r['run_id']}/cancel").json()["run"]
        assert run["status"] == "cancelled" and run["job"]["status"] == "cancelled"
        # … and through the job (job banner / Activity): the hook ends the row at once, before anyone reads it
        jobs.get(a["job"]["id"]).cancel()
        client.post(f"/api/jobs/{b['job']['id']}/cancel")
        for rid in [a["run_id"], *b["run_ids"]]:
            assert db.query_one("SELECT status FROM ai_runs WHERE id=?", (rid,))["status"] == "cancelled"
            assert client.get(f"/api/ai/runs/{rid}").json()["run"]["status"] == "cancelled"
    finally:
        for x in busy:
            jobs.get(x["job"]["id"]).cancel()
        for x in busy:
            _wait(x["job"]["id"])


def test_finish_unstarted_economy_run_and_shutdown(client, ds, monkeypatch):
    from fovea.ai import economy, runner
    from fovea.jobs import jobs
    ds_id, _ = ds
    rid = store.create_run(ds_id, "batch", "mock", "mock-fast", {"speed": "economy", "provider_batches": []})
    db.kv_set(economy.kv_key(rid), {"image_ids": [1], "batches": {}})
    monkeypatch.setattr(jobs.shutting_down, "is_set", lambda: True)
    runner.finish_unstarted([rid])                       # server stopping: those runs really were interrupted
    assert store.get_run(rid)["status"] == "running"
    monkeypatch.undo()
    runner.finish_unstarted([rid])
    assert store.get_run(rid)["status"] == "cancelled" and db.kv_get(economy.kv_key(rid)) is None


def _png_header(path: Path, w: int, h: int) -> Path:
    """A tiny PNG that DECLARES w x h pixels (a decompression bomb for whoever decodes it whole)."""
    import struct
    import zlib

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 1, 0, 0, 0, 0))
                     + chunk(b"IDAT", zlib.compress(b"\x00")) + chunk(b"IEND", b""))
    return path


def test_decompression_bombs_are_refused_before_any_local_decode(tmp_path):
    from fovea.ai import local
    from fovea.ai.assist import crop_jpeg
    from fovea.ai.base import ProviderError
    big = _png_header(tmp_path / "big.png", 16000, 16000)          # 256 MP: above the 250 MP cap
    huge = _png_header(tmp_path / "huge.png", 30000, 30000)        # 900 MP: Pillow itself refuses the header
    for p, dims in ((big, "16000×16000"), (huge, None)):
        with pytest.raises(ProviderError) as e:
            local.check_pixels(str(p))
        assert e.value.code == "too_large" and not e.value.fatal and "FOVEA_MAX_IMAGE_PIXELS" in str(e.value)
        assert dims is None or dims in str(e.value)
        r = local._read_bgr(str(p))                     # the chunk decoder: an error for that image, not a decode
        assert isinstance(r, ProviderError) and r.code == "too_large"
        with pytest.raises(ProviderError):
            local.load_bgr(str(p))
        with pytest.raises(ProviderError):
            crop_jpeg(str(p), (0.5, 0.5, 0.1, 0.1))
    ok = tmp_path / "ok.png"
    Image.new("RGB", (64, 48)).save(ok)
    local.check_pixels(str(ok))


def test_too_large_images_are_per_image_errors_in_jobs(client, ds, monkeypatch):
    from fovea import config as app_config
    from fovea.ai import assist, pipeline
    ds_id, root = ds
    big = root / "images" / "train" / "img_000.jpg"
    Image.new("RGB", (640, 480), (90, 90, 90)).save(big)          # larger than the (lowered) cap below
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    scan_dataset(ds_id, Job(id="s2", kind="scan"))
    monkeypatch.setattr(app_config, "MAX_IMAGE_PIXELS", 100_000)    # 640x480 = 307k > cap; 320x240 = 77k fits
    img = db.query_one("SELECT * FROM images WHERE dataset_id=? AND rel_path LIKE '%img_000%'", (ds_id,))
    assert img["width"] * img["height"] > 100_000
    for prov in ("yolo", "yoloe", "sam", "mock"):
        e = pipeline.pixel_error(img, prov)
        # a cloud provider decodes JPEGs at a reduced scale itself; local decoders never see the file
        assert (e is None) if prov == "mock" else (e is not None and e.code == "too_large")
    png = dict(img, abs_path=str(img["abs_path"]).replace(".jpg", ".png"))
    assert pipeline.pixel_error(png, "mock").code == "too_large"                 # cloud: only JPEGs are left to it
    assert pipeline.pixel_error(dict(img, width=None, height=None), "yolo").code == "too_large"   # header fallback
    # a job counts it as one image's error and goes on (never "fatal", never part of the 10-in-a-row streak)
    monkeypatch.setattr(registry, "is_cloud", lambda pid: pid != "mock" and pid in registry.CLOUD)
    res, job = _batch(client, ds_id)
    assert job.status == "done", job.error
    assert job.extra["errors"] == 1 and "too large to decode" in job.extra["last_error"] and job.extra["images_done"] == 11
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "mode": "full"})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done" and job.result["errors"] == 1 and job.result["images_done"] == 10
    r = client.post("/api/ai/audit", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "mode": "geometry", "class_family": {"person": [0, 1]}})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done" and job.result["errors"] == 1
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": img["id"], "provider": "mock",
                                              "model": "mock-fast"})
    assert r.status_code == 422 and r.json()["code"] == "too_large"
    # naming never picks it as an example
    assert img["id"] not in {e["image_id"] for c in range(3) for e in assist.pick_examples(ds_id, c, 8)}


def test_audit_matches_ground_truth_and_detections_one_to_one():
    from fovea.ai import runner
    det = {"cls": 0, "xc": 0.160698, "yc": 0.548819, "w": 0.055583, "h": 0.163697, "conf": 0.9}
    # rows 0 and 2: the same person labeled twice; row 1 another object the detector found exactly
    gt = [[0, 0.16, 0.55, 0.075, 0.20], [1, 0.6, 0.5, 0.2, 0.3], [0, 0.165, 0.56, 0.08, 0.22]]
    preds = [det, {"cls": 1, "xc": 0.6, "yc": 0.5, "w": 0.2, "h": 0.3, "conf": 0.95}]
    found = geometry.label_issues(gt, preds)
    loose = [f for f in found if f["kind"] == "loose"]
    assert [f["ref_row"] for f in loose] == [0]                      # only the row the detection fits best
    dup = [f for f in found if f["kind"] == "spurious"]
    assert len(dup) == 1 and dup[0]["ref_row"] == 2 and dup[0]["dup_of"] == 0 and "labeled twice" in dup[0]["reason"]
    # two different objects the detector merged: the second is not "tightened" onto the first one's box
    gt2 = [[0, 0.30, 0.5, 0.1, 0.3], [0, 0.38, 0.5, 0.1, 0.3]]
    found = geometry.label_issues(gt2, [{"cls": 0, "xc": 0.34, "yc": 0.5, "w": 0.18, "h": 0.3, "conf": 0.9}])
    assert len([f for f in found if f["kind"] == "loose"]) == 1 and not [f for f in found if f["kind"] == "spurious"]
    # older stored findings (two loose fixes onto one box): accepting both keeps the second pending
    rows = [[0, 0.16, 0.55, 0.075, 0.20], [0, 0.165, 0.56, 0.08, 0.22]]
    fixes = [{"id": 1, "kind": "loose", "ref_row": 0, "cls": 0, **{k: det[k] for k in ("xc", "yc", "w", "h")}},
             {"id": 2, "kind": "loose", "ref_row": 1, "cls": 0, **{k: det[k] for k in ("xc", "yc", "w", "h")}}]
    new_rows, outcome = runner.apply_to_rows(rows, fixes)
    assert outcome["modified"] == [1] and outcome["stale"] == [2] and new_rows[0][1:] != new_rows[1][1:]


def test_audit_duplicate_label_is_named_and_accept_all_removes_the_copy(client, ds):
    from fovea.ai import runner
    from fovea.ai.base import Usage
    ds_id, root = ds
    ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    img = db.query_one("SELECT * FROM images WHERE dataset_id=? AND rel_path LIKE '%img_001%'", (ds_id,))
    lp = Path(img["label_path"])
    lp.write_text("1 0.6 0.5 0.2 0.3\n0 0.16 0.55 0.075 0.2\n0 0.165 0.56 0.08 0.22\n")
    det = {"cls": 0, "xc": 0.160698, "yc": 0.548819, "w": 0.055583, "h": 0.163697, "conf": 0.9, "label": "fall"}
    cfg = runner.prepare_audit(ds_row, "mock", "mock-fast", 0.05, None, "full", None, {})
    run_id = store.create_run(ds_id, "audit", "mock", "mock-fast", {})
    res = runner.audit_image(ds_row, img, cfg, run_id, store_raw=False,
                             pre={0: {"preds": [det, {"cls": 1, "xc": 0.6, "yc": 0.5, "w": 0.2, "h": 0.3, "conf": 0.9}],
                                      "usage": Usage(), "covered": [0, 1, 2]}})
    kinds = sorted((p["kind"], p["ref_row"]) for p in res["preds"])
    assert kinds == [("loose", 1), ("spurious", 2)]
    sp = next(p for p in res["preds"] if p["kind"] == "spurious")
    assert "box #2" in sp["reason"]                                   # rows are numbered from 1, like Annotate
    r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [img["id"]],
                                                                       "kinds": list(store.AUDIT_KINDS)})
    assert _wait(r.json()["job"]["id"]).status == "done"
    out = [ln.split() for ln in lp.read_text().splitlines()]
    assert len(out) == 2 and len({tuple(r) for r in out}) == 2        # the copy is gone, no two identical rows


def test_audit_summary_lists_each_image_once(client, ds):
    ds_id, _ = ds
    ids = _ids(ds_id)[:3]
    run_id = store.create_run(ds_id, "audit", "mock", "mock-fast", {}, status="done")
    box = {"cls": 0, "xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.1, "conf": 0.9}
    for iid, kinds, score in ((ids[0], store.AUDIT_KINDS, 0.9), (ids[1], ("loose", "swap"), 0.8), (ids[2], ("missing",), 0.7)):
        store.insert_preds(run_id, ds_id, iid, [{**box, "kind": k, "ref_row": 0} for k in kinds], "pending")
        store.upsert_audit(ds_id, iid, run_id, score, {k: 1 for k in kinds})
    top = client.get(f"/api/datasets/{ds_id}/audit/summary", params={"top": 3}).json()["top"]
    assert [t["image_id"] for t in top] == ids                       # 3 rows asked, 3 distinct images given
    top = client.get(f"/api/datasets/{ds_id}/audit/summary", params={"top": 20}).json()["top"]
    assert len(top) == len({t["image_id"] for t in top}) == 3
    # an audit cancelled before it looked at any image is not "the last audit" the findings come from
    store.create_run(ds_id, "audit", "mock", "mock-accurate", {}, status="cancelled")
    assert client.get(f"/api/datasets/{ds_id}/audit/summary").json()["run"]["id"] == run_id


def test_economy_skips_images_too_large_to_decode(client, ds, monkeypatch):
    from fovea import config as app_config
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    ds_id, root = ds
    Image.new("RGB", (640, 480), (90, 90, 90)).save(root / "images" / "train" / "img_000.png")
    (root / "images" / "train" / "img_000.jpg").unlink()
    scan_dataset(ds_id, Job(id="s3", kind="scan"))
    monkeypatch.setattr(app_config, "MAX_IMAGE_PIXELS", 100_000)          # the PNG (307k px) is over the cap
    r = _economy(client, ds_id, monkeypatch)
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    res = client.get(f"/api/ai/runs/{r.json()['run_id']}").json()["run"]["result"]
    assert res["errors"] == 1 and "too large to decode" in res["last_error"] and res["images_done"] == 11


def test_batch_budget_waits_for_images_in_flight(client, ds, monkeypatch):
    """Reservations of images still in flight must not stop the run for good: with 4 in parallel it labels as many
    images as one at a time does."""
    from fovea.ai import runner
    ds_id, _ = ds
    monkeypatch.setattr(runner, "_per_image_hi", lambda *a, **k: 0.005)        # estimate 5x the real $0.001
    got = {}
    for conc in (1, 4):
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                                "budget_usd": 0.01, "concurrency": conc, "options": {"mock_latency_ms": 30}})
        job = _wait(r.json()["job"]["id"])
        run = store.get_run(r.json()["run_id"])
        assert job.status == "done" and job.result["stopped"] == "budget" and run["cost_usd"] <= 0.01 + 1e-9
        got[conc] = run["images"]
    assert got[4] == got[1] >= 5                                       # the old rule stopped at 2 with 4 in parallel


def test_cancel_keeps_paid_late_results_and_the_job_summary_matches_the_run(client, ds):
    from fovea.jobs import jobs
    ds_id, root = ds
    for target in ("proposals", "append"):
        before = [_label_text(root, k) for k in range(12)]
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock",
                                                "model": "mock-fast", "target": target, "concurrency": 4,
                                                "options": {"mock_latency_ms": 300}}).json()
        time.sleep(0.45)
        jobs.get(r["job"]["id"]).cancel()
        job = _wait(r["job"]["id"])
        run = store.get_run(r["run_id"])
        assert job.status == "cancelled" and run["status"] == "cancelled"
        ex = job.extra
        assert ex["images_done"] == run["images"] and ex["cost_usd"] == pytest.approx(run["cost_usd"], abs=1e-9)
        assert ex.get("late", 0) >= 1 and "after the cancel" in job.message
        pending = db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE run_id=? AND state='pending'", (r["run_id"],))["n"]
        if target == "proposals":
            assert pending == ex["boxes"]
        else:                     # late results are kept as proposals: no label file is written after the cancel
            late_imgs = {r["image_id"] for r in db.query("SELECT image_id FROM ai_preds WHERE run_id=? AND "
                                                         "state='pending'", (r["run_id"],))}
            assert pending > 0 and len(late_imgs) <= ex["late"]
            changed = sum(1 for k in range(12) if _label_text(root, k) != before[k])
            assert changed == ex["written"]
    # a fatal error: what finished afterwards is billed and shown too
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": "mock-fast",
                                            "concurrency": 4, "options": {"mock_latency_ms": 150, "mock_error": "auth",
                                                                          "mock_error_every": 4}}).json()
    job = _wait(r["job"]["id"])
    run = store.get_run(r["run_id"])
    assert job.status == "error" and job.extra["cost_usd"] == pytest.approx(run["cost_usd"], abs=1e-9)
    assert job.extra["images_done"] == run["images"]


def test_unpriced_cloud_model_cannot_bypass_budget_or_spend_cap(client, ds, monkeypatch):
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    m = "mock-unlisted-9-9"              # like a model "Refresh list" added: served, but not in the price table
    e = client.post("/api/ai/estimate", json={"dataset_id": ds_id, "count": 10, "provider": "mock", "model": m}).json()
    assert e["priced"] is False and e["unpriced"] == [m] and e["usd"] == [0, 0]
    assert any("no price on file" in a for a in e["assumptions"])
    assert client.post("/api/ai/estimate", json={"dataset_id": ds_id, "count": 10, "provider": "mock",
                                                  "model": "mock-fast"}).json()["priced"] is True
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "mock", "model": m,
                                            "budget_usd": 0.01})
    assert r.status_code == 422 and "No price on file" in r.json()["detail"] and "budget" in r.json()["detail"]
    two = {"pipeline": "two_stage", "detector": {"provider": "mock", "model": "mock-fast"},
           "classifier": {"provider": "mock", "model": m}}
    r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "", "options": two,
                                            "budget_usd": 1})
    assert r.status_code == 422 and m in r.json()["detail"]          # a two-stage classifier counts too
    # no budget, no cap: allowed (cost not tracked, and said so)
    res, job = _batch(client, ds_id, model=m, image_ids=_ids(ds_id)[:3])
    assert job.status == "done" and any("no price on file" in w for w in job.result["warnings"])
    # with a spend cap on the provider, every path refuses it
    monkeypatch.setattr(ai_config, "spend_cap", lambda pid: 2.5 if pid == "mock" else None)
    for path, body in (("/api/ai/batch", {"filters": {}, "provider": "mock", "model": m}),
                       ("/api/ai/propose", {"image_id": iid, "provider": "mock", "model": m}),
                       ("/api/ai/preview", {"image_ids": [iid], "provider": "mock", "model": m}),
                       ("/api/ai/classify", {"image_id": iid, "provider": "mock", "model": m,
                                             "boxes": [[0, 0.5, 0.5, 0.2, 0.2]]}),
                       ("/api/ai/audit", {"filters": {}, "provider": "mock", "model": m, "mode": "full"}),
                       ("/api/ai/review", {"image_id": iid, "provider": "mock", "model": m}),
                       ("/api/ai/benchmark", {"sample": 4, "configs": [{"provider": "mock", "model": m}]}),
                       ("/api/ai/name-classes", {"provider": "mock", "model": m, "per_class": 4}),
                       ("/api/ai/nl-filter", {"provider": "mock", "model": m, "text": "fall"})):
        r = client.post(path, json={"dataset_id": ds_id, **body})
        assert r.status_code == 422 and "spend cap" in r.json()["detail"], (path, r.status_code, r.text)
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 200                                        # a priced model is fine under the cap
    r = client.post("/api/ai/name-classes", json={"dataset_id": ds_id, "provider": "mock", "model": m, "per_class": 4,
                                                   "dry_run": True})
    assert r.status_code == 200 and r.json()["estimate"]["priced"] is False   # the estimate itself still shows


def test_local_job_estimate_uses_measured_throughput(client, ds, monkeypatch):
    from fovea.ai import estimate as E
    from fovea.ai import local
    db.init_db()
    monkeypatch.setattr(local, "is_imported", lambda: False)
    monkeypatch.setattr(local, "_throughput", {})
    monkeypatch.setattr(local, "_throughput_loaded", True)
    monkeypatch.setattr(local, "speed_ms", lambda fam, name: None)
    s, how = E._local_seconds("yolo", "yolo12m", {})
    assert how == "guess" and s * 7808 < 10 * 60                     # was 0.25 s/image: "about 33 min" for 7,808
    assert E._local_seconds("yolo", "yolo12x", {})[0] > s > E._local_seconds("yolo", "yolo26n", {})[0]
    assert E._local_seconds("yolo", "yolo12m", {"tiles": True})[0] > s      # one image per call: no chunking
    local.record_throughput("yolo", "yolo12m", "batch", 999, 40.6)
    assert E._local_seconds("yolo", "yolo12m", {}) == (pytest.approx(0.0406, abs=1e-4), "measured")
    assert db.kv_get(local.SPEED_KEY)["yolo|yolo12m|batch"]["s"] == pytest.approx(0.0406, abs=1e-4)
    # a local job records its throughput (the mock posing as a local batch-capable model)
    from fovea.ai.providers import mock
    monkeypatch.setattr(mock.MockProvider, "capabilities", {**mock.MockProvider.capabilities, "batch_infer": True})
    monkeypatch.setattr(mock.PROVIDER, "kind", "local", raising=False)
    from fovea.ai import runner
    monkeypatch.setattr(runner, "MIN_SPEED_IMAGES", 2)
    ds_id, _ = ds
    _res, job = _batch(client, ds_id, options={"infer_batch": 2, "mock_latency_ms": 250})
    assert job.status == "done" and local.job_seconds("mock", "mock-fast", "batch")


def test_naming_estimate_prices_each_crop_at_its_real_size(client, ds):
    from fovea.ai import assist
    import io as _io
    tall = {"xc": 0.5, "yc": 0.5, "w": 0.1, "h": 0.5, "width": 640, "height": 640}
    w, h = assist.crop_size(tall)
    assert h == assist.CROP_EDGE and w < h / 3

    class P:
        id = "mock"

        @staticmethod
        def image_tokens(model, sw, sh, options):
            return sw * sh // 750
    reqs = [([0], "Class 0 …", {})]
    est = assist.naming_estimate(P(), "mock-fast", reqs, {0: [tall] * 6})
    text = len(assist.NAME_SYSTEM) // 4 + len("Class 0 …") // 4 + 40
    assert est["tokens_in"][0] == text + 6 * (w * h // 750)
    assert est["tokens_in"][0] < text + 6 * (assist.CROP_EDGE ** 2 // 750) // 2      # was priced as square crops
    # the size is what crop_jpeg really sends
    ds_id, root = ds
    data = assist.crop_jpeg(str(root / "images" / "train" / "img_000.jpg"), (0.2, 0.5, 0.15, 0.3))
    assert Image.open(_io.BytesIO(data)).size == assist.crop_size(
        {"xc": 0.2, "yc": 0.5, "w": 0.15, "h": 0.3, "width": 320, "height": 240})


def test_runs_list_counts_read_a_covering_index(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    rid = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": iid, "provider": "mock",
                                                "model": "mock-fast"}).json()["run_id"]
    runs = client.get("/api/ai/runs", params={"dataset_id": ds_id}).json()["runs"]
    assert next(r for r in runs if r["id"] == rid)["preds"]["pending"] >= 1
    lean = client.get("/api/ai/runs", params={"dataset_id": ds_id, "counts": 0}).json()["runs"]
    assert lean and all("preds" not in r for r in lean) and any(r["id"] == rid for r in lean)
    plan = " ".join(str(dict(r)) for r in db.query(
        "EXPLAIN QUERY PLAN SELECT run_id, state, COUNT(*) n FROM ai_preds WHERE run_id IN (?,?) GROUP BY run_id, state",
        [rid, rid + 1]))
    assert "COVERING INDEX idx_ai_preds_run_state" in plan


def test_yoloe_text_encoder_message_names_the_real_settings_card():
    import inspect
    from fovea.ai.providers import yoloe
    src = inspect.getsource(yoloe)
    assert "Settings › Model weights" in src and "Your labels" in src and "Settings › AI models" not in src


# ------------------------------------------------------------------ round 4 (ai-core)

def _pred(cls, xc, yc, w, h, conf=0.9, kind="detect", **kw):
    return {"cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf, "label": "", "kind": kind, "reason": "", **kw}


def test_apply_to_rows_never_puts_two_classes_on_one_object():
    from fovea.ai import runner
    rows = [[0, 0.2, 0.5, 0.15, 0.3]]
    preds = [{**_pred(1, 0.2, 0.5, 0.15, 0.3, 0.95), "id": 1},        # on the class-0 box: a conflict
             {**_pred(0, 0.205, 0.5, 0.15, 0.3, 0.9), "id": 2},       # same class: already labeled
             {**_pred(1, 0.7, 0.5, 0.1, 0.2, 0.8), "id": 3},          # new object …
             {**_pred(2, 0.7, 0.5, 0.1, 0.2, 0.6), "id": 4},          # … proposed twice: the less confident one waits
             {**_pred(2, 0.45, 0.2, 0.1, 0.1, None), "id": 5}]        # no confidence: last, still added
    new_rows, out = runner.apply_to_rows(rows, preds)
    assert sorted(out["conflicts"]) == [1, 4] and out["duplicates"] == [2] and sorted(out["added"]) == [3, 5]
    assert [r[0] for r in new_rows] == [0, 1, 2]
    _rows, out = runner.apply_to_rows(rows, preds, skip_conflicts=False)       # only an explicit override writes them
    assert 1 in out["added"] and not out["conflicts"]


def test_bulk_accept_leaves_proposals_over_another_class_pending(client, ds):
    ds_id, root = ds
    iid = _ids(ds_id)[0]                                   # one row: class 0 at (0.2, 0.5, 0.15, 0.3)
    run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {}, status="done")
    stored = store.insert_preds(run_id, ds_id, iid, [_pred(1, 0.2, 0.5, 0.15, 0.3, 0.95), _pred(0, 0.205, 0.5, 0.15, 0.3),
                                                     _pred(2, 0.7, 0.5, 0.1, 0.2, 0.7)], "pending")
    r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid], "min_conf": 0.5})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    res = job.result
    assert (res["added"], res["duplicates"], res["conflicts"]) == (1, 1, 1) and res["conflict_images"] == [iid]
    assert "another class" in job.message
    rows = [ln.split() for ln in _label_text(root, 0).splitlines()]
    assert [r[0] for r in rows] == ["0", "2"]                             # never a class-1 box on the class-0 object
    state = {p["id"]: db.query_one("SELECT state FROM ai_preds WHERE id=?", (p["id"],))["state"] for p in stored}
    assert state[stored[0]["id"]] == "pending"                            # left for a person (⇄ in Annotate)


def test_append_batch_leaves_boxes_over_another_class_as_proposals(client, ds):
    from fovea.ai import runner
    ds_id, root = ds
    iid = _ids(ds_id)[0]
    ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    img = db.query_one("SELECT * FROM images WHERE id=?", (iid,))
    run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {})
    out = {"preds": [_pred(1, 0.2, 0.5, 0.15, 0.3, 0.95), _pred(0, 0.205, 0.5, 0.15, 0.3), _pred(2, 0.7, 0.5, 0.1, 0.2, 0.7)]}
    for _ in range(2):                                     # a repeated Append neither stacks boxes nor proposals
        stats = {"boxes": 0, "skipped": 0, "duplicates": 0, "written": 0, "conflicts": 0}
        sink = runner.BatchSink(ds_row, run_id, "append", False, stats)
        sink.apply(iid, img, out)
        sink.close()
        assert stats["conflicts"] == 1
    assert [ln.split()[0] for ln in _label_text(root, 0).splitlines()] == ["0", "2"]
    pend = db.query("SELECT cls, reason FROM ai_preds WHERE image_id=? AND state='pending'", (iid,))
    assert [p["cls"] for p in pend] == [1] and "another class" in pend[0]["reason"]
    store.finish_run(run_id, "done")


def test_audit_fixes_follow_their_box_after_rows_shift():
    from fovea.ai import runner
    a, b, c = [1, 0.2, 0.5, 0.1, 0.2], [0, 0.5, 0.5, 0.1, 0.2], [0, 0.8, 0.5, 0.1, 0.3]
    loose = {"id": 1, "kind": "loose", "cls": 0, "xc": 0.8, "yc": 0.5, "w": 0.08, "h": 0.25, "ref_row": 2, "ref_box": c[1:]}
    swap = {"id": 2, "kind": "swap", "cls": 1, "xc": b[1], "yc": b[2], "w": b[3], "h": b[4], "ref_row": 1, "ref_box": b[1:]}
    gone = {"id": 3, "kind": "spurious", "cls": 1, "xc": a[1], "yc": a[2], "w": a[3], "h": a[4], "ref_row": 0,
            "ref_box": a[1:]}
    rows = [list(b), list(c)]                               # row 0 (a) was deleted after the audit: b, c moved up
    new_rows, out = runner.apply_to_rows(rows, [loose, swap, gone])
    assert out["modified"] == [1, 2] and out["obsolete"] == [3] and not out["stale"]
    assert new_rows == [[1, 0.5, 0.5, 0.1, 0.2], [0, 0.8, 0.5, 0.08, 0.25]]
    # without a stored ref_box nothing can tell where the box went: left pending as before
    _r, out = runner.apply_to_rows(rows, [{**gone, "ref_box": None}])
    assert out["stale"] == [3]


def test_bulk_accept_applies_shifted_findings_and_resolves_gone_ones(client, ds):
    ds_id, root = ds
    ids = _ids(ds_id)
    iid = ids[2]                                            # rows: 2 @0.2, 0 @0.45, 1 @0.7
    lp = root / "labels" / "train" / "img_002.txt"
    old = [ln.split() for ln in lp.read_text().splitlines()]
    box = lambda r: [float(v) for v in r[1:5]]             # noqa: E731
    run_id = store.create_run(ds_id, "audit", "mock", "m", {}, status="done")
    store.insert_preds(run_id, ds_id, iid, [
        {**_pred(1, 0.7, 0.5, 0.12, 0.35, kind="loose"), "ref_row": 2, "ref_box": box(old[2])},
        {**_pred(2, *box(old[0]), conf=None, kind="spurious"), "ref_row": 0, "ref_box": box(old[0])}], "pending")
    lp.write_text("\n".join(" ".join(r) for r in old[1:]) + "\n")      # Objects deleted row 0
    from fovea.core.scanner import refresh_image
    refresh_image(iid, 3)
    r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid], "kinds": list(store.AUDIT_KINDS)})
    job = _wait(r.json()["job"]["id"])
    assert job.status == "done", job.error
    assert job.result["modified"] == 1 and job.result["obsolete"] == 1 and job.result["stale"] == 0
    rows = [ln.split() for ln in lp.read_text().splitlines()]
    assert len(rows) == 2 and float(rows[1][3]) == pytest.approx(0.12) and float(rows[1][4]) == pytest.approx(0.35)
    left = db.query_one("SELECT COUNT(*) n FROM ai_preds WHERE image_id=? AND state='pending'", (iid,))["n"]
    assert left == 0                                         # nothing stays a suspect for good


def test_loose_fix_after_its_detection_was_accepted_leaves_one_box(client, ds):
    from fovea.ai import runner
    gt, det = [0, 0.5, 0.5, 0.2, 0.4], [0.52, 0.48, 0.12, 0.3]     # IoU 0.45: the detection was accepted as new
    rows, out = runner.apply_to_rows([list(gt)], [{**_pred(0, *det), "id": 1}])
    assert out["added"] == [1] and len(rows) == 2
    fix = {**_pred(0, *det, kind="loose"), "id": 2, "ref_row": 0, "ref_box": gt[1:]}
    rows, out = runner.apply_to_rows(rows, [fix])
    assert out["merged"] == [2] and rows == [[0, *det]]       # the loose copy is gone, no two identical rows
    # through the API, in the order Inspect's ⇧Enter twice does it
    ds_id, root = ds
    iid = _ids(ds_id)[0]
    run_id = store.create_run(ds_id, "audit", "mock", "m", {}, status="done")
    store.insert_preds(run_id, ds_id, iid, [_pred(0, 0.21, 0.52, 0.09, 0.2, 0.9),
                                            {**_pred(0, 0.21, 0.52, 0.09, 0.2, kind="loose"), "ref_row": 0,
                                             "ref_box": [0.2, 0.5, 0.15, 0.3]}], "pending")
    for kinds in (list(store.DETECT_KINDS), list(store.AUDIT_KINDS)):
        r = client.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid], "kinds": kinds})
        job = _wait(r.json()["job"]["id"])
        assert job.status == "done", job.error
    assert job.result["merged"] == 1 and "removed" in job.message
    rows = [tuple(ln.split()) for ln in _label_text(root, 0).splitlines()]
    assert len(rows) == 1 and float(rows[0][3]) == pytest.approx(0.09)


def test_spend_cap_holds_economy_runs_waiting_at_the_provider(client, ds, monkeypatch):
    from fovea.ai import runner
    ds_id, _ = ds
    db.execute("DELETE FROM ai_runs")
    monkeypatch.setattr(ai_config, "spend_cap", lambda pid: 0.01 if pid == "mock" else None)
    first = _economy(client, ds_id, monkeypatch, delay=2.5)            # 12 images × $0.0006 held while it waits
    assert first.status_code == 200, first.text
    second = _economy(client, ds_id, monkeypatch, delay=2.5)
    assert second.status_code == 200, second.text
    p1, p2 = (store.get_run(r.json()["run_id"])["params"] for r in (first, second))
    assert store.get_run(second.json()["run_id"])["result"]["stopped"] == "cap"
    assert p1["images"] == 12 and p1["per_image_usd"] == pytest.approx(0.0006)
    assert p2["budget_usd"] == pytest.approx(0.01 - 12 * 0.0006, abs=1e-9) and p2["images"] == 4
    assert runner.reserved_usd("mock") == pytest.approx(16 * 0.0006, abs=1e-9)
    assert client.get("/api/ai/usage").json()["caps"]["mock"]["reserved_usd"] == pytest.approx(16 * 0.0006, abs=1e-6)
    res, job = _batch(client, ds_id)                                    # a Now run gets only what is left: $0.0004
    assert job.status == "done" and job.result["stopped"] == "cap" and job.result["images_done"] == 0
    assert "spend cap" in job.message
    for r in (first, second):
        assert _wait(r.json()["job"]["id"], timeout=60).status == "done"
    assert runner.spent_30d("mock") <= 0.01 + 1e-9 and runner.reserved_usd("mock") == 0
    db.execute("DELETE FROM ai_runs")


def test_running_now_runs_hold_their_remaining_spend(ds):
    from fovea.ai import runner
    ds_id, _ = ds
    rid = store.create_run(ds_id, "batch", "mock", "mock-fast", {"speed": "now"})
    runner.hold(rid, ["mock"], 0.004)
    assert runner.reserved_usd("mock") == pytest.approx(0.004) and runner.reserved_usd("mock", exclude_run=rid) == 0
    assert runner.reserved_usd("anthropic") == 0
    store.finish_run(rid, "done")                            # an ended run never holds money
    assert runner.reserved_usd("mock") == 0 and rid not in runner._holds


def test_remote_guard_covers_a_local_server_that_gets_the_owners_key(client, ds, monkeypatch, tmp_path):
    ds_id, _ = ds
    from fovea.api import ai as ai_api
    monkeypatch.setattr(secrets, "DATA_DIR", tmp_path)
    monkeypatch.delenv("OPENAI_COMPAT_API_KEY", raising=False)
    ai_config.update_provider_config("openai_compat", {"base_url": "http://127.0.0.1:9/v1"})   # nothing listens
    registry.get("openai_compat")
    try:
        assert registry.is_cloud("openai_compat") is False and registry.is_guarded("openai_compat") is False
        monkeypatch.setattr(ai_api, "_is_local", lambda req: False)
        monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
        assert client.post("/api/ai/providers/openai_compat/test").status_code == 200     # Ollama-style: no key
        secrets.set_key("openai_compat", "LocalCompatSecret9xQ2Zr")       # a LiteLLM gateway to a paid model
        assert registry.is_guarded("openai_compat") is True
        assert client.post("/api/ai/providers/openai_compat/test").status_code == 403
        r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": _ids(ds_id)[0],
                                                  "provider": "openai_compat", "model": "qwen3-vl"})
        assert r.status_code == 403
        r = client.post("/api/ai/batch", json={"dataset_id": ds_id, "filters": {}, "provider": "openai_compat",
                                                "model": "qwen3-vl"})
        assert r.status_code == 403
        entry = next(p for p in client.get("/api/ai/providers").json()["providers"] if p["id"] == "openai_compat")
        assert entry["guarded"] is True and "stay on this machine" not in entry["location"]
        assert "API key" in entry["location"] and "LocalCompat" not in json.dumps(entry)
    finally:
        ai_config.update_provider_config("openai_compat", {"base_url": ""})


def test_symlinked_weights_are_listed_and_loadable(tmp_path, monkeypatch):
    md, elsewhere = tmp_path / "model", tmp_path / "shared"
    md.mkdir()
    elsewhere.mkdir()
    (elsewhere / "yolo12m.pt").write_bytes(b"x" * 10)
    (elsewhere / "notes.txt").write_text("not weights")
    (md / "yolo12m.pt").symlink_to(elsewhere / "yolo12m.pt")
    (md / "sneaky.pt").symlink_to(elsewhere / "notes.txt")             # a link to something that is not weights
    (md / "dangling.pt").symlink_to(elsewhere / "missing.pt")
    monkeypatch.setattr(weights, "MODEL_DIR", md)
    assert [w["name"] for w in weights.local_weights()] == ["yolo12m"]
    p = weights.resolve("yolo12m", "yolo")
    assert p.parent == md.resolve() and p.name == "yolo12m.pt" and p.is_file()
    for bad in ("sneaky", "dangling", "../shared/yolo12m"):
        with pytest.raises(ValueError):
            weights.resolve(bad)
    assert {w["name"]: w["downloaded"] for w in weights.weights_list() if not w["curated"]} == {"yolo12m.pt": True}


def test_exemplar_keys_change_only_with_the_references(ds):
    from fovea.ai import pipeline
    from fovea.core.scanner import refresh_image
    ds_id, root = ds
    ds_row = db.query_one("SELECT * FROM datasets WHERE id=?", (ds_id,))
    specs = pipeline.specs_for(ds_row, None)
    ex, key, ck = pipeline.build_exemplars(ds_row, specs, per_class=2)
    used = {p for refs in ex.values() for p, _b in refs}
    img = next(r for r in db.query("SELECT * FROM images WHERE dataset_id=? AND has_label=1 AND n_boxes > 0 ORDER BY "
                                   "rel_path", (ds_id,)) if r["abs_path"] not in used)

    def nudge(row):
        lp = Path(row["label_path"])
        lines = [ln.split() for ln in lp.read_text().splitlines()]
        lines[0][1] = f"{float(lines[0][1]) + 0.004:.6f}"
        lp.write_text("\n".join(" ".join(x) for x in lines) + "\n")
        os.utime(lp, (time.time() + 5, time.time() + 5))
        refresh_image(row["id"], 3)
    nudge(img)                                               # a save on a frame no prototype uses
    ex2, key2, ck2 = pipeline.build_exemplars(ds_row, specs, per_class=2)
    assert key2 == key and ck2 == ck
    ref_path = ex[0][0][0]                                   # a reference image of class 0
    ref = db.query_one("SELECT * FROM images WHERE abs_path=?", (ref_path,))
    lp = Path(ref["label_path"])
    lines = [ln.split() for ln in lp.read_text().splitlines()]
    k = next(i for i, ln in enumerate(lines) if ln[0] == "0")
    lines[0], lines[k] = lines[k], lines[0]                  # nudge that image's class-0 box
    lp.write_text("\n".join(" ".join(x) for x in lines) + "\n")
    nudge(ref)
    _ex3, key3, ck3 = pipeline.build_exemplars(ds_row, specs, per_class=2)
    assert key3 != key and ck3[0] != ck[0]
    assert all(ck3[c] == ck[c] for c in ck if c != 0 and ref_path not in {p for p, _b in ex[c]})


def test_propose_run_lasts_as_long_as_the_request(client, ds):
    ds_id, _ = ds
    r = client.post("/api/ai/propose", json={"dataset_id": ds_id, "image_id": _ids(ds_id)[0], "provider": "mock",
                                              "model": "mock-fast", "options": {"mock_latency_ms": 400}})
    assert r.status_code == 200, r.text
    run = store.get_run(r.json()["run_id"])
    assert run["finished_at"] - run["created_at"] >= 0.35


def test_job_speed_ignores_pauses(monkeypatch):
    from fovea.ai import local, runner
    clock = {"t": 100.0}
    monkeypatch.setattr(runner.time, "monotonic", lambda: clock["t"])
    m = runner.SpeedMeter(16)
    m.start(8)
    for k in range(1, 20):                                  # 16 images per 0.8 s …
        clock["t"] += 0.8 + (600 if k == 7 else 0)           # … and one window with a 10-minute sleep in it
        m.tick(8 + 16 * k)
    m.tick(8 + 16 * 19, final=True)
    images, seconds = m.result()
    assert images == 16 * 18 and seconds == pytest.approx(0.8 * 18)
    # across runs: one slowed run cannot move the stored speed; two can
    monkeypatch.setattr(local, "_throughput", {"yolo|yolo12m|batch": {"s": 3.28, "device": "mps"}})
    monkeypatch.setattr(local, "_throughput_loaded", True)
    monkeypatch.setattr(local, "_ready", False)
    local.record_throughput("yolo", "yolo12m", "batch", 200, 10.0, "mps")      # 0.05 s/image
    assert local.job_seconds("yolo", "yolo12m", "batch") == pytest.approx(0.05)
    local.record_throughput("yolo", "yolo12m", "batch", 1000, 12900.0, "mps")  # a run that spanned a sleep
    assert local.job_seconds("yolo", "yolo12m", "batch") == pytest.approx(0.05)
    local.record_throughput("yolo", "yolo12m", "batch", 1000, 13000.0, "mps")  # slow twice: the machine is slower
    assert local.job_seconds("yolo", "yolo12m", "batch") == pytest.approx(12.9)


def test_audit_summary_counts_only_what_is_still_pending(client, ds):
    ds_id, _ = ds
    iid = _ids(ds_id)[0]
    run_id = store.create_run(ds_id, "audit", "mock", "mock-fast", {}, status="done")
    preds = store.insert_preds(run_id, ds_id, iid, [{**_pred(0, 0.5, 0.5, 0.1, 0.1), "kind": "missing"},
                                                    {**_pred(0, 0.2, 0.5, 0.15, 0.3), "kind": "spurious",
                                                     "ref_row": 0}], "pending")
    store.upsert_audit(ds_id, iid, run_id, 0.62, {"missing": 1, "spurious": 1})
    store.decide(ds_id, [preds[0]["id"]], [])                 # the missing object was accepted in Annotate
    top = client.get(f"/api/datasets/{ds_id}/audit/summary").json()["top"]
    assert next(t for t in top if t["image_id"] == iid)["counts"] == {"spurious": 1}


def test_usage_lists_only_real_providers(client, ds):
    ds_id, _ = ds
    db.execute("DELETE FROM ai_runs")
    store.create_run(ds_id, "interp", "fovea", "fill-gap", {}, status="done")
    store.create_run(ds_id, "batch", "yolo+mock", "yolo12m+mock-fast", {}, status="done")     # stopped before any call
    u = client.get("/api/ai/usage").json()
    assert "fovea" not in u["by_provider"] and "yolo+mock" not in u["by_provider"]
    assert u["by_provider"]["mock"]["runs"] == 1 and not any(k.startswith("fovea/") for k in u["by_model"])
    db.execute("DELETE FROM ai_runs")


def test_cloud_estimate_counts_parallel_requests(client, ds):
    ds_id, _ = ds
    ids = _ids(ds_id)
    body = {"dataset_id": ds_id, "provider": "mock", "model": "mock-fast"}
    e = client.post("/api/ai/estimate", json={**body, "image_ids": ids}).json()
    assert e["seconds_per_request"] == pytest.approx(0.05) and e["concurrency"] == 4
    assert e["seconds_per_image"] == pytest.approx(0.0125) and e["seconds_total"] == pytest.approx(0.15, abs=0.006)
    assert any("4 requests at a time" in a for a in e["assumptions"])
    e = client.post("/api/ai/estimate", json={**body, "image_ids": ids[:2]}).json()     # 2 images: 2 at a time
    assert e["seconds_per_image"] == pytest.approx(0.025)
    e = client.post("/api/ai/estimate", json={**body, "image_ids": ids, "concurrency": 1}).json()
    assert e["seconds_per_image"] == pytest.approx(0.05)
