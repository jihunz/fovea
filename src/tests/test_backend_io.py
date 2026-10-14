"""backend-io: security guards, jobs, thumbnails / serving, paths, exports."""
from __future__ import annotations

import io
import json
import os
import time
import uuid
import zipfile
from pathlib import Path

import pytest

os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from PIL import Image  # noqa: E402

from fovea import db, security  # noqa: E402
from fovea.core import layout  # noqa: E402
from fovea.jobs import Job, JobManager  # noqa: E402


# ---------------------------------------------------------------- helpers

def _img(path: Path, size=(64, 48), color=(120, 120, 120), **save):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", size, color).save(path, **save)
    return path


def _register(root: Path, classes=None) -> str:
    from fovea.core.scanner import scan_dataset
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "io-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(classes or []),
                "user" if classes else "inferred", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    if classes:
        db.execute("UPDATE datasets SET classes=?, classes_source='user' WHERE id=?", (db.dumps(classes), ds_id))
    _settle(ds_id)
    return ds_id


def _settle(ds_id, timeout=20.0):
    """Let jobs a scan queued in the background (e.g. the data-quality pass) finish — including a re-run the
    QA engine queues from a helper thread a few ms after its first job ends: only a quiet 0.1 s counts as
    settled (a test that then plants its own flags must not have them overwritten)."""
    from fovea.jobs import jobs
    deadline = time.monotonic() + timeout
    quiet = time.monotonic()
    while time.monotonic() < deadline:
        active = jobs.active_all(ds_id)
        if active:
            jobs.wait(active, max(0.0, deadline - time.monotonic()))
            quiet = time.monotonic()
        elif time.monotonic() - quiet >= 0.1:
            return
        time.sleep(0.02)


def _drop(ds_id):
    from fovea.jobs import jobs
    jobs.cancel_all(ds_id, wait=10)
    for t in ("image_flags", "image_hash", "boxes", "images", "reviews"):
        try:
            db.execute(f"DELETE FROM {t} WHERE dataset_id=?", (ds_id,))
        except Exception:  # noqa: BLE001
            pass
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _client():
    from fastapi.testclient import TestClient
    from fovea.main import app
    return TestClient(app)


def _client_from(addr, **kw):
    """A TestClient on the app whose requests come from `addr` (see testkit.client_from)."""
    from testkit import client_from
    from fovea.main import app
    return client_from(app, addr, **kw)


def _ids(ds_id):
    return {r["rel_path"]: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?", (ds_id,))}


# ---------------------------------------------------------------- security

def test_host_check_blocks_dns_rebinding_but_allows_ips_and_localhost():
    c = _client()
    assert c.get("/api/settings", headers={"host": "attacker.example"}).status_code == 400
    assert c.get("/api/settings", headers={"host": "attacker.example:8000"}).status_code == 400
    for ok in ("localhost:8000", "127.0.0.1:8000", "[::1]:8000", "192.168.1.20:8888", "app.localhost"):
        assert c.get("/api/settings", headers={"host": ok}).status_code == 200, ok
    assert security.host_allowed("LOCALHOST.")
    assert not security.host_allowed("")


def test_allowed_hosts_setting(monkeypatch):
    monkeypatch.setattr(security, "_ALLOWED_HOSTS", ["fovea.lab.example", "*.corp.example"])
    assert security.host_allowed("fovea.lab.example:443")
    assert security.host_allowed("a.b.corp.example")
    assert not security.host_allowed("corp.example.evil.com")


def test_cross_origin_writes_are_refused_whatever_the_content_type():
    c = _client()
    assert c.post("/api/cache/clear", headers={"origin": "https://attacker.example"}).status_code == 403
    assert c.post("/api/cache/clear", headers={"origin": "null"}).status_code == 403
    assert c.post("/api/cache/clear", headers={"sec-fetch-site": "cross-site"}).status_code == 403
    assert c.post("/api/cache/clear", headers={"sec-fetch-site": "same-site"}).status_code == 403
    assert c.put("/api/settings", content=b"x=1", headers={"origin": "http://evil.test", "content-type": "text/plain"}).status_code == 403
    # same origin (with or without the port spelled out) and non-browser clients pass
    assert c.post("/api/cache/clear", headers={"origin": "http://testserver", "sec-fetch-site": "same-origin"}).status_code == 200
    assert c.post("/api/cache/clear", headers={"host": "localhost:8000", "origin": "http://localhost:8000"}).status_code == 200
    assert c.post("/api/cache/clear").status_code == 200
    # reads are never refused by the origin check (cross-origin pages cannot read them anyway)
    assert c.get("/api/meta", headers={"origin": "https://attacker.example"}).status_code == 200


def test_security_headers_on_every_response():
    c = _client()
    for url in ("/", "/api/meta", "/static/app.js"):
        r = c.get(url)
        assert r.headers["x-frame-options"] == "DENY"
        assert "frame-ancestors 'none'" in r.headers["content-security-policy"]
        assert r.headers["x-content-type-options"] == "nosniff"


def test_is_local_request_and_remote_ai_guard(monkeypatch):
    assert security.is_local_request({"client": ("127.0.0.1", 5)})
    assert security.is_local_request({"client": ("::1", 5)})
    assert security.is_local_request({"client": ("::ffff:127.0.0.1", 5)})
    assert not security.is_local_request({"client": ("192.168.0.7", 5)})
    assert not security.is_local_request({"client": None})
    monkeypatch.setattr(security, "_LOCAL_NETS", [__import__("ipaddress").ip_network("172.16.0.0/12")])
    assert security.is_local_request({"client": ("172.17.0.1", 5)})
    from fastapi import HTTPException
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    with pytest.raises(HTTPException) as e:
        security.require_ai_access({"client": ("10.1.2.3", 9)})
    assert e.value.status_code == 403 and "FOVEA_ALLOW_REMOTE_AI" in e.value.detail
    security.require_ai_access({"client": ("127.0.0.1", 9)})
    monkeypatch.setenv("FOVEA_ALLOW_REMOTE_AI", "1")
    security.require_ai_access({"client": ("10.1.2.3", 9)})
    assert security.remote_ai_allowed()


def test_scrub_removes_keys_and_tokens():
    s = security.scrub("401 for sk-ant-api03-AbCdEf123456_xyz at https://x/v1?key=AIzaSyA1234567890abcdefghijk&alt=json "
                       "Authorization: Bearer abcdef.123456 sk-proj-ABCDEFGHIJKLMNOPQRST x-goog-api-key: AIzaZZZZZZZZZZZZZZZZZZZZZZ")
    for leak in ("AbCdEf123456", "AIzaSyA1234567890", "abcdef.123456", "ABCDEFGHIJKLMNOPQRST", "ZZZZZZZZ"):
        assert leak not in s, s
    assert "alt=json" in s
    security.register_secret("my-custom-local-key-123")
    assert "my-custom-local-key-123" not in security.scrub("failed with my-custom-local-key-123")
    assert security.scrub("plain text, 12 images") == "plain text, 12 images"


def test_error_responses_never_carry_secrets():
    from fastapi import HTTPException
    from fovea.main import app

    @app.get("/api/_test_leak_http")
    def _leak_http():
        raise HTTPException(502, "upstream said: invalid x-api-key sk-ant-api03-SECRETSECRETSECRET")

    @app.get("/api/_test_leak_500")
    def _leak_500():
        raise RuntimeError("GET https://generativelanguage.googleapis.com/v1/models?key=AIzaSECRETSECRETSECRETSECRET failed")

    from fastapi.testclient import TestClient
    c = TestClient(app, raise_server_exceptions=False)
    r = c.get("/api/_test_leak_http")
    assert r.status_code == 502 and "SECRETSECRET" not in r.text
    r = c.get("/api/_test_leak_500")
    assert r.status_code == 500 and "SECRETSECRET" not in r.text and "RuntimeError" in r.text


def test_token_required_from_other_machines(monkeypatch):
    import asyncio
    monkeypatch.setenv("FOVEA_TOKEN", "t0ken-for-tests")
    sent = []

    async def app(scope, receive, send):
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    async def call(client, path="/api/meta", headers=(), qs=b"", method="GET"):
        sent.clear()

        async def send(m):
            sent.append(m)

        scope = {"type": "http", "method": method, "path": path, "query_string": qs, "client": client, "scheme": "http",
                 "headers": [(b"host", b"192.168.1.5:8000")] + list(headers)}
        await security.GuardMiddleware(app)(scope, None, send)
        return sent[0]["status"], dict((k.decode(), v.decode()) for k, v in sent[0]["headers"])

    remote = ("192.168.1.9", 5000)
    assert asyncio.run(call(remote))[0] == 401
    assert asyncio.run(call(remote, headers=[(b"authorization", b"Bearer wrong")]))[0] == 401
    assert asyncio.run(call(remote, headers=[(b"authorization", b"Bearer t0ken-for-tests")]))[0] == 200
    assert asyncio.run(call(remote, headers=[(b"cookie", b"a=1; fovea_token=t0ken-for-tests")]))[0] == 200
    status, headers = asyncio.run(call(remote, path="/d/x", qs=b"token=t0ken-for-tests&tab=2"))
    assert status == 303 and headers["location"] == "/d/x?tab=2" and "fovea_token=t0ken-for-tests" in headers["set-cookie"]
    status, headers = asyncio.run(call(remote, path="/", headers=[(b"accept", b"text/html")]))
    assert status == 401 and headers["content-type"].startswith("text/html")
    assert asyncio.run(call(("127.0.0.1", 1)))[0] == 200            # this machine needs no token
    assert "t0ken-for-tests" not in security.scrub("token t0ken-for-tests")


# ---------------------------------------------------------------- settings / meta

def test_settings_refuse_secrets_and_merge_deeply():
    c = _client()
    db.kv_set("settings", {"old": {"api_key": "sk-ant-api03-LEAKEDLEAKED"}, "keep": 1})
    assert "LEAKED" not in c.get("/api/meta").text and "LEAKED" not in c.get("/api/settings").text
    r = c.put("/api/settings", json={"anthropic_api_key": "sk-ant-api03-abcdefghijkl"})
    assert r.status_code == 400 and "secrets" in r.json()["detail"]
    assert c.put("/api/settings", json={"x": {"note": "sk-ant-api03-abcdefghijkl"}}).status_code == 400
    assert c.put("/api/settings", json={"keyboard": {"layout": "qwerty"}}).status_code == 200
    c.put("/api/settings", json={"ai": {"provider": "yolo", "model": "a"}})
    assert c.put("/api/settings", json={"ai": {"model": "b"}}).json()["settings"]["ai"] == {"provider": "yolo", "model": "b"}
    assert c.put("/api/settings", json={"ai": {"provider": None}}).json()["settings"]["ai"] == {"model": "b"}
    assert c.put("/api/settings", json=[1, 2]).status_code == 400
    stored = db.kv_get("settings")
    assert "old" not in json.dumps(stored) or "LEAKED" not in json.dumps(stored)
    db.kv_set("settings", {})


def test_meta_is_cheap_and_never_imports_ultralytics(monkeypatch):
    import sys
    c = _client()
    before = "ultralytics" in sys.modules
    t = time.perf_counter()
    j = c.get("/api/meta").json()
    assert time.perf_counter() - t < 2.0
    assert ("ultralytics" in sys.modules) == before
    assert set(j["thumb_cache"]) >= {"files", "bytes", "pending"}
    assert j["server"]["remote_ai_allowed"] in (True, False)


# ---------------------------------------------------------------- jobs

def test_cancelling_a_queued_job_finishes_it_at_once_and_it_never_runs():
    import threading
    m = JobManager(workers={"default": 1})
    gate = threading.Event()
    ran = []
    a = m.submit("long", lambda j: gate.wait(5), dataset_id="d1")
    b = m.submit("export_copy", lambda j: ran.append(1), dataset_id="d1")
    time.sleep(0.05)
    assert b.status == "queued" and b.to_dict()["queue_position"] == 1 and b.to_dict()["elapsed"] == 0.0
    b.cancel()
    assert b.status == "cancelled" and b.finished_at is not None
    assert m.active_for("d1", "export_copy") is None
    gate.set()
    m.wait([a], 5)
    time.sleep(0.05)
    assert not ran and a.status == "done"
    assert a.to_dict()["started_at"] >= a.created_at


def test_cancel_all_and_shutdown_stop_every_job():
    m = JobManager(workers={"default": 2})

    def loop(j):
        for i in range(500):
            j.update(i, 500)
            time.sleep(0.01)

    jobs_ = [m.submit("scan", loop, dataset_id="dx"), m.submit("autolabel", loop, dataset_id="dx"),
             m.submit("export_copy", loop, dataset_id="dx")]
    time.sleep(0.1)
    hit = m.cancel_all("dx", wait=3)
    assert len(hit) == 3 and all(j.status == "cancelled" for j in jobs_)
    j2 = m.submit("scan", loop, dataset_id="dy")
    time.sleep(0.05)
    t = time.perf_counter()
    m.shutdown(wait=3)
    assert j2.status == "cancelled" and time.perf_counter() - t < 3
    assert m.submit("late", loop).status == "cancelled"


def test_job_errors_are_plain_and_scrubbed():
    m = JobManager(workers={"default": 1})
    j1 = m.submit("x", lambda j: (_ for _ in ()).throw(FileNotFoundError("Dataset files are not reachable")))
    j2 = m.submit("x", lambda j: (_ for _ in ()).throw(RuntimeError("bad key sk-ant-api03-SECRETSECRETSECRET")))
    m.wait([j1, j2], 3)
    assert j1.error == "Dataset files are not reachable"
    assert j2.error.startswith("RuntimeError:") and "SECRETSECRET" not in j2.error


def test_job_events_end_when_the_server_shuts_down(monkeypatch):
    import threading
    from fovea import jobs as jobs_mod
    m = JobManager(workers={"default": 1})
    monkeypatch.setattr("fovea.api.jobs.jobs", m)
    gate = threading.Event()
    job = m.submit("long", lambda j: gate.wait(10))
    c = _client()
    threading.Timer(0.5, m.begin_shutdown).start()
    t = time.perf_counter()
    with c.stream("GET", f"/api/jobs/{job.id}/events") as r:
        frames = [line for line in r.iter_lines() if line.startswith("data: ")]
    assert time.perf_counter() - t < 5 and frames
    gate.set()
    assert jobs_mod.jobs is not m


# ---------------------------------------------------------------- paths / fs

def test_browse_reports_missing_folders_with_the_nearest_parent(tmp_path):
    (tmp_path / "a" / "b").mkdir(parents=True)
    for i in range(30):
        _img(tmp_path / "a" / f"f{i:02d}.jpg", size=(8, 8))
    (tmp_path / "a" / "notes.txt").write_text("x")
    c = _client()
    r = c.get("/api/fs/browse", params={"path": str(tmp_path / "a" / "missing" / "deeper")})
    assert r.status_code == 404 and r.json()["nearest"] == str(tmp_path / "a")
    r = c.get("/api/fs/browse", params={"path": str(tmp_path / "a"), "limit": 10}).json()
    assert r["counts"] == {"dirs": 1, "files": 31, "images": 30} and r["truncated"] and len(r["files"]) == 10
    assert [d["name"] for d in r["dirs"]] == ["b"]
    r = c.get("/api/fs/browse", params={"path": str(tmp_path / "a"), "files": 0}).json()
    assert r["files"] == [] and r["counts"]["images"] == 30 and not r["truncated"]
    r = c.get("/api/fs/browse", params={"path": str(tmp_path / "a" / "f01.jpg")}).json()
    assert r["selected"] == "f01.jpg" and r["current"] == str(tmp_path / "a")


def test_host_paths_map_into_the_container_even_before_they_exist(tmp_path, monkeypatch):
    from fovea import paths
    mount = tmp_path / "host"
    (mount / "data").mkdir(parents=True)
    monkeypatch.setattr(paths, "PATH_MAPPINGS", [("/Users/someone", str(mount))])
    got = paths.resolve("/Users/someone/data/new-export")
    assert got == mount / "data" / "new-export" and not got.exists()
    assert paths.to_host(got) == "/Users/someone/data/new-export"
    monkeypatch.setattr(paths, "_HOST_IS_WINDOWS", True)
    monkeypatch.setattr(paths, "PATH_MAPPINGS", [("C:\\Data", str(mount)), ("/c/Data", str(mount))])
    assert paths.resolve("c:\\data\\data\\x") == mount / "data" / "x"
    assert paths.resolve("C:/DATA/data") == (mount / "data").resolve()
    assert paths.resolve("rel/p", base=tmp_path) == tmp_path / "rel" / "p"


def test_detect_names_yaml_and_list_sources_after_their_folder(tmp_path):
    from fovea.api.fs import suggested_name
    root = tmp_path / "낙상 데이터"
    _img(root / "images" / "train" / "a.jpg")
    (root / "data.yaml").write_text("train: images/train\nval: images/train\nnames: [a]\n")
    assert suggested_name(root / "data.yaml") == "낙상 데이터"
    assert suggested_name(root) == "낙상 데이터"


def test_new_ultralytics_formats_are_indexed():
    from fovea.config import IMAGE_EXTS
    assert ".mpo" in IMAGE_EXTS
    from PIL import features
    if features.check("avif"):
        assert ".avif" in IMAGE_EXTS
    if features.check("jpg_2000"):
        assert ".jp2" in IMAGE_EXTS
    assert ".dng" not in IMAGE_EXTS


# ---------------------------------------------------------------- thumbnails / serving

def _imgds(tmp_path, files):
    """Dataset with arbitrary image files: files = {rel: callable(path)}."""
    root = tmp_path / ("r" + uuid.uuid4().hex[:6])
    for rel, make in files.items():
        p = root / "images" / "train" / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        make(p)
    (root / "labels" / "train").mkdir(parents=True, exist_ok=True)
    return _register(root), root


def _u16(path, w=640, h=512, hi=4095, hot=False):
    import numpy as np
    a = (np.linspace(0, hi, w * h).reshape(h, w)).astype(np.uint16)
    if hot:
        a[0, 0] = 65535
    Image.fromarray(a).save(path)


def test_16bit_thumbnails_render_at_every_size_and_ignore_hot_pixels(tmp_path):
    ds, _root = _imgds(tmp_path, {"d16.png": lambda p: _u16(p, hot=True), "d16.tif": lambda p: _u16(p, 1200, 900)})
    try:
        c = _client()
        for rel, iid in _ids(ds).items():
            for s in (128, 256, 512):
                r = c.get(f"/api/thumb/{iid}", params={"s": s})
                assert r.status_code == 200, (rel, s, r.text)
                im = Image.open(io.BytesIO(r.content)).convert("L")
                lo, hi = im.getextrema()
                assert hi > 200 and lo < 40, (rel, s, lo, hi)          # a real gradient, not black
    finally:
        _drop(ds)


def test_high_bit_png_is_served_as_a_stretched_rendition(tmp_path):
    ds, _root = _imgds(tmp_path, {"d16.png": lambda p: _u16(p), "plain.png": lambda p: _img(p, (40, 30))})
    try:
        c = _client()
        ids = _ids(ds)
        r = c.get(f"/api/img/{ids['train/d16.png']}")
        assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg"
        assert Image.open(io.BytesIO(r.content)).convert("L").getextrema()[1] > 200
        r = c.get(f"/api/img/{ids['train/plain.png']}")
        assert r.headers["content-type"] == "image/png"
    finally:
        _drop(ds)


def test_full_images_revalidate_unless_versioned(tmp_path):
    ds, root = _imgds(tmp_path, {"a.jpg": lambda p: _img(p, (40, 30))})
    try:
        c = _client()
        iid = _ids(ds)["train/a.jpg"]
        r = c.get(f"/api/img/{iid}")
        assert r.headers["cache-control"] == "private, no-cache"
        etag = r.headers["etag"]
        assert c.get(f"/api/img/{iid}", headers={"if-none-match": etag}).status_code == 304
        assert "immutable" in c.get(f"/api/img/{iid}", params={"v": "123"}).headers["cache-control"]
        # replacing the file changes the ETag, so a revalidating browser fetches the new picture
        time.sleep(0.01)
        _img(root / "images" / "train" / "a.jpg", (41, 30), (10, 200, 10))
        os.utime(root / "images" / "train" / "a.jpg", (time.time() + 5, time.time() + 5))
        assert c.get(f"/api/img/{iid}", headers={"if-none-match": etag}).status_code == 200
    finally:
        _drop(ds)


def test_thumbnails_follow_exif_orientation_and_keep_the_colour_profile(tmp_path):
    from PIL import ImageCms

    def rotated(p):
        im = Image.new("RGB", (400, 200), (200, 30, 30))
        ex = Image.Exif()
        ex[0x0112] = 6
        im.save(p, exif=ex)

    def tagged(p):
        icc = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
        Image.new("RGB", (900, 600), (20, 120, 220)).save(p, icc_profile=icc)

    ds, _root = _imgds(tmp_path, {"rot.jpg": rotated, "icc.jpg": tagged})
    try:
        c = _client()
        ids = _ids(ds)
        im = Image.open(io.BytesIO(c.get(f"/api/thumb/{ids['train/rot.jpg']}", params={"s": 128}).content))
        assert im.size[1] > im.size[0]                       # portrait after applying orientation 6
        im = Image.open(io.BytesIO(c.get(f"/api/thumb/{ids['train/icc.jpg']}", params={"s": 128}).content))
        assert im.info.get("icc_profile")
    finally:
        _drop(ds)


def test_small_web_images_are_served_as_their_own_thumbnail(tmp_path):
    ds, root = _imgds(tmp_path, {"small.jpg": lambda p: _img(p, (300, 200))})
    try:
        c = _client()
        iid = _ids(ds)["train/small.jpg"]
        r = c.get(f"/api/thumb/{iid}", params={"s": 512, "v": "1.3"})
        assert r.content == (root / "images" / "train" / "small.jpg").read_bytes() and "immutable" in r.headers["cache-control"]
        r = c.get(f"/api/thumb/{iid}", params={"s": 128})
        assert Image.open(io.BytesIO(r.content)).size == (128, 85)
        assert c.get(f"/api/thumb/{iid}", params={"s": 2048}).status_code == 200
    finally:
        _drop(ds)


def test_decompression_bombs_are_refused_quickly(tmp_path, monkeypatch):
    from fovea.core import thumbs
    ds, _root = _imgds(tmp_path, {"big.png": lambda p: Image.new("L", (3000, 3000)).save(p, optimize=True)})
    try:
        monkeypatch.setattr(thumbs, "MAX_IMAGE_PIXELS", 4_000_000)
        c = _client()
        iid = _ids(ds)["train/big.png"]
        t = time.perf_counter()
        r = c.get(f"/api/thumb/{iid}", params={"s": 256})
        assert r.status_code == 415 and "MP" in r.json()["detail"] and time.perf_counter() - t < 2
        # the failure is remembered: no second decode attempt
        assert c.get(f"/api/thumb/{iid}", params={"s": 256}).status_code == 415
    finally:
        _drop(ds)


def test_thumbnail_cache_counter_and_trim(tmp_path, monkeypatch):
    from fovea.core import thumbs
    ds, _root = _imgds(tmp_path, {f"i{k}.png": (lambda p: _img(p, (600, 400))) for k in range(6)})
    try:
        thumbs.clear_cache()
        c = _client()
        for iid in _ids(ds).values():
            assert c.get(f"/api/thumb/{iid}", params={"s": 128}).status_code == 200
        st = thumbs.cache_stats()
        assert st["files"] == 6 and st["bytes"] > 0
        assert thumbs.cache_stats(refresh=True, wait=True)["files"] == 6
        old = time.time() - 3 * 86400
        for e in thumbs._walk_files():
            os.utime(e.path, (old, old))
        res = thumbs.trim_cache(1)
        assert res["removed"] == 6 and thumbs.cache_stats()["files"] == 0
    finally:
        _drop(ds)


def test_big_non_jpeg_sizes_derive_from_one_decode(tmp_path, monkeypatch):
    from fovea.core import thumbs
    ds, _root = _imgds(tmp_path, {"big.png": lambda p: Image.new("RGB", (2600, 2000), (9, 99, 199)).save(p)})
    try:
        thumbs.clear_cache()
        calls = []
        real = thumbs._decode
        monkeypatch.setattr(thumbs, "_decode", lambda *a, **k: (calls.append(a[1]), real(*a, **k))[1])
        c = _client()
        iid = _ids(ds)["train/big.png"]
        for s in (128, 256, 512):
            assert c.get(f"/api/thumb/{iid}", params={"s": s}).status_code == 200
        assert calls == [1024]
    finally:
        _drop(ds)


# ---------------------------------------------------------------- export

def _labeled(tmp_path, spec, classes=("person", "fall")):
    """spec: {"<split>/<name>.jpg": "label text" | None}"""
    root = tmp_path / ("x" + uuid.uuid4().hex[:6])
    for rel, text in spec.items():
        split, name = rel.split("/", 1)
        _img(root / "images" / split / name, (64, 48))
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        if text is not None:
            (root / "labels" / split / (Path(name).stem + ".txt")).write_text(text)
    return _register(root, list(classes)), root


def _zip_names(data: bytes):
    return zipfile.ZipFile(io.BytesIO(data)).namelist()


def test_export_resize_validation_is_a_400_before_anything_starts(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        for bad in ({"mode": "max", "size": 0}, {"mode": "max", "size": ""}, {"mode": "max"},
                    {"mode": "exact", "width": "640px", "height": 640}, {"mode": "exact", "width": 20000, "height": 64},
                    {"mode": "weird"}):
            for url in ("zip-link", "copy", "preview"):
                r = c.post(f"/api/datasets/{ds}/export/{url}", json={"resize": bad, "target_dir": str(tmp_path / "o")})
                assert r.status_code == 400, (bad, url, r.text)
    finally:
        _drop(ds)


def test_zip_link_has_exact_length_manifest_and_a_portable_yaml(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "train/b.jpg": None,
                                    "test-1854-포함/c.jpg": "1 0.5 0.5 0.2 0.2 0.9\n"})
    try:
        c = _client()
        link = c.post(f"/api/datasets/{ds}/export/zip-link", json={"filename": "내보내기"}).json()
        assert link["images"] == 3 and link["bytes"] > 0
        r = c.get(link["url"])
        assert r.status_code == 200 and int(r.headers["content-length"]) == len(r.content) == link["bytes"]
        assert "filename*=UTF-8''%EB%82%B4" in r.headers["content-disposition"]
        z = zipfile.ZipFile(io.BytesIO(r.content))
        assert z.testzip() is None
        import yaml
        doc = yaml.safe_load(z.read("data.yaml"))
        assert "path" not in doc and doc["train"] == "images/train" and doc["val"] == "images/train"
        assert doc["test"] == "images/test-1854-포함" and doc["names"] == {0: "person", 1: "fall"}
        assert z.read("labels/test-1854-포함/c.txt").decode() == "1 0.500000 0.500000 0.200000 0.200000\n"
        man = json.loads(z.read("fovea-export.json"))
        assert man["counts"]["images"] == 3 and man["dataset_id"] == ds
        info = z.getinfo("images/train/a.jpg")
        assert info.date_time[0] >= 2020
    finally:
        _drop(ds)


def test_exported_yaml_trains_from_any_working_directory(tmp_path, monkeypatch):
    pytest.importorskip("ultralytics")
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "val/b.jpg": "1 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        out = tmp_path / "unz"
        zipfile.ZipFile(io.BytesIO(c.get(c.post(f"/api/datasets/{ds}/export/zip-link", json={}).json()["url"]).content)).extractall(out)
        elsewhere = tmp_path / "cwd"
        elsewhere.mkdir()
        monkeypatch.chdir(elsewhere)
        from ultralytics.data.utils import check_det_dataset
        d = check_det_dataset(str(out / "data.yaml"), autodownload=False)
        assert Path(d["train"]).is_dir() and Path(d["val"]).is_dir()
    finally:
        _drop(ds)


def test_zip_stream_sends_large_chunks_not_one_per_write(tmp_path):
    from fovea.core.export import stream_zip
    spec = {f"train/{k:04d}.jpg": "0 0.5 0.5 0.2 0.2\n" for k in range(300)}
    ds, _root = _labeled(tmp_path, spec)
    try:
        chunks = list(stream_zip(ds, ["person"], {}))
        data = b"".join(chunks)
        assert len(_zip_names(data)) == 300 + 300 + 1
        assert len(chunks) <= 2 + len(data) // (256 * 1024) + 2, len(chunks)
    finally:
        _drop(ds)


def test_resized_export_is_upright_and_high_bit_images_are_not_black(tmp_path):
    root = tmp_path / "orient"
    im = Image.new("RGB", (400, 200), (0, 0, 0))
    im.paste((255, 255, 255), (300, 0, 400, 100))       # bright object top-right of the stored pixels
    ex = Image.Exif()
    ex[0x0112] = 6
    (root / "images" / "train").mkdir(parents=True)
    im.save(root / "images" / "train" / "r.jpg", exif=ex, quality=95)
    _u16(root / "images" / "train" / "d.png")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "r.txt").write_text("0 0.75 0.875 0.5 0.25\n")      # upright frame (200x400)
    ds = _register(root, ["obj"])
    try:
        from fovea.core.export import copy_subset
        out = tmp_path / "out"
        res = copy_subset(Job(id="c", kind="export_copy"), ds, ["obj"], {}, str(out), resize={"mode": "max", "size": 100})
        assert res["images"] == 2 and res["converted_8bit"] == 1
        r = Image.open(out / "images" / "train" / "r.jpg")
        assert r.size == (50, 100) and r.getexif().get(0x0112) in (None, 1)
        # the labelled region (x 0.5-1.0, y 0.75-1.0 of the upright image) is the bright object
        px = r.convert("L").crop((30, 80, 48, 98))
        assert sum(px.getdata()) / (18 * 18) > 180
        d = Image.open(out / "images" / "train" / "d.jpg").convert("L")
        assert d.getextrema()[1] > 200
    finally:
        _drop(ds)


def test_copy_refuses_a_non_empty_folder_and_replaces_its_own_previous_export(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "val/b.jpg": "1 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        out = tmp_path / "exp"
        out.mkdir()
        (out / "mine.txt").write_text("keep me")
        r = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(out)})
        assert r.status_code == 409 and r.json()["target"]["entries"] == 1
        from fovea.core.export import copy_subset
        res = copy_subset(Job(id="1", kind="export_copy"), ds, ["person", "fall"], {}, str(out),
                          opts=__import__("fovea.core.export", fromlist=["x"]).parse_options({"overwrite": True}))
        assert res["images"] == 2 and (out / "fovea-export.json").is_file()
        # a narrower re-export into the same folder removes the previous export's stale files only
        from fovea.core import export as ex
        res = ex.copy_subset(Job(id="2", kind="export_copy"), ds, ["person", "fall"], {"split": "val"}, str(out),
                             opts=ex.parse_options({"overwrite": True}))
        assert res["removed_stale"] == 2
        assert not (out / "images" / "train" / "a.jpg").exists() and (out / "images" / "val" / "b.jpg").exists()
        assert (out / "mine.txt").read_text() == "keep me"
        import yaml
        doc = yaml.safe_load((out / "data.yaml").read_text())
        assert doc["path"] == str(out.resolve()) and doc["val"] == "images/val" and doc["train"] == "images/val"
    finally:
        _drop(ds)


def test_copy_stops_with_a_clear_message_when_the_destination_is_not_writable(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    out = tmp_path / "ro"
    (out / "images" / "train").mkdir(parents=True)
    os.chmod(out / "images" / "train", 0o555)
    try:
        from fovea.core import export as ex
        from fovea.jobs import JobError
        with pytest.raises(JobError, match="permission denied"):
            ex.copy_subset(Job(id="1", kind="export_copy"), ds, ["person"], {}, str(out), opts=ex.parse_options({"overwrite": True}))
    finally:
        os.chmod(out / "images" / "train", 0o755)
        _drop(ds)


def test_missing_and_undecodable_images_are_reported(tmp_path):
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "train/b.jpg": None, "train/c.jpg": None})
    try:
        (root / "images" / "train" / "b.jpg").unlink()
        (root / "images" / "train" / "c.jpg").write_bytes(b"not an image")
        from fovea.core import export as ex
        res = ex.copy_subset(Job(id="1", kind="export_copy"), ds, ["person"], {}, str(tmp_path / "o"),
                             opts=ex.parse_options({"resize": {"mode": "max", "size": 32}}))
        assert res["images"] == 1 and res["missing"] == 1 and res["failed"] == 1
        reasons = {s["path"]: s["reason"] for s in res["skipped"]}
        assert reasons["train/b.jpg"] == "missing" and "decoded" in reasons["train/c.jpg"]
        data = b"".join(ex.stream_zip(ds, ["person"], {}, opts=ex.parse_options({"resize": {"mode": "max", "size": 32}}),
                                      include_manifest=True))
        man = json.loads(zipfile.ZipFile(io.BytesIO(data)).read("fovea-export.json"))
        assert man["counts"]["missing"] == 1 and man["counts"]["failed"] == 1
    finally:
        _drop(ds)


def test_same_stem_images_do_not_overwrite_each_other(tmp_path):
    root = tmp_path / "stem"
    _img(root / "images" / "train" / "a.jpg")
    _img(root / "images" / "train" / "a.png")
    (root / "labels" / "train").mkdir(parents=True)
    (root / "labels" / "train" / "a.txt").write_text("0 0.5 0.5 0.2 0.2\n")
    ds = _register(root, ["x"])
    try:
        from fovea.core import export as ex
        data = b"".join(ex.stream_zip(ds, ["x"], {}, resize={"mode": "exact", "width": 32, "height": 32}))
        names = _zip_names(data)
        assert len(names) == len(set(names))
        assert {"images/train/a.jpg", "images/train/a_png.jpg", "labels/train/a.txt", "labels/train/a_png.txt"} <= set(names)
    finally:
        _drop(ds)


def test_class_remap_drop_compact_and_unnamed_ids(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n2 0.6 0.6 0.1 0.1\n",
                                    "train/b.jpg": "\ufeff9 0.5 0.5 0.2 0.2\n1.5 0.1 0.1 0.1 0.1\n"
                                                   "0 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4\n"},
                         classes=("a", "b", "c"))
    try:
        from fovea.core import export as ex
        z = zipfile.ZipFile(io.BytesIO(b"".join(ex.stream_zip(ds, ["a", "b", "c"], {}, opts=ex.parse_options(
            {"class_map": {"0": 1, "1": 0, "2": None}})))))
        assert z.read("labels/train/a.txt").decode().split("\n")[:2] == ["1 0.500000 0.500000 0.200000 0.200000",
                                                                          "0 0.300000 0.300000 0.100000 0.100000"]
        rows_b = z.read("labels/train/b.txt").decode().splitlines()
        assert rows_b[0].startswith("9 ") and rows_b[1] == "1 0.1 0.1 0.4 0.1 0.4 0.4 0.1 0.4" and len(rows_b) == 2
        import yaml
        doc = yaml.safe_load(z.read("data.yaml"))
        assert doc["nc"] == 10 and doc["names"][0] == "b" and doc["names"][1] == "a" and doc["names"][9] == "class_9"
        # compact after dropping: ids renumbered densely
        cp = ex.class_plan(["a", "b", "c"], ex.parse_options({"class_map": {"0": None}, "compact": True}))
        assert cp.names == ["b", "c"] and cp.map(1) == 0 and cp.map(2) == 1 and cp.map(0) is None
        # bad_class=drop removes rows whose id has no name
        z = zipfile.ZipFile(io.BytesIO(b"".join(ex.stream_zip(ds, ["a", "b", "c"], {}, opts=ex.parse_options({"bad_class": "drop"})))))
        assert all(not r.startswith("9 ") for r in z.read("labels/train/b.txt").decode().splitlines())
        assert yaml.safe_load(z.read("data.yaml"))["nc"] == 3
        with pytest.raises(ValueError):
            ex.parse_options({"class_map": {"x": 1}})
    finally:
        _drop(ds)


def _flag_dups(ds, groups):
    db.execute("DELETE FROM image_flags WHERE dataset_id=? AND code='near_dup'", (ds,))
    for g, members in groups.items():
        for iid in members:
            db.execute("INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
                       (iid, ds, "near_dup", 0, json.dumps({"group": g, "dist": 2, "size": len(members)})))
    db.execute("INSERT OR REPLACE INTO image_hash(image_id, dataset_id, dhash) VALUES(?,?,0)", (next(iter(groups.values()))[0], ds))


def test_drop_near_duplicates_keeps_one_per_group(tmp_path):
    spec = {f"train/v1_{k:03d}.jpg": ("0 0.5 0.5 0.2 0.2\n" * (2 if k == 3 else 1)) for k in range(6)}
    ds, _root = _labeled(tmp_path, spec)
    try:
        ids = _ids(ds)
        _flag_dups(ds, {7: [ids[f"train/v1_{k:03d}.jpg"] for k in range(5)]})
        c = _client()
        p = c.post(f"/api/datasets/{ds}/export/preview", json={"drop_near_dups": True}).json()
        assert p["images"] == 2 and p["near_dups"] == {"available": True, "groups": 1, "dropped": 4}
        names = _zip_names(c.get(c.post(f"/api/datasets/{ds}/export/zip-link", json={"drop_near_dups": True}).json()["url"]).content)
        imgs = sorted(n for n in names if n.startswith("images/"))
        assert imgs == ["images/train/v1_003.jpg", "images/train/v1_005.jpg"]     # the member with most boxes stays
    finally:
        _drop(ds)


def test_split_builder_keeps_sequences_whole_and_is_reproducible(tmp_path):
    spec = {}
    for v in range(20):
        for f in range(5):
            spec[f"train/video{v:02d}_frame_{f:03d}.jpg"] = f"{v % 2} 0.5 0.5 0.2 0.2\n"
    ds, _root = _labeled(tmp_path, spec)
    try:
        from fovea.core import export as ex
        opts = ex.parse_options({"split_builder": {"ratios": {"train": 70, "val": 20, "test": 10}, "seed": 3}})
        p1 = ex.build_plan(ds, ["person", "fall"], {}, opts)
        p2 = ex.build_plan(ds, ["person", "fall"], {}, opts)
        assert [i.out_split for i in p1.items] == [i.out_split for i in p2.items]
        per_seq = {}
        for it in p1.items:
            per_seq.setdefault(it.seq, set()).add(it.out_split)
        assert len(per_seq) == 20 and all(len(s) == 1 for s in per_seq.values())
        counts = p1.split_counts
        assert counts["train"] == 70 and counts["val"] == 20 and counts["test"] == 10
        other = ex.build_plan(ds, ["person", "fall"], {}, ex.parse_options({"split_builder": {"ratios": {"train": 70, "val": 20, "test": 10}, "seed": 4}}))
        assert [i.out_split for i in other.items] != [i.out_split for i in p1.items]
        c = _client()
        pv = c.post(f"/api/datasets/{ds}/export/preview", json={"split_builder": {"train": 0.8, "val": 0.2}}).json()
        assert {s["name"] for s in pv["splits"]} == {"train", "val"} and pv["sequences"]["spanning_splits"] == 0
        assert "val: images/val" in pv["yaml"]
        z = zipfile.ZipFile(io.BytesIO(c.get(c.post(f"/api/datasets/{ds}/export/zip-link",
                                                     json={"split_builder": {"train": 0.8, "val": 0.2}}).json()["url"]).content))
        assert any(n.startswith("images/val/") for n in z.namelist()) and any(n.startswith("labels/val/") for n in z.namelist())
        with pytest.raises(ValueError):
            ex.parse_options({"split_builder": {"train": 0, "val": 1}})
    finally:
        _drop(ds)


def test_relative_lists_resolve_from_the_list_file(tmp_path):
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "테스트-1/b.jpg": None})
    try:
        c = _client()
        r = c.post(f"/api/datasets/{ds}/export/list", json={"style": "rel", "write": True, "filename": "목록-테스트-1"})
        assert r.status_code == 200
        out = root / "목록-테스트-1.txt"
        lines = out.read_text().splitlines()
        assert lines and all(l.startswith("./") and (root / l).is_file() for l in lines)
        r = c.post(f"/api/datasets/{ds}/export/list", json={"style": "rel", "write": True, "filename": "목록-테스트-1"})
        assert r.status_code == 409 and r.json()["exists"]
        assert c.post(f"/api/datasets/{ds}/export/list", json={"style": "rel", "write": True, "filename": "목록-테스트-1",
                                                               "overwrite": True}).status_code == 200
    finally:
        _drop(ds)


def test_rewriting_data_yaml_keeps_comments_relative_paths_and_a_backup(tmp_path):
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "val/b.jpg": None})
    original = ("# my training config\npath: .\n\n# splits\ntrain: images/train\nval: images/val\n"
                "kpt_shape: [17, 3]  # keep\nnames:\n  0: person\n")
    (root / "data.yaml").write_text(original)
    lay = layout.detect(str(root)).to_dict()
    db.execute("UPDATE datasets SET layout=? WHERE id=?", (db.dumps(lay), ds))
    try:
        c = _client()
        pv = c.get(f"/api/datasets/{ds}/export/yaml").json()
        assert pv["changed"] and pv["comments_kept"] and pv["current"] == original
        r = c.post(f"/api/datasets/{ds}/export/yaml").json()
        text = (root / "data.yaml").read_text()
        assert "# my training config" in text and "# splits" in text and "kpt_shape: [17, 3]  # keep" in text
        assert "train: images/train" in text and "val: images/val" in text
        assert f"path: {root.resolve()}" in text or f"path: {root}" in text
        import yaml
        assert yaml.safe_load(text)["names"] == {0: "person", 1: "fall"}
        assert (root / "data.yaml.bak").read_text() == original and r["backup"]
        # unchanged: no second rewrite, no new backup
        (root / "data.yaml.bak").unlink()
        c.post(f"/api/datasets/{ds}/export/yaml")
        assert not (root / "data.yaml.bak").exists()
    finally:
        _drop(ds)


def test_log_redaction_keeps_uvicorn_access_records_intact():
    import logging
    from uvicorn.logging import AccessFormatter
    rec = logging.LogRecord("uvicorn.access", logging.INFO, __file__, 1, '%s - "%s %s HTTP/%s" %d',
                            ("127.0.0.1:5", "GET", "/api/x?key=AIzaSyA1234567890abcdefghijk", "1.1", 200), None)
    assert security.RedactingFilter().filter(rec)
    line = AccessFormatter('%(client_addr)s - "%(request_line)s" %(status_code)s').format(rec)
    assert "AIzaSyA1234567890" not in line and "200" in line and "GET /api/x?key=" in line
    try:
        raise RuntimeError("Bearer abcdefghijklmnop")
    except RuntimeError:
        import sys
        rec = logging.LogRecord("fovea", logging.ERROR, __file__, 1, "boom %s", (ValueError("sk-ant-api03-zzzzzzzzzzzz"),), sys.exc_info())
    security.RedactingFilter().filter(rec)
    text = logging.Formatter().format(rec)
    assert "zzzzzzzzzzzz" not in text and "abcdefghijklmnop" not in text and "boom" in text


def test_deep_colour_tiff_and_cmyk_render_with_real_contrast(tmp_path):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")

    def rgb16(p):
        a = np.zeros((300, 400, 3), np.uint16)
        a[..., 0] = np.linspace(0, 4095, 400)[None, :]          # 12-bit data in a 16-bit container
        a[..., 1] = 2048
        tifffile.imwrite(str(p), a, photometric="rgb")

    def cmyk(p):
        Image.new("CMYK", (300, 200), (0, 200, 200, 0)).save(p, "JPEG")

    ds, _root = _imgds(tmp_path, {"rgb16.tif": rgb16, "cmyk.jpg": cmyk})
    try:
        c = _client()
        ids = _ids(ds)
        for path in (f"/api/thumb/{ids['train/rgb16.tif']}?s=128", f"/api/img/{ids['train/rgb16.tif']}"):
            r = c.get(path)
            assert r.status_code == 200 and r.headers["content-type"] == "image/jpeg", path
            red = Image.open(io.BytesIO(r.content)).convert("RGB").split()[0]
            assert red.getextrema()[1] > 200, path                     # not the 0..15 high byte
        r = c.get(f"/api/thumb/{ids['train/cmyk.jpg']}", params={"s": 128})
        px = Image.open(io.BytesIO(r.content)).convert("RGB").getpixel((10, 10))
        assert px[0] > 150 and px[1] < 120                              # red-ish, not inverted
    finally:
        _drop(ds)


def test_zip_with_a_promised_length_aborts_if_a_file_changes_mid_download(tmp_path):
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "train/b.jpg": None})
    try:
        from fovea.core import export as ex
        prep = ex.prepare_zip(ds, ["person"], {}, ex.parse_options({}))
        (root / "images" / "train" / "b.jpg").unlink()
        with pytest.raises(RuntimeError, match="failed part-way"):
            b"".join(ex.stream_prepared(prep))
    finally:
        _drop(ds)


def test_copy_export_job_through_the_api_reports_counts(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "val/b.jpg": "1 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        out = tmp_path / "copy-out"
        pv = c.post(f"/api/datasets/{ds}/export/preview", json={"target_dir": str(out)}).json()
        assert pv["target"]["exists"] is False and pv["images"] == 2
        job = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(out), "class_map": {"1": 0}}).json()["job"]
        from fovea.jobs import jobs
        j = jobs.get(job["id"])
        jobs.wait([j], 20)
        assert j.status == "done", j.error
        assert j.result["images"] == 2 and j.result["split_counts"] == {"train": 1, "val": 1}
        assert (out / "labels" / "val" / "b.txt").read_text().startswith("0 ")
        assert c.get("/api/jobs", params={"active": 1, "dataset_id": ds}).json()["jobs"] == []
        r = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(_root)})
        assert r.status_code == 400 and "overlaps" in r.json()["detail"]
    finally:
        _drop(ds)


def test_secret_name_detection_spares_ordinary_settings():
    for name in ("api_key", "anthropicApiKey", "hf_token", "token", "password", "access_key", "secret"):
        assert security.is_secret_name(name), name
    for name in ("max_tokens", "token_budget", "sort_key", "hotkeys", "keyboard", "theme"):
        assert not security.is_secret_name(name), name
    assert security.find_secrets({"ui": {"max_tokens": 512, "nested": [{"x": "AIzaSyA1234567890abcdefghijk"}]}}) == ["ui.nested[0].x"]


# ---------------------------------------------------------------- wave 2a: dataset card

def _card_dataset(tmp_path):
    spec = {}
    for v in range(4):
        for f in range(3):
            spec[f"train/v{v}_frame_{f:03d}.jpg"] = f"{v % 2} 0.5 0.5 0.2 0.2\n"
    spec["val/v9_frame_000.jpg"] = "1 0.5 0.5 0.2 0.2\n0 0.2 0.2 0.1 0.1\n"
    spec["test-1854-포함/x.jpg"] = None
    spec["train/bad.jpg"] = "0 1.5 0.5 0.2 0.2\n"
    ds, root = _labeled(tmp_path, spec, classes=("person", "넘어짐 | fall"))
    ids = _ids(ds)
    db.execute("INSERT INTO class_cards(dataset_id, cls, description, include, exclude, prompt, examples, updated_at)"
               " VALUES(?,?,?,?,?,?,?,?)", (ds, 1, "A person lying on the floor\nafter a fall | not sitting", "lying",
                                            "sitting *on purpose*", "fallen person", "[]", db.now()))
    _flag_dups(ds, {3: [ids["train/v0_frame_000.jpg"], ids["val/v9_frame_000.jpg"]]})
    db.execute("INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
               (ids["train/v1_frame_001.jpg"], ds, "blurry", 1.0, "{}"))
    for rel, st in (("train/v0_frame_000.jpg", "approved"), ("train/v1_frame_000.jpg", "flagged")):
        db.execute("INSERT INTO reviews(dataset_id, rel_path, status, note, updated_at) VALUES(?,?,?,?,?)", (ds, rel, st, "", db.now()))
    run = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, created_at) VALUES(?,?,?,?,?)",
                     (ds, "audit", "mock", "m", db.now())).lastrowid
    for kind, rel in (("swap", "train/v2_frame_000.jpg"), ("detect", "train/v3_frame_000.jpg")):
        db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind, state, created_at)"
                   " VALUES(?,?,?,?,?,?,?,?,?,?,?)", (run, ds, ids[rel], 0, .5, .5, .1, .1, kind, "pending", db.now()))
    return ds, root, ids


def _md_tables_are_well_formed(text: str):
    """Every row of every Markdown table has as many cells as its header (escaped pipes do not split cells)."""
    import re
    n = None
    for line in text.splitlines():
        if line.startswith("|"):
            cells = len(re.split(r"(?<!\\)\|", line)) - 2
            if n is None:
                n = cells
            assert cells == n, line
        else:
            n = None


def test_dataset_card_standalone_describes_the_dataset(tmp_path):
    ds, root, _ids_ = _card_dataset(tmp_path)
    try:
        c = _client()
        r = c.get(f"/api/datasets/{ds}/card.md")
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/markdown")
        assert "attachment" in r.headers["content-disposition"] and "DATASET_CARD.md" in r.headers["content-disposition"]
        md = r.text
        _md_tables_are_well_formed(md)
        assert md.startswith(f"# {ds} — dataset card")
        # splits in train, val, test order with their data.yaml key; the Korean split maps to test
        rows = [ln for ln in md.splitlines() if ln.startswith(("| train |", "| val |", "| test-1854-포함 |"))]
        assert rows[0].startswith("| train | train | 13 |") and rows[1].startswith("| val | val | 1 |")
        assert rows[2].startswith("| test-1854-포함 | test | 1 |")
        # classes with box/image counts and the class card text (escaped for tables)
        assert "| 1 | 넘어짐 \\| fall | 7 |" in md and "A person lying on the floor after a fall \\| not sitting" in md
        assert "- Excludes: sitting \\*on purpose\\*" in md and "- AI prompt: fallen person" in md
        # data quality, review progress, label issues, AI findings
        assert "| Near-duplicates across splits (leak) | 2 |" in md and "| Blurry image | 1 |" in md
        assert "2 of 15 images reviewed" in md and "| Approved | 1 |" in md and "| Flagged | 1 |" in md
        assert "| Box outside image | 1 |" in md and "AI: possibly wrong class" in md
        assert "1 image with AI-suggested boxes waiting for review" in md
        assert "## Export settings" not in md and "## Selection" not in md
        # filters narrow the card and are listed; inline view; unknown dataset
        r = c.get(f"/api/datasets/{ds}/card.md", params={"split": "val", "inline": 1})
        assert r.status_code == 200 and r.headers["content-type"].startswith("text/plain")
        assert r.headers["content-disposition"].startswith("inline")
        assert "## Selection" in r.text and "- Split: val" in r.text and "| Images | 1 (1 with a label file) |" in r.text
        assert c.get("/api/datasets/nope/card.md").status_code == 404
    finally:
        _drop(ds)


def test_dataset_card_goes_into_zip_and_copy_exports_only_when_asked(tmp_path):
    from fovea.core import export as ex_mod
    ds, root, ids = _card_dataset(tmp_path)
    body = {"dataset_card": True, "class_map": {"0": 1, "1": 0}, "drop_near_dups": True,
            "split_builder": {"ratios": {"train": 70, "val": 30}, "seed": 5}, "filename": "with-card"}
    try:
        c = _client()
        link = c.post(f"/api/datasets/{ds}/export/zip-link", json=body).json()
        r = c.get(link["url"])
        assert int(r.headers["content-length"]) == len(r.content) == link["bytes"]       # the card is in the length
        z = zipfile.ZipFile(io.BytesIO(r.content))
        assert z.testzip() is None and "DATASET_CARD.md" in z.namelist()
        md = z.read("DATASET_CARD.md").decode("utf-8")
        _md_tables_are_well_formed(md)
        assert "## Export settings" in md and "remapped 0 → 1, 1 → 0" in md
        assert "rebuilt by Fovea — train 70% · val 30%; seed 5; kept together: whole sequences" in md
        assert "Near-duplicates: dropped" in md and "1 image removed" in md
        assert "| 0 | 넘어짐 \\| fall | 1 |" in md               # remapped class keeps its card, source id listed
        assert "- Selection: every image of the dataset — 15 images matched" in md
        cp = ex_mod.class_plan(["a", "b", "c"], ex_mod.parse_options({"class_map": {"1": None}}))
        assert cp.mapping[1] is None
        assert json.loads(z.read("fovea-export.json"))["options"]["dataset_card"] is True
        img_total = sum(1 for n in z.namelist() if n.startswith("images/"))
        assert f"| Images | {img_total} (" in md
        pv = c.post(f"/api/datasets/{ds}/export/preview", json={"dataset_card": True, "class_map": {"0": None}}).json()
        assert "| 0 | person | – (dropped) | 0 |" in pv["card"] and "dropped 0" in pv["card"]
        # without the option: no card
        names = _zip_names(c.get(c.post(f"/api/datasets/{ds}/export/zip-link", json={}).json()["url"]).content)
        assert "DATASET_CARD.md" not in names
        # the streamed (resized) ZIP carries it too
        from fovea.core import export as ex
        data = b"".join(ex.stream_zip(ds, ["person", "넘어짐 | fall"], {}, opts=ex.parse_options(
            {"dataset_card": True, "resize": {"mode": "max", "size": 32}})))
        md2 = zipfile.ZipFile(io.BytesIO(data)).read("DATASET_CARD.md").decode()
        assert "longest side at most 32 px" in md2
        # preview returns the card text for the UI
        pv = c.post(f"/api/datasets/{ds}/export/preview", json=body).json()
        assert pv["card"].startswith(f"# {ds} — dataset card") and "## Export settings" in pv["card"]
        assert c.post(f"/api/datasets/{ds}/export/preview", json={}).json()["card"] is None
        # copy: the card is written beside data.yaml, listed in the manifest, and removed by a later
        # overwrite that does not ask for it
        out = tmp_path / "copy-card"
        from fovea.jobs import jobs
        job = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(out), "card": True}).json()["job"]
        j = jobs.get(job["id"])
        jobs.wait([j], 20)
        assert j.status == "done", j.error
        assert (out / "DATASET_CARD.md").is_file() and "DATASET_CARD.md" in json.loads((out / "fovea-export.json").read_text())["files"]
        assert "## Export settings" in (out / "DATASET_CARD.md").read_text(encoding="utf-8")
        job = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(out), "overwrite": True}).json()["job"]
        j = jobs.get(job["id"])
        jobs.wait([j], 20)
        assert j.status == "done" and not (out / "DATASET_CARD.md").exists()
    finally:
        _drop(ds)


def test_dataset_card_counts_leave_out_images_the_export_could_not_read(tmp_path):
    ds, root, ids = _card_dataset(tmp_path)
    try:
        (root / "images" / "train" / "v3_frame_002.jpg").unlink()
        from fovea.core import export as ex
        prep = ex.prepare_zip(ds, ["person", "넘어짐 | fall"], {}, ex.parse_options({"dataset_card": True}))
        assert prep.result["missing"] == 1
        assert "| Images | 14 (" in prep.card_text and "| Not exported | 1 missing, 0 unreadable" in prep.card_text
        data = b"".join(ex.stream_prepared(prep))
        assert len(data) == prep.length
    finally:
        _drop(ds)


def test_leaks_are_judged_on_the_data_yaml_split(tmp_path):
    """Two source folders feeding data.yaml's `test` (vfp290k: test + test-1854-1860-포함) sharing a video is
    not a leak; a video in train and val is."""
    spec = {f"test/v5_frame_{f:03d}.jpg": "0 0.5 0.5 0.2 0.2\n" for f in range(3)}
    spec.update({f"test-1854-포함/v5_frame_{f:03d}.jpg": "0 0.5 0.5 0.2 0.2\n" for f in range(3, 6)})
    spec.update({f"train/v7_frame_{f:03d}.jpg": "1 0.5 0.5 0.2 0.2\n" for f in range(3)})
    spec.update({f"val/v7_frame_{f:03d}.jpg": "1 0.5 0.5 0.2 0.2\n" for f in range(3, 6)})
    ds, _root = _labeled(tmp_path, spec)
    try:
        ids = _ids(ds)
        _flag_dups(ds, {1: [ids["test/v5_frame_000.jpg"], ids["test-1854-포함/v5_frame_003.jpg"]],
                        2: [ids["train/v7_frame_000.jpg"], ids["val/v7_frame_003.jpg"]]})
        c = _client()
        pv = c.post(f"/api/datasets/{ds}/export/preview", json={}).json()
        assert pv["sequences"] == {"total": 2, "spanning_splits": 1}
        md = c.get(f"/api/datasets/{ds}/card.md").text
        assert "| Sequences spanning splits | 6 | 1 of 2 sequences" in md
        assert "| Near-duplicates across splits (leak) | 2 | 1 group with members in more than one split |" in md
        assert "| Near-duplicate images | 4 | 2 groups of look-alike images |" in md
    finally:
        _drop(ds)


def test_card_markdown_escaping():
    from fovea.core import export as ex
    assert ex._md("a | b\n*c* `d` <e>") == "a \\| b \\*c\\* \\`d\\` \\<e>"
    assert ex._para("# not a heading") == "\\# not a heading" and ex._para("1. item") == "1\\. item"
    assert ex._code("C:\\data|x") == "`C:\\data\\|x`" and ex._code("a`b") == "`` a`b ``"
    assert ex._count(1, "group") == "1 group" and ex._count(2, "group") == "2 groups"


# ---------------------------------------------------------------- wave 2a: access log regression

def test_uvicorn_access_log_survives_the_redaction_filter(capsys):
    """Regression: a redaction filter that replaced record.args with None made uvicorn's AccessFormatter
    fail with 'cannot unpack non-iterable NoneType object' on every request ('--- Logging error ---')."""
    import logging
    from uvicorn.logging import AccessFormatter
    stream = io.StringIO()
    handler = logging.StreamHandler(stream)
    handler.setFormatter(AccessFormatter('%(levelprefix)s %(client_addr)s - "%(request_line)s" %(status_code)s',
                                         use_colors=False))
    lg = logging.getLogger("uvicorn.access")
    old = (lg.handlers[:], lg.level, lg.propagate)
    lg.handlers[:] = [handler]
    lg.setLevel(logging.INFO)
    lg.propagate = False
    try:
        security.install_log_redaction()
        assert security._filter in handler.filters
        security.install_log_redaction()                     # idempotent: one filter per handler
        assert handler.filters.count(security._filter) == 1
        # exactly how uvicorn's protocols log a request
        lg.info('%s - "%s %s HTTP/%s" %d', "127.0.0.1:5000", "GET", "/api/meta?key=AIzaSyA1234567890abcdefghijk", "1.1", 200)
        lg.info('%s - "%s %s HTTP/%s" %d', "", "POST", "/api/datasets", "1.1", 403)
    finally:
        lg.handlers[:], lg.level, lg.propagate = old
    out = stream.getvalue()
    err = capsys.readouterr().err
    assert "Logging error" not in err and "TypeError" not in err, err
    assert 'GET /api/meta?key=… HTTP/1.1" 200' in out and 'POST /api/datasets HTTP/1.1" 403' in out
    assert "AIzaSyA1234567890" not in out
    # whatever shape args has, the filter keeps it (None stays None, a tuple stays a tuple, a dict a dict)
    import logging as _l
    for args in ((), ("x",), ({"a": "sk-ant-api03-zzzzzzzzzzzz"},)):
        rec = _l.LogRecord("x", _l.INFO, __file__, 1, "m %s" if args else "m", args, None)
        before = type(rec.args)
        assert security.RedactingFilter().filter(rec) and type(rec.args) is before
    rec = _l.makeLogRecord({"msg": "plain", "args": None})
    assert security.RedactingFilter().filter(rec) and rec.args is None and rec.getMessage() == "plain"


def _free_port() -> int:
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


@pytest.mark.parametrize("mode", ["module", "uvicorn", "module-reload"])
def test_server_run_modes_log_requests_without_logging_errors(tmp_path, mode):
    """End to end in a real process: `python -m fovea`, the uvicorn CLI and --reload. Every request gets an
    access line, no '--- Logging error ---' / traceback appears, keys in URLs are scrubbed and Ctrl+C exits
    cleanly (python -m fovea used to end with a KeyboardInterrupt traceback)."""
    import signal
    import subprocess
    import sys
    import urllib.error
    import urllib.request
    port = _free_port()
    src = Path(__file__).resolve().parent.parent
    data = tmp_path / "data"
    cmd = {"module": [sys.executable, "-m", "fovea", "--port", str(port), "--data-dir", str(data)],
           "uvicorn": [sys.executable, "-m", "uvicorn", "fovea.main:app", "--port", str(port)],
           "module-reload": [sys.executable, "-m", "fovea", "--port", str(port), "--data-dir", str(data), "--reload"]}[mode]
    env = dict(os.environ, FOVEA_DATA_DIR=str(data), PYTHONUNBUFFERED="1")
    env.pop("FOVEA_TOKEN", None)
    log = tmp_path / "server.log"
    with open(log, "w") as fh:
        p = subprocess.Popen(cmd, cwd=src, env=env, stdout=fh, stderr=subprocess.STDOUT, start_new_session=True)

    def get(path, headers=None):
        req = urllib.request.Request(f"http://127.0.0.1:{port}{path}", headers=headers or {})
        try:
            with urllib.request.urlopen(req, timeout=10) as r:
                return r.status
        except urllib.error.HTTPError as e:
            return e.code

    try:
        t0 = time.time()
        while True:
            try:
                if get("/api/meta") == 200:
                    break
            except OSError:
                pass
            assert p.poll() is None and time.time() - t0 < 60, log.read_text()
            time.sleep(0.2)
        assert get("/") == 200
        assert get("/api/datasets/nope") == 404
        assert get("/api/meta", {"Host": "evil.example"}) == 400
        assert get("/api/meta?key=AIzaSyA1234567890abcdefghijk") == 200
        time.sleep(0.3)
    finally:
        os.killpg(p.pid, signal.SIGINT)
        try:
            p.wait(20)
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGKILL)
            p.wait()
    text = log.read_text(encoding="utf-8", errors="replace")
    assert "Logging error" not in text and "Traceback" not in text and "TypeError" not in text, text
    assert '"GET / HTTP/1.1" 200' in text and '"GET /api/datasets/nope HTTP/1.1" 404' in text
    assert '"GET /api/meta HTTP/1.1" 400' in text and "AIzaSyA1234567890" not in text
    assert p.returncode == 0, (p.returncode, text[-2000:])


# ---------------------------------------------------------------- integer range (SQLite INTEGER is 64-bit)

BIG = 2 ** 63               # one past SQLite's largest INTEGER: binding it used to be a 500 OverflowError


def test_out_of_range_integers_are_404_or_422_never_500(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "train/b.jpg": "1 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        iid = _ids(ds)["train/a.jpg"]
        # A path id that cannot exist is "not found", exactly like the largest one that can.
        for url in (f"/api/img/{BIG}", f"/api/thumb/{BIG}", f"/api/crop/{BIG}?box=0.5,0.5,0.1,0.1",
                    f"/api/datasets/{ds}/images/{BIG}", f"/api/datasets/{ds}/images/{BIG}/labels",
                    f"/api/datasets/{ds}/snapshots/{10 ** 30}", f"/api/ai/runs/{10 ** 30}"):
            assert c.get(url).status_code == 404, url
        assert c.get(f"/api/img/{BIG - 1}").status_code == 404
        assert c.delete(f"/api/ai/runs/{BIG}").status_code == 404
        assert c.post(f"/api/datasets/{ds}/images/{BIG}/fill-gap", json={}).status_code == 404
        # A query parameter out of range is the 422 FastAPI gives for Query(le=...), loc and all.
        r = c.get(f"/api/datasets/{ds}/images?offset={BIG}")
        assert r.status_code == 422, r.text
        err = r.json()["detail"][0]
        assert err["loc"] == ["query", "offset"] and err["type"] == "less_than_equal" and err["ctx"]["le"] == BIG - 1
        r = c.get(f"/api/datasets/{ds}/images/neighbor?id={-BIG - 1}")
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == "greater_than_equal"
        assert c.get(f"/api/datasets/{ds}/images/position?id={10 ** 25}").status_code == 422
        # Numbers inside filter strings are clamped where they are parsed: they simply match nothing.
        for q in (f"cls={10 ** 30}", f"only_cls={10 ** 30}", f"ids={10 ** 30}", f"min_boxes={10 ** 30}",
                  f"max_boxes=-{10 ** 30}", f"cls=0,{10 ** 30}&labeled=empty"):
            r = c.get(f"/api/datasets/{ds}/images?{q}")
            assert r.status_code == 200 and r.json()["total"] == 0, (q, r.text)
        assert c.get(f"/api/datasets/{ds}/images?cls=0,{10 ** 30}").json()["total"] == 1
        assert c.get(f"/api/datasets/{ds}/images/ids?cls={10 ** 30}").json()["ids"] == []
        assert c.get(f"/api/datasets/{ds}/boxes?cls={10 ** 30}").json()["total"] == 0
        assert c.get(f"/api/datasets/{ds}/boxes?cls=²").status_code == 200
        assert c.get(f"/api/datasets/{ds}/boxes?sort=area&after=[0.5,{10 ** 30}]").status_code == 400
        r = c.post(f"/api/datasets/{ds}/export/preview", json={"filters": {"cls": str(10 ** 30)}})
        assert r.status_code == 200 and r.json()["images"] == 0
        # JSON bodies: every number must fit too (an int, or a whole float that int() would accept).
        for url, body, loc in (
                (f"/api/datasets/{ds}/interpolate", {"from_image_id": 10 ** 30, "to_image_id": iid}, ["body", "from_image_id"]),
                (f"/api/datasets/{ds}/interpolate", {"from_image_id": iid, "to_image_id": 1e30}, ["body", "to_image_id"]),
                (f"/api/datasets/{ds}/proposals/decide", {"accept": [10 ** 30], "reject": []}, ["body", "accept", 0]),
                ("/api/ai/propose", {"dataset_id": ds, "image_id": 10 ** 30, "provider": "yolo", "model": "x"}, ["body", "image_id"]),
                (f"/api/datasets/{ds}/export/preview", {"filters": {"min_boxes": -(10 ** 30)}}, ["body", "filters", "min_boxes"])):
            r = c.post(url, json=body)
            assert r.status_code == 422 and r.json()["detail"][0]["loc"] == loc, (url, r.text)
        r = c.post(f"/api/datasets/{ds}/interpolate", content=b'{"from_image_id": Infinity, "to_image_id": 1}',
                   headers={"content-type": "application/json"})
        assert r.status_code == 422 and r.json()["detail"][0]["type"] == "finite_number"
        # In range, nothing changes.
        assert c.get(f"/api/datasets/{ds}/images?offset=0&limit=1").json()["items"][0]["id"] in _ids(ds).values()
        assert c.post(f"/api/datasets/{ds}/export/preview", json={"filters": {"min_boxes": 1}}).json()["images"] == 2
    finally:
        _drop(ds)


def test_no_route_answers_500_to_an_out_of_range_integer(tmp_path):
    """Every GET route (and DELETE by id) of the app and its plugins that declares an integer parameter,
    driven with 2^63 and -2^63-1: a route added later is covered without being listed here."""
    import re
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        spec = c.get("/api/openapi.json").json()

        def is_int(s):
            return bool(s) and (s.get("type") == "integer" or any(x.get("type") == "integer" for x in s.get("anyOf", [])))

        tried, bad = 0, []
        for path, ops in spec["paths"].items():
            for method, op in ops.items():
                if method not in ("get", "delete"):
                    continue
                params = op.get("parameters", [])
                base = {p["name"]: (ds if p["name"] in ("ds_id", "dataset_id", "ds") else "1")
                        for p in params if p["in"] == "path"}
                for p in params:
                    if not is_int(p.get("schema")) or (method == "delete" and p["in"] != "path"):
                        continue
                    for v in (str(BIG), str(-BIG - 1)):
                        vals = dict(base, **({p["name"]: v} if p["in"] == "path" else {}))
                        url = re.sub(r"\{(\w+)(:path)?\}", lambda m: vals.get(m.group(1), "x"), path)
                        if p["in"] == "query":
                            url += f"?{p['name']}={v}"
                        r = c.request(method.upper(), url)
                        tried += 1
                        if r.status_code >= 500:
                            bad.append((method, url, r.status_code, r.text[:120]))
                        elif p["in"] == "path":
                            assert r.status_code == 404, (url, r.status_code)
        assert tried > 30 and not bad, bad
    finally:
        _drop(ds)


# ---------------------------------------------------------------- jobs under concurrency

def test_job_list_and_queue_positions_survive_concurrent_submits():
    """queue_position() iterated the live jobs dict: a submit() on another thread made to_dict() — /api/jobs,
    the SSE stream, every dataset GET — raise 'dictionary changed size during iteration'."""
    import sys
    import threading
    m = JobManager(workers={"default": 1})
    gate = threading.Event()
    m.submit("block", lambda j: gate.wait(30))
    errors, stop = [], time.time() + 1.5
    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)

    def submitter():
        while time.time() < stop:
            m.submit("x", lambda j: None)

    def reader():
        while time.time() < stop:
            try:
                for j in m.list(active=True)[:20]:
                    j.to_dict()
            except RuntimeError as e:
                errors.append(str(e))
                return

    try:
        ts = [threading.Thread(target=submitter) for _ in range(2)] + [threading.Thread(target=reader) for _ in range(2)]
        for t in ts:
            t.start()
        for t in ts:
            t.join()
    finally:
        sys.setswitchinterval(old)
        gate.set()
        m.shutdown(1)
    assert not errors, errors[:2]


# ---------------------------------------------------------------- ZIP planning cost

def test_zip_length_arithmetic_matches_zipfile_including_zip64(monkeypatch):
    """The Content-Length of a prepared ZIP is computed, not dry-run: it must equal what zipfile writes for
    every shape — unicode names, empty members, ZIP64 local headers, sizes and offsets past the limit, and
    the ZIP64 end records for many members. Small limits stand in for 4 GB / 65,535 entries."""
    from fovea.core import export as ex

    def real(members):
        sink = ex._CountingFile()
        with zipfile.ZipFile(sink, "w", zipfile.ZIP_STORED, allowZip64=True) as zf:
            for name, size in members:
                ex._zip_put_bytes(zf, name, bytes(size), None)
        return sink.tell()

    cases = [[], [("data.yaml", 0)], [("images/학습/a.jpg", 1234), ("labels/학습/a.txt", 31), ("é" * 200 + ".png", 7)]]
    assert ex._length_model_ok()
    for members in cases:
        assert ex.zip_length(members) == real(members), members
    monkeypatch.setattr(zipfile, "ZIP64_LIMIT", 5000)
    monkeypatch.setattr(zipfile, "ZIP_FILECOUNT_LIMIT", 6)
    shapes = [
        [("a.jpg", 4800)],                                   # size*1.05 > limit: ZIP64 local header only
        [("a.jpg", 5001)],                                   # ZIP64 sizes in the central directory too
        [(f"i/{k}.jpg", 1500) for k in range(5)],            # header offsets past the limit
        [(f"i/{k}.jpg", 10) for k in range(9)],              # more members than the count limit
        [("big.bin", 9000)] + [(f"l/{k}.txt", 3) for k in range(8)],
    ]
    for members in shapes:
        assert ex.zip_length(members) == real(members), members


def test_zip_plans_reuse_rendered_labels_and_notice_every_change(tmp_path, monkeypatch):
    from fovea.core import export as ex
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n", "train/b.jpg": "1 0.5 0.5 0.2 0.2 0.9\n",
                                   "val/c.jpg": "0 0.1 0.1 0.1 0.1\n"})
    try:
        calls = []
        real = ex.transform_label_text
        monkeypatch.setattr(ex, "transform_label_text", lambda *a, **k: calls.append(1) or real(*a, **k))
        ex.RENDERED_LABELS.clear()
        names = ["person", "fall"]
        p1 = ex.prepare_zip(ds, names, {}, ex.parse_options({}))
        assert len(calls) == 3 and p1.result["label_rows"]["conf_stripped"] == 1
        # A second Download click reuses the whole plan; another selection reuses the rendered labels.
        assert ex.prepare_zip(ds, names, {}, ex.parse_options({})) is p1
        p_val = ex.prepare_zip(ds, names, {"split": "train"}, ex.parse_options({}))
        assert p_val is not p1 and len(calls) == 3 and p_val.result["images"] == 2
        # A label edited outside Fovea: new identity, so the plan is rebuilt and only that file is read again.
        lbl = root / "labels" / "train" / "a.txt"
        lbl.write_text("1 0.25 0.25 0.1 0.1\n0 0.5 0.5 0.2 0.2\n")
        p2 = ex.prepare_zip(ds, names, {}, ex.parse_options({}))
        assert p2 is not p1 and len(calls) == 4
        data = b"".join(ex.stream_prepared(p2))
        assert len(data) == p2.length
        assert zipfile.ZipFile(io.BytesIO(data)).read("labels/train/a.txt").decode().startswith("1 0.250000 0.250000")
        # An image whose size changed invalidates the plan (its announced length would be wrong).
        _img(root / "images" / "val" / "c.jpg", (96, 80))
        p3 = ex.prepare_zip(ds, names, {}, ex.parse_options({}))
        assert p3 is not p2 and len(b"".join(ex.stream_prepared(p3))) == p3.length
        # A label write through Fovea changes the index version: a new plan.
        c = _client()
        iid = _ids(ds)["train/b.jpg"]
        cur = c.get(f"/api/datasets/{ds}/images/{iid}/labels").json()
        r = c.put(f"/api/datasets/{ds}/images/{iid}/labels",
                  json={"boxes": [[0, 0.5, 0.5, 0.3, 0.3]], "version": cur.get("version")})
        assert r.status_code == 200, r.text
        p4 = ex.prepare_zip(ds, names, {}, ex.parse_options({}))
        assert p4 is not p3
        z = zipfile.ZipFile(io.BytesIO(b"".join(ex.stream_prepared(p4))))
        assert z.read("labels/train/b.txt").decode() == "0 0.500000 0.500000 0.300000 0.300000\n"
        # Different options are a different plan.
        assert ex.prepare_zip(ds, names, {}, ex.parse_options({"class_map": {"0": 1, "1": 0}})) is not p4
    finally:
        _drop(ds)


def test_zip_link_planning_stays_fast_on_a_large_dataset(tmp_path):
    """3,000 labelled images: the first plan reads each label once, a repeat click only stats the files."""
    from fovea.core import export as ex
    root = tmp_path / "many"
    src = _img(tmp_path / "one.jpg", (32, 24))
    for k in range(3000):
        split = ("train", "val", "test")[k % 3]
        d = root / "images" / split
        d.mkdir(parents=True, exist_ok=True)
        os.link(src, d / f"f{k:05d}.jpg")
        (root / "labels" / split).mkdir(parents=True, exist_ok=True)
        (root / "labels" / split / f"f{k:05d}.txt").write_text(f"{k % 2} 0.5 0.5 0.2 0.2\n1 0.3 0.3 0.1 0.1\n")
    ds = _register(root, ["a", "b"])
    try:
        c = _client()
        ex.RENDERED_LABELS.clear()
        t = time.perf_counter()
        first = c.post(f"/api/datasets/{ds}/export/zip-link", json={}).json()
        t_first = time.perf_counter() - t
        t = time.perf_counter()
        again = c.post(f"/api/datasets/{ds}/export/zip-link", json={"filename": "again"}).json()
        t_again = time.perf_counter() - t
        assert first["images"] == again["images"] == 3000 and first["bytes"] == again["bytes"]
        assert t_again < max(0.5, t_first / 2), (t_first, t_again)
        r = c.get(again["url"])
        assert r.status_code == 200 and len(r.content) == again["bytes"] == int(r.headers["content-length"])
        assert zipfile.ZipFile(io.BytesIO(r.content)).testzip() is None
    finally:
        _drop(ds)


def test_slow_storage_is_planned_in_parallel_with_the_same_result(tmp_path, monkeypatch):
    """On a network share (first chunk slow) the files are stat'ed and read by a thread pool, in plan order."""
    from fovea.core import export as ex
    spec = {f"train/{k:03d}.jpg": f"{k % 2} 0.5 0.5 0.2 0.2\n" for k in range(600)}      # three chunks
    ds, root = _labeled(tmp_path, spec)
    try:
        names, opts = ["person", "fall"], ex.parse_options({})
        serial = ex._prepare_zip(ds, names, {}, opts, True)
        pools = []

        class Pool(ex.ThreadPoolExecutor):
            def __init__(self, *a, **k):
                pools.append(1)
                super().__init__(*a, **k)

        monkeypatch.setattr(ex, "ThreadPoolExecutor", Pool)
        monkeypatch.setattr(ex, "_SLOW_FILE_S", -1.0)          # every storage counts as slow
        ex.RENDERED_LABELS.clear()
        par = ex._prepare_zip(ds, names, {}, opts, True)
        assert pools and par.length == serial.length
        assert [(it.img_rel, it.size, it.label_text) for it in par.plan.items] == \
               [(it.img_rel, it.size, it.label_text) for it in serial.plan.items]
        assert ex._files_unchanged(par.plan)
        (root / "images" / "train" / "599.jpg").unlink()
        assert not ex._files_unchanged(par.plan)
    finally:
        _drop(ds)


# ---------------------------------------------------------------- round 2: remote cancel, export input, own folders

def _economy_run(c, ds, monkeypatch):
    """A running Economy (provider batch) run of the mock provider, which the test declares a cloud provider."""
    from fovea.ai import economy, registry, store
    real = registry.is_cloud
    monkeypatch.setattr(registry, "is_cloud", lambda pid: pid == "mock" or real(pid))
    monkeypatch.setattr(economy, "POLL_S", 0.1)
    r = c.post("/api/ai/batch", json={"dataset_id": ds, "filters": {}, "provider": "mock", "model": "mock-fast",
                                      "target": "proposals", "speed": "economy", "options": {"mock_batch_delay_s": 60}})
    assert r.status_code == 200, r.text
    run_id, job_id = r.json()["run_id"], r.json()["job"]["id"]
    t0 = time.time()
    while not store.get_run(run_id)["params"].get("provider_batches") and time.time() - t0 < 10:
        time.sleep(0.05)
    return run_id, job_id, store.get_run(run_id)["params"]["batch_id"]


def test_remote_clients_cannot_cancel_the_owners_provider_batch_by_any_route(tmp_path, monkeypatch):
    """/api/ai/runs/{id}/cancel refused a remote client, but /api/jobs/{id}/cancel and DELETE dataset cancelled the
    same Economy batch at the provider with the owner's key. Relocate does not cancel it (it pauses the run and
    reconnects to the batch with the owner's key after the rescan) and is refused for that reason — its message used
    to claim it stopped the batch."""
    from fovea.ai import registry
    from fovea.ai.providers.mock import PROVIDER
    from fovea.api.jobs import RECONNECTS_BATCH, STOPS_BATCH
    from fovea.jobs import jobs
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    registry.reset()
    ds, root = _labeled(tmp_path, {f"train/{k}.jpg": "0 0.5 0.5 0.2 0.2\n" for k in range(3)})
    try:
        c = _client()
        remote = _client_from(("192.168.1.50", 50000))
        run_id, job_id, bid = _economy_run(c, ds, monkeypatch)
        assert c.get(f"/api/ai/runs/{run_id}").json()["run"]["status"] == "running"
        assert remote.post(f"/api/ai/runs/{run_id}/cancel").status_code == 403      # the earlier guard
        r = remote.post(f"/api/jobs/{job_id}/cancel")
        assert r.status_code == 403 and r.json()["detail"] == f"Cancel: {STOPS_BATCH} {security.REMOTE_AI_MESSAGE}"
        r = remote.delete(f"/api/datasets/{ds}")
        assert r.status_code == 403 and r.json()["detail"] == f"Delete: {STOPS_BATCH} {security.REMOTE_AI_MESSAGE}"
        r = remote.post(f"/api/datasets/{ds}/relocate", json={"path": str(root)})
        assert r.status_code == 403, r.text
        assert r.json()["detail"] == f"Relocate: {RECONNECTS_BATCH} {security.REMOTE_AI_MESSAGE}"
        assert "pauses an Economy batch and reconnects to it" in r.json()["detail"]
        assert "stops" not in r.json()["detail"]                    # relocate never cancels the batch
        assert jobs.get(job_id).status == "running" and c.get(f"/api/datasets/{ds}").status_code == 200
        assert PROVIDER.batch_status(bid)["counts"]["canceled"] == 0                  # nothing reached the provider
        # other jobs stay cancellable from anywhere: the guard is about the provider call only
        other = jobs.submit("scan", lambda j: [time.sleep(0.02) or j.check() for _ in range(500)], dataset_id=ds)
        assert remote.post(f"/api/jobs/{other.id}/cancel").status_code == 200
        assert jobs.wait([other], 5) and other.status == "cancelled"
        # a synchronous (non-Economy) AI job is not guarded either: cancelling it only stops spending
        from fovea.api.jobs import provider_batch_of
        assert provider_batch_of(other) is None and provider_batch_of(jobs.get(job_id)) == "mock"
        # the owner (or a server started with FOVEA_ALLOW_REMOTE_AI=1) still can
        monkeypatch.setenv("FOVEA_ALLOW_REMOTE_AI", "1")
        assert remote.post(f"/api/jobs/{job_id}/cancel").status_code == 200
        assert jobs.wait([jobs.get(job_id)], 20) and jobs.get(job_id).status == "cancelled"
        assert PROVIDER.batch_status(bid)["counts"]["canceled"] == 3
    finally:
        _drop(ds)
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))
        registry.reset()


def test_relocate_never_cancels_an_economy_batch_even_while_its_job_is_queued(tmp_path, monkeypatch):
    """Relocate detaches Economy runs and reconnects to their provider batches after the rescan. A re-attached job
    still queued in the ai_wait pool (all workers busy) is cancelled while queued, and 'cancelled before it started'
    used to finish its run: the job's unstarted-cancel hook — and any run view (the AI tab polls the runs list)
    shown during the rescan — cancelled the batch at the provider, while the job said "the provider batch keeps
    running"."""
    import threading
    from concurrent.futures import ThreadPoolExecutor
    from fovea.ai import registry, store
    from fovea.ai.providers.mock import PROVIDER
    from fovea.api import datasets as api_ds
    from fovea.jobs import jobs
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    registry.reset()
    ds, root = _labeled(tmp_path, {f"train/{k}.jpg": "0 0.5 0.5 0.2 0.2\n" for k in range(3)})
    gate = threading.Event()
    pool = ThreadPoolExecutor(max_workers=1)
    try:
        c = _client()
        run_id, job_id, bid = _economy_run(c, ds, monkeypatch)
        views, reattach = [], api_ds._reattach_economy

        def polled_during_rescan(ds_id):                        # the UI polls the runs while the rescan runs
            views.append(c.get(f"/api/ai/runs/{run_id}").json()["run"]["status"])
            views.append(next(r["status"] for r in c.get("/api/ai/runs", params={"dataset_id": ds_id}).json()["runs"]
                              if r["id"] == run_id))
            reattach(ds_id)

        monkeypatch.setattr(api_ds, "_reattach_economy", polled_during_rescan)
        monkeypatch.setitem(jobs._pools, "ai_wait", pool)           # one worker, kept busy below
        jobs.submit("blocker", lambda j: gate.wait(60), dataset_id="", pool="ai_wait")

        def relocate():
            r = c.post(f"/api/datasets/{ds}/relocate", json={"path": str(root)})
            assert r.status_code == 200, r.text
            assert jobs.wait([jobs.get(r.json()["job"]["id"])], 20)
            return store.get_run(run_id)

        run = relocate()                                            # 1: the running job is paused
        old = jobs.get(job_id)
        assert old.status == "cancelled" and "provider batch keeps running" in old.message
        assert run["status"] == "running" and run["job_id"] != job_id
        queued = jobs.get(run["job_id"])                            # re-attached, waiting for a worker
        assert queued.status == "queued"
        assert PROVIDER.batch_status(bid)["counts"]["canceled"] == 0

        run = relocate()                                            # 2: the queued job is paused too
        assert queued.status == "cancelled" and "provider batch keeps running" in queued.message
        time.sleep(0.3)                                             # the hook cancelled from a thread
        counts = PROVIDER.batch_status(bid)["counts"]
        assert counts["canceled"] == 0 and counts["processing"] == 3, counts
        assert run["status"] == "running" and run["job_id"] not in (job_id, queued.id)
        assert jobs.get(run["job_id"]).status == "queued"          # and re-attached again
        assert len(views) == 4 and "cancelled" not in views, views
    finally:
        gate.set()
        _drop(ds)
        pool.shutdown(wait=False)
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))
        registry.reset()


def test_export_drops_nan_and_inf_rows_like_the_index(tmp_path):
    from fovea.core.export import transform_label_text
    from fovea.core.labels import parse_label_rows
    text = ("0 nan 0.5 0.1 0.1\n1 0.5 inf 0.2 0.2\n0 0.4 0.4 0.1 0.1\n1 0.3 0.3 0.1 0.1 nan\n"
            "1 0.1 0.1 0.2 0.1 -inf 0.3 0.3\n1 0.1 0.1 0.2 0.1 0.2 0.3 0.3\n9999999999 0.5 0.5 0.1 0.1\n")
    out, st = transform_label_text(text)
    assert out.splitlines() == ["0 0.400000 0.400000 0.100000 0.100000", "1 0.300000 0.300000 0.100000 0.100000",
                                "1 0.1 0.1 0.2 0.1 0.2 0.3 0.3"]
    assert st.dropped_invalid == 4 and st.rows_out == 3 and st.conf_stripped == 1
    assert "nan" not in out.lower() and "inf" not in out.lower()
    rows, issues = parse_label_rows(text)
    assert len(rows) == st.rows_out and "bad_format" in issues          # the export and the index agree
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 nan 0.5 0.1 0.1\n0 0.4 0.4 0.1 0.1\n"})
    try:
        c = _client()
        prev = c.post(f"/api/datasets/{ds}/export/preview", json={}).json()
        assert any("leaves the unusable lines out" in w for w in prev["warnings"]), prev["warnings"]
        r = c.post(f"/api/datasets/{ds}/export/zip", json={})
        assert r.status_code == 200
        z = zipfile.ZipFile(io.BytesIO(r.content))
        assert z.read("labels/train/a.txt").decode() == "0 0.400000 0.400000 0.100000 0.100000\n"
    finally:
        _drop(ds)


def test_export_text_fields_must_be_strings(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        for bad in (-1, 5, 1.5, ["a"], {"a": 1}, True):
            for url in ("zip-link", "zip", "list"):
                r = c.post(f"/api/datasets/{ds}/export/{url}", json={"filename": bad})
                assert r.status_code == 400 and r.json()["detail"] == "filename must be a string", (bad, url, r.text)
            for url in ("copy", "preview"):
                r = c.post(f"/api/datasets/{ds}/export/{url}", json={"target_dir": bad})
                assert r.status_code == 400 and r.json()["detail"] == "target_dir must be a string", (bad, url, r.text)
        assert c.post(f"/api/datasets/{ds}/export/zip-link", json={"filename": "ok", "target_dir": None}).status_code == 200
        assert c.post(f"/api/datasets/{ds}/export/preview", json={"target_dir": None}).status_code == 200
    finally:
        _drop(ds)


def test_exports_never_write_into_foveas_own_folders(tmp_path, monkeypatch):
    """A copy into DATA_DIR made it a registrable dataset whose secrets.json the file endpoints then served."""
    from fovea import config
    data, model = tmp_path / "home" / "fovea-data", tmp_path / "home" / "fovea-model"
    for d in (data, model):
        d.mkdir(parents=True)
    (data / "secrets.json").write_text("{}")
    monkeypatch.setattr(config, "DATA_DIR", data)
    monkeypatch.setattr(config, "MODEL_DIR", model)
    ds, root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n"})
    try:
        c = _client()
        for target, word in ((data, "inside"), (data / "exports" / "x", "inside"), (model, "inside"),
                             (tmp_path / "home", "contains"), (tmp_path, "contains")):
            r = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(target), "overwrite": True})
            assert r.status_code == 400 and word in r.json()["detail"] and "Fovea's" in r.json()["detail"], (target, r.text)
            p = c.post(f"/api/datasets/{ds}/export/preview", json={"target_dir": str(target)}).json()
            assert "Fovea's" in p["target"]["error"], target
        link = tmp_path / "link-to-data"
        link.symlink_to(data)                                            # resolved before the comparison
        assert c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(link)}).status_code == 400
        assert sorted(os.listdir(data)) == ["secrets.json"]              # nothing was written
        ok = c.post(f"/api/datasets/{ds}/export/copy", json={"target_dir": str(tmp_path / "elsewhere")})
        assert ok.status_code == 200, ok.text
        # a dataset that lives in the data folder gets no image list or data.yaml written there either
        monkeypatch.setattr(config, "DATA_DIR", root)
        r = c.post(f"/api/datasets/{ds}/export/list", json={"write": True})
        assert r.status_code == 400 and "Fovea's own data folder" in r.json()["detail"], r.text
        r = c.post(f"/api/datasets/{ds}/export/yaml")
        assert r.status_code == 400 and "Fovea's own data folder" in r.json()["detail"], r.text
        assert not (root / "fovea-list.txt").exists()
        _settle(ds)
    finally:
        _drop(ds)


def test_host_allowlist_has_no_test_only_names(monkeypatch):
    """'testserver' was accepted in production; tests now allow it through FOVEA_ALLOWED_HOSTS (conftest.py)."""
    assert "testserver" not in security.LOOPBACK_NAMES
    monkeypatch.setattr(security, "_ALLOWED_HOSTS", [])
    assert not security.host_allowed("testserver:8000") and not security.host_allowed("testserver")
    c = _client()
    assert c.get("/api/meta", headers={"host": "testserver:8000"}).status_code == 400
    assert c.delete("/api/datasets/none", headers={"host": "testserver:8000",
                                                   "origin": "http://testserver:8000"}).status_code == 400
    assert c.get("/api/meta", headers={"host": "localhost:8000"}).status_code == 200


def test_api_docs_load_a_pinned_swagger_with_integrity_and_a_strict_csp():
    import base64
    import hashlib
    import re
    r = _client().get("/api/docs")
    assert r.status_code == 200
    html, csp = r.text, r.headers["content-security-policy"]
    assert "swagger-ui-dist@5/" not in html
    remote = re.findall(r'<(?:script|link)[^>]+(?:src|href)="(https://[^"]+)"[^>]*>', html)
    assert len(remote) == 2
    for tag in re.findall(r"<(?:script|link)[^>]+https://[^>]+>", html):
        assert re.search(r'integrity="sha(256|384|512)-[A-Za-z0-9+/=]{40,}"', tag) and 'crossorigin="anonymous"' in tag, tag
        assert re.search(r"swagger-ui-dist@\d+\.\d+\.\d+/", tag), tag                # an exact version
    inline = re.findall(r"<script>(.*?)</script>", html, re.S)
    assert len(inline) == 1
    digest = base64.b64encode(hashlib.sha256(inline[0].encode()).digest()).decode()
    directives = dict(d.strip().split(" ", 1) for d in csp.split(";") if " " in d.strip())
    assert directives["default-src"] == "'none'" and directives["frame-ancestors"] == "'none'"
    assert directives["script-src"].split() == [u for u in remote if u.endswith(".js")] + [f"'sha256-{digest}'"]
    assert directives["connect-src"] == "'self'"
    assert "/api/docs" not in _client().get("/api/openapi.json").json()["paths"]


def test_one_huge_class_id_cannot_make_extend_build_billions_of_names(tmp_path):
    ds, _root = _labeled(tmp_path, {"train/a.jpg": "0 0.5 0.5 0.2 0.2\n2000000000 0.5 0.5 0.1 0.1\n"})
    try:
        c = _client()
        t = time.perf_counter()
        for url in ("preview", "zip-link"):
            r = c.post(f"/api/datasets/{ds}/export/{url}", json={})
            assert r.status_code == 400 and "far beyond the class list" in r.json()["detail"], (url, r.text[:300])
        assert time.perf_counter() - t < 5
        r = c.post(f"/api/datasets/{ds}/export/zip", json={"bad_class": "drop"})
        assert r.status_code == 200
        z = zipfile.ZipFile(io.BytesIO(r.content))
        assert z.read("labels/train/a.txt").decode() == "0 0.500000 0.500000 0.200000 0.200000\n"
        p = c.post(f"/api/datasets/{ds}/export/preview", json={"bad_class": "drop"}).json()
        assert p["classes"]["nc"] == 2 and p["unnamed"]["rows"] == 1
    finally:
        _drop(ds)


def test_class_plan_is_linear_in_the_number_of_classes():
    """An inferred class list grows to the highest class id in the labels: 20k names made every export
    preview spend 5 s in a per-class scan of the whole mapping."""
    from fovea.core.export import class_plan, parse_options
    names = [f"n{i}" for i in range(50_000)]
    t = time.perf_counter()
    cp = class_plan(names, parse_options({"class_map": {"3": 0, "7": 0, "0": 9}}))
    assert time.perf_counter() - t < 1.0
    # 0 ← {3, 7}: the lowest source names it; 9 ← {0, 9}: the class that keeps its own id does
    assert cp.names[0] == "n3" and cp.names[9] == "n9" and cp.names[7] == "n7" and len(cp.names) == 50_000
    cp = class_plan(["a", "b", "c", "d"], parse_options({"class_map": {"0": 2, "2": 5}, "compact": True}))
    assert cp.names == ["b", "a", "d", "c"]


# ---------------------------------------------------------------- deep TIFFs: first page only, bounded memory

_PEAK = r"""
import json, os, resource, sys
sys.path.insert(0, sys.argv[1])
from fovea.core import thumbs
import numpy, tifffile  # noqa: F401 — loaded before the baseline
def rss():
    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return r if sys.platform == "darwin" else r * 1024
base = rss()
im, _ = thumbs.decode_display(sys.argv[2], int(sys.argv[3]))
print(json.dumps({"grew": rss() - base, "size": im.size, "red": im.split()[0].getextrema()}))
"""


def _peak_decode(tmp_path, path, side):
    import subprocess
    import sys
    env = dict(os.environ, FOVEA_DATA_DIR=str(tmp_path / "peak-data"))
    out = subprocess.run([sys.executable, "-c", _PEAK, str(Path(__file__).resolve().parents[1]), str(path), str(side)],
                         env=env, capture_output=True, text=True, timeout=120)
    assert out.returncode == 0, out.stderr[-2000:]
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_deep_tiff_thumbnail_reads_one_page_and_never_holds_the_16_bit_raster(tmp_path):
    """A 1 MB RGB16 TIFF used 2-3 GB for one thumbnail: tifffile.imread read every page of a stack at full
    resolution and then made a float32 copy of it all, and only Pillow's page-0 size was checked."""
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    page = np.zeros((1024, 1024, 3), np.uint16)
    page[..., 0] = np.linspace(0, 4095, 1024)[None, :]
    stack = tmp_path / "stack.tif"            # 120 pages x 6 MB decoded, a few hundred KB on disk
    tifffile.imwrite(str(stack), (page for _ in range(120)), shape=(120, 1024, 1024, 3), dtype="uint16",
                     photometric="rgb", compression="zlib")
    big = tmp_path / "big.tif"                # one 6000 x 6000 page: 216 MB decoded at 16 bits
    tifffile.imwrite(str(big), np.zeros((6000, 6000, 3), np.uint16), photometric="rgb", tile=(512, 512), compression="zlib")
    r = _peak_decode(tmp_path, stack, 256)
    assert r["size"] == [256, 256] and r["red"][1] > 200, r          # page 0's gradient, stretched
    assert r["grew"] < 250 * 2**20, r                                # was ~2.4 GB
    r = _peak_decode(tmp_path, big, 256)
    assert r["size"] == [256, 256] and r["grew"] < 250 * 2**20, r     # was ~1 GB


def _deep_reference(np, a, k):
    """Box mean by k (edge cells average the pixels they have) and the stretch every deep rendition uses."""
    h, w = a.shape[:2]
    sy, sx = max(1, h // 512), max(1, w // 512)
    out = np.zeros((-(-h // k), -(-w // k), 3), np.uint8)
    for c in range(3):
        ch = a[..., c].astype(np.float64)
        s = ch[::sy, ::sx].ravel()
        keep = s[(s != 0) & (s != 65535)]
        vals = keep if keep.size >= 0.01 * s.size else s
        lo, hi = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
        hi = hi if hi > lo else lo + 1.0
        sums = np.add.reduceat(np.add.reduceat(ch, np.arange(0, w, k), axis=1), np.arange(0, h, k), axis=0)
        cnt = np.outer(np.minimum(k, h - np.arange(0, h, k)), np.minimum(k, w - np.arange(0, w, k)))
        out[..., c] = np.clip((sums / cnt - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
    return out


@pytest.mark.parametrize("layout_kw", [{"tile": (128, 128)}, {"rowsperstrip": 7}, {"tile": (64, 64), "planarconfig": "separate"},
                                       {"tile": (128, 128), "byteorder": ">"}])
def test_deep_tiff_reduced_and_full_reads_match_a_full_precision_reference(tmp_path, layout_kw):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fovea.core import thumbs
    rng = np.random.default_rng(3)
    a = (rng.random((701, 1003, 3)) * np.array([4095, 1000, 300])).astype(np.uint16)   # 12-bit-ish, per-channel ranges
    a[:40, :40] = 0                                                                     # 'no data' corner
    data = np.moveaxis(a, -1, 0) if layout_kw.get("planarconfig") == "separate" else a
    p = tmp_path / "d.tif"
    tifffile.imwrite(str(p), data, photometric="rgb", compression="zlib", **layout_kw)
    small = thumbs._read_deep_tiff(str(p), (100, 70))                 # k = 10: 101 x 71 with partial edge cells
    assert small.size == (101, 71)
    assert np.abs(np.asarray(small, np.int16) - _deep_reference(np, a, 10)).max() <= 1
    full = thumbs._read_deep_tiff(str(p))
    assert full.size == (1003, 701)
    assert np.abs(np.asarray(full, np.int16) - _deep_reference(np, a, 1)).max() <= 1
    im, _ = thumbs.decode_display(str(p), 128)
    assert im.size == (128, 89)


def test_deep_tiff_thumbnail_reads_the_smallest_pyramid_level_that_covers_it(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fovea.core import thumbs
    full = np.zeros((2048, 3072, 3), np.uint16)
    full[..., 0] = np.linspace(0, 4095, 3072)[None, :]
    levels = [full, full[::2, ::2], full[::4, ::4], full[::8, ::8]]     # 3072, 1536, 768, 384 wide
    sub, seq = tmp_path / "subifds.tif", tmp_path / "reduced.tif"
    with tifffile.TiffWriter(str(sub)) as tw:
        tw.write(levels[0], photometric="rgb", tile=(256, 256), compression="zlib", subifds=3)
        for lv in levels[1:]:
            tw.write(lv, photometric="rgb", tile=(256, 256), compression="zlib", subfiletype=1)
    with tifffile.TiffWriter(str(seq)) as tw:
        for i, lv in enumerate(levels):
            tw.write(lv, photometric="rgb", tile=(256, 256), compression="zlib", subfiletype=1 if i else 0)
    read = []
    real = thumbs._deep_render
    monkeypatch.setattr(thumbs, "_deep_render", lambda np_, page, target: read.append(int(page.imagewidth)) or real(np_, page, target))
    for p in (sub, seq):
        read.clear()
        im, _ = thumbs.decode_display(str(p), 512)
        assert read == [768] and im.size == (512, 341), (p.name, read)
        assert im.split()[0].getextrema()[1] > 200
        read.clear()
        thumbs.decode_display(str(p))                                   # full size: page 0 itself
        assert read == [3072]


def test_deep_tiff_checks_its_own_header_and_counts_as_a_big_decode(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fovea.core import thumbs
    p = tmp_path / "d.tif"
    tifffile.imwrite(str(p), np.full((300, 400, 3), 1000, np.uint16), photometric="rgb")
    monkeypatch.setattr(thumbs, "MAX_IMAGE_PIXELS", 100_000)
    with pytest.raises(thumbs.ImageTooLarge):
        thumbs._read_deep_tiff(str(p), (64, 48))
    monkeypatch.setattr(thumbs, "MAX_IMAGE_PIXELS", 250_000_000)
    taken = []

    class Gate:
        def acquire(self):
            taken.append(1)

        def release(self):
            pass

    monkeypatch.setattr(thumbs, "_BIG_SLOTS", Gate())
    monkeypatch.setattr(thumbs, "BIG_PIXELS", 300 * 400 + 1)          # over it only at 2 bytes per sample
    thumbs.decode_display(str(p), 64)
    assert taken == [1]


def test_deep_tiff_stack_thumbnail_is_page_zero_only(tmp_path, monkeypatch):
    np = pytest.importorskip("numpy")
    tifffile = pytest.importorskip("tifffile")
    from fovea.core import thumbs
    first = np.zeros((200, 300, 3), np.uint16)
    first[..., 0] = np.linspace(0, 4095, 300)[None, :]
    rest = np.full((200, 300, 3), 4095, np.uint16)
    p = tmp_path / "stack.tif"
    tifffile.imwrite(str(p), np.stack([first] + [rest] * 30), photometric="rgb", compression="zlib")
    pages = []
    real = tifffile.TiffPage.segments

    def segments(self, *a, **kw):
        pages.append(self.index)
        return real(self, *a, **kw)

    monkeypatch.setattr(tifffile.TiffPage, "segments", segments)
    im, _ = thumbs.decode_display(str(p), 128)
    full, _ = thumbs.decode_display(str(p))
    assert set(pages) == {0} and full.size == (300, 200)
    red = np.asarray(full)[..., 0]
    assert red[:, 0].max() < 10 and red[:, -1].min() > 245             # page 0's ramp, not the white pages


# ---------------------------------------------------------------- access-token page keeps the deep link

def test_token_page_posts_to_its_own_url_so_deep_links_survive_and_no_token_reaches_a_url(monkeypatch):
    import asyncio
    monkeypatch.setenv("FOVEA_TOKEN", "t0ken-for-tests")
    reached = []

    async def app(scope, receive, send):
        reached.append(scope["path"])
        await send({"type": "http.response.start", "status": 200, "headers": [(b"content-type", b"text/plain")]})
        await send({"type": "http.response.body", "body": b"ok"})

    def call(path, qs=b"", method="GET", body=b"", headers=(), raw_path=None):
        sent = []

        async def send(m):
            sent.append(m)

        chunks = [{"type": "http.request", "body": body[:5], "more_body": True},
                  {"type": "http.request", "body": body[5:], "more_body": False}]

        async def receive():
            return chunks.pop(0) if chunks else {"type": "http.disconnect"}

        scope = {"type": "http", "method": method, "path": path, "query_string": qs, "client": ("10.20.30.40", 5000),
                 "scheme": "http", "headers": [(b"host", b"192.168.1.5:8000"), (b"accept", b"text/html,*/*")] + list(headers)}
        if raw_path is not None:
            scope["raw_path"] = raw_path
        asyncio.run(security.GuardMiddleware(app)(scope, receive, send))
        start = sent[0]
        hdrs = {k.decode(): v.decode() for k, v in start["headers"]}
        return start["status"], hdrs, b"".join(m.get("body", b"") for m in sent[1:]).decode()

    form = [(b"content-type", b"application/x-www-form-urlencoded"), (b"origin", b"http://192.168.1.5:8000")]
    status, _h, page = call("/d/vfp/annotate", b"img=5&filter=a%26b")
    assert status == 401 and '<form method="post" action="/d/vfp/annotate?img=5&amp;filter=a%26b">' in page
    assert "not right" not in page
    # a wrong token: shown again on the same URL, nothing in the address bar
    status, _h, page = call("/d/vfp/annotate", b"img=5&filter=a%26b", "POST", b"token=wrong-token", form)
    assert status == 401 and "not right" in page and 'action="/d/vfp/annotate?img=5&amp;filter=a%26b"' in page
    assert "wrong-token" not in page
    # the right one: back to exactly the link that was opened, with the cookie
    status, h, _ = call("/d/vfp/annotate", b"img=5&filter=a%26b", "POST", b"token=t0ken-for-tests", form)
    assert status == 303 and h["location"] == "/d/vfp/annotate?img=5&filter=a%26b"
    assert "fovea_token=t0ken-for-tests" in h["set-cookie"] and "HttpOnly" in h["set-cookie"]
    assert reached == []                                                # the form body never reaches the app
    # a link that carried a wrong token: the form posts to the URL without it
    status, _h, page = call("/d/vfp/annotate", b"token=typo&img=5")
    assert status == 401 and "not right" in page and 'action="/d/vfp/annotate?img=5"' in page
    # the encoded path the browser sent is kept; '//host' can never become an off-site redirect
    status, h, _ = call("/d/한/x", b"", "POST", b"token=t0ken-for-tests", form, raw_path=b"/d/%ED%95%9C/x")
    assert status == 303 and h["location"] == "/d/%ED%95%9C/x"
    status, h, _ = call("//evil.example/x", b"", "POST", b"token=t0ken-for-tests", form, raw_path=b"//evil.example/x")
    assert status == 303 and h["location"] == "/evil.example/x"
    # never a login for the API, and still cross-site protected
    status, h, _ = call("/api/meta", b"", "POST", b"token=t0ken-for-tests", form)
    assert status == 401 and h["content-type"] == "application/json"
    status, _h, _ = call("/d/x", b"", "POST", b"token=t0ken-for-tests",
                         [(b"content-type", b"application/x-www-form-urlencoded"), (b"origin", b"http://evil.example")])
    assert status == 403
    status, _h, page = call("/d/x", b"", "POST", b"token=" + b"x" * 9000, form)   # oversized body: just wrong
    assert status == 401 and "not right" in page
    assert reached == []


def test_token_login_end_to_end_through_the_app(monkeypatch):
    """The whole app behind the guard, as a browser on another machine sees it."""
    monkeypatch.setenv("FOVEA_TOKEN", "t0ken-for-tests")
    c = _client_from(("10.20.30.40", 5000), base_url="http://192.168.1.5:8000")
    r = c.get("/d/vfp/annotate?img=5", headers={"accept": "text/html"})
    assert r.status_code == 401 and 'method="post" action="/d/vfp/annotate?img=5"' in r.text
    r = c.post("/d/vfp/annotate?img=5", data={"token": "t0ken-for-tests"},
               headers={"accept": "text/html", "origin": "http://192.168.1.5:8000"}, follow_redirects=False)
    assert r.status_code == 303 and r.headers["location"] == "/d/vfp/annotate?img=5"
    r = c.get(r.headers["location"], headers={"accept": "text/html"})
    assert r.status_code == 200 and "<html" in r.text.lower()           # the app itself, on the same link
    assert c.get("/api/meta").status_code == 200


# ---------------------------------------------------------------- FastAPI compatibility

def test_server_code_imports_no_fastapi_internals():
    """fovea.main imported fastapi.dependencies.utils.get_flat_dependant, which newer FastAPI releases removed:
    a fresh install (pip / Docker, FastAPI 0.141+) could not start the server at all. Internal modules have no
    compatibility promise, so no server module may import them (checked on 0.115.6 and 0.143.0 by hand)."""
    import ast
    root = Path(__file__).resolve().parents[1] / "fovea"
    bad = []
    for f in root.rglob("*.py"):
        for node in ast.walk(ast.parse(f.read_text(encoding="utf-8"))):
            names = [node.module or ""] if isinstance(node, ast.ImportFrom) else \
                [a.name for a in node.names] if isinstance(node, ast.Import) else []
            bad += [f"{f.relative_to(root)}:{node.lineno} {n}" for n in names
                    if n.startswith(("fastapi.dependencies", "fastapi._compat", "starlette._"))]
    assert not bad, bad


def test_int_params_include_sub_dependencies():
    from fastapi import Depends, FastAPI, Query
    from fastapi.routing import APIRoute
    from fovea import main

    def paging(limit: int = Query(10), cursor: str = ""):
        return limit

    def scoped(page=Depends(paging), offset: int = 0):
        return page

    demo = FastAPI()

    @demo.get("/x/{item}/{name}")
    def handler(item: int, name: str, n=Depends(scoped), k: int | None = None, tag: str = ""):
        return {}

    route = next(r for r in demo.routes if isinstance(r, APIRoute) and r.path == "/x/{item}/{name}")
    path, query = main._int_params(route)
    assert path == ["item"] and sorted(query) == ["k", "limit", "offset"]


# ---------------------------------------------------------------- the version range requirements.txt allows

def test_route_walk_sees_nested_routers_mounts_and_sub_apps():
    """FastAPI 0.140+ keeps include_router() lazy: app.routes holds one router object per include, so a plain
    `for r in app.routes: r.path` misses every router endpoint (test_integration's frontend-path check failed on
    0.143 and test_ai_cloud's 'is /api/ai/propose there' check silently skipped). testkit.iter_routes must see the
    same full paths on every release."""
    from fastapi import APIRouter, FastAPI
    from starlette.staticfiles import StaticFiles
    from testkit import route_paths

    inner, mid, deep, app, sub = APIRouter(), APIRouter(), APIRouter(), FastAPI(), FastAPI()
    inner.get("/c/{x}")(lambda x: x)
    mid.post("/b")(lambda: 1)
    mid.include_router(inner, prefix="/b")                   # a router inside a router
    sub.get("/s")(lambda: 2)
    deep.include_router(inner, prefix="/deep")
    sub.include_router(deep, prefix="/r")
    app.include_router(mid, prefix="/a")
    app.mount("/static", StaticFiles(directory=str(Path(__file__).parent)), name="st")
    app.mount("/sub", sub)                                   # a sub-app with its own routers
    app.get("/top")(lambda: 3)
    docs = {"/openapi.json", "/docs", "/docs/oauth2-redirect", "/redoc"}
    assert route_paths(app) - docs - {"/sub" + d for d in docs} == {
        "/a/b", "/a/b/c/{x}", "/static/{path:path}", "/sub/s", "/sub/r/deep/c/{x}", "/top"}


def test_route_walk_covers_every_documented_app_route():
    """The walk over the real app agrees with FastAPI's own (public) OpenAPI listing and sees router endpoints
    from each kind of include: core routers, a plugin router under its prefix, the static mounts."""
    import re
    from testkit import route_paths
    from fovea.main import app
    paths = route_paths(app)
    norm = {re.sub(r"\{(\w+):\w+\}", r"{\1}", p) for p in paths}
    assert set(app.openapi()["paths"]) <= norm
    for p in ("/api/meta", "/api/jobs/{job_id}/cancel", "/api/datasets/{ds_id}/images", "/api/fs/browse",
              "/api/datasets/{ds_id}/export/zip", "/api/thumb/{image_id}", "/api/ai/propose",
              "/api/plugins/compare/evals", "/static/{path:path}", "/d/{rest:path}"):
        assert p in paths, p


def test_client_from_sets_the_peer_address_on_every_starlette(monkeypatch):
    """TestClient(client=...) exists only on Starlette 0.42+ (FastAPI 0.115.x brings 0.41): testkit.client_from sets
    the address in the ASGI scope instead, for plain apps and for a middleware-wrapped app alike."""
    monkeypatch.delenv("FOVEA_TOKEN", raising=False)
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from testkit import client_from
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    demo = Starlette(routes=[Route("/who", lambda request: JSONResponse(
        {"host": request.client.host, "local": security.is_local_request(request)}))])
    assert client_from(demo, ("192.168.1.50", 50000)).get("/who").json() == {"host": "192.168.1.50", "local": False}
    assert client_from(demo, ("127.0.0.1", 50000)).get("/who").json() == {"host": "127.0.0.1", "local": True}
    proxied = client_from(ProxyHeadersMiddleware(demo, trusted_hosts="127.0.0.1"), ("127.0.0.1", 50000))
    assert proxied.get("/who", headers={"X-Forwarded-For": "203.0.113.9"}).json()["host"] == "203.0.113.9"
    # and through the real app: a LAN client may look, but not touch the stored API keys
    lan = _client_from(("192.168.1.50", 50000))
    assert lan.get("/api/ai/providers").status_code == 200
    r = lan.delete("/api/ai/providers/anthropic/key")
    assert r.status_code == 403 and "FOVEA_ALLOW_REMOTE_AI" in r.text


def test_installed_uvicorn_trusts_a_proxy_cidr_as_the_readme_advises():
    """README (Docker behind a host proxy): FORWARDED_ALLOW_IPS=172.16.0.0/12,…. uvicorn reads CIDR ranges only from
    0.31 on; before that the range never matched, the header was ignored and every client looked like the bridge."""
    from starlette.applications import Starlette
    from starlette.responses import JSONResponse
    from starlette.routing import Route
    from testkit import client_from
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware

    demo = Starlette(routes=[Route("/who", lambda request: JSONResponse(request.client.host))])
    c = client_from(ProxyHeadersMiddleware(demo, trusted_hosts="172.16.0.0/12,192.168.65.0/24"), ("172.17.0.1", 5))
    assert c.get("/who", headers={"X-Forwarded-For": "203.0.113.9"}).json() == "203.0.113.9"


def test_requirements_cap_the_web_stack_at_tested_releases():
    """fastapi, starlette (recent FastAPI releases no longer cap it) and uvicorn each carry a floor and a ceiling,
    and the uvicorn floor reads CIDR proxy ranges (see above)."""
    import re
    req = (Path(__file__).resolve().parents[1] / "requirements.txt").read_text(encoding="utf-8")
    spec = {}
    for line in req.splitlines():
        m = re.match(r"([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?\s*([<>=!~][^#]*)", line.strip())
        if m:
            spec[m.group(1).lower()] = m.group(2).replace(" ", "")
    for name in ("fastapi", "starlette", "uvicorn"):
        assert re.search(r">=[\d.]+", spec.get(name, "")) and re.search(r"<[\d.]+", spec[name]), (name, spec.get(name))
    floor = tuple(int(x) for x in re.search(r">=([\d.]+)", spec["uvicorn"]).group(1).split("."))
    assert floor >= (0, 31)
