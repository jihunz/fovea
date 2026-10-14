"""Cross-role contracts found during integration: every API path the frontend calls exists, the QA engine and the
label parser agree on flag wording, proposal decisions can be undone, and AI runs record which images they
processed (Compare relies on it). Run: python -m pytest tests -q"""
from __future__ import annotations

import os
import re
import sys
import time
import uuid
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout  # noqa: E402

SRC = Path(__file__).resolve().parents[1]


def _make_dataset(tmp_path, n=6):
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"ds-{uuid.uuid4().hex[:6]}"
    for k in range(n):
        ip = root / "images" / "train" / f"img_{k:03d}.jpg"
        ip.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (160, 120), (90 + 20 * k, 100, 110)).save(ip)
        lp = root / "labels" / "train" / f"img_{k:03d}.txt"
        lp.parent.mkdir(parents=True, exist_ok=True)
        lp.write_text(f"{k % 2} 0.400000 0.500000 0.200000 0.300000\n1 0.700000 0.400000 0.150000 0.250000\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "int-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["person", "fall"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fastapi.testclient import TestClient
    from fovea.ai import registry
    from fovea.main import app
    registry.reset()
    return TestClient(app)


@pytest.fixture()
def ds(tmp_path):
    ds_id = _make_dataset(tmp_path)
    yield ds_id
    db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds_id,))
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def _wait(job_id, timeout=60):
    from fovea.jobs import jobs
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = jobs.get(job_id)
        if j and j.status in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError("job did not finish")


# ------------------------------------------------------------------ frontend ↔ backend paths

def _route_patterns():
    from fovea.main import app
    from testkit import iter_routes          # walks include_router() on every FastAPI (0.14x: app.routes is lazy)
    pats = []
    for path, _ in iter_routes(app):
        if not path or not path.startswith("/api/"):
            continue
        rx = re.sub(r"\{[^}:]+(:path)?\}", lambda m: ".+" if m.group(1) else "[^/]+", path)
        pats.append(re.compile("^" + rx + "$"))
    return pats


def _frontend_api_paths():
    """Every '/api/…' string or template literal in the browser code, with ${…} replaced by a sample value."""
    out = set()
    files = list((SRC / "static").rglob("*.js")) + list((SRC / "fovea" / "plugins").rglob("*.js"))
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in re.finditer(r"[`'\"](/api/[^`'\"\s]*)", text):
            p = m.group(1)
            p = re.sub(r"\$\{[^}]*\}", "x1", p).split("?")[0]
            if p.endswith("/") or p == "/api":
                continue                                   # a prefix used to build other paths
            out.add((f.relative_to(SRC).as_posix(), p))
        # plugin code builds its paths from a base constant: `${API}/evals/...`
        base = re.search(r"const API\s*=\s*['\"`](/api/[^'\"`]+)", text)
        base = base.group(1) if base else (f"/api/plugins/{f.parts[f.parts.index('plugins') + 1]}"
                                           if "plugins" in f.parts and "${API}" in text else None)
        if base:
            for m in re.finditer(r"`\$\{API\}(/[^`]*)`", text):
                p = re.sub(r"\$\{[^}]*\}", "x1", base + m.group(1)).split("?")[0]
                out.add((f.relative_to(SRC).as_posix(), p))
    return out


def test_every_frontend_api_path_has_a_route():
    pats = _route_patterns()
    paths = _frontend_api_paths()
    assert len(paths) > 60
    missing = sorted((f, p) for f, p in paths if not any(rx.match(p) for rx in pats))
    assert not missing, f"frontend calls paths the server does not serve: {missing}"


# ------------------------------------------------------------------ shared wording

def test_flag_labels_agree_between_parser_table_and_qa_engine():
    from fovea.core import labels, qa
    for code, text in qa.LABELS.items():
        assert labels.FLAG_LABELS.get(code) == text, code
    assert set(labels.FLAG_LABELS) == set(qa.LABELS)


def test_frontend_job_kinds_are_labelled():
    """Every job kind the server submits has a readable name in the shell (banner, Activity)."""
    kinds = set()
    for f in (SRC / "fovea").rglob("*.py"):
        kinds |= set(re.findall(r"\bsubmit\(\s*[\"']([a-z_]+)[\"']", f.read_text(encoding="utf-8")))
    app_js = (SRC / "static" / "app.js").read_text(encoding="utf-8")
    table = re.search(r"const JOB_KIND = \{(.*?)\n\};", app_js, re.S).group(1)
    named = set(re.findall(r"\b([a-z_]+):\s*\[", table))
    assert kinds and not (kinds - named), f"job kinds without a label in app.js: {sorted(kinds - named)}"


# ------------------------------------------------------------------ proposals / runs

def test_decide_pending_undoes_an_earlier_decision(client, ds):
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=? ORDER BY id", (ds,))["id"]
    r = client.post("/api/ai/propose", json={"dataset_id": ds, "image_id": iid, "provider": "mock", "model": "mock-fast"})
    assert r.status_code == 200, r.text
    ids = [p["id"] for p in r.json()["preds"]]
    assert ids
    r = client.post(f"/api/datasets/{ds}/proposals/decide", json={"accept": ids[:1], "reject": ids[1:2]})
    assert r.json()["updated"] == len(ids[:2])
    r = client.post(f"/api/datasets/{ds}/proposals/decide", json={"pending": ids[:2]})
    assert r.status_code == 200 and r.json()["updated"] == len(ids[:2])
    states = {x["state"] for x in db.query(f"SELECT state FROM ai_preds WHERE id IN ({','.join('?' * len(ids))})", ids)}
    assert states == {"pending"}
    r = client.post(f"/api/datasets/{ds}/proposals/decide", json={"accept": ids[:1], "pending": ids[:1]})
    assert r.status_code == 422


def test_batch_and_benchmark_runs_record_processed_images(client, ds):
    from fovea.plugins.compare.evaluate import _processed_ids
    ids = [r["id"] for r in db.query("SELECT id FROM images WHERE dataset_id=? ORDER BY id", (ds,))][:4]
    r = client.post("/api/ai/batch", json={"dataset_id": ds, "image_ids": ids, "provider": "mock", "model": "mock-fast",
                                           "target": "proposals"})
    assert r.status_code == 200, r.text
    assert _wait(r.json()["job"]["id"]).status == "done"
    run = db.query_one("SELECT * FROM ai_runs WHERE id=?", (r.json()["run_id"],))
    assert _processed_ids(db.loads(run["params"], {}), db.loads(run["result"], {})) == set(ids)
    r = client.post("/api/ai/benchmark", json={"dataset_id": ds, "image_ids": ids,
                                               "configs": [{"provider": "mock", "model": "mock-fast"}]})
    assert r.status_code == 200, r.text
    assert _wait(r.json()["job"]["id"]).status == "done"
    run = db.query_one("SELECT * FROM ai_runs WHERE dataset_id=? AND kind='benchmark' ORDER BY id DESC", (ds,))
    assert _processed_ids(db.loads(run["params"], {}), db.loads(run["result"], {})) == set(ids)


# ------------------------------------------------------------------ wave 2a contracts

def test_dataset_splits_are_nfc_like_the_image_rows(client, tmp_path):
    """macOS hands out folder names decomposed (NFD Hangul). images.split, filters and facet keys are NFC, so the
    dataset's public split list must be NFC too, or Explore / Overview look up a split that has no images."""
    import unicodedata
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    nfd = unicodedata.normalize("NFD", "test-포함")
    root = tmp_path / f"nfd-{uuid.uuid4().hex[:6]}"
    for split in ("train", nfd):
        for k in range(2):
            ip = root / "images" / split / f"a_{k}.jpg"
            ip.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (64, 48), (100, 100, 100)).save(ip)
            lp = root / "labels" / split / f"a_{k}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("0 0.5 0.5 0.2 0.2\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "nfd-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)", (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["a"]), "user", "",
                                                "new", now, now))
    try:
        scan_dataset(ds_id, Job(id="s", kind="scan"))
        d = client.get(f"/api/datasets/{ds_id}").json()["dataset"]
        nfc = unicodedata.normalize("NFC", nfd)
        assert d["splits"] == ["train", nfc]
        assert all(s["split"] == unicodedata.normalize("NFC", s["split"]) for s in d["layout"]["sources"])
        facets = client.get(f"/api/datasets/{ds_id}/images/facets").json()["facets"]["split"]
        assert facets.get(nfc) == 2
        assert client.get(f"/api/datasets/{ds_id}/images", params={"split": d["splits"][1], "limit": 1}).json()["total"] == 2
    finally:
        db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_ai_tab_takes_a_working_set_from_explore():
    """Explore hands selections over 1,500 images to the AI tab as ?set=<token>; the label section must offer it."""
    explore = (SRC / "static" / "views" / "explore.js").read_text(encoding="utf-8")
    ai = (SRC / "static" / "views" / "ai.js").read_text(encoding="utf-8")
    assert "/ai?s=label&${key}=" in explore
    assert "ctx.query.get('set')" in ai and "mk('set'" in ai
