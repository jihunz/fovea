"""Shell (static/app.js, static/lib/*, views home/dataset/overview/settings/files/export/issues): the pure helpers of
the dataset shell and Overview run in Node, and the server contracts those pages rely on run through the API.

What these pin down:
* The job status slot follows the right job (running before queued, never one it already saw finish), says what a
  finished job did, and the notice row knows a partial index that is being resumed (the server reads 'scanning').
* The Data quality card follows the QA job wherever it is reported — /qa/summary, the shell's job poll or
  dataset.job — so its numbers never freeze while the banner moves on (N2), and its mode matches the summary.
* Overview's latest-index report (dataset.last_scan) and the placeholder-name cue for "Suggest names".
* Every static module is in app.html's import map (cached and versioned), and every file the shell owns parses.
* Server contracts: /qa/summary status + job, dataset.last_scan problems, crop + thumbnail cache stats and clear,
  DATASET_CARD.md standalone and in the export preview, Objects accepting ?cls=.
* Review round 1: menus/popovers never cover their anchor and keep their CSS cap, the Flex switch only where a provider
  reads batch_mode, Overview links that say what they open (all images, suspects worst first, boxes vs images), the
  under-represented rule, scan notes not repeated, Files keys in ?, the rest reminder ignoring other layers' Escape,
  restore following its own job, the ZIP busy state, AA contrast of faint/status text and the shell CSS fixes.
Node-based tests are skipped when Node is not installed. Run: python -m pytest tests/test_shell.py -q"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout  # noqa: E402

SRC = Path(__file__).resolve().parents[1]
STATIC = SRC / "static"
NODE = shutil.which("node")

SHELL_FILES = ["app.js", "lib/router.js", "lib/store.js", "lib/api.js", "lib/ui.js", "lib/dom.js", "lib/icons.js",
               "lib/comfort.js", "lib/colors.js", "lib/overlay.js", "lib/keys.js", "views/home.js", "views/dataset.js",
               "views/settings.js", "views/files.js", "views/export.js", "views/overview.js", "views/issues.js"]


# ------------------------------------------------------------------ Node: pure helpers

SCENARIOS = r"""
import { jobsToFollow, doneLine, alertKind, DONE_LINE_MS } from './views/dataset.mjs';
import { qaJobFor, qaMode, isPlaceholderName, scanReport } from './views/overview.mjs';
const out = {};
const J = (id, kind, status, created_at, dataset_id = 'ds') => ({ id, kind, status, created_at, dataset_id });

// ---- status slot: which job, in what order
const jobs = [J('q1', 'qa', 'queued', 5), J('s1', 'scan', 'running', 9), J('x', 'qa', 'running', 1, 'other'), J('d', 'qa', 'done', 0), J('q0', 'qa', 'queued', 2)];
out.follow = jobsToFollow(jobs, 'ds').map(j => j.id);
out.followFinished = jobsToFollow(jobs, 'ds', { finished: new Set(['s1']) }).map(j => j.id);
out.followDsJob = jobsToFollow([], 'ds', { dsJob: { id: 'z', kind: 'scan', status: 'running', created_at: 3 } }).map(j => j.id);
out.followDsJobDup = jobsToFollow([J('z', 'scan', 'running', 3)], 'ds', { dsJob: { id: 'z', kind: 'scan', status: 'running' } }).length;
const num = (n) => Number(n).toLocaleString('en-US');
out.done = {
  scan: doneLine('Indexing', 'scan', { status: 'done', result: { images: 7808 } }, null, num),
  scanNoResult: doneLine('Indexing', 'scan', { status: 'done', result: null }, null, num),
  restore: doneLine('Restoring labels', 'restore', { status: 'done' }, null, num),
  qa: doneLine('Checking data quality', 'qa', { status: 'done' }, null, num),
  other: doneLine('AI labeling', 'ai_batch', { status: 'done' }, null, num),
  cancelled: doneLine('AI labeling', 'ai_batch', { status: 'cancelled' }, null, num),
  failed: doneLine('Exporting', 'export_copy', null, new Error('disk full'), num),
};
out.doneMs = DONE_LINE_MS;

// ---- notice row
const ds = (status, ok = true) => ({ id: 'ds', status, reachable: { ok } });
out.alert = {
  ready: alertKind(ds('ready'), []),
  partial: alertKind(ds('partial'), []),
  partialWithScan: alertKind(ds('partial'), [J('s', 'scan', 'running', 1)]),
  resumingReadsScanning: alertKind(ds('scanning'), [], 'partial'),
  resumingKept: alertKind(ds('scanning'), [], 'resuming'),
  scanningFresh: alertKind(ds('scanning'), [], null),
  unreachable: alertKind(ds('partial', false), []),
  finished: alertKind(ds('ready'), [], 'resuming'),
  none: alertKind(null, []),
};

// ---- Data quality card (N2)
const sum = (status, job = null, extra = {}) => ({ status, job, computed_at: 100, ...extra });
const queued = J('q-queued', 'qa', 'queued', 5), running = J('q-run', 'qa', 'running', 9);
out.qa = {
  fromSummary: (qaJobFor(sum('running', running), [], 'ds') || {}).id,
  pollBeatsQueuedSummary: (qaJobFor(sum('running', queued), [queued, running], 'ds') || {}).id,
  fromDatasetJob: (qaJobFor(sum('ok'), [], 'ds', running) || {}).id,
  otherDataset: qaJobFor(sum('ok'), [J('o', 'qa', 'running', 1, 'other')], 'ds'),
  notQa: qaJobFor(sum('ok'), [J('s', 'scan', 'running', 1)], 'ds'),
  freshestSnapshot: (qaJobFor(sum('running', J('a', 'qa', 'queued', 1)), [J('a', 'qa', 'running', 1)], 'ds') || {}).status,
  doneIgnored: qaJobFor(sum('ok', J('d', 'qa', 'done', 1)), [], 'ds'),
};
out.mode = {
  jobWins: qaMode(sum('ok'), running),
  summaryRunning: qaMode(sum('running'), null),
  scanning: qaMode({ status: 'never', scanning: true, computed_at: null }, null),
  never: qaMode({ status: 'never', computed_at: null }, null),
  stale: qaMode(sum('stale'), null),
  ok: qaMode(sum('ok'), null),
  empty: qaMode(null, null),
};

// ---- class names and the latest index
out.placeholder = ['class_0', 'class_12', '3', '', '  ', null, 'Class 4', 'class-5', 'fallen_person', 'person', 'class_a', '0 person'].map(isPlaceholderName);
out.report = scanReport({ at: 5, images: 7808, elapsed: 5.3, problems: [
  { code: 'labels_without_images', label: 'Label files with no matching images folder', severity: 'info', count: 1008, examples: ['a', 'b', 'c', 'd', 'e', 'f'] },
  { code: 'stat_error', label: 'Images that could not be read', severity: 'warn', count: 3, examples: ['x.jpg'] },
  { code: 'zero', label: 'nothing', severity: 'warn', count: 0 } ], notes: ['ignored when problems exist'] });
out.reportLegacy = scanReport({ at: 1, notes: ['3 list entries not found'] });
out.reportNone = scanReport(null);
out.reportClean = scanReport({ at: 1, images: 10, problems: [], notes: [] });
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_out(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("shell-node")
    (d / "views").mkdir()
    # Overview imports sortSplits from home.js, which pulls in browser-only modules: stub it.
    (d / "views" / "home.mjs").write_text("export const sortSplits = (l) => l.slice();\n")
    (d / "views" / "dataset.mjs").write_text((STATIC / "views" / "dataset.js").read_text(encoding="utf-8"), encoding="utf-8")
    # ... and the audit-kind words from ai.js (the AI tab, also browser-only): just its KIND_COUNT table.
    kinds = re.search(r"^export const KIND_COUNT = \{.*?\};", (STATIC / "views" / "ai.js").read_text(encoding="utf-8"), re.M).group(0)
    (d / "views" / "ai.mjs").write_text(kinds + "\n", encoding="utf-8")
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8").replace("'./home.js'", "'./home.mjs'").replace("'./ai.js'", "'./ai.mjs'")
    (d / "views" / "overview.mjs").write_text(ov, encoding="utf-8")
    (d / "run.mjs").write_text(SCENARIOS, encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.mjs")], capture_output=True, text=True, timeout=60)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_status_slot_follows_running_then_oldest_queued(node_out):
    assert node_out["follow"] == ["s1", "q0", "q1"], "running first, then queued oldest first; other datasets and done jobs never"
    assert node_out["followFinished"] == ["q0", "q1"], "a job the slot saw finish is never followed again"
    assert node_out["followDsJob"] == ["z"], "dataset.job counts before the job poll has caught up"
    assert node_out["followDsJobDup"] == 1


def test_finished_job_leaves_a_one_line_result(node_out):
    d = node_out["done"]
    assert d["scan"] == {"tone": "ok", "icon": "checkCircle", "text": "Index finished · 7,808 images"}
    assert d["scanNoResult"]["text"] == "Index finished"
    assert d["restore"]["text"] == "Labels restored"
    assert d["qa"] == {"tone": "ok", "icon": "checkCircle", "text": "Data quality checked"}
    assert d["other"]["text"] == "AI labeling finished"
    assert d["cancelled"]["tone"] == "muted" and d["cancelled"]["text"] == "AI labeling cancelled"
    assert d["failed"]["tone"] == "bad" and d["failed"]["text"] == "Exporting failed"
    assert 2000 <= node_out["doneMs"] <= 6000, "the done line stays a few seconds, not forever"


def test_notice_row_knows_a_resumed_partial_index(node_out):
    a = node_out["alert"]
    assert a["ready"] is None and a["none"] is None
    assert a["partial"] == "partial" and a["partialWithScan"] == "resuming"
    assert a["resumingReadsScanning"] == "resuming", "the server says 'scanning' while a partial index resumes"
    assert a["resumingKept"] == "resuming"
    assert a["scanningFresh"] is None, "an ordinary rescan is not a partial index"
    assert a["unreachable"] == "unreachable"
    assert a["finished"] is None


def test_quality_card_follows_the_qa_job_wherever_it_is_reported(node_out):
    q = node_out["qa"]
    assert q["fromSummary"] == "q-run"
    assert q["pollBeatsQueuedSummary"] == "q-run", "the running job wins over a queued one the summary reported (N2)"
    assert q["fromDatasetJob"] == "q-run"
    assert q["otherDataset"] is None and q["notQa"] is None and q["doneIgnored"] is None
    assert q["freshestSnapshot"] == "running"


def test_quality_card_mode(node_out):
    m = node_out["mode"]
    assert m == {"jobWins": "running", "summaryRunning": "running", "scanning": "scanning", "never": "never",
                 "stale": "stale", "ok": "ok", "empty": "never"}


def test_placeholder_class_names(node_out):
    assert node_out["placeholder"] == [True, True, True, True, True, True, True, True, False, False, False, False]


def test_latest_index_report(node_out):
    r = node_out["report"]
    assert [p["code"] for p in r["problems"]] == ["stat_error", "labels_without_images"], "warnings first, empty kinds dropped"
    assert (r["warn"], r["info"]) == (1, 1)
    assert len(r["problems"][1]["examples"]) == 5, "at most five examples per problem"
    assert r["notes"] == [] and r["images"] == 7808 and r["elapsed"] == 5.3
    legacy = node_out["reportLegacy"]
    assert legacy["notes"] == ["3 list entries not found"] and legacy["warn"] == 1
    assert node_out["reportNone"] is None
    clean = node_out["reportClean"]
    assert clean["problems"] == [] and clean["warn"] == 0


# ------------------------------------------------------------------ static checks

def test_shell_files_parse_as_modules():
    if not NODE:
        pytest.skip("node is not installed")
    for rel in SHELL_FILES:
        r = subprocess.run([NODE, "--input-type=module", "--check"], input=(STATIC / rel).read_text(encoding="utf-8"),
                           capture_output=True, text=True, timeout=60)
        assert r.returncode == 0, f"{rel}: {r.stderr}"


def test_every_static_module_is_in_the_import_map():
    """A module missing from app.html's list still loads, but unversioned — it is revalidated on every load and an
    edit can be served stale next to fresh modules."""
    html = (SRC / "templates" / "app.html").read_text(encoding="utf-8")
    listed = set(re.findall(r"'((?:lib|views)/[\w./-]+\.js)'", html))
    present = {p.relative_to(STATIC).as_posix() for d in ("lib", "views") for p in (STATIC / d).glob("*.js")}
    assert present <= listed, f"add to the mods list in templates/app.html: {sorted(present - listed)}"
    assert listed <= present, f"listed but missing: {sorted(listed - present)}"


def test_job_banner_never_reflows_the_shell():
    """N3: the job status lives in a fixed-height slot of the header's path line; no banner row is inserted."""
    js = (STATIC / "views" / "dataset.js").read_text(encoding="utf-8")
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    assert "ds-scan" not in js and ".ds-scan" not in css
    assert re.search(r"\.ds-header \.path \{[^}]*height: 28px", css), "the path line has a fixed height"
    assert re.search(r"\.ds-job \{[^}]*height: 26px", css)
    assert "s.jobSlot" in js and "el.appendChild(alert); el.appendChild(tabBar)" in js


def test_overview_links_examples_to_objects_by_class():
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "/objects?cls=${i}" in ov
    assert "openSuggestNames" in ov and "ai?tab=cards&suggest=1" in ov
    objects = (STATIC / "views" / "objects.js").read_text(encoding="utf-8")
    assert re.search(r"q0\.get\('cls'\)", objects), "Objects must keep reading ?cls= (Overview's Examples links)"


def test_export_sends_the_dataset_card_option_the_server_reads():
    from fovea.core.export import parse_options
    ex = (STATIC / "views" / "export.js").read_text(encoding="utf-8")
    assert "o.dataset_card = true" in ex
    assert parse_options({"dataset_card": True}).dataset_card is True
    assert parse_options({}).dataset_card is False


# ------------------------------------------------------------------ server contracts the shell relies on

def _make_dataset(tmp_path):
    """train: 4 images with labels, val: 2 images; a stray labels-only folder that the scan reports."""
    from fovea.core.scanner import scan_dataset
    from fovea.jobs import Job
    root = tmp_path / f"ds-{uuid.uuid4().hex[:6]}"
    for split, n in (("train", 4), ("val", 2)):
        for k in range(n):
            ip = root / "images" / split / f"{split}_{k:03d}.jpg"
            ip.parent.mkdir(parents=True, exist_ok=True)
            Image.new("RGB", (96, 72), (40 + 30 * k, 90, 120)).save(ip)
            lp = root / "labels" / split / f"{split}_{k:03d}.txt"
            lp.parent.mkdir(parents=True, exist_ok=True)
            lp.write_text("0 0.500000 0.500000 0.300000 0.300000\n")
    orphan = root / "labels" / "extra-only"
    orphan.mkdir(parents=True, exist_ok=True)
    (orphan / "a.txt").write_text("0 0.5 0.5 0.1 0.1\n")
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "shell-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(["class_0"]), "user", "", "new", now, now))
    scan_dataset(ds_id, Job(id="s", kind="scan"))
    return ds_id


@pytest.fixture()
def client(monkeypatch):
    monkeypatch.setenv("FOVEA_AI_MOCK", "1")
    from fastapi.testclient import TestClient
    from fovea.main import app
    return TestClient(app)


@pytest.fixture()
def ds(tmp_path):
    ds_id = _make_dataset(tmp_path)
    yield ds_id
    db.execute("DELETE FROM datasets WHERE id=?", (ds_id,))


def test_dataset_carries_the_latest_index_report(client, ds):
    r = client.get(f"/api/datasets/{ds}")
    assert r.status_code == 200, r.text
    last = r.json()["dataset"]["last_scan"]
    assert isinstance(last, dict) and last.get("images") == 6
    assert isinstance(last.get("problems"), list)
    for p in last["problems"]:
        assert {"code", "label", "severity", "count", "examples"} <= set(p), p
        assert p["severity"] in ("warn", "info") and len(p["examples"]) <= 5
    assert any(p["code"] == "labels_without_images" for p in last["problems"]), last["problems"]


def test_quality_summary_reports_status_and_job(client, ds):
    r = client.get(f"/api/datasets/{ds}/qa/summary", params={"auto": 0})
    assert r.status_code == 200, r.text
    s = r.json()
    assert s["status"] in ("running", "never", "stale", "ok")
    assert "job" in s and "scanning" in s and "computed_at" in s and "counts" in s
    if s["job"] is not None:
        assert s["status"] == "running" and s["job"]["kind"] == "qa"


def test_cache_stats_and_clear(client, ds):
    iid = client.get(f"/api/datasets/{ds}/images", params={"limit": 1}).json()["items"][0]["id"]
    assert client.get(f"/api/thumb/{iid}", params={"s": 128}).status_code == 200
    assert client.get(f"/api/crop/{iid}", params={"box": "0.5,0.5,0.3,0.3", "pad": 0.2, "s": 96}).status_code == 200
    crops = client.get("/api/crops/cache").json()
    assert crops["files"] >= 1 and crops["bytes"] > 0 and "limit_bytes" in crops
    assert client.get("/api/cache/stats").json()["thumb_cache"].keys() >= {"files", "bytes"}
    a = client.post("/api/cache/clear").json()
    b = client.post("/api/crops/cache/clear").json()
    assert isinstance(a["removed"], int) and b["removed"] >= 1
    assert client.get("/api/crops/cache").json()["files"] == 0


def test_dataset_card_standalone_and_in_the_export_preview(client, ds):
    r = client.get(f"/api/datasets/{ds}/card.md")
    assert r.status_code == 200 and r.text.startswith("# ") and "attachment" in r.headers.get("content-disposition", "")
    with_card = client.post(f"/api/datasets/{ds}/export/preview", json={"dataset_card": True}).json()
    assert isinstance(with_card.get("card"), str) and with_card["card"].startswith("# ")
    without = client.post(f"/api/datasets/{ds}/export/preview", json={}).json()
    assert without.get("card") is None


# ------------------------------------------------------------------ review round 1 (shell findings)

SCENARIOS_R1 = r"""
// Browser-only globals the modules touch at load time (listeners, platform sniffing); no DOM is built here.
globalThis.navigator = { platform: 'MacIntel' };
globalThis.document = { addEventListener() {}, removeEventListener() {} };
globalThis.window = { addEventListener() {}, removeEventListener() {} };
const ui = await import('./lib/ui.js');
const ov = await import('./views/overview.js');
const st = await import('./views/settings.js');
const keys = await import('./lib/keys.js');
const files = await import('./views/files.js');
const out = {};

// ---- anchored menus / popovers: never over their own anchor; the CSS cap is never raised
const R = (top, bottom, left = 300, right = 380) => ({ top, bottom, left, right });
out.place = {
  filters: ui.placement(R(100, 130), 260, 680, 1440, 900, 680),        // Explore Filters (tall, CSS-capped)
  pickerLow: ui.placement(R(640, 670), 300, 500, 1440, 900, 884),      // Annotate model picker low on the page
  tallMid: ui.placement(R(400, 430), 300, 884, 1440, 900, 884),        // taller than either side
  small: ui.placement(R(100, 130), 200, 120, 1440, 900, 884),
  cramped: ui.placement(R(130, 160), 200, 500, 1440, 300, 284),        // neither side has room: pinned
  rightEdge: ui.placement(R(100, 130, 1380, 1430), 260, 200, 1440, 900, 884),
};
// ---- Settings: the Flex switch only where a provider reads batch_mode
out.flex = [{ id: 'openai', capabilities: { batch_api: true } }, { id: 'anthropic', capabilities: { batch_api: true } },
  { id: 'gemini', capabilities: { batch_api: true } }, { id: 'mock', capabilities: { batch_api: true } },
  { id: 'future', capabilities: { flex: true } }, null].map(st.hasFlexTier);
// ---- Overview helpers
out.exploreAll = ov.exploreQuery({});
out.exploreIssue = ov.exploreQuery({ issue: 'dark' });
out.median = ov.medianBoxes([{ boxes: 20772 }, { boxes: 322 }, { boxes: 124 }, { boxes: 164 }, { boxes: 18 }, { boxes: 12 }, { boxes: 21 }, { boxes: 0 }, { boxes: 900 }, { boxes: 5000 }, { boxes: 700 }]);
out.medianEmpty = ov.medianBoxes([{ boxes: 0 }]);
out.suspect = ov.suspectLink('vfp', 5680);
out.tiles = [ov.auditTileText('spurious', 973, 509), ov.auditTileText('missing', 16, 15), ov.auditTileText('loose', 1, 1), ov.auditTileText('swap', 4, null)];
const rep = ov.scanReport({ at: 1, images: 7808, problems: [
  { code: 'labels_without_images', label: 'Label files with no matching images folder', severity: 'info', count: 1008, examples: ['labels/test-fall0class (1,008 files)'] },
  { code: 'stat_error', label: 'Images that could not be read', severity: 'warn', count: 2, examples: [] }] });
out.dedupe = ov.withoutLayoutDupes(rep, ['labels/test-fall0class: 1,008 label file(s) with no matching images folder — not indexed']);
out.dedupeNoNote = ov.withoutLayoutDupes(rep, ['Detected dataset in subfolder x']) === rep;
// ---- Files registers its keys for the ? dialog
files.install({ registerTab() {}, keys });
out.filesKeys = keys.keyGroups().filter(g => g.scope === 'files').map(g => ({ title: g.title, keys: g.items.map(i => i[0]) }));
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_r1(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("shell-r1")
    shutil.copytree(STATIC / "lib", d / "lib")
    shutil.copytree(STATIC / "views", d / "views")
    (d / "package.json").write_text('{"type": "module"}')
    (d / "views" / "home.js").write_text("export const sortSplits = (l) => l.slice();\n")   # pulls in browser-only modules
    (d / "run.js").write_text(SCENARIOS_R1, encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.js")], capture_output=True, text=True, timeout=60, cwd=d)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_menus_and_popovers_never_cover_their_anchor(node_r1):
    p = node_r1["place"]
    f = p["filters"]
    assert f["y"] >= 130 and f["maxHeight"] == 680, "below the Filters button, at the .filter-pop cap (not 884)"
    assert f["y"] + f["maxHeight"] <= 900 - 8
    low = p["pickerLow"]
    assert low["y"] + min(500, low["maxHeight"]) <= 640, "a tall menu low on the page opens above its anchor"
    assert low["y"] >= 8
    mid = p["tallMid"]
    assert mid["y"] >= 430 and mid["y"] + mid["maxHeight"] <= 892, "too tall for both sides: the roomier side, scrolling"
    assert (p["small"]["x"], p["small"]["y"]) == (300, 134), "a menu that fits opens just below its anchor"
    assert p["cramped"]["y"] == 8 and p["cramped"]["maxHeight"] <= 284, "pinned only when neither side has room"
    assert p["rightEdge"]["x"] + 260 <= 1440 - 8


def test_flex_switch_only_where_a_provider_reads_it(node_r1):
    assert node_r1["flex"] == [True, False, False, False, True, False]
    js = (STATIC / "views" / "settings.js").read_text(encoding="utf-8")
    assert "About half the price; results within 24 h')" not in js, "the old promise on every batch-capable provider"
    assert "hasFlexTier(p)" in js and "Economy" in js
    # the provider code is still the only reader of batch_mode, and it is OpenAI's
    readers = [p.name for p in (SRC / "fovea" / "ai" / "providers").glob("*.py") if "batch_mode" in p.read_text(encoding="utf-8")]
    assert readers == ["openai.py"], readers


def test_overview_links_say_what_they_open(node_r1):
    assert node_r1["exploreAll"] == "sort=id", "an empty query would restore Explore's last session filters"
    assert node_r1["exploreIssue"] == "issue=dark"
    assert node_r1["suspect"] == "/d/vfp/annotate?img=5680&ai=audit&sort=audit&order=desc"
    assert node_r1["tiles"] == [{"unit": "boxes", "where": "in 509 images"}, {"unit": "objects", "where": "in 15 images"},
                                {"unit": "box", "where": "in 1 image"}, {"unit": "boxes", "where": ""}]


def test_under_represented_rule_matches_the_server(node_r1):
    from fovea.core import stats
    assert stats.UNDER_REPRESENTED == 0.1, "the Health check text says 10% of the median class"
    assert node_r1["median"] == 322 and node_r1["medianEmpty"] == 0
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "< 5% of the largest class" not in ov and "10% of the median class" in ov


def test_scan_report_does_not_repeat_a_layout_note(node_r1):
    d = node_r1["dedupe"]
    assert [p["code"] for p in d["problems"]] == ["stat_error"] and d["shownAbove"] == 1
    assert (d["warn"], d["info"]) == (1, 0)
    assert node_r1["dedupeNoNote"] is True


def test_files_keys_are_listed_in_the_help(node_r1):
    assert node_r1["filesKeys"] == [{"title": "Files", "keys": ["↑ ↓", "PageUp PageDown", "Home End", "Enter", "Backspace"]}]


def test_rest_reminder_ignores_escapes_meant_for_other_layers():
    js = (STATIC / "lib" / "comfort.js").read_text(encoding="utf-8")
    on_input = re.search(r"function onInput\([^)]*\) \{(.*?)\n\}", js, re.S).group(1)
    assert "Escape" not in on_input, "the window-capture activity listener runs before every other handler"
    assert re.search(r"class: 'rest-banner'[^\n]*onKeydown: \(e\) => \{ if \(e\.key === 'Escape'", js), "Escape works with focus in the reminder"


def test_restore_follows_its_own_job():
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    body = re.search(r"function followRestore\(job, onDone\) \{(.*?)\n  \}\n", ov, re.S).group(1)
    assert "api.watchJob(job.id)" in body and "draw(); loadHistory();" in body and "refresh()" in body
    assert "followRestore(r && r.job, onDone)" in ov, "restore() hands its job to followRestore"
    assert "restore(s, [it.image_id], requery)" in ov, "the open diff is compared again after a per-image restore"


def test_zip_download_shows_that_it_is_working():
    ex = (STATIC / "views" / "export.js").read_text(encoding="utf-8")
    body = re.search(r"async function downloadZip\(\) \{(.*?)\n  \}\n", ex, re.S).group(1)
    assert "new AbortController()" in body and "signal: ctrl.signal" in body
    assert "}, 400);" in body and "ui.spinner()" in body and "ui.elapsedClock(t0)" in body and "ctrl.abort()" in body
    assert "if (zipCtrl) zipCtrl.abort()" in ex, "leaving the page abandons the request"


def _tokens(css: str, selector_re: str) -> dict:
    block = re.search(selector_re + r"\s*\{(.*?)\n\s*\}", css, re.S | re.M).group(1)
    return dict(re.findall(r"(--[\w-]+):\s*([^;]+);", block))


def _lum(rgb):
    def ch(v):
        v /= 255
        return v / 12.92 if v <= 0.03928 else ((v + .055) / 1.055) ** 2.4
    r, g, b = rgb
    return .2126 * ch(r) + .7152 * ch(g) + .0722 * ch(b)


def _rgb(v: str):
    v = v.strip()
    if v.startswith("#"):
        return tuple(int(v[i:i + 2], 16) for i in (1, 3, 5)), 1.0
    nums = [float(x) for x in re.findall(r"[\d.]+", v)]
    return tuple(int(n) for n in nums[:3]), (nums[3] if len(nums) > 3 else 1.0)


def _contrast(fg, bg):
    a, b = _lum(fg), _lum(bg)
    return (max(a, b) + .05) / (min(a, b) + .05)


def test_faint_and_status_text_reach_aa():
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    light = _tokens(css, r"^:root")
    dark = _tokens(css, r"^\[data-theme=\"dark\"\]")
    system = _tokens(css, r"\[data-theme=\"system\"\]")
    for t in (dark, system):
        surface, _ = _rgb(t["--surface"])
        acc, a = _rgb(t["--accent-soft"])
        active = tuple(round(c * a + s * (1 - a)) for c, s in zip(acc, surface))
        text3, _ = _rgb(t["--text-3"])
        assert _contrast(text3, active) >= 4.5, "faint counts on an active (accent-soft) row"
        assert _contrast(text3, _rgb(t["--surface-2"])[0]) >= 4.5
    ok, _ = _rgb(light["--ok-text"])
    for bg in ("--surface-2", "--surface", "--ok-soft", "--bg"):
        assert _contrast(ok, _rgb(light[bg])[0]) >= 4.5, bg


def test_shell_css_review_fixes():
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    focus = re.search(r"\.menu-item:focus-visible \{([^}]*)\}", css).group(1)
    assert "var(--accent" in focus, "keyboard focus in menus is more than a 1.1:1 tint"
    assert re.search(r"\.table td\.num \{[^}]*font-family: var\(--mono\)", css)
    assert not re.search(r"\.table \.num \{[^}]*font-family", css), "number headers stay in the text font"
    assert re.search(r"\.hbar \{[^}]*grid-template-columns: subgrid", css), "class bars share one scale"
    assert re.search(r"\.stat \.s \{[^}]*-webkit-line-clamp: 2", css)
    assert ":last-child:nth-child(4n+3) { grid-column: span 2; }" in css, "7 KPI tiles in 4 columns leave no hole"
    assert re.search(r"\.legend-swatch \{[^}]*flex: none", css)
    assert re.search(r"\.hbar \.track \.bar::after \{[^}]*border: 1px solid var\(--edge\)", css), "pale split segments keep a visible edge"
    ui = (STATIC / "lib" / "ui.js").read_text(encoding="utf-8")
    assert "title: typeof sub === 'string' ? sub : null" in ui, "a clamped tile subtitle is readable on hover"


def test_job_names_match_the_ai_tab():
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    assert "ai_batch: ['AI labeling'" in app and "Auto-labeling" not in app
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    ai = (STATIC / "views" / "ai.js").read_text(encoding="utf-8")
    for label in ("What should a model look for?", "Box it when…", "Don’t box…", "Example boxes"):
        assert label in ov and label in ai, label


def test_audit_tiles_count_boxes_and_images(client, ds):
    """Overview's audit tiles: findings (boxes) from /audit/summary, images (what the link opens) from the issue
    facet — two boxes on one image and one on another read "3 boxes in 2 images"."""
    ids = [it["id"] for it in client.get(f"/api/datasets/{ds}/images", params={"limit": 2}).json()["items"]]
    now = db.now()
    run_id = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, status, created_at, finished_at) "
                        "VALUES(?,?,?,?,?,?,?)", (ds, "audit", "mock", "m", "done", now, now)).lastrowid
    for iid in (ids[0], ids[0], ids[1]):
        db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind, state, created_at) "
                   "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (run_id, ds, iid, 0, .5, .5, .3, .3, "spurious", "pending", now))
    s = client.get(f"/api/datasets/{ds}/audit/summary", params={"top": 6}).json()
    assert s["counts"]["spurious"] == 3 and s["images_flagged"] == 2
    f = client.get(f"/api/datasets/{ds}/images/facets").json()["facets"]
    assert f["issue"].get("ai_spurious") == 2


def test_restore_answers_with_a_job_to_follow(client, ds):
    import time
    snap = client.post(f"/api/datasets/{ds}/snapshots", json={"name": "before"}).json()
    sid = snap.get("snapshot_id")
    if sid is None:
        job = snap["job"]["id"]
        for _ in range(100):
            j = client.get(f"/api/jobs/{job}").json()["job"]
            if j["status"] in ("done", "error", "cancelled"):
                break
            time.sleep(.05)
        sid = client.get(f"/api/datasets/{ds}/snapshots").json()["snapshots"][0]["id"]
    r = client.post(f"/api/datasets/{ds}/snapshots/{sid}/restore", json={})
    assert r.status_code == 200, r.text
    job = r.json()["job"]
    assert job["id"] and job["kind"] == "restore", "Overview follows this job itself (api.watchJob(job.id))"
    for _ in range(100):
        j = client.get(f"/api/jobs/{job['id']}").json()["job"]
        if j["status"] in ("done", "error", "cancelled"):
            break
        time.sleep(.05)
    assert j["status"] == "done", j


# ------------------------------------------------------------------ review round 2 (shell findings)

SCENARIOS_R2 = r"""
globalThis.navigator = { platform: 'Win32' };          // NumLock exists on Windows / Linux keyboards
globalThis.document = { addEventListener() {}, removeEventListener() {} };
globalThis.window = { addEventListener() {}, removeEventListener() {} };
const keys = await import('./lib/keys.js');
const ov = await import('./views/overview.js');
const home = await import('./views/home.js');
const { fmt } = await import('./lib/dom.js');
const out = {};
// ---- numeric keypad: a digit only while it types one (NumLock off sends navigation keys with the same codes)
const K = (key, code, extra = {}) => ({ key, code, location: code.startsWith('Numpad') ? 3 : 0, shiftKey: false, altKey: false, metaKey: false, ctrlKey: false, ...extra });
out.digit = {
  numLockOn: keys.digit(K('6', 'Numpad6')),
  arrowRight: keys.digit(K('ArrowRight', 'Numpad6')),
  pageUp: keys.digit(K('PageUp', 'Numpad9')),
  arrowUp: keys.digit(K('ArrowUp', 'Numpad8')),
  insert: keys.digit(K('Insert', 'Numpad0')),
  clear: keys.digit(K('Clear', 'Numpad5')),
  row: keys.digit(K('6', 'Digit6')),
  rowKorean: keys.digit(K('Process', 'Digit3')),
  synthetic: keys.digit({ key: '7' }),
};
out.isArrow = keys.is(K('ArrowRight', 'Numpad6'), 'ArrowRight');
out.isSix = keys.is(K('ArrowRight', 'Numpad6'), '6');
out.isSixOn = keys.is(K('6', 'Numpad6'), '6');
// ---- restore confirmation: what a full restore really rewrites
const f = { ...fmt, ago: () => '1m ago', date: () => 'Oct 14, 2026, 10:00' };
const snap = { created_at: 1, n_files: 7808 };
out.restore = {
  drawer: ov.restoreText(snap, null, { images_changed: 24 }, f),
  one: ov.restoreText(snap, null, { images_changed: 1 }, f),
  all: ov.restoreText({ created_at: 1, n_files: 3 }, null, { images_changed: 3 }, f),
  unknown: ov.restoreText(snap, null, null, f),
  image: ov.restoreText(snap, [5], null, f),
  images: ov.restoreText(snap, [5, 6], null, f),
};
out.boxes = [ov.diffBoxText({ boxes_added: 1, boxes_removed: 2 }, f), ov.diffBoxText({ boxes_added: 0, boxes_removed: 1 }, f), ov.diffBoxText({}, f)];
// ---- Dataset card split column
out.col = [ov.splitColWidth(['train', 'val', 'test', 'test-1854-1860-포함']), ov.splitColWidth(['train']), ov.splitColWidth([]), ov.splitColWidth(['x'.repeat(80)])];
// ---- Home card split list: whole names only
out.splits = home.splitLabels(['train', 'val', 'test', 'test-1854-1860-포함']);
out.splitsOne = home.splitLabels(['train']);
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_r2(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("shell-r2")
    shutil.copytree(STATIC / "lib", d / "lib")
    shutil.copytree(STATIC / "views", d / "views")
    (d / "package.json").write_text('{"type": "module"}')
    (d / "run.js").write_text(SCENARIOS_R2, encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.js")], capture_output=True, text=True, timeout=60, cwd=d)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_numpad_navigation_keys_never_pick_a_class(node_r2):
    """NumLock off: numpad 6 arrives as ArrowRight, 9 as PageUp — same codes as the digits. Objects and Annotate
    check digit() before their navigation keys, so a keypad code alone reassigned the box."""
    d = node_r2["digit"]
    assert d["numLockOn"] == "6" and d["row"] == "6" and d["synthetic"] == "7"
    assert d["rowKorean"] == "3", "the digit row is matched by position whatever the input source sends"
    assert [d[k] for k in ("arrowRight", "pageUp", "arrowUp", "insert", "clear")] == [""] * 5
    assert node_r2["isArrow"] is True and node_r2["isSix"] is False and node_r2["isSixOn"] is True


def test_restore_says_how_many_files_it_rewrites(node_r2):
    r = node_r2["restore"]
    assert r["drawer"] == ("24 changed label files will be put back as they were 1m ago (Oct 14, 2026, 10:00). "
                           "The other 7,784 already match it and are not touched.")
    assert r["one"].startswith("1 changed label file will be put back as it was 1m ago")
    assert r["all"] == "3 changed label files will be put back as they were 1m ago (Oct 14, 2026, 10:00).", "nothing 'other' to mention"
    assert "7,808 label files will be put back" not in r["unknown"] and "differs from this version" in r["unknown"]
    assert r["image"] == "This image’s label file will be put back as it was 1m ago (Oct 14, 2026, 10:00)."
    assert r["images"].startswith("2 label files will be put back as they were")
    assert node_r2["boxes"] == [{"added": "+1 box", "removed": "−2 boxes"}, {"added": None, "removed": "−1 box"}, {"added": None, "removed": None}]


def test_history_drawer_restore_all_waits_for_a_difference():
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    body = re.search(r"function openDiff\(s\) \{(.*?)\n    page\(\);\n  \}", ov, re.S).group(1)
    assert re.search(r"const restoreAll = h\('button', \{ class: 'btn btn-danger', disabled: true", body), "disabled until the comparison is in"
    assert "setRestoreAll(summary.images_changed > 0" in body
    assert "restore(s, null, null, summary)" in body, "the drawer hands its diff to the confirmation"
    assert "boxes`" not in body, "box counts go through fmt.plural (no '+1 boxes')"
    # from the ⋯ menu, the diff is asked for first and a version that matches is not restored at all
    rest = re.search(r"async function restore\(s, imageIds, onDone, summary\) \{(.*?)\n  \}\n", ov, re.S).group(1)
    assert "/snapshots/${s.id}/diff`, { limit: 1 }" in rest and "!summary.images_changed" in rest


def test_dataset_card_split_column_fits_its_names(node_r2):
    long, short, empty, huge = node_r2["col"]
    assert long == "max(110px, calc(19ch + 28px))", "test-1854-1860-포함: 17 characters, two of them wide (2 ch each)"
    assert short == "max(110px, calc(5ch + 28px))" and empty == "max(110px, calc(0ch + 28px))"
    assert huge == "max(110px, calc(32ch + 28px))", "capped; a % cap would make Chrome ignore the width (equal thirds)"
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "h('col', { style: 'width:110px' })" not in ov


def test_home_split_list_drops_whole_names(node_r2):
    nb = " / "
    assert node_r2["splits"] == [nb.join(["train", "val", "test", "test-1854-1860-포함"]), nb.join(["train", "val", "test", "+1"]),
                                 nb.join(["train", "val", "+2"]), nb.join(["train", "+3"]), "4 splits"]
    assert node_r2["splitsOne"] == ["train"]
    home = (STATIC / "views" / "home.js").read_text(encoding="utf-8")
    assert "ui.midText(splits.join" not in home, "a middle ellipsis over the joined list cut through a name"
    assert "new ResizeObserver" in home and "splitFit.disconnect()" in home
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    assert re.search(r"\.ds-card \.foot \.splits \{[^}]*flex: 1 1 0", css), "its width must not follow its text (no resize loop)"


def test_stat_tile_label_truncates_with_an_ellipsis():
    ui = (STATIC / "lib" / "ui.js").read_text(encoding="utf-8")
    assert "typeof label === 'string' ? h('span', { class: 'truncate', title: label }, label) : label" in ui
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    assert ".stat .k > .truncate { min-width: 0; }" in css


def test_class_name_typing_keeps_focus():
    """A keystroke must not rebuild the class rows: the first letter used to flip 'dirty', rebuild every input and
    leave focus on <body>, so the rest of the name ran global shortcuts (g e → Explore)."""
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    set_dirty = re.search(r"function setDirty\(v\) \{(.*?)\n  \}", ov, re.S).group(1)
    assert "renderClasses" not in set_dirty and "renderClassFoot()" in set_dirty
    on_input = re.search(r"onInput: \(e\) => \{ names\[i\] = e\.target\.value;(.*?)\},\n", ov, re.S).group(1)
    assert "renderClasses" not in on_input and "classList.toggle('dirty'" in on_input
    rc = re.search(r"function renderClasses\(\) \{\n    const back = classFocus\(\);(.*?)\n  \}\n", ov, re.S)
    assert rc and rc.group(1).rstrip().endswith("back();"), "a rebuild (cards loaded, Enter to save) puts the caret back"
    draw = re.search(r"async function draw\(\) \{(.*?)\n  \}\n", ov, re.S).group(1)
    assert draw.index("const back = classFocus()") < draw.index("build(stats)") < draw.index("back();")


def _handlers_reading_current_target_after_await(src: str):
    """async (x) => { … await … x.currentTarget … }: currentTarget is null once dispatch has ended."""
    bad = []
    for m in re.finditer(r"async \((\w+)\) => \{", src):
        var, i, depth = m.group(1), m.end(), 1
        while depth and i < len(src):
            depth += {"{": 1, "}": -1}.get(src[i], 0)
            i += 1
        body = src[m.end():i]
        a = body.find("await ")
        if a >= 0 and f"{var}.currentTarget" in body[a:]:
            bad.append(src[:m.start()].count("\n") + 1)
    return bad


def test_no_shell_handler_reads_current_target_after_await():
    found = {rel: _handlers_reading_current_target_after_await((STATIC / rel).read_text(encoding="utf-8")) for rel in SHELL_FILES}
    assert not {k: v for k, v in found.items() if v}, found
    assert _handlers_reading_current_target_after_await("x(async (e) => { e.currentTarget.disabled = true; await f(); e.currentTarget.disabled = false; })") == [1]


# ------------------------------------------------------------------ review round 3 (shell findings)

SCENARIOS_R3 = r"""
globalThis.navigator = { platform: 'MacIntel' };
globalThis.document = { addEventListener() {}, removeEventListener() {} };
globalThis.window = { addEventListener() {}, removeEventListener() {} };
const { fmt } = await import('./lib/dom.js');
const ov = await import('./views/overview.js');
const st = await import('./views/settings.js');
const { jobText } = await import('./jobtext.js');
const out = {};
// ---- job banner: a provider wait never repeats the counts the banner already shows
const J = (kind, done, total, message) => ({ kind, status: 'running', done, total, message });
out.job = {
  wait: jobText(J('ai_batch', 12, 100, 'Anthropic Claude rate limited — retrying in 12 s · 12/100 · 30 boxes · $0.01')),
  waitAudit: jobText(J('ai_audit', 1200, 5000, 'Rate limited — retrying in 3 s · 1,200/5,000 · 40 images with findings · $0.12')),
  plain: jobText(J('ai_batch', 12, 100, '12/100 · 30 boxes · $0.01')),
  index: jobText(J('scan', 6400, 7808, 'Indexing 6,400/7,808')),
  otherCount: jobText(J('ai_batch', 12, 100, 'Rate limited — retrying in 3 s · Reading Anthropic batch results · 50/120')),
  labelled: jobText(J('ai_benchmark', 3, 10, 'Rate limited — retrying in 5 s · YOLO n: 3/10')),
  queued: jobText({ kind: 'ai_batch', status: 'queued', queue_position: 2 }),
};
// ---- Settings → Usage: runs of a model with no price on file are "cost unknown", apart from the dollars
const U = (by_provider, unpriced, total_usd, by_model) => ({ total_usd, by_provider, by_model, unpriced });
out.usage = {
  none: st.usageSummary(U({ anthropic: { usd: 0.42, runs: 2, images: 300 } }, { runs: 0, images: 0, by_provider: {} }, 0.42), fmt),
  legacy: st.usageSummary(U({ mock: { usd: 5, runs: 1, images: 5000 } }, undefined, 5), fmt),
  all: st.usageSummary(U({ anthropic: { usd: 0, runs: 1, images: 200 } }, { runs: 1, images: 200, by_provider: { anthropic: { runs: 1, images: 200 } } }, 0), fmt),
  mixed: st.usageSummary(U({ mock: { usd: 5, runs: 1, images: 5000 }, anthropic: { usd: 0.42, runs: 2, images: 300 } },
    { runs: 1, images: 200, by_provider: { anthropic: { runs: 1, images: 200 } } }, 5.42), fmt),
  noImages: st.usageSummary(U({ gemini: { usd: 0.2, runs: 3, images: 10 } }, { runs: 1, images: 0, by_provider: { gemini: { runs: 1, images: 0 } } }, 0.2), fmt),
  // a two-stage run counts once in "across N runs" (by_model), although two providers billed it
  twoStage: st.usageSummary(U({ yoloe: { usd: 0, runs: 1, images: 4 }, anthropic: { usd: 0.1, runs: 1, images: 4 } }, { runs: 0, images: 0, by_provider: {} }, 0.1,
    { 'yoloe+anthropic/x': { runs: 1, usd: 0.1, images: 4 } }), fmt),
};
// ---- Overview: History retention copy, suspect counts
out.retention = ov.retentionText({ keep_auto: 20, keep_small: 50, small_files: 20 });
out.retentionLegacy = ov.retentionText({ keep_auto: 20 });
out.counts = [ov.auditCountsText({ missing: 1, swap: 0, loose: 2, spurious: 1 }), ov.auditCountsText({}), ov.auditCountsText(null)];
// ---- review round 4: the suspect rows speak the AI tab's words (ai.js KIND_COUNT), in the tiles' order
out.kindCount = (await import('./views/ai.js')).KIND_COUNT;
out.countsOrder = ov.auditCountsText({ spurious: 1, other: 3, swap: 2, missing: 1, loose: 0 });
// ---- review round 4: what Settings in a browser on another machine draws disabled
const PV = (id, kind, extra = {}) => ({ id, kind, needs_key: false, key: null, ...extra });
const claude = PV('anthropic', 'cloud', { needs_key: true, key: { present: true } });
out.locks = {
  cloudRemote: st.remoteLocks(claude, { request_is_local: false }, false),
  compatRemote: st.remoteLocks(PV('openai_compat', 'local', { key: { present: false } }), { request_is_local: false }, false),
  mockRemote: st.remoteLocks(PV('mock', 'cloud'), { request_is_local: false }, false),
  yoloRemote: st.remoteLocks(PV('yolo', 'local'), { request_is_local: false }, false),
  cloudAllowed: st.remoteLocks(claude, { request_is_local: false }, true),
  cloudLocal: st.remoteLocks(claude, { request_is_local: true }, false),
  noMeta: st.remoteLocks(claude, null, false),
};
// ---- final round: Overview Review links open the most suspicious first; the cap line says what runs still going hold
out.review = { qa: ov.exploreQuery(ov.flagReview('blurry')), kind: ov.exploreQuery(ov.auditReview({ issue: 'ai_missing' })),
  all: ov.exploreQuery(ov.auditReview({ ai: 'audit' })) };
const CAP = (cap_usd, spent_30d, reserved_usd) => ({ cap_usd, spent_30d, reserved_usd });
out.cap = {
  held: st.capUsageText(CAP(5, 0.23381, 0.916), 0, 0, fmt),
  idle: st.capUsageText(CAP(5, 0.2, 0), 0, 0, fmt),
  legacy: st.capUsageText({ cap_usd: 5, spent_30d: 1 }, 0, 0, fmt),            // a server without the field
  noCap: st.capUsageText(undefined, 0.42, 2, fmt),
  heldUnpriced: st.capUsageText(CAP(5, 1, 0.5), 0, 1, fmt),
  room: st.capStatus(CAP(5, 1, 0.5), fmt),
  reached: st.capStatus(CAP(5, 5.2, 0.5), fmt),
  takenByRuns: st.capStatus(CAP(1, 0.23, 0.92), fmt),
  exactlyHeld: st.capStatus(CAP(1, 0.5, 0.5), fmt),
  noCapStatus: st.capStatus(undefined, fmt),
  legacyStatus: st.capStatus({ cap_usd: 1, spent_30d: 0.5 }, fmt),
};
console.log(JSON.stringify(out));
"""


@pytest.fixture(scope="module")
def node_r3(tmp_path_factory):
    if not NODE:
        pytest.skip("node is not installed")
    d = tmp_path_factory.mktemp("shell-r3")
    shutil.copytree(STATIC / "lib", d / "lib")
    shutil.copytree(STATIC / "views", d / "views")
    (d / "package.json").write_text('{"type": "module"}')
    # app.js wires the whole browser shell on import; its job-text helper (and the label table it reads) is lifted out
    app = (STATIC / "app.js").read_text(encoding="utf-8")
    kinds = re.search(r"^const JOB_KIND = \{.*?\n\};\n", app, re.S | re.M).group(0)
    label = re.search(r"^const jobLabel = .*?;\n", app, re.M).group(0)
    fn = re.search(r"^function jobText\(j\) \{.*?\n\}\n", app, re.S | re.M).group(0)
    (d / "jobtext.js").write_text("import { fmt } from './lib/dom.js';\n" + kinds + label + "export " + fn, encoding="utf-8")
    (d / "run.js").write_text(SCENARIOS_R3, encoding="utf-8")
    r = subprocess.run([NODE, str(d / "run.js")], capture_output=True, text=True, timeout=60, cwd=d)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout.strip().splitlines()[-1])


def test_job_banner_does_not_repeat_counts_after_a_provider_wait(node_r3):
    j = node_r3["job"]
    assert j["wait"] == "AI labeling · 12 / 100 · Anthropic Claude rate limited — retrying in 12 s · 30 boxes · $0.01"
    assert j["waitAudit"] == "Label audit · 1,200 / 5,000 · Rate limited — retrying in 3 s · 40 images with findings · $0.12"
    assert j["plain"] == "AI labeling · 12 / 100 · 30 boxes · $0.01"
    assert j["index"] == "Indexing · 6,400 / 7,808"
    assert j["otherCount"].endswith("Reading Anthropic batch results · 50/120"), "a different count says something else"
    assert j["labelled"].endswith("YOLO n: 3/10"), "a count with its own label stays"
    assert j["queued"] == "AI labeling · queued (#2)"


def test_usage_shows_unpriced_runs_apart(node_r3):
    u = node_r3["usage"]
    assert u["none"] == {"total": "$0.42", "runs": "across 2 runs", "note": "", "rows": [{"id": "anthropic", "usd": 0.42, "text": "$0.42 · 300 images", "unpriced": 0}]}
    assert u["legacy"]["rows"][0]["text"] == "$5.00 · 5,000 images" and u["legacy"]["note"] == "", "a server without the field reads as before"
    a = u["all"]
    assert a["total"] == "Cost unknown" and a["rows"][0]["text"] == "cost unknown · 200 images", "never '$0 · 200 images'"
    assert a["note"] == "This run (200 images) used a model with no price on file, so its cost is unknown."
    m = u["mixed"]
    assert m["total"] == "$5.42" and m["runs"] == "across 3 runs"
    assert [r["id"] for r in m["rows"]] == ["mock", "anthropic"]
    assert m["rows"][1]["text"] == "$0.42 · 100 images + 200 images cost unknown"
    assert m["note"] == "1 run (200 images) used a model with no price on file, so its cost is unknown and not in this total."
    assert u["noImages"]["rows"][0]["text"] == "$0.20 · 10 images + 1 run cost unknown"
    assert u["twoStage"]["runs"] == "across 1 run", "runs are counted once, not once per provider"


def test_history_states_the_real_retention(node_r3):
    assert node_r3["retention"] == ("The newest 20 automatic versions of more than 20 files are kept, plus the newest 50 smaller ones; "
                                    "versions you save stay until you delete them.")
    assert node_r3["retentionLegacy"].startswith("The newest 20 automatic versions are kept;")
    from fovea.core import snapshots as snap
    assert (snap.KEEP_AUTO, snap.KEEP_SMALL, snap.SMALL_FILES) == (20, 50, 20), "update the copy test with the constants"
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "(the newest ', String(r.keep_auto" not in ov and "${retentionText(r)}" in ov
    assert "allVersions(snaps, r)" in ov and "retentionText(r)), snaps.map(snapRow)" in ov, "the drawer says it too"
    assert "/snapshots`, { limit: 200 }" in ov, "70 automatic versions plus saved ones: 50 hid some from 'All versions'"


def test_suspect_rows_never_paint_over_their_counts(node_r3):
    kc = node_r3["kindCount"]
    assert set(kc) == {"missing", "loose", "swap", "spurious"} and all(len(v) == 2 for v in kc.values())
    assert node_r3["counts"] == [f"1 {kc['missing'][0]} · 2 {kc['loose'][1]} · 1 {kc['spurious'][0]}", "", ""], \
        "the AI tab's words with plurals, never the raw kind keys ('1 missing · 2 loose · 1 spurious')"
    assert node_r3["countsOrder"] == f"1 {kc['missing'][0]} · 2 {kc['swap'][1]} · 1 {kc['spurious'][0]} · 3 other", \
        "the tiles' order; a kind the client does not know yet still shows"
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    mid = re.search(r"^\.mid \{([^}]*)\}", css, re.M).group(1)
    assert "overflow: hidden" in mid, "the tail is clipped inside .mid, never painted over a neighbour"
    head = re.search(r"^\.mid > \.h \{([^}]*)\}", css, re.M).group(1)
    tail = re.search(r"^\.mid > \.t \{([^}]*)\}", css, re.M).group(1)
    shrink = float(re.search(r"flex: 0 (\d+) auto", head).group(1))
    assert shrink >= 1e6, "the head gives way first; at 1000 the tail still lost ~0.03 px and showed '…'"
    assert "flex: none" not in tail and "min-width: 0" in tail and "text-overflow: ellipsis" in tail
    row = re.search(r"^\.top-list a \{([^}]*)\}", css, re.M).group(1)
    assert "flex-wrap: wrap" in row, "counts that do not fit beside the whole path go to a second line"
    assert re.search(r"\.top-list \.pth \{[^}]*flex: 1 1 auto; min-width: 0", css)
    assert re.search(r"\.top-list \.cnt \{[^}]*margin-left: auto[^}]*text-overflow: ellipsis", css)
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "findIndex(x => x.image_id === it.image_id)" not in ov, "the server's audit summary already drops repeats"
    assert "h('span', { class: 'pth' }, h('span', { class: 'sc' }" in ov and "class: 'cnt xs faint', title: counts" in ov


def test_splits_card_keeps_split_names_whole():
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    body = re.search(r"function splitsCard\(stats\) \{(.*?)\n  \}\n", ov, re.S).group(1)
    assert "h('table', { class: 'table snug' }" in body and body.rstrip().endswith("'span-5');")
    assert "ui.midText(s.split, { tail: 8 })" in body, "a longer name keeps both ends (full name in the tooltip)"
    heat = re.search(r"function heatCard\(stats\) \{(.*?)\n  \}\n", ov, re.S).group(1)
    assert heat.count("'span-3')") == 2, "the map beside it is at most 240 px wide either way"
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    assert ".table.snug th, .table.snug td { padding-left: 6px; padding-right: 6px; }" in css
    assert re.search(r"@container \(max-width: 1000px\) \{[^}]*\.ov-grid > \.span-3 \{ grid-column: span 6; \}", css), "narrow: two per row as before"


def test_usage_lists_unpriced_runs(client, ds):
    """Runs a cloud model with no price on file billed are booked at $0: the usage endpoint lists them apart so
    Settings says "cost unknown" instead of "$0" (a ledger share with tokens but no cost, or a model the price table
    does not know). Local runs and calls that never billed are not "unpriced"."""
    from fovea.ai import store
    from fovea.ai.base import Usage
    try:
        priced = store.create_run(ds, "batch", "anthropic", "claude-haiku-5-5", {}, status="done")
        store.add_usage(priced, Usage(1000, 100, 0.01), images=3)
        unknown = store.create_run(ds, "batch", "anthropic", "claude-not-on-file-9", {}, status="done")
        store.add_usage(unknown, Usage(2000, 200, 0.0), images=5)
        learned = store.create_run(ds, "propose", "anthropic", "claude-haiku-5-5", {}, status="done")  # booked before its price was on file
        store.add_usage(learned, Usage(500, 50, 0.0), images=1)
        committee = store.create_run(ds, "audit", "anthropic+gemini", "claude-haiku-5-5+gemini-not-on-file", {}, status="done")
        u = Usage(1800, 180, 0.02)
        u.by_provider = {"anthropic": [1000, 100, 0.02], "gemini": [800, 80, 0.0]}
        store.add_usage(committee, u, images=4)
        local = store.create_run(ds, "batch", "yolo", "yolo11n.pt", {}, status="done")
        store.add_usage(local, Usage(), images=7)
        store.create_run(ds, "batch", "anthropic", "claude-haiku-5-5", {}, status="error")   # failed before any call
        r = client.get("/api/ai/usage", params={"dataset_id": ds})
        assert r.status_code == 200, r.text
        un = r.json()["unpriced"]
        assert un["runs"] == 3 and un["images"] == 10, un
        assert un["by_provider"] == {"anthropic": {"runs": 2, "images": 6}, "gemini": {"runs": 1, "images": 4}}
        assert sorted(un["models"]) == ["anthropic+gemini/claude-haiku-5-5+gemini-not-on-file", "anthropic/claude-haiku-5-5",
                                        "anthropic/claude-not-on-file-9"]
        assert abs(r.json()["by_provider"]["anthropic"]["usd"] - 0.03) < 1e-9, "the dollar figures are unchanged"
    finally:
        db.execute("DELETE FROM ai_run_usage WHERE run_id IN (SELECT id FROM ai_runs WHERE dataset_id=?)", (ds,))
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))


# ------------------------------------------------------------------ review round 4

def test_a_clicked_tab_gives_the_arrows_back_to_the_view():
    """A dataset tab clicked with the mouse kept focus, so ←/→/Home/End walked the tab bar (and Enter opened another
    tab) instead of moving the image in Annotate or the tile in Explore / Objects. A pointer click (detail > 0) now
    blurs the tab; Enter (detail 0) keeps the tablist pattern for keyboard users. Live: flows/tabfocus.py."""
    js = (STATIC / "views" / "dataset.js").read_text(encoding="utf-8")
    bar = re.search(r"tabs\.forEach\(t => tabBar\.appendChild\(h\('a'.*?\n    \}\);\n", js, re.S).group(0)
    assert "if (e.detail > 0) e.currentTarget.blur();" in bar
    keys = bar[bar.index("tabBar.addEventListener('keydown'"):]
    assert "if (e.altKey || e.metaKey || e.ctrlKey || e.shiftKey) return;" in keys, "⌘← / Alt+← stay the browser's Back"
    assert "if (i < 0) return;" in keys, "only while a tab itself has focus"
    assert keys.index("if (i < 0) return;") < keys.index("e.preventDefault()")


def test_restore_copy_does_not_claim_to_be_the_only_deletion(node_r3):
    """Annotate's remove_if_empty also deletes a label file (draw then undo, Remove file), so the restore dialog must
    not call restore 'the only case where Fovea deletes a label file' (README names both cases)."""
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "the only case where Fovea deletes a label file" not in ov
    assert "A label file that did not exist in that version is removed.')" in ov
    snap = (SRC / "fovea" / "core" / "snapshots.py").read_text(encoding="utf-8")
    assert "the only case where" not in snap and "remove_if_empty" in snap


def test_class_names_keep_room_on_a_narrow_classes_card():
    """At 900 px the Classes card is about 430 px wide and the fixed 176 px Examples + Card column left 89 px for the
    name ('standing_per'). A query on the class list itself (the old page-width rule never matched where the card is
    narrow) keeps only the Examples icon there; the full name is in the field's title."""
    css = (STATIC / "fovea.css").read_text(encoding="utf-8")
    assert re.search(r"^\.class-list-scroll \{[^}]*container: class-list / inline-size;", css, re.M)
    m = re.search(r"@container class-list \(max-width: (\d+)px\) \{ \.class-edit \.lbl-wide \{ display: none; \} "
                  r"\.class-edit \{ grid-template-columns: 26px 10px minmax\(0, 1fr\) auto 120px; \} \}", css)
    assert m and int(m.group(1)) >= 500, "half-width cards (~370-490 px of list) get the icon-only button"
    assert "@container (min-width: 1001px) and (max-width: 1150px) { .class-edit" not in css
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "value: n, title: n, 'aria-label': `Name of class ${i}`" in ov and "e.target.title = e.target.value;" in ov
    assert "'aria-label': `Examples of class ${i}`" in ov, "the icon-only link keeps its name"


def test_settings_lock_rule(node_r3):
    L = node_r3["locks"]
    assert L["cloudRemote"] == {"remote": True, "key": True, "settings": True, "call": True}
    assert L["compatRemote"] == {"remote": True, "key": True, "settings": True, "call": False}, \
        "its URL decides where images go (locked); Test / Refresh of a server on this machine are allowed"
    assert L["mockRemote"] == {"remote": True, "key": False, "settings": True, "call": True}
    assert L["yoloRemote"] == {"remote": True, "key": False, "settings": False, "call": False}
    for k in ("cloudAllowed", "cloudLocal", "noMeta"):
        assert not any(L[k].values()), k


def test_remote_settings_lock_matches_what_the_server_refuses(client, monkeypatch, tmp_path):
    """Settings from another machine drew every cloud control enabled; each click then failed with a 403 whose
    'API keys:' prefix was wrong for an effort change. The page's rule (settings.remoteLocks) must disable exactly
    what api/ai.py refuses, and a refused settings change says 'Provider settings:'."""
    if not NODE:
        pytest.skip("node is not installed")
    from fovea.api import ai as ai_api
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    entries = client.get("/api/ai/providers").json()["providers"]
    assert {"anthropic", "openai_compat", "yolo", "mock"} <= {p["id"] for p in entries}
    (tmp_path / "settings.mjs").write_text((STATIC / "views" / "settings.js").read_text(encoding="utf-8"), encoding="utf-8")
    (tmp_path / "entries.json").write_text(json.dumps(entries), encoding="utf-8")
    (tmp_path / "run.mjs").write_text(
        "import { remoteLocks } from './settings.mjs';\nimport { readFileSync } from 'node:fs';\n"
        "const E = JSON.parse(readFileSync(new URL('./entries.json', import.meta.url)));\n"
        "console.log(JSON.stringify(Object.fromEntries(E.map(p => [p.id, remoteLocks(p, { request_is_local: false }, false)]))));\n",
        encoding="utf-8")
    r = subprocess.run([NODE, str(tmp_path / "run.mjs")], capture_output=True, text=True, timeout=60, cwd=tmp_path)
    assert r.returncode == 0, r.stderr
    locks = json.loads(r.stdout.strip().splitlines()[-1])

    monkeypatch.setattr(ai_api, "_is_local", lambda req: False)
    r = client.put("/api/ai/providers/anthropic/config", json={"effort": "medium"})
    assert r.status_code == 403 and r.json()["detail"].startswith("Provider settings: "), r.text
    r = client.delete("/api/ai/providers/anthropic/key")
    assert r.status_code == 403 and r.json()["detail"].startswith("API keys: "), r.text
    for p in entries:
        pid, lk = p["id"], locks[p["id"]]
        refused = client.put(f"/api/ai/providers/{pid}/config", json={}).status_code == 403
        assert lk["settings"] == refused, (pid, lk, refused)
        if p["key"] is not None:
            assert lk["key"] and client.put(f"/api/ai/providers/{pid}/key", json={"key": "x" * 24}).status_code == 403
        assert lk["call"] == (p["kind"] == "cloud"), pid
        if p["kind"] == "cloud":
            assert client.post(f"/api/ai/providers/{pid}/test").status_code == 403, pid
            assert client.get(f"/api/ai/providers/{pid}/models", params={"refresh": 1}).status_code == 403, pid


# ------------------------------------------------------------------ final round

def test_review_links_open_the_most_suspicious_first(node_r3):
    """Overview's Data quality and AI audit Review links say their order: QA flags by the flag score, audit findings
    by the audit score, both descending — Explore and Annotate start every other sort ascending and carry an explicit
    order on to the next view."""
    assert node_r3["review"] == {"qa": "issue=blurry&sort=flag&order=desc", "kind": "issue=ai_missing&sort=audit&order=desc",
                                 "all": "ai=audit&sort=audit&order=desc"}
    ov = (STATIC / "views" / "overview.js").read_text(encoding="utf-8")
    assert "link(explore(flagReview(c))" in ov and "link(explore({ issue: c })" not in ov, "every Data quality row"
    assert "explore(auditReview({ issue: `ai_${k}` }))" in ov and "explore(auditReview({ ai: 'audit' }))" in ov
    from fovea.core.imagesq import DEFAULT_DESC, SORT_KEYS
    assert {"flag", "audit"} <= set(SORT_KEYS) and {"flag", "audit"} <= DEFAULT_DESC


def test_review_link_query_ranks_by_score(client, ds):
    """The query a Review link sends lists the highest-scored image first, whatever the image ids are."""
    ids = [it["id"] for it in client.get(f"/api/datasets/{ds}/images", params={"limit": 6}).json()["items"]]
    scores = {ids[0]: 1.5, ids[1]: 9.0, ids[2]: 4.2, ids[4]: 0.3}
    try:
        for iid, sc in scores.items():
            db.execute("INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
                       (iid, ds, "blurry", sc, "{}"))
        db.execute("INSERT OR REPLACE INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
                   (ids[0], ds, "dark", 99.0, "{}"))      # another flag's score never ranks the blurry list
        got = client.get(f"/api/datasets/{ds}/images", params={"issue": "blurry", "sort": "flag", "order": "desc"}).json()
        assert [it["id"] for it in got["items"]] == [ids[1], ids[2], ids[0], ids[4]]
        now = db.now()
        run_id = db.execute("INSERT INTO ai_runs(dataset_id, kind, provider, model, status, created_at, finished_at) "
                            "VALUES(?,?,?,?,?,?,?)", (ds, "audit", "mock", "m", "done", now, now)).lastrowid
        for iid, sc in ((ids[3], 0.4), (ids[5], 0.9), (ids[1], 0.6)):
            db.execute("INSERT INTO ai_preds(run_id, dataset_id, image_id, cls, xc, yc, w, h, kind, state, created_at) "
                       "VALUES(?,?,?,?,?,?,?,?,?,?,?)", (run_id, ds, iid, 0, .5, .5, .3, .3, "spurious", "pending", now))
            db.execute("INSERT OR REPLACE INTO ai_audit(image_id, dataset_id, run_id, score, counts, created_at) VALUES(?,?,?,?,?,?)",
                       (iid, ds, run_id, sc, '{"spurious": 1}', now))
        for q in ({"issue": "ai_spurious"}, {"ai": "audit"}):
            got = client.get(f"/api/datasets/{ds}/images", params={**q, "sort": "audit", "order": "desc"}).json()
            assert [it["id"] for it in got["items"]] == [ids[5], ids[1], ids[3]], q
    finally:
        db.execute("DELETE FROM image_flags WHERE dataset_id=?", (ds,))
        db.execute("DELETE FROM ai_audit WHERE dataset_id=?", (ds,))
        db.execute("DELETE FROM ai_preds WHERE dataset_id=?", (ds,))
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))


def test_spend_cap_line_says_what_runs_still_going_hold(node_r3):
    c = node_r3["cap"]
    assert c["held"] == "spent $0.23 in the last 30 days · reserved by runs still going $0.92"
    assert c["idle"] == "spent $0.20 in the last 30 days", "nothing reserved: nothing said"
    assert c["legacy"] == "spent $1.00 in the last 30 days"
    assert c["noCap"] == "spent $0.42 in the last 30 days · 2 runs with no price on file not counted", "no cap: by_provider's figure"
    assert c["heldUnpriced"] == "spent $1.00 in the last 30 days · reserved by runs still going $0.50 · 1 run with no price on file not counted"
    assert c["room"] is None and c["noCapStatus"] is None and c["legacyStatus"] is None
    assert c["reached"] == "Spend cap reached ($5.20 of $5.00)"
    assert c["takenByRuns"] == "Spend cap held by runs still going ($0.92 of $1.00)", "the server refuses new runs here too"
    assert c["exactlyHeld"] == "Spend cap held by runs still going ($0.50 of $1.00)"
    st = (STATIC / "views" / "settings.js").read_text(encoding="utf-8")
    assert "capUsageText(capU, spent, unpricedRuns, fmt)" in st and "capStatus(usage.caps[p.id], fmt)" in st


def test_settings_reads_the_reserved_spend_the_usage_endpoint_reports(client, ds, monkeypatch, tmp_path):
    """GET /api/ai/usage fills caps[id].reserved_usd while an Economy run waits at the provider; Settings' cap line
    turns exactly that entry into words."""
    if not NODE:
        pytest.skip("node is not installed")
    from fovea.ai import config as ai_config, store
    monkeypatch.setattr(ai_config, "spend_cap", lambda pid: 5.0 if pid == "mock" else None)
    base = client.get("/api/ai/usage").json()["caps"]["mock"]["reserved_usd"]
    try:
        store.create_run(ds, "batch", "mock", "m", {"speed": "economy", "per_image_usd": 0.05, "images": 10}, status="running")
        cap = client.get("/api/ai/usage").json()["caps"]["mock"]
        assert cap["reserved_usd"] == pytest.approx(base + 0.5) and cap["cap_usd"] == 5.0
        tmp = tmp_path / "capline"
        shutil.copytree(STATIC / "lib", tmp / "lib")
        (tmp / "views").mkdir()
        (tmp / "views" / "settings.js").write_text((STATIC / "views" / "settings.js").read_text(encoding="utf-8"), encoding="utf-8")
        (tmp / "package.json").write_text('{"type": "module"}')
        (tmp / "run.js").write_text(
            "globalThis.navigator = { platform: 'MacIntel' };\n"
            "globalThis.document = { addEventListener() {}, removeEventListener() {} };\n"
            "globalThis.window = { addEventListener() {}, removeEventListener() {} };\n"
            "const { fmt } = await import('./lib/dom.js');\nconst st = await import('./views/settings.js');\n"
            f"const cap = {json.dumps(cap)};\n"
            "console.log(JSON.stringify([st.capUsageText(cap, 0, 0, fmt), st.capStatus(cap, fmt)]));\n", encoding="utf-8")
        r = subprocess.run([NODE, str(tmp / "run.js")], capture_output=True, text=True, timeout=60, cwd=tmp)
        assert r.returncode == 0, r.stderr
        line, status = json.loads(r.stdout.strip().splitlines()[-1])
        assert line.endswith(f" · reserved by runs still going ${cap['reserved_usd']:.2f}"), line
        assert status is None, "$5 cap, well under it"
    finally:
        db.execute("DELETE FROM ai_runs WHERE dataset_id=?", (ds,))
