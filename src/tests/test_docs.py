"""Docs stay true to the code: claims in README.md and docs/*.md that can drift, checked against the code that
backs them (install profile, Overview tiles, cost estimates and caps, effort gains, Health check and QA rules).
Run: python -m pytest src/tests -q"""
from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import uuid
from pathlib import Path

import pytest
from PIL import Image

SRC = Path(__file__).resolve().parents[1]
REPO = SRC.parent
sys.path.insert(0, str(SRC))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fovea import db  # noqa: E402
from fovea.core import layout, qa, scanner, stats  # noqa: E402
from fovea.jobs import Job  # noqa: E402


def _doc(name: str) -> str:
    return (REPO / name).read_text(encoding="utf-8")


def _line(text: str, needle: str) -> str:
    hits = [ln for ln in text.splitlines() if needle in ln]
    assert hits, f"no line with {needle!r}"
    return hits[0]


def _write(root: Path, split: str, name: str, im: Image.Image, rows) -> None:
    (root / "images" / split).mkdir(parents=True, exist_ok=True)
    (root / "labels" / split).mkdir(parents=True, exist_ok=True)
    im.save(root / "images" / split / f"{name}.jpg", quality=92)
    (root / "labels" / split / f"{name}.txt").write_text(
        "".join(f"{c} {x:.6f} {y:.6f} {w:.6f} {h:.6f}\n" for c, x, y, w, h in rows))


def _register(root: Path, names) -> str:
    db.init_db()
    lay = layout.detect(str(root))
    ds_id = "docs-" + uuid.uuid4().hex[:8]
    now = db.now()
    db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
               " VALUES(?,?,?,?,?,?,?,?,?,?)",
               (ds_id, ds_id, lay.root, db.dumps(lay.to_dict()), db.dumps(list(names)), "user", "", "new", now, now))
    return ds_id


def _scan(ds_id: str) -> None:
    """Index without the post-scan hooks (a queued QA job would outlive the test's dataset)."""
    hooks = list(scanner.POST_SCAN_HOOKS)
    scanner.POST_SCAN_HOOKS[:] = []
    try:
        scanner.scan_dataset(ds_id, Job(id="scan-" + ds_id, kind="scan"))
    finally:
        scanner.POST_SCAN_HOOKS[:] = hooks


def _drop(ds_id: str) -> None:
    for t in ("ai_preds", "ai_runs", "image_flags", "image_hash", "boxes", "images", "plugin_data", "datasets"):
        db.execute(f"DELETE FROM {t} WHERE {'id' if t == 'datasets' else 'dataset_id'}=?", (ds_id,))


# ============================================================== README quick start: the core-only install

def _core_requirements():
    out = []
    for ln in (SRC / "requirements.txt").read_text().splitlines():
        if ln.startswith("# Optional"):
            break
        ln = ln.split("#")[0].strip()
        if ln:
            out.append(ln)
    return out


def test_core_install_command_is_the_requirements_core_block_and_names_every_ai_dependency():
    line = _line(_doc("README.md"), "가볍게 쓰려면 핵심 패키지만")
    cmd = re.search(r"`pip install ([^`]+)`", line).group(1)
    assert re.findall(r'"([^"]+)"', cmd) == _core_requirements()
    # Every AI provider needs an optional package: the sentence must say which, not only "local models and Claude".
    for dep in ("ultralytics", "torch", "anthropic", "httpx"):
        assert f"`{dep}`" in line, dep
    for who in ("YOLO/YOLOE/SAM", "Claude", "Gemini", "OpenAI 호환"):
        assert who in line, who
    assert "AI 공급자는 모두 쓸 수 없습니다" in line
    for still in ("라벨링", "검수", "데이터 품질", "내보내기"):
        assert still in line


CORE_ONLY = r"""
import io, json, sys, zipfile
from pathlib import Path
for m in ("httpx", "anthropic", "numpy", "ultralytics", "torch", "tifffile", "pillow_heif", "clip"):
    sys.modules[m] = None          # what a core-only install leaves out: invisible to import AND find_spec
sys.path.insert(0, sys.argv[1])
root = Path(sys.argv[2])
from PIL import Image, ImageDraw
from fovea import db
from fovea.ai import registry
from fovea.core import layout, qa, scanner, stats, export
from fovea.jobs import Job
db.init_db()
out = {"status": {pid: registry.safe_status(registry.get(pid)) for pid in registry.ids()}}
im = Image.new("RGB", (160, 120), (90, 90, 90))
ImageDraw.Draw(im).rectangle([20, 20, 80, 70], fill=(200, 60, 60))
for k in (1, 2, 3):
    for sub in ("images/train", "labels/train"):
        (root / sub).mkdir(parents=True, exist_ok=True)
    im.save(root / "images/train" / f"cam_{k:04d}.jpg")
    (root / "labels/train" / f"cam_{k:04d}.txt").write_text("0 0.31 0.375 0.375 0.417\n")
lay = layout.detect(str(root))
now = db.now()
db.execute("INSERT INTO datasets(id,name,root,layout,classes,classes_source,description,status,created_at,updated_at)"
           " VALUES(?,?,?,?,?,?,?,?,?,?)", ("core", "core", lay.root, db.dumps(lay.to_dict()), db.dumps(["thing"]),
                                            "user", "", "new", now, now))
scanner.scan_dataset("core", Job(id="scan-core", kind="scan"))
res = qa.run_qa("core", Job(id="qa-core", kind="qa"))
out["qa"] = [res.get("images"), bool(res.get("deferred")), bool((qa.load_summary("core") or {}).get("computed_at"))]
out["boxes"] = stats.dataset_stats("core", ["thing"])["classes"][0]["boxes"]
from fovea.api import datasets as api_ds
iid = db.query_one("SELECT id FROM images WHERE dataset_id='core' ORDER BY rel_path")["id"]
out["saved"] = api_ds.put_labels("core", iid, {"boxes": [[0, 0.5, 0.5, 0.2, 0.2]]}).get("saved")
out["review"] = api_ds.set_review("core", {"status": "approved", "image_ids": [iid]}).get("updated")
z = zipfile.ZipFile(io.BytesIO(b"".join(export.stream_zip("core", ["thing"], {}))))
out["zip"] = sorted(n for n in z.namelist() if n.endswith((".jpg", ".txt", ".yaml")))
print("RESULT " + json.dumps(out))
"""


def test_core_only_install_disables_every_ai_provider_and_keeps_the_rest(tmp_path):
    """README: with only the core packages every AI provider is unavailable with the missing package named (Gemini /
    OpenAI included, not just local models and Claude), while labeling, review, data quality and export work."""
    script = tmp_path / "core_only.py"
    script.write_text(CORE_ONLY)
    env = {k: v for k, v in os.environ.items() if not k.endswith(("_API_KEY", "_BASE_URL")) and k != "FOVEA_AI_MOCK"}
    env["FOVEA_DATA_DIR"] = str(tmp_path / "data")
    for name in ("ANTHROPIC", "GEMINI", "OPENAI", "OPENAI_COMPAT"):       # nothing can reach a real API
        env[f"FOVEA_{name}_BASE_URL"] = "http://127.0.0.1:9/v1"
    r = subprocess.run([sys.executable, str(script), str(SRC), str(tmp_path / "ds")], cwd=tmp_path, env=env,
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-3000:]
    out = json.loads(r.stdout.split("RESULT ", 1)[1])
    need = {"yolo": "ultralytics", "yoloe": "ultralytics", "sam": "ultralytics", "anthropic": "anthropic",
            "gemini": "httpx", "openai": "httpx", "openai_compat": "httpx"}
    assert set(need) <= set(out["status"])
    for pid, pkg in need.items():
        st = out["status"][pid]
        assert st["available"] is False, (pid, st)
        assert "not installed" in st["reason"] and f"pip install {pkg}" in st["reason"], (pid, st)
    assert out["qa"] == [3, False, True] and out["boxes"] == 3
    assert out["saved"] == 1 and out["review"] == 1
    assert "data.yaml" in out["zip"] and sum(n.endswith(".jpg") for n in out["zip"]) == 3


# ============================================================== Overview KPI tiles

def _tiles():
    src = (SRC / "static/views/overview.js").read_text(encoding="utf-8")
    out = {}
    for ln in src.splitlines():
        m = re.search(r"ui\.statTile\(\{ label: '([^']+)'", ln)
        if not m:
            continue
        go = re.search(r"onClick: \(\) => go\((explore\(|`/d/\$\{enc\}/(\w+)`)", ln)
        out[m.group(1)] = None if not go else ("explore" if go.group(1) == "explore(" else go.group(2))
    return out


def test_overview_tiles_go_where_the_readme_says():
    tiles = _tiles()
    line = _line(_doc("README.md"), "**KPI 타일**")
    listed = re.search(r"\*\*KPI 타일\*\*: ([^.]+)\.", line).group(1).split(" · ")
    assert listed == list(tiles)
    assert tiles["Boxes"] == "objects" and "Boxes는 Objects로" in line
    assert tiles["Classes"] is None and "Classes 타일은 정보만" in line
    assert {k for k, v in tiles.items() if v == "explore"} == set(tiles) - {"Boxes", "Classes"}
    assert "Explore로" in line


# ============================================================== costs: which calls are estimated, which are capped

def test_docs_say_which_cloud_calls_show_an_estimate():
    readme, ai_doc = _doc("README.md"), _doc("docs/ai-labeling.md")
    assert "모든 클라우드 실행 전에" not in readme
    assert "클라우드 실행은 예상 금액, 실행별 예산" not in ai_doc
    cost = _line(readme, "- **비용**:")
    for call in ("Annotate `I`", "Ask AI to check the labels", "Explore **Ask AI**", "Preview on 3 images",
                 "Test on 4 labeled images"):
        assert call in cost, call
    assert "실행별 예산은 일괄 라벨링에만" in cost
    # the code agrees: the single-image calls never ask for an estimate ...
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    body = panel[panel.index("async function propose()"):]
    body = body[:re.search(r"\n  (async )?function ", body[1:]).start() + 1]
    assert "/api/ai/propose" in body and "/api/ai/estimate" not in body
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    assert "/api/ai/review" in explore and "/api/ai/nl-filter" in explore and "/api/ai/estimate" not in explore
    # ... and the batch form, two-stage, audit and benchmark do
    ai_tab = (SRC / "static/views/ai.js").read_text(encoding="utf-8")
    assert "/api/ai/estimate" in panel and ai_tab.count("/api/ai/estimate") >= 3
    assert "dry_run: true" in ai_tab                      # Suggest names: exact cost before anything is sent
    # ... and only a batch run takes a per-run budget
    api = (SRC / "fovea/api/ai.py").read_text(encoding="utf-8")
    routes = re.split(r"\n@router\.", api)
    assert [r.split("\n", 1)[0] for r in routes if "budget_usd" in r] == ['post("/api/ai/batch")']


@pytest.fixture()
def tiny_ds(tmp_path):
    root = tmp_path / "tiny"
    _write(root, "train", "a_0001", Image.new("RGB", (64, 48), (120, 120, 120)), [(0, 0.5, 0.5, 0.3, 0.3)])
    ds_id = _register(root, ["thing"])
    _scan(ds_id)
    yield ds_id
    _drop(ds_id)


def test_a_reached_spend_cap_blocks_single_image_cloud_calls(tiny_ds, monkeypatch):
    """README / ai-labeling.md: single-image calls have no estimate, but the 30-day spend cap still refuses them."""
    from testkit import client_from
    from fovea.ai import config as ai_config
    from fovea.main import app
    for name in ("ANTHROPIC", "GEMINI", "OPENAI"):
        monkeypatch.setenv(f"FOVEA_{name}_BASE_URL", "http://127.0.0.1:9")
    monkeypatch.setattr(ai_config, "spend_cap", lambda pid: 0.0 if pid == "anthropic" else None)
    c = client_from(app, ("127.0.0.1", 50000), base_url="http://127.0.0.1")
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=?", (tiny_ds,))["id"]
    calls = {"/api/ai/propose": {"dataset_id": tiny_ds, "image_id": iid, "provider": "anthropic"},
             "/api/ai/review": {"dataset_id": tiny_ds, "image_id": iid, "provider": "anthropic"},
             "/api/ai/nl-filter": {"dataset_id": tiny_ds, "text": "images with a thing", "provider": "anthropic"}}
    for url, body in calls.items():
        r = c.post(url, json=body)
        assert r.status_code == 402, (url, r.status_code, r.text[:300])
        assert "spend cap" in r.text, (url, r.text[:300])


# ============================================================== ai-labeling.md: effort gains and the model table

def _bench():
    prices = json.loads((SRC / "fovea/ai/prices.json").read_text(encoding="utf-8"))
    out = {}
    for pid, p in prices["providers"].items():
        for m in p.get("models") or []:
            out[m["name"]] = m.get("bench") or {}
    return out


def test_effort_gain_range_matches_the_price_table():
    gains = {n: round(b["map50_high"] - b["map50_low"], 1) for n, b in _bench().items()
             if b.get("map50_low") is not None and b.get("map50_high") is not None}
    line = _line(_doc("docs/ai-labeling.md"), "- **노력 수준**")
    lo, hi = re.search(r"\+([\d.]+)~\+([\d.]+)", line).groups()
    assert (float(lo), float(hi)) == (min(gains.values()), max(gains.values()))
    best = max(gains, key=gains.get)
    assert f"{best} +{gains[best]}" in line                    # the largest gain is named first
    for name, g in gains.items():                               # every model the sentence names, with its gain
        if name.split(" (")[0] in line:
            assert f"{name.split(' (')[0]} +{g}" in line, name


def test_model_table_scores_match_the_price_table():
    bench = {n.split(" (")[0]: b for n, b in _bench().items()}
    doc = _doc("docs/ai-labeling.md")
    table = doc[doc.index("| 모델 | mAP@50 |"):]
    table = table[:table.index("\n\n")]
    rows = [r.split("|")[1:3] for r in table.splitlines()[2:]]
    assert rows
    for name_cell, score_cell in rows:
        names = [n.split(" (")[0].strip() for n in name_cell.split(" · ")]
        scores = [s.strip() for s in score_cell.split(" · ")]
        assert len(names) == len(scores), name_cell
        for n, s in zip(names, scores):
            b = bench[n]
            want = f"{b['map50_low']} / {b['map50_high']}" if b.get("map50_high") is not None else f"{b['map50_low']}"
            assert s == want, (n, s, want)


# ============================================================== Health check: under-represented classes

def test_under_represented_is_ten_percent_of_the_median_class(tmp_path):
    """Counts 100, 60, 50, 8, 3, 0: the median of the non-zero per-class counts is 50, so 8 boxes is fine and 3 is
    flagged. Under '10% of the most common class' 8 would be flagged too."""
    root = tmp_path / "cc"
    counts = [100, 60, 50, 8, 3, 0]
    for cls, n in enumerate(counts):
        if n:
            rows = [(cls, 0.05 + (k % 10) * 0.1, 0.05 + (k // 10) * 0.1, 0.04, 0.04) for k in range(n)]
            _write(root, "train", f"c{cls}", Image.new("RGB", (64, 64), (100 + cls * 20,) * 3), rows)
    names = ["a", "b", "c", "d", "e", "f"]
    ds_id = _register(root, names)
    try:
        _scan(ds_id)
        st = stats.dataset_stats(ds_id, names)
        under = {c["cls"]: c["under_represented"] for c in st["classes"]}
        assert [c["boxes"] for c in sorted(st["classes"], key=lambda c: c["cls"])] == counts
        assert under == {0: False, 1: False, 2: False, 3: False, 4: True, 5: True}
    finally:
        _drop(ds_id)
    for doc in (_doc("README.md"), _doc("docs/data-quality.md")):
        assert "클래스별 박스 수 중앙값" in doc and "가장 흔한 클래스 중앙값" not in doc


# ============================================================== data-quality.md: seq_jump movement rule

def _item(i: int, name: str, boxes) -> qa.Item:
    it = qa.Item({"id": i, "rel_path": f"train/{name}.jpg", "split": "train", "seq": name.split("_")[0],
                  "width": 640, "height": 640, "size": 1, "has_label": 1, "n_boxes": len(boxes), "dhash": None,
                  "b0": None, "luma": None, "contrast": None, "sharpness": None})
    it.boxes = [list(b) for b in boxes]
    return it


def test_seq_jump_needs_the_larger_of_half_the_diagonal_and_8_px():
    """data-quality.md: the centre must move at least max(half the expected box's diagonal, 8 px). A 9 px box moved
    7 px (> half diagonal 6.4, < 8) and a 40 px box moved 22 px (> 8, < half diagonal 28.3) are not jumps."""
    cases = {"jA": (9, 7, False), "jB": (9, 9, True), "jC": (40, 22, False), "jD": (40, 30, True)}
    items, mid = [], {}
    for k, (name, (size, shift, _)) in enumerate(cases.items()):
        for f in (1, 2, 3):
            x = (200 + (shift if f == 2 else 0)) / 640
            iid = k * 10 + f
            items.append(_item(iid, f"{name}_{f:04d}", [(0, x, 300 / 640, size / 640, size / 640)]))
            if f == 2:
                mid[name] = iid
    flags: dict = {}
    qa.check_sequences(items, dict(qa.DEFAULTS), flags, {})
    jumped = {name for name, iid in mid.items() if (iid, "seq_jump") in flags}
    assert jumped == {n for n, c in cases.items() if c[2]}
    line = _line(_doc("docs/data-quality.md"), "**`seq_jump`**")
    assert "대각선의 절반과 8 px 중 큰 값" in line and "대각선의 절반 또는 8 px" not in line


# ============================================================== keyboard.md: the tables ARE the registered keymaps

KEY_DUMP = r"""
// Install every view that registers keys with a stub fovea, then print keys.keyGroups() as JSON.
const [root, ...mods] = process.argv.slice(2);
globalThis.window = globalThis;
globalThis.document = { querySelector: () => null, createElement: () => ({ dataset: {} }), head: { appendChild() {} },
                        addEventListener() {} };
Object.defineProperty(globalThis, 'navigator', { value: { platform: 'MacIntel', userAgent: 'Mac' }, configurable: true });
globalThis.localStorage = { getItem: () => null, setItem() {}, removeItem() {} };
const keys = await import(root + '/static/lib/keys.js');
const fovea = { keys, MOD: '⌘', registerTab() {}, registerCommand() {} };
for (const m of mods) {
  const mod = await import(root + '/' + m);
  (mod.install || mod.default)(fovea, { api: '/api/plugins/x', entry: '/plugins/x/x.js' });
}
console.log('RESULT ' + JSON.stringify(keys.keyGroups()));
"""


def _key_sites():
    """Every frontend file that calls registerKeys (keys.js itself defines it), relative to src/."""
    out = []
    for p in sorted([*(SRC / "static").rglob("*.js"), *(SRC / "fovea/plugins").glob("*/static/*.js")]):
        rel = p.relative_to(SRC).as_posix()
        if rel != "static/lib/keys.js" and "registerKeys(" in p.read_text(encoding="utf-8"):
            out.append(rel)
    return out


def _norm_keys(s: str) -> str:
    """'`⌘A`', '`Alt` + `←` `↑`', '`` ` ``', '`0`–`9`' (doc) and '⌘ A', 'Alt ←↑', '`', '0-9' (code) → one spelling."""
    s = re.sub(r"(`+)\s?(.+?)\s?\1", lambda m: m.group(2), s)
    return re.sub(r"[\s+]", "", s).replace("–", "-").lower()


def _registered_groups(tmp_path):
    import shutil
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    mirror = tmp_path / "mirror"                       # a package.json with type=module, so Node loads the ES modules
    shutil.copytree(SRC / "static", mirror / "static")
    for d in (SRC / "fovea/plugins").glob("*/static"):
        shutil.copytree(d, mirror / d.relative_to(SRC))
    (mirror / "package.json").write_text('{"type": "module"}')
    (tmp_path / "dump.mjs").write_text(KEY_DUMP, encoding="utf-8")
    views = [s for s in _key_sites() if s != "static/app.js"]
    r = subprocess.run([node, str(tmp_path / "dump.mjs"), str(mirror), *views], capture_output=True, text=True,
                       timeout=60, check=False)
    assert r.returncode == 0, r.stderr[-2000:]
    groups = json.loads(r.stdout.split("RESULT ", 1)[1])
    # The global group is registered by app.js, which needs a real page: read its table from the source.
    app = (SRC / "static/app.js").read_text(encoding="utf-8")
    block = app[app.index("const GLOBAL_KEYS = ["):]
    block = block[:block.index("];")]
    items = [[k.strip("`'").replace("${MOD}", "⌘"), d]
             for k, d in re.findall(r"\[\s*(`[^`]*`|'[^']*')\s*,\s*'([^']*)'\s*\]", block)]
    assert re.search(r"registerKeys\('global', 'Everywhere', GLOBAL_KEYS", app)
    return [{"scope": "global", "title": "Everywhere", "items": items}] + groups


def _doc_key_tables():
    out, cur = [], None
    for ln in _doc("docs/keyboard.md").splitlines():
        m = re.match(r"## (.+) \(`([\w-]+)`\)$", ln)
        if m:
            cur = {"scope": m.group(2), "title": m.group(1), "items": []}
            out.append(cur)
            continue
        if ln.startswith("## "):
            cur = None
        elif cur is not None and ln.startswith("| ") and not ln.startswith(("| 키 |", "| --- |")):
            cells = [c.strip() for c in ln.strip().strip("|").split(" | ")]
            cur["items"].append([cells[0], cells[1]])
    return out


def test_keyboard_doc_tables_are_the_registered_keymaps(tmp_path):
    """keyboard.md says its tables cannot drift from the `?` dialog: same groups, same order, same rows, same text."""
    want = [(g["scope"], g["title"], [(_norm_keys(k), d) for k, d in g["items"]]) for g in _registered_groups(tmp_path)]
    have = [(g["scope"], g["title"], [(_norm_keys(k), d) for k, d in g["items"]]) for g in _doc_key_tables()]
    assert [g[:2] for g in have] == [g[:2] for g in want]
    for (scope, _t, rows), (_s, _t2, ref) in zip(have, want):
        assert rows == ref, scope
    intro = _doc("docs/keyboard.md").split("\n\n")[1]
    for site in _key_sites():
        assert f"`{site}`" in intro, site
    # keys the ? dialog lists are not repeated as "not in the dialog"
    unlisted = _doc("docs/keyboard.md").split("## 목록에 없는 키보드 조작", 1)[1]
    assert "Files 목록" not in unlisted


def test_digit_wait_rule_in_keyboard_doc_matches_the_code():
    """A digit waits for a second one only when it can start a valid two-digit class (d × 10 < classes)."""
    objects = (SRC / "static/views/objects.js").read_text(encoding="utf-8")
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    assert "d > 0 && d * 10 < n" in objects and "setTimeout(() => commitDigit(), 550)" in objects
    assert "wait: one > 0 && one * 10 < n" in annotate and "setTimeout(clearClassBuf, 600)" in annotate
    doc = _doc("docs/keyboard.md")
    assert "11개 이상이면" not in doc and "클래스가 10개를 넘으면" not in doc
    assert doc.count("숫자 × 10 < 클래스 수") == 2
    assert "클래스가 13개 이상이면 `1` `2` → 클래스 12" in doc and "550 ms" in doc and "600 ms" in doc


# ============================================================== data-quality.md: flag and scan-report tables

def _md_table(text: str, header: str):
    t = text[text.index(header):]
    t = t[:t.index("\n\n")]
    return [[c.strip() for c in ln.strip().strip("|").split(" | ")] for ln in t.splitlines()[2:]]


def test_flag_table_lists_every_qa_flag_with_its_screen_name():
    doc = _doc("docs/data-quality.md")
    rows = _md_table(doc[doc.index("## 플래그"):], "| 코드 | 화면 이름 | 뜻 |")
    assert {r[0].strip("`"): r[1] for r in rows} == qa.LABELS


def test_scan_report_table_lists_every_scan_problem_with_its_severity():
    doc = _doc("docs/data-quality.md")
    rows = _md_table(doc[doc.index("## 인덱싱 보고서"):], "| 코드 | 심각도 | 뜻 |")
    assert {r[0].strip("`"): r[1] for r in rows} == {c: v[0] for c, v in scanner.SCAN_PROBLEMS.items()}


def _noise(seed: int) -> Image.Image:
    import random
    rnd = random.Random(seed)
    return Image.frombytes("L", (64, 48), bytes(rnd.randrange(256) for _ in range(64 * 48))).convert("RGB")


def test_leaks_are_judged_on_split_roles_as_documented(tmp_path):
    """data-quality.md: one frame in train and val is a leak (leak_split); one frame in test and test-1854-1860 (both
    feed test) is dup_same_role, not a leak, and that video is not listed as spanning splits."""
    root = tmp_path / "roles"
    a, b = _noise(1), _noise(2)
    _write(root, "train", "camA_0001", a, [(0, 0.5, 0.5, 0.2, 0.2)])
    _write(root, "val", "camA_0002", a, [(0, 0.5, 0.5, 0.2, 0.2)])
    _write(root, "test", "camB_0001", b, [(0, 0.5, 0.5, 0.2, 0.2)])
    _write(root, "test-1854-1860", "camB_0002", b, [(0, 0.5, 0.5, 0.2, 0.2)])
    for k in range(3, 9):                               # unrelated frames: nothing else may match
        _write(root, "train", f"other{k}_0001", _noise(k), [(0, 0.4, 0.4, 0.1, 0.1)])
    ds_id = _register(root, ["thing"])
    try:
        _scan(ds_id)
        qa.run_qa(ds_id, Job(id="qa-" + ds_id, kind="qa"))
        got = {(r["split"], r["code"]) for r in db.query(
            "SELECT i.split, f.code FROM image_flags f JOIN images i ON i.id = f.image_id WHERE f.dataset_id=? "
            "AND f.code IN ('leak_split', 'dup_same_role')", (ds_id,))}
        assert got == {("train", "leak_split"), ("val", "leak_split"),
                       ("test", "dup_same_role"), ("test-1854-1860", "dup_same_role")}
        span = (qa.load_summary(ds_id) or {}).get("split_spanning_sequences") or []
        assert {s["seq"] for s in span} <= {"camA"}
    finally:
        _drop(ds_id)
    doc = _doc("docs/data-quality.md")
    assert "**`dup_same_role`**" in doc and "`test-1854-1860-포함` → test" in doc
    assert "`leak_split` 0장, `dup_same_role` 1,664장" in doc and "`leak_split` 1,680장" not in doc
    assert "Same frame in two folders of one split" in _line(_doc("README.md"), "- **Data quality**")


def test_near_duplicate_keep_rule_is_exports_and_the_docs_say_so():
    """Approved first, then labeled, most boxes, first indexed (lowest id); no sharpness step, not 'oldest file'."""
    from fovea.core import export
    K = qa._KeepRow
    assert qa.keep_member([K(5, None, 1, 6), K(9, "approved", 1, 3)]) == 9
    assert qa.keep_member([K(5, None, 0, 0), K(9, None, 1, 1)]) == 9
    assert qa.keep_member([K(7, None, 1, 4), K(3, None, 1, 4)]) == 3
    assert "_keep_rank" in (SRC / "fovea/core/qa.py").read_text(encoding="utf-8") and export._keep_rank
    rule = "승인 > 라벨 있음 > 박스 많음 > 먼저 인덱싱된 것"
    for name in ("README.md", "docs/data-quality.md"):
        doc = _doc(name)
        assert rule in doc, name
        assert "오래된 순" not in doc and "선명함 >" not in doc, name
    near = _line(_doc("docs/data-quality.md"), "- **`near_dup`**")
    assert "/qa/duplicates`를 읽을 때 정한다" in near
    assert "내보내기가 남길 대표 이미지" not in _doc("docs/data-quality.md")


# ============================================================== README: when Fovea deletes a label file

def test_readme_names_both_label_file_deletions(tiny_ds):
    """README: Restore and Annotate's remove_if_empty (zero boxes again, or **Remove file**) delete label files; the
    latter takes no snapshot."""
    from testkit import client_from
    from fovea.main import app
    c = client_from(app, ("127.0.0.1", 50000), base_url="http://127.0.0.1")
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=?", (tiny_ds,))["id"]
    url = f"/api/datasets/{tiny_ds}/images/{iid}/labels"
    cur = c.get(url).json()
    lp = Path(cur["label_path"])
    r = c.put(url, json={"boxes": [], "base_version": cur["version"]})
    assert r.status_code == 200 and lp.is_file() and lp.stat().st_size == 0     # an empty file stays (background)
    r = c.put(url, json={"boxes": [], "base_version": r.json()["version"], "remove_if_empty": True})
    assert r.status_code == 200 and r.json()["item"]["has_label"] is False and not lp.exists()
    assert c.get(f"/api/datasets/{tiny_ds}/snapshots").json()["snapshots"] == []
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    assert "body.remove_if_empty = true" in annotate and "'Remove file'" in annotate
    line = _line(_doc("README.md"), "Fovea가 라벨 파일을 지우는 경우는")
    assert "유일한 경우" not in _doc("README.md")
    for part in ("**Restore**", "**Remove file**", "스냅샷을 남기지 않습니다", "**Mark as background**"):
        assert part in line, part


# ============================================================== README: reverse proxy and X-Forwarded-For

def test_reverse_proxy_needs_x_forwarded_for_as_the_readme_says(tiny_ds, monkeypatch):
    """Behind a same-host proxy that sends no X-Forwarded-For every client is 'this PC': no token is asked. With the
    header (nginx $proxy_add_x_forwarded_for appends the real address) the token and the remote-AI guard apply. uvicorn
    trusts the header from FORWARDED_ALLOW_IPS (default 127.0.0.1); '*' trusts the client-written first value."""
    import uvicorn
    from testkit import client_from
    from uvicorn.middleware.proxy_headers import ProxyHeadersMiddleware
    from fovea.main import app
    for name in ("ANTHROPIC", "GEMINI", "OPENAI", "OPENAI_COMPAT"):
        monkeypatch.setenv(f"FOVEA_{name}_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.setenv("FOVEA_TOKEN", "tok-docs")
    monkeypatch.delenv("FORWARDED_ALLOW_IPS", raising=False)
    trusted = uvicorn.Config(app, log_config=None).forwarded_allow_ips
    assert trusted.split(",")[0] == "127.0.0.1" and set(trusted.split(",")) <= {"127.0.0.1", "::1"}  # 0.53+ adds ::1

    def client(hosts):
        return client_from(ProxyHeadersMiddleware(app, trusted_hosts=hosts), ("127.0.0.1", 50000),
                           base_url="http://127.0.0.1")

    proxy = client(trusted)
    def code(c, xff=None, **headers):
        return c.get("/api/datasets", headers={**({"X-Forwarded-For": xff} if xff else {}), **headers}).status_code

    assert code(proxy) == 200                                        # no header: the proxy's 127.0.0.1 is "this PC"
    assert code(proxy, "203.0.113.9") == 401
    assert code(proxy, "127.0.0.1, 203.0.113.9") == 401              # a spoofed first value does not help
    assert code(proxy, "203.0.113.9", Authorization="Bearer tok-docs") == 200
    iid = db.query_one("SELECT id FROM images WHERE dataset_id=?", (tiny_ds,))["id"]
    r = proxy.post("/api/ai/propose", json={"dataset_id": tiny_ds, "image_id": iid, "provider": "anthropic"},
                   headers={"X-Forwarded-For": "203.0.113.9", "Authorization": "Bearer tok-docs"})
    assert r.status_code == 403 and "FOVEA_ALLOW_REMOTE_AI" in r.text
    assert code(client("*"), "127.0.0.1, 203.0.113.9") == 200        # why the README forbids '*'
    assert code(client("10.0.0.5"), "203.0.113.9") == 200            # a proxy not listed: its header is ignored
    readme = _doc("README.md")
    sec = readme[readme.index("## 보안"):readme.index("## 구조")]
    for part in ("proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;", "FORWARDED_ALLOW_IPS=<프록시 주소>",
                 "`*`는", "Caddy", "FOVEA_TOKEN`과 원격 AI 차단이 적용되지 않습니다",
                 "| `FORWARDED_ALLOW_IPS` | `127.0.0.1` (uvicorn 0.53+는 `127.0.0.1,::1`) |", "uvicorn 0.53부터는 `127.0.0.1,::1`"):
        assert part in sec, part


# ============================================================== README Home, ai-labeling.md Settings switch

def test_readme_home_describes_the_collection_picker_and_the_duplicate_dialog():
    home = (SRC / "static/views/home.js").read_text(encoding="utf-8")
    assert "This folder holds several datasets. Pick one:" in home
    pick = home[home.index("lay.candidates.map(c =>"):]
    pick = pick[:pick.index("))))")]
    assert "pathInput.value = c.path_host" in pick and "detect(true)" in pick and "/api/datasets" not in pick
    assert "`Open “${dup[1]}”`" in home and "'Register anyway'" in home and "Open it" not in home
    assert "'Register & index'" in home
    line = _line(_doc("README.md"), "여러 데이터셋이 든 폴더")
    assert "Open it" not in line and "*Open “이름”* / *Register anyway*" in line and "고른 뒤 **Register & index**" in line


def test_ai_labeling_names_the_openai_flex_tier_switch():
    settings = (SRC / "static/views/settings.js").read_text(encoding="utf-8")
    assert "FLEX_PROVIDERS = new Set(['openai'])" in settings and "row('Flex tier'" in settings
    assert "Batch mode" not in settings
    openai = (SRC / "fovea/ai/providers/openai.py").read_text(encoding="utf-8")
    gen = openai[openai.index("def call_generate"):openai.index("# ---- Batch API")]
    assert 'opts.setdefault("batch_mode", False)' in gen               # Ask AI / Suggest names: never Flex
    for name in ("README.md", "docs/ai-labeling.md"):
        assert "Batch mode" not in _doc(name), name
    line = _line(_doc("docs/ai-labeling.md"), "**Flex tier** 스위치")
    for part in ("**OpenAI에만**", "Ask AI(자연어 검색)와 Suggest names는 항상 표준 등급", "*Batch API*"):
        assert part in line, part


# ============================================================== r3: docs that had fallen behind the code
# Each test pins a statement the docs now make to the code (or the behaviour) behind it.

def _client():
    from testkit import client_from
    from fovea.main import app
    return client_from(app, ("127.0.0.1", 50000), base_url="http://127.0.0.1")


def _wait(c, job_id: str, timeout: float = 60) -> dict:
    import time
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        j = j.get("job") or j
        if j.get("status") in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def _label_ds(tmp_path, names=("thing", "other"), n=2):
    """`n` 64x48 frames in images/train, one class-0 box each; indexed without the post-scan QA hook."""
    root = tmp_path / "labels-ds"
    for k in range(n):
        _write(root, "train", f"f{k}_0001", _noise(40 + k), [(0, 0.5, 0.5, 0.3, 0.3)])
    ds_id = _register(root, names)
    _scan(ds_id)
    ids = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?",
                                                                (ds_id,))}
    return root, ds_id, ids


def _cleanup(ds_id: str) -> None:
    from fovea.core import snapshots as snap
    snap.settle()
    snap.drop_dataset(ds_id)
    _drop(ds_id)


def test_snapshot_retention_is_two_rings_as_the_readme_says(tmp_path):
    """README: automatic snapshots of more than 20 files keep the newest 20, those of 20 files or fewer the newest
    50, so a run of small edits never pushes out a bulk edit's undo point; manual versions stay."""
    from fovea.core import snapshots as snap
    assert (snap.KEEP_AUTO, snap.KEEP_SMALL, snap.SMALL_FILES) == (20, 50, 20)
    files = []
    for k in range(snap.SMALL_FILES + 1):
        f = tmp_path / f"l{k}.txt"
        f.write_text(f"0 0.5 0.5 0.{k % 9 + 1} 0.2\n")
        files.append(str(f))
    _root, ds_id, _ids = _label_ds(tmp_path, n=1)
    try:
        manual = snap.snapshot_before(ds_id, files[:1], "manual", kind="manual")
        big = [snap.snapshot_before(ds_id, files, "bulk set_class") for _ in range(snap.KEEP_AUTO + 2)]
        small = [snap.snapshot_before(ds_id, files[:1], "objects") for _ in range(snap.KEEP_SMALL + 5)]
        assert snap.settle()
        snap.prune(ds_id)
        left = {r["id"] for r in db.query("SELECT id FROM snapshots WHERE dataset_id=? AND kind!='deleting'", (ds_id,))}
        assert left == {manual} | set(big[-snap.KEEP_AUTO:]) | set(small[-snap.KEEP_SMALL:])
    finally:
        _cleanup(ds_id)
    line = _line(_doc("README.md"), "**스냅샷 보관**")
    for part in ("파일이 20개를 넘는 것은 최근 20개", "20개 이하인 작은 것", "최근 50개", "**Save version**", "지울 때까지"):
        assert part in line, part
    assert "최근 20개 + 직접 저장한 버전" not in _doc("README.md")


def test_op_undo_keeps_later_edits_and_history_restore_is_full(tmp_path):
    """README: an op's own Undo (mode 'undo') reverts only files still as the op left them and reports the rest as
    'edited since — kept'; History's Restore (mode 'full', the default) puts the whole version back."""
    root, ds_id, ids = _label_ds(tmp_path)
    c = _client()
    try:
        a, b = ids["f0_0001"], ids["f1_0001"]
        la, lb = root / "labels/train/f0_0001.txt", root / "labels/train/f1_0001.txt"
        orig = la.read_text()
        r = c.post(f"/api/datasets/{ds_id}/labels/bulk", json={"op": "set_class", "cls": 1, "image_ids": [a, b]})
        assert r.status_code == 200, r.text
        sid = r.json()["snapshot_id"]
        assert la.read_text().startswith("1 ") and lb.read_text().startswith("1 ")
        cur = c.get(f"/api/datasets/{ds_id}/images/{a}/labels").json()     # a hand edit after the bulk op
        r = c.put(f"/api/datasets/{ds_id}/images/{a}/labels",
                  json={"boxes": [[1, 0.5, 0.5, 0.3, 0.3], [0, 0.2, 0.2, 0.1, 0.1]], "base_version": cur["version"]})
        assert r.status_code == 200, r.text
        hand = la.read_text()
        j = _wait(c, c.post(f"/api/datasets/{ds_id}/snapshots/{sid}/restore", json={"mode": "undo"}).json()["job"]["id"])
        assert j["status"] == "done", j
        assert (j["result"]["mode"], j["result"]["restored"], j["result"]["changed_since"]) == ("undo", 1, 1)
        assert "edited since — kept" in j["message"]
        assert la.read_text() == hand and lb.read_text() == orig                # the edit survives, the rest is undone
        j = _wait(c, c.post(f"/api/datasets/{ds_id}/snapshots/{sid}/restore", json={}).json()["job"]["id"])
        assert j["status"] == "done" and j["result"]["mode"] == "full" and j["result"]["restored"] == 1, j
        assert la.read_text() == orig                                           # History: the whole version
    finally:
        _cleanup(ds_id)
    safety = _doc("README.md").split("## 라벨 파일 안전", 1)[1].split("\n## ", 1)[0]
    line = _line(safety, "**작업의 Undo와 History의 Restore는 다릅니다.**")
    for part in ("**Undo these changes**", "**Range edit**", "Objects의 `⌘Z`", "*N edited since — kept*",
                 "History의 **Restore**는 그 버전 전체를 되돌립니다"):
        assert part in line, part
    for src, needle in (("static/views/ai-panel.js", "'Undo these changes'"), ("static/views/annotate.js", "{ mode: 'undo' }"),
                        ("static/views/objects.js", "mode: 'undo' }")):
        assert needle in (SRC / src).read_text(encoding="utf-8"), src
    overview = (SRC / "static/views/overview.js").read_text(encoding="utf-8")
    assert "mode: 'undo'" not in overview                                       # History restores in full
    assert "edited since — kept" in _line(_doc("docs/ai-labeling.md"), "**Undo these changes**")


def test_a_file_saved_while_a_restore_runs_is_kept_and_reported(tmp_path, monkeypatch):
    """README: a label file saved elsewhere while a restore runs is not overwritten: the newer edit is kept and the
    result says 'changed meanwhile and kept'."""
    from fovea.core import snapshots as snap
    root, ds_id, ids = _label_ds(tmp_path, n=1)
    la = root / "labels/train/f0_0001.txt"
    try:
        sid = snap.snapshot_images(ds_id, [ids["f0_0001"]], op="test")
        la.write_text("1 0.5 0.5 0.3 0.3\n")                                  # differs from the version: to restore
        real = snap.snapshot_before

        def racing(*a, **kw):                    # the restore's own undo point is taken; then someone saves
            out = real(*a, **kw)
            la.write_text("1 0.4 0.4 0.2 0.2\n")
            return out
        monkeypatch.setattr(snap, "snapshot_before", racing)
        job = Job(id="restore-" + ds_id, kind="restore")
        res = snap.restore(ds_id, sid, job=job)
        assert (res["restored"], res["kept"]) == (0, 1)
        assert la.read_text() == "1 0.4 0.4 0.2 0.2\n"
        assert "changed meanwhile and kept" in job.message
    finally:
        monkeypatch.undo()
        _cleanup(ds_id)
    assert "*N changed meanwhile and kept*" in _line(_doc("README.md"), "되돌리는 동안 다른 곳")


def test_symlinked_label_files_are_written_through_inside_and_refused_outside(tmp_path):
    """README: a label symlink into the dataset is written through (the link stays); one pointing outside is never
    written — the editor gets 403 label_link_outside, a bulk op skips and counts it."""
    root = tmp_path / "links"
    for k, name in enumerate(("in_0001", "out_0001")):
        _write(root, "train", name, _noise(70 + k), [(0, 0.5, 0.5, 0.3, 0.3)])
    shared = root / "shared" / "in.txt"
    shared.parent.mkdir()
    shared.write_text("0 0.5 0.5 0.3 0.3\n")
    outside = tmp_path / "elsewhere" / "out.txt"
    outside.parent.mkdir()
    outside.write_text("0 0.5 0.5 0.3 0.3\n")
    lin, lout = root / "labels/train/in_0001.txt", root / "labels/train/out_0001.txt"
    lin.unlink()
    lin.symlink_to(shared)
    lout.unlink()
    lout.symlink_to(outside)
    ds_id = _register(root, ["thing", "other"])
    c = _client()
    try:
        _scan(ds_id)
        ids = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?",
                                                                    (ds_id,))}
        url = f"/api/datasets/{ds_id}/images/{ids['in_0001']}/labels"
        r = c.put(url, json={"boxes": [[1, 0.5, 0.5, 0.3, 0.3]], "base_version": c.get(url).json()["version"]})
        assert r.status_code == 200, r.text
        assert lin.is_symlink() and shared.read_text().startswith("1 ")      # written through, link kept
        url = f"/api/datasets/{ds_id}/images/{ids['out_0001']}/labels"
        r = c.put(url, json={"boxes": [[1, 0.5, 0.5, 0.3, 0.3]], "base_version": c.get(url).json()["version"]})
        assert r.status_code == 403 and "label_link_outside" in r.text, r.text
        r = c.post(f"/api/datasets/{ds_id}/labels/bulk", json={"op": "set_class", "cls": 1, "image_ids": [ids["out_0001"]]})
        assert r.status_code == 200 and r.json().get("skipped_links") == 1, r.text
        assert lout.is_symlink() and outside.read_text().startswith("0 ")    # never written
    finally:
        _cleanup(ds_id)
    line = _line(_doc("README.md"), "**심볼릭 링크인 라벨 파일**")
    for part in ("그 파일에 씁니다", "403 `label_link_outside`", "건너뛰고", "충돌로 표시", "`link_outside`"):
        assert part in line, part


def test_relocate_keeps_image_ids_and_what_refers_to_them(tmp_path, monkeypatch):
    """README: Move to a new path… re-indexes the same dataset; image ids stay, so review marks, QA flags, snapshots
    and working sets still point at the right images."""
    from fovea.core import snapshots as snap
    monkeypatch.setattr(scanner, "POST_SCAN_HOOKS", [])         # no QA job outliving the test
    root, ds_id, ids = _label_ds(tmp_path, n=3)
    c = _client()
    try:
        a = ids["f0_0001"]
        assert c.put(f"/api/datasets/{ds_id}/review", json={"status": "approved", "image_ids": [a]}).status_code == 200
        db.execute("INSERT INTO image_flags(image_id, dataset_id, code, score, detail) VALUES(?,?,?,?,?)",
                   (a, ds_id, "blurry", 1.0, "{}"))
        token = c.post(f"/api/datasets/{ds_id}/sets", json={"ids": [ids["f2_0001"], a]}).json()["token"]
        sid = snap.snapshot_images(ds_id, [a], op="test")
        moved = tmp_path / "moved-here"
        root.rename(moved)
        r = c.post(f"/api/datasets/{ds_id}/relocate", json={"path": str(moved)})
        assert r.status_code == 200, r.text
        j = _wait(c, r.json()["job"]["id"])
        assert j["status"] == "done", j
        now = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?",
                                                                    (ds_id,))}
        assert now == ids
        row = db.query_one("SELECT label_path FROM images WHERE id=?", (a,))
        assert row["label_path"].startswith(str(moved.resolve())) or row["label_path"].startswith(str(moved))
        assert db.query_one("SELECT 1 FROM image_flags WHERE image_id=? AND code='blurry'", (a,))
        got = c.get(f"/api/datasets/{ds_id}/images/ids", params={"set": token}).json()
        assert (got.get("ids") or got.get("items")) == [ids["f2_0001"], a]
        assert db.query_one("SELECT image_id FROM snapshot_files WHERE snapshot_id=?", (sid,))["image_id"] == a
        item = c.get(f"/api/datasets/{ds_id}/images/{a}").json()
        item = item.get("item") or item
        assert (item.get("review") or {}).get("status") == "approved"
    finally:
        _cleanup(ds_id)
    for src in ("static/views/home.js", "static/views/dataset.js"):
        assert "'Move to a new path…'" in (SRC / src).read_text(encoding="utf-8"), src
    line = _line(_doc("README.md"), "**폴더를 옮겼을 때**")
    for part in ("**Move to a new path…**", "**이미지 ID가 그대로 유지**", "리뷰 표시", "AI 제안과 감사 소견",
                 "데이터 품질 플래그", "스냅샷", "작업 묶음", "Economy 배치"):
        assert part in line, part


def test_ai_panel_effort_is_three_levels_defaulting_to_settings():
    """ai-labeling.md: the panel offers Fast / Balanced / Accurate like Settings, and a source without a choice uses
    its provider's Settings default."""
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    block = panel[panel.index("export const EFFORTS = ["):]
    block = block[:block.index("];")]
    assert re.findall(r"label: '(\w+)', send: '(\w+)'", block) == [("Fast", "low"), ("Balanced", "medium"),
                                                                   ("Accurate", "high")]
    assert "p.config && p.config.effort" in panel and "defaultEffort(provOf(P" in panel
    settings = (SRC / "static/views/settings.js").read_text(encoding="utf-8")
    assert "row('Effort'" in settings and "label: 'Balanced'" in settings
    line = _line(_doc("docs/ai-labeling.md"), "- **노력 수준**")
    assert "Fast / Balanced / Accurate" in line and "Settings 기본값" in line and "low / medium / high" in line
    assert "Fast(낮음) / Accurate(높음)" not in _doc("docs/ai-labeling.md")


def test_gateway_url_credentials_are_documented_as_basic_auth_never_shown(monkeypatch):
    """README / ai-labeling.md: FOVEA_<PROVIDER>_BASE_URL may carry user:password (sent as basic auth, shown as the
    host only); Settings refuses such a URL."""
    from fovea.ai import config as ai_config
    from fovea.ai import vlm
    from fovea.ai.providers.openai_compat import OpenAICompatProvider
    envs = sorted(set(re.findall(r'base_url_env = "(FOVEA_\w+_BASE_URL)"', "".join(
        p.read_text(encoding="utf-8") for p in (SRC / "fovea/ai/providers").glob("*.py")))))
    assert envs == ["FOVEA_ANTHROPIC_BASE_URL", "FOVEA_GEMINI_BASE_URL", "FOVEA_OPENAI_BASE_URL",
                    "FOVEA_OPENAI_COMPAT_BASE_URL"]
    with pytest.raises(ai_config.ConfigError, match="credentials"):
        ai_config._clean_value("base_url", "http://gw:Secret9@127.0.0.1:9/v1")
    monkeypatch.setenv("FOVEA_OPENAI_COMPAT_BASE_URL", "http://gw:Secret9@127.0.0.1:9/v1")     # nothing listens there
    p = OpenAICompatProvider(config={"base_url": ""}, key="")
    assert p.base_url() == "http://gw:Secret9@127.0.0.1:9/v1"                  # used to build requests …
    assert "Secret9" not in p.location and "127.0.0.1:9" in p.location        # … shown as the host only
    assert "Secret9" not in vlm.scrub("Cannot reach http://gw:Secret9@127.0.0.1:9/v1/chat/completions")
    row = _line(_doc("README.md"), "| `FOVEA_ANTHROPIC_BASE_URL`")
    for part in [*(f"`{e}`" for e in envs), "user:password@", "basic auth", "호스트만", "거부"]:
        assert part in row, part
    line = _line(_doc("docs/ai-labeling.md"), "- **게이트웨이 주소의 자격 증명**")
    for part in ("`user:password@`", "basic auth", "`host:port`만", "*Put credentials in the key field, not in the URL*"):
        assert part in line, part


def test_refused_unreadable_and_cut_off_answers_are_billed_as_documented():
    """ai-labeling.md: a refusal, unreadable output or a cut-off answer carries the provider's charge; a cut-off answer
    is asked again with the larger budget (both billed) and only complete items are kept."""
    from fovea.ai import vlm
    from fovea.ai.base import ProviderError, Usage, billed_usage
    from fovea.ai.providers.openai_compat import OpenAICompatProvider
    p = OpenAICompatProvider(config={"base_url": "http://127.0.0.1:9/v1"}, key="")
    for res, code in ((vlm.CallResult("", Usage(100, 3, 0.01), finish="refusal"), "refusal"),
                      (vlm.CallResult("no json here", Usage(100, 9, 0.02)), "bad_output")):
        with pytest.raises(ProviderError) as ei:
            p._parse(res, "boxes", {})
        assert ei.value.code == code and billed_usage(ei.value).input_tokens == 100
    budgets = []

    def cut(max_tokens):
        budgets.append(max_tokens)
        return vlm.CallResult('{"objects": [{"label": "a"}, {"lab', Usage(50, max_tokens // 100, 0.001), finish="truncated")
    stats: dict = {}
    obj, res, notes = p._call(cut, "boxes", stats)
    assert budgets == [p.max_tokens, p.max_tokens_retry]
    assert obj["objects"] == [{"label": "a"}] and res.usage.input_tokens == 100 and notes
    api = (SRC / "fovea/api/ai.py").read_text(encoding="utf-8")
    assert "def _book_failed(" in api and api.count("_book_failed(e,") >= 2
    runner = (SRC / "fovea/ai/runner.py").read_text(encoding="utf-8")
    assert "spent = billed_usage(out)" in runner and "store.add_usage(run_id, spent, images=0)" in runner
    line = _line(_doc("docs/ai-labeling.md"), "- **실패해도 과금된 답**")
    for part in ("거절", "읽을 수 없는", "잘린", "30일 지출 상한", "처리한 이미지 수에는 넣지 않는다", "두 번 모두 과금"):
        assert part in line, part
    assert "출력 한도에서 잘린 답도" in _line(_doc("README.md"), "- **비용**:")


def test_flex_tier_estimates_use_batch_prices_and_economy_saves_nothing_more():
    from fovea.ai.providers.openai import OpenAIProvider
    std = OpenAIProvider(config={"base_url": "http://127.0.0.1:9/v1"}, key="sk-test-docs").estimate(640, 640, "", 2, {})
    flex = OpenAIProvider(config={"base_url": "http://127.0.0.1:9/v1", "batch_mode": True},
                          key="sk-test-docs").estimate(640, 640, "", 2, {})
    assert "tier" not in std and std["usd"][1] > std["usd_batch"][1]
    assert flex["tier"] == "flex" and flex["usd"] == flex["usd_batch"]
    est = (SRC / "fovea/ai/estimate.py").read_text(encoding="utf-8")
    assert "the Flex tier (Settings): batch prices now" in est and "Economy costs" in est
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    assert "speedOpts[1].label = flex ? 'Economy' : 'Economy · −50%'" in panel
    assert "the same price as the Flex tier" in panel
    line = _line(_doc("docs/ai-labeling.md"), "**Flex tier** 스위치")
    for part in ("추정의 Now 가격도 배치 가격", "*the same price as the Flex tier*", "더 싸지지 않고"):
        assert part in line, part


def test_economy_never_sent_wording_matches_the_ui():
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    assert "never sent`" in panel and "x.not_submitted" in panel
    econ = (SRC / "fovea/ai/economy.py").read_text(encoding="utf-8")
    assert 'self.stats["not_submitted"]' in econ and "never resubmit on resume" in econ
    doc = _doc("docs/ai-labeling.md")
    line = _line(doc, "제출 도중 서버가 멈추면")
    assert "*N never sent*" in line and "`not_submitted`" in line and "*not submitted*" not in doc


def test_large_qa_runs_compute_in_a_child_process(tmp_path, monkeypatch):
    """data-quality.md: from 2,000 images the checks are computed in a short-lived child process that only reads the
    index (the server writes); smaller runs compute in the server."""
    assert qa.PROCESS_MIN_IMAGES == 2000 and qa.RECHECK_PROCESS_FRAMES == 2000
    root, ds_id, _ids = _label_ds(tmp_path, n=3)
    try:
        assert qa.run_qa(ds_id, Job(id="qa-a-" + ds_id, kind="qa"))["process"] is False
        if qa._worker_broken:
            pytest.skip("this machine cannot start the QA worker: " + qa._worker_broken[0])
        monkeypatch.setattr(qa, "PROCESS_MIN_IMAGES", 3)
        res = qa.run_qa(ds_id, Job(id="qa-b-" + ds_id, kind="qa"))
        assert res["process"] is True and res["images"] == 3
        assert not qa._live_workers                                # the child is gone once the run ends
    finally:
        _cleanup(ds_id)
    line = _line(_doc("docs/data-quality.md"), "**어디서 계산하나**")
    for part in ("**2,000장 이상**", "자식 프로세스", "2,000프레임", "서버 안에서 계산한다"):
        assert part in line, part


def test_keyboard_doc_explains_shift_n_skips_and_reject_hidden():
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    assert "for (let tries = 0; tries < 25; tries++)" in annotate
    assert "whose proposals are all hidden here" in annotate and "'Reject hidden')" in annotate
    for name, needle in (("docs/keyboard.md", "`⇧N`은 대기 중인 제안이 있는 다음 이미지로"),
                         ("docs/ai-labeling.md", "- `⇧N`은 대기 중인 제안이 있는 다음 이미지로")):
        line = _line(_doc(name), needle)
        assert "25장" in line and "건너뛰" in line, name
    assert "**Reject hidden**" in _line(_doc("docs/keyboard.md"), "`⇧N`은 대기 중인 제안이")
    assert "**Reject hidden**" in _line(_doc("docs/ai-labeling.md"), "*already labeled*로 숨는다")


# ============================================================== r4: Docker on the LAN, Fill from neighbours, Files
# focus, the two ghosts

def _compose_env() -> dict:
    """docker-compose.yml `environment:` entries: NAME -> value, or None for a pass-through (`- NAME`)."""
    out, inside = {}, False
    for ln in (SRC / "docker-compose.yml").read_text(encoding="utf-8").splitlines():
        if ln.strip() == "environment:":
            inside = True
            continue
        if inside:
            s = ln.split("#", 1)[0].strip()
            if not ln.strip() or ln.strip().startswith("#"):
                continue
            if not s.startswith("- "):
                break
            name, eq, value = s[2:].strip().partition("=")
            out[name] = value if eq else None
    return out


def test_docker_opened_to_the_lan_treats_the_host_browser_as_remote_as_the_readme_says(tmp_path, monkeypatch):
    """README Docker: with the port opened and FOVEA_LOCAL_NETS removed, the Docker host's own browser arrives from
    the bridge like any other machine — token page, 403 for keys and cloud AI — until FOVEA_ALLOW_REMOTE_AI=1, which
    (like every variable the docs tell Docker users to put in .env) compose passes through to the container."""
    import ipaddress
    from testkit import client_from
    from fovea import security
    from fovea.ai import secrets as ai_secrets
    from fovea.main import app
    env = _compose_env()
    passthrough = {k for k, v in env.items() if v is None}
    assert {"FOVEA_TOKEN", "FOVEA_ALLOWED_HOSTS", "FOVEA_ALLOWED_ORIGINS", "FOVEA_ALLOW_REMOTE_AI",
            "ANTHROPIC_API_KEY", "GEMINI_API_KEY", "OPENAI_API_KEY"} <= passthrough
    readme = _doc("README.md")
    docker = readme[readme.index("### Docker"):readme.index("## 화면별 기능")]
    listed = _line(docker, "`.env`의 값은")
    listed = listed[listed.index("전달됩니다:"):listed.index("세 API 키")]
    assert set(re.findall(r"`(FOVEA_[A-Z_]+)`", listed)) == {n for n in passthrough if n.startswith("FOVEA_")}
    compose_vars = {"HOST_PATH", "UID", "GID", "INSTALL_AI"}           # read by compose itself (mounts, user, build)
    for name in ("README.md", "docs/ai-labeling.md"):
        for var in re.findall(r"\.env`에 `([A-Z_]+)", _doc(name)):
            assert var in passthrough | compose_vars, (name, var)
    lan = _line(docker, "**같은 네트워크의 다른 PC에 열기**")
    for part in ("`FOVEA_LOCAL_NETS` 줄을 지우세요", "Docker를 실행한 PC의 브라우저도 다른 PC로 취급됩니다",
                 "토큰을 입력해야 하고", "403", "`FOVEA_ALLOW_REMOTE_AI=1`", "docker compose up -d"):
        assert part in lan, part
    row = _line(_doc("docs/ai-labeling.md"), "*Cloud AI providers and API keys can only be used from the computer")
    assert "Docker" in row and "`FOVEA_ALLOW_REMOTE_AI=1`" in row and "`localhost`" in row

    for name in ("ANTHROPIC", "GEMINI", "OPENAI", "OPENAI_COMPAT"):
        monkeypatch.setenv(f"FOVEA_{name}_BASE_URL", "http://127.0.0.1:9/v1")
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    monkeypatch.setattr(ai_secrets, "_path", lambda: tmp_path / "secrets.json")
    monkeypatch.setenv("FOVEA_TOKEN", "tok-docker")
    host_browser = client_from(app, ("172.17.0.1", 50000), base_url="http://localhost:8888")
    html, auth = {"accept": "text/html"}, {"Authorization": "Bearer tok-docker"}
    key = {"key": "sk-ant-api03-" + "d0cker" * 4}
    # compose as shipped (port on 127.0.0.1, FOVEA_LOCAL_NETS = the bridge ranges): the host browser is "this PC"
    monkeypatch.setattr(security, "_LOCAL_NETS", [ipaddress.ip_network(n) for n in env["FOVEA_LOCAL_NETS"].split(",")])
    assert host_browser.get("/", headers=html).status_code == 200
    assert host_browser.get("/api/meta").json()["server"]["request_is_local"] is True
    # opened to the LAN as documented: the same browser is another machine
    monkeypatch.setattr(security, "_LOCAL_NETS", [])
    page = host_browser.get("/", headers=html)
    assert page.status_code == 401 and 'name="token"' in page.text
    assert host_browser.get("/api/meta", headers=auth).json()["server"]["request_is_local"] is False
    r = host_browser.put("/api/ai/providers/anthropic/key", json=key, headers=auth)
    assert r.status_code == 403 and "FOVEA_ALLOW_REMOTE_AI=1" in r.text
    monkeypatch.setenv("FOVEA_ALLOW_REMOTE_AI", "1")
    r = host_browser.put("/api/ai/providers/anthropic/key", json=key, headers=auth)
    assert r.status_code == 200 and r.json()["key"]["present"] is True and key["key"] not in r.text
    assert host_browser.delete("/api/ai/providers/anthropic/key", headers=auth).status_code == 200


def test_fill_from_neighbours_is_the_one_click_seq_gap_fix_the_docs_describe(tmp_path):
    """data-quality.md / README / keyboard.md: next to a seq_gap chip Annotate has a Fill from neighbours button that
    POSTs …/fill-gap; the answer is pending interpolated proposals 'Filled from <before> and <after>' (Enter accepts)."""
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    chips = annotate[annotate.index("(item.flags || []).forEach((c) => {"):annotate.index("renderFlagInfo();\n    // Middle")]
    assert "c !== 'seq_gap'" in chips and "'Fill from neighbours'" in chips and "onClick: () => fillGap()" in chips
    fill = annotate[annotate.index("async function fillGap()"):]
    fill = fill[:fill.index("\n  }\n")]
    assert "/fill-gap`" in fill and "selectProp(shown[0])" in fill and "Enter accepts" in fill
    root = tmp_path / "seq"
    grey = Image.new("RGB", (64, 48), (110, 110, 110))
    _write(root, "train", "cam_0001", grey, [(0, 0.30, 0.5, 0.2, 0.3)])
    _write(root, "train", "cam_0002", grey, [])
    _write(root, "train", "cam_0003", grey, [(0, 0.50, 0.5, 0.2, 0.3)])
    ds_id = _register(root, ["thing"])
    try:
        _scan(ds_id)
        ids = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?",
                                                                    (ds_id,))}
        res = qa.fill_gap(ds_id, ids["cam_0002"])
        assert len(res["preds"]) == 1
        p = res["preds"][0]
        assert (p["kind"], p["state"], p["cls"]) == ("interp", "pending", 0) and abs(p["xc"] - 0.40) < 0.02
        assert p["reason"] == "Filled from cam_0001.jpg and cam_0003.jpg"
    finally:
        _drop(ds_id)
    line = _line(_doc("docs/data-quality.md"), "`seq_gap`의 경우")
    for part in ("**Fill from neighbours**", "한 번 클릭", "*Filled from <앞 프레임> and <뒤 프레임>*", "`Enter`로 수락",
                 "`K`로 키프레임 두 개"):
        assert part in line, part
    assert "**Fill from neighbours**" in _line(_doc("README.md"), "- **시퀀스 도구**")
    assert "**Fill from neighbours**" in _line(_doc("docs/keyboard.md"), "데이터 품질 플래그 칩을 누르면")


def test_files_list_is_one_tab_stop_entered_after_copy_path_as_keyboard_md_says():
    """keyboard.md (Files): the list is a single roving tab stop after the path row's Copy path button — one Tab from
    a fresh page lands on the sidebar — and opening a folder (Enter, click) or going up (Backspace) keeps the focus
    in the list, on the folder just left when going up."""
    files = (SRC / "static/views/files.js").read_text(encoding="utf-8")
    assert "role: 'listbox', 'aria-label': 'Folder contents', tabindex: '-1'" in files
    assert "role: 'option', tabindex: '-1'" in files                    # rows: roving tabindex, one stop for the list
    crumbs = files[files.index("function renderCrumbs"):files.index("function row(")]
    assert crumbs.rstrip().endswith("'Copy path'));\n  }") and "put(top," in crumbs   # Copy path ends the path row
    assert files.index("root.appendChild(top); root.appendChild(h('div', { class: 'files-body' }, list") > 0
    load = files[files.index("async function load("):files.index("const parentOf")]
    assert "const keepFocus = offset === 0 && list.contains(document.activeElement), from = path;" in load
    assert "r._entry.rel === from" in load and "t.focus()" in load
    doc = _doc("docs/keyboard.md")
    assert "`Tab` 한 번으로 목록에 들어간다" not in doc
    line = _line(doc, "폴더 목록에 포커스가 있을 때 동작한다")
    for part in ("행을 클릭하거나", "**Copy path** 버튼에서 `Tab`을 한 번 더", "`Tab` 한 번으로는 들어가지 않는다",
                 "포커스는 목록에 남고", "방금 있던 폴더 행"):
        assert part in line, part


def test_visual_comfort_describes_both_ghosts_as_drawn():
    """visual-comfort.md rule 10: the previous-frame ghost is a static two-tone dotted outline (dark 2 px dots, light
    1 px dots in the gaps), and a pressed QA flag chip draws a static amber (#f59e0b) ghost in Annotate and Inspect
    that Esc clears."""
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    g = annotate[annotate.index("if (opts.ghost && ghost && ghost.forId"):annotate.index("const fa = fillAlpha();")]
    assert "gc.setLineDash([3, 3]);" in g
    assert "gc.lineWidth = 2; gc.strokeStyle = 'rgba(0,0,0,.8)';" in g
    assert "gc.lineDashOffset = 3; gc.lineWidth = 1; gc.strokeStyle = 'rgba(255,255,255,.95)';" in g
    assert "lineDashOffset = 0" in g and "requestAnimationFrame" not in g
    assert "const FLAG_GHOST = '#f59e0b';" in annotate
    assert "if (flagFocus) { clearFlagFocus(); return; }" in annotate
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    assert "const GHOST = '#f59e0b';" in explore and "dashed: true, color: GHOST" in explore
    doc = _doc("docs/visual-comfort.md")
    assert "1 px 회색" not in doc
    ghost = _line(doc, "**이전 프레임 고스트**")
    for part in ("어두운 2 px 점과 밝은 1 px 점", "정적인 점선", "클래스 색을 쓰지 않아"):
        assert part in ghost, part
    flag = _line(doc, "**데이터 품질 플래그 고스트**")
    for part in ("Annotate", "Inspect", "`#f59e0b`", "주황 테두리", "주황 점선", "`Esc`", "IoU ≥ 0.5"):
        assert part in flag, part


# ============================================================== r5: round-4 behaviour changes the docs now describe

def test_audit_glyphs_in_the_docs_are_the_ones_drawn():
    """ai_spurious is drawn ✕ in every view (it used to be −) and ai_missing +: README, ai-labeling.md's audit table
    and visual-comfort.md rule 10 name exactly the glyphs Annotate, Explore and Objects draw."""
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    objects = (SRC / "static/views/objects.js").read_text(encoding="utf-8")
    head = annotate[annotate.index("const GLYPH = {"):]
    glyph = dict(re.findall(r"(missing|loose|swap|spurious): '([^']+)'", head[:head.index("};")]))
    assert glyph == {"missing": "+", "loose": "⤢", "swap": "⇄", "spurious": "✕"}
    assert "const AUDIT_GLYPH = { missing: '+', loose: '⤢', swap: '⇄', spurious: '✕' };" in explore
    assert "swap: { glyph: '⇄'" in objects and "loose: { glyph: '⤢'" in objects and "spurious: { glyph: '✕'" in objects
    named = "+ 빠짐, ⤢ 느슨함, ⇄ 클래스, ✕ 근거 없음"
    assert named in _line(_doc("README.md"), "AI 감사 의심(")
    assert named in _line(_doc("docs/visual-comfort.md"), "**감사 의심**")
    rows = _md_table(_doc("docs/ai-labeling.md"), "| 종류 | 표시 |")
    assert {r[0].split()[0].strip("`"): r[1] for r in rows} == {f"ai_{k}": v for k, v in glyph.items()}
    assert "⇄ ⤢ ✕" in _line(_doc("docs/ai-labeling.md"), "**Objects**: AI 감사 의심 박스에")
    for name in ("README.md", "docs/ai-labeling.md", "docs/visual-comfort.md"):
        doc = _doc(name)
        assert "＋" not in doc and "− 근거" not in doc and "Objects ✕" not in doc, name


def _p(i, cls, xc, yc, w, h, conf):
    return {"id": i, "cls": cls, "xc": xc, "yc": yc, "w": w, "h": h, "conf": conf, "label": "", "kind": "detect",
            "reason": ""}


def test_append_and_bulk_accept_skip_same_class_and_keep_other_class_pending_as_documented():
    """Append and every bulk accept (Explore Accept proposals, Inspect ⇧Enter, the AI tab): a box over one of its
    class (IoU >= 0.7) is a duplicate and is skipped; a box over a box of ANOTHER class is not written but stays a
    pending proposal (⇄) — not the old 'duplicates are skipped'."""
    from fovea.ai import geometry, runner
    assert runner.DUP_IOU == 0.7
    rows = [[0, 0.2, 0.5, 0.15, 0.3]]
    preds = [_p(1, 0, 0.205, 0.5, 0.15, 0.3, 0.9),        # same class, same object: already labeled
             _p(2, 1, 0.2, 0.5, 0.15, 0.3, 0.95),         # another class on the class-0 object: left to decide
             _p(3, 1, 0.7, 0.5, 0.1, 0.2, 0.8)]           # a new object: written
    new_rows, out = runner.apply_to_rows(rows, preds)     # the accept job
    assert (out["duplicates"], out["conflicts"], out["added"]) == ([1], [2], [3])
    assert [r[0] for r in new_rows] == [0, 1]
    ok, conflicts = runner.split_conflicts(preds, rows, runner.DUP_IOU)      # Append: the same two steps
    fresh, dups = geometry.split_duplicates(ok, rows)
    assert ([p["id"] for p in conflicts], [p["id"] for p in dups], [p["id"] for p in fresh]) == ([2], [1], [3])
    src = (SRC / "fovea/ai/runner.py").read_text(encoding="utf-8")
    apply = src[src.index("    def apply(self, iid: int, img: dict, out: dict)"):src.index("    def _park(")]
    assert "split_conflicts(preds, existing, DUP_IOU)" in apply and "self._park(iid, conflicts)" in apply
    park = src[src.index("    def _park("):src.index("    def keep(")]
    assert '"pending")' in park                                               # conflicts wait as proposals
    # the result lines the docs quote are the ones the UI shows
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    assert "duplicates skipped`" in panel and "over a box of another class left to decide in Annotate`" in panel
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    assert "already labeled`" in explore and "⇄ left to decide (over another class)`" in explore
    for s in (panel, explore, (SRC / "static/views/ai.js").read_text(encoding="utf-8")):
        assert "another class stays" in s                                    # every accept/Append form says so
    for name in ("README.md", "docs/ai-labeling.md"):
        assert "중복은 건너뜀" not in _doc(name), name
    ai_doc = _doc("docs/ai-labeling.md")
    append = _line(ai_doc, "**Append**(기존 박스 옆에 추가")
    for part in ("같은 클래스 박스와 IoU ≥ 0.7로 겹치는 박스는 중복으로 건너뛰고", "다른 클래스 박스에 겹치는 박스는 쓰지 않고",
                 "대기 제안(⇄)", "*N duplicates skipped*", "*N over a box of another class left to decide in Annotate*"):
        assert part in append, part
    bulk = _line(ai_doc, "**일괄 수락의 규칙**")
    for part in ("Explore의 *Accept proposals*", "Inspect `⇧Enter`", "AI 탭", "*already labeled*", "대기로 남겨",
                 "*N ⇄ left to decide (over another class)*"):
        assert part in bulk, part
    readme = _doc("README.md")
    assert "다른 클래스 박스에 겹친 ⇄ 제안은 대기로 남김" in _line(readme, "일괄 작업: Approve/Flag/Exclude")
    tab = _line(readme, "**Overview**(사용액, 제안·의심 수, 공급자 상태)")
    assert "*Append*는 같은 클래스 박스와 겹치는 AI 박스를 중복으로 건너뛰고" in tab and "일괄 수락도 같은 규칙" in tab


def test_undoing_a_bulk_accept_puts_its_proposals_back_to_pending_as_documented(tmp_path):
    """README / ai-labeling.md: a bulk accept's snapshot remembers what it accepted; History's Restore puts the label
    file back and those proposals become pending again ('N AI proposals pending again')."""
    from fovea.ai import store
    from fovea.core import snapshots as snap
    root, ds_id, ids = _label_ds(tmp_path, n=1)
    c = _client()
    try:
        iid = ids["f0_0001"]
        lp = root / "labels/train/f0_0001.txt"
        orig = lp.read_text()
        run_id = store.create_run(ds_id, "batch", "mock", "mock-fast", {}, status="done")
        (pred,) = store.insert_preds(run_id, ds_id, iid, [_p(None, 1, 0.15, 0.15, 0.1, 0.1, 0.9)], "pending")
        j = _wait(c, c.post(f"/api/datasets/{ds_id}/proposals/accept", json={"image_ids": [iid]}).json()["job"]["id"])
        assert j["status"] == "done", j
        assert lp.read_text() != orig and j["result"]["added"] == 1
        state = lambda: db.query_one("SELECT state FROM ai_preds WHERE id=?", (pred["id"],))["state"]  # noqa: E731
        assert state() == "accepted"
        sid = j["result"]["snapshot_id"]
        j = _wait(c, c.post(f"/api/datasets/{ds_id}/snapshots/{sid}/restore", json={}).json()["job"]["id"])
        assert j["status"] == "done", j
        assert lp.read_text() == orig and state() == "pending"
        assert "1 AI proposal pending again" in j["message"]
    finally:
        _cleanup(ds_id)
    assert snap._proposals_text({"pending": 2}) == ["2 AI proposals pending again"]
    assert "*N AI proposals pending again*" in _line(_doc("docs/ai-labeling.md"), "**일괄 수락의 규칙**")
    safety = _doc("README.md").split("## 라벨 파일 안전", 1)[1].split("\n## ", 1)[0]
    assert "*N AI proposals pending again*" in _line(safety, "AI 제안 일괄 수락의 스냅샷은")
    assert "다시 대기 상태로" in _line(_doc("README.md"), "- **History**:")


def test_keyframe_interpolation_pairs_a_fall_and_takes_the_nearer_keyframes_class(tmp_path):
    """K: same-class boxes pair wherever they went; a box that changes class in place (standing -> fallen) pairs
    too, and each frame in between takes the class of the nearer keyframe (the later one at exactly halfway).
    Fill from neighbours (seq_gap) still links same-class boxes only — as data-quality.md says."""
    root = tmp_path / "fall"
    grey = Image.new("RGB", (64, 48), (110, 110, 110))
    _write(root, "train", "cam_0001", grey, [(0, 0.30, 0.50, 0.10, 0.30), (2, 0.15, 0.20, 0.10, 0.10)])
    for k in (2, 3, 4):
        _write(root, "train", f"cam_000{k}", grey, [])
    _write(root, "train", "cam_0005", grey, [(1, 0.32, 0.55, 0.30, 0.10), (2, 0.85, 0.80, 0.10, 0.10)])
    ds_id = _register(root, ["standing", "fallen", "bag"])
    try:
        _scan(ds_id)
        ids = {Path(r["rel_path"]).stem: r["id"] for r in db.query("SELECT id, rel_path FROM images WHERE dataset_id=?",
                                                                    (ds_id,))}
        res = qa.interpolate(ds_id, ids["cam_0001"], ids["cam_0005"])
        assert res["matched"] == 2 and res["unmatched"] == {"from": 0, "to": 0}
        assert [(c["from_cls"], c["to_cls"]) for c in res["class_changes"]] == [(0, 1)]
        assert "nearer keyframe" in res["message"] and "standing → fallen" in res["message"]
        by_img = {}
        for p in res["preds"]:
            by_img.setdefault(p["image_id"], []).append(p)
        person = {k: next(p for p in by_img[ids[k]] if p["cls"] != 2) for k in ("cam_0002", "cam_0003", "cam_0004")}
        assert [person[k]["cls"] for k in ("cam_0002", "cam_0003", "cam_0004")] == [0, 1, 1]
        assert all("class changes standing → fallen" in p["reason"] for p in person.values())
        assert all(any(p["cls"] == 2 for p in by_img[ids[k]]) for k in ("cam_0002", "cam_0003", "cam_0004"))
    finally:
        _drop(ds_id)
    assert (qa.KEYFRAME_CHANGE_DIST, qa.KEYFRAME_CHANGE_AREA, qa.MAX_INTERP_FRAMES) == (0.75, 4.0, 2000)
    src = (SRC / "fovea/core/qa.py").read_text(encoding="utf-8")
    fill = src[src.index("def fill_gap("):src.index("class _Path")]
    rule = src[src.index("def missing_boxes("):src.index("def _explained(")]
    assert "missing_boxes(A, cur, B, W, H, t)" in fill                       # the fix fills what the flag reports
    assert "class_aware=True" in rule and "missing_boxes(p.boxes, cur.boxes, n.boxes" in src
    explained = src[src.index("def _explained("):src.index("def check_sequences(")]
    assert "v >= 0.1 or cd <= 0.75" in explained
    k = _line(_doc("docs/ai-labeling.md"), "**`K` 보간**")
    for part in ("**제자리에서만**", "0.75배", "4배", "**가까운 키프레임의 클래스**", "정확히 가운데면 뒤 키프레임",
                 "*class changes A → B: this frame takes the nearer keyframe's*", "2,000프레임", "IoU ≥ 0.5",
                 "기존 박스 하나는 보간 박스 하나만 대신한다", "*1 box without a partner · 1 already labeled · 1 class change*"):
        assert part in k, part
    assert qa.EXISTING_IOU == 0.5 and "already_labeled(boxes, existing[r[\"id\"]], compat)" in src
    assert "제자리에서 클래스가 바뀐 박스(넘어짐)도 짝지음" in _line(_doc("docs/ai-labeling.md"), "| `K` |")
    assert "가까운 키프레임의 클래스" in _line(_doc("README.md"), "- **시퀀스 도구**")
    assert "가까운 키프레임의 클래스" in _line(_doc("docs/keyboard.md"), "`K`는 이 프레임을 키프레임으로")
    dq = _doc("docs/data-quality.md")
    gap = _line(dq, "`seq_gap`의 경우")
    for part in ("`seq_gap`과 같은 규칙", "같은 클래스로 이어지는", "플래그가 알린 것을 그대로 채움", "같은 클래스끼리만 잇는다",
                 "`K` 보간"):
        assert part in gap, part
    flag = _line(dq, "- **`seq_gap`**:")
    for part in ("1:1", "IoU ≥ 0.1", "0.75배", "**하나**"):
        assert part in flag, part


def test_propose_per_image_and_the_no_cap_confirmation_are_documented():
    """Annotate's I runs per image (up to 3 at once; moving on keeps the request, its answer is saved for its own
    image; Stop only stops waiting), and a cloud batch or audit that no budget or cap could stop asks first once its
    high estimate reaches $5 (20 images for a model with no price on file)."""
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    assert "const MAX_RUNNING = 3;" in panel
    for s in ("`Already proposing for ${running.size} images", "— they show when you go back to it.`",
              "'Proposing for\\u00a0'", "If the model still finishes, its proposals are saved for"):
        assert s in panel, s
    assert "export const SPEND_CONFIRM_USD = 5;" in panel and "export const SPEND_CONFIRM_IMAGES = 20;" in panel
    for s in ("title: `Start a cloud ${what} with no spend cap?`", "'Submit without a cap'", "'Set a cap'",
              "title: `Start a ${what} whose cost is unknown?`", "No spend cap: nothing stops this run before it ends"):
        assert s in panel, s
    ai_tab = (SRC / "static/views/ai.js").read_text(encoding="utf-8")
    audit = ai_tab[ai_tab.index("async function startAudit()"):]
    audit = audit[:audit.index("const res = await fovea.api.post('/api/ai/audit'")]
    assert "!sourceCap(" in audit and "spendCheck(est)" in audit and "what: 'label audit'" in audit
    ai_doc = _doc("docs/ai-labeling.md")
    per = _line(ai_doc, "**`I`는 이미지마다 따로 돈다.**")
    for part in ("*Proposing for <파일>*", "**Stop**", "동시에 3장까지", "*Already proposing for 3 images*",
                 "they show when you go back to it"):
        assert part in per, part
    ask = _line(ai_doc, "**상한 없는 실행은 한 번 더 묻는다**")
    for part in ("$5 이상", "*Start a cloud run with no spend cap?*", "**Set a cap**", "*Submit without a cap*",
                 "*No spend cap: nothing stops this run before it ends…*", "20장 이상"):
        assert part in ask, part
    cap = _line(ai_doc, "**상한 없는 실행의 확인**")
    assert "라벨 감사" in cap and "$5 이상" in cap
    assert "**Stop**은 기다림만 끝낸다" in _line(_doc("docs/keyboard.md"), "`I`는 지금 이미지에 대한 요청이다")
    readme = _doc("README.md")
    assert "최대 3장 동시" in _line(readme, "- **AI 제안**:")
    assert "*Start a cloud run with no spend cap?*" in _line(readme, "- **비용**:")


def test_annotate_tab_cycles_only_where_keyboard_md_says():
    """Tab cycles visible proposals then boxes over the canvas; off the canvas only with a selection and no control
    focused, so keyboard users can still Tab to the panel. A control reached with Tab keeps Enter / Space."""
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    tab = annotate[annotate.index("if (e.key === 'Tab') {"):]
    tab = tab[:tab.index("\n    }\n")]
    assert "if (!mouse.inside && (onControl || !sel)) return;" in tab and "cycle(e.shiftKey ? -1 : 1)" in tab
    cyc = annotate[annotate.index("function cycle(dir)"):]
    cyc = cyc[:cyc.index("\n  }\n")]
    assert cyc.index("visProps.map") < cyc.index("boxes.map") and "hiddenCls.has" in cyc and "% order2.length" in cyc
    assert "if (kbdFocus && (e.key === 'Enter' || c === 'Space')) return;" in annotate
    assert "f.matches('input[type=\"range\"], select')" in annotate
    line = _line(_doc("docs/keyboard.md"), "`Tab`/`⇧Tab`은 보이는 제안을 먼저")
    for part in ("포인터가 그림 위에 있으면 항상", "무언가 선택되어 있고", "컨트롤에 있지 않을 때만", "기본 포커스 이동",
                 "`Enter`·`Space`가 그 컨트롤을 누르고", "슬라이더·선택 상자"):
        assert part in line, part
    assert "포인터가 그림 밖이면 무언가 선택되어 있을 때만" in _line(_doc("docs/ai-labeling.md"), "| `Tab` / `⇧Tab` |")


# ============================================================== round 5: default order, reservations, remote guard

def test_flag_and_audit_lists_open_most_suspicious_first_as_documented():
    """A list filtered by a QA flag opens by that flag's score (sort=flag, descending), audit findings by the audit
    score, anything else by index — in Explore, in Annotate and from the Overview links — as README and
    data-quality.md say. The flag sort is only offered when the issue filter names a QA flag."""
    from fovea.core import imagesq
    assert imagesq.DEFAULT_DESC == {"audit", "flag"}
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    overview = (SRC / "static/views/overview.js").read_text(encoding="utf-8")
    ai_tab = (SRC / "static/views/ai.js").read_text(encoding="utf-8")
    assert "['flag', 'Most suspicious first']" in explore and "flag: 'Most suspicious'" in explore
    assert "sort === 'audit' || sort === 'flag' ? 'desc' : 'asc'" in explore
    for src in (explore, annotate):        # same rule in both views: set, flag, audit, index
        body = src[src.index("(f) {\n  f = f || {};"):]
        body = body[:body.index("\n}\n")]
        assert [ln.strip() for ln in body.splitlines()[2:6]] == [
            "if (f.set) return 'set';", "if (flagSortable(f)) return 'flag';",
            "if (issueCodes(f).some(c => AUDIT_CODES.has(c)) || f.ai === 'audit') return 'audit';"
            if src is annotate else "if (csv(f.issue).some(c => AUDIT_CODES.has(c)) || f.ai === 'audit') return 'audit';",
            "return 'id';"]
    flag_codes = re.search(r"const FLAG_CODES = new Set\(\[([^\]]+)\]\)", annotate).group(1)
    assert sorted(re.findall(r"'(\w+)'", flag_codes)) == sorted(imagesq.FLAG_LABELS)
    assert "export const flagReview = (code) => ({ issue: code, sort: 'flag', order: 'desc' });" in overview
    assert "sort: 'audit', order: 'desc'" in overview and "sort: 'audit', order: 'desc'" in ai_tab
    readme, dq = _doc("README.md"), _doc("docs/data-quality.md")
    assert "Most suspicious first(데이터 품질 플래그로 거를 때만)" in _line(readme, "- 툴바:")
    rule = _line(readme, "**기본 정렬은 필터를 따릅니다**")
    for part in ("*Most suspicious ↓*", "*Most suspect ↓*", "Index ↑", "URL에 남아"):
        assert part in rule, part
    assert "가장 의심스러운 것부터" in _line(readme, "- **Data quality**:")
    assert "`?issue=<code>&sort=flag&order=desc`" in _line(dq, "Overview 카드의 **Review**")
    score = _line(dq, "**가장 의심스러운 것부터**")
    for part in ("`sort=flag`", "*class* > *count* > *boxes*", "`issue=any`는 점수가 없어 Index 순", "URL에 남는다"):
        assert part in score, part
    src = (SRC / "fovea/core/qa.py").read_text(encoding="utf-8")
    assert '_REASON_RANK = {"class": 3, "count": 2, "boxes": 1}' in src
    assert 'flags[(cur.id, "seq_gap")] = (min(1.0, len(gaps) / 2)' in src
    assert 'flags[(fr.id, "seq_flicker")] = (1.0 if run == 1 else 0.7' in src


def test_runs_still_going_hold_part_of_the_spend_cap_as_documented():
    """Money a running run has committed but not been billed for counts against the 30-day cap; Settings shows it."""
    from fovea.ai import runner
    src = (SRC / "fovea/ai/runner.py").read_text(encoding="utf-8")
    cap = src[src.index("def cap_remaining("):src.index("def _fmt_usd(")]
    assert "cap - spent_30d(provider) - reserved_usd(provider, exclude_run)" in cap
    assert callable(runner.reserved_usd) and "is held by runs still going" in src
    settings = (SRC / "static/views/settings.js").read_text(encoding="utf-8")
    assert " · reserved by runs still going " in settings and "`Spend cap held by runs still going (" in settings
    line = _line(_doc("docs/ai-labeling.md"), "- **예약**:")
    for part in ("Economy", "Now", "*spent $0.23 in the last 30 days · reserved by runs still going $0.92*",
                 "*Spend cap held by runs still going*", "*(… is held by runs still going)*"):
        assert part in line, part
    assert "*reserved by runs still going*" in _line(_doc("README.md"), "- **비용**:")


def test_remote_guard_covers_a_keyed_local_server_and_the_server_decides_where_the_browser_is(monkeypatch):
    """Other machines may not call cloud providers nor a local OpenAI-compatible server that is sent the owner's key;
    whether a browser is on another machine is the server's call (/api/meta request_is_local)."""
    from types import SimpleNamespace
    from fovea.ai import registry
    monkeypatch.setitem(registry._providers, "openai_compat", SimpleNamespace(kind="local"))   # on a loopback URL
    monkeypatch.setattr(registry.secrets, "get_key", lambda pid: None)
    assert registry.is_guarded("anthropic") and not registry.is_guarded("openai_compat")
    monkeypatch.setattr(registry.secrets, "get_key", lambda pid: "k-test")
    assert registry.is_guarded("openai_compat")
    panel = (SRC / "static/views/ai-panel.js").read_text(encoding="utf-8")
    explore = (SRC / "static/views/explore.js").read_text(encoding="utf-8")
    for s in ("'Cloud AI is off for browsers on other machines'", "'Uses the server’s saved API key — off for browsers on other machines'",
              "P.request_is_local = server ? server.request_is_local : null;"):
        assert s in panel, s
    assert "'Cloud AI off from this browser'" in (SRC / "static/views/ai.js").read_text(encoding="utf-8")
    assert "m.server.request_is_local" in explore and "p.guarded" in explore
    line = _line(_doc("docs/ai-labeling.md"), "- **원격 차단**:")
    for part in ("저장된 키를 보내는 OpenAI 호환 서버", "`request_is_local`", "*Cloud AI off from this browser*",
                 "*Cloud AI is off for browsers on other machines*", "*Uses the server’s saved API key — off for browsers on other machines*"):
        assert part in line, part
    sec = _line(_doc("README.md"), "- **원격 AI 차단**:")
    assert "저장된 키를 보내는 OpenAI 호환 서버" in sec and "`request_is_local`" in sec


def test_proposal_decisions_that_do_not_reach_the_server_are_kept_as_documented():
    """A failed decide request keeps the decisions queued ("Decisions not saved") and retries 4 s, 8 s, then every
    16 s; ⌘S retries now; the threshold keys are also buttons."""
    annotate = (SRC / "static/views/annotate.js").read_text(encoding="utf-8")
    assert "setSaveState('error', 'Decisions not saved')" in annotate
    assert "4000 * 2 ** Math.min(2, decFails - 1)" in annotate and "Date.now() - lastFailToast > 8000" in annotate
    assert "if (K.is(e, 'mod+s')) { e.preventDefault(); flush(); return; }" in annotate
    assert "class: 'kbd ann-thr-step'" in annotate and "disabled: d < 0 ? thr <= 0 : thr >= 1" in annotate
    assert "not saved — ${n === 1 ? 'it shows' : 'they show'} as pending again" in annotate
    ai_doc = _doc("docs/ai-labeling.md")
    line = _line(ai_doc, "- 결정은 이미지를 떠날 때 서버에 저장되고")
    for part in ("*Decisions not saved*", "4초·8초 뒤, 이후 16초마다", "`⌘S`는 바로", "8초에 한 번까지",
                 "*N proposal decisions were not saved — they show as pending again*"):
        assert part in line, part
    assert "`{` `}` 버튼" in _line(ai_doc, "| `{` / `}` |")
    assert "*Decisions not saved*" in _line(_doc("docs/keyboard.md"), "`⌘S`는 박스와 함께")
    assert "0과 1에서 비활성" in _line(_doc("docs/keyboard.md"), "`{` `}`는 패널 *Proposals* 칸")
