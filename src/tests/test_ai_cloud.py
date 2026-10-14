"""Cloud VLM providers (Claude / Gemini / OpenAI / OpenAI-compatible) against fake HTTP servers.

The fakes in ``fake_vlm.py`` look at the image they receive (synthetic scenes draw objects as colour blocks) and
answer in each provider's native coordinate frame for the size they actually received, so these tests check the real
coordinate maths end to end: pre-resize, tiling offsets, EXIF orientation, y-first vs x-first, pixel vs 0-1000 vs 0-1.
Run: python -m pytest src/tests/test_ai_cloud.py -q
"""
from __future__ import annotations

import json
import math
import os
import sys
import time
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))
os.environ.setdefault("FOVEA_DATA_DIR", str(Path(__file__).resolve().parent / ".tmp-data"))

from fake_vlm import KEYS, FakeVLM, find_blocks, make_scene, smart_resize as fake_smart_resize  # noqa: E402
from fake_vlm import _schema_problems  # noqa: E402

from fovea.ai import vlm  # noqa: E402
from fovea.ai.base import ClassifyRequest, ClassSpec, Crop, DetectRequest, ProviderError  # noqa: E402
from fovea.ai.providers.anthropic import AnthropicProvider  # noqa: E402
from fovea.ai.providers.gemini import GeminiProvider  # noqa: E402
from fovea.ai.providers.openai import OpenAIProvider  # noqa: E402
from fovea.ai.providers.openai_compat import OpenAICompatProvider, coord_from_model  # noqa: E402

RED, BLUE, GREEN, YELLOW, MAGENTA, CYAN = (230, 30, 30), (30, 60, 230), (30, 190, 40), (240, 220, 20), (200, 40, 220), (20, 210, 220)
CLASSES = [ClassSpec(0, "person"), ClassSpec(1, "car"), ClassSpec(2, "class_2", prompt="fallen person"),
           ClassSpec(3, "dog")]


# ---------------------------------------------------------------------------------------------- fixtures

@pytest.fixture(scope="module")
def fake():
    f = FakeVLM().start()
    yield f
    f.stop()


@pytest.fixture(autouse=True)
def _clean(fake, monkeypatch):
    fake.reset()
    waits = []
    monkeypatch.setattr(vlm, "_sleep", lambda s: waits.append(s))
    vlm._limiters.clear()
    vlm.VLMProvider._models_cache.clear()
    AnthropicProvider._unsupported.clear()
    GeminiProvider._unsupported.clear()
    OpenAICompatProvider._unsupported.clear()
    OpenAIProvider._unsupported.clear()
    yield waits


@pytest.fixture
def waits(_clean):
    return _clean


def anth(fake, **cfg):
    return AnthropicProvider(config={"base_url": fake.url, **cfg}, key=KEYS["anthropic"])


def gem(fake, **cfg):
    return GeminiProvider(config={"base_url": fake.url, **cfg}, key=KEYS["gemini"])


def oai(fake, **cfg):
    return OpenAIProvider(config={"base_url": fake.url + "/v1", **cfg}, key=KEYS["openai"])


def compat(fake, **cfg):
    return OpenAICompatProvider(config={"base_url": fake.url + "/v1", **cfg}, key="")


ALL = [("anthropic", anth), ("gemini", gem), ("openai", oai), ("compat", compat)]

SCENE_OBJS = [(RED, (100, 120, 400, 700)), (BLUE, (900, 500, 1500, 900)), (YELLOW, (1000, 100, 1200, 300))]
SCENE_GT = {0: (100, 120, 400, 700), 1: (900, 500, 1500, 900), 2: (1000, 100, 1200, 300)}


@pytest.fixture
def scene(tmp_path):
    p = tmp_path / "scene.jpg"
    make_scene(str(p), (1600, 1000), SCENE_OBJS)
    return str(p)


def px_boxes(dets, W, H):
    """{cls: (x1, y1, x2, y2) in pixels} (one box per class in these scenes)."""
    out = {}
    for d in dets:
        out.setdefault(d.cls, []).append(((d.xc - d.w / 2) * W, (d.yc - d.h / 2) * H, (d.xc + d.w / 2) * W,
                                          (d.yc + d.h / 2) * H))
    return out


def assert_boxes(dets, gt, W, H, tol_frac=0.012):
    got = px_boxes(dets, W, H)
    assert sorted(got) == sorted(gt), (got, gt)
    tol = max(W, H) * tol_frac
    for c, box in gt.items():
        assert len(got[c]) == 1, got[c]
        for a, b in zip(got[c][0], box):
            assert abs(a - b) <= tol, (c, got[c][0], box, tol)


# ---------------------------------------------------------------------------------------------- resize maths

def _fits(w, h, max_edge, max_tokens):
    return (math.ceil(w / 28) * 28 <= max_edge and math.ceil(h / 28) * 28 <= max_edge
            and math.ceil(w / 28) * math.ceil(h / 28) <= max_tokens)


@pytest.mark.parametrize("w,h", [(1920, 1080), (1080, 1920), (4000, 3000), (8000, 600), (640, 640), (3000, 3000),
                                 (2577, 100), (29, 29), (1, 5000)])
@pytest.mark.parametrize("tier", [(2576, 4784), (1568, 1568)])
def test_anthropic_resized_size_is_the_largest_fitting_size(w, h, tier):
    rw, rh = vlm.anthropic_resized_size(w, h, *tier)
    assert _fits(rw, rh, *tier)
    if (rw, rh) != (w, h):
        # aspect preserved (within rounding) and one more pixel on the long edge would not fit
        assert abs(rw / rh - w / h) <= max(1 / rh, 1 / rw) * (w / h) + 0.02
        if w >= h:
            assert not _fits(rw + 1, max(round((rw + 1) / (w / h)), 1), *tier)
        else:
            assert not _fits(max(round((rh + 1) * (w / h)), 1), rh + 1, *tier)
    else:
        assert _fits(w, h, *tier)


def test_anthropic_resized_size_documented_examples():
    # Anthropic's worked example: 1920x1080 stays on the high-res tier, becomes 1456x819 on the standard tier
    assert vlm.anthropic_resized_size(1920, 1080) == (1920, 1080)
    assert vlm.anthropic_resized_size(1920, 1080, 1568, 1568) == (1456, 819)
    assert vlm.anthropic_image_tokens(1456, 819) == 1560
    assert vlm.anthropic_image_tokens(1920, 1080) == 2691


@pytest.mark.parametrize("w,h", [(1920, 1080), (4000, 3000), (640, 480), (3000, 200), (200, 3000)])
def test_openai_patch_budget(w, h):
    sw, sh = vlm.patch_fit_size(w, h)
    assert max(sw, sh) <= 2048 and math.ceil(sw / 32) * math.ceil(sh / 32) <= 2500
    assert abs(sw / sh - w / h) / (w / h) < 0.02
    assert vlm.openai_image_tokens(1920, 1080) == 2448     # documented: 2040 patches x 1.2


@pytest.mark.parametrize("w,h", [(1920, 1080), (640, 640), (810, 1080), (4000, 3000), (50, 40)])
def test_qwen25_smart_resize_is_idempotent_on_its_own_grid(w, h):
    sw, sh = vlm.smart_resize(w, h)
    assert sw % 28 == 0 and sh % 28 == 0 and sw * sh <= 1280 * 28 * 28
    # the server's own smart_resize (larger max_pixels) must leave our pre-resized image unchanged
    assert fake_smart_resize(sw, sh) == (sw, sh)
    assert vlm.smart_resize(sw, sh) == (sw, sh)


def test_fit_long_edge_never_upscales():
    assert vlm.fit_long_edge(640, 480, 1536) == (640, 480)
    assert vlm.fit_long_edge(3000, 1500, 1500) == (1500, 750)


def test_tile_regions_cover_image_with_overlap():
    W, H = 3000, 2000
    tiles = vlm.tile_regions(W, H, 3, 2, 0.2)
    assert len(tiles) == 6
    tw, th = tiles[0][2], tiles[0][3]
    assert tiles[0][:2] == (0.0, 0.0)
    assert math.isclose(tiles[-1][0] + tw, W) and math.isclose(tiles[-1][1] + th, H)
    # neighbours overlap by 20% of a tile
    assert math.isclose((tiles[0][0] + tw) - tiles[1][0], 0.2 * tw, rel_tol=1e-6)
    assert vlm.tile_plan(W, H, False, 1536) == []
    assert vlm.tile_plan(1000, 800, "auto", 1536) == []            # small image: no tiling
    assert len(vlm.tile_plan(6000, 4000, True, 1536)) == 9           # 3.9x the native size -> 3x3
    assert len(vlm.tile_plan(W, H, 2, 1536)) == 4
    assert len(vlm.tile_plan(W, H, [3, 1], 1536)) == 3


# ---------------------------------------------------------------------------------------------- frames

def test_frame_orders_and_scales():
    assert vlm._to_xyxy_order([100, 200, 300, 400], vlm.YXYX_1000) == (200, 100, 400, 300)
    assert vlm._to_xyxy_order([100, 200, 300, 400], vlm.XYXY_1000) == (100, 200, 300, 400)
    assert vlm.to_unit((50, 25, 150, 75), "pixel", 200, 100) == (0.25, 0.25, 0.75, 0.75)
    assert vlm.to_unit((250, 500, 750, 1000), "k1000", 9, 9) == (0.25, 0.5, 0.75, 1.0)


@pytest.mark.parametrize("frame,values,size,scale,anomaly", [
    (vlm.PIXEL_XYXY, [10, 20, 600, 400], (640, 480), "pixel", False),
    (vlm.PIXEL_XYXY, [0.1, 0.2, 0.5, 0.9], (640, 480), "unit", True),
    (vlm.PIXEL_XYXY, [100, 200, 950, 990], (640, 480), "k1000", True),      # > image size but <= 1000
    (vlm.PIXEL_XYXY, [100, 200, 1900, 1000], (1920, 1080), "pixel", False),  # big image: pixels are > 1000
    (vlm.YXYX_1000, [100, 200, 900, 990], (1536, 864), "k1000", False),
    (vlm.YXYX_1000, [100, 200, 1400, 990], (1536, 864), "pixel", True),
    (vlm.YXYX_1000, [0.1, 0.2, 0.9, 0.95], (1536, 864), "unit", True),
    (vlm.XYXY_1000, [0, 0, 1000, 1000], (1280, 720), "k1000", False),
    (vlm.NORM_XYXY, [0.1, 0.2, 0.3, 1.0], (1280, 720), "unit", False),
    (vlm.NORM_XYXY, [100, 200, 300, 900], (1280, 720), "k1000", True),
    (vlm.NORM_XYXY, [100, 200, 1300, 700], (1280, 720), "pixel", True),
])
def test_scale_guard(frame, values, size, scale, anomaly):
    got, note = vlm.detect_scale(values, frame, *size)
    assert got == scale and bool(note) == anomaly


def test_region_to_full_offsets_tiles():
    prep = vlm.Prepared(b"", "image/jpeg", 500, 400, (1000.0, 600.0, 1000.0, 800.0), (3000, 2000))
    x1, y1, x2, y2 = vlm.region_to_full((0.0, 0.0, 0.5, 0.5), prep)
    assert (x1 * 3000, y1 * 2000, x2 * 3000, y2 * 2000) == (1000.0, 600.0, 1500.0, 1000.0)


# ---------------------------------------------------------------------------------------------- vocab / json

def test_vocab_resolution():
    v = vlm.Vocab.build(CLASSES)
    assert v.labels == ["person", "car", "fallen person", "dog"]
    for lab, cid in [("person", 0), ("Person", 0), ("persons", 0), ("PEOPLE", None), ("Fallen_Person", 2),
                     ("fallen-persons", 2), ("class_2", 2), ("cars", 1), ("  car ", 1), ("Dogs", 3),
                     ("cat", None), (None, None), ("", None)]:
        c = v.resolve(lab)
        assert (c.id if c else None) == cid, lab


def test_vocab_duplicate_prompts_stay_distinct():
    v = vlm.Vocab.build([ClassSpec(0, "a", prompt="person"), ClassSpec(1, "b", prompt="person")])
    assert len(set(v.labels)) == 2
    assert v.resolve(v.labels[0]).id == 0 and v.resolve(v.labels[1]).id == 1


def test_extract_json_variants():
    assert vlm.extract_json('{"objects": []}') == {"objects": []}
    assert vlm.extract_json('```json\n{"objects": [1]}\n```') == {"objects": [1]}
    assert vlm.extract_json('Here you go:\n[{"label": "a"}]\nThanks!') == [{"label": "a"}]
    assert vlm.extract_json('prefix {"objects": [{"label": "a", "box_2d": [1, 2, 3, 4]}]} suffix')["objects"][0]["label"] == "a"
    with pytest.raises(vlm.BadOutput):
        vlm.extract_json("I could not find anything.")
    with pytest.raises(vlm.BadOutput):
        vlm.extract_json("")


def test_salvage_truncated_items():
    text = '{"objects": [{"label": "a", "x1": 1, "y1": 2, "x2": 3, "y2": 4}, {"label": "b", "x1": 5, "y1"'
    assert vlm.salvage_items(text) == [{"label": "a", "x1": 1, "y1": 2, "x2": 3, "y2": 4}]
    assert vlm.salvage_items('{"objects": [{"lab') == []
    assert vlm.salvage_items('[{"a": 1}, {"a": 2}, {"a"', "objects") == [{"a": 1}, {"a": 2}]


def test_item_box_shapes():
    assert vlm.item_box({"x1": 1, "y1": 2, "x2": 3, "y2": 4}, vlm.PIXEL_XYXY) == (1, 2, 3, 4)
    assert vlm.item_box({"box_2d": [10, 20, 30, 40]}, vlm.YXYX_1000) == (20, 10, 40, 30)
    assert vlm.item_box({"bbox_2d": [10, 20, 30, 40]}, vlm.XYXY_1000) == (10, 20, 30, 40)
    assert vlm.item_box({"xmin": 1, "ymin": 2, "xmax": 3, "ymax": 4}, vlm.XYXY_1000) == (1, 2, 3, 4)
    assert vlm.item_box({"box_2d": [1, 2, 3]}, vlm.YXYX_1000) is None
    assert vlm.item_box({"box_2d": ["a", 2, 3, 4]}, vlm.YXYX_1000) is None


def test_schemas_match_each_api_dialect():
    labels = ["person", "car"]
    a = vlm.detect_schema(labels, "xyxy", bounds=False)
    assert _schema_problems(a, forbid=("minimum", "maximum", "maxItems")) == ""
    g = vlm.detect_schema(labels, "array", bounds=True, max_items=100, box_key="box_2d", bound_max=1000)
    assert g["properties"]["objects"]["maxItems"] == 100
    assert g["properties"]["objects"]["items"]["properties"]["box_2d"]["items"]["maximum"] == 1000
    assert _schema_problems(g, strict=True) == ""                    # OpenAI strict rules
    c = vlm.classify_schema(vlm.classify_labels(vlm.Vocab.build(CLASSES)), bounds=True, n=5)
    assert _schema_problems(c, strict=True) == ""
    assert c["properties"]["items"]["items"]["properties"]["label"]["enum"][-2:] == ["none", "uncertain"]


def test_prompt_states_frame_size_and_classes():
    v = vlm.Vocab.build([ClassSpec(0, "class_0", prompt="person lying on the floor", description="Exclude sitting.")])
    s = vlm.detect_system_prompt(v, vlm.PIXEL_XYXY, 100)
    assert '"person lying on the floor": Exclude sitting.' in s and "at most 100" in s and "pixel" in s
    assert "{" not in s.split("Coordinate format:")[1]          # no unformatted placeholders
    assert vlm.detect_system_prompt(v, vlm.PIXEL_XYXY, 100) == s  # identical for every image: cacheable
    u = vlm.detect_user_text(v, vlm.PIXEL_XYXY, 1456, 819)
    assert "1456x819" in u and "THIS 1456x819 image" in u
    assert "ymin" in vlm.detect_user_text(v, vlm.YXYX_1000, 1536, 864)


# ---------------------------------------------------------------------------------------------- retry / scrub / cost

def test_parse_retry_after_variants():
    assert vlm.parse_retry_after({"retry-after": "7"}) == 7.0
    assert vlm.parse_retry_after({"retry-after-ms": "1500"}) == 1.5
    from email.utils import formatdate
    d = vlm.parse_retry_after({"retry-after": formatdate(time.time() + 30, usegmt=True)})
    assert 25 <= d <= 31
    assert vlm.parse_retry_after({}, {"error": {"details": [{"@type": "x.RetryInfo", "retryDelay": "37s"}]}}) == 37.0
    assert vlm.parse_retry_after({"x-ratelimit-reset-requests": "6m0s"}) == 360.0
    assert vlm.parse_retry_after({"x-ratelimit-reset-tokens": "250ms"}) == 0.25
    assert vlm.parse_retry_after({}) is None


def test_scrub_removes_every_key_shape():
    text = ("x-api-key: sk-ant-api03-AbCdEf123456 Authorization: Bearer sk-proj-ABCDEFGHIJKLMNOPQRST "
            "url https://x/v1beta/models/m:generateContent?key=AIzaSyA1234567890abcdefghij&alt=json "
            "and AIzaSyB0987654321zyxwvutsr and sk-live-0123456789abcdef0123 and custom-secret-XYZ12345")
    s = vlm.scrub(text, "custom-secret-XYZ12345")
    for leaked in ("AbCdEf123456", "ABCDEFGHIJKLMNOPQRST", "AIzaSyA1234567890", "AIzaSyB0987654321", "0123456789abcdef0123",
                   "custom-secret-XYZ12345"):
        assert leaked not in s, s


def test_cost_usd():
    # Sonnet 5.5: $2 in / $10 out per MTok
    assert vlm.cost_usd("anthropic", "claude-sonnet-5-5", 1_000_000, 100_000) == pytest.approx(3.0)
    assert vlm.cost_usd("anthropic", "claude-sonnet-5-5", 1_000_000, 100_000, batch=True) == pytest.approx(1.5)
    # cache: 400k read at $0.10, 100k written at 1.25x, 500k plain
    assert vlm.cost_usd("anthropic", "claude-sonnet-5-5", 1_000_000, 0, cached_tokens=400_000,
                        cache_write_tokens=100_000) == pytest.approx(0.5 * 2 + 0.4 * 0.10 + 0.1 * 2.5)
    assert vlm.cost_usd("openai", "gpt-6.1-sol-2026-09-01", 1_000_000, 0) == pytest.approx(2.0)   # dated id
    assert vlm.cost_usd("gemini", "models/gemini-3.8-flash", 1_000_000, 1_000_000) == pytest.approx(4.5)
    assert vlm.cost_usd("anthropic", "claude-unknown-9", 10 ** 6, 10 ** 6) == 0.0
    assert vlm.prices_as_of() == "2026-10-13"
    for prov in ("anthropic", "gemini", "openai"):
        defaults = [m for m in vlm.static_models(prov) if m.get("default")]
        assert len(defaults) == 1 and defaults[0]["price"]["in"] > 0


def test_model_variants_do_not_cross_versions():
    assert vlm.is_variant("gpt-6.1-sol-2026-09-01", "gpt-6.1-sol")
    assert vlm.is_variant("claude-sonnet-5-5-20260901", "claude-sonnet-5-5")
    assert vlm.is_variant("qwen3.8:27b-q4_K_M", "qwen3.8:27b")
    assert not vlm.is_variant("claude-opus-5-5", "claude-opus-5")
    assert not vlm.is_variant("gpt-6-sol", "gpt-6")
    assert vlm.find_model("anthropic", "claude-opus-5-7") is None          # a future model: no guessed price
    assert vlm.find_model("anthropic", "claude-opus-5")["price"]["in"] == 5.0


def test_pick_effort():
    assert vlm.pick_effort("xhigh", ["low", "medium", "high"]) == "high"
    assert vlm.pick_effort("minimal", ["low", "medium", "high"]) == "low"
    assert vlm.pick_effort("medium", ["minimal", "low", "medium", "high"]) == "medium"
    assert vlm.pick_effort("max", []) == "max"


def test_nms_merge_and_votes():
    B = vlm.Box
    a, b = B(0, .1, .1, .5, .5), B(0, .11, .1, .5, .52)
    assert len(vlm.nms([a, b], 0.5)) == 1
    assert len(vlm.nms([a, B(1, .1, .1, .5, .5)], 0.5)) == 2                    # class-aware
    # a tile box cut by the tile border inside the full-image box is dropped
    full = B(0, .40, .10, .70, .60, tile=0)
    cut = B(0, .40, .10, .52, .60, edge=True, tile=1)
    assert vlm.merge_tiles([cut, full]) == [full]
    runs = [[B(0, .1, .1, .3, .3), B(1, .6, .6, .9, .9)], [B(0, .11, .1, .31, .3)], [B(0, .1, .12, .3, .3)]]
    m = vlm.vote_merge(runs)
    assert [(x.cls, x.conf) for x in m] == [(0, 1.0), (1, pytest.approx(1 / 3, abs=1e-3))]


def test_retries_honour_retry_after_and_report_waits(waits):
    calls, seen = [], []
    outcomes = [vlm.Transient("busy", "rate", 3.0, 429), vlm.Transient("busy", "overloaded", None, 529), "ok"]

    def call():
        calls.append(1)
        o = outcomes[len(calls) - 1]
        if isinstance(o, Exception):
            raise o
        return o

    with vlm.wait_listener(lambda s, why: seen.append((s, why))):
        assert vlm.with_retries(call, provider="X") == "ok"
    assert len(calls) == 3
    assert 3.0 <= sum(waits[:3]) <= 3.8                       # retry-after 3 s (+ <=25% jitter), slept in 1 s slices
    assert seen[0][0] >= 3 and "rate limited" in seen[0][1]
    assert any(s == 0 for s, _ in seen)


def test_current_waits_visible_from_other_threads(monkeypatch):
    import threading
    seen, go = [], threading.Event()

    def slow_sleep(s):
        seen.append(vlm.current_waits())
        go.set()

    monkeypatch.setattr(vlm, "_sleep", slow_sleep)
    t = threading.Thread(target=lambda: vlm._wait(2.0, "Anthropic rate limited"))
    t.start(), t.join()
    assert seen[0] == [{"reason": "Anthropic rate limited", "seconds_left": 2}]
    assert vlm.current_waits() == []


def test_retries_give_up_with_the_transient_code(waits):
    with pytest.raises(ProviderError) as ei:
        vlm.with_retries(lambda: (_ for _ in ()).throw(vlm.Transient("down", "overloaded", None, 503)),
                         policy=vlm.RetryPolicy(tries=3))
    assert ei.value.code == "overloaded" and "3 attempts" in str(ei.value)
    assert ei.value.fatal is False


def test_cancel_during_wait_aborts(waits):
    class Cancelled(Exception):
        pass

    def listener(s, why):
        raise Cancelled()

    with vlm.wait_listener(listener), pytest.raises(Cancelled):
        vlm.with_retries(lambda: (_ for _ in ()).throw(vlm.Transient("x", "rate", 60, 429)))


def test_adaptive_limiter_halves_on_rate_limit():
    lim = vlm.AdaptiveLimiter(8)
    lim.rate_limited()
    assert lim.limit == 4
    lim.rate_limited(), lim.rate_limited(), lim.rate_limited()
    assert lim.limit == 1
    for _ in range(10):
        lim.success()
    assert lim.limit == 2


# ---------------------------------------------------------------------------------------------- end to end per provider

@pytest.mark.parametrize("name,mk", ALL)
def test_detect_maps_boxes_back_exactly(fake, scene, name, mk):
    r = mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert_boxes(r.detections, SCENE_GT, 1600, 1000)
    assert all(d.conf is None for d in r.detections)               # no fake confidences
    assert r.warnings == [] and r.latency_ms >= 0
    assert r.usage.input_tokens > 0 and r.usage.output_tokens > 0
    if name != "compat":
        assert r.usage.cost_usd > 0
        assert r.served_by == vlm.default_model({"anthropic": "anthropic", "gemini": "gemini", "openai": "openai"}[name])
    else:
        assert r.usage.cost_usd == 0
    assert {d.label for d in r.detections} == {"person", "car", "fallen person"}


def test_anthropic_request_shape(fake, scene):
    r = anth(fake).detect(DetectRequest(scene, CLASSES, "claude-sonnet-5-5", options={"effort": "high"}))
    assert len(r.detections) == 3
    (call,) = fake.calls("anthropic")
    b = call["body"]
    assert call["auth"]["x-api-key"].startswith("sk-ant-api")
    assert "anthropic-version" in call["header_names"]
    assert "server-side-fallback-2026-07-01" in (call["anthropic_beta"] or "")     # refusal fallback opted in
    assert b["fallbacks"] == "default"
    assert not any(k in b for k in ("temperature", "top_p", "top_k", "thinking"))
    assert b["output_config"]["effort"] == "high"
    assert b["output_config"]["format"]["type"] == "json_schema"
    assert b["output_config"]["format"]["schema"]["properties"]["objects"]["items"]["properties"]["label"]["enum"] == \
        ["person", "car", "fallen person", "dog"]
    img = b["messages"][0]["content"][0]
    assert img["type"] == "image" and img["transformations"] == {"oversized_image": "error"}
    assert b["messages"][0]["content"][1]["text"].startswith("Image size: 1600x1000 px")
    assert call["images"] == [(1600, 1000)]
    assert b["max_tokens"] == 16000
    # cost from usage x price table
    assert r.usage.cost_usd == pytest.approx(vlm.cost_usd("anthropic", "claude-sonnet-5-5", r.usage.input_tokens,
                                                          r.usage.output_tokens))


def test_anthropic_large_image_is_pre_resized_to_claude_size(fake, tmp_path):
    p = str(tmp_path / "big.jpg")
    objs = [(RED, (200, 300, 1200, 2500)), (BLUE, (2500, 1500, 3800, 2900))]
    make_scene(p, (4000, 3000), objs)
    r = anth(fake).detect(DetectRequest(p, CLASSES, "claude-opus-5-5"))
    (call,) = fake.calls("anthropic")
    assert call["images"] == [vlm.anthropic_resized_size(4000, 3000)] == [(2212, 1659)]
    assert_boxes(r.detections, {0: objs[0][1], 1: objs[1][1]}, 4000, 3000)


def test_anthropic_standard_tier_model(fake, tmp_path):
    p = str(tmp_path / "hd.jpg")
    make_scene(p, (1920, 1080), [(GREEN, (300, 200, 900, 1000))])
    r = anth(fake).detect(DetectRequest(p, CLASSES, "claude-sonnet-4-6"))
    (call,) = fake.calls("anthropic")
    assert call["images"] == [(1456, 819)]
    assert "fallbacks" not in call["body"]                     # no fallback on 4.6
    assert_boxes(r.detections, {3: (300, 200, 900, 1000)}, 1920, 1080)


def test_anthropic_oversized_tripwire(fake, scene, monkeypatch):
    # a client that skipped the pre-resize must fail loudly instead of silently skewing every box
    monkeypatch.setattr(AnthropicProvider, "sent_size", lambda self, m, w, h, o: (w, h))
    p = Path(scene).with_name("huge.jpg")
    make_scene(str(p), (5000, 3000), [(RED, (10, 10, 100, 100))])
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(str(p), CLASSES, "claude-opus-5-5"))
    assert ei.value.code == "invalid" and "resize to" in str(ei.value)


def test_anthropic_effort_rules(fake, scene):
    anth(fake).detect(DetectRequest(scene, CLASSES, "claude-haiku-4-5", options={"effort": "high"}))
    anth(fake).detect(DetectRequest(scene, CLASSES, "claude-opus-4-6", options={"effort": "xhigh"}))
    anth(fake).detect(DetectRequest(scene, CLASSES, "claude-opus-5-5"))
    b1, b2, b3 = (c["body"] for c in fake.calls("anthropic"))
    assert "effort" not in b1["output_config"]               # Haiku 4.5 rejects effort
    assert b2["output_config"]["effort"] == "high"           # Opus 4.6 has no xhigh
    assert b3["output_config"]["effort"] == "medium"         # explicit, never the model's implicit default


@pytest.mark.parametrize("orientation", [6, 8, 3])
def test_exif_orientation_is_applied(fake, tmp_path, orientation):
    p = str(tmp_path / f"exif{orientation}.jpg")
    objs = [(RED, (60, 40, 300, 200)), (BLUE, (500, 300, 760, 560))]
    make_scene(p, (800, 600), objs, exif_orientation=orientation)
    r = gem(fake).detect(DetectRequest(p, CLASSES, ""))
    assert_boxes(r.detections, {0: objs[0][1], 1: objs[1][1]}, 800, 600)


@pytest.mark.parametrize("name,mk", ALL)
def test_tiles_offset_back_to_full_image(fake, tmp_path, name, mk):
    p = str(tmp_path / "wide.jpg")
    # small objects far apart (one per quadrant) + one big object crossing the tile borders
    objs = [(RED, (100, 100, 160, 220)), (GREEN, (2800, 150, 2880, 230)), (MAGENTA, (150, 1800, 250, 1900)),
            (BLUE, (1200, 700, 1900, 1300))]
    make_scene(p, (3000, 2000), objs)
    classes = CLASSES + [ClassSpec(4, "bicycle")]
    r = mk(fake).detect(DetectRequest(p, classes, "", options={"tiles": 2}))
    reqs = [c for c in fake.calls() if c["images"]]
    assert len(reqs) == 5                                       # full image + 2x2 tiles
    assert_boxes(r.detections, {0: objs[0][1], 3: objs[1][1], 4: objs[2][1], 1: objs[3][1]}, 3000, 2000,
                 tol_frac=0.006)
    assert r.stats["tiles"] == 4


def test_gemini_request_shape(fake, scene):
    gem(fake).detect(DetectRequest(scene, CLASSES, "gemini-3.8-flash", options={"effort": "minimal"}))
    (call,) = fake.calls("gemini")
    assert call["auth"]["x-goog-api-key"].startswith("AIza")
    assert "key" not in call["query"]                           # never in the URL
    b = call["body"]
    gc = b["generationConfig"]
    assert gc["responseMimeType"] == "application/json" and "responseJsonSchema" in gc
    assert gc["thinkingConfig"] == {"thinkingLevel": "LOW"}     # 3.8 Flash has no MINIMAL
    assert "temperature" not in gc and "topP" not in gc
    part = b["contents"][0]["parts"][0]
    assert part["mediaResolution"] == {"level": "MEDIA_RESOLUTION_HIGH"}
    assert call["images"] == [(1536, 960)]                      # long edge 1536 for media resolution high
    schema = gc["responseJsonSchema"]
    assert schema["properties"]["objects"]["items"]["properties"]["box_2d"]["maxItems"] == 4


def test_gemini_25_uses_thinking_budget(fake, scene):
    r = gem(fake).detect(DetectRequest(scene, CLASSES, "gemini-2.5-flash", options={"effort": "low"}))
    assert len(r.detections) == 3
    b = fake.calls("gemini")[0]["body"]
    assert b["generationConfig"]["thinkingConfig"] == {"thinkingBudget": 0}
    assert "mediaResolution" not in b["contents"][0]["parts"][0]


def test_openai_request_shape_and_flex(fake, scene):
    r = oai(fake, batch_mode=True).detect(DetectRequest(scene, CLASSES, "gpt-6.1-sol", options={"effort": "high"}))
    (call,) = fake.calls("openai")
    b = call["body"]
    assert call["auth"]["authorization"].startswith("Bearer sk-")
    assert b["store"] is False and b["service_tier"] == "flex"
    assert b["text"]["format"]["strict"] is True and b["reasoning"] == {"effort": "high"}
    assert b["input"][0]["content"][0]["detail"] == "high"
    assert "temperature" not in b
    assert call["images"] == [(1600, 1000)]
    assert r.usage.cost_usd == pytest.approx(vlm.cost_usd("openai", "gpt-6.1-sol", r.usage.input_tokens,
                                                          r.usage.output_tokens, batch=True))


def test_openai_flex_billed_by_the_tier_that_served(fake, scene):
    fake.set(flex_full=True)          # the server answered on the default tier: no discount
    r = oai(fake, batch_mode=True).detect(DetectRequest(scene, CLASSES, "gpt-6.1-sol"))
    assert r.usage.cost_usd == pytest.approx(vlm.cost_usd("openai", "gpt-6.1-sol", r.usage.input_tokens,
                                                          r.usage.output_tokens))


@pytest.mark.parametrize("coord,family", [("qwen3", "qwen3"), ("qwen25", "qwen25"), ("gemini", "box_2d"),
                                          ("norm", "norm")])
def test_compat_coordinate_conventions(fake, tmp_path, coord, family):
    p = str(tmp_path / "bus.jpg")
    objs = [(RED, (50, 230, 300, 900)), (BLUE, (400, 100, 780, 400))]
    make_scene(p, (810, 1080), objs)
    fake.set(family=family)
    r = compat(fake, coord=coord).detect(DetectRequest(p, CLASSES, "some-vlm"))
    (call,) = [c for c in fake.calls("compat") if c["images"]]
    if coord == "qwen25":
        sw, sh = call["images"][0]
        assert sw % 28 == 0 and sh % 28 == 0                  # pre-resized onto Qwen2.5's grid
    key = {"qwen3": "bbox_2d", "qwen25": "bbox_2d", "gemini": "box_2d", "norm": "bbox"}[coord]
    schema = call["body"]["response_format"]["json_schema"]["schema"]
    assert key in schema["properties"]["objects"]["items"]["properties"]
    assert_boxes(r.detections, {0: objs[0][1], 1: objs[1][1]}, 810, 1080, tol_frac=0.015)


def test_compat_auto_coord_from_model_name():
    assert coord_from_model("qwen2.5vl:7b") == "qwen25"
    assert coord_from_model("Qwen/Qwen2.5-VL-72B-Instruct") == "qwen25"
    assert coord_from_model("qwen3.8:27b") == "qwen3"
    assert coord_from_model("gemma4:31b") == "gemini"
    assert coord_from_model("llava") == "qwen3"


def test_compat_server_without_structured_output_falls_back_to_prompt_json(fake, scene):
    fake.set(reject_params=["response_format"])
    prov = compat(fake)
    r = prov.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    assert_boxes(r.detections, SCENE_GT, 1600, 1000)
    calls = [c for c in fake.calls("compat") if c["images"]]
    assert "response_format" in calls[0]["body"] and "response_format" not in calls[1]["body"]
    # remembered: the next request goes straight to prompt-only JSON (reply is prose + fenced JSON)
    prov.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    calls = [c for c in fake.calls("compat") if c["images"]]
    assert len(calls) == 3 and "response_format" not in calls[2]["body"]


def test_compat_kind_follows_base_url():
    assert OpenAICompatProvider(config={"base_url": "http://localhost:11434"}).kind == "local"
    assert OpenAICompatProvider(config={"base_url": "http://localhost:11434"}).base_url() == "http://localhost:11434/v1"
    assert OpenAICompatProvider(config={"base_url": "http://127.0.0.1:1234/v1/chat/completions"}).base_url() == \
        "http://127.0.0.1:1234/v1"
    remote = OpenAICompatProvider(config={"base_url": "https://openrouter.ai/api/v1"})
    assert remote.kind == "cloud" and "openrouter.ai" in remote.location
    assert remote.base_url() == "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------------------------- classification

@pytest.mark.parametrize("name,mk", ALL)
def test_classify_numbered_crops(fake, scene, name, mk):
    crops = [Crop(10, 250 / 1600, 410 / 1000, 300 / 1600, 580 / 1000), Crop(11, 1200 / 1600, .7, 600 / 1600, .4),
             Crop(12, 1100 / 1600, .2, 200 / 1600, .2), Crop(13, 0.05, 0.95, 0.04, 0.04)]
    r = mk(fake).classify(ClassifyRequest(scene, crops, CLASSES, ""))
    got = {it.index: (it.cls, it.verdict) for it in r.items}
    assert got == {10: (0, "label"), 11: (1, "label"), 12: (2, "label"), 13: (None, "none")}
    assert all(it.confidence in (0.35, 0.65, 0.9) for it in r.items)
    reqs = [c for c in fake.calls() if c["images"]]
    assert len(reqs) == 1 and len(reqs[0]["images"]) == 4
    assert all(336 <= max(s) <= 448 for s in reqs[0]["images"])
    if name == "gemini":                                   # crops do not need 1120 tokens each
        parts = reqs[0]["body"]["contents"][0]["parts"]
        assert {p["mediaResolution"]["level"] for p in parts if "inlineData" in p} == {"MEDIA_RESOLUTION_MEDIUM"}


def test_classify_chunks_over_20_crops(fake, scene):
    crops = [Crop(i, 250 / 1600, 410 / 1000, 300 / 1600, 580 / 1000) for i in range(23)]
    r = anth(fake).classify(ClassifyRequest(scene, crops, CLASSES, "claude-haiku-5-5"))
    assert [it.index for it in r.items] == list(range(23)) and all(it.cls == 0 for it in r.items)
    reqs = fake.calls("anthropic")
    assert [len(c["images"]) for c in reqs] == [20, 3]
    assert "fallbacks" not in reqs[0]["body"]                  # Haiku 5.5 has no server-side fallback


def test_classify_crop_has_context_and_outline():
    from PIL import Image
    im = Image.new("RGB", (1000, 800), (128, 128, 128))
    data, size = vlm.crop_for_classification(im, 0.5, 0.5, 0.2, 0.25, pad=0.25, edge=448)
    assert size == (336, 336)                              # 200x200 box + 25% each side, upscaled to the minimum edge
    crop = Image.open(__import__("io").BytesIO(data)).convert("RGB")
    r, g, b = crop.getpixel((57, 168))                     # left edge of the outlined box (300 px crop x 336/300)
    assert r > 200 and g < 80
    _, size = vlm.crop_for_classification(im, 0.5, 0.5, 0.6, 0.6, pad=0.25, edge=448)
    assert max(size) == 448                                # big boxes are capped
    _, size = vlm.crop_for_classification(im, 0.5, 0.5, 0.05, 0.1, pad=0.25, edge=448, min_edge=0)
    assert size == (76, 120)                               # upscaling can be turned off (edges rounded outward)


# ---------------------------------------------------------------------------------------------- failures

def test_429_retry_after_then_success(fake, scene, waits):
    fake.push_fault("anthropic", kind="rate_limit", retry_after=2)
    seen = []
    with vlm.wait_listener(lambda s, why: seen.append(s)):
        r = anth(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert len(r.detections) == 3
    assert len(fake.calls("anthropic")) == 2
    assert 2.0 <= sum(waits) <= 2.6 and seen[0] >= 2


@pytest.mark.parametrize("name,mk", ALL)
def test_overloaded_and_5xx_are_retried(fake, scene, waits, name, mk):
    fake.push_fault(name, kind="overloaded")
    fake.push_fault(name, kind="server_error")
    r = mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert len(r.detections) == 3
    assert len([c for c in fake.calls(name) if c["images"]]) == 3
    assert len(waits) >= 2


def test_gemini_retry_info_delay(fake, scene, waits):
    fake.push_fault("gemini", kind="rate_limit", retry_after=4)
    gem(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert 4.0 <= sum(waits) <= 5.1


def test_rate_limit_exhausted_is_not_fatal(fake, scene, waits):
    fake.push_fault("openai", kind="rate_limit", count=5)
    with pytest.raises(ProviderError) as ei:
        oai(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "rate" and not ei.value.fatal
    assert len(fake.calls("openai")) == 5


@pytest.mark.parametrize("name,mk", ALL[:3])
def test_bad_key_aborts_without_retry(fake, scene, waits, name, mk):
    fake.push_fault(name, kind="auth")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "auth" and ei.value.fatal
    assert len(fake.calls(name)) == 1 and waits == []


def test_wrong_key_is_rejected_by_fake(fake, scene):
    with pytest.raises(ProviderError) as ei:
        AnthropicProvider(config={"base_url": fake.url}, key="sk-ant-wrong-key-000000").detect(
            DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "auth" and "sk-ant-wrong" not in str(ei.value)


def test_missing_key_is_auth_error(fake, scene, monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    with pytest.raises(ProviderError) as ei:
        AnthropicProvider(config={"base_url": fake.url}, key="").detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "auth" and "Settings" in str(ei.value)
    assert fake.calls() == []


@pytest.mark.parametrize("name,mk", ALL[:3])
def test_spend_limit_aborts_the_job(fake, scene, waits, name, mk):
    fake.push_fault(name, kind="spend_limit")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "quota" and ei.value.fatal
    assert len(fake.calls(name)) == 1 and waits == []


@pytest.mark.parametrize("name,mk", ALL)
def test_error_text_never_leaks_the_key(fake, scene, name, mk):
    fake.push_fault(name, kind="leaky_error")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    msg = str(ei.value)
    for k in KEYS.values():
        if k:
            assert k not in msg
    assert "LEAKEDleaked" not in msg and "…" in msg


def test_refusal_without_fallback_skips_image(fake, scene):
    fake.push_fault("anthropic", kind="refusal")
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, CLASSES, "claude-haiku-5-5"))
    assert ei.value.code == "refusal" and not ei.value.fatal and "cyber" in str(ei.value)


def test_refusal_rescued_by_server_side_fallback(fake, scene):
    fake.push_fault("anthropic", kind="refusal")
    r = anth(fake).detect(DetectRequest(scene, CLASSES, "claude-opus-5-5"))
    assert r.served_by == "claude-opus-5" and len(r.detections) == 3
    # the declined hop bills at Opus 5.5 rates, the answer at Opus 5 rates
    (call,) = fake.calls("anthropic")
    assert r.usage.cost_usd == pytest.approx(
        vlm.cost_usd("anthropic", "claude-opus-5-5", 100, 0)
        + vlm.cost_usd("anthropic", "claude-opus-5", r.usage.input_tokens - 100, r.usage.output_tokens))


def test_fallback_beta_rejected_by_gateway_is_dropped(fake, scene):
    fake.set(reject_params=["fallback"])
    prov = anth(fake)
    r = prov.detect(DetectRequest(scene, CLASSES, "claude-opus-5-5"))
    assert len(r.detections) == 3
    calls = fake.calls("anthropic")
    assert "fallbacks" in calls[0]["body"] and "fallbacks" not in calls[1]["body"]
    prov.detect(DetectRequest(scene, CLASSES, "claude-opus-5-5"))
    assert "fallbacks" not in fake.calls("anthropic")[2]["body"]


def test_fallback_can_be_turned_off(fake, scene):
    anth(fake).detect(DetectRequest(scene, CLASSES, "claude-opus-5-5", options={"fallback": False}))
    assert "fallbacks" not in fake.calls("anthropic")[0]["body"]


@pytest.mark.parametrize("name,mk,param,model,opts,where", [
    ("anthropic", anth, "effort", "claude-future-9", {"effort": "high"}, lambda b: "effort" in b["output_config"]),
    ("openai", oai, "reasoning", "gpt-4.1", {"effort": "low"}, lambda b: "reasoning" in b),
    ("openai", oai, "service_tier", "gpt-6.1-sol", {"batch_mode": True}, lambda b: "service_tier" in b),
    ("gemini", gem, "mediaResolution", "gemini-3.8-flash", {}, lambda b: "mediaResolution" in b["contents"][0]["parts"][0]),
])
def test_optional_params_a_model_rejects_are_dropped_once(fake, scene, name, mk, param, model, opts, where):
    fake.set(reject_params=[param])
    prov = mk(fake)
    r = prov.detect(DetectRequest(scene, CLASSES, model, options=opts))
    assert_boxes(r.detections, SCENE_GT, 1600, 1000)
    calls = [c for c in fake.calls(name) if c["images"]]
    assert len(calls) == 2 and where(calls[0]["body"]) and not where(calls[1]["body"])
    prov.detect(DetectRequest(scene, CLASSES, model, options=opts))          # remembered: one request
    calls = [c for c in fake.calls(name) if c["images"]]
    assert len(calls) == 3 and not where(calls[2]["body"])
    if param == "mediaResolution":
        assert calls[2]["body"]["generationConfig"]["mediaResolution"] == "MEDIA_RESOLUTION_HIGH"
    if param == "service_tier":                                              # billed at standard prices then
        assert r.usage.cost_usd == pytest.approx(vlm.cost_usd("openai", model, r.usage.input_tokens,
                                                              r.usage.output_tokens))


@pytest.mark.parametrize("name,mk", [("gemini", gem), ("openai", oai)])
def test_refusal_other_providers(fake, scene, name, mk):
    fake.push_fault(name, kind="refusal")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "refusal"


@pytest.mark.parametrize("name,mk", ALL)
def test_truncated_output_retries_with_bigger_budget(fake, scene, name, mk):
    fake.push_fault(name, kind="truncate", keep=1, below=16000)
    r = mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert_boxes(r.detections, SCENE_GT, 1600, 1000)
    calls = [c for c in fake.calls(name) if c["images"]]
    assert len(calls) == 2
    budget = {"anthropic": "max_tokens", "gemini": None, "openai": "max_output_tokens", "compat": "max_tokens"}[name]
    if budget:
        assert calls[0]["body"][budget] == 16000 and calls[1]["body"][budget] == 32000
    else:
        assert calls[1]["body"]["generationConfig"]["maxOutputTokens"] == 32000
    assert r.stats["truncated"] == 1 and r.usage.input_tokens > 0


@pytest.mark.parametrize("name,mk", ALL[:3])
def test_still_truncated_keeps_complete_items(fake, scene, name, mk):
    fake.push_fault(name, kind="truncate", keep=2, count=2)
    r = mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert len(r.detections) == 2
    assert any("cut off" in w for w in r.warnings)


def test_truncated_before_any_item_is_bad_output(fake, scene):
    fake.push_fault("anthropic", kind="truncate", keep=0, count=2)
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "bad_output" and not ei.value.fatal


@pytest.mark.parametrize("name,mk", ALL)
def test_unreadable_output_is_bad_output(fake, scene, name, mk):
    fake.push_fault(name, kind="text", text="I see a person and a car.")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "bad_output"


@pytest.mark.parametrize("name,mk,scale", [("anthropic", anth, "k1000"), ("anthropic", anth, "unit"),
                                           ("gemini", gem, "pixel"), ("openai", oai, "unit"),
                                           ("compat", compat, "pixel")])
def test_wrong_scale_answers_are_rescaled(fake, tmp_path, name, mk, scale):
    # Only an answer that overshoots its frame is detectable: 0-1000 instead of pixels on a small image (values above
    # the pixel range), pixels instead of 0-1000 on a large one (values above 1000), 0-1 fractions anywhere.
    p = str(tmp_path / "s.jpg")
    if scale == "pixel":
        W, H, objs = 1600, 1000, SCENE_OBJS
    else:
        W, H = 640, 480
        objs = [(RED, (40, 60, 200, 400)), (BLUE, (360, 240, 600, 450)), (YELLOW, (400, 40, 480, 120))]
    make_scene(p, (W, H), objs)
    fake.set(scale=scale)
    r = mk(fake).detect(DetectRequest(p, CLASSES, ""))
    tol = 0.012 if scale != "unit" else 0.02
    assert_boxes(r.detections, {i: o[1] for i, o in enumerate(objs)}, W, H, tol_frac=tol)
    assert r.stats["scale_anomalies"] == 1
    assert any("rescaled" in w for w in r.warnings)


def test_unknown_labels_degenerate_and_duplicates(fake, scene):
    fake.set(duplicate=True, extra_items=[{"label": "cat", "box_2d": [10, 10, 100, 100]},
                                          {"label": "Persons", "box_2d": [500, 500, 500, 600]},
                                          {"label": "Persons", "box_2d": [900, 900, 990, 990]},
                                          {"label": "car", "box_2d": [-50, 950, 40, 1003]}])
    r = gem(fake).detect(DetectRequest(scene, CLASSES, ""))
    got = px_boxes(r.detections, 1600, 1000)
    assert len(got[0]) == 2                  # scene person + plural "Persons" (zero-height one dropped)
    assert len(got[1]) == 2                  # scene car + clamped out-of-range car
    x1, y1, x2, y2 = sorted(got[1], key=lambda b: b[0])[-1]
    assert x2 == 1600 and y1 == 0 and y2 == pytest.approx(40, abs=0.5)
    assert r.stats["unknown_labels"] == 1 and r.stats["degenerate"] == 1 and r.stats["duplicates"] == 1
    assert any("outside the class list" in w and "cat" in w for w in r.warnings)


def test_class_prompt_maps_to_dataset_id(fake, scene):
    # an unnamed dataset class with a class-card prompt gets the model's "fallen person" boxes
    classes = [ClassSpec(7, "class_7", prompt="fallen person")]
    r = oai(fake).detect(DetectRequest(scene, classes, ""))
    assert [(d.cls, d.label) for d in r.detections] == [(7, "fallen person")]
    assert any("outside the class list" in w for w in r.warnings)       # person / car were not asked for


def test_max_objects_cap(fake, scene):
    r = anth(fake).detect(DetectRequest(scene, CLASSES, "", options={"max_objects": 2}))
    assert len(r.detections) == 2 and r.stats["capped"] == 1


def test_self_consistency_samples_give_vote_confidence(fake, scene):
    r = gem(fake).detect(DetectRequest(scene, CLASSES, "", conf=0.5, options={"samples": 3}))
    assert len(fake.calls("gemini")) == 3
    assert sorted(d.conf for d in r.detections) == [1.0, 1.0, 1.0]
    assert r.usage.input_tokens == 3 * (1120 + 200)


def test_anthropic_samples_cache_the_image(fake, scene):
    r = anth(fake).detect(DetectRequest(scene, CLASSES, "claude-sonnet-5-5", options={"samples": 2}))
    calls = fake.calls("anthropic")
    assert len(calls) == 2 and all(c["body"]["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
                                   for c in calls)
    assert sorted(d.conf for d in r.detections) == [1.0, 1.0, 1.0]
    anth(fake).detect(DetectRequest(scene, CLASSES, "claude-sonnet-5-5"))
    assert "cache_control" not in fake.calls("anthropic")[2]["body"]["messages"][0]["content"][0]


def test_empty_class_list_is_rejected(fake, scene):
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, [], ""))
    assert ei.value.code == "invalid"


def test_missing_image_is_invalid(fake, tmp_path):
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(str(tmp_path / "nope.jpg"), CLASSES, ""))
    assert ei.value.code == "invalid"


def test_connection_refused_is_network_error(waits):
    prov = GeminiProvider(config={"base_url": "http://127.0.0.1:9"}, key=KEYS["gemini"])
    with pytest.raises(ProviderError) as ei:
        prov.detect(DetectRequest(str(Path(__file__)), CLASSES, ""))     # unreadable image fails first
    assert ei.value.code == "invalid"
    r = prov.test()
    assert r["ok"] is False and "127.0.0.1:9" in r["message"]


def test_local_server_not_running_fails_fast(scene, waits):
    prov = OpenAICompatProvider(config={"base_url": "http://127.0.0.1:9/v1"}, key="")
    t = time.monotonic()
    with pytest.raises(ProviderError) as ei:
        prov.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    assert ei.value.code == "network" and "is the server running" in str(ei.value)
    assert waits == [] and time.monotonic() - t < 5


def test_concurrent_requests_share_the_adaptive_limiter(fake, scene):
    import concurrent.futures as cf
    prov = anth(fake, concurrency=2)
    with cf.ThreadPoolExecutor(6) as ex:
        res = list(ex.map(lambda _: prov.detect(DetectRequest(scene, CLASSES, "")), range(6)))
    assert all(len(r.detections) == 3 for r in res)
    assert vlm.limiter_for(f"anthropic|{fake.url}", 2).active == 0


# ---------------------------------------------------------------------------------------------- models / test / estimate

def test_status_without_key(monkeypatch):
    for k in ("ANTHROPIC_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY", "OPENAI_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    for P in (AnthropicProvider, GeminiProvider, OpenAIProvider):
        st = P(config={}, key="").status()
        assert st["available"] is False and "key" in st["reason"]
    # no key needed for an OpenAI-compatible server — its status is its reachability (see the N4 tests below)
    assert OpenAICompatProvider(config={}, key="").needs_key is False


def test_anthropic_status_without_sdk(monkeypatch):
    import fovea.ai.providers.anthropic as mod
    monkeypatch.setattr(mod, "_sdk_available", lambda: False)
    st = AnthropicProvider(config={}, key="sk-ant-x").status()
    assert st["available"] is False and "pip install anthropic" in st["reason"]


def test_static_models_have_prices_and_one_default():
    for P in (AnthropicProvider, GeminiProvider, OpenAIProvider):
        ms = P(config={}, key="").models()
        assert sum(1 for m in ms if m["default"]) == 1
        assert all(m["price"] and m["price"]["in"] > 0 for m in ms)
        assert all(m["max_edge"] for m in ms)
    ms = AnthropicProvider(config={"default_model": "claude-haiku-5-5"}, key="").models()
    assert [m["id"] for m in ms if m["default"]] == ["claude-haiku-5-5"]
    ms = {m["id"]: m for m in AnthropicProvider(config={}, key="").models()}
    assert ms["claude-sonnet-4-6"]["max_edge"] == 1568 and ms["claude-opus-5-5"]["max_edge"] == 2576


@pytest.mark.parametrize("name,mk,live_only,dropped", [
    ("anthropic", anth, "claude-sonnet-9-9", None),
    ("gemini", gem, "gemini-9.0-ultra", "gemini-embedding-001"),
    ("openai", oai, "gpt-7-preview", "text-embedding-3-large"),
])
def test_live_model_listing_merges_with_price_table(fake, name, mk, live_only, dropped):
    prov = mk(fake)
    ms = {m["id"]: m for m in prov.models(refresh=True)}
    assert live_only in ms and ms[live_only]["price"] is None and "price table" in ms[live_only]["notes"]
    default = vlm.default_model(name)
    assert ms[default]["available"] is True and ms[default]["price"]
    unlisted = {"anthropic": "claude-opus-5", "gemini": "gemini-3.6-flash", "openai": "gpt-6-sol"}[name]
    assert ms[unlisted]["available"] is False
    if dropped:
        assert dropped not in ms
    n = len(fake.calls())
    prov.models()                                                  # cached for 10 minutes
    assert len(fake.calls()) == n


def test_live_listing_failure_is_reported(fake):
    fake.push_fault("anthropic", kind="auth")
    prov = AnthropicProvider(config={"base_url": fake.url}, key="sk-ant-api03-WRONGwrongWRONG")
    with pytest.raises(ProviderError) as ei:
        prov.models(refresh=True)
    assert ei.value.code == "auth"
    assert prov.models()[0]["id"] == "claude-opus-5-5"          # curated list still there


def test_openai_listing_flags_models_bad_at_boxes(fake):
    ms = {m["id"]: m for m in oai(fake).models(refresh=True)}
    assert "gpt-realtime" not in ms and "Poor at boxes" in ms["gpt-5.5"]["notes"]


def test_compat_discovery_keeps_only_vision_models(fake):
    ms = {m["id"]: m for m in compat(fake).models(refresh=True)}
    assert "qwen3.8:27b" in ms and "qwen2.5vl:7b" in ms and "llama3.2:3b" not in ms
    assert ms["qwen2.5vl:7b"]["coord"] == "qwen25" and ms["qwen3.8:27b"]["coord"] == "qwen3"
    assert all(m["price"] is None for m in ms.values())


def test_compat_discovery_without_ollama_endpoints(fake):
    fake.set(reject_params=["tags"], models=["my-vlm", "other"])
    ms = [m["id"] for m in compat(fake).models(refresh=True)]
    assert "my-vlm" in ms and "other" in ms


@pytest.mark.parametrize("name,mk", ALL)
def test_connection_test(fake, name, mk):
    r = mk(fake).test()
    assert r["ok"] is True and "Connected" in r["message"] and r["latency_ms"] >= 0


@pytest.mark.parametrize("P,key", [(AnthropicProvider, "sk-ant-api03-WRONGwrongWRONG"),
                                   (GeminiProvider, "AIzaWRONGwrongWRONGwrongWRONG"),
                                   (OpenAIProvider, "sk-proj-WRONGwrongWRONGwrong")])
def test_connection_test_bad_key(fake, P, key):
    base = fake.url + ("/v1" if P is OpenAIProvider else "")
    r = P(config={"base_url": base}, key=key).test()
    assert r["ok"] is False and key not in r["message"]
    assert "key" in r["message"].lower()


def test_estimate_matches_pricing_and_tiling():
    p = AnthropicProvider(config={}, key="")
    e = p.estimate(640, 640, "claude-sonnet-5-5", 4, {"effort": "low"})
    prompt = p.detect_prompt_tokens(vlm.PIXEL_XYXY, 4, 100, {})
    assert 300 < prompt < 400                                    # the system prompt + user text actually sent
    assert e["tokens_in"][0] == 529 + prompt                      # ceil(640/28)^2 image tokens + prompt
    assert 0 < e["usd"][0] < e["usd"][1]
    assert e["usd_batch"][0] == pytest.approx(e["usd"][0] / 2, rel=1e-3)
    assert e["calls"] == 1 and e["seconds"] > 0
    e2 = p.estimate(6000, 4000, "claude-sonnet-5-5", 4, {"tiles": True})
    assert e2["calls"] == 5 and e2["tokens_in"][0] > 4 * 4000      # 2.3x Claude's 2576 px: full pass + 2x2 tiles
    gp = GeminiProvider(config={}, key="")
    g = gp.estimate(1920, 1080, "gemini-3.8-flash", 2, {})
    assert g["tokens_in"][0] == 1120 + gp.detect_prompt_tokens(vlm.YXYX_1000, 2, 100, {})
    c = OpenAICompatProvider(config={}, key="").estimate(1920, 1080, "qwen3.8:27b", 2, {})
    assert c["usd"] == [0.0, 0.0] and c["usd_batch"] is None and c["priced"] is False
    op = OpenAIProvider(config={}, key="")
    s = op.estimate(1920, 1080, "gpt-6.1-sol", 2, {"samples": 3})
    assert s["calls"] == 3 and s["tokens_in"][0] == 3 * (2448 + op.detect_prompt_tokens(vlm.YXYX_1000, 2, 100, {}))


def test_location_and_kind():
    assert AnthropicProvider(config={}).location == "Images are sent to api.anthropic.com"
    assert "generativelanguage.googleapis.com" in GeminiProvider(config={}).location
    assert AnthropicProvider(config={"base_url": "https://gw.example.com/anthropic"}).location.endswith("gw.example.com")
    assert OpenAICompatProvider(config={}).kind == "local"


def test_base_url_env_override(monkeypatch):
    monkeypatch.setenv("FOVEA_ANTHROPIC_BASE_URL", "http://127.0.0.1:8955")
    assert AnthropicProvider(config={"base_url": ""}).base_url() == "http://127.0.0.1:8955"
    assert AnthropicProvider(config={"base_url": "https://x.example"}).base_url() == "https://x.example"


def test_fake_block_finder_is_exact():
    from PIL import Image
    im = Image.new("RGB", (300, 200), (128, 128, 128))
    from PIL import ImageDraw
    d = ImageDraw.Draw(im)
    d.rectangle([10, 20, 59, 79], fill=RED)
    d.rectangle([100, 20, 149, 79], fill=RED)
    d.rectangle([200, 100, 249, 189], fill=BLUE)
    got = sorted(find_blocks(im))
    assert got == [("car", (200, 100, 250, 190)), ("person", (10, 20, 60, 80)), ("person", (100, 20, 150, 80))]


# ---------------------------------------------------------------------------------------------- through the app

def test_propose_endpoint_end_to_end_with_fake_anthropic(fake, tmp_path, monkeypatch):
    """POST /api/ai/propose with the real Anthropic provider (FOVEA_AI_MOCK unset) pointed at the fake."""
    try:
        from fastapi.testclient import TestClient
        import fovea.main as main
        from fovea.ai import registry
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"app not importable yet: {e}")
    from testkit import route_paths         # walks include_router() on every FastAPI (0.14x: app.routes is lazy)
    if "/api/ai/propose" not in route_paths(main.app):
        pytest.skip("/api/ai/propose not available yet (ai-core)")
    monkeypatch.delenv("FOVEA_AI_MOCK", raising=False)
    monkeypatch.setenv("ANTHROPIC_API_KEY", KEYS["anthropic"])
    monkeypatch.setenv("FOVEA_ANTHROPIC_BASE_URL", fake.url)
    registry.reset()
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    make_scene(str(root / "images" / "a.jpg"), (1600, 1000), SCENE_OBJS)
    (root / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames: [person, car, fallen person, dog]\n")
    c = TestClient(main.app)
    r = c.post("/api/datasets", json={"path": str(root), "name": "cloud-e2e"})
    assert r.status_code in (200, 201), r.text
    ds = r.json().get("id") or r.json().get("dataset", {}).get("id")
    for _ in range(100):
        d = c.get(f"/api/datasets/{ds}").json()
        if (d.get("dataset") or d).get("status") == "ready":
            break
        time.sleep(0.1)
    imgs = c.get(f"/api/datasets/{ds}/images", params={"limit": 5}).json()
    iid = (imgs.get("items") or imgs.get("images"))[0]["id"]
    r = c.post("/api/ai/propose", json={"dataset_id": ds, "image_id": iid, "provider": "anthropic",
                                        "model": "claude-sonnet-5-5"})
    assert r.status_code == 200, r.text
    out = r.json()
    assert out["served_by"] == "claude-sonnet-5-5"
    assert_boxes([type("D", (), p) for p in out["preds"]], SCENE_GT, 1600, 1000)
    assert out["usage"]["cost_usd"] > 0
    assert KEYS["anthropic"] not in json.dumps(out)
    c.delete(f"/api/datasets/{ds}")


# ================================================================================================ wave 2a
# generate_json · provider batch APIs · crop-and-refine · ruler overlay · OpenAI-compatible reachability (N4)

NAME_SCHEMA = {"type": "object", "properties": {
    "classes": {"type": "array", "items": {"type": "object", "properties": {
        "cls": {"type": "integer", "minimum": 0},
        "name": {"type": "string", "pattern": "^[a-z][a-z0-9_]*$", "maxLength": 40},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "note": {"type": "string", "default": ""}},
        "required": ["cls", "name", "confidence"]}},
    "filters": {"type": "object", "additionalProperties": {"type": "string"}}},
    "required": ["classes", "filters"]}
NAME_ANSWER = {"classes": [{"cls": 0, "name": "standing_person", "confidence": 0.9},
                           {"cls": 1, "name": "fallen_person", "confidence": 0.7, "note": "lying on the floor"}],
               "filters": {"split": "train", "cls": "1"}}


def _schema_keywords(s, out=None):
    out = set() if out is None else out
    if isinstance(s, dict):
        out.update(s.keys())
        for v in s.values():
            _schema_keywords(v, out)
    elif isinstance(s, list):
        for v in s:
            _schema_keywords(v, out)
    return out


def _wire_schema(rec):
    b = rec["body"]
    if "output_config" in b:
        return b["output_config"]["format"]["schema"]
    if "generationConfig" in b:
        return b["generationConfig"]["responseJsonSchema"]
    if "text" in b:
        return b["text"]["format"]["schema"]
    return b["response_format"]["json_schema"]["schema"]


@pytest.mark.parametrize("name,mk", ALL)
def test_generate_json_each_provider(fake, scene, name, mk):
    fake.set(generate=NAME_ANSWER)
    p = mk(fake)
    assert p.capabilities["generate"] is True
    data = Path(scene).read_bytes()
    obj, usage = p.generate_json("You name classes.", "Name the classes in these crops.", NAME_SCHEMA, "",
                                 [data, data], {"effort": "low"})
    assert obj == NAME_ANSWER                      # maps decoded back from key/value pairs
    rec = fake.calls(name)[-1]
    assert len(rec["images"]) == 2
    wire = _wire_schema(rec)
    kw = _schema_keywords(wire)
    assert "default" not in kw
    if name == "anthropic":
        assert not kw & {"minimum", "maximum", "maxLength"}
        assert usage.cost_usd > 0
    if name == "openai":                           # strict: every property required, optional ones nullable
        item = wire["properties"]["classes"]["items"]
        assert set(item["required"]) == set(item["properties"])
        assert item["properties"]["note"]["type"] == ["string", "null"]
        assert "service_tier" not in rec["body"]
    if name == "gemini":
        assert "pattern" not in kw and usage.cost_usd > 0
    # the free-form map went out as key/value pairs (strict structured outputs cannot express maps)
    assert wire["properties"]["filters"]["type"] == "array"
    assert json.dumps(rec).find(KEYS.get(name) or "nokey") == -1


@pytest.mark.parametrize("name,mk", ALL)
def test_generate_json_array_root_and_text_only(fake, name, mk):
    fake.set(generate=["a", "b"])
    obj, _u = mk(fake).generate_json("", "List two words.", {"type": "array", "items": {"type": "string"}}, "")
    assert obj == ["a", "b"]
    rec = fake.calls(name)[-1]
    assert rec["images"] == []
    assert _wire_schema(rec)["properties"]["result"]["type"] == "array"


def test_generate_json_synthesised_answer_has_the_schema_shape(fake):
    obj, _u = gem(fake).generate_json("s", "u", NAME_SCHEMA, "gemini-3.5-flash-lite")
    assert set(obj) == {"classes", "filters"} and isinstance(obj["filters"], dict)


@pytest.mark.parametrize("name,mk", ALL)
def test_generate_json_wrong_shape_is_bad_output(fake, name, mk):
    fake.push_fault(name if name != "compat" else "compat", kind="text", text='{"something": "else"}')
    with pytest.raises(ProviderError) as ei:
        mk(fake).generate_json("s", "u", NAME_SCHEMA, "")
    assert ei.value.code == "bad_output" and "missing classes" in str(ei.value)


@pytest.mark.parametrize("name,mk", ALL[:3])
def test_generate_json_refusal_and_retries(fake, name, mk, waits):
    fake.push_fault(name, kind="rate_limit", retry_after=2)
    fake.set(generate=NAME_ANSWER)
    obj, _u = mk(fake).generate_json("s", "u", NAME_SCHEMA, "")
    assert obj == NAME_ANSWER and sum(waits) >= 2
    fake.push_fault(name, kind="refusal", fallback=False)
    with pytest.raises(ProviderError) as ei:
        mk(fake, fallback=False).generate_json("s", "u", NAME_SCHEMA, "")
    assert ei.value.code == "refusal"


def test_generate_json_truncated_twice_is_bad_output(fake):
    p = anth(fake)
    # first answer cut off (max_tokens) → retried with the bigger budget, which answers in full
    fake.set(generate=NAME_ANSWER)
    fake.push_fault("anthropic", kind="truncate", below=6000)
    obj, _u = p.generate_json("s", "u", NAME_SCHEMA, "", options={"max_tokens": 4000, "effort": "low"})
    assert obj == NAME_ANSWER
    bodies = [c["body"]["max_tokens"] for c in fake.calls("anthropic")]
    assert bodies == [4000 + 1200, 2 * (4000 + 1200)]          # answer budget + the thinking allowance of "low"
    fake.push_fault("anthropic", kind="truncate", count=2)
    with pytest.raises(ProviderError) as ei:
        p.generate_json("s", "u", NAME_SCHEMA, "")
    assert ei.value.code == "bad_output"


def test_generate_json_compat_without_structured_output(fake):
    fake.set(generate=NAME_ANSWER, reject_params=["response_format"])
    p = compat(fake)
    obj, _u = p.generate_json("s", "Name them.", NAME_SCHEMA, "qwen3.8:27b")
    assert obj == NAME_ANSWER
    last = fake.calls("compat")[-1]
    assert "response_format" not in last["body"]
    assert "matching this JSON schema" in last["text"]          # the schema travels in the prompt instead


def test_generate_json_reencodes_unusual_or_huge_images(fake, tmp_path):
    from PIL import Image
    big = tmp_path / "big.png"
    Image.new("RGB", (5000, 300), (10, 200, 30)).save(big)
    fake.set(generate={"classes": [], "filters": {}})
    anth(fake).generate_json("s", "u", NAME_SCHEMA, "claude-haiku-5-5", [big.read_bytes()])
    w, h = fake.calls("anthropic")[-1]["images"][0]
    assert max(w, h) <= 2048
    with pytest.raises(ProviderError) as ei:
        anth(fake).generate_json("s", "u", NAME_SCHEMA, "", [b"not an image"])
    assert ei.value.code == "invalid"
    with pytest.raises(ProviderError):
        anth(fake).generate_json("s", "  ", NAME_SCHEMA, "")


def test_schema_dialects_and_map_roundtrip():
    s = {"type": "object", "properties": {"a": {"type": "integer", "minimum": 1, "maximum": 5},
                                          "b": {"type": "string", "const": "x"},
                                          "m": {"type": "object", "additionalProperties": {
                                              "type": "object", "additionalProperties": {"type": "integer"}}},
                                          "l": {"type": "array", "items": {"type": "string"}, "minItems": 3}},
         "required": ["a"]}
    a = vlm.schema_for("anthropic", s)
    assert "minimum" not in a["properties"]["a"] and a["properties"]["l"]["minItems"] == 1
    assert a["additionalProperties"] is False
    o = vlm.schema_for("openai", s, strict=True)
    assert set(o["required"]) == {"a", "b", "m", "l"} and o["properties"]["b"]["type"] == ["string", "null"]
    assert o["properties"]["a"]["minimum"] == 1
    g = vlm.schema_for("gemini", s)
    assert g["properties"]["b"]["enum"] == ["x"] and "const" not in g["properties"]["b"]
    enc, paths = vlm.encode_maps(s)
    assert enc["properties"]["m"]["type"] == "array"
    reply = {"a": 2, "m": [{"key": "x", "value": [{"key": "y", "value": 3}]}], "l": ["p"]}
    assert vlm.decode_maps(reply, paths) == {"a": 2, "m": {"x": {"y": 3}}, "l": ["p"]}
    wrapped, unwrap = vlm.wrap_schema({"type": "array", "items": {"type": "integer"}})
    assert unwrap and wrapped["required"] == ["result"]
    assert vlm.check_json_shape({"x": 1}, {"type": "object", "required": ["y"]}) == "missing y"
    assert vlm.check_json_shape([1], {"type": "object"}).startswith("expected a JSON object")


# ---------------------------------------------------------------------------------------------- provider batches

@pytest.fixture
def batch_dir(tmp_path, monkeypatch):
    d = tmp_path / "ai_batches"
    monkeypatch.setattr(vlm, "_batch_dir", lambda: d)
    return d


BATCHERS = [("anthropic", anth), ("gemini", gem), ("openai", oai)]


def _scenes(tmp_path, n=3):
    out = []
    for i in range(n):
        p = tmp_path / f"b{i}.jpg"
        objs = [(RED, (100 + 40 * i, 120, 400 + 40 * i, 700)), (BLUE, (900, 500, 1500, 900))]
        make_scene(str(p), (1600, 1000), objs)
        out.append((str(p), {0: (100 + 40 * i, 120, 400 + 40 * i, 700), 1: (900, 500, 1500, 900)}))
    return out


@pytest.mark.parametrize("name,mk", BATCHERS)
def test_batch_end_to_end(fake, tmp_path, batch_dir, name, mk):
    fake.set(batch_polls=2)
    p = mk(fake)
    assert p.capabilities["batch_api"] is True
    scenes = _scenes(tmp_path)
    items = [{"custom_id": 500 + i, "request": DetectRequest(path, CLASSES, "", options={"tiles": 2} if i == 2 else {})}
             for i, (path, _gt) in enumerate(scenes)]
    bid = p.batch_submit(items)
    st = p.batch_status(bid)
    assert st["state"] == "running" and st["done"] is False and st["images"] == 3
    assert st["counts"]["total"] == 1 + 1 + 5            # the tiled image is a full pass + 4 tiles
    st = p.batch_status(bid)
    assert st["state"] == "ended" and st["done"] and st["counts"]["succeeded"] == 7
    got = dict(p.batch_results(bid))
    assert sorted(got) == [500, 501, 502]                # the caller's custom ids, original type kept
    sync = mk(fake).detect(DetectRequest(scenes[0][0], CLASSES, ""))
    for i, (_path, gt) in enumerate(scenes):
        res = got[500 + i]
        assert not isinstance(res, Exception), res
        assert_boxes(res.detections, gt, 1600, 1000)
        assert res.usage.cost_usd > 0
    # batch prices: half the synchronous price for the same tokens
    assert got[500].usage.cost_usd == pytest.approx(sync.usage.cost_usd / 2, rel=0.02)
    # the decode context survives a restart (a fresh provider instance) and holds no key
    files = list(batch_dir.glob("*.json"))
    assert len(files) == 1 and (KEYS.get(name) or "x") not in files[0].read_text()
    again = dict(mk(fake).batch_results(bid))
    assert sorted(again) == [500, 501, 502]


@pytest.mark.parametrize("name,mk", BATCHERS)
def test_batch_results_rebuilt_from_items_without_saved_context(fake, tmp_path, batch_dir, name, mk, monkeypatch):
    """ai-core passes the request list again: a lost context file must not lose the (already paid) results."""
    monkeypatch.setattr(type(mk(fake)), "batch_max_requests", 3)          # also split over two provider batches
    scenes = _scenes(tmp_path, 3)
    items = [{"custom_id": str(i), "request": DetectRequest(path, CLASSES, "", options={"ruler": i == 1, "tiles": 2 if i == 2 else None})}
             for i, (path, _gt) in enumerate(scenes)]
    items.insert(1, {"custom_id": "gone", "request": DetectRequest(str(tmp_path / "gone.jpg"), CLASSES, "")})
    p = mk(fake)
    bid = p.batch_submit(items)
    for f in batch_dir.glob("*.json"):
        f.unlink()
    with pytest.raises(ProviderError):
        p.batch_results(bid)
    got = dict(mk(fake).batch_results(bid, items=items))
    assert got["gone"].code == "invalid"
    for i, (_path, gt) in enumerate(scenes):
        assert_boxes(got[str(i)].detections, gt, 1600, 1000)


def test_anthropic_batch_request_shape(fake, tmp_path, batch_dir):
    (path, _gt), = _scenes(tmp_path, 1)
    p = anth(fake)
    p.batch_submit([{"custom_id": "img-1", "request": DetectRequest(path, CLASSES, "claude-sonnet-5-5")}])
    items = [c for c in fake.calls("anthropic") if c.get("batch", "").startswith("msgbatch")]
    assert len(items) == 1
    body = items[0]["body"]
    assert "fallbacks" not in body and body["max_tokens"] == p.max_tokens_retry
    assert body["output_config"]["format"]["type"] == "json_schema"
    img = body["messages"][0]["content"][0]
    assert img["type"] == "image" and img["transformations"] == {"oversized_image": "error"}
    assert "temperature" not in body


def test_batch_per_image_errors(fake, tmp_path, batch_dir):
    scenes = _scenes(tmp_path, 4)
    fake.push_fault("anthropic", kind="refusal", fallback=False)       # item 0
    fake.push_fault("anthropic", kind="overloaded")                    # item 1
    fake.push_fault("anthropic", kind="truncate", keep=1)              # item 2: complete part salvaged
    fake.set(batch_expire=["13"])                                      # item 3 never processed
    items = [{"custom_id": str(10 + i), "image_path": path, "classes": [{"id": 0, "name": "person"},
                                                                      {"id": 1, "name": "car"}], "model": ""}
             for i, (path, _gt) in enumerate(scenes)]
    items.append({"custom_id": "missing", "image_path": str(tmp_path / "gone.jpg"),
                  "classes": [{"id": 0, "name": "person"}], "model": ""})
    p = anth(fake)
    bid = p.batch_submit(items)
    st = p.batch_status(bid)
    assert st["counts"]["expired"] == 1 and st["counts"]["errored"] == 1
    got = dict(p.batch_results(bid))
    assert got["10"].code == "refusal"
    assert got["11"].code == "overloaded" and "overloaded_error" in str(got["11"])
    assert len(got["12"].detections) == 1 and any("cut off" in w for w in got["12"].warnings)
    assert got["13"].code == "expired"
    assert got["missing"].code == "invalid"


@pytest.mark.parametrize("name,mk", BATCHERS)
def test_batch_cancel(fake, tmp_path, batch_dir, name, mk):
    fake.set(batch_polls=5, batch_cancel_keep=1)
    p = mk(fake)
    items = [{"custom_id": i, "request": DetectRequest(path, CLASSES, "")} for i, (path, _gt) in
             enumerate(_scenes(tmp_path, 3))]
    bid = p.batch_submit(items)
    st = p.batch_cancel(bid)
    assert st["state"] in ("canceling", "ended")
    for _ in range(3):
        st = p.batch_status(bid)
    assert st["done"] is True and st["state"] == "ended"
    assert st["counts"]["canceled"] == 2 and st["counts"]["succeeded"] == 1
    got = dict(p.batch_results(bid))
    assert not isinstance(got[0], Exception)              # finished before the cancel: still delivered
    assert got[1].code == "canceled" and got[2].code == "canceled"


def test_batch_split_into_parts_and_partial_failure(fake, tmp_path, batch_dir, monkeypatch):
    monkeypatch.setattr(AnthropicProvider, "batch_max_requests", 2)
    p = anth(fake)
    scenes = _scenes(tmp_path, 3)
    items = [{"custom_id": i, "request": DetectRequest(path, CLASSES, "")} for i, (path, _gt) in enumerate(scenes)]
    bid = p.batch_submit(items)
    assert bid.count(",") == 1                            # two provider batches, one Fovea id
    st = p.batch_status(bid)
    assert st["parts"] == 2 and st["counts"]["total"] == 3 and st["done"]
    assert sorted(dict(p.batch_results(bid))) == [0, 1, 2]
    # the second part fails for good: the first one is canceled instead of left running
    calls_before = len(fake.calls("anthropic"))
    fake.push_fault("anthropic_batch", kind="pass")       # first create passes (a non-HTTP fault is a no-op)
    fake.push_fault("anthropic_batch", kind="auth")
    with pytest.raises(ProviderError) as ei:
        p.batch_submit(items)
    assert ei.value.code == "auth"
    cancels = [c for c in fake.calls("anthropic")[calls_before:] if str(c.get("batch", "")).startswith("/cancel")]
    assert len(cancels) == 1


def test_batch_create_rate_limit_is_retried_and_ids_sanitised(fake, tmp_path, batch_dir, waits):
    (path, gt), = _scenes(tmp_path, 1)
    fake.push_fault("anthropic_batch", kind="rate_limit", retry_after=3)
    p = anth(fake)
    bid = p.batch_submit([{"custom_id": "ds 7/img #5", "request": DetectRequest(path, CLASSES, "")}])
    assert sum(waits) >= 3
    item = [c for c in fake.calls("anthropic") if c.get("batch", "").startswith("msgbatch")]
    assert item
    got = dict(p.batch_results(bid))
    assert list(got) == ["ds 7/img #5"]
    assert_boxes(got["ds 7/img #5"].detections, gt, 1600, 1000)


def test_batch_errors_up_front(fake, tmp_path, batch_dir):
    (path, _gt), = _scenes(tmp_path, 1)
    with pytest.raises(ProviderError) as ei:
        compat(fake).batch_submit([{"custom_id": 1, "request": DetectRequest(path, CLASSES, "")}])
    assert ei.value.code == "invalid"
    with pytest.raises(ProviderError):
        anth(fake).batch_submit([])
    with pytest.raises(ProviderError):
        anth(fake).batch_submit([{"custom_id": 1, "request": DetectRequest(path, CLASSES, "")},
                                 {"custom_id": "1", "request": DetectRequest(path, CLASSES, "")}])
    with pytest.raises(ProviderError) as ei:
        anth(fake).batch_results("msgbatch_unknown")
    assert ei.value.code == "invalid"
    with pytest.raises(ProviderError) as ei:      # every image unusable: nothing is submitted
        anth(fake).batch_submit([{"custom_id": 1, "request": DetectRequest(str(tmp_path / "x.jpg"), CLASSES, "")}])
    assert ei.value.code == "invalid" and not [c for c in fake.calls("anthropic") if c.get("batch") == "create"]
    bad = AnthropicProvider(config={"base_url": fake.url}, key="sk-ant-api03-WRONGwrongWRONG")
    with pytest.raises(ProviderError) as ei:
        bad.batch_submit([{"custom_id": 1, "request": DetectRequest(path, CLASSES, "")}])
    assert ei.value.code == "auth"


def test_gemini_batch_results_from_a_responses_file(fake, tmp_path, batch_dir):
    fake.set(gemini_batch_file=True)
    scenes = _scenes(tmp_path, 2)
    p = gem(fake)
    bid = p.batch_submit([{"custom_id": i, "request": DetectRequest(path, CLASSES, "")}
                          for i, (path, _gt) in enumerate(scenes)])
    assert p.batch_status(bid)["done"]
    got = dict(p.batch_results(bid))
    for i, (_path, gt) in enumerate(scenes):
        assert_boxes(got[i].detections, gt, 1600, 1000)
    assert [c for c in fake.calls("gemini") if ":download" in c["path"]]


def test_openai_batch_file_shape(fake, tmp_path, batch_dir):
    (path, _gt), = _scenes(tmp_path, 1)
    p = oai(fake, batch_mode=True)               # Flex is a synchronous tier: never inside a batch body
    p.batch_submit([{"custom_id": 1, "request": DetectRequest(path, CLASSES, "gpt-6.1-sol")}])
    create = [c for c in fake.calls("openai") if c.get("batch") == "create"][0]["body"]
    assert create["endpoint"] == "/v1/responses" and create["completion_window"] == "24h"
    items = [c for c in fake.calls("openai") if str(c.get("batch", "")).startswith("batch_")]
    assert items and "service_tier" not in items[0]["body"] and items[0]["body"]["store"] is False


def test_batch_refine_option_is_reported_not_run(fake, tmp_path, batch_dir):
    (path, _gt), = _scenes(tmp_path, 1)
    p = anth(fake)
    bid = p.batch_submit([{"custom_id": 1, "request": DetectRequest(path, CLASSES, "", options={"refine": True})}])
    res = dict(p.batch_results(bid))[1]
    assert any("Economy" in w for w in res.warnings)


# ---------------------------------------------------------------------------------------------- crop-and-refine

SMALL_OBJS = [(RED, (700, 300, 760, 380)), (BLUE, (200, 800, 270, 840)), (BLUE, (900, 500, 1500, 900))]


@pytest.fixture
def small_scene(tmp_path):
    p = tmp_path / "small.jpg"
    make_scene(str(p), (1600, 1000), SMALL_OBJS)
    return str(p)


def _px_sorted(dets):
    return sorted((d.cls, round((d.xc - d.w / 2) * 1600), round((d.yc - d.h / 2) * 1000),
                   round((d.xc + d.w / 2) * 1600), round((d.yc + d.h / 2) * 1000)) for d in dets)


@pytest.mark.parametrize("name,mk", ALL)
def test_refine_tightens_small_loose_boxes(fake, small_scene, name, mk):
    fake.set(loose=0.3)
    first = mk(fake).detect(DetectRequest(small_scene, CLASSES, ""))
    n_first = len(fake.calls(name))
    res = mk(fake).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    got = _px_sorted(res.detections)
    want_small = {(0, 700, 300, 760, 380), (1, 200, 800, 270, 840)}
    assert want_small <= {tuple(g) for g in got}, got
    # the big car (> 2 % of the image) is not refined and keeps its first-pass box
    big_first = [b for b in _px_sorted(first.detections) if b[3] - b[1] > 400]
    assert big_first and big_first[0] in [tuple(g) for g in got]
    assert res.stats["refined"] == 2 and res.stats["refine_calls"] == 2
    crops = fake.calls(name)[n_first + 1:]
    assert len(crops) == 2 and all(vlm.REFINE_MARK in c["text"] for c in crops)
    assert all(max(c["images"][0]) >= 700 for c in crops)          # zoomed: small crops are upscaled
    if name != "compat":
        assert res.usage.cost_usd > first.usage.cost_usd           # refine calls are paid for


def test_refine_guards_keep_the_first_box(fake, small_scene):
    fake.set(loose=0.3, refine_answer={"label": "person", "box": [0.1, 0.1, 0.1, 0.1]})   # a box in the corner
    res = anth(fake).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    assert res.stats["refined"] == 0 and res.stats["refine_rejected"] == 2
    first = anth(fake).detect(DetectRequest(small_scene, CLASSES, ""))
    assert _px_sorted(res.detections) == _px_sorted(first.detections)


def test_refine_failures_keep_first_pass(fake, small_scene, waits):
    fake.set(loose=0.3)
    fake.push_fault("anthropic", kind="pass")                  # main pass goes through (a no-op fault)
    fake.push_fault("anthropic", kind="refusal", fallback=False)
    res = anth(fake, fallback=False).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    assert res.stats["refine_failed"] == 1 and res.stats["refined"] == 1 and len(res.detections) == 3
    fake.push_fault("anthropic", kind="pass")
    fake.push_fault("anthropic", kind="spend_limit")           # fatal: stop refining, keep the result, say why
    res = anth(fake).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    assert len(res.detections) == 3 and any("refine pass stopped" in w for w in res.warnings)


def test_refine_selection_settings():
    with pytest.raises(ProviderError):
        vlm.refine_settings({"max_boxes": "many"})
    assert vlm.refine_settings(False) is None and vlm.refine_settings("nope") is None
    cfg = vlm.refine_settings({"max_boxes": 99, "iterations": 7, "context": 9})
    assert cfg["max_boxes"] == 30 and cfg["iterations"] == 2 and cfg["context"] == 4.0
    assert vlm.refine_settings("all")["all"] is True
    old = vlm.Box(0, 0.40, 0.40, 0.44, 0.46)
    region = vlm.refine_region(old, 1000, 1000, 2.5, 256)
    assert region[2] == 256 and region[3] == 256            # min size wins for a tiny box
    assert vlm.refine_accept(old, vlm.Box(0, 0.405, 0.405, 0.44, 0.455), region, 1000, 1000)
    assert not vlm.refine_accept(old, vlm.Box(0, 0.30, 0.30, 0.50, 0.60), region, 1000, 1000)   # area x5
    assert not vlm.refine_accept(old, vlm.Box(0, 0.55, 0.55, 0.59, 0.61), region, 1000, 1000)   # elsewhere
    edge = vlm.refine_region(vlm.Box(0, 0.0, 0.0, 0.02, 0.02), 1000, 800, 2.5, 256)
    assert edge[0] == 0 and edge[1] == 0                    # shifted to stay inside the image


def test_refine_all_and_max_boxes(fake, small_scene):
    res = gem(fake).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": {"all": True, "max_boxes": 2}}))
    assert res.stats["refine_candidates"] == 2


def test_refine_low_agreement_boxes_with_samples(fake, small_scene):
    fake.set(loose=0.0)
    res = oai(fake).detect(DetectRequest(small_scene, CLASSES, "", options={"samples": 2, "refine": {"small": 0.0}}))
    assert res.stats["refine_candidates"] == 0               # every sample agreed: nothing to re-ask


def test_estimate_includes_refine_calls():
    p = AnthropicProvider(config={}, key="x")
    base = p.estimate(1920, 1080, "claude-sonnet-5-5", 4, {})
    ref = p.estimate(1920, 1080, "claude-sonnet-5-5", 4, {"refine": True})
    assert ref["usd"][0] == base["usd"][0] and ref["usd"][1] > base["usd"][1]
    assert ref["refine_calls"] == [0, 8] and ref["usd_batch"] == base["usd_batch"]


# ---------------------------------------------------------------------------------------------- ruler overlay

@pytest.mark.parametrize("name,mk", [("gemini", gem), ("openai", oai), ("compat", compat)])
def test_ruler_margin_is_removed(fake, scene, name, mk):
    res = mk(fake).detect(DetectRequest(scene, CLASSES, "", options={"ruler": True}))
    assert_boxes(res.detections, SCENE_GT, 1600, 1000)
    rec = fake.calls(name)[-1]
    w, h = rec["images"][0]
    plain = mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    pw, ph = fake.calls(name)[-1]["images"][0]
    assert w > pw and h > ph                                   # the margin is added on top of the photo
    assert "ruler" in rec["text"] and "photo occupies x" in rec["text"]
    assert not plain.warnings and not res.warnings


def test_ruler_with_scripted_boxes_and_tiles(fake, scene):
    fake.set(boxes=[{"label": "person", "box": [0.25, 0.5, 0.2, 0.4]}])
    res = gem(fake).detect(DetectRequest(scene, CLASSES, "", options={"ruler": True}))
    d = res.detections[0]
    # the 0-1000 integer frame quantises to 1/1000 of the (slightly larger) canvas
    assert all(abs(a - b) <= 0.0025 for a, b in zip((d.xc, d.yc, d.w, d.h), (0.25, 0.5, 0.2, 0.4))), d
    fake.set(boxes=None)
    res = oai(fake).detect(DetectRequest(scene, CLASSES, "", options={"ruler": True, "tiles": 2}))
    assert_boxes(res.detections, SCENE_GT, 1600, 1000)


def test_ruler_is_skipped_for_pixel_frames(fake, scene):
    res = anth(fake).detect(DetectRequest(scene, CLASSES, "", options={"ruler": True}))
    assert any("ruler" in w for w in res.warnings)
    assert fake.calls("anthropic")[-1]["images"][0] == (1600, 1000)
    assert_boxes(res.detections, SCENE_GT, 1600, 1000)


def test_add_ruler_is_black_and_white_only():
    from PIL import Image
    import numpy as np
    photo = Image.new("RGB", (800, 500), (128, 128, 128))
    canvas, pad = vlm.add_ruler(photo, vlm.YXYX_1000)
    l, t, r, b = pad
    assert r == 0 and b == 0 and canvas.size == (800 + l, 500 + t)
    margin = np.asarray(canvas)[:t, :, :].reshape(-1, 3)
    assert (np.abs(margin.astype(int) - margin.mean(axis=1, keepdims=True)) < 2).all()   # greys only
    assert find_blocks(canvas) == []                          # nothing the fake (or a model) could take for an object
    prep = vlm.Prepared(b"", "image/jpeg", canvas.size[0], canvas.size[1], (0, 0, 800, 500), (800, 500), pad)
    assert prep.photo_size == (800, 500)
    assert vlm.Prepared.from_geometry(prep.geometry()) == prep


def test_estimate_with_ruler_counts_the_margin():
    p = OpenAIProvider(config={}, key="x")
    assert p.estimate(1920, 1080, "gpt-6.1-sol", 3, {"ruler": True})["tokens_in"][0] > \
        p.estimate(1920, 1080, "gpt-6.1-sol", 3, {})["tokens_in"][0]


# ---------------------------------------------------------------------------------------------- N4: reachability

@pytest.fixture
def dead_url():
    import socket
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    return f"http://127.0.0.1:{port}/v1"


@pytest.fixture(autouse=True)
def _fresh_probes():
    OpenAICompatProvider._probes.clear()
    OpenAICompatProvider._probe_threads.clear()
    yield
    OpenAICompatProvider._probes.clear()
    OpenAICompatProvider._probe_threads.clear()


def test_compat_status_reflects_reachability(fake, dead_url):
    up = compat(fake)
    assert up.status() == {"available": True, "reason": None}
    n = len(fake.calls("compat"))
    assert up.status()["available"] is True and len(fake.calls("compat")) == n     # cached: no new request
    down = OpenAICompatProvider(config={"base_url": dead_url}, key="")
    st = down.status()
    assert st["available"] is False and st["reason"] == f"Not reachable at {dead_url[7:-3]}"
    for m in down.models():
        assert m["available"] is False and m["reason"].startswith("Not reachable")


def test_compat_first_listing_waits_for_the_first_probe(fake):
    """The provider list asks models() before status(): neither may answer "checking" for a server that is up."""
    p = compat(fake)
    models = p.models()
    assert p.status() == {"available": True, "reason": None}
    assert {m["id"]: m["available"] for m in models}["qwen3.8:27b"] is True


def test_compat_models_marked_pulled(fake):
    p = compat(fake)
    p.status()
    by_id = {m["id"]: m for m in p.models()}
    assert by_id["qwen3.8:27b"]["available"] is True and "reason" not in by_id["qwen3.8:27b"]
    not_pulled = [m for m in by_id.values() if m["available"] is False]
    assert not_pulled and all("ollama pull" in m["reason"] for m in not_pulled)
    fake.set(reject_params=["tags"])                 # a plain OpenAI-compatible server (no Ollama endpoints)
    OpenAICompatProvider._probes.clear()
    p.status()
    by_id = {m["id"]: m for m in p.models()}
    assert all("Not served by" in m["reason"] for m in by_id.values() if m["available"] is False)


def test_compat_probe_is_cached_and_refreshed_in_the_background(fake, monkeypatch):
    import fovea.ai.providers.openai_compat as mod
    p = compat(fake)
    assert p.status()["available"] is True
    n = len(fake.calls("compat"))
    monkeypatch.setattr(mod, "PROBE_TTL", 0.0)       # stale now: the next status() answers from cache, refreshes
    assert p.status()["available"] is True
    OpenAICompatProvider._probe_threads[p._probe_key()].join(5)
    assert len(fake.calls("compat")) > n


def test_compat_failed_request_flips_status_at_once(fake, dead_url, scene):
    p = OpenAICompatProvider(config={"base_url": dead_url}, key="")
    key = p._probe_key()
    OpenAICompatProvider._probes[key] = {"t": time.time(), "ok": True, "reason": None, "models": None}
    assert p.status()["available"] is True               # stale belief from the cache…
    with pytest.raises(ProviderError):
        p.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    assert p.status()["available"] is False              # …corrected by the failed request, no probe needed


def test_compat_probe_reports_a_rejected_key(fake):
    p = OpenAICompatProvider(config={"base_url": fake.url + "/v1"}, key="sk-proj-WRONGwrongWRONGwrong")
    st = p.status()                                     # the fake treats sk-proj-* bearer as a bad OpenAI key
    assert st["available"] is False and "rejected the API key" in st["reason"]
    assert "WRONG" not in json.dumps(st)


def test_capability_flags():
    assert AnthropicProvider.capabilities["batch_api"] and GeminiProvider.capabilities["batch_api"]
    assert OpenAIProvider.capabilities["batch_api"] and not OpenAICompatProvider.capabilities["batch_api"]
    assert all(P.capabilities["generate"] for P in (AnthropicProvider, GeminiProvider, OpenAIProvider,
                                                     OpenAICompatProvider))


# ================================================================================================ review fixes (r1)
# is_loopback_url decides openai_compat's kind (local/cloud), so the remote-AI guard and the "Images stay on this
# machine" line; a core-only install (no httpx) must say "not installed", never fail with HTTP 500.

@pytest.mark.parametrize("url,local", [
    ("http://localhost:11434/v1", True), ("http://LOCALHOST:11434", True), ("http://ollama.localhost/v1", True),
    ("http://127.0.0.1:1234/v1", True), ("http://127.0.0.2/v1", True), ("http://127.255.255.254:8000", True),
    ("http://[::1]:8000/v1", True), ("http://[::ffff:127.0.0.1]:8000/v1", True), ("http://0.0.0.0:11434", True),
    ("http://127.evil.example/v1", False), ("http://127.0.0.1.nip.io/v1", False), ("http://127.attacker.example:18934/v1", False),
    ("http://localhost.evil.example/v1", False), ("http://evil.example/localhost", False), ("http://localhostx/v1", False),
    ("http://[::ffff:10.0.0.5]/v1", False), ("http://192.168.1.20:11434/v1", False), ("http://10.0.0.1/v1", False),
    ("https://openrouter.ai/api/v1", False), ("http://user:pw@127.0.0.1.evil.example/v1", False),
    ("http://127.0.0.1@evil.example/v1", False), ("", False), ("not a url", False), ("http://[::1/v1", False),
])
def test_is_loopback_url_parses_the_host(url, local):
    assert vlm.is_loopback_url(url) is local


def test_compat_kind_is_cloud_for_lookalike_hosts():
    for url in ("http://127.attacker.example:18934/v1", "http://127.0.0.1.nip.io/v1"):
        p = OpenAICompatProvider(config={"base_url": url})
        assert p.kind == "cloud", url
        assert p.location.startswith("Images are sent to ") and "stay on this machine" not in p.location
    p = OpenAICompatProvider(config={"base_url": "http://127.0.0.2:8000/v1"})
    assert p.kind == "local" and p.location == "Images stay on this machine (127.0.0.2:8000)"


def _app():
    try:
        from fastapi.testclient import TestClient
        import fovea.main as main
        from fovea.ai import registry
    except Exception as e:  # noqa: BLE001
        pytest.skip(f"app not importable: {e}")
    return TestClient, main, registry


@pytest.fixture
def app_ds(tmp_path, monkeypatch):
    """A one-image dataset through the real app; restores openai_compat's config afterwards."""
    TestClient, main, registry = _app()
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    monkeypatch.delenv("FOVEA_AI_MOCK", raising=False)
    registry.reset()
    c = TestClient(main.app)
    before = c.get("/api/ai/providers").json()
    saved = {p["id"]: dict(p.get("config") or {}) for p in before["providers"]}
    root = tmp_path / "ds"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    make_scene(str(root / "images" / "a.jpg"), (640, 400), [(RED, (40, 50, 160, 280))])
    (root / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames: [person, car]\n")
    r = c.post("/api/datasets", json={"path": str(root), "name": "r1-ai-cloud"})
    assert r.status_code in (200, 201), r.text
    ds = r.json().get("id") or r.json().get("dataset", {}).get("id")
    for _ in range(200):
        d = c.get(f"/api/datasets/{ds}").json()
        if (d.get("dataset") or d).get("status") == "ready":
            break
        time.sleep(0.05)
    imgs = c.get(f"/api/datasets/{ds}/images", params={"limit": 5}).json()
    iid = (imgs.get("items") or imgs.get("images"))[0]["id"]
    from testkit import client_from         # TestClient(client=...) needs Starlette 0.42+
    yield c, client_from(main.app, ("192.168.1.50", 50000)), ds, iid
    for pid in ("openai_compat", "gemini"):
        if pid in saved:
            c.put(f"/api/ai/providers/{pid}/config", json={"base_url": saved[pid].get("base_url") or ""})
    c.delete(f"/api/datasets/{ds}")
    registry.reset()


def _compat_entry(c):
    return next(p for p in c.get("/api/ai/providers").json()["providers"] if p["id"] == "openai_compat")


def test_remote_client_cannot_point_compat_at_a_lookalike_host(fake, app_ds, monkeypatch):
    """The review's exploit: a LAN client sets base_url=http://127.attacker.example/v1 (its own DNS name), the
    provider calls itself local, and propose / batch ship every image and the compat key to that host."""
    import socket
    c, remote, ds, iid = app_ds
    port = int(fake.url.rsplit(":", 1)[1])
    real = socket.getaddrinfo

    def fake_dns(host, *a, **kw):            # the attacker's DNS: *.attacker.example → the fake sink
        if isinstance(host, str) and host.endswith(".attacker.example"):
            host = "127.0.0.1"
        return real(host, *a, **kw)

    monkeypatch.setattr(socket, "getaddrinfo", fake_dns)
    monkeypatch.setenv("OPENAI_COMPAT_API_KEY", "compatENVsecretVALUE77")
    evil = f"http://127.attacker.example:{port}/v1"
    assert c.put("/api/ai/providers/openai_compat/config", json={"base_url": fake.url + "/v1"}).status_code == 200
    assert _compat_entry(c)["kind"] == "local"
    # 1. a remote client may not re-point a local provider at another host — lookalike or plain
    for url in (evil, "http://evil.attacker.example/v1", ""):
        r = remote.put("/api/ai/providers/openai_compat/config", json={"base_url": url})
        assert r.status_code == 403, (url, r.text)
    assert _compat_entry(c)["config"]["base_url"] == fake.url + "/v1"
    # 2. set from this machine, the lookalike host is what it is: cloud, and the privacy line says where images go
    r = c.put("/api/ai/providers/openai_compat/config", json={"base_url": evil})
    assert r.status_code == 200, r.text
    e = r.json()
    assert e["kind"] == "cloud" and e["location"] == f"Images are sent to 127.attacker.example:{port}"
    from fovea.ai import registry
    assert registry.is_cloud("openai_compat") is True
    # 3. so a remote client cannot use it
    n = len(fake.calls(path="chat/completions"))
    r = remote.post("/api/ai/propose", json={"dataset_id": ds, "image_id": iid, "provider": "openai_compat",
                                             "model": "qwen3.8:27b"})
    assert r.status_code == 403, r.text
    r = remote.post("/api/ai/batch", json={"dataset_id": ds, "filters": {}, "provider": "openai_compat",
                                           "model": "qwen3.8:27b", "target": "proposals"})
    assert r.status_code == 403, r.text
    assert len(fake.calls(path="chat/completions")) == n            # nothing reached the attacker's host
    r = remote.put("/api/ai/providers/openai_compat/config", json={"base_url": fake.url + "/v1"})
    assert r.status_code == 403                                     # cloud now: its url is key-level config


def _block(monkeypatch, *mods):
    """Make ``import <mod>`` fail the way a core-only install does (find_spec → None, import → ImportError)."""
    for m in mods:
        monkeypatch.setitem(sys.modules, m, None)


def test_without_httpx_providers_say_not_installed(fake, scene, monkeypatch):
    _block(monkeypatch, "httpx")
    assert vlm.httpx_available() is False
    for p in (gem(fake), oai(fake), compat(fake), anth(fake)):
        st = p.status()
        assert st == {"available": False, "reason": vlm.HTTPX_MISSING}, (p.id, st)
        t = p.test()
        assert t["ok"] is False and t["message"] == vlm.HTTPX_MISSING, (p.id, t)
    for m in compat(fake).models():
        assert m["available"] is False and m["reason"] == vlm.HTTPX_MISSING
    assert "ModuleNotFoundError" not in json.dumps([compat(fake).status(), compat(fake).models()])
    for mk in (gem, oai, compat):
        with pytest.raises(ProviderError) as ei:
            mk(fake).detect(DetectRequest(scene, CLASSES, ""))
        assert ei.value.code == "unavailable" and str(ei.value) == vlm.HTTPX_MISSING and ei.value.fatal
        with pytest.raises(ProviderError) as ei:
            mk(fake).generate_json("s", "u", NAME_SCHEMA, "")
        assert ei.value.code == "unavailable"
    assert fake.calls() == []                                       # nothing was sent


def test_without_the_anthropic_sdk_calls_fail_cleanly(fake, scene, monkeypatch):
    _block(monkeypatch, "anthropic")
    from fovea.ai.providers import anthropic as mod
    p = anth(fake)
    assert p.status() == {"available": False, "reason": mod.SDK_MISSING}
    assert p.test()["message"] == mod.SDK_MISSING
    for call in (lambda: p.detect(DetectRequest(scene, CLASSES, "")), lambda: p.list_remote_models(),
                 lambda: p.generate_json("s", "u", NAME_SCHEMA, "")):
        with pytest.raises(ProviderError) as ei:
            call()
        assert ei.value.code == "unavailable" and str(ei.value) == mod.SDK_MISSING


def test_without_httpx_the_api_answers_not_installed(fake, app_ds, monkeypatch):
    """GET /providers says why; propose / test / nl-filter answer with a clear 4xx/5xx code, never 500 'internal'."""
    c, _remote, ds, iid = app_ds
    assert c.put("/api/ai/providers/gemini/config", json={"base_url": fake.url}).status_code == 200
    monkeypatch.setenv("GEMINI_API_KEY", KEYS["gemini"])         # env key: nothing written to secrets.json
    _block(monkeypatch, "httpx")
    by_id = {p["id"]: p for p in c.get("/api/ai/providers").json()["providers"]}
    for pid in ("gemini", "openai", "openai_compat"):
        assert by_id[pid]["status"] == {"available": False, "reason": vlm.HTTPX_MISSING}, (pid, by_id[pid]["status"])
    r = c.post("/api/ai/providers/gemini/test")
    assert r.status_code == 200 and r.json()["ok"] is False and r.json()["message"] == vlm.HTTPX_MISSING
    r = c.post("/api/ai/propose", json={"dataset_id": ds, "image_id": iid, "provider": "gemini",
                                        "model": "gemini-3.8-flash"})
    assert r.status_code == 503, r.text
    assert r.json()["code"] == "unavailable" and r.json()["detail"].endswith(vlm.HTTPX_MISSING)
    r = c.post("/api/ai/nl-filter", json={"dataset_id": ds, "text": "images with a person"})
    assert r.status_code in (400, 503) and r.json().get("code") == "unavailable", r.text
    assert "ModuleNotFoundError" not in r.text
    assert fake.calls() == []


# ================================================================================================ review fixes (r2)
# 1. A gateway URL with ``user:password@`` (FOVEA_<PROVIDER>_BASE_URL) keeps working — httpx sends it as basic auth —
#    but never reaches a client: provider location, status and model reasons, errors and saved batch files show
#    host[:port] only, and every scrubber removes URL credentials.
# 2. Failed calls the provider billed anyway (refusals, unreadable or cut-off output, earlier tiles / chunks / committee
#    members) carry their usage on the error and are booked: run cost, usage totals, budgets and the 30-day caps.
# 3. OpenAI's Flex tier bills synchronous requests at batch prices: estimates (and so budget reservations) say so, and
#    Economy is not offered as a further 50 % saving.

GW_USER, GW_SECRET = "gwuser", "GatewayPw9xq7"


def _no_creds(*things):
    blob = json.dumps(things, default=str)
    assert GW_SECRET not in blob and GW_USER not in blob, blob[:600]


@pytest.mark.parametrize("url,host", [
    (f"http://{GW_USER}:{GW_SECRET}@127.0.0.1:8997/v1", "127.0.0.1:8997"),
    (f"https://{GW_USER}:{GW_SECRET}@Gateway.Example/v1", "gateway.example"),
    (f"http://{GW_USER}@[::1]:8000/v1", "[::1]:8000"),
    (f"http://:{GW_SECRET}@localhost:11434", "localhost:11434"),
    ("https://api.openai.com/v1", "api.openai.com"),
    ("http://127.0.0.2:8000/v1", "127.0.0.2:8000"),
])
def test_url_host_and_redact_url_drop_credentials(url, host):
    assert vlm.url_host(url) == host
    red = vlm.redact_url(url)
    assert "@" not in red and vlm.url_host(red) == host
    assert red.endswith("/v1") == url.endswith("/v1")
    _no_creds(vlm.url_host(url), red)


def test_scrubbers_remove_credentials_inside_urls():
    from fovea import security
    text = f"Cannot reach http://{GW_USER}:{GW_SECRET}@10.0.0.5:8080/v1/chat/completions (timeout)"
    for s in (vlm.scrub(text), security.scrub(text)):
        _no_creds(s)
        assert "http://…@10.0.0.5:8080/v1/chat/completions" in s
    plain = "see https://example.com/a@b and mailto:x@y.z"
    assert vlm.scrub(plain) == plain and security.scrub(plain) == plain


def test_env_gateway_url_credentials_stay_server_side(fake, scene, dead_url, monkeypatch):
    port = fake.url.rsplit(":", 1)[1]
    monkeypatch.setenv("FOVEA_OPENAI_COMPAT_BASE_URL", f"http://{GW_USER}:{GW_SECRET}@127.0.0.1:{port}/v1")
    p = OpenAICompatProvider(config={"base_url": ""}, key="")
    assert p.location == f"Images stay on this machine (127.0.0.1:{port})"
    fake.set(models=["qwen3.8:27b"])
    st, models = p.status(), p.models()
    assert st["available"] is True
    assert any(m.get("reason", "").startswith(f"Not pulled on 127.0.0.1:{port}") for m in models)
    r = p.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    assert r.detections                                  # the gateway still authenticates the request…
    sent = fake.calls("compat", "chat/completions")[-1]["auth"].get("authorization") or ""
    assert sent.startswith("Basic ")                     # …with the URL's user:password as basic auth
    _no_creds(p.location, st, models, p.test(), r.warnings)
    # a gateway that does not answer: every reason and error names the host only
    dport = dead_url.split(":")[2].split("/")[0]
    monkeypatch.setenv("FOVEA_OPENAI_COMPAT_BASE_URL", f"http://{GW_USER}:{GW_SECRET}@127.0.0.1:{dport}/v1")
    down = OpenAICompatProvider(config={"base_url": ""}, key="")
    st = down.status()
    with pytest.raises(ProviderError) as ei:
        down.detect(DetectRequest(scene, CLASSES, "qwen3.8:27b"))
    assert st["available"] is False and st["reason"] == f"Not reachable at 127.0.0.1:{dport}"
    assert f"127.0.0.1:{dport}" in str(ei.value)
    _no_creds(down.location, st, down.models(), down.test(), str(ei.value))


def test_env_gateway_url_credentials_are_not_saved_with_a_batch(fake, tmp_path, batch_dir, monkeypatch):
    port = fake.url.rsplit(":", 1)[1]
    p = AnthropicProvider(config={"base_url": f"http://{GW_USER}:{GW_SECRET}@127.0.0.1:{port}"}, key=KEYS["anthropic"])
    assert p.location == f"Images are sent to 127.0.0.1:{port}"
    (path, _gt), = _scenes(tmp_path, 1)
    bid = p.batch_submit([{"custom_id": "7", "request": DetectRequest(path, CLASSES, "claude-haiku-5-5")}])
    saved = [f.read_text() for f in batch_dir.glob("*.json")]
    assert saved and json.loads(saved[0])["base_url"] == f"http://127.0.0.1:{port}"
    _no_creds(saved)
    assert dict(p.batch_results(bid))["7"].detections


def test_providers_api_never_shows_gateway_credentials(fake, app_ds, monkeypatch):
    c, remote, _ds, _iid = app_ds
    port = fake.url.rsplit(":", 1)[1]
    assert c.put("/api/ai/providers/openai_compat/config", json={"base_url": ""}).status_code == 200
    monkeypatch.setenv("FOVEA_OPENAI_COMPAT_BASE_URL", f"http://{GW_USER}:{GW_SECRET}@127.0.0.1:{port}/v1")
    OpenAICompatProvider._probes.clear()
    for client in (c, remote):                           # the listing is open to every client
        r = client.get("/api/ai/providers")
        assert r.status_code == 200
        _no_creds(r.text)
    e = _compat_entry(c)
    assert e["location"] == f"Images stay on this machine (127.0.0.1:{port})"
    # what Settings shows under an empty URL field: the environment's gateway, without its credentials
    assert e["endpoint"] == {"url": f"http://127.0.0.1:{port}/v1", "source": "env", "env": "FOVEA_OPENAI_COMPAT_BASE_URL"}
    r = c.post("/api/ai/providers/openai_compat/test")
    assert r.json()["ok"] is True and f"127.0.0.1:{port}" in r.json()["message"]
    _no_creds(r.text, c.get("/api/ai/providers/openai_compat/models?refresh=1").text)



def test_providers_api_says_where_an_empty_url_field_points(app_ds, monkeypatch):
    """Settings showed an empty Server URL with the Ollama placeholder even when FOVEA_OPENAI_COMPAT_BASE_URL sent
    the images elsewhere: the listing names the address in use and where it came from."""
    c, _remote, _ds, _iid = app_ds
    monkeypatch.delenv("FOVEA_OPENAI_COMPAT_BASE_URL", raising=False)
    assert c.put("/api/ai/providers/openai_compat/config", json={"base_url": ""}).status_code == 200
    OpenAICompatProvider._probes.clear()
    assert _compat_entry(c)["endpoint"] == {"url": "http://localhost:11434/v1", "source": "default",
                                            "env": "FOVEA_OPENAI_COMPAT_BASE_URL"}
    monkeypatch.setenv("FOVEA_OPENAI_COMPAT_BASE_URL", "http://127.0.0.1:9/v1?token=abc#frag")   # nothing listens
    e = _compat_entry(c)
    assert e["endpoint"] == {"url": "http://127.0.0.1:9/v1", "source": "env", "env": "FOVEA_OPENAI_COMPAT_BASE_URL"}
    assert "abc" not in json.dumps(e["endpoint"])
    assert c.put("/api/ai/providers/openai_compat/config", json={"base_url": "http://127.0.0.1:9/v1"}).status_code == 200
    assert _compat_entry(c)["endpoint"]["source"] == "settings"
    # local providers have no endpoint
    assert next(p for p in c.get("/api/ai/providers").json()["providers"] if p["id"] == "yolo")["endpoint"] is None

# ---- 2. billed failures carry their usage

def _plain(u):
    return int(u.input_tokens), int(u.output_tokens), round(float(u.cost_usd), 8)


def test_cut_off_twice_bills_both_calls(fake, scene):
    """The finding's case: two cut-off Opus answers (16k, then 32k budget) end in bad_output — both were billed."""
    fake.set(thinking_tokens=6000)
    fake.push_fault("anthropic", kind="truncate", keep=0, count=2)
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, CLASSES, "claude-opus-5-5"))
    e = ei.value
    assert e.code == "bad_output" and len(fake.calls("anthropic", "/v1/messages")) == 2
    assert e.usage is not None and e.usage.output_tokens >= 2 * 6000
    assert e.usage.cost_usd == pytest.approx(vlm.cost_usd("anthropic", "claude-opus-5-5", e.usage.input_tokens,
                                                          e.usage.output_tokens), rel=1e-6)
    assert e.usage.by_provider == {"anthropic": list(_plain(e.usage))}     # attributed for the usage ledger


REFUSAL_USAGE = {"anthropic": (120, 0), "gemini": (1200, 0), "openai": (900, 8)}


@pytest.mark.parametrize("name,mk", ALL[:3])
def test_refusal_carries_its_billed_usage(fake, scene, name, mk):
    fake.push_fault(name, kind="refusal", fallback=False)
    p = mk(fake, fallback=False)
    with pytest.raises(ProviderError) as ei:
        p.detect(DetectRequest(scene, CLASSES, ""))
    e = ei.value
    tin, tout = REFUSAL_USAGE[name]
    assert e.code == "refusal" and (e.usage.input_tokens, e.usage.output_tokens) == (tin, tout)
    assert e.usage.cost_usd == pytest.approx(vlm.cost_usd(p.id, p.model_id(""), tin, tout), rel=1e-6) and e.usage.cost_usd > 0
    assert set(e.usage.by_provider) == {p.id}


@pytest.mark.parametrize("name,mk", ALL)
def test_unreadable_output_carries_its_billed_usage(fake, scene, name, mk):
    fake.push_fault(name, kind="text", text="I see a person and a car but cannot draw boxes.")
    with pytest.raises(ProviderError) as ei:
        mk(fake).detect(DetectRequest(scene, CLASSES, ""))
    assert ei.value.code == "bad_output"
    assert ei.value.usage is not None and ei.value.usage.input_tokens > 0 and ei.value.usage.output_tokens > 0


def test_failure_on_a_later_tile_bills_the_earlier_calls(fake, scene):
    fake.push_fault("anthropic", kind="refusal", fallback=False)
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, CLASSES, "claude-haiku-5-5"))
    alone = _plain(ei.value.usage)
    assert alone[:2] == (120, 0)
    fake.reset()
    fake.push_fault("anthropic", kind="pass")                          # the full image answers…
    fake.push_fault("anthropic", kind="refusal", fallback=False)       # …the first tile is refused
    with pytest.raises(ProviderError) as ei:
        anth(fake).detect(DetectRequest(scene, CLASSES, "claude-haiku-5-5", options={"tiles": 2}))
    assert len(fake.calls("anthropic", "/v1/messages")) == 2
    u = ei.value.usage
    assert u.input_tokens > 120 and u.output_tokens > 0 and u.cost_usd > alone[2]


def test_failure_on_a_later_crop_chunk_bills_the_earlier_chunks(fake, scene):
    crops = [Crop(i, 250 / 1600, 410 / 1000, 300 / 1600, 580 / 1000) for i in range(23)]
    fake.push_fault("anthropic", kind="pass")
    fake.push_fault("anthropic", kind="text", text="nope")
    with pytest.raises(ProviderError) as ei:
        anth(fake).classify(ClassifyRequest(scene, crops, CLASSES, "claude-haiku-5-5"))
    assert [len(c["images"]) for c in fake.calls("anthropic")] == [20, 3]
    u = ei.value.usage
    assert ei.value.code == "bad_output" and u.input_tokens > 20 * 100 and set(u.by_provider) == {"anthropic"}


def test_refine_failures_are_billed_with_the_first_pass(fake, small_scene):
    fake.set(loose=0.3)
    first = anth(fake, fallback=False).detect(DetectRequest(small_scene, CLASSES, ""))
    fake.push_fault("anthropic", kind="pass")                          # first pass answers…
    fake.push_fault("anthropic", kind="refusal", fallback=False, count=2)   # …both refine crops are refused
    res = anth(fake, fallback=False).detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    assert res.stats["refine_failed"] == 2 and res.stats["refined"] == 0
    assert _px_sorted(res.detections) == _px_sorted(first.detections)
    assert (res.usage.input_tokens, res.usage.output_tokens) == (first.usage.input_tokens + 2 * 120,
                                                                  first.usage.output_tokens)
    assert res.usage.cost_usd == pytest.approx(first.usage.cost_usd + 2 * vlm.cost_usd(
        "anthropic", anth(fake).model_id(""), 120, 0), rel=1e-6)


@pytest.mark.parametrize("name,mk", ALL)
def test_generate_json_wrong_shape_and_refusal_carry_usage(fake, name, mk):
    fake.push_fault(name, kind="text", text='{"something": "else"}')
    with pytest.raises(ProviderError) as ei:
        mk(fake).generate_json("s", "u", NAME_SCHEMA, "")
    assert ei.value.code == "bad_output" and ei.value.usage is not None and ei.value.usage.input_tokens > 0
    if name != "compat":
        fake.push_fault(name, kind="refusal", fallback=False)
        with pytest.raises(ProviderError) as ei:
            mk(fake, fallback=False).generate_json("s", "u", NAME_SCHEMA, "")
        assert ei.value.code == "refusal" and ei.value.usage.input_tokens == REFUSAL_USAGE[name][0]


def test_batch_results_errors_carry_usage_at_the_batch_price(fake, tmp_path, batch_dir):
    scenes = _scenes(tmp_path, 3)
    fake.push_fault("anthropic", kind="refusal", fallback=False)       # item 0: refused (billed)
    fake.push_fault("anthropic", kind="text", text="no json here")     # item 1: unreadable (billed)
    fake.push_fault("anthropic", kind="overloaded")                    # item 2: errored at the provider (not billed)
    p = anth(fake)
    items = [{"custom_id": str(20 + i), "request": DetectRequest(path, CLASSES, "claude-haiku-5-5")}
             for i, (path, _gt) in enumerate(scenes)]
    got = dict(p.batch_results(p.batch_submit(items)))
    ref, bad, over = got["20"], got["21"], got["22"]
    assert ref.code == "refusal" and (ref.usage.input_tokens, ref.usage.output_tokens) == (120, 0)
    assert ref.usage.cost_usd == pytest.approx(vlm.cost_usd("anthropic", "claude-haiku-5-5", 120, 0, batch=True))
    assert bad.code == "bad_output" and bad.usage.output_tokens > 0 and set(bad.usage.by_provider) == {"anthropic"}
    assert over.code == "overloaded" and over.usage is None


# ---- 2b. … and booked by every caller (through the real app)

@pytest.fixture
def app_cloud(tmp_path, monkeypatch, fake):
    """Four 640x400 images through the real app, Claude / OpenAI pointed at the fake BEFORE any key or call. The
    providers' saved settings are restored afterwards, also when the setup itself fails."""
    TestClient, main, registry = _app()
    monkeypatch.delenv("FOVEA_ALLOW_REMOTE_AI", raising=False)
    monkeypatch.delenv("FOVEA_AI_MOCK", raising=False)
    registry.reset()
    c = TestClient(main.app)
    saved = {p["id"]: dict(p.get("config") or {}) for p in c.get("/api/ai/providers").json()["providers"]}
    ds = None
    try:
        for pid, url in (("anthropic", fake.url), ("openai", fake.url + "/v1")):
            assert c.put(f"/api/ai/providers/{pid}/config", json={"base_url": url}).status_code == 200
        monkeypatch.setenv("ANTHROPIC_API_KEY", KEYS["anthropic"])      # env keys: nothing written to secrets.json
        monkeypatch.setenv("OPENAI_API_KEY", KEYS["openai"])
        root = tmp_path / "ds4"
        (root / "images").mkdir(parents=True)
        (root / "labels").mkdir()
        for i in range(4):
            make_scene(str(root / "images" / f"f{i}.jpg"), (640, 400), [(RED, (40 + 10 * i, 50, 160 + 10 * i, 280))])
            (root / "labels" / f"f{i}.txt").write_text(f"0 {(100 + 10 * i) / 640:.6f} 0.412500 0.187500 0.575000\n")
        (root / "data.yaml").write_text("path: .\ntrain: images\nnames: [person, car]\n")
        r = c.post("/api/datasets", json={"path": str(root), "name": "r2-ai-cloud"})
        assert r.status_code in (200, 201), r.text
        ds = r.json().get("id") or r.json().get("dataset", {}).get("id")
        for _ in range(200):
            d = c.get(f"/api/datasets/{ds}").json()
            if (d.get("dataset") or d).get("status") == "ready":
                break
            time.sleep(0.05)
        imgs = c.get(f"/api/datasets/{ds}/images", params={"limit": 10}).json()
        ids = sorted(i["id"] for i in (imgs.get("items") or imgs.get("images")))
        assert len(ids) == 4
        yield c, ds, ids
    finally:
        for pid in ("anthropic", "openai"):
            cfg = saved.get(pid) or {}
            c.put(f"/api/ai/providers/{pid}/config", json={"base_url": cfg.get("base_url") or "",
                                                           "batch_mode": bool(cfg.get("batch_mode")),
                                                           "spend_cap_usd": cfg.get("spend_cap_usd")})
        if ds:
            c.delete(f"/api/datasets/{ds}")
        registry.reset()


def _runs(c, ds):
    return {r["id"]: r for r in c.get("/api/ai/runs", params={"dataset_id": ds, "limit": 200}).json()["runs"]}


def _wait_job(c, job_id, timeout=60):
    t0 = time.time()
    while time.time() - t0 < timeout:
        j = c.get(f"/api/jobs/{job_id}").json()
        j = j.get("job") or j
        if j.get("status") in ("done", "error", "cancelled"):
            return j
        time.sleep(0.05)
    raise AssertionError(f"job {job_id} did not finish")


def test_failed_propose_is_booked_and_counts_against_the_cap(fake, app_cloud):
    c, ds, ids = app_cloud
    total0 = c.get("/api/ai/usage", params={"dataset_id": ds}).json()["total_usd"]
    fake.set(thinking_tokens=6000)
    fake.push_fault("anthropic", kind="truncate", keep=0, count=2)
    r = c.post("/api/ai/propose", json={"dataset_id": ds, "image_id": ids[0], "provider": "anthropic",
                                        "model": "claude-opus-5-5"})
    assert r.status_code == 502 and r.json()["code"] == "bad_output", r.text
    assert len(fake.calls("anthropic", "/v1/messages")) == 2
    runs = [x for x in _runs(c, ds).values() if x["kind"] == "propose"]
    assert len(runs) == 1 and runs[0]["status"] == "error" and runs[0]["output_tokens"] >= 12000
    cost = runs[0]["cost_usd"]
    assert cost == pytest.approx(vlm.cost_usd("anthropic", "claude-opus-5-5", runs[0]["input_tokens"],
                                              runs[0]["output_tokens"]), rel=1e-6)
    u = c.get("/api/ai/usage", params={"dataset_id": ds}).json()
    assert u["total_usd"] == pytest.approx(total0 + cost, abs=1e-6)
    assert u["by_provider"]["anthropic"]["usd"] == pytest.approx(cost, abs=1e-6)
    # the 30-day cap sees it: a cap just under what was spent stops the next request before it is sent
    n = len(fake.calls())
    assert c.put("/api/ai/providers/anthropic/config", json={"spend_cap_usd": round(cost * 0.9, 4)}).status_code == 200
    r = c.post("/api/ai/propose", json={"dataset_id": ds, "image_id": ids[0], "provider": "anthropic",
                                        "model": "claude-opus-5-5"})
    assert r.status_code == 402 and r.json()["code"] == "quota", r.text
    assert len(fake.calls()) == n


def test_failed_classify_and_preview_are_booked(fake, app_cloud):
    c, ds, ids = app_cloud
    fake.push_fault("anthropic", kind="refusal", fallback=False)
    r = c.post("/api/ai/classify", json={"dataset_id": ds, "image_id": ids[0], "provider": "anthropic",
                                         "model": "claude-haiku-5-5", "boxes": [[0, 0.2, 0.4, 0.2, 0.5]]})
    assert r.status_code == 502 and r.json()["code"] == "refusal"
    fake.push_fault("anthropic", kind="text", text="nope")
    r = c.post("/api/ai/preview", json={"dataset_id": ds, "image_ids": ids[:2], "provider": "anthropic",
                                        "model": "claude-haiku-5-5"})
    assert r.status_code == 200 and sum(1 for it in r.json()["items"] if it.get("error")) == 1
    runs = list(_runs(c, ds).values())
    cl = [x for x in runs if x["kind"] == "classify"]
    assert len(cl) == 1 and cl[0]["status"] == "error" and cl[0]["input_tokens"] == 120 and cl[0]["cost_usd"] > 0
    pv = [x for x in runs if x["kind"] == "preview"]
    ok_item = next(it for it in r.json()["items"] if not it.get("error"))
    assert len(pv) == 1 and pv[0]["input_tokens"] > ok_item["usage"]["input_tokens"]     # the failed image is in
    assert r.json()["usage"]["input_tokens"] == pv[0]["input_tokens"]


def test_failed_review_name_classes_and_nl_filter_are_booked(fake, app_cloud):
    c, ds, ids = app_cloud
    # Ask AI about one labeled image: the run exists before the call — it keeps the refused call's cost
    fake.push_fault("anthropic", kind="refusal", fallback=False)
    r = c.post("/api/ai/review", json={"dataset_id": ds, "image_id": ids[0], "provider": "anthropic",
                                       "model": "claude-haiku-5-5"})
    assert r.status_code == 502 and r.json()["code"] == "refusal", r.text
    rv = [x for x in _runs(c, ds).values() if x["kind"] == "review"]
    assert len(rv) == 1 and rv[0]["status"] == "error" and rv[0]["input_tokens"] == 120 and rv[0]["cost_usd"] > 0
    # class names: a refused request still costs money
    fake.push_fault("anthropic", kind="refusal", fallback=False, count=4)
    r = c.post("/api/ai/name-classes", json={"dataset_id": ds, "provider": "anthropic", "model": "claude-haiku-5-5",
                                             "classes": [0], "per_class": 4})
    assert r.status_code == 200 and r.json()["items"][0].get("error"), r.text
    nc = [x for x in _runs(c, ds).values() if x["kind"] == "name_classes"]
    assert len(nc) == 1 and nc[0]["input_tokens"] >= 120 and nc[0]["cost_usd"] > 0
    # Ask AI search: refused → 502, and the question's cost is on the day's search row
    fake.reset()
    fake.push_fault("anthropic", kind="refusal", fallback=False)
    r = c.post("/api/ai/nl-filter", json={"dataset_id": ds, "text": "people lying down", "provider": "anthropic",
                                          "model": "claude-haiku-5-5"})
    assert r.status_code == 502 and r.json()["code"] == "refusal", r.text
    nl = [x for x in _runs(c, ds).values() if x["kind"] == "nl_filter"]
    assert len(nl) == 1 and nl[0]["input_tokens"] == 120 and nl[0]["cost_usd"] > 0


def _batch(c, ds, ids, **kw):
    body = {"dataset_id": ds, "image_ids": ids, "provider": "anthropic", "model": "claude-haiku-5-5",
            "target": "proposals", **kw}
    r = c.post("/api/ai/batch", json=body)
    assert r.status_code == 200, r.text
    job = _wait_job(c, r.json()["job"]["id"] if isinstance(r.json().get("job"), dict) else r.json()["job"])
    return job, _runs(c, ds)[r.json()["run_id"]]


def test_sync_batch_books_unreadable_answers(fake, app_cloud):
    c, ds, ids = app_cloud
    _job, clean = _batch(c, ds, ids)
    assert clean["status"] == "done" and clean["images"] == 4 and clean["input_tokens"] > 0
    fake.reset()
    fake.push_fault("anthropic", kind="text", text="cannot comply in JSON", count=2)
    job, run = _batch(c, ds, ids)
    assert len(fake.calls("anthropic", "/v1/messages")) == 4
    assert run["images"] == 2 and (run["result"] or {}).get("errors") == 2
    # all four calls are in the cost (same-size images: the same prompt tokens each)
    assert run["input_tokens"] == clean["input_tokens"]
    assert (run["result"] or {}).get("cost_usd") == pytest.approx(run["cost_usd"], abs=1e-6)
    assert (job.get("extra") or {}).get("cost_usd") == pytest.approx(run["cost_usd"], abs=1e-6)


def test_economy_books_unreadable_answers(fake, app_cloud, monkeypatch):
    from fovea.ai import economy
    monkeypatch.setattr(economy, "POLL_S", 0.1)
    c, ds, ids = app_cloud
    _job, clean = _batch(c, ds, ids, speed="economy")
    assert clean["status"] == "done" and clean["input_tokens"] > 0, clean
    fake.reset()
    fake.push_fault("anthropic", kind="text", text="cannot comply in JSON", count=2)
    _job, run = _batch(c, ds, ids, speed="economy")
    assert run["status"] == "done" and (run["result"] or {}).get("errors") == 2
    assert run["input_tokens"] == clean["input_tokens"]                  # the 2 failed items are billed too
    tin_each = clean["input_tokens"] // 4
    assert run["cost_usd"] >= 4 * vlm.cost_usd("anthropic", "claude-haiku-5-5", tin_each, 0, batch=True) - 1e-9


# ---- 3. OpenAI Flex tier

def test_flex_tier_is_estimated_at_the_price_it_is_billed(fake, scene):
    std = OpenAIProvider(config={"batch_mode": False}, key="")
    a = std.estimate(1920, 1080, "gpt-6.1-sol", 2, {})
    flex = OpenAIProvider(config={"batch_mode": True}, key="")
    b = flex.estimate(1920, 1080, "gpt-6.1-sol", 2, {})
    assert "tier" not in a and b["tier"] == "flex"
    assert a["usd"][0] == pytest.approx(2 * a["usd_batch"][0], rel=1e-3)         # standard: twice the batch price
    assert b["usd"] == b["usd_batch"] == a["usd_batch"]                            # Flex: the batch price already
    assert b["seconds"] == pytest.approx(a["seconds"] * vlm.FLEX_SLOWDOWN, abs=0.11)
    assert flex.estimate(1920, 1080, "gpt-6.1-sol", 2, {"batch_mode": False})["usd"] == a["usd"]
    # the estimate matches the bill
    p = oai(fake, batch_mode=True)
    r = p.detect(DetectRequest(scene, CLASSES, "gpt-6.1-sol"))
    assert fake.calls("openai", "/v1/responses")[-1]["body"]["service_tier"] == "flex"
    assert r.usage.cost_usd == pytest.approx(vlm.cost_usd("openai", "gpt-6.1-sol", r.usage.input_tokens,
                                                          r.usage.output_tokens, batch=True), rel=1e-6)
    # a model that turns the Flex tier down runs — and is estimated — at the standard price
    fake.set(reject_params=["service_tier"])
    p.detect(DetectRequest(scene, CLASSES, "gpt-6.1-sol"))
    e = p.estimate(1920, 1080, "gpt-6.1-sol", 2, {})
    assert "tier" not in e and e["usd"][0] == pytest.approx(2 * e["usd_batch"][0], rel=1e-3)
    assert OpenAIProvider.capabilities["flex"] is True


def test_flex_estimate_through_the_api(fake, app_cloud):
    c, ds, _ids = app_cloud
    body = {"dataset_id": ds, "count": 100, "provider": "openai", "model": "gpt-6.1-sol"}
    std = c.post("/api/ai/estimate", json=body).json()
    assert std.get("tier") is None and std["usd"][0] == pytest.approx(2 * std["usd_batch"][0], rel=1e-2)
    assert c.put("/api/ai/providers/openai/config", json={"batch_mode": True}).status_code == 200
    fl = c.post("/api/ai/estimate", json=body).json()
    assert fl["tier"] == "flex" and fl["usd"] == fl["usd_batch"] == std["usd_batch"]
    assert any("Flex tier" in a for a in fl["assumptions"])
    assert fl["seconds_per_image"] > std["seconds_per_image"]


def test_batch_dialog_price_line_for_the_flex_tier(tmp_path):
    import shutil
    import subprocess
    node = shutil.which("node")
    if not node:
        pytest.skip("node is not installed")
    static = Path(__file__).resolve().parent.parent / "static"
    (tmp_path / "views").mkdir()
    (tmp_path / "lib").mkdir()
    shutil.copy(static / "views" / "ai-panel.js", tmp_path / "views" / "ai-panel.js")
    shutil.copy(static / "lib" / "dom.js", tmp_path / "lib" / "dom.js")
    (tmp_path / "package.json").write_text('{"type":"module"}')
    (tmp_path / "run.mjs").write_text("""
globalThis.location = { hostname: 'localhost' };
globalThis.navigator = globalThis.navigator || { platform: 'MacIntel' };
const m = await import('./views/ai-panel.js');
const flex = { usd: [0.5, 1], usd_batch: [0.5, 1], tier: 'flex' }, std = { usd: [1, 2], usd_batch: [0.5, 1] };
const oai = (on) => ({ id: 'openai', capabilities: { flex: true }, config: { batch_mode: on } });
console.log(JSON.stringify({ flexNow: m.priceLine(flex, 'now'), flexEco: m.priceLine(flex, 'economy'),
  stdNow: m.priceLine(std, 'now'), on: m.flexOn(oai(true)), off: m.flexOn(oai(false)),
  anth: m.flexOn({ id: 'anthropic', capabilities: {}, config: { batch_mode: true } }) }));
""")
    p = subprocess.run([node, "run.mjs"], cwd=tmp_path, capture_output=True, text=True, timeout=60)
    assert p.returncode == 0, p.stderr
    out = json.loads(p.stdout.strip().splitlines()[-1])
    assert out["flexNow"]["main"] == "≈ $0.50–1.00" and "Economy costs the same" in out["flexNow"]["other"]
    assert "Flex tier" in out["flexEco"]["other"] and "now $" not in out["flexEco"]["other"]
    assert out["stdNow"]["other"].startswith("Economy ≈ $0.50–1.00")
    assert (out["on"], out["off"], out["anth"]) == (True, False, False)


# ---- the fake Messages API bills the user text too (it billed only the system prompt and the images, so the
#      suggest-names estimate looked 1.2-1.5x too high against it); detect / refine estimates cover the text sent

def test_fake_anthropic_bills_system_and_user_text(fake, scene):
    from fake_vlm import TURN_TOKENS
    fake.set(generate=NAME_ANSWER)
    p = anth(fake)
    data = Path(scene).read_bytes()
    system, short, long = "You name classes. " * 10, "Name them.", "Name the classes in these crops. " * 40
    _o, u1 = p.generate_json(system, short, NAME_SCHEMA, "claude-haiku-5-5", [data])
    _o, u2 = p.generate_json(system, long, NAME_SCHEMA, "claude-haiku-5-5", [data])
    img = sum(math.ceil(w / 28) * math.ceil(h / 28) for w, h in fake.calls("anthropic")[-1]["images"])
    assert u1.input_tokens == img + len(system) // 4 + len(short) // 4 + TURN_TOKENS
    assert u2.input_tokens - u1.input_tokens == len(long) // 4 - len(short) // 4      # the user text is billed
    # a bare-string user turn is read (and billed) like a list of blocks
    from fake_vlm import anthropic_text_tokens
    body = {"system": system, "messages": [{"role": "user", "content": long}]}
    assert anthropic_text_tokens(body) == len(system) // 4 + len(long) // 4 + TURN_TOKENS


@pytest.mark.parametrize("model", ["claude-haiku-5-5", "claude-sonnet-5-5"])
@pytest.mark.parametrize("n", [1, 4, 12])
def test_detect_estimate_covers_the_billed_prompt(fake, tmp_path, model, n):
    """The estimate counts the prompts actually sent: never below the bill, at most a few % above it."""
    img = tmp_path / "s.jpg"
    make_scene(str(img), (640, 640), [(RED, (100, 120, 300, 500)), (BLUE, (350, 300, 600, 600))])
    names = ["person", "car", "fallen person", "dog"]
    classes = [ClassSpec(i, f"class_{i}", prompt=names[i % 4]) for i in range(n)]
    p = anth(fake)
    r = p.detect(DetectRequest(str(img), classes, model))
    est = p.estimate(640, 640, model, n, {})["tokens_in"][0]
    assert r.usage.input_tokens <= est <= r.usage.input_tokens * 1.05, (r.usage.input_tokens, est)


def test_refine_estimate_covers_each_billed_refine_prompt(fake, small_scene):
    from fake_vlm import anthropic_text_tokens
    fake.set(loose=0.3)
    p = anth(fake)
    res = p.detect(DetectRequest(small_scene, CLASSES, "", options={"refine": True}))
    assert res.stats["refine_calls"] == 2
    crops = [c for c in fake.calls("anthropic") if vlm.REFINE_MARK in c["text"]]
    per_call = p.refine_prompt_tokens(vlm.PIXEL_XYXY, {})
    for c in crops:
        assert anthropic_text_tokens(c["body"]) <= per_call <= anthropic_text_tokens(c["body"]) * 1.15


def test_naming_estimate_matches_the_fake_bill(fake, scene):
    """The finding's case: suggest-names' estimate low end was 1.2-1.5x the fake bill because the fake dropped the
    user text; with the text billed the two agree (the estimate keeps a small per-request allowance)."""
    from fovea.ai import assist
    fake.set(generate={"name": "standing_person", "prompt": "person standing", "description": "upright",
                       "confidence": 0.9})
    W, H = 1600, 1000
    ex = [{"abs_path": scene, "rel_path": "scene.jpg", "width": W, "height": H, "xc": (100 + 400) / 2 / W,
           "yc": (120 + 700) / 2 / H, "w": 300 / W, "h": 580 / H}] * 3
    text = ("Dataset: demo.\nClass 0. Images 1-3 are its examples (3 crops from 3 different images). It has 1,234 "
            "labeled boxes in 987 images; boxes are tall and medium-sized.\n") * 3
    schema = {"type": "object", "properties": assist._name_props(),
              "required": ["name", "prompt", "description", "confidence"], "additionalProperties": False}
    for model in ("claude-haiku-5-5", "claude-sonnet-5-5"):
        p = anth(fake)
        est = assist.naming_estimate(p, model, [([0], text, schema)], {0: ex})
        imgs = [assist.crop_jpeg(e["abs_path"], (e["xc"], e["yc"], e["w"], e["h"])) for e in ex]
        _obj, usage = p.generate_json(assist.NAME_SYSTEM, text, schema, model, images=imgs,
                                      options={"max_tokens": 1500})
        assert usage.input_tokens <= est["tokens_in"][0] <= usage.input_tokens * 1.1, (usage.input_tokens, est)
