"""Deterministic test provider (registered only when ``FOVEA_AI_MOCK=1``).

Behaves like a cheap cloud VLM so tests and UI work can exercise every path without a key or a GPU:

* detect: when the image already has ground-truth rows in the index it returns them slightly perturbed — every 7th box
  dropped, every 11th class swapped, a loose box now and then and one spurious box on about a fifth of the images — so
  the audit and the benchmark produce realistic findings. Images without labels get 1-3 pseudo-random boxes.
* classify: verdict derived from a hash of the crop; ``hint`` (the current label) wins 2 times out of 3.
* costs $0.0010 per detect call and $0.0004 per classified crop, so budgets and the usage ledger can be tested.

Options (tests): ``mock_error`` = one of ProviderError's codes raised on every call, ``mock_error_every`` = raise only on
every n-th image (by hash), ``mock_latency_ms`` = sleep per call, ``mock_batch_delay_s`` = how long a provider batch
stays "running" (default 2 s).

* generate_json: answers the two structured tasks Fovea asks (class naming, natural-language filters) from the request
  text alone — box-shape hints for names, keyword rules (English and Korean) for filters — at $0.0005 per call.
* batch API: batches persist under ``DATA_DIR/ai_cache/mock_batches`` (they survive a server restart, like a real
  provider's) and cost half the synchronous price.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ... import config as app_config
from ... import db
from ..base import (ClassSpec, Classification, ClassifyRequest, ClassifyResult, DetectRequest, DetectResult, Detection,
                    Provider, ProviderError, Usage)

DETECT_USD = 0.0010
CROP_USD = 0.0004
GENERATE_USD = 0.0005
BATCH_DISCOUNT = 0.5
_batch_lock = threading.Lock()


def _h(*parts) -> int:
    return int(hashlib.sha1("|".join(str(p) for p in parts).encode()).hexdigest()[:12], 16)


def _frac(*parts) -> float:
    return (_h(*parts) % 10_000) / 10_000.0


_LYING = ("fall", "lying", "lie", "down", "prone", "floor", "ground")
_STANDING = ("stand", "upright", "walk", "person")


def _pick_pose(classes, lying: bool):
    if not classes:
        return None
    words = _LYING if lying else _STANDING
    for c in classes:
        txt = f"{c.name} {c.prompt}".lower()
        if any(w in txt for w in words) and not (not lying and any(w in txt for w in _LYING)):
            return c.id
    return classes[min(1, len(classes) - 1)].id if lying else classes[0].id


class MockProvider(Provider):
    id = "mock"
    name = "Mock (tests)"
    kind = "cloud"
    location = "Deterministic fake answers — nothing leaves this machine"
    needs_key = False
    capabilities = {"detect": True, "classify": True, "refine": False, "open_vocab": True, "exemplars": False,
                    "batch_api": True, "confidence": True, "generate": True}

    def models(self, refresh: bool = False) -> List[dict]:
        return [
            {"id": "mock-fast", "name": "Mock fast", "notes": "Instant, deterministic", "recommended": True, "default": True,
             "price": {"in": 0.5, "out": 2.0, "batch_in": 0.25, "batch_out": 1.0}, "max_edge": 1568},
            {"id": "mock-accurate", "name": "Mock accurate", "notes": "Same answers, pretends to think longer",
             "recommended": False, "default": False, "price": {"in": 2.0, "out": 8.0, "batch_in": 1.0, "batch_out": 4.0},
             "max_edge": 1568},
        ]

    def test(self) -> dict:
        return {"ok": True, "message": "Mock provider is always reachable", "latency_ms": 1}

    def priced(self, model: str) -> bool:
        """Like a real provider after "Refresh list": a model id missing from the price table is served but costs
        are not tracked (booked at $0, with a warning)."""
        return (model or "") in {m["id"] for m in self.models()}

    def estimate(self, width, height, model, n_classes, options):
        if not self.priced(model):
            return {"usd": [0.0, 0.0], "tokens_in": [1500, 1800], "tokens_out": [60, 400], "seconds": 0.05,
                    "usd_batch": None, "priced": False}
        return {"usd": [DETECT_USD * 0.8, DETECT_USD * 1.2], "tokens_in": [1500, 1800], "tokens_out": [60, 400],
                "seconds": 0.05, "usd_batch": [DETECT_USD * 0.4, DETECT_USD * 0.6], "priced": True}

    def _maybe_fail(self, path: str, options: dict) -> None:
        code = options.get("mock_error")
        if not code:
            return
        every = int(options.get("mock_error_every") or 1)
        if every <= 1 or _h(os.path.basename(path), "err") % every == 0:
            raise ProviderError(f"Mock {code} error", code, status={"auth": 401, "rate": 429, "quota": 429}.get(code))

    def detect(self, req: DetectRequest) -> DetectResult:
        t0 = time.time()
        opts = req.options or {}
        if opts.get("mock_latency_ms"):
            time.sleep(float(opts["mock_latency_ms"]) / 1000)
        self._maybe_fail(req.image_path, opts)
        if not req.classes:
            return DetectResult([], Usage(), 0, ["No classes to look for"], req.model)
        ids = [c.id for c in req.classes]
        by_id = {c.id: c for c in req.classes}
        gt = []
        try:
            row = db.query_one("SELECT id FROM images WHERE abs_path=? ORDER BY id LIMIT 1", (req.image_path,))
            if row:
                gt = db.query("SELECT cls, xc, yc, w, h FROM boxes WHERE image_id=? ORDER BY id", (row["id"],))
        except Exception:  # noqa: BLE001 — works without an index too
            gt = []
        dets: List[Detection] = []
        p = os.path.basename(req.image_path)        # stable across copies of the same dataset
        # A vocabulary unrelated to the labels (two-stage "person") sees every labeled object as its first class.
        foreign = bool(gt) and not any(int(b["cls"]) in by_id for b in gt)
        if gt:
            for k, b in enumerate(gt):
                if _h(p, k, "drop") % 7 == 0:
                    continue
                cls = ids[0] if foreign else int(b["cls"])
                if not foreign and _h(p, k, "swap") % 11 == 0 and len(ids) > 1:
                    cls = ids[(ids.index(cls) + 1) % len(ids)] if cls in ids else ids[0]
                if cls not in by_id:
                    continue
                j = 0.03 if _h(p, k, "loose") % 9 == 0 else 0.004
                xc = b["xc"] + (_frac(p, k, "x") - 0.5) * j
                yc = b["yc"] + (_frac(p, k, "y") - 0.5) * j
                w = b["w"] * (1 + (_frac(p, k, "w") - 0.5) * j * 6)
                h = b["h"] * (1 + (_frac(p, k, "h") - 0.5) * j * 6)
                conf = round(0.55 + 0.44 * _frac(p, k, "c"), 4)
                dets.append(Detection(cls, xc, yc, w, h, conf, by_id[cls].name))
            if not dets and gt and (foreign or int(gt[0]["cls"]) in by_id):
                b = gt[0]
                c0 = ids[0] if foreign else int(b["cls"])
                dets.append(Detection(c0, b["xc"], b["yc"], b["w"], b["h"], 0.9, by_id[c0].name))
            if _h(p, "spurious") % 5 == 0:
                c = ids[_h(p, "sc") % len(ids)]
                dets.append(Detection(c, 0.1 + 0.8 * _frac(p, "sx"), 0.1 + 0.8 * _frac(p, "sy"), 0.08, 0.12,
                                      round(0.6 + 0.3 * _frac(p, "sconf"), 4), by_id[c].name))
        else:
            n = 1 + _h(p, "n") % 3
            for k in range(n):
                c = ids[_h(p, k, "cls") % len(ids)]
                w = 0.08 + 0.25 * _frac(p, k, "w")
                h = 0.08 + 0.35 * _frac(p, k, "h")
                xc = w / 2 + (1 - w) * _frac(p, k, "x")
                yc = h / 2 + (1 - h) * _frac(p, k, "y")
                dets.append(Detection(c, xc, yc, w, h, round(0.3 + 0.68 * _frac(p, k, "c"), 4), by_id[c].name))
        dets = [d for d in dets if d.conf is None or d.conf >= (req.conf or 0)]
        priced = self.priced(req.model)
        usage = Usage(1500 + 12 * len(req.classes), 60 + 40 * len(dets), DETECT_USD if priced else 0.0)
        warn = [] if priced else [f"no price on file for {req.model}; cost not tracked"]
        return DetectResult(dets, usage, int((time.time() - t0) * 1000) + 5, warn, req.model)

    def classify(self, req: ClassifyRequest) -> ClassifyResult:
        t0 = time.time()
        opts = req.options or {}
        if opts.get("mock_latency_ms"):
            time.sleep(float(opts["mock_latency_ms"]) / 1000)
        self._maybe_fail(req.image_path, opts)
        items = []
        names = {c.name: c.id for c in req.classes}
        for cr in req.crops:
            hv = _h(os.path.basename(req.image_path), cr.index, round(cr.xc, 3), round(cr.yc, 3))
            if hv % 13 == 0:
                items.append(Classification(cr.index, None, "none", 0.7, "mock: no target object in this crop"))
                continue
            if hv % 17 == 0:
                items.append(Classification(cr.index, None, "uncertain", 0.3, "mock: too small to tell"))
                continue
            if cr.hint in names and hv % 3 != 0:
                cid = names[cr.hint]
            else:
                # wide boxes look "lying", tall ones "standing": gives a fall dataset something plausible
                cid = _pick_pose(req.classes, lying=cr.w > cr.h)
            conf = round(0.5 + 0.5 * _frac(req.image_path, cr.index, "cc"), 3)
            items.append(Classification(cr.index, cid, "label", conf,
                                        f"mock: {'wide' if cr.w > cr.h else 'tall'} box ({cr.w:.2f}×{cr.h:.2f})"))
        usage = Usage(400 + 300 * len(req.crops), 40 * len(req.crops),
                      CROP_USD * len(req.crops) if self.priced(req.model) else 0.0)
        return ClassifyResult(items, usage, int((time.time() - t0) * 1000) + 5, [])


    # ---------------------------------------------------------------- structured generation

    def generate_json(self, system: str, user: str, schema: dict, model: str, images: Optional[List[bytes]] = None,
                      options: Optional[Dict[str, Any]] = None) -> Tuple[dict, Usage]:
        opts = options or {}
        if opts.get("mock_latency_ms"):
            time.sleep(float(opts["mock_latency_ms"]) / 1000)
        code = opts.get("mock_error")
        if code:
            raise ProviderError(f"Mock {code} error", code)
        if not isinstance(schema, dict) or schema.get("type") != "object":
            raise ProviderError("generate_json needs an object schema", "invalid")
        props = schema.get("properties") or {}
        if "filters" in props:
            out = _mock_filters(system, user)
        elif "sources" in props:
            out = _mock_folders(user)
        elif "classes" in props:
            out = {"classes": [_mock_name(block) for block in _class_blocks(user)]}
        elif "name" in props:
            out = _mock_name(user)
        else:
            out = {k: _schema_default(v) for k, v in props.items()}
        n_img = len(images or [])
        usage = Usage(len(system) // 4 + len(user) // 4 + 260 * n_img, 80 + 40 * len(out.get("classes") or [1]),
                      (GENERATE_USD + 0.0001 * n_img) if self.priced(model) else 0.0)
        return out, usage

    # ---------------------------------------------------------------- batch API (persisted like a real provider's)

    @staticmethod
    def _batch_dir() -> Path:
        d = Path(app_config.DATA_DIR) / "ai_cache" / "mock_batches"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _batch_file(self, batch_id: str) -> Path:
        if not re.fullmatch(r"mockbatch_[0-9a-f]{16}", batch_id or ""):
            raise ProviderError(f"Unknown batch: {batch_id}", "invalid", status=404)
        return self._batch_dir() / f"{batch_id}.json"

    def _batch_load(self, batch_id: str) -> dict:
        p = self._batch_file(batch_id)
        try:
            return json.loads(p.read_text())
        except (OSError, ValueError):
            raise ProviderError(f"Unknown batch: {batch_id}", "invalid", status=404) from None

    def batch_submit(self, items: List[Dict[str, Any]]) -> str:
        if not items:
            raise ProviderError("A batch needs at least one request", "invalid")
        recs = []
        for it in items:
            req: DetectRequest = it["request"]
            opts = {k: v for k, v in (req.options or {}).items() if k not in ("exemplars",)}
            if opts.get("mock_submit_error"):
                raise ProviderError(f"Mock {opts['mock_submit_error']} error", opts["mock_submit_error"])
            recs.append({"custom_id": str(it["custom_id"]), "image_path": req.image_path, "model": req.model,
                         "conf": req.conf, "classes": [asdict(c) for c in req.classes], "options": opts})
        delay = float((items[0]["request"].options or {}).get("mock_batch_delay_s", 2.0))
        bid = "mockbatch_" + uuid.uuid4().hex[:16]
        rec = {"id": bid, "created_at": time.time(), "ready_at": time.time() + delay, "canceled_at": None,
               "items": recs}
        with _batch_lock:
            self._batch_file(bid).write_text(json.dumps(rec))
        return bid

    def batch_status(self, batch_id: str) -> Dict[str, Any]:
        rec = self._batch_load(batch_id)
        n = len(rec["items"])
        now = time.time()
        if rec.get("canceled_at"):
            done = n if rec["canceled_at"] >= rec["ready_at"] else 0
            return {"state": "ended", "counts": {"processing": 0, "succeeded": done, "errored": 0, "canceled": n - done,
                                                 "expired": 0}, "created_at": rec["created_at"]}
        if now < rec["ready_at"]:
            frac = max(0.0, min(1.0, (now - rec["created_at"]) / max(1e-6, rec["ready_at"] - rec["created_at"])))
            done = int(n * frac * 0.9)
            return {"state": "running", "counts": {"processing": n - done, "succeeded": done, "errored": 0,
                                                   "canceled": 0, "expired": 0}, "created_at": rec["created_at"]}
        return {"state": "ended", "counts": {"processing": 0, "succeeded": n, "errored": 0, "canceled": 0, "expired": 0},
                "created_at": rec["created_at"]}

    def batch_results(self, batch_id: str, items: Optional[List[Dict[str, Any]]] = None):
        rec = self._batch_load(batch_id)
        st = self.batch_status(batch_id)
        if st["state"] != "ended":
            raise ProviderError("The batch has not ended yet", "invalid")
        canceled = st["counts"]["canceled"] > 0
        for r in rec["items"]:
            if canceled:
                yield r["custom_id"], ProviderError("Canceled before it ran", "canceled")
                continue
            req = DetectRequest(image_path=r["image_path"], classes=[ClassSpec(**c) for c in r["classes"]],
                                model=r["model"], conf=r["conf"], options=r["options"])
            try:
                res = self.detect(req)
            except ProviderError as e:
                yield r["custom_id"], e
                continue
            res.usage.cost_usd = round(res.usage.cost_usd * BATCH_DISCOUNT, 8)
            yield r["custom_id"], res

    def batch_cancel(self, batch_id: str) -> Dict[str, Any]:
        with _batch_lock:
            rec = self._batch_load(batch_id)
            if not rec.get("canceled_at"):
                rec["canceled_at"] = time.time()
                self._batch_file(batch_id).write_text(json.dumps(rec))
        return self.batch_status(batch_id)


# ------------------------------------------------------------------ mock structured answers

def _schema_default(v: dict):
    t = (v or {}).get("type")
    if (v or {}).get("enum"):
        return v["enum"][0]
    return {"string": "", "number": 0.5, "integer": 0, "boolean": False, "array": [], "object": {}}.get(t, None)


_FOLDER_LINE = re.compile(r"^(\S+)\s+images=(\d+) txt=(\d+)")


def _mock_folders(user: str) -> dict:
    """Folder roles from the listing ai/folders.py sends: every folder with images directly inside is an image
    folder; its labels are the .txt folder sharing the most trailing path parts (imgs/train ↔ anns/train), the split
    the image folder's last part when it reads like one."""
    rows = [m.groups() for m in map(_FOLDER_LINE.match, user.splitlines()) if m]
    imgs = [r for r, n, _t in rows if int(n) > 0]
    txts = [r for r, _n, t in rows if int(t) > 0]

    def shared(a: str, b: str) -> tuple:
        pa, pb, head, tail = a.split("/"), b.split("/"), 0, 0
        while head < min(len(pa), len(pb)) and pa[head] == pb[head]:
            head += 1
        while tail < min(len(pa), len(pb)) - head and pa[-1 - tail] == pb[-1 - tail]:
            tail += 1
        word = lambda p: next((w for w in re.split(r"[^a-z]+", p.split("/")[-1].lower()) if w in ("train", "val", "test")), "")
        return (tail, word(a) == word(b), head, len(pa) == len(pb))
    out = []
    for r in imgs:
        lab = max(txts, key=lambda t: (shared(r, t), t == r)) if txts else ""
        last = r.split("/")[-1].lower()
        out.append({"split": last if last in ("train", "val", "valid", "test") else "", "images": r, "labels": lab})
    return {"sources": out, "reason": "mock: folders with images, paired with the closest .txt folder"}


def _class_blocks(text: str) -> List[str]:
    """Split a multi-class naming request into one block per "Class <id>" section."""
    parts = re.split(r"(?m)^(?=Class \d+\b)", text or "")
    return [p for p in parts if re.match(r"Class \d+", p)]


def _mock_name(text: str) -> dict:
    text = (text or "").split("Other classes")[0]
    m = re.search(r"Class (\d+)", text or "")
    cid = int(m.group(1)) if m else 0
    wide = bool(re.search(r"wider than tall", text or ""))
    tall = bool(re.search(r"taller than wide", text or ""))
    shape = "lying" if wide else ("upright" if tall else "compact")
    out = {"name": f"{shape}_object_{cid}", "prompt": f"{shape} object",
           "description": f"Mock suggestion from box shapes only ({shape}).",
           "confidence": "low" if not (wide or tall) else "medium"}
    if m:
        out["cls"] = cid
    return out


_SPLIT_WORDS = {"train": ("train", "training", "학습", "훈련"), "val": ("val", "valid", "validation", "검증"),
                "test": ("test", "테스트", "시험")}
_FLAG_WORDS = {"blurry": ("blurry", "blur", "흐린", "흐릿"), "dark": ("dark", "어두운"), "overexposed": ("overexposed", "밝은", "과노출"),
               "near_dup": ("duplicate", "near-duplicate", "중복"), "leak_split": ("leak", "누수")}


def _mock_filters(system: str, user: str) -> dict:
    t = (user or "").lower()
    f: Dict[str, Any] = {}
    why: List[str] = []
    classes = []
    for m in re.finditer(r"(?m)^class (\d+): ([^|\n]+)(.*)$", system or ""):
        pm = re.search(r"\| prompt: ([^|]+)", m.group(3) or "")
        classes.append((int(m.group(1)), m.group(2).strip().lower(), (pm.group(1) if pm else "").strip().lower()))
    hits = []
    for cid, name, prompt in classes:
        words = [w for w in re.split(r"[\s_]+", name) if len(w) > 2] + ([prompt] if prompt else [])
        if name in t or any(w and w in t for w in words):
            hits.append(cid)
    if any(w in t for w in ("fall", "fallen", "넘어", "쓰러")):
        hits += [cid for cid, name, prompt in classes if any(k in name + prompt for k in ("fall", "lying", "lie"))]
    if hits:
        f["cls"] = sorted(set(hits))
        why.append("classes " + ", ".join(str(c) for c in f["cls"]))
    for sp, words in _SPLIT_WORDS.items():
        if any(re.search(rf"(^|\W){re.escape(w)}(\W|$)", t) for w in words):
            f["split"] = [sp]
            why.append(f"split {sp}")
            break
    if any(w in t for w in ("unlabeled", "no label", "라벨 없는", "라벨이 없는", "미라벨")):
        f["labeled"] = "unlabeled"
        why.append("not labeled yet")
    flags = [code for code, words in _FLAG_WORDS.items() if any(w in t for w in words)]
    if flags:
        f["issue"] = flags
        why.append("flags " + ", ".join(flags))
    if any(w in t for w in ("not reviewed", "unreviewed", "검토 안", "미검토")):
        f["review"] = "none"
        why.append("not reviewed")
    elif any(w in t for w in ("flagged", "플래그")):
        f["review"] = "flagged"
        why.append("flagged")
    m = re.search(r"(?:more than|over|at least|이상)\s*(\d+)|(\d+)\s*(?:개 이상|or more|\+)", t)
    if m:
        f["min_boxes"] = int(m.group(1) or m.group(2))
        why.append(f"at least {f['min_boxes']} boxes")
    out: Dict[str, Any] = {"filters": f, "explanation": "Mock: " + ("; ".join(why) if why else "no filter recognised")}
    if any(w in t for w in ("suspicious", "most likely wrong", "의심")):
        out["sort"] = "audit"
    return out


PROVIDER = MockProvider()
