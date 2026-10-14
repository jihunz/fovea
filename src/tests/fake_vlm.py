"""Fake Anthropic / Gemini / OpenAI / OpenAI-compatible HTTP APIs for tests and live checks (no real key needed).

The fake *looks at* the image it receives: synthetic test scenes draw each object as a saturated colour block, and the
fake finds those blocks in the decoded upload and answers in the provider's own coordinate frame for the size it
actually received (Claude: pixels; Gemini / GPT: box_2d 0-1000 y-first; Qwen3: 0-1000 x-first; Qwen2.5: pixels on its
own 28-px resize grid). So a client that resizes, tiles, crops or orders coordinates wrongly gets wrong boxes back —
the same way it would from the real model. Scripted boxes (normalised to the received image) can replace the colour
search for real photos.

It also validates requests the way the real APIs do where it matters (auth headers, no key in the URL, Claude's
schema restrictions, OpenAI strict-schema rules, oversized images, no sampling params on Claude 4.7+, fallback beta)
and injects failures on demand: 429 with retry-after, 529/503, 500, 401, spend limits, refusals, truncated output,
wrong-scale coordinates, prose-wrapped JSON, servers that reject ``response_format``.

Beyond detection / classification it answers generic structured-output requests (``generate_json``: a scripted answer
or one synthesised from the schema), emulates the three batch APIs (Anthropic Message Batches, OpenAI ``/v1/files`` +
``/v1/batches``, Gemini ``batchGenerateContent``) by running every request through the same logic, and can play a
sloppy first pass (``loose``) that a zoomed refine request answers exactly.

Usage in tests::

    fake = FakeVLM().start()
    fake.push_fault("anthropic", kind="rate_limit", retry_after=1)
    ...  provider with config={"base_url": fake.url} ...
    fake.stop()

Live checks: ``python tests/fake_vlm.py --port 8955`` and drive it with ``POST /_fake/scenario`` / ``/_fake/fault``.
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import math
import re
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import parse_qs, urlparse

KEYS = {"anthropic": "sk-ant-api03-FAKEKEYfakekeyFAKEKEY0001", "gemini": "AIzaFAKEfakeFAKEfakeFAKEfake0001",
        "openai": "sk-proj-FAKEfakeFAKEfakeFAKE0001", "compat": None}

# colour → label. Saturated, far apart, robust to JPEG.
PALETTE = {
    (230, 30, 30): "person",
    (30, 60, 230): "car",
    (30, 190, 40): "dog",
    (240, 220, 20): "fallen person",
    (200, 40, 220): "bicycle",
    (20, 210, 220): "helmet",
}
TOL = 70

TURN_TOKENS = 7          # role / turn framing the Messages API bills per message

HIGH = (2576, 4784)
STANDARD = (1568, 1568)
STANDARD_MODELS = re.compile(r"-4-[0-6]\b|-4-5|haiku-4|sonnet-4")
NO_SAMPLING = re.compile(r"claude-(opus|sonnet|haiku|fable|mythos)-(5|4-[78])")
FALLBACK_MODELS = {"claude-opus-5-5": "claude-opus-5", "claude-sonnet-5-5": "claude-sonnet-5",
                   "claude-fable-5-1": "claude-opus-5", "claude-opus-5": "claude-opus-4-8", "claude-fable-5": "claude-opus-4-8"}
REFINE_MARK = "nearest the centre of this zoomed crop"
CUSTOM_ID = re.compile(r"^[a-zA-Z0-9_-]{1,64}$")


# ------------------------------------------------------------------------------------------- scene helpers

def make_scene(path: str, size: Tuple[int, int], objects: List[Tuple[Tuple[int, int, int], Tuple[int, int, int, int]]],
               *, background=(128, 128, 128), exif_orientation: Optional[int] = None, quality: int = 95) -> None:
    """Write a test image: ``objects`` = [(rgb, (x1, y1, x2, y2) pixel edges)]. With ``exif_orientation`` the pixels
    are stored rotated and the EXIF tag says how to rotate them back (so the *displayed* image matches ``objects``)."""
    from PIL import Image, ImageDraw
    im = Image.new("RGB", size, background)
    d = ImageDraw.Draw(im)
    for rgb, (x1, y1, x2, y2) in objects:
        d.rectangle([x1, y1, x2 - 1, y2 - 1], fill=rgb)
    if exif_orientation in (6, 8, 3):
        # store the inverse rotation; viewers apply the tag and get `im` back
        stored = {6: im.transpose(Image.ROTATE_90), 8: im.transpose(Image.ROTATE_270),
                  3: im.transpose(Image.ROTATE_180)}[exif_orientation]
        exif = Image.Exif()
        exif[0x0112] = exif_orientation
        stored.save(path, quality=quality, exif=exif.tobytes())
    else:
        im.save(path, quality=quality)


def find_blocks(im) -> List[Tuple[str, Tuple[int, int, int, int]]]:
    """[(label, (x1, y1, x2, y2) pixel edges)] of palette-coloured blocks in a decoded image."""
    import numpy as np
    a = np.asarray(im.convert("RGB")).astype(np.int16)
    out = []
    for rgb, label in PALETTE.items():
        mask = (np.abs(a - np.array(rgb, dtype=np.int16)) < TOL).all(axis=2)
        if mask.sum() < 4:
            continue
        for x1, y1, x2, y2 in _split(mask, 0, 0, mask.shape[1], mask.shape[0], 0):
            out.append((label, (x1, y1, x2, y2)))
    return out


def _runs(v, min_len=2):
    runs, start = [], None
    for i, on in enumerate(list(v) + [False]):
        if on and start is None:
            start = i
        elif not on and start is not None:
            if i - start >= min_len:
                runs.append((start, i))
            start = None
    return runs


def _split(mask, ox, oy, x2, y2, depth):
    sub = mask[oy:y2, ox:x2]
    cols = _runs(sub.any(axis=0))
    boxes = []
    for c0, c1 in cols:
        rows = _runs(sub[:, c0:c1].any(axis=1))
        for r0, r1 in rows:
            bx1, by1, bx2, by2 = ox + c0, oy + r0, ox + c1, oy + r1
            if depth < 3 and (len(cols) > 1 or len(rows) > 1):
                inner = _split(mask, bx1, by1, bx2, by2, depth + 1)
                if len(inner) > 1:
                    boxes.extend(inner)
                    continue
            if mask[by1:by2, bx1:bx2].sum() >= 4:
                boxes.append((bx1, by1, bx2, by2))
    return boxes


def resized_size(w, h, max_edge=2576, max_tokens=4784):
    def fits(a, b):
        return (math.ceil(a / 28) * 28 <= max_edge and math.ceil(b / 28) * 28 <= max_edge
                and math.ceil(a / 28) * math.ceil(b / 28) <= max_tokens)
    if fits(w, h):
        return w, h
    if h > w:
        rh, rw = resized_size(h, w, max_edge, max_tokens)
        return rw, rh
    ar = w / h
    lo, hi = 1, w
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if fits(mid, max(round(mid / ar), 1)):
            lo = mid
        else:
            hi = mid
    return lo, max(round(lo / ar), 1)


def smart_resize(w, h, factor=28, min_pixels=56 * 56, max_pixels=12845056):
    h_bar = max(factor, round(h / factor) * factor)
    w_bar = max(factor, round(w / factor) * factor)
    if h_bar * w_bar > max_pixels:
        beta = math.sqrt((h * w) / max_pixels)
        h_bar = max(factor, math.floor(h / beta / factor) * factor)
        w_bar = max(factor, math.floor(w / beta / factor) * factor)
    elif h_bar * w_bar < min_pixels:
        beta = math.sqrt(min_pixels / (h * w))
        h_bar = math.ceil(h * beta / factor) * factor
        w_bar = math.ceil(w * beta / factor) * factor
    return w_bar, h_bar


# ------------------------------------------------------------------------------------------- schema helpers

def schema_kind(schema: Any) -> str:
    """detect | classify | generate, from the structured-output schema a client sent."""
    props = (schema or {}).get("properties") or {} if isinstance(schema, dict) else {}
    objs = props.get("objects")
    if isinstance(objs, dict) and "label" in ((objs.get("items") or {}).get("properties") or {}):
        return "detect"
    items = props.get("items")
    if isinstance(items, dict) and {"index", "confidence"} <= set(((items.get("items") or {}).get("properties") or {})):
        return "classify"
    return "generate"


def synth(schema: Any, depth: int = 0) -> Any:
    """A value that satisfies ``schema`` (first enum value, minimum, one array item, every property)."""
    if not isinstance(schema, dict) or depth > 12:
        return None
    if "enum" in schema and schema["enum"]:
        return next((v for v in schema["enum"] if v is not None), schema["enum"][0])
    if "const" in schema:
        return schema["const"]
    if "anyOf" in schema and schema["anyOf"]:
        return synth(next((s for s in schema["anyOf"] if s.get("type") != "null"), schema["anyOf"][0]), depth + 1)
    t = schema.get("type")
    if isinstance(t, list):
        t = next((x for x in t if x != "null"), "null")
    if t == "object" or "properties" in schema:
        return {k: synth(v, depth + 1) for k, v in (schema.get("properties") or {}).items()}
    if t == "array":
        n = max(1, int(schema.get("minItems") or 1))
        return [synth(schema.get("items") or {}, depth + 1) for _ in range(n)]
    if t == "integer":
        return int(schema.get("minimum", 0))
    if t == "number":
        return float(schema.get("minimum", 0.5))
    if t == "boolean":
        return True
    if t == "string":
        return "example"
    return None


def conform(value: Any, schema: Any) -> Any:
    """Shape a natural scripted answer like the wire schema the client sent: wrap under ``result``, turn dicts into
    ``[{key, value}]`` where the client encoded a map as pairs."""
    if not isinstance(schema, dict):
        return value
    props = schema.get("properties") or {}
    if schema.get("type") == "object" and list(props) == ["result"] and not (isinstance(value, dict)
                                                                           and set(value) == {"result"}):
        return {"result": conform(value, props["result"])}
    if schema.get("type") == "array" and isinstance(value, dict):
        ip = ((schema.get("items") or {}).get("properties") or {})
        if set(ip) == {"key", "value"}:
            return [{"key": k, "value": conform(v, ip["value"])} for k, v in value.items()]
    if isinstance(value, dict) and props:
        return {k: conform(v, props.get(k, {})) for k, v in value.items()}
    if isinstance(value, list) and isinstance(schema.get("items"), dict):
        return [conform(v, schema["items"]) for v in value]
    return value


def _schema_problems(schema: Any, forbid=(), strict=False, path="$", structural=True) -> str:
    """'' when the schema is acceptable to the emulated API, else the first problem. ``structural=False`` checks only
    the forbidden keywords (Gemini takes any additionalProperties / minItems)."""
    if isinstance(schema, dict):
        for k in forbid:
            if k in schema:
                return f"{path}: keyword '{k}' is not supported"
        if structural and schema.get("minItems") not in (None, 0, 1) and not strict:
            return f"{path}: minItems > 1 is not supported"
        if structural and (schema.get("type") == "object"
                           or (isinstance(schema.get("type"), list) and "object" in schema["type"])):
            if schema.get("additionalProperties") is not False:
                return f"{path}: additionalProperties must be false"
            if strict and set(schema.get("required") or []) != set((schema.get("properties") or {}).keys()):
                return f"{path}: all properties must be required"
        for k, v in schema.items():
            if k in ("properties",) and isinstance(v, dict):
                for pk, pv in v.items():
                    p = _schema_problems(pv, forbid, strict, f"{path}.{pk}", structural)
                    if p:
                        return p
            elif k == "items":
                p = _schema_problems(v, forbid, strict, f"{path}[]", structural)
                if p:
                    return p
        if path == "$" and strict and schema.get("type") != "object":
            return "$: the root must be an object"
    return ""


# ------------------------------------------------------------------------------------------- the server

class FakeVLM:
    def __init__(self, host: str = "127.0.0.1", port: int = 0):
        self.lock = threading.Lock()
        self.requests: List[dict] = []
        self.faults: deque = deque()
        self.scenario: Dict[str, Any] = {}
        self.batches: Dict[str, dict] = {}
        self.files: Dict[str, bytes] = {}
        self.reset()
        fake = self

        class Handler(_Handler):
            server_fake = fake

        self.httpd = ThreadingHTTPServer((host, port), Handler)
        self.httpd.daemon_threads = True
        self.thread: Optional[threading.Thread] = None

    @property
    def url(self) -> str:
        h, p = self.httpd.server_address[:2]
        return f"http://{h}:{p}"

    def start(self) -> "FakeVLM":
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)
        self.thread.start()
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def reset(self) -> None:
        with self.lock:
            self.requests.clear()
            self.faults.clear()
            self.batches.clear()
            self.files.clear()
            self.scenario = {"boxes": None, "scale": None, "extra_items": [], "prose": False, "family": None,
                             "latency": 0.0, "thinking_tokens": 50, "reject_params": [], "classify": None,
                             "models": None, "duplicate": False, "generate": None, "generate_by_text": None, "loose": 0.0,
                             "refine_answer": None, "batch_polls": 1, "batch_expire": [], "batch_fail": None,
                             "batch_cancel_keep": 0, "gemini_batch_file": False}

    def set(self, **kw) -> None:
        with self.lock:
            self.scenario.update(kw)

    def push_fault(self, provider: str = "*", count: int = 1, **fault) -> None:
        with self.lock:
            for _ in range(max(1, count)):
                self.faults.append({"provider": provider, **fault})

    def take_fault(self, provider: str) -> Optional[dict]:
        with self.lock:
            for i, f in enumerate(self.faults):
                if f["provider"] in (provider, "*"):
                    del self.faults[i]
                    return f
        return None

    def calls(self, provider: Optional[str] = None, path: Optional[str] = None) -> List[dict]:
        with self.lock:
            return [r for r in self.requests if (provider is None or r["provider"] == provider)
                    and (path is None or path in r["path"])]


Reply = Tuple[int, Any, Dict[str, Any]]


class _Handler(BaseHTTPRequestHandler):
    server_fake: FakeVLM
    protocol_version = "HTTP/1.1"

    def log_message(self, *a):  # quiet
        pass

    # ---- plumbing
    def _send(self, status: int, body: Any, headers: Optional[dict] = None, ctype: str = "application/json") -> None:
        data = json.dumps(body).encode() if not isinstance(body, (bytes, str)) else (
            body.encode() if isinstance(body, str) else body)
        self.send_response(status)
        self.send_header("content-type", ctype)
        self.send_header("content-length", str(len(data)))
        for k, v in (headers or {}).items():
            self.send_header(k, str(v))
        self.end_headers()
        self.wfile.write(data)

    def _reply(self, r: Reply) -> None:
        self._send(r[0], r[1], r[2])

    def _raw_body(self) -> bytes:
        n = int(self.headers.get("content-length") or 0)
        return self.rfile.read(n) if n else b""

    def _body(self, raw: Optional[bytes] = None) -> Any:
        raw = self._raw_body() if raw is None else raw
        if not raw:
            return None
        try:
            return json.loads(raw)
        except ValueError:
            return None

    def _base(self) -> str:
        return f"http://{self.headers.get('Host') or '127.0.0.1'}"

    def _record(self, provider: str, body: Any, images: List[Tuple[int, int]], **extra) -> dict:
        fake = self.server_fake
        hdrs = {k.lower(): v for k, v in self.headers.items()}
        rec = {"provider": provider, "method": self.command, "path": self.path, "query": parse_qs(urlparse(self.path).query),
               "header_names": sorted(hdrs), "auth": {k: _redact(hdrs.get(k)) for k in
                                                      ("x-api-key", "x-goog-api-key", "authorization") if k in hdrs},
               "anthropic_beta": hdrs.get("anthropic-beta"), "body": _strip_images(body), "images": images,
               "t": time.time(), **extra}
        with fake.lock:
            fake.requests.append(rec)
        return rec

    def do_GET(self):
        self._route("GET")

    def do_POST(self):
        self._route("POST")

    def _route(self, method: str) -> None:
        fake = self.server_fake
        path = urlparse(self.path).path
        raw = self._raw_body() if method == "POST" else b""
        ctype = self.headers.get("content-type") or ""
        body = self._body(raw) if method == "POST" and "multipart" not in ctype else None
        try:
            if path.startswith("/_fake/"):
                return self._control(path, body)
            lat = float(fake.scenario.get("latency") or 0)
            if lat:
                time.sleep(lat)
            # Anthropic
            if path == "/v1/messages" and method == "POST":
                return self._reply(self._anthropic(body, check=True))
            if path == "/v1/messages/batches" and method == "POST":
                return self._anthropic_batch_create(body)
            m = re.fullmatch(r"/v1/messages/batches/([\w-]+)(/results|/cancel)?", path)
            if m:
                return self._anthropic_batch(m.group(1), m.group(2) or "", method)
            if path == "/v1/models" and "x-api-key" in {k.lower() for k in self.headers.keys()}:
                return self._anthropic_models()
            # Gemini
            if path.startswith("/v1beta/models") and path.endswith(":generateContent") and method == "POST":
                return self._gemini_http(body, path)
            if path.startswith("/v1beta/models") and path.endswith(":batchGenerateContent") and method == "POST":
                return self._gemini_batch_create(body, path)
            m = re.fullmatch(r"/v1beta/batches/([\w-]+)(:cancel)?", path)
            if m:
                return self._gemini_batch(m.group(1), m.group(2) or "", method)
            m = re.fullmatch(r"/download/v1beta/files/([\w-]+):download", path)
            if m:
                return self._gemini_download(m.group(1))
            if path == "/v1beta/models" and method == "GET":
                return self._gemini_models()
            # OpenAI
            if path == "/v1/responses" and method == "POST":
                return self._openai_http(body)
            if path == "/v1/files" and method == "POST":
                return self._openai_upload(raw, ctype)
            m = re.fullmatch(r"/v1/files/([\w-]+)/content", path)
            if m:
                return self._openai_file(m.group(1))
            if path == "/v1/batches" and method == "POST":
                return self._openai_batch_create(body)
            m = re.fullmatch(r"/v1/batches/([\w-]+)(/cancel)?", path)
            if m:
                return self._openai_batch(m.group(1), m.group(2) or "", method)
            # OpenAI-compatible / Ollama
            if path == "/v1/chat/completions" and method == "POST":
                return self._compat(body)
            if path == "/v1/models" and method == "GET":
                return self._openai_models()
            if path == "/api/tags":
                return self._ollama_tags()
            if path == "/api/show" and method == "POST":
                return self._ollama_show(body)
            self._send(404, {"error": {"message": f"no route {method} {path}"}})
        except Exception as e:  # noqa: BLE001
            self._send(500, {"error": {"type": "api_error", "message": f"fake crashed: {type(e).__name__}: {e}"}})

    def _control(self, path: str, body: Any) -> None:
        fake = self.server_fake
        if path == "/_fake/scenario":
            fake.set(**(body or {}))
            return self._send(200, {"ok": True, "scenario": fake.scenario})
        if path == "/_fake/fault":
            b = dict(body or {})
            fake.push_fault(b.pop("provider", "*"), b.pop("count", 1), **b)
            return self._send(200, {"ok": True, "faults": len(fake.faults)})
        if path == "/_fake/reset":
            fake.reset()
            return self._send(200, {"ok": True})
        if path == "/_fake/requests":
            return self._send(200, {"requests": fake.calls()})
        if path == "/_fake/batches":
            with fake.lock:
                return self._send(200, {"batches": {k: {kk: vv for kk, vv in v.items() if kk != "results"}
                                                    for k, v in fake.batches.items()}})
        self._send(404, {"error": "unknown control"})

    # ---- vision
    @staticmethod
    def _see(b64: str) -> Tuple[Any, Tuple[int, int]]:
        from PIL import Image
        raw = base64.b64decode(b64)
        im = Image.open(io.BytesIO(raw))
        im.load()
        return im.convert("RGB"), im.size

    def _objects(self, im, text: str = "") -> List[Tuple[str, Tuple[float, float, float, float]]]:
        """[(label, (x1, y1, x2, y2) as fractions of the received image)]."""
        sc = self.server_fake.scenario
        W, H = im.size
        refine = REFINE_MARK in text
        if refine and sc.get("refine_answer"):
            lab, (xc, yc, w, h) = sc["refine_answer"]["label"], sc["refine_answer"]["box"]
            return [(lab, (xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2))]
        if sc.get("boxes") and not refine:
            out = []
            # a ruler margin: scripted boxes are fractions of the photo, which starts at (x0, y0) of the 0-1000 frame
            m = re.search(r"photo occupies x (\d+)-1000 and y (\d+)-1000", text or "")
            fx, fy = (int(m.group(1)) / 1000, int(m.group(2)) / 1000) if m else (0.0, 0.0)
            for b in sc["boxes"]:
                xc, yc, w, h = b["box"]
                x1, y1, x2, y2 = xc - w / 2, yc - h / 2, xc + w / 2, yc + h / 2
                out.append((b["label"], (fx + x1 * (1 - fx), fy + y1 * (1 - fy), fx + x2 * (1 - fx),
                                         fy + y2 * (1 - fy))))
            return out
        found = [(lab, (x1 / W, y1 / H, x2 / W, y2 / H)) for lab, (x1, y1, x2, y2) in find_blocks(im)]
        loose = float(sc.get("loose") or 0)
        if loose and not refine:
            # a sloppy first pass: every box grown by `loose` x its size, off-centre (what refine is meant to fix)
            out = []
            for lab, (x1, y1, x2, y2) in found:
                w, h = x2 - x1, y2 - y1
                out.append((lab, (max(0.0, x1 - w * loose * 0.3), max(0.0, y1 - h * loose * 0.3),
                                  min(1.0, x2 + w * loose * 0.7), min(1.0, y2 + h * loose * 0.7))))
            return out
        return found

    def _classify_crop(self, im) -> Tuple[str, str]:
        import numpy as np
        sc = self.server_fake.scenario
        if sc.get("classify"):
            return sc["classify"], "scripted"
        W, H = im.size
        a = np.asarray(im.crop((W // 4, H // 4, max(W // 4 + 1, 3 * W // 4), max(H // 4 + 1, 3 * H // 4))))
        a = a.astype(np.int16)
        best, n_best = "none", 0
        for rgb, label in PALETTE.items():
            n = int((np.abs(a - np.array(rgb, dtype=np.int16)) < TOL).all(axis=2).sum())
            if n > n_best:
                best, n_best = label, n
        return (best if n_best >= max(4, a.shape[0] * a.shape[1] // 10) else "none"), "colour block"

    def _generate(self, schema: Any, crops: List[Any], text: str = "") -> str:
        sc = self.server_fake.scenario
        ans = sc.get("generate")
        by = sc.get("generate_by_text") or {}
        hits = [(k, v) for k, v in by.items() if k in (text or "")]
        if hits:
            # scripted per prompt text (e.g. {"Class 1.": {...}}); several hits fill a "classes" list (one per hit)
            props = (schema or {}).get("properties") or {}
            if "classes" in props:
                ans = {"classes": [dict(v, cls=int(re.search(r"\d+", k).group())) if re.search(r"\d+", k) else v
                                   for k, v in hits]}
            else:
                ans = hits[0][1]
        if ans is not None:
            if isinstance(ans, dict) and "_by_images" in ans:      # scripted per number of attached images
                ans = ans["_by_images"].get(str(len(crops)), ans.get("_default"))
            return json.dumps(conform(ans, schema))
        return json.dumps(synth(schema))

    def _render(self, objs, frame: str, sw: int, sh: int, *, labels_ok=None) -> List[dict]:
        """Objects (fractions of the received image) → items in ``frame``: anthropic | box_2d | qwen3 | qwen25 | norm."""
        sc = self.server_fake.scenario
        scale = sc.get("scale")       # force a wrong scale: "k1000" | "unit" | "pixel"
        items = []
        for lab, (x1, y1, x2, y2) in objs:
            if frame == "anthropic":
                if scale == "k1000":
                    v = [round(x1 * 1000), round(y1 * 1000), round(x2 * 1000), round(y2 * 1000)]
                elif scale == "unit":
                    v = [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]
                else:
                    v = [round(x1 * sw), round(y1 * sh), round(x2 * sw), round(y2 * sh)]
                items.append({"label": lab, "x1": v[0], "y1": v[1], "x2": v[2], "y2": v[3]})
            elif frame == "box_2d":
                if scale == "pixel":
                    v = [round(y1 * sh), round(x1 * sw), round(y2 * sh), round(x2 * sw)]
                elif scale == "unit":
                    v = [round(y1, 4), round(x1, 4), round(y2, 4), round(x2, 4)]
                else:
                    v = [round(y1 * 1000), round(x1 * 1000), round(y2 * 1000), round(x2 * 1000)]
                items.append({"label": lab, "box_2d": v})
            elif frame == "qwen3":
                if scale == "pixel":
                    v = [round(x1 * sw), round(y1 * sh), round(x2 * sw), round(y2 * sh)]
                else:
                    v = [round(x1 * 1000), round(y1 * 1000), round(x2 * 1000), round(y2 * 1000)]
                items.append({"label": lab, "bbox_2d": v})
            elif frame == "qwen25":
                rw, rh = smart_resize(sw, sh)       # the model's processor resizes what it got
                items.append({"label": lab, "bbox_2d": [round(x1 * rw), round(y1 * rh), round(x2 * rw), round(y2 * rh)]})
            elif frame == "norm":
                items.append({"label": lab, "bbox": [round(x1, 4), round(y1, 4), round(x2, 4), round(y2, 4)]})
        if sc.get("duplicate") and items:
            items.append(dict(items[0]))
        items.extend(sc.get("extra_items") or [])
        return items

    @staticmethod
    def _truncate(text: str, keep: int) -> str:
        """Cut a JSON reply in the middle of item ``keep`` (0-based)."""
        starts = [m.start() for m in re.finditer(r"\{\"label\"", text)]
        if keep < len(starts):
            cut = starts[keep] + 12
            return text[:cut]
        return text[:-3]

    @staticmethod
    def _fault_reply(provider: str, f: dict) -> Optional[Reply]:
        """An injected HTTP failure as (status, body, headers), or None when the fault is not an HTTP error (e.g.
        ``refusal`` / ``truncate`` / ``text`` change the answer instead; ``pass`` lets one request through)."""
        kind = f.get("kind")
        if "status" in f:
            return int(f["status"]), f.get("body") or {"error": {"message": "injected"}}, f.get("headers") or {}
        hdr = {"retry-after": f["retry_after"]} if f.get("retry_after") is not None else {}
        if kind == "rate_limit":
            body = {"anthropic": {"type": "error", "error": {"type": "rate_limit_error", "message": "Number of requests has exceeded your rate limit"}},
                    "gemini": {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Resource has been exhausted (e.g. check quota).",
                                         "details": ([{"@type": "type.googleapis.com/google.rpc.RetryInfo", "retryDelay": f"{f['retry_after']}s"}]
                                                     if f.get("retry_after") is not None else [])}},
                    "openai": {"error": {"message": "Rate limit reached for requests", "type": "requests", "code": "rate_limit_exceeded"}},
                    "compat": {"error": {"message": "Too many requests"}}}[provider]
            return 429, body, ({} if provider == "gemini" else hdr)
        if kind == "overloaded":
            if provider == "anthropic":
                return 529, {"type": "error", "error": {"type": "overloaded_error", "message": "Overloaded"}}, hdr
            return 503, {"error": {"code": 503, "status": "UNAVAILABLE", "message": "The model is overloaded."}}, hdr
        if kind == "server_error":
            return 500, {"type": "error", "error": {"type": "api_error", "message": "Internal server error"}}, {}
        if kind == "auth":
            if provider == "gemini":
                return 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key.",
                                       "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}}, {}
            if provider == "anthropic":
                return 401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}, {}
            return 401, {"error": {"message": "Incorrect API key provided: sk-proj-****0001", "type": "invalid_request_error", "code": "invalid_api_key"}}, {}
        if kind == "spend_limit":
            if provider == "anthropic":
                return 429, {"type": "error", "error": {"type": "rate_limit_error", "message": "You have reached your specified workspace spend limit.",
                                                        "details": {"error_code": "enforced_spend_limit_reached"}}}, {}
            if provider == "openai":
                return 429, {"error": {"message": "You exceeded your current quota, please check your plan and billing details.",
                                       "type": "insufficient_quota", "code": "insufficient_quota"}}, {}
            return 429, {"error": {"code": 429, "status": "RESOURCE_EXHAUSTED", "message": "Your prepayment credits are depleted. Check your billing."}}, {}
        if kind == "leaky_error":
            # an upstream error that echoes the caller's key back (some proxies do): the client must scrub it
            k = KEYS.get(provider) or "sk-compat-LEAKEDleakedLEAKED01"
            return 400, {"error": {"type": "invalid_request_error", "message": f"bad request for key {k} at ?key={k}"}}, {}
        return None

    # ---- Anthropic
    def _anthropic_models(self):
        if self.headers.get("x-api-key") != KEYS["anthropic"]:
            return self._send(401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}})
        self._record("anthropic", None, [])
        models = self.server_fake.scenario.get("models") or ["claude-opus-5-5", "claude-sonnet-5-5", "claude-haiku-5-5",
                                                             "claude-sonnet-9-9"]
        data = [{"type": "model", "id": m, "display_name": m.replace("-", " ").title(), "created_at": "2026-09-01T00:00:00Z",
                 "lifecycle": "active", "capabilities": {"image_input": {"supported": True}}} for m in models]
        self._send(200, {"data": data, "has_more": False, "first_id": models[0], "last_id": models[-1]})

    def _anthropic_auth(self) -> Optional[Reply]:
        if self.headers.get("x-api-key") != KEYS["anthropic"]:
            return 401, {"type": "error", "error": {"type": "authentication_error", "message": "invalid x-api-key"}}, {}
        if not self.headers.get("anthropic-version"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "anthropic-version header is required"}}, {}
        return None

    def _anthropic(self, body: dict, check: bool = True, batch: str = "") -> Reply:
        """One Messages API request → (status, body, headers). ``check``: validate auth headers (not for batch
        items, whose auth was checked when the batch was created)."""
        fake = self.server_fake
        if check:
            bad = self._anthropic_auth()
            if bad:
                self._record("anthropic", body, [])
                return bad
        model = body.get("model", "")
        images, crops = [], []
        content = body["messages"][0]["content"]
        if isinstance(content, str):                       # the API also takes a bare string as the user turn
            content = [{"type": "text", "text": content}]
        text = " ".join(str(b.get("text", "")) for b in content if isinstance(b, dict) and b.get("type") == "text")
        for blk in content:
            if isinstance(blk, dict) and blk.get("type") == "image":
                im, size = self._see(blk["source"]["data"])
                max_edge, max_tokens = STANDARD if STANDARD_MODELS.search(model) else HIGH
                target = resized_size(size[0], size[1], max_edge, max_tokens)
                if target != size:
                    if (blk.get("transformations") or {}).get("oversized_image") == "error":
                        self._record("anthropic", body, [size], batch=batch)
                        return 400, {"type": "error", "error": {"type": "invalid_request_error",
                                     "message": f"image exceeds the model's size limit; resize to {target[0]}x{target[1]}"}}, {}
                    from PIL import Image
                    im = im.resize(target, Image.BILINEAR)     # what Claude itself would do
                    size = target
                images.append(size)
                crops.append(im)
        self._record("anthropic", body, images, batch=batch, text=text[:2000])
        f = fake.take_fault("anthropic")
        if f:
            r = self._fault_reply("anthropic", f)
            if r:
                return r
        # request validation the real API performs
        for p in ("temperature", "top_p", "top_k"):
            if p in body and NO_SAMPLING.search(model):
                return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": f"{p}: Extra inputs are not permitted for this model"}}, {}
        th = body.get("thinking") or {}
        if th.get("type") in ("disabled", "enabled") and model in ("claude-opus-5-5", "claude-sonnet-5-5", "claude-fable-5-1"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": f'"thinking.type.{th.get("type")}" is not supported for this model.'}}, {}
        if "fallbacks" in body:
            beta = self.headers.get("anthropic-beta") or ""
            if batch:
                return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks: not supported in the Message Batches API"}}, {}
            if body["fallbacks"] == "default" and "server-side-fallback-2026-07-01" not in beta:
                return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "fallbacks: requires the server-side-fallback-2026-07-01 beta"}}, {}
            if "fallback" in (fake.scenario.get("reject_params") or []):
                return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "Unexpected value(s) `server-side-fallback-2026-07-01` for the `anthropic-beta` header."}}, {}
        oc = body.get("output_config") or {}
        if "effort" in (fake.scenario.get("reject_params") or []) and "effort" in oc:
            return 400, {"type": "error", "error": {"type": "invalid_request_error",
                         "message": "output_config.effort: effort is not supported for this model"}}, {}
        fmt = oc.get("format") or {}
        if fmt.get("type") != "json_schema" or not isinstance(fmt.get("schema"), dict):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "output_config.format.schema required"}}, {}
        bad = _schema_problems(fmt["schema"], forbid=("minimum", "maximum", "maxItems", "minLength", "maxLength",
                                                      "default"), strict=False)
        if bad:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": f"output_config.format.schema: {bad}"}}, {}
        if oc.get("effort") not in (None, "low", "medium", "high", "xhigh", "max"):
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "output_config.effort invalid"}}, {}
        if oc.get("effort") and "haiku-4-5" in model:
            return 400, {"type": "error", "error": {"type": "invalid_request_error", "message": "effort is not supported for this model"}}, {}
        served = model
        iterations = None
        if f and f.get("kind") == "refusal":
            if body.get("fallbacks") and model in FALLBACK_MODELS and f.get("fallback", True):
                served = FALLBACK_MODELS[model]
                iterations = [{"type": "message", "model": model, "input_tokens": 100, "output_tokens": 0,
                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0}]
            else:
                return 200, _anthropic_msg(model, [], "refusal", 120, 0,
                                           stop_details={"type": "refusal", "category": "cyber", "explanation": "declined"}), {}
        kind = schema_kind(fmt["schema"])
        if kind == "classify":
            answers = []
            for i, im in enumerate(crops):
                lab, why = self._classify_crop(im)
                answers.append({"index": i + 1, "label": lab, "confidence": "high", "reason": why})
            out_text = json.dumps({"items": answers})
        elif kind == "generate":
            out_text = self._generate(fmt["schema"], crops, text)
        else:
            objs = self._objects(crops[0], text) if crops else []
            items = self._render(objs, "anthropic", images[0][0], images[0][1]) if crops else []
            out_text = json.dumps({"objects": items})
        if f and f.get("kind") == "text":
            out_text = f["text"]
        stop = "end_turn"
        if f and f.get("kind") == "truncate" and (body.get("max_tokens", 0) <= int(f.get("below", 10 ** 9))):
            out_text, stop = self._truncate(out_text, int(f.get("keep", 1))), "max_tokens"
        tin = sum(math.ceil(w / 28) * math.ceil(h / 28) for w, h in images) + anthropic_text_tokens(body)
        tout = len(out_text) // 4 + int(fake.scenario.get("thinking_tokens") or 0)
        content_out = [{"type": "thinking", "thinking": "", "signature": "sig"}, {"type": "text", "text": out_text}]
        msg = _anthropic_msg(served, content_out, stop, tin, tout)
        if iterations is not None:
            iterations.append({"type": "fallback_message", "model": served, "input_tokens": tin, "output_tokens": tout,
                               "cache_read_input_tokens": 0, "cache_creation_input_tokens": 0})
            msg["usage"]["iterations"] = iterations
            msg["usage"]["input_tokens"] = tin + 100
        return 200, msg, {"request-id": "req_fake"}

    # Message Batches: every request is answered at creation time; status polls reveal them
    def _anthropic_batch_create(self, body: dict) -> None:
        fake = self.server_fake
        bad = self._anthropic_auth()
        self._record("anthropic", {"requests": len((body or {}).get("requests") or [])}, [], batch="create")
        if bad:
            return self._reply(bad)
        f = fake.take_fault("anthropic_batch")
        if f:
            r = self._fault_reply("anthropic", f)
            if r:
                return self._reply(r)
        reqs = (body or {}).get("requests") or []
        if not reqs:
            return self._send(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "requests: at least one request is required"}})
        ids = [r.get("custom_id") for r in reqs]
        if len(set(ids)) != len(ids):
            return self._send(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "custom_id values must be unique within a batch"}})
        for cid in ids:
            if not isinstance(cid, str) or not CUSTOM_ID.match(cid):
                return self._send(400, {"type": "error", "error": {"type": "invalid_request_error", "message": f"custom_id {cid!r} does not match ^[a-zA-Z0-9_-]{{1,64}}$"}})
        with fake.lock:
            bid = f"msgbatch_fake{len(fake.batches) + 1:04d}"
        results = []
        for r in reqs:
            cid = r["custom_id"]
            if cid in (fake.scenario.get("batch_expire") or []):
                results.append({"custom_id": cid, "result": {"type": "expired"}})
                continue
            status, out, _h = self._anthropic(r.get("params") or {}, check=False, batch=bid)
            if status == 200:
                results.append({"custom_id": cid, "result": {"type": "succeeded", "message": out}})
            else:
                err = out if isinstance(out, dict) and out.get("type") == "error" else \
                    {"type": "error", "error": {"type": "api_error", "message": str(out)}}
                results.append({"custom_id": cid, "result": {"type": "errored", "error": err}})
        with fake.lock:
            fake.batches[bid] = {"provider": "anthropic", "polls": 0, "status": "in_progress", "results": results,
                                 "n": len(reqs), "canceled": False, "created": time.time()}
        self._send(200, self._anthropic_batch_obj(bid))

    def _anthropic_batch_obj(self, bid: str) -> dict:
        b = self.server_fake.batches[bid]
        ended = b["status"] == "ended"
        counts = {"processing": 0 if ended else b["n"], "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        if ended:
            for r in b["results"]:
                counts[r["result"]["type"]] += 1
        return {"id": bid, "type": "message_batch", "processing_status": b["status"], "request_counts": counts,
                "created_at": "2026-10-13T00:00:00Z", "expires_at": "2026-10-14T00:00:00Z",
                "ended_at": "2026-10-13T00:10:00Z" if ended else None, "archived_at": None,
                "cancel_initiated_at": "2026-10-13T00:05:00Z" if b["canceled"] else None,
                "results_url": f"{self._base()}/v1/messages/batches/{bid}/results" if ended else None}

    def _anthropic_batch(self, bid: str, action: str, method: str) -> None:
        fake = self.server_fake
        bad = self._anthropic_auth()
        if bad:
            return self._reply(bad)
        with fake.lock:
            b = fake.batches.get(bid)
        self._record("anthropic", None, [], batch=f"{action or 'get'}:{bid}")
        if b is None:
            return self._send(404, {"type": "error", "error": {"type": "not_found_error", "message": f"batch {bid} not found"}})
        f = fake.take_fault("anthropic_batch")
        if f:
            r = self._fault_reply("anthropic", f)
            if r:
                return self._reply(r)
        if action == "/cancel" and method == "POST":
            with fake.lock:
                if b["status"] != "ended":
                    b["canceled"] = True
                    b["status"] = "canceling"
                    keep = int(fake.scenario.get("batch_cancel_keep") or 0)
                    for r in b["results"][keep:]:
                        r["result"] = {"type": "canceled"}
            return self._send(200, self._anthropic_batch_obj(bid))
        if action == "/results":
            if b["status"] != "ended":
                return self._send(400, {"type": "error", "error": {"type": "invalid_request_error", "message": "batch has not ended"}})
            lines = "".join(json.dumps(r) + "\n" for r in reversed(b["results"]))     # order is not guaranteed
            return self._send(200, lines.encode(), ctype="application/binary")
        with fake.lock:
            b["polls"] += 1
            if b["status"] == "canceling" or b["polls"] >= int(fake.scenario.get("batch_polls") or 1):
                b["status"] = "ended"
        self._send(200, self._anthropic_batch_obj(bid))

    # ---- Gemini
    def _gemini_models(self):
        if self.headers.get("x-goog-api-key") != KEYS["gemini"]:
            return self._send(400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid.",
                                              "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}})
        self._record("gemini", None, [])
        ms = self.server_fake.scenario.get("models") or ["gemini-3.8-flash", "gemini-3.5-flash-lite", "gemini-embedding-001", "gemini-9.0-ultra"]
        self._send(200, {"models": [{"name": f"models/{m}", "displayName": m, "supportedGenerationMethods":
                                     ["embedContent"] if "embedding" in m else ["generateContent", "countTokens"]} for m in ms]})

    def _gemini_auth(self) -> Optional[Reply]:
        q = parse_qs(urlparse(self.path).query)
        if "key" in q:
            return 400, {"error": {"code": 400, "message": "fake: API key must not be sent in the URL"}}, {}
        if self.headers.get("x-goog-api-key") != KEYS["gemini"]:
            return 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message": "API key not valid. Please pass a valid API key.",
                                   "details": [{"@type": "type.googleapis.com/google.rpc.ErrorInfo", "reason": "API_KEY_INVALID"}]}}, {}
        return None

    def _gemini_http(self, body: dict, path: str) -> None:
        bad = self._gemini_auth()
        if bad:
            self._record("gemini", body, [])
            return self._reply(bad)
        model = path.split("/models/", 1)[1].split(":", 1)[0]
        self._reply(self._gemini(body, model))

    def _gemini(self, body: dict, model: str, batch: str = "") -> Reply:
        fake = self.server_fake
        parts = body["contents"][0]["parts"]
        crops, images = [], []
        text = " ".join(str(p.get("text", "")) for p in parts if "text" in p)
        for p in parts:
            if "inlineData" in p:
                im, size = self._see(p["inlineData"]["data"])
                crops.append(im)
                images.append(size)
        self._record("gemini", body, images, batch=batch, text=text[:2000])
        f = fake.take_fault("gemini")
        if f:
            r = self._fault_reply("gemini", f)
            if r:
                return r
        gc = body.get("generationConfig") or {}
        rej = fake.scenario.get("reject_params") or []
        if "mediaResolution" in rej and any("mediaResolution" in p for p in parts):
            return 400, {"error": {"code": 400, "status": "INVALID_ARGUMENT", "message":
                         'Invalid JSON payload received. Unknown name "mediaResolution" at \'contents[0].parts[0]\': Cannot find field.'}}, {}
        if gc.get("responseMimeType") != "application/json" or not isinstance(gc.get("responseJsonSchema"), dict):
            return 400, {"error": {"code": 400, "message": "responseMimeType/responseJsonSchema required by fake"}}, {}
        bad = _schema_problems(gc["responseJsonSchema"], forbid=("pattern", "default", "minLength", "maxLength"),
                               structural=False)
        if bad:
            return 400, {"error": {"code": 400, "message": f"responseJsonSchema: {bad}"}}, {}
        lvl = (gc.get("thinkingConfig") or {}).get("thinkingLevel")
        if lvl == "MINIMAL" and model.startswith("gemini-3.8"):
            return 400, {"error": {"code": 400, "message": "thinking level MINIMAL is not supported for this model"}}, {}
        if lvl and model.startswith("gemini-2"):
            return 400, {"error": {"code": 400, "message": "thinkingLevel is not supported for Gemini 2.5; use thinkingBudget"}}, {}
        if "temperature" in gc:
            return 400, {"error": {"code": 400, "message": "fake: temperature must not be sent to Gemini 3"}}, {}
        if f and f.get("kind") == "refusal":
            return 200, {"candidates": [{"content": {"parts": [], "role": "model"}, "finishReason": "SAFETY"}],
                         "usageMetadata": {"promptTokenCount": 1200, "candidatesTokenCount": 0}, "modelVersion": model}, {}
        schema = gc["responseJsonSchema"]
        kind = schema_kind(schema)
        if kind == "classify":
            answers = [{"index": i + 1, "label": self._classify_crop(im)[0], "confidence": "medium", "reason": "colour"}
                       for i, im in enumerate(crops)]
            out_text = json.dumps({"items": answers})
        elif kind == "generate":
            out_text = self._generate(schema, crops, text)
        else:
            items = self._render(self._objects(crops[0], text), "box_2d", *images[0])
            out_text = json.dumps({"objects": items})
        if f and f.get("kind") == "text":
            out_text = f["text"]
        finish = "STOP"
        if f and f.get("kind") == "truncate" and gc.get("maxOutputTokens", 0) <= int(f.get("below", 10 ** 9)):
            out_text, finish = self._truncate(out_text, int(f.get("keep", 1))), "MAX_TOKENS"
        res_level = None
        for p in parts:
            if "mediaResolution" in p:
                res_level = p["mediaResolution"].get("level")
        tin = {"MEDIA_RESOLUTION_LOW": 280, "MEDIA_RESOLUTION_MEDIUM": 560, "MEDIA_RESOLUTION_ULTRA_HIGH": 2240}.get(res_level, 1120) * len(images) + 200
        thoughts = int(fake.scenario.get("thinking_tokens") or 0)
        return 200, {"candidates": [{"content": {"parts": [{"text": "thinking…", "thought": True}, {"text": out_text}], "role": "model"},
                                     "finishReason": finish}],
                     "usageMetadata": {"promptTokenCount": tin, "candidatesTokenCount": len(out_text) // 4, "thoughtsTokenCount": thoughts,
                                       "totalTokenCount": tin + len(out_text) // 4 + thoughts},
                     "modelVersion": model}, {}

    def _gemini_batch_create(self, body: dict, path: str) -> None:
        fake = self.server_fake
        bad = self._gemini_auth()
        self._record("gemini", {"batch": "create"}, [], batch="create")
        if bad:
            return self._reply(bad)
        f = fake.take_fault("gemini_batch")
        if f:
            r = self._fault_reply("gemini", f)
            if r:
                return self._reply(r)
        model = path.split("/models/", 1)[1].split(":", 1)[0]
        b = (body or {}).get("batch") or {}
        inp = b.get("inputConfig") or b.get("input_config") or {}
        reqs = ((inp.get("requests") or {}).get("requests")) or []
        raw = json.dumps(body).encode()
        if len(raw) > 20 * 1024 * 1024:
            return self._send(400, {"error": {"code": 400, "message": "inline batch requests are limited to 20 MB"}})
        with fake.lock:
            bid = f"fakebatch{len(fake.batches) + 1:04d}"
        results = []
        for r in reqs:
            key = (r.get("metadata") or {}).get("key")
            status, out, _h = self._gemini(r.get("request") or {}, model, batch=bid)
            line = {"metadata": {"key": key}} if key is not None else {}
            if status == 200:
                line["response"] = out
            else:
                err = (out or {}).get("error") or {"code": status, "message": "error"}
                line["error"] = {"code": err.get("code", status), "message": err.get("message", "error")}
            results.append(line)
        with fake.lock:
            fake.batches[bid] = {"provider": "gemini", "polls": 0, "status": "RUNNING", "results": results,
                                 "n": len(reqs), "model": model, "canceled": False}
        self._send(200, self._gemini_batch_obj(bid))

    def _gemini_batch_obj(self, bid: str) -> dict:
        fake = self.server_fake
        b = fake.batches[bid]
        st = b["status"]
        done = st in ("SUCCEEDED", "CANCELLED", "FAILED", "EXPIRED")
        ok = sum(1 for r in b["results"] if "response" in r) if done else 0
        bad = sum(1 for r in b["results"] if "error" in r) if done else 0
        op = {"name": f"batches/{bid}",
              "metadata": {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatch",
                           "model": f"models/{b['model']}", "name": f"batches/{bid}", "state": f"BATCH_STATE_{st}",
                           "batchStats": {"requestCount": str(b["n"]), "successfulRequestCount": str(ok),
                                          "failedRequestCount": str(bad),
                                          "pendingRequestCount": str(0 if done else b["n"])}},
              "done": done}
        if st == "SUCCEEDED" or (st == "CANCELLED" and b["results"]):
            if fake.scenario.get("gemini_batch_file"):
                fid = f"files/out-{bid}"
                fake.files[fid] = "".join(json.dumps({"key": r.get("metadata", {}).get("key"),
                                                      **{k: v for k, v in r.items() if k != "metadata"}}) + "\n"
                                          for r in b["results"]).encode()
                op["response"] = {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatchOutput",
                                  "responsesFile": fid}
            else:
                op["response"] = {"@type": "type.googleapis.com/google.ai.generativelanguage.v1main.GenerateContentBatchOutput",
                                  "inlinedResponses": {"inlinedResponses": list(b["results"])}}
        if st == "FAILED":
            op["error"] = {"code": 3, "message": "batch failed: invalid request file"}
        return op

    def _gemini_batch(self, bid: str, action: str, method: str) -> None:
        fake = self.server_fake
        bad = self._gemini_auth()
        if bad:
            return self._reply(bad)
        self._record("gemini", None, [], batch=f"{action or 'get'}:{bid}")
        with fake.lock:
            b = fake.batches.get(bid)
        if b is None:
            return self._send(404, {"error": {"code": 404, "message": f"batches/{bid} not found", "status": "NOT_FOUND"}})
        if action == ":cancel":
            with fake.lock:
                if b["status"] in ("PENDING", "RUNNING"):
                    keep = int(fake.scenario.get("batch_cancel_keep") or 0)
                    b["results"] = b["results"][:keep]
                    b["status"] = "CANCELLED"
            return self._send(200, {})
        with fake.lock:
            b["polls"] += 1
            if b["status"] in ("PENDING", "RUNNING") and b["polls"] >= int(fake.scenario.get("batch_polls") or 1):
                b["status"] = fake.scenario.get("batch_fail") or "SUCCEEDED"
        self._send(200, self._gemini_batch_obj(bid))

    def _gemini_download(self, fid: str) -> None:
        bad = self._gemini_auth()
        self._record("gemini", None, [], batch="download")
        if bad:
            return self._reply(bad)
        data = self.server_fake.files.get(f"files/{fid}")
        if data is None:
            return self._send(404, {"error": {"code": 404, "message": "file not found"}})
        self._send(200, data, ctype="application/jsonl")

    # ---- OpenAI
    def _openai_models(self):
        auth = self.headers.get("authorization") or ""
        if auth.startswith("Bearer ") and auth[7:] == KEYS["openai"]:
            self._record("openai", None, [])
            ms = self.server_fake.scenario.get("models") or ["gpt-6.1-sol", "gpt-6-luna", "gpt-5.5", "gpt-realtime", "text-embedding-3-large", "gpt-7-preview"]
            return self._send(200, {"object": "list", "data": [{"id": m, "object": "model"} for m in ms]})
        if auth.startswith("Bearer sk-proj-") or auth.startswith("Bearer sk-ant-"):
            self._record("openai", None, [])
            return self._send(401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error",
                                              "code": "invalid_api_key"}})
        # OpenAI-compatible servers (no / any key)
        self._record("compat", None, [])
        ms = self.server_fake.scenario.get("models") or ["qwen3.8:27b", "qwen2.5vl:7b", "llama3.2:3b"]
        if "models" in (self.server_fake.scenario.get("reject_params") or []):
            return self._send(404, {"error": {"message": "not found"}})
        self._send(200, {"object": "list", "data": [{"id": m, "object": "model", "owned_by": "library"} for m in ms]})

    def _openai_auth(self) -> Optional[Reply]:
        auth = self.headers.get("authorization") or ""
        if not auth.startswith("Bearer ") or auth[7:] != KEYS["openai"]:
            return 401, {"error": {"message": "Incorrect API key provided.", "type": "invalid_request_error", "code": "invalid_api_key"}}, {}
        return None

    def _openai_http(self, body: dict) -> None:
        bad = self._openai_auth()
        if bad:
            self._record("openai", body, [])
            return self._reply(bad)
        self._reply(self._openai(body))

    def _openai(self, body: dict, batch: str = "") -> Reply:
        fake = self.server_fake
        content = body["input"][0]["content"]
        crops, images = [], []
        text = " ".join(str(c.get("text", "")) for c in content if c.get("type") == "input_text")
        for c in content:
            if c.get("type") == "input_image":
                url = c["image_url"]
                im, size = self._see(url.split(",", 1)[1])
                crops.append(im)
                images.append(size)
        self._record("openai", body, images, batch=batch, text=text[:2000])
        f = fake.take_fault("openai")
        if f:
            r = self._fault_reply("openai", f)
            if r:
                return r
        fmt = ((body.get("text") or {}).get("format")) or {}
        if fmt.get("type") != "json_schema" or not fmt.get("strict") or not fmt.get("name"):
            return 400, {"error": {"message": "text.format must be a strict json_schema (fake)"}}, {}
        bad = _schema_problems(fmt["schema"], forbid=("default", "minLength", "maxLength"), strict=True)
        if bad:
            return 400, {"error": {"message": f"Invalid schema for response_format '{fmt['name']}': {bad}", "type": "invalid_request_error"}}, {}
        rej = fake.scenario.get("reject_params") or []
        if "reasoning" in rej and "reasoning" in body:
            return 400, {"error": {"message": "Unsupported parameter: 'reasoning.effort' is not supported with this model.",
                                   "type": "invalid_request_error", "param": "reasoning.effort"}}, {}
        if body.get("service_tier") == "flex" and batch:
            return 400, {"error": {"message": "service_tier is not supported in the Batch API", "type": "invalid_request_error"}}, {}
        if "service_tier" in rej and body.get("service_tier") == "flex":
            return 400, {"error": {"message": "Flex processing (service_tier 'flex') is not available for this model.",
                                   "type": "invalid_request_error", "param": "service_tier"}}, {}
        if "temperature" in body and str(body.get("model", "")).startswith("gpt-6"):
            return 400, {"error": {"message": "Unsupported parameter: 'temperature'"}}, {}
        model = body.get("model")
        if f and f.get("kind") == "refusal":
            return 200, {"id": "resp_fake", "object": "response", "status": "completed", "model": model,
                         "output": [{"type": "message", "role": "assistant", "content": [{"type": "refusal", "refusal": "I can't help with that."}]}],
                         "usage": {"input_tokens": 900, "output_tokens": 8}}, {}
        schema = fmt["schema"]
        kind = schema_kind(schema)
        if kind == "classify":
            out_text = json.dumps({"items": [{"index": i + 1, "label": self._classify_crop(im)[0], "confidence": "low", "reason": "colour"}
                                             for i, im in enumerate(crops)]})
        elif kind == "generate":
            out_text = self._generate(schema, crops, text)
        else:
            out_text = json.dumps({"objects": self._render(self._objects(crops[0], text), "box_2d", *images[0])})
        if f and f.get("kind") == "text":
            out_text = f["text"]
        status, inc = "completed", None
        if f and f.get("kind") == "truncate" and body.get("max_output_tokens", 0) <= int(f.get("below", 10 ** 9)):
            out_text, status, inc = self._truncate(out_text, int(f.get("keep", 1))), "incomplete", {"reason": "max_output_tokens"}
        tin = sum(math.ceil(math.ceil(w / 32) * math.ceil(h / 32) * 1.2) for w, h in images) + 150
        out = {"id": "resp_fake", "object": "response", "status": status, "model": model, "incomplete_details": inc,
               "output": [{"type": "reasoning", "summary": []},
                          {"type": "message", "role": "assistant", "content": [{"type": "output_text", "text": out_text, "annotations": []}]}],
               "usage": {"input_tokens": tin, "input_tokens_details": {"cached_tokens": 0},
                         "output_tokens": len(out_text) // 4 + 40, "output_tokens_details": {"reasoning_tokens": 40}}}
        out["service_tier"] = "default" if (fake.scenario.get("flex_full") or batch) else (body.get("service_tier") or "default")
        return 200, out, {}

    def _openai_upload(self, raw: bytes, ctype: str) -> None:
        from email.parser import BytesParser
        from email.policy import default as email_default
        fake = self.server_fake
        bad = self._openai_auth()
        self._record("openai", {"upload": len(raw)}, [], batch="upload")
        if bad:
            return self._reply(bad)
        if "multipart/form-data" not in ctype:
            return self._send(400, {"error": {"message": "files must be uploaded as multipart/form-data"}})
        msg = BytesParser(policy=email_default).parsebytes(b"Content-Type: " + ctype.encode() + b"\r\n\r\n" + raw)
        fields: Dict[str, Any] = {}
        for part in msg.iter_parts():
            name = part.get_param("name", header="content-disposition")
            fields[name] = part.get_payload(decode=True)
        if (fields.get("purpose") or b"").decode() != "batch" or not fields.get("file"):
            return self._send(400, {"error": {"message": "purpose=batch and a file are required"}})
        with fake.lock:
            fid = f"file-fake{len(fake.files) + 1:04d}"
            fake.files[fid] = fields["file"]
        self._send(200, {"id": fid, "object": "file", "bytes": len(fields["file"]), "purpose": "batch",
                         "filename": "fovea-batch.jsonl"})

    def _openai_file(self, fid: str) -> None:
        bad = self._openai_auth()
        if bad:
            return self._reply(bad)
        data = self.server_fake.files.get(fid)
        if data is None:
            return self._send(404, {"error": {"message": f"No such File object: {fid}"}})
        self._send(200, data, ctype="application/octet-stream")

    def _openai_batch_create(self, body: dict) -> None:
        fake = self.server_fake
        bad = self._openai_auth()
        self._record("openai", body, [], batch="create")
        if bad:
            return self._reply(bad)
        f = fake.take_fault("openai_batch")
        if f:
            r = self._fault_reply("openai", f)
            if r:
                return self._reply(r)
        if (body or {}).get("endpoint") != "/v1/responses" or (body or {}).get("completion_window") != "24h":
            return self._send(400, {"error": {"message": "endpoint must be /v1/responses and completion_window 24h (fake)"}})
        data = fake.files.get((body or {}).get("input_file_id") or "")
        if data is None:
            return self._send(400, {"error": {"message": "input_file_id not found"}})
        lines = [json.loads(x) for x in data.decode().splitlines() if x.strip()]
        ids = [ln.get("custom_id") for ln in lines]
        if len(set(ids)) != len(ids):
            return self._send(400, {"error": {"message": "custom_id must be unique"}})
        with fake.lock:
            bid = f"batch_fake{len(fake.batches) + 1:04d}"
        out_lines, err_lines = [], []
        for ln in lines:
            cid = ln["custom_id"]
            if ln.get("url") != "/v1/responses" or ln.get("method") != "POST":
                err_lines.append({"id": f"r_{cid}", "custom_id": cid, "response": None,
                                  "error": {"code": "invalid_url", "message": "url must be /v1/responses"}})
                continue
            if cid in (fake.scenario.get("batch_expire") or []):
                err_lines.append({"id": f"r_{cid}", "custom_id": cid, "response": None,
                                  "error": {"code": "batch_expired", "message": "This request could not be executed before the completion window expired."}})
                continue
            status, out, _h = self._openai(ln.get("body") or {}, batch=bid)
            rec = {"id": f"r_{cid}", "custom_id": cid, "response": {"status_code": status, "request_id": "req_fake", "body": out},
                   "error": None}
            (out_lines if status == 200 else err_lines).append(rec)
        with fake.lock:
            fake.batches[bid] = {"provider": "openai", "polls": 0, "status": "in_progress", "n": len(lines),
                                 "out": out_lines, "err": err_lines, "canceled": False, "results": out_lines + err_lines}
        self._send(200, self._openai_batch_obj(bid))

    def _openai_batch_obj(self, bid: str) -> dict:
        fake = self.server_fake
        b = fake.batches[bid]
        st = b["status"]
        done = st in ("completed", "expired", "cancelled", "failed")
        obj = {"id": bid, "object": "batch", "endpoint": "/v1/responses", "status": st, "completion_window": "24h",
               "request_counts": {"total": b["n"], "completed": len(b["out"]) if done else 0,
                                  "failed": len(b["err"]) if done else 0},
               "output_file_id": None, "error_file_id": None}
        if done and b["out"]:
            fid = f"file-out-{bid}"
            fake.files[fid] = "".join(json.dumps(x) + "\n" for x in reversed(b["out"])).encode()
            obj["output_file_id"] = fid
        if done and b["err"]:
            fid = f"file-err-{bid}"
            fake.files[fid] = "".join(json.dumps(x) + "\n" for x in b["err"]).encode()
            obj["error_file_id"] = fid
        if st == "failed":
            obj["errors"] = {"object": "list", "data": [{"code": "invalid_json_line", "message": "bad input file"}]}
        return obj

    def _openai_batch(self, bid: str, action: str, method: str) -> None:
        fake = self.server_fake
        bad = self._openai_auth()
        if bad:
            return self._reply(bad)
        self._record("openai", None, [], batch=f"{action or 'get'}:{bid}")
        with fake.lock:
            b = fake.batches.get(bid)
        if b is None:
            return self._send(404, {"error": {"message": f"No such batch: {bid}"}})
        if action == "/cancel" and method == "POST":
            with fake.lock:
                if b["status"] in ("validating", "in_progress", "finalizing"):
                    keep = int(fake.scenario.get("batch_cancel_keep") or 0)
                    b["out"], b["err"] = b["out"][:keep], []
                    b["status"] = "cancelling"
            return self._send(200, self._openai_batch_obj(bid))
        with fake.lock:
            b["polls"] += 1
            if b["status"] == "cancelling":
                b["status"] = "cancelled"
            elif b["status"] in ("validating", "in_progress") and b["polls"] >= int(fake.scenario.get("batch_polls") or 1):
                b["status"] = (fake.scenario.get("batch_fail") or "completed")
        self._send(200, self._openai_batch_obj(bid))

    # ---- OpenAI-compatible / Ollama
    def _ollama_tags(self):
        self._record("compat", None, [])
        if "tags" in (self.server_fake.scenario.get("reject_params") or []):
            return self._send(404, {"error": "not found"})
        ms = self.server_fake.scenario.get("models") or ["qwen3.8:27b", "qwen2.5vl:7b", "llama3.2:3b"]
        self._send(200, {"models": [{"name": m, "model": m, "size": 1} for m in ms]})

    def _ollama_show(self, body):
        self._record("compat", body, [])
        name = (body or {}).get("model", "")
        caps = ["completion"] + (["vision"] if re.search(r"qwen|vl|gemma|llava", name) and "llama3.2" not in name else [])
        self._send(200, {"capabilities": caps, "details": {"family": name.split(":")[0]}})

    def _compat(self, body: dict):
        fake = self.server_fake
        sc = fake.scenario
        msgs = body.get("messages") or []
        user = next((m for m in msgs if m.get("role") == "user"), {})
        crops, images = [], []
        text = " ".join(str(c.get("text", "")) for c in (user.get("content") or []) if isinstance(c, dict))
        for c in user.get("content") or []:
            if isinstance(c, dict) and c.get("type") == "image_url":
                im, size = self._see(c["image_url"]["url"].split(",", 1)[1])
                crops.append(im)
                images.append(size)
        self._record("compat", body, images, text=text[:4000])
        f = fake.take_fault("compat")
        if f:
            r = self._fault_reply("compat", f)
            if r:
                return self._reply(r)
        for p in sc.get("reject_params") or []:
            if p in body:
                return self._send(400, {"error": {"message": f"Unrecognized request argument supplied: {p} is not supported by this server"}})
        model = body.get("model", "")
        family = sc.get("family") or ("qwen25" if re.search(r"qwen2\.?5", model) else "qwen3")
        rf = body.get("response_format") or {}
        schema = ((rf.get("json_schema") or {}).get("schema")) or {}
        if not schema:
            m = re.search(r"matching this JSON schema: (\{.*\})\s*$", text, re.S)
            if m:
                try:
                    schema = json.loads(m.group(1))
                except ValueError:
                    schema = {}
        kind = schema_kind(schema) if schema else ("classify" if "Crop 1:" in text else "detect")
        if kind == "classify":
            out_text = json.dumps({"items": [{"index": i + 1, "label": self._classify_crop(im)[0], "confidence": "high", "reason": "colour"}
                                             for i, im in enumerate(crops)]})
        elif kind == "generate":
            out_text = self._generate(schema, crops, text)
        else:
            out_text = json.dumps({"objects": self._render(self._objects(crops[0], text), family, *images[0])})
        if sc.get("prose") or not rf:
            out_text = "Sure! Here is the answer:\n```json\n" + out_text + "\n```\nLet me know if you need anything else."
        if f and f.get("kind") == "text":
            out_text = f["text"]
        finish = "stop"
        if f and f.get("kind") == "truncate" and body.get("max_tokens", 0) <= int(f.get("below", 10 ** 9)):
            out_text, finish = self._truncate(out_text, int(f.get("keep", 1))), "length"
        self._send(200, {"id": "chatcmpl-fake", "object": "chat.completion", "model": model,
                         "choices": [{"index": 0, "message": {"role": "assistant", "content": out_text}, "finish_reason": finish}],
                         "usage": {"prompt_tokens": 600, "completion_tokens": len(out_text) // 4}})


def _block_text(v: Any) -> str:
    """The text of a Messages API ``system`` / ``content`` value: a plain string or a list of blocks."""
    if isinstance(v, str):
        return v
    if isinstance(v, list):
        return "".join(str(b.get("text", "")) for b in v if isinstance(b, dict) and b.get("type") == "text")
    return ""


def anthropic_text_tokens(body: dict) -> int:
    """Text input tokens the Messages API bills on top of the images: the system prompt AND every text block of
    every message (~4 characters a token), plus a few tokens of turn framing per message (so even a one-word request
    bills something). Only the text is counted, not the JSON wrapping around it."""
    msgs = [m for m in (body.get("messages") or []) if isinstance(m, dict)]
    return len(_block_text(body.get("system") or "")) // 4 + sum(
        len(_block_text(m.get("content"))) // 4 + TURN_TOKENS for m in msgs)


def _anthropic_msg(model, content, stop, tin, tout, stop_details=None):
    return {"id": "msg_fake", "type": "message", "role": "assistant", "model": model, "content": content,
            "stop_reason": stop, "stop_sequence": None, "stop_details": stop_details,
            "usage": {"input_tokens": tin, "output_tokens": tout, "cache_read_input_tokens": 0,
                      "cache_creation_input_tokens": 0}}


def _redact(v: Optional[str]) -> Optional[str]:
    if v is None:
        return None
    return v[:10] + "…" + v[-4:] if len(v) > 16 else "…"


def _strip_images(body: Any) -> Any:
    """Request body without base64 payloads (replaced by their length)."""
    if isinstance(body, dict):
        out = {}
        for k, v in body.items():
            if k == "data" and isinstance(v, str) and len(v) > 200:
                out[k] = f"<{len(v)} b64 chars>"
            elif k in ("url", "image_url") and isinstance(v, str) and v.startswith("data:"):
                out[k] = f"<data url {len(v)} chars>"
            else:
                out[k] = _strip_images(v)
        return out
    if isinstance(body, list):
        return [_strip_images(x) for x in body]
    return body


def main() -> None:
    ap = argparse.ArgumentParser(description="Fake cloud VLM APIs for Fovea live checks")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8955)
    a = ap.parse_args()
    fake = FakeVLM(a.host, a.port)
    print(f"fake VLM listening on {fake.url}", flush=True)
    print("keys:", json.dumps({k: v for k, v in KEYS.items() if v}), flush=True)
    try:
        fake.httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
