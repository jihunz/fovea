"""Shared machinery for cloud / server vision-language-model providers (Claude, Gemini, OpenAI, OpenAI-compatible).

Each provider module only knows its wire format. Everything that must be identical across providers lives here:

* image preparation — EXIF transpose, client-side pre-resize to the provider's native size (so coordinates map 1:1),
  SAHI-style tiles;
* the model's coordinate frame → normalised xyxy on the *sent* image → full-image YOLO geometry, with a guard that
  notices when a model answers on the wrong scale (0-1000 instead of pixels, 0-1 floats …);
* prompt + JSON-schema builders with the dataset classes as a closed ``enum``;
* robust JSON extraction (code fences, prose, truncated output);
* label → dataset class resolution, clamping, dedupe / NMS, self-consistency voting;
* retries (``retry-after`` / exponential backoff + jitter, abort codes), adaptive concurrency, key scrubbing;
* the price table (``prices.json``) and cost / estimate maths.

``VLMProvider`` implements ``detect`` / ``classify`` / ``estimate`` once on top of two small per-provider hooks
(``call_detect`` / ``call_classify``), so the coordinate maths is shared and tested in one place. On top of that:

* ``generate_json`` — one structured-output request (text + optional images) for naming classes / natural-language
  search; schemas are adapted per API dialect (``schema_for``), free-form maps travel as key/value pairs;
* provider batch APIs (``batch_submit`` / ``batch_status`` / ``batch_results`` / ``batch_cancel``, ~50 % off): the
  requests are built exactly like the synchronous ones and the decode context is saved in ``DATA_DIR/ai_batches``;
* ``options.refine`` — a crop-and-refine second pass for small / low-agreement boxes with geometric acceptance guards;
* ``options.ruler`` — an experimental tick ruler in a margin for 0-1000 frames (the margin is removed when decoding).
"""
from __future__ import annotations

import base64
import contextlib
import contextvars
import functools
import hashlib
import importlib.util
import io
import ipaddress
import json
import logging
import math
import os
import random
import re
import threading
import time
from dataclasses import dataclass, field
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Optional, Sequence, Tuple
from urllib.parse import urlparse

from .base import (Classification, ClassifyRequest, ClassifyResult, ClassSpec, DetectRequest, DetectResult,
                   Detection, Provider, ProviderError, Usage, billed_usage, carry_usage)

log = logging.getLogger("fovea.ai")

PRICES_PATH = Path(__file__).with_name("prices.json")

# ----------------------------------------------------------------------------------------------------------- prices


@functools.lru_cache(maxsize=1)
def load_prices() -> dict:
    with open(PRICES_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def prices_as_of() -> str:
    return load_prices().get("as_of", "")


def provider_table(provider: str) -> dict:
    return load_prices().get("providers", {}).get(provider, {})


def static_models(provider: str) -> List[dict]:
    return [dict(m) for m in provider_table(provider).get("models", [])]


def _norm_model_id(model: str) -> str:
    m = (model or "").strip()
    return m[len("models/"):] if m.startswith("models/") else m


_VARIANT = re.compile(r"[-@:](\d{4}-\d{2}-\d{2}|\d{8}|latest|v\d+(?::\d+)?|preview(?:-\d{2}-\d{4})?|"
                      r"q\d\w*|fp16|bf16|instruct)$", re.I)


def is_variant(model: str, base: str) -> bool:
    """``model`` is ``base`` with a date / tag / quantisation suffix (``gpt-6.1-sol-2026-09-01``, ``qwen3.8:27b-q4_K_M``)
    — but ``claude-opus-5-5`` is NOT a variant of ``claude-opus-5``."""
    return model == base or (model.startswith(base) and _VARIANT.fullmatch(model[len(base):]) is not None)


def find_model(provider: str, model: str) -> Optional[dict]:
    """Curated entry for a model id; dated / tagged variants of a curated id share its entry."""
    mid = _norm_model_id(model)
    models = provider_table(provider).get("models", [])
    for m in models:
        if m["id"] == mid:
            return m
    best = None
    for m in models:
        if is_variant(mid, m["id"]) and (best is None or len(m["id"]) > len(best["id"])):
            best = m
    return best


def default_model(provider: str) -> str:
    models = provider_table(provider).get("models", [])
    for m in models:
        if m.get("default"):
            return m["id"]
    return models[0]["id"] if models else ""


def model_price(provider: str, model: str) -> Optional[dict]:
    m = find_model(provider, model)
    return dict(m["price"]) if m and m.get("price") else None


def cost_usd(provider: str, model: str, input_tokens: int, output_tokens: int, *, cached_tokens: int = 0,
             cache_write_tokens: int = 0, batch: bool = False) -> float:
    """Cost of one call. ``input_tokens`` is the TOTAL prompt size including cached and cache-written tokens.
    Unknown models cost 0 (reported as "cost not tracked" by the caller)."""
    p = model_price(provider, model)
    if not p:
        return 0.0
    pin = p.get("batch_in", p["in"]) if batch else p["in"]
    pout = p.get("batch_out", p["out"]) if batch else p["out"]
    cached = max(0, min(int(cached_tokens or 0), int(input_tokens or 0)))
    write = max(0, min(int(cache_write_tokens or 0), int(input_tokens or 0) - cached))
    plain = max(0, int(input_tokens or 0) - cached - write)
    p_cache = p.get("cache_read", pin)
    p_write = pin * float(provider_table(provider).get("cache_write_multiplier", 1.0))
    usd = (plain * pin + cached * p_cache + write * p_write + int(output_tokens or 0) * pout) / 1e6
    return round(usd, 8)


# ------------------------------------------------------------------------------------------------ secrets / config

_KEY_PATTERNS = [
    (re.compile(r"sk-ant-[A-Za-z0-9_\-]{6,}"), "sk-ant-…"),
    (re.compile(r"\bsk-(?:proj-|svcacct-|admin-)?[A-Za-z0-9_\-]{12,}"), "sk-…"),
    (re.compile(r"\bAIza[0-9A-Za-z_\-]{16,}"), "AIza…"),
    (re.compile(r"(?i)\b(bearer\s+)[A-Za-z0-9._\-~+/]{6,}=*"), r"\1…"),
    (re.compile(r"(?i)\b(x-api-key|x-goog-api-key|api[_-]?key|authorization|access[_-]?token)(\"?'?\s*[:=]\s*\"?'?)"
                r"(?!…)[^\s'\",&}]+"), r"\1\2…"),
    (re.compile(r"(?i)([?&](?:key|api_key|apikey|token|access_token)=)[^&\s#]+"), r"\1…"),
    # credentials inside a URL (a gateway set as http://user:password@host/v1)
    (re.compile(r"(?i)\b([a-z][a-z0-9+.\-]*://)[^\s/?#@'\"<>]+@"), r"\1…@"),
]


def scrub(text: Any, *secrets: Optional[str]) -> str:
    """Remove anything that looks like an API key / bearer token from text that may reach the client or a log."""
    s = "" if text is None else str(text)
    for sec in secrets:
        if sec and len(sec) >= 6:
            s = s.replace(sec, "…")
    for rx, rep in _KEY_PATTERNS:
        s = rx.sub(rep, s)
    try:
        from . import secrets as _secrets   # also redacts every stored / env key verbatim
        s = _secrets.scrub(s)
    except Exception:  # noqa: BLE001 — scrubbing must never raise
        pass
    return s


def provider_key(provider: str, env_keys: Sequence[str] = ()) -> Optional[str]:
    """The API key for a provider, read at call time: ``fovea.ai.secrets`` (env var > secrets file)."""
    try:
        from .secrets import get_key
    except ImportError:
        get_key = None
    if get_key is not None:
        try:
            return get_key(provider) or None
        except Exception:  # noqa: BLE001 — a broken secrets file must not crash the provider list
            log.warning("secrets lookup failed for %s", provider)
    for name in env_keys:
        v = os.environ.get(name, "").strip()
        if v:
            return v
    return None


def provider_config(provider: str) -> dict:
    """Non-secret per-provider settings (base_url, default_model, effort, concurrency, batch_mode, coord, …)."""
    try:
        from .config import provider_config as _pc
        return dict(_pc(provider) or {})
    except Exception:  # noqa: BLE001 — no DB (unit tests without one) → defaults
        return {}


def is_loopback_host(host: str) -> bool:
    """True only for names that cannot leave this machine: ``localhost`` / ``*.localhost`` (RFC 6761) and IP literals
    that are loopback (127.0.0.0/8, ::1, ::ffff:127.x) or unspecified (0.0.0.0 / :: connect to this host). Decided by
    parsing, never by prefix: ``127.evil.example`` or ``127.0.0.1.nip.io`` are DNS names anyone can point anywhere."""
    host = (host or "").strip().lower()
    if host.startswith("[") and host.endswith("]"):
        host = host[1:-1]
    if not host:
        return False
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        ip = ipaddress.ip_address(host.split("%", 1)[0])       # drop an IPv6 zone id (fe80::1%en0)
    except ValueError:
        return False                                            # a DNS name: it resolves wherever its owner says
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    return ip.is_loopback or ip.is_unspecified


def is_loopback_url(url: str) -> bool:
    """Whether requests to ``url`` stay on this machine (decides openai_compat's local/cloud kind and privacy line)."""
    try:
        host = urlparse(str(url or "").strip()).hostname or ""
    except ValueError:
        return False
    return is_loopback_host(host)


def url_host(url: str) -> str:
    """``host[:port]`` of a URL for the UI and messages — never its user name or password (a gateway URL may carry
    ``user:password@``; httpx sends that as basic auth, but it must not reach a client or a log)."""
    try:
        p = urlparse(str(url or "").strip())
        host = p.hostname or ""
        port = p.port
    except ValueError:
        return scrub(url)
    if not host:
        return scrub(url)
    if ":" in host:                                 # IPv6 literal
        host = f"[{host}]"
    return f"{host}:{port}" if port else host


def redact_url(url: str) -> str:
    """``url`` without the ``user:password@`` part (for anything stored or shown)."""
    try:
        p = urlparse(str(url or "").strip())
        if p.username is None and p.password is None:
            return str(url or "")
        return p._replace(netloc=url_host(url)).geturl()
    except ValueError:
        return scrub(url)


# ---------------------------------------------------------------------------------------------------- images

@dataclass
class Prepared:
    """One image as sent to the model, plus where it came from in the original."""
    data: bytes
    mime: str
    width: int                   # sent size — the frame the model's pixel coordinates refer to
    height: int
    region: Tuple[float, float, float, float]   # (ox, oy, tw, th) in pixels of the decoded original
    full: Tuple[int, int]        # (W, H) of the decoded original
    pad: Tuple[int, int, int, int] = (0, 0, 0, 0)   # ruler margin (left, top, right, bottom) in sent pixels

    @property
    def b64(self) -> str:
        return base64.b64encode(self.data).decode("ascii")

    def data_url(self) -> str:
        return f"data:{self.mime};base64,{self.b64}"

    @property
    def photo_size(self) -> Tuple[int, int]:
        """Size of the photo inside the sent image (the sent size minus any ruler margin)."""
        l, t, r, b = self.pad
        return max(1, self.width - l - r), max(1, self.height - t - b)

    def geometry(self) -> list:
        """JSON-able decode context (everything but the pixels) — kept for provider batches."""
        return [self.width, self.height, list(self.region), list(self.full), list(self.pad)]

    @classmethod
    def from_geometry(cls, g: Sequence) -> "Prepared":
        w, h, region, full, pad = g
        return cls(b"", "image/jpeg", int(w), int(h), tuple(float(v) for v in region),  # type: ignore[arg-type]
                   (int(full[0]), int(full[1])), tuple(int(v) for v in pad))  # type: ignore[arg-type]


def open_image(path: str, need_edge: Optional[int] = None):
    """Decode an image the way Fovea displays it: EXIF orientation applied, 8-bit RGB.
    ``need_edge`` lets JPEG decoding use DCT scaling when we will downscale anyway (aspect is preserved)."""
    from PIL import Image, ImageOps
    try:
        im = Image.open(path)
    except FileNotFoundError:
        raise ProviderError(f"Image file not found: {Path(path).name}", "invalid")
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"Cannot read image {Path(path).name}: {e}", "invalid")
    try:
        if need_edge and im.format == "JPEG":
            w, h = im.size
            s = max(w, h)
            if s > need_edge * 2:
                scale = need_edge / s
                im.draft("RGB", (max(1, int(w * scale)), max(1, int(h * scale))))
        from ..config import MAX_IMAGE_PIXELS   # (qa-engine hook) never decode a raster beyond the cap
        if im.width * im.height > MAX_IMAGE_PIXELS:
            raise ProviderError(f"Image {Path(path).name} is too large to decode ({im.width}x{im.height} pixels; "
                                f"limit {MAX_IMAGE_PIXELS / 1e6:.0f} MP, FOVEA_MAX_IMAGE_PIXELS)", "invalid")
        im = ImageOps.exif_transpose(im)
        try:
            from ..core.thumbs import to_display_rgb
            im = to_display_rgb(im)
        except ImportError:
            im = im.convert("RGB")
        if im.mode != "RGB":
            im = im.convert("RGB")
        im.load()
    except ProviderError:
        raise
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"Cannot decode image {Path(path).name}: {e}", "invalid")
    return im


def fit_long_edge(w: int, h: int, max_edge: int) -> Tuple[int, int]:
    """Largest aspect-preserving size with long edge <= max_edge (never upscales)."""
    if max_edge <= 0 or max(w, h) <= max_edge:
        return w, h
    s = max_edge / max(w, h)
    return max(1, round(w * s)), max(1, round(h * s))


def anthropic_resized_size(w: int, h: int, max_edge: int = 2576, max_tokens: int = 4784) -> Tuple[int, int]:
    """Port of Anthropic's reference ``resized_size()``: the largest aspect-preserving size such that both edges
    (rounded up to 28 px) fit ``max_edge`` and ``ceil(w/28)*ceil(h/28) <= max_tokens``. Claude resizes to exactly
    this, so pre-resizing to it means Claude's pixel coordinates map 1:1 onto what we sent."""
    def fits(a: int, b: int) -> bool:
        return (math.ceil(a / 28) * 28 <= max_edge and math.ceil(b / 28) * 28 <= max_edge
                and math.ceil(a / 28) * math.ceil(b / 28) <= max_tokens)

    if fits(w, h):
        return w, h
    if h > w:
        rh, rw = anthropic_resized_size(h, w, max_edge, max_tokens)
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


def anthropic_image_tokens(w: int, h: int) -> int:
    return math.ceil(w / 28) * math.ceil(h / 28)


def patch_fit_size(w: int, h: int, max_edge: int = 2048, patch: int = 32, max_patches: int = 2500) -> Tuple[int, int]:
    """OpenAI ``detail:"high"`` budget: long edge <= max_edge and ceil(w/p)*ceil(h/p) <= max_patches."""
    w2, h2 = fit_long_edge(w, h, max_edge)
    if math.ceil(w2 / patch) * math.ceil(h2 / patch) <= max_patches:
        return w2, h2
    ar = w2 / h2
    lo, hi = 1, max(w2, h2)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        a, b = (mid, max(1, round(mid / ar))) if w2 >= h2 else (max(1, round(mid * ar)), mid)
        if math.ceil(a / patch) * math.ceil(b / patch) <= max_patches:
            lo = mid
        else:
            hi = mid
    return (lo, max(1, round(lo / ar))) if w2 >= h2 else (max(1, round(lo * ar)), lo)


def openai_image_tokens(w: int, h: int) -> int:
    return math.ceil(math.ceil(w / 32) * math.ceil(h / 32) * 1.2)


def smart_resize(w: int, h: int, factor: int = 28, min_pixels: int = 56 * 56, max_pixels: int = 1280 * 28 * 28
                 ) -> Tuple[int, int]:
    """Qwen2.5-VL processor resize (sides multiples of ``factor``, area within bounds). Pre-resizing to this makes the
    server's own smart_resize an identity, so Qwen2.5's absolute-pixel answers refer to exactly what we sent."""
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


def encode_image(im, size: Tuple[int, int], quality: int = 90) -> bytes:
    from PIL import Image
    if im.size != tuple(size):
        im = im.resize(size, Image.LANCZOS)
    buf = io.BytesIO()
    im.save(buf, "JPEG", quality=quality, optimize=True)
    return buf.getvalue()


def prepare(im, region: Optional[Tuple[float, float, float, float]], size_fn: Callable[[int, int], Tuple[int, int]],
            quality: int = 90, ruler: Optional[str] = None, encode: bool = True) -> Prepared:
    """Crop ``region`` (pixels of ``im``) and resize it with ``size_fn(w, h) -> (sw, sh)``. With ``ruler`` (a 0-1000
    frame) a margin with a tick ruler in that frame's units is added on the top and left (see ``add_ruler``).
    ``encode=False`` computes only the geometry (``data`` stays empty) — to decode batch results again."""
    from PIL import Image
    W, H = im.size
    if region is None:
        region = (0.0, 0.0, float(W), float(H))
    ox, oy, tw, th = region
    box = (int(round(ox)), int(round(oy)), int(round(ox + tw)), int(round(oy + th)))
    # use the integer crop actually taken so the offset maths matches the pixels sent
    region = (float(box[0]), float(box[1]), float(box[2] - box[0]), float(box[3] - box[1]))
    sw, sh = size_fn(box[2] - box[0], box[3] - box[1])
    if not encode:
        if ruler:
            left, top = ruler_margins(sw, sh)
            return Prepared(b"", "image/jpeg", sw + left, sh + top, region, (W, H), (left, top, 0, 0))
        return Prepared(b"", "image/jpeg", sw, sh, region, (W, H))
    crop = im if box == (0, 0, W, H) else im.crop(box)
    if ruler:
        photo = crop if crop.size == (sw, sh) else crop.resize((sw, sh), Image.LANCZOS)
        canvas, pad = add_ruler(photo, ruler)
        return Prepared(encode_image(canvas, canvas.size, quality), "image/jpeg", canvas.size[0], canvas.size[1],
                        region, (W, H), pad)
    return Prepared(encode_image(crop, (sw, sh), quality), "image/jpeg", sw, sh, region, (W, H))


def _ruler_font(size: int):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)
    except (TypeError, OSError, ValueError):     # Pillow < 10.1 or no FreeType: the fixed bitmap font
        return ImageFont.load_default()


def _ruler_metrics(pw: int, ph: int):
    m = max(24, int(round(0.05 * max(pw, ph))))
    return m, _ruler_font(max(10, int(round(m * 0.42)))), max(4, m // 4)


@functools.lru_cache(maxsize=256)
def ruler_margins(pw: int, ph: int) -> Tuple[int, int]:
    """(left, top) margin of the ruler canvas for a photo of pw x ph sent pixels."""
    from PIL import Image, ImageDraw
    m, font, tick = _ruler_metrics(pw, ph)
    x0, _y0, x1, _y1 = ImageDraw.Draw(Image.new("RGB", (8, 8))).textbbox((0, 0), "1000", font=font)
    return max(m, (x1 - x0) + tick + 6), m


def add_ruler(photo, frame: str):
    """Experimental ruler overlay for 0-1000 frames (SOTIF study: Gemini 3 mAP50 0.525 → 0.695 with a ruler).

    A white margin goes on the top and left of the photo with ticks every 50 units (labels every 100) of the frame the
    model answers in — measured over the WHOLE sent image, margin included, because that is the frame the models are
    trained to emit. The decoder removes the margin offset afterwards (``Prepared.pad``). Black on white only: the
    ruler never adds colours that could be mistaken for objects. Returns (canvas, (left, top, 0, 0))."""
    from PIL import Image, ImageDraw
    pw, ph = photo.size
    m, font, tick = _ruler_metrics(pw, ph)
    probe = ImageDraw.Draw(Image.new("RGB", (8, 8)))

    def text_size(s: str) -> Tuple[int, int]:
        x0, y0, x1, y1 = probe.textbbox((0, 0), s, font=font)
        return x1 - x0, y1 - y0

    left, top = ruler_margins(pw, ph)
    cw, ch = pw + left, ph + top
    canvas = Image.new("RGB", (cw, ch), (255, 255, 255))
    canvas.paste(photo, (left, top))
    d = ImageDraw.Draw(canvas)
    ink = (0, 0, 0)
    d.rectangle([left - 1, top - 1, cw - 1, ch - 1], outline=ink, width=1)
    for v in range(0, 1001, 50):
        major = v % 100 == 0
        x = int(round(v / 1000 * cw))
        if x >= left and x <= cw:
            x = min(x, cw - 1)
            d.line([(x, top - (tick if major else tick // 2) - 1), (x, top - 1)], fill=ink, width=1)
            if major:
                tw_, th_ = text_size(str(v))
                tx = min(max(x - tw_ // 2, left), cw - tw_ - 1)
                d.text((tx, max(0, top - tick - th_ - 4)), str(v), fill=ink, font=font)
        y = int(round(v / 1000 * ch))
        if y >= top and y <= ch:
            y = min(y, ch - 1)
            d.line([(left - (tick if major else tick // 2) - 1, y), (left - 1, y)], fill=ink, width=1)
            if major:
                tw_, th_ = text_size(str(v))
                ty = min(max(y - th_ // 2 - 2, top), ch - th_ - 3)
                d.text((max(1, left - tick - tw_ - 4), ty), str(v), fill=ink, font=font)
    return canvas, (left, top, 0, 0)


def ruler_sentence(prep: Prepared) -> str:
    """What the model is told about the ruler margin (and where the photo is, in the frame's units)."""
    l, t, _r, _b = prep.pad
    x0 = int(round(l / prep.width * 1000))
    y0 = int(round(t / prep.height * 1000))
    return (f" A ruler is drawn in the white margin above and to the left of the photo: its tick labels are positions "
            f"in the same 0-1000 coordinates you answer in, measured over the whole image including the margin. The "
            f"photo occupies x {x0}-1000 and y {y0}-1000; every object is inside it.")


def tile_regions(W: int, H: int, cols: int, rows: int, overlap: float = 0.2) -> List[Tuple[float, float, float, float]]:
    """SAHI-style grid: ``cols`` x ``rows`` tiles with ``overlap`` (fraction of a tile) between neighbours."""
    cols, rows = max(1, int(cols)), max(1, int(rows))
    overlap = min(max(float(overlap), 0.0), 0.5)
    tw = W / (cols - (cols - 1) * overlap)
    th = H / (rows - (rows - 1) * overlap)
    out = []
    for r in range(rows):
        for c in range(cols):
            ox = 0.0 if cols == 1 else c * (W - tw) / (cols - 1)
            oy = 0.0 if rows == 1 else r * (H - th) / (rows - 1)
            out.append((ox, oy, tw, th))
    return out


def tile_plan(W: int, H: int, tiles: Any, native_edge: int) -> List[Tuple[float, float, float, float]]:
    """Tiles to run in ADDITION to the full-image pass. ``tiles``: False/0 none · True/'auto' (grid only when the image
    is much larger than what the model sees) · int n (n x n) · [cols, rows]."""
    if not tiles:
        return []
    if tiles is True or tiles == "auto":
        ratio = max(W, H) / max(1, native_edge)
        if ratio < 1.5:
            return []
        n = 2 if ratio < 3 else 3
        cols = n if W >= H * 0.6 else max(1, n - 1)
        rows = n if H >= W * 0.6 else max(1, n - 1)
        return tile_regions(W, H, cols, rows, 0.2)
    if isinstance(tiles, (list, tuple)) and len(tiles) == 2:
        cols, rows = int(tiles[0]), int(tiles[1])
    else:
        try:
            cols = rows = int(tiles)
        except (TypeError, ValueError):
            return []
    cols, rows = min(max(cols, 1), 4), min(max(rows, 1), 4)
    if cols * rows <= 1:
        return []
    return tile_regions(W, H, cols, rows, 0.2)


def crop_for_classification(im, xc: float, yc: float, w: float, h: float, pad: float = 0.25, edge: int = 448,
                            mark: bool = True, min_edge: int = 336):
    """Crop a box with ``pad`` context on each side, long edge between ``min_edge`` and ``edge``. Small crops are
    upscaled: a 70 px person is only a handful of 28 px vision patches, too coarse for the model to judge a pose.
    With ``mark`` the box itself is outlined so the model knows which object is meant when the context has others."""
    from PIL import ImageDraw
    W, H = im.size
    bw, bh = max(w * W, 2.0), max(h * H, 2.0)
    cx, cy = xc * W, yc * H
    x1, y1 = max(0, int(math.floor(cx - bw / 2 - pad * bw))), max(0, int(math.floor(cy - bh / 2 - pad * bh)))
    x2, y2 = min(W, int(math.ceil(cx + bw / 2 + pad * bw))), min(H, int(math.ceil(cy + bh / 2 + pad * bh)))
    if x2 - x1 < 2 or y2 - y1 < 2:
        x1, y1, x2, y2 = max(0, int(cx) - 1), max(0, int(cy) - 1), min(W, int(cx) + 2), min(H, int(cy) + 2)
    crop = im.crop((x1, y1, x2, y2))
    sw, sh = fit_long_edge(crop.size[0], crop.size[1], edge)
    lo = min(int(min_edge or 0), edge)
    if max(sw, sh) < lo:
        s_up = lo / max(sw, sh)
        sw, sh = max(1, round(sw * s_up)), max(1, round(sh * s_up))
    if (sw, sh) != crop.size:
        from PIL import Image
        crop = crop.resize((sw, sh), Image.LANCZOS if (sw, sh) < crop.size else Image.BICUBIC)
    if mark:
        crop = crop.copy()
        sx, sy = sw / max(1, x2 - x1), sh / max(1, y2 - y1)
        bx1, by1 = (cx - bw / 2 - x1) * sx, (cy - bh / 2 - y1) * sy
        bx2, by2 = (cx + bw / 2 - x1) * sx, (cy + bh / 2 - y1) * sy
        lw = max(2, round(min(sw, sh) / 150))
        d = ImageDraw.Draw(crop)
        d.rectangle([bx1, by1, bx2 - 1, by2 - 1], outline=(255, 0, 64), width=lw)
    buf = io.BytesIO()
    crop.save(buf, "JPEG", quality=90)
    return buf.getvalue(), crop.size


# ------------------------------------------------------------------------------------------- coordinate frames

PIXEL_XYXY = "pixel_xyxy"    # Claude, Qwen2.5-VL: absolute pixels on the sent image
YXYX_1000 = "yxyx_1000"      # Gemini, GPT (box_2d): [ymin, xmin, ymax, xmax] in 0..1000
XYXY_1000 = "xyxy_1000"      # Qwen3+ (bbox_2d): [x1, y1, x2, y2] in 0..1000
NORM_XYXY = "norm_xyxy"      # [x1, y1, x2, y2] in 0..1
FRAMES = (PIXEL_XYXY, YXYX_1000, XYXY_1000, NORM_XYXY)

FRAME_SENTENCE = {
    PIXEL_XYXY: "integer pixel coordinates of THIS {w}x{h} image: (x1, y1) is the top-left corner and (x2, y2) the "
                "bottom-right corner; origin at the top-left, x to the right, y down",
    YXYX_1000: "box_2d = [ymin, xmin, ymax, xmax] as integers normalised to 0-1000 against the image height and "
               "width (0 = top/left edge, 1000 = bottom/right edge) — y comes first",
    XYXY_1000: "bbox_2d = [x1, y1, x2, y2] as integers in relative coordinates from 0 to 1000 (0 = left/top edge, "
               "1000 = right/bottom edge)",
    NORM_XYXY: "bbox = [x1, y1, x2, y2] as decimal fractions of the image width and height between 0 and 1",
}


def _to_xyxy_order(vals: Sequence[float], frame: str) -> Tuple[float, float, float, float]:
    a, b, c, d = (float(v) for v in vals[:4])
    if frame == YXYX_1000:
        return b, a, d, c
    return a, b, c, d


def detect_scale(all_values: Sequence[float], frame: str, sw: int, sh: int) -> Tuple[str, Optional[str]]:
    """Decide the scale a whole response is actually on. Returns (scale, anomaly) where scale is one of
    ``pixel`` / ``k1000`` / ``unit`` and ``anomaly`` a short note when the model ignored the requested frame."""
    vals = [abs(float(v)) for v in all_values if v is not None]
    expected = {PIXEL_XYXY: "pixel", YXYX_1000: "k1000", XYXY_1000: "k1000", NORM_XYXY: "unit"}[frame]
    if not vals:
        return expected, None
    vmax = max(vals)
    if expected == "pixel":
        if vmax <= 1.0 and max(sw, sh) > 4:
            return "unit", "0-1 fractions instead of pixels"
        if vmax > max(sw, sh) * 1.02 and vmax <= 1050:
            return "k1000", "a 0-1000 scale instead of pixels"
        return "pixel", None
    if expected == "k1000":
        if vmax > 1050:                     # a few units over 1000 is sloppiness, not pixels
            return "pixel", "pixels instead of the 0-1000 scale"
        if vmax <= 1.0:
            return "unit", "0-1 fractions instead of the 0-1000 scale"
        return "k1000", None
    # unit
    if vmax > 1.05:
        if vmax <= 1050:
            return "k1000", "a 0-1000 scale instead of 0-1 fractions"
        return "pixel", "pixels instead of 0-1 fractions"
    return "unit", None


def to_unit(xyxy: Tuple[float, float, float, float], scale: str, sw: int, sh: int) -> Tuple[float, float, float, float]:
    x1, y1, x2, y2 = xyxy
    if scale == "pixel":
        return x1 / sw, y1 / sh, x2 / sw, y2 / sh
    if scale == "k1000":
        return x1 / 1000.0, y1 / 1000.0, x2 / 1000.0, y2 / 1000.0
    return x1, y1, x2, y2


def region_to_full(xyxy_unit: Tuple[float, float, float, float], prep: Prepared) -> Tuple[float, float, float, float]:
    """Normalised xyxy on the sent (possibly tiled) image → normalised xyxy on the full original image."""
    ox, oy, tw, th = prep.region
    W, H = prep.full
    x1, y1, x2, y2 = xyxy_unit
    return ((ox + x1 * tw) / W, (oy + y1 * th) / H, (ox + x2 * tw) / W, (oy + y2 * th) / H)


# ------------------------------------------------------------------------------------------- prompts / schema

@dataclass
class Vocab:
    """The closed label vocabulary sent to the model and how each label maps back to a dataset class."""
    classes: List[ClassSpec]
    labels: List[str]                       # unique strings used in the prompt and the schema enum
    by_label: Dict[str, ClassSpec]

    @classmethod
    def build(cls, classes: Sequence[ClassSpec]) -> "Vocab":
        labels, by_label = [], {}
        for c in classes:
            q = (c.query or c.name or f"class_{c.id}").strip()
            if not q:
                q = f"class_{c.id}"
            base, k = q, 2
            while q in by_label:          # two classes with the same prompt: keep both distinguishable
                q = f"{base} ({c.name or c.id})" if k == 2 else f"{base} #{k}"
                k += 1
            labels.append(q)
            by_label[q] = c
        return cls(list(classes), labels, by_label)

    @staticmethod
    def _norm(s: str) -> str:
        s = (s or "").strip().strip("\"'`").casefold()
        s = re.sub(r"[\s_\-]+", " ", s)
        return re.sub(r"[^\w ]+", "", s).strip()

    @staticmethod
    def _singular(s: str) -> str:
        if s.endswith("ies") and len(s) > 4:
            return s[:-3] + "y"
        if s.endswith(("ches", "shes", "sses", "xes")):
            return s[:-2]
        if s.endswith("s") and not s.endswith("ss") and len(s) > 3:
            return s[:-1]
        return s

    @functools.cached_property
    def _index(self) -> Dict[str, ClassSpec]:
        idx: Dict[str, ClassSpec] = {}
        for lab, c in self.by_label.items():
            for key in (lab, c.name, c.prompt):
                if not key:
                    continue
                n = self._norm(key)
                idx.setdefault(n, c)
                idx.setdefault(self._singular(n), c)
        return idx

    def resolve(self, label: Any) -> Optional[ClassSpec]:
        if label is None:
            return None
        s = str(label)
        if s in self.by_label:
            return self.by_label[s]
        n = self._norm(s)
        c = self._index.get(n) or self._index.get(self._singular(n))
        if c is None and len(self.classes) == 1 and n in ("object", "objects", "item"):
            c = self.classes[0]
        return c


def detect_system_prompt(vocab: Vocab, frame: str, max_objects: int) -> str:
    """Stable across images (no per-image size in it) so prompt caching can reuse it for a whole batch."""
    lines = ["You are a meticulous object-detection annotator for a computer-vision training dataset.",
             "Find EVERY visible instance of the classes below and return one tight bounding box per instance.",
             "", "Classes (use the label text exactly as written):"]
    for lab in vocab.labels:
        c = vocab.by_label[lab]
        desc = (c.description or "").strip()
        lines.append(f"- {json.dumps(lab, ensure_ascii=False)}" + (f": {desc}" if desc else ""))
    frame_txt = (FRAME_SENTENCE[frame].replace("THIS {w}x{h} image", "the image exactly as you received it (its size "
                                                                     "is stated with the image)")
                 .format(w="W", h="H"))
    lines += ["", "Rules:",
              "- Use only these labels; ignore every other kind of object.",
              "- Boxes are tight around the visible extent of each object (do not extend them to hidden parts).",
              "- One box per instance; never one box around a group of objects.",
              "- If no listed object is visible, return an empty list.",
              f"- Return at most {max_objects} objects; if there are more, prefer the largest and clearest.",
              "", "Coordinate format: " + frame_txt + "."]
    return "\n".join(lines)


def detect_user_text(vocab: Vocab, frame: str, sw: int, sh: int, schema: Optional[dict] = None) -> str:
    names = ", ".join(json.dumps(l, ensure_ascii=False) for l in vocab.labels)
    txt = (f"Image size: {sw}x{sh} px. Detect every instance of: {names}. "
           f"Coordinates: {FRAME_SENTENCE[frame].format(w=sw, h=sh)}.")
    hint = shape_hint(schema) if schema else ""
    return txt + (f" Answer as JSON: {hint}" if hint else "")


def shape_hint(schema: dict) -> str:
    """One-line example of the expected JSON, derived from the schema (servers that ignore the schema still know the
    field names)."""
    try:
        props = schema["properties"]["objects"]["items"]["properties"]
    except (KeyError, TypeError):
        return ""
    if "x1" in props:
        item = '{"label": "<class>", "x1": 0, "y1": 0, "x2": 0, "y2": 0}'
    else:
        key = next((k for k in props if k != "label"), "box_2d")
        order = {"box_2d": "[ymin, xmin, ymax, xmax]"}.get(key, "[x1, y1, x2, y2]")
        item = '{"label": "<class>", "%s": %s}' % (key, order)
    return '{"objects": [' + item + ', ...]} (an empty list when nothing is visible).'


def box_fields(frame: str, key: Optional[str] = None) -> str:
    """The JSON field name used for a box in a given frame."""
    if key:
        return key
    return {PIXEL_XYXY: "xyxy", YXYX_1000: "box_2d", XYXY_1000: "bbox_2d", NORM_XYXY: "bbox"}[frame]


def detect_schema(labels: Sequence[str], style: str, *, bounds: bool, max_items: Optional[int] = None,
                  box_key: Optional[str] = None, number: bool = False, bound_max: Optional[int] = None) -> dict:
    """``{"objects": [...]}`` schema. ``style``: ``xyxy`` (named x1..y2 integers, for Claude whose schemas cannot bound
    array length) or ``array`` (``box_key: [4 numbers]``). ``bounds`` adds min/max/minItems/maxItems where supported."""
    label = {"type": "string", "enum": list(labels)}
    num_t = "number" if number else "integer"
    if style == "xyxy":
        props = {"label": label, "x1": {"type": num_t}, "y1": {"type": num_t}, "x2": {"type": num_t},
                 "y2": {"type": num_t}}
        req = ["label", "x1", "y1", "x2", "y2"]
    else:
        coord: Dict[str, Any] = {"type": num_t}
        if bounds and bound_max is not None:
            coord.update({"minimum": 0, "maximum": bound_max})
        arr: Dict[str, Any] = {"type": "array", "items": coord}
        if bounds:
            arr.update({"minItems": 4, "maxItems": 4})
        key = box_key or "box_2d"
        props = {"label": label, key: arr}
        req = ["label", key]
    objects: Dict[str, Any] = {"type": "array", "items": {"type": "object", "properties": props, "required": req,
                                                         "additionalProperties": False}}
    if bounds and max_items:
        objects["maxItems"] = int(max_items)
    return {"type": "object", "properties": {"objects": objects}, "required": ["objects"],
            "additionalProperties": False}


CONFIDENCE_WORDS = {"low": 0.35, "medium": 0.65, "high": 0.9}
NONE_LABEL, UNCERTAIN_LABEL = "none", "uncertain"


def classify_labels(vocab: Vocab) -> List[str]:
    extra = [x for x in (NONE_LABEL, UNCERTAIN_LABEL) if x not in vocab.by_label]
    return list(vocab.labels) + extra


def classify_schema(labels: Sequence[str], *, bounds: bool, n: Optional[int] = None) -> dict:
    item = {"type": "object",
            "properties": {"index": {"type": "integer"},
                           "label": {"type": "string", "enum": list(labels)},
                           "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
                           "reason": {"type": "string"}},
            "required": ["index", "label", "confidence", "reason"], "additionalProperties": False}
    arr: Dict[str, Any] = {"type": "array", "items": item}
    if bounds and n:
        arr["maxItems"] = int(n)
    return {"type": "object", "properties": {"items": arr}, "required": ["items"], "additionalProperties": False}


def classify_system_prompt(vocab: Vocab) -> str:
    lines = ["You are a careful image annotator verifying object classes for a computer-vision dataset.",
             "Each numbered image is a crop around ONE object, outlined with a thin red rectangle; the margin around "
             "it is context only.",
             "Decide which single class the outlined object belongs to.", "", "Classes:"]
    for lab in vocab.labels:
        c = vocab.by_label[lab]
        desc = (c.description or "").strip()
        lines.append(f"- {json.dumps(lab, ensure_ascii=False)}" + (f": {desc}" if desc else ""))
    lines += ["", f'Answer "{NONE_LABEL}" when the outlined object is none of these classes, and '
                  f'"{UNCERTAIN_LABEL}" when the crop does not show enough to decide.',
              "Give confidence low / medium / high and a reason of at most 15 words.",
              "Return exactly one item per crop, using the crop's number as index."]
    return "\n".join(lines)


REFINE_MARK = "nearest the centre of this zoomed crop"


def refine_system_prompt(frame: str) -> str:
    """Second pass on a zoomed context crop (research §7 item 7). Stable across crops so it can be cached."""
    frame_txt = (FRAME_SENTENCE[frame].replace("THIS {w}x{h} image", "the image exactly as you received it (its size "
                                                                     "is stated with the image)").format(w="W", h="H"))
    return "\n".join([
        "You are a meticulous object-detection annotator refining ONE bounding box.",
        "The image is a zoomed-in crop of a larger photo, centred on one object of the requested class.",
        "Return the tight box around the visible extent of that one object (do not extend it to hidden parts).",
        "If several objects of the class are visible, box only the one nearest the centre of the crop.",
        "If no object of the class is visible, return an empty list.",
        "", "Coordinate format: " + frame_txt + "."])


def refine_user_text(label: str, frame: str, sw: int, sh: int, schema: Optional[dict] = None) -> str:
    lab = json.dumps(label, ensure_ascii=False)
    txt = (f"Image size: {sw}x{sh} px. Return the tight bounding box of the single {lab} {REFINE_MARK}. "
           f"Coordinates: {FRAME_SENTENCE[frame].format(w=sw, h=sh)}.")
    hint = shape_hint(schema) if schema else ""
    return txt + (f" Answer as JSON: {hint}" if hint else "")


# ------------------------------------------------------------------------------------- generic JSON (generate_json)

_GEN_DROP = {
    # keywords each structured-output dialect rejects (validated client-side instead, by the caller)
    "anthropic": {"minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "maxItems", "minLength",
                  "maxLength", "minProperties", "maxProperties", "default", "examples", "uniqueItems"},
    "openai": {"default", "examples", "minLength", "maxLength", "uniqueItems", "minProperties", "maxProperties"},
    "gemini": {"pattern", "minLength", "maxLength", "default", "examples", "exclusiveMinimum", "exclusiveMaximum",
               "multipleOf", "uniqueItems", "minProperties", "maxProperties"},
    "compat": {"default", "examples", "uniqueItems", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
               "multipleOf", "minLength", "maxLength", "maxItems", "minProperties", "maxProperties"},
}


def wrap_schema(schema: dict) -> Tuple[dict, bool]:
    """Every API wants an object at the root; an array / scalar schema is wrapped as ``{"result": ...}``."""
    if not isinstance(schema, dict):
        raise ProviderError("generate_json needs a JSON schema object", "invalid")
    if schema.get("type") == "object" and isinstance(schema.get("properties"), dict):
        return schema, False
    return {"type": "object", "properties": {"result": schema}, "required": ["result"],
            "additionalProperties": False}, True


def schema_for(dialect: str, schema: Any, *, strict: bool = False) -> Any:
    """Adapt a caller's JSON schema to one API's structured-output dialect (deep copy).

    ``strict`` (OpenAI ``strict: true`` rules): every object lists all its properties in ``required`` — optional ones
    become nullable — and sets ``additionalProperties: false``. Claude also requires ``additionalProperties: false``
    and only supports ``minItems`` 0 / 1."""
    drop = _GEN_DROP.get(dialect, set())
    if isinstance(schema, list):
        return [schema_for(dialect, s, strict=strict) for s in schema]
    if not isinstance(schema, dict):
        return schema
    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in drop:
            continue
        if k == "minItems" and dialect == "anthropic" and isinstance(v, int) and v > 1:
            out[k] = 1
            continue
        if k == "const" and dialect == "gemini":
            out["enum"] = [v]
            continue
        if k in ("properties", "$defs", "definitions") and isinstance(v, dict):
            out[k] = {pk: schema_for(dialect, pv, strict=strict) for pk, pv in v.items()}
        elif k in ("items", "anyOf", "oneOf", "allOf", "prefixItems", "additionalProperties") and \
                isinstance(v, (dict, list)):
            out[k] = schema_for(dialect, v, strict=strict)
        else:
            out[k] = v
    t = out.get("type")
    if t == "object" or (isinstance(t, list) and "object" in t) or "properties" in out:
        props = out.get("properties") or {}
        if dialect in ("anthropic", "openai", "compat") or strict:
            out["additionalProperties"] = False
        if strict and props:
            req = list(out.get("required") or [])
            for pk in props:
                if pk not in req:
                    props[pk] = _nullable(props[pk])
            out["required"] = list(props.keys())
    return out


def encode_maps(schema: Any, path: Tuple = ()) -> Tuple[Any, List[Tuple]]:
    """Strict structured outputs cannot express free-form maps (``additionalProperties: {schema}`` without
    ``properties``). Each such map becomes an array of ``{key, value}`` pairs; ``decode_maps`` turns the reply back.
    Returns (schema, paths of converted maps)."""
    if not isinstance(schema, dict):
        return schema, []
    paths: List[Tuple] = []
    ap = schema.get("additionalProperties")
    if schema.get("type") == "object" and not schema.get("properties") and isinstance(ap, dict):
        val, sub = encode_maps(ap, path + ("[]", "value"))
        return ({"type": "array", "description": schema.get("description", "") or "key/value pairs",
                 "items": {"type": "object", "properties": {"key": {"type": "string"}, "value": val},
                           "required": ["key", "value"], "additionalProperties": False}},
                [path] + sub)
    out = dict(schema)
    if isinstance(schema.get("properties"), dict):
        props = {}
        for k, v in schema["properties"].items():
            props[k], sub = encode_maps(v, path + (k,))
            paths += sub
        out["properties"] = props
    if isinstance(schema.get("items"), dict):
        out["items"], sub = encode_maps(schema["items"], path + ("[]",))
        paths += sub
    return out, paths


def decode_maps(value: Any, paths: Sequence[Tuple]) -> Any:
    """Inverse of ``encode_maps`` on a parsed reply (deepest paths first so nested maps decode inside out)."""
    for p in sorted(paths, key=len, reverse=True):
        value = _decode_at(value, p)
    return value


def _decode_at(value: Any, path: Tuple) -> Any:
    if not path:
        if isinstance(value, list):
            out = {}
            for it in value:
                if isinstance(it, dict) and "key" in it:
                    out[str(it["key"])] = it.get("value")
            return out
        return value
    step, rest = path[0], path[1:]
    if step == "[]":
        return [_decode_at(v, rest) for v in value] if isinstance(value, list) else value
    if isinstance(value, dict) and step in value:
        value = dict(value)
        value[step] = _decode_at(value[step], rest)
    return value


def _nullable(s: Any) -> Any:
    if not isinstance(s, dict):
        return s
    s = dict(s)
    t = s.get("type")
    if isinstance(t, str) and t != "null":
        s["type"] = [t, "null"]
    elif isinstance(t, list) and "null" not in t:
        s["type"] = t + ["null"]
    elif "anyOf" in s and isinstance(s["anyOf"], list):
        s["anyOf"] = s["anyOf"] + [{"type": "null"}]
    if isinstance(s.get("enum"), list) and None not in s["enum"]:
        s["enum"] = s["enum"] + [None]
    return s


def check_json_shape(value: Any, schema: dict) -> Optional[str]:
    """Light structural check of a reply against the caller's schema (root type and required keys, one level deep):
    servers without constrained decoding may answer something JSON-shaped but wrong. None when it looks right."""
    t = schema.get("type")
    want = {"object": dict, "array": list, "string": str, "integer": int, "number": (int, float), "boolean": bool}
    if isinstance(t, str) and t in want and not isinstance(value, want[t]):
        return f"expected a JSON {t}, got {type(value).__name__}"
    if isinstance(value, dict):
        missing = [k for k in (schema.get("required") or []) if k not in value]
        if missing:
            return f"missing {', '.join(missing[:4])}"
    return None


def sniff_mime(data: bytes) -> Optional[str]:
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return "image/gif"
    return None


def prepare_blob(data: bytes, max_edge: int, max_bytes: int = 3_500_000) -> Tuple[bytes, str]:
    """An image for ``generate_json``: passed through when it is a common format within the size limits, otherwise
    decoded (EXIF-aware), shrunk to ``max_edge`` and re-encoded as JPEG."""
    from PIL import Image, ImageOps
    if not isinstance(data, (bytes, bytearray)) or not data:
        raise ProviderError("generate_json images must be non-empty bytes", "invalid")
    data = bytes(data)
    mime = sniff_mime(data)
    try:
        im = Image.open(io.BytesIO(data))
        size = im.size
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"Cannot read an attached image: {e}", "invalid") from None
    if mime in ("image/jpeg", "image/png", "image/webp") and len(data) <= max_bytes and max(size) <= max_edge:
        return data, mime
    try:
        im = ImageOps.exif_transpose(im)
        im = im.convert("RGB")
        im.thumbnail((max_edge, max_edge), Image.LANCZOS)
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=90)
    except Exception as e:  # noqa: BLE001
        raise ProviderError(f"Cannot decode an attached image: {e}", "invalid") from None
    return buf.getvalue(), "image/jpeg"


# --------------------------------------------------------------------------------------------- JSON extraction

class BadOutput(Exception):
    pass


_FENCE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str) -> Any:
    """Parse JSON from a model reply: plain JSON, fenced, or embedded in prose. Raises ``BadOutput``."""
    if text is None:
        raise BadOutput("empty reply")
    s = str(text).strip().lstrip("﻿")
    if not s:
        raise BadOutput("empty reply")
    try:
        return json.loads(s)
    except ValueError:
        pass
    for m in _FENCE.finditer(s):
        try:
            return json.loads(m.group(1).strip())
        except ValueError:
            continue
    dec = json.JSONDecoder()
    for i, ch in enumerate(s):
        if ch in "[{":
            try:
                obj, _ = dec.raw_decode(s, i)
                if isinstance(obj, (list, dict)):
                    return obj
            except ValueError:
                continue
    raise BadOutput("no JSON found in the reply")


def salvage_items(text: str, key: str = "objects") -> List[dict]:
    """Complete ``{...}`` items from a truncated JSON array (output cut off by the token limit)."""
    s = str(text or "")
    m = re.search(r'"%s"\s*:\s*\[' % re.escape(key), s)
    start = m.end() if m else (s.find("[") + 1 if "[" in s else -1)
    if start <= 0:
        return []
    dec = json.JSONDecoder()
    out, i, n = [], start, len(s)
    while i < n:
        while i < n and s[i] in " \t\r\n,":
            i += 1
        if i >= n or s[i] != "{":
            break
        try:
            obj, j = dec.raw_decode(s, i)
        except ValueError:
            break
        if isinstance(obj, dict):
            out.append(obj)
        i = j
    return out


def items_from(obj: Any, key: str = "objects") -> List[dict]:
    if isinstance(obj, list):
        return [x for x in obj if isinstance(x, dict)]
    if isinstance(obj, dict):
        for k in (key, "objects", "detections", "boxes", "items", "results", "predictions"):
            v = obj.get(k)
            if isinstance(v, list):
                return [x for x in v if isinstance(x, dict)]
        if any(k in obj for k in ("box_2d", "bbox_2d", "bbox", "x1", "box")):
            return [obj]
    return []


_ARRAY_KEYS = ("box_2d", "bbox_2d", "bbox", "box", "xyxy", "coordinates")


def item_box(item: dict, frame: str) -> Optional[Tuple[float, float, float, float]]:
    """Raw 4 numbers of an item in the provider's xyxy order (frame order already undone)."""
    try:
        if all(k in item for k in ("x1", "y1", "x2", "y2")):
            return tuple(float(item[k]) for k in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
        if all(k in item for k in ("xmin", "ymin", "xmax", "ymax")):
            return tuple(float(item[k]) for k in ("xmin", "ymin", "xmax", "ymax"))  # type: ignore[return-value]
        for k in _ARRAY_KEYS:
            v = item.get(k)
            if isinstance(v, (list, tuple)) and len(v) >= 4 and all(isinstance(x, (int, float)) for x in v[:4]):
                return _to_xyxy_order(v, frame)
            if isinstance(v, dict) and all(kk in v for kk in ("x1", "y1", "x2", "y2")):
                return tuple(float(v[kk]) for kk in ("x1", "y1", "x2", "y2"))  # type: ignore[return-value]
    except (TypeError, ValueError):
        return None
    return None


def item_label(item: dict) -> Any:
    for k in ("label", "class", "name", "category", "class_name"):
        if k in item:
            return item[k]
    return None


# --------------------------------------------------------------------------------------------- geometry / merge

@dataclass
class Box:
    cls: int
    x1: float
    y1: float
    x2: float
    y2: float
    label: str = ""
    conf: Optional[float] = None
    edge: bool = False           # touches an inner tile border (may be a cut-off part of a larger object)
    tile: int = 0                # 0 = full-image pass, k = k-th tile

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)


def iou_xyxy(a: Box, b: Box) -> float:
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    inter = ix * iy
    union = a.area + b.area - inter
    return inter / union if union > 0 else 0.0


def contained(a: Box, b: Box) -> float:
    """Fraction of ``a`` inside ``b``."""
    ix = max(0.0, min(a.x2, b.x2) - max(a.x1, b.x1))
    iy = max(0.0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return (ix * iy) / a.area if a.area > 0 else 0.0


def nms(boxes: List[Box], iou_thr: float = 0.5) -> List[Box]:
    """Class-aware NMS. Boxes keep their order of preference (callers put higher confidence / full-pass first)."""
    keep: List[Box] = []
    for b in boxes:
        if all(b.cls != k.cls or iou_xyxy(b, k) < iou_thr for k in keep):
            keep.append(b)
    return keep


def merge_tiles(boxes: List[Box], iou_thr: float = 0.5) -> List[Box]:
    """Merge full-image and tile boxes: NMS, then drop tile boxes that are cut by a tile border and lie mostly inside a
    same-class box from another pass (the full object)."""
    def pref(b: Box):
        return (b.conf if b.conf is not None else 1.0, -int(b.edge), b.area)
    ordered = sorted(boxes, key=pref, reverse=True)
    kept = nms(ordered, iou_thr)
    out = []
    for b in kept:
        if b.edge and any(o is not b and o.cls == b.cls and o.tile != b.tile and o.area > b.area
                          and contained(b, o) >= 0.7 for o in kept):
            continue
        out.append(b)
    return out


def vote_merge(runs: List[List[Box]], iou_thr: float = 0.5) -> List[Box]:
    """Self-consistency: match boxes of N independent runs (same class, IoU >= thr, greedy); conf = votes / N and the
    box is the mean of the matched boxes."""
    n = max(1, len(runs))
    clusters: List[List[Box]] = []
    for run in runs:
        used = set()
        for b in run:
            best, best_iou = None, iou_thr
            for ci, cl in enumerate(clusters):
                if ci in used or cl[0].cls != b.cls:
                    continue
                ref = _mean_box(cl)
                v = iou_xyxy(b, ref)
                if v >= best_iou:
                    best, best_iou = ci, v
            if best is None:
                clusters.append([b])
                used.add(len(clusters) - 1)
            else:
                clusters[best].append(b)
                used.add(best)
    out = []
    for cl in clusters:
        m = _mean_box(cl)
        m.conf = round(min(1.0, len(cl) / n), 4)
        out.append(m)
    out.sort(key=lambda b: (-(b.conf or 0), -b.area))
    return out


def _mean_box(cl: List[Box]) -> Box:
    k = len(cl)
    return Box(cl[0].cls, sum(b.x1 for b in cl) / k, sum(b.y1 for b in cl) / k, sum(b.x2 for b in cl) / k,
               sum(b.y2 for b in cl) / k, cl[0].label)


def to_detection(b: Box, nd: int = 6) -> Optional[Detection]:
    x1, x2 = sorted((min(max(b.x1, 0.0), 1.0), min(max(b.x2, 0.0), 1.0)))
    y1, y2 = sorted((min(max(b.y1, 0.0), 1.0), min(max(b.y2, 0.0), 1.0)))
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h <= 0:
        return None
    return Detection(cls=int(b.cls), xc=round((x1 + x2) / 2, nd), yc=round((y1 + y2) / 2, nd), w=round(w, nd),
                     h=round(h, nd), conf=b.conf, label=b.label)


# ---------------------------------------------------------------------------------------------- retries

class Transient(Exception):
    """A retryable failure (429, 5xx/529, timeouts, connection errors)."""

    def __init__(self, message: str, code: str = "overloaded", retry_after: Optional[float] = None,
                 status: Optional[int] = None):
        super().__init__(message)
        self.code = code            # rate | overloaded | network
        self.retry_after = retry_after
        self.status = status


@dataclass
class RetryPolicy:
    tries: int = 5               # total attempts
    base: float = 1.0
    cap: float = 60.0
    max_wait: float = 300.0      # a retry-after longer than this is not waited for (fail now, let the job decide)


RETRY = RetryPolicy()
_sleep = time.sleep              # tests patch this
_wait_listener: contextvars.ContextVar[Optional[Callable[[int, str], None]]] = contextvars.ContextVar(
    "fovea_vlm_wait_listener", default=None)


@contextlib.contextmanager
def wait_listener(fn: Optional[Callable[[int, str], None]]):
    """Report retry waits to a job: ``fn(seconds_left, reason)`` is called about once a second while a provider waits
    (e.g. ``job.update(message=f"Rate limited — waiting {s} s")``). If ``fn`` raises (``JobCancelled``) the wait is
    aborted and the exception propagates, so cancelling a job never waits out a long ``retry-after``."""
    tok = _wait_listener.set(fn)
    try:
        yield
    finally:
        _wait_listener.reset(tok)


def _notify(seconds: float, reason: str) -> None:
    fn = _wait_listener.get()
    if fn is not None:
        fn(max(0, int(math.ceil(seconds))), reason)


_waits: Dict[int, dict] = {}       # thread id -> {reason, until}: retry waits in progress, for job status lines


def current_waits() -> List[dict]:
    """Retry waits in progress in any thread: [{reason, seconds_left}] (longest first). Lets a batch job's status line
    say "rate limited — retrying in 12 s" without plumbing a listener into its worker threads."""
    now = time.time()
    out = [{"reason": w["reason"], "seconds_left": max(0, int(math.ceil(w["until"] - now)))}
           for w in list(_waits.values())]
    return sorted(out, key=lambda w: -w["seconds_left"])


def _wait(seconds: float, reason: str) -> None:
    left = float(seconds)
    tid = threading.get_ident()
    _waits[tid] = {"reason": reason, "until": time.time() + left}
    try:
        while left > 0:
            _notify(left, reason)
            step = min(1.0, left)
            _sleep(step)
            left -= step
        _notify(0, reason)
    finally:
        _waits.pop(tid, None)


def backoff_delay(attempt: int, policy: RetryPolicy = RETRY) -> float:
    """Exponential backoff with full jitter on the upper half: attempt 0 → ~0.5-1 s, 1 → 1-2 s, …, capped."""
    d = min(policy.cap, policy.base * (2 ** attempt))
    return d / 2 + random.uniform(0, d / 2)


def with_retries(call: Callable[[], Any], *, provider: str = "", policy: Optional[RetryPolicy] = None,
                 limiter: Optional["AdaptiveLimiter"] = None) -> Any:
    """Run ``call`` retrying ``Transient`` failures. ``ProviderError`` passes straight through (abort/skip codes)."""
    policy = policy or RETRY
    last: Optional[Transient] = None
    for attempt in range(max(1, policy.tries)):
        try:
            if limiter is not None:
                with limiter:
                    res = call()
            else:
                res = call()
            if limiter is not None:
                limiter.success()
            return res
        except Transient as t:
            last = t
            if limiter is not None and t.code == "rate":
                limiter.rate_limited()
            if attempt + 1 >= policy.tries:
                break
            delay = backoff_delay(attempt, policy)
            if t.retry_after is not None:
                if t.retry_after > policy.max_wait:
                    break
                delay = max(float(t.retry_after), 0.0) + random.uniform(0, 0.25 * max(1.0, float(t.retry_after)))
            delay = min(delay, policy.max_wait)
            why = {"rate": "rate limited", "overloaded": "provider busy", "network": "connection problem"}.get(t.code,
                                                                                                             t.code)
            log.info("%s: %s (%s); retry %d/%d in %.1f s", provider, why, t.status, attempt + 1, policy.tries - 1,
                     delay)
            _wait(delay, f"{provider + ' ' if provider else ''}{why}")
    assert last is not None
    code = last.code if last.code in ("rate", "overloaded", "network") else "overloaded"
    raise ProviderError(f"{last} (gave up after {policy.tries} attempts)", code, retry_after=last.retry_after,
                        status=last.status)


def parse_retry_after(headers: Any, body: Any = None) -> Optional[float]:
    """Seconds to wait from ``retry-after`` / ``retry-after-ms`` / OpenAI reset headers / Gemini ``RetryInfo``."""
    def h(name: str) -> Optional[str]:
        if headers is None:
            return None
        try:
            v = headers.get(name)
        except Exception:  # noqa: BLE001
            return None
        return v if v not in (None, "") else None

    v = h("retry-after-ms")
    if v:
        try:
            return max(0.0, float(v) / 1000.0)
        except ValueError:
            pass
    v = h("retry-after")
    if v:
        try:
            return max(0.0, float(v))
        except ValueError:
            try:
                dt = parsedate_to_datetime(v)
                return max(0.0, dt.timestamp() - time.time())
            except (TypeError, ValueError):
                pass
    # Gemini: error.details[].retryDelay "37s"
    try:
        details = (body or {}).get("error", {}).get("details") or []
        for d in details:
            rd = d.get("retryDelay") if isinstance(d, dict) else None
            if rd:
                return _parse_duration(rd)
    except AttributeError:
        pass
    for name in ("x-ratelimit-reset-requests", "x-ratelimit-reset-tokens"):
        v = h(name)
        if v:
            d = _parse_duration(v)
            if d is not None:
                return d
    return None


def _parse_duration(s: str) -> Optional[float]:
    """'37s', '1.5s', '6m0s', '250ms', '1h2m' → seconds."""
    s = str(s).strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        pass
    total, found = 0.0, False
    for num, unit in re.findall(r"([\d.]+)\s*(ms|h|m|s)", s):
        found = True
        total += float(num) * {"ms": 0.001, "h": 3600, "m": 60, "s": 1}[unit]
    return total if found else None


def error_message(body: Any, text: str = "") -> str:
    msg = ""
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            msg = err.get("message") or err.get("msg") or ""
        elif isinstance(err, str):
            msg = err
        if not msg:
            msg = body.get("message") or body.get("detail") or ""
    if not msg:
        msg = (text or "").strip()
    msg = re.sub(r"\s+", " ", str(msg))[:300]
    return msg


def http_failure(provider_name: str, status: int, body: Any, headers: Any, text: str = "", *,
                 quota: bool = False, auth: bool = False, secrets: Sequence[str] = ()) -> Exception:
    """Map an HTTP error to ``Transient`` (retry) or ``ProviderError`` (skip / abort).

    ``quota`` / ``auth``: provider-specific detection already decided this is a spend-cap / billing failure or a
    rejected key (e.g. Gemini returns 400 for a bad key, Anthropic 429 for an enforced spend limit)."""
    msg = scrub(error_message(body, text), *secrets)
    tail = f": {msg}" if msg else ""
    if auth or status == 401:
        return ProviderError(f"{provider_name} rejected the API key ({status}){tail}", "auth", status=status)
    if quota or status == 402:
        return ProviderError(f"{provider_name} spend limit or billing problem ({status}){tail}", "quota", status=status)
    if status == 403:
        return ProviderError(f"{provider_name} denied access ({status}){tail}", "auth", status=status)
    if status == 404:
        return ProviderError(f"{provider_name}: model or endpoint not found (404){tail}", "invalid", status=status)
    if status == 413:
        return ProviderError(f"{provider_name}: request too large (413){tail}", "invalid", status=status)
    if status == 429:
        return Transient(f"{provider_name} rate limit (429){tail}", "rate", parse_retry_after(headers, body), status)
    if status in (408, 409) or status >= 500:
        return Transient(f"{provider_name} unavailable ({status}){tail}", "overloaded", parse_retry_after(headers, body),
                         status)
    return ProviderError(f"{provider_name} rejected the request ({status}){tail}", "invalid", status=status)


_http_clients: Dict[Tuple[float, int], Any] = {}
_http_lock = threading.Lock()


HTTPX_MISSING = "The 'httpx' Python package is not installed (pip install httpx)"


def httpx_available() -> bool:
    """Whether the HTTP client the Gemini / OpenAI / OpenAI-compatible providers need is installed (no import: cheap
    enough for every provider listing). A core-only install leaves it out, and those providers must say so."""
    try:
        return importlib.util.find_spec("httpx") is not None
    except (ImportError, ValueError):
        return False


def require_httpx():
    """``import httpx``, or a user-facing ProviderError ('unavailable' aborts a batch: every image would fail alike)
    instead of a ModuleNotFoundError that reaches the client as HTTP 500."""
    try:
        import httpx
    except ImportError as e:
        if (getattr(e, "name", None) or "httpx").split(".")[0] == "httpx":
            raise ProviderError(HTTPX_MISSING, "unavailable") from None
        raise ProviderError(f"The 'httpx' Python package could not be loaded: its dependency '{e.name}' is missing "
                            "(pip install --force-reinstall httpx)", "unavailable") from None
    return httpx


def http_client(read_timeout: float = 180.0):
    """Shared pooled ``httpx.Client`` (thread-safe) per read timeout."""
    httpx = require_httpx()
    k = (float(read_timeout), os.getpid())
    with _http_lock:
        c = _http_clients.get(k)
        if c is None:
            c = _http_clients[k] = httpx.Client(timeout=httpx.Timeout(connect=10.0, read=read_timeout, write=60.0,
                                                                      pool=30.0),
                                                limits=httpx.Limits(max_connections=32, max_keepalive_connections=16),
                                                # gzip/deflate only: some Brotli/HTTP library pairings cannot decode
                                                # 'br' answers (see providers/anthropic.py); answers are small JSON
                                                headers={"Accept-Encoding": "gzip, deflate"},
                                                follow_redirects=False)
        return c


def _http(method: str, url: str, provider_name: str, *, headers: Optional[dict] = None, timeout: float = 180.0,
          secrets: Sequence[str] = (), classify: Optional[Callable[[int, Any, str], Tuple[bool, bool]]] = None,
          **kw) -> Any:
    """One HTTP request with Fovea's error mapping: raises ``Transient`` (retry) / ``ProviderError`` (scrubbed)."""
    httpx = require_httpx()
    try:
        r = http_client(timeout).request(method, url, headers=headers, **kw)
    except httpx.TimeoutException:
        raise Transient(f"{provider_name} request timed out", "network")
    except httpx.ConnectError as e:
        if is_loopback_url(url):        # nothing listening on this machine: retrying will not start it
            raise ProviderError(f"Cannot reach {url_host(url)} — is the server running?", "network") from None
        raise Transient(f"Cannot reach {url_host(url)}: {scrub(e, *secrets)}", "network")
    except httpx.HTTPError as e:
        raise Transient(f"Cannot reach {url_host(url)}: {scrub(e, *secrets)}", "network")
    if r.status_code >= 400:
        text = r.text if r.content else ""
        try:
            data = r.json() if text else None
        except ValueError:
            data = None
        quota, auth = classify(r.status_code, data, text) if classify else (False, False)
        raise http_failure(provider_name, r.status_code, data, r.headers, text, quota=quota, auth=auth,
                           secrets=secrets)
    return r


def http_json(method: str, url: str, provider_name: str, *, headers: Optional[dict] = None, body: Any = None,
              timeout: float = 180.0, secrets: Sequence[str] = (),
              classify: Optional[Callable[[int, Any, str], Tuple[bool, bool]]] = None, **kw) -> Tuple[Any, Any]:
    """One JSON request. Returns (json, response headers). Raises ``Transient`` / ``ProviderError`` (scrubbed).
    ``classify(status, json, text) -> (quota, auth)`` lets a provider flag spend-cap / bad-key responses.
    Extra keyword arguments (``files``, ``data`` for multipart uploads) go to httpx instead of a JSON body."""
    if body is not None:
        kw["json"] = body
    r = _http(method, url, provider_name, headers=headers, timeout=timeout, secrets=secrets, classify=classify, **kw)
    text = r.text if r.content else ""
    try:
        data = r.json() if text else None
    except ValueError:
        data = None
    if data is None:
        raise ProviderError(f"{provider_name} returned a non-JSON response ({r.status_code})", "bad_output",
                            status=r.status_code)
    return data, r.headers


def http_bytes(method: str, url: str, provider_name: str, *, headers: Optional[dict] = None, timeout: float = 300.0,
               secrets: Sequence[str] = (), classify: Optional[Callable[[int, Any, str], Tuple[bool, bool]]] = None
               ) -> bytes:
    """Raw response body (batch result files are JSONL, not JSON)."""
    return _http(method, url, provider_name, headers=headers, timeout=timeout, secrets=secrets,
                 classify=classify).content


def jsonl(data: bytes) -> Iterator[dict]:
    for line in (data or b"").splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


class AdaptiveLimiter:
    """Per-provider concurrency gate: starts at ``limit``, halves on each 429, creeps back up after successes."""

    def __init__(self, limit: int = 4):
        self.max = max(1, int(limit))
        self.limit = self.max
        self.active = 0
        self._ok = 0
        self._cv = threading.Condition()

    def set_max(self, limit: int) -> None:
        with self._cv:
            self.max = max(1, int(limit))
            self.limit = min(self.limit, self.max) if self.limit else self.max
            self._cv.notify_all()

    def __enter__(self):
        with self._cv:
            while self.active >= self.limit:
                self._cv.wait(timeout=1.0)
            self.active += 1
        return self

    def __exit__(self, *exc):
        with self._cv:
            self.active -= 1
            self._cv.notify_all()
        return False

    def rate_limited(self) -> None:
        with self._cv:
            self.limit = max(1, self.limit // 2)
            self._ok = 0

    def success(self) -> None:
        with self._cv:
            self._ok += 1
            if self._ok >= 10 and self.limit < self.max:
                self.limit += 1
                self._ok = 0
                self._cv.notify_all()


_limiters: Dict[str, AdaptiveLimiter] = {}
_limiters_lock = threading.Lock()


def limiter_for(key: str, limit: int) -> AdaptiveLimiter:
    with _limiters_lock:
        lim = _limiters.get(key)
        if lim is None:
            lim = _limiters[key] = AdaptiveLimiter(limit)
        elif lim.max != max(1, int(limit)):
            lim.set_max(limit)
        return lim


# -------------------------------------------------------------------------------------------- provider base

@dataclass
class CallResult:
    text: str
    usage: Usage = field(default_factory=Usage)
    served_by: str = ""
    finish: str = "stop"          # stop | truncated | refusal
    refusal: str = ""             # refusal category / explanation (no secrets)


EFFORTS = ("none", "minimal", "low", "medium", "high", "xhigh", "max")


def pick_effort(requested: str, supported: Sequence[str]) -> str:
    """The supported level closest to ``requested`` (rounding down, then up): ``xhigh`` on a model that stops at
    ``high`` becomes ``high``; ``minimal`` on a model that starts at ``low`` becomes ``low``."""
    sup = [e for e in EFFORTS if e in supported]
    if not sup:
        return requested
    if requested in sup:
        return requested
    r = EFFORTS.index(requested) if requested in EFFORTS else EFFORTS.index("medium")
    lower = [e for e in sup if EFFORTS.index(e) <= r]
    return lower[-1] if lower else sup[0]


FLEX_SLOWDOWN = 3.0                # Flex-tier answers queue behind standard traffic (minutes at busy times)
PROMPT_FRAMING_TOKENS = 10      # role / turn framing billed per request beyond the prompt text itself
THINKING_TOKENS = {"none": (0, 0), "minimal": (0, 300), "low": (0, 1200), "medium": (300, 3500), "high": (800, 7000),
                   "xhigh": (1500, 12000), "max": (3000, 20000)}


# --------------------------------------------------------------------------------------- detection plan / merge

@dataclass
class DetectPlan:
    """One image's detection, decided before any request (and rebuilt from JSON to decode provider-batch results)."""
    model: str
    frame: str
    vocab: Vocab
    options: Dict[str, Any]
    max_objects: int
    samples: int
    conf: float
    preps: List[Prepared]
    warnings: List[str] = field(default_factory=list)
    refine: Optional[dict] = None
    image: Any = None            # decoded original (refine crops); None when rebuilt for a batch

    def context(self, vocab_index: int) -> dict:
        return {"model": self.model, "frame": self.frame, "max_objects": self.max_objects, "samples": self.samples,
                "conf": self.conf, "preps": [p.geometry() for p in self.preps], "warnings": list(self.warnings),
                "vocab": vocab_index}

    @classmethod
    def from_context(cls, info: dict, vocab_rows: Sequence[Sequence[Any]]) -> "DetectPlan":
        specs = [ClassSpec(int(r[0]), str(r[1]), str(r[2] or ""), str(r[3] or "")) for r in vocab_rows]
        return cls(model=info["model"], frame=info["frame"], vocab=Vocab.build(specs), options={},
                   max_objects=int(info["max_objects"]), samples=int(info["samples"]),
                   conf=float(info.get("conf") or 0), preps=[Prepared.from_geometry(g) for g in info["preps"]], warnings=list(info.get("warnings") or []))


class DetectAccumulator:
    """Usage, warnings and counters of one image's detection across samples / tiles / refine calls."""

    def __init__(self, plan: DetectPlan):
        self.plan = plan
        self.stats: Dict[str, Any] = {"calls": 0, "scale_anomalies": 0, "unknown_labels": 0, "degenerate": 0,
                                      "truncated": 0, "salvaged": 0, "duplicates": 0, "capped": 0,
                                      "tiles": len(plan.preps) - 1, "samples": plan.samples}
        self.usage = Usage()
        self.served: List[str] = []
        self.warnings: List[str] = list(plan.warnings)
        self.unknown: Dict[str, int] = {}
        self.anomalies: List[str] = []
        self.runs: List[List[Box]] = []

    def add_usage(self, res: "CallResult", notes: Sequence[str] = ()) -> None:
        self.usage.add(res.usage)
        if res.served_by and res.served_by not in self.served:
            self.served.append(res.served_by)
        for n in notes:
            if n not in self.warnings:
                self.warnings.append(n)

    def add_run(self, run: List[Box]) -> None:
        if len(self.plan.preps) > 1:
            run = merge_tiles(run)
        else:
            before = len(run)
            run = nms(run, 0.9)
            self.stats["duplicates"] += before - len(run)
        self.runs.append(run)

    def merged(self) -> List[Box]:
        boxes = vote_merge(self.runs) if self.plan.samples > 1 else (self.runs[0] if self.runs else [])
        if self.plan.conf and self.plan.samples > 1:
            boxes = [b for b in boxes if b.conf is None or b.conf >= float(self.plan.conf)]
        return boxes


REFINE_DEFAULTS = {"small": 0.02, "max_boxes": 8, "iterations": 1, "context": 2.5, "min_px": 256, "zoom_edge": 768,
                   "all": False}


def refine_settings(value: Any) -> Optional[dict]:
    """``options.refine``: false/None (off) · true / "auto" (small < 2% of the image or low-agreement boxes) ·
    "all" (every box, up to max_boxes) · {small, max_boxes (<= 30), iterations (1-2), context (1.5-4), min_px,
    zoom_edge, all}."""
    if not value:
        return None
    cfg = dict(REFINE_DEFAULTS)
    if value == "all":
        cfg["all"] = True
    elif isinstance(value, dict):
        for k in cfg:
            if k in value and value[k] is not None:
                cfg[k] = value[k]
    elif value not in (True, "auto", 1, "1", "true", "on"):
        return None
    try:
        cfg["small"] = min(max(float(cfg["small"]), 0.0), 1.0)
        cfg["max_boxes"] = min(max(int(cfg["max_boxes"]), 1), 30)
        cfg["iterations"] = min(max(int(cfg["iterations"]), 1), 2)    # C2F-Space: no gain beyond two
        cfg["context"] = min(max(float(cfg["context"]), 1.5), 4.0)
        cfg["min_px"] = min(max(int(cfg["min_px"]), 64), 2048)
        cfg["zoom_edge"] = min(max(int(cfg["zoom_edge"]), 256), 2048)
        cfg["all"] = bool(cfg["all"])
    except (TypeError, ValueError):
        raise ProviderError("options.refine has an invalid value", "invalid") from None
    return cfg


def refine_region(b: Box, W: int, H: int, context: float, min_px: int) -> Tuple[float, float, float, float]:
    """Context window around a box (normalised coords) in original pixels: ``context`` x the box, at least
    ``min_px`` per side, shifted to stay inside the image."""
    bw, bh = (b.x2 - b.x1) * W, (b.y2 - b.y1) * H
    cx, cy = (b.x1 + b.x2) / 2 * W, (b.y1 + b.y2) / 2 * H
    cw = min(float(W), max(context * bw, float(min_px)))
    ch = min(float(H), max(context * bh, float(min_px)))
    ox = min(max(cx - cw / 2, 0.0), W - cw)
    oy = min(max(cy - ch / 2, 0.0), H - ch)
    return ox, oy, cw, ch


def refine_accept(old: Box, new: Box, region: Tuple[float, float, float, float], W: int, H: int) -> bool:
    """LFPR guards (arXiv 2608.19553): IoU(old, new) >= 0.25, area ratio within [0.5, 2.0], centre shift <= 0.35 of
    the context window. Otherwise the first box stays."""
    if iou_xyxy(old, new) < 0.25:
        return False
    if old.area <= 0 or not 0.5 <= new.area / old.area <= 2.0:
        return False
    _ox, _oy, cw, ch = region
    dx = abs((new.x1 + new.x2) - (old.x1 + old.x2)) / 2 * W
    dy = abs((new.y1 + new.y2) - (old.y1 + old.y2)) / 2 * H
    return dx <= 0.35 * cw and dy <= 0.35 * ch


# ------------------------------------------------------------------------------------------- provider batches

_CUSTOM_ID = re.compile(r"[A-Za-z0-9_-]{1,64}")


def _spec(c: Any) -> ClassSpec:
    if isinstance(c, ClassSpec):
        return c
    if isinstance(c, dict) and "id" in c:
        return ClassSpec(int(c["id"]), str(c.get("name") or f"class_{c['id']}"), str(c.get("prompt") or ""),
                         str(c.get("description") or ""))
    raise ProviderError("batch item classes must be ClassSpec or {id, name, prompt?, description?}", "invalid")


def _batch_item(it: Any) -> Tuple[Any, DetectRequest]:
    """``{"custom_id", "request": DetectRequest}`` · ``{"custom_id", "image_path", "classes", "model", "conf"?,
    "options"?}`` · ``(custom_id, DetectRequest)``."""
    cid, req = None, None
    if isinstance(it, (tuple, list)) and len(it) == 2:
        cid, req = it
    elif isinstance(it, dict):
        cid = it.get("custom_id", it.get("image_id", it.get("id")))
        req = it.get("request")
        if req is None and it.get("image_path"):
            req = DetectRequest(image_path=str(it["image_path"]), classes=[_spec(c) for c in it.get("classes") or []],
                                model=str(it.get("model") or ""), conf=it.get("conf") or 0.0,
                                options=dict(it.get("options") or {}))
        elif isinstance(req, dict):
            req = DetectRequest(image_path=str(req.get("image_path") or ""),
                                classes=[_spec(c) for c in req.get("classes") or []], model=str(req.get("model") or ""),
                                conf=req.get("conf") or 0.0, options=dict(req.get("options") or {}))
    if cid is None or str(cid) == "" or not isinstance(req, DetectRequest):
        raise ProviderError("each batch item needs a custom_id and a detection request", "invalid")
    return cid, req


def _sub_id(key: str, sample: int, tile: int, multi: bool, n: int) -> str:
    """Provider-side custom id of one request (Anthropic: ``^[a-zA-Z0-9_-]{1,64}$``)."""
    base = key if _CUSTOM_ID.fullmatch(key) else f"i{n}"
    sid = f"{base}-s{sample}t{tile}" if multi else base
    return sid if _CUSTOM_ID.fullmatch(sid) else f"r{n}"


BATCH_CTX_DAYS = 40              # provider results live 29 days (Anthropic) / 6 weeks (Gemini)


def _batch_dir() -> Path:
    from ..config import DATA_DIR
    return Path(DATA_DIR) / "ai_batches"


def _batch_ctx_path(provider: str, batch_id: str) -> Path:
    h = hashlib.sha256(f"{provider}|{batch_id}".encode("utf-8")).hexdigest()[:24]
    return _batch_dir() / f"{provider}-{h}.json"


def save_batch_ctx(provider: str, batch_id: str, ctx: dict) -> None:
    """Decode context of a submitted batch (no pixels, no keys), written atomically."""
    import tempfile
    d = _batch_dir()
    d.mkdir(parents=True, exist_ok=True)
    now = time.time()
    for old in d.glob("*.json"):                 # forget batches whose results the provider no longer keeps
        try:
            if now - old.stat().st_mtime > BATCH_CTX_DAYS * 86400:
                old.unlink()
        except OSError:
            pass
    ctx = {**ctx, "batch_id": batch_id}
    fd, tmp = tempfile.mkstemp(prefix=".batch-", suffix=".json", dir=str(d))
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(ctx, fh, separators=(",", ":"))
        os.replace(tmp, _batch_ctx_path(provider, batch_id))
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


def load_batch_ctx(provider: str, batch_id: str) -> dict:
    p = _batch_ctx_path(provider, batch_id)
    try:
        with open(p, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except (OSError, ValueError):
        raise ProviderError(f"Fovea has no record of batch {str(batch_id)[:80]} in this data folder", "invalid") \
            from None


class VLMProvider(Provider):
    """Shared detect / classify / estimate for VLM providers. Subclasses implement ``call_detect`` / ``call_classify``
    (one HTTP call; raise ``Transient`` or ``ProviderError``), ``frame_for``, ``sent_size`` and ``image_tokens``."""

    kind = "cloud"
    needs_key = True
    max_tokens = 16000
    max_tokens_retry = 32000
    max_crops_per_request = 20
    schema_bounds = True          # provider's JSON-schema dialect supports minimum/maximum/maxItems
    track_cost = True             # warn when a model has no price on file
    default_base_url = ""
    base_url_env = ""

    def __init__(self, config: Optional[dict] = None, key: Optional[str] = None):
        self._config_override = dict(config) if config is not None else None
        self._key_override = key

    # ---- config / key
    def config(self) -> dict:
        return dict(self._config_override) if self._config_override is not None else provider_config(self.id)

    def key(self) -> Optional[str]:
        if self._key_override is not None:
            return self._key_override or None
        return provider_key(self.id, self.env_keys)

    def base_url(self) -> str:
        """Where requests go. Settings refuse a URL with ``user:password@``, but FOVEA_<PROVIDER>_BASE_URL may name such
        a gateway: httpx sends it as basic auth, so the full URL is only ever used to build requests — anything shown
        or stored goes through :func:`url_host` / :func:`redact_url`, and every scrubber removes URL credentials."""
        cfg = self.config()
        url = (cfg.get("base_url") or "").strip() or (os.environ.get(self.base_url_env, "").strip()
                                                         if self.base_url_env else "") or self.default_base_url
        return url.rstrip("/")

    @property
    def location(self) -> str:  # type: ignore[override]
        host = url_host(self.base_url())
        return f"Images are sent to {host}" if host else "Images are sent to the provider"

    def model_id(self, model: Optional[str]) -> str:
        return (model or "").strip() or (self.config().get("default_model") or "").strip() or default_model(self.id)

    def effort_for(self, options: Dict[str, Any]) -> Optional[str]:
        e = (options or {}).get("effort") or self.config().get("effort")
        e = str(e).strip().lower() if e else None
        return {"fast": "low", "accurate": "high"}.get(e or "", e) if e else None

    def limiter(self) -> AdaptiveLimiter:
        try:
            n = int(self.config().get("concurrency") or 4)
        except (TypeError, ValueError):
            n = 4
        return limiter_for(f"{self.id}|{self.base_url()}", min(max(n, 1), 32))

    def require_key(self) -> str:
        k = self.key()
        if not k:
            raise ProviderError(f"No API key for {self.name}. Add one in Settings → AI providers.", "auth")
        return k

    # ---- hooks
    def frame_for(self, model: str, options: Dict[str, Any]) -> str:
        raise NotImplementedError

    def sent_size(self, model: str, w: int, h: int, options: Dict[str, Any]) -> Tuple[int, int]:
        raise NotImplementedError

    def native_edge(self, model: str, options: Dict[str, Any]) -> int:
        return 1536

    def image_tokens(self, model: str, sw: int, sh: int, options: Dict[str, Any]) -> int:
        return 1000

    def detect_schema_for(self, vocab: Vocab, frame: str, max_objects: int, options: Dict[str, Any]) -> dict:
        return detect_schema(vocab.labels, "array", bounds=self.schema_bounds, max_items=max_objects,
                             box_key=box_fields(frame), number=frame == NORM_XYXY,
                             bound_max={YXYX_1000: 1000, XYXY_1000: 1000, NORM_XYXY: 1}.get(frame))

    def call_detect(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                    options: Dict[str, Any], max_tokens: int) -> CallResult:
        raise NotImplementedError

    def call_classify(self, model: str, images: List[Tuple[int, bytes, Tuple[int, int]]], system: str,
                      user_text: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        raise NotImplementedError

    def call_generate(self, model: str, system: str, user_text: str, schema: dict, images: List[Tuple[bytes, str]],
                      options: Dict[str, Any], max_tokens: int) -> CallResult:
        """One structured-output request with optional images (``[(bytes, mime)]``) for ``generate_json``."""
        raise ProviderError(f"{self.name} cannot generate JSON", "invalid")

    # ---- shared plumbing
    def _billed(self, e: BaseException, spent: Optional[Usage] = None) -> BaseException:
        """Attach the money spent on this item to an error leaving the provider: ``spent`` (calls that succeeded
        before it) plus the failing call's own usage (set by ``_parse``) — refusals, unreadable and cut-off answers
        are billed like any other. Attributed to this provider for the usage ledger. Returns ``e``."""
        total = Usage()
        for u in (spent, billed_usage(e)):
            if u is not None:
                total.input_tokens += int(u.input_tokens or 0)
                total.output_tokens += int(u.output_tokens or 0)
                total.cost_usd += float(u.cost_usd or 0)
        if total.input_tokens or total.output_tokens or total.cost_usd:
            total.cost_usd = round(total.cost_usd, 8)
            try:
                e.usage = total.tagged(self.id)          # type: ignore[attr-defined]
            except AttributeError:                       # pragma: no cover — an exception type with __slots__
                pass
        return e

    def _call(self, fn: Callable[[int], CallResult], what: str, stats: dict,
              salvage: Sequence[str] = ("objects", "items"),
              budget: Optional[Tuple[int, int]] = None) -> Tuple[Any, CallResult, List[str]]:
        """One logical request with retries, the truncation retry and refusal handling. Returns (json, result, notes).
        ``budget`` = (max_tokens, max_tokens on the truncation retry); the provider's defaults otherwise. An error
        raised after a billed answer carries that answer's usage (``e.usage``, untagged)."""
        first, retry = budget or (self.max_tokens, self.max_tokens_retry)
        limiter = self.limiter()
        res = with_retries(lambda: fn(first), provider=self.name, limiter=limiter)
        stats["calls"] = stats.get("calls", 0) + 1
        usage = Usage().add(res.usage)
        if res.finish == "truncated":
            stats["truncated"] = stats.get("truncated", 0) + 1
            try:
                res2 = with_retries(lambda: fn(retry), provider=self.name, limiter=limiter)
            except BaseException as e:
                carry_usage(e, usage)                    # the first, cut-off answer was billed
                raise
            stats["calls"] += 1
            usage.add(res2.usage)
            res2.usage = usage
            res = res2
        else:
            res.usage = usage
        obj, notes = self._parse(res, what, stats, salvage)
        return obj, res, notes

    def _parse(self, res: CallResult, what: str, stats: dict,
               salvage: Sequence[str] = ("objects", "items")) -> Tuple[Any, List[str]]:
        """JSON out of one model reply (refusal → ``refusal``; unreadable → ``bad_output``; cut off → the complete
        items when ``salvage`` names the list keys). Shared by the synchronous path and provider batches. The errors
        carry ``res.usage``: the provider billed the reply all the same."""
        notes: List[str] = []
        u = res.usage or Usage()
        spent = Usage(int(u.input_tokens or 0), int(u.output_tokens or 0), float(u.cost_usd or 0))
        if res.finish == "refusal":
            why = f" ({res.refusal})" if res.refusal else ""
            raise ProviderError(f"{self.name} declined to process this image{why}", "refusal", usage=spent)
        if res.finish == "truncated":
            # Still cut off after the larger budget: keep every complete item (a full JSON parse of a truncated
            # reply would only find the first inner object).
            try:
                return json.loads(res.text), notes
            except (TypeError, ValueError):
                pass
            obj = {k: salvage_items(res.text, k) for k in salvage}
            if not any(obj.values()):
                raise ProviderError(f"{self.name} output was cut off before any complete {what}", "bad_output",
                                    usage=spent)
            stats["salvaged"] = stats.get("salvaged", 0) + 1
            notes.append("output was cut off at the token limit; kept the complete part")
            return obj, notes
        try:
            return extract_json(res.text), notes
        except BadOutput as e:
            raise ProviderError(f"{self.name} returned unreadable output ({e})", "bad_output", usage=spent) from None

    # ---- detection
    def _plan(self, req: DetectRequest, *, keep_image: bool = True, encode: bool = True) -> "DetectPlan":
        """Everything one image's detection needs before any request: model, frame, vocabulary, the prepared full
        image + tiles (+ ruler). Raises ProviderError for an unusable request or image."""
        if not req.classes:
            raise ProviderError("Choose at least one class to look for", "invalid")
        options = dict(req.options or {})
        model = self.model_id(req.model)
        frame = self.frame_for(model, options)
        native = self.native_edge(model, options)
        try:
            max_objects = min(max(int(options.get("max_objects") or 100), 1), 500)
        except (TypeError, ValueError):
            max_objects = 100
        try:
            samples = min(max(int(options.get("samples") or 1), 1), 5)
        except (TypeError, ValueError):
            samples = 1
        warnings: List[str] = []
        ruler = None
        if options.get("ruler"):
            if frame in (YXYX_1000, XYXY_1000):
                ruler = frame
            else:
                warnings.append("the ruler overlay is only used with 0-1000 coordinate frames; sent without it")
        vocab = Vocab.build(req.classes)
        tiles_opt = options.get("tiles")
        refine = refine_settings(options.get("refine"))
        # refine crops small boxes out of the original: decode it at full resolution
        need = None if refine else native * (3 if tiles_opt else 1)
        im = open_image(req.image_path, need_edge=need)
        W, H = im.size
        regions: List[Optional[Tuple[float, float, float, float]]] = [None] + list(tile_plan(W, H, tiles_opt, native))
        size_fn = functools.partial(self._size_with_cap, model, options)
        preps = [prepare(im, r, size_fn, ruler=ruler, encode=encode) for r in regions]
        try:
            conf = float(req.conf or 0)
        except (TypeError, ValueError):
            conf = 0.0
        return DetectPlan(model=model, frame=frame, vocab=vocab, options=options, max_objects=max_objects,
                          samples=samples, conf=conf, preps=preps, warnings=warnings, refine=refine,
                          image=im if keep_image else None)

    def _prompts(self, plan: "DetectPlan", prep: Prepared) -> Tuple[str, str, dict]:
        system = detect_system_prompt(plan.vocab, plan.frame, plan.max_objects)
        schema = self.detect_schema_for(plan.vocab, plan.frame, plan.max_objects, plan.options)
        user = detect_user_text(plan.vocab, plan.frame, prep.width, prep.height, schema)
        if any(prep.pad):
            user += ruler_sentence(prep)
        return system, user, schema

    def detect(self, req: DetectRequest) -> DetectResult:
        t0 = time.monotonic()
        plan = self._plan(req)
        acc = DetectAccumulator(plan)
        try:
            for _s in range(plan.samples):
                run: List[Box] = []
                for ti, prep in enumerate(plan.preps):
                    system, user, schema = self._prompts(plan, prep)
                    obj, res, notes = self._call(
                        lambda mt, prep=prep, system=system, user=user, schema=schema:
                            self.call_detect(plan.model, prep, system, user, schema, plan.options, mt), "box",
                        acc.stats)
                    acc.add_usage(res, notes)
                    run.extend(self._decode(items_from(obj, "objects"), plan.frame, prep, plan.vocab, ti, acc.stats,
                                            acc.unknown, acc.anomalies))
                acc.add_run(run)
            boxes = acc.merged()
            if plan.refine and boxes:
                boxes = self._refine(plan, boxes, acc)
        except Exception as e:
            self._billed(e, acc.usage)           # earlier tiles / samples / refine crops were billed too
            raise
        return self._finish(plan, boxes, acc, int((time.monotonic() - t0) * 1000))

    def _finish(self, plan: "DetectPlan", boxes: List["Box"], acc: "DetectAccumulator", latency_ms: int) -> DetectResult:
        stats, warnings = acc.stats, acc.warnings
        if len(boxes) > plan.max_objects:
            stats["capped"] = len(boxes) - plan.max_objects
            boxes = boxes[:plan.max_objects]
        dets = [d for d in (to_detection(b) for b in boxes) if d is not None]
        if acc.unknown:
            n = sum(acc.unknown.values())
            names = ", ".join(sorted(acc.unknown)[:5])
            warnings.append(f"{n} box{'es' if n != 1 else ''} with labels outside the class list dropped ({names})")
        if stats["degenerate"]:
            warnings.append(f"{stats['degenerate']} empty or inverted box{'es' if stats['degenerate'] != 1 else ''} "
                            "dropped")
        if acc.anomalies:
            warnings.append(f"{self.name} answered on {acc.anomalies[0]}; coordinates were rescaled")
        if stats["capped"]:
            warnings.append(f"kept the first {plan.max_objects} objects ({stats['capped']} more dropped)")
        if stats.get("refine_stopped"):
            warnings.append(f"refine pass stopped early ({stats['refine_stopped']})")
        served = acc.served
        if self.track_cost and not model_price(self.id, served[0] if served else plan.model):
            warnings.append(f"no price on file for {served[0] if served else plan.model}; cost not tracked")
        result = DetectResult(detections=dets, usage=acc.usage, latency_ms=latency_ms, warnings=warnings,
                              served_by=served[-1] if served else plan.model)
        result.stats = stats  # type: ignore[attr-defined]  — counters for batch job stats (not part of to_dict)
        return result

    def _size_with_cap(self, model: str, options: Dict[str, Any], w: int, h: int) -> Tuple[int, int]:
        cap = options.get("max_edge")
        if cap:
            try:
                w, h = fit_long_edge(w, h, max(64, int(cap)))
            except (TypeError, ValueError):
                pass
        return self.sent_size(model, w, h, options)

    def _decode(self, items: List[dict], frame: str, prep: Prepared, vocab: Vocab, tile: int, stats: dict,
                unknown: Dict[str, int], anomalies: List[str]) -> List[Box]:
        parsed = []
        for it in items:
            vals = item_box(it, frame)
            if vals is None or any(not math.isfinite(v) for v in vals):
                stats["degenerate"] = stats.get("degenerate", 0) + 1
                continue
            parsed.append((it, vals))
        scale, anomaly = detect_scale([v for _, vals in parsed for v in vals], frame, prep.width, prep.height)
        if anomaly:
            stats["scale_anomalies"] = stats.get("scale_anomalies", 0) + 1
            anomalies.append(anomaly)
            log.warning("%s: response on %s (frame %s, sent %dx%d)", self.name, anomaly, frame, prep.width,
                        prep.height)
        out: List[Box] = []
        ox, oy, tw, th = prep.region
        W, H = prep.full
        inner = (ox > 0.5, oy > 0.5, ox + tw < W - 0.5, oy + th < H - 0.5)
        pl, pt, _pr, _pb = prep.pad
        pw, ph = prep.photo_size
        for it, vals in parsed:
            lab = item_label(it)
            spec = vocab.resolve(lab)
            if spec is None:
                key = str(lab)[:40] if lab is not None else "(no label)"
                unknown[key] = unknown.get(key, 0) + 1
                stats["unknown_labels"] = stats.get("unknown_labels", 0) + 1
                continue
            x1, y1, x2, y2 = to_unit(vals, scale, prep.width, prep.height)
            if pl or pt or pw != prep.width or ph != prep.height:
                # ruler margin: the model answered over the whole sent image; keep only the photo part
                x1, x2 = (x1 * prep.width - pl) / pw, (x2 * prep.width - pl) / pw
                y1, y2 = (y1 * prep.height - pt) / ph, (y2 * prep.height - pt) / ph
            x1, x2 = sorted((min(max(x1, 0.0), 1.0), min(max(x2, 0.0), 1.0)))
            y1, y2 = sorted((min(max(y1, 0.0), 1.0), min(max(y2, 0.0), 1.0)))
            # under ~1 sent pixel in either direction: nothing to label
            if (x2 - x1) * pw < 1.0 or (y2 - y1) * ph < 1.0:
                stats["degenerate"] = stats.get("degenerate", 0) + 1
                continue
            edge = tile > 0 and ((inner[0] and x1 <= 0.01) or (inner[1] and y1 <= 0.01) or
                                 (inner[2] and x2 >= 0.99) or (inner[3] and y2 >= 0.99))
            fx1, fy1, fx2, fy2 = region_to_full((x1, y1, x2, y2), prep)
            out.append(Box(spec.id, fx1, fy1, fx2, fy2, str(lab), None, edge, tile))
        return out

    # ---- crop-and-refine second pass (research §7 item 7)
    def _refine(self, plan: "DetectPlan", boxes: List["Box"], acc: "DetectAccumulator") -> List["Box"]:
        """Re-ask small or low-agreement boxes on a zoomed context crop and accept the answer only when it passes the
        geometric guards (IoU with the first box >= 0.25, area ratio within [0.5, 2], centre shift <= 0.35 of the
        context window). Failures keep the first box; usage is accounted either way."""
        cfg = plan.refine
        im = plan.image
        if im is None or not cfg:
            return boxes
        W, H = im.size
        stats = acc.stats
        for k in ("refine_candidates", "refine_calls", "refined", "refine_rejected", "refine_failed"):
            stats.setdefault(k, 0)
        by_id = {c.id: c for c in plan.vocab.classes}
        order = []
        for i, b in enumerate(boxes):
            small = b.area < cfg["small"]
            disagree = plan.samples > 1 and b.conf is not None and b.conf < 1.0
            if cfg["all"] or small or disagree:
                order.append((0 if disagree else 1, b.area, i))
        order.sort()
        picked = [i for _p, _a, i in order[:cfg["max_boxes"]]]
        stats["refine_candidates"] += len(picked)
        out = list(boxes)
        zoom_fn = functools.partial(self._zoom_size, plan.model, plan.options, cfg["zoom_edge"])
        for i in picked:
            cur = out[i]
            spec = by_id.get(cur.cls)
            if spec is None:
                continue
            vocab1 = Vocab.build([spec])
            label = vocab1.labels[0]
            changed = False
            for _it in range(cfg["iterations"]):
                region = refine_region(cur, W, H, cfg["context"], cfg["min_px"])
                prep = prepare(im, region, zoom_fn)
                system = refine_system_prompt(plan.frame)
                schema = self.detect_schema_for(vocab1, plan.frame, 5, plan.options)
                user = refine_user_text(label, plan.frame, prep.width, prep.height, schema)
                scratch: Dict[str, Any] = {}
                try:
                    obj, res, _notes = self._call(
                        lambda mt, prep=prep, system=system, user=user, schema=schema:
                            self.call_detect(plan.model, prep, system, user, schema, plan.options, mt), "box", scratch)
                except ProviderError as e:
                    stats["refine_failed"] += 1
                    spent = billed_usage(e)             # a refused / unreadable refine answer is billed too
                    if spent is not None:
                        acc.usage.add(Usage(int(spent.input_tokens), int(spent.output_tokens), float(spent.cost_usd)))
                    if e.fatal or e.code in ("rate", "overloaded", "network"):
                        stats["refine_stopped"] = scrub(str(e))[:160]
                        return [o for o in out]
                    break
                stats["refine_calls"] += scratch.get("calls", 1)
                acc.usage.add(res.usage)
                cands = self._decode(items_from(obj, "objects"), plan.frame, prep, vocab1, 0, scratch, {}, [])
                best = max(cands, key=lambda c: iou_xyxy(c, cur), default=None)
                if best is None or not refine_accept(cur, best, region, W, H):
                    stats["refine_rejected"] += 1
                    break
                moved = iou_xyxy(best, cur)
                cur = Box(cur.cls, best.x1, best.y1, best.x2, best.y2, cur.label, cur.conf, cur.edge, cur.tile)
                changed = True
                if moved >= 0.9:          # converged: a second iteration would not move it
                    break
            if changed:
                stats["refined"] += 1
                out[i] = cur
        return out

    def _zoom_size(self, model: str, options: Dict[str, Any], zoom_edge: int, w: int, h: int) -> Tuple[int, int]:
        """A context crop is upscaled to ``zoom_edge`` on its long edge (a 60 px object is only a couple of vision
        patches otherwise), then fitted to the provider's own limits."""
        s = max(w, h)
        if s < zoom_edge:
            w, h = max(1, round(w * zoom_edge / s)), max(1, round(h * zoom_edge / s))
        return self.sent_size(model, w, h, options)

    # ---- generic structured output
    def generate_json(self, system: str, user: str, schema: dict, model: str = "",
                      images: Optional[List[bytes]] = None, options: Optional[Dict[str, Any]] = None
                      ) -> Tuple[Any, Usage]:
        """One structured-output request (text + optional images) → (parsed JSON, usage). Used for naming classes and
        natural-language search. The schema is adapted to the provider's dialect (unsupported keywords dropped, maps
        encoded as key/value arrays, a non-object root wrapped) and the reply is decoded back to the caller's shape."""
        options = dict(options or {})
        model = self.model_id(model)
        if not isinstance(system, str) or not isinstance(user, str) or not user.strip():
            raise ProviderError("generate_json needs a system and a user prompt", "invalid")
        wrapped, unwrap = wrap_schema(schema)
        wire, map_paths = encode_maps(wrapped)
        try:
            edge = int(options.get("max_edge") or 0) or min(self.native_edge(model, options), 2048)
        except (TypeError, ValueError):
            edge = 1568
        imgs = [prepare_blob(b, edge) for b in (images or [])]
        if len(imgs) > 20:
            raise ProviderError("generate_json accepts at most 20 images per request", "invalid")
        try:
            mt = int(options.get("max_tokens") or 0)
        except (TypeError, ValueError):
            mt = 0
        stats: Dict[str, Any] = {}
        budget = None
        if mt:
            # ``max_tokens`` sizes the answer; reasoning models spend output tokens on thinking first, so add that
            # allowance (billing is by tokens actually used, never by the cap)
            effort = self.effort_for(options) or self.default_effort(model)
            mt = min(max(mt, 256) + (THINKING_TOKENS.get(effort or "", (0, 0))[1] if effort else 0), 64000)
            budget = (mt, max(min(mt * 2, 64000), mt))
        try:
            obj, res, _notes = self._call(
                lambda m: self.call_generate(model, system, user, wire, imgs, options, m), "answer", stats,
                salvage=(), budget=budget)
        except Exception as e:
            self._billed(e)
            raise
        obj = decode_maps(obj, map_paths)
        if unwrap and isinstance(obj, dict) and "result" in obj:
            obj = obj["result"]
        problem = check_json_shape(obj, schema)
        if problem:
            raise self._billed(ProviderError(f"{self.name} answered in the wrong shape ({problem})", "bad_output"),
                               res.usage)
        if self.track_cost and not model_price(self.id, res.served_by or model):
            log.info("%s: no price on file for %s; cost not tracked", self.name, res.served_by or model)
        return obj, res.usage

    # ---- provider batch APIs (Economy mode: ~50% off, results within 24 h)
    batch_max_requests = 10000
    batch_max_bytes = 180 * 1024 * 1024

    def batch_request(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                      options: Dict[str, Any], max_tokens: int) -> dict:
        """The provider's request body for one detection inside a batch."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def _batch_create(self, model: str, reqs: List[Tuple[str, dict]]) -> str:
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def _batch_get(self, part: str) -> dict:
        """{state: running|canceling|ended|failed, counts: {...}, message?}"""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def _batch_fetch(self, part: str, ctx: dict) -> Iterator[Tuple[str, Any]]:
        """(sub_id, CallResult | ProviderError) for every request of one provider batch."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def _batch_cancel(self, part: str) -> None:
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def _batch_plans(self, reqs: List[Tuple[Any, DetectRequest]], ctx: dict, *, bodies: bool
                     ) -> Iterator[Tuple[str, str, List[Tuple[str, Optional[dict]]], int]]:
        """Plan every image of a batch in order, recording its decode context in ``ctx``; yields
        ``(key, model, [(sub_id, request body | None)], approx bytes)`` per usable image. Deterministic for the same
        items, so the context can be rebuilt from the request list when the saved one is missing."""
        vocab_idx: Dict[str, int] = {json.dumps(v): i for i, v in enumerate(ctx["vocabs"])}
        n_sub = 0
        for cid, req in reqs:
            key = str(cid)
            try:
                plan = self._plan(req, keep_image=False, encode=bodies)
            except ProviderError as e:
                ctx["errors"][key] = {"cid": cid, "code": e.code, "message": scrub(str(e))[:300]}
                continue
            if plan.refine:
                plan.warnings.append("crop-and-refine is not available in Economy (batch) mode; first-pass boxes kept")
            vk = json.dumps([[c.id, c.name, c.prompt, c.description] for c in plan.vocab.classes])
            if vk not in vocab_idx:
                vocab_idx[vk] = len(ctx["vocabs"])
                ctx["vocabs"].append(json.loads(vk))
            subs: List[Tuple[str, Optional[dict]]] = []
            size = 0
            multi = plan.samples * len(plan.preps) > 1
            for s in range(plan.samples):
                for ti, prep in enumerate(plan.preps):
                    n_sub += 1
                    sid = _sub_id(key, s, ti, multi, n_sub)
                    if sid in ctx["subs"]:
                        sid = f"r{n_sub}"
                    body = None
                    if bodies:
                        system, user, schema = self._prompts(plan, prep)
                        body = self.batch_request(plan.model, prep, system, user, schema, plan.options,
                                                  self.max_tokens_retry)
                        size += len(prep.data) * 4 // 3 + len(system) + len(user) + 4096
                    subs.append((sid, body))
                    ctx["subs"][sid] = [key, s, ti]
            ctx["images"][key] = {"cid": cid, **plan.context(vocab_idx[vk])}
            yield key, plan.model, subs, size

    @staticmethod
    def _new_ctx(provider: str, base_url: str) -> dict:
        return {"v": 1, "provider": provider, "base_url": base_url, "created": time.time(), "parts": [],
                "part_images": {}, "images": {}, "subs": {}, "errors": {}, "vocabs": []}

    @staticmethod
    def _check_items(items: Optional[List[Any]]) -> List[Tuple[Any, DetectRequest]]:
        reqs = [_batch_item(it) for it in items or []]
        seen = set()
        for cid, _r in reqs:
            if str(cid) in seen:
                raise ProviderError(f"Duplicate custom_id {cid!r} in the batch", "invalid")
            seen.add(str(cid))
        return reqs

    def batch_submit(self, items: List[Any]) -> str:
        """Submit detections as provider batch(es). ``items``: ``[{"custom_id", "request": DetectRequest}]`` or
        ``[{"custom_id", "image_path", "classes", "model", "conf"?, "options"?}]`` (custom_id = the image id).
        Returns the batch id (``id1,id2`` when the input had to be split over several provider batches). The decode
        context (sent sizes, tiles, ruler margins, vocabulary) is kept in ``DATA_DIR/ai_batches`` so results can be
        read after a restart. An image that cannot be prepared is reported by ``batch_results`` as an error."""
        if not (self.capabilities or {}).get("batch_api"):
            raise ProviderError(f"{self.name} has no batch API in Fovea", "invalid")
        self.require_key()
        reqs = self._check_items(items)
        if not reqs:
            raise ProviderError("Nothing to submit", "invalid")
        ctx = self._new_ctx(self.id, redact_url(self.base_url()))
        chunk: List[Tuple[str, dict]] = []
        chunk_ids: List[str] = []
        chunk_bytes = 0
        chunk_model = ""

        def create() -> str:
            try:
                return self._batch_create(chunk_model, chunk)
            except Transient as t:
                # an explicit 429 / 5xx means "not accepted": safe to retry. A timeout or dropped connection after
                # the upload is ambiguous — retrying could create (and bill) the same batch twice.
                if t.code == "network" or t.status is None:
                    raise ProviderError(f"{t} — not retried, because {self.name} may have created the batch anyway; "
                                        "check its console before submitting again", "network") from None
                raise

        def flush() -> None:
            nonlocal chunk, chunk_ids, chunk_bytes
            if not chunk:
                return
            try:
                part = with_retries(create, provider=self.name)
            except Exception:
                for p in ctx["parts"]:          # do not leave half a submission running (and billing)
                    try:
                        self._batch_cancel(p)
                    except Exception:  # noqa: BLE001
                        log.warning("%s: could not cancel batch part %s", self.name, p)
                raise
            ctx["parts"].append(part)
            ctx["part_images"][part] = list(chunk_ids)
            chunk, chunk_ids, chunk_bytes = [], [], 0

        for key, model, subs, size in self._batch_plans(reqs, ctx, bodies=True):
            if chunk and (model != chunk_model or chunk_bytes + size > self.batch_max_bytes
                          or len(chunk) + len(subs) > self.batch_max_requests):
                flush()
            chunk_model = model
            chunk.extend((sid, body) for sid, body in subs if body is not None)
            chunk_ids.append(key)
            chunk_bytes += size
        flush()
        if not ctx["parts"]:
            first = next(iter(ctx["errors"].values()), {})
            raise ProviderError(first.get("message") or "No image could be prepared for the batch",
                                first.get("code") or "invalid")
        batch_id = ",".join(ctx["parts"])
        save_batch_ctx(self.id, batch_id, ctx)
        return batch_id

    def batch_status(self, batch_id: str) -> dict:
        """``{state, done, counts: {total, processing, succeeded, errored, canceled, expired}, images, parts}``.
        state: ``running`` · ``canceling`` · ``ended`` (results can be read; some may have errored / expired) ·
        ``failed`` (the provider rejected the whole batch; ``message`` says why)."""
        parts = [p for p in str(batch_id or "").split(",") if p]
        if not parts:
            raise ProviderError("Unknown batch id", "invalid")
        counts = {"total": 0, "processing": 0, "succeeded": 0, "errored": 0, "canceled": 0, "expired": 0}
        states, messages = [], []
        for p in parts:
            st = with_retries(lambda p=p: self._batch_get(p), provider=self.name)
            states.append(st.get("state", "running"))
            for k in counts:
                counts[k] += int((st.get("counts") or {}).get(k) or 0)
            if st.get("message"):
                messages.append(str(st["message"]))
        if "failed" in states and all(s in ("failed", "ended") for s in states):
            state = "failed" if all(s == "failed" for s in states) else "ended"
        elif "canceling" in states:
            state = "canceling"
        elif all(s == "ended" for s in states):
            state = "ended"
        else:
            state = "running"
        try:
            ctx = load_batch_ctx(self.id, batch_id)
            images = len(ctx.get("images", {})) + len(ctx.get("errors", {}))
        except ProviderError:
            images = None
        out = {"state": state, "done": state in ("ended", "failed"), "counts": counts, "images": images,
               "parts": len(parts)}
        if messages:
            out["message"] = scrub("; ".join(messages))[:300]
        return out

    def batch_results(self, batch_id: str, items: Optional[List[Any]] = None) -> Iterator[Tuple[Any, Any]]:
        """``(custom_id, DetectResult | ProviderError)`` per submitted image, decoded exactly like the synchronous path
        (same frames, tiles, ruler margins, NMS / votes, class mapping) and costed at batch prices. Per-image errors
        use the usual codes plus ``expired`` / ``canceled`` (the provider never processed that request).
        The decode context saved at submit time is used; without it (another data folder, deleted file) ``items`` —
        the list given to ``batch_submit``, in the same order — rebuilds it from the images. An unknown batch raises
        ``ProviderError('invalid')`` here, before iteration starts."""
        try:
            ctx = load_batch_ctx(self.id, batch_id)
        except ProviderError:
            if not items:
                raise
            ctx = self._new_ctx(self.id, redact_url(self.base_url()))
            for _ in self._batch_plans(self._check_items(items), ctx, bodies=False):
                pass
            ctx["parts"] = [p for p in str(batch_id or "").split(",") if p]
        return self._iter_results(ctx)

    def _iter_results(self, ctx: dict) -> Iterator[Tuple[Any, Any]]:
        parts = list(ctx["parts"])
        known = all(p in ctx.get("part_images", {}) for p in parts)
        groups = [(p, [p]) for p in parts] if known else [(None, parts)]
        for part, fetch in groups:
            got: Dict[str, Dict[Tuple[int, int], Any]] = {}
            for fp in fetch:
                for sid, res in self._batch_fetch(fp, ctx):
                    m = ctx["subs"].get(sid)
                    if not m:
                        continue
                    got.setdefault(m[0], {})[(int(m[1]), int(m[2]))] = res
            keys = ctx["part_images"][part] if part is not None else list(ctx["images"])
            for key in keys:
                info = ctx["images"].get(key)
                if info is None:
                    continue
                yield info["cid"], self._batch_decode(ctx, info, got.get(key, {}))
        for e in ctx.get("errors", {}).values():
            yield e["cid"], ProviderError(e.get("message") or "image could not be prepared", e.get("code") or "invalid")

    def batch_cancel(self, batch_id: str) -> dict:
        """Ask the provider to stop the batch (requests already processed are still billed and readable)."""
        for p in [p for p in str(batch_id or "").split(",") if p]:
            with_retries(lambda p=p: self._batch_cancel(p), provider=self.name)
        return self.batch_status(batch_id)

    def _batch_decode(self, ctx: dict, info: dict, results: Dict[Tuple[int, int], Any]) -> Any:
        plan = DetectPlan.from_context(info, ctx["vocabs"][info["vocab"]])
        acc = DetectAccumulator(plan)

        def failed(err: ProviderError) -> ProviderError:
            # every answer the provider returned for this image was billed (at the batch price), the failing one too
            spent = Usage()
            for r in results.values():
                if isinstance(r, CallResult) and r.usage is not None:
                    spent.add(Usage(int(r.usage.input_tokens or 0), int(r.usage.output_tokens or 0),
                                    float(r.usage.cost_usd or 0)))
            err.usage = None
            self._billed(err, spent)
            return err

        for s in range(plan.samples):
            run: List[Box] = []
            for ti, prep in enumerate(plan.preps):
                res = results.get((s, ti))
                if res is None:
                    return failed(ProviderError(f"{self.name} returned no result for this image", "expired"))
                if isinstance(res, Exception):
                    return failed(res if isinstance(res, ProviderError) else
                                  ProviderError(scrub(str(res)), "bad_output"))
                try:
                    obj, notes = self._parse(res, "box", acc.stats)
                except ProviderError as e:
                    return failed(e)
                acc.stats["calls"] += 1
                acc.add_usage(res, notes)
                run.extend(self._decode(items_from(obj, "objects"), plan.frame, prep, plan.vocab, ti, acc.stats,
                                        acc.unknown, acc.anomalies))
            acc.add_run(run)
        return self._finish(plan, acc.merged(), acc, 0)

    def classify(self, req: ClassifyRequest) -> ClassifyResult:
        t0 = time.monotonic()
        if not req.classes:
            raise ProviderError("Choose at least one candidate class", "invalid")
        options = dict(req.options or {})
        model = self.model_id(req.model)
        vocab = Vocab.build(req.classes)
        labels = classify_labels(vocab)
        if not req.crops:
            return ClassifyResult(items=[], latency_ms=0)
        try:
            pad = float(options.get("pad", 0.25))
        except (TypeError, ValueError):
            pad = 0.25
        try:
            edge = int(options.get("crop_edge") or 448)
        except (TypeError, ValueError):
            edge = 448
        try:
            min_edge = int(options.get("crop_min_edge") or 336)
        except (TypeError, ValueError):
            min_edge = 336
        im = open_image(req.image_path)
        mark = options.get("mark", True) is not False
        crops = []
        for k, c in enumerate(req.crops):
            data, size = crop_for_classification(im, c.xc, c.yc, c.w, c.h, pad=min(max(pad, 0.0), 2.0),
                                                 edge=min(max(edge, 64), 1024), mark=mark, min_edge=min_edge)
            crops.append((k + 1, data, size, c))
        system = classify_system_prompt(vocab)
        usage, warnings, items = Usage(), [], []
        stats = {"calls": 0, "truncated": 0, "salvaged": 0}
        chunk = self.max_crops_per_request
        for start in range(0, len(crops), chunk):
            part = crops[start:start + chunk]
            schema = classify_schema(labels, bounds=self.schema_bounds, n=len(part))
            user = (f"{len(part)} numbered crops follow (1-{len(part)}). For each crop, classify the outlined object "
                    f"as one of: " + ", ".join(json.dumps(l, ensure_ascii=False) for l in labels) + ".")
            images = [(i - start, d, s) for i, d, s, _c in part]
            try:
                obj, res, notes = self._call(
                    lambda mt, images=images, schema=schema, user=user:
                        self.call_classify(model, images, system, user, schema, options, mt), "answer", stats)
            except Exception as e:
                self._billed(e, usage)           # earlier chunks of crops were billed too
                raise
            usage.add(res.usage)
            warnings.extend(n for n in notes if n not in warnings)
            answers = {}
            for it in items_from(obj, "items"):
                try:
                    answers[int(it.get("index"))] = it
                except (TypeError, ValueError):
                    continue
            for i, _d, _s, c in part:
                a = answers.get(i - start)
                if a is None:
                    items.append(Classification(index=c.index, cls=None, verdict="uncertain", confidence=None,
                                                reason="no answer for this crop"))
                    continue
                lab = a.get("label")
                conf = CONFIDENCE_WORDS.get(str(a.get("confidence", "")).strip().lower())
                reason = scrub(str(a.get("reason") or ""))[:200]
                norm = str(lab or "").strip().lower()
                spec = vocab.resolve(lab) if norm not in (NONE_LABEL, UNCERTAIN_LABEL) or lab in vocab.by_label else None
                if spec is not None:
                    items.append(Classification(index=c.index, cls=spec.id, verdict="label", confidence=conf,
                                                reason=reason))
                elif norm == NONE_LABEL:
                    items.append(Classification(index=c.index, cls=None, verdict="none", confidence=conf,
                                                reason=reason))
                else:
                    items.append(Classification(index=c.index, cls=None, verdict="uncertain", confidence=conf,
                                                reason=reason or f"unknown label {str(lab)[:40]!r}"))
        missing = sum(1 for it in items if it.reason == "no answer for this crop")
        if missing:
            warnings.append(f"{missing} crop{'s' if missing != 1 else ''} got no answer")
        return ClassifyResult(items=items, usage=usage, latency_ms=int((time.monotonic() - t0) * 1000),
                              warnings=warnings)

    def detect_prompt_tokens(self, frame: str, n_classes: int, max_objects: int, options: Dict[str, Any],
                             ruler: bool = False) -> int:
        """Text input tokens of one detect request, measured on the prompts actually sent (the API bills the system
        prompt and the user text, ~4 characters a token, plus turn framing). The estimate only knows how many classes
        there are, so their labels are placeholders of a typical length."""
        vocab = Vocab.build([ClassSpec(i, f"class name {i:02d}") for i in range(max(1, int(n_classes or 1)))])
        schema = self.detect_schema_for(vocab, frame, max_objects, options)
        user = detect_user_text(vocab, frame, 1000, 1000, schema)
        if ruler:
            user += ruler_sentence(Prepared(b"", "image/jpeg", 1060, 1060, (0.0, 0.0, 1000.0, 1000.0), (1000, 1000),
                                            pad=(60, 60, 0, 0)))
        return len(detect_system_prompt(vocab, frame, max_objects)) // 4 + len(user) // 4 + PROMPT_FRAMING_TOKENS

    def refine_prompt_tokens(self, frame: str, options: Dict[str, Any]) -> int:
        """Text input tokens of one crop-and-refine request (one class, a zoomed crop)."""
        vocab = Vocab.build([ClassSpec(0, "class name 00")])
        user = refine_user_text(vocab.labels[0], frame, 1000, 1000, self.detect_schema_for(vocab, frame, 5, options))
        return len(refine_system_prompt(frame)) // 4 + len(user) // 4 + PROMPT_FRAMING_TOKENS

    def estimate(self, width: int, height: int, model: str, n_classes: int, options: Dict[str, Any]) -> dict:
        options = dict(options or {})
        model = self.model_id(model)
        W, H = max(1, int(width or 0) or 1024), max(1, int(height or 0) or 768)
        native = self.native_edge(model, options)
        regions = [(0.0, 0.0, float(W), float(H))] + tile_plan(W, H, options.get("tiles"), native)
        try:
            samples = min(max(int(options.get("samples") or 1), 1), 5)
        except (TypeError, ValueError):
            samples = 1
        frame = self.frame_for(model, options)
        ruler = bool(options.get("ruler")) and frame in (YXYX_1000, XYXY_1000)
        img_tokens = 0
        for (_ox, _oy, tw, th) in regions:
            sw, sh = self._size_with_cap(model, options, max(1, int(round(tw))), max(1, int(round(th))))
            if ruler:                                  # the margin adds a few % of pixels
                m = max(24, int(round(0.05 * max(sw, sh))))
                sw, sh = sw + m + 12, sh + m
            img_tokens += self.image_tokens(model, sw, sh, options)
        calls = len(regions) * samples
        try:
            max_objects = min(max(int(options.get("max_objects") or 100), 1), 500)
        except (TypeError, ValueError):
            max_objects = 100
        prompt_tokens = self.detect_prompt_tokens(frame, n_classes, max_objects, options, ruler)
        tin = (img_tokens + prompt_tokens * len(regions)) * samples
        effort = self.effort_for(options) or self.default_effort(model)
        th_lo, th_hi = THINKING_TOKENS.get(effort or "medium", THINKING_TOKENS["medium"])
        out_lo, out_hi = (40 + th_lo) * calls, (900 + th_hi) * calls
        # crop-and-refine: none when no box is small, at most max_boxes x iterations zoomed crops
        try:
            rcfg = refine_settings(options.get("refine"))
        except ProviderError:
            rcfg = None
        r_calls, r_in, r_out = 0, 0, 0
        if rcfg:
            r_calls = rcfg["max_boxes"] * rcfg["iterations"]
            zw, zh = self._zoom_size(model, options, rcfg["zoom_edge"], rcfg["min_px"], rcfg["min_px"])
            r_in = r_calls * (self.image_tokens(model, zw, zh, options) + self.refine_prompt_tokens(frame, options))
            r_out = r_calls * (60 + th_hi // 4)        # one box on a crop: far less reasoning than a full image
        p = model_price(self.id, model)
        tier = self.billing_tier(model, options)
        usd = [0.0, 0.0]
        usd_batch = None
        if p:
            pin, pout = p["in"], p["out"]
            if tier == "flex":                        # synchronous requests billed at the batch price
                pin, pout = p.get("batch_in", pin), p.get("batch_out", pout)
            usd = [round((tin * pin + out_lo * pout) / 1e6, 6),
                   round(((tin + r_in) * pin + (out_hi + r_out) * pout) / 1e6, 6)]
            if "batch_in" in p:                       # refine does not run in batch mode
                usd_batch = [round((tin * p["batch_in"] + out_lo * p["batch_out"]) / 1e6, 6),
                             round((tin * p["batch_in"] + out_hi * p["batch_out"]) / 1e6, 6)]
        entry = find_model(self.id, model) or {}
        secs = entry.get("seconds")
        if secs:
            secs = float(secs) * {"none": 0.6, "minimal": 0.6, "low": 0.8, "medium": 1.0, "high": 1.5, "xhigh": 2.2,
                                  "max": 3.0}.get(effort or "medium", 1.0) * calls
            if tier == "flex":
                secs *= FLEX_SLOWDOWN
        out = {"usd": usd, "usd_batch": usd_batch, "tokens_in": [tin, tin + r_in],
               "tokens_out": [out_lo, out_hi + r_out], "seconds": round(secs, 1) if secs else None, "calls": calls,
               "priced": bool(p)}
        if tier:
            out["tier"] = tier
        if rcfg:
            out["refine_calls"] = [0, r_calls]
        return out

    def billing_tier(self, model: str, options: Dict[str, Any]) -> Optional[str]:
        """``"flex"`` when this provider's synchronous requests are billed at batch prices (OpenAI's Flex tier,
        slower answers); None for the standard price."""
        return None

    def default_effort(self, model: str) -> Optional[str]:
        return "medium"

    # ---- models
    _models_cache: Dict[str, Tuple[float, List[dict]]] = {}
    MODELS_TTL = 600.0

    def models(self, refresh: bool = False) -> List[dict]:
        """Curated list merged with the live listing. ``refresh`` fetches the live list now (needs the key) and
        raises ``ProviderError`` when that fails, so the caller can show why and fall back to the curated list."""
        cache_key = f"{self.id}|{self.base_url()}"
        live: Optional[List[dict]] = None
        if refresh and (self.key() or not self.needs_key):
            try:
                live = self.list_remote_models()
            except Transient as e:
                raise ProviderError(f"Could not list models: {e}", e.code, status=e.status) from None
            VLMProvider._models_cache[cache_key] = (time.time(), live)
        else:
            cached = VLMProvider._models_cache.get(cache_key)
            if cached and time.time() - cached[0] < self.MODELS_TTL:
                live = cached[1]
        return self.merge_models(live)

    def list_remote_models(self) -> List[dict]:
        """[{id, name?}] from the provider; raise ProviderError on failure."""
        return []

    def test(self) -> dict:
        """Auth + connectivity check without spending tokens: list the models the key can use."""
        t0 = time.monotonic()
        if not httpx_available():
            return {"ok": False, "message": HTTPX_MISSING, "latency_ms": 0}
        if self.needs_key and not self.key():
            return {"ok": False, "message": "No API key — add one first", "latency_ms": 0}
        try:
            models = self.list_remote_models()
        except (ProviderError, Transient) as e:
            return {"ok": False, "message": scrub(str(e), self.key()), "latency_ms": int((time.monotonic() - t0) * 1000)}
        VLMProvider._models_cache[f"{self.id}|{self.base_url()}"] = (time.time(), models)
        priced = sum(1 for m in models if find_model(self.id, m["id"]))
        where = url_host(self.base_url())
        msg = f"Connected to {where} — {len(models)} model{'s' if len(models) != 1 else ''} available"
        if models and priced < len(models):
            msg += f" ({priced} with prices on file)"
        return {"ok": True, "message": msg, "latency_ms": int((time.monotonic() - t0) * 1000)}

    def merge_models(self, live: Optional[List[dict]]) -> List[dict]:
        cfg_default = (self.config().get("default_model") or "").strip()
        static = static_models(self.id)
        out: List[dict] = []
        live_ids = {m["id"] for m in live} if live is not None else None
        for m in static:
            e = self._model_entry(m)
            if live_ids is not None:
                e["available"] = any(is_variant(i, m["id"]) for i in live_ids)
            out.append(e)
        if live is not None:
            known = {m["id"] for m in static}
            for lm in live:
                if lm["id"] in known:
                    continue
                cur = find_model(self.id, lm["id"])
                e = self._model_entry(cur or {"id": lm["id"]})
                e.update({"id": lm["id"], "name": lm.get("name") or lm["id"], "available": True, "default": False,
                          "recommended": False})
                if not cur:
                    e["notes"] = lm.get("notes") or "Not in Fovea's price table — cost is not tracked."
                out.append(e)
        if cfg_default:
            for e in out:
                e["default"] = e["id"] == cfg_default
            if not any(e["default"] for e in out):
                out.insert(0, {**self._model_entry(find_model(self.id, cfg_default) or {"id": cfg_default}),
                               "id": cfg_default, "name": cfg_default, "default": True})
        return out

    def _model_entry(self, m: dict) -> dict:
        p = m.get("price")
        return {"id": m["id"], "name": m.get("name") or m["id"], "notes": m.get("notes", ""),
                "recommended": bool(m.get("recommended")), "default": bool(m.get("default")),
                "price": dict(p) if p else None, "max_edge": self.native_edge(m["id"], {}),
                "efforts": list(m.get("efforts") or m.get("thinking_levels") or []),
                "bench": dict(m["bench"]) if m.get("bench") else None}

    def status(self) -> dict:
        if not httpx_available():
            return {"available": False, "reason": HTTPX_MISSING}
        if self.needs_key and not self.key():
            return {"available": False, "reason": f"Add an API key for {self.name} in Settings"}
        return {"available": True, "reason": None}
