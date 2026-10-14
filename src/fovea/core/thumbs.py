"""Thumbnail generation with an on-disk cache keyed by path + mtime + size.

Every display decode goes through decode_display(): EXIF orientation, high-bit-depth stretching, ICC colour
and the pixel limit are handled in one place, so thumbnails, full renditions and resized exports agree.
"""
from __future__ import annotations

import functools
import hashlib
import io
import logging
import os
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from PIL import Image, ImageOps

from ..config import MAX_IMAGE_PIXELS, THUMB_CACHE_MAX_BYTES, THUMB_DIR, THUMB_SIZES

log = logging.getLogger("fovea.thumbs")

_locks: dict = {}
_locks_guard = threading.Lock()
# Decoding is CPU- and memory-bound: more parallel decodes than cores only adds memory pressure, and a few
# huge rasters at once can exhaust RAM, so big ones get their own narrower gate.
_DECODE_SLOTS = threading.BoundedSemaphore(max(2, min(8, os.cpu_count() or 2)))
_BIG_SLOTS = threading.BoundedSemaphore(2)
BIG_PIXELS = 24_000_000


def snap_size(s: int) -> int:
    for cand in THUMB_SIZES:
        if s <= cand:
            return cand
    return THUMB_SIZES[-1]


HIGH_BIT_MODES = ("I", "I;16", "I;16B", "I;16L", "I;16N", "F")
WEB_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".avif"}   # what every browser decodes natively
_TRANSPOSE = {2: Image.Transpose.FLIP_LEFT_RIGHT, 3: Image.Transpose.ROTATE_180, 4: Image.Transpose.FLIP_TOP_BOTTOM,
              5: Image.Transpose.TRANSPOSE, 6: Image.Transpose.ROTATE_270, 7: Image.Transpose.TRANSVERSE,
              8: Image.Transpose.ROTATE_90}


class ImageTooLarge(Exception):
    """The image declares more pixels than FOVEA_MAX_IMAGE_PIXELS allows to decode."""


@dataclass(frozen=True)
class Probe:
    orient: int = 1
    mode: str = ""
    width: int = 0
    height: int = 0
    format: str = ""
    deep_rgb: bool = False       # 16-bit-per-channel colour: Pillow only sees the high byte
    icc: bool = False
    ok: bool = False

    @property
    def pixels(self) -> int:
        return self.width * self.height

    @property
    def high_bit(self) -> bool:
        return self.mode in HIGH_BIT_MODES or self.deep_rgb

    @property
    def display_size(self) -> Tuple[int, int]:
        return (self.height, self.width) if self.orient in (5, 6, 7, 8) else (self.width, self.height)


def _deep_rgb(im: Image.Image) -> bool:
    if im.mode not in ("RGB", "RGBA"):
        return False
    try:
        if im.format == "TIFF":
            bps = im.tag_v2.get(258)
            bps = bps if isinstance(bps, (tuple, list)) else (bps,)
            return any(int(b or 8) > 8 for b in bps)
        if im.format == "PNG" and im.tile:
            raw = im.tile[0].args if hasattr(im.tile[0], "args") else im.tile[0][3]
            raw = raw[0] if isinstance(raw, tuple) else raw
            return isinstance(raw, str) and ";16" in raw
    except Exception:  # noqa: BLE001
        return False
    return False


def png_over_cap(im: Image.Image) -> bool:
    """(qa-engine hook) True for a PNG beyond MAX_IMAGE_PIXELS whose EXIF is not in its header: reading it
    would decode every pixel."""
    return im.format == "PNG" and "exif" not in im.info and im.width * im.height > MAX_IMAGE_PIXELS


@functools.lru_cache(maxsize=200_000)
def _probe(abs_path: str, mtime: float) -> Probe:
    """Header facts, once per file version per process."""
    try:
        with Image.open(abs_path) as im:
            try:
                # (qa-engine hook) a PNG's getexif() decodes the whole raster looking for a trailing eXIf chunk:
                # never for one beyond the decode cap (a 389 KB PNG can declare 20000 x 20000)
                orient = 1 if png_over_cap(im) else int(im.getexif().get(0x0112) or 1)
            except Exception:  # noqa: BLE001 — a malformed EXIF block is simply "no orientation"
                orient = 1
            return Probe(orient=orient if 1 <= orient <= 8 else 1, mode=im.mode, width=im.size[0], height=im.size[1],
                         format=im.format or "", deep_rgb=_deep_rgb(im), icc=bool(im.info.get("icc_profile")), ok=True)
    except Exception:  # noqa: BLE001
        return Probe()


def probe(abs_path: str) -> Probe:
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return Probe()
    return _probe(abs_path, mtime)


def _key(abs_path: str, size, mtime: float, pr: Optional[Probe] = None) -> str:
    pr = pr or Probe()
    base = f"{abs_path}|{size}|{mtime:.3f}"
    # Only images whose rendering changed get a new key, so existing caches of ordinary images stay valid.
    if pr.orient not in (0, 1):
        base += f"|o{pr.orient}"
    if pr.mode in HIGH_BIT_MODES:
        base += "|n3"                  # n3: robust percentile stretch (n2 was min/max and failed for big 16-bit)
    if pr.deep_rgb:
        base += "|d1"
    if pr.icc:
        base += "|c1"
    return hashlib.sha1(base.encode("utf-8", "surrogateescape")).hexdigest()


# ------------------------------------------------------------------------------------------ high bit depth

def _robust_range(f: Image.Image, sentinel: Optional[float]) -> Optional[Tuple[float, float]]:
    """0.5-99.5 percentile window of a single-channel image, ignoring 0 and the type's max value (common
    'no data' / saturated sentinels) unless they are most of the image. One hot pixel no longer turns the
    whole picture black, as a plain min/max stretch did."""
    try:
        import numpy as np
        a = np.asarray(f, dtype=np.float64).ravel()
        a = a[np.isfinite(a)]
        if a.size == 0:
            return None
        if a.size > 2_000_000:
            a = a[:: a.size // 1_000_000]
        keep = a != 0
        if sentinel is not None:
            keep &= a != sentinel
        vals = a[keep] if keep.sum() >= 0.01 * a.size else a
        lo, hi = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
        if hi <= lo:
            lo, hi = float(vals.min()), float(vals.max())
        return lo, hi
    except ImportError:
        pass
    lo, hi = f.getextrema()
    if lo is None or hi is None or hi <= lo:
        return (lo, hi) if lo is not None else None
    # Two histogram passes (Pillow only): find the 0.5/99.5 % bins, then refine inside them.
    for _ in range(2):
        hist = f.histogram(extrema=(lo, hi))
        total = sum(hist)
        if not total:
            break
        step = (hi - lo) / len(hist)
        acc, a_bin, b_bin = 0, 0, len(hist) - 1
        for i, c in enumerate(hist):
            acc += c
            if acc >= total * 0.005:
                a_bin = i
                break
        acc = 0
        for i in range(len(hist) - 1, -1, -1):
            acc += hist[i]
            if acc >= total * 0.005:
                b_bin = i
                break
        nlo, nhi = lo + a_bin * step, lo + (b_bin + 1) * step
        if nhi <= nlo:
            break
        lo, hi = nlo, nhi
    return lo, hi


_SENTINEL = {"I;16": 65535.0, "I;16B": 65535.0, "I;16L": 65535.0, "I;16N": 65535.0, "I": 2147483647.0}


def _stretch(im: Image.Image) -> Image.Image:
    f = im.convert("F")
    rng = _robust_range(f, _SENTINEL.get(im.mode))
    if not rng or rng[1] <= rng[0]:
        return Image.new("L", im.size, 128)
    lo, hi = rng
    scale = 255.0 / (hi - lo)
    return f.point(lambda v: v * scale + (-lo * scale)).convert("L")    # convert("L") clamps to 0..255


def to_display_rgb(im: Image.Image, deep_rgb: bool = False) -> Image.Image:
    """8-bit RGB for display. High-bit-depth images (16-bit, 32-bit int, float) are stretched over a robust
    window of their real value range — a plain convert("RGB") clips them to a flat white or black rectangle."""
    if im.mode == "RGB":
        if deep_rgb:
            hi = max(b[1] for b in im.getextrema())
            if hi < 64:                  # 10-14-bit data in 16-bit containers: Pillow kept only the high byte
                return ImageOps.autocontrast(im, cutoff=0.5)
        return im
    if im.mode == "L":
        return im.convert("RGB")
    if im.mode in HIGH_BIT_MODES:
        return _stretch(im).convert("RGB")
    if im.mode in ("RGBA", "LA", "P", "PA") and deep_rgb:
        return to_display_rgb(im.convert("RGB"), deep_rgb=True)
    try:
        return im.convert("RGB")
    except Exception:  # noqa: BLE001
        return ImageOps.autocontrast(im.convert("L")).convert("RGB")


def _too_large(w: int, h: int) -> str:
    return (f"{w}x{h} = {w * h / 1e6:.0f} MP exceeds the {MAX_IMAGE_PIXELS / 1e6:.0f} MP decode limit "
            f"(set FOVEA_MAX_IMAGE_PIXELS to raise it)")


# A full-size deep-TIFF rendition keeps the decoded 16-bit tiles between its statistics pass and its mapping pass
# up to this many bytes; a larger page is decoded twice rather than held.
_DEEP_KEEP_BYTES = 128 << 20


def _deep_ok(page, like=None) -> bool:
    """A plain RGB raster tifffile can map: one plane deep, 3+ samples, numeric (and laid out like `like`)."""
    try:
        ok = (int(page.photometric) == 2 and int(getattr(page, "imagedepth", 1) or 1) == 1
              and 3 <= int(page.samplesperpixel) <= 16 and page.dtype is not None and page.dtype.kind in "uif")
        if ok and like is not None:
            ok = int(page.samplesperpixel) == int(like.samplesperpixel) and page.dtype == like.dtype
        return ok
    except Exception:  # noqa: BLE001
        return False


def _deep_level(tf, page0, target: Optional[Tuple[int, int]]):
    """The page to read for a rendition of at least `target` pixels: page 0, or the smallest reduced-resolution
    copy of it (SubIFDs, or 'reduced image' pages right after it, as pyramidal aerial / microscopy TIFFs carry)
    that still covers the target. Never another full-size page of a multi-page file."""
    if not target:
        return page0
    cands = []
    try:
        cands.extend(getattr(page0, "pages", None) or ())          # SubIFDs
    except Exception:  # noqa: BLE001
        pass
    for i in range(1, 32):
        try:
            p = tf.pages[i]
        except Exception:  # noqa: BLE001 — IndexError past the last page
            break
        if not getattr(p, "is_reduced", False):
            break
        cands.append(p)
    best = page0
    for p in cands:
        try:
            pw, ph = int(p.imagewidth), int(p.imagelength)
        except Exception:  # noqa: BLE001
            continue
        if (pw >= target[0] and ph >= target[1] and pw * ph < int(best.imagewidth) * int(best.imagelength)
                and _deep_ok(p, page0)):
            best = p
    return best


def _deep_blocks(page):
    """(first sample, y0, x0, block[rows, cols, samples]) per decoded tile or strip of `page`, cropped to the
    image. One segment is decoded at a time; an empty segment is skipped (it reads as 0)."""
    h, w = int(page.imagelength), int(page.imagewidth)
    for seg, idx, _shape in page.segments(maxworkers=1):
        if seg is None:
            continue
        s, _d, y0, x0 = (int(v) for v in idx[:4])
        yield s, y0, x0, seg[0, : h - y0, : w - x0]


def _deep_range(np, parts, dtype) -> Tuple[float, float]:
    """0.5-99.5 percentile window of one channel's sample, ignoring 0 and the type's max value (common
    'no data' / saturated sentinels) unless they are most of it."""
    vals = np.concatenate(parts) if parts else np.empty(0)
    if dtype.kind == "f":
        vals = vals[np.isfinite(vals)]
    if not vals.size:
        return 0.0, 1.0
    keep = vals != 0
    if dtype.kind in "ui":
        keep &= vals != np.iinfo(dtype).max
    vals = vals[keep] if keep.sum() >= 0.01 * vals.size else vals
    lo, hi = (float(x) for x in np.percentile(vals, [0.5, 99.5]))
    return lo, (hi if hi > lo else lo + 1.0)


def _read_deep_tiff(abs_path: str, target: Optional[Tuple[int, int]] = None) -> Optional[Image.Image]:
    """16-bit-per-channel TIFF at full precision (tifffile + numpy), stretched per channel to 8-bit RGB.

    Only the first page is read (what Pillow and the index see), and its size is checked against the decode
    limit before any pixel is. With `target` (w, h in stored orientation) the page is box-averaged by an integer
    factor while its tiles are decoded, so a thumbnail never holds the 16-bit raster: the result is at least
    `target` and the caller resizes it. Values map to 8 bits through a lookup table (no float copy of the image).
    None when the optional packages are missing or tifffile cannot decode the file: the caller then works from
    Pillow's 8-bit view. Raises ImageTooLarge."""
    try:
        import numpy as np
        import tifffile
    except ImportError:
        return None
    try:
        with tifffile.TiffFile(abs_path) as tf:
            page0 = tf.pages[0]
            w0, h0 = int(page0.imagewidth), int(page0.imagelength)
            if w0 * h0 > MAX_IMAGE_PIXELS:
                raise ImageTooLarge(_too_large(w0, h0))
            if not _deep_ok(page0):
                return None
            level = _deep_level(tf, page0, target)
            try:
                return _deep_render(np, level, target)
            except Exception:  # noqa: BLE001 — a pyramid level tifffile cannot decode: try the page itself
                if level is page0:
                    raise
                return _deep_render(np, page0, target)
    except ImageTooLarge:
        raise
    except Exception as e:  # noqa: BLE001 — e.g. a compression tifffile cannot decode without imagecodecs
        log.debug("tifffile could not read %s: %s", abs_path, e)
        return None


def _cell_starts(np, origin: int, n: int, k: int):
    """Offsets within a run of n pixels starting at `origin` where a new k-pixel cell of the reduced grid begins."""
    first = (-origin) % k
    return np.concatenate(([0], np.arange(first or k, n, k))).astype(np.intp)


def _deep_render(np, page, target: Optional[Tuple[int, int]]) -> Image.Image:
    h, w = int(page.imagelength), int(page.imagewidth)
    k = max(1, min(w // max(1, target[0]), h // max(1, target[1]))) if target else 1
    oh, ow = -(-h // k), -(-w // k)
    sy, sx = max(1, h // 512), max(1, w // 512)          # statistics: about 512 x 512 samples per channel
    parts: list = [[], [], []]
    acc = np.zeros((oh, ow, 3), np.float32) if k > 1 else None
    kept = [] if k == 1 and h * w * int(page.samplesperpixel) * page.dtype.itemsize <= _DEEP_KEEP_BYTES else None
    for s, y0, x0, block in _deep_blocks(page):
        nc = min(block.shape[-1], 3 - s)
        if nc <= 0:
            continue
        if kept is not None:
            kept.append((s, y0, x0, block))
        for j in range(nc):
            sample = block[(-y0) % sy::sy, (-x0) % sx::sx, j]
            if sample.size:
                parts[s + j].append(sample.flatten())        # a copy: a view would keep the whole segment alive
        if acc is not None:                                  # box sums into the reduced grid, a tile at a time
            rs, cs = _cell_starts(np, y0, block.shape[0], k), _cell_starts(np, x0, block.shape[1], k)
            sums = np.add.reduceat(np.add.reduceat(block[..., :nc], cs, axis=1, dtype=np.float32), rs, axis=0)
            acc[y0 // k: y0 // k + len(rs), x0 // k: x0 // k + len(cs), s: s + nc] += sums
    ranges = [_deep_range(np, parts[c], page.dtype) for c in range(3)]
    del parts
    out = np.zeros((oh, ow, 3), np.uint8)
    if acc is not None:
        cy = np.minimum(k, h - np.arange(oh) * k).astype(np.float32)        # pixels per cell (edges: fewer)
        cx = np.minimum(k, w - np.arange(ow) * k).astype(np.float32)
        acc /= (cy[:, None] * cx[None, :])[..., None]
        for c, (lo, hi) in enumerate(ranges):
            out[..., c] = np.clip((acc[..., c] - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
        return Image.fromarray(out)
    lut = None
    if page.dtype.kind == "u" and page.dtype.itemsize <= 2:
        n = 1 << (8 * page.dtype.itemsize)
        lut = [np.clip((np.arange(n, dtype=np.float32) - lo) * (255.0 / (hi - lo)), 0, 255).astype(np.uint8)
               for lo, hi in ranges]
    for s, y0, x0, block in (kept if kept is not None else _deep_blocks(page)):
        for j in range(min(block.shape[-1], 3 - s)):
            ch, c = block[..., j], s + j
            dst = out[y0: y0 + ch.shape[0], x0: x0 + ch.shape[1], c]
            if lut is not None:
                dst[...] = lut[c][ch]
            else:
                lo, hi = ranges[c]
                dst[...] = np.clip(np.nan_to_num((ch.astype(np.float32) - lo) * (255.0 / (hi - lo))), 0, 255).astype(np.uint8)
    return Image.fromarray(out)


# ------------------------------------------------------------------------------------------ colour

_SRGB = None


def _to_srgb(im: Image.Image, icc: bytes) -> Optional[Image.Image]:
    """Convert to sRGB through the embedded profile (CMYK and other non-RGB colour spaces)."""
    global _SRGB
    try:
        from PIL import ImageCms
        if _SRGB is None:
            _SRGB = ImageCms.createProfile("sRGB")
        src = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        return ImageCms.profileToProfile(im, src, _SRGB, outputMode="RGB")
    except Exception:  # noqa: BLE001 — a broken profile: fall back to Pillow's plain conversion
        return None


# ------------------------------------------------------------------------------------------ decode

def _fit(w: int, h: int, side: int) -> Tuple[int, int]:
    s = min(1.0, side / max(1, w, h))
    return max(1, round(w * s)), max(1, round(h * s))


def decode_display(abs_path: str, max_side: Optional[int] = None, pr: Optional[Probe] = None,
                   exact: Optional[Tuple[int, int]] = None, hq: bool = False) -> Tuple[Image.Image, Optional[bytes]]:
    """Decode for display: (8-bit RGB image upright per EXIF, ICC profile to embed or None).

    max_side shrinks to fit (never enlarges); exact resizes to (w, h). JPEGs are DCT-downscaled while
    decoding (draft), so a 6000 px photo costs a fraction of a full decode. hq selects Lanczos resampling
    (exports) over bilinear (thumbnails). Raises ImageTooLarge."""
    pr = pr or probe(abs_path)
    # What the decode really holds: a 16-bit colour TIFF is read at full precision (2 bytes per sample) and
    # mapped into a separate 8-bit raster, about twice an 8-bit decode.
    big = pr.pixels * (2 if pr.deep_rgb else 1) > BIG_PIXELS
    with _DECODE_SLOTS:
        if big:
            _BIG_SLOTS.acquire()
        try:
            return _decode(abs_path, max_side, pr, exact, hq)
        finally:
            if big:
                _BIG_SLOTS.release()


def _decode(abs_path: str, max_side: Optional[int], pr: Probe, exact: Optional[Tuple[int, int]], hq: bool):
    try:
        im = Image.open(abs_path)
    except Image.DecompressionBombError as e:
        raise ImageTooLarge(str(e))
    with im:
        target = None
        if exact:
            target = exact if pr.orient not in (5, 6, 7, 8) else (exact[1], exact[0])
        elif max_side:
            target = _fit(im.size[0], im.size[1], max_side)
        if target and im.format == "JPEG":
            try:
                im.draft("RGB" if im.mode not in ("L", "CMYK") else im.mode, target)
            except Exception:  # noqa: BLE001
                pass
        w, h = im.size
        if w * h > MAX_IMAGE_PIXELS:
            raise ImageTooLarge(_too_large(w, h))
        icc = im.info.get("icc_profile")
        deep = _read_deep_tiff(abs_path, target) if pr.deep_rgb and im.format == "TIFF" else None
        if deep is not None:
            work, deep_rgb = deep, False
        else:
            work, deep_rgb = im, pr.deep_rgb
        resample = Image.Resampling.LANCZOS if hq and work.mode not in HIGH_BIT_MODES else Image.Resampling.BILINEAR
        if exact or (deep is not None and target and work.size != target):
            # a deep TIFF arrives reduced by a whole factor (rounded up): size it like every other decode
            work = work.resize(target, resample)
        elif max_side and max(work.size) > max_side:
            # reducing_gap=None: Image.reduce() rejects 16-bit modes ("image has wrong mode").
            work.thumbnail((max_side, max_side), resample, reducing_gap=None if work.mode in HIGH_BIT_MODES else 2.0)
        else:
            work.load()
        if pr.orient in _TRANSPOSE:
            # Browsers rotate photos by their EXIF orientation; a rendition that did not would show the
            # picture sideways with its (upright-normalised) boxes on the wrong objects.
            work = work.transpose(_TRANSPOSE[pr.orient])
        embed: Optional[bytes] = None
        if icc and work.mode == "CMYK":
            conv = _to_srgb(work, icc)
            if conv is not None:
                work = conv
        elif icc and work.mode in ("RGB", "RGBA") and not deep_rgb:
            embed = icc                  # keep the colour space: browsers colour-manage tagged JPEGs
        out = to_display_rgb(work, deep_rgb=deep_rgb)
        if out is im:
            out = out.copy()
        return out, embed


# ------------------------------------------------------------------------------------------ cache

_stats_lock = threading.Lock()
_stats = {"files": 0, "bytes": 0, "known": False, "counting": False, "counted_at": None}
_trim_state = {"running": False, "last": 0.0}
_failures: "OrderedDict[str, Tuple[float, str]]" = OrderedDict()
_FAIL_TTL = 600.0


def _account(files: int, nbytes: int) -> None:
    with _stats_lock:
        _stats["files"] += files
        _stats["bytes"] += nbytes
        over = _stats["known"] and THUMB_CACHE_MAX_BYTES and _stats["bytes"] > THUMB_CACHE_MAX_BYTES
    if over:
        _maybe_trim()


def _remember_failure(key: str, reason: str) -> None:
    with _stats_lock:
        _failures[key] = (time.time(), reason)
        _failures.move_to_end(key)
        while len(_failures) > 20_000:
            _failures.popitem(last=False)


def _recent_failure(key: str) -> Optional[str]:
    with _stats_lock:
        hit = _failures.get(key)
        if hit and time.time() - hit[0] < _FAIL_TTL:
            return hit[1]
        if hit:
            _failures.pop(key, None)
    return None


def _cached(out: Path) -> bool:
    """A cache hit; refreshes the file's mtime at most daily so trimming removes the least recently used."""
    try:
        st = os.stat(out)
    except OSError:
        return False
    if time.time() - st.st_mtime > 86400:
        try:
            os.utime(out)
        except OSError:
            pass
    return True


def _render(abs_path: str, out: Path, key: str, size: Optional[int], quality: int, pr: Probe,
            source: Optional[str] = None) -> Tuple[Optional[Path], Optional[str]]:
    if _cached(out):
        return out, None
    failed = _recent_failure(key)
    if failed:
        return None, failed
    with _locks_guard:
        lock = _locks.setdefault(key, threading.Lock())
    with lock:
        if _cached(out):
            return out, None
        out.parent.mkdir(parents=True, exist_ok=True)
        tmp = out.with_name(f"{out.stem}.{threading.get_ident()}.tmp")
        try:
            if source:            # derive from a cached larger rendition instead of decoding the original again
                with Image.open(source) as base:
                    icc = base.info.get("icc_profile")
                    im = base.convert("RGB")
                im.thumbnail((size, size), Image.Resampling.BILINEAR)
            else:
                im, icc = decode_display(abs_path, size, pr)
            im.save(tmp, "JPEG", quality=quality, optimize=bool(size) and size <= 512,
                    **({"icc_profile": icc} if icc else {}))
            nbytes = os.path.getsize(tmp)
            os.replace(tmp, out)
            _account(1, nbytes)
        except ImageTooLarge as e:
            _cleanup(tmp)
            _remember_failure(key, str(e))
            return None, str(e)
        except Exception as e:  # noqa: BLE001
            _cleanup(tmp)
            reason = "This image could not be decoded" if not isinstance(e, OSError) else f"Could not read the image ({e.__class__.__name__})"
            _remember_failure(key, reason)
            log.info("thumbnail failed for %s: %s", abs_path, e)
            return None, reason
        finally:
            with _locks_guard:
                _locks.pop(key, None)
    return out, None


def _cleanup(tmp: Path) -> None:
    try:
        if tmp.exists():
            tmp.unlink()
    except OSError:
        pass


def _thumb_path(key: str) -> Path:
    return THUMB_DIR / key[:2] / f"{key}.jpg"


def thumb_render(abs_path: str, size: int) -> Tuple[Optional[Path], Optional[str]]:
    """(cached JPEG path, None) or (None, reason)."""
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return None, "Image file not found"
    size = snap_size(size)
    pr = _probe(abs_path, mtime)
    key = _key(abs_path, size, mtime, pr)
    out = _thumb_path(key)
    if _cached(out):
        return out, None
    quality = 82 if size <= 512 else 85
    largest = 1024
    # Formats without draft decoding (PNG, TIFF, WebP...) pay a full decode per size: decode once into the
    # 1024 rendition and derive the smaller sizes from it.
    if size < largest and pr.ok and pr.format != "JPEG" and pr.pixels > 4 * largest * largest:
        base, err = thumb_render(abs_path, largest)
        if base is not None:
            return _render(abs_path, out, key, size, quality, pr, source=str(base))
        if err:
            return None, err
    return _render(abs_path, out, key, size, quality, pr)


def thumb_for(abs_path: str, size: int) -> Optional[Path]:
    return thumb_render(abs_path, size)[0]


def passthrough_ok(abs_path: str, size: int, pr: Optional[Probe] = None) -> bool:
    """True when the original file can be served as the requested thumbnail as is: a browser-native image
    no larger than the tile, upright, 8-bit. Re-encoding it would only cost time and quality."""
    pr = pr or probe(abs_path)
    if not pr.ok or pr.high_bit or pr.orient not in (0, 1) or pr.mode == "CMYK":
        return False
    if Path(abs_path).suffix.lower() not in WEB_EXTS:
        return False
    return max(pr.width, pr.height) <= snap_size(size)


def needs_rendition(abs_path: str) -> bool:
    """Browsers cannot display the file as is: a non-web format (TIFF, JP2, HEIC) or a high-bit-depth
    PNG whose low-range data a browser would show near-black."""
    if Path(abs_path).suffix.lower() not in WEB_EXTS:
        return True
    pr = probe(abs_path)
    return pr.ok and pr.high_bit


def rendition_render(abs_path: str) -> Tuple[Optional[Path], Optional[str]]:
    """Full-resolution JPEG of an image browsers cannot display (TIFF, 16-bit), cached beside the thumbnails.
    Native resolution on purpose: a downscaled proxy would quietly cap the detail available to label."""
    try:
        mtime = os.stat(abs_path).st_mtime
    except OSError:
        return None, "Image file not found"
    pr = _probe(abs_path, mtime)
    key = _key(abs_path, "full", mtime, pr)
    return _render(abs_path, THUMB_DIR / "full" / key[:2] / f"{key}.jpg", key, None, 92, pr)


def rendition_for(abs_path: str) -> Optional[Path]:
    return rendition_render(abs_path)[0]


def render_tag(abs_path: str) -> str:
    """Short token describing how /api/img renders this file (part of its ETag)."""
    pr = probe(abs_path)
    return hashlib.sha1(f"{pr.orient}|{pr.mode}|{pr.deep_rgb}|{pr.icc}|r3".encode()).hexdigest()[:8]


def _walk_files():
    stack = [str(THUMB_DIR)]
    while stack:
        d = stack.pop()
        try:
            with os.scandir(d) as it:
                for e in it:
                    try:
                        if e.is_dir(follow_symlinks=False):
                            stack.append(e.path)
                        elif e.is_file(follow_symlinks=False):
                            yield e
                    except OSError:
                        continue
        except OSError:
            continue


def _count() -> None:
    files = total = 0
    for e in _walk_files():
        try:
            total += e.stat(follow_symlinks=False).st_size
            files += 1
        except OSError:
            pass
    with _stats_lock:
        # Renders that happened while walking were already added on top of the old numbers; the walk saw
        # most of them too, so take the walk as the truth.
        _stats.update(files=files, bytes=total, known=True, counting=False, counted_at=time.time())
    if THUMB_CACHE_MAX_BYTES and total > THUMB_CACHE_MAX_BYTES:
        _maybe_trim()


def cache_stats(refresh: bool = False, wait: bool = False) -> dict:
    """Cheap: numbers are kept by a running counter. The first call (or refresh) starts a background walk;
    until it finishes 'pending' is true. wait=True walks synchronously (for tests and the CLI)."""
    with _stats_lock:
        start = (refresh or not _stats["known"]) and not _stats["counting"]
        if start:
            _stats["counting"] = True
    if start:
        if wait:
            _count()
        else:
            threading.Thread(target=_count, name="fovea-thumb-count", daemon=True).start()
    with _stats_lock:
        return {"files": _stats["files"] if _stats["known"] else None,
                "bytes": _stats["bytes"] if _stats["known"] else None,
                "pending": _stats["counting"], "counted_at": _stats["counted_at"],
                "max_bytes": THUMB_CACHE_MAX_BYTES or None}


def _maybe_trim() -> None:
    with _stats_lock:
        if _trim_state["running"] or time.time() - _trim_state["last"] < 60:
            return
        _trim_state["running"] = True
    threading.Thread(target=_trim_worker, name="fovea-thumb-trim", daemon=True).start()


def _trim_worker() -> None:
    try:
        trim_cache(int(THUMB_CACHE_MAX_BYTES * 0.9))
    finally:
        with _stats_lock:
            _trim_state.update(running=False, last=time.time())


def trim_cache(target_bytes: int) -> dict:
    """Remove the least recently used cache files until the cache is at most target_bytes."""
    entries = []
    for e in _walk_files():
        try:
            st = e.stat(follow_symlinks=False)
        except OSError:
            continue
        entries.append((st.st_mtime, st.st_size, e.path))
    total = sum(s for _m, s, _p in entries)
    removed = freed = 0
    if total > target_bytes:
        entries.sort()
        cutoff = time.time() - 60          # never a file that is being written right now
        for m, s, p in entries:
            if total - freed <= target_bytes:
                break
            if m > cutoff:
                continue
            try:
                os.unlink(p)
                removed += 1
                freed += s
            except OSError:
                pass
    with _stats_lock:
        _stats.update(files=len(entries) - removed, bytes=total - freed, known=True, counted_at=time.time())
    if removed:
        log.info("thumbnail cache trimmed: %d files, %.1f MB", removed, freed / 1e6)
    return {"removed": removed, "freed": freed}


def clear_cache() -> int:
    n = 0
    for e in list(_walk_files()):
        try:
            os.unlink(e.path)
            n += 1
        except OSError:
            pass
    with _stats_lock:
        _stats.update(files=0, bytes=0, known=True, counted_at=time.time())
        _failures.clear()
    return n
