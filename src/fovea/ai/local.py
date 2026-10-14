"""Local inference runtime shared by the ultralytics-backed providers (yolo, yoloe, sam).

* ultralytics / torch are imported lazily, once, with downloads and auto-installs switched off
  (``YOLO_OFFLINE``/``YOLO_AUTOINSTALL``) and torch's thread count restored after ultralytics pins it to 1.
* device ``auto`` resolves to ``cuda:0`` > ``mps`` > ``cpu`` (ultralytics alone picks CPU on macOS — 17x slower here);
  the ``ai.settings.device`` setting overrides it.
* loaded models stay warm in a small LRU keyed by ``(family, weights, mode, device)``: ``ai.settings.keep_models``
  detectors (default 2) plus one SAM model in its own slot. Every entry carries its own lock, because ultralytics
  predictors keep per-call state (args, prompts, features) and are NOT safe to share between threads. Callers hold
  ``entry.lock`` around predict + result extraction. Each model runs one dummy inference right after loading, so
  MPS/CUDA kernel compilation is paid while loading, not on the first image.
* availability checks use ``importlib.util.find_spec`` (cached) and never import torch on a request thread.
"""
from __future__ import annotations

import gc
import importlib.util
import os
import statistics
import threading
import time
from collections import OrderedDict
from contextlib import contextmanager
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple

from . import config

_spec_cache: Dict[str, bool] = {}
_import_lock = threading.Lock()
_ready = False
_import_error: Optional[str] = None
_warm_thread: Optional[threading.Thread] = None


def setup_env() -> None:
    """Must run before ``import ultralytics``: no network, no pip installs, no console spam."""
    os.environ["YOLO_OFFLINE"] = "true"
    os.environ["YOLO_AUTOINSTALL"] = "false"
    os.environ.setdefault("YOLO_VERBOSE", "false")


setup_env()


def has_module(name: str) -> bool:
    v = _spec_cache.get(name)
    if v is None:
        try:
            v = importlib.util.find_spec(name) is not None
        except (ImportError, ValueError):
            v = False
        _spec_cache[name] = v
    return v


def ultralytics_status() -> Tuple[bool, Optional[str]]:
    """Cheap: never imports anything heavy."""
    if not has_module("ultralytics") or not has_module("torch"):
        return False, "ultralytics / torch are not installed (pip install ultralytics)"
    if _import_error:
        return False, f"ultralytics failed to import: {_import_error}"
    return True, None


def is_imported() -> bool:
    return _ready


def ensure_ultralytics() -> None:
    """Import ultralytics + torch once (seconds on first call)."""
    global _ready, _import_error
    if _ready:
        return
    with _import_lock:
        if _ready:
            return
        ok, reason = ultralytics_status()
        if not ok:
            from .base import ProviderError
            raise ProviderError(reason or "ultralytics is not available", "unavailable")
        setup_env()
        try:
            import torch
            import ultralytics  # noqa: F401  (sets OMP_NUM_THREADS=1 when unset)
            from ultralytics.utils import LOGGER
            LOGGER.setLevel(40)                                   # errors only: inference runs on request threads
            # ultralytics resolves YOLO / YOLOE / SAM lazily on first attribute access. Two request threads doing
            # that at once see a partially initialised ultralytics.models ("circular import"), so import every
            # entry point we use here, once, under the lock.
            from ultralytics import YOLO, YOLOE  # noqa: F401
            import ultralytics.models.sam  # noqa: F401
            import ultralytics.models.yolo.yoloe  # noqa: F401
            import ultralytics.nn.tasks  # noqa: F401
            import ultralytics.nn.text_model  # noqa: F401
        except Exception as e:  # noqa: BLE001
            _import_error = f"{type(e).__name__}: {e}"
            from .base import ProviderError
            raise ProviderError(f"ultralytics failed to import: {_import_error}", "unavailable") from e
        try:
            torch.set_num_threads(max(1, (os.cpu_count() or 2) - 1))
        except Exception:  # noqa: BLE001
            pass
        _ready = True


def warm_async() -> None:
    """Import the stack in the background so the first real request does not pay for it."""
    global _warm_thread
    if _ready or not ultralytics_status()[0]:
        return
    if _warm_thread and _warm_thread.is_alive():
        return

    def _run():
        try:
            ensure_ultralytics()
        except Exception:  # noqa: BLE001 — reported through ultralytics_status()
            pass

    _warm_thread = threading.Thread(target=_run, name="fovea-ai-warm", daemon=True)
    _warm_thread.start()


# ------------------------------------------------------------------ device

def available_devices() -> List[str]:
    """Only meaningful once torch is imported; ['cpu'] otherwise."""
    if not _ready:
        return ["cpu"]
    import torch
    out = ["cpu"]
    try:
        if torch.backends.mps.is_available():
            out.append("mps")
    except Exception:  # noqa: BLE001
        pass
    if torch.cuda.is_available():
        out.append("cuda")
    return out


def resolve_device(setting: Optional[str] = None) -> str:
    """auto -> cuda:0 > mps > cpu. An explicit device that is not available falls back to auto."""
    ensure_ultralytics()
    setting = (setting or config.ai_settings().get("device") or "auto").lower()
    avail = available_devices()
    if setting == "cuda" and "cuda" in avail:
        return "cuda:0"
    if setting == "mps" and "mps" in avail:
        return "mps"
    if setting == "cpu":
        return "cpu"
    if "cuda" in avail:
        return "cuda:0"
    if "mps" in avail:
        return "mps"
    return "cpu"


def device_info() -> dict:
    """For the providers list. Never blocks on importing torch: reports ``resolved: None`` while warming up."""
    setting = config.ai_settings().get("device", "auto")
    ok, reason = ultralytics_status()
    if not ok:
        return {"setting": setting, "resolved": None, "options": ["auto", "cpu"], "available": False, "reason": reason}
    if not _ready:
        warm_async()
        return {"setting": setting, "resolved": None, "options": ["auto", "cpu", "mps", "cuda"], "available": True,
                "reason": "Loading the AI runtime…"}
    import torch
    return {"setting": setting, "resolved": resolve_device(setting), "options": ["auto"] + available_devices(),
            "available": True, "reason": None, "threads": torch.get_num_threads()}


def free_device_memory() -> None:
    gc.collect()
    if not _ready:
        return
    import torch
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()
    except Exception:  # noqa: BLE001
        pass


# ------------------------------------------------------------------ model cache

class Loaded:
    """A model in memory. ``lock`` serialises every use of ``obj``; ``state`` is per-model scratch (e.g. the YOLOE
    vocabulary currently set) and must only be read or written under ``lock``."""

    def __init__(self, key: tuple, obj: Any, device: str, family: str, name: str):
        self.key = key
        self.obj = obj
        self.device = device
        self.family = family
        self.name = name
        self.lock = threading.RLock()
        self.state: Dict[str, Any] = {}
        self.loaded_at = time.time()
        self.last_used = self.loaded_at
        self.pins = 0
        self.ms_ema: Optional[float] = None       # moving average inference time per image (for estimates)

    def record_ms(self, ms: float) -> None:
        self.ms_ema = ms if self.ms_ema is None else 0.8 * self.ms_ema + 0.2 * ms
        _speed[(self.family, self.name)] = self.ms_ema

    def info(self) -> dict:
        return {"family": self.family, "name": self.name, "device": self.device, "mode": self.key[2] if len(self.key) > 2 else "",
                "loaded_at": self.loaded_at, "last_used": self.last_used,
                "ms_per_image": round(self.ms_ema) if self.ms_ema else None}


_speed: Dict[Tuple[str, str], float] = {}


def speed_ms(family: str, name: str) -> Optional[float]:
    return _speed.get((family, name))


# Whole-job throughput (images ÷ wall time, model loading excluded): what a batch or an audit really takes per image —
# chunked inference with the next chunk decoding meanwhile is several times faster than one call per image, so the
# per-call latency above overstates a job's time. Persisted per device in kv ``ai.speed`` (survives a restart).
SPEED_KEY = "ai.speed"
_throughput: Dict[str, dict] = {}
_throughput_loaded = False


def _speed_key(provider: str, model: str, mode: str) -> str:
    return f"{provider}|{model}|{mode}"


def _load_throughput() -> None:
    global _throughput_loaded
    if _throughput_loaded:
        return
    _throughput_loaded = True
    try:
        from .. import db
        v = db.kv_get(SPEED_KEY, {}) or {}
        if isinstance(v, dict):
            _throughput.update({k: e for k, e in v.items() if isinstance(e, dict) and e.get("s")})
    except Exception:  # noqa: BLE001 — a hint for estimates, never worth an error
        pass


def record_throughput(provider: str, model: str, mode: str, images: int, seconds: float, device: str = "") -> None:
    """Remember a finished job's seconds per image for ``(provider, model, mode)``; ``mode`` = 'batch' (chunked
    inference) or 'single' (one image per call: tiles, two-stage)."""
    if images < 2 or seconds <= 0:
        return
    _load_throughput()
    s = seconds / images
    key = _speed_key(provider, model, mode)
    old = _throughput.get(key)
    samples: List[float] = []
    if old and old.get("device") == device:           # (a record from before samples were kept is replaced)
        samples = [float(v) for v in (old.get("samples") or []) if isinstance(v, (int, float)) and v > 0]
    samples = (samples + [s])[-5:]
    # The lower median of the last few runs: one run slowed by a pause (sleep, a stopped process, another heavy job)
    # cannot move the estimate — pauses only ever make a run look slower — while a real slow-down shows in two runs.
    est = statistics.median_low(samples)
    _throughput[key] = {"s": round(est, 5), "samples": [round(v, 5) for v in samples], "device": device,
                        "images": int(images), "at": round(time.time())}
    try:
        from .. import db
        db.kv_set(SPEED_KEY, dict(_throughput))
    except Exception:  # noqa: BLE001
        pass


def job_seconds(provider: str, model: str, mode: str) -> Optional[float]:
    """Measured seconds per image of a job on this machine (None until one ran). A record from another device than
    the current one (Settings › device changed) is ignored when the current device is known."""
    _load_throughput()
    e = _throughput.get(_speed_key(provider, model, mode))
    if not e:
        return None
    if _ready and e.get("device"):
        try:
            if resolve_device() != e["device"]:
                return None
        except Exception:  # noqa: BLE001
            pass
    return float(e["s"])


class ModelCache:
    def __init__(self):
        self._entries: "OrderedDict[tuple, Loaded]" = OrderedDict()
        self._lock = threading.Lock()
        self._load_locks: Dict[tuple, threading.Lock] = {}

    def _keep(self) -> int:
        try:
            return int(config.ai_settings().get("keep_models", 2))
        except Exception:  # noqa: BLE001
            return 2

    def get(self, key: tuple, loader: Callable[[], Tuple[Any, str]], family: str, name: str) -> Loaded:
        """Return the cached entry or build it with ``loader() -> (obj, device)``. Loads of different models run in
        parallel; two requests for the same model wait for one load."""
        with self._lock:
            e = self._entries.get(key)
            if e is not None:
                self._entries.move_to_end(key)
                e.last_used = time.time()
                return e
            ll = self._load_locks.setdefault(key, threading.Lock())
        with ll:
            with self._lock:
                e = self._entries.get(key)
                if e is not None:
                    return e
            obj, device = loader()
            e = Loaded(key, obj, device, family, name)
            evicted: List[Loaded] = []
            with self._lock:
                self._entries[key] = e
                # SAM (interactive box tool) has its own slot so a detector batch never evicts it, and vice versa.
                group = family == "sam"
                keep = 1 if group else self._keep()
                same = [k for k, x in self._entries.items() if (x.family == "sam") == group]
                for k in same:
                    if len(same) - len(evicted) <= keep:
                        break
                    old = self._entries[k]
                    if old is e or old.pins > 0:
                        continue
                    evicted.append(self._entries.pop(k))
        if evicted:
            del evicted
            free_device_memory()
        return e

    @contextmanager
    def use(self, key: tuple, loader: Callable[[], Tuple[Any, str]], family: str, name: str) -> Iterator[Loaded]:
        """Pin an entry for the duration of a job so a burst of other loads cannot evict it mid-batch."""
        e = self.get(key, loader, family, name)
        with self._lock:
            e.pins += 1
        try:
            yield e
        finally:
            with self._lock:
                e.pins -= 1
                e.last_used = time.time()

    def loaded(self) -> List[dict]:
        with self._lock:
            return [e.info() for e in self._entries.values()]

    def is_loaded(self, family: str, name: str) -> bool:
        with self._lock:
            return any(e.family == family and e.name == name for e in self._entries.values())

    def unload(self, family: Optional[str] = None, name: Optional[str] = None) -> int:
        with self._lock:
            keys = [k for k, e in self._entries.items()
                    if (family is None or e.family == family) and (name is None or e.name == name) and e.pins == 0]
            for k in keys:
                self._entries.pop(k, None)
        if keys:
            free_device_memory()
        return len(keys)


CACHE = ModelCache()


def warmup_predict(model: Any, device: str, imgsz: int = 640) -> None:
    """One dummy inference right after loading: the first MPS/CUDA call compiles kernels and is several times slower
    than the rest, so pay that while loading instead of on the user's first image."""
    try:
        import numpy as np
        model.predict(np.zeros((imgsz, imgsz, 3), dtype=np.uint8), device=device, verbose=False, conf=0.5,
                      imgsz=imgsz, classes=None)
    except Exception:  # noqa: BLE001 — warm-up is best effort
        pass


# ------------------------------------------------------------------ decode guard

def too_large_error(name: str, w: Optional[int] = None, h: Optional[int] = None):
    """ProviderError 'too_large' (not fatal for a job: that image is counted as an error and skipped)."""
    from ..config import MAX_IMAGE_PIXELS
    from .base import ProviderError
    size = f"{w}×{h} = {w * h / 1e6:,.0f} MP" if w and h else f"more than {2 * MAX_IMAGE_PIXELS / 1e6:,.0f} MP"
    return ProviderError(f"{name} is too large to decode ({size}; the limit is {MAX_IMAGE_PIXELS / 1e6:,.0f} MP, "
                         "FOVEA_MAX_IMAGE_PIXELS)", "too_large")


def check_pixels(path: str) -> None:
    """Refuse an image whose header declares more pixels than FOVEA_MAX_IMAGE_PIXELS, before cv2 / ultralytics / PIL
    decode it whole: a 109 KB PNG can declare 30000×30000 px and take 2.3 GB per decode (four decoders run in
    parallel in a job). Reads the header only; a file Pillow cannot parse is left to the decoder to report."""
    import warnings
    from PIL import Image
    from ..config import MAX_IMAGE_PIXELS
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", Image.DecompressionBombWarning)
            with Image.open(path) as im:
                w, h = im.size
    except Image.DecompressionBombError:
        raise too_large_error(os.path.basename(str(path))) from None
    except Exception:  # noqa: BLE001
        return
    if w * h > MAX_IMAGE_PIXELS:
        raise too_large_error(os.path.basename(str(path)), w, h)


# ------------------------------------------------------------------ inference helpers (ultralytics detectors)

def load_bgr(path: str):
    """EXIF-transposed image as a BGR uint8 array (what ultralytics expects for ndarray sources) + (W, H)."""
    import numpy as np
    from PIL import Image, ImageOps
    check_pixels(path)
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        rgb = im.convert("RGB")
        arr = np.asarray(rgb)[:, :, ::-1]
        return np.ascontiguousarray(arr), rgb.width, rgb.height


def _boxes_of(r):
    b = getattr(r, "boxes", None)
    if b is not None and len(b):
        return b.xyxy.tolist(), [int(c) for c in b.cls.tolist()], [float(c) for c in b.conf.tolist()]
    o = getattr(r, "obb", None)                 # oriented boxes -> axis-aligned extent (finding C16)
    if o is not None and len(o):
        return o.xyxy.tolist(), [int(c) for c in o.cls.tolist()], [float(c) for c in o.conf.tolist()]
    return [], [], []


def raw_dets(results, W: Optional[float] = None, H: Optional[float] = None, offsets=None) -> List[dict]:
    """Ultralytics Results -> [{mcls, label, xc, yc, w, h, conf}] normalised to the full image (W x H). ``offsets``
    (one (ox, oy) per result) shift tile results back into the full image."""
    from .geometry import from_xyxy_px
    out: List[dict] = []
    for k, r in enumerate(results):
        oh, ow = r.orig_shape[:2]
        fw, fh = (W or ow), (H or oh)
        ox, oy = offsets[k] if offsets else (0, 0)
        names = getattr(r, "names", {}) or {}
        xyxy, cls, conf = _boxes_of(r)
        for (x0, y0, x1, y1), c, s in zip(xyxy, cls, conf):
            b = from_xyxy_px(x0 + ox, y0 + oy, x1 + ox, y1 + oy, fw, fh)
            if b is None:
                continue
            out.append({"mcls": c, "label": str(names.get(c, c)), "xc": b[0], "yc": b[1], "w": b[2], "h": b[3],
                        "conf": round(s, 4)})
    return out


def _cut_by_tile(d: dict, win, W: int, H: int, margin: float = 3.0) -> bool:
    """A tile box touching an inner tile edge is a truncated object; the neighbouring tile or the full frame has it."""
    x0, y0, x1, y1 = win
    bx0, bx1 = (d["xc"] - d["w"] / 2) * W, (d["xc"] + d["w"] / 2) * W
    by0, by1 = (d["yc"] - d["h"] / 2) * H, (d["yc"] + d["h"] / 2) * H
    return ((x0 > 0 and bx0 <= x0 + margin) or (x1 < W and bx1 >= x1 - margin)
            or (y0 > 0 and by0 <= y0 + margin) or (y1 < H and by1 >= y1 - margin))


def predict(entry: "Loaded", image_path: str, *, conf: float, iou: float, imgsz: int, classes=None, max_det: int = 300,
            tile=None, augment: bool = False) -> List[dict]:
    """Run an ultralytics detector with the FULL argument set on every call (predictor args are sticky between calls:
    a class filter left over from a previous request must never apply to this one — finding C2). Holds the model
    lock for inference and result extraction."""
    from . import geometry
    m = entry.obj
    kw = dict(conf=float(conf), iou=float(iou), imgsz=int(imgsz), classes=list(classes) if classes else None,
              device=entry.device, verbose=False, max_det=int(max_det), augment=bool(augment), half=False)
    t0 = time.time()
    check_pixels(image_path)
    if not tile:
        with entry.lock:
            res = m.predict(str(image_path), **kw)
            dets = raw_dets(res)
        entry.record_ms((time.time() - t0) * 1000)
        return dets
    im, W, H = load_bgr(image_path)
    size = int(tile) if not isinstance(tile, bool) and int(tile) >= 64 else int(imgsz)
    wins = geometry.tiles(W, H, size, 0.2)
    with entry.lock:
        dets = raw_dets(m.predict(im, **kw), W, H)                    # full frame keeps objects larger than a tile
        if len(wins) > 1:
            for i in range(0, len(wins), 8):
                part = wins[i:i + 8]
                res = m.predict([im[y0:y1, x0:x1] for x0, y0, x1, y1 in part], **{**kw, "imgsz": size})
                for r, win in zip(res, part):
                    dets += [d for d in raw_dets([r], W, H, [win[:2]]) if not _cut_by_tile(d, win, W, H)]
    entry.record_ms((time.time() - t0) * 1000)
    for d in dets:
        d["cls"] = d["mcls"]
    merged = geometry.nms(dets, 0.5, class_aware=True, ios_thr=0.7)
    for d in merged:
        d.pop("cls", None)
    return merged


_decode_pool = None
_decode_lock = threading.Lock()


def _decoder():
    global _decode_pool
    with _decode_lock:
        if _decode_pool is None:
            from concurrent.futures import ThreadPoolExecutor
            _decode_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="fovea-ai-decode")
        return _decode_pool


def _read_bgr(path: str):
    """BGR array as displayed (cv2 applies EXIF orientation, like ultralytics' own loader); PIL as the fallback for
    files cv2 cannot open. Returns the array or the exception (one unreadable image must not fail its chunk)."""
    try:
        check_pixels(path)
        import cv2
        arr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if arr is not None:
            return arr
        return load_bgr(path)[0]
    except Exception as e:  # noqa: BLE001
        return e


def decode_many(paths: List[str]) -> List[Any]:
    """Decode images in parallel (decoders release the GIL), outside any model lock: while one chunk runs on the
    GPU, the next chunk is being decoded. Each entry is an array or the exception for that file."""
    return list(_decoder().map(_read_bgr, paths))


def predict_arrays(entry: "Loaded", arrays: List[Any], *, conf: float, iou: float, imgsz: int, classes=None,
                   max_det: int = 300, augment: bool = False) -> List[List[dict]]:
    """One forward pass over several decoded images (``predict`` semantics: full argument set every call, model lock
    held). Measured on MPS (yolo12m, 640 px frames): 8.6 img/s one image per call, 14.8 img/s with chunks of 8 plus
    decoding the next chunk while this one runs."""
    if not arrays:
        return []
    m = entry.obj
    kw = dict(conf=float(conf), iou=float(iou), imgsz=int(imgsz), classes=list(classes) if classes else None,
              device=entry.device, verbose=False, max_det=int(max_det), augment=bool(augment), half=False)
    t0 = time.time()
    with entry.lock:
        res = m.predict(list(arrays), **kw)
        dets = [raw_dets([r]) for r in res]
    entry.record_ms((time.time() - t0) * 1000 / len(arrays))
    return dets


def image_size(path: str):
    """(W, H) as displayed, i.e. after EXIF orientation — header read only."""
    from PIL import Image
    with Image.open(path) as im:
        w, h = im.size
        try:
            o = im.getexif().get(0x0112, 1)
        except Exception:  # noqa: BLE001
            o = 1
    return (h, w) if o in (5, 6, 7, 8) else (w, h)
