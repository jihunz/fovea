"""Provider contract shared by every AI backend (local YOLO / YOLOE / SAM, Claude, Gemini, OpenAI, OpenAI-compatible).

A provider turns one image into boxes for the dataset's classes. Everything provider-specific — image resizing,
the model's native coordinate frame, the JSON schema, retries — stays inside the provider. What comes out is always
normalised YOLO geometry already mapped to *dataset* class ids, so callers never see a model's own vocabulary.

Nothing here writes label files. Callers store results as proposals (``ai_preds``) or, only when the user explicitly
chose a write mode, through ``core.labels.write_label_file`` under ``LABEL_LOCK``.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Iterator, List, Optional, Tuple, Union


class ProviderError(Exception):
    """A request failed in a way the user should see. ``code`` drives retry / abort decisions in batch jobs.

    code: auth (bad/missing key — abort job) · quota (spend cap / billing — abort job) · rate (429 — retried by the
    provider; raised only after retries are exhausted) · overloaded (5xx/529 after retries) · refusal (model declined
    this image — skip it) · bad_output (unparseable / truncated output — skip it) · unavailable (dependency or weights
    missing — abort) · invalid (bad request — abort) · network (connection failed after retries) · too_large (the
    image declares more pixels than FOVEA_MAX_IMAGE_PIXELS allows to decode — skip it)."""

    user_facing = True       # jobs.py: its message is the whole story — shown without the class name, logged in one line

    def __init__(self, message: str, code: str = "invalid", *, retry_after: Optional[float] = None,
                 status: Optional[int] = None, usage: Optional["Usage"] = None):
        super().__init__(message)
        self.code = code
        self.retry_after = retry_after
        self.status = status
        # --- ai-cloud hook: what the provider billed before this failed (a refusal, unreadable or cut-off output and
        # every earlier call for the same item are paid for). Callers book it like a success's usage (billed_usage).
        self.usage = usage

    @property
    def fatal(self) -> bool:
        """True when continuing a batch would fail the same way for every remaining image."""
        return self.code in ("auth", "quota", "unavailable", "invalid")


# --- ai-cloud hook: billed usage carried on errors -------------------------------------------------------------
def billed_usage(e: Any) -> Optional["Usage"]:
    """The spend an error carries (``ProviderError.usage``, also set on other exceptions by :func:`carry_usage`), or
    None when nothing was billed."""
    u = getattr(e, "usage", None)
    if isinstance(u, Usage) and (u.input_tokens or u.output_tokens or u.cost_usd):
        return u
    return None


def carry_usage(e: BaseException, spent: Optional["Usage"]) -> BaseException:
    """Fold ``spent`` — calls already billed for the same item before ``e`` happened (earlier tiles, the detector of a
    two-stage run, another committee member) — into ``e.usage``. Returns ``e`` so callers can ``raise``."""
    if spent is not None and (spent.input_tokens or spent.output_tokens or spent.cost_usd):
        prior = billed_usage(e)
        total = Usage().add(spent)
        if prior is not None:
            total.add(prior)
        try:
            e.usage = total            # type: ignore[attr-defined]
        except AttributeError:         # pragma: no cover — an exception type with __slots__
            pass
    return e
# --- end ai-cloud hook ------------------------------------------------------------------------------------------


@dataclass
class ClassSpec:
    """One dataset class as the model should see it."""
    id: int                      # dataset class id (the YOLO class index written to label files)
    name: str                    # dataset class name
    prompt: str = ""             # what to look for, e.g. "person lying on the floor"; empty -> name
    description: str = ""        # optional guideline text from the class card (include/exclude rules)

    @property
    def query(self) -> str:
        return (self.prompt or self.name).strip()


@dataclass
class DetectRequest:
    image_path: str              # absolute path of the original image on disk
    classes: List[ClassSpec]     # the classes to look for (subset of the dataset's classes)
    model: str                   # provider-specific model id
    conf: float = 0.25           # confidence floor (providers without calibrated confidence ignore it)
    options: Dict[str, Any] = field(default_factory=dict)
    # Common options (providers ignore what they do not support):
    #   effort: "low" | "medium" | "high"         (cloud reasoning effort; "Fast" = low, "Accurate" = high)
    #   max_edge: int                             (long-edge cap for the uploaded image; provider default otherwise)
    #   tiles: bool | int                         (SAHI-style tiled inference for small objects)
    #   imgsz: int, iou: float, device: str       (local ultralytics)
    #   class_map: {model_class_id: dataset_class_id}   (closed-vocabulary local models; unmapped -> skipped)
    #   prompt_mode: "text" | "exemplars"         (YOLOE: class-name prompts or visual prototypes from labels)
    #   max_objects: int                          (cap on returned objects; default 100)
    #   samples: int                              (cloud self-consistency samples -> vote-based confidence)
    #   infer_batch: int                          (jobs: images per local model call, 1-32, default 8)


@dataclass
class Detection:
    cls: int                     # dataset class id
    xc: float                    # normalised YOLO centre / size, clamped to [0, 1]
    yc: float
    w: float
    h: float
    conf: Optional[float] = None # None when the model gives no calibrated confidence
    label: str = ""              # the model's own label text (for provenance / debugging)

    def row(self) -> List[float]:
        r = [self.cls, self.xc, self.yc, self.w, self.h]
        return r + [self.conf] if self.conf is not None else r


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float = 0.0
    # Who billed what: {provider id: [input_tokens, output_tokens, cost_usd]}. A two-stage run or a committee audit
    # mixes providers in one total; spend caps and the usage ledger need each provider's own share. The part of
    # the totals not covered here belongs to the run's provider (see store.add_usage).
    by_provider: Dict[str, List[float]] = field(default_factory=dict, repr=False, compare=False)

    def add(self, other: "Usage") -> "Usage":
        self.input_tokens += other.input_tokens
        self.output_tokens += other.output_tokens
        self.cost_usd += other.cost_usd
        for pid, (tin, tout, usd) in (getattr(other, "by_provider", None) or {}).items():
            cur = self.by_provider.setdefault(pid, [0, 0, 0.0])
            cur[0] += int(tin)
            cur[1] += int(tout)
            cur[2] += float(usd)
        return self

    def tagged(self, provider: str) -> "Usage":
        """A copy whose whole amount is attributed to ``provider`` (unless it already carries attribution)."""
        u = Usage(int(self.input_tokens), int(self.output_tokens), float(self.cost_usd))
        parts = getattr(self, "by_provider", None) or {}
        if parts:
            u.by_provider = {k: list(v) for k, v in parts.items()}
        elif provider and (u.input_tokens or u.output_tokens or u.cost_usd):
            u.by_provider = {provider: [u.input_tokens, u.output_tokens, u.cost_usd]}
        return u


@dataclass
class DetectResult:
    detections: List[Detection]
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    warnings: List[str] = field(default_factory=list)   # e.g. "3 boxes with unknown labels dropped"
    served_by: str = ""                                 # model that actually answered (fallbacks)
    covered: Optional[List[int]] = None                 # dataset classes the model could actually find (None = all
                                                        # requested); the audit ignores ground truth of other classes

    def to_dict(self) -> dict:
        return {"detections": [asdict(d) for d in self.detections], "usage": asdict(self.usage),
                "latency_ms": self.latency_ms, "warnings": self.warnings, "served_by": self.served_by}


@dataclass
class Crop:
    """A region to classify (two-stage labeling / verifying existing labels)."""
    index: int                   # caller's index, echoed back
    xc: float                    # normalised box in the full image
    yc: float
    w: float
    h: float
    hint: str = ""               # optional: current label / detector label shown to nobody but kept for provenance


@dataclass
class ClassifyRequest:
    image_path: str
    crops: List[Crop]
    classes: List[ClassSpec]     # candidate classes; the model may also answer "none" / "uncertain"
    model: str
    options: Dict[str, Any] = field(default_factory=dict)   # effort, pad (context fraction, default 0.25), crop_edge


@dataclass
class Classification:
    index: int                   # Crop.index
    cls: Optional[int]           # dataset class id, or None for "none of these" / "uncertain"
    verdict: str = "label"       # label | none | uncertain
    confidence: Optional[float] = None
    reason: str = ""


@dataclass
class ClassifyResult:
    items: List[Classification]
    usage: Usage = field(default_factory=Usage)
    latency_ms: int = 0
    warnings: List[str] = field(default_factory=list)


class Provider:
    """Base class. Subclasses set the class attributes and override what they support.

    Threading: methods may be called concurrently from request threads and job threads. A provider that wraps a
    stateful model (ultralytics predictors) must serialise inference itself (a lock per loaded model)."""

    id: str = ""                 # registry key, e.g. "anthropic"
    name: str = ""               # UI name, e.g. "Anthropic Claude"
    kind: str = "local"          # local | cloud
    location: str = ""           # one line for the UI, e.g. "images are sent to Anthropic"
    needs_key: bool = False
    env_keys: tuple = ()         # environment variables that may hold the key, first match wins
    capabilities: Dict[str, bool] = {}   # detect, classify, refine, open_vocab, exemplars, generate, batch_api

    # ---- status / models
    def status(self) -> dict:
        """{available: bool, reason: str|None}. Must be cheap (no network): used to render the provider list."""
        return {"available": True, "reason": None}

    def models(self, refresh: bool = False) -> List[dict]:
        """[{id, name, notes, recommended: bool, price: {in, out, batch_in, batch_out} per MTok | None,
             max_edge: int|None, default: bool}]. May hit the network when refresh=True (key permitting)."""
        return []

    def test(self) -> dict:
        """Cheap connectivity / auth check: {ok: bool, message: str, latency_ms: int}."""
        return {"ok": True, "message": "ok", "latency_ms": 0}

    def estimate(self, width: int, height: int, model: str, n_classes: int, options: Dict[str, Any]) -> dict:
        """Per-image cost estimate: {usd: [lo, hi], tokens_in: [lo, hi], tokens_out: [lo, hi], seconds: float|None}.
        Local providers return usd [0, 0]."""
        return {"usd": [0.0, 0.0], "tokens_in": [0, 0], "tokens_out": [0, 0], "seconds": None}

    # ---- work
    def detect(self, req: DetectRequest) -> DetectResult:
        raise ProviderError(f"{self.name} cannot detect objects", "invalid")

    def detect_many(self, reqs: List[DetectRequest]) -> List[Union[DetectResult, ProviderError]]:
        """Several images at once where the backend batches inference (local ultralytics: one forward pass per chunk;
        capabilities["batch_infer"]). One entry per request, in order: its DetectResult or the ProviderError of that
        image. Default: one :meth:`detect` per request."""
        out: List[Union[DetectResult, ProviderError]] = []
        for r in reqs:
            try:
                out.append(self.detect(r))
            except ProviderError as e:
                out.append(e)
        return out

    def classify(self, req: ClassifyRequest) -> ClassifyResult:
        raise ProviderError(f"{self.name} cannot classify regions", "invalid")

    def refine(self, image_path: str, box: Optional[List[float]] = None, points: Optional[List[List[float]]] = None,
               model: str = "") -> dict:
        """Tighten a rough box / turn clicks into a box. ``box`` = normalised [xc, yc, w, h]; ``points`` =
        normalised [[x, y, 1|0], ...]. Returns {box, score, candidates: [{box, score}]} (normalised)."""
        raise ProviderError(f"{self.name} cannot refine boxes", "invalid")

    # ---- generic structured generation (capabilities["generate"])
    def generate_json(self, system: str, user: str, schema: dict, model: str, images: Optional[List[bytes]] = None,
                      options: Optional[Dict[str, Any]] = None) -> Tuple[dict, Usage]:
        """One structured answer: ``system`` + ``user`` text (+ optional encoded images, JPEG/PNG bytes, sent in
        order before the text) constrained to the JSON ``schema`` (an object schema; providers use their native
        structured-output mode and fall back to robust extraction). Returns ``(parsed_object, Usage)`` with cost.
        ``options``: effort (low|medium|high), max_tokens. Raises ProviderError (refusal / bad_output / auth …)."""
        raise ProviderError(f"{self.name} cannot generate text", "invalid")

    # ---- provider batch APIs (capabilities["batch_api"]): ~50% cheaper, results within 24 h
    def batch_submit(self, items: List[Dict[str, Any]]) -> str:
        """Submit detection requests as ONE provider batch and return its id (never a secret; stored in run params).
        ``items`` = ``[{"custom_id": str (the image id), "request": DetectRequest}, …]`` — every request uses the same
        model. Images are prepared (resized / encoded) here, exactly as the synchronous ``detect`` would."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def batch_status(self, batch_id: str) -> Dict[str, Any]:
        """``{state, counts, ...}``. ``state``: ``running`` (still processing) · ``canceling`` · ``ended`` (results
        ready — also after a cancel or expiry: finished items keep their results) · ``failed`` (no results at all).
        ``counts``: ``{processing, succeeded, errored, canceled, expired}``."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def batch_results(self, batch_id: str, items: Optional[List[Dict[str, Any]]] = None
                      ) -> Iterator[Tuple[str, Union[DetectResult, ProviderError]]]:
        """Yield ``(custom_id, DetectResult | ProviderError)`` once the batch has ended. ``items`` repeats the list
        given to ``batch_submit`` (rebuilt by the caller, also after a server restart) so coordinates can be decoded
        without provider-side state. ``DetectResult.usage.cost_usd`` must already be at the batch price."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")

    def batch_cancel(self, batch_id: str) -> Dict[str, Any]:
        """Ask the provider to stop the batch (best effort); returns the new :meth:`batch_status`."""
        raise ProviderError(f"{self.name} has no batch API", "invalid")
