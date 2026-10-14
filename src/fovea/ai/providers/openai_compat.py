"""Any OpenAI-compatible ``/v1/chat/completions`` server: Ollama, LM Studio, vLLM, llama.cpp, hosted gateways.

Open VLMs disagree about box coordinates, so the frame is a setting (``coord``):

* ``qwen3``  — Qwen3-VL / 3.5 / 3.6 / 3.8: ``bbox_2d = [x1, y1, x2, y2]`` relative 0-1000 (the default);
* ``qwen25`` — Qwen2.5-VL: absolute pixels on the processor's resized image. We pre-resize to the processor's own
  ``smart_resize`` grid (multiples of 28) so the server does not resize again and pixels map 1:1;
* ``gemini`` — Gemini-style ``box_2d = [ymin, xmin, ymax, xmax]`` 0-1000 (Gemma and other Gemini-trained models);
* ``norm``   — ``[x1, y1, x2, y2]`` as 0-1 fractions;
* ``auto``   — pick from the model name.

``response_format`` json_schema is used when the server supports it; servers that reject it fall back to prompt-only
JSON with robust extraction (fences / prose are fine). The scale guard catches models that drift between frames.

``status()`` reflects a cached (60 s) reachability probe of base_url (``/v1/models``, then Ollama's ``/api/tags``) run
in a background thread, so the provider list never calls a stopped server "ready"; ``models()`` marks which listed
models the server actually has (``available: false`` + ``reason`` such as "Not pulled on localhost:11434 — run: …").
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urlparse

from .. import vlm
from ..base import ProviderError, Usage
from ..vlm import CallResult, Prepared

log = logging.getLogger("fovea.ai")
PROBE_TTL = 60.0                 # seconds a reachability result stays fresh
PROBE_FIRST_WAIT = 1.0           # the very first status() waits this long for the first probe (loopback answers in ms)

COORDS = {
    "qwen3": (vlm.XYXY_1000, "bbox_2d"),
    "qwen25": (vlm.PIXEL_XYXY, "bbox_2d"),
    "gemini": (vlm.YXYX_1000, "box_2d"),
    "norm": (vlm.NORM_XYXY, "bbox"),
}
QWEN25_MAX_PIXELS = 1280 * 28 * 28


def coord_from_model(model: str) -> str:
    m = (model or "").lower()
    if re.search(r"qwen2\.?5|qwen25|qwen2_5", m):
        return "qwen25"
    if "gemma" in m or "gemini" in m:
        return "gemini"
    return "qwen3"


class OpenAICompatProvider(vlm.VLMProvider):
    id = "openai_compat"
    name = "OpenAI-compatible server"
    needs_key = False
    env_keys = ("OPENAI_COMPAT_API_KEY",)
    capabilities = {"detect": True, "classify": True, "refine": False, "open_vocab": True, "exemplars": False,
                    "batch_api": False, "generate": True}
    default_base_url = "http://localhost:11434/v1"
    base_url_env = "FOVEA_OPENAI_COMPAT_BASE_URL"
    schema_bounds = False          # grammar backends differ; clamp in code instead
    track_cost = False             # local / self-hosted: no price table
    _unsupported: Dict[str, set] = {}

    @property
    def kind(self) -> str:  # type: ignore[override]
        return "local" if vlm.is_loopback_url(self.base_url()) else "cloud"

    @property
    def location(self) -> str:  # type: ignore[override]
        host = vlm.url_host(self.base_url())
        return f"Images stay on this machine ({host})" if self.kind == "local" else f"Images are sent to {host}"

    # ---- URLs
    def base_url(self) -> str:
        url = super().base_url()
        url = re.sub(r"/chat/completions/?$", "", url)
        p = urlparse(url)
        if not re.search(r"/v\d+[a-z]*(/|$)", p.path or ""):
            url = url.rstrip("/") + "/v1"
        return url.rstrip("/")

    def root_url(self) -> str:
        p = urlparse(self.base_url())
        return f"{p.scheme}://{p.netloc}"

    # ---- frames / sizes
    def coord(self, model: str, options: Dict[str, Any]) -> str:
        c = str((options or {}).get("coord") or self.config().get("coord") or "auto").lower()
        if c in COORDS:
            return c
        m = vlm.find_model(self.id, model)
        if m and m.get("coord") in COORDS:
            return m["coord"]
        return coord_from_model(model)

    def frame_for(self, model: str, options: Dict[str, Any]) -> str:
        return COORDS[self.coord(model, options)][0]

    def native_edge(self, model: str, options: Dict[str, Any]) -> int:
        try:
            return min(max(int((options or {}).get("max_edge") or self.config().get("max_edge") or 1280), 256), 4096)
        except (TypeError, ValueError):
            return 1280

    def sent_size(self, model: str, w: int, h: int, options: Dict[str, Any]) -> Tuple[int, int]:
        edge = self.native_edge(model, options)
        if self.coord(model, options) == "qwen25":
            return vlm.smart_resize(w, h, 28, 56 * 56, min(QWEN25_MAX_PIXELS, edge * edge))
        return vlm.fit_long_edge(w, h, edge)

    def image_tokens(self, model: str, sw: int, sh: int, options: Dict[str, Any]) -> int:
        p = 28 if self.coord(model, options) == "qwen25" else 32
        return max(1, (sw // p) * (sh // p))

    def default_effort(self, model: str) -> Optional[str]:
        return None

    def detect_schema_for(self, vocab, frame, max_objects, options) -> dict:
        key = COORDS[self.coord_for_frame(frame)][1]
        return vlm.detect_schema(vocab.labels, "array", bounds=False, box_key=key, number=frame == vlm.NORM_XYXY)

    @staticmethod
    def coord_for_frame(frame: str) -> str:
        for k, (f, _key) in COORDS.items():
            if f == frame:
                return k
        return "qwen3"

    # ---- HTTP
    def _headers(self) -> dict:
        h = {"content-type": "application/json"}
        k = self.key()
        if k:
            h["Authorization"] = f"Bearer {k}"
        return h

    def _post(self, model: str, body: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        unsup = self._unsupported.setdefault(self.base_url(), set())
        for feat in unsup:
            body.pop(feat, None)
        k = self.key()
        try:
            data, _h = vlm.http_json("POST", f"{self.base_url()}/chat/completions", self.name, headers=self._headers(),
                                     body=body, timeout=600.0, secrets=[k] if k else [])
        except ProviderError as e:
            if e.code == "network":
                self._note_reachable(False, str(e))
            # Servers without structured output / reasoning_effort: drop the feature once and retry.
            if e.status in (400, 404, 422, 500, 501):
                feat = self._rejected_feature(body, unsup, str(e))
                if feat:
                    unsup.add(feat)
                    return self._post(model, body, options, max_tokens)
            raise
        except vlm.Transient as e:
            if e.status in (500, 503) and "response_format" in body and "response_format" not in unsup and \
                    re.search(r"json_schema|response_format|grammar", str(e), re.I):
                unsup.add("response_format")
                return self._post(model, body, options, max_tokens)
            raise
        return self._result(data, model)

    @staticmethod
    def _rejected_feature(body: dict, unsup: set, msg: str) -> Optional[str]:
        """Which optional request feature an error complains about (specific mention first, then a generic
        "unsupported" reply blames structured output, the feature servers most often lack)."""
        for feat, rx in (("reasoning_effort", r"reasoning[_ ]?effort|reasoning"),
                         ("response_format", r"response[_ ]?format|json[_ ]?schema|structured|grammar|guided")):
            if feat in body and feat not in unsup and re.search(rx, msg, re.I):
                return feat
        if "response_format" in body and "response_format" not in unsup and \
                re.search(r"unsupported|not supported|unknown (field|param)|extra (fields|inputs)|unrecognized", msg,
                          re.I):
            return "response_format"
        return None

    def _result(self, data: dict, model: str) -> CallResult:
        u = data.get("usage") or {}
        tin = int(u.get("prompt_tokens") or 0)
        tout = int(u.get("completion_tokens") or 0)
        served = data.get("model") or model
        usage = Usage(tin, tout, vlm.cost_usd(self.id, served, tin, tout))
        choices = data.get("choices") or []
        if not choices:
            return CallResult("", usage, served, "stop")
        ch = choices[0]
        msg = ch.get("message") or {}
        content = msg.get("content")
        if isinstance(content, list):
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict))
        text = content or ""
        fr = ch.get("finish_reason")
        if msg.get("refusal") and not text.strip():
            return CallResult("", usage, served, "refusal", vlm.scrub(msg.get("refusal"))[:160])
        if fr == "content_filter":
            return CallResult(text, usage, served, "refusal", "content filter")
        return CallResult(text, usage, served, "truncated" if fr == "length" else "stop")

    def _body(self, model: str, system: str, content: List[dict], schema: dict, name: str, options: Dict[str, Any],
              max_tokens: int) -> dict:
        body: Dict[str, Any] = {"model": model, "stream": False, "max_tokens": int(max_tokens),
                                "messages": [{"role": "system", "content": system},
                                             {"role": "user", "content": content}],
                                "response_format": {"type": "json_schema",
                                                    "json_schema": {"name": name, "strict": True, "schema": schema}}}
        effort = self.effort_for(options)
        if effort:
            body["reasoning_effort"] = {"minimal": "low", "none": "low", "xhigh": "high", "max": "high"}.get(effort,
                                                                                                              effort)
        cfg = self.config()
        if cfg.get("temperature") is not None:
            try:
                body["temperature"] = float(cfg["temperature"])
            except (TypeError, ValueError):
                pass
        return body

    def call_detect(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                    options: Dict[str, Any], max_tokens: int) -> CallResult:
        content = [{"type": "image_url", "image_url": {"url": prep.data_url()}},
                   {"type": "text", "text": user_text + " Reply with the JSON only."}]
        return self._post(model, self._body(model, system, content, schema, "detections", options, max_tokens),
                          options, max_tokens)

    def call_classify(self, model: str, images: List[Tuple[int, bytes, Tuple[int, int]]], system: str,
                      user_text: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        content: List[dict] = []
        for idx, data, _size in images:
            content.append({"type": "text", "text": f"Crop {idx}:"})
            content.append({"type": "image_url",
                            "image_url": {"url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")}})
        content.append({"type": "text", "text": user_text + ' Reply with the JSON only: {"items": [{"index": 1, '
                                                            '"label": "<class>", "confidence": "high", "reason": "..."}, '
                                                            '...]}.'})
        return self._post(model, self._body(model, system, content, schema, "classifications", options, max_tokens),
                          options, max_tokens)

    def call_generate(self, model: str, system: str, user_text: str, schema: dict, images: List[Tuple[bytes, str]],
                      options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        content: List[dict] = []
        for k, (data, mime) in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image {k + 1}:"})
            content.append({"type": "image_url",
                            "image_url": {"url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")}})
        # servers without structured output only see the prompt: always state the exact shape
        wire = vlm.schema_for("compat", schema, strict=True)
        content.append({"type": "text", "text": user_text + "\nReply with the JSON only, matching this JSON schema: "
                        + json.dumps(wire, ensure_ascii=False, separators=(",", ":"))})
        body = self._body(model, system or "Answer with JSON only.", content, wire, "result", options, max_tokens)
        return self._post(model, body, options, max_tokens)

    # ---- reachability (N4): status() must not say "ready" for a server that is not running
    _probes: Dict[str, dict] = {}
    _probe_lock = threading.Lock()

    def _probe_key(self) -> str:
        k = self.key() or ""
        return f"{self.base_url()}|{hashlib.sha256(k.encode()).hexdigest()[:12] if k else ''}"

    def _probe_now(self) -> dict:
        """GET /models (then Ollama's /api/tags): reachable or not, and which models the server has."""
        base, host = self.base_url(), vlm.url_host(self.base_url())
        k = self.key()
        st: Dict[str, Any] = {"t": time.time(), "ok": False, "reason": f"Not reachable at {host}", "models": None,
                              "ollama": False}
        try:
            httpx = vlm.require_httpx()
        except ProviderError as e:
            st["reason"] = str(e)
            return st
        try:
            with httpx.Client(timeout=httpx.Timeout(connect=2.0, read=4.0, write=4.0, pool=2.0),
                              follow_redirects=False) as c:
                r = c.get(f"{base}/models", headers=self._headers())
                if r.status_code in (401, 403):
                    st.update(ok=False, reason=f"{host} rejected the API key ({r.status_code})")
                    return st
                ids: Optional[List[str]] = None
                if r.status_code < 400:
                    try:
                        ids = [str(m.get("id")) for m in (r.json().get("data") or []) if m.get("id")]
                    except (ValueError, AttributeError):
                        ids = None
                reachable = r.status_code < 500
                try:
                    t = c.get(f"{self.root_url()}/api/tags", headers=self._headers())
                    if t.status_code < 400:
                        names = [str(m.get("name") or m.get("model")) for m in (t.json().get("models") or [])]
                        st["ollama"] = True
                        ids = sorted(set(ids or []) | set(names))
                        reachable = True
                except (httpx.HTTPError, ValueError, AttributeError):
                    pass
                if reachable:
                    st.update(ok=True, reason=None, models=ids)
                else:
                    st["reason"] = f"{host} answered {r.status_code}"
        except httpx.HTTPError:
            pass
        except Exception as e:  # noqa: BLE001 — a probe must never break the provider list
            st["reason"] = vlm.scrub(f"Cannot check {host}: {type(e).__name__}", k)
        return st

    _probe_threads: Dict[str, threading.Thread] = {}

    def probe(self, wait: float = 0.0) -> Optional[dict]:
        """Cached (60 s) reachability of base_url, refreshed in a background thread. ``wait`` bounds how long the
        caller waits for a probe that is still running when there is no result yet (the very first check)."""
        try:
            vlm.require_httpx()     # a cold import takes ~0.3 s: pay it here, not inside the bounded wait
        except ProviderError as e:                     # core-only install: nothing to probe with — say why
            return {"t": time.time(), "ok": False, "reason": str(e), "models": None, "ollama": False}
        key = self._probe_key()
        with self._probe_lock:
            st = self._probes.get(key)
            fresh = st is not None and "ok" in st and time.time() - st["t"] < PROBE_TTL
            th = self._probe_threads.get(key)
            if not fresh and (th is None or not th.is_alive()):
                def run(key=key):
                    res = self._probe_now()
                    with self._probe_lock:
                        self._probes[key] = res

                th = threading.Thread(target=run, name="fovea-compat-probe", daemon=True)
                self._probe_threads[key] = th
                th.start()
        if wait > 0 and not (st and "ok" in st) and th is not None and th.is_alive():
            th.join(wait)
        with self._probe_lock:
            st = self._probes.get(key)
        return st if st and "ok" in st else None

    def _note_reachable(self, ok: bool, why: str = "") -> None:
        """A real request just told us: refresh the cached state without another probe."""
        key = self._probe_key()
        host = vlm.url_host(self.base_url())
        with self._probe_lock:
            st = dict(self._probes.get(key) or {})
            if ok:
                st.update(t=time.time(), ok=True, reason=None)
                st.setdefault("models", None)
            else:
                st.update(t=time.time(), ok=False, reason=f"Not reachable at {host}", models=None)
            self._probes[key] = st

    def status(self) -> dict:
        if not vlm.httpx_available():
            return {"available": False, "reason": vlm.HTTPX_MISSING}
        st = self.probe(wait=PROBE_FIRST_WAIT)
        host = vlm.url_host(self.base_url())
        if st is None:
            return {"available": False, "reason": f"Checking {host}…"}
        if not st.get("ok"):
            return {"available": False, "reason": st.get("reason") or f"Not reachable at {host}"}
        return {"available": True, "reason": None}

    def test(self) -> dict:
        out = super().test()
        self._note_reachable(bool(out.get("ok")))
        if not out.get("ok"):
            with self._probe_lock:                     # keep the specific reason (e.g. a rejected key)
                self._probes.pop(self._probe_key(), None)
        return out

    # ---- discovery
    def list_remote_models(self) -> List[dict]:
        k = self.key()
        secrets = [k] if k else []
        ids: List[str] = []
        err: Optional[Exception] = None
        try:
            data, _h = vlm.http_json("GET", f"{self.base_url()}/models", self.name, headers=self._headers(),
                                     timeout=15.0, secrets=secrets)
            ids = [str(m.get("id")) for m in (data.get("data") or []) if m.get("id")]
        except (ProviderError, vlm.Transient) as e:
            err = e
        # Ollama: /api/tags lists models and /api/show tells which ones can see images
        caps: Dict[str, List[str]] = {}
        try:
            tags, _h = vlm.http_json("GET", f"{self.root_url()}/api/tags", self.name, headers=self._headers(),
                                     timeout=10.0, secrets=secrets)
            names = [str(m.get("name") or m.get("model")) for m in (tags.get("models") or [])]
            ids = ids or names
            for n in names[:40]:
                try:
                    show, _h = vlm.http_json("POST", f"{self.root_url()}/api/show", self.name,
                                             headers=self._headers(), body={"model": n}, timeout=10.0, secrets=secrets)
                    if isinstance(show.get("capabilities"), list):
                        caps[n] = [str(c) for c in show["capabilities"]]
                except (ProviderError, vlm.Transient):
                    continue
        except (ProviderError, vlm.Transient):
            if err is not None:
                raise err
        out = []
        for mid in ids:
            c = caps.get(mid)
            if c is not None and "vision" not in c:
                continue          # a text-only model cannot label images
            e = {"id": mid, "name": mid}
            if c is not None:
                e["notes"] = "Vision model reported by the server."
            out.append(e)
        return out

    def models(self, refresh: bool = False) -> List[dict]:
        out = super().models(refresh)
        st = self.probe(wait=PROBE_FIRST_WAIT)      # cached; waits only for the very first check
        host = vlm.url_host(self.base_url())
        served = (st or {}).get("models")
        for e in out:
            e["coord"] = self.coord(e["id"], {})
            e["price"] = None
            if str(e.get("notes", "")).startswith("Not in Fovea's price table"):
                e["notes"] = "" if self.kind == "local" else "Cost is not tracked for this server."
            if st is not None and not st.get("ok"):
                e["available"] = False
                e["reason"] = st.get("reason") or f"Not reachable at {host}"
            elif served is not None:
                have = any(vlm.is_variant(i, e["id"]) or vlm.is_variant(e["id"], i) for i in served)
                e["available"] = have
                if have:
                    e.pop("reason", None)
                else:
                    e["reason"] = (f"Not pulled on {host} — run: ollama pull {e['id']}" if st.get("ollama")
                                   else f"Not served by {host}")
            elif e.get("available") is False and "reason" not in e:
                e["reason"] = f"Not listed by {host}"
        return out


PROVIDER = OpenAICompatProvider()
