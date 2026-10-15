"""Anthropic Claude through the official ``anthropic`` SDK (Messages API).

Claude reads boxes best as absolute pixel ``x1,y1,x2,y2`` on the image it actually sees, so the image is pre-resized
client-side to exactly the size Claude would resize it to (``resized_size`` for the model's resolution tier) and sent
with ``transformations.oversized_image = "error"`` as a tripwire. Output is constrained with
``output_config.format`` (json_schema, label enum); no sampling parameters are sent (Claude 4.7+ rejects them).
The SDK's own retries are off (``max_retries=0``): ``vlm.with_retries`` retries instead so a job can show the wait.
Economy runs use the Message Batches API (``client.messages.batches``: create / retrieve / results / cancel) with the
same request per image, at half price; ``generate_json`` sends structured-output requests with optional images.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import logging
import re
import threading
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .. import vlm
from ..base import ProviderError, Usage

WEB_SEARCH_USD = 0.01        # web search is billed at $10 per 1,000 searches, on top of the tokens
from ..vlm import CallResult, Prepared, Transient

log = logging.getLogger("fovea.ai")

FALLBACK_BETA = "server-side-fallback-2026-07-01"
TIERS = {"high": (2576, 4784), "standard": (1568, 1568)}


SDK_MISSING = "The 'anthropic' Python package is not installed (pip install anthropic)"


def _sdk_available() -> bool:
    try:
        return importlib.util.find_spec("anthropic") is not None
    except (ImportError, ValueError):
        return False


def _import_sdk():
    """``import anthropic``, or a user-facing ProviderError ('unavailable') instead of an HTTP 500."""
    try:
        import anthropic
    except ImportError as e:
        missing = (getattr(e, "name", None) or "anthropic").split(".")[0]
        if missing == "httpx":
            raise ProviderError(vlm.HTTPX_MISSING, "unavailable") from None
        if missing == "anthropic":
            raise ProviderError(SDK_MISSING, "unavailable") from None
        raise ProviderError(f"The 'anthropic' Python package could not be loaded: its dependency '{missing}' is "
                            "missing (pip install --force-reinstall anthropic)", "unavailable") from None
    return anthropic


class AnthropicProvider(vlm.VLMProvider):
    id = "anthropic"
    name = "Anthropic Claude"
    kind = "cloud"
    needs_key = True
    env_keys = ("ANTHROPIC_API_KEY",)
    capabilities = {"detect": True, "classify": True, "refine": False, "open_vocab": True, "exemplars": False,
                    "batch_api": True, "generate": True, "web_search": True}
    default_base_url = "https://api.anthropic.com"
    base_url_env = "FOVEA_ANTHROPIC_BASE_URL"
    schema_bounds = False          # Claude's structured outputs reject minimum/maximum/maxItems

    _clients: Dict[Tuple[str, str, float], Any] = {}
    _clients_lock = threading.Lock()
    _unsupported: Dict[str, set] = {}   # base_url -> request features a gateway rejected (fallbacks, transformations)

    # ---- availability
    def status(self) -> dict:
        if not _sdk_available():
            return {"available": False, "reason": SDK_MISSING}
        return super().status()

    # ---- model facts
    def tier(self, model: str) -> Tuple[int, int]:
        m = vlm.find_model(self.id, model)
        if m and m.get("tier") in TIERS:
            return TIERS[m["tier"]]
        # unknown / newer ids: the 4.7+ generation is high-res; 4.6 / 4.5 era is standard
        if re.search(r"-(4-[0-6])\b|-4-5|haiku-4|sonnet-4|opus-4-[0-6]\b", model or ""):
            return TIERS["standard"]
        return TIERS["high"]

    def frame_for(self, model: str, options: Dict[str, Any]) -> str:
        return vlm.PIXEL_XYXY

    def native_edge(self, model: str, options: Dict[str, Any]) -> int:
        return self.tier(model)[0]

    def sent_size(self, model: str, w: int, h: int, options: Dict[str, Any]) -> Tuple[int, int]:
        max_edge, max_tokens = self.tier(model)
        return vlm.anthropic_resized_size(w, h, max_edge, max_tokens)

    def image_tokens(self, model: str, sw: int, sh: int, options: Dict[str, Any]) -> int:
        return min(vlm.anthropic_image_tokens(sw, sh), self.tier(model)[1])

    def detect_schema_for(self, vocab, frame, max_objects, options) -> dict:
        # named integer fields: Claude's schemas cannot pin an array to length 4
        return vlm.detect_schema(vocab.labels, "xyxy", bounds=False)

    def default_effort(self, model: str) -> Optional[str]:
        m = vlm.find_model(self.id, model) or {}
        return "medium" if m.get("efforts") else None

    # ---- client
    def _client(self, key: str, timeout: float = 180.0):
        anthropic = _import_sdk()
        base = self.base_url()
        ck = (hashlib.sha256(key.encode()).hexdigest()[:16], base, timeout)
        with self._clients_lock:
            c = self._clients.get(ck)
            if c is None:
                if len(self._clients) >= 8:          # keys / base URLs changed: drop old pools
                    for old in list(self._clients.values()):
                        try:
                            old.close()
                        except Exception:  # noqa: BLE001
                            pass
                    self._clients.clear()
                # gzip/deflate only: httpx2 2.13 calls brotlicffi < 1.2 with an argument it does not take, so a
                # Brotli-encoded answer fails as "Connection error" on such installs. API answers are small JSON.
                c = anthropic.Anthropic(api_key=key, base_url=base, max_retries=0,
                                        timeout=anthropic.Timeout(timeout, connect=10.0),
                                        default_headers={"Accept-Encoding": "gzip, deflate"})
                self._clients[ck] = c
            return c

    def _unsup(self) -> set:
        return self._unsupported.setdefault(self.base_url(), set())

    def _output_config(self, model: str, schema: dict, options: Dict[str, Any]) -> dict:
        oc: Dict[str, Any] = {"format": {"type": "json_schema", "schema": schema}}
        if f"effort:{model}" in self._unsup():
            return oc
        m = vlm.find_model(self.id, model)
        effort = self.effort_for(options) or "medium"   # explicit: Opus 5.5 / Haiku 5.5 default medium, Sonnet high
        if m is None:
            oc["effort"] = effort                         # unknown (newer) model: trust the user's choice
        elif m.get("efforts"):
            oc["effort"] = vlm.pick_effort(effort, m["efforts"])
        return oc

    def _image_block(self, b64: str) -> dict:
        blk: Dict[str, Any] = {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": b64}}
        if "transformations" not in self._unsup():
            blk["transformations"] = {"oversized_image": "error"}
        return blk

    def _use_fallback(self, model: str, options: Dict[str, Any]) -> bool:
        if options.get("fallback") is False or self.config().get("fallback") is False:
            return False
        m = vlm.find_model(self.id, model)
        return bool(m and m.get("fallback")) and "fallbacks" not in self._unsup()

    def _create(self, model: str, params: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        return self._result(self._send(model, params, options, max_tokens), model)

    def _send(self, model: str, params: dict, options: Dict[str, Any], max_tokens: int) -> Any:
        """One Messages API request → the SDK message (errors mapped to ProviderError / Transient)."""
        anthropic = _import_sdk()
        key = self.require_key()
        timeout = 600.0 if max_tokens > self.max_tokens or (self.effort_for(options) in ("high", "xhigh", "max")) else 180.0
        client = self._client(key, timeout)
        params = {**params, "model": model, "max_tokens": max_tokens}
        fallback = self._use_fallback(model, options)
        try:
            if fallback:
                msg = client.beta.messages.create(**params, betas=[FALLBACK_BETA], fallbacks="default")
            else:
                msg = client.messages.create(**params)
        except anthropic.APIStatusError as e:
            body = e.body if isinstance(e.body, dict) else {}
            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            text = vlm.error_message(body, getattr(e, "message", "") or "")
            # a gateway / org without these features: drop them once and retry immediately
            if e.status_code == 400 and fallback and re.search(r"fallback|anthropic-beta", text, re.I):
                self._unsup().add("fallbacks")
                return self._send(model, params, {**options, "fallback": False}, max_tokens)
            effort_key = f"effort:{model}"
            if e.status_code == 400 and re.search(r"\beffort\b", text, re.I) and effort_key not in self._unsup() \
                    and "effort" in (params.get("output_config") or {}):
                self._unsup().add(effort_key)          # an older model without effort control
                params = {**params, "output_config": {k: v for k, v in params["output_config"].items() if k != "effort"}}
                if not params["output_config"]:
                    params.pop("output_config")
                return self._send(model, params, options, max_tokens)
            if e.status_code == 400 and "transformations" in text and "transformations" not in self._unsup():
                self._unsup().add("transformations")
                for msg_ in params.get("messages", []):
                    for blk in msg_.get("content", []):
                        if isinstance(blk, dict):
                            blk.pop("transformations", None)
                return self._send(model, params, options, max_tokens)
            details = err.get("details") if isinstance(err.get("details"), dict) else {}
            code = details.get("error_code") or err.get("error_code") or ""
            etype = err.get("type") or ""
            quota = (code == "enforced_spend_limit_reached" or etype == "billing_error"
                     or (e.status_code == 400 and re.search(r"credit balance|spend(ing)? limit|billing", text, re.I)))
            raise vlm.http_failure("Anthropic", e.status_code, body, _headers(e), text, quota=bool(quota),
                                   auth=etype == "authentication_error", secrets=[key])
        except anthropic.APITimeoutError:
            raise Transient("Anthropic request timed out", "network")
        except anthropic.APIConnectionError as e:
            raise Transient(f"Cannot reach {vlm.url_host(self.base_url())}: {vlm.scrub(e, key)}", "network")
        return msg

    def _result(self, msg: Any, model: str, batch: bool = False) -> CallResult:
        stop = getattr(msg, "stop_reason", None)
        served = getattr(msg, "model", None) or model
        u = getattr(msg, "usage", None)
        usage = Usage()
        if u is not None:
            plain = int(getattr(u, "input_tokens", 0) or 0)
            cread = int(getattr(u, "cache_read_input_tokens", 0) or 0)
            cwrite = int(getattr(u, "cache_creation_input_tokens", 0) or 0)
            out = int(getattr(u, "output_tokens", 0) or 0)
            usage.input_tokens = plain + cread + cwrite
            usage.output_tokens = out
            iterations = getattr(u, "iterations", None) or []
            hops = [it for it in iterations if getattr(it, "type", "") in ("message", "fallback_message")]
            if hops:
                # a fallback ran: every hop bills at its own model's rates
                usage.cost_usd = sum(vlm.cost_usd(self.id, getattr(it, "model", None) or model,
                                                  int(getattr(it, "input_tokens", 0) or 0)
                                                  + int(getattr(it, "cache_read_input_tokens", 0) or 0)
                                                  + int(getattr(it, "cache_creation_input_tokens", 0) or 0),
                                                  int(getattr(it, "output_tokens", 0) or 0),
                                                  cached_tokens=int(getattr(it, "cache_read_input_tokens", 0) or 0),
                                                  cache_write_tokens=int(getattr(it, "cache_creation_input_tokens", 0)
                                                                         or 0), batch=batch)
                                     for it in hops)
            else:
                usage.cost_usd = vlm.cost_usd(self.id, served if vlm.find_model(self.id, served) else model,
                                              usage.input_tokens, out, cached_tokens=cread,
                                              cache_write_tokens=cwrite, batch=batch)
        if stop == "refusal":
            det = getattr(msg, "stop_details", None)
            cat = getattr(det, "category", None) if det is not None else None
            return CallResult("", usage, served, "refusal", f"safety category: {cat}" if cat else "")
        text = ""
        for blk in getattr(msg, "content", None) or []:
            if getattr(blk, "type", None) == "text":
                text = getattr(blk, "text", "") or ""
                break
        return CallResult(text, usage, served, "truncated" if stop == "max_tokens" else "stop")

    # ---- calls
    def call_detect(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                    options: Dict[str, Any], max_tokens: int) -> CallResult:
        img = self._image_block(prep.b64)
        try:
            repeats = int(options.get("samples") or 1) > 1
        except (TypeError, ValueError):
            repeats = False
        if repeats:
            # the same image goes out once per sample: cache it so repeats read it at ~0.1x the input price
            img["cache_control"] = {"type": "ephemeral"}
        params = {"system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  "messages": [{"role": "user", "content": [img, {"type": "text", "text": user_text}]}],
                  "output_config": self._output_config(model, schema, options)}
        return self._create(model, params, options, max_tokens)

    def call_classify(self, model: str, images: List[Tuple[int, bytes, Tuple[int, int]]], system: str,
                      user_text: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        content: List[dict] = []
        for idx, data, _size in images:
            content.append({"type": "text", "text": f"Crop {idx}:"})
            content.append(self._image_block(base64.b64encode(data).decode("ascii")))
        content.append({"type": "text", "text": user_text})
        params = {"system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                  "messages": [{"role": "user", "content": content}],
                  "output_config": self._output_config(model, schema, options)}
        return self._create(model, params, options, max_tokens)

    def _web_generate(self, model: str, system: str, user_text: str, schema: dict, options: Dict[str, Any],
                      max_tokens: int) -> CallResult:
        """A JSON answer researched with the server-side web search tool. Search answers carry citations, which
        structured outputs do not accept, so the schema goes into the prompt and the JSON is read from the text. A
        turn paused by the server's tool loop (pause_turn) is resumed by sending it back as it is."""
        dynamic = bool(re.search(r"claude-(opus-(5|4-[678])|sonnet-(5|4-6)|fable|mythos)", model))
        try:
            uses = max(1, min(int(options.get("max_searches") or 5), 10))
        except (TypeError, ValueError):
            uses = 5
        tool = {"type": "web_search_20260209" if dynamic else "web_search_20250305", "name": "web_search",
                "max_uses": uses}
        oc = {k: v for k, v in self._output_config(model, {}, options).items() if k != "format"}
        prompt = (user_text + "\n\nWhen you are done, answer with one JSON object that follows this JSON schema, and "
                  "nothing after it:\n" + json.dumps(schema, ensure_ascii=False))
        first = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
        params: Dict[str, Any] = {"system": [{"type": "text", "text": system or "You answer with JSON."}],
                                  "messages": first, "tools": [tool]}
        if oc:
            params["output_config"] = oc
        total, searches, res, msg = Usage(), 0, None, None
        for _ in range(4):
            msg = self._send(model, params, options, max_tokens)
            res = self._result(msg, model)
            total.add(res.usage)
            stu = getattr(getattr(msg, "usage", None), "server_tool_use", None)
            searches += int(getattr(stu, "web_search_requests", 0) or 0)
            if getattr(msg, "stop_reason", None) != "pause_turn":
                break
            params = {**params, "messages": first + [{"role": "assistant", "content": msg.content}]}
        total.cost_usd += searches * WEB_SEARCH_USD
        text = "".join(getattr(b, "text", "") or "" for b in (getattr(msg, "content", None) or [])
                       if getattr(b, "type", None) == "text")
        if res.finish == "refusal":
            return CallResult("", total, res.served_by, "refusal", res.refusal)
        return CallResult(text, total, res.served_by, res.finish)

    def call_generate(self, model: str, system: str, user_text: str, schema: dict, images: List[Tuple[bytes, str]],
                      options: Dict[str, Any], max_tokens: int) -> CallResult:
        if options.get("web_search") and not images:
            return self._web_generate(model, system, user_text, schema, options, max_tokens)
        import base64
        content: List[dict] = []
        for k, (data, mime) in enumerate(images):
            if len(images) > 1:
                content.append({"type": "text", "text": f"Image {k + 1}:"})
            content.append({"type": "image", "source": {"type": "base64", "media_type": mime,
                                                        "data": base64.b64encode(data).decode("ascii")}})
        content.append({"type": "text", "text": user_text})
        params = {"system": [{"type": "text", "text": system or "You answer with JSON only.",
                              "cache_control": {"type": "ephemeral"}}],
                  "messages": [{"role": "user", "content": content}],
                  "output_config": self._output_config(model, vlm.schema_for("anthropic", schema), options)}
        return self._create(model, params, options, max_tokens)

    # ---- Message Batches (official SDK: client.messages.batches)
    def batch_request(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                      options: Dict[str, Any], max_tokens: int) -> dict:
        # same request as the synchronous path minus the beta server-side fallback (not offered for batches)
        return {"model": model, "max_tokens": int(max_tokens),
                "system": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
                "messages": [{"role": "user", "content": [self._image_block(prep.b64),
                                                          {"type": "text", "text": user_text}]}],
                "output_config": self._output_config(model, schema, options)}

    def _sdk(self, fn, key: str):
        """Run one SDK call with Fovea's error mapping (Transient → retried by the caller; ProviderError → abort)."""
        anthropic = _import_sdk()
        try:
            return fn()
        except anthropic.APIStatusError as e:
            body = e.body if isinstance(e.body, dict) else {}
            err = body.get("error") if isinstance(body.get("error"), dict) else {}
            text = vlm.error_message(body, getattr(e, "message", "") or "")
            details = err.get("details") if isinstance(err.get("details"), dict) else {}
            code = details.get("error_code") or err.get("error_code") or ""
            etype = err.get("type") or ""
            quota = (code == "enforced_spend_limit_reached" or etype == "billing_error"
                     or (e.status_code == 400 and re.search(r"credit balance|spend(ing)? limit|billing", text, re.I)))
            raise vlm.http_failure("Anthropic", e.status_code, body, _headers(e), text, quota=bool(quota),
                                   auth=etype == "authentication_error", secrets=[key])
        except anthropic.APITimeoutError:
            raise Transient("Anthropic request timed out", "network")
        except anthropic.APIConnectionError as e:
            raise Transient(f"Cannot reach {vlm.url_host(self.base_url())}: {vlm.scrub(e, key)}", "network")
        except anthropic.AnthropicError as e:
            raise ProviderError(f"Anthropic: {vlm.scrub(e, key)}", "invalid")

    def _batch_create(self, model: str, reqs: List[Tuple[str, dict]]) -> str:
        key = self.require_key()
        client = self._client(key, 600.0)
        batch = self._sdk(lambda: client.messages.batches.create(
            requests=[{"custom_id": sid, "params": body} for sid, body in reqs]), key)
        return str(batch.id)

    def _batch_get(self, part: str) -> dict:
        key = self.require_key()
        b = self._sdk(lambda: self._client(key, 60.0).messages.batches.retrieve(part), key)
        rc = getattr(b, "request_counts", None)
        counts = {k: int(getattr(rc, k, 0) or 0) for k in ("processing", "succeeded", "errored", "canceled", "expired")}
        counts["total"] = sum(counts.values())
        status = getattr(b, "processing_status", "in_progress")
        state = {"in_progress": "running", "canceling": "canceling", "ended": "ended"}.get(status, "running")
        return {"state": state, "counts": counts}

    def _batch_fetch(self, part: str, ctx: dict) -> Iterator[Tuple[str, Any]]:
        key = self.require_key()
        client = self._client(key, 600.0)
        models = {info["model"] for info in ctx.get("images", {}).values()}
        model = next(iter(models)) if len(models) == 1 else ""
        results = self._sdk(lambda: client.messages.batches.results(part), key)
        it = iter(results)
        while True:
            try:
                r = self._sdk(lambda: next(it), key)
            except StopIteration:
                return
            res = getattr(r, "result", None)
            rtype = getattr(res, "type", "")
            sid = str(getattr(r, "custom_id", ""))
            if rtype == "succeeded":
                yield sid, self._result(res.message, model or getattr(res.message, "model", ""), batch=True)
            elif rtype == "errored":
                er = getattr(getattr(res, "error", None), "error", None)
                etype = str(getattr(er, "type", "") or "")
                msg = vlm.scrub(getattr(er, "message", "") or etype or "request failed", key)[:300]
                code = {"authentication_error": "auth", "permission_error": "auth", "billing_error": "quota",
                        "rate_limit_error": "rate", "overloaded_error": "overloaded", "api_error": "overloaded",
                        "timeout_error": "overloaded", "not_found_error": "invalid",
                        "request_too_large": "invalid"}.get(etype, "invalid")
                yield sid, ProviderError(f"Anthropic batch request failed ({etype or 'error'}): {msg}", code)
            elif rtype == "canceled":
                yield sid, ProviderError("The batch was canceled before this image was processed", "canceled")
            else:
                yield sid, ProviderError("This request expired before Anthropic processed it (24 h limit)", "expired")

    def _batch_cancel(self, part: str) -> None:
        key = self.require_key()
        self._sdk(lambda: self._client(key, 60.0).messages.batches.cancel(part), key)

    # ---- models / test
    def list_remote_models(self) -> List[dict]:
        anthropic = _import_sdk()
        key = self.require_key()
        try:
            out = []
            for m in self._client(key).models.list(limit=100):
                if getattr(m, "lifecycle", "active") == "retired":
                    continue
                caps = getattr(m, "capabilities", None)
                img = getattr(caps, "image_input", None) if caps is not None else None
                if img is not None and getattr(img, "supported", True) is False:
                    continue
                out.append({"id": m.id, "name": getattr(m, "display_name", None) or m.id})
            return out
        except anthropic.APIStatusError as e:
            raise vlm.http_failure("Anthropic", e.status_code, e.body, _headers(e), secrets=[key])
        except anthropic.APIConnectionError as e:
            raise ProviderError(f"Cannot reach {vlm.url_host(self.base_url())}: {vlm.scrub(e, key)}", "network")

    def test(self) -> dict:
        if not _sdk_available():
            return {"ok": False, "message": SDK_MISSING, "latency_ms": 0}
        return super().test()


def _headers(e: Any):
    resp = getattr(e, "response", None)
    return resp.headers if resp is not None else None


PROVIDER = AnthropicProvider()
