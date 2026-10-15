"""Google Gemini through the REST ``generateContent`` endpoint (httpx).

Gemini is trained to answer ``box_2d = [ymin, xmin, ymax, xmax]`` normalised to 0-1000 (y first). That frame is
scale-free, so the image only needs to be big enough for the chosen media resolution. The key travels in the
``x-goog-api-key`` header — never as ``?key=`` in a URL. No sampling parameters are sent (Gemini 3.6+ ignores them and
future models reject them); reasoning depth is ``thinkingLevel`` on Gemini 3 and ``thinkingBudget`` on 2.5.
Economy runs use ``batchGenerateContent`` with inline requests (a job over the 20 MB inline limit becomes several
batches under one Fovea id); results are read from ``inlinedResponses`` or a ``responsesFile``.
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple
from urllib.parse import quote

from .. import vlm
from ..base import ProviderError, Usage
from ..vlm import CallResult, Prepared

MEDIA_EDGE = {"low": 768, "medium": 1024, "high": 1536, "ultra_high": 3072}
MEDIA_TOKENS = {"low": 280, "medium": 560, "high": 1120, "ultra_high": 2240}
REFUSAL_FINISH = {"SAFETY", "RECITATION", "PROHIBITED_CONTENT", "IMAGE_SAFETY", "BLOCKLIST", "SPII",
                  "IMAGE_PROHIBITED_CONTENT", "IMAGE_RECITATION", "LANGUAGE"}


class GeminiProvider(vlm.VLMProvider):
    id = "gemini"
    name = "Google Gemini"
    kind = "cloud"
    needs_key = True
    env_keys = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    capabilities = {"detect": True, "classify": True, "refine": False, "open_vocab": True, "exemplars": False,
                    "batch_api": True, "generate": True, "web_search": True}
    default_base_url = "https://generativelanguage.googleapis.com"
    batch_max_bytes = 18 * 1024 * 1024     # inline batch requests are capped at 20 MB: larger jobs become several
    batch_max_requests = 5000
    base_url_env = "FOVEA_GEMINI_BASE_URL"
    schema_bounds = True
    _unsupported: Dict[str, set] = {}

    def generation(self, model: str) -> int:
        m = vlm.find_model(self.id, model)
        if m and m.get("generation"):
            return int(m["generation"])
        g = re.match(r"gemini-(\d+)", vlm._norm_model_id(model))
        return int(g.group(1)) if g else 3

    def media_resolution(self, options: Dict[str, Any]) -> str:
        r = str(options.get("media_resolution") or self.config().get("media_resolution") or "high").lower()
        return r if r in MEDIA_EDGE else "high"

    def frame_for(self, model: str, options: Dict[str, Any]) -> str:
        return vlm.YXYX_1000

    def native_edge(self, model: str, options: Dict[str, Any]) -> int:
        return MEDIA_EDGE[self.media_resolution(options or {})]

    def sent_size(self, model: str, w: int, h: int, options: Dict[str, Any]) -> Tuple[int, int]:
        return vlm.fit_long_edge(w, h, self.native_edge(model, options))

    def image_tokens(self, model: str, sw: int, sh: int, options: Dict[str, Any]) -> int:
        if self.generation(model) >= 3:
            return MEDIA_TOKENS[self.media_resolution(options or {})]
        if sw <= 384 and sh <= 384:
            return 258
        return 258 * math.ceil(sw / 768) * math.ceil(sh / 768)

    def default_effort(self, model: str) -> Optional[str]:
        return "medium" if self.generation(model) >= 3 else None

    def _thinking(self, model: str, options: Dict[str, Any]) -> Optional[dict]:
        effort = self.effort_for(options)
        if not effort:
            return None
        if self.generation(model) >= 3:
            m = vlm.find_model(self.id, model) or {}
            levels = m.get("thinking_levels") or ["low", "medium", "high"]
            return {"thinkingLevel": vlm.pick_effort(effort, levels).upper()}
        # Gemini 2.5: a token budget (0 disables thinking on Flash / Flash-Lite; Pro cannot go below 128)
        pro = "pro" in model
        budget = {"none": 0, "minimal": 0, "low": 0, "medium": 2048, "high": 8192, "xhigh": 16384,
                  "max": 24576}.get(effort, 2048)
        return {"thinkingBudget": max(budget, 128) if pro else budget}

    def _url(self, model: str) -> str:
        return f"{self.base_url()}/v1beta/models/{quote(vlm._norm_model_id(model), safe='.-_')}:generateContent"

    def _headers(self, key: str) -> dict:
        return {"x-goog-api-key": key, "content-type": "application/json"}

    @staticmethod
    def _classify_error(status: int, data: Any, text: str) -> Tuple[bool, bool]:
        err = (data or {}).get("error", {}) if isinstance(data, dict) else {}
        msg = str(err.get("message", "") if isinstance(err, dict) else "") + " " + (text or "")[:500]
        reasons = {str(d.get("reason")) for d in (err.get("details") or []) if isinstance(d, dict)} \
            if isinstance(err, dict) else set()
        auth = bool({"API_KEY_INVALID", "API_KEY_EXPIRED", "API_KEY_SERVICE_BLOCKED"} & reasons) or \
            (status == 400 and re.search(r"api key (not valid|expired|invalid)", msg, re.I) is not None)
        quota = re.search(r"billing|prepayment|credits? (are|is) depleted|spend(ing)? cap", msg, re.I) is not None \
            and status in (400, 402, 403, 429)
        return quota, auth

    def _strip_unsupported(self, body: dict) -> None:
        unsup = self._unsupported.setdefault(self.base_url(), set())
        if "mediaResolution" in unsup:
            # per-part resolution rejected: fall back to the request-wide setting (no ultra_high there)
            level = None
            for c in body.get("contents", []):
                for p in c.get("parts", []):
                    level = (p.pop("mediaResolution", None) or {}).get("level") or level
            if level and "generationConfig" in body and "generationMediaResolution" not in unsup:
                body["generationConfig"]["mediaResolution"] = level.replace("ULTRA_HIGH", "HIGH")
        if "generationMediaResolution" in unsup:
            body.get("generationConfig", {}).pop("mediaResolution", None)
        if "thinkingConfig" in unsup:
            body.get("generationConfig", {}).pop("thinkingConfig", None)

    def _post(self, model: str, body: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        key = self.require_key()
        timeout = 600.0 if max_tokens > self.max_tokens or self.effort_for(options) in ("high", "xhigh", "max") else 180.0
        unsup = self._unsupported.setdefault(self.base_url(), set())
        self._strip_unsupported(body)
        try:
            data, _h = vlm.http_json("POST", self._url(model), "Gemini", headers=self._headers(key), body=body,
                                     timeout=timeout, secrets=[key], classify=self._classify_error)
        except vlm.ProviderError as e:
            if e.status == 400:
                msg = str(e)
                gc = body.get("generationConfig", {})
                for feat, present, rx in (
                        ("mediaResolution", any("mediaResolution" in p for c in body.get("contents", [])
                                                for p in c.get("parts", [])), r"media_?resolution"),
                        ("generationMediaResolution", "mediaResolution" in gc, r"media_?resolution"),
                        ("thinkingConfig", "thinkingConfig" in gc, r"thinking")):
                    if present and feat not in unsup and re.search(rx, msg, re.I):
                        unsup.add(feat)
                        return self._post(model, body, options, max_tokens)
            raise
        return self._result(data, model)

    def _result(self, data: dict, model: str, batch: bool = False) -> CallResult:
        um = data.get("usageMetadata") or {}
        tin = int(um.get("promptTokenCount") or 0)
        tout = int(um.get("candidatesTokenCount") or 0) + int(um.get("thoughtsTokenCount") or 0)
        cached = int(um.get("cachedContentTokenCount") or 0)
        served = vlm._norm_model_id(data.get("modelVersion") or model)
        usage = Usage(tin, tout, vlm.cost_usd(self.id, served if vlm.find_model(self.id, served) else model, tin, tout,
                                              cached_tokens=cached, batch=batch))
        cands = data.get("candidates") or []
        block = (data.get("promptFeedback") or {}).get("blockReason")
        if not cands:
            if block:
                return CallResult("", usage, served, "refusal", f"blocked: {block}")
            return CallResult("", usage, served, "stop")
        c = cands[0]
        fr = str(c.get("finishReason") or "")
        parts = (c.get("content") or {}).get("parts") or []
        text = "".join(p.get("text", "") for p in parts if isinstance(p, dict) and not p.get("thought"))
        if fr in REFUSAL_FINISH or (block and not text):
            return CallResult(text, usage, served, "refusal", f"finish reason {fr or block}")
        return CallResult(text, usage, served, "truncated" if fr == "MAX_TOKENS" else "stop")

    def _gen_config(self, model: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> dict:
        gc: Dict[str, Any] = {"responseMimeType": "application/json", "responseJsonSchema": schema,
                              "maxOutputTokens": int(max_tokens)}
        th = self._thinking(model, options)
        if th:
            gc["thinkingConfig"] = th
        return gc

    def _image_part(self, model: str, b64: str, options: Dict[str, Any]) -> dict:
        part: Dict[str, Any] = {"inlineData": {"mimeType": "image/jpeg", "data": b64}}
        if self.generation(model) >= 3:
            part["mediaResolution"] = {"level": f"MEDIA_RESOLUTION_{self.media_resolution(options).upper()}"}
        return part

    def call_detect(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                    options: Dict[str, Any], max_tokens: int) -> CallResult:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [self._image_part(model, prep.b64, options),
                                                        {"text": user_text}]}],
                "generationConfig": self._gen_config(model, schema, options, max_tokens)}
        return self._post(model, body, options, max_tokens)

    def call_classify(self, model: str, images: List[Tuple[int, bytes, Tuple[int, int]]], system: str,
                      user_text: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        parts: List[dict] = []
        level = str(options.get("media_resolution") or "medium").lower()   # a <=448 px crop gains nothing from 1120 tokens
        for idx, data, _size in images:
            parts.append({"text": f"Crop {idx}:"})
            part: Dict[str, Any] = {"inlineData": {"mimeType": "image/jpeg", "data": base64.b64encode(data).decode("ascii")}}
            if self.generation(model) >= 3 and level in MEDIA_EDGE:
                part["mediaResolution"] = {"level": f"MEDIA_RESOLUTION_{level.upper()}"}
            parts.append(part)
        parts.append({"text": user_text})
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": parts}],
                "generationConfig": self._gen_config(model, schema, options, max_tokens)}
        return self._post(model, body, options, max_tokens)

    def call_generate(self, model: str, system: str, user_text: str, schema: dict, images: List[Tuple[bytes, str]],
                      options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        parts: List[dict] = []
        level = str(options.get("media_resolution") or "").lower()
        for k, (data, mime) in enumerate(images):
            if len(images) > 1:
                parts.append({"text": f"Image {k + 1}:"})
            part: Dict[str, Any] = {"inlineData": {"mimeType": mime, "data": base64.b64encode(data).decode("ascii")}}
            if level in MEDIA_EDGE and self.generation(model) >= 3:
                part["mediaResolution"] = {"level": f"MEDIA_RESOLUTION_{level.upper()}"}
            parts.append(part)
        web = bool(options.get("web_search")) and not images
        if web:
            # Grounding with Google Search: JSON mode and tools do not mix on every model, so the schema goes into
            # the prompt and the JSON is read from the text (search charges are not counted in the cost)
            user_text += ("\n\nWhen you are done, answer with one JSON object that follows this JSON schema, and "
                          "nothing after it:\n" + json.dumps(schema, ensure_ascii=False))
        parts.append({"text": user_text})
        gc = self._gen_config(model, vlm.schema_for("gemini", schema), options, max_tokens)
        if web:
            gc.pop("responseMimeType", None)
            gc.pop("responseJsonSchema", None)
        body: Dict[str, Any] = {"contents": [{"role": "user", "parts": parts}], "generationConfig": gc}
        if web:
            body["tools"] = [{"google_search": {}}]
        if system:
            body["systemInstruction"] = {"parts": [{"text": system}]}
        return self._post(model, body, options, max_tokens)

    # ---- Batch API (batchGenerateContent with inline requests; a large job is split into several batches)
    def batch_request(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                      options: Dict[str, Any], max_tokens: int) -> dict:
        body = {"systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"role": "user", "parts": [self._image_part(model, prep.b64, options),
                                                        {"text": user_text}]}],
                "generationConfig": self._gen_config(model, schema, options, max_tokens)}
        self._strip_unsupported(body)
        return body

    def _batch_create(self, model: str, reqs: List[Tuple[str, dict]]) -> str:
        key = self.require_key()
        url = f"{self.base_url()}/v1beta/models/{quote(vlm._norm_model_id(model), safe='.-_')}:batchGenerateContent"
        body = {"batch": {"displayName": f"fovea-{len(reqs)}-requests",
                          "inputConfig": {"requests": {"requests": [{"request": b, "metadata": {"key": sid}}
                                                                    for sid, b in reqs]}}}}
        data, _h = vlm.http_json("POST", url, "Gemini", headers=self._headers(key), body=body, timeout=600.0,
                                 secrets=[key], classify=self._classify_error)
        name = data.get("name") or (data.get("metadata") or {}).get("name") or ""
        if not name:
            raise ProviderError("Gemini did not return a batch name", "bad_output")
        return str(name)

    def _batch_op(self, part: str) -> dict:
        key = self.require_key()
        name = part if part.startswith("batches/") else f"batches/{part}"
        data, _h = vlm.http_json("GET", f"{self.base_url()}/v1beta/{name}", "Gemini", headers=self._headers(key),
                                 timeout=120.0, secrets=[key], classify=self._classify_error)
        return data

    @staticmethod
    def _batch_state(op: dict) -> str:
        meta = op.get("metadata") or {}
        return str(meta.get("state") or op.get("state") or "").upper()

    def _batch_get(self, part: str) -> dict:
        op = self._batch_op(part)
        st = self._batch_state(op)
        meta = op.get("metadata") or {}
        bs = meta.get("batchStats") or op.get("batchStats") or {}
        total = int(bs.get("requestCount") or 0)
        ok = int(bs.get("successfulRequestCount") or 0)
        bad = int(bs.get("failedRequestCount") or 0)
        pending = int(bs.get("pendingRequestCount") or max(0, total - ok - bad))
        counts = {"total": total, "processing": pending, "succeeded": ok, "errored": bad, "canceled": 0, "expired": 0}
        tail = st.split("_STATE_")[-1] if "_STATE_" in st else st
        rest = max(0, total - ok - bad)                # never processed: canceled / expired
        if tail in ("CANCELLED", "CANCELED"):
            counts["canceled"], counts["processing"] = rest, 0
        if tail == "EXPIRED":
            counts["expired"], counts["processing"] = rest, 0
        done = bool(op.get("done")) or tail in ("SUCCEEDED", "FAILED", "CANCELLED", "CANCELED", "EXPIRED")
        state = "running"
        if tail == "FAILED":
            state = "failed"
        elif done:
            state = "ended"
        out = {"state": state, "counts": counts}
        err = op.get("error")
        if isinstance(err, dict) and err.get("message"):
            out["message"] = vlm.scrub(err["message"])[:300]
        return out

    def _batch_fetch(self, part: str, ctx: dict) -> Iterator[Tuple[str, Any]]:
        key = self.require_key()
        op = self._batch_op(part)
        tail = self._batch_state(op).split("_STATE_")[-1]
        resp = op.get("response") or (op.get("metadata") or {}).get("output") or op.get("output") or {}
        inl = resp.get("inlinedResponses")
        if isinstance(inl, dict):
            inl = inl.get("inlinedResponses")
        lines: List[dict] = []
        if isinstance(inl, list):
            lines = [x for x in inl if isinstance(x, dict)]
        elif resp.get("responsesFile"):
            fname = str(resp["responsesFile"])
            raw = vlm.http_bytes("GET", f"{self.base_url()}/download/v1beta/{fname}:download?alt=media", "Gemini",
                                 headers=self._headers(key), secrets=[key], classify=self._classify_error)
            lines = list(vlm.jsonl(raw))
        models = {info["model"] for info in ctx.get("images", {}).values()}
        model = next(iter(models)) if len(models) == 1 else ""
        order = [sid for sid, m in ctx.get("subs", {}).items()
                 if m[0] in set(ctx.get("part_images", {}).get(part, []))]
        seen = set()
        for k, line in enumerate(lines):
            meta = line.get("metadata") or {}
            sid = str(meta.get("key") or line.get("key") or (order[k] if k < len(order) else ""))
            seen.add(sid)
            if isinstance(line.get("response"), dict):
                yield sid, self._result(line["response"], model, batch=True)
            else:
                err = line.get("error") if isinstance(line.get("error"), dict) else {}
                status = int(err.get("code") or 500)
                msg = vlm.scrub(err.get("message") or "request failed", key)[:300]
                code = "auth" if status in (401, 403) else ("rate" if status == 429 else
                                                             ("overloaded" if status >= 500 else "invalid"))
                yield sid, ProviderError(f"Gemini batch request failed ({status}): {msg}", code, status=status)
        if tail in ("CANCELLED", "CANCELED", "EXPIRED", "FAILED"):
            why = {"EXPIRED": ("expired", "This request expired before Gemini processed it"),
                   "FAILED": ("overloaded", "The Gemini batch failed")}.get(
                       tail, ("canceled", "The batch was canceled before this image was processed"))
            for sid in order:
                if sid not in seen:
                    yield sid, ProviderError(why[1], why[0])

    def _batch_cancel(self, part: str) -> None:
        key = self.require_key()
        name = part if part.startswith("batches/") else f"batches/{part}"
        vlm.http_json("POST", f"{self.base_url()}/v1beta/{name}:cancel", "Gemini", headers=self._headers(key),
                      body={}, timeout=60.0, secrets=[key], classify=self._classify_error)

    def list_remote_models(self) -> List[dict]:
        key = self.require_key()
        out: List[dict] = []
        token = ""
        for _ in range(10):
            url = f"{self.base_url()}/v1beta/models?pageSize=1000" + (f"&pageToken={quote(token)}" if token else "")
            data, _h = vlm.http_json("GET", url, "Gemini", headers=self._headers(key), timeout=30.0, secrets=[key],
                                     classify=self._classify_error)
            for m in data.get("models") or []:
                mid = vlm._norm_model_id(str(m.get("name", "")))
                methods = m.get("supportedGenerationMethods") or []
                if not mid.startswith("gemini") or (methods and "generateContent" not in methods):
                    continue
                if re.search(r"embedding|tts|-image|live|native-audio|aqa", mid):
                    continue
                out.append({"id": mid, "name": m.get("displayName") or mid})
            token = data.get("nextPageToken") or ""
            if not token:
                break
        return out


PROVIDER = GeminiProvider()
