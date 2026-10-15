"""OpenAI through the Responses API (httpx).

OpenAI documents no box convention; the frame every published GPT-6 detection benchmark used is Gemini-style
``box_2d = [ymin, xmin, ymax, xmax]`` in 0-1000, so that is what we ask for. Images are pre-resized to the
``detail:"high"`` budget (<= 2048 px, <= 2500 patches of 32 px) so the server never rescales. ``store: false`` keeps
requests out of OpenAI's stored responses; the strict json_schema format guarantees the shape.
Economy runs upload a JSONL file (``/v1/files``, purpose batch) and create a ``/v1/batches`` job over
``/v1/responses``; the output and error files are matched back by ``custom_id`` (their order is not preserved).
"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Iterator, List, Optional, Tuple

from .. import vlm
from ..base import ProviderError, Usage
from ..vlm import CallResult, Prepared

_SKIP = re.compile(r"audio|realtime|transcribe|tts|image|embedding|search|instruct|moderation|whisper|dall-e|sora|"
                   r"computer-use|codex")


class OpenAIProvider(vlm.VLMProvider):
    id = "openai"
    name = "OpenAI"
    kind = "cloud"
    needs_key = True
    env_keys = ("OPENAI_API_KEY",)
    capabilities = {"detect": True, "classify": True, "refine": False, "open_vocab": True, "exemplars": False,
                    "batch_api": True, "generate": True, "flex": True, "web_search": True}
    default_base_url = "https://api.openai.com/v1"
    batch_max_bytes = 180 * 1024 * 1024    # a batch input file may hold 200 MB / 50,000 requests
    batch_max_requests = 50000
    base_url_env = "FOVEA_OPENAI_BASE_URL"
    schema_bounds = True

    def frame_for(self, model: str, options: Dict[str, Any]) -> str:
        return vlm.YXYX_1000

    def native_edge(self, model: str, options: Dict[str, Any]) -> int:
        return 2048

    def sent_size(self, model: str, w: int, h: int, options: Dict[str, Any]) -> Tuple[int, int]:
        return vlm.patch_fit_size(w, h, 2048, 32, 2500)

    def image_tokens(self, model: str, sw: int, sh: int, options: Dict[str, Any]) -> int:
        return vlm.openai_image_tokens(sw, sh)

    def default_effort(self, model: str) -> Optional[str]:
        return "medium"

    def flex(self, options: Dict[str, Any]) -> bool:
        v = options.get("batch_mode", self.config().get("batch_mode"))
        return v in ("flex", True, "true", 1)

    def billing_tier(self, model: str, options: Dict[str, Any]) -> Optional[str]:
        """Flex on (Settings → Flex tier): requests go out with service_tier 'flex' and are billed at batch prices —
        unless this model already turned the Flex tier down (then it runs, and is billed, at the standard price)."""
        if not self.flex(options or {}):
            return None
        if "service_tier" in self._unsupported.get(f"{self.base_url()}|{model}", set()):
            return None
        return "flex"

    def _reasoning(self, model: str, options: Dict[str, Any]) -> Optional[dict]:
        effort = self.effort_for(options)
        m = vlm.find_model(self.id, model)
        if m is None:
            return {"effort": effort} if effort else None
        efforts = m.get("efforts") or []
        if not efforts:
            return None
        return {"effort": vlm.pick_effort(effort or "medium", efforts)}

    @staticmethod
    def _classify_error(status: int, data: Any, text: str) -> Tuple[bool, bool]:
        err = (data or {}).get("error", {}) if isinstance(data, dict) else {}
        code = str(err.get("code") or "") if isinstance(err, dict) else ""
        etype = str(err.get("type") or "") if isinstance(err, dict) else ""
        quota = code in ("insufficient_quota", "billing_hard_limit_reached", "billing_not_active") or \
            etype in ("insufficient_quota", "billing_error")
        auth = code in ("invalid_api_key", "invalid_organization") or status == 401
        return quota, auth

    _unsupported: Dict[str, set] = {}     # "base_url|model" -> optional params the model rejected

    def _post(self, model: str, body: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        key = self.require_key()
        unsup = self._unsupported.setdefault(f"{self.base_url()}|{model}", set())
        flex = self.flex(options) and "service_tier" not in unsup
        if flex:
            body["service_tier"] = "flex"
        for feat in unsup:
            body.pop(feat, None)
        timeout = 900.0 if flex else (600.0 if max_tokens > self.max_tokens
                                      or self.effort_for(options) in ("high", "xhigh", "max") else 180.0)
        try:
            data, _h = vlm.http_json("POST", f"{self.base_url()}/responses", "OpenAI",
                                     headers={"Authorization": f"Bearer {key}", "content-type": "application/json"},
                                     body=body, timeout=timeout, secrets=[key], classify=self._classify_error)
        except vlm.ProviderError as e:
            # older / non-reasoning models reject reasoning.effort; some models have no Flex tier
            if e.status == 400:
                for feat, rx in (("reasoning", r"reasoning"), ("service_tier", r"service[_ ]tier|flex")):
                    if feat in body and feat not in unsup and re.search(rx, str(e), re.I):
                        unsup.add(feat)
                        return self._post(model, body, options, max_tokens)
            raise
        return self._result(data, model, flex)

    def _result(self, data: dict, model: str, flex: bool, batch: bool = False) -> CallResult:
        u = data.get("usage") or {}
        tin = int(u.get("input_tokens") or 0)
        tout = int(u.get("output_tokens") or 0)
        cached = int((u.get("input_tokens_details") or {}).get("cached_tokens") or 0)
        served = data.get("model") or model
        price_model = served if vlm.find_model(self.id, served) else model
        if batch:                               # the Batch API bills every request at batch prices
            flex = True
        elif data.get("service_tier"):          # bill by the tier that actually served the request
            flex = data.get("service_tier") == "flex"
        usage = Usage(tin, tout, vlm.cost_usd(self.id, price_model, tin, tout, cached_tokens=cached, batch=flex))
        texts, refusal = [], None
        for item in data.get("output") or []:
            if not isinstance(item, dict) or item.get("type") != "message":
                continue
            for c in item.get("content") or []:
                if c.get("type") == "output_text":
                    texts.append(c.get("text") or "")
                elif c.get("type") == "refusal":
                    refusal = c.get("refusal") or "refused"
        text = "".join(texts)
        if refusal and not text.strip():
            return CallResult("", usage, served, "refusal", vlm.scrub(refusal)[:160])
        status = data.get("status")
        reason = (data.get("incomplete_details") or {}).get("reason")
        if status == "incomplete" and reason == "content_filter":
            return CallResult(text, usage, served, "refusal", "content filter")
        if status == "incomplete" and reason == "max_output_tokens":
            return CallResult(text, usage, served, "truncated")
        if status == "failed":
            err = data.get("error") if isinstance(data.get("error"), dict) else {}
            raise vlm.Transient(f"OpenAI response failed: {vlm.scrub(err.get('message') or 'server error')[:200]}",
                                "overloaded")
        return CallResult(text, usage, served, "stop")

    def _body(self, model: str, system: str, content: List[dict], schema: dict, name: str, options: Dict[str, Any],
              max_tokens: int) -> dict:
        body: Dict[str, Any] = {"model": model, "store": False, "instructions": system,
                                "input": [{"role": "user", "content": content}],
                                "max_output_tokens": int(max_tokens),
                                "text": {"format": {"type": "json_schema", "name": name, "strict": True,
                                                    "schema": schema}}}
        r = self._reasoning(model, options)
        if r:
            body["reasoning"] = r
        return body

    def call_detect(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                    options: Dict[str, Any], max_tokens: int) -> CallResult:
        content = [{"type": "input_image", "image_url": prep.data_url(), "detail": "high"},
                   {"type": "input_text", "text": user_text}]
        return self._post(model, self._body(model, system, content, schema, "detections", options, max_tokens),
                          options, max_tokens)

    def call_classify(self, model: str, images: List[Tuple[int, bytes, Tuple[int, int]]], system: str,
                      user_text: str, schema: dict, options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        content: List[dict] = []
        for idx, data, _size in images:
            content.append({"type": "input_text", "text": f"Crop {idx}:"})
            content.append({"type": "input_image", "detail": "high",
                            "image_url": "data:image/jpeg;base64," + base64.b64encode(data).decode("ascii")})
        content.append({"type": "input_text", "text": user_text})
        return self._post(model, self._body(model, system, content, schema, "classifications", options, max_tokens),
                          options, max_tokens)

    def call_generate(self, model: str, system: str, user_text: str, schema: dict, images: List[Tuple[bytes, str]],
                      options: Dict[str, Any], max_tokens: int) -> CallResult:
        import base64
        content: List[dict] = []
        for k, (data, mime) in enumerate(images):
            if len(images) > 1:
                content.append({"type": "input_text", "text": f"Image {k + 1}:"})
            content.append({"type": "input_image", "detail": str(options.get("detail") or "high"),
                            "image_url": f"data:{mime};base64," + base64.b64encode(data).decode("ascii")})
        web = bool(options.get("web_search")) and not images
        if web:
            # the web search tool: the schema goes into the prompt and the JSON is read from the answer text
            user_text += ("\n\nWhen you are done, answer with one JSON object that follows this JSON schema, and "
                          "nothing after it:\n" + json.dumps(schema, ensure_ascii=False))
        content.append({"type": "input_text", "text": user_text})
        opts = dict(options)
        opts.setdefault("batch_mode", False)       # interactive: never queue on the Flex tier unless asked to
        body = self._body(model, system or "Answer with JSON only.", content,
                          vlm.schema_for("openai", schema, strict=True), "result", opts, max_tokens)
        if web:
            body.pop("text", None)
            body["tools"] = [{"type": "web_search"}]
        return self._post(model, body, opts, max_tokens)

    # ---- Batch API (/v1/files + /v1/batches over /v1/responses)
    def batch_request(self, model: str, prep: Prepared, system: str, user_text: str, schema: dict,
                      options: Dict[str, Any], max_tokens: int) -> dict:
        content = [{"type": "input_image", "image_url": prep.data_url(), "detail": "high"},
                   {"type": "input_text", "text": user_text}]
        body = self._body(model, system, content, schema, "detections", options, max_tokens)
        for feat in self._unsupported.get(f"{self.base_url()}|{model}", set()):
            if feat != "service_tier":
                body.pop(feat, None)
        return body

    def _auth(self, key: str) -> dict:
        return {"Authorization": f"Bearer {key}"}

    def _batch_create(self, model: str, reqs: List[Tuple[str, dict]]) -> str:
        key = self.require_key()
        lines = "\n".join(json.dumps({"custom_id": sid, "method": "POST", "url": "/v1/responses", "body": body},
                                     separators=(",", ":")) for sid, body in reqs).encode("utf-8")
        f, _h = vlm.http_json("POST", f"{self.base_url()}/files", "OpenAI", headers=self._auth(key), timeout=600.0,
                              secrets=[key], classify=self._classify_error, data={"purpose": "batch"},
                              files={"file": ("fovea-batch.jsonl", lines, "application/jsonl")})
        fid = f.get("id")
        if not fid:
            raise ProviderError("OpenAI did not return a file id for the batch input", "bad_output")
        b, _h = vlm.http_json("POST", f"{self.base_url()}/batches", "OpenAI",
                              headers={**self._auth(key), "content-type": "application/json"}, timeout=120.0,
                              secrets=[key], classify=self._classify_error,
                              body={"input_file_id": fid, "endpoint": "/v1/responses", "completion_window": "24h",
                                    "metadata": {"source": "fovea"}})
        if not b.get("id"):
            raise ProviderError("OpenAI did not return a batch id", "bad_output")
        return str(b["id"])

    def _batch_obj(self, part: str) -> dict:
        key = self.require_key()
        data, _h = vlm.http_json("GET", f"{self.base_url()}/batches/{part}", "OpenAI", headers=self._auth(key),
                                 timeout=60.0, secrets=[key], classify=self._classify_error)
        return data

    def _batch_get(self, part: str) -> dict:
        b = self._batch_obj(part)
        st = str(b.get("status") or "")
        rc = b.get("request_counts") or {}
        total, done, failed = int(rc.get("total") or 0), int(rc.get("completed") or 0), int(rc.get("failed") or 0)
        rest = max(0, total - done - failed)
        counts = {"total": total, "processing": rest, "succeeded": done, "errored": failed, "canceled": 0, "expired": 0}
        state = "running"
        if st in ("completed", "expired", "cancelled"):
            state = "ended"
            counts["processing"] = 0
            if st == "expired":
                counts["expired"] += rest
            elif st == "cancelled":
                counts["canceled"] += rest
        elif st == "cancelling":
            state = "canceling"
        elif st == "failed":
            state = "failed"
        out = {"state": state, "counts": counts}
        errs = ((b.get("errors") or {}).get("data") or []) if isinstance(b.get("errors"), dict) else []
        if errs and isinstance(errs[0], dict):
            out["message"] = vlm.scrub(errs[0].get("message") or "")[:300]
        return out

    def _batch_fetch(self, part: str, ctx: dict) -> Iterator[Tuple[str, Any]]:
        key = self.require_key()
        b = self._batch_obj(part)
        models = {info["model"] for info in ctx.get("images", {}).values()}
        model = next(iter(models)) if len(models) == 1 else ""
        seen = set()
        for fid in (b.get("output_file_id"), b.get("error_file_id")):
            if not fid:
                continue
            raw = vlm.http_bytes("GET", f"{self.base_url()}/files/{fid}/content", "OpenAI", headers=self._auth(key),
                                 secrets=[key], classify=self._classify_error)
            for line in vlm.jsonl(raw):
                sid = str(line.get("custom_id") or "")
                seen.add(sid)
                resp = line.get("response") if isinstance(line.get("response"), dict) else {}
                status = int(resp.get("status_code") or 0)
                body = resp.get("body") if isinstance(resp.get("body"), dict) else None
                if body is not None and 200 <= status < 300:
                    try:
                        yield sid, self._result(body, model or body.get("model") or "", True, batch=True)
                    except vlm.Transient as e:
                        yield sid, ProviderError(str(e), e.code)
                    continue
                err = line.get("error") if isinstance(line.get("error"), dict) else \
                    ((body or {}).get("error") if isinstance((body or {}).get("error"), dict) else {})
                msg = vlm.scrub(err.get("message") or "request failed", key)[:300]
                ecode = str(err.get("code") or "")
                code = ("expired" if ecode == "batch_expired" else "canceled" if ecode == "batch_cancelled" else
                        "quota" if ecode in ("insufficient_quota", "billing_hard_limit_reached") else
                        "auth" if status in (401, 403) else "rate" if status == 429 else
                        "overloaded" if status >= 500 else "invalid")
                yield sid, ProviderError(f"OpenAI batch request failed ({status or ecode or 'error'}): {msg}", code,
                                         status=status or None)
        st = str(b.get("status") or "")
        if st in ("expired", "cancelled", "failed"):
            mine = set(ctx.get("part_images", {}).get(part, []))
            for sid, m in ctx.get("subs", {}).items():
                if m[0] in mine and sid not in seen:
                    yield sid, ProviderError("The batch ended before this image was processed",
                                             {"expired": "expired", "cancelled": "canceled"}.get(st, "overloaded"))

    def _batch_cancel(self, part: str) -> None:
        key = self.require_key()
        vlm.http_json("POST", f"{self.base_url()}/batches/{part}/cancel", "OpenAI", headers=self._auth(key),
                      timeout=60.0, secrets=[key], classify=self._classify_error)

    def list_remote_models(self) -> List[dict]:
        key = self.require_key()
        data, _h = vlm.http_json("GET", f"{self.base_url()}/models", "OpenAI",
                                 headers={"Authorization": f"Bearer {key}"}, timeout=30.0, secrets=[key],
                                 classify=self._classify_error)
        avoid = vlm.provider_table(self.id).get("avoid", {})
        out = []
        for m in data.get("data") or []:
            mid = str(m.get("id") or "")
            if not mid.startswith(("gpt-", "o")) or _SKIP.search(mid):
                continue
            e = {"id": mid, "name": mid}
            for k, note in avoid.items():
                if mid == k or mid.startswith(k + "-"):
                    e["notes"] = note
            out.append(e)
        out.sort(key=lambda e: e["id"])
        return out


PROVIDER = OpenAIProvider()
