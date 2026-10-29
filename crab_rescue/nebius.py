"""Real model clients for an OpenAI-compatible chat/completions endpoint (Nebius Token Factory).

Request shape (`POST {base}/chat/completions`, `Authorization: Bearer <key>`, body with `model`
and `messages`) is the one your working curl uses. Image input uses the OpenAI-style
`image_url` content part with a data URL; that part is UNVERIFIED per model until
`crab_rescue.discover --probe` confirms it. The key is read from the environment only and never
appears in an exception, a log line or a file.
"""

from __future__ import annotations

import base64
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any

from .clients import CallResult, DiskCache, ModelCallError
from .prompts import PLAN_SYSTEM, VLM_SYSTEM, plan_user_text, vlm_user_text


class ApiError(ModelCallError):
    """HTTP or transport failure. Carries status and a short body, never headers or the key."""


def env_credentials() -> tuple[str, str]:
    key, base = os.environ.get("NEBIUS_API_KEY", ""), os.environ.get("NEBIUS_BASE_URL", "")
    if not key or not base:
        raise SystemExit("Set NEBIUS_API_KEY and NEBIUS_BASE_URL in the environment "
                         "(see crab_rescue/.env.example). They are never read from files.")
    return key, base


class ChatTransport:
    def __init__(self, base_url: str, api_key: str, timeout_s: float = 90.0, max_retries: int = 2):
        self.base = base_url.rstrip("/")
        self._key = api_key
        self.timeout = timeout_s
        self.max_retries = max_retries

    def _request(self, method: str, path: str, body: dict[str, Any] | None) -> dict[str, Any]:
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(
            self.base + path, data=data, method=method,
            headers={"Content-Type": "application/json", "Accept": "*/*",
                     "Authorization": f"Bearer {self._key}"})
        last = "no attempt"
        for attempt in range(self.max_retries + 1):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as r:
                    return json.loads(r.read().decode())
            except urllib.error.HTTPError as e:
                text = e.read().decode(errors="replace")[:300]
                last = f"HTTP {e.code}: {text}"
                if e.code not in (429, 500, 502, 503, 504):
                    raise ApiError(last) from None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
                last = f"transport error: {type(e).__name__}"
            if attempt < self.max_retries:
                time.sleep(min(8.0, 1.0 * 2 ** attempt))        # bounded exponential backoff
        raise ApiError(f"gave up after {self.max_retries + 1} attempts ({last})")

    def chat(self, model: str, messages: list[dict[str, Any]], max_tokens: int) -> dict[str, Any]:
        return self._request("POST", "/chat/completions",
                             {"model": model, "messages": messages, "max_tokens": max_tokens,
                              "temperature": 0})

    def list_models(self) -> dict[str, Any]:
        return self._request("GET", "/models", None)


def _result(resp: dict[str, Any], model: str, latency: float, est_in: int) -> CallResult:
    msg = (resp.get("choices") or [{}])[0].get("message") or {}
    text = msg.get("content") or ""
    usage = resp.get("usage") or {}
    tin = int(usage.get("prompt_tokens") or est_in)
    tout = int(usage.get("completion_tokens") or max(1, len(text) // 4))
    return CallResult(text=text, tokens_in=tin, tokens_out=tout, latency_s=latency, model=model)


class NebiusVLM:
    def __init__(self, transport: ChatTransport, models: dict[str, str], max_tokens: int,
                 cache: DiskCache | None = None):
        self.t, self.models, self.max_tokens, self.cache = transport, models, max_tokens, cache

    def perceive(self, role: str, image_png: bytes, meta: dict[str, Any]) -> CallResult:
        model = self.models[role]
        b64 = base64.b64encode(image_png).decode()
        messages = [{"role": "system", "content": VLM_SYSTEM},
                    {"role": "user", "content": [
                        {"type": "text", "text": vlm_user_text(meta)},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{b64}"}}]}]
        key = self.cache.key(model, json.dumps(messages)) if self.cache else None
        if key and (hit := self.cache.get(key)):
            return CallResult(hit.text, 0, 0, 0.0, f"{model} (cached)")
        t0 = time.time()
        resp = self.t.chat(model, messages, self.max_tokens)
        res = _result(resp, model, time.time() - t0, est_in=1500)
        if key and res.text:
            self.cache.put(key, res)
        return res


class NebiusPlanner:
    def __init__(self, transport: ChatTransport, models: dict[str, str], max_tokens: int,
                 cache: DiskCache | None = None):
        self.t, self.models, self.max_tokens, self.cache = transport, models, max_tokens, cache

    def plan(self, role: str, context: dict[str, Any], feedback: list[str]) -> CallResult:
        model = self.models[role]
        user = plan_user_text(context, feedback)
        messages = [{"role": "system", "content": PLAN_SYSTEM}, {"role": "user", "content": user}]
        key = self.cache.key(model, json.dumps(messages)) if self.cache else None
        if key and (hit := self.cache.get(key)):
            return CallResult(hit.text, 0, 0, 0.0, f"{model} (cached)")
        t0 = time.time()
        resp = self.t.chat(model, messages, self.max_tokens)
        res = _result(resp, model, time.time() - t0, est_in=len(user) // 4)
        if key and res.text:
            self.cache.put(key, res)
        return res
