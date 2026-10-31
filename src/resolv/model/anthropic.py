"""Anthropic Messages API client (stdlib HTTP, no SDK dependency)."""

from __future__ import annotations

import copy
import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from resolv.budget import Usage
from resolv.errors import ModelError
from resolv.model import ModelResponse, ToolCall, ToolSpec

API_VERSION = "2023-06-01"
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_CACHE = {"type": "ephemeral"}


def _with_cache_breakpoint(messages: list[dict]) -> list[dict]:
    """Mark the end of the conversation so the growing prefix is cached between turns."""
    if not messages:
        return messages
    messages = list(messages)
    last = copy.deepcopy(messages[-1])
    content = last["content"]
    if isinstance(content, str):
        last["content"] = [{"type": "text", "text": content, "cache_control": _CACHE}]
    elif content:
        content[-1]["cache_control"] = _CACHE
    messages[-1] = last
    return messages


class AnthropicClient:
    provider = "anthropic"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = "https://api.anthropic.com",
        timeout: float = 300,
        max_retries: int = 4,
        transport: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/v1/messages"
        self._timeout = timeout
        self._max_retries = max_retries
        self._transport = transport or urllib.request.urlopen
        self._sleep = sleep

    def complete(self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_tokens: int) -> ModelResponse:
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "system": [{"type": "text", "text": system, "cache_control": _CACHE}],
            "messages": _with_cache_breakpoint(messages),
            "tools": [{"name": t.name, "description": t.description, "input_schema": t.input_schema} for t in tools],
        }
        data = self._post(body)
        content = data.get("content")
        if not isinstance(content, list):
            raise ModelError("response did not contain a content list")
        calls = [
            ToolCall(id=b["id"], name=b["name"], arguments=b.get("input") or {})
            for b in content
            if b.get("type") == "tool_use"
        ]
        raw = data.get("usage") or {}
        usage = Usage(
            input_tokens=raw.get("input_tokens") or 0,
            output_tokens=raw.get("output_tokens") or 0,
            cache_write_tokens=raw.get("cache_creation_input_tokens") or 0,
            cache_read_tokens=raw.get("cache_read_input_tokens") or 0,
        )
        return ModelResponse(content=content, tool_calls=calls, usage=usage, stop_reason=data.get("stop_reason"))

    def _post(self, body: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "content-type": "application/json",
            "x-api-key": self._api_key,
            "anthropic-version": API_VERSION,
        }
        last_error = "unknown error"
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
            retry_after: float | None = None
            try:
                with self._transport(request, timeout=self._timeout) as response:
                    return json.loads(response.read().decode("utf-8"))
            except urllib.error.HTTPError as exc:
                detail = exc.read().decode("utf-8", errors="replace")[:500]
                last_error = f"HTTP {exc.code}: {detail}"
                if exc.code not in RETRYABLE_STATUS:
                    raise ModelError(last_error) from exc
                header = exc.headers.get("retry-after") if exc.headers else None
                try:
                    retry_after = float(header) if header else None
                except ValueError:
                    retry_after = None
            except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
                last_error = f"network error: {exc}"
            except json.JSONDecodeError as exc:
                raise ModelError(f"invalid JSON response: {exc}") from exc
            if attempt < self._max_retries:
                self._sleep(min(60.0, retry_after if retry_after is not None else 2.0 * 2**attempt))
        raise ModelError(f"model request failed after {self._max_retries + 1} attempts: {last_error}")
