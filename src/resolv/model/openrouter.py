"""OpenRouter chat-completions client (stdlib HTTP, no SDK dependency).

resolv's conversation uses provider-neutral blocks (``text``, ``tool_use``,
``tool_result``); this client translates them to and from the OpenAI-style
chat format that OpenRouter accepts.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request
from collections.abc import Callable

from resolv.budget import Usage
from resolv.errors import ModelError
from resolv.model import ModelResponse, ToolCall, ToolSpec

DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
APP_URL = "https://github.com/himanshu-nakrani/resolv"
APP_TITLE = "resolv"
RETRYABLE_STATUS = frozenset({408, 409, 429, 500, 502, 503, 504, 529})
_CACHE = {"type": "ephemeral"}


def _text_of(content) -> str:
    if isinstance(content, str):
        return content
    return "\n".join(b.get("text", "") for b in content if b.get("type") == "text")


def to_openai_messages(system: str, messages: list[dict]) -> list[dict]:
    out: list[dict] = [{"role": "system", "content": system}]
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "assistant":
            blocks = content if isinstance(content, list) else [{"type": "text", "text": content}]
            entry: dict = {"role": "assistant", "content": _text_of(blocks) or None}
            calls = [
                {
                    "id": b["id"],
                    "type": "function",
                    "function": {
                        "name": b["name"],
                        # Unparseable arguments are kept as the raw string the model sent.
                        "arguments": b["input"] if isinstance(b["input"], str) else json.dumps(b["input"]),
                    },
                }
                for b in blocks
                if b.get("type") == "tool_use"
            ]
            if calls:
                entry["tool_calls"] = calls
            out.append(entry)
        elif isinstance(content, str):
            out.append({"role": role, "content": content})
        else:
            # Tool results become role=tool messages, which must directly follow the assistant turn.
            for block in content:
                if block.get("type") == "tool_result":
                    text = block.get("content", "")
                    if block.get("is_error"):
                        text = f"[tool error] {text}"
                    out.append({"role": "tool", "tool_call_id": block["tool_use_id"], "content": text})
            text = _text_of(content)
            if text:
                out.append({"role": role, "content": text})
    return out


def _parse_arguments(raw) -> dict | str:
    if isinstance(raw, dict):
        return raw
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return str(raw)
    return value if isinstance(value, dict) else str(raw)


def _usage(raw: dict) -> Usage:
    details = raw.get("prompt_tokens_details") or {}
    cached = details.get("cached_tokens") or 0
    written = details.get("cache_write_tokens") or 0
    prompt = raw.get("prompt_tokens") or 0
    cost = raw.get("cost")
    return Usage(
        # prompt_tokens includes cached and cache-write tokens.
        input_tokens=max(0, prompt - cached - written),
        output_tokens=raw.get("completion_tokens") or 0,
        cache_write_tokens=written,
        cache_read_tokens=cached,
        cost_usd=float(cost) if isinstance(cost, (int, float)) else None,
    )


class OpenRouterClient:
    provider = "openrouter"

    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: str = DEFAULT_BASE_URL,
        timeout: float = 300,
        data_collection: str = "deny",
        zero_data_retention: bool = False,
        max_retries: int = 4,
        transport: Callable | None = None,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        self.model = model
        self._api_key = api_key
        self._url = base_url.rstrip("/") + "/chat/completions"
        self._timeout = timeout
        self._routing = {"data_collection": data_collection, "require_parameters": True}
        if zero_data_retention:
            self._routing["zdr"] = True
        self._max_retries = max_retries
        self._transport = transport or urllib.request.urlopen
        self._sleep = sleep
        self.upstream_providers: set[str] = set()

    def complete(self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_tokens: int) -> ModelResponse:
        body: dict = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": to_openai_messages(system, messages),
            "tools": [
                {"type": "function", "function": {"name": t.name, "description": t.description, "parameters": t.input_schema}}
                for t in tools
            ],
            "provider": self._routing,
        }
        if self.model.startswith("anthropic/"):
            # Anthropic models cache only when asked; most other providers cache automatically.
            body["cache_control"] = _CACHE
        data = self._post(body)

        choices = data.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ModelError(f"response contained no choices: {json.dumps(data)[:500]}")
        choice = choices[0]
        if choice.get("error"):
            raise ModelError(f"provider error: {json.dumps(choice['error'])[:500]}")
        message = choice.get("message") or {}

        content: list[dict] = []
        text = message.get("content")
        if isinstance(text, str) and text.strip():
            content.append({"type": "text", "text": text})
        calls = []
        for raw in message.get("tool_calls") or []:
            fn = raw.get("function") or {}
            call = ToolCall(id=raw.get("id") or f"call_{len(calls)}", name=fn.get("name", ""), arguments=_parse_arguments(fn.get("arguments")))
            calls.append(call)
            content.append({"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments})
        if isinstance(data.get("provider"), str):
            self.upstream_providers.add(data["provider"])
        return ModelResponse(
            content=content, tool_calls=calls, usage=_usage(data.get("usage") or {}), stop_reason=choice.get("finish_reason")
        )

    def _post(self, body: dict) -> dict:
        payload = json.dumps(body).encode("utf-8")
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self._api_key}",
            "HTTP-Referer": APP_URL,
            "X-Title": APP_TITLE,
        }
        last_error = "unknown error"
        for attempt in range(self._max_retries + 1):
            request = urllib.request.Request(self._url, data=payload, headers=headers, method="POST")
            retry_after: float | None = None
            try:
                with self._transport(request, timeout=self._timeout) as response:
                    data = json.loads(response.read().decode("utf-8"))
                error = data.get("error") if isinstance(data, dict) else None
                if not error:
                    return data
                # OpenRouter can report upstream failures inside a 200 response.
                code = error.get("code") if isinstance(error, dict) else None
                last_error = f"error {code}: {json.dumps(error)[:500]}"
                if code not in RETRYABLE_STATUS:
                    raise ModelError(last_error)
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
