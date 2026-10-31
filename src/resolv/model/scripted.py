"""Deterministic model for tests and offline demos: replays scripted responses."""

from __future__ import annotations

import copy
import itertools
from collections.abc import Callable, Iterable

from resolv.budget import Usage
from resolv.errors import ModelError
from resolv.model import ModelResponse, ToolCall, ToolSpec

_ids = itertools.count(1)

Step = ModelResponse | Callable[[list[dict]], ModelResponse]


def call(name: str, **arguments) -> ToolCall:
    return ToolCall(id=f"toolu_{next(_ids):04d}", name=name, arguments=arguments)


def respond(*calls: ToolCall, text: str = "", usage: Usage | None = None) -> ModelResponse:
    content: list[dict] = [{"type": "text", "text": text}] if text else []
    content += [{"type": "tool_use", "id": c.id, "name": c.name, "input": c.arguments} for c in calls]
    return ModelResponse(content=content, tool_calls=list(calls), usage=usage or Usage(100, 50))


class ScriptedModel:
    provider = "scripted"
    model = "scripted"

    def __init__(self, steps: Iterable[Step]) -> None:
        self._steps = list(steps)
        self.requests: list[dict] = []

    @property
    def remaining(self) -> int:
        return len(self._steps)

    def complete(self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_tokens: int) -> ModelResponse:
        self.requests.append(
            {"system": system, "messages": copy.deepcopy(messages), "tools": [t.name for t in tools]}
        )
        if not self._steps:
            raise ModelError("scripted model has no more responses")
        step = self._steps.pop(0)
        return step(messages) if callable(step) else step
