"""Tool-use loop for one workflow stage.

The orchestrator owns the loop: it decides which tools exist, executes them,
caps their output, and ends the stage when the model calls a final tool. The
result of that final call is supplied later by the orchestrator (via resume),
so verification feedback reaches the model in the same conversation.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import dataclass

from resolv.budget import Budget
from resolv.errors import AgentTurnLimit, ToolError
from resolv.events import EventLog
from resolv.model import ModelClient, ToolSpec


@dataclass
class Tool:
    spec: ToolSpec
    handler: Callable[[dict], str]
    final: bool = False


@dataclass
class FinalCall:
    name: str
    arguments: dict


def _result(call_id: str, content: str, is_error: bool = False) -> dict:
    block = {"type": "tool_result", "tool_use_id": call_id, "content": content}
    if is_error:
        block["is_error"] = True
    return block


def _preview(arguments: dict, limit: int = 300) -> str:
    text = json.dumps(arguments, default=str)
    return text if len(text) <= limit else text[:limit] + "..."


class AgentSession:
    def __init__(
        self,
        *,
        stage: str,
        client: ModelClient,
        budget: Budget,
        events: EventLog,
        system: str,
        tools: list[Tool],
        max_turns: int,
        max_output_tokens: int,
        max_tool_output_chars: int,
    ) -> None:
        self.stage = stage
        self.client = client
        self.budget = budget
        self.events = events
        self.system = system
        self.tools = {t.spec.name: t for t in tools}
        self.specs = [t.spec for t in tools]
        self.final_names = sorted(t.spec.name for t in tools if t.final)
        self.max_turns = max_turns
        self.max_output_tokens = max_output_tokens
        self.max_tool_output_chars = max_tool_output_chars
        self.messages: list[dict] = []
        self.turns = 0
        self._pending: tuple[list[dict], int, str] | None = None

    def start(self, prompt: str) -> FinalCall:
        self.messages.append({"role": "user", "content": prompt})
        return self._loop()

    def resume(self, feedback: str) -> FinalCall:
        """Answer the pending final call with orchestrator feedback and continue."""
        assert self._pending is not None, "no pending final call"
        results, index, call_id = self._pending
        results[index] = _result(call_id, feedback)
        self.messages.append({"role": "user", "content": results})
        self._pending = None
        return self._loop()

    def _cap(self, text: str) -> str:
        limit = self.max_tool_output_chars
        if len(text) <= limit:
            return text
        return text[:limit] + f"\n... [output truncated at {limit} characters; request a narrower range]"

    def _loop(self) -> FinalCall:
        for _ in range(self.max_turns):
            self.budget.check()
            response = self.client.complete(
                system=self.system, messages=self.messages, tools=self.specs, max_tokens=self.max_output_tokens
            )
            self.budget.record(response.usage)
            self.turns += 1
            self.events.emit(
                "model_response",
                text=response.text[:2000],
                tool_calls=[{"name": c.name, "arguments": _preview(c.arguments)} for c in response.tool_calls],
                stop_reason=response.stop_reason,
                usage=vars(response.usage),
            )
            self.messages.append(
                {"role": "assistant", "content": response.content or [{"type": "text", "text": "(no content)"}]}
            )
            if not response.tool_calls:
                self.messages.append(
                    {
                        "role": "user",
                        "content": "Continue by calling a tool. To finish this stage call one of: "
                        + ", ".join(self.final_names) + ".",
                    }
                )
                continue

            results: list[dict] = []
            final: tuple[int, str, FinalCall] | None = None
            for call in response.tool_calls:
                tool = self.tools.get(call.name)
                if final is not None:
                    results.append(_result(call.id, "Not executed: this stage was already submitted.", True))
                    continue
                if tool is None:
                    results.append(_result(call.id, f"Unknown tool {call.name!r} in this stage.", True))
                    continue
                try:
                    output = tool.handler(call.arguments if isinstance(call.arguments, dict) else {})
                    is_error = False
                except (ToolError, OSError) as exc:
                    output, is_error = f"Error: {exc}", True
                self.events.emit(
                    "tool_call", name=call.name, arguments=_preview(call.arguments), is_error=is_error,
                    output=output[:500],
                )
                if tool.final and not is_error:
                    final = (len(results), call.id, FinalCall(call.name, call.arguments))
                    results.append({})  # filled by resume()
                    continue
                results.append(_result(call.id, self._cap(output), is_error))

            if final is not None:
                index, call_id, final_call = final
                self._pending = (results, index, call_id)
                return final_call
            self.messages.append({"role": "user", "content": results})
        raise AgentTurnLimit(f"{self.stage}: no final submission within {self.max_turns} model turns")
