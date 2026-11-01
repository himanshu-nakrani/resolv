"""Model gateway: provider-neutral types and client construction."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Protocol

from resolv.budget import Usage
from resolv.config import ModelConfig
from resolv.errors import ConfigError


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    input_schema: dict


@dataclass
class ToolCall:
    id: str
    name: str
    # A str holds raw arguments that were not a JSON object.
    arguments: dict | str


@dataclass
class ModelResponse:
    # Neutral content blocks (text / tool_use), echoed back verbatim in the conversation.
    content: list[dict]
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None

    @property
    def text(self) -> str:
        return "\n".join(b.get("text", "") for b in self.content if b.get("type") == "text").strip()


class ModelClient(Protocol):
    provider: str
    model: str

    def complete(
        self, *, system: str, messages: list[dict], tools: list[ToolSpec], max_tokens: int
    ) -> ModelResponse: ...


API_KEY_ENV = {"openrouter": "OPENROUTER_API_KEY"}


def create_client(config: ModelConfig, environ: Mapping[str, str]) -> ModelClient:
    if config.provider == "openrouter":
        from resolv.model.openrouter import DEFAULT_BASE_URL, OpenRouterClient

        key = environ.get(API_KEY_ENV["openrouter"])
        if not key:
            raise ConfigError("OPENROUTER_API_KEY is not set")
        return OpenRouterClient(
            model=config.name,
            api_key=key,
            base_url=environ.get("OPENROUTER_BASE_URL") or DEFAULT_BASE_URL,
            timeout=config.request_timeout_seconds,
            data_collection=config.data_collection,
            zero_data_retention=config.zero_data_retention,
        )
    raise ConfigError(f"unsupported model provider: {config.provider!r} (Phase 1 supports 'openrouter')")
