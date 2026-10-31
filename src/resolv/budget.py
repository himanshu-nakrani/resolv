"""Run budgets enforced by the orchestrator, outside the model's control."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from dataclasses import dataclass

from resolv.config import Limits, ModelConfig
from resolv.errors import BudgetExceeded, RunCancelled

# Anthropic prompt-cache pricing relative to the base input price.
CACHE_WRITE_MULTIPLIER = 1.25
CACHE_READ_MULTIPLIER = 0.1


@dataclass
class Usage:
    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens + self.cache_write_tokens + self.cache_read_tokens


class Budget:
    """Tracks time, tokens, and estimated spend; raises once any limit is reached.

    Limits are checked before every model call and command, so a single call
    can overshoot by at most its own size.
    """

    def __init__(self, limits: Limits, model: ModelConfig, clock: Callable[[], float] = time.monotonic) -> None:
        self.limits = limits
        self.model = model
        self._clock = clock
        self._start = clock()
        self.usage = Usage()
        self.model_calls = 0
        self.cancel_event = threading.Event()

    @property
    def elapsed_seconds(self) -> float:
        return self._clock() - self._start

    @property
    def remaining_seconds(self) -> float:
        return self.limits.run_timeout_seconds - self.elapsed_seconds

    @property
    def estimated_cost_usd(self) -> float:
        u = self.usage
        weighted_input = (
            u.input_tokens
            + CACHE_WRITE_MULTIPLIER * u.cache_write_tokens
            + CACHE_READ_MULTIPLIER * u.cache_read_tokens
        )
        return (
            weighted_input * self.model.input_cost_per_mtok + u.output_tokens * self.model.output_cost_per_mtok
        ) / 1_000_000

    def record(self, usage: Usage) -> None:
        self.model_calls += 1
        self.usage.input_tokens += usage.input_tokens
        self.usage.output_tokens += usage.output_tokens
        self.usage.cache_write_tokens += usage.cache_write_tokens
        self.usage.cache_read_tokens += usage.cache_read_tokens

    def check(self) -> None:
        if self.cancel_event.is_set():
            raise RunCancelled()
        if self.remaining_seconds <= 0:
            raise BudgetExceeded("time", f"run time limit of {self.limits.run_timeout_seconds}s reached")
        if self.usage.total_tokens >= self.limits.max_total_tokens:
            raise BudgetExceeded("tokens", f"token limit of {self.limits.max_total_tokens} reached")
        if self.estimated_cost_usd >= self.limits.max_cost_usd:
            raise BudgetExceeded(
                "cost",
                f"estimated model spend ${self.estimated_cost_usd:.2f} reached the ${self.limits.max_cost_usd:.2f} cap",
            )

    def command_timeout(self) -> float:
        self.check()
        return max(1.0, min(float(self.limits.command_timeout_seconds), self.remaining_seconds))

    def snapshot(self) -> dict:
        return {
            "elapsed_seconds": round(self.elapsed_seconds, 1),
            "model_calls": self.model_calls,
            "input_tokens": self.usage.input_tokens,
            "output_tokens": self.usage.output_tokens,
            "cache_write_tokens": self.usage.cache_write_tokens,
            "cache_read_tokens": self.usage.cache_read_tokens,
            "estimated_cost_usd": round(self.estimated_cost_usd, 4),
        }
