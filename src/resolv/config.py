"""Run configuration.

Configuration is supplied by the operator (CLI flags or a TOML file outside the
target repository). It is never read from the repository under repair, because
repository content is untrusted and must not be able to choose commands,
budgets, or policy.
"""

from __future__ import annotations

import shlex
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from resolv.errors import ConfigError


@dataclass(frozen=True)
class EnvironmentConfig:
    # Interpreter used to create the run venv (or used directly when venv is off).
    python: str = sys.executable
    venv: bool = True
    # Run in order inside the workspace after the venv is created, e.g.
    # "python -m pip install -e .[test]". Split with shlex; no shell is used.
    install_commands: tuple[str, ...] = ()
    # Must invoke pytest and accept extra arguments (plugins, node ids).
    test_command: str = "python -m pytest"
    # Host environment variables passed to commands (e.g. proxy settings).
    pass_env: tuple[str, ...] = ()


@dataclass(frozen=True)
class Limits:
    max_changed_files: int = 5
    max_changed_lines: int = 300
    max_patch_attempts: int = 3
    max_reproduction_attempts: int = 3
    max_turns_per_stage: int = 40
    run_timeout_seconds: int = 1800
    command_timeout_seconds: int = 600
    max_total_tokens: int = 3_000_000
    max_cost_usd: float = 3.0
    max_command_output_chars: int = 200_000
    max_tool_output_chars: int = 12_000
    max_issue_chars: int = 30_000


@dataclass(frozen=True)
class ModelConfig:
    provider: str = "openrouter"
    name: str = "anthropic/claude-sonnet-5.5"
    max_output_tokens: int = 8192
    # Fallback prices for the cost cap when the provider does not report a cost.
    input_cost_per_mtok: float = 2.0
    output_cost_per_mtok: float = 10.0
    request_timeout_seconds: int = 300
    # OpenRouter routing: "deny" only uses providers that do not store or train on prompts.
    data_collection: str = "deny"
    zero_data_retention: bool = False


@dataclass(frozen=True)
class RunConfig:
    environment: EnvironmentConfig = field(default_factory=EnvironmentConfig)
    limits: Limits = field(default_factory=Limits)
    model: ModelConfig = field(default_factory=ModelConfig)

    def validate(self) -> RunConfig:
        try:
            argv = shlex.split(self.environment.test_command)
        except ValueError as exc:
            raise ConfigError(f"test_command cannot be parsed: {exc}") from exc
        if not argv:
            raise ConfigError("test_command must not be empty")
        for command in self.environment.install_commands:
            try:
                if not shlex.split(command):
                    raise ConfigError("install_commands must not contain empty commands")
            except ValueError as exc:
                raise ConfigError(f"install command cannot be parsed: {command!r}: {exc}") from exc
        for f in fields(self.limits):
            if getattr(self.limits, f.name) <= 0:
                raise ConfigError(f"limits.{f.name} must be positive")
        for name in ("max_output_tokens", "request_timeout_seconds"):
            if getattr(self.model, name) <= 0:
                raise ConfigError(f"model.{name} must be positive")
        if self.model.data_collection not in ("allow", "deny"):
            raise ConfigError("model.data_collection must be 'allow' or 'deny'")
        for name in ("input_cost_per_mtok", "output_cost_per_mtok"):
            if getattr(self.model, name) < 0:
                raise ConfigError(f"model.{name} must not be negative")
        return self


def _coerce(section: str, key: str, default: Any, value: Any) -> Any:
    where = f"[{section}].{key}"
    if isinstance(default, bool):
        if not isinstance(value, bool):
            raise ConfigError(f"{where} must be a boolean")
        return value
    if isinstance(default, int):
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{where} must be an integer")
        return value
    if isinstance(default, float):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{where} must be a number")
        return float(value)
    if isinstance(default, tuple):
        if isinstance(value, str):
            value = [value]
        if not isinstance(value, list) or not all(isinstance(v, str) for v in value):
            raise ConfigError(f"{where} must be a list of strings")
        return tuple(value)
    if isinstance(default, str):
        if not isinstance(value, str) or not value.strip():
            raise ConfigError(f"{where} must be a non-empty string")
        return value
    raise ConfigError(f"{where} has an unsupported type")  # pragma: no cover


def _section(cls: type, data: Any, section: str) -> Any:
    if not isinstance(data, dict):
        raise ConfigError(f"[{section}] must be a table")
    defaults = cls()
    known = {f.name for f in fields(cls)}
    unknown = sorted(set(data) - known)
    if unknown:
        raise ConfigError(f"unknown key(s) in [{section}]: {', '.join(unknown)}")
    kwargs = {k: _coerce(section, k, getattr(defaults, k), v) for k, v in data.items()}
    return cls(**kwargs)


def load_config(path: Path | None) -> RunConfig:
    if path is None:
        return RunConfig().validate()
    try:
        data = tomllib.loads(path.read_text(encoding="utf-8"))
    except OSError as exc:
        raise ConfigError(f"cannot read config file {path}: {exc}") from exc
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(f"invalid TOML in {path}: {exc}") from exc
    unknown = sorted(set(data) - {"environment", "limits", "model"})
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    return RunConfig(
        environment=_section(EnvironmentConfig, data.get("environment", {}), "environment"),
        limits=_section(Limits, data.get("limits", {}), "limits"),
        model=_section(ModelConfig, data.get("model", {}), "model"),
    ).validate()
