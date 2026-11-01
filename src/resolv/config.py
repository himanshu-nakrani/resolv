"""Run configuration.

Configuration is supplied by the operator (CLI flags or a TOML file outside the
target repository). It is never read from the repository under repair, because
repository content is untrusted and must not be able to choose commands,
budgets, or policy.
"""

from __future__ import annotations

import re
import shlex
import sys
import tomllib
from dataclasses import dataclass, field, fields
from pathlib import Path
from typing import Any

from resolv.errors import ConfigError


@dataclass(frozen=True)
class EnvironmentConfig:
    # Python version for the Docker sandbox image (fills {python_version} in sandbox.image).
    python_version: str = "3.12"
    # Local sandbox only: interpreter used to create the run venv (or used directly when venv is off).
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
class SandboxConfig:
    # "docker" (isolated, default) or "local" (host execution; trusted repositories only).
    backend: str = "docker"
    image: str = "python:{python_version}"
    cpus: float = 2.0
    memory_mb: int = 4096
    pids_limit: int = 512
    file_size_limit_mb: int = 1024
    # Container writable-layer cap such as "10G"; needs a storage driver that supports --storage-opt size.
    storage_size: str = ""
    # Alternative OCI runtime such as "runsc" (gVisor) for stronger isolation.
    runtime: str = ""
    # Setup-phase network: "proxy" (allowlisted hosts only) or "none".
    setup_network: str = "proxy"
    allowed_hosts: tuple[str, ...] = ("pypi.org", "files.pythonhosted.org")
    user: str = "1000:1000"
    docker: str = "docker"


_HOST_RE = re.compile(r"^(?=.{1,253}$)(?!-)[a-z0-9-]{1,63}(?<!-)(\.(?!-)[a-z0-9-]{1,63}(?<!-))*$")


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
    sandbox: SandboxConfig = field(default_factory=SandboxConfig)

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
        if not re.fullmatch(r"\d+\.\d+", self.environment.python_version):
            raise ConfigError("environment.python_version must look like '3.12'")
        self._validate_sandbox()
        if self.model.data_collection not in ("allow", "deny"):
            raise ConfigError("model.data_collection must be 'allow' or 'deny'")
        for name in ("input_cost_per_mtok", "output_cost_per_mtok"):
            if getattr(self.model, name) < 0:
                raise ConfigError(f"model.{name} must not be negative")
        return self

    def _validate_sandbox(self) -> None:
        s = self.sandbox
        if s.backend not in ("docker", "local"):
            raise ConfigError("sandbox.backend must be 'docker' or 'local'")
        if s.setup_network not in ("proxy", "none"):
            raise ConfigError("sandbox.setup_network must be 'proxy' or 'none'")
        if s.cpus <= 0 or s.memory_mb < 128 or s.pids_limit < 16 or s.file_size_limit_mb < 1:
            raise ConfigError("sandbox limits are too small (cpus > 0, memory_mb >= 128, pids_limit >= 16, file_size_limit_mb >= 1)")
        if not re.fullmatch(r"\d+(:\d+)?", s.user):
            raise ConfigError("sandbox.user must be a numeric uid or uid:gid")
        if not s.image or any(ch.isspace() for ch in s.image):
            raise ConfigError("sandbox.image must be an image reference without spaces")
        for host in s.allowed_hosts:
            if not _HOST_RE.match(host):
                raise ConfigError(f"sandbox.allowed_hosts entry is not a host name: {host!r}")


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
        if not isinstance(value, str) or (default and not value.strip()):
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
    return config_from_dict(data)


def config_from_dict(data: dict) -> RunConfig:
    unknown = sorted(set(data) - {"environment", "limits", "model", "sandbox"})
    if unknown:
        raise ConfigError(f"unknown section(s): {', '.join(unknown)}")
    return RunConfig(
        environment=_section(EnvironmentConfig, data.get("environment", {}), "environment"),
        limits=_section(Limits, data.get("limits", {}), "limits"),
        model=_section(ModelConfig, data.get("model", {}), "model"),
        sandbox=_section(SandboxConfig, data.get("sandbox", {}), "sandbox"),
    ).validate()
