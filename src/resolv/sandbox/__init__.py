"""Execution sandboxes: where repository code (dependency installs and tests) runs.

The orchestrator keeps the authoritative checkout on the trusted side and
never executes it there. A sandbox receives the files it needs, runs commands,
and hands back output plus the pytest results file.

- ``DockerSandbox`` (default): every test command runs in a fresh, disposable
  container created from a snapshot of the installed environment, with no
  network, no credentials, no host mounts, no Docker socket, and CPU, memory,
  process, and file-size limits. Dependency installation runs earlier in a
  separate setup container whose only network route is an allowlisting proxy
  to the package index.
- ``LocalSandbox``: runs commands on the host with a scrubbed environment. For
  trusted repositories and development only.
"""

from __future__ import annotations

from pathlib import Path
from typing import Protocol

from resolv.budget import Budget
from resolv.config import RunConfig
from resolv.events import EventLog
from resolv.executor import CommandResult, CommandRunner
from resolv.redact import Redactor
from resolv.workspace import Workspace

PLUGIN_FILE = Path(__file__).resolve().parent.parent / "_pytest_plugin" / "resolv_pytest_results.py"


class SetupError(Exception):
    """Environment setup failed; ``code`` is a run reason code."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


class Sandbox(Protocol):
    backend: str
    runner: CommandRunner
    # Paths as seen by commands inside the sandbox.
    repo_root: str
    plugin_dir: str

    def setup(self) -> str:
        """Prepare the environment and run install commands; returns a summary or raises SetupError."""

    def results_path(self, name: str) -> str:
        """Where a command inside the sandbox should write a results file called ``name``."""

    def run(
        self, argv: list[str], *, label: str, env: dict[str, str] | None = None, collect: str | None = None
    ) -> tuple[CommandResult, bytes | None]:
        """Run ``argv`` against the current workspace; returns the result and the bytes of ``collect``, if any."""

    def describe(self) -> dict:
        """Isolation properties, for reports."""

    def close(self) -> None: ...


def create_sandbox(
    config: RunConfig,
    *,
    workspace: Workspace,
    temp_dir: Path,
    run_id: str,
    budget: Budget,
    events: EventLog,
    logs_dir: Path,
    redactor: Redactor,
) -> Sandbox:
    common = dict(
        config=config, workspace=workspace, temp_dir=temp_dir, run_id=run_id, budget=budget, events=events,
        logs_dir=logs_dir, redactor=redactor,
    )
    if config.sandbox.backend == "docker":
        from resolv.sandbox.docker import DockerSandbox

        return DockerSandbox(**common)
    from resolv.sandbox.local import LocalSandbox

    return LocalSandbox(**common)
