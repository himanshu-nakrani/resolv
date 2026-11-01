"""Local backend: commands run on the host with a scrubbed environment.

Use only for repositories you trust. There is no network isolation, and test
runs share one checkout, so a test can leave files behind for later runs.
"""

from __future__ import annotations

import shlex
import shutil
from pathlib import Path

from resolv.budget import Budget
from resolv.config import RunConfig
from resolv.events import EventLog
from resolv.executor import CommandResult, CommandRunner, build_env
from resolv.redact import Redactor
from resolv.sandbox import PLUGIN_FILE, SetupError
from resolv.workspace import Workspace


class LocalSandbox:
    backend = "local"

    def __init__(
        self,
        *,
        config: RunConfig,
        workspace: Workspace,
        temp_dir: Path,
        run_id: str,
        budget: Budget,
        events: EventLog,
        logs_dir: Path,
        redactor: Redactor,
    ) -> None:
        self.env_cfg = config.environment
        self.ws = workspace
        self.temp = temp_dir
        home, bin_dir, self.venv = temp_dir / "home", temp_dir / "bin", temp_dir / "venv"
        self.results_dir = temp_dir / "results"
        for d in (home, bin_dir, self.results_dir):
            d.mkdir(exist_ok=True)
        if not self.env_cfg.venv:
            # Shims make `python` in configured commands mean the configured interpreter.
            interpreter = shutil.which(self.env_cfg.python) or self.env_cfg.python
            for name in ("python", "python3"):
                shim = bin_dir / name
                shim.write_text(f'#!/bin/sh\nexec {shlex.quote(interpreter)} "$@"\n')
                shim.chmod(0o755)
        prefix = [self.venv / "bin"] if self.env_cfg.venv else [bin_dir]
        env = build_env(
            home=home, path_prefix=prefix, venv=self.venv if self.env_cfg.venv else None, pass_env=self.env_cfg.pass_env
        )
        self.runner = CommandRunner(
            cwd=workspace.root, env=env, budget=budget, events=events, logs_dir=logs_dir, redactor=redactor,
            max_output_chars=config.limits.max_command_output_chars,
        )
        self.repo_root = str(workspace.root)
        self.plugin_dir = str(PLUGIN_FILE.parent)

    def setup(self) -> str:
        if self.env_cfg.venv:
            created = self.runner.run([self.env_cfg.python, "-m", "venv", str(self.venv)], label="setup-venv")
            if not created.ok:
                raise SetupError(
                    "install_failed",
                    f"Creating the virtual environment failed ({created.describe_status()}); see {created.log_path}.",
                )
        for i, command in enumerate(self.env_cfg.install_commands, 1):
            result = self.runner.run(shlex.split(command), label=f"setup-install-{i}")
            if not result.ok:
                raise SetupError(
                    "install_failed",
                    f"Install command {command!r} failed ({result.describe_status()}); see {result.log_path}. "
                    "This is an environment problem, not a code failure.",
                )
        modified = self.ws.tracked_modifications()
        if modified:
            raise SetupError("environment_error", f"Environment setup modified tracked files: {', '.join(modified[:5])}.")
        ignored = self.ws.ignore_untracked()
        return f"Local environment ready; {len(ignored)} setup artifacts excluded from diffs"

    def results_path(self, name: str) -> str:
        return str(self.results_dir / name)

    def run(
        self, argv: list[str], *, label: str, env: dict[str, str] | None = None, collect: str | None = None
    ) -> tuple[CommandResult, bytes | None]:
        if collect:
            Path(collect).unlink(missing_ok=True)
        result = self.runner.run(argv, label=label, extra_env=env)
        data = None
        if collect and Path(collect).is_file():
            data = Path(collect).read_bytes()
            Path(collect).unlink()
        return result, data

    def describe(self) -> dict:
        return {
            "backend": "local",
            "network": "unrestricted",
            "isolation": "none: commands run on this machine as the current user with a scrubbed environment",
        }

    def close(self) -> None:
        pass
