"""Command execution with a scrubbed environment, timeouts, and output caps.

Phase 1 runs commands on the local machine. Commands get a minimal
environment (no API keys or tokens from the host), a private HOME, their own
process group so timeouts and cancellation kill the whole tree, and capped
output. Network isolation and VM-level sandboxing are Phase 3 work.
"""

from __future__ import annotations

import itertools
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from resolv.budget import Budget
from resolv.errors import RunCancelled
from resolv.events import EventLog
from resolv.redact import Redactor

_SAFE_HOST_VARS = ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")
_POLL_SECONDS = 0.25


@dataclass
class CommandResult:
    label: str
    argv: list[str]
    exit_code: int | None
    duration_seconds: float
    output: str
    timed_out: bool = False
    output_truncated: bool = False
    error: str | None = None
    log_path: str | None = None

    @property
    def ok(self) -> bool:
        return self.exit_code == 0 and not self.timed_out and self.error is None

    def describe_status(self) -> str:
        if self.error:
            return self.error
        if self.timed_out:
            return f"timed out after {self.duration_seconds:.0f}s"
        return f"exit code {self.exit_code}"

    def summary(self) -> dict:
        data = asdict(self)
        data.pop("output")
        data["duration_seconds"] = round(self.duration_seconds, 2)
        return data


def build_env(*, home: Path, path_prefix: list[Path], venv: Path | None, pass_env: tuple[str, ...] = ()) -> dict[str, str]:
    env = {k: os.environ[k] for k in (*_SAFE_HOST_VARS, *pass_env) if k in os.environ}
    env["PATH"] = os.pathsep.join([*(str(p) for p in path_prefix), os.environ.get("PATH", os.defpath)])
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if venv is not None:
        env["VIRTUAL_ENV"] = str(venv)
    return env


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


def _read_capped(fh, max_chars: int) -> tuple[str, bool]:
    fh.seek(0, os.SEEK_END)
    size = fh.tell()
    fh.seek(0)
    if size <= max_chars:
        return fh.read().decode("utf-8", errors="replace"), False
    half = max_chars // 2
    head = fh.read(half).decode("utf-8", errors="replace")
    fh.seek(size - half)
    tail = fh.read().decode("utf-8", errors="replace")
    return f"{head}\n\n... [{size - 2 * half} bytes of output truncated] ...\n\n{tail}", True


class CommandRunner:
    def __init__(
        self,
        *,
        cwd: Path,
        env: dict[str, str],
        budget: Budget,
        events: EventLog,
        logs_dir: Path,
        redactor: Redactor,
        max_output_chars: int,
    ) -> None:
        self.cwd = cwd
        self.env = env
        self.budget = budget
        self.events = events
        self.logs_dir = logs_dir
        self.redact = redactor
        self.max_output_chars = max_output_chars
        self.history: list[CommandResult] = []
        self._counter = itertools.count(1)

    def run(self, argv: list[str], *, label: str, extra_env: dict[str, str] | None = None) -> CommandResult:
        timeout = self.budget.command_timeout()
        env = {**self.env, **(extra_env or {})}
        executable = shutil.which(argv[0], path=env.get("PATH"))
        start = time.monotonic()
        if executable is None:
            result = CommandResult(label, list(argv), None, 0.0, "", error=f"command not found: {argv[0]}")
            return self._finish(result)

        timed_out = False
        with tempfile.TemporaryFile() as out:
            proc = subprocess.Popen(
                [executable, *argv[1:]],
                cwd=self.cwd,
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=out,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
            deadline = start + timeout
            try:
                while True:
                    try:
                        proc.wait(timeout=_POLL_SECONDS)
                        break
                    except subprocess.TimeoutExpired:
                        if self.budget.cancel_event.is_set():
                            raise RunCancelled() from None
                        if time.monotonic() >= deadline:
                            timed_out = True
                            _kill_group(proc)
                            break
            except BaseException:
                _kill_group(proc)
                self.events.emit("command_aborted", label=label, argv=argv)
                raise
            output, truncated = _read_capped(out, self.max_output_chars)

        result = CommandResult(
            label=label,
            argv=list(argv),
            exit_code=None if timed_out else proc.returncode,
            duration_seconds=time.monotonic() - start,
            output=self.redact(output),
            timed_out=timed_out,
            output_truncated=truncated,
        )
        return self._finish(result)

    def _finish(self, result: CommandResult) -> CommandResult:
        n = next(self._counter)
        safe_label = re.sub(r"[^A-Za-z0-9_.-]+", "-", result.label)[:60]
        log = self.logs_dir / f"{n:03d}-{safe_label}.log"
        log.write_text(
            f"$ {' '.join(result.argv)}\n{result.output}\n[{result.describe_status()}; "
            f"{result.duration_seconds:.2f}s]\n",
            encoding="utf-8",
        )
        result.log_path = str(log.relative_to(self.logs_dir.parent))
        self.history.append(result)
        self.events.emit("command", **result.summary())
        return result
