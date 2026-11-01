"""Command execution with timeouts, cancellation, and output caps.

The runner only drives processes on the orchestrator host. Where those
processes run repository code is decided by the sandbox (see resolv.sandbox):
directly on the host for the local backend, or via ``docker`` for the
isolated backend. Output is capped while it streams, so a noisy or malicious
command cannot exhaust host memory or disk.
"""

from __future__ import annotations

import collections
import itertools
import os
import re
import shutil
import signal
import subprocess
import threading
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from pathlib import Path

from resolv.budget import Budget
from resolv.errors import RunCancelled
from resolv.events import EventLog
from resolv.redact import Redactor

_SAFE_HOST_VARS = ("LANG", "LC_ALL", "LC_CTYPE", "TZ", "TERM")
_POLL_SECONDS = 0.25
_CHUNK = 65536


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
    """Minimal environment for commands run directly on the host (local backend)."""
    env = {k: os.environ[k] for k in (*_SAFE_HOST_VARS, *pass_env) if k in os.environ}
    env["PATH"] = os.pathsep.join([*(str(p) for p in path_prefix), os.environ.get("PATH", os.defpath)])
    env["HOME"] = str(home)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    if venv is not None:
        env["VIRTUAL_ENV"] = str(venv)
    return env


class CappedOutput:
    """Keeps the first and last ``limit / 2`` bytes of a stream."""

    def __init__(self, limit: int) -> None:
        self.half = max(1, limit // 2)
        self.head = bytearray()
        self.tail: collections.deque[bytes] = collections.deque()
        self.tail_size = 0
        self.total = 0

    def feed(self, data: bytes) -> None:
        self.total += len(data)
        room = self.half - len(self.head)
        if room > 0:
            self.head += data[:room]
            data = data[room:]
        if not data:
            return
        self.tail.append(data)
        self.tail_size += len(data)
        while self.tail and self.tail_size - len(self.tail[0]) >= self.half:
            self.tail_size -= len(self.tail.popleft())

    def text(self) -> tuple[str, bool]:
        tail = b"".join(self.tail)
        if len(tail) > self.half:
            tail = tail[-self.half :]
        omitted = self.total - len(self.head) - len(tail)
        head = self.head.decode("utf-8", errors="replace")
        if omitted <= 0:
            return head + tail.decode("utf-8", errors="replace"), False
        marker = f"\n\n... [{omitted} bytes of output truncated] ...\n\n"
        return head + marker + tail.decode("utf-8", errors="replace"), True


def _kill_group(proc: subprocess.Popen) -> None:
    try:
        os.killpg(proc.pid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass
    try:
        proc.wait(timeout=10)
    except subprocess.TimeoutExpired:  # pragma: no cover
        proc.kill()


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

    def run(
        self,
        argv: list[str],
        *,
        label: str,
        extra_env: dict[str, str] | None = None,
        exec_argv: list[str] | None = None,
        exec_env: dict[str, str] | None = None,
        kill: Callable[[], None] | None = None,
        on_exit: Callable[[CommandResult], None] | None = None,
        timeout: float | None = None,
    ) -> CommandResult:
        """Run a command; ``argv`` is what gets recorded, ``exec_argv`` what actually executes.

        ``kill`` is called on timeout or cancellation, e.g. to stop a container
        that outlives the client process. ``on_exit`` can amend the result
        before it is recorded.
        """
        limit = self.budget.command_timeout()
        if timeout is not None:
            limit = min(limit, timeout)
        if exec_argv is None:
            env = {**self.env, **(extra_env or {})}
            exec_argv = list(argv)
        else:
            env = dict(exec_env if exec_env is not None else self.env)
        start = time.monotonic()
        executable = shutil.which(exec_argv[0], path=env.get("PATH"))
        if executable is None:
            return self._finish(CommandResult(label, list(argv), None, 0.0, "", error=f"command not found: {exec_argv[0]}"))

        capped = CappedOutput(self.max_output_chars)
        proc = subprocess.Popen(
            [executable, *exec_argv[1:]],
            cwd=self.cwd,
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            start_new_session=True,
        )

        def pump() -> None:
            for chunk in iter(lambda: proc.stdout.read1(_CHUNK), b""):
                capped.feed(chunk)

        reader = threading.Thread(target=pump, daemon=True)
        reader.start()
        timed_out = False
        deadline = start + limit

        def stop() -> None:
            if kill is not None:
                try:
                    kill()
                except Exception:  # noqa: BLE001 - best effort
                    pass
            _kill_group(proc)

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
                        stop()
                        break
        except BaseException:
            stop()
            self.events.emit("command_aborted", label=label, argv=argv)
            raise
        finally:
            reader.join(timeout=10)
            proc.stdout.close()

        output, truncated = capped.text()
        result = CommandResult(
            label=label,
            argv=list(argv),
            exit_code=None if timed_out else proc.returncode,
            duration_seconds=time.monotonic() - start,
            output=self.redact(output),
            timed_out=timed_out,
            output_truncated=truncated,
        )
        if on_exit is not None and not timed_out:
            on_exit(result)
        return self._finish(result)

    def record(
        self, argv: list[str], *, label: str, output: str = "", exit_code: int | None = 0, error: str | None = None
    ) -> CommandResult:
        """Record an entry that did not go through run(), e.g. sandbox errors or collected service logs."""
        capped = CappedOutput(self.max_output_chars)
        capped.feed(output.encode("utf-8", errors="replace"))
        text, truncated = capped.text()
        return self._finish(
            CommandResult(label, list(argv), exit_code, 0.0, self.redact(text), output_truncated=truncated, error=error)
        )

    def record_error(self, argv: list[str], *, label: str, error: str, output: str = "") -> CommandResult:
        return self.record(argv, label=label, output=output, exit_code=None, error=error)

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
