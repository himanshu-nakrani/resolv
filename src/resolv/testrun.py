"""Running pytest and interpreting its results."""

from __future__ import annotations

import itertools
import json
import shlex
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path, PurePosixPath
from typing import TYPE_CHECKING

from resolv.executor import CommandResult

if TYPE_CHECKING:
    from resolv.sandbox import Sandbox

PLUGIN_NAME = "resolv_pytest_results"
FAILING = frozenset({"failed", "error"})

# Exceptions that indicate a broken test environment or test file rather than
# the reported behaviour.
ENVIRONMENT_EXCEPTIONS = frozenset(
    {"ImportError", "ModuleNotFoundError", "SyntaxError", "IndentationError", "TabError"}
)
# Raised inside the test file itself these usually mean the test is wrong.
TEST_AUTHORING_EXCEPTIONS = frozenset({"NameError", "UnboundLocalError"})

_PYTEST_EXIT = {
    0: "all tests passed",
    1: "some tests failed",
    2: "interrupted",
    3: "internal error",
    4: "usage error",
    5: "no tests collected",
}


@dataclass
class TestOutcome:
    __test__ = False  # not a pytest test class

    nodeid: str
    outcome: str
    when: str
    exc_type: str | None = None
    crash_path: str | None = None
    message: str = ""
    details: str = ""

    @property
    def failing(self) -> bool:
        return self.outcome in FAILING

    def headline(self) -> str:
        parts = [f"{self.nodeid}: {self.outcome}"]
        if self.when not in ("call", ""):
            parts.append(f"during {self.when}")
        if self.exc_type:
            parts.append(f"({self.exc_type})")
        if self.message:
            parts.append(f"- {self.message.strip().splitlines()[0][:300]}")
        return " ".join(parts)

    def to_dict(self, *, details: bool = True) -> dict:
        data = asdict(self)
        if not details:
            data.pop("details")
        return data


@dataclass
class TestRun:
    __test__ = False

    label: str
    command: CommandResult
    exit_status: int | None = None
    tests: dict[str, TestOutcome] = field(default_factory=dict)
    collection_errors: list[TestOutcome] = field(default_factory=list)
    environment_error: str | None = None

    @property
    def usable(self) -> bool:
        return self.environment_error is None

    def failing(self) -> list[TestOutcome]:
        return [*self.collection_errors, *(t for t in self.tests.values() if t.failing)]

    def counts(self) -> dict[str, int]:
        counts = Counter(t.outcome for t in self.tests.values())
        if self.collection_errors:
            counts["collection_error"] = len(self.collection_errors)
        return dict(sorted(counts.items()))

    def summary(self) -> str:
        if self.environment_error:
            return f"did not run: {self.environment_error}"
        counts = self.counts()
        if not counts:
            return "no tests collected"
        return ", ".join(f"{n} {k}" for k, n in counts.items())

    def to_dict(self) -> dict:
        return {
            "label": self.label,
            "command": self.command.summary(),
            "exit_status": self.exit_status,
            "environment_error": self.environment_error,
            "counts": self.counts(),
            "failing": [t.to_dict() for t in self.failing()],
        }


def describe_run(run: TestRun, *, max_failures: int = 10, max_detail_chars: int = 1500) -> str:
    """Human/model-readable description of a test run."""
    lines = [f"Result: {run.summary()} ({run.command.describe_status()})"]
    if run.environment_error:
        lines.append("The test command did not produce usable results. Output tail:")
        lines.append(run.command.output[-3000:])
        return "\n".join(lines)
    failing = run.failing()
    for outcome in failing[:max_failures]:
        lines.append(f"\n- {outcome.headline()}")
        if outcome.details:
            lines.append(outcome.details[-max_detail_chars:])
    if len(failing) > max_failures:
        lines.append(f"\n... and {len(failing) - max_failures} more failing tests")
    others = [t for t in run.tests.values() if not t.failing]
    if others and len(others) <= 30:
        lines.append("\nOther tests: " + ", ".join(f"{t.nodeid} ({t.outcome})" for t in others))
    elif others:
        lines.append(f"\n{len(others)} other tests did not fail.")
    return "\n".join(lines)


def reproduction_problem(outcome: TestOutcome, test_paths: set[str]) -> str | None:
    """Why a failing test does not count as reproducing the issue, or None if it does."""
    if outcome.outcome != "failed" or outcome.when != "call":
        phase = f" during {outcome.when}" if outcome.when != "call" else ""
        return f"{outcome.nodeid} did not fail in the test body (outcome: {outcome.outcome}{phase})"
    if outcome.exc_type in ENVIRONMENT_EXCEPTIONS:
        return (
            f"{outcome.nodeid} failed with {outcome.exc_type}, which indicates an import, dependency, or syntax "
            "problem rather than the reported behaviour"
        )
    if outcome.exc_type in TEST_AUTHORING_EXCEPTIONS and outcome.crash_path in test_paths:
        return f"{outcome.nodeid} failed with {outcome.exc_type} raised inside the test itself (a test bug)"
    return None


@dataclass
class SuiteComparison:
    new_failures: list[TestOutcome]
    pre_existing_failures: list[TestOutcome]
    fixed: list[str]
    missing: list[str]

    @property
    def clean(self) -> bool:
        return not self.new_failures and not self.missing

    def to_dict(self) -> dict:
        return {
            "new_failures": [t.to_dict(details=False) for t in self.new_failures],
            "pre_existing_failures": [t.to_dict(details=False) for t in self.pre_existing_failures],
            "fixed": self.fixed,
            "missing": self.missing,
        }


def compare_runs(baseline: TestRun, after: TestRun) -> SuiteComparison:
    baseline_failing = {t.nodeid for t in baseline.failing()}
    new, pre_existing = [], []
    for outcome in after.failing():
        (pre_existing if outcome.nodeid in baseline_failing else new).append(outcome)
    fixed = sorted(n for n in baseline_failing if n in after.tests and not after.tests[n].failing)
    missing = sorted(n for n in baseline.tests if n not in after.tests)
    return SuiteComparison(new, pre_existing, fixed, missing)


class PytestRunner:
    def __init__(self, sandbox: Sandbox, test_command: str) -> None:
        self.sandbox = sandbox
        self.base_argv = shlex.split(test_command)
        self._counter = itertools.count(1)

    def run(self, label: str, selectors: list[str] | tuple[str, ...] = ()) -> TestRun:
        results_path = self.sandbox.results_path(f"{next(self._counter):03d}-{label}.json")
        argv = [*self.base_argv, "-p", PLUGIN_NAME, "-p", "no:cacheprovider", *selectors]
        command, raw = self.sandbox.run(
            argv,
            label=label,
            env={"RESOLV_RESULTS_FILE": results_path, "PYTHONPATH": self.sandbox.plugin_dir},
            collect=results_path,
        )
        return self._parse(label, command, raw)

    def _relative(self, path: str | None) -> str | None:
        if not path:
            return None
        root = self.sandbox.repo_root
        for candidate, base in ((PurePosixPath(path), PurePosixPath(root)), (Path(path).resolve(), Path(root).resolve())):
            try:
                return candidate.relative_to(base).as_posix()
            except ValueError:
                continue
        return path

    def _outcome(self, raw: dict) -> TestOutcome:
        return TestOutcome(
            nodeid=raw["nodeid"],
            outcome=raw["outcome"],
            when=raw.get("when") or "",
            exc_type=raw.get("exc_type"),
            crash_path=self._relative(raw.get("crash_path")),
            message=raw.get("message") or "",
            details=raw.get("details") or "",
        )

    def _parse(self, label: str, command: CommandResult, raw: bytes | None) -> TestRun:
        run = TestRun(label=label, command=command)
        if command.error:
            run.environment_error = command.error
            return run
        if command.timed_out:
            run.environment_error = f"test command timed out after {command.duration_seconds:.0f}s"
            return run
        if raw is None:
            run.environment_error = (
                f"pytest produced no results ({command.describe_status()}); check that pytest is installed "
                "and that the test command invokes pytest"
            )
            return run
        try:
            data = json.loads(raw)
            run.exit_status = data.get("exitstatus")
            for item in data.get("tests", []):
                outcome = self._outcome(item)
                run.tests[outcome.nodeid] = outcome
            run.collection_errors = [self._outcome(item) for item in data.get("collect_errors", [])]
        except (ValueError, KeyError, TypeError, AttributeError) as exc:
            run.environment_error = f"the pytest results file could not be read ({exc})"
            return run
        status = run.exit_status
        if status in (3, 4):
            run.environment_error = f"pytest {_PYTEST_EXIT[status]} (exit {status})"
        elif status == 2 and not run.collection_errors:
            run.environment_error = "pytest run was interrupted (exit 2)"
        elif (problem := _consistency_problem(run, command)) is not None:
            run.environment_error = problem
        return run


def _consistency_problem(run: TestRun, command: CommandResult) -> str | None:
    """Detect results that disagree with the process exit code (e.g. a tampered results file)."""
    code = command.exit_code
    if code is None or code not in (0, 1):
        return None
    if run.exit_status is not None and run.exit_status != code:
        return f"pytest exited with code {code} but reported exit status {run.exit_status}; the results are not trustworthy"
    failing = bool(run.failing())
    if code == 0 and failing:
        return "pytest exited successfully but reported failing tests; the results are not trustworthy"
    if code == 1 and not failing:
        return "pytest exited with failures but reported none; the results are not trustworthy"
    return None
