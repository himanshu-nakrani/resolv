from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

from resolv.budget import Budget
from resolv.config import EnvironmentConfig, Limits, RunConfig, SandboxConfig
from resolv.events import EventLog
from resolv.model.scripted import ScriptedModel, call, respond
from resolv.orchestrator import Issue, Orchestrator, RunRequest
from resolv.redact import Redactor
from resolv.sandbox import create_sandbox
from resolv.testrun import PytestRunner
from resolv.workspace import Workspace

BUGGY_LINE = "    return year % 4 == 0\n"
FIXED_LINE = "    return year % 4 == 0 and (year % 100 != 0 or year % 400 == 0)\n"

TOY_FILES = {
    "pytest.ini": "[pytest]\n",
    "calendar_utils.py": "def is_leap_year(year):\n" + BUGGY_LINE,
    "tests/test_calendar_utils.py": (
        "from calendar_utils import is_leap_year\n\n\n"
        "def test_regular_leap_year():\n    assert is_leap_year(2024)\n\n\n"
        "def test_common_year():\n    assert not is_leap_year(2023)\n"
    ),
}

ISSUE_TEXT = "# is_leap_year(1900) returns True\n\n1900 was not a leap year, but `is_leap_year(1900)` returns True.\n"

REGRESSION_PATH = "tests/test_leap_regression.py"
REGRESSION_TEST = (
    "from calendar_utils import is_leap_year\n\n\n"
    "def test_century_is_not_leap_year():\n    assert not is_leap_year(1900)\n"
)
REGRESSION_ID = f"{REGRESSION_PATH}::test_century_is_not_leap_year"


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True).stdout


def make_repo(root: Path, files: dict[str, str]) -> Path:
    root.mkdir(parents=True)
    for rel, content in files.items():
        path = root / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    git(root, "init", "-q", "-b", "main")
    git(root, "add", "-A")
    git(root, "-c", "user.name=t", "-c", "user.email=t@example.com", "commit", "-qm", "init")
    return root


def make_config(*, environment: dict | None = None, sandbox: dict | None = None, **limits) -> RunConfig:
    env = {"python": sys.executable, "venv": False, **(environment or {})}
    return RunConfig(
        environment=EnvironmentConfig(**env),
        limits=Limits(**limits),
        sandbox=SandboxConfig(**{"backend": "local", **(sandbox or {})}),
    ).validate()


def run_resolv(repo: Path, steps, out: Path, *, config: RunConfig | None = None, issue: str = ISSUE_TEXT,
               check_only: bool = False, environ: dict | None = None, setup=None):
    model = ScriptedModel(steps)
    request = RunRequest(
        repo=repo,
        issue=None if check_only else Issue.from_text(issue, url="https://example.com/issues/1"),
        config=config or make_config(),
        out_dir=out,
        check_only=check_only,
    )
    orchestrator = Orchestrator(request, client=model, environ=environ or {})
    if setup:
        setup(orchestrator, model)
    record = orchestrator.run()
    return record, orchestrator.run_dir, model


# -- reusable scripted stages ----------------------------------------------

def investigate_steps():
    return [
        respond(call("read_file", path="calendar_utils.py")),
        respond(
            call(
                "submit_diagnosis",
                summary="is_leap_year treats century years as leap years.",
                cause="calendar_utils.is_leap_year only checks divisibility by 4.",
                relevant_files=["calendar_utils.py"],
                reproduction_plan="Assert that 1900 is not a leap year.",
            )
        ),
    ]


def reproduce_steps(content: str = REGRESSION_TEST, path: str = REGRESSION_PATH, node_id: str = REGRESSION_ID):
    return [
        respond(call("write_file", path=path, content=content)),
        respond(call("run_tests", node_ids=[path])),
        respond(call("submit_reproduction", test_node_ids=[node_id], expected_failure="AssertionError: 1900 is reported as a leap year")),
    ]


def patch_steps(new_line: str = FIXED_LINE):
    return [
        respond(call("replace_in_file", path="calendar_utils.py", old_text=BUGGY_LINE, new_text=new_line)),
        respond(call("submit_patch", changes="Apply the Gregorian century rule.", assumptions_and_risks="None known.")),
    ]


def tool_results(model: ScriptedModel, request_index: int) -> list[dict]:
    """Tool results the model received in a given request."""
    content = model.requests[request_index]["messages"][-1]["content"]
    return [b for b in content if isinstance(b, dict) and b.get("type") == "tool_result"]


@pytest.fixture
def toy_repo(tmp_path: Path) -> Path:
    return make_repo(tmp_path / "toy", TOY_FILES)


@pytest.fixture
def make_pytest_runner(tmp_path: Path):
    """Factory: PytestRunner over a repo, using a local sandbox and the current interpreter."""

    def factory(repo: Path) -> PytestRunner:
        sandbox_dir = tmp_path / "sandbox"
        logs = tmp_path / "logs"
        for d in (sandbox_dir, logs):
            d.mkdir(exist_ok=True)
        config = make_config()
        sandbox = create_sandbox(
            config,
            workspace=Workspace(repo, "HEAD"),
            temp_dir=sandbox_dir,
            run_id="test",
            budget=Budget(config.limits, config.model),
            events=EventLog(tmp_path / "events.jsonl", Redactor()),
            logs_dir=logs,
            redactor=Redactor(),
        )
        return PytestRunner(sandbox, "python -m pytest")

    return factory
