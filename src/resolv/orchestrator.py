"""Workflow orchestrator: preflight -> setup -> baseline -> investigate -> reproduce -> patch/validate -> deliver.

The orchestrator, not the model, decides stage order, verifies every claim the
model makes by running tests itself, enforces policy and budgets, and decides
the outcome. Phase 1 "delivery" writes a patch file and reports instead of
opening a pull request.
"""

from __future__ import annotations

import os
import secrets
import shutil
import tempfile
import time
import traceback
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path

from resolv import policy, prompts, tools
from resolv.agent import AgentSession, FinalCall, Tool
from resolv.budget import Budget
from resolv.config import RunConfig
from resolv.errors import (
    AgentTurnLimit,
    BudgetExceeded,
    ConfigError,
    ModelError,
    RunCancelled,
    ToolError,
    WorkspaceError,
)
from resolv.events import EventLog, utc_now
from resolv.model import API_KEY_ENV, ModelClient, create_client
from resolv.redact import Redactor
from resolv.report import RunRecord, write_reports
from resolv.sandbox import Sandbox, SetupError, create_sandbox
from resolv.testrun import (
    PytestRunner,
    TestRun,
    compare_runs,
    describe_run,
    reproduction_problem,
)
from resolv.workspace import (
    Workspace,
    git,
    has_uncommitted_changes,
    is_git_repo,
    resolve_commit,
)


class Outcome(StrEnum):
    PATCH_READY = "patch_ready"  # Phase 1 stand-in for "Draft PR created"
    BLOCKED = "blocked"
    FAILED = "failed"
    CANCELLED = "cancelled"
    CHECKED = "checked"  # `resolv check`: setup and baseline only


class StopRun(Exception):
    def __init__(self, outcome: Outcome, code: str, message: str) -> None:
        super().__init__(message)
        self.outcome = outcome
        self.code = code
        self.message = message


@dataclass
class Issue:
    title: str
    body: str
    url: str | None = None
    source: str | None = None

    @classmethod
    def from_text(cls, text: str, *, title: str | None = None, url: str | None = None, source: str | None = None) -> Issue:
        if title is None:
            first = next((line.strip() for line in text.splitlines() if line.strip()), "")
            title = first.lstrip("#").strip()[:200] or "Untitled issue"
        return cls(title=title, body=text, url=url, source=source)


@dataclass
class RunRequest:
    repo: Path
    issue: Issue | None
    config: RunConfig
    out_dir: Path
    ref: str = "HEAD"
    check_only: bool = False
    keep_workspace: bool = False
    run_id: str | None = None


@dataclass
class Reproduction:
    node_ids: list[str]
    expected_failure: str
    run: TestRun
    commit: str
    test_diff: str


@dataclass
class _Context:
    """Mutable per-run state shared between stages."""

    workspace: Workspace | None = None
    sandbox: Sandbox | None = None
    pytest: PytestRunner | None = None
    baseline: TestRun | None = None
    diagnosis: dict | None = None
    reproduction: Reproduction | None = None
    temp_dir: Path | None = None
    warnings: list[str] = field(default_factory=list)


def new_run_id() -> str:
    return datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ") + "-" + secrets.token_hex(3)


def _format_diagnosis(diagnosis: dict) -> str:
    files = ", ".join(diagnosis.get("relevant_files") or [])
    return (
        f"Summary: {diagnosis.get('summary', '')}\nCause: {diagnosis.get('cause', '')}\n"
        f"Relevant files: {files}\nReproduction plan: {diagnosis.get('reproduction_plan', '')}"
    )


class Orchestrator:
    def __init__(
        self,
        request: RunRequest,
        *,
        client: ModelClient | None = None,
        environ: Mapping[str, str] | None = None,
        listeners: list | None = None,
    ) -> None:
        self.request = request
        self.config = request.config
        self.limits = request.config.limits
        self.environ = dict(os.environ if environ is None else environ)
        self.client = client
        self.run_id = request.run_id or new_run_id()
        self.run_dir = request.out_dir.resolve() / self.run_id
        self.budget = Budget(self.limits, self.config.model)
        self.listeners = list(listeners or [])
        self.ctx = _Context()
        issue = request.issue
        self.record = RunRecord(
            run_id=self.run_id,
            started_at=utc_now(),
            repo=str(request.repo),
            ref=request.ref,
            mode="check" if request.check_only else "run",
            issue={"title": issue.title, "url": issue.url, "source": issue.source} if issue else None,
            environment={
                "sandbox": self.config.sandbox.backend,
                "python_version": self.config.environment.python_version,
                "install_commands": list(self.config.environment.install_commands),
                "test_command": self.config.environment.test_command,
            },
            limits=vars(self.limits).copy(),
        )

    def cancel(self) -> None:
        """Request cancellation; safe to call from another thread."""
        self.budget.cancel_event.set()

    # -- top level -------------------------------------------------------

    def run(self) -> RunRecord:
        self.run_dir.mkdir(parents=True, exist_ok=False)
        (self.run_dir / "logs").mkdir()
        secrets_in_env = [self.environ.get(name) for name in API_KEY_ENV.values()]
        self.redactor = Redactor(secrets_in_env)
        self.events = EventLog(self.run_dir / "events.jsonl", self.redactor, listeners=self.listeners)
        self.events.emit("run_started", run_id=self.run_id, repo=str(self.request.repo), ref=self.request.ref)
        try:
            self._execute()
        except StopRun as stop:
            self._finish(stop.outcome, stop.code, stop.message)
        except BudgetExceeded as exc:
            self._finish(Outcome.FAILED, f"budget_exhausted_{exc.kind}", f"Budget exhausted: {exc}. Nothing was published.")
        except (RunCancelled, KeyboardInterrupt):
            self._finish(Outcome.CANCELLED, "cancelled", "The run was cancelled. Active commands were terminated and nothing was published.")
        except AgentTurnLimit as exc:
            self._finish(Outcome.FAILED, "agent_turn_limit", str(exc))
        except ModelError as exc:
            self._finish(Outcome.FAILED, "model_error", f"Model provider error: {exc}")
        except Exception as exc:  # noqa: BLE001 - every failure must still produce a report
            self.events.emit("internal_error", error=repr(exc), traceback=traceback.format_exc())
            self._finish(Outcome.FAILED, "internal_error", f"Internal error: {exc!r}")
        finally:
            self._close()
        return self.record

    def _finish(self, outcome: Outcome, code: str, message: str) -> None:
        self.record.outcome = outcome.value
        self.record.reason_code = code
        self.record.reason = message
        self.events.emit("run_finished", outcome=outcome.value, reason_code=code, reason=message)

    def _close(self) -> None:
        if self.record.outcome is None:  # pragma: no cover - defensive
            self._finish(Outcome.FAILED, "internal_error", "run ended without an outcome")
        self.record.finished_at = utc_now()
        self.record.usage = self.budget.snapshot()
        upstream = getattr(self.client, "upstream_providers", None)
        if self.record.model is not None and upstream:
            self.record.model["upstream_providers"] = sorted(upstream)
        self.record.warnings = self.ctx.warnings
        if self.ctx.sandbox is not None:
            self.record.commands = [c.summary() for c in self.ctx.sandbox.runner.history]
            try:
                self.ctx.sandbox.close()
            except Exception as exc:  # noqa: BLE001 - cleanup must not hide the outcome
                self.events.emit("sandbox_cleanup_failed", error=repr(exc))
        write_reports(self.record, self.run_dir, self.redactor)
        self.events.close()
        if self.ctx.temp_dir is not None:
            if self.request.keep_workspace:
                self.record.artifacts["workspace"] = str(self.ctx.temp_dir)
                write_reports(self.record, self.run_dir, self.redactor)
            else:
                shutil.rmtree(self.ctx.temp_dir, ignore_errors=True)

    @contextmanager
    def _stage(self, name: str) -> Iterator[dict]:
        self.events.stage = name
        entry = {"name": name, "status": "running", "started_at": utc_now(), "summary": ""}
        self.record.stages.append(entry)
        start = time.monotonic()
        try:
            yield entry
            entry["status"] = "completed"
        except BaseException:
            entry["status"] = "stopped"
            raise
        finally:
            entry["duration_seconds"] = round(time.monotonic() - start, 2)

    def _execute(self) -> None:
        self._preflight()
        self._setup()
        self._run_baseline()
        if self.request.check_only:
            self._finish(Outcome.CHECKED, "checked", "Environment setup and baseline test run completed.")
            return
        self._investigate()
        self._reproduce()
        self._patch_and_validate()
        self._deliver()

    # -- stages ----------------------------------------------------------

    def _preflight(self) -> None:
        with self._stage("preflight") as stage:
            repo = self.request.repo
            if not is_git_repo(repo):
                raise StopRun(Outcome.BLOCKED, "unsupported_repository", f"{repo} is not a git repository checkout.")
            try:
                commit = resolve_commit(repo, self.request.ref)
            except WorkspaceError:
                raise StopRun(Outcome.BLOCKED, "preflight_rejected", f"Cannot resolve ref {self.request.ref!r}.") from None
            self.record.base_commit = commit
            if has_uncommitted_changes(repo):
                self.ctx.warnings.append(
                    "The source checkout has uncommitted changes; they were ignored. The run used the pinned commit only."
                )
            files = [f for f in git(repo, "ls-tree", "-r", "-z", "--name-only", commit).split("\0") if f]
            if not any(f.endswith(".py") for f in files):
                raise StopRun(Outcome.BLOCKED, "unsupported_repository", "The repository has no Python files at the pinned commit.")
            if not any(policy.is_test_path(f) for f in files):
                self.ctx.warnings.append("No pytest test files were found at the pinned commit.")
            if any(f == ".gitmodules" for f in files):
                self.ctx.warnings.append("Git submodules are not checked out in Phase 1.")
            if not self.request.check_only:
                self._preflight_issue()
                if self.client is None:
                    try:
                        self.client = create_client(self.config.model, self.environ)
                    except ConfigError as exc:
                        raise StopRun(Outcome.BLOCKED, "preflight_rejected", f"Model configuration error: {exc}.") from None
                self.record.model = {"provider": self.client.provider, "name": self.client.model}
            self.budget.check()
            stage["summary"] = f"Pinned {commit[:12]}"

    def _preflight_issue(self) -> None:
        issue = self.request.issue
        if issue is None or not issue.body.strip():
            raise StopRun(Outcome.BLOCKED, "preflight_rejected", "The issue text is empty.")
        if len(issue.body) > self.limits.max_issue_chars:
            issue.body = issue.body[: self.limits.max_issue_chars] + "\n[issue text truncated by resolv]"
            self.ctx.warnings.append(f"The issue text was truncated to {self.limits.max_issue_chars} characters.")

    def _setup(self) -> None:
        with self._stage("setup") as stage:
            temp = Path(tempfile.mkdtemp(prefix=f"resolv-{self.run_id}-"))
            self.ctx.temp_dir = temp
            ws = Workspace.clone(self.request.repo.resolve(), self.record.base_commit, temp / "repo")
            self.ctx.workspace = ws
            sandbox_dir = temp / "sandbox"
            sandbox_dir.mkdir()
            sandbox = create_sandbox(
                self.config,
                workspace=ws,
                temp_dir=sandbox_dir,
                run_id=self.run_id,
                budget=self.budget,
                events=self.events,
                logs_dir=self.run_dir / "logs",
                redactor=self.redactor,
            )
            self.ctx.sandbox = sandbox
            self.record.environment["isolation"] = sandbox.describe()
            self.ctx.pytest = PytestRunner(sandbox, self.config.environment.test_command)
            try:
                stage["summary"] = sandbox.setup()
            except SetupError as exc:
                raise StopRun(Outcome.BLOCKED, exc.code, exc.message) from None

    def _run_baseline(self) -> None:
        with self._stage("baseline") as stage:
            run = self.ctx.pytest.run("baseline")
            self.ctx.baseline = run
            self.record.baseline = run.to_dict()
            stage["summary"] = run.summary()
            if run.environment_error:
                raise StopRun(
                    Outcome.BLOCKED, "environment_error",
                    f"The baseline test run did not produce usable results: {run.environment_error}. "
                    f"This is an installation or test-environment problem, not a code failure; see {run.command.log_path}.",
                )
            if run.collection_errors:
                names = ", ".join(t.nodeid or "<root>" for t in run.collection_errors[:5])
                raise StopRun(
                    Outcome.BLOCKED, "environment_error",
                    f"The baseline test suite has collection errors ({names}), so pytest did not run any tests. "
                    f"Fix the environment or test command; see {run.command.log_path}.",
                )

    def _session(self, stage: str, stage_tools: list[Tool]) -> AgentSession:
        return AgentSession(
            stage=stage,
            client=self.client,
            budget=self.budget,
            events=self.events,
            system=prompts.SYSTEM,
            tools=stage_tools,
            max_turns=self.limits.max_turns_per_stage,
            max_output_tokens=self.config.model.max_output_tokens,
            max_tool_output_chars=self.limits.max_tool_output_chars,
        )

    def _issue_block(self) -> str:
        issue = self.request.issue
        return prompts.issue_block(issue.title, issue.body)

    def _blocked_by_agent(self, call: FinalCall, code: str) -> StopRun:
        category = call.arguments.get("category", "other")
        reason = str(call.arguments.get("reason", "")).strip()
        self.record.blocker = {"category": category, "reason": reason}
        return StopRun(Outcome.BLOCKED, code, f"The agent stopped ({category}): {reason}")

    def _investigate(self) -> None:
        ws = self.ctx.workspace
        with self._stage("investigate") as stage:
            session = self._session("investigate", [*tools.read_tools(ws), tools.diagnosis_tool(), tools.report_blocker_tool()])
            files = ws.all_files()
            listing = "\n".join(files[:300]) + (f"\n... {len(files) - 300} more files" if len(files) > 300 else "")
            call = session.start(
                prompts.investigate(issue=self._issue_block(), files=listing, baseline=self.ctx.baseline.summary())
            )
            if call.name == "report_blocker":
                raise self._blocked_by_agent(call, "agent_blocked")
            self.ctx.diagnosis = {
                "summary": call.arguments.get("summary", ""),
                "cause": call.arguments.get("cause", ""),
                "relevant_files": list(call.arguments.get("relevant_files") or []),
                "reproduction_plan": call.arguments.get("reproduction_plan", ""),
            }
            self.record.diagnosis = self.ctx.diagnosis
            stage["summary"] = self.ctx.diagnosis["summary"][:200]

    def _reproduce(self) -> None:
        ws = self.ctx.workspace
        max_attempts = self.limits.max_reproduction_attempts
        with self._stage("reproduce") as stage:
            session = self._session(
                "reproduce",
                [
                    *tools.read_tools(ws),
                    *tools.write_tools(ws, policy.REPRODUCE),
                    tools.run_tests_tool(ws, self.ctx.pytest, "agent-reproduce"),
                    tools.reproduction_tool(),
                    tools.report_blocker_tool(),
                ],
            )
            call = session.start(
                prompts.reproduce(
                    issue=self._issue_block(), diagnosis=_format_diagnosis(self.ctx.diagnosis), max_attempts=max_attempts
                )
            )
            attempts = []
            for attempt in range(1, max_attempts + 1):
                if call.name == "report_blocker":
                    raise self._blocked_by_agent(call, "not_reproducible")
                problem = self._verify_reproduction(call.arguments)
                attempts.append({"attempt": attempt, "accepted": problem is None, "feedback": problem})
                self.record.reproduction_attempts = attempts
                if problem is None:
                    stage["summary"] = f"{len(self.ctx.reproduction.node_ids)} regression test(s) fail on the base commit"
                    return
                if attempt == max_attempts:
                    break
                call = session.resume(
                    f"Reproduction rejected (submission {attempt} of {max_attempts}):\n{problem}\n"
                    "Fix the test and submit again, or call report_blocker."
                )
            raise StopRun(
                Outcome.BLOCKED, "not_reproducible",
                f"No valid reproduction after {max_attempts} submissions. Last problem: {attempts[-1]['feedback']}",
            )

    def _verify_reproduction(self, args: dict) -> str | None:
        """Run the submitted tests ourselves; return a problem description or None if accepted."""
        ws = self.ctx.workspace
        try:
            selectors = tools.validate_selectors(ws, [str(s) for s in args.get("test_node_ids") or []])
        except ToolError as exc:
            return str(exc)
        result = policy.check_patch(ws, stage=policy.REPRODUCE, limits=self.limits)
        if not result.files:
            return "No test changes were made. Write a new regression test first."
        if not result.ok:
            return "Policy check failed:\n" + result.describe()
        run = self.ctx.pytest.run("verify-reproduction", selectors)
        if run.environment_error:
            return f"The tests did not run: {run.environment_error}.\n{run.command.output[-2000:]}"
        if run.collection_errors:
            return "Collection errors (not a valid reproduction):\n" + describe_run(run)
        if not run.tests:
            return "The submitted node ids matched no tests."
        existing = [n for n in run.tests if n in self.ctx.baseline.tests]
        if existing:
            return (
                "These tests already existed before your change; submit only the new regression tests: "
                + ", ".join(existing)
            )
        test_paths = {f.path for f in result.files}
        problems = []
        for outcome in run.tests.values():
            if not outcome.failing:
                problems.append(f"{outcome.nodeid} {outcome.outcome} on the unchanged code; it must fail because of the bug")
            elif (problem := reproduction_problem(outcome, test_paths)) is not None:
                problems.append(problem)
        if problems:
            return "\n".join(f"- {p}" for p in problems) + "\n\n" + describe_run(run)
        commit = ws.commit_snapshot("resolv: regression test")
        self.ctx.reproduction = Reproduction(
            node_ids=sorted(run.tests),
            expected_failure=str(args.get("expected_failure", "")),
            run=run,
            commit=commit,
            test_diff=ws.diff(),
        )
        self.record.reproduction = {
            "node_ids": sorted(run.tests),
            "expected_failure": self.ctx.reproduction.expected_failure,
            "test_files": sorted(test_paths),
            "before": [t.to_dict() for t in run.tests.values()],
            "command": run.command.summary(),
        }
        return None

    def _patch_and_validate(self) -> None:
        ws = self.ctx.workspace
        repro = self.ctx.reproduction
        max_attempts = self.limits.max_patch_attempts
        test_stats = policy.check_patch(ws, stage=policy.REPRODUCE, limits=self.limits)
        with self._stage("patch") as stage:
            session = self._session(
                "patch",
                [
                    *tools.read_tools(ws),
                    *tools.write_tools(ws, policy.PATCH),
                    tools.run_tests_tool(ws, self.ctx.pytest, "agent-patch"),
                    tools.patch_tool(),
                    tools.report_blocker_tool(),
                ],
            )
            failure = "\n\n".join(f"{t.headline()}\n{t.details[-1500:]}" for t in repro.run.tests.values())
            call = session.start(
                prompts.patch(
                    issue=self._issue_block(),
                    diagnosis=_format_diagnosis(self.ctx.diagnosis),
                    regression_ids=repro.node_ids,
                    test_diff=repro.test_diff,
                    failure=failure,
                    baseline_failures=[t.nodeid for t in self.ctx.baseline.failing()],
                    max_attempts=max_attempts,
                    max_files=self.limits.max_changed_files,
                    max_lines=self.limits.max_changed_lines,
                    used_files=len(test_stats.files),
                    used_lines=test_stats.total_changed_lines,
                )
            )
            for attempt in range(1, max_attempts + 1):
                if call.name == "report_blocker":
                    raise self._blocked_by_agent(call, "agent_blocked")
                self.events.stage = "validate"
                problem = self._validate_attempt(attempt, call.arguments)
                self.events.stage = "patch"
                if problem is None:
                    stage["summary"] = f"Accepted on submission {attempt}"
                    return
                if attempt == max_attempts:
                    break
                call = session.resume(
                    f"Validation failed (submission {attempt} of {max_attempts}):\n{problem}\n"
                    "Revise the production code and submit again, or call report_blocker."
                )
            raise StopRun(
                Outcome.FAILED, "patch_attempts_exhausted",
                f"No candidate patch passed validation within {max_attempts} submissions.",
            )

    def _validate_attempt(self, attempt: int, args: dict) -> str | None:
        ws = self.ctx.workspace
        repro = self.ctx.reproduction
        record = {"attempt": attempt, "changes": str(args.get("changes", "")), "accepted": False}
        self.record.patch_attempts.append(record)

        result = policy.check_patch(ws, stage=policy.PATCH, limits=self.limits, reproduction_commit=repro.commit)
        record["policy"] = result.to_dict()
        if result.limit_violations and not result.violations:
            (self.run_dir / "candidate.diff").write_text(ws.diff(), encoding="utf-8")
            self.record.artifacts["candidate_diff"] = "candidate.diff"
            record["feedback"] = result.describe()
            raise StopRun(
                Outcome.BLOCKED, "patch_limit_exceeded",
                "The candidate patch exceeds the configured scope limits and needs maintainer review "
                f"(saved as candidate.diff, not as a proposed patch):\n{result.describe()}",
            )
        if not result.ok:
            record["feedback"] = "Policy check failed:\n" + result.describe()
            return record["feedback"]

        regression = self.ctx.pytest.run("validate-regression", repro.node_ids)
        record["regression"] = regression.to_dict()
        not_passing = [n for n in repro.node_ids if n not in regression.tests or regression.tests[n].outcome != "passed"]
        if regression.environment_error or regression.collection_errors or not_passing:
            record["feedback"] = "The regression tests do not pass with the patch:\n" + describe_run(regression)
            return record["feedback"]

        suite = self.ctx.pytest.run("validate-suite")
        record["suite"] = suite.to_dict()
        if suite.environment_error:
            record["feedback"] = f"The configured test suite did not run with the patch: {suite.environment_error}\n{suite.command.output[-2000:]}"
            return record["feedback"]
        comparison = compare_runs(self.ctx.baseline, suite)
        record["comparison"] = comparison.to_dict()
        if not comparison.clean:
            lines = [f"- new failure: {t.headline()}" for t in comparison.new_failures]
            lines += [f"- test disappeared from the suite: {n}" for n in comparison.missing]
            details = "\n".join(f"{t.headline()}\n{t.details[-1200:]}" for t in comparison.new_failures[:5])
            record["feedback"] = "The configured suite has new problems compared with the baseline:\n" + "\n".join(lines) + "\n\n" + details
            return record["feedback"]

        record["accepted"] = True
        not_in_suite = [n for n in repro.node_ids if n not in suite.tests]
        skipped = sorted(n for n, t in suite.tests.items() if t.outcome in ("skipped", "xfailed"))
        self.record.patch = {
            "changes": record["changes"],
            "assumptions_and_risks": str(args.get("assumptions_and_risks", "")),
            "files": result.to_dict()["files"],
            "total_changed_lines": result.total_changed_lines,
        }
        self.record.validation = {
            "regression_after": [regression.tests[n].to_dict(details=False) for n in repro.node_ids],
            "suite_counts": suite.counts(),
            "suite_command": suite.command.summary(),
            "baseline_counts": self.ctx.baseline.counts(),
            **comparison.to_dict(),
            "skipped_or_xfailed": skipped,
            "regression_not_in_suite": not_in_suite,
        }
        return None

    def _deliver(self) -> None:
        with self._stage("deliver") as stage:
            # Re-check budget and cancellation immediately before publishing.
            self.budget.check()
            diff = self.ctx.workspace.diff()
            (self.run_dir / "patch.diff").write_text(diff, encoding="utf-8")
            self.record.artifacts["patch"] = "patch.diff"
            stage["summary"] = "Wrote patch.diff and reports"
        self._finish(
            Outcome.PATCH_READY, "patch_ready",
            "A proposed patch with a reproduced failure and passing validation is ready for review.",
        )

