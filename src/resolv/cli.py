"""Command-line interface.

    resolv run   --repo PATH --issue ISSUE.md [options]   # full workflow
    resolv check --repo PATH [options]                    # setup + baseline only, no model calls
"""

from __future__ import annotations

import argparse
import dataclasses
import signal
import sys
from pathlib import Path

from resolv import __version__
from resolv.config import RunConfig, load_config
from resolv.errors import ConfigError
from resolv.orchestrator import Issue, Orchestrator, Outcome, RunRequest

EXIT_CODES = {
    Outcome.PATCH_READY: 0,
    Outcome.CHECKED: 0,
    Outcome.BLOCKED: 3,
    Outcome.FAILED: 4,
    Outcome.CANCELLED: 130,
}


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--repo", required=True, type=Path, help="Path to a local git checkout of the repository.")
    parser.add_argument("--ref", default="HEAD", help="Commit, branch, or tag to pin (default: HEAD).")
    parser.add_argument("--config", type=Path, help="TOML config file (keep it outside the target repository).")
    parser.add_argument("--out", type=Path, default=Path("resolv-runs"), help="Directory for run artifacts.")
    parser.add_argument("--python", help="Interpreter used to create the venv (or run tests without one).")
    parser.add_argument("--venv", action=argparse.BooleanOptionalAction, default=None, help="Create an isolated venv (default: on).")
    parser.add_argument("--install", action="append", metavar="CMD", help="Install command run in the workspace; repeatable.")
    parser.add_argument("--test-command", help="pytest invocation, e.g. 'python -m pytest -q'.")
    parser.add_argument("--keep-workspace", action="store_true", help="Keep the temporary checkout and venv for inspection.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="resolv", description="Turn a bug report into a validated, reviewable patch.")
    parser.add_argument("--version", action="version", version=f"resolv {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Investigate, reproduce, patch, and validate one issue.")
    _common(run)
    run.add_argument("--issue", required=True, help="Issue text file (Markdown), or '-' for stdin.")
    run.add_argument("--issue-title", help="Issue title (default: first line of the issue text).")
    run.add_argument("--issue-url", help="Link to the original bug report, included in the report.")
    run.add_argument("--model", help="Model name (default from config: claude-sonnet-5-5).")
    run.add_argument("--max-cost-usd", type=float, help="Model spend cap for this run.")

    check = sub.add_parser("check", help="Validate setup and run the baseline tests without calling a model.")
    _common(check)
    return parser


def _apply_overrides(config: RunConfig, args: argparse.Namespace) -> RunConfig:
    env = config.environment
    env_changes = {}
    if args.python:
        env_changes["python"] = args.python
    if args.venv is not None:
        env_changes["venv"] = args.venv
    if args.install:
        env_changes["install_commands"] = tuple(args.install)
    if args.test_command:
        env_changes["test_command"] = args.test_command
    config = dataclasses.replace(config, environment=dataclasses.replace(env, **env_changes))
    if getattr(args, "model", None):
        config = dataclasses.replace(config, model=dataclasses.replace(config.model, name=args.model))
    if getattr(args, "max_cost_usd", None) is not None:
        config = dataclasses.replace(config, limits=dataclasses.replace(config.limits, max_cost_usd=args.max_cost_usd))
    return config.validate()


def _read_issue(args: argparse.Namespace) -> Issue:
    if args.issue == "-":
        text, source = sys.stdin.read(), "<stdin>"
    else:
        path = Path(args.issue)
        text, source = path.read_text(encoding="utf-8"), str(path)
    return Issue.from_text(text, title=args.issue_title, url=args.issue_url, source=source)


def _raise_cancel(signum, frame):  # noqa: ARG001
    raise KeyboardInterrupt


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        config = _apply_overrides(load_config(args.config), args)
        issue = _read_issue(args) if args.command == "run" else None
    except (ConfigError, OSError) as exc:
        print(f"resolv: error: {exc}", file=sys.stderr)
        return 2

    request = RunRequest(
        repo=args.repo,
        issue=issue,
        config=config,
        out_dir=args.out,
        ref=args.ref,
        check_only=args.command == "check",
        keep_workspace=args.keep_workspace,
    )
    signal.signal(signal.SIGTERM, _raise_cancel)
    orchestrator = Orchestrator(request)
    print(f"resolv: run {orchestrator.run_id} started; artifacts in {orchestrator.run_dir}", file=sys.stderr)
    record = orchestrator.run()
    outcome = Outcome(record.outcome)
    print(f"resolv: {outcome.value} ({record.reason_code}): {record.reason}")
    print(f"resolv: report: {orchestrator.run_dir / 'report.md'}")
    if "patch" in record.artifacts:
        print(f"resolv: patch: {orchestrator.run_dir / record.artifacts['patch']}")
    return EXIT_CODES[outcome]


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
