"""Patch policy: protected files, test integrity, and patch-scope limits."""

from __future__ import annotations

import fnmatch
import re
from dataclasses import dataclass, field
from pathlib import PurePosixPath

from resolv.config import Limits
from resolv.errors import ToolError
from resolv.workspace import FileStat, Workspace

REPRODUCE = "reproduce"
PATCH = "patch"

PROTECTED_NAMES = frozenset(
    {
        "pyproject.toml", "setup.py", "setup.cfg", "tox.ini", "pytest.ini", "noxfile.py", "conftest.py",
        "MANIFEST.in", ".coveragerc", "Pipfile", "Pipfile.lock", "environment.yml", "environment.yaml",
        "Makefile", "Jenkinsfile", "Procfile", "Dockerfile",
    }
)
PROTECTED_NAME_GLOBS = (
    "requirements*.txt", "requirements*.in", "constraints*.txt", "*.lock", "Dockerfile*", "*.dockerfile",
    "docker-compose*.yml", "docker-compose*.yaml", "compose.yml", "compose.yaml", ".gitlab-ci.yml",
    ".travis.yml", "azure-pipelines.yml", ".pre-commit-config.yaml", "renovate.json", ".git*",
)
PROTECTED_TOP_DIRS = frozenset({".github", ".gitlab", ".circleci", ".devcontainer", ".buildkite"})
PROTECTED_ANY_DIRS = frozenset({"requirements"})

TEST_DIR_NAMES = frozenset({"tests", "test", "testing"})

SKIP_MARKER_PATTERNS = tuple(
    re.compile(p)
    for p in (
        r"\bpytest\.mark\.(?:skip|skipif|xfail)\b",
        r"\bpytest\.(?:skip|xfail|importorskip)\s*\(",
        r"\bunittest\.(?:skip|skipIf|skipUnless|expectedFailure)\b",
        r"@(?:skip|skipIf|skipUnless|expectedFailure)\b",
        r"\.skipTest\s*\(",
        r"\b__test__\s*=\s*False\b",
        r"\bcollect_ignore(?:_glob)?\b",
    )
)


# Agent-written tests run inside the same process as pytest and resolv's result
# collector, so they could fake their own outcome. Production code could detect
# that it is under test. These checks catch the obvious forms; reviewers and
# held-out tests remain the real backstop.
_RESULTS_ACCESS = (re.compile(r"RESOLV_|resolv_pytest_results"), "accesses resolv's result collection")
TEST_INTEGRITY_PATTERNS = (
    _RESULTS_ACCESS,
    (re.compile(r"\bos\._exit\b"), "calls os._exit"),
    (re.compile(r"\bpluginmanager\b"), "manipulates pytest plugins"),
    (re.compile(r"\bexitstatus\b"), "touches pytest's exit status"),
    (re.compile(r"^\s*(?:async\s+)?def\s+pytest_\w+\s*\("), "defines pytest hooks"),
    (re.compile(r"(?<![\w.])(?:exec|eval|compile|__import__)\s*\("), "uses dynamic code execution"),
    (re.compile(r"\batexit\b"), "registers exit handlers"),
)
PRODUCTION_INTEGRITY_PATTERNS = (
    _RESULTS_ACCESS,
    (
        re.compile(r"PYTEST_CURRENT_TEST|\b_pytest\b|['\"]pytest['\"]\s+(?:not\s+)?in\s+sys\.modules|sys\.modules\.get\(\s*['\"]pytest"),
        "behaves differently when running under pytest",
    ),
)


def integrity_problems(lines: list[str], *, test: bool) -> list[str]:
    patterns = TEST_INTEGRITY_PATTERNS if test else PRODUCTION_INTEGRITY_PATTERNS
    found = []
    for line in lines:
        for pattern, reason in patterns:
            if pattern.search(line):
                found.append(f"{reason}: {line.strip()[:120]}")
                break
    return found


def protected_reason(path: str) -> str | None:
    pure = PurePosixPath(path)
    name = pure.name
    if name in PROTECTED_NAMES or any(fnmatch.fnmatch(name, g) for g in PROTECTED_NAME_GLOBS):
        return f"{path} is a protected dependency, build, CI, or test-configuration file"
    if pure.parts and pure.parts[0] in PROTECTED_TOP_DIRS:
        return f"{path} is inside protected CI/infrastructure directory {pure.parts[0]}/"
    if any(part in PROTECTED_ANY_DIRS for part in pure.parts[:-1]):
        return f"{path} is inside a protected dependency directory"
    return None


def is_test_path(path: str) -> bool:
    pure = PurePosixPath(path)
    name = pure.name
    if name == "conftest.py":
        return True
    if name.endswith(".py") and (name.startswith("test_") or name.endswith("_test.py")):
        return True
    return any(part in TEST_DIR_NAMES for part in pure.parts[:-1])


def is_append_only(old: str, new: str) -> bool:
    """True when every original line is kept unchanged at the start of the new content."""
    old_lines = old.splitlines()
    return new.splitlines()[: len(old_lines)] == old_lines


def check_write_allowed(rel: str, stage: str) -> None:
    reason = protected_reason(rel)
    if reason:
        raise ToolError(f"Write rejected: {reason}.")
    if stage == REPRODUCE and not is_test_path(rel):
        raise ToolError(
            "Write rejected: during reproduction you may only create or extend test files (test_*.py, *_test.py, "
            "or files under a tests/ directory). Production code must stay unchanged."
        )
    if stage == PATCH and is_test_path(rel):
        raise ToolError(
            "Write rejected: test files are locked while patching so the reproduction stays fixed. "
            "Change production code only."
        )


def skip_markers(lines: list[str]) -> list[str]:
    return [line.strip() for line in lines if any(p.search(line) for p in SKIP_MARKER_PATTERNS)]


@dataclass(frozen=True)
class Violation:
    code: str
    path: str | None
    message: str

    def to_dict(self) -> dict:
        return {"code": self.code, "path": self.path, "message": self.message}


@dataclass
class PolicyResult:
    files: list[FileStat]
    violations: list[Violation] = field(default_factory=list)
    limit_violations: list[Violation] = field(default_factory=list)

    @property
    def total_changed_lines(self) -> int:
        return sum(f.changed_lines for f in self.files)

    @property
    def ok(self) -> bool:
        return not self.violations and not self.limit_violations

    def describe(self) -> str:
        return "\n".join(f"- [{v.code}] {v.message}" for v in [*self.violations, *self.limit_violations])

    def to_dict(self) -> dict:
        return {
            "files": [{"path": f.path, "added": f.added, "deleted": f.deleted, "binary": f.binary} for f in self.files],
            "total_changed_lines": self.total_changed_lines,
            "violations": [v.to_dict() for v in self.violations],
            "limit_violations": [v.to_dict() for v in self.limit_violations],
        }


def check_patch(ws: Workspace, *, stage: str, limits: Limits, reproduction_commit: str | None = None) -> PolicyResult:
    """Check the full change set against the pinned base commit."""
    result = PolicyResult(files=ws.numstat())
    add = result.violations.append
    for stat in result.files:
        path = stat.path
        test = is_test_path(path)
        current = ws.root / path
        if stat.binary:
            add(Violation("binary_change", path, f"{path}: binary changes are not allowed"))
            continue
        reason = protected_reason(path)
        if reason:
            add(Violation("protected_file", path, reason))
        if not current.exists():
            add(Violation("file_deleted", path, f"{path}: deleting files is not allowed"))
            continue
        if test and ws.exists_at_base(path):
            old = ws.content_at(ws.base_commit, path) or ""
            if not is_append_only(old, ws.read_text(path)):
                add(Violation("existing_test_modified", path, f"{path}: existing test content was changed or removed; only appending new tests is allowed"))
        if path.endswith(".py"):
            added = ws.added_lines(path)
            markers = skip_markers(added)
            if markers:
                add(Violation("skip_marker_added", path, f"{path}: adds skip/xfail markers: {markers[0][:120]}"))
            problems = integrity_problems(added, test=test)
            if problems:
                add(Violation("integrity_check", path, f"{path}: {problems[0]}"))
        if stage == REPRODUCE and not test:
            add(Violation("production_change_during_reproduction", path, f"{path}: production code must not change during reproduction"))

    if stage == PATCH:
        if reproduction_commit:
            for stat in ws.numstat(against=reproduction_commit):
                if is_test_path(stat.path):
                    add(Violation("reproduction_test_modified", stat.path, f"{stat.path}: test files changed after the reproduction was fixed"))
        if not any(not is_test_path(f.path) for f in result.files):
            add(Violation("empty_patch", None, "the patch does not change any production code"))

    if len(result.files) > limits.max_changed_files:
        result.limit_violations.append(
            Violation("too_many_files", None, f"{len(result.files)} changed files exceeds the limit of {limits.max_changed_files}")
        )
    if result.total_changed_lines > limits.max_changed_lines:
        result.limit_violations.append(
            Violation("too_many_lines", None, f"{result.total_changed_lines} changed lines exceeds the limit of {limits.max_changed_lines}")
        )
    return result
