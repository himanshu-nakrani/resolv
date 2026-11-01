"""Draft pull request publication.

The publisher trusts nothing the worker produced. Before anything is written
to GitHub it re-checks the evidence in the report, applies the stored patch to
a clean snapshot of the pinned commit, and re-runs every patch-policy check.
It then creates a dedicated ``resolv/...`` branch and a *draft* pull request
through the Git Data API. It never pushes to an existing branch, never
approves, and never merges.
"""

from __future__ import annotations

import base64
import re
import subprocess
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

from resolv import policy
from resolv.config import Limits
from resolv.service.github import GitHubClient, GitHubError
from resolv.workspace import Workspace, git

BRANCH_PREFIX = "resolv/"
MAX_PR_BODY = 60000


class PublicationRejected(Exception):
    """The artifacts do not meet the delivery criteria."""


class BaseBranchMoved(Exception):
    def __init__(self, expected: str, actual: str) -> None:
        super().__init__(f"base branch moved from {expected[:12]} to {actual[:12]}")
        self.expected = expected
        self.actual = actual


class PublicationCancelled(Exception):
    pass


@dataclass
class PublishResult:
    pr_number: int
    pr_url: str
    branch: str
    commit_sha: str


def branch_name(issue_number: int, run_id: str) -> str:
    suffix = run_id.rsplit("-", 1)[-1].lower()
    return f"{BRANCH_PREFIX}issue-{issue_number}-{suffix}"


def assert_safe_branch(branch: str, base_branch: str, default_branch: str | None) -> None:
    if not branch.startswith(BRANCH_PREFIX) or branch in (base_branch, default_branch):
        raise PublicationRejected(f"refusing to publish to branch {branch!r}")
    if ".." in branch or not re.fullmatch(r"[A-Za-z0-9._/-]+", branch) or branch.endswith((".lock", "/")):
        raise PublicationRejected(f"invalid branch name {branch!r}")


def check_evidence(report: dict) -> None:
    """Delivery criteria: failing before the fix, passing after, no new suite failures."""
    if report.get("outcome") != "patch_ready":
        raise PublicationRejected(f"run outcome is {report.get('outcome')!r}, not patch_ready")
    reproduction = report.get("reproduction") or {}
    validation = report.get("validation") or {}
    before = reproduction.get("before") or []
    after = validation.get("regression_after") or []
    if not before or any(t.get("outcome") != "failed" for t in before):
        raise PublicationRejected("missing evidence that the regression tests failed before the fix")
    if not after or any(t.get("outcome") != "passed" for t in after):
        raise PublicationRejected("missing evidence that the regression tests pass with the fix")
    if {t.get("nodeid") for t in before} != {t.get("nodeid") for t in after}:
        raise PublicationRejected("before/after evidence covers different tests")
    if validation.get("new_failures") or validation.get("missing"):
        raise PublicationRejected("the configured suite has new failures or missing tests")


def _patch_paths(root: Path, patch_file: Path) -> list[str]:
    out = git(root, "apply", "--numstat", "-z", str(patch_file))
    paths = []
    for record in out.split("\0"):
        if record.strip():
            paths.append(record.split("\t", 2)[2])
    return paths


def validate_artifacts(root: Path, base_commit: str, patch: str, report: dict, limits: Limits) -> tuple[Workspace, list[str]]:
    """Apply ``patch`` to the clean checkout at ``root`` and re-run all policy checks."""
    check_evidence(report)
    if not patch.strip():
        raise PublicationRejected("the patch is empty")
    patch_file = root.parent / "publish.patch"
    patch_file.write_text(patch, encoding="utf-8")
    head = git(root, "rev-parse", "HEAD").strip()
    if head != base_commit:
        raise PublicationRejected("snapshot does not match the pinned commit")
    try:
        git(root, "apply", "--check", str(patch_file))
        paths = _patch_paths(root, patch_file)
        git(root, "apply", str(patch_file))
    except Exception as exc:  # noqa: BLE001
        raise PublicationRejected(f"the patch does not apply to the pinned commit: {exc}") from None
    ws = Workspace(root, base_commit)
    ws.written = set(paths)
    result = policy.check_patch(ws, stage=policy.PATCH, limits=limits)
    if not result.ok:
        raise PublicationRejected("patch policy check failed:\n" + result.describe())
    return ws, sorted(paths)


def _mode(ws: Workspace, path: str) -> str:
    out = git(ws.root, "ls-tree", ws.base_commit, "--", path).strip()
    if out:
        mode = out.split()[0]
        if mode in ("100644", "100755"):
            return mode
    return "100644"


def publish(
    gh: GitHubClient,
    *,
    full_name: str,
    base_branch: str,
    default_branch: str | None,
    base_commit: str,
    branch: str,
    ws: Workspace,
    paths: list[str],
    title: str,
    body: str,
    commit_message: str,
    should_abort: Callable[[], bool] = lambda: False,
) -> PublishResult:
    assert_safe_branch(branch, base_branch, default_branch)
    head = gh.branch_head(full_name, base_branch)
    if head != base_commit:
        raise BaseBranchMoved(base_commit, head)
    base_tree = gh.get(f"/repos/{full_name}/git/commits/{base_commit}")["tree"]["sha"]
    entries = []
    for path in paths:
        content = base64.b64encode((ws.root / path).read_bytes()).decode("ascii")
        blob = gh.request("POST", f"/repos/{full_name}/git/blobs", {"content": content, "encoding": "base64"})
        entries.append({"path": path, "mode": _mode(ws, path), "type": "blob", "sha": blob["sha"]})
    tree = gh.request("POST", f"/repos/{full_name}/git/trees", {"base_tree": base_tree, "tree": entries})
    commit = gh.request(
        "POST", f"/repos/{full_name}/git/commits", {"message": commit_message, "tree": tree["sha"], "parents": [base_commit]}
    )
    if should_abort():
        raise PublicationCancelled()
    gh.request("POST", f"/repos/{full_name}/git/refs", {"ref": f"refs/heads/{branch}", "sha": commit["sha"]})
    try:
        if should_abort():
            raise PublicationCancelled()
        head = gh.branch_head(full_name, base_branch)
        if head != base_commit:
            raise BaseBranchMoved(base_commit, head)
        try:
            pr = gh.request(
                "POST", f"/repos/{full_name}/pulls",
                {"title": title, "head": branch, "base": base_branch, "body": body[:MAX_PR_BODY], "draft": True,
                 "maintainer_can_modify": True},
            )
        except GitHubError as exc:
            if exc.status == 422 and "draft" in exc.message.lower():
                raise PublicationRejected(f"GitHub refused to create a draft pull request: {exc.message}") from None
            raise
    except BaseException:
        try:
            gh.request("DELETE", f"/repos/{full_name}/git/refs/heads/{branch}")
        except GitHubError:
            pass
        raise
    return PublishResult(pr["number"], pr["html_url"], branch, commit["sha"])


def fetch_snapshot(url: str, sha: str, token: str | None, dest: Path) -> None:
    """Shallow-fetch exactly one commit into ``dest``; the token is passed via environment, never stored."""
    env = {"GIT_TERMINAL_PROMPT": "0", "GIT_CONFIG_NOSYSTEM": "1", "HOME": str(dest.parent), "PATH": "/usr/local/bin:/usr/bin:/bin"}
    if token:
        basic = base64.b64encode(f"x-access-token:{token}".encode()).decode()
        env.update(GIT_CONFIG_COUNT="1", GIT_CONFIG_KEY_0="http.extraheader", GIT_CONFIG_VALUE_0=f"AUTHORIZATION: basic {basic}")

    def run(*args: str) -> str:
        proc = subprocess.run(["git", *args], cwd=dest, env=env, capture_output=True, text=True, timeout=600)
        if proc.returncode != 0:
            message = (proc.stderr or proc.stdout).strip().replace(token or "\0", "[REDACTED]")
            raise RuntimeError(f"git {args[0]} failed: {message[:500]}")
        return proc.stdout

    dest.mkdir(parents=True, exist_ok=True)
    run("init", "-q")
    run("fetch", "-q", "--depth", "1", "--no-tags", url, sha)
    run("-c", "advice.detachedHead=false", "checkout", "-q", "--detach", "FETCH_HEAD")
    if run("rev-parse", "HEAD").strip() != sha:
        raise RuntimeError("fetched snapshot does not match the pinned commit")
