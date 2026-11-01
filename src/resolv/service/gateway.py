"""Gateway: the only component that holds the GitHub App private key.

For each queued run it verifies that the person who triggered it has write
access, checks repository eligibility and budgets, pins the base commit,
captures the issue and a snapshot of the repository for the worker, and posts
a status comment. When a worker finishes, it publishes approved patches as
draft pull requests (see resolv.service.publisher) or posts a blocker report.
"""

from __future__ import annotations

import io
import json
import logging
import tarfile
import tempfile
import threading
import traceback
from dataclasses import replace
from pathlib import Path

from resolv.config import RunConfig
from resolv.service import messages
from resolv.service.db import Database, now
from resolv.service.github import PUBLISH_PERMISSIONS, WRITE_ROLES, GitHubApp, GitHubClient, GitHubError
from resolv.service.publisher import (
    BaseBranchMoved,
    PublicationCancelled,
    PublicationRejected,
    branch_name,
    fetch_snapshot,
    publish,
    validate_artifacts,
)
from resolv.service.settings import ServiceSettings
from resolv.service.storage import Storage, run_prefix

log = logging.getLogger("resolv.gateway")

PREPARE_PERMISSIONS = {"contents": "read", "issues": "write", "metadata": "read"}
MAX_COMMENTS = 30


class Rejected(Exception):
    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.message = message


def compose_issue(issue: dict, comments: list[dict], max_chars: int) -> dict:
    """Issue title, body, and human comments as one untrusted document."""
    parts = [issue.get("body") or ""]
    human = [c for c in comments if (c.get("user") or {}).get("type") != "Bot" and messages.MARKER not in (c.get("body") or "")]
    if human:
        parts.append("\n\n--- Comments ---")
        for c in human[:MAX_COMMENTS]:
            author = (c.get("user") or {}).get("login", "unknown")
            parts.append(f"\n@{author} ({c.get('author_association', 'NONE')}):\n{c.get('body') or ''}")
    body = "".join(parts)
    if len(body) > max_chars:
        body = body[:max_chars] + "\n[truncated]"
    return {"title": issue.get("title") or "", "body": body, "url": issue.get("html_url")}


def tar_directory(directory: Path) -> bytes:
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(str(directory), arcname="repo")
    return buf.getvalue()


def _safe_member(member: tarfile.TarInfo, path: str) -> tarfile.TarInfo | None:
    try:
        return tarfile.data_filter(member, path)
    except tarfile.FilterError:
        return None  # e.g. a symlink pointing outside the repository: skip rather than fail


def extract_snapshot(data: bytes, dest: Path) -> Path:
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter=_safe_member)
    return dest / "repo"


class Gateway:
    def __init__(
        self,
        settings: ServiceSettings,
        db: Database,
        storage: Storage,
        app: GitHubApp,
        *,
        base_config: RunConfig | None = None,
    ) -> None:
        self.settings = settings
        self.db = db
        self.storage = storage
        self.app = app
        self.base_config = base_config or RunConfig()

    # -- loop ------------------------------------------------------------------

    def run_once(self) -> int:
        actions = 0
        for run in self.db.runs_with_status("pending"):
            self.prepare(run)
            actions += 1
        for run in self.db.runs_with_status("finished"):
            self.finalize(run)
            actions += 1
        self.update_progress()
        self.reap_stale()
        return actions

    def serve_forever(self, stop: threading.Event) -> None:
        log.info("gateway started")
        while not stop.is_set():
            try:
                if self.run_once() == 0:
                    stop.wait(self.settings.poll_seconds)
            except Exception:  # noqa: BLE001 - keep serving
                log.error("gateway iteration failed:\n%s", traceback.format_exc())
                stop.wait(self.settings.poll_seconds)

    def _client(self, run, permissions: dict[str, str]) -> GitHubClient:
        return self.app.client(run["installation_id"], run["repo_id"], permissions)

    # -- preparation ---------------------------------------------------------------

    def prepare(self, run) -> None:
        if not self.db.transition(run["id"], ["pending"], "preparing"):
            return
        self.db.add_event(run["id"], stage="queued", kind="preparing", summary="Verifying trigger and fetching inputs")
        gh = None
        try:
            gh = self._client(run, PREPARE_PERMISSIONS)
            repo, settings, issue = self._check(gh, run)
            head = gh.branch_head(repo["full_name"], settings.base_branch)
            prefix = run_prefix(run["installation_id"], run["repo_id"], run["id"])
            payload = compose_issue(issue, gh.issue_comments(repo["full_name"], run["issue_number"]), self.base_config.limits.max_issue_chars)
            with tempfile.TemporaryDirectory(prefix="resolv-snapshot-") as tmp:
                checkout = Path(tmp) / "repo"
                url = self.settings.git_url_template.format(full_name=repo["full_name"])
                fetch_snapshot(url, head, gh.token, checkout)
                self.storage.put_bytes(f"{prefix}/input/snapshot.tar.gz", tar_directory(checkout))
            self.storage.put_bytes(f"{prefix}/input/issue.json", json.dumps(payload).encode())
            run = self.db.get_run(run["id"])
            comment_id = gh.create_comment(
                repo["full_name"], run["issue_number"], messages.progress_comment(run, self.settings.run_url(run["id"]))
            )
            values = dict(
                base_commit=head, base_branch=settings.base_branch, settings_snapshot=json.dumps(settings.snapshot()),
                issue_title=issue.get("title"), issue_url=issue.get("html_url"), comment_id=comment_id,
                reported_stage="queued",
            )
            if self.db.transition(run["id"], ["preparing"], "ready", require_no_cancel=True, **values):
                self.db.add_event(run["id"], stage="queued", kind="ready", summary=f"Pinned {settings.base_branch} at {head[:12]}")
            else:
                self.db.update_run(run["id"], **values)
                self._complete(run["id"], ["preparing"], "cancelled", "cancelled", "Cancelled before execution started.", gh=gh)
        except Rejected as rejection:
            self._complete(run["id"], ["preparing"], "blocked", rejection.code, rejection.message, gh=gh)
        except Exception as exc:  # noqa: BLE001 - every run must end with a report
            log.error("preparing %s failed:\n%s", run["id"], traceback.format_exc())
            self._complete(run["id"], ["preparing"], "failed", "preparation_error", f"Could not prepare the run: {exc}", gh=gh)

    def _check(self, gh: GitHubClient, run):
        repo = self.db.get_repository(run["repo_id"])
        installation = self.db.get_installation(run["installation_id"])
        full_name = repo["full_name"]
        if installation is None or installation["removed"] or installation["suspended"] or repo["removed"]:
            raise Rejected("unsupported_repository", "The resolv GitHub App is not active for this repository.")
        role = gh.collaborator_role(full_name, run["triggered_by"])
        if role not in WRITE_ROLES:
            raise Rejected(
                "unauthorized",
                f"@{run['triggered_by']} has {role} access to {full_name}. Only users with write access or higher can start runs.",
            )
        if self.settings.require_opt_in and not repo["enabled"]:
            raise Rejected(
                "not_enrolled",
                f"{full_name} is not enrolled in the resolv pilot yet. An operator can enable it with "
                f"`resolv-service pilot enable {full_name}`.",
            )
        if repo["private"] and not self.settings.allow_private_repos:
            raise Rejected(
                "private_repository_disabled",
                "Private repositories are disabled until the operator has configured model-provider data handling "
                "(RESOLV_ALLOW_PRIVATE_REPOS).",
            )
        settings = self.db.get_settings(repo["id"])
        if settings is None or not settings.confirmed:
            raise Rejected(
                "settings_unconfirmed",
                f"A maintainer must confirm the validation settings first: {self.settings.public_url}/repos/{full_name}/settings",
            )
        spent = self.db.month_spend(run["installation_id"])
        if spent >= self.settings.monthly_budget_usd:
            raise Rejected(
                "budget_exhausted",
                f"This installation has used its monthly model budget (${spent:.2f} of ${self.settings.monthly_budget_usd:.2f}).",
            )
        issue = gh.get_issue(full_name, run["issue_number"])
        if issue.get("pull_request"):
            raise Rejected("not_an_issue", "resolv runs on issues, not pull requests.")
        if issue.get("state") != "open":
            raise Rejected("issue_closed", "The issue is closed.")
        labels = {label["name"] for label in issue.get("labels", []) if isinstance(label, dict)}
        missing = [name for name in settings.required_labels if name not in labels]
        if missing:
            raise Rejected("ineligible_issue", f"The issue is missing required label(s): {', '.join(missing)}.")
        excluded = [name for name in settings.excluded_labels if name in labels]
        if excluded:
            raise Rejected("ineligible_issue", f"The issue has label(s) that exclude it from resolv: {', '.join(excluded)}.")
        if len((issue.get("body") or "").strip()) < settings.min_issue_chars:
            raise Rejected(
                "ineligible_issue",
                f"The issue description is shorter than {settings.min_issue_chars} characters; add steps to reproduce and the expected behaviour.",
            )
        return repo, settings, issue

    # -- progress and finalization ---------------------------------------------------

    def update_progress(self) -> None:
        for run in self.db.runs_with_status("running"):
            if not run["comment_id"] or run["stage"] == run["reported_stage"]:
                continue
            repo = self.db.get_repository(run["repo_id"])
            try:
                self._client(run, PREPARE_PERMISSIONS).update_comment(
                    repo["full_name"], run["comment_id"], messages.progress_comment(run, self.settings.run_url(run["id"]))
                )
                self.db.update_run(run["id"], reported_stage=run["stage"])
            except GitHubError as exc:
                log.warning("progress comment for %s failed: %s", run["id"], exc)

    def reap_stale(self) -> None:
        for run in self.db.stale_running_runs(self.settings.stale_after_seconds):
            self.db.transition(
                run["id"], ["running"], "finished", outcome="failed", reason_code="worker_lost",
                reason="The worker stopped responding, so the run was abandoned. Nothing was published.", finished_at=now(),
            )

    def finalize(self, run) -> None:
        if run["outcome"] == "patch_ready" and not run["cancel_requested_at"]:
            if self.db.transition(run["id"], ["finished"], "publishing", require_no_cancel=True):
                self._publish(self.db.get_run(run["id"]))
            return
        if not self.db.transition(run["id"], ["finished"], "finalizing"):
            return
        outcome, code, reason = run["outcome"], run["reason_code"], run["reason"]
        if outcome in (None, "patch_ready", "checked"):
            outcome, code = "cancelled", "cancelled"
            reason = f"Cancelled by @{run['cancel_requested_by']} before publication; no pull request was created."
        self._complete(run["id"], ["finalizing"], outcome, code or outcome, reason or "")

    def _limits(self, run):
        snapshot = json.loads(run["settings_snapshot"] or "{}")
        return replace(
            self.base_config.limits,
            max_changed_files=snapshot.get("max_changed_files", self.base_config.limits.max_changed_files),
            max_changed_lines=snapshot.get("max_changed_lines", self.base_config.limits.max_changed_lines),
        )

    def _publish(self, run) -> None:
        run_id = run["id"]
        prefix = run_prefix(run["installation_id"], run["repo_id"], run_id)
        repo = self.db.get_repository(run["repo_id"])
        self.db.add_event(run_id, stage="deliver", kind="publishing", summary="Re-validating artifacts before publication")
        try:
            snapshot = self.storage.get_bytes(f"{prefix}/input/snapshot.tar.gz")
            patch = self.storage.get_bytes(f"{prefix}/artifacts/patch.diff")
            report_raw = self.storage.get_bytes(f"{prefix}/artifacts/report.json")
            if snapshot is None or patch is None or report_raw is None:
                raise PublicationRejected("run artifacts are missing")
            report = json.loads(report_raw)
            with tempfile.TemporaryDirectory(prefix="resolv-publish-") as tmp:
                root = extract_snapshot(snapshot, Path(tmp))
                ws, paths = validate_artifacts(root, run["base_commit"], patch.decode("utf-8"), report, self._limits(run))
                run_url = self.settings.run_url(run_id)
                gh = self._client(run, PUBLISH_PERMISSIONS)
                result = publish(
                    gh,
                    full_name=repo["full_name"],
                    base_branch=run["base_branch"],
                    default_branch=repo["default_branch"],
                    base_commit=run["base_commit"],
                    branch=branch_name(run["issue_number"], run_id),
                    ws=ws,
                    paths=paths,
                    title=messages.pr_title(run["issue_number"], run["issue_title"]),
                    body=messages.pr_body(report, issue_number=run["issue_number"], issue_url=run["issue_url"], run_url=run_url),
                    commit_message=messages.commit_message(run["issue_number"], run["issue_title"], run_url),
                    should_abort=lambda: self.db.cancel_requested(run_id),
                )
            self._complete(
                run_id, ["publishing"], "draft_pr_created", "draft_pr_created", f"Draft pull request #{result.pr_number} created.",
                gh=gh, pr_number=result.pr_number, pr_url=result.pr_url, pr_branch=result.branch, pr_created_at=now(),
                evidence_complete=True,
            )
        except BaseBranchMoved as exc:
            self._complete(
                run_id, ["publishing"], "blocked", "base_branch_moved",
                f"The base branch {run['base_branch']} moved from {exc.expected[:12]} to {exc.actual[:12]} during the run, "
                "so the patch was not published. Rerun to try again on the latest code.",
            )
        except PublicationCancelled:
            self._complete(
                run_id, ["publishing"], "cancelled", "cancelled",
                f"Cancelled by @{self.db.get_run(run_id)['cancel_requested_by']} during publication; no pull request was created.",
            )
        except PublicationRejected as exc:
            self._complete(run_id, ["publishing"], "failed", "publication_rejected", f"Publication was refused: {exc}")
        except Exception as exc:  # noqa: BLE001
            log.error("publishing %s failed:\n%s", run_id, traceback.format_exc())
            self._complete(run_id, ["publishing"], "failed", "publication_error", f"Publishing the draft pull request failed: {exc}")

    def _complete(self, run_id: str, from_statuses: list[str], outcome: str, code: str, reason: str, *, gh: GitHubClient | None = None, **extra) -> None:
        current = self.db.get_run(run_id)
        values = dict(outcome=outcome, reason_code=code, reason=reason, completed_at=now(), **extra)
        if current["finished_at"] is None:
            values["finished_at"] = now()
        if not self.db.transition(run_id, from_statuses, "completed", **values):
            return
        run = self.db.get_run(run_id)
        repo = self.db.get_repository(run["repo_id"])
        self.db.add_event(run_id, stage="complete", kind="run_completed", summary=f"{outcome}: {reason}"[:2000])
        try:
            gh = gh if gh is not None and outcome != "draft_pr_created" else self._client(run, PREPARE_PERMISSIONS)
            body = messages.final_comment(run, self.settings.run_url(run_id))
            if run["comment_id"]:
                gh.update_comment(repo["full_name"], run["comment_id"], body)
            else:
                self.db.update_run(run_id, comment_id=gh.create_comment(repo["full_name"], run["issue_number"], body))
            if run["trigger"] == "label":
                gh.remove_label(repo["full_name"], run["issue_number"], self.db.trigger_label_for(repo["id"], self.settings.trigger_label))
        except Exception as exc:  # noqa: BLE001 - reporting failures must not block completion
            log.warning("final comment for %s failed: %s", run_id, exc)
            self.db.add_event(run_id, stage="complete", kind="comment_failed", summary=str(exc)[:500])
        self.storage.delete_prefix(f"{run_prefix(run['installation_id'], run['repo_id'], run_id)}/input")
