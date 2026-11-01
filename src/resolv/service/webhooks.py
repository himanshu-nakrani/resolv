"""GitHub webhook handling.

The handler is deliberately thin: it verifies the signature, deduplicates
deliveries, records installation and repository changes, and enqueues runs.
Authorization of the person who triggered a run happens in the gateway, the
only component with GitHub credentials, before any work starts.
"""

from __future__ import annotations

import hashlib
import hmac
from dataclasses import dataclass

from resolv.orchestrator import new_run_id
from resolv.service.db import Database, now
from resolv.service.settings import ServiceSettings


@dataclass
class WebhookResult:
    status: int
    message: str
    run_id: str | None = None


def verify_signature(secret: str, body: bytes, header: str | None) -> bool:
    if not secret or not header or not header.startswith("sha256="):
        return False
    expected = "sha256=" + hmac.new(secret.encode("utf-8"), body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(expected, header)


def _is_bot(user: dict | None) -> bool:
    user = user or {}
    return user.get("type") == "Bot" or str(user.get("login", "")).endswith("[bot]")


def _sync_repository(db: Database, payload: dict) -> dict | None:
    repo = payload.get("repository")
    installation = payload.get("installation") or {}
    if not repo or not installation.get("id"):
        return None
    owner = (repo.get("owner") or {}).get("login") or repo["full_name"].split("/")[0]
    if db.get_installation(installation["id"]) is None:
        db.upsert_installation(installation["id"], owner)
    db.upsert_repository(
        repo["id"], installation["id"], repo["full_name"],
        default_branch=repo.get("default_branch"), private=repo.get("private"),
    )
    return repo


def handle_event(db: Database, settings: ServiceSettings, event: str, delivery_id: str, payload: dict) -> WebhookResult:
    action = payload.get("action")
    if not db.record_delivery(delivery_id, event, action):
        return WebhookResult(200, "duplicate delivery ignored")
    result = _dispatch(db, settings, event, action, delivery_id, payload)
    db.set_delivery_result(delivery_id, result.message)
    return result


def _dispatch(db: Database, settings: ServiceSettings, event: str, action: str | None, delivery_id: str, payload: dict) -> WebhookResult:
    if event == "ping":
        return WebhookResult(200, "pong")
    if event == "installation":
        return _installation(db, action, payload)
    if event == "installation_repositories":
        return _installation_repositories(db, action, payload)
    if event == "issues" and action == "labeled":
        return _issue_labeled(db, settings, delivery_id, payload)
    if event == "pull_request" and action == "closed":
        return _pull_request_closed(db, payload)
    if event == "pull_request_review" and action == "submitted":
        return _review_submitted(db, payload)
    if event in ("repository",) and payload.get("repository"):
        _sync_repository(db, payload)
        return WebhookResult(200, "repository updated")
    return WebhookResult(200, f"ignored {event}.{action}")


def _installation(db: Database, action: str | None, payload: dict) -> WebhookResult:
    inst = payload.get("installation") or {}
    account = inst.get("account") or {}
    if not inst.get("id"):
        return WebhookResult(400, "missing installation")
    if action == "deleted":
        db.set_installation_state(inst["id"], removed=True)
        return WebhookResult(200, "installation removed")
    db.upsert_installation(inst["id"], account.get("login", "unknown"), account.get("type"))
    if action == "suspend":
        db.set_installation_state(inst["id"], suspended=True)
    elif action == "unsuspend":
        db.set_installation_state(inst["id"], suspended=False)
    for repo in payload.get("repositories") or []:
        db.upsert_repository(repo["id"], inst["id"], repo["full_name"], private=repo.get("private"))
    return WebhookResult(200, f"installation {action}")


def _installation_repositories(db: Database, action: str | None, payload: dict) -> WebhookResult:
    inst = payload.get("installation") or {}
    account = inst.get("account") or {}
    db.upsert_installation(inst["id"], account.get("login", "unknown"), account.get("type"))
    for repo in payload.get("repositories_added") or []:
        db.upsert_repository(repo["id"], inst["id"], repo["full_name"], private=repo.get("private"))
    for repo in payload.get("repositories_removed") or []:
        db.remove_repository(repo["id"])
    return WebhookResult(200, f"repositories {action}")


def _issue_labeled(db: Database, settings: ServiceSettings, delivery_id: str, payload: dict) -> WebhookResult:
    repo = _sync_repository(db, payload)
    issue = payload.get("issue") or {}
    label = (payload.get("label") or {}).get("name", "")
    sender = payload.get("sender") or {}
    if repo is None or not issue:
        return WebhookResult(400, "missing repository or issue")
    if label != db.trigger_label_for(repo["id"], settings.trigger_label):
        return WebhookResult(200, "label is not the trigger label")
    if issue.get("pull_request"):
        return WebhookResult(200, "labels on pull requests do not trigger runs")
    if _is_bot(sender):
        return WebhookResult(200, "bot senders do not trigger runs")
    run = db.create_run(
        run_id=new_run_id(),
        repo_id=repo["id"],
        installation_id=payload["installation"]["id"],
        issue_number=issue["number"],
        issue_title=issue.get("title"),
        issue_url=issue.get("html_url"),
        trigger="label",
        triggered_by=sender.get("login", "unknown"),
        delivery_id=delivery_id,
    )
    if run is None:
        return WebhookResult(200, "issue already has an active run")
    return WebhookResult(202, "run queued", run["id"])


def _pull_request_closed(db: Database, payload: dict) -> WebhookResult:
    pr = payload.get("pull_request") or {}
    repo = payload.get("repository") or {}
    if not str((pr.get("head") or {}).get("ref", "")).startswith("resolv/"):
        return WebhookResult(200, "not a resolv pull request")
    run = db.find_run_by_pr(repo.get("id"), pr.get("number"))
    if run is None:
        return WebhookResult(200, "unknown resolv pull request")
    db.update_run(run["id"], pr_closed_at=now(), pr_merged=bool(pr.get("merged")))
    return WebhookResult(200, "pull request outcome recorded")


def _review_submitted(db: Database, payload: dict) -> WebhookResult:
    pr = payload.get("pull_request") or {}
    repo = payload.get("repository") or {}
    review = payload.get("review") or {}
    if _is_bot(review.get("user")):
        return WebhookResult(200, "bot review ignored")
    run = db.find_run_by_pr(repo.get("id"), pr.get("number"))
    if run is None:
        return WebhookResult(200, "not a resolv pull request")
    if run["pr_first_review_at"] is None:
        db.update_run(run["id"], pr_first_review_at=now())
    return WebhookResult(200, "review recorded")
