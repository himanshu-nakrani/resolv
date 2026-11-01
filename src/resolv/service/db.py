"""Run metadata, job queue, sessions, and pilot feedback (SQLAlchemy Core).

SQLite works for development and tests; PostgreSQL for deployments. The run
table doubles as the job queue: components claim work with conditional
status transitions, so concurrent workers or gateways never process the same
run twice.

Run lifecycle (status):
    pending -> preparing -> ready -> running -> finished -> publishing|finalizing -> completed
A run counts as active until it is completed; at most one active run may exist
per repository issue (enforced by a partial unique index).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from sqlalchemy import (
    BigInteger,
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    MetaData,
    String,
    Table,
    Text,
    and_,
    create_engine,
    delete,
    func,
    insert,
    select,
    update,
)
from sqlalchemy.engine import Engine, RowMapping
from sqlalchemy.exc import IntegrityError
from sqlalchemy.pool import StaticPool

ACTIVE_STATUSES = ("pending", "preparing", "ready", "running", "finished", "publishing", "finalizing")
FINAL_OUTCOMES = ("draft_pr_created", "blocked", "failed", "cancelled")

metadata = MetaData()

installations = Table(
    "installations",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=False),
    Column("account_login", String(255), nullable=False),
    Column("account_type", String(32)),
    Column("suspended", Boolean, nullable=False, default=False),
    Column("removed", Boolean, nullable=False, default=False),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
)

repositories = Table(
    "repositories",
    metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=False),
    Column("installation_id", BigInteger, ForeignKey("installations.id"), nullable=False, index=True),
    Column("full_name", String(255), nullable=False, index=True),
    Column("default_branch", String(255)),
    Column("private", Boolean, nullable=False, default=False),
    Column("enabled", Boolean, nullable=False, default=False),
    Column("removed", Boolean, nullable=False, default=False),
    Column("created_at", DateTime),
    Column("updated_at", DateTime),
)

repo_settings = Table(
    "repo_settings",
    metadata,
    Column("repo_id", BigInteger, ForeignKey("repositories.id"), primary_key=True, autoincrement=False),
    Column("base_branch", String(255), nullable=False),
    Column("python_version", String(16), nullable=False),
    Column("install_commands", Text, nullable=False),
    Column("test_command", Text, nullable=False),
    Column("trigger_label", String(100), nullable=False),
    Column("max_changed_files", Integer, nullable=False),
    Column("max_changed_lines", Integer, nullable=False),
    Column("max_cost_usd", Float, nullable=False),
    Column("required_labels", Text, nullable=False, default=""),
    Column("excluded_labels", Text, nullable=False, default=""),
    Column("min_issue_chars", Integer, nullable=False, default=0),
    Column("confirmed_by", String(255)),
    Column("confirmed_at", DateTime),
    Column("updated_at", DateTime),
)

runs = Table(
    "runs",
    metadata,
    Column("id", String(64), primary_key=True),
    Column("repo_id", BigInteger, ForeignKey("repositories.id"), nullable=False, index=True),
    Column("installation_id", BigInteger, nullable=False, index=True),
    Column("issue_number", Integer, nullable=False),
    Column("issue_title", Text),
    Column("issue_url", Text),
    Column("trigger", String(32), nullable=False),
    Column("triggered_by", String(255), nullable=False),
    Column("delivery_id", String(128)),
    Column("status", String(32), nullable=False, index=True),
    Column("stage", String(32)),
    Column("reported_stage", String(32)),
    Column("outcome", String(32)),
    Column("reason_code", String(64)),
    Column("reason", Text),
    Column("base_branch", String(255)),
    Column("base_commit", String(64)),
    Column("settings_snapshot", Text),
    Column("worker_id", String(128)),
    Column("heartbeat_at", DateTime),
    Column("cost_usd", Float, nullable=False, default=0.0),
    Column("elapsed_seconds", Float),
    Column("created_at", DateTime, nullable=False),
    Column("started_at", DateTime),
    Column("finished_at", DateTime),
    Column("completed_at", DateTime),
    Column("cancel_requested_at", DateTime),
    Column("cancel_requested_by", String(255)),
    Column("comment_id", BigInteger),
    Column("pr_number", Integer),
    Column("pr_url", Text),
    Column("pr_branch", String(255)),
    Column("pr_created_at", DateTime),
    Column("pr_first_review_at", DateTime),
    Column("pr_closed_at", DateTime),
    Column("pr_merged", Boolean),
    Column("evidence_complete", Boolean),
    Column("artifacts_deleted_at", DateTime),
)
Index(
    "uq_runs_active_issue",
    runs.c.repo_id,
    runs.c.issue_number,
    unique=True,
    sqlite_where=runs.c.status.in_(ACTIVE_STATUSES),
    postgresql_where=runs.c.status.in_(ACTIVE_STATUSES),
)

run_events = Table(
    "run_events",
    metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("run_id", String(64), ForeignKey("runs.id"), nullable=False, index=True),
    Column("ts", DateTime, nullable=False),
    Column("stage", String(32)),
    Column("kind", String(64), nullable=False),
    Column("summary", Text, nullable=False),
)

webhook_deliveries = Table(
    "webhook_deliveries",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("event", String(64), nullable=False),
    Column("action", String(64)),
    Column("received_at", DateTime, nullable=False),
    Column("result", Text),
)

web_sessions = Table(
    "web_sessions",
    metadata,
    Column("id", String(128), primary_key=True),
    Column("user_id", BigInteger),
    Column("login", String(255), nullable=False),
    Column("token", Text),
    Column("csrf", String(128), nullable=False),
    Column("created_at", DateTime, nullable=False),
    Column("expires_at", DateTime, nullable=False),
)

feedback = Table(
    "feedback",
    metadata,
    Column("run_id", String(64), ForeignKey("runs.id"), primary_key=True),
    Column("verdict", String(32), nullable=False),
    Column("reason_category", String(64)),
    Column("notes", Text),
    Column("recorded_by", String(255), nullable=False),
    Column("recorded_at", DateTime, nullable=False),
)


def now() -> datetime:
    """Naive UTC timestamp (stored consistently across SQLite and PostgreSQL)."""
    return datetime.now(UTC).replace(tzinfo=None)


@dataclass(frozen=True)
class RepoSettings:
    base_branch: str
    python_version: str
    install_commands: tuple[str, ...]
    test_command: str
    trigger_label: str
    max_changed_files: int = 5
    max_changed_lines: int = 300
    max_cost_usd: float = 3.0
    required_labels: tuple[str, ...] = ()
    excluded_labels: tuple[str, ...] = ()
    min_issue_chars: int = 0
    confirmed_by: str | None = None
    confirmed_at: datetime | None = None

    @property
    def confirmed(self) -> bool:
        return self.confirmed_at is not None

    def snapshot(self) -> dict:
        return {
            "base_branch": self.base_branch,
            "python_version": self.python_version,
            "install_commands": list(self.install_commands),
            "test_command": self.test_command,
            "max_changed_files": self.max_changed_files,
            "max_changed_lines": self.max_changed_lines,
            "max_cost_usd": self.max_cost_usd,
        }


def _labels(text: str | None) -> tuple[str, ...]:
    return tuple(part.strip() for part in (text or "").split(",") if part.strip())


class Database:
    def __init__(self, url: str) -> None:
        kwargs: dict = {}
        if url.startswith("sqlite"):
            kwargs["connect_args"] = {"check_same_thread": False}
            if url in ("sqlite://", "sqlite:///:memory:"):
                kwargs["poolclass"] = StaticPool
        self.engine: Engine = create_engine(url, **kwargs)
        self.is_postgres = self.engine.dialect.name == "postgresql"
        if self.engine.dialect.name == "sqlite":
            from sqlalchemy import event

            @event.listens_for(self.engine, "connect")
            def _sqlite_pragmas(dbapi_conn, _record):  # pragma: no cover - trivial
                cur = dbapi_conn.cursor()
                cur.execute("PRAGMA journal_mode=WAL")
                cur.execute("PRAGMA busy_timeout=10000")
                cur.execute("PRAGMA foreign_keys=ON")
                cur.close()

    def create_all(self) -> None:
        metadata.create_all(self.engine)

    # -- generic helpers ---------------------------------------------------

    def _one(self, stmt) -> RowMapping | None:
        with self.engine.connect() as conn:
            return conn.execute(stmt).mappings().first()

    def _all(self, stmt) -> list[RowMapping]:
        with self.engine.connect() as conn:
            return list(conn.execute(stmt).mappings().all())

    def _exec(self, stmt) -> int:
        with self.engine.begin() as conn:
            return conn.execute(stmt).rowcount

    # -- installations and repositories ---------------------------------------

    def upsert_installation(self, installation_id: int, account_login: str, account_type: str | None = None) -> None:
        with self.engine.begin() as conn:
            exists = conn.execute(select(installations.c.id).where(installations.c.id == installation_id)).first()
            values = {"account_login": account_login, "updated_at": now(), "removed": False}
            if account_type:
                values["account_type"] = account_type
            if exists:
                conn.execute(update(installations).where(installations.c.id == installation_id).values(**values))
            else:
                conn.execute(insert(installations).values(id=installation_id, created_at=now(), suspended=False, **values))

    def set_installation_state(self, installation_id: int, *, suspended: bool | None = None, removed: bool | None = None) -> None:
        values: dict = {"updated_at": now()}
        if suspended is not None:
            values["suspended"] = suspended
        if removed is not None:
            values["removed"] = removed
        self._exec(update(installations).where(installations.c.id == installation_id).values(**values))
        if removed:
            self._exec(update(repositories).where(repositories.c.installation_id == installation_id).values(removed=True))

    def get_installation(self, installation_id: int) -> RowMapping | None:
        return self._one(select(installations).where(installations.c.id == installation_id))

    def upsert_repository(
        self,
        repo_id: int,
        installation_id: int,
        full_name: str,
        *,
        default_branch: str | None = None,
        private: bool | None = None,
    ) -> None:
        with self.engine.begin() as conn:
            exists = conn.execute(select(repositories.c.id).where(repositories.c.id == repo_id)).first()
            values: dict = {"installation_id": installation_id, "full_name": full_name, "removed": False, "updated_at": now()}
            if default_branch:
                values["default_branch"] = default_branch
            if private is not None:
                values["private"] = private
            if exists:
                conn.execute(update(repositories).where(repositories.c.id == repo_id).values(**values))
            else:
                values.setdefault("private", False)
                conn.execute(insert(repositories).values(id=repo_id, enabled=False, created_at=now(), **values))

    def remove_repository(self, repo_id: int) -> None:
        self._exec(update(repositories).where(repositories.c.id == repo_id).values(removed=True, updated_at=now()))

    def get_repository(self, repo_id: int) -> RowMapping | None:
        return self._one(select(repositories).where(repositories.c.id == repo_id))

    def get_repository_by_name(self, full_name: str) -> RowMapping | None:
        return self._one(
            select(repositories).where(func.lower(repositories.c.full_name) == full_name.lower(), repositories.c.removed.is_(False))
        )

    def list_repositories(self, ids: Iterable[int] | None = None, *, enabled_only: bool = False) -> list[RowMapping]:
        stmt = select(repositories).where(repositories.c.removed.is_(False)).order_by(repositories.c.full_name)
        if ids is not None:
            stmt = stmt.where(repositories.c.id.in_(list(ids)))
        if enabled_only:
            stmt = stmt.where(repositories.c.enabled.is_(True))
        return self._all(stmt)

    def set_enabled(self, full_name: str, enabled: bool) -> bool:
        return self._exec(
            update(repositories).where(func.lower(repositories.c.full_name) == full_name.lower()).values(enabled=enabled, updated_at=now())
        ) > 0

    # -- settings --------------------------------------------------------------

    def get_settings(self, repo_id: int) -> RepoSettings | None:
        row = self._one(select(repo_settings).where(repo_settings.c.repo_id == repo_id))
        if row is None:
            return None
        return RepoSettings(
            base_branch=row["base_branch"],
            python_version=row["python_version"],
            install_commands=tuple(json.loads(row["install_commands"])),
            test_command=row["test_command"],
            trigger_label=row["trigger_label"],
            max_changed_files=row["max_changed_files"],
            max_changed_lines=row["max_changed_lines"],
            max_cost_usd=row["max_cost_usd"],
            required_labels=_labels(row["required_labels"]),
            excluded_labels=_labels(row["excluded_labels"]),
            min_issue_chars=row["min_issue_chars"],
            confirmed_by=row["confirmed_by"],
            confirmed_at=row["confirmed_at"],
        )

    def save_settings(self, repo_id: int, settings: RepoSettings, confirmed_by: str) -> None:
        values = {
            "base_branch": settings.base_branch,
            "python_version": settings.python_version,
            "install_commands": json.dumps(list(settings.install_commands)),
            "test_command": settings.test_command,
            "trigger_label": settings.trigger_label,
            "max_changed_files": settings.max_changed_files,
            "max_changed_lines": settings.max_changed_lines,
            "max_cost_usd": settings.max_cost_usd,
            "required_labels": ",".join(settings.required_labels),
            "excluded_labels": ",".join(settings.excluded_labels),
            "min_issue_chars": settings.min_issue_chars,
            "confirmed_by": confirmed_by,
            "confirmed_at": now(),
            "updated_at": now(),
        }
        with self.engine.begin() as conn:
            exists = conn.execute(select(repo_settings.c.repo_id).where(repo_settings.c.repo_id == repo_id)).first()
            if exists:
                conn.execute(update(repo_settings).where(repo_settings.c.repo_id == repo_id).values(**values))
            else:
                conn.execute(insert(repo_settings).values(repo_id=repo_id, **values))

    def trigger_label_for(self, repo_id: int, default: str) -> str:
        row = self._one(select(repo_settings.c.trigger_label).where(repo_settings.c.repo_id == repo_id))
        return row["trigger_label"] if row else default

    # -- webhooks ----------------------------------------------------------------

    def record_delivery(self, delivery_id: str, event: str, action: str | None) -> bool:
        """Returns False when this delivery was already processed (GitHub redelivery)."""
        try:
            self._exec(insert(webhook_deliveries).values(id=delivery_id, event=event, action=action, received_at=now()))
            return True
        except IntegrityError:
            return False

    def set_delivery_result(self, delivery_id: str, result: str) -> None:
        self._exec(update(webhook_deliveries).where(webhook_deliveries.c.id == delivery_id).values(result=result[:500]))

    # -- runs --------------------------------------------------------------------

    def create_run(
        self,
        *,
        run_id: str,
        repo_id: int,
        installation_id: int,
        issue_number: int,
        trigger: str,
        triggered_by: str,
        issue_title: str | None = None,
        issue_url: str | None = None,
        delivery_id: str | None = None,
    ) -> RowMapping | None:
        """Create a pending run, or return None if the issue already has an active run."""
        try:
            self._exec(
                insert(runs).values(
                    id=run_id, repo_id=repo_id, installation_id=installation_id, issue_number=issue_number,
                    issue_title=issue_title, issue_url=issue_url, trigger=trigger, triggered_by=triggered_by,
                    delivery_id=delivery_id, status="pending", stage="queued", cost_usd=0.0, created_at=now(),
                )
            )
        except IntegrityError:
            return None
        return self.get_run(run_id)

    def get_run(self, run_id: str) -> RowMapping | None:
        return self._one(select(runs).where(runs.c.id == run_id))

    def active_run_for_issue(self, repo_id: int, issue_number: int) -> RowMapping | None:
        return self._one(
            select(runs).where(runs.c.repo_id == repo_id, runs.c.issue_number == issue_number, runs.c.status.in_(ACTIVE_STATUSES))
        )

    def list_runs(self, repo_ids: Iterable[int] | None = None, *, limit: int = 100, status: str | None = None) -> list[RowMapping]:
        stmt = select(runs).order_by(runs.c.created_at.desc(), runs.c.id.desc()).limit(limit)
        if repo_ids is not None:
            stmt = stmt.where(runs.c.repo_id.in_(list(repo_ids)))
        if status is not None:
            stmt = stmt.where(runs.c.status == status)
        return self._all(stmt)

    def runs_with_status(self, *statuses: str, limit: int = 50) -> list[RowMapping]:
        return self._all(select(runs).where(runs.c.status.in_(statuses)).order_by(runs.c.created_at).limit(limit))

    def transition(self, run_id: str, from_statuses: Iterable[str], to_status: str, *, require_no_cancel: bool = False, **values) -> bool:
        """Atomically move a run between statuses; False if another component got there first."""
        cond = and_(runs.c.id == run_id, runs.c.status.in_(list(from_statuses)))
        if require_no_cancel:
            cond = and_(cond, runs.c.cancel_requested_at.is_(None))
        return self._exec(update(runs).where(cond).values(status=to_status, **values)) == 1

    def claim_run(self, from_status: str, to_status: str, worker_id: str) -> RowMapping | None:
        for candidate in self._all(
            select(runs.c.id).where(runs.c.status == from_status).order_by(runs.c.created_at).limit(5)
        ):
            if self.transition(candidate["id"], [from_status], to_status, worker_id=worker_id, started_at=now(), heartbeat_at=now()):
                return self.get_run(candidate["id"])
        return None

    def update_run(self, run_id: str, **values) -> None:
        self._exec(update(runs).where(runs.c.id == run_id).values(**values))

    def heartbeat(self, run_id: str, *, cost_usd: float | None = None) -> None:
        values: dict = {"heartbeat_at": now()}
        if cost_usd is not None:
            values["cost_usd"] = cost_usd
        self.update_run(run_id, **values)

    def request_cancel(self, run_id: str, by: str) -> str:
        """Record a cancellation request. Returns the run status afterwards."""
        stamp = now()
        self._exec(
            update(runs)
            .where(runs.c.id == run_id, runs.c.status.in_(ACTIVE_STATUSES), runs.c.cancel_requested_at.is_(None))
            .values(cancel_requested_at=stamp, cancel_requested_by=by)
        )
        # Runs no worker has picked up yet can be finished immediately.
        self.transition(
            run_id, ["pending", "ready"], "finished", outcome="cancelled", reason_code="cancelled",
            reason=f"Cancelled by @{by} before execution started.", finished_at=stamp,
        )
        run = self.get_run(run_id)
        return run["status"] if run else "missing"

    def cancel_requested(self, run_id: str) -> bool:
        row = self._one(select(runs.c.cancel_requested_at).where(runs.c.id == run_id))
        return bool(row and row["cancel_requested_at"])

    def stale_running_runs(self, older_than_seconds: int) -> list[RowMapping]:
        cutoff = now() - timedelta(seconds=older_than_seconds)
        return self._all(select(runs).where(runs.c.status == "running", runs.c.heartbeat_at < cutoff))

    def month_spend(self, installation_id: int) -> float:
        start = now().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
        row = self._one(
            select(func.coalesce(func.sum(runs.c.cost_usd), 0.0).label("total")).where(
                runs.c.installation_id == installation_id, runs.c.created_at >= start
            )
        )
        return float(row["total"]) if row else 0.0

    def find_run_by_pr(self, repo_id: int, pr_number: int) -> RowMapping | None:
        return self._one(select(runs).where(runs.c.repo_id == repo_id, runs.c.pr_number == pr_number))

    def runs_for_retention(self, older_than: datetime) -> list[RowMapping]:
        return self._all(
            select(runs).where(runs.c.status == "completed", runs.c.completed_at < older_than, runs.c.artifacts_deleted_at.is_(None))
        )

    # -- run events --------------------------------------------------------------

    def add_event(self, run_id: str, *, stage: str | None, kind: str, summary: str) -> None:
        self._exec(insert(run_events).values(run_id=run_id, ts=now(), stage=stage, kind=kind, summary=summary[:2000]))

    def list_events(self, run_id: str, *, limit: int = 300) -> list[RowMapping]:
        rows = self._all(select(run_events).where(run_events.c.run_id == run_id).order_by(run_events.c.id.desc()).limit(limit))
        return list(reversed(rows))

    # -- sessions ----------------------------------------------------------------

    def create_session(self, session_id: str, *, user_id: int | None, login: str, token: str | None, csrf: str, ttl_hours: int = 8) -> None:
        self._exec(
            insert(web_sessions).values(
                id=session_id, user_id=user_id, login=login, token=token, csrf=csrf, created_at=now(),
                expires_at=now() + timedelta(hours=ttl_hours),
            )
        )

    def get_session(self, session_id: str) -> RowMapping | None:
        row = self._one(select(web_sessions).where(web_sessions.c.id == session_id))
        if row is None or row["expires_at"] < now():
            return None
        return row

    def delete_session(self, session_id: str) -> None:
        self._exec(delete(web_sessions).where(web_sessions.c.id == session_id))

    def purge_sessions(self) -> None:
        self._exec(delete(web_sessions).where(web_sessions.c.expires_at < now()))

    # -- feedback (pilot) ---------------------------------------------------------

    def save_feedback(self, run_id: str, *, verdict: str, reason_category: str | None, notes: str, recorded_by: str) -> None:
        values = {"verdict": verdict, "reason_category": reason_category, "notes": notes[:5000], "recorded_by": recorded_by, "recorded_at": now()}
        with self.engine.begin() as conn:
            exists = conn.execute(select(feedback.c.run_id).where(feedback.c.run_id == run_id)).first()
            if exists:
                conn.execute(update(feedback).where(feedback.c.run_id == run_id).values(**values))
            else:
                conn.execute(insert(feedback).values(run_id=run_id, **values))

    def get_feedback(self, run_id: str) -> RowMapping | None:
        return self._one(select(feedback).where(feedback.c.run_id == run_id))

    def feedback_for_runs(self, run_ids: Iterable[str]) -> dict[str, RowMapping]:
        ids = list(run_ids)
        if not ids:
            return {}
        return {row["run_id"]: row for row in self._all(select(feedback).where(feedback.c.run_id.in_(ids)))}
