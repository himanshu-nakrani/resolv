"""Worker: runs the orchestrator for prepared runs. Holds no GitHub credentials.

Inputs (repository snapshot and issue text) come from artifact storage, where
the gateway put them. Repository code only ever executes inside the sandbox
configured in the operator's run config (Docker by default). Progress is
mirrored into the database for the dashboard; cancellation requests are polled
and terminate the sandbox.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import socket
import tempfile
import threading
import traceback
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

from resolv.config import RunConfig
from resolv.errors import ConfigError
from resolv.model import ModelClient
from resolv.orchestrator import Issue, Orchestrator, RunRequest
from resolv.service.db import Database, now
from resolv.service.gateway import extract_snapshot
from resolv.service.settings import ServiceSettings
from resolv.service.storage import Storage, put_dir, run_prefix

log = logging.getLogger("resolv.worker")

PROGRESS_STAGES = frozenset({"preflight", "setup", "baseline", "investigate", "reproduce", "patch", "validate", "deliver"})


def build_run_config(base: RunConfig, snapshot: dict) -> RunConfig:
    """Operator base config + maintainer settings; maintainers can lower but never raise the cost cap."""
    env = replace(
        base.environment,
        python_version=snapshot["python_version"],
        install_commands=tuple(snapshot["install_commands"]),
        test_command=snapshot["test_command"],
    )
    limits = replace(
        base.limits,
        max_changed_files=int(snapshot["max_changed_files"]),
        max_changed_lines=int(snapshot["max_changed_lines"]),
        max_cost_usd=min(float(snapshot["max_cost_usd"]), base.limits.max_cost_usd),
    )
    return replace(base, environment=env, limits=limits).validate()


def summarize_event(event: dict) -> str | None:
    kind = event.get("kind")
    if kind == "command":
        status = event.get("error") or ("timed out" if event.get("timed_out") else f"exit {event.get('exit_code')}")
        return f"$ {' '.join(event.get('argv', []))[:300]} → {status} ({event.get('duration_seconds')}s)"
    if kind == "tool_call":
        flag = " (rejected)" if event.get("is_error") else ""
        return f"{event.get('name')} {event.get('arguments', '')[:200]}{flag}"
    if kind == "stage_started":
        return f"Started {event.get('name')}"
    if kind == "stage_finished":
        summary = event.get("summary") or event.get("status", "")
        return f"Finished {event.get('name')}: {summary}"[:500]
    if kind == "run_finished":
        return f"{event.get('outcome')}: {event.get('reason')}"[:1000]
    if kind == "model_response" and event.get("text"):
        return "Agent: " + str(event["text"])[:300]
    return None


class Worker:
    def __init__(
        self,
        settings: ServiceSettings,
        db: Database,
        storage: Storage,
        base_config: RunConfig,
        *,
        environ: Mapping[str, str] | None = None,
        client_factory: Callable[[RunConfig], ModelClient] | None = None,
        worker_id: str | None = None,
    ) -> None:
        if base_config.sandbox.backend != "docker" and not settings.allow_local_sandbox:
            raise ConfigError(
                "the service requires the Docker sandbox; set RESOLV_ALLOW_LOCAL_SANDBOX=1 only for development with trusted repositories"
            )
        self.settings = settings
        self.db = db
        self.storage = storage
        self.base_config = base_config
        self.environ = dict(os.environ if environ is None else environ)
        self.client_factory = client_factory
        self.worker_id = worker_id or f"{socket.gethostname()}-{os.getpid()}"

    def run_once(self) -> bool:
        run = self.db.claim_run("ready", "running", self.worker_id)
        if run is None:
            return False
        self.process(run)
        return True

    def serve_forever(self, stop: threading.Event) -> None:
        log.info("worker %s started", self.worker_id)
        while not stop.is_set():
            try:
                if not self.run_once():
                    stop.wait(self.settings.poll_seconds)
            except Exception:  # noqa: BLE001
                log.error("worker iteration failed:\n%s", traceback.format_exc())
                stop.wait(self.settings.poll_seconds)

    def _listener(self, run_id: str):
        def on_event(event: dict) -> None:
            kind = event.get("kind")
            if kind == "stage_started" and event.get("name") in PROGRESS_STAGES:
                self.db.update_run(run_id, stage=event["name"])
            summary = summarize_event(event)
            if summary:
                self.db.add_event(run_id, stage=event.get("stage"), kind=kind, summary=summary)

        return on_event

    def process(self, run) -> None:
        run_id = run["id"]
        prefix = run_prefix(run["installation_id"], run["repo_id"], run_id)
        tmp = Path(tempfile.mkdtemp(prefix=f"resolv-worker-{run_id}-"))
        stop = threading.Event()
        orchestrator: Orchestrator | None = None
        try:
            snapshot = self.storage.get_bytes(f"{prefix}/input/snapshot.tar.gz")
            issue_raw = self.storage.get_bytes(f"{prefix}/input/issue.json")
            if snapshot is None or issue_raw is None:
                raise RuntimeError("run inputs are missing from storage")
            source = extract_snapshot(snapshot, tmp / "input")
            issue = json.loads(issue_raw)
            config = build_run_config(self.base_config, json.loads(run["settings_snapshot"]))
            request = RunRequest(
                repo=source,
                issue=Issue(title=issue["title"], body=issue["body"], url=issue.get("url"), source=f"issue #{run['issue_number']}"),
                config=config,
                out_dir=tmp / "out",
                ref=run["base_commit"],
                run_id=run_id,
            )
            client = self.client_factory(config) if self.client_factory else None
            orchestrator = Orchestrator(request, client=client, environ=self.environ, listeners=[self._listener(run_id)])
            watcher = threading.Thread(target=self._watch, args=(run_id, orchestrator, stop), daemon=True)
            watcher.start()
            record = orchestrator.run()
            stop.set()
            put_dir(self.storage, f"{prefix}/artifacts", orchestrator.run_dir)
            self.db.transition(
                run_id, ["running"], "finished", outcome=record.outcome, reason_code=record.reason_code,
                reason=record.reason, finished_at=now(), cost_usd=orchestrator.budget.estimated_cost_usd,
                elapsed_seconds=orchestrator.budget.elapsed_seconds, base_commit=record.base_commit or run["base_commit"],
            )
        except Exception as exc:  # noqa: BLE001 - every run must end with an outcome
            log.error("run %s failed in the worker:\n%s", run_id, traceback.format_exc())
            self.db.transition(
                run_id, ["running"], "finished", outcome="failed", reason_code="worker_error",
                reason=f"The worker could not complete the run: {exc}", finished_at=now(),
                cost_usd=orchestrator.budget.estimated_cost_usd if orchestrator else 0.0,
            )
        finally:
            stop.set()
            shutil.rmtree(tmp, ignore_errors=True)

    def _watch(self, run_id: str, orchestrator: Orchestrator, stop: threading.Event) -> None:
        while not stop.wait(2.0):
            try:
                self.db.heartbeat(run_id, cost_usd=orchestrator.budget.estimated_cost_usd)
                if self.db.cancel_requested(run_id):
                    orchestrator.cancel()
            except Exception:  # noqa: BLE001
                log.warning("watcher for %s failed:\n%s", run_id, traceback.format_exc())
