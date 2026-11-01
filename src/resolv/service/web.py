"""Web process: GitHub webhook endpoint and maintainer dashboard (Flask).

Dashboard users sign in with GitHub (OAuth through the GitHub App). Their own
user token decides what they may see and do: read access to view runs, write
access to start or cancel runs and record feedback, and maintain/admin access
to change a repository's validation settings. The web process never holds the
GitHub App private key.
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import json
import logging
import re
import secrets
import shlex
import time
from datetime import timedelta
from functools import wraps

from flask import Flask, Response, abort, flash, g, jsonify, redirect, render_template, request, session, url_for

from resolv.orchestrator import new_run_id
from resolv.service import metrics as metrics_mod
from resolv.service.db import ACTIVE_STATUSES, Database, RepoSettings, now
from resolv.service.github import GitHubClient, GitHubError, oauth_authorize_url, oauth_exchange_code
from resolv.service.messages import STAGES
from resolv.service.settings import SUPPORTED_PYTHON_VERSIONS, ServiceSettings
from resolv.service.storage import Storage, run_prefix
from resolv.service.webhooks import handle_event, verify_signature

log = logging.getLogger("resolv.web")

ROLE_ORDER = ("none", "read", "triage", "write", "maintain", "admin")
SESSION_COOKIE = "resolv_session"
_CACHE_SECONDS = 60
_BRANCH_RE = re.compile(r"^(?!/)(?!.*//)(?!.*\.\.)(?!.*@\{)[A-Za-z0-9._/-]{1,200}(?<![./])$")


def role_at_least(role: str, minimum: str) -> bool:
    return ROLE_ORDER.index(role if role in ROLE_ORDER else "none") >= ROLE_ORDER.index(minimum)


def _role_from_permissions(perms: dict | None) -> str:
    perms = perms or {}
    for role, key in (("admin", "admin"), ("maintain", "maintain"), ("write", "push"), ("triage", "triage"), ("read", "pull")):
        if perms.get(key):
            return role
    return "none"


class TokenBox:
    """Encrypts user OAuth tokens at rest with a key derived from RESOLV_SECRET_KEY."""

    def __init__(self, secret: str) -> None:
        from cryptography.fernet import Fernet

        key = base64.urlsafe_b64encode(hashlib.sha256(f"resolv-session-token:{secret}".encode()).digest())
        self._fernet = Fernet(key)

    def seal(self, token: str) -> str:
        return self._fernet.encrypt(token.encode()).decode()

    def open(self, sealed: str) -> str:
        return self._fernet.decrypt(sealed.encode()).decode()


def validate_settings_form(form, defaults: RepoSettings, max_cost: float) -> tuple[RepoSettings | None, list[str]]:
    errors: list[str] = []

    def integer(name: str, low: int, high: int, default: int) -> int:
        try:
            value = int(form.get(name, default))
        except (TypeError, ValueError):
            errors.append(f"{name.replace('_', ' ')} must be a whole number")
            return default
        if not low <= value <= high:
            errors.append(f"{name.replace('_', ' ')} must be between {low} and {high}")
        return value

    base_branch = form.get("base_branch", "").strip()
    if not _BRANCH_RE.match(base_branch):
        errors.append("base branch is not a valid branch name")
    python_version = form.get("python_version", "")
    if python_version not in SUPPORTED_PYTHON_VERSIONS:
        errors.append("unsupported Python version")
    install_commands = tuple(line.strip() for line in form.get("install_commands", "").splitlines() if line.strip())
    if len(install_commands) > 10:
        errors.append("at most 10 install commands are allowed")
    for command in install_commands:
        try:
            shlex.split(command)
        except ValueError:
            errors.append(f"install command cannot be parsed: {command[:80]}")
        if len(command) > 500:
            errors.append("install commands must be shorter than 500 characters")
    test_command = form.get("test_command", "").strip()
    try:
        argv = shlex.split(test_command)
    except ValueError:
        argv = []
    if not argv or not (argv[0] == "pytest" or argv[:3] == ["python", "-m", "pytest"]):
        errors.append("test command must start with 'python -m pytest' or 'pytest'")
    trigger_label = form.get("trigger_label", "").strip()
    if not trigger_label or len(trigger_label) > 50:
        errors.append("trigger label must be 1 to 50 characters")
    try:
        cost = float(form.get("max_cost_usd", defaults.max_cost_usd))
    except (TypeError, ValueError):
        cost = -1
    if not 0 < cost <= max_cost:
        errors.append(f"model spend cap must be between 0 and {max_cost:.2f} USD")
    settings = RepoSettings(
        base_branch=base_branch,
        python_version=python_version,
        install_commands=install_commands,
        test_command=test_command,
        trigger_label=trigger_label,
        max_changed_files=integer("max_changed_files", 1, 50, defaults.max_changed_files),
        max_changed_lines=integer("max_changed_lines", 10, 3000, defaults.max_changed_lines),
        max_cost_usd=cost,
        required_labels=tuple(x.strip() for x in form.get("required_labels", "").split(",") if x.strip()),
        excluded_labels=tuple(x.strip() for x in form.get("excluded_labels", "").split(",") if x.strip()),
        min_issue_chars=integer("min_issue_chars", 0, 10000, defaults.min_issue_chars),
    )
    if form.get("confirm") != "yes":
        errors.append("tick the confirmation box to confirm these settings")
    return (None if errors else settings), errors


def default_settings(repo, settings: ServiceSettings, max_cost: float) -> RepoSettings:
    return RepoSettings(
        base_branch=repo["default_branch"] or "main",
        python_version="3.12",
        install_commands=("python -m pip install -e .", "python -m pip install pytest"),
        test_command="python -m pytest",
        trigger_label=settings.trigger_label,
        max_cost_usd=max_cost,
        min_issue_chars=40,
    )


def create_app(
    settings: ServiceSettings,
    db: Database,
    storage: Storage,
    *,
    transport=None,
    model_disclosure: str = "",
    max_run_cost_usd: float = 3.0,
) -> Flask:
    app = Flask(__name__)
    secret = settings.secret_key
    if not secret:
        log.warning("RESOLV_SECRET_KEY is not set; using a random key (sessions reset on restart)")
        secret = secrets.token_urlsafe(32)
    app.config.update(
        SECRET_KEY=secret,
        MAX_CONTENT_LENGTH=25 * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True,
        SESSION_COOKIE_SAMESITE="Lax",
        SESSION_COOKIE_SECURE=settings.public_url.startswith("https://"),
        PERMANENT_SESSION_LIFETIME=timedelta(minutes=15),
    )
    tokens = TokenBox(secret)
    cache: dict[tuple, tuple[float, object]] = {}
    secure_cookie = settings.public_url.startswith("https://")

    def cached(key: tuple, producer):
        hit = cache.get(key)
        if hit and time.time() - hit[0] < _CACHE_SECONDS:
            return hit[1]
        value = producer()
        cache[key] = (time.time(), value)
        return value

    # -- session and authorization ---------------------------------------------------

    @app.before_request
    def load_user() -> None:
        g.user = None
        sid = request.cookies.get(SESSION_COOKIE)
        if sid:
            row = db.get_session(sid)
            if row is not None:
                g.user = {
                    "sid": sid, "login": row["login"], "user_id": row["user_id"], "csrf": row["csrf"],
                    "token": tokens.open(row["token"]) if row["token"] else None, "dev": row["token"] is None,
                }

    @app.after_request
    def security_headers(response: Response) -> Response:
        response.headers.setdefault("Content-Security-Policy", "default-src 'self'; style-src 'self'; img-src 'self' data:; form-action 'self' https:; frame-ancestors 'none'")
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "same-origin")
        return response

    @app.context_processor
    def inject() -> dict:
        return {
            "user": g.get("user"), "dev_login": settings.dev_login, "csrf_token": (g.user or {}).get("csrf", "") if g.get("user") else "",
            "outcome_labels": {"draft_pr_created": "Draft PR created", "blocked": "Blocked", "failed": "Failed", "cancelled": "Cancelled"},
        }

    def login_required(view):
        @wraps(view)
        def wrapper(*args, **kwargs):
            if g.user is None:
                return redirect(url_for("login", next=request.path))
            return view(*args, **kwargs)

        return wrapper

    def check_csrf() -> None:
        sent = request.form.get("csrf_token", "")
        if not g.user or not hmac.compare_digest(sent, g.user["csrf"]):
            abort(400, "invalid or missing CSRF token")

    def user_client() -> GitHubClient:
        return GitHubClient(g.user["token"], api_url=settings.api_url, transport=transport)

    def accessible_repos() -> list:
        if g.user["dev"]:
            return db.list_repositories()

        def fetch():
            ids: set[int] = set()
            client = user_client()
            for inst in client.paginate("/user/installations", key="installations"):
                for repo in client.paginate(f"/user/installations/{inst['id']}/repositories", key="repositories"):
                    ids.add(repo["id"])
            return ids

        try:
            ids = cached(("repos", g.user["sid"]), fetch)
        except GitHubError as exc:
            log.warning("listing repositories failed: %s", exc)
            ids = set()
        return db.list_repositories(ids)

    def repo_role(repo) -> str:
        if g.user["dev"]:
            return "admin"

        def fetch():
            try:
                data = user_client().get(f"/repos/{repo['full_name']}")
            except GitHubError as exc:
                if exc.status in (403, 404):
                    return "none"
                raise
            if data.get("id") != repo["id"]:
                return "none"
            return _role_from_permissions(data.get("permissions"))

        return cached(("role", g.user["sid"], repo["id"]), fetch)

    def load_repo(owner: str, name: str, minimum: str = "read"):
        repo = db.get_repository_by_name(f"{owner}/{name}")
        if repo is None:
            abort(404)
        role = repo_role(repo)
        if not role_at_least(role, minimum):
            abort(404 if role == "none" else 403)
        return repo, role

    def load_run(run_id: str, minimum: str = "read"):
        run = db.get_run(run_id)
        if run is None:
            abort(404)
        repo = db.get_repository(run["repo_id"])
        role = repo_role(repo)
        if not role_at_least(role, minimum):
            abort(404 if role == "none" else 403)
        return run, repo, role

    # -- webhook ------------------------------------------------------------------------

    @app.post("/webhook")
    def webhook():
        body = request.get_data(cache=False)
        if not verify_signature(settings.webhook_secret, body, request.headers.get("X-Hub-Signature-256")):
            return jsonify({"error": "invalid signature"}), 401
        event = request.headers.get("X-GitHub-Event", "")
        delivery = request.headers.get("X-GitHub-Delivery", "")
        if not event or not delivery:
            return jsonify({"error": "missing event headers"}), 400
        try:
            payload = json.loads(body)
        except ValueError:
            return jsonify({"error": "invalid JSON"}), 400
        result = handle_event(db, settings, event, delivery, payload)
        return jsonify({"message": result.message, "run_id": result.run_id}), result.status

    @app.get("/healthz")
    def healthz():
        return "ok"

    # -- login --------------------------------------------------------------------------

    def start_session(*, login: str, user_id: int | None, token: str | None):
        sid = secrets.token_urlsafe(32)
        db.create_session(sid, user_id=user_id, login=login, token=tokens.seal(token) if token else None, csrf=secrets.token_urlsafe(24))
        target = session.pop("next", None) or "/"
        if not target.startswith("/") or target.startswith("//"):
            target = "/"
        response = redirect(target)
        response.set_cookie(SESSION_COOKIE, sid, httponly=True, samesite="Lax", secure=secure_cookie, max_age=8 * 3600)
        return response

    @app.get("/login")
    def login():
        next_path = request.args.get("next", "/")
        session["next"] = next_path if next_path.startswith("/") and not next_path.startswith("//") else "/"
        if settings.dev_login:
            return render_template("login.html")
        if not settings.client_id:
            abort(503, "GitHub OAuth is not configured (GITHUB_CLIENT_ID)")
        url, state = oauth_authorize_url(settings.web_url, settings.client_id, settings.public_url + "/oauth/callback")
        session["oauth_state"] = state
        return redirect(url)

    @app.post("/login/dev")
    def login_dev():
        if not settings.dev_login:
            abort(404)
        return start_session(login="dev-admin", user_id=None, token=None)

    @app.get("/oauth/callback")
    def oauth_callback():
        state = session.pop("oauth_state", None)
        if not state or not hmac.compare_digest(state, request.args.get("state", "")):
            abort(400, "invalid OAuth state")
        try:
            token = oauth_exchange_code(
                settings.web_url, settings.client_id, settings.client_secret, request.args.get("code", ""),
                settings.public_url + "/oauth/callback", transport=transport,
            )
            me = GitHubClient(token, api_url=settings.api_url, transport=transport).get("/user")
        except GitHubError as exc:
            abort(502, f"GitHub sign-in failed: {exc.message}")
        return start_session(login=me["login"], user_id=me["id"], token=token)

    @app.post("/logout")
    @login_required
    def logout():
        check_csrf()
        db.delete_session(g.user["sid"])
        response = redirect(url_for("login"))
        response.delete_cookie(SESSION_COOKIE)
        return response

    # -- pages ----------------------------------------------------------------------------

    @app.get("/")
    @login_required
    def index():
        repos = accessible_repos()
        rows = []
        for repo in repos:
            repo_settings = db.get_settings(repo["id"])
            last = db.list_runs([repo["id"]], limit=1)
            rows.append({"repo": repo, "settings": repo_settings, "last_run": last[0] if last else None})
        return render_template("index.html", rows=rows, require_opt_in=settings.require_opt_in)

    @app.route("/repos/<owner>/<name>/settings", methods=["GET", "POST"])
    @login_required
    def repo_settings(owner: str, name: str):
        repo, role = load_repo(owner, name)
        current = db.get_settings(repo["id"]) or default_settings(repo, settings, max_run_cost_usd)
        can_edit = role_at_least(role, "maintain")
        errors: list[str] = []
        form_values = current
        if request.method == "POST":
            check_csrf()
            if not can_edit:
                abort(403)
            parsed, errors = validate_settings_form(request.form, current, max_run_cost_usd)
            if parsed is not None:
                db.save_settings(repo["id"], parsed, g.user["login"])
                flash("Settings confirmed. Runs can now be started for this repository.", "success")
                return redirect(url_for("repo_settings", owner=owner, name=name))
            form_values = None
        return render_template(
            "settings.html", repo=repo, role=role, can_edit=can_edit, current=db.get_settings(repo["id"]), values=form_values,
            form=request.form if request.method == "POST" else None, errors=errors, python_versions=SUPPORTED_PYTHON_VERSIONS,
            max_cost=max_run_cost_usd, disclosure=model_disclosure, require_opt_in=settings.require_opt_in,
            allow_private=settings.allow_private_repos,
        )

    @app.get("/repos/<owner>/<name>/runs")
    @login_required
    def repo_runs(owner: str, name: str):
        repo, role = load_repo(owner, name)
        return render_template(
            "runs.html", repo=repo, role=role, runs=db.list_runs([repo["id"]], limit=200),
            can_start=role_at_least(role, "write"), repo_settings=db.get_settings(repo["id"]), now=now(),
        )

    @app.post("/repos/<owner>/<name>/runs")
    @login_required
    def start_run(owner: str, name: str):
        check_csrf()
        repo, _ = load_repo(owner, name, "write")
        try:
            number = int(request.form.get("issue_number", ""))
            if number <= 0:
                raise ValueError
        except ValueError:
            flash("Enter a valid issue number.", "error")
            return redirect(url_for("repo_runs", owner=owner, name=name))
        run = db.create_run(
            run_id=new_run_id(), repo_id=repo["id"], installation_id=repo["installation_id"], issue_number=number,
            trigger="dashboard", triggered_by=g.user["login"],
        )
        if run is None:
            flash(f"Issue #{number} already has an active run.", "error")
            return redirect(url_for("repo_runs", owner=owner, name=name))
        return redirect(url_for("run_detail", run_id=run["id"]))

    def load_report(run) -> dict | None:
        raw = storage.get_bytes(f"{run_prefix(run['installation_id'], run['repo_id'], run['id'])}/artifacts/report.json")
        try:
            return json.loads(raw) if raw else None
        except ValueError:
            return None

    @app.get("/runs/<run_id>")
    @login_required
    def run_detail(run_id: str):
        run, repo, role = load_run(run_id)
        prefix = f"{run_prefix(run['installation_id'], run['repo_id'], run['id'])}/artifacts"
        report = load_report(run)
        diff_raw = storage.get_bytes(f"{prefix}/patch.diff") or storage.get_bytes(f"{prefix}/candidate.diff")
        artifacts = [key[len(prefix) + 1:] for key in storage.list(prefix)]
        if run["finished_at"] and run["started_at"]:
            elapsed = run["elapsed_seconds"] or (run["finished_at"] - run["started_at"]).total_seconds()
        elif run["started_at"]:
            elapsed = (now() - run["started_at"]).total_seconds()
        else:
            elapsed = None
        stage_names = [s for s, _ in STAGES]
        return render_template(
            "run.html", run=run, repo=repo, role=role, report=report, diff=diff_raw.decode("utf-8", errors="replace") if diff_raw else None,
            artifacts=artifacts, events=db.list_events(run_id), elapsed=elapsed, stages=STAGES,
            stage_index=stage_names.index(run["stage"]) if run["stage"] in stage_names else -1,
            active=run["status"] in ACTIVE_STATUSES, can_write=role_at_least(role, "write"),
            feedback=db.get_feedback(run_id), verdicts=metrics_mod.FEEDBACK_VERDICTS, reasons=metrics_mod.REJECTION_REASONS,
            github=settings.web_url,
        )

    @app.post("/runs/<run_id>/cancel")
    @login_required
    def cancel_run(run_id: str):
        check_csrf()
        run, _, _ = load_run(run_id, "write")
        if run["status"] not in ACTIVE_STATUSES:
            flash("This run has already finished.", "error")
        else:
            db.request_cancel(run_id, g.user["login"])
            db.add_event(run_id, stage=run["stage"], kind="cancel_requested", summary=f"Cancellation requested by @{g.user['login']}")
            flash("Cancellation requested. Running commands are terminated and nothing will be published.", "success")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/feedback")
    @login_required
    def run_feedback(run_id: str):
        check_csrf()
        run, _, _ = load_run(run_id, "write")
        verdict = request.form.get("verdict", "")
        reason = request.form.get("reason_category") or None
        if verdict not in metrics_mod.FEEDBACK_VERDICTS or (reason and reason not in metrics_mod.REJECTION_REASONS):
            abort(400, "invalid feedback")
        if verdict == "rejected" and not reason:
            flash("Choose a rejection reason.", "error")
            return redirect(url_for("run_detail", run_id=run_id))
        db.save_feedback(run_id, verdict=verdict, reason_category=reason if verdict == "rejected" else None,
                         notes=request.form.get("notes", ""), recorded_by=g.user["login"])
        flash("Feedback recorded. Thank you.", "success")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.post("/runs/<run_id>/delete-artifacts")
    @login_required
    def delete_artifacts(run_id: str):
        check_csrf()
        run, _, _ = load_run(run_id, "write")
        if run["status"] in ACTIVE_STATUSES:
            abort(409, "the run is still active")
        storage.delete_prefix(run_prefix(run["installation_id"], run["repo_id"], run_id))
        db.update_run(run_id, artifacts_deleted_at=now())
        flash("Run artifacts deleted.", "success")
        return redirect(url_for("run_detail", run_id=run_id))

    @app.get("/runs/<run_id>/artifacts/<path:name>")
    @login_required
    def artifact(run_id: str, name: str):
        run, _, _ = load_run(run_id)
        prefix = f"{run_prefix(run['installation_id'], run['repo_id'], run['id'])}/artifacts"
        if f"{prefix}/{name}" not in storage.list(prefix):
            abort(404)
        data = storage.get_bytes(f"{prefix}/{name}")
        return Response(data, mimetype="text/plain", headers={"Content-Disposition": "inline"})

    @app.get("/metrics")
    @login_required
    def metrics():
        repos = accessible_repos()
        per_repo = [(repo, metrics_mod.compute_metrics(db, [repo["id"]])) for repo in repos]
        return render_template(
            "metrics.html", total=metrics_mod.compute_metrics(db, [r["id"] for r in repos]), per_repo=per_repo,
            reasons=metrics_mod.REJECTION_REASONS, verdicts=metrics_mod.FEEDBACK_VERDICTS,
        )

    @app.errorhandler(400)
    @app.errorhandler(403)
    @app.errorhandler(404)
    @app.errorhandler(409)
    @app.errorhandler(502)
    @app.errorhandler(503)
    def error_page(exc):
        return render_template("error.html", error=exc), exc.code

    return app
