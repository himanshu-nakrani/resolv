"""Service settings, read from environment variables."""

from __future__ import annotations

import os
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path

from resolv.errors import ConfigError

SUPPORTED_PYTHON_VERSIONS = ("3.9", "3.10", "3.11", "3.12", "3.13")


def _bool(value: str | None, default: bool) -> bool:
    if value is None or value == "":
        return default
    return value.strip().lower() in ("1", "true", "yes", "on")


def _float(env: Mapping[str, str], name: str, default: float) -> float:
    try:
        return float(env.get(name, default))
    except ValueError as exc:
        raise ConfigError(f"{name} must be a number") from exc


def _int(env: Mapping[str, str], name: str, default: int) -> int:
    try:
        return int(env.get(name, default))
    except ValueError as exc:
        raise ConfigError(f"{name} must be an integer") from exc


@dataclass(frozen=True)
class ServiceSettings:
    database_url: str = "sqlite:///resolv-service.db"
    storage: str = "resolv-data"
    public_url: str = "http://localhost:8000"
    secret_key: str = ""
    # GitHub App (gateway)
    app_id: str = ""
    private_key: str = field(default="", repr=False)
    # Web
    webhook_secret: str = field(default="", repr=False)
    client_id: str = ""
    client_secret: str = field(default="", repr=False)
    dev_login: bool = False
    # GitHub endpoints (override for GitHub Enterprise Server or tests)
    api_url: str = "https://api.github.com"
    web_url: str = "https://github.com"
    git_url_template: str = "https://github.com/{full_name}.git"
    # Policy
    trigger_label: str = "ai-fix"
    require_opt_in: bool = True
    allow_private_repos: bool = False
    monthly_budget_usd: float = 100.0
    retention_days: int = 7
    run_config_path: str = ""
    allow_local_sandbox: bool = False
    poll_seconds: float = 2.0
    stale_after_seconds: int = 300

    @classmethod
    def from_env(cls, env: Mapping[str, str] | None = None) -> ServiceSettings:
        env = os.environ if env is None else env
        key = env.get("GITHUB_APP_PRIVATE_KEY", "")
        key_path = env.get("GITHUB_APP_PRIVATE_KEY_PATH", "")
        if not key and key_path:
            try:
                key = Path(key_path).read_text(encoding="utf-8")
            except OSError as exc:
                raise ConfigError(f"cannot read GITHUB_APP_PRIVATE_KEY_PATH: {exc}") from exc
        defaults = cls()
        return cls(
            database_url=env.get("RESOLV_DATABASE_URL", defaults.database_url),
            storage=env.get("RESOLV_STORAGE", defaults.storage),
            public_url=env.get("RESOLV_PUBLIC_URL", defaults.public_url).rstrip("/"),
            secret_key=env.get("RESOLV_SECRET_KEY", ""),
            app_id=env.get("GITHUB_APP_ID", ""),
            private_key=key,
            webhook_secret=env.get("GITHUB_WEBHOOK_SECRET", ""),
            client_id=env.get("GITHUB_CLIENT_ID", ""),
            client_secret=env.get("GITHUB_CLIENT_SECRET", ""),
            dev_login=_bool(env.get("RESOLV_DEV_LOGIN"), False),
            api_url=env.get("GITHUB_API_URL", defaults.api_url).rstrip("/"),
            web_url=env.get("GITHUB_WEB_URL", defaults.web_url).rstrip("/"),
            git_url_template=env.get("RESOLV_GIT_URL_TEMPLATE", defaults.git_url_template),
            trigger_label=env.get("RESOLV_TRIGGER_LABEL", defaults.trigger_label),
            require_opt_in=_bool(env.get("RESOLV_REQUIRE_OPT_IN"), True),
            allow_private_repos=_bool(env.get("RESOLV_ALLOW_PRIVATE_REPOS"), False),
            monthly_budget_usd=_float(env, "RESOLV_MONTHLY_BUDGET_USD", defaults.monthly_budget_usd),
            retention_days=_int(env, "RESOLV_ARTIFACT_RETENTION_DAYS", defaults.retention_days),
            run_config_path=env.get("RESOLV_RUN_CONFIG", ""),
            allow_local_sandbox=_bool(env.get("RESOLV_ALLOW_LOCAL_SANDBOX"), False),
            poll_seconds=_float(env, "RESOLV_POLL_SECONDS", defaults.poll_seconds),
            stale_after_seconds=_int(env, "RESOLV_STALE_AFTER_SECONDS", defaults.stale_after_seconds),
        )

    def require(self, *names: str) -> None:
        env_names = {
            "secret_key": "RESOLV_SECRET_KEY",
            "app_id": "GITHUB_APP_ID",
            "private_key": "GITHUB_APP_PRIVATE_KEY or GITHUB_APP_PRIVATE_KEY_PATH",
            "webhook_secret": "GITHUB_WEBHOOK_SECRET",
            "client_id": "GITHUB_CLIENT_ID",
            "client_secret": "GITHUB_CLIENT_SECRET",
        }
        missing = [env_names.get(n, n) for n in names if not getattr(self, n)]
        if missing:
            raise ConfigError("missing required settings: " + ", ".join(missing))

    def run_url(self, run_id: str) -> str:
        return f"{self.public_url}/runs/{run_id}"
