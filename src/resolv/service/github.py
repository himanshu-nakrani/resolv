"""GitHub REST client, GitHub App authentication, and OAuth (stdlib HTTP).

Installation tokens are minted per operation, scoped to a single repository
and to the minimum permissions that operation needs, and expire within an hour.
"""

from __future__ import annotations

import json
import secrets
import time
import urllib.error
import urllib.parse
import urllib.request
from collections.abc import Callable
from dataclasses import dataclass

USER_AGENT = "resolv-github-app"
WRITE_ROLES = frozenset({"admin", "maintain", "write"})
READ_PERMISSIONS = {"contents": "read", "issues": "write", "metadata": "read"}
PUBLISH_PERMISSIONS = {"contents": "write", "pull_requests": "write", "issues": "write", "metadata": "read"}


class GitHubError(Exception):
    def __init__(self, status: int, message: str, path: str = "") -> None:
        super().__init__(f"GitHub API {status} on {path}: {message}")
        self.status = status
        self.message = message
        self.path = path


Transport = Callable[..., object]


def _request(
    transport: Transport, method: str, url: str, *, token: str | None = None, auth_scheme: str = "Bearer",
    body: dict | None = None, form: dict | None = None, accept: str = "application/vnd.github+json",
    timeout: float = 30,
) -> tuple[int, dict | list | None]:
    headers = {"Accept": accept, "User-Agent": USER_AGENT, "X-GitHub-Api-Version": "2022-11-28"}
    data = None
    if token:
        headers["Authorization"] = f"{auth_scheme} {token}"
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        headers["Content-Type"] = "application/json"
    elif form is not None:
        data = urllib.parse.urlencode(form).encode("utf-8")
        headers["Content-Type"] = "application/x-www-form-urlencoded"
    request = urllib.request.Request(url, data=data, headers=headers, method=method)
    path = urllib.parse.urlsplit(url).path
    try:
        with transport(request, timeout=timeout) as response:
            raw = response.read()
            status = getattr(response, "status", 200)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        try:
            message = json.loads(detail).get("message", detail)
        except (ValueError, AttributeError):
            message = detail
        raise GitHubError(exc.code, str(message)[:500], path) from None
    except (urllib.error.URLError, TimeoutError, ConnectionError) as exc:
        raise GitHubError(0, f"network error: {exc}", path) from None
    if not raw:
        return status, None
    try:
        return status, json.loads(raw)
    except ValueError:
        raise GitHubError(status, "invalid JSON response", path) from None


class GitHubClient:
    """REST client bound to one token (installation or user)."""

    def __init__(self, token: str, *, api_url: str = "https://api.github.com", transport: Transport | None = None) -> None:
        self.token = token
        self.api_url = api_url.rstrip("/")
        self.transport = transport or urllib.request.urlopen

    def request(self, method: str, path: str, body: dict | None = None) -> dict | list | None:
        return _request(self.transport, method, self.api_url + path, token=self.token, body=body)[1]

    def get(self, path: str) -> dict | list | None:
        return self.request("GET", path)

    def paginate(self, path: str, *, key: str | None = None, max_pages: int = 10) -> list:
        items: list = []
        sep = "&" if "?" in path else "?"
        for page in range(1, max_pages + 1):
            data = self.get(f"{path}{sep}per_page=100&page={page}")
            batch = data.get(key, []) if key and isinstance(data, dict) else data or []
            items.extend(batch)
            if len(batch) < 100:
                break
        return items

    # -- repository helpers ----------------------------------------------------

    @staticmethod
    def _q(value: str) -> str:
        return urllib.parse.quote(value, safe="")

    def collaborator_role(self, full_name: str, login: str) -> str:
        """Effective role of a user on a repository: admin, maintain, write, triage, read, or none."""
        try:
            data = self.get(f"/repos/{full_name}/collaborators/{self._q(login)}/permission")
        except GitHubError as exc:
            if exc.status == 404:
                return "none"
            raise
        role = data.get("role_name") or data.get("permission") or "none"
        return "write" if role == "push" else role

    def get_repo(self, full_name: str) -> dict:
        return self.get(f"/repos/{full_name}")

    def get_issue(self, full_name: str, number: int) -> dict:
        return self.get(f"/repos/{full_name}/issues/{number}")

    def issue_comments(self, full_name: str, number: int) -> list[dict]:
        return self.paginate(f"/repos/{full_name}/issues/{number}/comments", max_pages=3)

    def branch_head(self, full_name: str, branch: str) -> str:
        data = self.get(f"/repos/{full_name}/git/ref/heads/{urllib.parse.quote(branch, safe='/')}")
        return data["object"]["sha"]

    def create_comment(self, full_name: str, number: int, body: str) -> int:
        return self.request("POST", f"/repos/{full_name}/issues/{number}/comments", {"body": body})["id"]

    def update_comment(self, full_name: str, comment_id: int, body: str) -> None:
        self.request("PATCH", f"/repos/{full_name}/issues/comments/{comment_id}", {"body": body})

    def remove_label(self, full_name: str, number: int, label: str) -> None:
        try:
            self.request("DELETE", f"/repos/{full_name}/issues/{number}/labels/{self._q(label)}")
        except GitHubError as exc:
            if exc.status != 404:
                raise


@dataclass
class _CachedToken:
    token: str
    expires: float


class GitHubApp:
    """Mints scoped installation tokens from the App's private key."""

    def __init__(
        self, app_id: str, private_key: str, *, api_url: str = "https://api.github.com", transport: Transport | None = None,
    ) -> None:
        self.app_id = app_id
        self.private_key = private_key
        self.api_url = api_url.rstrip("/")
        self.transport = transport or urllib.request.urlopen
        self._cache: dict[tuple, _CachedToken] = {}

    def jwt(self) -> str:
        import jwt  # PyJWT, service extra

        issued = int(time.time()) - 60
        return jwt.encode({"iat": issued, "exp": issued + 540, "iss": str(self.app_id)}, self.private_key, algorithm="RS256")

    def installation_token(self, installation_id: int, *, repository_ids: list[int], permissions: dict[str, str]) -> str:
        key = (installation_id, tuple(sorted(repository_ids)), tuple(sorted(permissions.items())))
        cached = self._cache.get(key)
        if cached and cached.expires - time.time() > 300:
            return cached.token
        _, data = _request(
            self.transport, "POST", f"{self.api_url}/app/installations/{installation_id}/access_tokens",
            token=self.jwt(), body={"repository_ids": repository_ids, "permissions": permissions},
        )
        token = data["token"]
        # Tokens last one hour; refresh well before expiry.
        self._cache[key] = _CachedToken(token, time.time() + 3300)
        return token

    def client(self, installation_id: int, repo_id: int, permissions: dict[str, str]) -> GitHubClient:
        token = self.installation_token(installation_id, repository_ids=[repo_id], permissions=permissions)
        return GitHubClient(token, api_url=self.api_url, transport=self.transport)


# -- OAuth (dashboard login) ----------------------------------------------------


def oauth_authorize_url(web_url: str, client_id: str, redirect_uri: str) -> tuple[str, str]:
    state = secrets.token_urlsafe(24)
    query = urllib.parse.urlencode({"client_id": client_id, "redirect_uri": redirect_uri, "state": state})
    return f"{web_url.rstrip('/')}/login/oauth/authorize?{query}", state


def oauth_exchange_code(
    web_url: str, client_id: str, client_secret: str, code: str, redirect_uri: str, transport: Transport | None = None
) -> str:
    _, data = _request(
        transport or urllib.request.urlopen, "POST", f"{web_url.rstrip('/')}/login/oauth/access_token",
        form={"client_id": client_id, "client_secret": client_secret, "code": code, "redirect_uri": redirect_uri},
        accept="application/json",
    )
    if not isinstance(data, dict) or "access_token" not in data:
        error = data.get("error_description") or data.get("error") if isinstance(data, dict) else "unknown error"
        raise GitHubError(400, f"OAuth code exchange failed: {error}", "/login/oauth/access_token")
    return data["access_token"]
