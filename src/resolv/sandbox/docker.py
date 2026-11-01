"""Docker backend: disposable, network-less containers for repository code.

Lifecycle for one run:

1. Setup: a setup container gets a copy of the checkout (plus the pytest
   plugin) and runs the configured install commands in a virtualenv. Its only
   network is an internal Docker network whose sole other member is the
   allowlisting proxy (resolv.sandbox.proxy), so installs reach the package
   index and nothing else. The container is then committed to a per-run image.
2. Every test command runs in a *fresh* container created from that image with
   ``--network none``, receives the files the agent has written, and is removed
   afterwards. Nothing a test does can persist into later runs.

Containers get no credentials (only an explicit, minimal environment), no
host mounts, no Docker socket, all capabilities dropped, no-new-privileges, a
non-root user, and CPU, memory, process-count, and file-size limits. An
optional runtime such as gVisor (``runsc``) adds a user-space kernel.
"""

from __future__ import annotations

import io
import itertools
import os
import shlex
import subprocess
import tarfile
import time
from pathlib import Path

from resolv.budget import Budget
from resolv.config import RunConfig
from resolv.events import EventLog
from resolv.executor import CommandResult, CommandRunner
from resolv.redact import Redactor
from resolv.sandbox import PLUGIN_FILE, SetupError
from resolv.workspace import Workspace

WORK = "/work"
REPO = f"{WORK}/repo"
PLUGIN_DIR = f"{WORK}/plugin"
RESULTS_DIR = f"{WORK}/results"
VENV = f"{WORK}/venv"
HOME = f"{WORK}/home"
PROXY_PORT = 3128
PROXY_SCRIPT = Path(__file__).resolve().parent / "proxy.py"
MAX_RESULTS_BYTES = 64 * 1024 * 1024
_DOCKER_HOST_VARS = (
    "PATH", "HOME", "DOCKER_HOST", "DOCKER_CONFIG", "DOCKER_CONTEXT", "DOCKER_CERT_PATH", "DOCKER_TLS_VERIFY",
    "XDG_RUNTIME_DIR",
)


class DockerError(Exception):
    pass


class DockerCLI:
    """Thin wrapper for short, trusted ``docker`` plumbing calls."""

    def __init__(self, command: str = "docker") -> None:
        self.command = command
        self.env = {k: os.environ[k] for k in _DOCKER_HOST_VARS if k in os.environ}

    def __call__(
        self, *args: str, input: bytes | None = None, timeout: float = 300, check: bool = True
    ) -> subprocess.CompletedProcess:
        try:
            proc = subprocess.run(
                [self.command, *args], input=input, capture_output=True, timeout=timeout, env=self.env
            )
        except FileNotFoundError as exc:
            raise DockerError(f"docker CLI not found ({self.command})") from exc
        except subprocess.TimeoutExpired as exc:
            raise DockerError(f"docker {args[0]} timed out") from exc
        if check and proc.returncode != 0:
            message = proc.stderr.decode("utf-8", errors="replace").strip() or proc.stdout.decode(errors="replace")
            raise DockerError(f"docker {args[0]} failed: {message[:1000]}")
        return proc

    def available(self) -> bool:
        try:
            return self("version", "--format", "{{.Server.Version}}", timeout=30, check=False).returncode == 0
        except DockerError:
            return False


def _owned(uid: int, gid: int):
    def apply(info: tarfile.TarInfo) -> tarfile.TarInfo:
        info.uid, info.gid, info.uname, info.gname = uid, gid, "", ""
        return info

    return apply


def _dir_entry(name: str, uid: int, gid: int) -> tarfile.TarInfo:
    info = tarfile.TarInfo(name)
    info.type = tarfile.DIRTYPE
    info.mode = 0o755
    info.uid, info.gid = uid, gid
    info.mtime = int(time.time())
    return info


class DockerSandbox:
    backend = "docker"
    repo_root = REPO
    plugin_dir = PLUGIN_DIR

    def __init__(
        self,
        *,
        config: RunConfig,
        workspace: Workspace,
        temp_dir: Path,
        run_id: str,
        budget: Budget,
        events: EventLog,
        logs_dir: Path,
        redactor: Redactor,
    ) -> None:
        self.cfg = config.sandbox
        self.env_cfg = config.environment
        self.ws = workspace
        self.temp = temp_dir
        self.events = events
        self.docker = DockerCLI(self.cfg.docker)
        self.image = self.cfg.image.format(python_version=self.env_cfg.python_version)
        slug = run_id.lower()
        self.prefix = f"resolv-{slug}"
        self.env_image = f"resolv-env:{slug}"
        uid, _, gid = self.cfg.user.partition(":")
        self.uid, self.gid = int(uid), int(gid or uid)
        self.labels = ["--label", "resolv.managed=true", "--label", f"resolv.run={run_id}"]
        self.runner = CommandRunner(
            cwd=temp_dir, env=self.docker.env, budget=budget, events=events, logs_dir=logs_dir, redactor=redactor,
            max_output_chars=config.limits.max_command_output_chars,
        )
        self._containers: set[str] = set()
        self._networks: list[str] = []
        self._image_built = False
        self._counter = itertools.count(1)
        self._env_files = itertools.count(1)
        self._pass_env = {k: os.environ[k] for k in self.env_cfg.pass_env if k in os.environ}

    # -- helpers ---------------------------------------------------------

    def _limits(self) -> list[str]:
        c = self.cfg
        args = [
            "--cpus", f"{c.cpus:g}",
            "--memory", f"{c.memory_mb}m",
            "--memory-swap", f"{c.memory_mb}m",
            "--pids-limit", str(c.pids_limit),
            "--ulimit", f"fsize={c.file_size_limit_mb * 1024 * 1024}",
            "--cap-drop", "ALL",
            "--security-opt", "no-new-privileges",
            "--user", f"{self.uid}:{self.gid}",
            "--hostname", "resolv-sandbox",
        ]
        if c.storage_size:
            args += ["--storage-opt", f"size={c.storage_size}"]
        if c.runtime:
            args += ["--runtime", c.runtime]
        return args

    def _env_file(self, extra: dict[str, str] | None) -> str:
        env = {
            "PATH": f"{VENV}/bin:/usr/local/bin:/usr/local/sbin:/usr/sbin:/usr/bin:/sbin:/bin",
            "HOME": HOME,
            "USER": "resolv",
            "LOGNAME": "resolv",
            "LANG": "C.UTF-8",
            "PYTHONDONTWRITEBYTECODE": "1",
            "PIP_DISABLE_PIP_VERSION_CHECK": "1",
            "PIP_NO_INPUT": "1",
            "PIP_NO_CACHE_DIR": "1",
            **self._pass_env,
            **(extra or {}),
        }
        if Path(f"{self.temp}/venv-ready").exists():
            env["VIRTUAL_ENV"] = VENV
        path = self.temp / f"env-{next(self._env_files):03d}"
        lines = []
        for key, value in env.items():
            if "\n" in value or "\r" in value:
                raise SetupError("environment_error", f"environment value for {key} contains a newline")
            lines.append(f"{key}={value}")
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        path.chmod(0o600)
        return str(path)

    def _tar_workspace(self) -> bytes:
        """The full checkout plus the pytest plugin, owned by the sandbox user."""
        buf = io.BytesIO()
        owned = _owned(self.uid, self.gid)
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for d in ("work", "work/plugin", "work/results", "work/home"):
                tar.addfile(_dir_entry(d, self.uid, self.gid))
            tar.add(str(self.ws.root), arcname="work/repo", filter=owned)
            tar.add(str(PLUGIN_FILE), arcname=f"work/plugin/{PLUGIN_FILE.name}", filter=owned)
        return buf.getvalue()

    def _tar_written(self) -> bytes | None:
        """Files the agent has written so far (current content from the trusted checkout)."""
        paths = sorted(p for p in self.ws.written if (self.ws.root / p).is_file())
        if not paths:
            return None
        buf = io.BytesIO()
        owned = _owned(self.uid, self.gid)
        dirs: set[str] = set()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            for rel in paths:
                parts = rel.split("/")[:-1]
                for i in range(1, len(parts) + 1):
                    d = "work/repo/" + "/".join(parts[:i])
                    if d not in dirs:
                        dirs.add(d)
                        tar.addfile(_dir_entry(d, self.uid, self.gid))
                tar.add(str(self.ws.root / rel), arcname=f"work/repo/{rel}", filter=owned, recursive=False)
        return buf.getvalue()

    def _copy_in(self, container: str, data: bytes) -> None:
        self.docker("cp", "-a", "-", f"{container}:/", input=data)

    def _copy_out(self, container: str, path: str) -> bytes | None:
        proc = self.docker("cp", f"{container}:{path}", "-", check=False)
        if proc.returncode != 0:
            return None
        with tarfile.open(fileobj=io.BytesIO(proc.stdout)) as tar:
            for member in tar.getmembers():
                if member.isfile() and member.size <= MAX_RESULTS_BYTES:
                    extracted = tar.extractfile(member)
                    return extracted.read() if extracted else None
        return None

    def _remove(self, container: str) -> None:
        self.docker("rm", "-f", "-v", container, check=False)
        self._containers.discard(container)

    def _kill(self, container: str):
        return lambda: self.docker("kill", container, check=False, timeout=60)

    # -- setup -----------------------------------------------------------

    def _ensure_image(self) -> None:
        if self.docker("image", "inspect", self.image, check=False).returncode == 0:
            return
        pulled = self.runner.run(
            ["docker", "pull", self.image], label="setup-pull-image",
            exec_argv=[self.docker.command, "pull", self.image], exec_env=self.docker.env,
        )
        if not pulled.ok:
            raise SetupError(
                "environment_error", f"Pulling sandbox image {self.image} failed ({pulled.describe_status()}); see {pulled.log_path}."
            )

    def _start_proxy(self) -> str:
        network = f"{self.prefix}-setup-net"
        self.docker("network", "create", "--internal", *self.labels, network)
        self._networks.append(network)
        proxy = f"{self.prefix}-proxy"
        allow = [arg for host in self.cfg.allowed_hosts for arg in ("--allow", host)]
        self.docker(
            "create", "--name", proxy, "--network", "bridge", *self.labels,
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--user", f"{self.uid}:{self.gid}",
            "--memory", "256m", "--pids-limit", "64", "--log-opt", "max-size=5m",
            self.image, "python", f"{WORK}/proxy.py", "--port", str(PROXY_PORT), *allow,
        )
        self._containers.add(proxy)
        buf = io.BytesIO()
        with tarfile.open(fileobj=buf, mode="w") as tar:
            tar.addfile(_dir_entry("work", self.uid, self.gid))
            tar.add(str(PROXY_SCRIPT), arcname="work/proxy.py", filter=_owned(self.uid, self.gid))
        self._copy_in(proxy, buf.getvalue())
        self.docker("network", "connect", network, proxy)
        self.docker("start", proxy)
        deadline = time.monotonic() + 20
        while time.monotonic() < deadline:
            logs = self.docker("logs", proxy, check=False)
            if b"listening" in logs.stdout:
                return network
            time.sleep(0.2)
        raise SetupError("environment_error", "The setup network proxy did not start.")

    def _exec(self, container: str, argv: list[str], *, label: str, env: dict[str, str] | None = None) -> CommandResult:
        env_file = self._env_file(env)
        return self.runner.run(
            argv, label=label,
            exec_argv=[self.docker.command, "exec", "--env-file", env_file, "-w", REPO, container, *argv],
            exec_env=self.docker.env, kill=self._kill(container),
        )

    def setup(self) -> str:
        try:
            return self._setup()
        except DockerError as exc:
            raise SetupError("environment_error", f"Docker sandbox setup failed: {exc}") from None

    def _setup(self) -> str:
        if not self.docker.available():
            raise SetupError(
                "environment_error",
                "Docker is not available. Install and start Docker, or use the local sandbox for trusted repositories.",
            )
        self._ensure_image()
        network = "none"
        proxy_env: dict[str, str] = {}
        uses_proxy = self.cfg.setup_network == "proxy" and bool(self.env_cfg.install_commands)
        if uses_proxy:
            network = self._start_proxy()
            url = f"http://{self.prefix}-proxy:{PROXY_PORT}"
            proxy_env = {"HTTP_PROXY": url, "HTTPS_PROXY": url, "http_proxy": url, "https_proxy": url}

        setup = f"{self.prefix}-setup"
        self.docker(
            "create", "--name", setup, "--network", network, *self._limits(), *self.labels,
            "--log-driver", "none", "-w", REPO, self.image, "sleep", "infinity",
        )
        self._containers.add(setup)
        self._copy_in(setup, self._tar_workspace())
        self.docker("start", setup)

        created = self._exec(setup, ["python", "-m", "venv", VENV], label="setup-venv")
        if not created.ok:
            raise SetupError(
                "install_failed", f"Creating the virtual environment failed ({created.describe_status()}); see {created.log_path}."
            )
        Path(f"{self.temp}/venv-ready").touch()
        try:
            for i, command in enumerate(self.env_cfg.install_commands, 1):
                result = self._exec(setup, shlex.split(command), label=f"setup-install-{i}", env=proxy_env)
                if not result.ok:
                    raise SetupError(
                        "install_failed",
                        f"Install command {command!r} failed ({result.describe_status()}); see {result.log_path}. "
                        "This is an environment problem, not a code failure.",
                    )
        finally:
            if uses_proxy:
                logs = self.docker("logs", f"{self.prefix}-proxy", check=False)
                self.runner.record(
                    ["setup network proxy"], label="setup-network-log",
                    output=(logs.stdout + logs.stderr).decode("utf-8", errors="replace"),
                )

        check = self.docker(
            "exec", setup, "sh", "-c",
            f"command -v git >/dev/null 2>&1 || exit 0; git -c safe.directory='*' -C {REPO} status --porcelain --untracked-files=no",
            check=False,
        )
        modified = [line[3:] for line in check.stdout.decode(errors="replace").splitlines() if line.strip()]
        if modified:
            raise SetupError("environment_error", f"Environment setup modified tracked files: {', '.join(modified[:5])}.")

        self.docker("commit", "--pause=true", setup, self.env_image, timeout=900)
        self._image_built = True
        self._remove(setup)
        self._teardown_network()
        network_note = (
            f"allowlisted proxy to {', '.join(self.cfg.allowed_hosts)}" if uses_proxy else "none"
        )
        return f"Docker sandbox ready ({self.image}); setup network: {network_note}; tests run with no network"

    def _teardown_network(self) -> None:
        proxy = f"{self.prefix}-proxy"
        if proxy in self._containers:
            self._remove(proxy)
        for network in self._networks:
            self.docker("network", "rm", network, check=False)
        self._networks.clear()

    # -- test execution --------------------------------------------------

    def results_path(self, name: str) -> str:
        return f"{RESULTS_DIR}/{name}"

    def run(
        self, argv: list[str], *, label: str, env: dict[str, str] | None = None, collect: str | None = None
    ) -> tuple[CommandResult, bytes | None]:
        name = f"{self.prefix}-t{next(self._counter):03d}"
        try:
            env_file = self._env_file(env)
            self.docker(
                "create", "--name", name, "--network", "none", *self._limits(), *self.labels,
                "--log-driver", "none", "--env-file", env_file, "-w", REPO, self.env_image, *argv,
            )
            self._containers.add(name)
            written = self._tar_written()
            if written:
                self._copy_in(name, written)
        except DockerError as exc:
            self._remove(name)
            return self.runner.record_error(argv, label=label, error=f"sandbox error: {exc}"), None

        def inspect(result: CommandResult) -> None:
            state = self.docker("inspect", "-f", "{{.State.OOMKilled}}", name, check=False)
            if state.stdout.strip() == b"true":
                result.error = f"killed: memory limit of {self.cfg.memory_mb} MB exceeded"

        try:
            result = self.runner.run(
                argv, label=label, exec_argv=[self.docker.command, "start", "-a", name],
                exec_env=self.docker.env, kill=self._kill(name), on_exit=inspect,
            )
            data = self._copy_out(name, collect) if collect and not result.timed_out else None
            return result, data
        finally:
            self._remove(name)

    # -- reporting and cleanup -------------------------------------------

    def describe(self) -> dict:
        c = self.cfg
        return {
            "backend": "docker",
            "image": self.image,
            "runtime": c.runtime or "default (runc)",
            "user": f"{self.uid}:{self.gid}",
            "network": "none for all test runs; setup "
            + (f"limited to {', '.join(c.allowed_hosts)} via proxy" if c.setup_network == "proxy" else "had no network"),
            "limits": {
                "cpus": c.cpus,
                "memory_mb": c.memory_mb,
                "pids": c.pids_limit,
                "max_file_size_mb": c.file_size_limit_mb,
                "storage": c.storage_size or "not limited by resolv",
            },
            "credentials": "none passed into containers",
            "mounts": "none (files are copied in; no host mounts or Docker socket)",
        }

    def close(self) -> None:
        for container in list(self._containers):
            self._remove(container)
        self._teardown_network()
        if self._image_built:
            self.docker("rmi", "-f", self.env_image, check=False)
            self._image_built = False


def cleanup_stale_resources(docker: DockerCLI | None = None, keep_runs: set[str] | frozenset[str] = frozenset()) -> int:
    """Remove containers, networks, and images left by crashed runs; returns how many containers were removed."""
    docker = docker or DockerCLI()
    removed = 0
    listing = docker("ps", "-a", "--filter", "label=resolv.managed=true", "--format", '{{.ID}} {{.Label "resolv.run"}}', check=False)
    for line in listing.stdout.decode().splitlines():
        cid, _, run = line.partition(" ")
        if cid and run not in keep_runs:
            docker("rm", "-f", "-v", cid, check=False)
            removed += 1
    nets = docker("network", "ls", "--filter", "label=resolv.managed=true", "--format", '{{.ID}} {{.Label "resolv.run"}}', check=False)
    for line in nets.stdout.decode().splitlines():
        nid, _, run = line.partition(" ")
        if nid and run not in keep_runs:
            docker("network", "rm", nid, check=False)
    images = docker("images", "resolv-env", "--format", "{{.Repository}}:{{.Tag}}", check=False)
    keep_tags = {r.lower() for r in keep_runs}
    for ref in images.stdout.decode().splitlines():
        if ref and ref.split(":", 1)[-1] not in keep_tags:
            docker("rmi", "-f", ref, check=False)
    return removed
