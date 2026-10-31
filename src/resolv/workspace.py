"""Disposable git checkout of the target repository pinned to one commit.

All file access by the agent goes through this class, which confines paths to
the checkout and records every path the agent writes. Diffs and policy checks
cover exactly those paths, so files created as side effects of running tests
or installing dependencies never leak into the proposed patch.
"""

from __future__ import annotations

import fnmatch
import os
import re
import subprocess
from dataclasses import dataclass
from pathlib import Path, PurePosixPath

from resolv.errors import ToolError, WorkspaceError

MAX_FILE_CHARS = 400_000
MAX_LISTED_FILES = 400
MAX_SEARCH_RESULTS = 100
_SEARCH_FILE_LIMIT = 2_000_000


def _git_env() -> dict[str, str]:
    env = {k: v for k, v in os.environ.items() if not k.startswith("GIT_")}
    env.update(GIT_TERMINAL_PROMPT="0", LC_ALL="C")
    return env


def git(cwd: Path, *args: str, check: bool = True) -> str:
    proc = subprocess.run(
        ["git", "--literal-pathspecs", "-c", "core.hooksPath=/dev/null", "-c", "core.quotepath=off", *args],
        cwd=cwd,
        capture_output=True,
        text=True,
        env=_git_env(),
    )
    if check and proc.returncode != 0:
        raise WorkspaceError(f"git {args[0]} failed: {proc.stderr.strip() or proc.stdout.strip()}")
    return proc.stdout


def resolve_commit(repo: Path, ref: str) -> str:
    return git(repo, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}").strip()


def is_git_repo(path: Path) -> bool:
    if not path.is_dir():
        return False
    proc = subprocess.run(
        ["git", "rev-parse", "--is-inside-work-tree"], cwd=path, capture_output=True, text=True, env=_git_env()
    )
    return proc.returncode == 0 and proc.stdout.strip() == "true"


def has_uncommitted_changes(repo: Path) -> bool:
    return bool(git(repo, "status", "--porcelain", "--untracked-files=no").strip())


@dataclass(frozen=True)
class FileStat:
    path: str
    added: int
    deleted: int
    binary: bool = False

    @property
    def changed_lines(self) -> int:
        return self.added + self.deleted


def _is_binary(data: bytes) -> bool:
    return b"\0" in data[:8192]


class Workspace:
    def __init__(self, root: Path, base_commit: str) -> None:
        self.root = root.resolve()
        self.base_commit = base_commit
        self.written: set[str] = set()

    @classmethod
    def clone(cls, source: Path, commit: str, dest: Path) -> Workspace:
        git(dest.parent, "clone", "--quiet", "--no-checkout", str(source), str(dest))
        git(dest, "checkout", "--quiet", "--detach", commit)
        return cls(dest, commit)

    # -- path handling -------------------------------------------------

    def rel(self, path: object) -> str:
        """Normalize an agent-supplied path; reject anything outside the checkout."""
        if not isinstance(path, str) or not path.strip():
            raise ToolError("path must be a non-empty string relative to the repository root")
        pure = PurePosixPath(path.strip().replace("\\", "/"))
        if pure.is_absolute():
            raise ToolError(f"absolute paths are not allowed: {path}")
        parts = [p for p in pure.parts if p not in ("", ".")]
        if ".." in parts:
            raise ToolError(f"path must not contain '..': {path}")
        if ".git" in parts:
            raise ToolError("the .git directory is not accessible")
        rel = "/".join(parts)
        if not rel:
            return ""
        full = self.root / rel
        if not full.resolve().is_relative_to(self.root):
            raise ToolError(f"path resolves outside the repository: {path}")
        return rel

    def _file(self, rel: str) -> Path:
        full = self.root / rel
        if not full.is_file():
            raise ToolError(f"file not found: {rel}")
        return full

    # -- read operations -----------------------------------------------

    def all_files(self) -> list[str]:
        out = git(self.root, "ls-files", "-z", "--cached", "--others", "--exclude-standard")
        return sorted({f for f in out.split("\0") if f and (self.root / f).is_file()})

    def list_files(self, path: str = "") -> str:
        rel = self.rel(path) if path else ""
        files = [f for f in self.all_files() if not rel or f == rel or f.startswith(rel + "/")]
        if not files:
            return f"No files under '{rel or '.'}'."
        shown = files[:MAX_LISTED_FILES]
        text = "\n".join(shown)
        if len(files) > len(shown):
            text += f"\n... {len(files) - len(shown)} more files; list a subdirectory to see them."
        return text

    def read_text(self, rel: str) -> str:
        data = self._file(rel).read_bytes()
        if _is_binary(data):
            raise ToolError(f"{rel} is a binary file")
        return data.decode("utf-8", errors="replace")

    def read_file(self, path: str, start_line: int | None = None, end_line: int | None = None) -> str:
        rel = self.rel(path)
        lines = self.read_text(rel).splitlines()
        total = len(lines)
        start = max(1, start_line or 1)
        end = min(total, end_line or total)
        if total == 0:
            return f"{rel} is empty."
        if start > total:
            raise ToolError(f"{rel} has only {total} lines")
        body = "\n".join(f"{i:>5}| {lines[i - 1]}" for i in range(start, end + 1))
        return f"{rel} (lines {start}-{end} of {total})\n{body}"

    def search(self, pattern: str, glob: str | None = None) -> str:
        try:
            regex = re.compile(pattern)
        except re.error as exc:
            raise ToolError(f"invalid regular expression: {exc}") from exc
        matches: list[str] = []
        for rel in self.all_files():
            if glob and not (fnmatch.fnmatch(rel, glob) or fnmatch.fnmatch(PurePosixPath(rel).name, glob)):
                continue
            full = self.root / rel
            if full.stat().st_size > _SEARCH_FILE_LIMIT:
                continue
            data = full.read_bytes()
            if _is_binary(data):
                continue
            for lineno, line in enumerate(data.decode("utf-8", errors="replace").splitlines(), 1):
                if regex.search(line):
                    matches.append(f"{rel}:{lineno}: {line.strip()[:200]}")
                    if len(matches) >= MAX_SEARCH_RESULTS:
                        return "\n".join(matches) + "\n... more matches omitted; refine the pattern or glob."
        return "\n".join(matches) if matches else "No matches."

    # -- write operations ----------------------------------------------

    def write_text(self, rel: str, content: str) -> None:
        if len(content) > MAX_FILE_CHARS:
            raise ToolError(f"content too large ({len(content)} characters)")
        full = self.root / rel
        if full.is_symlink():
            raise ToolError(f"refusing to write through a symlink: {rel}")
        if full.is_dir():
            raise ToolError(f"{rel} is a directory")
        full.parent.mkdir(parents=True, exist_ok=True)
        full.write_text(content, encoding="utf-8")
        self.written.add(rel)

    def replaced_text(self, rel: str, old: str, new: str) -> str:
        text = self.read_text(rel)
        if not old:
            raise ToolError("old_text must not be empty")
        count = text.count(old)
        if count == 0:
            raise ToolError(f"old_text was not found in {rel}; read the file and copy the text exactly")
        if count > 1:
            raise ToolError(f"old_text occurs {count} times in {rel}; include more context so it is unique")
        return text.replace(old, new, 1)

    # -- git state -------------------------------------------------------

    def _stage(self) -> list[str]:
        paths = sorted(self.written)
        if paths:
            git(self.root, "add", "-A", "--", *paths)
        return paths

    def diff(self, against: str | None = None) -> str:
        paths = self._stage()
        if not paths:
            return ""
        return git(
            self.root, "diff", "--cached", "--no-color", "--no-ext-diff", "--no-renames", "--binary",
            against or self.base_commit, "--", *paths,
        )

    def numstat(self, against: str | None = None) -> list[FileStat]:
        paths = self._stage()
        if not paths:
            return []
        out = git(self.root, "diff", "--cached", "--numstat", "--no-renames", against or self.base_commit, "--", *paths)
        stats = []
        for line in out.splitlines():
            added, deleted, path = line.split("\t", 2)
            if added == "-":
                stats.append(FileStat(path, 0, 0, binary=True))
            else:
                stats.append(FileStat(path, int(added), int(deleted)))
        return stats

    def added_lines(self, rel: str, against: str | None = None) -> list[str]:
        self._stage()
        out = git(self.root, "diff", "--cached", "-U0", "--no-color", against or self.base_commit, "--", rel)
        return [line[1:] for line in out.splitlines() if line.startswith("+") and not line.startswith("+++")]

    def content_at(self, rev: str, rel: str) -> str | None:
        proc = subprocess.run(
            ["git", "show", f"{rev}:{rel}"], cwd=self.root, capture_output=True, env=_git_env()
        )
        if proc.returncode != 0:
            return None
        return proc.stdout.decode("utf-8", errors="replace")

    def exists_at_base(self, rel: str) -> bool:
        return bool(git(self.root, "ls-tree", "--name-only", self.base_commit, "--", rel).strip())

    def commit_snapshot(self, message: str) -> str:
        self._stage()
        git(
            self.root, "-c", "user.name=resolv", "-c", "user.email=resolv@localhost", "-c", "commit.gpgsign=false",
            "commit", "--quiet", "--no-verify", "--allow-empty", "-m", message,
        )
        return git(self.root, "rev-parse", "HEAD").strip()

    def tracked_modifications(self) -> list[str]:
        return [p for p in git(self.root, "diff", "--name-only").splitlines() if p]

    def ignore_untracked(self) -> list[str]:
        """Add files created by environment setup to .git/info/exclude."""
        out = git(self.root, "ls-files", "-z", "--others", "--exclude-standard")
        untracked = sorted(f for f in out.split("\0") if f)
        if untracked:
            exclude = self.root / ".git" / "info" / "exclude"
            exclude.parent.mkdir(parents=True, exist_ok=True)
            escaped = ["/" + re.sub(r"([\[\]*?!#\\])", r"\\\1", f) for f in untracked]
            with exclude.open("a", encoding="utf-8") as fh:
                fh.write("\n# created during resolv environment setup\n" + "\n".join(escaped) + "\n")
        return untracked
