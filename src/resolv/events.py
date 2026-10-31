"""Append-only JSONL event log for a run."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path

from resolv.redact import Redactor


def utc_now() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


class EventLog:
    def __init__(self, path: Path, redactor: Redactor) -> None:
        self.path = path
        self._redact = redactor
        self._fh = path.open("a", encoding="utf-8")
        self.stage = "init"

    def emit(self, kind: str, **data) -> None:
        record = {"ts": utc_now(), "stage": self.stage, "kind": kind, **self._redact.deep(data)}
        self._fh.write(json.dumps(record, default=str) + "\n")
        self._fh.flush()

    def close(self) -> None:
        self._fh.close()
