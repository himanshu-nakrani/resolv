"""Best-effort redaction of likely secrets before logs and reports are stored."""

from __future__ import annotations

import re
from collections.abc import Iterable

REDACTED = "[REDACTED]"

_PATTERNS = [
    re.compile(r"-----BEGIN [A-Z ]*PRIVATE KEY-----.*?-----END [A-Z ]*PRIVATE KEY-----", re.S),
    re.compile(r"\b(?:ghp|gho|ghu|ghs|ghr)_[A-Za-z0-9]{20,}"),
    re.compile(r"\bgithub_pat_[A-Za-z0-9_]{20,}"),
    re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{16,}"),
    re.compile(r"\bsk-[A-Za-z0-9_\-]{20,}"),
    re.compile(r"\b(?:AKIA|ASIA)[A-Z0-9]{16}\b"),
    re.compile(r"\bxox[abprs]-[A-Za-z0-9-]{10,}"),
    re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b"),
]
# key = "value" style assignments; only quoted values to limit false positives.
_KEYED = re.compile(
    r"(?i)\b((?:password|passwd|secret|api[_-]?key|access[_-]?token|auth[_-]?token)\w*\s*[:=]\s*)(['\"])([^'\"\s]{8,})\2"
)


class Redactor:
    def __init__(self, literals: Iterable[str | None] = ()) -> None:
        self._literals = sorted({s for s in literals if s and len(s) >= 8}, key=len, reverse=True)

    def __call__(self, text: str) -> str:
        if not text:
            return text
        for literal in self._literals:
            text = text.replace(literal, REDACTED)
        for pattern in _PATTERNS:
            text = pattern.sub(REDACTED, text)
        return _KEYED.sub(lambda m: f"{m.group(1)}{m.group(2)}{REDACTED}{m.group(2)}", text)

    def deep(self, value):
        if isinstance(value, str):
            return self(value)
        if isinstance(value, dict):
            return {k: self.deep(v) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.deep(v) for v in value]
        return value
