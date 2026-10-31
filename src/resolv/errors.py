"""Exceptions shared across resolv components."""

from __future__ import annotations


class ConfigError(ValueError):
    """Invalid or incomplete run configuration."""


class ToolError(Exception):
    """A tool call was rejected or failed; the message is returned to the model."""


class BudgetExceeded(Exception):
    """A run budget enforced by the orchestrator was exhausted."""

    def __init__(self, kind: str, message: str) -> None:
        super().__init__(message)
        self.kind = kind


class RunCancelled(Exception):
    """The run was cancelled by the operator."""


class WorkspaceError(Exception):
    """A git or filesystem operation on the run workspace failed."""


class ModelError(Exception):
    """The model provider returned an error or an unusable response."""


class AgentTurnLimit(Exception):
    """The model did not finish a stage within its turn limit."""
