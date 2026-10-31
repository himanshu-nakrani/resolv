"""Tools exposed to the model, built per stage.

There is deliberately no general shell tool: the model can read and search
the checkout, write files allowed by the current stage's policy, and run the
configured pytest command on chosen tests.
"""

from __future__ import annotations

from resolv import policy
from resolv.agent import Tool
from resolv.errors import ToolError
from resolv.model import ToolSpec
from resolv.testrun import PytestRunner, describe_run
from resolv.workspace import Workspace

MAX_TEST_SELECTORS = 20


def _str(args: dict, key: str, *, required: bool = True, allow_empty: bool = False) -> str | None:
    value = args.get(key)
    if value is None:
        if required:
            raise ToolError(f"missing required argument {key!r}")
        return None
    if not isinstance(value, str):
        raise ToolError(f"argument {key!r} must be a string")
    if not allow_empty and not value.strip():
        raise ToolError(f"argument {key!r} must not be empty")
    return value


def _int(args: dict, key: str) -> int | None:
    value = args.get(key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 1:
        raise ToolError(f"argument {key!r} must be a positive integer")
    return value


def _str_list(args: dict, key: str, *, max_items: int) -> list[str]:
    value = args.get(key)
    if not isinstance(value, list) or not value or not all(isinstance(v, str) and v.strip() for v in value):
        raise ToolError(f"argument {key!r} must be a non-empty list of strings")
    if len(value) > max_items:
        raise ToolError(f"argument {key!r} accepts at most {max_items} items")
    return [v.strip() for v in value]


def _schema(properties: dict, required: list[str]) -> dict:
    return {"type": "object", "properties": properties, "required": required, "additionalProperties": False}


_STRING = {"type": "string"}
_STRINGS = {"type": "array", "items": {"type": "string"}}


def read_tools(ws: Workspace) -> list[Tool]:
    return [
        Tool(
            ToolSpec(
                "list_files",
                "List repository files (tracked and new), optionally under a directory.",
                _schema({"path": {"type": "string", "description": "Directory relative to the repo root."}}, []),
            ),
            lambda a: ws.list_files(_str(a, "path", required=False, allow_empty=True) or ""),
        ),
        Tool(
            ToolSpec(
                "read_file",
                "Read a text file with line numbers. Use start_line/end_line for large files.",
                _schema(
                    {"path": _STRING, "start_line": {"type": "integer"}, "end_line": {"type": "integer"}}, ["path"]
                ),
            ),
            lambda a: ws.read_file(_str(a, "path"), _int(a, "start_line"), _int(a, "end_line")),
        ),
        Tool(
            ToolSpec(
                "search_code",
                "Search file contents with a Python regular expression. Optional glob filters paths (e.g. '*.py').",
                _schema({"pattern": _STRING, "glob": _STRING}, ["pattern"]),
            ),
            lambda a: ws.search(_str(a, "pattern"), _str(a, "glob", required=False)),
        ),
    ]


def write_tools(ws: Workspace, stage: str) -> list[Tool]:
    def guarded(path: object) -> str:
        rel = ws.rel(path)
        if not rel:
            raise ToolError("path must name a file")
        policy.check_write_allowed(rel, stage)
        return rel

    def check_existing_tests(rel: str, content: str) -> None:
        if stage == policy.REPRODUCE and ws.exists_at_base(rel):
            original = ws.content_at(ws.base_commit, rel) or ""
            if not policy.is_append_only(original, content):
                raise ToolError(
                    f"{rel} is an existing test file: keep its original content unchanged and only append new "
                    "tests at the end (or create a new test file)."
                )

    def write_file(args: dict) -> str:
        rel = guarded(args.get("path"))
        content = _str(args, "content", allow_empty=True)
        check_existing_tests(rel, content)
        ws.write_text(rel, content)
        return f"Wrote {rel} ({len(content.splitlines())} lines)."

    def replace_in_file(args: dict) -> str:
        rel = guarded(args.get("path"))
        content = ws.replaced_text(rel, _str(args, "old_text"), _str(args, "new_text", allow_empty=True))
        check_existing_tests(rel, content)
        ws.write_text(rel, content)
        return f"Updated {rel}."

    return [
        Tool(
            ToolSpec(
                "write_file",
                "Create a file or replace its entire content.",
                _schema({"path": _STRING, "content": _STRING}, ["path", "content"]),
            ),
            write_file,
        ),
        Tool(
            ToolSpec(
                "replace_in_file",
                "Replace one exact, unique occurrence of old_text with new_text in an existing file.",
                _schema({"path": _STRING, "old_text": _STRING, "new_text": _STRING}, ["path", "old_text", "new_text"]),
            ),
            replace_in_file,
        ),
    ]


def validate_selectors(ws: Workspace, selectors: list[str]) -> list[str]:
    """Allow only node ids / paths inside the checkout, never pytest options."""
    for selector in selectors:
        if selector.startswith("-") or "\0" in selector:
            raise ToolError(f"invalid test selector {selector!r}: pass pytest node ids or test file paths only")
        path = selector.split("::", 1)[0]
        if not ws.rel(path):
            raise ToolError(f"invalid test selector {selector!r}")
    return selectors


def run_tests_tool(ws: Workspace, runner: PytestRunner, label: str) -> Tool:
    def handler(args: dict) -> str:
        selectors = validate_selectors(ws, _str_list(args, "node_ids", max_items=MAX_TEST_SELECTORS))
        return describe_run(runner.run(label, selectors))

    return Tool(
        ToolSpec(
            "run_tests",
            "Run the configured pytest command on specific tests, e.g. ['tests/test_x.py::test_y'] or "
            "['tests/test_x.py'].",
            _schema({"node_ids": _STRINGS}, ["node_ids"]),
        ),
        handler,
    )


def final_tool(name: str, description: str, properties: dict, required: list[str]) -> Tool:
    def handler(args: dict) -> str:
        for key in required:
            value = args.get(key)
            if properties[key].get("type") == "array":
                _str_list(args, key, max_items=MAX_TEST_SELECTORS)
            elif properties[key].get("enum"):
                if value not in properties[key]["enum"]:
                    raise ToolError(f"argument {key!r} must be one of {properties[key]['enum']}")
            else:
                _str(args, key)
        return "submitted"

    return Tool(ToolSpec(name, description, _schema(properties, required)), handler, final=True)


BLOCKER_CATEGORIES = ["ambiguous_issue", "not_a_bug", "insufficient_information", "out_of_scope", "cannot_reproduce", "other"]


def report_blocker_tool() -> Tool:
    return final_tool(
        "report_blocker",
        "Stop the run without a patch and explain concretely what prevents a reliable fix.",
        {"category": {"type": "string", "enum": BLOCKER_CATEGORIES}, "reason": _STRING},
        ["category", "reason"],
    )


def diagnosis_tool() -> Tool:
    return final_tool(
        "submit_diagnosis",
        "Finish investigation with a diagnosis grounded in the code.",
        {
            "summary": {"type": "string", "description": "One or two sentences restating the bug."},
            "cause": {"type": "string", "description": "Likely root cause with file and function references."},
            "relevant_files": _STRINGS,
            "reproduction_plan": {"type": "string", "description": "How a pytest test can demonstrate the bug."},
        },
        ["summary", "cause", "relevant_files", "reproduction_plan"],
    )


def reproduction_tool() -> Tool:
    return final_tool(
        "submit_reproduction",
        "Submit the new regression tests. The orchestrator runs them on unchanged production code; each must fail "
        "because of the reported bug.",
        {
            "test_node_ids": _STRINGS,
            "expected_failure": {"type": "string", "description": "How and why the tests fail on the current code."},
        },
        ["test_node_ids", "expected_failure"],
    )


def patch_tool() -> Tool:
    return final_tool(
        "submit_patch",
        "Submit the production-code fix for validation (regression tests plus the configured suite).",
        {
            "changes": {"type": "string", "description": "What changed and why."},
            "assumptions_and_risks": {"type": "string", "description": "Assumptions and risks for reviewers."},
        },
        ["changes", "assumptions_and_risks"],
    )

