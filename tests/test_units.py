from __future__ import annotations

import io
import json
import os
import urllib.error

import pytest
from conftest import make_repo

from resolv.config import load_config
from resolv.errors import ConfigError, ModelError, ToolError
from resolv.model import ToolSpec
from resolv.model.anthropic import AnthropicClient
from resolv.policy import is_append_only, is_test_path, protected_reason, skip_markers
from resolv.redact import Redactor
from resolv.testrun import compare_runs, reproduction_problem
from resolv.workspace import Workspace

# -- policy ------------------------------------------------------------------


@pytest.mark.parametrize(
    "path,expected",
    [
        ("tests/test_x.py", True),
        ("pkg/tests/helpers.py", True),
        ("test_top.py", True),
        ("pkg/thing_test.py", True),
        ("conftest.py", True),
        ("pkg/module.py", False),
        ("pkg/testing_utils.py", False),
    ],
)
def test_is_test_path(path, expected):
    assert is_test_path(path) is expected


@pytest.mark.parametrize(
    "path", ["pyproject.toml", "pkg/setup.cfg", "requirements-dev.txt", "uv.lock", ".github/workflows/ci.yml", "tests/conftest.py", ".gitignore", "Dockerfile.dev"]
)
def test_protected_paths(path):
    assert protected_reason(path)


@pytest.mark.parametrize("path", ["pkg/module.py", "docs/index.md", "tests/test_x.py"])
def test_unprotected_paths(path):
    assert protected_reason(path) is None


def test_append_only():
    old = "def test_a():\n    assert f() == 1"
    assert is_append_only(old, old + "\n\n\ndef test_b():\n    assert True\n")
    assert not is_append_only(old, old + "0\n")  # changes an existing line
    assert not is_append_only(old, "def test_a():\n    pass\n")
    assert is_append_only("", "anything")


def test_skip_markers():
    lines = [
        "@pytest.mark.skip(reason='later')",
        "    pytest.xfail('nope')",
        "@unittest.skip('x')",
        "        self.skipTest('x')",
        "collect_ignore = ['a.py']",
        "def skip_whitespace(text):",
        "x = mark_skip_count",
    ]
    assert len(skip_markers(lines)) == 5


# -- workspace path confinement --------------------------------------------


def test_workspace_rejects_paths_outside_checkout(tmp_path):
    repo = make_repo(tmp_path / "r", {"a.py": "x = 1\n"})
    (repo / "link").symlink_to(tmp_path)
    ws = Workspace(repo, "HEAD")
    for bad in ["/etc/passwd", "../x", "a/../../x", ".git/config", "link/secret", ""]:
        with pytest.raises(ToolError):
            ws.rel(bad)
    assert ws.rel("./pkg//mod.py") == "pkg/mod.py"


def test_workspace_replace_requires_unique_match(tmp_path):
    repo = make_repo(tmp_path / "r", {"a.py": "x = 1\nx = 1\ny = 2\n"})
    ws = Workspace(repo, "HEAD")
    with pytest.raises(ToolError, match="2 times"):
        ws.replaced_text("a.py", "x = 1", "x = 3")
    with pytest.raises(ToolError, match="not found"):
        ws.replaced_text("a.py", "z", "x")
    assert ws.replaced_text("a.py", "y = 2", "y = 3").endswith("y = 3\n")


# -- config ------------------------------------------------------------------


def test_config_loading(tmp_path):
    path = tmp_path / "c.toml"
    path.write_text('[environment]\ninstall_commands = "pip install -e ."\nvenv = false\n[limits]\nmax_cost_usd = 1\n')
    config = load_config(path)
    assert config.environment.install_commands == ("pip install -e .",)
    assert config.environment.venv is False
    assert config.limits.max_cost_usd == 1.0
    assert config.limits.max_changed_files == 5

    path.write_text("[limits]\nmax_cost = 1\n")
    with pytest.raises(ConfigError, match="unknown key"):
        load_config(path)
    path.write_text("[limits]\nmax_changed_files = 0\n")
    with pytest.raises(ConfigError, match="positive"):
        load_config(path)
    path.write_text('[limits]\nmax_changed_files = "5"\n')
    with pytest.raises(ConfigError, match="integer"):
        load_config(path)


# -- redaction ---------------------------------------------------------------


def test_redaction():
    redact = Redactor(["my-literal-secret-value"])
    text = (
        "key sk-ant-api03-abcdefghijklmnopqrstuvwxyz token ghp_" + "b" * 36
        + " aws AKIAABCDEFGHIJKLMNOP password = 'hunter2hunter2' my-literal-secret-value ok"
    )
    out = redact(text)
    assert "sk-ant" not in out and "ghp_" not in out and "AKIA" not in out
    assert "hunter2" not in out and "my-literal" not in out
    assert out.endswith("ok")
    assert redact.deep({"a": ["sk-ant-api03-abcdefghijklmnopqrstuv"]}) == {"a": ["[REDACTED]"]}


# -- pytest result collection and classification ----------------------------


CLASSIFY_FILES = {
    "pytest.ini": "[pytest]\n",
    "lib.py": "def broken():\n    return undefined_name\n\ndef value():\n    return 1\n",
    "tests/test_cases.py": (
        "import pytest\nfrom lib import broken, value\n\n\n"
        "@pytest.fixture\ndef bad_fixture():\n    raise RuntimeError('fixture boom')\n\n\n"
        "def test_assertion():\n    assert value() == 2\n\n\n"
        "def test_fixture_error(bad_fixture):\n    pass\n\n\n"
        "def test_name_error_in_test():\n    assert valeu() == 1\n\n\n"
        "def test_name_error_in_library():\n    broken()\n\n\n"
        "def test_ok():\n    assert value() == 1\n\n\n"
        "@pytest.mark.skip(reason='not today')\ndef test_skipped():\n    pass\n\n\n"
        "@pytest.mark.xfail\ndef test_xfail():\n    assert False\n"
    ),
    "tests/test_import.py": "import missing_dependency_xyz\n",
}


def test_plugin_records_outcomes_and_classifies_failures(tmp_path, make_pytest_runner):
    repo = make_repo(tmp_path / "r", CLASSIFY_FILES)
    runner = make_pytest_runner(repo)

    run = runner.run("cases", ["tests/test_cases.py"])
    assert run.environment_error is None
    t = run.tests
    prefix = "tests/test_cases.py::"
    assert t[prefix + "test_assertion"].outcome == "failed"
    assert t[prefix + "test_assertion"].exc_type == "AssertionError"
    assert t[prefix + "test_fixture_error"].outcome == "error"
    assert t[prefix + "test_fixture_error"].when == "setup"
    assert t[prefix + "test_ok"].outcome == "passed"
    assert t[prefix + "test_skipped"].outcome == "skipped"
    assert t[prefix + "test_xfail"].outcome == "xfailed"

    test_paths = {"tests/test_cases.py"}
    assert reproduction_problem(t[prefix + "test_assertion"], test_paths) is None
    assert "did not fail in the test body" in reproduction_problem(t[prefix + "test_fixture_error"], test_paths)
    assert "inside the test itself" in reproduction_problem(t[prefix + "test_name_error_in_test"], test_paths)
    # A NameError raised by the code under test is a legitimate bug signal.
    assert t[prefix + "test_name_error_in_library"].crash_path == "lib.py"
    assert reproduction_problem(t[prefix + "test_name_error_in_library"], test_paths) is None

    collect = runner.run("import", ["tests/test_import.py"])
    assert collect.environment_error is None
    assert collect.collection_errors[0].exc_type == "ModuleNotFoundError"

    missing = runner.run("missing", ["tests/test_nope.py::test_x"])
    assert missing.environment_error is not None


def test_compare_runs(tmp_path, make_pytest_runner):
    repo = make_repo(
        tmp_path / "r",
        {"pytest.ini": "[pytest]\n", "tests/test_a.py": "def test_pass():\n    pass\n\ndef test_fail():\n    assert 0\n\ndef test_gone():\n    pass\n"},
    )
    runner = make_pytest_runner(repo)
    baseline = runner.run("baseline")
    (repo / "tests/test_a.py").write_text("def test_pass():\n    assert 0\n\ndef test_fail():\n    assert 0\n")
    after = runner.run("after")
    comparison = compare_runs(baseline, after)
    assert [t.nodeid for t in comparison.new_failures] == ["tests/test_a.py::test_pass"]
    assert [t.nodeid for t in comparison.pre_existing_failures] == ["tests/test_a.py::test_fail"]
    assert comparison.missing == ["tests/test_a.py::test_gone"]
    assert not comparison.clean


# -- Anthropic client ----------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {"retry-after": "0"}, io.BytesIO(b'{"error": "busy"}'))


def test_anthropic_client_request_and_parsing():
    sent = []
    payload = {
        "content": [
            {"type": "text", "text": "Looking."},
            {"type": "tool_use", "id": "toolu_1", "name": "read_file", "input": {"path": "a.py"}},
        ],
        "stop_reason": "tool_use",
        "usage": {"input_tokens": 10, "output_tokens": 5, "cache_read_input_tokens": 100, "cache_creation_input_tokens": 7},
    }
    responses = [_http_error(529), _Response(json.dumps(payload).encode())]

    def transport(request, timeout):
        sent.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return item

    client = AnthropicClient(model="claude-test", api_key="k" * 20, transport=transport, sleep=lambda s: None)
    tool = ToolSpec("read_file", "Read", {"type": "object", "properties": {}})
    result = client.complete(system="sys", messages=[{"role": "user", "content": "hi"}], tools=[tool], max_tokens=100)

    assert len(sent) == 2  # retried once after 529
    request = sent[-1]
    assert request.full_url == "https://api.anthropic.com/v1/messages"
    assert request.get_header("X-api-key") == "k" * 20
    assert request.get_header("Anthropic-version") == "2023-06-01"
    body = json.loads(request.data)
    assert body["model"] == "claude-test"
    assert body["tools"][0]["name"] == "read_file"
    assert body["messages"][0]["content"][0]["cache_control"] == {"type": "ephemeral"}
    assert result.text == "Looking."
    assert result.tool_calls[0].arguments == {"path": "a.py"}
    assert (result.usage.cache_read_tokens, result.usage.cache_write_tokens) == (100, 7)


def test_anthropic_client_non_retryable_error():
    def transport(request, timeout):
        raise _http_error(400)

    client = AnthropicClient(model="m", api_key="k" * 20, transport=transport, sleep=lambda s: None)
    with pytest.raises(ModelError, match="HTTP 400"):
        client.complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10)


def test_cli_check_end_to_end(tmp_path, capsys):
    from conftest import TOY_FILES

    from resolv.cli import main

    repo = make_repo(tmp_path / "toy", TOY_FILES)
    code = main(["check", "--repo", str(repo), "--out", str(tmp_path / "out"), "--no-venv", "--python", os.environ.get("PYTHON", __import__("sys").executable)])
    assert code == 0
    assert "checked" in capsys.readouterr().out
