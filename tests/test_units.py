from __future__ import annotations

import io
import json
import os
import urllib.error

import pytest
from conftest import make_repo

from resolv.budget import Budget, Usage
from resolv.config import Limits, ModelConfig, RunConfig, load_config
from resolv.errors import ConfigError, ModelError, ToolError
from resolv.model import ToolSpec, create_client
from resolv.model.openrouter import OpenRouterClient, to_openai_messages
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
    path.write_text('[model]\ndata_collection = "sometimes"\n')
    with pytest.raises(ConfigError, match="data_collection"):
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


# -- OpenRouter client ---------------------------------------------------------


class _Response(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _http_error(code: int) -> urllib.error.HTTPError:
    return urllib.error.HTTPError("https://x", code, "err", {"retry-after": "0"}, io.BytesIO(b'{"error": "busy"}'))


def _client(responses: list, sent: list, model: str = "anthropic/claude-sonnet-5.5", **kwargs) -> OpenRouterClient:
    def transport(request, timeout):
        sent.append(request)
        item = responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return _Response(json.dumps(item).encode())

    return OpenRouterClient(model=model, api_key="sk-or-v1-" + "k" * 20, transport=transport, sleep=lambda s: None, **kwargs)


def test_openrouter_request_translation_and_parsing():
    payload = {
        "provider": "Anthropic",
        "choices": [
            {
                "finish_reason": "tool_calls",
                "message": {
                    "role": "assistant",
                    "content": "Looking.",
                    "tool_calls": [
                        {"id": "call_1", "type": "function", "function": {"name": "read_file", "arguments": '{"path": "a.py"}'}},
                        {"id": "call_2", "type": "function", "function": {"name": "read_file", "arguments": "{not json"}},
                    ],
                },
            }
        ],
        "usage": {"prompt_tokens": 120, "completion_tokens": 5, "prompt_tokens_details": {"cached_tokens": 100, "cache_write_tokens": 7}, "cost": 0.0123},
    }
    sent: list = []
    client = _client([_http_error(529), payload], sent)
    messages = [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": [{"type": "text", "text": "ok"}, {"type": "tool_use", "id": "t1", "name": "list_files", "input": {}}]},
        {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t1", "content": "a.py", "is_error": True}]},
    ]
    tool = ToolSpec("read_file", "Read", {"type": "object", "properties": {}})
    result = client.complete(system="sys", messages=messages, tools=[tool], max_tokens=100)

    assert len(sent) == 2  # retried once after 529
    request = sent[-1]
    assert request.full_url == "https://openrouter.ai/api/v1/chat/completions"
    assert request.get_header("Authorization") == "Bearer sk-or-v1-" + "k" * 20
    body = json.loads(request.data)
    assert body["model"] == "anthropic/claude-sonnet-5.5"
    assert body["provider"] == {"data_collection": "deny", "require_parameters": True}
    assert body["cache_control"] == {"type": "ephemeral"}
    assert body["tools"][0] == {"type": "function", "function": {"name": "read_file", "description": "Read", "parameters": {"type": "object", "properties": {}}}}
    assert body["messages"] == [
        {"role": "system", "content": "sys"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "ok", "tool_calls": [{"id": "t1", "type": "function", "function": {"name": "list_files", "arguments": "{}"}}]},
        {"role": "tool", "tool_call_id": "t1", "content": "[tool error] a.py"},
    ]

    assert result.text == "Looking."
    assert result.stop_reason == "tool_calls"
    assert result.tool_calls[0].arguments == {"path": "a.py"}
    assert result.tool_calls[1].arguments == "{not json"  # surfaced to the agent as a tool error
    u = result.usage
    assert (u.input_tokens, u.cache_read_tokens, u.cache_write_tokens, u.output_tokens) == (13, 100, 7, 5)
    assert u.cost_usd == 0.0123
    assert client.upstream_providers == {"Anthropic"}

    # Echoing the response back produces a valid assistant message.
    replay = to_openai_messages("s", [{"role": "assistant", "content": result.content}])[1]
    assert replay["tool_calls"][1]["function"]["arguments"] == "{not json"


def test_openrouter_routing_options_and_no_cache_control_for_other_models():
    sent: list = []
    ok = {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": "done"}}], "usage": {}}
    client = _client([ok], sent, model="openai/gpt-6.1-sol", data_collection="allow", zero_data_retention=True)
    result = client.complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10)
    body = json.loads(sent[0].data)
    assert "cache_control" not in body
    assert body["provider"] == {"data_collection": "allow", "require_parameters": True, "zdr": True}
    assert result.usage.cost_usd is None and result.tool_calls == []


def test_openrouter_errors():
    with pytest.raises(ModelError, match="HTTP 400"):
        _client([_http_error(400)], []).complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10)
    # Errors reported inside a 200 body: retryable codes are retried, others raised.
    sent: list = []
    ok = {"choices": [{"message": {"content": "fine"}}]}
    result = _client([{"error": {"code": 502, "message": "upstream"}}, ok], sent).complete(
        system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10
    )
    assert result.text == "fine" and len(sent) == 2
    with pytest.raises(ModelError, match="error 401"):
        _client([{"error": {"code": 401, "message": "bad key"}}], []).complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10)
    with pytest.raises(ModelError, match="no choices"):
        _client([{"choices": []}], []).complete(system="s", messages=[{"role": "user", "content": "x"}], tools=[], max_tokens=10)


def test_create_client_requires_openrouter_key():
    config = RunConfig().model
    with pytest.raises(ConfigError, match="OPENROUTER_API_KEY"):
        create_client(config, {})
    client = create_client(config, {"OPENROUTER_API_KEY": "sk-or-v1-abc", "OPENROUTER_BASE_URL": "http://localhost:9/v1/"})
    assert client.provider == "openrouter" and client._url == "http://localhost:9/v1/chat/completions"


def test_budget_prefers_reported_cost():
    budget = Budget(Limits(), ModelConfig(input_cost_per_mtok=2.0, output_cost_per_mtok=10.0))
    budget.record(Usage(input_tokens=1_000_000, output_tokens=0, cost_usd=0.5))
    assert budget.estimated_cost_usd == pytest.approx(0.5)
    budget.record(Usage(input_tokens=1_000_000, output_tokens=100_000))  # no reported cost: estimate
    assert budget.estimated_cost_usd == pytest.approx(0.5 + 2.0 + 1.0)
    assert budget.usage.input_tokens == 2_000_000


def test_cli_check_end_to_end(tmp_path, capsys):
    from conftest import TOY_FILES

    from resolv.cli import main

    repo = make_repo(tmp_path / "toy", TOY_FILES)
    code = main(["check", "--repo", str(repo), "--out", str(tmp_path / "out"), "--sandbox", "local", "--no-venv", "--python", os.environ.get("PYTHON", __import__("sys").executable)])
    assert code == 0
    assert "checked" in capsys.readouterr().out
