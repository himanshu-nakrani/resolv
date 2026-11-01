# resolv

resolv turns a bug report into a **reproducible failure → focused fix → reviewable evidence** bundle for
small-to-medium Python repositories tested with pytest.

This is **Phase 1: the local prototype**. You give it issue text and a local git checkout. It produces
a patch plus a test report, or it stops with a blocker report that explains why. It does not talk to
GitHub, and it never modifies your checkout.

## Quick start

```bash
uv sync                                  # or: pip install -e .
export OPENROUTER_API_KEY=...            # models are called through OpenRouter

# 1. Check that the environment installs and the baseline suite runs (no model calls)
uv run resolv check --repo ../myproject \
  --install "python -m pip install -e .[test]"

# 2. Run the full workflow on one issue
uv run resolv run --repo ../myproject --issue issue.md \
  --issue-url https://github.com/org/myproject/issues/123 \
  --install "python -m pip install -e .[test]"
```

Each run writes `resolv-runs/<run-id>/` with:

| File | Contents |
|---|---|
| `report.md` | Draft-PR-style report: issue, cause, changes, reproduction, validation, review notes, disclosure |
| `report.json` | The same data in machine-readable form |
| `patch.diff` | The proposed patch (regression test + fix). Written **only** when the run succeeds; apply it with `git apply` |
| `candidate.diff` | Written only when a candidate patch exceeds the scope limits and is held for review |
| `events.jsonl` | Every model turn, tool call, and command, with secrets redacted |
| `logs/` | Output of every command (install, baseline, reproduction, validation) |

Exit codes: `0` patch ready (or check passed), `3` blocked, `4` failed, `130` cancelled, `2` usage error.

## Workflow

The orchestrator controls the workflow. The model can only propose searches, edits, and test runs through
a fixed set of tools, and the orchestrator verifies every claim by running the tests itself.

| Stage | What happens | Stops the run when |
|---|---|---|
| Preflight | Pins the commit, checks that the repo is a Python git repo, validates config and model credentials | The repo is unsupported, the ref is invalid, the issue is empty, or no credentials are set |
| Setup | Clones the pinned commit into a temporary directory, creates a venv, runs the install commands | An install command fails (`install_failed`), which is reported separately from code failures |
| Baseline | Runs the configured test command on unchanged code and records per-test outcomes | Pytest cannot run or collect tests (`environment_error`) |
| Investigate | The model reads and searches code (read-only) and submits a diagnosis | The model reports the issue as ambiguous, not a bug, or out of scope |
| Reproduce | The model writes a new regression test. resolv runs it on unchanged production code | No valid reproduction after the allowed attempts (`not_reproducible`) |
| Patch / Validate | The model edits production code only. For each submission resolv runs policy checks, the regression tests, and the full suite, then compares the results with the baseline | Attempts are exhausted (`patch_attempts_exhausted`) or the patch exceeds the scope limits (`patch_limit_exceeded`) |
| Deliver | Rechecks cancellation and budget, then writes `patch.diff` and the reports | — |

The baseline runs before investigation so that a broken environment fails fast, before any model spend.

### What counts as a reproduction

Every submitted test must be **new** (absent from the baseline) and must **fail in the test body** on the
pinned commit. These failures are rejected:

- errors in setup, teardown, or collection, such as fixture errors
- `ImportError`, `ModuleNotFoundError`, `SyntaxError`, `IndentationError`, and `TabError`
- `NameError` or `UnboundLocalError` raised inside the test file itself

The model sees the rejection reason and can revise the test, up to `max_reproduction_attempts` times. The
exception type and the innermost traceback file come from a small pytest plugin
(`src/resolv/_pytest_plugin/resolv_pytest_results.py`) that resolv loads with `-p` for every test run.

### Publication criteria

A run ends as `patch_ready` only if all of these hold:

- the regression tests fail on the base commit and pass with the patch
- the configured suite has no new failures and no tests that disappeared, compared with the baseline
- every policy check passes

Tests that were already failing at baseline are disclosed prominently in the report.

### Patch policy (enforced outside the model)

- **Protected files cannot be written:** dependency, packaging, CI, and test configuration. Examples:
  `pyproject.toml`, `setup.py`, `setup.cfg`, `tox.ini`, `pytest.ini`, `conftest.py`, requirements and lock
  files, Dockerfiles, and `.github/`.
- **Reproduction stage:** only test files can be written. Existing test files are append-only, so every
  original line must stay unchanged.
- **Patch stage:** test files are locked, so the reproduction stays fixed. Only production code can change.
- **Always:** no skip or xfail markers (`pytest.mark.skip`, `pytest.skip(...)`, `unittest.skip`,
  `collect_ignore`, …), no deleted files, and no binary changes.
- **Scope:** at most 5 changed files and 300 changed lines by default, counting the regression test. If a
  candidate exceeds a limit, the run stops for review.
- **Diffs:** cover only files the model wrote through its tools. Artifacts from installs or test runs never
  leak into the patch.

### Safety controls in Phase 1

- **Configuration:** comes only from the CLI or a TOML file you supply, never from the target repository.
- **Untrusted content:** issue text, file contents, and test output are fenced as untrusted data in prompts.
  They cannot change the tools, budgets, or policy, which the orchestrator enforces in code.
- **No shell tool:** the model has no general shell. `run_tests` accepts only node IDs or paths inside the
  checkout, never pytest options.
- **Command environment:** commands run in a scrubbed environment. API keys and host variables are not
  passed through unless they are listed in `pass_env`. Each command gets a private `HOME` and its own process
  group, so timeouts and cancellation kill the whole process tree.
- **Budgets:** wall-clock time, tokens, estimated spend (USD 3 by default), per-command timeout, and command
  and tool output size are all enforced by the orchestrator.
- **Cancellation:** Ctrl-C or SIGTERM ends the run as `cancelled`, kills running commands, and writes no
  patch.
- **Redaction:** logs, events, and reports are redacted for likely secrets before they are written.

**Limitation:** Phase 1 runs repository code directly on your machine as your user, without network
isolation or VM sandboxing (that is Phase 3). Use it only on repositories you trust, or run it inside a
disposable VM or container.

## Configuration

Settings can come from CLI flags or a TOML file passed with `--config`. Keep the file outside the target
repository. All keys are optional; these are the defaults:

```toml
[environment]
python = "python3"                 # default: the interpreter running resolv
venv = true                        # create an isolated venv for the run
install_commands = []              # e.g. ["python -m pip install -e .[test]"]
test_command = "python -m pytest"  # must invoke pytest and accept extra args
pass_env = []                      # host env vars to pass through, e.g. ["HTTPS_PROXY"]

[limits]
max_changed_files = 5
max_changed_lines = 300
max_patch_attempts = 3
max_reproduction_attempts = 3
max_turns_per_stage = 40
run_timeout_seconds = 1800
command_timeout_seconds = 600
max_total_tokens = 3000000
max_cost_usd = 3.0
max_command_output_chars = 200000
max_tool_output_chars = 12000
max_issue_chars = 30000

[model]
provider = "openrouter"
name = "anthropic/claude-sonnet-5.5" # any OpenRouter model id that supports tool calling
max_output_tokens = 8192
input_cost_per_mtok = 2.0          # fallback prices, used only when OpenRouter reports no cost
output_cost_per_mtok = 10.0
request_timeout_seconds = 300
data_collection = "deny"           # route only to providers that don't store or train on prompts
zero_data_retention = false        # true restricts routing to zero-data-retention endpoints
```

### Model provider

resolv calls models through [OpenRouter](https://openrouter.ai)'s chat-completions API, using
`OPENROUTER_API_KEY` (and optionally `OPENROUTER_BASE_URL`). Choose a model with `--model` or
`[model].name`, for example `openai/gpt-6.1-sol` or `deepseek/deepseek-v4-pro`. The model must support
tool calling.

- **Cost cap:** counts the actual cost OpenRouter reports for each call. If a call reports no cost,
  resolv estimates it from the fallback prices above.
- **Routing:** every request sets `provider.require_parameters = true`, so OpenRouter only routes to
  endpoints that support tools. By default it also sets `data_collection = "deny"`.
- **Prompt caching:** `anthropic/*` models get prompt caching enabled. Most other providers cache
  automatically.
- **Disclosure:** the report names the upstream providers OpenRouter routed to.

With `venv = false`, resolv runs commands with the configured interpreter (`python` on `PATH` points to
it) instead of creating a venv.

## Development

```bash
uv sync
uv run pytest
```

The test suite drives the whole workflow with a deterministic `ScriptedModel`
(`src/resolv/model/scripted.py`) against throwaway git repositories, so it needs no API key or network
access.

## Layout

```
src/resolv/
  cli.py            CLI (run / check)
  orchestrator.py   stage workflow, verification, outcomes
  agent.py          per-stage tool-use loop
  tools.py          model-facing tools (read/search/write/run_tests/final submissions)
  policy.py         protected paths, test integrity, scope limits
  workspace.py      pinned disposable git checkout, path confinement, diffs
  testrun.py        pytest execution, result parsing, baseline comparison
  executor.py       scrubbed, time-limited, output-capped command runner
  budget.py         time / token / cost budgets and cancellation
  model/            model gateway (OpenRouter chat completions, scripted model)
  report.py         report.json and report.md
```
