"""Run record plus JSON and Markdown reports (Phase 1 stand-in for the draft PR body)."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from resolv.redact import Redactor

OUTCOME_LABELS = {
    "patch_ready": "Patch ready for review (not applied)",
    "blocked": "Blocked",
    "failed": "Failed",
    "cancelled": "Cancelled",
    "checked": "Environment check complete",
}


@dataclass
class RunRecord:
    run_id: str
    started_at: str
    repo: str
    ref: str
    mode: str
    issue: dict | None
    environment: dict
    limits: dict
    base_commit: str | None = None
    finished_at: str | None = None
    outcome: str | None = None
    reason_code: str | None = None
    reason: str | None = None
    model: dict | None = None
    stages: list[dict] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    baseline: dict | None = None
    diagnosis: dict | None = None
    blocker: dict | None = None
    reproduction: dict | None = None
    reproduction_attempts: list[dict] = field(default_factory=list)
    patch: dict | None = None
    patch_attempts: list[dict] = field(default_factory=list)
    validation: dict | None = None
    commands: list[dict] = field(default_factory=list)
    usage: dict = field(default_factory=dict)
    artifacts: dict = field(default_factory=dict)


ENVIRONMENT_LIMITATIONS = (
    "Phase 1 runs repository code directly on this machine as the current user, with a scrubbed environment "
    "but without network isolation or VM sandboxing. Run it only on repositories you trust or inside a "
    "disposable VM/container."
)


def _md_escape(text: str) -> str:
    return str(text).replace("|", "\\|").replace("\n", " ")


def _bullets(items, empty: str = "None.") -> list[str]:
    items = list(items)
    return [f"- {i}" for i in items] if items else [empty]


def _counts(counts: dict | None) -> str:
    if not counts:
        return "no tests"
    return ", ".join(f"{n} {k}" for k, n in counts.items())


def render_markdown(r: RunRecord) -> str:
    out: list[str] = []
    title = r.issue["title"] if r.issue else "environment check"
    out.append(f"# resolv report: {title}\n")
    if r.mode == "run":
        model = f"{r.model['name']} via {r.model['provider']}" if r.model else "an AI model"
        out.append(
            f"> **AI-generated proposed patch.** Produced by resolv (Phase 1 local prototype) using {model}. "
            "It has not been reviewed by a human; review it before applying.\n"
        )
    usage = r.usage or {}
    rows = [
        ("Outcome", OUTCOME_LABELS.get(r.outcome or "", r.outcome or "unknown")),
        ("Reason code", f"`{r.reason_code}`"),
        ("Repository", f"`{r.repo}` (ref `{r.ref}`)"),
        ("Base commit", f"`{r.base_commit}`" if r.base_commit else "not pinned"),
        ("Elapsed", f"{usage.get('elapsed_seconds', 0)}s"),
    ]
    if r.mode == "run":
        rows.append(
            (
                "Model usage",
                f"{usage.get('model_calls', 0)} calls, {usage.get('input_tokens', 0)} input + "
                f"{usage.get('cache_read_tokens', 0)} cached + {usage.get('cache_write_tokens', 0)} cache-write input, "
                f"{usage.get('output_tokens', 0)} output tokens; cost ${usage.get('estimated_cost_usd', 0):.2f} "
                f"(${usage.get('reported_cost_usd', 0):.2f} reported by the provider, the rest estimated)",
            )
        )
        if r.model and r.model.get("upstream_providers"):
            rows.append(("Upstream model providers", ", ".join(r.model["upstream_providers"])))
    out += ["| | |", "|---|---|", *(f"| {k} | {_md_escape(v)} |" for k, v in rows), ""]

    baseline_failing = (r.baseline or {}).get("failing") or []
    if baseline_failing:
        out.append(
            f"> **Warning:** {len(baseline_failing)} test(s) were already failing on the base commit before any "
            "change. They are listed in this report and are not caused by any change made by resolv.\n"
        )

    if r.outcome != "patch_ready":
        out += ["## Why the run stopped", "", r.reason or "", ""]

    if r.issue:
        out += ["## Issue", ""]
        if r.issue.get("url"):
            out.append(f"Original report: {r.issue['url']}")
        if r.issue.get("source"):
            out.append(f"Issue text source: `{r.issue['source']}`")
        out.append(f"Title: {r.issue['title']}\n")

    if r.diagnosis:
        d = r.diagnosis
        out += ["## Cause", "", d.get("cause", ""), "", "Relevant files: " + ", ".join(f"`{f}`" for f in d.get("relevant_files", [])), ""]

    if r.patch:
        out += ["## Changes", "", r.patch["changes"], "", "| File | + | - |", "|---|---|---|"]
        out += [f"| `{f['path']}` | {f['added']} | {f['deleted']} |" for f in r.patch["files"]]
        out += ["", f"Full diff: `{r.artifacts.get('patch', 'patch.diff')}` (apply with `git apply`).", ""]

    if r.reproduction:
        rep = r.reproduction
        out += ["## Reproduction", "", "Regression tests added by the agent:"]
        out += _bullets(f"`{n}`" for n in rep["node_ids"])
        out += ["", f"Expected failure (agent's description): {rep['expected_failure']}", "", "Observed on the unchanged base commit:"]
        for t in rep["before"]:
            message = (t.get("message") or "").strip().splitlines()
            out.append(f"- `{t['nodeid']}`: **{t['outcome']}** with `{t.get('exc_type')}`" + (f": {message[0][:300]}" if message else ""))
        out.append("")
    elif r.reproduction_attempts:
        out += ["## Reproduction attempts", ""]
        for a in r.reproduction_attempts:
            out += [f"**Submission {a['attempt']}:** {'accepted' if a['accepted'] else 'rejected'}", "", "```", (a.get("feedback") or "")[:3000], "```", ""]

    if r.validation:
        v = r.validation
        out += ["## Validation", "", "Regression tests with the patch:"]
        out += _bullets(f"`{t['nodeid']}`: **{t['outcome']}**" for t in v["regression_after"])
        cmd = " ".join(v["suite_command"]["argv"])
        out += [
            "",
            f"Configured suite (`{cmd}`): {_counts(v['suite_counts'])}. Baseline: {_counts(v['baseline_counts'])}.",
            "",
            "New failures compared with the baseline:",
            *_bullets(f"`{t['nodeid']}`" for t in v["new_failures"]),
            "",
            "Failing before and after (pre-existing, not caused by this patch):",
            *_bullets(f"`{t['nodeid']}`" for t in v["pre_existing_failures"]),
            "",
            "Previously failing tests that now pass:",
            *_bullets(f"`{n}`" for n in v["fixed"]),
            "",
            "Tests not run:",
        ]
        not_run = [f"`{n}` (skipped/xfailed)" for n in v["skipped_or_xfailed"]]
        if v["regression_not_in_suite"]:
            not_run.append("The regression tests are not collected by the configured test command; they were run separately.")
        not_run.append("Only the configured pytest command was run; linters, type checkers, and other suites were not.")
        out += _bullets(not_run) + [""]
    elif r.baseline:
        b = r.baseline
        out += ["## Baseline", "", f"Result: {_counts(b.get('counts'))}."]
        if b.get("environment_error"):
            out.append(f"Environment problem: {b['environment_error']}")
        if baseline_failing:
            out += ["", "Failing on the base commit:", *_bullets(f"`{t['nodeid']}` ({t['outcome']})" for t in baseline_failing)]
        out.append("")

    rejected = [a for a in r.patch_attempts if not a.get("accepted")]
    if rejected:
        out += ["## Rejected patch submissions", ""]
        for a in rejected:
            out += [f"**Submission {a['attempt']}:** {a.get('changes', '')[:500]}", "", "```", (a.get("feedback") or "")[:3000], "```", ""]

    if r.blocker:
        out += ["## Agent blocker report", "", f"Category: `{r.blocker['category']}`", "", r.blocker["reason"], ""]

    out += ["## Review notes", ""]
    notes = []
    if r.patch and r.patch.get("assumptions_and_risks"):
        notes.append(f"Agent-stated assumptions and risks: {r.patch['assumptions_and_risks']}")
    if r.mode == "run":
        notes.append("The regression test was written by the agent. A passing test is evidence, not proof, that the issue is solved.")
    notes += r.warnings
    notes.append(ENVIRONMENT_LIMITATIONS)
    out += _bullets(notes) + [""]

    out += ["## Run details", "", "| Stage | Status | Duration | Summary |", "|---|---|---|---|"]
    out += [
        f"| {s['name']} | {s['status']} | {s.get('duration_seconds', 0)}s | {_md_escape(s.get('summary', ''))[:200]} |"
        for s in r.stages
    ]
    out += ["", "| Command | Result | Duration | Log |", "|---|---|---|---|"]
    for c in r.commands:
        status = c["error"] or ("timed out" if c["timed_out"] else f"exit {c['exit_code']}")
        out.append(f"| `{_md_escape(' '.join(c['argv']))[:160]}` | {status} | {c['duration_seconds']}s | `{c['log_path']}` |")
    out += ["", "Event log: `events.jsonl`. Machine-readable report: `report.json`.", ""]

    if r.mode == "run":
        out += [
            "## Disclosure",
            "",
            "This is an AI-generated proposed patch and analysis. Repository code and issue text were sent to the model "
            "named above through OpenRouter and the upstream provider(s) it routed to. "
            "Do not merge without human review.",
            "",
        ]
    return "\n".join(out)


def write_reports(record: RunRecord, run_dir: Path, redactor: Redactor) -> None:
    data = redactor.deep(asdict(record))
    (run_dir / "report.json").write_text(json.dumps(data, indent=2, default=str) + "\n", encoding="utf-8")
    (run_dir / "report.md").write_text(redactor(render_markdown(record)), encoding="utf-8")
