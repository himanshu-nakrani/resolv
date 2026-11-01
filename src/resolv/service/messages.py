"""Text posted to GitHub: issue status comments and draft pull request bodies."""

from __future__ import annotations

from resolv.report import isolation_note, RunRecord

STAGES = (
    ("preflight", "Preflight"),
    ("setup", "Environment setup"),
    ("baseline", "Baseline tests"),
    ("investigate", "Investigation"),
    ("reproduce", "Reproduction"),
    ("patch", "Patch"),
    ("validate", "Validation"),
    ("deliver", "Delivery"),
)
OUTCOME_LABELS = {
    "draft_pr_created": "Draft PR created",
    "blocked": "Blocked",
    "failed": "Failed",
    "cancelled": "Cancelled",
}
MARKER = "<!-- resolv-run -->"


def safe_text(text: str | None, limit: int = 1500) -> str:
    """Model- or issue-derived text: truncate and neutralise @mentions so nobody gets pinged."""
    text = (text or "").strip()
    if len(text) > limit:
        text = text[:limit].rstrip() + " …"
    return text.replace("@", "@\u200b")


def progress_comment(run, run_url: str) -> str:
    current = run["stage"] or "queued"
    names = [name for name, _ in STAGES]
    position = names.index(current) if current in names else -1
    lines = [
        MARKER,
        f"**resolv** is working on this issue. [Follow the run]({run_url})",
        "",
    ]
    for i, (name, label) in enumerate(STAGES):
        mark = "x" if i < position else " "
        suffix = " ← in progress" if i == position else ""
        lines.append(f"- [{mark}] {label}{suffix}")
    lines += ["", "_An AI agent is investigating this report. It will open a draft pull request only if it can reproduce "
              "the bug with a failing test and validate a fix._"]
    return "\n".join(lines)


def final_comment(run, run_url: str) -> str:
    outcome = run["outcome"]
    label = OUTCOME_LABELS.get(outcome, outcome or "finished")
    if outcome == "draft_pr_created":
        body = [
            f"**resolv** opened a draft pull request: #{run['pr_number']}",
            "",
            "It contains a regression test that failed on the original code and a proposed fix that makes it pass "
            "without new failures in the configured test suite. This is an AI-generated proposal; please review it "
            "carefully before merging.",
        ]
    else:
        body = [
            f"**resolv** stopped without proposing a fix — **{label}**",
            "",
            f"> {safe_text(run['reason']).replace(chr(10), chr(10) + '> ')}",
        ]
        if run["reason_code"] == "base_branch_moved":
            body += ["", "Re-apply the trigger label to run again on the latest code."]
    body += ["", f"[Run details]({run_url}) · reason code `{run['reason_code']}`"]
    return MARKER + "\n" + "\n".join(body)


def _counts(counts: dict | None) -> str:
    if not counts:
        return "no tests"
    return ", ".join(f"{n} {k}" for k, n in counts.items())


def pr_title(issue_number: int, issue_title: str | None) -> str:
    title = (issue_title or "").strip() or f"issue #{issue_number}"
    text = f"Fix #{issue_number}: {title}"
    return text if len(text) <= 120 else text[:117] + "..."


def commit_message(issue_number: int, issue_title: str | None, run_url: str) -> str:
    return f"{pr_title(issue_number, issue_title)}\n\nAI-generated proposed fix by resolv.\nRun: {run_url}\n"


def pr_body(report: dict, *, issue_number: int, issue_url: str | None, run_url: str) -> str:
    r = RunRecord(**report)
    model = f"`{r.model['name']}` via {r.model['provider']}" if r.model else "an AI model"
    usage = r.usage or {}
    lines = [
        f"> [!NOTE]\n> **AI-generated proposed patch.** resolv produced this draft using {model}. It has not been "
        "reviewed by a human — review it before merging.",
        "",
        f"**Issue:** #{issue_number}" + (f" ({issue_url})" if issue_url else ""),
        f"**Run:** {run_url} · base commit `{(r.base_commit or '')[:12]}` · {usage.get('elapsed_seconds', 0)}s · "
        f"est. model cost ${usage.get('estimated_cost_usd', 0):.2f}",
        "",
    ]
    if r.diagnosis:
        files = ", ".join(f"`{f}`" for f in r.diagnosis.get("relevant_files", []))
        lines += ["## Cause", "", safe_text(r.diagnosis.get("cause"), 3000), "", f"Relevant files: {files}", ""]
    if r.patch:
        lines += ["## Changes", "", safe_text(r.patch.get("changes"), 3000), "", "| File | + | - |", "|---|---|---|"]
        lines += [f"| `{f['path']}` | {f['added']} | {f['deleted']} |" for f in r.patch.get("files", [])]
        lines.append("")
    if r.reproduction:
        rep = r.reproduction
        lines += ["## Reproduction", "", "Regression tests added in this PR:"]
        lines += [f"- `{n}`" for n in rep.get("node_ids", [])]
        lines += ["", "Result on the original code (before the fix):"]
        for t in rep.get("before", []):
            message = (t.get("message") or "").strip().splitlines()
            detail = f": {safe_text(message[0], 300)}" if message else ""
            lines.append(f"- `{t['nodeid']}`: **{t['outcome']}** with `{t.get('exc_type')}`{detail}")
        lines.append("")
    v = r.validation or {}
    if v:
        suite_cmd = " ".join(v.get("suite_command", {}).get("argv", []))
        lines += ["## Validation", "", "Regression tests with the fix:"]
        lines += [f"- `{t['nodeid']}`: **{t['outcome']}**" for t in v.get("regression_after", [])]
        lines += [
            "",
            f"Configured suite (`{suite_cmd}`): {_counts(v.get('suite_counts'))}. "
            f"Before the change: {_counts(v.get('baseline_counts'))}.",
            "",
            "- New failures: " + (", ".join(f"`{t['nodeid']}`" for t in v.get("new_failures", [])) or "none"),
        ]
        pre = v.get("pre_existing_failures", [])
        if pre:
            lines.append(
                f"- **Pre-existing failures** ({len(pre)}, also failing before this change): "
                + ", ".join(f"`{t['nodeid']}`" for t in pre[:20]) + (" …" if len(pre) > 20 else "")
            )
        not_run = v.get("skipped_or_xfailed", [])
        lines.append("- Tests not run (skipped/xfailed): " + (", ".join(f"`{n}`" for n in not_run[:20]) or "none"))
        if v.get("regression_not_in_suite"):
            lines.append("- The regression tests are not collected by the configured test command; they were run separately.")
        lines += ["- Only the configured pytest command was run; linters, type checkers, and other CI jobs were not.", ""]
    notes = []
    if r.patch and r.patch.get("assumptions_and_risks"):
        notes.append("Agent-stated assumptions and risks: " + safe_text(r.patch["assumptions_and_risks"], 2000))
    notes.append("The regression test was written by the agent. Passing tests are evidence, not proof, that the issue is solved.")
    notes += [safe_text(w, 500) for w in r.warnings]
    if r.environment.get("isolation"):
        notes.append(isolation_note(r))
    lines += ["## Review notes", "", *(f"- {n}" for n in notes), ""]
    lines += [
        "## Disclosure",
        "",
        "This pull request was generated by an AI system (resolv) as a proposed patch. Repository code and the issue "
        "text were sent to the model provider named above. resolv never merges or approves pull requests.",
    ]
    return "\n".join(lines)
