"""Pilot metrics (Phase 4): outcomes, acceptance, review time, rejection reasons.

The spec's targets (≥30% maintainer-accepted fixes, 100% evidence
completeness, median run under 15 minutes, USD 3 cap) are shown next to the
measured values so pilot reviews can compare them directly.
"""

from __future__ import annotations

import statistics
from collections import Counter
from collections.abc import Iterable

from resolv.service.db import Database

FEEDBACK_VERDICTS = {
    "accepted": "Accepted as is",
    "accepted_with_changes": "Accepted after changes",
    "rejected": "Rejected",
}
REJECTION_REASONS = {
    "incorrect_fix": "Fix is incorrect",
    "incomplete_fix": "Fix is incomplete",
    "wrong_approach": "Wrong approach or design",
    "inadequate_test": "Regression test is inadequate",
    "unrelated_changes": "Unnecessary or unrelated changes",
    "code_quality": "Code style or quality",
    "not_a_bug": "Issue was not a real bug",
    "superseded": "Fixed elsewhere or superseded",
    "other": "Other",
}
TARGETS = {
    "acceptance_rate": 0.30,
    "evidence_complete_rate": 1.0,
    "median_duration_minutes": 15.0,
    "max_cost_usd": 3.0,
}


def _median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def compute_metrics(db: Database, repo_ids: Iterable[int] | None = None) -> dict:
    runs = db.list_runs(repo_ids, limit=100000)
    completed = [r for r in runs if r["status"] == "completed"]
    outcomes = Counter(r["outcome"] for r in completed)
    published = [r for r in completed if r["outcome"] == "draft_pr_created"]
    feedback = db.feedback_for_runs(r["id"] for r in published)

    merged = [r for r in published if r["pr_merged"]]
    closed_unmerged = [r for r in published if r["pr_closed_at"] and not r["pr_merged"]]
    verdicts = Counter(f["verdict"] for f in feedback.values())
    accepted = {r["id"] for r in merged} | {rid for rid, f in feedback.items() if f["verdict"] != "rejected"}
    rejected = ({r["id"] for r in closed_unmerged} | {rid for rid, f in feedback.items() if f["verdict"] == "rejected"}) - accepted
    decided = len(accepted) + len(rejected)

    review_hours = []
    for r in published:
        reviewed = r["pr_first_review_at"] or r["pr_closed_at"]
        if reviewed and r["pr_created_at"]:
            review_hours.append((reviewed - r["pr_created_at"]).total_seconds() / 3600)
    durations = [r["elapsed_seconds"] / 60 for r in completed if r["elapsed_seconds"]]
    costs = [r["cost_usd"] or 0.0 for r in completed]
    stop_reasons = Counter(r["reason_code"] for r in completed if r["outcome"] in ("blocked", "failed"))

    return {
        "runs_total": len(runs),
        "runs_completed": len(completed),
        "runs_active": len(runs) - len(completed),
        "outcomes": dict(outcomes),
        "published": len(published),
        "merged": len(merged),
        "closed_unmerged": len(closed_unmerged),
        "awaiting_review": len([r for r in published if not r["pr_closed_at"] and r["id"] not in feedback]),
        "acceptance_rate": len(accepted) / decided if decided else None,
        "decided": decided,
        "evidence_complete_rate": (sum(1 for r in published if r["evidence_complete"]) / len(published)) if published else None,
        "median_review_hours": _median(review_hours),
        "median_duration_minutes": _median(durations),
        "within_15_minutes_rate": (sum(1 for d in durations if d <= 15) / len(durations)) if durations else None,
        "median_cost_usd": _median(costs),
        "max_cost_usd": max(costs) if costs else None,
        "total_cost_usd": sum(costs),
        "verdicts": dict(verdicts),
        "rejection_reasons": dict(Counter(f["reason_category"] for f in feedback.values() if f["verdict"] == "rejected" and f["reason_category"])),
        "stop_reasons": dict(stop_reasons.most_common()),
        "targets": TARGETS,
    }


def _pct(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.0%}"


def _num(value: float | None, unit: str = "") -> str:
    return "n/a" if value is None else f"{value:.1f}{unit}"


def render_pilot_report(db: Database, repos: list) -> str:
    """Markdown summary for pilot reviews (``resolv-service pilot report``)."""
    lines = ["# resolv pilot report", ""]
    total = compute_metrics(db, [r["id"] for r in repos]) if repos else compute_metrics(db, [])
    lines += [
        "| Metric | Target | Measured |",
        "|---|---|---|",
        f"| Maintainer-accepted fixes (of decided PRs) | ≥ 30% | {_pct(total['acceptance_rate'])} ({total['decided']} decided) |",
        f"| Published PRs with complete before/after evidence | 100% | {_pct(total['evidence_complete_rate'])} |",
        f"| Median run duration | < 15 min | {_num(total['median_duration_minutes'], ' min')} |",
        f"| Highest model spend per run | ≤ $3.00 | {'n/a' if total['max_cost_usd'] is None else f'${total[chr(109) + chr(97) + chr(120) + chr(95) + chr(99) + chr(111) + chr(115) + chr(116) + chr(95) + chr(117) + chr(115) + chr(100)]:.2f}'} |",
        f"| Median time to first review | (track) | {_num(total['median_review_hours'], ' h')} |",
        "",
        f"Runs: {total['runs_total']} ({total['runs_active']} active). Outcomes: "
        + (", ".join(f"{k} {v}" for k, v in sorted(total["outcomes"].items())) or "none") + ".",
        "",
        "## Why runs stopped",
        "",
        *([f"- `{code}`: {n}" for code, n in total["stop_reasons"].items()] or ["- none"]),
        "",
        "## Rejection reasons",
        "",
        *([f"- {REJECTION_REASONS.get(code, code)}: {n}" for code, n in total["rejection_reasons"].items()] or ["- none recorded"]),
        "",
        "## Per repository",
        "",
        "| Repository | Runs | Draft PRs | Accepted | Rejected | Median minutes |",
        "|---|---|---|---|---|---|",
    ]
    for repo in repos:
        m = compute_metrics(db, [repo["id"]])
        lines.append(
            f"| {repo['full_name']} | {m['runs_total']} | {m['published']} | {m['merged'] + m['verdicts'].get('accepted', 0)} | "
            f"{m['closed_unmerged'] + m['verdicts'].get('rejected', 0)} | {_num(m['median_duration_minutes'])} |"
        )
    lines += ["", "Use the stop reasons and rejection reasons to tune each repository's eligibility rules "
              "(required and excluded labels, minimum issue length) in its settings."]
    return "\n".join(lines) + "\n"
