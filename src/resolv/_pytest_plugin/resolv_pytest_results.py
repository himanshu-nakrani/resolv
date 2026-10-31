"""pytest plugin loaded by resolv with ``-p resolv_pytest_results``.

Records one JSON entry per test (outcome, failing phase, exception type, and
innermost traceback file) to the path in RESOLV_RESULTS_FILE. Must stay
compatible with a wide range of pytest versions and import nothing from resolv.
"""

from __future__ import annotations

import json
import os
import re

import pytest

_FAILING = ("failed", "error")
_MAX_DETAILS = 4000
_EXC_RE = re.compile(r"^E\s+([A-Za-z_][\w.]*(?:Error|Exception|Exit|Interrupt|Warning))\b", re.M)

_results: dict = {}
_order: list = []
_exceptions: dict = {}
_collect_errors: list = []


def _longrepr_text(report) -> str:
    try:
        return str(getattr(report, "longreprtext", "") or "")
    except Exception:
        return ""


def _crash_message(report) -> str:
    longrepr = getattr(report, "longrepr", None)
    if isinstance(longrepr, tuple) and len(longrepr) == 3:  # skip: (path, lineno, reason)
        return str(longrepr[2])
    crash = getattr(longrepr, "reprcrash", None)
    return str(getattr(crash, "message", "") or "")


def _guess_exception(text: str):
    found = _EXC_RE.findall(text)
    return found[-1].rsplit(".", 1)[-1] if found else None


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_makereport(item, call):
    excinfo = call.excinfo
    if excinfo is None:
        return None
    crash_path = None
    try:
        traceback = excinfo.traceback
        if len(traceback):
            crash_path = str(traceback[-1].path)
    except Exception:
        pass
    _exceptions[(item.nodeid, call.when)] = (excinfo.typename, crash_path)
    return None


def pytest_runtest_logreport(report):
    nodeid = report.nodeid
    if hasattr(report, "wasxfail"):
        outcome = "xfailed" if report.skipped else "xpassed"
    elif report.when == "call":
        outcome = report.outcome
    elif report.failed:
        outcome = "error"
    elif report.skipped:
        outcome = "skipped"
    else:
        return  # setup/teardown passed
    previous = _results.get(nodeid)
    if previous is not None and previous["outcome"] in _FAILING:
        return  # keep the first failure
    exc_type, crash_path = _exceptions.get((nodeid, report.when), (None, None))
    details = _longrepr_text(report) if outcome in _FAILING else ""
    if exc_type is None and outcome in _FAILING:
        exc_type = _guess_exception(details)
    if previous is None:
        _order.append(nodeid)
    _results[nodeid] = {
        "nodeid": nodeid,
        "outcome": outcome,
        "when": report.when,
        "exc_type": exc_type,
        "crash_path": crash_path,
        "message": _crash_message(report)[:1000],
        "details": details[-_MAX_DETAILS:],
    }


def pytest_collectreport(report):
    if not report.failed:
        return
    text = _longrepr_text(report)
    lines = [line for line in text.strip().splitlines() if line.strip()]
    _collect_errors.append(
        {
            "nodeid": report.nodeid,
            "outcome": "error",
            "when": "collect",
            "exc_type": _guess_exception(text),
            "crash_path": None,
            "message": (lines[-1] if lines else "collection error")[:1000],
            "details": text[-_MAX_DETAILS:],
        }
    )


def pytest_sessionfinish(session, exitstatus):
    if hasattr(session.config, "workerinput"):  # pytest-xdist worker
        return
    path = os.environ.get("RESOLV_RESULTS_FILE")
    if not path:
        return
    payload = {
        "exitstatus": int(exitstatus),
        "tests": [_results[n] for n in _order],
        "collect_errors": _collect_errors,
    }
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(payload, fh)
    os.replace(tmp, path)
