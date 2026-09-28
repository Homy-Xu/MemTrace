"""Freeze an official submission once its score is already good.

The official collector keeps the latest tag.  A later ``git tag -f`` replaces
a finished score even when the new tree is worse.  This module decides that
from ``summary.json`` counts only: it never reads test names.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

GOOD_SCORE = 90.0
_ATTEMPT = re.compile(r"^(.*)-retry(\d+)$")
_SHA = re.compile(r"^[0-9a-f]{40}$")



def _attempt_number(key: str, item: dict) -> int:
    match = _ATTEMPT.match(str(key))
    return int(match.group(2)) if match else int(item.get("attempt") or 0)


def first_attempt_is_complete(item: dict | None) -> bool:
    """Return whether one official attempt-0 result is safe to lock.

    The evaluator's ``passed`` status is the authoritative resolution signal;
    the count checks make the lock fail closed if a malformed or partial
    summary is published. Ordinary baseline failures are allowed when every
    required F2P/N2P/P2P unit passes.
    """

    if not isinstance(item, dict) or _attempt_number("", item) != 0:
        return False
    # ``summary.json`` marks an evaluator phase as ``passed`` before the
    # authoritative per-milestone evaluation_result is necessarily available.
    # A first-attempt lock is legal only after that receipt explicitly says
    # resolved=true; missing metadata fails closed.
    if item.get("eval_status") != "passed":
        return False
    if item.get("resolved") is not True:
        return False
    if item.get("official_receipt_state") != "SCORED":
        return False
    commit = str(item.get("commit") or item.get("tag_hash") or "")
    if not _SHA.fullmatch(commit) or item.get("receipt_error"):
        return False
    if item.get("infra_invalid") is True or item.get("infrastructure_failure"):
        return False
    if item.get("start_compile_error") or item.get("end_compile_error"):
        return False
    summary = item.get("test_summary")
    if not isinstance(summary, dict) or int(summary.get("total") or 0) <= 0:
        return False
    if int(summary.get("error") or 0) != 0:
        return False
    for achieved, required in (
        ("fail_to_pass_achieved", "fail_to_pass_required"),
        ("none_to_pass_achieved", "none_to_pass_required"),
        ("pass_to_pass_achieved", "pass_to_pass_required"),
    ):
        if int(summary.get(achieved) or 0) != int(summary.get(required) or 0):
            return False
    return (
        int(summary.get("pass_to_pass_failed") or 0) == 0
        and int(summary.get("pass_to_pass_missing") or 0) == 0
    )


def first_good_official_results(summary: dict) -> dict[str, dict]:
    """Select the first good attempt for each milestone, preferring attempt 0.

    A good attempt-0 result becomes a durable stop condition. If attempt 0
    was not good, later attempts remain eligible for the normal retry policy.
    """

    results = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(results, dict):
        return {}
    grouped: dict[str, list[tuple[int, dict]]] = {}
    for key, item in results.items():
        if not isinstance(item, dict):
            continue
        match = _ATTEMPT.match(str(key))
        milestone = match.group(1) if match else str(key)
        grouped.setdefault(milestone, []).append((_attempt_number(str(key), item), item))
    selected: dict[str, dict] = {}
    for milestone, attempts in grouped.items():
        attempts.sort(key=lambda pair: pair[0])
        first = next((item for attempt, item in attempts if attempt == 0), None)
        if first is not None and first_attempt_is_complete(first):
            selected[milestone] = {
                **first,
                "milestone_id": milestone,
                "attempt": 0,
                "score": score_reliable(first.get("test_summary")),
                "good": True,
                "lock_reason": "FIRST_ATTEMPT_LOCKED",
            }
            continue
        for attempt, item in reversed(attempts):
            status = item.get("eval_status")
            score = score_reliable(item.get("test_summary") if status != "error" else {"total": 0})
            if is_good_official_result(str(status) if status else None, score):
                selected[milestone] = {
                    **item,
                    "milestone_id": milestone,
                    "attempt": attempt,
                    "score": score,
                    "good": True,
                    "lock_reason": "RETRY_ATTEMPT_LOCKED",
                }
                break
    return selected


def score_reliable(test_summary: dict | None) -> float | None:
    """Official F1 over fix and regression counts, as a percentage.

    Returns ``None`` when the attempt has no test summary.  A zero-test
    attempt is a compilation or infrastructure failure and scores 0.
    """

    if not isinstance(test_summary, dict):
        return None
    total = int(test_summary.get("total") or 0)
    if total == 0:
        return 0.0
    fixed = int(test_summary.get("fail_to_pass_achieved") or 0) + int(
        test_summary.get("none_to_pass_achieved") or 0
    )
    target = int(test_summary.get("fail_to_pass_required") or 0) + int(
        test_summary.get("none_to_pass_required") or 0
    )
    broken = int(test_summary.get("pass_to_pass_failed") or 0) + int(
        test_summary.get("pass_to_pass_missing") or 0
    )
    recall = (1.0 if fixed == 0 else 0.0) if target == 0 else fixed / target
    precision = (fixed + 1.0) / (fixed + broken + 1.0)
    if precision == 0 and recall == 0:
        return 0.0
    return 100.0 * 2 * precision * recall / (precision + recall)


def is_good_official_result(eval_status: str | None, score: float | None) -> bool:
    """A passed result, or a score at or above the freeze threshold."""

    if eval_status == "passed":
        return True
    return score is not None and score >= GOOD_SCORE


def latest_official_results(summary: dict) -> dict[str, dict]:
    """One row per milestone: the highest attempt, counts only."""

    results = summary.get("results") if isinstance(summary, dict) else None
    if not isinstance(results, dict):
        return {}
    best: dict[str, tuple[int, dict]] = {}
    for key, item in results.items():
        if not isinstance(item, dict):
            continue
        match = _ATTEMPT.match(str(key))
        milestone = match.group(1) if match else str(key)
        attempt = int(match.group(2)) if match else int(item.get("attempt") or 0)
        status = item.get("eval_status")
        if status in {None, "not_run", "available", "blocked", "submitted", "unlocked"}:
            summary_counts = item.get("test_summary") or {}
            if not summary_counts and status != "error":
                continue
        score = score_reliable(item.get("test_summary") if status != "error" else {"total": 0})
        if status == "error":
            score = 0.0
        tag_hash = str(item.get("tag_hash") or "")
        row = {
            "milestone_id": milestone,
            "attempt": attempt,
            "eval_status": status,
            "score": score,
            "commit": tag_hash if _SHA.fullmatch(tag_hash) else "",
            "good": is_good_official_result(str(status) if status else None, score),
        }
        previous = best.get(milestone)
        if previous is None or attempt >= previous[0]:
            best[milestone] = (attempt, row)
    return {milestone: row for milestone, (_, row) in best.items()}


def load_official_results(path: Path) -> dict[str, dict]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return latest_official_results(payload if isinstance(payload, dict) else {})


def redact_end_compile_output(output: str, submitted_paths: tuple[str, ...]) -> str:
    """Keep compiler lines that name a submitted file; drop official test paths."""

    kept: list[str] = []
    for line in output.splitlines():
        if not any(path and path in line for path in submitted_paths):
            continue
        if _looks_like_official_test_path(line):
            continue
        kept.append(line)
    if not kept:
        return (
            "The official eval tree did not compile after overlaying the submitted "
            "sources. Repair the submitted files. Official test names are not included."
        )
    return "\n".join(kept[-40:])


def select_eval_image(images: list[str], project_id: str, milestone_id: str) -> str:
    """Pick the local eval image that still has this milestone's END tag.

    The agent repository deletes that tag before the model runs.  The eval
    image is a separate container, so its tree is not mounted where the model
    can read official tests.
    """

    project = project_id.lower()
    milestone = milestone_id.lower()
    needle = f"__{milestone}-eval-closure"
    matches = [
        image for image in images
        if needle in image.lower() and (not project or project in image.lower())
    ]
    return matches[-1] if matches else ""


def _looks_like_official_test_path(line: str) -> bool:
    markers = (
        "/src/test/",
        "/tests/",
        "_test.go",
        "test_",
        ".spec.ts",
        ".test.ts",
        "__tests__",
        "Test.java",
        "Tests.java",
    )
    return any(marker in line for marker in markers)
