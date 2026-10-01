"""Runtime-owned SWE-Milestone regression guard (``trusted_verifier`` protocol)."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import threading
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict
from functools import wraps
from pathlib import Path
from typing import Any

from .contract import SweMilestoneContract
from .git_scope import (
    BaselineWorktree,
    GitScopeError,
    SubmissionScope,
    overlay_submitted_paths,
    resolve_scope,
)
from .languages import (
    ENVIRONMENT_OUTCOME,
    GuardFailure,
    UnitOutcome,
    backends_for_contract,
    run_command,
)
from .official_freeze import redact_end_compile_output

VERIFIER_COMMAND = "homy-swe-milestone-verifier"
_PASSED = "PASSED"
# A unit with no tests to run (Go package without test files, jest
# ``--passWithNoTests``) cannot regress; only FAILED/ERROR/MISSING can.
_NON_REGRESSION_OUTCOMES = frozenset({_PASSED, "SKIPPED"})


class VerifierTimeout(RuntimeError):
    """The verification budget was exhausted before an outcome was known."""


def _normalize_typecheck_command(command: str) -> str:
    """Drop worktree-specific paths from a ``tsc --noEmit -p`` command."""
    normalized = re.sub(r"(?:^|\s)\S*tsc(?=\s|$)", " tsc", command).strip()
    return re.sub(r"(-p\s+)\S+", r"\1<project>", normalized)


def _typescript_diagnostics(output: str) -> set[str]:
    """Return stable TypeScript diagnostics without worktree line prefixes."""
    result: set[str] = set()
    for line in (output or "").splitlines():
        match = re.search(r"\berror\s+(TS\d+):\s*(.+)$", line, re.IGNORECASE)
        if match:
            result.add(f"{match.group(1).upper()}:{match.group(2).strip()}")
    return result


def _guard_is_preexisting(target: GuardFailure, baseline: GuardFailure) -> bool:
    """Whether a target guard is already present in the baseline tree.

    TypeScript diagnostics are compared by code/message because worktree
    prefixes and line numbers differ between disposable trees. Other guard
    output remains exact: newly introduced build/manifest failures must still
    block verification.
    """
    if target.kind != baseline.kind:
        return False
    if target.kind == "typecheck_failed":
        # Disposable worktrees put a different directory in both the tsc
        # binary path and ``-p``. Diagnostics are the stable identity.
        target_diagnostics = _typescript_diagnostics(target.output)
        baseline_diagnostics = _typescript_diagnostics(baseline.output)
        if target_diagnostics and baseline_diagnostics:
            return target_diagnostics.issubset(baseline_diagnostics)
        target_command = _normalize_typecheck_command(target.command)
        baseline_command = _normalize_typecheck_command(baseline.command)
        if target_command != baseline_command:
            return False
    elif target.command != baseline.command:
        return False
    return target.output.strip() == baseline.output.strip()


def _filter_preexisting_guards(
    target: list[GuardFailure], baseline: list[GuardFailure]
) -> tuple[list[GuardFailure], list[GuardFailure]]:
    """Separate new deterministic guards from baseline defects."""
    remaining = list(baseline)
    new: list[GuardFailure] = []
    ignored: list[GuardFailure] = []
    for item in target:
        match_index = next(
            (index for index, candidate in enumerate(remaining)
             if _guard_is_preexisting(item, candidate)),
            None,
        )
        if match_index is None:
            new.append(item)
        else:
            ignored.append(item)
            remaining.pop(match_index)
    return new, ignored


def _record_unavailable_attempt(method):
    """Durably distinguish a verifier crash from a failed regression check."""

    @wraps(method)
    def wrapped(self, *args, **kwargs):
        target_tag = self._target_tag
        started = time.monotonic()
        try:
            return method(self, *args, **kwargs)
        except Exception as exc:
            attempt = self._attempt
            path = self.receipt_dir / f"attempt-{attempt}.json"
            if attempt > 0 and not path.exists():
                summary = {
                    "schema": "homy/swe-milestone-verifier-receipt@1",
                    "attempt": attempt,
                    "passed": None,
                    "verification_scope": "VERIFIER_UNAVAILABLE",
                    "project_id": self.contract.project_id,
                    "languages": list(self.contract.languages),
                    "target_tag": target_tag,
                    "error_type": type(exc).__name__,
                    "units": [],
                    "regressions": [],
                    "guard_failures": [],
                    "timed_out": False,
                    "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
                }
                try:
                    self._write_receipt(attempt, summary, [])
                except OSError:
                    # Never replace the original failure with a receipt error.
                    pass
            raise

    return wrapped


class SweMilestoneVerifier:
    """Affected-scope regression verification for one SWE-Milestone project.

    The instance is the ``trusted_verifier`` callable handed to the memory
    runtime's ``RunCoordinator``. A call verifies the current working tree against the
    previous official submission (or the run baseline) and returns the
    immutable receipt mapping the acceptance kernel understands.  The optional
    :meth:`bind_submission` pins the next call to an already-created
    ``agent-impl-<milestone>`` tag so the post-tag guard checks exactly the
    tree the official evaluator will see.
    """

    # The official stream can end without a model-authored submission marker.
    # RunCoordinator checks this explicit capability before taking one final,
    # deterministic observation.  Generic/Python trusted verifiers keep their
    # existing boundary-only behavior because they do not expose this flag.
    terminal_observation_enabled = True

    def __init__(
        self,
        repository: str | Path,
        contract: SweMilestoneContract,
        run_root: str | Path,
        *,
        scratch_root: str | Path | None = None,
    ) -> None:
        self.repository = Path(repository).resolve()
        self.contract = contract
        self.run_root = Path(run_root).resolve()
        self.receipt_dir = self.run_root / "swe-milestone-verifier"
        self.receipt_dir.mkdir(parents=True, exist_ok=True)
        self.scratch_root = (
            Path(scratch_root).resolve()
            if scratch_root is not None
            else self._default_scratch_root(self.run_root)
        )
        self.backends = backends_for_contract(contract)
        for backend in self.backends:
            # Backends that need the live checkout (compiled Python
            # extensions) learn where it is; others ignore the attribute.
            setattr(backend, "repository", self.repository)
        self._target_tag: str | None = None
        self._attempt = max(
            (
                int(path.stem.removeprefix("attempt-"))
                for path in self.receipt_dir.glob("attempt-*.json")
                if path.stem.removeprefix("attempt-").isdigit()
            ),
            default=0,
        )
        self._consecutive_timeouts = 0
        self._initial_head = self._record_initial_head()
        self._tree_lock = threading.Lock()
        self._warmup: threading.Thread | None = self._start_warmup()

    @staticmethod
    def _default_scratch_root(run_root: Path) -> Path:
        """Worktrees and build caches live beside the run root, not inside it.

        The run root is archived whole on every process exit
        (``export_snapshot``); persistent checkouts and cargo target
        directories would make that archive tens of gigabytes.
        """

        sibling = run_root.parent / f"{run_root.name}-verifier-scratch"
        try:
            sibling.mkdir(parents=True, exist_ok=True)
            return sibling
        except OSError:
            return run_root / "swe-milestone-verifier" / "worktrees"

    # ---------------------------------------------------------------- warm-up
    def _shared_links(self) -> tuple[str, ...]:
        return tuple(dict.fromkeys(link for backend in self.backends for link in backend.shared_links))

    def _start_warmup(self) -> threading.Thread | None:
        """Build the baseline tree in the background for backends that ask for it.

        A cold cargo target compiles the whole workspace inside the first
        guard's budget and times out.  The warm-up creates both worktrees
        (so their files predate the build artifacts) and builds the baseline
        while the model is still reading the repository.
        """

        if os.environ.get("HOMY_SWE_MILESTONE_VERIFIER_NO_WARMUP") == "1":
            return None
        if not self._initial_head:
            return None
        warmers = [backend for backend in self.backends if callable(getattr(backend, "warm_baseline", None))]
        if not warmers or shutil.which("cargo") is None:
            return None
        thread = threading.Thread(target=self._warmup_body, args=(warmers,), name="swe-milestone-warmup", daemon=True)
        thread.start()
        return thread

    def _warmup_body(self, warmers: list[Any]) -> None:
        started = time.monotonic()
        receipt: dict[str, Any] = {"schema": "homy/swe-milestone-verifier-warmup@1", "baseline": self._initial_head}
        try:
            shared = self._shared_links()
            with self._tree_lock:
                BaselineWorktree(self.repository, self._initial_head, self.scratch_root, label="verify").ensure(shared)
                baseline_root = BaselineWorktree(
                    self.repository, self._initial_head, self.scratch_root, label="baseline"
                ).ensure(shared)
            for backend in warmers:
                timeout = float(getattr(backend, "warmup_timeout_seconds", 3600.0))
                result = backend.warm_baseline(baseline_root, timeout)
                receipt[backend.name] = {
                    "returncode": result.returncode,
                    "timed_out": result.timed_out,
                    "duration_ms": round(result.duration_ms, 3),
                    "tail": result.tail[-1500:],
                }
        except Exception as exc:  # noqa: BLE001 - background helper must never raise
            receipt["error"] = f"{type(exc).__name__}: {exc}"
        receipt["duration_ms"] = round((time.monotonic() - started) * 1000.0, 3)
        try:
            (self.receipt_dir / "warmup.json").write_text(json.dumps(receipt, ensure_ascii=False, sort_keys=True))
        except OSError:
            pass

    def _warmup_alive(self) -> bool:
        return self._warmup is not None and self._warmup.is_alive()

    def _join_warmup(self, timeout: float) -> None:
        if self._warmup is not None and self._warmup.is_alive() and timeout > 0:
            self._warmup.join(timeout)

    # ------------------------------------------------------------------ setup
    def _record_initial_head(self) -> str:
        marker = self.receipt_dir / "initial-head.txt"
        try:
            value = marker.read_text(encoding="ascii").strip()
            if re.fullmatch(r"[0-9a-f]{40}", value):
                return value
        except OSError:
            pass
        result = run_command(["git", "rev-parse", "HEAD"], cwd=self.repository, timeout=60)
        head = result.stdout.strip()
        if result.returncode == 0 and re.fullmatch(r"[0-9a-f]{40}", head):
            marker.write_text(head + "\n", encoding="ascii")
            return head
        return ""

    def bind_submission(self, tag: str | None) -> None:
        """Pin the next verification to an existing submission tag (or clear)."""

        if tag is not None and not tag.startswith(self.contract.submission_tag_prefix):
            raise ValueError(f"not a submission tag: {tag}")
        self._target_tag = tag

    @property
    def attempt(self) -> int:
        return self._attempt

    # --------------------------------------------------------------- protocol
    @_record_unavailable_attempt
    def __call__(self) -> Mapping[str, object]:
        self._attempt += 1
        attempt = self._attempt
        target_tag = self._target_tag
        self._target_tag = None
        started = time.monotonic()
        deadline = started + float(self.contract.total_timeout_seconds)

        def remaining() -> float:
            return max(5.0, deadline - time.monotonic())

        def unit_budget() -> float:
            return min(float(self.contract.unit_timeout_seconds), remaining())

        try:
            # The scored tree is baseline(run start) + every submitted path
            # since then; the official evaluator judges PASS_TO_PASS against
            # the start of the repository task, never against the previous
            # submission.  The incremental scope (since the previous tag) only
            # orders which units run first inside the budget.
            scope = resolve_scope(
                self.repository,
                self.contract,
                target_tag=target_tag,
                fallback_baseline=self._initial_head or None,
                baseline_override=self._initial_head or None,
            )
            incremental = resolve_scope(
                self.repository,
                self.contract,
                target_tag=target_tag,
                fallback_baseline=self._initial_head or None,
            )
        except GitScopeError as exc:
            raise RuntimeError(f"submission scope unavailable: {exc}") from exc
        if incremental.baseline_revision == scope.baseline_revision:
            incremental = scope
        historical_regressions = self._historical_regressions()

        guards: list[GuardFailure] = []
        if scope.out_of_scope:
            guards.append(
                GuardFailure(
                    "out_of_scope_paths",
                    "these changed files are outside repo_src_dirs/root manifests and will NOT be "
                    "submitted; the official evaluator runs without them, so any behaviour that "
                    "depends on them is lost: " + ", ".join(scope.out_of_scope[:20]),
                )
            )
        degraded = self._consecutive_timeouts >= 2
        units: list[str] = []
        units_by_backend: dict[str, list[str]] = {}
        outcomes: list[UnitOutcome] = []
        baseline_outcomes: dict[str, str] = {}
        regressions: list[str] = []
        pre_existing: list[str] = []
        new_unit_failures: list[str] = []
        environment_units: list[str] = []
        agent_environment_units: list[str] = []
        retag_regressions: list[str] = []
        previous_tag_passes_by_backend = self._previous_tag_passes_by_backend(target_tag)
        previous_tag_passes = set().union(*previous_tag_passes_by_backend.values())
        baseline_guard_failures: list[GuardFailure] = []
        pre_existing_guard_failures: list[GuardFailure] = []
        timed_out = False
        scope_label = "AFFECTED_SCOPE_REGRESSION"
        shared = self._shared_links()
        # The verification tree is what the official evaluator sees: the
        # baseline checkout (its own tests, its own tooling files) plus exactly
        # the submitted source paths and root manifests.  Locally edited or
        # added tests, snapshots and out-of-scope files never enter it.
        # Both checkouts persist between attempts so mtime-keyed build caches
        # stay warm; ``ensure`` resets everything except the submitted paths,
        # which the overlay rewrites only when their content changed.
        verify_tree = BaselineWorktree(
            self.repository, scope.baseline_revision, self.scratch_root, label="verify"
        )
        baseline_tree = BaselineWorktree(
            self.repository, scope.baseline_revision, self.scratch_root, label="baseline"
        )

        def build_budget(backend: Any, root: Path) -> float:
            cold = getattr(backend, "cold", None)
            if callable(cold) and cold(root):
                extra = float(getattr(backend, "cold_build_timeout_seconds", 0.0))
                return min(remaining(), max(unit_budget(), extra))
            return unit_budget()

        # A background warm-up may still be compiling the baseline; wait for
        # it while enough budget remains for the guards themselves.
        self._join_warmup(max(0.0, remaining() - 900.0))
        with self._tree_lock:
            verify_root = verify_tree.ensure(shared, keep_paths=scope.submitted_paths)
            baseline_root = baseline_tree.ensure(shared)
        overlay_submitted_paths(self.repository, verify_root, scope)

        # Calibrate deterministic build/typecheck guards against the
        # untouched baseline. Existing repository defects must be
        # reported, but they must not turn every later milestone into a
        # false regression.
        for backend in self.backends:
            baseline_guard_failures.extend(
                backend.manifest_guard(baseline_root, scope, unit_budget())
            )
        if not baseline_guard_failures:
            for backend in self.backends:
                baseline_guard_failures.extend(
                    backend.build_guard(baseline_root, scope, build_budget(backend, baseline_root))
                )
        if not self._warmup_alive():
            for backend in self.backends:
                seed = getattr(backend, "seed_target", None)
                if callable(seed):
                    seed(baseline_root, verify_root, scope.submitted_paths)

        target_guard_failures: list[GuardFailure] = []
        for backend in self.backends:
            target_guard_failures.extend(
                backend.manifest_guard(verify_root, scope, unit_budget())
            )
        if not target_guard_failures and not guards:
            for backend in self.backends:
                target_guard_failures.extend(
                    backend.build_guard(verify_root, scope, build_budget(backend, verify_root))
                )
        new_target_guards, pre_existing_guard_failures = _filter_preexisting_guards(
            target_guard_failures, baseline_guard_failures
        )
        guards.extend(new_target_guards)
        if not guards and scope.target_tag:
            guards.extend(self._end_tree_compile(scope, shared, unit_budget))
        if not guards and not degraded:
            try:
                units, outcomes, baseline_outcomes, units_by_backend = self._run_tests(
                    scope, verify_root, baseline_tree, shared, unit_budget, remaining,
                    incremental=incremental,
                    historical_regressions=historical_regressions,
                    previous_tag_passes=previous_tag_passes_by_backend,
                )
            except VerifierTimeout:
                timed_out = True
            if not timed_out:
                for item in outcomes:
                    if item.outcome in _NON_REGRESSION_OUTCOMES:
                        continue
                    baseline = baseline_outcomes.get(item.unit)
                    if item.outcome == ENVIRONMENT_OUTCOME:
                        # The offline cache lacks an artifact.  When the
                        # untouched baseline cannot run either, the
                        # environment is at fault and no verdict exists;
                        # when the baseline runs, the submission added
                        # the unresolvable dependency.
                        if baseline == ENVIRONMENT_OUTCOME:
                            environment_units.append(item.unit)
                        else:
                            agent_environment_units.append(item.unit)
                        continue
                    if baseline is not None and baseline in _NON_REGRESSION_OUTCOMES:
                        regressions.append(item.unit)
                    elif item.unit in previous_tag_passes:
                        # The earlier tree tagged for this same official
                        # ID ran this unit green.  ``git tag -f`` replaces
                        # that submission, so a unit that now fails is a
                        # regression against what the evaluator already
                        # had, whatever the run's initial baseline did
                        # (element maintenance_ui_ux 27/29 -> 0/29, go-zero
                        # M009 9/9 -> 7/9 in earlier runs.
                        retag_regressions.append(item.unit)
                    elif baseline is None:
                        new_unit_failures.append(item.unit)
                    else:
                        pre_existing.append(item.unit)
                regressions.extend(retag_regressions)
        elif degraded:
            scope_label = "GUARDS_ONLY_AFTER_REPEATED_TIMEOUT"
        if agent_environment_units:
            outputs = {item.unit: item.output for item in outcomes}
            guards.append(
                GuardFailure(
                    "maven_offline_resolution_failed",
                    "these units cannot run offline with your changes although the baseline runs "
                    "them: a dependency you added or changed is absent from the evaluator's offline "
                    "repository. Remove or revert it; the official evaluator has no network: "
                    + ", ".join(agent_environment_units[:20]),
                    "",
                    (outputs.get(agent_environment_units[0]) or "")[-1500:],
                )
            )
        environment_unavailable = bool(environment_units) and not guards and not regressions
        if timed_out:
            self._consecutive_timeouts += 1
            scope_label = "VERIFIER_TIMEOUT"
        else:
            self._consecutive_timeouts = 0
            if environment_unavailable:
                # Nothing about the submission was judged; the route is
                # neither held nor cleared by this run.
                scope_label = "VERIFIER_UNAVAILABLE"

        passed = not guards and not regressions and not timed_out and not environment_unavailable
        pass_to_pass: dict[str, str] = {
            item.unit: item.outcome
            for item in outcomes
            if item.outcome != "SKIPPED"
            and item.unit not in pre_existing
            and item.unit not in new_unit_failures
        }
        summary = {
            "schema": "homy/swe-milestone-verifier-receipt@1",
            "attempt": attempt,
            "passed": passed,
            "verification_scope": scope_label,
            "project_id": self.contract.project_id,
            "languages": list(self.contract.languages),
            "target_tag": target_tag,
            "baseline_revision": scope.baseline_revision,
            "baseline_source": scope.baseline_source,
            "incremental_baseline_revision": incremental.baseline_revision,
            "incremental_baseline_source": incremental.baseline_source,
            "incremental_changed_paths": list(incremental.changed_paths[:200]),
            "historical_regressions_rechecked": sorted(
                unit for units in historical_regressions.values() for unit in units
            ),
            "units_by_backend": units_by_backend,
            "target_revision": scope.target_revision,
            "working_tree_dirty": scope.working_tree_dirty,
            "changed_paths": list(scope.changed_paths[:200]),
            "submitted_source": list(scope.in_scope_source[:200]),
            "root_manifests": list(scope.root_manifests),
            "test_only_changes_not_submitted": list(scope.test_only[:100]),
            "out_of_scope_changes": list(scope.out_of_scope[:100]),
            "guard_failures": [asdict(item) for item in guards],
            "baseline_guard_failures": [asdict(item) for item in baseline_guard_failures],
            "pre_existing_guard_failures_ignored": [
                asdict(item) for item in pre_existing_guard_failures
            ],
            "units": units,
            "unit_outcomes": {item.unit: item.outcome for item in outcomes},
            "baseline_outcomes": baseline_outcomes,
            "regressions": regressions,
            "retag_regressions": retag_regressions,
            "previous_tag_passes_rechecked": sorted(previous_tag_passes),
            "pre_existing_failures_ignored": pre_existing,
            "new_unit_failures_not_submitted": new_unit_failures,
            "environment_unavailable_units": environment_units,
            "environment_unavailable": environment_unavailable,
            "timed_out": timed_out,
            "duration_ms": round((time.monotonic() - started) * 1000.0, 3),
        }
        self._write_receipt(attempt, summary, outcomes)
        stdout = json.dumps(self._model_facing_summary(summary), ensure_ascii=False, sort_keys=True)
        stderr = self._failure_text(
            guards,
            regressions,
            outcomes,
            timed_out,
            environment_units=environment_units,
            retag_regressions=retag_regressions,
        )
        return {
            "passed": passed,
            "command": f"{VERIFIER_COMMAND}[{','.join(self.contract.languages)}]",
            "returncode": 0 if passed else 1,
            "stdout": stdout,
            "stderr": stderr,
            "verification_scope": scope_label,
            "fail_to_pass_count": 0,
            "fail_to_pass_outcomes": {},
            "pass_to_pass_count": len(pass_to_pass),
            "pass_to_pass_outcomes": pass_to_pass,
            "regressions": list(regressions),
            "guard_failures": [item.kind for item in guards],
            "target_tag": target_tag,
            "receipt_path": str(self.receipt_dir / f"attempt-{attempt}.json"),
        }

    # ------------------------------------------------------------------ tests
    def _run_tests(
        self,
        scope: SubmissionScope,
        verify_root: Path,
        baseline_tree: BaselineWorktree,
        shared: tuple[str, ...],
        unit_budget,
        remaining,
        *,
        incremental: SubmissionScope | None = None,
        historical_regressions: Mapping[str, tuple[str, ...]] | None = None,
        previous_tag_passes: Mapping[str, set[str]] | None = None,
    ) -> tuple[list[str], list[UnitOutcome], dict[str, str], dict[str, list[str]]]:
        units: list[str] = []
        outcomes: list[UnitOutcome] = []
        per_backend: dict[str, list[str]] = {}
        history = historical_regressions or {}
        earlier_passes = previous_tag_passes or {}
        for backend in self.backends:
            # Priority inside the unit budget: what this submission touched,
            # then units an earlier guard already saw regress (they must stay
            # blocked until they pass again), then units the earlier tree of
            # this same tag ran green (a retag may not lose them), then the
            # accumulated diff.
            ordered: list[str] = []
            if incremental is not None and incremental is not scope:
                ordered.extend(backend.affected_units(verify_root, incremental, unit_budget()))
            ordered.extend(
                unit for unit in history.get(backend.name, ())
                if backend.unit_exists(verify_root, unit)
            )
            ordered.extend(
                unit for unit in sorted(earlier_passes.get(backend.name, ()))
                if backend.unit_exists(verify_root, unit)
            )
            ordered.extend(backend.affected_units(verify_root, scope, unit_budget()))
            found = list(dict.fromkeys(ordered))
            if len(found) > self.contract.max_units:
                found = found[: self.contract.max_units]
            per_backend[backend.name] = found
            units.extend(found)
        if not units:
            return units, outcomes, {}, per_backend
        for backend in self.backends:
            selected = per_backend.get(backend.name, [])
            if not selected:
                continue
            if remaining() <= 10.0:
                raise VerifierTimeout("no time left for affected tests")
            for item in backend.run_units(verify_root, selected, unit_budget()):
                if item.returncode == 124 and item.outcome == "ERROR":
                    raise VerifierTimeout(f"unit timed out: {item.unit}")
                outcomes.append(item)
        failing = [item for item in outcomes if item.outcome not in _NON_REGRESSION_OUTCOMES]
        # A unit that fails only when it shares the machine with its siblings
        # (port or temp-dir races) is not a regression the official evaluator
        # would score against the submission.  Confirm each failure in
        # isolation before it can hold the route; a pass on the isolated rerun
        # replaces the batched outcome.
        if failing:
            confirmed: list[UnitOutcome] = []
            for backend in self.backends:
                selected = set(per_backend.get(backend.name, ()))
                for item in failing:
                    if item.unit not in selected:
                        continue
                    if remaining() <= 10.0:
                        raise VerifierTimeout("no time left to confirm failing units")
                    rerun = backend.run_units(verify_root, [item.unit], unit_budget())
                    verdict = next((r for r in rerun if r.unit == item.unit), None)
                    if verdict is not None and verdict.returncode == 124 and verdict.outcome == "ERROR":
                        raise VerifierTimeout(f"unit timed out: {item.unit}")
                    if verdict is not None and verdict.outcome in _NON_REGRESSION_OUTCOMES:
                        outcomes[outcomes.index(item)] = UnitOutcome(
                            item.unit, verdict.outcome, verdict.returncode,
                            item.duration_ms + verdict.duration_ms, "",
                        )
                        continue
                    confirmed.append(verdict or item)
            failing = confirmed
        baseline_outcomes: dict[str, str] = {}
        if failing:
            if remaining() <= 10.0:
                raise VerifierTimeout("no time left for baseline calibration")
            base_path = baseline_tree.ensure(shared)
            for backend in self.backends:
                wanted = [
                    item.unit
                    for item in failing
                    if item.unit in per_backend.get(backend.name, ())
                    and backend.unit_exists(base_path, item.unit)
                ]
                if not wanted:
                    continue
                for item in backend.run_units(base_path, wanted, unit_budget()):
                    if item.returncode == 124 and item.outcome == "ERROR":
                        raise VerifierTimeout(f"baseline unit timed out: {item.unit}")
                    baseline_outcomes[item.unit] = item.outcome
        # Units that passed now need no calibration; an absent key means
        # "not calibrated", never "passed at baseline".
        return units, outcomes, baseline_outcomes, per_backend

    def _previous_tag_passes(self, target_tag: str | None) -> set[str]:
        """Units earlier receipts for this same tag ran PASSED.

        Only receipts pinned to the same ``target_tag`` count: a working-tree
        verification or another official ID says nothing about what the
        evaluator already holds for this submission.
        """

        return set().union(*self._previous_tag_passes_by_backend(target_tag).values())

    def _previous_tag_passes_by_backend(self, target_tag: str | None) -> dict[str, set[str]]:
        if not target_tag:
            return {}
        passes: dict[str, set[str]] = {}
        for path in sorted(self.receipt_dir.glob("attempt-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, Mapping) or data.get("target_tag") != target_tag:
                continue
            if data.get("timed_out") or data.get("environment_unavailable"):
                continue
            outcomes = data.get("unit_outcomes")
            by_backend = data.get("units_by_backend")
            if not isinstance(outcomes, Mapping) or not isinstance(by_backend, Mapping):
                continue
            for backend_name, units in by_backend.items():
                for unit in units or ():
                    if outcomes.get(str(unit)) == _PASSED:
                        passes.setdefault(str(backend_name), set()).add(str(unit))
        return passes

    def _historical_regressions(self) -> dict[str, tuple[str, ...]]:
        """Units earlier receipts recorded as regressions, per backend.

        The official evaluator re-scores every earlier submission tree as it
        stands, so a unit that regressed under milestone N stays a scored
        regression until it passes again; re-checking it on every later guard
        keeps the runtime's verdict aligned with that.
        """

        found: dict[str, list[str]] = {}
        for path in sorted(self.receipt_dir.glob("attempt-*.json")):
            try:
                data = json.loads(path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                continue
            if not isinstance(data, Mapping):
                continue
            regressions = set(map(str, data.get("regressions") or ()))
            if not regressions:
                continue
            by_backend = data.get("units_by_backend")
            if not isinstance(by_backend, Mapping):
                continue
            for backend_name, units in by_backend.items():
                for unit in units or ():
                    if str(unit) in regressions:
                        found.setdefault(str(backend_name), []).append(str(unit))
        return {name: tuple(dict.fromkeys(units)) for name, units in found.items()}

    # --------------------------------------------------------------- receipts
    def _write_receipt(self, attempt: int, summary: dict[str, Any], outcomes: list[UnitOutcome]) -> None:
        path = self.receipt_dir / f"attempt-{attempt}.json"
        payload = dict(summary)
        payload["unit_output_tails"] = {
            item.unit: item.output[-2000:] for item in outcomes if item.output and item.outcome != _PASSED
        }
        temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}")
        temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)

    @staticmethod
    def _model_facing_summary(summary: Mapping[str, Any]) -> dict[str, Any]:
        return {
            "verifier": VERIFIER_COMMAND,
            "passed": summary["passed"],
            "verification_scope": summary["verification_scope"],
            "baseline": summary["baseline_revision"][:12],
            "baseline_source": summary["baseline_source"],
            "target_tag": summary["target_tag"],
            "changed_source_files": len(summary["submitted_source"]),
            "root_manifests_changed": summary["root_manifests"],
            "test_only_changes_not_submitted": summary["test_only_changes_not_submitted"][:20],
            "out_of_scope_changes": summary["out_of_scope_changes"][:20],
            "units_run": len(summary["units"]),
            "regressions": summary["regressions"][:30],
            "pre_existing_failures_ignored": summary["pre_existing_failures_ignored"][:30],
            "new_unit_failures_not_submitted": summary["new_unit_failures_not_submitted"][:30],
            "guard_failures": [item["kind"] for item in summary["guard_failures"]],
        }

    def _host_end_compile(self, scope: SubmissionScope, milestone_id: str, budget) -> list[GuardFailure]:
        """Ask the host to compile on the eval image when the END tag is gone.

        The official harness deletes milestone tags and prunes those commits
        from the agent repository.  The host still has the eval image.  This
        writes a request the host fulfills outside the agent container, then
        waits for a redacted result.  No END tree is mounted here.
        """

        directory = Path(os.environ.get("HOMY_SM_END_COMPILE_DIR", "/e2e_workspace/homy-v2-state"))
        try:
            directory.mkdir(parents=True, exist_ok=True)
        except OSError:
            return []
        request_id = f"{scope.target_tag}:{scope.target_revision[:12]}"
        request = {
            "id": request_id,
            "project_id": self.contract.project_id,
            "milestone_id": milestone_id,
            "languages": list(self.contract.languages),
            "repo_src_dirs": list(self.contract.repo_src_dirs),
            "test_dirs": list(self.contract.test_dirs),
            "exclude_patterns": list(self.contract.exclude_patterns),
            "submitted_paths": list(scope.submitted_paths),
            "commit": scope.target_revision,
        }
        (directory / "end-compile-request.json").write_text(json.dumps(request), encoding="utf-8")
        deadline = time.monotonic() + max(1.0, min(float(budget()), 1500.0))
        result_path = directory / "end-compile-result.json"
        while time.monotonic() < deadline:
            try:
                payload = json.loads(result_path.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                time.sleep(1.0)
                continue
            if not isinstance(payload, dict) or payload.get("id") != request_id:
                time.sleep(1.0)
                continue
            if payload.get("skipped") or payload.get("ok") is True:
                return []
            return [
                GuardFailure(
                    kind="end_tree_compile_failed",
                    message=(
                        "submitted sources do not compile on the official eval tree; "
                        "the tag is not left for official scoring"
                    ),
                    command=str(payload.get("command") or ""),
                    output=str(payload.get("output") or ""),
                )
            ]
        return []

    def _end_tree_compile(self, scope: SubmissionScope, shared: tuple[str, ...], budget) -> list[GuardFailure]:
        """Production-compile submitted sources on the official END tree.

        The official evaluator checks out ``milestone-<id>-end`` and overlays
        the submitted sources.  A tag that does not compile there is not left
        for the official runner.  Missing END tags skip this step.  The text
        returned to the model names only submitted files.
        """

        tag = scope.target_tag or ""
        prefix = self.contract.submission_tag_prefix
        if not tag.startswith(prefix):
            return []
        end_tag = f"milestone-{tag[len(prefix):]}-end"
        probe = subprocess.run(
            ["git", "-C", str(self.repository), "rev-parse", "--verify", f"refs/tags/{end_tag}"],
            capture_output=True,
            text=True,
            check=False,
            timeout=30,
        )
        if probe.returncode != 0:
            return self._host_end_compile(scope, tag[len(prefix):], budget)
        revision = probe.stdout.strip()
        end_tree = BaselineWorktree(self.repository, revision, self.scratch_root, label="end")
        with self._tree_lock:
            root = end_tree.ensure(shared, keep_paths=scope.submitted_paths)
        overlay_submitted_paths(self.repository, root, scope)
        failures: list[GuardFailure] = []
        for backend in self.backends:
            failures.extend(backend.build_guard(root, scope, budget()))
        if not failures:
            return []
        raw = "\n".join(item.output for item in failures if item.output)
        return [
            GuardFailure(
                kind="end_tree_compile_failed",
                message=(
                    "submitted sources do not compile on the official eval tree; "
                    "the tag is not left for official scoring"
                ),
                command=failures[0].command,
                output=redact_end_compile_output(raw, scope.submitted_paths),
            )
        ]

    def _failure_text(
        self,
        guards: list[GuardFailure],
        regressions: list[str],
        outcomes: list[UnitOutcome],
        timed_out: bool,
        *,
        environment_units: Sequence[str] = (),
        retag_regressions: Sequence[str] = (),
    ) -> str:
        lines: list[str] = []
        if retag_regressions:
            lines.append(
                "[retag-regression] these units passed in the tree you tagged earlier for the SAME "
                "official ID and fail in the re-tagged tree; the official evaluator keeps only the "
                "latest tag, so this submission would score worse than the one it replaces. The "
                "runtime moves the tag back to the earlier tree; restore the earlier behaviour at "
                "HEAD (git diff <previous tag commit> -- <files>) before moving the tag again: "
                + ", ".join(list(retag_regressions)[:30])
            )
        if environment_units and not guards and not regressions:
            lines.append(
                "[environment] the offline dependency cache lacks an artifact the untouched "
                "baseline already needs, so no test verdict exists for: "
                + ", ".join(list(environment_units)[:20])
                + ". This is not caused by your change; do not try to fabricate or copy jars."
            )
        for guard in guards:
            lines.append(f"[{guard.kind}] {guard.message}")
            if guard.command:
                lines.append(f"  command: {guard.command}")
            if guard.output:
                lines.append("  " + guard.output.strip()[-1500:].replace("\n", "\n  "))
        if regressions:
            lines.append(
                "[regression] these units pass at the baseline revision but fail with your changes "
                "(official PASS_TO_PASS would fail): " + ", ".join(regressions[:30])
            )
            outputs = {item.unit: item.output for item in outcomes}
            for unit in regressions[:6]:
                tail = (outputs.get(unit) or "").strip()
                if tail:
                    lines.append(f"  --- {unit} ---")
                    lines.append("  " + tail[-1200:].replace("\n", "\n  "))
        if timed_out:
            lines.append(
                "[timeout] the verifier budget ran out before the affected tests finished; "
                "narrow your change or run the affected test units yourself and end the Turn."
            )
        return "\n".join(lines)
