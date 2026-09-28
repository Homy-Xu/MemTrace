from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from ..contracts import FactType

_PASSED = "PASSED"
_VALID_OUTCOMES = frozenset({_PASSED, "FAILED", "ERROR", "SKIPPED", "MISSING"})
_RESULT_EVIDENCE_TYPES = frozenset(
    {
        FactType.TOOL_RESULT.value,
        FactType.TEST_RESULT.value,
        FactType.VERIFIER_RESULT.value,
    }
)


def normalize_outcome_mapping(
    value: object,
    *,
    field: str,
) -> dict[str, str] | None:
    """Validate a trusted verifier's exact selector receipt.

    ``None`` means that an older verifier supplied only an aggregate result.
    An empty mapping is different: it is an exact receipt for an empty target
    set (for example, a benchmark instance with no calibrated regressions).
    """

    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise TypeError(f"trusted verifier {field} must be a Mapping")
    normalized: dict[str, str] = {}
    for raw_selector, raw_outcome in value.items():
        selector = str(raw_selector).strip()
        outcome = str(raw_outcome).strip().upper()
        if not selector:
            raise ValueError(f"trusted verifier {field} contains an empty selector")
        if outcome not in _VALID_OUTCOMES:
            raise ValueError(
                f"trusted verifier {field} contains an invalid outcome for {selector}: {outcome}"
            )
        if selector in normalized:
            raise ValueError(f"trusted verifier {field} contains a duplicate selector: {selector}")
        normalized[selector] = outcome
    return normalized


def _evidence_type(value: object) -> str:
    return str(getattr(value, "value", value))


def _selector_family(selector: str) -> str:
    """Return a pytest parameter-family address without guessing across tests."""

    head, separator, test_name = selector.rpartition("::")
    if not separator or "[" not in test_name or not test_name.endswith("]"):
        return selector
    return f"{head}::{test_name.split('[', 1)[0]}"


def _matched_targets(
    selectors: Sequence[str],
    outcomes: Mapping[str, str],
) -> tuple[str, ...]:
    """Match exact selectors and explicit pytest parameter families.

    Family completion is deliberately narrow. It lets a legacy Plan that named
    one parameterized node cover its sibling parameters, while never matching a
    different test function or relying on prose/entity similarity. New Plans
    are validated to enumerate every benchmark target exactly.
    """

    requested = set(selectors)
    requested_families = {_selector_family(selector) for selector in selectors}
    return tuple(
        target
        for target in outcomes
        if target in requested or _selector_family(target) in requested_families
    )


@dataclass(frozen=True, slots=True)
class CriterionVerificationProjection:
    criterion_id: str
    success: bool
    mode: str
    matched_fail_to_pass: tuple[str, ...]
    matched_pass_to_pass: tuple[str, ...]
    nonpassing_targets: tuple[str, ...]

    def as_mapping(self) -> dict[str, object]:
        return {
            "success": self.success,
            "mode": self.mode,
            "matched_fail_to_pass": list(self.matched_fail_to_pass),
            "matched_pass_to_pass": list(self.matched_pass_to_pass),
            "nonpassing_targets": list(self.nonpassing_targets),
        }


@dataclass(frozen=True, slots=True)
class TrustedVerificationProjection:
    passed: bool
    aggregate_passed: bool
    final_global_gate: bool
    regression_passed: bool | None
    criteria: tuple[CriterionVerificationProjection, ...]
    remaining_fail_to_pass: tuple[str, ...]
    regression_failures: tuple[str, ...]
    unrelated_nonpassing_targets: tuple[str, ...]

    @property
    def passed_criterion_ids(self) -> tuple[str, ...]:
        return tuple(item.criterion_id for item in self.criteria if item.success)

    @property
    def failed_criterion_ids(self) -> tuple[str, ...]:
        return tuple(item.criterion_id for item in self.criteria if not item.success)

    def as_mapping(self) -> dict[str, object]:
        return {
            "passed": self.passed,
            "aggregate_passed": self.aggregate_passed,
            "final_global_gate": self.final_global_gate,
            "regression_passed": self.regression_passed,
            "criteria": {
                item.criterion_id: item.as_mapping() for item in self.criteria
            },
            "remaining_fail_to_pass": list(self.remaining_fail_to_pass),
            "regression_failures": list(self.regression_failures),
            "unrelated_nonpassing_targets": list(self.unrelated_nonpassing_targets),
        }


def project_trusted_verification(
    *,
    aggregate_passed: bool,
    fail_to_pass_outcomes: Mapping[str, str] | None,
    pass_to_pass_outcomes: Mapping[str, str] | None,
    pass_to_pass_count: int,
    criteria: Sequence[Mapping[str, object]],
    final_global_gate: bool,
) -> TrustedVerificationProjection:
    """Project one immutable full-suite result onto the active acceptance scope.

    Non-final Milestones consume only their exact benchmark targets plus the
    global regression guard. The final Milestone additionally requires the
    aggregate full-suite result, so local progress can never become a false
    repository-level completion.
    """

    f2p = dict(fail_to_pass_outcomes) if fail_to_pass_outcomes is not None else None
    p2p = dict(pass_to_pass_outcomes) if pass_to_pass_outcomes is not None else None
    if p2p is not None:
        regression_passed: bool | None = all(outcome == _PASSED for outcome in p2p.values())
    elif pass_to_pass_count == 0 or aggregate_passed:
        regression_passed = True
    else:
        regression_passed = None

    regression_failures = tuple(
        selector for selector, outcome in (p2p or {}).items() if outcome != _PASSED
    )
    remaining_f2p = tuple(
        selector for selector, outcome in (f2p or {}).items() if outcome != _PASSED
    )
    criterion_projections: list[CriterionVerificationProjection] = []
    matched_by_active_scope: set[str] = set()
    for criterion in criteria:
        if not bool(criterion.get("required", True)):
            continue
        required_types = {
            _evidence_type(value)
            for value in criterion.get("required_evidence_types", ())
        }
        if not required_types.intersection(_RESULT_EVIDENCE_TYPES):
            continue
        criterion_id = str(criterion.get("criterion_id", "")).strip()
        if not criterion_id:
            continue
        selectors = tuple(map(str, criterion.get("test_selectors", ())))
        matched_f2p = _matched_targets(selectors, f2p or {}) if f2p is not None else ()
        matched_p2p = _matched_targets(selectors, p2p or {}) if p2p is not None else ()
        matched_by_active_scope.update(matched_f2p)
        matched_by_active_scope.update(matched_p2p)
        projectable = f2p is not None and bool(matched_f2p or matched_p2p)
        matched_outcomes = {
            **{selector: f2p[selector] for selector in matched_f2p},
            **{selector: p2p[selector] for selector in matched_p2p},
        }
        nonpassing = tuple(
            selector for selector, outcome in matched_outcomes.items() if outcome != _PASSED
        )
        if projectable:
            success = not nonpassing and regression_passed is True
            mode = "EXACT_TARGETS"
        else:
            # Old/non-benchmark verifier contracts remain readable, but an
            # aggregate failure can never be converted into positive evidence.
            success = aggregate_passed
            mode = "AGGREGATE_FALLBACK"
        if final_global_gate and not aggregate_passed:
            success = False
            mode = f"{mode}_WITH_FINAL_GLOBAL_GATE"
        criterion_projections.append(
            CriterionVerificationProjection(
                criterion_id=criterion_id,
                success=success,
                mode=mode,
                matched_fail_to_pass=matched_f2p,
                matched_pass_to_pass=matched_p2p,
                nonpassing_targets=nonpassing,
            )
        )

    passed = (
        all(item.success for item in criterion_projections)
        if criterion_projections
        else aggregate_passed
    )
    if final_global_gate:
        passed = passed and aggregate_passed
    unrelated = tuple(
        selector
        for selector in remaining_f2p
        if selector not in matched_by_active_scope
    )
    return TrustedVerificationProjection(
        passed=passed,
        aggregate_passed=aggregate_passed,
        final_global_gate=final_global_gate,
        regression_passed=regression_passed,
        criteria=tuple(criterion_projections),
        remaining_fail_to_pass=remaining_f2p,
        regression_failures=regression_failures,
        unrelated_nonpassing_targets=unrelated,
    )
