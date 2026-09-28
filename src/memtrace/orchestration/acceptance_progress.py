"""Milestone acceptance loop policy: claim signals, gap digests and budgets.

The execution boundary of a Milestone is governed by three durable keys, not by
Turn counting:

* the *gap* (which required criteria are unmet and which observation each one
  still lacks);
* the normalized digest of the evidence *bound* to the contract;
* the Milestone-scoped repository state.

A boundary that changes none of these produced no progress no matter how many
tool calls the model made.  Budgets below bound how long weak or absent
progress may keep a route open before the runtime records ``ROUTE_STALLED``
instead of materializing yet another Focus.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum

from ..contracts import digest

# Boundary reason the coordinator uses when a physical Epoch fence closes a
# Turn after ``no_progress`` consecutive Epochs left the semantic frontier
# (workspace revision, acceptance state, durable conclusions) unchanged.
PHYSICAL_NO_PROGRESS_BOUNDARY = "PHYSICAL_NO_PROGRESS_BUDGET"
# Boundary reason used when a physical fence closes a Turn after the model
# spent ``exploration_turns`` consecutive Turns of the current Milestone only
# reading (no workspace mutation, no test observation), was told to implement
# now, and still did not act.
EXPLORATION_BUDGET_BOUNDARY = "EXPLORATION_BUDGET_EXHAUSTED"


class ClaimSignal(str, Enum):
    """Why the runtime submitted the current Milestone for acceptance."""

    MODEL_BOUNDARY_REQUEST = "MODEL_BOUNDARY_REQUEST"
    # The model marked every navigation Step of the Milestone completed in
    # its native Plan: that is its own completion claim, stated in protocol.
    NATIVE_PLAN_MILESTONE_COMPLETED = "NATIVE_PLAN_MILESTONE_COMPLETED"
    FACTUAL_CONTRACT_SATISFIED = "FACTUAL_CONTRACT_SATISFIED"
    CRITERION_BOUND_OBSERVATION = "CRITERION_BOUND_OBSERVATION"
    # The runtime itself owns the decisive observation (host-managed
    # verifier); the Turn end is the moment to run it.
    RUNTIME_VERIFIER_READY = "RUNTIME_VERIFIER_READY"
    UNCLAIMED_BOUNDARY_BUDGET = "UNCLAIMED_BOUNDARY_BUDGET"
    # The Milestone already left IN_PROGRESS at an earlier boundary; the
    # acceptance loop below owns it now.
    ALREADY_SUBMITTED = "ALREADY_SUBMITTED"
    NONE = "NONE"


@dataclass(frozen=True, slots=True)
class AcceptanceBudgets:
    """Bounded retry policy for one Milestone acceptance loop."""

    # Consecutive claimed boundaries whose gap/evidence/revision keys changed
    # but the gap did not shrink.
    weak_progress: int = 4
    # Consecutive boundaries with identical acceptance keys.
    no_progress: int = 2
    # Model review rounds granted to semantic criteria before UNVERIFIED.
    semantic_review_rounds: int = 2
    # Natural Turn ends without any claim signal before the runtime submits
    # the Milestone anyway so the model receives exact acceptance diagnostics.
    unclaimed_boundaries: int = 3
    # Consecutive Turns of one Milestone (natural or fenced) that produce no
    # workspace mutation and no test observation before the Route Card carries
    # an IMPLEMENT_NOW directive; one more such Turn ending at a physical fence
    # is then an exhausted exploratory boundary (r5 M004 / r8 M008: 26-minute
    # read-only Turns until the context fence, three Epochs before any push).
    exploration_turns: int = 2

    @classmethod
    def from_mapping(cls, value: Mapping[str, object] | None) -> AcceptanceBudgets:
        if not value:
            return cls()
        base = cls()
        fields: dict[str, int] = {}
        for name in (
            "weak_progress",
            "no_progress",
            "semantic_review_rounds",
            "unclaimed_boundaries",
            "exploration_turns",
        ):
            raw = value.get(name)
            if raw is None:
                fields[name] = getattr(base, name)
                continue
            parsed = int(raw)
            if parsed < 1:
                raise ValueError(f"acceptance budget {name} must be >= 1")
            fields[name] = parsed
        return cls(**fields)


def gap_digest(
    unmet_criterion_ids: Sequence[str],
    missing_evidence_types: Mapping[str, Sequence[str]],
    rejection_reasons: Mapping[str, Sequence[str]] | None = None,
) -> str:
    """Digest the acceptance gap independent of event identity or ordering."""

    return digest(
        sorted(
            (
                str(criterion_id),
                sorted(map(str, missing_evidence_types.get(criterion_id, ()))),
                sorted(map(str, (rejection_reasons or {}).get(criterion_id, ()))),
            )
            for criterion_id in dict.fromkeys(map(str, unmet_criterion_ids))
        )
    )


def focus_frontier_key(
    *,
    milestone_identity_id: str,
    plan_version_id: str,
    focus_kind: str,
    gap: str,
) -> str:
    """Stable Focus identity: one Focus per (Milestone, contract version, gap)."""

    return digest(
        {
            "milestone_identity_id": milestone_identity_id,
            "plan_version_id": plan_version_id,
            "focus_kind": focus_kind,
            "gap_digest": gap,
        }
    )


def decide_claim(
    *,
    boundary_reason: str,
    factual_satisfied: bool,
    assessed_criteria: int,
    unmet_criteria: int,
    has_current_code_change: bool,
    unclaimed_boundaries: int,
    budgets: AcceptanceBudgets,
    runtime_verifier_ready: bool = False,
    native_plan_completed: bool = False,
) -> ClaimSignal:
    """Choose whether a natural Turn end submits the Milestone for acceptance.

    An exploratory Turn that produced no criterion-bound observation is not a
    completion claim; submitting it would only manufacture a Verification Focus
    for work the model never said was finished.  The unclaimed-boundary budget
    keeps that leniency finite.
    """

    if boundary_reason == "MILESTONE_BOUNDARY_REQUESTED":
        return ClaimSignal.MODEL_BOUNDARY_REQUEST
    if native_plan_completed:
        return ClaimSignal.NATIVE_PLAN_MILESTONE_COMPLETED
    if factual_satisfied:
        return ClaimSignal.FACTUAL_CONTRACT_SATISFIED
    if runtime_verifier_ready:
        return ClaimSignal.RUNTIME_VERIFIER_READY
    if assessed_criteria > unmet_criteria and has_current_code_change:
        return ClaimSignal.CRITERION_BOUND_OBSERVATION
    if boundary_reason in {PHYSICAL_NO_PROGRESS_BOUNDARY, EXPLORATION_BUDGET_BOUNDARY}:
        # Consecutive Epochs left the semantic frontier untouched, or the model
        # ignored an IMPLEMENT_NOW directive and filled another window with
        # reads only.  That is an exploratory boundary whose budget is spent.
        return ClaimSignal.UNCLAIMED_BOUNDARY_BUDGET
    if unclaimed_boundaries + 1 >= budgets.unclaimed_boundaries:
        return ClaimSignal.UNCLAIMED_BOUNDARY_BUDGET
    return ClaimSignal.NONE
