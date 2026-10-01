"""Runtime-owned Milestone contract freeze.

A future Milestone is planned as a set of semantic *directions*: what the
repository must do, without execution details.  When that Milestone becomes the
route target the runtime compiles the directions into an executable acceptance
contract and resolves natural addresses for every requirement, so the acceptance
kernel can bind real test/verifier results deterministically instead of asking
the model to transcribe Criterion IDs.

Address resolution follows a fixed priority chain and records its provenance:

1. addresses the model already declared on the direction;
2. addresses learned from accepted predecessor evidence and the current
   Milestone-scoped repository changes;
3. structural candidates supplied by the Repository State Graph (hint only);
4. addresses attached to the immutable Task requirement checklist.

A requirement that still has no address keeps its executable mode: it binds to
Milestone-scoped execution results during the address-learning window and can
never become an indefinitely blocking gate on its own.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace

from ..contracts import (
    TASK_FINAL_REQUIREMENT_ID,
    CommitmentLevel,
    CompletionCriterionSpec,
    CriterionVerificationMode,
    MilestoneSpec,
    PlanSpec,
    PlanStepSpec,
)

RichAddressResolver = Callable[[Sequence[str], Sequence[str]], Sequence[str]]
"""Return structural candidates for (requirement tokens, seed entity refs)."""

_STOP_WORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "with",
        "that",
        "this",
        "from",
        "into",
        "when",
        "then",
        "than",
        "must",
        "should",
        "shall",
        "not",
        "are",
        "was",
        "were",
        "has",
        "have",
        "does",
        "each",
        "every",
        "any",
        "all",
        "also",
        "only",
        "its",
        "their",
        "current",
        "existing",
        "code",
        "file",
        "files",
        "test",
        "tests",
        "repository",
        "module",
        "function",
        "class",
        "method",
        "behavior",
        "behaviour",
        "return",
        "returns",
        "value",
        "values",
        "use",
        "uses",
        "using",
        "new",
        "old",
        "same",
        "without",
        "within",
        "after",
        "before",
        "still",
        "correct",
        "correctly",
        "properly",
        "pass",
        "passes",
        "passing",
        "run",
        "runs",
        "running",
        "result",
        "results",
        "error",
        "errors",
        "raise",
        "raises",
        "raised",
        "instead",
        "case",
        "cases",
        "input",
        "output",
        "data",
        "object",
        "objects",
        "type",
        "types",
        "string",
        "list",
        "dict",
        "none",
        "true",
        "false",
        "self",
        "src",
        "lib",
        "main",
        "init",
        "py",
    }
)

_TOKEN = re.compile(r"[A-Za-z][A-Za-z0-9]+|\d+")
_CAMEL = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")


def semantic_address_tokens(text: str) -> frozenset[str]:
    """Split prose or identifiers into lower-cased address tokens.

    ``CardHeaderFormatter`` and ``card_header_formatter`` both contribute
    ``card``, ``header`` and ``formatter``.  Stop words and very short tokens
    are dropped so ordinary English never produces spurious address matches.
    """

    tokens: set[str] = set()
    for raw in _TOKEN.findall(text.replace("_", " ")):
        for part in _CAMEL.split(raw):
            lowered = part.casefold()
            if len(lowered) < 3 or lowered in _STOP_WORDS or lowered.isdigit():
                continue
            tokens.add(lowered)
    return frozenset(tokens)


def entity_address_tokens(entity_ref: str) -> frozenset[str]:
    """Return the tokens that name one file/symbol/test address."""

    normalized = entity_ref.strip().replace("\\", "/")
    for prefix in ("file:", "symbol:", "test:", "dir:"):
        if normalized.startswith(prefix):
            normalized = normalized[len(prefix) :]
            break
    if not normalized:
        return frozenset()
    if "::" in normalized:
        path, _, symbol = normalized.partition("::")
    elif ":" in normalized and "/" in normalized.split(":", 1)[0]:
        path, _, symbol = normalized.partition(":")
    else:
        path, symbol = normalized, ""
    tokens: set[str] = set()
    basename = path.rsplit("/", 1)[-1]
    stem = basename.split(".", 1)[0]
    if stem:
        tokens.update(semantic_address_tokens(stem))
    parent = path.rsplit("/", 2)
    if len(parent) >= 2:
        tokens.update(semantic_address_tokens(parent[-2]))
    if symbol:
        for piece in symbol.replace(".", " ").replace("::", " ").split():
            tokens.update(semantic_address_tokens(piece))
    return frozenset(tokens)


def _is_test_address(entity_ref: str) -> bool:
    lowered = entity_ref.casefold()
    return (
        lowered.startswith("test:")
        or "/tests/" in lowered
        or "/test/" in lowered
        or lowered.rsplit("/", 1)[-1].startswith("test_")
        or lowered.endswith("_test.py")
    )


@dataclass(frozen=True, slots=True)
class AddressHints:
    """Candidate addresses gathered by the runtime before a freeze."""

    predecessor_entities: tuple[str, ...] = ()
    scoped_change_entities: tuple[str, ...] = ()
    requirement_entities: tuple[str, ...] = ()
    rich_resolver: RichAddressResolver | None = None
    max_addresses: int = 6


@dataclass(frozen=True, slots=True)
class ResolvedCriterionAddress:
    criterion_id: str
    entity_refs: tuple[str, ...]
    source: str
    candidates_considered: int
    mode: CriterionVerificationMode


@dataclass(frozen=True, slots=True)
class FrozenContract:
    plan: PlanSpec
    milestone: MilestoneSpec
    resolutions: tuple[ResolvedCriterionAddress, ...] = field(default_factory=tuple)

    def address_resolution(self) -> dict[str, object]:
        return {
            "milestone_id": self.milestone.canonical_id,
            "criteria": [
                {
                    "criterion_id": item.criterion_id,
                    "entity_refs": list(item.entity_refs),
                    "source": item.source,
                    "candidates_considered": item.candidates_considered,
                    "verification_mode": item.mode.value,
                }
                for item in self.resolutions
            ],
        }


def _rank_candidates(
    requirement_tokens: frozenset[str],
    candidates: Iterable[str],
    *,
    limit: int,
) -> tuple[tuple[str, ...], int]:
    scored: list[tuple[int, int, str]] = []
    considered = 0
    for ordinal, candidate in enumerate(dict.fromkeys(item.strip() for item in candidates)):
        if not candidate:
            continue
        considered += 1
        overlap = len(requirement_tokens.intersection(entity_address_tokens(candidate)))
        if overlap <= 0:
            continue
        # Prefer implementation addresses over test addresses for the same
        # requirement: a test only observes the behaviour, the code holds it.
        penalty = 1 if _is_test_address(candidate) else 0
        scored.append((-overlap, penalty * 1000 + ordinal, candidate))
    scored.sort()
    return tuple(item[2] for item in scored[:limit]), considered


def resolve_criterion_addresses(
    criterion: CompletionCriterionSpec,
    hints: AddressHints,
) -> ResolvedCriterionAddress:
    """Resolve natural addresses for one compiled requirement."""

    mode = criterion.verification_mode or CriterionVerificationMode.EXECUTABLE
    if criterion.entity_refs:
        return ResolvedCriterionAddress(
            criterion_id=criterion.criterion_id,
            entity_refs=tuple(criterion.entity_refs),
            source="EXPLICIT",
            candidates_considered=0,
            mode=mode,
        )
    if criterion.requirement_id == TASK_FINAL_REQUIREMENT_ID or mode is (
        CriterionVerificationMode.SEMANTIC
    ):
        return ResolvedCriterionAddress(
            criterion_id=criterion.criterion_id,
            entity_refs=(),
            source="SEMANTIC_NO_ADDRESS",
            candidates_considered=0,
            mode=CriterionVerificationMode.SEMANTIC,
        )
    tokens = semantic_address_tokens(
        " ".join((criterion.requirement_text, criterion.observable_outcome))
    )
    if not tokens:
        return ResolvedCriterionAddress(
            criterion_id=criterion.criterion_id,
            entity_refs=(),
            source="ADDRESS_PENDING",
            candidates_considered=0,
            mode=mode,
        )
    total_considered = 0
    for source, candidates in (
        ("PREDECESSOR_EVIDENCE", (*hints.scoped_change_entities, *hints.predecessor_entities)),
    ):
        selected, considered = _rank_candidates(tokens, candidates, limit=hints.max_addresses)
        total_considered += considered
        if selected:
            return ResolvedCriterionAddress(
                criterion_id=criterion.criterion_id,
                entity_refs=selected,
                source=source,
                candidates_considered=total_considered,
                mode=mode,
            )
    if hints.rich_resolver is not None:
        seeds = tuple(dict.fromkeys((*hints.scoped_change_entities, *hints.predecessor_entities)))
        try:
            rich_candidates = tuple(hints.rich_resolver(tuple(sorted(tokens)), seeds))
        except Exception:  # pragma: no cover - a hint source must never block the route
            rich_candidates = ()
        selected, considered = _rank_candidates(
            tokens,
            rich_candidates,
            limit=hints.max_addresses,
        )
        total_considered += considered
        if selected:
            return ResolvedCriterionAddress(
                criterion_id=criterion.criterion_id,
                entity_refs=selected,
                source="RICH_GRAPH",
                candidates_considered=total_considered,
                mode=mode,
            )
    selected, considered = _rank_candidates(
        tokens,
        hints.requirement_entities,
        limit=hints.max_addresses,
    )
    total_considered += considered
    if selected:
        return ResolvedCriterionAddress(
            criterion_id=criterion.criterion_id,
            entity_refs=selected,
            source="REQUIREMENT_CHECKLIST",
            candidates_considered=total_considered,
            mode=mode,
        )
    return ResolvedCriterionAddress(
        criterion_id=criterion.criterion_id,
        entity_refs=(),
        source="ADDRESS_PENDING",
        candidates_considered=total_considered,
        mode=mode,
    )


def freeze_milestone_addresses(
    plan: PlanSpec,
    *,
    canonical_id: str,
    hints: AddressHints,
    prior_steps: Sequence[Mapping[str, object]] = (),
) -> FrozenContract:
    """Attach resolved addresses to an already materialized Milestone.

    ``plan`` must already carry the compiled (MILESTONE-level) criteria for
    ``canonical_id``; this pass only fills missing addresses, aligns the
    navigation Step with them and keeps every other Milestone untouched.
    """

    index = next(
        (
            ordinal
            for ordinal, item in enumerate(plan.milestones)
            if item.canonical_id == canonical_id
        ),
        None,
    )
    if index is None:
        raise KeyError(canonical_id)
    milestone = plan.milestones[index]
    if any(item.commitment_level is CommitmentLevel.DIRECTION for item in milestone.criteria):
        raise ValueError("address freeze requires compiled MILESTONE-level criteria")
    resolutions: list[ResolvedCriterionAddress] = []
    criteria: list[CompletionCriterionSpec] = []
    for criterion in milestone.criteria:
        resolution = resolve_criterion_addresses(criterion, hints)
        resolutions.append(resolution)
        criteria.append(
            replace(
                criterion,
                entity_refs=resolution.entity_refs,
                verification_mode=resolution.mode,
                required_evidence_types=(
                    ()
                    if resolution.mode is CriterionVerificationMode.SEMANTIC
                    else criterion.required_evidence_types
                ),
            )
        )
    entity_refs = tuple(
        dict.fromkeys(entity for item in criteria for entity in item.entity_refs)
    )
    steps = tuple(
        _align_navigation_step(step, criteria=criteria, entity_refs=entity_refs)
        for step in milestone.steps
    )
    steps = _reconcile_step_addresses(
        steps,
        canonical_id=canonical_id,
        prior_steps=prior_steps,
    )
    frozen_milestone = replace(
        milestone,
        criteria=tuple(criteria),
        completion_criteria=tuple(item.observable_outcome for item in criteria),
        entity_refs=entity_refs,
        steps=steps,
    )
    milestones = list(plan.milestones)
    milestones[index] = frozen_milestone
    frozen_plan = PlanSpec(
        goal=plan.goal,
        milestones=tuple(milestones),
        final_verification=plan.final_verification,
        final_acceptance=plan.final_acceptance,
        native_plan=plan.native_plan,
    )
    return FrozenContract(
        plan=frozen_plan,
        milestone=frozen_milestone,
        resolutions=tuple(resolutions),
    )


def _align_navigation_step(
    step: PlanStepSpec,
    *,
    criteria: Sequence[CompletionCriterionSpec],
    entity_refs: tuple[str, ...],
) -> PlanStepSpec:
    if step.corrective or step.minimum_acceptance:
        return step
    criterion_ids = tuple(item.criterion_id for item in criteria)
    return replace(step, criterion_ids=criterion_ids, entity_refs=entity_refs)


def _reconcile_step_addresses(
    steps: tuple[PlanStepSpec, ...],
    *,
    canonical_id: str,
    prior_steps: Sequence[Mapping[str, object]],
) -> tuple[PlanStepSpec, ...]:
    """Never rewrite an allocated Step contract; allocate a fresh suffix instead."""

    if not prior_steps:
        return steps
    prior_by_id = {str(item["step_id"]): item for item in prior_steps}
    suffixes = [
        int(match.group(1))
        for step_id in prior_by_id
        if (match := re.fullmatch(rf"{re.escape(canonical_id)}\.S(\d+)", step_id))
    ]
    next_suffix = max(suffixes, default=0) + 1
    reconciled: list[PlanStepSpec] = []
    for step in steps:
        prior = prior_by_id.get(step.step_id)
        if prior is None:
            reconciled.append(step)
            continue
        same_contract = (
            str(prior.get("expected_outcome", "")) == step.expected_outcome
            and bool(prior.get("minimum_acceptance", ())) == bool(step.minimum_acceptance)
            and tuple(map(str, prior.get("failure_signals", ()))) == tuple(step.failure_signals)
        )
        if same_contract:
            reconciled.append(step)
            continue
        reconciled.append(replace(step, step_id=f"{canonical_id}.S{next_suffix:03d}"))
        next_suffix += 1
    return tuple(reconciled)
