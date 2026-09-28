from __future__ import annotations

from collections.abc import Mapping, Sequence
import re

from .contracts import (
    ClaimType,
    CommitmentLevel,
    CompletionCriterionSpec,
    CriterionVerificationMode,
    FactType,
    default_verification_mode,
)

# A requirement category constrains which observed facts may support it. This
# is intentionally a small type-safety table, not a verifier/oracle language.
# Tests, repository observations and host receipts remain ordinary Evidence;
# the original requirement and the Milestone-boundary review remain semantic
# authority.
_CLAIM_PRIMARY_EVIDENCE_TYPES: dict[ClaimType, frozenset[FactType]] = {
    ClaimType.STRUCTURAL: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.CODE_OBSERVATION,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.BEHAVIORAL: frozenset({FactType.TEST_RESULT, FactType.VERIFIER_RESULT}),
    ClaimType.EQUIVALENCE: frozenset({FactType.TEST_RESULT, FactType.VERIFIER_RESULT}),
    ClaimType.INVARIANT: frozenset({FactType.TEST_RESULT, FactType.VERIFIER_RESULT}),
    ClaimType.REGRESSION: frozenset({FactType.TEST_RESULT, FactType.VERIFIER_RESULT}),
    ClaimType.PERFORMANCE: frozenset({FactType.VERIFIER_RESULT}),
    # A PROCESS claim is satisfied by the process receipt itself.  Requiring
    # CODE_OBSERVATION here turns a successful command into a second, unrelated
    # repository-audit obligation and reintroduces the Step gate we removed.
    ClaimType.PROCESS: frozenset({FactType.TOOL_RESULT}),
    ClaimType.DECISION: frozenset({FactType.IMPLEMENTATION_DECISION}),
    ClaimType.FAILURE_REPRODUCTION: frozenset({FactType.TEST_FAILURE}),
}

_CLAIM_COMPATIBLE_EVIDENCE_TYPES: dict[ClaimType, frozenset[FactType]] = {
    ClaimType.STRUCTURAL: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.CODE_OBSERVATION,
            FactType.IMPLEMENTATION_DECISION,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.BEHAVIORAL: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.CODE_OBSERVATION,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.EQUIVALENCE: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.CODE_OBSERVATION,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.INVARIANT: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.CODE_OBSERVATION,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.REGRESSION: frozenset(
        {
            FactType.CODE_CHANGE,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
    ),
    ClaimType.PERFORMANCE: frozenset({FactType.VERIFIER_RESULT}),
    ClaimType.PROCESS: frozenset({FactType.TOOL_RESULT}),
    ClaimType.DECISION: frozenset({FactType.IMPLEMENTATION_DECISION}),
    ClaimType.FAILURE_REPRODUCTION: frozenset({FactType.TEST_FAILURE}),
}

_DEFAULT_EVIDENCE_TYPE: dict[ClaimType, FactType] = {
    ClaimType.STRUCTURAL: FactType.CODE_CHANGE,
    ClaimType.BEHAVIORAL: FactType.TEST_RESULT,
    ClaimType.EQUIVALENCE: FactType.TEST_RESULT,
    ClaimType.INVARIANT: FactType.TEST_RESULT,
    ClaimType.REGRESSION: FactType.TEST_RESULT,
    ClaimType.PERFORMANCE: FactType.VERIFIER_RESULT,
    ClaimType.PROCESS: FactType.TOOL_RESULT,
    ClaimType.DECISION: FactType.IMPLEMENTATION_DECISION,
    ClaimType.FAILURE_REPRODUCTION: FactType.TEST_FAILURE,
}

_NON_MUTATING_STRUCTURAL_OUTCOME = re.compile(
    r"(?:\bno\b|\bwithout\b[^.]{0,80}\b(?:any\s+)?)(?:repository|repo|source|workspace|file|code)"
    r"[^.]{0,80}\b(?:change|edit|modif|mutation|write)",
    re.IGNORECASE,
)


def _default_evidence_type(
    claim_type: ClaimType,
    observable_outcome: str,
) -> FactType:
    # Observation-only structural stages are proven by a durable code
    # observation. Ordinary structural implementation stages still require a
    # code change.
    if (
        claim_type is ClaimType.STRUCTURAL
        and _NON_MUTATING_STRUCTURAL_OUTCOME.search(observable_outcome)
    ):
        return FactType.CODE_OBSERVATION
    return _DEFAULT_EVIDENCE_TYPE[claim_type]

_TERMINAL_CLAIMS = (
    ClaimType.STRUCTURAL,
    ClaimType.BEHAVIORAL,
    ClaimType.EQUIVALENCE,
    ClaimType.INVARIANT,
    ClaimType.REGRESSION,
    ClaimType.PERFORMANCE,
)


def _claim_type_from_evidence(evidence_types: Sequence[FactType]) -> ClaimType:
    selected = set(evidence_types)
    if FactType.TEST_FAILURE in selected:
        return ClaimType.FAILURE_REPRODUCTION
    if FactType.TEST_RESULT in selected or FactType.VERIFIER_RESULT in selected:
        return ClaimType.BEHAVIORAL
    if FactType.CODE_CHANGE in selected or FactType.CODE_OBSERVATION in selected:
        return ClaimType.STRUCTURAL
    if FactType.TOOL_RESULT in selected:
        return ClaimType.PROCESS
    if FactType.IMPLEMENTATION_DECISION in selected:
        return ClaimType.DECISION
    return ClaimType.STRUCTURAL


def behavioral_commitment_schema(*, terminal: bool) -> dict[str, object]:
    """Return the model-facing requirement-to-behaviour language.

    A model states what the repository must do and any currently known natural
    addresses. It does not publish Evidence types, predicates, verifier kinds,
    proof digests, or a separate oracle. ``test_selectors`` are optional hints:
    a test may not exist until the active Milestone creates it.
    """

    non_empty = {"type": "string", "minLength": 1}
    claim_types = _TERMINAL_CLAIMS if terminal else tuple(ClaimType)
    return {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "requirement_text": dict(non_empty),
            "observable_outcome": dict(non_empty),
            "claim_type": {
                "type": "string",
                "enum": [item.value for item in claim_types],
                "description": "The kind of repository behaviour stated by the requirement.",
            },
            "entity_refs": {
                "type": "array",
                "items": dict(non_empty),
                "description": (
                    "Known natural file, symbol, test, or result addresses. Leave empty when "
                    "the active Milestone still needs to inspect the repository."
                ),
            },
            "test_selectors": {
                "type": "array",
                "items": dict(non_empty),
                "description": (
                    "Optional tests already known to observe this requirement. Do not invent a "
                    "future path merely to satisfy the control protocol."
                ),
            },
        },
        "required": [
            "requirement_text",
            "observable_outcome",
            "claim_type",
            "entity_refs",
            "test_selectors",
        ],
    }


def acceptance_direction_schema() -> dict[str, object]:
    """Return the initial stage-skeleton language."""

    schema = behavioral_commitment_schema(terminal=True)
    properties = dict(schema["properties"])
    properties.pop("test_selectors")
    return {
        **schema,
        "properties": properties,
        "required": [
            "requirement_text",
            "observable_outcome",
            "claim_type",
            "entity_refs",
        ],
    }


def compile_behavioral_commitment(
    value: Mapping[str, object],
    *,
    criterion_id: str,
    entity_refs: Sequence[str],
    test_selectors: Sequence[str],
    level: CommitmentLevel,
    verification_mode: CriterionVerificationMode | None = None,
) -> CompletionCriterionSpec:
    """Compile a concise requirement statement into a typed Evidence boundary.

    The compilation only prevents evidence-category mistakes such as treating
    ``CODE_CHANGE`` as proof of behaviour. It does not freeze how the model
    must test the requirement and cannot replace Milestone semantic review.
    ``verification_mode`` may downgrade a requirement that has no executable
    address to a bounded semantic review; it can never upgrade one.
    """

    try:
        claim_type = ClaimType(str(value.get("claim_type", "")))
    except ValueError as error:
        raise ValueError("behavioral commitment has an unknown claim_type") from error
    if level in {CommitmentLevel.DIRECTION, CommitmentLevel.MILESTONE} and (
        claim_type not in _TERMINAL_CLAIMS
    ):
        raise ValueError(
            f"{claim_type.value} is supporting Step work, not a terminal Milestone outcome"
        )
    requirement_text = str(value.get("requirement_text", "")).strip()
    observable_outcome = str(value.get("observable_outcome", "")).strip()
    if not requirement_text or not observable_outcome:
        raise ValueError(
            "behavioral commitment requires requirement_text and observable_outcome"
        )
    normalized_entities = tuple(dict.fromkeys(item.strip() for item in entity_refs if item.strip()))
    normalized_selectors = tuple(
        dict.fromkeys(item.strip() for item in test_selectors if item.strip())
    )
    required_types = (
        ()
        if level is CommitmentLevel.DIRECTION
        or verification_mode is CriterionVerificationMode.SEMANTIC
        else (_default_evidence_type(claim_type, observable_outcome),)
    )
    criterion = CompletionCriterionSpec(
        criterion_id=criterion_id,
        requirement_id=criterion_id.replace(".C", ".R", 1),
        requirement_text=requirement_text,
        observable_outcome=observable_outcome,
        claim_type=claim_type,
        required_evidence_types=required_types,
        entity_refs=normalized_entities,
        test_selectors=normalized_selectors,
        required=True,
        commitment_level=level,
        verification_mode=verification_mode,
    )
    rejections = criterion_contract_rejections(criterion)
    if rejections:
        raise ValueError("invalid behavioral commitment: " + ", ".join(rejections))
    return criterion


def criterion_contract_rejections(
    criterion: Mapping[str, object] | CompletionCriterionSpec,
) -> tuple[str, ...]:
    """Reject only factual-strength contradictions in a commitment.

    Missing selectors are not errors: addresses are learned during execution,
    and the Milestone-boundary requirement review checks semantic coverage.
    """

    if isinstance(criterion, CompletionCriterionSpec):
        claim_type = criterion.claim_type
        level = criterion.commitment_level
        evidence_types = frozenset(criterion.required_evidence_types)
        mode = criterion.verification_mode
    else:
        raw_types = criterion.get("required_evidence_types", ())
        try:
            parsed_evidence_types = tuple(FactType(str(item)) for item in raw_types)
        except ValueError:
            return ("UNKNOWN_EVIDENCE_TYPE",)
        raw_claim_type = criterion.get("claim_type")
        try:
            claim_type = (
                ClaimType(str(raw_claim_type))
                if raw_claim_type
                else _claim_type_from_evidence(parsed_evidence_types)
            )
        except ValueError:
            return ("UNKNOWN_CLAIM_TYPE",)
        try:
            level = CommitmentLevel(
                str(criterion.get("commitment_level", CommitmentLevel.STEP.value))
            )
        except ValueError:
            return ("UNKNOWN_COMMITMENT_LEVEL",)
        evidence_types = frozenset(parsed_evidence_types)
        raw_mode = criterion.get("verification_mode")
        try:
            mode = (
                CriterionVerificationMode(str(raw_mode))
                if raw_mode
                else default_verification_mode(
                    claim_type,
                    requirement_id=str(criterion.get("requirement_id", "")),
                    required_evidence_types=parsed_evidence_types,
                    commitment_level=level,
                )
            )
        except ValueError:
            return ("UNKNOWN_VERIFICATION_MODE",)

    if level is CommitmentLevel.DIRECTION:
        return () if not evidence_types else ("DIRECTION_CANNOT_REQUIRE_EXECUTION_EVIDENCE",)
    if mode is CriterionVerificationMode.SEMANTIC:
        # A semantic requirement is decided by one bounded model review, not by
        # an execution result category.  It therefore carries no factual
        # evidence obligation and can never hold the route open on its own.
        return ()
    if not evidence_types:
        return ("MISSING_FACTUAL_EVIDENCE_TYPE",)
    primary = _CLAIM_PRIMARY_EVIDENCE_TYPES[claim_type]
    compatible = _CLAIM_COMPATIBLE_EVIDENCE_TYPES[claim_type]
    if not evidence_types.intersection(primary) or not evidence_types.issubset(compatible):
        return ("EVIDENCE_TOO_WEAK_FOR_REQUIREMENT",)
    return ()


def criteria_contract_rejections(
    criteria: Sequence[Mapping[str, object] | CompletionCriterionSpec],
) -> dict[str, tuple[str, ...]]:
    rejected: dict[str, tuple[str, ...]] = {}
    for criterion in criteria:
        reasons = criterion_contract_rejections(criterion)
        if isinstance(criterion, CompletionCriterionSpec):
            criterion_id = criterion.criterion_id
        else:
            criterion_id = str(criterion.get("criterion_id", ""))
        if reasons:
            rejected[criterion_id] = reasons
    return rejected


def compatible_evidence_types(claim_type: ClaimType) -> frozenset[FactType]:
    """Public acceptance-kernel mapping used by Evidence reduction."""

    return _CLAIM_COMPATIBLE_EVIDENCE_TYPES[claim_type]
