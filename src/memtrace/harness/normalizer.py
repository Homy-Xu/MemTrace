from __future__ import annotations

import json
import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from typing import Any

from ..acceptance import (
    acceptance_direction_schema,
    compile_behavioral_commitment,
    criterion_contract_rejections,
)
from ..contracts import (
    TASK_FINAL_REQUIREMENT_ID,
    ClaimType,
    CommitmentLevel,
    CompletionCriterionSpec,
    CriterionVerificationMode,
    FactType,
    MilestoneReviewDecision,
    MilestoneSpec,
    MilestoneStatus,
    NativePlanItemSpec,
    NativePlanSnapshot,
    PlanSpec,
    PlanStepSpec,
    PlanStepStatus,
    primitive,
)
from ..references import ReferenceIdentityFactory
from .command_semantics import command_evidence_semantics
from .memory_tools import preliminary_step_schema

_JSON_FENCE = re.compile(r"```(?:json)?\s*(\{.*?\})\s*```", re.DOTALL | re.IGNORECASE)
_MARKDOWN_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_MARKDOWN_LIST_ITEM = re.compile(r"^(\s*)(?:[-+*]|\d+[.)])\s+(.+?)\s*$")
_MARKDOWN_ORDERED_ITEM = re.compile(r"^\s*\d+[.)]\s+")
# The Provider is asked for an English ``## Ordered Execution Plan`` heading,
# but a model answering in the Task's or its own language may translate it
# (2.2.91 pressure run: ``## 有序执行计划``).  The heading is a route-order
# marker, not a contract, so its recognized spellings are language-aware and
# numbering/punctuation around them is ignored.
_NATIVE_PLAN_SEQUENCE_HEADINGS = frozenset(
    {
        "ordered execution plan",
        "execution plan",
        "execution order",
        "ordered execution steps",
        "execution steps",
        "有序执行计划",
        "执行计划",
        "执行顺序",
        "有序执行步骤",
        "执行步骤",
    }
)
_HEADING_DECORATION = re.compile(r"^(?:\d+[.)]|第[一二三四五六七八九十\d]+[步节章部分]?[、.:：]?)\s*")
_HEADING_TRAILER = re.compile(r"[\s:：。,，;；]+$")


def _sequence_heading_key(title: str) -> str:
    """Normalize a Markdown heading for the route-order heading whitelist."""

    text = " ".join(title.split()).strip("*_` ")
    text = _HEADING_DECORATION.sub("", text)
    text = _HEADING_TRAILER.sub("", text)
    return " ".join(text.casefold().split())

# This is an abuse guard for malformed Provider output, not a semantic target
# exposed to Planning. Real Milestone cardinality is chosen from the Task's
# natural acceptance-verifiable stage boundaries.
_MILESTONE_OPERATIONAL_SAFETY_LIMIT = 128
_MILESTONE_MANIFEST_BYTE_LIMIT = 1024 * 1024

# Completion contracts may name only facts that the execution path can
# produce before the Step/Milestone is accepted.  Route facts such as
# PLAN_DECISION and MILESTONE_STATE describe control state; accepting them as
# proof of that same control transition creates a circular contract and was a
# source of review loops in long-running tasks.
_ACCEPTANCE_EVIDENCE_TYPES = frozenset(
    {
        FactType.CODE_OBSERVATION,
        FactType.IMPLEMENTATION_DECISION,
        FactType.CODE_CHANGE,
        FactType.TOOL_RESULT,
        FactType.TEST_RESULT,
        FactType.TEST_FAILURE,
        FactType.VERIFIER_RESULT,
    }
)

_PRELIMINARY_STEP_KINDS = frozenset({"INSPECT", "IMPLEMENT", "VERIFY", "PROCESS"})


class MilestoneManifestRequired(ValueError):
    """The native flat Plan is too granular to allocate stable Milestones."""


@dataclass(frozen=True, slots=True)
class MilestoneReviewProposal:
    milestone_id: str
    decision: MilestoneReviewDecision
    reason: str
    future_plan: PlanSpec | None = None
    corrective_steps: tuple[PlanStepSpec, ...] = ()
    requirement_coverage: tuple[Mapping[str, str], ...] = ()


class CodexPlanNormalizer:
    """Fail-closed compiler from a native Codex Plan to the TPG route.

    The immutable native snapshot remains planning authority. The typed output
    groups source items into stage-level Milestones, records their observable
    outcomes, and preserves only the active stage's preliminary work as
    lightweight TPG cursors. These cursors guide ordinary Coding without becoming Provider
    Turn boundaries, approval gates, or a second proof language.
    """

    def __init__(self) -> None:
        self._trusted_verification_selectors: tuple[str, ...] = ()

    def bind_trusted_verification_contract(self, selectors: Sequence[str]) -> None:
        """Bind one immutable benchmark acceptance set before Planning starts."""

        normalized = tuple(
            dict.fromkeys(str(selector).strip() for selector in selectors if str(selector).strip())
        )
        if self._trusted_verification_selectors and (
            self._trusted_verification_selectors != normalized
        ):
            raise RuntimeError("Plan normalizer is already bound to another verification contract")
        self._trusted_verification_selectors = normalized

    @staticmethod
    def acceptance_evidence_types() -> tuple[str, ...]:
        return tuple(sorted(item.value for item in _ACCEPTANCE_EVIDENCE_TYPES))

    @staticmethod
    def initial_projection_schema() -> dict[str, Any]:
        """Expose only the semantic choices required to seed the initial TPG.

        Stable identities, linear dependencies, Criterion mappings, final
        acceptance and status are runtime-owned. This publication groups the
        frozen native Plan and declares observable stage outcomes. Only the
        first active stage carries concrete preliminary cursors; later stages
        remain semantic skeletons until their execution boundary. It never invents
        test selectors or opens a second Milestone-activation protocol.
        """

        non_empty_string: dict[str, Any] = {"type": "string", "minLength": 1}

        def string_list(*, minimum: int = 0) -> dict[str, Any]:
            return {
                "type": "array",
                "items": dict(non_empty_string),
                "minItems": minimum,
            }

        direction_claim = acceptance_direction_schema()
        direction_properties = direction_claim["properties"]
        milestone = {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "source_plan_item_ids": string_list(minimum=1),
                "task_requirement": {
                    **dict(non_empty_string),
                    "description": (
                        "The smallest verbatim excerpt of the original Task that makes this "
                        "stage necessary. Supporting or conditional native Plan work with no "
                        "such Task requirement must be grouped into the stage it serves."
                    ),
                },
                "target_outcome": {
                    **dict(non_empty_string),
                    "description": (
                        "The observable repository state this stage must reach. It may clarify "
                        "the quoted requirement but cannot add scope or strengthen its modality."
                    ),
                },
                "claim_type": dict(direction_properties["claim_type"]),
                "entity_refs": {
                    **dict(direction_properties["entity_refs"]),
                    "description": (
                        "Known natural entity addresses for the first active Milestone. "
                        "Omit or use an empty list for future Milestone skeletons."
                    ),
                },
                "non_goals": string_list(),
                "steps": {
                    "type": "array",
                    "items": preliminary_step_schema(),
                    "minItems": 0,
                    "description": (
                        "For the first active Milestone only: ordered preliminary execution "
                        "cursors. A coarse native Plan item may split into several Steps and "
                        "each Step keeps exactly one Nxxx source. Future Milestones must use "
                        "an empty list and are materialized only at their execution boundary."
                    ),
                },
            },
            "required": [
                "source_plan_item_ids",
                "task_requirement",
                "target_outcome",
                "claim_type",
                "non_goals",
                "steps",
            ],
        }
        return {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                "milestones": {
                    "type": "array",
                    "items": milestone,
                    "minItems": 1,
                }
            },
            "required": ["milestones"],
        }

    @staticmethod
    def output_schema() -> dict[str, Any]:
        """Return the compact model-facing contract for a stage Plan.

        The model declares semantic outcomes and references once. The
        normalizer deterministically expands Step and final references into the
        complete internal ``PlanSpec`` contract. This keeps internal identity,
        status and duplication mechanics out of the model's control protocol.
        """

        non_empty_string: dict[str, Any] = {"type": "string", "minLength": 1}

        def string_list(*, minimum: int = 0) -> dict[str, Any]:
            return {
                "type": "array",
                "items": dict(non_empty_string),
                "minItems": minimum,
            }

        def criterion_schema() -> dict[str, Any]:
            return {
                "type": "object",
                "properties": {
                    "criterion_id": dict(non_empty_string),
                    "observable_outcome": dict(non_empty_string),
                    "requirement_id": dict(non_empty_string),
                    "requirement_text": dict(non_empty_string),
                    "claim_type": {
                        "type": "string",
                        "enum": [item.value for item in ClaimType],
                    },
                    "required_evidence_types": {
                        "type": "array",
                        "items": {
                            "type": "string",
                            "enum": list(CodexPlanNormalizer.acceptance_evidence_types()),
                        },
                        "minItems": 1,
                        "description": (
                            "Conjunctive typed facts for one observable outcome. TOOL_RESULT is "
                            "a generic non-test command result and must be the only type in its "
                            "criterion. A test runner or executable Python assertion produces "
                            "TEST_RESULT, not TOOL_RESULT; put code observations, changes, and "
                            "test/verification outcomes in separate criteria."
                        ),
                    },
                    "entity_refs": string_list(),
                    "test_selectors": {
                        **string_list(),
                        "description": (
                            "Exact result addresses. Test runners and executable Python "
                            "assertion/raise probes must use TEST_RESULT; non-test commands "
                            "use TOOL_RESULT. Print-only inspection is diagnostic TOOL_RESULT "
                            "evidence. The runtime rejects a selector whose deterministic command "
                            "class disagrees with its declared Evidence type."
                        ),
                    },
                },
                "required": [
                    "criterion_id",
                    "observable_outcome",
                    "requirement_id",
                    "requirement_text",
                    "claim_type",
                    "required_evidence_types",
                    "entity_refs",
                    "test_selectors",
                ],
                "additionalProperties": False,
            }

        step_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "step_id": dict(non_empty_string),
                "title": dict(non_empty_string),
                "criterion_ids": {
                    **string_list(),
                    "description": (
                        "Stable mappings to criterion_id values declared by the enclosing "
                        "Milestone minimum_acceptance. These are graph contribution links, not "
                        "Step-local acceptance obligations; pre-change or diagnostic work may "
                        "use an empty list."
                    ),
                },
                "entity_refs": string_list(),
                "expected_outcome": dict(non_empty_string),
                "failure_signals": string_list(minimum=1),
                "historical_dependency_refs": string_list(),
            },
            "required": [
                "step_id",
                "title",
                "criterion_ids",
                "entity_refs",
                "expected_outcome",
                "failure_signals",
                "historical_dependency_refs",
            ],
            "additionalProperties": False,
        }
        milestone_schema: dict[str, Any] = {
            "type": "object",
            "properties": {
                "canonical_id": {
                    "type": "string",
                    "pattern": "^M[0-9]{3,}$",
                },
                "title": dict(non_empty_string),
                "objective": dict(non_empty_string),
                "target_outcome": dict(non_empty_string),
                "scope": dict(non_empty_string),
                "description": dict(non_empty_string),
                "depends_on": string_list(),
                "downstream_assumptions": string_list(),
                "non_goals": string_list(),
                "entity_refs": string_list(),
                "source_plan_items": {
                    **string_list(minimum=1),
                    "description": (
                        "Compatibility form: exact ordered labels from the native Codex Plan."
                    ),
                },
                "source_plan_item_ids": {
                    **string_list(minimum=1),
                    "description": (
                        "Stable Nxxx addresses from the authoritative Ordered Execution Plan. "
                        "Every address must belong to exactly one Milestone and remain in source "
                        "order. Prefer this field over source_plan_items."
                    ),
                },
                "verification": string_list(minimum=1),
                "minimum_acceptance": {
                    "type": "array",
                    "items": criterion_schema(),
                    "minItems": 1,
                },
                "steps": {
                    "type": "array",
                    "items": step_schema,
                    "description": (
                        "Concrete work for the first active Milestone only. Leave this empty for "
                        "future Milestones; their Steps are planned when that Milestone activates."
                    ),
                },
            },
            "required": [
                "canonical_id",
                "title",
                "objective",
                "target_outcome",
                "scope",
                "description",
                "depends_on",
                "downstream_assumptions",
                "non_goals",
                "entity_refs",
                "source_plan_item_ids",
                "verification",
                "minimum_acceptance",
                "steps",
            ],
            "additionalProperties": False,
        }
        return {
            "type": "object",
            "properties": {
                "goal": dict(non_empty_string),
                "milestones": {
                    "type": "array",
                    "items": milestone_schema,
                    "minItems": 1,
                },
                "final_verification": string_list(minimum=1),
                "final_criterion_ids": string_list(minimum=1),
            },
            "required": [
                "goal",
                "milestones",
                "final_verification",
                "final_criterion_ids",
            ],
            "additionalProperties": False,
        }

    @staticmethod
    def _expand_compact_manifest(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
        """Compile model semantics into the full internal Plan contract.

        Full manifests remain accepted for supplied/offline compatibility. A
        compact manifest is explicitly identified by ``final_criterion_ids``;
        the runtime never guesses which representation the model intended.
        """

        if "final_criterion_ids" not in manifest:
            return manifest
        raw_milestones = manifest.get("milestones")
        if not isinstance(raw_milestones, list) or not raw_milestones:
            raise MilestoneManifestRequired("compact MilestoneManifest contains no milestones")
        expanded_milestones: list[dict[str, Any]] = []
        global_criteria: dict[str, dict[str, Any]] = {}
        for raw_milestone in raw_milestones:
            if not isinstance(raw_milestone, Mapping):
                raise MilestoneManifestRequired("compact Milestone node is not an object")
            raw_acceptance = raw_milestone.get("minimum_acceptance")
            if not isinstance(raw_acceptance, list) or not raw_acceptance:
                raise MilestoneManifestRequired(
                    "compact Milestone requires non-empty minimum_acceptance"
                )
            milestone_criteria: dict[str, dict[str, Any]] = {}
            expanded_acceptance: list[dict[str, Any]] = []
            for raw_criterion in raw_acceptance:
                if not isinstance(raw_criterion, Mapping):
                    raise MilestoneManifestRequired("compact acceptance is not an object")
                criterion = {**dict(raw_criterion), "required": True}
                criterion_id = str(criterion.get("criterion_id", "")).strip()
                if not criterion_id:
                    raise MilestoneManifestRequired("compact acceptance has no criterion_id")
                if criterion_id in global_criteria:
                    raise MilestoneManifestRequired(
                        f"compact criterion_id is not globally unique: {criterion_id}"
                    )
                milestone_criteria[criterion_id] = criterion
                global_criteria[criterion_id] = criterion
                expanded_acceptance.append(criterion)
            raw_steps = raw_milestone.get("steps")
            if not isinstance(raw_steps, list):
                raise MilestoneManifestRequired("compact Milestone PlanSteps must be a JSON list")
            expanded_steps: list[dict[str, Any]] = []
            for raw_step in raw_steps:
                if not isinstance(raw_step, Mapping):
                    raise MilestoneManifestRequired("compact PlanStep is not an object")
                criterion_ids = raw_step.get("criterion_ids")
                if not isinstance(criterion_ids, list) or not all(
                    isinstance(item, str) and item.strip() for item in criterion_ids
                ):
                    raise MilestoneManifestRequired(
                        "compact PlanStep criterion_ids must be a JSON string list"
                    )
                unknown = set(map(str, criterion_ids)).difference(milestone_criteria)
                if unknown:
                    raise MilestoneManifestRequired(
                        f"compact PlanStep maps unknown Milestone criteria: {sorted(unknown)}"
                    )
                expanded_steps.append(
                    {
                        "step_id": raw_step.get("step_id"),
                        "title": raw_step.get("title"),
                        "status": "PENDING",
                        "criterion_ids": list(map(str, criterion_ids)),
                        "entity_refs": raw_step.get("entity_refs", []),
                        "expected_outcome": raw_step.get("expected_outcome"),
                        # Compatibility storage only. Step is a route cursor,
                        # never a second acceptance contract.
                        "minimum_acceptance": [],
                        "failure_signals": raw_step.get("failure_signals", []),
                        "historical_dependency_refs": raw_step.get(
                            "historical_dependency_refs", []
                        ),
                    }
                )
            expanded_milestones.append(
                {
                    **dict(raw_milestone),
                    "minimum_acceptance": expanded_acceptance,
                    "steps": expanded_steps,
                }
            )
        raw_final_ids = manifest.get("final_criterion_ids")
        if not isinstance(raw_final_ids, list) or not raw_final_ids:
            raise MilestoneManifestRequired("compact final_criterion_ids must be non-empty")
        unknown_final = set(map(str, raw_final_ids)).difference(global_criteria)
        if unknown_final:
            raise MilestoneManifestRequired(
                f"compact final acceptance maps unknown criteria: {sorted(unknown_final)}"
            )
        return {
            "goal": manifest.get("goal"),
            "milestones": expanded_milestones,
            "final_verification": manifest.get("final_verification"),
            "final_acceptance": [dict(global_criteria[str(item)]) for item in raw_final_ids],
        }

    def normalize(
        self,
        *,
        user_task: str,
        steps: Sequence[Mapping[str, Any]],
        final_plan_text: str | None = None,
        native_plan_text: str | None = None,
    ) -> PlanSpec:
        native_plan = self.native_plan_snapshot(steps, final_plan_text=native_plan_text)
        manifest = self._manifest(final_plan_text)
        if manifest is not None:
            return self._from_manifest(
                user_task=user_task,
                manifest=manifest,
                native_plan=native_plan,
            )

        if not steps and not final_plan_text:
            raise MilestoneManifestRequired(
                "Codex planning turn produced no structured or final Plan steps"
            )
        # A flat native Plan is a sequence of execution steps, not proof of
        # sparse task-stage boundaries. Allocate Milestone identities only
        # after the same Harness supplies the typed manifest.
        raise MilestoneManifestRequired(
            "a structured stage-level MilestoneManifest is required before stable IDs "
            "can be allocated"
        )

    def project_native_plan(
        self,
        *,
        user_task: str,
        steps: Sequence[Mapping[str, Any]],
        projection: Mapping[str, Any],
        native_plan_text: str | None = None,
    ) -> PlanSpec:
        """Compile one lightweight semantic projection into the runtime Plan.

        Full manifests remain readable for supplied/offline compatibility, but
        the live projection tool exposes ``initial_projection_schema`` and
        therefore cannot ask the model to construct runtime identities or
        cross-object acceptance mappings.
        """

        native_plan = self.native_plan_snapshot(steps, final_plan_text=native_plan_text)
        if native_plan is None:
            raise MilestoneManifestRequired(
                "native Planning completed without publishing any Plan items"
            )
        raw_milestones = projection.get("milestones")
        if isinstance(raw_milestones, list) and any(
            isinstance(item, Mapping)
            and ("canonical_id" in item or "objective" in item or "verification" in item)
            for item in raw_milestones
        ):
            return self._from_manifest(
                user_task=user_task,
                manifest=projection,
                native_plan=native_plan,
            )
        return self._from_initial_projection(
            user_task=user_task,
            projection=projection,
            native_plan=native_plan,
        )

    def _from_initial_projection(
        self,
        *,
        user_task: str,
        projection: Mapping[str, Any],
        native_plan: NativePlanSnapshot,
    ) -> PlanSpec:
        encoded_size = len(
            json.dumps(
                projection,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if encoded_size > _MILESTONE_MANIFEST_BYTE_LIMIT:
            raise MilestoneManifestRequired(
                "native Plan projection exceeds the operational serialized-size safety limit"
            )
        raw_milestones = projection.get("milestones")
        if not isinstance(raw_milestones, list) or not raw_milestones:
            raise MilestoneManifestRequired("native Plan projection contains no Milestones")
        if len(raw_milestones) > _MILESTONE_OPERATIONAL_SAFETY_LIMIT:
            raise MilestoneManifestRequired(
                "native Plan projection exceeds the operational node-count safety limit"
            )
        task_anchored_live_projection = all(
            isinstance(item, Mapping) and "task_requirement" in item
            for item in raw_milestones
        )

        expected_source_ids = tuple(item.source_step_id for item in native_plan.items)
        source_titles = {item.source_step_id: item.title for item in native_plan.items}
        projected_source_ids: list[str] = []
        grouped_source_ids: list[tuple[str, ...]] = []
        for ordinal, raw in enumerate(raw_milestones, start=1):
            if not isinstance(raw, Mapping):
                raise MilestoneManifestRequired(f"projected Milestone {ordinal} is not an object")
            source_ids = raw.get("source_plan_item_ids")
            if not isinstance(source_ids, list) or not source_ids:
                raise MilestoneManifestRequired(
                    f"projected Milestone {ordinal} has no native Plan addresses"
                )
            self._require_string_list(
                source_ids,
                scope=f"projected Milestone {ordinal} source_plan_item_ids",
            )
            normalized_source_ids = tuple(map(str, source_ids))
            grouped_source_ids.append(normalized_source_ids)
            projected_source_ids.extend(normalized_source_ids)
        if tuple(projected_source_ids) != expected_source_ids:
            raise MilestoneManifestRequired(
                "the lightweight projection must preserve every frozen native Plan address "
                f"exactly once and in order; expected {expected_source_ids}, "
                f"got {tuple(projected_source_ids)}"
            )

        compiled: list[MilestoneSpec] = []
        last_ordinal = len(raw_milestones)
        for ordinal, (raw, source_ids) in enumerate(
            zip(raw_milestones, grouped_source_ids, strict=True),
            start=1,
        ):
            assert isinstance(raw, Mapping)
            canonical_id = f"M{ordinal:03d}"
            scope_anchored_projection = "task_requirement" in raw
            grounded_route_projection = (
                scope_anchored_projection
                and ordinal == 1
                and bool(str(raw.get("target_outcome", "")).strip())
                and isinstance(raw.get("steps"), list)
                and bool(raw.get("steps"))
            )
            task_requirement = (
                self._task_requirement_anchor(
                    raw.get("task_requirement"),
                    user_task=user_task,
                    scope=canonical_id,
                )
                if scope_anchored_projection
                else ""
            )
            target_outcome = (
                str(raw.get("target_outcome", task_requirement)).strip()
                if scope_anchored_projection
                else str(raw.get("target_outcome", "")).strip()
            )
            title = (
                self._task_route_title(target_outcome)
                if scope_anchored_projection
                else str(raw.get("title", "")).strip()
            )
            if not title or not target_outcome:
                raise MilestoneManifestRequired(
                    f"{canonical_id} requires a title and target_outcome"
                )
            raw_non_goals = raw.get("non_goals")
            if not isinstance(raw_non_goals, list):
                raise MilestoneManifestRequired(f"{canonical_id} non_goals must be a JSON list")
            self._require_string_list(raw_non_goals, scope=f"{canonical_id} non_goals")

            lifecycle_projection = (
                scope_anchored_projection
                or "acceptance_directions" in raw
                or "active_execution" in raw
            )
            active_execution = None
            if scope_anchored_projection:
                criterion = self._task_anchored_criterion(
                    raw if ordinal == 1 else {**dict(raw), "entity_refs": []},
                    user_task=user_task,
                    criterion_id=f"{canonical_id}.C001",
                    scope=canonical_id,
                    level=(
                        CommitmentLevel.MILESTONE
                        if ordinal == 1
                        else CommitmentLevel.DIRECTION
                    ),
                )
                criteria = [criterion]
                raw_steps = raw.get("steps", [])
            elif lifecycle_projection:
                raw_directions = raw.get("acceptance_directions")
                if not isinstance(raw_directions, list) or not raw_directions:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} requires semantic acceptance_directions"
                    )
                active_execution = raw.get("active_execution")
                # Old durable WAL may still contain the retired activation
                # envelope. It remains readable below, but only the live
                # activation-free projection is constrained to one direction.
                if active_execution is None and len(raw_directions) != 1:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} requires exactly one concise semantic "
                        "acceptance direction"
                    )
                direction_criteria = [
                    self._acceptance_direction_criterion(
                        item,
                        criterion_id=f"{canonical_id}.C{index:03d}",
                        scope=f"{canonical_id} acceptance_directions",
                    )
                    for index, item in enumerate(raw_directions, start=1)
                ]
                if active_execution is not None:
                    # Transitional WAL and supplied projections from the
                    # previous protocol remain readable.  The live schema no
                    # longer exposes active_execution: route publication and
                    # M001 activation are separate lifecycle boundaries.
                    if ordinal != 1 or not isinstance(active_execution, Mapping):
                        raise MilestoneManifestRequired(
                            f"{canonical_id} is pending; active_execution must be null"
                        )
                    raw_acceptance = active_execution.get("minimum_acceptance")
                    raw_steps = active_execution.get("steps")
                    if not isinstance(raw_acceptance, list) or not raw_acceptance:
                        raise MilestoneManifestRequired(
                            f"{canonical_id} active_execution requires minimum_acceptance"
                        )
                    criteria = [
                        self._projection_criterion(
                            item,
                            criterion_id=f"{canonical_id}.C{index:03d}",
                            scope=f"{canonical_id} active minimum_acceptance",
                            terminal=True,
                        )
                        for index, item in enumerate(raw_acceptance, start=1)
                    ]
                    self._assert_directions_preserved(
                        direction_criteria,
                        criteria,
                        scope=f"{canonical_id} active_execution",
                    )
                else:
                    raw_steps = []
                    # Only the active stage receives an executable factual
                    # contract. Future stages remain pure semantic directions
                    # until the preceding Milestone review selects them.
                    criteria = (
                        [
                            self._materialize_direction(item)
                            for item in direction_criteria
                        ]
                        if ordinal == 1
                        else direction_criteria
                    )
            else:
                # Supplied/offline and old WAL projections remain readable.
                raw_acceptance = raw.get("minimum_acceptance")
                if not isinstance(raw_acceptance, list) or not raw_acceptance:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} requires observable minimum_acceptance"
                    )
                criteria = [
                    self._projection_criterion(
                        item,
                        criterion_id=f"{canonical_id}.C{index:03d}",
                        scope=f"{canonical_id} minimum_acceptance",
                        terminal=True,
                    )
                    for index, item in enumerate(raw_acceptance, start=1)
                ]
                raw_steps = raw.get("steps", [] if ordinal > 1 else None)

            if ordinal == last_ordinal and task_anchored_live_projection:
                criteria = list(
                    (
                        self._with_final_task_review
                        if ordinal == 1
                        else self._with_final_task_direction
                    )(
                        criteria,
                        canonical_id=canonical_id,
                        user_task=user_task,
                    )
                )
            if ordinal == last_ordinal and ordinal == 1:
                criteria = list(
                    self._with_trusted_verification(
                        criteria,
                        canonical_id=canonical_id,
                    )
                )

            if not isinstance(raw_steps, list):
                raise MilestoneManifestRequired(f"{canonical_id} steps must be a JSON list")
            if scope_anchored_projection and ordinal == 1 and not raw_steps:
                raise MilestoneManifestRequired(
                    f"{canonical_id} is the first active Milestone and requires ordered "
                    "preliminary Steps"
                )
            if scope_anchored_projection and ordinal > 1 and raw_steps:
                raise MilestoneManifestRequired(
                    f"{canonical_id} is a future Milestone; keep only its semantic skeleton "
                    "and materialize Steps at the Milestone boundary"
                )
            if (
                lifecycle_projection
                and not scope_anchored_projection
                and ordinal > 1
                and raw_steps
            ):
                raise MilestoneManifestRequired(
                    f"{canonical_id} is a future Milestone; plan its Steps only when activated"
                )
            if not lifecycle_projection and ordinal == 1 and not raw_steps:
                raise MilestoneManifestRequired("the first active Milestone requires concrete Steps")
            if not lifecycle_projection and ordinal > 1 and raw_steps:
                raise MilestoneManifestRequired(
                    f"{canonical_id} is a future Milestone; plan its Steps only when activated"
                )

            step_specs: list[PlanStepSpec] = []
            step_source_ids: list[str] = []
            if grounded_route_projection:
                step_specs = list(
                    self._preliminary_steps(
                        milestone_id=canonical_id,
                        raw_steps=raw_steps,
                        source_plan_item_ids=source_ids,
                        scope=canonical_id,
                    )
                )
                step_source_ids = [
                    source_id
                    for step in step_specs
                    for source_id in step.source_plan_item_ids
                ]
            for step_index, raw_step in enumerate(
                [] if grounded_route_projection else raw_steps,
                start=1,
            ):
                if not isinstance(raw_step, Mapping):
                    raise MilestoneManifestRequired(
                        f"{canonical_id} active Step {step_index} is not an object"
                    )
                raw_step_sources = raw_step.get("source_plan_item_ids")
                if not isinstance(raw_step_sources, list) or not raw_step_sources:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} active Step {step_index} has no native Plan addresses"
                    )
                self._require_string_list(
                    raw_step_sources,
                    scope=f"{canonical_id} active Step {step_index} source_plan_item_ids",
                )
                step_source_ids.extend(map(str, raw_step_sources))
                step_title = str(raw_step.get("title", "")).strip()
                expected_outcome = str(raw_step.get("expected_outcome", "")).strip()
                failure_signals = raw_step.get("failure_signals")
                historical_dependency_refs = raw_step.get("historical_dependency_refs", [])
                if not step_title or not expected_outcome:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} active Step {step_index} needs title/expected_outcome"
                    )
                if not isinstance(failure_signals, list) or not failure_signals:
                    raise MilestoneManifestRequired(
                        f"{canonical_id} active Step {step_index} needs failure_signals"
                    )
                self._require_string_list(
                    failure_signals,
                    scope=f"{canonical_id} active Step {step_index} failure_signals",
                )
                if not isinstance(historical_dependency_refs, list):
                    raise MilestoneManifestRequired(
                        f"{canonical_id} active Step {step_index} "
                        "historical_dependency_refs must be a JSON list"
                    )
                self._require_string_list(
                    historical_dependency_refs,
                    scope=(f"{canonical_id} active Step {step_index} historical_dependency_refs"),
                )
                step_specs.append(
                    PlanStepSpec(
                        step_id=f"{canonical_id}.S{step_index:03d}",
                        title=step_title,
                        entity_refs=self._normalize_entity_refs(
                            raw_step.get("entity_refs", ())
                        ),
                        expected_outcome=expected_outcome,
                        minimum_acceptance=(),
                        failure_signals=tuple(map(str, failure_signals)),
                        historical_dependency_refs=self._normalize_entity_refs(
                            historical_dependency_refs
                        ),
                        source_plan_item_ids=tuple(map(str, raw_step_sources)),
                    )
                )

            if (
                lifecycle_projection
                and not grounded_route_projection
                and active_execution is None
                and ordinal == 1
            ):
                # A route Step is an address/provenance owner, not a Provider
                # Turn or an approval gate.  It deliberately mirrors the
                # Milestone's broad requirements and leaves investigation,
                # implementation order, concrete tests, and natural Plan
                # updates to the Coding Harness.  This removes the former
                # read-only Activation Turn that duplicated repository work.
                navigation_step = self._navigation_step(
                    milestone_id=canonical_id,
                    title=title,
                    target_outcome=target_outcome,
                    source_plan_item_ids=source_ids,
                    criteria=criteria,
                )
                step_specs.append(navigation_step)
            if raw_steps and tuple(dict.fromkeys(step_source_ids)) != source_ids:
                raise MilestoneManifestRequired(
                    "the Steps must cover every native Plan address in their "
                    "Milestone at least once and preserve first-occurrence order; one native "
                    "item may derive multiple coherent Steps. Expected "
                    f"{source_ids}, got {tuple(step_source_ids)}"
                )

            entity_refs = tuple(
                dict.fromkeys(
                    (
                        *(
                            entity
                            for criterion in criteria
                            for entity in criterion.entity_refs
                        ),
                        *(entity for step in step_specs for entity in step.entity_refs),
                    )
                )
            )
            verification = tuple(
                dict.fromkeys(
                    (
                        "Review requirement coverage and supporting execution Evidence for: "
                        + criterion.observable_outcome
                    )
                    for criterion in criteria
                )
            )
            source_scope = "; ".join(source_titles[source_id] for source_id in source_ids)
            scope = target_outcome if scope_anchored_projection else source_scope
            downstream_assumptions = (
                (
                    (
                        f"M{ordinal + 1:03d} may start only from this verified target outcome: "
                        f"{target_outcome}"
                    ),
                )
                if ordinal < last_ordinal
                else ()
            )
            compiled.append(
                MilestoneSpec(
                    canonical_id=canonical_id,
                    title=title,
                    description=(
                        f"Task-backed stage goal: {target_outcome}; source requirement: "
                        f"{task_requirement}"
                        if grounded_route_projection
                        else f"Task-anchored stage requirement: {target_outcome}"
                        if scope_anchored_projection
                        else f"Projected from native Plan work: {scope}"
                    ),
                    completion_criteria=tuple(
                        criterion.observable_outcome for criterion in criteria
                    ),
                    verification=verification,
                    depends_on=((f"M{ordinal - 1:03d}",) if ordinal > 1 else ()),
                    entity_refs=entity_refs,
                    objective=target_outcome,
                    scope=scope,
                    target_outcome=target_outcome,
                    downstream_assumptions=downstream_assumptions,
                    non_goals=tuple(map(str, raw_non_goals)),
                    source_plan_item_ids=source_ids,
                    criteria=tuple(criteria),
                    steps=tuple(step_specs),
                )
            )

        # Final acceptance is the Task-wide requirement index.  Restricting it
        # to the terminal Milestone silently dropped requirements that were
        # implemented earlier in the route and made a weakened Plan self-
        # consistent.  Stable Criterion IDs let the final pass reuse the same
        # TPG/Page evidence without rescanning the repository.
        final_acceptance = self._final_acceptance_from_milestones(compiled)
        final_verification = tuple(
            dict.fromkeys(
                (
                    "Verify the final repository state through the terminal Milestone: "
                    f"{item.observable_outcome}"
                )
                for item in final_acceptance
            )
        )
        self._validate_final_evidence_coverage(
            milestones=compiled,
            final_acceptance=final_acceptance,
        )
        return PlanSpec(
            goal=user_task.strip(),
            milestones=tuple(compiled),
            final_verification=final_verification,
            final_acceptance=final_acceptance,
            native_plan=native_plan,
        )

    def _acceptance_direction_criterion(
        self,
        value: object,
        *,
        criterion_id: str,
        scope: str,
    ) -> CompletionCriterionSpec:
        """Persist a future stage's meaning without fabricating execution details.

        These non-executable directions are legal only while the Milestone has
        no Steps and is still pending. Milestone activation preserves their
        requirement semantics while materializing current work.
        """

        if not isinstance(value, Mapping):
            raise MilestoneManifestRequired(f"{scope} contains a non-object direction")
        requirement_text = str(value.get("requirement_text", "")).strip()
        observable_outcome = str(value.get("observable_outcome", "")).strip()
        try:
            claim_type = ClaimType(str(value.get("claim_type", "")))
        except ValueError as error:
            raise MilestoneManifestRequired(f"{scope} contains an unknown claim_type") from error
        if claim_type not in {
            ClaimType.STRUCTURAL,
            ClaimType.BEHAVIORAL,
            ClaimType.EQUIVALENCE,
            ClaimType.INVARIANT,
            ClaimType.REGRESSION,
            ClaimType.PERFORMANCE,
        }:
            raise MilestoneManifestRequired(
                f"{scope} contains supporting work instead of a terminal outcome"
            )
        if not requirement_text or not observable_outcome:
            raise MilestoneManifestRequired(
                f"{scope} requires requirement_text and observable_outcome"
            )
        raw_entities = value.get("entity_refs", [])
        if not isinstance(raw_entities, list):
            raise MilestoneManifestRequired(f"{scope} entity_refs must be a JSON list")
        self._require_string_list(raw_entities, scope=f"{scope} entity_refs")
        entities = self._normalize_entity_refs(raw_entities)
        return compile_behavioral_commitment(
            value,
            criterion_id=criterion_id,
            entity_refs=entities,
            test_selectors=(),
            level=CommitmentLevel.DIRECTION,
        )

    @staticmethod
    def _task_requirement_anchor(
        value: object,
        *,
        user_task: str,
        scope: str,
    ) -> str:
        """Return one immutable original-Task scope anchor.

        Native Plan items describe a proposed route, not new requirements.  A
        standalone Milestone therefore needs a verbatim (whitespace-normalized)
        Task excerpt.  This is a structural scope check, not another semantic
        review or repository-analysis protocol.
        """

        anchor = " ".join(str(value or "").split())
        normalized_task = " ".join(user_task.split())
        if not anchor:
            raise MilestoneManifestRequired(f"{scope} requires task_requirement")
        if anchor.casefold() not in normalized_task.casefold():
            raise MilestoneManifestRequired(
                f"{scope} task_requirement must be a verbatim excerpt of the original Task; "
                "Native Plan suggestions cannot create mandatory scope"
            )
        return anchor

    @staticmethod
    def _task_route_title(task_requirement: str) -> str:
        """Derive the execution-visible label from the immutable Task scope.

        Provider Plan titles remain useful provenance, but they are proposals
        about how to work. Reusing them as the TPG node label allowed a
        suggested "update docs" action to strengthen a Task requirement that
        only said "verify docs". The route label therefore comes from the
        already-validated Task excerpt and cannot add another action modality.
        """

        normalized = " ".join(task_requirement.split())
        if len(normalized) <= 180:
            return normalized
        return normalized[:177].rstrip() + "..."

    def _task_anchored_criterion(
        self,
        value: Mapping[str, Any],
        *,
        user_task: str,
        criterion_id: str,
        scope: str,
        level: CommitmentLevel,
    ) -> CompletionCriterionSpec:
        """Compile a lightweight Criterion whose semantics come only from the Task."""

        anchor = self._task_requirement_anchor(
            value.get("task_requirement"),
            user_task=user_task,
            scope=scope,
        )
        # Old durable task-anchored projections used the verbatim requirement
        # as their outcome. The live schema now requires an explicit stage
        # goal, but recovery must keep the former record readable.
        target_outcome = str(value.get("target_outcome", anchor)).strip()
        direction = self._acceptance_direction_criterion(
            {
                "requirement_text": anchor,
                "observable_outcome": target_outcome,
                "claim_type": value.get("claim_type"),
                "entity_refs": value.get("entity_refs", []),
            },
            criterion_id=criterion_id,
            scope=scope,
        )
        if level is CommitmentLevel.DIRECTION:
            return direction
        return compile_behavioral_commitment(
            {
                "requirement_text": direction.requirement_text,
                "observable_outcome": direction.observable_outcome,
                "claim_type": direction.claim_type.value,
            },
            criterion_id=criterion_id,
            entity_refs=direction.entity_refs,
            test_selectors=(),
            level=level,
        )

    def _preliminary_steps(
        self,
        *,
        milestone_id: str,
        raw_steps: object,
        source_plan_item_ids: Sequence[str],
        scope: str,
    ) -> tuple[PlanStepSpec, ...]:
        """Compile lightweight route cursors without compiling a proof protocol."""

        if not isinstance(raw_steps, list) or not raw_steps:
            raise MilestoneManifestRequired(f"{scope} requires preliminary Steps")
        allowed_sources = set(source_plan_item_ids)
        observed_sources: list[str] = []
        compiled: list[PlanStepSpec] = []
        for index, raw in enumerate(raw_steps, start=1):
            if not isinstance(raw, Mapping):
                raise MilestoneManifestRequired(f"{scope} Step {index} is not an object")
            raw_sources = raw.get("source_plan_item_ids")
            if not isinstance(raw_sources, list) or len(raw_sources) != 1:
                raise MilestoneManifestRequired(
                    f"{scope} Step {index} must keep exactly one native Plan address"
                )
            self._require_string_list(
                raw_sources,
                scope=f"{scope} Step {index} source_plan_item_ids",
            )
            source_id = str(raw_sources[0])
            if source_id not in allowed_sources:
                raise MilestoneManifestRequired(
                    f"{scope} Step {index} uses {source_id}, which is outside its Milestone"
                )
            observed_sources.append(source_id)

            title = str(raw.get("title", "")).strip()
            expected_outcome = str(raw.get("expected_outcome", "")).strip()
            work_kind = str(raw.get("work_kind", "")).upper().strip()
            if not title or not expected_outcome or work_kind not in _PRELIMINARY_STEP_KINDS:
                raise MilestoneManifestRequired(
                    f"{scope} Step {index} requires title, expected_outcome and a valid work_kind"
                )
            raw_entities = raw.get("entity_refs", [])
            raw_risks = raw.get("risk_checklist", [])
            raw_dependencies = raw.get("historical_dependency_refs", [])
            if not isinstance(raw_entities, list):
                raise MilestoneManifestRequired(f"{scope} Step {index} entity_refs must be a list")
            if not isinstance(raw_risks, list):
                raise MilestoneManifestRequired(
                    f"{scope} Step {index} risk_checklist must be a list"
                )
            if not isinstance(raw_dependencies, list):
                raise MilestoneManifestRequired(
                    f"{scope} Step {index} historical_dependency_refs must be a list"
                )
            self._require_string_list(raw_entities, scope=f"{scope} Step {index} entity_refs")
            self._require_string_list(raw_risks, scope=f"{scope} Step {index} risk_checklist")
            self._require_string_list(
                raw_dependencies,
                scope=f"{scope} Step {index} historical_dependency_refs",
            )
            entities = self._normalize_entity_refs(raw_entities)
            failure_signals = tuple(
                f"Unresolved risk: {value}" for value in map(str, raw_risks)
            ) or (f"Expected outcome not observed: {expected_outcome}",)
            compiled.append(
                PlanStepSpec(
                    step_id=f"{milestone_id}.S{index:03d}",
                    title=title,
                    entity_refs=entities,
                    expected_outcome=expected_outcome,
                    minimum_acceptance=(),
                    failure_signals=failure_signals,
                    historical_dependency_refs=self._normalize_entity_refs(raw_dependencies),
                    source_plan_item_ids=(source_id,),
                )
            )

        first_occurrences = tuple(dict.fromkeys(observed_sources))
        if first_occurrences != tuple(source_plan_item_ids):
            raise MilestoneManifestRequired(
                f"{scope} Steps must cover every Milestone native address in order; expected "
                f"{tuple(source_plan_item_ids)}, got {first_occurrences}"
            )
        # These are preliminary navigation cursors. Confirmed
        # Milestone/Evidence relations are established at the stage boundary;
        # a Step never compiles its own proof obligation.
        return tuple(compiled)

    @staticmethod
    def _reconcile_preliminary_step_identities(
        *,
        milestone_id: str,
        steps: Sequence[PlanStepSpec],
        prior_steps: Sequence[PlanStepSpec],
    ) -> tuple[PlanStepSpec, ...]:
        """Reuse immutable addresses only for semantically unchanged cursors."""

        if not prior_steps:
            return tuple(steps)
        prior_ids = {step.step_id for step in prior_steps}
        used_ids: set[str] = set()
        suffixes = [
            int(match.group(1))
            for step in prior_steps
            if (
                match := re.fullmatch(
                    rf"{re.escape(milestone_id)}\.S(\d+)",
                    step.step_id,
                )
            )
        ]
        next_suffix = max(suffixes, default=0) + 1
        reconciled: list[PlanStepSpec] = []
        for candidate in steps:
            reusable = next(
                (
                    prior
                    for prior in prior_steps
                    if prior.step_id not in used_ids
                    and replace(candidate, step_id=prior.step_id) == prior
                ),
                None,
            )
            if reusable is not None:
                reconciled.append(reusable)
                used_ids.add(reusable.step_id)
                continue
            if candidate.step_id not in prior_ids and candidate.step_id not in used_ids:
                reconciled.append(candidate)
                used_ids.add(candidate.step_id)
                continue
            fresh_id = f"{milestone_id}.S{next_suffix:03d}"
            next_suffix += 1
            reconciled.append(replace(candidate, step_id=fresh_id))
            used_ids.add(fresh_id)
        return tuple(reconciled)

    @staticmethod
    def _materialize_direction(
        criterion: CompletionCriterionSpec,
    ) -> CompletionCriterionSpec:
        """Compile one selected semantic direction into a factual boundary."""

        if criterion.commitment_level is not CommitmentLevel.DIRECTION:
            return criterion
        if criterion.requirement_id == TASK_FINAL_REQUIREMENT_ID:
            # TASK.FINAL is intentionally the one repository-wide semantic
            # review.  It becomes addressable at the terminal boundary, but it
            # must never be rewritten into a generic TEST_RESULT contract:
            # tests establish the factual substrate while the bounded review
            # decides whether the complete immutable Task was satisfied.
            return replace(
                criterion,
                commitment_level=CommitmentLevel.MILESTONE,
                required_evidence_types=(),
                entity_refs=(),
                test_selectors=(),
                verification_mode=CriterionVerificationMode.SEMANTIC,
            )
        compiled = compile_behavioral_commitment(
            {
                "requirement_text": criterion.requirement_text,
                "observable_outcome": criterion.observable_outcome,
                "claim_type": criterion.claim_type.value,
            },
            criterion_id=criterion.criterion_id,
            entity_refs=criterion.entity_refs,
            test_selectors=(),
            level=CommitmentLevel.MILESTONE,
        )
        return replace(compiled, requirement_id=criterion.requirement_id)

    @staticmethod
    def _navigation_step(
        *,
        milestone_id: str,
        title: str,
        target_outcome: str,
        source_plan_item_ids: Sequence[str],
        criteria: Sequence[CompletionCriterionSpec],
    ) -> PlanStepSpec:
        """Create the sole runtime-owned route owner for a stage skeleton.

        This Step contains no repository strategy and opens no Provider Turn.
        It simply gives naturally produced execution Evidence a stable TPG
        address until the Milestone boundary is reached.
        """

        return PlanStepSpec(
            step_id=f"{milestone_id}.S001",
            title=title,
            criterion_ids=tuple(item.criterion_id for item in criteria),
            entity_refs=tuple(
                dict.fromkeys(entity for item in criteria for entity in item.entity_refs)
            ),
            expected_outcome=target_outcome,
            minimum_acceptance=(),
            source_plan_item_ids=tuple(source_plan_item_ids),
        )

    def _with_trusted_verification(
        self,
        criteria: Sequence[CompletionCriterionSpec],
        *,
        canonical_id: str,
    ) -> tuple[CompletionCriterionSpec, ...]:
        """Attach runtime-owned benchmark selectors without model transcription."""

        compiled = list(criteria)
        existing_selectors = {
            selector for criterion in compiled for selector in criterion.test_selectors
        }
        for selector in self._trusted_verification_selectors:
            if selector in existing_selectors:
                continue
            ordinal = len(compiled) + 1
            compiled.append(
                CompletionCriterionSpec(
                    criterion_id=f"{canonical_id}.C{ordinal:03d}",
                    requirement_id=f"{canonical_id}.R{ordinal:03d}",
                    requirement_text=(
                        f"The benchmark contract requires {selector} to pass on the "
                        "final repository revision"
                    ),
                    observable_outcome=(
                        f"The runtime-owned final verification passes {selector} on the "
                        "current repository revision"
                    ),
                    claim_type=ClaimType.REGRESSION,
                    required_evidence_types=(FactType.VERIFIER_RESULT,),
                    test_selectors=(selector,),
                    required=True,
                    commitment_level=CommitmentLevel.MILESTONE,
                )
            )
        return tuple(compiled)

    @staticmethod
    def _with_final_task_direction(
        criteria: Sequence[CompletionCriterionSpec],
        *,
        canonical_id: str,
        user_task: str,
    ) -> tuple[CompletionCriterionSpec, ...]:
        """Preserve the immutable Task check without precompiling Evidence."""

        compiled = list(criteria)
        if any(item.requirement_id == "TASK.FINAL" for item in compiled):
            return tuple(compiled)
        ordinal = len(compiled) + 1
        compiled.append(
            CompletionCriterionSpec(
                criterion_id=f"{canonical_id}.C{ordinal:03d}",
                requirement_id="TASK.FINAL",
                requirement_text=" ".join(user_task.split()),
                observable_outcome=(
                    "The final repository state satisfies every explicit requirement in the "
                    "original Task, and the tests that actually ran do not contradict it"
                ),
                claim_type=ClaimType.REGRESSION,
                required=True,
                commitment_level=CommitmentLevel.DIRECTION,
            )
        )
        return tuple(compiled)

    @staticmethod
    def _with_final_task_review(
        criteria: Sequence[CompletionCriterionSpec],
        *,
        canonical_id: str,
        user_task: str,
    ) -> tuple[CompletionCriterionSpec, ...]:
        """Add one lightweight final check against the original Task.

        This is not an Oracle or proof language. Executable Milestone criteria
        establish the current-revision facts, then one bounded model review
        compares the final repository state with the immutable Task. Earlier
        Milestones remain historical; a detected gap is corrected under the
        terminal integration Milestone.
        """

        compiled = list(criteria)
        if any(item.requirement_id == "TASK.FINAL" for item in compiled):
            return tuple(compiled)
        ordinal = len(compiled) + 1
        compiled.append(
            CompletionCriterionSpec(
                criterion_id=f"{canonical_id}.C{ordinal:03d}",
                requirement_id="TASK.FINAL",
                requirement_text=" ".join(user_task.split()),
                observable_outcome=(
                    "The final repository state satisfies every explicit requirement in the "
                    "original Task, and the tests that actually ran do not contradict it"
                ),
                claim_type=ClaimType.REGRESSION,
                required=True,
                commitment_level=CommitmentLevel.MILESTONE,
                verification_mode=CriterionVerificationMode.SEMANTIC,
            )
        )
        return tuple(compiled)

    @staticmethod
    def _semantic_claim_signature(
        criterion: CompletionCriterionSpec,
    ) -> tuple[str, str, ClaimType]:
        return (
            " ".join(criterion.requirement_text.split()),
            " ".join(criterion.observable_outcome.split()),
            criterion.claim_type,
        )

    def _assert_directions_preserved(
        self,
        directions: Sequence[CompletionCriterionSpec],
        executable: Sequence[CompletionCriterionSpec],
        *,
        scope: str,
    ) -> None:
        """Prevent activation or replanning from silently weakening a stage."""

        required = {
            self._semantic_claim_signature(item)
            for item in directions
            if item.required
            and not set(item.test_selectors).intersection(
                self._trusted_verification_selectors
            )
        }
        available = {self._semantic_claim_signature(item) for item in executable}
        missing = required.difference(available)
        if missing:
            raise MilestoneManifestRequired(
                f"{scope} must preserve every semantic acceptance direction; "
                f"missing {sorted((requirement, outcome, kind.value) for requirement, outcome, kind in missing)}"
            )

    @staticmethod
    def _final_acceptance_from_terminal(
        criteria: Sequence[CompletionCriterionSpec],
    ) -> tuple[CompletionCriterionSpec, ...]:
        """Compatibility helper for imported single-Milestone plans."""

        return tuple(
            CompletionCriterionSpec(
                # Final acceptance is another view of the terminal task
                # contract, not a second proof address.  Keeping the stable
                # Criterion ID lets the same current-revision test receipt
                # support Milestone and task completion without copying or
                # heuristically rebinding Evidence.
                criterion_id=criterion.criterion_id,
                requirement_id=criterion.requirement_id,
                requirement_text=criterion.requirement_text,
                observable_outcome=criterion.observable_outcome,
                claim_type=criterion.claim_type,
                required_evidence_types=criterion.required_evidence_types,
                entity_refs=criterion.entity_refs,
                test_selectors=criterion.test_selectors,
                required=criterion.required,
                commitment_level=criterion.commitment_level,
                verification_mode=criterion.verification_mode,
            )
            for criterion in criteria
            if criterion.commitment_level
            in {CommitmentLevel.DIRECTION, CommitmentLevel.MILESTONE}
        )

    @classmethod
    def _final_acceptance_from_milestones(
        cls,
        milestones: Sequence[MilestoneSpec],
    ) -> tuple[CompletionCriterionSpec, ...]:
        addressed = tuple(
            (milestone.canonical_id, criterion)
            for milestone in milestones
            for criterion in cls._final_acceptance_from_terminal(
                milestone.minimum_acceptance
            )
        )
        variants: dict[str, set[CompletionCriterionSpec]] = {}
        for _milestone_id, criterion in addressed:
            variants.setdefault(criterion.criterion_id, set()).add(criterion)
        conflicting_ids = {
            criterion_id
            for criterion_id, definitions in variants.items()
            if len(definitions) > 1
        }
        indexed: dict[str, CompletionCriterionSpec] = {}
        for milestone_id, criterion in addressed:
            # Provider manifests commonly use Milestone-local IDs such as C1
            # in every stage. They are stable only inside that Milestone. Keep
            # a globally unique ID unchanged when possible; qualify only a
            # genuinely conflicting local ID so task-wide aggregation does
            # not reject an otherwise valid route or merge different claims.
            qualified = (
                replace(
                    criterion,
                    criterion_id=f"{milestone_id}.{criterion.criterion_id}",
                )
                if criterion.criterion_id in conflicting_ids
                else criterion
            )
            existing = indexed.get(qualified.criterion_id)
            if existing is not None and existing != qualified:
                raise MilestoneManifestRequired(
                    "Task-wide final acceptance contains conflicting qualified Criterion IDs: "
                    f"{qualified.criterion_id}"
                )
            indexed.setdefault(qualified.criterion_id, qualified)
        return tuple(indexed.values())

    def _projection_criterion(
        self,
        value: object,
        *,
        criterion_id: str,
        scope: str,
        terminal: bool,
    ) -> CompletionCriterionSpec:
        if not isinstance(value, Mapping):
            raise MilestoneManifestRequired(f"{scope} contains a non-object criterion")
        entity_refs = value.get("entity_refs")
        test_selectors = value.get("test_selectors")
        if not isinstance(entity_refs, list) or not isinstance(test_selectors, list):
            raise MilestoneManifestRequired(f"{scope} addresses must be JSON lists")
        self._require_string_list(entity_refs, scope=f"{scope} entity_refs")
        self._require_string_list(test_selectors, scope=f"{scope} test_selectors")
        normalized_entities = self._normalize_entity_refs(entity_refs)
        try:
            criterion = compile_behavioral_commitment(
                value,
                criterion_id=criterion_id,
                entity_refs=normalized_entities,
                test_selectors=tuple(map(str, test_selectors)),
                level=(
                    CommitmentLevel.MILESTONE
                    if terminal
                    else CommitmentLevel.STEP
                ),
            )
        except ValueError as error:
            raise MilestoneManifestRequired(
                f"{scope} has an invalid behavioral commitment: {error}"
            ) from error
        evidence_types = tuple(item.value for item in criterion.required_evidence_types)
        self._validate_acceptance_producibility(evidence_types, scope=scope)
        self._validate_result_binding(evidence_types, criterion.test_selectors, scope=scope)
        return criterion

    @staticmethod
    def native_plan_snapshot(
        steps: Sequence[Mapping[str, Any]],
        final_plan_text: str | None,
    ) -> NativePlanSnapshot | None:
        """Freeze the Harness Plan before any Homy route projection exists."""

        items = tuple(
            NativePlanItemSpec.from_dict(step, ordinal)
            for ordinal, step in enumerate(steps, start=1)
            if str(step.get("step", step.get("title", ""))).strip()
        )
        if not items:
            return None
        return NativePlanSnapshot(items=items, final_text=str(final_plan_text or ""))

    @staticmethod
    def native_plan_artifact_steps(text: str) -> tuple[Mapping[str, str], ...]:
        """Expose stable source blocks from an authoritative Codex Plan item.

        App Server's completed ``plan`` item is a Markdown artifact, not a
        structured Step array. Only its explicit ``Ordered Execution Plan``
        list defines route order; explanatory Key Changes, Test Plan, and
        Assumptions remain available in the preserved full text but cannot be
        mistaken for execution Steps.
        """

        source = text.strip()
        if not source:
            return ()
        try:
            decoded = json.loads(source)
        except json.JSONDecodeError:
            decoded = None
        if isinstance(decoded, Mapping) and isinstance(decoded.get("milestones"), list):
            # A projection serialized through a Plan-mode final item is Homy
            # output, not a replacement for the already observed native Plan.
            return ()

        # (indent, section ordinal, section heading key, numbered?, body)
        list_entries: list[tuple[int, int, str, bool, str]] = []
        section = ""
        section_ordinal = 0
        in_fence = False

        for raw_line in source.splitlines():
            stripped = raw_line.strip()
            if stripped.startswith("```") or stripped.startswith("~~~"):
                in_fence = not in_fence
                continue
            if in_fence:
                continue
            heading = _MARKDOWN_HEADING.match(raw_line)
            if heading is not None:
                level = len(heading.group(1))
                if level >= 2:
                    section = _sequence_heading_key(heading.group(2))
                    section_ordinal += 1
                continue
            listed = _MARKDOWN_LIST_ITEM.match(raw_line)
            if listed is not None:
                list_entries.append(
                    (
                        len(listed.group(1).expandtabs(4)),
                        section_ordinal,
                        section,
                        _MARKDOWN_ORDERED_ITEM.match(raw_line) is not None,
                        listed.group(2).strip(),
                    )
                )

        ordered = [
            entry for entry in list_entries if entry[2] in _NATIVE_PLAN_SEQUENCE_HEADINGS
        ]
        if not ordered and list_entries:
            if all(not entry[2] for entry in list_entries):
                # A bare top-level list is already an unambiguous ordered Plan.
                ordered = list_entries
            else:
                # Structural fallback for a translated or paraphrased heading:
                # the Plan is asked to *end* with one numbered execution list,
                # so the last section whose top-level items are all numbered
                # is the route source.  Bulleted Key Changes / Test Plan /
                # Assumptions sections never qualify.
                by_section: dict[int, list[tuple[int, int, str, bool, str]]] = {}
                for entry in list_entries:
                    by_section.setdefault(entry[1], []).append(entry)
                for ordinal in sorted(by_section, reverse=True):
                    entries = by_section[ordinal]
                    top = min(entry[0] for entry in entries)
                    top_level = [entry for entry in entries if entry[0] == top]
                    if len(top_level) >= 2 and all(entry[3] for entry in top_level):
                        ordered = entries
                        break
        if not ordered:
            return ()
        top_indent = min(entry[0] for entry in ordered)
        labels = [entry[4] for entry in ordered if entry[0] == top_indent]

        return tuple(
            {
                "source_step_id": f"N{ordinal:03d}",
                "step": " ".join(label.split()),
                "status": "pending",
            }
            for ordinal, label in enumerate(labels, start=1)
            if label.strip()
        )

    @staticmethod
    def _step_status(value: str) -> PlanStepStatus:
        return {
            "pending": PlanStepStatus.PENDING,
            "inprogress": PlanStepStatus.IN_PROGRESS,
            "in_progress": PlanStepStatus.IN_PROGRESS,
            "completed": PlanStepStatus.COMPLETED_CLAIMED,
            "completed_claimed": PlanStepStatus.COMPLETED_CLAIMED,
            "completed_verified": PlanStepStatus.COMPLETED_VERIFIED,
            "failed": PlanStepStatus.FAILED,
            "cancelled": PlanStepStatus.CANCELLED,
        }.get(value.replace("-", "_").casefold(), PlanStepStatus.PENDING)

    @staticmethod
    def canonical_step_address(milestone_id: str, step_id: str) -> str:
        """Translate a Provider-local Step ID into the graph's global address space."""

        milestone = milestone_id.strip().upper()
        raw = step_id.strip()
        if not raw:
            raise MilestoneManifestRequired(f"{milestone} PlanStep has an empty step_id")
        scoped = re.fullmatch(r"(M\d{3,})[.-](.+)", raw, re.IGNORECASE)
        if scoped is not None:
            owner = scoped.group(1).upper()
            if owner != milestone:
                raise MilestoneManifestRequired(
                    f"{milestone} PlanStep {raw!r} is scoped to a different Milestone"
                )
            local = scoped.group(2).strip()
        else:
            if re.match(r"M\d{3,}", raw, re.IGNORECASE):
                raise MilestoneManifestRequired(
                    f"{milestone} PlanStep {raw!r} has an invalid Milestone-qualified address"
                )
            local = raw
        if not local:
            raise MilestoneManifestRequired(f"{milestone} PlanStep has an empty local address")
        return f"{milestone}.{local}"

    @staticmethod
    def _manifest(text: str | None) -> Mapping[str, Any] | None:
        if not text:
            return None
        candidates = [match.group(1) for match in _JSON_FENCE.finditer(text)]
        stripped = text.strip()
        if stripped.startswith("{"):
            candidates.append(stripped)
        for raw in reversed(candidates):
            try:
                value = json.loads(raw)
            except json.JSONDecodeError:
                continue
            if not isinstance(value, Mapping):
                continue
            nested = value.get("milestone_manifest", value.get("MilestoneManifest", value))
            if isinstance(nested, Mapping) and isinstance(nested.get("milestones"), list):
                return nested
        return None

    def _from_manifest(
        self,
        *,
        user_task: str,
        manifest: Mapping[str, Any],
        native_plan: NativePlanSnapshot | None = None,
    ) -> PlanSpec:
        manifest = self._expand_compact_manifest(manifest)
        encoded_size = len(
            json.dumps(
                manifest,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        )
        if encoded_size > _MILESTONE_MANIFEST_BYTE_LIMIT:
            raise MilestoneManifestRequired(
                "MilestoneManifest exceeds the operational serialized-size safety limit"
            )
        raw_milestones = manifest.get("milestones")
        if not isinstance(raw_milestones, list) or not raw_milestones:
            raise MilestoneManifestRequired("MilestoneManifest contains no milestones")
        if len(raw_milestones) > _MILESTONE_OPERATIONAL_SAFETY_LIMIT:
            raise MilestoneManifestRequired(
                "MilestoneManifest exceeds the operational node-count safety limit"
            )
        implicit_native_projection: dict[int, tuple[str, ...]] = {}
        if native_plan is not None and not any(
            isinstance(item, Mapping)
            and bool(item.get("source_plan_item_ids") or item.get("source_plan_items"))
            for item in raw_milestones
        ):
            # Keep the adapter lightweight in the two unambiguous cases.  A
            # single stage owns the whole native Plan; equal counts project
            # one native item to each stage in order.  Only actual grouping or
            # splitting decisions require the model to declare boundaries.
            if len(raw_milestones) == 1:
                implicit_native_projection[1] = tuple(
                    item.source_step_id for item in native_plan.items
                )
            elif len(raw_milestones) == len(native_plan.items):
                implicit_native_projection = {
                    ordinal: (item.source_step_id,)
                    for ordinal, item in enumerate(native_plan.items, start=1)
                }
        milestones: list[MilestoneSpec] = []
        for ordinal, raw in enumerate(raw_milestones, start=1):
            if not isinstance(raw, Mapping):
                raise MilestoneManifestRequired("MilestoneManifest node is not an object")
            canonical = str(raw.get("canonical_id") or raw.get("milestone_id") or f"M{ordinal:03d}")
            match = re.fullmatch(r"M\d{3,}", canonical.upper())
            if match is None:
                raise MilestoneManifestRequired("MilestoneManifest contains an invalid stable ID")
            steps_value = raw.get("steps", ())
            if not isinstance(steps_value, list):
                raise MilestoneManifestRequired(f"{canonical} PlanSteps must be a JSON list")
            if native_plan is None and not steps_value:
                raise MilestoneManifestRequired(f"{canonical} contains no PlanSteps")
            if native_plan is not None:
                if ordinal == 1 and not steps_value:
                    raise MilestoneManifestRequired(
                        "the first active Milestone requires concrete PlanSteps"
                    )
                if ordinal > 1 and steps_value:
                    raise MilestoneManifestRequired(
                        f"{canonical} is a future Milestone; materialize its PlanSteps only "
                        "after the preceding Milestone review"
                    )
            steps: list[PlanStepSpec] = []
            for index, step in enumerate(steps_value, start=1):
                if isinstance(step, str) and step.strip():
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep must be a structured navigation object"
                    )
                if not isinstance(step, Mapping):
                    raise MilestoneManifestRequired(f"{canonical} has an invalid PlanStep")
                step_criterion_ids = step.get("criterion_ids", [])
                if not isinstance(step_criterion_ids, list) or not all(
                    isinstance(value, str) and value.strip() for value in step_criterion_ids
                ):
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep criterion_ids must be a JSON string list"
                    )
                expected_outcome = str(step.get("expected_outcome", "")).strip()
                failure_signals = step.get("failure_signals", ())
                historical_dependency_refs = step.get("historical_dependency_refs", [])
                if not expected_outcome:
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep requires an expected_outcome"
                    )
                if not isinstance(failure_signals, list) or not failure_signals:
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep requires bounded failure_signals"
                    )
                self._require_string_list(
                    failure_signals,
                    scope=f"{canonical} PlanStep failure_signals",
                )
                if not isinstance(historical_dependency_refs, list):
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep historical_dependency_refs must be a JSON list"
                    )
                self._require_string_list(
                    historical_dependency_refs,
                    scope=f"{canonical} PlanStep historical_dependency_refs",
                )
                step_spec = PlanStepSpec(
                    step_id=self.canonical_step_address(
                        canonical,
                        str(step.get("step_id") or f"S{index:03d}"),
                    ),
                    title=str(step.get("title") or step.get("step") or ""),
                    status=self._step_status(str(step.get("status", "PENDING"))),
                    criterion_ids=tuple(step_criterion_ids),
                    entity_refs=self._normalize_entity_refs(step.get("entity_refs", ())),
                    expected_outcome=expected_outcome,
                    # Old supplied manifests may still contain a Step-local
                    # acceptance field. It is deliberately ignored: a Step is
                    # navigation/provenance only, never a proof authority.
                    minimum_acceptance=(),
                    failure_signals=tuple(map(str, failure_signals)),
                    historical_dependency_refs=self._normalize_entity_refs(
                        historical_dependency_refs
                    ),
                )
                steps.append(step_spec)
            criteria = raw.get(
                "minimum_acceptance",
                raw.get("completion_criteria", ()),
            )
            verification = raw.get("verification", ())
            if isinstance(verification, str) and verification.strip():
                verification = [verification.strip()]
            if not isinstance(criteria, list) or not criteria:
                raise MilestoneManifestRequired(
                    f"{canonical} has no observable completion criteria"
                )
            if not isinstance(verification, list) or not verification:
                raise MilestoneManifestRequired(f"{canonical} has no verification method")
            self._require_string_list(verification, scope=f"{canonical} verification")
            milestone_entities = raw.get("entity_refs", ())
            if not isinstance(milestone_entities, list):
                raise MilestoneManifestRequired(f"{canonical} entity_refs must be a JSON list")
            normalized_milestone_entities = self._normalize_entity_refs(milestone_entities)
            normalized_criteria: list[dict[str, Any]] = []
            for criterion in criteria:
                if (
                    not isinstance(criterion, Mapping)
                    or not str(criterion.get("observable_outcome", "")).strip()
                ):
                    raise MilestoneManifestRequired(
                        f"{canonical} minimum acceptance must contain structured observable facts"
                    )
                evidence_types = criterion.get("required_evidence_types", ())
                if not isinstance(evidence_types, list) or not evidence_types:
                    raise MilestoneManifestRequired(
                        f"{canonical} minimum acceptance requires typed Evidence"
                    )
                self._require_string_list(
                    evidence_types,
                    scope=f"{canonical} required_evidence_types",
                )
                self._validate_evidence_types(evidence_types, scope=canonical)
                self._validate_acceptance_producibility(
                    evidence_types,
                    scope=f"{canonical} minimum_acceptance",
                )
                criterion_entities = criterion.get("entity_refs", milestone_entities)
                selectors = criterion.get("test_selectors", [])
                if not isinstance(criterion_entities, list) or not isinstance(selectors, list):
                    raise MilestoneManifestRequired(
                        f"{canonical} criterion selectors must be JSON lists"
                    )
                self._require_string_list(
                    criterion_entities,
                    scope=f"{canonical} criterion entity_refs",
                )
                self._require_string_list(
                    selectors,
                    scope=f"{canonical} criterion test_selectors",
                )
                if not criterion_entities and not selectors:
                    raise MilestoneManifestRequired(
                        f"{canonical} criterion needs entity_refs or test_selectors; "
                        "implicit binding is forbidden"
                    )
                self._validate_result_binding(
                    evidence_types,
                    selectors,
                    scope=f"{canonical} criterion",
                )
                normalized_criteria.append(
                    {
                        **dict(criterion),
                        "entity_refs": list(self._normalize_entity_refs(criterion_entities)),
                        "test_selectors": list(map(str, selectors)),
                    }
                )
            milestone_criterion_specs = tuple(
                CompletionCriterionSpec.from_value(item, index)
                for index, item in enumerate(normalized_criteria, start=1)
            )
            milestone_criterion_ids = {
                criterion.criterion_id for criterion in milestone_criterion_specs
            }
            if len(milestone_criterion_ids) != len(milestone_criterion_specs):
                raise MilestoneManifestRequired(
                    f"{canonical} Milestone acceptance criterion IDs must be unique"
                )
            for step in steps:
                unknown_mappings = sorted(
                    set(step.criterion_ids).difference(milestone_criterion_ids)
                )
                if unknown_mappings:
                    raise MilestoneManifestRequired(
                        f"{canonical} PlanStep {step.step_id} criterion_ids must reference "
                        "criterion_id values in the enclosing Milestone minimum_acceptance; "
                        f"unknown mappings: {unknown_mappings}. Local Step criterion IDs belong "
                        "in minimum_acceptance[].criterion_id"
                    )
            title = str(raw.get("title", "")).strip()
            objective = str(raw.get("objective", "")).strip()
            target_outcome = str(raw.get("target_outcome", "")).strip()
            if not title or not objective or not target_outcome:
                raise MilestoneManifestRequired(
                    f"{canonical} is missing title/objective/target_outcome"
                )
            for list_field in ("downstream_assumptions", "non_goals"):
                if not isinstance(raw.get(list_field, ()), list):
                    raise MilestoneManifestRequired(f"{canonical} {list_field} must be a JSON list")
                self._require_string_list(
                    raw.get(list_field, ()),
                    scope=f"{canonical} {list_field}",
                )
            milestone = MilestoneSpec.from_dict(
                {
                    **dict(raw),
                    "canonical_id": canonical.upper(),
                    "description": str(raw.get("description") or objective),
                    "scope": str(raw.get("scope") or objective),
                    "entity_refs": normalized_milestone_entities,
                    "verification": verification,
                    "source_plan_item_ids": self._resolve_native_plan_items(
                        raw,
                        native_plan=native_plan,
                        canonical_id=canonical,
                        implicit_ids=implicit_native_projection.get(ordinal, ()),
                    ),
                    "minimum_acceptance": normalized_criteria,
                    "steps": [
                        {
                            "step_id": item.step_id,
                            "title": item.title,
                            "status": item.status.value,
                            "criterion_ids": item.criterion_ids,
                            "entity_refs": item.entity_refs,
                            "expected_outcome": item.expected_outcome,
                            "minimum_acceptance": primitive(item.minimum_acceptance),
                            "failure_signals": item.failure_signals,
                            "historical_dependency_refs": (item.historical_dependency_refs),
                        }
                        for item in tuple(steps)
                    ],
                }
            )
            milestones.append(milestone)
        raw_final = manifest.get("final_acceptance", ())
        if isinstance(raw_final, Mapping):
            raw_final = [raw_final]
        if not isinstance(raw_final, list) or not raw_final:
            raise MilestoneManifestRequired(
                "MilestoneManifest requires structured final_acceptance"
            )
        normalized_final: list[dict[str, Any]] = []
        for criterion in raw_final:
            if (
                not isinstance(criterion, Mapping)
                or not str(criterion.get("observable_outcome", "")).strip()
            ):
                raise MilestoneManifestRequired(
                    "final_acceptance must contain structured observable facts"
                )
            evidence_types = criterion.get("required_evidence_types", ())
            entity_refs = criterion.get("entity_refs", [])
            selectors = criterion.get("test_selectors", [])
            if not isinstance(evidence_types, list) or not evidence_types:
                raise MilestoneManifestRequired("final_acceptance requires typed Evidence")
            self._require_string_list(
                evidence_types,
                scope="final_acceptance required_evidence_types",
            )
            self._validate_evidence_types(evidence_types, scope="final_acceptance")
            self._validate_acceptance_producibility(
                evidence_types,
                scope="final_acceptance",
            )
            if not isinstance(entity_refs, list) or not isinstance(selectors, list):
                raise MilestoneManifestRequired("final_acceptance selectors must be JSON lists")
            self._require_string_list(
                entity_refs,
                scope="final_acceptance entity_refs",
            )
            self._require_string_list(
                selectors,
                scope="final_acceptance test_selectors",
            )
            if not entity_refs and not selectors:
                raise MilestoneManifestRequired(
                    "final_acceptance needs entity_refs or test_selectors"
                )
            self._validate_result_binding(
                evidence_types,
                selectors,
                scope="final_acceptance",
            )
            normalized_final.append(
                {
                    **dict(criterion),
                    "entity_refs": list(self._normalize_entity_refs(entity_refs)),
                    "test_selectors": list(map(str, selectors)),
                }
            )
        final_verification = manifest.get(
            "final_verification",
            ["Verify every required Milestone and the final repository state"],
        )
        if isinstance(final_verification, str) and final_verification.strip():
            final_verification = [final_verification.strip()]
        if not isinstance(final_verification, list) or not final_verification:
            raise MilestoneManifestRequired("final_verification must be a non-empty JSON list")
        self._require_string_list(final_verification, scope="final_verification")
        final_acceptance = tuple(
            CompletionCriterionSpec.from_value(
                item,
                index,
                default_level=CommitmentLevel.MILESTONE,
            )
            for index, item in enumerate(normalized_final, start=1)
        )
        self._validate_trusted_verification_coverage(milestones=milestones)
        self._validate_stage_granularity(milestones=milestones)
        self._validate_final_evidence_coverage(
            milestones=milestones,
            final_acceptance=final_acceptance,
        )
        self._validate_native_plan_projection(
            milestones=milestones,
            native_plan=native_plan,
        )
        return PlanSpec(
            goal=str(manifest.get("goal") or user_task).strip(),
            milestones=tuple(milestones),
            final_verification=tuple(map(str, final_verification)),
            final_acceptance=final_acceptance,
            native_plan=native_plan,
        )

    @staticmethod
    def _resolve_native_plan_items(
        raw_milestone: Mapping[str, Any],
        *,
        native_plan: NativePlanSnapshot | None,
        canonical_id: str,
        allow_empty: bool = False,
        implicit_ids: tuple[str, ...] = (),
    ) -> tuple[str, ...]:
        explicit_ids = raw_milestone.get("source_plan_item_ids", ())
        if explicit_ids:
            if not isinstance(explicit_ids, list):
                raise MilestoneManifestRequired(
                    f"{canonical_id} source_plan_item_ids must be a JSON list"
                )
            return tuple(map(str, explicit_ids))
        raw_titles = raw_milestone.get("source_plan_items", ())
        if native_plan is None:
            return ()
        if implicit_ids:
            return implicit_ids
        if allow_empty and raw_titles in ((), []):
            return ()
        if not isinstance(raw_titles, list) or not raw_titles:
            raise MilestoneManifestRequired(
                f"{canonical_id} must map exact native Plan item titles"
            )
        by_title: dict[str, list[NativePlanItemSpec]] = {}
        for item in native_plan.items:
            by_title.setdefault(" ".join(item.title.casefold().split()), []).append(item)
        resolved: list[str] = []
        for title in raw_titles:
            normalized = " ".join(str(title).casefold().split())
            matches = by_title.get(normalized, ())
            if len(matches) != 1:
                raise MilestoneManifestRequired(
                    f"{canonical_id} native Plan mapping is "
                    + ("ambiguous" if matches else "unknown")
                    + f": {title!r}"
                )
            resolved.append(matches[0].source_step_id)
        return tuple(dict.fromkeys(resolved))

    @staticmethod
    def _validate_native_plan_projection(
        *,
        milestones: Sequence[MilestoneSpec],
        native_plan: NativePlanSnapshot | None,
    ) -> None:
        if native_plan is None:
            return
        expected = tuple(item.source_step_id for item in native_plan.items)
        projected = tuple(
            source_id for milestone in milestones for source_id in milestone.source_plan_item_ids
        )
        if projected != expected:
            raise MilestoneManifestRequired(
                "Milestone projection must preserve every native Plan item exactly once and "
                "in original order; "
                f"expected {expected}, got {projected}"
            )

    def _validate_trusted_verification_coverage(
        self,
        *,
        milestones: Sequence[MilestoneSpec],
    ) -> None:
        """Require a complete, unambiguous Milestone partition of benchmark targets.

        The benchmark owns the target set; the model owns only its semantic
        grouping into Milestones. Source paths, prose, and a convenient subset
        cannot replace exact result addresses at this trust boundary.
        """

        expected = set(self._trusted_verification_selectors)
        if not expected:
            return
        result_types = {
            FactType.TOOL_RESULT,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
        owners: dict[str, list[str]] = {selector: [] for selector in expected}
        unexpected: list[str] = []
        for milestone in milestones:
            for criterion in milestone.minimum_acceptance:
                if not criterion.required or not set(
                    criterion.required_evidence_types
                ).intersection(result_types):
                    continue
                for selector in criterion.test_selectors:
                    if selector in expected:
                        owners[selector].append(
                            f"{milestone.canonical_id}/{criterion.criterion_id}"
                        )
                    elif selector != "verify_current_milestone":
                        unexpected.append(
                            f"{milestone.canonical_id}/{criterion.criterion_id}:{selector}"
                        )
        missing = sorted(selector for selector, values in owners.items() if not values)
        duplicated = {selector: values for selector, values in owners.items() if len(values) > 1}
        problems: list[str] = []
        if missing:
            problems.append(f"missing exact targets: {missing}")
        if duplicated:
            problems.append(f"targets assigned more than once: {duplicated}")
        if unexpected:
            problems.append(f"unexpected host verification selectors: {sorted(unexpected)}")
        if problems:
            raise MilestoneManifestRequired(
                "trusted benchmark acceptance must be partitioned exactly once across required "
                "Milestone minimum_acceptance criteria; " + "; ".join(problems)
            )

    @staticmethod
    def _validate_stage_granularity(*, milestones: Sequence[MilestoneSpec]) -> None:
        """Reject execution-step chains disguised as stable Milestone nodes.

        The rule is based on the typed acceptance contract, not English verbs
        in titles. A multi-node mutation Plan needs a durable outcome at every
        stage; commands and verification results alone are work inside such a
        stage. The node that changes repository state must also carry its own
        post-change evidence rather than delegating acceptance to a later
        "run tests" node.

        Read-only tasks and a single verification-only task remain valid. A
        genuine architecture/decision stage can remain separate through
        IMPLEMENTATION_DECISION evidence produced during execution.
        """

        unsound_contracts: list[str] = []
        for milestone in milestones:
            for criterion in milestone.minimum_acceptance:
                if (
                    not milestone.steps
                    and criterion.commitment_level is CommitmentLevel.DIRECTION
                ):
                    # A pending skeleton preserves meaning only. Its executable
                    # contract is compiled atomically with its first Steps.
                    continue
                reasons = criterion_contract_rejections(primitive(criterion))
                if reasons:
                    unsound_contracts.append(
                        f"{milestone.canonical_id}/{criterion.criterion_id}:" + ",".join(reasons)
                    )
        if unsound_contracts:
            raise MilestoneManifestRequired(
                "acceptance claim strength is not matched by its Evidence type: "
                + "; ".join(unsound_contracts)
            )

        materialized = tuple(milestone for milestone in milestones if milestone.steps)
        evidence_by_milestone = {
            milestone.canonical_id: frozenset(
                evidence_type
                for criterion in milestone.minimum_acceptance
                if criterion.required
                for evidence_type in criterion.required_evidence_types
            )
            for milestone in materialized
        }
        result_evidence = {
            FactType.TOOL_RESULT,
            FactType.TEST_RESULT,
            FactType.TEST_FAILURE,
            FactType.VERIFIER_RESULT,
        }
        for milestone in materialized:
            required_criteria = {
                criterion.criterion_id: frozenset(criterion.required_evidence_types)
                for criterion in milestone.minimum_acceptance
                if criterion.required
            }
            impossible_failure_criteria = tuple(
                criterion_id
                for criterion_id, evidence_types in required_criteria.items()
                if FactType.TEST_FAILURE in evidence_types
            )
            if impossible_failure_criteria:
                raise MilestoneManifestRequired(
                    "required minimum_acceptance describes the completed end state and cannot "
                    f"require TEST_FAILURE in {milestone.canonical_id}: "
                    f"{impossible_failure_criteria}; keep baseline failures as historical "
                    "diagnostic Evidence"
                )
            positions_by_criterion: dict[str, list[int]] = {}
            for position, step in enumerate(milestone.steps):
                for criterion_id in step.criterion_ids:
                    positions_by_criterion.setdefault(criterion_id, []).append(position)
            change_positions = [
                position
                for criterion_id, positions in positions_by_criterion.items()
                if FactType.CODE_CHANGE in required_criteria.get(criterion_id, ())
                for position in positions
            ]
            if change_positions:
                first_change_position = min(change_positions)
                pre_change_results = tuple(
                    criterion_id
                    for criterion_id, evidence_types in required_criteria.items()
                    if evidence_types.intersection(result_evidence)
                    and positions_by_criterion.get(criterion_id)
                    and max(positions_by_criterion[criterion_id]) < first_change_position
                )
                if pre_change_results:
                    raise MilestoneManifestRequired(
                        f"{milestone.canonical_id} binds required result criteria "
                        f"{pre_change_results} only before its repository change; "
                        "minimum_acceptance is the terminal stage state, while reproduction and "
                        "other pre-change checks are PlanStep evidence"
                    )
        mutation_nodes = tuple(
            milestone
            for milestone in materialized
            if FactType.CODE_CHANGE in evidence_by_milestone[milestone.canonical_id]
        )
        if not mutation_nodes:
            return

        post_change_evidence = {
            FactType.CODE_OBSERVATION,
            FactType.TOOL_RESULT,
            FactType.TEST_RESULT,
            FactType.VERIFIER_RESULT,
        }
        unverified_mutations = tuple(
            milestone.canonical_id
            for milestone in mutation_nodes
            if not evidence_by_milestone[milestone.canonical_id].intersection(post_change_evidence)
        )
        if unverified_mutations:
            raise MilestoneManifestRequired(
                "mutation Milestones must include their own post-change acceptance Evidence; "
                f"merge implementation and verification PlanSteps in {unverified_mutations}"
            )

        if len(materialized) == 1:
            return
        durable_stage_evidence = {
            FactType.IMPLEMENTATION_DECISION,
            FactType.CODE_CHANGE,
        }
        step_only_nodes = tuple(
            milestone.canonical_id
            for milestone in materialized
            if not evidence_by_milestone[milestone.canonical_id].intersection(
                durable_stage_evidence
            )
        )
        if step_only_nodes:
            raise MilestoneManifestRequired(
                "a mutation Plan contains command, observation, or verification-only nodes "
                f"{step_only_nodes}; these are PlanSteps inside the independently deliverable "
                "Milestone they support, not stable Milestones"
            )

    @staticmethod
    def _validate_evidence_types(values: Sequence[object], *, scope: str) -> None:
        allowed = tuple(item.value for item in FactType)
        invalid = tuple(str(value) for value in values if str(value) not in allowed)
        if invalid:
            raise MilestoneManifestRequired(
                f"{scope} uses unsupported Evidence types {invalid}; allowed: {allowed}"
            )
        selected = set(map(str, values))
        if FactType.TOOL_RESULT.value in selected and len(selected) != 1:
            raise MilestoneManifestRequired(
                f"{scope} combines generic TOOL_RESULT with another typed fact; "
                "split independently observable outcomes into separate criteria"
            )

    @staticmethod
    def _validate_acceptance_producibility(
        values: Sequence[object],
        *,
        scope: str,
    ) -> None:
        """Reject route/input facts that would make completion self-referential."""

        selected = {FactType(str(value)) for value in values}
        circular = tuple(
            sorted(item.value for item in selected.difference(_ACCEPTANCE_EVIDENCE_TYPES))
        )
        if circular:
            raise MilestoneManifestRequired(
                f"{scope} requires control/input Evidence {circular}, which cannot prove its "
                "own completion transition. Use an execution-producible observable such as "
                "IMPLEMENTATION_DECISION, CODE_CHANGE, TEST_RESULT, or VERIFIER_RESULT"
            )

    @staticmethod
    def _require_string_list(values: Sequence[object], *, scope: str) -> None:
        if not all(isinstance(value, str) and value.strip() for value in values):
            raise MilestoneManifestRequired(f"{scope} must contain only non-empty strings")

    @staticmethod
    def _validate_result_binding(
        evidence_types: Sequence[object],
        selectors: Sequence[object],
        *,
        scope: str,
        require_observable_failure: bool = False,
    ) -> None:
        """Validate one result contract against execution's Evidence classifier.

        A selector is the stable address of the fact that will satisfy a
        Criterion.  Planning and execution must therefore classify that
        address identically before the Plan enters the WAL/TPG.  Accepting a
        generic ``TOOL_RESULT`` contract for a command that execution can only
        emit as ``TEST_RESULT`` creates an impossible route state: the action
        happened, but the Criterion can never observe its typed Evidence.

        This boundary deliberately rejects the mismatch instead of mutating a
        persisted contract or teaching the acceptance kernel a compatibility alias.
        The same deterministic classifier is used when execution emits the
        fact, so there is one success definition rather than a repair path.
        """

        result_types = {
            FactType.TOOL_RESULT.value,
            FactType.TEST_RESULT.value,
            FactType.TEST_FAILURE.value,
            FactType.VERIFIER_RESULT.value,
        }
        selected = set(map(str, evidence_types))
        normalized_selectors = tuple(
            str(value).strip() for value in selectors if str(value).strip()
        )
        if selected.intersection(result_types) and not normalized_selectors:
            # The active Step owns later result Evidence causally. A selector
            # is a useful address when already known, not a prerequisite that
            # Planning must invent before a test exists.
            return

        test_selectors = tuple(
            selector
            for selector in normalized_selectors
            if command_evidence_semantics(selector).is_test_observation
        )
        non_test_selectors = tuple(
            selector for selector in normalized_selectors if selector not in test_selectors
        )
        if FactType.TOOL_RESULT.value in selected and test_selectors:
            raise MilestoneManifestRequired(
                f"{scope} declares TOOL_RESULT for test-like selectors {test_selectors}; "
                "execution deterministically emits TEST_RESULT for test runners and executable "
                "Python assertion/raise probes. Declare TEST_RESULT instead"
            )
        if (
            require_observable_failure
            and FactType.TEST_FAILURE.value in selected
            and not test_selectors
        ):
            raise MilestoneManifestRequired(
                f"{scope} requires TEST_FAILURE but no selector can produce a typed failing "
                "observation. Use an executable test target or a Python assertion/raise probe "
                "that exits nonzero while the behavior is broken; print-only inspection is "
                "diagnostic TOOL_RESULT Evidence"
            )
        if (
            selected.intersection({FactType.TEST_RESULT.value, FactType.TEST_FAILURE.value})
            and non_test_selectors
        ):
            raise MilestoneManifestRequired(
                f"{scope} declares test Evidence for non-test selectors {non_test_selectors}; "
                "use an executable test target or Python assertion/raise probe, or declare "
                "TOOL_RESULT for a non-test command"
            )

    @classmethod
    def _validate_final_evidence_coverage(
        cls,
        *,
        milestones: Sequence[MilestoneSpec],
        final_acceptance: Sequence[CompletionCriterionSpec],
    ) -> None:
        """Require every final fact to be producible by a Milestone contract.

        ``final_acceptance`` is an aggregate over verified Milestones, not a
        hidden post-execution phase.  A type-valid but unscheduled final fact
        would otherwise leave a run permanently INCOMPLETE after the Harness
        has already stopped.  Binding compatibility mirrors the deterministic
        entity/selector rules used by final verification.
        """

        result_types = {
            FactType.TOOL_RESULT,
            FactType.TEST_RESULT,
            FactType.TEST_FAILURE,
            FactType.VERIFIER_RESULT,
        }
        milestone_bindings = tuple(
            criterion
            for milestone in milestones
            for criterion in milestone.minimum_acceptance
            if criterion.required
        )
        for final in final_acceptance:
            if not final.required:
                continue
            if final.commitment_level is CommitmentLevel.DIRECTION:
                # A pending terminal skeleton preserves the immutable task
                # meaning but has no execution Evidence contract yet. The
                # criterion is compiled when that Milestone becomes current.
                continue
            rejections = criterion_contract_rejections(primitive(final))
            if rejections:
                raise MilestoneManifestRequired(
                    f"final_acceptance {final.criterion_id} has an unsound Evidence contract: "
                    + ", ".join(rejections)
                )
            for evidence_type in set(final.required_evidence_types):
                if any(
                    evidence_type in milestone.required_evidence_types
                    and cls._criterion_bindings_overlap(
                        final,
                        milestone,
                        require_selector=(
                            evidence_type in result_types
                            and bool(final.test_selectors)
                        ),
                    )
                    for milestone in milestone_bindings
                ):
                    continue
                address_kind = "selector" if evidence_type in result_types else "entity or selector"
                raise MilestoneManifestRequired(
                    f"final_acceptance {final.criterion_id} requires {evidence_type.value} "
                    f"at a compatible {address_kind}, but it is not produced by any Milestone "
                    "criterion required for completion; final_acceptance only aggregates "
                    "scheduled Milestone Evidence"
                )

    @classmethod
    def _criterion_bindings_overlap(
        cls,
        final: CompletionCriterionSpec,
        milestone: CompletionCriterionSpec,
        *,
        require_selector: bool,
    ) -> bool:
        if (
            final.requirement_id == milestone.requirement_id
            and cls._semantic_claim_signature(final)
            == cls._semantic_claim_signature(milestone)
        ):
            return True
        final_selectors = set(final.test_selectors)
        milestone_selectors = set(milestone.test_selectors)
        if final_selectors.intersection(milestone_selectors):
            return True
        if require_selector:
            return False
        final_entities = {
            alias for value in final.entity_refs for alias in cls._binding_entity_aliases(value)
        }
        milestone_entities = {
            alias for value in milestone.entity_refs for alias in cls._binding_entity_aliases(value)
        }
        return bool(final_entities.intersection(milestone_entities))

    @staticmethod
    def _binding_entity_aliases(value: str) -> set[str]:
        normalized = value.strip().replace("\\", "/")
        while normalized.startswith("./"):
            normalized = normalized[2:]
        if not normalized:
            return set()
        aliases = {normalized}
        if normalized.startswith("file:"):
            aliases.add(normalized.removeprefix("file:"))
        elif normalized.startswith("symbol:"):
            symbol_address = normalized.removeprefix("symbol:")
            if ":" in symbol_address:
                path = symbol_address.rsplit(":", 1)[0]
                aliases.update({path, f"file:{path}"})
        elif ":" not in normalized and ("/" in normalized or "." in normalized):
            aliases.add(f"file:{normalized}")
        return aliases

    @staticmethod
    def normalize_entity_refs(values: object) -> tuple[str, ...]:
        """Translate model-facing paths into stable semantic addresses.

        Planning manifests and later corrective Plan fragments share this one
        address boundary.  A model may naturally emit ``src/runtime.py``;
        persisted contracts must consistently use ``file:src/runtime.py``.
        """

        if not isinstance(values, (list, tuple)):
            raise MilestoneManifestRequired("entity_refs must be a JSON list")
        file_suffixes = {
            ".c",
            ".cc",
            ".cpp",
            ".cs",
            ".go",
            ".gradle",
            ".h",
            ".hpp",
            ".java",
            ".js",
            ".json",
            ".jsx",
            ".kt",
            ".kts",
            ".md",
            ".php",
            ".proto",
            ".py",
            ".rb",
            ".rs",
            ".sh",
            ".sql",
            ".toml",
            ".ts",
            ".tsx",
            ".xml",
            ".yaml",
            ".yml",
        }
        result: list[str] = []
        for raw in values:
            if not isinstance(raw, str) or not raw.strip():
                raise MilestoneManifestRequired("entity_refs must contain only non-empty strings")
            value = raw.strip().replace("\\", "/")
            if ":" not in value:
                leaf = value.rsplit("/", 1)[-1]
                suffix = "." + leaf.rsplit(".", 1)[-1].casefold() if "." in leaf else ""
                if "/" in value or suffix in file_suffixes or leaf in {"Dockerfile", "Makefile"}:
                    try:
                        value = f"file:{ReferenceIdentityFactory.normalize_path(value)}"
                    except ValueError as exc:
                        raise MilestoneManifestRequired(
                            f"invalid repository-relative entity_ref: {raw!r}"
                        ) from exc
            result.append(value)
        return tuple(dict.fromkeys(result))

    # Kept for the internal manifest call sites while external Plan-fragment
    # consumers use the public spelling above.
    _normalize_entity_refs = normalize_entity_refs

    def review_from_mapping(
        self,
        *,
        value: Mapping[str, Any],
        user_task: str,
        expected_milestone_id: str,
        current_plan: PlanSpec,
        milestone_status: MilestoneStatus | str = MilestoneStatus.COMPLETED_VERIFIED,
    ) -> MilestoneReviewProposal:
        """Validate requirement coverage and one bounded route decision."""

        milestone_id = str(value.get("milestone_id", "")).upper().strip()
        if milestone_id != expected_milestone_id.upper():
            raise ValueError("MilestoneReview targets a different current Milestone")
        status = MilestoneStatus(str(milestone_status))
        raw_decision = str(value.get("decision", "")).upper().strip()
        if raw_decision == "UPDATE_FUTURE":
            raw_decision = MilestoneReviewDecision.REPLAN_FUTURE.value
        try:
            decision = MilestoneReviewDecision(raw_decision)
        except ValueError as exc:
            raise ValueError("MilestoneReview decision is not supported") from exc
        if status not in {
            MilestoneStatus.COMPLETED_CLAIMED,
            MilestoneStatus.COMPLETED_VERIFIED,
            MilestoneStatus.VERIFICATION_FAILED,
        }:
            raise ValueError(f"{status.value} is not a Milestone review boundary")
        reason = " ".join(str(value.get("reason", "")).split())
        if not reason:
            raise ValueError("MilestoneReview requires an evidence-grounded reason")
        _ = user_task
        current_milestone = next(
            item for item in current_plan.milestones if item.canonical_id == milestone_id
        )
        raw_coverage = value.get("requirement_coverage", ())
        if not isinstance(raw_coverage, (list, tuple)):
            raise ValueError("requirement_coverage must be an array")
        expected = {
            self._semantic_claim_signature(item): item
            for item in current_milestone.minimum_acceptance
            if item.required and item.commitment_level is CommitmentLevel.MILESTONE
        }
        coverage: list[Mapping[str, str]] = []
        observed: set[tuple[str, str, ClaimType]] = set()
        for raw in raw_coverage:
            if not isinstance(raw, Mapping):
                raise ValueError("requirement_coverage entries must be objects")
            requirement_text = " ".join(str(raw.get("requirement_text", "")).split())
            observable_outcome = " ".join(
                str(raw.get("observable_outcome", "")).split()
            )
            evidence_summary = " ".join(str(raw.get("evidence_summary", "")).split())
            raw_status = str(raw.get("status", "")).upper().strip()
            signatures = tuple(
                signature
                for signature in expected
                if signature[0] == requirement_text and signature[1] == observable_outcome
            )
            if len(signatures) != 1:
                raise ValueError(
                    "requirement_coverage must exactly match one current requirement/outcome"
                )
            signature = signatures[0]
            if signature in observed:
                raise ValueError("requirement_coverage contains a duplicate requirement")
            if raw_status not in {"SATISFIED", "NOT_SATISFIED"} or not evidence_summary:
                raise ValueError(
                    "requirement_coverage needs status and an evidence-grounded summary"
                )
            observed.add(signature)
            coverage.append(
                {
                    "criterion_id": expected[signature].criterion_id,
                    "requirement_id": expected[signature].requirement_id,
                    "requirement_text": requirement_text,
                    "observable_outcome": observable_outcome,
                    "status": raw_status,
                    "evidence_summary": evidence_summary,
                }
            )
        missing_coverage = set(expected).difference(observed)
        if missing_coverage:
            raise ValueError("requirement_coverage must review every current requirement")
        has_gap = any(item["status"] == "NOT_SATISFIED" for item in coverage)
        if has_gap and decision is not MilestoneReviewDecision.CORRECT_CURRENT:
            raise ValueError("unmet requirements require CORRECT_CURRENT")
        if not has_gap and decision is MilestoneReviewDecision.CORRECT_CURRENT:
            raise ValueError("CORRECT_CURRENT requires at least one unmet requirement")
        if status is MilestoneStatus.VERIFICATION_FAILED and (
            decision is not MilestoneReviewDecision.CORRECT_CURRENT
        ):
            raise ValueError("a failed Milestone requires CORRECT_CURRENT")

        raw_future = value.get("future_milestones")
        if decision is MilestoneReviewDecision.CONTINUE:
            if raw_future is not None:
                raise ValueError("CONTINUE cannot replace the pending Milestone route")
            prepared_plan = current_plan
        elif decision is MilestoneReviewDecision.REPLAN_FUTURE:
            if not isinstance(raw_future, (list, tuple)) or not raw_future:
                raise ValueError(
                    "REPLAN_FUTURE requires the ordered pending future_milestones only"
                )
            prepared_plan = self._future_plan_from_delta(
                current_plan=current_plan,
                current_milestone_id=milestone_id,
                raw_future=raw_future,
            )
        else:
            if raw_future is not None:
                raise ValueError("CORRECT_CURRENT cannot replace a future Milestone")
            prepared_plan = current_plan
        raw_corrective = value.get("corrective_steps", ())
        if not isinstance(raw_corrective, (list, tuple)):
            raise ValueError("corrective_steps must be an array")
        if decision is MilestoneReviewDecision.CORRECT_CURRENT:
            corrective_steps = self.corrective_steps_from_value(
                raw_corrective,
                milestone=current_milestone,
            )
            if not corrective_steps:
                raise ValueError("CORRECT_CURRENT requires concrete corrective Steps")
            unresolved_ids = tuple(
                str(item["criterion_id"])
                for item in coverage
                if item["status"] == "NOT_SATISFIED"
            )
            corrective_steps = tuple(
                replace(step, criterion_ids=unresolved_ids) for step in corrective_steps
            )
        else:
            if raw_corrective:
                raise ValueError("only CORRECT_CURRENT may append corrective Steps")
            corrective_steps = ()
        future_plan = (
            self._materialize_next_milestone_cursor(
                prepared_plan,
                current_milestone_id=milestone_id,
            )
            if decision is not MilestoneReviewDecision.CORRECT_CURRENT
            else None
        )
        return MilestoneReviewProposal(
            milestone_id=milestone_id,
            decision=decision,
            reason=reason,
            future_plan=future_plan,
            corrective_steps=corrective_steps,
            requirement_coverage=tuple(coverage),
        )

    def _future_plan_from_delta(
        self,
        *,
        current_plan: PlanSpec,
        current_milestone_id: str,
        raw_future: Sequence[object],
    ) -> PlanSpec:
        current_index = next(
            (
                index
                for index, milestone in enumerate(current_plan.milestones)
                if milestone.canonical_id == current_milestone_id
            ),
            None,
        )
        if current_index is None:
            raise ValueError("future route delta has no current Milestone anchor")
        existing = {item.canonical_id: item for item in current_plan.milestones}
        future = tuple(
            self._future_milestone_from_value(
                raw,
                current_plan.native_plan,
                existing=existing,
                user_task=current_plan.goal,
            )
            for raw in raw_future
        )
        # The pending route is linear and only the pending suffix may change.
        # Dependencies are therefore runtime state, not another model-facing
        # graph-edit protocol.
        prior_id = current_milestone_id
        linked_future: list[MilestoneSpec] = []
        for milestone in future:
            linked_future.append(replace(milestone, depends_on=(prior_id,)))
            prior_id = milestone.canonical_id
        future = tuple(linked_future)
        terminal = future[-1]
        terminal_criteria = self._with_final_task_direction(
            terminal.minimum_acceptance,
            canonical_id=terminal.canonical_id,
            user_task=current_plan.goal,
        )
        self._assert_directions_preserved(
            current_plan.milestones[-1].minimum_acceptance,
            terminal_criteria,
            scope="replanned terminal Milestone",
        )
        terminal_with_receipt = replace(
            terminal,
            criteria=terminal_criteria,
            completion_criteria=tuple(item.observable_outcome for item in terminal_criteria),
        )
        future = (*future[:-1], terminal_with_receipt)
        candidate_milestones = (
            *current_plan.milestones[: current_index + 1],
            *future,
        )
        final_acceptance = self._final_acceptance_from_milestones(
            candidate_milestones
        )
        final_verification = tuple(
            dict.fromkeys(
                "Verify the final repository state against the original Task: "
                + item.observable_outcome
                for item in final_acceptance
            )
        )
        candidate = PlanSpec(
            goal=current_plan.goal,
            milestones=candidate_milestones,
            final_verification=final_verification,
            final_acceptance=final_acceptance,
            native_plan=current_plan.native_plan,
        )
        self._validate_stage_granularity(milestones=candidate.milestones)
        self._validate_final_evidence_coverage(
            milestones=candidate.milestones,
            final_acceptance=candidate.final_acceptance,
        )
        self._validate_native_plan_projection(
            milestones=candidate.milestones,
            native_plan=candidate.native_plan,
        )
        return candidate

    def _future_milestone_from_value(
        self,
        raw: object,
        native_plan: NativePlanSnapshot | None,
        *,
        existing: Mapping[str, MilestoneSpec],
        user_task: str,
    ) -> MilestoneSpec:
        if not isinstance(raw, Mapping):
            raise TypeError("future Milestone must be an object")
        canonical_id = str(raw.get("canonical_id", "")).upper().strip()
        if re.fullmatch(r"M\d{3,}", canonical_id) is None:
            raise ValueError("future Milestone requires a stable M### identity")
        scope_anchored = "task_requirement" in raw
        if scope_anchored:
            anchored = self._task_anchored_criterion(
                {**dict(raw), "entity_refs": []},
                user_task=user_task,
                criterion_id=f"{canonical_id}.C001",
                scope=canonical_id,
                level=CommitmentLevel.DIRECTION,
            )
            milestone_entities = anchored.entity_refs
            normalized_criteria: list[dict[str, object]] = [primitive(anchored)]
        else:
            direction_only = "acceptance_directions" in raw
            raw_criteria = raw.get(
                "acceptance_directions" if direction_only else "minimum_acceptance",
                (),
            )
            if not isinstance(raw_criteria, (list, tuple)) or not raw_criteria:
                raise ValueError(f"{canonical_id} requires acceptance directions")
            milestone_entities = self.normalize_entity_refs(raw.get("entity_refs", ()))
            normalized_criteria = []
            for ordinal, criterion in enumerate(raw_criteria, start=1):
                if not isinstance(criterion, Mapping):
                    raise TypeError(f"{canonical_id} acceptance must be an object")
                if direction_only:
                    direction = self._acceptance_direction_criterion(
                        criterion,
                        criterion_id=f"{canonical_id}.C{ordinal:03d}",
                        scope=f"{canonical_id} acceptance_directions",
                    )
                    normalized_criteria.append(primitive(direction))
                    continue
                evidence_types = criterion.get("required_evidence_types", ())
                entities = criterion.get("entity_refs", milestone_entities)
                selectors = criterion.get("test_selectors", ())
                if not isinstance(entities, (list, tuple)) or not isinstance(
                    selectors, (list, tuple)
                ):
                    raise ValueError(f"{canonical_id} acceptance selectors must be arrays")
                normalized_entities = self.normalize_entity_refs(entities)
                if not normalized_entities and not selectors:
                    raise ValueError(f"{canonical_id} acceptance requires an Evidence address")
                if isinstance(evidence_types, (list, tuple)) and evidence_types:
                    # Supplied/offline full manifests remain readable, but live
                    # review tools no longer expose these internal mechanics.
                    self._validate_evidence_types(evidence_types, scope=canonical_id)
                    self._validate_acceptance_producibility(
                        evidence_types,
                        scope=f"{canonical_id} minimum_acceptance",
                    )
                    self._validate_result_binding(
                        evidence_types,
                        selectors,
                        scope=f"{canonical_id} minimum_acceptance",
                    )
                    normalized_criteria.append(
                        {
                            **dict(criterion),
                            "entity_refs": list(normalized_entities),
                            "test_selectors": list(map(str, selectors)),
                            "required": True,
                        }
                    )
                else:
                    compiled = compile_behavioral_commitment(
                        criterion,
                        criterion_id=f"{canonical_id}.C{ordinal:03d}",
                        entity_refs=normalized_entities,
                        test_selectors=tuple(map(str, selectors)),
                        level=CommitmentLevel.MILESTONE,
                    )
                    normalized_criteria.append(primitive(compiled))
        if not milestone_entities:
            milestone_entities = tuple(
                dict.fromkeys(
                    entity
                    for criterion in normalized_criteria
                    for entity in map(str, criterion.get("entity_refs", ()))
                )
            )
        verification = raw.get("verification", ())
        if not isinstance(verification, (list, tuple)):
            raise ValueError(f"{canonical_id} verification must be an array")
        if not verification:
            verification = tuple(
                f"Check the observable outcome: {criterion['observable_outcome']}"
                for criterion in normalized_criteria
            )
        source_fields_present = bool(
            raw.get("source_plan_item_ids") or raw.get("source_plan_items")
        )
        prior = existing.get(canonical_id)
        source_plan_item_ids = (
            self._resolve_native_plan_items(
                raw,
                native_plan=native_plan,
                canonical_id=canonical_id,
                allow_empty=True,
            )
            if source_fields_present or prior is None
            else prior.source_plan_item_ids
        )
        task_requirement = (
            self._task_requirement_anchor(
                raw.get("task_requirement"),
                user_task=user_task,
                scope=canonical_id,
            )
            if scope_anchored
            else ""
        )
        target_outcome = str(
            raw.get(
                "target_outcome",
                task_requirement if scope_anchored else raw.get("objective", ""),
            )
        ).strip()
        title = (
            self._task_route_title(target_outcome)
            if scope_anchored
            else str(raw.get("title", "")).strip()
        )
        if not title or not target_outcome:
            raise ValueError(f"{canonical_id} requires title and target_outcome")
        source_titles = {
            item.source_step_id: item.title
            for item in (native_plan.items if native_plan is not None else ())
        }
        source_scope = (
            "; ".join(source_titles[item] for item in source_plan_item_ids if item in source_titles)
            or title
        )
        scope = target_outcome if scope_anchored else source_scope
        milestone = MilestoneSpec.from_dict(
            {
                "canonical_id": canonical_id,
                "title": title,
                "objective": target_outcome,
                "target_outcome": target_outcome,
                "scope": (
                    scope
                    if scope_anchored
                    else str(raw.get("scope", scope)).strip() or scope
                ),
                "description": (
                    f"Task-backed stage goal: {target_outcome}; source requirement: "
                    f"{task_requirement}"
                    if scope_anchored
                    else str(
                        raw.get("description", f"Reviewed future stage: {scope}")
                    ).strip()
                ),
                "depends_on": list(raw.get("depends_on", prior.depends_on if prior else ())),
                "downstream_assumptions": list(
                    raw.get(
                        "downstream_assumptions",
                        prior.downstream_assumptions if prior else (),
                    )
                ),
                "non_goals": list(raw.get("non_goals", ())),
                "status": MilestoneStatus.PENDING.value,
                "entity_refs": list(milestone_entities),
                "verification": list(map(str, verification)),
                "minimum_acceptance": normalized_criteria,
                "source_plan_item_ids": source_plan_item_ids,
                "steps": [],
            }
        )
        raw_steps = raw.get("steps")
        if raw_steps not in (None, []):
            raise ValueError(
                f"{canonical_id} is a pending Milestone skeleton; future Steps are "
                "materialized only when it becomes next"
            )
        return replace(milestone, steps=())

    def _materialize_next_milestone_cursor(
        self,
        plan: PlanSpec,
        *,
        current_milestone_id: str,
    ) -> PlanSpec:
        """Materialize one lightweight cursor at the real stage boundary.

        Initial planning and route review retain only semantic future
        Milestones. Once the current Milestone is verified, the immediate
        successor receives one runtime-owned navigation address. More distant
        Milestones stay skeletons, so implementation strategy is never frozen
        several stages ahead.
        """

        current_index = next(
            (
                index
                for index, milestone in enumerate(plan.milestones)
                if milestone.canonical_id == current_milestone_id
            ),
            None,
        )
        if current_index is None or current_index + 1 >= len(plan.milestones):
            return plan
        successor = plan.milestones[current_index + 1]
        if successor.steps:
            return plan
        return self._materialize_milestone_at(plan, current_index + 1)

    def materialize_milestone_contract(
        self,
        plan: PlanSpec,
        *,
        canonical_id: str,
    ) -> PlanSpec:
        """Runtime-owned contract freeze for the Milestone that becomes current.

        This applies exactly the same compilation the model-driven route review
        would apply to the immediate successor, but it is triggered by the
        runtime at the real route boundary.  A Milestone whose criteria are
        already executable is returned unchanged.
        """

        index = next(
            (
                ordinal
                for ordinal, milestone in enumerate(plan.milestones)
                if milestone.canonical_id == canonical_id
            ),
            None,
        )
        if index is None:
            raise KeyError(canonical_id)
        target = plan.milestones[index]
        needs_compilation = any(
            item.commitment_level is CommitmentLevel.DIRECTION for item in target.minimum_acceptance
        )
        if not needs_compilation:
            # Steps are navigation only; a step-less Milestone whose criteria
            # are already MILESTONE-level is an executable contract as-is.
            return plan
        return self._materialize_milestone_at(plan, index)

    def _materialize_milestone_at(self, plan: PlanSpec, target_index: int) -> PlanSpec:
        successor = plan.milestones[target_index]
        materializes_directions = any(
            item.commitment_level is CommitmentLevel.DIRECTION
            for item in successor.minimum_acceptance
        )
        executable_criteria = tuple(
            self._materialize_direction(item) for item in successor.minimum_acceptance
        )
        terminal = target_index == len(plan.milestones) - 1
        if terminal and materializes_directions:
            # The live task-anchored projection deliberately postpones both
            # the Task-final check and trusted selectors until this real route
            # boundary.  Supplied/legacy manifests already carry a complete,
            # stable final contract; extending or rebuilding it here would
            # rewrite its historical Criterion addresses.
            executable_criteria = self._with_final_task_review(
                executable_criteria,
                canonical_id=successor.canonical_id,
                user_task=plan.goal,
            )
            executable_criteria = self._with_trusted_verification(
                executable_criteria,
                canonical_id=successor.canonical_id,
            )
        executable = replace(
            successor,
            criteria=executable_criteria,
            completion_criteria=tuple(
                item.observable_outcome for item in executable_criteria
            ),
            verification=tuple(
                dict.fromkeys(
                    "Review requirement coverage and supporting execution Evidence for: "
                    + item.observable_outcome
                    for item in executable_criteria
                )
            ),
            entity_refs=tuple(
                dict.fromkeys(
                    entity
                    for item in executable_criteria
                    for entity in item.entity_refs
                )
            ),
        )
        navigation = self._reconciled_navigation_step(executable, None)
        milestones = list(plan.milestones)
        milestones[target_index] = replace(executable, steps=(navigation,))
        final_acceptance = (
            self._final_acceptance_from_milestones(milestones)
            if terminal and materializes_directions
            else plan.final_acceptance
        )
        final_verification = (
            tuple(
                dict.fromkeys(
                    "Verify the final repository state against the original Task: "
                    + item.observable_outcome
                    for item in final_acceptance
                )
            )
            if terminal and materializes_directions
            else plan.final_verification
        )
        materialized = PlanSpec(
            goal=plan.goal,
            milestones=tuple(milestones),
            final_verification=final_verification,
            final_acceptance=final_acceptance,
            native_plan=plan.native_plan,
        )
        self._validate_final_evidence_coverage(
            milestones=materialized.milestones,
            final_acceptance=materialized.final_acceptance,
        )
        if terminal:
            self._validate_trusted_verification_coverage(
                milestones=materialized.milestones
            )
        return materialized

    def _reconciled_navigation_step(
        self,
        milestone: MilestoneSpec,
        prior: MilestoneSpec | None,
    ) -> PlanStepSpec:
        """Keep a stable address or allocate one for a changed future route."""

        navigation = self._navigation_step(
            milestone_id=milestone.canonical_id,
            title=milestone.title,
            target_outcome=milestone.target_outcome,
            source_plan_item_ids=milestone.source_plan_item_ids,
            criteria=milestone.minimum_acceptance,
        )
        if prior is not None and prior.steps:
            # An unchanged pending route keeps its stable Step address. A real
            # semantic change receives a fresh immutable address; the Registry
            # cancels the superseded still-PENDING node in the Plan transaction.
            reusable = next(
                (
                    step
                    for step in prior.steps
                    if replace(navigation, step_id=step.step_id) == step
                ),
                None,
            )
            if reusable is not None:
                navigation = reusable
            else:
                suffixes = [
                    int(match.group(1))
                    for step in prior.steps
                    if (
                        match := re.fullmatch(
                            rf"{re.escape(milestone.canonical_id)}\.S(\d+)",
                            step.step_id,
                        )
                    )
                ]
                navigation = replace(
                    navigation,
                    step_id=(
                        f"{milestone.canonical_id}.S{max(suffixes, default=0) + 1:03d}"
                    ),
                )
        return navigation

    @classmethod
    def corrective_steps_from_value(
        cls,
        values: Sequence[object],
        *,
        milestone: MilestoneSpec | None = None,
    ) -> tuple[PlanStepSpec, ...]:
        """Normalize model-owned corrective work into route deltas."""

        return cls._steps_from_value(
            values,
            corrective=True,
            milestone=milestone,
            maximum=4,
        )

    @classmethod
    def materialized_steps_from_value(
        cls,
        values: Sequence[object],
        *,
        milestone: MilestoneSpec,
    ) -> tuple[PlanStepSpec, ...]:
        return cls._steps_from_value(
            values,
            corrective=False,
            milestone=milestone,
            maximum=None,
        )

    @classmethod
    def _steps_from_value(
        cls,
        values: Sequence[object],
        *,
        corrective: bool,
        milestone: MilestoneSpec | None = None,
        maximum: int | None,
    ) -> tuple[PlanStepSpec, ...]:
        if maximum is not None and len(values) > maximum:
            raise ValueError(f"a review may add at most {maximum} local Steps")
        known_criteria = (
            {criterion.criterion_id for criterion in milestone.criteria}
            if milestone is not None
            else None
        )
        result: list[PlanStepSpec] = []
        for index, raw in enumerate(values, start=1):
            if not isinstance(raw, Mapping):
                raise TypeError("PlanStep must be an object")
            raw_step_id = (
                f"proposal-{index}"
                if corrective
                else cls.canonical_step_address(
                    milestone.canonical_id if milestone is not None else "",
                    str(raw.get("step_id") or f"S{index:03d}"),
                )
            )
            explicit_mappings = raw.get("criterion_ids")
            if explicit_mappings is not None:
                if not isinstance(explicit_mappings, (list, tuple)):
                    raise ValueError("PlanStep criterion_ids must be an array")
                criterion_ids = tuple(map(str, explicit_mappings))
            elif milestone is not None:
                # The enclosing Milestone review narrows these contribution
                # links to the failed requirement set. They remain graph hints,
                # not local proof obligations.
                criterion_ids = tuple(criterion.criterion_id for criterion in milestone.criteria)
            else:
                criterion_ids = ()
            step_entities = cls.normalize_entity_refs(raw.get("entity_refs", ()))
            proposal = PlanStepSpec.from_dict(
                {
                    **dict(raw),
                    "step_id": raw_step_id,
                    "criterion_ids": list(criterion_ids),
                    "entity_refs": list(step_entities),
                    "minimum_acceptance": [],
                    "historical_dependency_refs": list(
                        cls.normalize_entity_refs(raw.get("historical_dependency_refs", ()))
                    ),
                    "corrective": corrective,
                },
                index,
            )
            if not proposal.expected_outcome.strip() or not proposal.failure_signals:
                raise ValueError("PlanStep requires expected_outcome and bounded failure_signals")
            if known_criteria is not None:
                unknown = set(proposal.criterion_ids).difference(known_criteria)
                if unknown:
                    raise ValueError(f"PlanStep maps unknown Milestone criteria: {sorted(unknown)}")
            result.append(proposal)
        return tuple(result)

    @staticmethod
    def render_review_request(
        *,
        plan: PlanSpec,
        milestone_id: str,
        outcome: Mapping[str, object],
        milestone_statuses: Mapping[str, str],
        revision_id: str,
    ) -> str:
        """Build the bounded requirement and route review at a Milestone boundary."""

        current_index = next(
            index
            for index, milestone in enumerate(plan.milestones)
            if milestone.canonical_id == milestone_id
        )
        visible_indexes = set(
            range(max(0, current_index - 1), min(len(plan.milestones), current_index + 4))
        )
        visible_indexes.add(len(plan.milestones) - 1)
        route = []
        for index, milestone in enumerate(plan.milestones):
            if index not in visible_indexes:
                continue
            status = milestone_statuses.get(milestone.canonical_id, milestone.status)
            item: dict[str, object] = {
                "milestone_id": milestone.canonical_id,
                "title": milestone.title,
                "status": status,
                "target_outcome": milestone.target_outcome,
                "source_plan_item_ids": list(milestone.source_plan_item_ids),
            }
            if index == current_index:
                item["minimum_acceptance"] = primitive(milestone.minimum_acceptance)
            route.append(item)
        final_review = current_index == len(plan.milestones) - 1
        context = {
            "milestone_id": milestone_id,
            "verified_outcome": dict(outcome),
            "workspace_revision_id": revision_id,
            "frozen_goal": plan.goal,
            "frozen_final_acceptance": primitive(plan.final_acceptance),
            "task_level_final_review": final_review,
            "route": route,
        }
        return (
            "MILESTONE REQUIREMENT AND ROUTE REVIEW. Use the fixed Task, this Milestone's target "
            "outcome, the changes already made, and the tests or behavior observations that really "
            "ran. Do not rescan the repository or invent a separate proof protocol. A green test "
            "supports only the behavior its assertions actually check; a code change alone cannot "
            "prove behavior. If task_level_final_review is true, make one final pass over the full "
            "fixed Task and ensure no explicit requirement was lost across Milestones. Otherwise, "
            "also check whether the nearby pending route is still reasonable. Do not modify the "
            "repository or run implementation work in this review. "
            "Call review_current_milestone exactly once. Its fields are milestone_id, decision, "
            "reason, requirement_coverage, future_milestones, and corrective_steps. For every "
            "current requirement, copy its requirement_text and observable_outcome exactly, mark "
            "SATISFIED or NOT_SATISFIED, and summarize the observed support or gap. decision is "
            "CONTINUE or REPLAN_FUTURE only when all requirements are satisfied; otherwise use "
            "CORRECT_CURRENT with concrete causal corrective Steps. CONTINUE keeps "
            "future_milestones=null. "
            "REPLAN_FUTURE supplies only the complete ordered list of still-pending Milestone "
            "skeletons; never resubmit or rewrite completed history, the task goal, or final "
            "acceptance. Keep existing stable IDs where their stage remains and assign fresh "
            "monotonic IDs only to genuinely new stages. Every pending skeleton quotes the "
            "smallest verbatim original-Task requirement that makes it necessary; supporting or "
            "conditional native work without one stays inside the stage it serves. Include only "
            "the broad claim type and known natural entity addresses, not guessed selectors, "
            "tests, or future Steps. After this "
            "review is durable, the model continues natural Coding directly on the selected "
            "Milestone; there is no separate activation Turn. For successful decisions, "
            "corrective_steps must be empty.\n\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    @staticmethod
    def render_failure_review_request(
        *,
        milestone: MilestoneSpec,
        failure: Mapping[str, object],
        revision_id: str,
        semantic_route: Mapping[str, object],
    ) -> str:
        """Request one causal correction while the failed Milestone stays current."""

        context = {
            "milestone": primitive(milestone),
            "workspace_revision_id": revision_id,
            "failure": dict(failure),
            "semantic_route": dict(semantic_route),
        }
        return (
            "CURRENT MILESTONE VERIFICATION FAILED. The failed Milestone remains the "
            "authoritative current route node. Diagnose the bound failure and call "
            "review_current_milestone exactly once with decision=CORRECT_CURRENT, "
            "future_milestones=null, complete requirement_coverage, and one to four concrete "
            "corrective_steps. Mark each demonstrably unmet requirement NOT_SATISFIED and keep "
            "independently supported requirements SATISFIED. Each corrective Step "
            "must state real work, expected_outcome, known natural entity addresses, "
            "failure_signals, and natural historical dependencies. Do not copy the Milestone "
            "terminal contract into a Step, do not "
            "create a generic Repair Step, and do not modify future Milestones. The runtime "
            "links corrective navigation to the unmet Milestone requirements, "
            "persists the review, and only then exposes the first new Step as current.\n\n"
            + json.dumps(context, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        )

    @staticmethod
    def render_for_thread(plan: PlanSpec) -> str:
        lines = [
            "Canonical execution plan (use these stable Milestone IDs in later plan updates):",
            f"Goal: {plan.goal}",
        ]
        for item in plan.milestones:
            lines.append(f"- {item.canonical_id}: {item.title} [{item.status}]")
            lines.append(f"  Ideal outcome: {item.target_outcome}")
            lines.append("  Minimum acceptance:")
            for criterion in item.minimum_acceptance:
                lines.append(
                    "    - "
                    f"{criterion.criterion_id}: {criterion.observable_outcome} "
                    f"(Evidence: {', '.join(value.value for value in criterion.required_evidence_types)})"
                )
            for step in item.steps:
                lines.append(f"  - {step.step_id}: {step.title} [{step.status.value}]")
        lines.append("Do not renumber or reuse a Milestone ID.")
        return "\n".join(lines)
