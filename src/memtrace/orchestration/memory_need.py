from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Mapping, Sequence

from ..contracts import (
    COMPRESSED_MEMORY_REPRESENTATIONS,
    SEMANTIC_PAGE_QUERY_RELATIONS,
    ContextImage,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    RecallTemporalScope,
    digest,
)
from ..harness.command_semantics import (
    LogicalCommandOutcome,
    command_evidence_semantics,
    evidence_result_selector_matches,
    explicit_pipeline_test_outcome,
    logical_command_outcome,
)
from ..harness.contracts import HarnessEvent, HarnessEventType
from ..harness.memory_tools import (
    MILESTONE_REVIEW_TOOL,
    SEMANTIC_UPDATE_TOOL,
)
from ..rich_graph.models import STRUCTURAL_RELATIONS
from .models import MemoryNeed, MemoryUseAttribution

_MEMORY_BLOCK = re.compile(r"<memory_need>\s*(\{.*?\})\s*</memory_need>", re.DOTALL)
_SEMANTIC_BLOCK = re.compile(r"<semantic_update>\s*(\{.*?\})\s*</semantic_update>", re.DOTALL)
_COMPLETION_BLOCK = re.compile(
    r"<milestone_completion>\s*(\{.*?\})\s*</milestone_completion>",
    re.DOTALL,
)
_HANDOFF_BLOCK = re.compile(
    r"<handoff_consumed>\s*(\{.*?\})\s*</handoff_consumed>",
    re.DOTALL,
)
_MEMORY_USE_BLOCK = re.compile(r"<memory_use>\s*(\{.*?\})\s*</memory_use>", re.DOTALL)
_INTERNAL_RUNTIME_DYNAMIC_TOOLS = frozenset(
    {
        "recall_memory",
        "attribute_memory_use",
        "project_native_plan",
        "verify_current_milestone",
        SEMANTIC_UPDATE_TOOL,
        MILESTONE_REVIEW_TOOL,
    }
)


@dataclass(frozen=True, slots=True)
class MemoryNeedDetection:
    need: MemoryNeed | None
    triggers: tuple[str, ...]
    rejected_reason: str | None = None


class MemoryUseParser:
    """Read model attribution without exposing internal Page identities."""

    @staticmethod
    def detect(event: HarnessEvent) -> tuple[MemoryUseAttribution, ...]:
        direct = event.payload.get("memory_use")
        item = event.payload.get("item")
        item = item if isinstance(item, Mapping) else {}
        nested = item.get("memoryUse", item.get("memory_use"))
        values: object = direct if direct is not None else nested
        if values is None:
            text = item.get("text", item.get("content"))
            if isinstance(text, str):
                matches = tuple(_MEMORY_USE_BLOCK.finditer(text))
                parsed: list[object] = []
                for match in matches:
                    try:
                        parsed.append(json.loads(match.group(1)))
                    except json.JSONDecodeError:
                        continue
                values = parsed
        if isinstance(values, Mapping):
            values = values.get("uses", (values,))
        if not isinstance(values, (list, tuple)):
            return ()
        result: list[MemoryUseAttribution] = []
        for value in values:
            if not isinstance(value, Mapping):
                continue
            attribution = MemoryUseAttribution.from_mapping(value)
            if not attribution.delivery_id and not attribution.entity_refs:
                continue
            if not attribution.usage:
                continue
            result.append(attribution)
        return tuple(result[:8])


@dataclass(frozen=True, slots=True)
class MilestoneCompletionClaim:
    canonical_id: str
    completed_step_ids: tuple[str, ...]
    criterion_ids: tuple[str, ...]
    summary: str


@dataclass(frozen=True, slots=True)
class MilestoneCompletionDetection:
    claim: MilestoneCompletionClaim | None
    rejected_reason: str | None = None


@dataclass(frozen=True, slots=True)
class HandoffConsumption:
    predecessor_canonical_id: str
    used_information: tuple[str, ...]
    still_needed: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class HandoffConsumptionDetection:
    claim: HandoffConsumption | None
    rejected_reason: str | None = None


class HandoffConsumptionParser:
    """Parse the successor model's semantic acknowledgement of a handoff."""

    @staticmethod
    def detect(event: HarnessEvent) -> HandoffConsumptionDetection:
        if event.event_type is not HarnessEventType.ITEM_COMPLETED:
            return HandoffConsumptionDetection(None)
        item = event.payload.get("item")
        if not isinstance(item, Mapping) or str(item.get("type", "")) != "agentMessage":
            return HandoffConsumptionDetection(None)
        text = str(item.get("text", item.get("content", "")))
        matches = tuple(_HANDOFF_BLOCK.finditer(text))
        if not matches:
            return HandoffConsumptionDetection(None)
        if len(matches) != 1:
            return HandoffConsumptionDetection(None, "MULTIPLE_HANDOFF_BLOCKS")
        try:
            value = json.loads(matches[0].group(1))
        except json.JSONDecodeError:
            return HandoffConsumptionDetection(None, "INVALID_HANDOFF_JSON")
        if not isinstance(value, Mapping):
            return HandoffConsumptionDetection(None, "HANDOFF_NOT_OBJECT")
        predecessor = str(value.get("predecessor_canonical_id", "")).strip().upper()
        if re.fullmatch(r"M\d{3,}", predecessor) is None:
            return HandoffConsumptionDetection(None, "INVALID_HANDOFF_MILESTONE_ID")
        used_raw = value.get("used_information", ())
        needed_raw = value.get("still_needed", ())
        if not isinstance(used_raw, list) or not isinstance(needed_raw, list):
            return HandoffConsumptionDetection(None, "HANDOFF_FIELDS_MUST_BE_LISTS")
        used = tuple(dict.fromkeys(str(item).strip() for item in used_raw if str(item).strip()))
        needed = tuple(dict.fromkeys(str(item).strip() for item in needed_raw if str(item).strip()))
        if not used:
            return HandoffConsumptionDetection(None, "HANDOFF_USED_INFORMATION_EMPTY")
        return HandoffConsumptionDetection(HandoffConsumption(predecessor, used, needed))


class MilestoneCompletionParser:
    """Parse only the explicit typed completion handshake from an Agent message."""

    @staticmethod
    def detect(event: HarnessEvent) -> MilestoneCompletionDetection:
        if event.event_type is not HarnessEventType.ITEM_COMPLETED:
            return MilestoneCompletionDetection(None)
        item = event.payload.get("item")
        if not isinstance(item, Mapping) or str(item.get("type", "")) != "agentMessage":
            return MilestoneCompletionDetection(None)
        text = str(item.get("text", item.get("content", "")))
        matches = tuple(_COMPLETION_BLOCK.finditer(text))
        if not matches:
            return MilestoneCompletionDetection(None)
        if len(matches) != 1:
            return MilestoneCompletionDetection(None, "MULTIPLE_COMPLETION_BLOCKS")
        try:
            value = json.loads(matches[0].group(1))
        except json.JSONDecodeError:
            return MilestoneCompletionDetection(None, "INVALID_COMPLETION_JSON")
        if not isinstance(value, Mapping):
            return MilestoneCompletionDetection(None, "COMPLETION_NOT_OBJECT")
        canonical_id = str(value.get("canonical_id", "")).strip().upper()
        if re.fullmatch(r"M\d{3,}", canonical_id) is None:
            return MilestoneCompletionDetection(None, "INVALID_COMPLETION_MILESTONE_ID")
        raw_steps = value.get("completed_step_ids", [])
        raw_criteria = value.get("criterion_ids", [])
        if not isinstance(raw_steps, list) or not isinstance(raw_criteria, list):
            return MilestoneCompletionDetection(None, "COMPLETION_IDS_MUST_BE_LISTS")
        step_ids = tuple(str(item).strip() for item in raw_steps)
        criterion_ids = tuple(str(item).strip() for item in raw_criteria)
        if any(not item for item in (*step_ids, *criterion_ids)):
            return MilestoneCompletionDetection(None, "COMPLETION_IDS_MUST_NOT_CONTAIN_EMPTY")
        if len(set(step_ids)) != len(step_ids) or len(set(criterion_ids)) != len(criterion_ids):
            return MilestoneCompletionDetection(None, "DUPLICATE_COMPLETION_IDS")
        summary = " ".join(str(value.get("summary", "")).split())
        if not summary:
            return MilestoneCompletionDetection(None, "EMPTY_COMPLETION_SUMMARY")
        return MilestoneCompletionDetection(
            MilestoneCompletionClaim(
                canonical_id=canonical_id,
                completed_step_ids=step_ids,
                criterion_ids=criterion_ids,
                summary=summary,
            )
        )


class HarnessEvidenceExtractor:
    """Derive bounded, typed facts from actually observed Harness events."""

    @staticmethod
    def extract(
        event: HarnessEvent,
        *,
        active_criterion_ids: Sequence[str] = (),
        active_result_criteria: Sequence[Mapping[str, object]] = (),
        active_step_id: str | None = None,
    ) -> tuple[EvidenceDraft, ...]:
        item = event.payload.get("item")
        item = item if isinstance(item, Mapping) else {}
        revision = f"revision:{event.revision_id}"
        criterion_ids = tuple(
            dict.fromkeys(
                str(value).strip() for value in active_criterion_ids if str(value).strip()
            )
        )
        facts: list[EvidenceDraft] = []
        if event.event_type is HarnessEventType.MEMORY_TOOL_RESULT:
            metadata = event.payload.get("runtime_metadata")
            metadata = metadata if isinstance(metadata, Mapping) else {}
            if metadata.get("kind") == "EXTERNAL_VERIFICATION_RESULT":
                evidence_bindings = metadata.get("criterion_ids_by_evidence_type", {})
                criterion_bindings = metadata.get(
                    "criterion_bindings_by_evidence_type",
                    {},
                )
                criterion_bindings = (
                    criterion_bindings if isinstance(criterion_bindings, Mapping) else {}
                )
                task_final_bindings = metadata.get(
                    "task_final_criterion_ids_by_evidence_type",
                    {},
                )
                task_final_bindings = (
                    task_final_bindings
                    if isinstance(task_final_bindings, Mapping)
                    else {}
                )
                evidence_success_by_type = metadata.get("evidence_success_by_type", {})
                evidence_success_by_type = (
                    evidence_success_by_type
                    if isinstance(evidence_success_by_type, Mapping)
                    else {}
                )
                if isinstance(evidence_bindings, Mapping):
                    for raw_type, raw_criterion_ids in evidence_bindings.items():
                        try:
                            evidence_type = FactType(str(raw_type))
                        except ValueError:
                            continue
                        if evidence_type not in {
                            FactType.TOOL_RESULT,
                            FactType.TEST_RESULT,
                            FactType.TEST_FAILURE,
                            FactType.VERIFIER_RESULT,
                        }:
                            continue
                        criterion_ids = (
                            list(map(str, raw_criterion_ids))
                            if isinstance(raw_criterion_ids, (list, tuple))
                            else []
                        )
                        raw_criterion_bindings = criterion_bindings.get(
                            evidence_type.value,
                            (),
                        )
                        qualified_criterion_ids = (
                            list(map(str, raw_criterion_bindings))
                            if isinstance(raw_criterion_bindings, (list, tuple))
                            else []
                        )
                        raw_task_final_ids = task_final_bindings.get(
                            evidence_type.value,
                            (),
                        )
                        task_final_criterion_ids = (
                            list(map(str, raw_task_final_ids))
                            if isinstance(raw_task_final_ids, (list, tuple))
                            else []
                        )
                        evidence_success = bool(
                            evidence_success_by_type.get(
                                evidence_type.value,
                                metadata.get("success", False),
                            )
                        )
                        facts.append(
                            EvidenceDraft(
                                EvidenceKey(
                                    evidence_type,
                                    str(
                                        metadata.get(
                                            "canonical_entity_id",
                                            f"test:external-verification:{event.harness_event_id}",
                                        )
                                    ),
                                    "trusted_external_verification",
                                    revision,
                                    event.branch_id,
                                ),
                                {
                                    "tool_selector": metadata.get("tool_selector"),
                                    "command": metadata.get("command"),
                                    "command_digest": metadata.get("command_digest"),
                                    "success": evidence_success,
                                    "exit_code": metadata.get("exit_code"),
                                    "output_excerpt": metadata.get("output_excerpt", ""),
                                    "criterion_ids": criterion_ids,
                                    "criterion_bindings": qualified_criterion_ids,
                                    "task_final_criterion_ids": task_final_criterion_ids,
                                    "plan_step_id": metadata.get("plan_step_id"),
                                    "aggregate_success": metadata.get("aggregate_success"),
                                    "criterion_projection": metadata.get(
                                        "criterion_projection",
                                        {},
                                    ),
                                    "historical_milestones_reopened": bool(
                                        metadata.get("historical_milestones_reopened", False)
                                    ),
                                    "final_acceptance_scope": metadata.get(
                                        "final_acceptance_scope",
                                        "CURRENT_WORKSPACE_REVISION",
                                    ),
                                    "authority": "RUNTIME_TRUSTED_VERIFIER",
                                    "source_event_id": event.source_event_id,
                                },
                                must_preserve=not evidence_success,
                            )
                        )
            elif metadata.get("kind") == "SEMANTIC_UPDATE_ACCEPTED":
                raw_updates = metadata.get("updates", ())
                evidence_types = {
                    "implementation_decision": (
                        FactType.IMPLEMENTATION_DECISION,
                        "implementation_choice",
                    ),
                    "code_observation": (FactType.CODE_OBSERVATION, "agent_observation"),
                    "rejected_hypothesis": (
                        FactType.CODE_OBSERVATION,
                        "rejected_hypothesis",
                    ),
                    "constraint": (FactType.USER_CONSTRAINT, "active_constraint"),
                    "unresolved_question": (
                        FactType.UNRESOLVED_QUESTION,
                        "unresolved_question",
                    ),
                }
                if isinstance(raw_updates, (list, tuple)):
                    for update in raw_updates:
                        if not isinstance(update, Mapping):
                            continue
                        selected = evidence_types.get(str(update.get("kind", "")))
                        if selected is None:
                            continue
                        evidence_type, role = selected
                        entity_refs = tuple(map(str, update.get("entity_refs", ())))
                        for entity_ref in entity_refs:
                            facts.append(
                                EvidenceDraft(
                                    EvidenceKey(
                                        evidence_type,
                                        entity_ref,
                                        role,
                                        revision,
                                        event.branch_id,
                                    ),
                                    {
                                        "summary": update.get("summary"),
                                        "purpose": update.get("purpose"),
                                        "criterion_ids": list(
                                            map(str, update.get("criterion_ids", ()))
                                        ),
                                        "plan_step_id": active_step_id,
                                        "entity_refs": list(entity_refs),
                                        "source_event_id": event.source_event_id,
                                        "authority": "STRUCTURED_AGENT_OUTPUT",
                                    },
                                    must_preserve=evidence_type
                                    in {
                                        FactType.USER_CONSTRAINT,
                                        FactType.UNRESOLVED_QUESTION,
                                    },
                                )
                            )
            elif metadata.get("kind") in {
                "MILESTONE_REVIEW_ACCEPTED",
                "MILESTONE_BOUNDARY_REQUESTED",
            }:
                accepted_review = metadata.get("kind") == "MILESTONE_REVIEW_ACCEPTED"
                if accepted_review:
                    review_source: Mapping[str, object] = metadata
                else:
                    # A review stated with the boundary request carries the
                    # same runtime-validated requirement coverage; its
                    # statements are facts of this revision even though the
                    # route decision is applied only after acceptance.
                    raw_review = metadata.get("boundary_review")
                    review_source = raw_review if isinstance(raw_review, Mapping) else {}
                milestone_id = str(review_source.get("milestone_id", "")).strip()
                if milestone_id:
                    for coverage in review_source.get("requirement_coverage", ()):
                        if not isinstance(coverage, Mapping):
                            continue
                        criterion_id = str(coverage.get("criterion_id", "")).strip()
                        requirement_id = str(
                            coverage.get("requirement_id", criterion_id)
                        ).strip()
                        if not criterion_id:
                            continue
                        satisfied = str(coverage.get("status", "")) == "SATISFIED"
                        facts.append(
                            EvidenceDraft(
                                EvidenceKey(
                                    FactType.REQUIREMENT_REVIEW,
                                    f"requirement:{requirement_id}",
                                    "milestone_requirement_review",
                                    revision,
                                    event.branch_id,
                                ),
                                {
                                    "milestone_id": milestone_id,
                                    "criterion_ids": [criterion_id],
                                    "requirement_id": requirement_id,
                                    "requirement_text": coverage.get("requirement_text"),
                                    "observable_outcome": coverage.get(
                                        "observable_outcome"
                                    ),
                                    "success": satisfied,
                                    "evidence_summary": coverage.get("evidence_summary"),
                                    "source_event_id": event.source_event_id,
                                    "authority": "STRUCTURED_AGENT_REQUIREMENT_REVIEW",
                                },
                                must_preserve=not satisfied,
                            )
                        )
                    facts.append(
                        EvidenceDraft(
                            EvidenceKey(
                                FactType.PLAN_DECISION,
                                f"milestone:{milestone_id}",
                                "milestone_route_review",
                                revision,
                                event.branch_id,
                            ),
                            {
                                "milestone_id": milestone_id,
                                "decision": review_source.get("decision"),
                                "reason": review_source.get("reason"),
                                "future_plan_digest": digest(review_source.get("future_plan")),
                                "failure_signature": review_source.get("failure_signature"),
                                "corrective_steps": review_source.get("corrective_steps", ()),
                                "applied_at": (
                                    "REVIEW_RESULT"
                                    if accepted_review
                                    else "VERIFIED_MILESTONE_BOUNDARY"
                                ),
                                "source_event_id": event.source_event_id,
                                "authority": "STRUCTURED_AGENT_REVIEW_RUNTIME_VALIDATED",
                            },
                            must_preserve=True,
                        )
                    )
        elif event.event_type is HarnessEventType.PROVIDER_STALLED:
            facts.append(
                EvidenceDraft(
                    EvidenceKey(
                        FactType.PLAN_DECISION,
                        f"run:{event.run_id}",
                        "provider_stall_recovery",
                        revision,
                        event.branch_id,
                    ),
                    {
                        "turn_id": event.turn_id,
                        "idle_timeout_seconds": event.payload.get("idle_timeout_seconds"),
                        "recovery": event.payload.get("recovery"),
                        "source_event_id": event.source_event_id,
                        "authority": "RUNTIME_OBSERVED",
                    },
                )
            )
        elif event.event_type is HarnessEventType.PLAN_UPDATED:
            for ordinal, raw in enumerate(event.payload.get("plan", ()), start=1):
                if not isinstance(raw, Mapping):
                    continue
                step = " ".join(str(raw.get("step", "")).split())
                match = re.match(r"^(M\d{3})\s*[:\-]", step, re.IGNORECASE)
                evidence_type = FactType.MILESTONE_STATE if match else FactType.PLAN_DECISION
                canonical = (
                    f"milestone:{match.group(1).upper()}"
                    if match
                    # Native Provider Plan entries are observations, not the
                    # Registry's stable PlanStep identities. Address each
                    # snapshot item independently so repeated titles do not
                    # collapse into one Evidence Unit. The Registry and its
                    # Semantic projection remain the authoritative route.
                    else (f"plan-observation:{event.run_id}:{event.harness_event_id}:{ordinal:04d}")
                )
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            evidence_type,
                            canonical,
                            "provider_plan_observation",
                            revision,
                            event.branch_id,
                        ),
                        {
                            "step": step,
                            "status": raw.get("status"),
                            "ordinal": ordinal,
                            "explanation": event.payload.get("explanation"),
                            "plan_step_id": active_step_id,
                            "authority": "CODEX_PLAN_EVENT",
                            "provider_method": event.provider_method,
                            "source_event_id": event.source_event_id,
                        },
                    )
                )
        elif event.event_type is HarnessEventType.WORKSPACE_REVISION_ADVANCED:
            file_surfaces = {str(s.get("path")): s for s in event.payload.get("file_surfaces", ())
                             if isinstance(s, Mapping)}
            raw_changes = event.payload.get("changes", ())
            change_by_path = {
                str(change.get("path")): change
                for change in raw_changes
                if isinstance(raw_changes, (list, tuple))
                and isinstance(change, Mapping)
                and change.get("path")
            }
            for path in dict.fromkeys(map(str, event.payload.get("paths", ()))):
                change = change_by_path.get(path, {})
                key = EvidenceKey(
                    evidence_type=FactType.CODE_CHANGE,
                    canonical_entity_id=f"file:{path}",
                    semantic_role="workspace_change",
                    revision_constraint=revision,
                    branch_scope=event.branch_id,
                )
                facts.append(
                    EvidenceDraft(
                        key,
                        {
                            "path": path,
                            "provider_item_id": item.get("id"),
                            "provider_item_type": item.get("type"),
                            "diff": change.get("diff"),
                            "change_kind": change.get("kind"),
                            **({"code_surface": file_surfaces[path]} if path in file_surfaces else {}),
                            "previous_revision_id": event.payload.get("previous_revision_id"),
                            "revision_id": event.payload.get("revision_id", event.revision_id),
                            "source_event_id": event.source_event_id,
                            "authority": "PROVIDER_FILE_CHANGE",
                            # The route owner is attached after this raw fact is
                            # durably written. Native Plan progress may later
                            # merge a contiguous execution span, but it never
                            # rewrites this action's causal owner.
                            "in_progress_criterion_ids": list(criterion_ids),
                        },
                    )
                )
            raw_bindings = event.payload.get("symbol_bindings", ())
            if isinstance(raw_bindings, (list, tuple)):
                for binding in raw_bindings:
                    if not isinstance(binding, Mapping):
                        continue
                    canonical = str(binding.get("canonical_entity_id", "")).strip()
                    path = str(binding.get("path", "")).strip()
                    qualified = str(binding.get("qualified_name", "")).strip()
                    scope = str(binding.get("change_scope", "")).strip()
                    if not canonical.startswith("symbol:") or not path or not qualified:
                        continue
                    # Python keeps its frozen changed-symbol-only contract. For
                    # a touched non-Python file, retain source-backed unchanged
                    # symbol observations too: direct Page-in often needs the
                    # containing type/helper around the changed symbol. They
                    # remain CODE_OBSERVATION only; only an actual diff
                    # intersection becomes CODE_CHANGE.
                    language = str(binding.get("language") or "").strip().casefold()
                    if (
                        scope != "CHANGED_SYMBOL"
                        and language in {"python", "py", "python3", ""}
                    ):
                        continue
                    multilang_metadata = (
                        {
                            "language": binding.get("language"),
                            "symbol_kind": binding.get("symbol_kind"),
                            "parser_backend": binding.get("parser_backend"),
                            "parser_confidence": binding.get("parser_confidence"),
                        }
                        if language and language not in {"python", "py", "python3"}
                        else {}
                    )
                    shared = {
                        "path": path,
                        "qualified_name": qualified,
                        "reference_id": binding.get("reference_id"),
                        "line_start": binding.get("line_start"),
                        "line_end": binding.get("line_end"),
                        "change_scope": scope,
                        "source_event_id": event.source_event_id,
                        "authority": "DETERMINISTIC_WORKSPACE_SYMBOL_INDEX",
                        "in_progress_criterion_ids": list(criterion_ids),
                        **multilang_metadata,
                    }
                    facts.append(
                        EvidenceDraft(
                            EvidenceKey(
                                FactType.CODE_OBSERVATION,
                                canonical,
                                "workspace_symbol_binding",
                                revision,
                                event.branch_id,
                            ),
                            {
                                **shared,
                                "summary": f"{qualified} is defined in {path}",
                                **({"code_surface": binding["code_surface"]}
                                   if isinstance(binding.get("code_surface"), Mapping) else {}),
                            },
                        )
                    )
                    if scope == "CHANGED_SYMBOL":
                        facts.append(
                            EvidenceDraft(
                                EvidenceKey(
                                    FactType.CODE_CHANGE,
                                    canonical,
                                    "workspace_symbol_change",
                                    revision,
                                    event.branch_id,
                                ),
                                {
                                    **shared,
                                    "summary": (
                                        f"{qualified} changed in this workspace revision"
                                    ),
                                    **(
                                        {"code_surface": binding["code_surface"]}
                                        if multilang_metadata
                                        and isinstance(
                                            binding.get("code_surface"),
                                            Mapping,
                                        )
                                        else {}
                                    ),
                                },
                            )
                        )
        elif (
            event.event_type is HarnessEventType.ITEM_COMPLETED
            and str(item.get("type", "")) == "agentMessage"
        ):
            text = str(item.get("text", item.get("content", ""))).strip()
            if text:
                facts.extend(
                    HarnessEvidenceExtractor._structured_agent_facts(
                        text=text,
                        event=event,
                        revision=revision,
                        active_step_id=active_step_id,
                    )
                )
                progress_summary = HarnessEvidenceExtractor._agent_progress_summary(text)
                if progress_summary:
                    facts.append(
                        EvidenceDraft(
                            EvidenceKey(
                                FactType.CODE_OBSERVATION,
                                f"agent-progress:{event.harness_event_id}",
                                "agent_progress_observation",
                                revision,
                                event.branch_id,
                            ),
                            {
                                "summary": progress_summary,
                                "message_excerpt": text[:6_000],
                                "message_digest": digest({"text": text}),
                                "plan_step_id": active_step_id,
                                "authority": "OBSERVED_AGENT_MESSAGE",
                                "source_event_id": event.source_event_id,
                            },
                        )
                    )
        elif (
            event.event_type is HarnessEventType.TOOL_RESULT
            and not bool(event.payload.get("partial", False))
            and item
            and not (
                str(item.get("type", "")) == "dynamicToolCall"
                and str(item.get("tool", item.get("name", ""))) in _INTERNAL_RUNTIME_DYNAMIC_TOOLS
            )
        ):
            command = str(item.get("command", item.get("tool", item.get("name", ""))))
            command_semantics = command_evidence_semantics(command)
            looks_like_test = command_semantics.is_test_observation
            complete_output = HarnessEvidenceExtractor._item_output(item)
            explicit_test_outcome = (
                explicit_pipeline_test_outcome(
                    command,
                    observed_output=complete_output,
                )
                if looks_like_test
                else None
            )
            provider_success = HarnessEvidenceExtractor._tool_success(item)
            success = (
                explicit_test_outcome.success
                if explicit_test_outcome is not None
                else provider_success
            )
            success_exit_status_reliable = (
                explicit_test_outcome is not None
                or command_semantics.success_exit_status_reliable
                if looks_like_test
                else True
            )
            evidence_type = (
                FactType.TEST_RESULT
                if success and looks_like_test
                else FactType.TEST_FAILURE
                if not success and looks_like_test
                else FactType.TOOL_RESULT
            )
            role = "failure_forensics" if not success else "execution_result"
            identity = str(item.get("id", "unknown"))
            entity_prefix = "test" if looks_like_test else "tool"
            criterion_ids = item.get("criterionIds", item.get("criterion_ids", ()))
            logical_receipts: dict[str, tuple[LogicalCommandOutcome, list[str]]] = {}
            for criterion in active_result_criteria:
                if not bool(criterion.get("required", True)):
                    continue
                criterion_id = str(criterion.get("criterion_id", "")).strip()
                if not criterion_id:
                    continue
                for raw_selector in criterion.get("test_selectors", ()):
                    selector = str(raw_selector).strip()
                    if not selector:
                        continue
                    outcome = logical_command_outcome(
                        selector,
                        command,
                        observed_output=complete_output,
                    )
                    if outcome is None:
                        continue
                    recorded = logical_receipts.setdefault(selector, (outcome, []))
                    recorded[1].append(criterion_id)
            if not isinstance(criterion_ids, (list, tuple)):
                criterion_ids = ()
            if not criterion_ids:
                # The Coordinator supplies only contracts owned by the current
                # route node.  Within that causal boundary, one real result may
                # satisfy both a selector-addressed claim and a requirement
                # whose concrete test address was intentionally left open.
                # Treating these as fallbacks made the first selector match
                # hide every open requirement in the same Step.  Semantic
                # review still decides whether the observed test is sufficient;
                # this layer records only factual ownership.
                criterion_ids = tuple(
                    dict.fromkeys(
                        criterion_id
                        for criterion in active_result_criteria
                        if bool(criterion.get("required", True))
                        and evidence_type.value
                        in set(map(str, criterion.get("required_evidence_types", ())))
                        and (
                            not tuple(criterion.get("test_selectors", ()))
                            or any(
                                evidence_result_selector_matches(
                                    str(selector),
                                    content=item,
                                )
                                for selector in criterion.get("test_selectors", ())
                                if str(selector).strip()
                                and str(selector).strip() not in logical_receipts
                            )
                        )
                        and (criterion_id := str(criterion.get("criterion_id", "")).strip())
                    )
                )
            criterion_bound = bool(criterion_ids)
            if looks_like_test or not success or criterion_bound:
                excerpt_limit = 4_000 if not success else 1_200
                output_excerpt = complete_output[:excerpt_limit]
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            evidence_type=evidence_type,
                            canonical_entity_id=f"{entity_prefix}:{identity}",
                            semantic_role=role,
                            revision_constraint=revision,
                            branch_scope=event.branch_id,
                        ),
                        {
                            "command": command[:800],
                            "command_digest": digest({"command": command}),
                            "status": item.get("status"),
                            "success": success,
                            "success_exit_status_reliable": success_exit_status_reliable,
                            "exit_code": (
                                explicit_test_outcome.exit_code
                                if explicit_test_outcome is not None
                                else item.get("exitCode")
                            ),
                            "provider_success": provider_success,
                            "outcome_basis": (
                                explicit_test_outcome.basis
                                if explicit_test_outcome is not None
                                else "PROVIDER_PROCESS_EXIT"
                            ),
                            "output_excerpt": output_excerpt,
                            "complete_output": complete_output,
                            "complete_output_digest": digest({"complete_output": complete_output}),
                            "criterion_ids": (
                                list(map(str, criterion_ids))
                                if isinstance(criterion_ids, (list, tuple))
                                else []
                            ),
                            "authority": "PROVIDER_TOOL_RESULT",
                            "source_event_id": event.source_event_id,
                        },
                        # An unbound shell/tool failure is useful execution
                        # history, but it is not automatically a task failure.
                        # Pin only evidence that is a real test observation or
                        # is explicitly owned by an acceptance Criterion.
                        must_preserve=(looks_like_test or criterion_bound)
                        and (not success or not success_exit_status_reliable),
                    )
                )
            for selector, (outcome, bound_criterion_ids) in logical_receipts.items():
                selector_is_test = command_evidence_semantics(selector).is_test_observation
                logical_evidence_type = (
                    FactType.TEST_RESULT
                    if outcome.success and selector_is_test
                    else FactType.TEST_FAILURE
                    if not outcome.success and selector_is_test
                    else FactType.TOOL_RESULT
                )
                logical_role = (
                    "execution_result" if outcome.success else "failure_forensics"
                )
                logical_excerpt_limit = 1_200 if outcome.success else 4_000
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            evidence_type=logical_evidence_type,
                            canonical_entity_id=(
                                f"{'test' if selector_is_test else 'tool'}:{identity}:"
                                f"selector:{digest({'selector': selector})[:16]}"
                            ),
                            semantic_role=logical_role,
                            revision_constraint=revision,
                            branch_scope=event.branch_id,
                        ),
                        {
                            "command": selector[:800],
                            "command_digest": digest({"command": selector}),
                            "logical_command": selector,
                            "observed_command": command[:800],
                            "observed_command_digest": digest({"command": command}),
                            "status": item.get("status"),
                            "success": outcome.success,
                            "success_exit_status_reliable": True,
                            "exit_code": outcome.exit_code,
                            "outcome_basis": outcome.basis,
                            "output_excerpt": complete_output[:logical_excerpt_limit],
                            "complete_output": complete_output,
                            "complete_output_digest": digest(
                                {"complete_output": complete_output}
                            ),
                            "criterion_ids": list(
                                dict.fromkeys(
                                    (
                                        *bound_criterion_ids,
                                        *(
                                            criterion_id
                                            for criterion in active_result_criteria
                                            if bool(criterion.get("required", True))
                                            and not tuple(criterion.get("test_selectors", ()))
                                            and logical_evidence_type.value
                                            in set(
                                                map(
                                                    str,
                                                    criterion.get(
                                                        "required_evidence_types",
                                                        (),
                                                    ),
                                                )
                                            )
                                            and (
                                                criterion_id := str(
                                                    criterion.get("criterion_id", "")
                                                ).strip()
                                            )
                                        ),
                                    )
                                )
                            ),
                            "authority": "PROVIDER_LOGICAL_COMMAND_RECEIPT",
                            "source_event_id": event.source_event_id,
                        },
                        must_preserve=not outcome.success,
                    )
                )
            accessed_paths = tuple(
                dict.fromkeys(map(str, event.payload.get("accessed_paths", ())))
            )
            observation_ref: str | None = None
            if success and accessed_paths and complete_output:
                observation_ref = f"observation:{identity}"
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            FactType.CODE_OBSERVATION,
                            observation_ref,
                            "provider_observed_content",
                            revision,
                            event.branch_id,
                        ),
                        {
                            "command": command[:800],
                            "command_digest": digest({"command": command}),
                            "accessed_paths": list(accessed_paths),
                            "status": item.get("status"),
                            "success": True,
                            "exit_code": item.get("exitCode"),
                            "output_excerpt": complete_output[:1_200],
                            "complete_output": complete_output,
                            "complete_output_digest": digest(
                                {"complete_output": complete_output}
                            ),
                            "source_event_id": event.source_event_id,
                            "authority": "PROVIDER_TOOL_RESULT",
                        },
                    )
                )
            for path in accessed_paths if success else ():
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            FactType.CODE_OBSERVATION,
                            f"file:{path}",
                            "provider_observed_read",
                            revision,
                            event.branch_id,
                        ),
                        {
                            "path": path,
                            "command": command[:800],
                            "command_digest": digest({"command": command}),
                            "success": True,
                            "observation_ref": observation_ref,
                            "complete_output_digest": (
                                digest({"complete_output": complete_output})
                                if complete_output
                                else None
                            ),
                            "source_event_id": event.source_event_id,
                            "authority": "PROVIDER_TOOL_RESULT",
                        },
                    )
                )
            decision = item.get("implementationDecision")
            if isinstance(decision, Mapping):
                entity = str(decision.get("canonical_entity_id", f"decision:{item.get('id')}"))
                facts.append(
                    EvidenceDraft(
                        EvidenceKey(
                            FactType.IMPLEMENTATION_DECISION,
                            entity,
                            str(decision.get("semantic_role", "implementation_choice")),
                            revision,
                            event.branch_id,
                        ),
                        {
                            "decision": dict(decision),
                            "source_event_id": event.source_event_id,
                            "authority": "STRUCTURED_AGENT_OUTPUT",
                        },
                    )
                )
        return tuple(facts)

    @staticmethod
    def _item_output(item: Mapping[str, object]) -> str:
        for field in ("aggregatedOutput", "output", "stdout", "stderr", "text", "content"):
            value = item.get(field)
            if isinstance(value, str) and value:
                return "\n".join(line.rstrip() for line in value.splitlines())
        return ""

    @staticmethod
    def _structured_agent_facts(
        *,
        text: str,
        event: HarnessEvent,
        revision: str,
        active_step_id: str | None = None,
    ) -> tuple[EvidenceDraft, ...]:
        result: list[EvidenceDraft] = []
        for match in _COMPLETION_BLOCK.finditer(text):
            try:
                completion = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(completion, Mapping):
                continue
            milestone = str(completion.get("canonical_id", "")).strip().upper()
            summary = " ".join(str(completion.get("summary", "")).split())
            if re.fullmatch(r"M\d{3,}", milestone) is None or not summary:
                continue
            result.append(
                EvidenceDraft(
                    EvidenceKey(
                        FactType.MILESTONE_STATE,
                        f"milestone:{milestone}",
                        "completion_claim",
                        revision,
                        event.branch_id,
                    ),
                    {
                        "canonical_id": milestone,
                        "summary": summary[:1600],
                        "completed_step_ids": completion.get("completed_step_ids", ()),
                        "criterion_ids": completion.get("criterion_ids", ()),
                        "plan_step_id": active_step_id,
                        "source_event_id": event.source_event_id,
                        "authority": "STRUCTURED_AGENT_OUTPUT",
                    },
                    must_preserve=True,
                )
            )
        for match in _SEMANTIC_BLOCK.finditer(text):
            try:
                update = json.loads(match.group(1))
            except json.JSONDecodeError:
                continue
            if not isinstance(update, Mapping):
                continue
            categories = (
                ("decisions", FactType.IMPLEMENTATION_DECISION, "implementation_choice"),
                ("observations", FactType.CODE_OBSERVATION, "agent_observation"),
                (
                    "rejected_hypotheses",
                    FactType.CODE_OBSERVATION,
                    "rejected_hypothesis",
                ),
                ("constraints", FactType.USER_CONSTRAINT, "active_constraint"),
                (
                    "unresolved_questions",
                    FactType.UNRESOLVED_QUESTION,
                    "unresolved_question",
                ),
            )
            for field, evidence_type, role in categories:
                values = update.get(field, ())
                if not isinstance(values, list):
                    continue
                for index, value in enumerate(values):
                    if isinstance(value, str):
                        content: Mapping[str, object] = {"summary": value}
                    elif isinstance(value, Mapping):
                        content = dict(value)
                    else:
                        continue
                    summary = str(
                        content.get(
                            "summary",
                            content.get(
                                "question",
                                content.get("detail", content.get("description", "")),
                            ),
                        )
                    ).strip()
                    if not summary:
                        continue
                    supplied = str(content.get("canonical_entity_id", "")).strip()
                    canonical = supplied or (
                        f"semantic:{evidence_type.value.casefold()}:"
                        f"{digest({'summary': summary, 'index': index})[-16:]}"
                    )
                    result.append(
                        EvidenceDraft(
                            EvidenceKey(
                                evidence_type,
                                canonical,
                                str(content.get("semantic_role", role)),
                                revision,
                                event.branch_id,
                            ),
                            {
                                **dict(content),
                                "plan_step_id": content.get(
                                    "plan_step_id", active_step_id
                                ),
                                "source_event_id": event.source_event_id,
                                "authority": "STRUCTURED_AGENT_OUTPUT",
                            },
                            must_preserve=evidence_type
                            in {FactType.USER_CONSTRAINT, FactType.UNRESOLVED_QUESTION},
                        )
                    )
        return tuple(result)

    @staticmethod
    def _agent_progress_summary(text: str) -> str:
        """Keep a bounded natural-Coding handoff without making it proof.

        Codex already states findings and next actions during a normal Turn.
        Persisting that model-produced text is cheaper and more faithful than
        asking another model to summarize every event.  Internal control
        blocks are removed because their structured facts are stored
        separately; the remaining observation is explicitly non-authoritative
        for acceptance.
        """

        visible = text
        for block in (
            _MEMORY_BLOCK,
            _SEMANTIC_BLOCK,
            _COMPLETION_BLOCK,
            _HANDOFF_BLOCK,
            _MEMORY_USE_BLOCK,
        ):
            visible = block.sub(" ", visible)
        return " ".join(visible.split())[:1_600]

    @staticmethod
    def _tool_success(item: Mapping[str, object]) -> bool:
        return (
            item.get("success") is not False
            and item.get("exitCode") in (None, 0)
            and str(item.get("status", "")) == "completed"
        )


class MemoryNeedDetector:
    """Parse a model's semantic Recall question, then bind it to runtime EvidenceKeys."""

    _PROVIDER_INPUT_ITEM_TYPES = frozenset(
        {
            "userMessage",
            "developerMessage",
            "systemMessage",
            "contextCompaction",
        }
    )
    _MODEL_MESSAGE_ITEM_TYPES = frozenset({"agentMessage", "reasoning"})

    def detect(
        self,
        event: HarnessEvent,
        image: ContextImage,
        *,
        available_evidence: Sequence[EvidenceKey] = (),
        repeated_entities: Sequence[str] = (),
        resolved_ambiguities: Sequence[str] = (),
        newly_resident_entities: Sequence[str] = (),
        visible_nonresident_entities: Sequence[str] | None = None,
    ) -> MemoryNeedDetection:
        item = event.payload.get("item")
        item_type = str(item.get("type", "")) if isinstance(item, Mapping) else ""
        if item_type in self._PROVIDER_INPUT_ITEM_TYPES:
            # turn/steer, thread/inject_items and compaction control are
            # Provider inputs.  Their text can contain MemoryRefs and exact
            # recovered evidence, but that is visibility—not a model decision
            # to rely on the referenced fact.  Feeding these items back into
            # automatic detection creates a self-exciting Page-in loop.
            return MemoryNeedDetection(None, ())
        model_dependency_event = self._is_model_dependency_event(event, item_type=item_type)
        explicit_request_event = isinstance(event.payload.get("memory_need"), Mapping) or (
            event.event_type is HarnessEventType.ITEM_COMPLETED
            and item_type in self._MODEL_MESSAGE_ITEM_TYPES
        )
        raw, trigger = self._extract_request(event) if explicit_request_event else (None, "")
        if raw is None:
            referenced_address = (
                self.referenced_memory_ref(
                    event,
                    image,
                    newly_resident_entities=newly_resident_entities,
                    visible_nonresident_entities=visible_nonresident_entities,
                )
                if model_dependency_event
                else None
            )
            automatic = self.automatic_triggers(
                event,
                image,
                repeated_entities=repeated_entities,
                newly_resident_entities=newly_resident_entities,
                visible_nonresident_entities=visible_nonresident_entities,
                include_nonresident_reference=False,
            )
            if referenced_address is not None:
                automatic = (*automatic, "REFERENCED_MEMORY_REF")
            if not automatic:
                return MemoryNeedDetection(None, ())
            if referenced_address is None:
                # Failures, confidence and repeated reads remain attention
                # signals.  A reference to a NONRESIDENT semantic handle is
                # different: it is an observable memory access and faults
                # deterministically without asking the model to judge whether
                # the compressed summary is sufficient.
                return MemoryNeedDetection(
                    None,
                    automatic,
                    "MODEL_SEMANTIC_DECISION_REQUIRED",
                )
            raw = {
                "question": "Recover exact details for accessed nonresident semantic memory",
                "memory_ref": referenced_address[0],
                "entity_refs": list(referenced_address[1]),
                "desired_detail": "smallest exact fact slice for the referenced entity",
                "purpose": "dereference a compressed semantic handle before using it",
                "temporal_purpose": "memory_ref_detail",
            }
            trigger = "MEMORY_REF_ACCESS"
        required_value = raw.get("required_evidence", raw.get("requiredEvidence", []))
        if not isinstance(required_value, list):
            return MemoryNeedDetection(None, (trigger,), "INVALID_REQUIRED_EVIDENCE")
        if len(required_value) > 8:
            return MemoryNeedDetection(None, (trigger,), "EVIDENCE_KEY_LIMIT_EXCEEDED")
        keys: list[EvidenceKey] = []
        if required_value:
            # Backward-compatible transport for older clients. New model
            # prompts never expose these internal page-table keys.
            try:
                for value in required_value:
                    if not isinstance(value, Mapping):
                        raise ValueError("evidence key is not an object")
                    keys.append(
                        EvidenceKey(
                            evidence_type=FactType(str(value["evidence_type"])),
                            canonical_entity_id=str(value["canonical_entity_id"]),
                            semantic_role=str(value["semantic_role"]),
                            revision_constraint=str(
                                value.get("revision_constraint", f"revision:{event.revision_id}")
                            ),
                            branch_scope=str(value.get("branch_scope", event.branch_id)),
                            validity_requirement=str(value.get("validity_requirement", "CURRENT")),
                        )
                    )
            except (KeyError, TypeError, ValueError) as exc:
                return MemoryNeedDetection(
                    None,
                    (trigger,),
                    f"INVALID_EVIDENCE_KEY:{type(exc).__name__}",
                )
        else:
            keys.extend(available_evidence[:8])
        triggers = [trigger]
        if not all(self._resident(key, image) for key in keys):
            triggers.append("REQUIRED_EVIDENCE_NOT_RESIDENT")
        ambiguous = tuple(
            dict.fromkeys(
                (
                    *map(str, raw.get("ambiguous_entities", ())),
                    *map(str, resolved_ambiguities),
                )
            )
        )
        structural = tuple(
            map(
                str,
                raw.get(
                    "required_page_relations",
                    raw.get("required_structural_relations", ()),
                ),
            )
        )
        preferred_structural = tuple(
            map(
                str,
                raw.get(
                    "preferred_page_relations",
                    raw.get("preferred_structural_relations", ()),
                ),
            )
        )
        page_relations = tuple(
            dict.fromkeys(
                item.strip().upper()
                for item in (*structural, *preferred_structural)
                if item.strip()
            )
        )
        unknown_page_relations = set(page_relations).difference(SEMANTIC_PAGE_QUERY_RELATIONS)
        if unknown_page_relations:
            return MemoryNeedDetection(
                None,
                (trigger,),
                "INVALID_PAGE_RELATIONS:" + ",".join(sorted(unknown_page_relations)),
            )
        structural = tuple(item.strip().upper() for item in structural if item.strip())
        preferred_structural = tuple(
            item.strip().upper() for item in preferred_structural if item.strip()
        )
        rich_code_relations = tuple(
            dict.fromkeys(
                str(item).strip().upper()
                for item in raw.get("rich_code_relations", ())
                if str(item).strip()
            )
        )
        unknown_rich_relations = set(rich_code_relations).difference(STRUCTURAL_RELATIONS)
        if unknown_rich_relations:
            return MemoryNeedDetection(
                None,
                (trigger,),
                "INVALID_RICH_CODE_RELATIONS:" + ",".join(sorted(unknown_rich_relations)),
            )
        direction = str(
            raw.get(
                "page_relation_direction",
                raw.get("structural_relation_direction", raw.get("direction", "BOTH")),
            )
        ).upper()
        if direction not in {"INCOMING", "OUTGOING", "BOTH"}:
            return MemoryNeedDetection(None, (trigger,), "INVALID_RELATION_DIRECTION")
        entity_refs = tuple(map(str, raw.get("entity_refs", raw.get("entities", ()))))
        if len(entity_refs) > 16:
            return MemoryNeedDetection(None, (trigger,), "ENTITY_REF_LIMIT_EXCEEDED")
        memory_ref = str(raw.get("memory_ref", "")).strip() or None
        if memory_ref is not None and not memory_ref.startswith("memoryref_"):
            return MemoryNeedDetection(None, (trigger,), "INVALID_MEMORY_REF")
        if not keys and not entity_refs and memory_ref is None:
            return MemoryNeedDetection(None, (trigger,), "EMPTY_SEMANTIC_ADDRESS")
        confidence = raw.get("confidence")
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            if float(confidence) < 0.6:
                triggers.append("LOW_CONFIDENCE")
        if ambiguous:
            triggers.append("ENTITY_AMBIGUITY")
        if bool(raw.get("revision_mismatch", False)):
            triggers.append("REVISION_MISMATCH")
        if bool(raw.get("test_failure_requires_history", False)):
            triggers.append("TEST_FAILURE_HISTORY_BASIS")
        question = " ".join(str(raw.get("question", "")).split())
        if not question:
            return MemoryNeedDetection(None, tuple(dict.fromkeys(triggers)), "EMPTY_QUESTION")
        desired_detail = " ".join(
            str(raw.get("desired_detail", raw.get("detail", "smallest exact fact slice"))).split()
        )
        purpose = " ".join(str(raw.get("purpose", "continue the current Milestone")).split())
        if not desired_detail or not purpose:
            return MemoryNeedDetection(
                None,
                tuple(dict.fromkeys(triggers)),
                "EMPTY_RECALL_INTENT_FIELD",
            )
        raw_scope = str(raw.get("temporal_purpose", "")).strip().upper()
        if raw_scope:
            try:
                temporal_scope = RecallTemporalScope(raw_scope)
            except ValueError:
                return MemoryNeedDetection(
                    None,
                    tuple(dict.fromkeys(triggers)),
                    "INVALID_TEMPORAL_PURPOSE",
                )
            exact_revision = temporal_scope is RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
        else:
            exact_revision = bool(
                raw.get(
                    "require_exact_revision",
                    all(key.revision_constraint == f"revision:{event.revision_id}" for key in keys),
                )
            )
            temporal_scope = (
                RecallTemporalScope.CURRENT_WORKSPACE_TRUTH
                if exact_revision
                else RecallTemporalScope.HISTORICAL_EXECUTION
            )
        return MemoryNeedDetection(
            MemoryNeed(
                question=question,
                required_evidence=tuple(keys),
                entity_refs=entity_refs,
                memory_ref=memory_ref,
                section_handle=(
                    str(raw["section_handle"]).strip()
                    if raw.get("section_handle") is not None
                    else None
                ),
                continuation_token=(
                    str(raw["continuation_token"]).strip()
                    if raw.get("continuation_token") is not None
                    else None
                ),
                desired_detail=desired_detail,
                purpose=purpose,
                required_structural_relations=structural,
                preferred_structural_relations=preferred_structural,
                structural_relation_direction=direction,
                rich_code_relations=rich_code_relations,
                ambiguous_entities=ambiguous,
                resolution_state=("RESOLVED_EXACT" if keys else "ACCEPTED"),
                require_exact_revision=exact_revision,
                temporal_scope=temporal_scope,
                trigger_reasons=tuple(dict.fromkeys(triggers)),
            ),
            tuple(dict.fromkeys(triggers)),
        )

    @classmethod
    def automatic_triggers(
        cls,
        event: HarnessEvent,
        image: ContextImage,
        *,
        repeated_entities: Sequence[str] = (),
        newly_resident_entities: Sequence[str] = (),
        visible_nonresident_entities: Sequence[str] | None = None,
        include_nonresident_reference: bool = True,
    ) -> tuple[str, ...]:
        triggers: list[str] = []
        item = event.payload.get("item")
        item = item if isinstance(item, Mapping) else {}
        if (
            event.event_type is HarnessEventType.TOOL_RESULT
            and not bool(event.payload.get("partial", False))
            and item
            and not HarnessEvidenceExtractor._tool_success(item)
        ):
            command = str(item.get("command", item.get("tool", ""))).casefold()
            if any(marker in command for marker in ("pytest", " test", "test ", "cargo test")):
                triggers.append("TEST_FAILURE_HISTORY_BASIS")
        if bool(event.payload.get("revision_mismatch", False)):
            triggers.append("REVISION_MISMATCH")
        if include_nonresident_reference and cls.referenced_nonresident_entities(
            event,
            image,
            newly_resident_entities=newly_resident_entities,
            visible_nonresident_entities=visible_nonresident_entities,
        ):
            triggers.append("REFERENCED_NONRESIDENT_HANDLE")
        if repeated_entities:
            triggers.append("REPEATED_REPOSITORY_LOOKUP")
        confidence = item.get("confidence", event.payload.get("confidence"))
        if isinstance(confidence, (int, float)) and not isinstance(confidence, bool):
            if float(confidence) < 0.6:
                triggers.append("LOW_CONFIDENCE")
        return tuple(dict.fromkeys(triggers))

    @classmethod
    def _is_model_dependency_event(
        cls,
        event: HarnessEvent,
        *,
        item_type: str,
    ) -> bool:
        """Accept only model-authored decisions, never Provider lifecycle echoes."""

        if event.event_type is HarnessEventType.PLAN_UPDATED:
            return True
        if event.event_type is HarnessEventType.ITEM_COMPLETED:
            # A completed reasoning item precedes a later model/tool decision
            # and can therefore fault early enough to affect that decision.
            # A completed agentMessage can be the terminal answer; treating
            # its retrospective summary as a new dependency races the already
            # closed Turn and cannot improve the answer it just produced.
            # Explicit recall requests embedded in an agentMessage are still
            # parsed above and remain valid.
            return item_type == "reasoning"
        if event.event_type is HarnessEventType.TOOL_INTENT:
            # Bidirectional dynamic memory tools are executed once from the
            # durable ``item/tool/call`` request. Their started/completed
            # lifecycle events are acknowledgements, not new requests.
            return item_type != "dynamicToolCall"
        return False

    @staticmethod
    def referenced_nonresident_entities(
        event: HarnessEvent,
        image: ContextImage,
        *,
        newly_resident_entities: Sequence[str] = (),
        visible_nonresident_entities: Sequence[str] | None = None,
    ) -> tuple[str, ...]:
        address = MemoryNeedDetector.referenced_memory_ref(
            event,
            image,
            newly_resident_entities=newly_resident_entities,
            visible_nonresident_entities=visible_nonresident_entities,
        )
        return () if address is None else address[1]

    @staticmethod
    def referenced_memory_ref(
        event: HarnessEvent,
        image: ContextImage,
        *,
        newly_resident_entities: Sequence[str] = (),
        visible_nonresident_entities: Sequence[str] | None = None,
    ) -> tuple[str, tuple[str, ...]] | None:
        """Recognize an actual model-visible virtual-address access.

        A file or symbol name is ordinary repository work and is not proof
        that compressed history was used. Automatic context-refresh events are created
        only when a model-authored event contains the opaque MemoryRef itself;
        structured ``recall_memory`` requests are parsed separately above.
        """

        encoded = json.dumps(event.payload, ensure_ascii=False, sort_keys=True)
        mentioned = {
            item.casefold()
            for item in re.findall(r"\bmemoryref_[A-Za-z0-9_-]{8,}\b", encoded)
        }
        if not mentioned:
            return None
        visible_entities = (
            None
            if visible_nonresident_entities is None
            else {str(entity).casefold() for entity in visible_nonresident_entities if str(entity)}
        )
        resident = {str(entity).casefold() for entity in newly_resident_entities if str(entity)}
        matches: list[tuple[str, tuple[str, ...]]] = []
        for artifact in image.artifacts:
            if artifact.representation not in COMPRESSED_MEMORY_REPRESENTATIONS:
                continue
            memory_ref = artifact.memory_ref
            if not memory_ref or memory_ref.casefold() not in mentioned or not artifact.source_handles:
                continue
            focus_entities = tuple(
                entity
                for entity in artifact.entity_refs
                if entity.startswith(("file:", "symbol:", "test:", "tool:"))
            )
            if visible_entities is not None and not any(
                entity.casefold() in visible_entities for entity in focus_entities
            ):
                continue
            if focus_entities and all(entity.casefold() in resident for entity in focus_entities):
                continue
            matches.append((memory_ref, focus_entities[:8]))
        if len(matches) != 1:
            return None
        return matches[0]

    @staticmethod
    def _resident(key: EvidenceKey, image: ContextImage) -> bool:
        digest_value = key.key_digest
        return any(
            digest_value in artifact.content or key.canonical_entity_id in artifact.entity_refs
            for artifact in image.artifacts
            if artifact.content
        )

    @staticmethod
    def _extract_request(event: HarnessEvent) -> tuple[Mapping[str, object] | None, str]:
        direct = event.payload.get("memory_need")
        if isinstance(direct, Mapping):
            return direct, "CODEX_EXPLICIT_MEMORY_REQUEST"
        item = event.payload.get("item")
        if not isinstance(item, Mapping):
            return None, ""
        if str(item.get("type", "")) == "dynamicToolCall" and str(
            item.get("tool", item.get("name", ""))
        ) in {"recall_memory", "memory.recall"}:
            arguments = item.get("arguments", {})
            if isinstance(arguments, Mapping):
                return arguments, "CODEX_RECALL_TOOL_REQUEST"
        nested = item.get("memoryNeed", item.get("memory_need"))
        if isinstance(nested, Mapping):
            return nested, "CODEX_EXPLICIT_MEMORY_REQUEST"
        text = item.get("text", item.get("content"))
        if not isinstance(text, str):
            return None, ""
        match = _MEMORY_BLOCK.search(text)
        candidate = match.group(1) if match is not None else text.strip()
        if not candidate.startswith("{"):
            return None, ""
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            return None, ""
        if not isinstance(parsed, Mapping):
            return None, ""
        request = parsed.get("memory_need", parsed)
        return (
            (request, "CODEX_EXPLICIT_MEMORY_REQUEST")
            if isinstance(request, Mapping)
            and ("question" in request or "required_evidence" in request)
            else (None, "")
        )

    @classmethod
    def requested_entities(cls, event: HarnessEvent) -> tuple[str, ...]:
        """Expose only the model's natural semantic references to the runtime resolver."""

        raw, _ = cls._extract_request(event)
        if raw is None:
            return ()
        values = raw.get("entity_refs", raw.get("entities", ()))
        if not isinstance(values, (list, tuple)):
            return ()
        return tuple(dict.fromkeys(str(item) for item in values if str(item)))[:16]

    @classmethod
    def has_explicit_request(cls, event: HarnessEvent) -> bool:
        raw, _ = cls._extract_request(event)
        return raw is not None
