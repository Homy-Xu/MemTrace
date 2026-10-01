from __future__ import annotations

from typing import Any

from ..acceptance import acceptance_direction_schema

_RELATIONS = (
    "ADVANCES_TO",
    "BRANCHES_TO",
    "CORRECTED_BY",
    "VERIFIED_BY",
    "SUPERSEDED_BY",
    "DEPENDED_ON_BY",
    "RESOLVED_BY",
)

MILESTONE_MANIFEST_TOOL = "project_native_plan"
MILESTONE_REVIEW_TOOL = "review_current_milestone"
EXTERNAL_VERIFICATION_TOOL = "verify_current_milestone"
SEMANTIC_UPDATE_TOOL = "record_semantic_update"
CODE_GRAPH_SEARCH_TOOL = "search_code_graph"


def code_graph_search_dynamic_tool() -> dict[str, Any]:
    """Optional read-only Rich Graph navigation tool for multilingual runs."""

    return {
        "type": "function",
        "name": CODE_GRAPH_SEARCH_TOOL,
        "description": (
            "Search the current-revision Repository State Graph for navigation only. "
            "Results are bounded hints, never evidence or a Milestone gate. "
            "If the graph is still building or unavailable, continue normal "
            "repository work instead of retrying the same query."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": ["query"],
            "properties": {
                "query": {
                    "type": "string",
                    "minLength": 1,
                    "maxLength": 240,
                },
                "scope_entities": {
                    "type": "array",
                    "maxItems": 8,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 320,
                    },
                },
                "relation_kinds": {
                    "type": "array",
                    "maxItems": 4,
                    "items": {
                        "type": "string",
                        "minLength": 1,
                        "maxLength": 40,
                    },
                },
                "max_results": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 8,
                },
                "purpose": {
                    "type": "string",
                    "enum": [
                        "navigation",
                        "implementation",
                        "verification",
                        "history",
                    ],
                },
            },
        },
    }


def _planned_step_schema(*, include_source_plan_items: bool) -> dict[str, Any]:
    non_empty = {"type": "string", "minLength": 1}
    string_list = {"type": "array", "items": dict(non_empty)}
    required = [
        "title",
        "expected_outcome",
        "entity_refs",
        "failure_signals",
        "historical_dependency_refs",
    ]
    properties: dict[str, Any] = {
        "title": dict(non_empty),
        "expected_outcome": dict(non_empty),
        "entity_refs": {
            **string_list,
            "description": (
                "Currently known natural file or symbol addresses. Leave empty instead of "
                "guessing; the corrective work may discover more precise addresses."
            ),
        },
        "failure_signals": {
            "type": "array",
            "minItems": 1,
            "items": dict(non_empty),
        },
        "historical_dependency_refs": {
            **string_list,
            "description": (
                "Natural entities whose already-indexed detail is a hard dependency. "
                "Never provide trace or Memory Anchor IDs."
            ),
        },
    }
    if include_source_plan_items:
        required.insert(0, "source_plan_item_ids")
        properties["source_plan_item_ids"] = {
            **string_list,
            "minItems": 1,
        }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": required,
        "properties": properties,
    }


def preliminary_step_schema() -> dict[str, Any]:
    """Describe one lightweight executable cursor derived from the native Plan.

    It tells the MTG what work is current and what risks should remain visible.
    Completion follows observed native-Plan progress or the Milestone boundary;
    Planning never invents selectors, Evidence IDs, or a Step Oracle.
    """

    non_empty = {"type": "string", "minLength": 1}
    string_list = {"type": "array", "items": dict(non_empty)}
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "source_plan_item_ids",
            "title",
            "work_kind",
            "expected_outcome",
            "entity_refs",
            "risk_checklist",
            "historical_dependency_refs",
        ],
        "properties": {
            "source_plan_item_ids": {
                **string_list,
                "minItems": 1,
                "maxItems": 1,
                "description": (
                    "The one frozen Nxxx item that owns this cursor. One native item may "
                    "produce several consecutive Steps, but a Step never collapses multiple "
                    "native addresses."
                ),
            },
            "title": dict(non_empty),
            "work_kind": {
                "type": "string",
                "enum": ["INSPECT", "IMPLEMENT", "VERIFY", "PROCESS"],
                "description": (
                    "A display-only work category used to keep the route card concise. It never "
                    "selects a required Evidence type or controls execution."
                ),
            },
            "expected_outcome": dict(non_empty),
            "entity_refs": {
                **string_list,
                "description": (
                    "Currently known natural file or symbol addresses. Leave empty instead of "
                    "guessing; execution may discover more precise addresses."
                ),
            },
            "risk_checklist": {
                **string_list,
                "description": (
                    "Short requirement-derived omissions to keep visible while this Step is "
                    "current. These are navigation hints, not separate approval gates."
                ),
            },
            "historical_dependency_refs": {
                **string_list,
                "description": (
                    "Natural entities whose NONRESIDENT history is a declared dependency. "
                    "Never provide trace or Memory Anchor IDs."
                ),
            },
        },
    }


def native_plan_projection_tool(output_schema: dict[str, Any]) -> dict[str, Any]:
    """Project an observed native Harness Plan into the typed MTG route."""

    return {
        "type": "function",
        "name": MILESTONE_MANIFEST_TOOL,
        "description": (
            "Publish the one lightweight semantic projection of an already-frozen native Coding "
            "Harness Plan. Group contiguous native addresses into Task-backed stages. Every "
            "standalone stage quotes the smallest verbatim original-Task requirement that makes "
            "it necessary; supporting or conditional work without such a requirement is grouped "
            "into the stage it serves. The original Task remains scope authority: verification or "
            "investigation cannot be strengthened into a mandatory modification. "
            "The runtime, not the model, creates stable "
            "IDs, dependencies, Evidence typing, Criterion mappings, final "
            "acceptance, and status. Milestone acceptance contains delivery outcomes; setup, "
            "audit, diagnosis and failure reproduction stay supporting work. Publish ordered "
            "preliminary Steps only for the first active stage; every future stage must keep an "
            "empty Step list until its execution boundary. Each active Step keeps exactly one "
            "frozen native Plan address, while one coarse native item may be split into several "
            "Steps. Steps state "
            "work, an expected outcome and a risk checklist; they never state selectors, Evidence "
            "IDs or proof contracts. No separate repository-inspection or activation Turn follows "
            "this publication. "
            "It does not replace the native Plan or report execution progress."
        ),
        "inputSchema": output_schema,
    }


def milestone_review_dynamic_tool(plan_schema: dict[str, Any]) -> dict[str, Any]:
    """Publish the one post-verification route decision as a typed command."""

    _ = plan_schema  # Kept in the public adapter signature for compatibility.
    non_empty = {"type": "string", "minLength": 1}
    string_list = {"type": "array", "items": dict(non_empty)}
    acceptance_direction = acceptance_direction_schema()
    direction_properties = acceptance_direction["properties"]
    planned_step = _planned_step_schema(include_source_plan_items=False)
    future_milestone = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "canonical_id",
            "source_plan_item_ids",
            "task_requirement",
            "target_outcome",
            "claim_type",
            "non_goals",
        ],
        "properties": {
            "canonical_id": {"type": "string", "pattern": "^M[0-9]{3,}$"},
            "source_plan_item_ids": string_list,
            "task_requirement": {
                **dict(non_empty),
                "description": (
                    "The smallest verbatim original-Task excerpt that makes this pending stage "
                    "necessary. Native Plan suggestions cannot add mandatory scope."
                ),
            },
            "target_outcome": {
                **dict(non_empty),
                "description": (
                    "The observable repository state this stage must reach. It may clarify the "
                    "quoted Task requirement but cannot add scope or strengthen its modality."
                ),
            },
            "claim_type": dict(direction_properties["claim_type"]),
            "entity_refs": {
                **dict(direction_properties["entity_refs"]),
                "description": (
                    "Optional coarse address only; omit exact future file or Symbol guesses."
                ),
            },
            "non_goals": string_list,
        },
    }

    return {
        "type": "function",
        "name": MILESTONE_REVIEW_TOOL,
        "description": (
            "Milestone acceptance is decided by the runtime from durable Evidence at the "
            "Turn boundary; you normally finish the work and end the Turn without calling this "
            "tool. Call it during the current Milestone only when you (a) want to close the "
            "execution boundary now, or (b) need to narrow the still-pending route. The call "
            "never completes the Milestone by itself: the runtime fences the Turn, reduces "
            "Evidence once, and applies your recorded decision only after the Milestone is "
            "COMPLETED_VERIFIED. Review every current requirement against the actual changes "
            "and execution Evidence. Use CONTINUE or REPLAN_FUTURE only when every requirement "
            "is satisfied. REPLAN_FUTURE replaces only the still-pending Task-anchored route; "
            "each standalone future stage must quote an exact original-Task requirement and "
            "remain a semantic skeleton without future Steps. It never resubmits completed "
            "history. When the runtime opens a bounded semantic review Turn for requirements it "
            "cannot verify automatically, answer with this tool exactly once. Use "
            "CORRECT_CURRENT with concrete local corrective Steps only after a verification "
            "failure when a requirement is genuinely not satisfied. The runtime persists "
            "requirement coverage and the route decision before exposing a new node."
        ),
        "inputSchema": {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "milestone_id",
                "decision",
                "reason",
                "future_milestones",
                "requirement_coverage",
                "corrective_steps",
            ],
            "properties": {
                "milestone_id": {"type": "string", "pattern": "^M[0-9]{3,}$"},
                "decision": {
                    "type": "string",
                    "enum": ["CONTINUE", "REPLAN_FUTURE", "CORRECT_CURRENT"],
                },
                "reason": {"type": "string", "minLength": 1},
                "future_milestones": {
                    "anyOf": [
                        {
                            "type": "array",
                            "minItems": 1,
                            "items": future_milestone,
                        },
                        {"type": "null"},
                    ],
                    "description": (
                        "Null for CONTINUE and CORRECT_CURRENT. For REPLAN_FUTURE, the complete "
                        "ordered list of still-pending Milestone skeletons only."
                    ),
                },
                "requirement_coverage": {
                    "type": "array",
                    "minItems": 1,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": [
                            "requirement_text",
                            "observable_outcome",
                            "status",
                            "evidence_summary",
                        ],
                        "properties": {
                            "requirement_text": dict(non_empty),
                            "observable_outcome": dict(non_empty),
                            "status": {
                                "type": "string",
                                "enum": ["SATISFIED", "NOT_SATISFIED"],
                            },
                            "evidence_summary": dict(non_empty),
                        },
                    },
                },
                "corrective_steps": {
                    "type": "array",
                    "maxItems": 4,
                    "items": planned_step,
                },
            },
        },
    }


def memory_dynamic_tools() -> tuple[dict[str, Any], ...]:
    """Return the model-facing semantic memory protocol.

    The model describes its semantic intent. Page identifiers, EvidenceKeys,
    byte ranges and ranking mechanics remain private to the runtime.
    """

    relation_items = {"type": "string", "enum": list(_RELATIONS)}
    return (
        {
            "type": "function",
            "name": SEMANTIC_UPDATE_TOOL,
            "description": (
                "Persist one coherent semantic delta that tools cannot prove by themselves, "
                "such as a confirmed implementation conclusion, a hypothesis rejected by "
                "observed evidence, a durable constraint, or an unresolved question. Bind the "
                "update to natural file/symbol entities; the runtime maps it "
                "to compatible Milestone claims and records the current Step only as provenance. "
                "Call it once at a coherent reasoning boundary, not for every read, patch line, "
                "test result, or intended next action; those are either captured automatically or "
                "remain the model's free execution choice."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["updates"],
                "properties": {
                    "updates": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 8,
                        "items": {
                            "type": "object",
                            "additionalProperties": False,
                            "required": [
                                "kind",
                                "summary",
                                "purpose",
                                "entity_refs",
                            ],
                            "properties": {
                                "kind": {
                                    "type": "string",
                                    "enum": [
                                        "implementation_decision",
                                        "code_observation",
                                        "rejected_hypothesis",
                                        "constraint",
                                        "unresolved_question",
                                    ],
                                },
                                "summary": {"type": "string", "minLength": 1},
                                "purpose": {"type": "string", "minLength": 1},
                                "entity_refs": {
                                    "type": "array",
                                    "minItems": 1,
                                    # One semantic conclusion may legitimately cover
                                    # several touched files/symbols. Keep it bounded,
                                    # but do not reject a useful five-file handoff.
                                    "maxItems": 8,
                                    "items": {"type": "string", "minLength": 1},
                                },
                            },
                        },
                    }
                },
            },
        },
        {
            "type": "function",
            "name": "recall_memory",
            "description": (
                "Resolve a visible compressed Memory Anchor before relying on it for a code, test, "
                "tool, or irreversible decision. Also use for a new historical semantic need. "
                "State what is needed and why; never guess internal trace IDs. A returned section "
                "directory may expose a model-safe section_handle, and an incomplete section may "
                "expose a continuation_token. Copy those opaque values on a follow-up request "
                "instead of rereading the repository or repeating the first trace prefix."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "question",
                    "entity_refs",
                    "desired_detail",
                    "purpose",
                    "temporal_purpose",
                ],
                "properties": {
                    "memory_ref": {
                        "type": "string",
                        "pattern": "^memoryref_[a-f0-9]+$",
                        "description": (
                            "Copy the opaque visible Memory Anchor when loading compressed "
                            "history. The runtime resolves it; never provide an internal trace ID."
                        ),
                    },
                    "section_handle": {
                        "type": "string",
                        "pattern": "^section_[a-f0-9]+$",
                        "description": (
                            "Optional opaque section address copied from a prior direct trace frame. "
                            "It narrows the same Memory Anchor; never invent it."
                        ),
                    },
                    "continuation_token": {
                        "type": "string",
                        "pattern": "^sectioncontinuation_[0-9a-f]+_[0-9a-f]{24}$",
                        "description": (
                            "Optional opaque cursor copied from an incomplete section. It returns "
                            "the next exact chunk of that same immutable section."
                        ),
                    },
                    "question": {"type": "string", "minLength": 1},
                    "entity_refs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "desired_detail": {"type": "string", "minLength": 1},
                    "purpose": {"type": "string", "minLength": 1},
                    "temporal_purpose": {
                        "type": "string",
                        "description": (
                            "Whether the semantic need is current code truth, execution history, "
                            "a comparison, or detail behind a visible compressed Memory Anchor."
                        ),
                        "enum": [
                            "current_workspace_truth",
                            "historical_execution",
                            "compare_history_to_current",
                            "memory_ref_detail",
                        ],
                    },
                    "preferred_page_relations": {
                        "type": "array",
                        "maxItems": 4,
                        "items": relation_items,
                    },
                    "required_page_relations": {
                        "type": "array",
                        "maxItems": 4,
                        "items": relation_items,
                    },
                    "page_relation_direction": {
                        "type": "string",
                        "enum": ["outgoing", "incoming", "both"],
                        "default": "both",
                    },
                    "rich_code_relations": {
                        "type": "array",
                        "maxItems": 4,
                        "items": {"type": "string", "minLength": 1},
                    },
                },
            },
        },
        {
            "type": "function",
            "name": "attribute_memory_use",
            "description": (
                "Record which previously recalled semantic entities actually influenced the "
                "current implementation, verification, correction, dependency, or decision."
            ),
            "inputSchema": {
                "type": "object",
                "additionalProperties": False,
                "required": ["entity_refs", "usage"],
                "properties": {
                    "delivery_id": {"type": "string", "minLength": 1},
                    "entity_refs": {
                        "type": "array",
                        "minItems": 1,
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "evidence_handles": {
                        "type": "array",
                        "maxItems": 16,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "usage": {"type": "string", "minLength": 1},
                },
            },
        },
    )
