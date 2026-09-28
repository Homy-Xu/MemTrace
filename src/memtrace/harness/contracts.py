from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Mapping


class HarnessEventType(StrEnum):
    THREAD_STARTED = "THREAD_STARTED"
    THREAD_RESUMED = "THREAD_RESUMED"
    TURN_STARTED = "TURN_STARTED"
    PLAN_PROPOSED = "PLAN_PROPOSED"
    PLAN_UPDATED = "PLAN_UPDATED"
    MILESTONE_MANIFEST = "MILESTONE_MANIFEST"
    PLAN_PROJECTION_DECISION = "PLAN_PROJECTION_DECISION"
    CANONICAL_PLAN_INJECTION = "CANONICAL_PLAN_INJECTION"
    PLANNING_VALIDATED = "PLANNING_VALIDATED"
    PLANNING_TURN_RETRIED = "PLANNING_TURN_RETRIED"
    ITEM_STARTED = "ITEM_STARTED"
    ITEM_COMPLETED = "ITEM_COMPLETED"
    TOOL_INTENT = "TOOL_INTENT"
    TOOL_RESULT = "TOOL_RESULT"
    FILE_CHANGED = "FILE_CHANGED"
    WORKSPACE_REVISION_ADVANCED = "WORKSPACE_REVISION_ADVANCED"
    TOKEN_USAGE_UPDATED = "TOKEN_USAGE_UPDATED"
    CONTEXT_COMPACTED = "CONTEXT_COMPACTED"
    NATIVE_COMPACTION_TIMEOUT = "NATIVE_COMPACTION_TIMEOUT"
    PROVIDER_STALLED = "PROVIDER_STALLED"
    MEMORY_TOOL_RESULT = "MEMORY_TOOL_RESULT"
    TURN_QUIESCED = "TURN_QUIESCED"
    TURN_COMPLETED = "TURN_COMPLETED"
    PHYSICAL_CONTEXT_FAILURE = "PHYSICAL_CONTEXT_FAILURE"
    THREAD_UNRECOVERABLE = "THREAD_UNRECOVERABLE"
    SESSION_LOST = "SESSION_LOST"


@dataclass(frozen=True, slots=True)
class HarnessCapabilities:
    harness_name: str
    model_name: str
    tokenizer_id: str | None
    context_limit: int | None
    supports_plan_mode: bool
    supports_incremental_plan_updates: bool
    supports_thread_resume: bool
    supports_native_compaction: bool
    supports_token_usage_events: bool
    supports_tool_lifecycle_events: bool
    supports_context_replacement: bool


@dataclass(frozen=True, slots=True)
class HarnessEvent:
    harness_event_id: str
    event_type: HarnessEventType
    thread_id: str
    turn_id: str | None
    sequence: int
    provider_time_ms: int | None
    run_id: str
    branch_id: str
    revision_id: str
    source_event_id: str
    provider_method: str
    payload: Mapping[str, Any]
    raw_provider_summary: Mapping[str, Any]


SIDE_EFFECT_ITEM_TYPES = frozenset(
    {"commandExecution", "fileChange", "mcpToolCall", "dynamicToolCall"}
)


# Native context compaction may summarize a malformed tool call as if the tool
# itself had disappeared.  The registered schema, not that historical guess,
# is the durable ABI.  Keep one runtime-owned statement shared by the native
# compactor, Thread developer instructions, and the post-compaction Working
# Set refresh so an erroneous summary cannot become self-reinforcing.
COMPACTION_TOOL_ABI_INVARIANT = (
    "Registered Codex tool schemas remain authoritative after context compaction. "
    "An earlier invalid tool name or argument shape is not evidence that tools are "
    "unavailable. Never invent alternate tool names. Correct the call against the "
    "currently registered schema; for the pinned runtime, shell commands use "
    "exec_command with the cmd field and native plan entries contain only step and "
    "status. Use apply_patch only when it is actually registered; an unsupported "
    "apply_patch response does not make exec_command or the runtime dynamic tools "
    "unavailable. Only an error returned by a valid registered call is evidence of "
    "that tool's failure."
)
