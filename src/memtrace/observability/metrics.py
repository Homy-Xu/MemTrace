from __future__ import annotations

import math
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Iterator

_DURATION_NAMES = {
    "time_to_first_agent_action_ms",
    "planning_to_execution_wait_ms",
    "planning_provider_ms",
    "route_delivery_ms",
    "step_review_ms",
    "milestone_review_ms",
    "page_finalization_ms",
    "event_group_commit_ms",
    "page_seal_ms",
    "semantic_projection_ms",
    "semantic_lookup_ms",
    "page_open_ms",
    "page_slice_ms",
    "context_assembly_ms",
    "context_admission_ms",
    "fault_total_ms",
    "rich_graph_queue_ms",
    "rich_graph_projection_ms",
    "rich_graph_blocking_ms",
    "rich_code_search_ms",
}


class CounterName(StrEnum):
    """The single executable contract for runtime counters.

    Counter call sites use this type instead of duplicating string literals.
    A Provider stall is counted as a recovery *attempt* here because the
    interrupt and same-Thread continuation have not succeeded at detection
    time yet.
    """

    MEMORY_DECISION_CALL = "memory_decision_call_count"
    PAGE_IN_EXACT_SECTION_HIT = "page_in_exact_section_hit_count"
    PAGE_IN_EXECUTABLE_SURFACE_HIT = "page_in_executable_surface_hit_count"
    PAGE_IN_CONTINUATION_ISSUED = "page_in_continuation_issued_count"
    PAGE_IN_NO_EXECUTABLE_SURFACE = "page_in_no_executable_surface_count"
    RECALL_AFTER_REPEATED_FILE_READ = "recall_after_repeated_file_read_count"
    RECALL_FOLLOWED_BY_ROUTE_PROGRESS = "recall_followed_by_route_progress_count"
    COMPRESSION = "compression_count"
    NATIVE_COMPACTION = "native_compaction_count"
    EPOCH = "epoch_count"
    REPEATED_FILE_READ = "repeated_file_read_count"
    RECOVERED_CONTEXT_TOKENS = "recovered_context_tokens"
    PROVIDER_NATIVE_COMPACTION = "provider_native_compaction_count"
    SEMANTIC_GRAPH_EXPANSION = "semantic_graph_expansion_count"
    PROVIDER_STALL_RECOVERY_ATTEMPT = "provider_stall_recovery_attempt_count"
    PROVIDER_STALL_RECOVERY_SUCCESS = "provider_stall_recovery_success_count"
    ROUTE_DELTA = "route_delta_count"
    ROUTE_DELTA_TOKENS = "route_delta_tokens"
    STEP_REVIEW = "step_review_count"
    MILESTONE_REVIEW = "milestone_review_count"
    REJECTED_CONTROL = "rejected_control_count"
    DUPLICATE_CONTROL = "duplicate_control_count"
    PAGE_FAULT = "page_fault_count"
    PAGE_FINALIZATION = "page_finalization_count"
    # Rich Graph consumption funnel.  Queries count every time a route
    # decision asked the graph; hits count answers with at least one candidate;
    # "used" counts candidates that actually entered a frozen contract or a
    # result binding.  A run with many queries and zero uses shows the graph
    # consumed CPU without contributing (UNUSED), which is a first-class
    # finding rather than a silent waste.
    RICH_ADDRESS_QUERY = "rich_address_query_count"
    RICH_ADDRESS_HIT = "rich_address_hit_count"
    RICH_ADDRESS_USED = "rich_address_used_count"
    RICH_OBSERVER_QUERY = "rich_observer_query_count"
    RICH_OBSERVER_HIT = "rich_observer_hit_count"
    RICH_OBSERVER_USED = "rich_observer_used_count"
    RICH_CHANGED_SYMBOLS_ATTACHED = "rich_changed_symbols_attached_count"
    RICH_CODE_MAP_QUERY = "rich_code_map_query_count"
    RICH_CODE_MAP_HIT = "rich_code_map_hit_count"
    RICH_CODE_SEARCH_QUERY = "rich_code_search_query_count"
    RICH_CODE_SEARCH_HIT = "rich_code_search_hit_count"
    RICH_CODE_SEARCH_USED = "rich_code_search_used_count"
    RICH_CODE_SEARCH_FAILED = "rich_code_search_failed_count"
    RICH_NAVIGATION_CARD_OFFERED = "rich_navigation_card_offered_count"
    RUN_BUDGET_EXHAUSTED = "run_budget_exhausted_count"
    EPOCH_NO_PROGRESS_BUDGET_EXHAUSTED = "epoch_no_progress_budget_exhausted_count"
    PROVIDER_TURN_FAILURE_RETRY = "provider_turn_failure_retry_count"
    PROVIDER_TURN_FAILURES_EXHAUSTED = "provider_turn_failures_exhausted_count"
    # Route Cards that carried an IMPLEMENT_NOW directive because the model
    # spent ``exploration_turns`` Turns of one Milestone only reading.
    EXPLORATION_DIRECTIVE_ISSUED = "exploration_directive_issued_count"
    # Physical fences that closed a read-only Turn after that directive, so the
    # Milestone was submitted for acceptance instead of granted another window.
    EXPLORATION_BUDGET_EXHAUSTED = "exploration_budget_exhausted_count"
    # Model recall requests refused because the Turn already received its
    # recall delivery budget (re-orientation bursts after an Epoch handoff).
    RECALL_TURN_BUDGET_EXHAUSTED = "recall_turn_budget_exhausted_count"
    # Adaptive engagement moved one level up (PASSTHROUGH -> LIGHT -> FULL).
    ENGAGEMENT_ESCALATION = "engagement_escalation_count"
    # The runtime re-ran the model's own test runner because its exit status
    # was hidden (pipe/filter) or predates the current revision.  Layer-1
    # deterministic verification: no model judgement is involved.
    RUNTIME_TEST_REOBSERVATION = "runtime_test_reobservation_count"
    # SWE-Milestone: the runtime-owned regression guard ran because the model
    # created an official ``agent-impl-*`` submission tag.  PASSED/FAILED are
    # authoritative host verifier outcomes; UNAVAILABLE means the guard could
    # not produce evidence (timeout/internal error) and nothing was recorded.
    SUBMISSION_GUARD_PASSED = "submission_guard_passed_count"
    SUBMISSION_GUARD_FAILED = "submission_guard_failed_count"
    SUBMISSION_GUARD_UNAVAILABLE = "submission_guard_unavailable_count"
    # Route Cards that carried a REGRESSION_GUARD_FAILED directive.
    SUBMISSION_GUARD_DIRECTIVE_ISSUED = "submission_guard_directive_issued_count"


_COUNTER_NAMES = frozenset(item.value for item in CounterName)

_GAUGE_NAMES = {
    "page_hit_rate",
    "first_relevant_page_rank",
    "fallback_stage",
    "graph_freshness_lag",
    "logical_context_tokens",
    "logical_context_pressure",
    "provider_context_tokens",
    "provider_context_limit",
    "provider_context_pressure",
    "provider_compaction_state",
    "semantic_graph_pages_opened",
}


@dataclass(frozen=True, slots=True)
class MetricSnapshot:
    values: dict[str, Any]
    samples: dict[str, dict[str, float | int | None]]
    semantics: dict[str, str]


class MetricRecorder:
    """Records observed timings; absent behavior is never presented as a hit."""

    def __init__(self) -> None:
        self._durations: dict[str, list[float]] = {name: [] for name in _DURATION_NAMES}
        self._counters = {name: 0 for name in _COUNTER_NAMES}
        self._gauges: dict[str, Any] = {name: None for name in _GAUGE_NAMES}
        self._lock = threading.Lock()

    @contextmanager
    def timer(self, name: str) -> Iterator[None]:
        if name not in _DURATION_NAMES:
            raise KeyError(name)
        started = time.perf_counter_ns()
        try:
            yield
        finally:
            self.observe_ms(name, (time.perf_counter_ns() - started) / 1_000_000)

    def observe_ms(self, name: str, value: float) -> None:
        if name not in _DURATION_NAMES:
            raise KeyError(name)
        if value < 0:
            raise ValueError("duration cannot be negative")
        with self._lock:
            self._durations[name].append(float(value))

    def increment(self, name: CounterName, amount: int = 1) -> None:
        if not isinstance(name, CounterName):
            raise TypeError("counter name must be a CounterName")
        with self._lock:
            self._counters[name.value] += int(amount)

    def set(self, name: str, value: Any) -> None:
        if name not in _GAUGE_NAMES:
            raise KeyError(name)
        with self._lock:
            self._gauges[name] = value

    def snapshot(self) -> MetricSnapshot:
        with self._lock:
            durations = {name: list(values) for name, values in self._durations.items()}
            counters = dict(self._counters)
            gauges = dict(self._gauges)
        values: dict[str, Any] = {}
        samples: dict[str, dict[str, float | int | None]] = {}
        for name, observed in durations.items():
            ordered = sorted(observed)
            values[name] = sum(observed) if observed else None
            samples[name] = {
                "count": len(observed),
                "p50": ordered[len(ordered) // 2] if ordered else None,
                "p95": (ordered[max(0, math.ceil(len(ordered) * 0.95) - 1)] if ordered else None),
                "max": max(ordered) if ordered else None,
            }
        values.update(counters)
        values.update(gauges)
        semantics = {
            "durations": "null means the operation did not occur; values are measured cumulative local milliseconds",
            "counters": "zero means the observed behavior did not occur in this run",
            "gauges": "null means no qualifying observation was available",
            "page_hit_rate": "latest recall: delivered addressed Page sections or generic body-verified Evidence hits per Page read; a partial continuation is a section hit, not complete coverage",
            "rich_address_used_count": "exposed Rich address candidates used in durable bindings or successful recall delivery; does not measure correctness or causal benefit",
            "provider_time": "not included in local framework timings",
            "logical_context_tokens": (
                "serialized ContextImage prompt-body estimate; it does not claim Provider token deletion"
            ),
            "provider_context_tokens": (
                "latest Codex tokenUsage.last.totalTokens; null until the Provider reports it"
            ),
            "provider_compaction_state": (
                "a compaction event is distinct from verified physical token reduction"
            ),
        }
        return MetricSnapshot(values=values, samples=samples, semantics=semantics)
