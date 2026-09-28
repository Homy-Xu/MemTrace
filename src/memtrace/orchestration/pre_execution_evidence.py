from __future__ import annotations

from collections.abc import Mapping

from ..contracts import (
    Authority,
    Event,
    EventGroup,
    EvidenceDraft,
    FactType,
    PlanSpec,
    digest,
    stable_id,
)
from ..page_store import PageStore
from ..planning import PlanRegistry
from .models import RunRequest
from .trace import TraceRecorder


class PreExecutionEvidenceCoordinator:
    """Own the trusted runner-to-WAL historical evidence boundary.

    This component accepts only direct, exact benchmark observations supplied
    through the internal RunRequest field. It never manufactures current-code
    acceptance and never lets an offline scenario claim runner authority.
    """

    def __init__(
        self,
        request: RunRequest,
        registry: PlanRegistry,
        page_store: PageStore,
        trace: TraceRecorder,
    ) -> None:
        self.request = request
        self.registry = registry
        self.page_store = page_store
        self.trace = trace

    @staticmethod
    def host_managed(request: RunRequest) -> bool:
        """Whether benchmark preflight is explicitly part of the Agent contract.

        Ordinary benchmarks keep their official evaluator outside the Coding
        control plane. Merely configuring evaluator assets must not leak hidden
        selectors into Planning, turn them into Milestone acceptance, or let
        the runtime grade its own work. The legacy HOST_MANAGED task contract
        remains available for benchmarks that intentionally expose that tool.
        """

        return "HOST_MANAGED" in request.user_task.upper()

    @staticmethod
    def validated(request: RunRequest) -> dict[str, EvidenceDraft]:
        expected_revision = f"revision:{request.revision_id}"
        facts_by_selector: dict[str, EvidenceDraft] = {}
        for fact in request.trusted_pre_execution_evidence:
            content = fact.content
            selector = str(content.get("test_selector", "")).strip()
            if (
                fact.key.evidence_type is not FactType.TEST_FAILURE
                or fact.authority is not Authority.ASSERTED
                or fact.key.semantic_role != "trusted_pre_execution_failure"
                or fact.key.branch_scope != request.branch_id
                or fact.key.revision_constraint != expected_revision
                or fact.key.validity_requirement != "HISTORICAL_EXECUTION"
                or content.get("source_authority") != "RUNTIME_TRUSTED_BENCHMARK_PREFLIGHT"
                or content.get("outcome") != "FAILED"
                or not str(content.get("source_receipt_digest", "")).startswith("sha256:")
                or not selector
                or fact.key.canonical_entity_id != f"test:{selector}"
            ):
                raise ValueError("invalid trusted pre-execution Evidence contract")
            if selector in facts_by_selector:
                raise ValueError("duplicate trusted pre-execution test selector")
            facts_by_selector[selector] = fact
        return facts_by_selector

    @classmethod
    def planning_context(cls, request: RunRequest) -> str:
        if not cls.host_managed(request):
            return ""
        facts = cls.validated(request)
        if not facts:
            return ""
        lines = [
            "Runtime-trusted pre-execution context (not part of the user request):",
            "The benchmark runner already executed the declared FAIL_TO_PASS set in the "
            "untouched isolated workspace. The selectors below are only those with an exact "
            "FAILED JUnit outcome; noisy PASSED, ERROR, SKIPPED, or MISSING outcomes are not "
            "promoted to trusted failure facts. Do not repeat these exact baseline checks. "
            "These are historical facts only; they never verify modified current code. Every "
            "selector below is bound by the runtime to the terminal current-revision verifier "
            "contract. Use the failures as planning evidence, but do not copy, repartition, or "
            "retype these runtime-owned selector addresses in the Milestone projection.",
            "Exact benchmark acceptance selectors:",
            *(f"- {selector}" for selector in facts),
        ]
        return "\n\n" + "\n".join(lines)

    def _group(
        self,
        plan: PlanSpec,
    ) -> tuple[EventGroup | None, tuple[dict[str, object], ...]]:
        request = self.request
        if not request.trusted_pre_execution_evidence:
            return None, ()
        facts_by_selector = self.validated(request)
        current = self.registry.current(request.run_id)
        if not any(
            item.canonical_id == current.canonical_id for item in plan.milestones
        ):
            raise RuntimeError("current Milestone is absent from the active Plan")

        events: list[Event] = []
        summary: list[dict[str, object]] = []

        def append_fact(selector: str) -> None:
            source = facts_by_selector[selector]
            content = {
                **dict(source.content),
                "acceptance_scope": "HISTORICAL_CONTEXT_ONLY",
                "binding_authority": "RUNTIME_EXACT_TEST_SELECTOR",
            }
            event_id = stable_id(
                "event_",
                {
                    "run": request.run_id,
                    "kind": "TRUSTED_PRE_EXECUTION_EVIDENCE",
                    "selector": selector,
                    "source": source.key.key_digest,
                },
            )
            events.append(
                Event(
                    event_id=event_id,
                    event_type="TRUSTED_PRE_EXECUTION_EVIDENCE",
                    payload={
                        "schema": "codex-longterm-v2/trusted-pre-execution-evidence@1",
                        "source_authority": "RUNTIME_TRUSTED_BENCHMARK_PREFLIGHT",
                        "binding_authority": "RUNTIME_EXACT_TEST_SELECTOR",
                        "test_selector": selector,
                        "source_receipt_digest": content["source_receipt_digest"],
                    },
                    facts=(
                        EvidenceDraft(
                            key=source.key,
                            content=content,
                            authority=source.authority,
                            confidence=source.confidence,
                            must_preserve=source.must_preserve,
                        ),
                    ),
                    entity_refs=(source.key.canonical_entity_id,),
                    milestone_id=current.identity_id,
                    execution_phase="pre_execution_baseline",
                    revision_id=request.revision_id,
                )
            )
            summary.append(
                {
                    "test_selector": selector,
                    "outcome": "FAILED",
                }
            )

        for selector in facts_by_selector:
            append_fact(selector)

        return (
            EventGroup(
                group_id=stable_id(
                    "group_",
                    {
                        "run": request.run_id,
                        "kind": "TRUSTED_PRE_EXECUTION_EVIDENCE",
                        "milestone": current.identity_id,
                        "events": [item.event_id for item in events],
                    },
                ),
                group_type="TRUSTED_PRE_EXECUTION_EVIDENCE",
                run_id=request.run_id,
                branch_id=request.branch_id,
                revision_id=request.revision_id,
                events=tuple(events),
                milestone_id=current.identity_id,
                semantic_boundary=True,
            ),
            tuple(summary),
        )

    def commit(self, plan: PlanSpec) -> Mapping[str, object] | None:
        request = self.request
        if not self.host_managed(request):
            if request.trusted_pre_execution_evidence:
                self.trace.record(
                    "TRUSTED_PRE_EXECUTION_EVIDENCE_RESERVED_FOR_OFFICIAL_EVALUATOR",
                    fact_count=len(request.trusted_pre_execution_evidence),
                    model_visible=False,
                    route_acceptance=False,
                )
            return None
        current = self.registry.current(request.run_id)
        group, bindings = self._group(plan)
        if group is None:
            return None
        durable = tuple(self.page_store.has_durable_event(item.event_id) for item in group.events)
        if any(durable) and not all(durable):
            raise RuntimeError("trusted pre-execution EventGroup is only partially durable")
        if not all(durable):
            self.page_store.append_group(group, defer_seal=False)
            self.trace.record(
                "TRUSTED_PRE_EXECUTION_EVIDENCE_WAL_COMMITTED",
                group_id=group.group_id,
                event_count=len(group.events),
                current_milestone_id=current.identity_id,
            )
        else:
            self.trace.record(
                "TRUSTED_PRE_EXECUTION_EVIDENCE_RECOVERED_FROM_WAL",
                group_id=group.group_id,
                event_count=len(group.events),
                current_milestone_id=current.identity_id,
            )

        visible_bindings = bindings[:16]
        return {
            "kind": "TRUSTED_PRE_EXECUTION_EVIDENCE",
            "authority": "RUNTIME_TRUSTED_BENCHMARK_PREFLIGHT",
            "temporal_scope": "HISTORICAL_EXECUTION",
            "acceptance_scope": "HISTORICAL_CONTEXT_ONLY",
            "binding_rule": "EXACT_TEST_SELECTOR",
            "bound_fact_count": len(bindings),
            "binding_digest": digest(bindings),
            "visible_bindings": list(visible_bindings),
            "omitted_binding_count": len(bindings) - len(visible_bindings),
            "instruction": (
                "Do not repeat these baseline checks. Use them as historical diagnostic context; "
                "they neither advance a Step nor verify modified current code."
            ),
        }
