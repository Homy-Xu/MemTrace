"""Task-pressure-adaptive engagement of the state-consistent memory runtime.

The runtime always records traces, maintains Working Memory, performs Trace
Recall, and preserves Context Recovery; what varies is how deeply it steers
the model:

* ``PASSTHROUGH`` - one Milestone for the whole Task, no per-Turn Route Card,
  no intermediate acceptance.  Terminal verification, Epoch fences and the
  durable record remain.  For a 30-minute task this behaves like plain Codex.
* ``LIGHT`` - the projected Milestones are merged into at most
  ``light_max_milestones`` contiguous groups, the Route Card omits the CodeMap
  and predecessor handoff detail, and the previous Milestone's pages stay hot
  across the boundary instead of cooling immediately.
* ``FULL`` - the complete behaviour specified in the closure spec.

The initial level is decided once, right after Planning, from signals that
are available without touching the repository: native Plan item count,
extracted requirement count, Task length and projected Milestone count.
Execution Milestone granularity is fixed by that decision. During execution the level
can only move up (PASSTHROUGH -> LIGHT -> FULL) in response to real pressure:
the Provider context reaching URGENT/HARD, an Epoch replacement, or a
Milestone acceptance failure.  Escalation changes runtime guidance (Route
Card depth, Working Memory retention), never the frozen Plan.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import StrEnum
from typing import Any, Mapping, Sequence

from ..config import EngagementConfiguration
from ..contracts import CompletionCriterionSpec, MilestoneSpec, PlanSpec, PlanStepSpec


class EngagementLevel(StrEnum):
    PASSTHROUGH = "PASSTHROUGH"
    LIGHT = "LIGHT"
    FULL = "FULL"


_ORDER: tuple[EngagementLevel, ...] = (
    EngagementLevel.PASSTHROUGH,
    EngagementLevel.LIGHT,
    EngagementLevel.FULL,
)


def level_rank(level: EngagementLevel) -> int:
    return _ORDER.index(level)


@dataclass(frozen=True, slots=True)
class EngagementSignals:
    """Planning-time size estimate; every field is observable without tools."""

    plan_item_count: int
    requirement_count: int
    task_chars: int
    milestone_count: int

    def as_mapping(self) -> dict[str, int]:
        return {
            "plan_item_count": self.plan_item_count,
            "requirement_count": self.requirement_count,
            "task_chars": self.task_chars,
            "milestone_count": self.milestone_count,
        }


@dataclass(frozen=True, slots=True)
class EngagementDecision:
    level: EngagementLevel
    reason: str
    signals: EngagementSignals
    # Number of contiguous Milestone groups the projected Plan is folded into;
    # ``None`` keeps the projection as delivered by the model.
    milestone_groups: int | None


def decide_initial_level(
    config: EngagementConfiguration,
    signals: EngagementSignals,
) -> EngagementDecision:
    """Pick the starting engagement level from the Planning-time size estimate."""

    mode = config.mode
    if mode == "full":
        return EngagementDecision(EngagementLevel.FULL, "CONFIGURED_FULL", signals, None)
    if mode == "passthrough":
        return EngagementDecision(EngagementLevel.PASSTHROUGH, "CONFIGURED_PASSTHROUGH", signals, 1)
    if mode == "light":
        return EngagementDecision(
            EngagementLevel.LIGHT,
            "CONFIGURED_LIGHT",
            signals,
            min(config.light_max_milestones, max(1, signals.milestone_count)),
        )
    # adaptive
    if (
        signals.plan_item_count <= config.passthrough_max_plan_items
        and signals.requirement_count <= config.passthrough_max_requirements
        and signals.task_chars <= config.passthrough_max_task_chars
    ):
        return EngagementDecision(EngagementLevel.PASSTHROUGH, "SMALL_TASK_ESTIMATE", signals, 1)
    if signals.plan_item_count <= config.light_max_plan_items:
        return EngagementDecision(
            EngagementLevel.LIGHT,
            "MEDIUM_TASK_ESTIMATE",
            signals,
            min(config.light_max_milestones, max(1, signals.milestone_count)),
        )
    return EngagementDecision(EngagementLevel.FULL, "LARGE_TASK_ESTIMATE", signals, None)


def _contiguous_groups(count: int, groups: int) -> list[range]:
    """Split ``count`` ordered items into ``groups`` near-equal contiguous ranges."""

    groups = max(1, min(groups, count))
    base, extra = divmod(count, groups)
    ranges: list[range] = []
    start = 0
    for index in range(groups):
        size = base + (1 if index < extra else 0)
        ranges.append(range(start, start + size))
        start += size
    return ranges


def _unique(values: Sequence[str]) -> tuple[str, ...]:
    return tuple(dict.fromkeys(value for value in values if str(value).strip()))


def _joined_text(values: Sequence[str], *, separator: str = "\n") -> str:
    return separator.join(_unique(tuple(value.strip() for value in values)))


def regroup_milestones(plan: PlanSpec, groups: int) -> PlanSpec:
    """Fold the projected Milestones into ``groups`` contiguous Milestones.

    Everything the model committed to is preserved: every completion criterion
    (re-identified under the merged Milestone, requirement provenance kept),
    every verification method, entity reference, assumption, non-goal and
    native Plan source item.  Navigation Steps of the merged members are
    carried over in order.  Dependencies become the linear chain of merged
    Milestones.  Folding into one group yields the PASSTHROUGH single
    Milestone.
    """

    if groups <= 0:
        raise ValueError("milestone groups must be positive")
    members = plan.milestones
    if groups >= len(members):
        return plan
    merged: list[MilestoneSpec] = []
    for index, span in enumerate(_contiguous_groups(len(members), groups), start=1):
        group = members[span.start : span.stop]
        new_id = f"M{index:03d}"
        criterion_ids: dict[str, str] = {}
        criteria: list[CompletionCriterionSpec] = []
        for milestone in group:
            for criterion in milestone.criteria:
                local = criterion.criterion_id.split(".")[-1]
                candidate = f"{new_id}.{local}"
                if candidate in criterion_ids.values():
                    candidate = f"{new_id}.C{len(criteria) + 1:03d}"
                criterion_ids[f"{milestone.canonical_id}:{criterion.criterion_id}"] = candidate
                criteria.append(replace(criterion, criterion_id=candidate))
        steps: list[PlanStepSpec] = []
        for milestone in group:
            for step in milestone.steps:
                ordinal = len(steps) + 1
                remapped = tuple(
                    criterion_ids.get(f"{milestone.canonical_id}:{cid}", cid)
                    for cid in step.criterion_ids
                )
                steps.append(
                    replace(
                        step,
                        step_id=f"{new_id}.S{ordinal:03d}",
                        criterion_ids=_unique(remapped),
                    )
                )
        titles = _unique(tuple(item.title for item in group))
        title = titles[0] if len(titles) == 1 else " -> ".join(titles)
        if len(title) > 160:
            title = title[:157] + "..."
        merged.append(
            MilestoneSpec(
                canonical_id=new_id,
                title=title,
                description=_joined_text([item.description for item in group]),
                completion_criteria=(),
                verification=_unique(
                    tuple(method for item in group for method in item.verification)
                ),
                depends_on=(f"M{index - 1:03d}",) if index > 1 else (),
                status="pending",
                entity_refs=_unique(tuple(ref for item in group for ref in item.entity_refs)),
                objective=_joined_text([item.objective for item in group]),
                scope=_joined_text([item.scope for item in group]),
                target_outcome=_joined_text([item.target_outcome for item in group]),
                downstream_assumptions=_unique(
                    tuple(value for item in group for value in item.downstream_assumptions)
                ),
                non_goals=_unique(tuple(value for item in group for value in item.non_goals)),
                source_plan_item_ids=_unique(
                    tuple(value for item in group for value in item.source_plan_item_ids)
                ),
                criteria=tuple(criteria),
                steps=tuple(steps),
            )
        )
    return replace(plan, milestones=tuple(merged))


@dataclass(slots=True)
class EngagementState:
    """Mutable per-run engagement level with a monotonic escalation history."""

    initial: EngagementLevel
    reason: str
    signals: EngagementSignals
    level: EngagementLevel = field(init=False)
    transitions: list[dict[str, Any]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.level = self.initial

    @property
    def passthrough(self) -> bool:
        return self.level is EngagementLevel.PASSTHROUGH

    @property
    def light(self) -> bool:
        return self.level is EngagementLevel.LIGHT

    @property
    def full(self) -> bool:
        return self.level is EngagementLevel.FULL

    def escalate(self, *, reason: str, source_event_id: str | None = None) -> bool:
        """Move one level up; returns ``False`` when already FULL."""

        rank = level_rank(self.level)
        if rank >= len(_ORDER) - 1:
            return False
        previous = self.level
        self.level = _ORDER[rank + 1]
        self.transitions.append(
            {
                "from": previous.value,
                "to": self.level.value,
                "reason": reason,
                "source_event_id": source_event_id,
            }
        )
        return True

    def as_mapping(self) -> dict[str, Any]:
        return {
            "initial_level": self.initial.value,
            "level": self.level.value,
            "reason": self.reason,
            "signals": self.signals.as_mapping(),
            "transitions": list(self.transitions),
        }


def engagement_from_plan(
    config: EngagementConfiguration,
    *,
    plan: PlanSpec,
    user_task: str,
    requirement_count: int,
) -> tuple[EngagementDecision, PlanSpec]:
    """Decide the initial level and return the (possibly regrouped) Plan."""

    signals = EngagementSignals(
        plan_item_count=(
            len(plan.native_plan.items) if plan.native_plan is not None else len(plan.milestones)
        ),
        requirement_count=requirement_count,
        task_chars=len(user_task.strip()),
        milestone_count=len(plan.milestones),
    )
    decision = decide_initial_level(config, signals)
    if decision.milestone_groups is None:
        return decision, plan
    return decision, regroup_milestones(plan, decision.milestone_groups)


def summarize_engagement(state: EngagementState | None) -> Mapping[str, Any] | None:
    return None if state is None else state.as_mapping()
