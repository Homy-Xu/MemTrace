"""Opt-in native Plan input compatibility for non-Python benchmark runs.

This only exposes immutable source blocks. Milestones, Focus cursors and
acceptance are still produced by the ordinary CodexPlanNormalizer projection.
The Python normalizer deliberately remains unchanged.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from ..contracts import (
    ClaimType,
    CommitmentLevel,
    CompletionCriterionSpec,
    FactType,
    MilestoneSpec,
    PlanSpec,
)
from .memory_tools import EXTERNAL_VERIFICATION_TOOL
from .normalizer import CodexPlanNormalizer

REGRESSION_GUARD_CRITERION_SUFFIX = "HOST"
REGRESSION_GUARD_VERIFICATION = (
    f"HOST_MANAGED {EXTERNAL_VERIFICATION_TOOL}: the runtime-owned SWE-Milestone regression "
    "guard runs the affected test units offline, calibrates failures against the previous "
    "submission, and rejects manifests or out-of-scope edits the official evaluator cannot use"
)

_HEADING = re.compile(r"^\s{0,3}(#{1,6})\s+(.+?)\s*#*\s*$")
_ITEM = re.compile(r"^(\s*)(?:[-+*]|\d+[.)])\s+(.+)$")
_WORK_SECTIONS = frozenset(
    {
        "key changes",
        "implementation",
        "implementation changes",
        "implementation plan",
        "test plan",
        "testing",
        "verification",
        "validation",
        "changes",
        "关键变更",
        "实现计划",
        "实现改动",
        "测试计划",
        "验证计划",
    }
)


def use_multilang_plan(*, enabled: bool, language: str | None) -> bool:
    """Parser packages alone must not opt Python tasks into this policy."""
    return enabled and language in {"go", "rust", "typescript", "javascript", "java", "groovy"}


class MultilangPlanNormalizer(CodexPlanNormalizer):
    @staticmethod
    def native_plan_artifact_steps(text: str) -> tuple[Mapping[str, str], ...]:
        original = CodexPlanNormalizer.native_plan_artifact_steps(text)
        if original:
            return original
        # This method is called only for an App Server completed `plan` item,
        # not agentMessage, tool output or an unfinished streaming fragment.
        sections: list[tuple[str, list[str]]] = []
        heading = ""
        body: list[str] = []
        fence: str | None = None
        for line in text.splitlines():
            stripped = line.strip()
            marker = stripped[:3]
            if marker in {"```", "~~~"}:
                fence = None if fence == marker else marker if fence is None else fence
            match = _HEADING.match(line) if fence is None else None
            if match and len(match[1]) <= 2:
                sections.append((heading, body))
                heading, body = match[2].strip().strip("*_`:").casefold(), []
            else:
                body.append(line)
        sections.append((heading, body))

        blocks: list[str] = []
        for section, lines in sections:
            if section not in _WORK_SECTIONS:
                continue
            entries = []
            fence = None
            for index, line in enumerate(lines):
                marker = line.strip()[:3]
                if marker in {"```", "~~~"}:
                    fence = None if fence == marker else marker if fence is None else fence
                    continue
                match = _ITEM.match(line) if fence is None else None
                if match:
                    entries.append((match, index))
            if entries:
                top = min(len(match[1].expandtabs(4)) for match, _ in entries)
                starts = [index for match, index in entries if len(match[1].expandtabs(4)) == top]
                for start, end in zip(starts, [*starts[1:], len(lines)], strict=True):
                    # Include nested bullets and continuation lines in the source
                    # block; they are not extra execution Steps or contracts.
                    block = "\n".join(lines[start:end]).strip()
                    if block:
                        blocks.append(f"{section}:\n{block}")
            else:
                # A work section may use prose. Keep paragraphs as source
                # blocks, without inventing actions from Summary/Assumptions.
                blocks.extend(
                    f"{section}:\n{paragraph.strip()}"
                    for paragraph in re.split(r"\n\s*\n", "\n".join(lines))
                    if paragraph.strip()
                )
        return tuple(
            {"source_step_id": f"N{n:03d}", "step": block, "status": "pending"}
            for n, block in enumerate(blocks, 1)
        )


def regression_guard_criterion(canonical_id: str) -> CompletionCriterionSpec:
    """The one runtime-owned acceptance criterion every SWE-Milestone Milestone carries."""

    return CompletionCriterionSpec(
        criterion_id=f"{canonical_id}.{REGRESSION_GUARD_CRITERION_SUFFIX}",
        requirement_id=f"{canonical_id}.R-{REGRESSION_GUARD_CRITERION_SUFFIX}",
        requirement_text=(
            "The official SWE-Milestone evaluator applies only the submitted source tree "
            "and root manifests on top of the milestone end state and requires every "
            "previously passing test to keep passing"
        ),
        observable_outcome=(
            "The runtime-owned regression guard passes on the current repository "
            "revision: affected test units that pass at the previous submission still "
            "pass, root manifests resolve offline, affected code compiles, and no edit "
            "outside the submitted source directories is required"
        ),
        claim_type=ClaimType.REGRESSION,
        required_evidence_types=(FactType.VERIFIER_RESULT,),
        test_selectors=(EXTERNAL_VERIFICATION_TOOL,),
        required=True,
        commitment_level=CommitmentLevel.MILESTONE,
    )


def attach_regression_guard(plan: PlanSpec) -> PlanSpec:
    """Return ``plan`` with the HOST_MANAGED regression guard on every Milestone.

    The guard is additive: model-projected criteria, Steps, native Plan
    binding and ordering are untouched, so the frozen ``CodexPlanNormalizer``
    validation that already ran remains valid.  Milestones that already carry
    a ``verify_current_milestone`` selector are left as they are.
    """

    milestones: list[MilestoneSpec] = []
    for milestone in plan.milestones:
        has_guard = any(
            EXTERNAL_VERIFICATION_TOOL in criterion.test_selectors
            for criterion in milestone.criteria
        )
        if has_guard:
            milestones.append(milestone)
            continue
        verification = tuple(
            dict.fromkeys((*milestone.verification, REGRESSION_GUARD_VERIFICATION))
        )
        milestones.append(
            replace(
                milestone,
                completion_criteria=(),
                criteria=(*milestone.criteria, regression_guard_criterion(milestone.canonical_id)),
                verification=verification,
            )
        )
    final_has_guard = any(
        EXTERNAL_VERIFICATION_TOOL in criterion.test_selectors for criterion in plan.final_acceptance
    )
    final_acceptance = plan.final_acceptance
    final_verification = plan.final_verification
    if not final_has_guard:
        final_acceptance = (*plan.final_acceptance, regression_guard_criterion("TASK.FINAL"))
        final_verification = tuple(
            dict.fromkeys((*plan.final_verification, REGRESSION_GUARD_VERIFICATION))
        )
    return replace(
        plan,
        milestones=tuple(milestones),
        final_acceptance=final_acceptance,
        final_verification=final_verification,
    )


class SweMilestonePlanNormalizer(MultilangPlanNormalizer):
    """Multilang projection plus the host-managed SWE-Milestone regression guard.

    Official SWE-Milestone hides the FAIL_TO_PASS tests, so no exact benchmark
    selector can be bound before Planning the way SWE-EVO does.  The one thing
    the runtime *can* own is the regression contract, and it must be part of
    every Milestone's acceptance so that ``_run_automatic_trusted_verifier_if_ready``
    and the missing-evidence gate apply at each boundary.
    """

    def normalize(
        self,
        *,
        user_task: str,
        steps: Sequence[Mapping[str, Any]],
        final_plan_text: str | None = None,
        native_plan_text: str | None = None,
    ) -> PlanSpec:
        return attach_regression_guard(
            super().normalize(
                user_task=user_task,
                steps=steps,
                final_plan_text=final_plan_text,
                native_plan_text=native_plan_text,
            )
        )

    def project_native_plan(
        self,
        *,
        user_task: str,
        steps: Sequence[Mapping[str, Any]],
        projection: Mapping[str, Any],
        native_plan_text: str | None = None,
    ) -> PlanSpec:
        return attach_regression_guard(
            super().project_native_plan(
                user_task=user_task,
                steps=steps,
                projection=projection,
                native_plan_text=native_plan_text,
            )
        )
