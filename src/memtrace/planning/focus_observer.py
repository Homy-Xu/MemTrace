from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


@dataclass(frozen=True, slots=True)
class FocusSignal:
    """One durable execution observation offered to the route cursor.

    The signal contains only facts that already entered the WAL.  It is not an
    acceptance receipt and it cannot complete a Milestone.
    """

    action_type: str
    fact_types: tuple[str, ...] = ()
    command: str = ""
    tool_succeeded: bool | None = None
    modified_files: tuple[str, ...] = ()
    accessed_files: tuple[str, ...] = ()
    entity_refs: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class FocusObservation:
    step_id: str
    basis: str
    matched_entities: tuple[str, ...] = ()


class RouteFocusObserver:
    """Advance lightweight Focus from strong, deterministic work signals.

    A Plan Step is a navigation hint.  This observer deliberately recognizes
    only high-signal combinations (for example an audit conclusion, a real
    workspace revision, or a successful test).  Ambiguous activity leaves the
    cursor where it is; the natural Milestone boundary remains the fallback.
    """

    _PREPARE_WORDS = frozenset(
        {
            "branch",
            "checkout",
            "switch",
            "prepare",
            "setup",
            "初始化",
            "准备",
            "分支",
        }
    )
    _INVESTIGATE_WORDS = frozenset(
        {
            "audit",
            "inspect",
            "investigate",
            "analyze",
            "analyse",
            "understand",
            "locate",
            "trace",
            "review",
            "调查",
            "审计",
            "分析",
            "定位",
            "检查",
            "熟悉",
        }
    )
    _IMPLEMENT_WORDS = frozenset(
        {
            "implement",
            "add",
            "change",
            "modify",
            "fix",
            "repair",
            "refactor",
            "support",
            "wire",
            "实现",
            "增加",
            "添加",
            "修改",
            "修复",
            "重构",
            "支持",
        }
    )
    _VERIFY_WORDS = frozenset(
        {
            "test",
            "verify",
            "validate",
            "regression",
            "check",
            "测试",
            "验证",
            "回归",
        }
    )
    _DOCUMENT_WORDS = frozenset(
        {"doc", "docs", "documentation", "readme", "document", "文档", "说明"}
    )
    _BRANCH_COMMAND = re.compile(
        r"(?:^|[;&|]\s*)git\s+(?:switch\s+(?:-c|--create)|checkout\s+(?:-b|-B)|branch\s+)",
        re.IGNORECASE,
    )

    @staticmethod
    def _words(value: str) -> frozenset[str]:
        lowered = value.casefold()
        latin = re.findall(r"[a-z0-9_]+", lowered)
        return frozenset((*latin, *(char for char in lowered if "\u4e00" <= char <= "\u9fff")))

    @classmethod
    def _contains_any(cls, text: str, words: frozenset[str]) -> bool:
        lowered = text.casefold()
        tokens = cls._words(text)
        return bool(tokens.intersection(words)) or any(word in lowered for word in words)

    @staticmethod
    def _normalized_entities(values: Sequence[str]) -> frozenset[str]:
        return frozenset(str(value).strip() for value in values if str(value).strip())

    @classmethod
    def observe(
        cls,
        step: Mapping[str, object] | None,
        signal: FocusSignal,
    ) -> FocusObservation | None:
        if step is None:
            return None
        step_id = str(step.get("step_id", "")).strip()
        if not step_id:
            return None
        text = " ".join(
            value
            for value in (
                str(step.get("title", "")),
                str(step.get("expected_outcome", "")),
            )
            if value
        )
        fact_types = frozenset(signal.fact_types)
        step_entities = cls._normalized_entities(step.get("entity_refs", ()))
        action_entities = cls._normalized_entities(signal.entity_refs)
        matched_entities = tuple(sorted(step_entities.intersection(action_entities)))

        if (
            cls._contains_any(text, cls._PREPARE_WORDS)
            and signal.tool_succeeded is True
            and cls._BRANCH_COMMAND.search(signal.command)
        ):
            return FocusObservation(step_id, "SUCCESSFUL_BRANCH_PREPARATION", matched_entities)

        # A successful code read also produces CODE_OBSERVATION facts, but it
        # is only raw investigation material.  Do not move Focus after the
        # first ``sed``/``rg`` command: wait for the bounded semantic update
        # emitted after the investigation has reached a reusable conclusion.
        investigation_conclusion = (
            "IMPLEMENTATION_DECISION" in fact_types
            or (
                signal.action_type == "MEMORY_TOOL_RESULT"
                and "CODE_OBSERVATION" in fact_types
            )
        )
        if cls._contains_any(text, cls._INVESTIGATE_WORDS) and investigation_conclusion:
            return FocusObservation(step_id, "DURABLE_INVESTIGATION_CONCLUSION", matched_entities)

        modifies_workspace = bool(signal.modified_files) or "CODE_CHANGE" in fact_types
        if cls._contains_any(text, cls._DOCUMENT_WORDS) and modifies_workspace:
            documentation = tuple(
                path
                for path in signal.modified_files
                if path.casefold().endswith((".md", ".rst", ".txt"))
                or "doc" in path.casefold()
            )
            if documentation:
                return FocusObservation(step_id, "DOCUMENTATION_REVISION", matched_entities)

        if cls._contains_any(text, cls._VERIFY_WORDS):
            if fact_types.intersection({"TEST_RESULT", "VERIFIER_RESULT"}):
                return FocusObservation(step_id, "SUCCESSFUL_VERIFICATION_RESULT", matched_entities)
            if modifies_workspace and any(
                "test" in path.casefold() for path in signal.modified_files
            ):
                return FocusObservation(step_id, "TEST_IMPLEMENTATION_REVISION", matched_entities)

        if cls._contains_any(text, cls._IMPLEMENT_WORDS) and modifies_workspace:
            return FocusObservation(step_id, "WORKSPACE_IMPLEMENTATION_REVISION", matched_entities)

        # Entity-scoped custom Steps may not use one of the common verbs.  A
        # real mutation is still a strong observation when it touches the
        # Step's declared target.  Empty entity declarations never enter this
        # fallback, which prevents generic commands from skipping work.
        if modifies_workspace and matched_entities:
            return FocusObservation(step_id, "DECLARED_ENTITY_REVISION", matched_entities)
        return None
