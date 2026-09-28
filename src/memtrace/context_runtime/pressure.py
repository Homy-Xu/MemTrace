from __future__ import annotations

from dataclasses import dataclass

from ..contracts import PressureLevel


@dataclass(frozen=True, slots=True)
class ContextBudget:
    model_limit: int
    system_overhead: int
    tool_schema_tokens: int
    output_reserve: int
    safety_margin: int
    soft_ratio: float = 0.60
    urgent_ratio: float = 0.75
    hard_ratio: float = 0.85

    def __post_init__(self) -> None:
        if self.model_limit <= 0:
            raise ValueError("model_limit must be positive")
        if any(
            value < 0
            for value in (
                self.system_overhead,
                self.tool_schema_tokens,
                self.output_reserve,
                self.safety_margin,
            )
        ):
            raise ValueError("context budget reserves cannot be negative")
        if self.effective_limit <= 0:
            raise ValueError("effective context limit must be positive")
        if not 0 < self.soft_ratio < self.urgent_ratio < self.hard_ratio < 1:
            raise ValueError("pressure ratios must be strictly ordered")

    @property
    def effective_limit(self) -> int:
        return (
            self.model_limit
            - self.system_overhead
            - self.tool_schema_tokens
            - self.output_reserve
            - self.safety_margin
        )


class PressurePolicy:
    def __init__(self, budget: ContextBudget) -> None:
        self.budget = budget

    def level(self, physical_tokens: int) -> PressureLevel:
        if physical_tokens < 0:
            raise ValueError("physical_tokens cannot be negative")
        ratio = physical_tokens / self.budget.effective_limit
        if ratio < self.budget.soft_ratio:
            return PressureLevel.NORMAL
        if ratio < self.budget.urgent_ratio:
            return PressureLevel.SOFT
        if ratio < self.budget.hard_ratio:
            return PressureLevel.URGENT
        return PressureLevel.HARD
