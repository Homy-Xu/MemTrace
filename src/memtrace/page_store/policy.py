from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TailReason(StrEnum):
    RUN_END = "RUN_END"
    CRASH_RECOVERY = "CRASH_RECOVERY"
    FAULT_SAFE_CHECKPOINT = "FAULT_SAFE_CHECKPOINT"
    USER_CHECKPOINT = "USER_CHECKPOINT"
    ABSOLUTE_MAX_SAFETY = "ABSOLUTE_MAX_SAFETY"
    PAGE_SET_SEGMENT = "PAGE_SET_SEGMENT"


@dataclass(frozen=True, slots=True)
class PagePolicy:
    """Token policy for bounded Memory Trace bodies.

    Token counts are estimates over canonical, already-redacted EventGroups.
    Boundaries are evaluated only after a whole EventGroup has committed.
    """

    min_tokens: int = 2048
    target_tokens: int = 6144
    nominal_max_tokens: int = 8192
    absolute_max_tokens: int = 16384

    def __post_init__(self) -> None:
        values = (
            self.min_tokens,
            self.target_tokens,
            self.nominal_max_tokens,
            self.absolute_max_tokens,
        )
        if any(value <= 0 for value in values):
            raise ValueError("PagePolicy limits must be positive")
        if tuple(sorted(values)) != values or len(set(values)) != len(values):
            raise ValueError("PagePolicy must satisfy MIN < TARGET < NOMINAL_MAX < ABSOLUTE_MAX")


LEGAL_TAIL_REASONS = frozenset(TailReason)
