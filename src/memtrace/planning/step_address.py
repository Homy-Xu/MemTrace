from __future__ import annotations

import re
from dataclasses import dataclass

_STEP_ADDRESS = re.compile(
    r"^(?P<milestone>M\d{3,})[.\-](?P<kind>[SR])(?P<ordinal>\d+)$",
    re.IGNORECASE,
)
_STEP_LABEL_PREFIX = re.compile(
    r"^\s*(?P<milestone>M\d{3,})[.\-](?P<kind>[SR])(?P<ordinal>\d+)"
    r"(?=\s*(?::|\s-\s)|\s+|$)",
    re.IGNORECASE,
)


@dataclass(frozen=True, slots=True, order=True)
class PlanStepAddress:
    """Canonical identity key for a stage-local PlanStep address.

    Harnesses are free to render the same local ordinal as ``S1``, ``S01``
    or ``S001``.  Those are presentation aliases for one route node, not
    different Steps.  The Registry keeps the originally persisted ID for
    display and WAL compatibility, while this value object is the sole
    translation boundary for native Plan observations.
    """

    milestone_id: str
    kind: str
    ordinal: int

    @classmethod
    def parse(cls, value: str) -> PlanStepAddress | None:
        match = _STEP_ADDRESS.fullmatch(value.strip())
        return cls._from_match(match)

    @classmethod
    def parse_label(cls, value: str) -> PlanStepAddress | None:
        match = _STEP_LABEL_PREFIX.match(value)
        return cls._from_match(match)

    @classmethod
    def _from_match(cls, match: re.Match[str] | None) -> PlanStepAddress | None:
        if match is None:
            return None
        ordinal = int(match.group("ordinal"))
        if ordinal < 1:
            return None
        return cls(
            milestone_id=match.group("milestone").upper(),
            kind=match.group("kind").upper(),
            ordinal=ordinal,
        )
