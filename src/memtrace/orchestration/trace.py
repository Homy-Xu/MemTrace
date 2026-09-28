from __future__ import annotations

from typing import Any

from .models import TraceEvent


class TraceRecorder:
    """Ordered, factual production trace used by results and acceptance tests."""

    def __init__(self) -> None:
        self._events: list[TraceEvent] = []

    def record(self, name: str, **details: Any) -> TraceEvent:
        event = TraceEvent(len(self._events) + 1, name, details)
        self._events.append(event)
        return event

    def snapshot(self) -> tuple[TraceEvent, ...]:
        return tuple(self._events)
