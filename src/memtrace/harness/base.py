"""Provider-neutral harness contracts used by all benchmark adapters.

The memory runtime consumes durable :class:`HarnessEvent` objects.  A
backend may expose richer native features, but it must report the same basic
session, plan, event, usage, checkpoint, and close lifecycle.
"""
from __future__ import annotations

import time
from collections import deque
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol

from .contracts import HarnessCapabilities, HarnessEvent


@dataclass(frozen=True, slots=True)
class UsageSnapshot:
    """Provider usage collected for one run.

    Values are optional because providers and older trajectory formats do not
    always expose token accounting.  Missing values remain ``None`` rather
    than being presented as zero.
    """

    api_calls: int = 0
    input_tokens: int | None = None
    output_tokens: int | None = None
    total_tokens: int | None = None
    cost: float | None = None
    cost_available: bool | None = None
    cost_source: str | None = None
    wall_time_seconds: float | None = None
    context: Mapping[str, Any] | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "api_calls": self.api_calls,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "total_tokens": self.total_tokens,
            "cost": self.cost,
            "cost_available": self.cost_available,
            "cost_source": self.cost_source,
            "wall_time_seconds": self.wall_time_seconds,
            "context": dict(self.context) if self.context is not None else None,
        }


@dataclass(frozen=True, slots=True)
class HarnessSession:
    run_id: str
    branch_id: str
    thread_id: str
    started_at: float
    capabilities: HarnessCapabilities


@dataclass(frozen=True, slots=True)
class HarnessCheckpoint:
    run_id: str
    checkpoint_id: str
    revision_id: str
    payload: Mapping[str, Any] = field(default_factory=dict)


class HarnessBackend(Protocol):
    """Stable lifecycle implemented by Codex and mini-swe-agent backends."""

    def capabilities(self) -> HarnessCapabilities: ...

    def start_session(self, *, run_id: str, branch_id: str, thread_id: str | None = None) -> HarnessSession: ...

    def plan(self, task: str, **kwargs: Any) -> Mapping[str, Any]: ...

    def next_event(self) -> HarnessEvent | None: ...

    def execute(self, task: str, **kwargs: Any) -> Iterable[HarnessEvent]: ...

    def checkpoint(self, revision_id: str, **payload: Any) -> HarnessCheckpoint: ...

    def resume(self, checkpoint: HarnessCheckpoint) -> HarnessSession: ...

    def usage(self) -> UsageSnapshot: ...

    def close(self) -> None: ...


class EventQueueMixin:
    """Small append-only event queue shared by backend wrappers."""

    def _init_event_queue(self) -> None:
        self._event_queue: deque[HarnessEvent] = deque()

    def _queue_event(self, event: HarnessEvent) -> HarnessEvent:
        self._event_queue.append(event)
        return event

    def next_event(self) -> HarnessEvent | None:
        return self._event_queue.popleft() if self._event_queue else None

    @staticmethod
    def _started_at() -> float:
        return time.monotonic()
