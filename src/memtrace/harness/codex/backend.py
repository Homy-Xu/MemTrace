from __future__ import annotations

import time
from collections.abc import Iterable, Mapping
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any

from ...config import CodexProviderConfiguration
from ..adapter import CodexHarnessAdapter
from ..base import EventQueueMixin, HarnessCheckpoint, HarnessSession, UsageSnapshot
from ..contracts import HarnessCapabilities, HarnessEvent


class CodexBackend(EventQueueMixin):
    """Thin provider-neutral wrapper around the verified Codex adapter."""

    def __init__(
        self,
        *,
        repository_path: Path,
        model: str,
        run_root: Path,
        provider: CodexProviderConfiguration | None = None,
        executable: str | None = None,
        reasoning_effort: str | None = None,
        timeout_seconds: float = 600.0,
    ) -> None:
        self._init_event_queue()
        self.repository_path = Path(repository_path).resolve()
        self.run_root = Path(run_root).resolve()
        self.model = model
        self._started_at = time.monotonic()
        self._sequence = 0
        self._session: HarnessSession | None = None
        self._adapter = CodexHarnessAdapter(
            repository_path=self.repository_path,
            model=model,
            run_root=self.run_root,
            provider=provider,
            executable=executable,
            reasoning_effort=reasoning_effort,
            timeout_seconds=timeout_seconds,
        )

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            harness_name="codex-app-server",
            model_name=self.model,
            tokenizer_id=None,
            context_limit=None,
            supports_plan_mode=True,
            supports_incremental_plan_updates=True,
            supports_thread_resume=True,
            supports_native_compaction=False,
            supports_token_usage_events=True,
            supports_tool_lifecycle_events=True,
            supports_context_replacement=False,
        )

    def start_session(self, *, run_id: str, branch_id: str, thread_id: str | None = None) -> HarnessSession:
        actual = self._adapter.start_or_resume_thread(thread_id=thread_id)
        self._session = HarnessSession(run_id, branch_id, actual, time.time(), self.capabilities())
        return self._session

    def plan(self, task: str, **kwargs: Any) -> Mapping[str, Any]:
        result = self._adapter.plan(user_task=task, **kwargs)
        plan = asdict(result.plan) if is_dataclass(result.plan) else str(result.plan)
        return {
            "thread_id": result.thread_id,
            "turn_id": result.turn_id,
            "model": result.model,
            "plan": plan,
            "read_only_verified": result.read_only_verified,
        }

    def execute(self, task: str, **kwargs: Any) -> Iterable[HarnessEvent]:
        """Yield the verified App Server stream.

        The low-level driver owns workspace revisions and durable memory-tool
        callbacks, so a production caller passes either ``driver`` or the
        driver's required ``revision_tracker``.  Requiring that context here
        is intentional: silently starting a second Codex runner would break
        the five-stage WAL and make a benchmark receipt incomparable.
        """

        if self._session is None:
            raise RuntimeError("start_session must be called before execute")
        driver = kwargs.pop("driver", None)
        if driver is None:
            from ..driver import CodexHarnessDriver

            driver = CodexHarnessDriver(self._adapter)
        memory_tool_handler = kwargs.pop("memory_tool_handler", None)
        if memory_tool_handler is not None:
            observer = kwargs.pop("memory_tool_response_observer", None)
            driver.bind_memory_tool_handler(memory_tool_handler, observer)
        revision_tracker = kwargs.pop("revision_tracker", None)
        if revision_tracker is None:
            raise RuntimeError(
                "Codex execution requires the RunCoordinator WorkspaceRevisionTracker; "
                "pass revision_tracker=... or use the production RunCoordinator entry point"
            )
        yield from driver.events(
            user_task=task,
            run_id=self._session.run_id,
            branch_id=self._session.branch_id,
            revision_tracker=revision_tracker,
            initial_milestone=kwargs.pop("initial_milestone", None),
            initial_semantic_route=kwargs.pop("initial_semantic_route", None),
        )
        if kwargs:
            unknown = ", ".join(sorted(kwargs))
            raise TypeError(f"unsupported Codex execution arguments: {unknown}")

    def checkpoint(self, revision_id: str, **payload: Any) -> HarnessCheckpoint:
        if self._session is None:
            raise RuntimeError("start_session must be called before checkpoint")
        checkpoint_id = f"{self._session.run_id}:{revision_id}:{self._session.thread_id}"
        return HarnessCheckpoint(self._session.run_id, checkpoint_id, revision_id, payload)

    def resume(self, checkpoint: HarnessCheckpoint) -> HarnessSession:
        if self._session is None:
            return self.start_session(
                run_id=checkpoint.run_id,
                branch_id=str(checkpoint.payload.get("branch_id", "main")),
                thread_id=str(checkpoint.payload.get("thread_id", "")) or None,
            )
        thread_id = str(checkpoint.payload.get("thread_id", self._session.thread_id))
        actual = self._adapter.start_or_resume_thread(thread_id=thread_id)
        self._session = HarnessSession(
            self._session.run_id,
            self._session.branch_id,
            actual,
            self._session.started_at,
            self.capabilities(),
        )
        return self._session

    def usage(self) -> UsageSnapshot:
        return UsageSnapshot(wall_time_seconds=time.monotonic() - self._started_at)

    @property
    def native_adapter(self) -> CodexHarnessAdapter:
        return self._adapter

    def close(self) -> None:
        self._adapter.close()
