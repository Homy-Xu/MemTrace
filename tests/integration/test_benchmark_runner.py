from __future__ import annotations

from pathlib import Path

from memtrace.benchmarks import BenchmarkRunner
from memtrace.harness.base import EventQueueMixin, HarnessCheckpoint, HarnessSession, UsageSnapshot
from memtrace.harness.contracts import HarnessCapabilities, HarnessEvent, HarnessEventType


class FakeBackend(EventQueueMixin):
    def __init__(self) -> None:
        self._init_event_queue()
        self._session = None

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            harness_name="fake",
            model_name="fake",
            tokenizer_id=None,
            context_limit=None,
            supports_plan_mode=True,
            supports_incremental_plan_updates=False,
            supports_thread_resume=True,
            supports_native_compaction=False,
            supports_token_usage_events=True,
            supports_tool_lifecycle_events=True,
            supports_context_replacement=False,
        )

    def start_session(self, *, run_id: str, branch_id: str, thread_id: str | None = None) -> HarnessSession:
        self._session = HarnessSession(run_id, branch_id, thread_id or "fake-thread", 0.0, self.capabilities())
        return self._session

    def plan(self, task: str, **_: object) -> dict[str, str]:
        return {"task_digest": task}

    def execute(self, task: str, **_: object):
        assert self._session is not None
        yield HarnessEvent(
            harness_event_id="fake:1",
            event_type=HarnessEventType.TURN_COMPLETED,
            thread_id=self._session.thread_id,
            turn_id="turn-1",
            sequence=1,
            provider_time_ms=None,
            run_id=self._session.run_id,
            branch_id=self._session.branch_id,
            revision_id="rev-1",
            source_event_id="fake:1",
            provider_method="fake",
            payload={"status": "ok"},
            raw_provider_summary={},
        )

    def checkpoint(self, revision_id: str, **payload: object) -> HarnessCheckpoint:
        assert self._session is not None
        return HarnessCheckpoint(self._session.run_id, "cp", revision_id, payload)

    def resume(self, checkpoint: HarnessCheckpoint) -> HarnessSession:
        assert self._session is not None
        return self._session

    def usage(self) -> UsageSnapshot:
        return UsageSnapshot(api_calls=1, total_tokens=4, cost=0.01, wall_time_seconds=0.1)

    def close(self) -> None:
        pass


def test_runner_writes_one_redacted_receipt(tmp_path: Path) -> None:
    result = BenchmarkRunner(FakeBackend(), receipt_dir=tmp_path).run(
        benchmark="smoke",
        task_id="fake-1",
        task="implement the requested change",
    )
    assert result.events == 1
    receipt = tmp_path / "smoke-fake-1.json"
    assert receipt.is_file()
    assert '"status": "COMPLETED"' in receipt.read_text()
