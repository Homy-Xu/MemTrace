from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from memtrace.benchmarks.spec import BenchmarkTask
from memtrace.harness.base import EventQueueMixin, HarnessCheckpoint, HarnessSession, UsageSnapshot
from memtrace.harness.contracts import HarnessCapabilities, HarnessEvent, HarnessEventType
from memtrace.harness.mini_swe_agent import MiniSweAgentBackend


class FixtureCodexBackend(EventQueueMixin):
    """Deterministic App Server-shaped fixture for the offline smoke matrix."""

    def __init__(self) -> None:
        self._init_event_queue()
        self.session: HarnessSession | None = None

    def capabilities(self) -> HarnessCapabilities:
        return HarnessCapabilities(
            harness_name="codex-app-server",
            model_name="fixture-codex",
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
        self.session = HarnessSession(run_id, branch_id, thread_id or "fixture-thread", 0.0, self.capabilities())
        return self.session

    def plan(self, task: str, **_: object) -> dict[str, str]:
        return {"task_digest": str(hash(task))}

    def execute(self, task: str, **_: object):
        assert self.session is not None
        yield HarnessEvent(
            harness_event_id="fixture:turn",
            event_type=HarnessEventType.TURN_COMPLETED,
            thread_id=self.session.thread_id,
            turn_id="turn-1",
            sequence=1,
            provider_time_ms=1,
            run_id=self.session.run_id,
            branch_id=self.session.branch_id,
            revision_id="fixture-revision",
            source_event_id="fixture:turn",
            provider_method="turn/completed",
            payload={"status": "completed"},
            raw_provider_summary={"fixture": True},
        )

    def checkpoint(self, revision_id: str, **payload: object) -> HarnessCheckpoint:
        assert self.session is not None
        return HarnessCheckpoint(self.session.run_id, "fixture-checkpoint", revision_id, payload)

    def resume(self, checkpoint: HarnessCheckpoint) -> HarnessSession:
        assert self.session is not None
        return self.session

    def usage(self) -> UsageSnapshot:
        return UsageSnapshot(api_calls=1, input_tokens=3, output_tokens=2, total_tokens=5, cost=0.01, wall_time_seconds=0.01)

    def close(self) -> None:
        return None


@pytest.mark.parametrize("benchmark", ["swe-milestone", "deepswe", "swe-evo"])
def test_codex_benchmark_smoke_fixture(tmp_path: Path, benchmark: str) -> None:
    from memtrace.benchmarks import deepswe, swe_evo, swe_milestone

    task = BenchmarkTask(benchmark=benchmark, task_id="fixture", task="run the fixture")
    adapter = {"swe-milestone": swe_milestone, "deepswe": deepswe, "swe-evo": swe_evo}[benchmark]
    result = adapter.run_task(FixtureCodexBackend(), task, receipt_dir=tmp_path)
    assert result.receipt["status"] == "COMPLETED"
    assert result.receipt["evaluation"]["status"] == "PENDING"


@pytest.mark.parametrize("benchmark", ["swe-milestone", "deepswe", "swe-evo"])
def test_miniswe_benchmark_smoke_fixture(tmp_path: Path, benchmark: str, monkeypatch) -> None:
    from memtrace.benchmarks import deepswe, swe_evo, swe_milestone

    class FakeAgent:
        def __init__(self, config):
            self.config = config

        def run(self, task: str):
            self.config["output_path"].write_text(
                json.dumps({"info": {"model_stats": {"api_calls": 1, "instance_cost": 0.01}}}),
                encoding="utf-8",
            )
            return {"exit_status": "Submitted", "submission": ""}

    package = types.ModuleType("minisweagent")
    package.__version__ = "2.4.6"
    agents = types.ModuleType("minisweagent.agents")
    agents.get_agent = lambda model, environment, config, default_type="": FakeAgent(config)
    models = types.ModuleType("minisweagent.models")
    models.get_model = lambda config: object()
    environments = types.ModuleType("minisweagent.environments")
    environments.get_environment = lambda config, default_type="": object()
    for name, module in {
        "minisweagent": package,
        "minisweagent.agents": agents,
        "minisweagent.models": models,
        "minisweagent.environments": environments,
    }.items():
        monkeypatch.setitem(sys.modules, name, module)

    task = BenchmarkTask(benchmark=benchmark, task_id="fixture", task="run the fixture")
    backend = MiniSweAgentBackend(repository_path=tmp_path, model="fixture", run_root=tmp_path / "run")
    adapter = {"swe-milestone": swe_milestone, "deepswe": deepswe, "swe-evo": swe_evo}[benchmark]
    result = adapter.run_task(backend, task, receipt_dir=tmp_path / "receipts")
    assert result.receipt["status"] == "COMPLETED"
    assert result.receipt["harness_version"] == "2.4.6"
