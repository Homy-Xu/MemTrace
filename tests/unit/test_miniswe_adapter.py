from __future__ import annotations

import json
import sys
import types
from pathlib import Path

from memtrace.harness.contracts import HarnessEventType
from memtrace.harness.mini_swe_agent import MiniSweAgentBackend


def test_miniswe_adapter_normalizes_synthetic_trajectory(tmp_path: Path, monkeypatch) -> None:
    class FakeAgent:
        def __init__(self, config):
            self.config = config

        def run(self, task: str):
            output = self.config["output_path"]
            output.write_text(
                json.dumps(
                    {
                        "info": {"model_stats": {"api_calls": 2, "instance_cost": 0.12}},
                        "messages": [],
                    }
                )
            )
            return {"exit_status": "Submitted", "submission": ""}

    agents = types.ModuleType("minisweagent.agents")
    agents.get_agent = lambda model, environment, config, default_type="": FakeAgent(config)
    models = types.ModuleType("minisweagent.models")
    models.get_model = lambda config: object()
    environments = types.ModuleType("minisweagent.environments")
    environments.get_environment = lambda config, default_type="": object()
    package = types.ModuleType("minisweagent")
    monkeypatch.setitem(sys.modules, "minisweagent", package)
    monkeypatch.setitem(sys.modules, "minisweagent.agents", agents)
    monkeypatch.setitem(sys.modules, "minisweagent.models", models)
    monkeypatch.setitem(sys.modules, "minisweagent.environments", environments)

    backend = MiniSweAgentBackend(
        repository_path=tmp_path,
        model="openai/test",
        run_root=tmp_path / "run",
    )
    backend.start_session(run_id="r", branch_id="main")
    events = tuple(backend.execute("make the requested change"))
    assert [event.event_type for event in events] == [
        HarnessEventType.THREAD_STARTED,
        HarnessEventType.TURN_STARTED,
        HarnessEventType.TURN_COMPLETED,
        HarnessEventType.TOKEN_USAGE_UPDATED,
        HarnessEventType.WORKSPACE_REVISION_ADVANCED,
    ]
    assert backend.usage().api_calls == 2
    assert backend.usage().cost == 0.12
