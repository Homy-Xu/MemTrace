from __future__ import annotations

from pathlib import Path

from memtrace.harness import HarnessEventType, MiniSweAgentBackend


def test_miniswe_backend_has_explicit_version_and_capabilities(tmp_path: Path) -> None:
    backend = MiniSweAgentBackend(
        repository_path=tmp_path,
        model="openai/test-model",
        run_root=tmp_path / "run",
    )
    assert backend.capabilities().harness_name == "mini-swe-agent"
    assert backend.capabilities().supports_plan_mode is False
    session = backend.start_session(run_id="r1", branch_id="main")
    plan = backend.plan("fix the supplied repository")
    assert session.thread_id.startswith("mini-")
    assert plan["native_plan"] is False
    checkpoint = backend.checkpoint("rev-1")
    assert checkpoint.run_id == "r1"
    assert backend.next_event() is None


def test_event_type_contains_usage_and_workspace_boundaries() -> None:
    assert HarnessEventType.TOKEN_USAGE_UPDATED.value == "TOKEN_USAGE_UPDATED"
    assert HarnessEventType.WORKSPACE_REVISION_ADVANCED.value == "WORKSPACE_REVISION_ADVANCED"
