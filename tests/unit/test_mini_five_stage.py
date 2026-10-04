from __future__ import annotations

from pathlib import Path

import pytest

from memtrace.cli import main
from memtrace.config import load_config
from memtrace.harness.mini_five_stage import (
    MiniSweAgentHarnessAdapter,
    sanitize_mini_messages_for_api,
)


class _FakeModel:
    def format_message(self, *, role: str, content: str) -> dict[str, str]:
        return {"role": role, "content": content}


class _FakeAgent:
    def __init__(self) -> None:
        self.model = _FakeModel()
        self.messages: list[dict[str, str]] = []
        self.config = type("Config", (), {"output_path": None})()

    def add_messages(self, *messages: dict[str, str]) -> None:
        self.messages.extend(messages)

    def query(self) -> dict[str, object]:
        return {
            "content": "1. Inspect the failing test\n2. Implement the fix\n3. Run the tests",
            "extra": {"actions": []},
        }


def test_deepswe_five_stage_profile_matches_best_campaign() -> None:
    root = Path(__file__).parents[2]
    config = load_config(
        root / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    assert config.provider.model == "deepseek-v4-flash-0731"
    assert config.provider.api_key_env == "MEMTENSOR_DOMESTIC_API_KEY"
    assert config.provider.base_url == "https://api-int.memtensor.cn/v1"
    assert config.provider.model_context_window == 200_000
    assert config.provider.native_compaction_enabled is False
    assert config.codex_sandbox_mode == "danger-full-access"
    assert config.stages.planning is True
    assert config.stages.page_store is True
    assert config.stages.semantic_memory is True
    assert config.stages.recall is True
    assert config.stages.context_runtime is True
    assert config.stages.rich_graph is True
    assert config.page_policy.min_tokens == 2048
    assert config.page_policy.target_tokens == 6144
    assert config.page_policy.nominal_max_tokens == 8192
    assert config.page_policy.absolute_max_tokens == 16384
    assert config.context_budget.model_limit == 200_000
    assert config.recall_max_pages == 8
    assert config.recall_max_tokens == 8192
    assert config.rich_prefetch_budget == 4
    assert config.run_budget.max_execution_turns == 200
    assert config.run_budget.wall_clock_seconds == 14_400
    assert dict(config.acceptance_budgets) == {
        "weak_progress": 4,
        "no_progress": 2,
        "semantic_review_rounds": 2,
        "unclaimed_boundaries": 3,
    }
    assert config.engagement.mode == "full"


def test_mini_five_stage_projects_one_behavioral_plan(tmp_path: Path) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    (repository / "README.md").write_text("fixture\n", encoding="utf-8")
    config = load_config(
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    adapter = MiniSweAgentHarnessAdapter(
        repository_path=repository,
        model="deepseek-v4-flash-0731",
        run_root=tmp_path / "run",
        provider=config.provider,
        reasoning_effort="high",
        runtime_factory=lambda **_: _FakeAgent(),
    )
    result = adapter.plan(
        user_task="Fix the reported failure and leave the tests passing.",
        planning_context="",
        run_id="run-fixture",
        revision_id="planning-revision",
        workspace_receipt=None,
        inject=False,
    )
    assert result.thread_id
    criteria = [
        criterion
        for milestone in result.plan.milestones
        for criterion in milestone.criteria
    ]
    assert criteria
    claim_types = {criterion.claim_type for criterion in criteria}
    assert "BEHAVIORAL" in claim_types
    assert "STRUCTURAL" not in claim_types
    driver = adapter.build_driver(
        native_compaction_timeout_seconds=180,
        native_compaction_enabled=False,
    )
    assert driver.capabilities().supports_plan_mode is True
    assert driver.capabilities().supports_native_compaction is False
    assert driver.capabilities().supports_context_replacement is True


def test_mini_harness_rejects_codex_only_flags(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    repository = tmp_path / "repo"
    repository.mkdir()
    task = tmp_path / "task.txt"
    task.write_text("Fix the reported failure.\n", encoding="utf-8")
    config = (
        Path(__file__).parents[2]
        / "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    code = main(
        [
            "run",
            "--harness",
            "mini_swe_agent",
            "--multilang-plan",
            "--repository",
            str(repository),
            "--task-file",
            str(task),
            "--run-root",
            str(tmp_path / "run"),
            "--config",
            str(config),
        ]
    )
    assert code == 2
    assert "only available with the Codex Harness" in capsys.readouterr().err


def test_memtensor_replay_drops_rejected_response_fields() -> None:
    cleaned = sanitize_mini_messages_for_api(
        [
            {
                "object": "response",
                "created_at": 1,
                "output": [
                    {
                        "type": "reasoning",
                        "id": "rs_1",
                        "summary": [{"type": "summary_text", "text": "look around"}],
                    },
                    {
                        "type": "function_call",
                        "id": "fc_1",
                        "call_id": "call_1",
                        "name": "bash",
                        "arguments": "{}",
                        "caller": {"type": "direct"},
                        "namespace": "functions",
                    },
                ],
            },
            {"type": "function_call_output", "call_id": "call_1", "output": "ok"},
        ]
    )
    assert all(item.get("object") != "response" and "created_at" not in item for item in cleaned)
    calls = [item for item in cleaned if item.get("type") == "function_call"]
    assert len(calls) == 1
    assert "caller" not in calls[0]
    assert "namespace" not in calls[0]
    assert calls[0]["call_id"] == "call_1"
    assert any(item.get("role") == "assistant" for item in cleaned)
    assistant = next(item for item in cleaned if item.get("role") == "assistant")
    assert assistant["content"][0]["type"] == "input_text"
