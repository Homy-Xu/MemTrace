from __future__ import annotations

import pytest

from memtrace.harness.mini_swe_agent.context import (
    AgentStalledError,
    ContextBudget,
    ContextWindowGuard,
    ProgressGuard,
)


class _Model:
    def __init__(self) -> None:
        self.messages = []

    def query(self, messages, **kwargs):
        self.messages = list(messages)
        return {"message": {"role": "assistant", "content": "ok"}, "extra": {}}


def test_context_guard_preserves_task_prefix_and_recent_tool_pair() -> None:
    delegate = _Model()
    guard = ContextWindowGuard(
        delegate,
        model_name="test/model",
        budget=ContextBudget(
            model_limit=20_000,
            system_overhead=1_000,
            tool_schema_tokens=1_000,
            output_reserve=1_000,
            safety_margin=1_000,
        ),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        *({"role": "user", "content": "old " + ("x" * 5_000)} for _ in range(8)),
        {"role": "assistant", "tool_calls": [{"id": "call-last"}], "content": ""},
        {"role": "tool", "tool_call_id": "call-last", "content": "latest output"},
    ]
    result = guard.query(messages)
    kept = delegate.messages
    assert result["extra"]["context_guard"]["trimmed"] is True
    assert kept[:2] == messages[:2]
    assert messages[-2] in kept
    assert messages[-1] in kept
    assert kept[-1]["content"].startswith("The context guard compacted")
    assert result["extra"]["context_guard"]["kept_messages"] < len(messages)
    assert result["extra"]["context_guard"]["synopsis_included"] is True


def test_context_guard_includes_bounded_history_synopsis() -> None:
    delegate = _Model()
    guard = ContextWindowGuard(
        delegate,
        model_name="test/model",
        budget=ContextBudget(
            model_limit=20_000,
            system_overhead=1_000,
            tool_schema_tokens=1_000,
            output_reserve=1_000,
            safety_margin=1_000,
        ),
    )
    messages = [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "task"},
        {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "function": {
                        "name": "bash",
                        "arguments": '{"command":"cat config.go"}',
                    }
                }
            ],
        },
        {"role": "tool", "content": "config source", "extra": {"returncode": 0}},
        *({"role": "user", "content": "old " + ("x" * 5_000)} for _ in range(8)),
        {"role": "assistant", "tool_calls": [{"id": "call-last"}], "content": ""},
        {"role": "tool", "tool_call_id": "call-last", "content": "latest output"},
    ]
    result = guard.query(messages)
    synopsis = next(
        message
        for message in delegate.messages
        if message.get("content", "").startswith("The context guard compacted")
    )
    assert "cat config.go" in synopsis["content"]
    assert result["extra"]["context_guard"]["synopsis_commands"] >= 1


def test_progress_guard_stops_after_configured_repeated_cycles() -> None:
    guard = ProgressGuard(no_progress_limit=2)
    actions = [{"command": "pwd"}]
    outputs = [{"output": "/testbed"}]
    guard.observe(actions=actions, outputs=outputs, workspace_digest="same")
    with pytest.raises(AgentStalledError):
        guard.observe(actions=actions, outputs=outputs, workspace_digest="same")


def test_context_budget_rejects_invalid_reservations() -> None:
    with pytest.raises(ValueError):
        ContextBudget(model_limit=16_000, system_overhead=8_000, tool_schema_tokens=8_000).validate()
