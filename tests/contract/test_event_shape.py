from __future__ import annotations

from memtrace.harness.contracts import HarnessCapabilities, HarnessEvent, HarnessEventType


def test_provider_neutral_event_is_serializable() -> None:
    event = HarnessEvent(
        harness_event_id="r:1",
        event_type=HarnessEventType.TURN_STARTED,
        thread_id="thread",
        turn_id="turn",
        sequence=1,
        provider_time_ms=None,
        run_id="r",
        branch_id="main",
        revision_id="rev",
        source_event_id="source",
        provider_method="test",
        payload={"task_digest": "abc"},
        raw_provider_summary={"provider": "test"},
    )
    capabilities = HarnessCapabilities(
        harness_name="test",
        model_name="test-model",
        tokenizer_id=None,
        context_limit=None,
        supports_plan_mode=False,
        supports_incremental_plan_updates=False,
        supports_thread_resume=False,
        supports_native_compaction=False,
        supports_token_usage_events=True,
        supports_tool_lifecycle_events=True,
        supports_context_replacement=False,
    )
    assert event.event_type.value == "TURN_STARTED"
    assert capabilities.supports_token_usage_events
