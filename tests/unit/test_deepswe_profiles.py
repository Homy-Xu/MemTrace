from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def _profile(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def test_codex_and_miniswe_deepswe_profiles_are_identical() -> None:
    codex = _profile(
        "configs/codex/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )
    mini = _profile(
        "configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json"
    )

    assert codex == mini
    assert codex["provider"]["api_key_env"] == "MEMTENSOR_DOMESTIC_API_KEY"
    assert codex["provider"]["model"] == "deepseek-v4-flash-0731"
    assert codex["provider"]["model_context_window"] == 200_000
    assert codex["provider"]["native_compaction_enabled"] is False
    assert codex["run_budget"] == {
        "max_execution_turns": 200,
        "wall_clock_seconds": 14_400,
    }
