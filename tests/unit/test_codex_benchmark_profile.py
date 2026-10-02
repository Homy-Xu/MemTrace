from __future__ import annotations

from pathlib import Path

from memtrace.config import load_config
from memtrace.harness.provider import build_runtime_launch


def test_historical_codex_benchmark_profile_is_explicit_and_unbounded(
    monkeypatch, tmp_path: Path
) -> None:
    root = Path(__file__).parents[2]
    config = load_config(root / "configs/codex/memtensor-deepseek-v4-flash.json")
    assert config.provider.id == "memtensor_deepseek"
    assert config.provider.model == "deepseek-v4-flash"
    assert config.provider.model_context_window == 200_000
    assert config.provider.native_compaction_enabled is False
    assert config.codex_timeout_seconds == 10_800
    assert config.codex_reasoning_effort == "high"
    assert config.codex_sandbox_mode == "danger-full-access"
    assert config.run_budget.max_execution_turns is None
    assert config.run_budget.wall_clock_seconds is None
    monkeypatch.setenv("MEMTENSOR_API_KEY", "fixture-credential")
    sqlite_home = tmp_path / "sqlite"
    sqlite_home.mkdir()
    launch = build_runtime_launch(
        config.provider,
        executable="/usr/bin/true",
        sqlite_home=sqlite_home,
    )
    assert "200000" in " ".join(launch.app_server_args)
    assert "2147483647" in " ".join(launch.app_server_args)
