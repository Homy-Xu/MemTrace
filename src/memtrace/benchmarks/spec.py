"""Shared task contracts for the three benchmark integrations."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass(frozen=True, slots=True)
class BenchmarkTask:
    benchmark: str
    task_id: str
    task: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.benchmark.strip() or not self.task_id.strip():
            raise ValueError("benchmark and task_id must be non-empty")
        if not self.task.strip():
            raise ValueError("task text must be non-empty")
        forbidden = {"gold_patch", "test_patch", "hidden_tests", "all_patch"}
        leaked = forbidden.intersection(self.metadata)
        if leaked:
            raise ValueError(f"benchmark task contains protected fields: {sorted(leaked)}")
