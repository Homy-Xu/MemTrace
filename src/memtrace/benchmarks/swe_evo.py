"""SWE-EVO version-jump task bridge."""
from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from ..harness.base import HarnessBackend
from .runner import BenchmarkRun, BenchmarkRunner
from .spec import BenchmarkTask


def run_task(
    backend: HarnessBackend,
    task: BenchmarkTask,
    *,
    receipt_dir: Path,
    official_score: float | None = None,
    f2p: Mapping[str, int] | None = None,
    p2p: Mapping[str, int] | None = None,
    wheel_sha256: str | None = None,
    provenance: Mapping[str, Any] | None = None,
    execution_kwargs: Mapping[str, Any] | None = None,
) -> BenchmarkRun:
    task.validate()
    if task.benchmark != "swe-evo":
        raise ValueError("SWE-EVO adapter received a different benchmark")
    return BenchmarkRunner(backend, receipt_dir=receipt_dir).run(
        benchmark=task.benchmark,
        task_id=task.task_id,
        task=task.task,
        wheel_sha256=wheel_sha256,
        official_score=official_score,
        f2p=f2p,
        p2p=p2p,
        provenance={"official_evaluator": "external", **dict(provenance or {})},
        execution_kwargs=execution_kwargs,
    )
