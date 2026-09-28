"""Small standalone entry point for a mini-swe-agent smoke or task run."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import yaml

from ..harness.mini_swe_agent import MiniSweAgentBackend
from .runner import BenchmarkRunner


def main() -> int:
    parser = argparse.ArgumentParser(prog="python -m memtrace.benchmarks.mini")
    parser.add_argument("--model", default=os.environ.get("MEMTRACE_MINISWE_MODEL", ""))
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, help="mini-swe-agent YAML configuration")
    parser.add_argument("--benchmark", default="smoke")
    parser.add_argument("--task-id", default="local")
    args = parser.parse_args()
    if not args.model.strip():
        parser.error("--model or MEMTRACE_MINISWE_MODEL is required")
    task = args.task_file.read_text(encoding="utf-8")
    configuration = {}
    if args.config is not None:
        loaded = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            parser.error("--config must contain a YAML mapping")
        configuration = loaded
    backend = MiniSweAgentBackend(
        repository_path=args.repository,
        model=args.model,
        run_root=args.run_root,
        agent_config=configuration,
    )
    result = BenchmarkRunner(backend, receipt_dir=args.run_root / "receipts").run(
        benchmark=args.benchmark,
        task_id=args.task_id,
        task=task,
    )
    print(result.receipt)
    return 0 if result.receipt["status"] == "COMPLETED" else 2


if __name__ == "__main__":
    raise SystemExit(main())
