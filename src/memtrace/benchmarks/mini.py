"""DeepSWE entry for the mini-swe-agent 2.4.6 Memory Trace runtime."""
from __future__ import annotations

import argparse
import os
from pathlib import Path

from ..cli import main as cli_main

_DEFAULT_CONFIG = (
    Path(__file__).resolve().parents[3]
    / "configs"
    / "mini_swe_agent"
    / "memtensor-deepseek-v4-flash-0731.json"
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m memtrace.benchmarks.mini")
    parser.add_argument(
        "--model",
        default=os.environ.get("MEMTRACE_MINISWE_MODEL", "deepseek-v4-flash-0731"),
    )
    parser.add_argument("--repository", type=Path, required=True)
    parser.add_argument("--task-file", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=_DEFAULT_CONFIG,
        help="MemTrace runtime JSON configuration",
    )
    parser.add_argument(
        "--reasoning-effort",
        default=os.environ.get("MEMTRACE_MINISWE_REASONING_EFFORT", "high"),
    )
    parser.add_argument("--benchmark", default="deepswe")
    parser.add_argument("--task-id", default="local")
    args = parser.parse_args(argv)
    if not args.model.strip():
        parser.error("--model or MEMTRACE_MINISWE_MODEL is required")
    if not args.config.is_file():
        parser.error(f"runtime config not found: {args.config}")
    del args.benchmark, args.task_id
    return cli_main(
        [
            "run",
            "--harness",
            "mini_swe_agent",
            "--repository",
            str(args.repository),
            "--task-file",
            str(args.task_file),
            "--run-root",
            str(args.run_root),
            "--config",
            str(args.config),
            "--model",
            args.model,
            "--reasoning-effort",
            args.reasoning_effort,
        ]
    )


if __name__ == "__main__":
    raise SystemExit(main())
