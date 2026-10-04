# DeepSWE reproduction

This document describes the public path for a DeepSWE task on mini-swe-agent
2.4.6 with the five-stage runtime. It excludes private task prompts, cluster
paths, scheduler files, Docker sockets, trajectories, and credentials.

The five-stage chain is planning, trace persistence, semantic memory, trace
recall, and context runtime, with the repository graph enabled. Context
replacement, recall, and milestone acceptance stay inside the runtime.

## Environment

Use Python 3.11 or newer and install the pinned harness:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mini-swe-agent]'
```

The live provider configuration is supplied through a protected environment:

```bash
export MEMTENSOR_DOMESTIC_API_KEY='(read from a protected secret store)'
```

Do not put the key in a shell history, task file, receipt, or repository.

## Runtime contract

`configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json` is
the public DeepSWE profile:

| Setting | Value |
| --- | --- |
| Model | `deepseek-v4-flash-0731` |
| Credential variable | `MEMTENSOR_DOMESTIC_API_KEY` |
| Endpoint | `https://api-int.memtensor.cn/v1` |
| Wire API | Responses |
| Reasoning effort | `high` |
| Model context | 200,000 tokens |
| Native compaction | disabled |
| Stages | planning, trace store, semantic memory, recall, context runtime, repository graph |
| Trace size | 2048 / 6144 / 8192 / 16384 tokens |
| Recall | 8 traces, 8192 tokens |
| Execution budget | 200 turns, 14400 seconds |
| Acceptance | weak progress 4, no progress 2, semantic review rounds 2, unclaimed boundaries 3 |
| Engagement | `full` |
| Harness | mini-swe-agent 2.4.6 |

`no_progress` is the milestone acceptance budget. Non-Python tasks may set
`HOMY_MULTILANG_BASE_COMMIT` to the repository `HEAD` before launch. That
enables the same sectioned plan reading and code-graph tool used by the
multilingual campaign. Do not pass `--multilang-plan`; that flag belongs to
the Codex harness.

## Canary, then full run

Run one task from a clean repository checkout and an independent run root:

```bash
memtrace run \
  --harness mini_swe_agent \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json \
  --repository /path/to/clean/checkout \
  --task-file /path/to/task.txt \
  --run-root /path/to/runs/deepswe-canary \
  --model deepseek-v4-flash-0731 \
  --reasoning-effort high
```

`python -m memtrace.benchmarks.mini` calls the same command. Official scoring
stays with the external DeepSWE evaluator. A durable runtime result is a
successful process handoff even when the task verdict is incomplete, so the
evaluator can inspect the same workspace.

After the canary gate passes, submit the benchmark's full task manifest with
independent workers. Each worker gets its own checkout, container name,
scratch root, trajectory location, patch, and evaluation receipt. A task
failure terminates and cleans up that worker only.
