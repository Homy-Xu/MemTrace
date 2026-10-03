# DeepSWE reproduction

This document describes the public, provider-neutral path for running a
DeepSWE task with the mini-swe-agent harness. It deliberately excludes private
task prompts, cluster paths, scheduler files, Docker sockets, trajectories, and
credentials.

## Environment

Use Python 3.11 or newer and install the pinned harness:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mini-swe-agent]'
```

The live provider configuration is supplied through a protected environment:

```bash
export MEMTENSOR_API_KEY='(read from a protected secret store)'
export MEMTRACE_MINISWE_BASE_URL='https://api-int.memtensor.cn/v1'
export MEMTRACE_MINISWE_MODEL='deepseek-v4-flash'
```

Do not put the key in a shell history, task file, receipt, or repository. The
launcher should check the endpoint, model, credential availability, image
reference, run root, and scratch root before making a model request.

## Runtime contract

`configs/mini_swe_agent/memtensor-deepseek-v4-flash.yaml` is the public
reference configuration. It fixes the model context window at 200,000 tokens,
reserves space for system instructions, tool schemas, output, and a safety
margin, and records whether an input was trimmed. It leaves the numeric step
budget unset and limits each task to eight hours of wall-clock time.

The agent is required to use the bash tool for repository work and to run the
benchmark completion command supplied by the task runner. A provider-neutral
progress guard stops two consecutive identical action/output/workspace cycles;
the receipt classifies this as `AGENT_STALLED` rather than consuming the rest
of the wall-clock budget.

## Canary, then full run

Run one task from a clean repository checkout and an independent run root:

```bash
python -m memtrace.benchmarks.mini \
  --model "$MEMTRACE_MINISWE_MODEL" \
  --repository /path/to/clean/checkout \
  --task-file /path/to/task.txt \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash.yaml \
  --run-root /path/to/runs/deepswe-canary \
  --benchmark deepswe \
  --task-id actionlint-action-pinning-lint
```

The canary is a runtime gate. It must have a generation receipt, an official
evaluation receipt, and a final receipt with cleanup status. A score of zero
is retained as `MODEL_QUALITY_FAILURE` when generation and evaluation are
complete. Provider/auth, image/container, context, serialization, disk, and
agent-stall problems are recorded as infrastructure or agent failures and
must be fixed before scaling out.

After the canary gate passes, submit the benchmark's full task manifest with
12 independent workers. Each worker gets its own checkout, container name,
scratch root, trajectory location, patch, evaluation receipt, and final
receipt. A task failure terminates and cleans up that worker only.

## Receipt fields

Receipts distinguish generation from evaluation and score publication. Usage
contains API calls, input/output/total tokens, provider-reported cost when it
is available, a `cost_available` flag, and wall-clock time. Missing provider
pricing is represented by `cost: null`; it is never reported as a genuine
zero-cost run. Mini receipts also include the original and retained message
counts, estimated input tokens, configured input limit, and whether trimming
occurred. Public manifests contain source and wheel digests, task and
harness identity, official score when available, F2P/P2P counts, and the
failure class.

## Configuration parity

The earlier private server configuration and the public runtime have the same
observable guarantees:

| Behaviour | Public setting |
| --- | --- |
| Model context | `context_budget.model_limit: 200000` |
| Context admission | system/tool/output/safety reservations with input trimming |
| Memory runtime | planning, trace persistence, trace recall, context runtime, and graph runtime remain enabled by the benchmark launcher |
| Numeric steps | no numeric step limit (`step_limit: 0`) |
| Wall clock | `wall_time_limit_seconds: 28800` |
| Progress safety | `progress.no_progress_limit: 2` |
| Accounting | generation, evaluation, score, usage, cost, and cleanup receipts |

Private launchers may add scheduler and image details, but they must pass
strings for task IDs, commits, and image references into the public runner and
must not expose those private details in Git history.
