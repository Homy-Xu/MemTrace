# DeepSWE reproduction with mini-swe-agent 2.4.6

This document describes the public mini-swe-agent path for one DeepSWE task.
It is the release's A2-validated integration boundary. Private task prompts,
cluster paths, scheduler files, Docker sockets, raw trajectories, and provider
credentials are intentionally excluded.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mini-swe-agent]'
```

The package pins `mini-swe-agent==2.4.6`. Verify the installed version before a
paid run:

```bash
python - <<'PY'
import minisweagent
assert minisweagent.__version__ == '2.4.6'
print(minisweagent.__version__)
PY
```

## Provider profile

Use a protected environment variable for the MemTensor credential:

```bash
export MEMTENSOR_DOMESTIC_API_KEY='read-from-your-protected-secret-store'
```

The public profile is
`configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731.json`:

| Setting | Value |
| --- | --- |
| Harness | mini-swe-agent 2.4.6 |
| Model | `deepseek-v4-flash-0731` |
| Endpoint | `https://api-int.memtensor.cn/v1` |
| Wire API | Responses |
| Credential variable | `MEMTENSOR_DOMESTIC_API_KEY` |
| Model context | 200,000 tokens |
| Provider compaction | disabled; Context Refresh is runtime-owned |
| Numeric execution-turn limit | none in the agent profile |
| Memory runtime | Planning, Memory Trace formation, Trace Recall, Working Memory, MTG/RSG alignment |
| Acceptance budget | weak progress 4; no progress 2; semantic review 2; unclaimed boundaries 3 |
| Engagement | full |

The gateway adapter sanitizes the replayed request before every call. It
normalizes message IDs and input content, converts unsupported reasoning items
to assistant text, removes unsupported function-call fields, strips replayed
Responses envelopes, and appends a diagnostic output for an unmatched call.
The original trajectory remains the usage source.

## Run one clean task

```bash
memtrace run \
  --harness mini_swe_agent \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731.json \
  --model deepseek-v4-flash-0731 \
  --reasoning-effort high \
  --repository /path/to/clean/checkout \
  --task-file /path/to/deepswe-task.txt \
  --run-root /path/to/runs/deepswe/task-id
```

Or use the benchmark module:

```bash
python -m memtrace.benchmarks.mini \
  --model deepseek-v4-flash-0731 \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731.json \
  --repository /path/to/clean/checkout \
  --task-file /path/to/deepswe-task.txt \
  --run-root /path/to/runs/deepswe/task-id \
  --benchmark deepswe \
  --task-id task-id
```

Each task needs a new checkout and run root. The run root contains the event
ledger, Trace Store, Memory Index, checkpoints, trajectory, patch summary,
usage counters, and final receipt. A task-specific exception ends that task
and does not reuse another task's workspace.

## Receipt interpretation

The generation receipt is independent from the official evaluation receipt.
Read both before assigning a score:

- `generation_status` describes whether the Harness produced a terminal patch
  or a classified runtime failure.
- `evaluation_status` and `official_score` come only from the external DeepSWE
  evaluator.
- `input_tokens`, `output_tokens`, `total_tokens`, `api_calls`, `wall_time`
  and `provider_cost` describe usage. Unknown cost is `null` with
  `cost_available=false`.
- `failure_class` separates provider/auth, image/container, context-limit,
  agent-stall, evaluator, infrastructure, and model-quality failures.

A process exit of zero without an evaluator receipt is not a score. A reward of
zero with complete generation and evaluation is model-quality evidence; a
provider or container failure is not.

## Benchmark boundary

This entry point targets DeepSWE's repository-level tasks. SWE-Milestone uses
98 graded milestones grouped into seven continuous repository itineraries;
those milestones require ordered repository state, milestone-scoped official
receipts, predecessor handling, and an itinerary evaluator. Use the separate
SWE-Milestone launcher and its historical implementation provenance for that
benchmark. Do not convert a DeepSWE receipt into a SWE-Milestone result.
