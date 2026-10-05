# DeepSWE reproduction

This document describes the public DeepSWE path for Codex CLI and
mini-swe-agent 2.4.6. Both harnesses use the same Memory Trace runtime. It
excludes private task prompts, cluster paths, scheduler files, Docker sockets,
trajectories, and credentials.

The runtime seals Memory Traces into the Trace Store, links them in the Memory
Trace Graph (MTG), projects them onto the Repository State Graph (RSG), and
restores validated evidence into Working Memory. Context Refresh, Trace
Recall, and Execution Milestone acceptance stay inside the runtime.

This document covers the DeepSWE task boundary. It does not claim a
SWE-Milestone reproduction: that benchmark evaluates ordered milestone IDs and
attempts within a repository stream, with a different official evaluator and
receipt contract.

## Shared profile

Both profiles fix the same budget:

| Setting | Value |
| --- | --- |
| Model | `deepseek-v4-flash-0731` |
| Credential variable | `MEMTENSOR_DOMESTIC_API_KEY` |
| Endpoint | `https://api-int.memtensor.cn/v1` |
| Wire API | Responses |
| Reasoning effort | `high` |
| Model context | 200,000 tokens |
| Native compaction | disabled |
| Modules | Planning, Trace Store, MTG, RSG, Trace Recall, Working Memory |
| Trace size | 2048 / 6144 / 8192 / 16384 tokens |
| Trace Recall | 8 traces, 8192 tokens |
| Execution budget | 200 turns, 14400 seconds |
| Acceptance | weak progress 4, no progress 2, semantic review rounds 2, unclaimed boundaries 3 |
| Engagement | `full` |

`no_progress` is the Execution Milestone acceptance budget. Do not put the key
in a shell history, task file, receipt, or repository.

```bash
export MEMTENSOR_DOMESTIC_API_KEY='(read from a protected secret store)'
export HOMY_ENABLE_CODE_GRAPH_SEARCH=1
```

For a non-Python repository, record the commit that exists before the agent
starts:

```bash
export HOMY_MULTILANG_BASE_COMMIT="$(git -C /path/to/clean/checkout rev-parse HEAD)"
```

## Codex CLI

Use Python 3.11 or newer:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,codex,multilang]'
```

Codex CLI talks to the model through the App Server. Install
`openai-codex-cli-bin==0.144.4` with the `codex` extra. Add `--multilang-plan`
only for a non-Python repository.

```bash
memtrace validate-config \
  --config configs/codex/memtensor-deepseek-v4-flash-0731-five-stage.json

memtrace run \
  --harness codex \
  --config configs/codex/memtensor-deepseek-v4-flash-0731-five-stage.json \
  --repository /path/to/clean/checkout \
  --task-file /path/to/task.txt \
  --run-root /path/to/runs/deepswe-codex \
  --model deepseek-v4-flash-0731 \
  --reasoning-effort high
```

[configs/codex/deepseek.yaml](../configs/codex/deepseek.yaml) points at
DeepSeek's public endpoint and is not this profile.

## mini-swe-agent 2.4.6

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mini-swe-agent,multilang]'
```

mini-swe-agent plans once, then uses the same Trace Store, MTG, RSG, and
Working Memory. Do not pass `--multilang-plan`. In a container, read model
metadata locally:

```bash
export LITELLM_LOCAL_MODEL_COST_MAP=True

memtrace validate-config \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json

memtrace run \
  --harness mini_swe_agent \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json \
  --repository /path/to/clean/checkout \
  --task-file /path/to/task.txt \
  --run-root /path/to/runs/deepswe-mini \
  --model deepseek-v4-flash-0731 \
  --reasoning-effort high
```

`python -m memtrace.benchmarks.mini` calls the same command. Official scoring
stays with the external DeepSWE evaluator. A durable runtime result is a
successful process handoff even when the task verdict is incomplete, so the
evaluator can inspect the same workspace.

After one task completes, submit the benchmark's full task manifest with
independent workers. Each worker gets its own checkout, container name,
scratch root, trajectory location, patch, and evaluation receipt. A task
failure terminates and cleans up that worker only.
