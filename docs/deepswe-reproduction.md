# DeepSWE reproduction with mini-swe-agent 2.4.6

This document describes the mini-swe-agent adapter imported from the
author-provided `98e4e98` archive, reported as tested on A2. The source archive
and a new official benchmark reproduction have separate provenance; see
[validation status](reproducibility.md#validation-status). Private task
prompts, cluster paths, scheduler files, raw trajectories, and credentials are
excluded.

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
| mini-swe-agent numeric step limit | `0` (disabled) |
| Coordinator execution-turn budget | `200`; a runtime turn can contain multiple model/tool calls |
| Coordinator wall-clock budget | `14400` seconds (four hours) |
| Single shell-command timeout | `1800` seconds |
| Memory runtime | Planning, Memory Trace formation, Trace Recall, Working Memory, MTG/RSG alignment |
| Acceptance budget | weak progress 4; no progress 2; semantic review 2; unclaimed boundaries 3 |
| Engagement | full |

The gateway adapter sanitizes the replayed request before every call. It
normalizes message IDs and input content, converts unsupported reasoning items
to assistant text, removes unsupported function-call fields, strips replayed
Responses envelopes, and appends a diagnostic output for an unmatched call.
The original trajectory remains the usage source.

These are the checked-in archive defaults, not a claim that this profile
matches every paper experiment's budget. A launcher must enforce its own hard
deadline, including blocked provider requests and evaluator execution.

## Prepare the task environment

The adapter executes commands locally through mini-swe-agent's
`LocalEnvironment`. Run the CLI **inside** the task's prepared environment;
passing `--repository` does not load an image, create a container, install
dependencies, or restrict filesystem access.

The external launcher is responsible for the following:

- Lock the task ID, task text, baseline commit, image digest, and source digest.
- Provide the benchmark's dependencies and test tools before generation.
- Expose only the task workspace and permitted assets. Later upstream solution
  commits, other task workspaces, and hidden evaluator tests must be
  inaccessible.
- Use a separate writable run root and a hard task deadline. Do not expose a
  host Docker socket to the agent.
- Export the patch and run the official evaluator in a separate environment.

Use the task's declared working directory (for example `/app` or `/testbed`).
Installing MemTrace in a launcher environment does not install the task's own
dependencies in its command environment.

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
  --run-root /path/to/runs/deepswe/task-id
```

Each task needs a new checkout and run root. Runtime artifacts include
`v2-state.sqlite3`, `rich-graph.sqlite3`, `workspace-revisions/`,
`mini-swe-agent/trajectory.json`, archived context epochs, and `result.json`
when the coordinator finishes. The benchmark launcher must export the patch
and associate it with the exact task and evaluator result. Legacy
`--benchmark` and `--task-id` arguments are accepted by the module but ignored;
store those identities in the launcher's manifest.

## Receipt interpretation

The runtime result is independent from the official evaluation receipt. The
following fields are requirements for the launcher's combined manifest, not
fields guaranteed to be emitted together by this CLI:

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
zero can be classified as model-quality evidence only after environment and
evaluation validity are established. A provider, container, or setup failure
is not a valid quality score.

Inspect trajectory usage across all context epochs. Under
`cost_tracking=ignore_errors`, a raw `instance_cost=0.0` may mean that no
pricing information was available; it must not be presented as provider-
reported cost. An interrupted process may have no `result.json`, so the
launcher must also record its exit status and preserve partial artifacts.

## Benchmark boundary

This entry point targets DeepSWE's repository-level tasks. SWE-Milestone uses
98 graded milestones grouped into seven continuous repository itineraries;
those milestones require ordered repository state, milestone-scoped official
receipts, predecessor handling, and an itinerary evaluator. Use the separate
SWE-Milestone launcher and its historical implementation provenance for that
benchmark. Do not convert a DeepSWE receipt into a SWE-Milestone result.
