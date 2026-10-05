<div align="center">
  <h1>MemTrace</h1>
  <p><strong>State-consistent memory for long-horizon coding agents</strong></p>
  <p>
    <a href="https://github.com/Homy-Xu/MemTrace/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer">
    <img src="https://img.shields.io/badge/mini--swe--agent-2.4.6-4b8bbe?style=flat-square" alt="mini-swe-agent 2.4.6">
    <img src="https://img.shields.io/badge/adapter-DeepSWE-6f42c1?style=flat-square" alt="DeepSWE adapter">
  </p>
</div>

<div align="center">
  <img src="docs/overview.png" alt="MemTrace: memory traces, working memory, state alignment, and validated restoration" width="96%">
</div>

MemTrace is a provenance-aware memory runtime for coding agents working on
long-horizon repository tasks. It records completed execution as immutable
**Memory Traces**, binds each trace to the **Repository State** in which it was
created, and keeps the evidence needed for the next action in **Working Memory**.
A trace is restored only after **Trace Validation** and a **State Alignment
Check** against the current task and repository.

This repository contains two harness boundaries:

- **Codex App Server**, which consumes native JSONL events and provider-managed
  context operations.
- **mini-swe-agent 2.4.6**, with the DeepSWE runtime adapter from the
  author-provided `98e4e98` archive, reported as tested on A2. It records
  model/tool events and token usage through the shared memory runtime.

The archive's reported A2 run and a new reproduction from this repository are
separate evidence. Offline checks pass; a fresh isolated end-to-end run with an
official evaluator receipt has not yet been established for this publication.
See [validation status](docs/reproducibility.md#validation-status).

The mini-swe-agent adapter is intentionally published under
`src/memtrace/harness/mini_swe_agent/`. The older
`memtrace.harness.mini_five_stage` import is retained as a compatibility alias
for existing launchers; it is not a separate runtime or benchmark claim.

## Contents

- [Architecture](#architecture)
- [Installation](#installation)
- [Run the mini-swe-agent DeepSWE adapter](#run-the-mini-swe-agent-deepswe-adapter)
- [Integrate another task runner](#integrate-another-task-runner)
- [Benchmark scope](#benchmark-scope)
- [Receipts and reproducibility](#receipts-and-reproducibility)
- [Project layout](#project-layout)
- [Limitations](#limitations)
- [License](#license)

## Architecture

MemTrace addresses **Task-State Drift** and **State-Memory Misalignment**. The
runtime has three connected responsibilities:

1. **Memory Trace formation** seals observations, edits, test outcomes,
   decisions, and **Correction Traces** with Repository Anchors, Version Anchors,
   and provenance.
2. **MTG-RSG alignment** connects the **Memory Trace Graph (MTG)**, which records
   execution order and dependencies, with the **Repository State Graph (RSG)**,
   which describes files, symbols, and tests in the current repository.
3. **Validated restoration** performs **Trace Localization**, **Trace
   Validation**, **State Alignment Check**, and **Selective Memory Restoration**
   before **Cross-Context Recovery** resumes from the **Execution Frontier**.

The public Harness contract is provider-neutral:

```text
start_session -> plan -> next_event / execute -> checkpoint -> resume -> usage -> close
```

The lifecycle is shared by the Codex and mini-swe-agent boundaries, while each
backend keeps its native event and trajectory format. See
[docs/architecture.md](docs/architecture.md) and the paper-aligned vocabulary
in [docs/terminology.md](docs/terminology.md).

## Installation

MemTrace requires Python 3.11 or newer.

```bash
git clone https://github.com/Homy-Xu/MemTrace.git
cd MemTrace
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev,mini-swe-agent]'
```

The mini-swe-agent extra pins `mini-swe-agent==2.4.6`. The Codex extra pins the
Codex App Server binary used by the companion backend:

```bash
python -m pip install -e '.[codex]'
```

Credentials are read only from protected environment variables. Start from
[.env.example](.env.example), keep the populated file outside Git, and never
place a key in a task file, config JSON, command argument, receipt, or log.

## Run the mini-swe-agent DeepSWE adapter

This entry point runs generation for one DeepSWE repository task. Run it inside
the task's prepared, isolated environment, with its dependencies installed and
an independent run root. The adapter uses a **local command environment**:
`--repository` sets the working directory; it does not start a Docker container
or install task dependencies. The benchmark launcher must create that
environment and run the official evaluator separately.

The profile uses the MemTensor Responses endpoint and a 200,000-token context.
mini-swe-agent's numeric step limit is disabled (`step_limit=0`), while the
checked-in coordinator profile retains `max_execution_turns=200` and
`wall_clock_seconds=14400` (four hours). Those runtime turns differ from tool
calls. Single shell commands have a 1,800-second timeout; the external
launcher must also enforce a hard deadline covering provider calls and
evaluation.

```bash
export MEMTENSOR_DOMESTIC_API_KEY='read-from-your-protected-secret-store'

memtrace run \
  --harness mini_swe_agent \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731.json \
  --model deepseek-v4-flash-0731 \
  --reasoning-effort high \
  --repository /path/to/clean/checkout \
  --task-file /path/to/deepswe-task.txt \
  --run-root /path/to/runs/deepswe/task-id
```

The equivalent benchmark entry point is:

```bash
python -m memtrace.benchmarks.mini \
  --model deepseek-v4-flash-0731 \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731.json \
  --repository /path/to/clean/checkout \
  --task-file /path/to/deepswe-task.txt \
  --run-root /path/to/runs/deepswe/task-id
```

The task file is supplied by the benchmark launcher and must remain unchanged.
The run root receives runtime state, workspace revisions, the trajectory, and
`result.json` when the coordinator finishes. A launcher must bind these to the
benchmark task ID, export the generated patch, and retain the official
evaluation receipt. The module accepts legacy `--benchmark` and `--task-id`
arguments but does not propagate them into runtime metadata. Local test output
is not a substitute for an official result.

The adapter uses the public mini-swe-agent 2.4.6 APIs and sanitizes replayed
Responses items for the current MemTensor gateway. It removes unsupported
reasoning summaries and envelope fields, normalizes message content, preserves
function-call/output pairing. A provider exception or interrupted process is
not a model-quality score; the launcher must retain its failure status even
when no terminal runtime result is written.

## Integrate another task runner

A custom launcher should keep benchmark orchestration outside the runtime and
supply the same Harness lifecycle:

1. prepare the task image and dependencies, a benchmark-approved baseline
   checkout, and a new run root;
2. call `start_session` and `plan` with the exact task text;
3. forward model/tool boundaries through `next_event` or `execute`;
4. call `checkpoint` before a context refresh and `resume` only with that
   checkpoint;
5. close the run and persist usage, generation, evaluation, and failure
   receipts.

The DeepSWE CLI connects
`memtrace.harness.mini_swe_agent.MiniSweAgentHarnessAdapter` to `RunCoordinator`.
`MiniSweAgentBackend` is a separate, lower-level trajectory bridge; using it
alone does not run the complete memory runtime. Do not reuse a mutable
workspace, provider session, container identity, or receipt directory across
attempts. Keep evaluator tests and later solution commits outside the
generation environment. The adapter's local environment is not a sandbox.

## Benchmark scope

The published mini-swe-agent code is for **DeepSWE**. It is not the
SWE-Milestone adapter and should not be used to report SWE-Milestone scores.
The distinction is structural:

| Benchmark | Unit being evaluated | Official accounting |
| --- | --- | --- |
| DeepSWE | One repository-level engineering task | 113 tasks across 91 repositories; task-level `pass@1` |
| SWE-Milestone | A graded milestone inside a continuous repository itinerary | 98 milestones in 7 itineraries; `Score`, `Precision`, `Recall`, and `Resolved Rate` |

SWE-Milestone milestones share repository state and history across an
itinerary. A runner must preserve the ordered repository stream, milestone
predecessors, official IDs and tags, milestone-scoped acceptance, and the
external evaluator receipt. Solving a DeepSWE issue in one checkout does not
establish that the corresponding SWE-Milestone milestone sequence is valid.
MemTrace may also plan internal **Execution Milestones** for a single DeepSWE
task. Those are runtime work units, not the benchmark's official milestone IDs
and not independently scored examples.
The seven official milestone counts are go-zero 23, Navidrome 9,
element-web 18, Nushell 13, ripgrep 11, Dubbo 12, and scikit-learn 12.

The repository still contains compatibility contracts for Codex and
SWE-Milestone-related integrations, but this mini-swe-agent release does not
claim a new SWE-Milestone run. The paper records a separate historical runtime
identity (`MemTrace 2.2.97`, commit `0dc603d6d78c6cec479d061d5ac460dccffbc5bd`,
branch `fix/v2.2.90-five-stage-closure`, build
`build_61cc38bd4ec1da5ede5cda6097508c58`, and wheel SHA256
`322148bfb10b0993e41b894b00ab838726384f7dc7618b0b2feb0e98ebdb0e98`). Those
fields are provenance for the historical SWE-Milestone result; the matching
source, wheel, task receipts, and evaluator archive are not bundled here and
must not be inferred from this release.

## Receipts and reproducibility

Runtime outputs and raw trajectories are private artifacts. Before publishing
a result, the launcher must create a redacted manifest binding the task,
Harness, source digest, generation status, official evaluation, F2P/P2P,
wall-clock time, token usage, and failure class. Use provider-reported cost
when available. mini-swe-agent's raw `instance_cost=0.0` with
`cost_tracking=ignore_errors` is not evidence of a free run; report unavailable
cost as `null` with `cost_available=false` in the result manifest.

Use the offline checks before a provider run:

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
python -m memtrace --help
python -m memtrace.benchmarks.mini --help
```

The release manifest in [results/manifests/release-gate.json](results/manifests/release-gate.json)
contains only redacted metadata. Historical campaigns, score-only regrades,
model-quality failures, and infrastructure failures remain separate records.
See [docs/deepswe-reproduction.md](docs/deepswe-reproduction.md) for the
provider setup and [docs/reproducibility.md](docs/reproducibility.md) for the
release boundary.

## Project layout

```text
src/memtrace/
├── core contracts, persistence, and receipts
├── planning/                 Task State and Execution Milestone contracts
├── page_store/               compatibility implementation for the Trace Store
├── semantic_memory/          Trace Localization and state alignment
├── recall/                   Trace Recall and validated restoration
├── context_runtime/          Working Memory and Context Refresh
├── rich_graph/               Repository State Graph indexing
├── harness/codex/            Codex App Server backend
├── harness/mini_swe_agent/    mini-swe-agent 2.4.6 backends and DeepSWE adapter
└── benchmarks/               shared runner and benchmark bridges
```

The source-level `page_store` package, configuration keys, and historical
receipt fields remain for compatibility. The paper and public prose use Trace
Store, Memory Index, Memory Anchor, Memory Episode, and the other terms in
[docs/terminology.md](docs/terminology.md).

## Limitations

- Official benchmark scores require the corresponding external evaluator and
  benchmark assets; they are never inferred from local tests.
- The DeepSWE profile requires an authorized MemTensor credential and a task
  image/checkout supplied by the benchmark launcher.
- The public repository does not include private task prompts, hidden tests,
  cluster paths, scheduler files, Docker state, raw trajectories, or provider
  secrets.
- The historical SWE-Milestone result identity is recorded for provenance and
  is not a claim that this mini-swe-agent profile can reproduce it.

## License

MemTrace is released under the [Apache License 2.0](LICENSE). Optional Harness
and benchmark dependencies retain their upstream licenses.
