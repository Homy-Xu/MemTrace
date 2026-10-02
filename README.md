<div align="center">
  <h1>MemTrace</h1>
  <p><strong>State-consistent memory for long-horizon coding agents</strong></p>
  <p>
    <a href="https://github.com/Homy-Xu/MemTrace/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer">
    <img src="https://img.shields.io/badge/mini--swe--agent-2.4.6-4b8bbe?style=flat-square" alt="mini-swe-agent 2.4.6">
    <img src="https://img.shields.io/badge/benchmarks-SWE--Milestone%20%7C%20DeepSWE%20%7C%20SWE--EVO-6f42c1?style=flat-square" alt="Supported benchmarks">
  </p>
</div>

<div align="center">
  <img src="docs/overview.png" alt="MemTrace: memory traces, working memory, state alignment, and validated restoration" width="96%">
</div>

MemTrace helps coding agents continue long-horizon repository work after their
active context has been refreshed. It records execution as immutable **Memory
Traces**, binds each trace to the repository state in which it was produced,
and keeps only the evidence needed for the next action in **Working Memory**.
Historical evidence is restored only after it has been aligned with the current
task and repository state.

The runtime exposes the same provider-neutral lifecycle to two backends:

- **Codex App Server** — native JSONL events, planning, workspace revisions,
  memory-tool callbacks, and context recovery.
- **mini-swe-agent 2.4.6** — native trajectories with model/tool events,
  token usage, cost, wall-clock time, and exit status.

## 🧭 Contents

- [Architecture](#-architecture)
- [Installation](#-installation)
- [Run a Harness](#-run-a-harness)
- [Benchmark adapters](#-benchmark-adapters)
- [Receipts and reproducibility](#-receipts-and-reproducibility)
- [Terminology](#-terminology)
- [Project layout](#-project-layout)
- [Limitations](#-limitations)
- [License](#-license)

## 🧠 Architecture

MemTrace addresses **task-state drift** and **state-memory misalignment**. The
method has three connected responsibilities:

1. **Memory Trace formation** — consolidate completed observations, edits, test
   outcomes, decisions, and corrections with repository and provenance anchors.
2. **MTG–RSG alignment** — connect the Memory Trace Graph (MTG), which records
   execution relations, with the Repository State Graph (RSG), which describes
   the current files, symbols, and tests.
3. **Validated restoration** — localize Trace Candidates, perform Trace
   Validation and State Alignment Checks, and load only the evidence required
   to continue from the Execution Frontier.

The provider-neutral lifecycle is:

```text
start_session → plan → next_event / execute → checkpoint → resume → usage → close
```

See [docs/architecture.md](docs/architecture.md) for the method-level data
flow and [docs/overview.pdf](docs/overview.pdf) for the architecture
illustration.

## ⚡ Installation

MemTrace requires Python 3.11 or newer.

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install the backend you need:

```bash
python -m pip install -e '.[codex]'
# or
python -m pip install -e '.[mini-swe-agent]'
```

The Codex path pins `openai-codex-cli-bin==0.144.4`; the mini path pins
`mini-swe-agent==2.4.6`. Credentials are read from protected environment
variables. Literal keys are never accepted in configuration files or command
arguments. Start from [.env.example](.env.example) and keep the populated file
outside Git.

## 🚀 Run a Harness

Use a fresh run root for every task. A run root contains the event ledger,
durable trace state, checkpoints, trajectories, and receipts and is never
reused across attempts.

### Codex App Server

```bash
memtrace run \
  --harness codex \
  --model "$MEMTRACE_CODEX_MODEL" \
  --repository /path/to/repository \
  --task-file task.txt \
  --run-root /tmp/memtrace-codex-run
```

For DeepSeek's OpenAI-compatible Responses endpoint, use
[configs/codex/deepseek.yaml](configs/codex/deepseek.yaml):

```bash
export DEEPSEEK_API_KEY='set-this-locally'
export MEMTRACE_CODEX_MODEL='deepseek-flash'

memtrace run \
  --harness codex \
  --config configs/codex/deepseek.yaml \
  --model "$MEMTRACE_CODEX_MODEL" \
  --repository /path/to/repository \
  --task-file task.txt \
  --run-root /tmp/memtrace-codex-deepseek
```

For the long-horizon Codex contract used by the earlier DeepSWE,
SWE-Milestone, and SWE-EVO runs, use the pinned 200k-context profile and
private environment files:

```bash
memtrace run \
  --harness codex \
  --config configs/codex/memtensor-deepseek-v4-flash.json \
  --env-file /protected/provider.env \
  --env-file /protected/proxy.env \
  --provider-api-key-env MEMTENSOR_API_KEY \
  --model deepseek-v4-flash \
  --reasoning-effort high \
  --repository /path/to/repository \
  --task-file task.txt \
  --run-root /tmp/memtrace-codex-long-horizon
```

The profile details and its benchmark boundary are documented in
[docs/codex-benchmark-profile.md](docs/codex-benchmark-profile.md).

### mini-swe-agent 2.4.6

```bash
export DEEPSEEK_API_KEY='set-this-locally'
export MEMTRACE_MINISWE_MODEL='deepseek/deepseek-flash'

python -m memtrace.benchmarks.mini \
  --model "$MEMTRACE_MINISWE_MODEL" \
  --repository /path/to/repository \
  --task-file task.txt \
  --config configs/mini_swe_agent/default.yaml \
  --run-root /tmp/memtrace-mini-run \
  --benchmark smoke \
  --task-id local
```

DeepSeek currently documents `deepseek-flash` as the model identifier. The
legacy `deepseek-v4-flash` name remains accepted as an alias; use
`deepseek/deepseek-v4-flash` only when a deployment specifically requires that
legacy spelling.

The benchmark launchers accept only the shared Harness lifecycle. They do not
silently substitute a standalone runner or reuse another task's workspace,
container, provider socket, label, or receipt.

## 🧪 Benchmark adapters

The public adapters keep benchmark-specific concerns outside the memory
runtime:

- **SWE-Milestone** — repository streams, official evaluator handoff, and
  immutable attempt receipts.
- **DeepSWE** — multilingual task/image bridges, provider isolation, and
  infrastructure-failure classification.
- **SWE-EVO** — version-jump tasks, continuous Execution Milestones,
  host-managed verification, and fixture-contamination checks.

Full benchmark datasets, hidden tests, private prompts, cluster launch scripts,
and raw trajectories remain outside this repository.

## 📊 Receipts and reproducibility

Every task receipt records the benchmark, task identifier, Harness and version,
source digest, wheel digest, official score when available, F2P/P2P counts,
wall time, token usage, cost, generation status, evaluation status, and failure
class. Receipts are append-only and redact credentials and private filesystem
paths.

Public manifests distinguish a complete campaign, rerun, score-only regrade,
infrastructure failure, and model-quality failure. Historical results retain
their provenance and are never silently combined into a new benchmark claim.

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
```

The release smoke matrix and its current gate status are in
[docs/reproducibility.md](docs/reproducibility.md) and
[results/manifests/release-gate.json](results/manifests/release-gate.json).

## 📚 Terminology

The canonical vocabulary is documented in
[docs/terminology.md](docs/terminology.md). It is shared with the paper and
should be used in new documentation, experiment reports, and issue
discussions. Stable implementation identifiers and historical receipt fields
remain available for compatibility.

## 🗂️ Project layout

```text
src/memtrace/
├── core runtime contracts and persistence
├── planning/             task and Execution Milestone state
├── page_store/           compatibility implementation for the Trace Store
├── semantic_memory/      trace localization and repository alignment
├── recall/               Trace Recall and validated restoration
├── context_runtime/      Working Memory, compaction, checkpoints
├── rich_graph/           optional Repository State Graph indexing
├── harness/codex/        Codex App Server backend
├── harness/mini_swe_agent/  mini-swe-agent 2.4.6 backend
└── benchmarks/           shared runner and benchmark bridges
```

The source-level `page_store` name is retained for compatibility with existing
integrations; the public concept is **Trace Store**. The README structure
follows the concise research-code presentation used by
[RepoGraph](https://github.com/ozyyshr/RepoGraph) and
[Paper2Code](https://github.com/going-doer/Paper2Code). The paired
representation and reconstruction boundary is inspired by
[RPG-Encoder](https://github.com/microsoft/RPG-ZeroRepo/tree/main/zerorepo/rpg_encoder).

## 🔒 Limitations

- Official benchmark scores require the corresponding external evaluator and
  are never inferred from local tests.
- Real provider smoke runs require a model credential in the protected
  environment; the repository ships offline fixtures for contract testing.
- The Codex backend intentionally fails closed unless the verified
  `WorkspaceRevisionTracker` and memory-tool bridge are supplied.

## 📄 License

MemTrace is released under the [Apache License 2.0](LICENSE). Optional Harness
and benchmark dependencies retain their upstream licenses.
