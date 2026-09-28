<div align="center">
  <h1>MemTrace</h1>
  <p><strong>Addressable working context for long-running coding agents</strong></p>
  <p>
    <a href="https://github.com/Homy-Xu/MemTrace/blob/main/LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer">
    <img src="https://img.shields.io/badge/mini--swe--agent-2.4.6-4b8bbe?style=flat-square" alt="mini-swe-agent 2.4.6">
    <img src="https://img.shields.io/badge/benchmarks-SWE--Milestone%20%7C%20DeepSWE%20%7C%20SWE--EVO-6f42c1?style=flat-square" alt="Supported benchmarks">
  </p>
</div>

<div align="center">
  <img src="docs/overview.png" alt="MemTrace architecture: memory traces, addressable context, and validated recall" width="96%">
</div>

MemTrace gives coding agents a durable, addressable working context. It turns
execution history into memory pages, keeps a small active working set, and
recalls only the evidence needed for the current repository state.

The same five-stage runtime supports two execution backends:

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
- [Project layout](#-project-layout)
- [Limitations](#-limitations)
- [License](#-license)

## 🧠 Architecture

The runtime keeps the control plane and provider telemetry separate while
sharing one event and receipt contract:

1. **Planning and Milestone state** — freeze task scope and execution stages.
2. **WAL and page storage** — persist workspace facts and page updates.
3. **Semantic memory and Rich Graph** — index reusable conclusions and code
   relationships.
4. **Evidence-keyed recall** — admit only task-relevant, validated evidence.
5. **Context runtime** — manage working sets, compaction, checkpoints, and
   recovery.

The provider-neutral lifecycle is:

```text
start_session → plan → next_event / execute → checkpoint → resume → usage → close
```

See [docs/architecture.md](docs/architecture.md) for the data flow and
[docs/overview.pdf](docs/overview.pdf) for the full architecture illustration.

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

Use a fresh run root for every task. A run root contains WAL, SQLite state,
checkpoints, trajectories, and receipts and is never reused across attempts.

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

The public adapters keep benchmark-specific concerns outside the core runtime:

- **SWE-Milestone** — repository streams, official evaluator handoff, and
  immutable attempt receipts.
- **DeepSWE** — multilingual task/image bridges, provider isolation, and
  infrastructure-failure classification.
- **SWE-EVO** — version-jump tasks, continuous Milestones, host-managed
  verification, and fixture-contamination checks.

Full benchmark datasets, hidden tests, private prompts, cluster launch scripts,
and raw trajectories remain outside this repository.

## 📊 Receipts and reproducibility

Every task receipt records the benchmark, task identifier, Harness and version,
source digest, wheel digest, official score when available, F2P/P2P counts,
wall time, token usage, cost, generation status, evaluation status, and failure
class. Receipts are append-only and redact credentials and private filesystem
paths.

Public manifests distinguish a complete campaign, rerun, score-only regrade,
infrastructure failure, and model-quality failure. Historical results are kept
with provenance and are never silently combined into a new benchmark claim.

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
```

The release smoke matrix and its current gate status are in
[docs/reproducibility.md](docs/reproducibility.md) and
[results/manifests/release-gate.json](results/manifests/release-gate.json).

## 🗂️ Project layout

```text
src/memtrace/
├── core runtime contracts and persistence
├── planning/             Plan and Milestone state
├── page_store/           durable memory pages
├── semantic_memory/      reusable implementation conclusions
├── recall/               evidence retrieval and admission
├── context_runtime/      working sets, compaction, checkpoints
├── rich_graph/           optional structural indexing
├── harness/codex/        Codex App Server backend
├── harness/mini_swe_agent/  mini-swe-agent 2.4.6 backend
└── benchmarks/           shared runner and benchmark bridges
```

The README structure follows the concise research-code presentation used by
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
