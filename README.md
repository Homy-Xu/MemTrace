<div align="center">
  <h1>MemTrace</h1>
  <p><strong>State-consistent memory for long-horizon coding agents</strong></p>
  <p>
    <a href="https://github.com/Homy-Xu/MemTrace"><img src="https://img.shields.io/badge/repository-MemTrace-24292f?style=flat-square&logo=github" alt="MemTrace repository"></a>
    <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache--2.0-2ea44f?style=flat-square" alt="Apache 2.0 license"></a>
    <img src="https://img.shields.io/badge/python-3.11%2B-3776ab?style=flat-square&logo=python&logoColor=white" alt="Python 3.11 or newer">
    <img src="https://img.shields.io/badge/mini--swe--agent-2.4.6-4b8bbe?style=flat-square" alt="mini-swe-agent 2.4.6">
  </p>
  <p>
    <a href="#-overview">Overview</a> ·
    <a href="#-quick-start">Quick start</a> ·
    <a href="#-run-a-harness">Run a harness</a> ·
    <a href="#-benchmark-integrations">Benchmarks</a>
  </p>
</div>

<div align="center">
  <img src="docs/overview.png" alt="MemTrace memory traces, repository state, and validated restoration" width="94%">
</div>

## 📖 Overview

Long-running coding agents lose useful context as tasks grow. MemTrace gives an
agent a durable, state-aware memory layer that can survive context refreshes
without treating old observations as automatically valid.

MemTrace records execution as **Memory Traces**, anchors each trace to the
repository state that produced it, and restores only evidence that remains
relevant to the current task. The runtime is provider-neutral and can be used
with either of the following harnesses:

- **Codex App Server** for native JSONL events and workspace revisions.
- **mini-swe-agent 2.4.6** for model and tool trajectories with token, cost,
  wall-clock, and exit-status accounting.

## ✨ Highlights

- **State-aware memory** — connect task progress, repository changes, tests,
  decisions, and corrections to explicit state anchors.
- **Validated restoration** — perform Trace Localization, Trace Validation,
  and State Alignment Checks before loading historical evidence.
- **One runtime contract** — use the same session, checkpoint, resume, usage,
  and receipt interfaces across harnesses and benchmarks.
- **Reproducible runs** — keep source digests, evaluator status, usage, cost,
  and failure classification in append-only receipts.

## 🚀 Quick start

### Requirements

- Python 3.11 or newer
- Git
- A provider credential for live model runs

### Installation

```bash
git clone https://github.com/Homy-Xu/MemTrace.git
cd MemTrace

python -m venv .venv
source .venv/bin/activate
python -m pip install -e '.[dev]'
```

Install the harness you plan to use:

```bash
python -m pip install -e '.[codex]'
# or
python -m pip install -e '.[mini-swe-agent]'
```

Credentials are read from protected environment variables or private env
files. Start from [.env.example](.env.example), and never commit populated
credentials.

## 🧭 Run a harness

Each run should use a new repository workspace and run root. The run root keeps
the event ledger, Memory Traces, checkpoints, trajectories, and receipts for
that attempt.

### Codex App Server

```bash
memtrace run \
  --harness codex \
  --model "$MEMTRACE_CODEX_MODEL" \
  --repository /path/to/repository \
  --task-file task.txt \
  --run-root /tmp/memtrace-codex-run
```

For a provider with an OpenAI-compatible Responses endpoint, pass a private
configuration and env files:

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
  --run-root /tmp/memtrace-codex-run
```

See [docs/codex-benchmark-profile.md](docs/codex-benchmark-profile.md) for
the provider contract and benchmark-specific boundaries.

### mini-swe-agent 2.4.6

```bash
python -m memtrace.benchmarks.mini \
  --model "${MEMTRACE_MINISWE_MODEL:-deepseek-v4-flash}" \
  --repository /path/to/repository \
  --task-file task.txt \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash.yaml \
  --run-root /tmp/memtrace-mini-run \
  --benchmark smoke \
  --task-id local
```

The shared runner records the model trajectory and returns a receipt. It does
not reuse another task's workspace, provider socket, label, or run root.

For a protected MemTensor run, set `MEMTENSOR_API_KEY` and optionally
`MEMTRACE_MINISWE_BASE_URL` in the execution environment before starting the
command. The public configuration fixes the model context at 200,000 tokens,
leaves the numeric step budget unset, and uses an eight-hour wall-clock limit.
See [docs/deepswe-reproduction.md](docs/deepswe-reproduction.md) for the
canary and full-run procedure.

## 🧪 Benchmark integrations

MemTrace keeps benchmark-specific orchestration at the edge of the runtime:

| Benchmark | Adapter responsibilities |
| --- | --- |
| **SWE-Milestone** | Repository streams, official evaluator handoff, and immutable attempt receipts |
| **DeepSWE** | Task and image bridges, provider isolation, and infrastructure-failure classification |
| **SWE-EVO** | Version-jump tasks, continuous Execution Milestones, host-managed verification, and contamination checks |

The repository contains the public adapters and offline fixtures. Hidden tests,
private prompts, cluster launch scripts, and raw benchmark trajectories remain
outside the public source tree.

## 📊 Receipts and reproducibility

Every task receipt records the benchmark, task identifier, harness and version,
source digest, evaluator state, score when available, F2P/P2P counts, wall-clock
time, token usage, cost, and failure class. Credentials and private filesystem
paths are redacted.

Public manifests distinguish complete campaigns, reruns, score-only regrades,
infrastructure failures, and model-quality failures. Historical results keep
their provenance and are never silently combined into a new benchmark claim.

Run the local validation suite with:

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
```

See [docs/reproducibility.md](docs/reproducibility.md) and
[results/README.md](results/README.md) for the receipt format and release
checks.

## 🧠 Terminology

MemTrace uses the vocabulary defined in [docs/terminology.md](docs/terminology.md):

- **Memory Trace** and **Memory Episode** for durable execution evidence.
- **Trace Store** and **Memory Index** for persistence and lookup.
- **Working Memory** for the evidence admitted to the current context.
- **Memory Trace Graph (MTG)** and **Repository State Graph (RSG)** for linking
  execution history with repository state.
- **Memory Loading**, **Memory Consolidation**, and **Selective Memory
  Restoration** for the lifecycle of durable evidence.

Source-level compatibility names remain available where existing integrations
depend on them; they are documented as implementation aliases rather than new
paper concepts.

## 🗂️ Project layout

```text
src/memtrace/
├── core/                   runtime contracts and persistence
├── planning/               task and Execution Milestone state
├── page_store/             compatibility implementation for the Trace Store
├── semantic_memory/        trace localization and repository alignment
├── recall/                 Trace Recall and validated restoration
├── context_runtime/        Working Memory and checkpoints
├── rich_graph/             optional Repository State Graph indexing
├── harness/codex/          Codex App Server backend
├── harness/mini_swe_agent/ mini-swe-agent 2.4.6 backend
└── benchmarks/             shared runner and benchmark adapters
```

The public method description is in [docs/architecture.md](docs/architecture.md).

## 🤝 Contributing

Issues and pull requests are welcome. Before submitting a change:

1. Keep provider credentials, private prompts, raw trajectories, and cluster
   paths out of commits.
2. Add or update the relevant unit, contract, or offline smoke test.
3. Run `git diff --check` and the validation commands above.
4. Explain the source digest, evaluator status, and failure class for any new
   benchmark result.

## 📌 Limitations

- Official benchmark scores require the corresponding external evaluator.
- Live provider runs require credentials and network access in the protected
  environment.
- The Codex backend fails closed unless the verified workspace-revision and
  memory-tool bridges are available.

## 📄 License

MemTrace is released under the [Apache License 2.0](LICENSE). Optional harness
and benchmark dependencies retain their upstream licenses.
