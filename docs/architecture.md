# Architecture

MemTrace is a provenance-aware memory system for long-horizon coding agents.
It addresses **Task-State Drift**, where the agent loses track of completed and
unresolved work, and **State-Memory Misalignment**, where recovered evidence no
longer applies to the current Repository State.

The provider boundary is explicit. Codex App Server and mini-swe-agent 2.4.6
translate their native events into the same `HarnessEvent` contract, while the
runtime owns task state, Memory Trace persistence, repository alignment,
Selective Memory Restoration, checkpoints, and receipts.

```text
Codex App Server JSONL ─┐
                        ├─ HarnessEvent ─> state-consistent memory runtime ─> receipt
mini-swe-agent 2.4.6 ──┘
```

## Memory Trace formation

Completed observations, edits, test outcomes, decisions, and corrections are
consolidated as immutable **Memory Traces**. Each trace retains its evidence,
Repository State, referenced files/symbols/tests, provenance, and a **Memory
Anchor**. Ordered traces form **Memory Episodes** at task boundaries. Later
corrections create new traces and typed Trace Relations; earlier evidence is
preserved for provenance rather than silently rewritten.

The **Trace Store** is append-only. The **Memory Index** maps anchors and
repository entities to bounded trace sections. A **Trace Synopsis** and anchor
can remain in Working Memory after the full trace body is moved out of the
active context.

## MTG-RSG alignment

The **Memory Trace Graph (MTG)** records execution order, dependencies,
verification, correction, and supersession. The **Repository State Graph (RSG)**
represents the current files, symbols, tests, and structural relations. Shared
Repository Anchors connect the two views without treating structural hints as
proof that historical evidence remains valid.

The **Execution Frontier** is the latest task state and unresolved work. Trace
Localization starts from its anchors, task requirements, and repository
entities, then follows bounded MTG and RSG relations. Candidate selection is
request-specific; it does not replay the entire history.

## Validated restoration

When provider context is refreshed, MemTrace performs **Working-Memory
Reconstruction** in a bounded sequence:

```text
Trace Localization
  -> Memory Resolution
  -> Trace Validation
  -> State Alignment Check
  -> Selective Memory Restoration
  -> Cross-Context Recovery
```

Trace Validation checks integrity, source, and event lineage. Version
Validation checks the recorded repository revision and affected scope.
Provenance Validation checks event and repository bindings. The State Alignment
Check determines whether the evidence still applies to the current task and
Repository State, including whether a later Correction Trace supersedes it.
Only validated evidence that fits the remaining context budget is loaded into
Working Memory.

Completed evidence may undergo **Memory Offloading** as context pressure rises.
Its full body remains in the Trace Store, while a bounded Trace Synopsis and
Memory Anchor preserve recoverability. A **Context Refresh** replaces provider
visible context; it does not erase task state, repository bindings, or
provenance. Cross-Context Recovery resumes from the same Execution Frontier.

## Harness and receipts

Codex exposes native planning, incremental plan updates, thread recovery, and
provider capabilities when supported by its pinned App Server. The
mini-swe-agent DeepSWE adapter exposes model/tool events, trajectory usage,
Memory Episode checkpoints, and context replacement while reporting native
thread resume and native provider compaction as unavailable. The runtime records
those capabilities instead of fabricating parity.

The event ledger is append-only. Each event carries a run, branch, revision,
source event, provider method, normalized payload, and a raw-provider summary.
Credentials, private host paths, raw trajectories, and hidden prompts are
redacted or excluded from public receipts.

Stable source-level compatibility names remain in the implementation and
serialized schemas. The public vocabulary is defined in
[terminology.md](terminology.md).
