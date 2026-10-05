# Architecture

MemTrace is a provenance-aware memory system for long-horizon coding agents.
It addresses two coupled failure modes: **task-state drift**, where the agent
loses track of completed and unresolved work, and **state-memory misalignment**,
where recovered evidence no longer applies to the current repository.

The runtime is provider-neutral. Codex App Server and mini-swe-agent 2.4.6
translate their native events into the same `HarnessEvent` contract, while the
runtime owns task state, trace persistence, repository alignment, restoration,
checkpointing, and receipts.

```text
Codex App Server JSONL ─┐
                        ├─ HarnessEvent ─> state-consistent memory runtime ─> receipt
mini-swe-agent 2.4.6 ──┘
```

## Memory Trace formation

Completed observations, edits, test outcomes, decisions, and corrections are
consolidated as immutable **Memory Traces**. Each trace retains its execution
evidence, repository state, referenced files/symbols/tests, provenance, and a
**Memory Anchor**. Ordered traces are grouped into **Memory Episodes** at task
boundaries. Later corrections create new traces and typed relations; the
original evidence remains available for provenance and is never silently
rewritten.

The **Trace Store** is append-only. The **Memory Index** maps anchors and
repository entities to bounded trace sections. A **Trace Synopsis** and anchor
can remain available in Working Memory after the full trace body has been
offloaded.

## MTG–RSG alignment

The **Memory Trace Graph (MTG)** records execution order, dependencies,
verification, correction, and supersession. The **Repository State Graph (RSG)**
represents the current files, symbols, tests, and structural relations. Shared
repository anchors connect the two views without treating structural hints as
proof of historical validity.

The Execution Frontier is the latest task state and unresolved work. Trace
localization starts from its anchors, task requirements, and repository
entities, then follows bounded MTG and RSG relations. Candidate selection is
addressed and request-specific; it does not scan or replay the entire history.

## Validated restoration

When provider context is refreshed, MemTrace performs **Working-Memory
Reconstruction** in a bounded sequence:

```text
Trace Localization
  → Memory Resolution
  → Trace Validation
  → State Alignment Check
  → Selective Memory Restoration
  → Cross-Context Recovery
```

Trace Validation checks integrity, source, and event lineage. Version
Validation checks the recorded repository revision and affected scope.
Provenance Validation checks the event and repository bindings. The State
Alignment Check determines whether the evidence still applies to the current
task and Repository State, including whether a later Correction Trace has
superseded it. Only validated evidence that fits the remaining context budget
is loaded into Working Memory.

Completed evidence may undergo **Memory Offloading** as context pressure rises.
Its full body remains in the Trace Store, while a bounded Trace Synopsis and
Memory Anchor preserve recoverability. A **Context Refresh** replaces the
provider-visible context; it does not erase task state, repository bindings, or
provenance. The next run resumes from the same Execution Frontier through
Cross-Context Recovery.

## Provider boundary and receipts

Codex exposes native planning, incremental plan updates, thread recovery, and
provider capabilities when supported by the pinned App Server. DeepSWE launches
mini-swe-agent 2.4.6 through the same Memory Trace runtime. That driver exposes
plan mode, execution turns, trajectory usage, and Context Refresh. It reports
native thread resume and native compaction as unavailable. The runtime records
those capabilities instead of fabricating parity.

The event ledger is append-only. Each event carries a run, branch, revision,
source event, provider method, normalized payload, and a raw-provider summary.
Payloads are summarized before entering public receipts; credentials, private
host paths, raw trajectories, and hidden prompts are redacted or excluded.

Stable source-level compatibility names remain in the implementation and
serialized schemas. The public vocabulary is defined in
[terminology.md](terminology.md).

## Benchmark scope

This release documents and validates the DeepSWE task boundary. Each task owns
an independent checkout, provider session, run root, trajectory, patch, and
official evaluator receipt.

SWE-Milestone uses a repository stream with ordered milestone IDs, attempt
numbers, repository-state transitions, host-managed verification, and
milestone-level scoring. Those boundaries are not interchangeable with a
single DeepSWE task, so this release does not claim a completed SWE-Milestone
mini-swe-agent adapter. SWE-EVO is outside the validated release scope as
well.
