# MemTrace terminology

MemTrace uses the vocabulary in the paper, *MemTrace: State-Consistent Memory
for Long-Horizon Coding Agents*. The terms describe the method at the level of
the paper and the public documentation; they do not rename stable Python
interfaces or stored receipt fields.

| Concept | MemTrace term | Meaning |
| --- | --- | --- |
| A bounded, immutable unit of execution evidence | **Memory Trace** | A completed observation, edit, test result, decision, or correction bound to the repository state in which it was produced. |
| An ordered group of traces around a task boundary | **Memory Episode** | A logical execution segment that preserves order, goal, summary, metadata, and relations. |
| Durable trace persistence | **Trace Store** | The append-only store for trace bodies, provenance, relations, and summaries. |
| Stable lookup directory | **Memory Index** | The address directory that maps anchors and repository entities to trace sections. |
| Stable reference to a trace or section | **Memory Anchor** | An opaque, runtime-owned address that can be resolved without broad history search. |
| Bringing validated evidence into active context | **Memory Loading** | Selective restoration of trace evidence required by the next action. |
| Moving completed evidence out of active context | **Memory Offloading** | Context reduction that preserves the complete trace in the Trace Store. |
| Closing and indexing a completed episode | **Memory Consolidation** | The durable boundary at which trace bodies, summaries, anchors, and relations become recoverable. |
| Execution-history relation graph | **Memory Trace Graph (MTG)** | Relations for order, dependency, verification, correction, and supersession. |
| Current repository relation graph | **Repository State Graph (RSG)** | Files, symbols, tests, and structural relations in the current repository state. |
| Active bounded context | **Working Memory** | The task, checkpoint, repository scope, anchors, and validated evidence currently visible to the model. |
| Compact representation of offloaded evidence | **Trace Synopsis** | A bounded summary that preserves provenance and points back to a Memory Anchor. |
| Evidence selection | **Trace Recall** | The operation that locates, validates, and admits relevant traces. |
| Candidate evidence | **Trace Candidate** | A trace proposed for validation before it enters Working Memory. |
| Repository or symbol binding | **Repository Anchor** | A file, symbol, or test location used to connect traces with the repository. |
| Version binding | **Version Anchor** | The commit or state version in which evidence was produced. |
| Runtime state binding | **State Anchor** | A task, execution, or repository-state identity used for alignment. |
| Graph relation | **Trace Relation** | A typed relation between traces or between traces and repository entities. |
| Dependency relation | **Trace Dependency** | A relation showing that one trace relies on another trace or state. |
| Execution-order relation | **Execution Relation** | A relation showing the order in which evidence was produced. |
| Current unfinished work | **Execution Frontier** | The latest task state and unresolved work from which execution continues. |
| Repository snapshot | **Repository State** | The files, symbols, tests, and revisions visible at a point in execution. |
| Version snapshot | **State Snapshot** | A durable description of the state used to produce or validate evidence. |
| Candidate ranking and lookup | **Trace Localization** | Addressed selection of traces through anchors, MTG relations, RSG entities, and task state. |
| Address resolution | **Memory Resolution** | Translation of a Memory Anchor into a bounded trace section. |
| Evidence check | **Trace Validation** | Verification that a candidate is intact, sourced, and internally consistent. |
| Applicability check | **State Alignment Check** | Verification that evidence still applies to the current task and repository state. |
| Revision check | **Version Validation** | Verification that the recorded repository version matches the current validation scope. |
| Source check | **Provenance Validation** | Verification of origin, event lineage, and repository bindings. |
| Selective history restoration | **Selective Memory Restoration** | Loading only the validated evidence required for the next action. |
| New active context | **Context Refresh** | Replacing the provider context while preserving durable task state and trace provenance. |
| Rebuilding active context | **Working-Memory Reconstruction** | Reassembling task state, repository scope, anchors, and admitted evidence. |
| Resume after a refresh | **Cross-Context Recovery** | Continuing the same task from the reconstructed execution frontier. |
| Continuity across refreshes | **Cross-Context Continuity** | Preserving task identity and state alignment across provider context boundaries. |
| Progress mismatch | **Task-State Drift** | The active task representation no longer matches what has been completed or remains unresolved. |
| Repository mismatch | **Repository-State Drift** | The repository has changed enough that previous evidence may no longer apply. |
| Memory/state mismatch | **State-Memory Misalignment** | Recalled evidence and the current task or repository state disagree. |
| Acceptance mismatch | **Task-State Gap** | The recorded task state does not yet establish the required acceptance condition. |
| Benchmark or task boundary | **Execution Milestone** | A named, evaluable unit of progress in a continuous task stream. |
| Corrective branch | **Correction Trace** | New evidence that diagnoses, repairs, verifies, or supersedes an earlier trace. |

## Compatibility names

The implementation retains historical module paths, class names, configuration
keys, database columns, and receipt fields so existing integrations continue to
work. Those identifiers are compatibility details rather than the terminology
used to describe the method. Public prose should use the terms in this table;
schemas and migration code may retain their original names until a versioned
compatibility migration is introduced.
