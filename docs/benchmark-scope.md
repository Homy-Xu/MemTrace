# Benchmark scope and provenance

## Published mini-swe-agent path

The code under `src/memtrace/harness/mini_swe_agent/` publishes the
mini-swe-agent 2.4.6 adapter used for DeepSWE. It provides the provider-neutral
Harness lifecycle, Memory Trace persistence, Trace Recall, Working Memory,
Context Refresh, MTG/RSG alignment, usage accounting, and redacted receipts.
The public profile is configured for MemTensor's
`deepseek-v4-flash-0731` Responses endpoint with a 200,000-token context.

The archive source identity for this adapter is commit `98e4e98`, whose parent
`d77a1fb` introduced the DeepSWE runtime bridge. The public branch may contain
an equivalent cherry-picked commit with a different Git object ID; use the
source digest in the build receipt when binding an experiment to code.

## Why this is not SWE-Milestone

DeepSWE and SWE-Milestone have different task units and different evaluator
contracts. DeepSWE evaluates 113 repository-level tasks, normally one issue per
clean checkout, and reports task-level `pass@1`. SWE-Milestone evaluates 98
graded milestones within seven continuous repository itineraries:

- go-zero: 23 milestones
- Navidrome: 9 milestones
- element-web: 18 milestones
- Nushell: 13 milestones
- ripgrep: 11 milestones
- Dubbo: 12 milestones
- scikit-learn: 12 milestones

A SWE-Milestone run must preserve the repository stream between milestones,
track the active official milestone ID and predecessor state, create the
milestone-scoped patch/tag, and submit a separate official receipt. Its main
metrics are Score, Precision, Recall, and Resolved Rate. A DeepSWE patch,
trajectory, or score cannot be substituted for those receipts.

The runtime also creates internal **Execution Milestones** while planning a
DeepSWE task. These organize work within that task; they do not correspond to
SWE-Milestone's official IDs, evaluator tags, or scored units. A targeted
SWE-Milestone adapter must make that mapping explicit, preserve per-ID commit
anchors and retry accounting, and retain one official receipt per scored
attempt. The current mini-swe-agent CLI rejects the repository-stream and
SWE-Milestone verifier options rather than claiming those semantics.

The current repository keeps compatibility contracts and historical
SWE-Milestone integration code, but this release does not claim a new
mini-swe-agent SWE-Milestone run. The high-scoring historical implementation
is a separate provenance line. The paper records the following identity:

```text
MemTrace runtime: 2.2.97
Git commit: 0dc603d6d78c6cec479d061d5ac460dccffbc5bd
Branch: fix/v2.2.90-five-stage-closure
Build ID: build_61cc38bd4ec1da5ede5cda6097508c58
Source digest: 63cfb8b0359c085c701fad9a25240772faa0487023f448e89de0dacc81f9fecc
Wheel: homy_longterm_v2-2.2.97-py3-none-any.whl
Wheel SHA256: 322148bfb10b0993e41b894b00ab838726384f7dc7618b0b2feb0e98ebdb0e98
```

Those fields are historical provenance only. The matching source tree, wheel,
per-milestone receipts, and official evaluator archive are not included in
this public mini-swe-agent release. A reproduction claim requires all of them
to be available together.
