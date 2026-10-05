# Result manifests

This directory contains small, redacted manifests only. A manifest may record
the benchmark and task identifier, Harness and version, source and wheel
digests, official score, F2P/P2P counts, wall-clock time, token usage, cost,
generation status, evaluation status, and failure classification.

The public mini-swe-agent release is scoped to DeepSWE. A DeepSWE task receipt
must not be converted into a SWE-Milestone receipt: SWE-Milestone requires a
continuous repository stream, milestone-scoped official IDs, and its own
Score/Precision/Recall/Resolved Rate evaluation.

Receipts distinguish complete campaigns, reruns, score-only regrades,
infrastructure or agent failures, and model-quality failures. Historical
manifests retain their original provenance and are never silently combined into
a new benchmark claim.

Do not copy raw trajectories, private prompts, provider files, cluster logs,
Docker state, or evaluator workspaces here.
