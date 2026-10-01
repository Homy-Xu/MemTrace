# Result manifests

This directory is reserved for small, redacted JSON manifests. A manifest may
contain the benchmark and task identifier, source and wheel digests, official
score, F2P/P2P counts, wall-clock time, token usage, cost, generation status,
evaluation status, and failure classification.

Receipts distinguish complete campaigns, reruns, score-only regrades,
infrastructure or agent failures, and model-quality failures. Historical
manifests retain their original provenance and are never silently combined into
a new benchmark claim.

Do not copy raw trajectories, private prompts, provider files, cluster logs,
Docker state, or evaluator workspaces here.
