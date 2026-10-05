# Reproducibility

## Release checks

The release gate installs a clean Python 3.11 environment, runs the core
contract tests, and executes one real task through each harness and benchmark
combination:

| Harness | SWE-Milestone | DeepSWE | SWE-EVO |
| --- | --- | --- | --- |
| Codex App Server | smoke | smoke | smoke |
| mini-swe-agent 2.4.6 | smoke | smoke | smoke |

Each smoke uses a new run root, repository checkout, container identity,
provider session, and receipt path. The DeepSWE mini-swe-agent reference run
uses version 2.4.6 through the five-stage runtime: `deepseek-v4-flash-0731`,
a 200,000-token context window, native compaction disabled, 200 execution
turns, and a 14,400-second wall-clock budget. A smoke is complete only when generation,
official evaluation, score publication, usage accounting, and cleanup all have
receipts. A launcher, provider, container, evaluator, or network failure is
recorded as `INFRA_OR_AGENT_FAILURE` and is never converted into a model score.

The public release manifest records whether each combination is complete or
blocked. A blocked smoke is an explicit reproducibility status, not an inferred
benchmark result.

## Result provenance

`results/manifests/` contains only redacted metadata. Each row identifies the
benchmark, task, harness and version, source digest, wheel digest, official
score when available, F2P/P2P counts, wall-clock time, token usage, cost,
generation status, evaluation status, and failure classification. It points to
the official campaign receipt outside the public repository.

Cross-campaign summaries retain their component campaigns and are labeled as
summaries. A score-only regrade, rerun, complete campaign, infrastructure
failure, and model-quality failure remain distinct provenance classes.

## Reproduction boundary

Full task datasets, hidden tests, private prompts, provider configuration,
cluster paths, raw trajectories, Docker state, and evaluator workspaces are
intentionally excluded. Reproduction requires the corresponding external
benchmark assets and an authorized provider credential supplied through the
protected environment.

Use a fresh run root and independent repository checkout for each attempt. Do
not reuse a receipt directory, container identity, provider session, or mutable
workspace across attempts.
