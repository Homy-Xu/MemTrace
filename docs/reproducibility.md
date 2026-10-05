# Reproducibility

## Release checks

This release validates the two DeepSWE entry points in a clean Python 3.11+
environment. Both use `deepseek-v4-flash-0731`, a 200,000-token context
window, native compaction disabled, 200 execution turns, and a 14,400-second
wall-clock budget. The Codex CLI path uses the App Server; the mini-swe-agent
path uses version 2.4.6 and litellm.

The offline gate runs:

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
memtrace validate-config \
  --config configs/codex/memtensor-deepseek-v4-flash-0731-five-stage.json
memtrace validate-config \
  --config configs/mini_swe_agent/memtensor-deepseek-v4-flash-0731-five-stage.json
```

Any live smoke uses a new run root, repository checkout, provider session, and
receipt path. It is complete only when generation, official evaluation, score
publication, usage accounting, and cleanup all have receipts. A launcher,
provider, container, evaluator, or network failure is recorded as
`INFRA_OR_AGENT_FAILURE` and is never converted into a model score.

The public release manifest records whether each combination is complete or
blocked. A blocked smoke is an explicit reproducibility status, not an inferred
benchmark result.

SWE-Milestone and SWE-EVO are not release smoke targets here. SWE-Milestone
requires a repository stream, milestone-level evaluator, and attempt-scoped
receipts; those boundaries must be adapted and validated separately.

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
