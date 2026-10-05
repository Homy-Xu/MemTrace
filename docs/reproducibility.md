# Reproducibility

## Public release checks

This release publishes the mini-swe-agent 2.4.6 DeepSWE adapter. The offline
gate covers the shared runtime, both Harness contracts, the MemTensor request
sanitizer, receipt serialization, and the DeepSWE profile:

```bash
python -m compileall -q src
python -m pytest tests/unit tests/contract tests/integration
python -m memtrace --help
python -m memtrace.benchmarks.mini --help
```

A real provider run requires a protected MemTensor credential and the external
DeepSWE image, task manifest, and evaluator. A public checkout cannot claim an
official score from local tests. The release manifest records the offline gate
and whether a provider smoke was available at build time.

## A2 validation boundary

The uploaded source line was exercised on A2 with mini-swe-agent 2.4.6 and the
MemTensor Responses endpoint. Private run roots, image references, scheduler
configuration, credentials, raw trajectories, and evaluator workspaces remain
outside this repository. Re-run from a clean checkout and bind the resulting
receipt to the exact source digest and wheel SHA before reporting a result.

## Receipt requirements

Every task receipt should contain:

- generation status and patch/trajectory identity;
- evaluation status and official score, when the external evaluator publishes it;
- F2P/P2P counts when the benchmark defines them;
- API calls, input/output/total tokens, provider cost or `null` with
  `cost_available=false`, and wall-clock time;
- Harness version, source digest, wheel SHA, and model/provider identity;
- failure classification.

Provider/auth, image/container, context-limit, agent-stall, evaluator,
infrastructure, and model-quality failures are separate classes. A process
exit of zero without an evaluator receipt is not a score. A reward of zero
with complete generation and evaluation is model-quality evidence; a provider
or container failure is not.

## Benchmark boundary

The mini-swe-agent profile is for DeepSWE. SWE-Milestone has a different
repository-stream contract and milestone-level evaluator; see
[benchmark-scope.md](benchmark-scope.md) before attempting that benchmark.
Historical result manifests remain provenance records and are not silently
combined into a new campaign.
