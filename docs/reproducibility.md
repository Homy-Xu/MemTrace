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

## Validation status

The author-provided `98e4e98` archive is reported as tested on A2 with
mini-swe-agent 2.4.6. Its original full campaign archive is not bundled with
this publication. This provenance is distinct from the canary below.

A fresh check of publication commit `3e7174d` reached 186 MemTensor API calls
but was stopped and excluded from scoring. Its external launcher ran
generation on the host without the prepared task dependencies and retained
upstream Git refs beyond the task baseline; the agent accessed those refs.
No official evaluator result was produced. This is a launcher/environment
validation failure, not a measured model-quality result. The earlier request
serialization errors did not recur in that check, which is narrower evidence
than a successful end-to-end benchmark run.

The corrected Docker canary for publication commit `d2de921` used a clean
`a691069f` checkout, the pinned task image, mini-swe-agent 2.4.6, and the
MemTensor endpoint. Generation completed with 193 API calls in 1669.10 seconds;
the separate official evaluator reported reward 1, F2P 44/44, and P2P
2738/2738. The patch was recovered from the original container before it was
removed, and score-only evaluation made no additional model calls. This is a
valid single-task Docker canary for the generic `MiniSweAgentBackend +
BenchmarkRunner` integration. It is not a 113-task DeepSWE campaign and does
not validate the canonical local-environment `memtrace run` path's full memory
runtime. Provider cost was unavailable and is recorded as `null`.

A new full run must use the task's prepared environment, benchmark-approved
repository history, and separate evaluator. Bind its receipts to the source
digest, wheel SHA, task, and image before reporting a result. Private run roots,
scheduler configuration, credentials, raw trajectories, and evaluator
workspaces remain outside this repository.

## Receipt requirements

The external launcher's redacted task manifest should contain:

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
with valid generation and evaluation can be model-quality evidence. A provider,
container, or setup failure is not. A trajectory cost of zero under
`cost_tracking=ignore_errors` is insufficient to establish a measured cost.

## Benchmark boundary

The mini-swe-agent profile is for DeepSWE. SWE-Milestone has a different
repository-stream contract and milestone-level evaluator; see
[benchmark-scope.md](benchmark-scope.md) before attempting that benchmark.
Historical result manifests remain provenance records and are not silently
combined into a new campaign.
