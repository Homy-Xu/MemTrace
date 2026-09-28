# Reproducibility

## Release checks

The release gate runs the core contract tests, installs a clean Python 3.11
environment, and executes one real task through each of these combinations:

| Harness | SWE-Milestone | DeepSWE | SWE-EVO |
| --- | --- | --- | --- |
| Codex App Server | smoke | smoke | smoke |
| mini-swe-agent 2.4.6 | smoke | smoke | smoke |

Each smoke uses a new run root, repository checkout, container identity,
provider session, and receipt path. A smoke is complete only when generation,
official evaluation, score publication, usage accounting, and cleanup all have
receipts. A launcher or container failure is recorded as infrastructure
failure and is not converted into a reward value.

## Result provenance

`results/manifests/` contains only redacted metadata. A row identifies the
source and wheel digest used for that task and points to the official campaign
receipt outside the public repository. Cross-campaign summaries retain their
component campaigns and are labeled as summaries.

Full task datasets, hidden tests, private prompts, raw trajectories, provider
configuration, and cluster paths are intentionally excluded.
