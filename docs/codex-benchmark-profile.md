# Codex benchmark compatibility profile

The successful Codex benchmark runs used a fixed App Server contract.  The
public profile at
[`configs/codex/memtensor-deepseek-v4-flash.json`](../configs/codex/memtensor-deepseek-v4-flash.json)
captures the runtime settings that matter for long-horizon DeepSWE,
SWE-Milestone, and SWE-EVO tasks:

- Codex App Server `0.144.4` from the pinned project dependency;
- the custom Responses provider at `https://api-int.memtensor.cn/v1`;
- a 200,000-token model context and native compaction disabled;
- high reasoning effort unless the command line overrides it;
- a 10,800-second App Server idle wait, with no runtime step, turn, or tool
  truncation; and
- full access inside the task's isolated repository boundary.

The profile contains only the name of the credential environment variable.
Load the protected provider and proxy files without sourcing them in a shell:

```bash
memtrace run \
  --harness codex \
  --config configs/codex/memtensor-deepseek-v4-flash.json \
  --env-file /protected/provider.env \
  --env-file /protected/proxy.env \
  --provider-api-key-env MEMTENSOR_API_KEY \
  --model deepseek-v4-flash \
  --reasoning-effort high \
  --repository /path/to/checkout \
  --task-file task.txt \
  --run-root /tmp/memtrace-codex-task
```

Each environment file must be a regular file with mode `0600` or stricter
and contain only `NAME=value` assignments (an optional `export` prefix is
accepted).  MemTrace parses the assignments directly; it does not execute
the file, expand variables, or include values in receipts.

The three benchmark adapters still delegate containers, hidden tests, and
official scoring to their benchmark runners.  This profile aligns the Codex
transport and resource contract with the prior successful runs; it does not
turn a local smoke into an official score.  A real run must still provide the
corresponding task image, evaluator, isolated checkout, and valid provider
credential.

The historical campaign also used deployment-specific model aliases such as
`deepseek-v4-flash-0731`.  Pass that exact alias with `--model` when the
authorized endpoint advertises it; the profile's provider and context
settings remain unchanged.
