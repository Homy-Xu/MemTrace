# Architecture

MemTrace has one runtime and two Harness backends. The runtime owns planning,
WAL, pages, semantic addresses, recall, context admission, checkpoint, and
receipt publication. A backend owns provider-specific session and event
translation.

```text
Codex App Server JSONL ─┐
                        ├─ HarnessEvent ─> five-stage runtime ─> receipt
mini-swe-agent 2.4.6 ──┘
```

Codex exposes native planning, incremental plan updates, thread recovery, and
provider compaction capabilities when the pinned App Server supports them.
mini-swe-agent exposes execution turns and trajectory usage; its capability
record explicitly reports that native Codex plan mode and thread resume are
unsupported. The runtime never fabricates those capabilities.

The event ledger is append-only. Each event carries a run, branch, revision,
source event, provider method, normalized payload, and a raw-provider summary.
Payloads are summarized before they enter public receipts; credentials and
private host paths are redacted.
