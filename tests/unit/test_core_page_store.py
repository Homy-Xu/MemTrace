from __future__ import annotations

import json
from pathlib import Path

import pytest

from memtrace.contracts import (
    Event,
    EventGroup,
    EvidenceDraft,
    EvidenceKey,
    FactType,
    PageCandidate,
    PageState,
    RecallIntent,
)
from memtrace.database import StateDatabase
from memtrace.durability import SecretRedactor
from memtrace.page_store import (
    BranchVisibilityError,
    PageBoundaryError,
    PageIntegrityError,
    PagePolicy,
    PageStore,
    TailReason,
)
from memtrace.recall import PageSliceRetriever
from memtrace.semantic_memory.contracts import ExactEvidenceHit

TEST_POLICY = PagePolicy(
    min_tokens=300,
    target_tokens=500,
    nominal_max_tokens=700,
    absolute_max_tokens=1100,
)


def group(
    group_id: str,
    *,
    size: int = 80,
    semantic_boundary: bool = True,
    events: int = 1,
    branch_id: str = "main",
    revision_id: str = "rev-1",
    payload: dict[str, object] | None = None,
) -> EventGroup:
    return EventGroup(
        group_id=group_id,
        group_type="agent_action",
        run_id="run-1",
        branch_id=branch_id,
        revision_id=revision_id,
        events=tuple(
            Event(
                event_id=f"{group_id}-event-{index}",
                event_type="tool_result",
                payload=payload if payload is not None else {"stdout": "x" * size},
                entity_refs=("file:src/example.py",),
                milestone_id="milestone-1",
                revision_id=revision_id,
            )
            for index in range(events)
        ),
        milestone_id="milestone-1",
        semantic_boundary=semantic_boundary,
    )


@pytest.fixture
def state(tmp_path: Path):
    database = StateDatabase(tmp_path / "state.sqlite")
    try:
        yield database
    finally:
        database.close()


def test_default_policy_and_complete_group_state_machine(
    tmp_path: Path, state: StateDatabase
) -> None:
    defaults = PagePolicy()
    assert (
        defaults.min_tokens,
        defaults.target_tokens,
        defaults.nominal_max_tokens,
        defaults.absolute_max_tokens,
    ) == (2048, 6144, 8192, 16384)
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)

    first = group("group-1", size=80, semantic_boundary=False)
    assert first.token_count < TEST_POLICY.min_tokens
    assert store.append_group(first) is None
    assert store.page_state() == PageState.OPEN_BELOW_MIN

    second = group("group-2", size=80, semantic_boundary=False)
    assert store.append_group(second) is None
    assert store.page_state() == PageState.OPEN_NORMAL

    third = group("group-3", size=80, semantic_boundary=True)
    manifest = store.append_group(third)
    assert manifest is not None
    assert manifest.seal_reason == "TARGET_REACHED"
    assert manifest.tail is False
    assert manifest.token_count >= TEST_POLICY.target_tokens
    assert manifest.token_count <= TEST_POLICY.absolute_max_tokens
    assert manifest.event_group_ids == ("group-1", "group-2", "group-3")
    assert [item.group_id for item in store.open_page(manifest.page_id)] == [
        "group-1",
        "group-2",
        "group-3",
    ]
    assert store.page_state() == PageState.OPEN_BELOW_MIN


def test_semantic_boundary_above_min_waits_for_target(tmp_path: Path, state: StateDatabase) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    first = group("group-1", size=30, semantic_boundary=True)
    assert first.token_count < TEST_POLICY.min_tokens
    assert store.append_group(first) is None
    assert store.append_group(group("group-2", size=30, semantic_boundary=True)) is None
    assert store.page_state() == PageState.OPEN_NORMAL
    manifest = store.append_group(group("group-3", size=30, semantic_boundary=True))
    assert manifest is not None
    assert manifest.seal_reason == "TARGET_REACHED"
    assert manifest.tail is False
    assert manifest.token_count >= TEST_POLICY.min_tokens


def test_2076_token_semantic_group_does_not_seal_before_target(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main")
    item = next(
        (
            candidate
            for size in range(3000, 9000)
            if (candidate := group("exact-2076", size=size)).token_count == 2076
        ),
        None,
    )
    assert item is not None, "test fixture could not construct an exact 2076-token group"
    assert store.append_group(item) is None
    assert store.page_state() == PageState.OPEN_NORMAL


def test_64_short_actions_are_grouped_into_bounded_pages(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main")
    for ordinal in range(64):
        store.append_group(group(f"short-{ordinal}", size=5, semantic_boundary=True))
    store.close()
    manifests = store.list_manifests()
    assert 1 <= len(manifests) < 64
    assert all(item.token_count <= store.policy.absolute_max_tokens for item in manifests)


def test_event_group_is_one_synced_wal_record_and_projector_runs_once_per_page(
    tmp_path: Path, state: StateDatabase
) -> None:
    syncs: list[tuple[str, Path]] = []
    projected: list[tuple[str, tuple[str, ...], dict[str, tuple[int, int]]]] = []
    with state.transaction() as conn:
        conn.execute(
            "CREATE TABLE projection_probe(page_id TEXT PRIMARY KEY, event_count INTEGER NOT NULL)"
        )

    def projector(conn, manifest, groups, positions):
        assert conn.in_transaction
        projected.append(
            (manifest.page_id, tuple(item.group_id for item in groups), dict(positions))
        )
        conn.execute(
            "INSERT INTO projection_probe(page_id, event_count) VALUES(?, ?)",
            (manifest.page_id, len(positions)),
        )

    store = PageStore(
        tmp_path / "page-store",
        state,
        "run-1",
        "main",
        policy=TEST_POLICY,
        projector=projector,
        sync_hook=lambda kind, path: syncs.append((kind, path)),
    )
    item = group("complete-tool", size=700, events=3)
    manifest = store.append_group(item)
    assert manifest is not None
    assert sum(kind == "wal" for kind, _ in syncs) == 1
    assert len(store.wal_path.read_text().splitlines()) == 1
    assert len(projected) == 1
    assert projected[0][1] == ("complete-tool",)
    assert len(projected[0][2]) == 3
    assert (
        state.connection.execute(
            "SELECT event_count FROM projection_probe WHERE page_id=?", (manifest.page_id,)
        ).fetchone()["event_count"]
        == 3
    )


def test_oversized_payload_uses_immutable_cas_blob_and_stays_below_absolute(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    original = {"stdout": "large-output-" * 1000}
    manifest = store.append_group(group("large", payload=original))
    if manifest is None:
        manifest = store.close()
    assert manifest is not None
    assert manifest.token_count <= TEST_POLICY.absolute_max_tokens
    opened = store.open_page(manifest.page_id)
    external = opened[0].events[0].payload["external_payload"]
    assert external["blob_handle"].startswith("sha256:")
    assert external["byte_range"] == [0, external["byte_count"]]
    blob = store.open_blob(external["blob_handle"], tuple(external["byte_range"]))
    assert json.loads(blob) == original

    # CAS deduplicates the exact same redacted body without overwriting it.
    store2 = PageStore(tmp_path / "page-store", state, "run-1", "other", policy=TEST_POLICY)
    manifest2 = store2.append_group(group("large-2", branch_id="other", payload=original))
    if manifest2 is None:
        manifest2 = store2.close()
    handles = state.connection.execute("SELECT handle FROM v2_page_blobs").fetchall()
    assert len(handles) == 1


def test_multiple_moderate_payloads_are_externalized_without_splitting_group(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    item = group("many", size=1500, events=3)
    manifest = store.append_group(item)
    assert manifest is not None
    assert manifest.event_group_ids == ("many",)
    assert manifest.token_count <= TEST_POLICY.absolute_max_tokens
    opened = store.open_page(manifest.page_id)
    assert len(opened) == 1
    assert len(opened[0].events) == 3
    assert any("external_payload" in event.payload for event in opened[0].events)


def test_oversized_structural_metadata_is_one_atomic_multi_page_page_set(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    item = EventGroup(
        group_id="logical-large-metadata",
        group_type="PLANNING_VALIDATED",
        run_id="run-1",
        branch_id="main",
        revision_id="rev-1",
        milestone_id="milestone-1",
        events=(
            Event(
                event_id="logical-large-event",
                event_type="PLANNING_VALIDATED",
                payload={"summary": "bounded"},
                entity_refs=tuple(
                    f"symbol:src/module_{index}.py:Worker.run" for index in range(500)
                ),
                milestone_id="milestone-1",
                revision_id="rev-1",
            ),
        ),
    )
    assert item.token_count > TEST_POLICY.absolute_max_tokens

    manifest = store.append_group(item)

    assert manifest is not None
    assert len(store.wal_path.read_text(encoding="utf-8").splitlines()) == 1
    manifests = tuple(
        page for page in store.last_sealed if page.seal_reason == TailReason.PAGE_SET_SEGMENT.value
    )
    assert len(manifests) > 1
    assert all(page.token_count <= TEST_POLICY.absolute_max_tokens for page in manifests)
    page_set = state.connection.execute(
        "SELECT * FROM v2_page_sets WHERE logical_group_id=?",
        (item.group_id,),
    ).fetchone()
    assert page_set is not None
    segments = state.connection.execute(
        "SELECT segment_index,page_id FROM v2_page_set_segments "
        "WHERE page_set_id=? ORDER BY segment_index",
        (page_set["page_set_id"],),
    ).fetchall()
    assert [row["segment_index"] for row in segments] == list(range(len(manifests)))
    assert all(row["page_id"] for row in segments)
    synopsis = state.connection.execute(
        "SELECT * FROM v2_page_set_synopses WHERE page_set_id=?",
        (page_set["page_set_id"],),
    ).fetchone()
    assert synopsis is not None
    assert "PLANNING_VALIDATED" in synopsis["synopsis"]
    directory = state.connection.execute(
        "SELECT segment_index,title,summary,entity_refs_json "
        "FROM v2_page_set_segment_directory WHERE page_set_id=? ORDER BY segment_index",
        (page_set["page_set_id"],),
    ).fetchall()
    assert len(directory) == len(manifests)
    assert all(json.loads(row["entity_refs_json"]) for row in directory)
    assert store.durable_groups() == (item,)


def test_page_set_recovers_atomically_after_wal_fsync(tmp_path: Path, state: StateDatabase) -> None:
    root = tmp_path / "page-store"

    def crash_after_wal(kind: str, _path: Path) -> None:
        if kind == "wal":
            raise RuntimeError("simulated crash after PageSet WAL fsync")

    store = PageStore(
        root,
        state,
        "run-1",
        "main",
        policy=TEST_POLICY,
        sync_hook=crash_after_wal,
    )
    item = EventGroup(
        group_id="recover-large-metadata",
        group_type="PLANNING_VALIDATED",
        run_id="run-1",
        branch_id="main",
        revision_id="rev-1",
        events=(
            Event(
                event_id="recover-large-event",
                event_type="PLANNING_VALIDATED",
                payload={"summary": "bounded"},
                entity_refs=tuple(f"file:src/generated_{index}.py" for index in range(500)),
                revision_id="rev-1",
            ),
        ),
    )
    with pytest.raises(RuntimeError, match="PageSet WAL fsync"):
        store.append_group(item)
    assert len(store.wal_path.read_text(encoding="utf-8").splitlines()) == 1
    assert state.connection.execute("SELECT COUNT(*) FROM v2_page_sets").fetchone()[0] == 0

    reopened = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    report = reopened.recover()

    assert report.scanned_groups == 1
    assert report.recovered_groups == 1
    assert len(report.sealed_pages) > 1
    assert all(
        page.token_count <= TEST_POLICY.absolute_max_tokens for page in reopened.list_manifests()
    )
    assert reopened.durable_groups() == (item,)
    again = reopened.recover()
    assert again.recovered_groups == 0
    assert again.sealed_pages == ()


def test_large_fact_is_externalized_and_recall_reads_exact_blob_content(
    tmp_path: Path, state: StateDatabase
) -> None:
    key = EvidenceKey(
        FactType.TOOL_RESULT,
        "tool:large",
        "OUTPUT",
        "rev-1",
        "main",
    )
    expected = {"stdout": "fact-body-" * 2000}
    item = EventGroup(
        group_id="large-fact",
        group_type="TOOL_RESULT",
        run_id="run-1",
        branch_id="main",
        revision_id="rev-1",
        milestone_id="milestone-1",
        events=(
            Event(
                event_id="large-fact-event",
                event_type="TOOL_RESULT",
                payload={"summary": "large fact"},
                facts=(EvidenceDraft(key, expected),),
                milestone_id="milestone-1",
                revision_id="rev-1",
            ),
        ),
    )
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    store.append_group(item)
    manifest = store.close()
    assert manifest is not None
    opened_fact = store.open_page(manifest.page_id)[0].events[0].facts[0]
    reference = opened_fact.content["external_fact"]
    assert reference["byte_range"] == [0, reference["byte_count"]]

    candidate = PageCandidate(
        page_id=manifest.page_id,
        payload_digest=manifest.payload_digest,
        revision_id="rev-1",
        branch_id="main",
        event_range=manifest.event_range,
        anchor_ids=("anchor",),
        evidence_ids=("evidence",),
        evidence_key_digests=(key.key_digest,),
        estimated_tokens=manifest.token_count,
        freshness_cursor=1,
    )
    hit = ExactEvidenceHit(
        requested_key_digest=key.key_digest,
        stored_key_digest=key.key_digest,
        evidence_id="evidence",
        anchor_id="anchor",
        page_id=manifest.page_id,
        event_id="large-fact-event",
        event_group_id="large-fact",
        event_range=manifest.event_range,
        revision_id="rev-1",
        authority="ASSERTED",
    )
    scan = PageSliceRetriever(store).retrieve(
        candidate=candidate,
        intent=RecallIntent(
            recall_id="large-fact-recall",
            repository_id="repo",
            run_id="run-1",
            branch_id="main",
            revision_id="rev-1",
            required_evidence=(key,),
            current_milestone_id="milestone-1",
            question="recover full output",
            max_slice_tokens=16_384,
            max_recovered_block_tokens=20_000,
            max_context_admission_tokens=20_000,
        ),
        missing_key_digests=frozenset({key.key_digest}),
        hits=(hit,),
    )
    assert scan.body_evidence[0].content == expected
    assert scan.page_slice is not None
    assert scan.page_slice.content["events"][0]["facts"][0]["content"] == expected


def test_direct_file_address_restores_externalized_code_observation_body(
    tmp_path: Path, state: StateDatabase
) -> None:
    marker = "def route_handler(request):\n    return request.app.state.result\n"
    body_key = EvidenceKey(
        FactType.CODE_OBSERVATION,
        "observation:read-routing",
        "provider_observed_content",
        "rev-1",
        "main",
    )
    file_key = EvidenceKey(
        FactType.CODE_OBSERVATION,
        "file:src/routing.py",
        "provider_observed_read",
        "rev-1",
        "main",
    )
    body = {
        "command": "sed -n '1,400p' src/routing.py",
        "accessed_paths": ["src/routing.py"],
        "complete_output": ("# routing source\n" * 900) + marker,
        "complete_output_digest": "sha256:body",
        "success": True,
    }
    item = EventGroup(
        group_id="code-read",
        group_type="TOOL_RESULT",
        run_id="run-1",
        branch_id="main",
        revision_id="rev-1",
        milestone_id="milestone-1",
        events=(
            Event(
                event_id="code-read-event",
                event_type="TOOL_RESULT",
                payload={"summary": "read routing source"},
                facts=(
                    EvidenceDraft(body_key, body),
                    EvidenceDraft(
                        file_key,
                        {
                            "path": "src/routing.py",
                            "observation_ref": "observation:read-routing",
                            "success": True,
                        },
                    ),
                ),
                entity_refs=("file:src/routing.py",),
                milestone_id="milestone-1",
                revision_id="rev-1",
            ),
        ),
    )
    store = PageStore(tmp_path / "code-pages", state, "run-1", "main", policy=TEST_POLICY)
    store.append_group(item)
    manifest = store.close()
    assert manifest is not None
    opened = store.open_page(manifest.page_id)[0].events[0]
    assert "external_fact" in opened.facts[0].content

    candidate = PageCandidate(
        page_id=manifest.page_id,
        payload_digest=manifest.payload_digest,
        revision_id="rev-1",
        branch_id="main",
        event_range=manifest.event_range,
        anchor_ids=(),
        evidence_ids=(),
        evidence_key_digests=(body_key.key_digest, file_key.key_digest),
        estimated_tokens=manifest.token_count,
        freshness_cursor=1,
    )
    scan = PageSliceRetriever(store).retrieve_addressed(
        candidate=candidate,
        intent=RecallIntent(
            recall_id="direct-code-read",
            repository_id="repo",
            run_id="run-1",
            branch_id="main",
            revision_id="rev-1",
            required_evidence=(),
            current_milestone_id="milestone-1",
            question="restore routing implementation",
            entity_refs=("file:src/routing.py",),
            direct_page_ids=(manifest.page_id,),
            source_memory_ref="memoryref_code_read",
            max_slice_tokens=16_384,
            max_recovered_block_tokens=20_000,
            max_context_admission_tokens=20_000,
        ),
    )

    assert scan.page_slice is not None
    restored_facts = scan.page_slice.content["events"][0]["facts"]
    restored_body = next(
        fact["content"]
        for fact in restored_facts
        if fact["key"]["semantic_role"] == "provider_observed_content"
    )
    assert restored_body["complete_output"].endswith(marker)
    assert any(
        fact["key"]["canonical_entity_id"] == "observation:read-routing"
        for fact in restored_facts
    )


def test_below_min_page_requires_precise_legal_tail_reason(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    store.append_group(group("short", size=1, semantic_boundary=True))
    with pytest.raises(PageBoundaryError, match="legal Tail reason"):
        store.checkpoint("INTERNAL_MAINTENANCE")
    assert not store.list_manifests()

    manifest = store.checkpoint(TailReason.USER_CHECKPOINT)
    assert manifest is not None
    assert manifest.tail is True
    assert manifest.seal_reason == "USER_CHECKPOINT"


def test_absolute_safety_seals_existing_tail_before_next_complete_group(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    first = group("first", size=350, semantic_boundary=True)
    second = group("second", size=2000, semantic_boundary=False)
    assert first.token_count < TEST_POLICY.min_tokens
    assert second.token_count <= TEST_POLICY.absolute_max_tokens
    store.append_group(first)
    store.append_group(second)
    assert len(store.last_sealed) >= 1
    first_page = store.list_manifests()[0]
    assert first_page.event_group_ids == ("first",)
    assert first_page.tail is True
    assert first_page.seal_reason == "ABSOLUTE_MAX_SAFETY"
    assert all(
        item.token_count <= TEST_POLICY.absolute_max_tokens for item in store.list_manifests()
    )


def test_active_semantic_unit_spans_bounded_pages_at_complete_group_boundaries(
    tmp_path: Path, state: StateDatabase
) -> None:
    store = PageStore(tmp_path / "page-store", state, "run-1", "main", policy=TEST_POLICY)
    first = group("unit-start", size=350, semantic_boundary=False)
    middle = group("unit-middle", size=2000, semantic_boundary=False)
    end = group("unit-end", size=5, semantic_boundary=True)
    store.append_group(first)
    store.append_group(middle)
    first_page = store.list_manifests()[0]
    assert first_page.event_group_ids == ("unit-start",)
    assert first_page.seal_reason == "ABSOLUTE_MAX_SAFETY"
    assert first_page.tail is True

    manifest = store.append_group(end)
    assert manifest is not None
    assert manifest.event_group_ids == ("unit-middle", "unit-end")
    assert manifest.token_count <= TEST_POLICY.absolute_max_tokens
    assert manifest.seal_reason == "TARGET_REACHED"
    assert manifest.tail is False
    assert all(
        page.token_count <= TEST_POLICY.absolute_max_tokens
        for page in store.list_manifests()
    )


def test_recovery_replays_wal_and_creates_crash_tail_once(
    tmp_path: Path, state: StateDatabase
) -> None:
    root = tmp_path / "page-store"
    store = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    store.append_group(group("durable", size=1, semantic_boundary=False))
    assert len(store.wal_path.read_text().splitlines()) == 1
    # Model a crash after WAL fsync but before the Page Map transaction became
    # durable by removing the reconstructed row.
    with state.transaction() as conn:
        conn.execute("DELETE FROM v2_page_wal_groups WHERE run_id='run-1' AND branch_id='main'")

    reopened = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    report = reopened.recover()
    assert report.scanned_groups == 1
    assert report.recovered_groups == 1
    assert len(report.sealed_pages) == 1
    manifest = reopened.list_manifests()[0]
    assert manifest.tail is True
    assert manifest.seal_reason == "CRASH_RECOVERY"
    assert reopened.open_page(manifest.page_id)[0].group_id == "durable"

    again = reopened.recover()
    assert again.recovered_groups == 0
    assert again.sealed_pages == ()


def test_page_and_blob_tampering_are_detected(tmp_path: Path, state: StateDatabase) -> None:
    root = tmp_path / "page-store"
    store = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    manifest = store.append_group(group("large", payload={"stdout": "z" * 5000}))
    if manifest is None:
        manifest = store.close()
    assert manifest is not None
    page_row = state.connection.execute(
        "SELECT storage_path FROM v2_pages WHERE page_id=?", (manifest.page_id,)
    ).fetchone()
    page_path = root / page_row["storage_path"]
    original_page = page_path.read_bytes()
    page_path.write_bytes(original_page + b"tamper")
    with pytest.raises(PageIntegrityError, match="storage digest"):
        store.open_page(manifest.page_id)
    page_path.write_bytes(original_page)

    blob_row = state.connection.execute(
        "SELECT handle, storage_path FROM v2_page_blobs LIMIT 1"
    ).fetchone()
    blob_path = root / blob_row["storage_path"]
    blob_path.write_bytes(b"tampered")
    with pytest.raises(PageIntegrityError, match="Blob digest"):
        store.open_page(manifest.page_id)


def test_secret_is_redacted_before_wal_blob_page_database_and_index(
    tmp_path: Path,
) -> None:
    secret = "TOP-SECRET-credential-314159"
    database_path = tmp_path / "state.sqlite"
    state = StateDatabase(database_path)
    root = tmp_path / "page-store"
    try:
        store = PageStore(
            root,
            state,
            "run-1",
            "main",
            policy=TEST_POLICY,
            redactor=SecretRedactor((secret,)),
        )
        manifest = store.append_group(
            group(
                "secret",
                payload={
                    "stdout": (secret + " safe-output ") * 700,
                    "nested": {"token": secret},
                },
            )
        )
        if manifest is None:
            manifest = store.close()
        assert manifest is not None
        opened = store.open_page(manifest.page_id)
        external = opened[0].events[0].payload["external_payload"]
        blob = store.open_blob(external["blob_handle"])
        assert secret.encode() not in blob
        assert b"[REDACTED]" in blob
    finally:
        state.close()
    for path in tmp_path.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes(), path


def test_branch_visibility_is_inherited_to_fork_and_isolated_from_sibling(
    tmp_path: Path, state: StateDatabase
) -> None:
    root = tmp_path / "page-store"
    main = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    main.append_group(group("base", size=1))
    base = main.close()
    assert base is not None
    child = PageStore(
        root,
        state,
        "run-1",
        "child",
        policy=TEST_POLICY,
        parent_branch_id="main",
        parent_page_id=base.page_id,
    )
    assert child.open_page(base.page_id)[0].revision_id == "rev-1"
    child_manifest = child.append_group(
        group("child-event", branch_id="child", revision_id="rev-child")
    )
    if child_manifest is None:
        child_manifest = child.close()
    sibling = PageStore(root, state, "run-1", "sibling", policy=TEST_POLICY)
    with pytest.raises(BranchVisibilityError):
        sibling.open_page(child_manifest.page_id)


def test_wal_tampering_is_detected_during_recovery(tmp_path: Path, state: StateDatabase) -> None:
    root = tmp_path / "page-store"
    store = PageStore(root, state, "run-1", "main", policy=TEST_POLICY)
    store.append_group(group("durable", size=1, semantic_boundary=False))
    content = store.wal_path.read_text()
    store.wal_path.write_text(content.replace("tool_result", "tool_tampered", 1))
    with pytest.raises(PageIntegrityError, match="WAL record digest"):
        store.recover()
