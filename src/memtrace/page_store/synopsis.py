from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from ..contracts import PageManifest, digest, primitive, stable_id
from .descriptor import PageSemanticDescriptor, descriptor_payload

PAGE_SYNOPSIS_SCHEMA = "codex-longterm-v2/page-synopsis@1"


def memory_ref_for_page(page_id: str, page_digest: str) -> str:
    """Return the stable, model-visible virtual address for an immutable Page."""

    if not page_id or not page_digest:
        raise ValueError("Page identity and digest are required for a MemoryRef")
    # page_id remains inside the one-way opaque address.  Including it avoids
    # aliasing two distinct execution locations with identical payload bytes.
    return stable_id(
        "memoryref_",
        {"page_id": page_id, "page_digest": page_digest},
    )


@dataclass(frozen=True, slots=True)
class PageSynopsis:
    """Bounded, model-safe synopsis derived from one authoritative Page.

    Page IDs and relation targets deliberately remain runtime-private.  The model
    sees only a stable MemoryRef, confirmed semantic deltas, and the relation
    kinds that it may request when more detail is needed.
    """

    schema: str
    kind: str
    memory_ref: str
    summary: str
    outcome: str
    delta_kinds: tuple[str, ...]
    changed_files: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    verification_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    unresolved_refs: tuple[str, ...]
    entity_refs: tuple[str, ...]
    entity_ref_count: int
    revision_ids: tuple[str, ...]
    available_page_relations: tuple[str, ...]
    relation_counts: Mapping[str, int]
    descriptor_digest: str
    detail_state: str = "COMPRESSED_PAGE_SYNOPSIS"
    details_available_via_recall: bool = True
    recall_required_before_use: bool = True
    authoritative_payload: str = "PAGE_STORE_ONLY"
    summary_limitations: str = (
        "This bounded synopsis preserves confirmed execution deltas, not full code, "
        "tool output, rationale, or exact historical wording. Resolve its MemoryRef "
        "through recall_memory before relying on omitted detail."
    )


def build_page_synopsis(
    manifest: PageManifest,
    descriptor: PageSemanticDescriptor,
) -> PageSynopsis:
    if descriptor.page_id != manifest.page_id:
        raise ValueError("Page descriptor does not belong to the supplied manifest")
    relation_counts = {
        relation: len(targets)
        for relation, targets in sorted(descriptor.relation_targets.items())
        if targets
    }
    if manifest.previous_page_id is not None:
        relation_counts.setdefault("ADVANCES_TO", 1)
    focus_entities = tuple(
        dict.fromkeys(
            (
                *descriptor.changed_files,
                *descriptor.changed_symbols,
                *descriptor.test_refs,
                *descriptor.decision_refs,
                *descriptor.unresolved_refs,
                *(
                    entity
                    for entity in manifest.entity_refs
                    if entity.startswith(("file:", "symbol:", "test:", "tool:"))
                ),
            )
        )
    )
    return PageSynopsis(
        schema=PAGE_SYNOPSIS_SCHEMA,
        kind="MEMORY_REF",
        memory_ref=memory_ref_for_page(manifest.page_id, manifest.payload_digest),
        summary=descriptor.delta_summary[:1600],
        outcome=descriptor.outcome,
        delta_kinds=descriptor.delta_kinds,
        changed_files=descriptor.changed_files[:12],
        changed_symbols=descriptor.changed_symbols[:12],
        verification_refs=descriptor.test_refs[:12],
        decision_refs=descriptor.decision_refs[:12],
        unresolved_refs=descriptor.unresolved_refs[:12],
        entity_refs=focus_entities[:16],
        entity_ref_count=len(manifest.entity_refs),
        revision_ids=manifest.revision_ids[:4],
        available_page_relations=tuple(sorted(relation_counts)),
        relation_counts=relation_counts,
        descriptor_digest=digest(descriptor_payload(descriptor)),
    )


def synopsis_payload(synopsis: PageSynopsis) -> dict[str, object]:
    return primitive(synopsis)
