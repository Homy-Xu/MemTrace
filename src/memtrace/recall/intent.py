from __future__ import annotations

from dataclasses import replace

from ..contracts import SEMANTIC_PAGE_QUERY_RELATIONS, RecallIntent, unique_by_digest
from ..rich_graph.models import STRUCTURAL_RELATIONS


def normalize_intent(intent: RecallIntent) -> RecallIntent:
    """Return the canonical, exact-key form used by every V2 recall path.

    ``RecallIntent`` is the sole production contract.  This function deliberately
    does not translate exact keys into coarse ``FactType`` buckets.
    """

    required = unique_by_digest(intent.required_evidence)
    if len(required) != len(intent.required_evidence):
        # The contract normally rejects duplicates.  Keep the check here so a
        # deserialized/duck-typed caller cannot weaken Coverage semantics.
        raise ValueError("RecallIntent contains duplicate exact EvidenceKeys")
    question = " ".join(intent.question.split())
    if not question:
        raise ValueError("RecallIntent.question must be non-empty")
    desired_detail = " ".join(intent.desired_detail.split())
    purpose = " ".join(intent.purpose.split())
    if not desired_detail or not purpose:
        raise ValueError("RecallIntent detail and purpose must be non-empty")
    budgets = (
        intent.page_limit,
        intent.max_page_bytes_read,
        intent.max_blob_bytes_read,
        intent.max_slice_tokens,
        intent.recovered_block_limit,
        intent.admission_limit,
    )
    if any(value <= 0 for value in budgets):
        raise ValueError("Trace Recall and token budgets must be positive")
    if not all(
        value.strip()
        for value in (
            intent.recall_id,
            intent.repository_id,
            intent.run_id,
            intent.branch_id,
            intent.revision_id,
            intent.current_milestone_id,
        )
    ):
        raise ValueError("RecallIntent scope fields must be non-empty")
    required_page_relations = tuple(
        dict.fromkeys(
            item.strip().upper() for item in intent.required_structural_relations if item.strip()
        )
    )
    preferred_page_relations = tuple(
        dict.fromkeys(
            item.strip().upper() for item in intent.preferred_structural_relations if item.strip()
        )
    )
    unknown_page_relations = set((*required_page_relations, *preferred_page_relations)).difference(
        SEMANTIC_PAGE_QUERY_RELATIONS
    )
    if unknown_page_relations:
        raise ValueError(f"unknown Memory Trace relations: {sorted(unknown_page_relations)}")
    rich_code_relations = tuple(
        dict.fromkeys(item.strip().upper() for item in intent.rich_code_relations if item.strip())
    )
    unknown_rich_relations = set(rich_code_relations).difference(STRUCTURAL_RELATIONS)
    if unknown_rich_relations:
        raise ValueError(f"unknown Repository State Graph relations: {sorted(unknown_rich_relations)}")
    return replace(
        intent,
        required_evidence=required,
        direct_page_ids=tuple(dict.fromkeys(intent.direct_page_ids)),
        question=question,
        desired_detail=desired_detail,
        purpose=purpose,
        required_structural_relations=required_page_relations,
        preferred_structural_relations=preferred_page_relations,
        structural_relation_direction=intent.structural_relation_direction.upper(),
        rich_code_relations=rich_code_relations,
    )


def with_required_keys(intent: RecallIntent, required_key_digests: frozenset[str]) -> RecallIntent:
    """Narrow an intent for an actually executed fallback stage."""

    required = tuple(
        key for key in intent.required_evidence if key.key_digest in required_key_digests
    )
    if not required:
        raise ValueError("cannot create a fallback intent without missing evidence")
    return replace(intent, required_evidence=required, direct_page_ids=(), source_memory_ref=None)
