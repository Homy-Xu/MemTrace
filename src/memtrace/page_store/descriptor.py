from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from ..contracts import Event, EventGroup, FactType, PageManifest, primitive


@dataclass(frozen=True, slots=True)
class PageSemanticDescriptor:
    """Deterministic, bounded description of one immutable Memory Trace."""

    page_id: str
    delta_kinds: tuple[str, ...]
    delta_summary: str
    outcome: str
    changed_files: tuple[str, ...]
    changed_symbols: tuple[str, ...]
    test_refs: tuple[str, ...]
    decision_refs: tuple[str, ...]
    unresolved_refs: tuple[str, ...]
    supporting_event_ids: tuple[str, ...]
    relation_targets: Mapping[str, tuple[str, ...]]
    relation_provenance: Mapping[
        str,
        Mapping[str, tuple[Mapping[str, object], ...]],
    ]


_RELATION_FIELDS = {
    "CORRECTED_BY": "corrects_page_ids",
    "VERIFIED_BY": "verifies_page_ids",
    "SUPERSEDED_BY": "supersedes_page_ids",
    "DEPENDED_ON_BY": "depends_on_page_ids",
    "RESOLVED_BY": "resolves_page_ids",
}


def _strings(value: object) -> tuple[str, ...]:
    if isinstance(value, str):
        return (value,) if value.strip() else ()
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes)):
        return tuple(str(item) for item in value if str(item).strip())
    return ()


def _fact_summary(content: Mapping[str, object]) -> str | None:
    for key in ("summary", "decision", "rationale", "question", "path", "command"):
        value = content.get(key)
        if isinstance(value, str) and value.strip():
            return " ".join(value.split())[:240]
        if isinstance(value, Mapping):
            nested = value.get("summary", value.get("rationale"))
            if isinstance(nested, str) and nested.strip():
                return " ".join(nested.split())[:240]
    return None


def describe_page(
    manifest: PageManifest,
    groups: tuple[EventGroup, ...],
) -> PageSemanticDescriptor:
    """Describe only facts already present in the consolidated trace; never predict edges."""

    kinds: set[str] = set()
    changed_files: set[str] = set()
    changed_symbols: set[str] = set()
    tests: set[str] = set()
    decisions: set[str] = set()
    unresolved: set[str] = set()
    snippets: list[str] = []
    relations: dict[str, set[str]] = {edge: set() for edge in _RELATION_FIELDS}
    relation_provenance: dict[str, dict[str, list[Mapping[str, object]]]] = {
        edge: {} for edge in _RELATION_FIELDS
    }
    failed = False
    verified = False
    events: list[Event] = [event for group in groups for event in group.events]
    for event in events:
        for edge_type, field in _RELATION_FIELDS.items():
            relations[edge_type].update(_strings(event.payload.get(field)))
        raw_provenance = event.payload.get("relation_provenance")
        if isinstance(raw_provenance, Mapping):
            for edge_type, field in _RELATION_FIELDS.items():
                by_target = raw_provenance.get(field, raw_provenance.get(edge_type))
                if not isinstance(by_target, Mapping):
                    continue
                for target, records in by_target.items():
                    values = (
                        records
                        if isinstance(records, Sequence) and not isinstance(records, (str, bytes))
                        else (records,)
                    )
                    accepted = [dict(item) for item in values if isinstance(item, Mapping)]
                    if accepted:
                        relation_provenance[edge_type].setdefault(str(target), []).extend(accepted)
        for fact in event.facts:
            key = fact.key
            content = fact.content
            for edge_type, field in _RELATION_FIELDS.items():
                relations[edge_type].update(_strings(content.get(field)))
            entity = key.canonical_entity_id
            if key.evidence_type is FactType.CODE_CHANGE:
                kinds.add("IMPLEMENT")
                if entity.startswith("file:"):
                    changed_files.add(entity)
                elif entity.startswith("symbol:"):
                    changed_symbols.add(entity)
            elif key.evidence_type in {
                FactType.TEST_RESULT,
                FactType.VERIFIER_RESULT,
            }:
                kinds.add("VERIFY")
                verified = verified or content.get("success") is True
                if entity.startswith(("test:", "tool:")):
                    tests.add(entity)
                failed = failed or content.get("success") is False
            elif key.evidence_type is FactType.TOOL_RESULT:
                # Commands remain durable execution facts. They become
                # verification/failure semantics only when a Criterion owns
                # them; otherwise a typo, cancelled grep, or missing optional
                # utility would pollute the active Working Set as if the task
                # itself had failed.
                kinds.add("EXECUTE")
                criterion_ids = _strings(content.get("criterion_ids"))
                if criterion_ids:
                    kinds.add("VERIFY")
                    verified = verified or content.get("success") is True
                    failed = failed or content.get("success") is False
                    if entity.startswith("tool:"):
                        tests.add(entity)
            elif key.evidence_type is FactType.TEST_FAILURE:
                kinds.update(("DIAGNOSE", "VERIFY"))
                tests.add(entity)
                failed = True
            elif key.evidence_type in {
                FactType.IMPLEMENTATION_DECISION,
                FactType.PLAN_DECISION,
                FactType.USER_CONSTRAINT,
            }:
                kinds.add("REPLAN" if key.evidence_type is FactType.PLAN_DECISION else "DECIDE")
                decisions.add(entity)
            elif key.evidence_type is FactType.CODE_OBSERVATION:
                kinds.add("DIAGNOSE")
            elif key.evidence_type is FactType.UNRESOLVED_QUESTION:
                unresolved.add(entity)
            summary = _fact_summary(content)
            if summary and summary not in snippets:
                snippets.append(summary)

    if not kinds:
        kinds.add("EXECUTE")
    outcome = "FAILED" if failed else "SUCCESS" if verified else "PARTIAL"
    parts = ["/".join(sorted(kinds))]
    if changed_files:
        parts.append("changed " + ", ".join(sorted(changed_files)[:8]))
    if decisions:
        parts.append("decisions " + ", ".join(sorted(decisions)[:6]))
    if tests:
        parts.append("verification " + ", ".join(sorted(tests)[:6]))
    if snippets:
        parts.append("facts " + " | ".join(snippets[:4]))
    if unresolved:
        parts.append("unresolved " + ", ".join(sorted(unresolved)[:6]))
    summary = "; ".join(parts)[:1600]
    return PageSemanticDescriptor(
        page_id=manifest.page_id,
        delta_kinds=tuple(sorted(kinds)),
        delta_summary=summary,
        outcome=outcome,
        changed_files=tuple(sorted(changed_files)),
        changed_symbols=tuple(sorted(changed_symbols)),
        test_refs=tuple(sorted(tests)),
        decision_refs=tuple(sorted(decisions)),
        unresolved_refs=tuple(sorted(unresolved)),
        supporting_event_ids=tuple(event.event_id for event in events),
        relation_targets={
            edge: tuple(sorted(targets)) for edge, targets in relations.items() if targets
        },
        relation_provenance={
            edge: {
                target: tuple(records) for target, records in sorted(by_target.items()) if records
            }
            for edge, by_target in relation_provenance.items()
            if by_target
        },
    )


def descriptor_payload(descriptor: PageSemanticDescriptor) -> dict[str, object]:
    return primitive(descriptor)
