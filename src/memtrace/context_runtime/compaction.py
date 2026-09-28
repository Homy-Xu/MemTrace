from __future__ import annotations

import json
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Protocol

from ..contracts import (
    ContextArtifact,
    ContextImage,
    Representation,
    digest,
    stable_id,
)
from ..observability import CounterName, MetricRecorder
from .image import ContextImageBuilder, artifact_from_content


class NativeCompactionState(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"
    AVAILABLE = "AVAILABLE"
    FAILED = "FAILED"


@dataclass(frozen=True, slots=True)
class NativeCompactionCapabilityReceipt:
    state: NativeCompactionState
    adapter_name: str | None
    verification_supported: bool
    reason: str | None


class NativeCompactionAdapter(Protocol):
    """Provider adapter for verifiable, same-Thread native compaction."""

    def capability(self) -> NativeCompactionCapabilityReceipt: ...

    def compact(self, image: ContextImage, *, target_tokens: int) -> ContextImage | None: ...

    def verify(self, source: ContextImage, candidate: ContextImage) -> bool: ...


class Compactor:
    """Deterministic compaction that always changes the represented content."""

    def __init__(
        self,
        metrics: MetricRecorder,
        native_adapter: NativeCompactionAdapter | None = None,
    ) -> None:
        self.metrics = metrics
        self.native_adapter = native_adapter
        self._native_failure: str | None = None

    def native_capability(self) -> NativeCompactionCapabilityReceipt:
        if self.native_adapter is None:
            return NativeCompactionCapabilityReceipt(
                state=NativeCompactionState.UNAVAILABLE,
                adapter_name=None,
                verification_supported=False,
                reason="NO_NATIVE_COMPACTION_ADAPTER",
            )
        try:
            receipt = self.native_adapter.capability()
        except Exception as error:  # provider capability probes are not fatal
            return NativeCompactionCapabilityReceipt(
                state=NativeCompactionState.FAILED,
                adapter_name=type(self.native_adapter).__name__,
                verification_supported=False,
                reason=f"CAPABILITY_PROBE_FAILED:{type(error).__name__}",
            )
        if self._native_failure is not None:
            return replace(
                receipt,
                state=NativeCompactionState.FAILED,
                reason=self._native_failure,
            )
        if receipt.state == NativeCompactionState.AVAILABLE and (
            not receipt.adapter_name or not receipt.verification_supported
        ):
            return replace(
                receipt,
                state=NativeCompactionState.UNAVAILABLE,
                reason="NATIVE_VERIFICATION_UNAVAILABLE",
            )
        return receipt

    def native_compact(
        self,
        source: ContextImage,
        *,
        target_tokens: int,
        builder: ContextImageBuilder,
    ) -> ContextImage | None:
        """Try native compaction without weakening the deterministic fallback."""

        capability = self.native_capability()
        if capability.state != NativeCompactionState.AVAILABLE:
            return None
        assert self.native_adapter is not None
        if target_tokens < 0:
            self._native_failure = "INVALID_NATIVE_TARGET"
            return None
        try:
            candidate = self.native_adapter.compact(source, target_tokens=target_tokens)
            if candidate is None:
                self._native_failure = "NATIVE_ADAPTER_DECLINED"
                return None
            if not self._framework_verify_native(
                source, candidate, target_tokens=target_tokens, builder=builder
            ):
                self._native_failure = "FRAMEWORK_NATIVE_VERIFICATION_FAILED"
                return None
            if not self.native_adapter.verify(source, candidate):
                self._native_failure = "ADAPTER_NATIVE_VERIFICATION_FAILED"
                return None
        except Exception as error:  # safe fallback to deterministic result
            self._native_failure = f"NATIVE_COMPACTION_FAILED:{type(error).__name__}"
            return None
        self._native_failure = None
        self.metrics.increment(CounterName.NATIVE_COMPACTION)
        return candidate

    @staticmethod
    def _framework_verify_native(
        source: ContextImage,
        candidate: ContextImage,
        *,
        target_tokens: int,
        builder: ContextImageBuilder,
    ) -> bool:
        try:
            canonical = builder.build(
                thread_id=candidate.thread_id,
                artifacts=candidate.artifacts,
                current_milestone_id=candidate.current_milestone_id,
                revision_id=candidate.revision_id,
            )
        except (TypeError, ValueError):
            return False
        if canonical != candidate:
            return False
        if (
            candidate.thread_id != source.thread_id
            or candidate.current_milestone_id != source.current_milestone_id
            or candidate.revision_id != source.revision_id
            or candidate.total_tokens >= source.total_tokens
            or candidate.total_tokens > target_tokens
        ):
            return False

        candidate_by_id = {item.artifact_id: item for item in candidate.artifacts}
        source_ids = {item.artifact_id for item in source.artifacts}
        if any(
            item.artifact_id not in source_ids and not source_ids.intersection(item.derived_from)
            for item in candidate.artifacts
        ):
            return False
        candidate_handles = {
            handle for item in candidate.artifacts for handle in item.source_handles
        }
        for original in source.artifacts:
            retained = candidate_by_id.get(original.artifact_id)
            if original.must_preserve or original.current_milestone:
                if retained != original:
                    return False
                continue
            if retained == original:
                continue
            derived = any(original.artifact_id in item.derived_from for item in candidate.artifacts)
            if not derived:
                return False
            if not set(original.source_handles).issubset(candidate_handles):
                return False
        return True

    def demote(
        self,
        artifact: ContextArtifact,
        *,
        focus_terms: tuple[str, ...] = (),
    ) -> ContextArtifact | None:
        if (
            artifact.must_preserve
            or artifact.current_milestone
            or artifact.soft_pin_boundaries > 0
            or "context:milestone_handoff" in artifact.entity_refs
        ):
            return None
        if artifact.representation == Representation.FULL:
            target = Representation.SEMANTIC_SLICE
            content = self._slice(artifact.content, focus_terms)
            verified = artifact.verified
        elif artifact.representation == Representation.SEMANTIC_SLICE:
            if artifact.verified:
                target = Representation.VERIFIED_SUMMARY
                content = self._summary(artifact)
                verified = self._verify_summary(content, artifact)
                if not verified:
                    if not artifact.source_handles:
                        return None
                    target = Representation.HANDLE
                    content = self._handle_content(artifact)
            else:
                if not artifact.source_handles:
                    return None
                target = Representation.HANDLE
                content = self._handle_content(artifact)
                verified = False
        elif artifact.representation == Representation.VERIFIED_SUMMARY:
            if not artifact.source_handles:
                return None
            target = Representation.HANDLE
            content = self._handle_content(artifact)
            verified = artifact.verified
        elif artifact.representation == Representation.HANDLE:
            if not artifact.source_handles:
                return None
            target = Representation.NONRESIDENT
            # NONRESIDENT locators live in the typed artifact metadata, not in
            # the physical prompt body.
            content = ""
            verified = artifact.verified
        else:
            return None
        if content == artifact.content:
            raise RuntimeError("representation demotion did not change physical content")
        compacted = artifact_from_content(
            content=content,
            representation=target,
            milestone_ids=artifact.milestone_ids,
            entity_refs=artifact.entity_refs,
            source_handles=artifact.source_handles,
            must_preserve=artifact.must_preserve,
            current_milestone=artifact.current_milestone,
            verified=verified,
            derived_from=tuple(dict.fromkeys((*artifact.derived_from, artifact.artifact_id))),
            memory_ref=artifact.memory_ref,
            identity_seed={"source": artifact.artifact_id, "target": target.value},
        )
        self.metrics.increment(CounterName.COMPRESSION)
        return compacted

    @staticmethod
    def _slice(content: str, focus_terms: tuple[str, ...]) -> str:
        structured = Compactor._structured_source(content)
        if structured is not None:
            projection = Compactor._semantic_projection(structured, focus_terms=focus_terms)
            return json.dumps(
                {
                    "schema": "codex-longterm-v2/semantic-slice@1",
                    "representation": "SEMANTIC_SLICE",
                    "bounded_derivation": True,
                    "memory_ref": structured.get("memory_ref"),
                    "detail_state": "COMPRESSED_PAGE_SLICE",
                    "recall_required_before_use": bool(structured.get("memory_ref")),
                    "semantic_fields": projection,
                },
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
        lines = content.splitlines()
        selected: list[tuple[int, str]] = []
        normalized = tuple(term.casefold() for term in focus_terms if term)
        for index, line in enumerate(lines):
            if index in {0, len(lines) - 1} or any(term in line.casefold() for term in normalized):
                selected.append((index + 1, line[:512]))
        if len(selected) < min(4, len(lines)):
            for index, line in enumerate(lines[:4]):
                candidate = (index + 1, line[:512])
                if candidate not in selected:
                    selected.append(candidate)
        selected.sort()
        return json.dumps(
            {
                "schema": "codex-longterm-v2/semantic-slice@1",
                "representation": "SEMANTIC_SLICE",
                "bounded_derivation": True,
                "fallback": "TEXT_LINES",
                "lines": selected,
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _summary(artifact: ContextArtifact) -> str:
        claims = Compactor._summary_claims(artifact)
        try:
            source = json.loads(artifact.content)
        except json.JSONDecodeError:
            source = {}
        memory_ref = artifact.memory_ref
        if isinstance(source, dict) and isinstance(source.get("memory_ref"), str):
            memory_ref = str(source["memory_ref"])
        return json.dumps(
            {
                "representation": "VERIFIED_SUMMARY",
                "source_digest": artifact.content_digest,
                "claims": claims,
                "memory_ref": memory_ref,
                "detail_state": "COMPRESSED_PAGE_SYNOPSIS",
                "recall_required_before_use": memory_ref is not None,
                "verification": digest(
                    {"source_digest": artifact.content_digest, "claims": claims}
                ),
            },
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _summary_claims(artifact: ContextArtifact) -> list[object]:
        try:
            source = json.loads(artifact.content)
        except json.JSONDecodeError:
            source = {"text": artifact.content}
        if isinstance(source, dict) and isinstance(source.get("semantic_fields"), dict):
            return [source["semantic_fields"]]
        if isinstance(source, dict) and isinstance(source.get("lines"), list):
            return list(source["lines"][:6])
        else:
            text = json.dumps(source, ensure_ascii=False, sort_keys=True)
            return [text[:512]]

    @staticmethod
    def _verify_summary(content: str, source: ContextArtifact) -> bool:
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            return False
        claims = value.get("claims")
        return (
            value.get("source_digest") == source.content_digest
            and isinstance(claims, list)
            and claims == Compactor._summary_claims(source)
            and value.get("verification")
            == digest({"source_digest": source.content_digest, "claims": claims})
        )

    @staticmethod
    def _handle_content(artifact: ContextArtifact) -> str:
        memory_ref = artifact.memory_ref
        try:
            source = json.loads(artifact.content)
        except json.JSONDecodeError:
            source = None
        if isinstance(source, dict) and isinstance(source.get("memory_ref"), str):
            memory_ref = str(source["memory_ref"])
        if memory_ref is None:
            memory_ref = stable_id(
                "memoryref_",
                {
                    "content_digests": tuple(
                        dict.fromkeys(handle.content_digest for handle in artifact.source_handles)
                    ),
                    "revision_ids": tuple(
                        dict.fromkeys(handle.revision_id for handle in artifact.source_handles)
                    ),
                },
            )
        payload: dict[str, object] = {
            "schema": "codex-longterm-v2/memory-handle@1",
            "representation": "HANDLE",
            "memory_ref": memory_ref,
            "entity_refs": list(artifact.entity_refs),
            "recall_required_before_use": True,
            "source_content_digest": artifact.content_digest,
        }
        semantic_hint = Compactor._handle_semantic_hint(source)
        if semantic_hint:
            payload["semantic_hint"] = semantic_hint
        return json.dumps(
            payload,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        )

    @staticmethod
    def _structured_source(content: str) -> dict[str, object] | None:
        try:
            value = json.loads(content)
        except json.JSONDecodeError:
            marker = "<UNTRUSTED_RECOVERED_DATA>"
            closing = "</UNTRUSTED_RECOVERED_DATA>"
            if marker not in content or closing not in content:
                return None
            inner = content.split(marker, 1)[1].split(closing, 1)[0].strip()
            try:
                value = json.loads(inner)
            except json.JSONDecodeError:
                return None
        return dict(value) if isinstance(value, dict) else None

    @staticmethod
    def _semantic_projection(
        source: dict[str, object],
        *,
        focus_terms: tuple[str, ...],
    ) -> dict[str, object]:
        """Keep bounded semantic contracts instead of arbitrary first/last lines."""

        priority = (
            "schema",
            "kind",
            "goal",
            "objective",
            "target_outcome",
            "summary",
            "delta_summary",
            "outcome",
            "status",
            "purpose",
            "desired_detail",
            "memory_ref",
            "detail_state",
            "recall_required_before_use",
            "synopsis",
            "semantic_directory",
            "recall_id",
            "current_milestone",
            "minimum_acceptance",
            "verification",
            "verification_refs",
            "failed_tests",
            "unresolved",
            "unresolved_refs",
            "decisions",
            "decision_refs",
            "changed_files",
            "changed_symbols",
            "entity_refs",
            "available_page_relations",
            "relation_counts",
            "coverage",
            "source_evidence",
            "data",
            "harness_event",
            "payload",
            "event_type",
            "action_type",
            "execution_phase",
            "events",
            "facts",
            "slices",
            "content",
        )

        def bounded(value: object, depth: int = 0) -> object:
            if isinstance(value, str):
                return " ".join(value.split())[:800]
            if isinstance(value, (bool, int, float)) or value is None:
                return value
            if isinstance(value, (list, tuple)):
                limit = 8 if depth < 2 else 4
                return [bounded(item, depth + 1) for item in value[:limit]]
            if isinstance(value, dict) and depth < 5:
                selected: dict[str, object] = {}
                for key in priority:
                    if key in value:
                        selected[key] = bounded(value[key], depth + 1)
                if not selected:
                    for key in sorted(value)[:8]:
                        selected[str(key)] = bounded(value[key], depth + 1)
                return selected
            return str(value)[:400]

        projected = bounded(source)
        assert isinstance(projected, dict)
        if focus_terms:
            projected["focus_terms"] = [term[:120] for term in focus_terms[:12] if term]
        return projected

    @staticmethod
    def _handle_semantic_hint(source: object) -> dict[str, object]:
        if not isinstance(source, dict):
            return {}
        candidate: object = source
        claims = source.get("claims")
        if isinstance(claims, list) and claims and isinstance(claims[0], dict):
            candidate = claims[0]
        if not isinstance(candidate, dict):
            return {}
        keys = (
            "summary",
            "delta_summary",
            "outcome",
            "delta_kinds",
            "changed_files",
            "changed_symbols",
            "verification_refs",
            "unresolved_refs",
            "available_page_relations",
        )
        return {key: candidate[key] for key in keys if key in candidate}
