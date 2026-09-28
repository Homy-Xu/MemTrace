from __future__ import annotations

from dataclasses import replace

from ..contracts import (
    ContextArtifact,
    ContextHandle,
    ContextImage,
    Representation,
    digest,
    primitive,
    stable_id,
)


def count_tokens(content: str) -> int:
    """Return the physical prompt cost used by the local admission policy.

    Empty content is deliberately zero.  A NONRESIDENT artifact keeps its
    locator in ``source_handles`` and contributes no prompt body.
    """

    return (len(content.encode("utf-8")) + 2) // 3


def artifact_from_content(
    *,
    content: str,
    representation: Representation,
    milestone_ids: tuple[str, ...] = (),
    entity_refs: tuple[str, ...] = (),
    source_handles: tuple[ContextHandle, ...] = (),
    must_preserve: bool = False,
    current_milestone: bool = False,
    verified: bool = True,
    soft_pin_boundaries: int = 0,
    derived_from: tuple[str, ...] = (),
    memory_ref: str | None = None,
    identity_seed: object | None = None,
) -> ContextArtifact:
    if representation == Representation.NONRESIDENT and content:
        raise ValueError("NONRESIDENT artifacts cannot retain prompt content")
    content_digest = digest({"content": content})
    if memory_ref is None and source_handles:
        # A recoverable artifact gets one stable virtual address at creation.
        # Representation changes must never change the address the model saw.
        memory_ref = stable_id(
            "memoryref_",
            {
                "source_handles": tuple(
                    (
                        handle.page_id,
                        handle.event_range,
                        handle.blob_handle,
                        handle.blob_range,
                        handle.content_digest,
                        handle.revision_id,
                    )
                    for handle in source_handles
                ),
            },
        )
    return ContextArtifact(
        artifact_id=stable_id(
            "artifact_",
            {
                "content_digest": content_digest,
                "representation": representation.value,
                "seed": identity_seed,
            },
        ),
        representation=representation,
        content=content,
        token_count=count_tokens(content),
        content_digest=content_digest,
        milestone_ids=milestone_ids,
        entity_refs=entity_refs,
        source_handles=source_handles,
        must_preserve=must_preserve,
        current_milestone=current_milestone,
        verified=verified,
        soft_pin_boundaries=soft_pin_boundaries,
        derived_from=derived_from,
        memory_ref=memory_ref,
    )


class ContextImageBuilder:
    def build(
        self,
        *,
        thread_id: str,
        artifacts: tuple[ContextArtifact, ...],
        current_milestone_id: str,
        revision_id: str,
    ) -> ContextImage:
        if not thread_id or not current_milestone_id or not revision_id:
            raise ValueError("ContextImage scope identifiers must be non-empty")
        if len({item.artifact_id for item in artifacts}) != len(artifacts):
            raise ValueError("ContextImage cannot contain duplicate artifact ids")
        for item in artifacts:
            expected_tokens = count_tokens(item.content)
            if item.token_count != expected_tokens:
                raise ValueError(f"artifact {item.artifact_id} token_count is not physical content")
            if item.representation == Representation.NONRESIDENT and item.content:
                raise ValueError("NONRESIDENT artifact leaked physical content")
        total_tokens = sum(item.token_count for item in artifacts)
        image_digest = digest(
            {
                # The digest is a checkpoint of the complete physical and
                # residency state.  In particular, a soft-pin boundary change
                # must produce a different candidate Context digest.  Thread
                # identity is deliberately excluded: an Epoch replacement
                # must prove that it injects the exact same continuity image.
                "artifacts": [primitive(item) for item in artifacts],
                "total_tokens": total_tokens,
                "current_milestone_id": current_milestone_id,
                "revision_id": revision_id,
            }
        )
        return ContextImage(
            image_id=stable_id("context_image_", image_digest),
            thread_id=thread_id,
            artifacts=artifacts,
            total_tokens=total_tokens,
            image_digest=image_digest,
            current_milestone_id=current_milestone_id,
            revision_id=revision_id,
        )

    def advance_boundary(self, image: ContextImage) -> ContextImage:
        artifacts = tuple(
            replace(
                item,
                soft_pin_boundaries=max(0, item.soft_pin_boundaries - 1),
            )
            if item.soft_pin_boundaries
            else item
            for item in image.artifacts
        )
        return self.build(
            thread_id=image.thread_id,
            artifacts=artifacts,
            current_milestone_id=image.current_milestone_id,
            revision_id=image.revision_id,
        )
