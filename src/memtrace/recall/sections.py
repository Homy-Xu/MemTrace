from __future__ import annotations

import hmac
import re
from typing import Sequence

from ..contracts import digest, stable_id

_CONTINUATION_PATTERN = re.compile(
    r"^sectioncontinuation_(?P<offset>[0-9a-f]+)_(?P<proof>[0-9a-f]{24})$"
)


def semantic_section_handle(
    *,
    page_digest: str,
    revision_id: str,
    event_group_ids: Sequence[str],
    event_ids: Sequence[str],
) -> str:
    """Return the stable public address of one semantic trace section.

    A Memory Anchor addresses the immutable Memory Trace (or Episode segment). The section
    handle narrows that address to complete EventGroup boundaries without
    exposing an internal trace ID, byte range, or repository search key.
    """

    if not event_group_ids or not event_ids:
        raise ValueError("a semantic section requires EventGroup and Event identities")
    return stable_id(
        "section_",
        {
            "page_digest": page_digest,
            "revision_id": revision_id,
            "event_group_ids": tuple(event_group_ids),
            "event_ids": tuple(event_ids),
        },
    )


def section_continuation_token(
    *,
    section_handle: str,
    full_content_digest: str,
    next_character: int,
) -> str:
    """Create an opaque, restart-safe cursor within one immutable section."""

    if not section_handle.startswith("section_"):
        raise ValueError("continuation requires a semantic section handle")
    if next_character <= 0:
        raise ValueError("continuation cursor must advance beyond the section start")
    proof = digest(
        {
            "section_handle": section_handle,
            "full_content_digest": full_content_digest,
            "next_character": next_character,
        }
    ).removeprefix("sha256:")[:24]
    return f"sectioncontinuation_{next_character:x}_{proof}"


def section_continuation_offset(
    token: str,
    *,
    section_handle: str,
    full_content_digest: str,
    total_characters: int,
) -> int:
    """Validate an opaque continuation against the reopened immutable body."""

    match = _CONTINUATION_PATTERN.fullmatch(token)
    if match is None:
        raise ValueError("invalid semantic section continuation")
    offset = int(match.group("offset"), 16)
    if offset <= 0 or offset >= total_characters:
        raise ValueError("semantic section continuation is outside the section body")
    expected = section_continuation_token(
        section_handle=section_handle,
        full_content_digest=full_content_digest,
        next_character=offset,
    )
    if not hmac.compare_digest(expected, token):
        raise ValueError("semantic section continuation does not match the addressed body")
    return offset
