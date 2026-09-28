from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

from ..contracts import digest
from .workspace import run_command

IMAGE_RECEIPT_SCHEMA = "codex-longterm-v2/docker-image-artifact@1"
_VALID_DESCRIPTOR_MEDIA_TYPES = frozenset(
    {
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.docker.distribution.manifest.list.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
        "application/vnd.oci.image.index.v1+json",
    }
)


class DockerImageArtifactError(RuntimeError):
    """The requested benchmark image is absent, mutable, or not runnable."""


def _canonical_pull_ref(source_ref: str) -> str:
    value = source_ref.strip()
    if not value:
        raise ValueError("Docker image reference is empty")
    first = value.split("/", 1)[0]
    if first in {"docker.io", "index.docker.io"}:
        return "registry-1.docker.io/" + value.split("/", 1)[1]
    if first == "registry-1.docker.io" or "." in first or ":" in first or first == "localhost":
        return value
    if "/" not in value:
        value = "library/" + value
    return "registry-1.docker.io/" + value


def _repository(ref: str) -> str:
    value = ref.split("@", 1)[0]
    slash = value.rfind("/")
    colon = value.rfind(":")
    return value[:colon] if colon > slash else value


def _inspect(ref: str, *, cwd: Path) -> Mapping[str, object] | None:
    completed = run_command(
        ["docker", "image", "inspect", ref],
        cwd=cwd,
        check=False,
    )
    if completed.returncode != 0:
        return None
    try:
        value = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        raise DockerImageArtifactError("Docker image inspect returned invalid JSON") from exc
    if not isinstance(value, list) or len(value) != 1 or not isinstance(value[0], Mapping):
        raise DockerImageArtifactError("Docker image inspect returned an invalid object")
    return value[0]


def _validated_image(document: Mapping[str, object], *, reference: str) -> dict[str, object]:
    image_id = str(document.get("Id", ""))
    architecture = str(document.get("Architecture", ""))
    operating_system = str(document.get("Os", ""))
    rootfs = document.get("RootFS")
    rootfs = rootfs if isinstance(rootfs, Mapping) else {}
    layers = rootfs.get("Layers")
    layers = tuple(map(str, layers)) if isinstance(layers, list) else ()
    descriptor = document.get("Descriptor")
    descriptor = descriptor if isinstance(descriptor, Mapping) else {}
    media_type = str(descriptor.get("mediaType", ""))
    config = document.get("Config")
    repo_digests = document.get("RepoDigests")
    repo_digests = (
        tuple(sorted(map(str, repo_digests))) if isinstance(repo_digests, list) else ()
    )
    errors: list[str] = []
    if not image_id.startswith("sha256:") or len(image_id) != 71:
        errors.append("invalid image config digest")
    if operating_system != "linux":
        errors.append(f"unexpected operating system {operating_system!r}")
    if architecture not in {"amd64", "x86_64"}:
        errors.append(f"unexpected architecture {architecture!r}")
    if str(rootfs.get("Type", "")) != "layers" or not layers:
        errors.append("missing runnable RootFS layers")
    if not isinstance(config, Mapping) or not config:
        errors.append("missing image configuration")
    if media_type and media_type not in _VALID_DESCRIPTOR_MEDIA_TYPES:
        errors.append(f"invalid descriptor media type {media_type!r}")
    if errors:
        raise DockerImageArtifactError(
            f"Docker object {reference!r} is not a runnable SWE-EVO image: "
            + "; ".join(errors)
        )
    return {
        "image_id": image_id,
        "architecture": "amd64" if architecture == "x86_64" else architecture,
        "os": operating_system,
        "size": int(document.get("Size", 0)),
        "rootfs_layer_count": len(layers),
        "descriptor_media_type": media_type or None,
        "repo_digests": list(repo_digests),
    }


def _manifest_digest(
    repo_digests: object,
    *,
    preferred_repositories: tuple[str, ...] = (),
) -> str:
    values = tuple(map(str, repo_digests)) if isinstance(repo_digests, list) else ()
    parsed = tuple(
        item.rsplit("@", 1)
        for item in values
        if "@sha256:" in item and len(item.rsplit("@", 1)[1]) == 71
    )
    for repository in preferred_repositories:
        preferred = sorted({item[1] for item in parsed if item[0] == repository})
        if len(preferred) == 1:
            return preferred[0]
        if len(preferred) > 1:
            raise DockerImageArtifactError(
                f"Docker repository {repository!r} has multiple manifest digests"
            )
    digests = sorted({item[1] for item in parsed})
    if len(digests) != 1:
        raise DockerImageArtifactError(
            "Docker image does not resolve to one immutable repository manifest digest"
        )
    return digests[0]


def _validate_receipt(receipt: Mapping[str, object], *, source_ref: str) -> None:
    if receipt.get("schema") != IMAGE_RECEIPT_SCHEMA:
        raise DockerImageArtifactError("Docker image receipt schema is unsupported")
    if str(receipt.get("source_ref", "")) != source_ref:
        raise DockerImageArtifactError("Docker image receipt belongs to another source reference")
    body = {key: value for key, value in receipt.items() if key != "receipt_digest"}
    if str(receipt.get("receipt_digest", "")) != digest(body):
        raise DockerImageArtifactError("Docker image receipt digest mismatch")
    canonical_ref = _canonical_pull_ref(source_ref)
    manifest_digest = str(receipt.get("manifest_digest", ""))
    if (
        receipt.get("canonical_pull_ref") != canonical_ref
        or not manifest_digest.startswith("sha256:")
        or len(manifest_digest) != 71
        or receipt.get("immutable_ref")
        != _repository(canonical_ref) + "@" + manifest_digest
    ):
        raise DockerImageArtifactError("Docker image receipt immutable identity is invalid")


def ensure_image_artifact(
    source_ref: str,
    *,
    cwd: Path,
    expected_receipt: Mapping[str, object] | None = None,
) -> Mapping[str, object]:
    """Return one validated immutable image receipt, restoring it if necessary."""

    canonical_ref = _canonical_pull_ref(source_ref)
    pulled = False
    repaired_invalid_local_ref = False
    if expected_receipt is not None:
        _validate_receipt(expected_receipt, source_ref=source_ref)
        expected_id = str(expected_receipt.get("image_id", ""))
        immutable_ref = str(expected_receipt.get("immutable_ref", ""))
        document = _inspect(expected_id, cwd=cwd)
        if document is None:
            run_command(["docker", "pull", immutable_ref], cwd=cwd, timeout_seconds=3600)
            pulled = True
            document = _inspect(immutable_ref, cwd=cwd)
        if document is None:
            raise DockerImageArtifactError("receipt image could not be restored")
        validated = _validated_image(document, reference=immutable_ref or expected_id)
        if validated["image_id"] != expected_id:
            raise DockerImageArtifactError("restored image ID does not match immutable receipt")
        expected_manifest = str(expected_receipt.get("manifest_digest", ""))
        observed_manifests = {
            value.rsplit("@", 1)[1]
            for value in map(str, validated["repo_digests"])
            if "@sha256:" in value
        }
        if expected_manifest not in observed_manifests:
            raise DockerImageArtifactError(
                "local image manifest does not match immutable receipt"
            )
        run_command(["docker", "tag", expected_id, source_ref], cwd=cwd)
        current = _validated_image(
            _inspect(source_ref, cwd=cwd) or {},
            reference=source_ref,
        )
        if current["image_id"] != expected_id:
            raise DockerImageArtifactError("source tag does not point to receipt image ID")
        return dict(expected_receipt)

    document = _inspect(source_ref, cwd=cwd)
    if document is not None:
        try:
            validated = _validated_image(document, reference=source_ref)
            _manifest_digest(
                validated["repo_digests"],
                preferred_repositories=(
                    _repository(canonical_ref),
                    _repository(source_ref),
                ),
            )
        except DockerImageArtifactError:
            # Remove only the requested benchmark reference. Never prune or
            # remove an image ID that may have unrelated tags/containers.
            run_command(
                ["docker", "image", "rm", source_ref],
                cwd=cwd,
                check=False,
            )
            repaired_invalid_local_ref = True
            document = None
    if document is None:
        canonical_document = _inspect(canonical_ref, cwd=cwd)
        if canonical_document is not None:
            try:
                canonical_validated = _validated_image(
                    canonical_document,
                    reference=canonical_ref,
                )
                _manifest_digest(
                    canonical_validated["repo_digests"],
                    preferred_repositories=(_repository(canonical_ref),),
                )
            except DockerImageArtifactError:
                run_command(
                    ["docker", "image", "rm", canonical_ref],
                    cwd=cwd,
                    check=False,
                )
            else:
                document = canonical_document
        if document is None:
            run_command(["docker", "pull", canonical_ref], cwd=cwd, timeout_seconds=3600)
            pulled = True
            document = _inspect(canonical_ref, cwd=cwd)
    if document is None:
        raise DockerImageArtifactError("Docker pull completed without a local image")
    validated = _validated_image(document, reference=canonical_ref)
    manifest_digest = _manifest_digest(
        validated["repo_digests"],
        preferred_repositories=(
            _repository(canonical_ref),
            _repository(source_ref),
        ),
    )
    image_id = str(validated["image_id"])
    run_command(["docker", "tag", image_id, source_ref], cwd=cwd)
    tagged = _validated_image(_inspect(source_ref, cwd=cwd) or {}, reference=source_ref)
    if tagged["image_id"] != image_id:
        raise DockerImageArtifactError("validated image could not be pinned to source tag")
    immutable_ref = _repository(canonical_ref) + "@" + manifest_digest
    # RepoTags/RepoDigests are mutable local aliases. Only the canonical
    # manifest reference belongs in an immutable receipt; scorer aliases must
    # not change its digest or resume identity.
    validated["repo_digests"] = [immutable_ref]
    body: dict[str, object] = {
        "schema": IMAGE_RECEIPT_SCHEMA,
        "source_ref": source_ref,
        "canonical_pull_ref": canonical_ref,
        "immutable_ref": immutable_ref,
        "manifest_digest": manifest_digest,
        **validated,
        "pulled": pulled,
        "repaired_invalid_local_ref": repaired_invalid_local_ref,
    }
    return {**body, "receipt_digest": digest(body)}
