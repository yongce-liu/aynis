"""Integrity checks for resumable Phase-3 batch artifacts."""

from __future__ import annotations

from pathlib import Path

from .filesystem import remove_path
from .hashing import sha256_file
from .layout import BatchLayout
from .spec import read_json


def remove_legacy_batch_artifacts(batch_dir: Path) -> None:
    """Delete superseded per-batch Blender and ``genesis`` output locations."""
    for directory in (batch_dir / "blender", batch_dir / "genesis"):
        remove_path(directory)
    remove_path(batch_dir / "blender_build.log")


def batch_complete(
    batch_dir: Path,
    *,
    expected_identity_sha256: str | None = None,
    expected_sample_ids: list[int] | None = None,
) -> bool:
    """Verify a completed batch marker and all manifest-listed artifacts."""
    layout = BatchLayout(batch_dir)
    marker = layout.success_marker
    sim_dir = layout.sim
    manifest = sim_dir / "run_manifest.json"
    validation = sim_dir / "validation.json"
    trajectories = sim_dir / "trajectories.npz"
    if not (
        marker.is_file()
        and manifest.is_file()
        and validation.is_file()
        and trajectories.is_file()
    ):
        return False

    marker_data = read_json(marker)
    if marker_data.get("run_manifest_sha256") != sha256_file(manifest):
        return False

    manifest_data = read_json(manifest)
    for artifact in manifest_data.get("artifacts", {}).values():
        artifact_path = sim_dir / artifact["path"]
        if (
            not artifact_path.is_file()
            or sha256_file(artifact_path) != artifact["sha256"]
        ):
            return False
    if (
        expected_identity_sha256 is not None
        and marker_data.get("run_identity_sha256") != expected_identity_sha256
    ):
        return False
    if (
        expected_sample_ids is not None
        and manifest_data.get("sample_ids") != expected_sample_ids
    ):
        return False
    return True
