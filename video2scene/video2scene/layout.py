"""Canonical filesystem layout for every reconstructed scene."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SCENE_ID = "ball-and-block-fall"
DEFAULT_SCENE_ROOT = PROJECT_ROOT / "outputs" / DEFAULT_SCENE_ID


@dataclass(frozen=True)
class BatchLayout:
    """Paths owned by one Phase-3 batch."""

    root: Path

    @property
    def sim(self) -> Path:
        return self.root / "sim"

    @property
    def success_marker(self) -> Path:
        return self.root / "_SUCCESS"

    @property
    def incomplete_marker(self) -> Path:
        return self.root / "_INCOMPLETE"


@dataclass(frozen=True)
class SceneLayout:
    """Stable output contract shared by all pipeline stages."""

    root: Path

    @property
    def frames(self) -> Path:
        return self.root / "frames"

    @property
    def blender(self) -> Path:
        return self.root / "blender"

    @property
    def sim(self) -> Path:
        return self.root / "sim"

    @property
    def batches(self) -> Path:
        return self.root / "batches"

    def blender_variant(self, seed: int) -> Path:
        if seed < 0:
            raise ValueError("Variant seed must be non-negative")
        return self.blender / "variants" / f"seed_{seed:06d}"

    def batch(self, index: int) -> BatchLayout:
        if index < 0:
            raise ValueError("Batch index must be non-negative")
        return BatchLayout(self.batches / f"batch_{index:06d}")

    def prepare(self) -> None:
        """Create the canonical top-level directories without deleting artifacts."""
        for path in (self.frames, self.blender, self.sim, self.batches):
            path.mkdir(parents=True, exist_ok=True)

    def reference_frame(self, frame_index: int) -> Path:
        return self.frames / f"frame_{frame_index:04d}.png"


def resolve_scene_layout(scene_id: str, output_root: Path | None = None) -> SceneLayout:
    """Resolve a scene root from an explicit path or the repository default."""
    root = (
        output_root if output_root is not None else PROJECT_ROOT / "outputs" / scene_id
    )
    return SceneLayout(Path(root).resolve())
