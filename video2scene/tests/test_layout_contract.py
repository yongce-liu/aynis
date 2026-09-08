"""Tests for the canonical scene output directory contract."""

import tempfile
import unittest
from pathlib import Path

from video2scene.generate_dataset import DatasetConfig
from video2scene.layout import SceneLayout, resolve_scene_layout
from video2scene.simulation_config import (
    SimulationCommandConfig,
    resolve_simulation_command,
)


class SceneLayoutTest(unittest.TestCase):
    def test_prepare_creates_the_canonical_top_level_structure(self):
        with tempfile.TemporaryDirectory() as directory:
            layout = SceneLayout(Path(directory) / "scene")
            layout.prepare()
            self.assertEqual(
                {path.name for path in layout.root.iterdir()},
                {"frames", "blender", "sim", "batches"},
            )

    def test_variants_and_batches_stay_under_their_stage_directories(self):
        layout = SceneLayout(Path("outputs/example"))
        self.assertEqual(
            layout.blender_variant(7),
            Path("outputs/example/blender/variants/seed_000007"),
        )
        self.assertEqual(
            layout.batch(3).sim,
            Path("outputs/example/batches/batch_000003/sim"),
        )

    def test_default_scene_root_is_derived_from_scene_id(self):
        layout = resolve_scene_layout("another-scene")
        self.assertEqual(layout.root.name, "another-scene")
        self.assertEqual(layout.root.parent.name, "outputs")

    def test_phase_commands_resolve_paths_from_one_root(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "scene"
            simulation = resolve_simulation_command(
                SimulationCommandConfig(output_root=root)
            )
            dataset = DatasetConfig(output_root=root)
            self.assertEqual(simulation.blender_dir, root / "blender")
            self.assertEqual(simulation.source_frames_dir, root / "frames")
            self.assertEqual(simulation.sim_dir, root / "sim")
            self.assertEqual(dataset.batches_dir, root / "batches")


if __name__ == "__main__":
    unittest.main()
