"""Tests for Phase-2 rendering defaults and canonical artifact paths."""

import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from video2scene.simulation_artifacts import (
    source_reference_frame,
    write_source_simulation_grid,
)
from video2scene.simulation_config import SimulationRunConfig


class SimulationRenderConfigTest(unittest.TestCase):
    def make_config(self, root: Path) -> SimulationRunConfig:
        return SimulationRunConfig(
            blender_dir=root / "blender",
            source_frames_dir=root / "frames",
            sim_dir=root / "sim",
        )

    def test_default_lighting_uses_calibrated_bright_approximation(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(Path(directory))
            self.assertEqual(config.light_model, "point")
            self.assertAlmostEqual(config.ambient_light, 0.85)
            self.assertAlmostEqual(config.key_light_scale, 0.35)
            self.assertAlmostEqual(config.fill_light_scale, 0.45)

    def test_default_coupling_band_limits_box_embedding(self):
        with tempfile.TemporaryDirectory() as directory:
            config = self.make_config(Path(directory))
            self.assertAlmostEqual(config.coupling_softness_m, 0.015)

    def test_source_reference_uses_only_canonical_frames_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config.blender_dir.mkdir()
            config.source_frames_dir.mkdir()
            (config.blender_dir / "scene_manifest.json").write_text(
                json.dumps({"source": {"reference_frame": 0}})
            )
            expected = config.source_frames_dir / "frame_0000.png"
            expected.write_bytes(b"frame")
            self.assertEqual(source_reference_frame(config), expected)

    def test_source_simulation_grid_pairs_available_frames(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config.blender_dir.mkdir()
            config.source_frames_dir.mkdir()
            config.sim_dir.mkdir()
            (config.blender_dir / "scene_manifest.json").write_text(
                json.dumps({"source": {"fps": 30, "reference_frame": 0}})
            )
            simulation_frames = []
            times = [0.0, 0.1, 0.2]
            for index, time_s in enumerate(times):
                source_index = round(time_s * 30)
                Image.new("RGB", (64, 36), (20 + index, 30, 60)).save(
                    config.source_frames_dir / f"frame_{source_index:04d}.png"
                )
                frame = config.sim_dir / f"frame_{index:04d}.png"
                Image.new("RGB", (64, 36), (60, 30, 20 + index)).save(frame)
                simulation_frames.append(frame)
            write_source_simulation_grid(config, simulation_frames, times)
            output = config.sim_dir / "source_simulation_grid.jpg"
            self.assertTrue(output.is_file())
            with Image.open(output) as image:
                self.assertEqual(image.height, 408)

    def test_legacy_main_directory_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = self.make_config(root)
            config.blender_dir.mkdir()
            legacy_dir = root / "main"
            legacy_dir.mkdir()
            (config.blender_dir / "scene_manifest.json").write_text(
                json.dumps({"source": {"reference_frame": 0}})
            )
            (legacy_dir / "frame_0000.png").write_bytes(b"frame")
            with self.assertRaisesRegex(FileNotFoundError, "canonical"):
                source_reference_frame(config)


if __name__ == "__main__":
    unittest.main()
