"""Tests for deterministic Phase-3 batch planning and runtime resolution."""

import hashlib
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from video2scene.dataset_artifacts import batch_complete, remove_legacy_batch_artifacts
from video2scene.domain_randomization import (
    PER_BATCH,
    PER_ENV,
    derive_seed,
    parameter_execution,
    plan_batch,
    pointer_value,
    resolve_runtime_entities,
    serializable_batch_plan,
)
from video2scene.spec import DEFAULT_SPEC, read_json, validate


class DomainRandomizationPlanTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(DEFAULT_SPEC)
        ball = cls.spec["entities"]["ball"]
        radius = ball["geometry"]["radius_m"]
        volume = 4.0 * np.pi * radius**3 / 3.0
        cls.manifest = {
            "entities": cls.spec["entities"],
            "resolved_entities": {
                "ball": {
                    "physics_type": "rigid",
                    "position_m": ball["transform"]["position_m"],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                    "mass_kg": volume * ball["physics"]["parameters"]["density_kg_m3"],
                    "rest_bounds_local_m": [
                        [-radius, -radius, -radius],
                        [radius, radius, radius],
                    ],
                    "collision": {"representation": "sphere", "radius_m": radius},
                    "physics_parameters": ball["physics"]["parameters"],
                    "appearance": ball["appearance"],
                    "initial_state": ball["initial_state"],
                }
            },
        }

    def test_parameter_partition_covers_every_declared_range(self):
        execution = parameter_execution(self.spec)
        self.assertEqual(
            set(execution), set(self.spec["domain_randomization"]["parameters"])
        )
        self.assertEqual(
            sum(item["lane"] == PER_ENV for item in execution.values()), 10
        )
        self.assertEqual(
            sum(item["lane"] == PER_BATCH for item in execution.values()), 38
        )
        self.assertEqual(execution["ball.radius"]["lane"], PER_ENV)
        self.assertEqual(execution["cushion_left.young_modulus_pa"]["lane"], PER_BATCH)
        self.assertEqual(execution["camera.lens"]["lane"], PER_BATCH)

    def test_seed_derivation_and_batch_plan_are_repeatable(self):
        first = plan_batch(
            self.spec, root_seed=123, batch_index=2, num_envs=4, sample_start=8
        )
        second = plan_batch(
            self.spec, root_seed=123, batch_index=2, num_envs=4, sample_start=8
        )
        self.assertEqual(
            serializable_batch_plan(first), serializable_batch_plan(second)
        )
        self.assertNotEqual(derive_seed(123, "batch", 1), derive_seed(123, "batch", 2))
        radii = [
            item.spec["entities"]["ball"]["geometry"]["radius_m"]
            for item in first.samples
        ]
        self.assertGreater(len(set(radii)), 1)
        soft_moduli = [
            item.spec["entities"]["cushion_left"]["physics"]["parameters"][
                "young_modulus_pa"
            ]
            for item in first.samples
        ]
        self.assertEqual(len(set(soft_moduli)), 1)
        for item in first.samples:
            validate(item.spec, check_schema=False)

    def test_reuse_mode_preserves_batch_values_and_varies_runtime_values(self):
        anchor = self.spec
        plan = plan_batch(
            self.spec,
            root_seed=9,
            batch_index=0,
            num_envs=3,
            sample_start=0,
            batch_spec=anchor,
        )
        execution = parameter_execution(self.spec)
        for item in plan.samples:
            for name, metadata in execution.items():
                if metadata["lane"] == PER_BATCH:
                    self.assertEqual(
                        pointer_value(item.spec, metadata["path"]),
                        pointer_value(anchor, metadata["path"]),
                    )

    def test_runtime_resolution_scales_geometry_and_mass(self):
        anchor = self.spec
        plan = plan_batch(
            self.spec,
            root_seed=77,
            batch_index=0,
            num_envs=2,
            sample_start=0,
            batch_spec=anchor,
        )
        resolved = resolve_runtime_entities(
            self.manifest, [item.spec for item in plan.samples]
        )
        for env_index, item in enumerate(plan.samples):
            ball = resolved[env_index]["ball"]
            authored = item.spec["entities"]["ball"]
            radius_ratio = (
                authored["geometry"]["radius_m"]
                / anchor["entities"]["ball"]["geometry"]["radius_m"]
            )
            density_ratio = (
                authored["physics"]["parameters"]["density_kg_m3"]
                / anchor["entities"]["ball"]["physics"]["parameters"]["density_kg_m3"]
            )
            self.assertTrue(np.allclose(ball["visual_scale_xyz"], [radius_ratio] * 3))
            self.assertAlmostEqual(
                ball["mass_kg"],
                self.manifest["resolved_entities"]["ball"]["mass_kg"]
                * radius_ratio**3
                * density_ratio,
            )
            self.assertAlmostEqual(
                ball["collision"]["radius_m"], authored["geometry"]["radius_m"]
            )
            expected_coupling_friction = np.sqrt(
                authored["physics"]["parameters"]["friction"]
                * item.spec["entities"]["cushion_left"]["physics"]["parameters"][
                    "friction"
                ]
            )
            self.assertAlmostEqual(
                ball["physics_parameters"]["coupling_friction"],
                expected_coupling_friction,
            )


class DatasetOutputContractTest(unittest.TestCase):
    def test_success_marker_must_match_identity_and_sample_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            sim_dir = batch_dir / "sim"
            sim_dir.mkdir()
            for name in ("validation.json", "trajectories.npz"):
                (sim_dir / name).write_bytes(b"artifact")
            run_manifest = sim_dir / "run_manifest.json"
            run_manifest.write_text(json.dumps({"sample_ids": [4, 5]}))
            (batch_dir / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "run_identity_sha256": "expected",
                        "run_manifest_sha256": hashlib.sha256(
                            run_manifest.read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            self.assertTrue(
                batch_complete(
                    batch_dir,
                    expected_identity_sha256="expected",
                    expected_sample_ids=[4, 5],
                )
            )
            self.assertFalse(
                batch_complete(
                    batch_dir,
                    expected_identity_sha256="different",
                    expected_sample_ids=[4, 5],
                )
            )
            self.assertFalse(
                batch_complete(
                    batch_dir,
                    expected_identity_sha256="expected",
                    expected_sample_ids=[5, 6],
                )
            )

    def test_legacy_genesis_directory_is_not_accepted(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            legacy_dir = batch_dir / "genesis"
            legacy_dir.mkdir()
            for name in ("validation.json", "trajectories.npz"):
                (legacy_dir / name).write_bytes(b"artifact")
            run_manifest = legacy_dir / "run_manifest.json"
            run_manifest.write_text(json.dumps({"sample_ids": [0]}))
            (batch_dir / "_SUCCESS").write_text(
                json.dumps(
                    {
                        "run_identity_sha256": "expected",
                        "run_manifest_sha256": hashlib.sha256(
                            run_manifest.read_bytes()
                        ).hexdigest(),
                    }
                )
            )
            self.assertFalse(
                batch_complete(
                    batch_dir,
                    expected_identity_sha256="expected",
                    expected_sample_ids=[0],
                )
            )

    def test_legacy_batch_artifacts_are_removed_before_regeneration(self):
        with tempfile.TemporaryDirectory() as directory:
            batch_dir = Path(directory)
            for path in (
                batch_dir / "genesis" / "trace.json",
                batch_dir / "blender" / "scene.blend",
                batch_dir / "blender_build.log",
            ):
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("legacy")
            remove_legacy_batch_artifacts(batch_dir)
            self.assertFalse((batch_dir / "genesis").exists())
            self.assertFalse((batch_dir / "blender").exists())
            self.assertFalse((batch_dir / "blender_build.log").exists())


if __name__ == "__main__":
    unittest.main()
