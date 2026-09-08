"""Tests for solver selection and migration invariants."""

import copy
import unittest

from video2scene.genesis_plan import (
    MigrationError,
    choose_solver,
    rayleigh_damping,
    vertical_fov_deg,
)


class GenesisPlanTest(unittest.TestCase):
    def setUp(self):
        self.soft = {
            "physics_type": "soft",
            "collision": {"representation": "deformable_surface"},
            "physics_parameters": {
                "young_modulus_pa": 18000,
                "poisson_ratio": 0.22,
                "density_kg_m3": 65,
                "friction": 0.55,
                "damping_ratio": 0.25,
            },
            "deformable": {"representation": "closed_rest_surface"},
            "rest_bounds_local_m": [[-0.25, -0.35, -0.09], [0.25, 0.35, 0.09]],
        }

    def test_soft_volume_selects_fem_without_surface_decimation(self):
        decision = choose_solver("cushion", self.soft)
        self.assertEqual(decision["solver"], "FEMSolver")
        self.assertFalse(decision["discretization"]["surface_decimation"])

    def test_soft_volume_cannot_silently_become_rigid(self):
        entity = copy.deepcopy(self.soft)
        entity["collision"]["representation"] = "box"
        with self.assertRaisesRegex(MigrationError, "refusing"):
            choose_solver("cushion", entity)

    def test_unimplemented_nonrigid_class_fails_loudly(self):
        entity = copy.deepcopy(self.soft)
        entity["physics_type"] = "fluid"
        entity["collision"]["representation"] = "fluid_volume"
        with self.assertRaisesRegex(MigrationError, "dedicated"):
            choose_solver("liquid", entity)

    def test_rayleigh_mapping_matches_requested_ratio(self):
        damping = rayleigh_damping({"cushion": self.soft})
        omega = damping["reference_angular_frequency_rad_s"]
        ratio = damping["alpha"] / (2 * omega) + damping["beta"] * omega / 2
        self.assertAlmostEqual(ratio, 0.25)

    def test_camera_conversion_uses_vertical_fov(self):
        camera = {"resolution_px": [640, 360], "sensor_width_mm": 36, "lens_mm": 40}
        self.assertAlmostEqual(vertical_fov_deg(camera), 28.409, places=3)


if __name__ == "__main__":
    unittest.main()
