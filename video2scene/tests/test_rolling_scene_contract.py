"""Regression tests for the rigid wood-to-glass rolling scene contract."""

import copy
import math
import unittest
from pathlib import Path

import numpy as np

from video2scene.simulation_geometry import horizontal_half_extent, world_bounds
from video2scene.layout import PROJECT_ROOT
from video2scene.spec import read_json, sample, validate


ROLLING_SPEC = PROJECT_ROOT / "scenes/ball-rolls-on-glass.json"


class RollingSceneContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(ROLLING_SPEC)

    def test_nominal_rigid_rolling_scene_validates(self):
        validate(self.spec)
        rule = self.spec["physics_validation"]["checks"][0]
        self.assertEqual(rule["type"], "rolling_transition")
        self.assertEqual(rule["entity"], "rolling_sphere")
        self.assertEqual(rule["target_support"], "glass_plate")

    def test_initial_velocity_and_spin_encode_forward_rolling(self):
        sphere = self.spec["entities"]["rolling_sphere"]
        radius = sphere["geometry"]["radius_m"]
        velocity = sphere["initial_state"]["linear_velocity_m_s"][0]
        omega = sphere["initial_state"]["angular_velocity_rad_s"][1]
        self.assertAlmostEqual(velocity, radius * omega, delta=0.01)

    def test_rolling_scene_rejects_a_floating_sphere(self):
        spec = copy.deepcopy(self.spec)
        spec["entities"]["rolling_sphere"]["transform"]["position_m"][2] += 0.03
        with self.assertRaisesRegex(ValueError, "rest on"):
            validate(spec)

    def test_sampling_preserves_the_rolling_velocity_constraint(self):
        sampled = sample(self.spec, 37)
        sphere = sampled["entities"]["rolling_sphere"]
        velocity = sphere["initial_state"]["linear_velocity_m_s"][0]
        omega = sphere["initial_state"]["angular_velocity_rad_s"][1]
        self.assertAlmostEqual(velocity, sphere["geometry"]["radius_m"] * omega)

    def test_exported_bounds_support_rotated_boxes(self):
        half_angle = math.radians(45.0) / 2.0
        entity = {
            "rest_bounds_local_m": [[-1.0, -0.5, -0.25], [1.0, 0.5, 0.25]],
            "position_m": [2.0, 3.0, 4.0],
            "quaternion_wxyz": [math.cos(half_angle), 0.0, 0.0, math.sin(half_angle)],
            "collision": {"representation": "box", "dimensions_m": [2.0, 1.0, 0.5]},
        }
        bounds = world_bounds(entity)
        extent = bounds[1] - bounds[0]
        expected_extent = np.repeat(3.0 / math.sqrt(2.0), 2)
        self.assertTrue(np.allclose(extent[:2], expected_extent, atol=1e-7))
        self.assertTrue(
            np.allclose(horizontal_half_extent(entity), expected_extent / 2.0)
        )

    def test_spec_path_is_repository_local(self):
        self.assertEqual(
            ROLLING_SPEC, Path(PROJECT_ROOT) / "scenes/ball-rolls-on-glass.json"
        )


if __name__ == "__main__":
    unittest.main()


class RollingValidationRuleTest(unittest.TestCase):
    def test_rule_validator_accepts_supported_forward_rolling(self):
        from types import SimpleNamespace

        from video2scene.simulate import _validate_rolling_transition

        manifest = {
            "source": {"fps": 30},
            "semantics": {
                "events": [
                    {
                        "name": "sphere_center_enters_glass",
                        "source_frame_interval": [3, 5],
                    }
                ]
            },
            "physics_validation": {"event_tolerance_frames": 2.0},
            "resolved_entities": {
                "rolling_sphere": {
                    "collision": {"representation": "sphere", "radius_m": 0.035},
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "wood_table": {
                    "rest_bounds_local_m": [
                        [-0.36, -0.38, -0.21],
                        [0.36, 0.38, 0.21],
                    ],
                    "position_m": [-0.36, -0.055, -0.21],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
                "glass_plate": {
                    "rest_bounds_local_m": [
                        [-0.31, -0.275, -0.004],
                        [0.31, 0.275, 0.004],
                    ],
                    "position_m": [0.305, -0.03, -0.004],
                    "quaternion_wxyz": [1.0, 0.0, 0.0, 0.0],
                },
            },
        }
        rule = {
            "id": "wood_to_glass_roll",
            "type": "rolling_transition",
            "entity": "rolling_sphere",
            "supports": ["wood_table", "glass_plate"],
            "target_support": "glass_plate",
            "event": "sphere_center_enters_glass",
            "motion_axis": 0,
            "direction": "positive",
            "angular_axis": 1,
            "rolling_sign": 1.0,
            "support_gap_range_m": [-0.004, 0.012],
            "minimum_supported_fraction": 1.0,
            "minimum_displacement_m": 0.3,
            "minimum_direction_fraction": 0.95,
            "maximum_lateral_drift_m": 0.025,
            "maximum_vertical_excursion_m": 0.01,
            "minimum_rolling_speed_m_s": 0.1,
            "maximum_median_slip_ratio": 0.2,
            "minimum_final_speed_ratio": 0.5,
            "maximum_speed_gain_ratio": 1.2,
            "require_final_on_target": True,
        }
        times = [0.0, 4.0 / 30.0, 0.5]
        positions = [
            [-0.135, -0.005, 0.035],
            [0.01, -0.005, 0.035],
            [0.35, -0.005, 0.035],
        ]
        velocities = [[1.0, 0.0, 0.0], [0.98, 0.0, 0.0], [0.9, 0.0, 0.0]]
        records = []
        for time_s, position, velocity in zip(
            times, positions, velocities, strict=True
        ):
            records.append(
                {
                    "time_s": time_s,
                    "rigid": {
                        "rolling_sphere": {
                            "position_m": position,
                            "linear_velocity_m_s": velocity,
                            "angular_velocity_rad_s": [0.0, velocity[0] / 0.035, 0.0],
                            "bottom_z_m": 0.0,
                        }
                    },
                }
            )
        checks: dict[str, bool] = {}
        physics_keys: list[str] = []
        metrics: dict[str, object] = {}
        _validate_rolling_transition(
            SimpleNamespace(manifest=manifest),
            records,
            rule,
            checks,
            physics_keys,
            metrics,
        )
        self.assertTrue(physics_keys)
        self.assertTrue(all(checks[name] for name in physics_keys), checks)
        transition = metrics["rolling_transitions"]["wood_to_glass_roll"]
        self.assertAlmostEqual(transition["detected_entry_time_s"], 4.0 / 30.0)
