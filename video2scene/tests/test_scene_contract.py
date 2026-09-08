"""Guard the physical meaning of sampled reconstruction configurations."""

import copy
import unittest

from video2scene.spec import DEFAULT_SPEC, digest, read_json, sample, validate


class SceneContractTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.spec = read_json(DEFAULT_SPEC)

    def test_rejects_soft_cushion_replaced_by_rigid_box(self):
        spec = copy.deepcopy(self.spec)
        spec["entities"]["cushion_left"]["physics"]["type"] = "rigid"
        spec["entities"]["cushion_left"]["physics"]["collision"]["representation"] = (
            "box"
        )
        with self.assertRaisesRegex(ValueError, "deformable"):
            validate(spec)

    def test_rejects_initial_penetration(self):
        spec = copy.deepcopy(self.spec)
        spec["entities"]["ball"]["transform"]["position_m"][2] = 0.1
        with self.assertRaisesRegex(ValueError, "above"):
            validate(spec)

    def test_rejects_invalid_poisson_ratio(self):
        spec = copy.deepcopy(self.spec)
        spec["entities"]["cushion_left"]["physics"]["parameters"]["poisson_ratio"] = 0.5
        with self.assertRaisesRegex(ValueError, "Poisson"):
            validate(spec)

    def test_rejects_floating_soft_support(self):
        spec = copy.deepcopy(self.spec)
        spec["entities"]["cushion_left"]["transform"]["position_m"][2] += 0.01
        with self.assertRaisesRegex(ValueError, "rest on"):
            validate(spec)

    def test_seed_is_repeatable_without_mutating_nominal(self):
        before = digest(self.spec)
        self.assertEqual(digest(sample(self.spec, 17)), digest(sample(self.spec, 17)))
        self.assertNotEqual(
            digest(sample(self.spec, 17)), digest(sample(self.spec, 18))
        )
        self.assertEqual(digest(self.spec), before)

    def test_feasible_variants_keep_support_and_change_physics(self):
        for seed in range(24):
            spec = sample(self.spec, seed)
            validate(spec)
            pad = spec["entities"]["cushion_left"]
            self.assertAlmostEqual(
                pad["transform"]["position_m"][2],
                pad["geometry"]["dimensions_m"][2] / 2,
            )
            self.assertNotEqual(
                pad["physics"]["parameters"]["young_modulus_pa"],
                self.spec["entities"]["cushion_left"]["physics"]["parameters"][
                    "young_modulus_pa"
                ],
            )

    def test_impossible_randomization_fails_with_bounded_attempts(self):
        spec = copy.deepcopy(self.spec)
        spec["domain_randomization"]["parameters"]["ball.initial_z"]["range"] = [
            0.01,
            0.02,
        ]
        with self.assertRaisesRegex(ValueError, "No feasible sample after 3"):
            sample(spec, 0, max_attempts=3)


if __name__ == "__main__":
    unittest.main()
