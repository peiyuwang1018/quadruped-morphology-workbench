import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from quadruped_codesign import Morphology, evaluate_robot, generate_mjcf, generate_robot, write_robot
from quadruped_codesign.generator import (
    FOOT_MASS, HIP_Z, LEGS, LINK_DENSITY, SHANK_RADIUS, THIGH_RADIUS,
)


class GeneratorTests(unittest.TestCase):
    def test_invalid_lengths_rejected_before_compilation(self):
        for invalid in [0, -0.2, 0.099, 0.451, float("nan"), float("inf"), "0.2", True, None]:
            for parameter in ["thigh_length", "shank_length"]:
                with self.subTest(parameter=parameter, value=invalid):
                    with self.assertRaises(ValueError):
                        Morphology(**{parameter: invalid})

    def test_forward_kinematics_matches_independent_two_link_formula(self):
        for thigh, shank in [(0.1, 0.45), (0.22, 0.20), (0.45, 0.1), (0.30, 0.28)]:
            model = generate_robot(thigh, shank)
            data = mujoco.MjData(model)
            for hip, knee in [(0, 0), (0.45, -0.9), (-0.7, -1.4), (1.1, 0.1)]:
                for leg in LEGS:
                    data.qpos[model.joint(f"{leg}_hip").qposadr[0]] = hip
                    data.qpos[model.joint(f"{leg}_knee").qposadr[0]] = knee
                mujoco.mj_forward(model, data)
                for leg, (x, y) in LEGS.items():
                    expected = [x - thigh * math.sin(hip) - shank * math.sin(hip + knee), y,
                                model.qpos0[2] + HIP_Z - thigh * math.cos(hip) - shank * math.cos(hip + knee)]
                    with self.subTest(lengths=(thigh, shank), angles=(hip, knee), leg=leg):
                        np.testing.assert_allclose(data.site(f"{leg}_foot").xpos, expected, atol=1e-10)

    def test_masses_match_capsule_volume_and_fixed_foot_mass(self):
        for thigh, shank in [(0.22, 0.20), (0.30, 0.28)]:
            model = generate_robot(thigh, shank)
            upper_mass = LINK_DENSITY * (math.pi * THIGH_RADIUS**2 * thigh + 4 / 3 * math.pi * THIGH_RADIUS**3)
            lower_mass = LINK_DENSITY * (math.pi * SHANK_RADIUS**2 * shank + 4 / 3 * math.pi * SHANK_RADIUS**3) + FOOT_MASS
            for leg in LEGS:
                self.assertAlmostEqual(model.body(f"{leg}_thigh").mass[0], upper_mass, places=10)
                self.assertAlmostEqual(model.body(f"{leg}_shank").mass[0], lower_mass, places=10)
            self.assertAlmostEqual(model.body_mass.sum(), 8 + 4 * (upper_mass + lower_mass), places=10)

    def test_independent_parameters_propagate_geometry_and_inertia(self):
        base = generate_robot(0.22, 0.20)
        longer_thigh = generate_robot(0.30, 0.20)
        longer_shank = generate_robot(0.22, 0.28)
        for leg in LEGS:
            upper, lower = f"{leg}_thigh", f"{leg}_shank"
            np.testing.assert_allclose(longer_thigh.body(lower).pos, [0, 0, -0.30])
            np.testing.assert_allclose(longer_shank.site(f"{leg}_foot").pos, [0, 0, -0.28])
            np.testing.assert_allclose(longer_thigh.body(lower).inertia, base.body(lower).inertia)
            np.testing.assert_allclose(longer_shank.body(upper).inertia, base.body(upper).inertia)
            self.assertTrue(np.all(longer_thigh.body(upper).inertia > base.body(upper).inertia))
            self.assertTrue(np.all(longer_shank.body(lower).inertia > base.body(lower).inertia))
            self.assertAlmostEqual(longer_thigh.geom(f"{leg}_thigh_geom").size[1], 0.30 / 2)
            self.assertAlmostEqual(longer_shank.geom(f"{leg}_shank_geom").size[1], 0.28 / 2)

    def test_boundary_models_compile_and_survive_short_uncontrolled_rollout(self):
        for thigh in [0.10, 0.22, 0.45]:
            for shank in [0.10, 0.20, 0.45]:
                with self.subTest(thigh=thigh, shank=shank):
                    report = evaluate_robot(generate_robot(thigh, shank))
                    self.assertTrue(report["passed"], report)
                    self.assertAlmostEqual(report["unactuated_rollout_seconds"], 1.0)

    def test_deterministic_standalone_file_and_invalid_write_preserves_output(self):
        self.assertEqual(generate_mjcf(0.22, 0.2), generate_mjcf(0.22, 0.2))
        with tempfile.TemporaryDirectory(prefix="robot with spaces ") as directory:
            target = Path(directory) / "model.xml"
            write_robot(target)
            self.assertEqual(mujoco.MjModel.from_xml_path(str(target)).nu, 8)
            original = target.read_text()
            with self.assertRaises(ValueError):
                write_robot(target, -1, 0.2)
            self.assertEqual(target.read_text(), original)

    def test_toml_config_and_cli_override(self):
        with tempfile.TemporaryDirectory(prefix="robot config ") as directory:
            config = Path(directory) / "morphology.toml"
            output = Path(directory) / "robot.xml"
            config.write_text("thigh_length = 0.25\nshank_length = 0.18\n")
            command = [sys.executable, "-m", "quadruped_codesign.cli", "generate", "--config", str(config),
                       "--thigh", "0.30", "--output", str(output)]
            subprocess.run(command, check=True, capture_output=True, text=True)
            model = mujoco.MjModel.from_xml_path(str(output))
            np.testing.assert_allclose(model.numeric("morphology").data, [0.30, 0.18])
            result = subprocess.run(command + ["--shank", "nan"], capture_output=True, text=True)
            self.assertNotEqual(result.returncode, 0)


if __name__ == "__main__":
    unittest.main()
