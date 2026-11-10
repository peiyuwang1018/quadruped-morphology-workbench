import json
import math
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import mujoco
import numpy as np

from quadruped_codesign import Morphology, Payload, evaluate_robot, generate_robot
from quadruped_codesign.generator import HIP_Z, LEGS
from quadruped_codesign.mass_properties import inspect_mass
from quadruped_codesign.study import run_study

ROOT = Path(__file__).resolve().parents[1]


class PayloadAndFrontLegTests(unittest.TestCase):
    def test_default_matches_original_asset_physics(self):
        old = mujoco.MjModel.from_xml_path(str(ROOT / "assets/base_quadruped.xml"))
        new = generate_robot()
        for attribute in ("body_mass", "body_ipos", "body_inertia", "body_iquat", "body_pos",
                          "geom_size", "geom_pos", "jnt_range", "jnt_axis", "qpos0", "key_qpos",
                          "actuator_ctrlrange", "actuator_gear"):
            np.testing.assert_allclose(getattr(old, attribute), getattr(new, attribute), rtol=0, atol=1e-12)

    def test_front_parameters_are_independent_with_analytic_fk(self):
        base = generate_robot()
        model = generate_robot(front_thigh_length=0.26, front_shank_length=0.18)
        self.assertTrue(evaluate_robot(model)["passed"])
        data = mujoco.MjData(model)
        data.qpos[:3] = [0.1, -0.2, 1.0]
        for leg in LEGS:
            hip, knee = (0.3, -0.8) if leg.startswith("F") else (-0.2, -1.1)
            data.qpos[model.joint(f"{leg}_hip").qposadr[0]] = hip
            data.qpos[model.joint(f"{leg}_knee").qposadr[0]] = knee
        mujoco.mj_forward(model, data)
        for leg, (x, y) in LEGS.items():
            thigh, shank = (0.26, 0.18) if leg.startswith("F") else (0.22, 0.20)
            hip, knee = (0.3, -0.8) if leg.startswith("F") else (-0.2, -1.1)
            expected = [0.1+x-thigh*math.sin(hip)-shank*math.sin(hip+knee),
                        -0.2+y, 1+HIP_Z-thigh*math.cos(hip)-shank*math.cos(hip+knee)]
            np.testing.assert_allclose(data.site(f"{leg}_foot").xpos, expected, atol=1e-10)
        for leg in ("RL", "RR"):
            for part in ("thigh", "shank"):
                for attribute in ("mass", "ipos", "inertia", "pos"):
                    np.testing.assert_allclose(getattr(model.body(f"{leg}_{part}"), attribute),
                                               getattr(base.body(f"{leg}_{part}"), attribute), atol=1e-12)

    def test_payload_adds_no_dofs_and_moves_with_torso(self):
        cargo = Payload(mass=2, position=(0.12, 0, 0.12))
        model = generate_robot(payload=cargo)
        self.assertEqual((model.nq, model.nv, model.nu), (15, 14, 8))
        self.assertEqual(model.body_jntnum[model.body("payload").id], 0)
        data = mujoco.MjData(model)
        data.qpos[:7] = [0.1, 0.2, 1.0, math.cos(0.6), 0, math.sin(0.6), 0]
        mujoco.mj_forward(model, data)
        torso = data.body("torso")
        expected = torso.xpos + torso.xmat.reshape(3, 3) @ np.asarray(cargo.position)
        np.testing.assert_allclose(data.body("payload").xpos, expected, atol=1e-12)

    def test_payload_mass_com_and_box_inertia_match_analytic_model(self):
        empty = generate_robot()
        loaded = generate_robot(payload=Payload(mass=2, position=(0.12, 0, 0.12)))
        data0, data1 = mujoco.MjData(empty), mujoco.MjData(loaded)
        data0.qpos[:7] = data1.qpos[:7] = [0.1, -0.2, 1, math.cos(.3), 0, math.sin(.3), 0]
        mujoco.mj_forward(empty, data0)
        mujoco.mj_forward(loaded, data1)
        m0, m1 = inspect_mass(empty, data0), inspect_mass(loaded, data1)
        expected = (m0["total_mass_kg"] * np.array(m0["system_com_world_m"]) + 2 * data1.body("payload").xpos) / (m0["total_mass_kg"] + 2)
        np.testing.assert_allclose(m1["system_com_world_m"], expected, atol=1e-12)
        self.assertAlmostEqual(m1["total_mass_kg"] - m0["total_mass_kg"], 2)
        self.assertAlmostEqual(m1["bare_robot_mass_kg"], m0["total_mass_kg"])
        np.testing.assert_allclose(m1["bodies"]["payload"]["inertia_com_body_frame_kg_m2"],
                                   np.diag([2/12*(.16**2+.12**2), 2/12*(.20**2+.12**2), 2/12*(.20**2+.16**2)]), atol=1e-12)
        self.assertTrue(evaluate_robot(loaded)["passed"])

    def test_moving_payload_preserves_mass_and_moves_system_com(self):
        values = []
        for x in (0, .12):
            model = generate_robot(payload=Payload(mass=2, position=(x, 0, .12)))
            data = mujoco.MjData(model)
            mujoco.mj_forward(model, data)
            values.append(inspect_mass(model, data))
        self.assertEqual(values[0]["total_mass_kg"], values[1]["total_mass_kg"])
        shift = np.asarray(values[1]["system_com_world_m"]) - values[0]["system_com_world_m"]
        np.testing.assert_allclose(shift, [2*.12/values[0]["total_mass_kg"], 0, 0], atol=1e-12)

    def test_mass_and_com_errors_are_detected(self):
        for field, check in (("body_mass", "analytic_body_masses"), ("body_ipos", "analytic_body_centres_of_mass"),
                             ("body_inertia", "analytic_body_inertia_tensors")):
            model = generate_robot()
            getattr(model, field)[model.body("FL_thigh").id] *= 1.2
            report = evaluate_robot(model, steps=0)
            self.assertFalse(report["checks"][check])
            self.assertFalse(report["passed"])

    def test_invalid_inputs_and_payload_mount_rejected(self):
        for options in [{"mass":-1}, {"mass":True}, {"mass":float("nan")}, {"size":[0, .1, .1]},
                        {"size":[.1, .1]}, {"position":[0, 0, float("inf")]},
                        {"mass":2, "position":[0, 0, 0]}, {"mass":2, "position":[1, 0, .12]}]:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    Payload(**options)
        for options in [{"front_thigh_length":-1}, {"front_shank_length":float("nan")},
                        {"torso_mass":0}, {"torso_mass":True}, {"payload":{}}]:
            with self.assertRaises(ValueError):
                Morphology(**options)
        model = generate_robot(payload=Payload(mass=0))
        self.assertEqual(model.nbody, 10)

    def test_torso_mass_and_asymmetric_boundary_models(self):
        for front, hind in ((.10, .45), (.45, .10)):
            model = generate_robot(hind, hind, front_thigh_length=front, front_shank_length=front,
                                   torso_mass=10, payload=Payload(mass=2))
            self.assertAlmostEqual(model.body("torso").mass[0], 10)
            self.assertTrue(evaluate_robot(model)["passed"])

    def test_payload_toml_cli_and_study_report(self):
        with tempfile.TemporaryDirectory(prefix="payload study ") as folder:
            output = Path(folder) / "payload.xml"
            subprocess.run([sys.executable,"-m","quadruped_codesign.cli","generate","--config",
                            str(ROOT/"configs/payload.toml"),"--front-thigh","0.26","--output",str(output)],
                           check=True,capture_output=True,text=True)
            model = mujoco.MjModel.from_xml_path(str(output))
            self.assertAlmostEqual(model.body("payload").mass[0], 2)
            np.testing.assert_allclose(model.numeric("front_morphology").data, [.26, .20])
            run_study(Path(folder)/"study")
            report = json.loads((Path(folder)/"study/report.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(len(report["models"]), 4)


if __name__ == "__main__":
    unittest.main()
