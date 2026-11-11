from dataclasses import replace
import json
import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

from quadruped_codesign.fusion import FusionShape, Stance
from quadruped_codesign.generator import Morphology, Payload
from quadruped_codesign.inspection import descendants
from quadruped_codesign.panel_model import Preview, load_morphology
from quadruped_codesign.posture import align_projection
from quadruped_codesign.statics import paired_vertical_loads, evaluate_statics


class StaticTests(unittest.TestCase):
    def test_pair_sharing_matches_hand_calculation_and_rejects_tension(self):
        points = {"FL": [1, .5, 0], "FR": [1, -.5, 0],
                  "RL": [-1, .5, 0], "RR": [-1, -.5, 0]}
        result = paired_vertical_loads(100, [.25, 0, 2], points)
        self.assertEqual(result["status"], "balanced")
        self.assertEqual(result["trial_forces_world_n"]["FL"], [0, 0, 31.25])
        self.assertEqual(result["trial_forces_world_n"]["RR"], [0, 0, 18.75])
        np.testing.assert_allclose(result["moment_residual_about_com_world_nm"], 0, atol=1e-12)
        edge = paired_vertical_loads(100, [1, 0, 2], points)
        self.assertEqual(edge["status"], "balanced")
        self.assertEqual(edge["trial_forces_world_n"]["RL"][2], 0)
        outside = paired_vertical_loads(100, [1.2, 0, 2], points)
        self.assertEqual(outside["status"], "unbalanced")
        self.assertLess(outside["trial_forces_world_n"]["RL"][2], 0)
        self.assertIn("pull", outside["reason"])
        self.assertEqual(paired_vertical_loads(100, [0, .05, 2], points)["status"], "outside_assumptions")
        collapsed = {k: [0, v[1], 0] for k, v in points.items()}
        self.assertEqual(paired_vertical_loads(100, [0, 0, 2], collapsed)["status"], "outside_assumptions")

    def test_joint_torques_match_independent_subtree_moment_sums_in_both_poses(self):
        upright, _ = align_projection(FusionShape(stance=Stance(mode="hind_stand")))
        for shape in (FusionShape(), upright):
            p = Preview.build(shape)
            result = p.report["statics"]
            self.assertTrue(result["equilibrium_within_tolerance"], result["reason"])
            self.assertTrue(all(result["checks"].values()))
            base = result["base_required_wrench"]
            rotation = p.data.body("torso").xmat.reshape(3, 3)
            np.testing.assert_allclose(base["force_world_n"], -np.asarray(result["force_residual_world_n"]), atol=1e-10)
            np.testing.assert_allclose(rotation @ base["moment_base_axes_nm"], -np.asarray(result["moment_residual_about_com_world_nm"]), atol=1e-10)
            self.assertEqual(len(result["joint_torques"]), 12)
            self.assertEqual(p.model.nu, 0)
            for row in result["joint_torques"]:
                joint = p.model.joint(row["joint"])
                members = descendants(p.model, int(joint.bodyid[0]))
                origin, axis = p.data.xanchor[joint.id], p.data.xaxis[joint.id]
                gravity_moment = np.zeros(3)
                contact_moment = np.zeros(3)
                for body in members:
                    gravity_moment += np.cross(p.data.xipos[body]-origin, p.model.body_mass[body]*p.model.opt.gravity)
                for leg, point in result["contacts_world_m"].items():
                    if int(p.model.site(leg+"_foot").bodyid[0]) in members:
                        contact_moment += np.cross(np.asarray(point)-origin, result["trial_forces_world_n"][leg])
                self.assertAlmostEqual(row["gravity_only_nm"], -axis @ gravity_moment, places=9)
                self.assertAlmostEqual(row["contact_generalized_nm"], axis @ contact_moment, places=9)
                self.assertAlmostEqual(row["required_joint_output_nm"], -axis @ (gravity_moment+contact_moment), places=9)
            for pair in ("F", "R"):
                rows = {r["joint"]: r["required_joint_output_nm"] for r in result["joint_torques"]}
                self.assertAlmostEqual(rows[pair+"L_roll"], -rows[pair+"R_roll"], places=9)
                self.assertAlmostEqual(rows[pair+"L_hip"], rows[pair+"R_hip"], places=9)
                self.assertAlmostEqual(rows[pair+"L_knee"], rows[pair+"R_knee"], places=9)

    def test_joint_loads_match_finite_difference_virtual_work(self):
        p = Preview.build(FusionShape())
        s = p.report["statics"]
        d = mujoco.MjData(p.model)
        def energy(qpos):
            d.qpos[:] = qpos
            mujoco.mj_forward(p.model, d)
            potential = -sum(p.model.body_mass[b]*(d.xipos[b] @ p.model.opt.gravity) for b in range(p.model.nbody))
            work = sum(np.asarray(force) @ d.site(leg+"_foot").xpos for leg, force in s["trial_forces_world_n"].items())
            return potential-work
        epsilon = 1e-6
        for row in s["joint_torques"]:
            dof = int(p.model.joint(row["joint"]).dofadr[0])
            direction = np.zeros(p.model.nv); direction[dof] = 1
            plus, minus = p.data.qpos.copy(), p.data.qpos.copy()
            mujoco.mj_integratePos(p.model, plus, direction, epsilon)
            mujoco.mj_integratePos(p.model, minus, direction, -epsilon)
            self.assertAlmostEqual((energy(plus)-energy(minus))/(2*epsilon), row["required_joint_output_nm"], places=6)

    def test_upright_requires_alignment_and_arm_edits_can_break_it(self):
        shape = FusionShape(stance=Stance(mode="hind_stand"))
        p = Preview.build(shape)
        s = p.report["statics"]
        self.assertTrue(p.report["passed"])
        self.assertEqual(s["status"], "unbalanced")
        self.assertEqual(s["joint_torques"], [])
        self.assertGreater(np.linalg.norm(s["moment_residual_about_com_world_nm"]), 9)
        aligned, _ = align_projection(shape)
        good = Preview.build(aligned).report["statics"]
        self.assertTrue(good["equilibrium_within_tolerance"])
        self.assertFalse(good["controlled_stability_verified"])
        self.assertEqual(set(good["trial_forces_world_n"]), {"RL", "RR"})
        self.assertAlmostEqual(good["trial_forces_world_n"]["RL"][2], good["weight_n"]/2)
        for row in good["joint_torques"]:
            if row["joint"].startswith("F"):
                self.assertEqual(row["contact_generalized_nm"], 0)
                self.assertEqual(row["required_joint_output_nm"], row["gravity_only_nm"])
        moved = replace(aligned, stance=replace(aligned.stance, front_hip=.8))
        bad = Preview.build(moved).report["statics"]
        self.assertEqual(bad["status"], "unbalanced")
        self.assertEqual(bad["joint_torques"], [])
        self.assertEqual(moved.stance.upright_hind_x, aligned.stance.upright_hind_x)
        realigned, _ = align_projection(moved)
        self.assertTrue(Preview.build(realigned).report["statics"]["equilibrium_within_tolerance"])

    def test_payload_and_geometry_changes_update_forces_and_torques(self):
        shape = FusionShape()
        a = Preview.build(shape).report["statics"]
        b = Preview.build(replace(shape, payload=Payload(mass=4))).report["statics"]
        self.assertAlmostEqual(b["weight_n"]-a["weight_n"], 2*9.81)
        for leg in a["trial_forces_world_n"]:
            self.assertAlmostEqual(b["trial_forces_world_n"][leg][2]-a["trial_forces_world_n"][leg][2], 2*9.81/4)
        forward = Preview.build(replace(shape, payload=Payload(2, position=(.1, 0, .12)))).report["statics"]
        self.assertGreater(forward["trial_forces_world_n"]["FL"][2], a["trial_forces_world_n"]["FL"][2])
        self.assertLess(forward["trial_forces_world_n"]["RL"][2], a["trial_forces_world_n"]["RL"][2])
        longer = Preview.build(replace(shape, l_thigh=.36)).report["statics"]
        self.assertTrue(longer["equilibrium_within_tolerance"])
        self.assertNotAlmostEqual(longer["largest_absolute_joint_torque"]["nm"], a["largest_absolute_joint_torque"]["nm"], places=3)

    def test_outside_baseline_is_not_a_claim_of_physical_impossibility(self):
        lateral = Preview.build(FusionShape(payload=Payload(2, position=(0, .03, .12))))
        s = lateral.report["statics"]
        self.assertEqual(s["status"], "outside_assumptions")
        self.assertIn("unequal left/right", s["reason"])
        self.assertEqual(s["joint_torques"], [])
        self.assertTrue(lateral.report["passed"])
        legacy = Preview.build(Morphology())
        self.assertEqual(legacy.report["statics"]["status"], "outside_assumptions")
        p = Preview.build(FusionShape())
        p.data.qpos[2] += .01
        self.assertIn("not on the ground", evaluate_statics(p.model, p.data, p.report["support"])["reason"])
        p.model.opt.gravity[0] = 1
        self.assertIn("gravity", evaluate_statics(p.model, p.data, p.report["support"])["reason"])

    def test_pose_only_evaluation_does_not_modify_state_or_use_solver_reactions(self):
        p = Preview.build(FusionShape())
        expected = p.report["statics"]
        p.data.qvel[:] = .2
        p.data.qfrc_constraint[:] = 1234
        p.data.qfrc_applied[:] = 999
        before = [a.copy() for a in (p.data.qpos, p.data.qvel, p.data.qfrc_constraint, p.data.qfrc_applied)]
        actual = evaluate_statics(p.model, p.data, p.report["support"])
        self.assertEqual(actual, expected)
        for array, saved in zip((p.data.qpos, p.data.qvel, p.data.qfrc_constraint, p.data.qfrc_applied), before):
            np.testing.assert_array_equal(array, saved)

    def test_saved_reports_keep_balanced_unbalanced_and_legacy_results(self):
        for shape in (FusionShape(), FusionShape(stance=Stance(mode="hind_stand")), Morphology()):
            p = Preview.build(shape)
            with tempfile.TemporaryDirectory() as folder:
                saved = p.save(Path(folder), Image.new("RGB", (8, 8)))
                result = json.loads((saved/"validation.json").read_text())["statics"]
                self.assertEqual(result, p.report["statics"])
                self.assertEqual(Preview.build(load_morphology(saved/"config.toml")).report["statics"], result)


@unittest.skipUnless(os.environ.get("QUADRUPED_UI_TESTS") == "1", "Requires active desktop")
class StaticPanelTests(unittest.TestCase):
    def test_static_page_updates_alignment_rollback_and_profile_switch(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        root = tk.Tk(); root.withdraw()
        with tempfile.TemporaryDirectory() as folder:
            p = ParameterPanel(root, FusionShape(), Path(folder))
            try:
                p.update_model()
                p.inspect_active.set(True); p.toggle_inspector()
                p.inspector.tabs.select(3)
                def text():
                    return p.inspector.texts["Static"].get("1.0", "end")
                self.assertIn("Within tolerance", text())
                self.assertIn("RL_knee:", text())
                p.entries["l_thigh"].set("999"); p.update_model()
                self.assertFalse(p.valid)
                self.assertIn("Within tolerance", text())  # last valid preview
                p.entries["l_thigh"].set("340"); p.update_model()
                p.mode_var.set("Hind-leg upright"); p.change_mode()
                self.assertTrue(p.valid, p.message.get())
                self.assertIn("Not balanced", text())
                self.assertNotIn("RL_knee:", text())
                self.assertIn("Withheld", text())
                p.save()
                result = json.loads((next(Path(folder).iterdir())/"validation.json").read_text())
                self.assertEqual(result["statics"]["status"], "unbalanced")
                p.align_com()
                self.assertIn("Within tolerance", text())
                p.entries["stance.front_hip"].set("45"); p.update_model()
                self.assertIn("Not balanced", text())
                p.apply_config(Morphology()); p.update_model()
                self.assertTrue(p.valid, p.message.get())
                self.assertIn("Outside baseline", text())
            finally:
                p.close()
