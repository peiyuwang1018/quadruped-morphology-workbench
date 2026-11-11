"""Upright arm edits must change only the intended pose and derived CoM."""

from dataclasses import replace
import math
import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from PIL import Image

from quadruped_codesign.fusion import FusionShape, Stance, JointLimits, solve_pose
from quadruped_codesign import fusion_panel
from quadruped_codesign.panel_model import Preview, load_morphology, morphology_toml


class FrontPoseTests(unittest.TestCase):
    def setUp(self):
        self.shape = FusionShape(stance=Stance(mode="hind_stand"))

    def test_edits_move_mirrored_arms_and_com_without_moving_hind_feet(self):
        original = Preview.build(self.shape)
        for key, degrees in (("stance.front_roll", 10), ("stance.front_hip", 45),
                             ("stance.front_knee", -100)):
            with self.subTest(key=key):
                values = fusion_panel.panel_values(self.shape)
                values[key] = degrees
                p = Preview.build(fusion_panel.shape_from_panel(values, self.shape))
                self.assertEqual(replace(p.shape, stance=self.shape.stance), self.shape)
                np.testing.assert_array_equal(p.model.body_mass, original.model.body_mass)
                np.testing.assert_allclose(p.data.qpos[:7], original.data.qpos[:7], atol=1e-11)
                np.testing.assert_allclose(p.data.qpos[13:], original.data.qpos[13:], atol=1e-11)
                for leg in ("RL", "RR"):
                    np.testing.assert_allclose(p.data.site(leg+"_foot").xpos,
                                               original.data.site(leg+"_foot").xpos, atol=1e-11)
                left, right = p.data.site("FL_foot").xpos, p.data.site("FR_foot").xpos
                np.testing.assert_allclose(left, right*[1, -1, 1], atol=1e-10)
                self.assertGreater(np.linalg.norm(left-original.data.site("FL_foot").xpos), .01)
                com = np.asarray(p.report["support"]["com_world_m"])
                self.assertGreater(np.linalg.norm(com-original.report["support"]["com_world_m"]), 1e-5)
                np.testing.assert_allclose(com, p.data.subtree_com[p.model.body("torso").id], atol=1e-10)
                self.assertEqual(p.report["support"]["contact_legs"], ["RL", "RR"])
                self.assertFalse(p.report["support"]["static_stability_verified"])

    def test_saved_joint_limits_constrain_both_mirrored_roll_angles(self):
        shape = replace(self.shape, upright_limits=JointLimits(roll=(-.2, .4), hip=(-1.35, 2.8)))
        fields = fusion_panel.panel_fields(shape)
        np.testing.assert_allclose(fields["stance.front_roll"][2:4], np.degrees([-.2, .2]))
        for key, invalid in (("stance.front_roll", math.degrees(.3)),
                             ("stance.front_hip", 170), ("stance.front_knee", 0)):
            values = fusion_panel.panel_values(shape)
            values[key] = invalid
            with self.assertRaisesRegex(ValueError, "enter"):
                fusion_panel.shape_from_panel(values, shape)

    def test_in_range_arm_collision_is_rejected(self):
        values = fusion_panel.panel_values(self.shape)
        values.update({"stance.front_roll": -29, "stance.front_hip": 30, "stance.front_knee": -30})
        with self.assertRaisesRegex(ValueError, "Model parts overlap"):
            Preview.build(fusion_panel.shape_from_panel(values, self.shape))

    def test_degree_endpoint_roundoff_does_not_exceed_saved_joint_limits(self):
        for key in fusion_panel.FRONT_LIMB_SPECS:
            for edge in fusion_panel.panel_fields(self.shape)[key][2:4]:
                with self.subTest(key=key, edge=edge):
                    values = fusion_panel.panel_values(self.shape)
                    values[key] = f"{edge:.12g}"
                    shape = fusion_panel.shape_from_panel(values, self.shape)
                    solve_pose(shape)
                    low, high = fusion_panel.front_angle_limits(shape, key)
                    self.assertTrue(low <= fusion_panel.get_value(shape, key) <= high)

    def test_arm_settings_survive_quad_edits_and_saved_snapshots(self):
        values = fusion_panel.panel_values(self.shape)
        values.update({"stance.front_roll": 10, "stance.front_hip": 45, "stance.front_knee": -100})
        edited = fusion_panel.shape_from_panel(values, self.shape)
        quad = replace(edited, stance=replace(edited.stance, mode="quadruped"))
        self.assertFalse(set(fusion_panel.FRONT_LIMB_SPECS) & fusion_panel.panel_fields(quad).keys())
        quad_values = fusion_panel.panel_values(quad)
        quad_values["stance.hip_height"] = 510
        quad = fusion_panel.shape_from_panel(quad_values, quad)
        for key in fusion_panel.FRONT_LIMB_SPECS:
            self.assertEqual(fusion_panel.get_value(quad, key), fusion_panel.get_value(edited, key))
        self.assertTrue(Preview.build(quad).report["passed"])
        restored = replace(quad, stance=replace(quad.stance, mode="hind_stand"))
        p = Preview.build(restored)
        with tempfile.TemporaryDirectory() as directory:
            saved = p.save(Path(directory), Image.new("RGB", (8, 8)))
            loaded = load_morphology(saved / "config.toml")
            self.assertEqual(loaded, restored)
            np.testing.assert_array_equal(Preview.build(loaded).data.qpos, p.data.qpos)
            # Configurations from before upright arm fields existed still load.
            old = Path(directory) / "old.toml"
            old.write_text("\n".join(line for line in morphology_toml(self.shape).splitlines()
                                     if not line.startswith(('"front_roll"', '"front_hip"', '"front_knee"'))))
            self.assertEqual(load_morphology(old).stance, self.shape.stance)


@unittest.skipUnless(os.environ.get("QUADRUPED_UI_TESTS") == "1", "Requires active desktop")
class FrontPosePanelTests(unittest.TestCase):
    def test_live_edit_rollback_render_and_mode_roundtrip(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        root = tk.Tk()
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            p = ParameterPanel(root, FusionShape(stance=Stance(mode="hind_stand")), Path(directory))
            try:
                p.update_model()
                root.geometry("1000x720")
                p.tabs.select(1)
                root.deiconify()
                for index, keys in ((0, ("stance.upright_hip_height", "stance.upright_pitch", "stance.upright_hind_width", "stance.upright_hind_x")),
                                    (1, tuple(fusion_panel.FRONT_LIMB_SPECS))):
                    p.upright_tabs.select(index)
                    root.update_idletasks()
                    for key in keys:
                        slider = p.sliders[key]
                        self.assertGreaterEqual(slider.winfo_height(), slider.winfo_reqheight(), key)
                root.withdraw()
                before = p.preview
                for key, value in (("stance.front_roll", "10"), ("stance.front_hip", "45"),
                                   ("stance.front_knee", "-100")):
                    p.entries[key].set(value)
                p.update_model()
                self.assertTrue(p.valid, p.message.get())
                self.assertNotEqual(p.preview.report["support"]["com_world_m"], before.report["support"]["com_world_m"])
                edited = p.preview.shape
                image = np.asarray(p.pixels).copy()
                for _ in range(3):
                    p.cam.azimuth += 10; p.render_camera()
                    p.cam.azimuth -= 10; p.render_camera()
                    # Transparent CoM/packaging edges can differ by a few RGB
                    # levels after orbiting; large flips/stripes must still fail.
                    np.testing.assert_allclose(p.pixels, image, rtol=0, atol=4)
                previous = p.preview
                for values in ({"stance.front_hip": "170"},
                               {"stance.front_roll": "-29", "stance.front_hip": "30", "stance.front_knee": "-30"}):
                    for key, value in values.items():
                        p.entries[key].set(value)
                    p.update_model()
                    self.assertFalse(p.valid)
                    self.assertIs(p.preview, previous)
                    self.assertTrue(p.save_button.instate(["disabled"]))
                p._set_values(edited); p.update_model()
                p.mode_var.set("Four-foot stance"); p.change_mode()
                self.assertTrue(p.valid, p.message.get())
                self.assertNotIn("stance.front_roll", p.entries)
                p.mode_var.set("Hind-leg upright"); p.change_mode()
                self.assertTrue(p.valid, p.message.get())
                for key, value in (("stance.front_roll", 10), ("stance.front_hip", 45), ("stance.front_knee", -100)):
                    self.assertAlmostEqual(float(p.entries[key].get()), value)
                p.save()
                saved = next(Path(directory).iterdir())
                p.apply_config(load_morphology(saved / "config.toml")); p.update_model()
                self.assertTrue(p.valid, p.message.get())
                np.testing.assert_allclose(p.preview.data.qpos[7:13], previous.data.qpos[7:13], atol=1e-10)
            finally:
                p.close()
