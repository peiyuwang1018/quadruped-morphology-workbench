from dataclasses import replace
import copy
import json
import math
import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

from quadruped_codesign.fusion import FusionShape, Stance, UPPER_RADIUS
from quadruped_codesign.generator import Morphology, Payload
from quadruped_codesign.inspection import aggregate, descendants, add_selection_overlay
from quadruped_codesign.panel_model import Preview, load_morphology
from quadruped_codesign.visualization import camera


class InspectionTests(unittest.TestCase):
    def test_subtrees_match_mujoco_mass_and_com_in_both_poses(self):
        for mode in ("quadruped", "hind_stand"):
            p = Preview.build(FusionShape(stance=Stance(mode=mode)))
            for name, item in p.report["inspection"]["bodies"].items():
                body = p.model.body(name)
                direct, group = item["body_only"], item["subtree"]
                self.assertAlmostEqual(direct["mass_kg"], body.mass[0])
                self.assertAlmostEqual(sum(item["body_mass_breakdown_kg"].values()), direct["mass_kg"])
                np.testing.assert_allclose(direct["com_selected_body_frame_m"], body.ipos, atol=1e-12)
                np.testing.assert_allclose(direct["com_world_m"], p.data.xipos[body.id], atol=1e-12)
                self.assertAlmostEqual(group["mass_kg"], p.model.body_subtreemass[body.id])
                np.testing.assert_allclose(group["com_world_m"], p.data.subtree_com[body.id], atol=1e-12)
            pair = p.report["inspection"]["bodies"]["RL_carrier"]
            self.assertEqual(pair["body_only"]["mass_kg"], 2.8)
            self.assertEqual(pair["subtree"]["body_names"], ["RL_carrier", "RL_thigh", "RL_shank"])
            self.assertIn("sibling", pair["source"])

    def test_parallel_axis_inertia_with_rotated_body_and_inertial_frames(self):
        q = math.sqrt(.5)
        xml = f'''<mujoco><worldbody><body name="root" pos="2 3 4" quat="{q} 0 0 {q}">
          <inertial mass="2" pos="0 0 0" diaginertia="1 2 3"/>
          <body name="child" pos="1 0 0" quat="{q} 0 0 {q}">
            <inertial mass="3" pos="0 0 0" diaginertia="4 5 6"/>
          </body></body></worldbody></mujoco>'''
        model = mujoco.MjModel.from_xml_string(xml)
        data = mujoco.MjData(model); mujoco.mj_forward(model, data)
        result = aggregate(model, data, model.body("root").id, descendants(model, model.body("root").id))
        self.assertEqual(result["mass_kg"], 5)
        np.testing.assert_allclose(result["com_selected_body_frame_m"], [.6, 0, 0], atol=1e-12)
        np.testing.assert_allclose(result["com_world_m"], [2, 3.6, 4], atol=1e-12)
        # Child I rotates to diag(5,4,6). Shifts add 1.2 about y,z.
        np.testing.assert_allclose(result["inertia_com_selected_body_axes_kg_m2"], np.diag([6, 7.2, 10.2]), atol=1e-12)

    def test_length_change_updates_body_properties_and_derived_geometry(self):
        a = Preview.build(FusionShape())
        b = Preview.build(replace(a.shape, l_thigh=.36))
        a_body = a.report["inspection"]["bodies"]["RL_thigh"]["body_only"]
        b_body = b.report["inspection"]["bodies"]["RL_thigh"]["body_only"]
        self.assertAlmostEqual(b_body["mass_kg"]-a_body["mass_kg"], a.shape.link_density*math.pi*UPPER_RADIUS**2*.02)
        np.testing.assert_allclose(b_body["com_selected_body_frame_m"], [0,0,-.18], atol=1e-12)
        self.assertGreater(b_body["inertia_com_selected_body_axes_kg_m2"][0][0], a_body["inertia_com_selected_body_axes_kg_m2"][0][0])
        self.assertAlmostEqual(b.report["inspection"]["legs"]["RL"]["upper_length_m"], .36)
        self.assertAlmostEqual(b.report["inspection"]["foot_spacing_y_m"]["R"], .55)
        for leg in b.report["inspection"]["legs"].values():
            self.assertAlmostEqual(leg["foot_surface_height_m"], 0, places=9)
        pair_gaps = [g for g in b.report["inspection"]["motor_envelope_gaps"] if g["label"].endswith("Pitch pair")]
        np.testing.assert_allclose([g["distance_m"] for g in pair_gaps], .005, atol=1e-10)

    def test_pose_changes_world_properties_but_not_body_mass_or_local_inertia(self):
        a = Preview.build(FusionShape(stance=Stance(mode="hind_stand")))
        b = Preview.build(replace(a.shape, stance=replace(a.shape.stance, front_hip=.8)))
        x = a.report["inspection"]["bodies"]["FL_thigh"]["body_only"]
        y = b.report["inspection"]["bodies"]["FL_thigh"]["body_only"]
        self.assertEqual(x["mass_kg"], y["mass_kg"])
        np.testing.assert_allclose(x["inertia_com_selected_body_axes_kg_m2"], y["inertia_com_selected_body_axes_kg_m2"], atol=1e-12)
        self.assertGreater(np.linalg.norm(np.subtract(x["com_world_m"], y["com_world_m"])), .01)
        joint = next(j for j in b.report["inspection"]["joints"] if j["name"] == "FL_hip")
        self.assertAlmostEqual(joint["angle_rad"], .8)
        self.assertAlmostEqual(joint["limit_margin_rad"], 2.)

    def test_highlight_is_scene_only_and_does_not_accumulate(self):
        p = Preview.build(FusionShape())
        scene = mujoco.MjvScene(p.model, maxgeom=200)
        mass, rgba, qpos, report = p.model.body_mass.copy(), p.model.geom_rgba.copy(), p.data.qpos.copy(), copy.deepcopy(p.report)
        for _ in range(3):
            mujoco.mjv_updateScene(p.model,p.data,mujoco.MjvOption(),None,camera(),mujoco.mjtCatBit.mjCAT_ALL,scene)
            original_count = scene.ngeom
            add_selection_overlay(scene,p.model,p.report,"RL_carrier",True)
            self.assertEqual(scene.ngeom, original_count+1)
            self.assertEqual(scene.geoms[scene.ngeom-1].label, "Selected CoM")
            for geom in scene.geoms[:original_count]:
                if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM:
                    name = p.model.body(int(p.model.geom_bodyid[geom.objid])).name
                    if name in ("RL_carrier","RL_thigh","RL_shank"):
                        np.testing.assert_allclose(geom.rgba[:3], [1,.3,.08], atol=1e-7)
                        self.assertEqual(geom.rgba[3], rgba[geom.objid,3])
        np.testing.assert_array_equal(p.model.body_mass, mass)
        np.testing.assert_array_equal(p.model.geom_rgba, rgba)
        np.testing.assert_array_equal(p.data.qpos, qpos)
        self.assertEqual(p.report, report)

    def test_snapshot_export_and_legacy_body_accounting(self):
        for shape in (FusionShape(), Morphology(payload=Payload(mass=2))):
            p = Preview.build(shape)
            lower = p.report["inspection"]["bodies"]["RL_shank"]
            self.assertIn("foot", lower["label"])
            self.assertAlmostEqual(lower["body_mass_breakdown_kg"]["Foot"], .08)
            with tempfile.TemporaryDirectory() as folder:
                saved = p.save(Path(folder), Image.new("RGB",(8,8)))
                report = json.loads((saved/"validation.json").read_text())
                self.assertEqual(report["inspection"], p.report["inspection"])
                rebuilt = Preview.build(load_morphology(saved/"config.toml"))
                self.assertEqual(rebuilt.report["inspection"], report["inspection"])


@unittest.skipUnless(os.environ.get("QUADRUPED_UI_TESTS") == "1", "Requires active desktop")
class InspectorPanelTests(unittest.TestCase):
    def test_selection_refresh_rollback_profile_changes_and_pixels(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        root = tk.Tk(); root.withdraw()
        with tempfile.TemporaryDirectory() as folder:
            p = ParameterPanel(root,FusionShape(),Path(folder))
            try:
                p.update_model()
                original_pixels = np.asarray(p.pixels).copy()
                original_model, original_qpos = p.preview.model, p.preview.data.qpos.copy()
                p.inspect_active.set(True); p.toggle_inspector()
                self.assertEqual(p.selected_body,"RL_thigh")
                self.assertIs(p.preview.model, original_model)
                np.testing.assert_array_equal(p.preview.data.qpos, original_qpos)
                self.assertGreater(np.abs(np.asarray(p.pixels).astype(float)-original_pixels).sum(), 1000)
                p.inspect_active.set(False); p.toggle_inspector()
                np.testing.assert_allclose(p.pixels,original_pixels,rtol=0,atol=2)
                p.inspect_active.set(True); p.toggle_inspector()
                p.inspector.body_var.set("RL · Lower link + foot"); p.inspector.select()
                self.assertEqual(p.selected_body, "RL_shank")
                self.assertIn("Foot: 0.080000", p.inspector.texts["Mass"].get("1.0","end"))
                p.select_body("RL_thigh",False)
                p.entries["l_thigh"].set("360"); p.update_model()
                expected = f"{p.preview.model.body('RL_thigh').mass[0]:.6f}"
                text = p.inspector.texts["Mass"].get("1.0","end")
                self.assertIn(expected,text)
                p.entries["l_thigh"].set("999"); p.update_model()
                self.assertFalse(p.valid)
                self.assertEqual(p.inspector.texts["Mass"].get("1.0","end"),text)
                p.entries["l_thigh"].set("360"); p.update_model()
                p.select_body("RL_carrier",True)
                p.mode_var.set("Hind-leg upright"); p.change_mode()
                self.assertTrue(p.valid,p.message.get())
                self.assertEqual(p.selected_body,"RL_carrier")
                image = np.asarray(p.pixels).copy()
                for _ in range(2):
                    p.cam.azimuth += 5; p.render_camera()
                    p.cam.azimuth -= 5; p.render_camera()
                    np.testing.assert_allclose(p.pixels,image,rtol=0,atol=4)
                p.select_body("payload",False)
                p.entries["payload_mass"].set("0"); p.update_model()
                self.assertTrue(p.valid,p.message.get())
                self.assertEqual(p.selected_body,"torso")
                self.assertEqual(p.inspector.body_var.get(),"Frame")
                p.apply_config(Morphology()); p.update_model()
                self.assertTrue(p.valid,p.message.get())
                self.assertTrue(p.inspect_active.get())
                self.assertEqual(p.selected_body,"torso")
                p.save()
                saved = next(Path(folder).iterdir())
                self.assertIn("inspection",json.loads((saved/"validation.json").read_text()))
            finally:
                p.close()
