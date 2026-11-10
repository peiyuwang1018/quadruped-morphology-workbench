from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

from quadruped_codesign.fusion import (
    FusionShape, Motor, Stance, JointLimits, LEGS, FOOT_RADIUS,
    evaluate_fusion, shape_from_model, solve_pose, PROFILE, LEGACY_PROFILE,
)
from quadruped_codesign.panel_model import Preview, load_morphology
from quadruped_codesign.generator import Morphology


class FusionPhysicsTests(unittest.TestCase):
    def test_confirmed_datums_and_foot_contacts(self):
        preview = Preview.build(FusionShape())
        m, d = preview.model, preview.data
        self.assertEqual((m.nq, m.nv, m.nu), (19, 18, 0))
        front, rear = d.body('FL_carrier').xpos, d.body('RL_carrier').xpos
        np.testing.assert_allclose(front-rear, [.7, -.025, -.03], atol=1e-10)
        self.assertAlmostEqual(front[1]-d.body('FR_carrier').xpos[1], .45)
        self.assertAlmostEqual(rear[1]-d.body('RR_carrier').xpos[1], .50)
        self.assertAlmostEqual(rear[2], .55)
        for leg in LEGS:
            self.assertAlmostEqual(d.site(leg+'_foot').xpos[2], FOOT_RADIUS)
        self.assertEqual(shape_from_model(m), preview.shape)

    def test_ik_tracks_height_pitch_and_foot_targets_without_stretching_links(self):
        for stance in (Stance(hip_height=.50), Stance(body_pitch=.08, front_x=.04, hind_x=-.03),
                       Stance(front_width=.58, hind_width=.62)):
            shape = FusionShape(stance=stance, l_upperarm=.345)
            preview = Preview.build(shape)
            self.assertTrue(preview.report['passed'])
            for leg in LEGS:
                upper, lower = shape.lengths(leg)
                data = preview.data
                hip = data.body(leg+'_thigh').xpos
                knee = data.body(leg+'_shank').xpos
                foot = data.site(leg+'_foot').xpos
                self.assertAlmostEqual(np.linalg.norm(knee-hip), upper)
                self.assertAlmostEqual(np.linalg.norm(foot-knee), lower)
                self.assertLess(preview.report['stance']['foot_error_m'][leg], 1e-8)

    def test_motor_mass_stays_proximal_and_envelope_is_massless(self):
        initial = Preview.build(FusionShape())
        extra = Preview.build(FusionShape(motor=Motor(extra_mass=.2)))
        self.assertAlmostEqual(extra.report['robot_mass_kg']-initial.report['robot_mass_kg'], 12*.2)
        self.assertAlmostEqual(initial.report['mass_budget_kg']['nominal_motor_modules'], 16.8)
        for leg in LEGS:
            for suffix in ('thigh', 'shank'):
                np.testing.assert_array_equal(initial.model.body(leg+'_'+suffix).mass,
                                              extra.model.body(leg+'_'+suffix).mass)
            for suffix in ('hip', 'remote'):
                self.assertEqual(initial.model.geom(f'{leg}_{suffix}_module').bodyid,
                                 initial.model.body(leg+'_carrier').id)
        # Changing distal joint angles must not move their remotely placed motor mass.
        d, m = initial.data, initial.model
        module = d.geom('FL_remote_module').xpos.copy()
        foot = d.site('FL_foot').xpos.copy()
        d.qpos[m.joint('FL_hip').qposadr[0]] += .2
        d.qpos[m.joint('FL_knee').qposadr[0]] -= .1
        mujoco.mj_forward(m, d)
        np.testing.assert_allclose(d.geom('FL_remote_module').xpos, module, atol=1e-12)
        self.assertGreater(np.linalg.norm(d.site('FL_foot').xpos-foot), .01)
        bigger = Preview.build(FusionShape(motor=Motor(envelope_diameter=.120)))
        np.testing.assert_array_equal(bigger.model.body_mass, initial.model.body_mass)

    def test_invalid_targets_limits_bounds_and_overlapping_envelopes_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'unreachable'):
            Preview.build(FusionShape(l_upperarm=.295, l_lowerarm=.310,
                                      z_offset_hip_shoulder=-.01,
                                      stance=Stance(hip_height=.65, front_width=.65, front_x=.10)))
        with self.assertRaisesRegex(ValueError, 'joint limits'):
            Preview.build(FusionShape(limits=JointLimits(hip=(-.1,.1))))
        with self.assertRaisesRegex(ValueError, 'exploration range'):
            FusionShape(l_upperarm=.8)
        with self.assertRaisesRegex(ValueError, 'Motor envelopes overlap'):
            Preview.build(FusionShape(profile=LEGACY_PROFILE, motor=Motor(pair_half_spacing=.02)))
        with self.assertRaisesRegex(ValueError, 'non-negative'):
            Motor(axial_gap=-.001)
        with self.assertRaises(ValueError):
            FusionShape(stance=Stance(hip_height=float('nan')))

    def test_independent_mass_and_inertia_checks_detect_changes(self):
        preview = Preview.build(FusionShape())
        model = preview.model
        model.body_mass[model.body('FL_carrier').id] += .1
        model.body_inertia[model.body('RR_shank').id, 0] *= 1.1
        report = evaluate_fusion(model)
        self.assertFalse(report['passed'])
        self.assertFalse(report['checks']['analytic_body_masses'])
        self.assertFalse(report['checks']['analytic_inertia_tensors'])

    def test_snapshot_round_trip_includes_pose_bounds_and_hidden_mount_parameters(self):
        shape = FusionShape(motor=Motor(extra_mass=.1, axial_gap=.008, thigh_axial_offset=.003),
                            stance=Stance(hip_height=.53, body_pitch=.05))
        preview = Preview.build(shape)
        with tempfile.TemporaryDirectory() as directory:
            saved = preview.save(Path(directory), Image.new('RGB', (8,8)))
            restored = Preview.build(load_morphology(saved/'config.toml'))
            self.assertEqual(restored.shape, shape)
            for field in ('body_mass','body_inertia','key_qpos','geom_size'):
                np.testing.assert_array_equal(getattr(restored.model,field),getattr(preview.model,field))
            self.assertEqual(shape_from_model(restored.model), shape)

    def test_coaxial_stack_and_thigh_origin_in_rotated_pose(self):
        shape = FusionShape(motor=Motor(axial_gap=.008, thigh_axial_offset=.003),
                            stance=Stance(body_pitch=.08, front_width=.58, hind_width=.62))
        preview = Preview.build(shape)
        m,d = preview.model,preview.data
        for leg in LEGS:
            side = 1 if leg.endswith('L') else -1
            inner,outer = d.geom(leg+'_hip_module'),d.geom(leg+'_remote_module')
            # Inspect compiled coordinates, not the generation helper.
            np.testing.assert_allclose(m.geom(leg+'_hip_module').pos, [0, side*.03, 0])
            np.testing.assert_allclose(m.geom(leg+'_remote_module').pos, [0, side*.098, 0])
            np.testing.assert_allclose(m.body(leg+'_thigh').pos, [0, side*.101, 0])
            axis = inner.xmat.reshape(3,3)[:,2]
            np.testing.assert_allclose(np.cross(outer.xpos-inner.xpos,axis),0,atol=1e-10)
            self.assertAlmostEqual(np.linalg.norm(outer.xpos-inner.xpos),.068)
            trim = d.body(leg+'_carrier').xmat.reshape(3,3) @ [0,side*.003,0]
            np.testing.assert_allclose(d.body(leg+'_thigh').xpos,outer.xpos+trim,atol=1e-10)
            self.assertAlmostEqual(np.linalg.norm(d.site(leg+'_foot').xpos-
                preview.report['stance']['foot_targets_world_m'][leg]),0,places=8)
        # Validation must detect a geometric regression even if mass/pose data remain.
        m.geom_pos[m.geom('FL_remote_module').id,0] += .01
        self.assertFalse(evaluate_fusion(m)['checks']['coaxial_motor_axes'])

    def test_offset_ik_and_mounting_changes_keep_targets_fixed(self):
        initial = Preview.build(FusionShape())
        for trim in (-.01,.01):
            preview = Preview.build(FusionShape(motor=Motor(thigh_axial_offset=trim),
                                               stance=Stance(hip_height=.5,body_pitch=-.05)))
            self.assertTrue(preview.report['passed'])
            self.assertAlmostEqual(preview.report['robot_mass_kg'],initial.report['robot_mass_kg'])
            for leg in LEGS:
                self.assertLess(preview.report['stance']['foot_error_m'][leg],1e-8)
        with self.assertRaisesRegex(ValueError,'lateral-offset'):
            solve_pose(FusionShape(motor=Motor(pitch_outboard=.6)))
        with self.assertRaisesRegex(ValueError,'Thigh axial trim'):
            Motor(thigh_axial_offset=.04)

    def test_v1_snapshot_retains_old_placement_and_pose(self):
        shape = load_morphology(Path(__file__).parent/'fixtures/fusion_v1.toml')
        self.assertEqual(shape.profile,LEGACY_PROFILE)
        self.assertEqual(FusionShape().profile,PROFILE)
        preview = Preview.build(shape)
        for leg in LEGS:
            side = 1 if leg.endswith('L') else -1
            np.testing.assert_allclose(preview.model.geom(leg+'_hip_module').pos,[.075,side*.03,0])
            np.testing.assert_allclose(preview.model.geom(leg+'_remote_module').pos,[-.075,side*.03,0])
            np.testing.assert_array_equal(preview.model.body(leg+'_thigh').pos,0)
        np.testing.assert_allclose(preview.data.qpos[7:10],
            [.05056415031721006,.7525516136357842,-1.4645513083448745],atol=1e-10)
        with tempfile.TemporaryDirectory() as folder:
            saved=preview.save(Path(folder),Image.new('RGB',(8,8)))
            restored=Preview.build(load_morphology(saved/'config.toml'))
            self.assertEqual(restored.shape,shape)
            np.testing.assert_array_equal(restored.model.body_inertia,preview.model.body_inertia)


@unittest.skipUnless(os.environ.get('QUADRUPED_UI_TESTS') == '1', 'Requires active desktop')
class FusionPanelTests(unittest.TestCase):
    def test_live_ik_invalid_rollback_profile_switch_and_camera_pixels(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        root = tk.Tk()
        root.withdraw()
        with tempfile.TemporaryDirectory() as directory:
            panel = ParameterPanel(root, FusionShape(), Path(directory))
            try:
                panel.update_model()
                self.assertTrue(panel.valid, panel.message.get())
                lengths = panel.preview.shape.lengths('FL')
                for key, value in (('stance.hip_height','500'), ('w_shoulder','470'),
                                   ('stance.body_pitch','3'), ('payload_mass','0')):
                    panel.entries[key].set(value)
                    panel.update_model()
                    self.assertTrue(panel.valid, panel.message.get())
                    correct = np.asarray(panel.pixels).copy()
                    panel.render_camera()
                    # Transparent packaging can vary by two 8-bit levels at an edge.
                    np.testing.assert_allclose(panel.pixels, correct, rtol=0, atol=2)
                    panel.cam.azimuth += 5
                    panel.render_camera()
                    panel.cam.azimuth -= 5
                    panel.render_camera()
                    np.testing.assert_allclose(panel.pixels, correct, rtol=0, atol=2)
                self.assertEqual(panel.preview.shape.lengths('FL'),lengths)
                panel.save()
                saved = next(Path(directory).iterdir())
                self.assertEqual(load_morphology(saved/'config.toml'),panel.preview.shape)
                previous = panel.preview
                panel.entries['stance.hip_height'].set('999')
                panel.update_model()
                self.assertFalse(panel.valid)
                self.assertIs(panel.preview, previous)
                self.assertTrue(panel.save_button.instate(['disabled']))
                # Open config / Reset must preserve hidden data and change units safely.
                for shape in (Morphology(), FusionShape(motor=Motor(extra_mass=.1))):
                    panel.apply_config(shape)
                    panel.update_model()
                    self.assertTrue(panel.valid,panel.message.get())
                    self.assertEqual(panel.preview.shape,shape)
            finally:
                panel.close()


if __name__ == '__main__':
    unittest.main()
