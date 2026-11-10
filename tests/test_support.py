from dataclasses import replace
import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

from quadruped_codesign.fusion import FusionShape,Stance,JointLimits
from quadruped_codesign.generator import Payload
from quadruped_codesign.panel_model import Preview,load_morphology
from quadruped_codesign.posture import align_projection
from quadruped_codesign.support import projection_geometry,TOLERANCE


class SupportTests(unittest.TestCase):
    def test_polygon_margin_and_finite_support_segment(self):
        contacts=[[-1,-1,0],[1,-1,0],[1,1,0],[-1,1,0]]
        inside=projection_geometry([0,0,3],contacts)
        self.assertTrue(inside['inside_or_on_support'])
        self.assertAlmostEqual(inside['signed_margin_m'],1)
        outside=projection_geometry([2,0,3],contacts)
        self.assertFalse(outside['inside_or_on_support'])
        self.assertAlmostEqual(outside['projection_distance_m'],1)
        line=[[-1,0,0],[1,0,0]]
        on=projection_geometry([0,0,3],line)
        self.assertTrue(on['inside_or_on_support'])
        self.assertEqual(on['support_dimension'],1)
        self.assertEqual(on['signed_margin_m'],0)
        self.assertFalse(on['static_stability_verified'])
        self.assertFalse(projection_geometry([2,0,3],line)['inside_or_on_support'])
        self.assertAlmostEqual(projection_geometry([0,.2,3],line)['projection_distance_m'],.2)

    def test_upright_has_only_hind_contacts_and_explicit_different_limits(self):
        shape=FusionShape(stance=Stance(mode='hind_stand'))
        p=Preview.build(shape)
        self.assertEqual(p.report['support']['contact_legs'],['RL','RR'])
        self.assertEqual(p.report['support']['support_dimension'],1)
        for leg in ('FL','FR'):
            self.assertGreater(p.report['stance']['foot_surface_height_m'][leg],1.)
        for leg in ('RL','RR'):
            self.assertLess(p.report['stance']['foot_error_m'][leg],1e-8)
            self.assertGreater(p.data.qpos[p.model.joint(leg+'_hip').qposadr[0]],shape.limits.hip[1])
            np.testing.assert_allclose(p.model.joint(leg+'_hip').range,shape.upright_limits.hip)
        np.testing.assert_allclose(p.data.body('torso').xmat.reshape(3,3)[:,0],[0,0,1],atol=1e-10)
        self.assertEqual(shape.limits.hip,(-1.35,1.35))
        with self.assertRaisesRegex(ValueError,'joint limits'):
            Preview.build(replace(shape,upright_limits=JointLimits()))

    def test_com_uses_compiled_mass_and_payload_transform(self):
        for mode,position in (('quadruped',(.1,0,.12)),('hind_stand',(0,0,.14))):
            a=Preview.build(FusionShape(stance=Stance(mode=mode)))
            b=Preview.build(replace(a.shape,payload=Payload(2,position=position)))
            actual=np.asarray(b.report['support']['com_world_m'])-a.report['support']['com_world_m']
            expected=(2/a.report['robot_mass_kg'])*(a.data.body('torso').xmat.reshape(3,3) @
                       (np.asarray(position)-a.shape.payload.position))
            np.testing.assert_allclose(actual,expected,atol=1e-10)
            np.testing.assert_allclose(b.report['support']['com_world_m'],
                                      b.data.subtree_com[b.model.body('torso').id],atol=1e-10)

    def test_alignment_changes_only_foot_offsets_and_keeps_limits(self):
        for mode in ('quadruped','hind_stand'):
            original=FusionShape(stance=Stance(mode=mode))
            aligned,details=align_projection(original)
            p=Preview.build(aligned)
            self.assertTrue(p.report['support']['inside_or_on_support'])
            self.assertLessEqual(abs(details['fore_aft_residual_m']),TOLERANCE)
            self.assertEqual(replace(aligned,stance=original.stance),original)
            changed={k for k in vars(original.stance) if getattr(original.stance,k)!=getattr(aligned.stance,k)}
            self.assertEqual(changed,{'upright_hind_x'} if mode=='hind_stand' else {'front_x','hind_x'})
            if mode=='hind_stand':
                self.assertLessEqual(p.report['support']['projection_distance_m'],TOLERANCE)
                self.assertEqual(p.report['support']['signed_margin_m'],0)

    def test_failed_alignment_retains_input_and_does_not_widen_bounds(self):
        shape=FusionShape(stance=Stance(mode='hind_stand'))
        bounds=dict(shape.bounds);bounds['stance.upright_hind_x']=(.049,.051)
        shape=replace(shape,bounds=bounds)
        before=shape.to_dict()
        with self.assertRaisesRegex(ValueError,'No feasible CoM alignment'):
            align_projection(shape)
        self.assertEqual(shape.to_dict(),before)

    def test_reduced_body_side_gap_without_changing_pair_gap_or_leg_geometry(self):
        new=Preview.build(FusionShape())
        old=Preview.build(replace(new.shape,motor=replace(new.shape.motor,roll_inboard=.100)))
        for leg in ('FL','FR','RL','RR'):
            def clearance(p):
                return mujoco.mj_geomDistance(p.model,p.data,p.model.geom(leg+'_roll_module_envelope').id,
                                             p.model.geom(leg+'_hip_module_envelope').id,.2,None)
            self.assertGreater(clearance(new),0)
            self.assertGreater(clearance(old)-clearance(new),.019)
            np.testing.assert_array_equal(new.model.geom(leg+'_hip_module').pos,old.model.geom(leg+'_hip_module').pos)
        np.testing.assert_array_equal(new.data.qpos,old.data.qpos)
        self.assertAlmostEqual(new.report['robot_mass_kg'],old.report['robot_mass_kg'])

    def test_both_pose_settings_and_aligned_snapshot_round_trip(self):
        original=FusionShape(stance=Stance(mode='hind_stand',hip_height=.51,front_x=.03))
        shape,_=align_projection(original)
        p=Preview.build(shape)
        with tempfile.TemporaryDirectory() as folder:
            saved=p.save(Path(folder),Image.new('RGB',(8,8)))
            loaded=load_morphology(saved/'config.toml')
            self.assertEqual(loaded,shape)
            self.assertEqual(loaded.stance.hip_height,.51)
            self.assertEqual(loaded.stance.front_x,.03)
            rebuilt=Preview.build(loaded)
            np.testing.assert_array_equal(rebuilt.data.qpos,p.data.qpos)
            self.assertLessEqual(rebuilt.report['support']['projection_distance_m'],TOLERANCE)


@unittest.skipUnless(os.environ.get('QUADRUPED_UI_TESTS')=='1','Requires active desktop')
class SupportPanelTests(unittest.TestCase):
    def test_switch_align_overlay_and_reload(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        root=tk.Tk();root.withdraw()
        with tempfile.TemporaryDirectory() as folder:
            p=ParameterPanel(root,FusionShape(),Path(folder))
            try:
                p.update_model()
                p.entries['stance.hip_height'].set('510');p.update_model()
                p.mode_var.set('Hind-leg upright');p.change_mode()
                self.assertTrue(p.valid,p.message.get())
                self.assertEqual(p.preview.shape.stance.mode,'hind_stand')
                p.align_com()
                self.assertTrue(p.valid,p.message.get())
                self.assertLessEqual(p.preview.report['support']['projection_distance_m'],TOLERANCE)
                image=np.asarray(p.pixels).copy()
                model=p.preview.model;mass=model.body_mass.copy();qpos=p.preview.data.qpos.copy()
                rendered_count=p.renderer.scene.ngeom
                for _ in range(3):
                    p.render_camera()
                    self.assertEqual(p.renderer.scene.ngeom,rendered_count)
                    np.testing.assert_allclose(p.pixels,image,rtol=0,atol=2)
                p.show_com.set(False);p.render_camera()
                self.assertLess(p.renderer.scene.ngeom,rendered_count)
                self.assertGreater(np.abs(np.asarray(p.pixels).astype(float)-image).sum(),1000)
                p.show_com.set(True);p.render_camera()
                np.testing.assert_allclose(p.pixels,image,rtol=0,atol=2)
                np.testing.assert_array_equal(model.body_mass,mass)
                np.testing.assert_array_equal(p.preview.data.qpos,qpos)
                p.save();saved=next(Path(folder).iterdir())
                p.apply_config(load_morphology(saved/'config.toml'));p.update_model()
                self.assertTrue(p.valid,p.message.get())
                self.assertEqual(p.preview.shape.stance.mode,'hind_stand')
                p.mode_var.set('Four-foot stance');p.change_mode()
                self.assertTrue(p.valid,p.message.get())
                self.assertAlmostEqual(float(p.entries['stance.hip_height'].get()),510)
                self.assertEqual(len(p.preview.report['support']['contact_legs']),4)
            finally:
                p.close()


if __name__=='__main__':
    unittest.main()
