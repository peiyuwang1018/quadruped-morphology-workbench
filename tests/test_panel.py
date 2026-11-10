import json
import os
from pathlib import Path
import tempfile
import unittest

import mujoco
import numpy as np
from PIL import Image

from quadruped_codesign.generator import Morphology, Payload
from quadruped_codesign.panel_model import Preview, load_morphology, morphology_toml


class PanelSnapshotTests(unittest.TestCase):
    def test_saved_config_reproduces_compiled_physics(self):
        shape = Morphology(front_thigh_length=.26, torso_mass=9.5,
                           payload=Payload(2, (.2, .16, .12), (.12, 0, .12)))
        preview = Preview.build(shape)
        with tempfile.TemporaryDirectory() as temporary:
            path = preview.save(Path(temporary), Image.new("RGB", (32, 24)))
            restored = Preview.build(load_morphology(path / "config.toml"))
            compiled = mujoco.MjModel.from_xml_path(str(path / "robot.xml"))
            self.assertEqual(restored.shape, shape)
            for name in ("body_mass", "body_ipos", "body_inertia", "body_pos", "geom_size", "jnt_pos"):
                np.testing.assert_array_equal(getattr(compiled, name), getattr(restored.model, name))
            report = json.loads((path / "validation.json").read_text())
            self.assertTrue(report["passed"])
            self.assertEqual(report["unactuated_rollout_seconds"], 0)
            with Image.open(path / "preview.png") as picture:
                self.assertEqual(picture.size, (32, 24))
            # Repeated saves create independent snapshots, never overwrite a candidate.
            second = preview.save(Path(temporary), Image.new("RGB", (16, 16)))
            self.assertNotEqual(path, second)
            self.assertTrue((path / "robot.xml").is_file())

    def test_failed_save_publishes_no_partial_snapshot(self):
        class BrokenImage:
            def save(self, path):
                raise OSError("disk write failed")
        with tempfile.TemporaryDirectory() as temporary:
            preview = Preview.build(Morphology())
            with self.assertRaises(OSError):
                preview.save(Path(temporary), BrokenImage())
            self.assertEqual(list(Path(temporary).iterdir()), [])

    def test_config_backwards_compatibility_and_bad_fields(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "config.toml"
            path.write_text("thigh_length = 0.3\nshank_length = 0.28\n")
            shape = load_morphology(path)
            self.assertEqual(shape.lengths("FL"), (.3, .28))
            self.assertEqual(shape.payload.mass, 0)
            path.write_text(morphology_toml(shape))
            self.assertEqual(load_morphology(path), shape)
            path.write_text("torso_mas = 8\n")
            with self.assertRaises(TypeError):
                load_morphology(path)


@unittest.skipUnless(os.environ.get("QUADRUPED_UI_TESTS") == "1", "Opt-in: requires Tk and an active desktop")
class NativePanelTests(unittest.TestCase):
    def setUp(self):
        import tkinter as tk
        from quadruped_codesign.panel import ParameterPanel
        self.temporary = tempfile.TemporaryDirectory()
        self.root = tk.Tk()
        self.root.withdraw()
        self.panel = ParameterPanel(self.root, Morphology(payload=Payload(mass=2)), Path(self.temporary.name))
        self.panel.update_model()

    def tearDown(self):
        self.panel.close()
        self.temporary.cleanup()

    def test_live_payload_update_preserves_camera_and_recomputes_com(self):
        panel = self.panel
        initial = panel.preview.report["mass_properties_preview_pose"]
        first_image = np.asarray(panel.pixels).copy()
        panel.cam.azimuth = 47
        panel.cam.distance = 2.1
        panel.entries["position_x"].set("12")  # UI uses cm, model uses m.
        panel.update_model()
        moved = panel.preview.report["mass_properties_preview_pose"]
        self.assertTrue(panel.valid)
        self.assertEqual(panel.preview.shape.payload.position[0], .12)
        self.assertAlmostEqual(moved["total_mass_kg"], initial["total_mass_kg"])
        self.assertAlmostEqual(moved["system_com_torso_frame_m"][0] - initial["system_com_torso_frame_m"][0],
                               2 * .12 / initial["total_mass_kg"])
        self.assertEqual((panel.cam.azimuth, panel.cam.distance), (47, 2.1))
        self.assertGreater(np.abs(np.asarray(panel.pixels).astype(float) - first_image).mean(), 1)
        panel.save()
        path = next(Path(self.temporary.name).iterdir())
        self.assertEqual(load_morphology(path / "config.toml").payload.position[0], .12)

    def test_invalid_input_keeps_visible_model_and_disables_save_then_recovers(self):
        panel = self.panel
        original = panel.preview
        image = np.asarray(panel.pixels).copy()
        panel.entries["position_x"].set("25")  # Outside back for the default box.
        panel.update_model()
        self.assertFalse(panel.valid)
        self.assertIs(panel.preview, original)
        np.testing.assert_array_equal(panel.pixels, image)
        self.assertTrue(panel.save_button.instate(["disabled"]))
        panel.save()
        self.assertEqual(list(Path(self.temporary.name).iterdir()), [])
        panel.entries["position_x"].set("0")
        panel.update_model()
        self.assertTrue(panel.valid)
        # Removing/adding the payload changes body count without leaking a renderer.
        for mass in (0, 2, 0, 2):
            panel.entries["payload_mass"].set(str(mass))
            panel.update_model()
            self.assertTrue(panel.valid)
            self.assertEqual(panel.preview.model.nbody, 10 + int(mass > 0))

    def test_orbit_updates_without_clearing_the_displayed_image(self):
        from PIL import ImageTk
        panel = self.panel
        self.root.deiconify()
        self.root.update_idletasks()
        panel.show_image()
        item = panel.canvas.find_withtag("preview")
        image_name = panel.canvas.itemcget(item[0], "image")
        model = panel.preview.model
        first = np.asarray(ImageTk.getimage(panel.photo)).copy()
        for angle in range(95, 156, 5):
            panel.cam.azimuth = angle
            panel.render_camera()
            self.root.update_idletasks()
            self.assertEqual(panel.canvas.find_withtag("preview"), item)
            self.assertEqual(panel.canvas.itemcget(item[0], "image"), image_name)
            self.assertIs(panel.preview.model, model)
            displayed = np.asarray(ImageTk.getimage(panel.photo))[:, :, :3]
            self.assertGreater(np.ptp(displayed), 30)
        last = np.asarray(ImageTk.getimage(panel.photo))
        self.assertGreater(np.abs(first.astype(float) - last).mean(), 1)

    def test_model_replacement_preserves_rendered_pixels_and_drag_direction(self):
        from types import SimpleNamespace
        panel = self.panel
        # This sequence is essential: a new model is drawn before the previous
        # renderer is retired. A valid first image does not prove later frames work.
        for key, value in (("position_x", "12"), ("payload_mass", "0"),
                           ("front_thigh_length", "26"), ("payload_mass", "2")):
            panel.entries[key].set(value)
            panel.update_model()
            self.assertTrue(panel.valid)
            correct = np.asarray(panel.pixels).copy()
            for _ in range(3):
                panel.render_camera()
                # GPU antialiasing can differ by one 8-bit level at an edge.
                np.testing.assert_allclose(np.asarray(panel.pixels), correct, rtol=0, atol=1,
                                           err_msg="A stationary camera changed after replacing the model")
            azimuth, elevation = panel.cam.azimuth, panel.cam.elevation
            panel.drag_start(SimpleNamespace(x=300, y=200))
            # One-way horizontal drags must not change elevation or flip pixels.
            for x in (305, 310, 315, 320):
                panel.drag(SimpleNamespace(x=x, y=200))
                panel.render_camera()
                self.assertEqual(panel.cam.elevation, elevation)
            self.assertAlmostEqual(panel.cam.azimuth, azimuth - 8)
            # Return to the exact camera and compare the actual rendered RGB image.
            panel.cam.azimuth, panel.cam.elevation = azimuth, elevation
            panel.render_camera()
            np.testing.assert_allclose(np.asarray(panel.pixels), correct, rtol=0, atol=1)


if __name__ == "__main__":
    unittest.main()
