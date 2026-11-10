"""Native MuJoCo viewer and repeatable rendered morphology comparisons."""

from pathlib import Path
import time

import mujoco
import numpy as np

from .mass_properties import inspect_mass, leg_lengths


class ContextSafeRenderer(mujoco.Renderer):
    """Retire GPU resources in the context that created them.

    MuJoCo 3.17's Renderer.close() frees its MjrContext without first making its
    GL context current. When a replacement has already rendered, deleting the
    previous renderer can therefore delete the replacement's framebuffer and
    textures. Its next read then returns invalid RGB data (flips/colour stripes).

    Keep the version-specific private-context access confined to this adapter;
    the renderer lifecycle is covered by model-replacement regression tests.
    """

    def close(self):
        context = getattr(self, "_gl_context", None)
        if context is not None:
            context.make_current()
        super().close()


def preview_data(model: mujoco.MjModel) -> mujoco.MjData:
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("preview").id)
    mujoco.mj_forward(model, data)
    return data


def camera() -> mujoco.MjvCamera:
    cam = mujoco.MjvCamera()
    cam.lookat[:] = [0, 0, 0.30]
    cam.distance = 1.8
    cam.azimuth = 135
    cam.elevation = -22
    return cam


def view(path: str | Path, *, simulate: bool = False, seconds: float | None = None) -> None:
    """Default is a frozen preview; --simulate intentionally applies gravity without control."""
    import mujoco.viewer

    model = mujoco.MjModel.from_xml_path(str(Path(path).resolve()))
    data = preview_data(model)
    mode = "UNACTUATED DROP (no controller)" if simulate else "FROZEN MORPHOLOGY PREVIEW"
    print(f"{mode}\n{Path(path).resolve()}\nClose the window to exit.", flush=True)
    with mujoco.viewer.launch_passive(model, data, show_left_ui=False, show_right_ui=False) as viewer:
        with viewer.lock():
            cam = camera()
            viewer.cam.lookat[:] = cam.lookat
            viewer.cam.distance = cam.distance
            viewer.cam.azimuth = cam.azimuth
            viewer.cam.elevation = cam.elevation
        start = time.monotonic()
        while viewer.is_running() and (seconds is None or time.monotonic() - start < seconds):
            tick = time.monotonic()
            if simulate:
                mujoco.mj_step(model, data)
            else:
                mujoco.mj_forward(model, data)
            viewer.sync()
            dt = model.opt.timestep if simulate else 1 / 60
            time.sleep(max(0, dt - (time.monotonic() - tick)))


def render_comparison(paths: list[Path], output: Path, *, shared_qpos=None) -> Path:
    """Same camera and resolution for every candidate. Uses actual MuJoCo rendering."""
    from PIL import Image, ImageDraw

    width, height, header = 800, 540, 132
    columns = min(len(paths), 2)
    rows = (len(paths) + columns - 1) // columns
    sheet = Image.new("RGB", (width * columns, (height + header) * rows), "#17232e")
    draw = ImageDraw.Draw(sheet)
    for index, path in enumerate(paths):
        model = mujoco.MjModel.from_xml_path(str(path.resolve()))
        data = preview_data(model)
        if shared_qpos is not None:
            data.qpos[:] = shared_qpos
            mujoco.mj_forward(model, data)
        with ContextSafeRenderer(model, height=height, width=width) as renderer:
            renderer.update_scene(data, camera=camera())
            pixels = renderer.render().copy()
        if np.ptp(pixels) < 10:
            raise RuntimeError("Renderer returned an unexpectedly blank image")
        x, y = index % columns * width, index // columns * (height + header)
        sheet.paste(Image.fromarray(pixels), (x, y + header))
        front, hind = leg_lengths(model, "FL"), leg_lengths(model, "RL")
        mass = inspect_mass(model, data)
        draw.text((x + 24, y + 12), path.stem, fill="white", font_size=26)
        draw.text((x + 24, y + 45), f"Front: {front[0]:.2f} / {front[1]:.2f} m   Hind: {hind[0]:.2f} / {hind[1]:.2f} m", fill="#d9e5ed", font_size=20)
        draw.text((x + 24, y + 72), f"Total: {mass['total_mass_kg']:.3f} kg   Payload: {mass['payload_mass_kg']:.1f} kg   CoM x: {1000*mass['system_com_torso_frame_m'][0]:.2f} mm", fill="#a9b7c7", font_size=20)
        label = "Same base + joint pose; frozen inspection (not standing)" if shared_qpos is not None else "Frozen inspection; same camera; base height follows length"
        draw.text((x + 24, y + 101), label, fill="#a9b7c7", font_size=18)
    output.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(output)
    return output
