"""A single native Tk window, with real MuJoCo rendering on the main thread."""

import math
from dataclasses import replace
from pathlib import Path
import time
import tkinter as tk
from tkinter import filedialog, ttk

import mujoco
from PIL import Image, ImageTk

from .generator import Morphology, Payload
from .fusion import FusionShape, PROFILE
from . import fusion_panel
from .panel_model import Preview, load_morphology
from .visualization import ContextSafeRenderer, camera
from .support import add_support_overlay


# UI distances are centimetres; the generator and saved files always use SI units.
FIELDS = {
    "front_thigh_length": ("Front upper leg", "cm", 10, 45, 0.5),
    "front_shank_length": ("Front lower leg", "cm", 10, 45, 0.5),
    "thigh_length": ("Hind upper leg", "cm", 10, 45, 0.5),
    "shank_length": ("Hind lower leg", "cm", 10, 45, 0.5),
    "torso_mass": ("Torso mass", "kg", 1, 30, 0.1),
    "payload_mass": ("Payload mass", "kg", 0, 10, 0.1),
    "size_x": ("Box length", "cm", 5, 50, 0.5),
    "size_y": ("Box width", "cm", 4, 26, 0.5),
    "size_z": ("Box height", "cm", 2, 30, 0.5),
    "position_x": ("Fore-aft position (x)", "cm", -25, 25, 0.5),
    "position_y": ("Lateral position (y)", "cm", -10, 10, 0.5),
    "position_z": ("Centre height (z)", "cm", 7, 50, 0.5),
}


def values_from_shape(shape):
    values = {key: getattr(shape, key) * (100 if spec[1] == "cm" else 1)
              for key, spec in FIELDS.items() if hasattr(shape, key)}
    values["payload_mass"] = shape.payload.mass
    for index, axis in enumerate("xyz"):
        values[f"size_{axis}"] = shape.payload.size[index] * 100
        values[f"position_{axis}"] = shape.payload.position[index] * 100
    return values


def shape_from_values(values):
    checked = {}
    for key, (label, unit, low, high, _) in FIELDS.items():
        try:
            value = float(values[key])
        except (ValueError, TypeError):
            raise ValueError(f"{label}: enter a number.") from None
        if not math.isfinite(value) or not low <= value <= high:
            raise ValueError(f"{label}: enter a value from {low:g} to {high:g} {unit}.")
        checked[key] = value / 100 if unit == "cm" else value
    size = tuple(checked[f"size_{axis}"] for axis in "xyz")
    position = tuple(checked[f"position_{axis}"] for axis in "xyz")
    if checked["payload_mass"] > 0:
        if position[2] - size[2] / 2 < 0.06 - 1e-10:
            raise ValueError(f"Box overlaps the torso. Centre height must be at least {6 + size[2] * 50:g} cm. Use 'Centre on back' to place it on top.")
        if any(abs(position[i]) + size[i] / 2 > (0.28, 0.13)[i] + 1e-10 for i in (0, 1)):
            raise ValueError(f"Box extends beyond the back. For this size, x must be within ±{28-size[0]*50:g} cm and y within ±{13-size[1]*50:g} cm.")
    return Morphology(**{key: checked[key] for key in list(FIELDS)[:5]},
                      payload=Payload(checked["payload_mass"], size, position))


class ParameterPanel:
    def __init__(self, root, shape: Morphology, output: Path):
        self.root, self.output = root, output.resolve()
        self.edit_shape = shape
        self.fusion = isinstance(shape, FusionShape)
        self.fields = fusion_panel.panel_fields(shape) if self.fusion else FIELDS
        self.preview = self.renderer = self.pixels = self.photo = None
        self.update_job = self.camera_job = self.resize_job = None
        self.entries, self.sliders = {}, {}
        self.valid = False
        self.last_drag = None
        self.show_com = tk.BooleanVar(master=root, value=True)
        self.cam = camera()
        if self.fusion:
            self.cam.distance = 2.8 if shape.stance.upright else 2.4
        root.title("Quadruped Workbench · MuJoCo")
        width, height = min(1230, root.winfo_screenwidth() - 60), min(850, root.winfo_screenheight() - 100)
        root.geometry(f"{width}x{height}")
        root.minsize(min(1000, width), min(720, height))
        root.configure(bg="#f4f6f8")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._layout()
        self._set_values(shape)
        self.update_job = root.after(50, self.update_model)

    def _layout(self):
        style = ttk.Style(self.root)
        style.configure("TFrame", background="#f4f6f8")
        style.configure("TLabel", background="#f4f6f8", foreground="#23364a", font=("Helvetica", 12))
        style.configure("Heading.TLabel", font=("Helvetica", 20, "bold"))
        style.configure("Muted.TLabel", foreground="#65748a", font=("Helvetica", 11))
        style.configure("Metric.TLabel", font=("Helvetica", 22, "bold"))
        style.configure("TButton", font=("Helvetica", 12), padding=(8, 5))
        outer = ttk.Frame(self.root, padding=18)
        outer.pack(fill="both", expand=True)
        header = ttk.Frame(outer)
        header.pack(fill="x", pady=(0, 14))
        ttk.Label(header, text="Quadruped Workbench", style="Heading.TLabel").pack(side="left")
        subtitle = ("Coaxial motor layout" if self.edit_shape.profile == PROFILE else "Legacy motor layout") if self.fusion else "Explore shape and mass"
        ttk.Label(header, text=subtitle, style="Muted.TLabel").pack(side="left", padx=16)
        self.status = tk.StringVar(value="Preparing preview…")
        ttk.Label(header, textvariable=self.status, style="Muted.TLabel").pack(side="right")
        content = ttk.Frame(outer)
        content.pack(fill="both", expand=True)
        sidebar = ttk.Frame(content, width=340 if self.fusion else 305)
        sidebar.pack(side="left", fill="y", padx=(0, 18))
        sidebar.pack_propagate(False)
        # Let the notebook use the remaining height instead of pushing actions
        # below the window when a tab has more fields than the legacy panel.
        tabs = ttk.Notebook(sidebar, height=1)
        tabs.pack(fill="both", expand=True)
        self.tabs = tabs
        if self.fusion:
            motor = self.edit_shape.motor
            mounting = (f"Coaxial hip/knee motors. Thigh starts at outer module centre + {motor.thigh_axial_offset*1000:g} mm axial trim. Envelope gap: {motor.axial_gap*1000:g} mm (estimate)."
                        if self.edit_shape.profile == PROFILE else "Legacy fore/aft motor placement. Reset example loads the coaxial model.")
            descriptions = (
                ("Geometry", fusion_panel.GEOMETRY, "Negative z = lower shoulders.\nRanges are provisional."),
                ("Stance", fusion_panel.stance_fields(self.edit_shape),
                 (f"Upright concept hip range: {math.degrees(self.edit_shape.upright_limits.hip[0]):.0f} to {math.degrees(self.edit_shape.upright_limits.hip[1]):.0f} deg (unverified). Front limbs are held clear. A two-point support line has zero area."
                  if self.edit_shape.stance.upright else "Height uses the rear mounting datum. Foot x offsets use each datum's ground projection. Alignment moves foot targets, keeping geometry and masses fixed.")),
                ("Mass", fusion_panel.MASS,
                 f"12 × {motor.mass:g} kg nominal modules (AKE90-8 reference). "
                 f"Physical size: Ø{motor.diameter*1000:g} × {motor.length*1000:g} mm. "
                 f"Packaging: Ø{motor.envelope_diameter*1000:g} × {motor.envelope_length*1000:g} mm.\n\n"
                 + mounting + "\n\nExtra/module = 0 means omitted. Frame and link mass remain uncalibrated estimates."),
            )
            for title, keys, description in descriptions:
                tab = ttk.Frame(tabs, padding=10)
                tabs.add(tab, text=title)
                if title == 'Stance':
                    self.mode_var = tk.StringVar(value='Hind-leg upright' if self.edit_shape.stance.upright else 'Four-foot stance')
                    mode = ttk.Combobox(tab, textvariable=self.mode_var, state='readonly',
                                        values=('Four-foot stance','Hind-leg upright'))
                    mode.pack(fill='x',pady=(0,8))
                    mode.bind('<<ComboboxSelected>>',self.change_mode)
                for key in keys:
                    self._field(tab, key)
                if title == 'Stance':
                    ttk.Button(tab,text='Align CoM projection',command=self.align_com).pack(fill='x',pady=(4,0))
                note = ttk.Label(tab, text=description, style="Muted.TLabel", wraplength=250, justify="left")
                note.pack(fill="x", pady=8)
                note.bind("<Configure>", lambda event, label=note: label.configure(wraplength=max(100, event.width)))
        else:
            body = ttk.Frame(tabs, padding=10)
            tabs.add(body, text=" Legs & torso ")
            ttk.Label(body, text="Independent front / hind legs.\nLeft and right remain symmetric.", style="Muted.TLabel").pack(anchor="w", pady=(0, 10))
            for key in list(FIELDS)[:5]:
                self._field(body, key)
            ttk.Label(body, text="Lengths are joint-centre distances.\nLonger links change mass and inertia.\nTorso mass excludes legs and payload.", style="Muted.TLabel", wraplength=220, justify="left").pack(anchor="w", pady=14)
        cargo = ttk.Frame(tabs, padding=10)
        tabs.add(cargo, text="Payload")
        for key in fusion_panel.PAYLOAD if self.fusion else list(FIELDS)[5:]:
            self._field(cargo, key)
        ttk.Button(cargo, text="Centre on back", command=self.seat_payload).pack(fill="x", pady=(4, 6))
        ttk.Label(cargo, text="Full box dimensions; position refers to its centre. Set mass to 0 to remove it.", style="Muted.TLabel", wraplength=220, justify="left").pack(anchor="w")
        actions = ttk.Frame(sidebar)
        actions.pack(fill="x", pady=(12, 0))
        ttk.Button(actions, text="Open config…", command=self.load).pack(side="left", expand=True, fill="x")
        ttk.Button(actions, text="Reset example", command=self.reset).pack(side="left", expand=True, fill="x", padx=(5, 0))
        self.save_button = ttk.Button(sidebar, text="Save config & model", command=self.save, state="disabled")
        self.save_button.pack(fill="x", pady=(7, 0))
        self.right = ttk.Frame(content)
        self.right.pack(side="left", fill="both", expand=True)
        toolbar = ttk.Frame(self.right)
        toolbar.pack(fill="x", pady=(0, 8))
        ttk.Label(toolbar, text="Model preview", font=("Helvetica", 14, "bold")).pack(side="left")
        for label, angle in (("Perspective", (135, -22)), ("Side", (90, -12)), ("Top", (90, -89))):
            ttk.Button(toolbar, text=label, command=lambda a=angle: self.set_camera(a)).pack(side="right", padx=2)
        self.canvas = tk.Canvas(self.right, bg="#263441", highlightthickness=0, cursor="hand2")
        self.canvas.pack(fill="both", expand=True)
        self.preview_item = self.canvas.create_image(0, 0, tags="preview")
        self.canvas.bind("<Configure>", self.resize)
        self.canvas.bind("<ButtonPress-1>", self.drag_start)
        self.canvas.bind("<B1-Motion>", self.drag)
        self.canvas.bind("<ButtonRelease-1>", lambda event: setattr(self, "last_drag", None))
        self.canvas.bind("<MouseWheel>", self.zoom)
        hint = "Gold: CoM · Pink: projection · Cyan: support · Drag to orbit" if self.fusion else "Drag to orbit · Scroll to zoom     Purple: payload · Orange torso dot: front"
        legend = ttk.Frame(self.right)
        legend.pack(fill='x',pady=7)
        ttk.Label(legend, text=hint, style="Muted.TLabel").pack(side="left")
        if self.fusion:
            ttk.Checkbutton(legend,text='Show CoM',variable=self.show_com,command=self.render_camera).pack(side='right')
        metrics = ttk.Frame(self.right)
        metrics.pack(fill="x", pady=(4, 9))
        self.metrics = []
        labels = ("Model mass / kg", "World CoM x / mm", "World CoM y / mm", "World CoM z / mm") if self.fusion else ("Model mass / kg", "CoM x / mm", "CoM y / mm", "CoM z / mm")
        for index, label in enumerate(labels):
            frame = ttk.Frame(metrics)
            frame.grid(row=0, column=index, sticky="ew", padx=(0, 12))
            metrics.columnconfigure(index, weight=1)
            ttk.Label(frame, text=label, style="Muted.TLabel").pack(anchor="w")
            variable = tk.StringVar(value="—")
            ttk.Label(frame, textvariable=variable, style="Metric.TLabel").pack(anchor="w")
            self.metrics.append(variable)
        coordinates = "CoM in world coordinates · Ground z = 0 · +x reference forward, +y left, +z up." if self.fusion else "CoM = centre of mass, relative to torso. +x forward, +y left, +z up."
        ttk.Label(self.right, text=coordinates, style="Muted.TLabel").pack(anchor="w")
        self.message = tk.StringVar(value="")
        self.message_label = ttk.Label(self.right, textvariable=self.message, wraplength=700, justify="left")
        self.message_label.pack(fill="x", pady=(9, 0))
        footer = ("Geometric pose preview · Ideal point contacts · Uncalibrated mounting / masses / joint limits · No balance controller."
                  if self.fusion else "Frozen morphology preview · No standing or walking controller. Shorter legs may be above the ground.")
        ttk.Label(outer, text=footer, style="Muted.TLabel").pack(anchor="w", pady=(12, 0))

    def _field(self, parent, key):
        label, unit, low, high, step = self.fields[key]
        frame = ttk.Frame(parent)
        frame.pack(fill="x", pady=(0, 3 if self.fusion else 7))
        row = ttk.Frame(frame)
        row.pack(fill="x")
        ttk.Label(row, text=label).pack(side="left")
        ttk.Label(row, text=unit, style="Muted.TLabel", width=max(3, len(unit))).pack(side="right")
        entry_var = tk.StringVar()
        entry_width = 9 if key in ('stance.front_x','stance.hind_x','stance.upright_hind_x') else 6
        entry = ttk.Entry(row, textvariable=entry_var, width=entry_width, justify="right")
        entry.pack(side="right", padx=(3, 4))
        slider = ttk.Scale(frame, from_=low, to=high,
                           command=lambda value: self.slide(key, float(value), step))
        slider.pack(fill="x", pady=(0 if self.fusion else 3, 0))
        entry.bind("<KeyRelease>", lambda event: self.schedule_update())
        entry.bind("<Return>", lambda event: self.schedule_update(0))
        entry.bind("<FocusOut>", lambda event: self.schedule_update(0))
        self.entries[key], self.sliders[key] = entry_var, slider

    def _set_values(self, shape):
        self.edit_shape = shape
        values = fusion_panel.panel_values(shape) if self.fusion else values_from_shape(shape)
        for key, value in values.items():
            self.entries[key].set(f"{value:.12g}")
            # Setting a ttk Scale invokes its callback; suppress it during population.
            slider = self.sliders[key]
            callback = slider.cget("command")
            slider.configure(command="")
            slider.set(value)
            slider.configure(command=callback)

    def slide(self, key, value, step):
        low, high = self.fields[key][2:4]
        rounded = max(low, min(high, round(value / step)*step))
        self.entries[key].set(f"{rounded:.12g}")
        self.schedule_update()

    def schedule_update(self, delay=180):
        self.valid = False
        self.save_button.configure(state="disabled")
        self.status.set("Updating…")
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = self.root.after(delay, self.update_model)

    def update_model(self):
        if self.update_job:
            self.root.after_cancel(self.update_job)
        self.update_job = None
        candidate_renderer = None
        started = time.perf_counter()
        try:
            values = {key: variable.get() for key, variable in self.entries.items()}
            shape = fusion_panel.shape_from_panel(values, self.edit_shape) if self.fusion else shape_from_values(values)
            if self.preview and shape == self.preview.shape:
                self.valid = True
                self.save_button.configure(state="normal")
                self.status.set("Up to date")
                self.message.set(self.success_message())
                self.message_label.configure(foreground=self.success_color())
                return
            candidate = Preview.build(shape)
            candidate_renderer = ContextSafeRenderer(candidate.model, height=700, width=1000)
            # Only the vertical target follows model height. User orbit and zoom persist.
            self.cam.lookat[2] = candidate.data.body("torso").xpos[2] * 0.65
            candidate_renderer.update_scene(candidate.data, camera=self.cam)
            if self.show_com.get():
                add_support_overlay(candidate_renderer.scene,candidate.report)
            pixels = Image.fromarray(candidate_renderer.render().copy())
            old_renderer = self.renderer
            self.preview, self.renderer, self.pixels = candidate, candidate_renderer, pixels
            candidate_renderer = None
            if old_renderer:
                old_renderer.close()
            self._set_values(shape)
            self.show_image()
            mass = candidate.report["mass_properties_preview_pose"]
            com_key = 'system_com_world_m' if self.fusion else 'system_com_torso_frame_m'
            values = [mass["total_mass_kg"], *(v * 1000 for v in mass[com_key])]
            for variable, value in zip(self.metrics, values):
                variable.set(f"{value:.3f}")
            self.valid = True
            self.save_button.configure(state="normal")
            self.status.set(f"Updated · {(time.perf_counter()-started)*1000:.0f} ms")
            self.message.set(self.success_message())
            self.message_label.configure(foreground=self.success_color())
        except (ValueError, TypeError, RuntimeError, mujoco.FatalError) as error:
            self.valid = False
            self.save_button.configure(state="disabled")
            self.status.set("Check parameters")
            self.message.set(f"{error}\nPreview and values show the last valid model. Current inputs have not been applied.")
            self.message_label.configure(foreground="#a04722")
        finally:
            if candidate_renderer:
                candidate_renderer.close()

    def success_message(self):
        if not self.fusion:
            return "Mass, centre of mass, inertia and kinematics checks passed."
        residual = max(self.preview.report["stance"]["foot_error_m"].values())*1000
        support = self.preview.report['support']
        count = len(support['contact_legs'])
        if support['support_dimension'] == 1:
            condition = f"Line error: {support['projection_distance_m']*1000:.3f} mm · Zero-area support; balance NOT verified."
        else:
            status = 'Inside' if support['inside_or_on_support'] else 'Outside'
            condition = f"{status} support polygon · Signed margin: {support['signed_margin_m']*1000:.1f} mm · Geometry only."
        return f"{count} feet on ground · Max target error: {residual:.4f} mm\n{condition}"

    def success_color(self):
        if self.fusion and self.preview:
            support = self.preview.report['support']
            if support['support_dimension']==1 or not support['inside_or_on_support']:
                return '#a04722'
        return '#246652'

    def show_image(self):
        if self.pixels is None:
            return
        width, height = self.canvas.winfo_width(), self.canvas.winfo_height()
        image = self.pixels.copy()
        image.thumbnail((max(1, width), max(1, height)), Image.Resampling.LANCZOS)
        if self.photo is not None and (self.photo.width(), self.photo.height()) == image.size:
            # Keep the displayed Tk image alive rather than temporarily clearing
            # its canvas item. GPU context ownership is handled by the renderer.
            self.photo.paste(image)
        else:
            new_photo = ImageTk.PhotoImage(image, master=self.root)
            self.canvas.itemconfigure(self.preview_item, image=new_photo)
            self.photo = new_photo  # Release the old image only after the swap.
        self.canvas.coords(self.preview_item, width / 2, height / 2)

    def resize(self, event):
        if self.resize_job:
            self.root.after_cancel(self.resize_job)
        self.resize_job = self.root.after(60, self.show_image)
        self.message_label.configure(wraplength=max(200, event.width))

    def render_camera(self):
        if self.camera_job:
            self.root.after_cancel(self.camera_job)
        self.camera_job = None
        if self.renderer:
            self.renderer.update_scene(self.preview.data, camera=self.cam)
            if self.show_com.get():
                add_support_overlay(self.renderer.scene,self.preview.report)
            self.pixels = Image.fromarray(self.renderer.render().copy())
            self.show_image()

    def schedule_camera(self):
        if not self.camera_job:
            self.camera_job = self.root.after(30, self.render_camera)

    def set_camera(self, angles):
        self.cam.azimuth, self.cam.elevation = angles
        self.cam.distance = (2.8 if self.edit_shape.stance.upright else 2.4) if self.fusion else 1.8
        self.schedule_camera()

    def drag_start(self, event):
        self.last_drag = (event.x, event.y)

    def drag(self, event):
        if self.last_drag:
            self.cam.azimuth -= (event.x - self.last_drag[0]) * 0.4
            self.cam.elevation = max(-89, min(20, self.cam.elevation + (event.y-self.last_drag[1]) * 0.3))
            self.last_drag = (event.x, event.y)
            self.schedule_camera()

    def zoom(self, event):
        delta = max(-10, min(10, event.delta))
        self.cam.distance = max(0.7, min(4.0, self.cam.distance * math.exp(-delta * 0.025)))
        self.schedule_camera()

    def seat_payload(self):
        try:
            height = float(self.entries["size_z"].get())
            if not math.isfinite(height):
                raise ValueError
        except ValueError:
            self.message.set("Enter a valid box height first.")
            return
        for key, value in (("position_x", 0), ("position_y", 0), ("position_z", (60 if self.fusion else 6) + height / 2)):
            self.entries[key].set(f"{value:g}")
        self.schedule_update(0)

    def reset(self):
        self.apply_config(FusionShape() if self.fusion else Morphology(payload=Payload(mass=2)))

    def change_mode(self, event=None):
        requested = 'hind_stand' if self.mode_var.get()=='Hind-leg upright' else 'quadruped'
        self.update_model()
        if not self.valid:
            self.mode_var.set('Hind-leg upright' if self.edit_shape.stance.upright else 'Four-foot stance')
            return
        try:
            shape = replace(self.preview.shape,stance=replace(self.preview.shape.stance,mode=requested))
            self.apply_config(shape)
            self.update_model()
            self.tabs.select(1)
        except (ValueError,TypeError) as error:
            self.mode_var.set('Hind-leg upright' if self.edit_shape.stance.upright else 'Four-foot stance')
            self.message.set(f'Cannot switch pose: {error}')
            self.message_label.configure(foreground='#a04722')

    def align_com(self):
        from .posture import align_projection
        self.update_model()
        if not self.valid or not self.fusion:
            return
        try:
            shape,details = align_projection(self.preview.shape)
            self._set_values(shape)
            self.update_model()
            if self.valid:
                self.preview.report['alignment'] = details
        except ValueError as error:
            self.message.set(str(error))
            self.message_label.configure(foreground='#a04722')

    def apply_config(self, shape):
        """Validate first, then rebuild controls to match the profile and its bounds."""
        if isinstance(shape, FusionShape):
            fusion_panel.shape_from_panel(fusion_panel.panel_values(shape), shape)
        else:
            shape_from_values(values_from_shape(shape))
        Preview.build(shape)
        for job in (self.update_job, self.camera_job, self.resize_job):
            if job:
                self.root.after_cancel(job)
        self.update_job = self.camera_job = self.resize_job = None
        for widget in self.root.winfo_children():
            widget.destroy()
        self.photo = None
        self.entries, self.sliders = {}, {}
        self.edit_shape = shape
        self.fusion = isinstance(shape, FusionShape)
        self.fields = fusion_panel.panel_fields(shape) if self.fusion else FIELDS
        self.cam.distance = (2.8 if shape.stance.upright else 2.4) if self.fusion else 1.8
        self._layout()
        self._set_values(shape)
        # The controls were rebuilt even if the loaded config equals the preview.
        old_renderer = self.renderer
        self.preview = self.renderer = None
        if old_renderer:
            old_renderer.close()
        self.schedule_update(0)

    def load(self):
        path = filedialog.askopenfilename(parent=self.root, title="Open morphology config", filetypes=[("TOML config", "*.toml")])
        if path:
            try:
                shape = load_morphology(Path(path))
                self.apply_config(shape)
            except (ValueError, TypeError, OSError) as error:
                self.message.set(f"Could not open config: {error}")
                self.message_label.configure(foreground="#a04722")

    def save(self):
        if self.valid and self.preview:
            try:
                path = self.preview.save(self.output, self.pixels)
                self.message.set(f"Saved config, model, checks and image:\n{path}")
                self.message_label.configure(foreground="#246652")
                print(f"Saved: {path}", flush=True)
            except OSError as error:
                self.message.set(f"Could not save: {error}")
                self.message_label.configure(foreground="#a04722")

    def close(self):
        for job in (self.update_job, self.camera_job, self.resize_job):
            if job:
                self.root.after_cancel(job)
        if self.renderer:
            self.renderer.close()
        self.root.destroy()


def launch(config: Path | None = None, output: Path = Path("outputs/panel")):
    shape = load_morphology(config) if config else FusionShape()
    if isinstance(shape, FusionShape):
        fusion_panel.shape_from_panel(fusion_panel.panel_values(shape), shape)
    else:
        shape_from_values(values_from_shape(shape))
    root = tk.Tk()
    ParameterPanel(root, shape, output)
    root.mainloop()
