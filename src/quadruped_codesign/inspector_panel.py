"""Compact, read-only inspector that shares the editor's sidebar."""

import math
import tkinter as tk
from tkinter import ttk

from .inspection import selected_scope


def vector_mm(values):
    return "  ".join(f"{value*1000:.2f}" if abs(value) >= .000005 else "0.00" for value in values)


class InspectorPanel(ttk.Frame):
    def __init__(self, parent, on_select):
        super().__init__(parent, padding=8)
        self.on_select = on_select
        self.names = {}
        ttk.Label(self, text="Inspect model", font=("Helvetica", 13, "bold")).pack(anchor="w")
        ttk.Label(self, text="Read-only · Last valid preview", style="Muted.TLabel").pack(anchor="w", pady=(2, 8))
        self.body_var = tk.StringVar()
        self.body_choice = ttk.Combobox(self, textvariable=self.body_var, state="readonly")
        self.body_choice.pack(fill="x")
        self.body_choice.bind("<<ComboboxSelected>>", self.select)
        self.scope_var = tk.StringVar(value="Selected body")
        self.scope_choice = ttk.Combobox(self, textvariable=self.scope_var, state="readonly",
                                        values=("Selected body", "Body + descendants"))
        self.scope_choice.pack(fill="x", pady=(6, 8))
        self.scope_choice.bind("<<ComboboxSelected>>", self.select)
        self.tabs = ttk.Notebook(self, height=1)
        self.tabs.pack(fill="both", expand=True)
        self.texts = {}
        for title in ("Mass", "Pose", "Model", "Static"):
            frame = ttk.Frame(self.tabs)
            self.tabs.add(frame, text=title)
            text = tk.Text(frame, width=1, height=1, wrap="word", font=("Helvetica", 11),
                           bg="#f4f6f8", fg="#23364a", relief="flat", padx=5, pady=8,
                           borderwidth=0, highlightthickness=0, cursor="arrow", state="disabled")
            scroll = ttk.Scrollbar(frame, orient="vertical", command=text.yview)
            scroll.pack(side="right", fill="y")
            text.pack(side="left", fill="both", expand=True)
            text.configure(yscrollcommand=scroll.set)
            text.tag_configure("heading", font=("Helvetica", 11, "bold"), spacing1=7, spacing3=3)
            text.tag_configure("matrix", font=("Menlo", 9))
            text.tag_configure("balanced", foreground="#227353")
            text.tag_configure("unbalanced", foreground="#9b4b20")
            self.texts[title] = text
        ttk.Label(self, text="Orange: selection · White: its CoM", style="Muted.TLabel").pack(anchor="w", pady=(6, 0))

    def select(self, event=None):
        if self.body_var.get() in self.names:
            self.on_select(self.names[self.body_var.get()], self.scope_var.get() == "Body + descendants")

    def refresh(self, preview, name, include_children):
        report = preview.report
        info = report["inspection"]
        bodies = info["bodies"]
        self.names = {v["label"]: k for k, v in bodies.items()}
        self.body_choice.configure(values=list(self.names))
        name, scope = selected_scope(report, name, include_children)
        item = bodies[name]
        self.body_var.set(item["label"])
        self.scope_var.set("Body + descendants" if include_children else "Selected body")
        for text in self.texts.values():
            text.configure(state="normal")
            text.delete("1.0", "end")

        def section(tab, title, content, tag=None):
            text = self.texts[tab]
            text.insert("end", title+"\n", "heading")
            text.insert("end", content+"\n", tag or ())

        section("Mass", "Mass / kg", f"{scope['mass_kg']:.6f}")
        section("Mass", "Included bodies", ", ".join(scope["body_names"]))
        if include_children:
            breakdown = {n: bodies[n]["body_only"]["mass_kg"] for n in scope["body_names"]}
            source = "Sum of compiled body masses. See each body's Selected body view for its source assumptions. Descendants follow the model tree; sibling parts are excluded."
        else:
            breakdown = item["body_mass_breakdown_kg"]
            source = item["source"]
        section("Mass", "CoM · selected-body frame / mm", "x / y / z\n"+vector_mm(scope["com_selected_body_frame_m"]))
        section("Mass", "CoM · world frame / mm", "x / y / z\n"+vector_mm(scope["com_world_m"]))
        inertia = scope["inertia_com_selected_body_axes_kg_m2"]
        # Hide numerical rotation noise in the display; exports retain full precision.
        tolerance = max(abs(v) for row in inertia for v in row) * 1e-12
        matrix = "\n".join(" ".join(f"{(v if abs(v) > tolerance else 0): .2e}" for v in row) for row in inertia)
        section("Mass", "Inertia at CoM / kg·m²", "Selected-body x / y / z axes")
        self.texts["Mass"].insert("end", matrix+"\n", "matrix")
        section("Mass", "Model mass breakdown / kg", "\n".join(f"{n}: {m:.6f}" for n, m in breakdown.items()))
        section("Mass", "Source / approximation", source)

        section("Pose", "Body origin · world / mm", vector_mm(item["origin_world_m"]))
        section("Pose", "Parent body", item["parent"])
        joints = [j for j in info["joints"] if j["body"] in scope["body_names"]]
        if not joints:
            section("Pose", "Joint", "Floating base (6 DoF)." if name == "torso" else "Fixed to parent; no joint on this body.")
        for joint in joints:
            content = f"Angle: {math.degrees(joint['angle_rad']):.2f} deg"
            if joint["range_rad"]:
                low, high = map(math.degrees, joint["range_rad"])
                content += f"\nConcept limits: {low:.2f} … {high:.2f} deg\nNearest limit: {math.degrees(joint['limit_margin_rad']):.2f} deg"
            section("Pose", joint["name"], content)
        leg = name[:2]
        if leg in info["legs"]:
            g = info["legs"][leg]
            section("Pose", "Leg positions · world / mm", "Upper origin: "+vector_mm(g["upper_origin_world_m"])+"\nKnee / elbow: "+vector_mm(g["knee_world_m"])+"\nFoot centre: "+vector_mm(g["foot_center_world_m"]))
            section("Pose", "Derived geometry / mm", f"Upper link: {g['upper_length_m']*1000:.2f}\nLower link: {g['lower_length_m']*1000:.2f}\nHip to foot centre: {g['hip_to_foot_m']*1000:.2f}\nFoot surface above ground: {g['foot_surface_height_m']*1000:.2f}")
        section("Pose", "Definitions", "Positions use world axes. Foot surface height = sphere centre z minus radius. Joint limits are unverified concept settings. FL/FR: front left/right; RL/RR: rear left/right.")

        mass = report["mass_properties_preview_pose"]
        section("Model", "Mass budget / kg", f"Bare robot: {mass['bare_robot_mass_kg']:.6f}\nPayload: {mass['payload_mass_kg']:.6f}\nTotal: {mass['total_mass_kg']:.6f}")
        section("Model", "Actual foot spacing · world y / mm", "\n".join(f"{label}: {info['foot_spacing_y_m'][key]*1000:.2f}" for key, label in (("F", "Front"), ("R", "Hind"))))
        section("Model", "Upper-origin spacing · 3D / mm", "\n".join(f"{label}: {info['upper_origin_spacing_3d_m'][key]*1000:.2f}" for key, label in (("F", "Front"), ("R", "Hind"))))
        section("Model", "Foot surface height / mm", "\n".join(f"{leg}: {v['foot_surface_height_m']*1000:.2f}" for leg, v in info["legs"].items()))
        if info["motor_envelope_gaps"]:
            section("Model", "Named envelope gaps / mm", "\n".join(f"{v['label']}: {'≥ ' if v['distance_m'] >= v['capped_at_m'] else ''}{v['distance_m']*1000:.2f}" for v in info["motor_envelope_gaps"]))
        section("Model", "Scope", "Derived from this valid pose. Envelope gaps cover named proxy pairs only, not all collisions or CAD clearance. See Static for ideal joint loads under its stated assumptions. No controlled stability is verified.")
        self.static_sections(report["statics"], scope["body_names"], section)
        for text in self.texts.values():
            text.configure(state="disabled")
        return name

    def static_sections(self, result, selected_names, section):
        """Whole-model equilibrium is independent of the body-inspection scope."""
        def add(title, content, tag=None):
            section("Static", title, content, tag)

        balanced = result["equilibrium_within_tolerance"]
        label = {"balanced": "Within tolerance", "unbalanced": "Not balanced",
                 "outside_assumptions": "Outside baseline"}[result["status"]]
        add("Static check · whole model", label, "balanced" if balanced else "unbalanced")
        add("Meaning", result["reason"])
        if "weight_n" in result:
            add("Weight / N", f"{result['weight_n']:.3f}")
        forces = result["trial_forces_world_n"]
        if forces:
            title = "Vertical reactions / N" if balanced else "Trial vertical loads / N"
            add(title, "\n".join(f"{leg}: {force[2]:.3f}" for leg, force in forces.items()))
        if balanced:
            largest = result["largest_absolute_joint_torque"]
            add("Largest |joint torque| / N·m", f"{largest['joint']}: {largest['nm']:.3f}")
            add("Joint outputs / N·m", "\n".join(f"{j['joint']}: {j['required_joint_output_nm']:+.3f}" for j in result["joint_torques"]))
            selected = [j for j in result["joint_torques"] if j["body"] in selected_names]
            for joint in selected:
                add(joint["joint"]+" · components / N·m",
                    f"Gravity only: {joint['gravity_only_nm']:+.3f}\nContact term: {joint['contact_generalized_nm']:+.3f}\nRequired = gravity − contact")
        else:
            add("Joint outputs", "Withheld until the whole model passes balance checks. Trial loads are not a standing solution.")
        if result.get("force_residual_world_n") is not None:
            def norm(values):
                return math.sqrt(sum(v*v for v in values))
            add("Unbalanced force / N", f"{norm(result['force_residual_world_n']):.3e}\nTolerance: {result['force_tolerance_n']:.3e}")
            add("Unbalanced moment at CoM / N·m", f"{norm(result['moment_residual_about_com_world_nm']):.5f}\nTolerance: {result['moment_tolerance_nm']:.5f}")
            add("Numerical tolerance", f"Moment tolerance = weight × {result['position_tolerance_m']*1000:g} mm. This is not a stability margin.")
        add("Load-sharing rule", result["assumptions"])
        add("Signs and scope", "Positive torque follows each MJCF joint axis. FL/FR hip and knee denote shoulder pitch and elbow.\n"+result["scope"])
