"""Read-only mass, pose and clearance inspection of the compiled model.

All exported distances are metres. Inertia tensors are about the selected
scope's CoM, expressed in the selected body's axes, never about its origin.
"""

import mujoco
import numpy as np

from .fusion import FusionShape
from .generator import LEGS, FOOT_MASS, LINK_DENSITY
from .mass_properties import body_tensor, shift_tensor


def descendants(model, root):
    members = {root}
    # MuJoCo orders parents before children.
    for body in range(root + 1, model.nbody):
        if int(model.body_parentid[body]) in members:
            members.add(body)
    return sorted(members)


def aggregate(model, data, root, members):
    masses = model.body_mass[members]
    total = float(masses.sum())
    if total <= 0:
        raise ValueError("The selected inspection scope has no mass.")
    com = (masses[:, None] * data.xipos[members]).sum(axis=0) / total
    inertia_world = np.zeros((3, 3))
    for body, mass in zip(members, masses):
        rotation = data.xmat[body].reshape(3, 3)
        inertia_world += rotation @ body_tensor(model, body) @ rotation.T
        inertia_world += shift_tensor(mass, data.xipos[body] - com)
    frame = data.xmat[root].reshape(3, 3)
    return {
        "body_names": [model.body(i).name for i in members],
        "mass_kg": total,
        "com_world_m": com.tolist(),
        "com_selected_body_frame_m": (frame.T @ (com-data.xpos[root])).tolist(),
        "inertia_com_selected_body_axes_kg_m2": (frame.T @ inertia_world @ frame).tolist(),
    }


def body_description(name, mass, shape):
    """Labels describe accounting groups, not a claimed CAD bill of materials."""
    density = shape.link_density if isinstance(shape, FusionShape) else LINK_DENSITY
    if name == "torso":
        return "Frame", "Declared frame mass; uniform-box inertia estimate. Excludes separate motor, leg and payload bodies.", {"Frame estimate": mass}
    if name == "payload":
        return "Payload", "User-declared fixed payload; uniform-box mass distribution. Mounting hardware is omitted.", {"Payload": mass}
    leg, kind = name.split("_", 1)
    if kind == "roll_motor":
        return f"{leg} · Roll module", "Nominal motor + declared extra mass, lumped into a uniform cylinder. Clearance envelope adds no mass.", {"Roll module": mass}
    if kind == "carrier":
        drive = "Shoulder / elbow" if leg.startswith("F") else "Hip / knee"
        module_mass = shape.motor.mass + shape.motor.extra_mass
        return f"{leg} · Pitch drive pair", "Two proximal drive modules, including declared extra/module. Moving links are separate bodies; the roll motor is a sibling, not a descendant.", {f"{drive.split(' / ')[0]} module": module_mass, f"Remote {drive.split(' / ')[1].lower()} module": module_mass}
    if kind == "thigh":
        return f"{leg} · Upper link", f"Uniform capsule at {density:g} kg/m³ surrogate density. Not a CAD or measured mass.", {"Upper link": mass}
    if kind == "shank":
        return f"{leg} · Lower link + foot", f"One rigid body: capsule at {density:g} kg/m³ + {FOOT_MASS:g} kg spherical foot. Not a CAD or measured mass.", {"Lower link": mass-FOOT_MASS, "Foot": FOOT_MASS}
    return name, "Compiled rigid-body properties; provenance not specified.", {name: mass}


def inspect_model(model, data, shape):
    bodies = {}
    joints = []
    for i in range(model.njnt):
        if model.jnt_type[i] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        angle = float(data.qpos[model.jnt_qposadr[i]])
        limited = bool(model.jnt_limited[i])
        low, high = map(float, model.jnt_range[i])
        joints.append({"name": model.joint(i).name, "body": model.body(int(model.jnt_bodyid[i])).name,
                       "angle_rad": angle, "range_rad": [low, high] if limited else None,
                       "limit_margin_rad": min(angle-low, high-angle) if limited else None})
    for i in range(1, model.nbody):
        name = model.body(i).name
        label, source, parts = body_description(name, float(model.body_mass[i]), shape)
        members = descendants(model, i)
        bodies[name] = {
            "label": label, "parent": model.body(int(model.body_parentid[i])).name,
            "source": source, "body_mass_breakdown_kg": parts,
            "origin_world_m": data.xpos[i].tolist(),
            "body_only": aggregate(model, data, i, [i]),
            "subtree": aggregate(model, data, i, members),
        }
    legs, gaps = {}, []
    for leg in LEGS:
        hip = data.body(leg+"_thigh").xpos
        knee = data.body(leg+"_shank").xpos
        foot = data.site(leg+"_foot").xpos
        radius = float(model.geom(leg+"_foot_geom").size[0])
        legs[leg] = {"upper_origin_world_m": hip.tolist(), "knee_world_m": knee.tolist(),
                     "foot_center_world_m": foot.tolist(), "foot_surface_height_m": float(foot[2]-radius),
                     "upper_length_m": float(np.linalg.norm(knee-hip)),
                     "lower_length_m": float(np.linalg.norm(foot-knee)),
                     "hip_to_foot_m": float(np.linalg.norm(foot-hip))}
        if isinstance(shape, FusionShape):
            for first, second, label in (("roll", "hip", "Roll / inner pitch"),
                                         ("hip", "remote", "Pitch pair")):
                names = [f"{leg}_{s}_module_envelope" for s in (first, second)]
                distance = float(mujoco.mj_geomDistance(model, data, *(model.geom(n).id for n in names), 10., None))
                gaps.append({"label": f"{leg} · {label}", "geom_names": names,
                             "distance_m": distance, "capped_at_m": 10.})
    return {
        "schema_version": 1, "bodies": bodies, "joints": joints, "legs": legs,
        "foot_spacing_y_m": {g: abs(legs[g+"L"]["foot_center_world_m"][1]-legs[g+"R"]["foot_center_world_m"][1]) for g in ("F", "R")},
        "upper_origin_spacing_3d_m": {g: float(np.linalg.norm(np.subtract(legs[g+"L"]["upper_origin_world_m"], legs[g+"R"]["upper_origin_world_m"]))) for g in ("F", "R")},
        "motor_envelope_gaps": gaps,
        "scope": "Read-only compiled-model properties at the saved pose. Subtree means selected body plus its descendants, not necessarily a whole leg. Inertia is about the selected scope CoM in selected-body axes. Gaps cover only named proxy envelope pairs, not CAD or all-body clearance. Structural mass and joint limits remain uncalibrated. Forces and torques, if available, are in the separate statics section with its own assumptions.",
    }


def selected_scope(report, name, include_children=False):
    bodies = report["inspection"]["bodies"]
    name = name if name in bodies else "torso"
    return name, bodies[name]["subtree" if include_children else "body_only"]


def add_selection_overlay(scene, model, report, name, include_children=False):
    """Tint scene copies only; neither model geometry nor dynamics is modified."""
    _, scope = selected_scope(report, name, include_children)
    ids = {model.body(n).id for n in scope["body_names"]}
    for geom in scene.geoms[:scene.ngeom]:
        if geom.objtype == mujoco.mjtObj.mjOBJ_GEOM and geom.objid >= 0 and int(model.geom_bodyid[geom.objid]) in ids:
            geom.rgba[:3] = (1., .30, .08)
    if scene.ngeom >= scene.maxgeom:
        raise RuntimeError("Not enough render geometry capacity for selected CoM")
    marker = scene.geoms[scene.ngeom]
    mujoco.mjv_initGeom(marker, mujoco.mjtGeom.mjGEOM_SPHERE, np.array([.012, 0, 0]),
                       np.asarray(scope["com_world_m"]), np.eye(3).reshape(9), np.ones(4, dtype=np.float32))
    marker.label = "Selected CoM"
    scene.ngeom += 1
