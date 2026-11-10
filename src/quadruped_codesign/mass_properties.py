"""Independent primitive-shape calculations used to audit MuJoCo's compiler.

All tensors are about the centre of mass, expressed in the containing body frame.
The source of truth for dimensions is the generator's documented primitive model.
"""

import math

import mujoco
import numpy as np

from .generator import (
    LEGS, LINK_DENSITY, THIGH_RADIUS, SHANK_RADIUS, FOOT_MASS, FOOT_RADIUS, TORSO_SIZE,
)


def numeric(model, name, fallback):
    identifier = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_NUMERIC, name)
    return np.asarray(fallback) if identifier < 0 else model.numeric(identifier).data.copy()


def leg_lengths(model, leg):
    hind = numeric(model, "morphology", [0.22, 0.20])
    return numeric(model, "front_morphology", hind) if leg.startswith("F") else hind


def capsule(length, radius):
    """Cylinder + two hemispheres, endpoints at z=0 and z=-length."""
    cylinder_mass = LINK_DENSITY * math.pi * radius**2 * length
    cap_mass = LINK_DENSITY * 4 * math.pi * radius**3 / 3
    transverse = cylinder_mass * (3 * radius**2 + length**2) / 12
    transverse += cap_mass * (2 * radius**2 / 5 + 3 * length * radius / 8 + length**2 / 4)
    axial = cylinder_mass * radius**2 / 2 + cap_mass * 2 * radius**2 / 5
    return cylinder_mass + cap_mass, np.array([0., 0., -length / 2]), np.diag([transverse, transverse, axial])


def box(mass, full_size):
    x, y, z = full_size
    return mass, np.zeros(3), mass / 12 * np.diag([y*y + z*z, x*x + z*z, x*x + y*y])


def shift_tensor(mass, displacement):
    displacement = np.asarray(displacement)
    return mass * (np.dot(displacement, displacement) * np.eye(3) - np.outer(displacement, displacement))


def expected_body_properties(model):
    """No compiled body mass/inertia is used to compute the expected values."""
    torso_mass = float(numeric(model, "torso_mass", [8.0])[0])
    expected = {"torso": box(torso_mass, TORSO_SIZE)}
    for leg in LEGS:
        thigh, shank = leg_lengths(model, leg)
        expected[f"{leg}_thigh"] = capsule(thigh, THIGH_RADIUS)
        shaft_mass, shaft_com, shaft_inertia = capsule(shank, SHANK_RADIUS)
        foot_com = np.array([0., 0., -shank])
        mass = shaft_mass + FOOT_MASS
        com = (shaft_mass * shaft_com + FOOT_MASS * foot_com) / mass
        inertia = shaft_inertia + shift_tensor(shaft_mass, shaft_com - com)
        inertia += 2 / 5 * FOOT_MASS * FOOT_RADIUS**2 * np.eye(3) + shift_tensor(FOOT_MASS, foot_com - com)
        expected[f"{leg}_shank"] = mass, com, inertia
    payload = numeric(model, "payload_spec", [0, .20, .16, .12, 0, 0, .12])
    if payload[0] > 0:
        expected["payload"] = box(float(payload[0]), payload[1:4])
    return expected


def body_tensor(model, identifier):
    rotation = np.empty(9)
    mujoco.mju_quat2Mat(rotation, model.body_iquat[identifier])
    rotation = rotation.reshape(3, 3)
    return rotation @ np.diag(model.body_inertia[identifier]) @ rotation.T


def inspect_mass(model, data):
    """Mass, centre of mass and body-frame inertial data for export/inspection."""
    torso = data.body("torso")
    torso_rotation = torso.xmat.reshape(3, 3)
    payload_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "payload")
    payload_mass = float(model.body_mass[payload_id]) if payload_id >= 0 else 0.
    total = float(model.body_mass.sum())
    world_com = (model.body_mass[:, None] * data.xipos).sum(axis=0) / total
    torso_com = torso_rotation.T @ (world_com - torso.xpos)
    return {
        "total_mass_kg": total, "bare_robot_mass_kg": total - payload_mass,
        "payload_mass_kg": payload_mass,
        "system_com_world_m": world_com.tolist(), "system_com_torso_frame_m": torso_com.tolist(),
        "bodies": {model.body(i).name: {
            "parent": model.body(int(model.body_parentid[i])).name,
            "mass_kg": float(model.body_mass[i]),
            "com_body_frame_m": model.body_ipos[i].tolist(),
            "inertia_com_body_frame_kg_m2": body_tensor(model, i).tolist(),
        } for i in range(1, model.nbody)},
    }
