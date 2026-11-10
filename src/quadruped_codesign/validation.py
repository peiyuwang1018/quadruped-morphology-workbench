"""Small structural checks, not a certificate of hardware feasibility."""

import math

import mujoco
import numpy as np

from .generator import LEGS, HIP_Z
from .mass_properties import body_tensor, expected_body_properties, inspect_mass, leg_lengths, numeric


def evaluate_robot(model: mujoco.MjModel, *, steps: int = 500) -> dict:
    """Check our generator's model and run a one-second unactuated drop by default."""
    if isinstance(steps, bool) or not isinstance(steps, int) or steps < 0:
        raise ValueError("steps must be a non-negative integer")
    thigh, shank = leg_lengths(model, "RL")
    front_thigh, front_shank = leg_lengths(model, "FL")
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    feet = {leg: data.site(f"{leg}_foot").xpos.copy() for leg in LEGS}
    fk_error = max(float(np.max(np.abs(feet[leg] - [x, y, model.qpos0[2] + HIP_Z - sum(leg_lengths(model, leg))])))
                   for leg, (x, y) in LEGS.items())
    zero_mass = inspect_mass(model, data)
    expected = expected_body_properties(model)
    mass_errors, com_errors, inertia_errors = [], [], []
    for name, (mass, com, inertia) in expected.items():
        body = model.body(name)
        mass_errors.append(abs(float(body.mass[0]) - mass))
        com_errors.append(float(np.max(np.abs(body.ipos - com))))
        inertia_errors.append(float(np.max(np.abs(body_tensor(model, body.id) - inertia))))
    payload_spec = numeric(model, "payload_spec", [0, .20, .16, .12, 0, 0, .12])
    payload_present = payload_spec[0] > 0
    hinges = [model.joint(f"{leg}_{joint}").id for leg in LEGS for joint in ("hip", "knee")]
    checks = {
        "structure_8_hinges_floating_base": bool(model.nq == 15 and model.nv == 14 and model.nu == 8 and model.nbody == 10 + int(payload_present)),
        "analytic_body_masses": bool(max(mass_errors) < 1e-9),
        "analytic_body_centres_of_mass": bool(max(com_errors) < 1e-9),
        "analytic_body_inertia_tensors": bool(max(inertia_errors) < 1e-9),
        "positive_link_mass_and_inertia": bool(np.all(model.body_mass[1:] > 0) and np.all(model.body_inertia[1:] > 0)),
        "valid_inertia_triangle": bool(np.all(2 * np.max(model.body_inertia[1:], axis=1)
                                              <= model.body_inertia[1:].sum(axis=1) + 1e-12)),
        "finite_mass_and_inertia": bool(np.isfinite(model.body_mass).all() and np.isfinite(model.body_inertia).all()),
        "zero_pose_forward_kinematics": fk_error < 1e-9,
        "limited_hinges": bool(np.all(model.jnt_limited[hinges]) and np.all(model.jnt_range[hinges, 0] < model.jnt_range[hinges, 1])),
        "zero_pose_no_penetration": not any(c.dist < -1e-8 for c in data.contact),
    }
    if payload_present:
        body = model.body("payload")
        checks["payload_fixed_to_torso"] = bool(model.body_parentid[body.id] == model.body("torso").id and model.body_jntnum[body.id] == 0)
        checks["payload_position_matches_input"] = bool(np.allclose(body.pos, payload_spec[4:7], rtol=0, atol=1e-10))
    mujoco.mj_resetDataKeyframe(model, data, model.key("preview").id)
    mujoco.mj_forward(model, data)
    checks["preview_pose_no_penetration"] = not any(c.dist < -1e-8 for c in data.contact)
    checks["preview_inside_joint_limits"] = all(
        model.jnt_range[j, 0] <= data.qpos[model.jnt_qposadr[j]] <= model.jnt_range[j, 1]
        for j in hinges)
    preview_feet = {leg: data.site(f"{leg}_foot").xpos.tolist() for leg in LEGS}
    preview_mass = inspect_mass(model, data)
    finite = True
    for _ in range(steps):
        mujoco.mj_step(model, data)
        finite &= bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all() and np.isfinite(data.qacc).all())
        if not finite:
            break
    warnings = {mujoco.mjtWarning(i).name: int(w.number) for i, w in enumerate(data.warning) if w.number}
    if steps:
        checks["unactuated_rollout_finite"] = finite
        checks["unactuated_rollout_no_warnings"] = not warnings
        checks["unactuated_rollout_time_advanced"] = math.isclose(data.time, steps * model.opt.timestep, abs_tol=1e-8)
    return {
        "passed": all(checks.values()), "checks": checks,
        "morphology_m": {"thigh_length": float(thigh), "shank_length": float(shank),
                         "front_thigh_length": float(front_thigh), "front_shank_length": float(front_shank)},
        "payload_spec": {"mass_kg": float(payload_spec[0]), "full_size_m": payload_spec[1:4].tolist(), "position_torso_frame_m": payload_spec[4:7].tolist()},
        "mass_properties_zero_pose": zero_mass,
        "mass_properties_preview_pose": preview_mass,
        "analytic_max_errors": {"mass_kg": max(mass_errors), "com_m": max(com_errors), "inertia_kg_m2": max(inertia_errors)},
        "robot_mass_kg": float(model.body_mass.sum()),
        "body_mass_kg": {model.body(i).name: float(model.body_mass[i]) for i in range(1, model.nbody)},
        "body_principal_inertia_kg_m2": {model.body(i).name: model.body_inertia[i].tolist() for i in range(1, model.nbody)},
        "zero_pose_foot_centres_m": {leg: xyz.tolist() for leg, xyz in feet.items()},
        "preview_foot_centres_m": preview_feet,
        "max_zero_pose_fk_error_m": fk_error,
        "unactuated_rollout_seconds": float(data.time), "warnings": warnings,
        "scope": "Toy geometry; preview clears the longest leg and may leave shorter legs airborne. No support-pose solver, upright task, controller, motor hardware or Isaac Lab validation.",
    }
