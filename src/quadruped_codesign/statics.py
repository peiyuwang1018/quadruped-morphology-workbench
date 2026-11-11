"""Frozen-pose, gravity-only joint loads under an explicit contact-force rule.

No simulator contact reactions or actuator model are used. A failed balance
check is a reportable design result, not a failure to build the preview.
"""

import mujoco
import numpy as np

from .support import TOLERANCE


ASSUMPTIONS = (
    "Level ground; zero velocity and acceleration; gravity only. Ideal point feet "
    "supply nonnegative vertical forces, zero horizontal force and zero contact moment. "
    "Left/right feet share each pair load equally. Fore/hind sharing follows CoM x. "
    "Lateral CoM offsets require a different sharing rule and are not evaluated."
)
SCOPE = (
    "Ideal joint-output torques, not motor-shaft torques or capacity checks. "
    "No transmission coupling, efficiency, rotor effects, friction loss, compliance, "
    "external task wrench or feedback controller. Masses and joint limits remain "
    "uncalibrated. Two point feet have zero support area; balance within numerical "
    "tolerance does not establish stable standing."
)


def paired_vertical_loads(weight, com, contacts):
    """Solve equal-L/R pair loads and retain signed trial loads on failure."""
    com = np.asarray(com, dtype=float)
    points = {name: np.asarray(point, dtype=float) for name, point in contacts.items()}
    result = {"status": "outside_assumptions", "reason": "", "trial_forces_world_n": {},
              "force_residual_world_n": None, "moment_residual_about_com_world_nm": None,
              "force_tolerance_n": max(1e-8, weight*1e-8),
              "moment_tolerance_nm": weight*TOLERANCE,
              "position_tolerance_m": TOLERANCE}
    if set(points) not in ({"FL", "FR", "RL", "RR"}, {"RL", "RR"}):
        result["reason"] = "This rule requires four feet or the two hind feet."
        return result
    if weight <= 0 or not np.isfinite(weight) or not np.isfinite(com).all() or any(not np.isfinite(p).all() for p in points.values()):
        result["reason"] = "Positive finite weight and finite positions are required."
        return result
    pairs = ("F", "R") if "FL" in points else ("R",)
    centres = {}
    for pair in pairs:
        left, right = points[pair+"L"], points[pair+"R"]
        if abs(left[0]-right[0]) > 1e-8 or max(abs(left[2]), abs(right[2])) > 1e-6 or left[1]-right[1] <= 1e-8:
            result["reason"] = "Each pair must have level, laterally separated feet at the same world x."
            return result
        centres[pair] = (left+right)/2
    if "F" in centres and abs(centres["F"][1]-centres["R"][1]) > 1e-8:
        result["reason"] = "Front and hind pairs must share a lateral centreline."
        return result
    if abs(com[1]-centres["R"][1]) > TOLERANCE:
        result["reason"] = "Lateral CoM offset needs unequal left/right loads; outside this equal-sharing baseline."
        return result
    if "F" in centres:
        span = centres["F"][0]-centres["R"][0]
        if abs(span) < 1e-8:
            result["reason"] = "Front/hind x separation is too small for this pair-load rule."
            return result
        front = weight*(com[0]-centres["R"][0])/span
        pair_loads = {"F": front, "R": weight-front}
    else:
        pair_loads = {"R": weight}
    forces = {name: np.array([0., 0., pair_loads[name[0]]/2]) for name in points}
    # Only clamp round-off at zero, never conceal a required pulling contact.
    for force in forces.values():
        if -result["force_tolerance_n"] <= force[2] < 0:
            force[2] = 0.
    force_residual = np.sum(list(forces.values()), axis=0) + [0., 0., -weight]
    moment_residual = np.sum([np.cross(points[name]-com, force) for name, force in forces.items()], axis=0)
    result.update(trial_forces_world_n={name: value.tolist() for name, value in forces.items()},
                  force_residual_world_n=force_residual.tolist(),
                  moment_residual_about_com_world_nm=moment_residual.tolist())
    if any(force[2] < 0 for force in forces.values()):
        result.update(status="unbalanced", reason="CoM lies outside the fore/hind support span; a foot pair would have to pull on the ground.")
    elif np.linalg.norm(force_residual) > result["force_tolerance_n"] or np.linalg.norm(moment_residual) > result["moment_tolerance_nm"]:
        result.update(status="unbalanced", reason="The trial loads leave an unbalanced moment. Align the CoM projection or change the pose.")
    else:
        result.update(status="balanced", reason="Force and moment balance within the stated numerical tolerances.")
    return result


def joint_load_vectors(model, qpos, contacts, forces):
    """g(q) - sum(J_point.T @ force), with a private stationary data instance."""
    data = mujoco.MjData(model)
    data.qpos[:] = qpos
    mujoco.mj_forward(model, data)
    gravity = np.zeros(model.nv)
    mujoco.mj_rne(model, data, 0, gravity)
    contact = np.zeros(model.nv)
    jacobian = np.zeros((3, model.nv))
    for leg, point in contacts.items():
        body_id = int(model.site(leg+"_foot").bodyid[0])
        mujoco.mj_jac(model, data, jacobian, None, np.asarray(point, dtype=float), body_id)
        contact += jacobian.T @ np.asarray(forces[leg])
    return gravity, contact, gravity-contact


def evaluate_statics(model, data, support=None):
    report = {"schema_version": 1, "status": "outside_assumptions", "reason": "",
              "assumptions": ASSUMPTIONS, "scope": SCOPE,
              "rule": "equal_left_right_vertical_pair_loads",
              "equilibrium_within_tolerance": False, "controlled_stability_verified": False,
              "joint_torques": [], "trial_forces_world_n": {}, "checks": {}}
    if support is None:
        report["reason"] = "No declared ground-contact pose. Use a Fusion four-foot or hind-leg upright profile."
        return report
    if abs(model.opt.gravity[0])+abs(model.opt.gravity[1]) > 1e-12 or model.opt.gravity[2] >= 0:
        report["reason"] = "This baseline requires downward world-z gravity."
        return report
    free = np.flatnonzero(model.jnt_type == mujoco.mjtJoint.mjJNT_FREE)
    if len(free) != 1 or any(int(t) not in (int(mujoco.mjtJoint.mjJNT_FREE), int(mujoco.mjtJoint.mjJNT_HINGE)) for t in model.jnt_type):
        report["reason"] = "This baseline requires one floating base and hinge joints."
        return report
    if np.any(model.jnt_stiffness) or np.any(model.dof_frictionloss) or np.any(model.body_gravcomp) or model.ntendon or model.neq:
        report["reason"] = "Passive springs, friction loss, gravity compensation, tendons or equality constraints need a separate force model."
        return report
    # Recompute the pose rather than accepting simulator contact forces or stale CoM.
    stationary = mujoco.MjData(model)
    stationary.qpos[:] = data.qpos
    mujoco.mj_forward(model, stationary)
    mass = float(model.body_mass.sum())
    com = (model.body_mass[:, None]*stationary.xipos).sum(axis=0)/mass
    weight = mass*-float(model.opt.gravity[2])
    contacts = {}
    for leg in support["contact_legs"]:
        centre = stationary.site(leg+"_foot").xpos
        radius = float(model.geom(leg+"_foot_geom").size[0])
        if abs(centre[2]-radius) > 1e-6:
            report["reason"] = "A declared supporting foot is not on the ground."
            return report
        contacts[leg] = [float(centre[0]), float(centre[1]), 0.]
    report.update(weight_n=weight, com_world_m=com.tolist(), contacts_world_m=contacts,
                  gravity_world_m_s2=model.opt.gravity.tolist())
    report.update(paired_vertical_loads(weight, com, contacts))
    if not report["trial_forces_world_n"]:
        return report
    gravity, contact, required = joint_load_vectors(model, data.qpos, contacts, report["trial_forces_world_n"])
    start = int(model.jnt_dofadr[free[0]])
    base = required[start:start+6]
    report["base_required_wrench"] = {"force_world_n": base[:3].tolist(),
                                      "moment_base_axes_nm": base[3:].tolist(),
                                      "about": model.body(int(model.jnt_bodyid[free[0]])).name + " origin"}
    checks = {
        "nonnegative_reactions": all(f[2] >= 0 for f in report["trial_forces_world_n"].values()),
        "world_force_balance": bool(np.linalg.norm(report["force_residual_world_n"]) <= report["force_tolerance_n"]),
        "world_moment_balance": bool(np.linalg.norm(report["moment_residual_about_com_world_nm"]) <= report["moment_tolerance_nm"]),
        "floating_base_force_balance": bool(np.linalg.norm(base[:3]) <= report["force_tolerance_n"]),
        "floating_base_moment_balance": bool(np.linalg.norm(base[3:]) <= report["moment_tolerance_nm"]),
    }
    report["checks"] = checks
    balanced = report["status"] == "balanced" and all(checks.values())
    report["equilibrium_within_tolerance"] = balanced
    if not balanced:
        if report["status"] == "balanced":
            report.update(status="unbalanced", reason="Floating-base equilibrium check failed; joint torques withheld.")
        return report
    for j in range(model.njnt):
        if model.jnt_type[j] != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        dof = int(model.jnt_dofadr[j])
        report["joint_torques"].append({"joint": model.joint(j).name,
            "body": model.body(int(model.jnt_bodyid[j])).name,
            "gravity_only_nm": float(gravity[dof]), "contact_generalized_nm": float(contact[dof]),
            "required_joint_output_nm": float(required[dof])})
    largest = max(report["joint_torques"], key=lambda row: abs(row["required_joint_output_nm"]))
    report["largest_absolute_joint_torque"] = {"joint": largest["joint"], "nm": abs(largest["required_joint_output_nm"])}
    return report
