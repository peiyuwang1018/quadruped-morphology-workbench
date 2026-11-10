"""Fusion reference geometry and ground-contact poses, with explicit approximations.

This is a 12-joint concept model, not a CAD export or a transmission model.
The proximal hip/knee motors are coaxial and stacked laterally. The upper link
starts at the outer motor's axial centre (plus an explicit trim). Mounting
distances, bracket masses and mechanical ROM remain approximations. No
actuator-to-joint transmission is invented here. v1 snapshots remain readable.
"""

from dataclasses import asdict, dataclass, field, fields
import itertools
import json
import math
from numbers import Real
import xml.etree.ElementTree as ET

import mujoco
import numpy as np

from .generator import Payload, _fmt
from .mass_properties import box, capsule, shift_tensor, body_tensor, inspect_mass

LEGS = ("FL", "FR", "RL", "RR")
FOOT_RADIUS, FOOT_MASS = .026, .08
UPPER_RADIUS, LOWER_RADIUS = .020, .018
PROFILE = "fusion_reference_v2"
LEGACY_PROFILE = "fusion_reference_v1"

# These are adjustable exploration windows, NOT hardware feasibility limits.
DEFAULT_BOUNDS = {
    "l_upperarm": (.295, .355), "l_lowerarm": (.310, .370),
    "l_thigh": (.310, .370), "l_shank": (.370, .430),
    "w_shoulder": (.420, .480), "w_hip": (.470, .530),
    "d_hip_shoulder": (.650, .750), "z_offset_hip_shoulder": (-.050, -.010),
    "torso_mass": (1., 40.), "link_density": (100., 8000.),
    "stance.hip_height": (.400, .650), "stance.body_pitch": (-.15, .15),
    "stance.front_width": (.400, .650), "stance.hind_width": (.450, .700),
    "stance.front_x": (-.100, .100), "stance.hind_x": (-.100, .100),
    "payload.mass": (0., 20.), "motor.extra_mass": (0., 1.),
}
UPRIGHT_BOUNDS = {
    "stance.upright_hip_height": (.40, .65),
    "stance.upright_pitch": (-1.70, -1.05),
    "stance.upright_hind_width": (.60, .80),
    "stance.upright_hind_x": (-.15, .15),
}
DEFAULT_BOUNDS.update(UPRIGHT_BOUNDS)


def finite(value, name):
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class Motor:
    mass: float = 1.4
    diameter: float = .1075
    length: float = .0435
    envelope_diameter: float = .118
    envelope_length: float = .060
    extra_mass: float = 0.0  # Unknown housing/driver/bracket mass, omitted until supplied.
    pair_half_spacing: float = .075  # v1 only: fore/aft arrangement, retained for snapshots.
    roll_inboard: float = .080  # Reduced body-side gap; saved older configs retain .100.
    pitch_outboard: float = .030
    axial_gap: float = .005  # v2: gap between 60 mm packaging envelopes; unmeasured.
    thigh_axial_offset: float = 0.0  # v2: trim from outer motor centre, outward positive.

    def __post_init__(self):
        for item in fields(self):
            finite(getattr(self, item.name), f"motor.{item.name}")
        if any(getattr(self, name) <= 0 for name in ("mass", "diameter", "length", "envelope_diameter", "envelope_length")):
            raise ValueError("Motor dimensions and mass must be positive")
        if self.extra_mass < 0 or min(self.pair_half_spacing, self.roll_inboard, self.pitch_outboard, self.axial_gap) < 0:
            raise ValueError("Motor allowances must be non-negative")
        if abs(self.thigh_axial_offset) > self.envelope_length/2:
            raise ValueError("Thigh axial trim must stay within the outer motor packaging length")
        if self.envelope_diameter < self.diameter or self.envelope_length < self.length:
            raise ValueError("Motor packaging envelope cannot be smaller than the physical module")


@dataclass(frozen=True)
class Stance:
    hip_height: float = .55  # Rear mounting-datum height, before the lateral thigh offset.
    body_pitch: float = 0.0
    front_width: float = .50  # World lateral distance between foot centres.
    hind_width: float = .55
    front_x: float = 0.0  # World-horizontal offsets from the corresponding hip projection.
    hind_x: float = 0.0
    mode: str = "quadruped"
    upright_hip_height: float = .55
    upright_pitch: float = -math.pi/2
    upright_hind_width: float = .70
    upright_hind_x: float = .05
    # Free front limbs in the upright preview, not ground-contact targets.
    front_roll: float = 0.0
    front_hip: float = .45
    front_knee: float = -1.2

    def __post_init__(self):
        if self.mode not in ("quadruped", "hind_stand"):
            raise ValueError("stance.mode must be quadruped or hind_stand")
        for name, value in vars(self).items():
            if name != "mode":
                finite(value, f"stance.{name}")

    @property
    def upright(self):
        return self.mode == "hind_stand"

    @property
    def pitch(self):
        return self.upright_pitch if self.upright else self.body_pitch

    @property
    def height(self):
        return self.upright_hip_height if self.upright else self.hip_height

    @property
    def rear_width(self):
        return self.upright_hind_width if self.upright else self.hind_width

    @property
    def rear_x(self):
        return self.upright_hind_x if self.upright else self.hind_x


@dataclass(frozen=True)
class JointLimits:
    roll: tuple = (-.52, .52)
    hip: tuple = (-1.35, 1.35)
    knee: tuple = (-2.50, -.15)  # Negative-knee branch, away from straight-leg singularity.

    def __post_init__(self):
        for item in fields(self):
            value = getattr(self, item.name)
            if not isinstance(value, (list, tuple)) or len(value) != 2:
                raise ValueError(f"limits.{item.name} needs [min, max]")
            for number in value:
                finite(number, f"limits.{item.name}")
            if value[0] >= value[1]:
                raise ValueError(f"limits.{item.name}: min must be less than max")
            object.__setattr__(self, item.name, tuple(value))


@dataclass(frozen=True)
class FusionShape:
    l_upperarm: float = .325
    l_lowerarm: float = .340
    l_thigh: float = .340
    l_shank: float = .400
    w_shoulder: float = .450
    w_hip: float = .500
    d_hip_shoulder: float = .700
    z_offset_hip_shoulder: float = -.030  # Confirmed: shoulder is LOWER than rear hip.
    torso_mass: float = 8.0  # Uncalibrated structure estimate, excludes all motors and legs.
    link_density: float = 1000.0  # Surrogate link density, not solid metal/CAD mass.
    motor: Motor = field(default_factory=Motor)
    stance: Stance = field(default_factory=Stance)
    limits: JointLimits = field(default_factory=JointLimits)
    # Explicit concept ROM for upright inspection, NOT a measured hardware limit.
    upright_limits: JointLimits = field(default_factory=lambda: JointLimits(hip=(-1.35, 2.80)))
    payload: Payload = field(default_factory=lambda: Payload(mass=2))
    bounds: dict = field(default_factory=lambda: dict(DEFAULT_BOUNDS))
    profile: str = PROFILE

    def __post_init__(self):
        for item in fields(self):
            if item.name not in ("motor", "stance", "limits", "upright_limits", "payload", "bounds", "profile"):
                finite(getattr(self, item.name), item.name)
        if self.profile not in (PROFILE, LEGACY_PROFILE):
            raise ValueError("Unknown Fusion profile")
        for key, cls in (("motor", Motor), ("stance", Stance), ("limits", JointLimits), ("upright_limits", JointLimits), ("payload", Payload)):
            if not isinstance(getattr(self, key), cls):
                raise ValueError(f"{key} must be {cls.__name__}")
        if any(getattr(self, name) <= 0 for name in ("l_upperarm", "l_lowerarm", "l_thigh", "l_shank", "w_shoulder", "w_hip", "d_hip_shoulder", "torso_mass", "link_density")):
            raise ValueError("Structural dimensions, mass and density must be positive")
        if (min(self.stance.front_width, self.stance.hind_width, self.stance.upright_hind_width) <= 0
                or min(self.stance.hip_height, self.stance.upright_hip_height) <= FOOT_RADIUS):
            raise ValueError("Stance width and hip height must be positive and above the feet")
        if (not isinstance(self.bounds, dict) or set(self.bounds)-set(DEFAULT_BOUNDS)
                or (set(DEFAULT_BOUNDS)-set(self.bounds))-set(UPRIGHT_BOUNDS)):
            raise ValueError("bounds must declare every supported exploration parameter")
        # Older snapshots did not have upright settings. Fill only those additions.
        bounds = {**UPRIGHT_BOUNDS, **self.bounds}
        normalized = {}
        for name, interval in bounds.items():
            if not isinstance(interval, (tuple, list)) or len(interval) != 2:
                raise ValueError(f"bounds.{name} needs [min, max]")
            low, high = interval
            finite(low, name); finite(high, name)
            if low >= high:
                raise ValueError(f"bounds.{name}: min must be less than max")
            value = self
            for part in name.split("."):
                value = getattr(value, part)
            if not low - 1e-12 <= value <= high + 1e-12:
                raise ValueError(f"{name}={value:g} is outside the provisional exploration range [{low:g}, {high:g}]. Edit bounds in the config to change the study domain.")
            normalized[name] = (low, high)
        object.__setattr__(self, "bounds", normalized)

    @property
    def active_limits(self):
        return self.upright_limits if self.stance.upright else self.limits

    def lengths(self, leg):
        return ((self.l_upperarm, self.l_lowerarm) if leg.startswith("F") else (self.l_thigh, self.l_shank))

    def anchor(self, leg):
        front, side = leg.startswith("F"), (1 if leg.endswith("L") else -1)
        sign = 1 if front else -1
        return np.array([sign * self.d_hip_shoulder / 2,
                         side * (self.w_shoulder if front else self.w_hip) / 2,
                         sign * self.z_offset_hip_shoulder / 2])

    def to_dict(self):
        return {"profile": self.profile, **asdict(self)}

    def pitch_motor_positions(self, side):
        motor = self.motor
        if self.profile == LEGACY_PROFILE:
            return (np.array([motor.pair_half_spacing, side*motor.pitch_outboard, 0.]),
                    np.array([-motor.pair_half_spacing, side*motor.pitch_outboard, 0.]))
        inner = motor.pitch_outboard
        outer = inner + motor.envelope_length + motor.axial_gap
        return np.array([0., side*inner, 0.]), np.array([0., side*outer, 0.])

    def thigh_origin(self, side):
        if self.profile == LEGACY_PROFILE:
            return np.zeros(3)
        outer = self.pitch_motor_positions(side)[1]
        return outer + [0., side*self.motor.thigh_axial_offset, 0.]

    @classmethod
    def from_dict(cls, values):
        values = dict(values)
        if values.get("profile") not in (PROFILE, LEGACY_PROFILE):
            raise ValueError("Unknown Fusion profile")
        for name, constructor in (("motor", Motor), ("stance", Stance), ("limits", JointLimits), ("upright_limits", JointLimits), ("payload", Payload)):
            if name in values:
                values[name] = constructor(**values[name])
        return cls(**values)


def rotation_y(angle):
    c, s = math.cos(angle), math.sin(angle)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def solve_pose(shape):
    """Analytic 3R IK including the roll-to-thigh lateral offset."""
    stance = shape.stance
    rotation = rotation_y(stance.pitch)
    rear = rotation @ shape.anchor("RL")
    base = np.array([0., 0., stance.height - rear[2]])
    joints, targets = {}, {}
    for leg in LEGS:
        origin = base + rotation @ shape.anchor(leg)
        front, side = leg.startswith("F"), (1 if leg.endswith("L") else -1)
        if front and stance.upright:
            q = (side*stance.front_roll, stance.front_hip, stance.front_knee)
            for name, value in zip(("roll", "hip", "knee"), q):
                low, high = getattr(shape.active_limits, name)
                if not low <= value <= high:
                    raise ValueError(f"{leg}: free front {name} pose exceeds the upright concept limits")
            joints[leg] = q
            continue
        target = np.array([origin[0] + (stance.front_x if front else stance.rear_x),
                           side * (stance.front_width if front else stance.rear_width) / 2, FOOT_RADIUS])
        local = rotation.T @ (target - origin)
        if local[2] >= 0 and not stance.upright:
            raise ValueError(f"{leg}: foot must be below its hip in this stance model")
        offset = shape.thigh_origin(side)[1]
        lateral_reach = math.hypot(local[1], local[2])
        if lateral_reach <= abs(offset) + .001:
            raise ValueError(f"{leg}: foot target is inside the lateral-offset reach boundary. Change the foot target or mounting offset.")
        # p_yz = Rx(roll) @ [lateral_offset, -planar_drop]. The
        # offset rotates with the carrier; it is not a world-y translation.
        planar_drop = math.sqrt(lateral_reach**2 - offset**2)
        roll = math.atan2(local[1], -local[2]) - math.atan2(offset, planar_drop)
        planar_z = -planar_drop
        reach = math.hypot(local[0], planar_z)
        upper, lower = shape.lengths(leg)
        if not abs(upper-lower) + .001 < reach < upper+lower - .001:
            raise ValueError(f"{leg}: requested foot is unreachable or too close to a singular reach boundary ({reach*1000:.1f} mm). Change hip height or foot target; link lengths were not altered.")
        knee = -math.acos(np.clip((reach**2-upper**2-lower**2)/(2*upper*lower), -1, 1))
        hip = math.atan2(-local[0], -planar_z) - math.atan2(lower*math.sin(knee), upper+lower*math.cos(knee))
        q = (roll, hip, knee)
        for name, value in zip(("roll", "hip", "knee"), q):
            low, high = getattr(shape.active_limits, name)
            if not low <= value <= high:
                raise ValueError(f"{leg} {name}: {math.degrees(value):.1f} deg exceeds provisional joint limits [{math.degrees(low):.1f}, {math.degrees(high):.1f}] deg")
        joints[leg], targets[leg] = q, target
    quaternion = [math.cos(stance.pitch/2), 0, math.sin(stance.pitch/2), 0]
    return [*base, *quaternion, *(q for leg in LEGS for q in joints[leg])], targets, joints


def cylinder_properties(mass, radius, length, axis):
    transverse = mass*(3*radius**2+length**2)/12
    diagonal = [transverse]*3
    diagonal[axis] = mass*radius**2/2
    return np.diag(diagonal)


def expected_properties(shape):
    """Independent primitive formulas, without reading compiled body mass/inertia."""
    expected = {"torso": box(shape.torso_mass, (shape.d_hip_shoulder+.10, min(shape.w_shoulder, shape.w_hip)-.05, .12))}
    motor = shape.motor
    m = motor.mass + motor.extra_mass
    for leg in LEGS:
        side = 1 if leg.endswith("L") else -1
        expected[f"{leg}_roll_motor"] = (m, np.zeros(3), cylinder_properties(m, motor.diameter/2, motor.length, 0))
        if shape.profile == LEGACY_PROFILE:
            centre = np.array([0., side*motor.pitch_outboard, 0.])
            half_separation = np.array([motor.pair_half_spacing, 0., 0.])
        else:
            spacing = motor.envelope_length + motor.axial_gap
            centre = np.array([0., side*(motor.pitch_outboard+spacing/2), 0.])
            half_separation = np.array([0., spacing/2, 0.])
        expected[f"{leg}_carrier"] = (2*m, centre,
            2*cylinder_properties(m, motor.diameter/2, motor.length, 1) +
            2*shift_tensor(m, half_separation))
        upper, lower = shape.lengths(leg)
        mass, com, inertia = capsule(upper, UPPER_RADIUS)
        expected[f"{leg}_thigh"] = (mass*shape.link_density/1000, com, inertia*shape.link_density/1000)
        mass, com, inertia = capsule(lower, LOWER_RADIUS)
        mass *= shape.link_density/1000; inertia *= shape.link_density/1000
        foot_com = np.array([0., 0., -lower])
        total = mass+FOOT_MASS
        centre = (mass*com+FOOT_MASS*foot_com)/total
        tensor = inertia + shift_tensor(mass, com-centre)
        tensor += 2/5*FOOT_MASS*FOOT_RADIUS**2*np.eye(3) + shift_tensor(FOOT_MASS, foot_com-centre)
        expected[f"{leg}_shank"] = (total, centre, tensor)
    if shape.payload.mass:
        expected["payload"] = box(shape.payload.mass, shape.payload.size)
    return expected


def generate_fusion_mjcf(shape):
    qpos, targets, _ = solve_pose(shape)
    root = ET.Element("mujoco", model="fusion_reference_quadruped")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="auto", autolimits="true", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.002", integrator="implicitfast", gravity="0 0 -9.81")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "text", name="fusion_config", data=json.dumps(shape.to_dict()))
    ET.SubElement(custom, "numeric", name="morphology", data=_fmt(shape.l_thigh, shape.l_shank))
    ET.SubElement(custom, "numeric", name="front_morphology", data=_fmt(shape.l_upperarm, shape.l_lowerarm))
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.6", limited="true")
    ET.SubElement(default, "geom", friction="0.8 0.005 0.0001", condim="3")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker", rgb1="0.22 0.26 0.30", rgb2="0.28 0.32 0.36", width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="4 4")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", material="floor")
    torso = ET.SubElement(world, "body", name="torso", pos=_fmt(*qpos[:3]))
    ET.SubElement(torso, "freejoint", name="root")
    torso_inertia = expected_properties(shape)["torso"][2]
    ET.SubElement(torso, "inertial", pos="0 0 0", mass=_fmt(shape.torso_mass), diaginertia=_fmt(*np.diag(torso_inertia)))
    # Visible frame rails are proxies; the structure's declared box inertia is
    # independent of their cosmetic thickness. Motor bodies carry their own mass.
    for side in (1, -1):
        a = shape.anchor("RL" if side == 1 else "RR") + [0, -side*.12, 0]
        b = shape.anchor("FL" if side == 1 else "FR") + [0, -side*.12, 0]
        ET.SubElement(torso, "geom", name=f"rail_{side}", type="capsule", fromto=_fmt(*a, *b), size="0.014", mass="0", rgba="0.33 0.37 0.40 1")
    for group in ("F", "R"):
        a = shape.anchor(group+"L") + [0, -.10, 0]
        b = shape.anchor(group+"R") + [0, .10, 0]
        ET.SubElement(torso, "geom", name=f"crossbar_{group}", type="capsule", fromto=_fmt(*a, *b), size="0.018", mass="0", rgba="0.80 0.84 0.85 1")
    ET.SubElement(torso, "geom", name="payload_deck", type="box", size="0.28 0.13 0.008", pos="0 0 0.052", mass="0", rgba="0.72 0.77 0.8 0.45")
    ET.SubElement(torso, "site", name="front_marker", pos=_fmt(shape.d_hip_shoulder/2, 0, .05), size="0.02", rgba="1 .45 .15 1")
    motor = shape.motor

    def module(parent, name, position, axis, color="0.75 0.78 0.80 1"):
        quaternion = (math.sqrt(.5), 0, math.sqrt(.5), 0) if axis == 0 else (math.sqrt(.5), math.sqrt(.5), 0, 0)
        attributes = dict(type="cylinder", pos=_fmt(*position), quat=_fmt(*quaternion))
        ET.SubElement(parent, "geom", name=name, **attributes, size=_fmt(motor.diameter/2, motor.length/2),
                      mass=_fmt(motor.mass+motor.extra_mass), rgba=color)
        ET.SubElement(parent, "geom", name=name+"_envelope", **attributes,
                      size=_fmt(motor.envelope_diameter/2, motor.envelope_length/2), mass="0", contype="0", conaffinity="0", group="2", rgba="0.95 0.66 0.25 0.18")

    for leg in LEGS:
        side = 1 if leg.endswith("L") else -1
        anchor = shape.anchor(leg)
        fixed = ET.SubElement(torso, "body", name=f"{leg}_roll_motor", pos=_fmt(*(anchor+[0, -side*motor.roll_inboard, 0])))
        module(fixed, f"{leg}_roll_module", (0, 0, 0), 0)
        carrier = ET.SubElement(torso, "body", name=f"{leg}_carrier", pos=_fmt(*anchor))
        ET.SubElement(carrier, "joint", name=f"{leg}_roll", axis="1 0 0", range=_fmt(*shape.active_limits.roll))
        for suffix, position in zip(("hip", "remote"), shape.pitch_motor_positions(side)):
            color = ".40 .52 .65 1" if suffix == "remote" and shape.profile == PROFILE else ".75 .78 .80 1"
            module(carrier, f"{leg}_{suffix}_module", position, 1, color)
        upper, lower = shape.lengths(leg)
        thigh = ET.SubElement(carrier, "body", name=f"{leg}_thigh", pos=_fmt(*shape.thigh_origin(side)))
        ET.SubElement(thigh, "joint", name=f"{leg}_hip", axis="0 1 0", range=_fmt(*shape.active_limits.hip))
        ET.SubElement(thigh, "geom", name=f"{leg}_thigh_geom", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -upper), size=_fmt(UPPER_RADIUS), density=_fmt(shape.link_density), rgba=".22 .49 .91 1")
        shank = ET.SubElement(thigh, "body", name=f"{leg}_shank", pos=_fmt(0, 0, -upper))
        ET.SubElement(shank, "joint", name=f"{leg}_knee", axis="0 1 0", range=_fmt(*shape.active_limits.knee))
        ET.SubElement(shank, "geom", name=f"{leg}_shank_geom", type="capsule", fromto=_fmt(0, 0, 0, 0, 0, -lower), size=_fmt(LOWER_RADIUS), density=_fmt(shape.link_density), rgba=".12 .71 .64 1")
        ET.SubElement(shank, "geom", name=f"{leg}_foot_geom", type="sphere", pos=_fmt(0, 0, -lower), size=_fmt(FOOT_RADIUS), mass=_fmt(FOOT_MASS), rgba="1 .58 .20 1")
        ET.SubElement(shank, "site", name=f"{leg}_foot", pos=_fmt(0, 0, -lower), size=".006", rgba=".9 .2 .2 1")
        # Ground targets are visible, massless markers; they are not constraints.
        if leg in targets:
            ET.SubElement(world, "site", name=f"{leg}_target", pos=_fmt(targets[leg][0], targets[leg][1], .001), type="cylinder", size=".033 .001", rgba=".3 .95 .6 .5")
    if shape.payload.mass:
        cargo = ET.SubElement(torso, "body", name="payload", pos=_fmt(*shape.payload.position))
        ET.SubElement(cargo, "geom", name="payload_geom", type="box", size=_fmt(*(v/2 for v in shape.payload.size)), mass=_fmt(shape.payload.mass), rgba=".74 .39 .83 1")
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="preview", qpos=_fmt(*qpos))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode")+"\n"


def is_fusion_model(model):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "fusion_config") >= 0


def shape_from_model(model):
    index = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_TEXT, "fusion_config")
    if index < 0:
        raise ValueError("Not a Fusion reference model")
    start, size = int(model.text_adr[index]), int(model.text_size[index])
    raw = bytes(model.text_data[start:start+size]).rstrip(b"\0").decode()
    return FusionShape.from_dict(json.loads(raw))


def evaluate_fusion(model, shape=None):
    shape = shape or shape_from_model(model)
    qpos, targets, joints = solve_pose(shape)
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, model.key("preview").id)
    mujoco.mj_forward(model, data)
    errors = {leg: float(np.linalg.norm(data.site(leg+"_foot").xpos-targets[leg])) for leg in targets}
    heights = {leg: float(data.site(leg+"_foot").xpos[2]-FOOT_RADIUS) for leg in LEGS}
    mass_errors, com_errors, inertia_errors = [], [], []
    for name, (mass, com, inertia) in expected_properties(shape).items():
        body = model.body(name)
        mass_errors.append(abs(float(body.mass[0])-mass))
        com_errors.append(float(np.max(np.abs(body.ipos-com))))
        inertia_errors.append(float(np.max(np.abs(body_tensor(model, body.id)-inertia))))
    envelopes = [i for i in range(model.ngeom) if model.geom(i).name.endswith("_envelope")]
    overlaps = []
    for a, b in itertools.combinations(envelopes, 2):
        distance = mujoco.mj_geomDistance(model, data, a, b, .005, None)
        if distance < -1e-5:
            overlaps.append([model.geom(a).name, model.geom(b).name, float(distance)])
    penetrations = [[model.geom(c.geom1).name, model.geom(c.geom2).name, float(c.dist)]
                    for c in data.contact if c.dist < -1e-5]
    checks = {
        "twelve_hinges_floating_base": bool(model.nq == 19 and model.nv == 18 and model.njnt == 13),
        "transmission_actuation_explicitly_unmodeled": model.nu == 0,
        "foot_targets_reached": max(errors.values()) < 1e-8,
        "declared_foot_surfaces_on_ground": max(abs(heights[leg]) for leg in targets) < 1e-8,
        "free_front_feet_clear_ground": not shape.stance.upright or min(heights['FL'],heights['FR']) > .01,
        "hip_height_matches_target": abs(float((data.body("RL_carrier").xpos[2]+data.body("RR_carrier").xpos[2])/2)-shape.stance.height) < 1e-8,
        "analytic_body_masses": max(mass_errors) < 1e-8,
        "analytic_local_com": max(com_errors) < 1e-8,
        "analytic_inertia_tensors": max(inertia_errors) < 1e-8,
        "motor_envelopes_do_not_overlap_each_other": not overlaps,
        "active_proxy_contacts_not_penetrating": not penetrations,
        "keyframe_matches_solved_pose": bool(np.allclose(data.qpos, qpos, rtol=0, atol=1e-10)),
    }
    if shape.profile == PROFILE:
        axis_errors, spacing_errors, thigh_errors = [], [], []
        for leg in LEGS:
            inner, outer = data.geom(leg+'_hip_module'), data.geom(leg+'_remote_module')
            inner_axis, outer_axis = inner.xmat.reshape(3,3)[:,2], outer.xmat.reshape(3,3)[:,2]
            delta = outer.xpos-inner.xpos
            axis_errors.extend((np.linalg.norm(np.cross(inner_axis, outer_axis)),
                                np.linalg.norm(np.cross(delta, inner_axis))))
            spacing_errors.append(abs(np.linalg.norm(delta)-shape.motor.envelope_length-shape.motor.axial_gap))
            side = 1 if leg.endswith('L') else -1
            trim = data.body(leg+'_carrier').xmat.reshape(3,3) @ np.array([0., side*shape.motor.thigh_axial_offset, 0.])
            thigh_errors.append(np.linalg.norm(data.body(leg+'_thigh').xpos-outer.xpos-trim))
        checks.update(coaxial_motor_axes=bool(max(axis_errors) < 1e-9),
                      coaxial_motor_spacing=bool(max(spacing_errors) < 1e-9),
                      thigh_origin_at_outer_motor_plus_trim=bool(max(thigh_errors) < 1e-9))
    mass = inspect_mass(model, data)
    from .support import inspect_support
    support = inspect_support(model, data, tuple(targets))
    return {
        "passed": all(checks.values()), "checks": checks, "profile": shape.profile,
        "mass_properties_preview_pose": mass, "robot_mass_kg": mass["total_mass_kg"],
        "stance": {"mode": shape.stance.mode, "active_joint_limits_rad": asdict(shape.active_limits),
                   "target_hip_height_m": shape.stance.height, "foot_targets_world_m": {k:v.tolist() for k,v in targets.items()},
                   "foot_error_m": errors, "foot_surface_height_m": heights, "joint_angles_rad": joints},
        "mass_budget_kg": {"nominal_motor_modules": 12*shape.motor.mass, "extra_module_mass": 12*shape.motor.extra_mass,
                           "structure_estimate": shape.torso_mass, "payload": shape.payload.mass,
                           "links_and_feet_estimate": mass["total_mass_kg"]-12*(shape.motor.mass+shape.motor.extra_mass)-shape.torso_mass-shape.payload.mass},
        "analytic_max_errors": {"mass_kg": max(mass_errors), "com_m": max(com_errors), "inertia_kg_m2": max(inertia_errors)},
        "envelope_overlaps": overlaps, "proxy_penetrations": penetrations,
        "support": support,
        "mounting": {"layout": "coaxial" if shape.profile == PROFILE else "legacy_fore_aft",
                     "thigh_origins_carrier_m": {leg:shape.thigh_origin(1 if leg.endswith('L') else -1).tolist() for leg in LEGS},
                     "thigh_origins_world_m": {leg:data.body(leg+'_thigh').xpos.tolist() for leg in LEGS},
                     "hip_height_datum": "rear carrier anchor, before lateral thigh offset"},
        "scope": "Kinematic ground-contact pose only. See mounting.layout for motor arrangement. Axial gap, anchor-to-motor distances and upper-link trim are explicit mounting estimates; the v2 thigh offset is included in IK. Structural masses and joint limits are uncalibrated. Collision checks cover motor-envelope pairs and active proxy contacts in this pose only; parent-child joint interfaces and real CAD interference are not certified. No transmission ratios, rotor armature, actuator dynamics, static equilibrium, controller or Isaac Lab validation.",
        "unactuated_rollout_seconds": 0.0,
    }
