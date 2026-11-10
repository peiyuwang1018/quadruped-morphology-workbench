"""Generate one consistent kinematic tree from two lengths (SI units)."""

from dataclasses import dataclass, field
import math
from numbers import Real
from pathlib import Path
import xml.etree.ElementTree as ET

import mujoco

LEGS = {"FL": (0.22, 0.18), "FR": (0.22, -0.18),
        "RL": (-0.22, 0.18), "RR": (-0.22, -0.18)}
HIP_Z = -0.04
THIGH_RADIUS = 0.024
SHANK_RADIUS = 0.019
FOOT_RADIUS = 0.026
FOOT_MASS = 0.08
LINK_DENSITY = 1000.0  # Surrogate density, not a material/CAD measurement.
GROUND_CLEARANCE = 0.015
PREVIEW_HIP_ANGLE = 0.45
TORSO_SIZE = (0.56, 0.26, 0.12)  # Full dimensions, metres.


def _finite(value, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")


@dataclass(frozen=True)
class Payload:
    """Uniform box rigidly attached to torso; size is FULL dimensions in metres."""
    mass: float = 0.0  # zero removes the payload, rather than making a massless body
    size: tuple[float, float, float] = (0.20, 0.16, 0.12)
    position: tuple[float, float, float] = (0.0, 0.0, 0.12)  # centre in torso frame

    def __post_init__(self):
        _finite(self.mass, "payload.mass")
        if self.mass < 0:
            raise ValueError("payload.mass must be non-negative kg")
        for name in ("size", "position"):
            value = getattr(self, name)
            if not isinstance(value, (tuple, list)) or len(value) != 3:
                raise ValueError(f"payload.{name} must contain three numbers")
            for component in value:
                _finite(component, f"payload.{name}")
            object.__setattr__(self, name, tuple(value))
        if any(value <= 0 for value in self.size):
            raise ValueError("payload.size must contain positive FULL dimensions")
        # Parent/child collision filtering cannot detect a box embedded in torso.
        # This minimal model supports a box above the back, without rotation.
        if self.mass > 0:
            if self.position[2] - self.size[2] / 2 < TORSO_SIZE[2] / 2 - 1e-10:
                raise ValueError("payload bottom must be at or above the torso top (z=0.06 m)")
            if any(abs(self.position[i]) + self.size[i] / 2 > TORSO_SIZE[i] / 2 + 1e-10 for i in (0, 1)):
                raise ValueError("payload footprint must fit on the torso for this first version")


@dataclass(frozen=True)
class Morphology:
    # Legacy names are hind lengths; omitted front lengths inherit them.
    thigh_length: float = 0.22
    shank_length: float = 0.20
    front_thigh_length: float | None = None
    front_shank_length: float | None = None
    torso_mass: float = 8.0
    payload: Payload = field(default_factory=Payload)

    def __post_init__(self):
        for front, hind in (("front_thigh_length", "thigh_length"), ("front_shank_length", "shank_length")):
            if getattr(self, front) is None:
                object.__setattr__(self, front, getattr(self, hind))
        for name in ("thigh_length", "shank_length", "front_thigh_length", "front_shank_length"):
            value = getattr(self, name)
            if (isinstance(value, bool) or not isinstance(value, Real)
                    or not math.isfinite(value) or not 0.10 <= value <= 0.45):
                raise ValueError(f"{name} must be finite and within [0.10, 0.45] metres")
        _finite(self.torso_mass, "torso_mass")
        if self.torso_mass <= 0:
            raise ValueError("torso_mass must be positive kg")
        if not isinstance(self.payload, Payload):
            raise ValueError("payload must be a Payload instance")

    def lengths(self, leg: str) -> tuple[float, float]:
        if leg not in LEGS:
            raise ValueError(f"Unknown leg: {leg}")
        return ((self.front_thigh_length, self.front_shank_length) if leg.startswith("F")
                else (self.thigh_length, self.shank_length))


def _fmt(*values: float) -> str:
    return " ".join(f"{value:.12g}" for value in values)


def generate_mjcf(thigh_length: float = 0.22, shank_length: float = 0.20, **options) -> str:
    """Return standalone MJCF; lengths measure joint-centre distances.

    Each leg has hip pitch + knee pitch. Capsules extend by their radius
    beyond endpoints. MuJoCo recomputes mass and inertia from the geometry.
    """
    morphology = Morphology(thigh_length, shank_length, **options)
    thigh, shank = morphology.thigh_length, morphology.shank_length
    longest = max(sum(morphology.lengths(leg)) for leg in LEGS)
    root_z = longest - HIP_Z + FOOT_RADIUS + GROUND_CLEARANCE
    root = ET.Element("mujoco", model="parametric_quadruped")
    ET.SubElement(root, "compiler", angle="radian", inertiafromgeom="true", autolimits="true", fusestatic="false")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81", integrator="implicitfast")
    custom = ET.SubElement(root, "custom")
    ET.SubElement(custom, "numeric", name="morphology", data=_fmt(thigh, shank))
    ET.SubElement(custom, "numeric", name="front_morphology", data=_fmt(*morphology.lengths("FL")))
    ET.SubElement(custom, "numeric", name="torso_mass", data=_fmt(morphology.torso_mass))
    payload = morphology.payload
    ET.SubElement(custom, "numeric", name="payload_spec", data=_fmt(payload.mass, *payload.size, *payload.position))
    default = ET.SubElement(root, "default")
    ET.SubElement(default, "joint", damping="0.6", armature="0.015", limited="true")
    ET.SubElement(default, "geom", friction="0.8 0.005 0.0001", condim="3")
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1600", offheight="1000")
    ET.SubElement(visual, "headlight", diffuse="0.8 0.8 0.8", ambient="0.3 0.3 0.3")
    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "texture", name="grid", type="2d", builtin="checker",
                  rgb1="0.22 0.26 0.30", rgb2="0.28 0.32 0.36", width="512", height="512")
    ET.SubElement(asset, "material", name="floor", texture="grid", texrepeat="4 4", reflectance="0.1")
    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -2 3", dir="0 0 -1", directional="true")
    ET.SubElement(world, "geom", name="ground", type="plane", size="3 3 0.1", material="floor")
    torso = ET.SubElement(world, "body", name="torso", pos=_fmt(0, 0, root_z))
    ET.SubElement(torso, "freejoint", name="root")
    ET.SubElement(torso, "geom", name="torso_geom", type="box", size=_fmt(*(v / 2 for v in TORSO_SIZE)),
                  mass=_fmt(morphology.torso_mass), rgba="0.82 0.87 0.92 1")
    ET.SubElement(torso, "site", name="front_marker", pos="0.285 0 0.025",
                  size="0.018", rgba="1 0.46 0.16 1")
    actuators = ET.SubElement(root, "actuator")
    for leg, (x, y) in LEGS.items():
        thigh, shank = morphology.lengths(leg)
        upper = ET.SubElement(torso, "body", name=f"{leg}_thigh", pos=_fmt(x, y, HIP_Z))
        ET.SubElement(upper, "joint", name=f"{leg}_hip", type="hinge", axis="0 1 0", range="-1.2 1.2")
        ET.SubElement(upper, "geom", name=f"{leg}_thigh_geom", type="capsule",
                      fromto=_fmt(0, 0, 0, 0, 0, -thigh), size=str(THIGH_RADIUS),
                      density=str(LINK_DENSITY), rgba="0.22 0.49 0.91 1")
        lower = ET.SubElement(upper, "body", name=f"{leg}_shank", pos=_fmt(0, 0, -thigh))
        ET.SubElement(lower, "joint", name=f"{leg}_knee", type="hinge", axis="0 1 0", range="-2.4 0.2")
        ET.SubElement(lower, "geom", name=f"{leg}_shank_geom", type="capsule",
                      fromto=_fmt(0, 0, 0, 0, 0, -shank), size=str(SHANK_RADIUS),
                      density=str(LINK_DENSITY), rgba="0.12 0.71 0.64 1")
        ET.SubElement(lower, "geom", name=f"{leg}_foot_geom", type="sphere", pos=_fmt(0, 0, -shank),
                      size=str(FOOT_RADIUS), mass=str(FOOT_MASS), rgba="1 0.58 0.20 1")
        ET.SubElement(lower, "site", name=f"{leg}_foot", pos=_fmt(0, 0, -shank),
                      size="0.006", rgba="0.9 0.2 0.2 1")
        for joint in ("hip", "knee"):
            ET.SubElement(actuators, "motor", name=f"{leg}_{joint}_motor",
                          joint=f"{leg}_{joint}", gear="1", ctrllimited="true", ctrlrange="-25 25")
    if payload.mass > 0:
        cargo = ET.SubElement(torso, "body", name="payload", pos=_fmt(*payload.position))
        ET.SubElement(cargo, "geom", name="payload_geom", type="box",
                      size=_fmt(*(v / 2 for v in payload.size)), mass=_fmt(payload.mass),
                      rgba="0.74 0.39 0.83 1")
    # A bent pose for inspection only. No controller is holding this pose.
    preview_z = longest * math.cos(PREVIEW_HIP_ANGLE) - HIP_Z + FOOT_RADIUS + GROUND_CLEARANCE
    qpos = [0, 0, preview_z, 1, 0, 0, 0] + [PREVIEW_HIP_ANGLE, -2 * PREVIEW_HIP_ANGLE] * 4
    keys = ET.SubElement(root, "keyframe")
    ET.SubElement(keys, "key", name="preview", qpos=_fmt(*qpos))
    ET.indent(root, space="  ")
    return ET.tostring(root, encoding="unicode") + "\n"


def generate_robot(thigh_length: float = 0.22, shank_length: float = 0.20, **options) -> mujoco.MjModel:
    """Return a compiled MuJoCo model, ready for evaluation or simulation."""
    return mujoco.MjModel.from_xml_string(generate_mjcf(thigh_length, shank_length, **options))


def write_robot(path: str | Path, thigh_length: float = 0.22, shank_length: float = 0.20, **options) -> Path:
    """Compile before writing so invalid MJCF never replaces an output."""
    xml = generate_mjcf(thigh_length, shank_length, **options)
    mujoco.MjModel.from_xml_string(xml)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(xml, encoding="utf-8")
    return destination
