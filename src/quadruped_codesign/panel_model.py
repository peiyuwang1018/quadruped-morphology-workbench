"""Model creation and reproducible snapshots for the native parameter panel."""

from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import tempfile
import tomllib

import mujoco

from .generator import Morphology, Payload, generate_mjcf
from .fusion import FusionShape, generate_fusion_mjcf, evaluate_fusion
from .validation import evaluate_robot
from .visualization import preview_data


def load_morphology(path: Path) -> Morphology | FusionShape:
    with path.open("rb") as stream:
        values = tomllib.load(stream)
    if "profile" in values:
        return FusionShape.from_dict(values)
    if "payload" in values:
        if not isinstance(values["payload"], dict):
            raise ValueError("payload must be a TOML table.")
        values["payload"] = Payload(**values["payload"])
    return Morphology(**values)


def morphology_toml(shape: Morphology) -> str:
    if isinstance(shape, FusionShape):
        values = shape.to_dict()
        lines = ["# Fusion concept model. SI units: m, kg, s, rad.",
                 "# Motor positions, structure mass, density and joint limits are estimates.",
                 "# Bounds are provisional exploration ranges, not hardware limits.",
                 "# v2: coaxial hip/knee stack; axial_gap and thigh_axial_offset are mounting estimates.",
                 "# pair_half_spacing applies only to legacy v1 snapshots.",
                 "# z offset = shoulder minus hind hip; negative means shoulder lower."]
        for name, value in values.items():
            if not isinstance(value, dict):
                lines.append(f"{name} = {json.dumps(value)}")
        for name, value in values.items():
            if isinstance(value, dict):
                lines += ["", f"[{name}]"]
                lines += [f"{json.dumps(k)} = {json.dumps(v)}" for k, v in value.items()]
        return "\n".join(lines) + "\n"
    lines = ["# Saved parameter-panel model. SI units: m, kg.",
             "# Payload size is FULL dimensions; position is relative to torso."]
    for name in ("thigh_length", "shank_length", "front_thigh_length", "front_shank_length", "torso_mass"):
        lines.append(f"{name} = {getattr(shape, name)!r}")
    lines += ["", "[payload]", f"mass = {shape.payload.mass!r}",
              f"size = {list(shape.payload.size)!r}", f"position = {list(shape.payload.position)!r}", ""]
    return "\n".join(lines)


@dataclass
class Preview:
    shape: Morphology | FusionShape
    xml: str
    model: mujoco.MjModel
    data: mujoco.MjData
    report: dict

    @classmethod
    def build(cls, shape: Morphology):
        fusion = isinstance(shape, FusionShape)
        xml = generate_fusion_mjcf(shape) if fusion else generate_mjcf(**vars(shape))
        model = mujoco.MjModel.from_xml_string(xml)
        report = evaluate_fusion(model, shape) if fusion else evaluate_robot(model, steps=0)
        if not report["passed"]:
            if fusion and report["envelope_overlaps"]:
                first, second, distance = report["envelope_overlaps"][0]
                raise ValueError(f"Motor envelopes overlap: {first} / {second} ({-distance*1000:.2f} mm). Check module placement or stance.")
            if fusion and report["proxy_penetrations"]:
                first, second, distance = report["proxy_penetrations"][0]
                raise ValueError(f"Model parts overlap: {first} / {second} ({-distance*1000:.2f} mm). Check geometry or stance.")
            failed = [name for name, passed in report["checks"].items() if not passed]
            raise ValueError("Model checks failed: " + ", ".join(failed))
        return cls(shape, xml, model, preview_data(model), report)

    def save(self, parent: Path, image) -> Path:
        """Save a consistent, validated snapshot without overwriting prior experiments."""
        parent.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        destination = parent / name
        # Publish the directory only after every part has been written successfully.
        with tempfile.TemporaryDirectory(prefix=".pending-", dir=parent) as temporary:
            folder = Path(temporary)
            (folder / "config.toml").write_text(morphology_toml(self.shape), encoding="utf-8")
            (folder / "robot.xml").write_text(self.xml, encoding="utf-8")
            (folder / "validation.json").write_text(json.dumps(self.report, indent=2) + "\n", encoding="utf-8")
            image.save(folder / "preview.png")
            (folder / "README.txt").write_text(
                "Frozen morphology preview. Mass, CoM, inertia and kinematics checked.\n"
                "No controlled balance, walking, gravity-drop or Isaac Lab tests were run by the panel.\n"
                + (f"Pose: {self.shape.stance.mode}. Declared geometric foot contacts solved.\n"
                   "CoM/support overlay is render-only. A two-point support line has zero area.\n"
                   "Mounting proxies, masses and separate upright joint limits are uncalibrated.\n"
                   "12 nominal motor masses included; extra housing/driver mass only if supplied.\n"
                   "No transmission, actuator dynamics or hardware feasibility validation.\n"
                   if isinstance(self.shape, FusionShape) else "") +
                "Reopen config.toml in the panel or use generate --config. robot.xml is the corresponding MuJoCo model.\n",
                encoding="utf-8")
            folder.rename(destination)
        return destination
