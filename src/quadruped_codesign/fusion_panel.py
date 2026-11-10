"""English UI units and groups for the Fusion concept profile."""
from dataclasses import replace
import math

from .fusion import FusionShape
from .generator import Payload

# label, displayed unit, SI-to-display multiplier, step
SPECS = {
    "l_upperarm": ("Front upper length", "mm", 1000, 1),
    "l_lowerarm": ("Front lower length", "mm", 1000, 1),
    "l_thigh": ("Hind upper length", "mm", 1000, 1),
    "l_shank": ("Hind lower length", "mm", 1000, 1),
    "w_shoulder": ("Shoulder spacing", "mm", 1000, 1),
    "w_hip": ("Hip spacing", "mm", 1000, 1),
    "d_hip_shoulder": ("Shoulder–hip x", "mm", 1000, 1),
    "z_offset_hip_shoulder": ("Shoulder minus hip z", "mm", 1000, 1),
    "stance.hip_height": ("Hind hip height", "mm", 1000, 1),
    "stance.body_pitch": ("Body pitch", "deg", 180/math.pi, .1),
    "stance.front_width": ("Front foot spacing", "mm", 1000, 1),
    "stance.hind_width": ("Hind foot spacing", "mm", 1000, 1),
    "stance.front_x": ("Front foot x offset", "mm", 1000, 1),
    "stance.hind_x": ("Hind foot x offset", "mm", 1000, 1),
    "torso_mass": ("Frame mass estimate", "kg", 1, .1),
    "link_density": ("Link density estimate", "kg/m³", 1, 50),
    "motor.extra_mass": ("Extra per module (est.)", "kg", 1, .05),
}

GEOMETRY = list(SPECS)[:8]
STANCE = list(SPECS)[8:14]
MASS = list(SPECS)[14:]
PAYLOAD = ["payload_mass", *[f"size_{a}" for a in "xyz"], *[f"position_{a}" for a in "xyz"]]
UPRIGHT_SPECS = {
    "stance.upright_hip_height": ("Hind datum height", "mm", 1000, 1),
    "stance.upright_pitch": ("Body pitch", "deg", 180/math.pi, .1),
    "stance.upright_hind_width": ("Hind foot spacing", "mm", 1000, 1),
    "stance.upright_hind_x": ("Hind foot x offset", "mm", 1000, 1),
}


def specs(shape):
    if shape.stance.upright:
        return {**{k:v for k,v in SPECS.items() if k not in STANCE},**UPRIGHT_SPECS}
    return SPECS


def stance_fields(shape):
    return list(UPRIGHT_SPECS) if shape.stance.upright else STANCE


def get_value(shape, name):
    for part in name.split("."):
        shape = getattr(shape, part)
    return shape


def panel_fields(shape):
    result = {key: (label, unit, *(v*factor for v in shape.bounds[key]), step)
              for key, (label, unit, factor, step) in specs(shape).items()}
    result["payload_mass"] = ("Payload mass", "kg", *shape.bounds["payload.mass"], .1)
    for key, label, low, high in (
        ("size_x", "Box length", 50, 500), ("size_y", "Box width", 40, 260),
        ("size_z", "Box height", 20, 300), ("position_x", "Fore-aft position", -250, 250),
        ("position_y", "Lateral position", -100, 100), ("position_z", "Box centre height", 70, 500)):
        result[key] = (label, "mm", low, high, 1)
    return result


def panel_values(shape):
    values = {key: get_value(shape, key)*spec[2] for key, spec in specs(shape).items()}
    values["payload_mass"] = shape.payload.mass
    for i, axis in enumerate("xyz"):
        values[f"size_{axis}"] = shape.payload.size[i]*1000
        values[f"position_{axis}"] = shape.payload.position[i]*1000
    return values


def shape_from_panel(values, original):
    checked = {}
    active_specs = specs(original)
    for key, (label, unit, low, high, step) in panel_fields(original).items():
        try:
            value = float(values[key])
        except (TypeError, ValueError):
            raise ValueError(f"{label}: enter a number.") from None
        if not math.isfinite(value) or not low-1e-6 <= value <= high+1e-6:
            raise ValueError(f"{label}: enter {low:g} to {high:g} {unit} (provisional exploration range).")
        factor = active_specs[key][2] if key in active_specs else (1000 if unit == "mm" else 1)
        checked[key] = value/factor
    direct = {key: checked[key] for key in active_specs if "." not in key}
    stance = replace(original.stance, **{key.split(".")[1]: checked[key] for key in stance_fields(original)})
    motor = replace(original.motor, extra_mass=checked["motor.extra_mass"])
    payload = Payload(checked["payload_mass"], tuple(checked[f"size_{a}"] for a in "xyz"),
                      tuple(checked[f"position_{a}"] for a in "xyz"))
    return replace(original, **direct, stance=stance, motor=motor, payload=payload)
