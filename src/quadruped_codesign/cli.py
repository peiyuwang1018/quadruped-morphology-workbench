"""Small command-line entry point, also usable with macOS mjpython."""

import argparse
from dataclasses import fields, replace
import json
from pathlib import Path
import tomllib

import mujoco

from .generator import Morphology, Payload, write_robot
from .validation import evaluate_robot


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate, inspect and compare a simple quadruped")
    commands = parser.add_subparsers(dest="command", required=True)
    generate = commands.add_parser("generate", help="Generate from two lengths or a TOML config")
    generate.add_argument("--thigh", type=float)
    generate.add_argument("--shank", type=float)
    generate.add_argument("--front-thigh", type=float)
    generate.add_argument("--front-shank", type=float)
    generate.add_argument("--torso-mass", type=float)
    generate.add_argument("--config", type=Path)
    generate.add_argument("--output", type=Path, default=Path("outputs/robot.xml"))
    demo = commands.add_parser("demo", help="Generate and validate two morphologies")
    demo.add_argument("--output-dir", type=Path, default=Path("outputs"))
    demo.add_argument("--render", action="store_true", help="Also render comparison.png (requires graphics)")
    study = commands.add_parser("study", help="Compare front-only length and payload changes at a shared pose")
    study.add_argument("--output-dir", type=Path, default=Path("outputs/mass-study"))
    study.add_argument("--render", action="store_true")
    panel = commands.add_parser("panel", help="Native parameter sliders and a live MuJoCo preview")
    panel.add_argument("--config", type=Path)
    panel.add_argument("--output-dir", type=Path, default=Path("outputs/panel"))
    check = commands.add_parser("check", help="Validate a generated model")
    check.add_argument("model", type=Path)
    check.add_argument("--steps", type=int, help="Legacy drop steps (default 500); Fusion supports pose checks only (0)")
    check.add_argument("--output", type=Path)
    viewer = commands.add_parser("view", help="Frozen inspection pose; use mjpython on macOS")
    viewer.add_argument("model", type=Path)
    viewer.add_argument("--simulate", action="store_true", help="Uncontrolled gravity drop, not standing/walking")
    viewer.add_argument("--seconds", type=float, help="Close automatically after this many seconds")
    args = parser.parse_args()
    try:
        if args.command == "generate":
            params = {}
            if args.config:
                with args.config.open("rb") as stream:
                    params = tomllib.load(stream)
            if "profile" in params:
                from .fusion import FusionShape
                from .panel_model import Preview
                shape = FusionShape.from_dict(params)
                overrides = {parameter: getattr(args, argument) for argument, parameter in (
                    ("thigh", "l_thigh"), ("shank", "l_shank"),
                    ("front_thigh", "l_upperarm"), ("front_shank", "l_lowerarm"),
                    ("torso_mass", "torso_mass")) if getattr(args, argument) is not None}
                preview = Preview.build(replace(shape, **overrides))
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(preview.xml, encoding="utf-8")
                path = args.output
                print("Fusion concept: geometric foot contact checked; mounting/mass estimates and transmission remain uncalibrated.")
            else:
                unknown = set(params) - {f.name for f in fields(Morphology)}
                if unknown:
                    raise ValueError(f"Unknown configuration fields: {sorted(unknown)}")
                if "payload" in params:
                    if not isinstance(params["payload"], dict):
                        raise ValueError("payload must be a TOML table")
                    params["payload"] = Payload(**params["payload"])
                for argument, parameter in (("thigh", "thigh_length"), ("shank", "shank_length"),
                                        ("front_thigh", "front_thigh_length"), ("front_shank", "front_shank_length"),
                                        ("torso_mass", "torso_mass")):
                    if getattr(args, argument) is not None:
                        params[parameter] = getattr(args, argument)
                path = write_robot(args.output, **params)
            print(f"Generated and compiled: {path.resolve()}")
            from .mass_properties import inspect_mass
            from .visualization import preview_data
            model = mujoco.MjModel.from_xml_path(str(path.resolve()))
            mass = inspect_mass(model, preview_data(model))
            com = ", ".join(f"{1000 * value:.3f}" for value in mass["system_com_torso_frame_m"])
            print(f"Total mass: {mass['total_mass_kg']:.6f} kg; payload: {mass['payload_mass_kg']:.3f} kg")
            print(f"Frozen preview CoM in torso frame [x, y, z]: [{com}] mm")
        elif args.command == "demo":
            paths, reports = [], {}
            for name, thigh, shank in [("baseline", 0.22, 0.20), ("long_legs", 0.30, 0.28)]:
                path = write_robot(args.output_dir / f"{name}.xml", thigh, shank)
                report = evaluate_robot(mujoco.MjModel.from_xml_path(str(path.resolve())))
                paths.append(path)
                reports[name] = report
                print(f"{name}: thigh={thigh:.2f} m, shank={shank:.2f} m, mass={report['robot_mass_kg']:.3f} kg, checks={'PASS' if report['passed'] else 'FAIL'}")
            (args.output_dir / "validation.json").write_text(json.dumps(reports, indent=2) + "\n", encoding="utf-8")
            if not all(report["passed"] for report in reports.values()):
                raise SystemExit("Validation failed; see validation.json")
            if args.render:
                from .visualization import render_comparison
                print(render_comparison(paths, args.output_dir / "comparison.png").resolve())
        elif args.command == "study":
            from .study import run_study
            print(run_study(args.output_dir, render=args.render).resolve())
        elif args.command == "check":
            from .fusion import is_fusion_model, evaluate_fusion
            model = mujoco.MjModel.from_xml_path(str(args.model.resolve()))
            if is_fusion_model(model):
                if args.steps not in (None, 0):
                    raise ValueError("Fusion concept check supports frozen poses only; transmission/control are not modeled. Use --steps 0.")
                report = evaluate_fusion(model)
            else:
                report = evaluate_robot(model, steps=500 if args.steps is None else args.steps)
            result = json.dumps(report, indent=2) + "\n"
            if args.output:
                args.output.parent.mkdir(parents=True, exist_ok=True)
                args.output.write_text(result, encoding="utf-8")
            print(result)
            if not report["passed"]:
                raise SystemExit(1)
        elif args.command == "panel":
            from .panel import launch
            launch(args.config, args.output_dir)
        elif args.command == "view":
            if args.seconds is not None and not (0 < args.seconds < float("inf")):
                raise ValueError("--seconds must be a positive finite number")
            from .visualization import view
            view(args.model, simulate=args.simulate, seconds=args.seconds)
    except (ValueError, TypeError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
