# Quadruped Workbench — detailed controls

[Main README](README.md) · [Sources and provenance](SOURCES.md)

A native Mac tool for exploring robot geometry, proximal motor mass and fixed payloads. The default Fusion reference v2 profile is a 12-joint concept model with coaxial hip/knee drive modules rendered by MuJoCo. It is not a validated CAD digital twin.

## Open and try

Close the old window after a code update, then double-click **Open Workbench.command** in Finder, or run:

```sh
cd quadruped-morphology-workbench
.venv/bin/python -m quadruped_codesign.cli panel --config configs/fusion_reference.toml
```

1. Select **Stance → Hind-leg upright** to change the robot's actual pose.
2. Observe the **gold CoM**, **pink ground projection** and **cyan support segment**. Click **Align CoM projection** to adjust the hind foot targets and rerun IK.
3. Save the candidate, or switch back to **Four-foot stance**; its earlier settings are retained.

The new upright range is an unverified concept setting (hip −77° to +160°). Two ideal point feet form a zero-area support line: alignment does not verify balance or actuator feasibility. Four-foot mode displays a polygon and signed geometric margin. Readouts now use world coordinates. **Show CoM** only toggles the render overlay. See [pose switching and CoM projection](docs/support-preview.md).

The panel uses ordinary Python/Tk. The separate MuJoCo `view` command uses `mjpython` on macOS. No new dependencies are needed.

## Controls and definitions

| Tab | Parameters |
|---|---|
| Geometry | Independent front/hind upper/lower lengths; shoulder/hip lateral spacings; longitudinal spacing; shoulder-minus-hip height offset |
| Stance | Hind hip height; torso pitch; front/hind foot spacings and longitudinal foot offsets |
| Mass | Frame mass estimate; link surrogate density; estimated extra mass per module |
| Payload | Independent fixed box mass, full dimensions and centre position |

Distances in the Fusion panel use **mm**, angles **degrees** and masses **kg**. Saved configurations use SI units: metres, radians, kilograms. The body frame is +x forward, +y left, +z up, with origin midway between the shoulder/hip anchor groups. CoM readouts and markers use the world frame; exports retain both world and torso-frame CoM.

Shoulder-minus-hip z is **−30 mm**: the shoulders are lower. Positive torso pitch lowers the front. Hip height refers to the rear mounting datum before the lateral upper-link offset, not the body bottom or actual upper-link start after roll. Foot x offsets are measured from each mounting datum's world-horizontal projection; widths are full left–right foot-centre distances.

The stance solver changes joint angles and retains the requested dimensions. Upright mode solves only the hind contacts and keeps a declared free front-limb pose. Spherical feet have radius 26 mm. Green ground discs show foot targets. Geometric contact does not imply standing equilibrium or a controller.

Drag to orbit, scroll to zoom. Camera changes only rerender; parameter changes rebuild and validate after a short pause. Invalid inputs retain the previous preview/readouts and disable saving. The bounds are provisional local exploration windows, not verified hardware limits. Upright mode uses a separate `[upright_limits]` table, preserved in saved configs. They are stored in `[bounds]`; joint limits in `[limits]` are also provisional.

## Mass and modeling scope

Twelve nominal AKE90-8 module masses contribute **16.8 kg**. Knee/elbow modules are proximal. The Ø118 × 60 mm clearance envelopes add no mass; physical modules are approximated by uniform cylinders of Ø107.5 × 43.5 mm. The body-side roll module was moved 20 mm outward (`roll_inboard` 100 → 80 mm) to reduce its gap to the pitch stack; the coaxial pair gap remains 5 mm. Hip and knee drive modules are coaxial, with the knee drive outside. The upper link starts at that outer module's axial midpoint. Its lateral offset is included in IK. The 5 mm envelope gap and other mounting distances remain estimates, configurable as `motor.axial_gap` and `motor.thigh_axial_offset` (outward-positive trim from the outer centre, default 0). Shoulder/hip widths retain their mounting datums rather than measuring between the shifted upper-link starts.

Extra mass of zero means that unknown casing/driver/bracket mass has been omitted. An entered extra is lumped into the module cylinder. The default 8 kg frame and 1000 kg/m³ link surrogate density are uncalibrated. Default total model mass is approximately **30.53 kg**, including a 2 kg payload; it is not the real robot's predicted mass.

The model has 12 hinges but **no actuators**: transmission ratios/couplings, rotor inertia and actuator dynamics remain to be defined. The panel checks analytic mass/inertia, foot targets, motor-envelope pairs and active proxy contacts at the inspection pose. It does not validate complete CAD clearance, support forces, joint torques, payload capacity, standing/walking control or Isaac Lab migration. See [full assumptions and validation](docs/fusion-reference.md).

## Saved candidates and legacy models

Each save creates a new timestamped folder under `outputs/panel/` with `config.toml`, `robot.xml`, `validation.json`, `preview.png` and a scope note. It includes the stance targets, exploration bounds and hidden motor-placement parameters. Existing candidates and source configurations are not overwritten.

**Open config…** can switch between profiles. Saved Fusion v1 profiles retain the previous fore/aft motor arrangement and show “Legacy motor layout”; **Reset example** loads the corrected coaxial v2 baseline. The legacy eight-joint model remains available with `configs/payload.toml`; its panel uses cm and does not solve ground contact. `demo`, `study` and the original regression fixture retain their old meaning.

```sh
.venv/bin/python -m quadruped_codesign.cli generate --config configs/fusion_reference.toml --output outputs/fusion-reference/robot.xml
.venv/bin/python -m quadruped_codesign.cli check outputs/fusion-reference/robot.xml
```

Fusion checks are pose-only; a nonzero `check --steps` is rejected. The rendering path retains the GPU-context fix for repeated frames and orbiting after a model replacement.

## Tests

```sh
.venv/bin/python -m unittest discover -s tests -v
# Include native-window and rendered-pixel tests on an active desktop:
QUADRUPED_UI_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

The environment is recorded in `.python-version`, `pyproject.toml` and `uv.lock`. User practice and understanding are tracked separately from these engineering checks.
