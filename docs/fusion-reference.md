# Fusion reference concept — implemented scope

2026-11-10. Default profile: `fusion_reference_v2` (coaxial motors). Saved `fusion_reference_v1` profiles retain their original fore/aft motor layout. This is a parameterized concept model, not a CAD import or validated hardware model.

## Confirmed inputs

- Front links: 325 / 340 mm; hind links: 340 / 400 mm.
- Shoulder spacing: 450 mm; hip spacing: 500 mm; longitudinal shoulder–hip spacing: 700 mm.
- The user confirmed the shoulder outputs are **30 mm below** the hind hip outputs in a level body pose. `z_offset_hip_shoulder = -0.030` means shoulder z minus hip z.
- Each limb has three DOFs. The knee/elbow motor is proximal and drives the distal joint through a transmission. The user subsequently clarified that hip and knee drive modules are **coaxial**, with the knee drive on the outside, and that the upper link begins approximately at the outer module's axial midpoint. The same topology is used for the front shoulder/elbow pair in this concept model.
- The user confirmed Ø118 × 60 mm is the **packaging envelope**, including casing/driver/installation allowance. It does not specify additional mass.
- Nominal AKE90-8 KV35 module: 1.4 kg, Ø107.5 × 43.5 mm; twelve modules contribute 16.8 kg. [Manufacturer source](https://www.cubemars.com/product/ake90-8-kv35-quasi-direct-drive-actuator.html).

The longitudinal interpretation of 700 mm follows the existing glossary. Exact CAD measurement datums and roll-to-pitch joint offsets still need measurement. The screenshot alone does not establish those mounting transforms.

## Geometry, coordinates and pose

The body frame uses +x forward, +y left, +z up. Its origin is the midpoint of the front/hind pitch-anchor groups. Anchors are `(±d/2, ±w/2, ±dz/2)`. Relative to the glossary's rear-hip midpoint datum, the concept-frame origin is `(d/2, 0, dz/2)`; do not compare raw positions without this translation.

Each leg has roll about local x, followed by hip/shoulder pitch and knee/elbow pitch about local y. The upper-link pitch origin is displaced laterally from the roll datum, along the shared motor axis. Both the offset and the two motor centres rotate with the roll carrier. Links are measured between joint centres and from the distal joint to the spherical foot centre. The selected IK branch has negative knee angles for all four legs; the CAD's actual flexion branches and mechanical limits are not established.

`stance.hip_height` is the world height of the rear mounting anchors, before the lateral upper-link offset; it is not body-bottom clearance or the height of the actual upper-link start after roll. The existing width/height datums are retained rather than silently being redefined as the upper-link origins. Their exact association with CAD output faces still needs measurement. The report includes the actual upper-link world origins separately. `body_pitch` rotates about +y; positive values lower the front. The two stance widths are full world left–right distances between foot centres. Each foot x target uses the corresponding mounting anchor's world-horizontal projection plus `front_x` or `hind_x`.

The analytic inverse-kinematics solver adjusts joint angles while holding dimensions and targets fixed, including the roll-to-upper-link lateral offset. In the body-aligned local frame the foot vector is `Rx(roll) * ([0, d, 0] + Ry(hip)*[0,0,-L1] + Ry(hip+knee)*[0,0,-L2])`, where signed `d` is the upper-link offset. The solver uses `h = sqrt(y² + z² − d²)` and `roll = atan2(y, −z) − atan2(d, h)` before solving the planar two-link problem. Targets inside the lateral reach boundary are rejected. A translation only in world y would be incorrect when roll changes. Foot centres are at z = 26 mm, so the spherical surfaces touch z = 0. A 1 mm reach-boundary margin and configured provisional joint limits are checked. Invalid targets raise a leg-specific error; the panel retains the last valid model and disables saving. Target markers are visual sites, not physical constraints.

## Proximal motor mass and packaging

Each limb has a roll-module body fixed to the torso and two pitch-axis modules on a carrier after the roll joint, before the hip pitch joint. Thus the remote knee/elbow motor mass does not move with the lower leg. This ownership is a declared candidate approximation; proximal mounting alone does not identify the exact carrier in the CAD.

The v2 hip/knee modules share a single local y-axis: inner hip drive first, outer remotely driving knee module next. Their cylindrical centres differ only along that axis, with no fore/aft separation. Coaxial motors do not imply equal joint angles or define a transmission ratio; the distal knee joint is still at the end of the upper link.

Unmeasured mounting dimensions, expressed relative to the mounting anchor:

| Configuration | Default | Meaning |
|---|---:|---|
| `motor.roll_inboard` | 80 mm | Roll-module centre inward from the anchor; reduced from 100 mm after user feedback |
| `motor.pitch_outboard` | 30 mm | Inner hip/shoulder module centre outward from the anchor |
| `motor.axial_gap` | 5 mm | **Assumed** gap between the two 60 mm packaging envelopes |
| `motor.thigh_axial_offset` | 0 mm | Upper-link start relative to the outer module's axial midpoint; outward positive |
| `motor.pair_half_spacing` | 75 mm | Used **only in saved v1 profiles**, not in v2 |

With these defaults, the inner module centre is 30 mm outboard, the outer module centre and upper-link start are 95 mm outboard. At zero roll, upper-link origin spacing is consequently `w + 190 mm`; the input shoulder/hip width still describes the existing mounting datums. Physical module centres are separated by 65 mm, derived from 60 mm packaging length plus the provisional 5 mm gap. This is not a measured shaft length or CAD clearance. Changing `thigh_axial_offset` moves the upper-link plane and is included in IK; it must stay within the outer module's axial packaging span. Left/right placement is mirrored.

Mounting dimensions remain in `[motor]` in the TOML. They are not ordinary optimization sliders. Brackets, bearing interfaces and the radial joint/upper-link junction are still simplified; parent-child primitive overlaps at that junction are not a validated CAD fit. The Mass tab displays the arrangement, gap and trim. Old v1 snapshots show “Legacy motor layout”; “Reset example” loads the current coaxial baseline.

Module rigid-body inertia is approximated by a uniform cylinder of the nominal physical dimensions. The amber clearance cylinders carry zero mass and are checked pairwise even when MuJoCo's usual parent/same-body collision filtering would exclude them. Extra module mass defaults to zero **because it is unknown and omitted**. If entered, it is lumped into the same cylinder; this is not a measured casing inertia. Rotor inertia, reflected inertia, belts/linkages, ratios, coupled transmission equations and torque/speed limits are not implemented. The exported model has 12 hinges and **zero actuators**; it is not an RL-ready actuation model.

## Other mass assumptions

- `torso_mass = 8 kg` is an uncalibrated remaining-frame estimate, excluding motors, links, feet and payload. Its inertia uses a uniform box of full size `(d + 0.10, min(w_shoulder, w_hip) − 0.05, 0.12)` m. Visible rails and deck do not add mass on top.
- Link surrogate density is 1000 kg/m³; upper/lower capsule radii are 20 / 18 mm. Four spherical feet each have mass 0.08 kg and radius 26 mm.
- Payload is a separate fixed uniform box. Default 2 kg is a demonstration input. Its footprint must fit the 560 × 260 mm deck; its bottom must be at or above body-frame z = 60 mm. Raised payloads assume an ideal massless mount.
- Baseline mass budget: 16.8 kg nominal motors + 0 unknown extras entered + 8 kg frame estimate + 3.729541 kg links/feet + 2 kg payload = **30.529541 kg**. This is not a prediction for the approximately 50 kg hardware target.

## Bounds and validity

`[bounds]` in the TOML stores exploration intervals. Link lengths are ±30 mm around the screenshot; shoulder/hip widths ±30 mm; longitudinal spacing ±50 mm; vertical offset −50 to −10 mm. These deliberately local intervals reduce accidental extremes, and are **not measured hardware limits**. Stance, mass and density bounds are similarly provisional. Changing the study domain requires editing the saved bounds explicitly; no domain is enlarged silently.

Validation covers compiled forward kinematics against foot targets, foot-surface height, hip height, analytic masses/local centres/full inertia tensors, coaxiality and axial spacing of the motor pair, the upper-link start relative to the outer motor, pairwise motor-envelope overlap, and active proxy-contact penetration at this pose. It does not certify real CAD interference, parent-child structure clearances, all joint positions, static equilibrium, contact forces, torque feasibility or standing control. Range acceptance and geometric contact are insufficient evidence of hardware feasibility.

The panel now adds a separate [static-load baseline](static-loads.md) with explicit equal-left/right vertical reactions and conditional ideal joint-output torques. Its balance status is independent of the above geometry checks and is saved in `statics`. It does not certify motor capacity, transmission loads or controlled standing.

## Reproduce and inspect

```sh
.venv/bin/python -m quadruped_codesign.cli panel --config configs/fusion_reference.toml
.venv/bin/python -m quadruped_codesign.cli generate --config configs/fusion_reference.toml --output outputs/fusion-reference/robot.xml
.venv/bin/python -m quadruped_codesign.cli check outputs/fusion-reference/robot.xml --output outputs/fusion-reference/validation.json
QUADRUPED_UI_TESTS=1 .venv/bin/python -m unittest discover -s tests -v
```

The Fusion check is a pose check; explicit nonzero `--steps` is rejected. The legacy configuration remains available through `configs/payload.toml`, and the legacy model/fixtures are unchanged. Open config can switch profiles, preserving their units and hidden parameters. Saved candidates include the full bounds, pose and mounting assumptions.

The coaxial increment passed 33 tests, including ten Fusion tests: independent mass/inertia checks, three-DOF FK/IK under pitch and lateral targets, motor mass ownership, invalid target/limit/overlap rejection, snapshot round trips, native profile switching, rollback and repeated-render pixel checks; the coaxial update also covers world-space alignment under roll/pitch, offset-aware IK and faithful v1 snapshot reload. Existing 8-DOF and renderer regressions also passed. These engineering checks do not mark the user's learning task complete or record hours.

Next hands-on observation: change **Stance → Hind hip height** from 550 to 500 mm. Four foot surfaces should stay on the floor, joint bends should change, and the four link lengths should retain their values. Next engineering calibration: measure mounting offsets and rigid-body masses from CAD before support-force/torque evaluation.


## Subsequent pose/support increment

The panel now supports four-foot and hind-leg upright frozen poses, mass-weighted world CoM markers, planar support geometry and bounded foot-target alignment. The original four-foot pitch/limits are retained; upright mode has separate explicit concept limits. The later suite has 41 passing tests. See [support-preview.md](support-preview.md) for assumptions, parameter names, clearance measurements and scope. Neither mode supplies a balance controller.
