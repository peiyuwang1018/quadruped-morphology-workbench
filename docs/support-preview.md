# Pose switching and CoM projection

2026-11-10. Implemented in the Fusion parameter panel. These are frozen geometric inspection poses, not standing control or a sit-to-stand trajectory.

## Smaller body-side gap

The red-box request referred to the gap between the body-side roll module and the coaxial pitch-module stack, not the 5 mm gap between the two coaxial modules. The default `motor.roll_inboard` was reduced from 100 to 80 mm, moving the roll-module centre outward by 20 mm. The pitch-module stack, thigh origin, leg lengths and the 5 mm axial gap are unchanged.

At the default four-foot pose, MuJoCo's signed envelope distance for the front-left roll/hip pair decreased from approximately 39.98 to 20.19 mm; the rear-left pair changed from 40.10 to 20.28 mm. The default model passes the existing pairwise envelope and active proxy-contact checks. These are proxy-geometry distances, not CAD manufacturing clearances. Existing saved configs retain their explicit old offsets.

## Two stored poses

In **Stance**, choose **Four-foot stance** or **Hind-leg upright**. Each mode retains its own pitch, height and foot targets; switching back restores the earlier four-foot settings. Both modes use the same morphology, motor masses and payload. Body pose and joint angles genuinely change; the camera remains an independent control.

Four-foot mode uses the original `hip_height`, `body_pitch`, front/hind width and x offsets. All four foot surfaces contact the ground geometrically.

Upright mode uses the following additional fields in `[stance]`:

| Field | Default | Meaning |
|---|---:|---|
| `mode` | `quadruped` | `hind_stand` selects the upright pose |
| `upright_hip_height` | 0.55 m | Height of the rear mounting datum |
| `upright_pitch` | −π/2 rad | Body x-axis points upward |
| `upright_hind_width` | 0.70 m | Full left–right foot-centre spacing |
| `upright_hind_x` | 0.05 m | Foot x offset from rear mounting datum projection |
| `front_roll`, `front_hip`, `front_knee` | 0, 0.45, −1.2 rad | Free front-limb pose, not ground targets |

Only the two hind feet are ground targets in upright mode. The front feet must remain more than 10 mm above ground and active proxy contacts must not penetrate. The payload remains rigidly attached to the torso when it rotates; this is not a grasping task.

**Upright joint ranges are explicit concept assumptions.** The original four-foot hip range (−1.35 to +1.35 rad) cannot reach the selected upright pose on the current negative-knee branch. The separate `[upright_limits]` table uses hip range −1.35 to +2.80 rad (approximately −77° to +160°). Roll and knee limits retain their existing provisional values. The UI displays the upright hip range as unverified. No hardware ROM has been measured or certified. Supplying narrower limits causes pose rejection; the solver and alignment never enlarge them automatically.

Upright exploration windows are stored in `[bounds]`: height 0.40–0.65 m, pitch −1.70 to −1.05 rad, hind width 0.60–0.80 m, and hind x offset −0.15 to +0.15 m. These do not guarantee all parameter combinations are reachable or collision-free. Older configs receive the new inactive upright defaults while preserving their original four-foot pose.

## What the markers mean

- **Gold CoM:** the mass-weighted centre of all compiled robot bodies, including the payload. `sum(m_i * x_i) / sum(m_i)` uses body inertial-frame world positions, not body origins or geometric centres guessed from appearance.
- **Pink Projection:** that same point projected vertically to the z = 0 ground plane. The dashed gold line joins the two.
- **Cyan support:** the convex hull of the declared ground-contact points. Four feet usually form a polygon; two ideal point feet form a finite line segment.
- Numeric CoM readouts now use **world coordinates** in mm to match the markers. The saved report still includes both world-frame and torso-frame CoM.

The four-foot display reports the minimum signed inward distance to the polygon edges. An outside point has a negative margin. Upright mode instead reports distance to the **finite segment**, including its endpoints; a point on the infinite extension is not accepted. Even when aligned, a two-point support line has zero area and zero positive transverse margin. The display therefore keeps an amber “stability NOT verified” message, including when the separate ideal static-load check passes.

The overlay is made with MuJoCo render-only geoms, refreshed after every scene update. It adds no physical bodies, masses, contacts, joints or forces. **Show CoM** hides only the overlay. The panel's PNG includes it; the physical MJCF does not. The numeric support analysis is saved in `validation.json`.

## Align CoM projection

### Adjust the free front limbs first

In upright mode, **Stance** has **Body & feet** and **Front limbs** pages. The latter exposes `front_roll` (Shoulder roll), `front_hip` (Shoulder pitch) and `front_knee` (Elbow angle) in degrees. These are direct joint commands for a frozen pose, not front-endpoint IK or a controller. Defaults remain 0, 0.45 and −1.2 rad.

The left arm uses the entered roll; the right uses its negative. Shoulder pitch and elbow angle are shared. Each slider uses the corresponding saved `[upright_limits]`; mirrored roll uses the intersection valid for both signs. These remain unverified concept limits. Model checks still reject collisions even when angles lie inside the slider ranges.

An arm edit recomputes the model, CoM and projection error while preserving morphology, mass and hind foot targets. It does **not** silently run alignment. Use the alignment button separately to observe the difference. Invalid edits keep the previous valid preview and disable saving. Both mode switching and saved snapshots retain the arm settings; older configurations keep their saved angles or receive the existing defaults when those fields are absent.

### Run alignment

This button searches within the existing foot-x bounds:

- In four-foot mode, it shifts both front and hind foot-x targets equally, centring the CoM projection in the support polygon's fore/aft direction.
- In upright mode, it adjusts only `upright_hind_x` to place the CoM projection on the hind-foot segment within a 0.1 mm numerical tolerance.

At every trial it reruns IK, compiles the model, recomputes mass-weighted CoM, and checks joint limits and current proxy collisions. Morphology, mass, payload position, body height/pitch, lateral spacing and limits stay fixed. Alignment never moves a marker independently of the physical model. It also checks that the result lies between the segment endpoints / inside the polygon. A failed search retains the current configuration and explains that no alignment was found within this search domain; this does not prove global infeasibility.

In the default upright example, the fore/aft line error decreased from about **32.95 mm** to **0.064 mm**, with the hind foot offset changing from 50 to approximately 14.297 mm. This is an example of geometric alignment, not a measured stability or load result. The snapshots are under `outputs/support-preview/`.

## Verification and limits

41 tests passed, including the original 33 and eight new support/pose tests. Coverage includes polygon/finite-segment geometry, positive-vs-zero margin semantics, independent MuJoCo subtree-CoM comparison, rotated payload contribution, two hind contacts with raised front limbs, explicit upright limits, bounded alignment and rollback, reduced module clearance, dual-pose save/reload, UI switching and render-only overlay pixel checks. Screenshots of both modes and upright alignment were inspected.

Geometric support is only an initial screening step under flat-ground, gravity-only, ideal point-contact assumptions. The geometric overlay does not compute ground reactions. Friction reserves, motor torque limits, transmission dynamics, contact compliance and feedback control remain unverified. See [Feasible Region: an Actuation-Aware Extension of the Support Region](https://arxiv.org/abs/1903.07999) for why support geometry alone does not establish actuator feasibility. Rendering follows the official [MuJoCo visualization API](https://mujoco.readthedocs.io/en/3.17.0/APIreference/APIfunctions.html#mjv-initgeom).

The panel now adds a separate [Static page](static-loads.md): it assumes equal left/right vertical loads, checks equilibrium and conditionally reports ideal joint-output torques. This does not change the meaning of the geometric overlay or the alignment search. A geometrically aligned candidate can still be outside the static page's distribution rule, and no motor capacity or controlled stability is verified.

User learning completion and working hours are not inferred from these engineering checks.
