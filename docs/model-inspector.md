# Model inspector

The **Inspect** checkbox above the preview replaces the parameter sidebar with read-only properties of the last valid frozen model. The preview remains interactive. Uncheck it to edit geometry or pose again; the selected body is retained where available. Removing a selected payload falls back to **Frame**. Invalid input retains the last valid model and readouts and disables saving.

## Try one change

1. Open the default Fusion v2 configuration. Enable **Inspect**, select **RL · Upper link**, and keep **Selected body**. Its model mass is about **0.460767 kg** and its local CoM is at **z = −170 mm**.
2. Disable **Inspect**. In Geometry, increase the hind upper-link length from **340 to 360 mm**, keeping the density and radius unchanged. Reopen the inspector: mass increases by about **0.025133 kg**, the local CoM moves to **z = −180 mm**, and transverse inertia increases. Pose readouts show the changed length after IK updates the joint angles.
3. Select **Body + descendants** to include the lower link and foot. Compare the aggregate mass, CoM and inertia with the upper link alone. This scope follows the model's body tree; it does not automatically mean a complete physical leg assembly.

These values illustrate the capsule approximation used by the generator. They are not calibrated hardware masses. Changing pose can move the world CoM while leaving a single body's own mass and local inertia unchanged. A multi-body group's inertia can change with its joint configuration.

## Readouts

| Page | Contents |
|---|---|
| Mass | Scope mass, included body names, local/world CoM, inertia tensor, mass breakdown and approximation |
| Pose | Selected-body origin and parent, included hinge angles and distance to nearest concept limit; selected leg's upper origin, knee/elbow and foot positions; segment lengths and foot height |
| Model | Bare robot/payload/total mass, actual foot spacing in world y, 3D left–right upper-origin distance, foot surface height and named motor-envelope gaps |
| Static | Whole-model equilibrium, vertical foot loads and conditional joint-output torques under an explicit equal-left/right rule |

Distances are displayed in **mm**, angles in **degrees**, mass in **kg**, and inertia in **kg·m²**. The exported `validation.json` → `inspection` uses metres/radians and retains full numerical precision for **all** bodies and subtrees. Displayed inertia uses three significant digits and suppresses floating-point noise below 1e−12 of the largest tensor entry; saved tensors are unchanged.

The selected-body local frame uses that body's origin and axes, not the principal-inertia frame. Each 3×3 inertia tensor is **about the selected scope's CoM**, expressed along the selected body's x/y/z axes. Aggregation rotates each body tensor into common axes and applies the parallel-axis theorem before expressing the result in the selected frame. It is not a simple sum of diagonal values. World readouts use +z up with ground at z=0. Foot surface height is the sphere centre's z minus its radius; it is a geometry measurement, not a contact force.

The nearest joint-limit distance is the smaller distance to the lower/upper angle bounds. These bounds remain unverified concept settings. Foot spacing is measured from the current foot centres; in upright mode the front feet are airborne and are not support contacts. Upper-origin spacing includes the motor stack offset and roll pose, so it need not equal the shoulder/hip mounting datum width in Geometry.

## Mass ownership

| Body selection | What its mass includes |
|---|---|
| Frame | Declared frame mass only; separate motors, legs and payload excluded |
| Roll module | One nominal proximal module plus declared extra mass; fixed to torso |
| Pitch drive pair | Two proximal hip/knee or shoulder/elbow modules plus declared extra mass |
| Upper link | Uniform capsule at the configured surrogate density |
| Lower link + foot | Lower capsule and the 0.08 kg spherical foot as one rigid body |
| Payload | Declared fixed uniform box; mounting hardware omitted |

The roll module is a **sibling** of the pitch-drive carrier, so selecting a pitch drive pair with descendants includes its upper/lower links and foot but excludes the roll module. Selecting Frame with descendants includes the whole robot and any payload. The motor clearance envelopes have zero mass; their highlighted shapes do not add mass to the selected group. The nominal motor cylinders, extra mass distribution, frame mass and link density remain approximations.

Orange tints the selected scope's render geometry. A white sphere marks its CoM. These are scene-only overlays: they do not modify model colours, masses, joints or collision geometry. Gold/pink/cyan retain their whole-model CoM, projection and support meanings. Selection markers can be occluded by other geometry; orbit the camera to inspect them.

## Gaps and scope

Named gaps use MuJoCo's nearest signed geom distance for the roll/inner-pitch and coaxial pitch-pair envelopes on each leg. A negative value means overlap for those proxy shapes; a positive value is separation. Distances are capped at the query limit of 10 m and display **≥** if that limit is reached. These few pairs do not establish complete collision clearance or CAD assembly feasibility.

The separate **Static** page now evaluates vertical support reactions and ideal joint-output torques under a declared equal-left/right load-sharing rule. It withholds torque results when balance fails and records that outcome separately from the geometry checks. See [static-load definitions](static-loads.md). Transmission loads and controlled standing stability are still unverified; the Fusion profile has no actuators.

Implementation references: [MuJoCo model/data fields](https://mujoco.readthedocs.io/en/stable/APIreference/APItypes.html) and [mj_geomDistance](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-geomdistance).
