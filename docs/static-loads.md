# Static joint-load baseline

Open **Inspect → Static** in the Fusion profile. This page assesses the **whole model**, independent of the selected body. It reports vertical foot loads, balance residuals and, only after equilibrium checks pass, ideal joint-output torques. Scroll for all joints and the gravity/contact components on joints belonging to the current body selection.

## One experiment

1. Reset the default example, choose **Hind-leg upright**, then open **Inspect → Static**. The initial pose shows **Not balanced**: the trial hind-foot loads leave a moment of approximately 9.87 N·m.
2. Uncheck Inspect and click **Stance → Align CoM projection**. Return to Static: it should show **Within tolerance**, with about 149.75 N per hind foot and a largest joint demand of about 40.88 N·m at a hind knee.
3. In **Stance → Front limbs**, change **Shoulder pitch** to 45°. The hind targets stay fixed, so balance is lost again. Align separately and inspect the new torques. This separates the effect of arm pose from the effect of moving the supporting feet.

For comparison, the default four-foot pose gives about 71.54 N per front foot, 78.21 N per hind foot, and a largest joint demand of 19.66 N·m at a hind knee. These numbers use the same uncalibrated model geometry and mass estimates; they are examples, not motor specifications or optimized configurations.

## Assumptions and distribution rule

- The pose is frozen: velocity and acceleration are zero. Gravity acts downward along world z. No external task force or moment is applied.
- The ground is level. Spherical feet are represented by ideal contact points directly below their centres at world z=0. Forces are vertical and nonnegative; horizontal forces and contact moments are zero.
- Left/right feet share each front or hind pair's load equally. Both pairs must have mirrored foot positions about a common lateral centreline. The whole-model CoM must lie on that centreline within the stated numerical tolerance.
- Four-foot sharing follows the fore/aft CoM position. For weight W, front/hind contact x coordinates xF and xR, and CoM coordinate xC:

  `F_front_pair = W (xC − xR) / (xF − xR)`

  `F_hind_pair = W − F_front_pair`

  Each left/right foot receives half its pair's load. Four contacts alone do not uniquely determine every reaction; equal left/right sharing is an explicit choice.
- In hind-leg upright mode, each hind foot receives W/2. Their combined force line must pass through the CoM projection. Airborne front feet supply no contact forces, but the front limbs still require joint torque to hold their own weight.

If a pair would have to pull downward on the ground, the result is **Not balanced**. A lateral payload offset requiring unequal left/right loads is **Outside baseline**, not a declaration of physical impossibility. The evaluator does not silently change the load-sharing rule or move the feet. Coincident fore/hind x positions are also outside the current pair-load formula. Legacy eight-joint previews lack declared ground-contact poses and are not evaluated.

## Computation and checks

The stationary joint equation is:

`tau_required = g(q) − sum(J_contact(q).T @ F_contact)`

MuJoCo's recursive Newton–Euler routine supplies g(q) at zero velocity and acceleration. Each point Jacobian maps the specified ground reaction into generalized forces. This deliberately excludes the simulator's soft-contact solver reactions. The original preview data and model are not changed.

The evaluator checks both:

1. World-coordinate total force and moment balance about the whole-model CoM.
2. The six unactuated floating-base coordinates: their required force and moment must vanish within tolerance. A free body cannot supply its own holding wrench.

The force tolerance is `max(1e−8 N, W × 1e−8)`. The moment tolerance is `W × 0.0001 m`, consistent with the existing 0.1 mm geometric alignment tolerance. With the default 299.49 N weight, this permits about 0.02995 N·m residual moment. **Within tolerance is a numerical acceptance condition, not exact equilibrium or a positive stability margin.** Residuals and tolerances are displayed and exported. Two point feet still form a zero-area support line; none of this verifies feedback stability.

When balance fails, trial loads and residuals remain available, but the joint-torque list is empty. A geometrically valid unbalanced candidate can still be saved for comparison; it must not silently disappear from a future design study. The top-level report's `passed` flag describes geometry/kinematics, while `statics.equilibrium_within_tolerance` describes this separate mechanical check.

## Torque meaning and export

Torques are in N·m, positive along each MJCF joint's positive axis. FL/FR `hip` and `knee` names denote shoulder pitch and elbow. Roll signs can be opposite on mirrored limbs. Each exported joint row contains gravity-only demand, generalized contact contribution, and their difference. The UI also reports the largest absolute joint torque; this is a descriptive metric, not a capacity or optimality assessment.

The result is the ideal torque at each modeled **joint output**. The proximal hip/knee motors may have a coupled transmission; motor-shaft torque cannot be inferred until its kinematic mapping, ratios and efficiency are specified. No motor rating, friction-loss allowance, gearbox efficiency, elastic loading, rotor inertia or thermal capacity is assumed. Passive springs, friction loss, gravity compensation, tendons and equality constraints are rejected by this baseline rather than ignored. Joint-limit stop reactions are not used as supports. Current joint damping contributes no force at zero velocity.

Panel snapshots store the complete result under `validation.json` → `statics`, including the rule, assumptions, contacts, trial forces, residuals, checks and torque rows. No configuration parameters or actuator elements are added. The existing CLI generator/check commands retain their kinematic scope; programmatic static reports are available through `Preview.build(shape).report['statics']`.

Verification includes hand-computable pair loading, independent subtree moment sums in both poses, finite-difference virtual work, floating-base residual frames, payload/geometry changes, unbalanced and unsupported cases, save/reload, and native-window updates.

References: [MuJoCo equations of motion](https://mujoco.readthedocs.io/en/stable/computation/index.html#general-framework), [recursive Newton–Euler](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-rne), and [point Jacobian](https://mujoco.readthedocs.io/en/stable/APIreference/APIfunctions.html#mj-jac).
