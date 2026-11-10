"""Bounded geometric CoM alignment by changing foot targets, not mass or markers."""
from dataclasses import replace
import numpy as np

from .panel_model import Preview
from .support import TOLERANCE


def align_projection(shape):
    """Centre the projected CoM in x; retain y within the support segment/polygon.

    Quad mode moves both front/hind target x offsets by the same amount. Upright
    mode moves only the hind target x offset. Dimensions, masses, pitch, height,
    widths and joint limits stay fixed. A failed search changes nothing.
    """
    names = ('upright_hind_x',) if shape.stance.upright else ('front_x','hind_x')
    low = max(shape.bounds['stance.'+name][0]-getattr(shape.stance,name) for name in names)
    high = min(shape.bounds['stance.'+name][1]-getattr(shape.stance,name) for name in names)
    attempts = 0

    def sample(delta):
        nonlocal attempts
        attempts += 1
        stance = replace(shape.stance,**{n:getattr(shape.stance,n)+delta for n in names})
        candidate = replace(shape,stance=stance)
        preview = Preview.build(candidate)
        support = preview.report['support']
        centre_x = np.mean([p[0] for p in support['contacts_world_m'].values()])
        residual = support['com_world_m'][0]-centre_x
        return candidate, support, float(residual)

    def result(delta, sampled):
        candidate, support, residual = sampled
        if not support['inside_or_on_support']:
            raise ValueError("Fore-aft alignment cannot place this CoM inside the support segment/area. Check lateral payload position and foot spacing.")
        return candidate, {'foot_x_shift_m':float(delta), 'fore_aft_residual_m':residual,
                           'evaluations':attempts, 'method':'bounded foot-target search; kinematics only'}

    initial = sample(0.)
    if abs(initial[2]) <= TOLERANCE:
        return result(0.,initial)
    previous = None
    bracket = None
    for delta in sorted(set(np.linspace(low,high,41).tolist()+[0.])):
        try:
            current = sample(delta)
        except ValueError:
            previous = None  # Never bridge a known infeasible region.
            continue
        if abs(current[2]) <= TOLERANCE:
            return result(delta,current)
        if previous is not None and previous[1][2]*current[2] < 0:
            bracket = (previous,(delta,current))
            break
        previous = (delta,current)
    if bracket:
        (a,left),(b,right) = bracket
        for _ in range(35):
            middle = (a+b)/2
            try:
                current = sample(middle)
            except ValueError as error:
                raise ValueError(f"Alignment crossed an invalid pose; inputs were kept. {error}") from error
            if abs(current[2]) <= TOLERANCE:
                return result(middle,current)
            if left[2]*current[2] <= 0:
                b,right = middle,current
            else:
                a,left = middle,current
    raise ValueError("No feasible CoM alignment found within the current foot-x bounds, joint limits and pose. Adjust height/pitch or targets; no parameters were changed.")
