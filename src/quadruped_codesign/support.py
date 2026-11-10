"""CoM projection and planar point-contact geometry, without stability claims."""
import numpy as np
import mujoco

TOLERANCE = 1e-4  # 0.1 mm numerical alignment tolerance, not a stability margin.


def cross(a, b):
    return float(a[0]*b[1]-a[1]*b[0])


def convex_hull(points):
    ordered = sorted(set(tuple(map(float, p)) for p in points))
    if len(ordered) < 2:
        return np.asarray(ordered)
    def half(sequence):
        result = []
        for p in sequence:
            while len(result) >= 2 and cross(np.subtract(result[-1],result[-2]),
                                             np.subtract(p,result[-1])) <= 0:
                result.pop()
            result.append(p)
        return result
    return np.asarray(half(ordered)[:-1]+half(reversed(ordered))[:-1])


def segment_distance(point, a, b):
    delta = b-a
    t = np.clip(np.dot(point-a, delta)/np.dot(delta,delta),0,1)
    return float(np.linalg.norm(point-(a+t*delta)))


def projection_geometry(com, contacts):
    com = np.asarray(com, dtype=float)
    hull = convex_hull(np.asarray(contacts)[:,:2])
    p = com[:2]
    if len(hull) >= 3:
        edges = list(zip(hull, np.roll(hull,-1,axis=0)))
        margins = [cross(b-a,p-a)/np.linalg.norm(b-a) for a,b in edges]
        margin = float(min(margins))
        distance = 0. if margin >= 0 else min(segment_distance(p,a,b) for a,b in edges)
        dimension = 2
        inside = margin >= -TOLERANCE
    elif len(hull) == 2:
        distance = segment_distance(p,*hull)
        margin = 0. if distance <= TOLERANCE else -distance
        dimension, inside = 1, distance <= TOLERANCE
    else:
        raise ValueError("At least two distinct ground contacts are needed")
    return {"com_world_m": com.tolist(), "projection_world_m": [float(p[0]),float(p[1]),0.],
            "hull_xy_m": hull.tolist(), "support_dimension": dimension,
            "inside_or_on_support": bool(inside), "signed_margin_m": margin,
            "projection_distance_m": distance, "alignment_tolerance_m": TOLERANCE,
            "static_stability_verified": False,
            "assumptions": "Level ground, gravity only, ideal point contacts. A two-foot line has zero area and no positive transverse margin. This is geometric screening, not torque/friction feasibility or controlled balance."}


def inspect_support(model, data, legs):
    total = float(model.body_mass.sum())
    com = (model.body_mass[:,None]*data.xipos).sum(axis=0)/total
    feet = {leg:data.site(leg+'_foot').xpos.copy() for leg in legs}
    contacts = {leg:[float(p[0]),float(p[1]),0.] for leg,p in feet.items()}
    result = projection_geometry(com,list(contacts.values()))
    result['contact_legs'] = list(legs)
    result['contacts_world_m'] = contacts
    return result


def add_support_overlay(scene, report):
    """Append render-only geoms after update_scene; never add physical bodies/mass."""
    if 'support' not in report:
        return
    support = report['support']
    com = np.asarray(support['com_world_m'])
    projection = np.asarray(support['projection_world_m'])+[0,0,.008]
    cyan, gold, pink = (.15,.90,.90,1), (1,.76,.10,1), (1,.28,.63,1)

    def geom(kind, pos, size, color, label=''):
        if scene.ngeom >= scene.maxgeom:
            raise RuntimeError("Not enough render geometry capacity for CoM overlay")
        g = scene.geoms[scene.ngeom]
        mujoco.mjv_initGeom(g,kind,np.asarray(size,dtype=float),np.asarray(pos,dtype=float),
                           np.eye(3).reshape(9),np.asarray(color,dtype=np.float32))
        g.label = label
        scene.ngeom += 1
        return g

    def line(a,b,color,width=3):
        g = geom(mujoco.mjtGeom.mjGEOM_LINE,(0,0,0),(0,0,0),color)
        mujoco.mjv_connector(g,mujoco.mjtGeom.mjGEOM_LINE,width,np.asarray(a),np.asarray(b))

    hull = [np.array([x,y,.008]) for x,y in support['hull_xy_m']]
    edges = list(zip(hull,hull[1:]))
    if len(hull)>2:
        edges.append((hull[-1],hull[0]))
    for a,b in edges:
        line(a,b,cyan,4)
    for k in range(8):
        line(projection+(com-projection)*(k/8),projection+(com-projection)*((k+.55)/8),gold,2)
    geom(mujoco.mjtGeom.mjGEOM_SPHERE,com,(.019,0,0),gold,'CoM')
    geom(mujoco.mjtGeom.mjGEOM_SPHERE,projection,(.014,0,0),pink,'Projection')
