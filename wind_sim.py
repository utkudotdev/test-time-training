import mujoco
from mujoco._specs import MjSpec
import mujoco.viewer
import time
import numpy as np

GRID_N  = 10
GRID_W  = 2.0
GRID_H = 1.5
SCALE   = 0.15

def draw_line(scn, from_pos, to_pos, rgba, width=0.005):
    if scn.ngeom >= scn.maxgeom:
        return
    g = scn.geoms[scn.ngeom]
    mujoco.mjv_initGeom(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        np.zeros(3),
        np.zeros(3),
        np.zeros(9),
        np.array(rgba, dtype=np.float32),
    )
    mujoco.mjv_connector(
        g,
        mujoco.mjtGeom.mjGEOM_LINE,
        width,
        np.array(from_pos, dtype=np.float64),
        np.array(to_pos, dtype=np.float64),
    )
    scn.ngeom += 1

def update_wind_lines(viewer, model, data):
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        for body_id in range(1, model.nbody):
            body_pos = data.xpos[body_id]
            xs = np.linspace(body_pos[0] - GRID_W / 2, body_pos[0] + GRID_W / 2, GRID_N)
            ys = np.linspace(body_pos[1] - GRID_W / 2, body_pos[1] + GRID_W / 2, GRID_N)
            for x in xs:
                for y in ys:
                    pos = np.array([x, y, body_pos[2]])
                    u, v = wind_field(pos, data.time)
                    speed = np.sqrt(u*u + v*v)

                    tail = pos
                    tip  = pos + np.array([u, v, 0.0]) * SCALE

                    c = np.clip(speed / 1.5, 0, 1)
                    rgba = [c, 0.3, 1.0 - c, 1.0]

                    draw_line(scn, tail, tip, rgba, width=0.005)


def wind_field(pos, t, speed=1.0, turbulence=0.3):
    cx, cy = 0.0, 0.0
    dx, dy = pos[0] - cx, pos[1] - cy
    r = np.sqrt(dx * dx + dy * dy) + 0.001
    s = 1.0 / (r * 6 + 0.4)
    u = (-dy * s + dx * 0.18) * 1.8 * speed
    v = (dx * s + dy * 0.18) * 1.8 * speed
    u += np.sin(pos[0] * 4 + t * 3.0)* turbulence * 0.5
    v += np.cos(pos[1] * 4 + t * 3.0) * turbulence * 0.5
    return u, v