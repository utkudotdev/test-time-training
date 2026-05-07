import mujoco
from mujoco._specs import MjSpec
import mujoco.viewer
import time
import numpy as np

GRID_N  = 10
GRID_W  = 3.0
GRID_H = 2.5
SCALE   = 0.15

def update_field_angle(field_angle, target_angle, angle_cooldown, dt):
    angle_cooldown -= dt
    if angle_cooldown <= 0:
        target_angle   = np.random.uniform(0, 2 * np.pi)
        angle_cooldown = np.random.uniform(5.0, 8.0)

    field_angle += (target_angle - field_angle) * 0.002

    return field_angle, target_angle, angle_cooldown

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

def update_wind_lines(viewer, model, wind_field, data, speed, turbulence, field_angle):
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
                    u, v, w = wind_field(pos, data.time, speed, turbulence, field_angle)
                    spd = np.sqrt(u*u + v*v)

                    tail = pos
                    tip  = pos + np.array([u, v, w]) * SCALE

                    c = np.clip(spd / 1.5, 0, 1)
                    rgba = [c, 0.3, 1.0 - c, 1.0]

                    draw_line(scn, tail, tip, rgba, width=0.05)

def cap_pitch(u_r, v_r, w, max_deg=20):
    horiz = np.sqrt(u_r**2 + v_r**2)
    max_tan = np.tan(np.radians(max_deg))
    return np.clip(w, -horiz * max_tan, horiz * max_tan)

def wind_none(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    return 0, 0, 0

def wind_calm(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    u = np.cos(field_angle) * speed * 0.5
    v = np.sin(field_angle) * speed * 0.5

    u += np.sin(pos[0] * 4 + t * 3.0) * turbulence * 0.2
    v += np.cos(pos[1] * 4 + t * 3.0) * turbulence * 0.2
    w  = np.sin(pos[2] * 4 + t * 3.0) * turbulence * 0.1

    return u, v, w
 
def wind_cyclone(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    cx, cy = 0.0, 0.0
    dx, dy = pos[0] - cx, pos[1] - cy
    r = np.sqrt(dx*dx + dy*dy) + 0.001
    s = 1.0 / (r * 6 + 0.4)
    u = (-dy * s + dx * 0.18) * 1.8 * speed
    v = (dx * s + dy * 0.18) * 1.8 * speed

    w = np.sin(pos[2] * 2 + t * 1.5) * turbulence * 0.1
    ring_r     = 0.3
    ring_width = 0.08
    ring_phase = r - ring_r - (t * 0.2 % 0.6)
    ring       = np.exp(-(ring_phase**2) / ring_width)
    w += ring * 0.8 * speed

    u += np.sin(pos[0] * 4 + t * 3.0) * turbulence * 0.2
    v += np.cos(pos[1] * 4 + t * 3.0) * turbulence * 0.2
    w += np.sin(pos[2] * 2 + t * 1.5) * turbulence * 0.1

    horiz = np.sqrt(u**2 + v**2)
    w = np.clip(w, -horiz * np.tan(np.radians(20)), horiz * np.tan(np.radians(20)))

    return u, v, w

def wind_cold_front(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    cos_a = np.cos(field_angle)
    sin_a = np.sin(field_angle)
    rx =  pos[0] * cos_a + pos[1] * sin_a
    ry = -pos[0] * sin_a + pos[1] * cos_a

    wave    = np.sin(rx * np.pi * 2 + t * 0.4) * 0.3
    front_y = wave

    dist = ry - front_y

    cold_wgt = 1.0 - np.clip((dist + 0.3) / 0.6, 0, 1)
    warm_wgt = 1.0 - cold_wgt

    u_r = speed * (1.6 * cold_wgt + 0.4 * warm_wgt)
    v_r = speed * (-0.1 * cold_wgt + 0.2 * warm_wgt)

    front_proximity = np.exp(-dist**2 / 0.05)
    w  =  front_proximity * 1.2 * speed     
    w += -0.15 * speed * cold_wgt
    w +=  0.1  * speed * warm_wgt

    # Turbulence
    u_r += np.sin(rx * 4 + t * 3.0) * turbulence * 0.5
    v_r += np.cos(ry * 4 + t * 3.0) * turbulence * 0.5
    w   += np.sin(pos[2] * 3 + t * 2.0) * turbulence * 0.15

    # Rotate back to world frame
    u = u_r * cos_a - v_r * sin_a
    v = u_r * sin_a + v_r * cos_a
    return u, v, cap_pitch(u_r, v_r, w)  

def wind_squall(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    cos_a = np.cos(field_angle)
    sin_a = np.sin(field_angle)
    rx =  pos[0] * cos_a + pos[1] * sin_a
    ry = -pos[0] * sin_a + pos[1] * cos_a

    line_x    = (rx - t * 0.3) % 2.0 - 1.0
    sq_w      = 0.12
    front_str = np.exp(-(line_x * line_x) / (sq_w * sq_w))
    u_r = speed * (1.6 - front_str * 0.8)
    v_r = ry * front_str * 2.5 * speed
    v_r += np.sin(ry * np.pi * 4 + t) * 0.15 * speed
    u_r += np.sin(rx * 4 + t * 3.0) * turbulence * 0.5
    v_r += np.cos(ry * 4 + t * 3.0) * turbulence * 0.5

    ahead = np.exp(-((line_x - 0.15)**2) / (sq_w * sq_w))
    w  = -front_str * 1.0 * speed
    w +=  ahead     * 0.5 * speed
    w += np.sin(pos[2] * 3 + t * 2.0) * turbulence * 0.15

    u = u_r * cos_a - v_r * sin_a
    v = u_r * sin_a + v_r * cos_a
    return u, v, cap_pitch(u_r, v_r, w)

def wind_thermal(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    cos_a = np.cos(field_angle)
    sin_a = np.sin(field_angle)
    rx =  pos[0] * cos_a + pos[1] * sin_a
    ry = -pos[0] * sin_a + pos[1] * cos_a

    thermals = [(-0.6, -0.5), (0.4, 0.3), (0.8, -0.7)]
    u_r = 0.08 * speed
    v_r = 0.0
    w   = 0.0

    for tx, ty in thermals:
        dx, dy = rx - tx, ry - ty
        r2     = dx*dx + dy*dy
        bubble = np.exp(-r2 / 0.08)

        u_r += -dx * bubble * 0.6 * speed
        v_r += -dy * bubble * 0.6 * speed
        w += bubble * 1.8 * speed

    u_r += np.sin(rx * 4 + t * 3.0) * turbulence * 0.5
    v_r += np.cos(ry * 4 + t * 3.0) * turbulence * 0.5
    w   += np.sin(pos[2] * 3 + t * 2.0) * turbulence * 0.15

    u = u_r * cos_a - v_r * sin_a
    v = u_r * sin_a + v_r * cos_a
    return u, v, cap_pitch(u_r, v_r, w, 40)

def wind_jet_stream(pos, t, speed=1.0, turbulence=0.3, field_angle=0.0):
    cos_a = np.cos(field_angle)
    sin_a = np.sin(field_angle)
    rx =  pos[0] * cos_a + pos[1] * sin_a
    ry = -pos[0] * sin_a + pos[1] * cos_a

    jet_y     = np.sin(rx * np.pi * 2.5 + t * 0.25) * 0.4
    jet_width = 0.15
    dist      = ry - jet_y
    strength  = np.exp(-(dist ** 2) / (2 * jet_width * jet_width))

    u_r = (2.5 + np.sin(rx * 4 + t * 0.2) * 0.3) * speed * strength
    v_r = np.cos(rx * np.pi * 2.5 + t * 0.25) * 0.6 * speed * strength

    w  = -np.sign(dist) * strength * 0.3 * speed
    w += np.sin(pos[2] * 3 + t * 2.0) * turbulence * 0.15

    u_r += np.sin(rx * 4 + t * 3.0) * turbulence * 0.3
    v_r += np.cos(ry * 4 + t * 3.0) * turbulence * 0.3

    u = u_r * cos_a - v_r * sin_a
    v = u_r * sin_a + v_r * cos_a
    return u, v, cap_pitch(u_r, v_r, w)

WIND_MODES = {
    "1": ("calm", wind_calm),
    "2": ("cold_front", wind_cold_front),
    "3": ("squall", wind_squall),
    "4": ("thermal", wind_thermal), 
    "5": ("jet_stream", wind_jet_stream),
    "6": ("cyclone", wind_cyclone),
    "7": ("none", wind_none),
}