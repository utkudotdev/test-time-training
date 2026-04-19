import mujoco
import mujoco.viewer
import time
import numpy as np

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
        np.array(to_pos,   dtype=np.float64),
    )
    scn.ngeom += 1
GRID_N  = 10
GRID_W  = 2.0
SCALE   = 0.15   # wind vector → line length

def update_wind_lines(viewer, t):
    with viewer.lock():
        scn = viewer.user_scn
        scn.ngeom = 0

        xs = np.linspace(-GRID_W / 2, GRID_W / 2, GRID_N)
        ys = np.linspace(-GRID_W / 2, GRID_W / 2, GRID_N)

        for x in xs:
            for y in ys:
                pos = np.array([x, y, 0.1])  # slight z lift off ground
                u, v = wind_field(pos, t)
                speed = np.sqrt(u*u + v*v)

                tail = pos
                tip  = pos + np.array([u, v, 0.0]) * SCALE

                # color: blue (slow) → red (fast)
                c = np.clip(speed / 1.5, 0, 1)
                rgba = [c, 0.3, 1.0 - c, 0.8]

                draw_line(scn, tail, tip, rgba)

def wind_field(pos, t, speed=1.0, turbulence=0.3):
    cx, cy = 0.0, 0.0
    dx, dy = pos[0] - cx, pos[1] - cy
    r = np.sqrt(dx * dx + dy * dy) + 0.001
    s = 1.0 / (r * 6 + 0.4)
    u = (-dy * s + dx * 0.18) * 1.8 * speed
    v = (dx * s + dy * 0.18) * 1.8 * speed
    # turbulence
    u += np.sin(pos[0] * 4 + t * 0.5) * turbulence * 0.5
    v += np.cos(pos[1] * 4 + t * 0.5) * turbulence * 0.5
    return u, v

def get_sensor_readings(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim]

def main():
    with open("example.xml") as f:
        model = mujoco.MjModel.from_xml_string(f.read())

    data = mujoco.MjData(model)

    goal = data.geom("goal")

    paused = False

    def key_callback(keycode):
        if chr(keycode) == " ":
            nonlocal paused
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            goal_pos = goal.xpos

            if not paused:
                for body_id in range(1, model.nbody):
                    pos = data.xpos[body_id]
                    fx, fy = wind_field(pos, data.time)
                    data.xfrc_applied[body_id, 0] = 20 * fx
                    data.xfrc_applied[body_id, 1] = 20 * fy

                data.ctrl = 6 * np.ones(4)

                gyro   = get_sensor_readings(model, data, "gyro").copy()
                accel  = get_sensor_readings(model, data, "accelerometer").copy()
                quat   = get_sensor_readings(model, data, "framequat").copy()

                mujoco.mj_step(model, data)

            viewer.sync()
            update_wind_lines(viewer, data.time)

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
