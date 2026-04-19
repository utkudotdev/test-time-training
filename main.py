import mujoco
import mujoco.viewer
import time
import numpy as np


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


def main():
    with open("example.xml") as f:
        model = mujoco.MjModel.from_xml_string(f.read())

    data = mujoco.MjData(model)

    paused = False

    def key_callback(keycode):
        if chr(keycode) == " ":
            nonlocal paused
            paused = not paused

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                for body_id in range(1, model.nbody):
                    pos = data.xpos[body_id]
                    fx, fy = wind_field(pos, data.time)
                    data.xfrc_applied[body_id, 0] = 20 * fx
                    data.xfrc_applied[body_id, 1] = 20 * fy

                data.ctrl = 6 * np.ones(4)

                mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
