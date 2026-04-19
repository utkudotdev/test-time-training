import mujoco
import mujoco.viewer
import time


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
                mujoco.mj_step(model, data)
            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
