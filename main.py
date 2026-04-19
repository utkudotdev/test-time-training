import mujoco
from mujoco._specs import MjSpec
import mujoco.viewer
import time
import numpy as np
import wind_sim as wind
import random as rand

def get_sensor_readings(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim]


GOAL_POSITION = np.array([10.0, 0.0, 2.0])
NUM_OBSTACLES = 10
OBSTACLE_REGION = np.array([[0.5, -10.0, 0.0], [10.0, 10.0, 10.0]])
OBSTACLE_RADIUS_RANGE = np.array([0.2, 1.5])

alpha = 0.4
prev_speed = rand.uniform(0.5, 2.0)
prev_turbulence = rand.uniform(0.0, 1.0)

def build_obstacle_scene() -> MjSpec:
    spec = mujoco.MjSpec.from_file("example.xml")

    spec.worldbody.add_geom(
        name="goal",
        type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.1],
        rgba=[0.0, 0.7, 0.3, 0.5],
        pos=GOAL_POSITION.tolist(),
        contype=0,
        conaffinity=0,
    )

    rng = np.random.default_rng()
    obs_pos = rng.uniform(
        low=OBSTACLE_REGION[0], high=OBSTACLE_REGION[1], size=(NUM_OBSTACLES, 3)
    )
    obs_size = rng.uniform(
        low=OBSTACLE_RADIUS_RANGE[0], high=OBSTACLE_RADIUS_RANGE[1], size=NUM_OBSTACLES
    )

    for pos, radius in zip(obs_pos, obs_size):
        spec.worldbody.add_geom(
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[radius],
            rgba=[1.0, 0.0, 0.0, 0.5],
            pos=pos.tolist(),
            contype=0,
            conaffinity=0,
        )

    return spec


def main():
    global prev_speed, prev_turbulence
    spec = build_obstacle_scene()
    model = spec.compile()
    data = mujoco.MjData(model)

    paused = False
    wind_mode = "1"

    def key_callback(keycode):
        nonlocal paused, wind_mode
        if chr(keycode) == " ":
            paused = not paused
        elif chr(keycode) in wind.WIND_MODES:
            wind_mode = chr(keycode)
            name, _ = wind.WIND_MODES[chr(keycode)]
            print(f"[wind] switched to: {name}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()
            _, wind_field = wind.WIND_MODES[wind_mode]

            if not paused:
                speed = prev_speed * alpha + rand.uniform(0.5, 2.0) * (1 - alpha)
                turbulence = prev_turbulence * alpha + rand.uniform(0.0, 1.0) * (1 - alpha)
                for body_id in range(1, model.nbody):
                    pos = data.xpos[body_id]
                    fx, fy = wind_field(pos, data.time, speed=speed, turbulence=turbulence)
                    data.xfrc_applied[body_id, 0] = fx
                    data.xfrc_applied[body_id, 1] = fy

                data.ctrl = 6 * np.ones(4)

                gyro = get_sensor_readings(model, data, "gyro").copy()
                accel = get_sensor_readings(model, data, "accelerometer").copy()
                quat = get_sensor_readings(model, data, "framequat").copy()

                prev_speed = speed
                prev_turbulence = turbulence

                mujoco.mj_step(model, data)

            viewer.sync()
            wind.update_wind_lines(viewer, model, wind_field, data)

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
