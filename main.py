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

def update_speed(dt):
    global speed, target_speed, speed_cooldown
    speed_cooldown -= dt
    if speed_cooldown <= 0:
        target_speed  = np.random.uniform(0.5, 1.5)
        speed_cooldown = np.random.uniform(2.0, 5.0)

    speed += (target_speed - speed) * 0.01

def update_turbulence(dt):
    global turbulence, target_turbulence, turbulence_cooldown
    turbulence_cooldown -= dt
    if turbulence_cooldown <= 0:
        target_turbulence  = np.random.uniform(0.0, 1.0)
        turbulence_cooldown = np.random.uniform(5.0, 10.0)

    turbulence += (target_turbulence - turbulence) * 0.001

GOAL_POSITION = np.array([10.0, 0.0, 2.0])
NUM_OBSTACLES = 10
OBSTACLE_REGION = np.array([[0.5, -10.0, 0.0], [10.0, 10.0, 10.0]])
OBSTACLE_RADIUS_RANGE = np.array([0.2, 1.5])

angle = 0.0
target_angle = 0.0
angle_cooldown = 0.0
prev_angle = rand.uniform(0.0, 2 * np.pi)

target_speed = rand.uniform(0.5, 2.0)
speed = target_speed
speed_cooldown = 0.0

target_turbulence = rand.uniform(0.0, 1.0)
turbulence = target_turbulence
turbulence_cooldown = 0.0

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
    global angle_cooldown, angle, target_angle, prev_speed, prev_turbulence, speed, target_speed, speed_cooldown
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
                angle, target_angle, angle_cooldown = wind.update_field_angle(angle, target_angle, angle_cooldown, model.opt.timestep)
                update_speed(model.opt.timestep)
                update_turbulence(model.opt.timestep)
                for body_id in range(1, model.nbody):
                    pos = data.xpos[body_id]
                    fx, fy = wind_field(pos, data.time, speed=speed, turbulence=turbulence, field_angle=angle)
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
            wind.update_wind_lines(viewer, model, wind_field, data, speed, turbulence, angle)

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
