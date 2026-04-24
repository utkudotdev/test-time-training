"""Visualize trained PPO agent using MuJoCo viewer.

Run with: uv run mjpython visualize_mujoco.py
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO

import wind_sim as wind
from env import build_scene_spec, HOVER_THRUST, GOAL_POSITION, DroneDeliveryEnv
from controller import cascaded_control


MODEL_PATH = "models/best_model"  # falls back to ppo_delivery if not found
WITH_OBSTACLES = False
WITH_WIND = False                  # set True to test with weather (use WIND_TYPE below)
WIND_TYPE = "calm"                 # "none", "calm", "cold_front", "squall", "thermal", "jet_stream"
WIND_SPEED = 1.0
WIND_TURBULENCE = 0.3
SHOW_WIND_LINES = False            # toggle with `W` key


def get_sensor(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim].copy()


def build_observation(model, data, goal_geom_id, last_action):
    drone_pos = data.qpos[:3].copy()
    quat = data.qpos[3:7].copy()
    box_pos = data.qpos[7:10].copy()
    goal_pos = data.geom_xpos[goal_geom_id].copy()

    rot = DroneDeliveryEnv._rotate_by_conj_quat
    lin_vel_body = rot(data.qvel[:3].copy(), quat)
    box_rel_pos_body = rot(box_pos - drone_pos, quat)
    box_rel_vel_body = rot(data.qvel[6:9].copy() - data.qvel[:3].copy(), quat)
    goal_vec_body = rot(goal_pos - drone_pos, quat)

    return np.concatenate([
        [drone_pos[2]], quat, lin_vel_body,
        get_sensor(model, data, "body_gyro"),
        get_sensor(model, data, "body_linacc"),
        box_rel_pos_body, box_rel_vel_body, goal_vec_body,
        last_action,
    ]).astype(np.float32)


def load_policy():
    import os
    if os.path.exists(MODEL_PATH + ".zip"):
        print(f"Loading policy from {MODEL_PATH}.zip")
        return PPO.load(MODEL_PATH)
    print("Loading policy from models/ppo_delivery.zip")
    return PPO.load("models/ppo_delivery")


def main():
    spec = build_scene_spec(seed=0, with_obstacles=WITH_OBSTACLES)
    model = spec.compile()
    data = mujoco.MjData(model)

    mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)
    data.qpos[:3] = [0.0, 0.0, 1.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:10] = [0.0, 0.0, 0.8]
    data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    goal_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
    policy = load_policy()

    paused = False
    show_wind = SHOW_WIND_LINES
    step_count = 0

    def key_callback(keycode):
        nonlocal paused, show_wind
        c = chr(keycode)
        if c == " ":
            paused = not paused
        elif c == "R":
            mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)
            data.qpos[:3] = [0.0, 0.0, 1.5]
            data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
            data.qpos[7:10] = [0.0, 0.0, 0.8]
            data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
            data.qvel[:] = 0.0
            mujoco.mj_forward(model, data)
            print("Reset.")
        elif c == "W":
            show_wind = not show_wind
            print(f"Wind visualization: {'on' if show_wind else 'off'}")

    print("Controls: SPACE=pause  R=reset  W=toggle wind lines")

    last_action = np.zeros(4, dtype=np.float32)
    wind_field_fn = getattr(wind, f"wind_{WIND_TYPE}")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                if WITH_WIND:
                    for body_id in range(1, model.nbody):
                        pos = data.xpos[body_id]
                        fx, fy = wind_field_fn(
                            pos, data.time, WIND_SPEED, WIND_TURBULENCE, 0.0
                        )
                        data.xfrc_applied[body_id, 0] = 20 * fx
                        data.xfrc_applied[body_id, 1] = 20 * fy

                # Cascaded control: policy action → motors via PD
                obs = build_observation(model, data, goal_geom_id, last_action)
                action, _ = policy.predict(obs, deterministic=True)
                quat = data.qpos[3:7]
                gyro = get_sensor(model, data, "body_gyro")
                data.ctrl = cascaded_control(quat, gyro, action, HOVER_THRUST)

                mujoco.mj_step(model, data)
                last_action = action.astype(np.float32)
                step_count += 1

                if step_count % 100 == 0:
                    drone_pos = data.qpos[:3]
                    dist = np.linalg.norm(drone_pos - GOAL_POSITION)
                    print(
                        f"Step {step_count} | Drone ({drone_pos[0]:.2f}, "
                        f"{drone_pos[1]:.2f}, {drone_pos[2]:.2f}) | "
                        f"Dist to goal: {dist:.2f}"
                    )

            if show_wind:
                wind.update_wind_lines(
                    viewer, model, wind_field_fn, data,
                    WIND_SPEED, WIND_TURBULENCE, 0.0,
                )
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
