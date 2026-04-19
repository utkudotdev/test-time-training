"""Visualize trained PPO agent using MuJoCo viewer.

Run with: uv run mjpython visualize_mujoco.py
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO

import wind_sim as wind
from env import build_scene_spec, HOVER_THRUST, ACTION_SCALE, GOAL_POSITION


MODEL_PATH = "models/best_model"  # falls back to ppo_delivery if not found
WITH_OBSTACLES = False
WITH_WIND = True
SHOW_WIND_LINES = False  # toggle with `W` key


def get_sensor(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim].copy()


def build_observation(model, data, goal_geom_id):
    drone_qpos = data.qpos[:7].copy()
    drone_qvel = data.qvel[:6].copy()
    box_qpos = data.qpos[7:14].copy()
    box_qvel = data.qvel[6:12].copy()
    gyro = get_sensor(model, data, "body_gyro")
    accel = get_sensor(model, data, "body_linacc")
    quat = get_sensor(model, data, "body_quat")
    goal_pos = data.geom_xpos[goal_geom_id].copy()
    return np.concatenate(
        [drone_qpos, drone_qvel, box_qpos, box_qvel, gyro, accel, quat, goal_pos]
    ).astype(np.float32)


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

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                # Apply wind forces
                if WITH_WIND:
                    for body_id in range(1, model.nbody):
                        pos = data.xpos[body_id]
                        fx, fy = wind.wind_field(pos, data.time)
                        data.xfrc_applied[body_id, 0] = 20 * fx
                        data.xfrc_applied[body_id, 1] = 20 * fy

                # Policy action → residual thrust
                obs = build_observation(model, data, goal_geom_id)
                action, _ = policy.predict(obs, deterministic=True)
                ctrl = HOVER_THRUST + action.astype(np.float64) * ACTION_SCALE
                data.ctrl = np.clip(ctrl, 0.0, 13.0)

                mujoco.mj_step(model, data)
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
                wind.update_wind_lines(viewer, model, data)
            viewer.sync()

            dt = model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
