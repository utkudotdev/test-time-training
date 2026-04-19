"""Visualize trained PPO agent using MuJoCo viewer.

Run with: uv run mjpython visualize_mujoco.py
"""

import mujoco
import mujoco.viewer
import time
import numpy as np
from stable_baselines3 import PPO


def wind_field(pos, t, speed=1.0, turbulence=0.3):
    cx, cy = 0.0, 0.0
    dx, dy = pos[0] - cx, pos[1] - cy
    r = np.sqrt(dx * dx + dy * dy) + 0.001
    s = 1.0 / (r * 6 + 0.4)
    u = (-dy * s + dx * 0.18) * 1.8 * speed
    v = (dx * s + dy * 0.18) * 1.8 * speed
    u += np.sin(pos[0] * 4 + t * 0.5) * turbulence * 0.5
    v += np.cos(pos[1] * 4 + t * 0.5) * turbulence * 0.5
    return u, v


def get_sensor_readings(model, data, name):
    sensor_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sensor_id]
    dim = model.sensor_dim[sensor_id]
    return data.sensordata[adr : adr + dim].copy()


def build_observation(model, data, goal_geom_id):
    """Build observation matching env.py format (39D)."""
    drone_qpos = data.qpos[:7].copy()
    drone_qvel = data.qvel[:6].copy()
    box_qpos = data.qpos[7:14].copy()
    box_qvel = data.qvel[6:12].copy()
    gyro = get_sensor_readings(model, data, "body_gyro")
    accel = get_sensor_readings(model, data, "body_linacc")
    quat = get_sensor_readings(model, data, "body_quat")
    goal_pos = data.geom_xpos[goal_geom_id].copy()
    return np.concatenate(
        [drone_qpos, drone_qvel, box_qpos, box_qvel, gyro, accel, quat, goal_pos]
    ).astype(np.float32)


def main():
    # Load MuJoCo model
    with open("example.xml") as f:
        model = mujoco.MjModel.from_xml_string(f.read())
    data = mujoco.MjData(model)

    # Reset to hover keyframe
    mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)
    mujoco.mj_forward(model, data)

    # Get goal geom id
    goal_geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal")

    # Load trained PPO policy
    print("Loading trained PPO model...")
    policy = PPO.load("models/ppo_delivery")
    print("Model loaded. Launching viewer...")

    paused = False
    step_count = 0

    def key_callback(keycode):
        if chr(keycode) == " ":
            nonlocal paused
            paused = not paused
        elif chr(keycode) == "R":
            # Reset to hover keyframe
            mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)
            mujoco.mj_forward(model, data)

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                # Apply wind
                for body_id in range(1, model.nbody):
                    pos = data.xpos[body_id]
                    fx, fy = wind_field(pos, data.time)
                    data.xfrc_applied[body_id, 0] = 20 * fx
                    data.xfrc_applied[body_id, 1] = 20 * fy

                # Get observation and predict action
                obs = build_observation(model, data, goal_geom_id)
                action, _ = policy.predict(obs, deterministic=True)
                data.ctrl = np.clip(action, 0, 13)

                mujoco.mj_step(model, data)
                step_count += 1

                # Print status periodically
                if step_count % 100 == 0:
                    drone_pos = data.qpos[:3]
                    goal_pos = data.geom_xpos[goal_geom_id]
                    dist = np.linalg.norm(drone_pos - goal_pos)
                    print(
                        f"Step {step_count} | Drone: ({drone_pos[0]:.2f}, "
                        f"{drone_pos[1]:.2f}, {drone_pos[2]:.2f}) | "
                        f"Dist: {dist:.2f}"
                    )

            viewer.sync()

            time_until_next_step = model.opt.timestep - (time.time() - step_start)
            if time_until_next_step > 0:
                time.sleep(time_until_next_step)


if __name__ == "__main__":
    main()
