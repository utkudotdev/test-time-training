"""Visualize drone-only PPO policy in MuJoCo viewer.

Run: uv run mjpython visualize_drone.py
"""

import mujoco, mujoco.viewer, time, os
import numpy as np
from stable_baselines3 import PPO
from env_drone import DroneEnv, HOVER_THRUST, GOAL_POSITION
from controller import cascaded_control


def get_sensor(model, data, name):
    sid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SENSOR, name)
    adr = model.sensor_adr[sid]
    dim = model.sensor_dim[sid]
    return data.sensordata[adr : adr + dim].copy()


def rotate_by_conj_quat(v, q):
    w, x, y, z = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y + w * z),     2 * (x * z - w * y)],
        [2 * (x * y - w * z),     1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
        [2 * (x * z + w * y),     2 * (y * z - w * x),     1 - 2 * (x * x + y * y)],
    ])
    return R @ v


def build_obs(model, data, goal_geom_id, last_action):
    drone_pos = data.qpos[:3].copy()
    quat = data.qpos[3:7].copy()
    goal_pos = data.geom_xpos[goal_geom_id].copy()
    goal_vec_body = rotate_by_conj_quat(goal_pos - drone_pos, quat)
    lin_vel_body = rotate_by_conj_quat(data.qvel[:3].copy(), quat)
    return np.concatenate([
        [drone_pos[2]],
        quat,
        lin_vel_body,
        get_sensor(model, data, "body_gyro"),
        get_sensor(model, data, "body_linacc"),
        goal_vec_body,
        last_action,
    ]).astype(np.float32)


def main():
    spec = mujoco.MjSpec.from_file("x2_only.xml")
    spec.worldbody.add_geom(
        name="goal", type=mujoco.mjtGeom.mjGEOM_SPHERE,
        size=[0.15], rgba=[0.0, 0.8, 0.2, 0.6],
        pos=GOAL_POSITION.tolist(), contype=0, conaffinity=0,
    )
    model = spec.compile()
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    goal_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "goal")

    # Load best or final model
    path = "models_drone/best_model" if os.path.exists("models_drone/best_model.zip") \
           else "models_drone/ppo_drone_final"
    print(f"Loading {path}.zip")
    policy = PPO.load(path)

    paused = False

    def key_callback(keycode):
        nonlocal paused
        if chr(keycode) == " ":
            paused = not paused
        elif chr(keycode) == "R":
            mujoco.mj_resetDataKeyframe(model, data, 0)
            mujoco.mj_forward(model, data)
            print("Reset.")

    print("SPACE=pause  R=reset")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        # Use the "track" camera that follows the drone's center of mass
        cam_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "track")
        viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
        viewer.cam.fixedcamid = cam_id
        viewer.sync()
        step = 0
        last_action = np.zeros(4, dtype=np.float32)
        while viewer.is_running():
            t0 = time.time()
            if not paused:
                obs = build_obs(model, data, goal_geom, last_action)
                action, _ = policy.predict(obs, deterministic=True)
                quat = data.qpos[3:7]
                gyro = get_sensor(model, data, "body_gyro")
                data.ctrl = cascaded_control(quat, gyro, action, HOVER_THRUST)
                mujoco.mj_step(model, data)
                last_action = action.astype(np.float32)
                step += 1
                if step % 100 == 0:
                    pos = data.qpos[:3]
                    dist = np.linalg.norm(pos - GOAL_POSITION)
                    print(f"Step {step} | pos ({pos[0]:.2f},{pos[1]:.2f},{pos[2]:.2f}) | dist {dist:.2f}")
            viewer.sync()
            dt = model.opt.timestep - (time.time() - t0)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
