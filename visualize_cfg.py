"""Configurable MuJoCo visualizer — choose wind type, obstacles, and model via CLI.

Run with:
  uv run mjpython visualize_cfg.py
  uv run mjpython visualize_cfg.py --wind-type cyclone --obstacles
  uv run mjpython visualize_cfg.py --model models_ttt/best_model --wind-type squall --obstacles
  uv run mjpython visualize_cfg.py --model models/best_model --no-wind --obstacles

Controls (interactive):
  SPACE  — pause / resume
  R      — reset episode
  W      — toggle wind-force lines
  1–7    — switch wind type: 1=calm 2=cold_front 3=squall 4=thermal 5=jet_stream 6=cyclone 7=none
"""

import argparse
import time
import numpy as np
import mujoco
import mujoco.viewer
from stable_baselines3 import PPO

import wind_sim as wind
from env import build_scene_spec, HOVER_THRUST, GOAL_POSITION, DroneDeliveryEnv
from controller import cascaded_control

WIND_KEYS = {
    "1": "calm",
    "2": "cold_front",
    "3": "squall",
    "4": "thermal",
    "5": "jet_stream",
    "6": "cyclone",
    "7": "none",
}

WIND_CHOICES = ["none", "calm", "cold_front", "squall", "thermal", "jet_stream", "cyclone"]


def parse_args():
    p = argparse.ArgumentParser(description="Visualize a trained drone policy in MuJoCo.")
    p.add_argument("--model", default=None,
                   help="Path to model zip (without .zip). "
                        "Defaults to models_full/best_model if --obstacles, else models/best_model.")
    p.add_argument("--wind-type", default="calm", choices=WIND_CHOICES,
                   help="Initial wind type (default: calm).")
    p.add_argument("--wind-speed", type=float, default=1,
                   help="Wind speed scalar (default: 1.2).")
    p.add_argument("--wind-turbulence", type=float, default=0.3,
                   help="Wind turbulence intensity (default: 0.3).")
    p.add_argument("--obstacles", action="store_true", default=False,
                   help="Include obstacles in the scene.")
    p.add_argument("--no-wind", action="store_true", default=False,
                   help="Disable wind entirely (overrides --wind-type).")
    p.add_argument("--wind-lines", action="store_true", default=False,
                   help="Show wind-force lines on startup (toggle with W key).")
    return p.parse_args()


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
        [drone_pos[2]],
        quat,
        lin_vel_body,
        get_sensor(model, data, "body_gyro"),
        get_sensor(model, data, "body_linacc"),
        box_rel_pos_body,
        box_rel_vel_body,
        goal_vec_body,
        last_action,
    ]).astype(np.float32)


def reset_state(model, data):
    mujoco.mj_resetDataKeyframe(model, data, model.key("hover").id)
    data.qpos[:3] = [0.0, 0.0, 1.5]
    data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
    data.qpos[7:10] = [0.0, 0.0, 0.8]
    data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    args = parse_args()

    # Resolve model path
    if args.model is not None:
        model_path = args.model
    elif args.obstacles:
        model_path = "models_full/best_model"
    else:
        model_path = "models/best_model"

    import os
    if not os.path.exists(model_path + ".zip"):
        fallback = "models/ppo_delivery"
        print(f"Warning: {model_path}.zip not found, falling back to {fallback}.zip")
        model_path = fallback

    print(f"Loading policy: {model_path}.zip")
    policy_model = PPO.load(model_path, device="cpu")

    with_wind = not args.no_wind
    wind_type = args.wind_type if with_wind else "none"
    wind_speed = args.wind_speed
    wind_turbulence = args.wind_turbulence

    print(f"Scene:   obstacles={args.obstacles}")
    print(f"Wind:    enabled={with_wind}  type={wind_type}  speed={wind_speed}  turbulence={wind_turbulence}")
    print()
    print("Controls: SPACE=pause  R=reset  W=wind lines  1-7=switch wind type")
    print("  1=calm  2=cold_front  3=squall  4=thermal  5=jet_stream  6=cyclone  7=none")

    spec, obs_pos, obs_size = build_scene_spec(with_obstacles=args.obstacles)
    mj_model = spec.compile()
    mj_data = mujoco.MjData(mj_model)
    reset_state(mj_model, mj_data)

    goal_geom_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
    wind_field_fn = getattr(wind, f"wind_{wind_type}")
    current_wind_type = wind_type
    field_angle = 0.0

    paused = False
    show_wind = args.wind_lines
    step_count = 0
    last_action = np.zeros(4, dtype=np.float32)
    in_obstacle = False  # tracks whether we already printed the collision warning

    def key_callback(keycode):
        nonlocal paused, show_wind, wind_field_fn, current_wind_type, field_angle, step_count, last_action, in_obstacle
        c = chr(keycode)
        if c == " ":
            paused = not paused
            print(f"{'Paused' if paused else 'Resumed'}.")
        elif c == "R":
            reset_state(mj_model, mj_data)
            last_action[:] = 0.0
            step_count = 0
            in_obstacle = False
            print("Reset.")
        elif c == "W":
            show_wind = not show_wind
            print(f"Wind lines: {'on' if show_wind else 'off'}")
        elif c in WIND_KEYS:
            current_wind_type = WIND_KEYS[c]
            wind_field_fn = getattr(wind, f"wind_{current_wind_type}")
            field_angle = 0.0
            print(f"Wind → {current_wind_type}")

    with mujoco.viewer.launch_passive(mj_model, mj_data, key_callback=key_callback) as viewer:
        viewer.sync()

        while viewer.is_running():
            step_start = time.time()

            if not paused:
                if with_wind and current_wind_type != "none":
                    for body_id in range(1, mj_model.nbody):
                        pos = mj_data.xpos[body_id]
                        fx, fy, fz = wind_field_fn(
                            pos, mj_data.time, wind_speed, wind_turbulence, field_angle
                        )
                        mj_data.xfrc_applied[body_id, 0] = 2 * fx
                        mj_data.xfrc_applied[body_id, 1] = 2 * fy
                        mj_data.xfrc_applied[body_id, 2] = 2 * fz

                obs = build_observation(mj_model, mj_data, goal_geom_id, last_action)
                action, _ = policy_model.predict(obs, deterministic=True)
                quat = mj_data.qpos[3:7]
                gyro = get_sensor(mj_model, mj_data, "body_gyro")
                mj_data.ctrl = cascaded_control(quat, gyro, action, HOVER_THRUST)

                mujoco.mj_step(mj_model, mj_data)
                last_action = action.astype(np.float32)
                step_count += 1

                drone_pos = mj_data.qpos[:3]

                # Obstacle collision check — print once on entry, once on exit
                if obs_pos is not None and obs_size is not None:
                    box_pos = mj_data.qpos[7:10]
                    drone_dists = np.linalg.norm(obs_pos - drone_pos, axis=1) - obs_size
                    box_dists   = np.linalg.norm(obs_pos - box_pos,   axis=1) - obs_size
                    currently_in = bool(np.any(drone_dists < 0) or np.any(box_dists < 0))
                    if currently_in and not in_obstacle:
                        who = []
                        if np.any(drone_dists < 0):
                            who.append(f"drone (depth {float(np.max(-drone_dists)):.3f} m)")
                        if np.any(box_dists < 0):
                            who.append(f"box (depth {float(np.max(-box_dists)):.3f} m)")
                        print(
                            f"*** OBSTACLE HIT at step {step_count} | "
                            f"Drone ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}) | "
                            + " + ".join(who) + " ***"
                        )
                    elif not currently_in and in_obstacle:
                        print(f"    (exited obstacle at step {step_count})")
                    in_obstacle = currently_in

                if step_count % 100 == 0:
                    dist = np.linalg.norm(drone_pos - GOAL_POSITION)
                    print(
                        f"Step {step_count:5d} | wind={current_wind_type:<10} | "
                        f"Drone ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}) | "
                        f"Dist to goal: {dist:.2f}"
                    )

            if show_wind and current_wind_type != "none":
                wind.update_wind_lines(
                    viewer, mj_model, wind_field_fn, mj_data,
                    wind_speed, wind_turbulence, field_angle,
                )

            viewer.sync()
            dt = mj_model.opt.timestep - (time.time() - step_start)
            if dt > 0:
                time.sleep(dt)


if __name__ == "__main__":
    main()
