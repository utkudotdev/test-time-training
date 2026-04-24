"""Drone-only Gymnasium env — no box, no tendon.

Simpler task: fly to goal position. Use this to verify the drone
can learn to hover and navigate before tackling the delivery task.
"""

import gymnasium as gym
import numpy as np
import mujoco
import wind_sim as wind
from controller import cascaded_control, quat_to_euler


GOAL_POSITION = np.array([5.0, 0.0, 2.0])
HOVER_THRUST = 3.2495625  # per-motor thrust, holds drone at z=0.3

# Action semantics (cascaded): [thrust_delta, desired_roll, desired_pitch, desired_yaw_rate]
# All in [-1, 1]; controller.py scales to physical units via MAX_TILT / MAX_YAW_RATE / THRUST_DELTA.


class DroneEnv(gym.Env):
    """Drone-only navigation env. Zero policy action = stable hover."""

    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(self, max_episode_steps=1000, with_wind=True):
        self.max_episode_steps = max_episode_steps
        self.with_wind = with_wind
        self.step_count = 0

        spec = mujoco.MjSpec.from_file("x2_only.xml")
        spec.worldbody.add_geom(
            name="goal",
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=[0.15],
            rgba=[0.0, 0.8, 0.2, 0.6],
            pos=GOAL_POSITION.tolist(),
            contype=0,
            conaffinity=0,
        )
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        self.goal_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal"
        )

        # Cascaded action: [thrust_delta, roll_cmd, pitch_cmd, yaw_rate_cmd] in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Obs (21D): z(1) + quat(4) + lin_vel_body(3) + gyro(3) + accel(3) + goal_body(3) + prev_action(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(21,), dtype=np.float32
        )

        self._prev_dist = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)

    def _get_sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim].copy()

    def _get_obs(self):
        drone_pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()  # (w, x, y, z)
        goal_pos = self.data.geom_xpos[self.goal_geom].copy()
        goal_vec_world = goal_pos - drone_pos

        # Rotate goal vector into drone body frame (inverse rotation by quat)
        goal_vec_body = self._rotate_by_conj_quat(goal_vec_world, quat)

        # World linear velocity rotated into body frame
        lin_vel_body = self._rotate_by_conj_quat(self.data.qvel[:3].copy(), quat)

        return np.concatenate([
            [drone_pos[2]],                     # altitude (only useful world coord)
            quat,                               # orientation
            lin_vel_body,                       # linear velocity (body frame)
            self._get_sensor("body_gyro"),      # angular velocity (body frame)
            self._get_sensor("body_linacc"),    # linear acceleration (body frame)
            goal_vec_body,                      # goal direction (body frame)
            self._last_action,                  # previous action (for smoothness)
        ]).astype(np.float32)

    @staticmethod
    def _rotate_by_conj_quat(v, q):
        """Rotate vector v by quaternion conjugate (world → body frame). q = (w, x, y, z)."""
        w, x, y, z = q
        # v' = q* ⊗ v ⊗ q (using q* = (w, -x, -y, -z))
        # Equivalent: R^T @ v where R is rotation matrix from q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z),     2 * (x * z - w * y)],
            [2 * (x * y - w * z),     1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y),     2 * (y * z - w * x),     1 - 2 * (x * x + y * y)],
        ])
        return R @ v

    def _compute_reward(self):
        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        dist = float(np.linalg.norm(drone_pos - goal_pos))

        # Progress reward: strong positive when approaching, strong negative when fleeing
        progress = self._prev_dist - dist
        reward = 10.0 * progress

        # Survival bonus
        reward += 0.1

        # Exponentially-decaying proximity — gives a useful gradient at all distances
        # at d=0: +2.0,  d=2: +0.74,  d=5: +0.16,  d=10: +0.01
        reward += 2.0 * np.exp(-dist / 2.0)

        # Upright penalty (safety net on top of PD controller)
        _, qx, qy, _ = self.data.qpos[3:7]
        body_z_world = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = 1.0 - np.clip(body_z_world, -1.0, 1.0)
        reward -= 10.0 * tilt

        # Damp angular velocity
        reward -= 0.05 * np.linalg.norm(self.data.qvel[3:6])

        # Stillness near goal: penalize velocity only inside 1m radius
        if dist < 1.0:
            reward -= 1.0 * (1.0 - dist) * np.linalg.norm(self.data.qvel[:3])

        # Tiered proximity bonuses (visible signal at each threshold)
        if dist < 0.5:
            reward += 2.0
        if dist < 0.3:
            reward += 5.0
        if dist < 0.15:
            reward += 15.0

        self._prev_dist = dist
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0
        self._last_action = np.zeros(4, dtype=np.float32)

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        self._prev_dist = float(np.linalg.norm(drone_pos - goal_pos))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Cascaded control: high-level command → motor thrusts via PD
        quat = self.data.qpos[3:7]
        gyro = self._get_sensor("body_gyro")
        self.data.ctrl = cascaded_control(quat, gyro, action, HOVER_THRUST)

        if self.with_wind:
            for body_id in range(1, self.model.nbody):
                pos = self.data.xpos[body_id]
                fx, fy = wind.wind_field(pos, self.data.time)
                self.data.xfrc_applied[body_id, 0] = 20 * fx
                self.data.xfrc_applied[body_id, 1] = 20 * fy

        mujoco.mj_step(self.model, self.data)
        self._last_action = action

        obs = self._get_obs()
        reward = self._compute_reward()

        drone_z = self.data.qpos[2]
        drone_xy = self.data.qpos[:2]
        dist = float(np.linalg.norm(self.data.qpos[:3] - self.data.geom_xpos[self.goal_geom]))

        crashed = drone_z < 0.05
        out_of_bounds = np.linalg.norm(drone_xy) > 15.0 or drone_z > 10.0
        terminated = bool(crashed or out_of_bounds)
        truncated = self.step_count >= self.max_episode_steps

        # Strong penalty for crashing / flying off so policy can't abuse early termination
        if terminated:
            reward -= 100.0

        info = {"dist": dist, "crashed": crashed, "oob": out_of_bounds}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
