"""Drone-only Gymnasium env — no box, no tendon.

Simpler task: fly to goal position. Use this to verify the drone
can learn to hover and navigate before tackling the delivery task.
"""

import gymnasium as gym
import numpy as np
import mujoco
import wind_sim as wind


GOAL_POSITION = np.array([5.0, 0.0, 2.0])  # closer goal for easier learning
HOVER_THRUST = 3.2495625                    # verified: holds drone at z=0.3
ACTION_SCALE = 2.0                          # policy output [-1,1] → ctrl delta ±2


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

        # Action: residual from hover, policy outputs [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Obs: qpos(7) + qvel(6) + gyro(3) + accel(3) + quat(4) + goal_vec(3) = 26
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(26,), dtype=np.float32
        )

        self._prev_dist = None

    def _get_sensor(self, name):
        sid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sid]
        dim = self.model.sensor_dim[sid]
        return self.data.sensordata[adr : adr + dim].copy()

    def _get_obs(self):
        drone_pos = self.data.qpos[:3].copy()
        goal_pos = self.data.geom_xpos[self.goal_geom].copy()
        goal_vec = goal_pos - drone_pos  # relative vector to goal
        return np.concatenate([
            self.data.qpos[:7],     # pos + quat
            self.data.qvel[:6],     # linear + angular vel
            self._get_sensor("body_gyro"),
            self._get_sensor("body_linacc"),
            self._get_sensor("body_quat"),
            goal_vec,
        ]).astype(np.float32)

    def _compute_reward(self):
        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        dist = float(np.linalg.norm(drone_pos - goal_pos))

        # Progress reward (dense)
        progress = self._prev_dist - dist
        reward = 5.0 * progress

        # Survival bonus
        reward += 0.05

        # Proximity bonus
        reward += 0.5 / (1.0 + dist)

        # Upright penalty — quaternion w=qpos[3], drone is upright when
        # z-axis of body aligns with world z-axis
        qw, qx, qy, qz = self.data.qpos[3:7]
        # World z-component of drone's body z-axis
        body_z_world = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = 1.0 - np.clip(body_z_world, -1.0, 1.0)
        reward -= 10.0 * tilt  # strong penalty for tilting

        # Angular velocity penalty (prevent spinning)
        reward -= 0.05 * np.linalg.norm(self.data.qvel[3:6])

        # Control smoothness penalty
        reward -= 0.001 * np.sum((self.data.ctrl - HOVER_THRUST) ** 2)

        # Reach goal bonus
        if dist < 0.3:
            reward += 20.0

        self._prev_dist = dist
        return reward

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetDataKeyframe(self.model, self.data, 0)
        mujoco.mj_forward(self.model, self.data)

        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        self._prev_dist = float(np.linalg.norm(drone_pos - goal_pos))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Residual action around hover: zero action = stable hover
        ctrl = HOVER_THRUST + np.asarray(action, dtype=np.float64) * ACTION_SCALE
        self.data.ctrl = np.clip(ctrl, 0.0, 13.0)

        if self.with_wind:
            for body_id in range(1, self.model.nbody):
                pos = self.data.xpos[body_id]
                fx, fy = wind.wind_field(pos, self.data.time)
                self.data.xfrc_applied[body_id, 0] = 20 * fx
                self.data.xfrc_applied[body_id, 1] = 20 * fy

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()

        drone_z = self.data.qpos[2]
        dist = float(np.linalg.norm(self.data.qpos[:3] - self.data.geom_xpos[self.goal_geom]))

        terminated = bool(drone_z < 0.05 or np.linalg.norm(self.data.qpos[:2]) > 20.0)
        truncated = self.step_count >= self.max_episode_steps

        info = {"dist": dist, "crashed": drone_z < 0.05}
        return obs, float(reward), terminated, truncated, info

    def render(self):
        pass

    def close(self):
        pass
