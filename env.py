import gymnasium as gym
import numpy as np
import mujoco


class DroneDeliveryEnv(gym.Env):
    """Package delivery environment for Skydio X2 drone with PPO training."""

    metadata = {"render_modes": ["human"], "render_fps": 100}

    def __init__(self, render_mode=None, max_episode_steps=1000):
        self.render_mode = render_mode
        self.max_episode_steps = max_episode_steps
        self.step_count = 0

        with open("example.xml") as f:
            self.model = mujoco.MjModel.from_xml_string(f.read())
        self.data = mujoco.MjData(self.model)

        self.viewer = None

        # Action space: 4 thrust motors, range [0, 13]
        self.action_space = gym.spaces.Box(
            low=0, high=13, shape=(4,), dtype=np.float32
        )

        # Observation space: drone qpos/qvel (13), box qpos/qvel (13), sensors (10), goal pos (3)
        # Total: 39 dimensions
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )

        self.goal_geom = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal"
        )

    def _get_obs(self):
        """Get full observation: drone state + box state + sensors + goal."""
        # Drone body (first body after world)
        drone_qpos = self.data.qpos[:7].copy()  # pos(3) + quat(4)
        drone_qvel = self.data.qvel[:6].copy()  # linear(3) + angular(3)

        # Box body
        box_qpos = self.data.qpos[7:14].copy()  # pos(3) + quat(4)
        box_qvel = self.data.qvel[6:12].copy()  # linear(3) + angular(3)

        # Sensors
        gyro = self._get_sensor("body_gyro")
        accel = self._get_sensor("body_linacc")
        quat = self._get_sensor("body_quat")

        # Goal position
        goal_pos = self.data.geom_xpos[self.goal_geom].copy()

        obs = np.concatenate(
            [drone_qpos, drone_qvel, box_qpos, box_qvel, gyro, accel, quat, goal_pos]
        )
        return obs.astype(np.float32)

    def _get_sensor(self, name):
        """Extract sensor data by name."""
        sensor_id = mujoco.mj_name2id(
            self.model, mujoco.mjtObj.mjOBJ_SENSOR, name
        )
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr : adr + dim].copy()

    def _compute_reward(self):
        """Compute reward: distance to goal + task bonuses."""
        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        box_pos = self.data.qpos[7:10]

        # Distance rewards
        drone_to_goal = np.linalg.norm(drone_pos - goal_pos)
        box_to_goal = np.linalg.norm(box_pos - goal_pos)

        # Main reward: negative distance (encourage getting closer)
        reward = -0.1 * drone_to_goal

        # Bonus for box proximity to goal (delivery task)
        if box_to_goal < 0.5:
            reward += 1.0  # Bonus for successful delivery

        # Control penalty (fuel cost)
        reward -= 0.001 * np.sum(self.data.ctrl**2)

        return reward

    def _check_termination(self):
        """Check if episode should terminate."""
        drone_z = self.data.qpos[2]
        box_z = self.data.qpos[9]

        # Crashed (hit ground)
        if drone_z < 0.05 or box_z < 0.0:
            return True, "crashed"

        # Success (box at goal)
        box_pos = self.data.qpos[7:10]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        if np.linalg.norm(box_pos - goal_pos) < 0.15:
            return True, "delivered"

        return False, ""

    def reset(self, seed=None, options=None):
        """Reset environment to initial state."""
        super().reset(seed=seed)
        self.step_count = 0

        # Reset to hover keyframe
        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("hover").id)
        mujoco.mj_forward(self.model, self.data)

        obs = self._get_obs()
        info = {}
        return obs, info

    def step(self, action):
        """Execute one step of environment."""
        self.step_count += 1

        # Apply action (thrust motor commands)
        self.data.ctrl = np.clip(action, 0, 13)

        # Apply wind forces
        for body_id in range(1, self.model.nbody):
            pos = self.data.xpos[body_id]
            fx, fy = self._wind_field(pos, self.data.time)
            self.data.xfrc_applied[body_id, 0] = 20 * fx
            self.data.xfrc_applied[body_id, 1] = 20 * fy

        mujoco.mj_step(self.model, self.data)

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated, _ = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        info = {"step_count": self.step_count}

        if self.render_mode == "human":
            self.render()

        return obs, reward, terminated, truncated, info

    def _wind_field(self, pos, t, speed=1.0, turbulence=0.3):
        """Compute wind field at position."""
        cx, cy = 0.0, 0.0
        dx, dy = pos[0] - cx, pos[1] - cy
        r = np.sqrt(dx * dx + dy * dy) + 0.001
        s = 1.0 / (r * 6 + 0.4)
        u = (-dy * s + dx * 0.18) * 1.8 * speed
        v = (dx * s + dy * 0.18) * 1.8 * speed
        u += np.sin(pos[0] * 4 + t * 0.5) * turbulence * 0.5
        v += np.cos(pos[1] * 4 + t * 0.5) * turbulence * 0.5
        return u, v

    def render(self):
        """Render environment using MuJoCo viewer."""
        if self.viewer is None:
            self.viewer = mujoco.Viewer(self.model, self.data)
        self.viewer.sync()

    def close(self):
        """Close viewer."""
        if self.viewer is not None:
            self.viewer.close()
