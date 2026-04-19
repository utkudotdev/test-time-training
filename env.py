"""Gymnasium env for drone package delivery with PPO training."""

import gymnasium as gym
import numpy as np
import mujoco
import wind_sim as wind


GOAL_POSITION = np.array([10.0, 0.0, 2.0])
NUM_OBSTACLES = 10
OBSTACLE_REGION = np.array([[0.5, -10.0, 0.0], [10.0, 10.0, 10.0]])
OBSTACLE_RADIUS_RANGE = np.array([0.2, 1.5])

# Hover thrust per motor, tuned to hold drone+box at z≈0.3
# (original keyframe value 3.25 was for drone only; box adds ~2.0 per motor)
HOVER_THRUST = 5.256
# Max delta from hover that policy can command
ACTION_SCALE = 3.0


def build_scene_spec(seed=None, with_obstacles=True):
    """Build scene programmatically: floor, drone, box, goal, obstacles."""
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

    if with_obstacles:
        rng = np.random.default_rng(seed)
        obs_pos = rng.uniform(
            low=OBSTACLE_REGION[0], high=OBSTACLE_REGION[1], size=(NUM_OBSTACLES, 3)
        )
        obs_size = rng.uniform(
            low=OBSTACLE_RADIUS_RANGE[0],
            high=OBSTACLE_RADIUS_RANGE[1],
            size=NUM_OBSTACLES,
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


class DroneDeliveryEnv(gym.Env):
    """Package delivery environment for Skydio X2 drone.

    Policy outputs residual thrust deltas around hover, so an untrained policy
    (action ~ 0) still hovers instead of falling.
    """

    metadata = {"render_modes": [], "render_fps": 100}

    def __init__(
        self,
        max_episode_steps=1000,
        with_obstacles=False,
        with_wind=True,
        seed=None,
    ):
        self.max_episode_steps = max_episode_steps
        self.with_obstacles = with_obstacles
        self.with_wind = with_wind
        self.step_count = 0

        # Build scene and compile
        spec = build_scene_spec(seed=seed, with_obstacles=with_obstacles)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # Action: residual around hover. Policy outputs [-1, 1], scaled by ACTION_SCALE.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Observation: drone qpos(7) + qvel(6) + box qpos(7) + qvel(6) + sensors(10) + goal(3)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(39,), dtype=np.float32
        )

        self.goal_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
        self._prev_drone_to_goal = None

    def _get_sensor(self, name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr : adr + dim].copy()

    def _get_obs(self):
        drone_qpos = self.data.qpos[:7].copy()
        drone_qvel = self.data.qvel[:6].copy()
        box_qpos = self.data.qpos[7:14].copy()
        box_qvel = self.data.qvel[6:12].copy()
        gyro = self._get_sensor("body_gyro")
        accel = self._get_sensor("body_linacc")
        quat = self._get_sensor("body_quat")
        goal_pos = self.data.geom_xpos[self.goal_geom].copy()
        return np.concatenate(
            [drone_qpos, drone_qvel, box_qpos, box_qvel, gyro, accel, quat, goal_pos]
        ).astype(np.float32)

    def _compute_reward(self, drone_to_goal_prev):
        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        box_pos = self.data.qpos[7:10]

        drone_to_goal = np.linalg.norm(drone_pos - goal_pos)
        box_to_goal = np.linalg.norm(box_pos - goal_pos)

        # Dense shaping: progress toward goal (positive when getting closer)
        progress = drone_to_goal_prev - drone_to_goal
        reward = 5.0 * progress

        # Small survival bonus to avoid learning to crash quickly
        reward += 0.02

        # Proximity bonuses (dense, not sparse)
        reward += 1.0 / (1.0 + drone_to_goal)

        # Delivery bonus (box near goal)
        if box_to_goal < 0.5:
            reward += 10.0
        elif box_to_goal < 1.5:
            reward += 1.0 / (1.0 + box_to_goal)

        # Orientation penalty (keep drone upright)
        drone_quat = self.data.qpos[3:7]
        z_axis_world = 2 * (
            drone_quat[1] * drone_quat[3] + drone_quat[0] * drone_quat[2]
        )
        tilt = 1.0 - np.clip(z_axis_world, -1, 1) ** 2
        reward -= 0.1 * tilt

        # Small control effort penalty
        reward -= 0.001 * np.sum((self.data.ctrl - HOVER_THRUST) ** 2)

        return reward, drone_to_goal

    def _check_termination(self):
        drone_z = self.data.qpos[2]
        box_pos = self.data.qpos[7:10]
        goal_pos = self.data.geom_xpos[self.goal_geom]

        # Crashed
        if drone_z < 0.05:
            return True, "crashed"

        # Drifted too far (sanity)
        drone_xy = self.data.qpos[:2]
        if np.linalg.norm(drone_xy) > 30.0 or drone_z > 15.0:
            return True, "out_of_bounds"

        # Delivery success
        if np.linalg.norm(box_pos - goal_pos) < 0.2:
            return True, "delivered"

        return False, ""

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count = 0

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("hover").id)
        mujoco.mj_forward(self.model, self.data)

        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        self._prev_drone_to_goal = float(np.linalg.norm(drone_pos - goal_pos))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1

        # Residual thrust around hover: action in [-1, 1] → ctrl in [hover ± ACTION_SCALE]
        ctrl = HOVER_THRUST + np.asarray(action, dtype=np.float64) * ACTION_SCALE
        self.data.ctrl = np.clip(ctrl, 0.0, 13.0)

        if self.with_wind:
            for body_id in range(1, self.model.nbody):
                pos = self.data.xpos[body_id]
                fx, fy = wind.wind_field(pos, self.data.time)
                self.data.xfrc_applied[body_id, 0] = 20 * fx
                self.data.xfrc_applied[body_id, 1] = 20 * fy

        mujoco.mj_step(self.model, self.data)

        reward, self._prev_drone_to_goal = self._compute_reward(
            self._prev_drone_to_goal
        )
        terminated, reason = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        info = {"step_count": self.step_count, "termination": reason}
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def render(self):
        pass

    def close(self):
        pass
