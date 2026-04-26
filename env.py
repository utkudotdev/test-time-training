"""Gymnasium env for drone package delivery with PPO training."""

import gymnasium as gym
import numpy as np
import mujoco
import wind_sim as wind
from controller import cascaded_control


GOAL_POSITION = np.array([10.0, 0.0, 2.0])
NUM_OBSTACLES = 10
OBSTACLE_REGION = np.array([[0.5, -10.0, 0.0], [10.0, 10.0, 10.0]])
OBSTACLE_RADIUS_RANGE = np.array([0.2, 1.5])

# Calibrated hover thrust for drone + suspended box system (verified by binary search).
# At this per-motor thrust with cascaded PD control, drone+box holds altitude.
HOVER_THRUST = 5.702


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
        wind_type="calm",   # "none", "calm", "cold_front", "squall", "thermal", "jet_stream"
        wind_speed=1.0,
        wind_turbulence=0.3,
        seed=None,
    ):
        self.max_episode_steps = max_episode_steps
        self.with_obstacles = with_obstacles
        self.with_wind = with_wind
        self.wind_speed = wind_speed
        self.wind_turbulence = wind_turbulence
        self._wind_field_fn = getattr(wind, f"wind_{wind_type}")
        self._wind_angle = 0.0
        self.step_count = 0

        # Build scene and compile
        spec = build_scene_spec(seed=seed, with_obstacles=with_obstacles)
        self.model = spec.compile()
        self.data = mujoco.MjData(self.model)

        # Cascaded action: [thrust_delta, roll, pitch, yaw] in [-1, 1]
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

        # Obs (27D): z(1) + quat(4) + lin_vel_body(3) + gyro(3) + accel(3)
        #          + box_rel_pos_body(3) + box_rel_vel_body(3) + goal_body(3) + last_action(4)
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(27,), dtype=np.float32
        )

        self.goal_geom = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, "goal")
        self._prev_drone_to_goal = 0.0
        self._last_action = np.zeros(4, dtype=np.float32)

    def _get_sensor(self, name):
        sensor_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SENSOR, name)
        adr = self.model.sensor_adr[sensor_id]
        dim = self.model.sensor_dim[sensor_id]
        return self.data.sensordata[adr : adr + dim].copy()

    @staticmethod
    def _rotate_by_conj_quat(v, q):
        """Rotate v by conjugate of quat q=(w,x,y,z) — world → body frame."""
        w, x, y, z = q
        R = np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y + w * z),     2 * (x * z - w * y)],
            [2 * (x * y - w * z),     1 - 2 * (x * x + z * z), 2 * (y * z + w * x)],
            [2 * (x * z + w * y),     2 * (y * z - w * x),     1 - 2 * (x * x + y * y)],
        ])
        return R @ v

    def _get_obs(self):
        drone_pos = self.data.qpos[:3].copy()
        quat = self.data.qpos[3:7].copy()
        box_pos = self.data.qpos[7:10].copy()
        goal_pos = self.data.geom_xpos[self.goal_geom].copy()

        lin_vel_body = self._rotate_by_conj_quat(self.data.qvel[:3].copy(), quat)
        box_rel_pos_body = self._rotate_by_conj_quat(box_pos - drone_pos, quat)
        box_rel_vel_body = self._rotate_by_conj_quat(
            self.data.qvel[6:9].copy() - self.data.qvel[:3].copy(), quat
        )
        goal_vec_body = self._rotate_by_conj_quat(goal_pos - drone_pos, quat)

        return np.concatenate([
            [drone_pos[2]],
            quat,
            lin_vel_body,
            self._get_sensor("body_gyro"),
            self._get_sensor("body_linacc"),
            box_rel_pos_body,
            box_rel_vel_body,
            goal_vec_body,
            self._last_action,
        ]).astype(np.float32)

    def _compute_reward(self, drone_to_goal_prev):
        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        box_pos = self.data.qpos[7:10]

        drone_to_goal = float(np.linalg.norm(drone_pos - goal_pos))
        box_to_goal = float(np.linalg.norm(box_pos - goal_pos))

        # Progress reward: strong signal for moving the drone toward the goal
        progress = drone_to_goal_prev - drone_to_goal
        reward = 10.0 * progress

        # Survival bonus
        reward += 0.1

        # Exponential proximity — useful gradient at all distances
        reward += 2.0 * np.exp(-drone_to_goal / 3.0)

        # Upright penalty (safety net on top of PD controller)
        _, qx, qy, _ = self.data.qpos[3:7]
        body_z_world = 1.0 - 2.0 * (qx * qx + qy * qy)
        tilt = 1.0 - np.clip(body_z_world, -1.0, 1.0)
        reward -= 10.0 * tilt

        # Damp angular velocity
        reward -= 0.05 * np.linalg.norm(self.data.qvel[3:6])

        # Stillness near goal
        if drone_to_goal < 1.5:
            reward -= 0.5 * (1.5 - drone_to_goal) * np.linalg.norm(self.data.qvel[:3])

        # Tiered bonuses for drone near goal
        if drone_to_goal < 0.5: reward += 2.0
        if drone_to_goal < 0.3: reward += 5.0

        # Delivery bonus for BOX near goal (the actual task)
        if box_to_goal < 0.5: reward += 10.0
        if box_to_goal < 0.2: reward += 25.0

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
        self._last_action = np.zeros(4, dtype=np.float32)

        if getattr(self, "_randomize_wind", False):
            wtype = self.np_random.choice(self._rand_wind_types)
            spd = float(self.np_random.uniform(*self._rand_speed_range))
            turb = float(self.np_random.uniform(*self._rand_turbulence_range))
            self.set_wind(wtype, spd, turb)

        mujoco.mj_resetDataKeyframe(self.model, self.data, self.model.key("hover").id)

        # Drone at z=1.5, box hanging at z=0.8 → tendon fully taut (0.6m).
        self.data.qpos[:3] = [0.0, 0.0, 1.5]
        self.data.qpos[3:7] = [1.0, 0.0, 0.0, 0.0]
        self.data.qpos[7:10] = [0.0, 0.0, 0.8]
        self.data.qpos[10:14] = [1.0, 0.0, 0.0, 0.0]
        self.data.qvel[:] = 0.0

        mujoco.mj_forward(self.model, self.data)

        drone_pos = self.data.qpos[:3]
        goal_pos = self.data.geom_xpos[self.goal_geom]
        self._prev_drone_to_goal = float(np.linalg.norm(drone_pos - goal_pos))

        return self._get_obs(), {}

    def step(self, action):
        self.step_count += 1
        action = np.asarray(action, dtype=np.float32)

        # Cascaded PD control: high-level action → motor commands
        quat = self.data.qpos[3:7]
        gyro = self._get_sensor("body_gyro")
        self.data.ctrl = cascaded_control(quat, gyro, action, HOVER_THRUST)

        if self.with_wind:
            for body_id in range(1, self.model.nbody):
                pos = self.data.xpos[body_id]
                fx, fy = self._wind_field_fn(
                    pos, self.data.time, self.wind_speed,
                    self.wind_turbulence, self._wind_angle,
                )
                self.data.xfrc_applied[body_id, 0] = 2 * fx
                self.data.xfrc_applied[body_id, 1] = 2 * fy

        mujoco.mj_step(self.model, self.data)
        self._last_action = action

        reward, self._prev_drone_to_goal = self._compute_reward(
            self._prev_drone_to_goal
        )
        terminated, reason = self._check_termination()
        truncated = self.step_count >= self.max_episode_steps

        # Strong penalty for crash or going out of bounds, bonus for delivery
        if terminated:
            if reason == "delivered":
                reward += 100.0
            else:
                reward -= 100.0

        info = {"step_count": self.step_count, "termination": reason}
        return self._get_obs(), float(reward), bool(terminated), bool(truncated), info

    def set_wind(self, wind_type="none", speed=1.0, turbulence=0.3):
        """Hot-swap wind config (called by curriculum callback via env_method)."""
        self.with_wind = wind_type != "none"
        self._wind_field_fn = getattr(wind, f"wind_{wind_type}")
        self.wind_speed = speed
        self.wind_turbulence = turbulence

    def enable_wind_randomization(
        self,
        types=("calm", "cold_front", "squall", "thermal", "jet_stream"),
        speed_range=(0.5, 1.5),
        turbulence_range=(0.1, 0.5),
    ):
        """Per-episode random wind type + strength (domain randomization)."""
        self._randomize_wind = True
        self._rand_wind_types = list(types)
        self._rand_speed_range = speed_range
        self._rand_turbulence_range = turbulence_range

    def render(self):
        pass

    def close(self):
        pass
