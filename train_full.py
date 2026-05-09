"""Train PPO with wind curriculum, warm-started from models_obs/best_model.

Assumes train_obs.py has already run (obstacle avoidance is pre-baked).

Curriculum phases (timesteps from the START of this script):
  Phase 0 —       0 … 200 k:  calm speed=0.3      — barely-noticeable gusts
  Phase 1 —  200 k … 500 k:  calm speed=0.6      — moderate push
  Phase 2 —  500 k … 800 k:  calm speed=1.0      — full strength
  Phase 3 —  800 k … 1.3 M:  domain-randomised wind, speed 0.3–1.0

Run:
  uv run python train_full.py
Monitor:
  uv run tensorboard --logdir logs_full/
Visualize:
  uv run mjpython visualize_cfg.py --obstacles --wind-type calm
"""

import os
import shutil
import numpy as np
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import DroneDeliveryEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 1_300_000
LOG_FREQ = 25_000        # print a log line every this many timesteps
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs_full/{_RUN_TS}"
MODEL_DIR = f"models_full/{_RUN_TS}"
_MODEL_DIR_BASE = "models_full"

PRETRAIN_PATH = "models_obs/best_model"

TRAIN_WIND_TYPES = ["calm", "cold_front", "squall", "cyclone"]

CURRICULUM = [
    (0,       dict(phase="calm_gentle", wind_type="calm", speed=0.3, turbulence=0.05)),
    (200_000, dict(phase="calm_medium", wind_type="calm", speed=0.6, turbulence=0.15)),
    (500_000, dict(phase="calm_full",   wind_type="calm", speed=1.0, turbulence=0.3)),
    (800_000, dict(phase="randomize")),
]


def make_env(seed=0, with_obstacles=True):
    def _init():
        env = DroneDeliveryEnv(
            max_episode_steps=1000,
            with_obstacles=with_obstacles,
            with_wind=False,  # curriculum callback enables wind
        )
        env.reset(seed=seed)
        return env
    return _init


class CurriculumCallback(BaseCallback):
    """Advances wind curriculum."""

    def __init__(self, train_env, curriculum=None, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self._curriculum = curriculum if curriculum is not None else CURRICULUM
        self._current_phase_idx = -1

    def _apply_phase(self, cfg):
        phase = cfg["phase"]
        if phase == "randomize":
            self.train_env.env_method(
                "enable_wind_randomization",
                TRAIN_WIND_TYPES,
                (0.3, 1.0),
                (0.05, 0.3),
            )
            if self.verbose:
                print(f"\n[Curriculum] → randomize  types={TRAIN_WIND_TYPES}")
        else:
            self.train_env.env_method(
                "set_wind", cfg["wind_type"], cfg["speed"], cfg["turbulence"]
            )
            if self.verbose:
                print(
                    f"\n[Curriculum] → {phase}  "
                    f"wind={cfg['wind_type']}  speed={cfg['speed']}  turb={cfg['turbulence']}"
                )

    def _on_step(self) -> bool:
        for i, (start_ts, cfg) in enumerate(self._curriculum):
            if self.num_timesteps >= start_ts and i > self._current_phase_idx:
                self._current_phase_idx = i
                self._apply_phase(cfg)
        return True


class TrainLogCallback(BaseCallback):
    """Prints a compact log line every LOG_FREQ timesteps."""

    def __init__(self, curriculum, total_steps=TOTAL_TIMESTEPS, log_freq=LOG_FREQ):
        super().__init__(verbose=0)
        self._curriculum = curriculum
        self._total_steps = total_steps
        self._log_freq = log_freq
        self._last_log = 0

    def _phase_name(self):
        name = self._curriculum[0][1]["phase"]
        for start_ts, cfg in self._curriculum:
            if self.num_timesteps >= start_ts:
                name = cfg["phase"]
        return name

    def _on_step(self) -> bool:
        if self.num_timesteps - self._last_log < self._log_freq:
            return True
        self._last_log = self.num_timesteps

        buf = self.model.ep_info_buffer
        if not buf:
            return True

        mean_rew = float(np.mean([ep["r"] for ep in buf]))
        mean_len = float(np.mean([ep["l"] for ep in buf]))
        progress = self.num_timesteps / self._total_steps * 100
        print(
            f"[{self.num_timesteps:>9,} / {self._total_steps:,}  {progress:4.1f}%]  "
            f"phase={self._phase_name():<16}  "
            f"ep_rew={mean_rew:>8.1f}  ep_len={mean_len:>6.1f}"
        )
        return True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv(
        [make_env(i, with_obstacles=True) for i in range(N_ENVS)]
    )
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    eval_env = SubprocVecEnv([make_env(100, with_obstacles=True)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))
    eval_env.env_method("set_wind", "calm", 1.0, 0.3)

    if os.path.exists(PRETRAIN_PATH + ".zip"):
        print(f"Warm-starting from {PRETRAIN_PATH}.zip")
        model = PPO.load(
            PRETRAIN_PATH,
            env=train_env,
            tensorboard_log=LOG_DIR,
            learning_rate=1e-4,
            verbose=0,
        )
    else:
        print(f"No pretrained model at {PRETRAIN_PATH}.zip — training from scratch.")
        print("Tip: run train.py first for best results.")
        model = PPO(
            "MlpPolicy",
            train_env,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=256,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            policy_kwargs=dict(net_arch=[256, 256]),
            tensorboard_log=LOG_DIR,
            verbose=0,
        )

    print(f"\nTraining for {TOTAL_TIMESTEPS:,} steps  ({N_ENVS} envs)")
    print("Curriculum:")
    for ts, cfg in CURRICULUM:
        print(f"  {ts:>9,}  →  {cfg['phase']}")
    print()

    callbacks = [
        CurriculumCallback(train_env, verbose=1),
        TrainLogCallback(CURRICULUM, log_freq=LOG_FREQ),
        CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_full",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(50_000 // N_ENVS, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
            verbose=0,
        ),
    ]

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,
    )

    final_path = os.path.join(MODEL_DIR, "ppo_full")
    model.save(final_path)
    print(f"\nFinal  → {final_path}.zip")
    print(f"Best   → {MODEL_DIR}/best_model.zip")

    os.makedirs(_MODEL_DIR_BASE, exist_ok=True)
    src = os.path.join(MODEL_DIR, "best_model.zip")
    dst = os.path.join(_MODEL_DIR_BASE, "best_model.zip")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Canon  → {dst}")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
