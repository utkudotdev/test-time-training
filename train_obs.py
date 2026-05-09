"""Phase 1b: Fine-tune baseline on obstacles (no wind).

Starts from models/best_model (plain navigation, no wind, no obstacles) and
teaches obstacle avoidance in 200 k steps before any wind is introduced.
Output (models_obs/best_model) is the shared warm-start for train_full.py and
train_ttt.py (CALM_ONLY=True), which skip the obstacles_only phase entirely.

Workflow:
  1. uv run python train.py        → models/best_model
  2. uv run python train_obs.py    → models_obs/best_model
  3. uv run python train_full.py   → models_full/best_model
     uv run python train_ttt.py    → models_ttt/best_model

Run:
  uv run python train_obs.py
Monitor:
  uv run tensorboard --logdir logs_obs/
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


N_ENVS          = 8
TOTAL_TIMESTEPS = 200_000
LOG_FREQ        = 25_000
PRETRAIN_PATH   = "models/best_model"

_RUN_TS       = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR       = f"logs_obs/{_RUN_TS}"
MODEL_DIR     = f"models_obs/{_RUN_TS}"
_MODEL_DIR_BASE = "models_obs"


def make_env(seed=0):
    def _init():
        env = DroneDeliveryEnv(
            max_episode_steps=1000,
            with_obstacles=True,
            with_wind=False,
        )
        env.reset(seed=seed)
        return env
    return _init


class TrainLogCallback(BaseCallback):
    """Prints a compact log line every LOG_FREQ timesteps."""

    def __init__(self, total_steps=TOTAL_TIMESTEPS, log_freq=LOG_FREQ):
        super().__init__(verbose=0)
        self._total_steps = total_steps
        self._log_freq = log_freq
        self._last_log = 0

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
            f"[{self.num_timesteps:>7,} / {self._total_steps:,}  {progress:4.1f}%]  "
            f"phase=obstacles_only      "
            f"ep_rew={mean_rew:>8.1f}  ep_len={mean_len:>6.1f}"
        )
        return True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    eval_env = SubprocVecEnv([make_env(100)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))

    if not os.path.exists(PRETRAIN_PATH + ".zip"):
        raise FileNotFoundError(
            f"{PRETRAIN_PATH}.zip not found — run train.py first."
        )

    print(f"Warm-starting from {PRETRAIN_PATH}.zip")
    model = PPO.load(
        PRETRAIN_PATH,
        env=train_env,
        tensorboard_log=LOG_DIR,
        learning_rate=1e-4,
        verbose=0,
    )

    print(f"\nTraining {TOTAL_TIMESTEPS:,} steps  ({N_ENVS} envs)  phase=obstacles_only")

    callbacks = [
        TrainLogCallback(total_steps=TOTAL_TIMESTEPS, log_freq=LOG_FREQ),
        CheckpointCallback(
            save_freq=max(50_000 // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_obs",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(25_000 // N_ENVS, 1),
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

    final_path = os.path.join(MODEL_DIR, "ppo_obs")
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
