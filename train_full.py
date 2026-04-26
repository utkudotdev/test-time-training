"""Train PPO with wind curriculum, starting from a pretrained no-wind checkpoint.

Wind force scale: 2x (down from original 20x). With 2x the drone can physically
resist speed=1.0 calm wind at all positions along the flight path (max ~6.5 N vs
8.9 N lateral authority). At 20x this was impossible — 65 N at the goal.

Workflow:
  1. Run train.py first (or use an existing models/best_model.zip).
  2. Run this script — it loads that checkpoint then adds wind progressively.

Curriculum phases (timesteps from the START of this script):
  Phase 0 —      0 … 200 k: calm speed=0.3  — barely-noticeable gusts
  Phase 1 — 200 k … 500 k:  calm speed=0.6  — moderate push (~3.9 N at goal)
  Phase 2 — 500 k … 800 k:  calm speed=1.0  — full strength (~6.5 N at goal)
  Phase 3 — 800 k … 1.5 M:  domain-randomised wind, speed 0.5–1.0

Run:
  uv run python train_full.py
Monitor:
  uv run tensorboard --logdir logs_full/
Visualize:
  Change MODEL_PATH in visualize_mujoco.py to "models_full/best_model" then:
  uv run mjpython visualize_mujoco.py
"""

import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, EvalCallback
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import DroneDeliveryEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 1_500_000
LOG_DIR = "logs_full"
MODEL_DIR = "models_full"

# Pretrained no-wind checkpoint to warm-start from
PRETRAIN_PATH = "models/best_model"

ALL_WIND_TYPES = ["calm", "cold_front", "squall", "thermal", "jet_stream"]
TRAIN_WIND_TYPES = ["calm", "cold_front", "squall"]
TEST_WIND_TYPES = ["thermal", "jet_stream"]  # only in final randomization phase; not seen during training but still within drone's control authority, so should be manageable with good generalization. Note that the two policies may have different action scales, so even if the pretrained model performs worse with these unseen wind types, it can still be qualitatively successful (i.e. can learn to hover and navigate before tackling the delivery task).
# (start_timestep, config)
# Wind forces with 2x multiplier are now within the drone's 8.9 N lateral budget.
CURRICULUM = [
    (0,       dict(phase="calm_gentle", wind_type="calm", speed=0.3, turbulence=0.05)),
    (200_000, dict(phase="calm_medium", wind_type="calm", speed=0.6, turbulence=0.15)),
    (500_000, dict(phase="calm_full",   wind_type="calm", speed=1.0, turbulence=0.3)),
    (800_000, dict(phase="randomize")),
]


def make_env(rank, seed=0, with_obstacles=False):
    def _init():
        return DroneDeliveryEnv(
            max_episode_steps=1000,
            with_obstacles=with_obstacles,
            with_wind=False,   # curriculum callback enables wind
            seed=seed + rank,
        )
    return _init


class CurriculumCallback(BaseCallback):
    """Advances wind curriculum and logs phase index to tensorboard."""

    def __init__(self, train_env, verbose=1):
        super().__init__(verbose)
        self.train_env = train_env
        self._current_phase_idx = -1

    def _apply_phase(self, cfg):
        phase = cfg["phase"]
        if phase == "randomize":
            self.train_env.env_method(
                "enable_wind_randomization",
                TRAIN_WIND_TYPES,
                (0.3, 1.0),   # speed range within drone's lateral control authority
                (0.05, 0.3),
            )
            if self.verbose:
                print(f"\n[Curriculum] → domain-randomised wind {ALL_WIND_TYPES}")
        else:
            self.train_env.env_method(
                "set_wind", cfg["wind_type"], cfg["speed"], cfg["turbulence"]
            )
            if self.verbose:
                print(
                    f"\n[Curriculum] → {phase}  "
                    f"wind={cfg['wind_type']}  speed={cfg['speed']}  turb={cfg['turbulence']}"
                )
        self.logger.record("curriculum/phase_idx", self._current_phase_idx)

    def _on_step(self) -> bool:
        for i, (start_ts, cfg) in enumerate(CURRICULUM):
            if self.num_timesteps >= start_ts and i > self._current_phase_idx:
                self._current_phase_idx = i
                self._apply_phase(cfg)
        return True


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv(
        [make_env(i, with_obstacles=False) for i in range(N_ENVS)]
    )
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    # Eval env: fixed calm wind so comparison is consistent across all phases
    eval_env = SubprocVecEnv([make_env(100, with_obstacles=False)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))
    eval_env.env_method("set_wind", "calm", 1.0, 0.3)

    # Load pretrained checkpoint so phase 0 (basic navigation) is already solved
    if os.path.exists(PRETRAIN_PATH + ".zip"):
        print(f"Loading pretrained model from {PRETRAIN_PATH}.zip")
        model = PPO.load(
            PRETRAIN_PATH,
            env=train_env,
            tensorboard_log=LOG_DIR,
            # keep original hyperparams; lower LR for fine-tuning
            learning_rate=1e-4,
        )
    else:
        print(f"No pretrained model found at {PRETRAIN_PATH}.zip — training from scratch.")
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
            verbose=1,
        )

    callbacks = [
        CurriculumCallback(train_env, verbose=1),
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
        ),
    ]

    print(f"Fine-tuning for {TOTAL_TIMESTEPS:,} steps across {N_ENVS} envs")
    print("Curriculum:")
    for ts, cfg in CURRICULUM:
        print(f"  {ts:>8,}  →  {cfg['phase']}")

    model.learn(
        total_timesteps=TOTAL_TIMESTEPS,
        callback=callbacks,
        reset_num_timesteps=True,   # curriculum counts from 0 in this run
    )

    final_path = os.path.join(MODEL_DIR, "ppo_full")
    model.save(final_path)
    print(f"\nFinal model  → {final_path}.zip")
    print(f"Best model   → {MODEL_DIR}/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
