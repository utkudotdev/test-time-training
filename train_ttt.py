"""Phase 3: Train PPO + auxiliary dynamics-prediction head.

Loads models_obs/best_model (CALM_ONLY) or models_full/best_model, transfers
weights into WindAwarePolicy, then fine-tunes with:
  total_loss = ppo_loss  (handled by SB3)
  aux update = separate Adam step on AuxLossCallback (after each rollout)

The aux head is updated on a SEPARATE optimizer so it does not interfere with
PPO's clipped gradient update. Both touch the shared mlp_extractor, which is
intentional — the encoder gradually becomes wind-aware.

Run:
  uv run python train_ttt.py
Monitor:
  uv run tensorboard --logdir logs_ttt/
"""

import os
import shutil
import numpy as np
import torch as th
import torch.nn.functional as F
from datetime import datetime

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import DroneDeliveryEnv
from policy_ttt import WindAwarePolicy, VEL_IDX, ACCEL_IDX
from train_full import CurriculumCallback, TrainLogCallback, CURRICULUM


N_ENVS = 8
TOTAL_STEPS = 1_300_000
AUX_LR = 3e-4
AUX_COEF = 0.1  # weight on aux loss (relative to keeping ppo scale)
_RUN_TS = datetime.now().strftime("%Y%m%d_%H%M%S")
LOG_DIR = f"logs_ttt/{_RUN_TS}"
MODEL_DIR = f"models_ttt/{_RUN_TS}"
_MODEL_DIR_BASE = "models_ttt"  # canonical best_model lives here for adapt.py

# CALM_ONLY=True:  train only on calm wind; rely on TTT for OOD winds.
#                  Encoder specializes to calm → OOD aux loss stays high → strong adaptation signal.
# CALM_ONLY=False: full domain-randomised curriculum; encoder sees all winds during training.
CALM_ONLY = True

# CALM_ONLY=True  → start from obstacle-trained baseline (models_obs); encoder stays calm-only
# CALM_ONLY=False → start from full model (models_full); already has all-wind knowledge
PRETRAIN_PATH = "models_obs/best_model" if CALM_ONLY else "models_full/best_model"

CALM_CURRICULUM = [
    (0,       dict(phase="calm_gentle", wind_type="calm", speed=0.3, turbulence=0.05)),
    (200_000, dict(phase="calm_medium", wind_type="calm", speed=0.6, turbulence=0.15)),
    (500_000, dict(phase="calm_full",   wind_type="calm", speed=1.0, turbulence=0.30)),
]


def make_env(with_obstacles=False):
    def _init():
        return DroneDeliveryEnv(
            max_episode_steps=1000,
            with_obstacles=with_obstacles,
            with_wind=False,
        )

    return _init


class AuxLossCallback(BaseCallback):
    """After each PPO rollout, update aux_head + encoder with dynamics-prediction loss.

    Uses consecutive (obs_t, action_t, obs_{t+1}) pairs from the rollout buffer.
    Episode-boundary transitions are masked out via episode_starts.
    """

    def __init__(self, aux_coef: float = 0.1, lr: float = 3e-4, verbose: int = 0):
        super().__init__(verbose)
        self.aux_coef = aux_coef
        self.lr = lr
        self._aux_opt: th.optim.Optimizer | None = None

    def _on_training_start(self) -> None:
        policy = self.model.policy
        # Optimise both the aux_head and the shared encoder so the representation
        # becomes wind-aware. Use a lower LR than PPO to avoid dominance.
        params = list(policy.aux_head.parameters()) + list(
            policy.mlp_extractor.parameters()
        )
        self._aux_opt = th.optim.Adam(params, lr=self.lr)

    def _on_rollout_end(self) -> None:
        buf = self.model.rollout_buffer

        obs = th.tensor(buf.observations, dtype=th.float32)  # (T, N, obs_dim)
        acts = th.tensor(buf.actions, dtype=th.float32)  # (T, N, act_dim)
        starts = th.tensor(buf.episode_starts, dtype=th.bool)  # (T, N)

        # Consecutive pairs (t, t+1) — flatten over envs
        obs_t = obs[:-1].reshape(-1, obs.shape[-1])
        obs_t1 = obs[1:].reshape(-1, obs.shape[-1])
        acts_t = acts[:-1].reshape(-1, acts.shape[-1])
        valid = ~starts[1:].reshape(-1)  # mask episode-boundary steps

        if valid.sum() < 32:
            return

        obs_t = obs_t[valid]
        obs_t1 = obs_t1[valid]
        acts_t = acts_t[valid]

        # Target: observed dynamics residual (wind effect shows up here)
        target = th.cat(
            [
                obs_t1[:, VEL_IDX] - obs_t[:, VEL_IDX],
                obs_t1[:, ACCEL_IDX] - obs_t[:, ACCEL_IDX],
            ],
            dim=-1,
        ).detach()

        pred = self.model.policy.predict_aux(obs_t, acts_t)
        loss = F.mse_loss(pred, target)

        self._aux_opt.zero_grad()
        (self.aux_coef * loss).backward()
        self._aux_opt.step()

        self.logger.record("train/aux_loss", loss.item())

    def _on_step(self) -> bool:
        return True


def _transfer_weights(src_policy, dst_policy) -> None:
    """Copy shared layers from a plain MlpPolicy into a WindAwarePolicy.

    WindAwarePolicy has an additional aux_head which is NOT in the source, so
    we transfer only the parts that exist in both. Both policies use 27D obs so
    all layer shapes match exactly.
    """
    for name in ("features_extractor", "mlp_extractor", "action_net", "value_net"):
        getattr(dst_policy, name).load_state_dict(
            getattr(src_policy, name).state_dict(), strict=True
        )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv([make_env(with_obstacles=True) for _ in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    eval_env = SubprocVecEnv([make_env(with_obstacles=True)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))
    eval_env.env_method("set_wind", "calm", 1.0, 0.3)

    model = PPO(
        WindAwarePolicy,
        train_env,
        learning_rate=1e-4,  # fine-tuning LR
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[256, 256], aux_hidden=64),
        tensorboard_log=LOG_DIR,
        verbose=0,
        device="cpu",
    )

    if os.path.exists(PRETRAIN_PATH + ".zip"):
        print(f"Transferring pretrained weights from {PRETRAIN_PATH}.zip")
        old = PPO.load(PRETRAIN_PATH)
        _transfer_weights(old.policy, model.policy)
        del old
        print("Weights transferred — aux_head starts from random init.")
    else:
        print("No pretrained model found — training from scratch (run train.py first).")

    active_curriculum = CALM_CURRICULUM if CALM_ONLY else CURRICULUM

    callbacks = [
        AuxLossCallback(aux_coef=AUX_COEF, lr=AUX_LR, verbose=1),
        CurriculumCallback(train_env, curriculum=active_curriculum, verbose=1),
        TrainLogCallback(active_curriculum, total_steps=TOTAL_STEPS),
        CheckpointCallback(
            save_freq=max(100_000 // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_ttt",
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

    curriculum_label = "calm-only" if CALM_ONLY else "full domain-randomised"
    print(f"Training {TOTAL_STEPS:,} steps, {N_ENVS} envs  [{curriculum_label} curriculum]")
    print("Curriculum:")
    for ts, cfg in active_curriculum:
        print(f"  {ts:>8,}  →  {cfg['phase']}")

    model.learn(
        total_timesteps=TOTAL_STEPS, callback=callbacks, reset_num_timesteps=True
    )

    model.save(os.path.join(MODEL_DIR, "ppo_ttt"))
    print(f"Saved to {MODEL_DIR}/ppo_ttt.zip")
    print(f"Best model → {MODEL_DIR}/best_model.zip")

    # Copy best_model to canonical path so adapt.py can reference it directly.
    os.makedirs(_MODEL_DIR_BASE, exist_ok=True)
    src = os.path.join(MODEL_DIR, "best_model.zip")
    dst = os.path.join(_MODEL_DIR_BASE, "best_model.zip")
    if os.path.exists(src):
        shutil.copy2(src, dst)
        print(f"Canonical    → {dst}  (latest best)")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
