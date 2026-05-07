"""Phase 2: Train PPO + auxiliary dynamics-prediction head.

Loads the pretrained no-wind policy (models/best_model.zip), transfers its
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
import numpy as np
import torch as th
import torch.nn.functional as F

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import (
    BaseCallback,
    CheckpointCallback,
    EvalCallback,
)
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor

from env import DroneDeliveryEnv
from policy_ttt import WindAwarePolicy, VEL_IDX, ACCEL_IDX
from train_full import CurriculumCallback, CURRICULUM, ALL_WIND_TYPES


N_ENVS = 8
TOTAL_STEPS = 1_500_000
AUX_LR = 3e-4
AUX_COEF = 0.1  # weight on aux loss (relative to keeping ppo scale)
LOG_DIR = "logs_ttt"
MODEL_DIR = "models_ttt"
PRETRAIN_PATH = "models/best_model"  # no-wind pretrained checkpoint


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
    we transfer only the parts that exist in both.
    """
    for name in ("features_extractor", "mlp_extractor", "action_net", "value_net"):
        getattr(dst_policy, name).load_state_dict(
            getattr(src_policy, name).state_dict(), strict=True
        )


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv([make_env(with_obstacles=False) for _ in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    eval_env = SubprocVecEnv([make_env(with_obstacles=False)])
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
        verbose=1,
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

    callbacks = [
        AuxLossCallback(aux_coef=AUX_COEF, lr=AUX_LR, verbose=1),
        CurriculumCallback(train_env, verbose=1),
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

    print(f"Training {TOTAL_STEPS:,} steps, {N_ENVS} envs")
    print("Curriculum:")
    for ts, cfg in CURRICULUM:
        print(f"  {ts:>8,}  →  {cfg['phase']}")

    model.learn(
        total_timesteps=TOTAL_STEPS, callback=callbacks, reset_num_timesteps=True
    )

    model.save(os.path.join(MODEL_DIR, "ppo_ttt"))
    print(f"Saved to {MODEL_DIR}/ppo_ttt.zip")
    print(f"Best model → {MODEL_DIR}/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
