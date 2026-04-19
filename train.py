"""Train PPO agent for drone package delivery.

Features:
  - Residual action around hover thrust (stable baseline behavior)
  - Vectorized envs for faster rollout collection
  - Tensorboard logging, checkpointing, best-model saving

Run: uv run python train.py
Monitor: uv run tensorboard --logdir logs/
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import (
    CheckpointCallback,
    EvalCallback,
)
from env import DroneDeliveryEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 1_000_000
LOG_DIR = "logs"
MODEL_DIR = "models"


def make_env(rank, seed=0, with_obstacles=False):
    def _init():
        env = DroneDeliveryEnv(
            max_episode_steps=1000,
            with_obstacles=with_obstacles,
            with_wind=False,
            seed=seed + rank,
        )
        return env
    return _init


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    # Training envs (no obstacles initially for easier learning)
    train_env = SubprocVecEnv([make_env(i, with_obstacles=False) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train_monitor"))

    # Eval env (single, for periodic evaluation)
    eval_env = SubprocVecEnv([make_env(100, with_obstacles=False)])
    eval_env = VecMonitor(eval_env, filename=os.path.join(LOG_DIR, "eval_monitor"))

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
        CheckpointCallback(
            save_freq=max(50_000 // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_delivery",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(25_000 // N_ENVS, 1),
            n_eval_episodes=5,
            deterministic=True,
            render=False,
        ),
    ]

    print(f"Training for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)

    final_path = os.path.join(MODEL_DIR, "ppo_delivery")
    model.save(final_path)
    print(f"Final model saved to {final_path}.zip")
    print(f"Best model saved to {MODEL_DIR}/best_model.zip")

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
