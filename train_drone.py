"""Train PPO on drone-only navigation task (no box).

Use this to verify the drone learns to fly before tackling delivery.

Run: uv run python train_drone.py
Monitor: uv run tensorboard --logdir logs_drone/
"""

import os
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import SubprocVecEnv, VecMonitor
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback
from env_drone import DroneEnv


N_ENVS = 8
TOTAL_TIMESTEPS = 500_000
LOG_DIR = "logs_drone"
MODEL_DIR = "models_drone"


def make_env(rank, seed=0):
    def _init():
        return DroneEnv(max_episode_steps=500, with_wind=False)
    return _init


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    train_env = SubprocVecEnv([make_env(i) for i in range(N_ENVS)])
    train_env = VecMonitor(train_env, filename=os.path.join(LOG_DIR, "train"))

    eval_env = SubprocVecEnv([make_env(99)])
    eval_env = VecMonitor(eval_env)

    model = PPO(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        n_steps=1024,
        batch_size=256,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        policy_kwargs=dict(net_arch=[256, 256]),
        tensorboard_log=LOG_DIR,
        verbose=1,
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(50_000 // N_ENVS, 1),
            save_path=MODEL_DIR,
            name_prefix="ppo_drone",
        ),
        EvalCallback(
            eval_env,
            best_model_save_path=MODEL_DIR,
            log_path=LOG_DIR,
            eval_freq=max(20_000 // N_ENVS, 1),
            n_eval_episodes=5,
            deterministic=True,
        ),
    ]

    print(f"Training drone-only for {TOTAL_TIMESTEPS:,} timesteps across {N_ENVS} envs...")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=callbacks)
    model.save(os.path.join(MODEL_DIR, "ppo_drone_final"))

    train_env.close()
    eval_env.close()


if __name__ == "__main__":
    main()
