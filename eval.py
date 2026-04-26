"""Evaluate trained PPO agent for package delivery (headless)."""

import os
import numpy as np
from stable_baselines3 import PPO
from env import DroneDeliveryEnv

MODEL_PATH = "models/best_model" if os.path.exists("models/best_model.zip") else "models/ppo_delivery"
print(f"Loading {MODEL_PATH}.zip")
model = PPO.load(MODEL_PATH)

env = DroneDeliveryEnv(max_episode_steps=1000, with_obstacles=False, with_wind=False)

num_episodes = 5
rewards, lengths, outcomes = [], [], []

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0.0
    steps = 0
    done = False
    reason = "timeout"

    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        steps += 1
        if terminated:
            reason = info.get("termination", "terminated")
        done = terminated or truncated

    rewards.append(episode_reward)
    lengths.append(steps)
    outcomes.append(reason)
    print(f"Episode {episode + 1}: reward={episode_reward:.2f}  steps={steps}  end={reason}")

print(f"\nMean reward: {np.mean(rewards):.2f} ± {np.std(rewards):.2f}")
print(f"Mean length: {np.mean(lengths):.1f}")
print(f"Outcomes: {outcomes}")

env.close()
