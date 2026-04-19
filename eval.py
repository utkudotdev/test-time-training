"""Evaluate trained PPO agent for package delivery."""

from stable_baselines3 import PPO
from env import DroneDeliveryEnv
import numpy as np

# Load trained model
model = PPO.load("models/ppo_delivery")

# Create environment with rendering
env = DroneDeliveryEnv(render_mode="human", max_episode_steps=1000)

# Run episodes
num_episodes = 5
episode_rewards = []

for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False

    while not done:
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated

    episode_rewards.append(episode_reward)
    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}")

print(f"\nMean reward: {np.mean(episode_rewards):.2f} ± {np.std(episode_rewards):.2f}")

env.close()
