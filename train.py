"""Train PPO agent for package delivery task.

This trains for 500k timesteps with improved hyperparameters.
Takes ~20-30 minutes depending on hardware.
"""

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from env import DroneDeliveryEnv
import os

# Create environment
env = DroneDeliveryEnv()
env = DummyVecEnv([lambda: env])

# Create PPO agent
model = PPO(
    "MlpPolicy",
    env,
    learning_rate=5e-4,
    n_steps=2048,
    batch_size=128,
    n_epochs=20,
    gamma=0.99,
    gae_lambda=0.95,
    clip_range=0.2,
    verbose=1,
)

# Train
total_timesteps = 500000
print(f"Training for {total_timesteps} timesteps...")
model.learn(total_timesteps=total_timesteps)

# Save model
os.makedirs("models", exist_ok=True)
model.save("models/ppo_delivery")
print("Model saved to models/ppo_delivery.zip")

env.close()
