"""Visualize trained PPO agent with viser 3D viewer."""

from stable_baselines3 import PPO
from env import DroneDeliveryEnv
import viser
import numpy as np
import time

# Load trained model
model = PPO.load("models/ppo_delivery")
env = DroneDeliveryEnv(max_episode_steps=1000)

# Create viser server
server = viser.ViserServer(port=8080)
print("Open http://localhost:8080 in your browser")

# Ground grid (reference)
server.scene.add_grid(
    "ground",
    width=10.0,
    height=10.0,
    width_segments=10,
    height_segments=10,
    position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# World axes for orientation reference
server.scene.add_frame(
    "world",
    show_axes=True,
    axes_length=1.0,
    axes_radius=0.02,
    position=np.array([0.0, 0.0, 0.0], dtype=np.float32),
)

# Goal (bright green, large enough to see)
goal_marker = server.scene.add_icosphere(
    "goal_sphere",
    radius=0.3,
    color=(0, 255, 0),
)
goal_marker.position = np.array([5.0, 0.0, 2.0], dtype=np.float32)

# Drone (blue)
drone_marker = server.scene.add_icosphere(
    "drone_sphere",
    radius=0.25,
    color=(0, 100, 255),
)
drone_marker.position = np.array([0.0, 0.0, 0.3], dtype=np.float32)

# Box (red)
box_marker = server.scene.add_icosphere(
    "box_sphere",
    radius=0.2,
    color=(255, 50, 50),
)
box_marker.position = np.array([0.0, 0.0, 0.1], dtype=np.float32)

# Label for live status
status_label = server.gui.add_text("Status", initial_value="Waiting to start...")

print("Waiting 3 seconds for browser to connect...")
time.sleep(3)

# Run episodes
num_episodes = 3
for episode in range(num_episodes):
    obs, info = env.reset()
    episode_reward = 0
    done = False
    step = 0

    while not done:
        # Observation layout: drone_qpos(7), drone_qvel(6), box_qpos(7), box_qvel(6), gyro(3), accel(3), quat(4), goal(3)
        drone_pos = np.array(obs[:3], dtype=np.float32)
        box_pos = np.array(obs[13:16], dtype=np.float32)

        # Update visualizations
        drone_marker.position = drone_pos
        box_marker.position = box_pos

        # Predict and step
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        episode_reward += reward
        done = terminated or truncated
        step += 1

        # Update status every 20 steps
        if step % 20 == 0:
            dist = np.linalg.norm(drone_pos - np.array([5.0, 0.0, 2.0]))
            status_label.value = (
                f"Ep {episode + 1} | Step {step} | "
                f"Drone: ({drone_pos[0]:.2f}, {drone_pos[1]:.2f}, {drone_pos[2]:.2f}) | "
                f"Dist to goal: {dist:.2f}"
            )

        # Slow down so you can see it (real-time at ~30 FPS)
        time.sleep(0.03)

    print(f"Episode {episode + 1}: Reward = {episode_reward:.2f}, Steps = {step}")
    time.sleep(1)  # Pause between episodes

status_label.value = "Visualization complete."
print("Visualization complete. Keep browser open to view.")
while True:
    time.sleep(1)
