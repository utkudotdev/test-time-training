# Drone Package Delivery RL Training

Train a PPO agent to autonomously deliver a package using a Skydio X2 drone in MuJoCo.

## Setup

Create uv environment and install dependencies:

```bash
uv sync
```

## Files

- **env.py** — Gymnasium wrapper for the delivery task
  - Observation: drone state, box state, sensors, goal position (39D)
  - Action: 4 thrust motor commands [0, 13]
  - Reward: distance to goal + delivery bonus - control penalty

- **train.py** — Train PPO agent from scratch
  - 100k timesteps training
  - Saves model to `models/ppo_delivery.zip`

- **eval.py** — Evaluate trained agent
  - Renders 5 episodes
  - Shows final reward statistics

- **main.py** — Interactive viewer (not for training)
  - Run with `uv run mjpython main.py`
  - Shows wind field visualization
  - Press space to pause/resume

## Training

```bash
uv run python train.py
```

This will:

1. Create a DummyVecEnv wrapper
2. Train PPO on 100k timesteps
3. Save to `models/ppo_delivery.zip`

## Evaluation

```bash
uv run python eval.py
```

Renders 5 evaluation episodes with the trained policy (deterministic actions).

## Environment Details

**Task:** Navigate drone to deliver box at target goal position (5.0, 0, 2.0)

**Dynamics:**

- Drone: 6-DOF free joint (quaternion-based)
- Box: 6-DOF free joint, connected to drone via tendon
- 4 thrust motors with wind field

**Reward:**

- `-0.1 * distance_to_goal` — incentivize proximity
- `+1.0` — bonus when box within 0.15m of goal
- `-0.001 * sum(control^2)` — fuel cost

**Termination:**

- Crash (drone/box Z < threshold)
- Successful delivery (box near goal)
- Max episode steps (1000)

## Notes

- On macOS, main.py requires `mjpython`: `uv run mjpython main.py`
- Training uses headless mode (no rendering) for speed
- Adjust hyperparameters in `train.py` for tuning
