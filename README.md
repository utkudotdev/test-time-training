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
  - 1M timesteps, 8 parallel envs (SubprocVecEnv)
  - Saves best + checkpoints to `models/`
  - Tensorboard logs to `logs/`

- **eval.py** — Evaluate trained agent (headless)
  - Runs 5 episodes
  - Shows final reward statistics

- **visualize_mujoco.py** — Visualize trained agent in MuJoCo viewer
  - Run with `uv run mjpython visualize_mujoco.py`
  - Native MuJoCo rendering (full fidelity)
  - Press space to pause, R to reset

- **main.py** — Interactive viewer (not for training)
  - Run with `uv run mjpython main.py`
  - Shows wind field visualization:
    - Keybinds for toggling wind conditions: calm **[1]**, cold_front **[2]**, squall **[3]**, thermal **[4]**, jet stream **[5]**, no wind **[6]** 
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

Runs 5 evaluation episodes with the trained policy (deterministic actions).

## Environment Details

**Task:** Navigate drone to deliver box at target goal position (5.0, 0, 2.0)

**Dynamics:**

- Drone: 6-DOF free joint (quaternion-based)
- Box: 6-DOF free joint, connected to drone via tendon
- 4 thrust motors with wind field

**Reward:**

- `+5.0 * progress` — progress toward goal (delta distance)
- `+1/(1+dist)` — dense proximity bonus
- `+10` — delivery bonus (box within 0.5m of goal)
- `-0.1 * tilt` — keep upright penalty
- `-0.001 * (ctrl - hover)^2` — effort penalty

**Termination:**

- Crash (drone/box Z < threshold)
- Successful delivery (box near goal)
- Max episode steps (1000)

## Notes

- On macOS, main.py requires `mjpython`: `uv run mjpython main.py`
- Training uses headless mode (no rendering) for speed
- Adjust hyperparameters in `train.py` for tuning
