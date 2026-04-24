# Drone Package Delivery — RL with Test-Time Training

Train a PPO policy to fly a Skydio X2 drone (with a 1kg box attached by a tendon)
to a goal position in MuJoCo. Built for test-time-training research: the same
policy can be evaluated under unseen wind patterns to study adaptation.

## Setup

```bash
uv sync
```

## Architecture

A **cascaded controller** separates navigation from stabilization so the RL
policy doesn't have to learn low-level motor mixing.

```text
RL policy  →  [thrust_Δ, roll, pitch, yaw]   (high-level, 4 values in [-1, 1])
                        ↓
PD controller →  4 motor thrusts              (see controller.py)
                        ↓
             drone + box physics  ←  wind field
```

This factorization is ideal for TTT: the PD controller is wind-invariant and
stays fixed at test time; only the navigation policy adapts.

## Files

### Environments

- **[env.py](env.py)** — `DroneDeliveryEnv` (drone + box delivery, the main task)
  - Obs (27D): altitude, quat, body-frame velocities, gyro, accel, box-relative pose, goal in body frame, last action
  - Action (4D): cascaded `[thrust_Δ, roll, pitch, yaw]` in `[-1, 1]`
  - Goal: `[10, 0, 2]` with optional spherical obstacles
  - Calibrated `HOVER_THRUST = 5.702` for drone+suspended-box system

- **[env_drone.py](env_drone.py)** — `DroneEnv` (drone only, no box)
  - Simpler test env for verifying the control stack
  - Obs (21D), same 4D action
  - Uses `x2_only.xml` (drone without box/tendon)
  - `HOVER_THRUST = 3.2495625`

- **[controller.py](controller.py)** — Cascaded PD attitude controller
  - `cascaded_control(quat, gyro, action, hover_thrust)` → 4 motor commands
  - Angle control on roll/pitch/yaw, rate damping via gyro
  - Tunable gains: `KP_ATT`, `KD_RATE`, `KP_YAW`, `KD_YAW`

- **[wind_sim.py](wind_sim.py)** — Wind patterns for weather domain shift
  - `wind_none`, `wind_calm`, `wind_cold_front`, `wind_squall`, `wind_thermal`, `wind_jet_stream`
  - Selected in env via `wind_type="calm"`

- **[x2_only.xml](x2_only.xml)** / **[x2.xml](x2.xml)** / **[example.xml](example.xml)** — MuJoCo models (drone, drone+box, scene)

### Training & eval

- **[train.py](train.py)** — Train PPO on delivery task
  - 8 parallel envs, ~1M timesteps, tensorboard logging
  - Saves `models/best_model.zip` and periodic checkpoints

- **[train_drone.py](train_drone.py)** — Train PPO on drone-only navigation
  - Use this first to verify the control stack learns before tackling delivery

- **[eval.py](eval.py)** — Headless evaluation over 5 episodes

- **[visualize_mujoco.py](visualize_mujoco.py)** — Watch the delivery policy
  - Configurable `WITH_WIND` / `WIND_TYPE` at top of file
  - Keys: `SPACE` pause, `R` reset, `W` toggle wind lines

- **[visualize_drone.py](visualize_drone.py)** — Watch the drone-only policy

- **[main.py](main.py)** — Interactive scene with manual wind-type keybinds
  - Keys: **1**=calm, **2**=cold_front, **3**=squall, **4**=thermal, **5**=jet_stream, **6**=none, **SPACE**=pause

## Workflow

### 1. Verify control stack (drone only, no box)

```bash
uv run python train_drone.py            # ~20 min, saves to models_drone/
uv run mjpython visualize_drone.py      # watch it fly to the goal
```

### 2. Train delivery (drone + box)

```bash
uv run python train.py                  # saves to models/
uv run tensorboard --logdir logs/       # monitor rollout/ep_rew_mean
```

### 3. Visualize / evaluate

```bash
uv run mjpython visualize_mujoco.py     # MuJoCo viewer with trained policy
uv run python eval.py                   # headless, 5 episodes, prints stats
```

### 4. Test-time training experiments

Train under one wind condition, evaluate under another:

```python
train_env = DroneDeliveryEnv(wind_type="calm")      # training distribution
test_env  = DroneDeliveryEnv(wind_type="thermal")   # unseen at test time
```

The PD controller (motor-level stabilization) stays the same across weathers;
only the high-level navigation policy needs to adapt.

## Reward (delivery env)

| Term | Weight | Purpose |
| --- | --- | --- |
| `10 · Δdist_to_goal` | — | dense progress signal |
| Survival bonus | +0.1 | avoid crash-seeking |
| `2 · exp(-d/3)` | — | exponential proximity |
| Tilt penalty | -10 | safety net on PD |
| Stillness at goal | -0.5 · (1.5 - d) · \|v\| | stop at target |
| Tiered d-bonuses | +2 (d<0.5), +5 (d<0.3) | shaped signal near target |
| Box delivered | +10 (d<0.5), +25 (d<0.2) | the actual task |
| Terminal: crashed / OOB | -100 | strong deterrent |
| Terminal: delivered | +100 | strong completion bonus |

## macOS note

`mujoco.viewer.launch_passive` requires `mjpython` on macOS. Use `uv run mjpython <file>` for any script that opens a viewer.
