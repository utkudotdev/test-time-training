# Drone Package Delivery — RL + Test-Time Training

PPO policy for a Skydio X2 drone (with a 1 kg box on a tendon) flying to a goal in MuJoCo.
The main research question: can a policy trained on some wind conditions adapt to unseen ones at test time, using only self-supervised dynamics prediction — no reward signal?

## Setup

```bash
uv sync
```

## Architecture

```text
RL policy → [thrust_Δ, roll, pitch, yaw]  (4D, cascaded)
                      ↓
            PD attitude controller → 4 motor thrusts
                      ↓
            drone + box physics ← wind field
```

The PD controller handles low-level stabilization and stays frozen. Only the navigation policy adapts.

## Files

| File | Purpose |
| --- | --- |
| `env.py` | Drone + box delivery env (27D obs, 4D action) |
| `env_drone.py` | Drone-only env for sanity checking |
| `controller.py` | Cascaded PD attitude controller |
| `wind_sim.py` | Wind patterns: `calm`, `cold_front`, `squall`, `thermal`, `jet_stream`, `none` |
| `train.py` | PPO baseline — no wind, confirms delivery works |
| `train_full.py` | PPO with wind curriculum, warm-started from baseline |
| `train_ttt.py` | PPO + auxiliary dynamics-prediction head (Phase 2) |
| `adapt.py` | Test-time adaptation: fast, gradient, and adaptive modes |
| `eval.py` | Headless evaluation |
| `visualize_mujoco.py` | MuJoCo viewer — set `MODEL_PATH` and `WIND_TYPE` at top |
| `visualize_drone.py` | Viewer for drone-only policy |

## Workflow

### 1. Train baseline (no wind)

```bash
uv run python train.py
uv run tensorboard --logdir logs/ --host 127.0.0.1
```

Saves `models/best_model.zip`. Reaches ~8500 reward/episode.

### 2. Train with wind curriculum

```bash
uv run python train_full.py
```

Warm-starts from `models/best_model.zip`, then adds wind progressively:

| Steps | Wind |
| --- | --- |
| 0–200 k | calm, speed=0.3 |
| 200–500 k | calm, speed=0.6 |
| 500–800 k | calm, speed=1.0 |
| 800 k–1.5 M | domain-randomised (calm / cold\_front / squall) |

Saves to `models_full/`.

### 3. Train TTT policy (aux head)

```bash
uv run python train_ttt.py
```

Adds a dynamics-prediction auxiliary head to the policy:

```text
shared encoder → action head  (PPO)
              ↘ aux head → predicted Δ[lin_vel, accel]  (6D)
```

The aux head is trained jointly with PPO on a **separate optimizer** so it doesn't interfere with the policy gradient. The encoder learns wind-aware features as a side effect.

Saves to `models_ttt/`.

### 4. Test-time adaptation

```bash
uv run python adapt.py
```

Three modes, all **self-supervised** (no reward signal used):

| Mode | What it does |
| --- | --- |
| `fast` | 30 gradient steps on encoder + aux head |
| `adaptive` | gradient steps with early stopping + skip if aux loss < 0.35 |
| `gradient` | fixed 200 steps (reference) |

Results on `models_ttt/best_model`:

| Condition | Zero-shot | Adaptive | Steps |
| --- | --- | --- | --- |
| calm (in-dist) | 11821 | 11821 | 0 (skipped) |
| thermal (OOD) | 9047 | 9047 | 0 (skipped) |
| squall (OOD) | 1847 | 2122 | 42 |
| **jet\_stream (OOD)** | **289** | **4794** | **77** |

The aux loss doubles as a **wind-novelty detector**: if it's below 0.35, the wind is already familiar and adaptation would overfit. Above 0.35, the encoder shifts toward the new wind pattern.

### 5. Visualize

```bash
# Delivery policy
uv run mjpython visualize_mujoco.py

# Drone-only policy
uv run mjpython visualize_drone.py
```

Change `MODEL_PATH` at the top of the visualizer to switch between `models/`, `models_full/`, or `models_ttt/`.

Keys: `SPACE` pause · `R` reset · `W` toggle wind lines · `1–6` switch wind type (main.py only)

## Wind types

| Key | Name | Force at goal | Notes |
| --- | --- | --- | --- |
| 1 | calm | ~0.6 N | rotational, weakest |
| 2 | cold\_front | ~2 N | directional frontal |
| 3 | squall | ~3.2 N | moving front |
| 4 | thermal | ~0.2 N | localized near origin, near-zero at distance |
| 5 | jet\_stream | ~5.6 N | narrow directional burst, strongest |
| 6 | none | 0 N | — |

Wind force scale is `2×` the raw wind function output (calibrated so speed=1.0 stays within the drone's 8.9 N lateral authority).

## Reward (delivery env)

| Term | Value |
| --- | --- |
| Progress toward goal | `10 × Δdist` |
| Survival bonus | +0.1/step |
| Exponential proximity | `2 × exp(−d/3)` |
| Tilt penalty | `−10 × tilt` |
| Stillness near goal | `−0.5 × (1.5−d) × ‖v‖` if d < 1.5 |
| Tiered bonuses | +2 (d<0.5), +5 (d<0.3) |
| Delivery bonus | +10 (box<0.5 m), +25 (box<0.2 m) |
| Crash / out-of-bounds | −100 (terminal) |
| Delivered | +100 (terminal) |

## macOS note

`mujoco.viewer` requires `mjpython` on macOS. Use `uv run mjpython <file>` for any script that opens a viewer window.
