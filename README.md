# Drone Package Delivery — RL + Test-Time Training

PPO policy for a Skydio X2 drone (with a 1 kg box on a tendon) flying to a goal in MuJoCo.
The main research question: can a policy trained on calm wind adapt to unseen wind conditions at test time, using only self-supervised dynamics prediction — no reward signal?

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

### TTT policy architecture (train_ttt.py)

```text
obs(33D) = raw_obs(27D) ++ wind_context(6D)
                ↓
         shared encoder (MLP 256×256)
         ↙                    ↘
  action_net → π(a|s)     aux_head → Δ[lin_vel, accel]  (training only)
```

`wind_context` is a 6D rolling EMA of recent Δ[lin_vel, accel] maintained by the environment. It updates every step with no gradient required — the policy observes wind disturbance directly. The aux head still trains the encoder to be wind-aware, and gradient-based TTT remains available on top for strongly OOD conditions.

## Files

| File | Purpose |
| --- | --- |
| `env.py` | Drone + box delivery env (27D obs, 4D action) |
| `env_drone.py` | Drone-only env for sanity checking |
| `controller.py` | Cascaded PD attitude controller |
| `wind_sim.py` | Wind patterns: `calm`, `cold_front`, `squall`, `thermal`, `jet_stream`, `cyclone`, `none` |
| `train.py` | PPO baseline — no wind, no obstacles |
| `train_obs.py` | Fine-tune on obstacles (no wind); output is shared warm-start |
| `train_full.py` | PPO with wind curriculum, warm-started from `models_obs` |
| `train_ttt.py` | PPO + auxiliary dynamics-prediction head (Phase 3) |
| `adapt.py` | Test-time adaptation: fast, gradient, and adaptive modes |
| `eval.py` | Headless evaluation |
| `visualize_mujoco.py` | MuJoCo viewer — set `MODEL_PATH` and `WIND_TYPE` at top |
| `visualize_cfg.py` | Configurable MuJoCo viewer — wind type, obstacles, model via CLI args |
| `visualize_drone.py` | Viewer for drone-only policy |

## Workflow

### 1. Train baseline (no wind, no obstacles)

```bash
uv run python train.py
uv run tensorboard --logdir logs/ --host 127.0.0.1
```

Saves `models/best_model.zip`. Learns basic navigation (~8500 reward/episode).

### 2. Add obstacle avoidance

```bash
uv run python train_obs.py
uv run tensorboard --logdir logs_obs/
```

Warm-starts from `models/best_model.zip`, trains 200 k steps with obstacles and no wind.
Saves `models_obs/best_model.zip` — shared warm-start for all downstream scripts.

### 3. Train with wind curriculum

```bash
uv run python train_full.py
uv run tensorboard --logdir logs_full/
```

Warm-starts from `models_obs/best_model.zip`, then adds wind progressively:

| Steps | Wind |
| --- | --- |
| 0–200 k | calm, speed=0.3 |
| 200–500 k | calm, speed=0.6 |
| 500–800 k | calm, speed=1.0 |
| 800 k–1.3 M | domain-randomised (calm / cold\_front / squall / cyclone) |

Each run saves to a timestamped subdirectory `models_full/YYYYMMDD_HHMMSS/`. The best checkpoint is also copied to `models_full/best_model.zip` for easy access.

### 4. Train TTT policy (aux head)

```bash
uv run python train_ttt.py
uv run tensorboard --logdir logs_ttt/
```

Adds a dynamics-prediction auxiliary head to the policy:

```text
shared encoder → action head  (PPO)
              ↘ aux head → predicted Δ[lin_vel, accel]  (6D)
```

The aux head is trained jointly with PPO on a **separate optimizer** so it doesn't interfere with the policy gradient. The encoder learns wind-aware features as a side effect.

Each run saves to a timestamped subdirectory `models_ttt/YYYYMMDD_HHMMSS/`. The best checkpoint is also copied to `models_ttt/best_model.zip`.

Warm-starts from `models_obs/best_model.zip` (CALM_ONLY=True) or `models_full/best_model.zip` (CALM_ONLY=False).

#### Curriculum modes

Set `CALM_ONLY` at the top of `train_ttt.py`:

| `CALM_ONLY` | Pretrain | Curriculum | Use when |
| --- | --- | --- | --- |
| `True` (default) | `models_obs/best_model` | calm only (gentle → full strength) | rely entirely on TTT to handle all other winds |
| `False` | `models_full/best_model` | full domain-randomised (calm / cold\_front / squall / cyclone) | bake more wind resistance into the policy |

### 5. Test-time adaptation

```bash
uv run python adapt.py
# or point at a specific timestamped run:
uv run python adapt.py --model models_ttt/20260507_143022/best_model
```

Three modes, all **self-supervised** (no reward signal used):

| Mode | What it does |
| --- | --- |
| `fast` | 30 gradient steps on encoder + aux head; skips if aux loss < threshold |
| `adaptive` | gradient steps with early stopping (Δloss < 1e-3) + skip if aux loss < threshold |
| `gradient` | fixed 200 steps (reference) |

Results on `models_ttt/best_model` (calm-only training):

| Condition | AuxLoss | EpLen | Zero-shot | Adaptive | Steps |
| --- | --- | --- | --- | --- | --- |
| calm (in-dist) | 0.228 | 999 | 11391 | 11391 | skip |
| thermal (OOD) | 0.253 | 999 | 4176 | 4176 | skip |
| jet\_stream (OOD) | 0.503 | 999 | **11775** | 11775 | skip |
| squall (OOD) | 0.666 | 999 | 2674 | ~8100 | ~32 |
| cold\_front (OOD) | 0.701 | 229 | 238 | ~6000 | ~30–50 |
| cyclone (OOD) | 0.780 | 561 | −521 | −521 | skip\* |

\* Cyclone is a rotational vortex — wind direction depends on drone position, so the aux head sees contradictory Δvel signals from different parts of the field and cannot converge to a useful encoder shift. Needs cyclone in training (set `CALM_ONLY = False`) to handle properly.

The aux loss is a **wind-novelty detector**. The threshold `MIN_AUX_LOSS = 0.55` separates two regimes:

| AuxLoss range | Meaning | Action |
| --- | --- | --- |
| < 0.55 | Wind is familiar or already well-handled zero-shot | Skip — gradient steps fit noise |
| ≥ 0.55 | Wind is genuinely OOD and the encoder can improve | Adapt |

Key: high aux loss alone doesn't mean the policy is struggling (jet\_stream=0.503 with zero-shot 11775). The policy can navigate a dynamically novel wind without needing the encoder to shift. Only adapt when the aux loss is high *and* the task is failing zero-shot.

### 6. Visualize

```bash
# Configurable viewer (recommended) — choose wind, obstacles, model via flags
uv run mjpython visualize_cfg.py --wind-type cyclone --obstacles
uv run mjpython visualize_cfg.py --model models_ttt/best_model --wind-type squall --obstacles
uv run mjpython visualize_cfg.py --no-wind --obstacles

# Legacy viewer — edit MODEL_PATH / WIND_TYPE constants at top of file
uv run mjpython visualize_mujoco.py

# Drone-only policy
uv run mjpython visualize_drone.py
```

`visualize_cfg.py` flags: `--model`, `--wind-type`, `--wind-speed`, `--wind-turbulence`, `--obstacles`, `--no-wind`, `--wind-lines`. Defaults to `models_full/best_model` when `--obstacles` is set, else `models/best_model`.

Keys: `SPACE` pause · `R` reset · `W` toggle wind lines · `1–7` switch wind type live

## Wind types

| Key | Name | Force at goal | Notes |
| --- | --- | --- | --- |
| 1 | calm | ~0.6 N | rotational, weakest |
| 2 | cold\_front | ~2 N | directional frontal |
| 3 | squall | ~3.2 N | moving front |
| 4 | thermal | ~0.2 N | localized near origin, near-zero at distance |
| 5 | jet\_stream | ~5.6 N | narrow directional burst, strongest |
| 6 | cyclone | — | rotational spiral with updraft ring |
| 7 | none | 0 N | — |

Wind force scale is `2×` the raw wind function output (calibrated so speed=1.0 stays within the drone's 8.9 N lateral authority). All three force components (x, y, z) are applied — vertical wind matters for cyclone and thermal.

## Reward (delivery env)

| Term | Value |
| --- | --- |
| Progress toward goal | `10 × Δdist` |
| Survival bonus | +0.1/step |
| Exponential proximity | `2 × exp(−d/3)` |
| Tilt penalty | `−10 × tilt` |
| Angular velocity damp | `−0.05 × ‖ω‖` |
| Stillness near goal | `−0.5 × (1.5−d) × ‖v‖` if d < 1.5 |
| Tiered bonuses | +2 (d<0.5), +5 (d<0.3) |
| Delivery bonus | +25 (box<0.5 m) |
| Hit obstacle | −500 (terminal) |
| Crash / out-of-bounds | −100 (terminal) |
| Delivered | +100 (terminal) |

### Tuning tips

| Goal | What to change |
| --- | --- |
| Policy too aggressive, crashes in wind | Lower progress coef `10×` → `5×` |
| Policy hovers instead of navigating | Raise progress coef, lower survival bonus |
| Too much tilting in strong wind | Raise tilt penalty `10×` → `15×` |
| Overshoots goal | Raise stillness coef `0.5×` → `1.0×` |
| Drone still flies through obstacles | Raise obstacle penalty −500 → −1000 |
| Delivery never triggered | Check box\_to\_goal threshold (0.5 m) |

## Model directory layout

```
models/
  best_model.zip          ← baseline (no wind)
models_full/
  best_model.zip          ← latest best (canonical, overwritten each run)
  20260507_143022/        ← per-run checkpoints + best_model
  20260507_160500/
models_ttt/
  best_model.zip          ← latest best (canonical, overwritten each run)
  20260507_143022/
  20260507_160500/
```

## macOS note

`mujoco.viewer` requires `mjpython` on macOS. Use `uv run mjpython <file>` for any script that opens a viewer window.
