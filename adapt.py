"""Test-time adaptation for drone under unseen wind.

Comparison columns:
  Oracle     — models_full/best_model trained on all winds + obstacles (upper bound)
  Zero-shot  — models_ttt/best_model, calm-only training, no adaptation
  Adaptive   — models_ttt/best_model + gradient adapt with early stopping

The gap Oracle vs Zero-shot shows the OOD penalty.
The gap Zero-shot vs Adaptive shows how much TTT recovers.

Run:
  uv run python adapt.py
  uv run python adapt.py --model models_ttt/20260509_131210/best_model
"""

import copy
import argparse
import os
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO

from env import DroneDeliveryEnv
from policy_ttt import VEL_IDX, ACCEL_IDX

DEFAULT_TTT_PATH    = "models_ttt/best_model"
DEFAULT_ORACLE_PATH = "models_full/best_model"

TEST_CONDITIONS = [
    dict(wind_type="calm",       wind_speed=1.0, wind_turbulence=0.3, label="calm (in-dist)"),
    dict(wind_type="thermal",    wind_speed=1.0, wind_turbulence=0.3, label="thermal (OOD)"),
    dict(wind_type="jet_stream", wind_speed=1.0, wind_turbulence=0.3, label="jet_stream (OOD)"),
    dict(wind_type="squall",     wind_speed=1.0, wind_turbulence=0.3, label="squall (OOD)"),
    dict(wind_type="cold_front", wind_speed=1.0, wind_turbulence=0.3, label="cold_front (OOD)"),
    dict(wind_type="cyclone",    wind_speed=1.0, wind_turbulence=0.3, label="cyclone (OOD)"),
]

WITH_OBSTACLES   = True   # match training environment
N_ADAPT_EPISODES = 10
N_EVAL_EPISODES  = 5
GRAD_STEPS_MAX   = 50
GRAD_LR          = 5e-4
GRAD_CLIP        = 0.5
PROX_LAMBDA      = 1.0   # raised from 0.5 — stronger proximal anchor prevents encoder drift
EARLY_STOP_DELTA = 1e-3
# Set above in-dist (calm) aux_loss so calm is never adapted.
# Current calm aux_loss ≈ 0.26 → threshold 0.31 skips calm, adapts squall (0.34+).
MIN_AUX_LOSS     = 0.31


# ---------------------------------------------------------------------------
# Rollout / eval helpers
# ---------------------------------------------------------------------------

def make_env(wind_type, wind_speed, wind_turbulence):
    return DroneDeliveryEnv(
        max_episode_steps=1000,
        with_wind=True,
        wind_type=wind_type,
        wind_speed=wind_speed,
        wind_turbulence=wind_turbulence,
        with_obstacles=WITH_OBSTACLES,
    )


def collect_episodes(model, env, n_episodes: int):
    rollouts = []
    for _ in range(n_episodes):
        obs_seq, act_seq = [], []
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            next_obs, _, terminated, truncated, _ = env.step(action)
            obs_seq.append(obs.copy())
            act_seq.append(action.copy())
            obs = next_obs
            done = terminated or truncated
        if len(obs_seq) > 1:
            obs_arr = np.array(obs_seq)
            act_arr = np.array(act_seq)
            rollouts.append((obs_arr[:-1], act_arr[:-1], obs_arr[1:]))
    return rollouts


def evaluate(model, env, n_episodes: int) -> float:
    rewards = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r = 0.0
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, _ = env.step(action)
            ep_r += r
            done = terminated or truncated
        rewards.append(ep_r)
    return float(np.mean(rewards))


def rollouts_to_tensors(rollouts):
    obs_t  = th.tensor(np.concatenate([r[0] for r in rollouts]), dtype=th.float32)
    acts_t = th.tensor(np.concatenate([r[1] for r in rollouts]), dtype=th.float32)
    obs_t1 = th.tensor(np.concatenate([r[2] for r in rollouts]), dtype=th.float32)
    target = th.cat([
        obs_t1[:, VEL_IDX]   - obs_t[:, VEL_IDX],
        obs_t1[:, ACCEL_IDX] - obs_t[:, ACCEL_IDX],
    ], dim=-1)
    return obs_t, acts_t, target


# ---------------------------------------------------------------------------
# Adaptation
# ---------------------------------------------------------------------------

def adaptive_adapt(model, rollouts):
    """Gradient adapt with early stopping. Skips if aux loss < MIN_AUX_LOSS.

    Returns (adapted_model, steps_taken, init_aux_loss, skipped).
    """
    policy = model.policy
    enc_params = list(policy.mlp_extractor.parameters())
    aux_params = list(policy.aux_head.parameters())
    params = enc_params + aux_params
    obs_t, acts_t, target = rollouts_to_tensors(rollouts)
    anchor = [p.data.clone() for p in enc_params]

    with th.no_grad():
        init_loss = F.mse_loss(policy.predict_aux(obs_t, acts_t), target).item()

    if init_loss < MIN_AUX_LOSS:
        return model, 0, init_loss, True

    opt = th.optim.Adam(params, lr=GRAD_LR)
    prev_loss = float("inf")
    steps_taken = 0
    policy.set_training_mode(True)
    for _ in range(GRAD_STEPS_MAX):
        pred = policy.predict_aux(obs_t, acts_t)
        aux_loss = F.mse_loss(pred, target)
        prox_loss = sum((p - a).pow(2).sum() for p, a in zip(enc_params, anchor))
        loss = aux_loss + PROX_LAMBDA * prox_loss

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        steps_taken += 1

        if prev_loss - aux_loss.item() < EARLY_STOP_DELTA:
            break
        prev_loss = aux_loss.item()
    policy.set_training_mode(False)
    return model, steps_taken, init_loss, False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model",  default=DEFAULT_TTT_PATH,
                        help="TTT model path (without .zip).")
    parser.add_argument("--oracle", default=DEFAULT_ORACLE_PATH,
                        help="Oracle model path (without .zip). Pass 'none' to skip.")
    args = parser.parse_args()

    print(f"TTT model : {args.model}.zip")
    base_model = PPO.load(args.model, device="cpu")
    if not hasattr(base_model.policy, "aux_head"):
        raise RuntimeError("No aux_head found — train with train_ttt.py first.")

    oracle_model = None
    if args.oracle.lower() != "none" and os.path.exists(args.oracle + ".zip"):
        print(f"Oracle    : {args.oracle}.zip")
        oracle_model = PPO.load(args.oracle, device="cpu")
    else:
        print("Oracle    : not found — skipping oracle column")

    print(f"\nConfig: adapt_eps={N_ADAPT_EPISODES}  eval_eps={N_EVAL_EPISODES}  "
          f"obstacles={WITH_OBSTACLES}  min_aux={MIN_AUX_LOSS}  "
          f"max_steps={GRAD_STEPS_MAX}  lr={GRAD_LR}  prox={PROX_LAMBDA}")

    oracle_hdr = f"{'Oracle':>10}  " if oracle_model else ""
    header = (
        f"\n{'Condition':<22}  {'AuxLoss':>7}  {'EpLen':>5}  "
        f"{oracle_hdr}{'Zero-shot':>10}  {'Adaptive':>10}  {'Steps':>6}"
    )
    print(header)
    print("-" * len(header))

    for cond in TEST_CONDITIONS:
        label = cond.pop("label")
        env = make_env(**cond)

        # Oracle (zero-shot, trained on all winds)
        oracle_str = ""
        if oracle_model:
            oracle_r = evaluate(oracle_model, env, N_EVAL_EPISODES)
            oracle_str = f"{oracle_r:>10.1f}  "

        # TTT zero-shot
        base = copy.deepcopy(base_model)
        zs_r = evaluate(base, env, N_EVAL_EPISODES)

        # Collect rollouts for adaptation
        rollouts = collect_episodes(copy.deepcopy(base_model), env, N_ADAPT_EPISODES)
        mean_ep_len = float(np.mean([len(r[0]) for r in rollouts]))

        # TTT adaptive
        ada_model = copy.deepcopy(base_model)
        ada_model, steps, init_loss, skipped = adaptive_adapt(ada_model, rollouts)
        ada_r = evaluate(ada_model, env, N_EVAL_EPISODES)

        steps_str = "skip" if skipped else str(steps)
        print(
            f"{label:<22}  {init_loss:>7.3f}  {mean_ep_len:>5.0f}  "
            f"{oracle_str}{zs_r:>10.1f}  {ada_r:>10.1f}  {steps_str:>6}"
        )

        env.close()
        cond["label"] = label


if __name__ == "__main__":
    main()
