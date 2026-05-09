"""Compare all policy variants across all wind conditions.

Baselines evaluated:
  1. no-wind pretrained  (models/best_model)       — transfer lower-bound
  2. wind-curriculum     (models_full/best_model)  — domain-randomised PPO
  3. TTT zero-shot       (models_ttt/best_model)   — aux head, no adaptation
  4. TTT fast            (30 grad steps)
  5. TTT adaptive        (early stopping, up to 50 steps)

Usage:
  uv run python evaluate.py
  uv run python evaluate.py --ttt models_ttt/20260507_143022/best_model
  uv run python evaluate.py --n-eval 10 --n-adapt 20
  uv run python evaluate.py --no-ttt          # skip TTT variants
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

# ── Test conditions ────────────────────────────────────────────────────────────
TEST_CONDITIONS = [
    dict(wind_type="calm",       wind_speed=1.0, wind_turbulence=0.3, label="calm (in-dist)"),
    dict(wind_type="thermal",    wind_speed=1.0, wind_turbulence=0.3, label="thermal (OOD)"),
    dict(wind_type="jet_stream", wind_speed=1.0, wind_turbulence=0.3, label="jet_stream (OOD)"),
    dict(wind_type="squall",     wind_speed=1.0, wind_turbulence=0.3, label="squall (OOD)"),
    dict(wind_type="cold_front", wind_speed=1.0, wind_turbulence=0.3, label="cold_front (OOD)"),
    dict(wind_type="cyclone",    wind_speed=1.0, wind_turbulence=0.3, label="cyclone (OOD)"),
]

# TTT adaptation hyperparams — mirror adapt.py
GRAD_STEPS_FAST  = 30
GRAD_STEPS_FULL  = 50
GRAD_LR          = 5e-4
GRAD_CLIP        = 0.5
PROX_LAMBDA      = 0.5
EARLY_STOP_DELTA = 1e-3
MIN_AUX_LOSS     = 0.1


# ── Evaluation helpers ─────────────────────────────────────────────────────────

def evaluate_policy(model, env, n_episodes):
    """Returns (mean_r, std_r, delivery_rate, obs_hit_rate, crash_rate, mean_ep_len)."""
    rewards, delivered, crashes, lengths = [], [], [], []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_r, steps, reason = 0.0, 0, ""
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, terminated, truncated, info = env.step(action)
            ep_r += r
            steps += 1
            if terminated:
                reason = info.get("termination", "")
            done = terminated or truncated
        rewards.append(ep_r)
        delivered.append(1.0 if reason == "delivered" else 0.0)
        crashes.append(  1.0 if reason in ("hit_obstacle", "crashed") else 0.0)
        lengths.append(steps)
    return (
        float(np.mean(rewards)),
        float(np.std(rewards)),
        float(np.mean(delivered)),
        float(np.mean(crashes)),
        float(np.mean(lengths)),
    )


def collect_rollouts(model, env, n_episodes):
    """Collect (obs_t, act_t, obs_{t+1}) tuples for TTT adaptation."""
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
            arr = np.array(obs_seq)
            act = np.array(act_seq)
            rollouts.append((arr[:-1], act[:-1], arr[1:]))
    return rollouts


def _to_tensors(rollouts):
    obs_t  = th.tensor(np.concatenate([r[0] for r in rollouts]), dtype=th.float32)
    acts_t = th.tensor(np.concatenate([r[1] for r in rollouts]), dtype=th.float32)
    obs_t1 = th.tensor(np.concatenate([r[2] for r in rollouts]), dtype=th.float32)
    target = th.cat([
        obs_t1[:, VEL_IDX]   - obs_t[:, VEL_IDX],
        obs_t1[:, ACCEL_IDX] - obs_t[:, ACCEL_IDX],
    ], dim=-1)
    return obs_t, acts_t, target


def compute_aux_loss(model, rollouts) -> float:
    obs_t, acts_t, target = _to_tensors(rollouts)
    with th.no_grad():
        pred = model.policy.predict_aux(obs_t, acts_t)
        return F.mse_loss(pred, target).item()


# ── TTT adaptation ─────────────────────────────────────────────────────────────

def _adapt(model, rollouts, n_steps, early_stop=False):
    """Gradient-based encoder + aux_head adaptation with proximal penalty.

    Returns steps_taken.
    """
    policy = model.policy
    enc = list(policy.mlp_extractor.parameters())
    aux = list(policy.aux_head.parameters())
    opt = th.optim.Adam(enc + aux, lr=GRAD_LR)
    obs_t, acts_t, target = _to_tensors(rollouts)
    anchor = [p.data.clone() for p in enc]

    policy.set_training_mode(True)
    prev_loss = float("inf")
    steps_done = 0
    for _ in range(n_steps):
        pred = policy.predict_aux(obs_t, acts_t)
        aux_loss = F.mse_loss(pred, target)
        prox = sum((p - a).pow(2).sum() for p, a in zip(enc, anchor))
        loss = aux_loss + PROX_LAMBDA * prox
        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(enc + aux, GRAD_CLIP)
        opt.step()
        steps_done += 1
        if early_stop:
            cur = F.mse_loss(policy.predict_aux(obs_t, acts_t), target).detach().item()
            if prev_loss - cur < EARLY_STOP_DELTA:
                break
            prev_loss = cur
    policy.set_training_mode(False)
    return steps_done


def ttt_fast(model, rollouts):
    """30 fixed gradient steps; skips if aux loss already low."""
    init_loss = compute_aux_loss(model, rollouts)
    if init_loss < MIN_AUX_LOSS:
        return model, 0, init_loss
    steps = _adapt(model, rollouts, GRAD_STEPS_FAST, early_stop=False)
    return model, steps, init_loss


def ttt_adaptive(model, rollouts):
    """Up to 50 gradient steps with early stopping; skips if aux loss low."""
    init_loss = compute_aux_loss(model, rollouts)
    if init_loss < MIN_AUX_LOSS:
        return model, 0, init_loss
    steps = _adapt(model, rollouts, GRAD_STEPS_FULL, early_stop=True)
    return model, steps, init_loss


# ── Model loading ──────────────────────────────────────────────────────────────

def try_load(path, label):
    if not path:
        return None
    if not os.path.exists(path + ".zip"):
        print(f"  [skip] {label}: {path}.zip not found")
        return None
    print(f"  Loaded  {label}: {path}.zip")
    return PPO.load(path, device="cpu")


# ── Formatting ─────────────────────────────────────────────────────────────────

W_COND = 22
W_CELL = 22   # "RRRRR(d:DD%|o:OO%|c:CC%)"

def _cell(mean_r, del_rate, crash_rate):
    return f"{mean_r:>6.0f}(d:{del_rate*100:>2.0f}%|x:{crash_rate*100:>2.0f}%)"


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Evaluate all policy variants on all wind conditions")
    parser.add_argument("--baseline", default="models/best_model",
                        help="No-wind pretrained model (no .zip)")
    parser.add_argument("--full",     default="models_full/best_model",
                        help="Wind-curriculum model (no .zip)")
    parser.add_argument("--ttt",      default="models_ttt/best_model",
                        help="TTT model with aux_head (no .zip)")
    parser.add_argument("--n-eval",   type=int, default=5,
                        help="Episodes per evaluation (default: 5)")
    parser.add_argument("--n-adapt",  type=int, default=10,
                        help="Rollout episodes for TTT adaptation data (default: 10)")
    parser.add_argument("--no-ttt",   action="store_true",
                        help="Skip TTT variants (faster run)")
    args = parser.parse_args()

    print("\nLoading models...")
    m_base = try_load(args.baseline, "no-wind")
    m_full = try_load(args.full,     "full-wind")
    m_ttt  = None if args.no_ttt else try_load(args.ttt, "TTT")

    active_cols = (
        (["no-wind"]                          if m_base else []) +
        (["full-wind"]                        if m_full else []) +
        (["ttt-zs", "ttt-fast", "ttt-ada"]   if m_ttt  else [])
    )

    if not active_cols:
        print("No models found — nothing to evaluate.")
        return

    sep = "-" * (W_COND + len(active_cols) * (W_CELL + 2) + (17 if m_ttt else 0))

    # ── Table header ──────────────────────────────────────────────────────────
    print(f"\nMetrics: reward(d:del%|o:obs_hit%|c:crash%)   obstacles=True  n_eval={args.n_eval}  n_adapt={args.n_adapt}\n")
    print(f"{'Condition':<{W_COND}}", end="")
    for col in active_cols:
        print(f"  {col:^{W_CELL}}", end="")
    if m_ttt:
        print(f"  {'aux_loss':>8}  {'steps':>5}", end="")
    print()
    print(sep)

    # ── Per-condition evaluation ───────────────────────────────────────────────
    for cond in TEST_CONDITIONS:
        label = cond["label"]
        kw = {k: v for k, v in cond.items() if k != "label"}
        env = DroneDeliveryEnv(
            max_episode_steps=1000,
            with_wind=True,
            with_obstacles=True,
            wind_type=str(kw["wind_type"]),
            wind_speed=float(kw["wind_speed"]),
            wind_turbulence=float(kw["wind_turbulence"]),
        )

        cells = {}

        if m_base:
            r, _, d, c, _ = evaluate_policy(copy.deepcopy(m_base), env, args.n_eval)
            cells["no-wind"] = _cell(r, d, c)

        if m_full:
            r, _, d, c, _ = evaluate_policy(copy.deepcopy(m_full), env, args.n_eval)
            cells["full-wind"] = _cell(r, d, c)

        aux_str = step_str = ""
        if m_ttt:
            rollouts = collect_rollouts(copy.deepcopy(m_ttt), env, args.n_adapt)
            aux_loss = compute_aux_loss(copy.deepcopy(m_ttt), rollouts)

            r, _, d, c, _ = evaluate_policy(copy.deepcopy(m_ttt), env, args.n_eval)
            cells["ttt-zs"] = _cell(r, d, c)

            m_f = copy.deepcopy(m_ttt)
            m_f, _, _ = ttt_fast(m_f, rollouts)
            r, _, d, c, _ = evaluate_policy(m_f, env, args.n_eval)
            cells["ttt-fast"] = _cell(r, d, c)

            m_a = copy.deepcopy(m_ttt)
            m_a, ada_steps, _ = ttt_adaptive(m_a, rollouts)
            r, _, d, c, _ = evaluate_policy(m_a, env, args.n_eval)
            cells["ttt-ada"] = _cell(r, d, c)

            skipped = aux_loss < MIN_AUX_LOSS
            aux_str  = f"{aux_loss:>8.3f}"
            step_str = f"{'skip':>5}" if skipped else f"{ada_steps:>5}"

        print(f"{label:<{W_COND}}", end="")
        for col in active_cols:
            print(f"  {cells.get(col, 'N/A'):^{W_CELL}}", end="")
        if m_ttt:
            print(f"  {aux_str}  {step_str}", end="")
        print()

        env.close()

    print(sep)
    print(f"\nDelivery: box<0.5m  |  o=hit_obstacle  |  c=ground_crash  |  aux_skip<{MIN_AUX_LOSS}\n")


if __name__ == "__main__":
    main()
