"""Test-time adaptation for drone under unseen wind.

Why the previous version showed identical results for all modes:
  The aux_head is a SIDE BRANCH — the policy never routes through it during
  inference (action = action_net(mlp_extractor(obs)), no aux_head involved).
  Adapting only the aux_head has exactly zero effect on the policy's actions.

Fixed approach:
  gradient  — update the FULL shared encoder (mlp_extractor) + aux_head via Adam
              on the aux loss. Encoder change → pi_latent change → action change.
              200 steps, lr=1e-3, gradient clipping to prevent forgetting.
  fast      — same but only 30 steps. Speed vs quality tradeoff.

Note on Neural-Fly linear adapter:
  A true closed-form linear adapter (d̂ = Φ·c, update c via least squares)
  requires the policy to RECEIVE d̂ as part of its observation so it can act
  on the wind estimate. That needs architecture change: retrain train_ttt.py
  with obs_space extended to [obs(27D); d̂(6D)] = 33D. Without that, the
  aux_head is unreachable from the policy's action path.

Run:
  uv run python adapt.py
"""

import copy
import numpy as np
import torch as th
import torch.nn.functional as F
from stable_baselines3 import PPO

from env import DroneDeliveryEnv
from policy_ttt import WindAwarePolicy, VEL_IDX, ACCEL_IDX

MODEL_PATH = "models_ttt/best_model"

TEST_CONDITIONS = [
    dict(wind_type="calm", wind_speed=1.0, wind_turbulence=0.3, label="calm (in-dist)"),
    dict(
        wind_type="thermal", wind_speed=1.0, wind_turbulence=0.3, label="thermal (OOD)"
    ),
    dict(
        wind_type="jet_stream",
        wind_speed=1.0,
        wind_turbulence=0.3,
        label="jet_stream (OOD)",
    ),
    dict(wind_type="squall", wind_speed=1.0, wind_turbulence=0.3, label="squall (OOD)"),
]

N_ADAPT_EPISODES = 10
N_EVAL_EPISODES = 5
GRAD_STEPS_FULL = 200  # thorough adaptation
GRAD_STEPS_FAST = 30  # fast adaptation (fewer compute budget)
GRAD_LR = 5e-4
GRAD_CLIP = 0.5  # max norm
PROX_LAMBDA = 0.5  # L2 penalty toward pretrained weights (prevents forgetting)
EARLY_STOP_DELTA = 5e-4  # stop when aux loss improvement per step drops below this
MIN_AUX_LOSS = 0.35  # skip adaptation entirely if initial aux loss is below this
# (wind signal too weak → gradient steps fit noise, hurts policy)


# ---------------------------------------------------------------------------
# Rollout helpers
# ---------------------------------------------------------------------------


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
    obs_t = th.tensor(np.concatenate([r[0] for r in rollouts]), dtype=th.float32)
    acts_t = th.tensor(np.concatenate([r[1] for r in rollouts]), dtype=th.float32)
    obs_t1 = th.tensor(np.concatenate([r[2] for r in rollouts]), dtype=th.float32)
    target = th.cat(
        [
            obs_t1[:, VEL_IDX] - obs_t[:, VEL_IDX],
            obs_t1[:, ACCEL_IDX] - obs_t[:, ACCEL_IDX],
        ],
        dim=-1,
    )
    return obs_t, acts_t, target


# ---------------------------------------------------------------------------
# Adaptation modes
# ---------------------------------------------------------------------------


def gradient_adapt(
    model,
    rollouts,
    n_steps: int = GRAD_STEPS_FULL,
    lr: float = GRAD_LR,
    prox_lambda: float = PROX_LAMBDA,
):
    """Adapt the shared encoder + aux_head via gradient descent on aux loss.

    Uses a proximal (L2) penalty toward the pretrained encoder weights to prevent
    catastrophic forgetting of the base navigation policy:
        loss = MSE(aux_pred, dynamics_target) + prox_lambda * ||W - W_0||²

    The penalty keeps the encoder from drifting too far from what it learned
    during domain-randomised training, while still allowing wind-specific shifts.
    """
    policy = model.policy
    enc_params = list(policy.mlp_extractor.parameters())
    aux_params = list(policy.aux_head.parameters())
    params = enc_params + aux_params
    opt = th.optim.Adam(params, lr=lr)
    obs_t, acts_t, target = rollouts_to_tensors(rollouts)

    # Snapshot original encoder weights as the proximal anchor
    anchor = [p.data.clone() for p in enc_params]

    policy.set_training_mode(True)
    for _ in range(n_steps):
        pred = policy.predict_aux(obs_t, acts_t)
        aux_loss = F.mse_loss(pred, target)

        # Proximal penalty: pulls encoder back toward pretrained weights
        prox_loss = sum((p - a).pow(2).sum() for p, a in zip(enc_params, anchor))
        loss = aux_loss + prox_lambda * prox_loss

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
    policy.set_training_mode(False)
    return model


def fast_adapt(model, rollouts):
    """Same as gradient_adapt but only 30 steps — for speed-vs-quality comparison."""
    return gradient_adapt(model, rollouts, n_steps=GRAD_STEPS_FAST, lr=GRAD_LR)


def adaptive_adapt(
    model,
    rollouts,
    max_steps: int = GRAD_STEPS_FULL,
    lr: float = GRAD_LR,
    prox_lambda: float = PROX_LAMBDA,
    delta: float = EARLY_STOP_DELTA,
):
    """Gradient adapt with early stopping when aux loss improvement saturates.

    Automatically calibrates step count to the difficulty of the wind:
      - In-dist (aux loss already low): converges in ~10-20 steps → small encoder shift
      - Hard OOD (aux loss high): runs up to max_steps → large encoder shift needed
    """
    policy = model.policy
    enc_params = list(policy.mlp_extractor.parameters())
    aux_params = list(policy.aux_head.parameters())
    params = enc_params + aux_params
    opt = th.optim.Adam(params, lr=lr)
    obs_t, acts_t, target = rollouts_to_tensors(rollouts)
    anchor = [p.data.clone() for p in enc_params]

    # Measure initial aux loss — if wind signal is too weak, skip adaptation
    with th.no_grad():
        init_pred = policy.predict_aux(obs_t, acts_t)
        init_loss = F.mse_loss(init_pred, target).item()
    if init_loss < MIN_AUX_LOSS:
        return model, 0  # wind already predictable; gradient steps would fit noise

    prev_loss = float("inf")
    steps_taken = 0
    policy.set_training_mode(True)
    for _ in range(max_steps):
        pred = policy.predict_aux(obs_t, acts_t)
        aux_loss = F.mse_loss(pred, target)
        prox_loss = sum((p - a).pow(2).sum() for p, a in zip(enc_params, anchor))
        loss = aux_loss + prox_lambda * prox_loss

        opt.zero_grad()
        loss.backward()
        th.nn.utils.clip_grad_norm_(params, GRAD_CLIP)
        opt.step()
        steps_taken += 1

        if prev_loss - aux_loss.item() < delta:
            break
        prev_loss = aux_loss.item()
    policy.set_training_mode(False)
    return model, steps_taken


# ---------------------------------------------------------------------------
# Main evaluation loop
# ---------------------------------------------------------------------------


def main():
    print(f"Loading {MODEL_PATH}.zip")
    base_model = PPO.load(MODEL_PATH, device="cpu")

    if not hasattr(base_model.policy, "aux_head"):
        raise RuntimeError("No aux_head found — train with train_ttt.py first.")

    print(f"\nConfig:  adapt_eps={N_ADAPT_EPISODES}  eval_eps={N_EVAL_EPISODES}")
    print(
        f"         grad_steps_full={GRAD_STEPS_FULL}  fast={GRAD_STEPS_FAST}  lr={GRAD_LR}  prox_lambda={PROX_LAMBDA}"
    )

    header = f"\n{'Condition':<22}  {'Zero-shot':>10}  {'Fast(30)':>10}  {'Adaptive':>10}  {'Steps':>6}"
    print(header)
    print("-" * len(header))

    for cond in TEST_CONDITIONS:
        label = cond.pop("label")
        env = DroneDeliveryEnv(max_episode_steps=1000, with_wind=True, **cond)

        base = copy.deepcopy(base_model)
        zs_r = evaluate(base, env, N_EVAL_EPISODES)
        rollouts = collect_episodes(base, env, N_ADAPT_EPISODES)

        fast_model = copy.deepcopy(base_model)
        fast_adapt(fast_model, rollouts)
        fast_r = evaluate(fast_model, env, N_EVAL_EPISODES)

        ada_model = copy.deepcopy(base_model)
        ada_model, steps = adaptive_adapt(ada_model, rollouts)
        ada_r = evaluate(ada_model, env, N_EVAL_EPISODES)

        print(
            f"{label:<22}  {zs_r:>10.1f}  {fast_r:>10.1f}  {ada_r:>10.1f}  {steps:>6}"
        )

        env.close()
        cond["label"] = label


if __name__ == "__main__":
    main()

