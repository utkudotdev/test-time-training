"""WindAwarePolicy: ActorCriticPolicy with dynamics-prediction auxiliary head.

Architecture (use_wind_context=True, 33D obs):
  obs(33D) → features_extractor → mlp_extractor → pi_latent ──→ action_net → π(a|s)
                                                             └──→ value_net  → V(s)
             (pi_latent ⊕ action) → aux_head → Δ[lin_vel(3), accel(3)]

The rolling 6D wind context (obs[27:33]) is computed by the env each step as an
EMA of recent Δ[lin_vel, accel]. This gives the action_net an explicit wind signal
without requiring gradient-based TTT — adaptation happens through the rolling buffer.
The aux_head still trains the encoder to be wind-aware (useful for TTT on top).

Obs indices (first 27D; same positions in 33D obs):
  [0]     altitude
  [1:5]   quat
  [5:8]   lin_vel_body   ← aux target component 1
  [8:11]  gyro
  [11:14] body_linacc    ← aux target component 2
  [14:17] box_rel_pos
  [17:20] box_rel_vel
  [20:23] goal_vec
  [23:27] last_action
  [27:33] wind_context   ← EMA of [lin_vel(3), accel(3)] level (use_wind_context only)
                           tracks persistent wind-distorted state; delta-EMA would
                           converge to zero under constant wind (useless).
"""

import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy

VEL_IDX   = slice(5, 8)    # lin_vel_body within obs
ACCEL_IDX = slice(11, 14)  # body_linacc within obs
AUX_DIM   = 6              # Δlin_vel(3) + Δaccel(3)


class WindAwarePolicy(ActorCriticPolicy):
    """PPO policy with a 6D auxiliary dynamics-prediction head."""

    def __init__(self, *args, aux_hidden: int = 64, **kwargs):
        super().__init__(*args, **kwargs)
        pi_dim  = self.mlp_extractor.latent_dim_pi   # 256 with net_arch=[256,256]
        act_dim = self.action_space.shape[0]           # 4
        self.aux_head = nn.Sequential(
            nn.Linear(pi_dim + act_dim, aux_hidden),
            nn.ReLU(),
            nn.Linear(aux_hidden, AUX_DIM),
        )

    def predict_aux(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Predict Δ[lin_vel(3), accel(3)] from (obs, action).

        Gradients flow through mlp_extractor AND aux_head so both can be
        jointly optimised by the aux optimizer.
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        pi_latent, _ = self.mlp_extractor(features)
        return self.aux_head(th.cat([pi_latent, actions], dim=-1))

    def aux_features(self, obs: th.Tensor, actions: th.Tensor) -> th.Tensor:
        """Return Φ(s,a) — penultimate aux features (aux_hidden-dim).

        Used by the Neural-Fly linear adapter: d̂ = Φ(s,a) · c,
        where Φ is frozen at test time and only c is updated via least squares.
        """
        features = self.extract_features(obs, self.pi_features_extractor)
        pi_latent, _ = self.mlp_extractor(features)
        x = th.cat([pi_latent, actions], dim=-1)
        return th.relu(self.aux_head[0](x))   # (N, aux_hidden)
