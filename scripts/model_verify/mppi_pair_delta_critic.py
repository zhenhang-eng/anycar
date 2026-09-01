#!/usr/bin/env python3
"""Reusable absolute-action value Critic with an antisymmetric pair-delta head."""

from __future__ import annotations

import torch
from torch import nn

from car_foundation.mppi_proposal_policy import TemporalConvEncoder
from run_mppi_absolute_action_value_critic_cv import FLAT_SENSITIVITY, KNOT_TIMES


def knot_geometry(reference: torch.Tensor) -> torch.Tensor:
    samples = reference[:, KNOT_TIMES, :]
    return torch.stack((
        samples[..., 0], samples[..., 1],
        torch.atan2(samples[..., 2], samples[..., 3]), samples[..., 4],
    ), dim=-1)


class ConfigurableAbsoluteActionValueCritic(nn.Module):
    """Current scalar-value topology plus an optional antisymmetric delta head."""

    def __init__(self, wide: bool = False, pair_delta: bool = False) -> None:
        super().__init__()
        if wide:
            temporal_dim, current_dim = 200, 104
            state_dim, token_dim, feedforward_dim = 304, 96, 192
            value_hidden, value_tail = 304, 104
        else:
            temporal_dim, current_dim = 128, 64
            state_dim, token_dim, feedforward_dim = 192, 64, 128
            value_hidden, value_tail = 192, 64
        self.config = {
            "wide": bool(wide), "pair_delta": bool(pair_delta),
            "temporal_dim": temporal_dim, "current_dim": current_dim,
            "state_dim": state_dim, "token_dim": token_dim,
            "feedforward_dim": feedforward_dim, "transformer_layers": 2,
        }
        self.pair_delta_enabled = bool(pair_delta)
        self.history_encoder = TemporalConvEncoder(7, 8, temporal_dim, 0.0)
        self.reference_encoder = TemporalConvEncoder(5, 5, temporal_dim, 0.0)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, current_dim), nn.SiLU(),
            nn.Linear(current_dim, current_dim), nn.SiLU(),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(2 * temporal_dim + current_dim, state_dim), nn.SiLU(),
            nn.Linear(state_dim, state_dim), nn.SiLU(),
        )
        self.state_projection = nn.Linear(state_dim, token_dim)
        self.token_projection = nn.Linear(9, token_dim)
        layer = nn.TransformerEncoderLayer(
            d_model=token_dim, nhead=4, dim_feedforward=feedforward_dim,
            dropout=0.0, batch_first=True, activation="gelu",
        )
        self.action_encoder = nn.TransformerEncoder(layer, num_layers=2)
        self.value_head = nn.Sequential(
            nn.Linear(token_dim + state_dim + 16, value_hidden), nn.SiLU(),
            nn.Linear(value_hidden, value_tail), nn.SiLU(),
            nn.Linear(value_tail, 1),
        )
        if self.pair_delta_enabled:
            pair_input = state_dim + 2 * token_dim + 16
            self.pair_head = nn.Sequential(
                nn.Linear(pair_input, value_hidden), nn.SiLU(),
                nn.Linear(value_hidden, value_tail), nn.SiLU(),
                nn.Linear(value_tail, 1),
            )
        times = torch.tensor(KNOT_TIMES, dtype=torch.float32) / 49.0
        self.register_buffer("time_encoding", torch.stack((times, 1.0 - times), -1))
        sensitivity = torch.tensor(FLAT_SENSITIVITY).reshape(8, 2).sum(1)
        self.register_buffer("sensitivity", sensitivity / sensitivity.max())

    def encode_state(self, history, reference, current):
        return self.state_fusion(torch.cat((
            self.history_encoder(history), self.reference_encoder(reference),
            self.current_encoder(current),
        ), dim=-1))

    def encode_actions(self, state, reference, actions):
        batch, candidates = actions.shape[:2]
        geometry = knot_geometry(reference)
        times = self.time_encoding[None].expand(batch, -1, -1)
        sensitivity = self.sensitivity[None, :, None].expand(batch, -1, -1)
        static = torch.cat((
            actions,
            times[:, None].expand(-1, candidates, -1, -1),
            sensitivity[:, None].expand(-1, candidates, -1, -1),
            geometry[:, None].expand(-1, candidates, -1, -1),
        ), dim=-1)
        token = self.token_projection(static.reshape(batch * candidates, 8, 9))
        state_token = self.state_projection(state)[:, None, None, :].expand(
            -1, candidates, 8, -1
        ).reshape(batch * candidates, 8, -1)
        return self.action_encoder(token + state_token).mean(1).reshape(
            batch, candidates, -1
        )

    def values_from_encoded(self, state, encoded, actions):
        batch, candidates = actions.shape[:2]
        state_expanded = state[:, None].expand(-1, candidates, -1)
        return self.value_head(torch.cat((
            encoded, state_expanded, actions.reshape(batch, candidates, 16)
        ), dim=-1))[..., 0]

    def forward(self, history, reference, current, actions):
        state = self.encode_state(history, reference, current)
        encoded = self.encode_actions(state, reference, actions)
        return self.values_from_encoded(state, encoded, actions)

    def _raw_pair(self, state, left_encoded, right_encoded, left, right):
        return self.pair_head(torch.cat((
            state, left_encoded + right_encoded, left_encoded - right_encoded,
            (left - right).reshape(len(left), 16),
        ), dim=-1))[:, 0]

    def pair_delta(self, history, reference, current, left, right):
        if not self.pair_delta_enabled:
            raise RuntimeError("pair_delta head is not enabled")
        if left.shape != right.shape or left.ndim != 3:
            raise ValueError("left/right must both be [B,8,2]")
        state = self.encode_state(history, reference, current)
        encoded = self.encode_actions(
            state, reference, torch.stack((left, right), dim=1)
        )
        forward = self._raw_pair(
            state, encoded[:, 0], encoded[:, 1], left, right
        )
        reverse = self._raw_pair(
            state, encoded[:, 1], encoded[:, 0], right, left
        )
        return 0.5 * (forward - reverse)


def model_for_arm(arm: str) -> ConfigurableAbsoluteActionValueCritic:
    if arm == "base":
        return ConfigurableAbsoluteActionValueCritic()
    if arm == "wide":
        return ConfigurableAbsoluteActionValueCritic(wide=True)
    if arm == "pair_delta":
        return ConfigurableAbsoluteActionValueCritic(pair_delta=True)
    raise ValueError(arm)
