"""Lightweight PyTorch policy for bounded MPPI sampling-center residuals."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class MPPIProposalNormalization:
    """Train-split statistics used by both training and deployment."""

    history_mean: np.ndarray
    history_std: np.ndarray
    reference_mean: np.ndarray
    reference_std: np.ndarray
    current_mean: np.ndarray
    current_std: np.ndarray

    @classmethod
    def fit(
        cls, history: np.ndarray, reference: np.ndarray, current: np.ndarray
    ) -> "MPPIProposalNormalization":
        def statistics(value: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
            axes = tuple(range(value.ndim - 1))
            mean = value.mean(axis=axes).astype(np.float32)
            std = value.std(axis=axes).astype(np.float32)
            return mean, np.maximum(std, 1e-4).astype(np.float32)

        history_mean, history_std = statistics(history)
        reference_mean, reference_std = statistics(reference)
        current_mean, current_std = statistics(current)
        return cls(
            history_mean,
            history_std,
            reference_mean,
            reference_std,
            current_mean,
            current_std,
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            name: np.asarray(getattr(self, name), dtype=np.float32).tolist()
            for name in self.__dataclass_fields__
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "MPPIProposalNormalization":
        return cls(
            **{
                name: np.asarray(value[name], dtype=np.float32)
                for name in cls.__dataclass_fields__
            }
        )

    def normalize_numpy(
        self, history: np.ndarray, reference: np.ndarray, current: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        return (
            (history - self.history_mean) / self.history_std,
            (reference - self.reference_mean) / self.reference_std,
            (current - self.current_mean) / self.current_std,
        )


def ego_reference_features(
    reference_ego: np.ndarray, current_vx: float, horizon: int = 50
) -> np.ndarray:
    """Convert stored ego reference `[51,4]` to policy features `[50,5]`."""
    reference = np.asarray(reference_ego, dtype=np.float32)
    if reference.shape == (horizon + 1, 4):
        reference = reference[1:]
    if reference.shape != (horizon, 4):
        raise ValueError(
            f"reference_ego must have shape [{horizon},4] or [{horizon + 1},4]"
        )
    return np.stack(
        (
            reference[:, 0],
            reference[:, 1],
            np.sin(reference[:, 2]),
            np.cos(reference[:, 2]),
            reference[:, 3] - float(current_vx),
        ),
        axis=-1,
    ).astype(np.float32)


class TemporalConvEncoder(nn.Module):
    """Small fixed-shape temporal encoder designed for ONNX export."""

    def __init__(
        self,
        input_channels: int,
        pooled_steps: int,
        output_dim: int = 128,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.convolution = nn.Sequential(
            nn.Conv1d(input_channels, 32, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.SiLU(),
            nn.Conv1d(64, 96, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool1d(pooled_steps),
        )
        self.projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(96 * pooled_steps, output_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
        )

    def forward(self, sequence: torch.Tensor) -> torch.Tensor:
        return self.projection(self.convolution(sequence.transpose(1, 2)))


class TorchMPPIProposalPolicy(nn.Module):
    """Predict eight bounded acceleration/steering residual knots."""

    history_length = 250
    reference_length = 50
    knot_count = 8
    action_dim = 2

    def __init__(
        self,
        trust_scale: tuple[float, float] = (0.25, 0.35),
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU()
        )
        self.warm_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.output = nn.Linear(256, self.knot_count * self.action_dim)
        nn.init.zeros_(self.output.weight)
        nn.init.zeros_(self.output.bias)
        self.register_buffer(
            "trust_scale",
            torch.tensor(trust_scale, dtype=torch.float32).reshape(1, 1, 2),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        warm_knots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(warm_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("warm_knots must have shape [B,8,2]")
        features = torch.cat(
            (
                self.history_encoder(history),
                self.reference_encoder(reference),
                self.current_encoder(current),
                self.warm_encoder(warm_knots.flatten(1)),
            ),
            dim=1,
        )
        delta = torch.tanh(self.output(self.fusion(features))).reshape(
            -1, self.knot_count, self.action_dim
        )
        delta = delta * self.trust_scale
        center = torch.clamp(warm_knots + delta, -1.0, 1.0)
        effective_delta = center - warm_knots
        return effective_delta, center

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIProposalCritic(nn.Module):
    """Predict warm-relative proposal quality for an arbitrary sampling center.

    The scalar output is a normalized advantage: positive means that the proposed
    center is expected to have lower MPPI weighted-output cost than the warm
    center.  Keeping warm at the natural zero target makes the critic less
    sensitive to the large absolute-cost shift across speeds and recovery states.
    """

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        # These module names and shapes intentionally match the actor so the
        # state encoders can be initialized from a BC checkpoint.
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU()
        )
        self.warm_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        # Candidate features include both the absolute center and its standardized
        # displacement from warm.  The latter is the actor's natural action space.
        self.center_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim * 2, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 128),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 1),
        )
        self.register_buffer(
            "base_sigma",
            torch.tensor((0.25, 0.35), dtype=torch.float32).reshape(1, 1, 2),
        )

    def encode_state(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        warm_knots: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        expected_knots = (self.knot_count, self.action_dim)
        if tuple(warm_knots.shape[1:]) != expected_knots:
            raise ValueError("warm_knots must have shape [B,8,2]")
        return torch.cat(
            (
                self.history_encoder(history),
                self.reference_encoder(reference),
                self.current_encoder(current),
                self.warm_encoder(warm_knots.flatten(1)),
            ),
            dim=1,
        )

    def encode_center(
        self, warm_knots: torch.Tensor, center_knots: torch.Tensor
    ) -> torch.Tensor:
        expected_knots = (self.knot_count, self.action_dim)
        if tuple(warm_knots.shape[1:]) != expected_knots:
            raise ValueError("warm_knots must have shape [B,8,2]")
        if tuple(center_knots.shape[1:]) != expected_knots:
            raise ValueError("center_knots must have shape [B,8,2]")
        if len(warm_knots) != len(center_knots):
            raise ValueError("warm_knots and center_knots batch sizes differ")
        standardized_delta = (center_knots - warm_knots) / self.base_sigma
        center_features = torch.cat(
            (center_knots.flatten(1), standardized_delta.flatten(1)), dim=1
        )
        return self.center_encoder(center_features)

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        warm_knots: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.cat(
            (
                self.encode_state(history, reference, current, warm_knots),
                self.encode_center(warm_knots, center_knots),
            ),
            dim=1,
        )
        return self.fusion(features).squeeze(-1)

    def forward_center_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        warm_knots: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        """Score a per-state center bank while encoding state context once."""
        expected_knots = (self.knot_count, self.action_dim)
        if tuple(center_knots.shape[2:]) != expected_knots:
            raise ValueError("center_knots must have shape [B,K,8,2]")
        batch, center_count = center_knots.shape[:2]
        if len(history) != batch or len(warm_knots) != batch:
            raise ValueError("state and center bank batch sizes differ")
        state_features = self.encode_state(history, reference, current, warm_knots)
        expanded_warm = warm_knots[:, None].expand(-1, center_count, -1, -1)
        center_features = self.encode_center(
            expanded_warm.reshape(batch * center_count, *expected_knots),
            center_knots.reshape(batch * center_count, *expected_knots),
        ).reshape(batch, center_count, -1)
        fused = torch.cat(
            (
                state_features[:, None].expand(-1, center_count, -1),
                center_features,
            ),
            dim=2,
        )
        return self.fusion(fused.reshape(batch * center_count, -1)).reshape(
            batch, center_count
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPILocalQuadraticCritic(nn.Module):
    """Locally structured center critic with an explicitly supervised gradient.

    The fourth input is the local anchor center rather than the controller's raw
    warm start.  The output is anchor-relative normalized advantage, so it is
    exactly zero at the anchor.  Its gradient in standardized center coordinates
    is predicted directly from state context instead of emerging implicitly from
    an unconstrained value MLP.
    """

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.warm_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 64),
            nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gradient_head = nn.Linear(256, self.knot_count * self.action_dim)
        self.curvature_head = nn.Linear(256, 1)
        nn.init.zeros_(self.gradient_head.weight)
        nn.init.zeros_(self.gradient_head.bias)
        nn.init.zeros_(self.curvature_head.weight)
        nn.init.zeros_(self.curvature_head.bias)
        self.register_buffer(
            "base_sigma",
            torch.tensor((0.25, 0.35), dtype=torch.float32).reshape(1, 1, 2),
        )

    def encode_state(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        expected_knots = (self.knot_count, self.action_dim)
        if tuple(anchor_knots.shape[1:]) != expected_knots:
            raise ValueError("anchor_knots must have shape [B,8,2]")
        return self.fusion(
            torch.cat(
                (
                    self.history_encoder(history),
                    self.reference_encoder(reference),
                    self.current_encoder(current),
                    self.warm_encoder(anchor_knots.flatten(1)),
                ),
                dim=1,
            )
        )

    def local_parameters(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode_state(history, reference, current, anchor_knots)
        gradient = self.gradient_head(features).reshape(
            -1, self.knot_count, self.action_dim
        )
        curvature = self.curvature_head(features).squeeze(-1)
        return gradient, curvature

    @staticmethod
    def _quadratic_advantage(
        standardized_delta: torch.Tensor,
        gradient: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        linear = torch.sum(gradient * standardized_delta, dim=(-2, -1))
        radius_squared = torch.sum(
            standardized_delta.square(), dim=(-2, -1)
        )
        return linear + 0.5 * curvature * radius_squared

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(center_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("center_knots must have shape [B,8,2]")
        gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots
        )
        standardized_delta = (center_knots - anchor_knots) / self.base_sigma
        return self._quadratic_advantage(
            standardized_delta, gradient, curvature
        )

    def forward_center_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(center_knots.shape[2:]) != (self.knot_count, self.action_dim):
            raise ValueError("center_knots must have shape [B,K,8,2]")
        if len(center_knots) != len(anchor_knots):
            raise ValueError("state and center bank batch sizes differ")
        gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots
        )
        standardized_delta = (
            center_knots - anchor_knots[:, None]
        ) / self.base_sigma[:, None]
        return self._quadratic_advantage(
            standardized_delta,
            gradient[:, None],
            curvature[:, None],
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIFeedbackQuadraticCritic(nn.Module):
    """Local Q model conditioned on observations from a first MPPI pass."""

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim

    def __init__(self, feedback_dim: int = 74, dropout: float = 0.05) -> None:
        super().__init__()
        self.feedback_dim = int(feedback_dim)
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.warm_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(self.feedback_dim, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128, 256), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(256, 256), nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.gradient_head = nn.Linear(256, self.knot_count * self.action_dim)
        self.feedback_gradient_skip = nn.Linear(
            self.feedback_dim, self.knot_count * self.action_dim
        )
        self.curvature_head = nn.Linear(256, 1)
        nn.init.zeros_(self.gradient_head.weight)
        nn.init.zeros_(self.gradient_head.bias)
        nn.init.zeros_(self.feedback_gradient_skip.weight)
        nn.init.zeros_(self.feedback_gradient_skip.bias)
        nn.init.zeros_(self.curvature_head.weight)
        nn.init.zeros_(self.curvature_head.bias)
        self.register_buffer(
            "base_sigma",
            torch.tensor((0.25, 0.35), dtype=torch.float32).reshape(1, 1, 2),
        )

    def local_parameters(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        features = self.fusion(
            torch.cat(
                (
                    self.history_encoder(history),
                    self.reference_encoder(reference),
                    self.current_encoder(current),
                    self.warm_encoder(anchor_knots.flatten(1)),
                    self.feedback_encoder(feedback),
                ),
                dim=1,
            )
        )
        return (
            (
                self.gradient_head(features) + self.feedback_gradient_skip(feedback)
            ).reshape(-1, self.knot_count, self.action_dim),
            self.curvature_head(features).squeeze(-1),
        )

    @staticmethod
    def _quadratic_advantage(
        standardized_delta: torch.Tensor,
        gradient: torch.Tensor,
        curvature: torch.Tensor,
    ) -> torch.Tensor:
        return torch.sum(gradient * standardized_delta, dim=(-2, -1)) + 0.5 * curvature * torch.sum(
            standardized_delta.square(), dim=(-2, -1)
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots, feedback
        )
        delta = (center_knots - anchor_knots) / self.base_sigma
        return self._quadratic_advantage(delta, gradient, curvature)

    def forward_center_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        center_knots: torch.Tensor,
    ) -> torch.Tensor:
        gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots, feedback
        )
        delta = (center_knots - anchor_knots[:, None]) / self.base_sigma[:, None]
        return self._quadratic_advantage(
            delta, gradient[:, None], curvature[:, None]
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIFeedbackStepRiskCritic(nn.Module):
    """Calibrate mean and lower-tail reward along a feedback-critic direction."""

    context_dim = 74 + 16 + 16

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.context_encoder = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
        )
        self.step_encoder = nn.Sequential(
            nn.Linear(2, 64), nn.SiLU(), nn.Linear(64, 64), nn.SiLU()
        )
        self.fusion = nn.Sequential(
            nn.Linear(256 + 64, 192),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(192, 128),
            nn.SiLU(),
        )
        self.mean_rate_head = nn.Linear(128, 1)
        self.tail_gap_rate_head = nn.Linear(128, 1)
        self.win_logit_head = nn.Linear(128, 1)
        self.safe_logit_head = nn.Linear(128, 1)
        nn.init.zeros_(self.mean_rate_head.weight)
        nn.init.zeros_(self.mean_rate_head.bias)

    def forward_center_bank(
        self,
        context: torch.Tensor,
        signed_radius: torch.Tensor,
        old_linear_advantage: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if context.shape[1:] != (self.context_dim,):
            raise ValueError(f"context must have shape [B,{self.context_dim}]")
        if signed_radius.shape != old_linear_advantage.shape:
            raise ValueError("step feature banks must have identical shape")
        if signed_radius.ndim != 2 or len(signed_radius) != len(context):
            raise ValueError("step feature banks must have shape [B,K]")
        batch, center_count = signed_radius.shape
        context_features = self.context_encoder(context)
        step = torch.stack((signed_radius, old_linear_advantage), dim=2)
        step_features = self.step_encoder(step.reshape(batch * center_count, 2)).reshape(
            batch, center_count, -1
        )
        fused = self.fusion(
            torch.cat(
                (
                    context_features[:, None].expand(-1, center_count, -1),
                    step_features,
                ),
                dim=2,
            ).reshape(batch * center_count, -1)
        ).reshape(batch, center_count, -1)
        absolute_radius = signed_radius.abs()
        mean = absolute_radius * self.mean_rate_head(fused).squeeze(-1)
        tail_gap = absolute_radius * torch.nn.functional.softplus(
            self.tail_gap_rate_head(fused).squeeze(-1)
        )
        lower_tail = mean - tail_gap
        win_logit = self.win_logit_head(fused).squeeze(-1)
        safe_logit = self.safe_logit_head(fused).squeeze(-1)
        return mean, lower_tail, win_logit, safe_logit

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIFeedbackDiscreteStepCritic(nn.Module):
    """Predict one-step rewards for a small, fully covered radius bank.

    Every output action is evaluated by DBM rollout in the replay dataset.  This
    deliberately avoids differentiating a continuous Q model outside its action
    support while testing whether feedback-conditioned one-step policy learning
    can outperform the search teacher.
    """

    context_dim = 74 + 16 + 16

    def __init__(self, action_count: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.encoder = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
        )
        self.output = nn.Linear(128, self.action_count)

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[1:] != (self.context_dim,):
            raise ValueError(f"context must have shape [B,{self.context_dim}]")
        return self.output(self.encoder(context))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIFeedbackDiscreteStepActor(nn.Module):
    """Categorical one-step actor over a fixed bank of positive trust radii."""

    context_dim = TorchMPPIFeedbackDiscreteStepCritic.context_dim

    def __init__(self, action_count: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.action_count = int(action_count)
        self.policy = nn.Sequential(
            nn.Linear(self.context_dim, 256),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(128, self.action_count),
        )

    def forward(self, context: torch.Tensor) -> torch.Tensor:
        if context.shape[1:] != (self.context_dim,):
            raise ValueError(f"context must have shape [B,{self.context_dim}]")
        return self.policy(context)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIFeedbackDiscreteStateNetwork(nn.Module):
    """State- and feedback-conditioned values or logits for discrete steps."""

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    feedback_dim = 74
    gradient_context_dim = 32

    def __init__(self, output_count: int = 5, dropout: float = 0.05) -> None:
        super().__init__()
        self.output_count = int(output_count)
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(self.feedback_dim, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.gradient_encoder = nn.Sequential(
            nn.Linear(self.gradient_context_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128 + 64, 256), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(256, 192), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(192, self.output_count),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(anchor_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("anchor_knots must have shape [B,8,2]")
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        if gradient_context.shape[1:] != (self.gradient_context_dim,):
            raise ValueError(
                f"gradient_context must have shape [B,{self.gradient_context_dim}]"
            )
        return self.fusion(
            torch.cat(
                (
                    self.history_encoder(history),
                    self.reference_encoder(reference),
                    self.current_encoder(current),
                    self.anchor_encoder(anchor_knots.flatten(1)),
                    self.feedback_encoder(feedback),
                    self.gradient_encoder(gradient_context),
                ),
                dim=1,
            )
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPISequentialProbeActorCritic(nn.Module):
    """Discrete SAC network for repeated probes at one physical state.

    The physical state, history, reference, guided anchor, and first-pass
    feedback remain fixed during an internal search episode.  ``probe_value``
    and ``probe_mask`` change after each real model rollout.  The three heads
    implement a categorical actor and twin critics over the same fully covered
    center bank; no continuous action gradient is used.
    """

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    feedback_dim = 74
    gradient_context_dim = 32

    def __init__(
        self,
        action_count: int = 33,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        self.action_count = int(action_count)
        if self.action_count < 2:
            raise ValueError("action_count must include anchor and a probe action")
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(self.feedback_dim, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.gradient_encoder = nn.Sequential(
            nn.Linear(self.gradient_context_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.probe_encoder = nn.Sequential(
            nn.Linear(2 * self.action_count + 1, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 128), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128 + 64 + 128, 320),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(320, 192),
            nn.SiLU(),
        )
        self.actor_head = nn.Linear(192, self.action_count)
        self.q1_head = nn.Linear(192, self.action_count)
        self.q2_head = nn.Linear(192, self.action_count)

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        probe_value: torch.Tensor,
        probe_mask: torch.Tensor,
        remaining_fraction: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(anchor_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("anchor_knots must have shape [B,8,2]")
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        if gradient_context.shape[1:] != (self.gradient_context_dim,):
            raise ValueError(
                f"gradient_context must have shape [B,{self.gradient_context_dim}]"
            )
        expected_bank = (self.action_count,)
        if probe_value.shape[1:] != expected_bank:
            raise ValueError(f"probe_value must have shape [B,{self.action_count}]")
        if probe_mask.shape[1:] != expected_bank:
            raise ValueError(f"probe_mask must have shape [B,{self.action_count}]")
        if remaining_fraction.shape[1:] != (1,):
            raise ValueError("remaining_fraction must have shape [B,1]")
        probe = torch.cat(
            (probe_value, probe_mask.to(probe_value.dtype), remaining_fraction),
            dim=1,
        )
        fused = self.fusion(
            torch.cat(
                (
                    self.history_encoder(history),
                    self.reference_encoder(reference),
                    self.current_encoder(current),
                    self.anchor_encoder(anchor_knots.flatten(1)),
                    self.feedback_encoder(feedback),
                    self.gradient_encoder(gradient_context),
                    self.probe_encoder(probe),
                ),
                dim=1,
            )
        )
        return self.actor_head(fused), self.q1_head(fused), self.q2_head(fused)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIContinuousCenterEncoder(nn.Module):
    """Encode a frozen-state MPPI probe context for continuous-center RL."""

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    feedback_dim = 74
    gradient_context_dim = 32
    output_dim = 192

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.anchor_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        self.feedback_encoder = nn.Sequential(
            nn.Linear(self.feedback_dim, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 128), nn.SiLU(),
        )
        self.gradient_encoder = nn.Sequential(
            nn.Linear(self.gradient_context_dim, 64), nn.SiLU(),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(128 + 128 + 64 + 64 + 128 + 64, 320),
            nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(320, self.output_dim), nn.SiLU(),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(anchor_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("anchor_knots must have shape [B,8,2]")
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        if gradient_context.shape[1:] != (self.gradient_context_dim,):
            raise ValueError(
                f"gradient_context must have shape [B,{self.gradient_context_dim}]"
            )
        return self.fusion(
            torch.cat(
                (
                    self.history_encoder(history),
                    self.reference_encoder(reference),
                    self.current_encoder(current),
                    self.anchor_encoder(anchor_knots.flatten(1)),
                    self.feedback_encoder(feedback),
                    self.gradient_encoder(gradient_context),
                ),
                dim=1,
            )
        )


class TorchMPPIContinuousCenterActor(nn.Module):
    """Squashed-Gaussian Actor that directly generates a 16-D center residual.

    ``normalized_action`` is bounded to [-1, 1].  One unit represents
    ``maximum_delta_sigma`` source-MPPI sigmas, so the physical center remains a
    bounded residual around the runtime anchor while no fixed center bank limits
    its direction.
    """

    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    flat_action_dim = knot_count * action_dim

    def __init__(
        self,
        maximum_delta_sigma: float = 2.0,
        dropout: float = 0.05,
        minimum_log_std: float = -5.0,
        maximum_log_std: float = 1.0,
    ) -> None:
        super().__init__()
        if maximum_delta_sigma <= 0:
            raise ValueError("maximum_delta_sigma must be positive")
        if minimum_log_std >= maximum_log_std:
            raise ValueError("minimum_log_std must be below maximum_log_std")
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.mean_head = nn.Linear(self.encoder.output_dim, self.flat_action_dim)
        self.log_std_head = nn.Linear(self.encoder.output_dim, self.flat_action_dim)
        nn.init.zeros_(self.mean_head.weight)
        nn.init.zeros_(self.mean_head.bias)
        nn.init.zeros_(self.log_std_head.weight)
        # raw=0 maps to log_std=-2 with the default bounds: enough initial
        # exploration without immediately saturating the two-sigma trust box.
        nn.init.zeros_(self.log_std_head.bias)
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.register_buffer(
            "base_sigma",
            torch.tensor((0.25, 0.35), dtype=torch.float32).reshape(1, 1, 2),
        )
        self.register_buffer(
            "maximum_delta_sigma",
            torch.tensor(float(maximum_delta_sigma), dtype=torch.float32),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        mean = self.mean_head(feature).reshape(-1, self.knot_count, self.action_dim)
        raw_log_std = self.log_std_head(feature).reshape_as(mean)
        log_std = torch.tanh(raw_log_std)
        midpoint = 0.5 * (self.maximum_log_std + self.minimum_log_std)
        half_range = 0.5 * (self.maximum_log_std - self.minimum_log_std)
        log_std = midpoint + half_range * log_std
        return mean, log_std

    def center_from_action(
        self, anchor_knots: torch.Tensor, normalized_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(normalized_action.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("normalized_action must have shape [B,8,2]")
        requested_delta = (
            normalized_action * self.maximum_delta_sigma * self.base_sigma
        )
        center = torch.clamp(anchor_knots + requested_delta, -1.0, 1.0)
        effective_action = (center - anchor_knots) / (
            self.maximum_delta_sigma * self.base_sigma
        )
        return effective_action, center

    def sample(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        deterministic: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        mean, log_std = self(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        pre_tanh = mean if deterministic else mean + log_std.exp() * torch.randn_like(mean)
        normalized_action = torch.tanh(pre_tanh)
        effective_action, center = self.center_from_action(
            anchor_knots, normalized_action
        )
        # Log probability of the squashed Gaussian.  The fixed physical action
        # scale is omitted because it adds only a policy-independent constant.
        normal_log_probability = -0.5 * (
            ((pre_tanh - mean) / log_std.exp()).square()
            + 2.0 * log_std
            + math.log(2.0 * math.pi)
        )
        correction = torch.log(1.0 - normalized_action.square() + 1e-6)
        log_probability = (normal_log_probability - correction).flatten(1).sum(1)
        return effective_action, log_probability, center

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIDeterministicCenterActor(nn.Module):
    """Deterministic 16-D center Actor with one unique output per context.

    Exploration is deliberately external to this module.  Training code may
    perturb the returned normalized action with a frozen structured design, but
    deployment and direct-cost evaluation always use exactly ``tanh(raw_action)``.
    """

    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    flat_action_dim = knot_count * action_dim

    def __init__(
        self,
        maximum_delta_sigma: float = 2.0,
        dropout: float = 0.05,
    ) -> None:
        super().__init__()
        if maximum_delta_sigma <= 0:
            raise ValueError("maximum_delta_sigma must be positive")
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.action_head = nn.Linear(self.encoder.output_dim, self.flat_action_dim)
        nn.init.zeros_(self.action_head.weight)
        nn.init.zeros_(self.action_head.bias)
        self.register_buffer(
            "base_sigma",
            torch.tensor((0.25, 0.35), dtype=torch.float32).reshape(1, 1, 2),
        )
        self.register_buffer(
            "maximum_delta_sigma",
            torch.tensor(float(maximum_delta_sigma), dtype=torch.float32),
        )

    def center_from_action(
        self, anchor_knots: torch.Tensor, normalized_action: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if tuple(normalized_action.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("normalized_action must have shape [B,8,2]")
        requested_delta = (
            normalized_action * self.maximum_delta_sigma * self.base_sigma
        )
        center = torch.clamp(anchor_knots + requested_delta, -1.0, 1.0)
        effective_action = (center - anchor_knots) / (
            self.maximum_delta_sigma * self.base_sigma
        )
        return effective_action, center

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        feature = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        raw_action = self.action_head(feature).reshape(
            -1, self.knot_count, self.action_dim
        )
        return self.center_from_action(anchor_knots, torch.tanh(raw_action))

    def load_stochastic_actor_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Transfer the encoder and mean head from the old SAC Actor."""
        encoder_prefix = "encoder."
        encoder_state = {
            name[len(encoder_prefix):]: value
            for name, value in state_dict.items()
            if name.startswith(encoder_prefix)
        }
        self.encoder.load_state_dict(encoder_state, strict=True)
        self.action_head.load_state_dict(
            {
                "weight": state_dict["mean_head.weight"],
                "bias": state_dict["mean_head.bias"],
            },
            strict=True,
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPITrustAlphaPolicy(nn.Module):
    """Deterministic move/stay gate and scalar step along a frozen trust line.

    The old/proposal Actors and trust projection remain external and frozen.  This
    policy consumes their projected normalized direction plus its requested radius
    and projection scale, then predicts whether to move and the conditional
    ``alpha in [0,1]``.  A hard move threshold gives an exact alpha-zero fallback;
    the deployed center remains one unique deterministic output.
    """

    direction_dim = TorchMPPIDeterministicCenterActor.flat_action_dim + 2

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.direction_encoder = nn.Sequential(
            nn.Linear(self.direction_dim, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 64, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.SiLU(),
        )
        self.move_head = nn.Linear(64, 1)
        self.alpha_head = nn.Linear(64, 1)
        # Exactly reproduce the old Actor before training: sigmoid(0)=0.5 and
        # the hard policy moves only when probability is strictly above 0.5.
        nn.init.zeros_(self.move_head.weight)
        nn.init.zeros_(self.move_head.bias)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)

    def load_actor_encoder_state_dict(
        self, actor_state_dict: dict[str, torch.Tensor]
    ) -> None:
        prefix = "encoder."
        state = {
            name[len(prefix):]: value
            for name, value in actor_state_dict.items()
            if name.startswith(prefix)
        }
        self.encoder.load_state_dict(state, strict=True)

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        projected_direction: torch.Tensor,
        requested_rho: torch.Tensor,
        trust_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if projected_direction.shape[1:] != (8, 2):
            raise ValueError("projected_direction must have shape [B,8,2]")
        if requested_rho.shape[1:] != (1,) or trust_scale.shape[1:] != (1,):
            raise ValueError("requested_rho and trust_scale must have shape [B,1]")
        context = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        direction = self.direction_encoder(torch.cat((
            projected_direction.flatten(1), requested_rho, trust_scale,
        ), dim=1))
        feature = self.fusion(torch.cat((context, direction), dim=1))
        move_logit = self.move_head(feature).squeeze(1)
        conditional_alpha = torch.sigmoid(self.alpha_head(feature).squeeze(1))
        move_probability = torch.sigmoid(move_logit)
        return move_logit, move_probability, conditional_alpha

    @staticmethod
    def hard_alpha(
        move_probability: torch.Tensor,
        conditional_alpha: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        return torch.where(
            move_probability > threshold,
            conditional_alpha,
            torch.zeros_like(conditional_alpha),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPITrustAlphaSACPolicy(nn.Module):
    """Hybrid SAC policy for stay/move plus a continuous conditional alpha.

    The Bernoulli stay/move branch is enumerated in the Actor objective, so no
    biased gradient estimator is required for the discrete decision.  The move
    branch uses a reparameterized logit-normal alpha on ``(0, 1)``.  Deployment
    remains deterministic: threshold the move probability once and use the
    sigmoid of the conditional mean logit.  Training exploration is therefore
    absent from the serialized runtime output.
    """

    direction_dim = TorchMPPITrustAlphaPolicy.direction_dim

    def __init__(
        self,
        dropout: float = 0.05,
        minimum_log_std: float = -4.0,
        maximum_log_std: float = 1.0,
        initial_log_std: float = -0.5,
        alpha_logit_scale: float = 1.0,
    ) -> None:
        super().__init__()
        self.minimum_log_std = float(minimum_log_std)
        self.maximum_log_std = float(maximum_log_std)
        self.alpha_logit_scale = float(alpha_logit_scale)
        if self.alpha_logit_scale <= 0.0:
            raise ValueError("alpha_logit_scale must be positive")
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.direction_encoder = nn.Sequential(
            nn.Linear(self.direction_dim, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 64, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 64), nn.SiLU(),
        )
        self.move_head = nn.Linear(64, 1)
        self.alpha_head = nn.Linear(64, 1)
        self.alpha_log_std_head = nn.Linear(64, 1)
        nn.init.zeros_(self.move_head.weight)
        nn.init.zeros_(self.move_head.bias)
        nn.init.zeros_(self.alpha_head.weight)
        nn.init.zeros_(self.alpha_head.bias)
        nn.init.zeros_(self.alpha_log_std_head.weight)
        nn.init.constant_(self.alpha_log_std_head.bias, float(initial_log_std))

    def load_deterministic_policy_state_dict(
        self, state_dict: dict[str, torch.Tensor]
    ) -> None:
        """Initialize the shared policy and means without inventing old std state."""
        expected = set(self.state_dict()) - {
            "alpha_log_std_head.weight", "alpha_log_std_head.bias"
        }
        if set(state_dict) != expected:
            missing = sorted(expected - set(state_dict))
            unexpected = sorted(set(state_dict) - expected)
            raise RuntimeError(
                f"deterministic alpha-policy state mismatch: missing={missing}, "
                f"unexpected={unexpected}"
            )
        current = self.state_dict()
        current.update(state_dict)
        self.load_state_dict(current, strict=True)

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        projected_direction: torch.Tensor,
        requested_rho: torch.Tensor,
        trust_scale: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if projected_direction.shape[1:] != (8, 2):
            raise ValueError("projected_direction must have shape [B,8,2]")
        if requested_rho.shape[1:] != (1,) or trust_scale.shape[1:] != (1,):
            raise ValueError("requested_rho and trust_scale must have shape [B,1]")
        context = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        direction = self.direction_encoder(torch.cat((
            projected_direction.flatten(1), requested_rho, trust_scale,
        ), dim=1))
        feature = self.fusion(torch.cat((context, direction), dim=1))
        move_logit = self.move_head(feature).squeeze(1)
        # A scale below one is an optional training-time desaturation of a
        # deterministic predecessor whose sigmoid alpha head collapsed near 1.
        # It changes only the continuous-action parameterization, not the action
        # range or the unique deterministic deployment contract.
        mean_logit = self.alpha_logit_scale * self.alpha_head(feature).squeeze(1)
        log_std = torch.clamp(
            self.alpha_log_std_head(feature).squeeze(1),
            self.minimum_log_std,
            self.maximum_log_std,
        )
        return move_logit, torch.sigmoid(move_logit), mean_logit, log_std

    @staticmethod
    def deterministic_alpha(
        move_probability: torch.Tensor,
        mean_logit: torch.Tensor,
        threshold: float = 0.5,
    ) -> torch.Tensor:
        conditional = torch.sigmoid(mean_logit)
        return torch.where(
            move_probability > threshold,
            conditional,
            torch.zeros_like(conditional),
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPITrustAlphaCritic(nn.Module):
    """Continuous ``Q(context, alpha)`` on a frozen old-to-proposal trust line.

    The physical center direction remains an explicit input so the same state can
    be scored under a different frozen proposal Actor.  ``alpha`` is the only
    optimized action and is constrained to ``[0, 1]`` by the caller.  The bank
    method encodes each context once, which is important for the 21-point TR1
    replay curve.
    """

    direction_dim = TorchMPPITrustAlphaPolicy.direction_dim
    alpha_feature_dim = 5

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.direction_encoder = nn.Sequential(
            nn.Linear(self.direction_dim, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.state_fusion = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 64, 192), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(192, 128), nn.SiLU(),
        )
        self.alpha_encoder = nn.Sequential(
            nn.Linear(self.alpha_feature_dim, 32), nn.SiLU(),
            nn.Linear(32, 32), nn.SiLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(128 + 32, 128), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(128, 64), nn.SiLU(), nn.Linear(64, 1),
        )

    @staticmethod
    def alpha_features(alpha: torch.Tensor) -> torch.Tensor:
        if alpha.ndim != 1:
            raise ValueError("alpha must have shape [B]")
        return torch.stack((
            alpha,
            alpha.square(),
            torch.sin(math.pi * alpha),
            torch.cos(math.pi * alpha),
            torch.cos(2.0 * math.pi * alpha),
        ), dim=1)

    def encode_context(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        projected_direction: torch.Tensor,
        requested_rho: torch.Tensor,
        trust_scale: torch.Tensor,
    ) -> torch.Tensor:
        if projected_direction.shape[1:] != (8, 2):
            raise ValueError("projected_direction must have shape [B,8,2]")
        if requested_rho.shape[1:] != (1,) or trust_scale.shape[1:] != (1,):
            raise ValueError("requested_rho and trust_scale must have shape [B,1]")
        context = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        direction = self.direction_encoder(torch.cat((
            projected_direction.flatten(1), requested_rho, trust_scale,
        ), dim=1))
        return self.state_fusion(torch.cat((context, direction), dim=1))

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        projected_direction: torch.Tensor,
        requested_rho: torch.Tensor,
        trust_scale: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        state = self.encode_context(
            history, reference, current, anchor_knots, feedback, gradient_context,
            projected_direction, requested_rho, trust_scale,
        )
        action = self.alpha_encoder(self.alpha_features(alpha))
        return self.q_head(torch.cat((state, action), dim=1)).squeeze(1)

    def forward_alpha_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        projected_direction: torch.Tensor,
        requested_rho: torch.Tensor,
        trust_scale: torch.Tensor,
        alpha: torch.Tensor,
    ) -> torch.Tensor:
        if alpha.ndim != 2:
            raise ValueError("alpha bank must have shape [B,K]")
        state = self.encode_context(
            history, reference, current, anchor_knots, feedback, gradient_context,
            projected_direction, requested_rho, trust_scale,
        )
        batch, count = alpha.shape
        action = self.alpha_encoder(
            self.alpha_features(alpha.reshape(batch * count))
        )
        expanded = state[:, None].expand(-1, count, -1).reshape(batch * count, -1)
        return self.q_head(torch.cat((expanded, action), dim=1)).reshape(batch, count)

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIContinuousCenterCritic(nn.Module):
    """Scalar Q(s,a) for a continuous normalized 16-D center residual."""

    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.action_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 128), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(128, 128), nn.SiLU(),
        )
        self.q_head = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 128, 256), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(256, 128), nn.SiLU(),
            nn.Linear(128, 1),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(normalized_action.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("normalized_action must have shape [B,8,2]")
        state = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        action = self.action_encoder(normalized_action.flatten(1))
        return self.q_head(torch.cat((state, action), dim=1)).squeeze(-1)

    def forward_action_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        """Evaluate ``[B,K,8,2]`` actions while encoding each context once."""
        if normalized_action.ndim != 4 or tuple(normalized_action.shape[2:]) != (
            self.knot_count,
            self.action_dim,
        ):
            raise ValueError("normalized_action must have shape [B,K,8,2]")
        state = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        batch, count = normalized_action.shape[:2]
        action = self.action_encoder(normalized_action.reshape(batch * count, -1))
        expanded_state = state[:, None].expand(-1, count, -1).reshape(
            batch * count, -1
        )
        return self.q_head(torch.cat((expanded_state, action), dim=1)).reshape(
            batch, count
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIActorCenteredLocalCritic(nn.Module):
    """Explicit full-16D local Q model around a frozen Actor action.

    Unlike :class:`TorchMPPIContinuousCenterCritic`, the action derivative does
    not have to emerge through an unconstrained action MLP.  The state/context
    network directly predicts all sixteen gradient components plus a scalar
    radial curvature.  ``anchor_action`` is the deterministic Actor output used
    as the local expansion point; it is a Critic input only and does not change
    the Actor deployment contract.
    """

    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    flat_action_dim = knot_count * action_dim

    def __init__(self, dropout: float = 0.05) -> None:
        super().__init__()
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.anchor_action_encoder = nn.Sequential(
            nn.Linear(self.flat_action_dim, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(self.encoder.output_dim + 64, 256), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(256, 192), nn.SiLU(),
        )
        self.value_head = nn.Linear(192, 1)
        self.gradient_head = nn.Linear(192, self.flat_action_dim)
        self.curvature_head = nn.Linear(192, 1)
        nn.init.zeros_(self.value_head.weight)
        nn.init.zeros_(self.value_head.bias)
        nn.init.zeros_(self.gradient_head.weight)
        nn.init.zeros_(self.gradient_head.bias)
        nn.init.zeros_(self.curvature_head.weight)
        nn.init.zeros_(self.curvature_head.bias)

    def local_parameters(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        anchor_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(anchor_action.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("anchor_action must have shape [B,8,2]")
        state = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        action = self.anchor_action_encoder(anchor_action.flatten(1))
        feature = self.fusion(torch.cat((state, action), dim=1))
        value = self.value_head(feature).squeeze(-1)
        gradient = self.gradient_head(feature).reshape(
            -1, self.knot_count, self.action_dim
        )
        curvature = self.curvature_head(feature).squeeze(-1)
        return value, gradient, curvature

    @staticmethod
    def local_value(
        value: torch.Tensor,
        gradient: torch.Tensor,
        curvature: torch.Tensor,
        action_delta: torch.Tensor,
    ) -> torch.Tensor:
        linear = torch.sum(gradient * action_delta, dim=(-2, -1))
        radial = torch.sum(action_delta.square(), dim=(-2, -1))
        return value + linear + 0.5 * curvature * radial

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        anchor_action: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(normalized_action.shape[1:]) != (
            self.knot_count, self.action_dim
        ):
            raise ValueError("normalized_action must have shape [B,8,2]")
        value, gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots, feedback,
            gradient_context, anchor_action,
        )
        return self.local_value(
            value, gradient, curvature, normalized_action - anchor_action
        )

    def forward_action_bank(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        anchor_action: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        if normalized_action.ndim != 4 or tuple(normalized_action.shape[2:]) != (
            self.knot_count, self.action_dim
        ):
            raise ValueError("normalized_action must have shape [B,K,8,2]")
        value, gradient, curvature = self.local_parameters(
            history, reference, current, anchor_knots, feedback,
            gradient_context, anchor_action,
        )
        return self.local_value(
            value[:, None], gradient[:, None], curvature[:, None],
            normalized_action - anchor_action[:, None],
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPIAbsoluteCenterLocalCritic(TorchMPPIActorCenteredLocalCritic):
    """Local Critic whose action branch sees the absolute physical MPPI center.

    ``anchor_action`` remains the normalized residual coordinate used by the
    Actor and by the local Taylor expansion.  Only the action-conditioning
    feature changes: it is reconstructed as the unique absolute control center
    ``anchor_knots + residual * maximum_residual_sigma * base_sigma``.  This
    makes two residuals that land at the same physical center identical to the
    action encoder, while retaining the existing Actor/deployment contract.

    The current fixed-DBM dataset and deployment MPPI use the fixed noise sigma
    ``[0.25, 0.35]``.  Keeping it explicit here avoids silently interpreting a
    normalized residual as an absolute action location.
    """

    base_sigma = (0.25, 0.35)

    def __init__(
        self,
        dropout: float = 0.05,
        maximum_residual_sigma: float = 2.0,
    ) -> None:
        super().__init__(dropout)
        self.maximum_residual_sigma = float(maximum_residual_sigma)

    def local_parameters(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        anchor_action: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if tuple(anchor_action.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("anchor_action must have shape [B,8,2]")
        state = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        sigma = anchor_action.new_tensor(self.base_sigma).view(1, 1, 2)
        absolute_center = (
            anchor_knots
            + anchor_action * self.maximum_residual_sigma * sigma
        )
        action = self.anchor_action_encoder(absolute_center.flatten(1))
        feature = self.fusion(torch.cat((state, action), dim=1))
        value = self.value_head(feature).squeeze(-1)
        gradient = self.gradient_head(feature).reshape(
            -1, self.knot_count, self.action_dim
        )
        curvature = self.curvature_head(feature).squeeze(-1)
        return value, gradient, curvature


class TorchMPPISemanticStateActionEncoder(nn.Module):
    """Encode physical state and an explicit absolute MPPI action location.

    The legacy continuous-center encoder mixes the first-pass anchor, feedback,
    and gradient context.  This encoder makes the semantic contract explicit:
    history/reference/current describe the physical state, ``anchor_knots`` is
    the absolute Actor center where the local Q expansion is evaluated, and
    feedback is an optional ablation input.  Gradient context is accepted only
    to keep a common call signature and is intentionally ignored.
    """

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    feedback_dim = 74
    gradient_context_dim = 32
    output_dim = 192

    def __init__(self, include_feedback: bool, dropout: float = 0.05) -> None:
        super().__init__()
        self.include_feedback = bool(include_feedback)
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.absolute_action_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        if self.include_feedback:
            self.feedback_encoder = nn.Sequential(
                nn.Linear(self.feedback_dim, 128), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(128, 128), nn.SiLU(),
            )
        else:
            self.feedback_encoder = None
        input_dim = 128 + 128 + 64 + 64
        if self.include_feedback:
            input_dim += 128
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 320), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(320, self.output_dim), nn.SiLU(),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(anchor_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("absolute Actor center must have shape [B,8,2]")
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        if gradient_context.shape[1:] != (self.gradient_context_dim,):
            raise ValueError(
                f"gradient_context must have shape [B,{self.gradient_context_dim}]"
            )
        parts = [
            self.history_encoder(history),
            self.reference_encoder(reference),
            self.current_encoder(current),
            self.absolute_action_encoder(anchor_knots.flatten(1)),
        ]
        if self.feedback_encoder is not None:
            parts.append(self.feedback_encoder(feedback))
        return self.fusion(torch.cat(parts, dim=1))


class TorchMPPIStructuredLocalQCritic(nn.Module):
    """State-conditioned, integrable local quadratic Q in the full 16-D action.

    Candidate actions have no generic MLP path.  Their only direct influence is
    through ``Q0 + g0^T delta + 0.5 delta^T H delta``.  ``H`` is symmetric by
    construction, signed, and optionally augments a signed diagonal with a
    signed low-rank term.  Setting ``hessian_enabled=False`` is the same-loss H0
    control arm.
    """

    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    flat_action_dim = knot_count * action_dim

    def __init__(
        self,
        low_rank: int = 0,
        dropout: float = 0.05,
        hessian_scale: float = 256.0,
        hessian_enabled: bool = True,
    ) -> None:
        super().__init__()
        if low_rank < 0:
            raise ValueError("low_rank must be nonnegative")
        if hessian_scale <= 0:
            raise ValueError("hessian_scale must be positive")
        self.low_rank = int(low_rank)
        self.hessian_scale = float(hessian_scale)
        self.hessian_enabled = bool(hessian_enabled)
        self.encoder = TorchMPPIContinuousCenterEncoder(dropout)
        self.trunk = nn.Sequential(
            nn.Linear(self.encoder.output_dim, 256), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(256, 192), nn.SiLU(),
        )
        self.value_head = nn.Linear(192, 1)
        self.gradient_head = nn.Linear(192, self.flat_action_dim)
        self.diagonal_head = nn.Linear(192, self.flat_action_dim)
        if self.low_rank:
            self.low_rank_vector_head = nn.Linear(
                192, self.flat_action_dim * self.low_rank
            )
            self.low_rank_value_head = nn.Linear(192, self.low_rank)
        else:
            self.low_rank_vector_head = None
            self.low_rank_value_head = None
        for head in (self.value_head, self.gradient_head, self.diagonal_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        if self.low_rank_value_head is not None:
            nn.init.zeros_(self.low_rank_value_head.weight)
            nn.init.zeros_(self.low_rank_value_head.bias)

    def _signed_bounded(self, raw: torch.Tensor) -> torch.Tensor:
        return self.hessian_scale * torch.tanh(raw / self.hessian_scale)

    def local_parameters(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        state = self.encoder(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        feature = self.trunk(state)
        value = self.value_head(feature).squeeze(-1)
        gradient = self.gradient_head(feature)
        batch = len(feature)
        if not self.hessian_enabled:
            hessian = feature.new_zeros(
                batch, self.flat_action_dim, self.flat_action_dim
            )
        else:
            diagonal = self._signed_bounded(self.diagonal_head(feature))
            hessian = torch.diag_embed(diagonal)
            if self.low_rank:
                assert self.low_rank_vector_head is not None
                assert self.low_rank_value_head is not None
                vectors = self.low_rank_vector_head(feature).reshape(
                    batch, self.flat_action_dim, self.low_rank
                )
                vectors = F.normalize(vectors, p=2, dim=1, eps=1e-6)
                signed_value = self._signed_bounded(
                    self.low_rank_value_head(feature)
                )
                hessian = hessian + torch.einsum(
                    "bir,br,bjr->bij", vectors, signed_value, vectors
                )
            # Make the mathematical symmetry bitwise explicit in float32 so
            # structural gates do not depend on einsum accumulation order.
            hessian = 0.5 * (hessian + hessian.transpose(1, 2))
        return value, gradient, hessian

    @staticmethod
    def local_value(
        value: torch.Tensor,
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        action_delta: torch.Tensor,
    ) -> torch.Tensor:
        delta = action_delta.flatten(start_dim=-2)
        linear = torch.sum(gradient * delta, dim=-1)
        quadratic = torch.einsum("...i,...ij,...j->...", delta, hessian, delta)
        return value + linear + 0.5 * quadratic

    @staticmethod
    def local_gradient(
        gradient: torch.Tensor,
        hessian: torch.Tensor,
        action_delta: torch.Tensor,
    ) -> torch.Tensor:
        delta = action_delta.flatten(start_dim=-2)
        return gradient + torch.einsum("...ij,...j->...i", hessian, delta)

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
        reference_action: torch.Tensor,
        normalized_action: torch.Tensor,
    ) -> torch.Tensor:
        value, gradient, hessian = self.local_parameters(
            history, reference, current, anchor_knots, feedback, gradient_context
        )
        return self.local_value(
            value, gradient, hessian, normalized_action - reference_action
        )

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


class TorchMPPISemanticStructuredLocalQCritic(TorchMPPIStructuredLocalQCritic):
    """Structured local Q with a clean physical-state/action-location contract."""

    def __init__(
        self,
        include_feedback: bool,
        low_rank: int = 2,
        dropout: float = 0.05,
        hessian_scale: float = 256.0,
        hessian_enabled: bool = True,
    ) -> None:
        super().__init__(
            low_rank=low_rank,
            dropout=dropout,
            hessian_scale=hessian_scale,
            hessian_enabled=hessian_enabled,
        )
        self.include_feedback = bool(include_feedback)
        self.encoder = TorchMPPISemanticStateActionEncoder(
            include_feedback=self.include_feedback,
            dropout=dropout,
        )


class TorchMPPISemanticInteractionStateActionEncoder(nn.Module):
    """Semantic encoder with an explicit state-by-action bilinear interaction.

    Concatenation-only fusion can bury the absolute action signal; this
    encoder additionally projects the physical-state feature and the absolute
    action feature to a shared width and feeds their elementwise product (a
    low-rank bilinear form) to the fusion layer.
    """

    history_length = TorchMPPIProposalPolicy.history_length
    reference_length = TorchMPPIProposalPolicy.reference_length
    knot_count = TorchMPPIProposalPolicy.knot_count
    action_dim = TorchMPPIProposalPolicy.action_dim
    feedback_dim = 74
    gradient_context_dim = 32
    output_dim = 192

    def __init__(self, include_feedback: bool, dropout: float = 0.05) -> None:
        super().__init__()
        self.include_feedback = bool(include_feedback)
        self.history_encoder = TemporalConvEncoder(7, 8, 128, dropout)
        self.reference_encoder = TemporalConvEncoder(5, 5, 128, dropout)
        self.current_encoder = nn.Sequential(
            nn.Linear(4, 64), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(64, 64), nn.SiLU(),
        )
        self.absolute_action_encoder = nn.Sequential(
            nn.Linear(self.knot_count * self.action_dim, 64), nn.SiLU(),
            nn.Dropout(dropout), nn.Linear(64, 64), nn.SiLU(),
        )
        if self.include_feedback:
            self.feedback_encoder = nn.Sequential(
                nn.Linear(self.feedback_dim, 128), nn.SiLU(),
                nn.Dropout(dropout), nn.Linear(128, 128), nn.SiLU(),
            )
        else:
            self.feedback_encoder = None
        state_dim = 128 + 128 + 64
        self.state_projection = nn.Linear(state_dim, 64)
        self.action_projection = nn.Linear(64, 64)
        input_dim = state_dim + 64 + 64
        if self.include_feedback:
            input_dim += 128
        self.fusion = nn.Sequential(
            nn.Linear(input_dim, 320), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(320, self.output_dim), nn.SiLU(),
        )

    def forward(
        self,
        history: torch.Tensor,
        reference: torch.Tensor,
        current: torch.Tensor,
        anchor_knots: torch.Tensor,
        feedback: torch.Tensor,
        gradient_context: torch.Tensor,
    ) -> torch.Tensor:
        if tuple(history.shape[1:]) != (self.history_length, 7):
            raise ValueError("history must have shape [B,250,7]")
        if tuple(reference.shape[1:]) != (self.reference_length, 5):
            raise ValueError("reference must have shape [B,50,5]")
        if current.shape[1:] != (4,):
            raise ValueError("current must have shape [B,4]")
        if tuple(anchor_knots.shape[1:]) != (self.knot_count, self.action_dim):
            raise ValueError("absolute Actor center must have shape [B,8,2]")
        if feedback.shape[1:] != (self.feedback_dim,):
            raise ValueError(f"feedback must have shape [B,{self.feedback_dim}]")
        if gradient_context.shape[1:] != (self.gradient_context_dim,):
            raise ValueError(
                f"gradient_context must have shape [B,{self.gradient_context_dim}]"
            )
        state_feature = torch.cat((
            self.history_encoder(history),
            self.reference_encoder(reference),
            self.current_encoder(current),
        ), dim=1)
        action_feature = self.absolute_action_encoder(
            anchor_knots.flatten(1)
        )
        interaction = self.state_projection(state_feature) * (
            self.action_projection(action_feature)
        )
        parts = [state_feature, action_feature, interaction]
        if self.feedback_encoder is not None:
            parts.append(self.feedback_encoder(feedback))
        return self.fusion(torch.cat(parts, dim=1))


class TorchMPPISemanticInteractionStructuredLocalQCritic(
    TorchMPPIStructuredLocalQCritic
):
    """Structured local Q with an explicit state-action interaction encoder."""

    def __init__(
        self,
        include_feedback: bool,
        low_rank: int = 2,
        dropout: float = 0.05,
        hessian_scale: float = 256.0,
        hessian_enabled: bool = True,
    ) -> None:
        super().__init__(
            low_rank=low_rank,
            dropout=dropout,
            hessian_scale=hessian_scale,
            hessian_enabled=hessian_enabled,
        )
        self.include_feedback = bool(include_feedback)
        self.encoder = TorchMPPISemanticInteractionStateActionEncoder(
            include_feedback=self.include_feedback,
            dropout=dropout,
        )
