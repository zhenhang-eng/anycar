"""Runtime reconstruction of the trained deterministic residual MPPI Actor.

The current Actor is feedback-conditioned: one frozen, deterministic 128-candidate
first pass is part of inference.  This module contains the deployable PyTorch path
without importing research scripts or cached label files.  It uses only live
state/history/reference, the current warm knots, frozen checkpoints, and forward
rollouts through the controller backend.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import math
from pathlib import Path
import time
from typing import Any, Dict, Sequence

import numpy as np
import torch

from car_foundation.mppi_proposal_policy import (
    MPPIProposalNormalization,
    TorchMPPIDeterministicCenterActor,
    TorchMPPIFeedbackQuadraticCritic,
    TorchMPPIProposalPolicy,
    TorchMPPITrustAlphaSACPolicy,
    ego_reference_features,
)


@dataclass(frozen=True)
class ResidualActorRuntimeConfig:
    first_seed: int = 24001
    first_samples: int = 128
    fit_ridge: float = 0.10
    step_damping: float = 0.10
    max_standardized_step: float = 1.0

    def __post_init__(self) -> None:
        if self.first_samples < 4 or self.first_samples % 2:
            raise ValueError("first_samples must be an even integer >= 4")
        if self.fit_ridge <= 0 or self.step_damping <= 0:
            raise ValueError("ridge and damping must be positive")
        if self.max_standardized_step <= 0:
            raise ValueError("max_standardized_step must be positive")


@dataclass
class ResidualActorRuntimeOutput:
    center_knots: torch.Tensor
    action_sequence: torch.Tensor
    bc_center_knots: torch.Tensor
    guided_center_knots: torch.Tensor
    first_pass_feedback: torch.Tensor
    critic_gradient_mean: torch.Tensor
    critic_gradient_std: torch.Tensor
    old_center_knots: torch.Tensor
    proposal_center_knots: torch.Tensor
    base_center_knots: torch.Tensor
    move_probability: float
    base_alpha: float
    first_pass_cost: torch.Tensor
    first_pass_knots: torch.Tensor
    duration_s: float
    rollout_count: int


def reference_in_ego_frame(reference: np.ndarray, state: Sequence[float]) -> np.ndarray:
    """Transform global ``[x,y,yaw,vx]`` reference values into the current ego frame."""
    local = np.asarray(reference, dtype=np.float32).copy()
    if local.ndim != 2 or local.shape[1] != 4:
        raise ValueError("reference must have shape [50/51,4]")
    state = np.asarray(state, dtype=np.float32)
    delta = local[:, :2] - state[:2]
    yaw = float(state[2])
    cosine, sine = np.cos(yaw), np.sin(yaw)
    local[:, 0] = cosine * delta[:, 0] + sine * delta[:, 1]
    local[:, 1] = -sine * delta[:, 0] + cosine * delta[:, 1]
    yaw_delta = local[:, 2] - yaw
    local[:, 2] = np.arctan2(np.sin(yaw_delta), np.cos(yaw_delta))
    return local


def _as_numpy(value: Any, dtype=np.float32) -> np.ndarray:
    if torch.is_tensor(value):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=dtype)


def _antithetic_candidates(
    center: np.ndarray,
    sigma: np.ndarray,
    count: int,
    seed: int,
    action_min: np.ndarray,
    action_max: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    extra = rng.standard_normal(center.shape).astype(np.float32)
    second = center + extra * sigma
    pair_count = (count - 2) // 2
    direction = rng.standard_normal((pair_count, *center.shape)).astype(np.float32)
    perturbation = direction * sigma
    raw = np.concatenate(
        (
            center[None],
            second[None],
            center[None] + perturbation,
            center[None] - perturbation,
        ),
        axis=0,
    ).astype(np.float32)
    return raw, np.clip(raw, action_min, action_max).astype(np.float32)


def _weighted_residuals(
    controller,
    trajectories: torch.Tensor,
    actions: torch.Tensor,
    reference: torch.Tensor,
    current_action: torch.Tensor,
) -> torch.Tensor:
    weights = controller.cost_weights
    fields = [
        math.sqrt(weights.position)
        * (trajectories[..., 0:2] - reference[None, :, 0:2]),
        math.sqrt(weights.yaw)
        * controller._wrapped_angle_difference(
            trajectories[..., 2], reference[None, :, 2]
        ).unsqueeze(-1),
        math.sqrt(weights.vx)
        * (trajectories[..., 3] - reference[None, :, 3]).unsqueeze(-1),
    ]
    if reference.shape[1] == 5 and weights.yawrate != 0:
        fields.append(
            math.sqrt(weights.yawrate)
            * (trajectories[..., 4] - reference[None, :, 4]).unsqueeze(-1)
        )
    previous = torch.cat(
        (current_action.expand(actions.shape[0], 1, -1), actions[:, :-1]), dim=1
    )
    action_rate = actions - previous
    fields.extend(
        (
            math.sqrt(weights.acceleration_rate) * action_rate[..., 0:1],
            math.sqrt(weights.steering_rate) * action_rate[..., 1:2],
        )
    )
    return torch.cat(fields, dim=-1).reshape(actions.shape[0], -1)


def _load_deterministic_actor(payload: dict[str, Any], device: torch.device):
    actor = TorchMPPIDeterministicCenterActor(
        maximum_delta_sigma=float(payload["maximum_delta_sigma"]), dropout=0.0
    ).to(device)
    actor.load_state_dict(payload["actor_state_dict"], strict=True)
    return actor.eval()


class ResidualActorRuntime:
    """Frozen online center generator matching the selected residual checkpoint."""

    DEFAULT_BC = Path(
        "outputs/mppi_proposal/bc_t1_conv_diverse_20260805_v1/trust1_seed1.pt"
    )
    DEFAULT_FEEDBACK_CRITIC_DIR = Path(
        "outputs/mppi_proposal/critic_two_pass_feedback_20260805_v1"
    )

    def __init__(
        self,
        checkpoint: Path | str,
        *,
        bc_checkpoint: Path | str = DEFAULT_BC,
        feedback_critic_dir: Path | str = DEFAULT_FEEDBACK_CRITIC_DIR,
        config: ResidualActorRuntimeConfig | None = None,
        device: str | torch.device = "cuda",
    ) -> None:
        self.device = torch.device(device)
        self.config = config or ResidualActorRuntimeConfig()
        self.checkpoint_path = Path(checkpoint).resolve()
        residual_payload = torch.load(self.checkpoint_path, map_location="cpu")
        base_payload = torch.load(
            Path(residual_payload["base_alpha_checkpoint"]), map_location="cpu"
        )
        old_payload = torch.load(Path(base_payload["old_actor"]), map_location="cpu")
        proposal_payload = torch.load(
            Path(base_payload["proposal_actor"]), map_location="cpu"
        )

        self.old_actor = _load_deterministic_actor(old_payload, self.device)
        self.proposal_actor = _load_deterministic_actor(proposal_payload, self.device)
        self.base_policy = TorchMPPITrustAlphaSACPolicy(
            dropout=0.0,
            alpha_logit_scale=float(base_payload.get("alpha_logit_scale", 1.0)),
        ).to(self.device)
        self.base_policy.load_state_dict(base_payload["policy_state_dict"], strict=True)
        self.base_policy.eval()
        self.residual_actor = TorchMPPIDeterministicCenterActor(
            maximum_delta_sigma=float(residual_payload["maximum_residual_sigma"]),
            dropout=0.0,
        ).to(self.device)
        self.residual_actor.load_state_dict(
            residual_payload["actor_state_dict"], strict=True
        )
        self.residual_actor.eval()
        self.move_threshold = float(base_payload["move_threshold"])
        self.actor_normalization = MPPIProposalNormalization.from_dict(
            old_payload["state_normalization"]
        )
        self.actor_feedback_mean = np.asarray(old_payload["feedback_mean"], np.float32)
        self.actor_feedback_std = np.asarray(old_payload["feedback_std"], np.float32)
        self.actor_gradient_mean = np.asarray(old_payload["gradient_mean"], np.float32)
        self.actor_gradient_std = np.asarray(old_payload["gradient_std"], np.float32)

        labels_config = json.loads(
            (Path(residual_payload["labels"]) / "config.json").read_text()
        )
        self.trust_radius = float(labels_config["trust_radius_sigma_rms"])

        bc_payload = torch.load(Path(bc_checkpoint).resolve(), map_location="cpu")
        architecture = bc_payload["architecture"]
        self.bc_actor = TorchMPPIProposalPolicy(
            trust_scale=tuple(architecture["trust_scale"]), dropout=0.0
        ).to(self.device)
        self.bc_actor.load_state_dict(bc_payload["model_state_dict"], strict=True)
        self.bc_actor.eval()
        self.bc_normalization = MPPIProposalNormalization.from_dict(
            bc_payload["normalization"]
        )

        critic_summary = json.loads(
            (Path(feedback_critic_dir).resolve() / "training_summary.json").read_text()
        )
        self.feedback_critics = []
        self.feedback_critic_normalization = None
        self.feedback_mean = self.feedback_std = None
        for path_text in critic_summary["ensemble_checkpoints"]:
            payload = torch.load(Path(path_text), map_location="cpu")
            critic = TorchMPPIFeedbackQuadraticCritic(dropout=0.0).to(self.device)
            critic.load_state_dict(payload["model_state_dict"], strict=True)
            self.feedback_critics.append(critic.eval())
            if self.feedback_critic_normalization is None:
                self.feedback_critic_normalization = MPPIProposalNormalization.from_dict(
                    payload["state_normalization"]
                )
                self.feedback_mean = np.asarray(payload["feedback_mean"], np.float32)
                self.feedback_std = np.asarray(payload["feedback_std"], np.float32)
        if not self.feedback_critics:
            raise ValueError("feedback Critic ensemble is empty")

    def _policy_state_inputs(
        self,
        history: np.ndarray,
        reference_ego: np.ndarray,
        initial_state: np.ndarray,
        current_action: np.ndarray,
        normalization: MPPIProposalNormalization,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reference = ego_reference_features(reference_ego, float(initial_state[3]))
        current = np.asarray(
            (initial_state[3], initial_state[4], *current_action), np.float32
        )
        return normalization.normalize_numpy(history, reference, current)

    @torch.no_grad()
    def _bc_center(
        self,
        history: np.ndarray,
        reference_ego: np.ndarray,
        initial_state: np.ndarray,
        current_action: np.ndarray,
        warm_knots: np.ndarray,
    ) -> torch.Tensor:
        normalized = self._policy_state_inputs(
            history,
            reference_ego,
            initial_state,
            current_action,
            self.bc_normalization,
        )
        _, center = self.bc_actor(
            *(
                torch.from_numpy(np.asarray(value, np.float32)[None]).to(self.device)
                for value in normalized
            ),
            torch.from_numpy(warm_knots[None]).to(self.device),
        )
        return center[0]

    @torch.no_grad()
    def _first_pass(
        self,
        controller,
        initial_state: np.ndarray,
        current_action: np.ndarray,
        history: np.ndarray,
        reference: np.ndarray,
        base_center: torch.Tensor,
        first_seed: int,
    ) -> Dict[str, torch.Tensor]:
        sigma = np.asarray(controller.params.noise_sigma, np.float32).reshape(1, 2)
        raw, clipped = _antithetic_candidates(
            base_center.detach().cpu().numpy(),
            sigma,
            self.config.first_samples,
            first_seed,
            np.asarray(controller.params.action_min, np.float32),
            np.asarray(controller.params.action_max, np.float32),
        )
        knots = torch.from_numpy(clipped).to(self.device)
        actions = controller._interpolate_knots(knots)
        evaluation = controller.evaluate_action_sequences(
            initial_state, current_action, history[None], reference, actions
        )
        trajectories = evaluation["trajectories"]
        cost = evaluation["cost"]
        prepared_reference = controller._prepare_reference(reference)
        current_t = torch.from_numpy(current_action).to(self.device).reshape(1, 2)
        residuals = _weighted_residuals(
            controller,
            trajectories,
            actions,
            prepared_reference,
            current_t,
        )

        normalized_delta = ((knots - base_center) / torch.from_numpy(sigma).to(self.device)).reshape(
            self.config.first_samples, -1
        )
        baseline_residual = residuals[0]
        residual_delta = residuals - baseline_residual
        cost_scale = torch.quantile(cost, 0.5).clamp_min(1e-6)
        fit_weight = torch.exp(-(cost - cost.min()) / cost_scale)
        sqrt_weight = torch.sqrt(fit_weight / fit_weight.mean()).unsqueeze(1)
        weighted_input = normalized_delta * sqrt_weight
        weighted_output = residual_delta * sqrt_weight
        dimension = normalized_delta.shape[1]
        identity = torch.eye(dimension, dtype=knots.dtype, device=self.device)
        response = torch.linalg.solve(
            weighted_input.T @ weighted_input + self.config.fit_ridge * identity,
            weighted_input.T @ weighted_output,
        )
        standardized_step = -torch.linalg.solve(
            response @ response.T + self.config.step_damping * identity,
            response @ baseline_residual,
        ).clamp(
            -self.config.max_standardized_step,
            self.config.max_standardized_step,
        )
        guided = torch.clamp(
            base_center
            + standardized_step.reshape_as(base_center)
            * torch.from_numpy(sigma).to(self.device),
            -1.0,
            1.0,
        )

        weight = torch.softmax(-(cost - cost.min()) / controller.params.temperature, dim=0)
        weighted_sequence = torch.sum(weight[:, None, None] * actions, dim=0)
        weighted_output = controller.evaluate_action_sequences(
            initial_state,
            current_action,
            history[None],
            reference,
            weighted_sequence,
        )
        empirical_gradient = 2.0 * (response @ baseline_residual)
        empirical_hessian_diagonal = 2.0 * torch.sum(response.square(), dim=1)
        weighted_shift = torch.sum(weight[:, None] * normalized_delta, dim=0)
        softmin = cost.min() - controller.params.temperature * torch.log(
            torch.mean(torch.exp(-(cost - cost.min()) / controller.params.temperature))
        )
        predicted_delta = normalized_delta @ response
        fit_error = torch.linalg.vector_norm(
            (predicted_delta - residual_delta) * sqrt_weight
        )
        target_norm = torch.linalg.vector_norm(
            residual_delta * sqrt_weight
        ).clamp_min(1e-12)
        scalar = torch.stack(
            (
                cost[0],
                cost.min(),
                torch.quantile(cost, 0.10),
                torch.quantile(cost, 0.50),
                cost.mean(),
                weighted_output["cost"][0],
                softmin,
                1.0 / weight.square().sum() / self.config.first_samples,
                torch.tensor(np.mean(raw != clipped), device=self.device),
                fit_error / target_norm,
            )
        )
        feedback = torch.cat(
            (
                standardized_step,
                empirical_gradient,
                empirical_hessian_diagonal,
                weighted_shift,
                scalar,
            )
        ).to(torch.float32)
        if feedback.shape != (74,):
            raise AssertionError("first-pass feedback dimension mismatch")
        return {
            "knots": knots,
            "cost": cost,
            "guided": guided,
            "feedback": feedback,
        }

    @torch.no_grad()
    def _critic_context(
        self,
        history: np.ndarray,
        reference_ego: np.ndarray,
        initial_state: np.ndarray,
        current_action: np.ndarray,
        anchor: torch.Tensor,
        feedback: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        assert self.feedback_critic_normalization is not None
        assert self.feedback_mean is not None and self.feedback_std is not None
        normalized = self._policy_state_inputs(
            history,
            reference_ego,
            initial_state,
            current_action,
            self.feedback_critic_normalization,
        )
        tensors = tuple(
            torch.from_numpy(np.asarray(value, np.float32)[None]).to(self.device)
            for value in normalized
        ) + (
            anchor[None],
            ((feedback.cpu().numpy() - self.feedback_mean) / self.feedback_std),
        )
        tensors = tensors[:-1] + (
            torch.from_numpy(np.asarray(tensors[-1], np.float32)[None]).to(self.device),
        )
        gradients = torch.stack(
            [critic.local_parameters(*tensors)[0].flatten(1) for critic in self.feedback_critics]
        )
        return gradients.mean(0)[0], gradients.std(0, unbiased=False)[0]

    def _actor_inputs(
        self,
        history: np.ndarray,
        reference_ego: np.ndarray,
        initial_state: np.ndarray,
        current_action: np.ndarray,
        anchor: torch.Tensor,
        feedback: torch.Tensor,
        gradient_mean: torch.Tensor,
        gradient_std: torch.Tensor,
    ) -> tuple[torch.Tensor, ...]:
        state_values = self._policy_state_inputs(
            history,
            reference_ego,
            initial_state,
            current_action,
            self.actor_normalization,
        )
        normalized_feedback = (
            feedback.cpu().numpy() - self.actor_feedback_mean
        ) / self.actor_feedback_std
        gradient = np.concatenate(
            (gradient_mean.cpu().numpy(), gradient_std.cpu().numpy())
        )
        normalized_gradient = (
            gradient - self.actor_gradient_mean
        ) / self.actor_gradient_std
        values = (*state_values, anchor.cpu().numpy(), normalized_feedback, normalized_gradient)
        return tuple(
            torch.from_numpy(np.asarray(value, np.float32)[None]).to(self.device)
            for value in values
        )

    @torch.no_grad()
    def propose(
        self,
        controller,
        initial_state,
        current_action,
        history,
        reference,
        warm_knots,
        *,
        reference_ego: np.ndarray | None = None,
        first_seed: int | None = None,
    ) -> ResidualActorRuntimeOutput:
        """Generate one unique deterministic proposal sequence from live inputs."""
        started = time.perf_counter()
        initial = _as_numpy(initial_state).reshape(5)
        current = _as_numpy(current_action).reshape(2)
        history_np = _as_numpy(history).reshape(1, 250, 7)[0]
        reference_np = _as_numpy(reference)
        warm = _as_numpy(warm_knots).reshape(8, 2)
        if reference_ego is None:
            reference_ego = reference_in_ego_frame(reference_np, initial)
        else:
            reference_ego = _as_numpy(reference_ego)

        bc_center = self._bc_center(
            history_np, reference_ego, initial, current, warm
        )
        first = self._first_pass(
            controller,
            initial,
            current,
            history_np,
            reference_np,
            bc_center,
            self.config.first_seed if first_seed is None else int(first_seed),
        )
        gradient_mean, gradient_std = self._critic_context(
            history_np,
            reference_ego,
            initial,
            current,
            first["guided"],
            first["feedback"],
        )
        inputs = self._actor_inputs(
            history_np,
            reference_ego,
            initial,
            current,
            first["guided"],
            first["feedback"],
            gradient_mean,
            gradient_std,
        )
        _, old_center = self.old_actor(*inputs)
        _, proposal_center = self.proposal_actor(*inputs)
        sigma = torch.as_tensor(
            controller.params.noise_sigma, dtype=torch.float32, device=self.device
        ).reshape(1, 1, 2)
        requested = (proposal_center - old_center) / sigma
        requested_rho = torch.sqrt(torch.mean(requested.square(), dim=(1, 2)))
        trust_scale = torch.clamp(
            self.trust_radius / (requested_rho + 1e-8), max=1.0
        )
        direction = requested * trust_scale[:, None, None]
        _, move_probability, alpha_mean, _ = self.base_policy(
            *inputs,
            direction,
            requested_rho[:, None],
            trust_scale[:, None],
        )
        alpha = self.base_policy.deterministic_alpha(
            move_probability, alpha_mean, self.move_threshold
        )
        base_center = torch.clamp(
            old_center + alpha[:, None, None] * sigma * direction, -1.0, 1.0
        )
        residual_inputs = list(inputs)
        residual_inputs[3] = base_center
        _, final_center = self.residual_actor(*residual_inputs)
        action_sequence = controller._interpolate_knots(final_center[0])
        return ResidualActorRuntimeOutput(
            center_knots=final_center[0].detach(),
            action_sequence=action_sequence.detach(),
            bc_center_knots=bc_center.detach(),
            guided_center_knots=first["guided"].detach(),
            first_pass_feedback=first["feedback"].detach(),
            critic_gradient_mean=gradient_mean.detach(),
            critic_gradient_std=gradient_std.detach(),
            old_center_knots=old_center[0].detach(),
            proposal_center_knots=proposal_center[0].detach(),
            base_center_knots=base_center[0].detach(),
            move_probability=float(move_probability[0]),
            base_alpha=float(alpha[0]),
            first_pass_cost=first["cost"].detach(),
            first_pass_knots=first["knots"].detach(),
            duration_s=time.perf_counter() - started,
            rollout_count=self.config.first_samples + 1,
        )
