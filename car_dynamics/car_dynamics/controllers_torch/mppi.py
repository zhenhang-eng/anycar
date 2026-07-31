"""Pure PyTorch MPPI for the current deterministic AnyCar Query model.

This controller deliberately has no JAX dependency.  Dynamics evaluation is
provided by a callable backend, so the same MPPI implementation can use either
the native PyTorch Query graph or its ONNX Runtime export.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TorchMPPICostWeights:
    position: float = 5.0
    yaw: float = 5.0
    vx: float = 1.0
    yawrate: float = 0.0
    acceleration_rate: float = 0.05
    steering_rate: float = 0.10


@dataclass(frozen=True)
class TorchMPPIParams:
    """Parameters fixed to the current Query-model time/state protocol."""

    horizon: int = 50
    history_length: int = 250
    state_dim: int = 5
    action_dim: int = 2
    dt: float = 0.05
    num_samples: int = 256
    num_iterations: int = 1
    num_knots: int = 8
    temperature: float = 1.0
    mean_update_rate: float = 1.0
    noise_sigma: Tuple[float, float] = (0.25, 0.35)
    action_min: Tuple[float, float] = (-1.0, -1.0)
    action_max: Tuple[float, float] = (1.0, 1.0)
    seed: int = 3407

    def __post_init__(self):
        if self.horizon != 50 or self.history_length != 250 or self.dt != 0.05:
            raise ValueError(
                "Torch MPPI must follow the current Query protocol: "
                "history=250, horizon=50, dt=0.05"
            )
        if self.state_dim != 5 or self.action_dim != 2:
            raise ValueError("Query MPPI requires state_dim=5 and action_dim=2")
        if self.num_samples < 2:
            raise ValueError("num_samples must be at least 2")
        if self.num_iterations < 1:
            raise ValueError("num_iterations must be positive")
        if not 2 <= self.num_knots <= self.horizon:
            raise ValueError("num_knots must be within [2, horizon]")
        if self.temperature <= 0:
            raise ValueError("temperature must be positive")
        if not 0 < self.mean_update_rate <= 1:
            raise ValueError("mean_update_rate must be within (0, 1]")
        if any(value <= 0 for value in self.noise_sigma):
            raise ValueError("noise_sigma entries must be positive")


@dataclass
class TorchMPPIRunningState:
    """Warm-start state in the same eight-knot parameterization as sampling."""

    mean_knots: torch.Tensor


class TorchMPPIController:
    """MPPI optimizer shared by PyTorch and ONNX Query rollout backends."""

    def __init__(
        self,
        rollout_backend,
        params: Optional[TorchMPPIParams] = None,
        cost_weights: Optional[TorchMPPICostWeights] = None,
        device: Union[str, torch.device] = "cuda",
    ):
        self.rollout_backend = rollout_backend
        self.params = params or TorchMPPIParams()
        self.cost_weights = cost_weights or TorchMPPICostWeights()
        self.device = torch.device(device)
        self._action_min = torch.tensor(
            self.params.action_min, dtype=torch.float32, device=self.device
        )
        self._action_max = torch.tensor(
            self.params.action_max, dtype=torch.float32, device=self.device
        )
        self._noise_sigma = torch.tensor(
            self.params.noise_sigma, dtype=torch.float32, device=self.device
        )
        self._knot_indices = torch.linspace(
            0, self.params.horizon - 1, self.params.num_knots, device=self.device
        ).round().to(torch.long)
        self._generator = torch.Generator(device=self.device)
        self._generator.manual_seed(self.params.seed)

    def get_init_state(
        self, initial_action: Optional[Sequence[float]] = None
    ) -> TorchMPPIRunningState:
        if initial_action is None:
            action = torch.zeros(2, dtype=torch.float32, device=self.device)
        else:
            action = torch.as_tensor(
                initial_action, dtype=torch.float32, device=self.device
            ).reshape(2)
        action = torch.clamp(action, self._action_min, self._action_max)
        return TorchMPPIRunningState(
            mean_knots=action[None, :].repeat(self.params.num_knots, 1)
        )

    def _interpolate_knots(self, knots: torch.Tensor) -> torch.Tensor:
        """Linearly interpolate ``[..., K, 2]`` knots to 50 model actions."""
        original_shape = knots.shape[:-2]
        flat = knots.reshape(-1, self.params.num_knots, self.params.action_dim)
        full = F.interpolate(
            flat.transpose(1, 2),
            size=self.params.horizon,
            mode="linear",
            align_corners=True,
        ).transpose(1, 2)
        return full.reshape(*original_shape, self.params.horizon, self.params.action_dim)

    def _sequence_to_knots(self, sequence: torch.Tensor) -> torch.Tensor:
        return sequence.index_select(-2, self._knot_indices)

    def _prepare_reference(self, reference) -> torch.Tensor:
        reference = torch.as_tensor(
            reference, dtype=torch.float32, device=self.device
        )
        if reference.ndim != 2 or reference.shape[1] not in (4, 5):
            raise ValueError("reference must have shape [50, 4/5] or [51, 4/5]")
        if reference.shape[0] == self.params.horizon + 1:
            reference = reference[1:]
        if reference.shape[0] != self.params.horizon:
            raise ValueError("reference horizon must be 50 (or 51 including current)")
        return reference

    @staticmethod
    def _wrapped_angle_difference(lhs, rhs):
        difference = lhs - rhs
        return torch.atan2(torch.sin(difference), torch.cos(difference))

    def trajectory_cost(
        self,
        trajectory: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        current_action: torch.Tensor,
    ) -> torch.Tensor:
        components = self.trajectory_cost_components(
            trajectory, action, reference, current_action
        )
        return sum(components.values())

    def trajectory_cost_components(
        self,
        trajectory: torch.Tensor,
        action: torch.Tensor,
        reference: torch.Tensor,
        current_action: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Return the weighted, horizon-summed terms for every candidate."""
        if trajectory.shape != (
            action.shape[0],
            self.params.horizon,
            self.params.state_dim,
        ):
            raise ValueError("rollout backend returned an invalid trajectory shape")
        position_error = (
            trajectory[..., 0:2] - reference[None, :, 0:2]
        ).square().sum(dim=-1)
        yaw_error = self._wrapped_angle_difference(
            trajectory[..., 2], reference[None, :, 2]
        ).square()
        vx_error = (trajectory[..., 3] - reference[None, :, 3]).square()
        components = {
            "position": self.cost_weights.position * position_error.sum(dim=1),
            "yaw": self.cost_weights.yaw * yaw_error.sum(dim=1),
            "vx": self.cost_weights.vx * vx_error.sum(dim=1),
        }
        if reference.shape[1] == 5 and self.cost_weights.yawrate != 0:
            components["yawrate"] = self.cost_weights.yawrate * (
                trajectory[..., 4] - reference[None, :, 4]
            ).square().sum(dim=1)

        previous = torch.cat(
            (current_action.expand(action.shape[0], 1, -1), action[:, :-1]), dim=1
        )
        action_rate = action - previous
        components["acceleration_rate"] = (
            self.cost_weights.acceleration_rate
            * action_rate[..., 0].square().sum(dim=1)
        )
        components["steering_rate"] = (
            self.cost_weights.steering_rate
            * action_rate[..., 1].square().sum(dim=1)
        )
        return components

    def __call__(
        self,
        initial_state,
        current_action,
        history,
        reference,
        running_state: Optional[TorchMPPIRunningState] = None,
    ):
        initial_state = torch.as_tensor(
            initial_state, dtype=torch.float32, device=self.device
        ).reshape(1, self.params.state_dim)
        current_action = torch.as_tensor(
            current_action, dtype=torch.float32, device=self.device
        ).reshape(1, self.params.action_dim)
        history = torch.as_tensor(history, dtype=torch.float32, device=self.device)
        if tuple(history.shape) != (1, self.params.history_length, 7):
            raise ValueError("history must have shape [1, 250, 7]")
        reference = self._prepare_reference(reference)
        if running_state is None:
            running_state = self.get_init_state(current_action.reshape(-1))
        mean_knots = running_state.mean_knots.to(self.device)
        if tuple(mean_knots.shape) != (self.params.num_knots, self.params.action_dim):
            raise ValueError("running_state.mean_knots has an invalid shape")

        last = None
        for _ in range(self.params.num_iterations):
            noise = torch.randn(
                self.params.num_samples,
                self.params.num_knots,
                self.params.action_dim,
                generator=self._generator,
                device=self.device,
            ) * self._noise_sigma
            noise[0].zero_()  # Always retain the current mean as a candidate.
            sampled_knots = mean_knots[None, :, :] + noise
            sampled_knots = torch.clamp(
                sampled_knots, self._action_min, self._action_max
            )
            sampled_action = self._interpolate_knots(sampled_knots)

            trajectory = self.rollout_backend(
                history, initial_state, current_action, sampled_action
            ).to(self.device)
            cost_components = self.trajectory_cost_components(
                trajectory, sampled_action, reference, current_action
            )
            cost = sum(cost_components.values())
            weight = torch.softmax(
                -(cost - cost.min()) / self.params.temperature, dim=0
            )
            weighted_sequence = torch.sum(
                weight[:, None, None] * sampled_action, dim=0
            )
            candidate_mean_knots = self._sequence_to_knots(weighted_sequence)
            mean_knots = (
                self.params.mean_update_rate * candidate_mean_knots
                + (1.0 - self.params.mean_update_rate) * mean_knots
            )
            last = (
                sampled_action,
                trajectory,
                cost,
                cost_components,
                weight,
                weighted_sequence,
            )

        assert last is not None
        (
            sampled_action,
            trajectory,
            cost,
            cost_components,
            weight,
            weighted_sequence,
        ) = last
        action = weighted_sequence[0]
        shifted_sequence = torch.cat(
            (weighted_sequence[1:], weighted_sequence[-1:]), dim=0
        )
        new_state = TorchMPPIRunningState(
            mean_knots=self._sequence_to_knots(shifted_sequence).detach()
        )
        best_index = torch.argmin(cost)
        info: Dict[str, object] = {
            "optimized_action_sequence": weighted_sequence.detach(),
            "sampled_action_sequences": sampled_action.detach(),
            "best_sampled_action_sequence": sampled_action[best_index].detach(),
            "sampled_trajectories": trajectory.detach(),
            "trajectory": trajectory[best_index].detach(),
            "cost": cost.detach(),
            "cost_components": {
                name: value.detach() for name, value in cost_components.items()
            },
            "weight": weight.detach(),
            "best_cost": cost[best_index].detach(),
            "effective_sample_size": (1.0 / weight.square().sum()).detach(),
        }
        return action.detach(), new_state, info
